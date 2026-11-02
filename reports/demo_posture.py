#!/usr/bin/env python3
"""
Demo inference realtime cho model tư thế ngồi (CE190363_Posture.ipynb) - bản tinh chỉnh.

Pipeline (giữ nguyên như lúc train):
    khung hình -> YOLO-pose (17 keypoint COCO) -> bộ đệm theo thời gian
    -> lấy đều 32 frame -> đặc trưng "new" (hoặc "legacy") -> chuẩn hóa
    -> ensemble GRU 2 chiều -> trung bình xác suất -> nhãn tư thế

Cách chạy:
    # Webcam laptop
    python demo_posture.py --weights posture_gru_ensemble.pt --source 0

    # IP Webcam trên điện thoại
    python demo_posture.py --weights posture_gru_ensemble.pt --ip 192.168.1.23

    # HAI camera cùng lúc (giống lúc train: gộp xác suất của 2 góc nhìn -> chính xác hơn)
    python demo_posture.py --weights posture_gru_ensemble.pt --source 0 1
    python demo_posture.py --weights posture_gru_ensemble.pt --ip 192.168.1.23 192.168.1.24

    # File video (để test)
    python demo_posture.py --weights posture_gru_ensemble.pt --source clip.mp4

Phím: q / ESC = thoát, r = xóa bộ đệm (bắt đầu lại).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import unicodedata
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)
from torch import nn

# ---------------------------------------------------------------------------
# 1. Đặc trưng: sao chép đúng từ notebook (KHÔNG sửa - phải khớp lúc train)
# ---------------------------------------------------------------------------
EPS = 1e-6
N_GEO = 10


def frame_features(kp: np.ndarray, cf: np.ndarray, bx: np.ndarray, mode: str) -> np.ndarray:
    """kp[T,17,2], cf[T,17], bx[T,4] -> đặc trưng theo frame."""
    T = len(kp)
    w, h = np.maximum(bx[:, 2] - bx[:, 0], 1.0), np.maximum(bx[:, 3] - bx[:, 1], 1.0)
    if mode == "legacy":
        xy = (kp - bx[:, None, :2]) / np.stack([w, h], axis=1)[:, None, :]
        return np.concatenate([xy.reshape(T, -1), cf], axis=1)
    shoulder_mid = kp[:, 5:7].mean(axis=1)
    hip_mid = kp[:, 11:13].mean(axis=1)
    shoulder_w = np.linalg.norm(kp[:, 5] - kp[:, 6], axis=1)
    scale = np.maximum(shoulder_w, 0.25 * h)
    rel = np.clip((kp - shoulder_mid[:, None, :]) / scale[:, None, None], -6.0, 6.0)
    nose = rel[:, 0]
    nose_dist = np.linalg.norm(nose, axis=1)
    shoulder_vec = kp[:, 6] - kp[:, 5]
    hips_visible = np.minimum(cf[:, 11], cf[:, 12]) > 0.3
    wrist_up = np.where(cf[:, 9:11] > 0.2, -rel[:, 9:11, 1], -1.0).max(axis=1)
    geo = np.stack([
        nose[:, 0] / (nose_dist + EPS),
        -nose[:, 1] / (nose_dist + EPS),
        nose_dist,
        np.clip(shoulder_vec[:, 1] / (shoulder_w + EPS), -1.0, 1.0),
        shoulder_w / h,
        np.where(hips_visible, np.linalg.norm(shoulder_mid - hip_mid, axis=1) / h, 0.0),
        rel[:, 3:5, 1].mean(axis=1),
        wrist_up,
        np.log(w / h),
        0.5 * np.log(w * h),
    ], axis=1)
    return np.concatenate([rel.reshape(T, -1), cf, geo], axis=1)


def sequence_features(kp: np.ndarray, cf: np.ndarray, bx: np.ndarray, mode: str) -> np.ndarray:
    feats = frame_features(kp, cf, bx, mode)
    if mode == "new":
        feats[:, -1] -= feats[:, -1].mean()
        geo = feats[:, -N_GEO:]
        feats = np.concatenate([feats, np.diff(geo, axis=0, prepend=geo[:1])], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def derive_box(kp: np.ndarray, cf: np.ndarray) -> np.ndarray:
    """Suy bbox từ keypoint (giống nhánh dự phòng trong notebook)."""
    pts = kp[cf > 0.2] if (cf > 0.2).sum() >= 3 else kp
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.1 * np.maximum(hi - lo, 1.0)
    return np.concatenate([lo - pad, hi + pad]).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Model: giống notebook. Chạy trên CPU mặc định (model rất nhỏ, tránh đồng bộ GPU)
# ---------------------------------------------------------------------------
class PoseGRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, classes: int, dropout: float = 0.4) -> None:
        super().__init__()
        self.input_drop = nn.Dropout(0.1)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim * 4), nn.Dropout(dropout),
                                  nn.Linear(hidden_dim * 4, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.input_drop(x))
        return self.head(torch.cat([out.mean(dim=1), out.amax(dim=1)], dim=1))


class PostureEnsemble:
    """Nạp file .pt do notebook lưu (posture_gru_ensemble.pt) và dự đoán xác suất."""

    def __init__(self, path: str, device: torch.device) -> None:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # torch cũ không có weights_only
            ckpt = torch.load(path, map_location="cpu")
        self.device = device
        self.mode: str = ckpt["config"]["mode"]
        hidden: int = ckpt["config"]["hidden"]
        self.seq_len: int = int(ckpt.get("seq_len", 32))
        self.names: dict[int, str] = {int(k): v for k, v in ckpt["label_names"].items()}
        self.num_classes = len(self.names)
        # Siêu dữ liệu tuỳ chọn (xem hướng dẫn thêm vào notebook); không có thì dùng mặc định
        self.clip_seconds: float | None = ckpt.get("clip_seconds")
        self.bbox_mode: str | None = ckpt.get("bbox_mode")
        self.nets = []
        means, stds = [], []
        for m in ckpt["members"]:
            mean = np.asarray(m["mean"], dtype=np.float32)
            std = np.asarray(m["std"], dtype=np.float32)
            net = PoseGRUClassifier(len(mean), hidden, self.num_classes).to(device)
            net.load_state_dict(m["state"])
            net.eval()
            self.nets.append(net)
            means.append(mean)
            stds.append(std)
        self.means = np.stack(means)[:, None, :]  # [M,1,F] để broadcast với [M,T,F]
        self.stds = np.stack(stds)[:, None, :]
        print(f"[model] {ckpt.get('experiment')} | mode={self.mode} hidden={hidden} "
              f"| {len(self.nets)} thành viên | {self.num_classes} lớp | device={device}")

    @torch.inference_mode()
    def predict(self, kp: np.ndarray, cf: np.ndarray, bx: np.ndarray) -> np.ndarray:
        """kp[32,17,2], cf[32,17], bx[32,4] -> xác suất [num_classes]."""
        feats = sequence_features(kp, cf, bx, self.mode)  # tính 1 lần, dùng chung cho mọi thành viên
        xs = np.clip((feats[None] - self.means) / self.stds, -8.0, 8.0).astype(np.float32)  # [M,T,F]
        xs = torch.from_numpy(xs).to(self.device)
        probs = torch.stack([torch.softmax(net(xs[i:i + 1]), dim=1)[0] for i, net in enumerate(self.nets)])
        return probs.mean(dim=0).cpu().numpy()  # chỉ đồng bộ một lần


# ---------------------------------------------------------------------------
# 3. Bộ đệm keypoint theo thời gian
# ---------------------------------------------------------------------------
class PoseBuffer:
    def __init__(self, window_s: float, max_gap_s: float = 1.0) -> None:
        self.window = window_s
        self.max_gap = max_gap_s
        self.items: deque = deque()

    def __len__(self) -> int:
        return len(self.items)

    def prune(self, now: float) -> None:
        while self.items and now - self.items[0][0] > self.window:
            self.items.popleft()

    def add(self, t: float, kp: np.ndarray, cf: np.ndarray, bx: np.ndarray) -> None:
        # Mất người > max_gap giây rồi xuất hiện lại: bỏ dữ liệu cũ (tránh ghép 2 tư thế khác nhau)
        if self.items and t - self.items[-1][0] > self.max_gap:
            self.items.clear()
        self.items.append((t, kp, cf, bx))
        self.prune(t)

    def clear(self) -> None:
        self.items.clear()

    def progress(self, min_frames: int, fill: float) -> float:
        """Mức đã thu thập đủ dữ liệu (0..1)."""
        if len(self.items) < 2:
            return 0.0
        span = self.items[-1][0] - self.items[0][0]
        return float(min(span / (fill * self.window), len(self.items) / min_frames, 1.0))

    def ready(self, min_frames: int, fill: float) -> bool:
        return self.progress(min_frames, fill) >= 1.0

    def sample(self, seq_len: int):
        """Lấy đều seq_len frame THEO THỜI GIAN (không theo chỉ số), nên FPS dao động /
        frame bị rơi cũng không làm méo nhịp chuyển động như khi lấy theo index."""
        ts = np.fromiter((it[0] for it in self.items), dtype=np.float64, count=len(self.items))
        targets = np.linspace(ts[0], ts[-1], seq_len)
        idx = np.clip(np.searchsorted(ts, targets), 1, len(ts) - 1)
        idx = np.where(targets - ts[idx - 1] <= ts[idx] - targets, idx - 1, idx)
        picked = [self.items[i] for i in idx]
        return (np.stack([p[1] for p in picked]), np.stack([p[2] for p in picked]),
                np.stack([p[3] for p in picked]))


# ---------------------------------------------------------------------------
# 4. Nguồn video: webcam / IP Webcam (MJPEG) / RTSP / file
# ---------------------------------------------------------------------------
class FrameSource:
    """Webcam & luồng mạng: luôn lấy frame MỚI NHẤT bằng thread (tránh trễ tích lũy).
    File video: đọc tuần tự, timestamp theo FPS của video."""

    STREAM_PREFIXES = ("http://", "https://", "rtsp://", "rtmp://")

    def __init__(self, source: str, width: int = 0, height: int = 0, cam_fps: int = 30) -> None:
        self.source, self.width, self.height, self.cam_fps = source, width, height, cam_fps
        self.is_index = source.isdigit()
        self.live = self.is_index or source.lower().startswith(self.STREAM_PREFIXES)
        self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn video: {source}")
        self.finished = False
        self._frame_idx = 0
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.live:
            self._cond = threading.Condition()
            self._latest = None
            self._seq = 0
            self._last_seq = 0
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        if self.is_index:
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(self.source), backend)
            # MJPG: nhiều webcam (nhất là trên Windows/DSHOW) bị khóa ở ~10 FPS khi chạy 720p
            # với định dạng YUY2 mặc định; MJPG cho phép đủ 30 FPS.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            if self.width and self.height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.cam_fps:
                cap.set(cv2.CAP_PROP_FPS, self.cam_fps)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self) -> None:
        fails = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                fails += 1
                if fails >= 30:  # mất kết nối -> thử mở lại
                    print(f"[source] {self.source}: mất tín hiệu, đang kết nối lại...")
                    self.cap.release()
                    time.sleep(1.0)
                    self.cap = self._open()
                    fails = 0
                else:
                    time.sleep(0.03)
                continue
            fails = 0
            with self._cond:
                self._latest = (frame, time.monotonic())
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 0.05):
        """Trả về (frame, t_giây) hoặc None nếu chưa có frame mới / đã hết video.
        timeout ngắn để vòng lặp chính luôn kịp xử lý phím bấm."""
        if not self.live:
            ok, frame = self.cap.read()
            if not ok:
                self.finished = True
                return None
            t = self._frame_idx / self._fps
            self._frame_idx += 1
            return frame, t
        with self._cond:
            self._cond.wait_for(lambda: self._seq != self._last_seq, timeout=timeout)
            if self._seq == self._last_seq:
                return None
            self._last_seq = self._seq
            return self._latest

    def release(self) -> None:
        if self.live:
            self._stop.set()
            self._thread.join(timeout=1.0)
        self.cap.release()


# ---------------------------------------------------------------------------
# 5. Lấy người từ kết quả YOLO-pose (bám theo người đang theo dõi)
# ---------------------------------------------------------------------------
def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class PersonSelector:
    """Chọn 1 người mỗi frame. Nếu người đang theo dõi vẫn còn (IoU > 0.3 với frame trước) thì giữ
    nguyên; chỉ khi mất dấu mới chọn lại người có (diện tích x độ tin cậy) lớn nhất. Nhờ vậy khi có
    người khác đi ngang, cửa sổ thời gian không bị trộn keypoint của 2 người."""

    def __init__(self) -> None:
        self.prev: np.ndarray | None = None

    def reset(self) -> None:
        self.prev = None

    def extract(self, result, bbox_mode: str):
        """Trả về (kp[17,2], cf[17], bx[4]) hoặc None."""
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            self.prev = None
            return None
        xy = result.keypoints.xy.cpu().numpy().astype(np.float32)
        kconf = result.keypoints.conf
        cf_all = kconf.cpu().numpy().astype(np.float32) if kconf is not None else np.ones(xy.shape[:2], np.float32)
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        bconf = result.boxes.conf.cpu().numpy()
        i = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * bconf))
        if self.prev is not None:
            ious = np.array([box_iou(self.prev, b) for b in boxes])
            if ious.max() > 0.3:
                i = int(np.argmax(ious))
        kp, cf, bx = xy[i], cf_all[i], boxes[i]
        if (cf > 0.05).sum() < 3 or not np.isfinite(kp).all():
            self.prev = None
            return None
        self.prev = bx.copy()
        if bbox_mode == "keypoints" or bx[2] <= bx[0] or bx[3] <= bx[1]:
            bx = derive_box(kp, cf)
        return kp, cf, bx


# ---------------------------------------------------------------------------
# 6. Ổn định nhãn hiển thị (chống nhấp nháy + báo "không chắc chắn")
# ---------------------------------------------------------------------------
class LabelStabilizer:
    """Nhãn đang hiển thị chỉ đổi khi nhãn mới thắng `hold` lần phân loại liên tiếp.
    Xác suất cao nhất < min_conf -> trả None (UI hiện 'Không chắc chắn')."""

    def __init__(self, min_conf: float, hold: int) -> None:
        self.min_conf, self.hold = min_conf, max(1, hold)
        self.reset()

    def reset(self) -> None:
        self.current: int | None = None
        self.cand: int | None = None
        self.count = 0

    def update(self, probs: np.ndarray) -> int | None:
        k = int(np.argmax(probs))
        if probs[k] < self.min_conf:
            self.cand, self.count = None, 0
            if self.current is not None and probs[self.current] < 0.8 * self.min_conf:
                self.current = None
            return self.current
        if self.current is None or k == self.current:
            self.current, self.cand, self.count = k, None, 0
            return k
        self.count = self.count + 1 if k == self.cand else 1
        self.cand = k
        if self.count >= self.hold:
            self.current, self.cand, self.count = k, None, 0
        return self.current


# ---------------------------------------------------------------------------
# 7. Vẽ giao diện (hỗ trợ tiếng Việt có dấu qua PIL; fallback bỏ dấu nếu thiếu font)
# ---------------------------------------------------------------------------
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]
CLASS_COLORS_BGR = [(80, 200, 80), (0, 160, 255), (60, 60, 230), (200, 140, 40), (200, 80, 200), (230, 200, 40)]
GREY = (160, 160, 160)


def strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def find_font(user_font: str | None) -> str | None:
    cands = ([user_font] if user_font else []) + FONT_CANDIDATES
    try:
        import matplotlib
        cands.append(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"))
    except (ImportError, AttributeError, OSError) as exc:
        logger.debug("Unable to locate the matplotlib font directory: %s", exc)
    return next((c for c in cands if c and Path(c).is_file()), None)


class PanelRenderer:
    SCALE = 0.6  # 1.0 = kích thước cũ; muốn nhỏ hơn nữa thì giảm (vd 0.5), to hơn thì tăng

    def _px(self, v: float) -> int:
        return max(1, round(v * self.SCALE))

    def __init__(self, names: dict[int, str], font_path: str | None) -> None:
        px = self._px
        self.W, self.ROW = px(500), px(28)
        self.BAR_X, self.BAR_W = px(262), px(150)
        self.TOP = px(70)
        self.names = names
        self.font_path = find_font(font_path)
        self.pil = None
        self._cache_key = None
        self._cache_img: np.ndarray | None = None
        if self.font_path:
            try:
                from PIL import Image, ImageDraw, ImageFont
                self.ImageFont = ImageFont
                self.Image, self.ImageDraw = Image, ImageDraw
                self.f_big = ImageFont.truetype(self.font_path, max(14, px(26)))
                self.f_small = ImageFont.truetype(self.font_path, max(10, px(17)))
                self.pil = True
            except (ImportError, OSError) as exc:
                print(f"[ui] không dùng được PIL/font ({exc}); sẽ bỏ dấu tiếng Việt.")
        else:
            print("[ui] không tìm thấy font Unicode (dùng --font <file.ttf>); sẽ bỏ dấu tiếng Việt.")
        self.height = self.TOP + self.ROW * len(names) + px(10)

    def render(self, title: str, subtitle: str, probs: np.ndarray | None, color) -> np.ndarray:
        """Có cache: chỉ vẽ lại khi nội dung thật sự đổi."""
        key = (title, subtitle, tuple(color), None if probs is None else tuple(np.round(probs * 100).astype(int)))
        if key == self._cache_key and self._cache_img is not None:
            return self._cache_img
        self._cache_key, self._cache_img = key, self._draw(title, subtitle, probs, color)
        return self._cache_img

    def _draw(self, title: str, subtitle: str, probs: np.ndarray | None, color) -> np.ndarray:
        px = self._px
        panel = np.full((self.height, self.W, 3), 30, np.uint8)
        for i in range(len(self.names)):
            y = self.TOP + i * self.ROW
            p = float(probs[i]) if probs is not None else 0.0
            y1, y2 = y + px(5), y + px(21)
            cv2.rectangle(panel, (self.BAR_X, y1), (self.BAR_X + self.BAR_W, y2), (70, 70, 70), -1)
            cv2.rectangle(panel, (self.BAR_X, y1), (self.BAR_X + int(self.BAR_W * p), y2),
                          CLASS_COLORS_BGR[i % len(CLASS_COLORS_BGR)], -1)
        texts = [(title, (px(12), px(8)), "big", color),
                 (subtitle, (px(12), px(42)), "small", (200, 200, 200))]
        for i, name in sorted(self.names.items()):
            y = self.TOP + i * self.ROW
            p = float(probs[i]) if probs is not None else 0.0
            texts.append((name, (px(12), y + px(3)), "small", (235, 235, 235)))
            texts.append((f"{p * 100:3.0f}%", (self.BAR_X + self.BAR_W + px(10), y + px(3)),
                          "small", (235, 235, 235)))
        if self.pil:
            img = self.Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            draw = self.ImageDraw.Draw(img)
            for text, pos, size, col in texts:
                draw.text(pos, text, font=self.f_big if size == "big" else self.f_small, fill=col[::-1])
            return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        for text, (x, y), size, col in texts:  # fallback khi không có font Unicode
            cv2.putText(panel, strip_accents(text), (x, y + (px(24) if size == "big" else px(14))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.SCALE if size == "big" else 0.5 * self.SCALE,
                        col, 1, cv2.LINE_AA)
        return panel


def draw_person(frame: np.ndarray, kp: np.ndarray, cf: np.ndarray, bx: np.ndarray, color) -> None:
    x1, y1, x2, y2 = bx.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    for a, b in SKELETON:
        if cf[a] > 0.3 and cf[b] > 0.3:
            cv2.line(frame, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
    for j in range(17):
        if cf[j] > 0.3:
            cv2.circle(frame, tuple(kp[j].astype(int)), 4, (0, 0, 255), -1, cv2.LINE_AA)


def blit_panel(frame: np.ndarray, panel: np.ndarray, margin: int = 10, alpha: float = 0.8) -> None:
    h, w = panel.shape[:2]
    h, w = min(h, frame.shape[0] - margin), min(w, frame.shape[1] - margin)
    roi = frame[margin:margin + h, margin:margin + w]
    frame[margin:margin + h, margin:margin + w] = cv2.addWeighted(panel[:h, :w], alpha, roi, 1 - alpha, 0)


def tile_frames(frames: list[np.ndarray], height: int = 540) -> np.ndarray:
    """Ghép nhiều camera cạnh nhau (một camera thì trả nguyên frame, không tốn thêm gì)."""
    if len(frames) == 1:
        return frames[0]
    resized = [cv2.resize(f, (int(f.shape[1] * height / f.shape[0]), height)) for f in frames]
    return np.hstack(resized)


# ---------------------------------------------------------------------------
# 8. Mỗi camera = 1 Stream (nguồn + bộ đệm + bộ chọn người + xác suất làm mượt)
# ---------------------------------------------------------------------------
class Stream:
    def __init__(self, source: str, args: argparse.Namespace, window: float) -> None:
        self.source = source
        self.src = FrameSource(source, args.width, args.height, args.cam_fps)
        self.buf = PoseBuffer(window, args.max_gap)
        self.selector = PersonSelector()
        self.smooth: np.ndarray | None = None
        self.person = None
        self.last_infer = -1e9
        self.last_seen = -1e9

    def reset(self) -> None:
        self.buf.clear()
        self.selector.reset()
        self.smooth = None
        self.person = None
        self.last_infer = -1e9


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo realtime nhận dạng tư thế ngồi")
    p.add_argument("--weights", required=True, help="posture_gru_ensemble.pt (xuất từ notebook)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--source", nargs="+", default=["0"],
                     help="Một hoặc nhiều nguồn: index webcam (0 1 ..), URL (http://ip:8080/video, rtsp://...) "
                          "hoặc file video. Nhiều nguồn = gộp xác suất như lúc train với 2 camera")
    src.add_argument("--ip", nargs="+", help="IP điện thoại chạy app IP Webcam (có thể đưa 2 IP)")
    p.add_argument("--port", type=int, default=8080, help="Cổng IP Webcam (mặc định 8080)")
    p.add_argument("--user", help="Tài khoản IP Webcam (nếu có đặt mật khẩu)")
    p.add_argument("--password", help="Mật khẩu IP Webcam")
    p.add_argument("--yolo", default="yolo11n-pose.pt",
                   help="Model YOLO-pose (PHẢI dùng đúng loại đã dùng khi thu dataset)")
    p.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đưa vào YOLO")
    p.add_argument("--det-conf", type=float, default=0.3, help="Ngưỡng tin cậy phát hiện người")
    p.add_argument("--no-half", action="store_true", help="Tắt FP16 cho YOLO trên GPU")
    p.add_argument("--bbox-mode", choices=["yolo", "keypoints"], default=None,
                   help="Mặc định lấy theo file .pt (nếu có) rồi mới tới 'yolo'. "
                        "yolo: bbox của YOLO; keypoints: suy bbox từ keypoint")
    p.add_argument("--window", type=float, default=None,
                   help="Cửa sổ thời gian (giây). Mặc định lấy 'clip_seconds' trong file .pt, "
                        "nếu không có thì 3.0. PHẢI xấp xỉ độ dài trung vị của clip lúc train")
    p.add_argument("--fill", type=float, default=0.8,
                   help="Tỉ lệ cửa sổ cần đầy trước khi bắt đầu dự đoán (0.6 = nhanh hơn, 1.0 = chắc hơn)")
    p.add_argument("--max-gap", type=float, default=1.0, help="Mất người quá ngần này giây thì xóa bộ đệm")
    p.add_argument("--interval", type=float, default=0.3, help="Chu kỳ chạy phân loại (giây)")
    p.add_argument("--ema", type=float, default=0.6, help="Làm mượt xác suất (0 = tắt, gần 1 = mượt hơn)")
    p.add_argument("--min-conf", type=float, default=0.45,
                   help="Dưới ngưỡng này hiện 'Không chắc chắn' (0 = luôn hiện nhãn)")
    p.add_argument("--hold", type=int, default=2,
                   help="Số lần phân loại liên tiếp để đổi nhãn đang hiển thị (chống nhấp nháy)")
    p.add_argument("--width", type=int, default=1280, help="Độ phân giải webcam yêu cầu (chỉ khi dùng index)")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--cam-fps", type=int, default=30, help="FPS yêu cầu cho webcam index (0 = để mặc định)")
    p.add_argument("--device", default=None, help="Thiết bị cho YOLO: cuda:0 / cpu (mặc định tự chọn)")
    p.add_argument("--clf-device", default="cpu", help="Thiết bị cho ensemble GRU (model nhỏ, CPU là đủ)")
    p.add_argument("--font", default=None, help="File .ttf hỗ trợ tiếng Việt (nếu tự dò không ra)")
    p.add_argument("--profile", action="store_true", help="In thời gian trung bình YOLO / phân loại mỗi 5 giây")
    return p.parse_args()


def build_sources(args: argparse.Namespace) -> list[str]:
    if args.ip:
        auth = f"{args.user}:{args.password}@" if args.user else ""
        return [f"http://{auth}{ip}:{args.port}/video" for ip in args.ip]
    return list(args.source)


def mask(url: str, password: str | None) -> str:
    return url.replace(password, "***") if password else url


def main() -> int:
    args = parse_args()
    device_name = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    on_cuda = device_name.startswith("cuda")
    half = on_cuda and not args.no_half
    if on_cuda:
        torch.backends.cudnn.benchmark = True  # kích thước ảnh cố định -> chọn kernel nhanh nhất
    print(f"[init] YOLO device={device_name} half={half} | GRU device={args.clf_device}")

    from ultralytics import YOLO
    yolo = YOLO(args.yolo)
    clf = PostureEnsemble(args.weights, torch.device(args.clf_device))
    window = args.window or clf.clip_seconds or 3.0
    bbox_mode = args.bbox_mode or clf.bbox_mode or "yolo"
    min_frames = max(8, clf.seq_len // 3)
    print(f"[init] window={window:.2f}s fill={args.fill} bbox_mode={bbox_mode}")

    def detect(frame: np.ndarray):
        return yolo.predict(frame, imgsz=args.imgsz, conf=args.det_conf, device=device_name,
                            quantize=16 if half else None, max_det=5, verbose=False)[0]

    # Khởi động trước (nạp kernel / cudnn benchmark) để frame thật đầu tiên không bị giật
    detect(np.zeros((args.height or 720, args.width or 1280, 3), np.uint8))

    sources = build_sources(args)
    for s in sources:
        print(f"[init] nguồn video: {mask(s, args.password)}")
    streams = [Stream(s, args, window) for s in sources]
    ui = PanelRenderer(clf.names, args.font)
    stabilizer = LabelStabilizer(args.min_conf, args.hold)

    fps, t_prev, fps_shown, fps_updated = 0.0, time.perf_counter(), 0.0, 0.0
    t_yolo = t_clf = 0.0
    n_yolo = n_clf = 0
    prof_t0 = time.perf_counter()
    last_frames: list[np.ndarray | None] = [None] * len(streams)
    win = "Posture demo (q: thoat, r: reset)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            got_new = False
            for k, st in enumerate(streams):
                item = st.src.read()
                if item is None:
                    continue
                got_new = True
                frame, t = item
                t0 = time.perf_counter()
                person = st.selector.extract(detect(frame), bbox_mode)
                t_yolo += time.perf_counter() - t0
                n_yolo += 1
                st.person = person
                if person is not None:
                    st.buf.add(t, *person)
                    st.last_seen = t
                else:
                    st.buf.prune(t)
                    if t - st.last_seen > args.max_gap:  # mất người đủ lâu: bỏ kết quả cũ
                        st.smooth = None
                        st.buf.clear()
                if st.buf.ready(min_frames, args.fill) and t - st.last_infer >= args.interval:
                    t0 = time.perf_counter()
                    probs = clf.predict(*st.buf.sample(clf.seq_len))
                    t_clf += time.perf_counter() - t0
                    n_clf += 1
                    st.smooth = probs if (st.smooth is None or args.ema <= 0) \
                        else args.ema * st.smooth + (1 - args.ema) * probs
                    st.last_infer = t
                last_frames[k] = frame

            all_done = all(s.src.finished for s in streams)
            if all_done:
                break

            if got_new:
                # ---- gộp xác suất các camera đang thấy người ----
                live = [s.smooth for s in streams if s.smooth is not None]
                fused = np.mean(live, axis=0) if live else None
                if fused is None:
                    stabilizer.reset()
                    label = None
                else:
                    label = stabilizer.update(fused)
                seen_any = any(s.person is not None for s in streams)

                # ---- hiển thị ----
                if fused is not None and label is not None:
                    color = CLASS_COLORS_BGR[label % len(CLASS_COLORS_BGR)]
                    title = clf.names[label]
                    subtitle = f"Độ tin cậy {fused[label] * 100:.0f}%  |  {fps_shown:4.1f} FPS"
                elif fused is not None:
                    top = int(np.argmax(fused))
                    color, title = GREY, "Không chắc chắn"
                    subtitle = f"Gần nhất: {clf.names[top]} {fused[top] * 100:.0f}%  |  {fps_shown:4.1f} FPS"
                elif not seen_any:
                    color, title, subtitle = GREY, "Không thấy người", f"{fps_shown:4.1f} FPS"
                else:
                    prog = min(s.buf.progress(min_frames, args.fill) for s in streams if s.person is not None)
                    color, title = GREY, "Đang thu thập dữ liệu..."
                    subtitle = f"{prog * 100:.0f}%  |  {fps_shown:4.1f} FPS"

                shown = []
                for st, frame in zip(streams, last_frames):
                    if frame is None:
                        continue
                    img = frame if len(streams) == 1 else frame.copy()
                    if st.person is not None:
                        draw_person(img, *st.person, color)
                    shown.append(img)
                canvas = tile_frames(shown)
                blit_panel(canvas, ui.render(title, subtitle, fused, color))
                cv2.imshow(win, canvas)

                now = time.perf_counter()
                inst = 1.0 / max(now - t_prev, 1e-6)
                fps = 0.9 * fps + 0.1 * inst if fps else inst
                t_prev = now
                if now - fps_updated > 0.5:  # cập nhật số FPS hiển thị 2 lần/giây (giúp cache panel hiệu quả)
                    fps_shown, fps_updated = fps, now

            # luôn xử lý phím, kể cả khi chưa có frame mới (tránh cửa sổ bị "treo")
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                for st in streams:
                    st.reset()
                stabilizer.reset()

            if args.profile and time.perf_counter() - prof_t0 >= 5.0:
                print(f"[profile] YOLO {1000 * t_yolo / max(n_yolo, 1):.1f} ms/frame | "
                      f"GRU ensemble {1000 * t_clf / max(n_clf, 1):.1f} ms/lần | loop {fps:.1f} FPS")
                t_yolo = t_clf = 0.0
                n_yolo = n_clf = 0
                prof_t0 = time.perf_counter()
    finally:
        for st in streams:
            st.src.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
