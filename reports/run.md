
# Webcam laptop

python demo_posture.py --weights posture_gru_ensemble.pt --source 0

# IP Webcam trên điện thoại (cùng mạng Wi-Fi với máy tính)

python demo_posture.py --weights posture_gru_ensemble.pt --ip 192.168.1.23

# Hoặc đưa thẳng URL, hoặc một file video để test

python demo_posture.py --weights posture_gru_ensemble.pt --source http://192.168.1.23:8080/video
python demo_posture.py --weights posture_gru_ensemble.pt --source clip.mp4
