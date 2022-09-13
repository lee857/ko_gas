import torch
import cv2
import time
import requests

# relay module 연결
ip = '192.168.100.60'
ms = '1000'
addr = f'http://{ip}/index.html?p0={ms}'

# 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       'c:/users/user/desktop/내부/best.pt')

# Inference Setting
model.conf = 0.85  # NMS confidence threshold
model.iou = 0.25  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = 0
model.max_det = 1  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference

# RTSP 불러오기
url = 'rtsp://admin:admin135!@192.168.0.20:554/unicast/c5/s0/live' # camera ip = 223.171.146.61:554

temp = []
count = 0
while True:
    # 프레임 추출
    cap = cv2.VideoCapture(url)

    ret, frame = cap.read()

    if not ret:
        print('전송실패')
        continue

    frame = cv2.resize(frame, (640, 384))  # Resize
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB

    # Inference
    results = model(frame)

    results.show()

    # if count == 0:
    #     results.show()

    lst_results = results.xyxy[0].tolist()  # im predictions (tensor)

    lst_results = [[int(lst[i]) if i < 4 else lst[i]
                    for i in range(len(lst))] for lst in lst_results]

    if not lst_results:
        print('객체가 존재하지 않습니다.')
        # Relay Module 시그널 전송
        requests.get(addr)

        results.show()
        continue

    elif temp:
        for a, b in zip(lst_results[0][:4], temp):
            if a-b > 5:
                print('움직임이 발생했습니다.')
                # Relay Module 시그널 전송
                requests.get(addr)

                results.show()
                break
            else:
                print('비교문제 없음')
    else:
        print('변화 없음')

    temp = lst_results[0][:4]

    time.sleep(3)

    # count += 1
