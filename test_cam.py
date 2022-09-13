import torch
import cv2
import requests

# relay module 연결
ip = '192.168.100.60'
ms = '1000'
addr = f'http://{ip}/index.html?p0={ms}' # http://192.168.100.60/index.html

# 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       'c:/users/user/desktop/KoreaExpert/가스공사프로젝트/내부/best.pt')

# Inference Setting
model.conf = 0.70  # NMS confidence threshold
model.iou = 0.25  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = 0
model.max_det = 1  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference

# RTSP 불러오기
url = 'rtsp://admin:admin135!@192.168.0.20:554/unicast/c1/s0/live' # camera ip = 223.171.146.61:554
# url = 'rtsp://admin:admin135!@223.171.134.88:554/stream0' # camera ip = 223.171.146.61:554

temp = []
# count = 0
xy_position = ['x_min', 'y_min', 'x_max', 'y_max']

cap = cv2.VideoCapture(url)
fps = int(cap.get(cv2.CAP_PROP_FPS))

while True:
    # 프레임 추출
    ret, frame = cap.read()
    frameId = int(round(cap.get(1)))

    if not ret:
        print('전송실패')
        continue
    else:
        if frameId % (fps*20) == 0 or frameId == 1:
            # print(count)
            frame = cv2.resize(frame, (640, 384))  # Resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB

            # Inference
            results = model(frame)

            # results.show()

            if not temp:
                results.show()
            

            lst_results = results.xyxy[0].tolist()  # im predictions (tensor)

            lst_results = [[int(lst[i]) if i < 4 else lst[i]
                            for i in range(len(lst))] for lst in lst_results]

            if not lst_results:
                print('객체가 존재하지 않습니다.')
                # Relay Module 시그널 전송
                requests.get(addr)

                results.show()

                temp = []
                continue

            elif temp:
                for i,(a, b) in enumerate(zip(lst_results[0][:4], temp)):
                    if a-b > 10:
                        print('움직임이 발생했습니다.')
                        # Relay Module 시그널 전송
                        requests.get(addr)

                        results.show()
                        break
                    else:
                        print(xy_position[i] +' 비교문제 없음')
            else:
                print('변화 없음')

            temp = lst_results[0][:4]
        else:
            continue


        
