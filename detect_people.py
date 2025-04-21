'''
brew install python@3.10 11 ou 12 (o 13 não é compatível com a versão da ultralytics)
brew unlink python@3.y (caso tenhas uma versão anterior; se isto der problema, desinstala a versão anterior
brew uninstall python@3.y)

brew link --overwrite python@3.x

pip3 install ultralytics==8.2.103 -q (esta versão foi a que encontrei num script talvez haja uma mais recente)
python3.10 -m venv yolo_env (cria um ambiente virtual)
source yolo_env/bin/activate (ativa o ambiente virtual)
pip3 install torch torchvision torchaudio (instala o torch)
pip3 install ultralytics (instala o ultralytics)

'''

from ultralytics import YOLO
import cv2
import math
import csv
import os
import time
import torch
import torchvision
from datetime import datetime



'''

para ter cuda 

wget https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip3 install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip3 install torchvision==0.15.1
pip3 install ultralytics

problema:
/home/atlas/.local/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: '/home/atlas/.local/lib/python3.8/site-packages/torchvision/image.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSs'If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
  warn(

'''
#parametros de zoom
GOAL_W, GOAL_H = 480, 240    
TOLERANCE       = 0.15       

def zoom_logic(width, height, goal_width, goal_height, tol=TOLERANCE):
    ratio_w = max(width / goal_width,  goal_width / width)
    ratio_h = max(height / goal_height, goal_height / height)
    if ratio_w <= 1 + tol and ratio_h <= 1 + tol:
        return 0
    if ratio_h <= ratio_w:
        return 1 if goal_height > height else -1
    return 1 if goal_width  > width  else -1

print(torch.cuda.is_available())  # Should return True
device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.device(0)

# modelo
model = YOLO("best.pt")  

#  classes
classNames = ["car", "person", "tree"]

# start webcam
def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

print("Available cameras:", list_cameras())


# o segundo argumento é para mac, para windows acho que é cv2.CAP_DSHOW mas acho que não é necessário por sequer

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)


# isto era para o meu telemovel 
w_tel = 360
h_tel = 180

# medidas do monitor do pc (isto não é preciso ser exato, mas se for muito diferente do teu pc pode dar problemas)
'''w = 3072
h = 1920
cap.set(3, w)
cap.set(4, h)'''

# Set desired resolution & FPS
w, h, fps = 640, 480, 60  # Or 1280x720 for lower resolution

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Force H264
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, fps)

# Check if settings applied
actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Resolution: {int(actual_w)}x{int(actual_h)}, FPS: {actual_fps}")




if not cap.isOpened():
    print("Could not open webcam") 
    exit()

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

csv_file_path = os.path.join(
    output_dir,
    'dados.csv'
)
sl_csv_file_path = os.path.join(output_dir, 'single_line_bounding_box_centers.csv')
# reabrir csv
def open_main_csv():
    f = open(csv_file_path, mode='w', newline='')
    w = csv.writer(f)
    w.writerow(['Timestamp', 'Frame', 'Center X', 'Center Y', 'Zoom'])
    return f, w

csv_file, csv_writer = open_main_csv()
rows_written = 0            # conta linhas


frame_count = 0
detected_objects = []

while True:
    ret, img = cap.read()
    #print(img.is_cuda())
    if not ret:
        print("Failed to capture image")
        break
    frame_count += 1

    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    img_h, img_w = img.shape[:2]
  
    start_time = time.time()
    results = model(img)
    print(f"Elapsed time of inference: {time.time() - start_time: .6f} s")

    detection_written = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1

            confidence = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # centro absoluto
            rel_cx = cx - img_w // 2                 # origem no centro da imagem
            rel_cy = (img_h // 2) - cy

            zoom_dir = zoom_logic(bw, bh, GOAL_W, GOAL_H)

            label = f"{classNames[cls]} {confidence:.2f}"
            if confidence > 0.5:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, label, (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

            detected_objects.append(
                [timestamp, frame_count, rel_cx, rel_cy, zoom_dir]
            )

                # escreve nos csvs a cada 30 frames
            if frame_count % 30 == 0:
                csv_writer.writerow(
                    [timestamp, frame_count, rel_cx, rel_cy, zoom_dir]
                )
                rows_written += 1
                # CSV de uma só linha:
                with open(sl_csv_file_path, mode='w', newline='') as csv_file2:
                    csv.writer(csv_file2).writerow(
                        [timestamp, frame_count, rel_cx, rel_cy, zoom_dir]
                    )
            detection_written = True

    if not detection_written and frame_count % 20 == 0:
        zero_row = [timestamp, 0, 0, 0, 0]
        csv_writer.writerow(zero_row)
        rows_written += 1
        with open(sl_csv_file_path, mode='w', newline='') as csv_file2:
            csv.writer(csv_file2).writerow(zero_row)

    if rows_written >= 100:
        csv_file.close()           
        csv_file, csv_writer = open_main_csv()  
        rows_written = 0           


    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow('Frame', img) 

    #cv2.waitKey(3)
    if cv2.waitKey(2) == ord('q'): 
        break

cap.release() 
cv2.destroyAllWindows() 
csv_file.close()
