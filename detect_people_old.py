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

print(torch.cuda.is_available())  # Should return True

device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.device(0)

# modelo
model = YOLO('best.pt')  

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

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)


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

'''output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

csv_file_path = os.path.join(output_dir, 'bounding_box_centers.csv')
sl_csv_file_path = os.path.join(output_dir, 'single_line_bounding_box_centers.csv')

# csv file 1
csv_file = open(csv_file_path, mode = 'w', newline = '')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Class', 'Center X', 'Center Y'])

# csv file 2 (single line)
csv_file2 = open(sl_csv_file_path, mode = 'w', newline = '')
csv_writer2 = csv.writer(csv_file2)
csv_writer2.writerow(['Frame', 'Class', 'Center X', 'Center Y'])'''

frame_count = 0

detected_objects = []

while True:
    

    ret, img = cap.read()
    #print(img.is_cuda())
    if not ret:
        print("Failed to capture image")
        break
    frame_count += 1

    if (frame_count % 1 ==0):
        
        start_time = time.time()
        print("starting inference")
        #torch.cuda.synchronize()

        results = model(img)# deteta objetos na imagem
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Elapsed time of inference: {elapsed_time: .6f} seconds")
    
        
        
        for r in results:
            boxes = r.boxes 

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->", confidence)

                # nome da classe
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # centro da bounding box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                print(f"Center of {classNames[cls]}: ({cx}, {cy})")

                # texto e formatacao 
                label = f"{classNames[cls]} {confidence:.2f}"
                org = (x1, y1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                color = (255, 0, 0)
                thickness = 2

                if confidence > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # bounding box
                    cv2.putText(img, label, org, font, fontScale, color, thickness) # label da confidence 
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1) # centro da bounding box

                

                detected_objects.append([frame_count, classNames[cls], cx, cy])

                # escreve nos csvs a cada 30 frames

                '''if frame_count % 30 == 0:

                    csv_writer.writerow([frame_count, classNames[cls], cx, cy])
                    with open(sl_csv_file_path, mode = 'w', newline = '') as csv_file2:
                        csv_writer2 = csv.writer(csv_file2)
                        csv_writer2.writerow([frame_count, classNames[cls], cx, cy])'''


    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow('Frame', img) 

    #cv2.waitKey(3)
    if cv2.waitKey(2) == ord('q'): 
        break

cap.release() 
cv2.destroyAllWindows() 
#csv_file.close()

