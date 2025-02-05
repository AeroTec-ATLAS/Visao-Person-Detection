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

# modelo
model = YOLO('best.pt')  

#  classes
classNames = ["car", "person", "tree"]

# start webcam

'''
def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

print("Available cameras:", list_cameras())
'''

# o segundo argumento é para mac, para windows acho que é cv2.CAP_DSHOW mas acho que não é necessário por sequer

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION) 

# medidas do monitor do pc (isto não é preciso ser exato, mas se for muito diferente do teu pc pode dar problemas)
w = 3072
h = 1920
# isto era para o meu telemovel 
w_tel = 360
h_tel = 180

cap.set(3, w)
cap.set(4, h)


if not cap.isOpened():
    print("Could not open webcam") 
    exit()

output_dir = 'output'
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
csv_writer2.writerow(['Frame', 'Class', 'Center X', 'Center Y'])

frame_count = 0

detected_objects = []

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    frame_count += 1

    results = model(img) # deteta objetos na imagem

    for r in results:
        boxes = r.boxes 

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

            # poe um retangulo 
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

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
            fontScale = 2
            color = (255, 0, 0)
            thickness = 2

            if confidence > 0.5:
                cv2.putText(img, label, org, font, fontScale, color, thickness)

            # circle no centro bounding box
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

            detected_objects.append([frame_count, classNames[cls], cx, cy])

            # escreve nos csvs a cada 30 frames

            if frame_count % 30 == 0:

                csv_writer.writerow([frame_count, classNames[cls], cx, cy])
                with open(sl_csv_file_path, mode = 'w', newline = '') as csv_file2:
                    csv_writer2 = csv.writer(csv_file2)
                    csv_writer2.writerow([frame_count, classNames[cls], cx, cy])



    cv2.imshow('Webcam', img) 
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release() 
cv2.destroyAllWindows() 
csv_file.close()

