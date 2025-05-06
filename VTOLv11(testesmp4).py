import cv2
import torch
import os
import time
import csv
from datetime import datetime
from ultralytics import YOLO

#ZOOM Configs
GOAL_W, GOAL_H = 480, 240
TOLERANCE = 0.9  # 90% de tolerância
MAX_ROWS_CSV = 100  

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    torch.cuda.set_device(0)

model = YOLO("/home/atlas/atlas/visao/24-25/Visao-Person-Detection/yolo11n.pt")
class_names = model.names

#Paths
video_path = "/home/atlas/Downloads/pessoa.mp4"
csv_main_path = "/home/atlas/atlas/visao/24-25/Visao-Person-Detection/100linhas.csv"
csv_singleline_path = "/home/atlas/atlas/visao/24-25/Visao-Person-Detection/1linha.csv"
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

def zoom_logic(width, height, goal_width, goal_height, tol=TOLERANCE):
    width_pct  = width / goal_width
    height_pct = height / goal_height
    lower = 1 - tol
    upper = 1 + tol

    if lower <= width_pct <= upper and lower <= height_pct <= upper:
        return 0

    width_dev  = abs(1 - width_pct)
    height_dev = abs(1 - height_pct)

    if height_dev >= width_dev:
        return 1 if height_pct < 1 else -1
    else:
        return 1 if width_pct < 1 else -1

def write_main_row(row, max_rows=MAX_ROWS_CSV):
    try:
        with open(csv_main_path, newline='') as f:
            rows = list(csv.reader(f))
        if rows:
            header, data = rows[0], rows[1:]
        else:
            header, data = ['Timestamp', 'Frame', 'Center X', 'Center Y', 'Zoom'], []
    except FileNotFoundError:
        header, data = ['Timestamp', 'Frame', 'Center X', 'Center Y', 'Zoom'], []

    data.append(row)
    if len(data) > max_rows:
        data = data[-max_rows:]

    with open(csv_main_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

# ---------- ALTERAÇÃO AQUI: nova função para garantir cabeçalho no ficheiro 1linha.csv ----------
def write_singleline_csv(row):
    header = ['Timestamp', 'Frame', 'Center X', 'Center Y', 'Zoom']
    with open(csv_singleline_path, mode='w', newline='') as sl_csv:
        sl_writer = csv.writer(sl_csv)
        sl_writer.writerow(header)  # escreve sempre o cabeçalho
        sl_writer.writerow(row)
# ----------------------------------------------------------------------------------------------

# Setup video
cap = cv2.VideoCapture(video_path)
w, h, fps = 640, 480, 60
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, fps)

print(f"Resolution: {int(cap.get(3))}x{int(cap.get(4))}, FPS: {cap.get(5)}")
if not cap.isOpened():
    print("Could not open video.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    frame_count += 1
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    img_h, img_w = frame.shape[:2]

    start = time.time()
    results = model(frame)
    print(f"Frame {frame_count} - Inference time: {time.time() - start:.4f}s")

    detection_written = False
    best_box = None
    best_conf = 0

    for box in results[0].boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id]

        if label != "person" or conf < 0.3:
            continue

        if conf > best_conf:
            best_conf = conf
            best_box = box

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rel_cx = cx - img_w // 2
        rel_cy = (img_h // 2) - cy
        box_w = x2 - x1
        box_h = y2 - y1

        zoom_dir = zoom_logic(box_w, box_h, GOAL_W, GOAL_H)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{best_conf:.2f}, zoom {zoom_dir}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        if frame_count % 1 == 0:
            row = [timestamp, frame_count, rel_cx, rel_cy, zoom_dir]
            write_main_row(row)
            write_singleline_csv(row)  # ---------- ALTERADO AQUI: chamada à nova função ----------
            detection_written = True

    # Caso 0 deteções
    if not detection_written and frame_count % 20 == 0:
        zero_row = [timestamp, 0, 0, 0, 0]
        write_main_row(zero_row)
        write_singleline_csv(zero_row)  # ---------- ALTERADO AQUI também ----------


    #cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.imshow('Frame', frame)

    #if cv2.waitKey(2) == ord('q'):
    #    break

cap.release()
#cv2.destroyAllWindows()
