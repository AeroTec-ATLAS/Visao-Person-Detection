import cv2
import torch
import os
import time
import csv
from datetime import datetime
from ultralytics import YOLO

# ZOOM Configs
GOAL_W, GOAL_H = 480, 240
TOLERANCE = 0.10 
MAX_ROWS_CSV = 100  
RTSP_URL = "rtsp://root:atlas@192.168.0.10/stream=0"

GST_PIPELINE = (
    "rtspsrc location=rtsp://root:atlas@192.168.0.10/stream=0 latency=100 ! queue ! "
    "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! "
    "video/x-raw,width=640,height=480,format=BGR ! appsink drop=true sync=false"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    torch.cuda.set_device(0)

model = YOLO("/home/atlas/atlas/visao/24-25/Visao-Person-Detection/yolo11n.pt")
class_names = model.names

# Paths
base_path = "/home/atlas/atlas/visao/24-25/Visao-Person-Detection"
csv_main_path = os.path.join(base_path, "100linhas.csv")
csv_singleline_path = os.path.join(base_path, "1linha.csv")
os.makedirs(base_path, exist_ok=True)

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

# Setup video
cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("Failed to open video stream.")
print("Video stream opened successfully.")

frame_count = 0
header = ['Timestamp', 'Frame', 'Center X', 'Center Y', 'Zoom']

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame vazio - stream parada")
        continue

    frame_count += 1
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    img_h, img_w = frame.shape[:2]

    start = time.time()
    results = model(frame)
    print(f"Frame {frame_count} - Inference time: {time.time() - start:.3f}s")

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

    detection_written = False

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rel_cx = cx - img_w // 2
        rel_cy = (img_h // 2) - cy
        box_w = x2 - x1
        box_h = y2 - y1

        zoom_dir = zoom_logic(box_w, box_h, GOAL_W, GOAL_H)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.putText(frame, f"{best_conf:.2f}, zoom {zoom_dir}", (x1, y1 - 10),
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        if frame_count % 20 == 0:
            row = [timestamp, frame_count, rel_cx, rel_cy, zoom_dir]
            write_main_row(row)
            with open(csv_singleline_path, mode='w', newline='') as sl_csv:
                writer = csv.writer(sl_csv)
                writer.writerow(header)
                writer.writerow(row)
            detection_written = True

    if not detection_written and frame_count % 20 == 0:
        zero_row = [timestamp, 0, 0, 0, 0]
        write_main_row(zero_row)
        with open(csv_singleline_path, mode='w', newline='') as sl_csv:
            writer = csv.writer(sl_csv)
            writer.writerow(header)
            writer.writerow(zero_row)

    #cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.imshow('Frame', frame)

    #if cv2.waitKey(2) == ord('q'):
    #    break

cap.release()
#cv2.destroyAllWindows()