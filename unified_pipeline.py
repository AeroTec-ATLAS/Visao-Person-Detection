import os
import cv2
import csv
import time
import signal
import threading
import queue
import argparse
from ultralytics import YOLO
import torch
import logging
import json


# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Convert to dot-access for ease
class Config:
    def __init__(self, cfg): self.__dict__.update(cfg)
args = Config(config)


# =======================
# Logging
# =======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =======================
# Signal Handling
# =======================
stop_event = threading.Event()
def handle_sigint(signum, frame):
    logging.info("SIGINT received. Stopping threads...")
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)

# =======================
# Zoom Logic
# =======================
TOLERANCE = 0.15
GOAL_W, GOAL_H = 480, 240
def compute_zoom(box_w, box_h, goal_w=GOAL_W, goal_h=GOAL_H):
    rw, rh = max(box_w/goal_w, goal_w/box_w), max(box_h/goal_h, goal_h/box_h)
    if rw <= 1 + TOLERANCE and rh <= 1 + TOLERANCE:
        return 0
    return 1 if (rh <= rw and goal_h > box_h) or (rw < rh and goal_w > box_w) else -1

# =======================
# GStreamer
# =======================
def gst_input(rtsp_ip, rtsp_port, stream, latency):
    return f"rtspsrc location=rtsp://{rtsp_ip}:{rtsp_port}/{stream} latency={latency} ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"

def gst_output(udp_ip, udp_port, bitrate):
    return f"appsrc ! videoconvert ! x264enc bitrate={bitrate} speed-preset=superfast tune=zerolatency ! rtph264pay config-interval=1 pt=96 ! udpsink host={udp_ip} port={udp_port}"

# =======================
# Threads
# =======================
class FrameGrabber(threading.Thread):
    def __init__(self, cap, frame_queue):
        super().__init__(daemon=True)
        self.cap = cap
        self.frame_queue = frame_queue

    def run(self):
        while not stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Frame read failed. Reconnecting...")
                time.sleep(1)
                continue
            self.frame_queue.put(frame)
        self.cap.release()

class InferenceWorker(threading.Thread):
    def __init__(self, model, frame_queue, result_queue, skip):
        super().__init__(daemon=True)
        self.model = model
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.skip = skip
        self.counter = 0

    def run(self):
        while not stop_event.is_set():
            frame = self.frame_queue.get()
            if frame is None:
                break
            if self.counter % (self.skip + 1) == 0:
                result = self.model(frame)[0]
                self.result_queue.put((frame, result))
            self.counter += 1

class Sender(threading.Thread):
    def __init__(self, result_queue, writer, udp_ip, udp_port, fps):
        super().__init__(daemon=True)
        self.queue = result_queue
        self.writer = writer
        self.fps = fps
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.pipe = self._create_gst_pipe()

    def _create_gst_pipe(self):
        gst_str = (
            f'appsrc ! videoconvert ! video/x-raw,format=BGR ! '
            f'x264enc speed-preset=ultrafast tune=zerolatency bitrate=2000 ! '
            f'rtph264pay config-interval=1 pt=96 ! '
            f'udpsink host={self.udp_ip} port={self.udp_port}'
        )
        return cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, self.fps, (640, 480), True)

    def run(self):
        while not stop_event.is_set():
            try:
                frame, det = self.queue.get(timeout=1)
            except queue.Empty:
                continue
            timestamp = time.time()
            for *box, conf, cls in det.boxes.data.tolist():
                x1, y1, x2, y2 = map(int, box)
                zoom = compute_zoom(x2-x1, y2-y1)
                self.writer.writerow([timestamp, cls, x1, y1, x2, y2, zoom])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            self.pipe.write(frame)

# =======================
# Main
# =======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.model).to(device)
    if device.type == 'cuda':
        model.model.half()

    cap = cv2.VideoCapture(gst_input(args.rtsp_ip, args.rtsp_port, args.stream, args.latency), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        logging.error("Failed to open video stream.")
        return

    frame_q = queue.Queue(maxsize=10)
    result_q = queue.Queue(maxsize=10)

    with open(args.csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'class', 'x1', 'y1', 'x2', 'y2', 'zoom'])

        grabber = FrameGrabber(cap, frame_q)
        inf_worker = InferenceWorker(model, frame_q, result_q, args.skip_frames)
        sender = Sender(result_q, writer, args.udp_ip, args.udp_port, args.fps)

        grabber.start()
        inf_worker.start()
        sender.start()

        while not stop_event.is_set():
            time.sleep(0.1)

        frame_q.put(None)
        result_q.put(None)
        grabber.join()
        inf_worker.join()
        sender.join()

    logging.info("Pipeline terminated.")

if __name__ == "__main__":
    main()


'''
correr com:
python unified_pipeline.py 



correr na ground station:

sudo apt update
sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-base \
                    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
                    gstreamer1.0-plugins-ugly gstreamer1.0-libav

chmod +x receive_stream.sh
./receive_stream.sh

'''