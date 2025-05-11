import os
import csv
import json
import time
import signal
import threading
import queue
import argparse
from ultralytics import YOLO
import cv2
import torch
import logging
from datetime import datetime

# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# =============================================================================
# Signal Handling
# =============================================================================
stop_event = threading.Event()

def handle_sigint(signum, frame):
    logging.info("\nCtrl+C recebido, a parar…")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)

# =============================================================================
# Utility Functions
# =============================================================================

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def reconnect(cap_factory, max_backoff=32):
    backoff = 1
    while not stop_event.is_set():
        cap = cap_factory()
        logging.info("Reconnected to RTSP stream")
        if cap.isOpened():
            return cap
        logging.warning("Failed to open RTSP, retrying in %ss", backoff)
        time.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)
    return None

# Zoom logic
GOAL_W, GOAL_H = 480, 240
TOLERANCE = 0.15

def zoom_logic(box_w, box_h, w, h): FIXME # acho que os racios tem de ser normalizados para estarem entre 0 e 1
    ratio_w = max(box_w / w, w / box_w)
    ratio_h = max(box_h / h, h / box_h)
    if ratio_w <= 1 + TOLERANCE and ratio_h <= 1 + TOLERANCE:
        return 0
    if ratio_h <= ratio_w:
        return 1 if box_h < h else -1
    return 1 if box_w < w else -1

# =============================================================================
# GStreamer Pipelines
# =============================================================================

def build_input_pipeline(ip, port, stream, latency):
    return (
        f"rtspsrc location=rtsp://root:atlas@{ip}/{stream} latency={latency} ! "
        "queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! "
        "video/x-raw,width=640,height=480,format=BGR ! appsink drop=true sync=false"
    )


def build_output_pipeline(ip, port, w, h, fps, bitrate):
    return (
        f"appsrc ! videoconvert ! x264enc tune=zerolatency bitrate={bitrate} ! "
        "h264parse ! rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host=127.0.0.1 port={port} sync=false"
    )

# =============================================================================
# Frame Grabber Thread
# =============================================================================
class FrameGrabber(threading.Thread):
    def __init__(self, cap_factory, queue, stop_event, grab_interval=0.1, max_retries=None, retry_delay=1):
        super().__init__(daemon=True)
        self.cap_factory = cap_factory
        self.capture = cap_factory()
        self.queue = queue
        self.stop_event = stop_event
        self.dropped = 0
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.grab_interval = grab_interval 


    def run(self):
        retry = 0
        while not self.stop_event.is_set():
            if not (self.capture and self.capture.isOpened()):
                logging.warning("FrameGrabber: reconnecting...")
                self.capture = reconnect(self.cap_factory)
                retry += 1
                if self.max_retries and retry >= self.max_retries:
                    logging.error("Max retries reached, stopping grabber")
                    break
                time.sleep(self.retry_delay)
                continue
            ret, frame = self.capture.read()
            if not ret:
                logging.warning("FrameGrabber: read failed, reconnecting...")
                self.capture.release()
                continue
            try:
                self.queue.put_nowait((time.time(), frame))
            except queue.Full:
                self.dropped += 1
                if self.dropped % 100 == 0:
                    logging.warning("Dropped %d frames", self.dropped)
            time.sleep(self.grab_interval)

# =============================================================================
# CSV Helper
# =============================================================================
def write_single_line_csv(path, header, row):
    """Escreve um CSV sobrescrevendo com um header e uma única linha de dados."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Config params
    cam_ip = cfg['camera_ipv4']
    gs_ip = cfg['ground_station_ip']
    rtsp_port = cfg.get('rtsp_port', 8554)
    udp_port = cfg.get('udp_port', 5600)
    stream = cfg.get('stream_name', 'main.264')
    latency = cfg.get('latency', 100)
    fps = cfg.get('fps', 30)
    bitrate = cfg.get('bitrate', 4_000_000)
    global TOLERANCE
    TOLERANCE = cfg.get('tolerance', TOLERANCE)
    detect_int = cfg.get('detection_interval', 20)
    sel_cls = cfg.get('selected_clastime.time(),ses', [])
    max_det = cfg.get('max_det', None)

    # YOLOv11 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(cfg.get('model_path', 'yolo11n.pt')).to(device)
    names = cfg.get('class_names', ['car','person','tree'])

    # GStreamer setup
    inp = build_input_pipeline(cam_ip, rtsp_port, stream, latency)
    cap_factory = lambda: cv2.VideoCapture(inp, cv2.CAP_GSTREAMER)
    cap0 = cap_factory()
    if not cap0.isOpened(): raise RuntimeError('Cannot open RTSP')
    ret, frame = cap0.read(); cap0.release()
    if not ret: raise RuntimeError('No frame')
    h, w = frame.shape[:2]
    out_pipe = build_output_pipeline(gs_ip, udp_port, w, h, fps, bitrate)
    out = cv2.VideoWriter(out_pipe, cv2.CAP_GSTREAMER, fps, (w,h))
    if not out.isOpened(): raise RuntimeError('Cannot open Writer')

    # Single-line CSV setup
    out_dir = cfg.get('output_dir','output')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, '/home/atlas/atlas/visao/24-25/Visao-Person-Detection/output/1linha.csv')
    header = ['Timestamp','Frame','CenterX','CenterY','Zoom']

    # Start grabber
    q = queue.Queue(maxsize=10)
    grabber = FrameGrabber(cap_factory, q, stop_event)
    grabber.start()

    count = 0
    logging.info("Running...")
    while not stop_event.is_set():
        try:
            ts0, frame = q.get(timeout=1)
            if time.time()-ts0 > cfg.get('frame_age_thresh',1):
                continue
        except queue.Empty:
            if not grabber.is_alive(): break
            continue

        count += 1
        # detect
        res = model(
            frame,
            conf=cfg.get('conf_thresh',0.5),
            classes=sel_cls if sel_cls else None,
            max_det=max_det
        )
        # pick best
        best, best_conf = None, 0
        for box in res[0].boxes:
            c = float(box.conf[0]); cid=int(box.cls[0])
            if c<cfg.get('conf_thresh',0.5) or cid>=len(names): continue
            if sel_cls and cid not in sel_cls: continue
            if c>best_conf: best_conf, best = c, box

        # overlay on frame
        if best:
            x1,y1,x2,y2 = map(int,best.xyxy[0])
            bw,bh = x2-x1, y2-y1
            cx,cy = x1+bw//2, y1+bh//2
            zm = zoom_logic(bw, bh, GOAL_W, GOAL_H)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"Conf:{best_conf:.2f} Zoom:{zm}",(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        # stream
        out.write(frame)
        # CSV overwrite every detect_int
        if count % detect_int == 0:
            ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            if best:
                relx = cx - w//2
                rely = (h//2) - cy
                row = [ts, count, relx, rely, zm]
            else:
                row = [ts, 0, 0,0,0]
            write_single_line_csv(csv_path, header, row)

    logging.info("Stopping...")
    grabber.join(timeout=2)
    out.release()

if __name__=='__main__':
    main()
