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

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")


# =============================================================================
# Signal Handling
# =============================================================================

# Cria um Event para sinalizar stop
stop_event = threading.Event()

# Handler para SIGINT (Ctrl+C)
def handle_sigint(signum, frame):
    logging.info("\nCtrl+c recebido, a parar…")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)


#=============================================================================
# Utility Functions
# =============================================================================

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def reconnect(cap_factory, max_backoff=32):
    backoff = 1
    while not stop_event.is_set():
        cap = cap_factory()  # função que cria VideoCapture
        logging.info("Reconnected to RTSP stream")
        if cap.isOpened():
            return cap
        logging.warning("Failed to open RTSP, retrying in %ss", backoff)
        time.sleep(backoff)
        backoff = min(backoff*2, max_backoff)
    return None


GOAL_W, GOAL_H = 480, 240
TOLERANCE = 0.15

def zoom_logic(box_w, box_h, goal_w, goal_h):
    ratio_w = max(box_w / goal_w, goal_w / box_w)
    ratio_h = max(box_h / goal_h, goal_h / box_h)
    if ratio_w <= 1 + TOLERANCE and ratio_h <= 1 + TOLERANCE:
        return 0  # dentro da tolerância
    if ratio_h <= ratio_w:
        return 1 if goal_h > box_h else -1
    return 1 if goal_w > box_w else -1



# =================================================================================
# GStreamer Pipelines
# ================================================================================

def build_input_pipeline(ip, port, stream, latency):
    return (
        # Obter a stream RTSP da câmara SIYI pela rede (ethernet)
    f"rtspsrc location=rtsp://root:atlas@192.168.0.10/stream=0 latency=100 ! queue ! "
    "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! "
    "video/x-raw,width=640,height=480,format=BGR ! appsink drop=true sync=false"
    )

def build_output_pipeline(ip, port, width, height, fps, bitrate):
    return (
        # "appsrc" indica que vamos usar dados provenientes de uma fonte personalizada (neste caso, o OpenCV ou o código que gera o vídeo)
    f"appsrc ! "

    # "videoconvert" converte o formato do vídeo para algo que o resto da pipeline aceite
    "videoconvert ! "

    # "nvvidconv" é uma conversão otimizada para GPUs da NVIDIA
    "nvvidconv ! "

    # Adiciona o formato desejado: NVMM memory (nvidia memory management- forca uso da GPU), resolução e framerate
    f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "

    # "nvv4l2h264enc" é o codificador de vídeo H.264 da NVIDIA, que usa a aceleração por hardware (ou seja a GPU)
    # A "bitrate=4000000" define a taxa de bits para a compressão do vídeo (4 Mbps neste caso)
    # insert-sps-pps = true garante que as definicoes SPS e PPS sao enviadas (necessarias para descodificar o video H.264)
    # SPS (Sequence Parameter Set) - resolução da imagem, o formato da codificação, a taxa de compressão e outros parâmetros de nível superior (resolucao, bitrate, fps)
    # PPS (Picture Parameter Set) -  configuração de cada frame  dentro do vídeo, como a quantização, o tipo de bloco usado e a estrutura de codificação
        # Quantização: Diminui a precisão dos dados para reduzir o tamanho do arquivo; Tipo de bloco: Define como a imagem é dividida para compressão; Estrutura de codificação: Organiza a sequência de frames para otimizar a compressão e a decodificação.
    f"nvv4l2h264enc bitrate={bitrate} insert-sps-pps=true !"

    # "rtph264pay" encapsula o fluxo de vídeo H.264 em pacotes RTP 
    # com config_interval=1, as configuracoes SPS e PPS sao enviadas a cada segundo. Aumentar pode tornar a transmissao mais rapida, mas pode perder-se sincronizacao
    # "pt=96" define o payload type (valor numérico usado para identificar o tipo de video que está a ser transportado nos pacotes RTP); 96 esta reservado para h.264
    # poderia acrescentar-se um MTU (Maximum Transmission Unit) = 1400, por exemplo; assim manda-se pacotes pequenos (1400 bytes), o que evita fragmentacao;
        # em principio o rtph264pay ja limita o tamanho dos pacotes ao limite da rede; se houver fragmentacao, testamos MTUs
    "rtph264pay config-interval=1 pt=96 ! "

    # "udpsink" envia os pacotes RTP para a ground station via UDP
    # "host={receiver_ip}" define o endereço IP da ground station
    # "port=5000" define a porta a usar para enviar os pacotes
    f"udpsink host={ip} port={port}"
    )

def build_input_pipeline2(ip, port, stream, latency):
    return (
        # Obter a stream UDP da câmara pela rede (ethernet)
        f'udpsrc port={port} caps="application/x-rtp, media=(string)video, payload=(int)96, encoding-name=(string)H264" ! '

        # Desempacotar (extrair o conteúdo) do vídeo H.264 dos pacotes RTP
        "rtph264depay ! "

        # H264 parse para garantir que está pronto para descodificação
        "h264parse ! "

        # Decodificação de vídeo com aceleração de hardware ou software
        "avdec_h264 ! "  # Substituído por 'avdec_h264' para decodificação correta

        # Converte o formato para o OpenCV
        "videoconvert ! "

        # A OpenCV vai receber os frames aqui
        "appsink sync=false"
    )

def build_output_pipeline2(ip, port, width, height, fps, bitrate):
    return (
        # "appsrc" indica que vamos usar dados provenientes de uma fonte personalizada (neste caso, o OpenCV ou o código que gera o vídeo)
    f"appsrc ! "

    # "videoconvert" converte o formato do vídeo para algo que o resto da pipeline aceite
    "videoconvert ! "

    # "nvvidconv" é uma conversão otimizada para GPUs da NVIDIA
    # "nvvidconv ! "

    # Adiciona o formato desejado: NVMM memory (nvidia memory management- forca uso da GPU), resolução e framerate
    f"video/x-raw, format=NV12, width={width}, height={height}, framerate={fps}/1 ! "

    # "nvv4l2h264enc" é o codificador de vídeo H.264 da NVIDIA, que usa a aceleração por hardware (ou seja a GPU)
    # A "bitrate=4000000" define a taxa de bits para a compressão do vídeo (4 Mbps neste caso)
    # insert-sps-pps = true garante que as definicoes SPS e PPS sao enviadas (necessarias para descodificar o video H.264)
    # SPS (Sequence Parameter Set) - resolução da imagem, o formato da codificação, a taxa de compressão e outros parâmetros de nível superior (resolucao, bitrate, fps)
    # PPS (Picture Parameter Set) -  configuração de cada frame  dentro do vídeo, como a quantização, o tipo de bloco usado e a estrutura de codificação
        # Quantização: Diminui a precisão dos dados para reduzir o tamanho do arquivo; Tipo de bloco: Define como a imagem é dividida para compressão; Estrutura de codificação: Organiza a sequência de frames para otimizar a compressão e a decodificação.
    # f"nvv4l2h264enc bitrate={bitrate} insert-sps-pps=true !"
    f"x264enc tune=zerolatency bitrate={bitrate}  !"

    # "rtph264pay" encapsula o fluxo de vídeo H.264 em pacotes RTP 
    # com config_interval=1, as configuracoes SPS e PPS sao enviadas a cada segundo. Aumentar pode tornar a transmissao mais rapida, mas pode perder-se sincronizacao
    # "pt=96" define o payload type (valor numérico usado para identificar o tipo de video que está a ser transportado nos pacotes RTP); 96 esta reservado para h.264
    # poderia acrescentar-se um MTU (Maximum Transmission Unit) = 1400, por exemplo; assim manda-se pacotes pequenos (1400 bytes), o que evita fragmentacao;
        # em principio o rtph264pay ja limita o tamanho dos pacotes ao limite da rede; se houver fragmentacao, testamos MTUs
    
    "h264parse ! "

    "rtph264pay config-interval=1 pt=96 ! "

    # "udpsink" envia os pacotes RTP para a ground station via UDP
    # "host={receiver_ip}" define o endereço IP da ground station
    # "port=5000" define a porta a usar para enviar os pacotes
    f"udpsink host={ip} port={port} sync=false"
    )



# =============================================================================
# Frame Grabber Thread
# =============================================================================

class FrameGrabber(threading.Thread):
    """
    Background thread that continuously reads frames from a video source
    and puts them into a queue for processing.
    """
    def __init__(self, cap_factory, frame_queue, stop_event, max_retries=None, retry_delay=1):
        super().__init__(daemon=True) # daemon = True - faz com que a thread termine automaticamente quando o processo principal acabar
        self.cap_factory = cap_factory
        self.capture = self.cap_factory()
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.dropped = 0
        self.max_retries = max_retries  
        self.retry_delay = retry_delay
        
    def run(self):
        retry_count = 0
        while not self.stop_event.is_set():
        # Ensure the capture is valid and reconnect if necessary
            if self.capture is None or not self.capture.isOpened():
                logging.warning("Frame grabber: capture is not opened, attempting to reconnect...")
                self.capture = reconnect(self.cap_factory)
                if self.capture is None:
                    retry_count += 1
                    if self.max_retries is not None and retry_count >= self.max_retries:
                        logging.error("Frame grabber: maximum retries reached, exiting thread.")
                        break
                    logging.warning(f"Frame grabber: reconnect failed, retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                    continue
                retry_count = 0  # Reset retry count after successful reconnection

        # Read a frame
            ret, frame = self.capture.read()
            if not ret:
                logging.warning("Frame grabber: read failed, reconnecting...")
                self.capture.release()
                continue

            try:
                self.frame_queue.put_nowait((time.time(), frame))
            except queue.Full:
                self.dropped += 1
                if self.dropped % 100 == 0:
                    logging.warning("Dropped %d frames so far", self.dropped)
            # time.sleep(0.001)  small sleep to prevent busy-wait, not sure if i want it because it slows down the processing


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function to run the detection + streaming pipeline."""

    FRAME_AGE_THRESHOLD = 1  # in seconds

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to JSON config')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cam_ip = cfg['camera_ipv4']
    gs_ip = cfg['ground_station_ip']
    rtsp_port = cfg.get('rtsp_port', 8554) # ter aqui o numero das ports, nome da stream, etc sao fallbacks de seguranca, caso me esqueca de definir no config.json
    udp_port = cfg.get('udp_port', 5600)
    stream_name = cfg.get('stream_name', 'main.264')
    latency = cfg.get('latency', 100)
    fps = cfg.get('fps', 30)
    bitrate = cfg.get('bitrate', 4_000_000)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(cfg.get('model_path', 'best.pt')).to(device)
    class_names = cfg.get('class_names', ['car','person','tree'])

    # GStreamer setup
    inp_pipe = build_input_pipeline(cam_ip, rtsp_port, stream_name, latency)
    cap_factory = lambda: cv2.VideoCapture(inp_pipe, cv2.CAP_GSTREAMER)

    # Initial capture to get frame size
    cap_temp = cap_factory()
    if not cap_temp.isOpened():
        raise RuntimeError('Cannot open RTSP stream')
    ret, frame = cap_temp.read()
    cap_temp.release()
    if not ret:
        raise RuntimeError('Failed to read initial frame')
    
    # OpenCV returns (height, width), but GStreamer expects (width, height)
    height, width = frame.shape[:2]

    out_pipe = build_output_pipeline2(gs_ip, udp_port, width, height, fps, bitrate)
    out = cv2.VideoWriter(out_pipe, cv2.CAP_GSTREAMER, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError('Não foi possível abrir VideoWriter')

    # CSV setup
    os.makedirs('output', exist_ok=True)
    csv_path = os.path.join('output', 'bboxes.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp','Frame','Class','CenterX','CenterY','X1','Y1','X2','Y2','Zoom'])

    # Frame queue & grabber
    frame_queue = queue.Queue(maxsize=10)
    grabber = FrameGrabber(cap_factory, frame_queue, stop_event)
    grabber.start()

    frame_count = 0
    logging.info("Running... Press Ctrl+C to stop")

    while not stop_event.is_set():
        try:
            frame_timestamp, frame = frame_queue.get(timeout=1)  # Grab the next frame from the queue

            # If the frame is older than the threshold, discard it
            if time.time() - frame_timestamp > FRAME_AGE_THRESHOLD:
                logging.info(f"Discarding frame older than {FRAME_AGE_THRESHOLD} seconds.")
                continue  # Skip processing of this frame

        except queue.Empty:
            if not grabber.is_alive():
                logging.error("Frame grabber thread has stopped unexpectedly.")
                break
            continue

        frame_count += 1
        t0 = time.time()
        results = model(frame)
        inference_time = time.time() - t0
        logging.debug("Inference time: %.3fs", inference_time)

        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # (x1,y1)=top-left, (x2,y2)=bottom-right (nota: 0,0 = top left do frame)
                if x2 <= x1 or y2 <= y1:
                    continue
                box_w, box_h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if conf < cfg.get('conf_thresh', 0.5) or cls_id >= len(class_names):
                    continue

                zoom = zoom_logic(box_w, box_h, GOAL_W, GOAL_H)
                cx, cy = x1 + box_w // 2, y1 + box_h // 2

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,255), 2)
                cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
                cv2.putText(frame, f"{class_names[cls_id]} {conf:.2f}", (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                # CSV
                ts = time.time()
                csv_writer.writerow([ts, frame_count, class_names[cls_id], cx, cy,
                                     x1, y1, x2, y2, zoom])
                csv_file.flush()
                os.fsync(csv_file.fileno())

        try: 
            out.write(frame)
        except Exception as e:
            logging.error("Failed to write frame: %s", e)
            break

        # Periodic status log
        if frame_count % 100 == 0:
            logging.info("Frame %d, queue size %d, dropped %d", frame_count, frame_queue.qsize(), grabber.dropped)


    # Cleanup
    logging.info("Stopping grabber thread...")
    grabber.join(timeout=2)
    out.release()
    csv_file.close()
    logging.info("Shutdown complete.")

if __name__ == '__main__':
    main()





'''
correr com:
python pipeline.py --config config.json 

ou com verbose 
python pipeline.py --config config.json --verbose


correr na ground station:
gst-launch-1.0 -v \
  udpsrc port=5600 caps="application/x-rtp, media=(string)video, payload=(int)96, encoding-name=(string)H264" ! \
  rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
'''