from ultralytics import YOLO
import cv2
import math
import time
import torch
import os
import csv

# IPs, ports, stream e suas configuracoes
ip_tmp = "172.20.10.2"
camera_ipv4 = ip_tmp  # TODO: meter IP real
ground_station_ip = ip_tmp  # TODO: meter para IP real
rtsp_default_port = 8554
udp_default_port = 5600  # QGroundControl espera video nesta port por default
main_stream_name = "main.264"
latency = 100
bitrate = 4000000
fps = 30

# Modelo e zoom
model_path="best.pt"
classNames = ["car", "person", "tree"]
GOAL_W, GOAL_H = 480, 240
TOLERANCE = 0.15


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Using device:", device)
    return YOLO(model_path).to(device)

def save_to_csv(csv_writer, frame_count, class_name, cx, cy, x1, y1, x2, y2, zoom_dir):
    '''Grava os dados da deteção no CSV a cada 30 frames'''
    if frame_count % 30 == 0:
        timestamp = time.time()
        csv_writer.writerow([timestamp, frame_count, class_name, cx, cy, x1, y1, x2, y2, zoom_dir])

def zoom_logic(width, height, goal_width, goal_height):
    ratio_w = max(width / goal_width,  goal_width / width)
    ratio_h = max(height / goal_height, goal_height / height)
    if ratio_w <= 1 + TOLERANCE and ratio_h <= 1 + TOLERANCE:
        return 0
    if ratio_h <= ratio_w:
        return 1 if goal_height > height else -1
    return 1 if goal_width  > width  else -1

def build_gst_pipeline_input(ip, port, stream, latency):
        
    return (
        # Obter a stream RTSP da câmara SIYI pela rede (ethernet)
    f"rtspsrc location=rtsp://{camera_ipv4}:{rtsp_default_port}/{main_stream_name} latency={latency} ! "
    
    # Desempacotar (extrair o conteudo) do vídeo H.264 dos pacotes RTP
    "rtph264depay ! "
    
    # Analisa o fluxo H.264 para garantir que está pronto para descodificar
    "h264parse ! "
    
    # Descodifica o vídeo com aceleração de hardware do Jetson
    "nvv4l2decoder ! "
    
    # Converte o formato para o que precisamos
    "nvvidconv ! "
    
    # Define o formato como BGRx com canal alfa x, que serve para alinhar os dados em blocos de 4 bytes (facilita a conversão no hardware do Jetson)
    "video/x-raw, format=BGRx ! "
    
    # Converte para BGR, o formato que o OpenCV espera
    "videoconvert ! "
    
    # Envia o vídeo para o OpenCV processar
    "video/x-raw, format=BGR ! "
    
    # O appsink é onde o OpenCV vai receber os frames
    "appsink"
    )

def build_gst_pipeline_input2(ip, port, stream, latency):
    return (
        # Obter a stream UDP da câmara pela rede (ethernet)
        f'udpsrc port={port} caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! '

        # Desempacotar (extrair o conteúdo) do vídeo H.264 dos pacotes RTP
        "rtph264depay ! "

        # H264 parse para garantir que está pronto para descodificação
        "h264parse ! "

        # Decodificação de vídeo com aceleração de hardware ou software
        "nvv4l2decoder ! "  # Substituído por 'avdec_h264' para decodificação correta

        # Converte o formato para o OpenCV
        "nvvidconv ! "

        # A OpenCV vai receber os frames aqui
        "appsink sync=false"
    )


def build_gst_pipeline_output(ip, width, height, fps, bitrate):
    return (
        # "appsrc" indica que vamos usar dados provenientes de uma fonte personalizada (neste caso, o OpenCV ou o código que gera o vídeo)
    f"appsrc format=time is-live=true block=true ! "

    

    # "nvvidconv" é uma conversão otimizada para GPUs da NVIDIA
    # "nvvidconv ! "

    # Adiciona o formato desejado: NVMM memory (nvidia memory management- forca uso da GPU), resolução e framerate
    f"video/x-raw, format=I420, width=640, height=480, framerate=30/1 ! "

   # "videoconvert" converte o formato do vídeo para algo que o resto da pipeline aceite
    "videoconvert ! " 
    
    # "nvv4l2h264enc" é o codificador de vídeo H.264 da NVIDIA, que usa a aceleração por hardware (ou seja a GPU)
    # A "bitrate=4000000" define a taxa de bits para a compressão do vídeo (4 Mbps neste caso)
    # insert-sps-pps = true garante que as definicoes SPS e PPS sao enviadas (necessarias para descodificar o video H.264)
    # SPS (Sequence Parameter Set) - resolução da imagem, o formato da codificação, a taxa de compressão e outros parâmetros de nível superior (resolucao, bitrate, fps)
    # PPS (Picture Parameter Sestart_timet) -  configuração de cada frame  dentro do vídeo, como a quantização, o tipo de bloco usado e a estrutura de codificação
        # Quantização: Diminui a precisão dos dados para reduzir o tamanho do arquivo; Tipo de bloco: Define como a imagem é dividida para compressão; Estrutura de codificação: Organiza a sequência de frames para otimizar a compressão e a decodificação.
    f"x264enc tune=zerolatency speed-preset=ultrafast bitrate=800  !"



    "h264parse ! "

    # "rtph264pay" encapsula o fluxo de vídeo H.264 em pacotes RTP 
    # com config_interval=1, as configuracoes SPS e PPS sao enviadas a cada segundo. Aumentar pode tornar a transmissao mais rapida, mas pode perder-se sincronizacao
    # "pt=96" define o payload type (valor numérico usado para identificar o tipo de video que está a ser transportado nos pacotes RTP); 96 esta reservado para h.264
    # poderia acrescentar-se um MTU (Maximum Transmission Unit) = 1400, por exemplo; assim manda-se pacotes pequenos (1400 bytes), o que evita fragmentacao;
        # em principio o rtph264pay ja limita o tamanho dos pacotes ao limite da rede; se houver fragmentacao, testamos MTUs
    "rtph264pay config-interval=1 pt=96 ! "

    # "udpsink" envia os pacotes RTP para a ground station via UDP
    # "host={receiver_ip}" define o endereço IP da ground station
    # "port=5000" define a porta a usar para enviar os pacotes
    f"udpsink host={ip_tmp} port={udp_default_port} sync=false"
    )

def build_gst_pipeline_output2(ip, width, height, fps, bitrate):
    return (
        # "appsrc" indica que vamos usar dados provenientes de uma fonte personalizada (neste caso, o OpenCV ou o código que gera o vídeo)
    f"appsrc ! "

    # "videoconvert" converte o formato do vídeo para algo que o resto da pipeline aceite
    "videoconvert ! "

    # "nvvidconv" é uma conversão otimizada para GPUs da NVIDIA
    # "nvvidconv ! "

    # Adiciona o formato desejado: NVMM memory (nvidia memory management- forca uso da GPU), resolução e framerate
    # f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "

    # "nvv4l2h264enc" é o codificador de vídeo H.264 da NVIDIA, que usa a aceleração por hardware (ou seja a GPU)
    # A "bitrate=4000000" define a taxa de bits para a compressão do vídeo (4 Mbps neste caso)
    # insert-sps-pps = true garante que as definicoes SPS e PPS sao enviadas (necessarias para descodificar o video H.264)
    # SPS (Sequence Parameter Set) - resolução da imagem, o formato da codificação, a taxa de compressão e outros parâmetros de nível superior (resolucao, bitrate, fps)
    # PPS (Picture Parameter Set) -  configuração de cada frame  dentro do vídeo, como a quantização, o tipo de bloco usado e a estrutura de codificação
        # Quantização: Diminui a precisão dos dados para reduzir o tamanho do arquivo; Tipo de bloco: Define como a imagem é dividida para compressão; Estrutura de codificação: Organiza a sequência de frames para otimizar a compressão e a decodificação.
    # f"nvv4l2h264enc bitrate={bitrate} insert-sps-pps=true !"
    f"x264enc bitrate=800 !"

    # "rtph264pay" encapsula o fluxo de vídeo H.264 em pacotes RTP 
    # com config_interval=[32, 3, 3, 31, as configuracoes SPS e PPS sao enviadas a cada segundo. Aumentar pode tornar a transmissao mais rapida, mas pode perder-se sincronizacao
    # "pt=96" define o payload type (valor numérico usado para identificar o tipo de video que está a ser transportado nos pacotes RTP); 96 esta reservado para h.264
    # poderia acrescentar-se um MTU (Maximum Transmission Unit) = 1400, por exemplo; assim manda-se pacotes pequenos (1400 bytes), o que evita fragmentacao;
        # em principio o rtph264pay ja limita o tamanho dos pacotes ao limite da rede; se houver fragmentacao, testamos MTUs
    "rtph264pay config-interval=1 pt=96 ! "

    # "udpsink" envia os pacotes RTP para a ground station via UDP
    # "host={receiver_ip}" define o endereço IP da ground station
    # "port=5000" define a porta a usar para enviar os pacotes
    f"udpsink host={ip_tmp} port={udp_default_port} sync=false"
    )



def open_camera(pipeline_input):
    #print(f"Attempting to open camera with pipeline: {pipeline_input}")
    cap = cv2.VideoCapture(pipeline_input, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Failed to open camera stream. Check the pipeline and camera accessibility.")
        raise RuntimeError("Falha ao abrir a stream da câmera.")
    else:
        print("Camera stream opened successfully.")

    # Test reading a frame to ensure the stream is functional
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to read a frame from the camera stream. Ensure the camera is streaming and the pipeline is correct.")
        raise RuntimeError("Falha ao capturar frame da câmera.")
    else:
        print("Successfully read a frame from the camera stream.")

    return cap


def open_output(ip, width, height, fps, bitrate):
    pipeline_output = build_gst_pipeline_output2(ip, width, height, fps, bitrate)
    # print(f"GStreamer pipeline for output: {pipeline_output}", flush=True)  # Debugging info
    out = cv2.VideoWriter(pipeline_output, cv2.CAP_GSTREAMER, fps, (width, height))
    if not out.isOpened():
        print("Failed to open GStreamer VideoWriter. Check the pipeline configuration and GStreamer setup.")
        raise RuntimeError("Falha ao abrir GStreamer VideoWriter para envio.")
    return out


def process_frame(model, frame, frame_count, csv_writer, zoom_dir):
    '''start_time = time.time()
    results = model(frame)
    elapsed = time.time() - start_time
    print(f"Inferência em {elapsed:.3f}s")'''

    if frame.shape[2] == 4:  # If the frame has 4 channels (BGRA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    results = model(frame)
    for r in results:
        for box in r.boxes:

            # bounding boxes, confidence, class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf > 0.5 and cls_id < len(classNames):

                # centro da bounding box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Zoom
                

                # texto e formatacao
                label = f"{classNames[cls_id]} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color_rect = (255, 0, 255)
                color_text = (255, 0, 0)
                color_circle = (0, 255, 0)
                thickness_rect = 2
                thickness_text = 3
                thickness_circle = -1 # preenche o circulo
                radius = 5

                cv2.rectangle(frame, (x1, y1), (x2, y2), color_rect, thickness_rect)
                cv2.putText(frame, label, (x1, y1), font, font_scale, color_text, thickness_text)
                cv2.circle(frame, (cx, cy), radius, color_circle, thickness_circle)

                # print(f"Frame {frame_count}: {label} at ({cx}, {cy})")
                save_to_csv(csv_writer, frame_count, cls_id, cx, cy, x1, y1, x2, y2, zoom_dir)

            else:
               #print(f"Classe desconhecida: id {cls_id}")
               None
    return frame


def main():
    
    #print(torch.cuda.is_available())  # Should return True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # modelo
    print(device) # should return cuda
    model = YOLO('best.pt').to(device)
    model.eval()

    gst_in = build_gst_pipeline_input2(camera_ipv4, rtsp_default_port, main_stream_name, latency)
    cap = open_camera(gst_in)

    # Ler primeiro frame para obter dimensões
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar primeiro frame.")
        return

    h, w = frame.shape[:2]
    out = open_output(ground_station_ip, w, h, fps, bitrate)

    # Setup do CSV
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, "bounding_box_centers.csv")
    csv_file = open(csv_file_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'Frame', 'Class', 'Center X', 'Center Y', 'X1', 'Y1', 'X2', 'Y2', 'Zoom'])

   # print("Tudo pronto. A receber vídeo e a enviar deteções...")
    
    frame_count = 0
    
    while True:
       # print("entrou no while True")
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem")
            break
        
       # print("Vamos aumentar frame count")
        frame_count += 1
        zoom_dir = zoom_logic(w, h, GOAL_W, GOAL_H)
      #  print(f"Zoom: {zoom_dir}")
       # print("Zoom feito, vamos processar frame")

        inference_start = time.time()
        with torch.no_grad():
            annotated_frame = process_frame(model, frame, frame_count, csv_writer, zoom_dir)
        inference_time = time.time() - inference_start
        print(f"Inference time: {inference_time:.3f} seconds")


        send_start = time.time()
        out.write(annotated_frame)
        send_time = time.time() - send_start
        print(f"Video sending time: {send_time:.3f} seconds")

        # Measure total frame processing time
        total_time = time.time() - inference_start
        print(f"Total frame processing time: {total_time:.3f} seconds")


    cap.release()
    out.release()
    csv_file.close()
    print("Stream terminado.")


if __name__ == "__main__":
    main()


'''
correr na ground station:

gst-launch-1.0 -v \
  udpsrc port=5600 caps="application/x-rtp, media=(string)video, payload=(int)96, encoding-name=(string)H264" ! \
  rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
'''
