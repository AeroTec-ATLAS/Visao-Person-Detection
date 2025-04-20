from ultralytics import YOLO
import cv2
import math
import time
import torch

# Verificar se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Carregar o modelo e garantir que esta na GPU
model = YOLO('best.pt').to(device)

# classes
classNames = ["car", "person", "tree"]


# Pipeline GStreamer para receber vídeo RTSP da câmara SIYI A8 mini, ligada ao Jetson por cabo Ethernet
camera_ipv4 = 0 # FIXME meter ip da camera
rtsp_default_port = 8554
main_stream_name = "main.264" # assumindo o codec default, sera h.264
latency = 100 # tentamos 100 ms. Se estiver muito lento, baixamos. Se estivermos a perder muitos pacotes, aumentamos


gst_cam = (
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


cap = cv2.VideoCapture(gst_cam, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Falha ao abrir o stream da câmara.")
    exit()


# IP da ground station que vai receber o vídeo
ground_station_ip = "192.168.1.100"  # FIXME mudar para ip real da ground station
bitrate = 4000000 # 4 Mbps (baixo, mas permite HD)
fps = 30

# Pipeline GStreamer para enviar vídeo por UDP para a ground station
gst_out = (
    # 'appsrc' indica que vamos usar dados provenientes de uma fonte personalizada (neste caso, o OpenCV ou o código que gera o vídeo)
    f'appsrc ! '

    # 'videoconvert' converte o formato do vídeo para algo que o resto da pipeline aceite
    'videoconvert ! '

    # 'nvvidconv' é uma conversão otimizada para GPUs da NVIDIA
    'nvvidconv ! '

    # Adiciona o formato desejado: NVMM memory (nvidia memory management- forca uso da GPU), resolução e framerate
    f'video/x-raw(memory:NVMM), width=640, height=480, framerate={fps}/1 ! '

    # 'nvv4l2h264enc' é o codificador de vídeo H.264 da NVIDIA, que usa a aceleração por hardware (ou seja a GPU)
    # A 'bitrate=4000000' define a taxa de bits para a compressão do vídeo (4 Mbps neste caso)
    # insert-sps-pps = true garante que as definicoes SPS e PPS sao enviadas (necessarias para descodificar o video H.264)
    # SPS (Sequence Parameter Set) - resolução da imagem, o formato da codificação, a taxa de compressão e outros parâmetros de nível superior (resolucao, bitrate, fps)
    # PPS (Picture Parameter Set) -  configuração de cada frame  dentro do vídeo, como a quantização, o tipo de bloco usado e a estrutura de codificação
        # Quantização: Diminui a precisão dos dados para reduzir o tamanho do arquivo; Tipo de bloco: Define como a imagem é dividida para compressão; Estrutura de codificação: Organiza a sequência de frames para otimizar a compressão e a decodificação.
    f'nvv4l2h264enc bitrate={bitrate} insert-sps-pps=true !'

    # 'rtph264pay' encapsula o fluxo de vídeo H.264 em pacotes RTP 
    # com config_interval=1, as configuracoes SPS e PPS sao enviadas a cada segundo. Aumentar pode tornar a transmissao mais rapida, mas pode perder-se sincronizacao
    # 'pt=96' define o payload type (valor numérico usado para identificar o tipo de video que está a ser transportado nos pacotes RTP); 96 esta reservado para h.264
    # poderia acrescentar-se um MTU (Maximum Transmission Unit) = 1400, por exemplo; assim manda-se pacotes pequenos (1400 bytes), o que evita fragmentacao;
        # em principio o rtph264pay ja limita o tamanho dos pacotes ao limite da rede; se houver fragmentacao, testamos MTUs
    'rtph264pay config-interval=1 pt=96 ! '

    # 'udpsink' envia os pacotes RTP para a ground station via UDP
    # 'host={receiver_ip}' define o endereço IP da estação de recepção (ground station)
    # 'port=5000' define a porta a usar para enviar os pacotes
    f'udpsink host={ground_station_ip} port=5000'
)

# Abre o pipeline de envio
out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, fps, (w, h), True)
if not out.isOpened():
    print("Falha ao abrir GStreamer VideoWriter para envio.")
    exit()

frame_count = 0

print("Tudo pronto. A receber vídeo e a enviar deteções...")

while True:
    ret, img = cap.read()
    if not ret:
        print("Falha ao capturar imagem")
        break

    frame_count += 1
    start_time = time.time()

    # Inference
    results = model(img)
    elapsed = time.time() - start_time
    print(f"Inferência em {elapsed:.3f}s")

    for r in results:
        for box in r.boxes:
            # Caixa e classe
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf > 0.5 and cls_id < len(classNames):
                label = f"{classNames[cls_id]} {conf:.2f}"
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Desenho na imagem
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

                print(f"Frame {frame_count}: {label} at ({cx}, {cy})")

    # Envia frame anotado para a ground station
    out.write(img)

    # Sai com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpeza
cap.release()
out.release()
cv2.destroyAllWindows()




'''
gst-launch-1.0 -e nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=640, height=480, framerate=25/1'  insert-sps-pps=true ! rtph264pay mtu=1400 ! multiudpsink clients=127.0.0.1:5602,127.0.0.1:5603 sync=false async=false
'''