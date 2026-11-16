import cv2
from ultralytics import YOLO

# Carrega o modelo YOLOv8 "nano" - o mais leve, ótimo para rodar em CPU e em tempo real.
# Na primeira execução ele baixa o arquivo de pesos automaticamente (~6MB).
modelo = YOLO("yolov8n.pt")

# Tradução de algumas classes do dataset COCO (em inglês) para português.
TRADUCAO_CLASSES = {
    "person": "pessoa",
    "cat": "gato",
    "dog": "cachorro",
    "bird": "pássaro",
    "horse": "cavalo",
    "cow": "vaca",
    "car": "carro",
    "bicycle": "bicicleta",
    "motorcycle": "moto",
    "bottle": "garrafa",
    "cell phone": "celular",
    "backpack": "mochila",
    "chair": "cadeira",
    "laptop": "notebook",
}

# Reduz a imagem antes de rodar a IA - acelera bastante a inferência em webcams de
# resolução alta (ex: 1280x720), sem perder muita precisão para esse tipo de câmera.
LARGURA_MAXIMA_ANALISE = 640


def classificar_frame(frame, confianca_minima=0.4):
    """
    Roda a IA sobre o frame da câmera e retorna todas as detecções encontradas:
        (nome_da_classe, confianca, x, y, largura, altura)

    x, y, largura, altura já vêm na escala do frame ORIGINAL, prontos para desenhar.
    Retorna lista vazia se nada for reconhecido com confiança suficiente.
    """
    altura_original, largura_original = frame.shape[:2]

    escala = 1.0
    frame_para_ia = frame
    if largura_original > LARGURA_MAXIMA_ANALISE:
        escala = LARGURA_MAXIMA_ANALISE / largura_original
        nova_altura = int(altura_original * escala)
        frame_para_ia = cv2.resize(frame, (LARGURA_MAXIMA_ANALISE, nova_altura))

    resultados = modelo(frame_para_ia, verbose=False)

    deteccoes = []
    for resultado in resultados:
        for caixa in resultado.boxes:
            confianca = float(caixa.conf[0])
            if confianca < confianca_minima:
                continue

            classe_id = int(caixa.cls[0])
            nome_ingles = modelo.names[classe_id]
            nome_pt = TRADUCAO_CLASSES.get(nome_ingles, nome_ingles)

            # Desfaz o redimensionamento para a caixa bater com o frame original
            x1, y1, x2, y2 = caixa.xyxy[0]
            x1, y1, x2, y2 = x1 / escala, y1 / escala, x2 / escala, y2 / escala

            x, y = int(x1), int(y1)
            largura, altura = int(x2 - x1), int(y2 - y1)

            deteccoes.append((nome_pt, confianca, x, y, largura, altura))

    return deteccoes
