import cv2
import numpy as np

from ia_classificacao import classificar_frame
from analise_comportamento import RastreadorDeComportamento
from analise_fft import filtrar_ruido_por_frequencia

# --- Parâmetros ajustáveis (calibre testando com sua câmera) ---
LIMIAR_DIFERENCA = 25          # sensibilidade da comparação entre frames - menor = mais sensível
TAMANHO_MINIMO_CONTORNO = 30   # largura/altura mínima (px) para considerar "movimento real"
AREA_MINIMA_CONTORNO = 500     # área mínima (px²) do contorno - filtra ruído de sensor à noite
BRILHO_LIMIAR_NOTURNO = 80     # média de brilho (0-255) abaixo da qual consideramos "ambiente escuro"
RAIO_CORTE_FFT = 40            # raio do filtro passa-baixa no domínio da frequência

frame_anterior = None
rastreador = RastreadorDeComportamento()

# Guarda o último espectro de frequência calculado - usado só para mostrar na
# interface/slide o "antes e depois" do filtro FFT, não afeta a detecção.
ultimo_espectro_fft = None

# Detecções do último frame analisado - usadas para o resumo de atividade
# (base do chatbot para responder "como foi a última hora?").
ultimas_deteccoes = []


def resetar_estado():
    """Chame sempre que (re)ativar a segurança, para não comparar frames de momentos diferentes."""
    global frame_anterior, rastreador
    frame_anterior = None
    rastreador = RastreadorDeComportamento()


def _ambiente_escuro(cinza):
    return cinza.mean() < BRILHO_LIMIAR_NOTURNO


def _e_movimento_real(contorno):
    x, y, largura, altura = cv2.boundingRect(contorno)
    if largura < TAMANHO_MINIMO_CONTORNO or altura < TAMANHO_MINIMO_CONTORNO:
        return False
    return cv2.contourArea(contorno) > AREA_MINIMA_CONTORNO


def detectar_movimento(frame):
    """
    Analisa um frame e retorna (alertas, threshold):
      - alertas: lista de strings com comportamentos incomuns encontrados neste frame
      - threshold: imagem preto-e-branco mostrando onde houve movimento (branco = mudou)
    """
    global frame_anterior, ultimo_espectro_fft, ultimas_deteccoes
    ultimas_deteccoes = []

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    noite = _ambiente_escuro(cinza)
    if noite:
        # Realça o contraste - essencial à noite, quando a diferença de brilho entre
        # um objeto e o fundo costuma ser sutil demais para o detector enxergar.
        cinza = cv2.equalizeHist(cinza)

        # FFT: filtra o ruído de sensor (grão/flicker) no domínio da frequência.
        # À noite o sensor da câmera amplifica o sinal para compensar a pouca luz,
        # e isso produz ruído de alta frequência que o GaussianBlur sozinho não
        # remove bem sem borrar demais a imagem. O filtro passa-baixa em frequência
        # ataca justamente esse tipo de ruído, preservando as formas dos objetos.
        cinza, ultimo_espectro_fft = filtrar_ruido_por_frequencia(cinza, raio_corte=RAIO_CORTE_FFT)

    cinza = cv2.GaussianBlur(cinza, (21, 21), 0)

    if frame_anterior is None:
        frame_anterior = cinza
        return [], np.zeros_like(cinza)

    diferenca = cv2.absdiff(frame_anterior, cinza)
    frame_anterior = cinza

    _, threshold = cv2.threshold(diferenca, LIMIAR_DIFERENCA, 255, cv2.THRESH_BINARY)
    threshold = cv2.dilate(threshold, None, iterations=2)
    contornos, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not any(_e_movimento_real(c) for c in contornos):
        return [], threshold

    # A IA roda UMA única vez por frame (não uma vez por contorno!) - rodar várias
    # vezes no mesmo frame era o que causava o travamento da interface.
    confianca_minima = 0.3 if noite else 0.4
    deteccoes = classificar_frame(frame, confianca_minima=confianca_minima)
    ultimas_deteccoes = deteccoes

    for nome, confianca, x, y, largura, altura in deteccoes:
        cv2.rectangle(frame, (x, y), (x + largura, y + altura), (0, 255, 0), 2)
        cv2.putText(frame, f"{nome} ({confianca:.0%})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    alertas = rastreador.atualizar(deteccoes)
    for i, alerta in enumerate(alertas):
        cv2.putText(frame, f"ALERTA: {alerta}", (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return alertas, threshold
