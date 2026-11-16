"""
Gera evidencias REAIS para os slides do SafeScreen.AI e mede metricas simples.

Coloque este arquivo na mesma pasta do main.py e rode:
    python gerar_evidencias.py                  (usa a camera 0)
    python gerar_evidencias.py --forcar-noite   (aplica equalizeHist + FFT mesmo com luz)

Roteiro (o script avisa no terminal):
  1) FASE PARADA: fique FORA da imagem / nao mexa em nada por ~5 s.
     Qualquer contorno detectado aqui e um falso positivo (nao ha movimento real).
  2) FASE MOVIMENTO: entre em cena e ande por ~4 s.
Saidas na pasta "evidencias/": painel_segmentacao_real.png, painel_fft_real.png
e o resumo de metricas no terminal (copie para os slides).
"""
import argparse
import os
import time

import cv2
import numpy as np

from analise_fft import filtrar_ruido_por_frequencia

# Mesmos parametros do detectmov.py
LIMIAR_DIFERENCA = 25
TAMANHO_MINIMO_CONTORNO = 30
AREA_MINIMA_CONTORNO = 500
RAIO_CORTE_FFT = 40
PASTA = "evidencias"


def preparar(frame, noite, usar_fft):
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if noite:
        cinza = cv2.equalizeHist(cinza)
        if usar_fft:
            cinza, _ = filtrar_ruido_por_frequencia(cinza, RAIO_CORTE_FFT)
    return cv2.GaussianBlur(cinza, (21, 21), 0)


def segmentar(f_ant, f_atual, noite, usar_fft):
    a, b = preparar(f_ant, noite, usar_fft), preparar(f_atual, noite, usar_fft)
    dif = cv2.absdiff(a, b)
    _, th = cv2.threshold(dif, LIMIAR_DIFERENCA, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, None, iterations=2)
    contornos, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reais = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if w >= TAMANHO_MINIMO_CONTORNO and h >= TAMANHO_MINIMO_CONTORNO and cv2.contourArea(c) > AREA_MINIMA_CONTORNO:
            reais.append(c)
    return dif, th, reais


def rotulo(img, texto):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (18, 22, 28), -1)
    cv2.putText(img, texto, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 245, 242), 2)
    return img


def bgr(x):
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if x.ndim == 2 else x


def capturar(cam, n):
    frames = []
    while len(frames) < n:
        ok, f = cam.read()
        if ok:
            frames.append(f)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", default="0", help="indice da camera ou caminho de um video")
    ap.add_argument("--forcar-noite", action="store_true")
    args = ap.parse_args()

    os.makedirs(PASTA, exist_ok=True)
    fonte = int(args.fonte) if args.fonte.isdigit() else args.fonte
    cam = cv2.VideoCapture(fonte)
    if not cam.isOpened():
        raise SystemExit("Nao foi possivel abrir a camera/video.")
    capturar(cam, 20)  # aquecimento (auto-exposicao)

    print("FASE 1: FIQUE FORA DA IMAGEM e nao mexa em nada (5 segundos)...")
    parada = capturar(cam, 100)
    print("FASE 2: ENTRE EM CENA E ANDE (4 segundos)...")
    movimento = capturar(cam, 80)
    cam.release()

    brilho = float(np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in parada]))
    noite = args.forcar_noite or brilho < 80
    print(f"\nBrilho medio da cena: {brilho:.1f} (modo noturno = {'SIM' if noite else 'NAO'})")

    # ---- Metrica 1: falsos positivos em cena parada, SEM x COM FFT (mesmos frames)
    print("\n== Falsos positivos em cena parada (100 frames, nenhum movimento real) ==")
    for usar_fft in (False, True):
        fp = sum(1 for i in range(1, len(parada)) if segmentar(parada[i - 1], parada[i], noite, usar_fft)[2])
        print(f"  {'COM' if usar_fft else 'SEM'} FFT: {fp} de {len(parada) - 1} pares de frames com contorno detectado")

    # ---- Metrica 2: tempo de processamento
    print("\n== Tempo medio de processamento por frame ==")
    cinza = cv2.cvtColor(parada[0], cv2.COLOR_BGR2GRAY)
    t = time.perf_counter()
    for _ in range(30):
        filtrar_ruido_por_frequencia(cinza, RAIO_CORTE_FFT)
    print(f"  Filtro FFT:  {(time.perf_counter() - t) / 30 * 1000:.1f} ms")
    t = time.perf_counter()
    for i in range(1, 31):  # no sistema real cada frame e preparado uma unica vez (o anterior fica em cache)
        atual = preparar(parada[i], noite, usar_fft=noite)
        _, th_t = cv2.threshold(cv2.absdiff(atual, atual), LIMIAR_DIFERENCA, 255, cv2.THRESH_BINARY)
        cv2.findContours(cv2.dilate(th_t, None, iterations=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Pre-processamento + segmentacao de 1 frame: {(time.perf_counter() - t) / 30 * 1000:.1f} ms")
    try:
        from ia_classificacao import classificar_frame
        classificar_frame(parada[0])  # aquecimento do modelo
        t = time.perf_counter()
        for i in range(10):
            classificar_frame(movimento[i])
        print(f"  YOLOv8n (1 inferencia): {(time.perf_counter() - t) / 10 * 1000:.1f} ms")
    except Exception as erro:
        print(f"  YOLOv8n: nao medido ({erro})")

    # ---- Painel de segmentacao (par de frames com mais movimento)
    melhor = max(range(1, len(movimento)), key=lambda i: int((segmentar(movimento[i - 1], movimento[i], noite, noite)[1] > 0).sum()))
    ant, atual = movimento[melhor - 1], movimento[melhor]
    dif, th, reais = segmentar(ant, atual, noite, noite)
    final = atual.copy()
    for c in reais:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(final, (x, y), (x + w, y + h), (0, 255, 0), 3)
    dif_vis = cv2.normalize(dif, None, 0, 255, cv2.NORM_MINMAX)
    seg = np.hstack([rotulo(atual, "1. Frame original"), rotulo(bgr(dif_vis), "2. absdiff"),
                     rotulo(bgr(th), "3. threshold + dilate"), rotulo(final, "4. Contornos")])
    cv2.imwrite(os.path.join(PASTA, "painel_segmentacao_real.png"), seg)

    # ---- Painel FFT (cinza, espectro, apos mascara, apos IFFT)
    cz = cv2.cvtColor(atual, cv2.COLOR_BGR2GRAY)
    if noite:
        cz = cv2.equalizeHist(cz)
    filtrada, espectro = filtrar_ruido_por_frequencia(cz, RAIO_CORTE_FFT)
    h, w = cz.shape
    yy, xx = np.ogrid[:h, :w]
    mascara = np.sqrt((xx - w // 2) ** 2 + (yy - h // 2) ** 2) <= RAIO_CORTE_FFT
    esp_c = bgr(espectro.copy())
    cv2.circle(esp_c, (w // 2, h // 2), RAIO_CORTE_FFT, (0, 200, 255), 2)
    fft_p = np.hstack([rotulo(bgr(cz), "1. Cinza"), rotulo(esp_c, "2. Espectro |F| (log)"),
                       rotulo(bgr((espectro * mascara).astype(np.uint8)), "3. Apos mascara"),
                       rotulo(bgr(filtrada), "4. Apos IFFT")])
    cv2.imwrite(os.path.join(PASTA, "painel_fft_real.png"), fft_p)
    print(f"\nImagens salvas em '{PASTA}/'. Envie as 2 imagens e o resumo acima para atualizar os slides.")


if __name__ == "__main__":
    main()
