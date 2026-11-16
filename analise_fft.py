import numpy as np


def filtrar_ruido_por_frequencia(imagem_cinza, raio_corte=40):
    """
    Aplica um FILTRO PASSA-BAIXA no domínio da frequência para remover ruído
    de sensor (grão/flicker), que é especialmente forte em câmeras à noite.

    A ideia (FFT para processamento de sinais/imagens):
      1. A imagem é levada do domínio espacial (pixels) para o domínio da
         frequência com a Transformada Rápida de Fourier 2D (FFT2D).
      2. No domínio da frequência: ruído de sensor e detalhes finos/granulados
         aparecem como componentes de ALTA frequência; o formato geral dos
         objetos e do fundo aparece como componentes de BAIXA frequência.
      3. Construímos uma máscara circular que deixa passar só as frequências
         baixas (perto do centro do espectro) e corta as altas.
      4. Aplicamos a máscara e voltamos para o domínio espacial com a
         Transformada Inversa (IFFT2D) - o resultado é a imagem original,
         porém com o ruído de granulação atenuado.

    Retorna: (imagem_filtrada, espectro_visual)
      - imagem_filtrada: pronta para seguir no pipeline (pode substituir a
        entrada do cv2.GaussianBlur / cv2.absdiff).
      - espectro_visual: imagem do espectro de magnitude (em escala log, para
        ficar visível), útil para MOSTRAR NO SLIDE o antes/depois no domínio
        da frequência.
    """
    altura, largura = imagem_cinza.shape

    # 1. Domínio espacial -> domínio da frequência
    espectro = np.fft.fft2(imagem_cinza)
    espectro_centralizado = np.fft.fftshift(espectro)  # move a frequência 0 (DC) para o centro

    # 2. Máscara passa-baixa: um círculo de raio "raio_corte" no centro do espectro
    centro_y, centro_x = altura // 2, largura // 2
    y, x = np.ogrid[:altura, :largura]
    distancia_do_centro = np.sqrt((x - centro_x) ** 2 + (y - centro_y) ** 2)
    mascara = distancia_do_centro <= raio_corte

    # 3. Aplica a máscara (zera as frequências altas, fora do círculo)
    espectro_filtrado = espectro_centralizado * mascara

    # 4. Domínio da frequência -> domínio espacial de volta
    espectro_sem_shift = np.fft.ifftshift(espectro_filtrado)
    imagem_filtrada = np.fft.ifft2(espectro_sem_shift)
    imagem_filtrada = np.abs(imagem_filtrada)  # a IFFT devolve números complexos; pegamos a magnitude
    imagem_filtrada = np.clip(imagem_filtrada, 0, 255).astype(np.uint8)

    # Espectro de magnitude em escala log, normalizado para 0-255 - só para visualização/slide
    magnitude = np.abs(espectro_centralizado)
    espectro_visual = np.log1p(magnitude)
    espectro_visual = (espectro_visual / espectro_visual.max() * 255).astype(np.uint8)

    return imagem_filtrada, espectro_visual
