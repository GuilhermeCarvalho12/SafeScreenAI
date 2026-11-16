import queue
import threading
import time
from collections import Counter
from datetime import datetime

import cv2

import detectmov
from detectmov import detectar_movimento, resetar_estado
from buffer_eventos import GravadorDeEventos
from registro_eventos import registrar_atividade


class CapturaSafeScreen:
    """
    Roda a câmera, a detecção de movimento e a IA em uma thread separada da interface.

    Por quê: rodar a IA é lento (pode levar 100-300ms por frame). Se isso acontecesse
    na mesma thread do Tkinter, a janela travaria toda vez que algo se movesse.
    Aqui a thread de captura nunca para - ela só entrega o resultado mais recente
    pronto numa fila, e a interface pega o que estiver disponível sem esperar.
    """

    def __init__(self, indice_camera=0):
        self.camera = cv2.VideoCapture(indice_camera)
        if not self.camera.isOpened():
            raise RuntimeError("Nao foi possivel abrir a camera.")

        fps = self.camera.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 20

        self.gravador = GravadorDeEventos(fps_estimado=self.fps)

        # maxsize=1: a fila guarda só o resultado mais recente. Se a interface
        # estiver mais lenta que a câmera, frames antigos são descartados em vez
        # de se acumular (o que faria o vídeo exibido atrasar cada vez mais).
        self.saida = queue.Queue(maxsize=1)

        self.modo_seguranca = True
        self.ultimo_frame = None   # frame "limpo" mais recente (usado pelo chatbot)

        # Resumo de atividade do minuto atual (gravado em eventos/atividade.csv)
        self._minuto_atual = None
        self._frames_com_deteccao = 0
        self._max_objetos = Counter()

        self._rodando = False
        self._thread = None

    def iniciar(self):
        self._rodando = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def parar(self):
        self._rodando = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._salvar_atividade()
        self.gravador.finalizar()
        self.camera.release()

    def definir_seguranca(self, ativa):
        """Liga/desliga o modo segurança (usado pelo botão e pelo chatbot)."""
        if ativa != self.modo_seguranca:
            resetar_estado()  # evita falso alerta de "movimento" ao reativar
        self.modo_seguranca = ativa

    def _loop(self):
        while self._rodando:
            sucesso, frame = self.camera.read()
            if not sucesso:
                time.sleep(0.1)
                continue

            self.ultimo_frame = frame.copy()  # antes do detectmov desenhar as caixas

            if self.modo_seguranca:
                alertas, threshold = detectar_movimento(frame)
            else:
                alertas = []
                threshold = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self._acumular_atividade(detectmov.ultimas_deteccoes if self.modo_seguranca else [])
            self.gravador.atualizar(frame, alertas)
            self._publicar({"frame": frame, "threshold": threshold, "alertas": alertas})

    def _acumular_atividade(self, deteccoes):
        minuto = datetime.now().strftime("%Y-%m-%d %H:%M")
        if minuto != self._minuto_atual:
            self._salvar_atividade()
            self._minuto_atual = minuto
        if deteccoes:
            self._frames_com_deteccao += 1
            contagem = Counter(nome for nome, *_ in deteccoes)
            for nome, qtd in contagem.items():
                self._max_objetos[nome] = max(self._max_objetos[nome], qtd)

    def _salvar_atividade(self):
        # Só grava minutos em que algo foi detectado, para o arquivo não crescer à toa
        if self._minuto_atual and self._frames_com_deteccao:
            registrar_atividade(self._minuto_atual, self._frames_com_deteccao,
                                dict(self._max_objetos))
        self._frames_com_deteccao = 0
        self._max_objetos = Counter()

    def _publicar(self, resultado):
        # Descarta o resultado anterior (se a interface ainda não pegou) para
        # garantir que a fila nunca acumule frames atrasados.
        if self.saida.full():
            try:
                self.saida.get_nowait()
            except queue.Empty:
                pass
        self.saida.put(resultado)
