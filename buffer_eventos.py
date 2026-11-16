import cv2
import os
import queue
import threading
import time
from datetime import datetime

from collections import deque

from registro_eventos import registrar_evento
from notificacao import enviar_em_segundo_plano, notificar_evento
from ia_generativa import descrever_evento

PASTA_EVENTOS = "eventos"
SEGUNDOS_BUFFER_ANTES = 15    # quanto tempo "antes" do alerta fica guardado em memória
SEGUNDOS_GRAVACAO_DEPOIS = 5  # continua gravando por mais esse tempo após o último alerta
ALERTA_GRAVACAO_MANUAL = "gravacao solicitada pelo chat"


def _amostrar(frames, quantidade):
    """Escolhe `quantidade` frames espalhados igualmente pela lista."""
    if not frames:
        return []
    passo = len(frames) / quantidade
    return [frames[min(int(passo * (i + 0.5)), len(frames) - 1)] for i in range(quantidade)]


class GravadorDeEventos:
    """
    Mantém um buffer circular com os últimos N segundos de vídeo em memória.
    Quando um evento (alerta) acontece, salva esse buffer + os frames seguintes
    em um .mp4. Quando o evento termina, a IA GENERATIVA escreve um relato do
    evento a partir da imagem do momento do alerta; depois registra no CSV e
    notifica por WhatsApp.

    Também aceita gravações pedidas pelo chat (gravar_manual), mesmo sem alerta.
    """

    def __init__(self, fps_estimado=20):
        self.fps = max(int(fps_estimado), 1)
        self.buffer = deque(maxlen=self.fps * SEGUNDOS_BUFFER_ANTES)
        self.gravando = False
        self.writer = None
        self.caminho_video_atual = None
        self.alertas_do_evento = set()
        self.instante_ultimo_alerta = None
        self.frame_do_alerta = None      # imagem enviada para a IA descrever
        self._tem_frame_de_alerta = False

        # Gravação pedida pelo chat
        self._pedido_manual = None       # (segundos, avisar_whatsapp)
        self._fim_manual = 0.0
        self._avisar_whatsapp_manual = False

        # Mensagens para mostrar no chat da interface (eventos concluídos etc.)
        self.avisos = queue.Queue()
        os.makedirs(PASTA_EVENTOS, exist_ok=True)

    def gravar_manual(self, segundos, avisar_whatsapp=False):
        """Pedido do chat: grava agora por `segundos` (pode ser chamado de outra thread)."""
        self._pedido_manual = (segundos, avisar_whatsapp)

    def atualizar(self, frame, alertas):
        """Chame isso a cada frame. alertas: lista de strings (vazia se nenhum)."""
        self.buffer.append(frame.copy())
        agora = time.time()

        if self._pedido_manual is not None:
            segundos, avisar = self._pedido_manual
            self._pedido_manual = None
            self._fim_manual = max(self._fim_manual, agora + segundos)
            self._avisar_whatsapp_manual = self._avisar_whatsapp_manual or avisar
            self.alertas_do_evento.add(ALERTA_GRAVACAO_MANUAL)
            if self.frame_do_alerta is None:
                self.frame_do_alerta = frame.copy()
            if not self.gravando:
                self._iniciar_gravacao(frame)

        if alertas:
            self.instante_ultimo_alerta = agora
            self.alertas_do_evento.update(alertas)
            if not self._tem_frame_de_alerta:
                # guarda a imagem do primeiro alerta real (é a que a IA vai descrever)
                self.frame_do_alerta = frame.copy()
                self._tem_frame_de_alerta = True
            if not self.gravando:
                self._iniciar_gravacao(frame)

        if self.gravando:
            self.writer.write(frame)
            fim_por_alerta = (self.instante_ultimo_alerta or 0) + SEGUNDOS_GRAVACAO_DEPOIS
            if agora > max(fim_por_alerta, self._fim_manual):
                self._encerrar_gravacao()

    def _iniciar_gravacao(self, frame_atual):
        altura, largura = frame_atual.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.caminho_video_atual = os.path.join(PASTA_EVENTOS, f"evento_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.caminho_video_atual, fourcc, self.fps, (largura, altura))

        for frame_anterior in self.buffer:
            self.writer.write(frame_anterior)

        self.gravando = True

    def _encerrar_gravacao(self):
        self.writer.release()
        self.writer = None
        self.gravando = False

        evento = {
            "video": self.caminho_video_atual,
            "alertas": sorted(self.alertas_do_evento),
            # imagem do alerta + 3 imagens espalhadas pelos últimos segundos: a IA
            # vê o evento inteiro, não só um instante (ex: a arma aparece depois)
            "frames": [self.frame_do_alerta] + _amostrar(list(self.buffer), 3),
            "manual": ALERTA_GRAVACAO_MANUAL in self.alertas_do_evento,
            "avisar_whatsapp": self._avisar_whatsapp_manual,
        }
        print("[SafeScreen.AI] Gravacao do evento encerrada.")

        self.alertas_do_evento = set()
        self.caminho_video_atual = None
        self.frame_do_alerta = None
        self._tem_frame_de_alerta = False
        self.instante_ultimo_alerta = None
        self._fim_manual = 0.0
        self._avisar_whatsapp_manual = False

        # A IA demora alguns segundos: roda em outra thread para não travar a captura
        threading.Thread(target=self._finalizar_evento, args=(evento,), daemon=True).start()

    def _finalizar_evento(self, evento):
        relato = descrever_evento(evento["frames"], evento["alertas"])
        registrar_evento(evento["video"], evento["alertas"], relato)

        apenas_manual = evento["alertas"] == [ALERTA_GRAVACAO_MANUAL]
        if not apenas_manual:
            notificar_evento(evento["video"], evento["alertas"], relato)
        elif evento["avisar_whatsapp"]:
            enviar_em_segundo_plano(
                f"SafeScreen.AI - Gravação pedida pelo chat concluída.\n\n"
                f"Análise da IA:\n{relato}", evento["video"])

        titulo = "Gravação concluída" if apenas_manual else "Evento registrado"
        self.avisos.put(f"{titulo} ({', '.join(evento['alertas'])}).\n"
                        f"{relato}\nVídeo: {evento['video']}")

    def finalizar(self):
        """Chame ao encerrar o programa, caso ainda esteja gravando algo."""
        if self.gravando:
            self._encerrar_gravacao()
