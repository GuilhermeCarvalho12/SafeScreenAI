import os
import queue
import threading
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageDraw, ImageTk

from captura import CapturaSafeScreen
from chat_markdown import analisar_markdown
from ia_generativa import AssistenteSafeScreen, abrir_arquivo, chat_disponivel, descricao_motores
from registro_eventos import CAMINHO_REGISTRO, ler_eventos
from tema import (
    ALERTA, AVISO, BG, BORDA, PAINEL, PAINEL_2, TEXTO, TEXTO_SUAVE, VERDE, VERDE_CLARO,
    aplicar_tema, estilizar_tabela,
)

# Cores extras da interface (derivadas da paleta do tema)
VERDE_HOVER = "#37a07d"
ALERTA_HOVER = "#f06a6e"
BOLHA_USUARIO = "#26352f"     # balão das mensagens do usuário
FUNDO_CHAT = PAINEL
LINK = "#62b6e7"

FONTE_UI = "Segoe UI"
FONTE_UI_FORTE = "Segoe UI Semibold"
FONTE_MONO = "Consolas"

# Sugestões mostradas enquanto a conversa está vazia
SUGESTOES_CHAT = [
    "Como foi a última hora?",
    "O que está acontecendo agora?",
    "Grave 30 segundos e me mande no celular",
]
MAX_EVENTOS_TABELA = 60


def _criar_icone_camera(tamanho=64):
    """Desenha um icone de camera (sem arquivo .ico e sem depender de fonte de emoji)."""
    esc = 4  # desenha em alta resolucao e reduz, para as bordas ficarem suaves
    t = tamanho * esc
    img = Image.new("RGBA", (t, t), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    u = t / 64.0

    d.rounded_rectangle([20 * u, 12 * u, 44 * u, 24 * u], radius=4 * u, fill=VERDE_CLARO)
    d.rounded_rectangle([5 * u, 18 * u, 59 * u, 54 * u], radius=9 * u, fill=VERDE)
    d.ellipse([20 * u, 24 * u, 44 * u, 48 * u], fill=BG)
    d.ellipse([24 * u, 28 * u, 40 * u, 44 * u], fill=PAINEL_2)
    d.ellipse([29 * u, 33 * u, 35 * u, 39 * u], fill=VERDE_CLARO)
    d.ellipse([47 * u, 24 * u, 53 * u, 30 * u], fill="#f5f6f2")

    return img.resize((tamanho, tamanho), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Botão com efeito de hover (o tk.Button padrão não muda ao passar o mouse)
# ---------------------------------------------------------------------------
def criar_botao(pai, texto, comando, cor=VERDE, cor_hover=VERDE_HOVER, cor_texto="#ffffff",
                fonte=(FONTE_UI_FORTE, 10), padx=14, pady=7, **extra):
    botao = tk.Button(
        pai, text=texto, command=comando, bg=cor, fg=cor_texto, font=fonte,
        activebackground=cor_hover, activeforeground=cor_texto, relief="flat", bd=0,
        cursor="hand2", padx=padx, pady=pady, highlightthickness=0, **extra)
    botao.cor_base, botao.cor_hover = cor, cor_hover

    def entrar(_e):
        if str(botao["state"]) != "disabled":
            botao.config(bg=botao.cor_hover)

    def sair(_e):
        botao.config(bg=botao.cor_base)

    botao.bind("<Enter>", entrar)
    botao.bind("<Leave>", sair)
    return botao


def pintar_botao(botao, cor, cor_hover, cor_texto=None):
    botao.cor_base, botao.cor_hover = cor, cor_hover
    botao.config(bg=cor, activebackground=cor_hover)
    if cor_texto:
        botao.config(fg=cor_texto, activeforeground=cor_texto)


class InterfaceSafeScreen:
    """
    Só cuida da tela: pede o resultado mais recente ao CapturaSafeScreen (que roda
    em outra thread) e desenha. Nenhum processamento pesado acontece aqui - é isso
    que mantém a janela sempre responsiva.

    Layout (tudo redimensionável arrastando as divisórias):
        ┌───────────────────────────┬──────────────────┐
        │ câmera                    │                  │
        │ [status] [modos] [segur.] │  assistente IA   │
        ├───────────────────────────┤  (recolhível)    │
        │ eventos recentes          │                  │
        └───────────────────────────┴──────────────────┘
    """

    def __init__(self, janela):
        self.janela = janela
        self.janela.title("SafeScreen.AI")
        self.janela.minsize(980, 620)
        self._definir_icone()
        self.janela.protocol("WM_DELETE_WINDOW", self.ao_fechar)
        try:
            self.janela.state("zoomed")  # abre maximizada no Windows
        except tk.TclError:
            self.janela.geometry("1400x860")

        self.modo_visualizacao = "normal"  # ou "threshold"
        self._tamanho_video = (800, 600)

        self.captura = CapturaSafeScreen()
        self.captura.iniciar()

        # Chatbot (IA generativa). As respostas chegam por esta fila, vindas da
        # thread que conversa com a IA - a interface nunca espera a IA.
        self.assistente = AssistenteSafeScreen(self.captura)
        self.fila_chat = queue.Queue()
        self._pensando = False
        self._links = {}           # tag do link -> caminho do vídeo
        self._bolhas = []          # balões do usuário (para ajustar a largura)
        self._chat_visivel = True
        self._pos_divisoria_chat = None
        self._registro_mtime = None
        self._eventos_tabela = {}  # id da linha -> evento

        self._configurar_estilos()
        self._montar_layout()
        self._mensagem_inicial_chat()
        self._atualizar_lista_eventos()
        self.janela.after(200, self._posicionar_divisorias)
        self._loop_video()
        self._loop_registro()

    # ------------------------------------------------------------------
    # Estrutura
    # ------------------------------------------------------------------
    def _definir_icone(self):
        try:
            self._icone_pil = _criar_icone_camera(64)
            self._icone_tk = ImageTk.PhotoImage(self._icone_pil)
            self.janela.iconphoto(True, self._icone_tk)
        except Exception:
            self._icone_pil = None
            self._icone_tk = None

    def _configurar_estilos(self):
        estilo = ttk.Style(self.janela)
        try:
            # divisórias finas e discretas entre os painéis
            estilo.configure("Sash", sashthickness=8, gripcount=0, background=BG,
                             lightcolor=BG, darkcolor=BG, bordercolor=BG)
        except tk.TclError:
            pass
        estilo.configure("TPanedwindow", background=BG)

    def _montar_layout(self):
        self._montar_cabecalho()

        # divisória horizontal: lado esquerdo (câmera + eventos) | chat
        self.paineis = ttk.PanedWindow(self.janela, orient="horizontal")
        self.paineis.pack(fill="both", expand=True, padx=16, pady=(4, 14))

        # divisória vertical: câmera em cima, eventos embaixo
        self.lado_esquerdo = ttk.PanedWindow(self.paineis, orient="vertical")
        self.paineis.add(self.lado_esquerdo, weight=3)

        area_video = tk.Frame(self.lado_esquerdo, bg=BG)
        self.lado_esquerdo.add(area_video, weight=3)
        self._montar_video(area_video)

        area_eventos = tk.Frame(self.lado_esquerdo, bg=BG)
        self.lado_esquerdo.add(area_eventos, weight=2)
        self._montar_eventos(area_eventos)

        self.painel_chat = tk.Frame(self.paineis, bg=BG)
        self.paineis.add(self.painel_chat, weight=2)
        self._montar_chat(self.painel_chat)

    def _montar_cabecalho(self):
        cabecalho = tk.Frame(self.janela, bg=BG)
        cabecalho.pack(fill="x", padx=16, pady=(12, 6))

        if self._icone_pil is not None:
            try:
                self._icone_cab = ImageTk.PhotoImage(self._icone_pil.resize((34, 34), Image.LANCZOS))
                tk.Label(cabecalho, image=self._icone_cab, bg=BG).pack(side="left", padx=(0, 10))
            except Exception:
                pass

        titulo = tk.Frame(cabecalho, bg=BG)
        titulo.pack(side="left")
        linha_titulo = tk.Frame(titulo, bg=BG)
        linha_titulo.pack(anchor="w")
        tk.Label(linha_titulo, text="SafeScreen", font=(FONTE_MONO, 20, "bold"), fg=TEXTO, bg=BG).pack(side="left")
        tk.Label(linha_titulo, text=".AI", font=(FONTE_MONO, 20, "bold"), fg=VERDE, bg=BG).pack(side="left")
        tk.Label(titulo, text="MONITORAMENTO INTELIGENTE", font=(FONTE_MONO, 9),
                 fg=TEXTO_SUAVE, bg=BG).pack(anchor="w")

        self.botao_chat = criar_botao(
            cabecalho, "Ocultar assistente  »", self._alternar_chat,
            cor=PAINEL, cor_hover=PAINEL_2, cor_texto=TEXTO, fonte=(FONTE_UI, 10))
        self.botao_chat.pack(side="right")

    # ------------------------------------------------------------------
    # Câmera, status e botões
    # ------------------------------------------------------------------
    def _montar_video(self, pai):
        # Barra de controles (empacotada ANTES do vídeo para nunca ser espremida)
        barra = tk.Frame(pai, bg=BG)
        barra.pack(side="bottom", fill="x", pady=(10, 8))

        self.label_status = tk.Label(
            barra, text="●  Iniciando...", font=(FONTE_UI_FORTE, 11), fg=TEXTO_SUAVE,
            bg=PAINEL, padx=14, pady=7, anchor="w")
        self.label_status.pack(side="left", fill="x", expand=True, padx=(0, 10))

        # Seletor de modo de visualização (o ativo fica destacado)
        seletor = tk.Frame(barra, bg=BORDA, padx=1, pady=1)
        seletor.pack(side="left", padx=(0, 10))
        self.botao_normal = criar_botao(seletor, "Normal", self._modo_normal, padx=16)
        self.botao_normal.pack(side="left")
        self.botao_threshold = criar_botao(seletor, "Threshold", self._modo_threshold, padx=16)
        self.botao_threshold.pack(side="left")
        self._atualizar_seletor_modo()

        self.botao_seguranca = criar_botao(barra, "", self._alternar_seguranca, padx=16)
        self.botao_seguranca.pack(side="left")
        self._atualizar_botao_seguranca(forcar=True)

        moldura = tk.Frame(pai, bg="#000000", highlightthickness=1, highlightbackground=BORDA)
        moldura.pack(fill="both", expand=True)
        moldura.pack_propagate(False)  # o tamanho vem da divisória, não da imagem
        self.label_video = tk.Label(moldura, bg="#000000", text="Abrindo câmera...",
                                    fg=TEXTO_SUAVE, font=(FONTE_UI, 11))
        self.label_video.pack(fill="both", expand=True)
        moldura.bind("<Configure>", self._ao_redimensionar_video)

    def _ao_redimensionar_video(self, evento):
        self._tamanho_video = (max(evento.width - 2, 1), max(evento.height - 2, 1))

    def _encaixar_imagem(self, imagem):
        """Redimensiona mantendo a proporção para caber no espaço do vídeo."""
        largura_max, altura_max = self._tamanho_video
        altura, largura = imagem.shape[:2]
        escala = min(largura_max / largura, altura_max / altura)
        nova = (max(int(largura * escala), 1), max(int(altura * escala), 1))
        interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
        return cv2.resize(imagem, nova, interpolation=interp)

    def _modo_normal(self):
        self.modo_visualizacao = "normal"
        self._atualizar_seletor_modo()

    def _modo_threshold(self):
        self.modo_visualizacao = "threshold"
        self._atualizar_seletor_modo()

    def _atualizar_seletor_modo(self):
        for botao, modo in ((self.botao_normal, "normal"), (self.botao_threshold, "threshold")):
            if self.modo_visualizacao == modo:
                pintar_botao(botao, VERDE, VERDE_HOVER, "#ffffff")
            else:
                pintar_botao(botao, PAINEL, PAINEL_2, TEXTO_SUAVE)

    def _alternar_seguranca(self):
        self.captura.definir_seguranca(not self.captura.modo_seguranca)
        self._atualizar_botao_seguranca()

    def _atualizar_botao_seguranca(self, forcar=False):
        # Também é chamado no loop, porque o chatbot pode ligar/desligar a segurança
        ativa = self.captura.modo_seguranca
        texto = "●  Segurança ativa" if ativa else "○  Segurança desligada"
        if forcar or self.botao_seguranca["text"] != texto:
            self.botao_seguranca.config(text=texto)
            if ativa:
                pintar_botao(self.botao_seguranca, VERDE, VERDE_HOVER)
            else:
                pintar_botao(self.botao_seguranca, ALERTA, ALERTA_HOVER)

    # ------------------------------------------------------------------
    # Tabela de eventos
    # ------------------------------------------------------------------
    def _montar_eventos(self, pai):
        topo = tk.Frame(pai, bg=BG)
        topo.pack(fill="x", pady=(10, 6))
        tk.Label(topo, text="EVENTOS RECENTES", font=(FONTE_MONO, 11, "bold"),
                 fg=VERDE_CLARO, bg=BG).pack(side="left")
        tk.Label(topo, text="clique para ver o relato  •  duplo clique abre o vídeo",
                 font=(FONTE_UI, 9), fg=TEXTO_SUAVE, bg=BG).pack(side="right")

        # Painel de detalhes do evento selecionado (embaixo da tabela)
        detalhe = tk.Frame(pai, bg=PAINEL, highlightthickness=1, highlightbackground=BORDA)
        detalhe.pack(side="bottom", fill="x", pady=(8, 0))
        self.botao_abrir_video = criar_botao(detalhe, "Abrir vídeo", self._abrir_video_selecionado,
                                             padx=12, pady=5)
        self.botao_abrir_video.pack(side="right", padx=10, pady=10)
        self.botao_abrir_video.config(state="disabled")
        self.label_detalhe = tk.Label(
            detalhe, text="Selecione um evento para ver o relato completo da IA.",
            font=(FONTE_UI, 10), fg=TEXTO_SUAVE, bg=PAINEL, justify="left", anchor="w",
            padx=12, pady=10, wraplength=600)
        self.label_detalhe.pack(side="left", fill="x", expand=True)
        detalhe.bind("<Configure>", lambda e: self.label_detalhe.config(
            wraplength=max(e.width - 160, 200)))

        area_tabela = tk.Frame(pai, bg=BORDA, padx=1, pady=1)
        area_tabela.pack(fill="both", expand=True)

        self.lista_eventos = ttk.Treeview(area_tabela, columns=("data", "tipo", "relato"),
                                          show="headings", selectmode="browse")
        self.lista_eventos.heading("data", text="Data/Hora", anchor="w")
        self.lista_eventos.heading("tipo", text="Evento", anchor="w")
        self.lista_eventos.heading("relato", text="Relato da IA", anchor="w")
        self.lista_eventos.column("data", width=150, minwidth=120, stretch=False, anchor="w")
        self.lista_eventos.column("tipo", width=240, minwidth=120, stretch=False, anchor="w")
        self.lista_eventos.column("relato", width=400, minwidth=150, stretch=True, anchor="w")
        estilizar_tabela(self.lista_eventos)

        barra = ttk.Scrollbar(area_tabela, orient="vertical", command=self.lista_eventos.yview)
        self.lista_eventos.configure(yscrollcommand=barra.set)
        self.lista_eventos.pack(side="left", fill="both", expand=True)
        barra.pack(side="right", fill="y")

        self.lista_eventos.bind("<<TreeviewSelect>>", self._ao_selecionar_evento)
        self.lista_eventos.bind("<Double-1>", lambda _e: self._abrir_video_selecionado())

    def _atualizar_lista_eventos(self):
        selecionado = self.lista_eventos.selection()
        selecionado = self._eventos_tabela.get(selecionado[0], {}).get("data_hora") if selecionado else None

        for item in self.lista_eventos.get_children():
            self.lista_eventos.delete(item)
        self._eventos_tabela = {}

        eventos = ler_eventos()[-MAX_EVENTOS_TABELA:]
        for i, evento in enumerate(reversed(eventos)):  # mais recente primeiro
            tags = ["par" if i % 2 == 0 else "impar"]
            if "gravacao solicitada" in evento["alertas"]:
                tags.append("aviso")
            relato = " ".join(evento["relato_ia"].split())
            iid = self.lista_eventos.insert(
                "", "end", values=(evento["data_hora"], evento["alertas"], relato), tags=tags)
            self._eventos_tabela[iid] = evento
            if evento["data_hora"] == selecionado:
                self.lista_eventos.selection_set(iid)

    def _evento_selecionado(self):
        selecao = self.lista_eventos.selection()
        return self._eventos_tabela.get(selecao[0]) if selecao else None

    def _ao_selecionar_evento(self, _e=None):
        evento = self._evento_selecionado()
        if not evento:
            return
        relato = evento["relato_ia"].strip() or "(sem relato da IA para este evento)"
        self.label_detalhe.config(
            text=f"{evento['data_hora']}   •   {evento['alertas']}\n\n{relato}", fg=TEXTO)
        tem_video = bool(evento.get("video")) and os.path.exists(evento["video"])
        self.botao_abrir_video.config(state="normal" if tem_video else "disabled")

    def _abrir_video_selecionado(self):
        evento = self._evento_selecionado()
        if evento and evento.get("video") and os.path.exists(evento["video"]):
            abrir_arquivo(evento["video"])

    def _loop_registro(self):
        # O relato da IA chega alguns segundos depois do alerta: recarrega a
        # tabela só quando o CSV muda (verifica a cada 3 s, custo quase zero)
        try:
            mtime = os.path.getmtime(CAMINHO_REGISTRO)
        except OSError:
            mtime = None
        if mtime != self._registro_mtime:
            self._registro_mtime = mtime
            self._atualizar_lista_eventos()
        self.janela.after(3000, self._loop_registro)

    # ------------------------------------------------------------------
    # Chat com a IA generativa
    # ------------------------------------------------------------------
    def _montar_chat(self, pai):
        topo = tk.Frame(pai, bg=BG)
        topo.pack(fill="x", padx=(10, 0), pady=(0, 6))
        tk.Label(topo, text="ASSISTENTE IA", font=(FONTE_MONO, 11, "bold"),
                 fg=VERDE_CLARO, bg=BG).pack(side="left")
        self.label_motores = tk.Label(topo, text="", font=(FONTE_UI, 9), fg=TEXTO_SUAVE, bg=BG)
        self.label_motores.pack(side="left", padx=(10, 0))
        criar_botao(topo, "Nova conversa", self._nova_conversa, cor=PAINEL, cor_hover=PAINEL_2,
                    cor_texto=TEXTO, fonte=(FONTE_UI, 9), padx=10, pady=4).pack(side="right")

        # --- caixa de digitação (embaixo) ---
        self.caixa_entrada = tk.Frame(pai, bg=FUNDO_CHAT, highlightthickness=1,
                                      highlightbackground=BORDA)
        self.caixa_entrada.pack(side="bottom", fill="x", padx=(10, 0), pady=(8, 0))
        self.botao_enviar = criar_botao(self.caixa_entrada, "Enviar", self._enviar_chat,
                                        padx=16, pady=6)
        self.botao_enviar.pack(side="right", anchor="s", padx=8, pady=8)
        self.chat_entrada = tk.Text(
            self.caixa_entrada, height=2, wrap="word", bg=FUNDO_CHAT, fg=TEXTO,
            insertbackground=TEXTO, relief="flat", font=(FONTE_UI, 11), padx=12, pady=10,
            highlightthickness=0, bd=0)
        self.chat_entrada.pack(side="left", fill="both", expand=True)
        self.placeholder = tk.Label(
            self.caixa_entrada, text="Pergunte ao SafeScreen...   (Enter envia, Shift+Enter pula linha)",
            font=(FONTE_UI, 10), fg=TEXTO_SUAVE, bg=FUNDO_CHAT, cursor="xterm")
        self.placeholder.place(x=14, y=11)
        self.placeholder.bind("<Button-1>", lambda _e: self.chat_entrada.focus_set())
        self.chat_entrada.bind("<Return>", self._ao_apertar_enter)
        for evento in ("<KeyRelease>", "<FocusIn>", "<FocusOut>"):
            self.chat_entrada.bind(evento, self._atualizar_placeholder, add="+")

        # --- sugestões (aparecem só com a conversa vazia) ---
        self.frame_sugestoes = tk.Frame(pai, bg=BG)
        for texto in SUGESTOES_CHAT:
            criar_botao(self.frame_sugestoes, texto, lambda t=texto: self._enviar_chat(t),
                        cor=PAINEL, cor_hover=PAINEL_2, cor_texto=TEXTO, fonte=(FONTE_UI, 10),
                        padx=12, pady=6, anchor="w").pack(fill="x", pady=2)

        # --- histórico da conversa ---
        area = tk.Frame(pai, bg=BORDA, padx=1, pady=1)
        area.pack(fill="both", expand=True, padx=(10, 0))
        self.chat_texto = tk.Text(
            area, wrap="word", state="disabled", bg=FUNDO_CHAT, fg=TEXTO, relief="flat",
            padx=20, pady=16, font=(FONTE_UI, 11), cursor="arrow", highlightthickness=0,
            bd=0, spacing1=1, spacing3=1)
        barra = ttk.Scrollbar(area, orient="vertical", command=self.chat_texto.yview)
        self.chat_texto.configure(yscrollcommand=barra.set)
        barra.pack(side="right", fill="y")
        self.chat_texto.pack(side="left", fill="both", expand=True)
        self.chat_texto.bind("<Configure>", self._ajustar_bolhas)
        self._configurar_tags_chat()

    def _configurar_tags_chat(self):
        t = self.chat_texto
        # (as tags criadas depois têm prioridade: títulos vêm depois de "bold")
        t.tag_configure("p", spacing3=3)
        t.tag_configure("gap", spacing1=2, font=(FONTE_UI, 5))
        t.tag_configure("bold", font=(FONTE_UI_FORTE, 11))
        t.tag_configure("italic", font=(FONTE_UI, 11, "italic"))
        t.tag_configure("code", font=(FONTE_MONO, 10), background=PAINEL_2, foreground=VERDE_CLARO)
        t.tag_configure("codeblock", font=(FONTE_MONO, 10), background="#151a20",
                        lmargin1=12, lmargin2=12, rmargin=12)
        t.tag_configure("bullet0", lmargin1=8, lmargin2=28, spacing3=3)
        t.tag_configure("bullet1", lmargin1=30, lmargin2=50, spacing3=2)
        t.tag_configure("bullet_mark", foreground=VERDE_CLARO, font=(FONTE_UI_FORTE, 11))
        t.tag_configure("quote", lmargin1=14, lmargin2=14, foreground=TEXTO_SUAVE)
        t.tag_configure("table", font=(FONTE_MONO, 10), wrap="none", lmargin1=4, lmargin2=4)
        t.tag_configure("table_head", font=(FONTE_MONO, 10, "bold"), foreground=VERDE_CLARO)
        t.tag_configure("hr", foreground=BORDA)
        t.tag_configure("h3", font=(FONTE_UI_FORTE, 12), spacing1=8, spacing3=4)
        t.tag_configure("h2", font=(FONTE_UI_FORTE, 13), spacing1=10, spacing3=4)
        t.tag_configure("h1", font=(FONTE_UI_FORTE, 14), spacing1=10, spacing3=6)
        t.tag_configure("nome_ia", font=(FONTE_UI_FORTE, 9), foreground=VERDE_CLARO, spacing1=16,
                        spacing3=4)
        t.tag_configure("sys_titulo", font=(FONTE_UI_FORTE, 9), foreground=AVISO, spacing1=16,
                        spacing3=3)
        t.tag_configure("sys", font=(FONTE_UI, 10), foreground=TEXTO_SUAVE)
        t.tag_configure("direita", justify="right", spacing1=0, spacing3=2)
        t.tag_configure("nome_voce", font=(FONTE_UI_FORTE, 9), foreground=TEXTO_SUAVE,
                        justify="right", spacing1=16, spacing3=4)
        t.tag_configure("pensando", font=(FONTE_UI, 10, "italic"), foreground=TEXTO_SUAVE,
                        spacing1=16)
        t.tag_configure("link", foreground=LINK, underline=True)
        t.tag_raise("link")
        t.tag_bind("link", "<Enter>", lambda _e: t.config(cursor="hand2"))
        t.tag_bind("link", "<Leave>", lambda _e: t.config(cursor="arrow"))
        t.tag_bind("link", "<Button-1>", self._clicar_link)

    # ---- escrita no chat ----
    def _editar_chat(self, funcao):
        self.chat_texto.configure(state="normal")
        try:
            funcao()
        finally:
            self.chat_texto.configure(state="disabled")
            self.chat_texto.see("end")

    def _inserir_markdown(self, texto, tags_base=()):
        for trecho, tags, video in analisar_markdown(texto, tags_base):
            if video:
                tag_link = f"video_{len(self._links)}"
                self._links[tag_link] = video
                tags = tags + (tag_link,)
            self.chat_texto.insert("end", trecho, tags)
        self.chat_texto.insert("end", "\n")

    def _mensagem_ia(self, texto):
        def escrever():
            self.chat_texto.insert("end", "SafeScreen.AI\n", "nome_ia")
            self._inserir_markdown(texto)
        self._editar_chat(escrever)

    def _mensagem_sistema(self, texto):
        """Avisos do sistema (evento registrado, gravação concluída...)."""
        linhas = texto.strip().split("\n")
        titulo, corpo = linhas[0], "\n".join(linhas[1:])

        def escrever():
            self.chat_texto.insert("end", "●  " + titulo + "\n", "sys_titulo")
            if corpo.strip():
                self._inserir_markdown(corpo, ("sys",))
        self._editar_chat(escrever)

    def _mensagem_usuario(self, texto):
        bolha = tk.Label(self.chat_texto, text=texto, bg=BOLHA_USUARIO, fg=TEXTO,
                         font=(FONTE_UI, 11), justify="left", anchor="w", padx=14, pady=9,
                         wraplength=self._largura_bolha())
        bolha.bind("<MouseWheel>", lambda e: self.chat_texto.yview_scroll(int(-e.delta / 120), "units"))
        self._bolhas.append(bolha)

        def escrever():
            self.chat_texto.insert("end", "Você\n", "nome_voce")
            inicio = self.chat_texto.index("end-1c")
            self.chat_texto.window_create("end", window=bolha)
            self.chat_texto.insert("end", "\n")
            self.chat_texto.tag_add("direita", inicio, "end-1c")
        self._editar_chat(escrever)

    def _largura_bolha(self):
        return max(int(self.chat_texto.winfo_width() * 0.7), 220)

    def _ajustar_bolhas(self, _e=None):
        largura = self._largura_bolha()
        for bolha in self._bolhas:
            bolha.config(wraplength=largura)

    def _clicar_link(self, evento):
        indice = self.chat_texto.index(f"@{evento.x},{evento.y}")
        for tag in self.chat_texto.tag_names(indice):
            caminho = self._links.get(tag)
            if caminho:
                if os.path.exists(caminho):
                    abrir_arquivo(caminho)
                else:
                    self._mensagem_sistema(f"Vídeo não encontrado\n{caminho}")
                return

    # ---- indicador "pensando..." ----
    def _mostrar_pensando(self):
        self._pensando = True

        def escrever():
            self.chat_texto.mark_set("pensando_ini", "end-1c")
            self.chat_texto.mark_gravity("pensando_ini", "left")
            self.chat_texto.insert("end", "SafeScreen.AI está pensando", "pensando")
        self._editar_chat(escrever)
        self._animar_pensando(0)

    def _animar_pensando(self, passo):
        if not self._pensando:
            return

        def escrever():
            self.chat_texto.delete("pensando_ini", "end-1c")
            self.chat_texto.insert("end", "SafeScreen.AI está pensando" + "." * (passo % 4), "pensando")
        self._editar_chat(escrever)
        self.janela.after(400, self._animar_pensando, passo + 1)

    def _esconder_pensando(self):
        if not self._pensando:
            return
        self._pensando = False
        self._editar_chat(lambda: self.chat_texto.delete("pensando_ini", "end-1c"))

    # ---- envio ----
    def _atualizar_placeholder(self, _e=None):
        vazio = not self.chat_entrada.get("1.0", "end-1c").strip()
        if vazio and self.janela.focus_get() is not self.chat_entrada:
            self.placeholder.place(x=14, y=11)
        else:
            self.placeholder.place_forget()
        cor = VERDE if self.janela.focus_get() is self.chat_entrada else BORDA
        self.caixa_entrada.config(highlightbackground=cor)

    def _ao_apertar_enter(self, evento):
        if evento.state & 0x0001:  # Shift+Enter: quebra de linha normal
            return None
        self._enviar_chat()
        return "break"

    def _enviar_chat(self, texto=None):
        texto = (texto or self.chat_entrada.get("1.0", "end-1c")).strip()
        if not texto or self._pensando:
            return
        self.chat_entrada.delete("1.0", "end")
        self._atualizar_placeholder()
        self.frame_sugestoes.pack_forget()

        self._mensagem_usuario(texto)
        self._mostrar_pensando()
        self.botao_enviar.config(state="disabled")
        pintar_botao(self.botao_enviar, PAINEL_2, PAINEL_2, TEXTO_SUAVE)

        def conversar():
            self.fila_chat.put(self.assistente.perguntar(texto))

        threading.Thread(target=conversar, daemon=True).start()

    def _mensagem_inicial_chat(self):
        if chat_disponivel():
            self.label_motores.config(text=descricao_motores())
            self._mensagem_ia("Olá! Posso **resumir o que aconteceu**, olhar a câmera agora, "
                              "analisar o vídeo de um evento, gravar e te mandar no celular. "
                              "O que você precisa?")
        else:
            self.label_motores.config(text="IA desativada")
            self._mensagem_sistema("Chat indisponível\nConfigure CHAVE_GROQ ou CHAVE_GEMINI em "
                                   "ia_generativa.py. A detecção e os alertas continuam funcionando.")
        self.frame_sugestoes.pack(side="bottom", fill="x", padx=(10, 0), pady=(8, 0))

    def _nova_conversa(self):
        if self._pensando:
            return
        self.assistente.nova_conversa()
        for bolha in self._bolhas:
            bolha.destroy()
        self._bolhas, self._links = [], {}
        self._editar_chat(lambda: self.chat_texto.delete("1.0", "end"))
        self._mensagem_inicial_chat()

    def _verificar_chat(self):
        # Resposta da IA
        try:
            resposta = self.fila_chat.get_nowait()
        except queue.Empty:
            resposta = None
        if resposta is not None:
            self._esconder_pensando()
            self._mensagem_ia(resposta)
            self.botao_enviar.config(state="normal")
            pintar_botao(self.botao_enviar, VERDE, VERDE_HOVER, "#ffffff")

        # Avisos do gravador (evento registrado / gravação concluída). Esperam a IA
        # terminar de responder para não se misturarem com o "pensando...".
        if not self._pensando:
            try:
                while True:
                    self._mensagem_sistema(self.captura.gravador.avisos.get_nowait())
                    self._atualizar_lista_eventos()
            except queue.Empty:
                pass

    # ---- recolher / mostrar o chat ----
    def _alternar_chat(self):
        if self._chat_visivel:
            try:
                self._pos_divisoria_chat = self.paineis.sashpos(0)
            except tk.TclError:
                self._pos_divisoria_chat = None
            self.paineis.forget(self.painel_chat)
            self.botao_chat.config(text="Mostrar assistente  «")
        else:
            self.paineis.add(self.painel_chat, weight=2)
            self.botao_chat.config(text="Ocultar assistente  »")
            if self._pos_divisoria_chat:
                self.janela.after(50, lambda: self.paineis.sashpos(0, self._pos_divisoria_chat))
        self._chat_visivel = not self._chat_visivel

    def _posicionar_divisorias(self):
        try:
            largura = self.paineis.winfo_width()
            altura = self.lado_esquerdo.winfo_height()
            if largura > 100:
                self.paineis.sashpos(0, int(largura * 0.6))
            if altura > 100:
                self.lado_esquerdo.sashpos(0, int(altura * 0.66))
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Loop da câmera
    # ------------------------------------------------------------------
    def _loop_video(self):
        try:
            resultado = self.captura.saida.get_nowait()
        except queue.Empty:
            resultado = None

        if resultado is not None:
            self._exibir_resultado(resultado)

        self._verificar_chat()
        self._atualizar_botao_seguranca()

        # Só consulta a fila de novo em 15ms - não faz nenhum trabalho pesado aqui
        self.janela.after(15, self._loop_video)

    def _exibir_resultado(self, resultado):
        frame = resultado["frame"]
        threshold = resultado["threshold"]
        alertas = resultado["alertas"]

        if alertas:
            self.label_status.config(text=f"●  ALERTA: {alertas[0]}", fg=ALERTA)
        elif self.captura.gravador.gravando:
            self.label_status.config(text="●  Gravando...", fg=AVISO)
        elif self.captura.modo_seguranca:
            self.label_status.config(text="●  Tudo normal", fg=VERDE_CLARO)
        else:
            self.label_status.config(text="○  Modo câmera (sem detecção)", fg=TEXTO_SUAVE)

        if self.modo_visualizacao == "threshold":
            imagem = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
        else:
            imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        imagem_tk = ImageTk.PhotoImage(Image.fromarray(self._encaixar_imagem(imagem)))
        self.label_video.configure(image=imagem_tk, text="")
        self.label_video.image = imagem_tk  # evita que o garbage collector descarte a imagem

    def ao_fechar(self):
        self.captura.parar()
        self.janela.destroy()


def iniciar_interface():
    janela = tk.Tk()
    aplicar_tema(janela)
    InterfaceSafeScreen(janela)
    janela.mainloop()
