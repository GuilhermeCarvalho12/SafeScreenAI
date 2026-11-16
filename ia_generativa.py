"""
ia_generativa.py - a parte de IA GENERATIVA do SafeScreen.AI.

Usa DUAS IAs, cada uma no que faz melhor:
  - GROQ (modelos abertos, ex: gpt-oss / Llama): o CHAT e os comandos. Muito rápido.
  - GOOGLE GEMINI (multimodal): tudo que precisa OLHAR A IMAGEM da câmera.
Se só uma chave estiver configurada, o sistema usa a que existir.

Duas funções:

1. RELATO DE EVENTOS: quando um evento termina, o Gemini recebe a imagem do
   momento do alerta + os alertas do rastreador e escreve um relato curto em
   português ("Uma pessoa ficou parada perto da porta por 12 s...").

2. ASSISTENTE (CHATBOT): o usuário conversa com o sistema pela interface.
   A IA usa FUNCTION CALLING: ela decide sozinha qual função do sistema
   chamar para atender o pedido, por exemplo:
     "Como foi a última hora?"          -> consultar_historico(60)
     "O que está acontecendo agora?"     -> ver_camera_agora()
     "Grava 30 segundos e me manda"      -> gravar_video(30) + aviso no WhatsApp
     "Abre o vídeo do último evento"     -> abrir_video("")
     "Tinha arma no vídeo das 16:30?"    -> analisar_video("16:30", pergunta)
     "Manda um resumo no meu celular"    -> enviar_celular("...")

Configuração (chaves gratuitas - NÃO envie as chaves para o GitHub/professor):
  - Gemini: https://aistudio.google.com/apikey  -> CHAVE_GEMINI abaixo
  - Groq:   https://console.groq.com/keys         -> CHAVE_GROQ abaixo

Toda chamada às IAs (prompt, resposta, funções usadas e tempo) fica salva em
eventos/log_prompts.jsonl - material para o relatório ("prompts utilizados" e
"visão crítica").
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

import cv2
import requests

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# --- Configuração ---
CHAVE_GEMINI = os.getenv("GEMINI_API_KEY", "")   # ou cole aqui: "AIza..."
MODELO = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
PASTA_EVENTOS = "eventos"
CAMINHO_LOG_PROMPTS = os.path.join(PASTA_EVENTOS, "log_prompts.jsonl")
GRAVACAO_MAXIMA_SEGUNDOS = 300   # limite para pedidos de gravação pelo chat
TENTATIVAS = 1                   # tentativas no mesmo modelo antes de ir para o reserva
TEMPO_LIMITE_SEGUNDOS = 40       # tempo máximo de espera por resposta em cada tentativa

# --- Groq (chat) ---
CHAVE_GROQ = os.getenv("GROQ_API_KEY", "")       # ou cole aqui: "gsk_..."
# Ordem de preferência. Na hora de usar, o programa confere quais modelos a chave
# realmente tem disponíveis (lista em /models) e pula os que não existirem.
MODELOS_GROQ = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b",
                "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
URL_GROQ = "https://api.groq.com/openai/v1/chat/completions"
URL_GROQ_MODELOS = "https://api.groq.com/openai/v1/models"
LIMITE_RESULTADO_FUNCAO = 1500   # caracteres do resultado de uma função mantidos no histórico

# ---------------------------------------------------------------------------
# Prompts (documentar no relatório)
# ---------------------------------------------------------------------------
# Regras comuns de segurança/ética para toda análise de imagem
_REGRAS_VISAO = """Observe com atenção:
- objetos nas mãos: arma de fogo, faca, barra, ferramenta usada como ameaça;
- rosto coberto: máscara, balaclava, capuz puxado, capacete em local fechado;
- objetos sendo escondidos ou levados, porta/janela sendo forçada, briga ou queda.
Uma arma de brinquedo não pode ser diferenciada de uma real pela câmera: chame de
"arma aparente" e diga que não é possível confirmar se é real.
O detector automático (YOLO) só conhece 80 objetos comuns e NÃO reconhece armas:
ele costuma chamar uma arma de "celular", "controle" ou "secador". Por isso, use as
IMAGENS como fonte principal e os rótulos do YOLO só como pista.
Regras: descreva apenas o que é visível; não identifique pessoas; não deduza etnia,
idade ou outras características sensíveis; se a imagem estiver escura, borrada ou
ambígua, diga isso em vez de inventar detalhes."""

PROMPT_RELATO_EVENTO = """Você é o analista de segurança do SafeScreen.AI, instalado em
uma loja/empresa. Abaixo estão {n_imagens} imagem(ns) do evento, em ordem cronológica.

Alertas do detector automático (YOLOv8 + rastreador de comportamento):
{alertas}
Horário: {horario}.

""" + _REGRAS_VISAO + """

Responda EXATAMENTE neste formato, em português, uma informação por linha:
Cena: <o que aparece e onde, 1 frase>
Ação: <o que a pessoa/animal parece fazer, 1 frase>
Riscos: <"nenhum" ou lista curta: arma aparente, rosto coberto, objeto sendo levado...>
Atenção: <BAIXA, MÉDIA ou ALTA> - <justificativa curta>

Critérios: ALTA = arma aparente, ameaça, invasão ou rosto coberto com atitude suspeita;
MÉDIA = comportamento incomum sem ameaça clara; BAIXA = rotina."""

PROMPT_CAMERA_AGORA = """Esta é a imagem atual da câmera de segurança do SafeScreen.AI.
Pedido do usuário: "{pedido}".

""" + _REGRAS_VISAO + """

Responda em português, em até 3 frases, e termine com uma linha
"Riscos: ..." ("nenhum" se não houver)."""

PROMPT_ANALISE_VIDEO = """Estas são {n_imagens} imagens, em ordem cronológica, de um vídeo
da câmera de segurança do SafeScreen.AI (duração ~{duracao} s, evento de {data_hora}).
Pergunta do usuário: "{pergunta}"

""" + _REGRAS_VISAO + """

Responda à pergunta de forma objetiva, em português, com base APENAS nas imagens.
Se algo aparece só em algumas imagens, diga em qual momento (início, meio, fim).
Termine com duas linhas:
Riscos: <"nenhum" ou lista curta>
Atenção: <BAIXA, MÉDIA ou ALTA> - <justificativa curta>"""

PROMPT_SISTEMA_ASSISTENTE = """Você é o assistente do SafeScreen.AI, um sistema de
monitoramento por câmera para lojas e empresas. Você conversa em português, de forma
curta, clara e educada.

Você tem funções para consultar o sistema e executar ordens. Use-as sempre que o
pedido depender de dados reais: NUNCA invente eventos, horários ou o que a câmera está
mostrando. Se uma função não retornar nada, diga que não houve registro.

Orientações:
- Perguntas sobre "como foi", "aconteceu algo", "teve alguém" -> consultar_historico
  com o período pedido em minutos (1 hora = 60, hoje = minutos desde 00:00).
- Perguntas sobre o momento atual -> ver_camera_agora.
- Perguntas sobre o CONTEÚDO de um evento/vídeo ("tinha arma?", "a pessoa estava de
  máscara?", "o que a pessoa fez?") -> analisar_video com a data_hora do evento
  (consulte o histórico antes se precisar achar o evento). NÃO responda isso só pelo
  histórico nem pelos rótulos do YOLO: o YOLO não reconhece armas e erra nomes.
- Se o relato de um evento disser "Atenção: ALTA" ou citar riscos, destaque isso.
- Pedidos para gravar -> gravar_video. Se o usuário pedir para "mandar"/"enviar",
  use avisar_celular_ao_terminar=true.
- Pedidos para abrir/mostrar um vídeo -> abrir_video. Para um vídeo específico
  ("o do teclado", "o das 15:33"), consulte o histórico antes para achar a data_hora
  do evento e passe essa data_hora. Vazio = vídeo mais recente.
- NUNCA diga que abriu um vídeo sem ter chamado abrir_video.
- Pedidos para mandar algo no celular/Telegram/WhatsApp -> enviar_celular com um
  texto curto; para mandar também um vídeo, passe a data_hora do evento em
  data_hora_video ("" = sem vídeo).
- Ligar/desligar o monitoramento -> alterar_seguranca, e confirme o que foi feito.
- Cada mensagem do usuário começa com [Agora: data hora]; use para calcular períodos.
- Ao resumir o histórico, cite horários e quantidades, e destaque o que for incomum.
  Lembre que o detector (YOLO) às vezes erra o nome de objetos (ex: chama algo de
  "hot dog" ou "surfboard"): trate objetos improváveis como possível erro de detecção.

Formato das respostas (a tela mostra Markdown, como no ChatGPT):
- Comece direto pela resposta. Respostas simples: 1 a 3 frases, sem títulos.
- Use **negrito** com moderação: no máximo 2 ou 3 destaques por resposta.
- Use lista com "-" quando houver 3 ou mais itens.
- Tabela só para listar vários eventos com horário (máximo 8 linhas, 3 colunas curtas).
- Título (##) só em resumos longos, e no máximo um.
- Não use separadores (---), emojis em excesso nem barras invertidas antes de * ou _.
- Ao citar um vídeo, escreva o caminho exatamente como veio da função (ex:
  eventos\\evento_20261109_155122.mp4), sem negrito: ele vira um link clicável."""


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
_cliente = None


def ia_disponivel():
    """Gemini (visão) disponível?"""
    return genai is not None and bool(CHAVE_GEMINI)


def groq_disponivel():
    return bool(CHAVE_GROQ)


def chat_disponivel():
    return groq_disponivel() or ia_disponivel()


def descricao_motores():
    chat = "Groq" if groq_disponivel() else ("Gemini" if ia_disponivel() else "indisponível")
    visao = "Gemini" if ia_disponivel() else "indisponível (só YOLO)"
    try:
        from notificacao import canais_ativos
        avisos = ", ".join(canais_ativos()) or "nenhum"
    except Exception:
        avisos = "?"
    return f"Chat: {chat} • Imagens: {visao} • Avisos: {avisos}"


def motivo_indisponivel():
    if genai is None:
        return "biblioteca google-genai não instalada (pip install google-genai)"
    if not CHAVE_GEMINI:
        return "chave GEMINI_API_KEY não configurada"
    return ""


_modelo_atual = MODELO      # pode mudar para um modelo reserva se o principal estiver lotado
_reserva_procurada = False


def _obter_cliente():
    global _cliente
    if _cliente is None:
        _cliente = genai.Client(
            api_key=CHAVE_GEMINI,
            http_options=types.HttpOptions(timeout=TEMPO_LIMITE_SEGUNDOS * 1000))  # em ms
    return _cliente


def _versao(nome):
    """'gemini-3.8-flash-lite' -> 3.8 (para preferir os modelos mais novos)."""
    import re
    achado = re.search(r"gemini-(\d+(?:\.\d+)?)", nome)
    return float(achado.group(1)) if achado else 0.0


def _listar_modelos_reserva():
    """Pergunta ao Google quais modelos 'flash' esta chave pode usar, do mais novo para
    o mais antigo (modelos antigos às vezes aparecem na lista mas já foram desativados)."""
    global _reserva_procurada
    _reserva_procurada = True
    try:
        nomes = []
        for modelo in _obter_cliente().models.list():
            nome = (modelo.name or "").replace("models/", "")
            acoes = getattr(modelo, "supported_actions", None) or []
            if acoes and "generateContent" not in acoes:
                continue
            if "flash" not in nome or nome == _modelo_atual:
                continue
            if any(x in nome for x in ("image", "tts", "audio", "live", "embedding", "thinking")):
                continue
            nomes.append(nome)
    except Exception as erro:
        print(f"[SafeScreen.AI] Não foi possível listar modelos: {erro}")
        return []
    # mais novo primeiro; na mesma versão, "lite" primeiro (costuma estar menos lotado)
    nomes.sort(key=lambda n: (-_versao(n), "lite" not in n, "preview" in n, "exp" in n, n))
    return nomes[:5]


def _imagem_para_parte(frame, largura_max=960):
    """Converte o frame (BGR do OpenCV) em JPEG para mandar ao Gemini."""
    altura, largura = frame.shape[:2]
    if largura > largura_max:
        frame = cv2.resize(frame, (largura_max, int(altura * largura_max / largura)))
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg")


def registrar_prompt(tipo, prompt, resposta, latencia, ferramentas=None, modelo=None):
    os.makedirs(PASTA_EVENTOS, exist_ok=True)
    with open(CAMINHO_LOG_PROMPTS, "a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps({
            "quando": datetime.now().isoformat(timespec="seconds"),
            "tipo": tipo, "modelo": modelo or _modelo_atual, "latencia_s": round(latencia, 2),
            "funcoes_chamadas": ferramentas or [],
            "prompt": prompt, "resposta": resposta,
        }, ensure_ascii=False) + "\n")


def _erro_temporario(erro):
    """503 (sobrecarregado), 504 (demorou demais), 429 (limite de uso) e 500 costumam passar."""
    texto = str(erro)
    return any(c in texto for c in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500 ",
                                    "504", "DEADLINE_EXCEEDED", "timed out", "Timeout", "timeout"))


def _com_retentativa(chamada):
    """chamada(modelo) -> resposta. Em erro temporário tenta de novo após 1 s; se o
    modelo continuar lotado, troca para um modelo reserva e tenta mais uma vez."""
    global _modelo_atual
    for tentativa in range(TENTATIVAS):
        try:
            return chamada(_modelo_atual)
        except Exception as erro:
            if not _erro_temporario(erro):
                raise
            ultimo_erro = erro
            if tentativa < TENTATIVAS - 1:
                print(f"[SafeScreen.AI] IA ocupada ({_modelo_atual}), tentando de novo...")
                time.sleep(1)

    if _reserva_procurada:
        raise ultimo_erro
    principal = _modelo_atual
    for reserva in _listar_modelos_reserva():
        print(f"[SafeScreen.AI] {principal} lotado - tentando o modelo reserva {reserva}...")
        try:
            resposta = chamada(reserva)
        except Exception as erro:
            if "NOT_FOUND" in str(erro) or _erro_temporario(erro):
                print(f"[SafeScreen.AI] {reserva} também não serviu, tentando o próximo.")
                continue
            raise
        _modelo_atual = reserva  # as próximas chamadas já usam o reserva que funcionou
        return resposta
    raise ultimo_erro


def mensagem_de_erro(erro):
    """Texto curto e amigável para mostrar no chat (o erro completo vai para o terminal)."""
    texto = str(erro)
    print(f"[SafeScreen.AI] Erro da IA: {texto}")
    if "Groq" in texto and ("API key not valid" in texto or "401" in texto):
        return "a chave do Groq é inválida"
    if "Groq" in texto:
        if "429" in texto or "rate_limit" in texto:
            return "limite de uso gratuito do Groq atingido por agora, espere 1 minuto e tente de novo"
        return "o Groq não respondeu (detalhes no terminal)"
    if "API_KEY_INVALID" in texto or "API key not valid" in texto:
        return "a chave do Gemini é inválida"
    if "NOT_FOUND" in texto:
        return f"o modelo '{MODELO}' não está disponível - troque MODELO em ia_generativa.py"
    if _erro_temporario(erro):
        return "os servidores do Google estão lentos ou sobrecarregados agora, tente de novo em instantes"
    return texto[:200]


def _descrever(frames, prompt, tipo):
    """frames: uma imagem ou uma lista de imagens (em ordem cronológica)."""
    if not ia_disponivel():
        return None
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    inicio = time.time()
    try:
        largura = 960 if len(frames) == 1 else 640  # várias imagens: menores
        conteudo = [_imagem_para_parte(f, largura) for f in frames] + [prompt]
        resposta = _com_retentativa(lambda modelo: _obter_cliente().models.generate_content(
            model=modelo, contents=conteudo))
        texto = (resposta.text or "").strip() or "(a IA não retornou texto)"
    except Exception as erro:
        texto = f"(relato indisponível: {mensagem_de_erro(erro)})"
    registrar_prompt(tipo, prompt + f"\n[{len(frames)} imagem(ns) anexada(s)]", texto,
                     time.time() - inicio)
    return texto


def ler_quadros_video(caminho, quantidade=6):
    """Pega `quantidade` quadros espalhados pelo vídeo (início, meio e fim)."""
    video = cv2.VideoCapture(caminho)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = video.get(cv2.CAP_PROP_FPS) or 20
    quadros = []
    if total > 0:
        for i in range(quantidade):
            video.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / quantidade))
            ok, quadro = video.read()
            if ok:
                quadros.append(quadro)
    video.release()
    return quadros, round(total / fps) if total else 0


def analisar_video_arquivo(caminho, pergunta, data_hora="", quantidade=6):
    """Manda quadros do vídeo + a pergunta para o Gemini. Retorna o texto (ou None)."""
    quadros, duracao = ler_quadros_video(caminho, quantidade)
    if not quadros:
        return "(não foi possível ler o vídeo)"
    prompt = PROMPT_ANALISE_VIDEO.format(n_imagens=len(quadros), duracao=duracao,
                                         data_hora=data_hora or os.path.basename(caminho),
                                         pergunta=pergunta)
    return _descrever(quadros, prompt, "analise_video")


def extrair_campo(relato, campo):
    """Lê uma linha "Campo: valor" do relato (ex: Riscos, Atenção)."""
    import re
    achado = re.search(rf"{campo}\s*:\s*(.+)", relato or "", flags=re.IGNORECASE)
    return achado.group(1).strip() if achado else ""


# ---------------------------------------------------------------------------
# 1. Relato automático de eventos
# ---------------------------------------------------------------------------
def descrever_evento(frames, alertas):
    """frames: imagem do alerta ou lista de imagens do evento. Retorna o relato (str)."""
    if frames is not None and not isinstance(frames, (list, tuple)):
        frames = [frames]
    frames = [f for f in (frames or []) if f is not None]
    lista = "\n".join(f"- {a}" for a in alertas) if alertas else "- movimento não identificado"
    prompt = PROMPT_RELATO_EVENTO.format(alertas=lista, n_imagens=len(frames),
                                         horario=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    relato = _descrever(frames, prompt, "relato_evento") if frames else None
    if relato is None:
        relato = "Relato automático indisponível (" + (motivo_indisponivel() or "sem imagem") + ")."
    return relato


# ---------------------------------------------------------------------------
# 2. Assistente com function calling
# ---------------------------------------------------------------------------
class AssistenteSafeScreen:
    """
    captura: objeto CapturaSafeScreen (dá acesso à câmera, gravador e segurança).
    """

    def __init__(self, captura):
        self.captura = captura
        self._funcoes_chamadas = []
        self._chat = None
        self._modelo_do_chat = None
        self._chat_groq = None

    # ---- funções que o Gemini pode chamar ----------------------------------
    # Os nomes, docstrings e tipos dos parâmetros são lidos pelo SDK e enviados
    # ao modelo: é assim que ele "sabe" o que cada função faz.
    def _criar_ferramentas(self):
        from registro_eventos import ler_atividade, ler_eventos
        from notificacao import canais_ativos, enviar_mensagem
        assistente = self

        def achar_video(data_hora):
            """Acha o evento com vídeo pela data/hora (ou o mais recente se vazio)."""
            eventos = ler_eventos(datetime.now() - timedelta(days=30))
            videos = [e for e in eventos if e.get("video") and os.path.exists(e["video"])]
            if not videos:
                return None, {"erro": "nenhum vídeo encontrado"}
            procurado = (data_hora or "").strip()
            if procurado.endswith(".mp4") and os.path.exists(procurado):
                return {"video": procurado, "data_hora": ""}, None
            if procurado:
                candidatos = [e for e in videos if procurado in e["data_hora"]]
                if not candidatos:
                    return None, {"erro": f"nenhum vídeo com data/hora '{procurado}'",
                                  "videos_disponiveis": [e["data_hora"] for e in videos[-10:]]}
                return candidatos[-1], None
            return videos[-1], None

        def consultar_historico(minutos: int) -> dict:
            """Consulta o que o sistema registrou nos últimos `minutos` minutos:
            alertas/eventos gravados (com relato da IA e vídeo) e um resumo minuto a
            minuto (ou hora a hora, para períodos longos) dos objetos detectados."""
            assistente._funcoes_chamadas.append(f"consultar_historico({minutos})")
            desde = datetime.now() - timedelta(minutes=max(1, int(minutos)))
            eventos = ler_eventos(desde)
            atividade = ler_atividade(desde, agrupar_por_hora=minutos > 180)
            return {
                "periodo": f"desde {desde:%d/%m %H:%M} até agora",
                "seguranca_ativa": assistente.captura.modo_seguranca,
                "total_eventos": len(eventos),
                # resultado enxuto: planos gratuitos limitam tokens por minuto
                "eventos": [{"data_hora": e["data_hora"], "alertas_yolo": e["alertas"][:140],
                             "atencao": extrair_campo(e["relato_ia"], "Aten[çc][ãa]o"),
                             "riscos": extrair_campo(e["relato_ia"], "Riscos"),
                             "relato_ia": e["relato_ia"][:300], "video": e["video"]}
                            for e in eventos[-12:]],
                "atividade": atividade[-40:],
                "observacao": "atividade vazia = nada detectado (ou segurança desativada). "
                              "Rótulos do YOLO podem estar errados e ele não reconhece armas; "
                              "para saber o conteúdo de um vídeo use analisar_video.",
            }

        def ver_camera_agora(pedido: str) -> dict:
            """Olha a câmera neste exato momento e descreve o que aparece.
            `pedido` é o que o usuário quer saber sobre a imagem."""
            assistente._funcoes_chamadas.append("ver_camera_agora")
            frame = assistente.captura.ultimo_frame
            if frame is None:
                return {"erro": "câmera ainda sem imagem"}
            if ia_disponivel():
                prompt = PROMPT_CAMERA_AGORA.format(pedido=pedido)
                descricao = _descrever(frame.copy(), prompt, "camera_agora")
                if descricao and not descricao.startswith("(relato indisponível"):
                    return {"descricao": descricao, "fonte": "Gemini (análise da imagem)"}
            # Plano B: sem a IA de visão, responde com o que o YOLO está vendo
            import detectmov
            objetos = [nome for nome, *_ in detectmov.ultimas_deteccoes]
            return {"fonte": "YOLO (a IA de imagem não respondeu)",
                    "objetos_detectados_agora": objetos,
                    "observacao": "o YOLO só analisa quando há movimento; lista vazia "
                                  "pode significar cena parada"}

        def gravar_video(segundos: int, avisar_celular_ao_terminar: bool) -> dict:
            """Começa a gravar a câmera agora por `segundos` segundos (o vídeo inclui
            também os 15 s anteriores ao pedido). Se `avisar_celular_ao_terminar`
            for true, manda o aviso (e o vídeo, pelo Telegram) para o celular."""
            segundos = max(5, min(int(segundos), GRAVACAO_MAXIMA_SEGUNDOS))
            assistente._funcoes_chamadas.append(f"gravar_video({segundos})")
            assistente.captura.gravador.gravar_manual(segundos, _bool(avisar_celular_ao_terminar))
            return {"status": "gravação iniciada", "duracao_s": segundos,
                    "limite_s": GRAVACAO_MAXIMA_SEGUNDOS,
                    "aviso": "o usuário será avisado no chat quando terminar"}

        def abrir_video(data_hora: str) -> dict:
            """Abre no computador o vídeo de um evento/gravação. `data_hora` é a data e
            hora do evento como aparece no histórico (ex: "2026-11-09 15:33:57", ou só
            "15:33"); deixe vazio ("") para abrir o vídeo mais recente."""
            assistente._funcoes_chamadas.append(f"abrir_video({data_hora!r})")
            evento, erro = achar_video(data_hora)
            if erro:
                return erro
            abrir_arquivo(evento["video"])
            return {"aberto": evento["video"], "evento": evento.get("data_hora", "")}

        def analisar_video(data_hora: str, pergunta: str) -> dict:
            """Assiste ao vídeo de um evento (vários quadros do início ao fim) com a IA de
            visão e responde `pergunta` sobre o conteúdo: se havia arma, rosto coberto,
            o que a pessoa fez etc. `data_hora` como no histórico; "" = mais recente."""
            assistente._funcoes_chamadas.append(f"analisar_video({data_hora!r})")
            evento, erro = achar_video(data_hora)
            if erro:
                return erro
            if not ia_disponivel():
                return {"erro": "IA de visão (Gemini) indisponível; só há o relato salvo",
                        "relato_salvo": evento.get("relato_ia", "")}
            resposta = analisar_video_arquivo(evento["video"], pergunta, evento.get("data_hora", ""))
            return {"video": evento["video"], "data_hora": evento.get("data_hora", ""),
                    "analise": resposta}

        def enviar_celular(mensagem: str, data_hora_video: str) -> dict:
            """Envia uma mensagem para o celular do responsável (Telegram e/ou WhatsApp).
            Para anexar o vídeo de um evento, passe a data_hora dele em
            `data_hora_video` ("" = sem vídeo). O Telegram envia o vídeo; o WhatsApp só texto."""
            assistente._funcoes_chamadas.append("enviar_celular")
            video = None
            if (data_hora_video or "").strip():
                evento, erro = achar_video(data_hora_video)
                if erro:
                    return erro
                video = evento["video"]
            ok, detalhe = enviar_mensagem(mensagem, video)
            return {"enviado": ok, "detalhe": detalhe, "canais": canais_ativos()}

        def alterar_seguranca(ativa: bool) -> dict:
            """Liga (true) ou desliga (false) o modo segurança (detecção e alertas)."""
            assistente._funcoes_chamadas.append(f"alterar_seguranca({ativa})")
            assistente.captura.definir_seguranca(_bool(ativa))
            return {"seguranca_ativa": assistente.captura.modo_seguranca}

        return [consultar_historico, ver_camera_agora, gravar_video,
                abrir_video, analisar_video, enviar_celular, alterar_seguranca]

    def _obter_chat(self, modelo):
        if self._chat is None or self._modelo_do_chat != modelo:
            historico = []
            if self._chat is not None:  # trocou de modelo: mantém a conversa até aqui
                try:
                    historico = self._chat.get_history()
                except Exception:
                    historico = []
            config = types.GenerateContentConfig(
                system_instruction=PROMPT_SISTEMA_ASSISTENTE,
                tools=self._criar_ferramentas(),
                temperature=0.3,
            )
            self._chat = _obter_cliente().chats.create(model=modelo, config=config,
                                                       history=historico)
            self._modelo_do_chat = modelo
        return self._chat

    def perguntar(self, texto):
        """Envia a mensagem do usuário e devolve a resposta (str). Bloqueia:
        chame em uma thread separada para não travar a interface."""
        if not chat_disponivel():
            return "O chat não está disponível: configure CHAVE_GROQ ou CHAVE_GEMINI em ia_generativa.py."
        mensagem = f"[Agora: {datetime.now():%d/%m/%Y %H:%M}] {texto}"
        self._funcoes_chamadas = []
        inicio = time.time()

        if groq_disponivel():
            if self._chat_groq is None:
                self._chat_groq = ChatGroq(self._criar_ferramentas())
            try:
                texto_resposta = self._chat_groq.enviar(mensagem)
            except Exception as erro:
                texto_resposta = f"Não consegui falar com a IA agora: {mensagem_de_erro(erro)}."
            registrar_prompt("chat", mensagem, texto_resposta, time.time() - inicio,
                             self._funcoes_chamadas, modelo=f"groq:{self._chat_groq.modelo}")
            return texto_resposta

        try:
            resposta = _com_retentativa(
                lambda modelo: self._obter_chat(modelo).send_message(mensagem))
            texto_resposta = (resposta.text or "").strip() or "(sem resposta)"
        except Exception as erro:
            self._chat = None  # recomeça a conversa na próxima mensagem
            texto_resposta = f"Não consegui falar com a IA agora: {mensagem_de_erro(erro)}."
        registrar_prompt("chat", mensagem, texto_resposta, time.time() - inicio,
                         self._funcoes_chamadas)
        return texto_resposta

    def nova_conversa(self):
        self._chat = None
        self._chat_groq = None


# ---------------------------------------------------------------------------
# Chat pelo Groq (API compatível com a da OpenAI, chamada via requests)
# ---------------------------------------------------------------------------
# Descrição dos parâmetros de cada função, no formato JSON Schema que o Groq exige.
# (A descrição da função em si vem da docstring.)
PARAMETROS_FERRAMENTAS = {
    "consultar_historico": {"minutos": ("integer", "período em minutos (1 hora = 60)")},
    "ver_camera_agora": {"pedido": ("string", "o que o usuário quer saber sobre a imagem")},
    "gravar_video": {"segundos": ("integer", "duração da gravação em segundos"),
                     "avisar_celular_ao_terminar": ("boolean", "mandar aviso e vídeo para o celular ao terminar")},
    "abrir_video": {"data_hora": ("string", "data/hora do evento; vazio = mais recente")},
    "analisar_video": {"data_hora": ("string", "data/hora do evento; vazio = mais recente"),
                       "pergunta": ("string", "o que o usuário quer saber sobre o vídeo")},
    "enviar_celular": {"mensagem": ("string", "texto curto a enviar"),
                       "data_hora_video": ("string", "data/hora do vídeo a anexar; vazio = sem vídeo")},
    "alterar_seguranca": {"ativa": ("boolean", "true liga, false desliga")},
}
MAX_MENSAGENS_HISTORICO = 40


def _bool(valor):
    if isinstance(valor, str):
        return valor.strip().lower() in ("true", "1", "sim", "yes")
    return bool(valor)


def _esquema_ferramenta(funcao):
    import inspect
    params = PARAMETROS_FERRAMENTAS.get(funcao.__name__, {})
    return {"type": "function", "function": {
        "name": funcao.__name__,
        "description": " ".join((inspect.getdoc(funcao) or "").split()),
        "parameters": {
            "type": "object",
            "properties": {n: {"type": t, "description": d} for n, (t, d) in params.items()},
            "required": list(params),
        },
    }}


class ChatGroq:
    """Conversa com o Groq e executa as funções que o modelo pedir (function calling).

    O laço: manda a conversa -> se o modelo pedir uma função, executamos e devolvemos
    o resultado -> repete até ele responder em texto (no máximo 6 voltas)."""

    def __init__(self, ferramentas):
        self.funcoes = {f.__name__: f for f in ferramentas}
        self.esquemas = [_esquema_ferramenta(f) for f in ferramentas]
        self.mensagens = [{"role": "system", "content": PROMPT_SISTEMA_ASSISTENTE}]
        self.modelos = _modelos_groq_disponiveis()
        self.modelo = self.modelos[0]

    def _post(self, modelo):
        return requests.post(
            URL_GROQ, timeout=TEMPO_LIMITE_SEGUNDOS,
            headers={"Authorization": f"Bearer {CHAVE_GROQ}"},
            json={"model": modelo, "messages": self.mensagens, "tools": self.esquemas,
                  "tool_choice": "auto", "temperature": 0.3})

    def _requisitar(self):
        """Tenta o modelo atual; se falhar, tenta os outros disponíveis."""
        ultimo_erro = None
        inicio = self.modelos.index(self.modelo) if self.modelo in self.modelos else 0
        for modelo in self.modelos[inicio:] + self.modelos[:inicio]:
            try:
                resposta = self._post(modelo)
                if resposta.status_code == 413 or "context_length" in resposta.text:
                    # conversa grande demais para o limite: mantém só o essencial e tenta de novo
                    print("[SafeScreen.AI] Groq: conversa longa demais, resumindo o histórico...")
                    self._aparar_historico(forcar=True)
                    resposta = self._post(modelo)
            except requests.RequestException as erro:
                ultimo_erro = Exception(f"timeout/conexão com o Groq: {erro}")
                print(f"[SafeScreen.AI] Groq ({modelo}): {ultimo_erro}")
                continue
            if resposta.status_code == 200:
                if modelo != self.modelo:
                    print(f"[SafeScreen.AI] Groq: passando a usar o modelo {modelo}.")
                self.modelo = modelo
                return resposta.json()["choices"][0]["message"]
            if resposta.status_code == 401:
                raise Exception("401 API key not valid (Groq)")
            ultimo_erro = Exception(f"{resposta.status_code} Groq: {resposta.text[:300]}")
            print(f"[SafeScreen.AI] Groq ({modelo}) falhou: {resposta.status_code} {resposta.text[:300]}")
        raise ultimo_erro

    def _aparar_historico(self, forcar=False):
        # Mantém o prompt do sistema + as últimas mensagens, cortando sempre no início
        # de uma pergunta do usuário (para não separar uma função do seu resultado).
        limite = 6 if forcar else MAX_MENSAGENS_HISTORICO
        if len(self.mensagens) <= limite + 1:
            return
        resto = self.mensagens[-limite:]
        while len(resto) > 1 and resto[0]["role"] != "user":
            resto = resto[1:]
        self.mensagens = self.mensagens[:1] + resto

    def _encolher_resultados_antigos(self):
        """Depois que a IA já respondeu, os resultados das funções não precisam ficar
        inteiros no histórico (só ocupam tokens)."""
        for msg in self.mensagens:
            if msg["role"] == "tool" and len(msg["content"]) > LIMITE_RESULTADO_FUNCAO:
                msg["content"] = msg["content"][:LIMITE_RESULTADO_FUNCAO] + "...(resumido)"

    def enviar(self, texto):
        tamanho_antes = len(self.mensagens)
        self.mensagens.append({"role": "user", "content": texto})
        self._aparar_historico()
        try:
            for _ in range(6):
                msg = self._requisitar()
                chamadas = msg.get("tool_calls") or []
                self.mensagens.append({"role": "assistant", "content": msg.get("content") or "",
                                       **({"tool_calls": chamadas} if chamadas else {})})
                if not chamadas:
                    self._encolher_resultados_antigos()
                    return (msg.get("content") or "").strip() or "(sem resposta)"
                for chamada in chamadas:
                    nome = chamada["function"]["name"]
                    try:
                        argumentos = json.loads(chamada["function"].get("arguments") or "{}")
                        resultado = self.funcoes[nome](**argumentos)
                    except Exception as erro:
                        resultado = {"erro": f"falha ao executar {nome}: {erro}"}
                    self.mensagens.append({"role": "tool", "tool_call_id": chamada["id"],
                                           "content": json.dumps(resultado, ensure_ascii=False,
                                                                 default=str)})
            return "Não consegui concluir o pedido (muitas etapas). Pode reformular?"
        except Exception:
            # Desfaz a pergunta que falhou: um histórico "quebrado" (função sem
            # resposta) faria TODAS as próximas mensagens falharem também.
            if len(self.mensagens) > tamanho_antes:
                del self.mensagens[max(tamanho_antes, 1):]
            raise


_cache_modelos_groq = None


def _modelos_groq_disponiveis():
    """Cruza a lista de preferência com os modelos que a chave realmente pode usar."""
    global _cache_modelos_groq
    if _cache_modelos_groq is not None:
        return _cache_modelos_groq
    try:
        resposta = requests.get(URL_GROQ_MODELOS, timeout=15,
                                headers={"Authorization": f"Bearer {CHAVE_GROQ}"})
        existentes = {m["id"] for m in resposta.json().get("data", [])}
    except Exception as erro:
        print(f"[SafeScreen.AI] Não foi possível listar os modelos do Groq: {erro}")
        existentes = set()
    disponiveis = [m for m in MODELOS_GROQ if m in existentes]
    if existentes and not disponiveis:
        # nenhum da lista: usa os modelos de texto que existirem
        disponiveis = sorted(m for m in existentes
                             if not any(x in m for x in ("whisper", "tts", "guard", "vision")))
    _cache_modelos_groq = disponiveis or list(MODELOS_GROQ)
    print(f"[SafeScreen.AI] Modelos do Groq que serão usados: {', '.join(_cache_modelos_groq[:4])}")
    return _cache_modelos_groq


def abrir_arquivo(caminho):
    caminho = os.path.abspath(caminho)
    if sys.platform.startswith("win"):
        os.startfile(caminho)  # noqa: abre com o player padrão do Windows
    elif sys.platform == "darwin":
        subprocess.Popen(["open", caminho])
    else:
        subprocess.Popen(["xdg-open", caminho])
