"""
notificacao.py - envia os alertas para o celular.

Canais:
  - TELEGRAM (recomendado): gratuito, confiável e manda o VÍDEO do evento junto.
  - WhatsApp via CallMeBot (opcional): só texto; configure em notificacao_whatsapp.py.
Os dois podem funcionar ao mesmo tempo.

Como configurar o Telegram (5 minutos, uma vez só):
  1. No Telegram, procure por @BotFather e mande /newbot.
  2. Escolha um nome (ex: SafeScreen AI) e um usuário terminado em "bot"
     (ex: safescreen_guilherme_bot). O BotFather responde com um TOKEN
     parecido com 123456789:AAH...  -> cole em TOKEN_TELEGRAM abaixo.
  3. Abra a conversa com o seu bot (link que o BotFather mandou) e envie /start.
  4. Rode  python notificacao.py  : deve chegar uma mensagem de teste no Telegram.
     O sistema descobre sozinho o seu chat_id e guarda em eventos/telegram_chat_id.txt. Para avisar um grupo, adicione o bot
     ao grupo e mande qualquer mensagem lá antes do primeiro alerta.

NÃO envie o token para o GitHub nem para o professor.
"""
import os
import re
import threading

import requests

import notificacao_whatsapp as whatsapp

TOKEN_TELEGRAM = os.getenv("TELEGRAM_TOKEN", "")      # ou cole aqui: "123456789:AAH..."
CHAT_ID_TELEGRAM = os.getenv("TELEGRAM_CHAT_ID", "")  # opcional: descoberto sozinho
ARQUIVO_CHAT_ID = os.path.join("eventos", "telegram_chat_id.txt")
LIMITE_VIDEO_MB = 49  # limite de envio de arquivos de bots no Telegram é 50 MB


def telegram_configurado():
    return bool(TOKEN_TELEGRAM)


def whatsapp_configurado():
    return whatsapp.ENVIO_ATIVO and whatsapp.API_KEY_CALLMEBOT not in ("", "XXXXX")


def canais_ativos():
    canais = []
    if telegram_configurado():
        canais.append("Telegram")
    if whatsapp_configurado():
        canais.append("WhatsApp")
    return canais


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def _url(metodo):
    return f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/{metodo}"


def _chat_id():
    """Usa o chat_id configurado; senão, o salvo; senão, descobre pela última
    mensagem que alguém mandou para o bot (por isso é preciso mandar /start)."""
    if CHAT_ID_TELEGRAM:
        return CHAT_ID_TELEGRAM
    if os.path.exists(ARQUIVO_CHAT_ID):
        with open(ARQUIVO_CHAT_ID, encoding="utf-8") as arquivo:
            salvo = arquivo.read().strip()
        if salvo:
            return salvo
    try:
        atualizacoes = requests.get(_url("getUpdates"), timeout=15).json().get("result", [])
    except requests.RequestException:
        return None
    for atualizacao in reversed(atualizacoes):
        mensagem = (atualizacao.get("message") or atualizacao.get("channel_post")
                    or atualizacao.get("my_chat_member") or {})
        chat = mensagem.get("chat") or {}
        if chat.get("id"):
            os.makedirs(os.path.dirname(ARQUIVO_CHAT_ID), exist_ok=True)
            with open(ARQUIVO_CHAT_ID, "w", encoding="utf-8") as arquivo:
                arquivo.write(str(chat["id"]))
            print(f"[SafeScreen.AI] Telegram: chat_id {chat['id']} descoberto e salvo.")
            return str(chat["id"])
    return None


def _chamar_telegram(metodo, timeout=60, **kwargs):
    try:
        resposta = requests.post(_url(metodo), timeout=timeout, **kwargs)
        dados = resposta.json()
    except (requests.RequestException, ValueError) as erro:
        return False, f"erro de conexão com o Telegram: {erro}"
    if dados.get("ok"):
        return True, "enviado"
    return False, f"Telegram recusou: {dados.get('description', resposta.status_code)}"


LARGURA_VIDEO_ENVIO = 640   # o vídeo enviado é uma cópia reduzida (sobe rápido pelo celular)


def _converter_com_ffmpeg(video, destino):
    """Converte para H.264 com o ffmpeg que vem no pacote imageio-ffmpeg
    (pip install imageio-ffmpeg). É o formato que o Telegram toca no celular."""
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False
    comando = [ffmpeg, "-y", "-loglevel", "error", "-i", video,
               "-vf", f"scale='min({LARGURA_VIDEO_ENVIO},iw)':-2",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
               "-pix_fmt", "yuv420p",        # obrigatório para tocar em celular
               "-movflags", "+faststart",    # começa a tocar antes de baixar tudo
               "-an", destino]
    try:
        resultado = subprocess.run(comando, capture_output=True, timeout=120,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as erro:
        print(f"[SafeScreen.AI] ffmpeg falhou: {erro}")
        return False
    if resultado.returncode != 0:
        print(f"[SafeScreen.AI] ffmpeg falhou: {resultado.stderr.decode(errors='ignore')[:300]}")
        return False
    return os.path.exists(destino) and os.path.getsize(destino) > 0


def _converter_com_opencv(video, destino):
    """Plano B sem ffmpeg: regrava com o OpenCV tentando codecs H.264
    (Media Foundation do Windows, depois avc1). Retorna o codec usado ou None."""
    import cv2
    entrada = cv2.VideoCapture(video)
    fps = entrada.get(cv2.CAP_PROP_FPS) or 15
    largura = int(entrada.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    altura = int(entrada.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if not entrada.isOpened() or not largura:
        entrada.release()
        return None
    escala = min(1.0, LARGURA_VIDEO_ENVIO / largura)
    tamanho = (int(largura * escala) // 2 * 2, int(altura * escala) // 2 * 2)

    tentativas = []
    if hasattr(cv2, "CAP_MSMF"):
        tentativas.append(("H264 (Windows)", cv2.CAP_MSMF, "H264"))
    tentativas += [("avc1", cv2.CAP_ANY, "avc1"), ("mp4v", cv2.CAP_ANY, "mp4v")]
    saida, codec = None, None
    for nome, backend, fourcc in tentativas:
        try:
            saida = cv2.VideoWriter(destino, backend, cv2.VideoWriter_fourcc(*fourcc), fps, tamanho)
        except cv2.error:
            saida = None
        if saida is not None and saida.isOpened():
            codec = nome
            break
    if not codec:
        entrada.release()
        return None
    while True:
        ok, quadro = entrada.read()
        if not ok:
            break
        saida.write(cv2.resize(quadro, tamanho, interpolation=cv2.INTER_AREA))
    entrada.release()
    saida.release()
    return codec if os.path.exists(destino) and os.path.getsize(destino) > 0 else None


def _copia_reduzida(video):
    """Cria uma cópia menor, em H.264, só para o envio (o original fica intacto no PC).
    Retorna (caminho, tocavel_no_celular)."""
    destino = os.path.join(os.path.dirname(video) or ".", "_envio_" + os.path.basename(video))
    if _converter_com_ffmpeg(video, destino):
        return destino, True
    codec = _converter_com_opencv(video, destino)
    if codec:
        if codec == "mp4v":
            print("[SafeScreen.AI] Aviso: sem H.264 disponível; o vídeo pode não tocar no celular. "
                  "Instale:  pip install imageio-ffmpeg")
        return destino, codec != "mp4v"
    return None, False


def _enviar_video_telegram(chat_id, video):
    """Envia o vídeo; tenta como vídeo e, se recusado, como arquivo. Retorna (ok, detalhe)."""
    if not os.path.exists(video):
        return False, f"arquivo não encontrado: {os.path.abspath(video)}"

    copia, tocavel = None, False
    try:
        copia, tocavel = _copia_reduzida(video)
    except Exception as erro:
        print(f"[SafeScreen.AI] Telegram: não foi possível converter o vídeo ({erro}); enviando o original.")
    arquivo_envio = copia or video
    tamanho_mb = os.path.getsize(arquivo_envio) / 1_000_000
    print(f"[SafeScreen.AI] Telegram: enviando vídeo ({tamanho_mb:.1f} MB)...")
    if tamanho_mb > LIMITE_VIDEO_MB:
        return False, f"vídeo grande demais ({tamanho_mb:.0f} MB; limite de bots é 50 MB)"

    legenda = os.path.basename(video)
    if not tocavel:
        legenda += "  (se não tocar no celular, baixe o arquivo e abra no player)"
    try:
        with open(arquivo_envio, "rb") as arquivo:
            ok, detalhe = _chamar_telegram(
                "sendVideo", timeout=180,
                data={"chat_id": chat_id, "caption": legenda, "supports_streaming": "true"},
                files={"video": (legenda, arquivo, "video/mp4")})
        if not ok:  # alguns formatos não passam como vídeo: manda como arquivo
            print(f"[SafeScreen.AI] Telegram: sendVideo falhou ({detalhe}); tentando como arquivo...")
            with open(arquivo_envio, "rb") as arquivo:
                ok, detalhe = _chamar_telegram(
                    "sendDocument", timeout=180,
                    data={"chat_id": chat_id, "caption": legenda},
                    files={"document": (legenda, arquivo, "video/mp4")})
        return ok, detalhe
    finally:
        if copia and os.path.exists(copia):
            try:
                os.remove(copia)
            except OSError:
                pass


def enviar_telegram(texto, video=None):
    if not telegram_configurado():
        return False, "Telegram não configurado (TOKEN_TELEGRAM em notificacao.py)"
    chat_id = _chat_id()
    if not chat_id:
        return False, "abra o seu bot no Telegram, mande /start e tente de novo"

    ok, detalhe = _chamar_telegram("sendMessage", data={"chat_id": chat_id, "text": texto[:4000]})
    if ok and video:
        ok_video, detalhe_video = _enviar_video_telegram(chat_id, video)
        if ok_video:
            detalhe = "mensagem e vídeo enviados"
        else:
            detalhe = f"mensagem enviada; vídeo falhou ({detalhe_video})"
            # avisa no próprio Telegram, para não ficar esperando um vídeo que não vem
            _chamar_telegram("sendMessage", data={
                "chat_id": chat_id,
                "text": f"⚠️ Não consegui enviar o vídeo: {detalhe_video[:300]}\n"
                        f"Ele está salvo no computador: {video}"})
    print(f"[SafeScreen.AI] Telegram: {detalhe}")
    return ok, detalhe


# ---------------------------------------------------------------------------
# Interface única usada pelo resto do sistema
# ---------------------------------------------------------------------------
def enviar_mensagem(texto, video=None):
    """Envia agora (bloqueia) por todos os canais configurados.
    Retorna (ok, detalhe) - ok=True se pelo menos um canal funcionou."""
    resultados = []
    if telegram_configurado():
        resultados.append(("Telegram",) + enviar_telegram(texto, video))
    if whatsapp_configurado():
        texto_wpp = texto + (f"\n\nVideo salvo em: {video}" if video else "")
        resultados.append(("WhatsApp",) + whatsapp.enviar_mensagem(texto_wpp))
    if not resultados:
        return False, "nenhum canal configurado (Telegram em notificacao.py ou WhatsApp em notificacao_whatsapp.py)"
    ok = any(r[1] for r in resultados)
    return ok, "; ".join(f"{canal}: {detalhe}" for canal, _, detalhe in resultados)


def enviar_em_segundo_plano(texto, video=None):
    threading.Thread(target=enviar_mensagem, args=(texto, video), daemon=True).start()


def nivel_de_atencao(relato_ia):
    """Lê o nível da linha "Atenção: ALTA/MÉDIA/BAIXA" do relato da IA.
    Retorna "ALTA", "MÉDIA", "BAIXA" ou "" (relato sem nível, ex.: IA fora do ar)."""
    achado = re.search(r"aten[çc][ãa]o\s*[:\-]\s*\**\s*(alta|m[ée]dia|baixa)",
                       relato_ia or "", flags=re.IGNORECASE)
    if not achado:
        return ""
    nivel = achado.group(1).upper()
    return "MÉDIA" if nivel.startswith("M") else nivel


def notificar_evento(caminho_video, alertas, relato_ia=""):
    """Avisa no celular sobre um evento.
    - Atenção ALTA: mensagem de alerta + o VÍDEO do evento.
    - MÉDIA, BAIXA ou sem nível: só a mensagem de texto (o vídeo fica salvo no PC)."""
    resumo = "; ".join(alertas) if alertas else "movimento nao identificado"
    nivel = nivel_de_atencao(relato_ia)

    if nivel == "ALTA":
        texto = f"🚨 SafeScreen.AI - ALERTA DE RISCO ALTO\n\nDetector: {resumo}"
        video = caminho_video
    else:
        rotulo = f" (atenção {nivel.lower()})" if nivel else ""
        texto = f"SafeScreen.AI - Evento detectado{rotulo}\n\nDetector: {resumo}"
        video = None
    if relato_ia:
        texto += f"\n\nAnálise da IA:\n{relato_ia}"
    if not video:
        texto += f"\n\nVídeo salvo no computador: {caminho_video}"
    enviar_em_segundo_plano(texto, video)


# ---------------------------------------------------------------------------
# Teste da configuração:  python notificacao.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not telegram_configurado():
        raise SystemExit("Cole o token do @BotFather em TOKEN_TELEGRAM (notificacao.py) e rode de novo.")
    import glob
    videos = sorted(glob.glob(os.path.join("eventos", "evento_*.mp4")), key=os.path.getmtime)
    ultimo = videos[-1] if videos else None
    print("Enviando mensagem de teste para o Telegram"
          + (f" com o vídeo mais recente ({ultimo})..." if ultimo else " (nenhum vídeo em eventos/)..."))
    ok, detalhe = enviar_telegram("✅ SafeScreen.AI conectado! Os alertas vão chegar aqui.\n\n"
                                  "Risco ALTO: mensagem + vídeo.  Outros eventos: só mensagem.",
                                  ultimo)
    print(("OK: " if ok else "FALHOU: ") + detalhe)
