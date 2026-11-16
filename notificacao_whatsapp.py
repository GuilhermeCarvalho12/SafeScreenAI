import threading

import requests

# --- Configuração (veja o passo a passo abaixo) ---
NUMERO_WHATSAPP = ""   # seu número com DDI+DDD, só dígitos
API_KEY_CALLMEBOT = "XXXXX"
ENVIO_ATIVO = False  # mude para True depois de configurar

"""
Como configurar o envio de WhatsApp (gratuito, via CallMeBot):

1. Salve o número +34 644 59 71 92 nos seus contatos do celular.
2. Pelo WhatsApp, mande para esse número a mensagem exata:
   "I allow callmebot to send me messages"
3. Você vai receber de volta um "apikey" (um número). Cole-o em API_KEY_CALLMEBOT acima.
4. Troque NUMERO_WHATSAPP pelo seu número (o que vai RECEBER os alertas).
5. Mude ENVIO_ATIVO para True.

Limitação importante: o CallMeBot gratuito só envia TEXTO, não envia o arquivo de
vídeo em si. A mensagem inclui o caminho do vídeo salvo no seu computador. Enviar o
vídeo automaticamente por WhatsApp exigiria uma API paga (ex: Twilio) ou um serviço
próprio de bot de WhatsApp (ex: Evolution API) - mais complexo de configurar.
"""

LIMITE_CARACTERES = 900  # mensagens muito longas falham na URL do CallMeBot


def notificar_evento(caminho_video, alertas, relato_ia=""):
    """
    Dispara o envio da notificação em uma thread separada, para não atrasar
    a gravação/captura enquanto espera a resposta da internet.
    """
    resumo = "; ".join(alertas) if alertas else "movimento nao identificado"
    texto = f"SafeScreen.AI - Alerta: {resumo}."
    if relato_ia:
        texto += f"\n\nRelato da IA: {relato_ia}"
    texto += f"\n\nVideo salvo em: {caminho_video}"
    enviar_em_segundo_plano(texto)


def enviar_em_segundo_plano(texto):
    if not ENVIO_ATIVO:
        print("[SafeScreen.AI] Envio de WhatsApp desativado (configure notificacao_whatsapp.py).")
        return
    threading.Thread(target=enviar_mensagem, args=(texto,), daemon=True).start()


def enviar_mensagem(texto):
    """Envia na hora (bloqueia). Retorna (ok: bool, detalhe: str) - usado pelo chatbot."""
    if not ENVIO_ATIVO:
        return False, "envio de WhatsApp desativado em notificacao_whatsapp.py"
    if len(texto) > LIMITE_CARACTERES:
        texto = texto[:LIMITE_CARACTERES - 3] + "..."

    try:
        resposta = requests.get(
            "https://api.callmebot.com/whatsapp.php",
            params={"phone": NUMERO_WHATSAPP, "text": texto, "apikey": API_KEY_CALLMEBOT},
            timeout=15,
        )
        if resposta.status_code == 200:
            print("[SafeScreen.AI] Notificacao enviada por WhatsApp.")
            return True, "mensagem enviada"
        print(f"[SafeScreen.AI] Falha ao enviar WhatsApp (codigo {resposta.status_code}).")
        return False, f"CallMeBot respondeu com código {resposta.status_code}"
    except requests.RequestException as erro:
        print(f"[SafeScreen.AI] Erro ao enviar WhatsApp: {erro}")
        return False, f"erro de conexão: {erro}"
