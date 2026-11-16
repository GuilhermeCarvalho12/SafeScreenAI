import csv
import os
import threading
from datetime import datetime

PASTA_EVENTOS = "eventos"
CAMINHO_REGISTRO = os.path.join(PASTA_EVENTOS, "registro_eventos.csv")
# Resumo minuto a minuto do que a IA viu (mesmo sem alerta) - é o que permite ao
# chatbot responder "como foi a última hora?" com dados reais.
CAMINHO_ATIVIDADE = os.path.join(PASTA_EVENTOS, "atividade.csv")
FORMATO_DATA = "%Y-%m-%d %H:%M:%S"

_trava = threading.Lock()  # o registro é escrito por threads diferentes


def _garantir_arquivo_criado(caminho, cabecalho):
    os.makedirs(PASTA_EVENTOS, exist_ok=True)
    if not os.path.exists(caminho):
        with open(caminho, mode="w", newline="", encoding="utf-8") as arquivo:
            csv.writer(arquivo).writerow(cabecalho)


def registrar_evento(caminho_video, alertas, relato_ia=""):
    """
    Adiciona uma linha no CSV de registro toda vez que um evento termina de ser gravado.
    alertas: lista de strings com os motivos do alerta (ex: "pessoa #2 parado ha 12s").
    relato_ia: texto escrito pela IA generativa descrevendo o evento.
    """
    _garantir_arquivo_criado(CAMINHO_REGISTRO,
                             ["data_hora", "tipos_de_alerta", "arquivo_video", "relato_ia"])

    data_hora = datetime.now().strftime(FORMATO_DATA)
    tipos = "; ".join(alertas) if alertas else "movimento nao identificado"

    with _trava, open(CAMINHO_REGISTRO, mode="a", newline="", encoding="utf-8") as arquivo:
        csv.writer(arquivo).writerow([data_hora, tipos, caminho_video, relato_ia])

    print(f"[SafeScreen.AI] Evento registrado: {data_hora} - {tipos}")


def registrar_atividade(minuto, frames_com_deteccao, objetos):
    """objetos: dict {nome: maximo_simultaneo}, ex: {"pessoa": 2, "gato": 1}."""
    _garantir_arquivo_criado(CAMINHO_ATIVIDADE, ["minuto", "frames_com_deteccao", "objetos"])
    resumo = "; ".join(f"{nome} (max {qtd})" for nome, qtd in sorted(objetos.items()))
    with _trava, open(CAMINHO_ATIVIDADE, mode="a", newline="", encoding="utf-8") as arquivo:
        csv.writer(arquivo).writerow([minuto, frames_com_deteccao, resumo])


def _linhas(caminho):
    if not os.path.exists(caminho):
        return []
    with _trava, open(caminho, "r", encoding="utf-8") as arquivo:
        linhas = list(csv.reader(arquivo))
    return [l for l in linhas[1:] if l]  # pula o cabeçalho


def _data(texto, formato):
    try:
        return datetime.strptime(texto.strip(), formato)
    except ValueError:
        return None


def ler_eventos(desde=None):
    """Eventos registrados a partir de `desde` (datetime), do mais antigo ao mais novo.
    Funciona também com o CSV antigo, de 3 colunas (sem relato_ia)."""
    eventos = []
    for linha in _linhas(CAMINHO_REGISTRO):
        quando = _data(linha[0], FORMATO_DATA)
        if quando is None or (desde and quando < desde):
            continue
        eventos.append({
            "data_hora": linha[0],
            "alertas": linha[1] if len(linha) > 1 else "",
            "video": linha[2] if len(linha) > 2 else "",
            "relato_ia": linha[3] if len(linha) > 3 else "",
        })
    return eventos


def ler_atividade(desde=None, agrupar_por_hora=False):
    """Resumo de atividade a partir de `desde`. Com agrupar_por_hora=True junta os
    minutos de cada hora (para períodos longos não ficarem enormes)."""
    resultado = []
    for linha in _linhas(CAMINHO_ATIVIDADE):
        quando = _data(linha[0], "%Y-%m-%d %H:%M")
        if quando is None or (desde and quando < desde):
            continue
        resultado.append({"minuto": linha[0], "frames": int(linha[1] or 0),
                          "objetos": linha[2] if len(linha) > 2 else ""})

    if not agrupar_por_hora:
        return resultado

    horas = {}
    for item in resultado:
        hora = item["minuto"][:13] + "h"
        grupo = horas.setdefault(hora, {"hora": hora, "minutos_com_atividade": 0, "objetos": set()})
        grupo["minutos_com_atividade"] += 1
        for obj in item["objetos"].split("; "):
            if obj:
                grupo["objetos"].add(obj.split(" (")[0])
    return [{**g, "objetos": ", ".join(sorted(g["objetos"]))} for g in horas.values()]
