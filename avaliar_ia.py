"""
avaliar_ia.py - mede o quanto a IA de visão acerta, usando os vídeos de eventos.

Gera números reais para a "visão crítica" do relatório (taxa de acerto, falsos
alarmes, tempo de resposta) e compara o que o YOLO viu com o que o Gemini viu.

Como usar (na pasta do main.py, com a chave do Gemini configurada):

  1. Grave cenas de teste com o SafeScreen (pelo chat: "grave 15 segundos"), por exemplo:
       - cena vazia                    - pessoa normal passando
       - pessoa de máscara/capuz       - pessoa com arma de brinquedo na mão
       - pessoa com celular na mão     - pessoa pegando um objeto da mesa
  2. Rode:   python avaliar_ia.py            (analisa os 10 vídeos mais recentes)
             python avaliar_ia.py 20         (analisa os 20 mais recentes)
  3. Abra eventos/avaliacao_ia.csv no Excel e preencha a coluna "gabarito"
     (o que REALMENTE aconteceu) e "acertou" (sim/nao).
  4. Rode   python avaliar_ia.py --resumo    para calcular a taxa de acerto.
"""
import csv
import os
import sys
import time

from ia_generativa import (analisar_video_arquivo, extrair_campo, ia_disponivel,
                           motivo_indisponivel)
from registro_eventos import ler_eventos

ARQUIVO_SAIDA = os.path.join("eventos", "avaliacao_ia.csv")
PERGUNTA_PADRAO = ("Descreva o que acontece no vídeo. Há alguma arma aparente (de fogo ou "
                   "branca)? Alguém está com o rosto coberto? Algum objeto foi levado?")
COLUNAS = ["data_hora", "video", "alertas_yolo", "atencao_ia", "riscos_ia", "resposta_ia",
           "tempo_s", "gabarito", "acertou"]


def avaliar(quantidade):
    if not ia_disponivel():
        raise SystemExit(f"IA de visão indisponível: {motivo_indisponivel()}")

    eventos = [e for e in ler_eventos() if e.get("video") and os.path.exists(e["video"])]
    eventos = eventos[-quantidade:]
    if not eventos:
        raise SystemExit("Nenhum vídeo de evento encontrado em eventos/.")

    linhas = []
    for n, evento in enumerate(eventos, 1):
        print(f"[{n}/{len(eventos)}] {evento['data_hora']}  ({evento['video']})")
        inicio = time.time()
        resposta = analisar_video_arquivo(evento["video"], PERGUNTA_PADRAO, evento["data_hora"]) or ""
        tempo = round(time.time() - inicio, 1)
        linha = {
            "data_hora": evento["data_hora"], "video": evento["video"],
            "alertas_yolo": evento["alertas"],
            "atencao_ia": extrair_campo(resposta, "Aten[çc][ãa]o"),
            "riscos_ia": extrair_campo(resposta, "Riscos"),
            "resposta_ia": " ".join(resposta.split()), "tempo_s": tempo,
            "gabarito": "", "acertou": "",
        }
        print(f"     Atenção: {linha['atencao_ia'] or '?'}  |  Riscos: {linha['riscos_ia'] or '?'}  "
              f"|  {tempo}s\n")
        linhas.append(linha)

    os.makedirs("eventos", exist_ok=True)
    # utf-8-sig: o Excel abre com acentos corretos
    with open(ARQUIVO_SAIDA, "w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS, delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas)
    print(f"Resultados salvos em {ARQUIVO_SAIDA}. Preencha 'gabarito' e 'acertou' e rode "
          f"'python avaliar_ia.py --resumo'.")


def resumo():
    if not os.path.exists(ARQUIVO_SAIDA):
        raise SystemExit("Rode primeiro: python avaliar_ia.py")
    with open(ARQUIVO_SAIDA, encoding="utf-8-sig") as arquivo:
        linhas = list(csv.DictReader(arquivo, delimiter=";"))

    avaliadas = [l for l in linhas if l["acertou"].strip()]
    acertos = sum(l["acertou"].strip().lower() in ("sim", "s", "1", "x") for l in avaliadas)
    tempos = [float(l["tempo_s"].replace(",", ".")) for l in linhas if l["tempo_s"]]
    altas = sum("ALTA" in l["atencao_ia"].upper() for l in linhas)
    armas_yolo = sum(any(p in l["alertas_yolo"].lower() for p in ("arma", "gun", "knife", "faca"))
                     for l in linhas)
    armas_ia = sum("arma" in l["riscos_ia"].lower() for l in linhas)

    print(f"Vídeos analisados:            {len(linhas)}")
    if avaliadas:
        print(f"Taxa de acerto da IA:         {acertos}/{len(avaliadas)} "
              f"({100 * acertos / len(avaliadas):.0f}%)")
    else:
        print("Taxa de acerto da IA:         (preencha a coluna 'acertou' no CSV)")
    if tempos:
        print(f"Tempo médio de análise:       {sum(tempos) / len(tempos):.1f} s")
    print(f"Eventos com atenção ALTA:     {altas}")
    print(f"Armas citadas pelo YOLO:      {armas_yolo}   (o YOLO não tem classe de arma)")
    print(f"Armas aparentes vistas pela IA: {armas_ia}")


if __name__ == "__main__":
    if "--resumo" in sys.argv:
        resumo()
    else:
        numeros = [a for a in sys.argv[1:] if a.isdigit()]
        avaliar(int(numeros[0]) if numeros else 10)
