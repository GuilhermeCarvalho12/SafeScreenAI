"""
chat_markdown.py - transforma o texto em Markdown das respostas da IA em trechos
com estilo, para a interface desenhar bonito (negrito, listas, títulos, tabelas,
código e caminhos de vídeo clicáveis) em vez de mostrar ** e ### na tela.

Não depende do Tkinter: analisar_markdown(texto) devolve uma lista de
(trecho, tags, caminho_video) e a interface só insere cada trecho com suas tags.
"""
import re

# Caminho de vídeo de evento, ex: eventos\evento_20261109_155122.mp4
PADRAO_VIDEO = re.compile(r"((?:[A-Za-z]:[\\/])?(?:[^\s\\/`*]+[\\/])*?eventos[\\/]evento_[\w-]+\.mp4)")

PADRAO_INLINE = re.compile(
    r"\*\*(.+?)\*\*"            # **negrito**
    r"|__(.+?)__"                # __negrito__
    r"|`([^`]+)`"                # `código`
    r"|(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])"   # *itálico*
    r"|\[([^\]]+)\]\(([^)]+)\)"  # [texto](link)
)
PADRAO_ESCAPE = re.compile(r"\\([*_`\[\]#|>\-.!()])")

LARGURA_MAX_CELULA = 30


def _videos(texto, tags):
    """Separa caminhos de vídeo do resto do texto (viram links clicáveis)."""
    partes = []
    pos = 0
    for achado in PADRAO_VIDEO.finditer(texto):
        if achado.start() > pos:
            partes.append((texto[pos:achado.start()], tags, None))
        caminho = achado.group(1).strip()
        partes.append((caminho, tags + ("link",), caminho))
        pos = achado.end()
    if pos < len(texto):
        partes.append((texto[pos:], tags, None))
    return partes


def _inline(texto, tags):
    partes = []
    pos = 0
    for achado in PADRAO_INLINE.finditer(texto):
        if achado.start() > pos:
            partes += _videos(texto[pos:achado.start()], tags)
        negrito, negrito2, codigo, italico, link_texto, _url = achado.groups()
        if negrito or negrito2:
            partes += _videos(negrito or negrito2, tags + ("bold",))
        elif codigo:
            partes += _videos(codigo, tags + ("code",))
        elif italico:
            partes += _videos(italico, tags + ("italic",))
        elif link_texto:
            partes += _videos(link_texto, tags + ("bold",))
        pos = achado.end()
    if pos < len(texto):
        partes += _videos(texto[pos:], tags)
    return partes


def _limpar_celula(texto):
    texto = PADRAO_INLINE.sub(lambda m: next(g for g in m.groups() if g), texto).strip()
    if len(texto) > LARGURA_MAX_CELULA:
        texto = texto[:LARGURA_MAX_CELULA - 1] + "…"
    return texto


def _tabela(linhas):
    """Desenha a tabela em colunas alinhadas (fonte monoespaçada)."""
    linhas_celulas = []
    for linha in linhas:
        if re.fullmatch(r"\|?[\s:|-]+\|?", linha.strip()):
            continue  # linha separadora |---|---|
        celulas = [_limpar_celula(c) for c in linha.strip().strip("|").split("|")]
        linhas_celulas.append(celulas)
    if not linhas_celulas:
        return []
    colunas = max(len(l) for l in linhas_celulas)
    for l in linhas_celulas:
        l += [""] * (colunas - len(l))
    larguras = [max(len(l[i]) for l in linhas_celulas) for i in range(colunas)]

    partes = []
    for n, celulas in enumerate(linhas_celulas):
        texto = "   ".join(c.ljust(larguras[i]) for i, c in enumerate(celulas)).rstrip()
        tags = ("table", "table_head") if n == 0 else ("table",)
        partes.append((" " + texto + "\n", tags, None))
        if n == 0:
            partes.append((" " + "─" * min(sum(larguras) + 3 * (colunas - 1), 90) + "\n",
                           ("table", "hr"), None))
    return partes


def analisar_markdown(texto, tags_base=()):
    """Retorna lista de (trecho, tags, caminho_video_ou_None)."""
    texto = PADRAO_ESCAPE.sub(r"\1", texto.replace("\r", "")).strip("\n")
    linhas = texto.split("\n")
    base = tuple(tags_base)
    partes = []
    em_codigo = False
    linha_em_branco = False
    i = 0

    while i < len(linhas):
        linha = linhas[i]
        limpa = linha.strip()

        if limpa.startswith("```"):
            em_codigo = not em_codigo
            i += 1
            continue
        if em_codigo:
            partes.append((linha + "\n", base + ("codeblock",), None))
            i += 1
            continue

        # tabela: linhas seguidas começando com |
        if limpa.startswith("|") and limpa.count("|") >= 2:
            bloco = []
            while i < len(linhas) and linhas[i].strip().startswith("|"):
                bloco.append(linhas[i])
                i += 1
            partes += [(t, base + tg, v) for t, tg, v in _tabela(bloco)]
            linha_em_branco = False
            continue

        if not limpa:
            if not linha_em_branco and partes:
                partes.append(("\n", base + ("gap",), None))
            linha_em_branco = True
            i += 1
            continue
        linha_em_branco = False

        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", limpa):
            i += 1
            continue  # separadores ficam de fora: só poluem o chat

        titulo = re.match(r"^(#{1,6})\s+(.*)", limpa)
        lista = re.match(r"^(\s*)([-*•+])\s+(.*)", linha)
        numerada = re.match(r"^(\s*)(\d+)[.)]\s+(.*)", linha)
        citacao = re.match(r"^>\s?(.*)", limpa)

        if titulo:
            nivel = min(len(titulo.group(1)), 3)
            partes += _inline(titulo.group(2).strip("*# "), base + (f"h{nivel}",))
        elif lista:
            nivel = "bullet1" if len(lista.group(1)) >= 2 else "bullet0"
            partes.append(("•  ", base + (nivel, "bullet_mark"), None))
            partes += _inline(lista.group(3), base + (nivel,))
        elif numerada:
            nivel = "bullet1" if len(numerada.group(1)) >= 2 else "bullet0"
            partes.append((f"{numerada.group(2)}.  ", base + (nivel, "bullet_mark"), None))
            partes += _inline(numerada.group(3), base + (nivel,))
        elif citacao:
            partes += _inline(citacao.group(1), base + ("quote",))
        else:
            partes += _inline(linha.strip(), base + ("p",))

        partes.append(("\n", base, None))
        i += 1

    while partes and partes[-1][0] == "\n":
        partes.pop()
    return partes
