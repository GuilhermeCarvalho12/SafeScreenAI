"""
tema.py - tema visual do SafeScreen.AI (mesmas cores do slide).

Uso (2 linhas no interface.py):

    from tema import aplicar_tema
    ...
    root = tk.Tk()
    aplicar_tema(root)      # <- logo depois de criar a janela, ANTES de criar os widgets

Nao mexe em camera, thread, fila, YOLO nem FFT. So estilo.
Para desfazer: apague as 2 linhas acima (ou apague este arquivo e a linha do import).
"""
import tkinter as tk
from tkinter import ttk

# ---- Paleta do slide ----
BG = "#12161c"          # fundo da janela
PAINEL = "#1b2129"      # cards / paineis
PAINEL_2 = "#232b35"    # hover / linhas alternadas
BORDA = "#2c3541"
TEXTO = "#f5f6f2"
TEXTO_SUAVE = "#9aa3a0"
VERDE = "#2f8f6f"       # cor principal (.AI)
VERDE_CLARO = "#4fb894"
ALERTA = "#e5484d"      # vermelho para alertas
AVISO = "#e9a23b"       # amarelo para avisos

FONTE_TITULO = ("Consolas", 16, "bold")
FONTE_MONO = ("Consolas", 10)
FONTE = ("Segoe UI", 10)
FONTE_NEGRITO = ("Segoe UI", 10, "bold")

print("TEMA APLICADO")
def aplicar_tema(root):
    """Aplica o tema escuro verde em toda a janela. Chame antes de criar os widgets."""
    root.configure(bg=BG)

    # --- widgets classicos do tk (Label, Frame, Button, Canvas, Text...) ---
    # option_add vale para os widgets criados DEPOIS desta chamada e
    # nao sobrescreve cores que o seu codigo definir explicitamente.
    root.option_add("*Background", BG)
    root.option_add("*Foreground", TEXTO)
    root.option_add("*Font", FONTE)
    root.option_add("*Button.Background", VERDE)
    root.option_add("*Button.Foreground", "#ffffff")
    root.option_add("*Button.ActiveBackground", VERDE_CLARO)
    root.option_add("*Button.ActiveForeground", "#ffffff")
    root.option_add("*Button.Relief", "flat")
    root.option_add("*Button.BorderWidth", 0)
    root.option_add("*Button.Cursor", "hand2")
    root.option_add("*Button.Padx", 14)
    root.option_add("*Button.Pady", 6)
    root.option_add("*Entry.Background", PAINEL)
    root.option_add("*Entry.Foreground", TEXTO)
    root.option_add("*Entry.InsertBackground", TEXTO)
    root.option_add("*Text.Background", PAINEL)
    root.option_add("*Text.Foreground", TEXTO)
    root.option_add("*Canvas.Background", "#000000")
    root.option_add("*Canvas.HighlightThickness", 0)
    root.option_add("*Listbox.Background", PAINEL)
    root.option_add("*Listbox.Foreground", TEXTO)
    root.option_add("*Listbox.SelectBackground", VERDE)

    # --- widgets ttk (Treeview, Notebook, Scrollbar, TButton...) ---
    s = ttk.Style(root)
    try:
        s.theme_use("clam")
    except tk.TclError:
        pass

    s.configure(".", background=BG, foreground=TEXTO, font=FONTE,
                bordercolor=BORDA, lightcolor=BG, darkcolor=BG,
                troughcolor=PAINEL, focuscolor=BG)

    s.configure("TFrame", background=BG)
    s.configure("Card.TFrame", background=PAINEL)
    s.configure("TLabel", background=BG, foreground=TEXTO)
    s.configure("Titulo.TLabel", font=FONTE_TITULO, foreground=TEXTO)
    s.configure("Suave.TLabel", foreground=TEXTO_SUAVE)
    s.configure("TLabelframe", background=BG, bordercolor=BORDA, relief="solid")
    s.configure("TLabelframe.Label", background=BG, foreground=VERDE_CLARO,
                font=FONTE_NEGRITO)

    s.configure("TButton", background=VERDE, foreground="#ffffff",
                borderwidth=0, padding=(14, 7), font=FONTE_NEGRITO)
    s.map("TButton",
          background=[("active", VERDE_CLARO), ("disabled", PAINEL_2)],
          foreground=[("disabled", TEXTO_SUAVE)])
    s.configure("Perigo.TButton", background=ALERTA)
    s.map("Perigo.TButton", background=[("active", "#f06a6e")])

    s.configure("TEntry", fieldbackground=PAINEL, foreground=TEXTO,
                insertcolor=TEXTO, bordercolor=BORDA)
    s.configure("TCombobox", fieldbackground=PAINEL, background=PAINEL,
                foreground=TEXTO, arrowcolor=VERDE_CLARO)
    s.configure("TCheckbutton", background=BG, foreground=TEXTO)
    s.map("TCheckbutton", background=[("active", BG)])

    # tabela de eventos
    s.configure("Treeview", background=PAINEL, fieldbackground=PAINEL,
                foreground=TEXTO, rowheight=28, borderwidth=0, font=FONTE)
    s.map("Treeview",
          background=[("selected", VERDE)],
          foreground=[("selected", "#ffffff")])
    s.configure("Treeview.Heading", background=BG, foreground=VERDE_CLARO,
                font=FONTE_NEGRITO, relief="flat", padding=(8, 8))
    s.map("Treeview.Heading", background=[("active", PAINEL_2)])

    s.configure("TNotebook", background=BG, borderwidth=0)
    s.configure("TNotebook.Tab", background=PAINEL, foreground=TEXTO_SUAVE,
                padding=(16, 8), font=FONTE_NEGRITO)
    s.map("TNotebook.Tab",
          background=[("selected", VERDE)],
          foreground=[("selected", "#ffffff")])

    s.configure("Vertical.TScrollbar", background=PAINEL_2, arrowcolor=TEXTO_SUAVE,
                bordercolor=BG)
    s.configure("Horizontal.TScrollbar", background=PAINEL_2, arrowcolor=TEXTO_SUAVE,
                bordercolor=BG)
    s.configure("TProgressbar", background=VERDE, troughcolor=PAINEL)
    return s


def estilizar_tabela(tree):
    """Opcional: linhas alternadas e alertas coloridos em um ttk.Treeview.

    Depois de criar a tabela:   estilizar_tabela(self.tabela)
    Ao inserir uma linha:       tree.insert("", "end", values=..., tags=("alerta",))
    (tags aceitas: "par", "impar", "alerta", "aviso")
    """
    tree.tag_configure("par", background=PAINEL)
    tree.tag_configure("impar", background=PAINEL_2)
    tree.tag_configure("alerta", foreground=ALERTA)
    tree.tag_configure("aviso", foreground=AVISO)
