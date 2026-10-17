"""
Avatares circulares para o bar chart race.

- Com foto em assets/fotos/<slug>.jpg|png: recorte quadrado priorizando o rosto
  (terço superior), máscara circular e anel na cor do clube.
- Sem foto: círculo na cor do clube com as iniciais (fallback elegante, sem "buraco").

Tudo é desenhado em 4x e reduzido (supersampling) para bordas suaves.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

TAMANHO_PX = 128
_ESCALA = 4
EXTENSOES = (".jpg", ".jpeg", ".png", ".webp")


def slug(jogador: str, sigla: str) -> str:
    """'Gustavo Gómez', 'PAL' -> 'gustavo_gomez_pal' (nome de arquivo seguro, sem acento)."""
    sem_acento = unicodedata.normalize("NFKD", jogador).encode("ascii", "ignore").decode()
    base = "".join(c if c.isalnum() else "_" for c in sem_acento.lower()).strip("_")
    return f"{base}_{sigla.lower()}"


def _iniciais(jogador: str) -> str:
    partes = [p for p in jogador.split() if p[0].isupper()] or jogador.split()
    return (partes[0][0] + (partes[-1][0] if len(partes) > 1 else "")).upper()


def _fonte_negrito(tamanho: int) -> ImageFont.FreeTypeFont:
    caminho = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    return ImageFont.truetype(caminho, tamanho)


def _recorte_quadrado(img: Image.Image) -> Image.Image:
    """Quadrado centrado na horizontal e puxado para cima: em retrato, o rosto fica no topo."""
    w, h = img.size
    lado = min(w, h)
    esquerda = (w - lado) // 2
    topo = 0 if h > w * 1.15 else (h - lado) // 2
    return img.crop((esquerda, topo, esquerda + lado, topo + lado))


def gerar_avatar(pasta_fotos: Path, jogador: str, sigla: str, cor_clube: str) -> np.ndarray:
    """Devolve uma imagem RGBA (array float 0-1) pronta para o OffsetImage do matplotlib."""
    grande = TAMANHO_PX * _ESCALA
    contorno = int(grande * 0.035)
    anel = int(grande * 0.10)
    tela = Image.new("RGBA", (grande, grande), (0, 0, 0, 0))
    desenho = ImageDraw.Draw(tela)
    desenho.ellipse((0, 0, grande - 1, grande - 1), fill="#FFFFFF")          # contorno branco: destaca da barra
    desenho.ellipse((contorno, contorno, grande - contorno, grande - contorno), fill=cor_clube)  # anel do clube

    interno = grande - 2 * anel
    foto = next((pasta_fotos / f"{slug(jogador, sigla)}{ext}" for ext in EXTENSOES
                 if (pasta_fotos / f"{slug(jogador, sigla)}{ext}").exists()), None)
    if foto is not None:
        retrato = _recorte_quadrado(Image.open(foto).convert("RGB")).resize((interno, interno), Image.LANCZOS)
        mascara = Image.new("L", (interno, interno), 0)
        ImageDraw.Draw(mascara).ellipse((0, 0, interno - 1, interno - 1), fill=255)
        tela.paste(retrato, (anel, anel), mascara)
    else:
        desenho.ellipse((anel, anel, grande - anel, grande - anel), fill="#141C2B")
        texto = _iniciais(jogador)
        fonte = _fonte_negrito(int(grande * 0.40))
        desenho.text((grande / 2, grande / 2), texto, font=fonte, fill="#E8EEF6", anchor="mm")

    pequeno = tela.resize((TAMANHO_PX, TAMANHO_PX), Image.LANCZOS)
    return np.asarray(pequeno).astype(float) / 255
