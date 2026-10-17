"""
Bandeiras nacionais desenhadas por código (PIL), sem baixar imagem nenhuma.

Por que procedural?
  - Bandeiras nacionais são domínio público, mas os ARQUIVOS (SVG/PNG) que circulam
    têm autores e licenças variadas. Desenhando aqui, a origem é 100% nossa.
  - Zero dependência de rede: o render roda offline e é reprodutível.
  - No vídeo elas aparecem translúcidas e recortadas pelo contorno do país; uma versão
    simplificada (sem brasões detalhados) é a escolha certa para leitura em celular.

Uso:
    from bandeiras import gerar_bandeira
    rgba = gerar_bandeira("Brasil")      # np.ndarray (H, W, 4) float 0-1
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

_ALTURA = 600
_ESCALA = 2   # supersampling: desenha 2x maior e reduz (bordas suaves, sem serrilhado)


def _sol(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float, cor: str, raios: int = 16) -> None:
    """Sol de Maio (Argentina/Uruguai), simplificado: disco + raios triangulares."""
    for i in range(raios):
        a = 2 * math.pi * i / raios
        a1, a2 = a - math.pi / raios * 0.45, a + math.pi / raios * 0.45
        ponta = (cx + math.cos(a) * r * 1.9, cy + math.sin(a) * r * 1.9)
        d.polygon([(cx + math.cos(a1) * r, cy + math.sin(a1) * r), ponta,
                   (cx + math.cos(a2) * r, cy + math.sin(a2) * r)], fill=cor)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=cor)


def _estrela(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float, cor: str) -> None:
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.40
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    d.polygon(pts, fill=cor)


def _faixas_h(d, w, h, cores_e_pesos):
    total, y = sum(p for _, p in cores_e_pesos), 0.0
    for cor, peso in cores_e_pesos:
        alt = h * peso / total
        d.rectangle((0, y, w, y + alt + 1), fill=cor)
        y += alt


def _brasil(w, h, d, img):
    d.rectangle((0, 0, w, h), fill="#009C3B")
    m = h * 0.085
    d.polygon([(m * 1.2, h / 2), (w / 2, m), (w - m * 1.2, h / 2), (w / 2, h - m)], fill="#FFDF00")
    r = h * 0.25
    cx, cy = w / 2, h / 2
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#002776")
    # faixa branca curva ("Ordem e Progresso" sem texto: ilegível nesse tamanho)
    # A faixa é desenhada numa camada à parte e recortada pelo círculo (máscara),
    # o mesmo princípio do clip_path que usamos para encaixar a bandeira no país.
    faixa = Image.new("L", (int(w), int(h)), 0)
    rf = r * 2.0
    ImageDraw.Draw(faixa).ellipse((cx - rf - r * 0.45, cy - r * 0.35, cx + rf - r * 0.45, cy - r * 0.35 + rf * 2),
                                  outline=255, width=int(r * 0.17))
    disco = Image.new("L", (int(w), int(h)), 0)
    ImageDraw.Draw(disco).ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    mascara = Image.fromarray(np.minimum(np.asarray(faixa), np.asarray(disco)))
    img.paste((255, 255, 255, 255), (0, 0), mascara)
    rng = np.random.default_rng(7)            # estrelas fixas (seed) = render reprodutível
    for _ in range(18):
        ang, dist = rng.uniform(0.15 * math.pi, 0.95 * math.pi), rng.uniform(0.2, 0.85) * r
        x, y = cx + math.cos(ang) * dist, cy + math.sin(ang) * dist * 0.9
        rr = r * rng.uniform(0.025, 0.05)
        d.ellipse((x - rr, y - rr, x + rr, y + rr), fill="#FFFFFF")


def _argentina(w, h, d, img):
    _faixas_h(d, w, h, [("#74ACDF", 1), ("#FFFFFF", 1), ("#74ACDF", 1)])
    _sol(d, w / 2, h / 2, h * 0.075, "#F6B40E")


def _uruguai(w, h, d, img):
    _faixas_h(d, w, h, [("#FFFFFF" if i % 2 == 0 else "#0038A8", 1) for i in range(9)])
    lado = h * 5 / 9
    d.rectangle((0, 0, lado, lado), fill="#FFFFFF")
    _sol(d, lado / 2, lado / 2, lado * 0.17, "#FCD116")


def _paraguai(w, h, d, img):
    _faixas_h(d, w, h, [("#D52B1E", 1), ("#FFFFFF", 1), ("#0038A8", 1)])
    r = h * 0.13
    d.ellipse((w / 2 - r, h / 2 - r, w / 2 + r, h / 2 + r), outline="#1A1A1A", width=int(r * 0.12))
    _estrela(d, w / 2, h / 2, r * 0.45, "#FCD116")


def _colombia(w, h, d, img):
    _faixas_h(d, w, h, [("#FCD116", 2), ("#003893", 1), ("#CE1126", 1)])


def _equador(w, h, d, img):
    _faixas_h(d, w, h, [("#FFDD00", 2), ("#034EA2", 1), ("#ED1C24", 1)])
    r = h * 0.14     # brasão simplificado: medalhão
    cx, cy = w / 2, h * 0.47
    d.ellipse((cx - r, cy - r * 1.15, cx + r, cy + r * 1.15), fill="#7BB8E8", outline="#8B5A2B",
              width=int(r * 0.14))
    d.polygon([(cx - r * 0.55, cy + r * 0.5), (cx, cy - r * 0.3), (cx + r * 0.55, cy + r * 0.5)],
              fill="#2E7D32")


def _chile(w, h, d, img):
    d.rectangle((0, 0, w, h / 2), fill="#FFFFFF")
    d.rectangle((0, h / 2, w, h), fill="#D52B1E")
    d.rectangle((0, 0, h / 2, h / 2), fill="#0039A6")
    _estrela(d, h / 4, h / 4, h * 0.11, "#FFFFFF")


# proporção oficial (largura/altura) de cada bandeira
_DESENHOS = {
    "Brasil": (10 / 7, _brasil),
    "Argentina": (8 / 5, _argentina),
    "Uruguai": (3 / 2, _uruguai),
    "Paraguai": (5 / 3, _paraguai),
    "Colômbia": (3 / 2, _colombia),
    "Equador": (3 / 2, _equador),
    "Chile": (3 / 2, _chile),
}


def proporcao(pais: str) -> float:
    return _DESENHOS[pais][0]


def gerar_bandeira(pais: str, altura: int = _ALTURA) -> np.ndarray:
    """Devolve a bandeira como array RGBA float (0-1), pronto para o imshow.

    `altura` em pixels: gere no tamanho em que ela vai APARECER. Pixel a mais na fonte
    só custa tempo de reamostragem a cada quadro, sem ganho visual.
    """
    if pais not in _DESENHOS:
        raise KeyError(f"Bandeira não implementada: {pais}")
    razao, desenhar = _DESENHOS[pais]
    h = max(60, altura) * _ESCALA
    w = round(h * razao)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    desenhar(w, h, ImageDraw.Draw(img), img)
    img = img.resize((w // _ESCALA, h // _ESCALA), Image.LANCZOS)
    return np.asarray(img).astype(float) / 255


if __name__ == "__main__":       # gera uma folha de conferência: python bandeiras.py
    folha = Image.new("RGB", (7 * 330, 230), "#0B1220")
    for i, p in enumerate(_DESENHOS):
        b = Image.fromarray((gerar_bandeira(p) * 255).astype(np.uint8)).convert("RGB")
        b.thumbnail((300, 200))
        folha.paste(b, (15 + i * 330, 15))
    folha.save("bandeiras_revisao.png")
    print("bandeiras_revisao.png gerado")
