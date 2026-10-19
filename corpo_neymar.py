"""Quadro do vídeo "O corpo do Neymar": foto com marcadores + contador + placar por clube.

Cada quadro é uma função pura do estado: data_corte -> imagem.
A animação final só chama `desenhar_quadro` para uma sequência de estados.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

LARGURA, ALTURA = 1080, 1920
ASSETS = Path("assets")
FOTO = ASSETS / "neymar_recorte.png"
FONTE_IMPACTO = str(ASSETS / "fontes/Anton-Regular.ttf")
FONTE_TEXTO = str(ASSETS / "fontes/BarlowCondensed-Bold.ttf")
FONTE_TEXTO_LEVE = str(ASSETS / "fontes/BarlowCondensed-SemiBold.ttf")

FUNDO = (12, 14, 20)
BRANCO = (245, 245, 245)
BRANCO_PURO = (255, 255, 255)
CINZA = (150, 156, 168)
VERMELHO = (229, 37, 52)

# Coordenadas na foto recortada (932x2507). Foto de frente: o lado DIREITO dele fica à ESQUERDA da tela.
# "não informado" fica no eixo do corpo, com marcador vazado (não inventamos lado).
REGIAO_XY: dict[tuple[str, str], tuple[int, int]] = {
    ("cabeça", "direito"): (360, 205),
    ("costas", "centro"): (440, 930),
    ("costela", "não informado"): (560, 1080),
    ("virilha", "direito"): (370, 1590), ("virilha", "esquerdo"): (560, 1545),
    ("virilha", "não informado"): (440, 1535),
    ("coxa", "direito"): (320, 1760), ("coxa", "esquerdo"): (600, 1760),
    ("coxa", "não informado"): (460, 1740),
    ("posterior", "esquerdo"): (665, 1835),
    ("joelho", "esquerdo"): (600, 1965), ("joelho", "não informado"): (485, 1960),
    ("panturrilha", "não informado"): (480, 2120),
    ("tornozelo", "direito"): (390, 2270), ("tornozelo", "esquerdo"): (600, 2280),
    ("tornozelo", "não informado"): (495, 2230),
    ("pé", "direito"): (330, 2450), ("pé", "esquerdo"): (640, 2460), ("pé", "não informado"): (490, 2470),
}

CLUBES = {  # ordem de exibição = ordem da carreira
    "Santos": {"cor": (235, 235, 235), "escudo": "santos.png"},
    "Barcelona": {"cor": (165, 0, 68), "escudo": "barcelona.png"},
    "PSG": {"cor": (0, 95, 170), "escudo": "psg.png"},
    "Al-Hilal": {"cor": (31, 79, 191), "escudo": "al_hilal.png"},
}

ROTULO_REGIAO = {"cabeça": "olho", "virilha": "adutor", "posterior": "posterior da coxa"}
REGIOES_FEMININAS = {"coxa", "panturrilha", "costela", "posterior"}  # "posterior da coxa" concorda com coxa


def rotulo_local(regiao: str, lado: str) -> str:
    """'coxa' + 'direito' -> 'coxa direita'. O lado concorda em gênero com a região."""
    nome = ROTULO_REGIAO.get(regiao, regiao)
    if lado in ("não informado", "centro"):
        return nome
    if regiao in REGIOES_FEMININAS:
        lado = lado[:-1] + "a"
    return f"{nome} {lado}"


# ZONA SEGURA (Shorts + TikTok, tela 1080x1920): o app cobre o topo (~150 px, abas),
# a base (~420 px, título/legenda/música) e a coluna direita (x > ~900, botões, de y≈800 para baixo).
# Tudo que é dado fica em x 50..900 e y 150..1500.
SEGURA_TOPO, SEGURA_BASE, SEGURA_DIREITA = 150, 1500, 900

CANAL = "ARQUIVOLOGIA DO FUTEBOL"
TITULO = "O CORPO DO NEYMAR"


@dataclass(frozen=True)
class Layout:
    foto_altura: int = 760
    foto_topo: int = 395
    cartao_topo: int = 1168
    cartao_altura: int = 112
    barras_topo: int = 1296
    barra_altura: int = 38
    barra_passo: int = 52


def fonte(caminho: str, tamanho: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(caminho, tamanho)


def fmt_int(n: float) -> str:
    return f"{int(n):,}".replace(",", ".")


def intensidade(dias: float) -> tuple[int, int, int]:
    """Quanto mais dias acumulados no ponto, mais escuro e saturado o vermelho."""
    t = min(dias / 400, 1.0)
    return (int(255 - 60 * t), int(120 - 110 * t), int(110 - 100 * t))


TRAJETORIA = [("SANTOS", "2009"), ("BARCELONA", "2013"), ("PSG", "2017"), ("AL-HILAL", "2023"), ("SANTOS", "2025")]

# Opacidades do fundo (0-255). Tudo abaixo de ~12%: decora sem competir com foto e dados.
ALFA_CAMPO, ALFA_DEZ, ALFA_FAIXA, ALFA_FAIXA_ATUAL = 22, 26, 60, 170


def camada_campo() -> Image.Image:
    """Linhas de um campo visto de cima, ocupando a tela vertical."""
    cam = Image.new("RGBA", (LARGURA, ALTURA), (0, 0, 0, 0))
    d = ImageDraw.Draw(cam)
    cor, w = (255, 255, 255, ALFA_CAMPO), 3
    m, meio = 40, ALTURA // 2
    d.rectangle((m, m, LARGURA - m, ALTURA - m), outline=cor, width=w)
    d.line((m, meio, LARGURA - m, meio), fill=cor, width=w)
    d.ellipse((LARGURA / 2 - 170, meio - 170, LARGURA / 2 + 170, meio + 170), outline=cor, width=w)
    for topo, sinal in ((m, 1), (ALTURA - m, -1)):  # grande área + pequena área + arco, em cima e embaixo
        d.rectangle(sorted_box(220, topo, LARGURA - 220, topo + sinal * 300), outline=cor, width=w)
        d.rectangle(sorted_box(380, topo, LARGURA - 380, topo + sinal * 110), outline=cor, width=w)
        cy = topo + sinal * 300
        d.arc((LARGURA / 2 - 120, cy - 120, LARGURA / 2 + 120, cy + 120),
              start=0 if sinal == 1 else 180, end=180 if sinal == 1 else 360, fill=cor, width=w)
    return cam


def sorted_box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
    """Pillow exige y0 <= y1; a área de baixo é desenhada 'para cima'."""
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def camada_dez() -> Image.Image:
    """O '10' que ele vestiu em todos os clubes, gigante e só no contorno."""
    cam = Image.new("RGBA", (LARGURA, ALTURA), (0, 0, 0, 0))
    ImageDraw.Draw(cam).text((LARGURA / 2, 930), "10", font=fonte(FONTE_IMPACTO, 1000), anchor="mm",
                             fill=(0, 0, 0, 0), stroke_width=4, stroke_fill=(255, 255, 255, ALFA_DEZ))
    return cam


def camada_trajetoria(etapa_atual: int) -> Image.Image:
    """Faixa vertical com a carreira; a etapa atual fica mais acesa."""
    fnt = fonte(FONTE_TEXTO, 26)
    partes = [f"{clube} {ano}" for clube, ano in TRAJETORIA]
    sep = "   ·   "
    larg_total = int(sum(fnt.getlength(p) for p in partes) + fnt.getlength(sep) * (len(partes) - 1))
    faixa = Image.new("RGBA", (larg_total, 50), (0, 0, 0, 0))
    d, x = ImageDraw.Draw(faixa), 0.0
    for i, parte in enumerate(partes):
        alfa = ALFA_FAIXA_ATUAL if i == etapa_atual else ALFA_FAIXA
        d.text((x, 25), parte, font=fnt, fill=(255, 255, 255, alfa), anchor="lm")
        x += fnt.getlength(parte) + fnt.getlength(sep)
    girada = faixa.rotate(90, expand=True)  # lê de baixo para cima, como lombada de livro
    cam = Image.new("RGBA", (LARGURA, ALTURA), (0, 0, 0, 0))
    # alpha_composite, não paste(img, pos, img): usar a imagem como máscara de si mesma multiplica o alfa duas vezes.
    cam.alpha_composite(girada, (54, 395 + (760 - girada.height) // 2))
    return cam


def camada_canal() -> Image.Image:
    """Nome do canal + título na lateral direita, lendo de cima para baixo (espelho da trajetória)."""
    cam = Image.new("RGBA", (LARGURA, ALTURA), (0, 0, 0, 0))
    x = LARGURA - 54
    for texto, tam, alfa in ((CANAL, 28, 170), (TITULO, 24, 95)):
        fnt = fonte(FONTE_TEXTO, tam)
        faixa = Image.new("RGBA", (int(fnt.getlength(texto)) + 4, tam + 12), (0, 0, 0, 0))
        ImageDraw.Draw(faixa).text((2, (tam + 12) // 2), texto, font=fnt, fill=(255, 255, 255, alfa), anchor="lm")
        girada = faixa.rotate(-90, expand=True)
        x -= girada.width
        cam.alpha_composite(girada, (x, 410))
        x -= 6
    return cam


def etapa_da_carreira(data: pd.Timestamp) -> int:
    datas = [pd.Timestamp(f"{ano}-01-01") for _, ano in TRAJETORIA]
    return max(i for i, d in enumerate(datas) if data >= d)


def desenhar_fundo(clube_atual: str, data: pd.Timestamp) -> Image.Image:
    return fundo_cacheado(clube_atual, etapa_da_carreira(data)).copy()


@lru_cache(maxsize=16)
def fundo_cacheado(clube_atual: str, etapa: int) -> Image.Image:
    """O blur de 160 px é a parte mais lenta do quadro; só existem ~5 fundos diferentes no vídeo."""
    img = Image.new("RGBA", (LARGURA, ALTURA), (*FUNDO, 255))
    # Brilho atrás do corpo na cor do clube atual: o fundo "troca de camisa" junto com a carreira.
    cor = CLUBES[clube_atual]["cor"]
    brilho = Image.new("RGBA", (LARGURA, ALTURA), (0, 0, 0, 0))
    ImageDraw.Draw(brilho).ellipse((90, 450, 990, 1450), fill=(*cor, 70))
    img.alpha_composite(brilho.filter(ImageFilter.GaussianBlur(160)))
    for camada in (camada_campo(), camada_dez(), camada_trajetoria(etapa), camada_canal()):
        img.alpha_composite(camada)
    return img.convert("RGB")


CONTADOR_Y = 262  # centro do número grande; usado também pelo "+N" que sobe ao lado


def desenhar_contador(d: ImageDraw.ImageDraw, dias: float, rotulo: str) -> None:
    d.text((LARGURA // 2, 170), "DIAS FORA DE CAMPO POR LESÃO", font=fonte(FONTE_TEXTO, 40),
           fill=CINZA, anchor="mm")
    d.text((LARGURA // 2, CONTADOR_Y), fmt_int(dias), font=fonte(FONTE_IMPACTO, 160), fill=BRANCO, anchor="mm")
    d.text((LARGURA // 2, 362), rotulo, font=fonte(FONTE_TEXTO, 42), fill=VERMELHO, anchor="mm")


@lru_cache(maxsize=4)
def foto_redimensionada(altura: int) -> tuple[Image.Image, float]:
    """Redimensionar com LANCZOS é caro: num vídeo de 1.200 frames, faz uma vez só (memoização)."""
    foto = Image.open(FOTO)
    escala = altura / foto.height
    return foto.resize((round(foto.width * escala), altura), Image.LANCZOS), escala


def colar_foto(img: Image.Image, lay: Layout) -> tuple[float, int]:
    foto, escala = foto_redimensionada(lay.foto_altura)
    x0 = (LARGURA - foto.width) // 2
    img.paste(foto, (x0, lay.foto_topo), foto)
    return escala, x0


@lru_cache(maxsize=8)
def escudo(nome_arquivo: str) -> Image.Image:
    img = Image.open(ASSETS / "escudos" / nome_arquivo).convert("RGBA")
    img.thumbnail((46, 46))
    return img


def agrupar_marcadores(ate: pd.DataFrame) -> pd.DataFrame:
    return ate.groupby(["regiao", "lado"]).agg(n=("dias_total", "size"), dias=("dias_total", "sum"))


def posicao_tela(regiao: str, lado: str, escala: float, x0: int, lay: Layout) -> tuple[float, float] | None:
    if (regiao, lado) not in REGIAO_XY:
        return None  # região genérica ("não informada"): só entra no contador
    fx, fy = REGIAO_XY[(regiao, lado)]
    return x0 + fx * escala, lay.foto_topo + fy * escala


def desenhar_marcadores(img: Image.Image, grupos: pd.DataFrame, escala: float, x0: int, lay: Layout,
                        pop: dict[tuple[str, str], float] | None = None) -> None:
    """`pop` aumenta o raio de um marcador (efeito de 'cravar' quando a lesão acontece)."""
    d = ImageDraw.Draw(img, "RGBA")
    pop = pop or {}
    for (regiao, lado), g in grupos.iterrows():
        pos = posicao_tela(regiao, lado, escala, x0, lay)
        if pos is None:
            continue
        x, y = pos
        cor = intensidade(g["dias"])
        r = (10 + min(g["n"], 6) * 1.5) * pop.get((regiao, lado), 1.0)
        d.ellipse((x - r * 1.6, y - r * 1.6, x + r * 1.6, y + r * 1.6), fill=(*cor, 60))  # halo
        if lado == "não informado":
            d.ellipse((x - r, y - r, x + r, y + r), outline=cor, width=5)
        else:
            d.ellipse((x - r, y - r, x + r, y + r), fill=cor, outline=BRANCO, width=3)
        if g["n"] > 1:
            desenhar_badge(d, x, y, r, g["n"], lado)


def desenhar_badge(d: ImageDraw.ImageDraw, x: float, y: float, r: float, n: int, lado: str) -> None:
    """Badge "+N" empurrado para FORA do corpo, para não brigar com marcadores vizinhos.

    Lado direito do jogador = esquerda da tela. Marcador central ("não informado") leva o badge para cima.
    """
    larg, alt = 50, 34
    if lado == "direito":
        cx, cy = x - r - 12 - larg / 2, y
    elif lado == "esquerdo":
        cx, cy = x + r + 12 + larg / 2, y
    else:
        cx, cy = x, y - r - 8 - alt / 2
    d.rounded_rectangle((cx - larg / 2, cy - alt / 2, cx + larg / 2, cy + alt / 2), radius=10, fill=BRANCO)
    d.text((cx, cy), f"+{n}", font=fonte(FONTE_TEXTO, 28), fill=FUNDO, anchor="mm")


def desenhar_cartao(img: Image.Image, lesao: pd.Series, n: int, lay: Layout, dx: float = 0) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    x0, y0 = 60 + dx, lay.cartao_topo
    x1, y1 = SEGURA_DIREITA + dx, y0 + lay.cartao_altura
    d.rounded_rectangle((x0, y0, x1, y1), radius=20, fill=(20, 22, 30, 235), outline=VERMELHO, width=4)
    local = rotulo_local(lesao["regiao"], lesao["lado"])
    d.text((x0 + 28, y0 + 34), f"#{n}  {local.upper()}", font=fonte(FONTE_IMPACTO, 44), fill=BRANCO, anchor="lm")
    detalhe = f"{lesao['inicio']:%d/%m/%Y}  ·  {lesao['clube']}  ·  {fmt_int(lesao['dias_total'])} dias fora"
    d.text((x0 + 28, y0 + 84), detalhe, font=fonte(FONTE_TEXTO_LEVE, 36), fill=CINZA, anchor="lm")


def desenhar_placar(img: Image.Image, dias: dict[str, float], clube_atual: str, lay: Layout,
                    maximo: float) -> None:
    """`maximo` fixo (o maior total do vídeo): a barra cresce em escala absoluta, não "re-escala" a cada frame."""
    d = ImageDraw.Draw(img, "RGBA")
    largura_max, altura_barra, passo = 560, lay.barra_altura, lay.barra_passo
    for i, (nome, cfg) in enumerate(CLUBES.items()):
        y = lay.barras_topo + i * passo
        esc = escudo(cfg["escudo"])
        img.paste(esc, (72 + (46 - esc.width) // 2, y + (altura_barra - esc.height) // 2), esc)
        valor = dias.get(nome, 0)
        w = max(int(largura_max * valor / maximo), 6)
        ativo = nome == clube_atual
        # Clube atual em cor cheia; os outros "congelados": mesma cor misturada com cinza.
        cor = cfg["cor"] if ativo else tuple(int(c * 0.45 + 90 * 0.55) for c in cfg["cor"])
        d.rounded_rectangle((150, y, 150 + w, y + altura_barra), radius=10, fill=cor)
        if ativo:
            d.rounded_rectangle((146, y - 4, 154 + w, y + altura_barra + 4), radius=12, outline=BRANCO, width=3)
        # Número sempre em branco puro, com sombra, para destacar sobre o fundo decorado.
        pos = (150 + w + 18, y + altura_barra // 2)
        d.text((pos[0] + 2, pos[1] + 3), fmt_int(valor), font=fonte(FONTE_IMPACTO, 42), fill=(0, 0, 0, 160), anchor="lm")
        d.text(pos, fmt_int(valor), font=fonte(FONTE_IMPACTO, 42), fill=BRANCO_PURO, anchor="lm")


def desenhar_quadro(lesoes: pd.DataFrame, indice: int, lay: Layout = Layout()) -> Image.Image:
    """Quadro no momento em que a lesão `indice` (0-based, ordem cronológica) acaba de acontecer."""
    ate = lesoes.iloc[: indice + 1]
    atual = ate.iloc[-1]
    img = desenhar_fundo(atual["clube"], atual["inicio"])
    d = ImageDraw.Draw(img)
    desenhar_contador(d, ate["dias_total"].sum(), f"LESÃO {indice + 1} DE {len(lesoes)}")
    escala, x0 = colar_foto(img, lay)
    desenhar_marcadores(img, agrupar_marcadores(ate), escala, x0, lay)
    desenhar_cartao(img, atual, indice + 1, lay)
    maximo = lesoes.groupby("clube")["dias_total"].sum().max()
    desenhar_placar(img, ate.groupby("clube")["dias_total"].sum().to_dict(), atual["clube"], lay, maximo)
    return img


def carregar_lesoes(caminho: Path = Path("dados/neymar_lesoes.csv")) -> pd.DataFrame:
    return pd.read_csv(caminho, parse_dates=["inicio", "fim"]).sort_values("inicio").reset_index(drop=True)


if __name__ == "__main__":
    import sys

    lesoes = carregar_lesoes()
    data = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2023-10-19")
    idx = int(lesoes.index[lesoes["inicio"] <= data].max())
    saida = Path("render/quadro_teste.png")
    saida.parent.mkdir(exist_ok=True)
    desenhar_quadro(lesoes, idx).save(saida)
    print(f"quadro da lesão #{idx + 1} salvo em {saida}")
