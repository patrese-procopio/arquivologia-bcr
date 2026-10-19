"""
Encarada estilo UFC para o gancho dos vídeos: dois painéis de foto, um de frente para o outro,
separados por uma diagonal, com tratamento de cor de "pôster de luta".

Etapas por jogador (tudo feito UMA vez, antes do vídeo):
    foto -> detecção do rosto (OpenCV, opcional) -> recorte centrado no rosto
         -> espelhamento opcional (para os dois se olharem) -> color grading
         -> degradês (cor do clube na borda externa, escuro embaixo para o placar)
         -> máscara diagonal (canal alfa)

Sem foto disponível, o painel vira um bloco na cor do clube com as iniciais gigantes.

LGPD/direitos: as fotos vêm do Wikimedia Commons com licença livre (ver CREDITOS.md).
Recortar, espelhar e colorir é ADAPTAÇÃO: permitido em CC BY / CC BY-SA, desde que creditado.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from matplotlib import font_manager

from avatares import EXTENSOES, _iniciais, slug

log = logging.getLogger("encarada")

# Geometria em fração da LARGURA da tela: cada painel cobre 56% e a diagonal corta no meio
LARGURA_PAINEL = 0.56
DIAGONAL_TOPO, DIAGONAL_BASE = 0.54, 0.46     # x da linha divisória no topo e na base da faixa


def _detectar_rosto(img: Image.Image) -> tuple[float, float, float] | None:
    """(cx, cy, altura) do maior rosto, em pixels. None se não achar ou sem OpenCV instalado.

    Haar cascade: detector clássico (Viola-Jones), leve e sem download de modelo,
    já vem dentro do pacote opencv-python. Ótimo para foto de rosto frontal.
    """
    try:
        import cv2
    except ImportError:
        log.info("OpenCV não instalado: recorte pelo topo (pip install opencv-python-headless)")
        return None
    if not hasattr(cv2, "CascadeClassifier"):     # OpenCV 5 tirou o Haar do pacote principal
        log.info("Detector Haar indisponível nesta versão do OpenCV: recorte pelo topo "
                 "(pip install \"opencv-python-headless<5\")")
        return None
    pasta = os.environ.get("CASCATA_HAAR_DIR") or cv2.data.haarcascades
    cascata = cv2.CascadeClassifier(str(Path(pasta) / "haarcascade_frontalface_default.xml"))
    if cascata.empty():
        return None
    cinza = np.asarray(img.convert("L"))
    minimo = max(30, min(img.size) // 10)
    rostos = cascata.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=6, minSize=(minimo, minimo))
    if len(rostos) == 0:
        return None
    x, y, w, h = max(rostos, key=lambda r: r[2] * r[3])        # o maior rosto é o protagonista
    return x + w / 2, y + h / 2, float(h)


def _recortar_no_rosto(img: Image.Image, largura: int, altura: int) -> Image.Image:
    """Recorte com a proporção do painel, rosto ocupando ~34% da altura, olhos no terço superior."""
    w, h = img.size
    rosto = _detectar_rosto(img)
    alvo = largura / altura
    if rosto is None:
        # sem detecção: maior recorte possível, puxado para o topo (retrato = rosto em cima)
        cw = min(w, h * alvo)
        ch = cw / alvo
        x0, y0 = (w - cw) / 2, 0.0
    else:
        cx, cy, fh = rosto
        ch = min(h, fh / 0.34)
        cw = ch * alvo
        if cw > w:                                   # foto estreita: limita pela largura
            cw = w
            ch = cw / alvo
        x0 = min(max(cx - cw / 2, 0), w - cw)
        y0 = min(max(cy - ch * 0.38, 0), h - ch)
    return img.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch))).resize(
        (largura, altura), Image.LANCZOS)


def _painel_sem_foto(jogador: str, cor: str, largura: int, altura: int) -> Image.Image:
    base = np.array(to_rgb(cor)) * 255
    gy = np.linspace(1.0, 0.45, altura)[:, None, None]
    arr = np.clip(base * gy * np.ones((altura, largura, 1)), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    caminho = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    fonte = ImageFont.truetype(caminho, int(altura * 0.34))
    ImageDraw.Draw(img).text((largura / 2, altura * 0.42), _iniciais(jogador), font=fonte,
                             fill=(255, 255, 255), anchor="mm")
    return img


def _color_grading(img: Image.Image, cor: str) -> Image.Image:
    """Visual de pôster: mais contraste, menos saturação e sombras puxadas para a cor do clube."""
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Color(img).enhance(0.80)
    arr = np.asarray(img).astype(float) / 255
    lum = arr.mean(axis=2, keepdims=True)
    tinta = np.array(to_rgb(cor))
    sombra = (1 - lum) ** 2                                   # quanto mais escuro, mais tingido
    arr = arr * (1 - 0.35 * sombra) + tinta * 0.35 * sombra
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def montar_painel(pasta_fotos: Path, jogador: str, sigla: str, cor: str, lado: str,
                  largura_px: int, altura_px: int, espelhar: bool = False) -> np.ndarray:
    """Painel RGBA float (0-1) de um jogador. lado = "esquerda" | "direita"."""
    foto = next((pasta_fotos / f"{slug(jogador, sigla)}{ext}" for ext in EXTENSOES
                 if (pasta_fotos / f"{slug(jogador, sigla)}{ext}").exists()), None)
    largura = round(largura_px * LARGURA_PAINEL)
    if foto is None:
        log.info("Sem foto de %s: painel com iniciais", jogador)
        img = _painel_sem_foto(jogador, cor, largura, altura_px)
    else:
        img = _recortar_no_rosto(Image.open(foto).convert("RGB"), largura, altura_px)
        if espelhar:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = _color_grading(img, cor)
    return finalizar_painel(img, cor, lado, largura_px)


def painel_de_imagem(img: Image.Image, cor: str, lado: str, largura_px: int, altura_px: int) -> np.ndarray:
    """Painel a partir de qualquer imagem (ex.: bandeira): recorte "cover" centralizado, sem color grading."""
    largura = round(largura_px * LARGURA_PAINEL)
    img = ImageOps.fit(img.convert("RGB"), (largura, altura_px), method=Image.LANCZOS, centering=(0.5, 0.5))
    return finalizar_painel(img, cor, lado, largura_px, brilho_externo=0.25)


def finalizar_painel(img: Image.Image, cor: str, lado: str, largura_px: int,
                     brilho_externo: float = 0.45) -> np.ndarray:
    """Degradês + máscara diagonal: o que transforma uma imagem retangular em "lado da encarada"."""
    arr = np.asarray(img).astype(float) / 255
    h, w = arr.shape[:2]
    u = np.linspace(0, 1, w)[None, :, None]                   # 0 = borda esquerda do painel
    v = np.linspace(0, 1, h)[:, None, None]                   # 0 = topo
    externo = (1 - u) if lado == "esquerda" else u            # 1 na borda externa da tela
    tinta = np.array(to_rgb(cor))
    k = brilho_externo * externo ** 2
    arr = arr * (1 - k) + tinta * k                                             # brilho do lado por fora
    escuro = np.clip((v - 0.55) / 0.45, 0, 1) ** 1.3                           # base escura: placar legível
    arr = arr * (1 - 0.80 * escuro)

    # Máscara diagonal (canal alfa) com borda suave de ~2 px
    x_tela = (np.arange(w) + 0.5) / w * LARGURA_PAINEL + (0 if lado == "esquerda" else 1 - LARGURA_PAINEL)
    linha = DIAGONAL_TOPO + (DIAGONAL_BASE - DIAGONAL_TOPO) * np.linspace(0, 1, h)[:, None]
    dist_px = (linha - x_tela[None, :]) * largura_px          # >0 = à esquerda da diagonal
    if lado == "direita":
        dist_px = -dist_px
    alfa = np.clip(dist_px / 2 + 0.5, 0, 1)
    # Bordas de cima e de baixo dissolvem no fundo (sem "recorte de tesoura")
    v2 = v[..., 0]
    alfa = alfa * np.clip(v2 / 0.22, 0, 1) ** 1.2 * np.clip((1 - v2) / 0.22, 0, 1) ** 0.8
    return np.dstack([np.clip(arr, 0, 1), alfa])


# ---------------------------------------------------------------------------
# A cena completa (reutilizada pelo bar race e pelo mapa)
# ---------------------------------------------------------------------------
def _entrada(f: float, inicio: float, duracao: float) -> float:
    """0 -> 1 com "ease-out-back": passa um pouco do alvo e volta (efeito de "pop")."""
    x = min(max((f - inicio) / duracao, 0.0), 1.0)
    c = 1.70158
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


@dataclass
class CenaEncarada:
    """Abertura "encarada": frase grande + dois painéis frente a frente + placar que conta.

    Não sabe nada de futebol, foto ou bandeira: recebe 2 painéis RGBA prontos, cores e textos.
    Por isso serve para Gabigol x Pedro (fotos) e para Brasil x Argentina (bandeiras).
    Uso: cena = CenaEncarada(...); cena.montar(fig); a cada quadro: cena.desenhar(fase 0->1).
    """
    paineis: tuple[np.ndarray, np.ndarray]
    cores: tuple[str, str]
    nomes: tuple[str, str]
    valores: tuple[float, float]
    titulo: str
    subtitulo: str
    chamada: str = "Veja como chegamos aqui ↓"
    cor_titulo: str = "#F59E2B"
    cor_texto: str = "#E8EEF6"
    cor_chamada: str = "#3FC1E8"
    cor_fundo: str = "#0B1220"
    artistas: list = field(default_factory=list)

    Y0, ALTURA = 0.335, 0.385          # faixa dos painéis, em fração da altura da tela

    @classmethod
    def tamanho_px(cls, largura_tela_px: int, altura_tela_px: int) -> tuple[int, int]:
        """(largura da tela, altura da faixa) em px: o que montar_painel/painel_de_imagem precisam."""
        return largura_tela_px, round(altura_tela_px * cls.ALTURA)

    def montar(self, fig) -> None:
        import matplotlib.patheffects as pe
        from matplotlib.patches import Rectangle

        ax = fig.add_axes((0, self.Y0, 1, self.ALTURA), facecolor="none")
        self.ax = ax
        self.imagens = []
        for k, painel in enumerate(self.paineis):
            x0 = 0.0 if k == 0 else 1 - LARGURA_PAINEL
            im = ax.imshow(painel, extent=(x0, x0 + LARGURA_PAINEL, 0, 1), aspect="auto",
                           interpolation="bilinear", zorder=1)
            im.extent_base = (x0, x0 + LARGURA_PAINEL)
            self.imagens.append(im)
        # imshow mexe nos limites: fixamos e travamos o autoscale
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_autoscale_on(False)
        ax.set_axis_off()

        self.linha, = ax.plot([DIAGONAL_TOPO, DIAGONAL_BASE], [1, 0], color="#FFFFFF", lw=4, zorder=3,
                              solid_capstyle="butt",
                              path_effects=[pe.Stroke(linewidth=14, foreground=self.cor_titulo, alpha=0.6),
                                            pe.Normal()])
        self.flash = Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#FFFFFF", alpha=0.0, zorder=4)
        ax.add_patch(self.flash)

        self.numeros, self.rotulos = [], []
        for k, (cor, nome) in enumerate(zip(self.cores, self.nomes)):
            xc = 0.25 if k == 0 else 0.75
            # contorno duplo (escuro por fora, cor por dentro): legível sobre foto, bandeira ou fundo liso
            self.numeros.append(fig.text(xc, 0.392, "0", ha="center", va="center", color="#FFFFFF",
                                         fontsize=124, fontweight="black",
                                         path_effects=[pe.Stroke(linewidth=22, foreground=self.cor_fundo),
                                                       pe.Stroke(linewidth=11, foreground=cor), pe.Normal()]))
            self.rotulos.append(fig.text(xc, 0.347, nome.upper(), ha="center", va="center",
                                         color=self.cor_texto, fontsize=38, fontweight="bold",
                                         path_effects=[pe.withStroke(linewidth=5, foreground=self.cor_fundo)]))
        self.x = fig.text(0.5, 0.53, "X", ha="center", va="center", color="#FFFFFF", fontsize=96,
                          fontweight="black", path_effects=[pe.withStroke(linewidth=10, foreground=self.cor_titulo)])
        # fonte se ajusta ao comprimento: "FALTAM 3 GOLS" (13) cabe em 86 pt, "EMPATE HISTÓRICO" (16) não
        self.fs_titulo = min(86.0, 1120.0 / max(len(self.titulo), 1))
        self.t_titulo = fig.text(0.5, 0.800, self.titulo, ha="center", va="center", color=self.cor_titulo,
                                 fontsize=self.fs_titulo, fontweight="black")
        self.t_sub = fig.text(0.5, 0.748, self.subtitulo, ha="center", va="center", color=self.cor_texto,
                              fontsize=40, fontweight="bold")
        self.t_chamada = fig.text(0.5, 0.296, self.chamada, ha="center", va="center", color=self.cor_chamada,
                                  fontsize=34, fontweight="bold")
        self.artistas = [ax, self.t_titulo, self.t_sub, self.x, self.t_chamada, *self.numeros, *self.rotulos]
        # Medição REAL da largura: a estimativa por nº de letras falha quando a fonte instalada
        # é mais larga (ex.: DejaVu no lugar da Inter). Encolhe até caber em 92% da tela.
        self.fs_titulo = self._caber(fig, self.t_titulo, self.fs_titulo)
        self._caber(fig, self.t_sub, self.t_sub.get_fontsize())
        self.set_visible(False)

    @staticmethod
    def _caber(fig, texto, tamanho: float, largura_max: float = 0.92) -> float:
        """Reduz a fonte até o texto caber em `largura_max` da largura da figura. Devolve o tamanho final."""
        renderer = fig.canvas.get_renderer()
        largura_fig = fig.bbox.width
        for _ in range(20):
            texto.set_fontsize(tamanho)
            largura = texto.get_window_extent(renderer=renderer).width / largura_fig
            if largura <= largura_max:
                break
            tamanho *= largura_max / largura * 0.98
        return tamanho

    def set_visible(self, visivel: bool) -> None:
        for a in self.artistas:
            a.set_visible(visivel)

    def desenhar(self, f: float) -> None:
        """f = fase de 0 a 1 dentro da abertura.

        O QUADRO 0 JÁ PRECISA ESTAR CHEIO: é ele que decide se o dedo para.
        Por isso nada nasce do zero: os painéis começam 70% dentro da tela e "colidem" no centro.
        """
        saida = 1.0 - min(max((f - 0.88) / 0.12, 0.0), 1.0)      # some nos últimos 12%
        self.t_titulo.set_fontsize(self.fs_titulo * (0.85 + 0.15 * _entrada(f, 0.00, 0.12)))
        for a in (self.t_titulo, self.t_sub, self.x, self.linha):
            a.set_alpha(saida)

        avanco = _entrada(f, 0.0, 0.16)                           # aproximação com overshoot
        impacto = min(max((f - 0.14) / 0.10, 0.0), 1.0)
        tremor = 0.006 * np.sin(f * 220) * (1 - impacto) if f >= 0.14 else 0.0
        zoom = 1 + 0.03 * f                                       # push-in lento de câmera
        for k, im in enumerate(self.imagens):
            sentido = -1 if k == 0 else 1
            x0, x1 = im.extent_base
            desloc = sentido * 0.18 * (1 - avanco) + tremor
            meio, meia = (x0 + x1) / 2, (x1 - x0) / 2 * zoom
            im.set_extent((meio - meia + desloc, meio + meia + desloc, 0.5 - zoom / 2, 0.5 + zoom / 2))
            im.set_alpha(saida)
        self.flash.set_alpha(0.55 * (1 - impacto) * saida if f >= 0.14 else 0.0)

        conta = 1 - (1 - min(max(f / 0.50, 0.0), 1.0)) ** 3     # placar "conta" até o valor real
        for k, (numero, rotulo) in enumerate(zip(self.numeros, self.rotulos)):
            numero.set_text(f"{self.valores[k] * (0.4 + 0.6 * conta):.0f}")   # quadro 0 já tem número
            numero.set_alpha(saida)
            rotulo.set_alpha(saida)
        self.t_chamada.set_alpha((0.55 + 0.45 * np.sin(f * 4 * np.pi) ** 2)
                                 * min(1.0, max(0.0, (f - 0.45) / 0.1)) * saida)
