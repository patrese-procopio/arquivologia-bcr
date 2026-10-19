"""
"Momentos de herói": cartão de 1,3 s com o jogador decisivo de uma conquista, sobre o mapa.

Por que existe: o mapa é informativo, mas sem rostos. A cada ponto de virada da história,
um cartão com o herói daquele título cria um "pico" de emoção e quebra a monotonia.

Dados: dados/herois_libertadores.csv (um herói por ano; os fatos de cada linha precisam de fonte).
Imagens: assets/herois/. O crédito de cada foto vai NO CARTÃO (campo `credito` do CSV).

Padronização visual (fotos de origens e resoluções muito diferentes):
    recorte 4:5 -> mesmo tamanho -> contraste/nitidez -> tom do clube nas sombras
    -> grão de filme (disfarça diferença de resolução) -> borda branca -> leve inclinação
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

log = logging.getLogger("herois")

PROPORCAO = 4 / 5                 # largura / altura do cartão (padrão retrato do Instagram)


@dataclass(frozen=True)
class Heroi:
    temporada: int
    clube: str
    jogador: str
    arquivo: Path
    recorte: tuple[int, int, int, int] | None
    cores: tuple[str, str]        # cor principal e secundária do clube (texto e confete)
    credito: str
    motivo: str


def carregar_herois(caminho_csv: Path, pasta_imagens: Path) -> dict[int, Heroi]:
    """Lê o CSV e valida TUDO antes do render (fail fast): arquivo existe? recorte é válido?"""
    if not caminho_csv.exists():
        return {}
    herois, problemas = {}, []
    with caminho_csv.open(encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            arquivo = pasta_imagens / linha["arquivo"]
            if not arquivo.exists():
                # foto ainda não obtida: AVISA e segue (o vídeo sai sem esse cartão).
                # Erro de verdade (recorte inválido) continua interrompendo tudo.
                log.warning("Herói %s (%s) sem imagem em %s: cartão pulado", linha["jogador"],
                            linha["temporada"], arquivo)
                continue
            recorte = None
            if linha["recorte"].strip():
                recorte = tuple(int(v) for v in linha["recorte"].split(","))
                if len(recorte) != 4:
                    problemas.append(f"{linha['jogador']}: recorte precisa de 4 números (x0,y0,x1,y1)")
                    continue
            cores = tuple(c.strip() for c in linha["cores"].split(";"))
            herois[int(linha["temporada"])] = Heroi(
                int(linha["temporada"]), linha["clube"], linha["jogador"], arquivo, recorte,
                (cores[0], cores[-1]), linha["credito"], linha["motivo"])
    if problemas:
        raise ValueError("Heróis inválidos:\n- " + "\n- ".join(problemas))
    log.info("Heróis: %s", ", ".join(f"{h.jogador} ({a})" for a, h in sorted(herois.items())))
    return herois


def preparar_cartao(h: Heroi, largura_px: int) -> np.ndarray:
    """Foto -> cartão RGBA padronizado (array float 0-1), já inclinado.

    Retrato vira 4:5 (padrão). Foto de TIME (horizontal) mantém o formato até 3:2:
    forçar 4:5 cortaria metade do elenco.
    """
    img = Image.open(h.arquivo).convert("RGB")
    if h.recorte:
        img = img.crop(h.recorte)
    proporcao = min(max(img.width / img.height, PROPORCAO), 1.5)
    altura_px = round(largura_px / proporcao)
    img = ImageOps.fit(img, (largura_px, altura_px), method=Image.LANCZOS, centering=(0.5, 0.35))

    # Tratamento único para todas: assim fotos de 1962 e de 2025 parecem da mesma "coleção"
    img = ImageEnhance.Contrast(img).enhance(1.18)
    img = ImageEnhance.Color(img).enhance(0.92)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=2))
    arr = np.asarray(img).astype(float) / 255
    lum = arr.mean(axis=2, keepdims=True)
    sombra = (1 - lum) ** 2
    tinta = np.array(to_rgb(h.cores[0]))
    arr = arr * (1 - 0.22 * sombra) + tinta * 0.22 * sombra       # sombras puxadas para a cor do clube
    rng = np.random.default_rng(h.temporada)                      # grão fixo por herói (reprodutível)
    arr = arr + rng.normal(0, 0.035, arr.shape[:2])[..., None]    # grão de filme
    # vinheta leve: centro mais claro, foco no jogador
    yy, xx = np.mgrid[0:altura_px, 0:largura_px]
    d = np.sqrt(((xx / largura_px - 0.5) / 0.7) ** 2 + ((yy / altura_px - 0.45) / 0.8) ** 2)
    arr = arr * (1 - 0.35 * np.clip(d - 0.45, 0, 1)[..., None])
    foto = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

    # Borda branca de "figurinha" e leve inclinação (dinamismo)
    borda = max(6, largura_px // 40)
    cartao = Image.new("RGBA", (largura_px + 2 * borda, altura_px + 2 * borda), (255, 255, 255, 255))
    cartao.paste(foto, (borda, borda))
    cartao = cartao.rotate(-3, resample=Image.BICUBIC, expand=True)
    return np.asarray(cartao).astype(float) / 255


@dataclass
class CenaHeroi:
    """Cartão + carimbo "CAMPEÃO <ano>" + nome + confete, por cima do mapa escurecido."""
    herois: dict[int, Heroi]
    largura_tela_px: int
    altura_tela_px: int
    cor_carimbo: str = "#F59E2B"
    cor_texto: str = "#E8EEF6"
    cor_fundo: str = "#0B1220"
    n_confete: int = 46                  # pouco confete: impacto sem poluir
    artistas: list = field(default_factory=list)

    CENTRO = (0.5, 0.585)                # centro do cartão (fração da tela)
    LARGURA = 0.60                       # largura do cartão (fração da largura da tela)

    def montar(self, fig) -> None:
        import matplotlib.patheffects as pe
        from matplotlib.patches import Rectangle

        ax = fig.add_axes((0, 0, 1, 1), facecolor="none", zorder=50)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        self.ax = ax
        self.escuro = Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=self.cor_fundo, alpha=0, zorder=0)
        ax.add_patch(self.escuro)

        self.cartoes = {}
        larg_px = round(self.largura_tela_px * self.LARGURA)
        for ano, h in self.herois.items():
            rgba = preparar_cartao(h, larg_px)
            im = ax.imshow(rgba, extent=(0, 1, 0, 1), aspect="auto", interpolation="bilinear", zorder=2)
            im.set_visible(False)
            im.proporcao_tela = (rgba.shape[0] / rgba.shape[1]) * (self.largura_tela_px / self.altura_tela_px)
            self.cartoes[ano] = im
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_autoscale_on(False)

        self.flash = Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#FFFFFF", alpha=0, zorder=6)
        ax.add_patch(self.flash)
        rng = np.random.default_rng(11)
        self._conf_x0 = rng.uniform(0.05, 0.95, self.n_confete)
        self._conf_y0 = rng.uniform(1.02, 1.35, self.n_confete)
        self._conf_v = rng.uniform(0.75, 1.25, self.n_confete)
        self._conf_fase = rng.uniform(0, 2 * np.pi, self.n_confete)
        self._conf_tam = rng.uniform(40, 120, self.n_confete)
        self._conf_lado = rng.integers(0, 2, self.n_confete)
        self.confete = ax.scatter([], [], s=[], marker="s", zorder=5, linewidths=0)

        contorno = [pe.withStroke(linewidth=8, foreground=self.cor_fundo)]
        self.t_carimbo = fig.text(0.5, 0.315, "", ha="center", va="center", color=self.cor_carimbo,
                                  fontsize=64, fontweight="black", zorder=60, rotation=-3,
                                  path_effects=[pe.withStroke(linewidth=10, foreground=self.cor_fundo)])
        self.t_nome = fig.text(0.5, 0.262, "", ha="center", va="center", color="#FFFFFF",
                               fontsize=50, fontweight="black", zorder=60, path_effects=contorno)
        self.t_motivo = fig.text(0.5, 0.222, "", ha="center", va="center", color=self.cor_texto,
                                 fontsize=27, fontweight="bold", zorder=60, path_effects=contorno)
        self.t_credito = fig.text(0.5, 0.0, "", ha="center", va="top", color="#FFFFFF", alpha=0.75,
                                  fontsize=13, zorder=60, path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])
        self.textos = [self.t_carimbo, self.t_nome, self.t_motivo, self.t_credito]
        self.artistas = [ax, *self.textos]
        self.esconder()

    def esconder(self) -> None:
        for a in self.artistas:
            a.set_visible(False)

    def desenhar(self, ano: int, f: float) -> None:
        """f = fase 0 -> 1 dos 1,3 s do cartão."""
        import matplotlib.patheffects as pe
        h = self.herois[ano]
        self.ax.set_visible(True)
        for a in self.textos:
            a.set_visible(True)
        for k, im in self.cartoes.items():
            im.set_visible(k == ano)
        im = self.cartoes[ano]

        def rampa(a, b):
            return min(max((f - a) / (b - a), 0.0), 1.0)

        def pop(x):                                       # ease-out-back
            c = 1.70158
            return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2

        saida = rampa(0.86, 1.0)                          # 0 -> 1 nos últimos 14%
        visivel = 1 - saida
        self.escuro.set_alpha(0.80 * min(rampa(0.0, 0.10), visivel))

        # Cartão: entra GRANDE e assenta com "pancada"; na saída escorrega para a esquerda
        escala = 1.0 + 0.30 * (1 - pop(rampa(0.0, 0.18)))
        cx = self.CENTRO[0] - 0.9 * saida ** 2
        larg = self.LARGURA * escala
        alt = larg * im.proporcao_tela
        # a BASE do cartão fica sempre logo acima do carimbo (retrato ou foto horizontal de time)
        cy = 0.375 + self.LARGURA * im.proporcao_tela / 2
        im.set_extent((cx - larg / 2, cx + larg / 2, cy - alt / 2, cy + alt / 2))
        im.set_alpha(min(1.0, rampa(0.0, 0.06) * 1.5) * visivel)
        self.flash.set_alpha(0.45 * (1 - rampa(0.12, 0.26)) if f >= 0.12 else 0.0)

        # Carimbo "CAMPEÃO <ano>" bate depois do cartão
        self.t_carimbo.set_text(f"CAMPEÃO {ano}")
        self.t_carimbo.set_fontsize(64 * (1.0 + 0.6 * (1 - pop(rampa(0.14, 0.30)))))
        self.t_carimbo.set_alpha(rampa(0.14, 0.18) * visivel)
        self.t_nome.set_text(h.jogador.upper())
        # branco com contorno duplo (escuro + cor do clube): legível até com grená ou preto
        cor_clube = h.cores[0] if h.cores[0].upper() not in ("#F2F2F2", "#FFFFFF") else h.cores[1]
        self.t_nome.set_path_effects([pe.Stroke(linewidth=14, foreground=self.cor_fundo),
                                      pe.Stroke(linewidth=8, foreground=cor_clube), pe.Normal()])
        self.t_nome.set_alpha(rampa(0.20, 0.30) * visivel)
        self.t_nome.set_position((0.5 - 0.9 * saida ** 2, 0.262 - 0.02 * (1 - rampa(0.20, 0.30))))
        self.t_motivo.set_text(f"{h.clube} · {h.motivo}")
        self.t_motivo.set_alpha(rampa(0.26, 0.36) * visivel)
        self.t_credito.set_text(h.credito)
        self.t_credito.set_position((cx, cy - alt / 2 - 0.006))
        self.t_credito.set_alpha(0.75 * rampa(0.10, 0.20) * visivel)

        # Confete: cai do topo, balançando, a partir do impacto
        t = max(0.0, f - 0.12)
        x = self._conf_x0 + 0.03 * np.sin(self._conf_fase + t * 14)
        y = self._conf_y0 - self._conf_v * t * 1.35
        cores = [h.cores[i] for i in self._conf_lado]
        self.confete.set_offsets(np.column_stack([x, y]) if f >= 0.12 else np.empty((0, 2)))
        self.confete.set_sizes(self._conf_tam * np.abs(np.sin(self._conf_fase + t * 9)) + 15)
        self.confete.set_facecolors(cores)
        self.confete.set_alpha(visivel)
