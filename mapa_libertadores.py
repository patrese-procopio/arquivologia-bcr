"""
Arquivologia do Futebol — Mapa animado dos campeões da Libertadores (1080x1920).

Pipeline:
    campeoes.csv + cidades.csv + america_do_sul.geojson
        -> validação (toda cidade de campeão precisa de coordenada)
        -> estados acumulados por ano (títulos por cidade e por país)
        -> linha do tempo: gancho inicial + 1 bloco por ano (interpolado) + pausa final
        -> renderização (mapa + detalhe do Rio da Prata + placar Brasil x Argentina)
           efeitos: bandeiras recortadas no país, flash/+1 a cada título, placar que "pulsa",
           câmera com zoom lento, fundo com vinheta e grade de pontos
        -> MP4 H.264 (CRF 17, tune animation)

Uso:
    python mapa_libertadores.py --preview
    python mapa_libertadores.py --saida saida/mapa_libertadores.mp4
    python mapa_libertadores.py --fps 60 --saida saida/mapa_60fps.mp4   # mais fluido, 2x mais lento
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
import textwrap
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.patches import ConnectionPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path as CaminhoMpl  # noqa: E402

from bandeiras import gerar_bandeira, proporcao  # noqa: E402
from bar_race import FFMpegComFundo  # noqa: E402   (reuso: mesmo truque de fundo via ffmpeg)

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
log = logging.getLogger("mapa")


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TemaMapa:
    fundo: str = "#0B1220"
    oceano: str = "#0E1830"
    terra: str = "#1A2540"
    fronteira: str = "#3A4D72"
    texto: str = "#E8EEF6"
    texto_suave: str = "#8A9BB4"
    destaque: str = "#3FC1E8"
    laranja: str = "#F59E2B"
    fontes: tuple[str, ...] = ("Montserrat", "Inter", "DejaVu Sans")
    cores_paises: dict[str, str] = field(default_factory=lambda: {
        "Brasil": "#F4C430",      # amarelo
        "Argentina": "#8CC8F0",   # celeste
        "Uruguai": "#4169E1",
        "Paraguai": "#E4572E",
        "Colômbia": "#FF8C42",
        "Chile": "#D7263D",
        "Equador": "#B084F5",
    })
    siglas: dict[str, str] = field(default_factory=lambda: {
        "Brasil": "BRA", "Argentina": "ARG", "Uruguai": "URU", "Paraguai": "PAR",
        "Colômbia": "COL", "Chile": "CHI", "Equador": "EQU",
    })


@dataclass(frozen=True)
class ConfigMapa:
    largura_px: int = 1080
    altura_px: int = 1920
    fps: int = 30
    dpi: int = 100
    seg_por_ano: float = 0.82
    fator_evento: float = 1.9
    seg_gancho: float = 1.4       # estado final parado: só o tempo de ler a pergunta
    seg_rebobinar: float = 0.8    # transição animada final -> 1960
    seg_final: float = 5.0
    extensao: tuple[float, float, float, float] = (-82.0, -34.0, -56.0, 13.5)   # lon0, lon1, lat0, lat1
    extensao_prata: tuple[float, float, float, float] = (-59.3, -55.5, -35.85, -33.95)
    zoom_camera: float = 0.05            # aproximação total da câmera ao longo do vídeo (5%)
    alpha_bandeira: tuple[float, float] = (0.26, 0.46)   # 1º título -> 25 títulos
    crf: int = 17                        # qualidade x264: menor = melhor (17 ~ visualmente sem perda)


@dataclass(frozen=True)
class TextosMapa:
    selo: str = "MAPA DA GLÓRIA ETERNA"
    titulo: str = "Campeões da Libertadores"
    subtitulo: str = "1960–2025 · a cidade de cada campeão"
    gancho_1: str = "BRASIL x ARGENTINA"
    gancho_2: str = "Quem tem mais Libertadores?"
    final_1: str = "25 x 25: empate histórico"
    final_2: str = "O Brasil não lidera desde 1964. Em 28/11, em Montevidéu, isso pode mudar."
    fonte: str = "Campeões: lista oficial, conferida em 2 fontes · Mapa: Natural Earth"
    perfil: str = "@arquivologiadofutebol"


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def carregar_dados(pasta: Path, tema: TemaMapa) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    campeoes = pd.read_csv(pasta / "libertadores_campeoes.csv")
    cidades = pd.read_csv(pasta / "geo" / "cidades.csv").set_index("cidade")
    geo = json.loads((pasta / "geo" / "america_do_sul.geojson").read_text(encoding="utf-8"))

    problemas = []
    anos = campeoes["temporada"].tolist()
    if anos != list(range(anos[0], anos[-1] + 1)):
        problemas.append("temporadas faltando ou fora de ordem")
    sem_coord = set(campeoes["cidade"]) - set(cidades.index)
    if sem_coord:
        problemas.append(f"cidades sem coordenada: {sorted(sem_coord)}")
    sem_cor = set(campeoes["pais"]) - set(tema.cores_paises)
    if sem_cor:
        problemas.append(f"países sem cor no tema: {sorted(sem_cor)}")
    if problemas:
        raise ValueError("Dados inválidos:\n- " + "\n- ".join(problemas))

    log.info("Dados OK: %d campeões (%d-%d), %d cidades, %d países no mapa",
             len(campeoes), anos[0], anos[-1], campeoes["cidade"].nunique(), len(geo["features"]))
    return campeoes, cidades, geo["features"]


def carregar_eventos(caminho: Path | None) -> dict[int, str]:
    if caminho is None or not caminho.exists():
        return {}
    ev = pd.read_csv(caminho)
    return dict(zip(ev["temporada"].astype(int), ev["texto"].astype(str)))


# ---------------------------------------------------------------------------
# Linha do tempo
# ---------------------------------------------------------------------------
@dataclass
class QuadroMapa:
    por_cidade: dict[str, float]
    por_pais: dict[str, float]
    ano: int | None
    clube: str = ""
    cidade_nova: str = ""
    pais_novo: str = ""
    pulso: float = 1.0          # 0 -> 1 (anel expandindo)
    modo: str = "ano"           # "gancho" | "rebobinar" | "ano" | "final"
    progresso: float = 0.0      # 0 -> 1 ao longo do vídeo inteiro (câmera)
    fase: float = 0.0           # 0 -> 1 dentro do bloco atual (pulsação do placar no final)


def montar_linha_do_tempo(campeoes: pd.DataFrame, eventos: dict[int, str],
                          cfg: ConfigMapa) -> list[QuadroMapa]:
    cidades = sorted(campeoes["cidade"].unique())
    paises = sorted(campeoes["pais"].unique())
    acum_c = {c: 0.0 for c in cidades}
    acum_p = {p: 0.0 for p in paises}
    totais_c = campeoes["cidade"].value_counts().to_dict()
    totais_p = campeoes["pais"].value_counts().to_dict()

    quadros: list[QuadroMapa] = []
    # Gancho: o estado FINAL por alguns segundos, com a pergunta (o "clímax" primeiro)
    final_c = {c: float(totais_c.get(c, 0)) for c in cidades}
    final_p = {p: float(totais_p.get(p, 0)) for p in paises}
    n_gancho = round(cfg.seg_gancho * cfg.fps)
    quadros += [QuadroMapa(final_c, final_p, None, modo="gancho", fase=i / n_gancho) for i in range(n_gancho)]

    # Rebobinar: bolhas encolhem e o ano corre de trás pra frente até o início.
    # Movimento contínuo = o olho não percebe "pausa" entre o gancho e a história.
    ano_ini, ano_fim = int(campeoes["temporada"].iloc[0]), int(campeoes["temporada"].iloc[-1])
    passos_rew = max(2, round(cfg.seg_rebobinar * cfg.fps))
    for i in range(passos_rew):
        t = i / (passos_rew - 1)
        resta = (1 - t) ** 2                                  # ease-in: acelera no fim
        quadros.append(QuadroMapa({c: v * resta for c, v in final_c.items()},
                                  {p: v * resta for p, v in final_p.items()},
                                  round(ano_ini + (ano_fim - ano_ini) * resta), modo="rebobinar"))

    for _, linha in campeoes.iterrows():
        ano, cidade, pais = int(linha["temporada"]), linha["cidade"], linha["pais"]
        segundos = cfg.seg_por_ano * (cfg.fator_evento if ano in eventos else 1.0)
        passos = max(2, round(segundos * cfg.fps))
        for i in range(passos):
            t = i / (passos - 1)
            crescimento = min(1.0, t / 0.45)                    # a bolha cresce no começo do ano
            suave = 1 - (1 - crescimento) ** 3                 # ease-out
            por_c = dict(acum_c)
            por_c[cidade] = acum_c[cidade] + suave
            por_p = dict(acum_p)
            por_p[pais] = acum_p[pais] + suave
            quadros.append(QuadroMapa(por_c, por_p, ano, linha["clube"], cidade, pais, pulso=t))
        acum_c[cidade] += 1
        acum_p[pais] += 1

    n_final = round(cfg.seg_final * cfg.fps)
    quadros += [QuadroMapa(dict(acum_c), dict(acum_p), ano_fim, modo="final", fase=i / n_final)
                for i in range(n_final)]
    # Cada quadro é um objeto próprio (nunca `[q] * n`: a lista teria n referências ao MESMO objeto)
    total = len(quadros) - 1
    quadros = [replace(q, progresso=i / total) for i, q in enumerate(quadros)]
    log.info("Linha do tempo: %d quadros (%.1f s)", len(quadros), len(quadros) / cfg.fps)
    return quadros


# ---------------------------------------------------------------------------
# Renderização
# ---------------------------------------------------------------------------
def _aneis(geometria: dict) -> list[np.ndarray]:
    """Extrai os anéis externos de Polygon/MultiPolygon do GeoJSON."""
    if geometria["type"] == "Polygon":
        return [np.array(geometria["coordinates"][0])]
    return [np.array(poligono[0]) for poligono in geometria["coordinates"]]


def gerar_fundo(largura: int, altura: int, tema: TemaMapa) -> np.ndarray:
    """Fundo estático: vinheta radial + brilho de refletor no topo + grade de pontos discreta.

    Regra de ouro para fundo de dataviz: contraste BAIXO e nenhum movimento.
    O olho vai para o que muda; se o fundo não muda, ele não compete com os dados.
    """
    yy, xx = np.mgrid[0:altura, 0:largura].astype(float)
    u, v = xx / largura, yy / altura
    base = np.array(matplotlib.colors.to_rgb(tema.fundo))
    centro = np.array(matplotlib.colors.to_rgb("#16284A"))
    borda = np.array(matplotlib.colors.to_rgb("#05080F"))
    d = np.sqrt(((u - 0.5) / 0.75) ** 2 + ((v - 0.55) / 0.60) ** 2)
    t_centro = np.clip(1 - d, 0, 1)[..., None] ** 1.5
    t_borda = np.clip(d - 0.55, 0, 1)[..., None]
    img = base + (centro - base) * t_centro + (borda - base) * t_borda
    # brilho difuso de "refletor de estádio" vindo do topo
    brilho = np.exp(-(((u - 0.80) / 0.35) ** 2 + ((v + 0.05) / 0.22) ** 2))[..., None]
    img += np.array(matplotlib.colors.to_rgb(tema.destaque)) * brilho * 0.10
    # grade de pontos (estética de "prancheta de análise")
    passo = max(8, round(largura / 30))
    pontos = ((xx % passo) < passo * 0.09) & ((yy % passo) < passo * 0.09)
    img[pontos] += 0.045
    return np.clip(img, 0, 1)


def _suave(x: float) -> float:
    """smoothstep: começa e termina devagar (movimento de câmera natural)."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class RenderizadorMapa:
    AREA_MAPA = (0.02, 0.15, 0.80, 0.545)
    AREA_PRATA = (0.50, 0.165, 0.48, 0.20)

    def __init__(self, campeoes: pd.DataFrame, cidades: pd.DataFrame, paises_geo: list[dict],
                 eventos: dict[int, str], tema: TemaMapa, cfg: ConfigMapa, textos: TextosMapa,
                 caminho_mascote: Path | None):
        self.tema, self.cfg, self.textos, self.eventos = tema, cfg, textos, eventos
        self.cidades = cidades
        self.pais_da_cidade = campeoes.drop_duplicates("cidade").set_index("cidade")["pais"].to_dict()
        self.cidades_campeas = sorted(self.pais_da_cidade)

        plt.rcParams["font.family"] = list(tema.fontes)
        self.fig = plt.figure(figsize=(cfg.largura_px / 100, cfg.altura_px / 100),
                              dpi=cfg.dpi, facecolor=tema.fundo)
        escala = cfg.dpi / 100
        # O fundo NÃO é desenhado pelo matplotlib: a figura fica transparente e o ffmpeg
        # sobrepõe o vídeo a esta imagem (em C), economizando ~45 ms por quadro.
        self.fundo_rgb = gerar_fundo(round(cfg.largura_px * escala), round(cfg.altura_px * escala), tema)
        self.fig.patch.set_alpha(0)
        self.contorno = [pe.withStroke(linewidth=4, foreground=tema.fundo)]   # texto legível sobre bandeira
        self._montar_mapa(paises_geo)
        self._montar_prata(paises_geo)
        self._montar_textos(caminho_mascote)

    # -- montagem ---------------------------------------------------------------
    def _eixo_geografico(self, area, extensao, lat_ref: float):
        ax = self.fig.add_axes(area, facecolor="none")
        lon0, lon1, lat0, lat1 = extensao
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(lat0, lat1)
        ax.set_aspect(1 / np.cos(np.radians(lat_ref)))   # equirretangular corrigida na latitude média
        ax.set_anchor("W")                                 # encosta à esquerda: Atlântico livre à direita
        ax.set_xticks([])
        ax.set_yticks([])
        for lado in ax.spines.values():
            lado.set_visible(False)
        return ax

    def _desenhar_paises(self, ax, paises_geo, guardar: bool) -> dict[str, list[Polygon]]:
        patches: dict[str, list[Polygon]] = {}
        for feat in paises_geo:
            nome = feat["properties"]["nome"]
            for anel in _aneis(feat["geometry"]):
                p = Polygon(anel, closed=True, facecolor=self.tema.terra,
                            edgecolor=self.tema.fronteira, linewidth=1.2, zorder=1)
                ax.add_patch(p)
                if guardar:
                    patches.setdefault(nome, []).append(p)
        return patches

    def _montar_mapa(self, paises_geo) -> None:
        t = self.tema
        self.ax = self._eixo_geografico(self.AREA_MAPA, self.cfg.extensao, lat_ref=-20)
        self.patches_mapa = self._desenhar_paises(self.ax, paises_geo, guardar=True)

        # Grade de latitude/longitude (graticule): referência cartográfica bem discreta
        for lon in range(-90, -20, 10):
            self.ax.plot([lon, lon], [-70, 20], color=t.destaque, alpha=0.09, linewidth=0.9, zorder=0.5)
        for lat in range(-60, 20, 10):
            self.ax.plot([-100, -20], [lat, lat], color=t.destaque, alpha=0.09, linewidth=0.9, zorder=0.5)

        self._montar_bandeiras()
        # Fronteiras redesenhadas POR CIMA das bandeiras (senão a bandeira apaga a divisa)
        for patches in self.patches_mapa.values():
            for p in patches:
                self.ax.add_patch(Polygon(p.get_xy(), closed=True, fill=False, edgecolor=t.fronteira,
                                          linewidth=1.3, zorder=2))
        # imshow mexe em limites e aspecto: restauramos a geografia
        lon0, lon1, lat0, lat1 = self.cfg.extensao
        self.ax.set_xlim(lon0, lon1)
        self.ax.set_ylim(lat0, lat1)
        self.ax.set_aspect(1 / np.cos(np.radians(-20)))

        fora = self.cidades[self.cidades["detalhe_prata"] == "nao"]
        self.cid_mapa = [c for c in fora.index if c in self.pais_da_cidade]
        self.bolhas = self.ax.scatter(
            [fora.loc[c, "lon"] for c in self.cid_mapa], [fora.loc[c, "lat"] for c in self.cid_mapa],
            s=np.zeros(len(self.cid_mapa)), c=[t.cores_paises[self.pais_da_cidade[c]] for c in self.cid_mapa],
            edgecolors="white", linewidths=2.0, zorder=5)
        self.rotulos_mapa = {
            c: self.ax.annotate("", (fora.loc[c, "lon"], fora.loc[c, "lat"]),
                                xytext=(fora.loc[c, "dx"], fora.loc[c, "dy"]), textcoords="offset points",
                                ha="left" if fora.loc[c, "dx"] >= 0 else "right", va="center",
                                color=t.texto, fontsize=15, fontweight="bold", zorder=6,
                                path_effects=self.contorno)
            for c in self.cid_mapa}

        # Moldura do Rio da Prata + linhas até o detalhe ampliado
        lon0, lon1, lat0, lat1 = self.cfg.extensao_prata
        self.ax.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, fill=False,
                                    edgecolor=t.destaque, linewidth=1.6, zorder=7))
        self.pulso_mapa = self.ax.scatter([], [], s=[], facecolors="none", edgecolors="white",
                                          linewidths=2.5, zorder=8)
        self.mais1_mapa = self._criar_mais1(self.ax)

    def _criar_mais1(self, ax):
        """Texto "+1" que sobe e some sobre a cidade campeã (feedback visual de "ponto marcado")."""
        return ax.annotate("+1", (0, 0), xytext=(0, 0), textcoords="offset points", ha="center",
                           va="bottom", fontsize=24, fontweight="bold", color="white", zorder=9,
                           visible=False, path_effects=self.contorno, annotation_clip=False)

    def _montar_bandeiras(self) -> None:
        """Uma bandeira por país campeão, recortada pelo contorno do país (clip_path).

        Encaixe "cover": a bandeira mantém a proporção oficial e cobre toda a caixa do país,
        como o `background-size: cover` do CSS. O que sobra fica fora do recorte.
        """
        k = 1 / np.cos(np.radians(-20))          # mesma correção de aspecto do eixo
        # pixels de tela por unidade de mapa (o eixo é limitado pela altura OU pela largura)
        lon0, lon1, lat0, lat1 = self.cfg.extensao
        escala = self.cfg.dpi / 100
        px_por_unidade = min(self.AREA_MAPA[2] * self.cfg.largura_px * escala / (lon1 - lon0),
                             self.AREA_MAPA[3] * self.cfg.altura_px * escala / ((lat1 - lat0) * k))
        self.bandeiras = {}
        for pais in self.tema.cores_paises:
            patches = self.patches_mapa.get(pais)
            if not patches:
                continue
            xy = np.vstack([p.get_xy() for p in patches])
            (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
            larg, alt = x1 - x0, (y1 - y0) * k     # tamanho na tela (unidades iguais)
            razao = proporcao(pais)
            if larg / alt > razao:
                alt = larg / razao
            else:
                larg = alt * razao
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            # Chile é estreito: ancorar à esquerda mostra o quadrado azul com a estrela
            esq = x0 - 0.5 if pais == "Chile" else cx - larg / 2
            extent = (esq, esq + larg, cy - alt / k / 2, cy + alt / k / 2)
            folga_zoom = 1 / (1 - self.cfg.zoom_camera)   # a câmera aproxima: um pouco mais de resolução
            rgba = gerar_bandeira(pais, altura=round(alt * px_por_unidade * folga_zoom))
            # Máscara RASTER calculada uma única vez: cada pixel da bandeira está dentro do país?
            # (clip_path vetorial faria esse teste a cada quadro: ~110 ms/quadro a mais)
            rgba[..., 3] *= self._mascara_pais(patches, extent, rgba.shape[:2])
            im = self.ax.imshow(rgba, extent=extent, origin="upper", aspect="auto",
                                interpolation="bilinear", zorder=1.5, alpha=0.0)
            self.bandeiras[pais] = im

    @staticmethod
    def _mascara_pais(patches, extent, forma) -> np.ndarray:
        """1.0 onde o pixel cai dentro do contorno do país, 0.0 fora (teste ponto-no-polígono)."""
        alt, larg = forma
        x0, x1, y0, y1 = extent
        xs = x0 + (np.arange(larg) + 0.5) / larg * (x1 - x0)
        ys = y1 - (np.arange(alt) + 0.5) / alt * (y1 - y0)       # origin="upper": linha 0 = topo
        gx, gy = np.meshgrid(xs, ys)
        pontos = np.column_stack([gx.ravel(), gy.ravel()])
        dentro = np.zeros(len(pontos), dtype=bool)
        for p in patches:
            dentro |= CaminhoMpl(p.get_xy()).contains_points(pontos)
        return dentro.reshape(alt, larg).astype(float)

    def _montar_prata(self, paises_geo) -> None:
        t = self.tema
        self.ax_prata = self._eixo_geografico(self.AREA_PRATA, self.cfg.extensao_prata, lat_ref=-34.8)
        for lado in self.ax_prata.spines.values():
            lado.set_visible(True)
            lado.set_edgecolor(t.destaque)
            lado.set_linewidth(1.6)
        # O detalhe guarda os próprios polígonos: as cores acompanham o mapa principal
        self.patches_prata = self._desenhar_paises(self.ax_prata, paises_geo, guardar=True)

        # Linhas ligando a moldura no mapa aos cantos do detalhe ampliado
        lon0, lon1, lat0, lat1 = self.cfg.extensao_prata
        for lat in (lat1, lat0):
            self.fig.add_artist(ConnectionPatch(xyA=(lon1, lat), coordsA=self.ax.transData,
                                                xyB=(lon0, lat), coordsB=self.ax_prata.transData,
                                                color=t.destaque, linewidth=1.0, alpha=0.6))
        self.ax_prata.set_facecolor(t.oceano)
        self.ax_prata.text(1.0, 1.02, "Rio da Prata (ampliado)", transform=self.ax_prata.transAxes,
                           color=t.destaque, fontsize=15, fontweight="bold", va="bottom", ha="right", zorder=9)

        prata = self.cidades[self.cidades["detalhe_prata"] == "sim"]
        self.cid_prata = list(prata.index)
        xs, ys = prata["lon_detalhe"].to_numpy(), prata["lat_detalhe"].to_numpy()
        reais_x, reais_y = prata["lon"].to_numpy(), prata["lat"].to_numpy()
        # Marcador da posição real + linha fina até a bolha deslocada (deslocamento cartográfico)
        for rx, ry, x, y in zip(reais_x, reais_y, xs, ys):
            self.ax_prata.plot([rx, x], [ry, y], color=t.texto_suave, linewidth=0.8, zorder=4)
            self.ax_prata.plot(rx, ry, "o", color=t.texto, markersize=3, zorder=4)
        self.bolhas_prata = self.ax_prata.scatter(
            xs, ys, s=np.zeros(len(xs)), c=[t.cores_paises[self.pais_da_cidade[c]] for c in self.cid_prata],
            edgecolors="white", linewidths=1.5, zorder=5)
        self.rotulos_prata = {
            c: self.ax_prata.annotate("", (prata.loc[c, "lon_detalhe"], prata.loc[c, "lat_detalhe"]),
                                      xytext=(0, -24) if c == "Montevidéu" else (24, 0),
                                      textcoords="offset points",
                                      ha="center" if c == "Montevidéu" else "left",
                                      va="top" if c == "Montevidéu" else "center",
                                      color=t.texto, fontsize=15, fontweight="bold", zorder=6,
                                      path_effects=self.contorno)
            for c in self.cid_prata}
        self.pulso_prata = self.ax_prata.scatter([], [], s=[], facecolors="none", edgecolors="white",
                                                 linewidths=2.5, zorder=8)
        self.mais1_prata = self._criar_mais1(self.ax_prata)

    def _montar_textos(self, caminho_mascote: Path | None) -> None:
        t, tx, f = self.tema, self.textos, self.fig
        f.text(0.06, 0.935, tx.selo, color=t.destaque, fontsize=24, fontweight="bold")
        f.text(0.06, 0.902, tx.titulo, color=t.texto, fontsize=44, fontweight="bold")
        f.text(0.06, 0.878, tx.subtitulo, color=t.texto_suave, fontsize=24)

        # Placar Brasil x Argentina: o fio narrativo do vídeo
        cor_br, cor_ar = t.cores_paises["Brasil"], t.cores_paises["Argentina"]
        f.text(0.06, 0.832, "BRASIL", color=cor_br, fontsize=26, fontweight="bold", va="center")
        self.placar_br = f.text(0.45, 0.832, "0", color=cor_br, fontsize=56, fontweight="bold",
                                va="center", ha="right")
        f.text(0.50, 0.832, "x", color=t.texto_suave, fontsize=34, va="center", ha="center")
        self.placar_ar = f.text(0.55, 0.832, "0", color=cor_ar, fontsize=56, fontweight="bold",
                                va="center", ha="left")
        f.text(0.94, 0.832, "ARGENTINA", color=cor_ar, fontsize=26, fontweight="bold",
               va="center", ha="right")

        # Cartão do ano: ano + campeão + cidade
        self.card_ano = f.text(0.06, 0.778, "", color="#FFFFFF", fontsize=50, fontweight="bold", va="center")
        self.card_clube = f.text(0.30, 0.786, "", color=t.texto, fontsize=30, fontweight="bold", va="center")
        self.card_cidade = f.text(0.30, 0.766, "", color=t.texto_suave, fontsize=22, va="center")
        self.legenda = f.text(0.06, 0.738, "", color=t.laranja, fontsize=26, fontweight="bold",
                              va="center", linespacing=1.2)

        # Outros países (fora do duelo principal), no Atlântico ao lado do mapa
        outros = [p for p in t.cores_paises if p not in ("Brasil", "Argentina")]
        f.text(0.70, 0.672, "Outros países", color=t.texto_suave, fontsize=17, fontweight="bold")
        self.outros = {}
        for i, p in enumerate(outros):
            y = 0.650 - i * 0.022
            f.text(0.70, y, "■", color=t.cores_paises[p], fontsize=17, va="center")
            f.text(0.735, y, p, color=t.texto, fontsize=17, va="center")
            self.outros[p] = f.text(0.97, y, "0", color=t.texto, fontsize=17, fontweight="bold",
                                    va="center", ha="right")

        # Rodapé
        x_texto = 0.06
        if caminho_mascote and caminho_mascote.exists():
            img = OffsetImage(plt.imread(str(caminho_mascote)), zoom=0.30)
            f.add_artist(AnnotationBbox(img, (0.06, 0.082), xycoords="figure fraction",
                                        frameon=False, box_alignment=(0, 0.5)))
            x_texto = 0.19
        f.text(x_texto, 0.088, tx.perfil, color=t.destaque, fontsize=24, fontweight="bold", va="bottom")
        f.text(x_texto, 0.084, tx.fonte, color=t.texto_suave, fontsize=15, va="top")

    # -- atualização por quadro -------------------------------------------------
    @staticmethod
    def _tamanho(n: np.ndarray, escala: float) -> np.ndarray:
        """Área da bolha proporcional ao nº de títulos (raio ∝ √n: percepção honesta de área)."""
        return escala * n

    def desenhar(self, q: QuadroMapa):
        t = self.tema
        cfg = self.cfg
        titulo_agora = q.modo == "ano" and q.cidade_nova
        flash = max(0.0, 1 - q.pulso / 0.30) if titulo_agora else 0.0   # 1 -> 0 no início do ano

        # Câmera: zoom lento e contínuo (efeito Ken Burns) - dá vida sem distrair
        z = 1 - cfg.zoom_camera * _suave(q.progresso)
        lon0, lon1, lat0, lat1 = cfg.extensao
        cx, cy = (lon0 + lon1) / 2, (lat0 + lat1) / 2
        self.ax.set_xlim(cx - (lon1 - lon0) / 2 * z, cx + (lon1 - lon0) / 2 * z)
        self.ax.set_ylim(cy - (lat1 - lat0) / 2 * z, cy + (lat1 - lat0) / 2 * z)

        # Bandeiras: surgem no 1º título e ganham força com cada taça; piscam no ano da conquista
        a_min, a_max = cfg.alpha_bandeira
        for pais, im in self.bandeiras.items():
            n = q.por_pais.get(pais, 0.0)
            alpha = 0.0 if n <= 0 else min(1.0, n) * (a_min + (a_max - a_min) * min(n, 25) / 25)
            if pais == q.pais_novo:
                alpha += 0.25 * flash
            im.set_alpha(min(alpha, 0.95))

        # Detalhe do Rio da Prata: cor sólida (área pequena, bandeira viraria ruído)
        for pais, patches in self.patches_prata.items():
            n = q.por_pais.get(pais, 0.0)
            if n > 0:
                alpha = 0.20 + 0.55 * min(n, 25) / 25
                for p in patches:
                    p.set_facecolor(t.cores_paises[pais])
                    p.set_alpha(alpha)
            else:
                for p in patches:
                    p.set_facecolor(t.terra)
                    p.set_alpha(1.0)

        n_mapa = np.array([q.por_cidade.get(c, 0.0) for c in self.cid_mapa])
        self.bolhas.set_sizes(self._tamanho(n_mapa, 130))
        for c, n in zip(self.cid_mapa, n_mapa):
            self.rotulos_mapa[c].set_text(f"{c} {round(n)}" if n >= 0.5 else "")
        n_prata = np.array([q.por_cidade.get(c, 0.0) for c in self.cid_prata])
        self.bolhas_prata.set_sizes(self._tamanho(n_prata, 210))
        for c, n in zip(self.cid_prata, n_prata):
            self.rotulos_prata[c].set_text(f"{c} {round(n)}" if n >= 0.5 else "")

        # Anel de pulso na cidade campeã do ano
        for pulso in (self.pulso_mapa, self.pulso_prata):
            pulso.set_offsets(np.empty((0, 2)))
        if q.modo == "ano" and q.cidade_nova:
            linha = self.cidades.loc[q.cidade_nova]
            na_prata = linha["detalhe_prata"] == "sim"
            alvo = self.pulso_prata if na_prata else self.pulso_mapa
            xy = (linha["lon_detalhe"], linha["lat_detalhe"]) if na_prata else (linha["lon"], linha["lat"])
            alvo.set_offsets([xy])
            alvo.set_sizes([200 + 2600 * q.pulso])
            alvo.set_alpha(max(0.0, 1 - q.pulso))

        # "+1" subindo e sumindo sobre a cidade campeã
        for txt in (self.mais1_mapa, self.mais1_prata):
            txt.set_visible(False)
        if titulo_agora and q.pulso < 0.7:
            u = q.pulso / 0.7
            txt = self.mais1_prata if na_prata else self.mais1_mapa
            # sai para o lado OPOSTO ao rótulo da cidade, subindo pouco: não cruza outros nomes
            lado = 1 if na_prata or self.cidades.loc[q.cidade_nova, "dx"] < 0 else -1
            txt.xy = xy
            txt.xyann = (26 * lado, 6 + 26 * _suave(u))
            txt.set_color(t.cores_paises[q.pais_novo])
            txt.set_alpha(1 - u ** 2)
            txt.set_visible(True)

        br, ar = q.por_pais.get("Brasil", 0.0), q.por_pais.get("Argentina", 0.0)
        self.placar_br.set_text(f"{round(br)}")
        self.placar_ar.set_text(f"{round(ar)}")
        # Placar "pula" quando o número muda; no final, os dois pulsam juntos (empate!)
        esc_br = esc_ar = 1.0
        if titulo_agora and q.pulso < 0.35:
            pop = 1 + 0.35 * np.sin(np.pi * q.pulso / 0.35)
            esc_br = pop if q.pais_novo == "Brasil" else 1.0
            esc_ar = pop if q.pais_novo == "Argentina" else 1.0
        elif q.modo in ("final", "gancho"):
            esc_br = esc_ar = 1 + 0.07 * np.sin(2 * np.pi * q.fase * (6 if q.modo == "final" else 2))
        self.placar_br.set_fontsize(56 * esc_br)
        self.placar_ar.set_fontsize(56 * esc_ar)
        for p, texto in self.outros.items():
            texto.set_text(f"{round(q.por_pais.get(p, 0.0))}")

        if q.modo == "rebobinar":
            self.card_ano.set_text(str(q.ano))
            self.card_clube.set_text("voltando no tempo…")
            self.card_clube.set_color(t.texto_suave)
            self.card_cidade.set_text("")
            self.legenda.set_text("")
        elif q.modo == "gancho":
            self.card_ano.set_text("")
            self.card_clube.set_text("")
            self.card_cidade.set_text("")
            self.legenda.set_position((0.06, 0.770))
            self.legenda.set_text(f"{self.textos.gancho_1}\n{self.textos.gancho_2}")
        elif q.modo == "final":
            self.card_ano.set_text("")
            self.card_clube.set_text("")
            self.card_cidade.set_text("")
            self.legenda.set_position((0.06, 0.762))
            self.legenda.set_text(f"{self.textos.final_1}\n" + textwrap.fill(self.textos.final_2, 42))
        else:
            self.legenda.set_position((0.06, 0.733))
            self.card_ano.set_text(str(q.ano))
            self.card_clube.set_text(q.clube)
            self.card_clube.set_color(t.cores_paises[q.pais_novo])
            self.card_cidade.set_text(f"{q.cidade_nova} · {q.pais_novo}")
            self.legenda.set_text(self._legenda_ativa(q.ano))
        return []

    def _legenda_ativa(self, ano: int) -> str:
        anteriores = [a for a in self.eventos if a <= ano]
        if not anteriores or ano - max(anteriores) > 1:
            return ""
        return textwrap.fill(self.eventos[max(anteriores)], 40)

    # -- saída ------------------------------------------------------------------
    def exportar_video(self, quadros: list[QuadroMapa], saida: Path) -> None:
        anim = FuncAnimation(self.fig, lambda i: self.desenhar(quadros[i]), frames=len(quadros), blit=False)
        # CRF (qualidade constante) em vez de bitrate fixo: o x264 gasta bits onde há detalhe.
        # tune=animation: otimizado para áreas chapadas e bordas nítidas (nosso caso).
        # +faststart: o vídeo começa a tocar antes do download terminar (upload/preview mais rápidos).
        saida.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as pasta:
            caminho_fundo = Path(pasta) / "fundo.png"
            plt.imsave(caminho_fundo, self.fundo_rgb)
            writer = FFMpegComFundo(
                caminho_fundo, fps=self.cfg.fps, codec="libx264",
                extra_args=["-filter_complex", "[1:v][0:v]overlay=shortest=1,format=yuv420p",
                            "-preset", "slow", "-crf", str(self.cfg.crf), "-tune", "animation",
                            "-movflags", "+faststart"])
            anim.save(str(saida), writer=writer, dpi=self.cfg.dpi)
        log.info("Vídeo salvo em %s", saida)

    def exportar_capa(self, quadro: QuadroMapa, saida: Path) -> None:
        self.desenhar(quadro)
        # Na imagem estática o fundo entra via figimage (custo de 1 quadro só, não importa)
        fundo = self.fig.figimage(self.fundo_rgb, xo=0, yo=0, origin="upper", zorder=-10)
        try:
            self.fig.savefig(saida, dpi=self.cfg.dpi, facecolor=self.tema.fundo)
            log.info("Capa salva em %s", saida)
        except OSError as erro:   # ex.: arquivo aberto em outro programa no Windows
            log.warning("Capa não salva (%s). O vídeo está OK; feche o arquivo e rode de novo.", erro)
        finally:
            fundo.remove()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Mapa animado dos campeões da Libertadores")
    parser.add_argument("--dados", type=Path, default=Path("dados"), help="pasta com os CSVs e geo/")
    parser.add_argument("--eventos", type=Path, default=Path("dados/libertadores_campeoes_eventos.csv"))
    parser.add_argument("--saida", type=Path, default=Path("saida/mapa_libertadores.mp4"))
    parser.add_argument("--mascote", type=Path, default=Path("assets/mascote.png"))
    parser.add_argument("--preview", action="store_true", help="render rápido em 540x960")
    parser.add_argument("--fps", type=int, default=30, choices=(30, 60),
                        help="60 = animação mais fluida (render ~2x mais demorado)")
    args = parser.parse_args()
    if not args.dados.exists():
        parser.error(f"pasta de dados não encontrada: {args.dados}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    tema, textos = TemaMapa(), TextosMapa()
    cfg = ConfigMapa(dpi=50 if args.preview else 100, fps=args.fps)

    campeoes, cidades, paises = carregar_dados(args.dados, tema)
    eventos = carregar_eventos(args.eventos)
    quadros = montar_linha_do_tempo(campeoes, eventos, cfg)

    render = RenderizadorMapa(campeoes, cidades, paises, eventos, tema, cfg, textos, args.mascote)
    render.exportar_video(quadros, args.saida)
    render.exportar_capa(quadros[-1], args.saida.with_name(args.saida.stem + "_capa.png"))


if __name__ == "__main__":
    main()
