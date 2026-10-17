"""
Arquivologia do Futebol — Bar Chart Race vertical (1080x1920) para Reels/TikTok/Shorts.

Pipeline:
    CSV tidy (temporada, clube, jogador, gols)
        -> validação de qualidade de dados
        -> matriz acumulada (wide + cumsum)
        -> linha do tempo de quadros interpolados (valores + posições)
        -> renderização com artistas reaproveitados
        -> MP4 H.264 (yuv420p)

Uso:
    python bar_race.py --dados dados/exemplo_gols.csv --eventos dados/exemplo_eventos.csv
    python bar_race.py --dados dados/exemplo_gols.csv --preview      # render rápido em 540x960
"""
from __future__ import annotations

import argparse
import logging
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # renderiza sem janela: necessário em servidor/automação (n8n, cron)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
log = logging.getLogger("bar_race")

COLUNAS_OBRIGATORIAS = {"temporada", "clube", "jogador", "gols"}


# ---------------------------------------------------------------------------
# Configuração: tudo que é "decisão de design" fica aqui, longe da lógica.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tema:
    fundo: str = "#0B1220"            # azul-noite do logo
    texto: str = "#E8EEF6"
    texto_suave: str = "#8A9BB4"
    destaque: str = "#3FC1E8"         # ciano do escudo
    laranja: str = "#F59E2B"          # laranja do "FUTEBOL"
    grade: str = "#1C2A40"
    fontes: tuple[str, ...] = ("Montserrat", "Inter", "DejaVu Sans")
    cores_clubes: dict[str, str] = field(default_factory=lambda: {
        "Flamengo": "#D7262E",
        "Palmeiras": "#1E9E57",
    })
    siglas_clubes: dict[str, str] = field(default_factory=lambda: {
        "Flamengo": "FLA",
        "Palmeiras": "PAL",
    })


@dataclass(frozen=True)
class ConfigVideo:
    largura_px: int = 1080
    altura_px: int = 1920
    fps: int = 30
    top_n: int = 8
    seg_periodo_ativo: float = 1.1    # temporada em que algo muda
    seg_periodo_parado: float = 0.35  # temporada sem gols novos: passa rápido
    fator_evento: float = 1.8         # temporadas com legenda ficam mais lentas
    seg_segurar_final: float = 3.0    # pausa no ranking final (gancho para comentar)
    dpi: int = 100


@dataclass(frozen=True)
class Textos:
    selo: str = "ARQUIVO Nº 001"
    titulo: str = "Artilheiros na Libertadores"
    subtitulo: str = "Flamengo x Palmeiras · gols acumulados"
    fonte: str = "Fonte: dados de exemplo (fictícios)"
    perfil: str = "@arquivologiadofutebol"


# ---------------------------------------------------------------------------
# Camada de dados
# ---------------------------------------------------------------------------
class ErroValidacaoDados(ValueError):
    """Erro de qualidade de dados, com todos os problemas encontrados de uma vez."""


def carregar_e_validar(caminho: Path, clubes_conhecidos: set[str]) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    df.columns = [c.strip().lower() for c in df.columns]

    faltando = COLUNAS_OBRIGATORIAS - set(df.columns)
    if faltando:
        raise ErroValidacaoDados(f"Colunas ausentes: {sorted(faltando)}")

    df["jogador"] = df["jogador"].astype(str).str.strip()
    df["clube"] = df["clube"].astype(str).str.strip()

    problemas: list[str] = []
    if df[list(COLUNAS_OBRIGATORIAS)].isna().any().any():
        problemas.append("Há células vazias.")
    if not pd.api.types.is_integer_dtype(df["temporada"]):
        problemas.append("'temporada' precisa ser ano inteiro (ex.: 2019).")
    if not pd.api.types.is_integer_dtype(df["gols"]) or (df["gols"] < 0).any():
        problemas.append("'gols' precisa ser inteiro >= 0.")
    desconhecidos = set(df["clube"]) - clubes_conhecidos
    if desconhecidos:
        problemas.append(f"Clubes sem cor definida no tema: {sorted(desconhecidos)}")
    duplicados = df[df.duplicated(["temporada", "clube", "jogador"], keep=False)]
    if not duplicados.empty:
        problemas.append(
            "Linhas duplicadas (mesmo jogador/clube/temporada):\n"
            + duplicados.to_string(index=False)
        )
    if problemas:
        raise ErroValidacaoDados("\n- " + "\n- ".join(problemas))

    log.info("Dados OK: %d linhas, %d jogadores, temporadas %d-%d",
             len(df), df["jogador"].nunique(), df["temporada"].min(), df["temporada"].max())
    return df


def montar_acumulado(df: pd.DataFrame, siglas: dict[str, str]) -> tuple[pd.DataFrame, pd.Series]:
    """Tidy -> wide acumulado. Retorna (matriz temporada x rótulo, clube de cada rótulo)."""
    df = df.assign(rotulo=df["jogador"] + " · " + df["clube"].map(siglas))
    clube_por_rotulo = df.drop_duplicates("rotulo").set_index("rotulo")["clube"]

    temporadas = range(df["temporada"].min(), df["temporada"].max() + 1)
    acumulado = (
        df.pivot_table(index="temporada", columns="rotulo", values="gols",
                       aggfunc="sum", fill_value=0)
          .reindex(temporadas, fill_value=0)   # anos sem jogo continuam existindo no tempo
          .cumsum()
    )
    return acumulado, clube_por_rotulo.loc[acumulado.columns]


def carregar_eventos(caminho: Path | None) -> dict[int, str]:
    if caminho is None:
        return {}
    ev = pd.read_csv(caminho)
    return dict(zip(ev["temporada"].astype(int), ev["texto"].astype(str)))


# ---------------------------------------------------------------------------
# Linha do tempo: transforma N temporadas em milhares de quadros suaves
# ---------------------------------------------------------------------------
@dataclass
class Quadro:
    valores: np.ndarray
    posicoes: np.ndarray
    temporada: int


def _calcular_posicoes(valores: np.ndarray, rotulos: np.ndarray, top_n: int) -> np.ndarray:
    """Posição 0 = topo. Desempate estável pelo nome. Quem tem 0 gol fica fora da tela."""
    ordem = np.lexsort((rotulos, -valores))
    posicoes = np.empty(len(valores), dtype=float)
    posicoes[ordem] = np.arange(len(valores))
    posicoes[valores == 0] = top_n + 1
    return posicoes


def _suavizar(t: np.ndarray) -> np.ndarray:
    """Ease-in-out cúbico: começa e termina devagar, parece movimento físico."""
    return np.where(t < 0.5, 4 * t**3, 1 - (-2 * t + 2) ** 3 / 2)


def montar_linha_do_tempo(acumulado: pd.DataFrame, eventos: dict[int, str],
                          cfg: ConfigVideo) -> list[Quadro]:
    rotulos = acumulado.columns.to_numpy()
    zeros = np.zeros(len(rotulos))
    estados = [zeros] + [acumulado.loc[t].to_numpy(dtype=float) for t in acumulado.index]
    temporadas = [int(acumulado.index[0])] + [int(t) for t in acumulado.index]
    posicoes = [_calcular_posicoes(v, rotulos, cfg.top_n) for v in estados]

    quadros: list[Quadro] = []
    for i in range(1, len(estados)):
        v0, v1 = estados[i - 1], estados[i]
        p0, p1 = posicoes[i - 1], posicoes[i]
        segundos = cfg.seg_periodo_ativo if not np.array_equal(v0, v1) else cfg.seg_periodo_parado
        if temporadas[i] in eventos:
            segundos *= cfg.fator_evento
        passos = max(2, round(segundos * cfg.fps))

        for t in _suavizar(np.linspace(0, 1, passos, endpoint=False)):
            quadros.append(Quadro(v0 + (v1 - v0) * t, p0 + (p1 - p0) * t, temporadas[i]))

    final = Quadro(estados[-1], posicoes[-1], temporadas[-1])
    quadros.extend([final] * round(cfg.seg_segurar_final * cfg.fps))
    log.info("Linha do tempo: %d quadros (%.1f s)", len(quadros), len(quadros) / cfg.fps)
    return quadros


# ---------------------------------------------------------------------------
# Renderização
# ---------------------------------------------------------------------------
class RenderizadorCorrida:
    """
    Cria os artistas UMA vez e só atualiza posição/tamanho a cada quadro.
    Redesenhar o gráfico inteiro por quadro (ax.clear + barh) é o clichê;
    reaproveitar artistas é bem mais rápido em vídeos longos.
    """

    # Área segura: TikTok/Reels cobrem o topo, a base (legenda/botões) e a borda direita.
    AREA_GRAFICO = (0.07, 0.33, 0.76, 0.41)  # [x, y, largura, altura] em fração da figura

    def __init__(self, acumulado: pd.DataFrame, clube_por_rotulo: pd.Series,
                 eventos: dict[int, str], tema: Tema, cfg: ConfigVideo, textos: Textos,
                 caminho_mascote: Path | None):
        self.rotulos = list(acumulado.columns)
        self.cores = [tema.cores_clubes[c] for c in clube_por_rotulo]
        self.eventos = eventos
        self.tema, self.cfg, self.textos = tema, cfg, textos
        self._xmax = 5.0

        plt.rcParams["font.family"] = list(tema.fontes)
        # Tamanho em polegadas é fixo (base 100 dpi); só o dpi muda no preview.
        # Assim fontes e proporções ficam idênticas no preview e no render final.
        self.fig = plt.figure(figsize=(cfg.largura_px / 100, cfg.altura_px / 100),
                              dpi=cfg.dpi, facecolor=tema.fundo)
        self.ax = self.fig.add_axes(self.AREA_GRAFICO, facecolor=tema.fundo)
        self._montar_cabecalho_e_rodape(caminho_mascote)
        self._montar_eixo()
        self._montar_barras()

    # -- montagem estática ---------------------------------------------------
    def _montar_cabecalho_e_rodape(self, caminho_mascote: Path | None) -> None:
        t, tx, f = self.tema, self.textos, self.fig
        f.text(0.07, 0.905, tx.selo, color=t.destaque, fontsize=26, fontweight="bold")
        f.text(0.07, 0.868, tx.titulo, color=t.texto, fontsize=46, fontweight="bold")
        f.text(0.07, 0.842, tx.subtitulo, color=t.texto_suave, fontsize=27)

        x = 0.07
        for clube, cor in t.cores_clubes.items():
            f.patches.append(FancyBboxPatch((x, 0.812), 0.022, 0.012, boxstyle="round,pad=0.002",
                                            transform=f.transFigure, color=cor))
            f.text(x + 0.03, 0.813, clube, color=t.texto, fontsize=24, va="bottom")
            x += 0.26

        self.legenda = f.text(0.07, 0.772, "", color=t.laranja, fontsize=31,
                              fontweight="bold", va="center", linespacing=1.25)

        # Rodapé (assinatura): mascote + @ — vira marca d'água se o vídeo for repostado
        x_texto = 0.07
        if caminho_mascote and caminho_mascote.exists():
            img = OffsetImage(plt.imread(str(caminho_mascote)), zoom=0.30)
            f.add_artist(AnnotationBbox(img, (0.07, 0.285), xycoords="figure fraction",
                                        frameon=False, box_alignment=(0, 0.5)))
            x_texto = 0.20
        f.text(x_texto, 0.291, tx.perfil, color=t.destaque, fontsize=24,
               fontweight="bold", va="bottom")
        f.text(x_texto, 0.287, tx.fonte, color=t.texto_suave, fontsize=18, va="top")

    def _montar_eixo(self) -> None:
        ax, t = self.ax, self.tema
        ax.set_ylim(self.cfg.top_n - 0.4, -0.9)   # invertido: posição 0 no topo
        ax.set_yticks([])
        # Sem números no eixo: o valor já está na ponta de cada barra (menos poluição)
        ax.tick_params(axis="x", labelbottom=False, labeltop=False, length=0)
        ax.grid(axis="x", color=t.grade, linewidth=1.5)
        ax.set_axisbelow(True)
        for lado in ax.spines.values():
            lado.set_visible(False)
        self.ano = ax.text(0.98, 0.04, "", transform=ax.transAxes, ha="right", va="bottom",
                           color=t.texto, alpha=0.18, fontsize=130, fontweight="bold")

    def _montar_barras(self) -> None:
        zeros = np.zeros(len(self.rotulos))
        fora = np.full(len(self.rotulos), self.cfg.top_n + 1.0)
        self.barras = self.ax.barh(fora, zeros, height=0.52, color=self.cores)
        self.nomes = [self.ax.text(0, 0, r, color=self.tema.texto, fontsize=24,
                                   fontweight="semibold", va="bottom") for r in self.rotulos]
        self.valores = [self.ax.text(0, 0, "", color=self.tema.texto, fontsize=28,
                                     fontweight="bold", va="center") for _ in self.rotulos]

    # -- atualização por quadro ----------------------------------------------
    def _atualizar_escala(self, valores: np.ndarray, posicoes: np.ndarray) -> float:
        visiveis = valores[posicoes < self.cfg.top_n]
        alvo = max(5.0, (visiveis.max() if visiveis.size else 0) * 1.22)
        self._xmax += (alvo - self._xmax) * 0.12   # suavização exponencial: sem "tremedeira"
        self.ax.set_xlim(0, self._xmax)
        return self._xmax

    def desenhar(self, q: Quadro):
        xmax = self._atualizar_escala(q.valores, q.posicoes)
        respiro = xmax * 0.012
        for barra, nome, valor, v, p in zip(self.barras, self.nomes, self.valores,
                                            q.valores, q.posicoes):
            alpha = float(np.clip(self.cfg.top_n - p, 0, 1))  # fade ao entrar/sair
            barra.set_y(p - 0.26)
            barra.set_width(v)
            barra.set_alpha(alpha)
            nome.set_position((0, p - 0.31))
            nome.set_alpha(alpha)
            valor.set_position((v + respiro, p))
            valor.set_text(f"{v:.0f}")
            valor.set_alpha(alpha)

        self.ano.set_text(str(q.temporada))
        self.legenda.set_text(self._legenda_ativa(q.temporada))
        return []

    def _legenda_ativa(self, temporada: int) -> str:
        anteriores = [t for t in self.eventos if t <= temporada]
        if not anteriores:
            return ""
        ultimo = max(anteriores)
        if temporada - ultimo > 1:
            return ""
        return textwrap.fill(self.eventos[ultimo], width=34)  # quebra antes da borda direita

    # -- saída ---------------------------------------------------------------
    def exportar_video(self, quadros: list[Quadro], saida: Path) -> None:
        anim = FuncAnimation(self.fig, lambda i: self.desenhar(quadros[i]),
                             frames=len(quadros), blit=False)
        writer = FFMpegWriter(
            fps=self.cfg.fps, codec="libx264", bitrate=8000,
            extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"],  # compatível com celular
        )
        saida.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(saida), writer=writer, dpi=self.cfg.dpi)
        log.info("Vídeo salvo em %s", saida)

    def exportar_capa(self, quadro_final: Quadro, saida: Path) -> None:
        for _ in range(60):          # deixa a escala convergir antes da foto
            self.desenhar(quadro_final)
        self.fig.savefig(saida, dpi=self.cfg.dpi, facecolor=self.tema.fundo)
        log.info("Capa salva em %s", saida)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Bar chart race vertical do Arquivologia do Futebol")
    parser.add_argument("--dados", type=Path, required=True, help="CSV tidy: temporada,clube,jogador,gols")
    parser.add_argument("--eventos", type=Path, help="CSV opcional: temporada,texto")
    parser.add_argument("--saida", type=Path, default=Path("saida/corrida.mp4"))
    parser.add_argument("--mascote", type=Path, default=Path("assets/mascote.png"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--preview", action="store_true", help="render rápido em 540x960")
    args = parser.parse_args()
    if not args.dados.exists():
           parser.error(f"arquivo de dados não encontrado: {args.dados}")
    if args.eventos and not args.eventos.exists():
            parser.error(f"arquivo de eventos não encontrado: {args.eventos}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    tema, textos = Tema(), Textos()
    cfg = ConfigVideo(fps=args.fps, top_n=args.top_n, dpi=50 if args.preview else 100)

    df = carregar_e_validar(args.dados, set(tema.cores_clubes))
    acumulado, clube_por_rotulo = montar_acumulado(df, tema.siglas_clubes)
    eventos = carregar_eventos(args.eventos)
    quadros = montar_linha_do_tempo(acumulado, eventos, cfg)

    render = RenderizadorCorrida(acumulado, clube_por_rotulo, eventos, tema, cfg, textos, args.mascote)
    render.exportar_video(quadros, args.saida)
    render.exportar_capa(quadros[-1], args.saida.with_name(args.saida.stem + "_capa.png"))


if __name__ == "__main__":
    main()
