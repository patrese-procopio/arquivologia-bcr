"""
Arquivologia do Futebol — Bar Chart Race vertical (1080x1920) para Reels/TikTok/Shorts.

Pipeline:
    CSV tidy (temporada, clube, jogador, gols)
        -> validação de qualidade de dados
        -> matriz acumulada (wide + cumsum)
        -> gancho gerado a partir dos dados (os 2 líderes e a diferença entre eles)
        -> linha do tempo com ritmo por peso + duração-alvo (interpolação de valores e posições)
        -> renderização com artistas reaproveitados
        -> MP4 H.264 (yuv420p)

Lição aprendida (analytics do 1º post): a maioria saiu em 0:01. O 1º quadro era uma
"tabela" com texto pequeno. Agora o vídeo abre com um gancho tipográfico GRANDE e em
movimento desde o quadro 0, e a duração cai de ~56 s para ~30 s.

Uso:
    python bar_race.py --dados dados/exemplo_gols.csv --eventos dados/exemplo_eventos.csv
    python bar_race.py --dados dados/exemplo_gols.csv --preview      # render rápido em 540x960
    python bar_race.py --dados ... --duracao-alvo 30 --ano-foco 2019     # versão curta para TikTok
"""
from __future__ import annotations

import argparse
import logging
import tempfile
import textwrap
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # renderiza sem janela: necessário em servidor/automação (n8n, cron)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.collections import LineCollection, PatchCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

from avatares import gerar_avatar  # noqa: E402
from encarada import CenaEncarada, montar_painel  # noqa: E402

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
        "Flamengo": "#A3080E",        # vermelho sangue
        "Palmeiras": "#1E9E57",
    })
    # Fundo de torcida gerado por código (sem foto real: sem direito autoral nem de imagem)
    avatares_ativos: bool = True
    pasta_fotos: Path = Path("assets/fotos")
    torcida_ativa: bool = True
    intensidade_torcida: float = 0.22   # opacidade máxima, na base da tela
    cores_torcidas: dict[str, tuple[str, str]] = field(default_factory=lambda: {
        "Flamengo": ("#A3080E", "#000000"),    # rubro-negra (esquerda)
        "Palmeiras": ("#1E9E57", "#F2F2F2"),   # alviverde (direita)
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
    # --- ritmo para redes sociais ---
    duracao_alvo: float | None = None  # segundos totais; None = ritmo fixo (versão longa)
    seg_gancho: float = 2.4           # abertura com o gancho tipográfico (0 = sem gancho)
    ano_foco: int | None = None       # a partir deste ano o tempo corre 2x mais devagar (a parte que importa)
    peso_foco: float = 2.0
    seg_minimo_temporada: float = 0.12  # nenhuma temporada some rápido demais (sem "salto")
    barra_progresso: bool = True      # mostra quanto falta: aumenta a taxa de quem assiste até o fim


@dataclass(frozen=True)
class Gancho:
    """Abertura do vídeo: o "fim" da história primeiro, em letras grandes (padrão de retenção)."""
    titulo: str
    subtitulo: str
    protagonistas: tuple[str, str]      # rótulos "Jogador · SIG": líder e perseguidor
    valores: tuple[float, float]
    chamada: str = "Veja como chegamos aqui ↓"
    espelhar: tuple[bool, bool] = (False, False)   # vira a foto para os dois se olharem


def nome_curto(jogador: str, limite: int = 10) -> str:
    """Nome de tela: 'Raphael Veiga' -> 'Veiga'. Nome longo invade a lateral direita,
    que no TikTok fica embaixo dos botões de curtir/comentar."""
    return jogador if len(jogador) <= limite else jogador.split()[-1]


def gancho_dos_dados(acumulado: pd.DataFrame) -> Gancho:
    """Monta o gancho a partir do ranking FINAL: quem lidera, quem persegue e por quanto.

    Gerado dos dados (não escrito à mão): se o CSV for atualizado depois de um jogo,
    o gancho se atualiza junto, sem risco de o texto contradizer o gráfico.
    """
    final = acumulado.iloc[-1].sort_values(ascending=False, kind="stable")
    (r1, v1), (r2, v2) = list(final.items())[:2]
    n1, n2 = nome_curto(r1.split(" · ")[0]), nome_curto(r2.split(" · ")[0])
    diff = int(v1 - v2)
    if diff == 0:
        titulo, sub = "EMPATE NO TOPO", f"{n1} x {n2}: quem desempata?"
    else:
        faltam = diff + 1                         # empatar não basta: precisa passar
        titulo = f"FALTA{'M' if faltam > 1 else ''} {faltam} GO{'LS' if faltam > 1 else 'L'}"
        sub = f"para {n2} passar {n1}"
    return Gancho(titulo, sub, (r1, r2), (float(v1), float(v2)))


@dataclass(frozen=True)
class Textos:
    selo: str = "BATALHA DE ARTILHEIROS"
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
    modo: str = "corrida"       # "gancho" | "corrida"
    fase: float = 0.0           # 0 -> 1 dentro do gancho (anima a entrada dos elementos)
    progresso: float = 0.0      # 0 -> 1 no vídeo inteiro (barra de progresso)


def _calcular_posicoes(valores: np.ndarray, posicoes_anteriores: np.ndarray, top_n: int) -> np.ndarray:
    """
    Posição 0 = topo. Quem tem 0 gol fica fora da tela.

    Desempate esportivo: em caso de empate, quem já estava na frente continua
    na frente (= quem chegou primeiro àquela marca). np.lexsort ordena pela
    ÚLTIMA chave e usa as anteriores como desempate, e é estável: empates
    totais preservam a ordem original das colunas (alfabética).
    """
    ordem = np.lexsort((posicoes_anteriores, -valores))
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
    # Posições calculadas em sequência: cada temporada herda o desempate da anterior
    posicoes = [_calcular_posicoes(estados[0], np.arange(len(rotulos), dtype=float), cfg.top_n)]
    for v in estados[1:]:
        posicoes.append(_calcular_posicoes(v, posicoes[-1], cfg.top_n))

    duracoes = _duracoes_por_temporada(estados, temporadas, eventos, cfg)

    quadros: list[Quadro] = []
    n_gancho = round(cfg.seg_gancho * cfg.fps)
    for i in range(n_gancho):
        quadros.append(Quadro(estados[0], posicoes[0], temporadas[0], modo="gancho", fase=i / n_gancho))

    for i in range(1, len(estados)):
        v0, v1 = estados[i - 1], estados[i]
        p0, p1 = posicoes[i - 1], posicoes[i]
        passos = max(2, round(duracoes[i - 1] * cfg.fps))

        for t in _suavizar(np.linspace(0, 1, passos, endpoint=False)):
            quadros.append(Quadro(v0 + (v1 - v0) * t, p0 + (p1 - p0) * t, temporadas[i]))

    final = Quadro(estados[-1], posicoes[-1], temporadas[-1])
    quadros.extend([final] * round(cfg.seg_segurar_final * cfg.fps))
    # replace() cria um objeto por quadro: `[final] * n` seriam n referências ao MESMO objeto
    total = max(1, len(quadros) - 1)
    quadros = [replace(q, progresso=i / total) for i, q in enumerate(quadros)]
    log.info("Linha do tempo: %d quadros (%.1f s)", len(quadros), len(quadros) / cfg.fps)
    return quadros


def _duracoes_por_temporada(estados: list[np.ndarray], temporadas: list[int],
                            eventos: dict[int, str], cfg: ConfigVideo) -> np.ndarray:
    """Quanto tempo de tela cada temporada recebe.

    1. Cada temporada ganha um PESO: com gol = 1; sem gol = proporcional ao "parado";
       com legenda = x fator_evento; a partir do ano-foco = x peso_foco.
    2. Sem duração-alvo: peso x seg_periodo_ativo (ritmo fixo, versão longa).
    3. Com duração-alvo: os pesos são ESCALADOS para caber no tempo disponível.
       É o mesmo raciocínio de um orçamento: o total é fixo, os pesos dizem como repartir.
    """
    pesos = []
    for i in range(1, len(estados)):
        mudou = not np.array_equal(estados[i - 1], estados[i])
        peso = 1.0 if mudou else cfg.seg_periodo_parado / cfg.seg_periodo_ativo
        if temporadas[i] in eventos:
            peso *= cfg.fator_evento
        if cfg.ano_foco is not None and temporadas[i] >= cfg.ano_foco:
            peso *= cfg.peso_foco
        pesos.append(peso)
    pesos = np.array(pesos)

    if cfg.duracao_alvo is None:
        return pesos * cfg.seg_periodo_ativo
    disponivel = cfg.duracao_alvo - cfg.seg_gancho - cfg.seg_segurar_final
    if disponivel <= 0:
        raise ValueError("duração-alvo menor que gancho + pausa final")
    return np.maximum(cfg.seg_minimo_temporada, pesos * disponivel / pesos.sum())


# ---------------------------------------------------------------------------
# Saída: ffmpeg com fundo estático
# ---------------------------------------------------------------------------
class FFMpegComFundo(FFMpegWriter):
    """
    FFMpegWriter que recebe uma imagem de fundo como 2ª entrada do ffmpeg.

    O matplotlib envia só o gráfico (fundo transparente) e o ffmpeg faz a
    sobreposição em C, muito mais rápido do que misturar a transparência de
    uma imagem de tela inteira em Python a cada quadro.
    """

    def __init__(self, caminho_fundo: Path, **kwargs):
        super().__init__(**kwargs)
        self.caminho_fundo = caminho_fundo

    def _args(self):
        args = super()._args()
        depois_do_pipe = args.index("pipe:") + 1
        # A 2ª entrada precisa vir ANTES das opções de saída (codec, bitrate...)
        args[depois_do_pipe:depois_do_pipe] = ["-loop", "1", "-i", str(self.caminho_fundo)]
        return args


# ---------------------------------------------------------------------------
# Fundo: torcidas procedurais (silhuetas geradas por código)
# ---------------------------------------------------------------------------
def gerar_fundo_torcidas(cores_esq: tuple[str, str], cores_dir: tuple[str, str],
                         intensidade: float, semente: int = 7,
                         largura_px: int = 540, altura_px: int = 960) -> np.ndarray:
    """
    Desenha uma arquibancada em silhueta (cabeças, ombros, braços e bandeirões)
    e devolve uma imagem RGBA. Esquerda = torcida 1, direita = torcida 2.

    - Semente fixa: o fundo sai idêntico em toda execução (reprodutível).
    - Renderizado em resolução menor e ampliado: é fundo desfocado, não precisa de detalhe.
    - Opacidade cai de baixo para cima: forte na base, some antes do título.
    """
    rng = np.random.default_rng(semente)
    proporcao = altura_px / largura_px            # coordenadas: x em [0,1], y em [0,proporcao]
    fig = Figure(figsize=(largura_px / 100, altura_px / 100), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, proporcao)
    ax.set_axis_off()
    fig.patch.set_alpha(0)

    def cor_para(x: float) -> str:
        # transição suave no meio: perto do centro as torcidas se misturam
        p_esq = np.clip((0.56 - x) / 0.12, 0, 1)
        cores = cores_esq if rng.random() < p_esq else cores_dir
        return cores[0] if rng.random() < 0.6 else cores[1]

    pessoas, bracos, cores_bracos = [], [], []
    fileiras = 16
    for k in range(fileiras - 1, -1, -1):                 # de trás para frente
        r = 0.022 * (1 - 0.45 * k / fileiras)             # perspectiva: fundo menor
        y = 0.02 + k * 0.055
        x = rng.uniform(-r, r)
        while x < 1 + r:
            cor = cor_para(x)
            yy = y + rng.uniform(-0.25, 0.25) * r
            pessoas.append(Ellipse((x, yy + 0.4 * r), 2.8 * r, 2.4 * r, color=cor))   # ombros
            pessoas.append(Circle((x, yy + 1.9 * r), 0.85 * r, color=cor))          # cabeça
            if rng.random() < 0.3:                                                   # braços erguidos
                for lado in (-1, 1):
                    bracos.append([(x + lado * 1.1 * r, yy + 1.2 * r),
                                   (x + lado * rng.uniform(1.4, 2.2) * r, yy + rng.uniform(3.0, 4.0) * r)])
                    cores_bracos.append(cor)
            x += r * rng.uniform(2.2, 2.7)
    ax.add_collection(PatchCollection(pessoas, match_original=True))
    ax.add_collection(LineCollection(bracos, colors=cores_bracos, linewidths=3.2 * largura_px / 540, capstyle="round"))

    # Bandeirões listrados tremulando (onda senoidal)
    for x0, cores in [(0.10, cores_esq), (0.30, cores_esq), (0.70, cores_dir), (0.90, cores_dir)]:
        base = 0.55 + rng.uniform(-0.05, 0.05)
        topo = base + 0.42
        ax.plot([x0, x0], [base - 0.25, topo], color=cores[1] if cores[1] != "#000000" else cores[0], lw=2 * largura_px / 540)
        largura, altura, listras = 0.17, 0.16, 5
        lado = -1 if x0 > 0.5 else 1
        xs = np.linspace(0, largura, 30)
        onda = 0.018 * np.sin(xs / largura * 2 * np.pi + rng.uniform(0, 6))
        for i in range(listras):
            y_ini = topo - altura * i / listras
            y_fim = topo - altura * (i + 1) / listras
            contorno = ([(x0 + lado * a, y_ini + o) for a, o in zip(xs, onda)]
                        + [(x0 + lado * a, y_fim + o) for a, o in zip(xs[::-1], onda[::-1])])
            ax.add_patch(Polygon(contorno, color=cores[i % 2]))

    fig.canvas.draw()
    imagem = np.asarray(fig.canvas.buffer_rgba()).astype(float) / 255
    # Degradê de opacidade: linha 0 da imagem = topo da tela
    altura_rel = np.linspace(1, 0, imagem.shape[0])[:, None]          # 1 no topo, 0 na base
    fade = np.clip(1 - altura_rel / 0.62, 0, 1) ** 1.4                 # some acima de ~62% da altura
    imagem[..., 3] *= intensidade * fade
    return imagem


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
    AREA_GRAFICO = (0.07, 0.285, 0.76, 0.41)  # [x, y, largura, altura] em fração da figura

    def __init__(self, acumulado: pd.DataFrame, clube_por_rotulo: pd.Series,
                 eventos: dict[int, str], tema: Tema, cfg: ConfigVideo, textos: Textos,
                 caminho_mascote: Path | None, gancho: Gancho | None = None):
        self.rotulos = list(acumulado.columns)
        self.clubes = list(clube_por_rotulo)
        self.cores = [tema.cores_clubes[c] for c in clube_por_rotulo]
        self.eventos = eventos
        self.tema, self.cfg, self.textos = tema, cfg, textos
        self._xmax = 5.0
        self.fundo_rgb: np.ndarray | None = None

        plt.rcParams["font.family"] = list(tema.fontes)
        # Tamanho em polegadas é fixo (base 100 dpi); só o dpi muda no preview.
        # Assim fontes e proporções ficam idênticas no preview e no render final.
        self.fig = plt.figure(figsize=(cfg.largura_px / 100, cfg.altura_px / 100),
                              dpi=cfg.dpi, facecolor=tema.fundo)
        if tema.torcida_ativa:
            self._montar_fundo_torcidas()
        # facecolor "none": o gráfico fica transparente e a torcida aparece por trás
        self.ax = self.fig.add_axes(self.AREA_GRAFICO, facecolor="none")
        self._montar_cabecalho_e_rodape(caminho_mascote)
        self._montar_eixo()
        self._montar_barras()
        self.gancho = gancho
        self.cena: CenaEncarada | None = None
        if gancho is not None and cfg.seg_gancho > 0:
            self._montar_gancho(gancho)
        self.progresso = None
        if cfg.barra_progresso:
            self._montar_barra_progresso()

    # -- montagem estática ---------------------------------------------------
    def _montar_fundo_torcidas(self) -> None:
        """Gera a arquibancada UMA vez e reaproveita como imagem em todos os quadros (cache)."""
        cores = list(self.tema.cores_torcidas.values())
        esquerda, direita = cores[0], cores[-1]       # um clube só: a mesma torcida dos dois lados
        largura, altura = (int(v) for v in self.fig.canvas.get_width_height())
        torcida = gerar_fundo_torcidas(esquerda, direita, self.tema.intensidade_torcida,
                                       largura_px=largura, altura_px=altura)
        # Composição alpha "over": torcida por cima da cor de fundo -> imagem opaca
        cor_fundo = np.array(to_rgb(self.tema.fundo))
        alpha = torcida[..., 3:4]
        self.fundo_rgb = torcida[..., :3] * alpha + cor_fundo * (1 - alpha)
        # A figura vira transparente: o fundo entra depois, no ffmpeg
        self.fig.patch.set_alpha(0)

    def _montar_cabecalho_e_rodape(self, caminho_mascote: Path | None) -> None:
        t, tx, f = self.tema, self.textos, self.fig
        # Tudo que some durante o gancho fica nesta lista (liga/desliga em bloco)
        self.cabecalho: list = [
            f.text(0.07, 0.905, tx.selo, color=t.destaque, fontsize=28, fontweight="bold"),
            f.text(0.07, 0.868, tx.titulo, color=t.texto, fontsize=48, fontweight="bold"),
            f.text(0.07, 0.842, tx.subtitulo, color=t.texto_suave, fontsize=28),
        ]
        x = 0.07
        # legenda de cores só faz sentido com 2+ clubes (com um só, o título já diz qual é)
        for clube, cor in (t.cores_clubes.items() if len(t.cores_clubes) > 1 else []):
            caixa = FancyBboxPatch((x, 0.812), 0.022, 0.012, boxstyle="round,pad=0.002",
                                   transform=f.transFigure, color=cor)
            f.patches.append(caixa)
            self.cabecalho += [caixa, f.text(x + 0.03, 0.813, clube, color=t.texto, fontsize=26, va="bottom")]
            x += 0.26

        self.legenda = f.text(0.07, 0.772, "", color=t.laranja, fontsize=35,
                              fontweight="bold", va="center", linespacing=1.25)
        self.cabecalho.append(self.legenda)

        # Rodapé (assinatura): mascote + @ — vira marca d'água se o vídeo for repostado
        x_texto = 0.07
        if caminho_mascote and caminho_mascote.exists():
            img = OffsetImage(plt.imread(str(caminho_mascote)), zoom=0.30)
            f.add_artist(AnnotationBbox(img, (0.07, 0.175), xycoords="figure fraction",
                                        frameon=False, box_alignment=(0, 0.5)))
            x_texto = 0.20
        f.text(x_texto, 0.181, tx.perfil, color=t.destaque, fontsize=24,
               fontweight="bold", va="bottom")
        f.text(x_texto, 0.177, tx.fonte, color=t.texto_suave, fontsize=18, va="top")

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
        # Ano em branco e bem visível, mas ATRÁS das barras (zorder baixo):
        # se uma barra passar por cima, o dado continua legível
        self.ano = ax.text(0.98, 0.04, "", transform=ax.transAxes, ha="right", va="bottom",
                           color="#FFFFFF", alpha=0.85, fontsize=120, fontweight="bold", zorder=0.5)

    def _montar_barras(self) -> None:
        zeros = np.zeros(len(self.rotulos))
        fora = np.full(len(self.rotulos), self.cfg.top_n + 1.0)
        self.barras = self.ax.barh(fora, zeros, height=0.52, color=self.cores)
        self.nomes = [self.ax.text(0, 0, r, color=self.tema.texto, fontsize=27,
                                   fontweight="semibold", va="bottom", zorder=6) for r in self.rotulos]
        # Valor como anotação com deslocamento em PONTOS: fica sempre à direita do avatar,
        # independentemente da escala do eixo (que muda a cada quadro)
        desloc_valor = 60 if self.tema.avatares_ativos else 6
        self.valores = [self.ax.annotate("", xy=(0, 0), xytext=(desloc_valor, 0),
                                         textcoords="offset points", color=self.tema.texto,
                                         fontsize=31, fontweight="bold", va="center", zorder=5)
                        for _ in self.rotulos]
        self.avatares: list[AnnotationBbox | None] = [None] * len(self.rotulos)
        if self.tema.avatares_ativos:
            for i, (rotulo, clube, cor) in enumerate(zip(self.rotulos, self.clubes, self.cores)):
                jogador, sigla = rotulo.rsplit(" · ", 1)
                imagem = gerar_avatar(self.tema.pasta_fotos, jogador, sigla, cor)
                caixa = AnnotationBbox(OffsetImage(imagem, zoom=0.44), (0, 0), xybox=(3, 0),
                                       boxcoords="offset points", box_alignment=(0, 0.5),
                                       frameon=False, zorder=4)
                caixa.set_visible(False)
                self.ax.add_artist(caixa)
                self.avatares[i] = caixa

    def _montar_gancho(self, g: Gancho) -> None:
        """Abertura estilo "encarada de UFC" (cena reutilizável em encarada.py)."""
        cfg, t = self.cfg, self.tema
        escala = cfg.dpi / 100
        larg_px, alt_px = CenaEncarada.tamanho_px(round(cfg.largura_px * escala), round(cfg.altura_px * escala))
        paineis, cores, nomes = [], [], []
        for k, (rotulo, lado) in enumerate(zip(g.protagonistas, ("esquerda", "direita"))):
            jogador, sigla = rotulo.rsplit(" · ", 1)
            cor = self.cores[self.rotulos.index(rotulo)]
            paineis.append(montar_painel(t.pasta_fotos, jogador, sigla, cor, lado, larg_px, alt_px,
                                         espelhar=g.espelhar[k]))
            cores.append(cor)
            nomes.append(nome_curto(jogador))
        self.cena = CenaEncarada(tuple(paineis), tuple(cores), tuple(nomes), g.valores, g.titulo,
                                 g.subtitulo, g.chamada, cor_titulo=t.laranja, cor_texto=t.texto,
                                 cor_chamada=t.destaque, cor_fundo=t.fundo)
        self.cena.montar(self.fig)

    def _montar_barra_progresso(self) -> None:
        """Barra fina que enche ao longo do vídeo: quem vê que falta pouco, fica até o fim."""
        f, t = self.fig, self.tema
        x0, y0, larg, alt = 0.07, 0.232, 0.86, 0.0045
        f.patches.append(Rectangle((x0, y0), larg, alt, transform=f.transFigure, color=t.grade, zorder=1))
        self.progresso = Rectangle((x0, y0), 0, alt, transform=f.transFigure, color=t.destaque, zorder=2)
        f.patches.append(self.progresso)
        self._progresso_largura = larg

    # -- atualização por quadro ----------------------------------------------
    def _atualizar_escala(self, valores: np.ndarray, posicoes: np.ndarray) -> float:
        visiveis = valores[posicoes < self.cfg.top_n]
        folga = 1.40 if self.tema.avatares_ativos else 1.22   # espaço para avatar + número
        alvo = max(5.0, (visiveis.max() if visiveis.size else 0) * folga)
        self._xmax += (alvo - self._xmax) * 0.12   # suavização exponencial: sem "tremedeira"
        self.ax.set_xlim(0, self._xmax)
        return self._xmax

    def desenhar(self, q: Quadro):
        if self.progresso is not None:
            self.progresso.set_width(self._progresso_largura * q.progresso)
        no_gancho = q.modo == "gancho" and self.cena is not None
        if self.cena is not None:
            self.cena.set_visible(no_gancho)
        for a in self.cabecalho:
            a.set_visible(not no_gancho)
        self.ax.set_visible(not no_gancho)
        if no_gancho:
            self.cena.desenhar(q.fase)
            return []

        self._atualizar_escala(q.valores, q.posicoes)
        for barra, nome, valor, avatar, v, p in zip(self.barras, self.nomes, self.valores,
                                                    self.avatares, q.valores, q.posicoes):
            alpha = float(np.clip(self.cfg.top_n - p, 0, 1))  # fade ao entrar/sair
            visivel = alpha > 0
            # Otimização: artista com alpha=0 AINDA é desenhado pelo matplotlib.
            # Escondê-lo de verdade evita trabalho inútil (dezenas de jogadores fora do top).
            for artista in (barra, nome, valor, avatar):
                if artista is not None:
                    artista.set_visible(visivel)
            if not visivel:
                continue
            barra.set_y(p - 0.26)
            barra.set_width(v)
            barra.set_alpha(alpha)
            nome.set_position((0, p - 0.31))
            nome.set_alpha(alpha)
            valor.xy = (v, p)
            valor.set_text(f"{v:.0f}")
            valor.set_alpha(alpha)
            if avatar is not None:
                avatar.xy = (v, p)
                avatar.offsetbox.image.set_alpha(alpha)

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
        saida.parent.mkdir(parents=True, exist_ok=True)
        if self.fundo_rgb is None:
            writer = FFMpegWriter(
                fps=self.cfg.fps, codec="libx264", bitrate=8000,
                extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"],  # compatível com celular
            )
            anim.save(str(saida), writer=writer, dpi=self.cfg.dpi)
        else:
            with tempfile.TemporaryDirectory() as pasta:
                caminho_fundo = Path(pasta) / "fundo.png"
                plt.imsave(caminho_fundo, self.fundo_rgb)
                writer = FFMpegComFundo(
                    caminho_fundo, fps=self.cfg.fps, codec="libx264", bitrate=8000,
                    extra_args=["-filter_complex", "[1:v][0:v]overlay=shortest=1,format=yuv420p",
                                "-preset", "medium"],
                )
                anim.save(str(saida), writer=writer, dpi=self.cfg.dpi)
        log.info("Vídeo salvo em %s", saida)

    def exportar_capa(self, quadro_final: Quadro, saida: Path) -> None:
        for _ in range(60):          # deixa a escala convergir antes da foto
            self.desenhar(quadro_final)
        if self.fundo_rgb is None:
            self.fig.savefig(saida, dpi=self.cfg.dpi, facecolor=self.tema.fundo)
        else:
            self.fig.canvas.draw()
            frente = np.asarray(self.fig.canvas.buffer_rgba()).astype(float) / 255
            alpha = frente[..., 3:4]
            plt.imsave(saida, frente[..., :3] * alpha + self.fundo_rgb * (1 - alpha))
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
    # Textos por vídeo: o código não muda, só os parâmetros (configuração fora do código)
    textos_padrao = Textos()
    parser.add_argument("--selo", default=textos_padrao.selo)
    parser.add_argument("--titulo", default=textos_padrao.titulo)
    parser.add_argument("--subtitulo", default=textos_padrao.subtitulo)
    parser.add_argument("--fonte", default=textos_padrao.fonte, help="crédito das fontes no rodapé")
    parser.add_argument("--sem-torcida", action="store_true", help="desliga o fundo de torcidas")
    parser.add_argument("--sem-fotos", action="store_true", help="desliga os avatares dos jogadores")
    # Ritmo para redes sociais
    parser.add_argument("--duracao-alvo", type=float, help="duração total em segundos (ex.: 30)")
    parser.add_argument("--ano-foco", type=int, help="a partir deste ano o ritmo desacelera (ex.: 2019)")
    parser.add_argument("--sem-gancho", action="store_true", help="começa direto na corrida (versão longa)")
    parser.add_argument("--gancho-titulo", help="substitui o título gerado dos dados (ex.: 'FALTAM 3 GOLS')")
    parser.add_argument("--gancho-sub", help="substitui o subtítulo gerado dos dados")
    parser.add_argument("--sem-progresso", action="store_true", help="esconde a barra de progresso")
    parser.add_argument("--clubes", default="",
                        help="filtra clubes, separados por vírgula (ex.: 'Flamengo'). Vazio = todos")
    parser.add_argument("--espelhar", default="", help="espelha a foto: 'esquerda', 'direita' ou 'ambos'")
    args = parser.parse_args()

    # Fail fast: valida a entrada antes de qualquer processamento pesado
    if not args.dados.exists():
        parser.error(f"arquivo de dados não encontrado: {args.dados}")
    if args.eventos and not args.eventos.exists():
        parser.error(f"arquivo de eventos não encontrado: {args.eventos}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    tema = Tema(torcida_ativa=not args.sem_torcida, avatares_ativos=not args.sem_fotos)
    textos = Textos(selo=args.selo, titulo=args.titulo, subtitulo=args.subtitulo, fonte=args.fonte)
    cfg = ConfigVideo(fps=args.fps, top_n=args.top_n, dpi=50 if args.preview else 100,
                      duracao_alvo=args.duracao_alvo, ano_foco=args.ano_foco,
                      seg_gancho=0.0 if args.sem_gancho else ConfigVideo.seg_gancho,
                      barra_progresso=not args.sem_progresso)

    df = carregar_e_validar(args.dados, set(tema.cores_clubes))
    # Filtro por clube DEPOIS da validação: o CSV inteiro continua sendo checado
    clubes = [c.strip() for c in args.clubes.split(",") if c.strip()]
    if clubes:
        desconhecidos = set(clubes) - set(tema.cores_clubes)
        if desconhecidos:
            parser.error(f"clube(s) sem cor no tema: {sorted(desconhecidos)}")
        df = df[df["clube"].isin(clubes)]
        tema = replace(tema,
                       cores_clubes={c: tema.cores_clubes[c] for c in clubes},
                       cores_torcidas={c: tema.cores_torcidas[c] for c in clubes if c in tema.cores_torcidas})
        log.info("Filtro de clubes: %s (%d linhas)", ", ".join(clubes), len(df))
    acumulado, clube_por_rotulo = montar_acumulado(df, tema.siglas_clubes)
    eventos = carregar_eventos(args.eventos)
    quadros = montar_linha_do_tempo(acumulado, eventos, cfg)

    gancho = None
    if cfg.seg_gancho > 0:
        gancho = gancho_dos_dados(acumulado)
        gancho = replace(gancho, titulo=args.gancho_titulo or gancho.titulo,
                         subtitulo=args.gancho_sub or gancho.subtitulo,
                         espelhar=(args.espelhar in ("esquerda", "ambos"),
                                   args.espelhar in ("direita", "ambos")))
        log.info("Gancho: %s / %s", gancho.titulo, gancho.subtitulo)

    render = RenderizadorCorrida(acumulado, clube_por_rotulo, eventos, tema, cfg, textos,
                                 args.mascote, gancho)
    render.exportar_video(quadros, args.saida)
    render.exportar_capa(quadros[-1], args.saida.with_name(args.saida.stem + "_capa.png"))
    if gancho is not None:   # capa para TikTok/Shorts: o quadro da encarada, logo após o impacto
        i = round(cfg.seg_gancho * cfg.fps * 0.6)
        render.exportar_capa(quadros[i], args.saida.with_name(args.saida.stem + "_capa_gancho.png"))


if __name__ == "__main__":
    main()
