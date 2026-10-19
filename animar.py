"""Anima "O corpo do Neymar": 2010 -> 2026, uma lesão por vez.

Arquitetura:
  1. montar_roteiro()  -> lista de segmentos (intro, lesão, transferência, outro) com duração em frames
  2. desenhar_frame()  -> função pura (segmento, t) -> imagem; t vai de 0 a 1 dentro do segmento
  3. efeitos           -> intensidade sai de `dias_total` (motion design guiado por dados)
  4. main()            -> frames vão direto para o stdin do ffmpeg (sem gravar 1.200 PNGs no disco)
"""
from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

import corpo_neymar as cn

FPS = 30
LAY = cn.Layout()

# Gravidade -> tempo de tela. Lesão curta passa rápido; a grave para o vídeo.
DURACAO_POR_NIVEL = {1: 0.55, 2: 0.9, 3: 1.5, 4: 3.4}
DUR_INTRO, DUR_TRANSFER, DUR_OUTRO = 1.8, 0.8, 5.0
# Multiplicador global de ritmo (feedback da v1: "rápido demais"). Os efeitos são frações de t,
# então desaceleram junto, sem mexer em nenhuma outra linha.
RITMO = 1.25
ANO_CHEGADA = {"Santos": "2009", "Barcelona": "2013", "PSG": "2017", "Al-Hilal": "2023"}


def nivel(dias: float) -> int:
    """1: até 15 dias · 2: até 60 · 3: até 150 · 4: acima de 150."""
    return 1 if dias <= 15 else 2 if dias <= 60 else 3 if dias <= 150 else 4


# ---------- easing: curvas que dão "peso" ao movimento ----------
def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def ease_out_back(t: float) -> float:
    """Passa do ponto e volta: o "quique" de um carimbo."""
    t = min(max(t, 0.0), 1.0)
    c = 1.9
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2


@dataclass(frozen=True)
class Segmento:
    tipo: str            # intro | lesao | transferencia | outro
    inicio: int          # frame inicial
    frames: int
    idx: int = -1        # índice da lesão (para "lesao") ou da última lesão concluída
    clube_de: str = ""
    clube_para: str = ""


def montar_roteiro(lesoes: pd.DataFrame) -> list[Segmento]:
    segs: list[Segmento] = []
    cursor = 0

    def add(tipo: str, segundos: float, **kw) -> None:
        nonlocal cursor
        n = round(segundos * RITMO * FPS)
        segs.append(Segmento(tipo, cursor, n, **kw))
        cursor += n

    add("intro", DUR_INTRO, clube_para="Santos")
    clube = "Santos"
    for i, les in lesoes.iterrows():
        if les["clube"] != clube:
            add("transferencia", DUR_TRANSFER, idx=i - 1, clube_de=clube, clube_para=les["clube"])
            clube = les["clube"]
        add("lesao", DURACAO_POR_NIVEL[nivel(les["dias_total"])], idx=i, clube_para=clube)
    add("outro", DUR_OUTRO, idx=len(lesoes) - 1, clube_para=clube)
    return segs


class Contexto:
    """Pré-calcula acumulados uma vez; cada frame só consulta."""

    def __init__(self, lesoes: pd.DataFrame):
        self.lesoes = lesoes
        self.acum = lesoes["dias_total"].cumsum().tolist()
        self.maximo = lesoes.groupby("clube")["dias_total"].sum().max()
        self.total = len(lesoes)

    def dias_ate(self, idx: int) -> float:
        return self.acum[idx] if idx >= 0 else 0.0

    def por_clube_ate(self, idx: int) -> dict[str, float]:
        if idx < 0:
            return {}
        return self.lesoes.iloc[: idx + 1].groupby("clube")["dias_total"].sum().to_dict()


# ---------- camadas de texto especiais ----------
def texto_com_caixa(img: Image.Image, linhas: list[tuple[str, tuple, int]], centro_y: float,
                    alfa: float = 1.0, escala: float = 1.0) -> None:
    """Bloco de texto centralizado com tarja escura atrás (garante leitura sobre a foto)."""
    if alfa <= 0:
        return
    camada = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(camada)
    fontes = [cn.fonte(cn.FONTE_IMPACTO, max(int(tam * escala), 1)) for _, _, tam in linhas]
    alturas = [f.getbbox(txt)[3] - f.getbbox(txt)[1] for (txt, _, _), f in zip(linhas, fontes)]
    larguras = [f.getlength(txt) for (txt, _, _), f in zip(linhas, fontes)]
    gap = 18 * escala
    h_total = sum(alturas) + gap * (len(linhas) - 1)
    pad = 34 * escala
    w_max = max(larguras)
    y = centro_y - h_total / 2
    d.rounded_rectangle((cn.LARGURA / 2 - w_max / 2 - pad, y - pad, cn.LARGURA / 2 + w_max / 2 + pad,
                         y + h_total + pad), radius=int(26 * escala), fill=(8, 9, 13, int(225 * alfa)))
    for (txt, cor, _), f, h in zip(linhas, fontes, alturas):
        d.text((cn.LARGURA / 2, y + h / 2), txt, font=f, fill=(*cor, int(255 * alfa)), anchor="mm")
        y += h + gap
    img.alpha_composite(camada)


def banner_transferencia(img: Image.Image, clube: str, t: float) -> None:
    """Ocupa o lugar do cartão: escudo + 'AGORA NO <CLUBE> · ANO'."""
    camada = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(camada)
    dx = -1100 * (1 - ease_out(t / 0.35))
    y0 = LAY.cartao_topo
    cor = cn.CLUBES[clube]["cor"]
    h = LAY.cartao_altura
    d.rounded_rectangle((60 + dx, y0, cn.SEGURA_DIREITA + dx, y0 + h), radius=20, fill=(20, 22, 30, 240),
                        outline=cor, width=5)
    esc = Image.open(cn.ASSETS / "escudos" / cn.CLUBES[clube]["escudo"]).convert("RGBA")
    esc.thumbnail((84, 84))  # carrega do original: ampliar o escudo pequeno borraria
    camada.alpha_composite(esc, (int(95 + dx), int(y0 + (h - esc.height) / 2)))
    nome = "SANTOS DE NOVO" if clube == "Santos" else clube.upper()
    ano = "2025" if clube == "Santos" else ANO_CHEGADA[clube]
    d.text((210 + dx, y0 + 40), nome, font=cn.fonte(cn.FONTE_IMPACTO, 54), fill=cn.BRANCO, anchor="lm")
    d.text((210 + dx, y0 + 88), f"nova camisa em {ano}", font=cn.fonte(cn.FONTE_TEXTO_LEVE, 34),
           fill=cn.CINZA, anchor="lm")
    img.alpha_composite(camada)


# ---------- efeitos guiados por dados ----------
def onda_de_choque(img: Image.Image, pos: tuple[float, float], t: float, niv: int) -> None:
    if t > 0.4:
        return
    p = ease_out(t / 0.4)
    r = 14 + p * (50 + 30 * niv)
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse((pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r), outline=(255, 255, 255, int(230 * (1 - p))), width=6)


def mais_dias(img: Image.Image, dias: float, t: float) -> None:
    """'+91' vermelho subindo ao lado do contador."""
    if t > 0.75:
        return
    alfa = 1.0 if t < 0.5 else 1 - (t - 0.5) / 0.25
    y = cn.CONTADOR_Y + 20 - 50 * ease_out(t / 0.5)
    d = ImageDraw.Draw(img, "RGBA")
    d.text((850, y), f"+{cn.fmt_int(dias)}", font=cn.fonte(cn.FONTE_IMPACTO, 66),
           fill=(*cn.VERMELHO, int(255 * alfa)), anchor="mm")


def vinheta(img: Image.Image, forca: float) -> None:
    if forca <= 0:
        return
    mascara = vinheta_mascara()
    escuro = Image.new("RGBA", img.size, (0, 0, 0, 0))
    escuro.putalpha(mascara.point(lambda v: int(v * forca)))
    img.alpha_composite(escuro)


_VINHETA: list[Image.Image] = []


def vinheta_mascara() -> Image.Image:
    if not _VINHETA:
        m = Image.new("L", (cn.LARGURA, cn.ALTURA), 230)
        ImageDraw.Draw(m).ellipse((-150, 150, cn.LARGURA + 150, cn.ALTURA - 150), fill=0)
        _VINHETA.append(m.filter(ImageFilter.GaussianBlur(180)))
    return _VINHETA[0]


def flash(img: Image.Image, cor: tuple[int, int, int], alfa: float) -> None:
    if alfa > 0:
        img.alpha_composite(Image.new("RGBA", img.size, (*cor, int(alfa))))


def tremida_e_zoom(img: Image.Image, dx: float, dy: float, zoom: float) -> Image.Image:
    """Câmera: aplica zoom a partir do centro e desloca. Resolve tudo numa transformação afim."""
    if abs(dx) < 0.5 and abs(dy) < 0.5 and zoom <= 1.0005:
        return img
    w, h = img.size
    a = 1 / zoom
    cx, cy = w / 2, h / 2
    matriz = (a, 0, cx - a * cx - dx * a, 0, a, cy - a * cy - dy * a)
    return img.transform(img.size, Image.AFFINE, matriz, resample=Image.BILINEAR, fillcolor=(*cn.FUNDO, 255))


# ---------- o frame ----------
def base(ctx: Contexto, idx_concluido: int, clube: str, data: pd.Timestamp, contador: float,
         valores: dict[str, float], grupos: pd.DataFrame, rotulo: str,
         pop: dict | None = None, fundo: Image.Image | None = None) -> tuple[Image.Image, float, int]:
    img = (fundo if fundo is not None else cn.desenhar_fundo(clube, data)).convert("RGBA")
    cn.desenhar_contador(ImageDraw.Draw(img), contador, rotulo)
    escala, x0 = cn.colar_foto(img, LAY)
    if len(grupos):
        cn.desenhar_marcadores(img, grupos, escala, x0, LAY, pop)
    cn.desenhar_placar(img, valores, clube, LAY, ctx.maximo)
    return img, escala, x0


def desenhar_frame(ctx: Contexto, seg: Segmento, t: float, frame: int) -> Image.Image:
    L = ctx.lesoes
    if seg.tipo == "intro":
        img, _, _ = base(ctx, -1, "Santos", pd.Timestamp("2009-06-01"), 0, {}, L.iloc[:0], f"{ctx.total} LESÕES. VAMOS CONTAR?")
        a = ease_out(t / 0.15)
        texto_com_caixa(img, [("QUANTOS DIAS O NEYMAR", cn.BRANCO, 78), ("PASSOU MACHUCADO?", cn.VERMELHO, 88)],
                        770, alfa=a, escala=0.85 + 0.15 * ease_out_back(t / 0.25))
        return img.convert("RGB")

    if seg.tipo == "transferencia":
        les = L.iloc[seg.idx]
        f_de = cn.desenhar_fundo(seg.clube_de, les["inicio"])
        data_para = pd.Timestamp(f"{'2025' if seg.clube_para == 'Santos' else ANO_CHEGADA[seg.clube_para]}-06-01")
        f_para = cn.desenhar_fundo(seg.clube_para, data_para)
        fundo = Image.blend(f_de, f_para, ease_out(t / 0.6))  # o fundo "troca de camisa"
        grupos = cn.agrupar_marcadores(L.iloc[: seg.idx + 1])
        img, _, _ = base(ctx, seg.idx, seg.clube_para, data_para, ctx.dias_ate(seg.idx),
                         ctx.por_clube_ate(seg.idx), grupos, f"LESÃO {seg.idx + 1} DE {ctx.total}", fundo=fundo)
        banner_transferencia(img, seg.clube_para, t)
        return img.convert("RGB")

    if seg.tipo == "outro":
        grupos = cn.agrupar_marcadores(L)
        ult = L.iloc[-1]
        img, _, _ = base(ctx, seg.idx, ult["clube"], ult["inicio"], ctx.dias_ate(seg.idx),
                         ctx.por_clube_ate(seg.idx), grupos, f"{ctx.total} LESÕES")
        vinheta(img, min(t / 0.15, 1) * 0.55)
        anos = ctx.dias_ate(seg.idx) / 365.25
        if t > 0.06:
            p = (t - 0.06) / 0.12
            texto_com_caixa(img, [(f"{anos:.1f} ANOS".replace(".", ","), cn.VERMELHO, 165),
                                  ("NO DEPARTAMENTO MÉDICO", cn.BRANCO, 64)],
                            740, alfa=min(p * 2, 1), escala=0.6 + 0.4 * ease_out_back(p))
        if t > 0.42:
            p = ease_out((t - 0.42) / 0.12)
            texto_com_caixa(img, [("QUAL LESÃO MAIS DOEU EM VOCÊ?", cn.BRANCO, 46), ("COMENTA AÍ", cn.VERMELHO, 46)],
                            1060, alfa=p)
        d = ImageDraw.Draw(img)
        d.text((cn.LARGURA // 2, LAY.cartao_topo + 60), "Fontes: Transfermarkt · Soccerway · O Globo · Estadão · TMC",
               font=cn.fonte(cn.FONTE_TEXTO_LEVE, 30), fill=cn.CINZA, anchor="mm")
        img = img.convert("RGB")
        if 0.06 < t < 0.14:  # impacto do carimbo
            img = tremida_e_zoom(img.convert("RGBA"), *_tremor(frame, 18 * (1 - (t - 0.06) / 0.08)), 1.0).convert("RGB")
        return img

    # ---- lesão ----
    les = L.iloc[seg.idx]
    niv = nivel(les["dias_total"])
    tc = ease_out(t / 0.7)  # contador e barra "rolam" nos primeiros 70% do segmento
    contador = ctx.dias_ate(seg.idx - 1) + les["dias_total"] * tc
    valores = ctx.por_clube_ate(seg.idx - 1)
    valores[les["clube"]] = valores.get(les["clube"], 0) + les["dias_total"] * tc
    grupos = cn.agrupar_marcadores(L.iloc[: seg.idx + 1])
    chave = (les["regiao"], les["lado"])
    pop = {chave: 1 + 1.4 * (1 - ease_out(t / 0.25))}
    img, escala, x0 = base(ctx, seg.idx, les["clube"], les["inicio"], contador, valores, grupos,
                           f"LESÃO {seg.idx + 1} DE {ctx.total}", pop)

    pos = cn.posicao_tela(les["regiao"], les["lado"], escala, x0, LAY)
    if pos:
        onda_de_choque(img, pos, t, niv)
    cn.desenhar_cartao(img, les, seg.idx + 1, LAY, dx=-1100 * (1 - ease_out(t / 0.18)))
    mais_dias(img, les["dias_total"], t)

    if niv == 4:  # câmera lenta dramática: vinheta + carimbo
        forca = min(t / 0.12, 1) * (1 if t < 0.85 else (1 - t) / 0.15)
        vinheta(img, forca)
        if t > 0.12:
            p = (t - 0.12) / 0.15
            texto_com_caixa(img, [("MAIS DE 1 ANO", cn.VERMELHO, 120), ("SEM JOGAR", cn.BRANCO, 80)],
                            760, alfa=min(p * 2, 1) * (1 if t < 0.88 else (1 - t) / 0.12),
                            escala=0.6 + 0.4 * ease_out_back(p))
    if niv >= 3:
        flash(img, cn.VERMELHO, {3: 80, 4: 120}[niv] * max(0, 1 - t / 0.18))

    amp = {1: 0, 2: 7, 3: 16, 4: 26}[niv] * max(0, 1 - t / 0.3) ** 2
    zoom = 1 + {1: 0, 2: 0.01, 3: 0.035, 4: 0.06}[niv] * (1 - ease_out(t / 0.3))
    return tremida_e_zoom(img, *_tremor(frame, amp), zoom).convert("RGB")


def _tremor(frame: int, amp: float) -> tuple[float, float]:
    rnd = random.Random(frame)  # semente = frame: o vídeo sai igual a cada render (reprodutível)
    return rnd.uniform(-amp, amp), rnd.uniform(-amp, amp)


def main(saida: Path = Path("render/neymar_lesoes.mp4"), audio: Path | None = Path("render/sfx.wav"),
         so_frames: list[float] | None = None) -> None:
    lesoes = cn.carregar_lesoes()
    ctx = Contexto(lesoes)
    roteiro = montar_roteiro(lesoes)
    total = roteiro[-1].inicio + roteiro[-1].frames
    print(f"{len(roteiro)} segmentos, {total} frames, {total / FPS:.1f} s")

    def segmento_do(frame: int) -> Segmento:
        return next(s for s in roteiro if s.inicio <= frame < s.inicio + s.frames)

    if so_frames:  # modo prévia: só alguns frames em PNG
        for seg_t in so_frames:
            f = int(seg_t * FPS)
            seg = segmento_do(f)
            desenhar_frame(ctx, seg, (f - seg.inicio) / seg.frames, f).save(f"render/previa_{seg_t:05.1f}s.png")
        return

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{cn.LARGURA}x{cn.ALTURA}", "-r", str(FPS), "-i", "-"]
    if audio and audio.exists():
        cmd += ["-i", str(audio), "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(saida)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(total):
        seg = segmento_do(f)
        proc.stdin.write(desenhar_frame(ctx, seg, (f - seg.inicio) / seg.frames, f).tobytes())
        if f % 150 == 0:
            print(f"  frame {f}/{total}")
    proc.stdin.close()
    proc.wait()
    print(f"✔ {saida}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "previa":
        main(so_frames=[float(x) for x in sys.argv[2:]])
    else:
        main()
