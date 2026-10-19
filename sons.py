"""Trilha de efeitos sonoros sintetizada (numpy), sincronizada com o roteiro do animar.py.

Nada de sample baixado: sem risco de direito autoral e o som também é guiado pelos dados
(lesão mais grave = impacto mais grave).
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

import animar
import corpo_neymar as cn

SR = 44_100
RNG = np.random.default_rng(10)  # semente fixa: trilha reprodutível


def _t(seg: float) -> np.ndarray:
    return np.arange(int(seg * SR)) / SR


def passa_baixa(x: np.ndarray, corte_hz: float) -> np.ndarray:
    """Filtro de 1 polo: suaviza o ruído branco em 'estrondo'."""
    a = np.exp(-2 * np.pi * corte_hz / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = (1 - a) * v + a * acc
        y[i] = acc
    return y


def impacto(nivel: int) -> np.ndarray:
    """Bumbo grave com queda de afinação + estalo de ruído. Nível 1..4 controla peso e cauda."""
    dur = {1: 0.25, 2: 0.45, 3: 0.9, 4: 2.2}[nivel]
    t = _t(dur)
    f0 = {1: 160, 2: 110, 3: 75, 4: 55}[nivel]
    freq = f0 + f0 * 2.5 * np.exp(-t * 30)                   # pitch drop: o "tum" do impacto
    fase = 2 * np.pi * np.cumsum(freq) / SR
    corpo = np.sin(fase) * np.exp(-t * {1: 18, 2: 9, 3: 5, 4: 2.2}[nivel])
    estalo = RNG.standard_normal(len(t)) * np.exp(-t * 70) * {1: 0.25, 2: 0.4, 3: 0.6, 4: 0.7}[nivel]
    som = corpo + estalo
    if nivel == 4:  # ronco grave longo para a câmera lenta
        som += passa_baixa(RNG.standard_normal(len(t)), 90) * 6 * np.exp(-t * 1.2)
    return som * {1: 0.35, 2: 0.55, 3: 0.8, 4: 1.0}[nivel]


def whoosh(dur: float = 0.6) -> np.ndarray:
    """Passagem de ar (transferência): ruído com envelope em sino."""
    t = _t(dur)
    env = np.sin(np.pi * t / dur) ** 2
    return passa_baixa(RNG.standard_normal(len(t)), 1800) * env * 1.6


def tic() -> np.ndarray:
    t = _t(0.03)
    return np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 180) * 0.12


def subida(dur: float) -> np.ndarray:
    """Riser: tom que sobe e cresce, prepara o carimbo final."""
    t = _t(dur)
    freq = 200 + 900 * (t / dur) ** 2
    return np.sin(2 * np.pi * np.cumsum(freq) / SR) * (t / dur) ** 2 * 0.25


def colocar(trilha: np.ndarray, som: np.ndarray, em_seg: float) -> None:
    i = int(em_seg * SR)
    fim = min(i + len(som), len(trilha))
    trilha[i:fim] += som[: fim - i]


def gerar(saida: Path = Path("render/sfx.wav")) -> None:
    lesoes = cn.carregar_lesoes()
    roteiro = animar.montar_roteiro(lesoes)
    total = (roteiro[-1].inicio + roteiro[-1].frames) / animar.FPS
    trilha = np.zeros(int(total * SR) + SR)

    for seg in roteiro:
        ini, dur = seg.inicio / animar.FPS, seg.frames / animar.FPS
        if seg.tipo == "intro":
            colocar(trilha, impacto(2), 0.05)
        elif seg.tipo == "transferencia":
            colocar(trilha, whoosh(), ini)
        elif seg.tipo == "lesao":
            niv = animar.nivel(lesoes.iloc[seg.idx]["dias_total"])
            colocar(trilha, impacto(niv), ini)
            if niv >= 2:  # tiques do contador rolando
                for k in np.arange(0.08, dur * 0.65, 0.07):
                    colocar(trilha, tic(), ini + k)
        elif seg.tipo == "outro":
            carimbo = ini + 0.06 * dur  # instante em que o "4,2 ANOS" bate na tela
            colocar(trilha, subida(0.6), carimbo - 0.6)
            colocar(trilha, impacto(4), carimbo)

    trilha = np.tanh(trilha * 1.4)  # soft clip: soma de sons não estoura
    trilha /= np.max(np.abs(trilha)) / 0.9
    pcm = (trilha * 32767).astype(np.int16)
    saida.parent.mkdir(exist_ok=True)
    with wave.open(str(saida), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"✔ {saida} ({total:.1f} s)")


if __name__ == "__main__":
    gerar()
