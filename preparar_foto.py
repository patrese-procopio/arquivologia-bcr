"""Prepara a foto do jogador para o vídeo: amplia 4x, recorta o fundo e corta nas bordas do corpo.

Tudo roda localmente (OpenCV + rembg): a imagem não sai da máquina.
Uso: python preparar_foto.py assets/neymar_psg2023.jpg assets/neymar_recorte.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
from rembg import new_session, remove

MODELO_SR = Path("modelos/EDSR_x4.pb")


def ampliar(img_bgr: np.ndarray) -> np.ndarray:
    """Super-resolução com rede neural (EDSR): reconstrói detalhe em vez de só esticar pixels."""
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(MODELO_SR))
    sr.setModel("edsr", 4)
    return sr.upsample(img_bgr)


def recortar(img_rgb: Image.Image) -> Image.Image:
    """Remove o fundo com um modelo treinado em pessoas e corta no contorno do corpo."""
    sessao = new_session("u2net_human_seg")
    sem_fundo = remove(img_rgb, session=sessao, post_process_mask=True)
    # A borda da máscara carrega pixels do fundo (franja verde da grama):
    # MinFilter "come" 2 px do contorno e o blur suaviza o corte.
    alfa = sem_fundo.getchannel("A").filter(ImageFilter.MinFilter(5)).filter(ImageFilter.GaussianBlur(1))
    sem_fundo.putalpha(alfa)
    return remover_spill(sem_fundo).crop(sem_fundo.getbbox())  # getbbox ignora pixels transparentes


def remover_spill(img: Image.Image, faixa_px: int = 15) -> Image.Image:
    """Despill: na faixa do contorno, limita o verde à média de R e B (reflexo da grama)."""
    rgba = np.array(img).astype(np.int16)
    alfa = rgba[..., 3].astype(np.uint8)
    nucleo = cv2.erode(alfa, np.ones((faixa_px, faixa_px), np.uint8))  # miolo do corpo, intocado
    borda = (alfa > 0) & (nucleo < 255)
    r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
    limite = (r + b) // 2
    rgba[..., 1] = np.where(borda & (g > limite), limite, g)
    return Image.fromarray(rgba.astype(np.uint8), "RGBA")


def main(entrada: Path, saida: Path) -> None:
    original = cv2.imread(str(entrada))
    if original is None:
        raise FileNotFoundError(entrada)
    grande = ampliar(original)
    rgb = Image.fromarray(cv2.cvtColor(grande, cv2.COLOR_BGR2RGB))
    final = recortar(rgb)
    saida.parent.mkdir(parents=True, exist_ok=True)
    final.save(saida)
    print(f"{original.shape[1]}x{original.shape[0]} -> ampliada {grande.shape[1]}x{grande.shape[0]} "
          f"-> recorte {final.width}x{final.height} em {saida}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
