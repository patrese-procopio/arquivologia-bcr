"""
Gera CREDITOS.md a partir de assets/fotos/creditos.csv.

A página fica no repositório e é o destino do link "Créditos das imagens"
na descrição dos vídeos. Fonte única da verdade: nunca edite CREDITOS.md à mão,
rode este script depois de cada `python baixar_fotos.py`.

Uso:
    python gerar_creditos.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ORIGEM = Path("assets/fotos/creditos.csv")
DESTINO = Path("CREDITOS.md")
LIVRES_DE_ATRIBUICAO = ("cc0", "public domain", "pd")


def _celula(texto: str) -> str:
    """Escapa o caractere que quebraria a tabela Markdown."""
    return (texto or "").replace("|", "\\|").strip()


def main() -> None:
    if not ORIGEM.exists():
        raise SystemExit(f"{ORIGEM} não encontrado: rode antes `python baixar_fotos.py`")

    with ORIGEM.open(encoding="utf-8") as f:
        creditos = sorted(csv.DictReader(f), key=lambda c: (c["clube"], c["jogador"]))

    linhas = [
        "# Créditos das imagens",
        "",
        "As fotos dos jogadores usadas nos vídeos do **Arquivologia do Futebol** vêm do "
        "[Wikimedia Commons](https://commons.wikimedia.org) e são usadas conforme as licenças abaixo. "
        "As imagens foram recortadas em formato circular (adaptação); nenhuma outra alteração foi feita.",
        "",
        "| Jogador | Clube | Autor | Licença | Arquivo original |",
        "|---|---|---|---|---|",
    ]
    for c in creditos:
        licenca = _celula(c["licenca"])
        if c.get("url_licenca"):
            licenca = f"[{licenca}]({c['url_licenca']})"
        autor = _celula(c["autor"])
        if autor.lower() == "desconhecido":
            autor = "não informado (ver arquivo original)"
        linhas.append(f"| {_celula(c['jogador'])} | {c['clube']} | {autor} | {licenca} | "
                      f"[Commons]({c['fonte']}) |")

    atribuicao = sum(not c["licenca"].lower().startswith(LIVRES_DE_ATRIBUICAO) for c in creditos)
    linhas += [
        "",
        f"_{len(creditos)} imagens · {atribuicao} exigem atribuição (CC BY / CC BY-SA) · "
        f"as demais são CC0 ou domínio público._",
        "",
        "Jogadores sem foto de licença livre aparecem com as iniciais (avatar gerado por código).",
        "",
    ]
    DESTINO.write_text("\n".join(linhas), encoding="utf-8")
    print(f"{DESTINO} gerado com {len(creditos)} imagens ({atribuicao} exigem atribuição).")


if __name__ == "__main__":
    main()
