# Arquivologia do Futebol · Bar Chart Race

Pipeline em Python que transforma dados históricos de futebol em vídeos verticais animados
(1080×1920) para TikTok, Reels e YouTube Shorts, **com rastreabilidade de fonte em cada número
e de licença em cada imagem**.

Primeiro episódio: **Batalha de Artilheiros**, os maiores goleadores de Flamengo e Palmeiras
na Copa Libertadores, de 1968 a 2026.

> Créditos das imagens: [CREDITOS.md](CREDITOS.md) · Metodologia dos dados: [dados/FONTES.md](dados/FONTES.md)

---

## Arquitetura

```
dados/*.csv (tidy, com fonte e confiança por linha)
   │
   ├─► validação (data quality gate: tipos, duplicatas, células vazias, clubes conhecidos)
   ├─► matriz acumulada (pivot + reindex de anos sem jogo + cumsum)
   ├─► linha do tempo (interpolação de valores E posições, ease-in-out,
   │                    ritmo adaptativo, desempate "quem chegou primeiro")
   ├─► renderização matplotlib (artistas reaproveitados, só o top N é desenhado)
   └─► ffmpeg (H.264 yuv420p + fundo estático sobreposto via filtro overlay)

baixar_fotos.py ─► Wikipedia/Commons API ─► filtro de licença ─► creditos.csv ─► CREDITOS.md
```

| Módulo | Responsabilidade |
|---|---|
| `bar_race.py` | Validação, transformação, linha do tempo, renderização e CLI |
| `avatares.py` | Avatar circular a partir da foto, ou iniciais quando não há foto livre |
| `baixar_fotos.py` | Coleta de imagens com checagem de licença, retry com backoff e folha de revisão |
| `gerar_creditos.py` | Gera `CREDITOS.md` a partir de `creditos.csv` (fonte única da verdade) |

## Decisões técnicas que valem a leitura

- **Desempate esportivo:** em empate, quem chegou primeiro à marca fica na frente. Implementado
  com `np.lexsort` estável, usando a posição anterior como chave secundária.
- **Performance:** artistas com `alpha=0` ainda são desenhados pelo matplotlib. Escondê-los com
  `set_visible(False)` reduziu o render de **9m41s para 2m42s**.
- **Composição no ffmpeg:** misturar a transparência de um fundo de tela inteira em Python, a cada
  quadro, custava cerca de 110 ms por quadro. A classe `FFMpegComFundo` herda do `FFMpegWriter` e
  injeta o fundo como segunda entrada do ffmpeg (`overlay`). O tempo voltou ao patamar sem fundo.
- **Fundo de torcida procedural:** silhuetas geradas por código, sem foto real. Assim não há direito
  autoral de terceiros nem direito de imagem de torcedores identificáveis (LGPD).
- **Área segura:** o layout respeita as zonas cobertas pela interface do TikTok e do Reels.

## Qualidade e governança dos dados

- Cada linha do CSV tem as colunas `fonte` e `confianca` (alta/média).
- Os totais por jogador são conferidos automaticamente contra os totais publicados por fontes
  independentes: **22 de 22 batem**.
- Uma análise de sensibilidade mostra que nenhum dado de confiança média altera o topo do ranking final.
- Hierarquia de fontes: clube oficial e CONMEBOL (primárias) > acervo histórico > imprensa (corroboração).

## Imagens e licenças

- Só são baixadas licenças **CC0, domínio público, CC BY ou CC BY-SA**. As licenças NC (uso não
  comercial), ND (sem adaptação) e os arquivos de "uso justo" são recusados automaticamente.
- Cada imagem tem autor, licença e link registrados em `assets/fotos/creditos.csv`.
- O script gera uma folha de revisão (`revisao.png`) para conferência humana, porque um nome
  ambíguo pode trazer a foto de outra pessoa.

## Como rodar

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt        # + ffmpeg no sistema: winget install --id Gyan.FFmpeg

python baixar_fotos.py                 # fotos com licença livre + revisao.png
python gerar_creditos.py               # atualiza CREDITOS.md

python bar_race.py --dados dados/libertadores_fla_pal.csv `
                   --eventos dados/libertadores_fla_pal_eventos.csv `
                   --top-n 10 --preview          # prévia rápida em 540×960
```

Opções úteis: `--selo`, `--titulo`, `--subtitulo`, `--fonte`, `--sem-torcida`, `--sem-fotos`
(veja todas com `python bar_race.py -h`).

## Licenças deste repositório

- **Código:** MIT.
- **Dados compilados** (`dados/`): CC BY 4.0. Cite "Arquivologia do Futebol" e as fontes listadas em `FONTES.md`.
- **Imagens de terceiros:** licenças individuais, listadas em [CREDITOS.md](CREDITOS.md).
