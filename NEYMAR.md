# O corpo do Neymar: 39 lesões em vídeo, guiado por dados

Short vertical (1080×1920, 57 s) que mostra cada lesão de Neymar de 2010 a 2026: o ponto do corpo atingido, o clube, a data e os dias fora, com contador contínuo e placar de dias por clube. **Tudo no vídeo (ritmo, efeitos, som) é gerado a partir dos dados.**

## Números

| Métrica | Valor |
|---|---|
| Lesões físicas | 39 (2010–2026) |
| Dias fora | 1.530 (4,2 anos) |
| Por clube | PSG 759 · Al-Hilal 411 · Santos 223 · Barcelona 137 |
| Lesões com lado do corpo confirmado | 18 de 39 |

## Pipeline

```
dados/neymar_lesoes_bruto.csv ─┐
dados/neymar_lados.csv ────────┼─> reconciliar_lesoes.py ─> dados/neymar_lesoes.csv
                               │      normalizar → consolidar episódios → enriquecer lado → auditar
assets/neymar_psg2023.jpg ─────┴─> preparar_foto.py ─> assets/neymar_recorte.png
                                     super-resolução EDSR 4× → recorte (rembg) → despill
dados/neymar_lesoes.csv ─> animar.py (+ corpo_neymar.py) ─> render/neymar_lesoes.mp4
                       └─> sons.py ─> render/sfx.wav (trilha sintetizada, mesmo roteiro)
```

## Decisões de engenharia de dados

- **Regra de negócio explícita:** só lesões físicas. Covid, doença, fadiga e descanso ficam fora (8 registros).
- **Reconciliação de fontes:** a matéria do Transfermarkt/Soccerway dizia "37 lesões", mas listava 34. O script audita a lista contra o total publicado.
- **Consolidação de episódios:** registros sobrepostos (folga ≤ 2 dias) viram uma lesão só. Validação externa: março/2025 consolidou em 39 dias, e a TMC publicou "39 a 40 dias"; o Santos fechou em 8 lesões, igual à manchete da TMC.
- **Correção de lesão escondida:** o TM somava 131 dias de "recuperação" à lesão de nov/2022, mas era a lesão de fev/2023 (Globo). Separadas, com a origem anotada na coluna `obs`.
- **Join aproximado (`merge_asof`, tolerância de 3 dias):** fontes diferentes datam a mesma lesão com 1–2 dias de diferença (jogo × exame).
- **Fail fast:** o pipeline para se duas fontes discordarem da região do corpo ou se uma linha de lado não casar com nenhuma lesão.
- **Linhagem por linha:** cada lesão carrega `fonte` e `fonte_lado`.

## Decisões de vídeo

- **Quadro = função pura do estado** (`desenhar_quadro(lesoes, indice)`), como UI = f(estado) no React.
- **Gravidade → efeitos:** `nivel(dias)` define tempo de tela, tremida, zoom, flash e peso do som.
- **`RITMO` global:** um multiplicador desacelera tudo, porque os efeitos são frações de `t`.
- **Zona segura Shorts/TikTok** em constantes (`SEGURA_TOPO/BASE/DIREITA`).
- **Reprodutível:** tremida com semente = nº do frame; trilha com semente fixa.
- **Performance:** `lru_cache` no fundo (blur) e na foto; frames vão direto para o stdin do ffmpeg.

## LGPD

Dado de saúde é **dado sensível** (art. 5º, II). Aqui ele já foi tornado público, é de figura pública e o uso tem fim jornalístico/informativo (art. 4º, II, "a"). Boas práticas adotadas:
- só o que foi publicado, com fonte por linha;
- **nenhum lado do corpo inventado**: sem fonte, o marcador fica no eixo do corpo, vazado;
- imagem processada localmente (super-resolução e recorte sem API de terceiros).

## Fontes

Transfermarkt (via Soccerway, 2024) · perfil Soccerway (2026) · O Globo, "Raio-X das lesões" (13/03/2023) · Estadão (17/04/2014) · TMC (04/09/2026) · site do Atlético-MG e Bahia Já (lesão no olho, 2010).

## Como rodar

```bash
pip install -r requirements.txt
# modelo de super-resolução (38 MB), só para preparar_foto.py:
curl -L -o modelos/EDSR_x4.pb https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x4.pb

python reconciliar_lesoes.py                                   # base reconciliada
python preparar_foto.py assets/neymar_psg2023.jpg assets/neymar_recorte.png
python corpo_neymar.py 2023-10-19                              # quadro estático de teste
python animar.py previa 2.4 37.4 52                            # PNGs de prévia em segundos
python sons.py && python animar.py                             # vídeo final (ffmpeg necessário)
```
