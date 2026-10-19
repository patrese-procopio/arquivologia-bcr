"""Reconcilia o histórico de lesões do Neymar a partir de várias fontes.

Pipeline:  bruto -> normalizar -> consolidar episódios -> enriquecer com lado -> auditar -> gravar

Fontes:
  - TM_via_soccerway_2024: lista do Transfermarkt publicada pelo Soccerway (até nov/2024),
    com dias e jogos perdidos, mas sem data de fim.
  - soccerway_perfil_2026: perfil do jogador no Soccerway, com início e fim, sem jogos.
  - neymar_lados.csv: lado do corpo (Globo 2023, Soccerway 2024, TMC 2026), uma fonte por linha.

Regra de negócio: contar só lesões físicas. Doença, Covid, fadiga e descanso ficam de fora.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ENTRADA = Path("dados/neymar_lesoes_bruto.csv")
LADOS = Path("dados/neymar_lados.csv")
SAIDA = Path("dados/neymar_lesoes.csv")

NAO_FISICAS = {"Indisposição", "Coronavírus", "Fadiga", "Illness", "Rest"}

# Ordem importa: o primeiro termo que casar define a região.
REGIOES: list[tuple[str, str]] = [
    ("ligamentos cruzados", "joelho"), ("knee", "joelho"),
    ("metatarso", "pé"), ("pé", "pé"),  # antes de "tarso": "metatarso" contém "tarso"
    ("tarso", "tornozelo"), ("tornozelo", "tornozelo"), ("ankle", "tornozelo"),
    ("adutor", "virilha"),
    ("hamstring", "posterior"),
    ("calf", "panturrilha"),
    ("coxa", "coxa"), ("thigh", "coxa"),
    ("costal", "costela"),
    ("coluna", "costas"),
    ("olho", "cabeça"),
]

# Dois registros separados por até N dias são o mesmo episódio (ex.: diagnóstico + agravamento).
FOLGA_EPISODIO = pd.Timedelta(days=2)
# Fontes diferentes datam a mesma lesão com 1 ou 2 dias de diferença (jogo x exame).
TOLERANCIA_JOIN = pd.Timedelta(days=3)

PASSAGENS = [("Santos", "2009-03-01"), ("Barcelona", "2013-06-01"), ("PSG", "2017-08-03"),
             ("Al-Hilal", "2023-08-15"), ("Santos", "2025-01-31")]

TM_PUBLICADO = {"lesoes": 37, "dias": 1385, "jogos": 240}


def regiao(tipo: str) -> str:
    t = tipo.lower()
    return next((r for termo, r in REGIOES if termo in t), "não informada")


def clube_na_data(data: pd.Timestamp) -> str:
    """Regra: a lesão conta para o clube com que ele tinha contrato na data."""
    return [nome for nome, ini in PASSAGENS if data >= pd.Timestamp(ini)][-1]


def normalizar(caminho: Path) -> pd.DataFrame:
    df = pd.read_csv(caminho, parse_dates=["inicio", "fim"])
    df["dias_recuperacao"] = df["dias_recuperacao"].fillna(0)
    # Fonte TM não traz fim: deriva do início + dias. Fonte Soccerway não traz dias: deriva das datas.
    sem_fim = df["fim"].isna()
    df.loc[sem_fim, "fim"] = df["inicio"] + pd.to_timedelta(df["dias"] + df["dias_recuperacao"], unit="D")
    df["dias_total"] = (df["fim"] - df["inicio"]).dt.days
    df["fisica"] = ~df["tipo_original"].isin(NAO_FISICAS)
    df["regiao"] = df["tipo_original"].map(regiao)
    return df.sort_values("inicio").reset_index(drop=True)


def consolidar_episodios(df: pd.DataFrame) -> pd.DataFrame:
    """Junta registros físicos que se sobrepõem (ou quase) num único episódio.

    O tipo e a região vêm do registro mais longo do grupo, que é o diagnóstico principal.
    """
    fis = df[df["fisica"]].sort_values("inicio").copy()
    fim_acumulado = fis["fim"].cummax().shift()
    fis["episodio"] = (fis["inicio"] > fim_acumulado + FOLGA_EPISODIO).cumsum()

    def resumir(g: pd.DataFrame) -> pd.Series:
        principal = g.loc[g["dias_total"].idxmax()]
        inicio, fim = g["inicio"].min(), g["fim"].max()
        return pd.Series({
            "inicio": inicio, "fim": fim, "dias_total": (fim - inicio).days,
            "tipo_original": principal["tipo_original"], "regiao": principal["regiao"],
            "jogos": g["jogos"].sum(min_count=1), "fonte": principal["fonte"],
            "registros": len(g),
        })

    return fis.groupby("episodio").apply(resumir, include_groups=False).reset_index(drop=True)


def enriquecer_lado(ep: pd.DataFrame, caminho: Path) -> pd.DataFrame:
    """Join aproximado por data (merge_asof) e checagem de que a região bate."""
    lados = pd.read_csv(caminho, parse_dates=["data_fonte"]).sort_values("data_fonte")
    lados = lados.rename(columns={"regiao": "regiao_fonte", "fonte": "fonte_lado", "obs": "obs_lado"})
    out = pd.merge_asof(ep.sort_values("inicio"), lados, left_on="inicio", right_on="data_fonte",
                        direction="nearest", tolerance=TOLERANCIA_JOIN)

    casou = out["lado"].notna()
    conflito = casou & (out["regiao"] != out["regiao_fonte"])
    if conflito.any():  # fail fast: duas fontes dizendo partes do corpo diferentes
        raise ValueError(f"Região diverge entre fontes:\n{out.loc[conflito, ['inicio', 'regiao', 'regiao_fonte']]}")

    nao_usadas = set(lados["data_fonte"]) - set(out.loc[casou, "data_fonte"])
    if nao_usadas:  # cada linha da tabela de lados precisa encontrar a sua lesão
        raise ValueError(f"Linhas de lado sem lesão correspondente: {sorted(d.date() for d in nao_usadas)}")

    out["lado"] = out["lado"].fillna("não informado")
    out["clube"] = out["inicio"].map(clube_na_data)
    return out.drop(columns=["data_fonte", "regiao_fonte"])


def auditar_fonte_tm(df: pd.DataFrame) -> None:
    tm = df[df["fonte"] == "TM_via_soccerway_2024"]
    print(f"  lista do TM: {len(tm)} lesões, {int(tm['jogos'].sum())} jogos | matéria diz: {TM_PUBLICADO}")


def resumo(ep: pd.DataFrame, n_bruto: int, n_nao_fisicas: int) -> None:
    dias = int(ep["dias_total"].sum())
    print(f"\n{n_bruto} registros -> {n_nao_fisicas} não físicos fora -> {len(ep)} lesões (episódios)")
    print(f"DIAS FORA: {dias} = {dias / 365.25:.2f} anos")
    print(f"Com lado conhecido: {(ep['lado'] != 'não informado').sum()} de {len(ep)}")

    print("\nEPISÓDIOS CONSOLIDADOS (mais de um registro):")
    print(ep.loc[ep["registros"] > 1, ["inicio", "fim", "tipo_original", "dias_total", "registros"]].to_string(index=False))

    print("\nPOR CLUBE:")
    print(ep.groupby("clube", sort=False).agg(lesoes=("clube", "size"), dias=("dias_total", "sum")).to_string())

    print("\nMARCADORES NO CORPO (região + lado):")
    print(ep.groupby(["regiao", "lado"]).size().rename("marcas").sort_values(ascending=False).to_string())


def main() -> None:
    bruto = normalizar(ENTRADA)
    print("AUDITORIA DA FONTE TRANSFERMARKT")
    auditar_fonte_tm(bruto)

    episodios = enriquecer_lado(consolidar_episodios(bruto), LADOS)
    resumo(episodios, len(bruto), int((~bruto["fisica"]).sum()))

    colunas = ["inicio", "fim", "tipo_original", "regiao", "lado", "clube", "dias_total", "jogos",
               "registros", "fonte", "fonte_lado", "obs_lado"]
    episodios[colunas].to_csv(SAIDA, index=False, date_format="%Y-%m-%d")
    print(f"\n✔ {SAIDA} gravado ({len(episodios)} lesões)")


if __name__ == "__main__":
    main()
