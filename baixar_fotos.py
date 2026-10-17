"""
Coleta fotos de jogadores na Wikipedia/Wikimedia Commons COM checagem de licença.

Fluxo para cada linha de assets/fotos/mapa_fotos.csv:
  1. Wikipedia (pt, depois en): confere se a página existe e não é desambiguação,
     e pega a imagem principal do artigo.
  2. Commons: lê a licença do arquivo. Só baixa licença livre (CC0, domínio
     público, CC BY, CC BY-SA). Arquivo "fair use"/local é descartado.
  3. Salva a miniatura e registra autor + licença em assets/fotos/creditos.csv.
  4. Gera assets/fotos/revisao.png: folha com todas as fotos para conferência humana.

Uso:
    python baixar_fotos.py              # baixa o que falta
    python baixar_fotos.py --refazer    # baixa tudo de novo
    python baixar_fotos.py --sem-sa     # recusa CC BY-SA (só CC0 / PD / CC BY)

Só usa biblioteca padrão + Pillow (já instalado com o matplotlib).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from avatares import slug

PASTA = Path("assets/fotos")
MAPA = PASTA / "mapa_fotos.csv"
CREDITOS = PASTA / "creditos.csv"
# A Wikimedia exige um User-Agent identificável (boa prática de scraping ético)
# Política da Wikimedia: User-Agent com contato real (URL ou e-mail). Genéricos são limitados mais cedo.
USER_AGENT = "ArquivologiaDoFutebol/1.1 (https://www.tiktok.com/@arquivologiadofutebol) Python-urllib"
PAUSA_ENTRE_CHAMADAS = 1.0   # segundos
TENTATIVAS = 5


@dataclass
class Resultado:
    jogador: str
    clube: str
    status: str
    detalhe: str = ""


def _requisitar(url: str) -> bytes:
    """
    GET com retry e backoff exponencial para 429 (limite) e 5xx (instabilidade).
    Respeita o cabeçalho Retry-After quando o servidor informa quanto esperar.
    """
    for tentativa in range(TENTATIVAS):
        time.sleep(PAUSA_ENTRE_CHAMADAS)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as erro:
            if erro.code != 429 and erro.code < 500:
                raise                                   # 404, 403...: tentar de novo não resolve
            espera = float(erro.headers.get("Retry-After") or 2 ** (tentativa + 1))
            print(f"   ... servidor pediu calma (HTTP {erro.code}), aguardando {espera:.0f}s")
            time.sleep(espera)
    raise RuntimeError(f"desisti após {TENTATIVAS} tentativas: {url}")


def _get_json(url: str, params: dict) -> dict:
    consulta = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    return json.loads(_requisitar(f"{url}?{consulta}"))


def _imagem_principal(titulos: str) -> tuple[str | None, str]:
    """
    Aceita vários candidatos separados por '|' (ex.: apelido|nome de registro).
    Cada um pode ter prefixo de idioma (pt:/en:); sem prefixo, tenta pt e depois en.
    Devolve (arquivo, motivos) — os motivos de TODAS as tentativas, para diagnóstico.
    """
    motivos: list[str] = []
    for candidato in (t.strip() for t in titulos.split("|") if t.strip()):
        wikis, titulo = ("pt", "en"), candidato
        if candidato[:3] in ("pt:", "en:"):
            wikis, titulo = (candidato[:2],), candidato[3:]
        for wiki in wikis:
            dados = _get_json(f"https://{wiki}.wikipedia.org/w/api.php", {
                "action": "query", "titles": titulo, "redirects": 1,
                "prop": "pageimages|pageprops", "piprop": "name", "ppprop": "disambiguation",
            })
            pagina = dados["query"]["pages"][0]
            if pagina.get("missing"):
                motivos.append(f"{wiki}:{titulo} não existe")
            elif "disambiguation" in pagina.get("pageprops", {}):
                motivos.append(f"{wiki}:{titulo} é desambiguação")
            elif not pagina.get("pageimage"):
                motivos.append(f"{wiki}:{titulo} sem foto")
            else:
                return pagina["pageimage"], ""
    return None, "; ".join(motivos)


def _limpar_html(texto: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", texto or ""))).strip()


def _info_commons(arquivo: str, largura: int = 400) -> dict | None:
    dados = _get_json("https://commons.wikimedia.org/w/api.php", {
        "action": "query", "titles": f"File:{arquivo}", "prop": "imageinfo",
        "iiprop": "url|extmetadata", "iiurlwidth": largura,
    })
    pagina = dados["query"]["pages"][0]
    if pagina.get("missing") or not pagina.get("imageinfo"):
        return None              # arquivo não está no Commons -> provavelmente não livre
    info = pagina["imageinfo"][0]
    meta = info.get("extmetadata", {})
    return {
        "thumb": info.get("thumburl") or info["url"],
        "pagina": info.get("descriptionurl", ""),
        "licenca": _limpar_html(meta.get("LicenseShortName", {}).get("value", "")),
        "url_licenca": meta.get("LicenseUrl", {}).get("value", ""),
        "autor": _limpar_html(meta.get("Artist", {}).get("value", "")) or "desconhecido",
    }


def _licenca_aceita(licenca: str, aceitar_sa: bool) -> bool:
    """Aceita CC0, domínio público e CC BY (com ou sem SA). Recusa NC/ND (sem uso comercial/edição)."""
    lic = licenca.lower().replace("-", " ")
    livre = lic.startswith(("cc0", "cc by", "pd")) or "public domain" in lic
    proibida = " nc" in lic or " nd" in lic
    share_alike = " sa" in lic
    return livre and not proibida and (aceitar_sa or not share_alike)


def _baixar(url: str, destino: Path) -> None:
    destino.write_bytes(_requisitar(url))


def _folha_de_revisao(linhas: list[dict]) -> Path:
    """Grade com todas as fotos baixadas + nome, para checar se é a pessoa certa."""
    lado, colunas = 180, 6
    fotos = [l for l in linhas if l["arquivo"]]
    linhas_grade = max(1, -(-len(fotos) // colunas))
    folha = Image.new("RGB", (colunas * lado, linhas_grade * (lado + 30)), "#0B1220")
    desenho = ImageDraw.Draw(folha)
    for i, l in enumerate(fotos):
        x, y = (i % colunas) * lado, (i // colunas) * (lado + 30)
        img = Image.open(PASTA / l["arquivo"]).convert("RGB")
        img.thumbnail((lado - 10, lado - 10))
        folha.paste(img, (x + 5, y + 5))
        desenho.text((x + 5, y + lado), f'{l["jogador"]} ({l["clube"]})', fill="#E8EEF6")
    destino = PASTA / "revisao.png"
    folha.save(destino)
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa fotos com licença livre e registra créditos")
    parser.add_argument("--refazer", action="store_true", help="baixa de novo mesmo se já existir")
    parser.add_argument("--sem-sa", action="store_true", help="recusa licenças CC BY-SA")
    args = parser.parse_args()

    if not MAPA.exists():
        parser.error(f"mapa não encontrado: {MAPA}")
    with MAPA.open(encoding="utf-8") as f:
        mapa = list(csv.DictReader(f))

    anteriores: dict[tuple[str, str], dict] = {}
    if CREDITOS.exists() and not args.refazer:
        with CREDITOS.open(encoding="utf-8") as f:
            anteriores = {(c["jogador"], c["clube"]): c for c in csv.DictReader(f)
                          if (PASTA / c["arquivo"]).exists()}

    creditos: list[dict] = []
    resultados: list[Resultado] = []
    for item in mapa:
        jogador, sigla, titulo = item["jogador"], item["sigla"], item["titulo_wikipedia"].strip()
        base = slug(jogador, sigla)
        if (jogador, sigla) in anteriores:             # já baixado antes: reaproveita, sem chamar a API
            creditos.append(anteriores[(jogador, sigla)])
            resultados.append(Resultado(jogador, sigla, "OK", "já baixada (reaproveitada)"))
            continue
        if not titulo:
            resultados.append(Resultado(jogador, sigla, "PULADO", "sem título no mapa (usa iniciais)"))
            continue
        try:
            arquivo, motivo = _imagem_principal(titulo)
            if not arquivo:
                resultados.append(Resultado(jogador, sigla, "SEM FOTO", motivo))
                continue
            info = _info_commons(arquivo)
            if info is None:
                resultados.append(Resultado(jogador, sigla, "RECUSADA", f"{arquivo} não está no Commons (não livre)"))
                continue
            if not _licenca_aceita(info["licenca"], aceitar_sa=not args.sem_sa):
                resultados.append(Resultado(jogador, sigla, "RECUSADA", f"licença '{info['licenca']}'"))
                continue
            extensao = Path(urllib.parse.urlparse(info["thumb"]).path).suffix.lower() or ".jpg"
            destino = PASTA / f"{base}{extensao}"
            if args.refazer or not destino.exists():
                _baixar(info["thumb"], destino)
            creditos.append({"jogador": jogador, "clube": sigla, "arquivo": destino.name,
                             "autor": info["autor"], "licenca": info["licenca"],
                             "url_licenca": info["url_licenca"], "fonte": info["pagina"]})
            resultados.append(Resultado(jogador, sigla, "OK", f"{info['licenca']} · {info['autor']}"))
        except Exception as erro:  # rede, JSON inesperado etc.: registra e segue
            resultados.append(Resultado(jogador, sigla, "ERRO", repr(erro)))

    with CREDITOS.open("w", encoding="utf-8", newline="") as f:
        campos = ["jogador", "clube", "arquivo", "autor", "licenca", "url_licenca", "fonte"]
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(creditos)

    for r in resultados:
        print(f"[{r.status:8}] {r.jogador} ({r.clube}) {('- ' + r.detalhe) if r.detalhe else ''}")
    ok = sum(r.status == "OK" for r in resultados)
    print(f"\n{ok}/{len(resultados)} com foto livre. Créditos em {CREDITOS}")
    if creditos:
        print(f"CONFIRA A FOLHA DE REVISÃO (pessoa certa?): {_folha_de_revisao(creditos)}")


if __name__ == "__main__":
    main()
