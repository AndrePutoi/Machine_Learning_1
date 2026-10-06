"""
llm_embeddings.py
==================
Representações latentes do TEXTO de cada listagem, geradas por um LLM
pequeno de embeddings, a partir do dataset CRU (data/listings.csv).

  1. Seleciona automaticamente todas as colunas de texto (não numéricas),
     exceto as realmente inúteis ou proibidas:
       - identificação / links (URLs, nome do anfitrião, licença)
       - metadados do scraping (datas de recolha, origem)
       - o próprio preço e as cotações de preço (seria leakage do target)
  2. Junta-as num único texto por listagem, no formato "Campo: valor"
     (limpa HTML, converte t/f em yes/no e a lista de amenities em texto).
  3. Passa o texto pelo modelo Qwen/Qwen3-Embedding-0.6B (LLM de 0.6B
     parâmetros, embeddings de 1024 dimensões).
  4. Gera 3 versões do espaço latente — 256, 512 e 1024 dimensões — e
     guarda cada uma num CSV (id, price, emb_0 ... emb_{d-1}).

Porquê um único modelo para as 3 dimensões: o Qwen3-Embedding foi treinado
com Matryoshka Representation Learning (MRL) — as primeiras d componentes
do vetor de 1024 são, por si só, um embedding válido de dimensão d. Basta
truncar e voltar a normalizar (norma L2 = 1). Usar 3 modelos diferentes
misturaria o efeito da dimensão com o efeito do modelo.

Precisa de: torch, sentence-transformers (ver requirements-embeddings.txt).
Na primeira execução descarrega o modelo da Hugging Face (~1.2 GB).

Autor: <o teu nome>
Disciplina: Machine Learning - Practical Project 1
"""

import html
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DIMENSOES = [256, 512, 1024]

# Colunas de texto excluídas — e porquê
COLS_TEXTO_EXCLUIDAS = {
    # identificação / links
    "listing_url": "link", "picture_url": "link", "host_url": "link",
    "host_profile_url": "link", "host_picture_url": "link",
    "host_name": "identificação", "license": "identificação (nº de registo)",
    # metadados do scraping (iguais ou quase iguais em todas as linhas)
    "last_scraped": "metadado do scraping", "calendar_last_scraped": "metadado do scraping",
    "source": "metadado do scraping",
    # target e derivados -> leakage
    "price": "target (leakage)", "price_quote_raw": "contém o preço (leakage)",
    "price_quote_checkin_date": "cotação de preço (leakage)",
    "price_quote_checkout_date": "cotação de preço (leakage)",
}

# Ordem no texto: o mais informativo primeiro (o modelo trunca a max_seq_length tokens)
ORDEM_PREFERIDA = [
    "name", "property_type", "room_type", "neighbourhood_group_cleansed",
    "neighbourhood_cleansed", "bathrooms_text", "description", "amenities",
    "host_is_superhost", "host_identity_verified", "host_has_profile_pic",
    "has_availability", "first_review", "last_review", "host_location", "host_about",
]


# ============================================================
# 1. SELEÇÃO E CONSTRUÇÃO DO TEXTO
# ============================================================

def selecionar_colunas_texto(df: pd.DataFrame) -> list[str]:
    """Todas as colunas não numéricas, exceto as de COLS_TEXTO_EXCLUIDAS."""
    texto = [c for c in df.columns
             if not pd.api.types.is_numeric_dtype(df[c]) and c not in COLS_TEXTO_EXCLUIDAS]
    ordem = {c: i for i, c in enumerate(ORDEM_PREFERIDA)}
    return sorted(texto, key=lambda c: ordem.get(c, len(ordem)))


def _limpar_valor(col: str, valor) -> str:
    if pd.isna(valor):
        return ""
    s = str(valor)
    if s in ("t", "f"):
        return "yes" if s == "t" else "no"
    if col == "amenities":
        try:
            return ", ".join(json.loads(s))
        except (json.JSONDecodeError, TypeError):
            return s
    s = html.unescape(re.sub(r"<br\s*/?>", "\n", s))
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"[ \t]+", " ", s).strip()


def construir_texto(df: pd.DataFrame, colunas: list[str]) -> pd.Series:
    """Um texto por listagem: 'Campo: valor' por linha, campos vazios omitidos."""
    def _linha(row):
        partes = []
        for col in colunas:
            v = _limpar_valor(col, row[col])
            if v:
                partes.append(f"{col.replace('_', ' ').capitalize()}: {v}")
        return "\n".join(partes)
    return df[colunas].apply(_linha, axis=1)


def preco_numerico(price: pd.Series) -> pd.Series:
    """'$1,234.00' -> 1234.0 (NaN se em falta)."""
    if pd.api.types.is_numeric_dtype(price):
        return price.astype(float)
    return pd.to_numeric(price.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


# ============================================================
# 2. EMBEDDINGS
# ============================================================

def gerar_embeddings(textos: list[str], model_name: str = MODEL_NAME,
                     batch_size: int = 16, max_seq_length: int = 512) -> np.ndarray:
    """
    Embeddings de dimensão máxima (1024), normalizados (norma L2 = 1).
    Usa GPU com float16 se houver CUDA; senão CPU (muito mais lento).
    """
    # Os kernels Triton "nativos" do torch são compilados com gcc e precisam dos
    # headers do Python (python3-dev); desligá-los usa as ops normais do CUDA.
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs)
    model.max_seq_length = max_seq_length
    print(f"[info] modelo {model_name} em {device}, max_seq_length={max_seq_length}")

    emb = model.encode(textos, batch_size=batch_size, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    return emb.astype(np.float32)


def truncar_matryoshka(emb: np.ndarray, dim: int) -> np.ndarray:
    """Primeiras `dim` componentes, renormalizadas para norma L2 = 1."""
    sub = emb[:, :dim]
    return sub / np.linalg.norm(sub, axis=1, keepdims=True)


# ============================================================
# 3. PIPELINE: DATASET CRU -> 3 CSVs
# ============================================================

def caminho_csv(outdir, dim: int) -> Path:
    return Path(outdir) / f"listings_emb_{dim}.csv"


def gerar_csvs_embeddings(caminho_cru="data/listings.csv", outdir="Data_clean/embeddings",
                          dimensoes=DIMENSOES, **kwargs_embeddings) -> dict:
    """
    Lê o dataset cru, constrói o texto, gera embeddings e guarda um CSV por
    dimensão. Guarda também o texto usado (auditoria). Devolve {dim: caminho}.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(caminho_cru)
    colunas = selecionar_colunas_texto(df)
    print(f"[info] {len(df)} listagens; {len(colunas)} colunas de texto usadas: {colunas}")

    textos = construir_texto(df, colunas)
    pd.DataFrame({"id": df["id"], "texto": textos}).to_csv(outdir / "listings_texto.csv", index=False)

    emb = gerar_embeddings(textos.tolist(), **kwargs_embeddings)

    caminhos = {}
    base = pd.DataFrame({"id": df["id"], "price": preco_numerico(df["price"])})
    for dim in dimensoes:
        emb_d = truncar_matryoshka(emb, dim)
        df_d = pd.concat([base, pd.DataFrame(emb_d, columns=[f"emb_{i}" for i in range(dim)])], axis=1)
        caminhos[dim] = caminho_csv(outdir, dim)
        df_d.to_csv(caminhos[dim], index=False, float_format="%.6f")
        print(f"[info] guardado {caminhos[dim]}  shape={df_d.shape}")
    return caminhos


def carregar_embeddings(dim: int, outdir="Data_clean/embeddings"):
    """Lê o CSV de uma dimensão; devolve (df, lista das colunas de embedding)."""
    df = pd.read_csv(caminho_csv(outdir, dim))
    return df, [c for c in df.columns if c.startswith("emb_")]


def adicionar_quadrados(df: pd.DataFrame, emb_cols: list[str]):
    """
    Regressão 'quadrática' em embeddings: junta x_i² a cada componente
    (sem termos cruzados x_i·x_j). Devolve (df_expandido, colunas_finais).
    """
    quad = df[emb_cols].pow(2)
    quad.columns = [f"{c}_sq" for c in emb_cols]
    return pd.concat([df, quad], axis=1), emb_cols + list(quad.columns)
