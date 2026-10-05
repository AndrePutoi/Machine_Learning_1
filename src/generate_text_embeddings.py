"""
generate_text_embeddings.py
=============================
1. Lê o CSV de listagens (caminho fornecido pelo utilizador).
2. Mantém só as colunas de texto (name, description, neighborhood_overview,
   host_about) — descarta todas as numéricas.
3. Cria uma coluna 'combined_text' que junta todo o texto de cada listagem.
4. Passa esse texto por um modelo de embeddings (representações latentes)
   e guarda os vetores resultantes num CSV, um vetor por listagem.

Sobre o "LLM" para embeddings
------------------------------
A Anthropic não disponibiliza uma API pública de embeddings (as Claude models
são só para geração de texto). Duas opções válidas:

  A) 'sentence-transformers' (recomendado, precisa de internet no PC onde
     correres o script, para descarregar o modelo pré-treinado da Hugging
     Face na primeira utilização).
  B) 'tfidf' (fallback totalmente local, sem downloads, usa TF-IDF +
     Truncated SVD para produzir um espaço latente de dimensão reduzida —
     mais fraco semanticamente, mas útil se não tiveres acesso de rede
     ou quiseres algo rápido/determinístico para testar o pipeline).

Uso:
    python generate_text_embeddings.py --input listings.csv --method tfidf
    python generate_text_embeddings.py --input listings.csv --method sentence-transformers

Autor: <o teu nome>
Disciplina: Machine Learning - Practical Project 1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURAÇÃO DAS COLUNAS DE TEXTO
# ============================================================

TEXT_COLS = ["name", "description", "neighborhood_overview", "host_about"]

# Coluna(s) identificadoras a preservar no CSV final (ajusta se o teu
# dataset não tiver 'id' — usa-se para poderes fazer join de volta ao
# dataframe principal depois).
ID_COLS = ["id"]


# ============================================================
# 2. CARREGAR E ISOLAR TEXTO
# ============================================================

def load_text_only(csv_path: str, text_cols=TEXT_COLS, id_cols=ID_COLS) -> pd.DataFrame:
    """
    Lê o CSV e devolve um dataframe só com as colunas identificadoras
    e as colunas de texto pedidas (ignora todas as numéricas/categóricas).
    """
    df = pd.read_csv(csv_path)

    cols_presentes = [c for c in text_cols if c in df.columns]
    faltam = [c for c in text_cols if c not in df.columns]
    if faltam:
        print(f"[aviso] colunas de texto não encontradas no CSV, a ignorar: {faltam}")

    ids_presentes = [c for c in id_cols if c in df.columns]
    if not ids_presentes:
        # se não houver coluna de id, usa o índice do dataframe como id
        df = df.reset_index().rename(columns={"index": "row_id"})
        ids_presentes = ["row_id"]
        print("[aviso] nenhuma coluna de id encontrada, a usar o índice das linhas.")

    df_text = df[ids_presentes + cols_presentes].copy()
    return df_text, cols_presentes, ids_presentes


# ============================================================
# 3. CONCATENAR TEXTO
# ============================================================

def build_combined_text(df_text: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    """
    Cria a coluna 'combined_text', juntando todas as colunas de texto
    de cada linha. NaN é tratado como string vazia (não "None" literal,
    para não poluir o texto com ruído léxico).
    """
    df_text = df_text.copy()
    for col in text_cols:
        df_text[col] = df_text[col].fillna("").astype(str)

    df_text["combined_text"] = df_text[text_cols].agg(" ".join, axis=1).str.strip()

    n_vazio = (df_text["combined_text"].str.len() == 0).sum()
    if n_vazio > 0:
        print(f"[aviso] {n_vazio} listagens ficaram com texto totalmente vazio "
              f"(todas as colunas de texto em falta nessa linha).")

    return df_text


# ============================================================
# 4a. EMBEDDINGS — sentence-transformers (modelo pré-treinado, "LLM" leve)
# ============================================================

def compute_embeddings_sentence_transformers(texts: list[str],
                                              model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Gera embeddings com um modelo pré-treinado da biblioteca sentence-transformers.
    Precisa de 'pip install sentence-transformers' e de internet na primeira
    utilização (descarrega o modelo da Hugging Face, fica em cache localmente
    depois disso).

    all-MiniLM-L6-v2: modelo leve (80MB), rápido, 384 dimensões, boa relação
    qualidade/custo para este tipo de tarefa.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return np.array(embeddings)


# ============================================================
# 4b. EMBEDDINGS — TF-IDF + SVD (fallback local, sem downloads)
# ============================================================

def compute_embeddings_tfidf(texts: list[str], n_components: int = 100,
                              random_state: int = 42) -> np.ndarray:
    """
    Representação latente via TF-IDF seguido de Truncated SVD (LSA).
    Não precisa de rede nem de modelos pré-treinados — corre sempre,
    em qualquer ambiente. Mais fraco semanticamente que um modelo tipo
    sentence-transformers (não capta sinónimos/contexto), mas é uma
    baseline válida e totalmente reprodutível.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    # textos vazios ficam com vetor TF-IDF nulo; substitui por um placeholder
    # neutro para evitar linhas totalmente a zero, que distorcem o SVD
    texts_safe = [t if t.strip() else "sem informação disponível" for t in texts]

    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=2
    )
    X_tfidf = vectorizer.fit_transform(texts_safe)

    n_components = min(n_components, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_latent = svd.fit_transform(X_tfidf)

    variancia_explicada = svd.explained_variance_ratio_.sum()
    print(f"[info] TF-IDF+SVD: {n_components} componentes, "
          f"variância explicada acumulada = {variancia_explicada:.2%}")

    return X_latent


# ============================================================
# 5. PIPELINE PRINCIPAL
# ============================================================

def run_pipeline(input_path: str, output_path: str, method: str = "tfidf",
                  n_components: int = 100, model_name: str = "all-MiniLM-L6-v2"):

    print(f"A carregar: {input_path}")
    df_text, cols_presentes, ids_presentes = load_text_only(input_path)
    print(f"Colunas de texto usadas: {cols_presentes}")
    print(f"Coluna(s) de id: {ids_presentes}")

    df_text = build_combined_text(df_text, cols_presentes)

    textos = df_text["combined_text"].tolist()
    print(f"A gerar embeddings para {len(textos)} listagens, método='{method}'...")

    if method == "sentence-transformers":
        embeddings = compute_embeddings_sentence_transformers(textos, model_name=model_name)
    elif method == "tfidf":
        embeddings = compute_embeddings_tfidf(textos, n_components=n_components)
    else:
        raise ValueError(f"Método desconhecido: {method}. Usa 'tfidf' ou 'sentence-transformers'.")

    n_dims = embeddings.shape[1]
    print(f"Embeddings gerados: shape={embeddings.shape}")

    df_embeddings = pd.DataFrame(
        embeddings, columns=[f"text_emb_{i}" for i in range(n_dims)]
    )
    df_final = pd.concat(
        [df_text[ids_presentes].reset_index(drop=True), df_embeddings], axis=1
    )

    df_final.to_csv(output_path, index=False)
    print(f"Guardado em: {output_path}  (shape={df_final.shape})")

    # guarda também o texto combinado (útil para auditoria/debug, separado
    # dos embeddings para não inchar o CSV principal)
    combined_path = str(Path(output_path).with_name(Path(output_path).stem + "_combined_text.csv"))
    df_text[ids_presentes + ["combined_text"]].to_csv(combined_path, index=False)
    print(f"Texto combinado guardado em: {combined_path}")

    return df_final


# ============================================================
# 6. ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Gera embeddings a partir das colunas de texto do Airbnb")
    parser.add_argument("--input", type=str, required=True, help="Caminho para o CSV de listagens")
    parser.add_argument("--output", type=str, default="text_embeddings.csv", help="Caminho do CSV de saída")
    parser.add_argument("--method", type=str, default="tfidf",
                         choices=["tfidf", "sentence-transformers"],
                         help="Método de geração de embeddings")
    parser.add_argument("--n_components", type=int, default=100,
                         help="Nº de dimensões latentes (só para method=tfidf)")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2",
                         help="Nome do modelo (só para method=sentence-transformers)")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, method=args.method,
                 n_components=args.n_components, model_name=args.model_name)


if __name__ == "__main__":
    main()