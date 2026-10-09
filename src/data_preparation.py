"""
data_preparation.py
=====================
Pipeline completo de preparação de dados para o dataset Airbnb Lisboa,
replicando os passos do notebook `Airbnb_Lisboa_Predicao_Preco.ipynb`:

  1. Carregamento dos dados a partir de um ficheiro .zip (contém o CSV de
     listagens)
  2. Limpeza inicial de colunas (remove identificadores, URLs, metadados de
     scraping e texto livre)
  3. Tratamento de valores em falta (missing values)
  4. Deteção e tratamento de outliers em price, minimum_nights e
     accommodates (método MAD), segundo o tipo escolhido: "none", "cap" ou
     "filter"
  5. Preparação automática de features (prepare_features.py) — SEM fase de
     seleção de "melhores features" (sem núcleo por domínio, sem VIF/Lasso):
     usam-se diretamente todas as features produzidas automaticamente
  6. Guarda o dataset já tratado (features + target) num CSV dentro da
     pasta `Data_clean/`, com o nome do zip de origem e a estratégia de
     outliers usada (ex: `listings_cap.csv`)

Este módulo NÃO treina modelos — só prepara e guarda os dados tratados.

Uso (linha de comandos):
    python data_preparation.py data/listings.zip cap

Ou a partir de outro script/notebook:
    from data_preparation import preparar_dados, guardar_dados_tratados

    df_final, feature_cols, scale_cols = preparar_dados("data/listings.zip", tipo="cap")
    guardar_dados_tratados(df_final, "data/listings.zip", tipo="cap")

Autor: <o teu nome>
Disciplina: Machine Learning - Practical Project 1
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))  # permite importar os módulos irmãos

from prepare_features import (
    prepare_features, combine_host_time_features, KNOWN_REDUNDANT_COLS,
    adicionar_distancia_ao_centro,
)


# ============================================================
# 1. CARREGAMENTO DOS DADOS A PARTIR DO ZIP
# ============================================================

def carregar_csv_do_zip(caminho_zip: str) -> pd.DataFrame:
    """
    Lê o CSV de listagens a partir do ficheiro .zip indicado.

    O pandas descomprime automaticamente um .zip que contenha um único CSV
    lá dentro, por isso basta indicar o caminho do .zip (não é preciso
    extrair nada manualmente).
    """
    caminho_zip = Path(caminho_zip)
    if not caminho_zip.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {caminho_zip}")

    print(f"[info] a carregar dados de '{caminho_zip}'...")
    df = pd.read_csv(caminho_zip)
    print(f"[info] dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


# ============================================================
# 2. LIMPEZA INICIAL DE COLUNAS
# ============================================================

COLS_TO_DROP = [
    # Identificadores / URLs
    'id', 'listing_url', 'scrape_id', 'host_name', 'source', 'picture_url',
    'host_id', 'host_url', 'host_profile_id', 'host_profile_url',
    'host_thumbnail_url', 'host_picture_url', 'license',
    # Metadados de scraping
    'last_scraped', 'calendar_updated', 'calendar_last_scraped',
]
TEXT_COLS = ['name', 'description', 'neighborhood_overview', 'host_about']


def limpeza_inicial(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove identificadores/URLs, metadados de scraping e colunas de texto
    livre — não contribuem para a previsão do preço ou seriam leakage.
    """
    cols_existentes = [c for c in COLS_TO_DROP + TEXT_COLS if c in df.columns]
    df = df.drop(columns=cols_existentes)
    print(f"[info] limpeza inicial: {len(cols_existentes)} colunas removidas, restam {df.shape[1]}")
    return df


# ============================================================
# 3. VALORES EM FALTA (MISSING VALUES)
# ============================================================

REVIEW_SCORE_COLS = [
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value',
]

# Colunas com poucos NaN (erros de recolha) onde se opta por remover as
# instâncias em vez de imputar — impacto negligenciável no tamanho do dataset
DROPS_PONTUAIS = ['minimum_nights', 'hosts_time_as_user_years', 'bathrooms_text', 'has_availability']


def _imputar_por_grupo(df: pd.DataFrame, col: str, grupos=('room_type', 'accommodates')) -> pd.Series:
    """Imputa NaN pela mediana do grupo (room_type, accommodates); usa a mediana global como último recurso."""
    grupos = list(grupos)
    mediana_global = df[col].median()
    valores = df.groupby(grupos)[col].transform(lambda x: x.fillna(x.median()))
    return valores.fillna(mediana_global)


def tratar_valores_em_falta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata os valores em falta (NaN), coluna a coluna:
      - Remove colunas totalmente vazias
      - Remove instâncias sem 'price' (o target)
      - Drops pontuais em colunas com poucos NaN (erros de recolha)
      - host_location: NaN -> "Não especificado" (missing estrutural)
      - bathrooms/bedrooms/beds: imputação pela mediana do grupo
        (room_type, accommodates)
      - Bloco de reviews (MNAR estrutural): cria a flag 'has_reviews',
        imputa reviews_per_month a 0, e usa sentinela -1 nos
        review_scores_* (fora do range real 0-5, para não confundir com
        uma nota real)
      - Remove first_review / last_review (datas cruas, não usadas)
    """
    df = df.copy()

    # Colunas totalmente vazias
    n_rows = len(df)
    cols_vazias = df.columns[df.isna().sum() == n_rows]
    if len(cols_vazias) > 0:
        df = df.drop(columns=cols_vazias)
        print(f"[info] colunas totalmente vazias removidas: {list(cols_vazias)}")

    # Instâncias sem price
    df = df.dropna(subset=['price'])

    # Drops pontuais — erros de recolha, impacto negligenciável
    for col in DROPS_PONTUAIS:
        if col in df.columns:
            df = df.dropna(subset=[col])
    print(f"[info] linhas após drops pontuais: {len(df)}")

    # host_location — categórica, missing estrutural
    if 'host_location' in df.columns:
        df['host_location'] = df['host_location'].fillna('Não especificado')

    # bathrooms, bedrooms, beds — imputação por mediana condicional
    for col in ['bathrooms', 'bedrooms', 'beds']:
        if col in df.columns:
            n_antes = df[col].isnull().sum()
            df[col] = _imputar_por_grupo(df, col)
            n_depois = df[col].isnull().sum()
            print(f"[info] {col}: {n_antes} NaN antes -> {n_depois} NaN depois (imputação por mediana de grupo)")

    # Bloco de reviews — MNAR estrutural (opção: sem NaN no dataset final)
    if 'number_of_reviews' in df.columns:
        df['has_reviews'] = df['number_of_reviews'] > 0
    if 'reviews_per_month' in df.columns:
        df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    cols_scores_presentes = [c for c in REVIEW_SCORE_COLS if c in df.columns]
    if cols_scores_presentes:
        df[cols_scores_presentes] = df[cols_scores_presentes].fillna(-1)

    # first_review / last_review — datas cruas, removidas
    df = df.drop(columns=[c for c in ['first_review', 'last_review'] if c in df.columns])

    nans_restantes = df.isnull().sum()
    nans_restantes = nans_restantes[nans_restantes > 0]
    if nans_restantes.empty:
        print("[info] sem NaNs no dataset final.")
    else:
        print(f"[aviso] ainda há NaNs em:\n{nans_restantes}")

    print(f"[info] shape após tratamento de valores em falta: {df.shape}")
    return df


# ============================================================
# 4. DETEÇÃO E TRATAMENTO DE OUTLIERS (MAD)
# ============================================================

def detect_outliers_mad(series: pd.Series, threshold: float = 3.5):
    """Deteção robusta via z-score modificado (MAD) — Iglewicz & Hoaglin (1993)."""
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        mad = (series - median).abs().mean()
    if mad == 0:
        return pd.Series(False, index=series.index), series.min(), series.max()

    modified_z = 0.6745 * (series - median) / mad
    mask = modified_z.abs() > threshold
    lower_bound = median - (threshold * mad / 0.6745)
    upper_bound = median + (threshold * mad / 0.6745)
    return mask, lower_bound, upper_bound


def apply_strategy(df: pd.DataFrame, col: str, strategy: str, **kwargs) -> pd.DataFrame:
    """strategy: 'none' (não faz nada) | 'cap' (winsorization) | 'filter' (remove linhas)."""
    df = df.copy()
    if strategy == "none":
        return df

    mask, lower, upper = detect_outliers_mad(df[col], **kwargs)

    if strategy == "cap":
        df[col] = df[col].clip(lower=lower, upper=upper)
    elif strategy == "filter":
        df = df.loc[~mask].copy()
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy!r}. Usa 'none', 'cap' ou 'filter'.")
    return df


# Colunas alvo do tratamento de outliers, aplicadas sequencialmente
TARGET_OUTLIER_COLS = ["price", "minimum_nights", "accommodates"]


def tratar_outliers(df: pd.DataFrame, tipo: str) -> pd.DataFrame:
    """
    Aplica a estratégia de outliers escolhida (`tipo`) sobre price,
    minimum_nights e accommodates, sequencialmente (método MAD, threshold=3.5).

    tipo : 'none' (mantém tudo), 'cap' (winsorization) ou 'filter' (remove
        as linhas com outliers).
    """
    if tipo not in ("none", "cap", "filter"):
        raise ValueError(f"tipo de outlier desconhecido: {tipo!r}. Usa 'none', 'cap' ou 'filter'.")

    df = df.copy()

    # Limpeza mínima do price, se ainda estiver em string (ex: "$1,234.00")
    if not pd.api.types.is_numeric_dtype(df['price']):
        df['price'] = df['price'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)

    n_antes = len(df)
    for col in TARGET_OUTLIER_COLS:
        if col in df.columns:
            df = apply_strategy(df, col, strategy=tipo)

    print(f"[info] outliers (tipo='{tipo}'): {n_antes} -> {len(df)} linhas")
    return df


# ============================================================
# 5. PREPARAÇÃO DE FEATURES (sem seleção de "melhores features")
# ============================================================

def preparar_features_finais(df: pd.DataFrame, target_col: str = "price", force_continuous=None,
                              usar_distancia_centro: bool = False):
    """
    Usa prepare_features() para classificar automaticamente cada coluna
    (contínua / binária / categórica / excluir) e usa TODAS as features
    resultantes diretamente como conjunto final — sem qualquer fase de
    seleção manual ou estatística (sem núcleo por domínio, sem
    correlação/VIF/Lasso).

    force_continuous : colunas a forçar como contínuas (scale_cols), mesmo
        que a heurística automática (baseada no nº de valores únicos NESTE
        dataframe) as classificasse como discretas/one-hot. Por omissão só
        força 'minimum_nights'. IMPORTANTE: ao preparar um dataset novo para
        avaliar um modelo já treinado noutro dataset, passa aqui o
        `scale_cols` desse modelo — caso contrário, colunas como
        bathrooms/bedrooms/beds podem ter cardinalidade baixa no dataset
        novo (ex: amostra pequena) e ser classificadas como discretas em vez
        de contínuas, ficando com nomes de coluna diferentes dos esperados
        pelo modelo (e a avaliação perde essa informação).
    usar_distancia_centro : se True, substitui latitude/longitude absolutas
        por 'dist_centro_km' (distância ao centro geográfico DESTE dataset —
        ver adicionar_distancia_ao_centro em prepare_features.py). Torna a
        feature geográfica comparável entre cidades diferentes, ao contrário
        de coordenadas absolutas.
    """
    if force_continuous is None:
        force_continuous = ["minimum_nights"]

    df = combine_host_time_features(df)
    if usar_distancia_centro:
        df, centro = adicionar_distancia_ao_centro(df)
        print(f"[info] latitude/longitude substituídas por dist_centro_km (centro usado: {centro})")

    df_final, feature_cols, scale_cols = prepare_features(
        df, target_col=target_col,
        exclude_cols=KNOWN_REDUNDANT_COLS,
        force_continuous=force_continuous,
    )
    df_final = df_final.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    print(f"[info] features finais (sem seleção): {len(feature_cols)} ({len(scale_cols)} contínuas)")
    return df_final, feature_cols, scale_cols


# ============================================================
# 6. PIPELINE DE PREPARAÇÃO COMPLETO
# ============================================================

def preparar_dados(caminho_zip: str, tipo: str = "cap", target_col: str = "price", force_continuous=None,
                    usar_distancia_centro: bool = False):
    """
    Pipeline completo: zip -> limpeza inicial -> valores em falta ->
    outliers -> features.

    Parâmetros
    ----------
    caminho_zip : caminho para o .zip que contém o CSV de listagens
    tipo : estratégia de tratamento de outliers: 'none', 'cap' ou 'filter'
    target_col : nome da coluna alvo (default: 'price')
    force_continuous : ver preparar_features_finais() — passa aqui o
        `scale_cols` de um modelo já treinado quando este dataset se destina
        a ser avaliado contra esse modelo (ver evaluate_saved_model() em
        linear_regression.py)
    usar_distancia_centro : ver preparar_features_finais() — substitui
        latitude/longitude por dist_centro_km, comparável entre cidades

    Retorna
    -------
    df_final : dataframe pronto a usar (já sem NaN, features automáticas
        geradas, sem fase de seleção de features)
    feature_cols : lista de todas as colunas de features
    scale_cols : subconjunto de feature_cols a escalar/expandir
        polinomialmente
    """
    df = carregar_csv_do_zip(caminho_zip)
    df = limpeza_inicial(df)
    df = tratar_valores_em_falta(df)
    df = tratar_outliers(df, tipo=tipo)
    df_final, feature_cols, scale_cols = preparar_features_finais(
        df, target_col=target_col, force_continuous=force_continuous,
        usar_distancia_centro=usar_distancia_centro,
    )
    return df_final, feature_cols, scale_cols


# ============================================================
# 7. GUARDAR OS DADOS TRATADOS EM Data_clean/
# ============================================================

def guardar_dados_tratados(df_final: pd.DataFrame, caminho_zip: str, tipo: str,
                            outdir: str = "Data_clean") -> Path:
    """
    Guarda o dataframe já tratado (features + target, sem NaN) num CSV
    dentro da pasta `outdir`, com o nome do zip de origem e a estratégia de
    outliers usada, ex: 'listings_cap.csv'.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # nome do ficheiro sem qualquer extensão (usa a primeira parte antes do
    # primeiro ponto, para lidar bem com extensões duplas como .csv.gz)
    nome_base = Path(caminho_zip).name.split(".")[0]
    caminho_saida = outdir / f"{nome_base}_{tipo}.csv"
    df_final.to_csv(caminho_saida, index=False)

    print(f"[info] dados tratados guardados em: {caminho_saida} (shape={df_final.shape})")
    return caminho_saida


# ============================================================
# 8. ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepara os dados Airbnb (a partir de um .zip) e guarda o dataset tratado em Data_clean/ "
                    "(sem seleção de features)."
    )
    parser.add_argument("caminho_zip", type=str, help="Caminho para o ficheiro .zip com o CSV de listagens")
    parser.add_argument("tipo", type=str, choices=["none", "cap", "filter"],
                         help="Estratégia de tratamento de outliers: 'none', 'cap' ou 'filter'")
    parser.add_argument("--outdir", type=str, default="Data_clean",
                         help="Pasta onde guardar o CSV tratado (default: 'Data_clean')")
    args = parser.parse_args()

    df_final, feature_cols, scale_cols = preparar_dados(args.caminho_zip, tipo=args.tipo)
    guardar_dados_tratados(df_final, args.caminho_zip, args.tipo, outdir=args.outdir)


if __name__ == "__main__":
    main()
