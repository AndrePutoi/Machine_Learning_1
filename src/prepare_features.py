"""
prepare_features.py
=====================
Prepara AUTOMATICAMENTE todas as colunas de um dataframe (já tratado na
Fase 2 - outliers) para serem usadas no linear_regression.py, sem teres
de escolher feature_cols/scale_cols à mão.

Decide, coluna a coluna:
  - EXCLUIR: leakage conhecido, IDs, texto livre/alta cardinalidade, datas
  - CONTÍNUA (scale_cols): numérica com muitos valores distintos -> escalada
    (sklearn StandardScaler) e expandida polinomialmente
  - BINÁRIA/CATEGÓRICA (passthrough): 't'/'f', True/False, ou categórica de
    baixa cardinalidade -> convertida para 0/1 ou one-hot, sem escala nem
    expansão polinomial

Uso:
    from prepare_features import prepare_features
    df_final, feature_cols, scale_cols = prepare_features(df, target_col="price")
"""

import pandas as pd
import numpy as np


# Colunas conhecidas a excluir sempre (ajusta consoante o teu dataset real)
DEFAULT_EXCLUDE = {
    # identificadores
    "id", "listing_id", "host_id", "scrape_id",
    # leakage claro (derivado do próprio price_quote, é essencialmente o preço)
    "price_quote_total_price", "price_quote_price_per_night", "price_quote_raw",
    "price_quote_checkin_date", "price_quote_checkout_date",
    "estimated_revenue_l365d",  # suspeito de leakage -> testa correlação antes de reintroduzir
    # texto livre / alta cardinalidade sem valor direto como feature numérica
    "amenities", "name", "description", "neighborhood_overview", "host_about",
    "host_location", "neighbourhood_cleansed", "bathrooms_text",
    "listing_url", "picture_url", "host_url", "host_thumbnail_url", "host_picture_url",
    # datas cruas (usa-as só se as converteres para algo numérico, ex: dias desde)
    "first_review", "last_review", "host_since", "calendar_last_scraped", "last_scraped",
}


def prepare_features(df, target_col, exclude_cols=None, force_continuous=None,
                      max_onehot_cardinality=20, min_continuous_unique=10,
                      verbose=True):
    """
    Prepara automaticamente as colunas de df para usar no modelo.

    Parâmetros
    ----------
    df : DataFrame já tratado (Fase 2 - outliers)
    target_col : nome da coluna alvo (excluída automaticamente das features)
    exclude_cols : set/list adicional de colunas a excluir, além do DEFAULT_EXCLUDE
    force_continuous : set/list de colunas a forçar para scale_cols mesmo que
        a heurística automática as classifique como "discretas" (útil quando
        sabes que a variável é semanticamente contínua, ex: minimum_nights
        que ficou com poucos valores únicos depois do capping de outliers)
    max_onehot_cardinality : categóricas com mais categorias que isto são
        EXCLUÍDAS (não fazem one-hot, para não explodir dimensões)
    min_continuous_unique : numéricas com menos valores distintos que isto
        são tratadas como binárias/discretas (passthrough), não contínuas
    verbose : imprime um resumo do que foi feito com cada coluna

    Retorna
    -------
    df_final : dataframe só com as colunas finais prontas a usar
    feature_cols : lista de todas as colunas de features (scale + passthrough)
    scale_cols : subconjunto de feature_cols a passar a scale_cols no
                 run_experiment / run_kfold_experiment
    """
    exclude = set(DEFAULT_EXCLUDE)
    if exclude_cols:
        exclude |= set(exclude_cols)
    exclude.add(target_col)

    force_continuous = set(force_continuous) if force_continuous else set()

    df_final = pd.DataFrame(index=df.index)
    scale_cols = []
    passthrough_cols = []
    resumo = []

    for col in df.columns:
        if col in exclude:
            resumo.append((col, "excluída", "lista de exclusão"))
            continue

        series = df[col]

        # --- booleanas True/False ---
        if series.dtype == bool:
            df_final[col] = series.astype(int)
            passthrough_cols.append(col)
            resumo.append((col, "passthrough (bool->0/1)", ""))
            continue

        # --- 't'/'f' estilo Airbnb ---
        if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) \
                and set(series.dropna().unique()) <= {"t", "f"}:
            df_final[col] = series.map({"t": 1, "f": 0})
            passthrough_cols.append(col)
            resumo.append((col, "passthrough (t/f->0/1)", ""))
            continue

        # --- numéricas ---
        if pd.api.types.is_numeric_dtype(series):
            n_unique = series.nunique(dropna=True)
            if col in force_continuous:
                df_final[col] = series
                scale_cols.append(col)
                resumo.append((col, "scale_cols (contínua, forçada)", f"{n_unique} valores únicos"))
            elif n_unique <= 2:
                df_final[col] = series
                passthrough_cols.append(col)
                resumo.append((col, "passthrough (numérica binária)", f"{n_unique} valores únicos"))
            elif n_unique < min_continuous_unique:
                dummies = pd.get_dummies(series, prefix=col, drop_first=True).astype(int)
                df_final = pd.concat([df_final, dummies], axis=1)
                passthrough_cols.extend(dummies.columns.tolist())
                resumo.append((col, f"one-hot ({len(dummies.columns)} cols)",
                                f"{n_unique} valores únicos, tratada como discreta"))
            else:
                df_final[col] = series
                scale_cols.append(col)
                resumo.append((col, "scale_cols (contínua)", f"{n_unique} valores únicos"))
            continue

        # --- categóricas (object/category) ---
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) \
                or str(series.dtype) == "category":
            n_unique = series.nunique(dropna=True)
            if n_unique > max_onehot_cardinality:
                resumo.append((col, "excluída", f"alta cardinalidade ({n_unique} categorias)"))
                continue
            dummies = pd.get_dummies(series, prefix=col, drop_first=True).astype(int)
            df_final = pd.concat([df_final, dummies], axis=1)
            passthrough_cols.extend(dummies.columns.tolist())
            resumo.append((col, f"one-hot ({len(dummies.columns)} cols)", f"{n_unique} categorias"))
            continue

        resumo.append((col, "excluída", f"dtype não tratado: {series.dtype}"))

    df_final[target_col] = df[target_col]
    feature_cols = scale_cols + passthrough_cols

    if verbose:
        print(f"{'coluna':35s} {'ação':30s} {'nota'}")
        print("-" * 90)
        for col, acao, nota in resumo:
            print(f"{col:35s} {acao:30s} {nota}")
        print("-" * 90)
        print(f"Total: {len(scale_cols)} contínuas (scale_cols), "
              f"{len(passthrough_cols)} passthrough, "
              f"{len(feature_cols)} features finais.")

    return df_final, feature_cols, scale_cols


# ============================================================
# Combinação de colunas de tempo de host (reduz multicolinearidade)
# ============================================================

def combine_host_time_features(df):
    """
    Combina hosts_time_as_user_years + hosts_time_as_user_months numa só
    coluna (host_user_total_months), e o mesmo para host. Reduz 4 colunas
    correlacionadas a 2, evitando multicolinearidade desnecessária.
    Chamar isto ANTES de prepare_features().
    """
    df = df.copy()
    cols_user = ["hosts_time_as_user_years", "hosts_time_as_user_months"]
    cols_host = ["hosts_time_as_host_years", "hosts_time_as_host_months"]

    if all(c in df.columns for c in cols_user):
        df["host_user_total_months"] = df["hosts_time_as_user_years"] * 12 + df["hosts_time_as_user_months"]
        df = df.drop(columns=cols_user)

    if all(c in df.columns for c in cols_host):
        df["host_host_total_months"] = df["hosts_time_as_host_years"] * 12 + df["hosts_time_as_host_months"]
        df = df.drop(columns=cols_host)

    return df


# ============================================================
# Distância ao centro da cidade (substitui latitude/longitude absolutas)
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Distância Haversine (em km) entre pontos (lat1, lon1) e (lat2, lon2)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def adicionar_distancia_ao_centro(df, lat_col="latitude", lon_col="longitude", drop_coords=True):
    """
    Substitui latitude/longitude absolutas por uma única feature contínua
    'dist_centro_km': a distância (Haversine, em km) de cada listagem ao
    centro geográfico do PRÓPRIO dataset (centróide de lat/lon desse
    dataset).

    Torna a feature comparável entre cidades diferentes: '0' passa a
    significar sempre "no centro da cidade em causa", em vez de depender de
    coordenadas absolutas (que variam completamente de cidade para cidade e
    tornam o modelo inútil fora da cidade onde foi treinado — ver secção
    sobre o scaler ajustado a Lisboa aplicado a Nova Iorque).

    Chamar isto ANTES de prepare_features(), tal como
    combine_host_time_features().

    Retorna
    -------
    df : dataframe com 'dist_centro_km' (e sem lat/lon, se drop_coords=True)
    centro : tuplo (lat_centro, lon_centro) usado, por transparência/debug
    """
    df = df.copy()
    lat_centro = df[lat_col].mean()
    lon_centro = df[lon_col].mean()
    df["dist_centro_km"] = haversine_km(df[lat_col], df[lon_col], lat_centro, lon_centro)
    if drop_coords:
        df = df.drop(columns=[lat_col, lon_col])
    return df, (lat_centro, lon_centro)


# ============================================================
# Lista de colunas redundantes conhecidas neste dataset (Airbnb Lisboa)
# ============================================================

KNOWN_REDUNDANT_COLS = {
    # variações de nights redundantes com minimum_nights / maximum_nights
    "minimum_minimum_nights", "maximum_minimum_nights",
    "minimum_maximum_nights", "maximum_maximum_nights",
    "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
    # disponibilidade altamente correlacionada -> mantém-se só availability_365
    "availability_30", "availability_60", "availability_90", "availability_eoy",
    # reviews redundante -> mantém-se só number_of_reviews
    "number_of_reviews_ltm", "number_of_reviews_l30d", "number_of_reviews_ly",
    # subtotal de calculated_host_listings_count, soma ≈ ao total
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    # constante (1 único valor) -> zero informação
    "has_availability",
}