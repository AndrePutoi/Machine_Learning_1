"""
inspect_columns.py
=====================
Corre isto sobre um dos CSVs reais gerados pela Fase 2 (outlier_versions/
listings_none.csv, listings_cap.csv ou listings_filter.csv) para obteres
um resumo de todas as colunas: dtype, nº de valores únicos, e uma sugestão
automática de scale (contínua) vs passthrough (binária/categórica) vs
excluir (leakage/texto/alta cardinalidade).

Uso:
    python inspect_columns.py --input outlier_versions/listings_cap.csv

Ou numa célula Jupyter:
    from inspect_columns import inspect
    resumo = inspect(df)
"""

import argparse
import pandas as pd
import numpy as np


DEFAULT_EXCLUDE_HINTS = {
    "id", "listing_id", "host_id", "scrape_id",
    "price_quote_total_price", "price_quote_price_per_night", "price_quote_raw",
    "price_quote_checkin_date", "price_quote_checkout_date",
    "estimated_revenue_l365d",
    "amenities", "name", "description", "neighborhood_overview", "host_about",
    "host_location", "neighbourhood_cleansed", "bathrooms_text",
    "listing_url", "picture_url", "host_url", "host_thumbnail_url", "host_picture_url",
    "first_review", "last_review", "host_since", "calendar_last_scraped", "last_scraped",
}


def sugestao_para_coluna(col, series, max_onehot_cardinality=20, min_continuous_unique=10):
    """Sugere uma categoria para a coluna, sem decidir nada definitivamente —
    é só um ponto de partida para revisares."""
    if col in DEFAULT_EXCLUDE_HINTS:
        return "excluir (na lista de exclusão conhecida)"

    if series.dtype == bool:
        return "passthrough (bool -> 0/1)"

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        valores_unicos = set(series.dropna().unique())
        if valores_unicos <= {"t", "f"}:
            return "passthrough (t/f -> 0/1)"
        n_unique = series.nunique(dropna=True)
        if n_unique > max_onehot_cardinality:
            return f"excluir (texto/alta cardinalidade, {n_unique} categorias)"
        return f"one-hot / passthrough ({n_unique} categorias)"

    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique(dropna=True)
        if n_unique <= 2:
            return "passthrough (numérica binária)"
        if n_unique < min_continuous_unique:
            return f"discreta ({n_unique} valores) -> avaliar: passthrough/one-hot ou contínua"
        return "scale_cols (contínua)"

    return f"rever manualmente (dtype: {series.dtype})"


def inspect(df, max_onehot_cardinality=20, min_continuous_unique=10):
    """Devolve um dataframe-resumo: coluna, dtype, n_unique, n_nulls, sugestão, exemplos."""
    linhas = []
    for col in df.columns:
        series = df[col]
        n_unique = series.nunique(dropna=True)
        n_nulls = series.isnull().sum()
        sugestao = sugestao_para_coluna(col, series, max_onehot_cardinality, min_continuous_unique)

        try:
            exemplos = series.dropna().unique()[:5].tolist()
        except TypeError:
            exemplos = list(series.dropna().unique()[:5])

        linhas.append({
            "coluna": col,
            "dtype": str(series.dtype),
            "n_unique": n_unique,
            "n_nulls": n_nulls,
            "sugestao": sugestao,
            "exemplos": exemplos,
        })

    resumo = pd.DataFrame(linhas)
    return resumo


def main():
    parser = argparse.ArgumentParser(description="Inspeção de colunas de um CSV")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--max_onehot_cardinality", type=int, default=20)
    parser.add_argument("--min_continuous_unique", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    resumo = inspect(df, args.max_onehot_cardinality, args.min_continuous_unique)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", 60)
    print(resumo.to_string(index=False))

    out_path = args.input.replace(".csv", "_column_inspection.csv")
    resumo.to_csv(out_path, index=False)
    print(f"\nResumo guardado em: {out_path}")


if __name__ == "__main__":
    main()