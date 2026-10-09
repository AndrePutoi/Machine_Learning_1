"""
linear_regression.py
=====================
Regressão linear com Gradient Descent implementado do zero, full-batch.

Simplificado deliberadamente: sem early stopping, sem parâmetros de
gradient clipping/divergência configuráveis. Corre sempre até n_epochs;
se divergir (NaN/Inf), para e avisa, restaurando o último checkpoint
válido (comportamento fixo, não configurável).

Este módulo NÃO trata outliers — assume que recebe um dataframe já tratado
(um dos CSVs gerados por save_outlier_strategies.py).

Autor: <o teu nome>
Disciplina: Machine Learning - Practical Project 1
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold


# ============================================================
# 1. REGRESSÃO LINEAR COM GRADIENT DESCENT (from scratch, full-batch)
# ============================================================

class LinearRegressionGD:
    """
    Regressão linear treinada por full-batch gradient descent, com
    regularização L2 (Ridge) opcional.

    Parâmetros
    ----------
    learning_rate : float
        Taxa de aprendizagem (alpha).
    n_epochs : int
        Número de épocas — corre sempre este número completo (sem early
        stopping).
    l2_lambda : float
        Força da regularização L2. 0.0 = sem regularização (OLS puro).
    random_state : int
        Semente para reprodutibilidade.
    verbose : bool
        Se True, imprime o loss a cada `print_every` épocas.
    print_every : int
        Frequência (em épocas) da impressão quando verbose=True.
    checkpoint_on : {"train", "val"}
        Loss usada para escolher o "melhor checkpoint".
        "train" (default): usa a loss de treino -> a validação NÃO influencia
        os pesos finais (necessário para a validação ser uma estimativa
        honesta). "val": escolhe a época pela loss de validação (equivale a
        early stopping sobre o fold de validação -> métricas otimistas).
    """

    def __init__(self, learning_rate=0.01, n_epochs=1000, l2_lambda=0.0,
                 random_state=42, verbose=False, print_every=10,
                 checkpoint_on="train"):
        if checkpoint_on not in ("train", "val"):
            raise ValueError(f"checkpoint_on deve ser 'train' ou 'val', recebido: {checkpoint_on!r}")
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2_lambda = l2_lambda
        self.random_state = random_state
        self.verbose = verbose
        self.print_every = print_every
        self.checkpoint_on = checkpoint_on

        self.weights_ = None
        self.bias_ = None
        self.loss_history_ = []
        self.val_loss_history_ = []
        self.diverged_ = False
        self.diverged_epoch_ = None

        self.best_weights_ = None
        self.best_bias_ = None
        self.best_loss_ = np.inf
        self.best_epoch_ = None

    def _compute_loss(self, X, y, weights, bias):
        """MSE + termo de regularização L2 (não penaliza o bias)."""
        n = len(y)
        y_pred = X @ weights + bias
        mse = np.mean((y_pred - y) ** 2)
        l2_term = self.l2_lambda * np.sum(weights ** 2) / n
        return mse + l2_term

    def _restore_checkpoint(self, motivo=""):
        if self.best_weights_ is not None:
            if self.verbose:
                print(f"[checkpoint] A restaurar pesos da época {self.best_epoch_} "
                      f"(loss={self.best_loss_:.4f}). {motivo}")
            self.weights_ = self.best_weights_
            self.bias_ = self.best_bias_
        else:
            # divergiu antes de existir qualquer checkpoint válido: volta a zeros
            # (evita prever NaN); o flag diverged_ fica True para o registo.
            self.weights_ = np.zeros_like(self.weights_)
            self.bias_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None,
            y_train_original=None, y_val_original=None, inverse_transform=None):
        """
        y_train_original / y_val_original: valores REAIS do target (ex: price
        em euros), quando y é uma versão transformada (ex: log1p(price)).
        inverse_transform: função que desfaz a transformação (ex: np.expm1).
        Usados só para reporte (MAE/RMSE em escala original durante o print).
        """
        n_samples, n_features = X.shape

        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_ = []
        self.val_loss_history_ = []
        self.diverged_ = False
        self.diverged_epoch_ = None
        self.best_weights_ = None
        self.best_bias_ = None
        self.best_loss_ = np.inf
        self.best_epoch_ = None

        for epoch in range(self.n_epochs):
            # full-batch: um único update por época, usando todo o treino
            y_pred = X @ self.weights_ + self.bias_
            error = y_pred - y

            grad_w = (2 / n_samples) * (X.T @ error) + (2 * self.l2_lambda / n_samples) * self.weights_
            grad_b = (2 / n_samples) * np.sum(error)

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

            # ---- proteção mínima contra divergência (não configurável) ----
            if not np.all(np.isfinite(self.weights_)) or not np.isfinite(self.bias_):
                self.diverged_ = True
                self.diverged_epoch_ = epoch
                print(f"[DIVERGÊNCIA] Treino divergiu na época {epoch} "
                      f"(pesos/bias tornaram-se NaN/Inf). Reduz learning_rate.")
                self._restore_checkpoint("(divergência)")
                return self

            epoch_loss = self._compute_loss(X, y, self.weights_, self.bias_)
            if not np.isfinite(epoch_loss):
                self.diverged_ = True
                self.diverged_epoch_ = epoch
                print(f"[DIVERGÊNCIA] Loss tornou-se NaN/Inf na época {epoch}.")
                self._restore_checkpoint("(divergência)")
                return self

            self.loss_history_.append(epoch_loss)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._compute_loss(X_val, y_val, self.weights_, self.bias_)
                self.val_loss_history_.append(val_loss)

            # ---- checkpoint do melhor modelo visto ----
            # Por defeito usa a loss de TREINO: a validação só mede, não decide.
            if self.checkpoint_on == "val" and val_loss is not None:
                loss_ref = val_loss
            else:
                loss_ref = epoch_loss

            if loss_ref < self.best_loss_:
                self.best_loss_ = loss_ref
                self.best_weights_ = self.weights_.copy()
                self.best_bias_ = self.bias_
                self.best_epoch_ = epoch

            if self.verbose and epoch % self.print_every == 0:
                msg = f"Época {epoch:4d} | loss(interno) treino: {epoch_loss:.4f}"
                if self.val_loss_history_:
                    msg += f" | loss(interno) val: {self.val_loss_history_[-1]:.4f}"

                if y_train_original is not None:
                    pred_train_raw = X @ self.weights_ + self.bias_
                    if inverse_transform is not None:
                        pred_train_raw = inverse_transform(pred_train_raw)
                    mae_t = np.mean(np.abs(y_train_original - pred_train_raw))
                    rmse_t = np.sqrt(np.mean((y_train_original - pred_train_raw) ** 2))
                    msg += f" || MAE(orig) treino: {mae_t:.2f} | RMSE(orig) treino: {rmse_t:.2f}"

                if y_val_original is not None and X_val is not None:
                    pred_val_raw = X_val @ self.weights_ + self.bias_
                    if inverse_transform is not None:
                        pred_val_raw = inverse_transform(pred_val_raw)
                    mae_v = np.mean(np.abs(y_val_original - pred_val_raw))
                    rmse_v = np.sqrt(np.mean((y_val_original - pred_val_raw) ** 2))
                    msg += f" | MAE(orig) val: {mae_v:.2f} | RMSE(orig) val: {rmse_v:.2f}"

                print(msg)

        self._restore_checkpoint("(fim do treino)")
        return self

    def predict(self, X: np.ndarray):
        return X @ self.weights_ + self.bias_

    def plot_convergence(self, title="Curva de convergência", ax=None):
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(self.loss_history_, label="Loss treino", color="tab:blue")
        if self.val_loss_history_:
            ax.plot(self.val_loss_history_, label="Loss validação", color="tab:orange")
        if self.best_epoch_ is not None:
            ax.axvline(self.best_epoch_, color="green", linestyle=":", alpha=0.7,
                        label=f"Melhor checkpoint (época {self.best_epoch_})")
        ax.set_xlabel("Época")
        ax.set_ylabel("MSE (+ L2)")
        ax.set_title(title)
        ax.legend()

        if own_fig:
            plt.tight_layout()
            plt.show()
        return ax


# ============================================================
# 2. FEATURE ENGINEERING — EXPANSÃO POLINOMIAL + ESCALA SELETIVA
# ============================================================

def build_design_matrix(df_train, df_val, df_test, scale_cols, passthrough_cols, degree=1):
    """
    Constrói a matriz de design, tratando de forma diferente:
      - scale_cols: variáveis contínuas -> expandidas polinomialmente
        (PolynomialFeatures) e depois standardizadas (sklearn StandardScaler,
        fit só no treino).
      - passthrough_cols: variáveis que NÃO fazem sentido escalar nem
        expandir (ex: binárias/indicadoras) -> passam tal como estão.
    """
    scale_cols = list(scale_cols) if scale_cols else []
    passthrough_cols = list(passthrough_cols) if passthrough_cols else []

    if scale_cols:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(df_train[scale_cols])

        # copy=False: escala in-place sobre a matriz polinomial (no grau 4 são
        # ~2.5 GB por cópia); o resultado numérico é o mesmo
        scaler = StandardScaler(copy=False)
        X_train_scaled = scaler.fit_transform(X_train_poly)

        X_val_scaled = None
        if df_val is not None:
            X_val_poly = poly.transform(df_val[scale_cols])
            X_val_scaled = scaler.transform(X_val_poly)

        X_test_scaled = None
        if df_test is not None:
            X_test_poly = poly.transform(df_test[scale_cols])
            X_test_scaled = scaler.transform(X_test_poly)
    else:
        poly, scaler = None, None
        X_train_scaled = np.empty((len(df_train), 0))
        X_val_scaled = np.empty((len(df_val), 0)) if df_val is not None else None
        X_test_scaled = np.empty((len(df_test), 0)) if df_test is not None else None

    def _concat(scaled_part, df_source):
        if df_source is None:
            return None
        if passthrough_cols:
            passthrough_part = df_source[passthrough_cols].to_numpy(dtype=float)
            return np.hstack([scaled_part, passthrough_part])
        return scaled_part

    X_train = _concat(X_train_scaled, df_train)
    X_val = _concat(X_val_scaled, df_val)
    X_test = _concat(X_test_scaled, df_test)

    return X_train, X_val, X_test, poly, scaler


# ============================================================
# 3. MÉTRICAS
# ============================================================

def compute_metrics(y_true, y_pred):
    """MSE, RMSE, MAE e R2 (escala original do target)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


# ============================================================
# 4. VALIDAÇÃO DE INPUTS
# ============================================================

def validate_inputs(df, feature_cols, target_col, degree=1,
                     test_size=None, val_size=None):
    cols_em_falta = [c for c in feature_cols + [target_col] if c not in df.columns]
    if cols_em_falta:
        raise ValueError(f"Colunas não encontradas no dataframe: {cols_em_falta}")

    cols_usadas = feature_cols + [target_col]
    subset = df[cols_usadas]

    nao_numericas = [c for c in cols_usadas if not pd.api.types.is_numeric_dtype(subset[c])]
    if nao_numericas:
        raise ValueError(
            f"As colunas seguintes não são numéricas: {nao_numericas}. "
            f"Codifica-as primeiro (one-hot, label encoding, etc.)."
        )

    n_nan = subset.isnull().sum()
    cols_com_nan = n_nan[n_nan > 0]
    if len(cols_com_nan) > 0:
        raise ValueError(
            f"Existem valores em falta (NaN) nas colunas usadas:\n"
            f"{cols_com_nan.to_dict()}\n"
            f"Trata os NaN antes de chamar esta função."
        )

    n_inf = np.isinf(subset.to_numpy(dtype=float)).sum()
    if n_inf > 0:
        raise ValueError(f"Existem {n_inf} valores infinitos nas colunas usadas.")

    if not isinstance(degree, int) or degree < 1:
        raise ValueError(f"degree tem de ser um inteiro >= 1, recebido: {degree!r}")

    if test_size is not None and val_size is not None:
        if test_size < 0 or val_size < 0 or (test_size + val_size) >= 1:
            raise ValueError(
                f"test_size + val_size tem de ser < 1. Recebido: "
                f"test_size={test_size}, val_size={val_size}, soma={test_size + val_size}"
            )


# ============================================================
# 5. PIPELINE COMPLETO — UMA ÚNICA CONFIGURAÇÃO
# ============================================================

def run_experiment(df, feature_cols, target_col, degree=1,
                    scale_cols=None, test_size=0.2, val_size=0.2,
                    learning_rate=0.01, n_epochs=1000, l2_lambda=0.0,
                    random_state=42, verbose=False,
                    log_target=False, plot=True, checkpoint_on="train"):
    """
    Pipeline completo, a partir de um dataframe JÁ TRATADO:
      0. Validação de inputs
      1. Split train/val/test (sem leakage)
      2. Expansão polinomial + StandardScaler, só em scale_cols
      3. Treino full-batch com Gradient Descent, L2
      4. Avaliação em treino/val/teste

    scale_cols : lista ou None
        Subconjunto de feature_cols a expandir polinomialmente e escalar.
        Se None, todas as feature_cols são escaladas.
    """
    validate_inputs(df, feature_cols, target_col, degree=degree,
                     test_size=test_size, val_size=val_size)

    if log_target and (df[target_col] < 0).any():
        raise ValueError(f"log_target=True mas '{target_col}' tem valores negativos.")

    if scale_cols is None:
        scale_cols = list(feature_cols)
    passthrough_cols = [c for c in feature_cols if c not in scale_cols]

    rng = np.random.default_rng(random_state)
    n = len(df)
    indices = rng.permutation(n)

    n_test = int(n * test_size)
    n_val = int(n * val_size)
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    y_train = df_train[target_col].to_numpy(dtype=float)
    y_val = df_val[target_col].to_numpy(dtype=float)
    y_test = df_test[target_col].to_numpy(dtype=float)

    y_train_model = np.log1p(y_train) if log_target else y_train
    y_val_model = np.log1p(y_val) if log_target else y_val

    X_train, X_val, X_test, poly, scaler = build_design_matrix(
        df_train, df_val, df_test, scale_cols, passthrough_cols, degree=degree
    )

    model = LinearRegressionGD(
        learning_rate=learning_rate, n_epochs=n_epochs, l2_lambda=l2_lambda,
        random_state=random_state, verbose=verbose, checkpoint_on=checkpoint_on
    )
    model.fit(
        X_train, y_train_model, X_val=X_val, y_val=y_val_model,
        y_train_original=y_train, y_val_original=y_val,
        inverse_transform=np.expm1 if log_target else None,
    )

    def _predict_original_scale(X):
        y_pred = model.predict(X)
        return np.expm1(y_pred) if log_target else y_pred

    y_pred_train = _predict_original_scale(X_train)
    y_pred_val = _predict_original_scale(X_val)
    y_pred_test = _predict_original_scale(X_test)

    metrics = {
        "train": compute_metrics(y_train, y_pred_train),
        "val": compute_metrics(y_val, y_pred_val),
        "test": compute_metrics(y_test, y_pred_test),
    }

    if plot:
        model.plot_convergence(title=f"Convergência — grau={degree}, λ={l2_lambda}")

    return {
        "model": model, "scaler": scaler, "poly": poly, "metrics": metrics,
        "scale_cols": scale_cols, "passthrough_cols": passthrough_cols,
        "n_features_final": X_train.shape[1],
        "config": {
            "degree": degree, "l2_lambda": l2_lambda, "learning_rate": learning_rate,
            "n_epochs": n_epochs, "log_target": log_target,
            "checkpoint_on": checkpoint_on,
        },
    }


# ============================================================
# 6. PIPELINE COM K-FOLD CV
# ============================================================

def run_kfold_experiment(df, feature_cols, target_col, degree=1,
                          scale_cols=None, n_splits=5, learning_rate=0.01,
                          n_epochs=1000, l2_lambda=0.0,
                          random_state=42, log_target=False,
                          verbose=False, print_every=10, return_history=True,
                          checkpoint_on="train"):
    """
    Versão com k-fold CV, a partir de um dataframe JÁ TRATADO.

    scale_cols : lista ou None
        Subconjunto de feature_cols a expandir polinomialmente e escalar
        (sklearn StandardScaler, fit só no fold de treino).
    return_history : bool
        Se True (default), devolve também um dataframe "longo" (uma linha
        por época x fold) com o histórico de loss de treino/validação.

    Retorna
    -------
    resultados : lista de dicts, um por fold, com as métricas finais
    history_df : (só se return_history=True) DataFrame [fold, epoch, loss_train, loss_val]
    """
    validate_inputs(df, feature_cols, target_col, degree=degree)

    if log_target and (df[target_col] < 0).any():
        raise ValueError(f"log_target=True mas '{target_col}' tem valores negativos.")

    if n_splits < 2:
        raise ValueError(f"n_splits tem de ser >= 2, recebido: {n_splits}")
    if n_splits > len(df):
        raise ValueError(f"n_splits ({n_splits}) > nº de linhas ({len(df)}).")

    if scale_cols is None:
        scale_cols = list(feature_cols)
    passthrough_cols = [c for c in feature_cols if c not in scale_cols]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    resultados = []
    historico_linhas = []

    df = df.reset_index(drop=True)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df), start=1):
        if verbose:
            print(f"\n----- Fold {fold_idx}/{n_splits} (grau={degree}, l2={l2_lambda}, "
                  f"log_target={log_target}) -----")

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        y_train = df_train[target_col].to_numpy(dtype=float)
        y_val = df_val[target_col].to_numpy(dtype=float)

        y_train_model = np.log1p(y_train) if log_target else y_train
        y_val_model = np.log1p(y_val) if log_target else y_val

        X_train, X_val, _, poly, scaler = build_design_matrix(
            df_train, df_val, None, scale_cols, passthrough_cols, degree=degree
        )

        model = LinearRegressionGD(
            learning_rate=learning_rate, n_epochs=n_epochs, l2_lambda=l2_lambda,
            random_state=random_state, verbose=verbose, print_every=print_every,
            checkpoint_on=checkpoint_on
        )
        model.fit(
            X_train, y_train_model, X_val=X_val, y_val=y_val_model,
            y_train_original=y_train, y_val_original=y_val,
            inverse_transform=np.expm1 if log_target else None,
        )

        if return_history:
            for epoch_idx, loss_t in enumerate(model.loss_history_):
                loss_v = model.val_loss_history_[epoch_idx] if model.val_loss_history_ else None
                historico_linhas.append({
                    "fold": fold_idx, "degree": degree, "l2_lambda": l2_lambda,
                    "log_target": log_target, "epoch": epoch_idx,
                    "loss_train": loss_t, "loss_val": loss_v,
                })

        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        if log_target:
            y_pred_train = np.expm1(y_pred_train)
            y_pred_val = np.expm1(y_pred_val)

        m_train = compute_metrics(y_train, y_pred_train)
        m_val = compute_metrics(y_val, y_pred_val)

        resultados.append({
            "fold": fold_idx, "degree": degree, "l2_lambda": l2_lambda,
            "log_target": log_target, "checkpoint_on": checkpoint_on,
            "n_train": len(df_train), "n_val": len(df_val),
            "n_features_final": X_train.shape[1],
            "mse_train": m_train["mse"], "mae_train": m_train["mae"],
            "rmse_train": m_train["rmse"], "r2_train": m_train["r2"],
            "mse_val": m_val["mse"], "mae_val": m_val["mae"],
            "rmse_val": m_val["rmse"], "r2_val": m_val["r2"],
            "diverged": model.diverged_, "diverged_epoch": model.diverged_epoch_,
            "best_epoch": model.best_epoch_,
        })

    if return_history:
        history_df = pd.DataFrame(historico_linhas)
        return resultados, history_df

    return resultados


# ============================================================
# 7. GUARDAR / CARREGAR PESOS E HISTÓRICO DE TREINO
# ============================================================

def save_experiment(resultado: dict, filepath) -> Path:
    """
    Guarda em disco (via joblib) os pesos, o histórico de loss e os objetos
    de transformação (poly, scaler) de um resultado devolvido por
    run_experiment(), para poderes recarregar o modelo (prever, inspecionar
    a curva de convergência, etc.) sem ter de retreinar.

    Parâmetros
    ----------
    resultado : dict devolvido por run_experiment()
    filepath : caminho do ficheiro a criar (ex: "modelos/cap_grau2.joblib");
        as pastas em falta são criadas automaticamente

    Retorna
    -------
    filepath : Path do ficheiro guardado
    """
    model = resultado["model"]
    payload = {
        # pesos finais (após restaurar o melhor checkpoint no fim do treino)
        "weights": model.weights_,
        "bias": model.bias_,
        # melhor checkpoint visto durante o treino (igual a weights/bias, guardado à parte por clareza)
        "best_weights": model.best_weights_,
        "best_bias": model.best_bias_,
        "best_epoch": model.best_epoch_,
        # histórico de loss por época (treino e validação)
        "loss_history": model.loss_history_,
        "val_loss_history": model.val_loss_history_,
        "diverged": model.diverged_,
        "diverged_epoch": model.diverged_epoch_,
        # objetos de transformação, necessários para reaplicar o mesmo pré-processamento
        "poly": resultado["poly"],
        "scaler": resultado["scaler"],
        "scale_cols": resultado["scale_cols"],
        "passthrough_cols": resultado["passthrough_cols"],
        # métricas e configuração usadas neste treino
        "metrics": resultado["metrics"],
        "config": resultado["config"],
    }

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, filepath)
    return filepath


def load_experiment(filepath) -> dict:
    """Carrega um ficheiro guardado por save_experiment()."""
    return joblib.load(Path(filepath))


def evaluate_saved_model(payload: dict, df_novo: pd.DataFrame, target_col: str = "price") -> dict:
    """
    Avalia um modelo guardado (dict devolvido por load_experiment()) sobre
    um dataframe NOVO já tratado (mesmo formato produzido por
    prepare_features()), usando o dataframe INTEIRO como teste — não faz
    nenhum split, não faz fit de nada.

    Reindexa as colunas de df_novo para coincidirem exatamente com as
    scale_cols + passthrough_cols usadas no treino original (colunas do
    modelo que não existem em df_novo ficam a 0; colunas extra em df_novo
    são ignoradas). Isto é necessário porque um dataset diferente (ex: outra
    cidade) pode gerar colunas one-hot diferentes das do treino. Reaplica
    (transform, NÃO fit) o poly/scaler guardados no treino original.

    Retorna
    -------
    dict com mse, mae, rmse, r2 (sobre o dataframe completo).
    """
    scale_cols = payload["scale_cols"]
    passthrough_cols = payload["passthrough_cols"]
    poly = payload["poly"]
    scaler = payload["scaler"]
    weights = payload["weights"]
    bias = payload["bias"]

    colunas_modelo = scale_cols + passthrough_cols
    faltam = [c for c in colunas_modelo if c not in df_novo.columns]
    if faltam:
        print(f"[aviso] {len(faltam)} colunas do modelo não existem no novo dataset "
              f"(assumidas como 0): {faltam}")

    df_alinhado = df_novo.reindex(columns=colunas_modelo, fill_value=0)

    if scale_cols:
        X_scale = poly.transform(df_alinhado[scale_cols])
        X_scale = scaler.transform(X_scale)
    else:
        X_scale = np.empty((len(df_alinhado), 0))

    if passthrough_cols:
        X_pass = df_alinhado[passthrough_cols].to_numpy(dtype=float)
        X = np.hstack([X_scale, X_pass])
    else:
        X = X_scale

    y_true = df_novo[target_col].to_numpy(dtype=float)
    y_pred = X @ weights + bias

    return compute_metrics(y_true, y_pred)