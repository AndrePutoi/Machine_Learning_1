"""
logistic_regression.py
=======================
Regressão logística multinomial (softmax) com Gradient Descent implementada
do zero, full-batch — só com numpy (sem sklearn para o modelo nem para as
métricas). Mesmo estilo que linear_regression.py.

Problema: classificar o preço em 4 categorias, definidas pelos quartis do
preço no conjunto de TREINO (evita leakage dos quartis de val/teste):

    0 = barato        (price <= Q1)
    1 = medio-barato  (Q1 < price <= Q2)
    2 = medio-caro    (Q2 < price <= Q3)
    3 = caro          (price > Q3)

A expansão polinomial + StandardScaler reutiliza build_design_matrix() de
linear_regression.py (é só pré-processamento, não faz parte do modelo).

Autor: <o teu nome>
Disciplina: Machine Learning - Practical Project 1
"""

import numpy as np
import matplotlib.pyplot as plt

from linear_regression import build_design_matrix, validate_inputs


CLASS_NAMES = ["barato", "medio-barato", "medio-caro", "caro"]


# ============================================================
# 1. TARGET CATEGÓRICO
# ============================================================

def calcular_limites_quartis(price):
    """Q1, Q2 (mediana) e Q3 do preço — usados como fronteiras das classes."""
    return np.quantile(np.asarray(price, dtype=float), [0.25, 0.5, 0.75])


def categorizar_preco(price, limites):
    """
    Converte preço em classe 0..3 dados os limites [Q1, Q2, Q3].
    Intervalos fechados à direita: price == Q1 conta como 'barato'.
    """
    return np.searchsorted(np.asarray(limites), np.asarray(price, dtype=float), side="left")


# ============================================================
# 2. REGRESSÃO LOGÍSTICA MULTINOMIAL (SOFTMAX) COM GRADIENT DESCENT
# ============================================================

def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # estabilidade numérica
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def _one_hot(y, n_classes):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


class SoftmaxRegressionGD:
    """
    Regressão logística multinomial treinada por full-batch gradient
    descent, com loss de entropia cruzada e regularização L2 opcional.

    Parâmetros
    ----------
    learning_rate : float
        Taxa de aprendizagem (alpha).
    n_epochs : int
        Número de épocas — corre sempre este número completo.
    l2_lambda : float
        Força da regularização L2 (não penaliza o bias). 0.0 = sem L2.
    n_classes : int
        Número de classes.
    """

    def __init__(self, learning_rate=0.1, n_epochs=1000, l2_lambda=0.0,
                 n_classes=4, verbose=False, print_every=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2_lambda = l2_lambda
        self.n_classes = n_classes
        self.verbose = verbose
        self.print_every = print_every

        self.weights_ = None
        self.bias_ = None
        self.loss_history_ = []
        self.val_loss_history_ = []
        self.best_epoch_ = None

    def _compute_loss(self, X, Y_onehot, weights, bias):
        """Entropia cruzada média + termo L2."""
        n = len(X)
        probs = _softmax(X @ weights + bias)
        ce = -np.sum(Y_onehot * np.log(probs + 1e-12)) / n
        l2_term = self.l2_lambda * np.sum(weights ** 2) / n
        return ce + l2_term

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        Y = _one_hot(y, self.n_classes)
        Y_val = _one_hot(y_val, self.n_classes) if y_val is not None else None

        self.weights_ = np.zeros((n_features, self.n_classes))
        self.bias_ = np.zeros(self.n_classes)
        self.loss_history_ = []
        self.val_loss_history_ = []

        best_loss = np.inf
        best_weights, best_bias = self.weights_.copy(), self.bias_.copy()

        for epoch in range(self.n_epochs):
            probs = _softmax(X @ self.weights_ + self.bias_)
            error = probs - Y  # derivada da entropia cruzada em ordem aos logits

            grad_w = (X.T @ error) / n_samples + (2 * self.l2_lambda / n_samples) * self.weights_
            grad_b = error.sum(axis=0) / n_samples

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

            epoch_loss = self._compute_loss(X, Y, self.weights_, self.bias_)
            if not np.isfinite(epoch_loss):
                print(f"[DIVERGÊNCIA] Loss tornou-se NaN/Inf na época {epoch}. Reduz learning_rate.")
                break
            self.loss_history_.append(epoch_loss)

            if X_val is not None:
                self.val_loss_history_.append(
                    self._compute_loss(X_val, Y_val, self.weights_, self.bias_)
                )

            # checkpoint pela loss de TREINO (a validação só mede, não decide)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_weights, best_bias = self.weights_.copy(), self.bias_.copy()
                self.best_epoch_ = epoch

            if self.verbose and epoch % self.print_every == 0:
                msg = f"Época {epoch:4d} | CE treino: {epoch_loss:.4f}"
                if self.val_loss_history_:
                    msg += f" | CE val: {self.val_loss_history_[-1]:.4f}"
                print(msg)

        self.weights_, self.bias_ = best_weights, best_bias
        return self

    def predict_proba(self, X):
        return _softmax(X @ self.weights_ + self.bias_)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def plot_convergence(self, title="Curva de convergência", ax=None):
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.loss_history_, label="Loss treino", color="tab:blue")
        if self.val_loss_history_:
            ax.plot(self.val_loss_history_, label="Loss validação", color="tab:orange")
        ax.set_xlabel("Época")
        ax.set_ylabel("Entropia cruzada (+ L2)")
        ax.set_title(title)
        ax.legend()
        if own_fig:
            plt.tight_layout()
            plt.show()
        return ax


# ============================================================
# 3. MÉTRICAS DE CLASSIFICAÇÃO
# ============================================================

def confusion_matrix(y_true, y_pred, n_classes=4):
    """Linhas = classe real, colunas = classe prevista."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def classification_metrics(y_true, y_pred, n_classes=4):
    """Accuracy, precision/recall/F1 por classe e macro F1."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0)
    real_pos = cm.sum(axis=1)

    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, real_pos, out=np.zeros_like(tp), where=real_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    # "erro de uma classe": prever uma categoria vizinha (ex: barato vs medio-barato)
    distancia = np.abs(np.asarray(y_true) - np.asarray(y_pred))

    return {
        "accuracy": tp.sum() / cm.sum(),
        "balanced_accuracy": recall.mean(),
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_f1": f1.mean(),
        "accuracy_pm1": np.mean(distancia <= 1),
        "mae_classes": distancia.mean(),
        "kappa_quadratico": cohen_kappa(y_true, y_pred, n_classes, pesos="quadratico"),
        "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": cm,
    }


def cohen_kappa(y_true, y_pred, n_classes=4, pesos=None):
    """
    Kappa de Cohen: concordância acima do acaso (0 = acaso, 1 = perfeito).
    pesos="quadratico" -> penaliza erros pelo quadrado da distância entre
    classes; adequado aqui porque as classes de preço são ORDINAIS
    (confundir barato com caro é pior do que com medio-barato).
    """
    cm = confusion_matrix(y_true, y_pred, n_classes).astype(float)
    n = cm.sum()
    esperado = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    i, j = np.indices((n_classes, n_classes))
    if pesos == "quadratico":
        w = (i - j) ** 2 / (n_classes - 1) ** 2
    else:
        w = (i != j).astype(float)
    return 1 - (w * cm).sum() / (w * esperado).sum()


# ============================================================
# 3b. MÉTRICAS PROBABILÍSTICAS — ROC / AUC / PRECISION-RECALL
# ============================================================

def roc_curve(y_binario, scores):
    """
    Curva ROC de um problema binário (one-vs-rest). Ordena por score
    decrescente e acumula TP/FP; scores empatados contam como um único
    limiar. Devolve fpr, tpr, limiares (começa em (0, 0)).
    """
    y_binario = np.asarray(y_binario, dtype=float)
    scores = np.asarray(scores, dtype=float)
    ordem = np.argsort(-scores, kind="mergesort")
    scores, y_binario = scores[ordem], y_binario[ordem]

    # índice do último elemento de cada grupo de scores iguais
    fim_grupo = np.r_[np.where(np.diff(scores))[0], len(scores) - 1]
    tp = np.cumsum(y_binario)[fim_grupo]
    fp = (fim_grupo + 1) - tp

    tpr = np.r_[0.0, tp / tp[-1]]
    fpr = np.r_[0.0, fp / fp[-1]]
    return fpr, tpr, np.r_[np.inf, scores[fim_grupo]]


def precision_recall_curve(y_binario, scores):
    """Curva precision-recall (one-vs-rest). Começa em recall=0, precision=1."""
    y_binario = np.asarray(y_binario, dtype=float)
    scores = np.asarray(scores, dtype=float)
    ordem = np.argsort(-scores, kind="mergesort")
    scores, y_binario = scores[ordem], y_binario[ordem]

    fim_grupo = np.r_[np.where(np.diff(scores))[0], len(scores) - 1]
    tp = np.cumsum(y_binario)[fim_grupo]
    preditos_pos = fim_grupo + 1

    precision = np.r_[1.0, tp / preditos_pos]
    recall = np.r_[0.0, tp / tp[-1]]
    return precision, recall


def auc_trapezios(x, y):
    """Área sob a curva pela regra dos trapézios."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def average_precision(y_binario, scores):
    """AP = soma de (R_n - R_{n-1}) * P_n — resumo da curva precision-recall."""
    precision, recall = precision_recall_curve(y_binario, scores)
    return float(np.sum(np.diff(recall) * precision[1:]))


def log_loss(y_true, proba):
    """Entropia cruzada média das probabilidades previstas (sem L2)."""
    p = np.clip(proba[np.arange(len(y_true)), np.asarray(y_true)], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def probabilistic_metrics(y_true, proba):
    """
    AUC-ROC one-vs-rest por classe, macro (média simples) e micro (todas as
    decisões classe×linha juntas), average precision por classe e log loss.
    """
    y_true = np.asarray(y_true)
    n_classes = proba.shape[1]
    Y = _one_hot(y_true, n_classes)

    auc_por_classe = np.array([auc_trapezios(*roc_curve(Y[:, k], proba[:, k])[:2])
                               for k in range(n_classes)])
    ap_por_classe = np.array([average_precision(Y[:, k], proba[:, k])
                              for k in range(n_classes)])
    fpr_micro, tpr_micro, _ = roc_curve(Y.ravel(), proba.ravel())

    return {
        "auc_macro": auc_por_classe.mean(),
        "auc_micro": auc_trapezios(fpr_micro, tpr_micro),
        "ap_macro": ap_por_classe.mean(),
        "log_loss": log_loss(y_true, proba),
        "auc": auc_por_classe, "ap": ap_por_classe,
    }


# ============================================================
# 4. PIPELINE COMPLETO
# ============================================================

def run_classification_experiment(df, feature_cols, price_col="price", degree=1,
                                  scale_cols=None, test_size=0.15, val_size=0.15,
                                  learning_rate=0.1, n_epochs=1000, l2_lambda=0.0,
                                  random_state=42, verbose=False, plot=True):
    """
    Mesmo split que linear_regression.run_experiment (mesma semente -> as
    mesmas linhas em treino/val/teste), quartis calculados SÓ no treino,
    expansão polinomial + StandardScaler em scale_cols, treino softmax.
    """
    validate_inputs(df, feature_cols, price_col, degree=degree,
                    test_size=test_size, val_size=val_size)

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

    limites = calcular_limites_quartis(df_train[price_col])
    y_train = categorizar_preco(df_train[price_col], limites)
    y_val = categorizar_preco(df_val[price_col], limites)
    y_test = categorizar_preco(df_test[price_col], limites)

    X_train, X_val, X_test, poly, scaler = build_design_matrix(
        df_train, df_val, df_test, scale_cols, passthrough_cols, degree=degree
    )

    model = SoftmaxRegressionGD(
        learning_rate=learning_rate, n_epochs=n_epochs, l2_lambda=l2_lambda,
        n_classes=len(CLASS_NAMES), verbose=verbose,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    y_true = {"train": y_train, "val": y_val, "test": y_test}
    proba = {"train": model.predict_proba(X_train), "val": model.predict_proba(X_val),
             "test": model.predict_proba(X_test)}

    metrics = {}
    for split in ("train", "val", "test"):
        y_pred = np.argmax(proba[split], axis=1)
        metrics[split] = {**classification_metrics(y_true[split], y_pred),
                          **probabilistic_metrics(y_true[split], proba[split])}

    if plot:
        model.plot_convergence(title=f"Softmax — grau={degree}, λ={l2_lambda}")

    return {
        "model": model, "poly": poly, "scaler": scaler, "metrics": metrics,
        "y_true": y_true, "proba": proba,
        "limites": limites, "n_features_final": X_train.shape[1],
        "class_counts_train": np.bincount(y_train, minlength=len(CLASS_NAMES)),
        "config": {"degree": degree, "l2_lambda": l2_lambda,
                   "learning_rate": learning_rate, "n_epochs": n_epochs},
    }
