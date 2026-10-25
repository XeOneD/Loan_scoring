import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve


def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def ks(y, p) -> float:
    """Колмогоров-Смирнов: максимальное расхождение распределений скора у плохих и хороших."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def bootstrap_ci(y, p, metric=gini, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Доверительный интервал метрики: на тесте около 200 дефолтов, и разница
    в 2-3 пункта Gini между моделями может оказаться шумом."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            values.append(metric(y[idx], p[idx]))
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def summary(y, p, probabilities: bool = True) -> dict:
    """probabilities=False для сырых скоров (например, бюро): тогда только ранговые метрики."""
    lo, hi = bootstrap_ci(y, p)
    out = {
        "gini": gini(y, p),
        "gini_ci": [lo, hi],
        "ks": ks(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "bad_rate": float(np.mean(y)),
    }
    if probabilities:
        out["brier"] = float(brier_score_loss(y, p))
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
        out["mean_pd"] = float(np.mean(p))
    return out


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index. < 0.1 - стабильно, 0.1-0.25 - сдвиг, > 0.25 - сильный сдвиг."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, expected), minlength=len(edges) + 1) / len(expected)
    a = np.bincount(np.searchsorted(edges, actual), minlength=len(edges) + 1) / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def cutoff_table(y, p, approval_rates=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)) -> pd.DataFrame:
    """Что будет при разных долях одобрения: одобряем заявки с наименьшим PD."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p, kind="stable")
    rows = []
    for rate in approval_rates:
        n = int(round(rate * len(p)))
        approved = order[:n]
        rows.append(
            {
                "approval_rate": rate,
                "pd_cutoff": float(p[order[n - 1]]),
                "bad_rate_approved": float(y[approved].mean()),
                "bads_rejected": float(1 - y[approved].sum() / y.sum()),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(y, p, bins: int = 5) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False)
    out = frame.groupby("bin").agg(n=("y", "size"), mean_pd=("p", "mean"), bad_rate=("y", "mean"))
    return out.reset_index(drop=True)
