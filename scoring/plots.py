from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from scoring.metrics import calibration_table, cutoff_table, gini  # noqa: E402


def save_all(folder: Path, y, predictions: dict, scores) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y)

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, p in predictions.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name}, Gini {gini(y, p):.3f}")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
    ax.set(xlabel="Доля хороших, получивших отказ", ylabel="Доля отсечённых дефолтов", title="ROC на тесте")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(folder / "roc.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, p in predictions.items():
        t = calibration_table(y, p, bins=5)
        ax.plot(t["mean_pd"], t["bad_rate"], marker="o", label=name)
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], color="grey", lw=0.8, ls="--")
    ax.set(xlabel="Предсказанный PD", ylabel="Фактическая доля дефолтов", title="Калибровка на тесте")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "calibration.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(np.percentile(scores, 0.5), np.percentile(scores, 99.5), 40)
    ax.hist(scores[y == 0], bins=bins, density=True, alpha=0.6, label="без дефолта")
    ax.hist(scores[y == 1], bins=bins, density=True, alpha=0.6, label="дефолт")
    ax.set(xlabel="Скор, баллы", ylabel="Плотность", title="Распределение скора на тесте")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "score_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    rates = np.linspace(0.3, 1.0, 36)
    for name, p in predictions.items():
        t = cutoff_table(y, p, approval_rates=rates)
        ax.plot(100 * t["approval_rate"], 100 * t["bad_rate_approved"], label=name)
    ax.set(xlabel="Доля одобрения, %", ylabel="Дефолтов среди одобренных, %", title="Одобрение и риск")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "approval_vs_bad_rate.png", dpi=120)
    plt.close(fig)
