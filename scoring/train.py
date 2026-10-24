"""Обучение, оценка на отложенном срезе по времени и сохранение моделей.

    python -m scoring.train

Результат: models/ (скоркарта, бустинг, метаданные) и reports/ (отчёт, таблица баллов, графики).
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from scoring import data, features, metrics, plots
from scoring.gbm import GBMModel
from scoring.scorecard import Scorecard

log = logging.getLogger(__name__)
MODELS = Path("models")
REPORTS = Path("reports")
PSI_MAX = 0.25
# Пороги решения: доли заявок валидации, которые одобряются автоматически
# и уходят на ручную проверку. Остальные — отказ.
AUTO_APPROVE_SHARE = 0.7
REVIEW_SHARE = 0.2


def stable_features(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[list[str], dict]:
    """Признаки, распределение которых сильно сдвинулось уже между train и valid,
    не переживут внедрение. Решение принимается без теста."""
    psi = {c: metrics.psi(train[c].dropna(), valid[c].dropna()) for c in features.NUMERIC}
    return [c for c in features.FEATURES if psi.get(c, 0) <= PSI_MAX], psi


def run() -> dict:
    df = data.load()
    split = data.time_split(df)
    X = {k: features.build(getattr(split, k)) for k in ("train", "valid", "test")}
    y = {k: getattr(split, k)["bad"].to_numpy() for k in ("train", "valid", "test")}

    cols, psi_valid = stable_features(X["train"], X["valid"])
    categorical = [c for c in features.CATEGORICAL if c in cols]
    log.info("Исключены как нестабильные: %s", [c for c in features.FEATURES if c not in cols])

    scorecard = Scorecard().fit(X["train"][cols], y["train"], categorical)
    scorecard.recalibrate(X["valid"][cols], y["valid"])
    gbm = GBMModel(categorical).fit(X["train"][cols], y["train"], X["valid"][cols], y["valid"])

    test = X["test"][cols]
    p_sc, p_gbm = scorecard.predict_proba(test), gbm.predict_proba(test)
    bureau = -X["test"]["bureau_score"].fillna(X["train"]["bureau_score"].median()).to_numpy()
    results = {
        "Скоринг бюро (сырой балл)": metrics.summary(y["test"], bureau, probabilities=False),
        "Скоркарта (WoE + логрегрессия)": metrics.summary(y["test"], p_sc),
        "Градиентный бустинг": metrics.summary(y["test"], p_gbm),
    }
    overfit = {
        "Скоркарта": {k: metrics.gini(y[k], scorecard.predict_proba(X[k][cols])) for k in ("train", "valid")},
        "Бустинг": {k: metrics.gini(y[k], gbm.predict_proba(X[k][cols])) for k in ("train", "valid")},
    }

    p_valid = scorecard.predict_proba(X["valid"][cols])
    thresholds = {
        "approve_below": float(np.quantile(p_valid, AUTO_APPROVE_SHARE)),
        "decline_above": float(np.quantile(p_valid, AUTO_APPROVE_SHARE + REVIEW_SHARE)),
    }
    importance = permutation_importance(
        gbm,
        test,
        y["test"],
        scoring=lambda m, X_, y_: roc_auc_score(y_, m.predict_proba(X_)),
        n_repeats=5,
        random_state=0,
    )

    MODELS.mkdir(exist_ok=True)
    joblib.dump(scorecard, MODELS / "scorecard.joblib")
    joblib.dump(gbm, MODELS / "gbm.joblib")
    metadata = {
        "champion": "scorecard",
        "features": cols,
        "scorecard_features": scorecard.features,
        "thresholds": thresholds,
        "target": f"просрочка >= {data.DEFAULT_DPD} дней",
        "test_gini": results["Скоркарта (WoE + логрегрессия)"]["gini"],
    }
    (MODELS / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "split": {
            k: {"n": len(getattr(split, k)), "bad_rate": float(getattr(split, k)["bad"].mean())}
            for k in ("train", "valid", "test", "excluded")
        },
        "psi_train_valid": psi_valid,
        "psi_train_test": {c: metrics.psi(X["train"][c].dropna(), X["test"][c].dropna()) for c in features.NUMERIC},
        "score_psi_train_test": metrics.psi(scorecard.score(X["train"][cols]), scorecard.score(test)),
        "results": results,
        "overfit": overfit,
        "gbm_iterations": gbm.n_iter_,
        "thresholds": thresholds,
        "iv": scorecard.binner.iv().to_dict(),
        "scorecard_features": scorecard.features,
        "dropped": {**scorecard.dropped, "unstable": [c for c in features.FEATURES if c not in cols]},
        "permutation_importance": dict(
            sorted(zip(cols, importance.importances_mean.tolist(), strict=True), key=lambda x: -x[1])
        ),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "report.md").write_text(
        render(report, metrics.cutoff_table(y["test"], p_sc), metrics.calibration_table(y["test"], p_sc)),
        encoding="utf-8",
    )
    (REPORTS / "scorecard.md").write_text(render_points(scorecard), encoding="utf-8")
    plots.save_all(REPORTS / "figures", y["test"], {"Скоркарта": p_sc, "Бустинг": p_gbm}, scorecard.score(test))
    log.info(
        "Gini на тесте: скоркарта %.3f, бустинг %.3f", metrics.gini(y["test"], p_sc), metrics.gini(y["test"], p_gbm)
    )
    return report


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render(r: dict, cutoffs: pd.DataFrame, calibration: pd.DataFrame) -> str:
    lines = ["# Отчёт по модели", "", "## Данные", "", "| Часть | Заявок | Доля дефолтов |", "|---|---|---|"]
    names = {"train": "обучение", "valid": "валидация", "test": "тест", "excluded": "исключено (незрелые)"}
    lines += [f"| {names[k]} | {v['n']} | {_pct(v['bad_rate'])} |" for k, v in r["split"].items()]

    lines += [
        "",
        "## Качество на тесте",
        "",
        "| Модель | Gini [95% ДИ] | KS | PR-AUC | Brier |",
        "|---|---|---|---|---|",
    ]
    for name, m in r["results"].items():
        lo, hi = m["gini_ci"]
        brier = f"{m['brier']:.4f}" if "brier" in m else "—"
        lines.append(f"| {name} | {m['gini']:.3f} [{lo:.3f}–{hi:.3f}] | {m['ks']:.3f} | {m['pr_auc']:.3f} | {brier} |")

    lines += ["", "Gini на обучении и валидации (разрыв с тестом показывает переобучение):", ""]
    lines += ["| Модель | Обучение | Валидация |", "|---|---|---|"]
    lines += [f"| {n} | {v['train']:.3f} | {v['valid']:.3f} |" for n, v in r["overfit"].items()]

    lines += ["", "## Стратегия отсечения (скоркарта, тест)", ""]
    lines += ["| Доля одобрения | Порог PD | Дефолтов среди одобренных | Отсечено дефолтов |", "|---|---|---|---|"]
    lines += [
        f"| {_pct(row.approval_rate)} | {row.pd_cutoff:.3f} | {_pct(row.bad_rate_approved)} "
        f"| {_pct(row.bads_rejected)} |"
        for row in cutoffs.itertuples()
    ]

    lines += ["", "## Калибровка (скоркарта, тест, квинтили PD)", "", "| Средний PD | Факт |", "|---|---|"]
    lines += [f"| {_pct(row.mean_pd)} | {_pct(row.bad_rate)} |" for row in calibration.itertuples()]

    lines += ["", "## Стабильность (PSI)", "", f"PSI скора обучение → тест: {r['score_psi_train_test']:.3f}", ""]
    lines += ["| Признак | обучение → валидация | обучение → тест |", "|---|---|---|"]
    lines += [f"| {c} | {v:.3f} | {r['psi_train_test'][c]:.3f} |" for c, v in r["psi_train_valid"].items()]

    lines += ["", "## Information Value", "", "| Признак | IV | В скоркарте |", "|---|---|---|"]
    lines += [f"| {c} | {v:.3f} | {'да' if c in r['scorecard_features'] else ''} |" for c, v in r["iv"].items()]

    lines += ["", "## Важность признаков бустинга (permutation, падение AUC на тесте)", ""]
    lines += ["| Признак | ΔAUC |", "|---|---|"]
    lines += [f"| {c} | {v:.4f} |" for c, v in list(r["permutation_importance"].items())[:10]]
    lines += ["", "Графики: `reports/figures/`. Таблица баллов: `reports/scorecard.md`."]
    return "\n".join(lines) + "\n"


def render_points(sc: Scorecard) -> str:
    table = sc.points_table()
    lines = ["# Таблица баллов", "", "Скор = сумма баллов. 600 баллов = шансы 50:1, +20 баллов удваивают шансы.", ""]
    for feature, group in table.groupby("feature", sort=False):
        lines += [f"## {feature}", "", "| Значение | Заявок | Доля дефолтов | WoE | Баллы |", "|---|---|---|---|---|"]
        lines += [
            f"| {r.bin} | {r.count} | {_pct(r.bad_rate)} | {r.woe:+.3f} | {r.points:.0f} |" for r in group.itertuples()
        ]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run()


if __name__ == "__main__":
    main()
