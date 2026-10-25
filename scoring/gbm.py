"""Градиентный бустинг - модель-претендент к скоркарте.

HistGradientBoostingClassifier из scikit-learn - тот же гистограммный алгоритм, что
в LightGBM, с нативной поддержкой категорий и пропусков. Монотонные ограничения
заданы там, где направление риска очевидно и подтверждается данными: модель не
сможет выучить «доход вырос - риск вырос» на шуме.

Число деревьев выбирается по log loss на валидации (отдельный срез по времени),
на нём же модель калибруется (Platt), чтобы её PD можно было сравнивать с порогами.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss

MONOTONIC = {"income": -1, "bureau_score": -1, "limit_to_income": 1}

PARAMS = {
    "learning_rate": 0.03,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 80,
    "l2_regularization": 1.0,
    "max_features": 0.8,
}


class GBMModel:
    def __init__(self, categorical: list[str], max_iter: int = 1500, seed: int = 42):
        self.categorical = categorical
        self.max_iter = max_iter
        self.seed = seed

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.categorical:
            # Незнакомая категория становится пропуском, а не ошибкой.
            out[col] = pd.Categorical(out[col].astype(str), categories=self.categories_[col])
        return out

    def _estimator(self, n_iter: int) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            **PARAMS,
            max_iter=n_iter,
            early_stopping=False,
            categorical_features="from_dtype",
            monotonic_cst={k: v for k, v in MONOTONIC.items() if k in self.columns_},
            random_state=self.seed,
        )

    def fit(self, X: pd.DataFrame, y, X_valid: pd.DataFrame, y_valid) -> "GBMModel":
        self.columns_ = list(X.columns)
        self.categories_ = {c: sorted(X[c].astype(str).unique()) for c in self.categorical}
        Xt, Xv = self._frame(X), self._frame(X_valid)

        probe = self._estimator(self.max_iter).fit(Xt, y)
        losses = [log_loss(y_valid, p[:, 1]) for p in probe.staged_predict_proba(Xv)]
        self.n_iter_ = int(np.argmin(losses)) + 1

        self.model_ = self._estimator(self.n_iter_).fit(Xt, y)
        self.calibrated_ = CalibratedClassifierCV(FrozenEstimator(self.model_), method="sigmoid").fit(Xv, y_valid)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrated_.predict_proba(self._frame(X[self.columns_]))[:, 1]
