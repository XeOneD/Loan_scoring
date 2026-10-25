"""Скоркарта: WoE-биннинг + логистическая регрессия + перевод в баллы.

Стандарт розничного кредитования: каждое значение признака даёт фиксированное
число баллов, сумма баллов - скор, а по баллам видно, почему заявка получила
низкий скор (причины отказа).

Шкала: 600 баллов = шансы 50:1 (хороший:плохой), +20 баллов удваивают шансы.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scoring.woe import WoEBinner


class Scorecard:
    def __init__(
        self,
        iv_min: float = 0.02,
        corr_max: float = 0.7,
        C: float = 1.0,
        base_score: float = 600,
        base_odds: float = 50,
        pdo: float = 20,
        **binner_params,
    ):
        self.iv_min = iv_min
        self.corr_max = corr_max
        self.C = C
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
        self.binner = WoEBinner(**binner_params)

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical: list[str]) -> "Scorecard":
        self.binner.fit(X, y, categorical)
        iv = self.binner.iv()
        candidates = [c for c in iv.index if iv[c] >= self.iv_min]
        woe = self.binner.transform(X, candidates)

        # Из пары сильно коррелирующих признаков оставляем более сильный (по IV).
        selected: list[str] = []
        for col in candidates:
            if all(abs(np.corrcoef(woe[col], woe[s])[0, 1]) < self.corr_max for s in selected):
                selected.append(col)

        # WoE растёт с риском, поэтому все коэффициенты должны быть положительными.
        # Отрицательный коэффициент - признак мультиколлинеарности; такой признак убираем.
        while True:
            self.model = LogisticRegression(C=self.C, max_iter=1000).fit(woe[selected], y)
            coef = self.model.coef_[0]
            if (coef > 0).all() or len(selected) == 1:
                break
            selected.pop(int(np.argmin(coef)))
        self.features = selected
        self.dropped = {"low_iv": [c for c in iv.index if iv[c] < self.iv_min]}
        self.dropped["correlated_or_negative"] = [c for c in candidates if c not in selected]
        return self

    def recalibrate(self, X: pd.DataFrame, y: pd.Series) -> "Scorecard":
        """Platt-калибровка на отложенных данных: logit' = a·logit + b.
        Линейное преобразование логита, поэтому таблица баллов остаётся таблицей баллов."""
        logit = self.logit(X).reshape(-1, 1)
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
        a, b = platt.coef_[0, 0], platt.intercept_[0]
        self.model.coef_ = self.model.coef_ * a
        self.model.intercept_ = self.model.intercept_ * a + b
        return self

    def logit(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.decision_function(self.binner.transform(X, self.features))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Вероятность дефолта (PD)."""
        return 1 / (1 + np.exp(-self.logit(X)))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.offset - self.factor * self.logit(X)

    def points_table(self) -> pd.DataFrame:
        n = len(self.features)
        base = (self.offset - self.factor * self.model.intercept_[0]) / n
        rows = []
        for coef, col in zip(self.model.coef_[0], self.features, strict=True):
            table = self.binner.bins_[col].table.copy()
            table["points"] = base - self.factor * coef * table["woe"]
            rows.append(table)
        return pd.concat(rows, ignore_index=True)

    def explain(self, X: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        """Причины низкого скора: признаки, где заявка недобрала больше всего баллов
        до лучшего значения этого признака."""
        points = self.points_table()
        best = points.groupby("feature")["points"].max()
        n = len(self.features)
        base = (self.offset - self.factor * self.model.intercept_[0]) / n
        woe = self.binner.transform(X, self.features)
        result = []
        for i in range(len(X)):
            lost = []
            for coef, col in zip(self.model.coef_[0], self.features, strict=True):
                pts = base - self.factor * coef * woe[col].iloc[i]
                value = X[col].iloc[i]
                value = "нет данных" if pd.isna(value) else (f"{value:.4g}" if isinstance(value, float) else str(value))
                lost.append({"feature": col, "value": value, "points_lost": round(float(best[col] - pts), 1)})
            lost.sort(key=lambda r: -r["points_lost"])
            result.append([r for r in lost[:top] if r["points_lost"] > 0])
        return result
