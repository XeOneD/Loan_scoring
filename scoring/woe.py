"""Биннинг признаков и Weight of Evidence.

WoE бина = ln(доля плохих в бине / доля хороших в бине). Чем больше, тем рискованнее.
IV признака = Σ (доля плохих − доля хороших) · WoE — сила признака в целом.

Правила биннинга (стандартные для скоркарт):
  - в каждом бине не меньше min_share наблюдений, иначе WoE — шум;
  - для числовых признаков WoE монотонен: «доход выше — риск ниже» без зигзагов,
    которые нельзя объяснить и которые не переживут новые данные;
  - не больше max_bins бинов;
  - пропуск — отдельный бин; незнакомая категория получает WoE бина other или 0.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING = "missing"
OTHER = "other"


@dataclass
class _Group:
    members: list
    n: int
    bad: int

    @property
    def rate(self) -> float:
        return self.bad / self.n


def _merge(groups: list[_Group], i: int) -> None:
    a, b = groups[i], groups.pop(i + 1)
    groups[i] = _Group(a.members + b.members, a.n + b.n, a.bad + b.bad)


def _merge_small(groups: list[_Group], min_n: int) -> None:
    while len(groups) > 1:
        i = min(range(len(groups)), key=lambda k: groups[k].n)
        if groups[i].n >= min_n:
            return
        if i == 0:
            _merge(groups, 0)
        elif i == len(groups) - 1:
            _merge(groups, i - 1)
        else:
            left = abs(groups[i - 1].rate - groups[i].rate)
            right = abs(groups[i + 1].rate - groups[i].rate)
            _merge(groups, i - 1 if left <= right else i)


def _merge_monotonic(groups: list[_Group]) -> None:
    rates = np.array([g.rate for g in groups])
    sizes = np.array([g.n for g in groups])
    trend = np.sign(np.cov(np.arange(len(groups)), rates, aweights=sizes)[0, 1]) if len(groups) > 1 else 1
    trend = trend or 1
    while True:
        violations = [i for i in range(len(groups) - 1) if (groups[i + 1].rate - groups[i].rate) * trend < 0]
        if not violations:
            return
        _merge(groups, violations[0])


def _merge_to_max(groups: list[_Group], max_bins: int) -> None:
    while len(groups) > max_bins:
        i = min(range(len(groups) - 1), key=lambda k: abs(groups[k + 1].rate - groups[k].rate))
        _merge(groups, i)


def _fmt(x: float) -> str:
    return f"{x:.4g}"


@dataclass
class FeatureBins:
    feature: str
    kind: str  # numeric | categorical
    labels: list[str]
    woe: dict[str, float]
    table: pd.DataFrame
    cuts: np.ndarray = field(default_factory=lambda: np.array([]))
    mapping: dict = field(default_factory=dict)

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())

    def assign(self, x: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            values = pd.to_numeric(x, errors="coerce")
            idx = np.searchsorted(self.cuts, values.to_numpy(), side="left")
            labels = np.array(self.labels_numeric)[np.minimum(idx, len(self.labels_numeric) - 1)]
            return pd.Series(np.where(values.isna(), MISSING, labels), index=x.index)
        fallback = OTHER if OTHER in self.mapping else None
        mapped = x.astype(str).map(self.mapping)
        if fallback is not None:
            mapped = mapped.fillna(self.mapping[OTHER])
        return mapped.fillna("unseen")

    @property
    def labels_numeric(self) -> list[str]:
        edges = [-np.inf, *self.cuts, np.inf]
        return [f"({_fmt(lo)}, {_fmt(hi)}]" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]

    def transform(self, x: pd.Series) -> pd.Series:
        return self.assign(x).map(self.woe).fillna(0.0).astype(float)


class WoEBinner:
    def __init__(self, max_bins: int = 6, min_share: float = 0.05, prebins: int = 20, smoothing: float = 0.5):
        self.max_bins = max_bins
        self.min_share = min_share
        self.prebins = prebins
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical: list[str]) -> "WoEBinner":
        y = np.asarray(y)
        self.total_bad = int(y.sum())
        self.total_good = int(len(y) - y.sum())
        self.min_n = int(np.ceil(self.min_share * len(y)))
        self.bins_ = {
            col: self._fit_categorical(col, X[col], y) if col in categorical else self._fit_numeric(col, X[col], y)
            for col in X.columns
        }
        return self

    def _woe_table(self, feature, labels, counts, bads) -> tuple[pd.DataFrame, dict]:
        counts, bads = np.asarray(counts, float), np.asarray(bads, float)
        goods = counts - bads
        s = self.smoothing
        bad_share = (bads + s) / (self.total_bad + s * len(labels))
        good_share = (goods + s) / (self.total_good + s * len(labels))
        woe = np.log(bad_share / good_share)
        table = pd.DataFrame(
            {
                "feature": feature,
                "bin": labels,
                "count": counts.astype(int),
                "share": counts / (self.total_bad + self.total_good),
                "bad_rate": np.divide(bads, counts, out=np.zeros_like(bads), where=counts > 0),
                "woe": woe,
                "iv": (bad_share - good_share) * woe,
            }
        )
        return table, dict(zip(labels, woe.tolist(), strict=True))

    def _fit_numeric(self, col: str, x: pd.Series, y: np.ndarray) -> FeatureBins:
        values = pd.to_numeric(x, errors="coerce").to_numpy(float)
        present = ~np.isnan(values)
        v, t = values[present], y[present]
        uniq = np.unique(v)
        if len(uniq) <= self.prebins:
            edges = (uniq[:-1] + uniq[1:]) / 2
        else:
            edges = np.unique(np.quantile(v, np.linspace(0, 1, self.prebins + 1)[1:-1]))
        pre = np.searchsorted(edges, v, side="left")
        groups = [
            _Group([k], int((pre == k).sum()), int(t[pre == k].sum()))
            for k in range(len(edges) + 1)
            if (pre == k).any()
        ]
        _merge_small(groups, self.min_n)
        _merge_monotonic(groups)
        _merge_to_max(groups, self.max_bins)
        cuts = np.array([edges[max(g.members)] for g in groups[:-1]])

        fb = FeatureBins(col, "numeric", [], {}, pd.DataFrame(), cuts=cuts)
        labels = fb.labels_numeric
        counts = [g.n for g in groups]
        bads = [g.bad for g in groups]
        if (~present).any():
            labels = [*labels, MISSING]
            counts.append(int((~present).sum()))
            bads.append(int(y[~present].sum()))
        fb.table, fb.woe = self._woe_table(col, labels, counts, bads)
        fb.labels = labels
        return fb

    def _fit_categorical(self, col: str, x: pd.Series, y: np.ndarray) -> FeatureBins:
        x = x.astype(str)
        stats = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].agg(["size", "sum"])
        rare = stats[stats["size"] < self.min_n]
        stats = stats[stats["size"] >= self.min_n]
        groups = [_Group([cat], int(r["size"]), int(r["sum"])) for cat, r in stats.iterrows()]
        if len(rare):
            groups.append(_Group([OTHER, *rare.index], int(rare["size"].sum()), int(rare["sum"].sum())))
        groups.sort(key=lambda g: g.rate)
        _merge_small(groups, self.min_n)
        _merge_to_max(groups, self.max_bins)

        labels, mapping = [], {}
        for g in groups:
            names = [m for m in g.members if m != OTHER]
            label = ", ".join(names[:4]) + (f" и ещё {len(names) - 4}" if len(names) > 4 else "")
            if OTHER in g.members:
                label = f"{label} + редкие" if names else "редкие"
            labels.append(label)
            mapping.update({m: label for m in g.members})
        fb = FeatureBins(col, "categorical", labels, {}, pd.DataFrame(), mapping=mapping)
        fb.table, fb.woe = self._woe_table(col, labels, [g.n for g in groups], [g.bad for g in groups])
        return fb

    def transform(self, X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        cols = columns or list(self.bins_)
        return pd.DataFrame({c: self.bins_[c].transform(X[c]) for c in cols}, index=X.index)

    def iv(self) -> pd.Series:
        return pd.Series({c: b.iv for c, b in self.bins_.items()}).sort_values(ascending=False)
