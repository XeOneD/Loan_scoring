import numpy as np
import pandas as pd

from scoring.woe import WoEBinner


def _toy(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(7, 0.5, n)
    city = rng.choice(["a", "b", "c", "rare1", "rare2"], n, p=[0.5, 0.3, 0.18, 0.01, 0.01])
    logit = -2 - 0.002 * (income - income.mean()) + np.where(city == "c", 1.0, 0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    income[:100] = np.nan
    return pd.DataFrame({"income": income, "city": city}), y


def test_numeric_woe_is_monotonic_and_bins_are_large_enough():
    X, y = _toy()
    binner = WoEBinner(min_share=0.05).fit(X, y, categorical=["city"])
    table = binner.bins_["income"].table
    numeric = table[table["bin"] != "missing"]
    assert (np.diff(numeric["woe"]) < 0).all()  # доход выше - риск ниже
    assert (numeric["share"] >= 0.05).all()
    assert "missing" in set(table["bin"])


def test_rare_categories_are_pooled_and_unseen_map_to_their_bin():
    X, y = _toy()
    bins = WoEBinner(min_share=0.05).fit(X, y, categorical=["city"]).bins_["city"]
    assert bins.mapping["rare1"] == bins.mapping["rare2"]
    unseen = bins.transform(pd.Series(["never_seen"]))
    assert unseen.iloc[0] == bins.woe[bins.mapping["other"]]


def test_iv_matches_definition():
    X, y = _toy()
    binner = WoEBinner(smoothing=0.0).fit(X, y, categorical=["city"])
    t = binner.bins_["city"].table
    bads = t["bad_rate"] * t["count"]
    goods = t["count"] - bads
    bad_share, good_share = bads / bads.sum(), goods / goods.sum()
    expected = ((bad_share - good_share) * np.log(bad_share / good_share)).sum()
    assert abs(binner.bins_["city"].iv - expected) < 1e-9
