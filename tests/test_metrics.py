import numpy as np

from scoring import metrics


def test_psi_is_zero_for_same_distribution():
    x = np.random.default_rng(0).normal(size=10_000)
    assert metrics.psi(x, x) < 1e-9
    assert metrics.psi(x, x + 1) > 0.25


def test_gini_and_ks_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.gini(y, p) == 1.0 and metrics.ks(y, p) == 1.0


def test_cutoff_table_full_approval_equals_base_rate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    t = metrics.cutoff_table(y, rng.random(1000))
    last = t.iloc[-1]
    assert last["approval_rate"] == 1.0 and abs(last["bad_rate_approved"] - y.mean()) < 1e-12
    assert last["bads_rejected"] == 0.0
