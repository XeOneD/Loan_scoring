import numpy as np

from scoring import features, metrics


def test_all_coefficients_positive(scorecard):
    assert (scorecard.model.coef_[0] > 0).all()


def test_points_add_up_to_score(scorecard, split):
    X = features.build(split.test.head(50))
    table = scorecard.points_table()
    total = np.zeros(len(X))
    for col in scorecard.features:
        labels = scorecard.binner.bins_[col].assign(X[col])
        points = table[table["feature"] == col].set_index("bin")["points"]
        total += labels.map(points).fillna(0).to_numpy()
    np.testing.assert_allclose(total, scorecard.score(X), atol=1e-6)


def test_score_scale(scorecard, split):
    X = features.build(split.test.head(50))
    pd_ = scorecard.predict_proba(X)
    odds_good = (1 - pd_) / pd_
    expected = 600 + 20 * np.log2(odds_good / 50)
    np.testing.assert_allclose(scorecard.score(X), expected, atol=1e-6)


def test_recalibration_keeps_ranking(split, X_train):
    from scoring.scorecard import Scorecard

    sc = Scorecard().fit(X_train, split.train["bad"], features.CATEGORICAL)
    X = features.build(split.test)
    before = sc.predict_proba(X)
    sc.recalibrate(features.build(split.valid), split.valid["bad"])
    after = sc.predict_proba(X)
    assert abs(metrics.gini(split.test["bad"], before) - metrics.gini(split.test["bad"], after)) < 1e-9


def test_explain_returns_top_reasons(scorecard, split):
    reasons = scorecard.explain(features.build(split.test.head(5)))
    assert len(reasons) == 5
    for r in reasons:
        assert len(r) <= 3
        assert all(x["points_lost"] > 0 for x in r)
        assert [x["points_lost"] for x in r] == sorted((x["points_lost"] for x in r), reverse=True)


def test_scorecard_beats_bureau_score_on_test(scorecard, split):
    X = features.build(split.test)
    y = split.test["bad"]
    bureau = -X["bureau_score"].fillna(X["bureau_score"].median())
    assert metrics.gini(y, scorecard.predict_proba(X)) > metrics.gini(y, bureau) + 0.1
