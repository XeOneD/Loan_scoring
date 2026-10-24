import numpy as np

from scoring import data, features


def test_post_issue_columns_never_reach_features(split):
    built = features.build(split.train)
    for col in ["Задолженность", "Просрочка, дни", "client_id", "bad"]:
        assert col not in built.columns


def test_split_is_ordered_by_client_id(split):
    parts = [split.train, split.valid, split.test, split.excluded]
    for older, newer in zip(parts, parts[1:], strict=False):
        assert older["client_id"].max() <= newer["client_id"].min()


def test_client_never_in_two_parts(split):
    sets = [set(p["client_id"]) for p in (split.train, split.valid, split.test, split.excluded)]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not sets[i] & sets[j]


def test_target_is_90_plus_dpd():
    df = data.load()
    assert set(np.unique(df["bad"])) == {0, 1}
    assert 0.03 < df["bad"].mean() < 0.06
