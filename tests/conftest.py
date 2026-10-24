import pytest

from scoring import data, features
from scoring.scorecard import Scorecard


@pytest.fixture(scope="session")
def split():
    return data.time_split(data.load())


@pytest.fixture(scope="session")
def X_train(split):
    return features.build(split.train)


@pytest.fixture(scope="session")
def scorecard(split, X_train):
    return Scorecard().fit(X_train, split.train["bad"], features.CATEGORICAL)


@pytest.fixture
def service(scorecard, tmp_path):
    import json

    import joblib

    from scoring.service import ScoringService

    joblib.dump(scorecard, tmp_path / "scorecard.joblib")
    meta = {"champion": "scorecard", "test_gini": 0.5, "thresholds": {"approve_below": 0.05, "decline_above": 0.1}}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    return ScoringService(tmp_path)
