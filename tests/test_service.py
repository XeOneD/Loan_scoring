import json

import pandas as pd


def test_scores_example_application(service):
    application = json.loads(open("examples/application.json", encoding="utf-8").read())
    result = service.score(pd.DataFrame([application]))[0]
    assert 0 < result["pd"] < 1
    assert 300 < result["score"] < 900
    assert result["decision"] in {"approve", "manual_review", "decline"}


def test_decision_thresholds(service):
    assert service.decide(0.01) == "approve"
    assert service.decide(0.07) == "manual_review"
    assert service.decide(0.5) == "decline"
