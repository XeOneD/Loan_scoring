import pandas as pd
import pytest

from scoring import features


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("г. Минск", "Минск"),
        ("МИНСК", "Минск"),
        ("МИНСКАЯ ОБЛАСТЬ", "Минская"),
        ("Минская", "Минская"),
        ("МОГИЛЁВСКАЯ", "Могилевская"),
        ("Гродненская область", "Гродненская"),
        (None, "unknown"),
    ],
)
def test_region_normalization(raw, expected):
    assert features.normalize_region(raw) == expected


def _application(**overrides):
    base = {
        "birth_date": "1990-01-01",
        "income": 1500.0,
        "limit": 3000.0,
        "term": 36,
        "debt_to_income": 0.3,
        "education": "**Послевузовское",
        "family_status": 3,
        "children": 0,
        "industry": "Торговля",
        "region": None,
        "settlement": "минск",
        "credit_history_rating": "-1",
        "bureau_score": None,
        "telecom_score": -2.0,
    }
    return pd.DataFrame([{**base, **overrides}])


def test_cleaning_rules():
    row = features.build(_application()).iloc[0]
    assert row["education"] == "Высшее"
    assert row["family_status"] == "other"
    assert row["credit_history_rating"] == "unknown"
    assert row["telecom_status"] == "not_found" and pd.isna(row["telecom_score"])
    assert row["lives_in_capital"] == "yes"
    assert row["limit_to_income"] == 2.0
    assert 33 < row["age"] < 35


def test_telecom_not_requested_differs_from_not_found():
    assert features.build(_application(telecom_score=None)).iloc[0]["telecom_status"] == "not_requested"


def test_sex_is_not_a_feature():
    assert "sex" not in features.FEATURES
