"""Признаки заявки. Работает и с таблицей из CSV, и с одной заявкой из API."""

import re

import numpy as np
import pandas as pd

REFERENCE_DATE = pd.Timestamp("2024-01-01")  # даты заявки нет; возраст — на эту дату

NUMERIC = [
    "age",
    "income",
    "limit",
    "term",
    "limit_to_income",
    "debt_to_income",
    "bureau_score",
    "telecom_score",
    "children",
]
# Пол в модель не входит намеренно: это защищённый признак, и решение о кредите
# не должно от него зависеть (к тому же его IV на этих данных ≈ 0).
CATEGORICAL = [
    "education",
    "family_status",
    "industry",
    "region",
    "credit_history_rating",
    "telecom_status",
    "lives_in_capital",
]
FEATURES = NUMERIC + CATEGORICAL

REGIONS = {
    "брест": "Брестская",
    "витеб": "Витебская",
    "гомел": "Гомельская",
    "гродн": "Гродненская",
    "могил": "Могилевская",
}


def normalize_region(value) -> str:
    """'МИНСКАЯ ОБЛАСТЬ', 'Минская', 'Минская область' -> одно значение; 'г. Минск' -> 'Минск'."""
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    s = value.lower().replace("ё", "е").strip()
    if s.startswith("г.") or s == "минск":
        return "Минск"
    if s.startswith("минск"):
        return "Минская"
    for prefix, name in REGIONS.items():
        if s.startswith(prefix):
            return name
    return "other"


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    birth = pd.to_datetime(df["birth_date"], errors="coerce")
    out["age"] = (REFERENCE_DATE - birth).dt.days / 365.25
    out["income"] = df["income"].where(df["income"] > 0)
    out["limit"] = df["limit"]
    out["term"] = df["term"]
    out["limit_to_income"] = out["limit"] / out["income"]
    out["debt_to_income"] = df["debt_to_income"]
    out["bureau_score"] = df["bureau_score"]
    out["children"] = df["children"]

    # -2 — абонент не найден у оператора, пусто — запрос не делали. Это разные ситуации.
    telecom = df["telecom_score"]
    out["telecom_score"] = telecom.where(telecom >= 0)
    out["telecom_status"] = np.select([telecom.isna(), telecom < 0], ["not_requested", "not_found"], "found")

    out["education"] = (
        df["education"].fillna("unknown").map(lambda s: re.sub(r"^\*+", "", s)).replace({"Послевузовское": "Высшее"})
    )
    out["family_status"] = df["family_status"].map({1: "1", 2: "2"}).fillna("other")
    out["industry"] = df["industry"].fillna("unknown")
    out["region"] = df["region"].map(normalize_region)
    rating = df["credit_history_rating"].astype("string")
    out["credit_history_rating"] = rating.where(rating.str.fullmatch(r"[A-E][1-3]"), "unknown").astype(str)
    out["lives_in_capital"] = (df["settlement"].fillna("").str.upper().str.strip() == "МИНСК").map(
        {True: "yes", False: "no"}
    )
    return out[FEATURES]
