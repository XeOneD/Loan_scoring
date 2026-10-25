"""Загрузка заявок, целевая переменная и разбиение по времени.

Целевая: просрочка 90+ дней (определение дефолта по Базелю).

Даты выдачи в данных нет. Её заменяет CLIENTID: номера выдаются по порядку, и чем
больше номер, тем меньше доля погашенного долга (новые кредиты). Поэтому:
  - разбиение делается по CLIENTID: модель учится на старых клиентах, а проверяется
    на более новых - так же, как она будет работать после внедрения;
  - самые новые 10% клиентов исключены: их кредиты ещё не успели дойти до 90+ дней
    просрочки (доля дефолтов там в 3 раза ниже), и они бы испортили целевую.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/bank_credit_scoring.csv")
DEFAULT_DPD = 90

# Колонки, известные только после выдачи кредита. В признаки не попадают никогда.
POST_ISSUE_COLUMNS = ["Задолженность", "Просрочка, дни"]

RENAME = {
    "Первоначльный лимит": "limit",
    "BIRTHDATE": "birth_date",
    "SEX": "sex",
    "EDU": "education",
    "INCOME": "income",
    "TERM": "term",
    "Рейтинг кредитной истории": "credit_history_rating",
    "LV_AREA": "region",
    "LV_SETTLEMENTNAME": "settlement",
    "INDUSTRYNAME": "industry",
    "PDN": "debt_to_income",
    "CLIENTID": "client_id",
    "SCORINGMARK": "bureau_score",
    "UNDERAGECHILDRENCOUNT": "children",
    "VELCOMSCORING": "telecom_score",
    "FAMILYSTATUS": "family_status",
}


def load(path: Path = DATA_PATH, default_dpd: int = DEFAULT_DPD) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = raw.drop(columns=POST_ISSUE_COLUMNS).rename(columns=RENAME)
    df["bad"] = (raw["Просрочка, дни"] >= default_dpd).astype(np.int8)
    return df


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    excluded: pd.DataFrame


def time_split(df: pd.DataFrame, valid_from=0.6, test_from=0.7, immature_from=0.9) -> Split:
    """Границы - квантили CLIENTID по строкам. Все кредиты клиента попадают в одну часть."""
    edges = df["client_id"].quantile([valid_from, test_from, immature_from]).to_numpy()
    part = np.digitize(df["client_id"], edges)
    return Split(*(df[part == i].reset_index(drop=True) for i in range(4)))
