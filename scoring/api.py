"""HTTP API для скоринга заявки.

uvicorn scoring.api:app --port 8000
curl -s localhost:8000/score -H 'Content-Type: application/json' -d @examples/application.json
"""

from datetime import date
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from scoring.service import ScoringService

app = FastAPI(title="Credit scoring", version="2.0.0")


class Application(BaseModel):
    birth_date: date
    income: float = Field(gt=0, description="Ежемесячный доход")
    limit: float = Field(gt=0, description="Запрошенный лимит")
    term: int = Field(gt=0, le=120, description="Срок, месяцев")
    debt_to_income: float = Field(ge=0, le=5, description="Показатель долговой нагрузки (ПДН)")
    education: Literal["Высшее", "Неоконченное высшее", "Среднее специальное", "Среднее", "Послевузовское"]
    family_status: int = Field(ge=1, le=4)
    children: int = Field(ge=0, le=10)
    industry: str
    region: str | None = None
    settlement: str | None = None
    credit_history_rating: str | None = Field(None, description="Рейтинг кредитной истории, A1…E3")
    bureau_score: float | None = Field(None, description="Балл кредитного бюро")
    telecom_score: float | None = Field(None, description="Скоринг оператора связи; -2 — абонент не найден")


class Reason(BaseModel):
    feature: str
    value: str
    points_lost: float


class ScoreResponse(BaseModel):
    pd: float = Field(description="Вероятность просрочки 90+ дней")
    score: int
    decision: Literal["approve", "manual_review", "decline"]
    reasons: list[Reason] = Field(description="Что сильнее всего снизило скор")


@lru_cache
def service() -> ScoringService:
    return ScoringService()


@app.post("/score", response_model=ScoreResponse)
def score(application: Application) -> dict:
    frame = pd.DataFrame([application.model_dump()])
    return service().score(frame)[0]


@app.get("/health")
def health() -> dict:
    meta = service().meta
    return {"status": "ok", "model": meta["champion"], "test_gini": round(meta["test_gini"], 3)}
