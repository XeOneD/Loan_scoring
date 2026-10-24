"""Скоринг заявок сохранённой скоркартой. Используется API, не зависит от FastAPI."""

import json
from pathlib import Path

import joblib
import pandas as pd

from scoring import features


class ScoringService:
    def __init__(self, models_dir: Path = Path("models")):
        self.scorecard = joblib.load(models_dir / "scorecard.joblib")
        self.meta = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))

    def decide(self, pd_value: float) -> str:
        t = self.meta["thresholds"]
        if pd_value < t["approve_below"]:
            return "approve"
        if pd_value > t["decline_above"]:
            return "decline"
        return "manual_review"

    def score(self, applications: pd.DataFrame) -> list[dict]:
        X = features.build(applications.reset_index(drop=True))
        pds = self.scorecard.predict_proba(X)
        scores = self.scorecard.score(X)
        reasons = self.scorecard.explain(X)
        return [
            {"pd": round(float(p), 4), "score": int(round(s)), "decision": self.decide(p), "reasons": r}
            for p, s, r in zip(pds, scores, reasons, strict=True)
        ]
