import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from scoring import api  # noqa: E402


def test_score_endpoint(service, monkeypatch):
    monkeypatch.setattr(api, "service", lambda: service)
    client = TestClient(api.app)
    application = json.loads(open("examples/application.json", encoding="utf-8").read())
    response = client.post("/score", json=application)
    assert response.status_code == 200
    assert set(response.json()) == {"pd", "score", "decision", "reasons"}
    assert client.post("/score", json={**application, "income": -1}).status_code == 422
