"""API smoke tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_inventory_endpoint_with_auth():
    with TestClient(app) as client:
        token_response = client.post("/auth/token", json={"username": "admin", "password": "admin123"})
        token = token_response.json()["access_token"]

        response = client.post(
            "/inventory/reorder",
            headers={"Authorization": f"Bearer {token}", "x-api-key": "neuralretail-local-key"},
            json={"lead_time_days": 7, "ordering_cost": 120.0, "holding_cost": 6.0},
        )
        assert response.status_code == 200
        assert len(response.json()["recommendations"]) > 0
