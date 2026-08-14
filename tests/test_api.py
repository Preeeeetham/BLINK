"""
FastAPI REST endpoint tests using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.server import app

client = TestClient(app)


def test_root_ui_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "BLINK" in res.text


def test_health_endpoint():
    res = client.get("/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "operational"
    assert data["project_name"] == "BLINK"
    assert "device" in data


def test_channels_endpoint():
    res = client.get("/v1/channels")
    assert res.status_code == 200
    data = res.json()
    assert "IMG_VIS" in data["channels"]
    assert "IMG_TIR1" in data["channels"]


def test_simulate_scenario_endpoint():
    payload = {
        "scenario": "cyclone",
        "grid_size": 128,
        "cadence_steps": 3,
    }
    res = client.post("/v1/simulate/scenario", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["scenario"] == "cyclone"
    assert len(data["synthesized_frames"]) == 2
    assert "metrics" in data
    assert "psnr_db" in data["metrics"]
    assert "ssim" in data["metrics"]
    assert "fluid_divergence" in data["metrics"]
    assert data["metrics"]["psnr_db"] > 5.0
    assert data["metrics"]["ssim"] > 0.0
    assert len(data["t0_base64"]) > 50
