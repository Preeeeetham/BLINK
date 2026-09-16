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
    assert data["metrics"]["has_ground_truth"] is True
    assert "psnr_db" in data["metrics"]
    assert "ssim" in data["metrics"]
    assert "fluid_divergence" in data["metrics"]
    assert data["metrics"]["psnr_db"] > 5.0
    assert data["metrics"]["ssim"] > 0.0
    assert len(data["t0_base64"]) > 50
    assert data["operating_mode"] == "TEMPORAL_INTERPOLATION"
    assert data["data_source_mode"] == "SYNTHETIC_DEMONSTRATION"


def test_fetch_query_endpoint():
    payload = {
        "source": "MOSDAC_INSAT3DS",
        "date": "2026-08-28",
        "region": "indian_subcontinent",
    }
    res = client.post("/v1/fetch/query", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "available_scans" in data
    assert len(data["available_scans"]) > 0


def test_fetch_realtime_endpoint():
    payload = {
        "source": "REAL_SATELLITE",
        "date": "2026-08-28",
        "time": "10:00",
        "region": "bay_of_bengal",
        "cadence_steps": 3,
        "grid_size": 64,
    }
    res = client.post("/v1/fetch/realtime", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "synthesized_frames" in data
    assert len(data["synthesized_frames"]) == 2
    assert "storm_track" in data
    assert "convective_nowcast" in data

    # Scientific validation checks on real satellite endpoint:
    # 1. Operational held-out intermediate observation triplet verification:
    assert "metrics" in data
    assert data["metrics"]["has_ground_truth"] is True
    assert data["metrics"]["psnr_db"] is not None and data["metrics"]["psnr_db"] > 0.0
    assert data["metrics"]["ssim"] is not None and data["metrics"]["ssim"] > 0.0
    assert data["metrics"]["rmse_k"] is not None and data["metrics"]["rmse_k"] > 0.0
    assert "verified" in data["metrics"]["validation_status"].lower() or "verification" in data["metrics"]["validation_status"].lower()

    # 2. Internal physical diagnostics must be valid
    assert data["metrics"]["radiance_conservation_pct"] >= 0.0
    assert data["metrics"]["physical_bt_compliance_pct"] >= 0.0
    assert "mean_bt_drift_k" in data["metrics"]
    assert "fluid_divergence" in data["metrics"]

    # 3. Operating mode and NETRA honesty
    assert data["operating_mode"] == "TEMPORAL_INTERPOLATION"
    assert data["convective_nowcast"]["is_calibrated_probability"] is False
    assert data["convective_nowcast"]["extreme_rain_probability_pct"] is None
    assert "cloud_top_bt_tendency_k_15m" in data["convective_nowcast"]


def test_config_mosdac_endpoint():
    payload = {
        "username": "researcher@example.org",
        "api_token": "sec_token_999",
    }
    res = client.post("/v1/config/mosdac", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["is_configured"] is True
