"""
Tests for scientifically grounded nowcasting:
- NETRA Convective Activity Index (Shukla et al., 2017 & IMD physical indicators)
- Cloud-top cooling tendency and experimental overshooting-top candidates
- Dvorak ADT & Courtney-Knaff (2009) wind-pressure physics
"""

import numpy as np
import pytest
import torch

from src.pipeline.nowcasting import ConvectiveNowcaster, StormTrackPredictor


def test_dvorak_intensity_estimation():
    # Synthetic convective frame (TIR-1 inverted: cold cloud top = 1.0)
    t1 = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
    # Put an intense storm core at center
    t1[:, 2, 50:78, 50:78] = 0.95  # cold core (~198 K)
    t1[:, 2, 60:68, 60:68] = 0.50  # warm eye (~245 K)

    is_active, ci, v_kmh, p_hpa, cat = StormTrackPredictor.estimate_dvorak_intensity(
        tensor_t1=t1,
        flow_mag_max=22.0,
        cx_px=64.0,
        cy_px=64.0,
    )

    assert isinstance(is_active, bool)
    assert 1.0 <= ci <= 8.0
    assert 12.0 <= v_kmh <= 300.0
    assert 880.0 <= p_hpa <= 1010.0
    assert isinstance(cat, str)


def test_netra_convective_activity_nowcasting():
    # T0 and T1 frames
    t0 = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
    t1 = torch.zeros((1, 3, 128, 128), dtype=torch.float32)

    # Convective explosion between T0 and T1
    t0[:, 2, 40:88, 40:88] = 0.30  # ~266 K at T0
    t1[:, 2, 40:88, 40:88] = 0.92  # ~201 K at T1 (severe cooling dT/dt ~ -65 K / 15min)

    flow_01 = torch.zeros((1, 2, 128, 128), dtype=torch.float32)
    flow_01[:, 0] = 3.5

    report = ConvectiveNowcaster.evaluate_convective_risk(t0, t1, flow_01)

    # Scientific honesty checks:
    # 1. Uncalibrated probability MUST be suppressed
    assert report.is_calibrated_probability is False
    assert report.extreme_rain_probability_pct is None
    assert report.precipitation_ground_truth_status == "UNVALIDATED_NO_SURFACE_RADAR"

    # 2. Convective Activity Index is a high indicator for explosive cooling
    assert report.convective_activity_index >= 60.0
    assert report.overall_convective_level == "SEVERE_CONVECTIVE_UPDRAFT"
    assert report.cloud_top_bt_tendency_k_15m < -5.0

    # 3. Active clusters
    assert len(report.active_threat_clusters) > 0
    top_cluster = report.active_threat_clusters[0]
    assert top_cluster.min_brightness_temp_k < 220.0
    assert top_cluster.convective_activity_index >= 60.0
    assert top_cluster.precipitation_status == "UNVALIDATED_NO_SURFACE_RADAR"
    assert top_cluster.estimated_rainfall_mm_hr is None


def test_coastal_landfall_raycasting():
    # Storm with cold eyewall ring and warm eye moving northwest in Bay of Bengal
    t0 = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
    t1 = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
    t0[:, 2, 45:85, 45:85] = 0.95
    t0[:, 2, 60:68, 60:68] = 0.50
    t1[:, 2, 40:80, 40:80] = 0.95
    t1[:, 2, 56:64, 56:64] = 0.50

    flow_01 = torch.zeros((1, 2, 128, 128), dtype=torch.float32)
    flow_01[:, 0] = -22.0  # Westward drift
    flow_01[:, 1] = -18.0  # Northward drift

    geo_bounds = {"latNorth": 22.0, "latSouth": 8.0, "lonWest": 80.0, "lonEast": 95.0}

    report = StormTrackPredictor.predict_track_and_cone(t0, t1, flow_01, geo_bounds=geo_bounds)
    assert report.is_active_cyclone is True
    assert report.landfall_estimate is not None
    assert "estimated_region" in report.landfall_estimate
    assert "estimated_eta_hours" in report.landfall_estimate
    assert report.landfall_estimate["estimated_eta_hours"] > 0.0
    assert "distance_to_coast_km" in report.landfall_estimate
    assert report.landfall_estimate["distance_to_coast_km"] >= 0.0
