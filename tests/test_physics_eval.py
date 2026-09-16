"""
Unit tests for physics evaluation, ground-truth verification, and baseline comparisons.
"""

import numpy as np
import pytest
import torch

from src.pipeline.physics_eval import PhysicsEvaluator


def test_psnr_and_ssim_identical():
    img = torch.rand(1, 3, 64, 64)
    psnr = PhysicsEvaluator.compute_psnr(img, img, data_range=1.0)
    ssim = PhysicsEvaluator.compute_ssim(img, img, data_range=1.0)

    assert psnr >= 90.0  # Identical images yield near infinity / capped at 100
    assert abs(ssim - 1.0) < 1e-4


def test_psnr_data_range():
    img0 = torch.zeros(1, 1, 32, 32)
    img1 = torch.ones(1, 1, 32, 32) * 10.0
    # MSE is 100.0, data_range is 140.0
    # PSNR = 10 * log10(140^2 / 100) = 10 * log10(196) ~ 22.92 dB
    psnr = PhysicsEvaluator.compute_psnr(img0, img1, data_range=140.0)
    assert 22.0 <= psnr <= 24.0

    with pytest.raises(ValueError):
        PhysicsEvaluator.compute_psnr(img0, img1, data_range=0.0)


def test_fluid_divergence():
    # Zero flow -> Zero divergence
    zero_flow = torch.zeros(1, 2, 64, 64)
    div_zero = PhysicsEvaluator.compute_fluid_divergence(zero_flow)
    assert div_zero == 0.0

    # Divergent flow field: u_x = x, u_y = y -> du_x/dx = 1, du_y/dy = 1 -> div = 2
    y, x = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
    div_flow = torch.stack([x, y], dim=0).unsqueeze(0)
    div_val = PhysicsEvaluator.compute_fluid_divergence(div_flow)
    assert div_val > 0.0


def test_radiance_conservation():
    f0 = torch.ones(1, 3, 32, 32) * 0.4
    f1 = torch.ones(1, 3, 32, 32) * 0.8
    # Exact linear mass at t=0.5 is 0.6
    pred_perfect = torch.ones(1, 3, 32, 32) * 0.6
    cons = PhysicsEvaluator.compute_radiance_conservation(pred_perfect, f0, f1, t_normalized=0.5)
    assert abs(cons - 100.0) < 1e-3


def test_physical_rmse_kelvin():
    t0_k = torch.ones(1, 1, 32, 32) * 250.0
    t1_k = torch.ones(1, 1, 32, 32) * 253.5
    rmse = PhysicsEvaluator.compute_rmse_k(t0_k, t1_k)
    assert abs(rmse - 3.5) < 1e-4


def test_bt_compliance_and_drift():
    # Compliant frame in [180, 330] K
    bt_good = torch.ones(1, 1, 32, 32) * 240.0
    comp = PhysicsEvaluator.compute_physical_bt_compliance(bt_good)
    assert comp == 100.0

    # Non-compliant pixel (unphysical numerical artifact)
    bt_artifact = bt_good.clone()
    bt_artifact[0, 0, 0, 0] = 120.0  # < 180 K
    comp_bad = PhysicsEvaluator.compute_physical_bt_compliance(bt_artifact)
    assert comp_bad < 100.0

    # Drift test
    t0_k = torch.ones(1, 1, 32, 32) * 240.0
    t1_k = torch.ones(1, 1, 32, 32) * 260.0
    # Expected mid is 250.0. If pred is 252.0, drift is +2.0 K
    pred_k = torch.ones(1, 1, 32, 32) * 252.0
    drift = PhysicsEvaluator.compute_mean_bt_drift(pred_k, t0_k, t1_k, t_normalized=0.5)
    assert abs(drift - 2.0) < 1e-4


def test_ground_truth_safety_assertions():
    f0 = torch.ones(1, 3, 32, 32) * 0.2
    f1 = torch.ones(1, 3, 32, 32) * 0.8
    synth = torch.ones(1, 3, 32, 32) * 0.5
    lin_mid = 0.5 * (f0 + f1)

    # 1. Target cannot be None
    with pytest.raises(ValueError, match="cannot be None"):
        PhysicsEvaluator.assert_independent_ground_truth(synth, None)

    # 2. Target cannot be prediction itself
    with pytest.raises(ValueError, match="identical in memory to prediction"):
        PhysicsEvaluator.assert_independent_ground_truth(synth, synth)

    # 3. Target cannot be frame_0 (persistence leakage)
    with pytest.raises(ValueError, match="persistence leakage"):
        PhysicsEvaluator.assert_independent_ground_truth(synth, f0, frame_0=f0, frame_1=f1)

    # 4. Target cannot be frame_1 (endpoint leakage)
    with pytest.raises(ValueError, match="endpoint leakage"):
        PhysicsEvaluator.assert_independent_ground_truth(synth, f1, frame_0=f0, frame_1=f1)

    # 5. Target cannot be linear blend of endpoints
    with pytest.raises(ValueError, match="synthetic linear blend"):
        PhysicsEvaluator.assert_independent_ground_truth(synth, lin_mid, frame_0=f0, frame_1=f1)


def test_evaluate_synthesis_without_ground_truth():
    f0 = torch.rand(1, 3, 32, 32)
    f1 = torch.rand(1, 3, 32, 32)
    synth = 0.5 * (f0 + f1)

    report = PhysicsEvaluator.evaluate_synthesis(
        synthesized=synth,
        ground_truth=None,
        frame_0=f0,
        frame_1=f1,
        t_normalized=0.5,
    )

    assert report.has_ground_truth is False
    assert report.psnr_db is None
    assert report.ssim is None
    assert report.rmse_k is None
    assert "suppressed" in report.validation_status.lower()
    assert report.physical_bt_compliance_pct >= 0.0
    assert report.radiance_conservation_pct >= 0.0


def test_evaluate_with_baselines():
    f0 = torch.ones(1, 3, 32, 32) * 0.2
    f1 = torch.ones(1, 3, 32, 32) * 0.8
    synth = torch.ones(1, 3, 32, 32) * 0.52
    # Independent held-out observation (neither f0, f1, nor linear blend)
    gt = torch.ones(1, 3, 32, 32) * 0.55

    res = PhysicsEvaluator.evaluate_with_baselines(synth, gt, f0, f1, t_normalized=0.5)

    assert "blink" in res
    assert "persistence" in res
    assert "linear_blend" in res
    assert res["blink"]["psnr_db"] > 0
    assert res["persistence"]["psnr_db"] > 0
    assert res["linear_blend"]["psnr_db"] > 0
