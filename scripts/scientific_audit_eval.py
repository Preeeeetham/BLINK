"""
Scientific Audit & Ground-Truth Verification Report Generator for Project BLINK.
Generates machine-readable docs/EVALUATION_REPORT.json.
"""

import json
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from src.ingestion.mosdac_parser import MOSDACParser, SyntheticMOSDACSimulator
from src.pipeline.interpolator import AeroInterpolator
from src.pipeline.nowcasting import ConvectiveNowcaster, StormTrackPredictor
from src.pipeline.physics_eval import PhysicsEvaluator


def run_scientific_audit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = MOSDACParser()
    interpolator = AeroInterpolator(device=device, channels=["IMG_VIS", "IMG_WV", "IMG_TIR1"])

    report = {
        "project": "BLINK",
        "audit_timestamp_iso": "2026-09-16T17:40:00Z",
        "scientific_standards": ["IMD Operational Standards", "ISRO SAC NETRA (2017)", "WMO Verification Guidelines"],
        "evaluations": [],
    }

    # --------------------------------------------------------------------------
    # Evaluation 1: Real INSAT-3DS Archive Observations
    # --------------------------------------------------------------------------
    h5_t0 = root_dir / "MOSDAC" / "3SIMG_30AUG2026_2300_L1B_STD_V01R00.h5"
    h5_t1 = root_dir / "MOSDAC" / "3SIMG_30AUG2026_2330_L1B_STD_V01R00.h5"

    eval_real = {
        "dataset_name": "MOSDAC INSAT-3DS Imager L1B Standard",
        "data_tier": "OBSERVED",
        "files_ingested": [h5_t0.name if h5_t0.exists() else "NONE", h5_t1.name if h5_t1.exists() else "NONE"],
        "scan_cadence_minutes": 30.0,
        "operating_mode": "TEMPORAL_INTERPOLATION",
        "independent_ground_truth_available": False,
        "ground_truth_absence_reason": "Consecutive observations are separated by 30 min (23:00 to 23:30 UTC). No intermediate 23:15 UTC observation exists in local archive. Quantitative validation metrics (PSNR, SSIM, RMSE) are strictly suppressed to prevent circular evaluation.",
        "validation_metrics": {
            "psnr_db": None,
            "ssim": None,
            "rmse_k": None,
        },
    }

    if h5_t0.exists() and h5_t1.exists():
        d0, bounds0 = parser.read_hdf5_sector(str(h5_t0), target_size=(256, 256))
        d1, bounds1 = parser.read_hdf5_sector(str(h5_t1), target_size=(256, 256))
        t0 = parser.to_normalized_tensor(d0, device=device)
        t1 = parser.to_normalized_tensor(d1, device=device)

        with torch.no_grad():
            res = interpolator.interpolate(t0, t1, sub_timesteps=[0.5])

        mid_synth = res.synthesized_frames[0]
        phys_eval = PhysicsEvaluator.evaluate_synthesis(
            synthesized=mid_synth,
            ground_truth=None,
            frame_0=t0,
            frame_1=t1,
            t_normalized=0.5,
            flow=res.flow_01,
            latency_ms=res.mean_latency_ms,
        )

        nowcast = ConvectiveNowcaster.evaluate_convective_risk(t0, t1, res.flow_01, geo_bounds=bounds1)

        eval_real["internal_physical_diagnostics"] = {
            "flux_conservation_pct": round(phys_eval.radiance_conservation_pct, 2),
            "physical_bt_compliance_pct": round(phys_eval.physical_bt_compliance_pct, 1),
            "mean_bt_drift_k": round(phys_eval.mean_bt_drift_k, 2),
            "fluid_divergence": round(phys_eval.fluid_divergence, 6),
            "mean_latency_ms": round(res.mean_latency_ms, 2),
        }
        eval_real["convective_indicators"] = {
            "convective_activity_index": round(nowcast.convective_activity_index, 1),
            "convective_activity_level": nowcast.overall_convective_level,
            "is_calibrated_probability": nowcast.is_calibrated_probability,
            "cloud_top_bt_tendency_k_15m": round(nowcast.cloud_top_bt_tendency_k_15m, 1),
            "experimental_ot_candidates_count": nowcast.experimental_ot_candidates_count,
            "precipitation_ground_truth_status": nowcast.precipitation_ground_truth_status,
        }

    report["evaluations"].append(eval_real)

    # --------------------------------------------------------------------------
    # Evaluation 2: Continuous Synthetic Ground Truth Benchmark
    # --------------------------------------------------------------------------
    subdivisions = 15
    sub_timesteps = [round(i / subdivisions, 4) for i in range(1, subdivisions)]

    d0_sim = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=(256, 256), t_normalized=0.0)
    d1_sim = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=(256, 256), t_normalized=1.0)
    t0_sim = parser.to_normalized_tensor(d0_sim, device=device)
    t1_sim = parser.to_normalized_tensor(d1_sim, device=device)

    with torch.no_grad():
        res_sim = interpolator.interpolate(t0_sim, t1_sim, sub_timesteps=sub_timesteps)

    horizon_evals = []
    blink_psnrs, pers_psnrs, lin_psnrs = [], [], []

    for i, t in enumerate(sub_timesteps):
        d_gt = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=(256, 256), t_normalized=t)
        t_gt = parser.to_normalized_tensor(d_gt, device=device)
        synth_t = res_sim.synthesized_frames[i]

        b_metrics = PhysicsEvaluator.evaluate_with_baselines(
            synthesized=synth_t,
            ground_truth=t_gt,
            frame_0=t0_sim,
            frame_1=t1_sim,
            t_normalized=t,
            data_range=1.0,
        )

        horizon_min = round(t * 15.0, 1)
        horizon_evals.append({
            "horizon_minutes": horizon_min,
            "normalized_t": t,
            "blink_psnr_db": b_metrics["blink"]["psnr_db"],
            "linear_blend_psnr_db": b_metrics["linear_blend"]["psnr_db"],
            "persistence_psnr_db": b_metrics["persistence"]["psnr_db"],
            "psnr_gain_over_linear_db": round(b_metrics["blink"]["psnr_db"] - b_metrics["linear_blend"]["psnr_db"], 2),
            "blink_ssim": b_metrics["blink"]["ssim"],
            "linear_ssim": b_metrics["linear_blend"]["ssim"],
            "persistence_ssim": b_metrics["persistence"]["ssim"],
        })
        blink_psnrs.append(b_metrics["blink"]["psnr_db"])
        pers_psnrs.append(b_metrics["persistence"]["psnr_db"])
        lin_psnrs.append(b_metrics["linear_blend"]["psnr_db"])

    eval_sim = {
        "dataset_name": "Continuous Atmospheric Simulation (Cyclone Scenario)",
        "data_tier": "MODEL OUTPUT / SYNTHETIC DEMONSTRATION",
        "operating_mode": "TEMPORAL_INTERPOLATION",
        "independent_ground_truth_available": True,
        "sample_count_n": len(sub_timesteps),
        "aggregate_scorecard": {
            "mean_blink_psnr_db": round(float(np.mean(blink_psnrs)), 2),
            "mean_linear_blend_psnr_db": round(float(np.mean(lin_psnrs)), 2),
            "mean_persistence_psnr_db": round(float(np.mean(pers_psnrs)), 2),
            "mean_psnr_advantage_over_linear_db": round(float(np.mean(blink_psnrs) - np.mean(lin_psnrs)), 2),
        },
        "horizon_breakdown": horizon_evals,
    }

    report["evaluations"].append(eval_sim)

    # Write report
    out_path = root_dir / "docs" / "EVALUATION_REPORT.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Generated machine-readable audit report: {out_path}")
    print(f"   Evaluation 1 (Real Data): Independent GT Available = {eval_real['independent_ground_truth_available']} (Suppressed)")
    print(f"   Evaluation 2 (Synthetic GT): N = {eval_sim['sample_count_n']} frames, Mean Gain = +{eval_sim['aggregate_scorecard']['mean_psnr_advantage_over_linear_db']} dB")


if __name__ == "__main__":
    run_scientific_audit()
