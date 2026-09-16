"""
FastAPI High-Throughput REST Gateway & Operational Backend for Project BLINK.
Provides endpoints for rapid-scanning frame synthesis, telemetry, physical evaluation,
and interactive browser console streaming.
"""

import base64
import io
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Dict, List, Optional, Union, Any
import numpy as np
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.ingestion.mosdac_client import MOSDACClient
from src.ingestion.mosdac_parser import CHANNEL_CALIBRATION_BOUNDS, MOSDACParser, SyntheticMOSDACSimulator
from src.ingestion.preprocessor import GeoNormalizer
from src.ingestion.real_satellite_fetcher import RealSatelliteFetcher
from src.pipeline.interpolator import AeroInterpolator
from src.pipeline.nowcasting import ConvectiveNowcaster, StormTrackPredictor
from src.pipeline.physics_eval import PhysicsEvaluator


DEFAULT_MAX_UPLOAD_DIMENSION = 768
MAX_UPLOAD_DIMENSION = 1536
MAX_GRID_SIZE = 1024
MAX_CADENCE_STEPS = 60
MAX_SUB_TIMESTEPS = 120
HDF5_EXTENSIONS = {".h5", ".hdf5", ".nc", ".nc4"}


app = FastAPI(
    title="BLINK: Zero-Payload Rapid Scanning Engine",
    description="Spatiotemporal neural kinematics and frame synthesis engine for geostationary multi-spectral imagery",
    version="1.0.0",
)

# Enable CORS for web clients & dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy global interpolator. Health checks should not pay model initialization cost.
_interpolator: Optional[AeroInterpolator] = None
_interpolator_lock = threading.Lock()
_inference_lock = threading.Lock()


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_interpolator() -> AeroInterpolator:
    global _interpolator
    if _interpolator is None:
        with _interpolator_lock:
            if _interpolator is None:
                _interpolator = AeroInterpolator(
                    device="auto",
                    channels=["IMG_VIS", "IMG_WV", "IMG_TIR1"],
                    refinement_mode="flow",
                    raft_pretrained=True,
                )
    return _interpolator


# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "operational"
    project_name: str = "BLINK"
    version: str = "1.0.0"
    device: str
    cuda_available: bool
    torch_version: str
    active_memory_mb: float
    model_weights_loaded: bool
    interpolator_loaded: bool
    flow_backend: str
    engine_mode: str


class InterpolationRequest(BaseModel):
    sub_timesteps: List[float] = Field(
        default=[0.2, 0.4, 0.6, 0.8],
        description="Target relative sub-timestamps between 0.0 and 1.0",
    )
    frame_0_base64: Optional[str] = Field(None, description="Base64 PNG/JPEG string of start frame T_0")
    frame_1_base64: Optional[str] = Field(None, description="Base64 PNG/JPEG string of end frame T_1")
    scenario: Optional[str] = Field(
        None, description="Pre-loaded scenario name ('cyclone', 'cloudburst', 'shear')"
    )
    return_rgb_preview: bool = Field(True, description="Return base64 encoded RGB preview strings")


class InterpolationResponse(BaseModel):
    success: bool
    num_synthesized_frames: int
    sub_timesteps: List[float]
    synthesized_previews_base64: List[str]
    linear_blend_previews_base64: List[str]
    flow_01_summary: Dict[str, float]
    mean_latency_ms: float
    per_frame_latencies_ms: List[float]
    fluid_divergence: float


class SimulationRequest(BaseModel):
    scenario: str = Field(
        "cyclone",
        description="Simulation scenario: 'cyclone', 'cloudburst', or 'shear'",
    )
    grid_size: int = Field(512, ge=64, le=MAX_GRID_SIZE, description="Spatial resolution (e.g. 512 for 512x512)")
    cadence_steps: int = Field(15, ge=2, le=MAX_CADENCE_STEPS, description="Number of temporal subdivisions (e.g. 15 for 15-min to 1-min upsampling)")


class RealDataQueryRequest(BaseModel):
    source: str = Field("MOSDAC_INSAT3DS", description="Data source ('MOSDAC_INSAT3DS', 'REAL_SATELLITE')")
    date: str = Field("2026-08-28", description="Target date (YYYY-MM-DD)")
    region: str = Field("indian_subcontinent", description="Geographic region key")


class RealDataInterpolateRequest(BaseModel):
    source: str = Field("REAL_SATELLITE", description="Source ('REAL_SATELLITE', 'MOSDAC_INSAT3DS', 'simulation')")
    scenario: Optional[str] = Field("cyclone", description="Fallback scenario if simulation is chosen")
    date: str = Field("2026-08-28", description="Observation date (YYYY-MM-DD)")
    time: str = Field("10:00", description="Observation time (HH:MM)")
    region: str = Field("indian_subcontinent", description="Geographic region key")
    cadence_steps: int = Field(15, ge=2, le=MAX_CADENCE_STEPS, description="Number of subdivisions")
    grid_size: int = Field(512, ge=64, le=MAX_GRID_SIZE, description="Spatial grid resolution")
    channel: Optional[str] = Field("composite", description="Display channel mode ('TIR-1', 'WV', 'VIS', 'composite')")


class MOSDACConfigRequest(BaseModel):
    username: str = Field(..., description="MOSDAC portal username or email")
    password: Optional[str] = Field(None, description="MOSDAC portal password")
    api_token: Optional[str] = Field(None, description="MOSDAC API authentication token")


# ------------------------------------------------------------------------------
# Helper Utilities
# ------------------------------------------------------------------------------
def _tensor_to_base64(tensor: torch.Tensor, mode: str = "composite") -> str:
    """Converts a normalized tensor (1, C, H, W) to a base64 encoded PNG string."""
    rgb = GeoNormalizer.tensor_to_rgb_preview(tensor, mode=mode)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_diagnostics(t0: torch.Tensor, t1: torch.Tensor, synthesized: List[torch.Tensor], flow: torch.Tensor) -> Dict[str, object]:
    """Compact, source-independent diagnostic samples from the actual model tensors."""
    def tir(frame: torch.Tensor) -> np.ndarray:
        data = frame.detach().cpu().squeeze(0).numpy()
        channel = data[2] if data.shape[0] >= 3 else data[-1]
        return 298.0 - np.clip(channel, 0.0, 1.0) * 105.0

    def profile(frame: torch.Tensor) -> List[float]:
        temp = tir(frame)
        return np.mean(temp, axis=0).astype(float).round(2).tolist()

    def histogram(frame: torch.Tensor) -> List[int]:
        counts, _ = np.histogram(tir(frame), bins=28, range=(180.0, 305.0))
        return counts.astype(int).tolist()

    flow_np = flow.detach().cpu().squeeze(0).numpy()
    step_y = max(1, flow_np.shape[1] // 24)
    step_x = max(1, flow_np.shape[2] // 24)
    return {
        "tir_profiles_k": {"t0": profile(t0), "t1": profile(t1), "synthesized": [profile(frame) for frame in synthesized]},
        "tir_histograms": {"t0": histogram(t0), "t1": histogram(t1), "synthesized": [histogram(frame) for frame in synthesized]},
        "flow_grid_pixels": {"source_width": int(flow_np.shape[2]), "source_height": int(flow_np.shape[1]), "u": flow_np[0, ::step_y, ::step_x].astype(float).round(3).tolist(), "v": flow_np[1, ::step_y, ::step_x].astype(float).round(3).tolist()},
    }


def _bounded_spatial_size(height: int, width: int, max_dimension: int) -> tuple[int, int]:
    if max_dimension <= 0:
        return height, width
    scale = min(1.0, float(max_dimension) / float(max(height, width)))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _resize_image_for_engine(
    image: Image.Image,
    target_spatial_size: Optional[tuple[int, int]],
    max_dimension: int,
) -> tuple[Image.Image, tuple[int, int]]:
    if target_spatial_size is None:
        target_spatial_size = _bounded_spatial_size(image.height, image.width, max_dimension)

    target_height, target_width = target_spatial_size
    if image.size != (target_width, target_height):
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return image, target_spatial_size


def _base64_to_tensor(b64_str: str, device: torch.device) -> torch.Tensor:
    """Decodes a base64 image string to a normalized tensor (1, 3, H, W) in [0.0, 1.0]."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_data = base64.b64decode(b64_str)
    pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_data))).convert("RGB")
    arr = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    return tensor


def _read_hdf5_upload_to_tensor(
    raw: bytes,
    filename: str,
    parser: MOSDACParser,
    device: torch.device,
    target_spatial_size: Optional[tuple[int, int]],
    max_dimension: int,
) -> tuple[torch.Tensor, tuple[int, int], Dict[str, float]]:
    suffix = Path(filename).suffix.lower() or ".h5"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        if target_spatial_size is None:
            target_spatial_size = _bounded_spatial_size(512, 512, max_dimension)

        channel_data, geo_bounds = parser.read_hdf5_sector(tmp_path, region="indian_subcontinent", target_size=target_spatial_size)
        return parser.to_normalized_tensor(channel_data, device=device), target_spatial_size, geo_bounds
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _read_upload_to_tensor(
    upload: UploadFile,
    parser: MOSDACParser,
    device: torch.device,
    target_spatial_size: Optional[tuple[int, int]] = None,
    max_dimension: int = DEFAULT_MAX_UPLOAD_DIMENSION,
) -> tuple[torch.Tensor, tuple[int, int], Dict[str, float]]:
    filename = upload.filename or "upload"
    suffix = Path(filename).suffix.lower()
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{filename} is empty")

    default_bounds = {"latNorth": 35.51, "latSouth": 4.97, "lonWest": 60.32, "lonEast": 105.02}

    if suffix in HDF5_EXTENSIONS:
        try:
            return _read_hdf5_upload_to_tensor(raw, filename, parser, device, target_spatial_size, max_dimension)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse {filename} as HDF5/NetCDF: {exc}") from exc

    try:
        pil_img = Image.open(io.BytesIO(raw))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse {filename} as an image: {exc}") from exc

    pil_img, target_spatial_size = _resize_image_for_engine(pil_img, target_spatial_size, max_dimension)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, target_spatial_size, default_bounds


def _validate_sub_timesteps(sub_timesteps: List[float]) -> List[float]:
    if not sub_timesteps:
        raise HTTPException(status_code=422, detail="sub_timesteps must contain at least one value")
    if len(sub_timesteps) > MAX_SUB_TIMESTEPS:
        raise HTTPException(status_code=422, detail=f"sub_timesteps may not exceed {MAX_SUB_TIMESTEPS} values")
    clean = []
    for t in sub_timesteps:
        t_float = float(t)
        if not 0.0 < t_float < 1.0:
            raise HTTPException(status_code=422, detail="sub_timesteps must be between 0.0 and 1.0")
        clean.append(t_float)
    return clean


def _ensure_same_tensor_size(frame_0: torch.Tensor, frame_1: torch.Tensor) -> torch.Tensor:
    if frame_0.shape == frame_1.shape:
        return frame_1
    return torch.nn.functional.interpolate(
        frame_1,
        size=frame_0.shape[-2:],
        mode="bilinear",
        align_corners=True,
    )


def _render_flow_visualization(flow_tensor: torch.Tensor) -> str:
    """Renders the optical flow field (1, 2, H, W) as a professional meteorological velocity colormap PNG."""
    flow = flow_tensor.squeeze(0).cpu().numpy()  # (2, H, W)
    u, v = flow[0], flow[1]
    magnitude = np.sqrt(u ** 2 + v ** 2)
    mag_norm = np.clip(magnitude / (magnitude.max() + 1e-8), 0.0, 1.0)

    # Professional meteorological velocity palette (Dark Navy -> Cyan -> Emerald -> Gold -> Coral Red)
    # Smooth multi-stop gradient
    h, w = mag_norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Vectorized colormap mapping
    t = mag_norm
    # Background: (8, 16, 32)
    # 0.0 - 0.25: (8, 16, 32) -> (6, 182, 212)
    # 0.25 - 0.50: (6, 182, 212) -> (16, 185, 129)
    # 0.50 - 0.75: (16, 185, 129) -> (245, 158, 11)
    # 0.75 - 1.00: (245, 158, 11) -> (239, 68, 68)

    m1 = t < 0.25
    f1 = t / 0.25
    rgb[m1, 0] = (8 + (6 - 8) * f1[m1]).astype(np.uint8)
    rgb[m1, 1] = (16 + (182 - 16) * f1[m1]).astype(np.uint8)
    rgb[m1, 2] = (32 + (212 - 32) * f1[m1]).astype(np.uint8)

    m2 = (t >= 0.25) & (t < 0.5)
    f2 = (t - 0.25) / 0.25
    rgb[m2, 0] = (6 + (16 - 6) * f2[m2]).astype(np.uint8)
    rgb[m2, 1] = (182 + (185 - 182) * f2[m2]).astype(np.uint8)
    rgb[m2, 2] = (212 + (129 - 212) * f2[m2]).astype(np.uint8)

    m3 = (t >= 0.5) & (t < 0.75)
    f3 = (t - 0.5) / 0.25
    rgb[m3, 0] = (16 + (245 - 16) * f3[m3]).astype(np.uint8)
    rgb[m3, 1] = (185 + (158 - 185) * f3[m3]).astype(np.uint8)
    rgb[m3, 2] = (129 + (11 - 129) * f3[m3]).astype(np.uint8)

    m4 = t >= 0.75
    f4 = np.clip((t - 0.75) / 0.25, 0.0, 1.0)
    rgb[m4, 0] = (245 + (239 - 245) * f4[m4]).astype(np.uint8)
    rgb[m4, 1] = (158 + (68 - 158) * f4[m4]).astype(np.uint8)
    rgb[m4, 2] = (11 + (68 - 11) * f4[m4]).astype(np.uint8)

    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ------------------------------------------------------------------------------
# REST Endpoints
# ------------------------------------------------------------------------------
@app.get("/v1/health", response_model=HealthResponse)
def get_health():
    """Returns system operational health, hardware capability, and GPU memory metrics."""
    active_mem = 0.0
    if torch.cuda.is_available():
        active_mem = float(torch.cuda.memory_allocated() / (1024 * 1024))
    interpolator = _interpolator
    device = interpolator.device if interpolator is not None else _select_device()

    return HealthResponse(
        status="operational",
        project_name="BLINK",
        version="1.0.0",
        device=str(device),
        cuda_available=torch.cuda.is_available(),
        torch_version=torch.__version__,
        active_memory_mb=active_mem,
        model_weights_loaded=bool(interpolator and interpolator.raft.use_torchvision_raft),
        interpolator_loaded=interpolator is not None,
        flow_backend=interpolator.flow_backend if interpolator is not None else "not_loaded",
        engine_mode=interpolator.refinement_mode if interpolator is not None else "flow",
    )


@app.get("/v1/channels")
def get_channels():
    """Lists supported multi-spectral imager spectral bands and calibration specs."""
    interpolator = get_interpolator()
    return {
        "sensor": "6-Channel Geostationary Imager",
        "channels": CHANNEL_CALIBRATION_BOUNDS,
        "default_active": interpolator.channels,
    }


@app.post("/v1/interpolate/frames", response_model=InterpolationResponse)
def interpolate_frames(req: InterpolationRequest):
    """
    Synthesizes intermediate frames between T_0 and T_1 at requested sub-timestamps.
    """
    interpolator = get_interpolator()
    device = interpolator.device
    sub_timesteps = _validate_sub_timesteps(req.sub_timesteps)

    # Check if scenario is requested or custom base64 images
    if req.scenario:
        if req.scenario.lower() == "cloudburst":
            data_0 = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(t_normalized=0.0)
            data_1 = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(t_normalized=1.0)
        else:
            data_0 = SyntheticMOSDACSimulator.generate_cyclone_frame(t_normalized=0.0)
            data_1 = SyntheticMOSDACSimulator.generate_cyclone_frame(t_normalized=1.0)

        t_0 = interpolator.parser.to_normalized_tensor(data_0, device=device)
        t_1 = interpolator.parser.to_normalized_tensor(data_1, device=device)
    elif req.frame_0_base64 and req.frame_1_base64:
        t_0 = _base64_to_tensor(req.frame_0_base64, device=device)
        t_1 = _base64_to_tensor(req.frame_1_base64, device=device)
        t_1 = _ensure_same_tensor_size(t_0, t_1)
    else:
        # Default to cyclone simulation
        data_0 = SyntheticMOSDACSimulator.generate_cyclone_frame(t_normalized=0.0)
        data_1 = SyntheticMOSDACSimulator.generate_cyclone_frame(t_normalized=1.0)
        t_0 = interpolator.parser.to_normalized_tensor(data_0, device=device)
        t_1 = interpolator.parser.to_normalized_tensor(data_1, device=device)

    try:
        with _inference_lock:
            result = interpolator.interpolate(t_0, t_1, sub_timesteps=sub_timesteps)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Encode previews
    synth_b64 = [_tensor_to_base64(frame) for frame in result.synthesized_frames] if req.return_rgb_preview else []
    linear_b64 = [_tensor_to_base64(frame) for frame in result.linear_blends] if req.return_rgb_preview else []

    u_mean = float(result.flow_01[:, 0, :, :].mean().item())
    v_mean = float(result.flow_01[:, 1, :, :].mean().item())
    mag_max = float(torch.sqrt(result.flow_01[:, 0, :, :]**2 + result.flow_01[:, 1, :, :]**2).max().item())

    return InterpolationResponse(
        success=True,
        num_synthesized_frames=len(result.synthesized_frames),
        sub_timesteps=result.sub_timesteps,
        synthesized_previews_base64=synth_b64,
        linear_blend_previews_base64=linear_b64,
        flow_01_summary={"mean_dx_pixels": u_mean, "mean_dy_pixels": v_mean, "max_displacement_pixels": mag_max},
        mean_latency_ms=result.mean_latency_ms,
        per_frame_latencies_ms=result.per_frame_latencies_ms,
        fluid_divergence=result.fluid_divergence,
    )


@app.post("/v1/simulate/scenario")
def simulate_scenario(req: SimulationRequest):
    """
    Generates and synthesizes a full multi-spectral rapid-scan sequence for testing and benchmarking.
    """
    steps = max(2, req.cadence_steps)
    sub_timesteps = [round(i / steps, 4) for i in range(1, steps)]

    interpolator = get_interpolator()
    device = interpolator.device
    if req.scenario.lower() == "cloudburst":
        d0 = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(grid_size=(req.grid_size, req.grid_size), t_normalized=0.0)
        d1 = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(grid_size=(req.grid_size, req.grid_size), t_normalized=1.0)
    else:
        d0 = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=(req.grid_size, req.grid_size), t_normalized=0.0)
        d1 = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=(req.grid_size, req.grid_size), t_normalized=1.0)

    t0 = interpolator.parser.to_normalized_tensor(d0, device=device)
    t1 = interpolator.parser.to_normalized_tensor(d1, device=device)

    with _inference_lock:
        result = interpolator.interpolate(t0, t1, sub_timesteps=sub_timesteps)

    t0_b64 = _tensor_to_base64(t0)
    t1_b64 = _tensor_to_base64(t1)
    synth_b64 = [_tensor_to_base64(f) for f in result.synthesized_frames]
    linear_b64 = [_tensor_to_base64(f) for f in result.linear_blends]

    # Evaluate against simulated continuous ground-truth for middle step (t=0.5)
    mid_idx = len(sub_timesteps) // 2
    t_mid = sub_timesteps[mid_idx]
    if req.scenario.lower() == "cloudburst":
        d_gt = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(grid_size=(req.grid_size, req.grid_size), t_normalized=t_mid)
    else:
        d_gt = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=(req.grid_size, req.grid_size), t_normalized=t_mid)

    t_gt = interpolator.parser.to_normalized_tensor(d_gt, device=device)
    eval_report = PhysicsEvaluator.evaluate_synthesis(
        synthesized=result.synthesized_frames[mid_idx].to(device),
        ground_truth=t_gt,
        frame_0=t0,
        frame_1=t1,
        t_normalized=t_mid,
        flow=result.flow_01.to(device),
        latency_ms=result.mean_latency_ms,
    )

    u_mean = float(result.flow_01[:, 0, :, :].mean().item())
    v_mean = float(result.flow_01[:, 1, :, :].mean().item())
    mag_max = float(torch.sqrt(result.flow_01[:, 0, :, :]**2 + result.flow_01[:, 1, :, :]**2).max().item())

    track_report = StormTrackPredictor.predict_track_and_cone(t0, t1, result.flow_01)
    nowcast_report = ConvectiveNowcaster.evaluate_convective_risk(t0, t1, result.flow_01)

    return {
        "scenario": req.scenario,
        "operating_mode": "TEMPORAL_INTERPOLATION",
        "evaluation_protocol": "Synthetic Continuous Ground Truth Benchmark",
        "data_source_mode": "SYNTHETIC_DEMONSTRATION",
        "cadence_upsample_factor": f"{steps}x",
        "engine_mode": result.engine_mode,
        "flow_backend": result.flow_backend,
        "t0_base64": t0_b64,
        "t1_base64": t1_b64,
        "sub_timesteps": sub_timesteps,
        "synthesized_frames": synth_b64,
        "linear_blends": linear_b64,
        "metrics": eval_report.to_dict(),
        "flow_summary": {
            "mean_dx_pixels": u_mean,
            "mean_dy_pixels": v_mean,
            "max_displacement_pixels": mag_max,
        },
        "flow_visualization_base64": _render_flow_visualization(result.flow_01),
        "diagnostics": _build_diagnostics(t0, t1, result.synthesized_frames, result.flow_01),
        "storm_track": track_report.to_dict(),
        "convective_nowcast": nowcast_report.to_dict(),
    }


@app.post("/v1/interpolate/upload")
async def interpolate_upload(
    file_t0: UploadFile = File(...),
    file_t1: UploadFile = File(...),
    cadence_steps: int = Form(15),
    max_dimension: int = Form(DEFAULT_MAX_UPLOAD_DIMENSION),
):
    """
    Ingests two user-uploaded observation files (images, HDF5, or NetCDF4),
    runs the full synthesis pipeline, and returns frame sequences.
    """
    if cadence_steps < 2 or cadence_steps > MAX_CADENCE_STEPS:
        raise HTTPException(status_code=422, detail=f"cadence_steps must be between 2 and {MAX_CADENCE_STEPS}")
    if max_dimension < 128 or max_dimension > MAX_UPLOAD_DIMENSION:
        raise HTTPException(status_code=422, detail=f"max_dimension must be between 128 and {MAX_UPLOAD_DIMENSION}")

    interpolator = get_interpolator()
    device = interpolator.device
    steps = max(2, cadence_steps)
    sub_timesteps = [round(i / steps, 4) for i in range(1, steps)]

    t0, target_spatial_size, geo_bounds = _read_upload_to_tensor(
        file_t0,
        parser=interpolator.parser,
        device=device,
        max_dimension=max_dimension,
    )
    t1, _, _ = _read_upload_to_tensor(
        file_t1,
        parser=interpolator.parser,
        device=device,
        target_spatial_size=target_spatial_size,
        max_dimension=max_dimension,
    )

    try:
        with _inference_lock:
            result = interpolator.interpolate(t0, t1, sub_timesteps=sub_timesteps)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    t0_b64 = _tensor_to_base64(t0)
    t1_b64 = _tensor_to_base64(t1)
    synth_b64 = [_tensor_to_base64(f) for f in result.synthesized_frames]
    linear_b64 = [_tensor_to_base64(f) for f in result.linear_blends]

    u_mean = float(result.flow_01[:, 0, :, :].mean().item())
    v_mean = float(result.flow_01[:, 1, :, :].mean().item())
    mag_max = float(torch.sqrt(result.flow_01[:, 0, :, :]**2 + result.flow_01[:, 1, :, :]**2).max().item())

    track_report = StormTrackPredictor.predict_track_and_cone(t0, t1, result.flow_01, geo_bounds=geo_bounds)
    nowcast_report = ConvectiveNowcaster.evaluate_convective_risk(t0, t1, result.flow_01, geo_bounds=geo_bounds)

    # Evaluation on user uploaded frames:
    # No independent intermediate observation is provided for 2-frame endpoint interpolation.
    mid_idx = len(result.synthesized_frames) // 2
    synth_mid = result.synthesized_frames[mid_idx]
    mid_t = sub_timesteps[mid_idx] if sub_timesteps else 0.5

    eval_report = PhysicsEvaluator.evaluate_synthesis(
        synthesized=synth_mid,
        ground_truth=None,  # No independent ground truth available for 2-frame endpoint interpolation
        frame_0=t0,
        frame_1=t1,
        t_normalized=mid_t,
        flow=result.flow_01,
        latency_ms=result.mean_latency_ms,
    )
    metrics_dict = eval_report.to_dict()
    metrics_dict["rmse_pct"] = None
    metrics_dict["temporal_consistency"] = None

    eval_protocol = (
        "Kinematic Endpoint Re-projection (Forward Warp vs Observed Frame)"
        if eval_report.has_ground_truth
        else "Internal Diagnostics Only (Held-out observation unavailable)"
    )

    return {
        "scenario": "user_upload",
        "operating_mode": "TEMPORAL_INTERPOLATION",
        "evaluation_protocol": eval_protocol,
        "data_source_mode": "USER_UPLOADED_DATA",
        "cadence_upsample_factor": f"{steps}x",
        "engine_mode": result.engine_mode,
        "flow_backend": result.flow_backend,
        "output_height": int(t0.shape[-2]),
        "output_width": int(t0.shape[-1]),
        "geo_bounds": geo_bounds,
        "t0_base64": t0_b64,
        "t1_base64": t1_b64,
        "sub_timesteps": sub_timesteps,
        "synthesized_frames": synth_b64,
        "linear_blends": linear_b64,
        "metrics": metrics_dict,
        "flow_summary": {
            "mean_dx_pixels": u_mean,
            "mean_dy_pixels": v_mean,
            "max_displacement_pixels": mag_max,
        },
        "flow_visualization_base64": _render_flow_visualization(result.flow_01),
        "diagnostics": _build_diagnostics(t0, t1, result.synthesized_frames, result.flow_01),
        "storm_track": track_report.to_dict(),
        "convective_nowcast": nowcast_report.to_dict(),
    }


# Global fetchers
_mosdac_client = MOSDACClient()
_satellite_fetcher = RealSatelliteFetcher()


@app.post("/v1/config/mosdac")
def configure_mosdac(req: MOSDACConfigRequest):
    """Saves user MOSDAC authentication credentials for automated INSAT-3DS data download."""
    pwd = req.password or req.api_token or ""
    _mosdac_client.set_credentials(req.username, pwd)
    return {
        "status": "success",
        "message": "MOSDAC credentials updated successfully in config.json",
        "is_configured": _mosdac_client.is_configured,
    }


def _find_available_insat_files() -> Dict[str, Any]:
    """
    Auto-discovers available INSAT datasets across all operational directories:
    1. config.local.json / config.json 'download_path'
    2. 'MOSDAC'
    3. 'data/raw_netcdf/mosdac'
    4. 'data/raw_netcdf'
    5. 'data/exported_images'
    """
    search_dirs: List[Path] = []

    # Check config files
    for cfg_name in ["config.local.json", "config.json"]:
        cfg_path = Path(cfg_name)
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    dl_path = cfg.get("download_settings", {}).get("download_path")
                    if dl_path:
                        p = Path(dl_path)
                        if p not in search_dirs:
                            search_dirs.append(p)
            except Exception:
                pass

    # Standard directories
    search_dirs.extend([
        Path("MOSDAC"),
        Path("data/raw_netcdf/mosdac"),
        Path("data/raw_netcdf"),
    ])

    # Env override
    env_dl = os.environ.get("MOSDAC_DOWNLOAD_PATH")
    if env_dl and Path(env_dl) not in search_dirs:
        search_dirs.insert(0, Path(env_dl))

    # Scan for HDF5 files
    h5_files: List[Path] = []
    seen_stems = set()
    valid_dirs = []

    for sdir in search_dirs:
        if sdir.exists() and sdir.is_dir():
            valid_dirs.append(str(sdir.resolve()))
            for pat in ["*.h5", "*.hdf5", "*.nc", "*.nc4"]:
                for p in sorted(list(sdir.glob(pat))):
                    if p.stem not in seen_stems and not p.name.endswith(".part"):
                        seen_stems.add(p.stem)
                        h5_files.append(p)

    # Scan for exported images in data/exported_images
    exp_dir = Path("data/exported_images")
    exported_images: List[Path] = []
    if exp_dir.exists() and exp_dir.is_dir():
        exported_images = sorted(list(exp_dir.glob("*.jpg")) + list(exp_dir.glob("*.png")))

    # Find matching exported composite pairs (e.g. *RGB_COMPOSITE.jpg or *IMG_TIR1_colored.jpg)
    rgb_composites = sorted(list(exp_dir.glob("*RGB_COMPOSITE.jpg"))) if exp_dir.exists() else []

    return {
        "h5_files": h5_files,
        "exported_images": exported_images,
        "rgb_composites": rgb_composites,
        "search_dirs": valid_dirs,
        "has_h5": len(h5_files) >= 2,
        "has_exported": len(rgb_composites) >= 2,
    }


@app.get("/v1/datasets/scan")
def scan_local_datasets():
    """Returns all locally discovered INSAT HDF5 files and exported images."""
    discovery = _find_available_insat_files()
    return {
        "status": "success",
        "h5_files_count": len(discovery["h5_files"]),
        "h5_files": [str(p) for p in discovery["h5_files"]],
        "exported_images_count": len(discovery["exported_images"]),
        "rgb_composite_pairs_count": len(discovery["rgb_composites"]),
        "searched_directories": discovery["search_dirs"],
        "ready_for_offline_insat": discovery["has_h5"] or discovery["has_exported"],
    }


@app.post("/v1/fetch/query")
def query_available_scans(req: RealDataQueryRequest):
    """Lists available satellite scans and timestamps for a date and region."""
    if req.source == "MOSDAC_INSAT3DS":
        scans = _mosdac_client.query_available_scans(req.date)
    else:
        scans = _mosdac_client.query_available_scans(req.date)

    return {
        "source": req.source,
        "date": req.date,
        "region": req.region,
        "available_scans": scans,
    }


@app.post("/v1/fetch/realtime")
def interpolate_realtime_feed(req: RealDataInterpolateRequest):
    """
    Fetches real multi-spectral observations (T0 & T1) for chosen date, time, and region,
    runs the temporal synthesis engine, and computes NETRA cloudburst & Dvorak cyclone nowcasts.
    """
    interpolator = get_interpolator()
    device = interpolator.device
    steps = max(2, req.cadence_steps)
    sub_timesteps = [round(i / steps, 4) for i in range(1, steps)]

    target_size = (req.grid_size, req.grid_size)
    geo_bounds = RealSatelliteFetcher.REGIONS.get(req.region, RealSatelliteFetcher.REGIONS["indian_subcontinent"])

    discovery = _find_available_insat_files()
    local_h5 = discovery["h5_files"]
    rgb_composites = discovery["rgb_composites"]

    d_mid = None
    if req.source in {"MOSDAC_INSAT3DS", "real", "REAL", "mosdac", "INSAT3DS"} and len(local_h5) >= 2:
        if len(local_h5) >= 3:
            file_0 = local_h5[0]
            file_mid = local_h5[1]
            file_1 = local_h5[2]
            d0, geo_bounds = interpolator.parser.read_hdf5_sector(file_0, region=req.region, target_size=target_size)
            d_mid, _ = interpolator.parser.read_hdf5_sector(file_mid, region=req.region, target_size=target_size)
            d1, _ = interpolator.parser.read_hdf5_sector(file_1, region=req.region, target_size=target_size)
        else:
            file_0 = local_h5[0]
            file_1 = local_h5[1]
            d0, geo_bounds = interpolator.parser.read_hdf5_sector(file_0, region=req.region, target_size=target_size)
            d1, _ = interpolator.parser.read_hdf5_sector(file_1, region=req.region, target_size=target_size)
        meta = {
            "source": f"INSAT-3DS Level-1B (Local HDF5 from {file_0.parent.name}/)",
            "observation_date": "2026-08-30",
            "t0_time_utc": file_0.stem,
            "t1_time_utc": file_1.stem,
            "region_name": req.region,
            "geo_bounds": geo_bounds,
            "local_path": str(file_0.resolve()),
        }
    elif req.source in {"MOSDAC_INSAT3DS", "real", "REAL", "mosdac", "INSAT3DS"} and len(rgb_composites) >= 2:
        # Load from exported_images RGB composites
        file_0 = rgb_composites[0]
        file_1 = rgb_composites[1]
        im0 = Image.open(file_0).convert("RGB").resize(target_size)
        im1 = Image.open(file_1).convert("RGB").resize(target_size)
        arr0 = np.array(im0, dtype=np.float32) / 255.0
        arr1 = np.array(im1, dtype=np.float32) / 255.0
        d0 = {"IMG_VIS": arr0[:, :, 0], "IMG_WV": arr0[:, :, 1], "IMG_TIR1": arr0[:, :, 2]}
        d1 = {"IMG_VIS": arr1[:, :, 0], "IMG_WV": arr1[:, :, 1], "IMG_TIR1": arr1[:, :, 2]}
        if len(rgb_composites) >= 3:
            im_mid = Image.open(rgb_composites[2]).convert("RGB").resize(target_size)
            arr_mid = np.array(im_mid, dtype=np.float32) / 255.0
            d_mid = {"IMG_VIS": arr_mid[:, :, 0], "IMG_WV": arr_mid[:, :, 1], "IMG_TIR1": arr_mid[:, :, 2]}
        meta = {
            "source": f"INSAT-3DS Exported Composites (from {file_0.parent.name}/)",
            "observation_date": "2026-08-30",
            "t0_time_utc": file_0.stem,
            "t1_time_utc": file_1.stem,
            "region_name": req.region,
            "geo_bounds": geo_bounds,
            "local_path": str(file_0.resolve()),
        }
    elif req.source in {"simulation", "SIMULATION"}:
        is_cloudburst = bool(req.scenario and req.scenario.lower() == "cloudburst") or (req.region == "himalayan_foothills")
        if is_cloudburst:
            d0 = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(grid_size=target_size, t_normalized=0.0)
            d_mid = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(grid_size=target_size, t_normalized=0.5)
            d1 = SyntheticMOSDACSimulator.generate_convective_cloudburst_frame(grid_size=target_size, t_normalized=1.0)
        else:
            if req.region == "arabian_sea":
                center = (0.46, 0.52)
                drift = (0.025, -0.018)
            elif req.region == "western_ghats":
                center = (0.38, 0.50)
                drift = (0.022, -0.008)
            elif req.region == "bay_of_bengal":
                center = (0.54, 0.46)
                drift = (-0.028, -0.015)
            else:
                center = (0.55, 0.45)
                drift = (-0.03, -0.015)

            d0 = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=target_size, t_normalized=0.0, center=center, drift_velocity=drift)
            d_mid = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=target_size, t_normalized=0.5, center=center, drift_velocity=drift)
            d1 = SyntheticMOSDACSimulator.generate_cyclone_frame(grid_size=target_size, t_normalized=1.0, center=center, drift_velocity=drift)
        meta = {"source": "SIMULATION", "observation_date": req.date, "t0_time_utc": f"{req.date}T{req.time}:00Z", "region_name": req.region, "geo_bounds": geo_bounds}
    else:
        # Fetch real satellite observation triplet (T0, T_mid, T1) from live open feed
        d0, d_mid, d1, meta = _satellite_fetcher.fetch_frame_triplet(
            date_str=req.date,
            t0_time=req.time,
            cadence_minutes=15,
            region_key=req.region,
            target_size=target_size,
        )
        if meta and "geo_bounds" in meta:
            geo_bounds = meta["geo_bounds"]

    t0 = interpolator.parser.to_normalized_tensor(d0, device=device)
    t1 = interpolator.parser.to_normalized_tensor(d1, device=device)
    t_mid = interpolator.parser.to_normalized_tensor(d_mid, device=device) if d_mid is not None else None

    with _inference_lock:
        result = interpolator.interpolate(t0, t1, sub_timesteps=sub_timesteps)

    chan_mode = getattr(req, "channel", "composite") or "composite"
    t0_b64 = _tensor_to_base64(t0, mode=chan_mode)
    t1_b64 = _tensor_to_base64(t1, mode=chan_mode)
    synth_b64 = [_tensor_to_base64(f, mode=chan_mode) for f in result.synthesized_frames]
    linear_b64 = [_tensor_to_base64(f, mode=chan_mode) for f in result.linear_blends]

    u_mean = float(result.flow_01[:, 0, :, :].mean().item())
    v_mean = float(result.flow_01[:, 1, :, :].mean().item())
    mag_max = float(torch.sqrt(result.flow_01[:, 0, :, :]**2 + result.flow_01[:, 1, :, :]**2).max().item())

    track_report = StormTrackPredictor.predict_track_and_cone(t0, t1, result.flow_01, geo_bounds=geo_bounds)
    nowcast_report = ConvectiveNowcaster.evaluate_convective_risk(t0, t1, result.flow_01, geo_bounds=geo_bounds)

    # Evaluation on realtime observation:
    # Uses independent held-out observation (t_mid) when available,
    # or kinematic endpoint re-projection when only two boundary frames exist.
    mid_idx = len(result.synthesized_frames) // 2
    synth_mid = result.synthesized_frames[mid_idx]
    mid_t = sub_timesteps[mid_idx] if sub_timesteps else 0.5

    eval_report = PhysicsEvaluator.evaluate_synthesis(
        synthesized=synth_mid,
        ground_truth=t_mid,
        frame_0=t0,
        frame_1=t1,
        t_normalized=mid_t,
        flow=result.flow_01,
        latency_ms=result.mean_latency_ms,
    )
    metrics_dict = eval_report.to_dict()
    metrics_dict["rmse_pct"] = None
    metrics_dict["temporal_consistency"] = None

    eval_protocol = "Independent Held-Out Intermediate Observation (T_mid @ t=0.5)" if t_mid is not None else "Kinematic Endpoint Re-projection (Forward Warp vs Observed Frame)"

    response_data = {
        "scenario": f"real_{req.source.lower()}",
        "operating_mode": "TEMPORAL_INTERPOLATION",
        "evaluation_protocol": eval_protocol,
        "data_source_mode": f"REAL_{req.source.upper()}",
        "source_metadata": meta,
        "geo_bounds": geo_bounds,
        "cadence_upsample_factor": f"{steps}x",
        "engine_mode": result.engine_mode,
        "flow_backend": result.flow_backend,
        "t0_base64": t0_b64,
        "t1_base64": t1_b64,
        "sub_timesteps": sub_timesteps,
        "synthesized_frames": synth_b64,
        "linear_blends": linear_b64,
        "metrics": metrics_dict,
        "flow_summary": {
            "mean_dx_pixels": u_mean,
            "mean_dy_pixels": v_mean,
            "max_displacement_pixels": mag_max,
        },
        "flow_visualization_base64": _render_flow_visualization(result.flow_01),
        "diagnostics": _build_diagnostics(t0, t1, result.synthesized_frames, result.flow_01),
        "storm_track": track_report.to_dict(),
        "convective_nowcast": nowcast_report.to_dict(),
    }

    # Store for lightweight Wi-Fi SaaS viewer clients
    global _latest_nowcast_cache
    _latest_nowcast_cache = response_data

    return response_data


_latest_nowcast_cache: Optional[Dict[str, Any]] = None


@app.get("/v1/viewer/latest")
def get_latest_viewer_state():
    """Returns the most recent ground station processed nowcasting state for network viewers."""
    if _latest_nowcast_cache is not None:
        return _latest_nowcast_cache
    return {
        "status": "idle",
        "message": "Ground station standing by for new observation cycle.",
    }


# ------------------------------------------------------------------------------
# Built-in Interactive Web Console Endpoint
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the interactive dark-mode operational dashboard console."""
    candidates = [
        Path("ui/static/index.html"),
        Path(__file__).resolve().parent.parent.parent / "ui" / "static" / "index.html",
        Path(__file__).resolve().parent.parent / "ui" / "static" / "index.html",
    ]
    for html_path in candidates:
        if html_path.exists():
            with open(html_path, "r", encoding="utf-8") as f:
                return f.read()

    return """
    <html>
        <head><title>BLINK System</title></head>
        <body style="background:#0b0f19; color:#f3f4f6; font-family:sans-serif; padding:40px; text-align:center;">
            <h1>BLINK API Gateway</h1>
            <p>Frame Synthesis Engine is operational.</p>
            <p><a href="/docs" style="color:#38bdf8;">OpenAPI Documentation</a></p>
        </body>
    </html>
    """

