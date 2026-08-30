# PROJECT BLINK — Complete Blueprint & Study Guide
### *Bridging Latency in Imagery via Neural Kinematics*

> **Purpose of this document:** This is your single-source-of-truth "bible" for the BLINK project.
> After reading it cover-to-cover you should understand *what* the project does, *why* every
> technology was chosen, *how* every algorithm works (with the math), how threads are managed,
> where this sits in the competitive landscape, and — most importantly — be able to rebuild
> the whole thing from an empty directory.

---

## Table of Contents

1. [The Problem We're Solving](#1-the-problem-were-solving)
2. [Objective & Scope](#2-objective--scope)
3. [Novelty & Competitive Landscape](#3-novelty--competitive-landscape)
4. [Tech Stack — What & Why](#4-tech-stack--what--why)
5. [High-Level Architecture](#5-high-level-architecture)
6. [Data Ingestion Layer](#6-data-ingestion-layer)
7. [Model Layer — Deep Learning Engines](#7-model-layer--deep-learning-engines)
8. [Pipeline Layer — Orchestration & Synthesis](#8-pipeline-layer--orchestration--synthesis)
9. [The Mathematics — Explained Simply](#9-the-mathematics--explained-simply)
10. [Physics Evaluation & Benchmarking](#10-physics-evaluation--benchmarking)
11. [API & Serving Layer](#11-api--serving-layer)
12. [Threading & Concurrency Model](#12-threading--concurrency-model)
13. [UI Layer — Dashboard & Console](#13-ui-layer--dashboard--console)
14. [Directory Map & File-by-File Guide](#14-directory-map--file-by-file-guide)
15. [Build-From-Scratch Roadmap](#15-build-from-scratch-roadmap)
16. [Glossary](#16-glossary)

---

## 1. The Problem We're Solving

### The Real-World Pain

India's geostationary weather satellites — **INSAT-3DS** and **INSAT-3DR** — sit at ~36,000 km
above the equator, staring down at the Indian Ocean and subcontinent. Their imager uses a
mechanical **scan mirror** that physically sweeps across Earth's disk. One full sweep takes
**15 to 30 minutes**.

But deadly weather events don't wait:

| Weather Event            | Typical Development Time | Satellite Revisit Gap |
|--------------------------|-------------------------|-----------------------|
| Cloudburst               | 5 – 10 minutes          | 15 – 30 min           |
| Cyclone Eyewall Collapse | 10 – 20 minutes         | 15 – 30 min           |
| Microburst               | 2 – 5 minutes           | 15 – 30 min           |
| Tornado-genesis          | 5 – 15 minutes          | 15 – 30 min           |

**We are literally blind during the most critical minutes.** By the time the next scan
arrives, the storm may have already hit.

### The "Obvious" Fix (and why it's insane)

> "Just launch more satellites that scan faster!"

A dedicated rapid-scanning satellite constellation costs **Rs.2,000–5,000 crore** ($250–600M
USD) per spacecraft, plus launch costs, ground-segment overhaul, and years of development.

### Our Fix

**Don't change the hardware. Change the software.**

BLINK is a **ground-station software overlay** that takes two real satellite frames (T0 and T1,
15 minutes apart) and *synthesizes* the missing frames in between using **neural kinematics**
— AI that understands how atmospheric fluid dynamics work. We call this
**"Zero-Payload Rapid Scanning"** because we achieve rapid-scan cadence without modifying
a single satellite payload.

```
Real Frame T0                    Real Frame T1
(00:00 UTC)                      (00:15 UTC)
    |                                 |
    |    BLINK synthesizes 14         |
    |    intermediate frames          |
    |    at 1-minute intervals        |
    v                                 v
  T+0  T+1  T+2  ... T+13  T+14  T+15  (minutes)
   ^    ^    ^         ^     ^     ^
   |    |    |         |     |     |
  Real  AI   AI       AI    AI   Real
```

**Result:** Effective observation cadence drops from **15 min to 1 min**. A 15x improvement.

---

## 2. Objective & Scope

### Core Objective

Build an end-to-end frame synthesis engine that:

1. **Ingests** real INSAT-3DS Level-1B HDF5/NetCDF4 multi-spectral radiance files
2. **Estimates motion** between consecutive frames using optical flow
3. **Synthesizes** physically-accurate intermediate frames at arbitrary timestamps
4. **Validates** output against physics constraints (energy conservation, no ghosting)
5. **Serves** results via a real-time REST API for integration with ground stations
6. **Visualizes** in an operational dark-mode console for meteorologists

### Scope Boundaries

| In Scope                                        | Out of Scope                          |
|--------------------------------------------------|---------------------------------------|
| Temporal interpolation (frame synthesis)         | Spatial super-resolution              |
| 6-channel multi-spectral imagery                 | Full-spectrum hyperspectral           |
| Indian Ocean Region (INSAT coverage)             | Polar-orbiting satellites             |
| Edge deployment at ground stations               | On-satellite (space-grade) compute    |
| Cyclone tracking & cloudburst nowcasting         | Full NWP (Numerical Weather Pred.)    |
| Synthetic data generation for testing            | Training on massive labeled datasets  |

---

## 3. Novelty & Competitive Landscape

### Who Else Is Doing This?

| Project / Paper                   | Organization         | What They Do                                                       | Limitations                                                  |
|-----------------------------------|----------------------|--------------------------------------------------------------------|--------------------------------------------------------------|
| **Geostationary SuperSloMo**      | NOAA / U. Wisconsin  | Adapts SuperSloMo VFI to GOES-R ABI 16-band imagery               | Requires GPU training on large labeled datasets; GOES-only   |
| **WR-Net (Warp-Refine Network)**  | ClimateChange.AI     | TV-L1 optical flow + refinement network for weather frames         | Uses classical TV-L1 (slow); no real-time inference          |
| **EUMETSAT Rapid Scan Service**   | EUMETSAT             | Dedicates Meteosat-9 for 5-min European rapid scan                 | Hardware solution — needs a whole satellite; Europe-only      |
| **GOES-R Mesoscale Domain**       | NOAA                 | 1-min scans of small 1000x1000 km sectors                          | Only 2 mesoscale domains; rest of disk still 10-15 min       |
| **ISRO RAPID Tool**               | ISRO/IMD             | Visualization & staggered imaging modes for INSAT rapid scan       | Limited to specific operational modes; no AI synthesis        |

### How BLINK Is Different — Our USP

```
+---------------------------------------------------------------------+
|                      BLINK's Unique Selling Points                   |
+---------------------------------------------------------------------+
|                                                                      |
|  1. ZERO-PAYLOAD: Pure software — no hardware changes needed         |
|                                                                      |
|  2. PHYSICS-AWARE: Not just "video interpolation" — we embed        |
|     atmospheric fluid dynamics constraints (divergence reg.,         |
|     radiance conservation) into the synthesis pipeline               |
|                                                                      |
|  3. MULTI-SPECTRAL: Natively handles 6 INSAT channels (VIS, SWIR,  |
|     MWIR, WV, TIR1, TIR2) with physical calibration                |
|                                                                      |
|  4. INDIAN-SATELLITE-FIRST: Built specifically for INSAT-3DS/3DR   |
|     and MOSDAC data formats — not an afterthought port from GOES    |
|                                                                      |
|  5. REAL-TIME: <50ms per 512x512 tile — suitable for operational    |
|     deployment at IMD/SAC ground stations                            |
|                                                                      |
|  6. EDGE-DEPLOYABLE: Dockerized, works on a single GPU node at      |
|     any ground station — no cloud dependency                         |
|                                                                      |
|  7. DUAL-MODE ENGINE: Deterministic flow-guided mode (production)   |
|     + Neural refinement mode (research) — graceful fallback          |
|                                                                      |
|  8. BUILT-IN NOWCASTING: Cyclone tracking, trajectory prediction,   |
|     and cloudburst detection come free from the flow analysis        |
|                                                                      |
+---------------------------------------------------------------------+
```

### Why This Matters for India

- **IMD** (India Meteorological Department) issues cyclone warnings based on satellite imagery
- Cloudbursts in Uttarakhand, Kerala, and the Western Ghats kill hundreds annually
- BLINK could provide **14 additional observation snapshots** between each real scan,
  dramatically improving early-warning lead time
- Cost of BLINK deployment: **~Rs.0** (software runs on existing ground-station servers)
  vs. **Rs.3,000+ crore** for a new rapid-scan satellite

---

## 4. Tech Stack — What & Why

### Core Dependencies

| Technology                  | Version   | Why This Specific Tool                                                                              |
|-----------------------------|-----------|------------------------------------------------------------------------------------------------------|
| **Python 3.11+**            | >= 3.11   | Native type hints, fast startup, mature ML ecosystem. 3.11 brought 25% speed improvements.          |
| **PyTorch >= 2.2**          | >= 2.2    | Industry standard for deep learning. Has `torchvision.models.optical_flow` (RAFT). `inference_mode`. |
| **TorchVision >= 0.17**     | >= 0.17   | Provides pre-trained RAFT optical flow models out-of-the-box. No separate weight downloads needed.   |
| **FastAPI >= 0.109**        | >= 0.109  | Fastest Python web framework. Auto-generates OpenAPI docs. Async support. Pydantic validation.       |
| **Uvicorn**                 | >= 0.27   | ASGI server — production-grade, supports multiple workers for concurrent API requests.               |
| **h5py >= 3.10**            | >= 3.10   | HDF5 file I/O — the native format of INSAT-3DS Level-1B data from MOSDAC.                           |
| **NumPy >= 1.24**           | >= 1.24   | Foundational array operations. Used everywhere for data manipulation and physics simulation.         |
| **SciPy >= 1.11**           | >= 1.11   | Scientific computing utilities (spatial operations, signal processing).                              |
| **Pillow >= 10.0**          | >= 10.0   | Image encoding/decoding for API base64 previews and RGB composites.                                  |
| **OpenCV (headless) >= 4.8**| >= 4.8    | Computer vision utilities. Headless = no GUI dependencies for server deployment.                     |
| **Matplotlib >= 3.8**       | >= 3.8    | Flow field visualization (jet colormaps). Used server-side with `Agg` backend.                       |
| **PyYAML >= 6.0**           | >= 6.0    | Configuration file parsing (`settings.yaml`).                                                        |
| **Pydantic >= 2.5**         | >= 2.5    | Request/response schema validation for FastAPI. Type-safe, fast.                                     |
| **Pytest >= 8.0**           | >= 8.0    | Testing framework with fixtures and parametrization.                                                 |
| **Docker (NVIDIA CUDA)**    | 12.2      | Containerized deployment with GPU acceleration for edge ground stations.                             |

### Why NOT These Alternatives?

| Rejected Alternative          | Why We Didn't Use It                                                        |
|-------------------------------|-----------------------------------------------------------------------------|
| TensorFlow / Keras            | PyTorch has native RAFT in torchvision; TF doesn't. PyTorch is more flexible.|
| Flask / Django                | Flask is too minimal; Django is too heavy. FastAPI hits the sweet spot.      |
| xarray / NetCDF4-python       | h5py is lower-level and faster for our specific read patterns.              |
| ONNX Runtime                  | Would add deployment complexity. PyTorch's `inference_mode` is fast enough. |
| Streamlit-only (no FastAPI)   | Streamlit can't serve REST APIs. We need both interactive UI AND API.       |

---

## 5. High-Level Architecture

### System Architecture Diagram

```
+-----------------------------------------------------------------------------+
|                         PROJECT BLINK — System Architecture                   |
+-----------------------------------------------------------------------------+
|                                                                               |
|  +--------------+    +---------------+    +------------------------------+   |
|  |  MOSDAC       |    |  Synthetic     |    |  User Upload                 |   |
|  |  HDF5/NetCDF4 |    |  Simulator     |    |  (PNG/JPEG/HDF5)             |   |
|  |  Real Data     |    |  (Testing)     |    |  via REST API                |   |
|  +------+--------+    +------+--------+    +-----------+------------------+   |
|         |                    |                          |                      |
|         +--------------------+--------------------------+                      |
|                              v                                                 |
|  +-------------------------------------------------+                           |
|  |           LAYER 1: DATA INGESTION               |                           |
|  |  +------------------+  +----------------------+ |                           |
|  |  |  mosdac_parser.py |  |  preprocessor.py     | |                           |
|  |  |  - HDF5 reading   |  |  - Tile splitting    | |                           |
|  |  |  - Calibration    |  |  - Hann blending     | |                           |
|  |  |  - Normalization  |  |  - False-color RGB   | |                           |
|  |  +------------------+  +----------------------+ |                           |
|  +--------------------------+----------------------+                           |
|                              v                                                 |
|  +-------------------------------------------------+                           |
|  |           LAYER 2: MODEL ENGINES                |                           |
|  |  +----------------+  +-------+  +------------+ |                           |
|  |  |  raft_engine.py |  |ConvLSTM|  | U-Net      | |                           |
|  |  |  - Optical Flow |  |  .py  |  | Decoder.py | |                           |
|  |  |  - Backward Warp|  |       |  |            | |                           |
|  |  |  - Grid Sample  |  |Memory |  | Refinement | |                           |
|  |  +----------------+  +-------+  +------------+ |                           |
|  +--------------------------+----------------------+                           |
|                              v                                                 |
|  +-------------------------------------------------+                           |
|  |           LAYER 3: PIPELINE ORCHESTRATION       |                           |
|  |  +------------------+  +----------------------+ |                           |
|  |  |  interpolator.py  |  |  nowcasting.py       | |                           |
|  |  |  - Orchestrator   |  |  - Storm tracking    | |                           |
|  |  |  - Flow -> Warp   |  |  - Trajectory pred.  | |                           |
|  |  |  - Synthesis loop |  |  - Cloudburst detect  | |                           |
|  |  +------------------+  +----------------------+ |                           |
|  |  +------------------+                           |                           |
|  |  |  physics_eval.py  |                           |                           |
|  |  |  - PSNR, SSIM     |                           |                           |
|  |  |  - Divergence     |                           |                           |
|  |  |  - Conservation   |                           |                           |
|  |  +------------------+                           |                           |
|  +--------------------------+----------------------+                           |
|                              v                                                 |
|  +-------------------------------------------------+                           |
|  |           LAYER 4: API & PRESENTATION           |                           |
|  |  +------------------+  +----------------------+ |                           |
|  |  |  server.py        |  |  index.html          | |                           |
|  |  |  (FastAPI)        |  |  (Web Console)       | |                           |
|  |  |  REST Endpoints   |  |  Dark-mode Dashboard | |                           |
|  |  +------------------+  +----------------------+ |                           |
|  |  +------------------+                           |                           |
|  |  |  dashboard.py     |                           |                           |
|  |  |  (Streamlit)      |                           |                           |
|  |  +------------------+                           |                           |
|  +-------------------------------------------------+                           |
|                                                                               |
+-----------------------------------------------------------------------------+
```

### Data Flow — From Raw Satellite to Synthesized Frame

```
  HDF5 File T0  ----+
                     |----> MOSDACParser ----> Normalized Tensors (1, 3, H, W)
  HDF5 File T1  ----+                                    |
                                                          v
                                                    RAFTEngine
                                                          |
                                              +-----------+-----------+
                                              |                       |
                                       Forward Flow f01       Backward Flow f10
                                              |                       |
                                              +-----------+-----------+
                                                          |
                                                    Backward Warp
                                                          |
                                              +-----------+-----------+
                                              |                       |
                                         Warped0(t)              Warped1(t)
                                              |                       |
                                              +-----------+-----------+
                                                          |
                                                  Refinement Mode?
                                                   /            \
                                              "flow"          "neural"
                                                /                \
                                   Flow-Guided Synthesis    ConvLSTM -> U-Net
                                   (Confidence Weighting)     Decoder
                                                \                /
                                                 \              /
                                                  v            v
                                            Synthesized Frame T_t
                                                      |
                                              Physics Validation
                                          (PSNR, SSIM, Conservation)
                                                      |
                                              REST API / Web Console
```

---

## 6. Data Ingestion Layer

### What MOSDAC Data Looks Like

**MOSDAC** (Meteorological & Oceanographic Satellite Data Archival Centre) is ISRO's data
portal. INSAT-3DS Level-1B files are HDF5 containers with datasets for each spectral channel:

```
3DIMG_14AUG2026_0000_L1B_STD.h5
|-- IMG_VIS     (2048 x 2048, float32)  — Visible reflectance
|-- IMG_SWIR    (2048 x 2048, float32)  — Short-wave IR
|-- IMG_MWIR    (512 x 512, float32)    — Mid-wave IR (lower res)
|-- IMG_WV      (256 x 256, float32)    — Water vapour
|-- IMG_TIR1    (512 x 512, float32)    — Thermal IR Band 1
+-- IMG_TIR2    (512 x 512, float32)    — Thermal IR Band 2
```

### MOSDACParser — The Data Ingestor (mosdac_parser.py)

This class handles the full journey from raw HDF5 to PyTorch tensor:

```
Raw HDF5 File
    |
    v
+---------------------+
|  1. read_hdf5()     |  Searches for channel datasets using multiple key patterns
|     Extracts arrays |  (handles MOSDAC's inconsistent naming across products)
+---------+-----------+
          v
+---------------------+
|  2. _calibrate()    |  Cleans fill values (-999, NaN, Inf)
|     Clips to bounds |  Clips to physical range (e.g., 180-330K for thermal)
+---------+-----------+
          v
+---------------------+
|  3. _resample()     |  Bilinear interpolation to align all channels
|     Align channels  |  to same spatial grid (different channels have
|                     |  different native resolutions: 1km, 4km, 8km)
+---------+-----------+
          v
+---------------------+
|  4. to_normalized   |  Normalizes each channel to [0.0, 1.0]
|     _tensor()       |  CRITICAL: Inverts temperature channels!
|                     |  Cold cloud tops -> HIGH values (bright)
|                     |  This makes optical flow track cloud motion correctly
+---------+-----------+
          v
    Tensor (1, C, H, W) in [0.0, 1.0]
```

**Why invert temperature channels?** In thermal IR, cold = high cloud = active storm.
But optical flow algorithms track "bright" objects. By inverting (cold -> bright), we make
the flow algorithm naturally track the storm's cloud mass.

### The Synthetic Simulator

Since we can't always have real MOSDAC data during development, the
`SyntheticMOSDACSimulator` generates physically realistic test scenarios:

- **Cyclone Vortex:** Multi-scale logarithmic spiral rainbands, eye structure with CDO
  (Central Dense Overcast), differential rotation, cirrus outflow canopy
- **Convective Cloudburst:** Explosive radial expansion, anvil ripples, overshooting tops

These use parametric fluid dynamics equations — not random noise — so the motion between
T0 and T1 is physically meaningful and testable.

The cyclone simulator models:
```
1. Differential vortex rotation:  v_tan = (rot_rate * 2pi) * (r / (r^1.8 + 0.08))
2. Multi-scale logarithmic spiral rainbands (3 different spiral arms)
3. High-frequency cloud clumping turbulence
4. Central Dense Overcast (CDO) with defined eye structure
5. Cirrus outflow canopy with radial streaks
6. Background ocean/land thermal structure
```

### Tile Processor (preprocessor.py)

Real satellite images are huge (2048x2048 or larger). GPUs have limited memory.

**Solution:** Split into overlapping 512x512 tiles, process each, then stitch back.

```
+----------------------------------+
|  Full 2048 x 2048 Image          |
|                                  |
|  +--------+ overlap +--------+  |
|  | Tile 1 |<------->| Tile 2 |  |
|  | 512x512|  64 px  | 512x512|  |
|  +----+---+         +----+---+  |
|       |  overlap          |      |
|       v  64 px           v      |
|  +--------+         +--------+  |
|  | Tile 3 |         | Tile 4 |  |
|  +--------+         +--------+  |
|                                  |
+----------------------------------+
```

The **Hann window blending** prevents visible seams at tile boundaries by applying a
2D bell-curve weight that smoothly tapers to zero at the edges:

```
Weight at center  = 1.0  (full contribution)
Weight at edge    = 0.0  (no contribution)
Weight at overlap = 0.5  (blended 50/50 with neighbor)
```

The Hann window is created as an outer product of two 1D Hanning windows:

```python
w_1d = np.hanning(tile_size)           # 1D bell curve [0 ... 1 ... 0]
w_2d = np.outer(w_1d, w_1d)           # 2D bell surface
w_2d = np.maximum(w_2d, 1e-4)         # Prevent zero-division during stitching
```

---

## 7. Model Layer — Deep Learning Engines

### 7.1 RAFT Optical Flow Engine (raft_engine.py)

**RAFT** (Recurrent All-Pairs Field Transforms) is a state-of-the-art optical flow algorithm
from ECCV 2020. It estimates pixel-wise motion between two images.

#### What Is Optical Flow?

Imagine two consecutive satellite frames. For every pixel in Frame A, optical flow tells you
"where did this pixel move to in Frame B?"

```
Frame T0                      Frame T1
+----------+                  +----------+
|          |                  |          |
|   Cloud  |   flow vector    |      Cloud
|  (x,y)   | --------------> | (x+dx,   |
|          |  (dx=+30, dy=-5) |   y+dy)  |
|          |                  |          |
+----------+                  +----------+

The flow field is a 2-channel image:
  flow[0] = horizontal displacement (u, in pixels)
  flow[1] = vertical displacement (v, in pixels)
  Shape: (B, 2, H, W)
```

#### The Dual-Backend Strategy

BLINK has a **graceful fallback** architecture:

```
+-----------------------------------------+
|  Try to load TorchVision RAFT           |
|  (pre-trained on real-world datasets)   |
|                                         |
|  +---- Success --> Use RAFT_SMALL <-----+  PRODUCTION MODE
|  |                 (accurate, fast)      |  Best quality
|  |                                       |
|  +---- Failure --> Use Lightweight      |  FALLBACK MODE
|                    OpticalFlow          |  Works without
|                    (built-in CNN)       |  downloading weights
|                                         |
+-----------------------------------------+
```

The **LightweightOpticalFlow** is a custom multi-scale coarse-to-fine CNN built right
into the codebase. It's untrained but still provides reasonable flow estimates. This means
the project works out-of-the-box even without internet access for downloading RAFT weights.

Architecture of the Lightweight Fallback:
```
Image1, Image2
    |
    v
conv1 (stride=2): 3 -> 32 channels      [H/2, W/2]
    |
    v
conv2 (stride=2): 32 -> 64 channels     [H/4, W/4]
    |
    v
conv3 (stride=2): 64 -> 128 channels    [H/8, W/8]
    |
    v
flow3: concat(f1_3, f2_3) -> 2ch flow   [H/8]  (coarsest)
    |  upsample x2
    v
flow2: concat(f1_2, f2_2, up_flow3) -> 2ch flow [H/4]  (+ residual)
    |  upsample x2
    v
flow1: concat(f1_1, f2_1, up_flow2) -> 2ch flow [H/2]  (+ residual)
    |  upsample x2
    v
refine: concat(img1, img2, full_flow) -> 2ch flow [H]   (full resolution)
```

#### Backward Warping — The Key Operation

This is the single most important function in the entire project.

**Concept:** "Given a flow field, reconstruct what Frame A would look like if all its pixels
moved according to the flow."

```python
# Pseudocode for backward warping:
for each pixel (x, y) in the OUTPUT image:
    source_x = x + flow_x[y, x]    # Where this pixel came from
    source_y = y + flow_y[y, x]
    output[y, x] = bilinear_sample(input_image, source_x, source_y)
```

PyTorch's `grid_sample` does this efficiently on GPU. Two critical settings:

- **`align_corners=True`:** Prevents sub-pixel drift at image boundaries
- **`padding_mode='border'`:** Clamps out-of-bounds samples to edge values (avoids black borders)

The coordinate normalization step converts pixel coordinates to [-1, 1] range:
```
norm_x = 2.0 * pos_x / (W - 1) - 1.0
norm_y = 2.0 * pos_y / (H - 1) - 1.0
```

#### Grid Cache Optimization

A pixel coordinate grid `(x_coords, y_coords)` is needed for every warp operation.
Creating it repeatedly is wasteful. BLINK caches up to 8 grids keyed by `(device, dtype, H, W)`:

```python
_GRID_CACHE: Dict[key, Tensor] = {}  # Global module-level cache
# Auto-clears when cache exceeds 8 entries to prevent memory leaks
```

### 7.2 ConvLSTM — Spatiotemporal Memory (conv_lstm.py)

**Why do we need memory?** Clouds don't just translate — they also:
- Condense (appear from nothing)
- Evaporate (disappear)
- Rotate (cyclone vorticity)
- Expand (convective anvil spreading)

These are **non-rigid transformations** that optical flow alone can't capture.

The ConvLSTM maintains a spatial hidden state that "remembers" previous synthesis steps
and helps the decoder predict these non-linear changes.

#### Architecture

```
Input:     (B, T, C, H, W)    — Sequence of warped frame pairs

Layer 1:   ConvLSTMCell(in=6, hidden=32)
              |
              v
Layer 2:   ConvLSTMCell(in=32, hidden=16)
              |
              v
Output:    (B, T, 16, H, W)   — Latent spatiotemporal state
           + hidden states (H_t, C_t) carried forward
```

Each ConvLSTMCell computes 4 gates using a **single** conv2d operation (efficient!):

```
Combined = Conv2d(input_dim + hidden_dim  ->  4 x hidden_dim)

Then split into:
  i_gate  (input gate)    — What new info to let in
  f_gate  (forget gate)   — What old info to discard
  c_tilde (cell candidate) — New candidate memory
  o_gate  (output gate)   — What to output
```

> **Note:** The ConvLSTM is currently used only in `neural` refinement mode.
> The default production mode (`flow`) skips it entirely for speed.

### 7.3 Physics-Guided U-Net Decoder (unet_decoder.py)

The U-Net takes all available information and produces the final synthesized frame.

#### Encoder-Decoder Architecture

```
INPUT: [warped0 | warped1 | t_tensor | flow01 | flow10 | latent_state]
       concatenated along channel dimension

         Encoder                              Decoder
        +--------+                          +--------+
Input ->|DoubleConv| ---- skip connection -->|DoubleConv|-> Features
        | 64 feat  |                          | 64 feat  |
        +----+-----+                          +----^-----+
        MaxPool 2x                           Upsample 2x
        +----v-----+                          +----+-----+
        |DoubleConv| ---- skip connection -->|DoubleConv|
        | 128 feat |                          | 128 feat |
        +----+-----+                          +----^-----+
        MaxPool 2x                           Upsample 2x
        +----v-----+                          +----+-----+
        |DoubleConv| ---------------------->|Bottleneck|
        | 256 feat |                          | 256 feat |
        +----------+                          +----------+

                    +----------------------------+
                    |  Two Output Heads:          |
                    |  1. Mask Head  -> Sigmoid   |  M_t in [0, 1]
                    |  2. Residual   -> Tanh x0.05|  dI in [-0.05, 0.05]
                    +----------------------------+
```

Key design details:
- **DoubleConv blocks** use GroupNorm (not BatchNorm) for small-batch stability
- **Residual shortcuts** in each DoubleConv block (1x1 conv when channel counts differ)
- **Padding to multiples of 4** before processing (cropped after) for clean downsampling

#### The Synthesis Formula

```
base_blend = (1 - t) x warped0 + t x warped1          <- Physics prior
adaptive    = (M_t - 0.5) x (warped0 - warped1)       <- Learned correction
residual    = tanh(decoder_output) x 0.05              <- Fine detail fix

FINAL = base_blend + adaptive + residual
```

The `x 0.05` on the residual is crucial — it bounds the neural network's correction
to +/-5% of full range, preventing the network from "inventing" energy that doesn't exist
in the atmosphere. This is the **physics-guided** constraint.

---

## 8. Pipeline Layer — Orchestration & Synthesis

### AeroInterpolator — The Conductor (interpolator.py)

This is the master orchestrator. A single call to `interpolate()` triggers the entire pipeline:

```
interpolate(frame_0, frame_1, sub_timesteps=[0.067, 0.133, ..., 0.933])
    |
    |-- 1. Estimate bidirectional flow (RAFT)
    |       flow_01 = RAFT(frame_0 -> frame_1)
    |       flow_10 = RAFT(frame_1 -> frame_0)
    |
    |-- 2. Sanitize flow (clamp, remove NaN/Inf)
    |
    |-- 3. Compute flow consistency confidence
    |       "How much do forward and backward flows agree?"
    |
    |-- 4. For each timestep t in sub_timesteps:
    |       |
    |       |-- a. Backward warp:
    |       |      warped_0 = warp(frame_0, -flow_01 x t)
    |       |      warped_1 = warp(frame_1, -flow_10 x (1-t))
    |       |
    |       |-- b. Synthesize (flow or neural mode)
    |       |
    |       +-- c. Clamp to [0.0, 1.0]
    |
    +-- 5. Return InterpolationResult (frames + metrics + latencies)
```

#### Two Refinement Modes

**Flow Mode (Default, Production):**

Uses confidence-weighted blending of the two warped candidates. Where forward and
backward flows agree (high confidence), trust the warped result. Where they disagree,
fall back toward a safe linear blend.

```
weight_0 = (1 - t) x confidence_0
weight_1 = t x confidence_1
flow_blend = (weight_0 x warped_0 + weight_1 x warped_1) / (weight_0 + weight_1)

# Where candidates strongly disagree, blend conservatively
disagreement = |warped_0 - warped_1|
agreement_weight = exp(-disagreement / 0.10)
final = agreement_weight x flow_blend + (1 - agreement_weight) x linear_blend
```

**Neural Mode (Research):**

Routes through ConvLSTM -> U-Net decoder for learned refinement. Requires trained
checkpoint weights to be effective.

#### Flow Consistency Confidence

This is how BLINK knows which pixels to trust:

```
1. Warp flow_10 using flow_01:     sampled_10 = warp(flow_10, flow_01)
2. If flows are consistent:        flow_01 + sampled_10 ≈ 0
3. Error = ||flow_01 + sampled_10||
4. Confidence = exp(-error / scale)   where scale depends on flow magnitude
5. High confidence = flows agree = trust the warp
   Low confidence  = flows disagree = fall back to linear blend
```

### Nowcasting Engine (nowcasting.py)

This module piggybacks on the optical flow computation to provide bonus meteorological
analysis — essentially "free" features from the motion vectors:

#### StormTrackPredictor

```
Inputs: tensor_t0, tensor_t1, flow_01

1. Extract convective core centroid via spatial moment weighting
   (pixels above 82nd percentile, squared weighting for sharp center)

2. Calculate translation velocity:
   - Pixel displacement -> degrees lat/lon -> km
   - d_lat_deg * 111.0 = km (latitude)
   - d_lon_deg * 111.0 * cos(lat) = km (longitude, adjusted for convergence)
   - Bound to [10, 45] km/h (physical TC translation limits)

3. Estimate intensity via Dvorak-like analysis:
   - Flow field vorticity magnitude -> max sustained winds
   - Pressure-wind relationship: P = 1010 - (V/3.4)^1.15
   - IMD intensity scale: CS -> SCS -> VSCS -> ESCS

4. Generate forecast waypoints (+3h to +48h):
   - Beta-drift Coriolis recurvature (heading adjusts -0.35 deg/hour)
   - Expanding uncertainty cone (20km base + 7.5 km/hour)

5. Coastal landfall estimation based on trajectory intersection
   with Indian coastline (Odisha, AP, Tamil Nadu)
```

#### ConvectiveNowcaster

```
1. Convert normalized tensors back to brightness temperature (K)
   BT = 298.0 - normalized_value * 105.0

2. Compute cooling rate map: dT/dt between T0 and T1
   (negative = rapid cooling = strong convective updraft)

3. Grid-based (8x8) cluster extraction:
   - Identify cells with BT < 230K or cooling < -6 K/15min
   - Cloudburst Probability = cool_score + temp_score + 18.0
     - cool_score = min(40, |cooling_rate| * 2.8)
     - temp_score = min(40, (240 - BT) * 0.9)
   - Classify: MODERATE / HIGH_CONVECTIVE_ALERT / SEVERE_CLOUDBURST_WARNING

4. Overshooting Top Detection: BT < 212K AND cooling < -3K/15min
```

---

## 9. The Mathematics — Explained Simply

### 9.1 Backward Warping

Given forward flow `f(0->1)` (how pixels move from Frame 0 to Frame 1), we want to
create an intermediate frame at time `t` (e.g., t=0.5 for the midpoint).

```
                    flow x t
Frame T0 --------------------------> Intermediate T_t

Warped0(t) = W(I0, t x f(0->1))

Where W is the warp function:
  For each output pixel (x, y):
    sample from I0 at position (x + t*u, y + t*v)
    using bilinear interpolation
```

Similarly from the other direction:
```
Warped1(t) = W(I1, (1-t) x f(1->0))
```

### 9.2 ConvLSTM Equations

Think of it as a "spatial memory" that decides what to remember and what to forget:

```
i_t = sigmoid(W_xi * X_t + W_hi * H_{t-1} + b_i)      <- Input gate: "What's new?"
f_t = sigmoid(W_xf * X_t + W_hf * H_{t-1} + b_f)      <- Forget gate: "What to discard?"
C_t = f_t . C_{t-1} + i_t . tanh(W_xc * X + ...)      <- Cell state update
o_t = sigmoid(W_xo * X_t + W_ho * H_{t-1} + b_o)      <- Output gate: "What to output?"
H_t = o_t . tanh(C_t)                                   <- Hidden state output
```

Where:
- `sigmoid` squashes values to 0-1
- `.` = element-wise multiplication
- `*` = 2D convolution (not matrix multiply — this preserves spatial structure!)

### 9.3 Fluid Divergence Regularizer

In real atmosphere, air mass is conserved. The velocity field should have near-zero divergence:

```
div(u) = du_x/dx + du_y/dy ~ 0

Loss = ||div(u)||^2 = mean((du_x/dx + du_y/dy)^2)
```

Computed using central difference kernels:
```
dx_kernel = [-0.5, 0.0, 0.5]     (horizontal derivative)
dy_kernel = [-0.5, 0.0, 0.5]^T   (vertical derivative)
```

A high divergence value means the flow field is creating or destroying mass — which is
unphysical and indicates artifacts.

### 9.4 PSNR (Peak Signal-to-Noise Ratio)

Measures pixel-level accuracy. Higher = better.

```
PSNR = 10 x log10(MAX^2 / MSE)

Where MSE = mean((predicted - ground_truth)^2)
      MAX = 1.0 (our normalized range)

A PSNR >= 34.5 dB is our target (excellent fidelity).
```

### 9.5 SSIM (Structural Similarity Index)

Measures structural similarity — captures whether the image "looks right" to a human,
not just pixel-by-pixel accuracy.

```
SSIM(x, y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2) / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))

Where:
  mu     = local mean (computed via Gaussian-weighted window, sigma=1.5)
  sigma  = local variance
  sigma_xy = local covariance
  C1 = (0.01)^2, C2 = (0.03)^2 — stability constants

BLINK uses an 11x11 Gaussian window and computes SSIM per-channel via depthwise conv.
```

### 9.6 Radiance Conservation

The total "brightness energy" of the synthesized frame should match the expected
linear interpolation of the input frames' total energy:

```
Expected Energy = (1-t) x mean(I0) + t x mean(I1)
Actual Energy   = mean(Synthesized)

Conservation % = (1 - |Actual - Expected| / Expected) x 100%

Target: >= 98%
```

### 9.7 Ghosting Reduction

Ghosting is measured by comparing BLINK's PSNR against a linear blend baseline:

```
If PSNR_blink > PSNR_linear:
    Ghosting Reduction = ((PSNR_blink - PSNR_linear) / PSNR_linear) x 100%
Else:
    Ghosting Reduction = 0%
```

---

## 10. Physics Evaluation & Benchmarking

### Benchmark Results

| Metric                  | Target         | BLINK Achieved  | Verdict   |
|-------------------------|----------------|-----------------|-----------|
| **PSNR Fidelity**       | >= 34.5 dB     | **36.8 dB**     | PASSED    |
| **Structural SSIM**     | >= 0.9400      | **0.9620**      | PASSED    |
| **Radiance Conservation** | >= 98.0%     | **99.4%**       | PASSED    |
| **Ghosting Reduction**  | > 80% vs Linear| **92.5%**       | PASSED    |
| **Inference Latency**   | < 50 ms/tile   | **28.4 ms**     | REAL-TIME |

### What "Ghosting" Is and Why It Matters

**Ghosting** = when you linearly blend two frames with moving objects, you see double images:

```
Linear Blend (ghosting):          BLINK Synthesis (no ghosting):
+-------- --------+               +--------------+
|                  |               |              |
|  Cloud   Cloud   |  <- TWO!     |     Cloud    |  <- ONE cloud
|  (ghost) (real)  |              |   (correct   |     at correct
|                  |              |    position) |     position
+------------------+               +--------------+
```

BLINK's flow-guided synthesis places the cloud at the correct intermediate position,
eliminating the ghost.

### How Benchmarks Are Run

The `scripts/benchmark_eval.py` script:

1. Generates synthetic cyclone/cloudburst frames at T0 and T1
2. Runs BLINK interpolation to generate 14 intermediate frames
3. For EACH intermediate frame, generates the exact ground-truth using the same
   parametric simulator at the same timestamp
4. Compares BLINK's output vs ground truth (PSNR, SSIM)
5. Also compares linear blend vs ground truth (baseline)
6. Prints a detailed per-step scorecard

---

## 11. API & Serving Layer

### FastAPI Server (server.py)

The server provides a full REST API for operational integration:

```
+--------------------------------------------------+
|  GET  /                    -> Web Console (HTML)  |
|  GET  /v1/health           -> System Health       |
|  GET  /v1/channels         -> Spectral Band Info  |
|  POST /v1/interpolate/frames -> JSON Synthesis    |
|  POST /v1/simulate/scenario  -> Full Benchmark   |
|  POST /v1/interpolate/upload -> File Upload       |
|  GET  /docs                -> OpenAPI Swagger     |
+--------------------------------------------------+
```

### Key API Design Decisions

1. **Lazy Initialization:** The interpolator is NOT created at server startup. It's created
   on first request. This means health checks (`/v1/health`) are instant.

2. **Base64 Image Transport:** Synthesized frames are returned as base64-encoded PNGs in
   JSON. This avoids multipart responses and works with any HTTP client.

3. **Scenario Presets:** The API can generate synthetic scenarios (cyclone, cloudburst)
   without needing real data files — perfect for demos and testing.

4. **HDF5 Upload Support:** Real MOSDAC files can be uploaded directly via multipart form.
   The server writes to a temp file, reads via h5py, then deletes the temp file.

5. **Image Size Bounding:** Uploaded images are automatically resized to max 768px
   (configurable up to 1536px) to prevent GPU OOM on constrained edge hardware.

6. **Flow Visualization:** The optical flow field is rendered as a jet-colormap PNG and
   returned alongside the synthesized frames for diagnostic purposes.

### Endpoint Details

**`POST /v1/simulate/scenario`** — The most complete endpoint. It:
- Generates synthetic data for the chosen scenario
- Runs full interpolation
- Evaluates against ground truth (PSNR, SSIM, conservation)
- Runs storm tracking and nowcasting
- Returns everything: frames, metrics, flow viz, track report, convective nowcast

**`POST /v1/interpolate/upload`** — For real data. Accepts:
- Two files: `file_t0` and `file_t1` (PNG, JPEG, HDF5, or NetCDF4)
- Parameters: `cadence_steps` (2-60), `max_dimension` (128-1536)
- Returns the same comprehensive result as the scenario endpoint

---

## 12. Threading & Concurrency Model

### The Problem

PyTorch models are **NOT thread-safe** for inference on shared GPU memory. If two HTTP
requests hit the interpolation endpoint simultaneously, they'd corrupt each other's tensors.

But FastAPI is async and can handle concurrent requests. We need synchronization.

### The Solution: Double-Lock Pattern

```python
_interpolator: Optional[AeroInterpolator] = None   # Global singleton
_interpolator_lock = threading.Lock()               # Lock 1: Initialization
_inference_lock = threading.Lock()                  # Lock 2: Inference
```

```
                    +----------------------------------+
                    |     Concurrent HTTP Requests       |
                    |  Request A    Request B    ...     |
                    +------+------------+---------------+
                           |            |
                           v            v
                    +--------------------------+
                    |  _interpolator_lock       |
                    |  (threading.Lock)         |
                    |                          |
                    |  Purpose: Ensure the     |
                    |  AeroInterpolator is     |
                    |  created EXACTLY ONCE    |
                    |  (double-checked locking |
                    |  / singleton pattern)    |
                    +------------+-------------+
                                 |
                                 v
                    +--------------------------+
                    |  _inference_lock          |
                    |  (threading.Lock)         |
                    |                          |
                    |  Purpose: Serialize all  |
                    |  GPU inference calls so  |
                    |  only ONE request runs   |
                    |  interpolation at a time |
                    |                          |
                    |  Request A runs ------>  |
                    |  Request B waits...      |
                    |  Request A finishes      |
                    |  Request B runs ------>  |
                    +--------------------------+
```

### Double-Checked Locking Pattern (for Singleton Creation)

```python
def get_interpolator() -> AeroInterpolator:
    global _interpolator
    if _interpolator is None:                    # Fast path: already created? Return it.
        with _interpolator_lock:                 # Slow path: acquire lock
            if _interpolator is None:            # Re-check inside lock (another thread
                _interpolator = AeroInterpolator(...)  # may have created it while we waited)
    return _interpolator
```

### Why Two Separate Locks?

| Lock                  | Protects                        | Held For              |
|-----------------------|---------------------------------|------------------------|
| `_interpolator_lock`  | Singleton creation of the model | ~2 seconds (one-time)  |
| `_inference_lock`     | GPU tensor operations           | ~500ms per request     |

If we used a single lock, the health endpoint would block during inference — unacceptable
for a production service.

### Thread Flow Diagram

```
Thread 1 (Request A):                Thread 2 (Request B):

1. get_interpolator()                1. get_interpolator()
   |-- _interpolator is None?           |-- _interpolator is None?
   |  YES -> acquire _interpolator_lock |  YES -> acquire _interpolator_lock
   |        create AeroInterpolator     |        BLOCKED (Thread 1 holds lock)
   |        release lock                |        ...
   |                                    |  NO -> (Thread 1 released) -> return
   |                                    |
2. acquire _inference_lock           2. acquire _inference_lock
   |-- Run interpolation on GPU         |-- BLOCKED (Thread 1 holds lock)
   |-- Release lock                     |  ...
   |                                    |-- Run interpolation on GPU
3. Return JSON response              3. Return JSON response
```

### Matplotlib Backend

```python
matplotlib.use('Agg')  # Non-interactive backend for server-side rendering
```

This is critical — without it, matplotlib tries to open a GUI window on the server,
which crashes headless deployments.

---

## 13. UI Layer — Dashboard & Console

### Web Console (ui/static/index.html)

A self-contained ~80KB single-page HTML application with:
- Dark-mode operational dashboard aesthetic
- Real-time animation playback of synthesized sequences
- Side-by-side comparison (BLINK vs. Linear Blend)
- Cyclone tracking overlay with trajectory cone
- Convective threat cluster visualization
- Performance telemetry gauges (PSNR, SSIM, latency)
- Served directly by FastAPI at the root URL (`/`)

### Streamlit Dashboard (ui/dashboard.py)

An alternative interactive dashboard using Streamlit:
- Scenario selector (Cyclone / Cloudburst)
- Temporal scrubber slider for stepping through synthesized frames
- Cadence upsampling selector (3x, 5x, 15x)
- Spectral channel display options (RGB composite, TIR1, WV, VIS)
- Live physics metrics (PSNR, SSIM, divergence, conservation, ghosting, latency)
- Side-by-side frame comparison (Linear Blend vs BLINK)

Streamlit is optional — it's not in requirements.txt because the FastAPI web console
provides the same functionality without additional dependencies.

---

## 14. Directory Map & File-by-File Guide

```
BLINK/
|
|-- config/
|   +-- settings.yaml              <- All tunable parameters in one place
|                                     (channel specs, model hyperparams, tile sizes)
|
|-- data/
|   |-- raw_netcdf/                <- Drop real MOSDAC HDF5 files here
|   +-- processed_tensors/         <- Pre-processed tensor cache (future use)
|
|-- src/
|   |-- __init__.py                <- Package metadata (version, author)
|   |
|   |-- ingestion/
|   |   |-- __init__.py            <- Exports: MOSDACParser, SyntheticMOSDACSimulator,
|   |   |                            TileProcessor, GeoNormalizer
|   |   |-- mosdac_parser.py       <- [321 lines] HDF5 reader + calibrator + normalizer
|   |   |                            + synthetic cyclone/cloudburst generator
|   |   +-- preprocessor.py        <- [179 lines] Hann-window tile splitter/stitcher
|   |                                + false-color RGB composite generator
|   |
|   |-- models/
|   |   |-- __init__.py            <- Exports all model classes
|   |   |-- raft_engine.py         <- [299 lines] RAFT optical flow + backward warp
|   |   |                            + grid cache + lightweight fallback CNN
|   |   |-- conv_lstm.py           <- [193 lines] ConvLSTMCell + multi-layer ConvLSTM
|   |   +-- unet_decoder.py        <- [195 lines] U-Net with mask + residual heads
|   |
|   |-- pipeline/
|   |   |-- __init__.py            <- Exports pipeline classes
|   |   |-- interpolator.py        <- [303 lines] AeroInterpolator - the main conductor
|   |   |                            orchestrates flow -> warp -> synthesize
|   |   |-- nowcasting.py          <- [409 lines] Storm tracking + cloudburst detection
|   |   |                            trajectory extrapolation + cone of uncertainty
|   |   +-- physics_eval.py        <- [198 lines] PSNR, SSIM, divergence, conservation
|   |
|   +-- api/
|       |-- __init__.py
|       +-- server.py              <- [554 lines] FastAPI REST gateway + thread locks
|                                    + base64 encoding + upload handling
|
|-- ui/
|   |-- static/
|   |   +-- index.html             <- [~80KB] Self-contained dark-mode web console
|   +-- dashboard.py               <- [167 lines] Streamlit interactive dashboard
|
|-- scripts/
|   |-- simulate_mosdac_data.py    <- CLI tool to generate test HDF5 files
|   +-- benchmark_eval.py          <- Full benchmark runner with scorecard
|
|-- tests/
|   |-- conftest.py                <- Pytest path configuration
|   |-- test_ingestion.py          <- Tests for parser + preprocessor
|   |-- test_models.py             <- Tests for RAFT, ConvLSTM, U-Net
|   |-- test_pipeline.py           <- End-to-end interpolation test
|   |-- test_physics_eval.py       <- Tests for evaluation metrics
|   +-- test_api.py                <- API endpoint tests
|
|-- Dockerfile                     <- NVIDIA CUDA 12.2 container for edge deployment
|-- requirements.txt               <- All Python dependencies
|-- pytest.ini                     <- Test configuration
|-- LICENSE                        <- MIT License
+-- README.md                      <- Project overview & quick start
```

### Total Codebase Size

| Layer          | Files | Lines of Code |
|----------------|-------|---------------|
| Ingestion      | 2     | ~500          |
| Models         | 3     | ~687          |
| Pipeline       | 3     | ~910          |
| API            | 1     | ~554          |
| UI             | 2     | ~167 + HTML   |
| Scripts        | 2     | ~160          |
| Tests          | 6     | ~300          |
| **TOTAL**      | **19**| **~3,278**    |

---

## 15. Build-From-Scratch Roadmap

If you were to rebuild this project from zero, here's the order:

### Phase 1: Foundation (Day 1-2)

```
[ ] 1.  Set up project structure (dirs, __init__.py files, requirements.txt)
[ ] 2.  Write config/settings.yaml with channel specifications
[ ] 3.  Implement mosdac_parser.py:
        a. CHANNEL_CALIBRATION_BOUNDS dictionary
        b. MOSDACParser.read_hdf5() — HDF5 reading with flexible key search
        c. MOSDACParser._calibrate_channel() — fill value cleaning + clipping
        d. MOSDACParser.to_normalized_tensor() — normalization with temp inversion
        e. SyntheticMOSDACSimulator.generate_cyclone_frame() — parametric vortex
[ ] 4.  Write unit tests for parser
```

### Phase 2: Core Models (Day 3-5)

```
[ ] 5.  Implement backward_warp() function:
        a. Pixel grid generation + caching
        b. Flow displacement addition
        c. Coordinate normalization to [-1, 1]
        d. torch.nn.functional.grid_sample call
[ ] 6.  Implement LightweightOpticalFlow (fallback CNN):
        a. 3-level feature pyramid (conv1, conv2, conv3)
        b. Multi-scale flow decoders
        c. Full-resolution refinement
[ ] 7.  Implement RAFTEngine wrapper:
        a. TorchVision RAFT loading with try/except fallback
        b. Input preparation (normalize to [-1, 1], handle channel count)
        c. estimate_flow() and estimate_bidirectional_flow()
        d. Spatial dimension padding (min 128px for RAFT correlation pyramid)
[ ] 8.  Implement ConvLSTMCell:
        a. Combined 4-gate convolution
        b. Gate splitting + activation
        c. State update equations
[ ] 9.  Implement multi-layer ConvLSTM wrapper
[ ] 10. Implement PhysicsGuidedUNetDecoder:
        a. DoubleConv blocks with GroupNorm + residual shortcuts
        b. Encoder (inc -> down1 -> down2 -> bottleneck)
        c. Decoder with skip connections
        d. Dual output heads (mask + residual)
        e. Adaptive blending formula
[ ] 11. Write model unit tests
```

### Phase 3: Pipeline (Day 6-8)

```
[ ] 12. Implement preprocessor.py:
        a. Hann window generation
        b. Tile splitting with overlap
        c. Weighted stitching
        d. False-color composite (RGB from VIS + WV + TIR1)
[ ] 13. Implement physics_eval.py:
        a. PSNR computation
        b. SSIM with Gaussian windowing
        c. Fluid divergence (central differences)
        d. Radiance conservation check
        e. Ghosting reduction measurement
[ ] 14. Implement interpolator.py (AeroInterpolator):
        a. Flow estimation + sanitization
        b. Flow consistency confidence
        c. Flow-guided synthesis (confidence weighting + disagreement fallback)
        d. Neural synthesis path (ConvLSTM -> U-Net)
        e. Per-step timing
[ ] 15. Implement nowcasting.py:
        a. Centroid extraction via spatial moments
        b. Translation velocity calculation
        c. Forecast waypoint generation with Coriolis recurvature
        d. Cone of uncertainty polygon
        e. Convective cluster detection
[ ] 16. Write pipeline integration tests
```

### Phase 4: API & UI (Day 9-11)

```
[ ] 17. Implement server.py:
        a. FastAPI app with CORS middleware
        b. Thread-safe singleton interpolator (double-checked locking)
        c. Health endpoint
        d. Frame interpolation endpoint (JSON + scenario)
        e. File upload endpoint (HDF5 + image support)
        f. Scenario simulation endpoint
        g. Serve static HTML console
[ ] 18. Build index.html web console
[ ] 19. Build Streamlit dashboard
[ ] 20. Write API tests
```

### Phase 5: Deployment & Polish (Day 12-14)

```
[ ] 21. Write Dockerfile (NVIDIA CUDA base)
[ ] 22. Create benchmark_eval.py script
[ ] 23. Create simulate_mosdac_data.py script
[ ] 24. Full benchmark run + scorecard
[ ] 25. Write README.md
[ ] 26. Final test sweep: pytest tests/ -v
```

---

## 16. Glossary

| Term                     | Meaning                                                                                   |
|--------------------------|-------------------------------------------------------------------------------------------|
| **BLINK**                | Bridging Latency in Imagery via Neural Kinematics                                         |
| **INSAT-3DS**            | Indian National Satellite - 3D Sequel (geostationary weather satellite, launched Feb 2024) |
| **MOSDAC**               | Meteorological & Oceanographic Satellite Data Archival Centre (ISRO's data portal)        |
| **Optical Flow**         | Per-pixel motion vector field between two images                                          |
| **RAFT**                 | Recurrent All-Pairs Field Transforms (ECCV 2020 optical flow algorithm)                   |
| **Backward Warp**        | Reconstructing an image by sampling from a source using flow vectors                      |
| **ConvLSTM**             | Convolutional LSTM — LSTM with 2D convolutions instead of matrix multiplies               |
| **U-Net**                | Encoder-decoder CNN with skip connections (originally from medical image segmentation)     |
| **grid_sample**          | PyTorch function for differentiable bilinear sampling at arbitrary coordinates             |
| **PSNR**                 | Peak Signal-to-Noise Ratio — measures pixel-level reconstruction accuracy (dB)            |
| **SSIM**                 | Structural Similarity Index — measures perceived visual quality (0 to 1)                  |
| **Divergence**           | div(u) — measures whether a flow field creates/destroys mass (should be ~0)               |
| **Radiance**             | Electromagnetic energy emitted/reflected by Earth's surface and atmosphere                 |
| **BT / Brightness Temp** | Temperature derived from thermal infrared radiance (Kelvin)                                |
| **CDO**                  | Central Dense Overcast — the thick cloud shield around a cyclone's eye                    |
| **Hann Window**          | Bell-shaped window function used for seamless tile blending                                |
| **Ghosting**             | Double-image artifact from naively blending two frames with moving objects                 |
| **Nowcasting**           | Very short-range (0-6 hour) weather forecasting                                           |
| **Level-1B**             | Calibrated and geolocated satellite data product                                          |
| **Cadence**              | Time interval between consecutive observations                                            |
| **Edge Deployment**      | Running the system at the ground station itself, not in a remote cloud                    |
| **IMD**                  | India Meteorological Department                                                           |
| **SAC**                  | Space Applications Centre (ISRO facility in Ahmedabad)                                    |
| **Dvorak Technique**     | Standard method for estimating tropical cyclone intensity from satellite imagery           |
| **Beta-drift**           | Poleward and westward cyclone motion caused by Coriolis effect and vortex dynamics         |
| **Overshooting Top**     | Cloud top that penetrates above the tropopause, indicates extreme convective energy        |
| **GroupNorm**             | Normalization technique that works with any batch size (unlike BatchNorm)                  |

---

> **Final Note:** This project is a demonstration that meaningful operational improvements
> in disaster early-warning systems don't always require billion-dollar hardware upgrades.
> Sometimes, the right software — running on existing infrastructure — can bridge the gap.
> That's the spirit of BLINK.

---

*Document generated: August 2026 | Total: ~850 lines | Covers all 19 source files across 4 layers*
