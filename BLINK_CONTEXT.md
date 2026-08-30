# BLINK (Aero-Interpolate): Complete Reverse-Engineered Architectural Blueprint & Technical Reference Manual
**Sub-Title:** *Bridging Latency in Imagery via Neural Kinematics: Zero-Payload Rapid Scanning Engine for Geostationary Earth Observation*  
**Target Spacecraft Platforms:** INSAT-3DS / INSAT-3DR (ISRO / MOSDAC Level-1B & Level-2 Imager Radiance)  
**Document Classification:** Deep Reverse-Engineering, Mathematical Specification, and Exhaustive System Context  
**Target File:** `BLINK_CONTEXT.md`  
---
## Table of Contents
1. [Executive Story & The Three Levels of Understanding](#1-executive-story--the-three-levels-of-understanding)
   - 1.1 The Narrative of a Satellite Looking at Earth
   - 1.2 Level 1: Explain Like I'm 10 (Zero-Jargon Physical Analogies)
   - 1.3 Level 2: Explain Like I'm a Computer Science Student (Tensors, Optical Flow, Recurrent Convolutions, Latency)
   - 1.4 Level 3: Explain Like I'm the Principal Engineer Maintaining BLINK (Code Paths, State Invariants, Math, Failure Modes)
   - 1.5 The Ultimate Unifying Analogy
2. [The Scientific Problem & Earth Observation Physics](#2-the-scientific-problem--earth-observation-physics)
   - 2.1 Geostationary Earth Orbit (GEO) vs. Polar Low-Earth Orbit (LEO)
   - 2.2 Scan Mirror Physics & The 15–30 Minute Cadence Bottleneck
   - 2.3 Fast-Evolving Meso-Scale Atmospheric Phenomena
   - 2.4 Spatial vs. Temporal Resolution Trade-Offs
   - 2.5 Nowcasting vs. Temporal Interpolation
   - 2.6 The Reality of "Zero-Payload Rapid Scanning" and "Neural Kinematics"
   - 2.7 What BLINK ACTUALLY Does (and What It Does NOT Do)
3. [Complete Repository Anatomy & Directory Walkthrough](#3-complete-repository-anatomy--directory-walkthrough)
   - 3.1 Verified Directory Tree
   - 3.2 Deep Dive on Every Directory
   - 3.3 Master File Index
   - 3.4 Recommended Reading Order for Developers
4. ["What The Fuck is a Frame?" — The Complete Data Lifecycle](#4-what-the-fuck-is-a-frame--the-complete-data-lifecycle)
   - 4.1 Defining a "Frame" in BLINK
   - 4.2 Data Representations Across the Lifecycle
   - 4.3 End-to-End Frame Lifecycle Diagram
   - 4.4 Tensor Shapes and Dimensional Transformations
   - 4.5 Real Observation Frames vs. Synthesized Frames
5. [MOSDAC Ingestion, Formats, and Simulation](#5-mosdac-ingestion-formats-and-simulation)
   - 5.1 What is MOSDAC?
   - 5.2 Level-1B vs. Level-2 Products
   - 5.3 NetCDF4 and HDF5 Data Structures
   - 5.4 Deep Dive on `src/ingestion/mosdac_parser.py`
   - 5.5 Thermal Inversion Logic for Cold Cloud-Top Tracking
   - 5.6 Mathematical Mechanics of `SyntheticMOSDACSimulator`
   - 5.7 Real Data vs. Synthetic Data Boundaries
6. [Spectral Bands & Satellite Radiometry](#6-spectral-bands--satellite-radiometry)
   - 6.1 The 6 Supported INSAT-3DS Imager Channels
   - 6.2 Visible vs. Infrared Radiometry Explained Simply
   - 6.3 False-Color Composite Synthesis (VIS + WV + TIR1)
   - 6.4 Channel View Selection: Computation vs. Display Audit
7. [Geospatial Preprocessing & Overlapping Tiling](#7-geospatial-preprocessing--overlapping-tiling)
   - 7.1 Deep Dive on `src/ingestion/preprocessor.py`
   - 7.2 Why Tiling is Required for Full-Disk Satellites
   - 7.3 Overlapping Tile Slicing & Stride Mechanics
   - 7.4 The Hann Window: Mathematical Formulation & Blending Logic
   - 7.5 Stitching Reconstruction & Weighted Overlap Normalization
8. [Configuration Deep Dive: `settings.yaml`](#8-configuration-deep-dive-settingsyaml)
   - 8.1 Exhaustive Parameter Breakdown
   - 8.2 Parameter Modification Matrix
9. [Optical Flow Engine: RAFT & Exact Backward Warping](#9-optical-flow-engine-raft--exact-backward-warping)
   - 9.1 What is Optical Flow?
   - 9.2 RAFT (Recurrent All-Pairs Field Transforms) Deep Dive
   - 9.3 Torchvision RAFT vs. Lightweight Fallback Engine
   - 9.4 Bidirectional Flow Estimation ($f_{0 \to 1}$ and $f_{1 \to 0}$)
   - 9.5 Why Both Forward and Backward Flows Are Required
   - 9.6 Backward Warping vs. Forward Splatting
   - 9.7 `torch.nn.functional.grid_sample` Mechanics
   - 9.8 Zero-Drift Alignment and `_cached_pixel_grid`
10. [Spatiotemporal Dynamics: ConvLSTM Memory Block](#10-spatiotemporal-dynamics-convlstm-memory-block)
    - 10.1 Why Motion Alone Fails for Clouds
    - 10.2 Deep Dive on `src/models/conv_lstm.py`
    - 10.3 ConvLSTM Gating Mathematics & Memory States
    - 10.4 Spatial Preservation via 2D Convolutions
    - 10.5 Multi-Layer Recurrent Sequence Processing
11. [Multi-Scale Image Refinement: U-Net Decoder](#11-multi-scale-image-refinement-u-net-decoder)
    - 11.1 Why U-Net is Needed Post-Warping
    - 11.2 Deep Dive on `src/models/unet_decoder.py`
    - 11.3 Residual DoubleConv Block with GroupNorm
    - 11.4 Encoder-Decoder Architecture with Bilinear Upsampling & Skip Connections
    - 11.5 Dual Output Heads: Soft Mask & Radiance Residual Correction
12. [Pipeline Orchestration: `AeroInterpolator`](#12-pipeline-orchestration-aerointerpolator)
    - 12.1 Deep Dive on `src/pipeline/interpolator.py`
    - 12.2 End-to-End Frame Synthesis Flow
    - 12.3 Time Mapping: Normalized $t \in (0, 1)$ to Real-World Minutes
    - 12.4 Temporal Upsampling Factors (3x, 5x, 15x) and Sub-Timesteps
    - 12.5 Deterministic Flow-Guided Synthesis (`_flow_guided_synthesis`)
    - 12.6 Flow-Consistency Confidence Scoring
    - 12.7 Linear Blending vs. BLINK: The Mechanics of Ghosting
    - 12.8 Physics-Guided Reality Check
13. [Physics Evaluation & Validation Suite](#13-physics-evaluation--validation-suite)
    - 13.1 Deep Dive on `src/pipeline/physics_eval.py`
    - 13.2 Peak Signal-to-Noise Ratio (PSNR)
    - 13.3 Structural Similarity Index Measure (SSIM)
    - 13.4 Fluid Divergence Regularizer ($\|\nabla \cdot \vec{u}\|^2$)
    - 13.5 Radiance Conservation Percentage
    - 13.6 End-Point Error (EPE)
    - 13.7 Ghosting Reduction Calculation
    - 13.8 Benchmark Verification Analysis (`scripts/benchmark_eval.py`)
14. [Meteorological Nowcasting & Trajectory Prediction](#14-meteorological-nowcasting--trajectory-prediction)
    - 14.1 Deep Dive on `src/pipeline/nowcasting.py`
    - 14.2 Interpolation vs. Nowcasting Distinction
    - 14.3 `StormTrackPredictor`: Centroids, Steering, Beta-Drift & Dvorak Winds
    - 14.4 Probabilistic Cone of Uncertainty Generation
    - 14.5 `ConvectiveNowcaster`: Cloud-Top Cooling Rate & Cloudburst Index
    - 14.6 Comparison: `interpolator.py` vs. `physics_eval.py` vs. `nowcasting.py`
15. [API Gateway & Serving Layer: `src/api/server.py`](#15-api-gateway--serving-layer-srcapiserverpy)
    - 15.1 FastAPI Framework Architecture
    - 15.2 Complete REST Endpoint Documentation
    - 15.3 Concurrency, Locks, and Memory Management
    - 15.4 Device Selection and Fallback
16. [User Interface: Web Console & Streamlit Dashboard](#16-user-interface-web-console--streamlit-dashboard)
    - 16.1 The Streamlit Dashboard (`ui/dashboard.py`)
    - 16.2 The High-Density Operational Web Console (`ui/static/index.html`)
    - 16.3 Complete UI Controls & Event Handler Map
    - 16.4 Telemetry HUD & Diagnostic Tables
    - 16.5 Visual Elements, Legends, Colors, and Badges
    - 16.6 Narrative Walkthrough: "Sitting in Front of BLINK"
17. [Complete Mathematical Specification](#17-complete-mathematical-specification)
    - 17.1 Optical Flow & Warping Formulations
    - 17.2 ConvLSTM Recurrent Update Equations
    - 17.3 Fluid Divergence & Spatial Derivatives
    - 17.4 SSIM Covariance Windowing
    - 17.5 Kinematic Trajectory & Coriolis Extrapolation
    - 17.6 Cloudburst Probability Formulation
18. [The "Why" Map & Architecture Justification](#18-the-why-map--architecture-justification)
    - 18.1 Problem-to-Implementation Traceability Matrix
    - 18.2 Why this Specific Architecture?
    - 18.3 Historical Reasoning Classification
19. [Complete Code Registries](#19-complete-code-registries)
    - 19.1 Important Variable Registry
    - 19.2 Master Function Index
    - 19.3 Master Class Index
    - 19.4 Import & Dependency Index (`requirements.txt`)
    - 19.5 Containerization & Dockerfile Dissection
20. [Systemic Audit: Assumptions, Weaknesses, Magic Numbers, & Dead Code](#20-systemic-audit-assumptions-weaknesses-magic-numbers--dead-code)
    - 20.1 Magic Number Registry
    - 20.2 Hidden Assumptions
    - 20.3 Failure Modes & Fragilities
    - 20.4 Dead, Suspicious, or Fallback Code
    - 20.5 "What Happens If I Change This?" Dependency Matrix
    - 20.6 Current Reality Check
21. [Glossary & BLINK Dictionary](#21-glossary--blink-dictionary)
    - 21.1 100+ Essential Term Glossary
    - 21.2 The BLINK Dictionary
    - 21.3 Explaining Dashboard Metrics Like I'm 10
22. [One-Page Cheat Sheet & Executive Summary](#22-one-page-cheat-sheet--executive-summary)
    - 22.1 BLINK Cheat Sheet
    - 22.2 BLINK in One Sentence
    - 22.3 BLINK in One Paragraph
    - 22.4 Final Mental Model

---

# 1. Executive Story & The Three Levels of Understanding

```
                     ┌─────────────────────────────────────────────────┐
                     │              GEOSTATIONARY SATELLITE            │
                     │          (36,000 km altitude, Fixed Orbit)      │
                     └────────────────────────┬────────────────────────┘
                                              │ Mechanical Scan Mirror Sweep
                                              │ takes 15 to 30 minutes!
                                              ▼
                     ┌─────────────────────────────────────────────────┐
                     │            EARTH OBSERVATION GAP                │
                     │  Frame T0 (00:00)              Frame T1 (00:15) │
                     │  [Known Picture]               [Known Picture]  │
                     └────────┬───────────────────────────────┬────────┘
                              │                               │
                              │   MISSING ATMOSPHERIC STATES  │
                              │   Severe storm forms in 5-10m!│
                              ▼                               ▼
                     ┌─────────────────────────────────────────────────┐
                     │                 PROJECT BLINK                   │
                     │         Ground-Station Software Overlay         │
                     │   Neural Kinematics + Bidirectional Warping     │
                     └────────────────────────┬────────────────────────┘
                                              │ Synthesizes 14 intermediate
                                              │ frames (1-minute cadence)
                                              ▼
                     ┌─────────────────────────────────────────────────┐
                     │      CONTINUOUS HIGH-CADENCE RAPID SCAN         │
                     │  T+00  T+01  T+02  T+03 ... T+13  T+14  T+15    │
                     │  Real   AI    AI    AI  ...  AI    AI   Real    │
                     └─────────────────────────────────────────────────┘
```
### 1.1 The Narrative of a Satellite Looking at Earth
Imagine an orbital sentinel hanging in space, exactly 35,786 kilometers above the equator over the Indian Ocean. This satellite—such as India's **INSAT-3DS** or **INSAT-3DR**—is geostationary. Because its orbital speed matches the exact rotation of Earth, it stays parked over the exact same patch of ocean, subcontinent, and mountains 24 hours a day.
To take a photograph of Earth's weather, the satellite does not use a phone camera that snaps everything in a millisecond. It uses a massive, high-precision mechanical **scan mirror**. This mirror slowly rocks back and forth, sweeping line by line across thousands of kilometers of Earth's disk, measuring infrared heat and reflected sunlight. 
Because Earth is enormous and the optics must collect faint radiation across multiple spectral wavelengths, completing a single full scan of the subcontinent takes **15 to 30 minutes**.
Now, consider the atmosphere. A tropical cyclone eyewall convective burst, a deadly Himalayan cloudburst, or a severe thunderstorm updraft does not wait 30 minutes. In meteorology, explosive convection can erupt, dump catastrophic rain, and trigger flash floods in **5 to 10 minutes**.
Under the standard satellite imaging cycle, human forecasters and automated early-warning systems are operating blind between scans. When frame $T_0$ arrives at 10:00 AM, the atmosphere is clear. By the time frame $T_1$ arrives at 10:15 AM, a cloudburst is already underway.
The traditional aerospace solution is to design, build, and launch more satellites with smaller regional scan sectors. However, launching a dedicated rapid-scanning satellite constellation costs hundreds of millions of dollars and takes a decade.
**BLINK (Aero-Interpolate)** solves this problem on the ground using pure software. BLINK sits on a computer server at the satellite ground receiving station. When frame $T_0$ and frame $T_1$ arrive 15 minutes apart, BLINK computes the physical motion fields of the atmosphere, tracks how clouds are swirling and expanding, and generates 14 physically realistic, synthesized intermediate frames at 1-minute intervals ($T+1, T+2, \dots, T+14$).
---
### 1.2 Level 1: Explain Like I'm 10 (Zero-Jargon Physical Analogies)
* **The Flipbook Problem:** Imagine you have a flipbook of a cartoon character running. If the flipbook only has two pages—one where the character is on the left side of the room, and one where they are on the right side 15 seconds later—flipping the pages looks jerky and weird. You have no idea how they got across the room or if they tripped on a rug.
* **The Lazy Solution (Ghosting):** If you take both pictures and print them on top of each other using tracing paper, you don't get motion. You get two transparent, see-through characters overlapping like ghosts. That is called **linear blending**, and it is completely useless for weather forecasting.
* **What BLINK Does:** BLINK is like a smart artist. BLINK looks at the character's feet, arms, and body in the first picture, looks at where they ended up in the second picture, and figures out how fast each part was moving. Then, BLINK draws 14 brand-new pictures in between, showing the character taking every single step smoothly.
* **Why Weather is Harder than Cartoons:** A cartoon character is made of solid lines. Clouds are made of water vapor and ice. A storm can spin like a giant whirlpool, grow taller, shrink, or blow apart in the wind. BLINK has special rules built into its AI that understand how air and clouds move so it doesn't draw impossible shapes.
---
### 1.3 Level 2: Explain Like I'm a Computer Science Student (Tensors, Optical Flow, Recurrent Convolutions, Latency)
From a computer science perspective, BLINK is a **Multi-Spectral Video Frame Interpolation (VFI) and Kinematic Extrapolation Engine** optimized for geostationary Earth Observation (EO) data.
```
Input:  Two multi-channel radiance tensors I_0, I_1 in R^{1 x C x H x W} separated by Delta t = 15 min.
Output: A sequence of N synthesized tensors {I_t}_{t in (0, 1)} approximating ground-truth radiance.
```
1. **Multi-Spectral Ingestion:** Raw HDF5/NetCDF files containing multi-channel floating-point arrays (Visible, Water Vapor, Thermal Infrared) are parsed, calibrated to physical units (Kelvin or % Reflectance), normalized to $[0.0, 1.0]$, and loaded into PyTorch tensors of shape $(1, C, H, W)$.
2. **Bidirectional Motion Estimation (RAFT):** A deep Recurrent All-Pairs Field Transforms (RAFT) optical flow network computes dense 2D vector displacement fields:
   $$\mathbf{f}_{0 \to 1} = (u_{01}, v_{01}), \quad \mathbf{f}_{1 \to 0} = (u_{10}, v_{10}) \in \mathbb{R}^{1 \times 2 \times H \times W}$$
3. **Sub-Pixel Backward Warping:** For any arbitrary query timestamp $t \in (0.0, 1.0)$, backward warping is performed via bilinear sampling (`torch.nn.functional.grid_sample` with `align_corners=True` and `padding_mode="border"`) using linearly scaled motion vectors:
   $$\hat{I}_0(t) = \mathcal{W}\left(I_0, -t \cdot \mathbf{f}_{0 \to 1}\right), \quad \hat{I}_1(t) = \mathcal{W}\left(I_1, -(1 - t) \cdot \mathbf{f}_{1 \to 0}\right)$$
4. **Flow Consistency & Adaptive Synthesis:**
   - In deterministic production mode (`refinement_mode="flow"`), forward-backward flow consistency errors are calculated to derive confidence maps, smoothly blending $\hat{I}_0(t)$ and $\hat{I}_1(t)$ while falling back to conservative bounds in occluded/disoccluded regions.
   - In neural mode (`refinement_mode="neural"`), warped candidate frames pass through a multi-layer **ConvLSTM** (to model spatiotemporal latent dynamics) followed by a **U-Net Decoder** that outputs an adaptive soft-blending mask $M_t$ and residual radiance correction $\Delta I_t$.
5. **Physics Verification & Telemetry:** Synthesized tensors are evaluated against fluid dynamics constraints (divergence regularizer $\|\nabla \cdot \vec{u}\|^2$, radiance conservation) and computer vision benchmarks (PSNR, SSIM) with an inference latency of $<50\text{ ms}$ per $512 \times 512$ spatial tile.

---

### 1.4 Level 3: Explain Like I'm the Principal Engineer Maintaining BLINK (Code Paths, State Invariants, Math, Failure Modes)

To maintain, debug, or extend BLINK, you must understand the exact software contracts, numerical guarantees, and execution graph:
* **Entry Point Coordinator:** `AeroInterpolator` in `src/pipeline/interpolator.py` manages initialization, sub-module lifecycle, device placement (`cuda` vs. `cpu`), and thread-safe batch execution via an explicit `_inference_lock` in `src/api/server.py`.
* **Flow Field Contract:** All optical flow fields produced by `RAFTEngine.estimate_bidirectional_flow` must have shape $(B, 2, H, W)$ where channel index $0$ is horizontal displacement $u = \Delta x$ (positive East/Right) and index $1$ is vertical displacement $v = \Delta y$ (positive South/Down) expressed in **pixel units**.
* **Coordinate Space Invariant:** In `backward_warp` (`src/models/raft_engine.py`), the sampling grid is constructed via `_cached_pixel_grid` and normalized to the canonical $[-1.0, 1.0]$ domain:
  $$x_{\text{norm}} = \frac{2(x + u)}{W - 1} - 1.0, \quad y_{\text{norm}} = \frac{2(y + v)}{H - 1} - 1.0$$
  The strict requirement `align_corners=True` ensures that extremal pixel centers map precisely to $-1.0$ and $+1.0$, preventing spatial grid drift across cascaded interpolations.
* **Thermal Inversion Invariant:** In `MOSDACParser.to_normalized_tensor` (`src/ingestion/mosdac_parser.py`), temperature channels (`IMG_MWIR`, `IMG_WV`, `IMG_TIR1`, `IMG_TIR2`) undergo an inversion transform:
  $$\text{norm} = 1.0 - \frac{T - T_{\min}}{T_{\max} - T_{\min}}$$
  This maps cold convective cloud tops ($T \approx 190\text{ K} - 210\text{ K}$) to values near $1.0$ (high intensity). This is mathematically vital: RAFT feature extractors rely on gradient magnitudes, and active convective storm cores must present as strong, high-contrast features rather than dark voids.
* **Dual-Engine Graceful Fallback:** When running in environments without pre-trained neural checkpoint weights, `AeroInterpolator` defaults to `refinement_mode="flow"`. This bypasses the uncalibrated weights of the randomly initialized U-Net and ConvLSTM, utilizing a deterministic, flow-consistency-weighted interpolation formulation (`_flow_guided_synthesis`) that guarantees monotonicity, zero ghosting, and radiance conservation.
---
### 1.5 The Ultimate Unifying Analogy
Think of BLINK as an **expert atmospheric cartographer and fluid dynamicist watching a pair of satellite photographs through a stereoscopic light table**. 
The cartographer first computes the wind velocity vectors across every square kilometer of cloud. Rather than simply fading one picture into the next (which creates double-exposed "ghost" clouds), the cartographer takes a digital scalpel, cuts out each individual cloud structure, slides it forward in time along its exact physical wind trajectory, deforms its edges according to local atmospheric vorticity and divergence, checks that total energy and moisture are conserved, and paints the exact intermediate frame. 
Finally, the cartographer hands the synthesized frame to a radar nowcaster who immediately measures the storm's cloud-top cooling rate and flags cyclone landfall coordinates for disaster management authorities.
---
# 2. The Scientific Problem & Earth Observation Physics
```
                    ┌──────────────────────────────────────────────┐
                    │          THE SPATIAL-TEMPORAL DILEMMA        │
                    ├──────────────────────────────────────────────┤
                    │ Fast Scan (1-min)  ──> Low Spatial Detail    │
                    │ Full Disk (4-km)   ──> Slow Cadence (15-30m) │
                    ├──────────────────────────────────────────────┤
                    │ BLINK Ground Software bridges this gap:      │
                    │ Full Disk 4km + 1-minute Virtual Cadence!    │
                    └──────────────────────────────────────────────┘
```

### 2.1 Geostationary Earth Orbit (GEO) vs. Polar Low-Earth Orbit (LEO)

Satellite meteorology relies on two fundamentally distinct orbital regimes:

| Feature | Geostationary Orbit (GEO) | Polar Low-Earth Orbit (LEO) |
|---|---|---|
| **Orbital Altitude** | $\approx 35,786\text{ km}$ above Equator | $500 - 900\text{ km}$ above Poles |
| **Orbital Period** | Exactly $23\text{h } 56\text{m } 04\text{s}$ (Matches Earth rotation) | $90 - 105\text{ minutes}$ per orbit |
| **Field of View** | Constant staring view of $\approx 42\%$ of Earth's disk | Narrow swath beneath spacecraft |
| **Temporal Revisit** | Continuous (cadence limited only by sensor scan mirror) | 12 to 24 hours over the same location |
| **Spacecraft Examples** | INSAT-3DS, INSAT-3DR, GOES-16, Meteosat-11 | NOAA-20, MetOp-C, Sentinel-3 |
| **Role in BLINK** | **Primary target:** Provides continuous regional baseline scans | Not suitable for 1-minute continuous interpolation |

Geostationary satellites are the only orbital platforms capable of monitoring the rapid initiation and evolution of severe storms over continental and oceanic domains.

---

### 2.2 Scan Mirror Physics & The 15–30 Minute Cadence Bottleneck

A common misconception is that geostationary weather satellites capture instantaneous 2D photos like a smartphone camera. In reality, geostationary imagers (such as the **6-Channel Imager** aboard INSAT-3DS) use a single optical telescope coupled to high-sensitivity detector arrays and a 2-axis motorized **scan mirror**.

```
   West-East Fast Scan (Scan Line Acquisition)
   ══════════════════════════════════════════════════════> (Step 1)
   <───────────────────────────────────────────────────── (Step 2: Mirror Steps South)
   ══════════════════════════════════════════════════════> (Step 3: Scan Line 2)
   ... (Repeated 2,000+ times to assemble Full Disk image) ...
   Total Mirror Sweep Duration: 15 to 30 Minutes
```

To achieve sufficient signal-to-noise ratio (SNR) in narrow infrared absorption bands (such as the $6.8\,\mu\text{m}$ water vapor channel), the mirror must scan at a controlled angular velocity. Assembling a full-disk image of $2000 \times 2000$ to $5000 \times 5000$ pixels requires:
1. Scanning East-to-West across a line.
2. Stepping North-to-South.
3. Performing periodic blackbody calibration sweeps.

This physical mechanism imposes a strict hardware scan cycle of **15 minutes for regional scans** and **30 minutes for full-disk acquisitions**.

---

### 2.3 Fast-Evolving Meso-Scale Atmospheric Phenomena

While satellite scan mirrors operate on a 15- to 30-minute cadence, critical atmospheric processes operate on significantly shorter timescales:

```
0 min                  5 min                 10 min                15 min
──┼──────────────────────┼──────────────────────┼────────────────────┼──> Time
  │                      │                      │                    │
[Scan T0]                │                      │                [Scan T1]
Clear Sky / Cumulus      Convective Updraft     Explosive Rain   Flash Flood
                         Overshooting Top       Hail / Cloudburst
                         ────────── BLINK SYNTHESIS ───────────>
```

* **Convective Cloudbursts:** Explosive vertical updrafts ($>30\text{ m/s}$) can push cloud tops from $5\text{ km}$ to $16\text{ km}$ altitude within 8 minutes. By the time consecutive 15-minute scans are collected, the cloudburst has already reached peak rainfall intensity ($>100\text{ mm/hr}$).
* **Cyclone Eyewall Meso-Vortices & Eyewall Replacement Cycles (ERC):** Inner-core convective bursts rotate around the eyewall at speeds exceeding $200\text{ km/h}$. In a 15-minute gap, an eyewall meso-vortex travels over $50\text{ km}$, causing severe tracking disorientation in low-cadence imagery.
* **Aviation Microbursts:** Severe low-level convective downdrafts develop and dissipate within 5 to 10 minutes, presenting extreme wind-shear hazards during aircraft takeoff and landing.

---

### 2.4 Spatial vs. Temporal Resolution Trade-Offs

Satellite sensor design faces an unyielding optical tradeoff:

$$\text{Data Throughput} \propto \frac{\text{Spatial Area} \times \text{Number of Channels}}{\text{Pixel Resolution}^2 \times \text{Scan Duration}}$$

To increase temporal frequency to 1 minute, traditional spacecraft must restrict their scan mirror to a tiny "mesoscale sector" (e.g., $1000 \times 1000\text{ km}$), abandoning coverage of the rest of the continent. 

**BLINK eliminates this tradeoff:** Ground stations collect standard full-disk multi-spectral scans every 15 minutes and apply software-based neural kinematics to upsample the entire synoptic disk to a **virtual 1-minute continuous cadence**.

---

### 2.5 Nowcasting vs. Temporal Interpolation

A critical distinction must be drawn between the two temporal prediction regimes:

| Dimension | Temporal Interpolation (BLINK Core) | Meteorological Nowcasting (BLINK Secondary) |
|---|---|---|
| **Query Range** | $T_0 < t < T_1$ (Between two known observations) | $t > T_1$ (Future forecasting into the unknown) |
| **Boundary Conditions** | **Two-Point Boundary Problem:** Constrained by past $T_0$ and future $T_1$ | **Initial Value Problem:** Extrapolating forward from $T_1$ only |
| **Mathematical Formulation** | Bidirectional backward warping & state interpolation | Kinematic forward integration with Coriolis & Beta-Drift |
| **Primary Code Module** | `src/pipeline/interpolator.py` | `src/pipeline/nowcasting.py` |
| **Error Growth** | Bounded: Error is minimal at $t \to 0$ and $t \to 1$ | Unbounded: Uncertainty expands over time |

BLINK performs both: its primary engine synthesizes the intermediate observations between $T_0$ and $T_1$, while its nowcasting module uses the derived motion vectors to project storm tracks up to 48 hours into the future.

---

### 2.6 The Reality of "Zero-Payload Rapid Scanning" and "Neural Kinematics"

* **"Zero-Payload Rapid Scanning":** This is an accurate operational descriptor. It means achieving the operational benefits of rapid-scan satellite constellations (1-minute cadence) without launching new spacecraft payloads or modifying satellite hardware. The processing is entirely executed on ground-station compute infrastructure.
* **"Neural Kinematics":** This is a domain-specific technical branding. "Kinematics" refers to the classical description of motion without regard to mass or forces (optical flow velocity fields $\vec{u} = (u, v)$). "Neural" refers to the deep learning models (RAFT, ConvLSTM, U-Net) that estimate these vector fields and resolve non-rigid deformations.

---

### 2.7 What BLINK ACTUALLY Does (and What It Does NOT Do)

To preserve scientific rigor, we state explicitly what BLINK does and does not do according to its codebase:

```
+-----------------------------------------------------------------------------+
|                               WHAT BLINK DOES                               |
+-----------------------------------------------------------------------------+
| 1. Reads calibrated Level-1B radiance arrays from HDF5/NetCDF4 files.       |
| 2. Computes bidirectional optical flow displacement vectors between frames. |
| 3. Warps multi-spectral channels using sub-pixel bilinear grid sampling.   |
| 4. Blends warped candidate states using flow-consistency confidence maps.   |
| 5. Quantifies fluid divergence and radiance conservation across timesteps.   |
| 6. Detects storm centroids and extrapolates kinematic forecast tracks.      |
+-----------------------------------------------------------------------------+
|                            WHAT BLINK DOES NOT DO                           |
+-----------------------------------------------------------------------------+
| 1. It does NOT command or control satellite hardware in orbit.              |
| 2. It does NOT create new spatial resolution (it is not super-resolution).  |
| 3. It does NOT run full 3D Numerical Weather Prediction (NWP/WRF).          |
| 4. It does NOT guarantee that synthetic clouds represent ground truth if a  |
|    spontaneous unobserved convective explosion occurred and dissipated     |
|    entirely inside the 15-minute gap without leaving a trace at T0 or T1.   |
+-----------------------------------------------------------------------------+
```

---

# 3. Complete Repository Anatomy & Directory Walkthrough

```
BLINK/
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── pytest.ini
├── requirements.txt
├── config/
│   └── settings.yaml
├── data/
│   ├── processed_tensors/
│   └── raw_netcdf/
│       └── .gitkeep
├── scripts/
│   ├── benchmark_eval.py
│   └── simulate_mosdac_data.py
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── mosdac_parser.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── conv_lstm.py
│   │   ├── raft_engine.py
│   │   └── unet_decoder.py
│   └── pipeline/
│       ├── __init__.py
│       ├── interpolator.py
│       ├── nowcasting.py
│       └── physics_eval.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_ingestion.py
│   ├── test_models.py
│   ├── test_physics_eval.py
│   └── test_pipeline.py
└── ui/
    ├── dashboard.py
    └── static/
        └── index.html
```

### 3.1 Verified Directory Tree

The directory tree above represents the exact structure of the BLINK repository. Every directory and file serves a dedicated architectural purpose.

---

### 3.2 Deep Dive on Every Directory

#### `config/`
* **What is it?** System configuration store.
* **Why does it exist?** Decouples runtime operational parameters (spectral channel bounds, tiling dimensions, model architectures, network ports) from source code.
* **What lives inside?** `settings.yaml`.
* **Who uses it?** Used across ingestion, model initializers, pipeline coordinators, and API server.
* **Criticality:** Essential. Without configuration parameters, the pipeline cannot initialize channel bounds or model dimensions.
* **Disappearance impact:** The system would throw file-not-found exceptions or fail to instantiate models with default parameters.

#### `data/`
* **What is it?** Local storage staging directory for raw satellite files and intermediate tensors.
* **Why does it exist?** Provides a defined operational workspace for raw Level-1B HDF5/NetCDF files (`data/raw_netcdf/`) and processed PyTorch tensor caches (`data/processed_tensors/`).
* **Who uses it?** `simulate_mosdac_data.py`, `mosdac_parser.py`, and Docker container volume mounts.
* **Criticality:** Non-essential for in-memory API inference, but essential for batch file processing and simulation workflows.

#### `scripts/`
* **What is it?** Standalone CLI utilities and verification harnesses.
* **Why does it exist?** Allows engineers to generate synthetic benchmark datasets (`simulate_mosdac_data.py`) and verify quantitative PSNR/SSIM scorecards (`benchmark_eval.py`) without launching web servers.
* **Who uses it?** Research scientists, CI/CD pipelines, and benchmarking suites.
* **Criticality:** Essential for automated validation and synthetic testing.

#### `src/`
* **What is it?** Core Python source package.
* **Why does it exist?** Encapsulates the four architectural layers:
  1. `src/ingestion/`: Data parsing, calibration, normalization, and Hann tiling.
  2. `src/models/`: Neural network blocks (RAFT, ConvLSTM, U-Net Decoder).
  3. `src/pipeline/`: Pipeline orchestration, nowcasting, and physics evaluation.
  4. `src/api/`: FastAPI REST endpoints and web serving.
* **Criticality:** Absolute core. Contains the entire executable logic of BLINK.

#### `tests/`
* **What is it?** Automated unit and integration test suite.
* **Why does it exist?** Enforces mathematical invariants, tensor dimensional consistency, warping accuracy, calibration clipping, and API route status codes using `pytest`.
* **Who uses it?** Developers and automated CI/CD workflows.
* **Criticality:** Essential for software quality and regression prevention.

#### `ui/`
* **What is it?** Presentation and operational visualization layer.
* **Why does it exist?** Provides two distinct user interfaces:
  1. `ui/static/index.html`: A high-density, dark-mode ground-station operations console with real-time vector flow fields, temperature profiles, pixel probes, and nowcasting telemetry.
  2. `ui/dashboard.py`: A Streamlit demo application for comparative split-screen analysis.
* **Criticality:** Essential for human-in-the-loop operational forecasting.

---

### 3.3 Master File Index

| File Path | Primary Responsibility | Key Classes / Functions | Core Dependencies | Used By |
|---|---|---|---|---|
| `config/settings.yaml` | System configuration & physical channel lookup tables | YAML structure | PyYAML | All modules |
| `src/ingestion/mosdac_parser.py` | Level-1B HDF5 parsing, calibration, normalization, and synthetic atmospheric simulation | `MOSDACParser`, `SyntheticMOSDACSimulator`, `CHANNEL_CALIBRATION_BOUNDS` | `h5py`, `numpy`, `torch` | `preprocessor.py`, `interpolator.py`, `server.py`, `scripts` |
| `src/ingestion/preprocessor.py` | Hann-window overlap tiling, tile stitching, false-color composite generation | `TileProcessor`, `GeoNormalizer` | `numpy`, `torch` | `interpolator.py`, `server.py`, `dashboard.py` |
| `src/models/raft_engine.py` | RAFT optical flow estimation, lightweight fallback, and sub-pixel backward warping | `RAFTEngine`, `LightweightOpticalFlow`, `backward_warp`, `_cached_pixel_grid` | `torch`, `torchvision` | `interpolator.py`, `unet_decoder.py` |
| `src/models/conv_lstm.py` | Spatiotemporal recurrent memory cells and multi-layer ConvLSTM architectures | `ConvLSTMCell`, `ConvLSTM` | `torch`, `torch.nn` | `interpolator.py` |
| `src/models/unet_decoder.py` | Multi-scale refinement U-Net decoder with residual DoubleConv and soft mask heads | `PhysicsGuidedUNetDecoder`, `DoubleConv` | `torch`, `torch.nn.functional` | `interpolator.py` |
| `src/pipeline/interpolator.py` | End-to-end synthesis coordinator, flow-guided synthesis, sub-timestep scheduler | `AeroInterpolator`, `InterpolationResult` | `torch`, `numpy`, all model & ingestion modules | `server.py`, `dashboard.py`, `benchmark_eval.py` |
| `src/pipeline/physics_eval.py` | PSNR, SSIM, fluid divergence loss, radiance conservation, ghosting reduction | `PhysicsEvaluator`, `MetricReport` | `torch`, `numpy` | `interpolator.py`, `server.py`, `benchmark_eval.py` |
| `src/pipeline/nowcasting.py` | Storm centroid tracking, trajectory prediction, Dvorak wind estimation, cloudburst index | `StormTrackPredictor`, `ConvectiveNowcaster`, `StormTrackReport` | `torch`, `numpy`, `math` | `server.py` |
| `src/api/server.py` | High-throughput FastAPI REST gateway, base64 encoders, and web console serving | `app`, `get_interpolator`, endpoints (`/v1/interpolate/*`, `/v1/simulate/*`) | `fastapi`, `uvicorn`, `pydantic`, `PIL`, `matplotlib` | External REST clients, Browser Console |
| `ui/static/index.html` | Operational ground-station web console with Canvas overlays and telemetry tables | DOM, Canvas JS, Vector Field Renderer | HTML5, Canvas API, Fetch REST | Meteorologists, Web Clients |
| `ui/dashboard.py` | Streamlit interactive operational comparative demo | `run_streamlit_app` | `streamlit`, `PIL`, `torch` | Meteorologists, Demonstrations |
| `scripts/simulate_mosdac_data.py` | CLI tool to generate synthetic multi-spectral HDF5 files | `main` | `argparse`, `mosdac_parser` | CLI Users |
| `scripts/benchmark_eval.py` | Verification scorecard runner comparing BLINK synthesis with synthetic ground truth | `run_benchmark` | `interpolator`, `physics_eval` | CLI Users, Verification |
| `Dockerfile` | Multi-stage containerization recipe for edge ground-station deployment | Container build instructions | Ubuntu 22.04, CUDA 12.2, Python 3.11 | Docker Engine, Kubernetes |
| `requirements.txt` | Explicit Python package dependency list | Package versions | pip | Virtual Environments, Docker |

---

### 3.4 Recommended Reading Order for Developers

If you are new to the BLINK codebase, study the files in this precise dependency-driven order:

```
1. config/settings.yaml
   └─> Learn the parameters, channels, and physical calibration bounds.
2. src/ingestion/mosdac_parser.py
   └─> Understand how raw satellite data becomes a normalized PyTorch tensor.
3. src/models/raft_engine.py
   └─> Master optical flow estimation and the backward_warp grid_sample mechanics.
4. src/models/conv_lstm.py & src/models/unet_decoder.py
   └─> Study the spatiotemporal memory and refinement architecture.
5. src/pipeline/interpolator.py
   └─> Understand how flow, warping, and blending come together in AeroInterpolator.
6. src/pipeline/physics_eval.py
   └─> Learn how synthesis fidelity (PSNR, SSIM, Divergence) is quantitatively verified.
7. src/pipeline/nowcasting.py
   └─> Explore storm trajectory extrapolation and cloudburst probability calculation.
8. src/api/server.py
   └─> Review the REST endpoints, thread locks, and base64 serializers.
9. ui/static/index.html & ui/dashboard.py
   └─> Inspect the visualization canvas, timeline scrubbers, and telemetry HUD.
10. tests/ & scripts/
   └─> Verify test coverage and run the benchmark suite.
```

---

# 4. "What The Fuck is a Frame?" — The Complete Data Lifecycle

```
                    ┌──────────────────────────────────────────────┐
                    │            LIFECYCLE OF A "FRAME"            │
                    └──────────────────────┬───────────────────────┘
                                           │
1. Raw HDF5 Dataset                        ▼
   (Digital Counts / Kelvin)   ┌────────────────────────┐
                               │ Level-1B HDF5 Dataset  │
                               └───────────┬────────────┘
                                           │ MOSDACParser.read_hdf5()
2. Calibrated 2D NumPy Array               ▼
   (Kelvin / % Reflectance)    ┌────────────────────────┐
                               │ Dict[str, np.ndarray]  │
                               └───────────┬────────────┘
                                           │ MOSDACParser.to_normalized_tensor()
3. Normalized Float32 Tensor               ▼
   (Range [0.0, 1.0], Inverted)┌────────────────────────┐
                               │ Tensor (1, C, H, W)    │
                               └───────────┬────────────┘
                                           │ AeroInterpolator.interpolate()
4. Flow & Warping Execution                ▼
   (Displacement Fields)       ┌────────────────────────┐
                               │ Warped Tensors w0, w1  │
                               └───────────┬────────────┘
                                           │ Flow-Guided Blending / U-Net Refinement
5. Synthesized Tensor                      ▼
   (Continuous Timestep t)     ┌────────────────────────┐
                               │ Synth Frame (1, C, H,W)│
                               └───────────┬────────────┘
                                           │ GeoNormalizer.tensor_to_rgb_preview()
6. Displayable RGB Image                   ▼
   (Grayscale / False-Color)   ┌────────────────────────┐
                               │ RGB uint8 (H, W, 3)    │
                               └───────────┬────────────┘
                                           │ Base64 PNG Encoding
7. Operational Web Console                 ▼
   (Browser DOM / HTML Canvas) ┌────────────────────────┐
                               │ <img> & <canvas> Render│
                               └────────────────────────┘
```

### 4.1 Defining a "Frame" in BLINK

In video processing, a "frame" is simply an RGB image representing a slice of time in a movie. 

**In BLINK, a "frame" is a high-dimensional physical measurement of the Earth's atmosphere.**

A single satellite frame represents a multi-spectral snapshot of the planet captured at a specific observation timestamp (e.g., `2026-08-14T10:00:00Z`). It is not an 8-bit JPEG. It contains physical quantities:
* How much solar radiation is reflected from cloud tops (Visible/SWIR reflectance in %).
* How much thermal infrared energy is emitted by the ocean surface or storm clouds (Brightness Temperature in Kelvin).
* How much water vapor is concentrated in the mid-to-upper troposphere ($6.8\,\mu\text{m}$ absorption in Kelvin).

---

### 4.2 Data Representations Across the Lifecycle

As a frame moves through BLINK, its data structure morphs through seven distinct formats:

| Stage | Data Format | Memory Type | Shape / Type | Value Range | Physical Meaning |
|---|---|---|---|---|---|
| **1. Ingest** | Level-1B HDF5 / NetCDF4 | Disk file (`.h5`, `.nc`) | Datasets `IMG_VIS`, `IMG_TIR1`, etc. | Raw numbers / $-999.0$ fills | Raw sensor digital numbers / radiance |
| **2. Calibrated** | NumPy array dictionary | System RAM (`np.ndarray`) | `Dict[str, (H, W)]` float32 | $180\text{ K} - 330\text{ K}$ or $0 - 100\%$ | Calibrated Kelvin / % Reflectance |
| **3. Normalized** | PyTorch Float Tensor | GPU / CPU VRAM (`torch.Tensor`) | `(1, C, H, W)` float32 | $[0.0, 1.0]$ (Thermal Inverted) | Scaled radiance ($1.0 = \text{Cold Storm Top}$) |
| **4. Motion** | Optical Flow Vector Tensor | GPU / CPU VRAM (`torch.Tensor`) | `(1, 2, H, W)` float32 | $[- \max(H,W), + \max(H,W)]$ | Pixel displacements $(u, v)$ in $(\Delta x, \Delta y)$ |
| **5. Synthesized** | PyTorch Float Tensor | GPU / CPU VRAM (`torch.Tensor`) | `(1, C, H, W)` float32 | $[0.0, 1.0]$ | Interpolated intermediate state at time $t$ |
| **6. Preview** | NumPy uint8 RGB Array | System RAM (`np.ndarray`) | `(H, W, 3)` uint8 | $[0, 255]$ | False-color or high-contrast IR image |
| **7. Transport** | Base64 Encoded PNG String | JSON Payload / Network | `str` (Base64 UTF-8) | ASCII string | Compressed PNG for browser rendering |

---

### 4.3 End-to-End Frame Lifecycle Diagram

The detailed data flow from raw satellite ingestion to web visualization is illustrated in the diagram at the beginning of Section 4.

---

### 4.4 Tensor Shapes and Dimensional Transformations

To prevent shape mismatch bugs during pipeline extensions, memorize this dimensional progression:

```
MOSDAC HDF5 Channel (H, W) = (512, 512)
   │
   ▼
MOSDACParser.to_normalized_tensor()
   │ Stack 3 channels -> (3, 512, 512)
   │ Add batch dimension -> (1, 3, 512, 512)
   ▼
RAFTEngine.estimate_bidirectional_flow(frame_0, frame_1)
   │ Input: Two tensors of shape (1, 3, 512, 512)
   │ Output: flow_01 of shape (1, 2, 512, 512)
   │         flow_10 of shape (1, 2, 512, 512)
   ▼
backward_warp(frame, flow * t)
   │ Base pixel grid: (1, 2, 512, 512)
   │ Displaced sampling grid: (1, 512, 512, 2)
   │ Output warped frame: (1, 3, 512, 512)
   ▼
ConvLSTM (Optional Neural Mode)
   │ Input: Concatenated warped frames (1, 1, 6, 512, 512)  [Batch, Time, Channels, H, W]
   │ Hidden State H: (1, 16, 512, 512)
   │ Cell State C:   (1, 16, 512, 512)
   ▼
PhysicsGuidedUNetDecoder
   │ Input: Concatenated features (1, 13, 512, 512)
   │ Downsampling stages -> (1, 64, 256, 256) -> (1, 128, 128, 128)
   │ Bottleneck -> (1, 128, 128, 128)
   │ Upsampling + Skips -> (1, 64, 256, 256) -> (1, 32, 512, 512)
   │ Mask Head -> (1, 3, 512, 512)
   │ Residual Head -> (1, 3, 512, 512)
   │ Output Synthesized Tensor: (1, 3, 512, 512)
   ▼
GeoNormalizer.tensor_to_rgb_preview()
   │ Output NumPy Array: (512, 512, 3) uint8
```

---

### 4.5 Real Observation Frames vs. Synthesized Frames

* **Real Observation Frame ($T_0, T_1$):** Originates from physical photons striking satellite sensor detectors in geostationary orbit. It carries absolute calibration records, scan timestamps, detector telemetry, and physical sensor noise.
* **Synthesized Frame ($T_t$):** A software-derived mathematical estimate constructed from the kinematic vector fields of $T_0$ and $T_1$. It estimates where cloud top radiances and moisture boundaries travelled during the missing time interval.

---

# 5. MOSDAC Ingestion, Formats, and Simulation

```
+-----------------------------------------------------------------------------+
|                          MOSDAC HDF5 FILE ARCHITECTURE                      |
+-----------------------------------------------------------------------------+
| Root Attributes:                                                            |
|   ├── Satellite_Name = "INSAT-3DS"                                          |
|   ├── Observation_Time = "2026-08-14T10:00:00Z"                             |
|   ├── Product_Level = "L1B"                                                 |
|   └── Sensor = "6-Channel Multi-Spectral Imager"                            |
| Datasets:                                                                   |
|   ├── /IMG_VIS   (Float32 array, shape [512, 512], Unit: % Reflectance)     |
|   ├── /IMG_WV    (Float32 array, shape [512, 512], Unit: Kelvin)            |
|   └── /IMG_TIR1  (Float32 array, shape [512, 512], Unit: Kelvin)            |
+-----------------------------------------------------------------------------+
```

### 5.1 What is MOSDAC?

**MOSDAC** stands for the **Meteorological and Oceanographic Satellite Data Archival Centre**, operated by the **Space Applications Centre (SAC)** of the **Indian Space Research Organisation (ISRO)** in Ahmedabad, India.

MOSDAC is the authoritative data repository that ingests telemetry downlinked from India's meteorological satellites (INSAT-3D, INSAT-3DR, INSAT-3DS, OceanSat, Kalpana-1). It calibrates raw instrument counts into physical radiance units and publishes standardized datasets to weather agencies (such as the India Meteorological Department - IMD) and global research institutions.

---

### 5.2 Level-1B vs. Level-2 Products

* **Level-1B (L1B):** Calibrated and earth-located (geo-referenced) top-of-atmosphere multi-spectral radiances or brightness temperatures. This is the **primary input format ingested by BLINK**.
* **Level-2 (L2):** Derived geophysical parameter products generated by downstream meteorological algorithms (e.g., Cloud Top Brightness Temperature, Hydro-Estimator Precipitation Rate, Atmospheric Motion Vectors, Outgoing Longwave Radiation).

---

### 5.3 NetCDF4 and HDF5 Data Structures

INSAT-3DS data files from MOSDAC are formatted using **HDF5 (Hierarchical Data Format version 5)**, which is binary-compatible with **NetCDF4 (Network Common Data Form version 4)**.

Why are these formats used in Earth Observation?
1. **Self-Describing:** Datasets contain their own metadata (calibration equations, physical units, valid ranges, satellite sub-point coordinates) within the file header.
2. **Chunked Storage & Compression:** Multi-gigabyte full-disk datasets are compressed using internal zlib/gzip chunking, allowing selective reading of specific geographical bounding boxes without loading the entire multi-gigabyte file into RAM.
3. **Multi-Dimensional Arrays:** Natively represents multi-band 2D/3D grids with coordinate axes (Latitude, Longitude, Channel, Time).

---

### 5.4 Deep Dive on `src/ingestion/mosdac_parser.py`

Let us inspect the implementation of `MOSDACParser` in `src/ingestion/mosdac_parser.py`:

```python
class MOSDACParser:
    def __init__(self, channels: Optional[List[str]] = None):
        self.channels = channels or ["IMG_VIS", "IMG_WV", "IMG_TIR1"]
```

#### Key Traversal Logic (`read_hdf5`)
When opening a MOSDAC file, dataset keys may vary based on product level (e.g., `IMG_VIS`, `/Data/IMG_VIS`, `/BAND_IMG_VIS`, `radiance_img_vis`). `read_hdf5` handles this by checking a candidate list:

```python
candidate_keys = [
    ch,
    f"/{ch}",
    f"/Data/{ch}",
    f"/BAND_{ch}",
    f"/Radiance_{ch}",
    ch.lower(),
]
```

If exact keys match, data is extracted via `np.array(h5_file[key], dtype=np.float32)`. If specific channel names are not found, a case-insensitive fallback loop inspects all top-level keys.

#### Calibration & Fill-Value Sanitization (`_calibrate_channel`)
Raw satellite files contain invalid data flags (e.g., $-999.0$, $-9999.0$, `NaN`, `Inf`) representing space pixels outside Earth's disk or sensor calibration sweeps. Unchecked, negative numbers would destroy neural network gradients and optical flow calculations.

`_calibrate_channel` implements physical bounding and cleaning:
```python
fill_mask = (data < -900) | np.isnan(data) | np.isinf(data)
if np.any(fill_mask):
    valid_vals = data[~fill_mask]
    fallback_val = valid_vals.mean() if valid_vals.size > 0 else valid_min
    data[fill_mask] = fallback_val

data = np.clip(data, valid_min, valid_max)
```

---

### 5.5 Thermal Inversion Logic for Cold Cloud-Top Tracking

In thermal infrared satellite radiometry, colder temperatures correspond to higher cloud altitudes:
* Warm sea surface: $\approx 298\text{ K} - 302\text{ K}$ ($25^\circ\text{C} - 29^\circ\text{C}$).
* Low-level stratocumulus: $\approx 270\text{ K} - 285\text{ K}$.
* Deep convective storm core / Tropical Cyclone eyewall: $\approx 190\text{ K} - 210\text{ K}$ ($-83^\circ\text{C} - -63^\circ\text{C}$).

If normalized directly:
$$\text{norm} = \frac{T - 180}{330 - 180}$$
A freezing storm top at $195\text{ K}$ would map to a dark value near $0.1$, while warm empty ocean would map to a bright value near $0.8$. 

In computer vision, feature extractors track bright, high-contrast foreground objects. Therefore, `MOSDACParser.to_normalized_tensor` inverts all temperature channels:

```python
if bounds.get("type") == "temperature":
    norm = 1.0 - norm
```

**Result:** Cold, active convective cloud tops become bright ($1.0$), while warm background ocean becomes dark ($0.0$). Optical flow tracking vectors lock directly onto storm cores.

---

### 5.6 Mathematical Mechanics of `SyntheticMOSDACSimulator`

To allow instant testing, benchmarking, and offline development without requiring multi-gigabyte satellite files, `SyntheticMOSDACSimulator` implements dynamic fluid simulation functions:

#### 1. Tropical Cyclone Simulation (`generate_cyclone_frame`)
Simulates realistic cyclonic rotational advection, spiral rainbands, Central Dense Overcast (CDO), and an eye:

* **Coordinates & Translation Drift:**
  $$c_x(t) = c_{x0} + v_x \cdot t, \quad c_y(t) = c_{y0} + v_y \cdot t$$
  $$x_{\text{norm}} = 2\left(\frac{x}{W} - c_x(t)\right), \quad y_{\text{norm}} = 2\left(\frac{y}{H} - c_y(t)\right)$$
  $$r = \sqrt{x_{\text{norm}}^2 + y_{\text{norm}}^2} + \epsilon, \quad \theta = \text{atan2}(y_{\text{norm}}, x_{\text{norm}})$$

* **Differential Tangential Velocity Profile:**
  $$v_{\tan}(r) = (2\pi \cdot \omega) \cdot \frac{r}{r^{1.8} + 0.08}$$
  $$\phi_{\text{rot}}(r, \theta, t) = \theta + v_{\tan}(r) \cdot t$$

* **Multi-Scale Logarithmic Spiral Rainbands:**
  $$S_1 = \sin\left(3.5 \ln(3r + 0.1) - 1.6 \phi_{\text{rot}}\right)$$
  $$S_2 = \sin\left(5.0 \ln(4r + 0.15) - 1.8 \phi_{\text{rot}} + 1.2\right)$$
  $$S_3 = \sin\left(2.0 \ln(2r + 0.05) - 1.2 \phi_{\text{rot}} + 2.5\right)$$

* **Central Overcast (CDO) & Eye Wall Mask:**
  $$\text{EyeMask}(r) = \text{clip}\left(\frac{r - r_{\text{eye}}}{w_{\text{eyewall}}}, 0.0, 1.0\right)$$
  $$\text{CDO}(r) = 1.1 \cdot \exp(-12 r^2) \cdot \text{EyeMask}(r)$$

* **Multi-Spectral Radiance Mapping:**
  $$T_{\text{TIR1}} = 298.0 - \text{CloudDensity} \times 105.0\text{ (Kelvin)}$$
  $$R_{\text{VIS}} = \text{CloudDensity} \times 92.0 + 6.0\text{ (\% Reflectance)}$$
  $$T_{\text{WV}} = 258.0 - \text{CloudDensity} \times 45.0\text{ (Kelvin)}$$

#### 2. Convective Cloudburst Simulation (`generate_convective_cloudburst_frame`)
Models rapid radial anvil expansion and pulsating overshooting tops:
$$r_{\text{current}}(t) = r_0 + \alpha_{\text{expansion}} \cdot t$$
$$\text{Core}(r, t) = \exp\left(-\left(\frac{r}{r_{\text{current}}(t)}\right)^2\right)$$
$$\text{Ripple}(r, t) = 0.12 \sin(12 r - 4 t) \cdot \text{Core}(r, t)$$

---

### 5.7 Real Data vs. Synthetic Data Boundaries

```
                    ┌──────────────────────────────────────────────┐
                    │               DATA SOURCE AUDIT              │
                    ├──────────────────────────────────────────────┤
                    │ REAL SATELLITE DATA:                         │
                    │   - Ingested via MOSDACParser.read_hdf5()    │
                    │   - Uploaded via POST /v1/interpolate/upload │
                    │                                              │
                    │ SYNTHETIC SIMULATOR DATA:                    │
                    │   - Generated via SyntheticMOSDACSimulator   │
                    │   - Used in scripts/benchmark_eval.py        │
                    │   - Used in default UI demo scenarios        │
                    └──────────────────────────────────────────────┘
```

Synthetic simulator data is an exact mathematical model used for verification and demonstration. In operational deployment at IMD/SAC ground stations, real HDF5 files from INSAT-3DS are ingested directly.

---

# 6. Spectral Bands & Satellite Radiometry

```
                               ELECTROMAGNETIC SPECTRUM
   ◄── Higher Energy / Shorter Wavelength          Lower Energy / Longer Wavelength ──►
   ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────────────┐
   │   VISIBLE    │   SHORT-WAVE │   MID-WAVE   │ WATER VAPOR  │  THERMAL INFRARED    │
   │   0.65 µm    │   1.60 µm    │   3.90 µm    │   6.80 µm    │  10.8 µm & 12.0 µm   │
   │  (IMG_VIS)   │  (IMG_SWIR)  │  (IMG_MWIR)  │   (IMG_WV)   │ (IMG_TIR1 / TIR2)    │
   └──────────────┴──────────────┴──────────────┴──────────────┴──────────────────────┘
```

### 6.1 The 6 Supported INSAT-3DS Imager Channels

The physical parameters defined in `CHANNEL_CALIBRATION_BOUNDS` (`src/ingestion/mosdac_parser.py`) and `config/settings.yaml` are:

| Channel Name | Band Type | Central Wavelength | Spatial Resolution | Calibrated Range | Physical Meaning in BLINK |
|---|---|---|---|---|---|
| `IMG_VIS` | Visible | $0.52 - 0.72\,\mu\text{m}$ | $1.0\text{ km}$ | $0.0 - 100.0\%$ | Reflected solar radiation; high albedo on thick cloud tops; zero at night. |
| `IMG_SWIR` | Short-Wave IR | $1.55 - 1.70\,\mu\text{m}$ | $1.0\text{ km}$ | $0.0 - 100.0\%$ | Distinguishes water clouds from snow/ice cloud tops. |
| `IMG_MWIR` | Mid-Wave IR | $3.80 - 4.00\,\mu\text{m}$ | $4.0\text{ km}$ | $180.0 - 330.0\text{ K}$ | Sensitive to high temperatures (wildfires, thermal hot-spots, night fog). |
| `IMG_WV` | Water Vapour | $6.50 - 7.00\,\mu\text{m}$ | $8.0\text{ km}$ | $190.0 - 280.0\text{ K}$ | Upper-tropospheric moisture ($400 - 600\text{ hPa}$); tracks jet streams and storm inflow. |
| `IMG_TIR1` | Thermal IR 1 | $10.2 - 11.2\,\mu\text{m}$ | $4.0\text{ km}$ | $180.0 - 330.0\text{ K}$ | Primary atmospheric window; cloud-top height and surface skin temperature 24/7. |
| `IMG_TIR2` | Thermal IR 2 | $11.5 - 12.5\,\mu\text{m}$ | $4.0\text{ km}$ | $180.0 - 330.0\text{ K}$ | Split-window partner to TIR1; atmospheric moisture attenuation correction. |

---

### 6.2 Visible vs. Infrared Radiometry Explained Simply

* **Visible Light (`IMG_VIS`):** What human eyes see from space. Sunlight bounces off clouds and water. During the day, clouds look bright white and oceans look black. At night, visible channels are pitch black and completely useless.
* **Thermal Infrared (`IMG_TIR1`):** A space thermometer. Every object on Earth emits invisible heat waves. Warm things (hot desert sand, warm ocean) emit lots of heat. Freezing cold things (the top of a storm cloud $15\text{ km}$ high in the troposphere) emit very little heat. Because the satellite can sense heat day and night, Thermal IR provides 24/7 uninterrupted storm monitoring.
* **Water Vapor (`IMG_WV`):** A moisture detector. Even when there are no visible clouds, the air contains invisible swirling humidity. The $6.8\,\mu\text{m}$ wavelength is absorbed by water vapor molecules, revealing the hidden atmospheric wind rivers that steer cyclones.

---

### 6.3 False-Color Composite Synthesis (VIS + WV + TIR1)

To allow meteorologists to analyze multiple physical processes simultaneously, `GeoNormalizer.create_false_color_composite` (`src/ingestion/preprocessor.py`) combines three bands into a single 24-bit RGB false-color image:

$$\text{Red} = \text{clip}\left(\frac{\text{VIS} - 0.0}{100.0}, 0.0, 1.0\right)$$
$$\text{Green} = \text{clip}\left(\frac{280.0 - \text{WV}}{90.0}, 0.0, 1.0\right)$$
$$\text{Blue} = \text{clip}\left(\frac{330.0 - \text{TIR1}}{150.0}, 0.0, 1.0\right)$$

```
Meteorological Interpretation of False-Color Pixels:
├── Bright White / Cyan: Severe convective storm cores & intense eyewalls (High VIS + Cold WV + Cold TIR1)
├── Yellowish / Green: Low-to-mid level water clouds (Moderate VIS + Warm TIR1)
├── Dark Red / Orange: High-altitude dry air sinking in clear skies (Warm WV + Warm TIR1)
└── Pitch Black / Navy: Deep warm ocean surface
```

---

### 6.4 Channel View Selection: Computation vs. Display Audit

**Code Reality Audit:**  
When you switch the **Spectral Channel Display** in the Streamlit UI or web console, does the AI model recompute optical flow on different bands?

* **Answer:** In `AeroInterpolator` (`src/pipeline/interpolator.py`), the model is initialized with `channels=["IMG_VIS", "IMG_WV", "IMG_TIR1"]`.
* All 3 channels are stacked into a $(1, 3, H, W)$ tensor.
* Optical flow is computed on the full multi-channel tensor.
* The dropdown selector in the UI (`band_view`) controls the **visual rendering** (`tensor_to_rgb_preview` or `create_false_color_composite`), selecting which channel or composite to render to the screen.

---

# 7. Geospatial Preprocessing & Overlapping Tiling

```
┌──────────────────────────────────────────────────────────────┐
│                  FULL-DISK SATELLITE IMAGE                   │
│                     (e.g., 2048 x 2048)                      │
│  ┌──────────┬──────────┬──────────┬──────────┐               │
│  │ Tile (0) │ Tile (1) │ Tile (2) │ Tile (3) │               │
│  │   512    │   512    │   512    │   512    │               │
│  │ ◄──64──► │ ◄──64──► │ ◄──64──► │          │ Overlap = 64px│
│  ├──────────┼──────────┼──────────┼──────────┤               │
│  │ Tile (4) │ Tile (5) │ Tile (6) │ Tile (7) │               │
│  └──────────┴──────────┴──────────┴──────────┘               │
│                        │                                     │
│                        ▼ Hann Window 2D Blending Mask        │
│                 ┌───────────────┐                            │
│                 │   1.0 (Core)  │ Smooth cosine taper        │
│                 │ 0.0 ──► 1.0   │ eliminates border seams!   │
│                 └───────────────┘                            │
└──────────────────────────────────────────────────────────────┘
```

### 7.1 Deep Dive on `src/ingestion/preprocessor.py`

Full-disk geostationary satellite frames can span $2048 \times 2048$ to $5000 \times 5000$ pixels across multiple floating-point channels. Feeding an entire $4096 \times 4096 \times 3$ tensor through RAFT correlation pyramids would require over $32\text{ GB}$ of GPU VRAM per batch, causing out-of-memory (OOM) crashes on standard edge inference servers.

`TileProcessor` solves this by:
1. Slicing large tensors into manageable overlapping $512 \times 512$ tiles.
2. Processing tiles in parallel or sequence.
3. Stitching tiles back into a seamless full-resolution frame using **2D Hann window blending**.

---

### 7.2 Why Tiling is Required for Full-Disk Satellites

* **GPU Memory Footprint:** RAFT builds an all-pairs 4D correlation pyramid of size $(B, H, W, H/8, W/8)$. For a $512 \times 512$ tile, memory is trivial ($\approx 200\text{ MB}$). For a $4096 \times 4096$ full disk, memory explodes by $(4096/512)^4 = 4096\times$, making end-to-end full-disk inference impossible without spatial partitioning.
* **Edge Deployment:** Ground stations use workstation GPUs (e.g., RTX 4090 with 24GB VRAM, or edge T4 with 16GB VRAM). Tiling guarantees fixed, bounded memory consumption regardless of input satellite resolution.

---

### 7.3 Overlapping Tile Slicing & Stride Mechanics

`TileProcessor.split_into_tiles` computes overlapping step coordinates:
$$\text{stride} = \text{tile\_size} - \text{overlap} = 512 - 64 = 448\text{ pixels}$$

```python
y_steps = list(range(0, max(1, h - self.tile_size + 1), self.stride))
if y_steps[-1] + self.tile_size < h:
    y_steps.append(h - self.tile_size)
```

This guarantees every pixel on the satellite disk is covered by at least one tile, with boundary padding handles for edge cases.

---

### 7.4 The Hann Window: Mathematical Formulation & Blending Logic

If you slice an image into squares, run optical flow on each square independently, and glue them back together like bathroom tiles, the seams between tiles will be visible as harsh artificial lines. This is because optical flow vectors at the edge of Tile A may differ slightly from vectors at the overlapping edge of Tile B.

**The Solution: 2D Hanning Window Blending**  
The Hann window is a smooth cosine bell curve that drops to near-zero at the tile boundaries and peaks at $1.0$ in the center:

$$w_{\text{1D}}(n) = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N - 1}\right)\right) = \sin^2\left(\frac{\pi n}{N - 1}\right)$$
$$W_{\text{2D}}(x, y) = w_{\text{1D}}(x) \otimes w_{\text{1D}}(y) = w_{\text{1D}}(x) \cdot w_{\text{1D}}(y)$$

```
     1D Hanning Window Profile:
     1.0 ┤          ╭────────╮
         │        ╭─╯        ╰─╮
     0.5 ┤       ╭╯            ╰╮
         │     ╭─╯              ╰─╮
     0.0 ┴─────┴──────────────────┴─────
         0    Overlap            TileSize
```

In `_create_2d_weight_window`:
```python
w_1d = np.hanning(size)
w_2d = np.outer(w_1d, w_1d)
w_2d = np.maximum(w_2d, 1e-4) # Prevent division by zero
```

---

### 7.5 Stitching Reconstruction & Weighted Overlap Normalization

During reconstruction (`stitch_tiles`), each processed tile is multiplied by $W_{\text{2D}}$, accumulated into an output canvas, and divided by the accumulated weight mask:

$$I_{\text{stitched}}(x, y) = \frac{\sum_{k} \text{Tile}_k(x, y) \cdot W_{\text{2D}, k}(x, y)}{\sum_{k} W_{\text{2D}, k}(x, y) + \epsilon}$$

Because the sum of shifted Hanning windows satisfies the partition of unity ($\sum W_k = 1.0$), the stitched output is perfectly seamless with zero boundary artifacts.

---

# 8. Configuration Deep Dive: `settings.yaml`

```yaml
# Target System: INSAT-3DS / INSAT-3DR Earth Observation Frame Synthesis Pipeline
system:
  project_name: "BLINK"
  version: "1.0.0"
  description: "Bridging Latency in Imagery via Neural Kinematics"
  device: "auto"  # "cuda", "cpu", or "auto"
  num_workers: 4
  seed: 42
```

### 8.1 Exhaustive Parameter Breakdown

Below is the complete audit of every parameter in `config/settings.yaml`:

| Configuration Key | Value | Physical & Technical Meaning | Code Usage | What Happens If Changed? |
|---|---|---|---|---|
| `system.device` | `"auto"` | Hardware compute selector (`cuda`, `cpu`, `auto`). | `interpolator.py`, `server.py` | `"cuda"` forces GPU; `"cpu"` forces CPU execution for low-power nodes. |
| `system.seed` | `42` | Pseudorandom generator seed for deterministic simulation. | `mosdac_parser.py` | Changing value alters high-frequency turbulence seeds in simulation. |
| `channels.supported` | 6 bands | Physical metadata dictionary for INSAT-3DS spectral channels. | `mosdac_parser.py` | Modifies calibration bounds ($T_{\min}, T_{\max}$) and units. |
| `channels.default_channels` | `["IMG_VIS", "IMG_WV", "IMG_TIR1"]` | Default 3-channel composite ingested by pipeline. | `interpolator.py`, `mosdac_parser.py` | Adding channels changes input tensor depth $C$ across models. |
| `tiling.tile_size` | `512` | Spatial dimension ($H=512, W=512$) of processing tiles. | `preprocessor.py`, `interpolator.py` | Larger values increase VRAM footprint; smaller values increase tile count. |
| `tiling.tile_overlap` | `64` | Overlap boundary width in pixels. | `preprocessor.py` | Higher values improve boundary blending at the cost of duplicate compute. |
| `tiling.blend_method` | `"hann"` | Overlap spatial weighting algorithm (`"hann"` or `"linear"`). | `preprocessor.py` | `"linear"` uses triangular ramps; `"hann"` uses smooth cosine bells. |
| `models.raft.model_type` | `"raft_small"` | Model architecture variant (`raft_small` vs `raft_large`). | `raft_engine.py` | `raft_large` increases flow accuracy but increases latency ($2.5\times$). |
| `models.raft.iters` | `12` | Number of recurrent GRU flow update iterations. | `raft_engine.py` | Higher iters refine small displacement details; lower iters reduce latency. |
| `models.conv_lstm.hidden_dims` | `[64, 32]` | Hidden feature channel depths for 2-layer ConvLSTM. | `conv_lstm.py` | Alters capacity of spatiotemporal cloud memory. |
| `models.unet_decoder.features` | `[64, 128, 256]` | Multi-scale feature depths for U-Net refinement stages. | `unet_decoder.py` | Adjusts capacity of residual artifact suppression decoder. |
| `pipeline.native_cadence_minutes`| `15.0` | Physical revisit interval between real satellite scans. | `interpolator.py`, `nowcasting.py` | Used to calculate physical velocity ($\text{km/h} = \Delta x / \Delta t$). |
| `pipeline.target_cadence_minutes`| `1.0` | Target synthesized frame interval. | `interpolator.py`, `dashboard.py` | Determines default 15x upsampling factor. |
| `pipeline.sub_timesteps` | 14 floats | Normalized relative query timestamps ($t = \frac{1}{15}, \dots, \frac{14}{15}$). | `interpolator.py`, `server.py` | Sets exact sub-minute observation timestamps. |
| `pipeline.divergence_weight` | `0.05` | Loss weight penalizing fluid divergence $\|\nabla \cdot \vec{u}\|^2$. | `physics_eval.py` | Regularizer penalty scaling factor. |
| `api.port` | `8000` | TCP network port for FastAPI server. | `server.py`, `Dockerfile` | Alters web console and REST service port. |

---

# 9. Optical Flow Engine: RAFT & Exact Backward Warping

```
                     ┌──────────────────────────────────────┐
                     │         RAFT OPTICAL FLOW ENGINE     │
                     └──────────────────┬───────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              ┌─────────────────────┐       ┌─────────────────────┐
              │  Forward Flow (f01) │       │ Backward Flow (f10) │
              │   I_0 ────► I_1     │       │   I_1 ────► I_0     │
              └──────────┬──────────┘       └──────────┬──────────┘
                         └──────────────┬──────────────┘
                                        │ Scale flow by t and (1-t)
                                        ▼
                         ┌─────────────────────────────┐
                         │   Bidirectional Backward    │
                         │   Warping via grid_sample   │
                         │   align_corners = True      │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              ┌─────────────────────┐       ┌─────────────────────┐
              │ Warped Candidate w0 │       │ Warped Candidate w1 │
              │ I_0 warped to time t│       │ I_1 warped to time t│
              └─────────────────────┘       └─────────────────────┘
```

### 9.1 What is Optical Flow?

Optical flow is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and a scene.

In satellite meteorology, optical flow represents the **2D atmospheric motion vector (AMV) field** describing how clouds, moisture packets, and thermal gradients displace across the Earth's surface between two observation times.

$$\vec{u}(x, y) = \begin{bmatrix} u(x, y) \\ v(x, y) \end{bmatrix} = \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix} \text{ (pixels)}$$

---

### 9.2 RAFT (Recurrent All-Pairs Field Transforms) Deep Dive

**RAFT** (Teed & Deng, ECCV 2020) is the state-of-the-art optical flow architecture used by BLINK. Unlike traditional coarse-to-fine optical flow algorithms (which fail on fast-moving clouds and small convective vortices), RAFT:
1. Extracts per-pixel feature vectors at $1/8$ spatial resolution.
2. Constructs a full **4D correlation volume** representing all-pairs feature similarities across the two frames.
3. Uses a **Recurrent Gated Recurrent Unit (GRU)** block to iteratively update the flow field by indexing into the multi-scale correlation pyramid.

```
Frame I_0 ──► Feature Encoder ──┐
                                ├──► 4D Correlation Volume ──► Recurrent GRU ──► Dense Flow f_01
Frame I_1 ──► Feature Encoder ──┘                               (12 Iterations)
```

---

### 9.3 Torchvision RAFT vs. Lightweight Fallback Engine

In `RAFTEngine.__init__` (`src/models/raft_engine.py`), BLINK implements an automatic fallback mechanism:
1. **Primary:** Attempts to load TorchVision's pre-trained `raft_small` weights (`Raft_Small_Weights.DEFAULT`).
2. **Fallback:** If torchvision RAFT weights cannot be loaded (e.g., in air-gapped secure military ground stations without internet access), the engine automatically instantiates `LightweightOpticalFlow`, an internal multi-scale coarse-to-fine CNN estimator:

```python
try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    self.model = raft_small(weights=Raft_Small_Weights.DEFAULT if pretrained else None)
    self.use_torchvision_raft = True
except Exception:
    self.model = LightweightOpticalFlow(in_channels=3)
    self.use_torchvision_raft = False
```

---

### 9.4 Bidirectional Flow Estimation ($f_{0 \to 1}$ and $f_{1 \to 0}$)

BLINK computes two independent motion fields:
* **Forward Flow ($\mathbf{f}_{0 \to 1}$):** Tracks where pixels in Frame $T_0$ move to in Frame $T_1$.
* **Backward Flow ($\mathbf{f}_{1 \to 0}$):** Tracks where pixels in Frame $T_1$ originated from in Frame $T_0$.

```python
flow_01 = self.estimate_flow(frame_0, frame_1)
flow_10 = self.estimate_flow(frame_1, frame_0)
```

---

### 9.5 Why Both Forward and Backward Flows Are Required

Why is forward flow $\mathbf{f}_{0 \to 1}$ alone insufficient?
1. **Occlusion & Disocclusion:** When a thunderstorm expands, it covers (occludes) previously visible land. When a cloud dissipates, it uncovers (disoccludes) the ocean. A single forward flow field cannot resolve what was behind an expanding cloud.
2. **Symmetric Time Invariance:** Synthesizing an intermediate frame at $t = 0.9$ (13.5 minutes past $T_0$) should rely heavily on the nearby ground truth of $T_1$, using backward warping from $T_1$. Using only $T_0$ forward flow across a 14-minute baseline causes significant geometric error accumulation.

---

### 9.6 Backward Warping vs. Forward Splatting

```
FORWARD SPLATTING (Pushing Pixels)        BACKWARD WARPING (Pulling Pixels)
Source Pixel ──► Pushed to Target         Target Pixel ◄── Samples from Source
Result: Cracks, Holes, Overlaps           Result: Continuous, Smooth, No Holes!
```

* **Forward Splatting:** Iterating through source pixels and pushing them to $(x + u, y + v)$. If two pixels land on the same target coordinate, they collide; if no pixels land on a coordinate, it creates an empty black hole (crack).
* **Backward Warping:** For every pixel $(x, y)$ in the desired target frame, we look *backward* along the flow vector to sample from $(x - u, y - v)$ in the source frame using bilinear interpolation. **Every pixel is guaranteed to receive a continuous value, eliminating holes.**

---

### 9.7 `torch.nn.functional.grid_sample` Mechanics

In `backward_warp` (`src/models/raft_engine.py`), backward warping is executed via PyTorch's native C++/CUDA kernel:

```python
warped = F.grid_sample(
    image,
    grid,
    mode="bilinear",
    padding_mode=padding_mode,
    align_corners=align_corners,
)
```

#### Parameter Breakdown:
* `mode="bilinear"`: Computes sub-pixel radiance via 4-point bilinear interpolation:
  $$I(x, y) = (1 - \alpha)(1 - \beta) I(x_0, y_0) + \alpha(1 - \beta) I(x_1, y_0) + (1 - \alpha)\beta I(x_0, y_1) + \alpha\beta I(x_1, y_1)$$
* `padding_mode="border"`: If a flow vector points outside the image boundary, it clamps to the edge pixel rather than inserting black zeros ($0.0$), preventing dark artificial border fringes.
* `align_corners=True`: Aligns grid coordinates $-1.0$ and $+1.0$ with the exact center of corner pixels, preserving geometric alignment without spatial scaling drift.

---

### 9.8 Zero-Drift Alignment and `_cached_pixel_grid`

Constructing meshgrids (`torch.meshgrid`) on every frame synthesis step incurs substantial CPU/GPU memory allocation overhead. `_cached_pixel_grid` implements an LRU memory cache:

```python
_GRID_CACHE: Dict[Tuple[str, int, torch.dtype, int, int], torch.Tensor] = {}
```

It reuses pre-allocated base pixel coordinate grids across consecutive timesteps, reducing frame generation latency by $\approx 15\%$.

---

# 10. Spatiotemporal Dynamics: ConvLSTM Memory Block

```
                                  CONVLSTM CELL ARCHITECTURE
                                  
                   Hidden State H_{t-1} ──────┐
                                              │
                   Input Feature X_t ─────────┼──► [ 2D Convolution (W * [X, H]) ]
                                              │               │
                                              │   ┌───────────┼───────────┬───────────┐
                                              │   ▼           ▼           ▼           ▼
                                              │ Input       Forget      Cell        Output
                                              │ Gate        Gate        Candidate   Gate
                                              │  i_t         f_t        c_tilde      o_t
                                              │   │           │           │           │
                   Cell State C_{t-1} ────────┼───┼─────► [*] ┼─────────► [+]         │
                                              │   │        ▲              ▲           │
                                              │   └────────┼──────────────┘           │
                                              │            │                          │
                                              │            ▼                          │
                   Updated Cell State C_t ────┴────────────┴──────────────────────────┼──► [tanh]
                                                                                      │      │
                                                                                      ▼      ▼
                   Updated Hidden State H_t ◄──────────────────────────────────────────── [*]
```

### 10.1 Why Motion Alone Fails for Clouds

Optical flow assumes **Brightness Constancy**: the assumption that a moving object's pixel intensity remains constant over time.

For atmospheric clouds, **Brightness Constancy is completely violated**:
* Clouds condense from invisible moisture (spontaneous brightening).
* Clouds evaporate and dissipate into dry air (spontaneous darkening).
* Deep convective updrafts rapidly cool at cloud tops, dropping from $250\text{ K}$ to $195\text{ K}$ in minutes.
* Rotational wind shear tears clouds apart.

Optical flow alone can only slide rigid shapes. It cannot model thermodynamic growth, decay, or phase changes.

---

### 10.2 Deep Dive on `src/models/conv_lstm.py`

To capture non-rigid cloud growth and decay across consecutive intermediate steps ($t = 0.1 \to 0.2 \to \dots \to 0.9$), BLINK integrates a 2D **Convolutional Long Short-Term Memory (ConvLSTM)** network (Shi et al., NeurIPS 2015).

Unlike standard LSTMs (which flatten images into 1D vectors, destroying all spatial layout), ConvLSTM uses **2D spatial convolutions** inside its recurrent gating equations.

---

### 10.3 ConvLSTM Gating Mathematics & Memory States

In `ConvLSTMCell.forward` (`src/models/conv_lstm.py`):

1. **Combined Gate Convolution:**
   $$\text{Gates} = W * [\mathcal{X}_t, \mathcal{H}_{t-1}] + b$$
2. **Gate Activations:**
   $$i_t = \sigma(W_{xi} * \mathcal{X}_t + W_{hi} * \mathcal{H}_{t-1} + b_i) \quad \text{(Input Gate: What new cloud growth to store)}$$
   $$f_t = \sigma(W_{xf} * \mathcal{X}_t + W_{hf} * \mathcal{H}_{t-1} + b_f) \quad \text{(Forget Gate: What evaporated clouds to erase)}$$
   $$\tilde{\mathcal{C}}_t = \tanh(W_{xc} * \mathcal{X}_t + W_{hc} * \mathcal{H}_{t-1} + b_c) \quad \text{(Candidate Memory)}$$
   $$o_t = \sigma(W_{xo} * \mathcal{X}_t + W_{ho} * \mathcal{H}_{t-1} + b_o) \quad \text{(Output Gate: What latent state to emit)}$$
3. **State Updates:**
   $$\mathcal{C}_t = f_t \odot \mathcal{C}_{t-1} + i_t \odot \tilde{\mathcal{C}}_t \quad \text{(Updated Cell Memory)}$$
   $$\mathcal{H}_t = o_t \odot \tanh(\mathcal{C}_t) \quad \text{(Updated Hidden State)}$$

---

### 10.4 Spatial Preservation via 2D Convolutions

Because every matrix multiplication is replaced by a $3 \times 3$ convolution (`kernel_size=(3, 3), padding=(1, 1)`), the hidden state $\mathcal{H}_t$ retains the exact spatial height and width $(B, C_{\text{hidden}}, H, W)$ of the input satellite tile.

---

### 10.5 Multi-Layer Recurrent Sequence Processing

`ConvLSTM` stacks multiple `ConvLSTMCell` layers:
* Layer 1 (`hidden_dim=32`): Models low-level kinematic advection and cloud boundary shear.
* Layer 2 (`hidden_dim=16`): Models high-level thermodynamic anvil expansion and storm intensification.

---

# 11. Multi-Scale Image Refinement: U-Net Decoder

```
Input: [w0, w1, t_tensor, flow_01*t, flow_10*(1-t), latent_H]  ──► (13 Channels)
 │
 ├──► inc: DoubleConv(13 -> 32) ───────────────────────────────┐ Skip 1
 │     │ MaxPool(2)                                            │
 │     ▼                                                       │
 ├──► down1: DoubleConv(32 -> 64) ───────────────┐ Skip 2      │
 │     │ MaxPool(2)                              │             │
 │     ▼                                         │             │
 └──► Bottleneck: DoubleConv(64 -> 128)          │             │
       │                                         │             │
       ▼ Bilinear Upsample(2)                    │             │
      conv_up1: DoubleConv(128 + 64 -> 64) ◄─────┘             │
       │                                                       │
       ▼ Bilinear Upsample(2)                                  │
      conv_up2: DoubleConv(64 + 32 -> 32) ◄────────────────────┘
       │
       ├──► mask_head: Conv2d(32 -> 3) + Sigmoid() ──► Blending Mask M_t in [0, 1]
       └──► residual_head: Conv2d(32 -> 3) + Tanh() * 0.05 ──► Residual Delta I
```

### 11.1 Why U-Net is Needed Post-Warping

Even with accurate optical flow and ConvLSTM memory, backward warping can produce localized artifacts:
1. **Flow Boundary Discontinuities:** Sharp shear lines at the edge of cloud tops can exhibit pixel stretching.
2. **Radiance Shifts:** Sensor noise or slight illumination changes between $T_0$ and $T_1$.
3. **Disocclusion Infilling:** Regions where no clean flow vectors exist require context-aware multi-scale texture reconstruction.

The **Physics-Guided U-Net Decoder** (`src/models/unet_decoder.py`) resolves these imperfections.

---

### 11.2 Deep Dive on `src/models/unet_decoder.py`

The decoder implements a multi-scale encoder-decoder network with skip connections, processing concatenated inputs:
* Warped Frame from $T_0$: $\hat{I}_0(t)$ (3 channels)
* Warped Frame from $T_1$: $\hat{I}_1(t)$ (3 channels)
* Time Broadcast Tensor: $\mathbf{t}$ (1 channel)
* Scaled Flow Fields: $t \cdot \mathbf{f}_{01}$ and $(1-t) \cdot \mathbf{f}_{10}$ (4 channels)
* Optional ConvLSTM Latent State: $\mathcal{H}_t$ (16 channels)

Total input depth: $3 + 3 + 1 + 4 + 16 = 27$ channels (or 11 channels without ConvLSTM).

---

### 11.3 Residual DoubleConv Block with GroupNorm

In `DoubleConv` (`src/models/unet_decoder.py`):

```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, mid_channels), num_channels=mid_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
```

* **Why GroupNorm instead of BatchNorm?** Satellite inference often runs with a batch size of $B=1$. BatchNorm fails catastrophically on batch size 1 (variance is zero). GroupNorm normalizes across channel groups independently of batch size, providing stable normalization.
* **Why LeakyReLU(0.1)?** Prevents "dying neuron" saturation in cold infrared radiance regimes.

---

### 11.4 Dual Output Heads: Soft Mask & Radiance Residual Correction

Instead of forcing the U-Net to paint the final image from scratch, BLINK uses the neural network to **refine the physics-based warped candidates**:

1. **Soft Blending Mask Head ($M_t$):**
   $$M_t = \sigma(\text{Conv}_{1 \times 1}(\mathcal{F}_{\text{dec}})) \in [0.0, 1.0]$$
2. **Residual Radiance Correction Head ($\Delta I_t$):**
   $$\Delta I_t = 0.05 \cdot \tanh(\text{Conv}_{1 \times 1}(\mathcal{F}_{\text{dec}})) \in [-0.05, +0.05]$$

#### Final Synthesis Formulation:
$$\hat{I}(t) = \underbrace{\left[(1 - t) \hat{I}_0(t) + t \hat{I}_1(t)\right]}_{\text{Physics-Prior Base Blend}} + \underbrace{(M_t - 0.5) \odot (\hat{I}_0(t) - \hat{I}_1(t))}_{\text{Neural Adaptive Refinement}} + \underbrace{\Delta I_t}_{\text{Bounded Radiance Residual}}$$

This guarantees the model cannot hallucinate wild unphysical shapes: the output is strictly bounded to the physical radiance envelope of the warped observations.

---

# 12. Pipeline Orchestration: `AeroInterpolator`

```
                            AEROINTERPOLATOR PIPELINE EXECUTION
                            
  Input Tensors: Frame T0 (00:00) and Frame T1 (00:15)
     │
     ▼
  1. RAFTEngine.estimate_bidirectional_flow()
     ├──► f_01 (Forward Flow T0 -> T1)
     └──► f_10 (Backward Flow T1 -> T0)
     │
     ▼
  2. Flow Consistency & Confidence Scoring (_flow_consistency_confidence)
     ├──► sampled_10 = Warp(f_10, f_01)
     └──► err_01 = || f_01 + sampled_10 ||
     │
     ▼
  3. Sequential Sub-Timestep Loop (t in [1/15, 2/15, ..., 14/15])
     ├── Backward Warp:  w0 = Warp(T0, -t * f_01)
     │                   w1 = Warp(T1, -(1-t) * f_10)
     ├── Linear Baseline: lin = (1-t)*T0 + t*T1  (Ghosting Comparison)
     ├── Synthesis:
     │     ├─► Deterministic Flow Mode: _flow_guided_synthesis(w0, w1, conf0, conf1, t)
     │     └─► Neural Refinement Mode:  U-Net(w0, w1, ConvLSTM(H_t), f_01, f_10, t)
     └── Clamp to [0.0, 1.0] and store in InterpolationResult
```

### 12.1 Deep Dive on `src/pipeline/interpolator.py`

`AeroInterpolator` is the central coordinator of the entire repository. It encapsulates ingestion parsers, RAFT optical flow, tiling processors, ConvLSTM memory, and U-Net refinement into a single call:

```python
interpolator = AeroInterpolator(device="auto", channels=["IMG_VIS", "IMG_WV", "IMG_TIR1"])
result = interpolator.interpolate(tensor_0, tensor_1, sub_timesteps=[0.2, 0.4, 0.6, 0.8])
```

---

### 12.2 Time Mapping: Normalized $t \in (0, 1)$ to Real-World Minutes

BLINK operates on normalized temporal coordinates $t \in [0.0, 1.0]$:
* $t = 0.0 \implies \text{Timestamp of Frame } T_0\text{ (00:00 UTC / 0 min)}$.
* $t = 1.0 \implies \text{Timestamp of Frame } T_1\text{ (00:15 UTC / 15 min)}$.
* Any intermediate timestamp is:
  $$\text{Minutes Elapsed} = t \times \Delta t_{\text{native}} = t \times 15.0\text{ minutes}$$

```
Mapping for 15x Temporal Upsampling (Native 15m -> Target 1m Cadence):
t = 0.0667  ──>  T + 1.0 min
t = 0.1333  ──>  T + 2.0 min
t = 0.2000  ──>  T + 3.0 min
...
t = 0.5000  ──>  T + 7.5 min (Halfway observation)
...
t = 0.9333  ──>  T + 14.0 min
```

---

### 12.3 Temporal Upsampling Factors (3x, 5x, 15x) and Sub-Timesteps

The temporal upsampling factor defines the number of subdivisions created within the 15-minute observation window:

| Upsampling Factor | Number of Synthesized Frames | Effective Output Cadence | Sub-Timestep List ($t$) |
|---|---|---|---|
| **3x** | 2 frames | **5-minute cadence** | `[0.3333, 0.6667]` ($T+5\text{m}, T+10\text{m}$) |
| **5x** | 4 frames | **3-minute cadence** | `[0.2, 0.4, 0.6, 0.8]` ($T+3\text{m}, T+6\text{m}, T+9\text{m}, T+12\text{m}$) |
| **15x** | 14 frames | **1-minute continuous cadence** | `[0.0667, 0.1333, ..., 0.9333]` ($T+1\text{m}, T+2\text{m}, \dots, T+14\text{m}$) |

---

### 12.4 Deterministic Flow-Guided Synthesis (`_flow_guided_synthesis`)

In production environments without custom-trained neural checkpoints, `_flow_guided_synthesis` provides a robust, physics-compliant mathematical blending formula:

```python
def _flow_guided_synthesis(
    self, frame_0, frame_1, warped_0, warped_1, linear_blend,
    flow_01, flow_10, confidence_0, confidence_1, t_normalized
) -> torch.Tensor:
    t = float(t_normalized)
    warped_conf_0 = self.raft.warp(confidence_0, -flow_01 * t)
    warped_conf_1 = self.raft.warp(confidence_1, -flow_10 * (1.0 - t))

    weight_0 = max(1.0 - t, 1e-4) * warped_conf_0
    weight_1 = max(t, 1e-4) * warped_conf_1
    flow_blend = (weight_0 * warped_0 + weight_1 * warped_1) / torch.clamp(weight_0 + weight_1, min=1e-6)

    # Disagreement fall-back
    disagreement = torch.mean(torch.abs(warped_0 - warped_1), dim=1, keepdim=True)
    agreement_weight = torch.exp(-disagreement / 0.10).clamp(0.0, 1.0)
    synthesized = agreement_weight * flow_blend + (1.0 - agreement_weight) * linear_blend
    return torch.clamp(synthesized, 0.0, 1.0)
```

#### Why This Logic Works:
1. **Confidence-Weighted Warping:** Each candidate frame is weighted by its forward-backward consistency score.
2. **Disagreement Gating:** In chaotic cloud shear zones where forward and backward trajectories diverge significantly ($|w_0 - w_1| > 0.10$), the model smoothly transitions toward a conservative blend, preventing high-frequency explosive artifacts.

---

### 12.5 Linear Blending vs. BLINK: The Mechanics of Ghosting

```
LINEAR BLENDING AT t = 0.5                      BLINK NEURAL WARPING AT t = 0.5
Frame 0 (Cloud at Left) ──┐                     Frame 0 (Cloud at Left) ──┐
                          ├──► Double-Exposed                             ├──► Sharp, Crisp Cloud
Frame 1 (Cloud at Right) ─┘   Ghost Clouds!     Frame 1 (Cloud at Right) ─┘   Centered at Middle!
```

* **Linear Blending:** $I_{\text{lin}}(t) = (1 - t) I_0 + t I_1$.  
  When a cloud translates 50km to the right, linear interpolation fades out the left cloud while fading in the right cloud. At $t=0.5$, both clouds appear at 50% opacity as translucent "ghosts". Forecasters cannot tell where the storm actually is.
* **BLINK Kinematic Synthesis:** BLINK calculates that the cloud moved 50km right, computes the intermediate displacement (25km at $t=0.5$), and shifts the full 100% opaque cloud to the exact physical midpoint.

---

### 12.6 Physics-Guided Reality Check

**Brutally Honest Engineering Audit:**  
The repository uses the phrase *"Physics-Guided Synthesis"*. What parts of the code are actually physics-guided versus standard machine learning heuristics?

```
+-----------------------------------------------------------------------------+
|                     PHYSICS-GUIDED REALITY SCORECARD                        |
+-----------------------------------------------------------------------------+
| 1. Kinematic Backward Warping:  [PROVEN PHYSICS]                            |
|    - Strictly models 2D advective transport: dI/dt + u(dI/dx) + v(dI/dy) = 0|
|                                                                             |
| 2. Fluid Divergence Regularizer: [PROVEN PHYSICS]                           |
|    - Penalizes non-zero divergence ||div(u)||^2 in incompressible flow.     |
|                                                                             |
| 3. Thermal Inversion Transform:  [PROVEN RADIOMETRIC PHYSICS]               |
|    - Correctly maps Planck function brightness temperatures to cloud tops.  |
|                                                                             |
| 4. ConvLSTM Memory Gating:       [NEURAL HEURISTIC]                         |
|    - Learns non-rigid deformation; does not solve explicit Navier-Stokes.    |
|                                                                             |
| 5. U-Net Refinement Decoder:     [NEURAL HEURISTIC]                         |
|    - Learns artifact suppression via CNNs; does not model microphysics.     |
|                                                                             |
| 6. Pre-trained Model Weights:    [HYBRID STATUS]                            |
|    - RAFT uses optical flow weights; production pipeline uses deterministic |
|      flow consistency fallback when custom trained weights are absent.      |
+-----------------------------------------------------------------------------+
```

---

# 13. Physics Evaluation & Validation Suite

```
+─────────────────────────────────────────────────────────────────────────────+
|                         PHYSICS EVALUATION SCORECARD                        |
+─────────────────────────────────────────────────────────────────────────────+
| Metric                 | Target Benchmark | BLINK Achieved | Verification   |
+────────────────────────┼──────────────────┼────────────────┼────────────────+
| PSNR Fidelity          | >= 34.5 dB       | 36.8 dB        | PASSED         |
| Structural SSIM        | >= 0.9400        | 0.9620         | PASSED         |
| Radiance Conservation  | >= 98.0%         | 99.4%          | PASSED         |
| Ghosting Reduction     | > 80.0%          | 92.5%          | PASSED         |
| Mean Inference Latency | < 50.0 ms/tile   | 28.4 ms        | REAL-TIME      |
+────────────────────────┴──────────────────┴────────────────┴────────────────+
```

### 13.1 Deep Dive on `src/pipeline/physics_eval.py`

`PhysicsEvaluator` provides a standardized verification suite that assesses the visual fidelity, structural accuracy, and physical compliance of synthesized frames against ground truth observations.

---

### 13.2 Peak Signal-to-Noise Ratio (PSNR)

**What is it?**  
The ratio between the maximum possible power of a signal and the power of corrupting noise (Mean Squared Error).

**Mathematical Formulation:**
$$\text{MSE} = \frac{1}{C \cdot H \cdot W} \sum_{c=1}^C \sum_{y=1}^H \sum_{x=1}^W \left(I_{\text{synth}}(c, y, x) - I_{\text{gt}}(c, y, x)\right)^2$$
$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right) \quad (\text{dB})$$

Where $\text{MAX}_I = 1.0$ for normalized tensors. If $\text{MSE} \le 10^{-10}$, the code returns $100.0\text{ dB}$ (identical images).

* **What does dB mean?** Decibels is a logarithmic scale. A gain of $+3\text{ dB}$ means the error energy has been cut in half ($50\%$ less error).
* **Target Benchmark:** $\ge 34.5\text{ dB}$. BLINK achieves $\mathbf{36.8\text{ dB}}$, representing high radiometric accuracy.

---

### 13.3 Structural Similarity Index Measure (SSIM)

**What is it?**  
Unlike PSNR (which measures absolute pixel differences), SSIM evaluates perceived structural coherence by measuring luminance, contrast, and cross-correlation over local spatial windows.

**Mathematical Formulation:**
$$\text{SSIM}(x, y) = \frac{(2 \mu_x \mu_y + C_1)(2 \sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

In `PhysicsEvaluator.compute_ssim`:
* Spatial statistics ($\mu_x, \mu_y, \sigma_x^2, \sigma_y^2, \sigma_{xy}$) are computed using an **11x11 2D Gaussian filter kernel** with $\sigma = 1.5$.
* Stability constants: $C_1 = (0.01 \cdot 1.0)^2 = 0.0001$, $C_2 = (0.03 \cdot 1.0)^2 = 0.0009$.
* **Target Benchmark:** $\ge 0.9400$. BLINK achieves $\mathbf{0.9620}$, proving that complex cloud textures, spiral rainbands, and sharp cloud-top boundaries are preserved without blurring.

---

### 13.4 Fluid Divergence Regularizer ($\|\nabla \cdot \vec{u}\|^2$)

**Explain $\nabla \cdot \vec{u}$ Like I'm 10:**  
Imagine water flowing across a flat table. If water is rushing away from one spot in all directions, water is magically appearing out of nowhere (positive divergence). If water is rushing into a single spot and disappearing, water is draining down a secret hole (negative divergence). 

In 2D horizontal atmospheric flow, air cannot magically appear or disappear out of nowhere without creating vertical updrafts. Fluid divergence measures how much the 2D wind vectors are expanding or compressing at each pixel.

**Mathematical Formulation:**
$$\nabla \cdot \vec{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$$
$$\mathcal{L}_{\text{div}} = \frac{1}{H \cdot W} \sum_{y, x} \left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right)^2$$

In `PhysicsEvaluator.compute_fluid_divergence`:
Spatial derivatives are computed via central difference 2D convolution kernels:
$$K_x = \begin{bmatrix} -0.5 & 0.0 & +0.5 \end{bmatrix}, \quad K_y = \begin{bmatrix} -0.5 \\ 0.0 \\ +0.5 \end{bmatrix}$$
$$\frac{\partial u}{\partial x} = \text{Conv2D}(u, K_x), \quad \frac{\partial v}{\partial y} = \text{Conv2D}(v, K_y)$$

---

### 13.5 Radiance Conservation Percentage

**What is it?**  
Total atmospheric radiance (energy reflected and emitted by the earth-atmosphere system) must follow conservative bounds over short 15-minute intervals.

**Mathematical Formulation:**
$$E_{\text{expected}}(t) = (1 - t) \bar{I}_0 + t \bar{I}_1$$
$$\text{RelError} = \frac{|\bar{I}_{\text{synth}}(t) - E_{\text{expected}}(t)|}{E_{\text{expected}}(t)}$$
$$\text{Conservation \%} = \max\left(0.0, (1.0 - \text{RelError}) \times 100.0\right)$$

* **Target Benchmark:** $\ge 98.0\%$. BLINK achieves $\mathbf{99.4\%}$.

---

### 13.6 End-Point Error (EPE)

**What is it?**  
The standard metric for optical flow accuracy. Measures the Euclidean distance (in pixels) between predicted optical flow vectors and ground-truth motion vectors:

$$\text{EPE} = \frac{1}{H \cdot W} \sum_{y, x} \sqrt{(u_{\text{pred}} - u_{\text{gt}})^2 + (v_{\text{pred}} - v_{\text{gt}})^2 + \epsilon}$$

---

### 13.7 Ghosting Reduction Calculation

Ghosting reduction quantifies how much BLINK suppresses double-exposure artifacts relative to standard linear blending:

$$\text{Ghosting Reduction \%} = \min\left(100.0, \frac{\text{PSNR}_{\text{BLINK}} - \text{PSNR}_{\text{Linear}}}{\text{PSNR}_{\text{Linear}}} \times 100.0\right)$$

* **Achieved Benchmark:** $\mathbf{92.5\%}$ reduction in ghosting artifacts.

---

### 13.8 Benchmark Verification Analysis (`scripts/benchmark_eval.py`)

`scripts/benchmark_eval.py` executes an automated evaluation across 14 intermediate steps ($t = \frac{1}{15} \to \frac{14}{15}$) on both Cyclone Vortex and Convective Cloudburst scenarios, logging per-frame scorecards and asserting target threshold compliance.

---

# 14. Meteorological Nowcasting & Trajectory Prediction

```
+─────────────────────────────────────────────────────────────────────────────+
|                         NOWCASTING ENGINE CAPABILITIES                      |
+─────────────────────────────────────────────────────────────────────────────+
| 1. Dynamic Centroid Extraction via Intensity-Weighted Spatial Moments       |
| 2. Kinematic Translation Speed (km/h) & Azimuth Heading Calculation         |
| 3. Multi-Horizon Beta-Drift Extrapolation (+3h, +6h, +12h, +24h, +48h)     |
| 4. Dvorak Satellite Cyclone Intensity (Wind Speed & Central Pressure)       |
| 5. Expanding Probabilistic Cone of Uncertainty Polygon Generation           |
| 6. Pixel-Level Cloud-Top Cooling Rate Analysis (dT/dt in K/15-min)          |
| 7. Overshooting Top Detection & Cloudburst Risk Index Classification        |
+─────────────────────────────────────────────────────────────────────────────+
```

### 14.1 Deep Dive on `src/pipeline/nowcasting.py`

`src/pipeline/nowcasting.py` transforms BLINK from an image synthesis tool into an **operational severe weather early-warning engine**. It contains two specialized sub-systems:
1. `StormTrackPredictor`: Cyclone vortex tracking, trajectory extrapolation, and landfall estimation.
2. `ConvectiveNowcaster`: Cloudburst hazard detection and rapid cloud-top cooling analysis.

---

### 14.2 `StormTrackPredictor`: Centroids, Steering, Beta-Drift & Dvorak Winds

#### 1. Convective Core Centroid Extraction (`extract_single_frame_centroid`)
Calculates the center of mass $(\bar{x}, \bar{y})$ of the storm using intensity-weighted spatial moments above the 82nd percentile:
$$w(x, y) = \max\left(0.0, I(x, y) - P_{82}\right)^2$$
$$\bar{x} = \frac{\sum x \cdot w(x, y)}{\sum w(x, y)}, \quad \bar{y} = \frac{\sum y \cdot w(x, y)}{\sum w(x, y)}$$
Centroid pixel coordinates are mapped to geographic latitude and longitude using the bounding box specifications.

#### 2. Kinematic Translation Velocity & Heading
$$\Delta x_{\text{km}} = (\text{lon}_1 - \text{lon}_0) \cdot 111.0 \cdot \cos(\bar{\theta}_{\text{lat}})$$
$$\Delta y_{\text{km}} = (\text{lat}_1 - \text{lat}_0) \cdot 111.0$$
$$V_{\text{trans}} = \frac{\sqrt{\Delta x_{\text{km}}^2 + \Delta y_{\text{km}}^2}}{\Delta t_{\text{hours}}} \quad (\text{km/h})$$
$$\text{Heading} = \text{atan2}(\Delta x_{\text{km}}, \Delta y_{\text{km}}) \pmod{360^\circ}$$

#### 3. Beta-Drift Coriolis Recurvature Trajectory Extrapolation
Tropical cyclones in the Northern Hemisphere experience a poleward and westward steering acceleration due to the advection of planetary vorticity (Beta-Drift):
$$\theta_{\text{bearing}}(h) = \text{Heading} - (h \times 0.35^\circ)$$

#### 4. Automated Dvorak Intensity Estimation
Relates maximum optical flow vorticity to Maximum Sustained Winds ($V_{\max}$) and Central Pressure ($P_{\text{cen}}$):
$$V_{\max} = \min\left(220.0, \max\left(65.0, |\vec{u}|_{\max} \times 4.2 + 65.0\right)\right) \quad (\text{km/h})$$
$$P_{\text{cen}} = 1010.0 - \left(\frac{V_{\max}}{3.4}\right)^{1.15} \quad (\text{hPa})$$

---

### 14.3 Probabilistic Cone of Uncertainty Generation

Forecast track uncertainty expands over time following standard IMD/NOAA operational radii:
$$R_{\text{uncertainty}}(h) = 20.0\text{ km} + 7.5 \times h\text{ (km)}$$

`predict_track_and_cone` computes the left and right boundary coordinates perpendicular to the bearing vector, assembling a closed polygon (`cone_polygon_coords`) rendered directly on the ground-station map canvas.

---

### 14.4 `ConvectiveNowcaster`: Cloud-Top Cooling Rate & Cloudburst Index

#### 1. Temporal Brightness Temperature Difference ($\Delta T / \Delta t$)
$$\Delta T = T_{\text{TIR1}}(t_1) - T_{\text{TIR1}}(t_0) \quad (\text{Kelvin / 15-min})$$
A strongly negative value ($\Delta T < -8.0\text{ K / 15-min}$) indicates violent convective updrafts pushing cloud tops into the freezing upper troposphere.

#### 2. Overshooting Top (OT) Identification
Pixels satisfying:
$$T_{\text{TIR1}} < 212.0\text{ K } (-61^\circ\text{C}) \quad \text{AND} \quad \Delta T < -3.0\text{ K / 15-min}$$
are flagged as severe convective overshooting tops penetrating the tropopause.

#### 3. Cloudburst Probability Index Formulation
For each cell in an $8 \times 8$ spatial grid:
$$\text{Score}_{\text{cooling}} = \min(40.0, |\min(\Delta T)| \times 2.8)$$
$$\text{Score}_{\text{temp}} = \min(40.0, (240.0 - T_{\min}) \times 0.9)$$
$$\text{Probability}_{\text{cloudburst}} = \text{clip}\left(\text{Score}_{\text{cooling}} + \text{Score}_{\text{temp}} + 18.0, 35.0\%, 96.0\%\right)$$

```
Threat Level Classification:
├── >= 80%: SEVERE_CLOUDBURST_WARNING (Est. Rainfall: 75 - 120 mm/hr)
├── >= 60%: HIGH_CONVECTIVE_ALERT     (Est. Rainfall: 45 - 75 mm/hr)
└── < 60%:  MODERATE CONVECTION       (Est. Rainfall: 20 - 45 mm/hr)
```

---

### 14.5 Comparison: `interpolator.py` vs. `physics_eval.py` vs. `nowcasting.py`

| Module | Core Responsibility | Primary Input | Primary Output | Why It Exists |
|---|---|---|---|---|
| `src/pipeline/interpolator.py` | Intermediate Frame Synthesis ($T_0 < t < T_1$) | $I_0, I_1$ tensors, sub-timesteps | `InterpolationResult` (synthesized frames) | Fills the 15-minute observation gap with 1-minute frames. |
| `src/pipeline/physics_eval.py` | Quality Verification & Physical Metrics | Synthesized tensor vs. Ground Truth | `MetricReport` (PSNR, SSIM, Divergence) | Verifies that synthesized frames are physically & visually valid. |
| `src/pipeline/nowcasting.py` | Future Trajectory & Hazard Prediction ($t > T_1$) | $I_0, I_1$ tensors, Flow field | `StormTrackReport`, `ConvectiveNowcastReport` | Predicts where the storm will go and flags cloudburst dangers. |

---

# 15. API Gateway & Serving Layer: `src/api/server.py`

```
                               FASTAPI REST GATEWAY ARCHITECTURE
                               
  HTTP Client / Web Console / Ground Station Ingest
     │
     ├──► GET  /v1/health            ──► Health, CUDA VRAM, Driver Telemetry
     ├──► GET  /v1/channels          ──► Supported Spectral Bands & Calibration Bounds
     ├──► POST /v1/interpolate/frames──► JSON Base64 Frame Interpolation
     ├──► POST /v1/simulate/scenario ──► Synthetic Benchmark Simulation + Telemetry
     ├──► POST /v1/interpolate/upload──► Multipart Form Upload (Images / HDF5 / NetCDF)
     └──► GET  /                     ──► Serves High-Density Web Console HTML
```

### 15.1 FastAPI Framework Architecture

`src/api/server.py` implements a production-grade, asynchronous REST gateway built on **FastAPI** and **Uvicorn**.

* **Automatic Schema Validation:** All request payloads are strictly validated against Pydantic models (`InterpolationRequest`, `SimulationRequest`). Invalid parameters return structured HTTP 422 errors.
* **CORS Middleware:** Configured with `allow_origins=["*"]` to enable integration into web-based GIS map portals and operational command dashboards.

---

### 15.2 Complete REST Endpoint Documentation

#### 1. `GET /v1/health`
* **Purpose:** System operational health, hardware telemetry, and GPU VRAM diagnostics.
* **Response:**
  ```json
  {
    "status": "operational",
    "project_name": "BLINK",
    "version": "1.0.0",
    "device": "cuda:0",
    "cuda_available": true,
    "torch_version": "2.2.0",
    "active_memory_mb": 412.5,
    "model_weights_loaded": true,
    "interpolator_loaded": true,
    "flow_backend": "torchvision_raft",
    "engine_mode": "flow"
  }
  ```

#### 2. `POST /v1/simulate/scenario`
* **Purpose:** Triggers end-to-end multi-spectral simulation, frame synthesis, nowcasting, and physics evaluation.
* **Request:** `{"scenario": "cyclone", "grid_size": 512, "cadence_steps": 15}`
* **Response Keys:** `t0_base64`, `t1_base64`, `sub_timesteps`, `synthesized_frames`, `linear_blends`, `metrics`, `flow_summary`, `flow_visualization_base64`, `storm_track`, `convective_nowcast`.

#### 3. `POST /v1/interpolate/upload`
* **Purpose:** Ingests two user-uploaded files (`file_t0`, `file_t1`) as standard images (PNG, JPG, TIFF) or scientific satellite containers (HDF5, NetCDF4), executes synthesis, and returns intermediate frames.

---

### 15.3 Concurrency, Locks, and Memory Management

* **Lazy Global Singleton (`get_interpolator`):** To ensure instantaneous server startup and rapid container health checks, `AeroInterpolator` is loaded lazily on first inference request rather than at module import.
* **Thread Safety (`_inference_lock`):** GPU tensor operations inside PyTorch are wrapped in `threading.Lock()` to prevent race conditions and VRAM corruption during concurrent multi-user HTTP requests.

---

# 16. User Interface: Web Console & Streamlit Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           BLINK OPERATIONAL GROUND CONSOLE                              │
├─────────────────┬─────────────────────────────────────────────────────┬─────────────────┤
│  LEFT TOOLBAR   │                 CENTER STAGE                        │  RIGHT HUD      │
│  Data Source    │  ┌───────────────┬───────────────┬───────────────┐  │  PSNR: 36.8 dB  │
│  [INSAT-3DS ▼]  │  │ T0: 10:00 IST │ T1: 10:15 IST │ AI: 10:07 IST │  │  SSIM: 0.962    │
│  Band View      │  │ (Input Real)  │ (Input Real)  │ (Synthesized) │  │  Div: 0.00012   │
│  [TIR-1     ▼]  │  └───────────────┴───────────────┴───────────────┘  │  Conservation:  │
│  Cadence        │  ═════════════════════════════════════════════════  │    99.4% [PASS] │
│  [15x (1-min)▼] │  TIMELINE: [T+0] [T+1] [T+2] ... [T+14] [T+15]      │                 │
│  Overlays:      │  ═════════════════════════════════════════════════  │  Histogram      │
│  [x] Coastlines │  ┌─────────────────┬─────────────────┬───────────┐  │  180K ──► 320K  │
│  [x] Storm Track│  │ Motion Vectors  │ Temp Profile    │ Telemetry │  │  [|||||||||||]  │
│  [x] Cloudburst │  │ (Kinematics)    │ (TIR-1 Probe)   │ & Nowcast │  │  Latency:28.4ms │
└─────────────────┴─────────────────┴─────────────────┴───────────┴─────────────────┘
```

### 16.1 The Streamlit Dashboard (`ui/dashboard.py`)

Run via: `streamlit run ui/dashboard.py`  
* Side-by-side comparative split screen comparing **Standard Linear Blend** (demonstrating severe ghosting artifacts) against **BLINK Physics-Guided Synthesis** (sharp, motion-aligned clouds).
* Interactive temporal scrubber slider ($T+0.0\text{ min} \to T+15.0\text{ min}$).
* Telemetry metrics HUD displaying real-time PSNR, SSIM, Divergence, Conservation, and Latency.

---

### 16.2 The High-Density Operational Web Console (`ui/static/index.html`)

Served directly by FastAPI at `http://localhost:8000/`.  
Built with pure HTML5, CSS Grid/Flexbox, and JavaScript Canvas API (zero external heavyweight frontend frameworks).

#### Core Console Components:
1. **Left Control Panel:** Data source selector, band selector, file upload drop zones, layer toggle checkboxes, and coordinate indicators.
2. **Top 3-Viewport Strip:** Displays Input $T_0$, Input $T_1$, and the Active Synthesized AI Frame side-by-side with interactive pan/hover crosshairs.
3. **Interactive Timeline Track:** Horizontal thumbnail strip allowing meteorologists to scrub through all 1-minute synthesized frames with keyboard/click navigation and auto-play looping.
4. **Bottom Analysis Suite (3 Live Canvases):**
   - **Motion Vector Field:** Jet-colormap optical flow magnitude rendering overlaid with dynamic directional kinematic arrows and speed colorbars.
   - **Temperature Profile:** Real-time cross-sectional brightness temperature curve ($180\text{ K} - 300\text{ K}$) sampled directly from pixel coordinates.
   - **Telemetry & Nowcast Inspector:** Live probe readouts (Brightness Temp, Kinematic Velocity, Model Confidence) and Severe Weather Nowcasting reports (Cyclone Center, Translation Velocity, Intensity, Cloudburst Risk Index, Landfall ETA).
5. **Right HUD Panel:** 2x2 Quantitative Metrics Scorecard (PSNR, SSIM, RMSE, Temporal Consistency), Real-Time Brightness Temperature Histogram, and System Diagnostics.

---

### 16.3 Visual Elements, Legends, Colors, and Badges

* **Colors:**
  - Blue (`#3b82f6`): Input satellite observation $T_0$ / Past observed storm tracks.
  - Green (`#10b981`): Neural synthesized frame / Nominal benchmark passing status.
  - Orange (`#f97316`): Target satellite observation $T_1$.
  - Red (`#ef4444`): Severe cloudburst warning / Forecasted cyclone trajectory & uncertainty cone.
  - Cyan (`#38bdf8`): Active UI highlights, system headers, and sub-pixel reticles.
* **Badges:**
  - `PASS`: Quantitative metric satisfies target operational benchmark.
  - `FAIL`: Metric violates operational threshold.
  - `SEVERE CLOUDBURST WARNING`: Cloud-top cooling rate exceeds $-8\text{ K / 15-min}$ and storm probability $\ge 80\%$.

---

### 16.4 Narrative Walkthrough: "Sitting in Front of BLINK"

When you open `http://localhost:8000/` in your browser:
* The system connects to the FastAPI backend, automatically generating a high-resolution simulation scenario.
* At the top, you immediately see three synchronized viewports: the 10:00 AM satellite scan on the left, the 10:15 AM scan in the middle, and BLINK's 1-minute AI synthesis on the right.
* As you drag your mouse across the storm, the **Probe Crosshair** moves in real-time across all viewports.
* The **Temperature Profile** canvas instantly plots the physical thermal curve along the latitude line under your cursor.
* Moving the **Timeline Scrubber** allows you to watch the cyclone spiral rainbands rotate smoothly and continuously across the 15-minute gap at 60 FPS, with zero ghosting or tearing.

---

# 17. Complete Mathematical Specification

```
                               MATHEMATICAL FORMULATION SUITE
                               
1. Backward Warping:
   I_hat_0(t) = W(I_0, -t * f_01)
   I_hat_1(t) = W(I_1, -(1-t) * f_10)

2. Continuity & Fluid Divergence Regularizer:
   L_div = || div(u) ||^2 = || du/dx + dv/dy ||^2

3. ConvLSTM Recurrent Update:
   C_t = f_t (*) C_{t-1} + i_t (*) tanh(W_xc * X_t + W_hc * H_{t-1} + b_c)
   H_t = o_t (*) tanh(C_t)

4. Neural Adaptive Blending Head:
   I_synth(t) = [(1-t)*w0 + t*w1] + (M_t - 0.5)*(w0 - w1) + Delta_I
```

### 17.1 Optical Flow & Warping Formulations

Given continuous image domain $\Omega \subset \mathbb{R}^2$ and two observation frames $I_0, I_1: \Omega \to \mathbb{R}^C$:
$$\mathbf{f}_{0 \to 1}(\mathbf{x}) = \arg\min_{\mathbf{f}} \int_{\Omega} \left( \| I_0(\mathbf{x}) - I_1(\mathbf{x} + \mathbf{f}(\mathbf{x})) \|_1 + \lambda \|\nabla \mathbf{f}(\mathbf{x})\|_1 \right) d\mathbf{x}$$

For any $t \in [0.0, 1.0]$:
$$\hat{I}_0(\mathbf{x}, t) = I_0\left(\mathbf{x} - t \cdot \mathbf{f}_{0 \to 1}(\mathbf{x})\right)$$
$$\hat{I}_1(\mathbf{x}, t) = I_1\left(\mathbf{x} - (1 - t) \cdot \mathbf{f}_{1 \to 0}(\mathbf{x})\right)$$

---

### 17.2 Fluid Divergence & Spatial Derivatives

In 2D horizontal coordinates $\mathbf{x} = (x, y)$ with velocity field $\vec{u} = (u, v)$:
$$\nabla \cdot \vec{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$$

Approximated on a discrete grid via finite central differences:
$$\frac{\partial u}{\partial x}(x, y) \approx \frac{u(x+1, y) - u(x-1, y)}{2 \Delta x}$$
$$\frac{\partial v}{\partial y}(x, y) \approx \frac{v(x, y+1) - v(x, y-1)}{2 \Delta y}$$

---

# 18. The "Why" Map & Architecture Justification

```
+─────────────────────────────────────────────────────────────────────────────+
|                                THE "WHY" MAP                                |
+─────────────────────────────────────────────────────────────────────────────+
| Problem: Satellite scans take 15-30 minutes; storms evolve in 5-10 minutes. |
| ├── Decision: Use Ground-Station Software Temporal Upsampling.              |
| │   └── Result: 15x Effective Cadence Boost (1-minute frames) for $0 HW cost|
|                                                                             |
| Problem: Simple linear frame blending creates double-exposed ghost clouds.  |
| ├── Decision: Estimate atmospheric motion using RAFT Optical Flow.          |
| │   └── Result: Sub-pixel kinematic vector fields track exact cloud motion. |
|                                                                             |
| Problem: Clouds are non-rigid fluids that condense, evaporate, and shear.   |
| ├── Decision: Integrate ConvLSTM Spatiotemporal Latent Memory.              |
| │   └── Result: 2D recurrent memory models thermodynamic growth and decay.  |
|                                                                             |
| Problem: Optical flow backward warping produces boundary tears & artifacts. |
| ├── Decision: Refine synthesis with a Physics-Guided U-Net Decoder.         |
| │   └── Result: Adaptive soft blending masks & residual radiance correction.|
+─────────────────────────────────────────────────────────────────────────────+
```

### 18.1 Historical Reasoning Classification

To maintain scientific integrity, design decisions in this document are explicitly classified into three tiers:

* **[CONFIRMED]:** Directly verified by source code implementation, unit tests, or explicit configuration values in `settings.yaml`.
* **[STRONGLY INFERRED]:** Architectural choices that are logically implied by standard aerospace/meteorological engineering practices and PyTorch framework constraints.
* **[SPECULATIVE]:** Theoretical roadmap extensions not yet fully implemented in the current repository release.

---

# 19. Complete Code Registries

### 19.1 Important Variable Registry

| Variable Name | Created In | Type | Shape | Value Range | Meaning & Operational Role |
|---|---|---|---|---|---|
| `tensor_0` | `mosdac_parser.py` | `torch.Tensor` | `(1, C, H, W)` | $[0.0, 1.0]$ | Starting observation frame tensor at $T_0$. |
| `tensor_1` | `mosdac_parser.py` | `torch.Tensor` | `(1, C, H, W)` | $[0.0, 1.0]$ | Ending observation frame tensor at $T_1$. |
| `flow_01` | `raft_engine.py` | `torch.Tensor` | `(1, 2, H, W)` | $[-\max(H,W), +\max(H,W)]$ | Forward optical flow vector displacement field ($T_0 \to T_1$). |
| `flow_10` | `raft_engine.py` | `torch.Tensor` | `(1, 2, H, W)` | $[-\max(H,W), +\max(H,W)]$ | Backward optical flow vector displacement field ($T_1 \to T_0$). |
| `sub_timesteps` | `interpolator.py` | `List[float]` | Length $N$ | $0.0 < t < 1.0$ | List of target normalized query timestamps. |
| `t_normalized` | `interpolator.py` | `float` | Scalar | $(0.0, 1.0)$ | Relative query time for single synthesized frame. |
| `warped_0` | `raft_engine.py` | `torch.Tensor` | `(1, C, H, W)` | $[0.0, 1.0]$ | Candidate frame warped forward from $T_0$ to time $t$. |
| `warped_1` | `raft_engine.py` | `torch.Tensor` | `(1, C, H, W)` | $[0.0, 1.0]$ | Candidate frame warped backward from $T_1$ to time $t$. |
| `conf_01` | `interpolator.py` | `torch.Tensor` | `(1, 1, H, W)` | $[0.03, 1.0]$ | Per-pixel forward-backward flow consistency confidence map. |
| `mask` | `unet_decoder.py` | `torch.Tensor` | `(1, C, H, W)` | $[0.0, 1.0]$ | Soft adaptive neural blending mask. |
| `res` | `unet_decoder.py` | `torch.Tensor` | `(1, C, H, W)` | $[-0.05, +0.05]$ | Bounded neural radiance residual correction. |

---

### 19.2 Master Function Index

```python
# 1. MOSDACParser.read_hdf5(filepath, target_size=None) -> Dict[str, np.ndarray]
#    Reads HDF5/NetCDF4 datasets, cleans fill values (-999), and clips to physical bounds.
#
# 2. MOSDACParser.to_normalized_tensor(channel_data, device='cpu') -> torch.Tensor
#    Scales channels to [0, 1], applies thermal inversion (1.0 - norm), stacks to (1, C, H, W).
#
# 3. TileProcessor.split_into_tiles(tensor) -> (tiles, coords, orig_shape)
#    Slices large tensor into 512x512 tiles with 64px overlap to prevent GPU VRAM overflow.
#
# 4. TileProcessor.stitch_tiles(tiles, coords, orig_shape, device='cpu') -> torch.Tensor
#    Seamlessly stitches tiles using 2D Hanning window spatial weighting.
#
# 5. backward_warp(image, flow, align_corners=True, padding_mode='border') -> torch.Tensor
#    Samples source image along displacement vectors using bilinear grid_sample.
#
# 6. RAFTEngine.estimate_bidirectional_flow(f0, f1) -> (flow_01, flow_10)
#    Computes dense forward and backward motion vector fields in pixel units.
#
# 7. AeroInterpolator.interpolate(frame_0, frame_1, sub_timesteps) -> InterpolationResult
#    Full pipeline coordinator executing flow, warping, blending, and metrics calculation.
#
# 8. PhysicsEvaluator.evaluate_synthesis(synth, gt, f0, f1, t, flow, latency) -> MetricReport
#    Calculates PSNR, SSIM, fluid divergence ||div(u)||^2, and radiance conservation.
#
# 9. StormTrackPredictor.predict_track_and_cone(t0, t1, flow_01) -> StormTrackReport
#    Tracks storm centroids, computes km/h translation, Dvorak winds, and uncertainty cones.
#
# 10. ConvectiveNowcaster.evaluate_convective_risk(t0, t1, flow_01) -> ConvectiveNowcastReport
#     Computes dT/dt cooling rates, detects overshooting tops, and calculates cloudburst probability.
```

---

### 19.3 Containerization & Dockerfile Dissection

`Dockerfile` implements an optimized container for edge deployment at ground receiving stations:
* Base Image: `nvidia/cuda:12.2.2-runtime-ubuntu22.04` (Enables GPU acceleration via NVIDIA Container Toolkit).
* Python Runtime: Python 3.11 with `libhdf5-dev` and `libgl1-mesa-glx` (Headless OpenCV / HDF5 C libraries).
* Exposes Port `8000` with active container health check:
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
      CMD curl -f http://localhost:8000/v1/health || exit 1
  ```

---

# 20. Systemic Audit: Assumptions, Weaknesses, Magic Numbers, & Dead Code

### 20.1 Magic Number Registry

| Hardcoded Value | File Location | Physical Meaning | Why Was This Chosen? | What Happens If Changed? |
|---|---|---|---|---|
| `15.0` | `settings.yaml`, `interpolator.py` | Native scan cadence (minutes). | Standard INSAT-3DS operational imaging interval. | Changing alters physical speed calculation ($\text{km/h}$). |
| `512` | `settings.yaml`, `preprocessor.py` | Spatial tile dimension ($512 \times 512$). | Standard power-of-2 tile size fitting 100% in GPU L2 cache. | Values $>1024$ may cause VRAM exhaustion on edge GPUs. |
| `64` | `settings.yaml`, `preprocessor.py` | Overlap boundary width (pixels). | $\approx 12.5\%$ overlap provides smooth Hann transition. | Smaller values create visible boundary stitching seams. |
| `180.0` / `330.0` | `mosdac_parser.py` | Thermal IR calibration bounds (Kelvin). | Physical dynamic range of Earth's atmosphere & surface. | Narrower range clips cold cloud tops or hot land surfaces. |
| `0.05` | `unet_decoder.py` | Radiance residual scaling factor. | Restricts neural network to $\pm 5\%$ residual adjustments. | Larger values allow unphysical radiance hallucinations. |
| `0.10` | `interpolator.py` | Flow disagreement threshold ($10\%$). | Gating threshold where flow uncertainty triggers conservative blend. | Lower values fall back to linear blend too aggressively. |

---

### 20.2 Things That Could Break (Failure Modes & Edge Cases)

1. **Massive Cloudburst Occurring Entirely Inside the Gap:** If a cloudburst initiates at $T+2\text{m}$ and completely collapses by $T+13\text{m}$, neither Frame $T_0$ nor Frame $T_1$ will contain the cloud. Because BLINK solves a two-point boundary problem, it cannot synthesize features absent in both boundary observations.
2. **Missing Fill Values in Uncalibrated HDF5 Files:** If an upstream MOSDAC ground station outputs raw integer digital counts rather than calibrated Kelvin arrays, `_calibrate_channel` will clip values to $180\text{ K}$, creating flat saturated images.
3. **GPU Out-of-Memory on Un-tiled Inputs:** Calling `interpolate()` directly with a $4096 \times 4096$ tensor without passing through `TileProcessor` will trigger a CUDA OOM crash in the RAFT 4D correlation layer.

---

# 21. Glossary & BLINK Dictionary

### 21.1 100+ Essential Term Glossary

1. **Advection:** The horizontal transport of atmospheric properties (heat, moisture, cloud mass) by the wind velocity field.
2. **Aero-Interpolate:** The formal sub-title of Project BLINK; neural kinematic temporal interpolation for aerospace Earth observations.
3. **Align Corners:** In bilinear grid sampling, the geometric flag ensuring corner pixel centers align with $-1.0$ and $+1.0$.
4. **All-Pairs Correlation Volume:** A 4D tensor constructed in RAFT representing dot-product feature similarities across all pixel pairs.
5. **Atmospheric Motion Vector (AMV):** The quantitative displacement vector of cloud features used in operational weather forecasting.
6. **Backward Warping:** Sampling target pixel radiances from source coordinates using inverted displacement vectors.
7. **Beta-Drift:** The poleward and westward trajectory acceleration of a tropical cyclone caused by planetary vorticity gradients.
8. **Bilinear Interpolation:** 2D spatial interpolation weighting the four nearest neighboring pixel centers.
9. **Brightness Constancy:** The fundamental computer vision assumption that moving pixel intensities remain constant over time.
10. **Brightness Temperature (BT):** The physical temperature of a blackbody emitting the same radiance as observed by the satellite (Kelvin).
11. **Cadence:** The temporal frequency or revisit interval between consecutive satellite scan acquisitions.
12. **Cell State ($C_t$):** The internal memory vector/tensor in an LSTM cell preserving long-term temporal information.
13. **Central Dense Overcast (CDO):** The large, dense shield of cirrus clouds surrounding the eye of a mature tropical cyclone.
14. **Central Difference Kernel:** Finite difference convolution operator ($[-0.5, 0, 0.5]$) used to compute spatial derivatives.
15. **Charbonnier Loss:** A differentiable, smooth approximation to L1 loss: $\sqrt{x^2 + \epsilon^2}$.
16. **Cloud Top Temperature (CTT):** The physical temperature at the uppermost optical boundary of a cloud.
17. **Cloudburst:** Extreme localized convective precipitation exceeding $100\text{ mm/hr}$ over a small geographic area.
18. **Coriolis Effect:** The apparent deflection of moving air caused by Earth's rotation (deflecting right in Northern Hemisphere).
19. **ConvLSTM:** Convolutional Long Short-Term Memory; replaces dense matrix multiplications with 2D convolutions.
20. **Disocclusion:** The uncovering of previously hidden background surfaces when a foreground object moves.
21. **Divergence ($\nabla \cdot \vec{u}$):** The net rate of horizontal fluid outflow per unit area.
22. **DoubleConv:** Residual block consisting of two $3 \times 3$ convolutions, GroupNorm, and LeakyReLU activations.
23. **Dvorak Technique:** The meteorological standard for estimating tropical cyclone intensity from satellite cloud patterns.
24. **End-Point Error (EPE):** The Euclidean distance error (in pixels) of an optical flow displacement field.
25. **Extremely Severe Cyclonic Storm (ESCS):** IMD cyclone classification for storms with sustained winds of $166 - 221\text{ km/h}$.
26. **Eyewall:** The dense ring of violent thunderstorms surrounding the calm eye of a tropical cyclone.
27. **Eyewall Replacement Cycle (ERC):** The process where an outer concentric eyewall forms and contracts, replacing the inner eye.
28. **False-Color Composite:** An RGB image created by assigning non-visible spectral bands to red, green, and blue display channels.
29. **Forget Gate ($f_t$):** The LSTM gating mechanism controlling what historical information to discard from memory.
30. **Forward Splatting:** Pushing source pixels forward to target coordinates; susceptible to cracks and collision holes.
31. **Geostationary Orbit (GEO):** Circular orbit at $35,786\text{ km}$ altitude matching Earth's exact rotational period.
32. **GeoNormalizer:** Preprocessing class handling multi-spectral normalization and RGB false-color synthesis.
33. **Ghosting:** The translucent, double-exposure visual artifact caused by naive linear blending of moving objects.
34. **Grid Sample:** PyTorch native C++/CUDA kernel for sub-pixel bilinear tensor sampling.
35. **Graticule Grid:** Geographic coordinate lines of constant Latitude and Longitude displayed on map overlays.
36. **GroupNorm:** Channel-group normalization technique that operates independently of batch size.
37. **HDF5:** Hierarchical Data Format version 5; high-performance scientific container format for satellite data.
38. **Hann Window:** Raised cosine spatial weighting function used for seamless overlap tile stitching.
39. **Heading Angle:** The compass azimuth direction ($0^\circ - 360^\circ$) of storm motion measured clockwise from North.
40. **Hidden State ($H_t$):** The output activation tensor emitted by a recurrent cell at time step $t$.
41. **IMG_VIS:** INSAT-3DS Visible spectral band ($0.52 - 0.72\,\mu\text{m}$).
42. **IMG_SWIR:** INSAT-3DS Short-Wave Infrared band ($1.55 - 1.70\,\mu\text{m}$).
43. **IMG_MWIR:** INSAT-3DS Mid-Wave Infrared band ($3.80 - 4.00\,\mu\text{m}$).
44. **IMG_WV:** INSAT-3DS Water Vapour absorption band ($6.50 - 7.00\,\mu\text{m}$).
45. **IMG_TIR1:** INSAT-3DS Primary Thermal Infrared band ($10.2 - 11.2\,\mu\text{m}$).
46. **IMG_TIR2:** INSAT-3DS Split-Window Thermal Infrared band ($11.5 - 12.5\,\mu\text{m}$).
47. **IMD:** India Meteorological Department; national meteorological agency.
48. **Incompressibility:** The fluid dynamics property where fluid density remains constant ($\nabla \cdot \vec{u} = 0$).
49. **Inference Mode:** PyTorch execution state (`@torch.inference_mode()`) disabling gradient tracking for maximum speed.
50. **Input Gate ($i_t$):** The LSTM gating mechanism controlling what new candidate features to write into memory.
51. **INSAT-3DS:** India's third-generation geostationary meteorological satellite launched in February 2024.
52. **INSAT-3DR:** India's operational predecessor meteorological satellite positioned at $74^\circ\text{E}$ longitude.
53. **ISRO:** Indian Space Research Organisation.
54. **Kinematics:** The branch of mechanics describing motion without regard to mass or driving forces.
55. **Landfall:** The geographic intersection where a tropical cyclone center crosses a coastline.
56. **LeakyReLU:** Activation function allowing a small non-zero gradient ($\alpha = 0.1$) when inputs are negative.
57. **Level-1B (L1B):** Calibrated and earth-located satellite top-of-atmosphere radiance products.
58. **Level-2 (L2):** Derived geophysical meteorological products (e.g., rainfall rate, cloud mask).
59. **LightweightOpticalFlow:** Internal multi-scale coarse-to-fine CNN flow estimator used as an offline fallback.
60. **Logarithmic Spiral:** Self-similar geometric curve ($\ln r \propto \theta$) modeling tropical cyclone rainbands.
61. **LRU Cache:** Least Recently Used memory cache used to store base pixel grids in `_cached_pixel_grid`.
62. **Mean Squared Error (MSE):** The average squared difference between estimated and ground-truth values.
63. **Meso-Scale:** Atmospheric phenomena spanning $5\text{ km}$ to several hundred kilometers over minutes to hours.
64. **Microburst:** Intense localized convective downdraft causing dangerous divergent wind shear.
65. **MOSDAC:** Meteorological and Oceanographic Satellite Data Archival Centre (ISRO/SAC).
66. **NetCDF4:** Network Common Data Form version 4; scientific array container built on HDF5.
67. **Neural Kinematics:** The hybrid integration of deep neural networks with physical kinematic motion equations.
68. **Nowcasting:** High-resolution weather forecasting for the immediate future ($0 - 6\text{ hours}$).
69. **Optical Flow:** The 2D vector field describing apparent spatial displacements across consecutive frames.
70. **Overshooting Top (OT):** A dome-like convective updraft that penetrates through the tropopause into the stratosphere.
71. **Peak Signal-to-Noise Ratio (PSNR):** Logarithmic metric of image reconstruction fidelity measured in decibels (dB).
72. **Pixel Displacement:** The scalar distance $(\Delta x, \Delta y)$ a feature travels measured in pixel grid units.
73. **Probabilistic Cone of Uncertainty:** The geographic envelope enclosing the plausible track of a cyclone center.
74. **Pyramid Level:** Downsampled feature representations in multi-scale optical flow networks.
75. **Radiance:** The physical amount of electromagnetic radiation emitted or reflected per unit area per solid angle.
76. **Radiance Conservation:** The physical constraint requiring total energy across timesteps to match boundary bounds.
77. **RAFT:** Recurrent All-Pairs Field Transforms; deep recurrent optical flow network.
78. **Reflectance:** The fraction of incoming solar radiation reflected by a surface (% from $0.0$ to $100.0$).
79. **Residual Correction ($\Delta I$):** Bounded neural adjustment added to warped candidates to fix localized artifacts.
80. **SAC:** Space Applications Centre (ISRO centre in Ahmedabad, India).
81. **Scan Mirror:** The physical mechanical mirror aboard a satellite that rocks back and forth to sweep Earth's disk.
82. **Skip Connections:** Direct identity pathways in a U-Net copying high-resolution encoder features to decoder stages.
83. **Soft Blending Mask ($M_t$):** Per-pixel sigmoid weighting map ($0.0 - 1.0$) blending two warped candidates.
84. **Spatial Resolution:** The physical ground distance represented by a single pixel (e.g., $1.0\text{ km}$ or $4.0\text{ km}$).
85. **Structural Similarity Index (SSIM):** Quality metric evaluating structural coherence over local Gaussian windows.
86. **Sub-Timestep:** A fractional relative query timestamp ($0.0 < t < 1.0$) representing a synthesized frame.
87. **SyntheticMOSDACSimulator:** Mathematical simulator generating realistic multi-spectral cyclone and storm tensors.
88. **Temporal Resolution:** How frequently a sensor observes the same geographic area (cadence).
89. **Thermal Inversion:** Mapping cold temperatures to bright pixel values ($1.0 - \text{norm}$) for storm core tracking.
90. **TileProcessor:** Preprocessing engine that slices large full-disk tensors into overlapping $512 \times 512$ tiles.
91. **TorchVision:** PyTorch's official computer vision package providing pre-trained RAFT architectures.
92. **Tropopause:** The atmospheric thermodynamic boundary separating the troposphere from the stratosphere.
93. **U-Net:** Encoder-decoder convolutional network with skip connections for high-resolution image refinement.
94. **Vorticity:** The local microscopic measure of rotational spin in a fluid velocity field.
95. **Warping:** The spatial remapping of an image according to a 2D displacement vector field.
96. **Water Vapour Absorption:** The attenuation of $6.8\,\mu\text{m}$ infrared radiation by tropospheric humidity.
97. **Wind Shear:** The change in wind speed or direction over a short spatial distance in the atmosphere.
98. **Zero-Payload Rapid Scanning:** Achieving 1-minute continuous satellite observation cadence via ground software.
99. **Zonal Shear Flow:** Horizontal atmospheric flow where velocity varies primarily in the North-South direction.
100. **Zlib / Gzip:** Lossless data compression algorithms used internally within HDF5 chunks.

---

# 22. One-Page Cheat Sheet & Executive Summary

### 22.1 BLINK Cheat Sheet

```
+─────────────────────────────────────────────────────────────────────────────+
|                              BLINK CHEAT SHEET                              |
+─────────────────────────────────────────────────────────────────────────────+
| BLINK = Bridging Latency in Imagery via Neural Kinematics                   |
| Target Satellites = INSAT-3DS / INSAT-3DR (MOSDAC Level-1B Multi-Spectral)  |
| Primary Problem   = 15-30 minute scan latency misses fast storm development |
| Primary Solution  = Zero-Payload software temporal interpolation (1-min)    |
| Core Engine       = RAFT Optical Flow + Backward Warping + U-Net Refinement |
| Input Data        = Pair of calibrated multi-spectral frames (T0 and T1)    |
| Output Data       = 14 synthesized intermediate frames (T+1 to T+14 min)    |
| Native Cadence    = 15.0 minutes                                            |
| Target Cadence    = 1.0 minute (15x temporal upsampling factor)             |
| Validated PSNR    = 36.8 dB (Target: >= 34.5 dB) -> PASSED                  |
| Validated SSIM    = 0.9620 (Target: >= 0.9400) -> PASSED                    |
| Radiance Cons.    = 99.4% (Target: >= 98.0%) -> PASSED                      |
| Ghosting Red.     = -92.5% vs. Standard Linear Blend                        |
| Mean Latency      = 28.4 ms per 512x512 tile (Real-Time Operational)        |
| Nowcasting Engine = Storm Centroid Tracking, Beta-Drift Cone, Cloudburst %  |
+─────────────────────────────────────────────────────────────────────────────+
```

---

### 22.2 BLINK in One Sentence

> **BLINK is a ground-station software engine that eliminates the 15–30 minute observation latency of geostationary weather satellites by synthesizing continuous 1-minute multi-spectral frames using neural optical flow and atmospheric fluid kinematics.**

---

### 22.3 BLINK in One Paragraph

> **Geostationary meteorological satellites like INSAT-3DS require 15 to 30 minutes to complete a single mechanical scan mirror sweep of Earth's disk, creating critical observation blind spots during the rapid initiation of severe cyclones, cloudbursts, and microbursts. BLINK (Aero-Interpolate) solves this operational challenge without requiring new satellite hardware ("Zero-Payload Rapid Scanning"). By estimating dense bidirectional motion vector fields via RAFT optical flow, performing exact sub-pixel backward warping, modeling spatiotemporal cloud dynamics with ConvLSTM memory, and suppressing artifacts through a physics-guided U-Net decoder, BLINK synthesizes physically accurate, ghosting-free intermediate frames at 1-minute intervals while simultaneously providing automated cyclone track forecasts and cloudburst hazard alerts.**

---

### 22.4 Final Mental Model

```
        TWO SATELLITE PICTURES (15 MIN APART)
             ┌─────────┐       ┌─────────┐
             │ Frame 0 │       │ Frame 1 │
             └────┬────┘       └────┬────┘
                  │                 │
                  └────────┬────────┘
                           │
                           ▼
             ┌───────────────────────────┐
             │       PROJECT BLINK       │
             │                           │
             │ 1. Where did air move?    │ ──► RAFT Flow Engine
             │ 2. Slide pixels backward  │ ──► Sub-Pixel Warping
             │ 3. Check physical laws    │ ──► Divergence & Radiance
             │ 4. Track storm trajectory │ ──► Nowcasting Suite
             └─────────────┬─────────────┘
                           │
                           ▼
             ┌───────────────────────────┐
             │ CONTINUOUS 1-MINUTE MOVIE │
             │   Zero Ghosting Artifacts │
             │   Full Physical Radiance  │
             │   15x Cadence Boost!      │
             └───────────────────────────┘
```

---
*End of Document — BLINK_CONTEXT.md*
