# Project BLINK: Scientific Validation & Methodological Grounding

**Document Classification:** Internal Technical & Scientific Specification  
**Version:** 2.0.0 (Scientific Audit Release)  
**Authors:** Project BLINK Research Engineering & Meteorological Data Science Team  
**Review Status:** Operational Standards Aligned (IMD / WMO / SAC-ISRO)

---

## 1. Scientific Scope of BLINK

Project BLINK is an atmospheric motion and radiance synthesis framework designed for geostationary meteorological satellite observations over the Indian subcontinent and surrounding oceanic basins. The system addresses the fundamental spatiotemporal observation gap between standard satellite scan cadences (e.g., 15-minute or 30-minute full-disk / sector scans from INSAT-3DS) and the microphysical timescales of rapidly evolving convective hazards (e.g., localized convective initiation occurring within 3 to 10 minutes).

The primary functional objective is twofold:
1. **High-Cadence Temporal Interpolation (Mode A):** Given two consecutive geostationary observations separated by $\Delta t$ (e.g., 15 or 30 minutes), reconstruct intermediate atmospheric state representations at 1-minute to 3-minute effective cadences by estimating non-linear, compressible atmospheric fluid flow fields.
2. **Convective Nowcasting & Hazard Indication (Mode B):** Compute diagnostic kinematic and thermodynamic tendency indicators (such as cloud-top cooling rates, cold-cloud anvil extent, and cyclone circulation metrics) to provide objective situational awareness to operational meteorologists.

---

## 2. Interpolation Definition (Mode A: Temporal Reconstruction)

### Definition & Mathematical Formulation
Temporal interpolation is a **bidirectional boundary-value problem**. The model is provided with two verified, temporally adjacent observations:
$$I_0 = I(t_0) \quad \text{and} \quad I_1 = I(t_1)$$
where $t_1 - t_0 = \Delta t$ (typically 15 or 30 minutes).

For any normalized intermediate query timestep $\tau \in (0, 1)$, where $t_\tau = t_0 + \tau(t_1 - t_0)$, the system computes the intermediate radiance or brightness temperature field $\hat{I}(t_\tau)$ using bidirectional optical flow vector fields:
$$\vec{u}_{0 \to 1} = (u, v) \quad \text{and} \quad \vec{u}_{1 \to 0} = (-u, -v)$$
with backward and forward splatting/warping:
$$\hat{I}(t_\tau) = \mathcal{W}_{\text{bidirectional}}(I_0, I_1, \vec{u}_{0 \to 1}, \tau)$$

### Critical Operational Boundary
Because Mode A utilizes *both past ($t_0$) and future ($t_1$) boundary conditions*, intermediate synthesized frames $\hat{I}(t_\tau)$ are **retrospective re-analyses / reconstructions**, **not future forecasts**.
- **Valid Evaluation Protocol:** Mode A can be quantitatively evaluated **only** when an independent, true intermediate observation $I_{\text{obs}}(t_\tau)$ was recorded by the sensor but **withheld** from the model at inference time.
- **Forbidden Practice:** Evaluating $\hat{I}(t_\tau)$ against linear combinations of the boundary inputs (e.g., $(1-\tau)I_0 + \tau I_1$) is strictly circular and invalid.

---

## 3. Forecast / Nowcast Definition (Mode B: Kinematic Extrapolation)

### Definition & Mathematical Formulation
Kinematic forecasting is an **initial-value extrapolation problem**. The model is provided strictly with historical observations:
$$\{I(t_{-k}), \dots, I(t_{-1}), I(t_0)\}$$
and predicts future atmospheric fields at lead times $\Delta t > 0$:
$$\hat{I}(t_0 + \Delta t) = \mathcal{F}_{\text{extrapolate}}(I(t_0), \vec{u}_{\text{historical}}, \Delta t)$$

In BLINK, nowcasting also includes kinematic cyclone track forecasting ($\text{Track}(t + \Delta t)$ for $\Delta t \in [3\text{h}, 48\text{h}]$) using atmospheric steering flow vectors corrected by Coriolis beta-drift recurvature.

### Critical Operational Boundary
At inference time, future states are completely unknown to the system.
- **Valid Evaluation Protocol:** Nowcasts must be evaluated strictly against future observations that materialize after the forecast initialization timestamp:
  $$t_{\text{target}} > t_{\text{initialization}}$$
- Mode A interpolation metrics must **never** be mixed or reported interchangeably with Mode B forecast metrics.

---

## 4. INSAT-3DS Data Used

BLINK operates directly on Level-1B (L1B) Standard Products from the INSAT-3DS Imager payload, archived and distributed by the Meteorological and Oceanographic Satellite Data Archival Centre (MOSDAC, SAC/ISRO):
- **Product Name Convention:** `3SIMG_DDMMMYYYY_HHMM_L1B_STD_V01R00.h5`
- **File Format:** Hierarchical Data Format 5 (HDF5).
- **Structure:**
  - `/IMG_VIS`: Visible reflectance channel raw digital counts.
  - `/IMG_SWIR`: Short-Wave Infrared counts.
  - `/IMG_MIR`: Mid-Wave Infrared counts.
  - `/IMG_TIR1`: Thermal Infrared-1 counts.
  - `/IMG_TIR2`: Thermal Infrared-2 counts.
  - `/IMG_WV`: Water Vapor counts.
  - `/Latitude` & `/Longitude`: Geodetic coordinate grids (4 km nadir for IR, 1 km for VIS).
  - `/Calibration`: Radiometric lookup tables and Planck inversion coefficients.

---

## 5. Calibration Assumptions & Radiometric Conversion

Raw 10-bit digital counts ($DN$) recorded by the INSAT-3DS Imager detectors are converted to physical quantities through the official SAC/ISRO calibration tables:

### Thermal Infrared & Water Vapor Channels (TIR-1, TIR-2, MIR, WV)
1. Digital counts are mapped to spectral radiance $L_\lambda$ ($\text{mW} \cdot \text{m}^{-2} \cdot \text{sr}^{-1} \cdot (\text{cm}^{-1})^{-1}$) via on-board blackbody calibration lookup tables:
   $$L_\lambda = \text{LUT}_{\text{calib}}[DN]$$
2. Radiance is inverted to effective Brightness Temperature ($T_{\text{B}}$ in Kelvin) via the Planck function adapted with sensor-specific central wavenumber $\nu_c$ ($\text{cm}^{-1}$) and temperature calibration coefficients ($A, B$):
   $$T^* = \frac{c_2 \cdot \nu_c}{\ln\left(1 + \frac{c_1 \cdot \nu_c^3}{L_\lambda}\right)}$$
   $$T_{\text{B}} = \frac{T^* - A}{B}$$
   where:
   - $c_1 = 1.191042 \times 10^{-5} \text{ mW} \cdot \text{m}^{-2} \cdot \text{sr}^{-1} \cdot \text{cm}^4$
   - $c_2 = 1.4387752 \text{ cm} \cdot \text{K}$

### Normalization Scale in BLINK
For internal neural and optical-flow processing, thermal channels are linearly normalized:
$$I_{\text{norm}} = \frac{298.0 - T_{\text{B}}}{105.0} \quad \iff \quad T_{\text{B}} = 298.0 - 105.0 \cdot I_{\text{norm}}$$
- $I_{\text{norm}} = 1.0 \implies T_{\text{B}} = 193.0\text{ K}$ (Extremely cold convective cloud top)
- $I_{\text{norm}} = 0.0 \implies T_{\text{B}} = 298.0\text{ K}$ (Warm surface/ocean)

---

## 6. Channel & Resolution Specifications

| Channel Name | Wavelength ($\mu\text{m}$) | Spatial Resolution (Nadir) | Primary Meteorological Target | Physical Unit |
| :--- | :--- | :--- | :--- | :--- |
| **VIS** | 0.55 – 0.75 | 1.0 km | Cloud albedo, low-level cumulus, daytime texture | Bidirectional Reflectance Factor [0, 1] |
| **SWIR** | 1.55 – 1.70 | 1.0 km | Cloud phase discrimination (ice vs water), fog, snow | Reflectance / Radiance |
| **MIR** | 3.80 – 4.00 | 4.0 km | Hot spot detection, nocturnal low clouds, fog | Brightness Temperature (K) |
| **TIR-1** | 10.3 – 11.3 | 4.0 km | Atmospheric window, cloud-top $T_{\text{B}}$, SST | Brightness Temperature (K) |
| **TIR-2** | 11.5 – 12.5 | 4.0 km | Split-window moisture attenuation, dust, cirrus | Brightness Temperature (K) |
| **WV** | 6.50 – 7.10 | 8.0 km | Mid-to-upper tropospheric water vapor, jet streams | Brightness Temperature (K) |

> [!IMPORTANT]
> VIS has a 1 km grid, while TIR-1 has a 4 km grid and WV has an 8 km grid. In the BLINK ingestion pipeline (`mosdac_parser.py`), VIS is downsampled or TIR is aligned using coordinate-aware bilinear regridding. Channels must not be assumed to have identical native spatial resolutions without explicit geometric resampling.

---

## 7. NETRA Scientific Basis & Taxonomy

NETRA (Nowcasting of Extreme Events using Tracking of Convective Clouds, Shukla et al., SAC/ISRO 2017) was conceptualized as a satellite-based convective storm tracking technique for the Indian region.

### Underlying Physical Mechanism
Deep moist convection exhibits distinct thermodynamic signatures in geostationary thermal infrared imagery:
1. **Vertical Ascent:** Strong convective updrafts transport cloud tops rapidly through the troposphere.
2. **Adiabatic Expansion & Cloud-Top Cooling:** As the cloud top rises, its radiative temperature drops rapidly, creating an intense negative local temperature tendency:
   $$\frac{\partial T_{\text{B}}}{\partial t} < 0$$
3. **Cold Anvil Spreading:** Reaching the neutral buoyancy level (anvil formation), an extensive cirrus shield develops with $T_{\text{B}} \le 220\text{ K}$.

### Taxonomy Classification: Heuristic Indicator
Because BLINK currently computes NETRA indicators directly from satellite $T_{\text{B}}$ without a machine-learning model trained on event-labeled rainfall observations, NETRA outputs are classified strictly as **Heuristic Indicators**:
- **Term Used:** `Convective Activity Index (CAI)` (0 to 100).
- **Categorical Levels:** `NOMINAL`, `MODERATE`, `ELEVATED`, `SEVERE_CONVECTIVE_UPDRAFT`.
- **Disclaimer:** *Heuristic diagnostic indicator. Not a calibrated event-level probability of surface precipitation.*

---

## 8. Cloudburst Target Definition: Physical Disconnect

### IMD Operational Definition of Cloudburst
Under official criteria established by the India Meteorological Department (IMD):
$$\text{Cloudburst} \iff \text{Rainfall rate} \ge 100\text{ mm/hr} \text{ over a localized area of } 20\text{--}30\text{ km}^2$$

### The Physical Reality: Cold Cloud Top $\ne$ Cloudburst
Satellite thermal infrared radiometers measure the **radiating temperature of the cloud top at the top of the atmosphere (TOA)**. They do **not** measure:
- Rainfall rate reaching the surface.
- Raindrop size distribution (DSD).
- Sub-cloud evaporation (virga).
- Orographic coalescence efficiency in mountain valleys.

Many mesoscale convective systems (MCS) exhibit extremely cold cloud tops ($T_{\text{B}} < 200\text{ K}$) and rapid cooling rates over broad areas without producing a localized 100 mm/hr cloudburst at the surface. Conversely, orographic cloudbursts in the Western Ghats and Himalayas can occur under warm-rain processes with modest cloud tops ($T_{\text{B}} \approx 235\text{ K}$).

> [!CAUTION]
> Equating a satellite cooling rate or cold cloud top to a "Cloudburst Probability" without ground-truth radar or rain-gauge validation is scientifically indefensible. The BLINK system **prohibits** claiming a "Cloudburst Risk %" until a verified event dataset is incorporated.

---

## 9. Overshooting-Top (OT) Methodology & Scientific Limits

### Authoritative Literature
- **Bedka et al. (2010)**, *Objective Satellite-Based Detection of Overshooting Tops*, Journal of Applied Meteorology and Climatology.
- **Proud (2015)**, *Analysis of Overshooting Top Detections by MSG SEVIRI*, Geophysical Research Letters.

### Criteria for Real Operational OT Detection
1. **IR Cold Spot:** The candidate pixel must have an IR brightness temperature colder than the surrounding anvil:
   $$T_{\text{anvil}} - T_{\text{candidate}} \ge 6.5\text{ K}$$
2. **NWP Tropopause Penetration:** The candidate must be colder than the local tropopause temperature ($T_{\text{candidate}} \le T_{\text{trop}}$) derived from numerical weather prediction (NWP) analysis or atmospheric soundings.
3. **Water Vapor Resonance:** In overshooting tops, tropospheric water vapor is injected into the lower stratosphere, causing the $6.7\,\mu\text{m}$ WV channel brightness temperature to equal or exceed the $10.8\,\mu\text{m}$ window temperature ($T_{\text{WV}} - T_{\text{TIR}} \ge 0\text{ K}$).
4. **Spatial Compactness:** The feature must be a distinct spatial core with diameter $\le 15\text{ km}$.

### BLINK's Current Implementation & Labeling
BLINK does not currently ingest real-time GFS/ECMWF NWP tropopause sounding fields. Therefore:
- The detector uses a 2D Gaussian spatial filter ($\sigma = 3.0$) to estimate the local background anvil temperature $\bar{T}_{\text{anvil}}$ and searches for relative minima:
  $$T_{\text{B}} \le \bar{T}_{\text{anvil}} - 4.5\text{ K} \quad \text{and} \quad T_{\text{B}} < 210\text{ K}$$
- **Mandatory Scientific Labeling:** This is designated in the UI and API as an **`Experimental OT Proxy / Candidate Updraft Core`**. The system does not claim operational certainty.

---

## 10. Cloud-Top Brightness-Temperature Tendency (Cooling Rate)

### Formulation
Cloud-top cooling is computed as the Lagrangian or Eulerian difference between two verified observation frames separated by known scan delta $\Delta t$:
$$\text{Tendency} = \left(\frac{\Delta T_{\text{B}}}{\Delta t}\right) = \frac{T_{\text{B}}(t_1, x, y) - T_{\text{B}}(t_0, x, y)}{t_1 - t_0}$$

### Rigorous Specification
- **Variable Evaluated:** Top-of-atmosphere Equivalent Blackbody Brightness Temperature ($T_{\text{B}}$) in the Thermal Infrared-1 (TIR-1, $10.8\,\mu\text{m}$) channel.
- **Units:** Explicitly reported in **$\text{K / 15-min}$** or **$\text{K / 30-min}$** matching sensor scan cadence.
- **Sign Convention:** Negative values denote cooling (upward convective ascent); positive values denote warming (cloud dissipation or subsidence).
- **Physical Significance:**
  - $\Delta T_{\text{B}}/\Delta t \in [-4\text{ K}, 0\text{ K}]$ per 15-min: Nominal / weak convection.
  - $\Delta T_{\text{B}}/\Delta t \in [-10\text{ K}, -4\text{ K}]$ per 15-min: Developing convective cell.
  - $\Delta T_{\text{B}}/\Delta t \le -15\text{ K}$ per 15-min: Vigorous, explosive deep convective updraft (Roberts & Rutledge, 2003).

---

## 11. Peak Signal-to-Noise Ratio (PSNR)

### Formulation
For an independent, held-out ground truth frame $I_{\text{gt}}$ and a model prediction $\hat{I}$:
$$\text{MSE} = \frac{1}{|\Omega|} \sum_{(x, y) \in \Omega} (\hat{I}(x, y) - I_{\text{gt}}(x, y))^2$$
$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right) \quad (\text{dB})$$

### Operational Constraints
1. **Explicit Dynamic Range:**
   - In normalized domain: $\text{MAX}_I = 1.0$.
   - In physical temperature domain: $\text{MAX}_I = 140.0\text{ K}$ (spanned range $[180\text{ K}, 320\text{ K}]$).
   - Normalized and physical PSNR values have different numerical scales and must **never** be mixed.
2. **Valid Mask $\Omega$:** Masked to valid sensor pixels, excluding space-look background pixels and missing-line fill values.
3. **Ground Truth Rule:** If no independent ground truth frame exists for the evaluation timestep, $\text{PSNR} = \text{None}$ (displayed as `N/A`).

---

## 12. Structural Similarity Index (SSIM)

### Formulation
Computed over local spatial windows $w$ using Gaussian weighting ($\sigma = 1.5$, window size 11):
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
with stability constants:
$$C_1 = (K_1 \cdot \text{MAX}_I)^2, \quad C_2 = (K_2 \cdot \text{MAX}_I)^2, \quad K_1 = 0.01, \quad K_2 = 0.03$$

### Operational Constraints
- Same ground truth rules apply: requires an independent held-out observation. If absent, $\text{SSIM} = \text{None}$ (`N/A`).

---

## 13. Root Mean Square Error (RMSE)

### Physical Domain RMSE
For thermal infrared channels, RMSE must be computed in physical units of **Kelvin (K)**:
$$\text{RMSE}_{\text{BT}} = \sqrt{\frac{1}{|\Omega|} \sum_{(x, y) \in \Omega} (\hat{T}_{\text{B}}(x, y) - T_{\text{B, gt}}(x, y))^2} \quad [\text{K}]$$

### Normalized RMSE (NRMSE)
If a percentage is reported, it must explicitly specify its normalizing denominator:
$$\text{NRMSE}_{\text{range}} = \frac{\text{RMSE}_{\text{BT}}}{T_{\max} - T_{\min}} \times 100\% \quad \text{or} \quad \text{NRMSE}_{\text{mean}} = \frac{\text{RMSE}_{\text{BT}}}{\bar{T}_{\text{B, gt}}} \times 100\%$$
- Under no circumstances will endpoint differences $\sqrt{\text{mean}((T_0 - T_1)^2)} \times 100$ be called RMSE.

---

## 14. Temperature Consistency & Internal Diagnostics

When no independent temporal ground truth frame exists, the system reports **Internal Physical Diagnostics** rather than fake validation scores:

1. **Radiance Flux Conservation (%):**
   Evaluates whether the total integrated radiative flux of the intermediate frame complies with conservative linear advection bounds:
   $$\text{Mass}_{\text{exp}} = (1 - \tau) \sum I_0 + \tau \sum I_1$$
   $$\text{Conservation \%} = \max\left(0, 100 \times \left(1 - \frac{|\sum \hat{I} - \text{Mass}_{\text{exp}}|}{\text{Mass}_{\text{exp}}}\right)\right)$$
2. **Physical BT Range Compliance (%):**
   Percentage of synthesized pixels falling within physically viable atmospheric limits:
   $$\text{Compliance \%} = \frac{1}{N} \sum_{(x, y)} \mathbb{I}[180.0\text{ K} \le \hat{T}_{\text{B}}(x, y) \le 330.0\text{ K}] \times 100\%$$
3. **Mean BT Drift (K):**
   The mean temperature offset of the synthesized frame relative to the expected interpolated mean:
   $$\text{Drift} = \text{Mean}(\hat{T}_{\text{B}}) - \left((1 - \tau)\text{Mean}(T_0) + \tau \text{Mean}(T_1)\right) \quad [\text{K}]$$

---

## 15. Benchmark Baselines

No validation metric may ever be reported in isolation. Project BLINK benchmarks against two fundamental reference baselines:
1. **Persistence Baseline:**
   Assumes zero temporal movement or evolution:
   $$\hat{I}_{\text{persistence}}(t_\tau) = I_0$$
2. **Linear Interpolation Baseline:**
   Direct pixel-wise linear alpha blend:
   $$\hat{I}_{\text{linear}}(t_\tau) = (1 - \tau) I_0 + \tau I_1$$
3. **Verification Question:**
   *Does BLINK outperform linear blending and persistence on independent held-out observations?*

---

## 16. Dataset Splits & Leakage Prevention

To prevent data leakage:
- **Temporal Separation:** Adjacent satellite frames (within 6 hours) must never be split across training and test sets.
- **Event-Level Splitting:** Storm events (e.g., Cyclone Dana 2024, Cyclone Biparjoy 2023, Uttarakhand heavy rain episodes) are quarantined into entire independent test sets.
- **Geographic Partitioning:** Specific regional sub-domains (e.g., Himalayan Orographic Belt vs Bay of Bengal Maritime) are tracked separately.

---

## 17. Ground-Truth Sources for Operational Nowcasting

To achieve genuine event-level verification, future expansions of Project BLINK will ingest:
1. **IMD Doppler Weather Radar (DWR):** Radar reflectivity (MAXZ $\ge 45\text{ dBZ}$) and Surface Precipitation Intensity (SPI mm/hr).
2. **IMD Automatic Weather Stations (AWS):** 15-minute tipping bucket rain gauge records.
3. **GPM / IMERG:** Integrated Multi-satellitE Retrievals for GPM (0.1° half-hourly calibrated precipitation).

---

## 18. Current System Limitations

1. **Absence of Real-Time NWP Tropopause Soundings:** Prevents definitive confirmation of overshooting tops; candidates remain experimental proxies.
2. **Top-of-Atmosphere Radiative Constraint:** Infrared observations cannot penetrate thick cloud decks to observe low-level precipitation beneath anvils.
3. **Spatial Resolution (4 km):** Sub-kilometer convective feeder cells are unresolved in TIR-1.
4. **Single-Interval Ground Truth Absence:** In normal 2-frame operation, no intermediate observation exists; validation metrics must report `N/A`.

---

## 19. Claims Project BLINK is NOT Allowed to Make

- **FORBIDDEN:** *"BLINK detects cloudbursts with 96% accuracy."* (No event-level ground truth exists).
- **FORBIDDEN:** *"PSNR is 36.42 dB"* (when computed by comparing prediction against a linear blend).
- **FORBIDDEN:** *"RMSE is 3.21%"* (when computed by comparing $T_0$ to $T_1$).
- **FORBIDDEN:** *"Overshooting tops detected with operational certainty."* (No NWP tropopause data ingested).
- **FORBIDDEN:** Showing high decimal scores (e.g., `0.9955`) under the ambiguous label "Consistency".

---

## 20. Permissible Scientific Claims

- **PERMISSIBLE:** *"BLINK reconstructs physically smooth, divergence-regularized intermediate radiance fields from 15-minute INSAT-3DS observations."*
- **PERMISSIBLE:** *"The Convective Activity Index highlights regions of rapid cloud-top brightness-temperature cooling ($\Delta T_{\text{B}}/\Delta t \le -12\text{ K / 15-min}$) and intense anvil development."*
- **PERMISSIBLE:** *"When evaluated on held-out INSAT-3DS observations, BLINK achieves a PSNR gain of $+X\text{ dB}$ over linear frame interpolation."*
- **PERMISSIBLE:** *"Validation metrics are reported as N/A when independent observations are unavailable."*
