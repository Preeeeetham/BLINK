# Operational Methodology: Tropical Cyclone Landfall Prediction via INSAT-3D/3DR/3DS Geostationary Satellite Data

**Project BLINK Technical Research Report**  
**Author**: Meteorological Data Science & Deep Learning Research Group  
**Target Domain**: North Indian Ocean (Bay of Bengal & Arabian Sea)  
**Standard Authorities**: India Meteorological Department (IMD), ISRO Space Applications Centre (SAC), WMO/ESCAP Panel on Tropical Cyclones  

---

## Executive Summary

Tropical Cyclones (TCs) over the North Indian Ocean (NIO)—encompassing the Bay of Bengal and the Arabian Sea—pose catastrophic risks to densely populated coastal corridors in India, Bangladesh, and Myanmar. The **India Meteorological Department (IMD)**, designated by the World Meteorological Organization (WMO) as the Regional Specialized Meteorological Centre (RSMC) New Delhi, operationalizes a standardized satellite-driven forecasting pipeline to predict cyclone genesis, central location, intensity, track trajectory, and **coastal landfall point and time (ETA)**.

Central to this operational capability is the constellation of Indian geostationary Earth observation satellites: **INSAT-3D**, **INSAT-3DR**, and **INSAT-3DS** (stationed at $74^\circ\text{E}$ and $82^\circ\text{E}$). This document details the exact physics, sensor channels, mathematical formulations, and operational standard operating procedures (SOP) used to derive cyclone landfall forecasts from INSAT multi-spectral radiance data.

---

## 1. INSAT-3D / 3DR / 3DS Multi-Spectral Imager Payloads

The INSAT-3D-class Imager provides optical and thermal infrared observations across 6 discrete spectral bands:

| Channel Identifier | Central Wavelength ($\lambda$) | Spatial Resolution at Nadir | Primary Meteorological Function in Cyclone Monitoring |
| :--- | :--- | :--- | :--- |
| **Visible (VIS)** | $0.55 - 0.75\,\mu\text{m}$ ($0.65\,\mu\text{m}$) | $1.0\,\text{km} \times 1.0\,\text{km}$ | Daytime eye structure, low-level cumulus cloud streaks, convective feeder band texture. |
| **Short-Wave IR (SWIR)** | $1.55 - 1.70\,\mu\text{m}$ ($1.62\,\mu\text{m}$) | $1.0\,\text{km} \times 1.0\,\text{km}$ | Cloud-phase discrimination (ice vs. supercooled liquid water), snow/cloud differentiation. |
| **Mid-Wave IR (MIR)** | $3.80 - 4.00\,\mu\text{m}$ ($3.90\,\mu\text{m}$) | $4.0\,\text{km} \times 4.0\,\text{km}$ | Nocturnal low-level circulation detection, sea surface temperature (SST) boundary contrast. |
| **Water Vapour (WV)** | $6.50 - 7.10\,\mu\text{m}$ ($6.80\,\mu\text{m}$) | $8.0\,\text{km} \times 8.0\,\text{km}$ | Upper-to-mid tropospheric moisture flux, environmental dry-air intrusion, upper-level divergence. |
| **Thermal IR-1 (TIR-1)** | $10.3 - 11.3\,\mu\text{m}$ ($10.8\,\mu\text{m}$) | $4.0\,\text{km} \times 4.0\,\text{km}$ | **Primary Dvorak / ADT channel**: Cloud-top brightness temperature ($T_B$), eye-wall thermal gradient. |
| **Thermal IR-2 (TIR-2)** | $11.5 - 12.5\,\mu\text{m}$ ($12.0\,\mu\text{m}$) | $4.0\,\text{km} \times 4.0\,\text{km}$ | Split-window low-level moisture estimation, atmospheric path attenuation correction. |

### Operational Scan Cadence & Rapid Scan Mode (RSM)
- **Standard Mode**: Full-disk imagery covering the entire Indian Ocean basin ($45^\circ\text{S} - 45^\circ\text{N}, 30^\circ\text{E} - 120^\circ\text{E}$) is collected every **30 minutes** (e.g. 00:00, 00:30 UTC).
- **Staggered Tandem Mode**: Combining INSAT-3D and INSAT-3DR/3DS provides interlaced observations every **15 minutes**.
- **Rapid Scan Mode (RSM)**: When a cyclonic disturbance intensifies into a Deep Depression ($V_{\max} \ge 28\,\text{knots}$) or approaches within $500\,\text{km}$ of the Indian coastline, ISRO/IMD activates the **Rapid Scan Mode** on INSAT-3DR or INSAT-3DS:
  - Sector: $10^\circ \times 10^\circ$ or $15^\circ \times 15^\circ$ flexible regional window centered over the cyclone.
  - Scan Interval: **4.7 minutes** (~12.8 scans per hour).
  - Purpose: Captures high-frequency eye-wall replacement cycles (ERC), center displacement jitter, and rapid intensification bursts.

---

## 2. Vortex Center Identification & Fixing Algorithms

Accurate landfall prediction is conditioned on establishing the initial storm position $(x_0, y_0, t_0)$ with minimum geodetic error.

```
       INSAT TIR-1 (10.8 µm) & VIS (0.65 µm) Radiance
                             │
                             ▼
                    Eye Pattern Present?
                     /                \
                   YES                 NO
                   /                    \
                  ▼                      ▼
      Min BT Gradient Center       Logarithmic Spiral Fitting
     ∇T_B = 0, T_eye - T_wall       r(θ) = r_0 · exp(b·θ)
                  \                      /
                   ▼                    ▼
                Centroid Consensus Cross-Check
                             │
                             ▼
                 Fixed Center (Lat_0, Lon_0)
```

### 2.1 Eye-Pattern Center Fixing (Mature Systems: CI $\ge$ 4.0)
For mature cyclones exhibiting a clear eye (e.g., Very Severe Cyclonic Storms and above):
1. **Brightness Temperature Minimum Gradient**: The eye is characterized by a warm local maximum ($T_{\text{eye}} \sim 240 - 275\,\text{K}$) surrounded by a ring of extremely cold convective cloud tops ($T_{\text{wall}} \sim 190 - 215\,\text{K}$).
2. The center is fixed at the centroid of the warmest pixels inside the closed eye isotherm contour:
   $$\vec{x}_{\text{center}} = \frac{\sum_{i \in \text{Eye}} T_{B}(i) \cdot \vec{x}_i}{\sum_{i \in \text{Eye}} T_{B}(i)}$$
3. Verification: The surrounding eye-wall temperature gradient must satisfy $|\nabla T_B| \ge 2.5\,\text{K/km}$.

### 2.2 Curved Band & Central Dense Overcast (CDO) Center Fixing (Early/Weak Systems)
For systems lacking an eye (Depression, Deep Depression, Cyclonic Storm):
1. **Logarithmic Spiral Overlay**: The spiral convective rainbands follow a 10° or 15° logarithmic spiral formulation in polar coordinates:
   $$r(\theta) = r_0 \, e^{b\theta} \quad \text{where } b = \tan\alpha$$
   where $\alpha \approx 10^\circ - 15^\circ$ is the crossing angle of cyclonic inflow.
2. Forecasters and automated algorithms fit the logarithmic spiral to the outer curvature of the $T_B < 235\,\text{K}$ cloud shield in the TIR-1 channel to pinpoint the low-level circulation center (LLCC).
3. **Kinematic Optical Flow Centroid**: In Project BLINK, the optical flow field derived by RAFT computes the rotational center as the zero-crossing singularity of the vector field:
   $$\vec{u}(\vec{x}_{\text{center}}) \approx 0, \quad \nabla \times \vec{u} = \zeta_{\max} > 0$$

---

## 3. Objective Intensity Estimation via Digital Dvorak & ADT

The intensity of the cyclone governs its physical radius of maximum winds, storm surge potential, and resistance to environmental vertical wind shear.

### 3.1 Advanced Dvorak Technique (ADT)
The objective ADT evaluates:
1. **Eye Temperature ($T_{\text{eye}}$)**: Warmest pixel within the eye radius ($r \le 50\,\text{km}$).
2. **Surrounding Ring Temperature ($T_{\text{ring}}$)**: Coldest uniform ring surrounding the eye at radius $r_{\text{ring}}$.
3. **Thermal Contrast**:
   $$\Delta T = T_{\text{eye}} - T_{\text{ring}}$$
4. **Current Intensity (CI) Number Calculation**:
   $$\text{CI} = f(T_{\text{ring}}) + g(\Delta T) - \text{Latitudinal Correction}$$

### 3.2 Wind-Pressure Formulation (North Indian Ocean Basin)
IMD utilizes the empirical relationship formulated by **Courtney & Knaff (2009)** and calibrated for the North Indian Ocean:
$$V_{\max} (\text{knots}) = 2.3 \times (1010 - P_c)^{0.76}$$
Inverting to solve for the central surface atmospheric pressure $P_c$:
$$P_c (\text{hPa}) = 1010.0 - \left(\frac{V_{\max}}{2.3}\right)^{1.3158}$$

### 3.3 IMD Cyclone Classification Matrix

| Category | Abbreviation | 3-min Sustained Wind ($V_{\max}$) | Central Pressure Deficit ($\Delta P$) | Typical Landfall Damage Profile |
| :--- | :--- | :--- | :--- | :--- |
| **Depression** | D | $17 - 27\,\text{kt}$ ($31 - 49\,\text{km/h}$) | $1.5 - 3.0\,\text{hPa}$ | Minor coastal sea disturbance. |
| **Deep Depression** | DD | $28 - 33\,\text{kt}$ ($50 - 61\,\text{km/h}$) | $3.0 - 4.5\,\text{hPa}$ | Heavy rainfall, squally winds. |
| **Cyclonic Storm** | CS | $34 - 47\,\text{kt}$ ($62 - 88\,\text{km/h}$) | $4.5 - 8.5\,\text{hPa}$ | Uprooting of trees, thatched roof damage. |
| **Severe Cyclonic Storm** | SCS | $48 - 63\,\text{kt}$ ($89 - 117\,\text{km/h}$) | $8.5 - 15.0\,\text{hPa}$ | Disruption of communication lines, major storm surge. |
| **Very Severe Cyclonic Storm** | VSCS | $64 - 89\,\text{kt}$ ($118 - 165\,\text{km/h}$) | $15.0 - 28.0\,\text{hPa}$ | Extensive structural damage, $2 - 4\,\text{m}$ surge. |
| **Extremely Severe Cyclonic Storm** | ESCS | $90 - 119\,\text{kt}$ ($166 - 221\,\text{km/h}$) | $28.0 - 55.0\,\text{hPa}$ | Catastrophic structural destruction, $4 - 6\,\text{m}$ surge. |
| **Super Cyclonic Storm** | SuCS | $\ge 120\,\text{kt}$ ($\ge 222\,\text{km/h}$) | $> 55.0\,\text{hPa}$ | Complete devastation of coastal infrastructure, $> 6\,\text{m}$ surge. |

---

## 4. Steering Flow & Atmospheric Motion Vectors (AMVs)

A tropical cyclone is advected primarily by the large-scale environmental winds in which the vortex is embedded. This is known as the **Steering Current**.

### 4.1 INSAT Atmospheric Motion Vectors (AMVs)
ISRO MOSDAC derives AMVs by tracking small cloud tracers and water vapor moisture gradients across successive 15-min or 30-min scans in three channels:
1. **Visible AMV (Low Level)**: Derived at $850\,\text{hPa}$ ($1.5\,\text{km}$) from low-level cumulus cloud tracers. Captures environmental boundary layer inflow.
2. **Thermal IR AMV (Mid-to-Upper Level)**: Derived at $700 - 400\,\text{hPa}$ ($3 - 7\,\text{km}$) from stratiform cloud elements.
3. **Water Vapor AMV (Clear-Air Upper Level)**: Derived at $300 - 150\,\text{hPa}$ ($9 - 14\,\text{km}$) by cross-correlating upper-tropospheric water vapor features in the $6.8\,\mu\text{m}$ channel. Captures anticyclonic outflow and jet streams.

### 4.2 Deep-Layer Mean (DLM) Steering Flow Formulation
The cyclone translation vector $\vec{V}_{\text{steering}}$ is calculated by integrating the environmental wind vectors $\vec{V}_{\text{env}}(p)$ mass-weighted across pressure levels between $850\,\text{hPa}$ and $200\,\text{hPa}$:
$$\vec{V}_{\text{steering}} = \frac{1}{\Delta p} \int_{200\,\text{hPa}}^{850\,\text{hPa}} \vec{V}_{\text{env}}(p) \, dp = \frac{\sum_{k} w_k \cdot \vec{V}_k}{\sum_{k} w_k}$$
Typical operational vertical pressure weights:
- $w_{850} = 0.25$
- $w_{700} = 0.25$
- $w_{500} = 0.30$
- $w_{300} = 0.15$
- $w_{200} = 0.05$

---

## 5. Beta-Drift & Coriolis Recurvature Dynamics

A cyclone does not move strictly along the environmental steering current. Because the Earth is rotating, the Coriolis parameter $f$ varies with latitude:
$$f = 2\Omega \sin\phi \implies \beta = \frac{df}{dy} = \frac{2\Omega \cos\phi}{R_{\text{Earth}}}$$

### 5.1 Asymmetric Beta Gyres
The cyclonic vortex transports high-vorticity polar air southward to the west of the center and low-vorticity tropical air northward to the east of the center. This differential advection of planetary vorticity induces two secondary circulation gyres:
- A **cyclonic gyre** southwest of the center.
- An **anticyclonic gyre** northeast of the center.

### 5.2 Beta-Drift Vector
The secondary flow between these beta gyres exerts a net ventilation across the cyclone core, inducing an autonomous translation velocity component:
$$\vec{V}_{\beta} \approx 2.0 - 3.5\,\text{m/s} \quad (\sim 7 - 13\,\text{km/h}) \text{ oriented towards the North-West } (315^\circ)$$
Thus, the total instantaneous cyclone translation vector is:
$$\vec{V}_{\text{track}} = \vec{V}_{\text{steering}} + \vec{V}_{\beta}$$

### 5.3 Recurvature Mechanism
In the Northern Hemisphere (Bay of Bengal / Arabian Sea):
- When a cyclone moves north of $\sim 18^\circ - 20^\circ\text{N}$, it encounters the western periphery of the subtropical anticyclone (subtropical ridge).
- The westerly winds of mid-latitude troughs interact with the storm, causing the track to **recurve** from a northwestward heading to north-northeastward (towards Odisha, West Bengal, Bangladesh, or Gujarat/Pakistan).

---

## 6. Cone of Uncertainty (COU) & Landfall Raycasting Formulation

IMD's operational **Cone of Uncertainty** reflects empirical 5-year track forecast errors.

### 6.1 Empirical Forecast Error Envelopes (IMD Operational Standard)
The radius of the circular error probability at lead time $\tau$ hours is:
$$R_{\text{err}}(\tau) = 20.0\,\text{km} + 7.5\,\text{km/hr} \times \tau$$

| Forecast Horizon ($\tau$) | IMD Empirical Track Error Radius ($R_{\text{err}}$) | Typical Coastal Warning Lead Time |
| :--- | :--- | :--- |
| **+12 Hours** | $110\,\text{km}$ | Port warning, evacuation of low-lying areas. |
| **+24 Hours** | $200\,\text{km}$ | Coastal alert, suspension of fishing operations. |
| **+36 Hours** | $290\,\text{km}$ | State disaster management mobilization. |
| **+48 Hours** | $380\,\text{km}$ | Pre-cyclone watch bulletin. |

### 6.2 Geodetic Coastal Raycasting Algorithm
To determine the exact point of landfall $(\phi_{\text{landfall}}, \lambda_{\text{landfall}})$ and the Estimated Time of Arrival ($\text{ETA}$):
1. **Discrete Coastal Polyline**: Let the Indian coastline be represented by an ordered sequence of geodetic vertices:
   $$\mathcal{C} = \{\vec{c}_1, \vec{c}_2, \dots, \vec{c}_M\}, \quad \vec{c}_j = (\phi_j, \lambda_j)$$
2. **Forecast Track Segments**: Let the predicted trajectory waypoints be:
   $$\mathcal{T} = \{\vec{w}_0, \vec{w}_1, \dots, \vec{w}_K\}, \quad \vec{w}_k = (\phi_k, \lambda_k, \tau_k)$$
3. **Segment-Segment Raycasting**: For each segment $S_{\text{track}} = (\vec{w}_k, \vec{w}_{k+1})$ and coastal segment $S_{\text{coast}} = (\vec{c}_j, \vec{c}_{j+1})$:
   $$\vec{p}(s) = \vec{w}_k + s(\vec{w}_{k+1} - \vec{w}_k), \quad s \in [0, 1]$$
   $$\vec{q}(t) = \vec{c}_j + t(\vec{c}_{j+1} - \vec{c}_j), \quad t \in [0, 1]$$
4. The intersection point occurs where $\vec{p}(s^*) = \vec{q}(t^*)$ for $s^*, t^* \in [0, 1]$.
5. **Exact ETA Calculation**:
   $$\text{ETA} = \tau_k + s^* \cdot (\tau_{k+1} - \tau_k) \quad \text{[Hours from observation $T_0$]}$$
6. **Geodetic Distance to Landfall**:
   Using the Haversine spherical distance metric:
   $$D = 2 R_{\text{Earth}} \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_0 \cos\phi_{\text{landfall}} \sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$
   $$\text{ETA} \approx \frac{D}{|\vec{V}_{\text{along-track}}|}$$

---

## 7. Coastal Subdivision Classification & Sector Naming

IMD defines specific coastal meteorological subdivisions and maritime landmarks for warning dissemination:

```
                            INDIAN COASTLINE SECTORS
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        ▼                                                             ▼
   WEST COAST                                                    EAST COAST
   (Arabian Sea Basin)                                      (Bay of Bengal Basin)
   ├─ Gujarat: Kutch Coast                                   ├─ West Bengal: Sundarbans / Sagar Island
   ├─ Gujarat: Saurashtra (Okha, Porbandar, Veraval, Diu)    ├─ West Bengal: Digha / Contai
   ├─ South Gujarat (Surat, Daman)                          ├─ North Odisha (Balasore, Chandipur, Bhadrak)
   ├─ North Konkan (Palghar, Mumbai, Raigad)                 ├─ Central Odisha (Paradip, Kendrapara, Puri)
   ├─ South Konkan (Ratnagiri, Sindhudurg)                   ├─ South Odisha (Gopalpur, Ganjam)
   ├─ Goa Coast (Panaji, Mormugao)                          ├─ North Andhra (Visakhapatnam, Kalingapatnam)
   ├─ Karnataka (Karwar, Udupi, Mangalore)                  ├─ Central Andhra (Machilipatnam, Kakinada, Bapatla)
   └─ Kerala: Malabar & Travancore Coasts                   ├─ South Andhra (Nellore, Prakasam, Sriharikota)
                                                            ├─ North Tamil Nadu: Coromandel (Chennai, Cuddalore)
                                                            └─ South Tamil Nadu: Palk Strait & Gulf of Mannar
```

---

## 8. Inland Intensity Decay Modeling

Upon crossing the coastline, the cyclone is severed from its thermodynamic energy source—the warm ocean surface latent heat flux ($Q_{\text{latent}} = \rho C_E U (q_{\text{sst}} - q_a)$)—and experiences elevated surface friction ($z_0 \approx 0.1 - 1.0\,\text{m}$ over land vs. $0.001\,\text{m}$ over smooth ocean).

### 8.1 Kaplan & DeMaria (1995) Inland Decay Formulation
IMD adapts the inland filling model:
$$V(t) = V_{\text{background}} + (R \cdot V_0 - V_{\text{background}}) e^{-\alpha t} - \Delta V_{\text{terrain}}(t)$$
where:
- $V_0$: Intensity at moment of landfall ($\text{knots}$).
- $R \approx 0.90$: Reduction factor representing core friction adjustment during the final hour of coastal crossing.
- $\alpha \approx 0.095\,\text{hr}^{-1}$: Decay rate constant (half-life of intensity $\approx 7.3\,\text{hours}$).
- $V_{\text{background}} \approx 20\,\text{knots}$: Residual background tropical depression wind speed.

---

## 9. Project BLINK Implementation Architecture

In Project BLINK, this operational meteorology pipeline is translated into a real-time, GPU-accelerated kinematic engine:

```
MOSDAC INSAT-3DS / NASA VIIRS Multi-Spectral Granules (T0 & T1)
                            │
                            ▼
              `StormTrackPredictor` Engine
                            │
     ┌──────────────────────┼──────────────────────┐
     ▼                      ▼                      ▼
Vortex Centroid      Dvorak ADT & CI       RAFT Kinematic Optical Flow
(Spatial Moments)    (Eye/Wall Contrast)   (Advection Field & Steering)
     │                      │                      │
     └──────────────────────┼──────────────────────┘
                            ▼
                 Trajectory Extrapolation
              (Beta-Drift Recurvature Model)
                            │
                            ▼
        `COASTAL_POLYLINES_INDIA` Vector Raycaster
                            │
                            ▼
    ┌─────────────────────────────────────────────────┐
    │          Operational Landfall Telemetry         │
    │  - Coastal Sector & Nearest Port Landmark        │
    │  - Landfall Point (Latitude, Longitude)         │
    │  - Estimated Time of Arrival (ETA in Hours)      │
    │  - Expected Intensity & Max Wind at Landfall    │
    └─────────────────────────────────────────────────┘
```

---

## References

1. **Shukla, B. P., et al. (2017)**: *Algorithm Theoretical Basis Document (ATBD) for Rapid Scanning Products of INSAT-3D/3DR*. Space Applications Centre (ISRO), Ahmedabad.
2. **Courtney, J. B., & Knaff, J. A. (2009)**: *Adapting the Knaff and Zehr wind-pressure relationship for operational use in Tropical Cyclone warning centres*. Australian Meteorological and Oceanographic Journal, 58(3), 167-179.
3. **India Meteorological Department (2021)**: *Standard Operating Procedure for Cyclone Warning in India*. Cyclone Warning Division, Earth System Science Organisation, Ministry of Earth Sciences, New Delhi.
4. **Dvorak, V. F. (1984)**: *Tropical cyclone intensity analysis and forecasting from satellite imagery*. NOAA Technical Report NESDIS 11.
5. **Kaplan, J., & DeMaria, M. (1995)**: *A Simple Empirical Model for Predicting the Decay of Tropical Cyclone Winds after Landfall*. Journal of Applied Meteorology, 34(11), 2499-2512.
6. **Mohapatra, M., et al. (2013)**: *Verification of official tropical cyclone track and intensity forecasts issued by India Meteorological Department*. Journal of Earth System Science, 122(4), 1157-1171.
