/**
 * Project BLINK: Convective Nowcasting (NETRA Algorithm & Convective Activity Index).
 * Scientifically grounded convective cloud shield segmentation and updraft kinetics.
 *
 * Implements strict scientific guardrails:
 * - Never outputs uncalibrated precipitation probabilities
 * - Explicit physical units: dT_B/dt in K/15-min
 * - Labeled as Experimental Updraft Proxies
 */

/**
 * Computes the Convective Activity Index (CAI, 0 to 100).
 * Measures physical updraft kinetics (cooling rate) and tropospheric penetration.
 *
 * @param {Object} options
 * @param {number} options.minCoolingK15m - Maximum cooling rate (negative value in K/15-min, e.g. -45.0)
 * @param {number} options.minBtKelvin - Minimum cloud-top brightness temperature in Kelvin (e.g. 195.0)
 * @param {number} options.areaKm2 - Convective core area in km² (e.g. 3200.0)
 * @returns {Object} Convective activity report with scientific guardrails
 */
function computeConvectiveActivityIndex({
  minCoolingK15m = 0.0,
  minBtKelvin = 260.0,
  areaKm2 = 500.0,
}) {
  const coolingTerm = Math.abs(Math.min(0.0, minCoolingK15m));
  const tempTerm = Math.max(0.0, 220.0 - minBtKelvin);
  const areaFactor = Math.min(2.0, Math.max(0.4, Math.sqrt(areaKm2 / 2500.0)));

  const rawCai = 12.0 * Math.sqrt(coolingTerm * (tempTerm + 1.0)) * areaFactor;
  const caiScore = Math.max(5.0, Math.min(100.0, rawCai));

  let convectiveLevel;
  let threatColor;
  if (caiScore >= 80.0) {
    convectiveLevel = "SEVERE_CONVECTIVE_UPDRAFT";
    threatColor = "#ef4444";
  } else if (caiScore >= 60.0) {
    convectiveLevel = "ELEVATED";
    threatColor = "#f97316";
  } else if (caiScore >= 40.0) {
    convectiveLevel = "MODERATE";
    threatColor = "#eab308";
  } else {
    convectiveLevel = "NOMINAL";
    threatColor = "#10b981";
  }

  return {
    convective_activity_index: Math.round(caiScore * 10) / 10,
    convective_level: convectiveLevel,
    cloud_top_bt_tendency_k_15m: Math.round(minCoolingK15m * 10) / 10,
    min_bt_kelvin: Math.round(minBtKelvin * 10) / 10,
    area_km2: Math.round(areaKm2 * 10) / 10,
    threat_color: threatColor,
    is_calibrated_probability: false,
    precipitation_status: "UNVALIDATED_NO_SURFACE_RADAR",
    disclaimer:
      "Experimental Heuristic Indicator · Not Calibrated Precipitation Probability",
  };
}

/**
 * Bedka et al. (2010) Overshooting Top (OT) proxy detection criteria.
 * Identifies local convective thermal anomalies against surrounding anvil cloud shield.
 */
function isOvershootingTopProxy({
  btKelvin,
  smoothedAnvilBtKelvin,
  coolingRateK15m = -3.0,
  minAnomalyK = 4.5,
}) {
  const isColdEnough = btKelvin < 210.0;
  const hasCoolingUpdraft = coolingRateK15m < -2.0;
  const isLocalThermalAnomaly = btKelvin <= smoothedAnvilBtKelvin - minAnomalyK;

  return isColdEnough && hasCoolingUpdraft && isLocalThermalAnomaly;
}

module.exports = {
  computeConvectiveActivityIndex,
  isOvershootingTopProxy,
};
