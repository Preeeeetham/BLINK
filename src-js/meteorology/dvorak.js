/**
 * Project BLINK: Dvorak ADT & Courtney-Knaff (2009) Cyclone Intensity Engine.
 * Formulates wind-pressure dynamics and IMD official operational classifications.
 */

/**
 * Computes maximum sustained wind speed (V_max) from central pressure (P_c)
 * using Courtney & Knaff (2009) empirical formulation:
 * V_max(kt) = 2.3 * (1010 - P_c)^0.76
 */
function courtneyKnaffVmax(centralPressureHpa) {
  const pDrop = Math.max(0.0, 1010.0 - centralPressureHpa);
  const vKnots = 2.3 * Math.pow(pDrop, 0.76);
  const vKmh = vKnots * 1.852;
  return {
    v_knots: Math.round(vKnots * 10) / 10,
    v_kmh: Math.round(vKmh * 10) / 10,
  };
}

/**
 * Inverts Courtney & Knaff (2009) to compute central pressure (P_c) from V_max in knots.
 */
function centralPressureFromVmax(vmaxKnots) {
  if (vmaxKnots <= 0) return 1010.0;
  const pDrop = Math.pow(vmaxKnots / 2.3, 1.0 / 0.76);
  const pc = Math.max(900.0, Math.min(1010.0, 1010.0 - pDrop));
  return Math.round(pc * 10) / 10;
}

/**
 * Translates Current Intensity (CI) number (2.0 to 8.0) into sustained winds and pressure.
 */
function dvorakCiLookup(ciNumber) {
  const ciClamped = Math.max(1.0, Math.min(8.0, ciNumber));
  const vKnots = Math.max(30.0, Math.min(155.0, 35.0 + (ciClamped - 2.5) * 23.0));
  const vKmh = vKnots * 1.852;
  const centralPressure = centralPressureFromVmax(vKnots);
  const category = classifyImdIntensity(vKmh);

  return {
    ci_number: ciClamped,
    v_knots: Math.round(vKnots * 10) / 10,
    v_kmh: Math.round(vKmh * 10) / 10,
    central_pressure_hpa: centralPressure,
    category,
  };
}

/**
 * Categorizes system into official India Meteorological Department (IMD) intensity scale.
 */
function classifyImdIntensity(vmaxKmh) {
  if (vmaxKmh >= 222.0) {
    return "Super Cyclonic Storm (SuCS)";
  } else if (vmaxKmh >= 166.0) {
    return "Extremely Severe Cyclonic Storm (ESCS)";
  } else if (vmaxKmh >= 118.0) {
    return "Very Severe Cyclonic Storm (VSCS)";
  } else if (vmaxKmh >= 89.0) {
    return "Severe Cyclonic Storm (SCS)";
  } else if (vmaxKmh >= 62.0) {
    return "Cyclonic Storm (CS)";
  } else if (vmaxKmh >= 50.0) {
    return "Deep Depression (DD)";
  } else {
    return "Depression (D)";
  }
}

/**
 * Applies Kaplan & DeMaria inland decay model to maximum winds after landfall.
 */
function applyKaplanDemariaDecay(vmaxKmh, hoursInland) {
  if (hoursInland <= 0) return vmaxKmh;
  const alpha = 0.095; // Decay rate constant (hr^-1)
  const vBackground = 35.0; // Background friction residual (km/h)
  const decayed = vBackground + (vmaxKmh - vBackground) * Math.exp(-alpha * hoursInland);
  return Math.max(vBackground, Math.round(decayed * 10) / 10);
}

module.exports = {
  courtneyKnaffVmax,
  centralPressureFromVmax,
  dvorakCiLookup,
  classifyImdIntensity,
  applyKaplanDemariaDecay,
};
