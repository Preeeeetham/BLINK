/**
 * Project BLINK: Radiometric Planck Law Inversion & Calibration for INSAT-3D/3DR/3DS.
 * Converts multi-spectral radiance to equivalent blackbody brightness temperature (T_B).
 *
 * Calibration Standards: ISRO Space Applications Centre (SAC) Level-1B.
 */

const PLANCK_C1 = 1.191042e8; // mW / (m² · sr · cm^-4)
const PLANCK_C2 = 1.4387752;  // K · cm

const INSAT_CHANNELS = {
  TIR1: {
    name: "Thermal Infrared 1",
    wavelength_um: 10.8,
    central_wavenumber_cm1: 925.9,
    calibration_A: 0.9983,
    calibration_B: 0.45,
    resolution_km: 4.0,
  },
  TIR2: {
    name: "Thermal Infrared 2",
    wavelength_um: 12.0,
    central_wavenumber_cm1: 833.3,
    calibration_A: 0.998,
    calibration_B: 0.52,
    resolution_km: 4.0,
  },
  MIR: {
    name: "Mid-Wave Infrared",
    wavelength_um: 3.9,
    central_wavenumber_cm1: 2564.1,
    calibration_A: 0.9992,
    calibration_B: 0.12,
    resolution_km: 4.0,
  },
  WV: {
    name: "Water Vapour",
    wavelength_um: 6.8,
    central_wavenumber_cm1: 1470.6,
    calibration_A: 0.9975,
    calibration_B: 0.68,
    resolution_km: 8.0,
  },
  VIS: {
    name: "Visible",
    wavelength_um: 0.65,
    resolution_km: 1.0,
  },
  SWIR: {
    name: "Short-Wave Infrared",
    wavelength_um: 1.62,
    resolution_km: 1.0,
  },
};

/**
 * Converts spectral radiance into equivalent blackbody brightness temperature (Kelvin).
 */
function radianceToBrightnessTemperature(radiance, channelKey = "TIR1") {
  const ch = INSAT_CHANNELS[channelKey] || INSAT_CHANNELS.TIR1;
  if (!ch.central_wavenumber_cm1) {
    throw new Error(`Channel ${channelKey} is optical/reflective and does not emit thermal radiance.`);
  }

  const nu = ch.central_wavenumber_cm1;
  const num = PLANCK_C1 * Math.pow(nu, 3);
  const ratio = Math.max(1e-12, radiance);
  const tStar = (PLANCK_C2 * nu) / Math.log(1.0 + num / ratio);

  const tb = ch.calibration_A * tStar + ch.calibration_B;
  return Math.max(170.0, Math.min(340.0, tb));
}

/**
 * Inverts brightness temperature (Kelvin) to spectral radiance.
 */
function brightnessTemperatureToRadiance(btKelvin, channelKey = "TIR1") {
  const ch = INSAT_CHANNELS[channelKey] || INSAT_CHANNELS.TIR1;
  if (!ch.central_wavenumber_cm1) {
    throw new Error(`Channel ${channelKey} does not have Planck thermal calibration.`);
  }

  const nu = ch.central_wavenumber_cm1;
  const tStar = (btKelvin - ch.calibration_B) / ch.calibration_A;
  const exponent = (PLANCK_C2 * nu) / Math.max(10.0, tStar);
  const radiance = (PLANCK_C1 * Math.pow(nu, 3)) / (Math.exp(exponent) - 1.0);
  return Math.max(0.0, radiance);
}

module.exports = {
  PLANCK_C1,
  PLANCK_C2,
  INSAT_CHANNELS,
  radianceToBrightnessTemperature,
  brightnessTemperatureToRadiance,
};
