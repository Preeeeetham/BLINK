/**
 * Project BLINK: Operational Satellite Kinematics, Optical Flow Interpolation &
 * INSAT Tropical Cyclone Landfall Prediction Engine.
 *
 * Official Node.js & Browser JavaScript SDK.
 */

const {
  COASTAL_SECTORS_INDIA,
  haversineKm,
  initialBearingDeg,
  destinationPoint,
  raycastCoastalLandfall,
  predictTrackAndCone,
} = require("./meteorology/landfall.js");

const {
  courtneyKnaffVmax,
  centralPressureFromVmax,
  dvorakCiLookup,
  classifyImdIntensity,
  applyKaplanDemariaDecay,
} = require("./meteorology/dvorak.js");

const {
  computeConvectiveActivityIndex,
  isOvershootingTopProxy,
} = require("./meteorology/netra.js");

const {
  PLANCK_C1,
  PLANCK_C2,
  INSAT_CHANNELS,
  radianceToBrightnessTemperature,
  brightnessTemperatureToRadiance,
} = require("./radiometry/planck.js");

const {
  computePsnr,
  computePhysicalRmseKelvin,
  computeSsim,
  computeBtCompliance,
  computeRadianceFluxConservation,
} = require("./evaluation/physics.js");

const { BlinkClient } = require("./client/client.js");

module.exports = {
  // Client SDK
  BlinkClient,

  // Meteorology & Landfall
  COASTAL_SECTORS_INDIA,
  haversineKm,
  initialBearingDeg,
  destinationPoint,
  raycastCoastalLandfall,
  predictTrackAndCone,

  // Dvorak & Pressure-Wind Dynamics
  courtneyKnaffVmax,
  centralPressureFromVmax,
  dvorakCiLookup,
  classifyImdIntensity,
  applyKaplanDemariaDecay,

  // NETRA Convective Risk
  computeConvectiveActivityIndex,
  isOvershootingTopProxy,

  // Radiometric Inversion
  PLANCK_C1,
  PLANCK_C2,
  INSAT_CHANNELS,
  radianceToBrightnessTemperature,
  brightnessTemperatureToRadiance,

  // Physical Evaluation
  computePsnr,
  computePhysicalRmseKelvin,
  computeSsim,
  computeBtCompliance,
  computeRadianceFluxConservation,
};
