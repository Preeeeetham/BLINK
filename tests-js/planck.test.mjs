import { test } from "node:test";
import assert from "node:assert/strict";

import {
  INSAT_CHANNELS,
  radianceToBrightnessTemperature,
  brightnessTemperatureToRadiance,
} from "../src-js/radiometry/planck.js";

test("INSAT channels contain official spectral calibrations", () => {
  assert.equal(INSAT_CHANNELS.TIR1.wavelength_um, 10.8);
  assert.equal(INSAT_CHANNELS.TIR2.wavelength_um, 12.0);
  assert.equal(INSAT_CHANNELS.MIR.wavelength_um, 3.9);
  assert.equal(INSAT_CHANNELS.WV.wavelength_um, 6.8);
  assert.equal(INSAT_CHANNELS.VIS.wavelength_um, 0.65);
});

test("Planck inversion roundtrips radiance and brightness temperature", () => {
  const originalBt = 285.0; // Kelvin (~12°C)
  const radiance = brightnessTemperatureToRadiance(originalBt, "TIR1");
  assert.ok(radiance > 0.0, "Radiance must be positive");

  const invertedBt = radianceToBrightnessTemperature(radiance, "TIR1");
  assert.ok(Math.abs(invertedBt - originalBt) < 0.05, `Planck roundtrip error: ${invertedBt} vs ${originalBt}`);
});
