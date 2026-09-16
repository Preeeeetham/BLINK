import { test } from "node:test";
import assert from "node:assert/strict";

import {
  computeConvectiveActivityIndex,
  isOvershootingTopProxy,
} from "../src-js/meteorology/netra.js";

test("computeConvectiveActivityIndex outputs high score for severe updraft with physical units", () => {
  const result = computeConvectiveActivityIndex({
    minCoolingK15m: -46.0,
    minBtKelvin: 194.0,
    areaKm2: 3200.0,
  });

  assert.ok(result.convective_activity_index >= 80.0, `Expected severe CAI (>=80), got ${result.convective_activity_index}`);
  assert.equal(result.convective_level, "SEVERE_CONVECTIVE_UPDRAFT");
  assert.equal(result.is_calibrated_probability, false);
  assert.equal(result.precipitation_status, "UNVALIDATED_NO_SURFACE_RADAR");
  assert.ok(result.disclaimer.includes("Experimental Heuristic"));
});

test("computeConvectiveActivityIndex outputs nominal score for quiescent clouds", () => {
  const result = computeConvectiveActivityIndex({
    minCoolingK15m: -0.5,
    minBtKelvin: 275.0,
    areaKm2: 200.0,
  });

  assert.ok(result.convective_activity_index < 40.0, `Expected nominal score (<40), got ${result.convective_activity_index}`);
  assert.equal(result.convective_level, "NOMINAL");
});

test("isOvershootingTopProxy satisfies Bedka et al. (2010) criteria", () => {
  // Anvil = 205 K, Core = 198 K (anomaly = 7 K >= 4.5 K), Cooling = -4 K/15m
  const isOt = isOvershootingTopProxy({
    btKelvin: 198.0,
    smoothedAnvilBtKelvin: 205.0,
    coolingRateK15m: -4.0,
    minAnomalyK: 4.5,
  });
  assert.equal(isOt, true);

  // Warm cloud (230 K) is not an overshooting top
  const isWarmOt = isOvershootingTopProxy({
    btKelvin: 230.0,
    smoothedAnvilBtKelvin: 240.0,
    coolingRateK15m: -5.0,
  });
  assert.equal(isWarmOt, false);
});
