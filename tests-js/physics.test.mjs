import { test } from "node:test";
import assert from "node:assert/strict";

import {
  computePsnr,
  computePhysicalRmseKelvin,
  computeSsim,
  computeBtCompliance,
  computeRadianceFluxConservation,
} from "../src-js/evaluation/physics.js";

test("computePsnr returns 100 dB for identical arrays and finite dB for differences", () => {
  const arr1 = [0.5, 0.6, 0.7, 0.8];
  const arr2 = [0.5, 0.6, 0.7, 0.8];
  assert.equal(computePsnr(arr1, arr2), 100.0);

  const arrDiff = [0.52, 0.58, 0.71, 0.79];
  const psnr = computePsnr(arr1, arrDiff);
  assert.ok(psnr > 30.0 && psnr < 45.0, `Expected PSNR 30-45 dB, got ${psnr}`);
});

test("computePhysicalRmseKelvin scales normalized differences to physical Kelvin", () => {
  // If difference is 0.02 normalized across 105 K range, RMSE should be 0.02 * 105 = 2.1 K
  const a = [0.5, 0.5, 0.5];
  const b = [0.52, 0.52, 0.52];
  const rmseK = computePhysicalRmseKelvin(a, b, 105.0);
  assert.ok(Math.abs(rmseK - 2.1) < 0.01, `Expected 2.1 K, got ${rmseK}`);
});

test("computeSsim returns 1.0 for identical distributions", () => {
  const a = [0.1, 0.3, 0.5, 0.7, 0.9];
  const ssim = computeSsim(a, a);
  assert.ok(Math.abs(ssim - 1.0) < 1e-4, `Expected SSIM ~1.0, got ${ssim}`);
});

test("computeBtCompliance verifies physical atmospheric viability", () => {
  const validArray = [200.0, 240.0, 280.0, 310.0];
  assert.equal(computeBtCompliance(validArray), 100.0);

  const mixedArray = [150.0, 240.0, 280.0, 350.0]; // 2 out of 4 invalid (< 180 or > 330)
  assert.equal(computeBtCompliance(mixedArray), 50.0);
});

test("computeRadianceFluxConservation measures flux conservation against bounds", () => {
  const f0 = [0.4, 0.4];
  const f1 = [0.6, 0.6];
  const mid = [0.5, 0.5]; // perfect linear mass conservation at t=0.5
  assert.equal(computeRadianceFluxConservation(mid, f0, f1, 0.5), 100.0);
});
