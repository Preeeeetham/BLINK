import { test } from "node:test";
import assert from "node:assert/strict";

import {
  courtneyKnaffVmax,
  centralPressureFromVmax,
  dvorakCiLookup,
  classifyImdIntensity,
  applyKaplanDemariaDecay,
} from "../src-js/meteorology/dvorak.js";

test("courtneyKnaffVmax computes accurate sustained winds from pressure drop", () => {
  // P_c = 960 hPa -> Delta P = 50 hPa -> (50)^0.76 * 2.3 = 19.55 * 2.3 = 45.0 knots (~83.3 km/h)
  const res = courtneyKnaffVmax(960.0);
  assert.ok(res.v_knots > 40 && res.v_knots < 50, `Expected ~45 kt, got ${res.v_knots}`);
  assert.ok(res.v_kmh > 75 && res.v_kmh < 95, `Expected ~83 km/h, got ${res.v_kmh}`);
});

test("centralPressureFromVmax correctly inverts Courtney-Knaff formula", () => {
  const originalPc = 965.0;
  const winds = courtneyKnaffVmax(originalPc);
  const invertedPc = centralPressureFromVmax(winds.v_knots);
  assert.ok(Math.abs(invertedPc - originalPc) < 1.0, `Inversion error: ${invertedPc} vs ${originalPc}`);
});

test("classifyImdIntensity follows official IMD criteria", () => {
  assert.equal(classifyImdIntensity(230.0), "Super Cyclonic Storm (SuCS)");
  assert.equal(classifyImdIntensity(180.0), "Extremely Severe Cyclonic Storm (ESCS)");
  assert.equal(classifyImdIntensity(130.0), "Very Severe Cyclonic Storm (VSCS)");
  assert.equal(classifyImdIntensity(100.0), "Severe Cyclonic Storm (SCS)");
  assert.equal(classifyImdIntensity(70.0), "Cyclonic Storm (CS)");
  assert.equal(classifyImdIntensity(55.0), "Deep Depression (DD)");
  assert.equal(classifyImdIntensity(40.0), "Depression (D)");
});

test("applyKaplanDemariaDecay models inland friction decay", () => {
  const initialVmax = 180.0;
  const v12h = applyKaplanDemariaDecay(initialVmax, 12.0);
  const v24h = applyKaplanDemariaDecay(initialVmax, 24.0);

  assert.ok(v12h < initialVmax, `Winds at 12h (${v12h}) should be lower than initial (${initialVmax})`);
  assert.ok(v24h < v12h, `Winds at 24h (${v24h}) should be lower than at 12h (${v12h})`);
  assert.ok(v24h >= 35.0, "Decay should respect background friction minimum (35 km/h)");
});
