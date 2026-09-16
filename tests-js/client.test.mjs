import { test } from "node:test";
import assert from "node:assert/strict";

import { BlinkClient } from "../src-js/client/client.js";

test("BlinkClient initializes with default or custom options", () => {
  const defaultClient = new BlinkClient();
  assert.equal(defaultClient.baseUrl, "http://127.0.0.1:8000");

  const customClient = new BlinkClient({ baseUrl: "https://blink.isro.gov.in/", timeoutMs: 15000 });
  assert.equal(customClient.baseUrl, "https://blink.isro.gov.in");
  assert.equal(customClient.timeoutMs, 15000);
});

test("BlinkClient executes offline landfall predictions", () => {
  const client = new BlinkClient();
  const track = client.predictLandfall({
    centerLat: 15.5,
    centerLon: 84.5,
    headingDeg: 310.0,
    speedKmh: 24.0,
    vMaxKmh: 150.0,
    centralPressureHpa: 970.0,
  });

  assert.equal(track.is_active_cyclone, true);
  assert.ok(track.forecast_waypoints.length > 0);
  assert.ok(track.landfall_estimate !== null);
});

test("BlinkClient executes offline convective risk evaluations", () => {
  const client = new BlinkClient();
  const netra = client.calculateConvectiveRisk({
    minCoolingK15m: -42.0,
    minBtKelvin: 195.0,
    areaKm2: 2800.0,
  });

  assert.ok(netra.convective_activity_index > 75.0);
  assert.equal(netra.convective_level, "SEVERE_CONVECTIVE_UPDRAFT");
});

test("BlinkClient computes offline physical evaluation metrics", () => {
  const client = new BlinkClient();
  const synth = [0.5, 0.6, 0.7];
  const gt = [0.51, 0.59, 0.72];

  const res = client.evaluateSynthesis(synth, gt);
  assert.ok(res.psnr_db > 30.0);
  assert.ok(res.ssim > 0.95);
  assert.ok(res.rmse_k < 3.0);
});
