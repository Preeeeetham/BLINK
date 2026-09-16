import { test } from "node:test";
import assert from "node:assert/strict";

import {
  haversineKm,
  initialBearingDeg,
  destinationPoint,
  raycastCoastalLandfall,
  predictTrackAndCone,
  COASTAL_SECTORS_INDIA,
} from "../src-js/meteorology/landfall.js";

test("haversineKm calculates accurate Great Circle distance", () => {
  // Chennai (13.08, 80.27) to Port Blair (11.62, 92.73) ~ 1367 km
  const dist = haversineKm(13.08, 80.27, 11.62, 92.73);
  assert.ok(dist > 1350 && dist < 1390, `Distance ${dist} should be ~1367 km`);

  // Zero distance for identical points
  assert.equal(haversineKm(20.0, 85.0, 20.0, 85.0), 0.0);
});

test("initialBearingDeg calculates correct azimuth heading", () => {
  // Northward
  const brgNorth = initialBearingDeg(10.0, 80.0, 15.0, 80.0);
  assert.ok(Math.abs(brgNorth - 0.0) < 0.1 || Math.abs(brgNorth - 360.0) < 0.1);

  // Eastward
  const brgEast = initialBearingDeg(10.0, 80.0, 10.0, 85.0);
  assert.ok(Math.abs(brgEast - 90.0) < 0.5);

  // Westward / Northwestward
  const brgNw = initialBearingDeg(15.0, 85.0, 18.0, 82.0);
  assert.ok(brgNw > 300 && brgNw < 330, `Bearing ${brgNw} should be northwestward (~315°)`);
});

test("destinationPoint projects forward geodesic coordinates", () => {
  const startLat = 15.0;
  const startLon = 85.0;
  const dest = destinationPoint(startLat, startLon, 111.0, 0.0); // 111 km due North
  assert.ok(Math.abs(dest.lat - 16.0) < 0.1, `Expected ~16.0°N, got ${dest.lat}`);
  assert.ok(Math.abs(dest.lon - 85.0) < 0.1, `Expected ~85.0°E, got ${dest.lon}`);
});

test("raycastCoastalLandfall detects direct intersection on Andhra Pradesh coast", () => {
  // Waypoints moving from Bay of Bengal towards Krishna-Godavari Delta (Andhra Pradesh)
  const waypoints = [
    { lead_hours: 6, lat: 15.8, lon: 83.0, lead_hours: 6.0 },
    { lead_hours: 12, lat: 16.0, lon: 81.8, lead_hours: 12.0 },
    { lead_hours: 18, lat: 16.3, lon: 80.4, lead_hours: 18.0 },
  ];

  const result = raycastCoastalLandfall(15.5, 84.5, waypoints, 140.0, "Very Severe Cyclonic Storm (VSCS)");
  assert.equal(result.status, "DIRECT_COASTAL_INTERSECTION");
  assert.ok(result.estimated_region.includes("Andhra Pradesh"), `Expected AP sector, got ${result.estimated_region}`);
  assert.ok(result.estimated_eta_hours > 6.0 && result.estimated_eta_hours < 18.0);
  assert.ok(result.distance_to_coast_km > 0.0);
  assert.ok(result.nearest_landmarks.length > 0);
});

test("predictTrackAndCone produces valid waypoints and Cone of Uncertainty", () => {
  const report = predictTrackAndCone({
    centerLat: 16.0,
    centerLon: 85.0,
    headingDeg: 310.0,
    speedKmh: 20.0,
    vMaxKmh: 135.0,
    centralPressureHpa: 978.0,
  });

  assert.equal(report.is_active_cyclone, true);
  assert.ok(report.forecast_waypoints.length >= 7);
  assert.ok(report.cone_polygon_coords.length > 10);
  assert.ok(report.landfall_estimate !== null);
  assert.equal(report.intensity_category, "Very Severe Cyclonic Storm (VSCS)");
});
