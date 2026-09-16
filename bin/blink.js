#!/usr/bin/env node

/**
 * Project BLINK: Unified Command-Line Interface.
 * Orchestrates Python backend, provides standalone meteorological nowcasting,
 * and hosts the operational satellite dashboard.
 */

const { spawn, execSync } = require("child_process");
const path = require("path");
const fs = require("fs");
const http = require("http");

const {
  predictTrackAndCone,
  raycastCoastalLandfall,
} = require("../src-js/meteorology/landfall.js");
const {
  computeConvectiveActivityIndex,
} = require("../src-js/meteorology/netra.js");
const {
  courtneyKnaffVmax,
  classifyImdIntensity,
} = require("../src-js/meteorology/dvorak.js");

const ROOT_DIR = path.resolve(__dirname, "..");
const ARGS = process.argv.slice(2);
const COMMAND = ARGS[0] ? ARGS[0].toLowerCase() : "help";

function printBanner() {
  console.log(`
========================================================================
   ____  _     ___ _   _ _  __
  | __ )| |   |_ _| \\ | | |/ /   PROJECT BLINK v1.0.0
  |  _ \\| |    | ||  \\| | ' /    Operational Satellite Kinematics &
  | |_) | |___ | || |\\  | . \\    INSAT Tropical Cyclone Landfall Engine
  |____/|_____|___|_| \\_|_|\\_\\   ISRO SAC / IMD Standards
========================================================================
`);
}

function parseArgValue(flag, defaultValue) {
  const idx = ARGS.indexOf(flag);
  if (idx !== -1 && idx + 1 < ARGS.length) {
    return ARGS[idx + 1];
  }
  return defaultValue;
}

function hasFlag(flag) {
  return ARGS.includes(flag);
}

async function handleStatus() {
  printBanner();
  console.log("Checking Project BLINK Server Status...\n");

  const req = http.get("http://127.0.0.1:8000/v1/health", (res) => {
    let data = "";
    res.on("data", (chunk) => (data += chunk));
    res.on("end", () => {
      try {
        const json = JSON.parse(data);
        console.log(`[ONLINE] BLINK Backend is Active!`);
        console.log(`- Status:     ${json.status}`);
        console.log(`- Device:     ${json.device}`);
        console.log(`- CUDA:       ${json.cuda_available ? "YES" : "NO (CPU Fallback)"}`);
        console.log(`- PyTorch:    v${json.torch_version}`);
        console.log(`- Models:     RAFT Kinematics / Triplet Ground-Truth Evaluator`);
        console.log(`- Memory:     ${json.active_memory_mb.toFixed(1)} MB`);
        console.log(`- Dashboard:  http://127.0.0.1:8000/\n`);
      } catch (e) {
        console.log(`[ONLINE] HTTP 200 OK (Raw response received)`);
      }
    });
  });

  req.on("error", () => {
    console.log(`[OFFLINE] BLINK Server is not responding on http://127.0.0.1:8000/`);
    console.log(`Run 'blink start' or 'npx blink-satellite start' to launch.`);
  });
}

function handleStart() {
  printBanner();
  const port = parseArgValue("--port", "8000");
  const host = hasFlag("--host") ? "0.0.0.0" : "127.0.0.1";

  console.log(`Launching BLINK Server daemon on http://${host}:${port}...`);
  try {
    const pythonExe = process.platform === "win32" ? "python" : "python3";
    const res = execSync(`${pythonExe} blink.py start`, {
      cwd: ROOT_DIR,
      encoding: "utf-8",
    });
    console.log(res);
  } catch (err) {
    console.error(`Error launching BLINK server: ${err.message}`);
    if (err.stdout) console.log(err.stdout);
    if (err.stderr) console.error(err.stderr);
  }
}

function handleStop() {
  printBanner();
  console.log("Stopping BLINK Server daemon...");
  try {
    const pythonExe = process.platform === "win32" ? "python" : "python3";
    const res = execSync(`${pythonExe} blink.py stop`, {
      cwd: ROOT_DIR,
      encoding: "utf-8",
    });
    console.log(res);
  } catch (err) {
    console.error(`Error stopping server: ${err.message}`);
  }
}

function handleNowcast() {
  printBanner();
  const lat = parseFloat(parseArgValue("--lat", "16.2"));
  const lon = parseFloat(parseArgValue("--lon", "84.5"));
  const heading = parseFloat(parseArgValue("--heading", "305.0"));
  const speed = parseFloat(parseArgValue("--speed", "22.0"));
  const vmax = parseFloat(parseArgValue("--vmax", "145.0"));
  const pressure = parseFloat(parseArgValue("--pressure", "972.0"));

  console.log("Operational Cyclone Landfall Prediction Engine");
  console.log("--------------------------------------------------");
  console.log(`Initial Position:   ${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E`);
  console.log(`Steering Heading:   ${heading.toFixed(1)}° (Beta-drift adjusted)`);
  console.log(`Translation Speed:  ${speed.toFixed(1)} km/h`);
  console.log(`Max Sustained Wind: ${vmax.toFixed(1)} km/h`);
  console.log(`Central Pressure:   ${pressure.toFixed(1)} hPa`);
  console.log(`IMD Category:       ${classifyImdIntensity(vmax)}\n`);

  const report = predictTrackAndCone({
    centerLat: lat,
    centerLon: lon,
    headingDeg: heading,
    speedKmh: speed,
    vMaxKmh: vmax,
    centralPressureHpa: pressure,
  });

  const lf = report.landfall_estimate;
  if (lf) {
    console.log("================ LANDFALL FORECAST ================");
    console.log(`Status:             ${lf.status}`);
    console.log(`Confidence:         ${lf.confidence}`);
    console.log(`Coastal Sector:     ${lf.estimated_region}`);
    console.log(`Estimated ETA:      ${lf.estimated_eta_hours} hours`);
    console.log(`Landfall Point:     ${lf.landfall_lat}°N, ${lf.landfall_lon}°E`);
    console.log(`Distance to Coast:  ${lf.distance_to_coast_km} km`);
    console.log(`Nearest Landmarks:  ${lf.nearest_landmarks}`);
    console.log(`Landfall Intensity: ${lf.estimated_intensity_at_landfall}`);
    console.log("===================================================\n");
  }

  console.log("72-Hour Kinematic Forecast Waypoints:");
  console.log(" Lead   |   Lat   |   Lon   | Wind (km/h) | Pressure | Cone Rad");
  console.log("-------+---------+---------+-------------+----------+---------");
  for (const wp of report.forecast_waypoints) {
    console.log(
      ` ${wp.timestamp_offset.padEnd(5)} | ${wp.lat.toFixed(2).padStart(7)} | ${wp.lon.toFixed(2).padStart(7)} | ${wp.max_sustained_winds_kmh.toFixed(1).padStart(11)} | ${wp.central_pressure_hpa.toFixed(1).padStart(8)} | ${wp.uncertainty_radius_km.toFixed(1).padStart(6)} km`
    );
  }
  console.log();
}

function handleNetra() {
  printBanner();
  const cooling = parseFloat(parseArgValue("--cooling", "-46.0"));
  const temp = parseFloat(parseArgValue("--temp", "194.0"));
  const area = parseFloat(parseArgValue("--area", "3200.0"));

  console.log("NETRA Convective Activity Index (CAI) Calculator");
  console.log("--------------------------------------------------");
  console.log(`Cloud-Top Cooling:  ${cooling.toFixed(1)} K / 15-min (dT_B/dt)`);
  console.log(`Minimum Core Temp:  ${temp.toFixed(1)} K (TIR-1 10.8 µm)`);
  console.log(`Convective Core:    ${area.toFixed(1)} km²\n`);

  const result = computeConvectiveActivityIndex({
    minCoolingK15m: cooling,
    minBtKelvin: temp,
    areaKm2: area,
  });

  console.log("================ CONVECTIVE REPORT ================");
  console.log(`CAI Score:          ${result.convective_activity_index} / 100`);
  console.log(`Threat Category:    ${result.convective_level}`);
  console.log(`Calibrated Prob:    ${result.is_calibrated_probability ? "YES" : "NO (Suppressed)"}`);
  console.log(`Precipitation:      ${result.precipitation_status}`);
  console.log(`Scientific Note:    ${result.disclaimer}`);
  console.log("===================================================\n");
}

function handleServe() {
  const serveScript = path.join(__dirname, "serve.js");
  require(serveScript);
}

function printHelp() {
  printBanner();
  console.log(`
Usage: blink <command> [options]
   or: npx blink-satellite <command> [options]

Commands:
  start          Starts the BLINK backend daemon (FastAPI + PyTorch)
  stop           Terminates the running BLINK server
  status         Checks server health, device status, and PID
  nowcast        Runs offline cyclone track and coastal landfall raycasting
  netra          Calculates Convective Activity Index (CAI) and updraft risk
  serve          Starts the Node.js dashboard server (default: port 3000)
  help           Displays this guidance

Options for 'nowcast':
  --lat <deg>       Storm initial latitude (default: 16.2)
  --lon <deg>       Storm initial longitude (default: 84.5)
  --heading <deg>   Initial azimuth heading in degrees (default: 305)
  --speed <km/h>    Translation speed in km/h (default: 22)
  --vmax <km/h>     Maximum sustained winds in km/h (default: 145)
  --pressure <hPa>  Central barometric pressure in hPa (default: 972)

Options for 'netra':
  --cooling <K>     Cloud-top cooling rate in K/15-min (default: -46.0)
  --temp <K>        Minimum brightness temperature in Kelvin (default: 194.0)
  --area <km2>      Convective core shield area in km² (default: 3200)

Options for 'serve':
  --port <number>   Port to listen on (default: 3000)
`);
}

switch (COMMAND) {
  case "start":
  case "launch":
    handleStart();
    break;
  case "stop":
    handleStop();
    break;
  case "status":
    handleStatus();
    break;
  case "nowcast":
  case "cyclone":
  case "landfall":
    handleNowcast();
    break;
  case "netra":
  case "convective":
    handleNetra();
    break;
  case "serve":
  case "ui":
  case "console":
    handleServe();
    break;
  case "help":
  case "--help":
  case "-h":
  default:
    printHelp();
    break;
}
