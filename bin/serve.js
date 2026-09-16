#!/usr/bin/env node

/**
 * Project BLINK: Standalone Node.js Console & Proxy Server.
 * Serves the satellite kinematics dashboard UI and proxies API requests to the Python backend
 * or handles them directly via the local JavaScript meteorological engine.
 */

const http = require("http");
const fs = require("fs");
const path = require("path");

const { predictTrackAndCone } = require("../src-js/meteorology/landfall.js");
const { computeConvectiveActivityIndex } = require("../src-js/meteorology/netra.js");

const ROOT_DIR = path.resolve(__dirname, "..");
const HTML_FILE = path.join(ROOT_DIR, "ui", "static", "index.html");

const args = process.argv.slice(2);
const portIdx = args.indexOf("--port");
const PORT = portIdx !== -1 && args[portIdx + 1] ? parseInt(args[portIdx + 1], 10) : 3000;
const BACKEND_URL = "http://127.0.0.1:8000";

const server = http.createServer(async (req, res) => {
  const url = req.url || "/";

  // Root: serve dashboard HTML
  if (url === "/" || url === "/index.html") {
    try {
      const content = fs.readFileSync(HTML_FILE, "utf-8");
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(content);
      return;
    } catch (err) {
      res.writeHead(500, { "Content-Type": "text/plain" });
      res.end(`Failed to read dashboard template: ${err.message}`);
      return;
    }
  }

  // Health endpoint
  if (url === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        status: "healthy",
        server: "BLINK-Node-Engine",
        version: "1.0.0",
        node_version: process.version,
      })
    );
    return;
  }

  // Offline nowcasting fallback endpoint
  if (url.startsWith("/v1/nowcasting/telemetry") && req.method === "POST") {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      try {
        const params = body ? JSON.parse(body) : {};
        const cyclone = predictTrackAndCone({
          centerLat: params.centerLat || 16.2,
          centerLon: params.centerLon || 84.5,
          headingDeg: params.headingDeg || 305.0,
          speedKmh: params.speedKmh || 22.0,
          vMaxKmh: params.vMaxKmh || 140.0,
          centralPressureHpa: params.centralPressureHpa || 975.0,
        });

        const netra = computeConvectiveActivityIndex({
          minCoolingK15m: -46.0,
          minBtKelvin: 194.0,
          areaKm2: 3200.0,
        });

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ cyclone, convective: netra }));
      } catch (err) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // Transparent proxy to Python backend on port 8000
  try {
    const targetUrl = new URL(url, BACKEND_URL);
    const proxyReq = http.request(
      targetUrl,
      {
        method: req.method,
        headers: { ...req.headers, host: "127.0.0.1:8000" },
      },
      (proxyRes) => {
        res.writeHead(proxyRes.statusCode || 200, proxyRes.headers);
        proxyRes.pipe(res);
      }
    );

    proxyReq.on("error", () => {
      res.writeHead(502, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          error: "BLINK Python backend is offline.",
          hint: "Start backend with 'blink start' or 'npx blink-satellite start'",
        })
      );
    });

    req.pipe(proxyReq);
  } catch (err) {
    res.writeHead(500, { "Content-Type": "text/plain" });
    res.end(`Proxy error: ${err.message}`);
  }
});

server.listen(PORT, () => {
  console.log(`
========================================================================
  Project BLINK Node Server Active
  Dashboard: http://127.0.0.1:${PORT}/
  Backend:   ${BACKEND_URL}
========================================================================
`);
});

module.exports = server;
