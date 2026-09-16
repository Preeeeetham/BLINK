/**
 * Project BLINK: Official JavaScript & TypeScript Client SDK.
 * Interacts with the BLINK FastAPI backend and provides local offline meteorological computations.
 */

const { raycastCoastalLandfall, predictTrackAndCone } = require("../meteorology/landfall.js");
const { computeConvectiveActivityIndex } = require("../meteorology/netra.js");
const { courtneyKnaffVmax, classifyImdIntensity } = require("../meteorology/dvorak.js");
const { computePsnr, computeSsim, computePhysicalRmseKelvin } = require("../evaluation/physics.js");

class BlinkClient {
  constructor(options = {}) {
    this.baseUrl = (options.baseUrl || "http://127.0.0.1:8000").replace(/\/$/, "");
    this.timeoutMs = options.timeoutMs || 30000;
  }

  async _request(path, method = "GET", body = null) {
    const url = `${this.baseUrl}${path}`;
    const headers = { "Accept": "application/json" };
    if (body) headers["Content-Type"] = "application/json";

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const resp = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(`HTTP ${resp.status} on ${path}: ${errText}`);
      }
      return await resp.json();
    } catch (err) {
      clearTimeout(timer);
      throw err;
    }
  }

  /**
   * Checks the health and hardware acceleration of the BLINK backend.
   */
  async health() {
    return await this._request("/health");
  }

  /**
   * Fetches real-time INSAT observation triplets (T_0, T_mid, T_1) and runs neural synthesis
   * with genuine held-out ground truth verification.
   */
  async fetchRealtime(options = {}) {
    return await this._request("/v1/fetch/realtime", "POST", {
      date: options.date || new Date().toISOString().split("T")[0],
      cadence_minutes: options.cadenceMinutes || 15,
      region_key: options.regionKey || "INDIA_FULL_DISK",
      target_size: options.targetSize || 512,
    });
  }

  /**
   * Submits two frames for neural temporal interpolation and physical evaluation.
   */
  async interpolate(options = {}) {
    return await this._request("/v1/interpolate/upload", "POST", {
      frame_0: options.frame0,
      frame_1: options.frame1,
      steps: options.steps || 1,
      evaluation_protocol: options.evaluationProtocol || "HELD_OUT_OBSERVATION",
    });
  }

  /**
   * Fetches the latest nowcasting and cyclone telemetry from the backend.
   */
  async getNowcastingTelemetry(options = {}) {
    return await this._request("/v1/nowcasting/telemetry", "POST", options);
  }

  /**
   * Fetches the verified scientific benchmark report.
   */
  async getBenchmarkReport() {
    return await this._request("/v1/benchmark/report");
  }

  /**
   * Offline / Standalone Coastal Landfall Prediction.
   * Runs directly in Node.js or browser without needing a running Python server.
   */
  predictLandfall(stormParams) {
    return predictTrackAndCone(stormParams);
  }

  /**
   * Offline / Standalone Convective Activity Index (NETRA) calculation.
   */
  calculateConvectiveRisk(cloudParams) {
    return computeConvectiveActivityIndex(cloudParams);
  }

  /**
   * Evaluates image synthesis fidelity against ground truth (PSNR, SSIM, and RMSE in K).
   */
  evaluateSynthesis(synthArray, gtArray, dataRange = 1.0) {
    return {
      psnr_db: computePsnr(synthArray, gtArray, dataRange),
      ssim: computeSsim(synthArray, gtArray, dataRange),
      rmse_k: computePhysicalRmseKelvin(synthArray, gtArray, 105.0),
    };
  }
}

module.exports = {
  BlinkClient,
};
