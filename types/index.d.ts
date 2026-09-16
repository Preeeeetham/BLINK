/**
 * Project BLINK: TypeScript Declarations.
 * Operational Satellite Kinematics & INSAT Tropical Cyclone Landfall Prediction Engine.
 */

export interface CoastalSector {
  region_name: string;
  landmarks: string;
  coords: [number, number][];
}

export interface TrackWaypoint {
  lead_hours: number;
  timestamp_offset: string;
  lat: number;
  lon: number;
  max_sustained_winds_knots: number;
  max_sustained_winds_kmh: number;
  central_pressure_hpa: number;
  uncertainty_radius_km: number;
}

export interface LandfallEstimate {
  estimated_region: string;
  estimated_eta_hours: number;
  landfall_lat: number;
  landfall_lon: number;
  distance_to_coast_km: number;
  nearest_landmarks: string;
  estimated_intensity_at_landfall: string;
  status: "DIRECT_COASTAL_INTERSECTION" | "COASTAL_SKIRTING_TRAJECTORY" | "RECURVING_OPEN_OCEAN" | "NO_WAYPOINTS";
  confidence: string;
}

export interface StormTrackReport {
  is_active_cyclone: boolean;
  current_center_lat: number;
  current_center_lon: number;
  translation_speed_kmh: number;
  heading_deg: number;
  intensity_category: string;
  max_winds_kmh: number;
  max_winds_knots: number;
  central_pressure_hpa: number;
  landfall_estimate: LandfallEstimate | null;
  forecast_waypoints: TrackWaypoint[];
  cone_polygon_coords: [number, number][];
}

export interface ConvectiveActivityReport {
  convective_activity_index: number;
  convective_level: "NOMINAL" | "MODERATE" | "ELEVATED" | "SEVERE_CONVECTIVE_UPDRAFT";
  cloud_top_bt_tendency_k_15m: number;
  min_bt_kelvin: number;
  area_km2: number;
  threat_color: string;
  is_calibrated_probability: false;
  precipitation_status: string;
  disclaimer: string;
}

export interface PhysicsEvaluationResult {
  psnr_db: number;
  ssim: number;
  rmse_k: number;
}

export interface InsatChannelConfig {
  name: string;
  wavelength_um: number;
  central_wavenumber_cm1?: number;
  calibration_A?: number;
  calibration_B?: number;
  resolution_km: number;
}

export interface BlinkClientOptions {
  baseUrl?: string;
  timeoutMs?: number;
}

export declare class BlinkClient {
  baseUrl: string;
  timeoutMs: number;
  constructor(options?: BlinkClientOptions);
  health(): Promise<any>;
  fetchRealtime(options?: {
    date?: string;
    cadenceMinutes?: number;
    regionKey?: string;
    targetSize?: number;
  }): Promise<any>;
  interpolate(options?: {
    frame0: any;
    frame1: any;
    steps?: number;
    evaluationProtocol?: string;
  }): Promise<any>;
  getNowcastingTelemetry(options?: any): Promise<any>;
  getBenchmarkReport(): Promise<any>;
  predictLandfall(stormParams: {
    centerLat: number;
    centerLon: number;
    headingDeg?: number;
    speedKmh?: number;
    vMaxKmh?: number;
    centralPressureHpa?: number;
    leadHoursList?: number[];
  }): StormTrackReport;
  calculateConvectiveRisk(cloudParams: {
    minCoolingK15m?: number;
    minBtKelvin?: number;
    areaKm2?: number;
  }): ConvectiveActivityReport;
  evaluateSynthesis(
    synthArray: number[] | Float32Array,
    gtArray: number[] | Float32Array,
    dataRange?: number
  ): PhysicsEvaluationResult;
}

export declare const COASTAL_SECTORS_INDIA: CoastalSector[];
export declare function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number;
export declare function initialBearingDeg(lat1: number, lon1: number, lat2: number, lon2: number): number;
export declare function destinationPoint(lat: number, lon: number, distanceKm: number, bearingDeg: number): { lat: number; lon: number };
export declare function raycastCoastalLandfall(
  curLat: number,
  curLon: number,
  waypoints: TrackWaypoint[],
  vMaxKmh: number,
  category: string,
  coastalSectors?: CoastalSector[]
): LandfallEstimate;
export declare function predictTrackAndCone(options: {
  centerLat: number;
  centerLon: number;
  headingDeg?: number;
  speedKmh?: number;
  vMaxKmh?: number;
  centralPressureHpa?: number;
  leadHoursList?: number[];
}): StormTrackReport;

export declare function courtneyKnaffVmax(centralPressureHpa: number): { v_knots: number; v_kmh: number };
export declare function centralPressureFromVmax(vmaxKnots: number): number;
export declare function dvorakCiLookup(ciNumber: number): {
  ci_number: number;
  v_knots: number;
  v_kmh: number;
  central_pressure_hpa: number;
  category: string;
};
export declare function classifyImdIntensity(vmaxKmh: number): string;
export declare function applyKaplanDemariaDecay(vmaxKmh: number, hoursInland: number): number;

export declare function computeConvectiveActivityIndex(options: {
  minCoolingK15m?: number;
  minBtKelvin?: number;
  areaKm2?: number;
}): ConvectiveActivityReport;
export declare function isOvershootingTopProxy(options: {
  btKelvin: number;
  smoothedAnvilBtKelvin: number;
  coolingRateK15m?: number;
  minAnomalyK?: number;
}): boolean;

export declare const PLANCK_C1: number;
export declare const PLANCK_C2: number;
export declare const INSAT_CHANNELS: Record<string, InsatChannelConfig>;
export declare function radianceToBrightnessTemperature(radiance: number, channelKey?: string): number;
export declare function brightnessTemperatureToRadiance(btKelvin: number, channelKey?: string): number;

export declare function computePsnr(synth: number[] | Float32Array, target: number[] | Float32Array, dataRange?: number): number;
export declare function computePhysicalRmseKelvin(synthNorm: number[] | Float32Array, targetNorm: number[] | Float32Array, kelvinRange?: number): number;
export declare function computeSsim(synth: number[] | Float32Array, target: number[] | Float32Array, dataRange?: number): number;
export declare function computeBtCompliance(btArrayKelvin: number[] | Float32Array): number;
export declare function computeRadianceFluxConservation(
  pred: number[] | Float32Array,
  frame0: number[] | Float32Array,
  frame1: number[] | Float32Array,
  tNormalized?: number
): number;
