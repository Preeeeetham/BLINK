/**
 * Project BLINK: Physical Evaluation Suite (PSNR, SSIM, and Kelvin RMSE).
 * Grounded in independent verification against held-out observations.
 */

/**
 * Computes Peak Signal-to-Noise Ratio (dB) with explicit data_range.
 */
function computePsnr(synth, target, dataRange = 1.0) {
  if (synth.length !== target.length) {
    throw new Error(`Array length mismatch: ${synth.length} vs ${target.length}`);
  }
  let sumSq = 0.0;
  for (let i = 0; i < synth.length; i++) {
    const diff = synth[i] - target[i];
    sumSq += diff * diff;
  }
  const mse = sumSq / synth.length;
  if (mse <= 1e-10) return 100.0;
  return 10.0 * Math.log10((dataRange * dataRange) / mse);
}

/**
 * Computes physical Root Mean Square Error in Kelvin (K).
 * Normalized input [0, 1] maps to [298 K, 193 K] with 105 K range.
 */
function computePhysicalRmseKelvin(synthNorm, targetNorm, kelvinRange = 105.0) {
  if (synthNorm.length !== targetNorm.length) {
    throw new Error(`Array length mismatch: ${synthNorm.length} vs ${targetNorm.length}`);
  }
  let sumSqK = 0.0;
  for (let i = 0; i < synthNorm.length; i++) {
    const diffNorm = synthNorm[i] - targetNorm[i];
    const diffK = diffNorm * kelvinRange;
    sumSqK += diffK * diffK;
  }
  const mseK = sumSqK / synthNorm.length;
  return Math.sqrt(Math.max(0.0, mseK));
}

/**
 * Computes global Structural Similarity Index (SSIM) on flattened arrays.
 */
function computeSsim(synth, target, dataRange = 1.0) {
  if (synth.length !== target.length) {
    throw new Error(`Array length mismatch: ${synth.length} vs ${target.length}`);
  }
  const N = synth.length;
  let sumX = 0.0;
  let sumY = 0.0;
  for (let i = 0; i < N; i++) {
    sumX += synth[i];
    sumY += target[i];
  }
  const meanX = sumX / N;
  const meanY = sumY / N;

  let varX = 0.0;
  let varY = 0.0;
  let covXY = 0.0;
  for (let i = 0; i < N; i++) {
    const dx = synth[i] - meanX;
    const dy = target[i] - meanY;
    varX += dx * dx;
    varY += dy * dy;
    covXY += dx * dy;
  }
  varX /= (N - 1);
  varY /= (N - 1);
  covXY /= (N - 1);

  const c1 = (0.01 * dataRange) ** 2;
  const c2 = (0.03 * dataRange) ** 2;

  const numerator = (2.0 * meanX * meanY + c1) * (2.0 * covXY + c2);
  const denominator = (meanX * meanX + meanY * meanY + c1) * (varX + varY + c2);

  if (denominator <= 1e-12) return 1.0;
  return Math.max(-1.0, Math.min(1.0, numerator / denominator));
}

/**
 * Calculates percentage of pixels falling within physically viable atmospheric limits [180 K, 330 K].
 */
function computeBtCompliance(btArrayKelvin) {
  if (btArrayKelvin.length === 0) return 100.0;
  let validCount = 0;
  for (let i = 0; i < btArrayKelvin.length; i++) {
    if (btArrayKelvin[i] >= 180.0 && btArrayKelvin[i] <= 330.0) {
      validCount++;
    }
  }
  return (validCount / btArrayKelvin.length) * 100.0;
}

/**
 * Calculates percentage total radiative flux conservation against conservative linear bounds.
 */
function computeRadianceFluxConservation(pred, frame0, frame1, tNormalized = 0.5) {
  let sumPred = 0.0;
  let sumF0 = 0.0;
  let sumF1 = 0.0;
  const N = pred.length;

  for (let i = 0; i < N; i++) {
    sumPred += pred[i];
    sumF0 += frame0[i];
    sumF1 += frame1[i];
  }

  const predMean = sumPred / N;
  const expectedMean = (1.0 - tNormalized) * (sumF0 / N) + tNormalized * (sumF1 / N);

  if (expectedMean < 1e-6) return 100.0;

  const relError = Math.abs(predMean - expectedMean) / expectedMean;
  return Math.max(0.0, (1.0 - relError) * 100.0);
}

module.exports = {
  computePsnr,
  computePhysicalRmseKelvin,
  computeSsim,
  computeBtCompliance,
  computeRadianceFluxConservation,
};
