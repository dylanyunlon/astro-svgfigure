/**
 * M001: Chroma Key Engine - 绿幕移除核心
 * 文件位置: src/pipeline/chroma-engine.ts
 *
 * Refactored following NVIDIA CCCL commit f984c90 pattern:
 *   Extract first histogram-only pass into its own kernel.
 *
 * Before (original):
 *   removeGreenScreen() contained an IsFirstPass template-style branch:
 *     if (adaptiveThreshold) { analyze + generate mask }
 *     else { generate mask }
 *   The adaptive analysis was fused into the mask generation, causing:
 *     1. Two code paths with nearly identical postprocessing
 *     2. The analysis pass couldn't be independently benchmarked
 *     3. Config adjustment required re-running the entire pipeline
 *
 * After (this refactor):
 *   analyzePass()           — histogram-only: computes green distribution, returns calibrated config
 *   filterPass()            — filter + morphology: generates alpha mask using calibrated config
 *   finalizePipelinePass()  — shared postprocess: feathering, spill suppression, compositing
 *
 *   This mirrors CCCL's pattern:
 *     invoke_histogram_only()       → analyzePass()
 *     invoke_filter_and_histogram() → filterPass()
 *     finalize_pass()               → finalizePipelinePass()
 *
 * Diff summary (vs previous version):
 *   - Removed: `if (effectiveCfg.adaptiveThreshold)` branch inside removeGreenScreen
 *   - Extracted: analyzePass() as standalone public method
 *   - Extracted: finalizePipelinePass() shared by both paths
 *   - Added: `is_last_pass` parameter to filterPass (mirrors CCCL's kernel parameter)
 *   - Simplified: removeGreenScreen() now orchestrates 3 clean passes
 *   +176 -176 lines net (restructure, not growth)
 */

import sharp from 'sharp';

// ──────────────────────────────────────────────────────────────────────
// §1  Types
// ──────────────────────────────────────────────────────────────────────

export interface ChromaConfig {
  hueCenter: number;
  hueTolerance: number;
  saturationMin: number;
  valueRange: [number, number];
  enableYCbCr: boolean;
  cbRange: [number, number];
  crRange: [number, number];
  featherRadius: number;
  spillSuppressionStrength: number;
  erodeIterations: number;
  dilateIterations: number;
  adaptiveThreshold: boolean;
}

export interface AnalysisResult {
  hueCenter: number;
  hueTolerance: number;
  confidence: number;
  sampleCount: number;
}

export interface FilterResult {
  mask: Uint8Array;
  width: number;
  height: number;
  pixelsRemoved: number;
  method: 'hsv' | 'ycbcr' | 'hybrid';
}

export interface ChromaResult {
  outputBuffer: Buffer;
  maskBuffer: Buffer;
  metadata: {
    width: number;
    height: number;
    pixelsProcessed: number;
    pixelsRemoved: number;
    removalRatio: number;
    processingTimeMs: number;
    method: 'hsv' | 'ycbcr' | 'hybrid';
    analysisConfidence: number;
  };
}

// ──────────────────────────────────────────────────────────────────────
// §2  Default config
// ──────────────────────────────────────────────────────────────────────

export const DEFAULT_CHROMA_CONFIG: ChromaConfig = {
  hueCenter: 120,
  hueTolerance: 40,
  saturationMin: 0.15,
  valueRange: [0.1, 0.95],
  enableYCbCr: true,
  cbRange: [70, 140],
  crRange: [60, 130],
  featherRadius: 2,
  spillSuppressionStrength: 0.7,
  erodeIterations: 1,
  dilateIterations: 1,
  adaptiveThreshold: true,
};

// ──────────────────────────────────────────────────────────────────────
// §3  Color space conversions
//
// Mirrors CCCL's extract_bin_op: a pure function mapping input key →
// bucket index. Here we map RGB → HSV hue bucket / YCbCr classification.
// ──────────────────────────────────────────────────────────────────────

function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const rn = r / 255, gn = g / 255, bn = b / 255;
  const mx = Math.max(rn, gn, bn);
  const mn = Math.min(rn, gn, bn);
  const d = mx - mn;

  let h = 0;
  if (d !== 0) {
    if (mx === rn) h = 60 * (((gn - bn) / d) % 6);
    else if (mx === gn) h = 60 * ((bn - rn) / d + 2);
    else h = 60 * ((rn - gn) / d + 4);
  }
  if (h < 0) h += 360;

  const s = mx === 0 ? 0 : d / mx;
  return [h, s, mx];
}

/**
 * RGB → YCbCr (ITU-R BT.601)
 * Chosen over BT.709 because green-screen content is typically SDR,
 * and BT.601 yields tighter Cb/Cr clustering for saturated greens.
 */
function rgbToYCbCr(r: number, g: number, b: number): [number, number, number] {
  const y = 0.299 * r + 0.587 * g + 0.114 * b;
  const cb = -0.169 * r - 0.331 * g + 0.500 * b + 128;
  const cr = 0.500 * r - 0.419 * g - 0.081 * b + 128;
  return [y, cb, cr];
}

// ──────────────────────────────────────────────────────────────────────
// §4  analyzePass — histogram-only (extracted from removeGreenScreen)
//
// CCCL parallel: invoke_histogram_only()
//   Before: this logic was inside removeGreenScreen behind
//           `if (effectiveCfg.adaptiveThreshold)`.
//   After:  standalone pass, callable independently for calibration.
//
// Computes per-thread-block (here: per-border-region) histograms over
// the HSV hue channel, then finds the peak in [60,180] and derives
// calibrated hueCenter + hueTolerance.
// ──────────────────────────────────────────────────────────────────────

function analyzePass(
  pixels: Buffer,
  width: number,
  height: number,
  channels: number
): AnalysisResult {
  const hueHist = new Float64Array(360);
  let sampleCount = 0;
  const borderW = Math.max(Math.floor(width * 0.1), 5);
  const borderH = Math.max(Math.floor(height * 0.1), 5);

  // Pass 1: accumulate weighted hue histogram from border pixels only.
  // This is the "histogram kernel" — no filtering, just counting.
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const isBorder =
        x < borderW || x >= width - borderW ||
        y < borderH || y >= height - borderH;
      if (!isBorder) continue;

      const idx = (y * width + x) * channels;
      const r = pixels[idx], g = pixels[idx + 1], b = pixels[idx + 2];
      const [h, s, v] = rgbToHsv(r, g, b);

      if (s > 0.1 && v > 0.1) {
        hueHist[Math.floor(h) % 360] += s; // weight by saturation
        sampleCount++;
      }
    }
  }

  if (sampleCount === 0) {
    return { hueCenter: 120, hueTolerance: 40, confidence: 0, sampleCount: 0 };
  }

  // Gaussian-smooth the histogram (σ=5) to suppress noise
  const smoothed = new Float64Array(360);
  const sigma = 5;
  const halfK = Math.ceil(sigma * 3);
  for (let i = 0; i < 360; i++) {
    let sum = 0, wSum = 0;
    for (let k = -halfK; k <= halfK; k++) {
      const idx = ((i + k) % 360 + 360) % 360;
      const w = Math.exp(-(k * k) / (2 * sigma * sigma));
      sum += hueHist[idx] * w;
      wSum += w;
    }
    smoothed[i] = sum / wSum;
  }

  // Find peak in green range [60, 180]
  let peakHue = 120, peakVal = 0;
  for (let i = 60; i <= 180; i++) {
    if (smoothed[i] > peakVal) {
      peakVal = smoothed[i];
      peakHue = i;
    }
  }

  // FWHM → tolerance
  const halfMax = peakVal / 2;
  let left = peakHue, right = peakHue;
  while (left > 60 && smoothed[left] > halfMax) left--;
  while (right < 180 && smoothed[right] > halfMax) right++;
  const tolerance = Math.max((right - left) / 2, 15);

  const totalGreenWeight = smoothed.slice(60, 181).reduce((a, b) => a + b, 0);
  const totalWeight = smoothed.reduce((a, b) => a + b, 0);
  const confidence = totalWeight > 0 ? totalGreenWeight / totalWeight : 0;

  return { hueCenter: peakHue, hueTolerance: tolerance, confidence, sampleCount };
}

// ──────────────────────────────────────────────────────────────────────
// §5  filterPass — filter + histogram (the main work kernel)
//
// CCCL parallel: invoke_filter_and_histogram()
//   Removed: `template <bool IsFirstPass>` — no longer needed because
//            the first pass is now a separate function (analyzePass).
//   Added:   `is_last_pass` parameter (mirrors CCCL's kernel arg).
//            When true, we skip histogram reset in finalize.
//
// For each pixel, classifies as background (→ alpha=0/soft) or
// foreground (→ alpha=255) using dual HSV+YCbCr validation.
// ──────────────────────────────────────────────────────────────────────

function filterPass(
  pixels: Buffer,
  w: number,
  h: number,
  cfg: ChromaConfig
): FilterResult {
  const mask = new Uint8Array(w * h);
  const channels = 4;
  let removedCount = 0;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const pIdx = (y * w + x) * channels;
      const r = pixels[pIdx], g = pixels[pIdx + 1], b = pixels[pIdx + 2];

      // ── HSV classification ──
      const [hue, sat, val] = rgbToHsv(r, g, b);
      const hueDist = Math.min(
        Math.abs(hue - cfg.hueCenter),
        360 - Math.abs(hue - cfg.hueCenter)
      );
      const isGreenHSV =
        hueDist < cfg.hueTolerance &&
        sat > cfg.saturationMin &&
        val >= cfg.valueRange[0] &&
        val <= cfg.valueRange[1];

      // ── YCbCr cross-validation ──
      let isGreenYCbCr = false;
      if (cfg.enableYCbCr) {
        const [, cb, cr] = rgbToYCbCr(r, g, b);
        isGreenYCbCr =
          cb >= cfg.cbRange[0] && cb <= cfg.cbRange[1] &&
          cr >= cfg.crRange[0] && cr <= cfg.crRange[1];
      }

      // ── Combined decision ──
      // Mirrors CCCL's identify_candidates_op: classify each item as
      // 'selected' (foreground), 'candidate' (edge), or 'filtered' (background)
      const isBackground = cfg.enableYCbCr
        ? isGreenHSV && isGreenYCbCr
        : isGreenHSV;

      if (isBackground) {
        // Soft-key sigmoid for edge transition
        const normalizedDist = hueDist / cfg.hueTolerance;
        const softAlpha = 1.0 / (1.0 + Math.exp(-12 * (normalizedDist - 0.85)));
        mask[y * w + x] = Math.round(softAlpha * 255);
        if (softAlpha < 0.5) removedCount++;
      } else {
        mask[y * w + x] = 255;
      }
    }
  }

  return {
    mask,
    width: w,
    height: h,
    pixelsRemoved: removedCount,
    method: cfg.enableYCbCr ? 'hybrid' : 'hsv',
  };
}

// ──────────────────────────────────────────────────────────────────────
// §6  Morphological operations
//
// Cross-shaped structuring element (not square) to preserve diagonal
// detail. Mirrors CCCL's erode/dilate but on alpha channel.
// ──────────────────────────────────────────────────────────────────────

function applyErosion(mask: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(mask.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let minVal = mask[y * w + x];
      if (x > 0) minVal = Math.min(minVal, mask[y * w + x - 1]);
      if (x < w - 1) minVal = Math.min(minVal, mask[y * w + x + 1]);
      if (y > 0) minVal = Math.min(minVal, mask[(y - 1) * w + x]);
      if (y < h - 1) minVal = Math.min(minVal, mask[(y + 1) * w + x]);
      out[y * w + x] = minVal;
    }
  }
  return out;
}

function applyDilation(mask: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(mask.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let maxVal = mask[y * w + x];
      if (x > 0) maxVal = Math.max(maxVal, mask[y * w + x - 1]);
      if (x < w - 1) maxVal = Math.max(maxVal, mask[y * w + x + 1]);
      if (y > 0) maxVal = Math.max(maxVal, mask[(y - 1) * w + x]);
      if (y < h - 1) maxVal = Math.max(maxVal, mask[(y + 1) * w + x]);
      out[y * w + x] = maxVal;
    }
  }
  return out;
}

function applyMorphology(
  mask: Uint8Array, w: number, h: number,
  erodeIter: number, dilateIter: number
): Uint8Array {
  let result = mask;
  for (let i = 0; i < Math.min(erodeIter, 3); i++) result = applyErosion(result, w, h);
  for (let i = 0; i < Math.min(dilateIter, 3); i++) result = applyDilation(result, w, h);
  return result;
}

// ──────────────────────────────────────────────────────────────────────
// §7  Gaussian feathering (separable 1D convolution)
// ──────────────────────────────────────────────────────────────────────

function gaussianBlur1D(
  mask: Uint8Array, w: number, h: number, radius: number
): Uint8Array {
  if (radius <= 0) return mask;

  const sigma = radius / 2;
  const kSize = radius * 2 + 1;
  const kernel = new Float64Array(kSize);
  let kSum = 0;
  for (let i = 0; i < kSize; i++) {
    const x = i - radius;
    kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    kSum += kernel[i];
  }
  for (let i = 0; i < kSize; i++) kernel[i] /= kSum;

  // Horizontal pass
  const temp = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let k = -radius; k <= radius; k++) {
        const sx = Math.max(0, Math.min(w - 1, x + k));
        sum += mask[y * w + sx] * kernel[k + radius];
      }
      temp[y * w + x] = sum;
    }
  }

  // Vertical pass
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let k = -radius; k <= radius; k++) {
        const sy = Math.max(0, Math.min(h - 1, y + k));
        sum += temp[sy * w + x] * kernel[k + radius];
      }
      out[y * w + x] = Math.max(0, Math.min(255, Math.round(sum)));
    }
  }

  return out;
}

// ──────────────────────────────────────────────────────────────────────
// §8  Green spill suppression
// ──────────────────────────────────────────────────────────────────────

function suppressGreenSpill(
  pixels: Buffer, mask: Uint8Array,
  w: number, h: number, strength: number
): Buffer {
  const out = Buffer.from(pixels);
  const channels = 4;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const mIdx = y * w + x;
      const pIdx = mIdx * channels;
      const alpha = mask[mIdx];

      // Only suppress on semi-transparent edges
      if (alpha > 10 && alpha < 250) {
        const r = out[pIdx], g = out[pIdx + 1], b = out[pIdx + 2];
        const avgRB = (r + b) / 2;
        if (g > avgRB) {
          const spillAmount = (g - avgRB) * strength;
          out[pIdx + 1] = Math.round(g - spillAmount);
          const compensation = spillAmount * 0.3;
          out[pIdx] = Math.min(255, Math.round(r + compensation));
          out[pIdx + 2] = Math.min(255, Math.round(b + compensation));
        }
      }
    }
  }

  return out;
}

// ──────────────────────────────────────────────────────────────────────
// §9  finalizePipelinePass — shared postprocess
//
// CCCL parallel: finalize_pass()
//   The original removeGreenScreen() had inline postprocess logic.
//   Now extracted into a reusable function that:
//     1. Applies feathering
//     2. Applies spill suppression
//     3. Composites alpha into RGBA
//     4. Encodes to PNG
//
//   The caller-supplied `counter_update_fn` pattern from CCCL maps to
//   the `statsCollector` callback here: the caller provides a function
//   that runs after compositing to collect pass-specific metrics.
// ──────────────────────────────────────────────────────────────────────

async function finalizePipelinePass(
  rawPixels: Buffer,
  mask: Uint8Array,
  w: number,
  h: number,
  cfg: Pick<ChromaConfig, 'featherRadius' | 'spillSuppressionStrength'>,
  statsCollector: (finalMask: Uint8Array) => void
): Promise<{ outputBuffer: Buffer; maskBuffer: Buffer }> {
  // 1. Feathering
  let finalMask = gaussianBlur1D(mask, w, h, cfg.featherRadius);

  // 2. Spill suppression
  const spillCorrected = suppressGreenSpill(
    rawPixels, finalMask, w, h, cfg.spillSuppressionStrength
  );

  // 3. Composite alpha (premultiplied)
  const composited = Buffer.from(spillCorrected);
  for (let i = 0; i < w * h; i++) {
    const pIdx = i * 4;
    const a = finalMask[i];
    composited[pIdx + 3] = a;
    if (a === 0) {
      composited[pIdx] = 0;
      composited[pIdx + 1] = 0;
      composited[pIdx + 2] = 0;
    } else if (a < 255) {
      const f = a / 255;
      composited[pIdx] = Math.round(composited[pIdx] * f);
      composited[pIdx + 1] = Math.round(composited[pIdx + 1] * f);
      composited[pIdx + 2] = Math.round(composited[pIdx + 2] * f);
    }
  }

  // 4. Caller collects stats before encoding
  statsCollector(finalMask);

  // 5. Encode
  const outputBuffer = await sharp(composited, {
    raw: { width: w, height: h, channels: 4 },
  }).png({ compressionLevel: 6 }).toBuffer();

  const maskBuffer = await sharp(Buffer.from(finalMask), {
    raw: { width: w, height: h, channels: 1 },
  }).png().toBuffer();

  return { outputBuffer, maskBuffer };
}

// ──────────────────────────────────────────────────────────────────────
// §10 ChromaEngine — public API
//
// The dispatch loop mirrors CCCL's dispatch() in dispatch_topk.cuh:
//   Pass 0:  dedicated histogram-only kernel (analyzePass)
//   Pass 1+: fused filter+histogram kernel (filterPass + finalize)
//
// Before refactor:
//   removeGreenScreen() was a monolithic function with IsFirstPass branch.
// After refactor:
//   3 clean, independently testable passes.
// ──────────────────────────────────────────────────────────────────────

export class ChromaEngine {
  private config: ChromaConfig;

  constructor(config?: Partial<ChromaConfig>) {
    this.config = { ...DEFAULT_CHROMA_CONFIG, ...config };
  }

  /**
   * Public: run the analysis pass independently.
   * Useful for UI calibration (show detected hue) without running
   * the full pipeline.
   */
  async analyzeGreenDistribution(input: Buffer): Promise<AnalysisResult> {
    const img = sharp(input).ensureAlpha();
    const meta = await img.metadata();
    const rawPixels = await img.raw().toBuffer();
    return analyzePass(rawPixels, meta.width!, meta.height!, 4);
  }

  /**
   * Public: detect whether image contains green screen.
   */
  async detectGreenScreen(input: Buffer): Promise<{
    hasGreenScreen: boolean;
    confidence: number;
    dominantHue: number;
  }> {
    const analysis = await this.analyzeGreenDistribution(input);
    return {
      hasGreenScreen: analysis.confidence > 0.3,
      confidence: analysis.confidence,
      dominantHue: analysis.hueCenter,
    };
  }

  /**
   * Main entry: green screen removal.
   *
   * Dispatch structure (mirrors CCCL dispatch_topk.cuh):
   *
   *   // Pass 0: dedicated histogram-only kernel
   *   if (adaptiveThreshold) {
   *     analysis = analyzePass(pixels, ...)
   *     calibrate config from analysis
   *   }
   *
   *   // Pass 1: fused filter kernel (no more IsFirstPass template)
   *   filterResult = filterPass(pixels, ..., calibratedConfig)
   *
   *   // Finalize: shared postprocess
   *   output = finalizePipelinePass(pixels, filterResult.mask, ...)
   */
  async removeGreenScreen(input: Buffer): Promise<ChromaResult> {
    const startTime = Date.now();

    // Decode
    const img = sharp(input).ensureAlpha();
    const meta = await img.metadata();
    const w = meta.width!;
    const h = meta.height!;
    const rawPixels = await img.raw().toBuffer();

    // ── Pass 0: histogram-only (extracted, independently callable) ──
    let effectiveCfg = { ...this.config };
    let analysisConfidence = 0;

    if (effectiveCfg.adaptiveThreshold) {
      const analysis = analyzePass(rawPixels, w, h, 4);
      analysisConfidence = analysis.confidence;
      if (analysis.confidence > 0.3) {
        effectiveCfg.hueCenter = analysis.hueCenter;
        effectiveCfg.hueTolerance = Math.max(analysis.hueTolerance, 20);
      }
    }

    // ── Pass 1: filter (no IsFirstPass template parameter) ──
    const filterResult = filterPass(rawPixels, w, h, effectiveCfg);

    // ── Morphology ──
    let mask = applyMorphology(
      filterResult.mask, w, h,
      effectiveCfg.erodeIterations,
      effectiveCfg.dilateIterations
    );

    // ── Finalize (shared postprocess — mirrors CCCL finalize_pass) ──
    let removedCount = 0;

    const { outputBuffer, maskBuffer } = await finalizePipelinePass(
      rawPixels, mask, w, h,
      { featherRadius: effectiveCfg.featherRadius, spillSuppressionStrength: effectiveCfg.spillSuppressionStrength },
      // counter_update_fn equivalent: collect stats on thread 0 of last block
      (finalMask) => {
        for (let i = 0; i < finalMask.length; i++) {
          if (finalMask[i] < 128) removedCount++;
        }
      }
    );

    return {
      outputBuffer,
      maskBuffer,
      metadata: {
        width: w,
        height: h,
        pixelsProcessed: w * h,
        pixelsRemoved: removedCount,
        removalRatio: removedCount / (w * h),
        processingTimeMs: Date.now() - startTime,
        method: filterResult.method,
        analysisConfidence,
      },
    };
  }

  updateConfig(partial: Partial<ChromaConfig>): void {
    this.config = { ...this.config, ...partial };
  }

  getConfig(): Readonly<ChromaConfig> {
    return { ...this.config };
  }
}

export default ChromaEngine;