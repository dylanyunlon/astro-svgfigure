// src/lib/sph/color-palette.ts
// M566: Dynamic color palette per QoS profile
//
// Colour logic is a TypeScript port of three lygia GLSL modules:
//   • upstream/lygia/color/blend.glsl   → blendScreen / blendOverlay
//   • upstream/lygia/color/hueShift.glsl → hueShift (rgb↔hsl round-trip)
//   • upstream/lygia/color/dither.glsl  → Vlachos / IGN dither (anti-aliasing)
//
// Each QoS profile owns a base colour theme; per-particle tint is derived
// from its SPH density and velocity magnitude via the ported blend/hueShift
// math, and a scalar dither offset is added before quantising to 8-bit to
// suppress banding artefacts on the canvas renderer.




// ─────────────────────────────────────────────────────────────────────────────
// Colour type (linear, 0-1 range)
// ─────────────────────────────────────────────────────────────────────────────




import type { QoSProfileName } from './qosSpatial';

export interface RGB {
  r: number;
  g: number;
  b: number;
}

export interface RGBA extends RGB {
  a: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// QoS → base-colour theme
// SENSOR_DATA  — 冷蓝  (cool azure, high-frequency sensor bursts)
// PARAMETERS   — 暖橙  (warm amber, slow reliable param streams)
// TF_STATIC    — 翡翠绿 (jade green, persistent static transforms)
// TOPO_CHANGE  — 品红  (magenta, disruptive topology events)
// DEFAULT      — neutral slate
// ─────────────────────────────────────────────────────────────────────────────

export interface ThemePalette {
  /** Base particle colour (rest density, zero velocity). */
  base: RGB;
  /** Highlight mixed in at high velocity (blendScreen). */
  highlight: RGB;
  /** Shadow mixed in at high density (blendOverlay). */
  shadow: RGB;
  /** Maximum hue-shift angle (radians) applied as density → velocity rises. */
  maxHueShift: number;
}

export const QOS_THEME: Record<QoSProfileName, ThemePalette> = {
  // 冷蓝 — sensor streams: cool, fast, turbulent
  SENSOR_DATA: {
    base:        { r: 0.10, g: 0.45, b: 0.90 },  // #1A73E6  cool blue
    highlight:   { r: 0.55, g: 0.85, b: 1.00 },  // #8CD9FF  icy highlight
    shadow:      { r: 0.02, g: 0.10, b: 0.40 },  // #051A66  deep navy shadow
    maxHueShift: Math.PI * 0.18,                   // subtle cyan drift
  },

  // 暖橙 — parameters: stable, persistent, amber glow
  PARAMETERS: {
    base:        { r: 0.95, g: 0.55, b: 0.10 },  // #F28C1A  warm orange
    highlight:   { r: 1.00, g: 0.88, b: 0.50 },  // #FFE080  golden highlight
    shadow:      { r: 0.45, g: 0.15, b: 0.02 },  // #732604  dark sienna shadow
    maxHueShift: Math.PI * 0.10,                   // gentle red-orange drift
  },

  // 翡翠绿 — TF_STATIC: grounded, crystalline, still
  TF_STATIC: {
    base:        { r: 0.07, g: 0.72, b: 0.42 },  // #12B86B  jade green
    highlight:   { r: 0.60, g: 1.00, b: 0.75 },  // #99FFBF  mint highlight
    shadow:      { r: 0.02, g: 0.28, b: 0.18 },  // #05472E  forest shadow
    maxHueShift: Math.PI * 0.12,                   // teal-to-emerald drift
  },

  // 品红 — TOPO_CHANGE: disruptive, energetic, vivid
  TOPO_CHANGE: {
    base:        { r: 0.88, g: 0.10, b: 0.65 },  // #E01AA6  magenta
    highlight:   { r: 1.00, g: 0.60, b: 0.92 },  // #FF99EB  pink highlight
    shadow:      { r: 0.40, g: 0.02, b: 0.30 },  // #66054D  deep plum shadow
    maxHueShift: Math.PI * 0.22,                   // bold hue sweep
  },

  // Neutral default
  DEFAULT: {
    base:        { r: 0.55, g: 0.60, b: 0.68 },  // #8C99AD  slate
    highlight:   { r: 0.85, g: 0.88, b: 0.95 },  // #D9E0F2  pale lavender
    shadow:      { r: 0.18, g: 0.20, b: 0.28 },  // #2E3347  dark slate shadow
    maxHueShift: Math.PI * 0.08,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Lygia port — HSL ↔ RGB  (space/rgb2hsl.glsl + space/hsl2rgb.glsl)
// ─────────────────────────────────────────────────────────────────────────────

function hue2rgb(p: number, q: number, t: number): number {
  let tt = t;
  if (tt < 0) tt += 1;
  if (tt > 1) tt -= 1;
  if (tt < 1 / 6) return p + (q - p) * 6 * tt;
  if (tt < 1 / 2) return q;
  if (tt < 2 / 3) return p + (q - p) * (2 / 3 - tt) * 6;
  return p;
}

function rgb2hsl(c: RGB): [number, number, number] {
  const max = Math.max(c.r, c.g, c.b);
  const min = Math.min(c.r, c.g, c.b);
  const l   = (max + min) / 2;
  if (max === min) return [0, 0, l];

  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h: number;
  if (max === c.r)      h = (c.g - c.b) / d + (c.g < c.b ? 6 : 0);
  else if (max === c.g) h = (c.b - c.r) / d + 2;
  else                  h = (c.r - c.g) / d + 4;
  return [h / 6, s, l];
}

function hsl2rgb(h: number, s: number, l: number): RGB {
  if (s === 0) return { r: l, g: l, b: l };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return {
    r: hue2rgb(p, q, h + 1 / 3),
    g: hue2rgb(p, q, h),
    b: hue2rgb(p, q, h - 1 / 3),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Lygia port — hueShift.glsl
// angle is in radians (same as the GLSL version using TAU normalisation)
// ─────────────────────────────────────────────────────────────────────────────

function hueShift(color: RGB, angle: number): RGB {
  const TAU = Math.PI * 2;
  const [h, s, l] = rgb2hsl(color);
  // mirrors: hsl.r = fract((hsl.r * TAU + a) / TAU)
  const newH = ((h * TAU + angle) / TAU) % 1;
  return hsl2rgb(newH < 0 ? newH + 1 : newH, s, l);
}

// ─────────────────────────────────────────────────────────────────────────────
// Lygia port — blend/screen.glsl   blendScreen(base, blend)
// formula:  1 - (1 - base) * (1 - blend)
// ─────────────────────────────────────────────────────────────────────────────

function blendScreenChannel(base: number, blend: number): number {
  return 1 - (1 - base) * (1 - blend);
}

function blendScreen(base: RGB, blend: RGB, opacity: number): RGB {
  return {
    r: blendScreenChannel(base.r, blend.r) * opacity + base.r * (1 - opacity),
    g: blendScreenChannel(base.g, blend.g) * opacity + base.g * (1 - opacity),
    b: blendScreenChannel(base.b, blend.b) * opacity + base.b * (1 - opacity),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Lygia port — blend/overlay.glsl  blendOverlay(base, blend)
// formula:  base < 0.5 ? 2*base*blend : 1 - 2*(1-base)*(1-blend)
// ─────────────────────────────────────────────────────────────────────────────

function blendOverlayChannel(base: number, blend: number): number {
  return base < 0.5
    ? 2 * base * blend
    : 1 - 2 * (1 - base) * (1 - blend);
}

function blendOverlay(base: RGB, blend: RGB, opacity: number): RGB {
  return {
    r: blendOverlayChannel(base.r, blend.r) * opacity + base.r * (1 - opacity),
    g: blendOverlayChannel(base.g, blend.g) * opacity + base.g * (1 - opacity),
    b: blendOverlayChannel(base.b, blend.b) * opacity + base.b * (1 - opacity),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Lygia port — dither/interleavedGradientNoise.glsl  (IGN)
// Jimenez 2014, used on mobile targets.  Returns scalar offset in [-0.5, 0.5].
// ─────────────────────────────────────────────────────────────────────────────

function ditherIGN(x: number, y: number): number {
  // IGN: magic coefficients from Jorge Jimenez "Interleaved Gradient Noise"
  const noise =
    (0.06711056 * x + 0.00583715 * y) % 1;
  // map to [-0.5, 0.5] so dithering is unbiased
  return noise - 0.5;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lygia port — dither/vlachos.glsl  (desktop default)
// Vlachos 2016 "Advanced VR Rendering".  Uses a 4×4 Bayer-like magic matrix.
// ─────────────────────────────────────────────────────────────────────────────

const VLACHOS_MAGIC: ReadonlyArray<number> = [
   0, 8, 2,10,
  12, 4,14, 6,
   3,11, 1, 9,
  15, 7,13, 5,
];

function ditherVlachos(x: number, y: number): number {
  const idx = ((y & 3) << 2) | (x & 3);
  // normalise to [-0.5, 0.5]
  return VLACHOS_MAGIC[idx] / 16 - 0.5;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API — particle colour resolver
// ─────────────────────────────────────────────────────────────────────────────

export interface ParticleColorInput {
  /** QoS profile name that spawned this particle. */
  profile: QoSProfileName;
  /** Normalised SPH density: 0 = vacuum, 1 = rest density, >1 = compressed. */
  densityNorm: number;
  /** Normalised velocity magnitude: 0 = still, 1 = max expected speed. */
  velocityNorm: number;
  /**
   * Pixel-grid position used for dither.  Pass the canvas pixel coordinate
   * (integer) so neighbouring particles dither with spatial coherence.
   */
  pixelX: number;
  pixelY: number;
}

/**
 * Resolves the final RGBA colour for a single SPH particle.
 *
 * Pipeline (mirrors the GLSL passes in a fragment shader):
 *  1. Start from `theme.base`.
 *  2. Screen-blend `theme.highlight` weighted by `velocityNorm`      (bright, additive)
 *  3. Overlay-blend `theme.shadow`   weighted by `densityNorm`        (depth, contrast)
 *  4. hueShift by `densityNorm × velocityNorm × maxHueShift`          (energy colour shift)
 *  5. Add IGN/Vlachos dither offset (1/255 ≈ 0.004) before clamping  (anti-banding)
 *  6. Alpha: high density → more opaque; low density → translucent
 */
export function resolveParticleColor(input: ParticleColorInput): RGBA {
  const theme = QOS_THEME[input.profile] ?? QOS_THEME.DEFAULT;

  const vn = Math.min(Math.max(input.velocityNorm, 0), 1);
  const dn = Math.min(Math.max(input.densityNorm,  0), 2);   // allow slight over-pressure

  // Step 1+2: screen-blend highlight by velocity (faster → brighter)
  let color = blendScreen(theme.base, theme.highlight, vn * 0.75);

  // Step 3: overlay-blend shadow by density (denser → deeper tone)
  const densityT = Math.min(dn / 2, 1);                       // remap 0-2 → 0-1
  color = blendOverlay(color, theme.shadow, densityT * 0.60);

  // Step 4: hueShift — energy grows with density×velocity product
  const shiftAngle = dn * vn * theme.maxHueShift;
  color = hueShift(color, shiftAngle);

  // Step 5: dither — 1 LSB (1/255) amplitude, use Vlachos for quality
  const ditherAmp = 1 / 255;
  const dv = ditherVlachos(input.pixelX, input.pixelY) * ditherAmp;

  const clamp01 = (v: number) => Math.min(Math.max(v, 0), 1);

  // Step 6: alpha — denser particles are more opaque (0.3 floor, 1.0 ceiling)
  const alpha = 0.30 + 0.70 * Math.min(dn, 1);

  return {
    r: clamp01(color.r + dv),
    g: clamp01(color.g + dv),
    b: clamp01(color.b + dv),
    a: clamp01(alpha),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: convert RGBA (0-1) → CSS rgba() string for Canvas2D
// ─────────────────────────────────────────────────────────────────────────────

export function rgbaToCss(c: RGBA): string {
  const r = Math.round(c.r * 255);
  const g = Math.round(c.g * 255);
  const b = Math.round(c.b * 255);
  return `rgba(${r},${g},${b},${c.a.toFixed(3)})`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: pack RGBA into a Uint8 [r, g, b, a] for GPU texture writes
// ─────────────────────────────────────────────────────────────────────────────

export function rgbaToU8(c: RGBA): [number, number, number, number] {
  return [
    Math.round(c.r * 255),
    Math.round(c.g * 255),
    Math.round(c.b * 255),
    Math.round(c.a * 255),
  ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch helper: fill a pre-allocated Float32Array with per-particle colour data
// suitable for a WebGPU storage buffer (r,g,b,a interleaved, stride 4).
//
//   colorBuf  — Float32Array of length count × 4
//   profiles  — QoSProfileName[] per particle (length = count)
//   densities — Float32Array normalised densities
//   velocities — Float32Array normalised velocity magnitudes
//   pixelsX/Y  — Int32Array pixel coordinates (for dither coherence)
// ─────────────────────────────────────────────────────────────────────────────

export function batchResolveColors(
  colorBuf:   Float32Array,
  profiles:   QoSProfileName[],
  densities:  Float32Array,
  velocities: Float32Array,
  pixelsX:    Int32Array,
  pixelsY:    Int32Array,
  count:      number,
): void {
  for (let i = 0; i < count; i++) {
    const c = resolveParticleColor({
      profile:      profiles[i],
      densityNorm:  densities[i],
      velocityNorm: velocities[i],
      pixelX:       pixelsX[i],
      pixelY:       pixelsY[i],
    });
    const base = i * 4;
    colorBuf[base]     = c.r;
    colorBuf[base + 1] = c.g;
    colorBuf[base + 2] = c.b;
    colorBuf[base + 3] = c.a;
  }
}
