/**
 * lut-generator.ts — QoS-driven LUT (Look-Up Table) color grading
 *
 * Each QoS profile maps to a different color tone for the entire world.
 * AT uses pre-baked LUT textures; we generate them dynamically based on
 * the active QoS profile so the world color shifts when QoS changes.
 *
 * LUT format: 16x16x16 = 4096 colors, stored as 256x16 strip (16 slices × 16px each)
 */

export interface LUTProfile {
  name: string;
  // Color adjustments
  temperatureShift: number;   // -1 (cool/blue) to +1 (warm/amber)
  saturation: number;         // 0 = grayscale, 1 = normal, 2 = vivid
  contrast: number;           // 0.5 = flat, 1 = normal, 2 = punchy
  gamma: number;              // 0.5 = bright, 1 = normal, 2 = dark
  // Tint
  shadowTint: [number, number, number];   // RGB tint for dark areas
  highlightTint: [number, number, number]; // RGB tint for bright areas
}

// QoS → LUT mapping (AT has fixed LUT; ours is QoS-reactive)
export const QOS_LUT_PROFILES: Record<string, LUTProfile> = {
  DEFAULT: {
    name: 'Default', temperatureShift: 0, saturation: 1.0, contrast: 1.0, gamma: 1.0,
    shadowTint: [0.0, 0.0, 0.02], highlightTint: [0.02, 0.01, 0.0],
  },
  SENSOR_DATA: {
    name: 'Sensor (cool)', temperatureShift: -0.4, saturation: 0.8, contrast: 0.9, gamma: 1.1,
    shadowTint: [0.0, 0.02, 0.05], highlightTint: [0.0, 0.02, 0.03],
  },
  PARAMETERS: {
    name: 'Parameters (warm)', temperatureShift: 0.3, saturation: 1.1, contrast: 1.1, gamma: 0.95,
    shadowTint: [0.03, 0.01, 0.0], highlightTint: [0.04, 0.02, 0.0],
  },
  SERVICES_DEFAULT: {
    name: 'Services (high contrast)', temperatureShift: 0.0, saturation: 1.2, contrast: 1.4, gamma: 1.0,
    shadowTint: [0.0, 0.0, 0.0], highlightTint: [0.0, 0.0, 0.0],
  },
  TF_STATIC: {
    name: 'TF Static (desaturated)', temperatureShift: -0.1, saturation: 0.6, contrast: 0.85, gamma: 1.05,
    shadowTint: [0.01, 0.01, 0.02], highlightTint: [0.01, 0.01, 0.01],
  },
  TOPO_CHANGE: {
    name: 'Topo Change (vivid)', temperatureShift: 0.15, saturation: 1.4, contrast: 1.2, gamma: 0.9,
    shadowTint: [0.02, 0.0, 0.03], highlightTint: [0.03, 0.02, 0.0],
  },
};

// sRGB ↔ Linear conversion
function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
function linearToSrgb(c: number): number {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
}

// Temperature shift: positive = warm (boost red, reduce blue), negative = cool
function applyTemperature(r: number, g: number, b: number, shift: number): [number, number, number] {
  return [
    Math.min(1, Math.max(0, r + shift * 0.1)),
    g,
    Math.min(1, Math.max(0, b - shift * 0.1)),
  ];
}

// Apply all LUT adjustments to a single color
function gradeColor(r: number, g: number, b: number, profile: LUTProfile): [number, number, number] {
  // To linear
  let lr = srgbToLinear(r), lg = srgbToLinear(g), lb = srgbToLinear(b);

  // Temperature
  [lr, lg, lb] = applyTemperature(lr, lg, lb, profile.temperatureShift);

  // Saturation (luminance-preserving)
  const lum = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
  lr = lum + (lr - lum) * profile.saturation;
  lg = lum + (lg - lum) * profile.saturation;
  lb = lum + (lb - lum) * profile.saturation;

  // Contrast (pivot at 0.18 gray)
  const pivot = 0.18;
  lr = pivot + (lr - pivot) * profile.contrast;
  lg = pivot + (lg - pivot) * profile.contrast;
  lb = pivot + (lb - pivot) * profile.contrast;

  // Gamma
  const invGamma = 1.0 / profile.gamma;
  lr = Math.pow(Math.max(0, lr), invGamma);
  lg = Math.pow(Math.max(0, lg), invGamma);
  lb = Math.pow(Math.max(0, lb), invGamma);

  // Shadow/highlight tint (blend by luminance)
  const lumFinal = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
  const shadowW = 1.0 - lumFinal;
  const highW = lumFinal;
  lr += profile.shadowTint[0] * shadowW + profile.highlightTint[0] * highW;
  lg += profile.shadowTint[1] * shadowW + profile.highlightTint[1] * highW;
  lb += profile.shadowTint[2] * shadowW + profile.highlightTint[2] * highW;

  // Back to sRGB
  return [
    linearToSrgb(Math.min(1, Math.max(0, lr))),
    linearToSrgb(Math.min(1, Math.max(0, lg))),
    linearToSrgb(Math.min(1, Math.max(0, lb))),
  ];
}

/**
 * Generate a 16x16x16 LUT as a 256x16 Uint8Array (RGBA).
 * Ready to upload as a WebGL texture.
 */
export function generateLUT(profileName: string): Uint8Array {
  const profile = QOS_LUT_PROFILES[profileName] || QOS_LUT_PROFILES.DEFAULT;
  const size = 16;
  const data = new Uint8Array(size * size * size * 4); // 256x16 RGBA

  let idx = 0;
  for (let b = 0; b < size; b++) {
    for (let g = 0; g < size; g++) {
      for (let r = 0; r < size; r++) {
        const ri = r / (size - 1);
        const gi = g / (size - 1);
        const bi = b / (size - 1);
        const [ro, go, bo] = gradeColor(ri, gi, bi, profile);
        data[idx++] = Math.round(ro * 255);
        data[idx++] = Math.round(go * 255);
        data[idx++] = Math.round(bo * 255);
        data[idx++] = 255;
      }
    }
  }
  return data;
}

/** Crossfade between two LUT textures (for QoS transitions) */
export function blendLUTs(lutA: Uint8Array, lutB: Uint8Array, t: number): Uint8Array {
  const out = new Uint8Array(lutA.length);
  for (let i = 0; i < lutA.length; i++) {
    out[i] = Math.round(lutA[i] * (1 - t) + lutB[i] * t);
  }
  return out;
}

export const LUT_SIZE = 16;
export const LUT_TEXTURE_WIDTH = 256;  // 16 slices × 16px
export const LUT_TEXTURE_HEIGHT = 16;
