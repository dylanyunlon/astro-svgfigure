// ═══════════════════════════════════════════════════════════════════════════════
// M015: Cell Color Palette — PixiJS Color for species palette management
//
// Mirrors upstream/pixijs-engine/src/color/Color.ts.
// Maps SPECIES_METADATA colours to PixiJS Color objects with:
//   - HSL manipulation for hover/select/disabled states
//   - Gradient stops for cell body fills
//   - Species-aware colour interpolation for epoch transitions
//
// ═══════════════════════════════════════════════════════════════════════════════

import { Color } from 'pixi.js';

// ── Species palette from SPECIES_METADATA ───────────────────────────────────

export interface SpeciesColorSet {
  fill: Color;
  stroke: Color;
  glow: Color;
  bg: Color;
  text: Color;
}

/** Colour definitions matching channels/rendering/species/species_port.py */
const SPECIES_HEX: Record<string, { color: string; bg_color: string }> = {
  'cil-eye':         { color: '#3F51B5', bg_color: '#E8EAF6' },
  'cil-vector':      { color: '#2E7D32', bg_color: '#E8F5E9' },
  'cil-bolt':        { color: '#E65100', bg_color: '#FFF3E0' },
  'cil-plus':        { color: '#C62828', bg_color: '#FCE4EC' },
  'cil-arrow-right': { color: '#455A64', bg_color: '#ECEFF1' },
  'cil-filter':      { color: '#7B1FA2', bg_color: '#F3E5F5' },
  'cil-code':        { color: '#2E7D32', bg_color: '#E8F5E9' },
  'cil-layers':      { color: '#1565C0', bg_color: '#E3F2FD' },
  'cil-loop':        { color: '#F57F17', bg_color: '#FFF8E1' },
  'cil-graph':       { color: '#00695C', bg_color: '#E0F2F1' },
};

const _cache = new Map<string, SpeciesColorSet>();

/**
 * Get the PixiJS Color set for a species.
 * Cached after first call per species.
 */
export function getSpeciesColors(species: string): SpeciesColorSet {
  const cached = _cache.get(species);
  if (cached) return cached;

  const hex = SPECIES_HEX[species] ?? { color: '#666666', bg_color: '#F5F5F5' };
  const fill = new Color(hex.color);
  const bg = new Color(hex.bg_color);

  // Derive glow from fill with increased lightness
  const [h, s, l] = rgbToHsl(fill.red, fill.green, fill.blue);
  const glowRgb = hslToRgb(h, Math.min(1, s * 1.2), Math.min(1, l + 0.15));
  const glow = new Color({ r: glowRgb[0] * 255, g: glowRgb[1] * 255, b: glowRgb[2] * 255 });

  // Text colour: white on dark fill, dark on light fill
  const textColor = l < 0.5 ? new Color('#FFFFFF') : new Color('#212121');

  const set: SpeciesColorSet = {
    fill,
    stroke: new Color(hex.color),
    glow,
    bg,
    text: textColor,
  };

  _cache.set(species, set);
  return set;
}

// ── State variants ──────────────────────────────────────────────────────────

/** Lighten a Color for hover state */
export function hoverColor(base: Color, amount = 0.12): Color {
  const [h, s, l] = rgbToHsl(base.red, base.green, base.blue);
  const rgb = hslToRgb(h, s, Math.min(1, l + amount));
  return new Color({ r: rgb[0] * 255, g: rgb[1] * 255, b: rgb[2] * 255 });
}

/** Desaturate a Color for disabled state */
export function disabledColor(base: Color, amount = 0.6): Color {
  const [h, s, l] = rgbToHsl(base.red, base.green, base.blue);
  const rgb = hslToRgb(h, s * (1 - amount), l);
  return new Color({ r: rgb[0] * 255, g: rgb[1] * 255, b: rgb[2] * 255 });
}

/** Interpolate between two Colors (for epoch transitions) */
export function lerpColor(a: Color, b: Color, t: number): Color {
  return new Color({
    r: (a.red + (b.red - a.red) * t) * 255,
    g: (a.green + (b.green - a.green) * t) * 255,
    b: (a.blue + (b.blue - a.blue) * t) * 255,
  });
}

// ── HSL helpers ─────────────────────────────────────────────────────────────

function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h = 0;
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
  else if (max === g) h = ((b - r) / d + 2) / 6;
  else h = ((r - g) / d + 4) / 6;
  return [h, s, l];
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  if (s === 0) return [l, l, l];
  const hue2rgb = (p: number, q: number, t: number) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [hue2rgb(p, q, h + 1 / 3), hue2rgb(p, q, h), hue2rgb(p, q, h - 1 / 3)];
}
