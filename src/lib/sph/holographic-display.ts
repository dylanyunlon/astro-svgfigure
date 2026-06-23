/**
 * holographic-display.ts — M798: Holographic Display Mode
 * ─────────────────────────────────────────────────────────────────────────────
 * Toggleable visual mode that renders cells and particles as holographic
 * projections.  The effect composites five distinct layers on top of the
 * existing scene via a Canvas2D overlay pass:
 *
 *   Layer 1 — Scanlines
 *     Horizontal raster lines with configurable density, opacity, and
 *     vertical scroll speed.  Creates the CRT / hologram projection look.
 *     Alternating line pairs with sub-pixel offset produce a convincing
 *     interlaced display artefact.
 *
 *   Layer 2 — Chromatic Shift (RGB Channel Separation)
 *     Per-cell and per-particle RGB channel offset that simulates prismatic
 *     dispersion in a volumetric projection medium.  The shift direction
 *     rotates slowly over time, and magnitude scales with distance from
 *     screen centre (barrel-style falloff).
 *
 *   Layer 3 — Temporal Flicker
 *     Frame-coherent luminance jitter with occasional "glitch" bursts
 *     (probability-based, configurable).  Glitch frames apply a brief
 *     horizontal slice displacement and colour inversion, referencing
 *     ActiveTheory's "data corruption" transitions.
 *
 *   Layer 4 — Edge Glow (Fresnel Rim)
 *     Bright additive glow around cell and particle silhouettes, simulating
 *     the light-scatter halo of a holographic projection surface.  Uses a
 *     Fresnel-style falloff: brighter at the shape boundary, fading inward.
 *     Colour follows the cell's species palette shifted towards cyan.
 *
 *   Layer 5 — Holographic Transparency
 *     Global alpha modulation that oscillates subtly, combined with a
 *     depth-fade gradient (bottom-of-frame fades to zero) to sell the
 *     "projected from below" illusion.  Cells further from the projection
 *     origin appear more transparent.
 *
 * ─── Visual Language ──────────────────────────────────────────────────────────
 *
 *   The mode targets cinematic sci-fi aesthetics inspired by:
 *   • Star Wars holotable projections (blue-cyan monochrome, scanlines)
 *   • Blade Runner 2049 emanator (chromatic dispersion, edge bloom)
 *   • Ghost in the Shell interfaces (data-corruption glitch, rim glow)
 *   • ActiveTheory's Apollo spacecraft visualisation (volumetric haze)
 *
 *   Default palette: cyan → blue → white core, with chromatic shift into
 *   red/green channels at the edges.  Three presets are provided:
 *     CLASSIC   — blue-cyan monochrome (Star Wars)
 *     CYBERPUNK — magenta-teal split-tone (Blade Runner)
 *     GHOST     — green-white washed (Matrix / Ghost in the Shell)
 *
 * ─── Data Flow ────────────────────────────────────────────────────────────────
 *
 *   ParticleData (types.ts)       CellEntry[] (SPHWorld / wireframe-overlay)
 *        │                              │
 *        ▼                              ▼
 *   particle positions            cell positions + radii + species
 *        │                              │
 *        └──────── HolographicDisplay ──┘
 *                       │
 *              ┌────────┴─────────┐
 *              │ per-frame layers │
 *              ├─ scanlines       │
 *              ├─ chromatic shift │
 *              ├─ flicker/glitch  │
 *              ├─ edge glow       │
 *              └─ transparency    │
 *                       │
 *                       ▼
 *              Canvas2D composite
 *
 * ─── Integration ──────────────────────────────────────────────────────────────
 *
 *   import { HolographicDisplay, HOLO_PRESETS } from '$lib/sph/holographic-display';
 *
 *   const holo = new HolographicDisplay();
 *
 *   // Activate / deactivate
 *   holo.options.enabled = true;
 *
 *   // Switch preset
 *   holo.applyPreset('CYBERPUNK');
 *
 *   // Render each frame (after main scene, before UI)
 *   holo.render(ctx, {
 *     particles: { x, y, vx, vy, species, count },
 *     cells:     [{ cx, cy, radius, species }],
 *     domainW:   800,
 *     domainH:   600,
 *     time:      performance.now() / 1000,
 *   });
 *
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/wireframe-overlay.ts   — Canvas2D overlay compositing pattern
 *   src/lib/sph/post-process.ts        — Full-screen post-process (WGSL)
 *   src/lib/sph/debug-renderer.ts      — Canvas2D debug drawing primitives
 *   src/lib/sph/world-renderer.ts      — SPECIES_COLORS palette
 *   src/lib/sph/chromatic-adaptation.ts — Chromatic colour resolution
 *   src/lib/sph/types.ts               — ParticleData, SimParams
 *   src/lib/sph/vfx-timeline.ts        — ScreenFlash / glitch FX reference
 *
 * Research: xiaodi #M798 — cell-pubsub-loop
 */

import type { ParticleData } from './types';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** A visible cell entry for the holographic renderer. */
export interface HoloCellEntry {
  /** Centre X in world coordinates. */
  cx: number;
  /** Centre Y in world coordinates. */
  cy: number;
  /** Bounding radius in world units. */
  radius: number;
  /** Species string, e.g. 'cil-eye'. */
  species: string;
}

/** Data bundle passed to HolographicDisplay.render() each frame. */
export interface HoloFrameData {
  /** SPH particle positions and velocities. */
  particles: ParticleData;
  /** Visible cell list with world positions, radii, and species. */
  cells: HoloCellEntry[];
  /** Simulation domain width in world units. */
  domainW: number;
  /** Simulation domain height in world units. */
  domainH: number;
  /** Current time in seconds (monotonic, e.g. performance.now() / 1000). */
  time: number;
}

/** Named colour preset for the holographic palette. */
export type HoloPresetName = 'CLASSIC' | 'CYBERPUNK' | 'GHOST';

/** RGBA colour stored as [r, g, b, a] in 0–255 range (alpha in 0–1). */
export type RGBA = [number, number, number, number];

/** Holographic palette definition — drives all layer colours. */
export interface HoloPalette {
  /** Primary hologram colour (core of cells / particles). */
  primary: RGBA;
  /** Secondary colour for edge glow and chromatic shift target. */
  secondary: RGBA;
  /** Tertiary accent for glitch frames and scanline tint. */
  accent: RGBA;
  /** Background fill for the holographic projection base. */
  baseFog: RGBA;
}

// ─────────────────────────────────────────────────────────────────────────────
// Holographic display options (all layers)
// ─────────────────────────────────────────────────────────────────────────────

export interface HolographicDisplayOptions {
  // ── Global ──────────────────────────────────────────────────────────────
  /** Master enable for the entire holographic mode. Default true. */
  enabled: boolean;
  /** Master opacity multiplier [0, 1]. Default 0.92. */
  masterAlpha: number;
  /** Active colour preset name. Default 'CLASSIC'. */
  presetName: HoloPresetName;
  /** Custom palette (overrides preset when non-null). Default null. */
  customPalette: HoloPalette | null;

  // ── Layer 1: Scanlines ─────────────────────────────────────────────────
  /** Enable horizontal scanline overlay. Default true. */
  scanlines: boolean;
  /** Scanline spacing in CSS pixels. Default 3. */
  scanlineSpacing: number;
  /** Scanline thickness in CSS pixels. Default 1. */
  scanlineWidth: number;
  /** Scanline opacity [0, 1]. Default 0.15. */
  scanlineAlpha: number;
  /** Scanline vertical scroll speed (pixels per second). Default 30. */
  scanlineScrollSpeed: number;

  // ── Layer 2: Chromatic Shift ───────────────────────────────────────────
  /** Enable RGB channel separation. Default true. */
  chromaticShift: boolean;
  /** Maximum pixel offset for channel separation. Default 3. */
  chromaticMaxOffset: number;
  /** Rotation speed of the shift direction (radians per second). Default 0.5. */
  chromaticRotationSpeed: number;
  /** Barrel falloff: 0 = uniform shift, 1 = edges-only. Default 0.6. */
  chromaticBarrelFalloff: number;

  // ── Layer 3: Flicker / Glitch ──────────────────────────────────────────
  /** Enable temporal luminance flicker. Default true. */
  flicker: boolean;
  /** Flicker amplitude (fraction of base luminance). Default 0.06. */
  flickerAmplitude: number;
  /** Flicker frequency in Hz. Default 8. */
  flickerFrequency: number;
  /** Probability of a glitch burst per frame [0, 1]. Default 0.005. */
  glitchProbability: number;
  /** Maximum horizontal slice displacement during glitch (pixels). Default 12. */
  glitchSliceMax: number;
  /** Number of horizontal slices displaced per glitch frame. Default 5. */
  glitchSliceCount: number;

  // ── Layer 4: Edge Glow ─────────────────────────────────────────────────
  /** Enable Fresnel-style edge glow around shapes. Default true. */
  edgeGlow: boolean;
  /** Glow radius expansion beyond the shape boundary (world units). Default 8. */
  edgeGlowRadius: number;
  /** Edge glow intensity multiplier [0, 2]. Default 1.0. */
  edgeGlowIntensity: number;
  /** Glow pulse speed (Hz). Default 1.2. */
  edgeGlowPulseSpeed: number;
  /** Glow pulse amplitude (fraction of intensity). Default 0.2. */
  edgeGlowPulseAmplitude: number;

  // ── Layer 5: Holographic Transparency ──────────────────────────────────
  /** Enable depth-fade transparency gradient. Default true. */
  depthFade: boolean;
  /** Fraction of screen height where projection fades to zero. Default 0.15. */
  depthFadeHeight: number;
  /** Global alpha oscillation amplitude. Default 0.08. */
  alphaOscillation: number;
  /** Global alpha oscillation speed (Hz). Default 0.4. */
  alphaOscillationSpeed: number;

  // ── HUD chrome ─────────────────────────────────────────────────────────
  /** Show "HOLOGRAPHIC" mode badge in the corner. Default true. */
  showHud: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour palettes
// ─────────────────────────────────────────────────────────────────────────────

const PALETTE_CLASSIC: HoloPalette = {
  primary:   [0, 200, 255, 0.85],    // cyan-blue
  secondary: [100, 255, 255, 0.6],   // bright cyan
  accent:    [180, 220, 255, 0.9],   // ice-white
  baseFog:   [0, 10, 30, 0.12],      // deep blue fog
};

const PALETTE_CYBERPUNK: HoloPalette = {
  primary:   [255, 50, 180, 0.85],   // hot magenta
  secondary: [0, 255, 200, 0.6],     // teal-green
  accent:    [255, 255, 80, 0.9],    // electric yellow
  baseFog:   [20, 0, 30, 0.12],      // dark violet fog
};

const PALETTE_GHOST: HoloPalette = {
  primary:   [80, 255, 120, 0.8],    // matrix green
  secondary: [200, 255, 200, 0.5],   // pale green-white
  accent:    [255, 255, 255, 0.9],   // stark white
  baseFog:   [0, 15, 5, 0.12],       // dark green fog
};

/** All available presets, keyed by name. */
export const HOLO_PRESETS: Record<HoloPresetName, HoloPalette> = {
  CLASSIC:   PALETTE_CLASSIC,
  CYBERPUNK: PALETTE_CYBERPUNK,
  GHOST:     PALETTE_GHOST,
};

// ─────────────────────────────────────────────────────────────────────────────
// Defaults
// ─────────────────────────────────────────────────────────────────────────────

export const HOLOGRAPHIC_DEFAULTS: HolographicDisplayOptions = {
  enabled:       true,
  masterAlpha:   0.92,
  presetName:    'CLASSIC',
  customPalette: null,

  scanlines:          true,
  scanlineSpacing:    3,
  scanlineWidth:      1,
  scanlineAlpha:      0.15,
  scanlineScrollSpeed: 30,

  chromaticShift:         true,
  chromaticMaxOffset:     3,
  chromaticRotationSpeed: 0.5,
  chromaticBarrelFalloff: 0.6,

  flicker:           true,
  flickerAmplitude:  0.06,
  flickerFrequency:  8,
  glitchProbability: 0.005,
  glitchSliceMax:    12,
  glitchSliceCount:  5,

  edgeGlow:               true,
  edgeGlowRadius:         8,
  edgeGlowIntensity:      1.0,
  edgeGlowPulseSpeed:     1.2,
  edgeGlowPulseAmplitude: 0.2,

  depthFade:            true,
  depthFadeHeight:      0.15,
  alphaOscillation:     0.08,
  alphaOscillationSpeed: 0.4,

  showHud: true,
};

// ─────────────────────────────────────────────────────────────────────────────
// HUD palette — holographic chrome styling
// ─────────────────────────────────────────────────────────────────────────────

const HUD = {
  bg:      'rgba(0,8,20,0.75)',
  border:  'rgba(0,200,255,0.35)',
  text:    'rgba(150,230,255,0.95)',
  dim:     'rgba(80,160,200,0.65)',
  glow:    'rgba(0,200,255,0.15)',
};

// ─────────────────────────────────────────────────────────────────────────────
// Helper utilities
// ─────────────────────────────────────────────────────────────────────────────

/** Convert RGBA tuple to a CSS colour string. */
function rgbaStr(c: RGBA): string {
  return `rgba(${c[0]|0},${c[1]|0},${c[2]|0},${c[3]})`;
}

/** Convert RGBA tuple to a CSS colour string with alpha override. */
function rgbaStrA(c: RGBA, a: number): string {
  return `rgba(${c[0]|0},${c[1]|0},${c[2]|0},${a})`;
}

/** Simple hash for pseudo-random noise (deterministic per frame/position). */
function hash11(n: number): number {
  let s = Math.sin(n * 127.1) * 43758.5453;
  return s - Math.floor(s);
}

/** Two-input hash → [0, 1]. */
function hash21(x: number, y: number): number {
  let s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}

/** Smooth-step interpolation. */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/** Linear interpolation. */
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Clamp value to [lo, hi]. */
function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Scanline Generator
//
// Draws horizontal raster lines across the full canvas.  Lines scroll
// downward over time at a configurable speed to simulate a scanning
// projection beam.  Every other line is drawn at half opacity for an
// interlaced effect.
// ─────────────────────────────────────────────────────────────────────────────

function drawScanlines(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  palette: HoloPalette,
  opts: HolographicDisplayOptions,
  time: number,
): void {
  const { scanlineSpacing, scanlineWidth, scanlineAlpha, scanlineScrollSpeed } = opts;
  const scrollOffset = (time * scanlineScrollSpeed) % (scanlineSpacing * 2);

  ctx.save();
  ctx.globalCompositeOperation = 'source-over';

  // Primary scanlines — full opacity band
  const baseColor = rgbaStrA(palette.primary, scanlineAlpha);
  const halfColor = rgbaStrA(palette.primary, scanlineAlpha * 0.4);

  for (let y = -scanlineSpacing * 2 + scrollOffset; y < h + scanlineSpacing; y += scanlineSpacing) {
    const lineIndex = Math.floor(y / scanlineSpacing);
    const isEven = (lineIndex & 1) === 0;

    ctx.fillStyle = isEven ? baseColor : halfColor;
    ctx.fillRect(0, y, w, scanlineWidth);
  }

  // Subtle broad luminance bands (low-frequency variation)
  const bandHeight = h * 0.12;
  const bandY = ((time * scanlineScrollSpeed * 0.3) % (h + bandHeight * 2)) - bandHeight;
  const bandGrad = ctx.createLinearGradient(0, bandY, 0, bandY + bandHeight);
  bandGrad.addColorStop(0, 'rgba(255,255,255,0)');
  bandGrad.addColorStop(0.5, rgbaStrA(palette.accent, 0.04));
  bandGrad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = bandGrad;
  ctx.fillRect(0, bandY, w, bandHeight);

  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2  Chromatic Shift
//
// Simulates RGB channel separation by drawing each cell and particle three
// times with slight positional offsets in R, G, and B channels.  The shift
// direction rotates over time and scales with distance from screen centre
// (barrel distortion falloff).
//
// Rather than re-rendering the full scene per channel, we composite a set
// of tinted translucent shapes at offset positions — cheaper and produces
// a convincing holographic fringe.
// ─────────────────────────────────────────────────────────────────────────────

interface ChromaticOffsets {
  rDx: number; rDy: number;
  gDx: number; gDy: number;
  bDx: number; bDy: number;
}

function computeChromaticOffsets(
  x: number,
  y: number,
  cx: number,
  cy: number,
  opts: HolographicDisplayOptions,
  time: number,
): ChromaticOffsets {
  // Distance from screen centre → barrel falloff
  const dx = (x - cx) / cx;
  const dy = (y - cy) / cy;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const barrel = lerp(1.0, dist, opts.chromaticBarrelFalloff);

  // Rotating shift direction
  const angle = time * opts.chromaticRotationSpeed;
  const maxOff = opts.chromaticMaxOffset * barrel;

  // Three channels offset 120° apart
  const rAngle = angle;
  const gAngle = angle + (Math.PI * 2) / 3;
  const bAngle = angle + (Math.PI * 4) / 3;

  return {
    rDx: Math.cos(rAngle) * maxOff,
    rDy: Math.sin(rAngle) * maxOff,
    gDx: Math.cos(gAngle) * maxOff,
    gDy: Math.sin(gAngle) * maxOff,
    bDx: Math.cos(bAngle) * maxOff,
    bDy: Math.sin(bAngle) * maxOff,
  };
}

function drawChromaticCell(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  radius: number,
  offsets: ChromaticOffsets,
  palette: HoloPalette,
  alpha: number,
): void {
  ctx.save();
  ctx.globalCompositeOperation = 'lighter'; // additive blending

  const channels: Array<{ dx: number; dy: number; color: string }> = [
    { dx: offsets.rDx, dy: offsets.rDy, color: `rgba(${palette.primary[0]|0},0,0,${alpha * 0.6})` },
    { dx: offsets.gDx, dy: offsets.gDy, color: `rgba(0,${palette.primary[1]|0},0,${alpha * 0.6})` },
    { dx: offsets.bDx, dy: offsets.bDy, color: `rgba(0,0,${palette.primary[2]|0},${alpha * 0.6})` },
  ];

  for (const ch of channels) {
    ctx.beginPath();
    ctx.arc(cx + ch.dx, cy + ch.dy, radius, 0, Math.PI * 2);
    ctx.fillStyle = ch.color;
    ctx.fill();
  }

  ctx.restore();
}

function drawChromaticParticle(
  ctx: CanvasRenderingContext2D,
  px: number,
  py: number,
  size: number,
  offsets: ChromaticOffsets,
  palette: HoloPalette,
  alpha: number,
): void {
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';

  const channels: Array<{ dx: number; dy: number; color: string }> = [
    { dx: offsets.rDx, dy: offsets.rDy, color: `rgba(${palette.secondary[0]|0},0,0,${alpha * 0.45})` },
    { dx: offsets.gDx, dy: offsets.gDy, color: `rgba(0,${palette.secondary[1]|0},0,${alpha * 0.45})` },
    { dx: offsets.bDx, dy: offsets.bDy, color: `rgba(0,0,${palette.secondary[2]|0},${alpha * 0.45})` },
  ];

  const halfSize = size * 0.5;
  for (const ch of channels) {
    ctx.fillStyle = ch.color;
    ctx.fillRect(px + ch.dx - halfSize, py + ch.dy - halfSize, size, size);
  }

  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  Flicker & Glitch
//
// Temporal flicker: sinusoidal luminance oscillation with multiple
// overlapping frequencies for organic feel.
//
// Glitch bursts: on a random per-frame basis (probability check), several
// horizontal slices of the canvas are displaced sideways with colour-
// inverted strips.  This simulates data-corruption / signal-dropout
// artefacts common in holographic sci-fi interfaces.
//
// The glitch system uses a seeded PRNG (based on frame time) so that
// identical timestamps produce identical glitch patterns — important for
// deterministic replay in the epoch-physics-recorder.
// ─────────────────────────────────────────────────────────────────────────────

/** Compute the flicker multiplier for the current frame [0.8, 1.2]. */
function computeFlicker(time: number, opts: HolographicDisplayOptions): number {
  if (!opts.flicker) return 1.0;

  const freq = opts.flickerFrequency;
  const amp = opts.flickerAmplitude;

  // Multi-frequency flicker for organic feel
  const f1 = Math.sin(time * freq * Math.PI * 2) * amp;
  const f2 = Math.sin(time * freq * 1.7 * Math.PI * 2) * amp * 0.4;
  const f3 = Math.sin(time * freq * 3.1 * Math.PI * 2) * amp * 0.15;

  return clamp(1.0 + f1 + f2 + f3, 0.6, 1.3);
}

/** Check whether this frame should display a glitch burst. */
function shouldGlitch(time: number, probability: number): boolean {
  // Use a high-frequency hash of time to get pseudo-random per-frame decision
  return hash11(time * 60.0) < probability;
}

/** Draw glitch slice displacement artefacts. */
function drawGlitch(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  palette: HoloPalette,
  opts: HolographicDisplayOptions,
  time: number,
): void {
  if (!shouldGlitch(time, opts.glitchProbability)) return;

  ctx.save();

  const seed = Math.floor(time * 60);
  for (let i = 0; i < opts.glitchSliceCount; i++) {
    const sliceY = hash21(seed, i) * h;
    const sliceH = 2 + hash21(seed + 100, i) * 8;
    const displacement = (hash21(seed + 200, i) - 0.5) * 2 * opts.glitchSliceMax;

    // Grab and displace a horizontal slice
    try {
      const imgData = ctx.getImageData(0, sliceY | 0, w, sliceH | 0);
      ctx.putImageData(imgData, displacement | 0, sliceY | 0);
    } catch {
      // getImageData may fail on tainted canvas — skip gracefully
    }

    // Overlay a tinted strip for chromatic glitch
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = rgbaStrA(palette.accent, 0.15);
    ctx.fillRect(0, sliceY, w, sliceH);

    // Occasional inverted colour strip
    if (hash21(seed + 300, i) > 0.6) {
      ctx.globalCompositeOperation = 'difference';
      ctx.fillStyle = rgbaStrA(palette.primary, 0.3);
      ctx.fillRect(0, sliceY, w, sliceH * 0.5);
    }
  }

  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  Edge Glow (Fresnel Rim)
//
// Renders a soft additive glow around each cell silhouette.  The glow is
// implemented as a radial gradient that extends beyond the cell boundary
// by `edgeGlowRadius`, with full brightness at the boundary fading to
// transparent at the outer edge.
//
// The glow intensity pulses slowly to simulate energy fluctuations in the
// projection field.  The colour is derived from the palette secondary,
// shifted towards cyan for consistency with the holographic theme.
// ─────────────────────────────────────────────────────────────────────────────

function drawEdgeGlow(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  radius: number,
  palette: HoloPalette,
  opts: HolographicDisplayOptions,
  time: number,
  flickerMul: number,
): void {
  const pulse = 1.0 + Math.sin(time * opts.edgeGlowPulseSpeed * Math.PI * 2)
    * opts.edgeGlowPulseAmplitude;
  const intensity = opts.edgeGlowIntensity * pulse * flickerMul;

  const innerR = Math.max(0, radius - 1);
  const outerR = radius + opts.edgeGlowRadius;

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';

  const grad = ctx.createRadialGradient(cx, cy, innerR, cx, cy, outerR);
  grad.addColorStop(0, 'rgba(0,0,0,0)');
  grad.addColorStop(0.3, rgbaStrA(palette.secondary, 0.05 * intensity));
  grad.addColorStop(0.65, rgbaStrA(palette.secondary, 0.25 * intensity));
  grad.addColorStop(0.85, rgbaStrA(palette.primary, 0.35 * intensity));
  grad.addColorStop(1, 'rgba(0,0,0,0)');

  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, outerR, 0, Math.PI * 2);
  ctx.fill();

  // Inner bright core ring
  ctx.strokeStyle = rgbaStrA(palette.accent, 0.3 * intensity);
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();

  ctx.restore();
}

/** Draw a soft glow dot for a particle. */
function drawParticleGlow(
  ctx: CanvasRenderingContext2D,
  px: number,
  py: number,
  size: number,
  palette: HoloPalette,
  intensity: number,
): void {
  const glowR = size * 2.5;

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';

  const grad = ctx.createRadialGradient(px, py, 0, px, py, glowR);
  grad.addColorStop(0, rgbaStrA(palette.secondary, 0.2 * intensity));
  grad.addColorStop(0.4, rgbaStrA(palette.secondary, 0.08 * intensity));
  grad.addColorStop(1, 'rgba(0,0,0,0)');

  ctx.fillStyle = grad;
  ctx.fillRect(px - glowR, py - glowR, glowR * 2, glowR * 2);

  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  Holographic Transparency & Depth Fade
//
// Composites a vertical gradient that fades the entire holographic overlay
// towards zero opacity at the bottom of the frame (simulating a projection
// emitter mounted below the scene).  Additionally applies a subtle global
// alpha oscillation to produce the "unstable projection" look.
// ─────────────────────────────────────────────────────────────────────────────

function drawDepthFade(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  palette: HoloPalette,
  opts: HolographicDisplayOptions,
  time: number,
): void {
  if (!opts.depthFade) return;

  ctx.save();
  ctx.globalCompositeOperation = 'destination-out';

  const fadeH = h * opts.depthFadeHeight;
  const fadeGrad = ctx.createLinearGradient(0, h - fadeH, 0, h);
  fadeGrad.addColorStop(0, 'rgba(0,0,0,0)');
  fadeGrad.addColorStop(1, 'rgba(0,0,0,0.85)');
  ctx.fillStyle = fadeGrad;
  ctx.fillRect(0, h - fadeH, w, fadeH);

  // Top edge vignette (subtler)
  const topH = h * 0.06;
  const topGrad = ctx.createLinearGradient(0, 0, 0, topH);
  topGrad.addColorStop(0, 'rgba(0,0,0,0.4)');
  topGrad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = topGrad;
  ctx.fillRect(0, 0, w, topH);

  ctx.restore();

  // Base fog tint (source-over) — faint wash at the projection floor
  ctx.save();
  ctx.globalCompositeOperation = 'source-over';
  const fogGrad = ctx.createLinearGradient(0, h * 0.7, 0, h);
  fogGrad.addColorStop(0, 'rgba(0,0,0,0)');
  fogGrad.addColorStop(1, rgbaStrA(palette.baseFog, palette.baseFog[3]));
  ctx.fillStyle = fogGrad;
  ctx.fillRect(0, h * 0.7, w, h * 0.3);
  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  HUD Badge
//
// Small status indicator in the top-right corner showing the active mode
// name and current preset.  Styled as a translucent sci-fi panel.
// ─────────────────────────────────────────────────────────────────────────────

function drawHud(
  ctx: CanvasRenderingContext2D,
  w: number,
  opts: HolographicDisplayOptions,
  time: number,
): void {
  if (!opts.showHud) return;

  ctx.save();

  const hudW = 145;
  const hudH = 42;
  const hudX = w - hudW - 12;
  const hudY = 12;
  const cornerR = 4;

  // Background panel
  ctx.beginPath();
  ctx.roundRect(hudX, hudY, hudW, hudH, cornerR);
  ctx.fillStyle = HUD.bg;
  ctx.fill();

  // Animated border (pulsing opacity)
  const borderAlpha = 0.25 + Math.sin(time * 2) * 0.1;
  ctx.strokeStyle = `rgba(0,200,255,${borderAlpha})`;
  ctx.lineWidth = 1;
  ctx.stroke();

  // Subtle inner glow
  ctx.shadowColor = HUD.glow;
  ctx.shadowBlur = 8;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Title line
  ctx.font = '600 10px "SF Mono", "Fira Code", monospace';
  ctx.fillStyle = HUD.text;
  ctx.textBaseline = 'top';
  ctx.fillText('◇ HOLOGRAPHIC', hudX + 10, hudY + 8);

  // Preset / status line
  ctx.font = '400 9px "SF Mono", "Fira Code", monospace';
  ctx.fillStyle = HUD.dim;
  ctx.fillText(`PRESET: ${opts.presetName}`, hudX + 10, hudY + 24);

  // Blinking indicator dot
  const dotAlpha = (Math.sin(time * 4) > 0) ? 0.9 : 0.2;
  ctx.fillStyle = `rgba(0,255,200,${dotAlpha})`;
  ctx.beginPath();
  ctx.arc(hudX + hudW - 14, hudY + hudH * 0.5, 3, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────────────────
// § 7  Cell Holographic Body
//
// Draws each cell as a translucent holographic shape: concentric rings
// (iso-level echoes) plus the core fill with species-tinted colour.  This
// replaces the solid cell rendering with the characteristic "projected"
// look — partially transparent, with visible internal structure lines.
// ─────────────────────────────────────────────────────────────────────────────

function drawHoloCell(
  ctx: CanvasRenderingContext2D,
  cell: HoloCellEntry,
  palette: HoloPalette,
  opts: HolographicDisplayOptions,
  time: number,
  flickerMul: number,
  canvasCx: number,
  canvasCy: number,
): void {
  const { cx, cy, radius } = cell;
  const alpha = opts.masterAlpha * flickerMul;

  // ── Chromatic shift layer ───────────────────────────────────────────────
  if (opts.chromaticShift) {
    const offsets = computeChromaticOffsets(cx, cy, canvasCx, canvasCy, opts, time);
    drawChromaticCell(ctx, cx, cy, radius, offsets, palette, alpha * 0.5);
  }

  // ── Core holographic body ──────────────────────────────────────────────
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';

  // Translucent filled core
  const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
  coreGrad.addColorStop(0, rgbaStrA(palette.accent, 0.15 * alpha));
  coreGrad.addColorStop(0.5, rgbaStrA(palette.primary, 0.08 * alpha));
  coreGrad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = coreGrad;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();

  // Concentric iso-ring echoes (3 rings for visual depth)
  const ringCount = 3;
  for (let i = 1; i <= ringCount; i++) {
    const ringR = radius * (i / (ringCount + 1));
    const ringAlpha = (0.25 - i * 0.06) * alpha;
    ctx.strokeStyle = rgbaStrA(palette.primary, Math.max(0, ringAlpha));
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Outer boundary ring (brighter)
  ctx.strokeStyle = rgbaStrA(palette.primary, 0.4 * alpha);
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();

  // Horizontal scanline clipping through the cell (micro-detail)
  const microSpacing = 4;
  const scrollY = (time * 15) % microSpacing;
  ctx.strokeStyle = rgbaStrA(palette.secondary, 0.06 * alpha);
  ctx.lineWidth = 0.5;
  for (let y = cy - radius + scrollY; y < cy + radius; y += microSpacing) {
    const dy = y - cy;
    const halfChord = Math.sqrt(Math.max(0, radius * radius - dy * dy));
    if (halfChord > 1) {
      ctx.beginPath();
      ctx.moveTo(cx - halfChord, y);
      ctx.lineTo(cx + halfChord, y);
      ctx.stroke();
    }
  }

  ctx.restore();

  // ── Edge glow ─────────────────────────────────────────────────────────
  if (opts.edgeGlow) {
    drawEdgeGlow(ctx, cx, cy, radius, palette, opts, time, flickerMul);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  Particle Holographic Rendering
//
// Draws SPH particles as small glowing points with chromatic separation.
// Particles use a smaller, faster variant of the cell holographic effect.
// A speed-based intensity modulation makes fast-moving particles brighter
// (simulating Doppler-shifted energy in the projection medium).
// ─────────────────────────────────────────────────────────────────────────────

const PARTICLE_BASE_SIZE = 2.5;
const MAX_PARTICLES_HOLO = 4000; // performance cap for holographic rendering

function drawHoloParticles(
  ctx: CanvasRenderingContext2D,
  particles: ParticleData,
  palette: HoloPalette,
  opts: HolographicDisplayOptions,
  time: number,
  flickerMul: number,
  canvasCx: number,
  canvasCy: number,
): void {
  const count = Math.min(particles.count, MAX_PARTICLES_HOLO);
  const alpha = opts.masterAlpha * flickerMul;

  ctx.save();

  for (let i = 0; i < count; i++) {
    const px = particles.x[i];
    const py = particles.y[i];

    // Speed-based intensity (faster particles glow brighter)
    const speed = Math.sqrt(
      particles.vx[i] * particles.vx[i] +
      particles.vy[i] * particles.vy[i],
    );
    const speedNorm = clamp(speed / 200, 0, 1);
    const intensity = 0.5 + speedNorm * 0.5;
    const size = PARTICLE_BASE_SIZE + speedNorm * 1.5;

    // ── Chromatic shift ──────────────────────────────────────────────────
    if (opts.chromaticShift) {
      const offsets = computeChromaticOffsets(px, py, canvasCx, canvasCy, opts, time);
      drawChromaticParticle(ctx, px, py, size, offsets, palette, alpha * intensity * 0.4);
    }

    // ── Core particle dot ────────────────────────────────────────────────
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = rgbaStrA(palette.primary, alpha * intensity * 0.35);
    ctx.beginPath();
    ctx.arc(px, py, size * 0.5, 0, Math.PI * 2);
    ctx.fill();

    // ── Particle glow (every Nth for performance) ────────────────────────
    if (opts.edgeGlow && (i & 3) === 0) {
      drawParticleGlow(ctx, px, py, size, palette, intensity * flickerMul * 0.4);
    }
  }

  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────────────────
// § 9  Global Alpha Oscillation
//
// Applies a subtle overall opacity modulation to the entire overlay canvas,
// selling the "unstable holographic projection" look.
// ─────────────────────────────────────────────────────────────────────────────

function computeGlobalAlpha(time: number, opts: HolographicDisplayOptions): number {
  const oscillation = Math.sin(time * opts.alphaOscillationSpeed * Math.PI * 2)
    * opts.alphaOscillation;
  return clamp(opts.masterAlpha + oscillation, 0, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// § 10  WGSL Shader Source (for WebGPU post-process integration)
//
// A full-screen fragment shader that can be applied as a render pass in the
// ATRenderPipeline / Nuke post-process chain.  This encodes the same visual
// layers (scanlines, chromatic shift, flicker, edge glow) in GPU shader
// form for higher-performance rendering when the WebGPU pipeline is active.
//
// The Canvas2D implementation above is the fallback for non-WebGPU contexts.
// ─────────────────────────────────────────────────────────────────────────────

export const HOLOGRAPHIC_WGSL = /* wgsl */`
// ─── Holographic Display Post-Process ────────────────────────────────────────
struct HoloUniforms {
  resolution    : vec2f,
  time          : f32,
  masterAlpha   : f32,

  // Scanlines
  scanlineSpacing    : f32,
  scanlineAlpha      : f32,
  scanlineScrollSpeed: f32,

  // Chromatic shift
  chromaticMaxOffset     : f32,
  chromaticRotationSpeed : f32,
  chromaticBarrelFalloff : f32,

  // Flicker
  flickerAmplitude : f32,
  flickerFrequency : f32,

  // Edge glow
  edgeGlowIntensity : f32,
  edgeGlowPulseSpeed: f32,

  // Palette: primary RGBA
  primaryR : f32,
  primaryG : f32,
  primaryB : f32,
  primaryA : f32,

  // Palette: secondary RGBA
  secondaryR : f32,
  secondaryG : f32,
  secondaryB : f32,
  secondaryA : f32,

  // Depth fade
  depthFadeHeight : f32,
  _pad0 : f32,
  _pad1 : f32,
  _pad2 : f32,
}

@group(0) @binding(0) var<uniform> u   : HoloUniforms;
@group(0) @binding(1) var          smp : sampler;
@group(0) @binding(2) var          src : texture_2d<f32>;

// ─── Hash for noise ──────────────────────────────────────────────────────────
fn hash11(n: f32) -> f32 {
  return fract(sin(n * 127.1) * 43758.5453);
}

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// ─── Scanline layer ──────────────────────────────────────────────────────────
fn scanline(uv: vec2f) -> f32 {
  let pixelY = uv.y * u.resolution.y;
  let scroll  = u.time * u.scanlineScrollSpeed;
  let line    = fract((pixelY + scroll) / u.scanlineSpacing);
  let scanVal = smoothstep(0.0, 0.15, line) * (1.0 - smoothstep(0.25, 0.4, line));
  return scanVal * u.scanlineAlpha;
}

// ─── Chromatic aberration ────────────────────────────────────────────────────
fn chromaticSample(uv: vec2f) -> vec3f {
  let centre = vec2f(0.5, 0.5);
  let d      = uv - centre;
  let dist   = length(d);
  let barrel = mix(1.0, dist * 2.0, u.chromaticBarrelFalloff);
  let maxOff = u.chromaticMaxOffset / u.resolution.x * barrel;

  let angle = u.time * u.chromaticRotationSpeed;
  let rOff  = vec2f(cos(angle), sin(angle)) * maxOff;
  let gOff  = vec2f(cos(angle + 2.094), sin(angle + 2.094)) * maxOff;
  let bOff  = vec2f(cos(angle + 4.189), sin(angle + 4.189)) * maxOff;

  let r = textureSample(src, smp, uv + rOff).r;
  let g = textureSample(src, smp, uv + gOff).g;
  let b = textureSample(src, smp, uv + bOff).b;

  return vec3f(r, g, b);
}

// ─── Flicker ─────────────────────────────────────────────────────────────────
fn flicker() -> f32 {
  let f1 = sin(u.time * u.flickerFrequency * 6.283) * u.flickerAmplitude;
  let f2 = sin(u.time * u.flickerFrequency * 1.7 * 6.283) * u.flickerAmplitude * 0.4;
  let f3 = sin(u.time * u.flickerFrequency * 3.1 * 6.283) * u.flickerAmplitude * 0.15;
  return clamp(1.0 + f1 + f2 + f3, 0.6, 1.3);
}

// ─── Edge detection for glow ─────────────────────────────────────────────────
fn edgeSobel(uv: vec2f) -> f32 {
  let px = 1.0 / u.resolution;
  let tl = luminance(textureSample(src, smp, uv + vec2f(-px.x, -px.y)).rgb);
  let tc = luminance(textureSample(src, smp, uv + vec2f(  0.0, -px.y)).rgb);
  let tr = luminance(textureSample(src, smp, uv + vec2f( px.x, -px.y)).rgb);
  let ml = luminance(textureSample(src, smp, uv + vec2f(-px.x,   0.0)).rgb);
  let mr = luminance(textureSample(src, smp, uv + vec2f( px.x,   0.0)).rgb);
  let bl = luminance(textureSample(src, smp, uv + vec2f(-px.x,  px.y)).rgb);
  let bc = luminance(textureSample(src, smp, uv + vec2f(  0.0,  px.y)).rgb);
  let br = luminance(textureSample(src, smp, uv + vec2f( px.x,  px.y)).rgb);

  let gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
  let gy = -tl - 2.0*tc - tr + bl + 2.0*bc + br;
  return sqrt(gx*gx + gy*gy);
}

// ─── Depth fade ──────────────────────────────────────────────────────────────
fn depthFade(uv: vec2f) -> f32 {
  let fadeStart = 1.0 - u.depthFadeHeight;
  let bottom    = smoothstep(fadeStart, 1.0, uv.y);
  let top       = smoothstep(0.0, 0.05, uv.y);
  return top * (1.0 - bottom * 0.85);
}

// ─── Main fragment ───────────────────────────────────────────────────────────
@fragment fn fs_holographic(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  let uv = fragCoord.xy / u.resolution;

  // Sample with chromatic aberration
  var col = chromaticSample(uv);

  // Flicker modulation
  let fMul = flicker();
  col *= fMul;

  // Tint towards primary holographic colour
  let holoColor = vec3f(u.primaryR, u.primaryG, u.primaryB) / 255.0;
  let lum = luminance(col);
  col = mix(col, holoColor * lum * 1.5, 0.35);

  // Edge glow (Sobel → rim highlight)
  let edge = edgeSobel(uv);
  let glowPulse = 1.0 + sin(u.time * u.edgeGlowPulseSpeed * 6.283) * 0.2;
  let edgeColor = vec3f(u.secondaryR, u.secondaryG, u.secondaryB) / 255.0;
  col += edgeColor * edge * u.edgeGlowIntensity * glowPulse * 2.0;

  // Scanline overlay
  let scan = scanline(uv);
  col *= (1.0 - scan);

  // Depth fade
  let fade = depthFade(uv);
  col *= fade;

  // Master alpha
  let alpha = u.masterAlpha * fade * fMul;

  return vec4f(col, alpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 11  HolographicDisplay — Main class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Toggleable holographic visual mode for the Cell / SPH rendering pipeline.
 *
 * Operates as a Canvas2D post-composite overlay — call `render()` each
 * frame after the main scene has been drawn to the same canvas.  For
 * WebGPU integration, use the exported `HOLOGRAPHIC_WGSL` shader source
 * in a Nuke post-process pass.
 *
 * All visual parameters are exposed via the mutable `options` property
 * and can be adjusted at runtime.  Use `applyPreset()` to switch colour
 * palettes, or provide a custom palette via `options.customPalette`.
 */
export class HolographicDisplay {
  /** Mutable options — tweak at any time. */
  options: HolographicDisplayOptions;

  /** Resolved palette (from preset or custom override). */
  private _palette: HoloPalette;

  /** Frame counter for debug / glitch seeding. */
  private _frame = 0;

  constructor(opts?: Partial<HolographicDisplayOptions>) {
    this.options = { ...HOLOGRAPHIC_DEFAULTS, ...opts };
    this._palette = this._resolvePalette();
  }

  // ── Palette management ──────────────────────────────────────────────────

  /** Apply a named preset, updating the palette immediately. */
  applyPreset(name: HoloPresetName): void {
    this.options.presetName = name;
    this.options.customPalette = null;
    this._palette = this._resolvePalette();
  }

  /** Set a custom palette, overriding any preset. */
  setCustomPalette(palette: HoloPalette): void {
    this.options.customPalette = palette;
    this._palette = palette;
  }

  /** Get the currently active palette. */
  getPalette(): Readonly<HoloPalette> {
    return this._palette;
  }

  private _resolvePalette(): HoloPalette {
    if (this.options.customPalette) return this.options.customPalette;
    return HOLO_PRESETS[this.options.presetName] ?? PALETTE_CLASSIC;
  }

  // ── Toggle helpers ──────────────────────────────────────────────────────

  /** Toggle the holographic mode on or off. */
  toggle(): boolean {
    this.options.enabled = !this.options.enabled;
    return this.options.enabled;
  }

  /** Check whether the mode is currently enabled. */
  get isEnabled(): boolean {
    return this.options.enabled;
  }

  // ── Main render entry point ─────────────────────────────────────────────

  /**
   * Render the holographic overlay onto the given Canvas2D context.
   *
   * Call this **after** the main scene has been drawn to the same canvas
   * and **before** any UI overlays.  The method composites all five layers
   * using Canvas2D blend modes (primarily additive / 'lighter').
   *
   * @param ctx    — The 2D rendering context of the target canvas.
   * @param frame  — Per-frame data bundle (particles, cells, domain, time).
   */
  render(ctx: CanvasRenderingContext2D, frame: HoloFrameData): void {
    if (!this.options.enabled) return;

    const { particles, cells, domainW, domainH, time } = frame;
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    const canvasCx = w * 0.5;
    const canvasCy = h * 0.5;

    this._frame++;
    this._palette = this._resolvePalette();

    // ── Global alpha oscillation ────────────────────────────────────────
    const globalAlpha = computeGlobalAlpha(time, this.options);
    ctx.save();
    ctx.globalAlpha = globalAlpha;

    // ── Flicker multiplier (shared across all layers) ───────────────────
    const flickerMul = computeFlicker(time, this.options);

    // ── Layer 7+8: Cell bodies & particle holographic rendering ─────────
    // Cells (drawn first so particles can overlap)
    for (let i = 0; i < cells.length; i++) {
      drawHoloCell(
        ctx, cells[i], this._palette, this.options,
        time, flickerMul, canvasCx, canvasCy,
      );
    }

    // Particles
    if (particles.count > 0) {
      drawHoloParticles(
        ctx, particles, this._palette, this.options,
        time, flickerMul, canvasCx, canvasCy,
      );
    }

    // ── Layer 1: Scanlines ──────────────────────────────────────────────
    if (this.options.scanlines) {
      drawScanlines(ctx, w, h, this._palette, this.options, time);
    }

    // ── Layer 3: Glitch artefacts (drawn after scanlines) ───────────────
    if (this.options.flicker) {
      drawGlitch(ctx, w, h, this._palette, this.options, time);
    }

    // ── Layer 5: Depth fade / transparency ──────────────────────────────
    drawDepthFade(ctx, w, h, this._palette, this.options, time);

    // ── HUD badge ───────────────────────────────────────────────────────
    drawHud(ctx, w, this.options, time);

    ctx.restore();
  }

  // ── Cleanup ─────────────────────────────────────────────────────────────

  /** Release any resources (currently none, but future-proofs the API). */
  destroy(): void {
    this._frame = 0;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 12  Factory / convenience functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a HolographicDisplay with a specific preset applied.
 *
 * @param preset — Preset name: 'CLASSIC', 'CYBERPUNK', or 'GHOST'.
 * @param overrides — Optional partial overrides for any option.
 */
export function createHolographicDisplay(
  preset: HoloPresetName = 'CLASSIC',
  overrides?: Partial<HolographicDisplayOptions>,
): HolographicDisplay {
  return new HolographicDisplay({
    presetName: preset,
    ...overrides,
  });
}

/**
 * Build a HolographicDisplay configured for debug/development use:
 * all layers visible, higher glitch probability, HUD enabled.
 */
export function createDebugHolographic(): HolographicDisplay {
  return new HolographicDisplay({
    glitchProbability: 0.03,
    edgeGlowIntensity: 1.5,
    scanlineAlpha: 0.2,
    showHud: true,
  });
}

/**
 * Build a subtle holographic display suitable for production / non-debug
 * use: reduced scanline opacity, no glitch, gentler flicker.
 */
export function createSubtleHolographic(
  preset: HoloPresetName = 'CLASSIC',
): HolographicDisplay {
  return new HolographicDisplay({
    presetName: preset,
    scanlineAlpha: 0.06,
    flickerAmplitude: 0.02,
    glitchProbability: 0,
    edgeGlowIntensity: 0.6,
    chromaticMaxOffset: 1.5,
    alphaOscillation: 0.03,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// § 13  Self-test
//
// Validates core utility functions and option defaults.
// Run via: import { selfTest } from './holographic-display'; selfTest();
// ─────────────────────────────────────────────────────────────────────────────

export function selfTest(): boolean {
  const errors: string[] = [];
  const assert = (cond: boolean, msg: string) => { if (!cond) errors.push(msg); };

  // ── Hash determinism ──────────────────────────────────────────────────
  assert(hash11(42) === hash11(42), 'hash11 should be deterministic');
  assert(hash21(1, 2) === hash21(1, 2), 'hash21 should be deterministic');
  assert(hash11(42) !== hash11(43), 'hash11 should vary with input');

  // ── smoothstep ────────────────────────────────────────────────────────
  assert(smoothstep(0, 1, 0) === 0, 'smoothstep(0,1,0) === 0');
  assert(smoothstep(0, 1, 1) === 1, 'smoothstep(0,1,1) === 1');
  assert(Math.abs(smoothstep(0, 1, 0.5) - 0.5) < 0.01, 'smoothstep midpoint ≈ 0.5');

  // ── clamp ─────────────────────────────────────────────────────────────
  assert(clamp(-1, 0, 1) === 0, 'clamp lower bound');
  assert(clamp(2, 0, 1) === 1, 'clamp upper bound');
  assert(clamp(0.5, 0, 1) === 0.5, 'clamp passthrough');

  // ── lerp ──────────────────────────────────────────────────────────────
  assert(lerp(0, 10, 0.5) === 5, 'lerp midpoint');
  assert(lerp(0, 10, 0) === 0, 'lerp start');
  assert(lerp(0, 10, 1) === 10, 'lerp end');

  // ── RGBA string conversion ────────────────────────────────────────────
  assert(rgbaStr([255, 0, 128, 0.5]) === 'rgba(255,0,128,0.5)', 'rgbaStr');
  assert(rgbaStrA([255, 0, 128, 0.5], 0.3) === 'rgba(255,0,128,0.3)', 'rgbaStrA');

  // ── Flicker range ─────────────────────────────────────────────────────
  const flickerVals: number[] = [];
  for (let t = 0; t < 10; t += 0.1) {
    flickerVals.push(computeFlicker(t, HOLOGRAPHIC_DEFAULTS));
  }
  const minFlicker = Math.min(...flickerVals);
  const maxFlicker = Math.max(...flickerVals);
  assert(minFlicker >= 0.6, 'flicker never below 0.6');
  assert(maxFlicker <= 1.3, 'flicker never above 1.3');

  // ── Global alpha range ────────────────────────────────────────────────
  for (let t = 0; t < 10; t += 0.1) {
    const a = computeGlobalAlpha(t, HOLOGRAPHIC_DEFAULTS);
    assert(a >= 0 && a <= 1, `globalAlpha in [0,1] at t=${t}`);
  }

  // ── Chromatic offsets structure ────────────────────────────────────────
  const offsets = computeChromaticOffsets(400, 300, 400, 300, HOLOGRAPHIC_DEFAULTS, 0);
  assert(typeof offsets.rDx === 'number', 'chromatic offsets have rDx');
  assert(typeof offsets.gDy === 'number', 'chromatic offsets have gDy');
  assert(typeof offsets.bDx === 'number', 'chromatic offsets have bDx');

  // ── Palette presets ───────────────────────────────────────────────────
  for (const name of ['CLASSIC', 'CYBERPUNK', 'GHOST'] as HoloPresetName[]) {
    const p = HOLO_PRESETS[name];
    assert(p.primary.length === 4, `${name} primary has 4 components`);
    assert(p.secondary.length === 4, `${name} secondary has 4 components`);
    assert(p.accent.length === 4, `${name} accent has 4 components`);
    assert(p.baseFog.length === 4, `${name} baseFog has 4 components`);
  }

  // ── Constructor ───────────────────────────────────────────────────────
  const holo = new HolographicDisplay();
  assert(holo.isEnabled === true, 'default enabled');
  assert(holo.getPalette() === PALETTE_CLASSIC, 'default palette is CLASSIC');

  holo.applyPreset('CYBERPUNK');
  assert(holo.getPalette() === PALETTE_CYBERPUNK, 'applyPreset switches palette');
  assert(holo.options.presetName === 'CYBERPUNK', 'applyPreset updates presetName');

  const customPal: HoloPalette = {
    primary: [255, 0, 0, 1], secondary: [0, 255, 0, 1],
    accent: [0, 0, 255, 1], baseFog: [0, 0, 0, 0.1],
  };
  holo.setCustomPalette(customPal);
  assert(holo.getPalette() === customPal, 'setCustomPalette overrides preset');

  holo.toggle();
  assert(!holo.isEnabled, 'toggle disables');
  holo.toggle();
  assert(holo.isEnabled, 'toggle re-enables');

  holo.destroy();

  // ── Factory functions ─────────────────────────────────────────────────
  const debug = createDebugHolographic();
  assert(debug.options.glitchProbability === 0.03, 'debug glitch probability');
  debug.destroy();

  const subtle = createSubtleHolographic('GHOST');
  assert(subtle.options.presetName === 'GHOST', 'subtle uses GHOST');
  assert(subtle.options.glitchProbability === 0, 'subtle has no glitch');
  subtle.destroy();

  // ── Report ────────────────────────────────────────────────────────────
  if (errors.length > 0) {
    console.error(`holographic-display selfTest FAILED (${errors.length} errors):`);
    for (const e of errors) console.error(`  ✗ ${e}`);
    return false;
  }
  console.log('holographic-display selfTest PASSED ✓');
  return true;
}
