/**
 * particle-life-color.ts — M783: Particle Lifecycle Color
 * ─────────────────────────────────────────────────────────────────────────────
 * 粒子生命周期颜色系统 — 每个 SPH 粒子根据其年龄/阶段渐变出生→成长→成熟→衰亡：
 *
 *   BIRTH   (0.0–0.15)  白色闪光 → 品种色渐显 (新生)
 *   YOUNG   (0.15–0.40) 品种色最鲜亮 (充满活力)
 *   MATURE  (0.40–0.75) 品种色渐深, 对比度增强 (稳定期)
 *   DYING   (0.75–1.0)  暗红衰败, alpha渐出 (消亡)
 *
 * 叠加调制:
 *   速度 → 色温偏移 (快→暖白/橙, 慢→冷蓝)
 *   密度 → 饱和度增减 (密→高饱和, 疏→低饱和/灰)
 *
 * 品种色来源:
 *   每个 cil-* species 拥有独立的基色 (SPECIES_BASE_HUE), 映射自
 *   species-shader-registry 的 10 个 species 类型 + SPECIES_COLORS fallback。
 *
 * 与现有系统的关系:
 *   • color-palette.ts        — QoS → 粒子色, 本模块增加 *时间维度*
 *   • chromatic-adaptation.ts — 密度/速度/温度渐变, 本模块增加 *生死叙事*
 *   • spline-particle-life.ts — SPAWN/FLOW/DECAY/DEAD 状态机, 本模块提供颜色
 *   • cell-visual-identity.ts — cell 视觉, 本模块作用于 particle 级
 *
 * 上游引用:
 *   src/lib/sph/color-palette.ts           — RGB, RGBA, rgbaToCss, rgbaToU8
 *   src/lib/sph/species-shader-registry.ts — species ID 清单
 *   src/lib/sph/types.ts                   — ParticleData
 *   src/lib/sph/world-renderer.ts          — SPECIES_COLORS (legacy numeric)
 */




// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle phase
// ─────────────────────────────────────────────────────────────────────────────


import type { RGB, RGBA } from './color-palette';

<<<<<<< HEAD
// [orphan-precise] /**
// [orphan-precise]  * 粒子生命周期的四个阶段。
// [orphan-precise]  * 由归一化年龄 ageNorm ∈ [0, 1] 决定:
// [orphan-precise]  *   BIRTH   — [0.00, 0.15)  初生, 白色闪光渐变至品种色
// [orphan-precise]  *   YOUNG   — [0.15, 0.40)  年轻, 品种色最鲜明
// [orphan-precise]  *   MATURE  — [0.40, 0.75)  成熟, 品种色加深
// [orphan-precise]  *   DYING   — [0.75, 1.00]  衰亡, 向暗红过渡并淡出
// [orphan-precise]  */
=======
/**
 * 粒子生命周期的四个阶段。
 * 由归一化年龄 ageNorm ∈ [0, 1] 决定:
 *   BIRTH   — [0.00, 0.15)  初生, 白色闪光渐变至品种色
 *   YOUNG   — [0.15, 0.40)  年轻, 品种色最鲜明
 *   MATURE  — [0.40, 0.75)  成熟, 品种色加深
 *   DYING   — [0.75, 1.00]  衰亡, 向暗红过渡并淡出
 */




>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
export type LifecyclePhase = 'BIRTH' | 'YOUNG' | 'MATURE' | 'DYING';

/** 阶段边界常量 — 保持与文档一致 */
const PHASE_BIRTH_END  = 0.15;
const PHASE_YOUNG_END  = 0.40;
const PHASE_MATURE_END = 0.75;
// DYING runs from PHASE_MATURE_END to 1.0

/** 判定粒子所处阶段。 */
export function getLifecyclePhase(ageNorm: number): LifecyclePhase {
  if (ageNorm < PHASE_BIRTH_END)  return 'BIRTH';
  if (ageNorm < PHASE_YOUNG_END)  return 'YOUNG';
  if (ageNorm < PHASE_MATURE_END) return 'MATURE';
  return 'DYING';
}

// ─────────────────────────────────────────────────────────────────────────────
// Species base hue (HSL hue angle, 0–1 range)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 每个 cil-* species 的「品种色」以 HSL hue 角度 + 固有饱和度/亮度定义。
 * hue 值映射自 species-shader-registry 的语义:
 *   cil-eye          → 注意力/感知   →  深靛蓝  (indigo)
 *   cil-bolt         → 能量/激活     →  琥珀橙  (amber)
 *   cil-vector       → 方向/嵌入     →  翠绿    (emerald)
 *   cil-plus         → 残差/叠加     →  紫罗兰  (violet)
 *   cil-arrow-right  → 前馈/传输     →  青色    (cyan)
 *   cil-filter       → 注意力过滤    →  品红    (magenta)
 *   cil-code          → 计算/FFN     →  天蓝    (sky blue)
 *   cil-layers       → 层叠/归一化   →  橄榄绿  (olive)
 *   cil-loop         → 循环/递归     →  珊瑚红  (coral)
 *   cil-graph        → 拓扑/结构     →  青绿    (teal)
 */
export interface SpeciesBaseColor {
  /** HSL hue, 0–1 (0=red, 0.33=green, 0.67=blue) */
  hue: number;
  /** Base saturation at YOUNG phase peak, 0–1 */
  saturation: number;
  /** Base lightness at YOUNG phase peak, 0–1 */
  lightness: number;
}

export const SPECIES_BASE_COLOR: Record<string, SpeciesBaseColor> = {
  'cil-eye':         { hue: 0.69, saturation: 0.72, lightness: 0.52 },  // 深靛蓝
  'cil-bolt':        { hue: 0.09, saturation: 0.85, lightness: 0.55 },  // 琥珀橙
  'cil-vector':      { hue: 0.40, saturation: 0.68, lightness: 0.45 },  // 翠绿
  'cil-plus':        { hue: 0.78, saturation: 0.65, lightness: 0.50 },  // 紫罗兰
  'cil-arrow-right': { hue: 0.52, saturation: 0.75, lightness: 0.50 },  // 青色
  'cil-filter':      { hue: 0.88, saturation: 0.78, lightness: 0.52 },  // 品红
  'cil-code':        { hue: 0.58, saturation: 0.70, lightness: 0.55 },  // 天蓝
  'cil-layers':      { hue: 0.22, saturation: 0.55, lightness: 0.45 },  // 橄榄绿
  'cil-loop':        { hue: 0.03, saturation: 0.80, lightness: 0.58 },  // 珊瑚红
  'cil-graph':       { hue: 0.47, saturation: 0.60, lightness: 0.42 },  // 青绿
};

/** 回退品种色 (未注册 species 使用中性灰蓝) */
const FALLBACK_BASE: SpeciesBaseColor = {
  hue: 0.60, saturation: 0.30, lightness: 0.55,
};

/** 衰亡暗红色 (dying target) */
const DYING_COLOR: SpeciesBaseColor = {
  hue: 0.00,       // 红色
  saturation: 0.45, // 低饱和暗红
  lightness: 0.18,  // 很暗
};

/** 初生白色 (birth flash) */
const BIRTH_WHITE: RGB = { r: 0.96, g: 0.97, b: 1.0 };

// ─────────────────────────────────────────────────────────────────────────────
// HSL ↔ RGB (与 color-palette.ts 使用相同算法, 此处独立以避免循环依赖)
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

// ─────────────────────────────────────────────────────────────────────────────
// Utility — clamp, lerp, smoothstep
// ─────────────────────────────────────────────────────────────────────────────

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpRGB(a: RGB, b: RGB, t: number): RGB {
  return {
    r: lerp(a.r, b.r, t),
    g: lerp(a.g, b.g, t),
    b: lerp(a.b, b.b, t),
  };
}

/** Hermite smoothstep for smooth phase transitions. */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

// ─────────────────────────────────────────────────────────────────────────────
// Velocity → Color Temperature (色温偏移)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 速度归一化值 → hue 偏移量 (在品种色基础上叠加)。
 *   velocityNorm = 0   →  冷蓝偏移 (hue += +0.08, 向蓝端)
 *   velocityNorm = 0.5 →  无偏移 (中性)
 *   velocityNorm = 1   →  暖橙偏移 (hue += -0.06, 向红/橙端)
 *
 * 同时影响亮度: 高速 → 更亮 (能量感), 低速 → 略暗。
 */
interface VelocityColorShift {
  /** Hue delta (additive, wraps in [0,1]) */
  hueDelta: number;
  /** Lightness multiplier (1.0 = no change) */
  lightnessMul: number;
}

function velocityToColorTemperature(velocityNorm: number): VelocityColorShift {
  const vn = clamp01(velocityNorm);

  // hue: 冷蓝 (+0.08) at v=0  →  暖橙 (-0.06) at v=1
  // uses smoothstep for perceptually linear transition
  const hueDelta = lerp(0.08, -0.06, smoothstep(0, 1, vn));

  // lightness: slow particles slightly dimmer, fast particles brighter
  const lightnessMul = lerp(0.88, 1.15, smoothstep(0, 1, vn));

  return { hueDelta, lightnessMul };
}

// ─────────────────────────────────────────────────────────────────────────────
// Density → Saturation (饱和度调制)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 密度归一化值 → 饱和度乘数。
 *   densityNorm = 0   →  低饱和 (0.35): 稀疏区域颜色发灰
 *   densityNorm = 1   →  全饱和 (1.0):  正常密度最鲜明
 *   densityNorm > 1   →  过饱和 (1.1+): 压缩区域颜色更浓烈
 */
function densityToSaturationMul(densityNorm: number): number {
  const dn = Math.min(Math.max(densityNorm, 0), 2);

  if (dn <= 1) {
    // 0→0.35, 1→1.0 with smoothstep easing
    return lerp(0.35, 1.0, smoothstep(0, 1, dn));
  }
  // over-pressure: subtle increase up to 1.15 at density=2
  return lerp(1.0, 1.15, smoothstep(1, 2, dn));
}

// ─────────────────────────────────────────────────────────────────────────────
// Core lifecycle color resolution
// ─────────────────────────────────────────────────────────────────────────────

/** Input to the lifecycle color resolver. */
export interface LifecycleColorInput {
  /** Species identifier (e.g. 'cil-eye'). Falls back to neutral if unknown. */
  speciesId: string;

  /**
   * Normalised age of this particle: 0.0 = just born, 1.0 = about to die.
   * Computed as `(currentTime - birthTime) / maxLifespan`.
   */
  ageNorm: number;

  /** Normalised velocity magnitude: 0 = still, 1 = max expected speed. */
  velocityNorm: number;

  /** Normalised SPH density: 0 = vacuum, 1 = rest density, >1 = compressed. */
  densityNorm: number;
}

/** Output of the lifecycle color resolver. */
export interface LifecycleColor extends RGBA {
  /** Which lifecycle phase the particle is currently in. */
  phase: LifecyclePhase;

  /**
   * Raw luminance (0–1) for bloom/emissive pass.
   * BIRTH flash is high luminance; DYING is very low.
   */
  luminance: number;
}

/**
 * Resolve the full RGBA colour for a particle based on its lifecycle stage,
 * velocity, and density.
 *
 * Pipeline:
 *   1. Determine lifecycle phase from ageNorm.
 *   2. Compute phase-specific base colour (white → species → dark → dying red).
 *   3. Apply velocity → color temperature shift (hue + lightness).
 *   4. Apply density → saturation modulation.
 *   5. Compute alpha (full during life, fade-out during DYING).
 *   6. Compute luminance for bloom pass.
 */
export function resolveLifecycleColor(input: LifecycleColorInput): LifecycleColor {
  const age   = clamp01(input.ageNorm);
  const phase = getLifecyclePhase(age);
  const base  = SPECIES_BASE_COLOR[input.speciesId] ?? FALLBACK_BASE;

  // ── Step 1: Phase-specific HSL target ──────────────────────────────────

  let h: number;
  let s: number;
  let l: number;

  switch (phase) {
    case 'BIRTH': {
      // White flash → species colour over [0, PHASE_BIRTH_END]
      const t = smoothstep(0, PHASE_BIRTH_END, age);
      // Start from white (low sat, high light) → species base
      h = base.hue;                              // hue snaps to species immediately
      s = lerp(0.05, base.saturation, t);         // saturation rises from near-zero
      l = lerp(0.95, base.lightness, t);          // lightness drops from white
      break;
    }
    case 'YOUNG': {
      // Species colour at full vibrancy — slight saturation/lightness boost
      const t = smoothstep(PHASE_BIRTH_END, PHASE_YOUNG_END, age);
      h = base.hue;
      s = lerp(base.saturation, base.saturation * 1.08, t);  // tiny sat increase
      l = lerp(base.lightness, base.lightness * 0.95, t);    // slight darkening toward mature
      break;
    }
    case 'MATURE': {
      // Species colour deepens: saturation stable, lightness drops
      const t = smoothstep(PHASE_YOUNG_END, PHASE_MATURE_END, age);
      h = base.hue;
      s = lerp(base.saturation * 1.08, base.saturation * 0.90, t);
      l = lerp(base.lightness * 0.95, base.lightness * 0.65, t);  // noticeably darker
      break;
    }
    case 'DYING': {
      // Deep species → dying dark red
      const t = smoothstep(PHASE_MATURE_END, 1.0, age);
      h = lerp(base.hue, DYING_COLOR.hue, t);
      // handle hue wrapping: if base.hue > 0.5 and target is 0, go via 1.0
      if (base.hue > 0.5) {
        const wrappedTarget = DYING_COLOR.hue + 1.0;
        h = lerp(base.hue, wrappedTarget, t) % 1.0;
      }
      s = lerp(base.saturation * 0.90, DYING_COLOR.saturation, t);
      l = lerp(base.lightness * 0.65, DYING_COLOR.lightness, t);
      break;
    }
  }

  // ── Step 2: Velocity → color temperature shift ─────────────────────────
  const velShift = velocityToColorTemperature(input.velocityNorm);
  h = ((h + velShift.hueDelta) % 1 + 1) % 1;  // wrap to [0, 1]
  l = clamp01(l * velShift.lightnessMul);

  // ── Step 3: Density → saturation modulation ────────────────────────────
  const satMul = densityToSaturationMul(input.densityNorm);
  s = clamp01(s * satMul);

  // ── Step 4: Convert HSL → RGB ──────────────────────────────────────────
  let color = hsl2rgb(h, s, l);

  // ── Step 5: BIRTH phase — blend toward pure white for the initial flash
  if (phase === 'BIRTH') {
    const flashT = 1 - smoothstep(0, PHASE_BIRTH_END * 0.6, age);
    color = lerpRGB(color, BIRTH_WHITE, flashT * 0.7);
  }

  // ── Step 6: Alpha ──────────────────────────────────────────────────────
  let alpha: number;
  if (phase === 'DYING') {
    // Smooth fade-out over the dying phase
    const t = smoothstep(PHASE_MATURE_END, 1.0, age);
    alpha = lerp(1.0, 0.0, t * t);  // quadratic ease-out for gentle fade
  } else if (phase === 'BIRTH') {
    // Quick fade-in at birth
    alpha = smoothstep(0, PHASE_BIRTH_END * 0.3, age);
  } else {
    alpha = 1.0;
  }

  // ── Step 7: Luminance ──────────────────────────────────────────────────
  // Perceptual luminance (Rec. 709) with phase-specific boost
  const baseLum = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
  let luminance: number;
  switch (phase) {
    case 'BIRTH':
      // Birth flash has high luminance for bloom
      luminance = lerp(0.9, baseLum, smoothstep(0, PHASE_BIRTH_END, age));
      break;
    case 'YOUNG':
      luminance = baseLum * 1.1;  // slight bloom
      break;
    case 'MATURE':
      luminance = baseLum;
      break;
    case 'DYING':
      luminance = baseLum * 0.5;  // dim
      break;
  }

  return {
    r: clamp01(color.r),
    g: clamp01(color.g),
    b: clamp01(color.b),
    a: clamp01(alpha),
    phase,
    luminance: clamp01(luminance),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience — CSS string output
// ─────────────────────────────────────────────────────────────────────────────

/** Convert LifecycleColor to CSS rgba() string for Canvas2D rendering. */
export function lifecycleColorToCss(c: LifecycleColor): string {
  const r = Math.round(c.r * 255);
  const g = Math.round(c.g * 255);
  const b = Math.round(c.b * 255);
  return `rgba(${r},${g},${b},${c.a.toFixed(3)})`;
}

/** Pack LifecycleColor to Uint8 [r,g,b,a] for GPU texture writes. */
export function lifecycleColorToU8(c: LifecycleColor): [number, number, number, number] {
  return [
    Math.round(c.r * 255),
    Math.round(c.g * 255),
    Math.round(c.b * 255),
    Math.round(c.a * 255),
  ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy species index → species ID mapping
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Maps numeric species index (from ParticleData.species Uint32Array) to
 * cil-* species ID string. Matches SPECIES_COLORS ordering in world-renderer.
 */
const SPECIES_INDEX_TO_ID: ReadonlyArray<string> = [
  'cil-eye',          // 0 — #3F51B5 indigo
  'cil-bolt',         // 1 — #FF6F00 amber
  'cil-vector',       // 2 — #2E7D32 green
  'cil-loop',         // 3 — #C62828 red
  'cil-arrow-right',  // 4 — #455A64 grey-blue
  'cil-filter',       // 5 — #7B1FA2 purple
  'cil-code',         // 6 — #1565C0 blue
];

/** Resolve numeric species index to cil-* species ID. */
export function speciesIndexToId(index: number): string {
  return SPECIES_INDEX_TO_ID[index] ?? '';
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch resolution — fill a Float32Array for GPU upload
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Batch-resolve lifecycle colours for an array of particles.
 *
 * Fills `colorBuf` (stride 4: r, g, b, a) and optionally `luminanceBuf`
 * (stride 1) for bloom pass.
 *
 * @param colorBuf      Float32Array of length count × 4 (r,g,b,a interleaved)
 * @param speciesIds    Species ID per particle (string[])
 * @param ages          Float32Array of normalised ages [0,1] per particle
 * @param velocities    Float32Array of normalised velocity magnitudes
 * @param densities     Float32Array of normalised densities
 * @param count         Number of active particles
 * @param luminanceBuf  Optional Float32Array of length count for bloom luminance
 */
export function batchResolveLifecycleColors(
  colorBuf:      Float32Array,
  speciesIds:    string[],
  ages:          Float32Array,
  velocities:    Float32Array,
  densities:     Float32Array,
  count:         number,
  luminanceBuf?: Float32Array,
): void {
  for (let i = 0; i < count; i++) {
    const c = resolveLifecycleColor({
      speciesId:    speciesIds[i],
      ageNorm:      ages[i],
      velocityNorm: velocities[i],
      densityNorm:  densities[i],
    });
    const base = i * 4;
    colorBuf[base]     = c.r;
    colorBuf[base + 1] = c.g;
    colorBuf[base + 2] = c.b;
    colorBuf[base + 3] = c.a;

    if (luminanceBuf) {
      luminanceBuf[i] = c.luminance;
    }
  }
}

/**
 * Batch variant accepting numeric species indices (from ParticleData.species).
 * Converts internally via speciesIndexToId().
 */
export function batchResolveLifecycleColorsIndexed(
  colorBuf:     Float32Array,
  speciesIdx:   Uint32Array,
  ages:         Float32Array,
  velocities:   Float32Array,
  densities:    Float32Array,
  count:        number,
  luminanceBuf?: Float32Array,
): void {
  for (let i = 0; i < count; i++) {
    const c = resolveLifecycleColor({
      speciesId:    speciesIndexToId(speciesIdx[i]),
      ageNorm:      ages[i],
      velocityNorm: velocities[i],
      densityNorm:  densities[i],
    });
    const base = i * 4;
    colorBuf[base]     = c.r;
    colorBuf[base + 1] = c.g;
    colorBuf[base + 2] = c.b;
    colorBuf[base + 3] = c.a;

    if (luminanceBuf) {
      luminanceBuf[i] = c.luminance;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Debug / visualisation helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generate a lifecycle color ramp for a given species — useful for
 * debug overlays, colour legend UI, and gradient texture generation.
 *
 * @param speciesId   Species to preview (e.g. 'cil-eye')
 * @param steps       Number of sample points along the age axis (default 64)
 * @param velocity    Fixed velocityNorm for the preview (default 0.5)
 * @param density     Fixed densityNorm for the preview (default 1.0)
 * @returns           Array of LifecycleColor, one per step
 */
export function generateLifecycleRamp(
  speciesId: string,
  steps = 64,
  velocity = 0.5,
  density = 1.0,
): LifecycleColor[] {
  const ramp: LifecycleColor[] = [];
  for (let i = 0; i < steps; i++) {
    const ageNorm = i / (steps - 1);
    ramp.push(resolveLifecycleColor({
      speciesId,
      ageNorm,
      velocityNorm: velocity,
      densityNorm: density,
    }));
  }
  return ramp;
}

/**
 * Generate a 1D gradient texture (Uint8Array, RGBA, width × 1) encoding the
 * lifecycle ramp for a species. Suitable for uploading as a WebGPU/WebGL
 * texture to sample in a shader via `texture(lifecycleLUT, vec2(ageNorm, 0))`.
 *
 * @param speciesId   Species to encode
 * @param width       Texture width in pixels (default 256)
 * @param velocity    Fixed velocityNorm
 * @param density     Fixed densityNorm
 * @returns           Uint8Array of length width × 4
 */
export function generateLifecycleLUT(
  speciesId: string,
  width = 256,
  velocity = 0.5,
  density = 1.0,
): Uint8Array {
  const buf = new Uint8Array(width * 4);
  for (let x = 0; x < width; x++) {
    const ageNorm = x / (width - 1);
    const c = resolveLifecycleColor({
      speciesId,
      ageNorm,
      velocityNorm: velocity,
      densityNorm: density,
    });
    const offset = x * 4;
    buf[offset]     = Math.round(c.r * 255);
    buf[offset + 1] = Math.round(c.g * 255);
    buf[offset + 2] = Math.round(c.b * 255);
    buf[offset + 3] = Math.round(c.a * 255);
  }
  return buf;
}
