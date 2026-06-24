// src/lib/sph/chromatic-adaptation.ts
// M584: Chromatic Adaptation Color System
//
// 色彩自适应系统 — 粒子颜色根据物理状态 (密度/速度/温度) 在自然色彩梯度中渐变。
//
// 移植自三个 lygia WGSL 模块:
//   • upstream/lygia/color/palette.wgsl         → IQ cosine palette (Inigo Quilez)
//   • upstream/lygia/color/palette/heatmap.wgsl → heatmap(v) polynomial
//   • upstream/lygia/color/palette/fire.wgsl    → fire(x) exponential ramp
//   • upstream/lygia/color/palette/water.wgsl   → water(x) power ramp
//   • upstream/lygia/color/palette/spectral/zucconi6.wgsl → visible spectrum
//   • upstream/lygia/color/hueShift.wgsl        → hsl hue rotation
//   • upstream/lygia/color/blend/screen.wgsl    → blendScreen
//   • upstream/lygia/math/bump.wgsl             → bump() gaussian kernel
//
// 三条自然梯度轨迹 (ChromaticMode):
//   SUNSET  — 日落: 深蓝暮色 → 橙红地平 → 金黄顶光  (低→高密度)
//   ABYSS   — 深海: 午夜黑 → 海沟蓝 → 生物发光青绿 (低→高速度)
//   AURORA  — 极光: 暗绿 → 青蓝 → 品红 → 白 (温度驱动光谱游走)
//
// 每条轨迹由三个 IQ cosine palette 控制点合成, 与 heatmap() 做 screen 混合
// 添加高频细节, 最终用 zucconi6 光谱发光核在温度峰值叠加白色发光。




// ─────────────────────────────────────────────────────────────────────────────
// Public API types
// ─────────────────────────────────────────────────────────────────────────────

<<<<<<< HEAD
// [orphan-precise] /** 三种自然界色彩模式。 */
=======
/** 三种自然界色彩模式。 */



import type { RGB, RGBA } from './color-palette';

>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
export type ChromaticMode = 'SUNSET' | 'ABYSS' | 'AURORA';

/** 驱动色彩自适应的粒子物理量。 */
export interface ParticlePhysics {
  /** 归一化密度   0=真空  1=静止密度  >1=压缩 (上限 2.0) */
  densityNorm:  number;
  /** 归一化速度   0=静止  1=最大预期速度 */
  velocityNorm: number;
  /**
   * 归一化温度   0=冷  1=极热
   * 对于非热力学 SPH 可用 velocityNorm 的平方近似: T ≈ v²
   */
  tempNorm:     number;
  /** 像素坐标 (整数), 用于 Vlachos dither 避免色阶带 */
  pixelX:       number;
  pixelY:       number;
}

/** resolveChromatic 的完整输出。 */
export interface ChromaticColor extends RGBA {
  /** 0-1 亮度, 可送入后处理 bloom pass */
  luminance: number;
  /**
   * 光谱发光强度 [0,1], 来自 zucconi6 bump 核。
   * 可用于 WebGPU emissive channel 或 Canvas2D globalCompositeOperation:'screen'
   */
  emissive:  number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers (ported from lygia/math/)
// ─────────────────────────────────────────────────────────────────────────────

const TAU = Math.PI * 2;

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** bump(x, k) — lygia/math/bump.wgsl: saturate((1 - x*x) - k) */
function bump(x: number, k: number): number {
  return clamp01(1 - x * x - k);
}

/** Component-wise bump for vec3. */
function bump3(
  x: [number, number, number],
  k: [number, number, number],
): [number, number, number] {
  return [bump(x[0], k[0]), bump(x[1], k[1]), bump(x[2], k[2])];
}

// ─────────────────────────────────────────────────────────────────────────────
// Lygia port — palette.wgsl  (Inigo Quilez cosine palette)
// fn palette(t, a, b, c, d) -> a + b * cos(TAU * (c*t + d))
// ─────────────────────────────────────────────────────────────────────────────

interface Vec3 { r: number; g: number; b: number }

function iqPalette(t: number, a: Vec3, b: Vec3, c: Vec3, d: Vec3): RGB {
  return {
    r: a.r + b.r * Math.cos(TAU * (c.r * t + d.r)),
    g: a.g + b.g * Math.cos(TAU * (c.g * t + d.g)),
    b: a.b + b.b * Math.cos(TAU * (c.b * t + d.b)),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Lygia port — palette/heatmap.wgsl
// fn heatmap(v) -> 1 - (v*2.1 - vec3(1.8, 1.14, 0.3))²
// ─────────────────────────────────────────────────────────────────────────────

function heatmap(v: number): RGB {
  const rv = v * 2.1 - 1.8;
  const gv = v * 2.1 - 1.14;
  const bv = v * 2.1 - 0.3;
  return {
    r: clamp01(1 - rv * rv),
    g: clamp01(1 - gv * gv),
    b: clamp01(1 - bv * bv),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Lygia port — palette/fire.wgsl
// fn fire(x) -> vec3(1, 0.25, 0.0625) * exp(4*x - 1)
// ─────────────────────────────────────────────────────────────────────────────

function firePalette(x: number): RGB {
  const e = Math.exp(4 * x - 1);
  return {
    r: clamp01(1.0    * e),
    g: clamp01(0.25   * e),
    b: clamp01(0.0625 * e),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Lygia port — palette/water.wgsl
// fn water(x) -> pow(vec3(.1,.7,.8), vec3(4 * saturate(1-x)))
// ─────────────────────────────────────────────────────────────────────────────

function waterPalette(x: number): RGB {
  const exp = 4 * clamp01(1 - x);
  return {
    r: Math.pow(0.1, exp),
    g: Math.pow(0.7, exp),
    b: Math.pow(0.8, exp),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Lygia port — palette/spectral/zucconi6.wgsl
// Gaussian bump approximation of the visible spectrum (Alan Zucconi)
// Returns emissive glow contribution [0,1]³
// ─────────────────────────────────────────────────────────────────────────────

function spectralZucconi6(x: number): RGB {
  // c1, x1, y1 — first set of 3 bumps
  const c1: [number, number, number] = [3.54585104, 2.93225262, 2.41593945];
  const x1: [number, number, number] = [0.69549072, 0.49228336, 0.27699880];
  const y1: [number, number, number] = [0.02312639, 0.15225084, 0.52607955];
  // c2, x2, y2 — second set
  const c2: [number, number, number] = [3.90307140, 3.21182957, 3.96587128];
  const x2: [number, number, number] = [0.11748627, 0.86755042, 0.66077860];
  const y2: [number, number, number] = [0.84897130, 0.88445281, 0.73949448];

  const dx1: [number, number, number] = [
    c1[0] * (x - x1[0]),
    c1[1] * (x - x1[1]),
    c1[2] * (x - x1[2]),
  ];
  const dx2: [number, number, number] = [
    c2[0] * (x - x2[0]),
    c2[1] * (x - x2[1]),
    c2[2] * (x - x2[2]),
  ];

  const b1 = bump3(dx1, y1);
  const b2 = bump3(dx2, y2);

  return {
    r: clamp01(b1[0] + b2[0]),
    g: clamp01(b1[1] + b2[1]),
    b: clamp01(b1[2] + b2[2]),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Lygia port — blend/screen.wgsl  blendScreen(base, blend, opacity)
// ─────────────────────────────────────────────────────────────────────────────

function blendScreen(base: RGB, blend: RGB, opacity: number): RGB {
  const s = (b: number, bl: number) => 1 - (1 - b) * (1 - bl);
  return {
    r: s(base.r, blend.r) * opacity + base.r * (1 - opacity),
    g: s(base.g, blend.g) * opacity + base.g * (1 - opacity),
    b: s(base.b, blend.b) * opacity + base.b * (1 - opacity),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Lygia port — hueShift.wgsl
// hsl round-trip, shifts hue by `angle` radians
// ─────────────────────────────────────────────────────────────────────────────

function rgb2hsl(c: RGB): [number, number, number] {
  const max = Math.max(c.r, c.g, c.b);
  const min = Math.min(c.r, c.g, c.b);
  const l   = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h = 0;
  if      (max === c.r) h = (c.g - c.b) / d + (c.g < c.b ? 6 : 0);
  else if (max === c.g) h = (c.b - c.r) / d + 2;
  else                  h = (c.r - c.g) / d + 4;
  return [h / 6, s, l];
}

function hue2rgb(p: number, q: number, t: number): number {
  let tt = t;
  if (tt < 0) tt += 1;
  if (tt > 1) tt -= 1;
  if (tt < 1/6) return p + (q - p) * 6 * tt;
  if (tt < 1/2) return q;
  if (tt < 2/3) return p + (q - p) * (2/3 - tt) * 6;
  return p;
}

function hsl2rgb(h: number, s: number, l: number): RGB {
  if (s === 0) return { r: l, g: l, b: l };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return { r: hue2rgb(p, q, h + 1/3), g: hue2rgb(p, q, h), b: hue2rgb(p, q, h - 1/3) };
}

function hueShift(color: RGB, angle: number): RGB {
  const [h, s, l] = rgb2hsl(color);
  // mirrors WGSL: hsl.r = fract((hsl.r * TAU + a) / TAU)
  let newH = ((h * TAU + angle) / TAU) % 1;
  if (newH < 0) newH += 1;
  return hsl2rgb(newH, s, l);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dither — Vlachos 2016 (ported from color-palette.ts, kept for banding)
// ─────────────────────────────────────────────────────────────────────────────

const VLACHOS: ReadonlyArray<number> = [
   0, 8, 2,10,
  12, 4,14, 6,
   3,11, 1, 9,
  15, 7,13, 5,
];

function ditherVlachos(x: number, y: number): number {
  return VLACHOS[((y & 3) << 2) | (x & 3)] / 16 - 0.5;
}

// ─────────────────────────────────────────────────────────────────────────────
// Luminance helper (BT.601 weights)
// ─────────────────────────────────────────────────────────────────────────────

function luminance(c: RGB): number {
  return 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
}

// ─────────────────────────────────────────────────────────────────────────────
// IQ cosine palette presets for each mode
// All coefficients tuned to reference imagery:
//   SUNSET — 加利福尼亚日落照片色谱
//   ABYSS  — 马里亚纳海沟 ROV 拍摄 + 生物发光数据
//   AURORA — 挪威特罗姆瑟极光频谱
// ─────────────────────────────────────────────────────────────────────────────

interface PaletteCoeffs {
  a: Vec3; b: Vec3; c: Vec3; d: Vec3;
}

const IQ_PRESETS: Record<ChromaticMode, PaletteCoeffs> = {
  //
  // SUNSET: t=0 (低密度) → 深靛蓝暮色, t=0.5 → 橙红地平, t=1 → 金黄顶光
  // 参考 iq 日落 palette: a=(0.5,0.4,0.3) b=(0.5,0.4,0.3) c=(1,1,1) d=(0.0,0.08,0.25)
  //
  SUNSET: {
    a: { r: 0.52, g: 0.38, b: 0.30 },
    b: { r: 0.52, g: 0.38, b: 0.32 },
    c: { r: 1.00, g: 0.90, b: 0.85 },
    d: { r: 0.00, g: 0.08, b: 0.22 },
  },

  //
  // ABYSS: t=0 → 午夜黑蓝, t=0.5 → 深海蓝, t=1 → 生物发光青绿
  // 参考 deep-sea bioluminescence: 最高速度时出现发光冷绿
  //
  ABYSS: {
    a: { r: 0.05, g: 0.12, b: 0.28 },
    b: { r: 0.05, g: 0.18, b: 0.22 },
    c: { r: 0.60, g: 0.80, b: 1.00 },
    d: { r: 0.40, g: 0.22, b: 0.00 },
  },

  //
  // AURORA: t=0 → 暗绿, t=0.33 → 青蓝, t=0.66 → 品红, t=1 → 白热
  // 极光多线模拟: cos 频率拉高制造快速色相切换
  //
  AURORA: {
    a: { r: 0.30, g: 0.60, b: 0.50 },
    b: { r: 0.30, g: 0.40, b: 0.50 },
    c: { r: 2.00, g: 1.60, b: 1.20 },
    d: { r: 0.00, g: 0.25, b: 0.50 },
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Core gradient functions — one per ChromaticMode
//
// 每个函数将 [densityNorm, velocityNorm, tempNorm] 映射到 RGB + emissive。
// 混合策略来自 lygia palette + heatmap + screen blend 的组合。
// ─────────────────────────────────────────────────────────────────────────────

/**
 * SUNSET 日落梯度
 * 密度驱动 IQ cosine palette 在暮色→地平→顶光之间移动。
 * 速度叠加 heatmap 高光, 温度触发微弱红移 hueShift。
 */
function gradientSunset(dn: number, vn: number, tn: number): { color: RGB; emissive: number } {
  // 密度决定梯度位置 (0=冷暗, 1=暖亮)
  const t = clamp01(dn);
  const base = iqPalette(t, IQ_PRESETS.SUNSET.a, IQ_PRESETS.SUNSET.b,
                             IQ_PRESETS.SUNSET.c, IQ_PRESETS.SUNSET.d);

  // 速度 → heatmap 叠加 (高速时出现红橙热区)
  const heat = heatmap(vn * 0.85);
  let color  = blendScreen(base, heat, vn * 0.50);

  // 温度 → 暖红移 (日落下沉感)
  color = hueShift(color, tn * Math.PI * -0.08);

  // 高速+高温 → 轻微光谱发光 (地平线镜面反光)
  const emT   = clamp01(vn * tn);
  const emRGB = spectralZucconi6(0.68 + tn * 0.10);   // 可见光偏红端
  color        = blendScreen(color, emRGB, emT * 0.25);
  const emissive = luminance(emRGB) * emT * 0.4;

  return { color, emissive };
}

/**
 * ABYSS 深海梯度
 * 速度驱动 IQ cosine palette 在黑暗→蓝→生物发光青绿之间移动。
 * 密度叠加 water() 压力色, 高温激活发光核。
 */
function gradientAbyss(dn: number, vn: number, tn: number): { color: RGB; emissive: number } {
  // 速度决定梯度 (静止=深渊黑, 高速=生物发光)
  const t = clamp01(vn);
  const base = iqPalette(t, IQ_PRESETS.ABYSS.a, IQ_PRESETS.ABYSS.b,
                             IQ_PRESETS.ABYSS.c, IQ_PRESETS.ABYSS.d);

  // 密度 → water palette 压力叠加 (高密度时深蓝调更蓝更暗)
  const water = waterPalette(dn);
  let color   = blendScreen(base, water, clamp01(dn * 0.55));

  // 速度冷蓝移 (深海流动感)
  color = hueShift(color, -vn * Math.PI * 0.06);

  // 高温 + 高速 → 生物发光青绿光谱爆发 (zucconi6 可见光 490nm 峰)
  const emT   = clamp01(vn * tn * 2.0);
  const emRGB = spectralZucconi6(0.22 + vn * 0.08);   // 可见光蓝绿端
  color        = blendScreen(color, emRGB, emT * 0.55);
  const emissive = luminance(emRGB) * emT * 0.8;

  return { color, emissive };
}

/**
 * AURORA 极光梯度
 * 温度驱动 IQ cosine palette 在暗绿→青蓝→品红→白之间快速游走。
 * 密度调节饱和度, 速度叠加 fire() 能量带, zucconi6 光谱在峰值白化。
 */
function gradientAurora(dn: number, vn: number, tn: number): { color: RGB; emissive: number } {
  // 温度 = 极光能量, 驱动快速色相摆动
  const t = clamp01(tn);
  const base = iqPalette(t, IQ_PRESETS.AURORA.a, IQ_PRESETS.AURORA.b,
                             IQ_PRESETS.AURORA.c, IQ_PRESETS.AURORA.d);

  // 速度 → fire() 能量带 (极光柱内部的亮带)
  const fire = firePalette(clamp01(vn * 0.7));
  let color  = blendScreen(base, fire, vn * 0.35);

  // 密度降低饱和度 → 低密度区极光更透明、去饱和
  const [h, s, l] = rgb2hsl(color);
  const sAdj = s * (0.55 + dn * 0.45);
  color = hsl2rgb(h, clamp01(sAdj), l);

  // zucconi6 光谱白化: 温度峰值时整个可见光范围发光
  const emT   = clamp01(tn * tn);                      // 二次曲线: 低温不发光
  const emRGB = spectralZucconi6(0.15 + tn * 0.65);    // 从绿到紫扫描
  color        = blendScreen(color, emRGB, emT * 0.65);
  const emissive = luminance(emRGB) * emT * 0.95;       // 极光最亮

  return { color, emissive };
}

// ─────────────────────────────────────────────────────────────────────────────
// Mode dispatcher
// ─────────────────────────────────────────────────────────────────────────────

type GradientFn = (dn: number, vn: number, tn: number) => { color: RGB; emissive: number };

const GRADIENT_FNS: Record<ChromaticMode, GradientFn> = {
  SUNSET: gradientSunset,
  ABYSS:  gradientAbyss,
  AURORA: gradientAurora,
};

// ─────────────────────────────────────────────────────────────────────────────
// Public — resolveChromatic
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 为单个 SPH 粒子解析最终色彩自适应颜色。
 *
 * 管线:
 *  1. 根据 ChromaticMode 选择梯度函数 (SUNSET/ABYSS/AURORA)
 *  2. 梯度函数内部完成 IQ palette + heatmap/water/fire screen blend
 *  3. 最终用 zucconi6 叠加光谱发光
 *  4. Vlachos dither (1/255 振幅) 消除色阶带
 *  5. Alpha = 密度驱动不透明度 + 速度微调
 *
 * @param mode    自然界色彩模式
 * @param physics 粒子物理量
 */
export function resolveChromatic(
  mode:    ChromaticMode,
  physics: ParticlePhysics,
): ChromaticColor {
  // Clamp inputs
  const dn = clamp01(physics.densityNorm  / 2) * 2;   // 允许 densityNorm 到 2
  const vn = clamp01(physics.velocityNorm);
  const tn = clamp01(physics.tempNorm);

  // Gradient
  const grad    = GRADIENT_FNS[mode];
  const { color, emissive } = grad(clamp01(dn / 2), vn, tn);

  // Dither (Vlachos)
  const dither  = ditherVlachos(physics.pixelX, physics.pixelY) / 255;

  // Alpha: 密度为主 (0.25 地板), 速度补充高亮
  const alpha   = clamp01(0.25 + 0.55 * clamp01(dn / 2) + 0.20 * vn);

  return {
    r:         clamp01(color.r + dither),
    g:         clamp01(color.g + dither),
    b:         clamp01(color.b + dither),
    a:         alpha,
    luminance: luminance(color),
    emissive:  clamp01(emissive),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch helper — 填充 WebGPU storage buffer (r,g,b,a, luminance, emissive 交错)
//
//   stride = 6 floats per particle
//   colorBuf[i*6+0..3] = r,g,b,a
//   colorBuf[i*6+4]    = luminance
//   colorBuf[i*6+5]    = emissive
// ─────────────────────────────────────────────────────────────────────────────

export const CHROMATIC_STRIDE = 6;

export function batchResolveChromatic(
  colorBuf:   Float32Array,
  modes:      ChromaticMode[],
  densities:  Float32Array,
  velocities: Float32Array,
  temps:      Float32Array,
  pixelsX:    Int32Array,
  pixelsY:    Int32Array,
  count:      number,
): void {
  for (let i = 0; i < count; i++) {
    const c = resolveChromatic(modes[i], {
      densityNorm:  densities[i],
      velocityNorm: velocities[i],
      tempNorm:     temps[i],
      pixelX:       pixelsX[i],
      pixelY:       pixelsY[i],
    });
    const base = i * CHROMATIC_STRIDE;
    colorBuf[base]     = c.r;
    colorBuf[base + 1] = c.g;
    colorBuf[base + 2] = c.b;
    colorBuf[base + 3] = c.a;
    colorBuf[base + 4] = c.luminance;
    colorBuf[base + 5] = c.emissive;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: ChromaticColor → CSS rgba() string
// ─────────────────────────────────────────────────────────────────────────────

export function chromaticToCss(c: ChromaticColor): string {
  return `rgba(${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)},${c.a.toFixed(3)})`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: ChromaticColor → CSS rgba() + additive glow for Canvas2D
// 先画粒子本体 (source-over), 再画发光层 (lighter mode)
// ─────────────────────────────────────────────────────────────────────────────

export function chromaticToGlowCss(c: ChromaticColor): { body: string; glow: string } {
  const body = chromaticToCss(c);
  // 发光层: 在 emissive 强度下白色叠加
  const ge = c.emissive;
  const gr = Math.round(clamp01(c.r + ge * 0.5) * 255);
  const gg = Math.round(clamp01(c.g + ge * 0.5) * 255);
  const gb = Math.round(clamp01(c.b + ge * 0.5) * 255);
  const ga = (ge * c.a).toFixed(3);
  const glow = `rgba(${gr},${gg},${gb},${ga})`;
  return { body, glow };
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: QoS profile name → 推荐 ChromaticMode
// 让 chromatic-adaptation 与既有 color-palette.ts QoS 主题协同工作
// ─────────────────────────────────────────────────────────────────────────────

export type QoSChromaMap = Record<string, ChromaticMode>;

export const QOS_DEFAULT_CHROMA: QoSChromaMap = {
  SENSOR_DATA:  'ABYSS',    // 传感器: 深海发光 — 高频蓝绿脉冲
  PARAMETERS:   'SUNSET',   // 参数流: 日落暖橙 — 稳定的琥珀梯度
  TF_STATIC:    'AURORA',   // 静态变换: 极光绿 — 冻结的光谱带
  TOPO_CHANGE:  'AURORA',   // 拓扑变化: 极光品红 — 剧烈的色相扫描
  DEFAULT:      'SUNSET',   // 默认: 日落
};

/**
 * 根据 QoS 名称查找推荐 ChromaticMode。
 * 允许在运行时覆盖默认映射。
 */
export function qosToChromaMode(
  profileName: string,
  overrides?: Partial<QoSChromaMap>,
): ChromaticMode {
  const map = overrides ? { ...QOS_DEFAULT_CHROMA, ...overrides } : QOS_DEFAULT_CHROMA;
  return map[profileName] ?? 'SUNSET';
}
