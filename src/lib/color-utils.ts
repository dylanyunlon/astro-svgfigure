/**
 * color-utils.ts — 完整颜色工具库
 *
 * AT Color 模块 (107 次引用) 实现:
 *   Color           核心 RGBA 颜色类，工厂方法 + 混合模式 + 格式转换
 *   ColorHSL        HSL 色彩空间操作封装
 *   lerp            颜色线性插值 (动画 / 渐变)
 *   BlendMode       multiply / add / screen / overlay 混合
 *   SPECIES_PALETTE 10 种 cell species 预定义颜色方案 (与 proton-particles / pixi-cell-renderer 保持一致)
 *
 * 设计原则:
 *   - 所有计算在 [0,1] float 域完成，仅在 I/O 边界转换为 0-255 或 hex
 *   - 不可变操作返回新实例，便于函数式管道
 *   - Float32Array 输出直接送入 WebGL uniform
 *
 * Author: dylanyunlon <dogechat@163.com>
 */

// ── 内部工具 ─────────────────────────────────────────────────────────────────

/** 钳制到 [min, max] */
function clamp(v: number, min = 0, max = 1): number {
  return Math.max(min, Math.min(max, v));
}

/** 线性插值 */
function mix(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** hex 字符串 (无 #) 解析为 0-255 整数三元组 */
function parseHex6(hex: string): [number, number, number] {
  const n = parseInt(hex.replace('#', ''), 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}

// ── ColorHSL ──────────────────────────────────────────────────────────────────

/**
 * HSL 色彩空间封装.
 *   h ∈ [0, 360), s ∈ [0, 1], l ∈ [0, 1]
 */
export class ColorHSL {
  constructor(
    public h: number,
    public s: number,
    public l: number,
  ) {}

  /** HSL → RGB (all in [0,1]) */
  toRGB(): [number, number, number] {
    const { h, s, l } = this;
    if (s === 0) return [l, l, l];

    const hue2rgb = (p: number, q: number, t: number): number => {
      let tt = ((t % 1) + 1) % 1;
      if (tt < 1 / 6) return p + (q - p) * 6 * tt;
      if (tt < 1 / 2) return q;
      if (tt < 2 / 3) return p + (q - p) * (2 / 3 - tt) * 6;
      return p;
    };

    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    const hn = h / 360;

    return [
      hue2rgb(p, q, hn + 1 / 3),
      hue2rgb(p, q, hn),
      hue2rgb(p, q, hn - 1 / 3),
    ];
  }

  /** 色相旋转 delta 度 */
  rotate(delta: number): ColorHSL {
    return new ColorHSL(((this.h + delta) % 360 + 360) % 360, this.s, this.l);
  }

  /** 调整饱和度 (乘以因子) */
  saturate(factor: number): ColorHSL {
    return new ColorHSL(this.h, clamp(this.s * factor), this.l);
  }

  /** 调整亮度 */
  lighten(delta: number): ColorHSL {
    return new ColorHSL(this.h, this.s, clamp(this.l + delta));
  }

  /** 互补色 */
  complement(): ColorHSL {
    return this.rotate(180);
  }

  clone(): ColorHSL {
    return new ColorHSL(this.h, this.s, this.l);
  }

  toString(): string {
    return `hsl(${this.h.toFixed(1)}, ${(this.s * 100).toFixed(1)}%, ${(this.l * 100).toFixed(1)}%)`;
  }
}

// ── Color ─────────────────────────────────────────────────────────────────────

/**
 * RGBA 颜色类 — AT Color 核心.
 *   r, g, b, a ∈ [0, 1]
 */
export class Color {
  readonly r: number;
  readonly g: number;
  readonly b: number;
  readonly a: number;

  // ── 构造 ──────────────────────────────────────────────────────────────────

  constructor(r: number, g: number, b: number, a = 1) {
    this.r = clamp(r);
    this.g = clamp(g);
    this.b = clamp(b);
    this.a = clamp(a);
  }

  // ── 工厂方法 ──────────────────────────────────────────────────────────────

  /** 从 hex 字符串创建, 支持 #rrggbb / #rrggbbaa / #rgb / #rgba */
  static fromHex(hex: string): Color {
    const h = hex.replace('#', '');
    if (h.length === 3 || h.length === 4) {
      const [r, g, b, a] = h.split('').map(c => parseInt(c + c, 16) / 255);
      return new Color(r, g, b, a ?? 1);
    }
    if (h.length === 8) {
      const [r, g, b] = parseHex6(h.slice(0, 6));
      const a = parseInt(h.slice(6, 8), 16);
      return new Color(r / 255, g / 255, b / 255, a / 255);
    }
    const [r, g, b] = parseHex6(h);
    return new Color(r / 255, g / 255, b / 255);
  }

  /**
   * 从 HSL 创建.
   *   h ∈ [0, 360), s ∈ [0, 1], l ∈ [0, 1], a ∈ [0, 1]
   */
  static fromHSL(h: number, s: number, l: number, a = 1): Color {
    const hsl = new ColorHSL(h, s, l);
    const [r, g, b] = hsl.toRGB();
    return new Color(r, g, b, a);
  }

  /**
   * 从 CSS 颜色字符串创建.
   *   支持: #hex, rgb(...), rgba(...), hsl(...), hsla(...)
   */
  static fromCSS(css: string): Color {
    const s = css.trim();

    if (s.startsWith('#')) return Color.fromHex(s);

    // rgb(r, g, b) / rgba(r, g, b, a)
    const rgbaMatch = s.match(/rgba?\(\s*([\d.]+%?)\s*,\s*([\d.]+%?)\s*,\s*([\d.]+%?)(?:\s*,\s*([\d.]+))?\s*\)/);
    if (rgbaMatch) {
      const parse = (v: string, max: number) => v.endsWith('%') ? parseFloat(v) / 100 : parseFloat(v) / max;
      const r = parse(rgbaMatch[1], 255);
      const g = parse(rgbaMatch[2], 255);
      const b = parse(rgbaMatch[3], 255);
      const a = rgbaMatch[4] !== undefined ? parseFloat(rgbaMatch[4]) : 1;
      return new Color(r, g, b, a);
    }

    // hsl(h, s%, l%) / hsla(h, s%, l%, a)
    const hslaMatch = s.match(/hsla?\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%(?:\s*,\s*([\d.]+))?\s*\)/);
    if (hslaMatch) {
      const h = parseFloat(hslaMatch[1]);
      const sat = parseFloat(hslaMatch[2]) / 100;
      const l = parseFloat(hslaMatch[3]) / 100;
      const a = hslaMatch[4] !== undefined ? parseFloat(hslaMatch[4]) : 1;
      return Color.fromHSL(h, sat, l, a);
    }

    // CSS 命名色 (常用子集)
    const named: Record<string, string> = {
      black: '#000000', white: '#ffffff', red: '#ff0000',
      green: '#00ff00', blue: '#0000ff', transparent: '#00000000',
    };
    if (named[s]) return Color.fromHex(named[s]);

    console.warn(`[Color.fromCSS] Unrecognised format: "${css}", returning black`);
    return new Color(0, 0, 0);
  }

  /** 从 0xRRGGBB 整数 (PixiJS 风格) 创建 */
  static fromInt(hex: number, a = 1): Color {
    return new Color(
      ((hex >> 16) & 0xff) / 255,
      ((hex >> 8)  & 0xff) / 255,
      (hex & 0xff)         / 255,
      a,
    );
  }

  // ── 输出格式 ──────────────────────────────────────────────────────────────

  /** → #rrggbb */
  toHex(): string {
    const r = Math.round(this.r * 255).toString(16).padStart(2, '0');
    const g = Math.round(this.g * 255).toString(16).padStart(2, '0');
    const b = Math.round(this.b * 255).toString(16).padStart(2, '0');
    return `#${r}${g}${b}`;
  }

  /** → #rrggbbaa */
  toHexA(): string {
    const a = Math.round(this.a * 255).toString(16).padStart(2, '0');
    return `${this.toHex()}${a}`;
  }

  /** → CSS rgba(...) */
  toRGBA(): string {
    return `rgba(${Math.round(this.r * 255)}, ${Math.round(this.g * 255)}, ${Math.round(this.b * 255)}, ${this.a.toFixed(4)})`;
  }

  /** → Float32Array [r, g, b, a]  — 直接送 WebGL uniform4fv */
  toFloat32Array(): Float32Array {
    return new Float32Array([this.r, this.g, this.b, this.a]);
  }

  /** → 0xRRGGBB 整数 (PixiJS tint 兼容) */
  toInt(): number {
    return (
      (Math.round(this.r * 255) << 16) |
      (Math.round(this.g * 255) << 8)  |
      Math.round(this.b * 255)
    );
  }

  /** → ColorHSL */
  toHSL(): ColorHSL {
    const { r, g, b } = this;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const l = (max + min) / 2;
    if (max === min) return new ColorHSL(0, 0, l);

    const d = max - min;
    const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    let h: number;
    switch (max) {
      case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
      case g: h = ((b - r) / d + 2) / 6; break;
      default: h = ((r - g) / d + 4) / 6;
    }
    return new ColorHSL(h * 360, s, l);
  }

  // ── 混合模式 ──────────────────────────────────────────────────────────────

  /**
   * multiply: dst * src  (变暗)
   * C_out = C_src * C_dst
   */
  multiply(other: Color): Color {
    return new Color(
      this.r * other.r,
      this.g * other.g,
      this.b * other.b,
      this.a * other.a,
    );
  }

  /**
   * add: dst + src  (增亮 / additive blend — 粒子特效)
   * C_out = clamp(C_src + C_dst)
   */
  add(other: Color): Color {
    return new Color(
      this.r + other.r,
      this.g + other.g,
      this.b + other.b,
      clamp(this.a + other.a),
    );
  }

  /**
   * screen: 1 - (1-src)*(1-dst)  (增亮，不超曝)
   */
  screen(other: Color): Color {
    return new Color(
      1 - (1 - this.r) * (1 - other.r),
      1 - (1 - this.g) * (1 - other.g),
      1 - (1 - this.b) * (1 - other.b),
      1 - (1 - this.a) * (1 - other.a),
    );
  }

  /**
   * overlay: 暗部 multiply, 亮部 screen
   *   base < 0.5 → 2*base*blend
   *   base ≥ 0.5 → 1 - 2*(1-base)*(1-blend)
   */
  overlay(other: Color): Color {
    const ch = (base: number, blend: number) =>
      base < 0.5
        ? 2 * base * blend
        : 1 - 2 * (1 - base) * (1 - blend);
    return new Color(
      ch(this.r, other.r),
      ch(this.g, other.g),
      ch(this.b, other.b),
      this.a,
    );
  }

  // ── 颜色调整 ──────────────────────────────────────────────────────────────

  /** alpha 缩放 */
  withAlpha(a: number): Color {
    return new Color(this.r, this.g, this.b, clamp(a));
  }

  /** 乘以标量 (亮度缩放，保持 alpha) */
  scale(factor: number): Color {
    return new Color(this.r * factor, this.g * factor, this.b * factor, this.a);
  }

  /** 反色 */
  invert(): Color {
    return new Color(1 - this.r, 1 - this.g, 1 - this.b, this.a);
  }

  /** 灰度 (luminance weighted) */
  grayscale(): Color {
    const lum = 0.2126 * this.r + 0.7152 * this.g + 0.0722 * this.b;
    return new Color(lum, lum, lum, this.a);
  }

  clone(): Color {
    return new Color(this.r, this.g, this.b, this.a);
  }

  equals(other: Color, eps = 1e-6): boolean {
    return (
      Math.abs(this.r - other.r) < eps &&
      Math.abs(this.g - other.g) < eps &&
      Math.abs(this.b - other.b) < eps &&
      Math.abs(this.a - other.a) < eps
    );
  }

  toString(): string {
    return this.toRGBA();
  }
}

// ── lerp ─────────────────────────────────────────────────────────────────────

/**
 * lerp(a, b, t) — 颜色线性插值.
 *   t=0 → a, t=1 → b.  常用于动画过渡和渐变.
 */
export function lerp(a: Color, b: Color, t: number): Color {
  const tt = clamp(t);
  return new Color(
    mix(a.r, b.r, tt),
    mix(a.g, b.g, tt),
    mix(a.b, b.b, tt),
    mix(a.a, b.a, tt),
  );
}

// ── SPECIES_PALETTE ───────────────────────────────────────────────────────────

/**
 * 10 种 cell species 的完整颜色方案.
 *
 * 每个 entry 包含:
 *   base    主体颜色 (cell fill)
 *   border  边框颜色
 *   label   标签文字颜色
 *   glow    粒子发光颜色 (additive blend)
 *   dim     低活跃度暗化版 (α=0.4)
 *
 * 与 proton-particles.ts SPECIES_TINT 保持 toInt() 一致.
 */

export interface SpeciesPalette {
  /** cell 主体填充色 */
  base: Color;
  /** cell 边框颜色 */
  border: Color;
  /** 标签文字颜色 (需在 base 上可读) */
  label: Color;
  /** 粒子发光 / additive 特效颜色 */
  glow: Color;
  /** 低活跃度暗化版 */
  dim: Color;
}

export const SPECIES_PALETTE: Record<string, SpeciesPalette> = {
  'cil-eye': {
    base:   Color.fromHex('#7986CB'),   // Indigo 400
    border: Color.fromHex('#5C6BC0'),
    label:  Color.fromHex('#ffffff'),
    glow:   Color.fromHex('#9FA8DA').withAlpha(0.8),
    dim:    Color.fromHex('#7986CB').withAlpha(0.4),
  },
  'cil-vector': {
    base:   Color.fromHex('#81C784'),   // Green 300
    border: Color.fromHex('#66BB6A'),
    label:  Color.fromHex('#1B5E20'),
    glow:   Color.fromHex('#A5D6A7').withAlpha(0.8),
    dim:    Color.fromHex('#81C784').withAlpha(0.4),
  },
  'cil-bolt': {
    base:   Color.fromHex('#FFCC80'),   // Orange 200
    border: Color.fromHex('#FFA726'),
    label:  Color.fromHex('#4E342E'),
    glow:   Color.fromHex('#FFE0B2').withAlpha(0.9),
    dim:    Color.fromHex('#FFCC80').withAlpha(0.4),
  },
  'cil-plus': {
    base:   Color.fromHex('#F48FB1'),   // Pink 200
    border: Color.fromHex('#EC407A'),
    label:  Color.fromHex('#880E4F'),
    glow:   Color.fromHex('#F8BBD9').withAlpha(0.8),
    dim:    Color.fromHex('#F48FB1').withAlpha(0.4),
  },
  'cil-arrow-right': {
    base:   Color.fromHex('#B0BEC5'),   // Blue Grey 200
    border: Color.fromHex('#90A4AE'),
    label:  Color.fromHex('#263238'),
    glow:   Color.fromHex('#CFD8DC').withAlpha(0.7),
    dim:    Color.fromHex('#B0BEC5').withAlpha(0.4),
  },
  'cil-filter': {
    base:   Color.fromHex('#CE93D8'),   // Purple 200
    border: Color.fromHex('#AB47BC'),
    label:  Color.fromHex('#4A148C'),
    glow:   Color.fromHex('#E1BEE7').withAlpha(0.8),
    dim:    Color.fromHex('#CE93D8').withAlpha(0.4),
  },
  'cil-code': {
    base:   Color.fromHex('#80CBC4'),   // Teal 200
    border: Color.fromHex('#26A69A'),
    label:  Color.fromHex('#004D40'),
    glow:   Color.fromHex('#B2DFDB').withAlpha(0.8),
    dim:    Color.fromHex('#80CBC4').withAlpha(0.4),
  },
  'cil-layers': {
    base:   Color.fromHex('#90CAF9'),   // Blue 200
    border: Color.fromHex('#42A5F5'),
    label:  Color.fromHex('#0D47A1'),
    glow:   Color.fromHex('#BBDEFB').withAlpha(0.8),
    dim:    Color.fromHex('#90CAF9').withAlpha(0.4),
  },
  'cil-loop': {
    base:   Color.fromHex('#FFE082'),   // Amber 200
    border: Color.fromHex('#FFB300'),
    label:  Color.fromHex('#4E342E'),
    glow:   Color.fromHex('#FFF9C4').withAlpha(0.9),
    dim:    Color.fromHex('#FFE082').withAlpha(0.4),
  },
  'cil-graph': {
    base:   Color.fromHex('#EF9A9A'),   // Red 200
    border: Color.fromHex('#EF5350'),
    label:  Color.fromHex('#B71C1C'),
    glow:   Color.fromHex('#FFCDD2').withAlpha(0.8),
    dim:    Color.fromHex('#EF9A9A').withAlpha(0.4),
  },
};

/** 安全取 species 调色板，未知 species 返回中性灰 */
export function getSpeciesPalette(species: string): SpeciesPalette {
  return SPECIES_PALETTE[species] ?? {
    base:   new Color(0.7, 0.7, 0.7),
    border: new Color(0.5, 0.5, 0.5),
    label:  new Color(0.1, 0.1, 0.1),
    glow:   new Color(0.9, 0.9, 0.9, 0.6),
    dim:    new Color(0.7, 0.7, 0.7, 0.4),
  };
}

// ── Gradient ──────────────────────────────────────────────────────────────────

/**
 * 多色渐变 — stops 按 t ∈ [0,1] 排列.
 * 用于 epoch 颜色过渡和热力图.
 */
export class ColorGradient {
  private readonly stops: Array<{ t: number; color: Color }>;

  constructor(stops: Array<{ t: number; color: Color }>) {
    this.stops = [...stops].sort((a, b) => a.t - b.t);
  }

  /** 在 t 处采样颜色 */
  sample(t: number): Color {
    const tt = clamp(t);
    const { stops } = this;

    if (tt <= stops[0].t) return stops[0].color;
    if (tt >= stops[stops.length - 1].t) return stops[stops.length - 1].color;

    for (let i = 1; i < stops.length; i++) {
      if (tt <= stops[i].t) {
        const prev = stops[i - 1];
        const curr = stops[i];
        const local = (tt - prev.t) / (curr.t - prev.t);
        return lerp(prev.color, curr.color, local);
      }
    }

    return stops[stops.length - 1].color;
  }

  /** 采样 N 个等间距颜色 → Float32Array (layout: r0,g0,b0,a0, r1,g1,b1,a1, ...) */
  toFloat32Array(n: number): Float32Array {
    const out = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      const c = this.sample(i / (n - 1));
      out[i * 4]     = c.r;
      out[i * 4 + 1] = c.g;
      out[i * 4 + 2] = c.b;
      out[i * 4 + 3] = c.a;
    }
    return out;
  }
}

// ── 常用预设颜色 ──────────────────────────────────────────────────────────────

export const Colors = {
  WHITE:       new Color(1, 1, 1),
  BLACK:       new Color(0, 0, 0),
  TRANSPARENT: new Color(0, 0, 0, 0),
  RED:         new Color(1, 0, 0),
  GREEN:       new Color(0, 1, 0),
  BLUE:        new Color(0, 0, 1),

  /** 调试用红色高亮 */
  DEBUG:       Color.fromHex('#FF3333'),

  /** epoch 进度渐变: 冷蓝 → 暖橙 → 亮白 */
  EPOCH_GRADIENT: new ColorGradient([
    { t: 0,   color: Color.fromHex('#1A237E') },
    { t: 0.5, color: Color.fromHex('#FF6D00') },
    { t: 1,   color: Color.fromHex('#FFFFFF') },
  ]),
} as const;
