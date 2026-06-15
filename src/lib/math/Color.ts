/**
 * Color — RGB color (0–1 range internally)
 * Supports hex / css-rgb / hsl conversion and lerp.
 * AT gap-fill: math/utility (#75 xiaodi)
 */
export class Color {
  constructor(public r = 0, public g = 0, public b = 0, public a = 1) {}

  // ── factory ──────────────────────────────────────────────────────────────
  static black(): Color  { return new Color(0, 0, 0); }
  static white(): Color  { return new Color(1, 1, 1); }
  static red(): Color    { return new Color(1, 0, 0); }
  static green(): Color  { return new Color(0, 1, 0); }
  static blue(): Color   { return new Color(0, 0, 1); }

  /** Parse "#rrggbb", "#rgb", "#rrggbbaa" or "#rgba" hex strings. */
  static fromHex(hex: string): Color {
    const s = hex.replace('#', '');
    let r: number, g: number, b: number, a = 1;
    if (s.length === 3 || s.length === 4) {
      r = parseInt(s[0] + s[0], 16) / 255;
      g = parseInt(s[1] + s[1], 16) / 255;
      b = parseInt(s[2] + s[2], 16) / 255;
      if (s.length === 4) a = parseInt(s[3] + s[3], 16) / 255;
    } else if (s.length === 6 || s.length === 8) {
      r = parseInt(s.slice(0, 2), 16) / 255;
      g = parseInt(s.slice(2, 4), 16) / 255;
      b = parseInt(s.slice(4, 6), 16) / 255;
      if (s.length === 8) a = parseInt(s.slice(6, 8), 16) / 255;
    } else {
      throw new Error(`Color.fromHex: invalid hex "${hex}"`);
    }
    return new Color(r, g, b, a);
  }

  /** From 0–255 integer channels. */
  static fromRGB255(r: number, g: number, b: number, a = 255): Color {
    return new Color(r / 255, g / 255, b / 255, a / 255);
  }

  /** From HSL (h: 0–360, s: 0–1, l: 0–1). */
  static fromHSL(h: number, s: number, l: number, a = 1): Color {
    if (s === 0) { return new Color(l, l, l, a); }
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    const hk = h / 360;
    const r = Color._hue2rgb(p, q, hk + 1/3);
    const g = Color._hue2rgb(p, q, hk);
    const b = Color._hue2rgb(p, q, hk - 1/3);
    return new Color(r, g, b, a);
  }
  private static _hue2rgb(p: number, q: number, t: number): number {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1/6) return p + (q - p) * 6 * t;
    if (t < 1/2) return q;
    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
    return p;
  }

  // ── setters ───────────────────────────────────────────────────────────────
  set(r: number, g: number, b: number, a = 1): this {
    this.r = r; this.g = g; this.b = b; this.a = a; return this;
  }
  copy(c: Color): this { this.r = c.r; this.g = c.g; this.b = c.b; this.a = c.a; return this; }
  clone(): Color { return new Color(this.r, this.g, this.b, this.a); }

  // ── conversion ────────────────────────────────────────────────────────────
  /** Returns "#rrggbb" (no alpha) or "#rrggbbaa" when alpha < 1. */
  toHex(includeAlpha = false): string {
    const toHex2 = (v: number) =>
      Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).padStart(2, '0');
    const base = `#${toHex2(this.r)}${toHex2(this.g)}${toHex2(this.b)}`;
    return includeAlpha ? base + toHex2(this.a) : base;
  }

  /** Returns 0xRRGGBB integer. */
  toInt(): number {
    return (
      (Math.round(this.r * 255) << 16) |
      (Math.round(this.g * 255) << 8)  |
       Math.round(this.b * 255)
    );
  }

  /** Returns { h: 0–360, s: 0–1, l: 0–1, a }. */
  toHSL(): { h: number; s: number; l: number; a: number } {
    const max = Math.max(this.r, this.g, this.b);
    const min = Math.min(this.r, this.g, this.b);
    const l = (max + min) / 2;
    let h = 0, s = 0;
    if (max !== min) {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      switch (max) {
        case this.r: h = ((this.g - this.b) / d + (this.g < this.b ? 6 : 0)) / 6; break;
        case this.g: h = ((this.b - this.r) / d + 2) / 6; break;
        case this.b: h = ((this.r - this.g) / d + 4) / 6; break;
      }
    }
    return { h: h * 360, s, l, a: this.a };
  }

  /** Returns { r, g, b, a } in 0–255 range (integer). */
  toRGB255(): { r: number; g: number; b: number; a: number } {
    return {
      r: Math.round(this.r * 255),
      g: Math.round(this.g * 255),
      b: Math.round(this.b * 255),
      a: Math.round(this.a * 255),
    };
  }

  /** CSS rgba() string. */
  toCSSString(): string {
    const { r, g, b, a } = this.toRGB255();
    return `rgba(${r}, ${g}, ${b}, ${(this.a).toFixed(3)})`;
  }

  /** Flat Float32Array [r, g, b, a] for WebGL uniforms. */
  toFloat32Array(): Float32Array {
    return new Float32Array([this.r, this.g, this.b, this.a]);
  }

  // ── operations ────────────────────────────────────────────────────────────
  lerp(c: Color, t: number): Color {
    return new Color(
      this.r + (c.r - this.r) * t,
      this.g + (c.g - this.g) * t,
      this.b + (c.b - this.b) * t,
      this.a + (c.a - this.a) * t,
    );
  }
  lerpSelf(c: Color, t: number): this {
    this.r += (c.r - this.r) * t;
    this.g += (c.g - this.g) * t;
    this.b += (c.b - this.b) * t;
    this.a += (c.a - this.a) * t;
    return this;
  }
  add(c: Color): Color { return new Color(this.r+c.r, this.g+c.g, this.b+c.b, this.a+c.a); }
  mul(s: number): Color { return new Color(this.r*s, this.g*s, this.b*s, this.a); }
  /** Gamma → linear (sRGB approximate). */
  toLinear(): Color {
    return new Color(this.r**2.2, this.g**2.2, this.b**2.2, this.a);
  }
  /** Linear → gamma (sRGB approximate). */
  toGamma(): Color {
    return new Color(this.r**(1/2.2), this.g**(1/2.2), this.b**(1/2.2), this.a);
  }

  // ── comparison ────────────────────────────────────────────────────────────
  equals(c: Color, eps = 1e-9): boolean {
    return (
      Math.abs(this.r - c.r) <= eps &&
      Math.abs(this.g - c.g) <= eps &&
      Math.abs(this.b - c.b) <= eps &&
      Math.abs(this.a - c.a) <= eps
    );
  }

  toString(): string { return this.toCSSString(); }
}
