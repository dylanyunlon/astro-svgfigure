/**
 * pixi-cell-renderer.ts — Method 1: PixiJS Graphics cell renderer
 *
 * Cell 只输出参数（species、bbox、z），PixiJS 负责所有视觉：
 * - 圆角矩形 (Graphics.roundRect)
 * - Species 内部图案 (procedural Graphics draw calls)
 * - 每个 cell 背后的 bloom glow（BlurFilter + additive blend）
 * - 贝塞尔曲线连线 (Graphics.bezierCurveTo)
 * - Anti-aliasing（GPU 子像素平滑）
 *
 * Live Poll 模式 (pollCellChannels):
 * - 每 500ms fetch /api/cells 拉取最新 CellDescriptor[]
 * - 位置变化 → lerp 平滑过渡 (alpha += (target - current) * 0.1)
 * - 新 cell fade in (alpha 0→1)，消失 cell fade out (alpha 1→0) 后销毁
 * - edge layer 每帧跟着 cell 当前位置实时重绘
 *
 * Upstream reference:
 *   upstream/pixijs-engine/src/scene/graphics/shared/Graphics.ts
 *   upstream/pixijs-engine/src/filters/defaults/blur/
 *   skills/pixijs/pixijs-filters/SKILL.md
 *   skills/pixijs/pixijs-graphics/SKILL.md
 */

import { Application } from '../../upstream/pixijs-engine/src/app/Application';
import { Container } from '../../upstream/pixijs-engine/src/scene/container/Container';
import { Graphics } from '../../upstream/pixijs-engine/src/scene/graphics/shared/Graphics';
import { Text } from '../../upstream/pixijs-engine/src/scene/text/Text';
import { TextStyle } from '../../upstream/pixijs-engine/src/scene/text/TextStyle';
import { Ticker } from '../../upstream/pixijs-engine/src/ticker/Ticker';
import { AdvancedBloomFilter } from '../../../upstream/pixijs-filters/src/advanced-bloom';
import { DropShadowFilter } from '../../../upstream/pixijs-filters/src/drop-shadow';

// ── params.json schema (M152 output) ───────────────────────────────────────

export interface ParamsJson {
  cell_id: string;
  species: string;
  bbox: { x: number; y: number; w: number; h: number; z?: number };
  z?: number;
  /** crowding_opacity — sets container.alpha */
  opacity?: number;
  /** hex string e.g. "#3F51B5" */
  fill_color?: string;
  /** hex string e.g. "#3F51B5" */
  stroke_color?: string;
  label?: string;
  font_size?: number;
  shadow?: {
    dx: number;
    dy: number;
    blur: number;
    opacity: number;
  };
  species_params?: Record<string, unknown>;
  epoch?: number;
}

// ── Cell descriptor — this is ALL the LLM needs to produce ──────────────────

export interface CellDescriptor {
  cell_id: string;
  label: string;
  species: string;
  bbox: { x: number; y: number; w: number; h: number };
  z: number;
  topology: {
    incoming_edges: string[];
    outgoing_edges: string[];
  };
  /** Optional params.json overlay (M152 output) — merged at build time or runtime */
  params?: ParamsJson;
}

export interface EdgeDescriptor {
  id: string;
  source: string;
  target: string;
  type: 'normal' | 'skip_connection';
}

// ── Species colour palette ──────────────────────────────────────────────────

const SPECIES_COLOURS: Record<string, { fill: number; stroke: number; glow: number }> = {
  'cil-eye':         { fill: 0x5C6BC0, stroke: 0x3949AB, glow: 0x7986CB },
  'cil-vector':      { fill: 0x66BB6A, stroke: 0x388E3C, glow: 0x81C784 },
  'cil-bolt':        { fill: 0xFFA726, stroke: 0xF57C00, glow: 0xFFCC80 },
  'cil-plus':        { fill: 0xEC407A, stroke: 0xC62828, glow: 0xF48FB1 },
  'cil-arrow-right': { fill: 0x78909C, stroke: 0x455A64, glow: 0xB0BEC5 },
  'cil-filter':      { fill: 0xAB47BC, stroke: 0x7B1FA2, glow: 0xCE93D8 },
  'cil-code':        { fill: 0x26A69A, stroke: 0x00796B, glow: 0x80CBC4 },
  'cil-layers':      { fill: 0x42A5F5, stroke: 0x1565C0, glow: 0x90CAF9 },
  'cil-loop':        { fill: 0xFFCA28, stroke: 0xF9A825, glow: 0xFFE082 },
  'cil-graph':       { fill: 0x78909C, stroke: 0x37474F, glow: 0xB0BEC5 },
};

function hexToNum(hex: string): number {
  return parseInt(hex.replace('#', ''), 16);
}

function getColours(
  species: string,
  fillOverride?: string,
  strokeOverride?: string,
): { fill: number; stroke: number; glow: number } {
  const base = SPECIES_COLOURS[species] ?? { fill: 0x90A4AE, stroke: 0x607D8B, glow: 0xB0BEC5 };
  return {
    fill:   fillOverride   ? hexToNum(fillOverride)   : base.fill,
    stroke: strokeOverride ? hexToNum(strokeOverride) : base.stroke,
    glow:   base.glow,
  };
}

// ── Species pattern drawers ─────────────────────────────────────────────────

type PatternDrawer = (g: Graphics, w: number, h: number, col: number, sp?: Record<string, unknown>) => void;

const SPECIES_PATTERNS: Record<string, PatternDrawer> = {
  'cil-eye': (g, w, h, col, sp) => {
    const cx = w / 2, cy = h / 2;
    const rOuter = (sp?.r_outer as number | undefined) ?? Math.min(w, h) * 0.35;
    const ringCount = (sp?.ring_count as number | undefined) ?? 3;
    const pupilRadius = (sp?.pupil_radius as number | undefined) ?? rOuter * 0.15;
    const rInnerRatio = (sp?.r_inner_ratio as number | undefined) ?? 0.3;
    for (let i = ringCount; i >= 1; i--) {
      g.circle(cx, cy, rOuter * (i / ringCount));
      g.fill({ color: col, alpha: 0.08 * i * (3 / ringCount) });
    }
    g.circle(cx, cy, pupilRadius);
    g.fill({ color: col, alpha: 0.5 });
    // draw iris line using r_inner_ratio
    g.circle(cx, cy, rOuter * rInnerRatio);
    g.stroke({ color: col, width: 0.8, alpha: 0.25 });
  },

  'cil-vector': (g, w, h, col, sp) => {
    const pad = 8;
    const arrowCount = (sp?.arrow_count as number | undefined) ?? 1;
    const arrowLength = (sp?.arrow_length as number | undefined) ?? (w - pad * 4);
    const angleSpread = (sp?.angle_spread as number | undefined) ?? 0;
    const cx = w / 2, cy = h / 2;
    const startX = cx - arrowLength / 2;
    for (let i = 0; i < arrowCount; i++) {
      const angle = arrowCount > 1 ? angleSpread * (i / (arrowCount - 1) - 0.5) : 0;
      const endX = startX + arrowLength * Math.cos(angle);
      const endY = cy + arrowLength * Math.sin(angle);
      g.moveTo(startX, cy);
      g.lineTo(endX, endY);
      g.moveTo(endX, endY);
      g.lineTo(endX - 6 * Math.cos(angle - 0.4), endY - 6 * Math.sin(angle - 0.4));
      g.moveTo(endX, endY);
      g.lineTo(endX - 6 * Math.cos(angle + 0.4), endY - 6 * Math.sin(angle + 0.4));
      g.stroke({ color: col, width: 1.5, alpha: 0.4 });
    }
  },

  'cil-bolt': (g, w, h, col, sp) => {
    const n = (sp?.zigzag_segments as number | undefined) ?? (sp?.num_rays as number | undefined) ?? 5;
    const dy = h / (n + 1), amp = w * 0.15;
    g.moveTo(w / 2, 6);
    for (let i = 1; i <= n; i++) {
      const x = w / 2 + (i % 2 === 1 ? amp : -amp);
      g.lineTo(x, dy * i + 6);
    }
    g.lineTo(w / 2, h - 6);
    g.stroke({ color: col, width: 1.5, alpha: 0.35 });
  },

  'cil-plus': (g, w, h, col, sp) => {
    const cx = w / 2, cy = h / 2;
    const arm = (sp?.arm_length as number | undefined) ?? Math.min(w, h) * 0.3;
    const strokeWidth = (sp?.stroke_width as number | undefined) ?? 2;
    const dashCorners = (sp?.dash_corners as boolean | undefined) ?? false;
    g.moveTo(cx - arm, cy); g.lineTo(cx + arm, cy);
    g.moveTo(cx, cy - arm); g.lineTo(cx, cy + arm);
    g.stroke({ color: col, width: strokeWidth, alpha: 0.3 });
    if (dashCorners) {
      const d = arm * 0.6;
      for (const [ox, oy] of [[-d,-d],[d,-d],[d,d],[-d,d]]) {
        g.circle(cx + ox, cy + oy, 1.5);
        g.fill({ color: col, alpha: 0.2 });
      }
    }
  },

  'cil-arrow-right': (g, w, h, col, sp) => {
    const cx = w / 2, cy = h / 2, sz = Math.min(w, h) * 0.25;
    g.moveTo(cx - sz, cy - sz);
    g.lineTo(cx + sz * 0.5, cy);
    g.lineTo(cx - sz, cy + sz);
    g.stroke({ color: col, width: 2, alpha: 0.4 });
  },

  'cil-filter': (g, w, h, col, sp) => {
    const pad = 10, gw = (w - pad * 2) / 3, gh = (h - pad * 2) / 3;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        g.rect(pad + c * gw + 1, pad + r * gh + 1, gw - 2, gh - 2);
        g.stroke({ color: col, width: 0.8, alpha: 0.2 });
      }
    }
  },

  'cil-code': (g, w, h, col, sp) => {
    const bx = 12, by = 8;
    g.moveTo(bx, by); g.lineTo(bx - 4, h / 2); g.lineTo(bx, h - by);
    g.moveTo(w - bx, by); g.lineTo(w - bx + 4, h / 2); g.lineTo(w - bx, h - by);
    g.stroke({ color: col, width: 1.5, alpha: 0.3 });
  },

  'cil-layers': (g, w, h, col, sp) => {
    for (let i = 0; i < 3; i++) {
      const off = i * 4;
      g.roundRect(6 + off, 6 + off, w - 12 - off * 2, h - 12 - off * 2, 3);
      g.stroke({ color: col, width: 1, alpha: 0.15 + i * 0.1 });
    }
  },

  'cil-loop': (g, w, h, col, sp) => {
    const cx = w / 2, cy = h / 2, r = Math.min(w, h) * 0.3;
    g.arc(cx, cy, r, -Math.PI * 0.8, Math.PI * 0.5);
    g.stroke({ color: col, width: 1.5, alpha: 0.4 });
    const ax = cx + r * Math.cos(Math.PI * 0.5);
    const ay = cy + r * Math.sin(Math.PI * 0.5);
    g.moveTo(ax - 4, ay - 4); g.lineTo(ax, ay); g.lineTo(ax + 4, ay - 4);
    g.stroke({ color: col, width: 1.5, alpha: 0.4 });
  },

  'cil-graph': (g, w, h, col, sp) => {
    const pts = [[w*0.25, h*0.3], [w*0.6, h*0.25], [w*0.75, h*0.6], [w*0.35, h*0.7]];
    for (const [x, y] of pts) {
      g.circle(x, y, 3);
      g.fill({ color: col, alpha: 0.35 });
    }
    g.moveTo(pts[0][0], pts[0][1]); g.lineTo(pts[1][0], pts[1][1]);
    g.lineTo(pts[2][0], pts[2][1]); g.lineTo(pts[3][0], pts[3][1]);
    g.lineTo(pts[0][0], pts[0][1]);
    g.stroke({ color: col, width: 1, alpha: 0.2 });
  },
};

// ── Bloom glow factory ──────────────────────────────────────────────────────

const SPECIES_BLOOM_SCALE: Record<string, number> = {
  'cil-eye':  2.0,
  'cil-bolt': 1.8,
  'cil-code': 0.8,
};

function createGlowSprite(w: number, h: number, glowColor: number, species: string): Graphics {
  const glow = new Graphics();
  const pad = 20;
  glow.roundRect(-pad, -pad, w + pad * 2, h + pad * 2, 12);
  glow.fill({ color: glowColor, alpha: 0.25 });
  const bloomScale = SPECIES_BLOOM_SCALE[species] ?? 1.5;
  const bloom = new AdvancedBloomFilter({ bloomScale, threshold: 0.5, blur: 8, quality: 4 });
  glow.filters = [bloom];
  return glow;
}

// ── Cell container builder ──────────────────────────────────────────────────

function buildCellContainer(desc: CellDescriptor): Container {
  const { bbox, species, label, z } = desc;
  const { w, h } = bbox;

  // ── Resolve params.json overrides ──────────────────────────────────────
  const p = desc.params;
  const cols = getColours(species, p?.fill_color, p?.stroke_color);
  const speciesParams = p?.species_params;
  const crowdingOpacity = (p?.opacity !== undefined) ? Math.max(0.15, Math.min(1, p.opacity)) : 1;
  const fontSize = p?.font_size ?? 11;

  const container = new Container();
  container.position.set(bbox.x, bbox.y);
  container.zIndex = z;

  const glow = createGlowSprite(w, h, cols.glow, species);
  container.addChild(glow);

  const body = new Graphics();
  body.roundRect(0, 0, w, h, 8);
  body.fill({ color: cols.fill, alpha: 0.9 });
  body.roundRect(0, 0, w, h, 8);
  body.stroke({ color: cols.stroke, width: 1.5, alpha: 0.8 });

  // ── DropShadowFilter from params.json shadow field ──────────────────────
  if (p?.shadow) {
    const { dx, dy, blur, opacity } = p.shadow;
    const dropShadow = new DropShadowFilter({
      offset: { x: dx, y: dy },
      blur,
      alpha: opacity,
      color: 0x000000,
      quality: 4,
    });
    // Compose with bloom on the body: [dropShadow, ...existing]
    body.filters = [dropShadow];
  }

  container.addChild(body);

  const pattern = new Graphics();
  const drawer = SPECIES_PATTERNS[species] ?? SPECIES_PATTERNS['cil-code'];
  drawer(pattern, w, h, cols.stroke, speciesParams);
  container.addChild(pattern);

  const style = new TextStyle({
    fontFamily: 'Inter, system-ui, sans-serif',
    fontSize,
    fill: 0xFFFFFF,
    fontWeight: '500',
  });
  const txt = new Text({ text: label, style });
  txt.anchor.set(0.5);
  txt.position.set(w / 2, h / 2);
  container.addChild(txt);

  // ── crowding_opacity: applies after fade-in completes ──────────────────
  // Store as userData so the poll loop can respect it when fading in
  (container as any).__targetAlpha = crowdingOpacity;
  container.alpha = 0; // caller sets to 0 for fade-in; will stop at __targetAlpha

  return container;
}

// ── Edge renderer ───────────────────────────────────────────────────────────

function drawEdges(
  g: Graphics,
  edges: EdgeDescriptor[],
  cellMap: Map<string, { x: number; y: number; w: number; h: number }>,
): void {
  g.clear();
  for (const edge of edges) {
    const src = cellMap.get(edge.source);
    const tgt = cellMap.get(edge.target);
    if (!src || !tgt) continue;

    const sx = src.x + src.w / 2;
    const sy = src.y + src.h;
    const tx = tgt.x + tgt.w / 2;
    const ty = tgt.y;

    if (edge.type === 'skip_connection') {
      const cx = Math.max(sx, tx) + 80;
      g.moveTo(sx, sy);
      g.bezierCurveTo(cx, sy, cx, ty, tx, ty);
      g.stroke({ color: 0x4CAF50, width: 2, alpha: 0.6 });
    } else {
      const mid_y = (sy + ty) / 2;
      g.moveTo(sx, sy);
      g.bezierCurveTo(sx, mid_y, tx, mid_y, tx, ty);
      g.stroke({ color: 0x999999, width: 1.5, alpha: 0.5 });
    }

    const angle = Math.atan2(ty - sy, tx - sx);
    const arrLen = 8;
    g.moveTo(tx - arrLen * Math.cos(angle - 0.4), ty - arrLen * Math.sin(angle - 0.4));
    g.lineTo(tx, ty);
    g.lineTo(tx - arrLen * Math.cos(angle + 0.4), ty - arrLen * Math.sin(angle + 0.4));
    g.stroke({ color: 0x999999, width: 1.5, alpha: 0.5 });
  }
}

// ── EdgeParticleSystem — WebGL2 Transform Feedback GPU 粒子流 ──────────────
//
// 架构：双缓冲 Ping-Pong Transform Feedback（参考 upstream/webgl2-particles-2）
//   - Simulation Pass: RASTERIZER_DISCARD + transformFeedback，更新粒子沿 bezier 路径的 t 值
//   - Render Pass: gl.POINTS，gl_PointSize 动态，fragment 输出发光点精灵
//   - 每条 edge 最多 PARTICLES_PER_EDGE 颗粒子，全部驻留 GPU 显存
//
// 粒子属性（per vertex，packed 进单 vec4）：
//   xyz = bezier 参数化坐标 (x, y, t)   w = lifetime [0..1]
//
// Simulation Shader:
//   - 沿 bezier 曲线步进 t += speed（速度略随机）
//   - t > 1.0 时粒子在起点复位，随机偏移以避免同步
//
// Render Shader:
//   - 坐标从 NDC 映射：position = (x/W*2-1, -(y/H*2-1))
//   - gl_PointSize 按 lifetime 曲线缩放（中段最大，两端收缩）
//   - fragment: 圆形点精灵 + 径向渐变发光

export class EdgeParticleSystem {
  private gl: WebGL2RenderingContext;
  private simProgram: WebGLProgram;
  private renderProgram: WebGLProgram;
  private tf: WebGLTransformFeedback;

  // Ping-Pong buffers — each holds all particle data as vec4 arrays
  private bufA: WebGLBuffer;
  private bufB: WebGLBuffer;
  private vaoA: WebGLVertexArrayObject;  // reads A, writes B
  private vaoB: WebGLVertexArrayObject;  // reads B, writes A
  private currentPing = 0; // 0 = read A write B, 1 = read B write A

  private particleCount = 0;
  private canvasW: number;
  private canvasH: number;

  // Per-edge bezier control points (uploaded as uniform arrays)
  private edgeUniforms: Float32Array; // [sx, sy, cx1, cy1, cx2, cy2, tx, ty] * MAX_EDGES
  private edgeCount = 0;

  static readonly PARTICLES_PER_EDGE = 32;
  static readonly MAX_EDGES          = 128;
  private static readonly VERT_FLOATS = 4; // vec4: x, y, t, lifetime

  // ── Shader sources ────────────────────────────────────────────────────────

  private static SIM_VERT = /* glsl */`#version 300 es
precision highp float;

// Input state (ping)
in vec4 a_particle; // x, y, t, lifetime

// Bezier edge data: packed as pairs of vec4
// [p0.xy, cp1.xy] and [cp2.xy, p1.xy] for each edge
uniform vec4 u_edges_a[${EdgeParticleSystem.MAX_EDGES}]; // p0.xy, cp1.xy
uniform vec4 u_edges_b[${EdgeParticleSystem.MAX_EDGES}]; // cp2.xy, p1.xy
uniform float u_speed_base;
uniform float u_time;

// Output state (pong) — captured by Transform Feedback
out vec4 v_particle;

// Simple LCG hash for per-particle pseudo-random speed jitter
float lcg(float seed) {
  return fract(sin(seed * 127.1 + 311.7) * 43758.5453);
}

// Cubic bezier position
vec2 bezier(vec2 p0, vec2 cp1, vec2 cp2, vec2 p1, float tt) {
  float u  = 1.0 - tt;
  float u2 = u * u;
  float u3 = u2 * u;
  float t2 = tt * tt;
  float t3 = t2 * tt;
  return u3*p0 + 3.0*u2*tt*cp1 + 3.0*u*t2*cp2 + t3*p1;
}

void main() {
  float px        = a_particle.x;
  float py        = a_particle.y;
  float t         = a_particle.z;
  float lifetime  = a_particle.w;

  // Each particle belongs to edge index = gl_VertexID / PARTICLES_PER_EDGE
  int edgeIdx = gl_VertexID / ${EdgeParticleSystem.PARTICLES_PER_EDGE};
  float particleSlot = float(gl_VertexID % ${EdgeParticleSystem.PARTICLES_PER_EDGE});

  vec4 ea = u_edges_a[edgeIdx]; // p0.xy, cp1.xy
  vec4 eb = u_edges_b[edgeIdx]; // cp2.xy, p1.xy
  vec2 p0  = ea.xy;
  vec2 cp1 = ea.zw;
  vec2 cp2 = eb.xy;
  vec2 p1  = eb.zw;

  // Per-particle speed jitter using LCG seeded by slot
  float jitter = 0.5 + lcg(particleSlot + float(edgeIdx) * 31.0) * 0.5;
  float speed  = u_speed_base * jitter;

  t += speed;

  // Respawn at staggered offsets so particles don't all bunch at start
  if (t > 1.0) {
    float phase = lcg(particleSlot * 17.0 + u_time * 0.001 + float(edgeIdx));
    t = phase * 0.35; // respawn in first third to maintain visual density
    lifetime = 0.0;
  } else {
    lifetime = min(1.0, lifetime + 0.04);
  }

  vec2 pos = bezier(p0, cp1, cp2, p1, clamp(t, 0.0, 1.0));

  v_particle = vec4(pos.x, pos.y, t, lifetime);
}
`;

  private static SIM_FRAG = /* glsl */`#version 300 es
precision highp float;
void main() {}  // RASTERIZER_DISCARD — fragment never executes
`;

  private static RENDER_VERT = /* glsl */`#version 300 es
precision highp float;

in vec4 a_particle; // x, y, t, lifetime

uniform vec2 u_resolution; // canvas W, H
uniform float u_point_size_max;
uniform vec4 u_color; // r, g, b, a base tint

out float v_lifetime;
out vec4  v_color;

void main() {
  float px = a_particle.x;
  float py = a_particle.y;
  float lifetime = a_particle.w;

  // Screen → NDC
  vec2 ndc = vec2(
    px / u_resolution.x * 2.0 - 1.0,
    -(py / u_resolution.y * 2.0 - 1.0)
  );
  gl_Position = vec4(ndc, 0.0, 1.0);

  // Size bell curve: peaks at lifetime ~ 0.5
  float bell = lifetime * (1.0 - lifetime) * 4.0;
  gl_PointSize = max(1.5, u_point_size_max * bell);

  v_lifetime = lifetime;
  v_color    = u_color;
}
`;

  private static RENDER_FRAG = /* glsl */`#version 300 es
precision highp float;

in float v_lifetime;
in vec4  v_color;
out vec4 fragColor;

void main() {
  // gl_PointCoord: [0,1]^2, center at (0.5, 0.5)
  vec2 uv   = gl_PointCoord - 0.5;
  float d   = dot(uv, uv); // squared distance from center
  if (d > 0.25) discard;   // clip to circle (r=0.5)

  // Radial glow: bright core, soft halo
  float core = 1.0 - smoothstep(0.0, 0.08, d);
  float halo = 1.0 - smoothstep(0.08, 0.25, d);
  float glow = core * 0.9 + halo * 0.3;

  fragColor = vec4(v_color.rgb * glow, v_color.a * glow * v_lifetime);
}
`;

  // ── Constructor ────────────────────────────────────────────────────────────

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', {
      alpha: true,
      premultipliedAlpha: false,
      antialias: false,
    });
    if (!gl) throw new Error('[EdgeParticleSystem] WebGL2 not available');
    this.gl = gl;
    this.canvasW = canvas.width;
    this.canvasH = canvas.height;

    // ── Compile simulation program (with transform feedback) ──────────────
    this.simProgram = this._buildSimProgram();

    // ── Compile render program ─────────────────────────────────────────────
    this.renderProgram = this._compileProgram(
      EdgeParticleSystem.RENDER_VERT,
      EdgeParticleSystem.RENDER_FRAG,
    );

    // ── Transform Feedback object ──────────────────────────────────────────
    this.tf = gl.createTransformFeedback()!;

    // ── Allocate max ping-pong buffers (MAX_EDGES * PARTICLES_PER_EDGE * 4 floats) ──
    const maxParticles = EdgeParticleSystem.MAX_EDGES * EdgeParticleSystem.PARTICLES_PER_EDGE;
    const bytes = maxParticles * EdgeParticleSystem.VERT_FLOATS * 4;
    this.bufA = this._allocBuf(bytes);
    this.bufB = this._allocBuf(bytes);
    this.vaoA = this._makeVAO(this.bufA);
    this.vaoB = this._makeVAO(this.bufB);

    // ── Edge uniform scratch ───────────────────────────────────────────────
    this.edgeUniforms = new Float32Array(EdgeParticleSystem.MAX_EDGES * 8);
  }

  // ── Upload edge bezier data and seed particles ─────────────────────────────

  setEdges(
    edges: EdgeDescriptor[],
    cellMap: Map<string, { x: number; y: number; w: number; h: number }>,
  ): void {
    const { PARTICLES_PER_EDGE, MAX_EDGES, VERT_FLOATS } = EdgeParticleSystem;
    const validEdges: Array<{
      sx: number; sy: number;
      cx1: number; cy1: number;
      cx2: number; cy2: number;
      tx: number; ty: number;
      color: [number, number, number];
    }> = [];

    for (const edge of edges) {
      if (validEdges.length >= MAX_EDGES) break;
      const src = cellMap.get(edge.source);
      const tgt = cellMap.get(edge.target);
      if (!src || !tgt) continue;

      const sx  = src.x + src.w / 2;
      const sy  = src.y + src.h;
      const tx  = tgt.x + tgt.w / 2;
      const ty  = tgt.y;

      let cx1: number, cy1: number, cx2: number, cy2: number;
      if (edge.type === 'skip_connection') {
        const cx = Math.max(sx, tx) + 80;
        cx1 = cx; cy1 = sy;
        cx2 = cx; cy2 = ty;
      } else {
        const mid_y = (sy + ty) / 2;
        cx1 = sx; cy1 = mid_y;
        cx2 = tx; cy2 = mid_y;
      }

      const color: [number, number, number] = edge.type === 'skip_connection'
        ? [0.3, 0.85, 0.4]  // green tint for skip connections
        : [0.5, 0.75, 1.0]; // blue-white for normal edges

      validEdges.push({ sx, sy, cx1, cy1, cx2, cy2, tx, ty, color });
    }

    this.edgeCount    = validEdges.length;
    this.particleCount = this.edgeCount * PARTICLES_PER_EDGE;

    if (this.particleCount === 0) return;

    // Build packed u_edges_a / u_edges_b uniform data
    for (let i = 0; i < validEdges.length; i++) {
      const e = validEdges[i];
      const base = i * 8;
      this.edgeUniforms[base + 0] = e.sx;
      this.edgeUniforms[base + 1] = e.sy;
      this.edgeUniforms[base + 2] = e.cx1;
      this.edgeUniforms[base + 3] = e.cy1;
      this.edgeUniforms[base + 4] = e.cx2;
      this.edgeUniforms[base + 5] = e.cy2;
      this.edgeUniforms[base + 6] = e.tx;
      this.edgeUniforms[base + 7] = e.ty;
    }

    // Seed initial particle state with staggered t values
    const seed = new Float32Array(this.particleCount * VERT_FLOATS);
    for (let e = 0; e < validEdges.length; e++) {
      const ev = validEdges[e];
      for (let p = 0; p < PARTICLES_PER_EDGE; p++) {
        const idx  = (e * PARTICLES_PER_EDGE + p) * VERT_FLOATS;
        const t    = p / PARTICLES_PER_EDGE;  // evenly staggered
        // Evaluate bezier at initial t to get starting position
        const pos  = this._evalBezier(ev.sx, ev.sy, ev.cx1, ev.cy1, ev.cx2, ev.cy2, ev.tx, ev.ty, t);
        seed[idx + 0] = pos[0];
        seed[idx + 1] = pos[1];
        seed[idx + 2] = t;
        seed[idx + 3] = t; // lifetime starts at same phase
      }
    }

    // Upload to both buffers so first ping-pong has valid data
    const gl = this.gl;
    const byteLen = this.particleCount * VERT_FLOATS * 4;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufA);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, seed.subarray(0, this.particleCount * VERT_FLOATS));
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufB);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, seed.subarray(0, this.particleCount * VERT_FLOATS));
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Store per-edge colors for render pass (simple: use first edge color for all, or cycle)
    this._edgeColors = validEdges.map(e => e.color);
  }

  private _edgeColors: Array<[number, number, number]> = [];

  // ── Tick: one simulation step + one render pass ────────────────────────────

  tick(time: number): void {
    if (this.particleCount === 0) return;
    const gl = this.gl;

    // WebGL2 blending for additive glow
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive — brightens on overlap

    this._simPass(time);
    this._renderPass();

    gl.disable(gl.BLEND);
  }

  // ── Simulation pass (Transform Feedback) ──────────────────────────────────

  private _simPass(time: number): void {
    const gl = this.gl;
    const { PARTICLES_PER_EDGE, MAX_EDGES, VERT_FLOATS } = EdgeParticleSystem;

    gl.useProgram(this.simProgram);

    // Upload edge bezier data
    const locA = gl.getUniformLocation(this.simProgram, 'u_edges_a[0]');
    const locB = gl.getUniformLocation(this.simProgram, 'u_edges_b[0]');
    // Pack into two separate Float32Arrays (vec4 arrays)
    const ua = new Float32Array(MAX_EDGES * 4);
    const ub = new Float32Array(MAX_EDGES * 4);
    for (let i = 0; i < this.edgeCount; i++) {
      const base = i * 8;
      ua[i*4+0] = this.edgeUniforms[base+0]; // p0.x
      ua[i*4+1] = this.edgeUniforms[base+1]; // p0.y
      ua[i*4+2] = this.edgeUniforms[base+2]; // cp1.x
      ua[i*4+3] = this.edgeUniforms[base+3]; // cp1.y
      ub[i*4+0] = this.edgeUniforms[base+4]; // cp2.x
      ub[i*4+1] = this.edgeUniforms[base+5]; // cp2.y
      ub[i*4+2] = this.edgeUniforms[base+6]; // p1.x
      ub[i*4+3] = this.edgeUniforms[base+7]; // p1.y
    }
    gl.uniform4fv(locA, ua);
    gl.uniform4fv(locB, ub);
    gl.uniform1f(gl.getUniformLocation(this.simProgram, 'u_speed_base'), 0.004);
    gl.uniform1f(gl.getUniformLocation(this.simProgram, 'u_time'), time);

    // Bind read VAO and write buffer for TF
    const [readVAO, writeBuf] = this.currentPing === 0
      ? [this.vaoA, this.bufB]
      : [this.vaoB, this.bufA];

    gl.bindVertexArray(readVAO);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, writeBuf);

    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, this.particleCount);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindVertexArray(null);

    // Flip ping-pong
    this.currentPing ^= 1;
  }

  // ── Render pass (gl.POINTS sprite) ────────────────────────────────────────

  private _renderPass(): void {
    const gl = this.gl;
    const { PARTICLES_PER_EDGE } = EdgeParticleSystem;

    gl.useProgram(this.renderProgram);
    gl.uniform2f(gl.getUniformLocation(this.renderProgram, 'u_resolution'), this.canvasW, this.canvasH);
    gl.uniform1f(gl.getUniformLocation(this.renderProgram, 'u_point_size_max'), 7.0);

    // Read VAO is now the pong side (we just wrote to it in sim pass)
    const readVAO = this.currentPing === 0 ? this.vaoA : this.vaoB;
    gl.bindVertexArray(readVAO);

    const colorLoc = gl.getUniformLocation(this.renderProgram, 'u_color');

    // Draw each edge's particle band with its own color
    for (let e = 0; e < this.edgeCount; e++) {
      const col = this._edgeColors[e] ?? [0.6, 0.8, 1.0];
      gl.uniform4f(colorLoc, col[0], col[1], col[2], 0.85);
      const offset = e * PARTICLES_PER_EDGE;
      gl.drawArrays(gl.POINTS, offset, PARTICLES_PER_EDGE);
    }

    gl.bindVertexArray(null);
  }

  // ── destroy ────────────────────────────────────────────────────────────────

  destroy(): void {
    const gl = this.gl;
    gl.deleteBuffer(this.bufA);
    gl.deleteBuffer(this.bufB);
    gl.deleteVertexArray(this.vaoA);
    gl.deleteVertexArray(this.vaoB);
    gl.deleteTransformFeedback(this.tf);
    gl.deleteProgram(this.simProgram);
    gl.deleteProgram(this.renderProgram);
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  private _evalBezier(
    sx: number, sy: number,
    cx1: number, cy1: number,
    cx2: number, cy2: number,
    tx: number, ty: number,
    t: number,
  ): [number, number] {
    const u  = 1 - t;
    const u2 = u * u;
    const u3 = u2 * u;
    const t2 = t * t;
    const t3 = t2 * t;
    return [
      u3*sx + 3*u2*t*cx1 + 3*u*t2*cx2 + t3*tx,
      u3*sy + 3*u2*t*cy1 + 3*u*t2*cy2 + t3*ty,
    ];
  }

  private _allocBuf(bytes: number): WebGLBuffer {
    const gl = this.gl;
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, bytes, gl.DYNAMIC_COPY);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return buf;
  }

  private _makeVAO(buf: WebGLBuffer): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    // attrib 0: a_particle (vec4)
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return vao;
  }

  private _buildSimProgram(): WebGLProgram {
    const gl = this.gl;

    const vs = this._compileShader(gl.VERTEX_SHADER, EdgeParticleSystem.SIM_VERT);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, EdgeParticleSystem.SIM_FRAG);

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);

    // Declare Transform Feedback varyings BEFORE linking
    gl.transformFeedbackVaryings(prog, ['v_particle'], gl.SEPARATE_ATTRIBS);

    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('[EdgeParticleSystem] Sim program link error: ' + gl.getProgramInfoLog(prog));
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  private _compileProgram(vertSrc: string, fragSrc: string): WebGLProgram {
    const gl = this.gl;
    const vs = this._compileShader(gl.VERTEX_SHADER, vertSrc);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, fragSrc);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('[EdgeParticleSystem] Render program link error: ' + gl.getProgramInfoLog(prog));
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  private _compileShader(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error('[EdgeParticleSystem] Shader compile error: ' + gl.getShaderInfoLog(sh));
    }
    return sh;
  }
}

// ── Live cell state (used by poll loop) ────────────────────────────────────

interface LiveCell {
  desc: CellDescriptor;
  /** current rendered position (lerp target) */
  curX: number;
  curY: number;
  /** target position from latest poll */
  tgtX: number;
  tgtY: number;
  container: Container;
  /** fade direction: +1 = fading in, -1 = fading out, 0 = stable */
  fadeDir: 0 | 1 | -1;
}

const LERP_FACTOR  = 0.1;   // position lerp per frame
const FADE_SPEED   = 0.05;  // alpha change per frame

// ── pollCellChannels ────────────────────────────────────────────────────────

/**
 * pollCellChannels — starts a 500ms polling loop against /api/cells.
 *
 * Behaviour:
 *   1. Every 500ms fetch /api/cells → CellDescriptor[]
 *   2. New cells: spawn container at alpha=0, fade in to 1
 *   3. Removed cells: fade out to 0, then destroy
 *   4. Existing cells: lerp position toward new bbox (alpha += (target-current)*0.1)
 *   5. Edge layer redraws every frame based on current live positions
 *
 * @param app        Running PixiJS Application
 * @param edges      EdgeDescriptor[] (static topology — edges don't change)
 * @param edgeLayer  Graphics node dedicated to edge drawing
 * @returns          stop() to cancel polling + animation
 */
export function pollCellChannels(
  app: Application,
  edges: EdgeDescriptor[],
  edgeLayer: Graphics,
  particleSystem?: EdgeParticleSystem | null,
): () => void {
  // Map of live cells keyed by cell_id
  const live = new Map<string, LiveCell>();

  let pollHandle: ReturnType<typeof setInterval> | null = null;
  let stopped = false;

  // ── Per-frame tick: lerp positions + fade + redraw edges ───────────────
  function tick(_ticker: Ticker): void {
    if (stopped) return;

    // Build bbox snapshot for edge drawing
    const bboxSnap = new Map<string, { x: number; y: number; w: number; h: number }>();

    for (const [id, lc] of live) {
      // Lerp position
      lc.curX += (lc.tgtX - lc.curX) * LERP_FACTOR;
      lc.curY += (lc.tgtY - lc.curY) * LERP_FACTOR;
      lc.container.position.set(lc.curX, lc.curY);

      // Fade
      const targetAlpha: number = (lc.container as any).__targetAlpha ?? 1;
      if (lc.fadeDir === 1) {
        lc.container.alpha = Math.min(targetAlpha, lc.container.alpha + FADE_SPEED);
        if (lc.container.alpha >= targetAlpha) lc.fadeDir = 0;
      } else if (lc.fadeDir === -1) {
        lc.container.alpha = Math.max(0, lc.container.alpha - FADE_SPEED);
        if (lc.container.alpha <= 0) {
          // Fully faded — destroy and remove
          app.stage.removeChild(lc.container);
          lc.container.destroy({ children: true });
          live.delete(id);
          continue;
        }
      }

      // Record current rendered bbox for edge drawing
      const { w, h } = lc.desc.bbox;
      bboxSnap.set(id, { x: lc.curX, y: lc.curY, w, h });
    }

    // Redraw edges at current (lerped) positions
    drawEdges(edgeLayer, edges, bboxSnap);

    // Sync particle paths to current cell positions (every frame — lerp smoothness)
    if (particleSystem && bboxSnap.size > 0) {
      particleSystem.setEdges(edges, bboxSnap);
    }
  }

  app.ticker.add(tick);

  // ── Poll loop: fetch /api/cells every 500ms ────────────────────────────
  async function fetchAndReconcile(): Promise<void> {
    if (stopped) return;
    try {
      const res = await fetch('/api/cells');
      if (!res.ok) return;
      const incoming: CellDescriptor[] = await res.json();

      const seen = new Set<string>();

      for (const desc of incoming) {
        seen.add(desc.cell_id);
        const tgtX = desc.bbox.x;
        const tgtY = desc.bbox.y;

        if (live.has(desc.cell_id)) {
          // Existing cell — update target position
          const lc = live.get(desc.cell_id)!;
          lc.tgtX = tgtX;
          lc.tgtY = tgtY;
          // Also update desc so edge dimensions stay correct
          lc.desc = desc;
          // Cancel any ongoing fade-out if cell reappears
          if (lc.fadeDir === -1) lc.fadeDir = 1;
        } else {
          // New cell — spawn at target, alpha=0, fade in
          const container = buildCellContainer(desc);
          container.alpha = 0;
          app.stage.addChild(container);

          const lc: LiveCell = {
            desc,
            curX: tgtX,
            curY: tgtY,
            tgtX,
            tgtY,
            container,
            fadeDir: 1,
          };
          live.set(desc.cell_id, lc);
        }
      }

      // Cells in live but NOT in incoming → fade out
      for (const [id, lc] of live) {
        if (!seen.has(id) && lc.fadeDir !== -1) {
          lc.fadeDir = -1;
        }
      }
    } catch (err) {
      console.warn('[pollCellChannels] fetch error:', err);
    }
  }

  // Initial fetch, then schedule
  fetchAndReconcile();
  pollHandle = setInterval(fetchAndReconcile, 500);

  // ── Return stop handle ─────────────────────────────────────────────────
  return () => {
    stopped = true;
    if (pollHandle !== null) clearInterval(pollHandle);
    app.ticker.remove(tick);
    if (particleSystem) particleSystem.destroy();
  };
}

// ── Main renderer (static, one-shot) ───────────────────────────────────────

export async function renderCellGraph(
  canvas: HTMLCanvasElement,
  cells: CellDescriptor[],
  edges: EdgeDescriptor[],
): Promise<Application> {
  const app = new Application();
  await app.init({
    canvas,
    width: canvas.width,
    height: canvas.height,
    backgroundColor: 0x1A1A2E,
    antialias: true,
    resolution: window.devicePixelRatio || 1,
    autoDensity: true,
  });

  app.stage.sortableChildren = true;

  const cellMap = new Map<string, CellDescriptor>();
  for (const c of cells) cellMap.set(c.cell_id, c);

  // Build bbox map for edge drawing
  const bboxMap = new Map<string, { x: number; y: number; w: number; h: number }>();
  for (const c of cells) bboxMap.set(c.cell_id, c.bbox);

  // Draw edges first (behind cells)
  const edgeLayer = new Graphics();
  edgeLayer.zIndex = 0;
  drawEdges(edgeLayer, edges, bboxMap);
  app.stage.addChild(edgeLayer);

  // Draw cells
  for (const cell of cells) {
    const container = buildCellContainer(cell);
    // In static mode: skip fade-in, go straight to target alpha
    container.alpha = (container as any).__targetAlpha ?? 1;
    app.stage.addChild(container);
  }

  // ── GPU edge particle system (WebGL2 Transform Feedback) ───────────────
  try {
    const particleSystem = new EdgeParticleSystem(canvas);
    particleSystem.setEdges(edges, bboxMap);
    const t0 = performance.now();
    app.ticker.add(() => particleSystem.tick(performance.now() - t0));
    (app as any).__edgeParticleSystem = particleSystem;
  } catch (err) {
    // WebGL2 not available — silently degrade to static edges only
    console.warn('[renderCellGraph] EdgeParticleSystem unavailable:', err);
  }

  return app;
}

// ── Live poll renderer (uses pollCellChannels) ──────────────────────────────

/**
 * renderCellGraphLive — initialise a PixiJS canvas in live-poll mode.
 *
 * No initial cells are rendered; the poll loop populates the stage.
 * Returns both the Application and a stop() handle.
 */
export async function renderCellGraphLive(
  canvas: HTMLCanvasElement,
  edges: EdgeDescriptor[],
): Promise<{ app: Application; stop: () => void }> {
  const app = new Application();
  await app.init({
    canvas,
    width: canvas.width,
    height: canvas.height,
    backgroundColor: 0x1A1A2E,
    antialias: true,
    resolution: window.devicePixelRatio || 1,
    autoDensity: true,
  });

  app.stage.sortableChildren = true;

  const edgeLayer = new Graphics();
  edgeLayer.zIndex = 0;
  app.stage.addChild(edgeLayer);

  // ── GPU edge particle system (WebGL2 Transform Feedback) ───────────────
  let particleSystem: EdgeParticleSystem | null = null;
  try {
    particleSystem = new EdgeParticleSystem(canvas);
    const t0 = performance.now();
    app.ticker.add(() => particleSystem!.tick(performance.now() - t0));
  } catch (err) {
    console.warn('[renderCellGraphLive] EdgeParticleSystem unavailable:', err);
  }

  const stop = pollCellChannels(app, edges, edgeLayer, particleSystem);

  return { app, stop };
}

// ── params.json loader helper ───────────────────────────────────────────────

/**
 * mergeParamsJson — fetches a params.json file for a given cell_id and
 * merges it into a CellDescriptor so that fill_color, stroke_color, shadow,
 * opacity, and species_params are all available to buildCellContainer.
 *
 * Usage:
 *   const cell = await mergeParamsJson(desc, '/channels/cell/self_attn/params.json');
 */
export async function mergeParamsJson(
  desc: CellDescriptor,
  paramsUrl: string,
): Promise<CellDescriptor> {
  try {
    const res = await fetch(paramsUrl);
    if (!res.ok) return desc;
    const p: ParamsJson = await res.json();
    return { ...desc, params: p };
  } catch {
    return desc;
  }
}
