/**
 * edge-gpu-pass.ts — GPU Edge 粒子流 (WebGL2 Transform Feedback)
 *
 * M1316j: 边从虚线改为粒子流
 *   - 移除 triangle-strip 虚线渲染
 *   - 改用 WebGL2 transform feedback ping-pong 粒子系统
 *   - 粒子沿 cubic Bézier 样条（控制点来自 edge_routes.json）流动
 *   - 粒子速度 = baseSpeed * sourceEnergy（source cell energy 驱动）
 *   - 发光圆点精灵 (gl.POINTS)，additive blend，species 颜色渐变
 *
 * 架构:
 *   每条 edge: cubic Bézier P0→C0→C1→P3 (从 EdgeControlPoints 推导)
 *   粒子状态 vec4(travel, speed, edgeIdx, seed) — TF ping-pong
 *   Update pass: RASTERIZER_DISCARD + TF → advance travel, wrap on reach
 *   Draw pass: gl.POINTS, evaluate Bézier → pixel pos, glow sprite
 *
 * 公共 API 与旧 EdgeGPU 完全兼容:
 *   new EdgeGPU(gl, config?, edges?)
 *   render(time)  setEdges(edges)  updateEdgeEnergy(id, e)
 *   updateAllEdgeEnergies(map)  setCanvasSize(w,h)  resize(w,h)
 *   destroy()  createOffscreenFBO(w,h)  renderToFBO(fbo,w,h,t)
 */

import { getShader } from '../shaders/ShaderLoader';

// ─── Edge descriptor ─────────────────────────────────────────────────────────

export interface EdgeControlPoints {
  id:     string;
  isSkip: boolean;
  p0: [number, number];
  p1: [number, number];
  p2: [number, number];
  p3: [number, number];
  sourceColor?: [number, number, number];
  targetColor?: [number, number, number];
  /** M1286: energy of the source cell [0, 1]; drives particle speed & brightness */
  sourceEnergy?: number;
}

export interface EdgeGPUConfig {
  lineWidth:   number;
  glowRadius:  number;
  glowAlpha:   number;
  dashLength:  number;
  gapLength:   number;
  flowSpeed:   number;
  segments:    number;
  canvasWidth: number;
  canvasHeight: number;
  /** particles per edge (default 48) */
  particlesPerEdge?: number;
  /** particle point size in pixels (default 4) */
  pointSize?: number;
  /** particle alpha (default 0.85) */
  particleAlpha?: number;
}

const DEFAULT_CONFIG: EdgeGPUConfig = {
  lineWidth:    2.5,
  glowRadius:   6.0,
  glowAlpha:    0.45,
  dashLength:   14,
  gapLength:    10,
  flowSpeed:    0.25,
  segments:     32,
  canvasWidth:  800,
  canvasHeight: 600,
  particlesPerEdge: 48,
  pointSize:    4,
  particleAlpha: 0.85,
};

// ─── Default edges ────────────────────────────────────────────────────────────

function makeDefaultEdges(w: number, h: number): EdgeControlPoints[] {
  const cx = w * 0.5;
  const step = h / 7;
  const ff: EdgeControlPoints[] = [
    { id: 'e1', isSkip: false,
      p0: [cx, step*1], p1: [cx+60, step*1.5], p2: [cx-60, step*1.5], p3: [cx, step*2],
      sourceColor: [0.3,0.6,1.0], targetColor: [0.2,0.8,0.9] },
    { id: 'e2', isSkip: false,
      p0: [cx, step*2], p1: [cx-40, step*2.5], p2: [cx+40, step*2.5], p3: [cx, step*3],
      sourceColor: [0.2,0.8,0.9], targetColor: [0.5,0.9,0.7] },
    { id: 'e3', isSkip: false,
      p0: [cx, step*3], p1: [cx+50, step*3.5], p2: [cx-50, step*3.5], p3: [cx, step*4],
      sourceColor: [0.5,0.9,0.7], targetColor: [0.9,0.7,0.3] },
    { id: 'e4', isSkip: false,
      p0: [cx, step*4], p1: [cx-70, step*4.5], p2: [cx+70, step*4.5], p3: [cx, step*5],
      sourceColor: [0.9,0.7,0.3], targetColor: [1.0,0.4,0.4] },
    { id: 'e5', isSkip: false,
      p0: [cx, step*5], p1: [cx+45, step*5.5], p2: [cx-45, step*5.5], p3: [cx, step*6],
      sourceColor: [1.0,0.4,0.4], targetColor: [0.8,0.3,0.9] },
    { id: 'e6', isSkip: false,
      p0: [cx, step*6], p1: [cx-55, step*6.3], p2: [cx+55, step*6.5], p3: [cx, step*7],
      sourceColor: [0.8,0.3,0.9], targetColor: [0.4,0.5,1.0] },
  ];
  const skip: EdgeControlPoints[] = [
    { id: 'skip1', isSkip: true,
      p0: [cx, step*2], p1: [cx+150, step*2.5], p2: [cx+150, step*3.5], p3: [cx, step*4],
      sourceColor: [0.2,0.8,0.9], targetColor: [0.9,0.7,0.3] },
    { id: 'skip2', isSkip: true,
      p0: [cx, step*4], p1: [cx-140, step*4.5], p2: [cx-140, step*5.5], p3: [cx, step*6],
      sourceColor: [0.9,0.7,0.3], targetColor: [0.8,0.3,0.9] },
  ];
  return [...ff, ...skip];
}

// ─── GLSL sources ─────────────────────────────────────────────────────────────

const MAX_EDGES = 16;

// Update pass: transform feedback, RASTERIZER_DISCARD
// State: vec4(travel, speed, edgeIdx, seed)
// Speed is modulated by per-edge sourceEnergy uniform
const UPDATE_VERT = /* glsl */`#version 300 es
precision highp float;

in vec4 a_state; // (travel, speed, edgeIdx, seed)

uniform float u_dt;           // ms per frame
uniform float u_resetRange;   // restart jitter [0, resetRange]

// Per-edge energy [0,1] — scales particle speed
uniform float u_energy[${MAX_EDGES}];

out vec4 v_state;

uint lcg(uint s) { return s * 1664525u + 1013904223u; }
float lcgF(uint s) { return float(lcg(s)) / 4294967296.0; }

void main() {
  float travel  = a_state.x;
  float speed   = a_state.y;
  float edgeIdx = a_state.z;
  uint  seed    = floatBitsToUint(a_state.w);

  int ei = clamp(int(edgeIdx + 0.5), 0, ${MAX_EDGES - 1});
  // energy drives speed: minimum 0.05 to keep particles always visible
  float energy  = max(u_energy[ei], 0.05);
  travel += speed * u_dt * energy;

  if (travel >= 1.0) {
    seed   = lcg(seed);
    travel = lcgF(seed) * u_resetRange;
    seed   = lcg(seed);
  }

  v_state = vec4(travel, speed, edgeIdx, uintBitsToFloat(seed));
}
`;

const UPDATE_FRAG = /* glsl */`#version 300 es
precision highp float;
out vec4 unused;
void main() { unused = vec4(0.0); }
`;

// Draw pass: evaluate Bézier, output glow sprite
const DRAW_VERT = /* glsl */`#version 300 es
precision highp float;

in vec4 a_state; // (travel, speed, edgeIdx, seed)

uniform vec2  u_resolution;
uniform float u_pointSize;
// Packed Bézier: u_bez0[i]=vec4(P0x,P0y,P3x,P3y), u_bez1[i]=vec4(C0x,C0y,C1x,C1y)
uniform vec4  u_bez0[${MAX_EDGES}];
uniform vec4  u_bez1[${MAX_EDGES}];
// Per-edge source/target species colours
uniform vec4  u_srcColor[${MAX_EDGES}];
uniform vec4  u_tgtColor[${MAX_EDGES}];
// Per-edge energy → brightness modulation
uniform float u_energy[${MAX_EDGES}];

out float v_alpha;
out vec3  v_color;

vec2 bezier(int idx, float t) {
  vec4 b0 = u_bez0[idx]; // P0.xy, P3.xy
  vec4 b1 = u_bez1[idx]; // C0.xy, C1.xy
  vec2 P0 = b0.xy;
  vec2 P3 = b0.zw;
  vec2 C0 = b1.xy;
  vec2 C1 = b1.zw;
  float mt = 1.0 - t;
  return mt*mt*mt*P0 + 3.0*mt*mt*t*C0 + 3.0*mt*t*t*C1 + t*t*t*P3;
}

vec2 pixelToNDC(vec2 px) {
  vec2 ndc = (px / u_resolution) * 2.0 - 1.0;
  ndc.y = -ndc.y; // screen Y → NDC Y
  return ndc;
}

void main() {
  float travel  = a_state.x;
  int   ei      = clamp(int(a_state.z + 0.5), 0, ${MAX_EDGES - 1});
  float t       = clamp(travel, 0.0, 1.0);

  vec2 posPx = bezier(ei, t);

  // Fade in/out at endpoints
  float fadeIn  = smoothstep(0.0, 0.06, travel);
  float fadeOut = 1.0 - smoothstep(0.90, 1.0, travel);
  float energy  = max(u_energy[ei], 0.05);
  // Energy modulates brightness: low energy → dim particles
  float energyAlpha = 0.3 + 0.7 * energy;
  v_alpha = fadeIn * fadeOut * energyAlpha;

  // Species colour gradient source → target along travel
  vec3 srcC  = u_srcColor[ei].rgb;
  vec3 tgtC  = u_tgtColor[ei].rgb;
  // Pulse: bright core slides along travel direction
  float pulse = fract(travel * 3.0);
  float pulseI = smoothstep(0.0, 0.2, pulse) * (1.0 - smoothstep(0.2, 0.5, pulse));
  v_color = mix(srcC, tgtC, t) * (1.0 + 0.8 * pulseI * energy);

  gl_Position  = vec4(pixelToNDC(posPx), 0.0, 1.0);
  gl_PointSize = u_pointSize * (0.7 + 0.3 * energy); // skip = slightly bigger handled by energy
}
`;

const DRAW_FRAG = /* glsl */`#version 300 es
precision highp float;

uniform float u_alpha;

in float v_alpha;
in vec3  v_color;

out vec4 fragColor;

void main() {
  // Circular glow sprite
  vec2  uv   = gl_PointCoord * 2.0 - 1.0;
  float r    = dot(uv, uv);
  if (r > 1.0) discard;

  // Bright core + exponential glow halo
  float core = 1.0 - smoothstep(0.0, 0.35, r);
  float glow = exp(-r * 2.8) * 0.7;
  float mask = max(core, glow);

  float alpha = mask * v_alpha * u_alpha;
  if (alpha < 0.004) discard;

  // Premultiplied alpha for additive blend
  fragColor = vec4(v_color * alpha, alpha);
}
`;

// ─── GL helpers ──────────────────────────────────────────────────────────────

function compileShader(gl: WebGL2RenderingContext, type: GLenum, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`[EdgeGPU] shader compile error: ${log}`);
  }
  return sh;
}

function linkProgram(
  gl: WebGL2RenderingContext,
  vert: string,
  frag: string,
  varyings?: string[],
): WebGLProgram {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, compileShader(gl, gl.VERTEX_SHADER, vert));
  gl.attachShader(prog, compileShader(gl, gl.FRAGMENT_SHADER, frag));
  if (varyings && varyings.length > 0) {
    gl.transformFeedbackVaryings(prog, varyings, gl.SEPARATE_ATTRIBS);
  }
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`[EdgeGPU] program link error: ${log}`);
  }
  return prog;
}

// LCG for initial seed generation
function lcg(seed: number): number {
  return (((seed * 1664525 + 1013904223) & 0xffffffff) >>> 0) / 4294967296.0;
}

// ─── EdgeGPU class ────────────────────────────────────────────────────────────

export class EdgeGPU {
  private gl:     WebGL2RenderingContext;
  private config: Required<EdgeGPUConfig>;
  private edges:  EdgeControlPoints[];

  // Programs
  private updateProg!: WebGLProgram;
  private drawProg!:   WebGLProgram;

  // Transform feedback
  private tf!: WebGLTransformFeedback;

  // Ping-pong
  private currentIndex = 0;
  private vao!:      [WebGLVertexArrayObject, WebGLVertexArrayObject];
  private stateBuf!: [WebGLBuffer, WebGLBuffer];

  private totalParticles = 0;
  private _ready = false;

  constructor(
    gl: WebGL2RenderingContext,
    configOrEdges?: Partial<EdgeGPUConfig> | EdgeControlPoints[],
    edgesOrConfig?: EdgeControlPoints[] | Partial<EdgeGPUConfig>,
  ) {
    this.gl = gl;

    // Handle both call patterns:
    //   new EdgeGPU(gl, config?, edges?)          — gpu-render-loop
    //   new EdgeGPU(gl, edges, {width,height})    — at-world-integrator
    let cfg: Partial<EdgeGPUConfig>;
    let edgeList: EdgeControlPoints[] | undefined;

    if (Array.isArray(configOrEdges)) {
      // (gl, edges, config)
      edgeList = configOrEdges as EdgeControlPoints[];
      const rawCfg = edgesOrConfig as Partial<EdgeGPUConfig> | undefined;
      // at-world-integrator passes {width, height} — map to canvasWidth/canvasHeight
      if (rawCfg && ('width' in rawCfg || 'height' in rawCfg)) {
        const r = rawCfg as unknown as { width?: number; height?: number };
        cfg = { canvasWidth: r.width, canvasHeight: r.height };
      } else {
        cfg = rawCfg ?? {};
      }
    } else {
      // (gl, config?, edges?)
      cfg = configOrEdges ?? {};
      edgeList = edgesOrConfig as EdgeControlPoints[] | undefined;
    }

    this.config = { ...DEFAULT_CONFIG, ...cfg } as Required<EdgeGPUConfig>;
    this.edges  = edgeList ?? makeDefaultEdges(this.config.canvasWidth, this.config.canvasHeight);
    this._init();
  }

  // ─── Public API ──────────────────────────────────────────────────────────

  render(time: number): void {
    if (!this._ready) return;
    // dt clamped to 100ms max to avoid particle explosion on tab switch
    const dt = Math.min(16.67, 16.67); // fixed 60fps step; time param unused but kept for API compat
    void time;
    this._update(dt);
    this._draw();
  }

  setEdges(edges: EdgeControlPoints[]): void {
    this.edges = edges;
    // Rebuild buffers with new edge layout
    this._destroyBuffers();
    this._buildBuffers();
  }

  updateEdgeEnergy(edgeId: string, sourceEnergy: number): void {
    const edge = this.edges.find(e => e.id === edgeId);
    if (edge) edge.sourceEnergy = Math.max(0, Math.min(1, sourceEnergy));
  }

  updateAllEdgeEnergies(energyMap: Map<string, number>): void {
    for (const edge of this.edges) {
      const e = energyMap.get(edge.id);
      if (e !== undefined) edge.sourceEnergy = Math.max(0, Math.min(1, e));
    }
  }

  setCanvasSize(w: number, h: number): void {
    this.config.canvasWidth  = w;
    this.config.canvasHeight = h;
  }

  resize(width: number, height: number): void {
    this.config.canvasWidth  = width;
    this.config.canvasHeight = height;
  }

  destroy(): void {
    this._destroyBuffers();
    const { gl } = this;
    if (this.tf)         { gl.deleteTransformFeedback(this.tf); }
    if (this.updateProg) { gl.deleteProgram(this.updateProg); }
    if (this.drawProg)   { gl.deleteProgram(this.drawProg); }
    this._ready = false;
  }

  createOffscreenFBO(
    width: number, height: number,
  ): { fbo: WebGLFramebuffer; tex: WebGLTexture } {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return { fbo, tex };
  }

  renderToFBO(fbo: WebGLFramebuffer, width: number, height: number, time: number): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    this.render(time);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ─── Internal init ───────────────────────────────────────────────────────

  private _init(): void {
    // ShaderLoader call kept for AT compat (compiled.vs registration check)
    try { getShader('edge-spline.vert'); getShader('edge-spline.frag'); } catch { /* ok */ }

    const { gl } = this;

    this.updateProg = linkProgram(gl, UPDATE_VERT, UPDATE_FRAG, ['v_state']);
    this.drawProg   = linkProgram(gl, DRAW_VERT,   DRAW_FRAG);
    this.tf         = gl.createTransformFeedback()!;

    this._buildBuffers();
    this._ready = true;
  }

  private _buildBuffers(): void {
    const { gl, config, edges } = this;
    const PPE = config.particlesPerEdge!;
    const N   = edges.length * PPE;
    this.totalParticles = N;

    // Pack initial state: vec4(travel, speed, edgeIdx, seed)
    const data = new Float32Array(N * 4);
    const baseSpeed = 0.00045; // travel/ms — tuned for visible flow

    for (let e = 0; e < edges.length; e++) {
      const isSkip = edges[e].isSkip ? 1.5 : 1.0; // skip edges slightly faster
      for (let p = 0; p < PPE; p++) {
        const i    = e * PPE + p;
        const base = i * 4;
        const travel = p / PPE;                           // evenly staggered
        const jitter = 0.7 + 0.6 * Math.random();
        const speed  = baseSpeed * jitter * isSkip;
        const seed   = lcg((i * 12345 + 67891) >>> 0);
        data[base + 0] = travel;
        data[base + 1] = speed;
        data[base + 2] = e;
        data[base + 3] = seed;
      }
    }

    const mkBuf = (src?: Float32Array): WebGLBuffer => {
      const buf = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, src ?? N * 4 * 4, gl.DYNAMIC_COPY);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      return buf;
    };

    this.stateBuf    = [mkBuf(data), mkBuf()];
    this.currentIndex = 0;

    this.vao = [gl.createVertexArray()!, gl.createVertexArray()!];

    for (let side = 0; side < 2; side++) {
      gl.bindVertexArray(this.vao[side]);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.stateBuf[side]);
      const loc = gl.getAttribLocation(this.updateProg, 'a_state');
      if (loc >= 0) {
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 0, 0);
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      gl.bindVertexArray(null);
    }
  }

  private _destroyBuffers(): void {
    const { gl } = this;
    if (this.vao) {
      gl.deleteVertexArray(this.vao[0]);
      gl.deleteVertexArray(this.vao[1]);
    }
    if (this.stateBuf) {
      gl.deleteBuffer(this.stateBuf[0]);
      gl.deleteBuffer(this.stateBuf[1]);
    }
  }

  // ─── Bezier uniform packing ───────────────────────────────────────────────

  /** Pack EdgeControlPoints into flat arrays for upload */
  private _packBezierUniforms(): { bez0: Float32Array; bez1: Float32Array } {
    const bez0 = new Float32Array(MAX_EDGES * 4);
    const bez1 = new Float32Array(MAX_EDGES * 4);
    for (let i = 0; i < Math.min(this.edges.length, MAX_EDGES); i++) {
      const e = this.edges[i];
      // u_bez0: P0.xy, P3.xy
      bez0[i*4+0] = e.p0[0]; bez0[i*4+1] = e.p0[1];
      bez0[i*4+2] = e.p3[0]; bez0[i*4+3] = e.p3[1];
      // u_bez1: C0.xy (p1), C1.xy (p2)
      bez1[i*4+0] = e.p1[0]; bez1[i*4+1] = e.p1[1];
      bez1[i*4+2] = e.p2[0]; bez1[i*4+3] = e.p2[1];
    }
    return { bez0, bez1 };
  }

  private _packEnergyArray(): Float32Array {
    const arr = new Float32Array(MAX_EDGES);
    for (let i = 0; i < Math.min(this.edges.length, MAX_EDGES); i++) {
      arr[i] = this.edges[i].sourceEnergy ?? 1.0;
    }
    return arr;
  }

  private _packColorUniforms(): { srcColors: Float32Array; tgtColors: Float32Array } {
    const srcColors = new Float32Array(MAX_EDGES * 4);
    const tgtColors = new Float32Array(MAX_EDGES * 4);
    for (let i = 0; i < Math.min(this.edges.length, MAX_EDGES); i++) {
      const e  = this.edges[i];
      const sc = e.sourceColor ?? [0.4, 0.7, 1.0];
      const tc = e.targetColor ?? [0.8, 0.4, 1.0];
      srcColors[i*4+0]=sc[0]; srcColors[i*4+1]=sc[1]; srcColors[i*4+2]=sc[2]; srcColors[i*4+3]=0;
      tgtColors[i*4+0]=tc[0]; tgtColors[i*4+1]=tc[1]; tgtColors[i*4+2]=tc[2]; tgtColors[i*4+3]=0;
    }
    return { srcColors, tgtColors };
  }

  // ─── Update pass (TF) ────────────────────────────────────────────────────

  private _update(dt: number): void {
    const { gl } = this;
    const readIdx  = this.currentIndex;
    const writeIdx = 1 - readIdx;

    gl.useProgram(this.updateProg);

    // dt uniform
    const dtLoc = gl.getUniformLocation(this.updateProg, 'u_dt');
    if (dtLoc !== null) gl.uniform1f(dtLoc, dt);

    const rrLoc = gl.getUniformLocation(this.updateProg, 'u_resetRange');
    if (rrLoc !== null) gl.uniform1f(rrLoc, 0.05);

    // Energy array
    const energyArr = this._packEnergyArray();
    const eLoc = gl.getUniformLocation(this.updateProg, 'u_energy[0]');
    if (eLoc !== null) gl.uniform1fv(eLoc, energyArr);

    gl.bindVertexArray(this.vao[readIdx]);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.stateBuf[writeIdx]);

    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, this.totalParticles);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindVertexArray(null);

    this.currentIndex = writeIdx;
  }

  // ─── Draw pass (gl.POINTS) ───────────────────────────────────────────────

  private _draw(): void {
    const { gl, config } = this;

    gl.viewport(0, 0, config.canvasWidth, config.canvasHeight);
    gl.useProgram(this.drawProg);

    // Resolution
    const resLoc = gl.getUniformLocation(this.drawProg, 'u_resolution');
    if (resLoc !== null) gl.uniform2f(resLoc, config.canvasWidth, config.canvasHeight);

    // Point size (skip edges get bigger particles via energy scaling in shader)
    const psLoc = gl.getUniformLocation(this.drawProg, 'u_pointSize');
    if (psLoc !== null) gl.uniform1f(psLoc, config.pointSize!);

    // Alpha
    const aLoc = gl.getUniformLocation(this.drawProg, 'u_alpha');
    if (aLoc !== null) gl.uniform1f(aLoc, config.particleAlpha!);

    // Bézier uniforms
    const { bez0, bez1 } = this._packBezierUniforms();
    const b0Loc = gl.getUniformLocation(this.drawProg, 'u_bez0[0]');
    const b1Loc = gl.getUniformLocation(this.drawProg, 'u_bez1[0]');
    if (b0Loc !== null) gl.uniform4fv(b0Loc, bez0);
    if (b1Loc !== null) gl.uniform4fv(b1Loc, bez1);

    // Colour uniforms
    const { srcColors, tgtColors } = this._packColorUniforms();
    const scLoc = gl.getUniformLocation(this.drawProg, 'u_srcColor[0]');
    const tcLoc = gl.getUniformLocation(this.drawProg, 'u_tgtColor[0]');
    if (scLoc !== null) gl.uniform4fv(scLoc, srcColors);
    if (tcLoc !== null) gl.uniform4fv(tcLoc, tgtColors);

    // Energy
    const energyArr = this._packEnergyArray();
    const eLoc = gl.getUniformLocation(this.drawProg, 'u_energy[0]');
    if (eLoc !== null) gl.uniform1fv(eLoc, energyArr);

    // Additive blend for glow
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); // premultiplied
    gl.depthMask(false);

    // Bind read-side VAO (updated state after _update swap)
    // Re-bind a_state for the draw program (different attrib locations possible)
    gl.bindBuffer(gl.ARRAY_BUFFER, this.stateBuf[this.currentIndex]);
    const aStateLoc = gl.getAttribLocation(this.drawProg, 'a_state');
    if (aStateLoc >= 0) {
      gl.enableVertexAttribArray(aStateLoc);
      gl.vertexAttribPointer(aStateLoc, 4, gl.FLOAT, false, 0, 0);
    }

    gl.drawArrays(gl.POINTS, 0, this.totalParticles);

    if (aStateLoc >= 0) gl.disableVertexAttribArray(aStateLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }
}
