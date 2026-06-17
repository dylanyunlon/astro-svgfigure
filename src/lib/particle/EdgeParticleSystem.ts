/**
 * EdgeParticleSystem.ts — WebGL2 Transform Feedback 粒子系统
 *
 * M041: 修复 EdgeParticleSystem WebGL2 transform feedback
 *
 * 粒子沿 edge 贝塞尔曲线 source→target 流动。
 * 每条 edge (route.json) 对应一条三次贝塞尔曲线，粒子从 P0 出发沿曲线
 * 向 P1 流动，到达终点后随机 offset 重新从起点出发。
 *
 * 架构 (参考 upstream/RESEARCH_121_webgl2_particles.md &
 *       upstream/RESEARCH_122_webgl2_particles_clean.md):
 *
 *   Ping-Pong Buffer (SEPARATE_ATTRIBS 模式):
 *     bufA → (update pass / TF) → bufB   帧 N
 *     bufB → (update pass / TF) → bufA   帧 N+1
 *     currentIndex ^= 1 切换读写侧
 *
 *   每个粒子状态 (4 floats — vec4):
 *     .x = travel  : 沿贝塞尔曲线的归一化进度 [0, 1]
 *     .y = speed   : 粒子速度 (每帧 travel += speed * dt)
 *     .z = edgeIdx : 归属的 edge 索引 (float, 整数语义)
 *     .w = seed    : LCG 随机种子，到达终点时重新生成初始 offset
 *
 *   Update Pass (transform feedback, RASTERIZER_DISCARD):
 *     vertex shader 更新 travel → out v_state
 *     到达 1.0 → 用 LCG 重置到随机 [0, resetRange] 并更新 seed
 *
 *   Draw Pass (gl.POINTS):
 *     vertex shader 用 travel + edge 贝塞尔参数评估 XY 屏幕坐标
 *     fragment shader 输出发光点精灵
 *
 *   Ticker 集成:
 *     每帧调用 update(deltaMS) + draw() 即可；
 *     EdgeRenderer.tick() 同样调用此处 tick(dt) 以统一时间轴。
 *
 * 上游参考:
 *   upstream/RESEARCH_121_webgl2_particles.md  — TF ping-pong 机制
 *   upstream/RESEARCH_122_webgl2_particles_clean.md — 简洁 TF 架构
 *   src/lib/renderers/antimatter-compute.ts — AntimatterPass TF 实现
 *   src/lib/EdgeRenderer.ts — route.json 格式与贝塞尔控制点推导
 *   src/lib/renderers/epoch-ticker.ts — Ticker 集成方式
 */

// ── Public types ──────────────────────────────────────────────────────────────

/** 从 channels/edge/{id}/route.json 读取的路由数据 (与 EdgeRenderer 共用格式) */
export interface EdgeRoute {
  edge_id:       string;
  sources:       string[];
  targets:       string[];
  advanced:      { semanticType?: string; routing?: string; curvature?: number };
  points:        Array<{ x: number; y: number }>;
  z:             number;
  rerouted_epoch: number;
}

export interface EdgeParticleSystemConfig {
  gl:             WebGL2RenderingContext;
  /** canvas 宽度 (像素), 用于 NDC 变换 */
  width:          number;
  /** canvas 高度 (像素) */
  height:         number;
  /** 每条 edge 分配的粒子数量 (default: 32) */
  particlesPerEdge?: number;
  /** 粒子速度范围 [min, max], 单位: travel/ms (default: [0.0003, 0.0007]) */
  speedRange?: [number, number];
  /** 粒子点精灵大小 (像素, default: 3) */
  pointSize?: number;
  /** 粒子颜色 RGB [0,1] (default: [0.39, 0.71, 0.96] — #64B5F6 天蓝) */
  color?: [number, number, number];
  /** skip 连接粒子颜色 (default: [1.0, 0.72, 0.30] — #FFB74D 琥珀) */
  skipColor?: [number, number, number];
  /** 粒子透明度 (default: 0.7) */
  alpha?: number;
  /** 到达终点后随机重置到 [0, resetRange], 产生持续流动感 (default: 0.05) */
  resetRange?: number;
}

// ── Bezier helpers ────────────────────────────────────────────────────────────

/** 单条 edge 的贝塞尔参数 (上传到 GPU uniform array) */
interface EdgeBezier {
  p0x: number; p0y: number;
  p1x: number; p1y: number;
  c0x: number; c0y: number;
  c1x: number; c1y: number;
  /** 1 = skip connection, 0 = straight edge */
  isSkip: number;
}

/**
 * 从 route.json points[] 和 curvature 推导三次贝塞尔控制点。
 * 与 EdgeRenderer.deriveControlPoints 完全等价，保持一致。
 */
function deriveControlPoints(
  p0: { x: number; y: number },
  p1: { x: number; y: number },
  mid: { x: number; y: number },
  curvature: number,
): { c0: { x: number; y: number }; c1: { x: number; y: number } } {
  return {
    c0: {
      x: p0.x + (mid.x - p0.x) * curvature,
      y: p0.y + (mid.y - p0.y) * curvature,
    },
    c1: {
      x: p1.x + (mid.x - p1.x) * curvature,
      y: p1.y + (mid.y - p1.y) * curvature,
    },
  };
}

function buildEdgeBezier(route: EdgeRoute): EdgeBezier {
  const pts = route.points;
  const p0  = pts[0];
  const p1  = pts[pts.length - 1];
  const mid = pts[Math.floor(pts.length / 2)] ?? {
    x: (p0.x + p1.x) * 0.5,
    y: (p0.y + p1.y) * 0.5,
  };

  const isSkip =
    route.advanced?.routing === 'SPLINES' ||
    route.advanced?.semanticType === 'skip_connection'
      ? 1
      : 0;

  const curvature = route.advanced?.curvature ?? (isSkip ? 0.5 : 0.33);
  const { c0, c1 } = deriveControlPoints(p0, p1, mid, curvature);

  return { p0x: p0.x, p0y: p0.y, p1x: p1.x, p1y: p1.y, c0x: c0.x, c0y: c0.y, c1x: c1.x, c1y: c1.y, isSkip };
}

// ── LCG random (Microsoft VC++ params, same as RESEARCH_121) ─────────────────

function lcg(seed: number): number {
  // returns value in [0, 1)
  return (((seed * 1664525 + 1013904223) & 0xffffffff) >>> 0) / 4294967296.0;
}

// ── GL helpers ────────────────────────────────────────────────────────────────

function compileShader(gl: WebGL2RenderingContext, type: GLenum, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`[EdgeParticleSystem] Shader compile error:\n${log}`);
  }
  return sh;
}

function createProgram(
  gl: WebGL2RenderingContext,
  vertSrc: string,
  fragSrc: string,
  varyings?: string[],
): WebGLProgram {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, compileShader(gl, gl.VERTEX_SHADER,   vertSrc));
  gl.attachShader(prog, compileShader(gl, gl.FRAGMENT_SHADER, fragSrc));

  if (varyings && varyings.length > 0) {
    // Must be called BEFORE linkProgram
    gl.transformFeedbackVaryings(prog, varyings, gl.SEPARATE_ATTRIBS);
  }

  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`[EdgeParticleSystem] Program link error:\n${log}`);
  }
  return prog;
}

// ── GLSL sources ──────────────────────────────────────────────────────────────

/**
 * Update vertex shader — runs under RASTERIZER_DISCARD, outputs via TF.
 *
 * Each "vertex" = one particle state vec4(travel, speed, edgeIdx, seed).
 * Advances travel by speed * u_dt. On wrap (travel >= 1), resets to
 * a small random offset using LCG so particles stagger their restarts.
 *
 * Bezier parameters are packed into a uniform array indexed by edgeIdx.
 * MAX_EDGES must match the JS constant below.
 */
const MAX_EDGES = 16; // enough for e1-e6 + skip1-skip2 with room

const UPDATE_VERT = /* glsl */`#version 300 es
precision highp float;

// Input particle state: (travel, speed, edgeIdx, seed)
in vec4 a_state;

// Time delta in milliseconds
uniform float u_dt;
// reset range: particles restart in [0, u_resetRange]
uniform float u_resetRange;

// Bezier parameters packed: [p0x,p0y, p1x,p1y, c0x,c0y, c1x,c1y] per edge
// 8 floats × MAX_EDGES
uniform vec4 u_bez0[${MAX_EDGES}]; // p0x,p0y, p1x,p1y
uniform vec4 u_bez1[${MAX_EDGES}]; // c0x,c0y, c1x,c1y

// Transform feedback output
out vec4 v_state;

// LCG: Microsoft VC++ multiplier / increment
uint lcg(uint seed) {
  return seed * 1664525u + 1013904223u;
}

float lcgF(uint seed) {
  return float(lcg(seed)) / 4294967296.0;
}

void main() {
  float travel  = a_state.x;
  float speed   = a_state.y;
  float edgeIdx = a_state.z;
  uint  seed    = floatBitsToUint(a_state.w);

  // Advance particle along its edge
  travel += speed * u_dt;

  if (travel >= 1.0) {
    // Reached target — reset to [0, resetRange] with LCG jitter
    seed   = lcg(seed);
    travel = lcgF(seed) * u_resetRange;
  }

  v_state = vec4(travel, speed, edgeIdx, uintBitsToFloat(seed));
}
`;

const UPDATE_FRAG = /* glsl */`#version 300 es
precision highp float;
out vec4 unused;
void main() { unused = vec4(0.0); }
`;

/**
 * Draw vertex shader — evaluates bezier at travel, outputs gl_Position.
 * Uses gl.POINTS; fragment outputs a circular glow sprite.
 */
const DRAW_VERT = /* glsl */`#version 300 es
precision highp float;

in vec4  a_state;     // (travel, speed, edgeIdx, seed)

uniform vec2  u_resolution;
uniform float u_pointSize;
uniform vec4  u_bez0[${MAX_EDGES}]; // p0x,p0y, p1x,p1y
uniform vec4  u_bez1[${MAX_EDGES}]; // c0x,c0y, c1x,c1y

out float v_alpha;
out float v_skip;

// Evaluate cubic bezier at t
vec2 bezier(int idx, float t) {
  vec4 b0 = u_bez0[idx]; // p0, p1
  vec4 b1 = u_bez1[idx]; // c0, c1
  vec2 P0 = b0.xy;
  vec2 P1 = b0.zw;
  vec2 C0 = b1.xy;
  vec2 C1 = b1.zw;

  float mt  = 1.0 - t;
  float mt2 = mt * mt;
  float t2  = t  * t;
  return mt2*mt*P0 + 3.0*mt2*t*C0 + 3.0*mt*t2*C1 + t2*t*P1;
}

// pixel → NDC (Y flipped for screen-space convention)
vec2 pixelToNDC(vec2 px) {
  vec2 ndc = (px / u_resolution) * 2.0 - 1.0;
  ndc.y = -ndc.y;
  return ndc;
}

void main() {
  float travel  = a_state.x;
  int   edgeIdx = int(a_state.z + 0.5);

  // Clamp index to valid range (defensive)
  edgeIdx = clamp(edgeIdx, 0, ${MAX_EDGES - 1});

  vec2 posPx = bezier(edgeIdx, clamp(travel, 0.0, 1.0));

  // Fade in / out at endpoints for smooth appearance
  float fadeIn  = smoothstep(0.0, 0.05, travel);
  float fadeOut = 1.0 - smoothstep(0.92, 1.0, travel);
  v_alpha = fadeIn * fadeOut;

  // isSkip flag is encoded in b0.xy == b0.zw trick — instead we use
  // the edgeIdx we already know from config upload (u_skip uniform unused).
  // For simplicity, the fragment shader receives v_alpha; colour is per-draw.
  v_skip  = 0.0; // see note — colour chosen by draw call, not per-particle

  gl_Position  = vec4(pixelToNDC(posPx), 0.0, 1.0);
  gl_PointSize = u_pointSize;
}
`;

const DRAW_FRAG = /* glsl */`#version 300 es
precision highp float;

uniform vec3  u_color;
uniform float u_alpha;

in float v_alpha;
in float v_skip;

out vec4 fragColor;

void main() {
  // Circular soft sprite (gl_PointCoord in [0,1]²)
  vec2  uv   = gl_PointCoord * 2.0 - 1.0;
  float r    = dot(uv, uv);
  if (r > 1.0) discard;

  // Glow falloff
  float glow = exp(-r * 2.5);
  float core = (1.0 - smoothstep(0.0, 0.5, r));
  float mask = max(glow * 0.6, core);

  float alpha = mask * v_alpha * u_alpha;
  if (alpha < 0.004) discard;

  // Pre-multiplied alpha (additive blend in draw())
  fragColor = vec4(u_color * alpha, alpha);
}
`;

// ── EdgeParticleSystem ────────────────────────────────────────────────────────

export class EdgeParticleSystem {
  private readonly gl:         WebGL2RenderingContext;
  private width:                number;
  private height:               number;
  private readonly cfg:         Required<EdgeParticleSystemConfig>;

  // ── GL objects ──────────────────────────────────────────────────────────
  private updateProg: WebGLProgram | null = null;
  private drawProg:   WebGLProgram | null = null;

  /** Single reused TransformFeedback object (pwambach pattern) */
  private tf: WebGLTransformFeedback | null = null;

  /** Ping-pong: index 0 or 1 is the current READ side */
  private currentIndex = 0;
  /** Two VAOs for ping-pong input binding */
  private vao:    [WebGLVertexArrayObject, WebGLVertexArrayObject] | null = null;
  /** Two buffers: state[currentIndex] = input, state[1-currentIndex] = output */
  private stateBuf: [WebGLBuffer, WebGLBuffer] | null = null;

  // ── Per-edge bezier params ───────────────────────────────────────────────
  private edges:      EdgeBezier[] = [];
  private edgeRoutes: EdgeRoute[]  = [];
  private totalParticles = 0;

  // ── Timing ──────────────────────────────────────────────────────────────
  private elapsed = 0;

  // ── State ────────────────────────────────────────────────────────────────
  private _ready = false;

  constructor(config: EdgeParticleSystemConfig) {
    this.gl     = config.gl;
    this.width  = config.width;
    this.height = config.height;

    // Apply defaults
    this.cfg = {
      gl:              config.gl,
      width:           config.width,
      height:          config.height,
      particlesPerEdge: config.particlesPerEdge ?? 32,
      speedRange:      config.speedRange      ?? [0.0003, 0.0007],
      pointSize:       config.pointSize       ?? 3,
      color:           config.color           ?? [0.39, 0.71, 0.96],
      skipColor:       config.skipColor       ?? [1.00, 0.72, 0.30],
      alpha:           config.alpha           ?? 0.7,
      resetRange:      config.resetRange      ?? 0.05,
    };
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /** True after init() completes successfully. */
  get ready(): boolean { return this._ready; }

  /**
   * Load route data and compile shaders.
   * Call once; after this, call tick() every frame.
   */
  async loadRoutes(basePath = '/channels/edge'): Promise<void> {
    const edgeIds = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'skip1', 'skip2'];
    const routes: EdgeRoute[] = [];
    await Promise.all(
      edgeIds.map(async (id) => {
        try {
          const r = await fetch(`${basePath}/${id}/route.json`);
          if (r.ok) routes.push(await r.json());
        } catch {
          // missing edge is fine — skip silently
        }
      }),
    );
    routes.sort((a, b) => edgeIds.indexOf(a.edge_id) - edgeIds.indexOf(b.edge_id));
    this.initFromRoutes(routes);
  }

  /**
   * Initialise directly from pre-loaded EdgeRoute array.
   * Use this when route data is already available (e.g. from EdgeRenderer).
   */
  initFromRoutes(routes: EdgeRoute[]): void {
    if (routes.length === 0) {
      console.warn('[EdgeParticleSystem] No routes provided — system inactive.');
      return;
    }
    if (routes.length > MAX_EDGES) {
      console.warn(
        `[EdgeParticleSystem] ${routes.length} edges exceeds MAX_EDGES=${MAX_EDGES}; ` +
        `truncating to first ${MAX_EDGES}.`,
      );
      routes = routes.slice(0, MAX_EDGES);
    }

    this.edgeRoutes = routes;
    this.edges      = routes.map(buildEdgeBezier);

    this._compilePrograms();
    this._buildBuffers();
    this._ready = true;

    console.log(
      `[EdgeParticleSystem] Ready — ${routes.length} edges, ` +
      `${this.totalParticles} particles, ` +
      `speedRange=[${this.cfg.speedRange}]`,
    );
  }

  /**
   * Advance simulation by dt milliseconds + draw particles.
   * Call once per frame from your Ticker callback.
   *
   * @example
   * ```ts
   * ticker.onFrame(({ deltaMS }) => eps.tick(deltaMS));
   * ```
   */
  tick(deltaMS: number): void {
    if (!this._ready) return;
    this.elapsed += deltaMS;
    this._update(deltaMS);
    this._draw();
  }

  /**
   * Update canvas resolution (call on resize).
   */
  resize(w: number, h: number): void {
    this.width  = w;
    this.height = h;
  }

  /**
   * Release all WebGL resources.
   */
  dispose(): void {
    const { gl } = this;
    if (this.updateProg) { gl.deleteProgram(this.updateProg); this.updateProg = null; }
    if (this.drawProg)   { gl.deleteProgram(this.drawProg);   this.drawProg   = null; }
    if (this.tf)         { gl.deleteTransformFeedback(this.tf); this.tf = null; }
    if (this.vao) {
      gl.deleteVertexArray(this.vao[0]);
      gl.deleteVertexArray(this.vao[1]);
      this.vao = null;
    }
    if (this.stateBuf) {
      gl.deleteBuffer(this.stateBuf[0]);
      gl.deleteBuffer(this.stateBuf[1]);
      this.stateBuf = null;
    }
    this._ready = false;
  }

  // ── Private: compile ──────────────────────────────────────────────────────

  private _compilePrograms(): void {
    const { gl } = this;

    // Update program with transform feedback varyings
    this.updateProg = createProgram(
      gl,
      UPDATE_VERT,
      UPDATE_FRAG,
      ['v_state'],           // SEPARATE_ATTRIBS: one buffer per varying
    );

    // Draw program — no TF
    this.drawProg = createProgram(gl, DRAW_VERT, DRAW_FRAG);

    // Create single TF object (reused every frame — pwambach #122 pattern)
    this.tf = gl.createTransformFeedback()!;
  }

  // ── Private: buffers ──────────────────────────────────────────────────────

  private _buildBuffers(): void {
    const { gl, cfg, edges } = this;
    const PPE = cfg.particlesPerEdge;

    this.totalParticles = edges.length * PPE;
    const N = this.totalParticles;

    // Pack initial particle state: vec4(travel, speed, edgeIdx, seed)
    const initData = new Float32Array(N * 4);

    for (let e = 0; e < edges.length; e++) {
      for (let p = 0; p < PPE; p++) {
        const i     = e * PPE + p;
        const base  = i * 4;

        // Distribute particles evenly along edge at start
        const travel = p / PPE;
        // Random speed in [speedRange[0], speedRange[1]]
        const speed  = cfg.speedRange[0] +
                       Math.random() * (cfg.speedRange[1] - cfg.speedRange[0]);
        // LCG seed — non-zero, deterministic per particle
        const seed   = (i * 12345 + 67891) >>> 0;

        initData[base + 0] = travel;
        initData[base + 1] = speed;
        initData[base + 2] = e;                     // edgeIdx
        initData[base + 3] = lcg(seed);             // initial seed float
      }
    }

    // Two ping-pong buffers
    const createBuf = (data?: Float32Array): WebGLBuffer => {
      const buf = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, data ?? N * 4 * 4, gl.DYNAMIC_COPY);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      return buf;
    };

    this.stateBuf = [createBuf(initData), createBuf()];

    // Two VAOs — one per ping-pong side — for the update (TF) and draw passes
    // Each VAO binds a_state → the corresponding stateBuf
    this.vao = [
      gl.createVertexArray()!,
      gl.createVertexArray()!,
    ];

    for (let side = 0; side < 2; side++) {
      gl.bindVertexArray(this.vao[side]);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.stateBuf[side]);

      // a_state layout: 4 floats, no stride/offset
      const loc = gl.getAttribLocation(this.updateProg!, 'a_state');
      if (loc >= 0) {
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 0, 0);
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      gl.bindVertexArray(null);
    }
  }

  // ── Private: upload bezier uniforms ──────────────────────────────────────

  private _uploadBezierUniforms(prog: WebGLProgram): void {
    const { gl, edges } = this;

    // Pack u_bez0[i] = vec4(p0x, p0y, p1x, p1y)
    // Pack u_bez1[i] = vec4(c0x, c0y, c1x, c1y)
    const bez0 = new Float32Array(MAX_EDGES * 4);
    const bez1 = new Float32Array(MAX_EDGES * 4);

    for (let i = 0; i < edges.length; i++) {
      const e = edges[i];
      bez0[i * 4 + 0] = e.p0x; bez0[i * 4 + 1] = e.p0y;
      bez0[i * 4 + 2] = e.p1x; bez0[i * 4 + 3] = e.p1y;
      bez1[i * 4 + 0] = e.c0x; bez1[i * 4 + 1] = e.c0y;
      bez1[i * 4 + 2] = e.c1x; bez1[i * 4 + 3] = e.c1y;
    }

    const loc0 = gl.getUniformLocation(prog, 'u_bez0[0]');
    const loc1 = gl.getUniformLocation(prog, 'u_bez1[0]');
    if (loc0) gl.uniform4fv(loc0, bez0);
    if (loc1) gl.uniform4fv(loc1, bez1);
  }

  // ── Private: update pass (transform feedback) ─────────────────────────────

  private _update(deltaMS: number): void {
    const { gl } = this;
    if (!this.updateProg || !this.tf || !this.vao || !this.stateBuf) return;

    const readIdx  = this.currentIndex;
    const writeIdx = 1 - readIdx;

    gl.useProgram(this.updateProg);

    // ── Set uniforms ──────────────────────────────────────────────────
    const dtLoc = gl.getUniformLocation(this.updateProg, 'u_dt');
    if (dtLoc !== null) gl.uniform1f(dtLoc, deltaMS);

    const rrLoc = gl.getUniformLocation(this.updateProg, 'u_resetRange');
    if (rrLoc !== null) gl.uniform1f(rrLoc, this.cfg.resetRange);

    this._uploadBezierUniforms(this.updateProg);

    // ── Bind input VAO (read side) ────────────────────────────────────
    gl.bindVertexArray(this.vao[readIdx]);

    // ── Bind TF output buffer (write side) ────────────────────────────
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.stateBuf[writeIdx]);

    // ── RASTERIZER_DISCARD: skip fragment stage — TF output only ─────
    gl.enable(gl.RASTERIZER_DISCARD);

    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, this.totalParticles);
    gl.endTransformFeedback();

    gl.disable(gl.RASTERIZER_DISCARD);

    // ── Unbind ────────────────────────────────────────────────────────
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindVertexArray(null);

    // Ping-pong: swap read/write
    this.currentIndex = writeIdx;
  }

  // ── Private: draw pass (gl.POINTS) ────────────────────────────────────────

  private _draw(): void {
    const { gl, cfg, edges } = this;
    if (!this.drawProg || !this.vao || !this.stateBuf) return;

    gl.useProgram(this.drawProg);

    // Resolution uniform
    const resLoc = gl.getUniformLocation(this.drawProg, 'u_resolution');
    if (resLoc !== null) gl.uniform2f(resLoc, this.width, this.height);

    // Point size
    const psLoc = gl.getUniformLocation(this.drawProg, 'u_pointSize');
    if (psLoc !== null) gl.uniform1f(psLoc, cfg.pointSize);

    // Alpha
    const aLoc = gl.getUniformLocation(this.drawProg, 'u_alpha');
    if (aLoc !== null) gl.uniform1f(aLoc, cfg.alpha);

    // Bezier params
    this._uploadBezierUniforms(this.drawProg);

    // Additive blend for glow effect (AT standard for particles)
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.depthMask(false);

    // ── Draw normal edges ──────────────────────────────────────────────
    const cLoc = gl.getUniformLocation(this.drawProg, 'u_color');

    const normalEdges  = edges.filter(e => !e.isSkip);
    const skipEdges    = edges.filter(e => e.isSkip);
    const PPE          = cfg.particlesPerEdge;

    // Bind the READ-side VAO (updated state after _update swap)
    // We need to re-create a minimal VAO pointing at the current read buffer
    // because our pre-built VAOs were created for the update program's a_state.
    // The draw program also uses a_state so we can reuse the same VAO.
    gl.bindVertexArray(this.vao[this.currentIndex]);

    // Normal edges (straight feed-forward)
    if (normalEdges.length > 0) {
      if (cLoc !== null) gl.uniform3fv(cLoc, new Float32Array(cfg.color));
      for (let e = 0; e < edges.length; e++) {
        if (edges[e].isSkip) continue;
        const first = e * PPE;
        gl.drawArrays(gl.POINTS, first, PPE);
      }
    }

    // Skip connections (different colour)
    if (skipEdges.length > 0) {
      if (cLoc !== null) gl.uniform3fv(cLoc, new Float32Array(cfg.skipColor));
      for (let e = 0; e < edges.length; e++) {
        if (!edges[e].isSkip) continue;
        const first = e * PPE;
        gl.drawArrays(gl.POINTS, first, PPE);
      }
    }

    gl.bindVertexArray(null);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }
}

// ── Factory ───────────────────────────────────────────────────────────────────

/**
 * Create and load an EdgeParticleSystem from a canvas element.
 *
 * @example
 * ```ts
 * const canvas = document.querySelector<HTMLCanvasElement>('#gl-canvas')!;
 * const eps = await createEdgeParticleSystem(canvas, {
 *   particlesPerEdge: 48,
 *   color: [0.39, 0.71, 0.96],
 * });
 *
 * // In your Ticker:
 * ticker.onFrame(({ deltaMS }) => eps.tick(deltaMS));
 * ```
 */
export async function createEdgeParticleSystem(
  canvas: HTMLCanvasElement,
  opts: Omit<EdgeParticleSystemConfig, 'gl' | 'width' | 'height'> = {},
  basePath = '/channels/edge',
): Promise<EdgeParticleSystem> {
  const gl = canvas.getContext('webgl2');
  if (!gl) throw new Error('[EdgeParticleSystem] WebGL2 not supported.');

  const eps = new EdgeParticleSystem({
    gl,
    width:  canvas.width,
    height: canvas.height,
    ...opts,
  });

  await eps.loadRoutes(basePath);
  return eps;
}
