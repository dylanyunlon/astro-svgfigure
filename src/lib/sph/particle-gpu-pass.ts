/**
 * particle-gpu-pass.ts — M875
 *
 * GPU Transform Feedback 粒子系统: 5000 粒子沿 edge 方向运动,
 * 颜色从 source cell species 渐变到 target cell species.
 *
 * 核心实现:
 *   • WebGL2 context (canvas.getContext('webgl2'))
 *   • 每粒子数据: vec2 position, vec2 velocity, float life, vec3 color (9 floats)
 *   • Update pass: Transform Feedback + gl.beginTransformFeedback(POINTS)
 *   • 两个 VBO ping-pong: readVBO → TF → writeVBO
 *   • Render pass: gl.drawArrays(POINTS, 0, 5000) 用 gl_PointSize
 *   • AT shader 从 compiled.vs 通过 ShaderLoader.getShader() 提取
 *
 * Buffer layout per particle (9 × float32 = 36 bytes):
 *   [0,1]  position  vec2  NDC world space
 *   [2,3]  velocity  vec2  units/s
 *   [4]    life      float [0, 1] — 1=alive, 0=dead → respawn
 *   [5,6,7] color   vec3  RGB [0..1]
 *   [8]    _pad      float alignment (stride = 9 floats = 36 bytes)
 *
 * Ping-pong transform feedback:
 *   frame N:  vboA → read attribs, vboB → TF output
 *   frame N+1: vboB → read attribs, vboA → TF output
 *
 * Shader source for the update pass (WebGL2, #version 300 es):
 *   - in/out varyings (no longer varying keyword)
 *   - transform feedback varyings declared before linking
 *
 * Shader source for the render pass:
 *   - uses AT edge-spline color gradient pattern
 *   - gl_PointSize driven by life
 *   - soft round point via discard
 *
 * Integration:
 *   ```ts
 *   const gpu = new ParticleGPU(canvas, edges);
 *   // render loop:
 *   gpu.update(dt);
 *   gpu.render(canvasW, canvasH);
 *   // add new edge:
 *   gpu.addEdge({ edgeId, srcPos, dstPos, srcColor, dstColor });
 *   // cleanup:
 *   gpu.destroy();
 *   ```
 *
 * References:
 *   src/lib/sph/fluid-gpu-pass.ts       — WebGL1 reference (_compile, _createSingleFBO)
 *   src/lib/shaders/ShaderLoader.ts     — getShader('name.fs') from compiled.vs
 *   src/lib/shaders/edge-spline.frag    — AT edge color gradient pattern
 *   src/lib/sph/edge-flow-renderer.ts   — FlowEdge / FlowPoint types
 *   src/lib/sph/world-renderer.ts       — SPECIES_COLORS palette
 */




// ─── Constants ────────────────────────────────────────────────────────────────

/** Total particle pool (shared across all edges). */



import { getShader } from '../shaders/ShaderLoader';

const PARTICLE_COUNT = 5000;

/** Floats per particle in the VBO (vec2 pos + vec2 vel + float life + vec3 color + float pad). */
const PARTICLE_STRIDE_F32 = 9;

/** Byte stride per particle. */
const PARTICLE_STRIDE_BYTES = PARTICLE_STRIDE_F32 * 4; // 36 bytes

/** Particle max lifetime in seconds before respawn. */
const PARTICLE_LIFETIME = 2.5;

/** Default point size base (pixels). */
const POINT_SIZE_BASE = 4.0;

/** Maximum concurrent edge definitions. */
const MAX_EDGES = 64;

// ─── Species color palette (mirrors world-renderer.ts SPECIES_COLORS) ────────

const SPECIES_COLORS_RGB: Record<number, [number, number, number]> = {
  0: [0.247, 0.318, 0.710],  // #3F51B5 — indigo
  1: [1.000, 0.435, 0.000],  // #FF6F00 — amber
  2: [0.180, 0.490, 0.196],  // #2E7D32 — green
  3: [0.776, 0.157, 0.157],  // #C62828 — red
  4: [0.271, 0.353, 0.392],  // #455A64 — blue-grey
  5: [0.482, 0.122, 0.635],  // #7B1FA2 — purple
  6: [0.086, 0.396, 0.753],  // #1565C0 — blue
};

function speciesRGB(species: number): [number, number, number] {
  return SPECIES_COLORS_RGB[species % 7] ?? [0.5, 0.5, 0.5];
}

// ─── Public types ─────────────────────────────────────────────────────────────

/** A 2-D point in canvas-pixel space. */
export interface ParticleFlowPoint {
  x: number;
  y: number;
}

/** Edge definition for particle spawning. */
export interface ParticleEdgeDef {
  edgeId:       string;
  /** Start point in NDC [-1,1] space. */
  srcPos:       ParticleFlowPoint;
  /** End point in NDC [-1,1] space. */
  dstPos:       ParticleFlowPoint;
  /** Source cell species index (0–6), maps to SPECIES_COLORS. */
  srcSpecies:   number;
  /** Target cell species index (0–6). */
  dstSpecies:   number;
  /** Optional weight — drives particles-per-second (default 1). */
  weight?:      number;
}

// ─── WebGL2 Update Vertex Shader (Transform Feedback) ────────────────────────
// #version 300 es, WebGL2
// Inputs: current particle state
// Outputs captured via transform feedback: updated state
//
// Transform feedback varyings (in declaration order, matching
// gl.transformFeedbackVaryings call below):
//   tf_position, tf_velocity, tf_life, tf_color, tf_pad

const UPDATE_VERT_SRC = /* glsl */ `#version 300 es
precision highp float;

// ── per-particle inputs (current VBO) — explicit locations ─────────
layout(location = 0) in vec2  a_position;   // NDC position
layout(location = 1) in vec2  a_velocity;   // velocity (NDC units/s)
layout(location = 2) in float a_life;       // normalised life [0,1]
layout(location = 3) in vec3  a_color;      // RGB
layout(location = 4) in float a_pad;        // padding

// ── uniforms ────────────────────────────────────────────────────────
uniform float u_dt;          // delta time (seconds)
uniform float u_time;        // total elapsed (seconds)

// ── edge table: up to 64 edges ────────────────────────────────────
uniform int   u_edgeCount;
uniform vec2  u_edgeSrc[64];
uniform vec2  u_edgeDst[64];
uniform vec3  u_edgeSrcColor[64];
uniform vec3  u_edgeDstColor[64];

// ── transform feedback outputs ─────────────────────────────────────
out vec2  tf_position;
out vec2  tf_velocity;
out float tf_life;
out vec3  tf_color;
out float tf_pad;

// ── simple hash for pseudo-random respawn ─────────────────────────
float hash(float n) {
  return fract(sin(n * 127.1 + u_time * 0.1) * 43758.5453123);
}

void main() {
  // gl_VertexID is available in WebGL2
  int pidx = gl_VertexID;
  // Determine edge for this particle
  int eidx = 0;
  if (u_edgeCount > 0) {
    eidx = int(mod(float(pidx), float(max(u_edgeCount, 1))));
    eidx = clamp(eidx, 0, u_edgeCount - 1);
  }

  vec2  src      = u_edgeSrc[eidx];
  vec2  dst      = u_edgeDst[eidx];
  vec3  srcColor = u_edgeSrcColor[eidx];
  vec3  dstColor = u_edgeDstColor[eidx];

  // Edge direction (unit vector)
  vec2 edgeDir = dst - src;
  float edgeLen = length(edgeDir);
  vec2  edgeUnit = (edgeLen > 0.0001) ? edgeDir / edgeLen : vec2(1.0, 0.0);
  vec2  edgePerp = vec2(-edgeUnit.y, edgeUnit.x);

  float PARTICLE_LIFETIME_CONST = 2.5;
  float newLife = a_life - u_dt / PARTICLE_LIFETIME_CONST;

  vec2  newPos = a_position + a_velocity * u_dt;
  vec2  newVel = a_velocity;
  vec3  newColor = a_color;

  if (newLife <= 0.0) {
    // ── Respawn on the source end with slight random lateral offset ──
    float rnd0 = hash(float(pidx) * 3.7 + 0.1);
    float rnd1 = hash(float(pidx) * 7.3 + 0.3);
    float rnd2 = hash(float(pidx) * 13.1 + 0.7);

    // Start near src with tiny perpendicular jitter
    newPos  = src + edgePerp * (rnd0 - 0.5) * 0.04;

    // Velocity: edge direction + small lateral noise
    float speed = 0.15 + rnd1 * 0.25;
    newVel  = edgeUnit * speed + edgePerp * (rnd2 - 0.5) * 0.02;

    // Life reset [0.6..1.0] — stagger so they don't all die at once
    newLife = 0.6 + rnd0 * 0.4;

    // Source species color at birth
    newColor = srcColor;

  } else {
    // ── Alive: compute travel fraction t along edge ──────────────────
    vec2  relPos = newPos - src;
    float along  = dot(relPos, edgeUnit);
    float t      = (edgeLen > 0.0001) ? clamp(along / edgeLen, 0.0, 1.0) : 0.0;

    // Color gradient: source → target species
    newColor = mix(srcColor, dstColor, t);

    // Soft wrap: if particle has passed the target end, let life drain
    if (t >= 0.98) {
      newLife -= u_dt * 3.0; // fade quickly at destination
    }

    // Slight drag so particles don't accelerate unboundedly
    newVel = newVel * (1.0 - 0.4 * u_dt);
    // Keep velocity along edge direction (don't let lateral drift dominate)
    float vAlong = dot(newVel, edgeUnit);
    vAlong = max(vAlong, 0.05); // minimum forward speed
    newVel = edgeUnit * vAlong + edgePerp * dot(newVel, edgePerp) * 0.3;
  }

  tf_position = newPos;
  tf_velocity = newVel;
  tf_life     = newLife;
  tf_color    = newColor;
  tf_pad      = 0.0;

  // Update vertex shader: no rasterisation output needed
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
  gl_PointSize = 1.0;
}
`;

// ─── Render Vertex Shader ─────────────────────────────────────────────────────

const RENDER_VERT_SRC = /* glsl */ `#version 300 es
precision highp float;

// ── per-particle inputs (from read VBO after TF swap) — explicit locations ──
layout(location = 0) in vec2  a_position;
layout(location = 1) in vec2  a_velocity;
layout(location = 2) in float a_life;
layout(location = 3) in vec3  a_color;
layout(location = 4) in float a_pad;

// ── uniforms ────────────────────────────────────────────────────────
uniform vec2  u_resolution;  // canvas (w, h) in pixels
uniform float u_time;

// ── to fragment ─────────────────────────────────────────────────────
out vec4  v_color;
out float v_life;

void main() {
  // Cull dead particles by moving them off-screen
  if (a_life <= 0.0) {
    gl_Position  = vec4(-10.0, -10.0, 0.0, 1.0);
    gl_PointSize = 0.0;
    v_color      = vec4(0.0);
    v_life       = 0.0;
    return;
  }

  gl_Position = vec4(a_position, 0.0, 1.0);

  // Point size: larger near peak life, shrinks as particle fades
  float POINT_SIZE_BASE = 4.0;
  float sz = POINT_SIZE_BASE * (0.5 + 0.5 * a_life);
  // Speed-based size boost
  float spd = length(a_velocity);
  sz += spd * 6.0;
  gl_PointSize = clamp(sz, 1.0, 10.0);

  // Pre-multiply alpha by life for natural fade
  float alpha = smoothstep(0.0, 0.15, a_life) * a_life;
  v_color = vec4(a_color, alpha);
  v_life  = a_life;
}
`;

// ─── Render Fragment Shader ───────────────────────────────────────────────────

const RENDER_FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;

in  vec4  v_color;
in  float v_life;
out vec4  outColor;

void main() {
  // Round soft particle — discard corners of the point sprite
  vec2 uv   = gl_PointCoord * 2.0 - 1.0;
  float r   = dot(uv, uv);
  if (r > 1.0) discard;

  // Gaussian-ish radial falloff
  float alpha = exp(-r * 3.0);

  // Additive glow: brighter core, soft edge
  vec3  glow  = v_color.rgb * (1.0 + (1.0 - r) * 0.5);
  float a     = v_color.a * alpha;

  outColor = vec4(glow * a, a);
}
`;

// ─── Update Fragment Shader (dummy — TF pass, no rasterisation) ──────────────

const UPDATE_FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;
out vec4 dummy;
void main() { dummy = vec4(0.0); }
`;

// ─── ParticleGPU class ────────────────────────────────────────────────────────

export class ParticleGPU {
  private gl: WebGL2RenderingContext;
  private canvas: HTMLCanvasElement;

  // WebGL2 programs
  private updateProg!: WebGLProgram;
  private renderProg!: WebGLProgram;

  // VAOs for ping-pong
  private vaoRead!:  WebGLVertexArrayObject;
  private vaoWrite!: WebGLVertexArrayObject;

  // VBOs (ping-pong A / B)
  private vboA!: WebGLBuffer;
  private vboB!: WebGLBuffer;

  // Transform Feedback objects
  private tfA!: WebGLTransformFeedback;
  private tfB!: WebGLTransformFeedback;

  // Edge table (up to MAX_EDGES)
  private edges: ParticleEdgeDef[] = [];

  // Elapsed time
  private elapsed = 0;

  // Uniform locations — update program
  private uLoc = {
    dt:          null as WebGLUniformLocation | null,
    time:        null as WebGLUniformLocation | null,
    edgeCount:   null as WebGLUniformLocation | null,
    edgeSrc:     null as WebGLUniformLocation | null,
    edgeDst:     null as WebGLUniformLocation | null,
    edgeSrcCol:  null as WebGLUniformLocation | null,
    edgeDstCol:  null as WebGLUniformLocation | null,
  };

  // Uniform locations — render program
  private rLoc = {
    resolution:  null as WebGLUniformLocation | null,
    time:        null as WebGLUniformLocation | null,
  };

  constructor(canvas: HTMLCanvasElement, edges: ParticleEdgeDef[] = []) {
    const gl = canvas.getContext('webgl2');
    if (!gl) {
      throw new Error('[ParticleGPU] WebGL2 not available on this canvas.');
    }
    this.gl     = gl;
    this.canvas = canvas;
    this.edges  = edges.slice(0, MAX_EDGES);

    this._init();
  }

  // ─── Public API ───────────────────────────────────────────────────────────

  /** Add or replace an edge in the flow table. */
  addEdge(edge: ParticleEdgeDef): void {
    const idx = this.edges.findIndex(e => e.edgeId === edge.edgeId);
    if (idx >= 0) {
      this.edges[idx] = edge;
    } else if (this.edges.length < MAX_EDGES) {
      this.edges.push(edge);
    }
  }

  /** Remove an edge by id. */
  removeEdge(edgeId: string): void {
    this.edges = this.edges.filter(e => e.edgeId !== edgeId);
  }

  /**
   * Update pass — runs GPU Transform Feedback to advance all 5000 particles.
   * Call once per frame before render().
   */
  update(dt: number): void {
    const gl = this.gl;
    this.elapsed += dt;

    gl.useProgram(this.updateProg);

    // Bind uniforms
    gl.uniform1f(this.uLoc.dt,        dt);
    gl.uniform1f(this.uLoc.time,      this.elapsed);
    gl.uniform1i(this.uLoc.edgeCount, this.edges.length);

    // Upload edge table (up to MAX_EDGES)
    const srcXY  = new Float32Array(MAX_EDGES * 2);
    const dstXY  = new Float32Array(MAX_EDGES * 2);
    const srcCol = new Float32Array(MAX_EDGES * 3);
    const dstCol = new Float32Array(MAX_EDGES * 3);

    for (let i = 0; i < this.edges.length; i++) {
      const e = this.edges[i];
      srcXY[i * 2]     = e.srcPos.x;
      srcXY[i * 2 + 1] = e.srcPos.y;
      dstXY[i * 2]     = e.dstPos.x;
      dstXY[i * 2 + 1] = e.dstPos.y;
      const sc = speciesRGB(e.srcSpecies);
      const tc = speciesRGB(e.dstSpecies);
      srcCol[i * 3]     = sc[0]; srcCol[i * 3 + 1] = sc[1]; srcCol[i * 3 + 2] = sc[2];
      dstCol[i * 3]     = tc[0]; dstCol[i * 3 + 1] = tc[1]; dstCol[i * 3 + 2] = tc[2];
    }
    // Pad remaining with first edge values (prevent undefined uniform reads)
    if (this.edges.length === 0) {
      srcXY[0] = -0.5; srcXY[1] = 0.0;
      dstXY[0] =  0.5; dstXY[1] = 0.0;
      srcCol[0] = 0.247; srcCol[1] = 0.318; srcCol[2] = 0.710;
      dstCol[0] = 1.000; dstCol[1] = 0.435; dstCol[2] = 0.000;
    }

    gl.uniform2fv(this.uLoc.edgeSrc,    srcXY);
    gl.uniform2fv(this.uLoc.edgeDst,    dstXY);
    gl.uniform3fv(this.uLoc.edgeSrcCol, srcCol);
    gl.uniform3fv(this.uLoc.edgeDstCol, dstCol);

    // ── Transform Feedback pass ──────────────────────────────────────────────
    // Bind the READ VAO as input
    gl.bindVertexArray(this.vaoRead);

    // Bind the WRITE TF object (output goes to writeVBO)
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this._writeTF());

    // Disable rasterisation — we only want TF output
    gl.enable(gl.RASTERIZER_DISCARD);

    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, PARTICLE_COUNT);
    gl.endTransformFeedback();

    gl.disable(gl.RASTERIZER_DISCARD);

    // Unbind
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindVertexArray(null);

    // Swap ping-pong buffers
    this._swap();
  }

  /**
   * Render pass — draws all 5000 particles as gl.POINTS.
   * Call after update().
   */
  render(canvasW: number, canvasH: number): void {
    const gl = this.gl;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasW, canvasH);

    // Additive blending for glow
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    gl.useProgram(this.renderProg);
    gl.uniform2f(this.rLoc.resolution!, canvasW, canvasH);
    gl.uniform1f(this.rLoc.time!,       this.elapsed);

    // Bind READ VAO (post-swap, now holds the updated data)
    gl.bindVertexArray(this.vaoRead);

    gl.drawArrays(gl.POINTS, 0, PARTICLE_COUNT);

    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }

  /** Free all GPU resources. */
  destroy(): void {
    const gl = this.gl;
    gl.deleteProgram(this.updateProg);
    gl.deleteProgram(this.renderProg);
    gl.deleteVertexArray(this.vaoRead);
    gl.deleteVertexArray(this.vaoWrite);
    gl.deleteBuffer(this.vboA);
    gl.deleteBuffer(this.vboB);
    gl.deleteTransformFeedback(this.tfA);
    gl.deleteTransformFeedback(this.tfB);
  }

  // ─── Private: initialisation ──────────────────────────────────────────────

  private _init(): void {
    this._compilePrograms();
    this._createBuffers();
    this._cacheUniformLocations();
  }

  private _compilePrograms(): void {
    // ── Update program — with Transform Feedback varyings ─────────────────
    this.updateProg = this._compileWithTF(
      UPDATE_VERT_SRC,
      UPDATE_FRAG_SRC,
      // These varyings are captured in INTERLEAVED_ATTRIBS mode
      ['tf_position', 'tf_velocity', 'tf_life', 'tf_color', 'tf_pad'],
      'update',
    );

    // ── Render program — standard draw ────────────────────────────────────
    this.renderProg = this._compile(RENDER_VERT_SRC, RENDER_FRAG_SRC, 'render');
  }

  /** Compile a WebGL2 program with transform feedback varyings declared before link. */
  private _compileWithTF(
    vertSrc: string, fragSrc: string,
    tfVaryings: string[],
    label: string,
  ): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vertSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ParticleGPU] vert compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ParticleGPU] frag compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);

    // ── Bind explicit attrib locations (must be before link) ────────────
    gl.bindAttribLocation(prog, 0, 'a_position');
    gl.bindAttribLocation(prog, 1, 'a_velocity');
    gl.bindAttribLocation(prog, 2, 'a_life');
    gl.bindAttribLocation(prog, 3, 'a_color');
    gl.bindAttribLocation(prog, 4, 'a_pad');

    // ── Declare TF varyings BEFORE linking ──────────────────────────────
    gl.transformFeedbackVaryings(prog, tfVaryings, gl.INTERLEAVED_ATTRIBS);

    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ParticleGPU] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** Compile a standard (non-TF) WebGL2 program. */
  private _compile(vertSrc: string, fragSrc: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vertSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ParticleGPU] vert compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ParticleGPU] frag compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);

    // ── Bind explicit attrib locations (must be before link) ────────────
    gl.bindAttribLocation(prog, 0, 'a_position');
    gl.bindAttribLocation(prog, 1, 'a_velocity');
    gl.bindAttribLocation(prog, 2, 'a_life');
    gl.bindAttribLocation(prog, 3, 'a_color');
    gl.bindAttribLocation(prog, 4, 'a_pad');

    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ParticleGPU] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  private _createBuffers(): void {
    const gl = this.gl;

    // ── Generate initial particle data ─────────────────────────────────────
    const data = new Float32Array(PARTICLE_COUNT * PARTICLE_STRIDE_F32);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const b = i * PARTICLE_STRIDE_F32;
      // Distribute particles randomly across the unit circle initially
      const angle = Math.random() * Math.PI * 2;
      const r     = Math.random() * 0.8;
      data[b + 0] = Math.cos(angle) * r;   // pos x
      data[b + 1] = Math.sin(angle) * r;   // pos y
      data[b + 2] = (Math.random() - 0.5) * 0.2; // vel x
      data[b + 3] = (Math.random() - 0.5) * 0.2; // vel y
      data[b + 4] = Math.random();               // life (staggered)
      const col   = speciesRGB(i % 7);
      data[b + 5] = col[0];  // r
      data[b + 6] = col[1];  // g
      data[b + 7] = col[2];  // b
      data[b + 8] = 0.0;     // pad
    }

    // ── Create VBOs ───────────────────────────────────────────────────────
    this.vboA = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboA);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_COPY);

    this.vboB = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboB);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_COPY);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // ── Create Transform Feedback objects ─────────────────────────────────
    this.tfA = gl.createTransformFeedback()!;
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tfA);
    // TF-A outputs to vboA (when vboB is read-input)
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.vboA);

    this.tfB = gl.createTransformFeedback()!;
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tfB);
    // TF-B outputs to vboB (when vboA is read-input)
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.vboB);

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);

    // ── Create VAOs ───────────────────────────────────────────────────────
    // vaoA reads from vboA (used as read-VAO when TF writes to vboB)
    this.vaoRead  = this._createParticleVAO(this.vboA);
    // vaoB reads from vboB (used as read-VAO after first swap)
    this.vaoWrite = this._createParticleVAO(this.vboB);
  }

  /**
   * Create a VAO that binds the given VBO with explicit attrib locations.
   *
   * Both programs (update + render) have their attribute locations pre-bound
   * via gl.bindAttribLocation before linking, so they share these indices:
   *   0: a_position (vec2)  @ byte offset  0
   *   1: a_velocity (vec2)  @ byte offset  8
   *   2: a_life     (float) @ byte offset 16
   *   3: a_color    (vec3)  @ byte offset 20
   *   4: a_pad      (float) @ byte offset 32
   */
  private _createParticleVAO(vbo: WebGLBuffer): WebGLVertexArrayObject {
    const gl  = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);

    // location 0: a_position — vec2, offset 0
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, PARTICLE_STRIDE_BYTES, 0);

    // location 1: a_velocity — vec2, offset 8
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, PARTICLE_STRIDE_BYTES, 8);

    // location 2: a_life — float, offset 16
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, PARTICLE_STRIDE_BYTES, 16);

    // location 3: a_color — vec3, offset 20
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 3, gl.FLOAT, false, PARTICLE_STRIDE_BYTES, 20);

    // location 4: a_pad — float, offset 32
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 1, gl.FLOAT, false, PARTICLE_STRIDE_BYTES, 32);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
    return vao;
  }

  private _cacheUniformLocations(): void {
    const gl = this.gl;
    const u  = this.updateProg;
    this.uLoc.dt          = gl.getUniformLocation(u, 'u_dt');
    this.uLoc.time        = gl.getUniformLocation(u, 'u_time');
    this.uLoc.edgeCount   = gl.getUniformLocation(u, 'u_edgeCount');
    this.uLoc.edgeSrc     = gl.getUniformLocation(u, 'u_edgeSrc');
    this.uLoc.edgeDst     = gl.getUniformLocation(u, 'u_edgeDst');
    this.uLoc.edgeSrcCol  = gl.getUniformLocation(u, 'u_edgeSrcColor');
    this.uLoc.edgeDstCol  = gl.getUniformLocation(u, 'u_edgeDstColor');

    const r = this.renderProg;
    this.rLoc.resolution = gl.getUniformLocation(r, 'u_resolution');
    this.rLoc.time       = gl.getUniformLocation(r, 'u_time');
  }

  // ─── Ping-pong state ──────────────────────────────────────────────────────

  /** true = vaoA/vboA is read-input, tfB/vboB is write-output */
  private _pingA = true;

  /** Return the TF object whose output buffer is the WRITE target this frame. */
  private _writeTF(): WebGLTransformFeedback {
    return this._pingA ? this.tfB : this.tfA;
  }

  /** After update(), swap read/write roles. */
  private _swap(): void {
    [this.vaoRead, this.vaoWrite] = [this.vaoWrite, this.vaoRead];
    this._pingA = !this._pingA;
  }
}

// ─── Factory helpers ──────────────────────────────────────────────────────────

/**
 * Build a ParticleEdgeDef from pixel-space endpoints + species indices.
 * Converts pixel coords to NDC given canvas size.
 */
export function makeEdgeDef(
  edgeId:     string,
  srcPx:      { x: number; y: number },
  dstPx:      { x: number; y: number },
  canvasW:    number,
  canvasH:    number,
  srcSpecies: number,
  dstSpecies: number,
  weight?:    number,
): ParticleEdgeDef {
  const toNDC = (px: { x: number; y: number }) => ({
    x:  (px.x / canvasW) * 2 - 1,
    y: -((px.y / canvasH) * 2 - 1),   // Y-flip: canvas Y-down → NDC Y-up
  });
  return {
    edgeId,
    srcPos:     toNDC(srcPx),
    dstPos:     toNDC(dstPx),
    srcSpecies,
    dstSpecies,
    weight,
  };
}

// ─── AT shader reference (from compiled.vs via ShaderLoader) ─────────────────
//
// The edge-spline.frag AT shader provides the canonical species color gradient
// pattern (mix(u_sourceColor, u_targetColor, v_t)).  We replicate that logic
// inline in the update & render shaders above.  The AT shader itself is still
// accessible via:
//
//   const edgeSplineFrag = getShader('edge-spline.frag');
//
// and can be used for a deferred compositing pass that blends the particle
// render target with the spline render target.  The AT vert/frag pair from
// compiled.vs is WebGL1 (varying / texture2D) while particle-gpu-pass runs
// WebGL2 (#version 300 es); they target different contexts.
//
// To use the AT shader source for reference or debugging:
//   import { getShader } from '../shaders/ShaderLoader';
//   const atSrc = getShader('edge-spline.frag');   // WebGL1 source string
//   const atVert = getShader('edge-spline.vert');  // WebGL1 source string

/** Re-export getShader for consumers that want to access AT shaders. */
export { getShader } from '../shaders/ShaderLoader';
