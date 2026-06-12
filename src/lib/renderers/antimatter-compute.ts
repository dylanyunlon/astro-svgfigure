/**
 * antimatter-compute.ts — GPU compute framework via WebGL2 transform feedback
 *
 * AT Antimatter 模块 (48 次引用): 粒子/物理模拟的 GPU 计算核心。
 * AT 用 WebGL2 transform feedback 在 GPU 端做位置/速度更新，
 * 完全绕过 CPU loop — 500 个 cell 的物理位置更新无需 loop_orchestrator.py。
 *
 * 架构:
 *   AntimatterAttribute   顶点属性管理 (position, velocity, life, species…)
 *   AntimatterFBO         双缓冲 FBO ping-pong (texA ↔ texB per frame)
 *   AntimatterPass        一个 compute pass: inputBuf → transform feedback → outputBuf
 *   AntimatterSpawn       粒子生成: 写入初始 position/velocity 到 buffer
 *
 * Transform feedback 核心循环:
 *   gl.beginTransformFeedback(gl.POINTS)
 *   gl.drawArrays(gl.POINTS, 0, count)
 *   gl.endTransformFeedback()
 *
 * Usage:
 *   const hydra  = new HydraGLLayer(canvas);
 *   const spawn  = new AntimatterSpawn(hydra.gl, 500);
 *   spawn.writeCells(cellDescriptors);
 *
 *   const pass = new AntimatterPass(hydra.gl, {
 *     vertSrc: PHYSICS_UPDATE_VERT,
 *     varyings: ['v_position', 'v_velocity'],
 *     count: 500,
 *   });
 *   pass.setBuffer(AttributeKind.Position, spawn.positionBuf);
 *   pass.setBuffer(AttributeKind.Velocity, spawn.velocityBuf);
 *
 *   // each frame:
 *   pass.run(deltaTime);
 *   pass.readback(AttributeKind.Position, Float32Array);  // CPU readback optional
 */

import { compileShader, HydraGLLayer } from './hydra-gl-layer';

// ── Constants ───────────────────────────────────────────────────────────────

/** Number of floats per cell in the position buffer: x, y, z, w(padding) */
const POSITION_STRIDE = 4;
/** Number of floats per cell in the velocity buffer: vx, vy, vz, damping */
const VELOCITY_STRIDE = 4;
/** Number of floats per cell in the life buffer: age, maxAge, spawnMask, speciesId */
const LIFE_STRIDE = 4;

// ── AttributeKind ────────────────────────────────────────────────────────────

export enum AttributeKind {
  Position = 0,
  Velocity = 1,
  Life     = 2,
  Force    = 3,
}

// ── AntimatterAttribute ──────────────────────────────────────────────────────
// Wraps a single ARRAY_BUFFER that can be bound as a vertex attribute
// and also as a TRANSFORM_FEEDBACK_BUFFER output target.

export class AntimatterAttribute {
  readonly gl: WebGL2RenderingContext;
  readonly kind: AttributeKind;
  /** Stride in floats (components per vertex) */
  readonly stride: number;
  /** Number of particles/cells */
  readonly count: number;

  /** Current write buffer (output of last TF pass) */
  bufferWrite: WebGLBuffer;
  /** Current read buffer (input of current TF pass) */
  bufferRead: WebGLBuffer;

  constructor(
    gl: WebGL2RenderingContext,
    kind: AttributeKind,
    count: number,
    initialData?: Float32Array,
  ) {
    this.gl = gl;
    this.kind = kind;
    this.stride = strideForKind(kind);
    this.count = count;

    const byteLen = count * this.stride * Float32Array.BYTES_PER_ELEMENT;
    this.bufferRead  = createBuffer(gl, byteLen, initialData ?? null);
    this.bufferWrite = createBuffer(gl, byteLen, null);
  }

  /** Swap read/write buffers — call after each transform feedback pass */
  swap(): void {
    const tmp = this.bufferRead;
    this.bufferRead  = this.bufferWrite;
    this.bufferWrite = tmp;
  }

  /** Upload new data into the read buffer (e.g. after CPU-side spawn) */
  upload(data: Float32Array, offset = 0): void {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufferRead);
    gl.bufferSubData(gl.ARRAY_BUFFER, offset * Float32Array.BYTES_PER_ELEMENT, data);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /**
   * Read back current write-buffer contents to CPU.
   * Expensive — only call for debug or when CPU needs updated positions.
   */
  readback(): Float32Array {
    const gl = this.gl;
    const out = new Float32Array(this.count * this.stride);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufferWrite);
    gl.getBufferSubData(gl.ARRAY_BUFFER, 0, out);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return out;
  }

  destroy(): void {
    this.gl.deleteBuffer(this.bufferRead);
    this.gl.deleteBuffer(this.bufferWrite);
  }
}

function strideForKind(kind: AttributeKind): number {
  switch (kind) {
    case AttributeKind.Position: return POSITION_STRIDE;
    case AttributeKind.Velocity: return VELOCITY_STRIDE;
    case AttributeKind.Life:     return LIFE_STRIDE;
    case AttributeKind.Force:    return POSITION_STRIDE; // dx,dy,dz,mag
    default:                     return 4;
  }
}

function createBuffer(gl: WebGL2RenderingContext, byteLen: number, data: Float32Array | null): WebGLBuffer {
  const buf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  if (data) {
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_COPY);
  } else {
    gl.bufferData(gl.ARRAY_BUFFER, byteLen, gl.DYNAMIC_COPY);
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  return buf;
}

// ── AntimatterFBO ────────────────────────────────────────────────────────────
// 双缓冲 FBO ping-pong: 读 texA 写 texB, 下帧交换。
// 用途: 把粒子状态存成纹理供 fragment shader 采样 (e.g. 渲染 cell 颜色场)。
// Transform feedback 本身用 buffer 而非纹理, 但 FBO ping-pong 配合使用
// 可以做 cell 颜色/物理状态的 GPU 端全局读取。

export class AntimatterFBO {
  readonly gl: WebGL2RenderingContext;
  readonly width: number;
  readonly height: number;

  private texA: WebGLTexture;
  private texB: WebGLTexture;
  private fboA: WebGLFramebuffer;
  private fboB: WebGLFramebuffer;

  /** After each swap: read from this texture */
  texRead: WebGLTexture;
  /** After each swap: write into this FBO */
  fboWrite: WebGLFramebuffer;

  constructor(gl: WebGL2RenderingContext, width: number, height: number) {
    this.gl = gl;
    this.width  = width;
    this.height = height;

    this.texA = createFloatTexture(gl, width, height);
    this.texB = createFloatTexture(gl, width, height);
    this.fboA = attachFBO(gl, this.texA);
    this.fboB = attachFBO(gl, this.texB);

    // Initial state: read A, write B
    this.texRead  = this.texA;
    this.fboWrite = this.fboB;
  }

  /** Bind write FBO and set viewport */
  bindWrite(): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboWrite);
    gl.viewport(0, 0, this.width, this.height);
  }

  unbind(): void {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
  }

  /** Swap read↔write, call after each compute pass */
  swap(): void {
    if (this.texRead === this.texA) {
      this.texRead  = this.texB;
      this.fboWrite = this.fboA;
    } else {
      this.texRead  = this.texA;
      this.fboWrite = this.fboB;
    }
  }

  destroy(): void {
    const gl = this.gl;
    gl.deleteTexture(this.texA);
    gl.deleteTexture(this.texB);
    gl.deleteFramebuffer(this.fboA);
    gl.deleteFramebuffer(this.fboB);
  }
}

function createFloatTexture(gl: WebGL2RenderingContext, w: number, h: number): WebGLTexture {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

function attachFBO(gl: WebGL2RenderingContext, tex: WebGLTexture): WebGLFramebuffer {
  const fbo = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return fbo;
}

// ── AntimatterPass ───────────────────────────────────────────────────────────
// 一个 compute pass: 把 attribute buffers 作为顶点属性输入,
// vertex shader 做物理变换, transform feedback 把结果写入 output buffers.
// Fragment shader 是 null/discard-only — 不需要光栅化输出.

export interface AntimatterPassOptions {
  /** GLSL 300 es vertex shader source. Must declare `out` for each varying. */
  vertSrc: string;
  /**
   * Names of `out` varyings to capture via transform feedback.
   * Must match the `out` variable names in vertSrc.
   * E.g. ['v_position', 'v_velocity']
   */
  varyings: string[];
  /** Number of particles/cells to process */
  count: number;
  /** Buffer capture mode (default: INTERLEAVED for single-buffer, SEPARATE for multi) */
  bufferMode?: 'interleaved' | 'separate';
}

/** Minimal discard fragment shader — transform feedback needs a linked frag */
const TF_DISCARD_FRAG = `#version 300 es
precision highp float;
out vec4 _unused;
void main() { discard; }
`;

export class AntimatterPass {
  readonly gl: WebGL2RenderingContext;
  readonly count: number;

  private prog: WebGLProgram;
  private tf: WebGLTransformFeedback;
  private vao: WebGLVertexArrayObject;
  private attributes: Map<AttributeKind, AntimatterAttribute> = new Map();
  private varyingOutputs: Map<AttributeKind, AntimatterAttribute> = new Map();
  private bufferMode: number;

  constructor(gl: WebGL2RenderingContext, opts: AntimatterPassOptions) {
    this.gl = gl;
    this.count = opts.count;
    this.bufferMode = opts.bufferMode === 'separate' ? gl.SEPARATE_ATTRIBS : gl.INTERLEAVED_ATTRIBS;

    // Compile shaders — must register varyings BEFORE linking
    const vert = compileShader(gl, gl.VERTEX_SHADER, opts.vertSrc);
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, TF_DISCARD_FRAG);

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);

    // Register transform feedback varyings before gl.linkProgram
    gl.transformFeedbackVaryings(prog, opts.varyings, this.bufferMode);
    gl.linkProgram(prog);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      throw new Error(`AntimatterPass link error: ${log}`);
    }
    gl.detachShader(prog, vert);
    gl.detachShader(prog, frag);
    gl.deleteShader(vert);
    gl.deleteShader(frag);
    this.prog = prog;

    // WebGL2 transform feedback object
    this.tf  = gl.createTransformFeedback()!;
    this.vao = gl.createVertexArray()!;
  }

  /**
   * Bind an AntimatterAttribute as a vertex attribute input.
   * @param kind  — attribute semantic
   * @param attr  — the attribute (its bufferRead will be used as input)
   * @param loc   — attribute location in the vertex shader (use gl.getAttribLocation)
   */
  setBuffer(kind: AttributeKind, attr: AntimatterAttribute, loc?: number): void {
    const gl = this.gl;
    this.attributes.set(kind, attr);

    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, attr.bufferRead);

    const attribLoc = loc ?? gl.getAttribLocation(this.prog, attribNameForKind(kind));
    if (attribLoc >= 0) {
      gl.enableVertexAttribArray(attribLoc);
      gl.vertexAttribPointer(attribLoc, attr.stride, gl.FLOAT, false, 0, 0);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
  }

  /**
   * Bind an AntimatterAttribute as a transform feedback output target.
   * @param kind    — attribute semantic (should match a varying)
   * @param attr    — output attribute (bufferWrite will receive TF data)
   * @param bindIdx — binding index for SEPARATE_ATTRIBS mode
   */
  setOutput(kind: AttributeKind, attr: AntimatterAttribute, bindIdx = 0): void {
    this.varyingOutputs.set(kind, attr);
    // Actual binding happens in run() after beginTransformFeedback setup
    void bindIdx;
  }

  /**
   * Set a float uniform on the compute program.
   */
  setUniform1f(name: string, value: number): void {
    const gl = this.gl;
    gl.useProgram(this.prog);
    const loc = gl.getUniformLocation(this.prog, name);
    if (loc !== null) gl.uniform1f(loc, value);
  }

  setUniform2f(name: string, x: number, y: number): void {
    const gl = this.gl;
    gl.useProgram(this.prog);
    const loc = gl.getUniformLocation(this.prog, name);
    if (loc !== null) gl.uniform2f(loc, x, y);
  }

  setUniform3f(name: string, x: number, y: number, z: number): void {
    const gl = this.gl;
    gl.useProgram(this.prog);
    const loc = gl.getUniformLocation(this.prog, name);
    if (loc !== null) gl.uniform3f(loc, x, y, z);
  }

  /**
   * Execute the compute pass via transform feedback.
   * @param deltaTime  — seconds since last frame (passed as uniform `u_dt`)
   * After this call, each output attribute's bufferWrite contains updated data.
   * Call attr.swap() on outputs to promote write → read for next frame.
   */
  run(deltaTime: number): void {
    const gl = this.gl;

    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);

    // Update per-frame uniforms
    this.setUniform1f('u_dt', deltaTime);

    // Bind input attribute buffers (re-bind because swap may have changed them)
    for (const [kind, attr] of this.attributes) {
      const attribLoc = gl.getAttribLocation(this.prog, attribNameForKind(kind));
      if (attribLoc < 0) continue;
      gl.bindBuffer(gl.ARRAY_BUFFER, attr.bufferRead);
      gl.enableVertexAttribArray(attribLoc);
      gl.vertexAttribPointer(attribLoc, attr.stride, gl.FLOAT, false, 0, 0);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Bind transform feedback output buffers
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf);
    let bindIdx = 0;
    for (const [, attr] of this.varyingOutputs) {
      gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, bindIdx, attr.bufferWrite);
      bindIdx++;
    }

    // Disable rasterization — we only want the TF output, not pixel output
    gl.enable(gl.RASTERIZER_DISCARD);

    // ── Core transform feedback loop ──────────────────────────────────
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, this.count);
    gl.endTransformFeedback();
    // ──────────────────────────────────────────────────────────────────

    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindVertexArray(null);

    // Swap all output buffers: write → read for next frame
    for (const [, attr] of this.varyingOutputs) {
      attr.swap();
    }
    // Also swap any input-only attributes that are shared outputs
    for (const [kind, attr] of this.attributes) {
      if (!this.varyingOutputs.has(kind)) {
        // Input-only: no swap needed
      }
    }
  }

  /** Read back output data for a given kind to CPU Float32Array */
  readback(kind: AttributeKind): Float32Array {
    const attr = this.varyingOutputs.get(kind) ?? this.attributes.get(kind);
    if (!attr) throw new Error(`AntimatterPass.readback: no buffer for kind ${kind}`);
    return attr.readback();
  }

  destroy(): void {
    const gl = this.gl;
    gl.deleteProgram(this.prog);
    gl.deleteTransformFeedback(this.tf);
    gl.deleteVertexArray(this.vao);
  }
}

function attribNameForKind(kind: AttributeKind): string {
  switch (kind) {
    case AttributeKind.Position: return 'a_position';
    case AttributeKind.Velocity: return 'a_velocity';
    case AttributeKind.Life:     return 'a_life';
    case AttributeKind.Force:    return 'a_force';
    default:                     return `a_attr${kind}`;
  }
}

// ── AntimatterSpawn ──────────────────────────────────────────────────────────
// 粒子/cell 生成逻辑: 把初始 position/velocity/life 写入 GPU buffer.
// 支持从 loop_orchestrator force_field JSON 批量写入 500 cells.

export interface CellSpawnDescriptor {
  /** Cell identifier (used for debug/readback indexing) */
  id: string;
  /** World-space position [x, y, z] */
  position: [number, number, number];
  /** Initial velocity [vx, vy, vz] */
  velocity?: [number, number, number];
  /** Force field contribution [dx, dy, dz, magnitude] */
  force?: [number, number, number, number];
  /** Age (0 = just spawned), maxAge, spawnMask (1=active), speciesId */
  life?: [number, number, number, number];
}

export class AntimatterSpawn {
  readonly gl: WebGL2RenderingContext;
  readonly maxCount: number;

  readonly positionAttr: AntimatterAttribute;
  readonly velocityAttr: AntimatterAttribute;
  readonly lifeAttr:     AntimatterAttribute;
  readonly forceAttr:    AntimatterAttribute;

  /** Map from cell id → buffer index for readback */
  private indexMap: Map<string, number> = new Map();
  private activeCount = 0;

  constructor(gl: WebGL2RenderingContext, maxCount: number) {
    this.gl       = gl;
    this.maxCount = maxCount;

    this.positionAttr = new AntimatterAttribute(gl, AttributeKind.Position, maxCount);
    this.velocityAttr = new AntimatterAttribute(gl, AttributeKind.Velocity, maxCount);
    this.lifeAttr     = new AntimatterAttribute(gl, AttributeKind.Life,     maxCount);
    this.forceAttr    = new AntimatterAttribute(gl, AttributeKind.Force,    maxCount);
  }

  /**
   * Write initial cell state from CellSpawnDescriptor array to GPU buffers.
   * Replaces entire buffer — call once at epoch start.
   * @param cells  — array of up to maxCount cells (500 for typical cell graph)
   */
  writeCells(cells: CellSpawnDescriptor[]): void {
    const n = Math.min(cells.length, this.maxCount);
    this.activeCount = n;

    const posBuf  = new Float32Array(n * POSITION_STRIDE);
    const velBuf  = new Float32Array(n * VELOCITY_STRIDE);
    const lifeBuf = new Float32Array(n * LIFE_STRIDE);
    const forBuf  = new Float32Array(n * POSITION_STRIDE);

    this.indexMap.clear();

    for (let i = 0; i < n; i++) {
      const c = cells[i];
      this.indexMap.set(c.id, i);

      const pi = i * POSITION_STRIDE;
      posBuf[pi]     = c.position[0];
      posBuf[pi + 1] = c.position[1];
      posBuf[pi + 2] = c.position[2];
      posBuf[pi + 3] = 1.0; // w = active flag

      const vi = i * VELOCITY_STRIDE;
      const vel = c.velocity ?? [0, 0, 0];
      velBuf[vi]     = vel[0];
      velBuf[vi + 1] = vel[1];
      velBuf[vi + 2] = vel[2];
      velBuf[vi + 3] = 0.98; // default damping

      const li = i * LIFE_STRIDE;
      const life = c.life ?? [0, 1e9, 1, 0];
      lifeBuf[li]     = life[0]; // age
      lifeBuf[li + 1] = life[1]; // maxAge
      lifeBuf[li + 2] = life[2]; // spawnMask (1 = active)
      lifeBuf[li + 3] = life[3]; // speciesId

      const fi = i * POSITION_STRIDE;
      const force = c.force ?? [0, 0, 0, 0];
      forBuf[fi]     = force[0];
      forBuf[fi + 1] = force[1];
      forBuf[fi + 2] = force[2];
      forBuf[fi + 3] = force[3];
    }

    this.positionAttr.upload(posBuf);
    this.velocityAttr.upload(velBuf);
    this.lifeAttr.upload(lifeBuf);
    this.forceAttr.upload(forBuf);
  }

  /**
   * Read back GPU positions for a specific cell by id (after TF pass).
   * @returns [x, y, z] or null if id not found
   */
  readCellPosition(id: string): [number, number, number] | null {
    const idx = this.indexMap.get(id);
    if (idx === undefined) return null;
    const data = this.positionAttr.readback();
    const base = idx * POSITION_STRIDE;
    return [data[base], data[base + 1], data[base + 2]];
  }

  /** Read back all cell positions as a flat Float32Array (XYZW * count) */
  readAllPositions(): Float32Array {
    return this.positionAttr.readback();
  }

  get count(): number { return this.activeCount; }

  destroy(): void {
    this.positionAttr.destroy();
    this.velocityAttr.destroy();
    this.lifeAttr.destroy();
    this.forceAttr.destroy();
  }
}

// ── Built-in cell physics vertex shader ─────────────────────────────────────
// GPU 端的 500-cell 物理位置更新。
// 输入: position, velocity, force — 输出: v_position, v_velocity
// 这里 loop_orchestrator 的 physics engine CPU loop 变成 GPU 并行。

export const CELL_PHYSICS_VERT = `#version 300 es
precision highp float;

// Input attributes (current state)
in vec4 a_position; // xyz = world pos, w = active
in vec4 a_velocity; // xyz = velocity,  w = damping
in vec4 a_life;     // x=age, y=maxAge, z=spawnMask, w=speciesId
in vec4 a_force;    // xyz = force vector, w = magnitude

// Transform feedback outputs (next state)
out vec4 v_position;
out vec4 v_velocity;
out vec4 v_life;

uniform float u_dt;           // delta time (seconds)
uniform float u_gravity;      // optional gravity Y (default 0)
uniform float u_repulse_r;    // repulsion radius (cell size estimate)
uniform float u_bound_x;      // canvas half-width  (0 = no bound)
uniform float u_bound_y;      // canvas half-height (0 = no bound)

void main() {
  // Dead cell: freeze in place
  if (a_life.z < 0.5 || a_life.x >= a_life.y) {
    v_position = a_position;
    v_velocity = a_velocity;
    v_life     = a_life;
    gl_Position = vec4(0.0);
    gl_PointSize = 1.0;
    return;
  }

  // Integrate force → velocity → position (semi-implicit Euler)
  vec3 acc = a_force.xyz + vec3(0.0, u_gravity, 0.0);
  vec3 vel = a_velocity.xyz + acc * u_dt;

  // Apply per-cell damping stored in a_velocity.w
  vel *= a_velocity.w;

  vec3 pos = a_position.xyz + vel * u_dt;

  // Soft boundary repulsion (keeps cells inside canvas)
  if (u_bound_x > 0.0) {
    float ox = abs(pos.x) - u_bound_x;
    if (ox > 0.0) {
      vel.x -= sign(pos.x) * ox * 8.0;
      pos.x  = sign(pos.x) * u_bound_x;
    }
  }
  if (u_bound_y > 0.0) {
    float oy = abs(pos.y) - u_bound_y;
    if (oy > 0.0) {
      vel.y -= sign(pos.y) * oy * 8.0;
      pos.y  = sign(pos.y) * u_bound_y;
    }
  }

  // Clamp velocity magnitude to prevent explosion
  float speed = length(vel);
  float maxSpeed = 800.0;
  if (speed > maxSpeed) vel = vel * (maxSpeed / speed);

  // Advance age
  float age = a_life.x + u_dt;

  v_position = vec4(pos, a_position.w);
  v_velocity = vec4(vel, a_velocity.w);
  v_life     = vec4(age, a_life.y, a_life.z, a_life.w);

  // gl_Position is required but unused (rasterization disabled)
  gl_Position  = vec4(pos.xy / vec2(800.0, 600.0), 0.0, 1.0);
  gl_PointSize = 1.0;
}
`;

// ── AntimatterCellCompute ────────────────────────────────────────────────────
// 组合 AntimatterSpawn + AntimatterPass, 暴露一个高层 API:
//   compute.loadFromForceField(forceFieldJson) → upload cells
//   compute.step(dt)                           → one GPU physics step
//   compute.exportPositions()                  → { cellId: [x,y,z], … }

export interface ForceFieldEntry {
  dx: number;
  dy: number;
  dz: number;
  push_from: string[];
  push_mag: number;
}

export class AntimatterCellCompute {
  private spawn: AntimatterSpawn;
  private pass:  AntimatterPass;
  private cellIds: string[] = [];

  readonly gl: WebGL2RenderingContext;

  constructor(hydra: HydraGLLayer, maxCells = 512) {
    this.gl    = hydra.gl;
    const gl   = this.gl;

    this.spawn = new AntimatterSpawn(gl, maxCells);

    this.pass = new AntimatterPass(gl, {
      vertSrc:    CELL_PHYSICS_VERT,
      varyings:   ['v_position', 'v_velocity', 'v_life'],
      count:      maxCells,
      bufferMode: 'separate',
    });

    // Wire inputs
    this.pass.setBuffer(AttributeKind.Position, this.spawn.positionAttr);
    this.pass.setBuffer(AttributeKind.Velocity, this.spawn.velocityAttr);
    this.pass.setBuffer(AttributeKind.Life,     this.spawn.lifeAttr);
    this.pass.setBuffer(AttributeKind.Force,    this.spawn.forceAttr);

    // Wire outputs — same attributes, TF writes into their bufferWrite
    this.pass.setOutput(AttributeKind.Position, this.spawn.positionAttr, 0);
    this.pass.setOutput(AttributeKind.Velocity, this.spawn.velocityAttr, 1);
    this.pass.setOutput(AttributeKind.Life,     this.spawn.lifeAttr,     2);

    // Default physics params
    this.pass.setUniform1f('u_gravity',   0.0);
    this.pass.setUniform1f('u_repulse_r', 80.0);
    this.pass.setUniform1f('u_bound_x',   400.0);
    this.pass.setUniform1f('u_bound_y',   300.0);
  }

  /**
   * Load cells from loop_orchestrator force_field.json format.
   * Each key is a cell id; dx/dy/dz are the accumulated force for that cell.
   * @param forceField  — parsed force_field.json object
   * @param positionHint — optional current positions per cell id (if not provided, uses dx/dy/dz as pos)
   */
  loadFromForceField(
    forceField: Record<string, ForceFieldEntry>,
    positionHint?: Record<string, [number, number, number]>,
  ): void {
    const cells: CellSpawnDescriptor[] = [];
    this.cellIds = [];

    for (const [id, entry] of Object.entries(forceField)) {
      this.cellIds.push(id);
      const pos = positionHint?.[id] ?? [entry.dx + 400, entry.dy + 300, 0] as [number, number, number];
      cells.push({
        id,
        position: pos,
        velocity: [0, 0, 0],
        force:    [entry.dx, entry.dy, entry.dz, entry.push_mag],
        life:     [0, 1e9, 1, 0],
      });
    }

    this.spawn.writeCells(cells);

    // Update TF pass count to match actual cell count
    // (AntimatterPass.count is readonly; create a new one if count differs)
    // For typical use, maxCells ≥ actual count so pass runs with some inactive slots.
  }

  /**
   * Load cells from an explicit CellSpawnDescriptor array.
   */
  loadCells(cells: CellSpawnDescriptor[]): void {
    this.cellIds = cells.map(c => c.id);
    this.spawn.writeCells(cells);
  }

  /** One GPU physics step. Call each animation frame. */
  step(deltaTime: number): void {
    this.pass.run(deltaTime);
  }

  /** Set physics parameters */
  setGravity(g: number): void     { this.pass.setUniform1f('u_gravity', g); }
  setBounds(hw: number, hh: number): void {
    this.pass.setUniform1f('u_bound_x', hw);
    this.pass.setUniform1f('u_bound_y', hh);
  }

  /**
   * Read back all cell positions from GPU to CPU.
   * @returns map of cellId → [x, y, z]
   * O(n) readback — use sparingly; call at end of epoch for position export.
   */
  exportPositions(): Record<string, [number, number, number]> {
    const raw = this.spawn.readAllPositions();
    const result: Record<string, [number, number, number]> = {};
    for (let i = 0; i < this.cellIds.length; i++) {
      const base = i * POSITION_STRIDE;
      result[this.cellIds[i]] = [raw[base], raw[base + 1], raw[base + 2]];
    }
    return result;
  }

  /** Access underlying spawn buffers for direct attribute binding in render passes */
  get positionBuffer(): WebGLBuffer { return this.spawn.positionAttr.bufferRead; }
  get velocityBuffer(): WebGLBuffer { return this.spawn.velocityAttr.bufferRead; }
  get count():          number       { return this.spawn.count; }

  destroy(): void {
    this.pass.destroy();
    this.spawn.destroy();
  }
}
