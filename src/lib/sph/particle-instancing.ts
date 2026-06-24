/**
 * particle-instancing.ts — M745: GPU instanced particle rendering
 * ─────────────────────────────────────────────────────────────────────────────
 * Replaces the per-particle Canvas2D draw loop in world-renderer.ts with a
 * single-draw-call WebGL2 instanced pipeline.
 *
 * Architecture
 * ─────────────────────────────────────────────────────────────────────────────
 * Each particle is a screen-aligned quad (2 triangles, 6 vertices) drawn via
 * `gl.drawArraysInstanced(TRIANGLES, 0, 6, count)`.  Per-instance data is
 * packed into a single interleaved Float32Array:
 *
 *   [ posX, posY, velX, velY, species(float), density ]   — 6 floats / particle
 *
 * The vertex shader expands the quad in clip-space using the projection matrix
 * and a speed-dependent radius.  The fragment shader renders an anti-aliased
 * disc with species-based colour and a soft velocity glow.
 *
 * Usage
 * ─────────────────────────────────────────────────────────────────────────────
 *   const instancer = new ParticleInstancer(50000);
 *   instancer.attach(canvas);
 *
 *   // Each frame:
 *   instancer.updateBuffer(packedFloat32, liveCount);
 *   instancer.render(projectionMatrix);
 *
 *   // Teardown:
 *   instancer.dispose();
 *
 * Data packing helper
 * ─────────────────────────────────────────────────────────────────────────────
 * Use `packParticleData(particles)` to convert a `Particle[]` (world-stepper)
 * into the interleaved Float32Array expected by `updateBuffer`.
 *
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   upstream/webgl2-particles   — transform feedback instancing pattern
 *   src/lib/sph/ParticleRenderer.ts — WebGPU instanced quads (design parallel)
 *   src/lib/sph/world-renderer.ts   — Canvas2D renderer this replaces
 *   src/lib/sph/types.ts            — MAX_PARTICLES, ParticleData
 */




// ─── Stride ────────────────────────────────────────────────────────────────
/** Floats per particle instance in the interleaved buffer. */



import { MAX_PARTICLES } from './types';

export const INSTANCE_STRIDE = 6; // posX, posY, velX, velY, species, density

// ─── Species colour palette (matches world-renderer SPECIES_COLORS) ────────
// Encoded as normalised RGB triplets for the shader.
const SPECIES_PALETTE: [number, number, number][] = [
  [0.247, 0.318, 0.710],  // 0  #3F51B5
  [1.000, 0.435, 0.000],  // 1  #FF6F00
  [0.180, 0.490, 0.196],  // 2  #2E7D32
  [0.776, 0.157, 0.157],  // 3  #C62828
  [0.271, 0.353, 0.392],  // 4  #455A64
  [0.482, 0.122, 0.635],  // 5  #7B1FA2
  [0.084, 0.396, 0.753],  // 6  #1565C0
  [1.000, 1.000, 1.000],  // 7  fallback white
];

// ─── Shader source ─────────────────────────────────────────────────────────

const VERT_SRC = /* glsl */ `#version 300 es
precision highp float;

// ── Projection matrix (orthographic or perspective) ──
uniform mat4 u_projection;

// ── Per-particle base radius and speed-radius gain ──
uniform float u_baseRadius;
uniform float u_speedRadiusGain;

// ── Quad vertices (2 triangles forming a [-1,1]² quad) ──
// Encoded as a constant array; vertex_index selects which corner.
const vec2 QUAD[6] = vec2[6](
  vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
  vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

// ── Per-instance attributes (interleaved, one per particle) ──
layout(location = 0) in vec2 a_position;   // posX, posY
layout(location = 1) in vec2 a_velocity;   // velX, velY
layout(location = 2) in float a_species;   // species index (float-encoded)
layout(location = 3) in float a_density;   // density scalar

// ── Outputs to fragment shader ──
out vec2 v_uv;          // [-1,1] local quad coordinate
out float v_speed;      // |velocity|
out float v_species;    // species index (for colour lookup)
out float v_density;    // density value
out float v_radius;     // computed radius in world units

void main() {
  vec2 uv = QUAD[gl_VertexID];

  float speed = length(a_velocity);
  float radius = u_baseRadius + min(speed * u_speedRadiusGain, u_baseRadius);

  // Expand the quad around the particle center
  vec2 worldPos = a_position + uv * radius;

  gl_Position = u_projection * vec4(worldPos, 0.0, 1.0);

  v_uv      = uv;
  v_speed   = speed;
  v_species = a_species;
  v_density = a_density;
  v_radius  = radius;
}
`;

const FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;

// ── Species colour palette uploaded as a uniform array ──
uniform vec3 u_palette[8];
uniform float u_maxSpeed;

in vec2 v_uv;
in float v_speed;
in float v_species;
in float v_density;
in float v_radius;

out vec4 fragColor;

// Speed-mapped colour: slow → species colour, fast → white-hot
vec3 speedTint(vec3 base, float t) {
  vec3 hot = vec3(1.0, 0.95, 0.85);
  return mix(base, hot, t * t * 0.6);
}

void main() {
  float dist = length(v_uv);

  // Discard fragments outside the unit circle
  if (dist > 1.0) discard;

  // Anti-aliased disc edge (2px feather in UV space)
  float edge = 1.0 - smoothstep(0.85, 1.0, dist);

  // Species colour lookup (clamp index to palette size)
  int idx = clamp(int(v_species + 0.5), 0, 7);
  vec3 baseColor = u_palette[idx];

  // Speed normalisation
  float t = clamp(v_speed / max(u_maxSpeed, 0.001), 0.0, 1.0);
  vec3 color = speedTint(baseColor, t);

  // Soft inner glow (brighter core)
  float core = 1.0 - smoothstep(0.0, 0.55, dist);
  color += core * 0.25;

  // Edge glow ring (species-tinted corona)
  float ring = smoothstep(0.6, 0.85, dist) * (1.0 - smoothstep(0.85, 1.0, dist));
  color += baseColor * ring * 0.4;

  // Density-dependent opacity boost (denser regions appear more opaque)
  float densityAlpha = clamp(v_density * 0.05, 0.0, 0.3);

  float alpha = edge * (0.75 + densityAlpha);

  fragColor = vec4(color, alpha);
}
`;

// ─── Helpers ───────────────────────────────────────────────────────────────

function compileShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('ParticleInstancer: failed to create shader');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`ParticleInstancer: shader compile error:\n${info}`);
  }
  return shader;
}

function linkProgram(
  gl: WebGL2RenderingContext,
  vert: WebGLShader,
  frag: WebGLShader,
): WebGLProgram {
  const program = gl.createProgram();
  if (!program) throw new Error('ParticleInstancer: failed to create program');
  gl.attachShader(program, vert);
  gl.attachShader(program, frag);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`ParticleInstancer: program link error:\n${info}`);
  }
  return program;
}

// ─── Public interface ──────────────────────────────────────────────────────

export interface ParticleInstancerOptions {
  /** Particle base radius in world units (default 3.0). */
  baseRadius?: number;
  /** Extra radius per unit speed (default 0.4). */
  speedRadiusGain?: number;
  /** Speed that maps to maximum glow (default 60). */
  maxSpeed?: number;
  /** Enable premultiplied-alpha blending (default true). */
  premultipliedAlpha?: boolean;
}

/**
 * GPU instanced particle renderer — single draw call for up to `maxParticles`.
 *
 * Renders all particles as anti-aliased discs using WebGL2 hardware instancing.
 * Each particle occupies 6 floats in the interleaved instance buffer
 * (posX, posY, velX, velY, species, density).
 *
 * Typical frame loop:
 * ```
 *   instancer.updateBuffer(packedData, liveCount);
 *   instancer.render(orthoMatrix);
 * ```
 */
export class ParticleInstancer {
  // ── Configuration ──────────────────────────────────────────────────────
  readonly maxParticles: number;
  private baseRadius: number;
  private speedRadiusGain: number;
  private maxSpeed: number;

  // ── WebGL2 state ───────────────────────────────────────────────────────
  private gl: WebGL2RenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private instanceBuffer: WebGLBuffer | null = null;

  // ── Uniform locations ──────────────────────────────────────────────────
  private uProjection: WebGLUniformLocation | null = null;
  private uBaseRadius: WebGLUniformLocation | null = null;
  private uSpeedRadiusGain: WebGLUniformLocation | null = null;
  private uMaxSpeed: WebGLUniformLocation | null = null;
  private uPalette: WebGLUniformLocation | null = null;

  // ── CPU staging buffer ─────────────────────────────────────────────────
  private stagingBuffer: Float32Array;

  // ── Live particle count for the current frame ──────────────────────────
  private liveCount = 0;

  constructor(maxParticles: number = MAX_PARTICLES, options: ParticleInstancerOptions = {}) {
    this.maxParticles = maxParticles;
    this.baseRadius = options.baseRadius ?? 3.0;
    this.speedRadiusGain = options.speedRadiusGain ?? 0.4;
    this.maxSpeed = options.maxSpeed ?? 60;

    // Pre-allocate staging buffer for the worst-case particle count
    this.stagingBuffer = new Float32Array(maxParticles * INSTANCE_STRIDE);
  }

  // ─── Lifecycle ─────────────────────────────────────────────────────────

  /**
   * Attach to a canvas and initialise WebGL2 resources.
   * Call once before the first `render`.
   *
   * @param canvas  Target canvas element (or offscreen canvas).
   * @param options WebGL context attributes (optional).
   * @returns `this` for chaining.
   */
  attach(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    contextAttrs: WebGLContextAttributes = {},
  ): this {
    const gl = canvas.getContext('webgl2', {
      alpha: true,
      premultipliedAlpha: true,
      antialias: false,
      ...contextAttrs,
    }) as WebGL2RenderingContext | null;

    if (!gl) throw new Error('ParticleInstancer: WebGL2 not available');
    this.gl = gl;

    // ── Compile & link program ─────────────────────────────────────────
    const vs = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
    this.program = linkProgram(gl, vs, fs);
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    // ── Resolve uniform locations ──────────────────────────────────────
    this.uProjection = gl.getUniformLocation(this.program, 'u_projection');
    this.uBaseRadius = gl.getUniformLocation(this.program, 'u_baseRadius');
    this.uSpeedRadiusGain = gl.getUniformLocation(this.program, 'u_speedRadiusGain');
    this.uMaxSpeed = gl.getUniformLocation(this.program, 'u_maxSpeed');
    this.uPalette = gl.getUniformLocation(this.program, 'u_palette');

    // ── Create instance buffer (GPU-side) ──────────────────────────────
    this.instanceBuffer = gl.createBuffer();
    if (!this.instanceBuffer) throw new Error('ParticleInstancer: failed to create buffer');

    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      this.stagingBuffer.byteLength,
      gl.DYNAMIC_DRAW,
    );

    // ── Build VAO ──────────────────────────────────────────────────────
    this.vao = gl.createVertexArray();
    if (!this.vao) throw new Error('ParticleInstancer: failed to create VAO');
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);

    const stride = INSTANCE_STRIDE * 4; // bytes per instance

    // location 0 — a_position (vec2: posX, posY)
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, stride, 0);
    gl.vertexAttribDivisor(0, 1); // per-instance

    // location 1 — a_velocity (vec2: velX, velY)
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, stride, 8);
    gl.vertexAttribDivisor(1, 1);

    // location 2 — a_species (float)
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, stride, 16);
    gl.vertexAttribDivisor(2, 1);

    // location 3 — a_density (float)
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 1, gl.FLOAT, false, stride, 20);
    gl.vertexAttribDivisor(3, 1);

    gl.bindVertexArray(null);

    return this;
  }

  // ─── Data upload ───────────────────────────────────────────────────────

  /**
   * Upload particle data to the GPU instance buffer.
   *
   * `particles` must be an interleaved Float32Array laid out as:
   *   [ posX₀, posY₀, velX₀, velY₀, species₀, density₀,
   *     posX₁, posY₁, velX₁, velY₁, species₁, density₁,
   *     … ]
   *
   * Only the first `count` particles are uploaded and rendered.
   *
   * @param particles  Interleaved per-particle data.
   * @param count      Number of live particles (not floats).
   */
  updateBuffer(particles: Float32Array, count: number): void {
    if (!this.gl || !this.instanceBuffer) return;
    const gl = this.gl;

    this.liveCount = Math.min(count, this.maxParticles);
    const floatCount = this.liveCount * INSTANCE_STRIDE;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);

    // Sub-buffer upload: only push the live region to avoid copying the
    // entire max-sized allocation every frame.
    if (floatCount <= particles.length) {
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, particles, 0, floatCount);
    } else {
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, particles);
    }
  }

  // ─── Render ────────────────────────────────────────────────────────────

  /**
   * Render all live particles in a single instanced draw call.
   *
   * @param projectionMatrix  4×4 column-major projection matrix (Float32Array[16]).
   *                          Typically an orthographic matrix mapping world coords
   *                          to clip space: `ortho(0, width, height, 0, -1, 1)`.
   */
  render(projectionMatrix: Float32Array): void {
    if (!this.gl || !this.program || !this.vao || this.liveCount === 0) return;
    const gl = this.gl;

    // ── State setup ────────────────────────────────────────────────────
    gl.useProgram(this.program);

    // Alpha blending (additive-friendly for glow)
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // No depth test for 2D particle overlay
    gl.disable(gl.DEPTH_TEST);

    // ── Upload uniforms ────────────────────────────────────────────────
    gl.uniformMatrix4fv(this.uProjection, false, projectionMatrix);
    gl.uniform1f(this.uBaseRadius, this.baseRadius);
    gl.uniform1f(this.uSpeedRadiusGain, this.speedRadiusGain);
    gl.uniform1f(this.uMaxSpeed, this.maxSpeed);

    // Flatten palette to a contiguous Float32Array for uniform upload
    const flat = new Float32Array(SPECIES_PALETTE.length * 3);
    for (let i = 0; i < SPECIES_PALETTE.length; i++) {
      flat[i * 3]     = SPECIES_PALETTE[i][0];
      flat[i * 3 + 1] = SPECIES_PALETTE[i][1];
      flat[i * 3 + 2] = SPECIES_PALETTE[i][2];
    }
    gl.uniform3fv(this.uPalette, flat);

    // ── Draw instanced ─────────────────────────────────────────────────
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.liveCount);
    gl.bindVertexArray(null);
  }

  // ─── Cleanup ───────────────────────────────────────────────────────────

  /** Release all WebGL2 resources. Safe to call multiple times. */
  dispose(): void {
    const gl = this.gl;
    if (!gl) return;

    if (this.vao) { gl.deleteVertexArray(this.vao); this.vao = null; }
    if (this.instanceBuffer) { gl.deleteBuffer(this.instanceBuffer); this.instanceBuffer = null; }
    if (this.program) { gl.deleteProgram(this.program); this.program = null; }

    this.gl = null;
    this.liveCount = 0;
  }

  // ─── Runtime tunables ──────────────────────────────────────────────────

  /** Update the base particle radius (world units). */
  setBaseRadius(r: number): void { this.baseRadius = r; }

  /** Update the speed-dependent radius gain. */
  setSpeedRadiusGain(g: number): void { this.speedRadiusGain = g; }

  /** Update the max-speed normalisation value. */
  setMaxSpeed(s: number): void { this.maxSpeed = s; }

  /** Return the current WebGL2 context (or null if not attached). */
  getContext(): WebGL2RenderingContext | null { return this.gl; }

  /** Return the current live particle count. */
  getCount(): number { return this.liveCount; }
}

// ─── Data packing utilities ────────────────────────────────────────────────

/**
 * Interleaved particle data suitable for `ParticleInstancer.updateBuffer`.
 */
export interface PackedParticleResult {
  /** Interleaved Float32Array: [posX, posY, velX, velY, species, density] × count. */
  buffer: Float32Array;
  /** Number of particles packed. */
  count: number;
}

/**
 * Pack an array of `Particle` objects (from world-stepper) into the
 * interleaved Float32Array layout expected by `ParticleInstancer.updateBuffer`.
 *
 * Species strings are mapped to numeric indices via a simple keyword table
 * that matches the SPECIES_PALETTE order.  Unknown species default to index 7.
 */
export function packParticleData(
  particles: ReadonlyArray<{
    x: number;
    y: number;
    vx?: number;
    vy?: number;
    species?: string | number;
    density?: number;
    isBoundary?: boolean;
  }>,
  options: { includeBoundary?: boolean } = {},
): PackedParticleResult {
  const includeBoundary = options.includeBoundary ?? false;
  const out = new Float32Array(particles.length * INSTANCE_STRIDE);
  let count = 0;

  for (let i = 0; i < particles.length; i++) {
    const p = particles[i];
    if (!includeBoundary && p.isBoundary) continue;

    const offset = count * INSTANCE_STRIDE;
    out[offset]     = p.x;
    out[offset + 1] = p.y;
    out[offset + 2] = p.vx ?? 0;
    out[offset + 3] = p.vy ?? 0;
    out[offset + 4] = resolveSpeciesIndex(p.species);
    out[offset + 5] = p.density ?? 0;
    count++;
  }

  return { buffer: out, count };
}

/**
 * Pack Structure-of-Arrays particle data (as in types.ts `ParticleData`) into
 * the interleaved layout.  This avoids the per-particle object overhead when
 * data comes from the WebGPU compute readback path.
 */
export function packParticleDataSOA(
  x: Float32Array,
  y: Float32Array,
  vx: Float32Array,
  vy: Float32Array,
  species: Uint32Array,
  density: Float32Array | null,
  count: number,
): PackedParticleResult {
  const n = Math.min(count, x.length, y.length);
  const out = new Float32Array(n * INSTANCE_STRIDE);

  for (let i = 0; i < n; i++) {
    const offset = i * INSTANCE_STRIDE;
    out[offset]     = x[i];
    out[offset + 1] = y[i];
    out[offset + 2] = vx[i];
    out[offset + 3] = vy[i];
    out[offset + 4] = species[i];
    out[offset + 5] = density ? density[i] : 0;
  }

  return { buffer: out, count: n };
}

// ─── Species string → numeric index mapping ────────────────────────────────

const SPECIES_KEYWORD_MAP: [string[], number][] = [
  // index 0-6 match SPECIES_COLORS in world-renderer.ts
  // These also handle cell-kind keywords for transformer layers
  [['attention', 'attn', 'self_attn'], 0],
  [['ffn', 'feed_forward', 'mlp'], 1],
  [['layernorm', 'layer_norm', 'add_norm', 'norm'], 2],
  [['embedding', 'embed', 'pos_encode', 'input_embed'], 3],
  [['softmax', 'output'], 4],
  [['residual', 'skip'], 5],
  [['dropout', 'mask'], 6],
];

function resolveSpeciesIndex(species: string | number | undefined): number {
  if (species == null) return 7;
  if (typeof species === 'number') return Math.min(species, 7);

  const lower = species.toLowerCase();
  for (const [keywords, idx] of SPECIES_KEYWORD_MAP) {
    for (const kw of keywords) {
      if (lower.includes(kw)) return idx;
    }
  }
  return 7;
}

// ─── Projection matrix helper ──────────────────────────────────────────────

/**
 * Build a column-major 4×4 orthographic projection matrix suitable for
 * the 2D particle renderer.
 *
 * Maps world coordinates `[left..right, bottom..top]` to clip space `[-1,1]`.
 * Matches the Canvas2D convention where (0,0) is top-left and Y increases
 * downward when called as `ortho(0, width, height, 0)`.
 */
export function ortho(
  left: number,
  right: number,
  bottom: number,
  top: number,
  near = -1,
  far = 1,
): Float32Array {
  const lr = 1 / (left - right);
  const bt = 1 / (bottom - top);
  const nf = 1 / (near - far);

  // Column-major layout
  return new Float32Array([
    -2 * lr,       0,             0,             0,
    0,             -2 * bt,       0,             0,
    0,             0,             2 * nf,        0,
    (left + right) * lr,
    (top + bottom) * bt,
    (near + far) * nf,
    1,
  ]);
}
