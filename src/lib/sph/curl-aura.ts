/**
 * curl-aura.ts — M749: Curl-noise aura halos for cell bodies
 * ─────────────────────────────────────────────────────────────────────────────
 * Renders a soft, animated aura glow around each cell using curl-noise
 * distortion of concentric SDF rings.  Extends the M745 instanced particle
 * pipeline with an aura-specific draw pass that shares the same WebGL2
 * context and projection matrix.
 *
 * Architecture
 * ─────────────────────────────────────────────────────────────────────────────
 * Each cell aura is a screen-aligned quad (same 6-vertex instanced pattern as
 * particle-instancing.ts) sized to enclose the cell bbox plus an aura radius.
 * The fragment shader evaluates:
 *
 *   1. Concentric SDF rings from the cell center
 *   2. Curl-noise perturbation of the ring coordinates (divergence-free
 *      distortion — organic, swirling motion with no pile-ups)
 *   3. fBm turbulence modulating ring opacity and width
 *   4. Species-keyed color with velocity-driven hue shift
 *   5. Soft radial falloff with inner glow and outer corona
 *
 * Per-instance data (10 floats / aura):
 *   [ centerX, centerY, halfW, halfH, species, phase,
 *     velocity, density, opacity, pulseRate ]
 *
 * The aura pass runs AFTER the particle instancing pass (additive blend on
 * top of particle discs) and BEFORE the post-process bloom, so bloom picks
 * up the aura glow naturally.
 *
 * Integration
 * ─────────────────────────────────────────────────────────────────────────────
 *   import { ParticleInstancer, ortho } from './particle-instancing';
 *
 *   const particles = new ParticleInstancer(50000);
 *   const aura = new CurlAuraRenderer(128);  // max 128 cells
 *   particles.attach(canvas);
 *   aura.attach(particles.getContext()!);     // share GL context
 *
 *   // frame loop:
 *   aura.setTime(elapsed);
 *   aura.updateBuffer(packedAuraData, cellCount);
 *   particles.render(proj);
 *   aura.render(proj);                        // additive on top
 *
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/particle-instancing.ts    — M745 instanced quad pattern
 *   src/lib/sph/curl-flow-field.ts        — M606 WebGPU curl noise
 *   src/lib/shaders/curl-trail.frag       — GLSL curl + fBm (LYGIA inlined)
 *   upstream/lygia/generative/curl.glsl   — divergence-free curl definition
 *   upstream/pixijs-filters               — GlowFilter / BloomFilter reference
 *   channels/rendering/species/           — species visual identity
 *
 * Research: xiaodi #M749 — cell-pubsub-loop
 */

// ─── Constants ─────────────────────────────────────────────────────────────

/** Floats per aura instance in the interleaved buffer. */
export const AURA_STRIDE = 10;
// centerX, centerY, halfW, halfH, species, phase,
// velocity, density, opacity, pulseRate

// ─── Species colour palette (normalised RGB, matches M745) ─────────────────

const SPECIES_AURA_PALETTE: [number, number, number][] = [
  [0.247, 0.318, 0.710],  // 0  cil-eye       #3F51B5
  [1.000, 0.435, 0.000],  // 1  cil-bolt      #FF6F00
  [0.180, 0.490, 0.196],  // 2  cil-vector    #2E7D32
  [0.776, 0.157, 0.157],  // 3  cil-plus      #C62828
  [0.271, 0.353, 0.392],  // 4  cil-arrow     #455A64
  [0.482, 0.122, 0.635],  // 5  cil-filter    #7B1FA2
  [0.180, 0.490, 0.196],  // 6  cil-code      #2E7D32
  [0.084, 0.396, 0.753],  // 7  cil-layers    #1565C0
  [0.961, 0.498, 0.090],  // 8  cil-loop      #F57F17
  [0.216, 0.278, 0.310],  // 9  cil-graph     #37474F
  [1.000, 1.000, 1.000],  // 10 fallback white
];

// ─── Shader source ─────────────────────────────────────────────────────────

const AURA_VERT = /* glsl */ `#version 300 es
precision highp float;

uniform mat4 u_projection;
uniform float u_time;

// Quad vertices (2 triangles, [-1,1]² quad)
const vec2 QUAD[6] = vec2[6](
  vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
  vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

// Per-instance attributes
layout(location = 0) in vec2  a_center;     // centerX, centerY
layout(location = 1) in vec2  a_halfSize;   // halfW, halfH
layout(location = 2) in float a_species;
layout(location = 3) in float a_phase;
layout(location = 4) in float a_velocity;
layout(location = 5) in float a_density;
layout(location = 6) in float a_opacity;
layout(location = 7) in float a_pulseRate;

// Outputs
out vec2  v_uv;
out float v_species;
out float v_phase;
out float v_velocity;
out float v_density;
out float v_opacity;
out float v_pulseRate;
out vec2  v_worldPos;

void main() {
  vec2 uv = QUAD[gl_VertexID];

  // Expand quad: cell half-size + aura radius (50% overshoot)
  vec2 auraHalf = a_halfSize * 1.5 + vec2(20.0);
  vec2 worldPos = a_center + uv * auraHalf;

  gl_Position = u_projection * vec4(worldPos, 0.0, 1.0);

  v_uv        = uv;
  v_species   = a_species;
  v_phase     = a_phase;
  v_velocity  = a_velocity;
  v_density   = a_density;
  v_opacity   = a_opacity;
  v_pulseRate = a_pulseRate;
  v_worldPos  = worldPos;
}
`;

const AURA_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform float u_time;
uniform vec3  u_palette[11];

in vec2  v_uv;
in float v_species;
in float v_phase;
in float v_velocity;
in float v_density;
in float v_opacity;
in float v_pulseRate;
in vec2  v_worldPos;

out vec4 fragColor;

// ── Inlined LYGIA-style curl noise (2D slice for aura distortion) ──────────

vec2 _hash22(vec2 p) {
  p = fract(p * vec2(443.8975, 441.4234));
  p += dot(p, p.yx + 19.19);
  return fract((p.xx + p.yx) * p.xy);
}

float _vnoise2(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = dot(_hash22(i + vec2(0,0)), vec2(1.0));
  float b = dot(_hash22(i + vec2(1,0)), vec2(1.0));
  float c = dot(_hash22(i + vec2(0,1)), vec2(1.0));
  float d = dot(_hash22(i + vec2(1,1)), vec2(1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// 2D curl of a scalar potential field — returns a divergence-free vec2
vec2 curlNoise2D(vec2 p) {
  float e = 0.01;
  float n  = _vnoise2(p + vec2(0, e));
  float s  = _vnoise2(p - vec2(0, e));
  float ea = _vnoise2(p + vec2(e, 0));
  float w  = _vnoise2(p - vec2(e, 0));
  // curl of scalar field: ( ∂f/∂y, -∂f/∂x )
  return vec2((n - s), -(ea - w)) / (2.0 * e);
}

// fBm for turbulence modulation
float fbm2(vec2 p) {
  float v = 0.0, a = 0.5;
  mat2 m = mat2(0.8, 0.6, -0.6, 0.8);
  for (int i = 0; i < 4; i++) {
    v += a * _vnoise2(p);
    p = m * p * 2.0;
    a *= 0.5;
  }
  return v;
}

// ── HSV→RGB for species colour blending ────────────────────────────────────

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// ── Main ───────────────────────────────────────────────────────────────────

void main() {
  float dist = length(v_uv);

  // Hard discard beyond unit circle
  if (dist > 1.0) discard;

  // ── 1. Curl-noise distortion of radial coordinate ───────────────────
  float t = u_time * 0.08 + v_phase;
  vec2 curlDomain = v_worldPos * 0.015 + vec2(t * 0.7, t * 0.3);
  vec2 curl = curlNoise2D(curlDomain);

  // Distort the UV for organic swirl (stronger at edges)
  float edgeFactor = smoothstep(0.2, 0.8, dist);
  vec2 distortedUV = v_uv + curl * 0.12 * edgeFactor;
  float distortedDist = length(distortedUV);

  // ── 2. fBm turbulence ───────────────────────────────────────────────
  vec2 fbmDomain = v_worldPos * 0.008 + vec2(t * 0.3, t * 0.15);
  float turb = fbm2(fbmDomain);

  // ── 3. Concentric ring SDF ──────────────────────────────────────────
  // Pulsing ring frequency driven by species pulse rate
  float pulse = sin(u_time * v_pulseRate + v_phase) * 0.5 + 0.5;
  float ringFreq = 3.0 + pulse * 1.5 + turb * 2.0;
  float rings = sin(distortedDist * ringFreq * 3.14159) * 0.5 + 0.5;
  rings = smoothstep(0.3, 0.7, rings);

  // ── 4. Radial falloff layers ────────────────────────────────────────
  // Inner glow: bright core near cell body
  float innerGlow = 1.0 - smoothstep(0.0, 0.45, dist);
  innerGlow = innerGlow * innerGlow;  // quadratic falloff

  // Mid aura: the curl-distorted ring zone
  float midAura = smoothstep(0.2, 0.5, dist) * (1.0 - smoothstep(0.6, 0.95, dist));
  midAura *= rings * 0.6;

  // Outer corona: soft edge glow
  float corona = smoothstep(0.55, 0.75, dist) * (1.0 - smoothstep(0.85, 1.0, dist));
  corona *= (0.3 + 0.7 * turb);

  // ── 5. Species colour with velocity hue shift ──────────────────────
  int idx = clamp(int(v_species + 0.5), 0, 10);
  vec3 baseColor = u_palette[idx];

  // Velocity drives warm shift: fast cells glow hotter
  float vel = clamp(v_velocity, 0.0, 1.0);
  vec3 hotColor = vec3(1.0, 0.85, 0.6);
  vec3 auraColor = mix(baseColor, hotColor, vel * vel * 0.4);

  // Turbulence subtly shifts hue for organic variation
  auraColor = mix(auraColor, baseColor * 1.3, turb * 0.15);

  // ── 6. Density-responsive intensity ─────────────────────────────────
  // Denser regions (more overlapping cells) get brighter aura
  float densityBoost = clamp(v_density * 0.03, 0.0, 0.25);

  // ── 7. Compose layers ───────────────────────────────────────────────
  float intensity = innerGlow * 0.55
                  + midAura
                  + corona * 0.4;
  intensity *= v_opacity;
  intensity += densityBoost;

  // Velocity adds overall brightness
  intensity *= (0.7 + vel * 0.3);

  float alpha = clamp(intensity, 0.0, 1.0);

  // Fade outer edge
  alpha *= 1.0 - smoothstep(0.88, 1.0, dist);

  if (alpha < 0.003) discard;

  // Inner core gets species color, outer gets softer desaturated version
  vec3 finalColor = mix(auraColor, auraColor * 0.6 + 0.1, smoothstep(0.3, 0.8, dist));

  // Tip brightness at core
  finalColor += innerGlow * baseColor * 0.3;

  // Premultiplied alpha for additive-friendly compositing
  fragColor = vec4(finalColor * alpha, alpha);
}
`;

// ─── Helpers ───────────────────────────────────────────────────────────────

function compileShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('CurlAura: failed to create shader');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`CurlAura: shader compile error:\n${info}`);
  }
  return shader;
}

function linkProgram(
  gl: WebGL2RenderingContext,
  vert: WebGLShader,
  frag: WebGLShader,
): WebGLProgram {
  const program = gl.createProgram();
  if (!program) throw new Error('CurlAura: failed to create program');
  gl.attachShader(program, vert);
  gl.attachShader(program, frag);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`CurlAura: program link error:\n${info}`);
  }
  return program;
}

// ─── Public interface ──────────────────────────────────────────────────────

export interface CurlAuraOptions {
  /** Default aura opacity (0–1). Default 0.65. */
  defaultOpacity?: number;
  /** Default pulse rate (radians/sec). Default 1.2. */
  defaultPulseRate?: number;
  /** Enable additive blending (default true, false = standard alpha). */
  additiveBlend?: boolean;
}

/**
 * CurlAuraRenderer — GPU-instanced curl-noise aura halos for cell bodies.
 *
 * Shares a WebGL2 context with ParticleInstancer (M745).  Each cell gets
 * a single instanced quad with a fragment shader that evaluates curl-noise
 * distorted concentric rings, fBm turbulence, and species-keyed color.
 *
 * Typical frame loop (after ParticleInstancer.render):
 * ```
 *   aura.setTime(elapsed);
 *   aura.updateBuffer(packedData, cellCount);
 *   aura.render(projectionMatrix);
 * ```
 */
export class CurlAuraRenderer {
  readonly maxCells: number;
  private additiveBlend: boolean;

  // WebGL2 state
  private gl: WebGL2RenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private instanceBuffer: WebGLBuffer | null = null;

  // Uniform locations
  private uProjection: WebGLUniformLocation | null = null;
  private uTime: WebGLUniformLocation | null = null;
  private uPalette: WebGLUniformLocation | null = null;

  // Staging
  private stagingBuffer: Float32Array;
  private liveCount = 0;
  private currentTime = 0;

  constructor(maxCells = 128, options: CurlAuraOptions = {}) {
    this.maxCells = maxCells;
    this.additiveBlend = options.additiveBlend ?? true;
    this.stagingBuffer = new Float32Array(maxCells * AURA_STRIDE);
  }

  // ─── Lifecycle ─────────────────────────────────────────────────────────

  /**
   * Attach to an existing WebGL2 context (shared with ParticleInstancer).
   * Call `instancer.getContext()` and pass it here.
   */
  attach(gl: WebGL2RenderingContext): this {
    this.gl = gl;

    // Compile & link
    const vs = compileShader(gl, gl.VERTEX_SHADER, AURA_VERT);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, AURA_FRAG);
    this.program = linkProgram(gl, vs, fs);
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    // Resolve uniforms
    this.uProjection = gl.getUniformLocation(this.program, 'u_projection');
    this.uTime       = gl.getUniformLocation(this.program, 'u_time');
    this.uPalette    = gl.getUniformLocation(this.program, 'u_palette');

    // Instance buffer
    this.instanceBuffer = gl.createBuffer();
    if (!this.instanceBuffer) throw new Error('CurlAura: failed to create buffer');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.stagingBuffer.byteLength, gl.DYNAMIC_DRAW);

    // Build VAO
    this.vao = gl.createVertexArray();
    if (!this.vao) throw new Error('CurlAura: failed to create VAO');
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);

    const stride = AURA_STRIDE * 4; // bytes

    // location 0 — a_center (vec2: centerX, centerY)
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, stride, 0);
    gl.vertexAttribDivisor(0, 1);

    // location 1 — a_halfSize (vec2: halfW, halfH)
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, stride, 8);
    gl.vertexAttribDivisor(1, 1);

    // location 2 — a_species (float)
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, stride, 16);
    gl.vertexAttribDivisor(2, 1);

    // location 3 — a_phase (float)
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 1, gl.FLOAT, false, stride, 20);
    gl.vertexAttribDivisor(3, 1);

    // location 4 — a_velocity (float)
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 1, gl.FLOAT, false, stride, 24);
    gl.vertexAttribDivisor(4, 1);

    // location 5 — a_density (float)
    gl.enableVertexAttribArray(5);
    gl.vertexAttribPointer(5, 1, gl.FLOAT, false, stride, 28);
    gl.vertexAttribDivisor(5, 1);

    // location 6 — a_opacity (float)
    gl.enableVertexAttribArray(6);
    gl.vertexAttribPointer(6, 1, gl.FLOAT, false, stride, 32);
    gl.vertexAttribDivisor(6, 1);

    // location 7 — a_pulseRate (float)
    gl.enableVertexAttribArray(7);
    gl.vertexAttribPointer(7, 1, gl.FLOAT, false, stride, 36);
    gl.vertexAttribDivisor(7, 1);

    gl.bindVertexArray(null);
    return this;
  }

  /**
   * Attach to a canvas directly (creates own WebGL2 context).
   * Use this when NOT sharing context with ParticleInstancer.
   */
  attachCanvas(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    contextAttrs: WebGLContextAttributes = {},
  ): this {
    const gl = canvas.getContext('webgl2', {
      alpha: true,
      premultipliedAlpha: true,
      antialias: false,
      ...contextAttrs,
    }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error('CurlAura: WebGL2 not available');
    return this.attach(gl);
  }

  // ─── Time ──────────────────────────────────────────────────────────────

  /** Set elapsed time in seconds (drives curl animation). */
  setTime(t: number): void {
    this.currentTime = t;
  }

  // ─── Data upload ───────────────────────────────────────────────────────

  /**
   * Upload cell aura data to the GPU.
   *
   * `data` is an interleaved Float32Array:
   *   [ centerX, centerY, halfW, halfH, species, phase,
   *     velocity, density, opacity, pulseRate ] × count
   */
  updateBuffer(data: Float32Array, count: number): void {
    if (!this.gl || !this.instanceBuffer) return;
    const gl = this.gl;

    this.liveCount = Math.min(count, this.maxCells);
    const floatCount = this.liveCount * AURA_STRIDE;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    if (floatCount <= data.length) {
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, data, 0, floatCount);
    } else {
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    }
  }

  // ─── Render ────────────────────────────────────────────────────────────

  /**
   * Render all cell auras. Call AFTER ParticleInstancer.render() for correct
   * compositing order (auras behind particles, picked up by bloom).
   *
   * @param projectionMatrix  Same 4×4 column-major ortho matrix used by
   *                          ParticleInstancer.
   */
  render(projectionMatrix: Float32Array): void {
    if (!this.gl || !this.program || !this.vao || this.liveCount === 0) return;
    const gl = this.gl;

    gl.useProgram(this.program);

    // Blending
    gl.enable(gl.BLEND);
    if (this.additiveBlend) {
      // Additive: aura glow accumulates (brighter where cells overlap)
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    } else {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    }
    gl.disable(gl.DEPTH_TEST);

    // Upload uniforms
    gl.uniformMatrix4fv(this.uProjection, false, projectionMatrix);
    gl.uniform1f(this.uTime, this.currentTime);

    // Flatten palette
    const flat = new Float32Array(SPECIES_AURA_PALETTE.length * 3);
    for (let i = 0; i < SPECIES_AURA_PALETTE.length; i++) {
      flat[i * 3]     = SPECIES_AURA_PALETTE[i][0];
      flat[i * 3 + 1] = SPECIES_AURA_PALETTE[i][1];
      flat[i * 3 + 2] = SPECIES_AURA_PALETTE[i][2];
    }
    gl.uniform3fv(this.uPalette, flat);

    // Draw
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.liveCount);
    gl.bindVertexArray(null);

    // Restore standard alpha blend for subsequent passes
    if (this.additiveBlend) {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    }
  }

  // ─── Cleanup ───────────────────────────────────────────────────────────

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

  setAdditiveBlend(v: boolean): void { this.additiveBlend = v; }
  getContext(): WebGL2RenderingContext | null { return this.gl; }
  getCount(): number { return this.liveCount; }
}

// ─── Data packing utilities ────────────────────────────────────────────────

export interface CellAuraDescriptor {
  /** Cell center X in world coordinates. */
  centerX: number;
  /** Cell center Y in world coordinates. */
  centerY: number;
  /** Cell half-width (bbox.w / 2). */
  halfW: number;
  /** Cell half-height (bbox.h / 2). */
  halfH: number;
  /** Species index (0–9). */
  species?: number | string;
  /** Phase offset for animation decorrelation (radians). */
  phase?: number;
  /** Normalised velocity (0–1). */
  velocity?: number;
  /** Local density scalar (from SPH or neighbour count). */
  density?: number;
  /** Aura opacity override (0–1). Default 0.65. */
  opacity?: number;
  /** Pulse rate (radians/sec). Default 1.2. */
  pulseRate?: number;
}

export interface PackedAuraResult {
  buffer: Float32Array;
  count: number;
}

/**
 * Pack cell descriptors into the interleaved Float32Array for
 * `CurlAuraRenderer.updateBuffer()`.
 */
export function packAuraData(
  cells: ReadonlyArray<CellAuraDescriptor>,
  defaults: { opacity?: number; pulseRate?: number } = {},
): PackedAuraResult {
  const defOpacity   = defaults.opacity   ?? 0.65;
  const defPulseRate = defaults.pulseRate ?? 1.2;

  const out = new Float32Array(cells.length * AURA_STRIDE);
  let count = 0;

  for (const cell of cells) {
    const off = count * AURA_STRIDE;
    out[off]     = cell.centerX;
    out[off + 1] = cell.centerY;
    out[off + 2] = cell.halfW;
    out[off + 3] = cell.halfH;
    out[off + 4] = resolveSpeciesIdx(cell.species);
    out[off + 5] = cell.phase ?? (count * 2.399);  // golden angle decorrelation
    out[off + 6] = cell.velocity ?? 0;
    out[off + 7] = cell.density ?? 0;
    out[off + 8] = cell.opacity ?? defOpacity;
    out[off + 9] = cell.pulseRate ?? defPulseRate;
    count++;
  }

  return { buffer: out, count };
}

/**
 * Pack cell data from Structure-of-Arrays layout (matching cell registry
 * / physics channel outputs).
 */
export function packAuraDataSOA(
  centerX:   Float32Array,
  centerY:   Float32Array,
  halfW:     Float32Array,
  halfH:     Float32Array,
  species:   Uint32Array,
  velocity:  Float32Array | null,
  density:   Float32Array | null,
  count:     number,
  opacity?:  number,
  pulseRate?: number,
): PackedAuraResult {
  const n = Math.min(count, centerX.length);
  const out = new Float32Array(n * AURA_STRIDE);
  const op = opacity ?? 0.65;
  const pr = pulseRate ?? 1.2;

  for (let i = 0; i < n; i++) {
    const off = i * AURA_STRIDE;
    out[off]     = centerX[i];
    out[off + 1] = centerY[i];
    out[off + 2] = halfW[i];
    out[off + 3] = halfH[i];
    out[off + 4] = species[i];
    out[off + 5] = i * 2.399;  // golden angle phase
    out[off + 6] = velocity ? velocity[i] : 0;
    out[off + 7] = density ? density[i] : 0;
    out[off + 8] = op;
    out[off + 9] = pr;
  }

  return { buffer: out, count: n };
}

// ─── Species string → index ────────────────────────────────────────────────

const SPECIES_MAP: [string[], number][] = [
  [['eye', 'attn', 'attention', 'self_attn'], 0],
  [['bolt', 'ffn', 'feed_forward', 'mlp'], 1],
  [['vector', 'embed', 'input_embed', 'pos_encode'], 2],
  [['plus', 'residual', 'add_norm'], 3],
  [['arrow', 'output', 'softmax'], 4],
  [['filter', 'mask', 'dropout'], 5],
  [['code', 'function'], 6],
  [['layers', 'layer_norm', 'norm'], 7],
  [['loop', 'feedback', 'recurrent'], 8],
  [['graph', 'network', 'struct'], 9],
];

function resolveSpeciesIdx(species: string | number | undefined): number {
  if (species == null) return 10;
  if (typeof species === 'number') return Math.min(species, 10);
  const lower = species.toLowerCase();
  for (const [keywords, idx] of SPECIES_MAP) {
    for (const kw of keywords) {
      if (lower.includes(kw)) return idx;
    }
  }
  return 10;
}
