/**
 * at-jellyfish-cell.ts — M807: AT jellyfish.bin as Organic Cell Shape
 * ─────────────────────────────────────────────────────────────────────────────
 * Replaces the rectangular rigid-body visual with AT's jellyfish geometry.
 * The collision body remains an AABB (unchanged), but the *render* path now
 * draws a Draco-decoded jellyfish mesh with per-species variants.
 *
 * Mapping to Transformer architecture:
 *   self_attn  → large jellyfish  (attention = sensory perception, wide bell)
 *   ffn        → small jellyfish swarm  (feed-forward = parallel processing)
 *
 * Shader pipeline:
 *   Vertex:   AT JellyShader noise-based pulse + sway (cnoise displacement)
 *   Fragment: translucent body with Fresnel rim glow + species colour tinting
 *
 * The .bin format:
 *   [uint32 LE: headerSize]  (unused padding bytes follow)
 *   [10 bytes offset → JSON header: {name, type, attributes}]
 *   [Draco payload immediately after closing '}']
 *
 * Upstream references:
 *   upstream/activetheory-assets/geometry/jellyfish.bin  — Draco mesh
 *   upstream/activetheory-assets/compiled.vs §JellyShader.glsl
 *   src/lib/threed-pipeline.ts                          — DracoThread decoder
 *   src/lib/sph/cell-visual-identity.ts                 — Morphology, VisualProfile
 *   src/lib/sph/cell-body-bridge.ts                     — CellPhysicsConfig (AABB)
 *   src/lib/sph/instanced-cell-renderer.ts              — existing rect renderer
 *
 * Research: xiaodi #M807 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────


import type { GeometryDescriptor } from '../threed-pipeline';
import { DracoThread }             from '../threed-pipeline';
import type { Morphology, VisualProfile } from './cell-visual-identity';

// [orphan-precise] /** Path to AT jellyfish geometry (Draco-compressed). */
const JELLYFISH_BIN_PATH = '/upstream/activetheory-assets/geometry/jellyfish.bin';

/**
 * Byte offset where the JSON header begins inside jellyfish.bin.
 * Format: [4-byte uint32 LE headerSize][6 bytes padding][JSON...][Draco payload]
 */
const BIN_JSON_OFFSET = 10;

/** Maximum instances in a single instanced draw call. */
const MAX_INSTANCES = 128;

/** Floats per instance in the instance VBO (mat4 + colour + params). */
const FLOATS_PER_INSTANCE = 28;
//  0-15  mat4  modelMatrix
// 16-19  vec4  colour (RGBA)
// 20     float scale
// 21     float pulsePhase
// 22     float tentacleLength
// 23     float translucency
// 24-25  vec2  swayOffset
// 26     float bellRadius
// 27     float speciesIdx (for shader branching)

// ─────────────────────────────────────────────────────────────────────────────
// Species → Jellyfish variant config
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Per-species jellyfish variant parameters.
 * Derived from the Morphology/Transformer role:
 *   self_attn → large sensory jellyfish (big bell, long tentacles, slow pulse)
 *   ffn       → small processing jellyfish (compact, short tentacles, fast pulse)
 */
export interface JellyfishVariantConfig {
  /** Uniform scale multiplier relative to cell bbox. */
  scale: number;
  /** Tentacle length multiplier (1.0 = geometry default). */
  tentacleLength: number;
  /** Bell dome radius multiplier. */
  bellRadius: number;
  /** Pulse frequency multiplier (Hz-like). */
  pulseFrequency: number;
  /** Body translucency 0–1 (0 = opaque, 1 = fully transparent). */
  translucency: number;
  /** Sway amplitude (world units) — lateral drift from noise. */
  swayAmplitude: number;
  /** Base tint colour RGBA (linear). */
  tint: [number, number, number, number];
}

/**
 * Built-in variant presets keyed by species string.
 * Unknown species fall back to 'default'.
 */
const VARIANT_PRESETS: Record<string, JellyfishVariantConfig> = {
  // ── self_attn: large sensory jellyfish ────────────────────────────────────
  // Attention = perception. Big bell to "sense" wide context.
  // Slow, majestic pulse. Long trailing tentacles = attention span.
  'cil-eye': {
    scale:           1.4,
    tentacleLength:  1.6,
    bellRadius:      1.3,
    pulseFrequency:  0.35,
    translucency:    0.65,
    swayAmplitude:   0.12,
    tint: [0.3, 0.55, 0.95, 0.85],   // deep blue, translucent
  },

  // ── ffn: small processing jellyfish swarm ─────────────────────────────────
  // Feed-forward = rapid parallel processing. Compact bodies, fast pulse.
  // Short tentacles = quick, decisive computation.
  'cil-bolt': {
    scale:           0.55,
    tentacleLength:  0.6,
    bellRadius:      0.8,
    pulseFrequency:  1.2,
    translucency:    0.45,
    swayAmplitude:   0.08,
    tint: [0.85, 0.45, 0.2, 0.9],    // warm amber
  },

  // ── Embedding / encoding: petal-like jellyfish ────────────────────────────
  'cil-vector': {
    scale:           0.9,
    tentacleLength:  1.0,
    bellRadius:      1.1,
    pulseFrequency:  0.6,
    translucency:    0.55,
    swayAmplitude:   0.1,
    tint: [0.45, 0.8, 0.5, 0.88],    // sea green
  },

  // ── Layer norm / structural ───────────────────────────────────────────────
  'cil-layers': {
    scale:           0.75,
    tentacleLength:  0.8,
    bellRadius:      1.0,
    pulseFrequency:  0.5,
    translucency:    0.35,
    swayAmplitude:   0.06,
    tint: [0.7, 0.7, 0.75, 0.92],    // silver-grey, more opaque
  },

  // ── Residual / skip connection: mycelium-like ─────────────────────────────
  'cil-plus': {
    scale:           0.65,
    tentacleLength:  1.3,
    bellRadius:      0.7,
    pulseFrequency:  0.8,
    translucency:    0.6,
    swayAmplitude:   0.15,
    tint: [0.6, 0.35, 0.8, 0.8],     // purple, networked feel
  },

  // ── Routing / skip ────────────────────────────────────────────────────────
  'cil-arrow-right': {
    scale:           0.7,
    tentacleLength:  1.1,
    bellRadius:      0.85,
    pulseFrequency:  0.7,
    translucency:    0.5,
    swayAmplitude:   0.18,
    tint: [0.4, 0.65, 0.85, 0.82],   // cyan directional
  },

  // ── Filter / attention gate ───────────────────────────────────────────────
  'cil-filter': {
    scale:           0.8,
    tentacleLength:  0.9,
    bellRadius:      1.05,
    pulseFrequency:  0.55,
    translucency:    0.58,
    swayAmplitude:   0.09,
    tint: [0.35, 0.7, 0.9, 0.86],    // light blue
  },

  // ── Default fallback ─────────────────────────────────────────────────────
  default: {
    scale:           0.8,
    tentacleLength:  1.0,
    bellRadius:      1.0,
    pulseFrequency:  0.6,
    translucency:    0.5,
    swayAmplitude:   0.1,
    tint: [0.5, 0.6, 0.8, 0.85],
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// JellyfishInstance — runtime state for one jellyfish
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Per-jellyfish instance data.  Created via `createVariant()` and updated
 * each frame by `animate()`.
 */
export interface JellyfishInstance {
  /** Cell identifier this jellyfish represents. */
  cellId: string;

  /** Species string (for variant lookup). */
  species: string;

  /** Variant configuration (immutable after creation). */
  config: JellyfishVariantConfig;

  /** World-space position (centre of the cell AABB). */
  position: Float32Array;  // [x, y, z]

  /** Current scale (animated). */
  scale: number;

  /** Current rotation angle (radians, Y-axis). */
  rotation: number;

  /** Pulse phase accumulator (radians). */
  pulsePhase: number;

  /** Current bell expansion factor (0–1, driven by pulse). */
  bellExpansion: number;

  /** Column-major model matrix (computed each frame). */
  modelMatrix: Float32Array;  // 16 floats
}

// ─────────────────────────────────────────────────────────────────────────────
// Shaders — AT JellyShader-inspired WebGL2 GLSL 300 es
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Vertex shader.
 *
 * Adapted from AT JellyShader.glsl:
 *   • cnoise displacement → simplex-based bell pulse + lateral sway
 *   • pos.y += cnoise(pos * scale + time) — vertical ripple
 *   • pos.xz += sin/cos(pos.y + time) — lateral sway
 *
 * Instance attributes encode per-jellyfish transform and variant params.
 */
const JELLYFISH_VERT = /* glsl */ `#version 300 es
precision highp float;

// ── Per-vertex (from decoded jellyfish.bin geometry) ────────────────────────
in vec3 a_position;
in vec3 a_normal;
in vec2 a_uv;

// ── Per-instance (divisor = 1) ──────────────────────────────────────────────
in mat4  a_modelMatrix;   // locations 3–6
in vec4  a_color;         // location 7:  species tint RGBA
in float a_scale;         // location 8:  uniform scale
in float a_pulsePhase;    // location 9:  current pulse phase
in float a_tentacleLen;   // location 10: tentacle length multiplier
in float a_translucency;  // location 11: body translucency
in vec2  a_swayOffset;    // location 12: lateral sway offset (world)
in float a_bellRadius;    // location 13: bell dome radius multiplier
in float a_speciesIdx;    // location 14: species index for fragment

// ── Uniforms ────────────────────────────────────────────────────────────────
uniform mat4  u_view;
uniform mat4  u_projection;
uniform float u_time;

// ── Outputs → fragment ──────────────────────────────────────────────────────
out vec3  v_worldPos;
out vec3  v_normal;
out vec2  v_uv;
out vec4  v_color;
out float v_translucency;
out float v_bellExpansion;
out float v_fresnel;
out float v_depth;

// ─── Simplex noise (AT cnoise equivalent, simplified 3D) ────────────────────

vec3 mod289(vec3 x) { return x - floor(x / 289.0) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x / 289.0) * 289.0; }
vec4 permute(vec4 x) { return mod289((x * 34.0 + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float cnoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  vec3 i  = floor(v + dot(v, vec3(C.y)));
  vec3 x0 = v - i + dot(i, vec3(C.x));
  vec3 g  = step(x0.yzx, x0.xyz);
  vec3 l  = 1.0 - g;
  vec3 i1 = min(g, l.zxy);
  vec3 i2 = max(g, l.zxy);
  vec3 x1 = x0 - i1 + C.x;
  vec3 x2 = x0 - i2 + C.y;
  vec3 x3 = x0 - 0.5;
  i = mod289(i);
  vec4 p = permute(permute(permute(
      i.z + vec4(0.0, i1.z, i2.z, 1.0))
    + i.y + vec4(0.0, i1.y, i2.y, 1.0))
    + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  vec4 j = p - 49.0 * floor(p / 49.0);
  vec4 x_ = floor(j / 7.0);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x  = (x_ * 2.0 + 0.5) / 7.0 - 1.0;
  vec4 y  = (y_ * 2.0 + 0.5) / 7.0 - 1.0;
  vec4 h  = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 g0 = vec3(a0.xy, h.x);
  vec3 g1 = vec3(a0.zw, h.y);
  vec3 g2 = vec3(a1.xy, h.z);
  vec3 g3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(g0,g0), dot(g1,g1), dot(g2,g2), dot(g3,g3)));
  g0 *= norm.x; g1 *= norm.y; g2 *= norm.z; g3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3)));
}

void main() {
  vec3 pos = a_position;

  // ── AT JellyShader vertex displacement ──────────────────────────────────
  // Bell pulse: vertical noise-based ripple (from compiled.vs JellyShader)
  float t = u_time * 0.35;
  pos.y += cnoise(pos * vec3(0.1, 0.5, 0.1) * 0.8 + t) * 0.6;

  // Lateral sway (sinusoidal, AT JellyShader style)
  pos.x += sin(pos.y + u_time * 0.1) * 0.1;
  pos.z += cos(pos.y + u_time * 0.1) * 0.1;

  // Slow drift (AT large-scale sway)
  pos.x += sin(pos.y * 0.04 + u_time * 0.2) * 1.0;
  pos.z += cos(pos.y * 0.04 + u_time * 0.2) * 1.0;

  // ── Pulse expansion: bell dome breathes ─────────────────────────────────
  float pulse = sin(a_pulsePhase) * 0.5 + 0.5;  // 0–1
  // Only expand the upper part (bell) — vertices above y=0
  float bellMask = smoothstep(-0.1, 0.3, a_position.y);
  pos.xz *= 1.0 + bellMask * pulse * 0.15 * a_bellRadius;
  v_bellExpansion = pulse;

  // ── Tentacle stretch: lower vertices extend ─────────────────────────────
  float tentacleMask = smoothstep(0.0, -0.5, a_position.y);
  pos.y -= tentacleMask * (a_tentacleLen - 1.0) * abs(a_position.y);

  // ── Apply instance scale ────────────────────────────────────────────────
  pos *= a_scale;

  // ── Apply instance sway offset ──────────────────────────────────────────
  pos.xz += a_swayOffset;

  // ── World transform ─────────────────────────────────────────────────────
  vec4 worldPos = a_modelMatrix * vec4(pos, 1.0);
  v_worldPos = worldPos.xyz;

  // Normal transform (upper 3x3 of model matrix)
  mat3 normalMatrix = mat3(a_modelMatrix);
  v_normal = normalize(normalMatrix * a_normal);

  v_uv           = a_uv;
  v_color        = a_color;
  v_translucency = a_translucency;

  // Fresnel approximation (view-dependent)
  vec4 viewPos = u_view * worldPos;
  vec3 viewDir = normalize(-viewPos.xyz);
  v_fresnel = pow(1.0 - max(dot(v_normal, viewDir), 0.0), 3.0);
  v_depth   = -viewPos.z;

  gl_Position = u_projection * viewPos;
}
`;

/**
 * Fragment shader.
 *
 * AT JellyShader fragment adapted for WebGL2 / instanced:
 *   • Fresnel rim glow (AT getFresnel pow 5 → pow 3 for softer look)
 *   • Translucent body with species tint
 *   • Bell expansion modulates internal glow intensity
 *   • Rainbow colour function from AT JellyShader for accent highlights
 */
const JELLYFISH_FRAG = /* glsl */ `#version 300 es
precision highp float;

in vec3  v_worldPos;
in vec3  v_normal;
in vec2  v_uv;
in vec4  v_color;
in float v_translucency;
in float v_bellExpansion;
in float v_fresnel;
in float v_depth;

uniform float u_time;

out vec4 fragColor;

// ── AT JellyShader rainbowColor function (from compiled.vs) ─────────────────
vec3 rainbowColor(float t) {
  t = fract(t);
  if (t < 0.03) return mix(vec3(0.5, 0.0, 0.5), vec3(0.5, 0.0, 1.0), t / 0.03);
  else if (t < 0.06) return mix(vec3(0.5, 0.0, 1.0), vec3(0.0, 0.0, 1.0), (t - 0.03) / 0.03);
  else if (t < 0.09) return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), (t - 0.06) / 0.03);
  else if (t < 0.12) return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.09) / 0.03);
  else if (t < 0.18) return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.12) / 0.06);
  else if (t < 0.24) return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), (t - 0.18) / 0.06);
  else return mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.24) / 0.06);
}

void main() {
  // Base species colour
  vec3 baseColor = v_color.rgb;

  // ── Fresnel rim glow (AT JellyShader: pow(getFresnel(...), 5.0)) ────────
  // Softer exponent (3.0) for the 2D-projected cell context
  vec3 rimColor = mix(baseColor * 1.5, vec3(1.0), 0.3);
  vec3 fresnelGlow = rimColor * v_fresnel * 0.9;

  // ── Internal glow modulated by bell pulse ───────────────────────────────
  // When the bell contracts (low expansion), internal bioluminescence peaks
  float internalGlow = (1.0 - v_bellExpansion) * 0.3;
  vec3 glowColor = baseColor * (1.0 + internalGlow);

  // ── Subtle rainbow accent (AT JellyShader style) ────────────────────────
  float rainbowT = v_uv.y * 0.3 + u_time * 0.02;
  vec3 rainbow = rainbowColor(rainbowT) * 0.08;

  // ── Combine ─────────────────────────────────────────────────────────────
  vec3 color = glowColor + fresnelGlow + rainbow;

  // Soft-light blend to lift mid-tones (AT blendSoftLight analogue)
  color = pow(color * 1.3, vec3(1.4));

  // ── Translucency ────────────────────────────────────────────────────────
  // Bell centre is more translucent, edges (fresnel) are more opaque
  float alpha = v_color.a * (1.0 - v_translucency * (1.0 - v_fresnel * 0.6));

  // Depth-based fade for distant instances
  float depthFade = smoothstep(500.0, 100.0, v_depth);
  alpha *= depthFade;

  fragColor = vec4(color, alpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// ATJellyfishCell — main class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * ATJellyfishCell
 *
 * Loads the AT jellyfish.bin Draco geometry and renders it as the visual
 * representation of cells, replacing the rectangular rigid-body draw.
 *
 * The collision body (AABB) is unchanged — only the visual rendering
 * is replaced. The jellyfish mesh is centred on the cell's AABB centre
 * and scaled to fit within it.
 *
 * Usage:
 * ```ts
 * const jelly = new ATJellyfishCell();
 * await jelly.load();
 *
 * // Create a variant for a specific cell species
 * const inst = jelly.createVariant('self_attn', 'cil-eye', 320, 200);
 *
 * // In the render loop:
 * jelly.animate(time, inst);
 * jelly.render(gl, [inst], viewMatrix);
 * ```
 */
export class ATJellyfishCell {
  // ── GL state ────────────────────────────────────────────────────────────
  private gl: WebGL2RenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;

  // ── Geometry buffers ────────────────────────────────────────────────────
  private vboPositions: WebGLBuffer | null = null;
  private vboNormals: WebGLBuffer | null = null;
  private vboUVs: WebGLBuffer | null = null;
  private ibo: WebGLBuffer | null = null;
  private vboInstances: WebGLBuffer | null = null;

  // ── Decoded geometry ────────────────────────────────────────────────────
  private geometry: GeometryDescriptor | null = null;
  private indexCount = 0;
  private vertexCount = 0;

  // ── Instance buffer (CPU-side) ──────────────────────────────────────────
  private instanceData: Float32Array;
  private maxInstances: number;

  // ── Uniform locations ───────────────────────────────────────────────────
  private uView: WebGLUniformLocation | null = null;
  private uProjection: WebGLUniformLocation | null = null;
  private uTime: WebGLUniformLocation | null = null;

  // ── Draco decoder ───────────────────────────────────────────────────────
  private dracoThread: DracoThread | null = null;

  // ── Load state ──────────────────────────────────────────────────────────
  private _loaded = false;
  private _loadPromise: Promise<void> | null = null;

  constructor(maxInstances = MAX_INSTANCES) {
    this.maxInstances = maxInstances;
    this.instanceData = new Float32Array(maxInstances * FLOATS_PER_INSTANCE);
  }

  /** Whether the geometry has been loaded and GPU resources are ready. */
  get loaded(): boolean {
    return this._loaded;
  }

  // ───────────────────────────────────────────────────────────────────────
  // load() — fetch jellyfish.bin, Draco-decode, upload to GPU
  // ───────────────────────────────────────────────────────────────────────

  /**
   * Load the AT jellyfish.bin geometry and prepare GPU resources.
   *
   * The .bin format (AT convention):
   *   Bytes 0-3:   uint32 LE — header size (informational)
   *   Bytes 4-9:   padding (zeroes)
   *   Bytes 10-N:  JSON header {"name","type","attributes":[...]}
   *   Bytes N+1-:  Draco-compressed mesh
   *
   * The Draco payload starts immediately after the JSON closing brace.
   *
   * @param gl   WebGL2 rendering context (stored for subsequent renders)
   * @param binUrl  Override URL for the jellyfish.bin asset
   */
  async load(
    gl: WebGL2RenderingContext,
    binUrl: string = JELLYFISH_BIN_PATH,
  ): Promise<void> {
    // Deduplicate concurrent loads
    if (this._loadPromise) return this._loadPromise;
    this._loadPromise = this._doLoad(gl, binUrl);
    return this._loadPromise;
  }

  private async _doLoad(gl: WebGL2RenderingContext, binUrl: string): Promise<void> {
    this.gl = gl;

    // ── 1. Fetch the .bin ────────────────────────────────────────────────
    const response = await fetch(binUrl);
    if (!response.ok) {
      throw new Error(`ATJellyfishCell: failed to fetch ${binUrl} (${response.status})`);
    }
    const fullBuffer = await response.arrayBuffer();

    // ── 2. Parse the AT .bin header to find the Draco payload ───────────
    const bytes = new Uint8Array(fullBuffer);
    // Scan for the closing '}' of the JSON header starting from byte 10
    let jsonEnd = BIN_JSON_OFFSET;
    while (jsonEnd < bytes.length && bytes[jsonEnd] !== 0x7D /* '}' */) {
      jsonEnd++;
    }
    jsonEnd++; // include the '}'

    // The Draco payload starts right after the JSON header
    const dracoBuffer = fullBuffer.slice(jsonEnd);

    // ── 3. Decode via DracoThread (off-main-thread) ─────────────────────
    this.dracoThread = new DracoThread();
    try {
      this.geometry = await this.dracoThread.decode(dracoBuffer);
    } catch (err) {
      // DracoThread requires the Draco WASM decoder to be served at /draco/
      // If unavailable, fall back to a procedural jellyfish bell
      console.warn('ATJellyfishCell: Draco decode failed, using procedural fallback', err);
      this.geometry = this._createProceduralJellyfish();
    }

    this.vertexCount = this.geometry.vertexCount;
    this.indexCount = this.geometry.indices?.length ?? 0;

    // ── 4. Compile shaders ──────────────────────────────────────────────
    this.program = this._compileProgram(gl, JELLYFISH_VERT, JELLYFISH_FRAG);
    gl.useProgram(this.program);

    // ── 5. Cache uniform locations ──────────────────────────────────────
    this.uView       = gl.getUniformLocation(this.program, 'u_view');
    this.uProjection = gl.getUniformLocation(this.program, 'u_projection');
    this.uTime       = gl.getUniformLocation(this.program, 'u_time');

    // ── 6. Upload geometry to GPU ───────────────────────────────────────
    this._uploadGeometry(gl);

    // ── 7. Build VAO ────────────────────────────────────────────────────
    this._buildVAO(gl);

    this._loaded = true;
  }

  // ───────────────────────────────────────────────────────────────────────
  // createVariant() — create a jellyfish instance for a cell
  // ───────────────────────────────────────────────────────────────────────

  /**
   * Create a jellyfish instance for a specific cell.
   *
   * The variant is chosen by species string.  The instance is positioned
   * at (worldX, worldY) which should be the centre of the cell's AABB.
   *
   * @param cellId    Unique cell identifier.
   * @param species   Cell species string (e.g. 'cil-eye').
   * @param worldX    Cell centre X in world coordinates.
   * @param worldY    Cell centre Y in world coordinates.
   * @param worldZ    Optional Z depth (default 0).
   * @returns         Mutable JellyfishInstance for animate() and render().
   */
  createVariant(
    cellId: string,
    species: string,
    worldX: number,
    worldY: number,
    worldZ = 0,
  ): JellyfishInstance {
    const config = VARIANT_PRESETS[species] ?? VARIANT_PRESETS['default'];

    return {
      cellId,
      species,
      config,
      position: new Float32Array([worldX, worldY, worldZ]),
      scale: config.scale,
      rotation: Math.random() * Math.PI * 2, // random initial yaw
      pulsePhase: Math.random() * Math.PI * 2, // desynchronise pulses
      bellExpansion: 0.5,
      modelMatrix: new Float32Array(16),
    };
  }

  /**
   * Create multiple small jellyfish instances for an FFN cell (swarm).
   *
   * FFN cells produce parallel computation, visualised as a cluster of
   * small jellyfish within the cell AABB.
   *
   * @param cellId    Unique cell identifier.
   * @param species   Cell species string.
   * @param worldX    Cell centre X.
   * @param worldY    Cell centre Y.
   * @param cellW     Cell AABB width.
   * @param cellH     Cell AABB height.
   * @param count     Number of jellyfish in the swarm (default 5).
   * @returns         Array of JellyfishInstances.
   */
  createSwarm(
    cellId: string,
    species: string,
    worldX: number,
    worldY: number,
    cellW: number,
    cellH: number,
    count = 5,
  ): JellyfishInstance[] {
    const instances: JellyfishInstance[] = [];
    const baseConfig = VARIANT_PRESETS[species] ?? VARIANT_PRESETS['default'];

    for (let i = 0; i < count; i++) {
      // Scatter positions within the cell AABB (golden-ratio spiral)
      const angle = i * Math.PI * 2 * 0.618033988749895; // golden angle
      const radius = 0.3 * Math.sqrt((i + 0.5) / count);
      const offsetX = Math.cos(angle) * radius * cellW;
      const offsetY = Math.sin(angle) * radius * cellH;

      // Scale variation: 70%–100% of the base swarm scale
      const scaleVariation = 0.7 + Math.random() * 0.3;

      const config: JellyfishVariantConfig = {
        ...baseConfig,
        scale: baseConfig.scale * scaleVariation * 0.7, // swarm members are smaller
      };

      instances.push({
        cellId: `${cellId}_swarm_${i}`,
        species,
        config,
        position: new Float32Array([worldX + offsetX, worldY + offsetY, 0]),
        scale: config.scale,
        rotation: Math.random() * Math.PI * 2,
        pulsePhase: Math.random() * Math.PI * 2,
        bellExpansion: 0.5,
        modelMatrix: new Float32Array(16),
      });
    }

    return instances;
  }

  // ───────────────────────────────────────────────────────────────────────
  // animate() — update jellyfish animation state each frame
  // ───────────────────────────────────────────────────────────────────────

  /**
   * Update a single jellyfish instance's animation state.
   *
   * Advances the pulse phase, computes bell expansion, applies sway,
   * and rebuilds the model matrix.  Should be called once per frame
   * per instance before render().
   *
   * @param time     Elapsed time in seconds (monotonic).
   * @param instance The instance to animate.
   * @param dt       Delta time since last frame (seconds). Default 1/60.
   */
  animate(time: number, instance: JellyfishInstance, dt = 1 / 60): void {
    const cfg = instance.config;

    // ── Advance pulse ─────────────────────────────────────────────────
    instance.pulsePhase += cfg.pulseFrequency * Math.PI * 2 * dt;
    instance.bellExpansion = Math.sin(instance.pulsePhase) * 0.5 + 0.5;

    // ── Slow rotation drift ───────────────────────────────────────────
    instance.rotation += dt * 0.15;

    // ── Sway offset (AT JellyShader sin/cos lateral drift) ────────────
    const swayX = Math.sin(time * 0.3 + instance.pulsePhase) * cfg.swayAmplitude;
    const swayY = Math.cos(time * 0.2 + instance.pulsePhase * 0.7) * cfg.swayAmplitude * 0.5;

    // ── Build model matrix (translate × rotateY × scale) ──────────────
    const m = instance.modelMatrix;
    const s = instance.scale;
    const c = Math.cos(instance.rotation);
    const sn = Math.sin(instance.rotation);
    const px = instance.position[0] + swayX;
    const py = instance.position[1] + swayY;
    const pz = instance.position[2];

    // Column-major mat4: scale × rotateY × translate
    m[0]  = c * s;   m[1]  = 0;     m[2]  = -sn * s;  m[3]  = 0;
    m[4]  = 0;       m[5]  = s;     m[6]  = 0;         m[7]  = 0;
    m[8]  = sn * s;  m[9]  = 0;     m[10] = c * s;     m[11] = 0;
    m[12] = px;      m[13] = py;    m[14] = pz;        m[15] = 1;
  }

  // ───────────────────────────────────────────────────────────────────────
  // render() — draw all jellyfish instances in a single instanced call
  // ───────────────────────────────────────────────────────────────────────

  /**
   * Render jellyfish instances with GPU instancing.
   *
   * All instances are drawn in a single `gl.drawElementsInstanced()` call.
   * The instance buffer is updated each frame from the JellyfishInstance
   * data (model matrix + variant parameters).
   *
   * @param gl          WebGL2 context.
   * @param instances   Array of animated JellyfishInstances.
   * @param viewMatrix  4×4 view matrix (column-major Float32Array).
   * @param projMatrix  4×4 projection matrix (column-major Float32Array).
   * @param time        Current time in seconds (for shader animation).
   */
  render(
    gl: WebGL2RenderingContext,
    instances: JellyfishInstance[],
    viewMatrix: Float32Array,
    projMatrix?: Float32Array,
    time = 0,
  ): void {
    if (!this._loaded || !this.program || !this.vao || instances.length === 0) return;

    const count = Math.min(instances.length, this.maxInstances);

    // ── Pack instance data ────────────────────────────────────────────
    for (let i = 0; i < count; i++) {
      const inst = instances[i];
      const off = i * FLOATS_PER_INSTANCE;

      // mat4 model matrix (16 floats)
      this.instanceData.set(inst.modelMatrix, off);

      // vec4 colour
      const tint = inst.config.tint;
      this.instanceData[off + 16] = tint[0];
      this.instanceData[off + 17] = tint[1];
      this.instanceData[off + 18] = tint[2];
      this.instanceData[off + 19] = tint[3];

      // scale
      this.instanceData[off + 20] = inst.scale;
      // pulsePhase
      this.instanceData[off + 21] = inst.pulsePhase;
      // tentacleLength
      this.instanceData[off + 22] = inst.config.tentacleLength;
      // translucency
      this.instanceData[off + 23] = inst.config.translucency;
      // swayOffset (recomputed from instance position delta — already baked into modelMatrix,
      // but passed for additional vertex-shader fine-grain sway)
      this.instanceData[off + 24] = 0; // sway is in modelMatrix
      this.instanceData[off + 25] = 0;
      // bellRadius
      this.instanceData[off + 26] = inst.config.bellRadius;
      // speciesIdx (for future per-species fragment branching)
      this.instanceData[off + 27] = 0;
    }

    // ── Upload instance data ──────────────────────────────────────────
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboInstances);
    gl.bufferSubData(
      gl.ARRAY_BUFFER, 0,
      this.instanceData.subarray(0, count * FLOATS_PER_INSTANCE),
    );

    // ── Bind program & set uniforms ───────────────────────────────────
    gl.useProgram(this.program);

    gl.uniformMatrix4fv(this.uView, false, viewMatrix);
    if (projMatrix && this.uProjection) {
      gl.uniformMatrix4fv(this.uProjection, false, projMatrix);
    }
    if (this.uTime) {
      gl.uniform1f(this.uTime, time);
    }

    // ── Draw ──────────────────────────────────────────────────────────
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false); // translucent — don't write depth

    gl.bindVertexArray(this.vao);

    if (this.indexCount > 0) {
      gl.drawElementsInstanced(
        gl.TRIANGLES,
        this.indexCount,
        gl.UNSIGNED_INT,
        0,
        count,
      );
    } else {
      gl.drawArraysInstanced(gl.TRIANGLES, 0, this.vertexCount, count);
    }

    gl.bindVertexArray(null);
    gl.depthMask(true);
  }

  // ───────────────────────────────────────────────────────────────────────
  // dispose() — clean up GPU resources
  // ───────────────────────────────────────────────────────────────────────

  dispose(): void {
    const gl = this.gl;
    if (!gl) return;

    if (this.vao)          gl.deleteVertexArray(this.vao);
    if (this.vboPositions) gl.deleteBuffer(this.vboPositions);
    if (this.vboNormals)   gl.deleteBuffer(this.vboNormals);
    if (this.vboUVs)       gl.deleteBuffer(this.vboUVs);
    if (this.ibo)          gl.deleteBuffer(this.ibo);
    if (this.vboInstances) gl.deleteBuffer(this.vboInstances);
    if (this.program)      gl.deleteProgram(this.program);

    this.dracoThread?.dispose();

    this.vao = null;
    this.vboPositions = null;
    this.vboNormals = null;
    this.vboUVs = null;
    this.ibo = null;
    this.vboInstances = null;
    this.program = null;
    this.gl = null;
    this._loaded = false;
    this._loadPromise = null;
  }

  // ───────────────────────────────────────────────────────────────────────
  // Accessors
  // ───────────────────────────────────────────────────────────────────────

  /** Get variant config for a species (or default). */
  static getVariantConfig(species: string): JellyfishVariantConfig {
    return VARIANT_PRESETS[species] ?? VARIANT_PRESETS['default'];
  }

  /** Check if a morphology should use jellyfish rendering. */
  static isJellyfishMorphology(morphology: Morphology): boolean {
    return morphology === 'jellyfish';
  }

  /**
   * Map a VisualProfile to jellyfish rendering parameters.
   * Merges the profile's species-derived colours with the jellyfish
   * variant config for a unified visual.
   */
  static profileToVariant(
    profile: VisualProfile,
    species: string,
  ): JellyfishVariantConfig {
    const base = VARIANT_PRESETS[species] ?? VARIANT_PRESETS['default'];
    return {
      ...base,
      tint: [
        profile.colorPalette.base[0],
        profile.colorPalette.base[1],
        profile.colorPalette.base[2],
        base.tint[3] * (1 - profile.glowIntensity * 0.1),
      ],
      translucency: base.translucency * (1 - profile.borderSharpness * 0.3),
      // Faster pulse for higher flow speed
      pulseFrequency: base.pulseFrequency * (0.8 + profile.flowSpeed * 0.4),
    };
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Private helpers
  // ═══════════════════════════════════════════════════════════════════════

  /** Compile & link a WebGL2 program from vertex + fragment source. */
  private _compileProgram(
    gl: WebGL2RenderingContext,
    vertSrc: string,
    fragSrc: string,
  ): WebGLProgram {
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vertSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(vs);
      gl.deleteShader(vs);
      throw new Error(`ATJellyfishCell vertex shader error:\n${log}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(fs);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error(`ATJellyfishCell fragment shader error:\n${log}`);
    }

    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    gl.deleteShader(vs);
    gl.deleteShader(fs);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`ATJellyfishCell program link error:\n${log}`);
    }

    return program;
  }

  /** Upload decoded geometry to GPU buffers. */
  private _uploadGeometry(gl: WebGL2RenderingContext): void {
    const geo = this.geometry!;

    // Positions
    this.vboPositions = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboPositions);
    gl.bufferData(gl.ARRAY_BUFFER, geo.positions, gl.STATIC_DRAW);

    // Normals
    if (geo.normals) {
      this.vboNormals = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vboNormals);
      gl.bufferData(gl.ARRAY_BUFFER, geo.normals, gl.STATIC_DRAW);
    }

    // UVs
    if (geo.uvs) {
      this.vboUVs = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vboUVs);
      gl.bufferData(gl.ARRAY_BUFFER, geo.uvs, gl.STATIC_DRAW);
    }

    // Index buffer
    if (geo.indices) {
      this.ibo = gl.createBuffer()!;
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, geo.indices, gl.STATIC_DRAW);
    }

    // Instance buffer (dynamic, updated each frame)
    this.vboInstances = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboInstances);
    gl.bufferData(gl.ARRAY_BUFFER, this.instanceData.byteLength, gl.DYNAMIC_DRAW);
  }

  /** Build VAO wiring all vertex + instance attributes. */
  private _buildVAO(gl: WebGL2RenderingContext): void {
    const program = this.program!;
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);

    // ── Per-vertex attributes ─────────────────────────────────────────

    // a_position (location 0)
    const aPos = gl.getAttribLocation(program, 'a_position');
    if (aPos >= 0 && this.vboPositions) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vboPositions);
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    }

    // a_normal (location 1)
    const aNorm = gl.getAttribLocation(program, 'a_normal');
    if (aNorm >= 0 && this.vboNormals) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vboNormals);
      gl.enableVertexAttribArray(aNorm);
      gl.vertexAttribPointer(aNorm, 3, gl.FLOAT, false, 0, 0);
    }

    // a_uv (location 2)
    const aUV = gl.getAttribLocation(program, 'a_uv');
    if (aUV >= 0 && this.vboUVs) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vboUVs);
      gl.enableVertexAttribArray(aUV);
      gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, 0, 0);
    }

    // ── Index buffer ──────────────────────────────────────────────────
    if (this.ibo) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    }

    // ── Per-instance attributes (divisor = 1) ─────────────────────────
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboInstances);

    const stride = FLOATS_PER_INSTANCE * 4; // bytes

    // a_modelMatrix: mat4 → 4 × vec4 at locations 3–6
    const aModel = gl.getAttribLocation(program, 'a_modelMatrix');
    if (aModel >= 0) {
      for (let col = 0; col < 4; col++) {
        const loc = aModel + col;
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, stride, (col * 4) * 4);
        gl.vertexAttribDivisor(loc, 1);
      }
    }

    // a_color: vec4 at offset 16
    const aColor = gl.getAttribLocation(program, 'a_color');
    if (aColor >= 0) {
      gl.enableVertexAttribArray(aColor);
      gl.vertexAttribPointer(aColor, 4, gl.FLOAT, false, stride, 16 * 4);
      gl.vertexAttribDivisor(aColor, 1);
    }

    // a_scale: float at offset 20
    const aScale = gl.getAttribLocation(program, 'a_scale');
    if (aScale >= 0) {
      gl.enableVertexAttribArray(aScale);
      gl.vertexAttribPointer(aScale, 1, gl.FLOAT, false, stride, 20 * 4);
      gl.vertexAttribDivisor(aScale, 1);
    }

    // a_pulsePhase: float at offset 21
    const aPulse = gl.getAttribLocation(program, 'a_pulsePhase');
    if (aPulse >= 0) {
      gl.enableVertexAttribArray(aPulse);
      gl.vertexAttribPointer(aPulse, 1, gl.FLOAT, false, stride, 21 * 4);
      gl.vertexAttribDivisor(aPulse, 1);
    }

    // a_tentacleLen: float at offset 22
    const aTentacle = gl.getAttribLocation(program, 'a_tentacleLen');
    if (aTentacle >= 0) {
      gl.enableVertexAttribArray(aTentacle);
      gl.vertexAttribPointer(aTentacle, 1, gl.FLOAT, false, stride, 22 * 4);
      gl.vertexAttribDivisor(aTentacle, 1);
    }

    // a_translucency: float at offset 23
    const aTranslucency = gl.getAttribLocation(program, 'a_translucency');
    if (aTranslucency >= 0) {
      gl.enableVertexAttribArray(aTranslucency);
      gl.vertexAttribPointer(aTranslucency, 1, gl.FLOAT, false, stride, 23 * 4);
      gl.vertexAttribDivisor(aTranslucency, 1);
    }

    // a_swayOffset: vec2 at offset 24
    const aSway = gl.getAttribLocation(program, 'a_swayOffset');
    if (aSway >= 0) {
      gl.enableVertexAttribArray(aSway);
      gl.vertexAttribPointer(aSway, 2, gl.FLOAT, false, stride, 24 * 4);
      gl.vertexAttribDivisor(aSway, 1);
    }

    // a_bellRadius: float at offset 26
    const aBell = gl.getAttribLocation(program, 'a_bellRadius');
    if (aBell >= 0) {
      gl.enableVertexAttribArray(aBell);
      gl.vertexAttribPointer(aBell, 1, gl.FLOAT, false, stride, 26 * 4);
      gl.vertexAttribDivisor(aBell, 1);
    }

    // a_speciesIdx: float at offset 27
    const aSpecies = gl.getAttribLocation(program, 'a_speciesIdx');
    if (aSpecies >= 0) {
      gl.enableVertexAttribArray(aSpecies);
      gl.vertexAttribPointer(aSpecies, 1, gl.FLOAT, false, stride, 27 * 4);
      gl.vertexAttribDivisor(aSpecies, 1);
    }

    gl.bindVertexArray(null);
  }

  /**
   * Procedural jellyfish fallback when Draco decoding is unavailable.
   *
   * Generates a simple bell + tentacle mesh:
   *   Bell:      hemisphere (16 rings × 24 segments)
   *   Tentacles: 6 trailing strips hanging from the bell rim
   */
  private _createProceduralJellyfish(): GeometryDescriptor {
    const RINGS    = 16;
    const SEGMENTS = 24;
    const TENTACLES = 6;
    const TENT_SEGMENTS = 8;

    // Bell vertices
    const bellVerts  = (RINGS + 1) * (SEGMENTS + 1);
    const tentVerts  = TENTACLES * (TENT_SEGMENTS + 1) * 2; // strip pairs
    const totalVerts = bellVerts + tentVerts;

    const positions = new Float32Array(totalVerts * 3);
    const normals   = new Float32Array(totalVerts * 3);
    const uvs       = new Float32Array(totalVerts * 2);

    let vi = 0;

    // ── Bell hemisphere ───────────────────────────────────────────────
    for (let r = 0; r <= RINGS; r++) {
      const phi = (r / RINGS) * Math.PI * 0.5; // 0 to π/2
      const y = Math.cos(phi);
      const ringRadius = Math.sin(phi);

      for (let s = 0; s <= SEGMENTS; s++) {
        const theta = (s / SEGMENTS) * Math.PI * 2;
        const x = ringRadius * Math.cos(theta);
        const z = ringRadius * Math.sin(theta);

        positions[vi * 3]     = x;
        positions[vi * 3 + 1] = y * 0.6; // squash to dome shape
        positions[vi * 3 + 2] = z;

        normals[vi * 3]     = x;
        normals[vi * 3 + 1] = y;
        normals[vi * 3 + 2] = z;

        uvs[vi * 2]     = s / SEGMENTS;
        uvs[vi * 2 + 1] = r / RINGS;

        vi++;
      }
    }

    // ── Tentacles (hanging strips) ────────────────────────────────────
    for (let t = 0; t < TENTACLES; t++) {
      const baseAngle = (t / TENTACLES) * Math.PI * 2;
      const bx = Math.cos(baseAngle);
      const bz = Math.sin(baseAngle);

      for (let seg = 0; seg <= TENT_SEGMENTS; seg++) {
        const frac = seg / TENT_SEGMENTS;
        const hangY = -frac * 1.5; // hang down
        const sway = Math.sin(frac * Math.PI) * 0.1;
        const width = 0.03 * (1.0 - frac * 0.7);

        // Left edge
        positions[vi * 3]     = bx + sway - bz * width;
        positions[vi * 3 + 1] = hangY;
        positions[vi * 3 + 2] = bz + sway + bx * width;
        normals[vi * 3] = -bz; normals[vi * 3 + 1] = 0; normals[vi * 3 + 2] = bx;
        uvs[vi * 2] = 0; uvs[vi * 2 + 1] = frac;
        vi++;

        // Right edge
        positions[vi * 3]     = bx + sway + bz * width;
        positions[vi * 3 + 1] = hangY;
        positions[vi * 3 + 2] = bz + sway - bx * width;
        normals[vi * 3] = bz; normals[vi * 3 + 1] = 0; normals[vi * 3 + 2] = -bx;
        uvs[vi * 2] = 1; uvs[vi * 2 + 1] = frac;
        vi++;
      }
    }

    // ── Indices ────────────────────────────────────────────────────────
    const bellFaces = RINGS * SEGMENTS * 2;
    const tentFaces = TENTACLES * TENT_SEGMENTS * 2;
    const indices = new Uint32Array((bellFaces + tentFaces) * 3);
    let ii = 0;

    // Bell triangles
    for (let r = 0; r < RINGS; r++) {
      for (let s = 0; s < SEGMENTS; s++) {
        const a = r * (SEGMENTS + 1) + s;
        const b = a + SEGMENTS + 1;
        indices[ii++] = a;
        indices[ii++] = b;
        indices[ii++] = a + 1;
        indices[ii++] = a + 1;
        indices[ii++] = b;
        indices[ii++] = b + 1;
      }
    }

    // Tentacle triangles (triangle strips → pairs)
    const tentBase = bellVerts;
    for (let t = 0; t < TENTACLES; t++) {
      const tOff = tentBase + t * (TENT_SEGMENTS + 1) * 2;
      for (let seg = 0; seg < TENT_SEGMENTS; seg++) {
        const a = tOff + seg * 2;
        indices[ii++] = a;
        indices[ii++] = a + 2;
        indices[ii++] = a + 1;
        indices[ii++] = a + 1;
        indices[ii++] = a + 2;
        indices[ii++] = a + 3;
      }
    }

    return {
      positions,
      normals,
      uvs,
      indices: indices.subarray(0, ii),
      vertexCount: totalVerts,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factory
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create and load an ATJellyfishCell instance.
 *
 * ```ts
 * const jelly = await createATJellyfishCell(gl);
 * const inst = jelly.createVariant('self_attn', 'cil-eye', 320, 200);
 * ```
 */
export async function createATJellyfishCell(
  gl: WebGL2RenderingContext,
  binUrl?: string,
): Promise<ATJellyfishCell> {
  const cell = new ATJellyfishCell();
  await cell.load(gl, binUrl);
  return cell;
}
