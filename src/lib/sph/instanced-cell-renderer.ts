/**
 * instanced-cell-renderer.ts — M769: GPU Instanced Cell Renderer
 * ─────────────────────────────────────────────────────────────────────────────
 * Renders ALL cells in a single `gl.drawElementsInstanced()` call, regardless
 * of species count.  Every per-cell visual dimension — SDF shape, material
 * type, base colour, reaction-diffusion texture coordinates, physics-driven
 * modulation — is encoded as per-instance vertex attributes (divisor = 1).
 *
 * Performance design
 * ─────────────────────────────────────────────────────────────────────────────
 * • ONE draw call for the entire cell population (vs N per species before).
 * • Zero GPU state switches between cells — same program, same VAO, same
 *   textures for the full batch.
 * • The fragment shader branches on integer `v_sdfShape` / `v_materialType`
 *   to evaluate the correct SDF and shading model.  Modern GPUs handle this
 *   efficiently because all cells within an 8×8 warp/wavefront typically
 *   share the same species (spatial coherence from physics layout).
 * • Reaction-diffusion texture is bound once as a shared sampler; per-cell
 *   RD lookup offset + scale is passed via instance attributes (v_rdOffset,
 *   v_rdScale) so each cell samples its own sub-region.
 * • Instance buffer is a single interleaved Float32Array uploaded once per
 *   frame via `gl.bufferSubData` — no per-cell GPU buffer binds.
 *
 * Instance attribute layout (per cell, FLOATS_PER_CELL = 32 floats)
 * ─────────────────────────────────────────────────────────────────────────────
 *   Offset  Count  Name              Description
 *   ──────  ─────  ────              ───────────
 *    0      16     a_modelMatrix     Column-major mat4 (position/scale/rotation)
 *   16       4     a_color           RGBA base colour (linear, 0–1)
 *   20       1     a_opacity         Independent opacity multiplier
 *   21       1     a_sdfShape        SDF shape index (float-encoded integer)
 *   22       4     a_sdfParams       Shape-specific params (petals/iterations/etc)
 *   26       1     a_materialType    Material type index (0=matcap, 1=pbr, 2=iridescence)
 *   27       1     a_patternShader   Pattern shader index (0=none, 1=grayscott, …)
 *   28       2     a_rdOffset        Reaction-diffusion UV offset for this cell's sub-region
 *   30       2     a_rdScale         Reaction-diffusion UV scale for this cell's sub-region
 *
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/renderer/InstancedMesh.ts       — base WebGL2 instancing pattern
 *   src/lib/renderer/CellInstanceManager.ts — per-species instanced (replaced by this)
 *   src/lib/sph/particle-instancing.ts      — particle instanced pipeline (design parallel)
 *   src/lib/sph/species-shader-registry.ts  — SdfShape, MaterialType, PatternShader enums
 *   src/lib/sph/cell-visual-identity.ts     — VisualProfile derivation
 *   src/lib/sph/cell-material-system.ts     — CellSpecies material definitions
 *   src/lib/sph/reaction-diffusion.ts       — RD texture source
 *   src/lib/sph/organic-sdf.ts              — CPU-side SDF evaluation (GPU mirrors)
 *   src/lib/sph/physics-uniform-bridge.ts   — PhysicsUniforms driving runtime modulation
 *
 * Research: xiaodi #M769 — cell-pubsub-loop
 */








import type { SdfShape, MaterialType, PatternShader, SdfShapeParams, MaterialParams }
import type { VisualProfile }     from './cell-visual-identity';
import type { PhysicsUniforms }   from './physics-uniform-bridge';

// [orphan-precise]   from './species-shader-registry';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// [orphan-precise] /** Floats per cell instance in the interleaved buffer. */
export const FLOATS_PER_CELL = 32;

/** Default maximum cells the buffer can hold (resizable). */
const DEFAULT_MAX_CELLS = 256;

// ─── SDF shape → integer index (matches fragment shader switch) ──────────────

const SDF_SHAPE_INDEX: Record<SdfShape, number> = {
  flower:     0,
  koch:       1,
  julia:      2,
  supershape: 3,
  capsule:    4,
  hexagon:    5,
  star:       6,
  roundbox:   7,
  polygon:    8,
};

// ─── Material type → integer index ────────────────────────────────────────────

const MATERIAL_TYPE_INDEX: Record<MaterialType, number> = {
  matcap:      0,
  pbr:         1,
  iridescence: 2,
};

// ─── Pattern shader → integer index ──────────────────────────────────────────

const PATTERN_SHADER_INDEX: Record<PatternShader, number> = {
  'none':               0,
  'grayscott-species':  1,
  'supershape-species': 2,
  'voronoi-membrane':   3,
  'voronoi-natural':    4,
  'iq-palette-species': 5,
  'julia-background':   6,
  'turing-pattern':     7,
  'curl-trail':         8,
  'fluid-surface':      9,
  'caustics':          10,
};

// ─────────────────────────────────────────────────────────────────────────────
// Cell descriptor — the input data structure for each cell
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Bounding box in world space (canvas coordinate system: origin top-left,
 * Y down).  The renderer builds a model matrix from this.
 */
export interface CellBBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Complete cell descriptor.  Everything the renderer needs to draw one cell
 * in the instanced batch.
 */
export interface CellInstanceDescriptor {
  /** Unique cell identifier (for updateCell / hit-testing). */
  cellId: string;

  /** World-space bounding box. */
  bbox: CellBBox;

  /** Base fill colour RGBA (linear, 0–1). */
  color: [number, number, number, number];

  /** Opacity multiplier 0–1. */
  opacity: number;

  /** SDF silhouette shape. */
  sdfShape: SdfShape;

  /** SDF shape-specific tuning parameters. */
  sdfParams: SdfShapeParams;

  /** Surface material model. */
  materialType: MaterialType;

  /** Pattern shader rendered inside the SDF mask. */
  patternShader: PatternShader;

  /**
   * Reaction-diffusion texture sub-region offset (normalised UV).
   * [0, 0] = sample from top-left of the RD texture.
   */
  rdOffset: [number, number];

  /**
   * Reaction-diffusion texture sub-region scale (normalised UV).
   * [1, 1] = sample the full RD texture; smaller = zoom into a sub-tile.
   */
  rdScale: [number, number];
}

// ─────────────────────────────────────────────────────────────────────────────
// Vertex shader — single-draw-call instanced cell rendering
// ─────────────────────────────────────────────────────────────────────────────

export const CELL_INSTANCED_VERT = /* glsl */ `#version 300 es
precision highp float;

// ── Per-vertex (shared geometry — unit quad) ─────────────────────────────────
in vec2 a_position;      // [-0.5, 0.5] unit quad corners
in vec2 a_uv;            // [0, 1] UV coordinates

// ── Per-instance (divisor = 1) ───────────────────────────────────────────────
in mat4  a_modelMatrix;  // 4 × vec4 slots (location 2–5)
in vec4  a_color;        // RGBA base colour
in float a_opacity;      // opacity multiplier
in float a_sdfShape;     // SDF shape index (integer encoded as float)
in vec4  a_sdfParams;    // shape-specific params (petals, iterations, etc)
in float a_materialType; // material type index
in float a_patternShader;// pattern shader index
in vec2  a_rdOffset;     // reaction-diffusion UV offset
in vec2  a_rdScale;      // reaction-diffusion UV scale

// ── Uniforms ─────────────────────────────────────────────────────────────────
uniform mat4  u_view;
uniform mat4  u_projection;
uniform float u_time;

// ── Outputs to fragment ──────────────────────────────────────────────────────
out vec2  v_uv;
out vec4  v_color;
out float v_opacity;
out float v_sdfShape;
out vec4  v_sdfParams;
out float v_materialType;
out float v_patternShader;
out vec2  v_rdUV;           // final RD texture UV (offset + scaled)
out vec3  v_worldPos;
out float v_time;

void main() {
  v_uv            = a_uv;
  v_color         = a_color;
  v_opacity       = a_opacity;
  v_sdfShape      = a_sdfShape;
  v_sdfParams     = a_sdfParams;
  v_materialType  = a_materialType;
  v_patternShader = a_patternShader;
  v_time          = u_time;

  // Compute reaction-diffusion UV from the cell's sub-region
  v_rdUV = a_rdOffset + a_uv * a_rdScale;

  // World position (for pattern evaluation / lighting)
  vec4 worldPos = a_modelMatrix * vec4(a_position, 0.0, 1.0);
  v_worldPos = worldPos.xyz;

  gl_Position = u_projection * u_view * worldPos;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Fragment shader — SDF evaluation + material shading + pattern overlay
// ─────────────────────────────────────────────────────────────────────────────

export const CELL_INSTANCED_FRAG = /* glsl */ `#version 300 es
precision highp float;

in vec2  v_uv;
in vec4  v_color;
in float v_opacity;
in float v_sdfShape;
in vec4  v_sdfParams;
in float v_materialType;
in float v_patternShader;
in vec2  v_rdUV;
in vec3  v_worldPos;
in float v_time;

uniform sampler2D u_rdTexture;       // reaction-diffusion texture (R=u, G=v)
uniform sampler2D u_matcapTexture;   // matcap environment map
uniform vec3      u_lightDir;        // directional light (normalised)
uniform float     u_fresnelPower;    // global Fresnel power
uniform float     u_bloomThreshold;  // emission threshold for bloom pass

out vec4 fragColor;

// ─── Constants ───────────────────────────────────────────────────────────────
const float PI     = 3.14159265359;
const float TWO_PI = 6.28318530718;

// ─── SDF primitives ──────────────────────────────────────────────────────────

float sdRoundBox(vec2 p, vec2 b, float r) {
  vec2 q = abs(p) - b + r;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

float sdFlower(vec2 p, float petals) {
  float r = length(p) * 2.0;
  float a = atan(p.y, p.x);
  float v = petals * 0.5;
  return 1.0 - (abs(cos(a * v)) * 0.5 + 0.5) / max(r, 0.001);
}

float sdHexagon(vec2 p, float radius) {
  vec2 q = abs(p);
  float d = max(q.x * 0.866025 + q.y * 0.5, q.y) - radius;
  return d;
}

float sdStar(vec2 p, float r, float points, float innerRatio) {
  float an = PI / points;
  float en = PI / (points >= 3.0 ? points : 3.0);
  vec2 acs = vec2(cos(an), sin(an));
  vec2 ecs = vec2(cos(en), sin(en));
  float bn = mod(atan(p.y, p.x), 2.0 * an) - an;
  p = length(p) * vec2(cos(bn), abs(sin(bn)));
  p -= r * acs;
  p += ecs * clamp(-dot(p, ecs), 0.0, r * acs.y / ecs.y);
  return length(p) * sign(p.x);
}

float sdSupershape(vec2 p, float m, float n1, float n2, float n3) {
  float a = atan(p.y, p.x);
  float r = length(p);
  float t = m * a / 4.0;
  float cs = pow(abs(cos(t)), n2);
  float sn = pow(abs(sin(t)), n3);
  float rn = pow(cs + sn, -1.0 / n1);
  return r - rn * 0.45;
}

float sdCapsule(vec2 p, vec2 a, vec2 b, float r) {
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

float sdKoch(vec2 p, float iterations) {
  // Simplified Koch snowflake SDF approximation
  p = abs(p);
  float d = dot(p, normalize(vec2(1.0, 1.732)));
  float scale = 1.0;
  for (float i = 0.0; i < 4.0; i++) {
    if (i >= iterations) break;
    p = abs(p);
    p -= vec2(0.5, 0.2887);
    p = abs(p);
    d = min(d, dot(p, normalize(vec2(1.0, 1.732))));
    p *= 2.0;
    scale *= 2.0;
  }
  return (d - 0.5) / scale;
}

float sdJulia(vec2 p, float cx, float cy) {
  // Escape-time Julia set SDF approximation
  vec2 z = p * 2.5;
  vec2 c = vec2(cx, cy);
  float dz2 = 1.0;
  float z2 = dot(z, z);
  for (int i = 0; i < 32; i++) {
    dz2 *= 4.0 * z2;
    z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
    z2 = dot(z, z);
    if (z2 > 256.0) break;
  }
  return 0.25 * sqrt(z2 / max(dz2, 1e-6)) * log(max(z2, 1e-6)) - 0.01;
}

float sdPolygon(vec2 p, float sides) {
  float an = PI / sides;
  float bn = mod(atan(p.y, p.x), 2.0 * an) - an;
  float r = length(p);
  return r * cos(bn) - 0.45;
}

// Evaluate the SDF for a cell based on its shape index and params
float evaluateSDF(vec2 uv, int shape, vec4 params) {
  vec2 p = uv * 2.0 - 1.0; // map [0,1] → [-1,1]

  // shape 0: flower
  if (shape == 0) return sdFlower(p, max(params.x, 3.0));
  // shape 1: koch
  if (shape == 1) return sdKoch(p, clamp(params.x, 1.0, 4.0));
  // shape 2: julia
  if (shape == 2) return sdJulia(p, params.x, params.y);
  // shape 3: supershape
  if (shape == 3) return sdSupershape(p, params.x, params.y, params.z, params.w);
  // shape 4: capsule
  if (shape == 4) return sdCapsule(p, vec2(-0.35, 0.0), vec2(0.35, 0.0), 0.25);
  // shape 5: hexagon
  if (shape == 5) return sdHexagon(p, max(params.x, 0.4));
  // shape 6: star
  if (shape == 6) return sdStar(p, 0.45, max(params.x, 5.0), clamp(params.y, 0.2, 0.8));
  // shape 7: roundbox (default)
  if (shape == 7) return sdRoundBox(p, vec2(0.85, 0.75), clamp(params.x, 0.02, 0.4));
  // shape 8: polygon
  if (shape == 8) return sdPolygon(p, max(params.x, 3.0));

  // Fallback: rounded box
  return sdRoundBox(p, vec2(0.85, 0.75), 0.12);
}

// ─── Pattern evaluation ──────────────────────────────────────────────────────

// Simplified Gray-Scott reaction-diffusion lookup
vec3 patternGrayScott(vec2 rdUV) {
  vec2 rd = texture(u_rdTexture, rdUV).rg;
  // Visualise: U channel as blue, V channel as warm orange
  float u = rd.r;
  float v = rd.g;
  vec3 col = mix(
    vec3(0.02, 0.04, 0.12),  // low-U background
    vec3(0.95, 0.60, 0.15),  // high-V active spots
    smoothstep(0.15, 0.55, v)
  );
  col = mix(col, vec3(0.20, 0.45, 0.85), smoothstep(0.3, 0.8, u) * 0.4);
  return col;
}

// Voronoi pattern (GPU-evaluated)
float voronoiHash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

vec3 patternVoronoi(vec2 uv, float time) {
  vec2 ip = floor(uv * 5.0);
  vec2 fp = fract(uv * 5.0);
  float minDist = 1.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 neighbor = vec2(float(x), float(y));
      vec2 point = vec2(
        voronoiHash(ip + neighbor),
        voronoiHash(ip + neighbor + vec2(37.0, 59.0))
      );
      point = 0.5 + 0.5 * sin(time * 0.4 + TWO_PI * point);
      float d = length(neighbor + point - fp);
      minDist = min(minDist, d);
    }
  }
  float edge = smoothstep(0.0, 0.06, minDist);
  return mix(vec3(0.7, 0.85, 0.95), vec3(0.15, 0.20, 0.30), edge);
}

// Supershape noise pattern
vec3 patternSupershape(vec2 uv, float time) {
  vec2 p = uv * 2.0 - 1.0;
  float a = atan(p.y, p.x);
  float r = length(p);
  float n = sin(a * 6.0 + time * 0.5) * 0.5 + 0.5;
  n *= sin(r * 8.0 - time * 0.3) * 0.5 + 0.5;
  return mix(vec3(0.08, 0.12, 0.25), vec3(0.85, 0.45, 0.90), n);
}

// IQ cosine palette
vec3 iqPalette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
  return a + b * cos(TWO_PI * (c * t + d));
}

vec3 patternIQPalette(vec2 uv, float time) {
  float t = length(uv - 0.5) * 2.0 + time * 0.15;
  return iqPalette(t,
    vec3(0.5, 0.5, 0.5),
    vec3(0.5, 0.5, 0.5),
    vec3(1.0, 1.0, 1.0),
    vec3(0.00, 0.33, 0.67)
  );
}

// Curl-trail noise (simplified)
vec3 patternCurlTrail(vec2 uv, float time) {
  vec2 p = uv * 4.0;
  float n = sin(p.x + sin(p.y * 1.3 + time * 0.3)) *
            cos(p.y + cos(p.x * 0.7 + time * 0.2));
  n = n * 0.5 + 0.5;
  return mix(vec3(0.02, 0.08, 0.15), vec3(0.55, 0.80, 1.0), n * n);
}

// Fluid surface normal approximation
vec3 patternFluidSurface(vec2 uv, float time) {
  vec2 p = uv * 3.0;
  float w1 = sin(p.x * 2.1 + time * 0.5) * cos(p.y * 1.7 + time * 0.3);
  float w2 = sin(p.x * 1.3 - time * 0.4) * cos(p.y * 2.9 + time * 0.6);
  float h = (w1 + w2) * 0.25 + 0.5;
  return mix(vec3(0.05, 0.15, 0.35), vec3(0.50, 0.80, 0.95), h);
}

// Caustic light transport approximation
vec3 patternCaustics(vec2 uv, float time) {
  vec2 p = uv * 6.0;
  float c = 0.0;
  for (float i = 1.0; i < 4.0; i++) {
    p = vec2(
      p.x + sin(p.y * i + time * (0.3 + i * 0.1)),
      p.y + cos(p.x * i + time * (0.2 + i * 0.1))
    );
    c += 1.0 / length(fract(p) - 0.5);
  }
  c = c / 3.0;
  c = pow(clamp(c * 0.15, 0.0, 1.0), 1.5);
  return mix(vec3(0.02, 0.06, 0.18), vec3(0.70, 0.90, 1.00), c);
}

// Turing spot/stripe pattern
vec3 patternTuring(vec2 uv, float time) {
  vec2 p = uv * 8.0;
  float v = sin(p.x + sin(p.y * 2.0 + time * 0.1)) *
            sin(p.y + sin(p.x * 2.0 - time * 0.1));
  v = v * 0.5 + 0.5;
  v = smoothstep(0.35, 0.65, v);
  return mix(vec3(0.10, 0.06, 0.02), vec3(0.85, 0.75, 0.55), v);
}

// Julia background field
vec3 patternJulia(vec2 uv, float time) {
  vec2 z = (uv - 0.5) * 3.0;
  vec2 c = vec2(-0.7 + sin(time * 0.05) * 0.1, 0.27 + cos(time * 0.07) * 0.05);
  float iter = 0.0;
  for (int i = 0; i < 24; i++) {
    z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
    if (dot(z, z) > 4.0) break;
    iter += 1.0;
  }
  float t = iter / 24.0;
  return iqPalette(t,
    vec3(0.2, 0.1, 0.4),
    vec3(0.5, 0.5, 0.5),
    vec3(1.0, 1.0, 0.5),
    vec3(0.80, 0.90, 0.30)
  );
}

// Dispatch pattern evaluation based on pattern index
vec3 evaluatePattern(int pattern, vec2 uv, vec2 rdUV, float time) {
  if (pattern == 0)  return vec3(1.0);                         // none
  if (pattern == 1)  return patternGrayScott(rdUV);            // grayscott-species
  if (pattern == 2)  return patternSupershape(uv, time);       // supershape-species
  if (pattern == 3)  return patternVoronoi(uv, time);          // voronoi-membrane
  if (pattern == 4)  return patternVoronoi(uv * 1.5, time);   // voronoi-natural (denser)
  if (pattern == 5)  return patternIQPalette(uv, time);        // iq-palette-species
  if (pattern == 6)  return patternJulia(uv, time);            // julia-background
  if (pattern == 7)  return patternTuring(uv, time);           // turing-pattern
  if (pattern == 8)  return patternCurlTrail(uv, time);        // curl-trail
  if (pattern == 9)  return patternFluidSurface(uv, time);     // fluid-surface
  if (pattern == 10) return patternCaustics(uv, time);         // caustics
  return vec3(1.0);
}

// ─── Material shading ────────────────────────────────────────────────────────

// Approximate Fresnel (Schlick)
float fresnelSchlick(float cosTheta, float F0) {
  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// Matcap shading: sample matcap texture from view-space normal
vec3 shadingMatcap(vec3 baseColor, vec2 uv) {
  // Derive a pseudo-normal from the UV quad (sphere-map approximation)
  vec2 n = uv * 2.0 - 1.0;
  float z2 = 1.0 - dot(n, n);
  if (z2 < 0.0) return baseColor;
  vec2 matcapUV = n * 0.5 + 0.5;
  vec3 matcap = texture(u_matcapTexture, matcapUV).rgb;
  return baseColor * matcap * 1.4;
}

// PBR Cook-Torrance approximation
vec3 shadingPBR(vec3 baseColor, vec2 uv, float time) {
  vec2 p = uv * 2.0 - 1.0;
  float z2 = max(0.0, 1.0 - dot(p, p));
  vec3 N = normalize(vec3(p, sqrt(z2)));
  vec3 V = vec3(0.0, 0.0, 1.0);
  vec3 L = normalize(u_lightDir);

  float NdotL = max(dot(N, L), 0.0);
  float NdotV = max(dot(N, V), 0.0);

  // Diffuse (Lambert)
  vec3 diffuse = baseColor * NdotL * 0.6;

  // Specular (Blinn-Phong approximation of Cook-Torrance)
  vec3 H = normalize(L + V);
  float NdotH = max(dot(N, H), 0.0);
  float spec = pow(NdotH, 64.0) * 0.8;

  // Fresnel rim
  float fresnel = fresnelSchlick(NdotV, 0.04) * u_fresnelPower;

  vec3 ambient = baseColor * 0.25;
  return ambient + diffuse + vec3(spec) + vec3(fresnel * 0.3);
}

// Iridescence: PBR base + thin-film interference
vec3 shadingIridescence(vec3 baseColor, vec2 uv, float time) {
  vec3 pbr = shadingPBR(baseColor, uv, time);

  // Thin-film interference approximation
  vec2 p = uv * 2.0 - 1.0;
  float z2 = max(0.0, 1.0 - dot(p, p));
  vec3 N = normalize(vec3(p, sqrt(z2)));
  vec3 V = vec3(0.0, 0.0, 1.0);
  float cosI = max(dot(N, V), 0.0);

  // Film thickness modulated by time and position
  float thickness = 420.0 + sin(time * 0.38 + uv.x * PI * 3.1) * 55.0;

  // Approximate spectral decomposition into RGB
  float phase = TWO_PI * thickness / 550.0 * (1.0 - cosI * cosI);
  vec3 irid = vec3(
    0.5 + 0.5 * cos(phase),
    0.5 + 0.5 * cos(phase + TWO_PI / 3.0),
    0.5 + 0.5 * cos(phase + 2.0 * TWO_PI / 3.0)
  );

  return pbr + irid * 0.35;
}

// Dispatch material shading
vec3 evaluateMaterial(int matType, vec3 baseColor, vec2 uv, float time) {
  if (matType == 0) return shadingMatcap(baseColor, uv);
  if (matType == 1) return shadingPBR(baseColor, uv, time);
  if (matType == 2) return shadingIridescence(baseColor, uv, time);
  return baseColor; // fallback
}

// ─── Main fragment ───────────────────────────────────────────────────────────

void main() {
  int sdfIdx     = int(v_sdfShape + 0.5);
  int matIdx     = int(v_materialType + 0.5);
  int patternIdx = int(v_patternShader + 0.5);

  // ── SDF evaluation ─────────────────────────────────────────────────────────
  float d = evaluateSDF(v_uv, sdfIdx, v_sdfParams);

  // Anti-aliased alpha from SDF distance
  float aaWidth = fwidth(d) * 1.5;
  float alpha = 1.0 - smoothstep(-aaWidth, aaWidth, d);

  if (alpha < 0.005) discard;

  // ── Pattern overlay ────────────────────────────────────────────────────────
  vec3 patternColor = evaluatePattern(patternIdx, v_uv, v_rdUV, v_time);

  // Modulate base colour by pattern (multiplicative blend for 'none')
  vec3 baseColor = v_color.rgb * patternColor;

  // ── Material shading ───────────────────────────────────────────────────────
  vec3 shadedColor = evaluateMaterial(matIdx, baseColor, v_uv, v_time);

  // ── Edge glow (soft Fresnel-like rim on the SDF boundary) ──────────────────
  float edgeDist = abs(d);
  float edgeGlow = exp(-edgeDist * 18.0) * 0.35;
  shadedColor += v_color.rgb * edgeGlow;

  // ── Final composite ────────────────────────────────────────────────────────
  float finalAlpha = alpha * v_opacity * v_color.a;
  fragColor = vec4(shadedColor, finalAlpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Parse "#RRGGBB" or "#RGB" → [r, g, b, 1] in linear 0–1 range. */
export function hexToLinearRGBA(hex: string): [number, number, number, number] {
  const h = hex.replace('#', '');
  let r: number, g: number, b: number;
  if (h.length === 3) {
    r = parseInt(h[0] + h[0], 16) / 255;
    g = parseInt(h[1] + h[1], 16) / 255;
    b = parseInt(h[2] + h[2], 16) / 255;
  } else {
    r = parseInt(h.slice(0, 2), 16) / 255;
    g = parseInt(h.slice(2, 4), 16) / 255;
    b = parseInt(h.slice(4, 6), 16) / 255;
  }
  // sRGB → linear approximation (gamma 2.2)
  return [Math.pow(r, 2.2), Math.pow(g, 2.2), Math.pow(b, 2.2), 1.0];
}

/**
 * Build a column-major model matrix from a CellBBox.
 * Translates to the bbox centre and scales to (w, h).
 */
function bboxToModelMatrix(bbox: CellBBox, out: Float32Array, offset: number): void {
  const cx = bbox.x + bbox.w / 2;
  const cy = bbox.y + bbox.h / 2;
  const w  = bbox.w;
  const h  = bbox.h;

  // Column-major 4×4:
  // [ w  0  0  0 ]   col 0
  // [ 0  h  0  0 ]   col 1
  // [ 0  0  1  0 ]   col 2
  // [ cx cy 0  1 ]   col 3
  out[offset +  0] = w;   out[offset +  1] = 0;   out[offset +  2] = 0;   out[offset +  3] = 0;
  out[offset +  4] = 0;   out[offset +  5] = h;   out[offset +  6] = 0;   out[offset +  7] = 0;
  out[offset +  8] = 0;   out[offset +  9] = 0;   out[offset + 10] = 1;   out[offset + 11] = 0;
  out[offset + 12] = cx;  out[offset + 13] = cy;  out[offset + 14] = 0;   out[offset + 15] = 1;
}

/**
 * Pack SDF shape parameters into 4 floats based on shape type.
 */
function packSdfParams(shape: SdfShape, params: SdfShapeParams): [number, number, number, number] {
  switch (shape) {
    case 'flower':
      return [params.petals ?? 6, 0, 0, 0];
    case 'koch':
      return [params.kochIterations ?? 3, 0, 0, 0];
    case 'julia':
      return [params.juliaRe ?? -0.7, params.juliaIm ?? 0.27, 0, 0];
    case 'supershape':
      return [
        params.supershapeM  ?? 6,
        params.supershapeN1 ?? 1,
        params.supershapeN2 ?? 1,
        params.supershapeN3 ?? 1,
      ];
    case 'capsule':
      return [0, 0, 0, 0];
    case 'hexagon':
      return [params.radius ?? 0.45, 0, 0, 0];
    case 'star':
      return [params.starPoints ?? 5, params.starRatio ?? 0.5, 0, 0];
    case 'roundbox':
      return [params.cornerRadius ?? 0.12, 0, 0, 0];
    case 'polygon':
      return [params.radius ?? 6, 0, 0, 0]; // sides count
    default:
      return [0.12, 0, 0, 0];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// InstancedCellRenderer — the main class
// ─────────────────────────────────────────────────────────────────────────────

export class InstancedCellRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;

  // ── Geometry ────────────────────────────────────────────────────────────────
  private vao: WebGLVertexArrayObject;
  private vboGeom: WebGLBuffer;
  private ibo: WebGLBuffer;
  private indexCount = 6;

  // ── Instancing ──────────────────────────────────────────────────────────────
  private vboInstances: WebGLBuffer;
  private instanceBuffer: Float32Array;
  private _maxCells: number;
  private _cellCount = 0;

  // ── Cell index ──────────────────────────────────────────────────────────────
  /** cellId → instance index for O(1) updates */
  private cellIndex = new Map<string, number>();

  // ── Uniforms ────────────────────────────────────────────────────────────────
  private uView:           WebGLUniformLocation | null;
  private uProjection:     WebGLUniformLocation | null;
  private uTime:           WebGLUniformLocation | null;
  private uRdTexture:      WebGLUniformLocation | null;
  private uMatcapTexture:  WebGLUniformLocation | null;
  private uLightDir:       WebGLUniformLocation | null;
  private uFresnelPower:   WebGLUniformLocation | null;
  private uBloomThreshold: WebGLUniformLocation | null;

  // ── Textures ────────────────────────────────────────────────────────────────
  private rdTexture: WebGLTexture | null = null;
  private matcapTexture: WebGLTexture | null = null;

  // ── Identity matrix ────────────────────────────────────────────────────────
  private static IDENTITY = new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);

  // ── Attribute locations (cached for bind) ──────────────────────────────────
  private aPosition     = -1;
  private aUV           = -1;
  private aModelMatrix  = -1;
  private aColor        = -1;
  private aOpacity      = -1;
  private aSdfShape     = -1;
  private aSdfParams    = -1;
  private aMaterialType = -1;
  private aPatternShader= -1;
  private aRdOffset     = -1;
  private aRdScale      = -1;

  // ──────────────────────────────────────────────────────────────────────────
  // Constructor
  // ──────────────────────────────────────────────────────────────────────────

  constructor(
    gl: WebGL2RenderingContext,
    maxCells: number = DEFAULT_MAX_CELLS,
  ) {
    this.gl = gl;
    this._maxCells = maxCells;
    this.instanceBuffer = new Float32Array(maxCells * FLOATS_PER_CELL);

    // ── Compile program ────────────────────────────────────────────────────
    this.program = this._compileProgram(CELL_INSTANCED_VERT, CELL_INSTANCED_FRAG);

    // ── Cache attribute locations ──────────────────────────────────────────
    this.aPosition      = gl.getAttribLocation(this.program, 'a_position');
    this.aUV            = gl.getAttribLocation(this.program, 'a_uv');
    this.aModelMatrix   = gl.getAttribLocation(this.program, 'a_modelMatrix');
    this.aColor         = gl.getAttribLocation(this.program, 'a_color');
    this.aOpacity       = gl.getAttribLocation(this.program, 'a_opacity');
    this.aSdfShape      = gl.getAttribLocation(this.program, 'a_sdfShape');
    this.aSdfParams     = gl.getAttribLocation(this.program, 'a_sdfParams');
    this.aMaterialType  = gl.getAttribLocation(this.program, 'a_materialType');
    this.aPatternShader = gl.getAttribLocation(this.program, 'a_patternShader');
    this.aRdOffset      = gl.getAttribLocation(this.program, 'a_rdOffset');
    this.aRdScale       = gl.getAttribLocation(this.program, 'a_rdScale');

    // ── Cache uniform locations ────────────────────────────────────────────
    this.uView           = gl.getUniformLocation(this.program, 'u_view');
    this.uProjection     = gl.getUniformLocation(this.program, 'u_projection');
    this.uTime           = gl.getUniformLocation(this.program, 'u_time');
    this.uRdTexture      = gl.getUniformLocation(this.program, 'u_rdTexture');
    this.uMatcapTexture  = gl.getUniformLocation(this.program, 'u_matcapTexture');
    this.uLightDir       = gl.getUniformLocation(this.program, 'u_lightDir');
    this.uFresnelPower   = gl.getUniformLocation(this.program, 'u_fresnelPower');
    this.uBloomThreshold = gl.getUniformLocation(this.program, 'u_bloomThreshold');

    // ── Build geometry (unit quad: 4 vertices, 6 indices) ──────────────────
    // prettier-ignore
    const quadVerts = new Float32Array([
      // x      y      u    v
      -0.5,  -0.5,   0.0, 0.0,
       0.5,  -0.5,   1.0, 0.0,
       0.5,   0.5,   1.0, 1.0,
      -0.5,   0.5,   0.0, 1.0,
    ]);
    const quadIdx = new Uint16Array([0, 1, 2, 2, 3, 0]);

    this.vboGeom = gl.createBuffer()!;
    this.ibo     = gl.createBuffer()!;

    // ── Build VAO ──────────────────────────────────────────────────────────
    this.vao = gl.createVertexArray()!;
    this.vboInstances = gl.createBuffer()!;

    gl.bindVertexArray(this.vao);

    // Geometry buffer
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboGeom);
    gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, quadIdx, gl.STATIC_DRAW);

    const geomStride = 4 * 4; // 4 floats × 4 bytes
    if (this.aPosition >= 0) {
      gl.enableVertexAttribArray(this.aPosition);
      gl.vertexAttribPointer(this.aPosition, 2, gl.FLOAT, false, geomStride, 0);
    }
    if (this.aUV >= 0) {
      gl.enableVertexAttribArray(this.aUV);
      gl.vertexAttribPointer(this.aUV, 2, gl.FLOAT, false, geomStride, 8);
    }

    // Instance buffer
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboInstances);
    gl.bufferData(gl.ARRAY_BUFFER, this.instanceBuffer.byteLength, gl.DYNAMIC_DRAW);
    this._bindInstanceAttribs();

    gl.bindVertexArray(null);

    // ── Create placeholder textures ────────────────────────────────────────
    this._createPlaceholderTextures();
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Public API — Data loading
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Load all cells from an array of descriptors.  Rebuilds the instance buffer.
   * Idempotent — safe to call repeatedly (e.g. on topology change).
   */
  loadCells(descriptors: CellInstanceDescriptor[]): void {
    this._ensureCapacity(descriptors.length);
    this.cellIndex.clear();
    this._cellCount = descriptors.length;

    for (let i = 0; i < descriptors.length; i++) {
      this.cellIndex.set(descriptors[i].cellId, i);
      this._writeCellInstance(i, descriptors[i]);
    }

    this._uploadBuffer();
  }

  /**
   * Construct cell descriptors from VisualProfile data (from CellVisualIdentity)
   * and load them.  Convenience bridge between the physics-derived visual system
   * and this renderer.
   */
  loadFromVisualProfiles(
    profiles: Map<string, VisualProfile>,
    bboxes: Map<string, CellBBox>,
  ): void {
    const descriptors: CellInstanceDescriptor[] = [];

    for (const [cellId, profile] of profiles) {
      const bbox = bboxes.get(cellId);
      if (!bbox) continue;

      descriptors.push({
        cellId,
        bbox,
        color: [...profile.colorPalette.base, 1.0],
        opacity: 1.0,
        sdfShape: profile.sdfShape,
        sdfParams: {}, // use defaults from packSdfParams
        materialType: profile.materialType,
        patternShader: profile.patternShader,
        rdOffset: [0, 0],
        rdScale: [1, 1],
      });
    }

    this.loadCells(descriptors);
  }

  /**
   * Update a single cell's instance data in-place (no full rebuild).
   * Useful for per-frame animation: position lerp, opacity fade, colour shift.
   */
  updateCell(
    cellId: string,
    patch: Partial<Pick<CellInstanceDescriptor,
      'bbox' | 'color' | 'opacity' | 'sdfParams' | 'rdOffset' | 'rdScale'
    >>,
  ): boolean {
    const idx = this.cellIndex.get(cellId);
    if (idx === undefined) return false;

    const base = idx * FLOATS_PER_CELL;
    const buf  = this.instanceBuffer;

    if (patch.bbox) {
      bboxToModelMatrix(patch.bbox, buf, base);
    }

    if (patch.color) {
      buf[base + 16] = patch.color[0];
      buf[base + 17] = patch.color[1];
      buf[base + 18] = patch.color[2];
      buf[base + 19] = patch.color[3];
    }

    if (patch.opacity !== undefined) {
      buf[base + 20] = patch.opacity;
    }

    if (patch.sdfParams) {
      // Read current shape from buffer
      const shapeIdx = Math.round(buf[base + 21]);
      const shapeNames: SdfShape[] = [
        'flower', 'koch', 'julia', 'supershape', 'capsule',
        'hexagon', 'star', 'roundbox', 'polygon',
      ];
      const shape = shapeNames[shapeIdx] ?? 'roundbox';
      const packed = packSdfParams(shape, patch.sdfParams);
      buf[base + 22] = packed[0];
      buf[base + 23] = packed[1];
      buf[base + 24] = packed[2];
      buf[base + 25] = packed[3];
    }

    if (patch.rdOffset) {
      buf[base + 28] = patch.rdOffset[0];
      buf[base + 29] = patch.rdOffset[1];
    }

    if (patch.rdScale) {
      buf[base + 30] = patch.rdScale[0];
      buf[base + 31] = patch.rdScale[1];
    }

    // Partial upload — just this cell's range
    const { gl } = this;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboInstances);
    gl.bufferSubData(
      gl.ARRAY_BUFFER,
      base * 4, // byte offset
      buf,
      base,
      FLOATS_PER_CELL,
    );
    return true;
  }

  /**
   * Batch-update all cells' positions from a physics snapshot.
   * Writes only the model-matrix portion (first 16 floats per cell) for speed.
   */
  updatePositions(bboxes: Map<string, CellBBox>): void {
    for (const [cellId, bbox] of bboxes) {
      const idx = this.cellIndex.get(cellId);
      if (idx === undefined) continue;
      bboxToModelMatrix(bbox, this.instanceBuffer, idx * FLOATS_PER_CELL);
    }
    this._uploadBuffer();
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Public API — Texture binding
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Bind an external reaction-diffusion texture for pattern sampling.
   * Expected format: RGBA or RG float texture (R=u, G=v from Gray-Scott).
   */
  setReactionDiffusionTexture(texture: WebGLTexture): void {
    this.rdTexture = texture;
  }

  /**
   * Bind an external matcap environment-map texture.
   * Expected format: RGB or RGBA, square, typically 256×256 or 512×512.
   */
  setMatcapTexture(texture: WebGLTexture): void {
    this.matcapTexture = texture;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Public API — Rendering
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Draw ALL cells in a single instanced draw call.
   *
   * @param time        Simulation / animation time (seconds).
   * @param view        Column-major view matrix (defaults to identity).
   * @param projection  Column-major projection matrix (defaults to identity).
   * @param lightDir    Normalised directional light vector (defaults to [0.3, 0.5, 0.8]).
   * @param fresnelPow  Global Fresnel power (defaults to 3.0).
   */
  draw(
    time       = 0,
    view       = InstancedCellRenderer.IDENTITY,
    projection = InstancedCellRenderer.IDENTITY,
    lightDir: [number, number, number] = [0.3, 0.5, 0.8],
    fresnelPow = 3.0,
  ): void {
    if (this._cellCount === 0) return;

    const { gl } = this;

    gl.useProgram(this.program);

    // ── Uniforms ──────────────────────────────────────────────────────────
    gl.uniformMatrix4fv(this.uView,       false, view);
    gl.uniformMatrix4fv(this.uProjection,  false, projection);
    gl.uniform1f(this.uTime, time);
    gl.uniform3f(this.uLightDir, lightDir[0], lightDir[1], lightDir[2]);
    gl.uniform1f(this.uFresnelPower, fresnelPow);
    gl.uniform1f(this.uBloomThreshold, 0.8);

    // ── Bind textures ────────────────────────────────────────────────────
    // Texture unit 0: reaction-diffusion
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.rdTexture);
    gl.uniform1i(this.uRdTexture, 0);

    // Texture unit 1: matcap
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.matcapTexture);
    gl.uniform1i(this.uMatcapTexture, 1);

    // ── Draw ─────────────────────────────────────────────────────────────
    gl.bindVertexArray(this.vao);
    gl.drawElementsInstanced(
      gl.TRIANGLES,
      this.indexCount,
      gl.UNSIGNED_SHORT,
      0,
      this._cellCount,
    );
    gl.bindVertexArray(null);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Public API — Introspection
  // ──────────────────────────────────────────────────────────────────────────

  /** Current number of cells in the batch. */
  get cellCount(): number { return this._cellCount; }

  /** Maximum capacity before resize. */
  get maxCells(): number { return this._maxCells; }

  /** Ordered list of cell IDs currently in the batch. */
  get cellIds(): string[] {
    const result: string[] = new Array(this._cellCount);
    for (const [id, idx] of this.cellIndex) {
      result[idx] = id;
    }
    return result;
  }

  /** Number of WebGL draw calls per frame (always 1 for this renderer). */
  get drawCallCount(): number { return this._cellCount > 0 ? 1 : 0; }

  /**
   * Hit-test: find which cell (if any) a screen-space point falls within.
   * Uses the CPU-side bbox stored in the instance buffer — not pixel-perfect
   * SDF but fast enough for interaction.
   *
   * @param wx World-space X coordinate.
   * @param wy World-space Y coordinate.
   * @returns  cellId of the hit cell, or null.
   */
  hitTest(wx: number, wy: number): string | null {
    // Walk cells back-to-front (last drawn = on top)
    for (let i = this._cellCount - 1; i >= 0; i--) {
      const base = i * FLOATS_PER_CELL;
      // Model matrix col3 = translation (cx, cy), col0[0] = w, col1[1] = h
      const cx = this.instanceBuffer[base + 12];
      const cy = this.instanceBuffer[base + 13];
      const w  = this.instanceBuffer[base +  0];
      const h  = this.instanceBuffer[base +  5];

      const halfW = w / 2;
      const halfH = h / 2;

      if (wx >= cx - halfW && wx <= cx + halfW &&
          wy >= cy - halfH && wy <= cy + halfH) {
        // Reverse-lookup cellId from index
        for (const [id, idx] of this.cellIndex) {
          if (idx === i) return id;
        }
      }
    }
    return null;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Public API — Lifecycle
  // ──────────────────────────────────────────────────────────────────────────

  dispose(): void {
    const { gl } = this;
    gl.deleteProgram(this.program);
    gl.deleteBuffer(this.vboGeom);
    gl.deleteBuffer(this.vboInstances);
    gl.deleteBuffer(this.ibo);
    gl.deleteVertexArray(this.vao);
    if (this.rdTexture)     gl.deleteTexture(this.rdTexture);
    if (this.matcapTexture) gl.deleteTexture(this.matcapTexture);
    this.cellIndex.clear();
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Private — Instance buffer packing
  // ──────────────────────────────────────────────────────────────────────────

  private _writeCellInstance(i: number, desc: CellInstanceDescriptor): void {
    const base = i * FLOATS_PER_CELL;
    const buf  = this.instanceBuffer;

    // [0–15] mat4 model matrix from bbox
    bboxToModelMatrix(desc.bbox, buf, base);

    // [16–19] RGBA colour
    buf[base + 16] = desc.color[0];
    buf[base + 17] = desc.color[1];
    buf[base + 18] = desc.color[2];
    buf[base + 19] = desc.color[3];

    // [20] opacity
    buf[base + 20] = desc.opacity;

    // [21] SDF shape index
    buf[base + 21] = SDF_SHAPE_INDEX[desc.sdfShape] ?? 7; // default roundbox

    // [22–25] SDF params
    const packed = packSdfParams(desc.sdfShape, desc.sdfParams);
    buf[base + 22] = packed[0];
    buf[base + 23] = packed[1];
    buf[base + 24] = packed[2];
    buf[base + 25] = packed[3];

    // [26] material type index
    buf[base + 26] = MATERIAL_TYPE_INDEX[desc.materialType] ?? 0;

    // [27] pattern shader index
    buf[base + 27] = PATTERN_SHADER_INDEX[desc.patternShader] ?? 0;

    // [28–29] RD texture UV offset
    buf[base + 28] = desc.rdOffset[0];
    buf[base + 29] = desc.rdOffset[1];

    // [30–31] RD texture UV scale
    buf[base + 30] = desc.rdScale[0];
    buf[base + 31] = desc.rdScale[1];
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Private — GPU buffer management
  // ──────────────────────────────────────────────────────────────────────────

  private _uploadBuffer(): void {
    const { gl } = this;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboInstances);
    gl.bufferSubData(
      gl.ARRAY_BUFFER,
      0,
      this.instanceBuffer,
      0,
      this._cellCount * FLOATS_PER_CELL,
    );
  }

  /**
   * Grow the instance buffer if the requested count exceeds current capacity.
   * Uses a 2× growth strategy to amortise reallocations.
   */
  private _ensureCapacity(count: number): void {
    if (count <= this._maxCells) return;

    const newMax = Math.max(count, this._maxCells * 2);
    const newBuf = new Float32Array(newMax * FLOATS_PER_CELL);

    // Copy existing data
    newBuf.set(this.instanceBuffer.subarray(0, this._cellCount * FLOATS_PER_CELL));
    this.instanceBuffer = newBuf;
    this._maxCells = newMax;

    // Reallocate GPU buffer
    const { gl } = this;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboInstances);
    gl.bufferData(gl.ARRAY_BUFFER, newBuf.byteLength, gl.DYNAMIC_DRAW);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Private — VAO attribute binding
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Bind per-instance attributes with vertexAttribDivisor = 1.
   * Must be called with vboInstances bound to ARRAY_BUFFER and the VAO active.
   */
  private _bindInstanceAttribs(): void {
    const { gl } = this;
    const byteStride = FLOATS_PER_CELL * 4; // 32 floats × 4 bytes = 128 bytes

    // ── mat4 a_modelMatrix → 4 consecutive vec4 slots ─────────────────────
    if (this.aModelMatrix >= 0) {
      for (let col = 0; col < 4; col++) {
        const loc = this.aModelMatrix + col;
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, byteStride, col * 16);
        gl.vertexAttribDivisor(loc, 1);
      }
    }

    // ── vec4 a_color ──────────────────────────────────────────────────────
    if (this.aColor >= 0) {
      gl.enableVertexAttribArray(this.aColor);
      gl.vertexAttribPointer(this.aColor, 4, gl.FLOAT, false, byteStride, 16 * 4);
      gl.vertexAttribDivisor(this.aColor, 1);
    }

    // ── float a_opacity ───────────────────────────────────────────────────
    if (this.aOpacity >= 0) {
      gl.enableVertexAttribArray(this.aOpacity);
      gl.vertexAttribPointer(this.aOpacity, 1, gl.FLOAT, false, byteStride, 20 * 4);
      gl.vertexAttribDivisor(this.aOpacity, 1);
    }

    // ── float a_sdfShape ──────────────────────────────────────────────────
    if (this.aSdfShape >= 0) {
      gl.enableVertexAttribArray(this.aSdfShape);
      gl.vertexAttribPointer(this.aSdfShape, 1, gl.FLOAT, false, byteStride, 21 * 4);
      gl.vertexAttribDivisor(this.aSdfShape, 1);
    }

    // ── vec4 a_sdfParams ──────────────────────────────────────────────────
    if (this.aSdfParams >= 0) {
      gl.enableVertexAttribArray(this.aSdfParams);
      gl.vertexAttribPointer(this.aSdfParams, 4, gl.FLOAT, false, byteStride, 22 * 4);
      gl.vertexAttribDivisor(this.aSdfParams, 1);
    }

    // ── float a_materialType ──────────────────────────────────────────────
    if (this.aMaterialType >= 0) {
      gl.enableVertexAttribArray(this.aMaterialType);
      gl.vertexAttribPointer(this.aMaterialType, 1, gl.FLOAT, false, byteStride, 26 * 4);
      gl.vertexAttribDivisor(this.aMaterialType, 1);
    }

    // ── float a_patternShader ─────────────────────────────────────────────
    if (this.aPatternShader >= 0) {
      gl.enableVertexAttribArray(this.aPatternShader);
      gl.vertexAttribPointer(this.aPatternShader, 1, gl.FLOAT, false, byteStride, 27 * 4);
      gl.vertexAttribDivisor(this.aPatternShader, 1);
    }

    // ── vec2 a_rdOffset ───────────────────────────────────────────────────
    if (this.aRdOffset >= 0) {
      gl.enableVertexAttribArray(this.aRdOffset);
      gl.vertexAttribPointer(this.aRdOffset, 2, gl.FLOAT, false, byteStride, 28 * 4);
      gl.vertexAttribDivisor(this.aRdOffset, 1);
    }

    // ── vec2 a_rdScale ────────────────────────────────────────────────────
    if (this.aRdScale >= 0) {
      gl.enableVertexAttribArray(this.aRdScale);
      gl.vertexAttribPointer(this.aRdScale, 2, gl.FLOAT, false, byteStride, 30 * 4);
      gl.vertexAttribDivisor(this.aRdScale, 1);
    }
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Private — Placeholder textures
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Create 1×1 placeholder textures so the shader doesn't error
   * if no external RD / matcap texture has been bound yet.
   */
  private _createPlaceholderTextures(): void {
    const { gl } = this;

    // RD placeholder: neutral (u=0.5, v=0.0)
    this.rdTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.rdTexture);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA8,
      1, 1, 0,
      gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([128, 0, 0, 255]),
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Matcap placeholder: neutral grey sphere
    this.matcapTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.matcapTexture);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA8,
      1, 1, 0,
      gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([180, 180, 180, 255]),
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Private — Shader compilation
  // ──────────────────────────────────────────────────────────────────────────

  private _compileProgram(vertSrc: string, fragSrc: string): WebGLProgram {
    const { gl } = this;
    const vert = this._compileShader(gl.VERTEX_SHADER,   vertSrc);
    const frag = this._compileShader(gl.FRAGMENT_SHADER, fragSrc);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      gl.deleteShader(vert);
      gl.deleteShader(frag);
      throw new Error(`InstancedCellRenderer link error: ${log}`);
    }
    gl.deleteShader(vert);
    gl.deleteShader(frag);
    return prog;
  }

  private _compileShader(type: number, src: string): WebGLShader {
    const { gl } = this;
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(s);
      gl.deleteShader(s);
      throw new Error(`InstancedCellRenderer shader error: ${log}`);
    }
    return s;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create an InstancedCellRenderer from a canvas element.
 *
 * @param canvas  Target canvas element.
 * @param maxCells  Initial instance buffer capacity (grows dynamically).
 * @returns  InstancedCellRenderer instance, or null if WebGL2 is unavailable.
 *
 * @example
 * ```ts
 * import { createInstancedCellRenderer } from '$lib/sph/instanced-cell-renderer';
 *
 * const renderer = createInstancedCellRenderer(canvas);
 * if (!renderer) throw new Error('WebGL2 required');
 *
 * renderer.loadCells(descriptors);
 *
 * function frame(t: number) {
 *   renderer.draw(t / 1000);
 *   requestAnimationFrame(frame);
 * }
 * requestAnimationFrame(frame);
 * ```
 */
export function createInstancedCellRenderer(
  canvas: HTMLCanvasElement,
  maxCells: number = DEFAULT_MAX_CELLS,
): InstancedCellRenderer | null {
  const gl = canvas.getContext('webgl2', {
    alpha: true,
    premultipliedAlpha: false,
    antialias: true,
    powerPreference: 'high-performance',
  });
  if (!gl) return null;
  return new InstancedCellRenderer(gl, maxCells);
}

/**
 * Bridge helper: convert a CellVisualIdentity-derived VisualProfile + physics
 * bounding boxes into CellInstanceDescriptors suitable for `loadCells()`.
 *
 * @param profiles  Map of cellId → VisualProfile from CellVisualIdentity.fromRegistry().
 * @param bboxes    Map of cellId → CellBBox (from cell_registry.json / physics layout).
 * @param rdGrid    Optional per-cell RD sub-region assignment (cellId → { offset, scale }).
 * @returns         Array of CellInstanceDescriptor.
 */
export function visualProfilesToDescriptors(
  profiles: Map<string, VisualProfile>,
  bboxes: Map<string, CellBBox>,
  rdGrid?: Map<string, { offset: [number, number]; scale: [number, number] }>,
): CellInstanceDescriptor[] {
  const descriptors: CellInstanceDescriptor[] = [];

  for (const [cellId, profile] of profiles) {
    const bbox = bboxes.get(cellId);
    if (!bbox) continue;

    const rd = rdGrid?.get(cellId);

    descriptors.push({
      cellId,
      bbox,
      color: [...profile.colorPalette.base, 1.0],
      opacity: 1.0,
      sdfShape: profile.sdfShape,
      sdfParams: {},
      materialType: profile.materialType,
      patternShader: profile.patternShader,
      rdOffset: rd?.offset ?? [0, 0],
      rdScale:  rd?.scale  ?? [1, 1],
    });
  }

  return descriptors;
}

/**
 * Compute per-cell RD sub-region assignments given a total cell count.
 * Tiles cells in a grid layout across the RD texture so each cell gets a
 * unique sub-region.
 *
 * @param cellIds  Array of cell identifiers.
 * @returns        Map of cellId → { offset, scale } for RD UV mapping.
 */
export function computeRDGridAssignment(
  cellIds: string[],
): Map<string, { offset: [number, number]; scale: [number, number] }> {
  const n = cellIds.length;
  if (n === 0) return new Map();

  // Compute grid dimensions (square-ish)
  const cols = Math.ceil(Math.sqrt(n));
  const rows = Math.ceil(n / cols);

  const tileW = 1.0 / cols;
  const tileH = 1.0 / rows;

  const result = new Map<string, { offset: [number, number]; scale: [number, number] }>();

  for (let i = 0; i < n; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    result.set(cellIds[i], {
      offset: [col * tileW, row * tileH],
      scale:  [tileW, tileH],
    });
  }

  return result;
}
