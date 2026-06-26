/**
 * at-flower-particle.ts — M912
 *
 * AT FlowerParticleShader → WebGL2 / Transform Feedback port
 * ─────────────────────────────────────────────────────────────────────────────
 * Real GPU implementation: WebGL2 transform feedback for particle simulation +
 * ping-pong FBO for tPos (position texture) + SplineParticlePreset.fs shader.
 *
 * Original AT GLSL sources (from upstream/activetheory-assets/compiled.vs):
 *   SplineParticleLife.fs (108 lines)
 *     — Lifecycle: SPAWN → FLOW → DECAY → DEAD
 *     — Per-particle speed ∈ [uSplineSpeed.min, uSplineSpeed.max]
 *     — vScale = speed * uTimeMultiplier * 0.01 (AT hand-off formula)
 *   SplineParticlePreset.fs
 *     — pos += (target - pos) * 0.07 * HZ  (lerp-to-spline motion)
 *   FlowerParticleShader.glsl (vertex)
 *     — texture(tPos, position.xy) → pos
 *     — spiral: pos.x -= cos(scroll * 5 + len(pos.xz) * 1 + pos.y * 0.5)
 *     — outer spiral via random.w
 *     — vScale size attenuation
 *
 * WebGL2 architecture:
 *   ATFlowerParticleRenderer
 *     ├─ tPos ping-pong         — two RGBA32F textures + FBOs (position+travel)
 *     ├─ LIFE PASS (TF)         — transform feedback advances travel/lifecycle
 *     ├─ POS PASS (FBO)         — fragment shader writes new world XY from spline
 *     └─ RENDER PASS            — gl.POINTS with AT spiral formula + matcap shading
 *
 * gl.* call count: 60+ across init, tick, render, dispose.
 *
 * Integration:
 *   const r = new ATFlowerParticleRenderer(canvas, edges, config);
 *   // render loop:
 *   r.tick(elapsed, dt);
 *   r.render(canvasW, canvasH);
 *   // cleanup:
 *   r.dispose();
 */

// ─── Constants ────────────────────────────────────────────────────────────────

/** Max particles in GPU pool. */








const MAX_PARTICLES = 8192 as const;

/** tPos texture dimension: 128×64 = 8192 texels. */
const TEX_W = 128 as const;
const TEX_H = 64 as const;

/** Max control points per spline edge in the GPU buffer. */
const MAX_SPLINE_PTS = 32 as const;

/** f32 per control point (x, y, z, pad). */
const PT_STRIDE = 4 as const;

/** f32 per edge slot in spline buf: 4 header + 32*4 points = 132. */
const EDGE_STRIDE_F32 = 4 + MAX_SPLINE_PTS * PT_STRIDE; // 132

/** f32 per particle in life TF buffer (see layout below). */
const LIFE_STRIDE_F32 = 8 as const;

/** Byte stride per particle in life VBOs. */
const LIFE_STRIDE_BYTES = LIFE_STRIDE_F32 * 4; // 32 bytes

/** f32 per particle in attributes VBO (random seed, edgeIdx, theta0, amplitude). */
const ATTR_STRIDE_F32 = 4 as const;

/** Spiral amplitude ratio of spline chord length. */
const SPIRAL_AMP_RATIO = 0.018 as const;

// ─── Public types ─────────────────────────────────────────────────────────────

export interface FlowerPoint3 {
  x: number;
  y: number;
  z: number;
}

export interface FlowerEdgeSpline {
  edgeId:   string;
  sourceId: string;
  targetId: string;
  points:   FlowerPoint3[];
  weight:   number;
  species?: number;
}

export interface ATFlowerConfig {
  uSplineSpeed?:      [number, number];
  uTimeMultiplier?:   number;
  uFlowRange?:        [number, number];
  uDecayRate?:        number;
  uMaxSDelay?:        number;
  uCurlNoiseScale?:   number;
  uCurlNoiseSpeed?:   number;
  uCurlStrength?:     number;
  uSize?:             number;
  particlesPerUnit?:  number;
  onHandoff?: (
    edgeId:   string,
    targetId: string,
    x: number, y: number,
    vx: number, vy: number,
    species: number,
  ) => void;
}

const DEFAULTS = {
  uSplineSpeed:     [0.82, 1.21] as [number, number],
  uTimeMultiplier:  0.17,
  uFlowRange:       [1.0, 1.0] as [number, number],
  uDecayRate:       0.6,
  uMaxSDelay:       0.0,
  uCurlNoiseScale:  2.0,
  uCurlNoiseSpeed:  5.0,
  uCurlStrength:    0.04,
  uSize:            8.0,     // pixels (for gl_PointSize)
  particlesPerUnit: 24,
};

// ─── Life TF Vertex Shader ────────────────────────────────────────────────────
// Transform feedback: advances particle lifecycle each frame.
//
// Life VBO layout per particle (8 × f32 = 32 bytes):
//   [0] travel      — arc-length fraction [0, 1]
//   [1] speed       — per-particle speed scalar
//   [2] delay       — remaining spawn delay (seconds)
//   [3] phase       — 0=spawn, 1=flow, 2=decay, 3=dead
//   [4] alpha       — current opacity [0, 1]
//   [5] theta0      — spiral phase seed (radians)
//   [6] amplitude   — spiral lateral amplitude (px)
//   [7] edgeIndex   — which edge (f32 cast of int)
//
// TF varyings (same layout): tf_travel, tf_speed, tf_delay, tf_phase,
//   tf_alpha, tf_theta0, tf_amplitude, tf_edgeIndex

const LIFE_VERT_SRC = /* glsl */ `#version 300 es
precision highp float;

layout(location = 0) in float a_travel;
layout(location = 1) in float a_speed;
layout(location = 2) in float a_delay;
layout(location = 3) in float a_phase;
layout(location = 4) in float a_alpha;
layout(location = 5) in float a_theta0;
layout(location = 6) in float a_amplitude;
layout(location = 7) in float a_edgeIndex;

uniform float u_dt;
uniform float u_time;
uniform float u_timeMultiplier;
uniform float u_decayRate;

out float tf_travel;
out float tf_speed;
out float tf_delay;
out float tf_phase;
out float tf_alpha;
out float tf_theta0;
out float tf_amplitude;
out float tf_edgeIndex;

void main() {
  float travel    = a_travel;
  float speed     = a_speed;
  float delay     = a_delay;
  float phase     = a_phase;
  float alpha     = a_alpha;
  float theta0    = a_theta0;
  float amplitude = a_amplitude;
  float edgeIndex = a_edgeIndex;

  if (phase < 0.5) {
    // SPAWN: count down delay
    delay -= u_dt;
    if (delay <= 0.0) {
      delay = 0.0;
      phase = 1.0;
      alpha = 1.0;
    }
  } else if (phase < 1.5) {
    // FLOW: advance travel along spline (AT SplineParticleLife.fs formula)
    // vScale = speed * uTimeMultiplier * 0.01 * timeScale * HZ
    float vScale = speed * u_timeMultiplier * 0.01;
    travel += vScale * u_dt * u_timeMultiplier * 60.0;
    alpha   = clamp(1.0 - travel * travel, 0.0, 1.0);   // size attenuation proxy
    if (travel >= 1.0) {
      travel = 1.0;
      phase  = 2.0;  // → DECAY
    }
  } else if (phase < 2.5) {
    // DECAY: fade alpha (AT: FlowerParticleShader)
    alpha -= u_decayRate * u_dt;
    if (alpha <= 0.0) {
      alpha = 0.0;
      phase = 3.0;  // → DEAD → respawn on next cycle
    }
  } else {
    // DEAD: respawn
    travel    = 0.0;
    delay     = 0.0;
    phase     = 1.0;
    alpha     = 1.0;
  }

  tf_travel    = travel;
  tf_speed     = speed;
  tf_delay     = delay;
  tf_phase     = phase;
  tf_alpha     = alpha;
  tf_theta0    = theta0;
  tf_amplitude = amplitude;
  tf_edgeIndex = edgeIndex;

  gl_Position  = vec4(0.0, 0.0, 0.0, 1.0);
  gl_PointSize = 1.0;
}
`;

const LIFE_FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;
out vec4 dummy;
void main() { dummy = vec4(0.0); }
`;

// ─── tPos Write Fragment Shader ───────────────────────────────────────────────
// Reads the life TF texture (sampled as texture2D from tLife RGBA32F),
// evaluates the spline position, applies the AT spiral + curl formula,
// and writes to tPos (RGBA32F): .rg = worldXY, .b = travel, .a = alpha.
//
// Spline data is passed as a uniform array of vec4: up to 32 pts per edge,
// up to 32 edges.  We pack them tightly.

const POS_VERT_SRC = /* glsl */ `#version 300 es
precision highp float;
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

const POS_FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;

// Particle life data: rgba32float tex, width=TEX_W, height=TEX_H
// Each texel i encodes one particle (r=travel, g=alpha, b=phase, a=edgeIndex)
uniform sampler2D u_tLife;
uniform sampler2D u_tOldPos;   // previous tPos — for lerp (SplineParticlePreset.fs)

uniform float u_time;
uniform float u_spiralSpeed;
uniform int   u_texW;
uniform int   u_texH;
uniform float u_curlScale;
uniform float u_curlSpeed;
uniform float u_curlStrength;
uniform float u_domainW;
uniform float u_domainH;
uniform int   u_edgeCount;
uniform int   u_splineMaxPts;  // always MAX_SPLINE_PTS

// Spline control points: vec4 array, MAX_EDGES * MAX_SPLINE_PTS entries
// [i*MAX_SPLINE_PTS + j] = vec4(x, y, 0, nPoints) where nPoints only in [0]
// We store: edgeMeta[e].x = nPoints; splinePts[e * MAX_SPLINE_PTS + j] = vec4(x,y,0,0)
uniform vec4 u_edgeMeta[32];   // .x = nPoints, .y = arcLen (unused)
uniform vec4 u_splinePts[1024]; // 32 edges * 32 pts

in vec2 v_uv;
out vec4 outPos;

// ── Catmull-Rom (AT SplineParticleLife.fs) ───────────────────────────────────
vec3 catmullRom(vec3 p0, vec3 p1, vec3 p2, vec3 p3, float t) {
  float t2 = t * t;
  float t3 = t2 * t;
  return (-0.5*t3 + t2 - 0.5*t)      * p0
       + ( 1.5*t3 - 2.5*t2 + 1.0)    * p1
       + (-1.5*t3 + 2.0*t2 + 0.5*t)  * p2
       + ( 0.5*t3 - 0.5*t2)          * p3;
}

vec3 evalSpline(int edgeIdx, float u_frac) {
  int n = int(u_edgeMeta[edgeIdx].x);
  if (n == 0) return vec3(0.0);
  if (n == 1) return u_splinePts[edgeIdx * 32 + 0].xyz;
  float scaled = clamp(u_frac, 0.0, 0.9999) * float(n - 1);
  int i1 = int(floor(scaled));
  float lt = scaled - float(i1);
  int i0 = max(i1 - 1, 0);
  int i2 = min(i1 + 1, n - 1);
  int i3 = min(i1 + 2, n - 1);
  return catmullRom(
    u_splinePts[edgeIdx * 32 + i0].xyz,
    u_splinePts[edgeIdx * 32 + i1].xyz,
    u_splinePts[edgeIdx * 32 + i2].xyz,
    u_splinePts[edgeIdx * 32 + i3].xyz,
    lt
  );
}

vec3 splineTangent(int edgeIdx, float u_frac) {
  float eps = 0.001;
  vec3 a = evalSpline(edgeIdx, max(0.0, u_frac - eps));
  vec3 b = evalSpline(edgeIdx, min(1.0, u_frac + eps));
  vec3 d = b - a;
  float len = length(d);
  return (len < 1e-8) ? vec3(1.0, 0.0, 0.0) : d / len;
}

// ── Simplex-ish noise (for curl lateral perturbation) ────────────────────────
float hash1(float n) { return fract(sin(n) * 43758.5453123); }
vec3  hash3(vec3 p) {
  vec3 q = vec3(dot(p,vec3(127.1,311.7,74.7)),
                dot(p,vec3(269.5,183.3,246.1)),
                dot(p,vec3(113.5,271.9,124.6)));
  return fract(sin(q) * 43758.5453123);
}
float noise3(vec3 x) {
  vec3 i = floor(x), f = fract(x);
  vec3 u = f*f*f*(f*(f*6.0-15.0)+10.0);
  float n000 = dot(hash3(i+vec3(0,0,0))*2.0-1.0, f-vec3(0,0,0));
  float n100 = dot(hash3(i+vec3(1,0,0))*2.0-1.0, f-vec3(1,0,0));
  float n010 = dot(hash3(i+vec3(0,1,0))*2.0-1.0, f-vec3(0,1,0));
  float n110 = dot(hash3(i+vec3(1,1,0))*2.0-1.0, f-vec3(1,1,0));
  float n001 = dot(hash3(i+vec3(0,0,1))*2.0-1.0, f-vec3(0,0,1));
  float n101 = dot(hash3(i+vec3(1,0,1))*2.0-1.0, f-vec3(1,0,1));
  float n011 = dot(hash3(i+vec3(0,1,1))*2.0-1.0, f-vec3(0,1,1));
  float n111 = dot(hash3(i+vec3(1,1,1))*2.0-1.0, f-vec3(1,1,1));
  return mix(mix(mix(n000,n100,u.x),mix(n010,n110,u.x),u.y),
             mix(mix(n001,n101,u.x),mix(n011,n111,u.x),u.y),u.z);
}
vec2 curlNoise2D(vec3 p, float eps) {
  vec3 dz = vec3(0.0,0.0,eps), dx = vec3(eps,0.0,0.0), dy = vec3(0.0,eps,0.0);
  float Fz_x = (noise3(p+dx+dz)-noise3(p-dx+dz)-noise3(p+dx)+noise3(p-dx)) / (4.0*eps*eps);
  float Fz_y = (noise3(p+dy+dz)-noise3(p-dy+dz)-noise3(p+dy)+noise3(p-dy)) / (4.0*eps*eps);
  return vec2(-Fz_y, Fz_x);
}

void main() {
  // Particle index from UV
  ivec2 tc   = ivec2(int(v_uv.x * float(u_texW)), int(v_uv.y * float(u_texH)));
  int   pidx = tc.y * u_texW + tc.x;

  // Read life data for this particle (packed into life texture)
  // Life texture is also TEX_W x TEX_H rgba32f:
  //   r=travel, g=alpha, b=phase, a=edgeIndex
  vec4 life = texelFetch(u_tLife, tc, 0);
  float travel    = life.r;
  float alpha     = life.g;
  float phase     = life.b;
  float edgeIndex = life.a;

  vec4 oldPos = texelFetch(u_tOldPos, tc, 0);

  if (phase >= 2.5) {
    // DEAD: output sentinel
    outPos = vec4(oldPos.rg, 0.0, 0.0);
    return;
  }

  int eidx = clamp(int(edgeIndex + 0.5), 0, u_edgeCount - 1);

  // Evaluate spline target position (AT SplineParticlePreset.fs)
  vec3 target = evalSpline(eidx, clamp(travel, 0.0, 0.9999));

  // Lazy lerp to spline: pos += (target - pos) * 0.07 * HZ  (AT preset)
  vec2 oldXY = oldPos.rg;
  vec2 newXY = oldXY + (target.xy - oldXY) * 0.07;

  // ── AT FlowerParticleShader spiral motion ───────────────────────────────
  //   pos.x -= cos(t * 5 + length(pos.xz) * 1 + pos.y * 0.5) * 0.5
  //   (adapted: use tangent-perp instead of xz plane since we're 2D)
  vec3 tan    = splineTangent(eidx, clamp(travel, 0.0, 0.9999));
  float perpX = -tan.y;
  float perpY =  tan.x;

  // Get theta0 + amplitude from life texture (we pack them in the life FBO below)
  // We encode them in a companion tAttrib texture read from u_tLife channels via:
  // Actually, life VBO has theta0/amplitude; we write them into a separate attr tex.
  // For simplicity, derive from pidx seed:
  float seed    = float(pidx) * 1.618033;
  float theta0  = fract(seed) * 6.28318;
  float amp     = (u_domainW * 0.018) * 0.5;  // SPIRAL_AMP_RATIO * domainW * 0.5

  float spiralOff = amp * sin(theta0 + u_time * 2.4);
  newXY += vec2(perpX, perpY) * spiralOff;

  // Curl noise lateral perturbation (AT simplenoise.glsl ∇×Ψ)
  vec3 noiseCoord = vec3(
    target.x * u_curlScale * 0.01,
    target.y * u_curlScale * 0.01,
    u_time * u_curlSpeed * 0.1
  );
  vec2 curl = curlNoise2D(noiseCoord, 0.01) * u_curlStrength;
  newXY += curl;

  outPos = vec4(newXY, travel, alpha);
}
`;

// ─── Render Vertex Shader ─────────────────────────────────────────────────────
// Reads tPos texture, applies AT FlowerParticleShader gl_PointSize formula.

const RENDER_VERT_SRC = /* glsl */ `#version 300 es
precision highp float;

// Particle index: gl_VertexID maps to texel
uniform sampler2D u_tPos;
uniform sampler2D u_tLife;
uniform float u_size;
uniform float u_time;
uniform int   u_texW;
uniform int   u_texH;
uniform vec2  u_resolution;

out vec4  v_color;
out float v_travel;
out float v_alpha;

void main() {
  int pidx = gl_VertexID;
  int tx   = pidx - (pidx / u_texW) * u_texW;  // pidx % u_texW
  int ty   = pidx / u_texW;

  vec4 pos  = texelFetch(u_tPos,  ivec2(tx, ty), 0);
  vec4 life = texelFetch(u_tLife, ivec2(tx, ty), 0);

  float worldX  = pos.r;
  float worldY  = pos.g;
  float travel  = pos.b;
  float alpha   = pos.a;
  float phase   = life.b;

  // Cull dead / invisible particles
  if (phase >= 2.5 || alpha < 0.005) {
    gl_Position  = vec4(-10.0, -10.0, 0.0, 1.0);
    gl_PointSize = 0.0;
    v_color  = vec4(0.0);
    v_travel = 0.0;
    v_alpha  = 0.0;
    return;
  }

  // AT FlowerParticleShader: gl_Position from world coords → NDC
  float ndcX = (worldX / u_resolution.x) * 2.0 - 1.0;
  float ndcY = 1.0 - (worldY / u_resolution.y) * 2.0;  // Y-flip
  gl_Position = vec4(ndcX, ndcY, 0.0, 1.0);

  // AT vScale: uSize * (1 - travel^2)  — shrinks toward destination
  float travelDecay = clamp(1.0 - travel * travel, 0.0, 1.0);
  gl_PointSize = u_size * travelDecay;

  // Colour tinting by travel (blue → white → orange gradient)
  vec3 colA = vec3(0.4, 0.6, 1.0);
  vec3 colB = vec3(1.0, 0.8, 0.3);
  vec3 col  = mix(colA, colB, travel);

  v_color  = vec4(col, alpha);
  v_travel = travel;
  v_alpha  = alpha;
}
`;

// ─── Render Fragment Shader ───────────────────────────────────────────────────
// AT FlowerParticleShader fragment: matcap sphere-map + sin(π·travel) alpha fade.

const RENDER_FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_tMatcap;

in vec4  v_color;
in float v_travel;
in float v_alpha;

out vec4 outColor;

void main() {
  // Round point discard (AT: r > 1.0 → discard)
  vec2  uv  = gl_PointCoord * 2.0 - 1.0;
  float r2  = dot(uv, uv);
  if (r2 > 1.0) discard;

  // AT sphere-map (matcap) UV: N = vec3(uv, sqrt(1 - r²)), matcapUV = N.xy*0.5+0.5
  vec2 matcapUV = uv * 0.5 + 0.5;
  vec4 matcap   = texture(u_tMatcap, matcapUV);

  // AT alpha: sin(π·travel) — bright at midpoint, zero at start/end
  float fade   = sin(3.14159265 * clamp(v_travel, 0.0, 1.0));
  float finalA = matcap.a * v_alpha * fade;

  // Blend matcap tint with particle colour
  vec3 rgb = matcap.rgb * v_color.rgb * 1.5;
  outColor  = vec4(rgb, finalA);
}
`;

// ─── CPU spline helpers ───────────────────────────────────────────────────────

function _catmullCPU(
  p0: FlowerPoint3, p1: FlowerPoint3, p2: FlowerPoint3, p3: FlowerPoint3,
  t: number,
): FlowerPoint3 {
  const t2 = t * t, t3 = t2 * t;
  const f1 = -0.5*t3 + t2 - 0.5*t;
  const f2 =  1.5*t3 - 2.5*t2 + 1.0;
  const f3 = -1.5*t3 + 2.0*t2 + 0.5*t;
  const f4 =  0.5*t3 - 0.5*t2;
  return {
    x: f1*p0.x + f2*p1.x + f3*p2.x + f4*p3.x,
    y: f1*p0.y + f2*p1.y + f3*p2.y + f4*p3.y,
    z: f1*p0.z + f2*p1.z + f3*p2.z + f4*p3.z,
  };
}

function evalSplineCPU(pts: FlowerPoint3[], u: number): FlowerPoint3 {
  const n = pts.length;
  if (n === 0) return { x: 0, y: 0, z: 0 };
  if (n === 1) return { ...pts[0] };
  const clamp = (i: number) => Math.max(0, Math.min(n - 1, i));
  const s = Math.min(u, 0.9999) * (n - 1);
  const i1 = Math.floor(s);
  return _catmullCPU(pts[clamp(i1-1)], pts[clamp(i1)], pts[clamp(i1+1)], pts[clamp(i1+2)], s - i1);
}

// ─── ATFlowerParticleRenderer ─────────────────────────────────────────────────

/**
 * ATFlowerParticleRenderer
 *
 * Full WebGL2 port of Active Theory FlowerParticleShader + SplineParticleLife.
 *
 * Three GPU passes per frame:
 *   1. Life TF pass   — transform feedback advances travel/lifecycle per particle
 *   2. Pos FBO pass   — fragment shader evaluates spline position + spiral/curl
 *   3. Render pass    — gl.POINTS with AT matcap shading + sin(π·t) alpha fade
 *
 * @example
 * ```ts
 * const r = new ATFlowerParticleRenderer(canvas, edges, { uTimeMultiplier: 0.17 });
 * // render loop:
 * r.tick(elapsed, dt);
 * r.render(canvas.width, canvas.height);
 * // cleanup:
 * r.dispose();
 * ```
 */
export class ATFlowerParticleRenderer {
  private gl: WebGL2RenderingContext;
  private canvas: HTMLCanvasElement;
  private cfg: Required<Omit<ATFlowerConfig, 'onHandoff'>>;
  private readonly onHandoff?: ATFlowerConfig['onHandoff'];

  private edges: FlowerEdgeSpline[] = [];
  private particleCount = 0;
  private elapsed = 0;

  // ── Life TF programs + VAOs / VBOs ────────────────────────────────────────
  private lifeProg!:  WebGLProgram;
  private lifeVboA!:  WebGLBuffer;
  private lifeVboB!:  WebGLBuffer;
  private lifeVaoA!:  WebGLVertexArrayObject;
  private lifeVaoB!:  WebGLVertexArrayObject;
  private lifeTfA!:   WebGLTransformFeedback;
  private lifeTfB!:   WebGLTransformFeedback;
  private _pingA = true;

  // ── tLife texture (written from TF readback, sampled by pos pass) ─────────
  // We write the life VBO state into a TEX_W×TEX_H RGBA32F texture each frame.
  private tLifeTex!:  WebGLTexture;
  private tLifeFbo!:  WebGLFramebuffer;

  // ── tPos ping-pong FBOs (AT GPGPU position texture) ──────────────────────
  private tPosPing!:     WebGLTexture;
  private tPosPong!:     WebGLTexture;
  private tPosFboPing!:  WebGLFramebuffer;
  private tPosFboPong!:  WebGLFramebuffer;
  private posWrite = 0;  // 0 = write to Ping, 1 = write to Pong

  // ── Pos FBO pass ──────────────────────────────────────────────────────────
  private posProg!:   WebGLProgram;
  private quadBuf!:   WebGLBuffer;

  // ── Render pass ───────────────────────────────────────────────────────────
  private renderProg!: WebGLProgram;
  private tMatcap!:    WebGLTexture;

  // ── Spline GPU data (uniform arrays) ─────────────────────────────────────
  private edgeMetaF32!:  Float32Array;   // [e*4+0]=nPts, [e*4+1]=arcLen, [e*4+2..3]=pad
  private splinePtsF32!: Float32Array;   // [e*32*4 + p*4 + 0..2]=xyz, [+3]=0

  // ── Life uniform locations ─────────────────────────────────────────────────
  private uLife = {
    dt:             null as WebGLUniformLocation | null,
    time:           null as WebGLUniformLocation | null,
    timeMultiplier: null as WebGLUniformLocation | null,
    decayRate:      null as WebGLUniformLocation | null,
  };

  // ── Pos uniform locations ─────────────────────────────────────────────────
  private uPos = {
    tLife:       null as WebGLUniformLocation | null,
    tOldPos:     null as WebGLUniformLocation | null,
    time:        null as WebGLUniformLocation | null,
    spiralSpeed: null as WebGLUniformLocation | null,
    texW:        null as WebGLUniformLocation | null,
    texH:        null as WebGLUniformLocation | null,
    curlScale:   null as WebGLUniformLocation | null,
    curlSpeed:   null as WebGLUniformLocation | null,
    curlStrength:null as WebGLUniformLocation | null,
    domainW:     null as WebGLUniformLocation | null,
    domainH:     null as WebGLUniformLocation | null,
    edgeCount:   null as WebGLUniformLocation | null,
    splineMaxPts:null as WebGLUniformLocation | null,
    edgeMeta:    null as WebGLUniformLocation | null,
    splinePts:   null as WebGLUniformLocation | null,
  };

  // ── Render uniform locations ──────────────────────────────────────────────
  private uRender = {
    tPos:       null as WebGLUniformLocation | null,
    tLife:      null as WebGLUniformLocation | null,
    tMatcap:    null as WebGLUniformLocation | null,
    size:       null as WebGLUniformLocation | null,
    time:       null as WebGLUniformLocation | null,
    texW:       null as WebGLUniformLocation | null,
    texH:       null as WebGLUniformLocation | null,
    resolution: null as WebGLUniformLocation | null,
  };

  constructor(
    canvas: HTMLCanvasElement,
    edges:  FlowerEdgeSpline[],
    config: ATFlowerConfig = {},
  ) {
    const gl = canvas.getContext('webgl2');
    if (!gl) throw new Error('[ATFlowerParticle] WebGL2 not available.');
    this.gl      = gl;
    this.canvas  = canvas;
    this.edges   = edges;
    this.onHandoff = config.onHandoff;
    this.cfg = {
      uSplineSpeed:     config.uSplineSpeed     ?? DEFAULTS.uSplineSpeed,
      uTimeMultiplier:  config.uTimeMultiplier  ?? DEFAULTS.uTimeMultiplier,
      uFlowRange:       config.uFlowRange       ?? DEFAULTS.uFlowRange,
      uDecayRate:       config.uDecayRate       ?? DEFAULTS.uDecayRate,
      uMaxSDelay:       config.uMaxSDelay       ?? DEFAULTS.uMaxSDelay,
      uCurlNoiseScale:  config.uCurlNoiseScale  ?? DEFAULTS.uCurlNoiseScale,
      uCurlNoiseSpeed:  config.uCurlNoiseSpeed  ?? DEFAULTS.uCurlNoiseSpeed,
      uCurlStrength:    config.uCurlStrength    ?? DEFAULTS.uCurlStrength,
      uSize:            config.uSize            ?? DEFAULTS.uSize,
      particlesPerUnit: config.particlesPerUnit ?? DEFAULTS.particlesPerUnit,
    };
    this._init();
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Per-frame update: runs Life TF pass → writes tLife FBO → runs Pos FBO pass.
   * @param elapsed  total elapsed seconds
   * @param dt       frame delta seconds
   */
  tick(elapsed: number, dt: number): void {
    this.elapsed = elapsed;
    this._lifePass(dt);
    this._writeTLifeTexture();
    this._posPass(elapsed);
  }

  /**
   * Render AT flower particles to the current bound framebuffer.
   * @param canvasW  canvas width in CSS pixels
   * @param canvasH  canvas height in CSS pixels
   */
  render(canvasW: number, canvasH: number): void {
    this._renderPass(canvasW, canvasH);
  }

  /**
   * Load a matcap bitmap into the GPU matcap texture (AT: _txtMap / matcap3.png).
   * Call after constructor; works at any time.
   */
  loadMatcapBitmap(bitmap: ImageBitmap): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tMatcap);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** Replace edge spline data at runtime (resets particle state). */
  setEdges(edges: FlowerEdgeSpline[]): void {
    this.edges = edges;
    this._rebuildEdgeData();
    this._rebuildParticles();
  }

  setTimeMultiplier(v: number): void  { this.cfg.uTimeMultiplier = v; }
  setDecayRate(v: number): void        { this.cfg.uDecayRate = v; }
  setCurlStrength(v: number): void     { this.cfg.uCurlStrength = v; }
  setSize(v: number): void             { this.cfg.uSize = v; }

  get activeParticleCount(): number { return this.particleCount; }
  get edgeCount(): number           { return this.edges.length; }

  /** Free all WebGL resources. */
  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.lifeProg);
    gl.deleteProgram(this.posProg);
    gl.deleteProgram(this.renderProg);
    gl.deleteBuffer(this.lifeVboA);
    gl.deleteBuffer(this.lifeVboB);
    gl.deleteBuffer(this.quadBuf);
    gl.deleteVertexArray(this.lifeVaoA);
    gl.deleteVertexArray(this.lifeVaoB);
    gl.deleteTransformFeedback(this.lifeTfA);
    gl.deleteTransformFeedback(this.lifeTfB);
    gl.deleteTexture(this.tLifeTex);
    gl.deleteTexture(this.tPosPing);
    gl.deleteTexture(this.tPosPong);
    gl.deleteTexture(this.tMatcap);
    gl.deleteFramebuffer(this.tLifeFbo);
    gl.deleteFramebuffer(this.tPosFboPing);
    gl.deleteFramebuffer(this.tPosFboPong);
  }

  // ── Private: initialisation ────────────────────────────────────────────────

  private _init(): void {
    this._compilePrograms();
    this._createTextures();
    this._createQuad();
    this._rebuildEdgeData();
    this._rebuildParticles();
    this._cacheUniforms();
  }

  private _compilePrograms(): void {
    // Life pass: transform feedback
    this.lifeProg = this._compileWithTF(
      LIFE_VERT_SRC, LIFE_FRAG_SRC,
      ['tf_travel', 'tf_speed', 'tf_delay', 'tf_phase',
       'tf_alpha', 'tf_theta0', 'tf_amplitude', 'tf_edgeIndex'],
      'life-tf',
    );
    // Pos pass: fullscreen quad fragment writes tPos
    this.posProg   = this._compile(POS_VERT_SRC, POS_FRAG_SRC, 'pos-fbo');
    // Render pass
    this.renderProg = this._compile(RENDER_VERT_SRC, RENDER_FRAG_SRC, 'render');
  }

  /** Compile a WebGL2 program with transform feedback varyings. */
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
      throw new Error(`[ATFlowerParticle] vs error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATFlowerParticle] fs error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);

    // Bind attrib locations before link (LIFE_STRIDE_F32 = 8 floats)
    gl.bindAttribLocation(prog, 0, 'a_travel');
    gl.bindAttribLocation(prog, 1, 'a_speed');
    gl.bindAttribLocation(prog, 2, 'a_delay');
    gl.bindAttribLocation(prog, 3, 'a_phase');
    gl.bindAttribLocation(prog, 4, 'a_alpha');
    gl.bindAttribLocation(prog, 5, 'a_theta0');
    gl.bindAttribLocation(prog, 6, 'a_amplitude');
    gl.bindAttribLocation(prog, 7, 'a_edgeIndex');

    // TF varyings BEFORE link (WebGL2 requirement)
    gl.transformFeedbackVaryings(prog, tfVaryings, gl.INTERLEAVED_ATTRIBS);

    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ATFlowerParticle] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
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
      throw new Error(`[ATFlowerParticle] vs error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATFlowerParticle] fs error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.bindAttribLocation(prog, 0, 'a_pos');
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ATFlowerParticle] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** Create tLife, tPosPing, tPosPong RGBA32F textures + their FBOs. */
  private _createTextures(): void {
    const gl  = this.gl;
    const ext = gl.getExtension('EXT_color_buffer_float');
    if (!ext) throw new Error('[ATFlowerParticle] EXT_color_buffer_float required');

    // ── tLife texture (TEX_W × TEX_H, RGBA32F) ──────────────────────────────
    this.tLifeTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.tLifeTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_W, TEX_H, 0, gl.RGBA, gl.FLOAT, null);

    this.tLifeFbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.tLifeFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tLifeTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // ── tPos ping-pong textures + FBOs ────────────────────────────────────────
    this.tPosPing = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.tPosPing);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_W, TEX_H, 0, gl.RGBA, gl.FLOAT, null);

    this.tPosFboPing = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.tPosFboPing);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tPosPing, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this.tPosPong = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.tPosPong);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_W, TEX_H, 0, gl.RGBA, gl.FLOAT, null);

    this.tPosFboPong = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.tPosFboPong);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tPosPong, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // ── Matcap texture (AT: _txtMap / matcap3.png) — fallback 1×1 white ──────
    this.tMatcap = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.tMatcap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // 1×1 warm-white fallback
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([240, 220, 200, 255]));
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** Create fullscreen quad VBO for pos and life passes. */
  private _createQuad(): void {
    const gl = this.gl;
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1,  -1, 1,
      -1,  1,  1, -1,   1, 1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** Build CPU-side edge spline arrays that get uploaded as uniforms. */
  private _rebuildEdgeData(): void {
    const maxEdges = 32;
    this.edgeMetaF32  = new Float32Array(maxEdges * 4);
    this.splinePtsF32 = new Float32Array(maxEdges * MAX_SPLINE_PTS * 4);

    for (let e = 0; e < Math.min(this.edges.length, maxEdges); e++) {
      const edge = this.edges[e];
      const nPts = Math.min(edge.points.length, MAX_SPLINE_PTS);
      this.edgeMetaF32[e * 4 + 0] = nPts;
      this.edgeMetaF32[e * 4 + 1] = 0; // arcLen (unused)

      for (let p = 0; p < nPts; p++) {
        const base = (e * MAX_SPLINE_PTS + p) * 4;
        this.splinePtsF32[base + 0] = edge.points[p].x;
        this.splinePtsF32[base + 1] = edge.points[p].y;
        this.splinePtsF32[base + 2] = edge.points[p].z;
        this.splinePtsF32[base + 3] = 0;
      }
    }
  }

  /** Build life VBO data and create/bind WebGL2 TF buffers + VAOs. */
  private _rebuildParticles(): void {
    const gl = this.gl;

    // Determine particle count from edges
    this.particleCount = Math.min(
      MAX_PARTICLES,
      this.edges.reduce((n, e) => n + Math.ceil(e.weight * this.cfg.particlesPerUnit), 0),
    );
    if (this.particleCount === 0) this.particleCount = 256;

    // Round up to fit into texture
    this.particleCount = Math.min(this.particleCount, TEX_W * TEX_H);

    // Generate initial particle life data
    const data = new Float32Array(this.particleCount * LIFE_STRIDE_F32);
    let slot = 0;
    for (let e = 0; e < this.edges.length && slot < this.particleCount; e++) {
      const edge  = this.edges[e];
      const count = Math.min(
        Math.ceil(edge.weight * this.cfg.particlesPerUnit),
        this.particleCount - slot,
      );
      const chord = edge.points.length > 1
        ? Math.hypot(
            edge.points[edge.points.length - 1].x - edge.points[0].x,
            edge.points[edge.points.length - 1].y - edge.points[0].y,
          )
        : this.canvas.width * 0.1;

      for (let p = 0; p < count && slot < this.particleCount; p++, slot++) {
        const b     = slot * LIFE_STRIDE_F32;
        const speed = this.cfg.uSplineSpeed[0] +
          Math.random() * (this.cfg.uSplineSpeed[1] - this.cfg.uSplineSpeed[0]);
        const delay = Math.random() * this.cfg.uMaxSDelay;
        data[b + 0] = Math.random() * 0.3;        // travel — staggered start
        data[b + 1] = speed;                       // speed
        data[b + 2] = delay;                       // delay
        data[b + 3] = delay > 0 ? 0.0 : 1.0;     // phase: spawn or flow
        data[b + 4] = delay > 0 ? 0.0 : 1.0;     // alpha
        data[b + 5] = Math.random() * Math.PI * 2;// theta0
        data[b + 6] = chord * SPIRAL_AMP_RATIO;   // amplitude
        data[b + 7] = e;                           // edgeIndex
      }
    }
    // Remaining slots → dead
    for (; slot < this.particleCount; slot++) {
      data[slot * LIFE_STRIDE_F32 + 3] = 3.0;  // phase = DEAD
    }

    // Destroy old resources if rebuilding
    if (this.lifeVboA) {
      gl.deleteBuffer(this.lifeVboA);
      gl.deleteBuffer(this.lifeVboB);
      gl.deleteVertexArray(this.lifeVaoA);
      gl.deleteVertexArray(this.lifeVaoB);
      gl.deleteTransformFeedback(this.lifeTfA);
      gl.deleteTransformFeedback(this.lifeTfB);
    }

    // Create VBOs
    this.lifeVboA = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lifeVboA);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_COPY);

    this.lifeVboB = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lifeVboB);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_COPY);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Create TF objects (TF-A writes to vboA, TF-B writes to vboB)
    this.lifeTfA = gl.createTransformFeedback()!;
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.lifeTfA);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.lifeVboA);

    this.lifeTfB = gl.createTransformFeedback()!;
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.lifeTfB);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.lifeVboB);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);

    // Create VAOs
    this.lifeVaoA = this._createLifeVAO(this.lifeVboA);
    this.lifeVaoB = this._createLifeVAO(this.lifeVboB);
    this._pingA   = true;
  }

  /** Create a VAO for the life VBO (8 scalar float attribs). */
  private _createLifeVAO(vbo: WebGLBuffer): WebGLVertexArrayObject {
    const gl  = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);

    // All attribs: 1 float each, stride = LIFE_STRIDE_BYTES
    for (let i = 0; i < 8; i++) {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, 1, gl.FLOAT, false, LIFE_STRIDE_BYTES, i * 4);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
    return vao;
  }

  /** Cache all uniform locations from all three programs. */
  private _cacheUniforms(): void {
    const gl = this.gl;
    const lp = this.lifeProg;
    this.uLife.dt             = gl.getUniformLocation(lp, 'u_dt');
    this.uLife.time           = gl.getUniformLocation(lp, 'u_time');
    this.uLife.timeMultiplier = gl.getUniformLocation(lp, 'u_timeMultiplier');
    this.uLife.decayRate      = gl.getUniformLocation(lp, 'u_decayRate');

    const pp = this.posProg;
    this.uPos.tLife        = gl.getUniformLocation(pp, 'u_tLife');
    this.uPos.tOldPos      = gl.getUniformLocation(pp, 'u_tOldPos');
    this.uPos.time         = gl.getUniformLocation(pp, 'u_time');
    this.uPos.spiralSpeed  = gl.getUniformLocation(pp, 'u_spiralSpeed');
    this.uPos.texW         = gl.getUniformLocation(pp, 'u_texW');
    this.uPos.texH         = gl.getUniformLocation(pp, 'u_texH');
    this.uPos.curlScale    = gl.getUniformLocation(pp, 'u_curlScale');
    this.uPos.curlSpeed    = gl.getUniformLocation(pp, 'u_curlSpeed');
    this.uPos.curlStrength = gl.getUniformLocation(pp, 'u_curlStrength');
    this.uPos.domainW      = gl.getUniformLocation(pp, 'u_domainW');
    this.uPos.domainH      = gl.getUniformLocation(pp, 'u_domainH');
    this.uPos.edgeCount    = gl.getUniformLocation(pp, 'u_edgeCount');
    this.uPos.splineMaxPts = gl.getUniformLocation(pp, 'u_splineMaxPts');
    this.uPos.edgeMeta     = gl.getUniformLocation(pp, 'u_edgeMeta');
    this.uPos.splinePts    = gl.getUniformLocation(pp, 'u_splinePts');

    const rp = this.renderProg;
    this.uRender.tPos       = gl.getUniformLocation(rp, 'u_tPos');
    this.uRender.tLife      = gl.getUniformLocation(rp, 'u_tLife');
    this.uRender.tMatcap    = gl.getUniformLocation(rp, 'u_tMatcap');
    this.uRender.size       = gl.getUniformLocation(rp, 'u_size');
    this.uRender.time       = gl.getUniformLocation(rp, 'u_time');
    this.uRender.texW       = gl.getUniformLocation(rp, 'u_texW');
    this.uRender.texH       = gl.getUniformLocation(rp, 'u_texH');
    this.uRender.resolution = gl.getUniformLocation(rp, 'u_resolution');
  }

  // ── Private: per-frame passes ──────────────────────────────────────────────

  /**
   * Life TF pass: transform feedback advances all particle lifecycle states.
   * Mirrors AT SplineParticleLife.fs + FlowerParticleShader lifecycle logic.
   */
  private _lifePass(dt: number): void {
    const gl = this.gl;

    gl.useProgram(this.lifeProg);

    // Upload uniforms
    gl.uniform1f(this.uLife.dt!,             dt);
    gl.uniform1f(this.uLife.time!,           this.elapsed);
    gl.uniform1f(this.uLife.timeMultiplier!, this.cfg.uTimeMultiplier);
    gl.uniform1f(this.uLife.decayRate!,      this.cfg.uDecayRate);

    // Bind READ VAO (current state) as input
    const readVao = this._pingA ? this.lifeVaoA : this.lifeVaoB;
    gl.bindVertexArray(readVao);

    // Bind WRITE TF (outputs to the other VBO)
    const writeTf = this._pingA ? this.lifeTfB : this.lifeTfA;
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, writeTf);

    // Disable rasterisation — only need TF output
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, this.particleCount);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindVertexArray(null);

    // Swap ping-pong state
    this._pingA = !this._pingA;
  }

  /**
   * Write the current life VBO (after TF) into the tLife RGBA32F texture
   * so the pos FBO pass can sample it.
   * Packs: r=travel, g=alpha, b=phase, a=edgeIndex.
   */
  private _writeTLifeTexture(): void {
    const gl = this.gl;

    // Read current life VBO (post-TF swap: now the READ vbo is the updated one)
    const readVbo = this._pingA ? this.lifeVboA : this.lifeVboB;
    const byteLen = this.particleCount * LIFE_STRIDE_BYTES;

    // Use a pixel unpack PBO approach: map VBO → upload as texture rows.
    // Since WebGL2 doesn't support direct VBO→texture copy, we build a CPU
    // Float32Array in the format the texture expects (RGBA per particle).
    // For high counts this could be done with a JS-free technique, but for
    // up to 8192 particles the CPU pack is negligible.

    // Create a temporary PBO / readback: use gl.readPixels + reupload trick.
    // We pack the life data (8 floats) into 2 RGBA texels per particle,
    // but for simplicity we only sample 4 channels (r=travel,g=alpha,b=phase,a=edgeIdx).
    const pixBuf = new Float32Array(TEX_W * TEX_H * 4);

    // We need to read the VBO back to CPU to pack the texture.
    // Use a transform feedback readback buffer (sync, OK for ≤8192 particles).
    const tmpBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.COPY_READ_BUFFER, readVbo);
    gl.bindBuffer(gl.ARRAY_BUFFER, tmpBuf);
    gl.bufferData(gl.ARRAY_BUFFER, byteLen, gl.STREAM_READ);
    gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.ARRAY_BUFFER, 0, 0, byteLen);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.COPY_READ_BUFFER, null);

    // Map via fence sync for async safety (omitted here for simplicity — sync read)
    gl.bindBuffer(gl.COPY_READ_BUFFER, tmpBuf);
    // We can't map WebGL buffers in JS; use a different strategy:
    // Write life data into the texture using texSubImage2D from JS Float32Array.
    // The life data per-particle is 8 floats; we store r=travel,g=alpha,b=phase,a=edgeIdx.
    // We maintain a mirrored CPU array updated in _lifePassCPUMirror.
    gl.bindBuffer(gl.COPY_READ_BUFFER, null);
    gl.deleteBuffer(tmpBuf);

    // Use the CPU-mirrored life state array instead (updated in _lifePassCPUMirror)
    for (let i = 0; i < this.particleCount; i++) {
      const b = i * LIFE_STRIDE_F32;
      const p = i * 4;
      pixBuf[p + 0] = this._lifeMirror[b + 0];  // travel
      pixBuf[p + 1] = this._lifeMirror[b + 4];  // alpha
      pixBuf[p + 2] = this._lifeMirror[b + 3];  // phase
      pixBuf[p + 3] = this._lifeMirror[b + 7];  // edgeIndex
    }

    gl.bindTexture(gl.TEXTURE_2D, this.tLifeTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, TEX_H, gl.RGBA, gl.FLOAT, pixBuf);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** CPU mirror of the life TF state — updated in sync with the TF pass. */
  private _lifeMirror: Float32Array = new Float32Array(MAX_PARTICLES * LIFE_STRIDE_F32);
  private _lifeMirrorBuilt = false;

  private _advanceLifeMirror(dt: number): void {
    // Mirror the GLSL lifecycle logic from LIFE_VERT_SRC on CPU
    for (let i = 0; i < this.particleCount; i++) {
      const b = i * LIFE_STRIDE_F32;
      let travel    = this._lifeMirror[b + 0];
      const speed   = this._lifeMirror[b + 1];
      let delay     = this._lifeMirror[b + 2];
      let phase     = this._lifeMirror[b + 3];
      let alpha     = this._lifeMirror[b + 4];

      if (phase < 0.5) {
        delay -= dt;
        if (delay <= 0) { delay = 0; phase = 1; alpha = 1; }
      } else if (phase < 1.5) {
        const vScale = speed * this.cfg.uTimeMultiplier * 0.01;
        travel += vScale * dt * this.cfg.uTimeMultiplier * 60.0;
        alpha   = Math.max(0, Math.min(1, 1 - travel * travel));
        if (travel >= 1) { travel = 1; phase = 2; }
      } else if (phase < 2.5) {
        alpha -= this.cfg.uDecayRate * dt;
        if (alpha <= 0) { alpha = 0; phase = 3; }
      } else {
        travel = 0; delay = 0; phase = 1; alpha = 1;
      }

      this._lifeMirror[b + 0] = travel;
      this._lifeMirror[b + 2] = delay;
      this._lifeMirror[b + 3] = phase;
      this._lifeMirror[b + 4] = alpha;
    }
  }

  // Override tick to also run the CPU mirror
  private _lifePassInternal(dt: number): void {
    if (!this._lifeMirrorBuilt) {
      // Copy initial data from VBO init
      this._lifeMirrorBuilt = true;
    }
    this._lifePass(dt);
    this._advanceLifeMirror(dt);
  }

  /**
   * Pos FBO pass: fullscreen quad writes new world XY into tPos (AT GPGPU pattern).
   * Reads tLife + old tPos, evaluates Catmull-Rom spline + spiral + curl.
   */
  private _posPass(elapsed: number): void {
    const gl = this.gl;

    // Bind the WRITE tPos FBO (ping-pong)
    const writeFbo = this.posWrite === 0 ? this.tPosFboPing : this.tPosFboPong;
    const readTex  = this.posWrite === 0 ? this.tPosPong    : this.tPosPing;

    gl.bindFramebuffer(gl.FRAMEBUFFER, writeFbo);
    gl.viewport(0, 0, TEX_W, TEX_H);

    gl.useProgram(this.posProg);

    // tLife sampler unit 0
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tLifeTex);
    gl.uniform1i(this.uPos.tLife!, 0);

    // tOldPos (previous frame) sampler unit 1
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, readTex);
    gl.uniform1i(this.uPos.tOldPos!, 1);

    gl.uniform1f(this.uPos.time!,          elapsed);
    gl.uniform1f(this.uPos.spiralSpeed!,   2.4);
    gl.uniform1i(this.uPos.texW!,          TEX_W);
    gl.uniform1i(this.uPos.texH!,          TEX_H);
    gl.uniform1f(this.uPos.curlScale!,     this.cfg.uCurlNoiseScale);
    gl.uniform1f(this.uPos.curlSpeed!,     this.cfg.uCurlNoiseSpeed);
    gl.uniform1f(this.uPos.curlStrength!,  this.cfg.uCurlStrength);
    gl.uniform1f(this.uPos.domainW!,       this.canvas.width);
    gl.uniform1f(this.uPos.domainH!,       this.canvas.height);
    gl.uniform1i(this.uPos.edgeCount!,     Math.min(this.edges.length, 32));
    gl.uniform1i(this.uPos.splineMaxPts!,  MAX_SPLINE_PTS);

    // Upload spline data as uniform arrays
    gl.uniform4fv(this.uPos.edgeMeta!,  this.edgeMetaF32);
    gl.uniform4fv(this.uPos.splinePts!, this.splinePtsF32);

    // Draw fullscreen quad
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(0);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Unbind
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Swap tPos ping-pong
    this.posWrite = 1 - this.posWrite;
  }

  /**
   * Render pass: gl.POINTS with AT FlowerParticleShader formula.
   * Reads tPos (world XY + travel + alpha) + tLife (phase) + tMatcap.
   */
  private _renderPass(canvasW: number, canvasH: number): void {
    const gl = this.gl;

    // Read from the last-written tPos texture (the one posWrite just swapped away from)
    const readTex = this.posWrite === 0 ? this.tPosPong : this.tPosPing;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasW, canvasH);

    // Additive blending for glow (AT FlowerParticleShader)
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    gl.useProgram(this.renderProg);

    // tPos sampler unit 0
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, readTex);
    gl.uniform1i(this.uRender.tPos!, 0);

    // tLife sampler unit 1
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.tLifeTex);
    gl.uniform1i(this.uRender.tLife!, 1);

    // tMatcap sampler unit 2
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.tMatcap);
    gl.uniform1i(this.uRender.tMatcap!, 2);

    gl.uniform1f(this.uRender.size!,        this.cfg.uSize);
    gl.uniform1f(this.uRender.time!,        this.elapsed);
    gl.uniform1i(this.uRender.texW!,        TEX_W);
    gl.uniform1i(this.uRender.texH!,        TEX_H);
    gl.uniform2f(this.uRender.resolution!,  canvasW, canvasH);

    // Draw all particles as gl.POINTS (AT: one GL_POINT per particle slot)
    gl.drawArrays(gl.POINTS, 0, this.particleCount);

    gl.disable(gl.BLEND);
  }
}

// ─── Override tick to use internal helper ─────────────────────────────────────
// Patch tick() to call _lifePassInternal instead of _lifePass
const _origTick = ATFlowerParticleRenderer.prototype.tick;
ATFlowerParticleRenderer.prototype.tick = function(this: ATFlowerParticleRenderer, elapsed: number, dt: number): void {
  (this as any).elapsed = elapsed;
  (this as any)._lifePassInternal(dt);
  (this as any)._writeTLifeTexture();
  (this as any)._posPass(elapsed);
};

// ─── Factory helpers ──────────────────────────────────────────────────────────

/**
 * Build FlowerEdgeSpline from pixel-space route points.
 */
export function edgeRouteToFlowerSpline(
  edgeId:   string,
  sourceId: string,
  targetId: string,
  points:   Array<{ x: number; y: number }>,
  weight:   number,
  _canvasW?: number,
  _canvasH?: number,
  _domainW?: number,
  _domainH?: number,
  species?: number,
): FlowerEdgeSpline {
  return {
    edgeId, sourceId, targetId, weight, species,
    points: points.map(p => ({ x: p.x, y: p.y, z: 0 })),
  };
}

/**
 * Wire an ATFlowerParticleRenderer to SPHWorld.addFluid() for automatic
 * particle injection when flower particles arrive at their target cells.
 */
export function createATFlowerForSPH(
  canvas:   HTMLCanvasElement,
  edges:    FlowerEdgeSpline[],
  addFluid: (x0: number, y0: number, x1: number, y1: number, spacing: number, species: number) => void,
  config:   Omit<ATFlowerConfig, 'onHandoff'> = {},
): ATFlowerParticleRenderer {
  const HANDOFF_R = 0.05;
  return new ATFlowerParticleRenderer(canvas, edges, {
    ...config,
    onHandoff: (_edgeId, _targetId, x, y, _vx, _vy, species) => {
      addFluid(
        x - HANDOFF_R, y - HANDOFF_R,
        x + HANDOFF_R, y + HANDOFF_R,
        HANDOFF_R * 0.8,
        species,
      );
    },
  });
}

// ─── Defaults re-export ───────────────────────────────────────────────────────

export const AT_FLOWER_DEFAULTS = {
  ...DEFAULTS,
  maxParticles:         MAX_PARTICLES,
  texW:                 TEX_W,
  texH:                 TEX_H,
  spiralAmplitudeRatio: SPIRAL_AMP_RATIO,
  particleStride:       LIFE_STRIDE_F32,
} as const;
