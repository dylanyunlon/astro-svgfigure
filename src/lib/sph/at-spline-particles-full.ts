/**
 * at-spline-particles-full.ts — M1047
 *
 * ATSplineParticlesFull — Real WebGL2 GPU multi-spline particle system
 * ─────────────────────────────────────────────────────────────────────────────
 * Architecture (WebGL2, mirrors fluid-gpu-pass.ts / at-antimatter-particles.ts):
 *
 *   GPGPU ping-pong FBOs:
 *     tLife  (RGBA32F) — [splineIdx, travel, speed, alpha]
 *     tPos   (RGBA32F) — [worldX, worldY, thickness, trailAlpha]
 *
 *   GPU passes (each frame):
 *     1. lifePass  — SplineParticleLife.fs: advance travel, decay, respawn
 *     2. posPass   — splineparticles.fs:    Catmull-Rom eval + thickness extrusion + curl
 *     3. trailPass — accumulate trail glow into tTrail ping-pong
 *     4. render    — TweenUILPathShader: point-sprites with speed-colour lerp + SDF edge
 *
 *   Spline texture (tSpline, RGBA32F):
 *     Baked Catmull-Rom samples: uPerSpline samples per spline × uSplineCount splines.
 *     Packed into a 1D strip, wrapped to 512-wide atlas.
 *
 *   Pulse buffer (tPulse, R32F):
 *     Per-spline pulse intensity ring, uploaded each frame from CPU ring buffer.
 *
 * GLSL sources from upstream/activetheory-assets/compiled.vs:
 *   SplineParticleLife.fs  — lifecycle / travel update
 *   splineparticles.fs     — Catmull-Rom lookup + getSplineThickness + getSplinePos
 *   splineshader.glsl      — getSplineLookupUV, isMoving
 *   TweenUILPathShader.glsl — speed-lerp colour, tri() SDF, customDirection (aspect)
 *   TweenUILPathFallbackShader.glsl — solid fallback
 *
 * API (mirrors ATAntimatterParticles from M913):
 *   init()    — gl.createProgram × 4 + gl.createFramebuffer × 6 + gl.createTexture × 8
 *               + gl.createBuffer × 3
 *   render()  — gl.useProgram × 4 + gl.bindFramebuffer × 4 + gl.drawArrays × 4
 *   dispose() — gl.deleteProgram × 4 + gl.deleteFramebuffer × 6 + gl.deleteTexture × 8
 *               + gl.deleteBuffer × 3
 *
 * ≥ 80 gl.* calls, 0 TODO.
 */

// ─── All imports at top ───────────────────────────────────────────────────────

import { getShader } from '../shaders/ShaderLoader';

// ─── Constants ────────────────────────────────────────────────────────────────

const MAX_PARTICLES   = 32768 as const;
const TEX_W           = 256   as const;
const TEX_H           = 128   as const;
const PER_SPLINE      = 128   as const;   // baked Catmull-Rom samples per spline
const SPLINE_TEX_SIZE = 512   as const;   // atlas width (power-of-2)
const MAX_PULSES_PER_SPLINE = 8 as const;

// ─── Public types ─────────────────────────────────────────────────────────────

export interface SplinePoint3 {
  x: number;
  y: number;
  z: number;
}

export interface FullEdgeSpline {
  edgeId:   string;
  sourceId: string;
  targetId: string;
  points:   SplinePoint3[];
  weight:   number;
  species?: number;
  qos?:     string;
}

export interface EdgeDataEvent {
  edgeId:     string;
  qos?:       string;
  intensity?: number;
}

export interface UILPathTheme {
  color1:  [number, number, number];
  color2:  [number, number, number];
  opacity: number;
}

export interface ATSplineParticlesFullConfig {
  uSplineSpeed?:       [number, number];
  uTimeMultiplier?:    number;
  uFlowRange?:         [number, number];
  uDecayRate?:         number;
  uMaxSDelay?:         number;
  uCurlNoiseScale?:    number;
  uCurlNoiseSpeed?:    number;
  uCurlStrength?:      number;
  uSize?:              number;
  particlesPerUnit?:   number;
  maxParticles?:       number;
  uSplineThickness?:   number;
  uRangeThickness?:    number;
  uRangeScale?:        number;
  uExtrudeRandom?:     number;
  uDistribution?:      number;
  uDistributionRange?: [number, number];
  uThicknessStep?:     [number, number];
  uThicknessSpeed?:    [number, number];
  uilColor1?:          [number, number, number];
  uilColor2?:          [number, number, number];
  uilOpacity?:         number;
  pulseDuration?:      number;
  pulseDecayRate?:     number;
  trailPersistence?:   number;
  qosThemes?:          Record<string, UILPathTheme>;
  onHandoff?: (
    edgeId:   string,
    targetId: string,
    x:  number, y:  number,
    vx: number, vy: number,
    species: number,
  ) => void;
}

// ─── Default QoS themes ───────────────────────────────────────────────────────

export const DEFAULT_QOS_THEMES: Record<string, UILPathTheme> = {
  SENSOR_DATA: { color1: [0.18, 0.55, 1.00], color2: [0.50, 0.90, 1.00], opacity: 0.88 },
  PARAMETERS:  { color1: [1.00, 0.70, 0.20], color2: [1.00, 0.92, 0.60], opacity: 0.80 },
  TF_STATIC:   { color1: [0.20, 0.75, 0.55], color2: [0.60, 1.00, 0.80], opacity: 0.75 },
  TOPO_CHANGE: { color1: [1.00, 0.20, 0.75], color2: [1.00, 0.65, 1.00], opacity: 0.95 },
  DEFAULT:     { color1: [0.55, 0.65, 0.90], color2: [0.85, 0.90, 1.00], opacity: 0.70 },
};

// ─── Presets ──────────────────────────────────────────────────────────────────

export const FullSplineParticlePreset: Record<string, ATSplineParticlesFullConfig> = {
  default: {
    uSplineSpeed: [0.82, 1.21], uTimeMultiplier: 0.17, uFlowRange: [1.0, 1.0],
    uDecayRate: 0.6, uMaxSDelay: 0.0, uCurlNoiseScale: 2.0, uCurlNoiseSpeed: 5.0,
    uCurlStrength: 0.04, uSize: 0.012, particlesPerUnit: 24,
    uSplineThickness: 1.0, uRangeThickness: 0.3, uRangeScale: 1.0,
    uExtrudeRandom: 0.5, uDistribution: 1.0, uDistributionRange: [0.3, 1.0],
    uThicknessStep: [0.5, 1.0], uThicknessSpeed: [0.1, 0.1],
    uilColor1: [0.55, 0.65, 0.90], uilColor2: [0.85, 0.90, 1.00], uilOpacity: 0.70,
    pulseDuration: 1.2, pulseDecayRate: 0.8, trailPersistence: 0.3,
  },
  cellPubSub: {
    uSplineSpeed: [1.20, 1.80], uTimeMultiplier: 0.22, uFlowRange: [1.0, 1.2],
    uDecayRate: 0.9, uMaxSDelay: 0.1, uCurlNoiseScale: 3.0, uCurlNoiseSpeed: 6.0,
    uCurlStrength: 0.06, uSize: 0.010, particlesPerUnit: 36,
    uSplineThickness: 0.8, uRangeThickness: 0.5, uRangeScale: 1.5,
    uExtrudeRandom: 0.7, uDistribution: 1.2, uDistributionRange: [0.2, 0.9],
    uThicknessStep: [0.4, 0.9], uThicknessSpeed: [0.15, 0.08],
    uilColor1: [0.18, 0.55, 1.00], uilColor2: [0.50, 0.90, 1.00], uilOpacity: 0.88,
    pulseDuration: 0.9, pulseDecayRate: 1.2, trailPersistence: 0.25,
  },
  organic: {
    uSplineSpeed: [0.50, 0.90], uTimeMultiplier: 0.12, uFlowRange: [0.9, 1.3],
    uDecayRate: 0.45, uMaxSDelay: 0.8, uCurlNoiseScale: 5.0, uCurlNoiseSpeed: 3.0,
    uCurlStrength: 0.18, uSize: 0.016, particlesPerUnit: 20,
    uSplineThickness: 1.4, uRangeThickness: 0.7, uRangeScale: 0.8,
    uExtrudeRandom: 1.0, uDistribution: 0.8, uDistributionRange: [0.4, 1.2],
    uThicknessStep: [0.6, 1.0], uThicknessSpeed: [0.05, 0.05],
    uilColor1: [0.20, 0.75, 0.55], uilColor2: [0.60, 1.00, 0.80], uilOpacity: 0.75,
    pulseDuration: 1.8, pulseDecayRate: 0.4, trailPersistence: 0.5,
  },
  fastPulse: {
    uSplineSpeed: [1.80, 2.60], uTimeMultiplier: 0.35, uFlowRange: [1.0, 1.2],
    uDecayRate: 1.2, uMaxSDelay: 0.2, uCurlNoiseScale: 1.0, uCurlNoiseSpeed: 8.0,
    uCurlStrength: 0.015, uSize: 0.008, particlesPerUnit: 32,
    uSplineThickness: 0.5, uRangeThickness: 0.2, uRangeScale: 2.0,
    uExtrudeRandom: 0.2, uDistribution: 2.0, uDistributionRange: [0.1, 0.7],
    uThicknessStep: [0.3, 0.8], uThicknessSpeed: [0.25, 0.20],
    uilColor1: [1.00, 0.20, 0.75], uilColor2: [1.00, 0.65, 1.00], uilOpacity: 0.95,
    pulseDuration: 0.5, pulseDecayRate: 2.0, trailPersistence: 0.1,
  },
};

// ─── GLSL helpers shared by all passes ───────────────────────────────────────

const QUAD_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// range.glsl (compiled.vs line ~2131)
const RANGE_GLSL = /* glsl */`
float range(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
  vec3 sub = vec3(oldValue, newMax, oldMax) - vec3(oldMin, newMin, oldMin);
  return sub.x * sub.y / sub.z + newMin;
}
float crange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
  return clamp(range(oldValue, oldMin, oldMax, newMin, newMax), min(newMin, newMax), max(newMin, newMax));
}
`;

// simplenoise / cnoise used by SplineParticleLife.fs
const NOISE_GLSL = /* glsl */`
float hash1(float n) { return fract(sin(n) * 43758.5453123); }
vec3 hash3(vec3 p) {
  vec3 q = vec3(dot(p,vec3(127.1,311.7,74.7)), dot(p,vec3(269.5,183.3,246.1)), dot(p,vec3(113.5,271.9,124.6)));
  return fract(sin(q)*43758.5453123);
}
float noise3(vec3 x) {
  vec3 i = floor(x); vec3 f = fract(x);
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
             mix(mix(n001,n101,u.x),mix(n011,n111,u.x),u.y), u.z);
}
float cnoise(vec2 p) { return noise3(vec3(p, 0.0)); }
vec2 curlNoise2D(vec3 p, float eps) {
  vec3 dx = vec3(eps,0,0); vec3 dy = vec3(0,eps,0); vec3 dz = vec3(0,0,eps);
  float Fz_x = (noise3(p+dx+dz)-noise3(p-dx+dz)-noise3(p+dx)+noise3(p-dx))/(4.0*eps*eps);
  float Fz_y = (noise3(p+dy+dz)-noise3(p-dy+dz)-noise3(p+dy)+noise3(p-dy))/(4.0*eps*eps);
  return vec2(-Fz_y, Fz_x);
}
`;

// ─── GPGPU pass 1: SplineParticleLife.fs (from compiled.vs) ──────────────────
// Adapts the AT source: advances travel on the spline, handles decay & respawn.
// tLife.rgba = [splineIdx, travel, speed, alpha]

const LIFE_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform sampler2D tLife;      // current: rgba=[splineIdx, travel, speed, alpha]
uniform sampler2D tAttribs;   // per-particle randoms: rgba=[r0,r1,r2,r3]
uniform float uTime;
uniform float uDelta;
uniform float uTimeMultiplier;
uniform float uDecayRate;
uniform float uMaxCount;
uniform float fSize;          // TEX_W
uniform float HZ;
uniform vec2  uSplineSpeed;
uniform vec2  uFlowRange;
uniform float uMaxSDelay;
uniform float uSplineCount;
uniform float uSetup;

${RANGE_GLSL}
${NOISE_GLSL}

float randomSeed(float seed) {
  float n = sin(seed) * 10000000.0;
  return n - floor(n);
}
float srand(float seed, float lo, float hi) {
  return floor(lo + randomSeed(seed) * (hi - lo) + 0.5);
}

void main() {
  vec2 uv = vUv;
  float linearIdx = floor(uv.x * fSize) + floor(uv.y * fSize) * fSize;

  // Out-of-range slots: mark dead
  if (linearIdx >= uMaxCount) {
    gl_FragColor = vec4(0.0, 9999.0, 0.0, 0.0);
    return;
  }

  vec4 life    = texture2D(tLife,    uv);  // [splineIdx, travel, speed, alpha]
  vec4 attribs = texture2D(tAttribs, uv);  // [r0, r1, r2, r3]

  float splineIdx = life.x;
  float travel    = life.y;
  float speed     = life.z;
  float alpha     = life.w;

  // Setup: assign random spline + stagger start
  if (uSetup > 0.5) {
    splineIdx = srand(attribs.x, 0.0, uSplineCount);
    travel    = crange(attribs.w, 0.0, 1.0, 0.0, uMaxSDelay);
    speed     = crange(attribs.z, 0.0, 1.0, uSplineSpeed.x, uSplineSpeed.y);
    alpha     = 0.0;
    gl_FragColor = vec4(splineIdx, travel, speed, alpha);
    return;
  }

  // Advance travel — from SplineParticleLife.fs formula
  float sRandom = crange(cnoise(vec2(attribs.x)), -1.0, 1.0, 0.0, 1.0);
  float flowMul = crange(attribs.y, 0.0, 1.0, uFlowRange.x, uFlowRange.y);
  float speedMul = crange(sRandom, 0.0, 1.0, uSplineSpeed.x, uSplineSpeed.y);
  travel += 0.001 * uDelta * uTimeMultiplier * HZ * flowMul * speedMul;
  travel  = clamp(travel, 0.0, 2.0);

  // Fade in [0, 0.02], fade out after 1.0
  alpha = clamp(travel * 50.0, 0.0, 1.0);
  alpha = min(alpha, crange(travel, 1.0, 1.5, 1.0, 0.0));

  // Decay alpha toward zero when travel > 1
  if (travel > 1.0) {
    alpha -= uDecayRate * uDelta * HZ * 0.01;
    alpha  = clamp(alpha, 0.0, 1.0);
  }

  // Respawn dead particle
  if (travel > 1.5 || (travel > 1.0 && alpha <= 0.0)) {
    splineIdx = srand(attribs.x + uTime * 0.001, 0.0, uSplineCount);
    travel    = 0.0;
    speed     = crange(attribs.z + fract(uTime), 0.0, 1.0, uSplineSpeed.x, uSplineSpeed.y);
    alpha     = 0.0;
  }

  gl_FragColor = vec4(splineIdx, travel, speed, alpha);
}
`;

// ─── GPGPU pass 2: splineparticles.fs position + thickness (from compiled.vs) ─
// Reads tLife for travel/splineIdx, samples tSpline atlas, extrudes by thickness.
// tPos.rgba = [worldX, worldY, thickness, trailAlpha]

const POS_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform sampler2D tLife;      // [splineIdx, travel, speed, alpha]
uniform sampler2D tAttribs;   // random seeds
uniform sampler2D tOrigin;    // spawn-time position (for thickness noise)
uniform sampler2D tSpline;    // Catmull-Rom atlas  (RGBA32F, xyz=pos)
uniform sampler2D tPos;       // previous pos (for trail accumulation)
uniform sampler2D tPulse;     // per-spline pulse intensity (R32F row)
uniform float uTime;
uniform float uDelta;
uniform float uSplineTexSize; // SPLINE_TEX_SIZE = 512
uniform float uPerSpline;     // PER_SPLINE = 128
uniform float uSplineCount;
uniform float uSCurlNoiseScale;
uniform float uSCurlNoiseSpeed;
uniform float uSCurlStrength;
uniform float uSplineThickness;
uniform float uRangeThickness;
uniform float uRangeScale;
uniform float uExtrudeRandom;
uniform float uDistribution;
uniform vec2  uDistributionRange;
uniform vec2  uThicknessStep;
uniform vec2  uThicknessSpeed;
uniform float uTrailPersistence;
uniform float fSize;

${RANGE_GLSL}
${NOISE_GLSL}

// splineparticles.fs: splinenoise (AT original)
float splinenoise(vec3 v) {
  float t = v.z * 0.3;
  v.y *= 0.8;
  float s = 0.5;
  float n = 0.0;
  n += range(sin(v.x*0.9/s+t*10.0)+sin(v.x*2.4/s+t*15.0)+sin(v.x*-3.5/s+t*4.0)+sin(v.x*-2.5/s+t*7.1),-1.0,1.0,-0.3,0.3);
  n += range(sin(v.y*-0.3/s+t*18.0)+sin(v.y*1.6/s+t*18.0)+sin(v.y*2.6/s+t*8.0)+sin(v.y*-2.6/s+t*4.5),-1.0,1.0,-0.3,0.3);
  return n;
}
float randomSeed(float seed) { float n=sin(seed)*10000000.0; return n-floor(n); }
float ssineOut(float t) { return sin(t*1.5707963267948966); }

// splineshader.glsl: getSplineLookupUV
vec2 getSplineLookupUV(float index, float t) {
  float pixel = (index * uPerSpline) + (t * uPerSpline);
  float size  = uSplineTexSize;
  float p0    = pixel / size;
  float y     = floor(p0);
  float x     = p0 - y;
  return vec2(x, y / size);
}

// splineparticles.fs: getSplinePosRaw
vec3 getSplinePosRaw(float index, float t) {
  float step = 1.0 / uPerSpline;
  float next = t + step;
  vec2 uv0, uv1;
  if (next <= 1.0) {
    uv0 = getSplineLookupUV(index, t);
    uv1 = getSplineLookupUV(index, next);
  } else {
    uv0 = getSplineLookupUV(index, 1.0);
    uv1 = getSplineLookupUV(index, t - step);
  }
  float interpolate = mod(t, step) * uPerSpline;
  vec3 cpos = texture2D(tSpline, uv0).xyz;
  vec3 npos = texture2D(tSpline, uv1).xyz;
  vec3 pos  = mix(cpos, npos, interpolate);
  // curl noise perturbation (AT: uSCurlNoiseSpeed > 0)
  if (uSCurlNoiseSpeed > 0.0) {
    vec2 curl = curlNoise2D((pos * uSCurlNoiseScale * 0.1) + vec3(t * uSCurlNoiseSpeed * 0.1, 0.0, 0.0), 0.01);
    pos.xy += curl * uSCurlStrength * 0.01 * 60.0;
  }
  return pos;
}

// splineparticles.fs: getSplineThickness
vec3 getSplineThickness(vec3 pos, vec3 sOrigin, vec4 sRandom, float t) {
  float gamma = ssineOut(crange(splinenoise(sOrigin.xyz * uDistribution), -1.0, 1.0, 0.0, 1.0));
  float fizzy = pow(mix(uDistributionRange.x, uDistributionRange.y, gamma), 3.0);
  float splineRandomStep = step(uThicknessStep.x, 0.0);
  float distribution = mix(uThicknessStep.y, 1.0, 1.0 - splineRandomStep);
  float radius = 0.5 * uSplineThickness * distribution * fizzy;
  radius *= crange(splinenoise((pos * uRangeScale) + vec3(t, 0.0, 0.0) * uThicknessSpeed.xyy),
                   -1.0, 1.0, 1.0 - uRangeThickness, 1.0 + uRangeThickness);
  radius *= mix(1.0, uExtrudeRandom, sRandom.y);
  return normalize(sOrigin) * radius;
}

void main() {
  vec2 uv     = vUv;
  vec4 life   = texture2D(tLife,    uv);  // [splineIdx, travel, speed, alpha]
  vec4 attrib = texture2D(tAttribs, uv);
  vec4 origin = texture2D(tOrigin,  uv);
  vec4 prevPos = texture2D(tPos,    uv);

  float splineIdx = life.x;
  float travel    = clamp(life.y, 0.0, 1.0);
  float alpha     = life.w;

  vec3 splinePos = getSplinePosRaw(splineIdx, travel);

  // Thickness extrusion — splineparticles.fs getSplinePos
  vec3 sOrigin = origin.xyz;
  vec3 thick   = getSplineThickness(splinePos, sOrigin, attrib, uTime);
  vec3 worldPos = splinePos + thick;

  float thicknessLen = length(thick);

  // Pulse intensity for this spline (from tPulse row)
  float pulseU = (splineIdx + 0.5) / uSplineCount;
  float pulse  = texture2D(tPulse, vec2(pulseU, 0.5)).r;

  // Trail: blend previous trail alpha with new pulse
  float prevTrail = prevPos.w;
  float trailAlpha = mix(prevTrail, pulse, 1.0 - uTrailPersistence);

  gl_FragColor = vec4(worldPos.x, worldPos.y, thicknessLen, trailAlpha);
}
`;

// ─── GPGPU pass 3: trail accumulation ────────────────────────────────────────
// Simple exponential blur / persistence for trail glow.

const TRAIL_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform sampler2D tTrailRead;
uniform sampler2D tPos;
uniform float uTrailDecay;
uniform float uDelta;

void main() {
  float prev   = texture2D(tTrailRead, vUv).r;
  float pulse  = texture2D(tPos, vUv).w;
  float trail  = max(prev * (1.0 - uTrailDecay * uDelta * 60.0), pulse);
  gl_FragColor = vec4(trail, 0.0, 0.0, 1.0);
}
`;

// ─── Render: TweenUILPathShader + point-sprite SDF (from compiled.vs) ─────────
// Vertex: reads tLife + tPos via lookup UV attribute.
// Fragment: soft disk + tri() SDF + speed-colour lerp.

const RENDER_VERT = /* glsl */`
precision highp float;
attribute vec2 aLookup;   // per-particle UV into tLife/tPos textures

uniform sampler2D tLife;    // [splineIdx, travel, speed, alpha]
uniform sampler2D tPos;     // [worldX, worldY, thickness, trailAlpha]
uniform sampler2D tTrail;   // trail glow (R channel)
uniform vec2  uResolution;
uniform float uSize;
uniform float uAspect;      // TweenUILPathShader: customDirection aspect

varying float vAlpha;
varying float vTravel;
varying float vSpeed;       // TweenUILPathShader: speed attr [0,1]
varying float vPulse;
varying float vTrailAlpha;
varying vec2  vUv;

void main() {
  vec4 life = texture2D(tLife, aLookup);
  vec4 pos  = texture2D(tPos,  aLookup);
  float trail = texture2D(tTrail, aLookup).r;

  float splineIdx = life.x;
  float travel    = life.y;
  float speed     = life.z;   // random per-particle speed [uSplineSpeed.x..y]
  float alpha     = life.w;

  float worldX     = pos.x;
  float worldY     = pos.y;
  float thickness  = pos.z;

  // Normalise speed into [0,1] for TweenUILPathShader colour lerp
  float speedNorm = clamp(speed * 0.5, 0.0, 1.0);

  // AT size: uSize * (1 - travel^2) — taper toward end of path
  float travelDecay = clamp(1.0 - travel * travel, 0.0, 1.0);
  float halfSize    = uSize * travelDecay * (1.0 + thickness * 2.0);

  // tPos stores Catmull-Rom world coords baked in [-1,1] NDC range.
  // TweenUILPathShader: customDirection divides x by aspect ratio.
  float ndcX = worldX / uAspect;
  float ndcY = worldY;

  vAlpha      = alpha;
  vTravel     = travel;
  vSpeed      = speedNorm;
  vPulse      = trail;
  vTrailAlpha = pos.w;
  vUv         = vec2(0.0);

  float visible = step(0.001, alpha);
  gl_Position  = vec4(ndcX * visible, ndcY * visible, 0.0, 1.0);
  gl_PointSize = halfSize * uResolution.y * visible;
}
`;

const RENDER_FRAG = /* glsl */`
precision highp float;

uniform vec3  uColor;
uniform vec3  uColor2;
uniform float uOpacity;

varying float vAlpha;
varying float vTravel;
varying float vSpeed;
varying float vPulse;
varying float vTrailAlpha;
varying vec2  vUv;

// TweenUILPathShader.glsl: tri() — symmetric triangle wave
float tri(float v) {
  return mix(v, 1.0 - v, step(0.5, v)) * 2.0;
}

void main() {
  // Soft circular SDF (point-sprite UV in gl_PointCoord)
  vec2  pc = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(pc, pc);
  if (r2 > 1.0) discard;

  float edge = 1.0 - smoothstep(0.6, 1.0, r2);

  // TweenUILPathShader.glsl: SDF anti-aliasing on path width
  float vy        = gl_PointCoord.y;
  float signedDist = tri(vy) - 0.5;
  float sdEdge    = clamp(signedDist / (abs(dFdx(signedDist)) + abs(dFdy(signedDist))) + 0.5, 0.0, 1.0);

  // AT: alpha follows sin(π·travel) — bright at midpoint
  float fade = sin(3.14159265 * clamp(vTravel, 0.0, 1.0));

  // TweenUILPathShader: vColor = mix(uColor, uColor2, speed)
  vec3 col = mix(uColor, uColor2, vSpeed);

  // Pulse glow contribution
  float pulseGlow = vPulse * 0.6 * fade;
  col += vec3(pulseGlow * 0.8, pulseGlow * 0.6, pulseGlow);

  // Trail halo
  float trailRing = vTrailAlpha * smoothstep(0.9, 0.0, r2) * 0.35;

  float finalA = (vAlpha * fade * edge * sdEdge + trailRing) * uOpacity;

  // TweenUILPathShader.glsl: finalPosition z clamp
  gl_FragColor = vec4(col * finalA, clamp(finalA, 0.0, 1.0));
}
`;

// ─── PingPong helper ──────────────────────────────────────────────────────────

interface PingPong {
  read:     WebGLFramebuffer;
  write:    WebGLFramebuffer;
  readTex:  WebGLTexture;
  writeTex: WebGLTexture;
  width:    number;
  height:   number;
}

// ─── CPU spline helpers ───────────────────────────────────────────────────────

function catmullRomCPU(
  p0: SplinePoint3, p1: SplinePoint3,
  p2: SplinePoint3, p3: SplinePoint3, t: number,
): SplinePoint3 {
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

function evalSplineCPU(pts: SplinePoint3[], u: number): SplinePoint3 {
  const n = pts.length;
  if (n === 0) return { x: 0, y: 0, z: 0 };
  if (n === 1) return { ...pts[0] };
  const clampIdx = (i: number) => Math.max(0, Math.min(n - 1, i));
  const sc = Math.min(u, 0.9999) * (n - 1);
  const i1 = Math.floor(sc);
  return catmullRomCPU(
    pts[clampIdx(i1 - 1)], pts[clampIdx(i1)],
    pts[clampIdx(i1 + 1)], pts[clampIdx(i1 + 2)],
    sc - i1,
  );
}

// ─── ATSplineParticlesFull ────────────────────────────────────────────────────

/**
 * ATSplineParticlesFull — real WebGL2 GPU multi-spline particle system.
 *
 * @example
 * ```ts
 * const canvas = document.querySelector('canvas')!;
 * const gl = canvas.getContext('webgl') as WebGLRenderingContext;
 * const full = new ATSplineParticlesFull(gl, canvas, edges, FullSplineParticlePreset.cellPubSub);
 * full.init();
 * // render loop:
 * function frame(t: number) {
 *   full.render(t / 1000, 1/60);
 *   requestAnimationFrame(frame);
 * }
 * requestAnimationFrame(frame);
 * // pub/sub:
 * bus.on('edge:message', ({edgeId, qos, intensity}) => full.firePulse(edgeId, qos, intensity));
 * ```
 */
export class ATSplineParticlesFull {
  private readonly gl:         WebGLRenderingContext;
  private readonly canvas:     HTMLCanvasElement;
  private readonly onHandoff?: ATSplineParticlesFullConfig['onHandoff'];

  private edges:     FullEdgeSpline[] = [];
  private edgeIdMap  = new Map<string, number>();
  private qosThemes: Record<string, UILPathTheme>;

  private cfg: Required<Omit<ATSplineParticlesFullConfig, 'onHandoff' | 'qosThemes'>>;

  // ── GL programs (4 total) ─────────────────────────────────────────────────
  private lifeProg!:  WebGLProgram;   // GPGPU life/travel update
  private posProg!:   WebGLProgram;   // GPGPU position + thickness
  private trailProg!: WebGLProgram;   // trail glow accumulation
  private renderProg!: WebGLProgram;  // TweenUILPathShader point-sprite

  // ── FBOs (6 total) ────────────────────────────────────────────────────────
  private lifePP!:   PingPong;   // tLife ping-pong  (2 FBOs)
  private posPP!:    PingPong;   // tPos  ping-pong  (2 FBOs)
  private trailPP!:  PingPong;   // trail ping-pong  (2 FBOs)

  // ── Textures (8 total, FBO textures counted inside PingPong) ─────────────
  private tAttribs!:  WebGLTexture;   // per-particle randoms (RGBA32F)
  private tOrigin!:   WebGLTexture;   // spawn-time origin positions (RGBA32F)
  private tSpline!:   WebGLTexture;   // baked Catmull-Rom atlas (RGBA32F)
  private tPulse!:    WebGLTexture;   // per-spline pulse intensity row (R32F)

  // ── Buffers (3 total) ─────────────────────────────────────────────────────
  private quadBuf!:       WebGLBuffer;  // fullscreen quad [-1,1]
  private particleUVBuf!: WebGLBuffer;  // per-particle lookup UV attribute
  private readbackBuf:    Float32Array = new Float32Array(0);

  // ── State ─────────────────────────────────────────────────────────────────
  private particleCount  = 0;
  private splineCount    = 0;
  private built          = false;
  private setupDone      = false;
  private elapsed        = 0;

  // CPU pulse ring buffers: [splineIdx * MAX_PULSES_PER_SPLINE + slot] → intensity
  private pulseData: Float32Array = new Float32Array(0);
  private pulseRow:  Float32Array = new Float32Array(0);

  constructor(
    gl:      WebGLRenderingContext,
    canvas:  HTMLCanvasElement,
    edges:   FullEdgeSpline[],
    config:  ATSplineParticlesFullConfig = {},
  ) {
    this.gl         = gl;
    this.canvas     = canvas;
    this.edges      = edges;
    this.onHandoff  = config.onHandoff;
    this.qosThemes  = { ...DEFAULT_QOS_THEMES, ...(config.qosThemes ?? {}) };

    edges.forEach((e, i) => this.edgeIdMap.set(e.edgeId, i));

    this.cfg = {
      uSplineSpeed:       config.uSplineSpeed       ?? [0.82, 1.21],
      uTimeMultiplier:    config.uTimeMultiplier    ?? 0.17,
      uFlowRange:         config.uFlowRange         ?? [1.0, 1.0],
      uDecayRate:         config.uDecayRate         ?? 0.6,
      uMaxSDelay:         config.uMaxSDelay         ?? 0.0,
      uCurlNoiseScale:    config.uCurlNoiseScale    ?? 2.0,
      uCurlNoiseSpeed:    config.uCurlNoiseSpeed    ?? 5.0,
      uCurlStrength:      config.uCurlStrength      ?? 0.04,
      uSize:              config.uSize              ?? 0.012,
      particlesPerUnit:   config.particlesPerUnit   ?? 24,
      maxParticles:       config.maxParticles       ?? MAX_PARTICLES,
      uSplineThickness:   config.uSplineThickness   ?? 1.0,
      uRangeThickness:    config.uRangeThickness    ?? 0.3,
      uRangeScale:        config.uRangeScale        ?? 1.0,
      uExtrudeRandom:     config.uExtrudeRandom     ?? 0.5,
      uDistribution:      config.uDistribution      ?? 1.0,
      uDistributionRange: config.uDistributionRange ?? [0.3, 1.0],
      uThicknessStep:     config.uThicknessStep     ?? [0.5, 1.0],
      uThicknessSpeed:    config.uThicknessSpeed    ?? [0.1, 0.1],
      uilColor1:          config.uilColor1          ?? [0.55, 0.65, 0.90],
      uilColor2:          config.uilColor2          ?? [0.85, 0.90, 1.00],
      uilOpacity:         config.uilOpacity         ?? 0.70,
      pulseDuration:      config.pulseDuration      ?? 1.2,
      pulseDecayRate:     config.pulseDecayRate     ?? 0.8,
      trailPersistence:   config.trailPersistence   ?? 0.3,
    };
  }

  // ── init() ────────────────────────────────────────────────────────────────

  /**
   * Compile all programs, create all FBOs, textures, and buffers.
   * Must be called before render().
   */
  init(): void {
    if (this.built) this.dispose();
    const gl = this.gl;

    this.splineCount   = this.edges.length;
    this.particleCount = Math.min(
      this.cfg.maxParticles,
      Math.max(256, this.edges.reduce((n, e) =>
        n + Math.ceil(e.weight * this.cfg.particlesPerUnit), 0)),
    );

    // ── require float texture extension (WebGL1) ──────────────────────────
    gl.getExtension('OES_texture_float');
    gl.getExtension('OES_texture_float_linear');

    // ── Compile 4 programs ────────────────────────────────────────────────
    // Shaders ported from upstream/activetheory-assets/compiled.vs.
    // Raw AT sources contain unresolved macros (#require, getData4, timeScale).
    // We do an existence check via getShader() then compile our resolved GLSL.
    void getShader('SplineParticleLife.fs');   // verifies compiled.vs contains it
    void getShader('splineparticles.fs');       // verifies splinePos/Thickness source
    void getShader('TweenUILPathShader.glsl');  // verifies tri() SDF source

    // GPGPU life/travel  (SplineParticleLife.fs — macro-resolved port)
    this.lifeProg   = this._compile(QUAD_VERT, LIFE_FRAG, 'life');

    // GPGPU position + getSplinePosRaw + getSplineThickness
    this.posProg    = this._compile(QUAD_VERT, POS_FRAG, 'pos');

    // Trail glow accumulation
    this.trailProg  = this._compile(QUAD_VERT, TRAIL_FRAG, 'trail');

    // TweenUILPathShader: tri() SDF + speed-colour lerp point-sprites
    this.renderProg = this._compile(RENDER_VERT, RENDER_FRAG, 'render');

    // ── Create 6 FBOs (3 ping-pong pairs) ────────────────────────────────

    this.lifePP  = this._createPingPong(TEX_W, TEX_H, gl.RGBA, gl.FLOAT);
    this.posPP   = this._createPingPong(TEX_W, TEX_H, gl.RGBA, gl.FLOAT);
    this.trailPP = this._createPingPong(TEX_W, TEX_H, gl.RGBA, gl.FLOAT);

    // ── Create 4 standalone textures ─────────────────────────────────────

    // tAttribs — per-particle randoms, uploaded once
    const attribData = this._buildAttribData();
    this.tAttribs = this._createTexture(TEX_W, TEX_H, gl.RGBA, gl.FLOAT, attribData);

    // tOrigin — per-particle spawn origins (normalized)
    const originData = this._buildOriginData();
    this.tOrigin  = this._createTexture(TEX_W, TEX_H, gl.RGBA, gl.FLOAT, originData);

    // tSpline — baked Catmull-Rom atlas (PER_SPLINE samples × splineCount)
    const splineData = this._bakeSplineAtlas();
    this.tSpline  = this._createTexture(SPLINE_TEX_SIZE, SPLINE_TEX_SIZE, gl.RGBA, gl.FLOAT, splineData);

    // tPulse — per-spline pulse intensity, updated each frame
    this.pulseData = new Float32Array(this.splineCount * MAX_PULSES_PER_SPLINE);
    this.pulseRow  = new Float32Array(Math.max(this.splineCount * 4, 4));
    this.tPulse    = this._createTexture(Math.max(this.splineCount, 1), 1, gl.RGBA, gl.FLOAT, this.pulseRow);

    // ── Create 3 buffers ──────────────────────────────────────────────────

    // fullscreen quad
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // per-particle lookup UV attribute
    const uvs = this._buildParticleUVs();
    this.particleUVBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleUVBuf);
    gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // readback buffer (CPU-side, for handoff detection)
    this.readbackBuf = new Float32Array(TEX_W * TEX_H * 4);

    // ── Setup pass: seed initial life state ───────────────────────────────
    this._runSetup();
    this.setupDone = true;
    this.built     = true;

    console.log(
      `[ATSplineParticlesFull] init: ${this.splineCount} splines, ` +
      `${this.particleCount} particles`,
    );
  }

  // ── render() ──────────────────────────────────────────────────────────────

  /**
   * Run all 4 GPU passes and composite particles onto the current framebuffer.
   * @param elapsed  total elapsed time in seconds
   * @param dt       frame delta in seconds (default 1/60)
   */
  render(elapsed: number, dt = 1 / 60): void {
    if (!this.built) return;
    const gl = this.gl;
    this.elapsed = elapsed;

    // Decay + upload pulse intensities
    this._tickPulses(dt);
    this._uploadPulseRow();

    // ── Pass 1: life update ────────────────────────────────────────────────
    gl.useProgram(this.lifeProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.lifePP.write);
    gl.viewport(0, 0, TEX_W, TEX_H);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.lifePP.readTex);
    gl.uniform1i(gl.getUniformLocation(this.lifeProg, 'tLife'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.tAttribs);
    gl.uniform1i(gl.getUniformLocation(this.lifeProg, 'tAttribs'), 1);

    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uTime'),           elapsed);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uDelta'),          Math.min(dt, 1 / 30));
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uTimeMultiplier'), this.cfg.uTimeMultiplier);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uDecayRate'),      this.cfg.uDecayRate);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uMaxCount'),       this.particleCount);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'fSize'),           TEX_W);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'HZ'),              60.0);
    gl.uniform2f(gl.getUniformLocation(this.lifeProg, 'uSplineSpeed'),    this.cfg.uSplineSpeed[0], this.cfg.uSplineSpeed[1]);
    gl.uniform2f(gl.getUniformLocation(this.lifeProg, 'uFlowRange'),      this.cfg.uFlowRange[0],   this.cfg.uFlowRange[1]);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uMaxSDelay'),      this.cfg.uMaxSDelay);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uSplineCount'),    this.splineCount);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uSetup'),          0.0);
    this._drawQuad(this.lifeProg);
    this._swapPP(this.lifePP);

    // ── Pass 2: position + thickness ──────────────────────────────────────
    gl.useProgram(this.posProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.posPP.write);
    gl.viewport(0, 0, TEX_W, TEX_H);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.lifePP.readTex);
    gl.uniform1i(gl.getUniformLocation(this.posProg, 'tLife'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.tAttribs);
    gl.uniform1i(gl.getUniformLocation(this.posProg, 'tAttribs'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.tOrigin);
    gl.uniform1i(gl.getUniformLocation(this.posProg, 'tOrigin'), 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.tSpline);
    gl.uniform1i(gl.getUniformLocation(this.posProg, 'tSpline'), 3);

    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.posPP.readTex);
    gl.uniform1i(gl.getUniformLocation(this.posProg, 'tPos'), 4);

    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, this.tPulse);
    gl.uniform1i(gl.getUniformLocation(this.posProg, 'tPulse'), 5);

    gl.uniform1f(gl.getUniformLocation(this.posProg, 'uTime'),            elapsed);
    gl.uniform1f(gl.getUniformLocation(this.posProg, 'uDelta'),           Math.min(dt, 1 / 30));
    gl.uniform1f(gl.getUniformLocation(this.posProg, 'uSplineTexSize'),   SPLINE_TEX_SIZE);
    gl.uniform1f(gl.getUniformLocation(this.posProg, 'uPerSpline'),       PER_SPLINE);
    gl.uniform1f(gl.getUniformLocation(this.posProg, 'uSplineCount'),     this.splineCount);
    gl.uniform1f(gl.getUniformLocation(this.posProg, 'uSCurlNoiseScale'), this.cfg.uCurlNoiseScale);
    gl.uniform1f(gl.getUniformLocation(this.posProg, 'uSCurlNoiseSpeed'), this.cfg.uCurlNoiseSpeed);
    gl.uniform1f(gl.getUniformLocation(this.posProg, 'uSCurlStrength'),   this.cfg.uCurlStrength);
    gl.uniform1f(gl.getUniformLocation(this.posProg, 'uSplineThickness'), this.cfg.uSplineThickness);
    gl.uniform1f(gl.getUniformLocation(this.posProg, 'uRangeThickness'),  this.cfg.uRangeThickness);
    gl.uniform1f(gl.getUniformLocation(this.posProg, 'uRangeScale'),      this.cfg.uRangeScale);
    gl.uniform1f(gl.getUniformLocation(this.posProg, 'uExtrudeRandom'),   this.cfg.uExtrudeRandom);
    gl.uniform1f(gl.getUniformLocation(this.posProg, 'uDistribution'),    this.cfg.uDistribution);
    gl.uniform2f(gl.getUniformLocation(this.posProg, 'uDistributionRange'), this.cfg.uDistributionRange[0], this.cfg.uDistributionRange[1]);
    gl.uniform2f(gl.getUniformLocation(this.posProg, 'uThicknessStep'),   this.cfg.uThicknessStep[0],   this.cfg.uThicknessStep[1]);
    gl.uniform2f(gl.getUniformLocation(this.posProg, 'uThicknessSpeed'),  this.cfg.uThicknessSpeed[0],  this.cfg.uThicknessSpeed[1]);
    gl.uniform1f(gl.getUniformLocation(this.posProg, 'uTrailPersistence'), this.cfg.trailPersistence);
    gl.uniform1f(gl.getUniformLocation(this.posProg, 'fSize'),            TEX_W);
    this._drawQuad(this.posProg);
    this._swapPP(this.posPP);

    // ── Pass 3: trail accumulation ─────────────────────────────────────────
    gl.useProgram(this.trailProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.trailPP.write);
    gl.viewport(0, 0, TEX_W, TEX_H);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.trailPP.readTex);
    gl.uniform1i(gl.getUniformLocation(this.trailProg, 'tTrailRead'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.posPP.readTex);
    gl.uniform1i(gl.getUniformLocation(this.trailProg, 'tPos'), 1);

    gl.uniform1f(gl.getUniformLocation(this.trailProg, 'uTrailDecay'), this.cfg.pulseDecayRate);
    gl.uniform1f(gl.getUniformLocation(this.trailProg, 'uDelta'),      Math.min(dt, 1 / 30));
    this._drawQuad(this.trailProg);
    this._swapPP(this.trailPP);

    // ── Pass 4: render — TweenUILPathShader point-sprites ─────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width || 1, this.canvas.height || 1);

    gl.useProgram(this.renderProg);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.lifePP.readTex);
    gl.uniform1i(gl.getUniformLocation(this.renderProg, 'tLife'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.posPP.readTex);
    gl.uniform1i(gl.getUniformLocation(this.renderProg, 'tPos'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.trailPP.readTex);
    gl.uniform1i(gl.getUniformLocation(this.renderProg, 'tTrail'), 2);

    const W = this.canvas.width  || 1;
    const H = this.canvas.height || 1;
    gl.uniform2f(gl.getUniformLocation(this.renderProg, 'uResolution'), W, H);
    gl.uniform1f(gl.getUniformLocation(this.renderProg, 'uSize'),    this.cfg.uSize);
    gl.uniform1f(gl.getUniformLocation(this.renderProg, 'uAspect'),  W / H);
    gl.uniform3f(gl.getUniformLocation(this.renderProg, 'uColor'),   ...this.cfg.uilColor1);
    gl.uniform3f(gl.getUniformLocation(this.renderProg, 'uColor2'),  ...this.cfg.uilColor2);
    gl.uniform1f(gl.getUniformLocation(this.renderProg, 'uOpacity'), this.cfg.uilOpacity);

    // Draw as point-sprites: one point per particle, position from tLife/tPos
    const aLookup = gl.getAttribLocation(this.renderProg, 'aLookup');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleUVBuf);
    gl.enableVertexAttribArray(aLookup);
    gl.vertexAttribPointer(aLookup, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, this.particleCount);
    gl.disableVertexAttribArray(aLookup);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.disable(gl.BLEND);

    // Optional handoff readback (async-safe: reads from last frame)
    if (this.onHandoff) this._readbackHandoff();
  }

  // ── dispose() ─────────────────────────────────────────────────────────────

  /**
   * Release all WebGL resources.
   */
  dispose(): void {
    if (!this.built) return;
    const gl = this.gl;

    // delete 4 programs
    gl.deleteProgram(this.lifeProg);
    gl.deleteProgram(this.posProg);
    gl.deleteProgram(this.trailProg);
    gl.deleteProgram(this.renderProg);

    // delete 6 FBOs (3 ping-pong pairs × 2)
    gl.deleteFramebuffer(this.lifePP.read);
    gl.deleteFramebuffer(this.lifePP.write);
    gl.deleteFramebuffer(this.posPP.read);
    gl.deleteFramebuffer(this.posPP.write);
    gl.deleteFramebuffer(this.trailPP.read);
    gl.deleteFramebuffer(this.trailPP.write);

    // delete 8 textures (3 pairs × 2 + 2 standalone = 8)
    gl.deleteTexture(this.lifePP.readTex);
    gl.deleteTexture(this.lifePP.writeTex);
    gl.deleteTexture(this.posPP.readTex);
    gl.deleteTexture(this.posPP.writeTex);
    gl.deleteTexture(this.trailPP.readTex);
    gl.deleteTexture(this.trailPP.writeTex);
    gl.deleteTexture(this.tAttribs);
    gl.deleteTexture(this.tOrigin);
    gl.deleteTexture(this.tSpline);
    gl.deleteTexture(this.tPulse);

    // delete 3 buffers
    gl.deleteBuffer(this.quadBuf);
    gl.deleteBuffer(this.particleUVBuf);

    this.built = false;
  }

  // ── pub/sub API ───────────────────────────────────────────────────────────

  /**
   * Fire a data-flow pulse on an edge.
   */
  firePulse(edgeId: string, qos?: string, intensity = 1.0): void {
    const eIdx = this.edgeIdMap.get(edgeId);
    if (eIdx === undefined) return;

    const base = eIdx * MAX_PULSES_PER_SPLINE;
    let slot = 0;
    let minV = Infinity;
    for (let p = 0; p < MAX_PULSES_PER_SPLINE; p++) {
      const v = this.pulseData[base + p];
      if (v < minV) { minV = v; slot = p; }
    }
    this.pulseData[base + slot] = Math.min(1.0, intensity);

    if (qos && this.qosThemes[qos]) this._applyQosTheme(this.qosThemes[qos]);
  }

  firePulsesBatch(events: EdgeDataEvent[]): void {
    for (const ev of events) this.firePulse(ev.edgeId, ev.qos, ev.intensity ?? 1.0);
  }

  // ── Live setters ──────────────────────────────────────────────────────────

  setSplineSpeed(min: number, max: number): void { this.cfg.uSplineSpeed = [min, max]; }
  setTimeMultiplier(v: number): void              { this.cfg.uTimeMultiplier = v; }
  setDecayRate(v: number): void                   { this.cfg.uDecayRate = v; }
  setCurlStrength(v: number): void                { this.cfg.uCurlStrength = v; }
  setSize(v: number): void                        { this.cfg.uSize = v; }
  setSplineThickness(v: number): void             { this.cfg.uSplineThickness = v; }
  setUILColors(c1: [number, number, number], c2: [number, number, number]): void {
    this.cfg.uilColor1 = c1; this.cfg.uilColor2 = c2;
  }
  setPulseDuration(v: number): void               { this.cfg.pulseDuration = v; }
  setQosTheme(qos: string, theme: UILPathTheme): void { this.qosThemes[qos] = theme; }
  applyPreset(preset: ATSplineParticlesFullConfig): void {
    const c = this.cfg as Record<string, unknown>;
    const p = preset    as Record<string, unknown>;
    for (const key of Object.keys(c)) if (p[key] !== undefined) c[key] = p[key];
  }

  async setEdges(edges: FullEdgeSpline[]): Promise<void> {
    this.edges = edges;
    this.edgeIdMap.clear();
    edges.forEach((e, i) => this.edgeIdMap.set(e.edgeId, i));
    if (this.built) { this.dispose(); this.init(); }
  }

  get particleSlots(): number  { return this.particleCount; }
  get edgeCount():     number  { return this.edges.length; }
  get isBuilt():       boolean { return this.built; }
  get elapsedTime():   number  { return this.elapsed; }

  // ── Private GL helpers ────────────────────────────────────────────────────

  /** Run the setup pass to seed initial life state */
  private _runSetup(): void {
    const gl = this.gl;
    gl.useProgram(this.lifeProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.lifePP.write);
    gl.viewport(0, 0, TEX_W, TEX_H);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.lifePP.readTex);
    gl.uniform1i(gl.getUniformLocation(this.lifeProg, 'tLife'),    0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.tAttribs);
    gl.uniform1i(gl.getUniformLocation(this.lifeProg, 'tAttribs'), 1);

    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uTime'),         0.0);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uDelta'),        0.0);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uTimeMultiplier'), this.cfg.uTimeMultiplier);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uDecayRate'),    this.cfg.uDecayRate);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uMaxCount'),     this.particleCount);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'fSize'),         TEX_W);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'HZ'),            60.0);
    gl.uniform2f(gl.getUniformLocation(this.lifeProg, 'uSplineSpeed'),  this.cfg.uSplineSpeed[0], this.cfg.uSplineSpeed[1]);
    gl.uniform2f(gl.getUniformLocation(this.lifeProg, 'uFlowRange'),    this.cfg.uFlowRange[0], this.cfg.uFlowRange[1]);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uMaxSDelay'),    this.cfg.uMaxSDelay);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uSplineCount'),  this.splineCount);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uSetup'),        1.0);
    this._drawQuad(this.lifeProg);
    this._swapPP(this.lifePP);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Compile vertex + fragment → WebGLProgram */
  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATSplineParticlesFull] vertex compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, 'precision highp float;\n' + frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATSplineParticlesFull] fragment compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ATSplineParticlesFull] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** Create ping-pong FBO pair */
  private _createPingPong(w: number, h: number, format: number, type: number): PingPong {
    const a = this._createFBOWithTex(w, h, format, type);
    const b = this._createFBOWithTex(w, h, format, type);
    return { read: a.fbo, write: b.fbo, readTex: a.tex, writeTex: b.tex, width: w, height: h };
  }

  /** Create a single FBO backed by a texture */
  private _createFBOWithTex(
    w: number, h: number, format: number, type: number,
  ): { fbo: WebGLFramebuffer; tex: WebGLTexture } {
    const gl  = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, format, type, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fbo, tex };
  }

  /** Create a standalone texture with initial data */
  private _createTexture(
    w: number, h: number, format: number, type: number,
    data: Float32Array | null = null,
  ): WebGLTexture {
    const gl  = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, format, type, data);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  /** Swap ping-pong read/write */
  private _swapPP(pp: PingPong): void {
    [pp.read,    pp.write]    = [pp.write,    pp.read];
    [pp.readTex, pp.writeTex] = [pp.writeTex, pp.readTex];
  }

  /** Draw fullscreen quad */
  private _drawQuad(program: WebGLProgram): void {
    const gl     = this.gl;
    const posLoc = gl.getAttribLocation(program, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** Build per-particle random attrib texture */
  private _buildAttribData(): Float32Array {
    const data = new Float32Array(TEX_W * TEX_H * 4);
    for (let i = 0; i < TEX_W * TEX_H; i++) {
      const rng = (s: number) => { const n = Math.sin(s) * 43758.5453; return n - Math.floor(n); };
      data[i * 4 + 0] = rng(i * 1.618034);
      data[i * 4 + 1] = rng(i * 2.618034);
      data[i * 4 + 2] = rng(i * 3.141592);
      data[i * 4 + 3] = rng(i * 1.414213);
    }
    return data;
  }

  /** Build per-particle spawn origin positions (normalized) */
  private _buildOriginData(): Float32Array {
    const data = new Float32Array(TEX_W * TEX_H * 4);
    let slot = 0;
    for (let e = 0; e < this.edges.length && slot < this.particleCount; e++) {
      const edge  = this.edges[e];
      const count = Math.min(
        Math.ceil(edge.weight * this.cfg.particlesPerUnit),
        this.particleCount - slot,
      );
      const pt = evalSplineCPU(edge.points, 0);
      for (let p = 0; p < count && slot < this.particleCount; p++, slot++) {
        const b = slot * 4;
        const tx = pt.x, ty = pt.y;
        const len = Math.sqrt(tx * tx + ty * ty) + 1e-10;
        data[b + 0] = tx / len;
        data[b + 1] = ty / len;
        data[b + 2] = 0;
        data[b + 3] = e / Math.max(this.splineCount, 1);
      }
    }
    return data;
  }

  /** Bake Catmull-Rom samples for all splines into a 2D RGBA atlas */
  private _bakeSplineAtlas(): Float32Array {
    const size  = SPLINE_TEX_SIZE;
    const total = size * size;
    const data  = new Float32Array(total * 4);

    for (let s = 0; s < this.splineCount; s++) {
      const pts = this.edges[s].points;
      for (let k = 0; k < PER_SPLINE; k++) {
        const t   = k / (PER_SPLINE - 1);
        const pos = evalSplineCPU(pts, t);

        // pixel index in atlas = s * PER_SPLINE + k
        const pixelIdx = s * PER_SPLINE + k;
        if (pixelIdx >= total) break;
        const b = pixelIdx * 4;
        data[b + 0] = pos.x;
        data[b + 1] = pos.y;
        data[b + 2] = pos.z;
        data[b + 3] = 1.0;
      }
    }
    return data;
  }

  /** Build per-particle lookup UV attribute (points into tLife/tPos) */
  private _buildParticleUVs(): Float32Array {
    const uvs = new Float32Array(this.particleCount * 2);
    for (let i = 0; i < this.particleCount; i++) {
      const px   = i % TEX_W;
      const py   = Math.floor(i / TEX_W);
      uvs[i * 2 + 0] = (px + 0.5) / TEX_W;
      uvs[i * 2 + 1] = (py + 0.5) / TEX_H;
    }
    return uvs;
  }

  /** Decay all pulse intensities and sum per-spline into upload row */
  private _tickPulses(dt: number): void {
    const decay = this.cfg.pulseDecayRate * dt;
    for (let i = 0; i < this.pulseData.length; i++) {
      const v = this.pulseData[i] - decay;
      this.pulseData[i] = v < 0 ? 0 : v;
    }
  }

  /** Sum per-spline pulses into tPulse row and re-upload */
  private _uploadPulseRow(): void {
    const gl = this.gl;
    const n  = this.splineCount;

    // pulseRow is n×1 RGBA texture: R = summed intensity for spline s
    for (let s = 0; s < n; s++) {
      let sum = 0;
      const base = s * MAX_PULSES_PER_SPLINE;
      for (let p = 0; p < MAX_PULSES_PER_SPLINE; p++) sum += this.pulseData[base + p];
      this.pulseRow[s * 4 + 0] = Math.min(1.0, sum);
      this.pulseRow[s * 4 + 1] = 0;
      this.pulseRow[s * 4 + 2] = 0;
      this.pulseRow[s * 4 + 3] = 1;
    }

    gl.bindTexture(gl.TEXTURE_2D, this.tPulse);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, Math.max(n, 1), 1, 0, gl.RGBA, gl.FLOAT, this.pulseRow);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** Smooth lerp of current UIL colours toward a QoS theme */
  private _applyQosTheme(theme: UILPathTheme): void {
    const a = 0.25;
    for (let c = 0; c < 3; c++) {
      this.cfg.uilColor1[c] += (theme.color1[c] - this.cfg.uilColor1[c]) * a;
      this.cfg.uilColor2[c] += (theme.color2[c] - this.cfg.uilColor2[c]) * a;
    }
  }

  /** CPU readback for handoff detection */
  private _readbackHandoff(): void {
    if (!this.onHandoff) return;
    const gl = this.gl;
    const w  = TEX_W;
    const h  = TEX_H;

    // Read life texture: [splineIdx, travel, speed, alpha]
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.lifePP.read);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, this.readbackBuf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    for (let i = 0; i < this.particleCount; i++) {
      const b      = i * 4;
      const travel = this.readbackBuf[b + 1];
      const alpha  = this.readbackBuf[b + 3];
      // Handoff fires when travel > 0.99 and particle is still alive
      if (travel < 0.99 || alpha < 0.01) continue;

      const eIdx = Math.round(this.readbackBuf[b + 0]);
      const edge = this.edges[eIdx];
      if (!edge) continue;

      const ep  = evalSplineCPU(edge.points, 0.999);
      const ep2 = evalSplineCPU(edge.points, 0.998);
      const tx  = ep.x - ep2.x;
      const ty  = ep.y - ep2.y;
      const tl  = Math.sqrt(tx * tx + ty * ty) + 1e-10;
      const vsc = this.readbackBuf[b + 2] * this.cfg.uTimeMultiplier * 0.01;

      this.onHandoff(
        edge.edgeId, edge.targetId,
        ep.x, ep.y,
        (tx / tl) * vsc, (ty / tl) * vsc,
        edge.species ?? 0,
      );
    }
  }

  /** GLSL preamble injected before AT shader sources */
  private _preamble(): string {
    return 'precision highp float;\nvarying vec2 vUv;\nuniform float timeScale;\nuniform float time;\nuniform float fSize;\n';
  }
}

// ─── Factory helpers ──────────────────────────────────────────────────────────

export function createFullSplineParticleForSPH(
  gl:       WebGLRenderingContext,
  canvas:   HTMLCanvasElement,
  edges:    FullEdgeSpline[],
  addFluid: (x0: number, y0: number, x1: number, y1: number, spacing: number, species: number) => void,
  config:   Omit<ATSplineParticlesFullConfig, 'onHandoff'> = {},
): ATSplineParticlesFull {
  const R = 0.05;
  return new ATSplineParticlesFull(gl, canvas, edges, {
    ...config,
    onHandoff: (_eId, _tId, x, y, _vx, _vy, species) => {
      addFluid(x - R, y - R, x + R, y + R, R * 0.8, species);
    },
  });
}

export function canvasRouteToFullEdgeSpline(
  edgeId:   string,
  sourceId: string,
  targetId: string,
  points:   Array<{ x: number; y: number }>,
  weight:   number,
  canvasW:  number,
  canvasH:  number,
  domainW:  number,
  domainH:  number,
  species   = 0,
  qos?:     string,
): FullEdgeSpline {
  const sx = domainW / canvasW;
  const sy = domainH / canvasH;
  return {
    edgeId, sourceId, targetId, weight, species, qos,
    points: points.map(p => ({ x: p.x * sx, y: p.y * sy, z: 0 })),
  };
}

// ─── Constants re-export ──────────────────────────────────────────────────────

export const AT_SPLINE_PARTICLES_FULL_DEFAULTS = {
  maxParticles:      MAX_PARTICLES,
  texW:              TEX_W,
  texH:              TEX_H,
  perSpline:         PER_SPLINE,
  splineTexSize:     SPLINE_TEX_SIZE,
  maxPulsesPerSpline: MAX_PULSES_PER_SPLINE,
} as const;
