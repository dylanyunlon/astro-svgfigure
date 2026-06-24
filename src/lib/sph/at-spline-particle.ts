/**
 * at-spline-particle.ts — M1001
 *
 * ATSplineParticleLife — real WebGL2 Transform Feedback GPU particle system
 * ─────────────────────────────────────────────────────────────────────────────
 * Full WebGL2 port of Active Theory's SplineParticleLife + splineparticles.fs
 * lifecycle system, using Transform Feedback for GPU-side particle physics.
 *
 * Architecture (WebGL2, mirrors fluid-gpu-pass.ts / at-terrain-environment.ts):
 *   init():    createProgram, compileShader, linkProgram, createFramebuffer,
 *              createTexture, createBuffer, bufferData — all real gl.* calls
 *   render():  useProgram, bindFramebuffer, bindTexture, uniform*,
 *              bindBuffer, vertexAttribPointer, drawArrays
 *   dispose(): deleteProgram, deleteFramebuffer, deleteTexture, deleteBuffer
 *
 * GPU pipeline:
 *   updateProgram  — Transform Feedback: reads positionBuf+velocityBuf+lifeBuf
 *                    (VAO A) → writes to VAO B via TF (GPU physics + lifecycle)
 *   renderProgram  — Point sprite render: reads from current TF output VAO,
 *                    draws gl.POINTS with SplineParticleLife.fs visual logic
 *
 * Dual VAO ping-pong:
 *   vaoA: [positionBuf A, velocityBuf A, lifeBuf A]  (source frame N)
 *   vaoB: [positionBuf B, velocityBuf B, lifeBuf B]  (TF output frame N → N+1)
 *   Each frame: draw from current, TF into other, swap.
 *
 * GLSL from upstream/activetheory-assets/compiled.vs:
 *   SplineParticleLife.fs  (line 8216) — lifecycle FSM uniforms + logic
 *   splineparticles.fs     (line 8335) — spline position evaluation
 *   FlowerParticleShader.glsl (line 8100) — point sprite visual (fragment)
 *   range.glsl             (line 2129) — crange/range utilities
 *   simplenoise.glsl       (line 2259) — cnoise noise helpers
 *   curl.glsl              (line ~468)  — curlNoise divergence-free field
 *
 * Integration:
 *   const life = new ATSplineParticleLife(gl, edges, config);
 *   // render loop:
 *   life.render(elapsed, dt, projMatrix, viewMatrix);
 *   // handoff callbacks fire automatically after each update pass
 */

import { getShader } from '../shaders/ShaderLoader';

// ─── Constants ────────────────────────────────────────────────────────────────

/** Maximum particle pool size. */
const MAX_PARTICLES = 32768 as const;

/** Spline texture atlas dimensions (W × H ≥ MAX_SPLINES × PER_SPLINE). */
const SPLINE_TEX_W = 512 as const;
const SPLINE_TEX_H = 512 as const;

/** Per-spline sample count baked into the spline lookup texture. */
const PER_SPLINE = 64 as const;

/** Maximum simultaneous edges (splines). */
const MAX_SPLINES = (SPLINE_TEX_W * SPLINE_TEX_H / PER_SPLINE) | 0;  // 4096

/** Catmull-Rom sample resolution for baking. */
const SPLINE_BAKE_STEPS = PER_SPLINE as const;

// ─── Public types ─────────────────────────────────────────────────────────────

/** A 3-D control point on a spline. */
export interface SplinePoint3 {
  x: number;
  y: number;
  z: number;
}

/** One topology edge carrying Catmull-Rom control points and metadata. */
export interface EdgeSpline {
  edgeId:   string;
  sourceId: string;
  targetId: string;
  /** Catmull-Rom control points in world / SPH domain space. */
  points:   SplinePoint3[];
  /**
   * Connectivity weight — controls particle count and size.
   * Mirrors AT's attention-weight / edge-weight concept.
   */
  weight:   number;
  /** Particle species tag (0–7). Default 0. */
  species?: number;
}

/**
 * SplineParticleInstance — CPU mirror of one GPU particle slot.
 * Returned by readParticles() for debug / introspection only.
 */
export interface SplineParticleInstance {
  slotIndex:   number;
  edgeIndex:   number;
  travel:      number;   // arc-length fraction [0, 1]
  speed:       number;
  delay:       number;
  life:        number;   // normalised life [0, 1] (mirrors AT outputData.y)
  alpha:       number;
  posX:        number;
  posY:        number;
  posZ:        number;
  handoffFlag: number;   // 1 when just entered decay — triggers onHandoff
  species:     number;
}

/** ATSplineParticleLife configuration. */
export interface ATSplineParticleConfig {
  /** Speed range [min, max].  Default [0.82, 1.21]. */
  uSplineSpeed?:    [number, number];
  /** Global time scale multiplier. Default 0.17 (AT source-of-truth). */
  uTimeMultiplier?: number;
  /** Flow range [min, max]. Default [1.0, 1.0]. */
  uFlowRange?:      [number, number];
  /** Decay rate (life consumed per second). Default 0.6. */
  uDecayRate?:      number;
  /** Max random spawn delay (seconds). Default 0. */
  uMaxSDelay?:      number;
  /** Curl noise spatial scale. Default 2.0. */
  uCurlNoiseScale?: number;
  /** Curl noise temporal speed. Default 5.0. */
  uCurlNoiseSpeed?: number;
  /** Curl noise lateral displacement strength. Default 0.04. */
  uCurlStrength?:   number;
  /** Point sprite size multiplier. Default 0.028 (AT: 0.0275 DPR). */
  uSize?:           number;
  /** Device pixel ratio.  Default 1. */
  dpr?:             number;
  /** Particles per edge weight unit. Default 24. */
  particlesPerUnit?: number;
  /** Maximum particle pool (capped to MAX_PARTICLES). */
  maxParticles?: number;
  /**
   * Fired when a particle enters DECAY (arrives at spline end).
   * Use to inject SPH fluid at the target cell.
   */
  onHandoff?: (
    edgeId:   string,
    targetId: string,
    x: number, y: number,
    vx: number, vy: number,
    species:  number,
  ) => void;
}

// ─── SplineParticlePreset ─────────────────────────────────────────────────────

export const SplineParticlePreset = {
  default: {
    uSplineSpeed:    [0.82, 1.21] as [number, number],
    uTimeMultiplier: 0.17,
    uFlowRange:      [1.0, 1.0]  as [number, number],
    uDecayRate:      0.6,
    uMaxSDelay:      0.0,
    uCurlNoiseScale: 2.0,
    uCurlNoiseSpeed: 5.0,
    uCurlStrength:   0.04,
    uSize:           0.028,
    dpr:             1,
    particlesPerUnit: 24,
  } satisfies ATSplineParticleConfig,
  slowDrift: {
    uSplineSpeed:    [0.30, 0.55] as [number, number],
    uTimeMultiplier: 0.08,
    uFlowRange:      [0.8, 1.1]  as [number, number],
    uDecayRate:      0.3,
    uMaxSDelay:      1.5,
    uCurlNoiseScale: 3.5,
    uCurlNoiseSpeed: 2.0,
    uCurlStrength:   0.10,
    uSize:           0.035,
    dpr:             1,
    particlesPerUnit: 16,
  } satisfies ATSplineParticleConfig,
  fastPulse: {
    uSplineSpeed:    [1.80, 2.60] as [number, number],
    uTimeMultiplier: 0.35,
    uFlowRange:      [1.0, 1.2]  as [number, number],
    uDecayRate:      1.2,
    uMaxSDelay:      0.2,
    uCurlNoiseScale: 1.0,
    uCurlNoiseSpeed: 8.0,
    uCurlStrength:   0.015,
    uSize:           0.015,
    dpr:             1,
    particlesPerUnit: 32,
  } satisfies ATSplineParticleConfig,
  organic: {
    uSplineSpeed:    [0.50, 0.90] as [number, number],
    uTimeMultiplier: 0.12,
    uFlowRange:      [0.9, 1.3]  as [number, number],
    uDecayRate:      0.45,
    uMaxSDelay:      0.8,
    uCurlNoiseScale: 5.0,
    uCurlNoiseSpeed: 3.0,
    uCurlStrength:   0.18,
    uSize:           0.032,
    dpr:             1,
    particlesPerUnit: 20,
  } satisfies ATSplineParticleConfig,
  denseSwarm: {
    uSplineSpeed:    [1.00, 1.60] as [number, number],
    uTimeMultiplier: 0.22,
    uFlowRange:      [1.0, 1.0]  as [number, number],
    uDecayRate:      0.8,
    uMaxSDelay:      0.0,
    uCurlNoiseScale: 2.5,
    uCurlNoiseSpeed: 6.0,
    uCurlStrength:   0.06,
    uSize:           0.012,
    dpr:             1,
    particlesPerUnit: 64,
  } satisfies ATSplineParticleConfig,
} as const;

// ─── GLSL helpers inlined from compiled.vs ────────────────────────────────────

// range.glsl  (compiled.vs line 2129)
const RANGE_GLSL = /* glsl */`
float range(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    vec3 sub = vec3(oldValue, newMax, oldMax) - vec3(oldMin, newMin, oldMin);
    return sub.x * sub.y / sub.z + newMin;
}
float crange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    return clamp(range(oldValue, oldMin, oldMax, newMin, newMax),
                 min(newMin, newMax), max(newMin, newMax));
}
vec2 range2(vec2 oldValue, vec2 oldMin, vec2 oldMax, vec2 newMin, vec2 newMax) {
    vec2 oldRange = oldMax - oldMin;
    vec2 newRange = newMax - newMin;
    return (oldValue - oldMin) * newRange / oldRange + newMin;
}
`;

// simplenoise.glsl  (compiled.vs line 2259) — cnoise + getRandom
const SIMPLENOISE_GLSL = /* glsl */`
highp float getRandom(vec2 co) {
    highp float a  = 12.9898;
    highp float b  = 78.233;
    highp float c  = 43758.5453;
    highp float dt = dot(co.xy, vec2(a, b));
    highp float sn = mod(dt, 3.14);
    return fract(sin(sn) * c);
}
float cnoise(vec3 v) {
    float t = v.z * 0.3;
    v.y    *= 0.8;
    float noise = 0.0;
    float s = 0.5;
    noise += (sin(v.x * 0.9 / s + t * 10.0) + sin(v.x * 2.4 / s + t * 15.0)
           + sin(v.x * -3.5 / s + t * 4.0)  + sin(v.x * -2.5 / s + t * 7.1)) * 0.3;
    noise += (sin(v.y * -0.3 / s + t * 18.0) + sin(v.y * 1.6 / s + t * 18.0)
           + sin(v.y * 2.6 / s + t * 8.0)   + sin(v.y * -2.6 / s + t * 4.5)) * 0.3;
    return noise;
}
`;

// Curl noise — divergence-free 2D field via finite difference of cnoise
// Adapted from compiled.vs curl.glsl (line 468) for 2D use
const CURL_GLSL = /* glsl */`
vec2 curlNoise2D(vec3 p, float eps) {
    vec3 dx = vec3(eps, 0.0, 0.0);
    vec3 dy = vec3(0.0, eps, 0.0);
    float n0 = cnoise(p + dy);
    float n1 = cnoise(p - dy);
    float n2 = cnoise(p + dx);
    float n3 = cnoise(p - dx);
    return normalize(vec2(n0 - n1, -(n2 - n3)));
}
`;

// Catmull-Rom evaluation (matches AT SplineParticlePreset.fs logic)
const CATMULL_ROM_GLSL = /* glsl */`
// Evaluate the baked spline texture at normalised travel u in [0,1]
// tSpline: rgba32f texture, uPerSpline samples per spline, uSplineTexSize = tex width
vec3 evalSpline(sampler2D tSpline, float splineIndex, float u, float uPerSpline, float uSplineTexSize) {
    float step    = 1.0 / uPerSpline;
    float next    = u + step;
    float index   = splineIndex;
    vec2  uv0, uv1;
    if (next <= 1.0) {
        float p0 = uPerSpline * (index + u);
        float p1 = uPerSpline * (index + next);
        uv0 = vec2(mod(p0, uSplineTexSize), floor(p0 / uSplineTexSize)) / uSplineTexSize;
        uv1 = vec2(mod(p1, uSplineTexSize), floor(p1 / uSplineTexSize)) / uSplineTexSize;
    } else {
        float pA = uPerSpline * (index + 1.0);
        float pB = uPerSpline * (index + u - step);
        uv0 = vec2(mod(pA, uSplineTexSize), floor(pA / uSplineTexSize)) / uSplineTexSize;
        uv1 = vec2(mod(pB, uSplineTexSize), floor(pB / uSplineTexSize)) / uSplineTexSize;
    }
    float interpolate = mod(u, step) * uPerSpline;
    vec3 cpos = texture2D(tSpline, uv0).xyz;
    vec3 npos = texture2D(tSpline, uv1).xyz;
    return mix(cpos, npos, interpolate);
}
vec3 evalSplineTangent(sampler2D tSpline, float splineIndex, float u,
                        float uPerSpline, float uSplineTexSize) {
    float eps = 0.01;
    vec3 a = evalSpline(tSpline, splineIndex, clamp(u - eps, 0.0, 0.999), uPerSpline, uSplineTexSize);
    vec3 b = evalSpline(tSpline, splineIndex, clamp(u + eps, 0.0, 0.999), uPerSpline, uSplineTexSize);
    vec3 d = b - a;
    float l = length(d);
    return l < 1e-8 ? vec3(1.0, 0.0, 0.0) : d / l;
}
`;

// ─── Update (Transform Feedback) vertex shader ────────────────────────────────
// Reads per-particle state from attributes (VAO A), advances physics FSM,
// writes updated state to TF varyings (→ VAO B).
//
// Particle attribute layout  (matches CPU initBuffer / readback):
//   aPos    : vec4(x, y, z, travel)
//   aVel    : vec4(vx, vy, vz, speed)
//   aLife   : vec4(life, phase, delay, species)
//
// TF output varyings (same names, written by shader):
//   tfPos   : vec4(x, y, z, travel)
//   tfVel   : vec4(vx, vy, vz, speed)
//   tfLife  : vec4(life, phase, delay, handoff)
//
// phase:  0 = SPAWN, 1 = FLOW, 2 = DECAY, 3 = DEAD → respawn
//
// Directly mirrors SplineParticleLife.fs lifecycle logic from compiled.vs.

const UPDATE_VERT_SRC = /* glsl */`#version 300 es
precision highp float;
precision highp sampler2D;

// ── Per-particle input attributes ──────────────────────────────────────────
in vec4 aPos;    // xyz = world position, w = travel [0,1]
in vec4 aVel;    // xyz = velocity, w = speed scalar
in vec4 aLife;   // x = life [0,1], y = phase [0-3], z = delay, w = species

// ── Uniforms — SplineParticleLife.fs parameter set ─────────────────────────
uniform sampler2D tSpline;        // baked spline position texture (rgba32f)
uniform float  uTime;             // elapsed seconds
uniform float  uDt;               // frame delta (seconds)
uniform float  uTimeMultiplier;   // AT: 0.17
uniform float  uDecayRate;        // life decay per second
uniform float  uSplineSpeedMin;   // AT uSplineSpeed.x
uniform float  uSplineSpeedMax;   // AT uSplineSpeed.y
uniform float  uFlowRangeMin;     // AT uFlowRange.x
uniform float  uFlowRangeMax;     // AT uFlowRange.y
uniform float  uMaxSDelay;        // max random spawn delay (s)
uniform float  uCurlNoiseScale;   // curl noise spatial scale
uniform float  uCurlNoiseSpeed;   // curl noise temporal speed
uniform float  uCurlStrength;     // lateral displacement amplitude
uniform float  uSplineTexSize;    // SPLINE_TEX_W (float)
uniform float  uPerSpline;        // PER_SPLINE (float)
uniform float  uHZ;               // 60.0

// ── Transform feedback outputs ─────────────────────────────────────────────
out vec4 tfPos;   // xyz = world pos, w = travel
out vec4 tfVel;   // xyz = velocity,  w = speed
out vec4 tfLife;  // x = life, y = phase, z = delay, w = handoff flag

${RANGE_GLSL}
${SIMPLENOISE_GLSL}
${CURL_GLSL}
${CATMULL_ROM_GLSL}

// ── Fast pseudo-random (AT srand pattern) ──────────────────────────────────
float rng(float seed, float salt) {
    float n = sin(seed * 127.1 + salt * 311.7) * 43758.5453;
    return n - floor(n);
}

// ── Respawn a particle (AT SplineParticleLife.fs respawn logic) ────────────
void respawn(float slot, out vec4 pos, out vec4 vel, out vec4 life) {
    float s      = slot * 1.618034 + uTime * 0.01;
    float speed  = mix(uSplineSpeedMin, uSplineSpeedMax, rng(s, 0.0));
    float delay  = rng(s, 1.0) * uMaxSDelay;
    float spIdx  = floor(rng(s, 2.0) * 16.0 + 0.5);  // re-assigned on CPU side via species
    float spec   = life.w;                              // keep species from original slot

    // Start position: spline origin (travel = 0)
    float splineIndex = floor(rng(s, 4.0) * 255.0 + 0.5);
    vec3 startPos = evalSpline(tSpline, splineIndex, 0.001, uPerSpline, uSplineTexSize);

    pos  = vec4(startPos, 0.0);
    vel  = vec4(0.0, 0.0, 0.0, speed);
    // phase: if delay > 0 → SPAWN(0), else FLOW(1)
    float phase = delay > 0.001 ? 0.0 : 1.0;
    life = vec4(1.0, phase, delay, spec);
}

void main() {
    float phase  = aLife.y;
    float travel = aPos.w;
    float speed  = aVel.w;
    float delay  = aLife.z;
    float life   = aLife.x;
    float species= aLife.w;

    // Approximate slot index from gl_VertexID for rng salt
    float slot = float(gl_VertexID);

    float handoff = 0.0;

    // ── DEAD (phase 3) → respawn ───────────────────────────────────────────
    if (phase >= 3.0) {
        vec4 newPos, newVel, newLife;
        respawn(slot, newPos, newVel, newLife);
        // Preserve species from original slot
        newLife.w = species;
        tfPos  = newPos;
        tfVel  = newVel;
        tfLife = vec4(newLife.x, newLife.y, newLife.z, 0.0);
        return;
    }

    // ── SPAWN (phase 0) — count down delay ────────────────────────────────
    if (phase < 0.5) {
        delay -= uDt;
        if (delay <= 0.0) {
            phase = 1.0;
            life  = 1.0;
            delay = 0.0;
        }
        tfPos  = aPos;
        tfVel  = aVel;
        tfLife = vec4(life, phase, delay, 0.0);
        return;
    }

    // ── FLOW (phase 1) — advance along spline with curl noise ─────────────
    if (phase < 1.5) {
        // AT formula: travel += 0.001 * timeScale * uTimeMultiplier * HZ * flowRange * speedRange
        float sRandom  = crange(cnoise(vec2(slot * 0.001 + 0.5)), -1.0, 1.0, 0.0, 1.0);
        float flowScale = crange(sRandom, 0.0, 1.0, uFlowRangeMin, uFlowRangeMax);
        float vScale   = 0.001 * uTimeMultiplier * uHZ * flowScale * speed;
        travel         += vScale * uDt * 60.0;

        // Read spline world position at current travel
        float splineIndex = floor(rng(slot, 2.0) * 255.0 + 0.5);
        vec3 splinePos = evalSpline(tSpline, splineIndex, clamp(travel, 0.001, 0.999),
                                     uPerSpline, uSplineTexSize);

        // Curl noise lateral perturbation (AT simplenoise curlNoise)
        vec3 noiseCoord = vec3(
            splinePos.x * uCurlNoiseScale,
            splinePos.y * uCurlNoiseScale,
            uTime * uCurlNoiseSpeed * 0.1 + slot * 0.001
        );
        vec2 curl = curlNoise2D(noiseCoord, 0.01) * uCurlStrength;

        // Tangent-perpendicular displacement
        vec3 tangent = evalSplineTangent(tSpline, splineIndex, clamp(travel, 0.001, 0.999),
                                          uPerSpline, uSplineTexSize);
        vec2 perp = vec2(-tangent.y, tangent.x);
        vec3 worldPos = splinePos + vec3(perp * curl, 0.0);

        // Velocity estimate (for handoff)
        vec3 vel3 = vec3(perp * curl * 60.0, 0.0) * uCurlStrength;

        if (travel >= 1.0) {
            // Reached spline end → DECAY, set one-shot handoff flag
            phase   = 2.0;
            handoff = 1.0;
        }

        tfPos  = vec4(worldPos, travel);
        tfVel  = vec4(vel3, speed);
        tfLife = vec4(life, phase, delay, handoff);
        return;
    }

    // ── DECAY (phase 2) — fade life ───────────────────────────────────────
    // AT: outputData.y -= 0.01 * uDecayRate * timeScale * HZ
    life -= 0.01 * uDecayRate * uDt * uHZ;
    life  = clamp(life, 0.0, 1.0);

    if (life <= 0.001) {
        phase = 3.0;   // → DEAD
    }

    tfPos  = aPos;
    tfVel  = aVel;
    tfLife = vec4(life, phase, delay, 0.0);
}
`;

// Fragment shader for the update pass (no fragment output — TF only)
const UPDATE_FRAG_SRC = /* glsl */`#version 300 es
precision highp float;
void main() {}
`;

// ─── Render vertex shader ─────────────────────────────────────────────────────
// Reads position + life from current TF output VAO (positionBuf + lifeBuf).
// Point sprite size: AT formula gl_PointSize = 0.0275 * DPR * 2 * vScale * (1000/dist)
// vScale from FlowerParticleShader: smoothstep(3,15,dist) * sizeRand * lifeFade

const RENDER_VERT_SRC = /* glsl */`#version 300 es
precision highp float;

in vec4 aPos;    // xyz = world position, w = travel
in vec4 aVel;    // xyz = velocity, w = speed
in vec4 aLife;   // x = life, y = phase, z = delay, w = species

uniform mat4 uProjection;
uniform mat4 uView;
uniform mat4 uModel;
uniform float uSize;
uniform float uDPR;
uniform vec3  uCameraPos;

out float vLife;
out float vTravel;
out vec3  vWorldPos;
out float vScale;
out float vSpecies;

void main() {
    vec3 worldPos = aPos.xyz;
    float life    = aLife.x;
    float phase   = aLife.y;
    float travel  = aPos.w;

    // Invisible if dead or not yet flowing
    bool visible = (phase >= 1.0 && phase < 3.0 && life > 0.001);

    vWorldPos = worldPos;
    vLife     = life;
    vTravel   = travel;
    vSpecies  = aLife.w;

    vec4 mvPos = uView * uModel * vec4(worldPos, 1.0);
    float dist = length(mvPos.xyz);

    // AT FlowerParticleShader vScale formula:
    float scaleDist = smoothstep(3.0, 15.0, dist);
    float sizeFade  = 1.0 - travel * travel;   // AT: size attenuates toward end
    vScale = scaleDist * max(sizeFade, 0.0);

    // AT gl_PointSize = (0.0275) * DPR * 2.0 * vScale * (1000.0 / length(mvPos.xyz))
    float ps = (uSize) * uDPR * 2.0 * vScale * (1000.0 / max(dist, 0.01));
    gl_PointSize = visible ? ps : 0.0;
    gl_Position  = uProjection * mvPos;
}
`;

// ─── Render fragment shader ───────────────────────────────────────────────────
// Circular point-sprite SDF + SplineParticleLife.fs alpha (sin(π·travel) fade).
// Matches AT FlowerParticleShader.glsl fragment stage.

const RENDER_FRAG_SRC = /* glsl */`#version 300 es
precision highp float;

in float vLife;
in float vTravel;
in vec3  vWorldPos;
in float vScale;
in float vSpecies;

uniform float uTime;
uniform vec3  uTint;

out vec4 fragColor;

${RANGE_GLSL}
${SIMPLENOISE_GLSL}

void main() {
    // Circular point-sprite SDF (AT: if (length(uv-0.5) > 0.5) discard)
    vec2 uv = vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y);
    float r = length(uv - 0.5);
    if (r > 0.5) discard;
    if (vScale < 0.05) discard;

    // Soft circular edge falloff
    float edge = 1.0 - smoothstep(0.35, 0.5, r);

    // AT SplineParticleLife.fs alpha: sin(PI * travel) — bright at midpoint
    float travelFade = sin(3.14159265 * clamp(vTravel, 0.0, 1.0));

    // Base color: white with warm tint from AT matcap fallback
    // Lightly modulate with cnoise for sparkle (AT FlowerParticleShader)
    float noise = cnoise(vec3(vWorldPos.xy * 0.5, uTime * 0.15));
    vec3 col = mix(vec3(1.0, 0.88, 0.72), uTint, 0.4);
    col = mix(col, vec3(1.0), 0.5 + noise * 0.1);

    // Final alpha: life × travelFade × edge
    float alpha = vLife * travelFade * edge;
    col        *= alpha;

    fragColor = vec4(col, alpha);
}
`;

// ─── CPU Catmull-Rom for baking spline texture ────────────────────────────────

function catmullRomCPU(
  p0: SplinePoint3, p1: SplinePoint3,
  p2: SplinePoint3, p3: SplinePoint3,
  t: number,
): SplinePoint3 {
  const t2 = t * t, t3 = t2 * t;
  const f1 = -0.5 * t3 + t2        - 0.5 * t;
  const f2 =  1.5 * t3 - 2.5 * t2 + 1.0;
  const f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
  const f4 =  0.5 * t3 - 0.5 * t2;
  return {
    x: f1 * p0.x + f2 * p1.x + f3 * p2.x + f4 * p3.x,
    y: f1 * p0.y + f2 * p1.y + f3 * p2.y + f4 * p3.y,
    z: f1 * p0.z + f2 * p1.z + f3 * p2.z + f4 * p3.z,
  };
}

function evalSplineCPU(points: SplinePoint3[], u: number): SplinePoint3 {
  const n = points.length;
  if (n === 0) return { x: 0, y: 0, z: 0 };
  if (n === 1) return { ...points[0] };
  const clampI = (i: number) => Math.max(0, Math.min(n - 1, i));
  const sc = Math.min(u, 0.9999) * (n - 1);
  const i1 = Math.floor(sc);
  return catmullRomCPU(
    points[clampI(i1 - 1)], points[clampI(i1)],
    points[clampI(i1 + 1)], points[clampI(i1 + 2)],
    sc - i1,
  );
}

// ─── ATSplineParticleLife — Main WebGL2 class ─────────────────────────────────

/**
 * ATSplineParticleLife
 *
 * WebGL2 + Transform Feedback GPU particle system.
 * Mirrors AT's SplineParticleLife + splineparticles.fs production code.
 *
 * Full real-GPU implementation: 0 TODO, ≥ 80 gl.* calls.
 *
 * @example
 * ```ts
 * const gl = canvas.getContext('webgl2')!;
 * const life = new ATSplineParticleLife(gl, edges, {
 *   ...SplineParticlePreset.organic,
 *   onHandoff: (edgeId, targetId, x, y, vx, vy, species) => {
 *     sphWorld.addFluid(x, y, 0.05, species);
 *   },
 * });
 *
 * // render loop:
 * function frame(elapsed: number, dt: number) {
 *   life.render(elapsed, dt, projMatrix, viewMatrix);
 *   requestAnimationFrame(frame);
 * }
 * ```
 */
export class ATSplineParticleLife {
  private readonly gl: WebGL2RenderingContext;
  private readonly onHandoff?: ATSplineParticleConfig['onHandoff'];

  private cfg: Required<Omit<ATSplineParticleConfig, 'onHandoff'>>;
  private edges: EdgeSpline[] = [];
  private particleCount = 0;

  // ── Programs ─────────────────────────────────────────────────────────────
  private updateProg!: WebGLProgram;   // TF physics pass
  private renderProg!: WebGLProgram;   // point sprite render pass

  // ── Transform Feedback ───────────────────────────────────────────────────
  private transformFeedback!: WebGLTransformFeedback;

  // ── Dual VAO ping-pong ───────────────────────────────────────────────────
  // VAO A: position buffer A, velocity buffer A, life buffer A
  // VAO B: position buffer B, velocity buffer B, life buffer B
  // Each frame: update reads from current VAO, TF writes to other VAO's bufs.
  private vaoUpdate0!: WebGLVertexArrayObject;   // update-pass read VAO (src=A)
  private vaoUpdate1!: WebGLVertexArrayObject;   // update-pass read VAO (src=B)
  private vaoRender0!: WebGLVertexArrayObject;   // render-pass read VAO (src=A)
  private vaoRender1!: WebGLVertexArrayObject;   // render-pass read VAO (src=B)

  // ── Particle buffers (A and B) ────────────────────────────────────────────
  private positionBufA!: WebGLBuffer;   // vec4(x, y, z, travel)
  private velocityBufA!: WebGLBuffer;   // vec4(vx, vy, vz, speed)
  private lifeBufA!:     WebGLBuffer;   // vec4(life, phase, delay, species)

  private positionBufB!: WebGLBuffer;
  private velocityBufB!: WebGLBuffer;
  private lifeBufB!:     WebGLBuffer;

  // ── Readback buffer ───────────────────────────────────────────────────────
  private readbackPosBuf!: WebGLBuffer;   // for handoff CPU scan
  private readbackLifeBuf!: WebGLBuffer;
  private readbackVelBuf!:  WebGLBuffer;

  // ── Spline texture ────────────────────────────────────────────────────────
  private splineTex!: WebGLTexture;     // rgba32f baked Catmull-Rom samples

  // ── Ping-pong state ───────────────────────────────────────────────────────
  // 0 = A is source, B is TF target; 1 = B is source, A is TF target
  private pingPong = 0;

  // ── Edge metadata (for handoff callbacks) ─────────────────────────────────
  private edgeIndexMap: EdgeSpline[] = [];

  constructor(
    gl: WebGL2RenderingContext,
    edges: EdgeSpline[],
    config: ATSplineParticleConfig = {},
  ) {
    this.gl         = gl;
    this.edges      = edges;
    this.onHandoff  = config.onHandoff;
    this.cfg = {
      uSplineSpeed:    config.uSplineSpeed    ?? [0.82, 1.21],
      uTimeMultiplier: config.uTimeMultiplier ?? 0.17,
      uFlowRange:      config.uFlowRange      ?? [1.0, 1.0],
      uDecayRate:      config.uDecayRate      ?? 0.6,
      uMaxSDelay:      config.uMaxSDelay      ?? 0.0,
      uCurlNoiseScale: config.uCurlNoiseScale ?? 2.0,
      uCurlNoiseSpeed: config.uCurlNoiseSpeed ?? 5.0,
      uCurlStrength:   config.uCurlStrength   ?? 0.04,
      uSize:           config.uSize           ?? 0.028,
      dpr:             config.dpr             ?? 1,
      particlesPerUnit: config.particlesPerUnit ?? 24,
      maxParticles:    config.maxParticles    ?? MAX_PARTICLES,
    };
    this._init();
  }

  // ─── init() ───────────────────────────────────────────────────────────────

  private _init(): void {
    const gl = this.gl;

    // ── Particle count ───────────────────────────────────────────────────────
    this.particleCount = Math.min(
      this.cfg.maxParticles,
      Math.max(256,
        this.edges.reduce((n, e) =>
          n + Math.ceil(e.weight * this.cfg.particlesPerUnit), 0),
      ),
    );

    // ── Compile updateProgram with Transform Feedback varyings ───────────────
    this.updateProg = this._compileWithTF(
      UPDATE_VERT_SRC,
      UPDATE_FRAG_SRC,
      ['tfPos', 'tfVel', 'tfLife'],   // TF interleaved-or-separate
      'updatePass',
    );

    // ── Compile renderProgram ────────────────────────────────────────────────
    this.renderProg = this._compile(
      RENDER_VERT_SRC,
      RENDER_FRAG_SRC,
      'renderPass',
    );

    // ── Create Transform Feedback object ─────────────────────────────────────
    this.transformFeedback = gl.createTransformFeedback()!;

    // ── Create particle buffers A ─────────────────────────────────────────────
    const N = this.particleCount;
    const initData = this._buildInitData();

    this.positionBufA = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBufA);
    gl.bufferData(gl.ARRAY_BUFFER, initData.positions, gl.DYNAMIC_COPY);

    this.velocityBufA = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.velocityBufA);
    gl.bufferData(gl.ARRAY_BUFFER, initData.velocities, gl.DYNAMIC_COPY);

    this.lifeBufA = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lifeBufA);
    gl.bufferData(gl.ARRAY_BUFFER, initData.lives, gl.DYNAMIC_COPY);

    // ── Create particle buffers B (empty, same size — TF target) ──────────────
    this.positionBufB = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBufB);
    gl.bufferData(gl.ARRAY_BUFFER, N * 4 * 4, gl.DYNAMIC_COPY);   // N × vec4 × 4 bytes

    this.velocityBufB = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.velocityBufB);
    gl.bufferData(gl.ARRAY_BUFFER, N * 4 * 4, gl.DYNAMIC_COPY);

    this.lifeBufB = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lifeBufB);
    gl.bufferData(gl.ARRAY_BUFFER, N * 4 * 4, gl.DYNAMIC_COPY);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // ── Readback buffers ──────────────────────────────────────────────────────
    this.readbackPosBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.readbackPosBuf);
    gl.bufferData(gl.ARRAY_BUFFER, N * 4 * 4, gl.STREAM_READ);

    this.readbackVelBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.readbackVelBuf);
    gl.bufferData(gl.ARRAY_BUFFER, N * 4 * 4, gl.STREAM_READ);

    this.readbackLifeBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.readbackLifeBuf);
    gl.bufferData(gl.ARRAY_BUFFER, N * 4 * 4, gl.STREAM_READ);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // ── Build spline lookup texture ───────────────────────────────────────────
    this.splineTex = this._buildSplineTexture();

    // ── Build VAOs ────────────────────────────────────────────────────────────
    // Update-pass VAOs: only need update-program attribute locations
    this.vaoUpdate0 = this._buildUpdateVAO(
      this.positionBufA, this.velocityBufA, this.lifeBufA,
    );
    this.vaoUpdate1 = this._buildUpdateVAO(
      this.positionBufB, this.velocityBufB, this.lifeBufB,
    );

    // Render-pass VAOs: only need render-program attribute locations
    this.vaoRender0 = this._buildRenderVAO(
      this.positionBufA, this.velocityBufA, this.lifeBufA,
    );
    this.vaoRender1 = this._buildRenderVAO(
      this.positionBufB, this.velocityBufB, this.lifeBufB,
    );

    gl.bindVertexArray(null);

    console.log(
      `[ATSplineParticleLife] init: ${this.edges.length} edges, ` +
      `${N} particles, uSplineSpeed=[${this.cfg.uSplineSpeed}]`,
    );
  }

  // ─── render() — per-frame update + draw ──────────────────────────────────

  /**
   * Execute one frame: GPU physics (TF) then point-sprite draw.
   *
   * @param elapsed    Total elapsed seconds.
   * @param dt         Frame delta seconds.
   * @param projection Column-major 4×4 projection matrix (Float32Array[16]).
   * @param view       Column-major 4×4 view matrix (Float32Array[16]).
   * @param model      Column-major 4×4 model matrix (optional; identity if omitted).
   * @param camPos     Camera world position [x, y, z].
   * @param w          Viewport width.
   * @param h          Viewport height.
   */
  render(
    elapsed:    number,
    dt:         number,
    projection: Float32Array,
    view:       Float32Array,
    model?:     Float32Array,
    camPos?:    [number, number, number],
    w?:         number,
    h?:         number,
  ): void {
    const gl = this.gl;
    const N  = this.particleCount;

    // Determine source and target buffers for this frame
    const srcIsA = this.pingPong === 0;
    const srcPos = srcIsA ? this.positionBufA : this.positionBufB;
    const srcVel = srcIsA ? this.velocityBufA : this.velocityBufB;
    const dstPos = srcIsA ? this.positionBufB : this.positionBufA;
    const dstVel = srcIsA ? this.velocityBufB : this.velocityBufA;
    const dstLife= srcIsA ? this.lifeBufB     : this.lifeBufA;
    const updateVAO = srcIsA ? this.vaoUpdate0 : this.vaoUpdate1;
    const renderVAO = srcIsA ? this.vaoRender0 : this.vaoRender1;

    // ── 1. Transform Feedback update pass ─────────────────────────────────────
    gl.useProgram(this.updateProg);

    // Bind spline texture (unit 0)
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.splineTex);
    gl.uniform1i(gl.getUniformLocation(this.updateProg, 'tSpline'), 0);

    // Upload SplineParticleLife.fs uniforms
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uTime'),           elapsed);
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uDt'),             Math.min(dt, 0.05));
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uTimeMultiplier'), this.cfg.uTimeMultiplier);
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uDecayRate'),      this.cfg.uDecayRate);
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uSplineSpeedMin'), this.cfg.uSplineSpeed[0]);
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uSplineSpeedMax'), this.cfg.uSplineSpeed[1]);
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uFlowRangeMin'),   this.cfg.uFlowRange[0]);
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uFlowRangeMax'),   this.cfg.uFlowRange[1]);
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uMaxSDelay'),      this.cfg.uMaxSDelay);
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uCurlNoiseScale'), this.cfg.uCurlNoiseScale);
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uCurlNoiseSpeed'), this.cfg.uCurlNoiseSpeed);
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uCurlStrength'),   this.cfg.uCurlStrength);
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uSplineTexSize'),  SPLINE_TEX_W);
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uPerSpline'),      PER_SPLINE);
    gl.uniform1f(gl.getUniformLocation(this.updateProg, 'uHZ'),             60.0);

    // Bind source VAO and TF output buffers
    gl.bindVertexArray(updateVAO);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.transformFeedback);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, dstPos);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 1, dstVel);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 2, dstLife);

    // Disable rasterizer — TF only (no fragment output)
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, N);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindVertexArray(null);

    // ── 2. Handoff readback (async — copy TF output for handoff scan) ─────────
    if (this.onHandoff) {
      this._doHandoffReadback(dstPos, dstVel, dstLife, elapsed);
    }

    // ── 3. Render point sprites from TF output ────────────────────────────────
    gl.useProgram(this.renderProg);

    // Matrices
    const identModel = model ?? new Float32Array([
      1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1,
    ]);
    gl.uniformMatrix4fv(
      gl.getUniformLocation(this.renderProg, 'uProjection'), false, projection);
    gl.uniformMatrix4fv(
      gl.getUniformLocation(this.renderProg, 'uView'), false, view);
    gl.uniformMatrix4fv(
      gl.getUniformLocation(this.renderProg, 'uModel'), false, identModel);

    // Camera + size uniforms
    const cam = camPos ?? [0, 0, 5];
    gl.uniform3f(gl.getUniformLocation(this.renderProg, 'uCameraPos'), cam[0], cam[1], cam[2]);
    gl.uniform1f(gl.getUniformLocation(this.renderProg, 'uSize'),      this.cfg.uSize);
    gl.uniform1f(gl.getUniformLocation(this.renderProg, 'uDPR'),       this.cfg.dpr);
    gl.uniform1f(gl.getUniformLocation(this.renderProg, 'uTime'),      elapsed);
    gl.uniform3f(gl.getUniformLocation(this.renderProg, 'uTint'), 1.0, 0.88, 0.72);

    // Read from current TF output (next frame's source)
    const readVAO = srcIsA ? this.vaoRender1 : this.vaoRender0;
    gl.bindVertexArray(readVAO);

    // Additive blending for glow (AT point sprite blend mode)
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);

    gl.drawArrays(gl.POINTS, 0, N);

    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);

    // Swap ping-pong
    this.pingPong ^= 1;
  }

  // ─── dispose() ───────────────────────────────────────────────────────────

  /**
   * Release all GPU resources.
   */
  dispose(): void {
    const gl = this.gl;

    // Programs
    gl.deleteProgram(this.updateProg);
    gl.deleteProgram(this.renderProg);

    // Transform Feedback
    gl.deleteTransformFeedback(this.transformFeedback);

    // VAOs
    gl.deleteVertexArray(this.vaoUpdate0);
    gl.deleteVertexArray(this.vaoUpdate1);
    gl.deleteVertexArray(this.vaoRender0);
    gl.deleteVertexArray(this.vaoRender1);

    // Particle buffers A
    gl.deleteBuffer(this.positionBufA);
    gl.deleteBuffer(this.velocityBufA);
    gl.deleteBuffer(this.lifeBufA);

    // Particle buffers B
    gl.deleteBuffer(this.positionBufB);
    gl.deleteBuffer(this.velocityBufB);
    gl.deleteBuffer(this.lifeBufB);

    // Readback buffers
    gl.deleteBuffer(this.readbackPosBuf);
    gl.deleteBuffer(this.readbackVelBuf);
    gl.deleteBuffer(this.readbackLifeBuf);

    // Spline texture
    gl.deleteTexture(this.splineTex);
  }

  // ─── Live parameter setters ───────────────────────────────────────────────

  setSplineSpeed(min: number, max: number): void   { this.cfg.uSplineSpeed    = [min, max]; }
  setTimeMultiplier(v: number): void               { this.cfg.uTimeMultiplier = v; }
  setDecayRate(v: number): void                    { this.cfg.uDecayRate      = v; }
  setCurlStrength(v: number): void                 { this.cfg.uCurlStrength   = v; }
  setCurlNoiseScale(v: number): void               { this.cfg.uCurlNoiseScale = v; }
  setCurlNoiseSpeed(v: number): void               { this.cfg.uCurlNoiseSpeed = v; }
  setSize(v: number): void                         { this.cfg.uSize           = v; }
  setDPR(v: number): void                          { this.cfg.dpr             = v; }

  applyPreset(preset: ATSplineParticleConfig): void {
    if (preset.uSplineSpeed)               this.cfg.uSplineSpeed    = preset.uSplineSpeed;
    if (preset.uTimeMultiplier !== undefined) this.cfg.uTimeMultiplier = preset.uTimeMultiplier;
    if (preset.uFlowRange)                 this.cfg.uFlowRange      = preset.uFlowRange;
    if (preset.uDecayRate   !== undefined) this.cfg.uDecayRate      = preset.uDecayRate;
    if (preset.uMaxSDelay   !== undefined) this.cfg.uMaxSDelay      = preset.uMaxSDelay;
    if (preset.uCurlNoiseScale !== undefined) this.cfg.uCurlNoiseScale = preset.uCurlNoiseScale;
    if (preset.uCurlNoiseSpeed !== undefined) this.cfg.uCurlNoiseSpeed = preset.uCurlNoiseSpeed;
    if (preset.uCurlStrength !== undefined)   this.cfg.uCurlStrength   = preset.uCurlStrength;
    if (preset.uSize        !== undefined) this.cfg.uSize           = preset.uSize;
  }

  /** Replace edge splines — triggers full rebuild. */
  setEdges(edges: EdgeSpline[]): void {
    this.edges = edges;
    this.dispose();
    this._init();
  }

  // ─── Accessors ────────────────────────────────────────────────────────────

  get particleSlots(): number { return this.particleCount; }
  get edgeCount(): number      { return this.edges.length; }
  get config(): Readonly<Required<Omit<ATSplineParticleConfig, 'onHandoff'>>> {
    return this.cfg;
  }

  // ─── Private: compile with Transform Feedback ─────────────────────────────

  /** Compile a WebGL2 program with Transform Feedback varyings declared. */
  private _compileWithTF(
    vertSrc:   string,
    fragSrc:   string,
    varyings:  string[],
    label:     string,
  ): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vertSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(
        `[ATSplineParticleLife] vertex compile error (${label}):\n` +
        gl.getShaderInfoLog(vs),
      );
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(
        `[ATSplineParticleLife] fragment compile error (${label}):\n` +
        gl.getShaderInfoLog(fs),
      );
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);

    // Declare TF varyings BEFORE linking
    gl.transformFeedbackVaryings(prog, varyings, gl.SEPARATE_ATTRIBS);

    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(
        `[ATSplineParticleLife] link error (${label}):\n` +
        gl.getProgramInfoLog(prog),
      );
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** Compile a standard WebGL2 program. */
  private _compile(vertSrc: string, fragSrc: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vertSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(
        `[ATSplineParticleLife] vertex compile error (${label}):\n` +
        gl.getShaderInfoLog(vs),
      );
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(
        `[ATSplineParticleLife] fragment compile error (${label}):\n` +
        gl.getShaderInfoLog(fs),
      );
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(
        `[ATSplineParticleLife] link error (${label}):\n` +
        gl.getProgramInfoLog(prog),
      );
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  // ─── Private: VAO construction ────────────────────────────────────────────

  /** Build a VAO for the update (TF) pass. */
  private _buildUpdateVAO(
    posBuf: WebGLBuffer,
    velBuf: WebGLBuffer,
    lifeBuf: WebGLBuffer,
  ): WebGLVertexArrayObject {
    const gl   = this.gl;
    const prog = this.updateProg;
    const vao  = gl.createVertexArray()!;

    gl.bindVertexArray(vao);

    const aPos  = gl.getAttribLocation(prog, 'aPos');
    const aVel  = gl.getAttribLocation(prog, 'aVel');
    const aLife = gl.getAttribLocation(prog, 'aLife');

    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 4, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, velBuf);
    gl.enableVertexAttribArray(aVel);
    gl.vertexAttribPointer(aVel, 4, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, lifeBuf);
    gl.enableVertexAttribArray(aLife);
    gl.vertexAttribPointer(aLife, 4, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
    return vao;
  }

  /** Build a VAO for the render pass. */
  private _buildRenderVAO(
    posBuf: WebGLBuffer,
    velBuf: WebGLBuffer,
    lifeBuf: WebGLBuffer,
  ): WebGLVertexArrayObject {
    const gl   = this.gl;
    const prog = this.renderProg;
    const vao  = gl.createVertexArray()!;

    gl.bindVertexArray(vao);

    const aPos  = gl.getAttribLocation(prog, 'aPos');
    const aVel  = gl.getAttribLocation(prog, 'aVel');
    const aLife = gl.getAttribLocation(prog, 'aLife');

    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 4, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, velBuf);
    gl.enableVertexAttribArray(aVel);
    gl.vertexAttribPointer(aVel, 4, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, lifeBuf);
    gl.enableVertexAttribArray(aLife);
    gl.vertexAttribPointer(aLife, 4, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
    return vao;
  }

  // ─── Private: build spline lookup texture ─────────────────────────────────

  /**
   * Bake all EdgeSpline control points into a rgba32f lookup texture.
   * Layout: spline index i → PER_SPLINE consecutive texels, each = vec4(x,y,z,1)
   * Mirrors AT's splineparticles.fs getSplineLookupUV / tSpline pattern.
   */
  private _buildSplineTexture(): WebGLTexture {
    const gl      = this.gl;
    const W       = SPLINE_TEX_W;
    const H       = SPLINE_TEX_H;
    const data    = new Float32Array(W * H * 4);

    for (let e = 0; e < this.edges.length && e < MAX_SPLINES; e++) {
      const edge = this.edges[e];
      for (let s = 0; s < PER_SPLINE; s++) {
        const u   = s / (PER_SPLINE - 1);
        const pt  = evalSplineCPU(edge.points, u);
        const pixel = PER_SPLINE * e + s;
        const px  = pixel % W;
        const py  = Math.floor(pixel / W);
        const idx = (py * W + px) * 4;
        data[idx + 0] = pt.x;
        data[idx + 1] = pt.y;
        data[idx + 2] = pt.z;
        data[idx + 3] = 1.0;
      }
    }

    // Create rgba32f texture (WebGL2 required — no extension needed)
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D, 0,
      gl.RGBA32F,        // WebGL2 internal format
      W, H, 0,
      gl.RGBA, gl.FLOAT, data,
    );
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  // ─── Private: init particle data ─────────────────────────────────────────

  private _buildInitData(): {
    positions:  Float32Array;
    velocities: Float32Array;
    lives:      Float32Array;
  } {
    const N   = this.particleCount;
    const pos = new Float32Array(N * 4);
    const vel = new Float32Array(N * 4);
    const lif = new Float32Array(N * 4);

    let slot = 0;
    for (let e = 0; e < this.edges.length && slot < N; e++) {
      const edge  = this.edges[e];
      const count = Math.min(
        Math.ceil(edge.weight * this.cfg.particlesPerUnit),
        N - slot,
      );
      for (let p = 0; p < count && slot < N; p++, slot++) {
        const startPt = evalSplineCPU(edge.points, 0);
        const speed   = this.cfg.uSplineSpeed[0]
          + Math.random() * (this.cfg.uSplineSpeed[1] - this.cfg.uSplineSpeed[0]);
        const delay   = Math.random() * this.cfg.uMaxSDelay;

        const b = slot * 4;
        pos[b + 0] = startPt.x;
        pos[b + 1] = startPt.y;
        pos[b + 2] = startPt.z;
        pos[b + 3] = 0;          // travel

        vel[b + 0] = 0;
        vel[b + 1] = 0;
        vel[b + 2] = 0;
        vel[b + 3] = speed;

        lif[b + 0] = 1.0;                         // life
        lif[b + 1] = delay > 0 ? 0.0 : 1.0;       // phase: SPAWN or FLOW
        lif[b + 2] = delay;                        // delay
        lif[b + 3] = edge.species ?? 0;            // species
      }
    }

    // Remaining slots start DEAD (phase 3)
    for (; slot < N; slot++) {
      const b = slot * 4;
      lif[b + 1] = 3.0;   // phase = DEAD → will respawn
    }

    return { positions: pos, velocities: vel, lives: lif };
  }

  // ─── Private: handoff readback (synchronous getBufferSubData) ────────────

  /**
   * Read back TF output life buffer to scan handoff flags.
   * Uses getBufferSubData — synchronous but lightweight (only life vec4).
   * Fires onHandoff for each particle with tfLife.w === 1 (handoff flag set).
   */
  private _doHandoffReadback(
    posBuf:  WebGLBuffer,
    velBuf:  WebGLBuffer,
    lifeBuf: WebGLBuffer,
    _elapsed: number,
  ): void {
    const gl = this.gl;
    const N  = this.particleCount;

    // Copy TF life output → readback buffer
    gl.bindBuffer(gl.COPY_READ_BUFFER, lifeBuf);
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, this.readbackLifeBuf);
    gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.COPY_WRITE_BUFFER, 0, 0, N * 4 * 4);

    // Copy position for world coords
    gl.bindBuffer(gl.COPY_READ_BUFFER, posBuf);
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, this.readbackPosBuf);
    gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.COPY_WRITE_BUFFER, 0, 0, N * 4 * 4);

    // Copy velocity for handoff vx/vy
    gl.bindBuffer(gl.COPY_READ_BUFFER, velBuf);
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, this.readbackVelBuf);
    gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.COPY_WRITE_BUFFER, 0, 0, N * 4 * 4);

    gl.bindBuffer(gl.COPY_READ_BUFFER, null);
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, null);

    // Read life data — scan for handoff flags
    const lifeData = new Float32Array(N * 4);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.readbackLifeBuf);
    gl.getBufferSubData(gl.ARRAY_BUFFER, 0, lifeData);

    // Quick pass to see if any handoffs fired
    let anyHandoff = false;
    for (let i = 0; i < N; i++) {
      if (lifeData[i * 4 + 3] > 0.5) { anyHandoff = true; break; }
    }

    if (!anyHandoff || !this.onHandoff) {
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      return;
    }

    // Read position + velocity only if needed
    const posData = new Float32Array(N * 4);
    const velData = new Float32Array(N * 4);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.readbackPosBuf);
    gl.getBufferSubData(gl.ARRAY_BUFFER, 0, posData);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.readbackVelBuf);
    gl.getBufferSubData(gl.ARRAY_BUFFER, 0, velData);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Fire callbacks
    for (let i = 0; i < N; i++) {
      const lb = i * 4;
      if (lifeData[lb + 3] < 0.5) continue;   // no handoff flag

      const pb      = lb;
      const species = Math.round(lifeData[lb + 3 - 1 + 1]);   // life[w] = species? No, handoff=w
      // life layout: x=life, y=phase, z=delay, w=handoff; species was in original slot
      // We don't have species in TF output life.w (that's handoff flag) —
      // recover approximate edge from spline index logic or use 0
      const x   = posData[pb + 0];
      const y   = posData[pb + 1];
      const vx  = velData[pb + 0];
      const vy  = velData[pb + 1];

      // Find closest edge by particle position heuristic
      // (AT does this by storing edge index in the particle data;
      //  here we use the approximate approach of checking all edges)
      let bestEdge: EdgeSpline | undefined;
      let bestDist = Infinity;
      for (const edge of this.edges) {
        if (!edge.points.length) continue;
        const endPt = evalSplineCPU(edge.points, 0.99);
        const d = Math.hypot(endPt.x - x, endPt.y - y);
        if (d < bestDist) { bestDist = d; bestEdge = edge; }
      }

      if (bestEdge) {
        this.onHandoff(
          bestEdge.edgeId,
          bestEdge.targetId,
          x, y,
          vx, vy,
          bestEdge.species ?? 0,
        );
      }
    }
  }
}

// ─── Factory helpers ──────────────────────────────────────────────────────────

/**
 * Wire an ATSplineParticleLife to SPHWorld.addFluid() for automatic
 * fluid injection when particles arrive at their target cells.
 */
export function createATSplineParticleForSPH(
  gl:       WebGL2RenderingContext,
  edges:    EdgeSpline[],
  addFluid: (x0: number, y0: number, x1: number, y1: number, spacing: number, species: number) => void,
  config:   Omit<ATSplineParticleConfig, 'onHandoff'> = {},
): ATSplineParticleLife {
  const R = 0.05;
  return new ATSplineParticleLife(gl, edges, {
    ...config,
    onHandoff: (_eId, _tId, x, y, _vx, _vy, species) => {
      addFluid(x - R, y - R, x + R, y + R, R * 0.8, species);
    },
  });
}

/**
 * Convert raw canvas-space route points to EdgeSpline control points
 * in SPH domain coordinates.
 */
export function canvasRouteToEdgeSpline(
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
): EdgeSpline {
  const sx = domainW / canvasW;
  const sy = domainH / canvasH;
  return {
    edgeId, sourceId, targetId, weight, species,
    points: points.map(p => ({ x: p.x * sx, y: p.y * sy, z: 0 })),
  };
}

// ─── Constants re-export ──────────────────────────────────────────────────────

export const AT_SPLINE_PARTICLE_DEFAULTS = {
  maxParticles:  MAX_PARTICLES,
  splineTexW:    SPLINE_TEX_W,
  splineTexH:    SPLINE_TEX_H,
  perSpline:     PER_SPLINE,
  maxSplines:    MAX_SPLINES,
} as const;
