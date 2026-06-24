/**
 * dynamic-environment.ts — M765: Dynamic Environment System
 * ─────────────────────────────────────────────────────────────────────────────
 * Drives the sky through a continuous dawn → day → dusk → night cycle with
 * physically-motivated ambient occlusion and environment-light colour that
 * tracks the current phase.  Every visual parameter is smoothly interpolated
 * so transitions never pop.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Sky Cycle Phases
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   Phase      t-range     Sun Elevation   Key Colour Mood
 *   ──────── ─────────── ─────────────── ────────────────────────────
 *   DAWN     [0.00,0.25)  0 → 0.65       warm peach → pale gold
 *   DAY      [0.25,0.50)  0.65 → 0.65    bright blue sky, high sun
 *   DUSK     [0.50,0.75)  0.65 → 0       amber → deep crimson
 *   NIGHT    [0.75,1.00)  0 → 0          indigo, stars + moonlight
 *
 *   t wraps modulo 1.0, so the cycle repeats indefinitely.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Per-frame Pipeline (WebGPU)
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Compute Pass: AMBIENT OCCLUSION ──────────────────────────────────────┐
 *   │  depthField → aoTex (half-res)                                         │
 *   │  Multi-sample horizon-based AO using the density field as a depth      │
 *   │  proxy.  Kernel radius and intensity modulated by time-of-day: night   │
 *   │  tightens the kernel (moonlit hard shadows), dawn/dusk widens it       │
 *   │  (soft diffuse scatter).                                               │
 *   └────────────────────────────────────────────────────────────────────────┘
 *                │ aoTex
 *                ▼
 *   ┌─ Render Pass: SKY GRADIENT + ENV LIGHT COMPOSITE ─────────────────────┐
 *   │  sceneTex × aoTex + skyGradient → dst                                  │
 *   │  The sky gradient is a vertical blend of three colour stops that       │
 *   │  smoothly evolve with the cycle phase.  The environment light tints    │
 *   │  the entire scene through a soft multiply.  AO darkens crevices.      │
 *   └────────────────────────────────────────────────────────────────────────┘
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Design References
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   src/lib/sph/atmosphere.ts              — Rayleigh/Mie scatter + exp² fog
 *   src/lib/sph/environment-fog.ts         — depth fog + god ray composite
 *   src/lib/sph/environment-fx.ts          — compute→render dual pass pattern
 *   src/lib/sph/at-volumetric-light.ts     — RT/BGL/pipeline helpers
 *   src/lib/sph/at-shader-utils.ts         — WGSL eases, range, blend modes
 *   src/lib/sph/tone-mapping.ts            — ACES filmic tone mapping
 *   upstream/lygia/lighting/atmosphere.wgsl — Rayleigh phase function
 *   upstream/lygia/color/blend/*.wgsl      — blend mode reference
 *   GPU Gems 2 Ch.16 "Accurate Atmospheric Scattering"
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Quick Start
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   const dynEnv = await DynamicEnvironment.create(device, format, w, h);
 *   dynEnv.configure({ cycleDuration: 120, aoIntensity: 0.6 });
 *
 *   // per frame:
 *   dynEnv.tick(dt);
 *   dynEnv.render(encoder, sceneTex, depthFieldTex, dstView);
 *
 *   // query current state for other systems (atmosphere, fog, etc.):
 *   const snap = dynEnv.snapshot();
 *   atmospherePass.setParams({ sunElevation: snap.sunElevation, fogColor: snap.fogColor });
 *
 * Research: xiaodi #M765 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** The four phases of the sky cycle. */








export type SkyPhase = 'dawn' | 'day' | 'dusk' | 'night';

/** RGB triplet in linear sRGB [0,1]. */
export type Color3 = [number, number, number];

/**
 * Immutable snapshot of the environment state at a given moment.
 * Other systems (atmosphere, fog, PBR material) can read this to
 * keep their parameters synchronised with the sky cycle.
 */
export interface EnvironmentSnapshot {
  /** Current phase label. */
  phase: SkyPhase;
  /** Normalised cycle position [0,1). */
  cycleT: number;
  /** Sun elevation [0,1]: 0 = horizon, 1 = zenith. */
  sunElevation: number;
  /** Sun azimuth angle in radians [0, 2π). */
  sunAzimuth: number;
  /** Primary environment-light colour (linear sRGB). */
  envLightColor: Color3;
  /** Environment-light intensity multiplier [0,∞). */
  envLightIntensity: number;
  /** Sky zenith colour (linear sRGB). */
  skyZenith: Color3;
  /** Sky horizon colour (linear sRGB). */
  skyHorizon: Color3;
  /** Sky nadir colour (linear sRGB) — below horizon fill. */
  skyNadir: Color3;
  /** Fog colour derived from current sky state (linear sRGB). */
  fogColor: Color3;
  /** Fog density [0,1]. */
  fogDensity: number;
  /** AO intensity in effect [0,1]. */
  aoIntensity: number;
  /** Star visibility [0,1] — fades in at night, out at dawn. */
  starVisibility: number;
}

/** User-configurable parameters for the dynamic environment system. */
export interface DynamicEnvironmentConfig {
  /**
   * Duration of one full dawn→day→dusk→night cycle in seconds.
   * @default 120
   */
  cycleDuration?: number;

  /**
   * Initial phase offset [0,1) — lets you start at a specific time of day.
   * 0 = start of dawn, 0.25 = start of day, 0.5 = start of dusk, 0.75 = night.
   * @default 0
   */
  phaseOffset?: number;

  /**
   * If true, the cycle pauses and cycleT is driven solely by `setCycleT()`.
   * @default false
   */
  paused?: boolean;

  /**
   * Ambient occlusion intensity [0,1].  0 disables AO entirely.
   * @default 0.55
   */
  aoIntensity?: number;

  /**
   * AO sample radius in UV-space.  Larger = softer, more diffuse.
   * @default 0.012
   */
  aoRadius?: number;

  /**
   * Number of AO sample taps per pixel (4–16).
   * Higher = smoother but more expensive.
   * @default 8
   */
  aoSamples?: number;

  /**
   * Environment-light intensity multiplier.  Scales the tint applied to the
   * entire scene.  1.0 = physically derived, >1 = exaggerated mood.
   * @default 1.0
   */
  envLightScale?: number;

  /**
   * Sky gradient vertical bias.  0.5 = horizon at screen centre,
   * lower = horizon drops, showing more sky.
   * @default 0.5
   */
  horizonBias?: number;

  /**
   * Star density for the night sky (particles per unit area).
   * @default 0.0008
   */
  starDensity?: number;

  /**
   * Fog density range [min, max] — interpolated across phases.
   * Dawn/dusk have higher fog; midday and deep night are clearer.
   * @default [0.08, 0.38]
   */
  fogDensityRange?: [number, number];
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal constants
// ─────────────────────────────────────────────────────────────────────────────

const PI  = Math.PI;
const TAU = PI * 2;

/** Uniform buffer layout: 48 floats = 192 bytes. */
const UNI_F32 = 48;
const UNI_BYTES = UNI_F32 * 4;

/**
 * Phase keyframes — each entry defines the target state at the *start*
 * of the named phase.  Interpolation happens smoothly between consecutive
 * entries using Hermite (smoothstep) blending within each quarter.
 */
interface PhaseKeyframe {
  sunElevation: number;
  envLight: Color3;
  envIntensity: number;
  skyZenith: Color3;
  skyHorizon: Color3;
  skyNadir: Color3;
  fogColor: Color3;
  fogDensity: number;
  aoMul: number;        // multiplier on base AO intensity
  aoRadiusMul: number;  // multiplier on base AO radius
  starVis: number;
}

const KEYFRAMES: Record<SkyPhase, PhaseKeyframe> = {
  dawn: {
    sunElevation: 0.0,
    envLight:     [1.0, 0.72, 0.48],
    envIntensity: 0.65,
    skyZenith:    [0.18, 0.22, 0.55],
    skyHorizon:   [0.95, 0.65, 0.38],
    skyNadir:     [0.12, 0.08, 0.18],
    fogColor:     [0.85, 0.62, 0.42],
    fogDensity:   0.30,
    aoMul:        0.8,
    aoRadiusMul:  1.3,
    starVis:      0.25,
  },
  day: {
    sunElevation: 0.65,
    envLight:     [1.0, 0.98, 0.92],
    envIntensity: 1.0,
    skyZenith:    [0.22, 0.45, 0.85],
    skyHorizon:   [0.62, 0.78, 0.95],
    skyNadir:     [0.15, 0.20, 0.32],
    fogColor:     [0.72, 0.82, 0.95],
    fogDensity:   0.10,
    aoMul:        1.0,
    aoRadiusMul:  1.0,
    starVis:      0.0,
  },
  dusk: {
    sunElevation: 0.65,
    envLight:     [1.0, 0.58, 0.28],
    envIntensity: 0.72,
    skyZenith:    [0.25, 0.15, 0.42],
    skyHorizon:   [0.92, 0.42, 0.18],
    skyNadir:     [0.10, 0.06, 0.15],
    fogColor:     [0.82, 0.48, 0.28],
    fogDensity:   0.35,
    aoMul:        0.85,
    aoRadiusMul:  1.2,
    starVis:      0.15,
  },
  night: {
    sunElevation: 0.0,
    envLight:     [0.35, 0.42, 0.72],
    envIntensity: 0.28,
    skyZenith:    [0.02, 0.03, 0.10],
    skyHorizon:   [0.05, 0.06, 0.18],
    skyNadir:     [0.01, 0.01, 0.05],
    fogColor:     [0.06, 0.08, 0.18],
    fogDensity:   0.15,
    aoMul:        1.3,
    aoRadiusMul:  0.7,
    starVis:      1.0,
  },
};

/** Ordered phase list for cyclic lookup. */
const PHASE_ORDER: SkyPhase[] = ['dawn', 'day', 'dusk', 'night'];

// ─────────────────────────────────────────────────────────────────────────────
// WGSL Shaders
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compute shader: horizon-based ambient occlusion.
 *
 * Uses the density field as a depth proxy — high density ≈ near surface,
 * low density ≈ far/empty.  Samples a disc of neighbours and accumulates
 * occlusion based on relative depth difference + distance falloff.
 */
const AO_COMPUTE_WGSL = /* wgsl */`
struct AOUniforms {
  width      : f32,
  height     : f32,
  radius     : f32,   // sample radius in UV space
  intensity  : f32,   // strength multiplier
  sampleCount: f32,   // number of taps (f32 for uniform alignment)
  frameIndex : f32,   // for per-frame jitter rotation
  _pad0      : f32,
  _pad1      : f32,
}

@group(0) @binding(0) var<uniform>      u         : AOUniforms;
@group(0) @binding(1) var               fieldTex  : texture_2d<f32>;
@group(0) @binding(2) var               smp       : sampler;
@group(0) @binding(3) var               aoOut     : texture_storage_2d<r8unorm, write>;

const PI : f32 = 3.14159265358979;

// ── Interleaved gradient noise (Jorge Jimenez, SIGGRAPH 2014) ──────────────
fn interleavedGradientNoise(fragCoord: vec2f) -> f32 {
  let f = fract(vec3f(fragCoord.xyx) * vec3f(0.06711056, 0.00583715, 0.1387128));
  return fract(52.9829189 * fract(dot(f, vec3f(1.0))));
}

// ── Vogel disc sample (uniform angular distribution) ───────────────────────
fn vogelDisc(index: u32, total: u32, phi: f32) -> vec2f {
  let goldenAngle = 2.399963; // PI * (3 - sqrt(5))
  let r  = sqrt((f32(index) + 0.5) / f32(total));
  let th = f32(index) * goldenAngle + phi;
  return vec2f(cos(th), sin(th)) * r;
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let outSize = vec2u(textureDimensions(aoOut));
  if (gid.x >= outSize.x || gid.y >= outSize.y) { return; }

  let uv    = (vec2f(gid.xy) + 0.5) / vec2f(outSize);
  let depth = textureSampleLevel(fieldTex, smp, uv, 0.0).r;

  // Skip truly empty regions (no geometry to occlude)
  if (depth < 0.005) {
    textureStore(aoOut, gid.xy, vec4f(1.0));
    return;
  }

  let samples = u32(u.sampleCount);
  // Per-pixel jitter angle to break banding — rotated by frame index
  let jitter  = interleavedGradientNoise(vec2f(gid.xy)) * PI * 2.0
              + u.frameIndex * 2.399963;

  var occlusion = 0.0;
  let aspect    = u.width / u.height;
  let radUV     = vec2f(u.radius, u.radius * aspect);

  for (var i = 0u; i < samples; i++) {
    let offset    = vogelDisc(i, samples, jitter);
    let sampleUV  = uv + offset * radUV;
    let sampleD   = textureSampleLevel(fieldTex, smp, sampleUV, 0.0).r;

    // Range check: occlusion only from surfaces that are *closer*
    // (higher density) than the current fragment, with a falloff.
    let diff = sampleD - depth;
    let rangeCheck = smoothstep(0.0, 0.08, abs(diff));
    occlusion += step(0.02, diff) * rangeCheck;
  }

  occlusion = occlusion / f32(samples);
  let ao    = 1.0 - clamp(occlusion * u.intensity, 0.0, 1.0);

  textureStore(aoOut, gid.xy, vec4f(ao, ao, ao, 1.0));
}
`;

/**
 * Fragment shader: sky gradient + environment light composite.
 *
 * Reads the scene colour, multiplies by AO, applies environment light tint,
 * and blends in a sky gradient behind transparent regions.  Includes a
 * simple procedural star field for the night phase.
 */
const COMPOSITE_WGSL = /* wgsl */`
struct EnvUniforms {
  // Resolution
  width         : f32,
  height        : f32,
  // Sky gradient colours (linear sRGB)
  zenithR       : f32, zenithG : f32, zenithB : f32,
  horizonR      : f32, horizonG: f32, horizonB: f32,
  nadirR        : f32, nadirG  : f32, nadirB  : f32,
  // Environment light
  envLightR     : f32, envLightG: f32, envLightB: f32,
  envLightInt   : f32,
  // Sun position (normalised screen-space)
  sunX          : f32, sunY      : f32,
  sunElevation  : f32,
  // Horizon bias & star visibility
  horizonBias   : f32,
  starVis       : f32,
  starDensity   : f32,
  // Fog
  fogR          : f32, fogG : f32, fogB : f32,
  fogDensity    : f32,
  // Time-of-day for animated effects
  time          : f32,
  // AO influence on env light [0,1]
  aoInfluence   : f32,
  _pad0         : f32, _pad1: f32,
}

@group(0) @binding(0) var<uniform> u       : EnvUniforms;
@group(0) @binding(1) var          smp     : sampler;
@group(0) @binding(2) var          sceneTex: texture_2d<f32>;
@group(0) @binding(3) var          aoTex   : texture_2d<f32>;

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
}

// Full-screen triangle
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
  let y = f32( vi         & 2u) * 2.0 - 1.0;
  var o: VSOut;
  o.pos = vec4f(x, y, 0.0, 1.0);
  o.uv  = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
  return o;
}

// ── Hash for star field ────────────────────────────────────────────────────
fn hash21(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2f) -> vec2f {
  let n = sin(vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3))));
  return fract(n * 43758.5453);
}

// ── Procedural star field ──────────────────────────────────────────────────
fn stars(uv: vec2f, density: f32, time: f32) -> f32 {
  let scale  = 1.0 / max(density, 0.0001);
  let gridUV = uv * scale;
  let cell   = floor(gridUV);
  let frac   = fract(gridUV);

  var brightness = 0.0;
  // Check 3×3 neighbourhood for closest star
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let neighbour = cell + vec2f(f32(dx), f32(dy));
      let starPos   = hash22(neighbour);
      let diff      = starPos - frac + vec2f(f32(dx), f32(dy));
      let dist      = length(diff);

      // Size variation from hash
      let starSize = 0.02 + 0.03 * hash21(neighbour + vec2f(42.0));
      // Twinkle: slow sinusoidal on per-star phase
      let phase    = hash21(neighbour + vec2f(7.13)) * 6.2832;
      let twinkle  = 0.6 + 0.4 * sin(time * (1.5 + hash21(neighbour) * 2.0) + phase);

      let star = smoothstep(starSize, 0.0, dist) * twinkle;
      brightness = max(brightness, star);
    }
  }
  return brightness;
}

// ── Smooth sky gradient ────────────────────────────────────────────────────
fn skyGradient(uv: vec2f) -> vec3f {
  let zenith  = vec3f(u.zenithR,  u.zenithG,  u.zenithB);
  let horizon = vec3f(u.horizonR, u.horizonG, u.horizonB);
  let nadir   = vec3f(u.nadirR,   u.nadirG,   u.nadirB);

  // Vertical position relative to horizon
  let vy = (uv.y - u.horizonBias) * 2.0;

  if (vy >= 0.0) {
    // Above horizon: horizon → zenith
    let t = smoothstep(0.0, 1.0, clamp(vy, 0.0, 1.0));
    return mix(horizon, zenith, t);
  } else {
    // Below horizon: horizon → nadir
    let t = smoothstep(0.0, 1.0, clamp(-vy, 0.0, 1.0));
    return mix(horizon, nadir, t);
  }
}

// ── Sun disc + atmospheric glow ────────────────────────────────────────────
fn sunGlow(uv: vec2f) -> vec3f {
  let sunPos = vec2f(u.sunX, u.sunY);
  let dist   = length(uv - sunPos);

  // Only show sun glow when it's above the horizon
  let elevFade = smoothstep(0.0, 0.15, u.sunElevation);

  // Core disc
  let disc = smoothstep(0.035, 0.015, dist) * 0.4 * elevFade;
  // Atmospheric halo
  let halo = exp(-dist * 8.0) * 0.35 * elevFade;
  // Colour: warm white core, env-tinted halo
  let sunCol = vec3f(1.0, 0.95, 0.85);
  let envCol = vec3f(u.envLightR, u.envLightG, u.envLightB);

  return sunCol * disc + envCol * halo;
}

// ── ACES Narkowicz (simple filmic) ─────────────────────────────────────────
fn acesToneMap(x: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  let mapped = (x * (a * x + b)) / (x * (c * x + d) + e);
  return clamp(mapped, vec3f(0.0), vec3f(1.0));
}

// ── Fragment main ──────────────────────────────────────────────────────────
@fragment fn fs_main(in: VSOut) -> @location(0) vec4f {
  let uv = in.uv;

  // 1. Sample scene + AO
  let scene = textureSampleLevel(sceneTex, smp, uv, 0.0);
  let ao    = textureSampleLevel(aoTex,    smp, uv, 0.0).r;

  // 2. Sky background
  var sky = skyGradient(uv);
  sky    += sunGlow(uv);

  // 3. Star layer (night only)
  if (u.starVis > 0.001) {
    let starBright = stars(uv, u.starDensity, u.time) * u.starVis;
    // Stars are white-ish with slight blue tint
    sky += vec3f(0.85, 0.90, 1.0) * starBright;
  }

  // 4. Environment light tint
  let envLight = vec3f(u.envLightR, u.envLightG, u.envLightB) * u.envLightInt;

  // 5. Compose: scene with AO, tinted by environment light
  let aoFactor  = mix(1.0, ao, u.aoInfluence);
  var sceneCol  = scene.rgb * aoFactor;
  // Soft tint: multiply-blend environment colour
  sceneCol = sceneCol * mix(vec3f(1.0), envLight, 0.35);

  // 6. Blend scene over sky (alpha composite)
  var outCol = mix(sky, sceneCol, scene.a);

  // 7. Depth fog on scene regions only
  if (scene.a > 0.01) {
    let fogCol    = vec3f(u.fogR, u.fogG, u.fogB);
    // Use AO as depth proxy — lower AO ≈ deeper crevices ≈ more fog
    let fogDepth  = 1.0 - ao;
    let fogFactor = 1.0 - exp(-u.fogDensity * u.fogDensity * fogDepth * fogDepth * 6.0);
    outCol = mix(outCol, fogCol, fogFactor * 0.5 * scene.a);
  }

  // 8. Tone mapping + gamma
  outCol = acesToneMap(outCol);
  outCol = pow(clamp(outCol, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));

  return vec4f(outCol, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function lerpC3(a: Color3, b: Color3, t: number): Color3 {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Attempt at implementing smoothstep. */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function phaseAtT(t: number): { phase: SkyPhase; localT: number; nextPhase: SkyPhase } {
  const idx     = Math.floor(t * 4) % 4;
  const localT  = (t * 4) % 1;
  const nextIdx = (idx + 1) % 4;
  return {
    phase    : PHASE_ORDER[idx],
    localT,
    nextPhase: PHASE_ORDER[nextIdx],
  };
}

function interpolateKeyframes(kfA: PhaseKeyframe, kfB: PhaseKeyframe, t: number): {
  sunElevation: number;
  envLight: Color3;
  envIntensity: number;
  skyZenith: Color3;
  skyHorizon: Color3;
  skyNadir: Color3;
  fogColor: Color3;
  fogDensity: number;
  aoMul: number;
  aoRadiusMul: number;
  starVis: number;
} {
  // Use smoothstep for perceptually smooth transitions
  const s = smoothstep(0, 1, t);
  return {
    sunElevation: lerp(kfA.sunElevation, kfB.sunElevation, s),
    envLight:     lerpC3(kfA.envLight, kfB.envLight, s),
    envIntensity: lerp(kfA.envIntensity, kfB.envIntensity, s),
    skyZenith:    lerpC3(kfA.skyZenith, kfB.skyZenith, s),
    skyHorizon:   lerpC3(kfA.skyHorizon, kfB.skyHorizon, s),
    skyNadir:     lerpC3(kfA.skyNadir, kfB.skyNadir, s),
    fogColor:     lerpC3(kfA.fogColor, kfB.fogColor, s),
    fogDensity:   lerp(kfA.fogDensity, kfB.fogDensity, s),
    aoMul:        lerp(kfA.aoMul, kfB.aoMul, s),
    aoRadiusMul:  lerp(kfA.aoRadiusMul, kfB.aoRadiusMul, s),
    starVis:      lerp(kfA.starVis, kfB.starVis, s),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// DynamicEnvironment
// ─────────────────────────────────────────────────────────────────────────────

export class DynamicEnvironment {
  private readonly device: any /*GPUDevice*/;
  private readonly format: GPUTextureFormat;

  // Dimensions (updated on resize)
  private width  = 0;
  private height = 0;

  // Config (merged with defaults)
  private cfg: Required<DynamicEnvironmentConfig> = {
    cycleDuration  : 120,
    phaseOffset    : 0,
    paused         : false,
    aoIntensity    : 0.55,
    aoRadius       : 0.012,
    aoSamples      : 8,
    envLightScale  : 1.0,
    horizonBias    : 0.5,
    starDensity    : 0.0008,
    fogDensityRange: [0.08, 0.38],
  };

  // Cycle accumulator
  private elapsed    = 0;
  private cycleT     = 0;
  private frameIndex = 0;

  // Current interpolated state
  private currentPhase: SkyPhase = 'dawn';
  private interp = interpolateKeyframes(KEYFRAMES.dawn, KEYFRAMES.day, 0);

  // ── GPU resources — AO compute pass ──
  private aoComputePipeline!: any /*GPUComputePipeline*/;
  private aoBGL!: GPUBindGroupLayout;
  private aoUniBuf!: any /*GPUBuffer*/;
  private aoTex!: GPUTexture;
  private aoTexView!: any /*GPUTextureView*/;
  private aoSampler!: GPUSampler;
  private aoBG: any /*GPUBindGroup*/ | null = null;
  private lastFieldView: any /*GPUTextureView*/ | null = null;

  // ── GPU resources — composite render pass ──
  private compositePipeline!: any /*GPURenderPipeline*/;
  private compositeBGL!: GPUBindGroupLayout;
  private compositeUniBuf!: any /*GPUBuffer*/;
  private compositeSampler!: GPUSampler;
  private compositeBG: any /*GPUBindGroup*/ | null = null;
  private lastSceneView: any /*GPUTextureView*/ | null = null;
  private lastAoView: any /*GPUTextureView*/ | null = null;

  // ─────────────────────────────────────────────────────────────────────────
  // Construction
  // ─────────────────────────────────────────────────────────────────────────

  private constructor(device: any /*GPUDevice*/, format: GPUTextureFormat, w: number, h: number) {
    this.device = device;
    this.format = format;
    this.width  = w;
    this.height = h;
  }

  /**
   * Factory — creates all GPU resources asynchronously.
   */
  static async create(
    device: any /*GPUDevice*/,
    format: GPUTextureFormat,
    width : number,
    height: number,
  ): Promise<DynamicEnvironment> {
    const env = new DynamicEnvironment(device, format, width, height);
    await env._initAOPipeline();
    await env._initCompositePipeline();
    env._createAOTexture();
    return env;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API
  // ─────────────────────────────────────────────────────────────────────────

  /** Merge user config into current settings. */
  configure(cfg: DynamicEnvironmentConfig): void {
    Object.assign(this.cfg, cfg);
  }

  /** Manually set the cycle position [0,1).  Useful when `paused` is true. */
  setCycleT(t: number): void {
    this.cycleT = ((t % 1) + 1) % 1;
    this._updateInterpolation();
  }

  /**
   * Advance the cycle clock.  Call once per frame *before* render().
   * @param dt — frame delta in seconds.
   */
  tick(dt: number): void {
    if (!this.cfg.paused) {
      this.elapsed += dt;
      const raw     = this.elapsed / Math.max(this.cfg.cycleDuration, 0.001);
      this.cycleT   = ((raw + this.cfg.phaseOffset) % 1 + 1) % 1;
    }
    this.frameIndex++;
    this._updateInterpolation();
  }

  /**
   * Execute the AO compute pass + sky/env composite render pass.
   *
   * @param encoder     Current frame command encoder.
   * @param sceneView   Scene colour texture view (particle render output).
   * @param fieldView   Density field texture view (depth proxy for AO).
   * @param dstView     Final output target view.
   */
  render(
    encoder  : any /*GPUCommandEncoder*/,
    sceneView: any /*GPUTextureView*/,
    fieldView: any /*GPUTextureView*/,
    dstView  : any /*GPUTextureView*/,
  ): void {
    // ── Pass 1: AO compute ─────────────────────────────────────────────
    this._uploadAOUniforms();
    this._ensureAOBindGroup(fieldView);

    const aoW = Math.ceil(this.width  / 2);
    const aoH = Math.ceil(this.height / 2);
    const cPass = encoder.beginComputePass({ label: 'dynenv-ao-compute' });
    cPass.setPipeline(this.aoComputePipeline);
    cPass.setBindGroup(0, this.aoBG!);
    cPass.dispatchWorkgroups(Math.ceil(aoW / 8), Math.ceil(aoH / 8));
    cPass.end();

    // ── Pass 2: Composite render ───────────────────────────────────────
    this._uploadCompositeUniforms();
    this._ensureCompositeBindGroup(sceneView, this.aoTexView);

    const rPass = encoder.beginRenderPass({
      label           : 'dynenv-composite',
      colorAttachments: [{
        view   : dstView,
        loadOp : 'load',
        storeOp: 'store',
      }],
    });
    rPass.setPipeline(this.compositePipeline);
    rPass.setBindGroup(0, this.compositeBG!);
    rPass.draw(3);  // full-screen triangle
    rPass.end();
  }

  /** Resize internal textures.  Call when the canvas dimensions change. */
  resize(width: number, height: number): void {
    this.width  = width;
    this.height = height;
    // Recreate half-res AO texture
    this.aoTex?.destroy();
    this._createAOTexture();
    // Invalidate cached bind groups
    this.aoBG        = null;
    this.compositeBG = null;
  }

  /** Read-only snapshot for other systems to synchronise. */
  snapshot(): EnvironmentSnapshot {
    const i = this.interp;
    const sunAzimuth = this.cycleT * TAU;
    return {
      phase            : this.currentPhase,
      cycleT           : this.cycleT,
      sunElevation     : i.sunElevation,
      sunAzimuth,
      envLightColor    : [...i.envLight] as Color3,
      envLightIntensity: i.envIntensity * this.cfg.envLightScale,
      skyZenith        : [...i.skyZenith] as Color3,
      skyHorizon       : [...i.skyHorizon] as Color3,
      skyNadir         : [...i.skyNadir] as Color3,
      fogColor         : [...i.fogColor] as Color3,
      fogDensity       : lerp(this.cfg.fogDensityRange[0], this.cfg.fogDensityRange[1], i.fogDensity),
      aoIntensity      : this.cfg.aoIntensity * i.aoMul,
      starVisibility   : i.starVis,
    };
  }

  /** Release all GPU resources. */
  destroy(): void {
    this.aoUniBuf?.destroy();
    this.aoTex?.destroy();
    this.compositeUniBuf?.destroy();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal — interpolation update
  // ─────────────────────────────────────────────────────────────────────────

  private _updateInterpolation(): void {
    const { phase, localT, nextPhase } = phaseAtT(this.cycleT);
    this.currentPhase = phase;
    this.interp = interpolateKeyframes(KEYFRAMES[phase], KEYFRAMES[nextPhase], localT);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal — GPU initialisation
  // ─────────────────────────────────────────────────────────────────────────

  private async _initAOPipeline(): Promise<void> {
    const d  = this.device;
    const sm = d.createShaderModule({ label: 'dynenv-ao-shader', code: AO_COMPUTE_WGSL });

    this.aoBGL = d.createBindGroupLayout({
      label  : 'dynenv-ao-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer : { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r8unorm', viewDimension: '2d' } },
      ],
    });

    this.aoComputePipeline = await d.createComputePipelineAsync({
      label  : 'dynenv-ao-pipeline',
      layout : d.createPipelineLayout({ bindGroupLayouts: [this.aoBGL] }),
      compute: { module: sm, entryPoint: 'cs_main' },
    });

    this.aoUniBuf = d.createBuffer({
      label: 'dynenv-ao-uni',
      size : 8 * 4,  // 8 floats
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.aoSampler = d.createSampler({
      label    : 'dynenv-ao-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
  }

  private async _initCompositePipeline(): Promise<void> {
    const d  = this.device;
    const sm = d.createShaderModule({ label: 'dynenv-composite-shader', code: COMPOSITE_WGSL });

    this.compositeBGL = d.createBindGroupLayout({
      label  : 'dynenv-composite-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer : { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
      ],
    });

    this.compositePipeline = await d.createRenderPipelineAsync({
      label : 'dynenv-composite-pipeline',
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.compositeBGL] }),
      vertex  : { module: sm, entryPoint: 'vs_main' },
      fragment: {
        module    : sm,
        entryPoint: 'fs_main',
        targets   : [{
          format: this.format,
          blend : {
            color: { srcFactor: 'one', dstFactor: 'zero', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'zero', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.compositeUniBuf = d.createBuffer({
      label: 'dynenv-composite-uni',
      size : UNI_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.compositeSampler = d.createSampler({
      label    : 'dynenv-composite-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
  }

  private _createAOTexture(): void {
    const aoW = Math.max(1, Math.ceil(this.width  / 2));
    const aoH = Math.max(1, Math.ceil(this.height / 2));

    this.aoTex = this.device.createTexture({
      label : 'dynenv-ao-tex',
      size  : [aoW, aoH],
      format: 'r8unorm',
      usage : GPUTextureUsage.STORAGE_BINDING
            | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.aoTexView = this.aoTex.createView({ label: 'dynenv-ao-view' });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal — uniform upload
  // ─────────────────────────────────────────────────────────────────────────

  private _uploadAOUniforms(): void {
    const i   = this.interp;
    const arr = new Float32Array(8);

    arr[0] = this.width;
    arr[1] = this.height;
    arr[2] = this.cfg.aoRadius * i.aoRadiusMul;
    arr[3] = this.cfg.aoIntensity * i.aoMul;
    arr[4] = Math.min(16, Math.max(4, this.cfg.aoSamples));
    arr[5] = this.frameIndex;
    arr[6] = 0;  // _pad0
    arr[7] = 0;  // _pad1

    this.device.queue.writeBuffer(this.aoUniBuf, 0, arr);
  }

  private _uploadCompositeUniforms(): void {
    const i   = this.interp;
    const arr = new Float32Array(UNI_F32);

    // Sun screen-space position from azimuth + elevation
    const sunScreenX = 0.5 + Math.cos(this.cycleT * TAU) * 0.35;
    const sunScreenY = 0.5 - i.sunElevation * 0.4;

    let idx = 0;
    arr[idx++] = this.width;                             // width
    arr[idx++] = this.height;                            // height
    // Sky zenith
    arr[idx++] = i.skyZenith[0];
    arr[idx++] = i.skyZenith[1];
    arr[idx++] = i.skyZenith[2];
    // Sky horizon
    arr[idx++] = i.skyHorizon[0];
    arr[idx++] = i.skyHorizon[1];
    arr[idx++] = i.skyHorizon[2];
    // Sky nadir
    arr[idx++] = i.skyNadir[0];
    arr[idx++] = i.skyNadir[1];
    arr[idx++] = i.skyNadir[2];
    // Environment light
    arr[idx++] = i.envLight[0];
    arr[idx++] = i.envLight[1];
    arr[idx++] = i.envLight[2];
    arr[idx++] = i.envIntensity * this.cfg.envLightScale;
    // Sun position
    arr[idx++] = sunScreenX;
    arr[idx++] = sunScreenY;
    arr[idx++] = i.sunElevation;
    // Horizon bias & stars
    arr[idx++] = this.cfg.horizonBias;
    arr[idx++] = i.starVis;
    arr[idx++] = this.cfg.starDensity;
    // Fog
    arr[idx++] = i.fogColor[0];
    arr[idx++] = i.fogColor[1];
    arr[idx++] = i.fogColor[2];
    arr[idx++] = lerp(this.cfg.fogDensityRange[0], this.cfg.fogDensityRange[1], i.fogDensity);
    // Time + AO influence
    arr[idx++] = this.elapsed;
    arr[idx++] = Math.min(1, Math.max(0, this.cfg.aoIntensity * i.aoMul));
    arr[idx++] = 0;  // _pad0
    arr[idx++] = 0;  // _pad1

    this.device.queue.writeBuffer(this.compositeUniBuf, 0, arr);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal — bind group management
  // ─────────────────────────────────────────────────────────────────────────

  private _ensureAOBindGroup(fieldView: any /*GPUTextureView*/): void {
    if (this.aoBG && this.lastFieldView === fieldView) return;

    this.aoBG = this.device.createBindGroup({
      label  : 'dynenv-ao-bg',
      layout : this.aoBGL,
      entries: [
        { binding: 0, resource: { buffer: this.aoUniBuf } },
        { binding: 1, resource: fieldView },
        { binding: 2, resource: this.aoSampler },
        { binding: 3, resource: this.aoTexView },
      ],
    });
    this.lastFieldView = fieldView;
  }

  private _ensureCompositeBindGroup(sceneView: any /*GPUTextureView*/, aoView: any /*GPUTextureView*/): void {
    if (this.compositeBG && this.lastSceneView === sceneView && this.lastAoView === aoView) return;

    this.compositeBG = this.device.createBindGroup({
      label  : 'dynenv-composite-bg',
      layout : this.compositeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.compositeUniBuf } },
        { binding: 1, resource: this.compositeSampler },
        { binding: 2, resource: sceneView },
        { binding: 3, resource: aoView },
      ],
    });
    this.lastSceneView = sceneView;
    this.lastAoView    = aoView;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Presets
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Ready-made configurations for common scenarios.
 *
 * REALTIME        — 2-minute full cycle, standard quality AO
 * CINEMATIC       — 4-minute cycle, high AO samples, exaggerated env light
 * FAST_PREVIEW    — 30-second cycle, reduced AO for performance
 * FROZEN_DAWN     — paused at the dawn phase (phaseOffset=0)
 * FROZEN_NIGHT    — paused at the night phase (phaseOffset=0.75)
 * ETERNAL_DAY     — paused at the midday phase
 */
export const DYNAMIC_ENV_PRESETS = {
  REALTIME: {
    cycleDuration  : 120,
    aoIntensity    : 0.55,
    aoSamples      : 8,
    envLightScale  : 1.0,
  },
  CINEMATIC: {
    cycleDuration  : 240,
    aoIntensity    : 0.70,
    aoRadius       : 0.016,
    aoSamples      : 12,
    envLightScale  : 1.3,
    fogDensityRange: [0.10, 0.45] as [number, number],
  },
  FAST_PREVIEW: {
    cycleDuration  : 30,
    aoIntensity    : 0.35,
    aoRadius       : 0.010,
    aoSamples      : 4,
    envLightScale  : 0.9,
  },
  FROZEN_DAWN: {
    paused     : true,
    phaseOffset: 0.0,
    aoIntensity: 0.50,
  },
  FROZEN_NIGHT: {
    paused      : true,
    phaseOffset : 0.75,
    aoIntensity : 0.65,
    starDensity : 0.0012,
    envLightScale: 0.6,
  },
  ETERNAL_DAY: {
    paused      : true,
    phaseOffset : 0.375,  // midway through the day phase
    aoIntensity : 0.55,
    envLightScale: 1.0,
  },
} satisfies Record<string, DynamicEnvironmentConfig>;

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factory
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a DynamicEnvironment with a named preset applied on top.
 *
 * @example
 * ```ts
 * const env = await createDynamicEnvironment(
 *   device, format, canvas.width, canvas.height, 'CINEMATIC',
 * );
 * ```
 */
export async function createDynamicEnvironment(
  device : any /*GPUDevice*/,
  format : GPUTextureFormat,
  width  : number,
  height : number,
  preset?: keyof typeof DYNAMIC_ENV_PRESETS,
): Promise<DynamicEnvironment> {
  const env = await DynamicEnvironment.create(device, format, width, height);
  if (preset) {
    env.configure(DYNAMIC_ENV_PRESETS[preset]);
  }
  return env;
}
