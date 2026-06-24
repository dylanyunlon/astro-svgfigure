/**
 * rain-snow-system.ts — M796: Weather Particle System (Rain + Snow)
 * ─────────────────────────────────────────────────────────────────────────────
 * GPU compute-driven weather particles for rain and snow effects.  A 2-D wind
 * field texture drives lateral drift so particles react coherently to gusts,
 * turbulence, and directional wind.  Rain falls fast with slight lateral
 * streaking; snow drifts slowly with curl-noise wobble.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Pipeline (per frame)
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Compute Pass 0: WIND FIELD UPDATE ──────────────────────────────────────┐
 *   │  windFieldTex (R16G16Float, 64×64)                                       │
 *   │  Procedural wind: base direction + Perlin gusts + time-varying curl.     │
 *   │  The field is a 2-D vector map sampled by each particle to obtain        │
 *   │  local wind acceleration.  Resolution is intentionally low — wind is     │
 *   │  a broad atmospheric effect, not per-pixel.                              │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *                │ windFieldTex
 *                ▼
 *   ┌─ Compute Pass 1: PARTICLE SIMULATE ─────────────────────────────────────┐
 *   │  posA/velA + windFieldTex + uniforms → posB/velB (ping-pong)            │
 *   │                                                                          │
 *   │  Rain:  high terminal velocity (≈ 6–9 m/s), narrow size variance,       │
 *   │         streak length ∝ speed, minimal lateral wobble.                   │
 *   │  Snow:  low terminal velocity (≈ 0.5–2 m/s), wide size variance,       │
 *   │         curl-noise wobble, flutter amplitude modulated by wind speed.    │
 *   │                                                                          │
 *   │  Both types share the same buffer layout and shader; a per-particle      │
 *   │  `kind` flag selects the physics branch.                                 │
 *   │                                                                          │
 *   │  Wind influence: sample windFieldTex at particle UV → acceleration.      │
 *   │  Gravity: constant downward pull (configurable).                         │
 *   │  Damping: kind-dependent air resistance.                                 │
 *   │  Respawn: particles that exit the domain are teleported to the top       │
 *   │           edge with randomised X position, preserving wind-coherent      │
 *   │           entry angles.                                                  │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *                │ posB
 *                ▼
 *   ┌─ Render Pass: DRAW PARTICLES ───────────────────────────────────────────┐
 *   │  Instanced quads — each particle is a screen-aligned rectangle.          │
 *   │  Rain: elongated vertically by velocity magnitude (motion blur).         │
 *   │  Snow: circular soft dot, radius ∝ size attribute.                       │
 *   │  Alpha: fade near domain edges + depth-based atmospheric fade.           │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Design References
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   src/lib/sph/boids-compute.ts             — ping-pong GPU compute pattern
 *   src/lib/sph/dynamic-environment.ts       — sky cycle + env light
 *   src/lib/sph/atmosphere.ts                — fog + scatter post-process
 *   src/lib/sph/environment-fog.ts           — depth fog composite
 *   src/lib/sph/curl-flow-field.ts           — curl noise generation
 *   src/lib/sph/noise-flow-field.ts          — Perlin noise overlay
 *   src/lib/sph/particle-effect-system.ts    — GPU particle stride patterns
 *   upstream/lygia/generative/snoise.wgsl    — simplex noise reference
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Quick Start
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   const weather = await RainSnowSystem.create(device, presentFormat, w, h);
 *   weather.configure({ mode: 'rain', intensity: 0.7, windAngle: -0.3 });
 *
 *   // per frame:
 *   weather.tick(dt);
 *   weather.render(encoder, dstView);
 *
 *   // switch to snow mid-scene:
 *   weather.configure({ mode: 'snow', intensity: 0.5, windAngle: 0.1 });
 *
 *   // blend rain + snow simultaneously:
 *   weather.configure({ mode: 'mixed', rainRatio: 0.6 });
 *
 * Research: xiaodi #M796 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────









export const WEATHER_WORKGROUP    = 256;
export const WEATHER_MAX          = 32768;
export const WIND_FIELD_SIZE      = 64;    // 64×64 texels
export const WIND_WORKGROUP       = 8;     // 8×8 threads per workgroup

// Per-particle stride: posX, posY, velX, velY, size, alpha, life, kind
export const PARTICLE_STRIDE      = 8;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** Weather particle type. */
export type WeatherKind = 0 | 1;  // 0 = rain, 1 = snow
export const RAIN: WeatherKind = 0;
export const SNOW: WeatherKind = 1;

/** Weather mode. */
export type WeatherMode = 'rain' | 'snow' | 'mixed' | 'off';

/** Configuration for the weather system. */
export interface RainSnowConfig {
  /** Weather mode: rain-only, snow-only, mixed, or off. */
  mode: WeatherMode;
  /** Particle count (clamped to WEATHER_MAX). Default 8192. */
  count?: number;
  /** Intensity [0, 1] — scales opacity, density, and spawn rate. Default 0.6. */
  intensity?: number;
  /** Wind direction angle in radians. 0 = rightward, -π/2 = downward. Default -0.2. */
  windAngle?: number;
  /** Wind base speed in world units / second. Default 1.5. */
  windSpeed?: number;
  /** Wind gust amplitude — random variation on top of base. Default 0.8. */
  gustAmplitude?: number;
  /** Gust frequency (Hz). Default 0.3. */
  gustFrequency?: number;
  /** Gravity strength (positive = downward). Default 9.8. */
  gravity?: number;
  /** Domain width in world units. Default 2.0 (NDC). */
  domainW?: number;
  /** Domain height in world units. Default 2.0 (NDC). */
  domainH?: number;
  /** Rain-to-snow ratio when mode='mixed'. Default 0.5. */
  rainRatio?: number;
  /** Rain terminal velocity. Default 7.0. */
  rainTerminalVel?: number;
  /** Snow terminal velocity. Default 1.2. */
  snowTerminalVel?: number;
  /** Snow curl-noise wobble amplitude. Default 0.4. */
  snowWobble?: number;
  /** Global alpha multiplier. Default 0.7. */
  globalAlpha?: number;
  /** Rain streak length multiplier. Default 0.06. */
  rainStreakLength?: number;
  /** Snow base radius in NDC. Default 0.004. */
  snowBaseRadius?: number;
}

/** Snapshot for querying from other systems. */
export interface WeatherSnapshot {
  mode: WeatherMode;
  windAngleRad: number;
  windSpeedCurrent: number;
  activeCount: number;
  elapsedTime: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Defaults
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULTS: Required<RainSnowConfig> = {
  mode:             'rain',
  count:            8192,
  intensity:        0.6,
  windAngle:        -0.2,
  windSpeed:        1.5,
  gustAmplitude:    0.8,
  gustFrequency:    0.3,
  gravity:          9.8,
  domainW:          2.0,
  domainH:          2.0,
  rainRatio:        0.5,
  rainTerminalVel:  7.0,
  snowTerminalVel:  1.2,
  snowWobble:       0.4,
  globalAlpha:      0.7,
  rainStreakLength:  0.06,
  snowBaseRadius:   0.004,
};

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — shared noise utilities
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_NOISE = /* wgsl */`
// Simplex-ish 2D noise (hash-based, good enough for wind)
fn hash2(p: vec2<f32>) -> vec2<f32> {
  var q = vec2<f32>(
    dot(p, vec2<f32>(127.1, 311.7)),
    dot(p, vec2<f32>(269.5, 183.3))
  );
  return fract(sin(q) * 43758.5453) * 2.0 - 1.0;
}

fn snoise2(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);

  let a = dot(hash2(i + vec2<f32>(0.0, 0.0)), f - vec2<f32>(0.0, 0.0));
  let b = dot(hash2(i + vec2<f32>(1.0, 0.0)), f - vec2<f32>(1.0, 0.0));
  let c = dot(hash2(i + vec2<f32>(0.0, 1.0)), f - vec2<f32>(0.0, 1.0));
  let d = dot(hash2(i + vec2<f32>(1.0, 1.0)), f - vec2<f32>(1.0, 1.0));

  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Curl of scalar noise field → divergence-free 2D vector
fn curlNoise2(p: vec2<f32>) -> vec2<f32> {
  let eps = 0.01;
  let dx = snoise2(p + vec2<f32>(eps, 0.0)) - snoise2(p - vec2<f32>(eps, 0.0));
  let dy = snoise2(p + vec2<f32>(0.0, eps)) - snoise2(p - vec2<f32>(0.0, eps));
  return vec2<f32>(dy, -dx) / (2.0 * eps);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — shared uniform struct
// ─────────────────────────────────────────────────────────────────────────────

const UNIFORMS_WGSL = /* wgsl */`
struct WeatherUniforms {
  count             : u32,     //  0
  mode              : u32,     //  4   0=rain, 1=snow, 2=mixed
  _pad0             : u32,     //  8
  _pad1             : u32,     // 12
  dt                : f32,     // 16
  time              : f32,     // 20
  gravity           : f32,     // 24
  intensity         : f32,     // 28
  windAngle         : f32,     // 32
  windSpeed         : f32,     // 36
  gustAmplitude     : f32,     // 40
  gustFrequency     : f32,     // 44
  domainW           : f32,     // 48
  domainH           : f32,     // 52
  rainTerminalVel   : f32,     // 56
  snowTerminalVel   : f32,     // 60
  snowWobble        : f32,     // 64
  globalAlpha       : f32,     // 68
  rainRatio         : f32,     // 72
  rainStreakLength   : f32,     // 76
  snowBaseRadius    : f32,     // 80
  _pad2             : f32,     // 84
  _pad3             : f32,     // 88
  _pad4             : f32,     // 92
}
`;

// Byte offsets — must mirror the struct above (std140, 16-byte rows)
const U_COUNT               =  0;
const U_MODE                =  4;
const U_DT                  = 16;
const U_TIME                = 20;
const U_GRAVITY             = 24;
const U_INTENSITY           = 28;
const U_WIND_ANGLE          = 32;
const U_WIND_SPEED          = 36;
const U_GUST_AMP            = 40;
const U_GUST_FREQ           = 44;
const U_DOMAIN_W            = 48;
const U_DOMAIN_H            = 52;
const U_RAIN_TERMINAL       = 56;
const U_SNOW_TERMINAL       = 60;
const U_SNOW_WOBBLE         = 64;
const U_GLOBAL_ALPHA        = 68;
const U_RAIN_RATIO          = 72;
const U_RAIN_STREAK         = 76;
const U_SNOW_RADIUS         = 80;
const UNIFORMS_BYTE_SIZE    = 96;   // 6 × vec4 rows

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Compute Pass 0: wind field update
// ─────────────────────────────────────────────────────────────────────────────

const WIND_FIELD_SHADER = /* wgsl */`
${WGSL_NOISE}
${UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform> u : WeatherUniforms;
@group(1) @binding(0) var windField : texture_storage_2d<rg32float, write>;

@compute @workgroup_size(${WIND_WORKGROUP}, ${WIND_WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = ${WIND_FIELD_SIZE}u;
  if (gid.x >= size || gid.y >= size) { return; }

  // Normalised UV [0,1]
  let uv = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5) / f32(size);

  // Base wind direction from angle
  let baseDir = vec2<f32>(cos(u.windAngle), sin(u.windAngle));

  // Perlin gust: large-scale spatial + temporal variation
  let gustPhase = u.time * u.gustFrequency;
  let gustSample = snoise2(uv * 3.0 + vec2<f32>(gustPhase, gustPhase * 0.7));
  let gustVec = baseDir * (1.0 + gustSample * u.gustAmplitude);

  // Curl turbulence: small-scale swirls (divergence-free so it looks physical)
  let curlP = uv * 5.0 + vec2<f32>(u.time * 0.15, u.time * 0.12);
  let curl = curlNoise2(curlP) * 0.35;

  // Combined wind vector
  let wind = gustVec * u.windSpeed + curl;

  textureStore(windField, vec2<i32>(gid.xy), vec4<f32>(wind, 0.0, 0.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Compute Pass 1: particle simulation
// ─────────────────────────────────────────────────────────────────────────────

const SIMULATE_SHADER = /* wgsl */`
${WGSL_NOISE}
${UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform> u : WeatherUniforms;

// Ping-pong particle buffers (SoA: stride = PARTICLE_STRIDE floats per particle)
// Layout per particle: [posX, posY, velX, velY, size, alpha, life, kind]
@group(1) @binding(0) var<storage, read>       particlesIn  : array<f32>;
@group(1) @binding(1) var<storage, read_write> particlesOut : array<f32>;

// Wind field texture (read via textureLoad)
@group(2) @binding(0) var windField : texture_2d<f32>;

// PCG-style hash for per-particle randomness
fn pcgHash(inp: u32) -> u32 {
  var state = inp * 747796405u + 2891336453u;
  var word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn randFloat(seed: u32) -> f32 {
  return f32(pcgHash(seed)) / 4294967295.0;
}

@compute @workgroup_size(${WEATHER_WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }

  let stride = ${PARTICLE_STRIDE}u;
  let base   = i * stride;

  // Read current state
  var px    = particlesIn[base + 0u];
  var py    = particlesIn[base + 1u];
  var vx    = particlesIn[base + 2u];
  var vy    = particlesIn[base + 3u];
  let size  = particlesIn[base + 4u];
  var alpha = particlesIn[base + 5u];
  var life  = particlesIn[base + 6u];
  let kind  = particlesIn[base + 7u];  // 0.0 = rain, 1.0 = snow

  let isSnow = kind > 0.5;
  let dt     = u.dt;

  // ── Sample wind field at particle position ──────────────────────────────
  let uvX = clamp((px / u.domainW + 0.5), 0.0, 1.0);
  let uvY = clamp((py / u.domainH + 0.5), 0.0, 1.0);
  let fieldSize = ${WIND_FIELD_SIZE};
  let texCoord = vec2<i32>(
    i32(uvX * f32(fieldSize - 1)),
    i32(uvY * f32(fieldSize - 1))
  );
  let windSample = textureLoad(windField, texCoord, 0);
  let windAccel  = windSample.xy;

  // ── Physics integration (Verlet-ish: semi-implicit Euler) ──────────────

  if (isSnow) {
    // Snow: low gravity, high drag, curl wobble
    let gravityForce = vec2<f32>(0.0, -u.gravity * 0.08);
    let drag         = 3.5;  // heavy air resistance
    let termVel      = u.snowTerminalVel;

    // Curl wobble — gives snowflakes their characteristic lateral drift
    let wobbleP = vec2<f32>(px * 2.0 + u.time * 0.8, py * 2.0 + u.time * 0.5);
    let wobble  = curlNoise2(wobbleP) * u.snowWobble;

    // Acceleration: gravity + wind + wobble - drag
    let ax = windAccel.x * 0.7 + wobble.x - vx * drag;
    let ay = gravityForce.y + windAccel.y * 0.3 + wobble.y - vy * drag;

    vx += ax * dt;
    vy += ay * dt;

    // Clamp to terminal velocity
    let speed = sqrt(vx * vx + vy * vy);
    if (speed > termVel) {
      let scale = termVel / speed;
      vx *= scale;
      vy *= scale;
    }
  } else {
    // Rain: high gravity, moderate drag, minimal lateral movement
    let gravityForce = vec2<f32>(0.0, -u.gravity);
    let drag         = 1.2;
    let termVel      = u.rainTerminalVel;

    let ax = windAccel.x * 0.4 - vx * drag;
    let ay = gravityForce.y + windAccel.y * 0.15 - vy * drag;

    vx += ax * dt;
    vy += ay * dt;

    // Clamp to terminal velocity (mainly vertical)
    let speed = abs(vy);
    if (speed > termVel) {
      vy = -termVel;  // force downward
    }
  }

  // ── Position update ─────────────────────────────────────────────────────
  px += vx * dt;
  py += vy * dt;

  // ── Life decay ──────────────────────────────────────────────────────────
  life -= dt;

  // ── Domain wrap / respawn ───────────────────────────────────────────────
  let halfW = u.domainW * 0.5;
  let halfH = u.domainH * 0.5;

  var needRespawn = life <= 0.0
                 || py < -halfH   // fell below bottom
                 || px < -halfW - 0.1
                 || px >  halfW + 0.1;

  if (needRespawn) {
    // Respawn at top with randomised position
    let seed = i * 1973u + u32(u.time * 1000.0);
    px   = (randFloat(seed) - 0.5) * u.domainW;
    py   = halfH + randFloat(seed + 1u) * 0.3;       // slightly above top edge
    vx   = windAccel.x * 0.3;                         // inherit current wind
    vy   = select(-u.rainTerminalVel * 0.3, -u.snowTerminalVel * 0.2, isSnow);
    life = 3.0 + randFloat(seed + 2u) * 4.0;           // 3–7 seconds
    alpha = u.intensity * u.globalAlpha;
  } else {
    // Fade near edges
    let edgeFadeX = smoothstep(0.0, 0.15, halfW - abs(px));
    let edgeFadeY = smoothstep(0.0, 0.15, py + halfH);  // fade near bottom
    let topFade   = smoothstep(0.0, 0.1, halfH - py);   // fade near top
    alpha = u.intensity * u.globalAlpha * edgeFadeX * edgeFadeY * topFade;
  }

  // ── Write output ────────────────────────────────────────────────────────
  particlesOut[base + 0u] = px;
  particlesOut[base + 1u] = py;
  particlesOut[base + 2u] = vx;
  particlesOut[base + 3u] = vy;
  particlesOut[base + 4u] = size;
  particlesOut[base + 5u] = alpha;
  particlesOut[base + 6u] = life;
  particlesOut[base + 7u] = kind;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Render: instanced quads (vertex + fragment)
// ─────────────────────────────────────────────────────────────────────────────

const RENDER_VERTEX_WGSL = /* wgsl */`
${UNIFORMS_WGSL}

struct VertexOut {
  @builtin(position) pos     : vec4<f32>,
  @location(0)       localUV : vec2<f32>,
  @location(1)       alpha   : f32,
  @location(2)       kind    : f32,
  @location(3)       speed   : f32,
}

@group(0) @binding(0) var<uniform> u : WeatherUniforms;
@group(1) @binding(0) var<storage, read> particles : array<f32>;

// Unit quad: 6 vertices (2 triangles) — positions in [-0.5, 0.5]
var<private> QUAD_POS : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
  vec2<f32>(-0.5, -0.5),
  vec2<f32>( 0.5, -0.5),
  vec2<f32>( 0.5,  0.5),
  vec2<f32>(-0.5, -0.5),
  vec2<f32>( 0.5,  0.5),
  vec2<f32>(-0.5,  0.5),
);

@vertex
fn main(
  @builtin(vertex_index)   vertIdx : u32,
  @builtin(instance_index) instIdx : u32,
) -> VertexOut {
  let stride = ${PARTICLE_STRIDE}u;
  let base   = instIdx * stride;

  let px    = particles[base + 0u];
  let py    = particles[base + 1u];
  let vx    = particles[base + 2u];
  let vy    = particles[base + 3u];
  let size  = particles[base + 4u];
  let alpha = particles[base + 5u];
  let kind  = particles[base + 7u];

  let speed = sqrt(vx * vx + vy * vy);
  let isSnow = kind > 0.5;

  // Quad local position
  let lp = QUAD_POS[vertIdx % 6u];

  // Scale: rain is elongated along velocity, snow is circular
  var scaleX : f32;
  var scaleY : f32;

  if (isSnow) {
    let r = u.snowBaseRadius * size;
    scaleX = r;
    scaleY = r;
  } else {
    // Rain streak: thin horizontally, stretched vertically by speed
    scaleX = 0.001;
    scaleY = u.rainStreakLength * (1.0 + speed * 0.3);
  }

  // For rain, rotate quad to align with velocity vector
  var worldPos : vec2<f32>;
  if (!isSnow && speed > 0.01) {
    let dir = vec2<f32>(vx, vy) / speed;
    let perp = vec2<f32>(-dir.y, dir.x);
    worldPos = vec2<f32>(px, py)
             + dir  * lp.y * scaleY
             + perp * lp.x * scaleX;
  } else {
    worldPos = vec2<f32>(px + lp.x * scaleX, py + lp.y * scaleY);
  }

  // Map from world space [-domainW/2, domainW/2] to NDC [-1, 1]
  let ndcX = worldPos.x / (u.domainW * 0.5);
  let ndcY = worldPos.y / (u.domainH * 0.5);

  var out : VertexOut;
  out.pos     = vec4<f32>(ndcX, ndcY, 0.0, 1.0);
  out.localUV = lp + 0.5;  // [0, 1]
  out.alpha   = alpha;
  out.kind    = kind;
  out.speed   = speed;
  return out;
}
`;

const RENDER_FRAGMENT_WGSL = /* wgsl */`
struct FragIn {
  @location(0) localUV : vec2<f32>,
  @location(1) alpha   : f32,
  @location(2) kind    : f32,
  @location(3) speed   : f32,
}

@fragment
fn main(in: FragIn) -> @location(0) vec4<f32> {
  let isSnow = in.kind > 0.5;
  let uv     = in.localUV;

  if (isSnow) {
    // Soft circular dot
    let d = length(uv - vec2<f32>(0.5));
    let softEdge = 1.0 - smoothstep(0.3, 0.5, d);
    if (softEdge < 0.01) { discard; }
    // Snow is white with slight blue tint
    let col = vec3<f32>(0.90, 0.93, 1.0);
    return vec4<f32>(col, in.alpha * softEdge * 0.85);
  } else {
    // Rain streak: bright white/light-blue, sharp vertically, soft at tips
    let tipFade = smoothstep(0.0, 0.15, uv.y) * smoothstep(1.0, 0.85, uv.y);
    let centerFade = 1.0 - abs(uv.x - 0.5) * 2.0;
    let mask = tipFade * max(centerFade, 0.0);
    if (mask < 0.01) { discard; }
    // Rain colour: very pale blue-white, brighter at high speed
    let brightness = 0.7 + clamp(in.speed * 0.05, 0.0, 0.3);
    let col = vec3<f32>(0.75, 0.82, 0.95) * brightness;
    return vec4<f32>(col, in.alpha * mask * 0.9);
  }
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Class
// ─────────────────────────────────────────────────────────────────────────────

export class RainSnowSystem {
  // ── GPU handles ──────────────────────────────────────────────────────────
  private device!:        GPUDevice;
  private presentFormat!: GPUTextureFormat;

  // Uniform
  private uniformBuf!:    GPUBuffer;
  private uniformBGL!:    GPUBindGroupLayout;
  private uniformBG!:     GPUBindGroup;

  // Particle ping-pong (single flat f32 array per side)
  private particleBuf:    GPUBuffer[] = [];
  private particleSrcBGL!: GPUBindGroupLayout;
  private particleDstBGL!: GPUBindGroupLayout;
  private simSrcBG:       (GPUBindGroup | null)[] = [null, null];
  private simDstBG:       (GPUBindGroup | null)[] = [null, null];

  // Render read-only bind group
  private renderParticleBGL!: GPUBindGroupLayout;
  private renderParticleBG:   (GPUBindGroup | null)[] = [null, null];

  // Wind field (rg32float storage texture)
  private windTex!:        GPUTexture;
  private windTexView!:    GPUTextureView;
  private windStoreBGL!:   GPUBindGroupLayout;
  private windStoreBG!:    GPUBindGroup;
  private windReadBGL!:    GPUBindGroupLayout;
  private windReadBG!:     GPUBindGroup;

  // Compute pipelines
  private windPipeline!:     GPUComputePipeline;
  private simulatePipeline!: GPUComputePipeline;

  // Render pipeline
  private renderPipeline!: GPURenderPipeline;

  // State
  private ping        = 0;
  private elapsed     = 0;
  private destroyed   = false;
  private cfg!:       Required<RainSnowConfig>;
  private width       = 0;
  private height      = 0;

  // ── Factory ──────────────────────────────────────────────────────────────

  static async create(
    device:        GPUDevice,
    presentFormat: GPUTextureFormat,
    width:         number,
    height:        number,
    config?:       Partial<RainSnowConfig>,
  ): Promise<RainSnowSystem> {
    const sys = new RainSnowSystem();
    sys.device        = device;
    sys.presentFormat = presentFormat;
    sys.width         = width;
    sys.height        = height;
    sys.cfg           = { ...DEFAULTS, ...config };
    sys.cfg.count     = Math.min(sys.cfg.count, WEATHER_MAX);

    sys.allocateBuffers();
    sys.buildPipelines();
    sys.buildBindGroups();
    sys.initParticles();

    return sys;
  }

  // ── Public API ──────────────────────────────────────────────────────────

  /** Update config at runtime (partial patches allowed). */
  configure(patch: Partial<RainSnowConfig>): void {
    Object.assign(this.cfg, patch);
    if (this.cfg.count > WEATHER_MAX) this.cfg.count = WEATHER_MAX;
    // If count changed we need to re-init particles
    if (patch.count !== undefined) {
      this.initParticles();
    }
    this.writeUniforms();
  }

  /** Advance simulation by dt seconds. Encodes wind + sim compute passes. */
  tick(dt: number): void {
    if (this.destroyed || this.cfg.mode === 'off') return;
    this.elapsed += dt;
    this.cfg = { ...this.cfg };  // trigger uniform write
    this.writeUniforms();

    const cmd = this.device.createCommandEncoder({ label: 'weather:tick' });
    const n   = this.cfg.count;
    const p   = this.ping;
    const q   = 1 - p;

    // Pass 0 — wind field
    {
      const pass = cmd.beginComputePass({ label: 'weather:wind' });
      pass.setPipeline(this.windPipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.windStoreBG);
      const disp = Math.ceil(WIND_FIELD_SIZE / WIND_WORKGROUP);
      pass.dispatchWorkgroups(disp, disp);
      pass.end();
    }

    // Pass 1 — simulate particles
    {
      const pass = cmd.beginComputePass({ label: 'weather:simulate' });
      pass.setPipeline(this.simulatePipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.simSrcBG[p]!);
      pass.setBindGroup(2, this.windReadBG);
      const disp = Math.ceil(n / WEATHER_WORKGROUP);
      pass.dispatchWorkgroups(disp);
      pass.end();
    }

    this.device.queue.submit([cmd.finish()]);
    this.ping = q;
  }

  /** Render weather particles into the given render target view. */
  render(
    encoder: GPUCommandEncoder,
    dstView: GPUTextureView,
    loadOp:  GPULoadOp = 'load',
  ): void {
    if (this.destroyed || this.cfg.mode === 'off') return;

    const p = this.ping;  // current source (just written by tick)
    const pass = encoder.beginRenderPass({
      label: 'weather:render',
      colorAttachments: [{
        view:       dstView,
        loadOp,
        storeOp:    'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
    });

    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.uniformBG);
    pass.setBindGroup(1, this.renderParticleBG[p]!);
    pass.draw(6, this.cfg.count);   // 6 verts per quad × count instances
    pass.end();
  }

  /** Query current state for cross-system coordination. */
  snapshot(): WeatherSnapshot {
    return {
      mode:             this.cfg.mode,
      windAngleRad:     this.cfg.windAngle,
      windSpeedCurrent: this.cfg.windSpeed,
      activeCount:      this.cfg.count,
      elapsedTime:      this.elapsed,
    };
  }

  /** Resize the render target dimensions. */
  resize(w: number, h: number): void {
    this.width  = w;
    this.height = h;
  }

  /** Release all GPU resources. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.particleBuf[0].destroy();
    this.particleBuf[1].destroy();
    this.uniformBuf.destroy();
    this.windTex.destroy();
  }

  // ── Internals ───────────────────────────────────────────────────────────

  private allocateBuffers(): void {
    const n      = WEATHER_MAX;
    const bytes  = n * PARTICLE_STRIDE * 4;

    // Ping-pong particle buffers
    for (let i = 0; i < 2; i++) {
      this.particleBuf[i] = this.device.createBuffer({
        label: `weather:particles:${i}`,
        size:  bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }

    // Uniform buffer
    this.uniformBuf = this.device.createBuffer({
      label: 'weather:uniforms',
      size:  UNIFORMS_BYTE_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Wind field storage texture
    this.windTex = this.device.createTexture({
      label:  'weather:windField',
      size:   [WIND_FIELD_SIZE, WIND_FIELD_SIZE],
      format: 'rg32float',
      usage:  GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.windTexView = this.windTex.createView({ label: 'weather:windFieldView' });
  }

  private buildPipelines(): void {
    const device = this.device;

    // ── Bind-group layouts ────────────────────────────────────────────────

    this.uniformBGL = device.createBindGroupLayout({
      label:   'weather:uniformBGL',
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });

    // Sim source: read + read_write (ping → pong)
    this.particleSrcBGL = device.createBindGroupLayout({
      label:   'weather:particleSrcBGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    // Wind field — storage texture write
    this.windStoreBGL = device.createBindGroupLayout({
      label:   'weather:windStoreBGL',
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rg32float', viewDimension: '2d' } }],
    });

    // Wind field — texture read (for simulate pass)
    this.windReadBGL = device.createBindGroupLayout({
      label:   'weather:windReadBGL',
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } }],
    });

    // Render: read-only particle storage
    this.renderParticleBGL = device.createBindGroupLayout({
      label:   'weather:renderParticleBGL',
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }],
    });

    // ── Shader modules ───────────────────────────────────────────────────

    const windSM     = device.createShaderModule({ label: 'weather:wind:sm',     code: WIND_FIELD_SHADER });
    const simSM      = device.createShaderModule({ label: 'weather:sim:sm',      code: SIMULATE_SHADER });
    const vertSM     = device.createShaderModule({ label: 'weather:vert:sm',     code: RENDER_VERTEX_WGSL });
    const fragSM     = device.createShaderModule({ label: 'weather:frag:sm',     code: RENDER_FRAGMENT_WGSL });

    // ── Compute pipelines ────────────────────────────────────────────────

    const windLayout = device.createPipelineLayout({
      label:            'weather:wind:layout',
      bindGroupLayouts: [this.uniformBGL, this.windStoreBGL],
    });

    const simLayout = device.createPipelineLayout({
      label:            'weather:sim:layout',
      bindGroupLayouts: [this.uniformBGL, this.particleSrcBGL, this.windReadBGL],
    });

    this.windPipeline = device.createComputePipeline({
      label:   'weather:wind:pipeline',
      layout:  windLayout,
      compute: { module: windSM, entryPoint: 'main' },
    });

    this.simulatePipeline = device.createComputePipeline({
      label:   'weather:sim:pipeline',
      layout:  simLayout,
      compute: { module: simSM, entryPoint: 'main' },
    });

    // ── Render pipeline ──────────────────────────────────────────────────

    const renderLayout = device.createPipelineLayout({
      label:            'weather:render:layout',
      bindGroupLayouts: [this.uniformBGL, this.renderParticleBGL],
    });

    this.renderPipeline = device.createRenderPipeline({
      label:  'weather:render:pipeline',
      layout: renderLayout,
      vertex: {
        module:     vertSM,
        entryPoint: 'main',
      },
      fragment: {
        module:     fragSM,
        entryPoint: 'main',
        targets: [{
          format: this.presentFormat,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  private buildBindGroups(): void {
    const device = this.device;

    // Uniform
    this.uniformBG = device.createBindGroup({
      label:   'weather:uniformBG',
      layout:  this.uniformBGL,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuf } }],
    });

    // Sim ping-pong (p reads from buf[p], writes to buf[1-p])
    for (let p = 0; p < 2; p++) {
      const q = 1 - p;
      this.simSrcBG[p] = device.createBindGroup({
        label:   `weather:simSrcBG:${p}`,
        layout:  this.particleSrcBGL,
        entries: [
          { binding: 0, resource: { buffer: this.particleBuf[p] } },
          { binding: 1, resource: { buffer: this.particleBuf[q] } },
        ],
      });
    }

    // Wind store (write-only storage texture)
    this.windStoreBG = device.createBindGroup({
      label:   'weather:windStoreBG',
      layout:  this.windStoreBGL,
      entries: [{ binding: 0, resource: this.windTexView }],
    });

    // Wind read (texture)
    this.windReadBG = device.createBindGroup({
      label:   'weather:windReadBG',
      layout:  this.windReadBGL,
      entries: [{ binding: 0, resource: this.windTexView }],
    });

    // Render read-only
    for (let p = 0; p < 2; p++) {
      this.renderParticleBG[p] = device.createBindGroup({
        label:   `weather:renderBG:${p}`,
        layout:  this.renderParticleBGL,
        entries: [{ binding: 0, resource: { buffer: this.particleBuf[p] } }],
      });
    }
  }

  /** Populate initial particle data on CPU and upload to both ping-pong buffers. */
  private initParticles(): void {
    const n    = this.cfg.count;
    const data = new Float32Array(n * PARTICLE_STRIDE);

    for (let i = 0; i < n; i++) {
      const base = i * PARTICLE_STRIDE;

      // Assign kind based on mode
      let kind: number;
      if (this.cfg.mode === 'rain')       kind = 0;
      else if (this.cfg.mode === 'snow')  kind = 1;
      else                                kind = Math.random() < this.cfg.rainRatio ? 0 : 1;

      const isSnow = kind === 1;

      // Random initial position across domain
      const px = (Math.random() - 0.5) * this.cfg.domainW;
      const py = (Math.random() - 0.5) * this.cfg.domainH;

      // Initial velocity: slight downward
      const vx = 0;
      const vy = isSnow ? -this.cfg.snowTerminalVel * 0.5 : -this.cfg.rainTerminalVel * 0.5;

      // Size: snow has wider variance
      const size = isSnow
        ? 0.6 + Math.random() * 0.8   // [0.6, 1.4]
        : 0.8 + Math.random() * 0.4;  // [0.8, 1.2]

      const alpha = this.cfg.intensity * this.cfg.globalAlpha;
      const life  = 2.0 + Math.random() * 5.0;

      data[base + 0] = px;
      data[base + 1] = py;
      data[base + 2] = vx;
      data[base + 3] = vy;
      data[base + 4] = size;
      data[base + 5] = alpha;
      data[base + 6] = life;
      data[base + 7] = kind;
    }

    // Upload to both buffers
    this.device.queue.writeBuffer(this.particleBuf[0], 0, data);
    this.device.queue.writeBuffer(this.particleBuf[1], 0, data);
    this.writeUniforms();
  }

  private writeUniforms(): void {
    const buf = new ArrayBuffer(UNIFORMS_BYTE_SIZE);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);

    const modeMap: Record<WeatherMode, number> = { rain: 0, snow: 1, mixed: 2, off: 0 };

    u32[U_COUNT / 4]           = this.cfg.count;
    u32[U_MODE / 4]            = modeMap[this.cfg.mode];
    f32[U_DT / 4]              = 1 / 60;  // fixed dt; caller's dt used for elapsed
    f32[U_TIME / 4]            = this.elapsed;
    f32[U_GRAVITY / 4]         = this.cfg.gravity;
    f32[U_INTENSITY / 4]       = this.cfg.intensity;
    f32[U_WIND_ANGLE / 4]      = this.cfg.windAngle;
    f32[U_WIND_SPEED / 4]      = this.cfg.windSpeed;
    f32[U_GUST_AMP / 4]        = this.cfg.gustAmplitude;
    f32[U_GUST_FREQ / 4]       = this.cfg.gustFrequency;
    f32[U_DOMAIN_W / 4]        = this.cfg.domainW;
    f32[U_DOMAIN_H / 4]        = this.cfg.domainH;
    f32[U_RAIN_TERMINAL / 4]   = this.cfg.rainTerminalVel;
    f32[U_SNOW_TERMINAL / 4]   = this.cfg.snowTerminalVel;
    f32[U_SNOW_WOBBLE / 4]     = this.cfg.snowWobble;
    f32[U_GLOBAL_ALPHA / 4]    = this.cfg.globalAlpha;
    f32[U_RAIN_RATIO / 4]      = this.cfg.rainRatio;
    f32[U_RAIN_STREAK / 4]     = this.cfg.rainStreakLength;
    f32[U_SNOW_RADIUS / 4]     = this.cfg.snowBaseRadius;

    this.device.queue.writeBuffer(this.uniformBuf, 0, buf);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Presets
// ─────────────────────────────────────────────────────────────────────────────

export const WEATHER_PRESETS: Record<string, Partial<RainSnowConfig>> = {
  /** Light drizzle — sparse, slow rain. */
  drizzle: {
    mode:            'rain',
    count:           2048,
    intensity:       0.35,
    windSpeed:       0.6,
    gustAmplitude:   0.3,
    rainTerminalVel: 4.0,
    globalAlpha:     0.5,
  },
  /** Heavy downpour — dense, fast rain with strong wind. */
  downpour: {
    mode:            'rain',
    count:           24000,
    intensity:       0.95,
    windAngle:       -0.4,
    windSpeed:       3.5,
    gustAmplitude:   1.5,
    gustFrequency:   0.5,
    rainTerminalVel: 9.0,
    globalAlpha:     0.85,
    rainStreakLength: 0.09,
  },
  /** Gentle snowfall — sparse, calm. */
  gentleSnow: {
    mode:            'snow',
    count:           4096,
    intensity:       0.5,
    windSpeed:       0.4,
    gustAmplitude:   0.2,
    snowTerminalVel: 0.8,
    snowWobble:      0.5,
    globalAlpha:     0.65,
    snowBaseRadius:  0.005,
  },
  /** Blizzard — dense snow with high wind. */
  blizzard: {
    mode:            'snow',
    count:           20000,
    intensity:       0.9,
    windAngle:       -0.6,
    windSpeed:       4.0,
    gustAmplitude:   2.0,
    gustFrequency:   0.6,
    snowTerminalVel: 2.5,
    snowWobble:      0.8,
    globalAlpha:     0.8,
    snowBaseRadius:  0.006,
  },
  /** Wintry mix — rain and snow together. */
  wintryMix: {
    mode:            'mixed',
    count:           12000,
    intensity:       0.6,
    rainRatio:       0.4,
    windSpeed:       1.8,
    gustAmplitude:   0.7,
    rainTerminalVel: 5.5,
    snowTerminalVel: 1.0,
    snowWobble:      0.35,
    globalAlpha:     0.7,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Factory helper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a RainSnowSystem from a preset name.
 *
 * @example
 *   const weather = await createWeatherSystem(device, format, w, h, 'downpour');
 */
export async function createWeatherSystem(
  device:        GPUDevice,
  presentFormat: GPUTextureFormat,
  width:         number,
  height:        number,
  preset?:       keyof typeof WEATHER_PRESETS,
): Promise<RainSnowSystem> {
  const cfg = preset ? WEATHER_PRESETS[preset] : {};
  return RainSnowSystem.create(device, presentFormat, width, height, cfg);
}
