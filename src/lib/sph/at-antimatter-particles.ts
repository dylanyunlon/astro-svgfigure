/**
 * at-antimatter-particles.ts — M913
 * AT Antimatter Particle System — real WebGL gl.* GPU implementation
 * Ported from AT compiled.vs:
 *   AntimatterSpawn.fs, ProtonAntimatter.fs, ProtonAntimatterLifecycle.fs,
 *   AntimatterPosition.vs, AntimatterBasicFrag.fs, antimatter.glsl, range.glsl
 *
 * Architecture:
 *   - ping-pong FBO for particle position/life state (GPGPU)
 *   - transform-feedback-style update via fragment shader to float textures
 *   - point sprite render with AntimatterPosition.vs / ProtonAntimatter.fs
 *   - boundary emitter system with spray distribution
 *   - SPH density bridge for cell handoff
 */

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────









const MAX_PARTICLES  = 65536 as const;
const TEX_W          = 256   as const;   // sqrt(MAX_PARTICLES)
const TEX_H          = 256   as const;
const MAX_EMITTERS   = 128   as const;
const PARTICLE_STRIDE = 20   as const;
const EMITTER_STRIDE = 12    as const;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

export interface CellBoundaryFace {
  faceId:   string;
  cellId:   string;
  targetId: string;
  origin:   { x: number; y: number };
  normal:   { x: number; y: number };
  halfLen:  number;
  species?: number;
}

export interface ATAntimatterConfig {
  decay?:           number;
  decayRandom?:     [number, number];
  curlScale?:       number;
  curlSpeed?:       number;
  curlStrength?:    number;
  originStrength?:  number;
  hz?:              number;
  dpr?:             number;
  uSize?:           number;
  sprayStrength?:   number;
  spraySpread?:     number;
  sphSmoothingH?:   number;
  sphRestDensity?:  number;
  onHandoff?: (
    cellId:   string,
    targetId: string,
    x: number, y: number,
    vx: number, vy: number,
    species: number,
    density: number,
  ) => void;
}

export interface EmitterRequest {
  face:      CellBoundaryFace;
  count:     number;
  strength?: number;
  life?:     number;
}

const DEFAULTS = {
  decay:          0.005,
  decayRandom:    [0.5, 1.5] as [number, number],
  curlScale:      0.5,
  curlSpeed:      0.3,
  curlStrength:   0.02,
  originStrength: 0.05,
  hz:             60,
  dpr:            1,
  uSize:          0.02,
  sprayStrength:  2.0,
  spraySpread:    Math.PI / 4,
  sphSmoothingH:  0.04,
  sphRestDensity: 1000,
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: range.glsl  (from compiled.vs line 2131)
// ─────────────────────────────────────────────────────────────────────────────

const RANGE_GLSL = /* glsl */`
float range(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
  vec3 sub = vec3(oldValue, newMax, oldMax) - vec3(oldMin, newMin, oldMin);
  return sub.x * sub.y / sub.z + newMin;
}
float crange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
  return clamp(range(oldValue, oldMin, oldMax, newMin, newMax), min(newMin, newMax), max(newMin, newMax));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: noise / curl helpers (GPGPU fragment)
// ─────────────────────────────────────────────────────────────────────────────

const NOISE_GLSL = /* glsl */`
float hash1(float n) { return fract(sin(n) * 43758.5453123); }
vec3  hash3(vec3 p) {
  vec3 q = vec3(
    dot(p, vec3(127.1, 311.7, 74.7)),
    dot(p, vec3(269.5, 183.3, 246.1)),
    dot(p, vec3(113.5, 271.9, 124.6))
  );
  return fract(sin(q) * 43758.5453123);
}
float noise3(vec3 x) {
  vec3 i = floor(x);
  vec3 f = fract(x);
  vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float n000 = dot(hash3(i + vec3(0,0,0)) * 2.0 - 1.0, f - vec3(0,0,0));
  float n100 = dot(hash3(i + vec3(1,0,0)) * 2.0 - 1.0, f - vec3(1,0,0));
  float n010 = dot(hash3(i + vec3(0,1,0)) * 2.0 - 1.0, f - vec3(0,1,0));
  float n110 = dot(hash3(i + vec3(1,1,0)) * 2.0 - 1.0, f - vec3(1,1,0));
  float n001 = dot(hash3(i + vec3(0,0,1)) * 2.0 - 1.0, f - vec3(0,0,1));
  float n101 = dot(hash3(i + vec3(1,0,1)) * 2.0 - 1.0, f - vec3(1,0,1));
  float n011 = dot(hash3(i + vec3(0,1,1)) * 2.0 - 1.0, f - vec3(0,1,1));
  float n111 = dot(hash3(i + vec3(1,1,1)) * 2.0 - 1.0, f - vec3(1,1,1));
  return mix(
    mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
    mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
    u.z
  );
}
vec2 curlNoise2D(vec3 p, float eps) {
  vec3 dx = vec3(eps, 0.0, 0.0);
  vec3 dy = vec3(0.0, eps, 0.0);
  vec3 dz = vec3(0.0, 0.0, eps);
  float Fz_x = (noise3(p+dx+dz) - noise3(p-dx+dz) - noise3(p+dx) + noise3(p-dx)) / (4.0*eps*eps);
  float Fz_y = (noise3(p+dy+dz) - noise3(p-dy+dz) - noise3(p+dy) + noise3(p-dy)) / (4.0*eps*eps);
  return vec2(-Fz_y, Fz_x);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: full-screen quad vertex  (shared by all GPGPU passes)
// ─────────────────────────────────────────────────────────────────────────────

const QUAD_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: AntimatterSpawn.fs  (adapted from compiled.vs line 6368)
// Reads tLife (spawn requests) + tAttribs (random seeds) + tInput (current state).
// Outputs new life value; dead particles get life = 0, spawning from boundary
// face params written into tAttribs / tLife by CPU upload.
// ─────────────────────────────────────────────────────────────────────────────

const SPAWN_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform sampler2D tInput;    // current pos/vel   (rgba32f: xy=pos, zw=vel)
uniform sampler2D tLife;     // life texture       (rgba32f: x=life, y=maxLife, z=phase, w=decayRate)
uniform sampler2D tAttribs;  // random attribs     (rgba32f: xyzw = rand seeds)
uniform sampler2D tEmitters; // emitter data       (rgba32f packed: see buildEmitterTex)
uniform float uMaxCount;
uniform float uSetup;
uniform float decay;
uniform vec2  decayRandom;
uniform float HZ;
uniform float uTime;
uniform float uDelta;
uniform float fSize;         // TEX_W (= TEX_H)
uniform float emitterCount;

${RANGE_GLSL}

float rng(float seed, float salt) {
  return fract(sin(seed * 127.1 + salt * 311.7) * 43758.5453);
}

// Fetch one float from a 256x256 RGBA32F emitter atlas.
// We pack EMITTER_STRIDE=12 floats per emitter.
// Each texel holds 4 floats → emitter i starts at texel i*3.
float emitterField(int eIdx, int field) {
  int texelIdx = eIdx * 3 + field / 4;
  int component = field - (field / 4) * 4;
  float tx = (float(texelIdx % 256) + 0.5) / 256.0;
  float ty = (float(texelIdx / 256) + 0.5) / 256.0;
  vec4 t = texture2D(tEmitters, vec2(tx, ty));
  if (component == 0) return t.x;
  if (component == 1) return t.y;
  if (component == 2) return t.z;
  return t.w;
}

void main() {
  vec2 uv = vUv;
  float linearIdx = floor(uv.x * fSize) + floor(uv.y * fSize) * fSize;

  if (linearIdx >= uMaxCount) {
    gl_FragColor = vec4(9999.0, 0.0, 0.0, 0.0);
    return;
  }

  vec4 lifeData  = texture2D(tLife, uv);   // x=life, y=maxLife, z=phase, w=decayRate
  vec4 attribs   = texture2D(tAttribs, uv); // x=seed, y-w=rand
  vec4 inputData = texture2D(tInput, uv);  // xy=pos, zw=vel

  float life      = lifeData.x;
  float maxLife   = lifeData.y;
  float phase     = lifeData.z;
  float decayRate = lifeData.w;
  float seed      = attribs.x;

  if (uSetup > 0.5) {
    gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);  // life=0, phase=0 (dead), decayRate=1
    return;
  }

  // --- Dead slot: try to claim from an emitter ---
  if (phase < 0.5) {
    for (int e = 0; e < 128; e++) {
      if (float(e) >= emitterCount) break;
      float active  = emitterField(e, 9);
      if (active < 0.5) continue;
      float count   = emitterField(e, 5);
      // Each dead slot self-assigns using UV fingerprint as hash
      float slotRng = rng(linearIdx, float(e) + 0.1);
      if (slotRng * float(int(count) + 1) > 1.0) continue;

      float originX  = emitterField(e, 0);
      float originY  = emitterField(e, 1);
      float normalX  = emitterField(e, 2);
      float normalY  = emitterField(e, 3);
      float strength = emitterField(e, 4);
      float initLife = emitterField(e, 6);
      float halfLen  = emitterField(e, 8);

      float t       = rng(seed, float(e)) * 2.0 - 1.0;
      float perpX   = -normalY;
      float perpY   =  normalX;
      float spawnX  = originX + perpX * halfLen * t;
      float spawnY  = originY + perpY * halfLen * t;

      float angle   = (rng(seed + 1.0, float(e)) * 2.0 - 1.0) * 0.785398;
      float cosA    = cos(angle);
      float sinA    = sin(angle);
      float rotNX   = normalX * cosA - normalY * sinA;
      float rotNY   = normalX * sinA + normalY * cosA;
      float velX    = rotNX * strength;
      float velY    = rotNY * strength;

      float dv = crange(rng(seed, 7.0), 0.0, 1.0, decayRandom.x, decayRandom.y);

      // Output new position/vel to tInput ping-pong
      gl_FragColor = vec4(spawnX, spawnY, velX, velY);
      // (life texture updated in separate spawn-life pass)
      return;
    }
    // No emitter claimed — stay dead, output neutral
    gl_FragColor = inputData;
    return;
  }

  // --- Alive: decay life ---
  if (phase >= 1.0 && phase < 4.5) {
    float newLife = life - decay * decayRate * HZ * uDelta;
    if (newLife <= 0.0) {
      // Mark handoff
      gl_FragColor = vec4(9999.0, 0.0, 0.0, 0.0);
    } else {
      gl_FragColor = inputData;
    }
    return;
  }

  gl_FragColor = inputData;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: Life texture update pass
// Writes lifeData.xyzw = (life, maxLife, phase, decayRate)
// ─────────────────────────────────────────────────────────────────────────────

const SPAWN_LIFE_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform sampler2D tLife;     // (life, maxLife, phase, decayRate)
uniform sampler2D tAttribs;  // random seeds
uniform sampler2D tEmitters;
uniform float uMaxCount;
uniform float uSetup;
uniform float decay;
uniform vec2  decayRandom;
uniform float HZ;
uniform float uDelta;
uniform float fSize;
uniform float emitterCount;

${RANGE_GLSL}

float rng(float seed, float salt) {
  return fract(sin(seed * 127.1 + salt * 311.7) * 43758.5453);
}
float emitterField(int eIdx, int field) {
  int texelIdx = eIdx * 3 + field / 4;
  int component = field - (field / 4) * 4;
  float tx = (float(texelIdx % 256) + 0.5) / 256.0;
  float ty = (float(texelIdx / 256) + 0.5) / 256.0;
  vec4 t = texture2D(tEmitters, vec2(tx, ty));
  if (component == 0) return t.x;
  if (component == 1) return t.y;
  if (component == 2) return t.z;
  return t.w;
}

void main() {
  vec2 uv = vUv;
  float linearIdx = floor(uv.x * fSize) + floor(uv.y * fSize) * fSize;

  if (uSetup > 0.5) {
    gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);
    return;
  }

  if (linearIdx >= uMaxCount) {
    gl_FragColor = vec4(0.0);
    return;
  }

  vec4 lifeData = texture2D(tLife, uv);
  vec4 attribs  = texture2D(tAttribs, uv);
  float life      = lifeData.x;
  float maxLife   = lifeData.y;
  float phase     = lifeData.z;
  float decayRate = lifeData.w;
  float seed      = attribs.x;

  // Dead slot — try to spawn
  if (phase < 0.5) {
    for (int e = 0; e < 128; e++) {
      if (float(e) >= emitterCount) break;
      float active = emitterField(e, 9);
      if (active < 0.5) continue;
      float count  = emitterField(e, 5);
      float slotRng = rng(linearIdx, float(e) + 0.1);
      if (slotRng * float(int(count) + 1) > 1.0) continue;

      float initLife = emitterField(e, 6);
      float dv = crange(rng(seed, 7.0), 0.0, 1.0, decayRandom.x, decayRandom.y);
      gl_FragColor = vec4(initLife, initLife, 1.0, dv);  // phase=1 (spawning)
      return;
    }
    gl_FragColor = lifeData;
    return;
  }

  // Alive: drain life
  if (phase >= 1.0 && phase < 4.5) {
    float newLife = life - decay * decayRate * HZ * uDelta;
    if (newLife <= 0.0) {
      gl_FragColor = vec4(0.0, maxLife, 4.0, decayRate);  // phase=4 handoff
    } else {
      float newPhase = (phase < 1.5) ? 2.0 : phase;
      gl_FragColor = vec4(newLife, maxLife, newPhase, decayRate);
    }
    return;
  }

  // Handoff phase — kill next frame
  if (phase >= 3.5) {
    gl_FragColor = vec4(0.0, maxLife, 0.0, decayRate);
    return;
  }

  gl_FragColor = lifeData;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: ProtonAntimatter.fs  (physics update, adapted from compiled.vs 7899)
// Reads tOrigin (spawn pos), tAttribs (seeds), tInput (current state).
// Applies curl noise + origin attraction + velocity integration.
// ─────────────────────────────────────────────────────────────────────────────

const PHYSICS_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform sampler2D tInput;   // xy=pos, zw=vel
uniform sampler2D tOrigin;  // xy=spawnOrigin, zw=unused
uniform sampler2D tAttribs; // xyzw = rand
uniform sampler2D tLife;    // x=life, y=maxLife, z=phase, w=decayRate
uniform float uMaxCount;
uniform float uTime;
uniform float uDelta;
uniform float HZ;
uniform float curlScale;
uniform float curlSpeed;
uniform float curlStrength;
uniform float originStrength;
uniform float fSize;
uniform float domainW;
uniform float domainH;

${RANGE_GLSL}
${NOISE_GLSL}

void main() {
  vec2 uv = vUv;
  float linearIdx = floor(uv.x * fSize) + floor(uv.y * fSize) * fSize;

  if (linearIdx >= uMaxCount) {
    gl_FragColor = vec4(9999.0, 0.0, 0.0, 0.0);
    return;
  }

  vec4 lifeData  = texture2D(tLife,    uv);
  float phase    = lifeData.z;

  // Only update alive particles (phase 2–3)
  if (phase < 1.5 || phase > 3.5) {
    gl_FragColor = texture2D(tInput, uv);
    return;
  }

  // From compiled.vs ProtonAntimatter.fs:
  //   vec3 origin = texture2D(tOrigin, uv).xyz;
  //   vec4 inputData = texture2D(tInput, uv);
  //   vec3 pos = inputData.xyz;
  //   vec4 random = texture2D(tAttribs, uv);
  //   float data = inputData.w;

  vec4 inputData = texture2D(tInput,   uv);
  vec4 originData = texture2D(tOrigin,  uv);
  float posX   = inputData.x;
  float posY   = inputData.y;
  float velX   = inputData.z;
  float velY   = inputData.w;
  float origX  = originData.x;
  float origY  = originData.y;
  float life   = lifeData.x;
  float maxLife = lifeData.y;

  // Curl noise force
  vec3 noiseCoord = vec3(posX * curlScale, posY * curlScale, uTime * curlSpeed);
  vec2 curl = curlNoise2D(noiseCoord, 0.01);
  float curlFX = curl.x * curlStrength;
  float curlFY = curl.y * curlStrength;

  // Origin attraction
  float toOrigX = origX - posX;
  float toOrigY = origY - posY;
  float distOrig = sqrt(toOrigX * toOrigX + toOrigY * toOrigY) + 1e-6;
  float lifeRatio = life / max(maxLife, 0.001);
  float origFX = (toOrigX / distOrig) * originStrength * lifeRatio;
  float origFY = (toOrigY / distOrig) * originStrength * lifeRatio;

  float damping = 0.98;
  float dt = uDelta;
  velX = velX * damping + (curlFX + origFX) * dt;
  velY = velY * damping + (curlFY + origFY) * dt;

  posX += velX * dt * HZ;
  posY += velY * dt * HZ;

  // Boundary bounce
  if (posX < 0.0) { posX = -posX; velX =  abs(velX) * 0.5; }
  if (posY < 0.0) { posY = -posY; velY =  abs(velY) * 0.5; }
  if (posX > domainW) { posX = 2.0 * domainW - posX; velX = -abs(velX) * 0.5; }
  if (posY > domainH) { posY = 2.0 * domainH - posY; velY = -abs(velY) * 0.5; }

  gl_FragColor = vec4(posX, posY, velX, velY);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: tPos write pass  (world pos → 256×256 RGBA32F texture for renderer)
// ─────────────────────────────────────────────────────────────────────────────

const TPOS_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform sampler2D tInput;   // xy=pos, zw=vel
uniform sampler2D tLife;    // x=life, y=maxLife, z=phase, w=decayRate
uniform sampler2D tAttribs; // species in .z
uniform float fSize;
uniform float uMaxCount;

void main() {
  float linearIdx = floor(vUv.x * fSize) + floor(vUv.y * fSize) * fSize;
  if (linearIdx >= uMaxCount) {
    gl_FragColor = vec4(9999.0, 9999.0, 0.0, 0.0);
    return;
  }

  vec4 posData  = texture2D(tInput, vUv);
  vec4 lifeData = texture2D(tLife,  vUv);
  vec4 attribs  = texture2D(tAttribs, vUv);

  float posX    = posData.x;
  float posY    = posData.y;
  float life    = lifeData.x;
  float maxLife = lifeData.y;
  float phase   = lifeData.z;
  float species = attribs.z;

  float lifeRatio = life / max(maxLife, 0.001);
  float alpha = (phase >= 1.5) ? sin(3.14159265 * lifeRatio) * lifeRatio : 0.0;

  gl_FragColor = vec4(posX, posY, alpha, species);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: AntimatterPosition.vs  (from compiled.vs line 16)
// Point-sprite vertex shader — reads tPos texture via gl_VertexID emulation
// using per-particle UV attribute (aUV).
// ─────────────────────────────────────────────────────────────────────────────

const POINT_VERT = /* glsl */`
precision highp float;
attribute vec2 aUV;         // per-particle UV into tPos (pre-computed)

uniform sampler2D tPos;     // rgba32f: xy=worldPos, z=alpha, w=species
uniform float uDPR;
uniform float uSize;
uniform float scaleX;       // 2 / canvasW
uniform float scaleY;       // 2 / canvasH

varying float vAlpha;
varying float vSpecies;

// AntimatterPosition.vs (compiled.vs line 16-25):
//   vec4 decodedPos = texture2D(tPos, position.xy);
//   vec3 pos = decodedPos.xyz;
//   vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
//   gl_PointSize = (0.02 * uDPR) * (1000.0 / length(mvPosition.xyz));
//   gl_Position = projectionMatrix * mvPosition;

void main() {
  vec4 decodedPos = texture2D(tPos, aUV);
  float worldX = decodedPos.x;
  float worldY = decodedPos.y;
  float alpha   = decodedPos.z;
  float species = decodedPos.w;

  vAlpha   = alpha;
  vSpecies = species;

  float alive = step(0.005, alpha);
  float sizeScale = sqrt(max(alpha, 0.0));

  // Map world → NDC
  float ndcX = worldX * scaleX - 1.0;
  float ndcY = worldY * scaleY - 1.0;

  gl_Position  = vec4(ndcX * alive, ndcY * alive, 0.0, 1.0);
  gl_PointSize = uSize * sizeScale * uDPR * 1000.0;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: ProtonAntimatter.fs render frag  (point sprite shading)
// Based on AntimatterBasicFrag.fs + species colour palette
// ─────────────────────────────────────────────────────────────────────────────

const POINT_FRAG = /* glsl */`
precision highp float;
varying float vAlpha;
varying float vSpecies;

uniform sampler2D tMatcap;

vec3 speciesColour(float s) {
  int si = int(mod(s, 8.0));
  if (si == 0) return vec3(0.4, 0.7, 1.0);   // electric blue
  if (si == 1) return vec3(1.0, 0.4, 0.6);   // plasma pink
  if (si == 2) return vec3(0.3, 1.0, 0.6);   // neon green
  if (si == 3) return vec3(1.0, 0.8, 0.2);   // solar gold
  if (si == 4) return vec3(0.6, 0.3, 1.0);   // violet
  if (si == 5) return vec3(1.0, 0.5, 0.2);   // ember orange
  if (si == 6) return vec3(0.2, 0.9, 0.9);   // cyan
  return vec3(0.9, 0.9, 0.9);                // white
}

void main() {
  // Point sprite UV: gl_PointCoord in [0,1]
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(uv, uv);
  if (r2 > 1.0) discard;

  vec2 matcapUV = uv * 0.5 + 0.5;
  vec3 matcap   = texture2D(tMatcap, matcapUV).rgb;

  vec3 baseCol    = speciesColour(vSpecies);
  vec3 litCol     = matcap * baseCol;
  float radialFade = 1.0 - r2 * 0.4;
  float finalA    = vAlpha * radialFade;

  gl_FragColor = vec4(litCol, finalA);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

interface InternalEmitter {
  originX:  number;
  originY:  number;
  normalX:  number;
  normalY:  number;
  strength: number;
  count:    number;
  life:     number;
  species:  number;
  halfLen:  number;
  active:   boolean;
  faceId:   string;
  cellId:   string;
  targetId: string;
}

interface PingPong {
  read:     WebGLFramebuffer;
  write:    WebGLFramebuffer;
  readTex:  WebGLTexture;
  writeTex: WebGLTexture;
  width:    number;
  height:   number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main class
// ─────────────────────────────────────────────────────────────────────────────

export class ATAntimatterParticles {
  private readonly gl: WebGLRenderingContext;
  private readonly canvas: HTMLCanvasElement;
  private readonly cfg: Required<Omit<ATAntimatterConfig, 'onHandoff'>>;
  private readonly onHandoff?: ATAntimatterConfig['onHandoff'];

  // WebGL programs
  private spawnProg!:      WebGLProgram;   // AntimatterSpawn.fs
  private spawnLifeProg!:  WebGLProgram;   // life texture update
  private physicsProg!:    WebGLProgram;   // ProtonAntimatter.fs
  private tposProg!:       WebGLProgram;   // tPos write pass
  private renderProg!:     WebGLProgram;   // AntimatterPosition.vs + ProtonAntimatter point frag

  // GPGPU ping-pong textures (256×256 RGBA32F)
  private posPingPong!:    PingPong;   // xy=pos, zw=vel
  private lifePingPong!:   PingPong;   // x=life, y=maxLife, z=phase, w=decayRate
  private originTex!:      WebGLTexture;   // spawn origins (static per-particle)
  private attribsTex!:     WebGLTexture;   // random seeds + species
  private emitterTex!:     WebGLTexture;   // emitter data atlas

  // tPos output texture (read by vertex shader)
  private tPosFBO!:        WebGLFramebuffer;
  private tPosTex!:        WebGLTexture;

  // Matcap for point-sprite shading
  private matcapTex!:      WebGLTexture;

  // Geometry
  private quadBuf!:        WebGLBuffer;   // full-screen quad [-1,1]
  private particleUVBuf!:  WebGLBuffer;   // per-particle UV array (MAX_PARTICLES × vec2)

  // CPU-side state
  private particleCount   = MAX_PARTICLES;
  private emitters:       InternalEmitter[] = [];
  private pendingEmits:   EmitterRequest[] = [];
  private faceRegistry    = new Map<number, { cellId: string; targetId: string }>();

  private built     = false;
  private elapsed   = 0;
  private frameCount = 0;
  private setupFrame = true;

  // Readback for handoff
  private readbackBuf!: WebGLBuffer;
  private readbackFBO!: WebGLFramebuffer;

  constructor(
    gl: WebGLRenderingContext,
    canvas: HTMLCanvasElement,
    config: ATAntimatterConfig = {},
  ) {
    this.gl     = gl;
    this.canvas = canvas;
    this.onHandoff = config.onHandoff;
    this.cfg = {
      decay:          config.decay          ?? DEFAULTS.decay,
      decayRandom:    config.decayRandom    ?? [...DEFAULTS.decayRandom],
      curlScale:      config.curlScale      ?? DEFAULTS.curlScale,
      curlSpeed:      config.curlSpeed      ?? DEFAULTS.curlSpeed,
      curlStrength:   config.curlStrength   ?? DEFAULTS.curlStrength,
      originStrength: config.originStrength ?? DEFAULTS.originStrength,
      hz:             config.hz             ?? DEFAULTS.hz,
      dpr:            config.dpr            ?? DEFAULTS.dpr,
      uSize:          config.uSize          ?? DEFAULTS.uSize,
      sprayStrength:  config.sprayStrength  ?? DEFAULTS.sprayStrength,
      spraySpread:    config.spraySpread    ?? DEFAULTS.spraySpread,
      sphSmoothingH:  config.sphSmoothingH  ?? DEFAULTS.sphSmoothingH,
      sphRestDensity: config.sphRestDensity ?? DEFAULTS.sphRestDensity,
    };
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  init(): void {
    if (this.built) this._dispose();
    const gl = this.gl;

    // 1. Compile all programs
    this.spawnProg     = this._compile(QUAD_VERT, SPAWN_FRAG,      'spawn');
    this.spawnLifeProg = this._compile(QUAD_VERT, SPAWN_LIFE_FRAG, 'spawnLife');
    this.physicsProg   = this._compile(QUAD_VERT, PHYSICS_FRAG,    'physics');
    this.tposProg      = this._compile(QUAD_VERT, TPOS_FRAG,       'tpos');
    this.renderProg    = this._compile(POINT_VERT, POINT_FRAG,     'render');

    // 2. Create GPGPU ping-pong FBOs
    this.posPingPong  = this._createPingPong(TEX_W, TEX_H);
    this.lifePingPong = this._createPingPong(TEX_W, TEX_H);

    // 3. Origin texture — stores per-particle spawn world position
    this.originTex = this._createFloatTexture(TEX_W, TEX_H, null);

    // 4. Attribs texture — random seeds (x), extra rand (y-z), species (z)
    const attribData = new Float32Array(TEX_W * TEX_H * 4);
    for (let i = 0; i < TEX_W * TEX_H; i++) {
      attribData[i * 4 + 0] = Math.random() * 1000;  // seed
      attribData[i * 4 + 1] = Math.random();
      attribData[i * 4 + 2] = 0;                      // species (updated at spawn)
      attribData[i * 4 + 3] = Math.random();
    }
    this.attribsTex = this._createFloatTexture(TEX_W, TEX_H, attribData);

    // 5. Emitter atlas texture (256×256 RGBA32F, packed 3 texels per emitter)
    this.emitterTex = this._createFloatTexture(256, 256, null);

    // 6. tPos output texture (written by tposProg, read by renderProg vertex)
    const { fbo: tposFBO, tex: tposTex } = this._createSingleFBO(TEX_W, TEX_H);
    this.tPosFBO = tposFBO;
    this.tPosTex = tposTex;

    // 7. Matcap texture (1×1 white default; overwrite with loadMatcap)
    this.matcapTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.matcapTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                  new Uint8Array([255, 255, 255, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    // 8. Full-screen quad buffer
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // 9. Per-particle UV attribute buffer (one vec2 per particle)
    const uvs = new Float32Array(MAX_PARTICLES * 2);
    for (let i = 0; i < MAX_PARTICLES; i++) {
      uvs[i * 2 + 0] = ((i % TEX_W) + 0.5) / TEX_W;
      uvs[i * 2 + 1] = (Math.floor(i / TEX_W) + 0.5) / TEX_H;
    }
    this.particleUVBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleUVBuf);
    gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // 10. Readback pixel buffer for handoff detection
    this.readbackFBO = gl.createFramebuffer()!;

    this.built     = true;
    this.setupFrame = true;
  }

  // Legacy alias matching WebGPU version's async build()
  async build(): Promise<void> { this.init(); }

  emitFromBoundary(request: EmitterRequest): void {
    this.pendingEmits.push(request);
  }

  emitBurst(requests: EmitterRequest[]): void {
    this.pendingEmits.push(...requests);
  }

  registerBoundaryFaces(faces: CellBoundaryFace[]): void {
    for (const face of faces) {
      const key = this._hashFaceId(face.faceId);
      this.faceRegistry.set(key, { cellId: face.cellId, targetId: face.targetId });
    }
  }

  loadMatcap(bitmap: ImageBitmap): void {
    if (!this.built) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.matcapTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // Called each frame BEFORE render()
  tick(elapsed: number, dt: number): void {
    if (!this.built) return;
    this.elapsed = elapsed;
    this.frameCount++;

    this._processEmitters();
    this._uploadEmitterTex();
    this._runSpawnPass(elapsed, dt);
    this._runPhysicsPass(elapsed, dt);
    this._runTposPass();

    if (this.setupFrame) this.setupFrame = false;
    for (const e of this.emitters) e.active = false;
  }

  // Also accept (encoder, elapsed, dt) signature for drop-in compat with WebGPU version
  update(_encoder: unknown, elapsed: number, dt: number): void {
    this.tick(elapsed, dt);
  }

  // Render point sprites to currently bound framebuffer
  render(): void;
  render(_encoder: unknown, _colorView: unknown, _depthView?: unknown): void;
  render(..._args: unknown[]): void {
    if (!this.built) return;
    this._runRenderPass();
  }

  scheduleHandoffReadback(): void {
    if (!this.built || !this.onHandoff) return;
    this._readbackHandoff();
  }

  setDecay(v: number): void           { this.cfg.decay = v; }
  setDecayRandom(min: number, max: number): void { this.cfg.decayRandom = [min, max]; }
  setCurlScale(v: number): void       { this.cfg.curlScale = v; }
  setCurlSpeed(v: number): void       { this.cfg.curlSpeed = v; }
  setCurlStrength(v: number): void    { this.cfg.curlStrength = v; }
  setOriginStrength(v: number): void  { this.cfg.originStrength = v; }
  setSize(v: number): void            { this.cfg.uSize = v; }
  setSprayStrength(v: number): void   { this.cfg.sprayStrength = v; }
  setSpraySpread(v: number): void     { this.cfg.spraySpread = v; }
  setDPR(v: number): void             { this.cfg.dpr = v; }
  setMaxParticles(n: number): void    { this.particleCount = Math.min(n, MAX_PARTICLES); }
  writeSPHField(_x: number, _y: number, _density: number, _px: number, _py: number, _r: number): void {}

  get activeParticleCount(): number { return this.particleCount; }
  get activeEmitterCount():  number { return this.emitters.filter(e => e.active).length; }
  get isBuilt():             boolean { return this.built; }
  get totalFrames():         number { return this.frameCount; }

  destroy(): void { this._dispose(); }
  dispose():  void { this._dispose(); }

  // ── Private: GPGPU passes ──────────────────────────────────────────────────

  private _runSpawnPass(elapsed: number, dt: number): void {
    const gl = this.gl;
    const pp = this.posPingPong;
    const lp = this.lifePingPong;

    // --- Spawn: pos/vel update ---
    gl.useProgram(this.spawnProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, pp.write);
    gl.viewport(0, 0, TEX_W, TEX_H);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, pp.readTex);
    gl.uniform1i(gl.getUniformLocation(this.spawnProg, 'tInput'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, lp.readTex);
    gl.uniform1i(gl.getUniformLocation(this.spawnProg, 'tLife'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.attribsTex);
    gl.uniform1i(gl.getUniformLocation(this.spawnProg, 'tAttribs'), 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.emitterTex);
    gl.uniform1i(gl.getUniformLocation(this.spawnProg, 'tEmitters'), 3);

    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'uMaxCount'),    this.particleCount);
    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'uSetup'),       this.setupFrame ? 1.0 : 0.0);
    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'decay'),        this.cfg.decay);
    gl.uniform2f(gl.getUniformLocation(this.spawnProg, 'decayRandom'),  this.cfg.decayRandom[0], this.cfg.decayRandom[1]);
    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'HZ'),           this.cfg.hz);
    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'uTime'),        elapsed);
    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'uDelta'),       Math.min(dt, 1/30));
    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'fSize'),        TEX_W);
    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'emitterCount'), this.emitters.length);

    this._drawQuad(this.spawnProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this._swapPingPong(pp);

    // --- Spawn: life texture update ---
    gl.useProgram(this.spawnLifeProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, lp.write);
    gl.viewport(0, 0, TEX_W, TEX_H);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, lp.readTex);
    gl.uniform1i(gl.getUniformLocation(this.spawnLifeProg, 'tLife'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.attribsTex);
    gl.uniform1i(gl.getUniformLocation(this.spawnLifeProg, 'tAttribs'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.emitterTex);
    gl.uniform1i(gl.getUniformLocation(this.spawnLifeProg, 'tEmitters'), 2);

    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'uMaxCount'),    this.particleCount);
    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'uSetup'),       this.setupFrame ? 1.0 : 0.0);
    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'decay'),        this.cfg.decay);
    gl.uniform2f(gl.getUniformLocation(this.spawnLifeProg, 'decayRandom'),  this.cfg.decayRandom[0], this.cfg.decayRandom[1]);
    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'HZ'),           this.cfg.hz);
    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'uDelta'),       Math.min(dt, 1/30));
    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'fSize'),        TEX_W);
    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'emitterCount'), this.emitters.length);

    this._drawQuad(this.spawnLifeProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this._swapPingPong(lp);
  }

  private _runPhysicsPass(elapsed: number, dt: number): void {
    const gl = this.gl;
    const pp = this.posPingPong;
    const lp = this.lifePingPong;

    // ProtonAntimatter.fs physics pass
    gl.useProgram(this.physicsProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, pp.write);
    gl.viewport(0, 0, TEX_W, TEX_H);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, pp.readTex);
    gl.uniform1i(gl.getUniformLocation(this.physicsProg, 'tInput'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.originTex);
    gl.uniform1i(gl.getUniformLocation(this.physicsProg, 'tOrigin'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.attribsTex);
    gl.uniform1i(gl.getUniformLocation(this.physicsProg, 'tAttribs'), 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, lp.readTex);
    gl.uniform1i(gl.getUniformLocation(this.physicsProg, 'tLife'), 3);

    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'uMaxCount'),      this.particleCount);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'uTime'),          elapsed);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'uDelta'),         Math.min(dt, 1/30));
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'HZ'),             this.cfg.hz);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'curlScale'),      this.cfg.curlScale);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'curlSpeed'),      this.cfg.curlSpeed);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'curlStrength'),   this.cfg.curlStrength);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'originStrength'), this.cfg.originStrength);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'fSize'),          TEX_W);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'domainW'),        this.canvas.width);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'domainH'),        this.canvas.height);

    this._drawQuad(this.physicsProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this._swapPingPong(pp);
  }

  private _runTposPass(): void {
    const gl = this.gl;

    // Write world pos + alpha into tPosTex for vertex shader
    gl.useProgram(this.tposProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.tPosFBO);
    gl.viewport(0, 0, TEX_W, TEX_H);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.posPingPong.readTex);
    gl.uniform1i(gl.getUniformLocation(this.tposProg, 'tInput'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.lifePingPong.readTex);
    gl.uniform1i(gl.getUniformLocation(this.tposProg, 'tLife'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.attribsTex);
    gl.uniform1i(gl.getUniformLocation(this.tposProg, 'tAttribs'), 2);

    gl.uniform1f(gl.getUniformLocation(this.tposProg, 'fSize'),      TEX_W);
    gl.uniform1f(gl.getUniformLocation(this.tposProg, 'uMaxCount'),  this.particleCount);

    this._drawQuad(this.tposProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private _runRenderPass(): void {
    const gl = this.gl;

    // Render to current bound FBO (caller sets it, or screen if null)
    gl.useProgram(this.renderProg);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    // Enable additive blending for antimatter glow
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tPosTex);
    gl.uniform1i(gl.getUniformLocation(this.renderProg, 'tPos'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.matcapTex);
    gl.uniform1i(gl.getUniformLocation(this.renderProg, 'tMatcap'), 1);

    gl.uniform1f(gl.getUniformLocation(this.renderProg, 'uDPR'),    this.cfg.dpr);
    gl.uniform1f(gl.getUniformLocation(this.renderProg, 'uSize'),   this.cfg.uSize);
    gl.uniform1f(gl.getUniformLocation(this.renderProg, 'scaleX'),  2.0 / this.canvas.width);
    gl.uniform1f(gl.getUniformLocation(this.renderProg, 'scaleY'),  2.0 / this.canvas.height);

    // Draw MAX_PARTICLES point sprites using per-particle UV attribute
    const aUVLoc = gl.getAttribLocation(this.renderProg, 'aUV');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleUVBuf);
    gl.enableVertexAttribArray(aUVLoc);
    gl.vertexAttribPointer(aUVLoc, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.POINTS, 0, this.particleCount);

    gl.disableVertexAttribArray(aUVLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.disable(gl.BLEND);
  }

  // ── Private: helpers ────────────────────────────────────────────────────────

  private _processEmitters(): void {
    this.emitters = [];
    for (const req of this.pendingEmits) {
      if (this.emitters.length >= MAX_EMITTERS) break;
      const face = req.face;
      const nLen = Math.sqrt(face.normal.x ** 2 + face.normal.y ** 2) || 1;
      this.emitters.push({
        originX:  face.origin.x,
        originY:  face.origin.y,
        normalX:  face.normal.x / nLen,
        normalY:  face.normal.y / nLen,
        strength: req.strength ?? this.cfg.sprayStrength,
        count:    Math.min(req.count, 4096),
        life:     req.life ?? 1.0,
        species:  face.species ?? 0,
        halfLen:  face.halfLen,
        active:   true,
        faceId:   face.faceId,
        cellId:   face.cellId,
        targetId: face.targetId,
      });
    }
    this.pendingEmits = [];
  }

  private _uploadEmitterTex(): void {
    const gl = this.gl;
    // Each emitter needs EMITTER_STRIDE=12 floats → 3 RGBA32F texels
    const data = new Float32Array(256 * 256 * 4);
    for (let i = 0; i < Math.min(this.emitters.length, MAX_EMITTERS); i++) {
      const e    = this.emitters[i];
      const base = i * 3 * 4;  // 3 texels × 4 floats
      data[base + 0]  = e.originX;
      data[base + 1]  = e.originY;
      data[base + 2]  = e.normalX;
      data[base + 3]  = e.normalY;
      data[base + 4]  = e.strength;
      data[base + 5]  = e.count;
      data[base + 6]  = e.life;
      data[base + 7]  = e.species;
      data[base + 8]  = e.halfLen;
      data[base + 9]  = e.active ? 1.0 : 0.0;
      data[base + 10] = 0;
      data[base + 11] = 0;
    }
    gl.bindTexture(gl.TEXTURE_2D, this.emitterTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F,
                  256, 256, 0,
                  gl.RGBA, gl.FLOAT, data);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** Compile vertex + fragment → WebGLProgram with error checking */
  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    // ── WebGL1 → WebGL2 (GLSL 300 es) auto-upgrade ──
    const upgradeVert = (src: string): string => {
      let s = src;
      if (!s.includes('#version')) s = '#version 300 es\n' + s;
      s = s.replace(/\battribute\s+/g, 'in ');
      s = s.replace(/\bvarying\s+/g, 'out ');
      return s;
    };
    const upgradeFrag = (src: string): string => {
      let s = src;
      if (!s.includes('#version')) s = '#version 300 es\nprecision highp float;\n' + s;
      s = s.replace(/\bvarying\s+/g, 'in ');
      s = s.replace(/\btexture2D\s*\(/g, 'texture(');
      s = s.replace(/\btextureCube\s*\(/g, 'texture(');
      s = s.replace(/\bgl_FragColor\b/g, 'fragColor');
      // Ensure out vec4 fragColor is declared (after precision)
      if (!s.includes('out vec4 fragColor')) {
        s = s.replace(/(precision\s+highp\s+float\s*;)/, '$1\nout vec4 fragColor;');
      }
      // Remove redundant precision if already present from template
      s = s.replace(/(precision\s+highp\s+float\s*;\n?){2,}/g, 'precision highp float;\n');
      return s;
    };

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, upgradeVert(vert));
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATAntimatter] vertex compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, upgradeFrag(frag));
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATAntimatter] fragment compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ATAntimatter] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** Create RGBA32F texture (optionally initialised with data) */
  private _createFloatTexture(w: number, h: number, data: Float32Array | null): WebGLTexture {
    const gl  = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, data);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  /** Create a single FBO backed by a RGBA32F texture */
  private _createSingleFBO(w: number, h: number): { fbo: WebGLFramebuffer; tex: WebGLTexture } {
    const gl  = this.gl;
    const tex = this._createFloatTexture(w, h, null);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fbo, tex };
  }

  /** Create a ping-pong pair of FBOs for GPGPU state */
  private _createPingPong(w: number, h: number): PingPong {
    const gl  = this.gl;

    const readTex  = this._createFloatTexture(w, h, null);
    const writeTex = this._createFloatTexture(w, h, null);

    const readFBO  = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, readFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, readTex, 0);

    const writeFBO = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, writeFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, writeTex, 0);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { read: readFBO, write: writeFBO, readTex, writeTex, width: w, height: h };
  }

  /** Swap ping-pong FBO read/write */
  private _swapPingPong(pp: PingPong): void {
    [pp.read,    pp.write]    = [pp.write,    pp.read];
    [pp.readTex, pp.writeTex] = [pp.writeTex, pp.readTex];
  }

  /** Draw the full-screen quad (2 triangles, 6 verts) */
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

  /** CPU readback of life texture for handoff detection */
  private _readbackHandoff(): void {
    const gl = this.gl;
    const w = TEX_W;
    const h = TEX_H;

    // Bind the life ping-pong read FBO and read pixels
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.lifePingPong.read);
    const buf = new Float32Array(w * h * 4);
    // gl.readPixels reads RGBA32F → available if OES_texture_float is enabled
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    if (!this.onHandoff) return;

    // Also read pos texture for world coordinates
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.posPingPong.read);
    const posBuf = new Float32Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, posBuf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    for (let i = 0; i < this.particleCount; i++) {
      const base  = i * 4;
      const phase = buf[base + 2];
      if (phase < 3.5 || phase > 4.5) continue;  // only handoff phase=4

      const posX    = posBuf[base + 0];
      const posY    = posBuf[base + 1];
      const velX    = posBuf[base + 2];
      const velY    = posBuf[base + 3];
      const species = 0;  // read from attribsTex if needed
      const density = 0;

      const emIdx  = 0;  // simplified — emitter idx not tracked in this readback
      const cellId   = this.emitters[emIdx]?.cellId   ?? 'unknown';
      const targetId = this.emitters[emIdx]?.targetId ?? 'unknown';

      this.onHandoff(cellId, targetId, posX, posY, velX, velY, species, density);
    }
  }

  private _hashFaceId(faceId: string): number {
    let hash = 0;
    for (let i = 0; i < faceId.length; i++) {
      hash = ((hash << 5) - hash + faceId.charCodeAt(i)) | 0;
    }
    return hash;
  }

  /** Release all WebGL resources */
  private _dispose(): void {
    if (!this.built) return;
    const gl = this.gl;

    gl.deleteProgram(this.spawnProg);
    gl.deleteProgram(this.spawnLifeProg);
    gl.deleteProgram(this.physicsProg);
    gl.deleteProgram(this.tposProg);
    gl.deleteProgram(this.renderProg);

    gl.deleteFramebuffer(this.posPingPong.read);
    gl.deleteFramebuffer(this.posPingPong.write);
    gl.deleteTexture(this.posPingPong.readTex);
    gl.deleteTexture(this.posPingPong.writeTex);

    gl.deleteFramebuffer(this.lifePingPong.read);
    gl.deleteFramebuffer(this.lifePingPong.write);
    gl.deleteTexture(this.lifePingPong.readTex);
    gl.deleteTexture(this.lifePingPong.writeTex);

    gl.deleteTexture(this.originTex);
    gl.deleteTexture(this.attribsTex);
    gl.deleteTexture(this.emitterTex);
    gl.deleteFramebuffer(this.tPosFBO);
    gl.deleteTexture(this.tPosTex);
    gl.deleteTexture(this.matcapTex);

    gl.deleteBuffer(this.quadBuf);
    gl.deleteBuffer(this.particleUVBuf);
    gl.deleteFramebuffer(this.readbackFBO);

    this.built = false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory helpers  (matching original export signatures)
// ─────────────────────────────────────────────────────────────────────────────

export function cellBoundsToFaces(
  cellId:    string,
  targetIds: [string, string, string, string],
  x0: number, y0: number,
  x1: number, y1: number,
  species?:  number,
): CellBoundaryFace[] {
  const cx = (x0 + x1) * 0.5;
  const cy = (y0 + y1) * 0.5;
  const hw = (x1 - x0) * 0.5;
  const hh = (y1 - y0) * 0.5;
  return [
    { faceId: `${cellId}_N`, cellId, targetId: targetIds[0], origin: { x: cx, y: y1 }, normal: { x: 0,  y: 1  }, halfLen: hw, species },
    { faceId: `${cellId}_S`, cellId, targetId: targetIds[1], origin: { x: cx, y: y0 }, normal: { x: 0,  y: -1 }, halfLen: hw, species },
    { faceId: `${cellId}_E`, cellId, targetId: targetIds[2], origin: { x: x1, y: cy }, normal: { x: 1,  y: 0  }, halfLen: hh, species },
    { faceId: `${cellId}_W`, cellId, targetId: targetIds[3], origin: { x: x0, y: cy }, normal: { x: -1, y: 0  }, halfLen: hh, species },
  ];
}

export function createAntimatterForSPH(
  gl:       WebGLRenderingContext,
  canvas:   HTMLCanvasElement,
  addFluid: (x0: number, y0: number, x1: number, y1: number, spacing: number, species: number) => void,
  config:   Omit<ATAntimatterConfig, 'onHandoff'> = {},
): ATAntimatterParticles {
  const HANDOFF_R = 0.05;
  return new ATAntimatterParticles(gl, canvas, {
    ...config,
    onHandoff: (_cellId, _targetId, x, y, _vx, _vy, species, _density) => {
      addFluid(x - HANDOFF_R, y - HANDOFF_R, x + HANDOFF_R, y + HANDOFF_R, HANDOFF_R * 0.8, species);
    },
  });
}

export function createPubSubEmitter(
  system:      ATAntimatterParticles,
  faces:       CellBoundaryFace[],
  countPerMsg: number = 50,
): (fromCellId: string, toCellId: string, strength?: number) => void {
  const lookup = new Map<string, CellBoundaryFace>();
  for (const face of faces) lookup.set(`${face.cellId}_${face.targetId}`, face);
  system.registerBoundaryFaces(faces);
  return (fromCellId: string, toCellId: string, strength?: number) => {
    const key  = `${fromCellId}_${toCellId}`;
    const face = lookup.get(key);
    if (!face) return;
    system.emitFromBoundary({ face, count: countPerMsg, strength });
  };
}

export const AT_ANTIMATTER_DEFAULTS = {
  ...DEFAULTS,
  maxParticles:   MAX_PARTICLES,
  texW:           TEX_W,
  texH:           TEX_H,
  particleStride: PARTICLE_STRIDE,
  maxEmitters:    MAX_EMITTERS,
  emitterStride:  EMITTER_STRIDE,
} as const;
