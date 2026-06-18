/**
 * antimatter.ts — Active Theory Antimatter GPU Particle System
 *
 * Ported from Active Theory's Antimatter module (compiled.vs):
 *   AntimatterCopy.fs/vs    — FBO texture copy pass (ping-pong blit)
 *   AntimatterPass.vs       — Full-screen quad vertex for GPGPU passes
 *   AntimatterPosition.vs   — Point-sprite renderer (position from tPos texture)
 *   AntimatterBasicFrag.fs  — Solid white fallback fragment
 *   AntimatterSpawn.fs      — Lifecycle: spawn/decay with random attribs
 *   ProtonAntimatter.fs     — Physics update: curl noise + origin attraction
 *   antimatter.glsl          — getData/getData4 texture sampling helpers
 *   range.glsl               — range/crange remapping utilities
 *   curl.glsl                — Analytic curl noise (36 trig calls)
 *
 * Architecture:
 *   GPU double-buffered FBO ping-pong for particle state (position + spawn/life).
 *   Each frame runs N shader passes as full-screen quad draws into FBOs:
 *     1. Spawn pass  — reads tLife(attribs) + tInput(prev spawn), writes spawn FBO
 *     2. Physics pass — reads tInput(prev pos) + tOrigin + tAttribs + tSpawn, writes position FBO
 *     3. Copy pass   — optional: blit an FBO for readback or multi-consumer
 *   Render pass uses AntimatterPosition.vs to scatter point sprites from tPos.
 *
 * References:
 *   upstream/webgl2-particles          (Transform Feedback ping-pong — RESEARCH_122)
 *   src/lib/gpgpu/constraint-texture.ts (FBO ping-pong pattern)
 *   src/lib/renderers/antimatter-compute.ts (AntimatterFBO class)
 *   src/lib/particle/ParticleSystem.ts  (GPGPU particle system, same codebase)
 *   src/lib/particle/CurlNoise.ts       (AT curl.glsl port)
 *
 * Shader source: curl -s https://activetheory.net/assets/shaders/compiled.vs
 *
 * AT parameter naming (from at_uil_params.json):
 *   uMaxCount     — active particle ceiling
 *   decay         — life drain speed
 *   decayRandom   — per-particle decay variance [min, max]
 *   uCurlScale    — curl noise spatial frequency
 *   uCurlSpeed    — curl noise time evolution rate
 *   HZ            — frame-rate normalisation (AT uses 60)
 *   uDPR          — device pixel ratio for gl_PointSize
 */

import { CURL_NOISE_GLSL } from './CurlNoise.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface AntimatterConfig {
  gl: WebGL2RenderingContext;
  /** Number of particles (will be rounded up to perfect square) */
  particleCount: number;
  /** AT: decay speed (default 0.005) */
  decay?: number;
  /** AT: decayRandom [min, max] variance (default [0.5, 1.5]) */
  decayRandom?: [number, number];
  /** AT: curl noise spatial scale (default 0.5) */
  curlScale?: number;
  /** AT: curl noise time speed (default 0.3) */
  curlSpeed?: number;
  /** AT: curl force strength (default 0.02) */
  curlStrength?: number;
  /** AT: origin attraction strength (default 0.05) */
  originStrength?: number;
  /** AT: HZ frame-rate normalisation base (default 60) */
  hz?: number;
  /** Device pixel ratio for point size (default window.devicePixelRatio) */
  dpr?: number;
  /** Optional custom physics fragment shader override */
  physicsShaderSrc?: string;
}

export interface AntimatterUniforms {
  uTime: number;
  uDelta: number;
  uMaxCount: number;
  decay: number;
  decayRandom: [number, number];
  uCurlScale: number;
  uCurlSpeed: number;
  uCurlStrength: number;
  uOriginStrength: number;
  HZ: number;
  uDPR: number;
  uSetup: number;
}

/** RGBA32F texture + associated FBO for ping-pong */
export interface AntimatterFBOPair {
  texA: WebGLTexture;
  texB: WebGLTexture;
  fboA: WebGLFramebuffer;
  fboB: WebGLFramebuffer;
  /** 0 = read A / write B, 1 = read B / write A */
  phase: number;
  width: number;
  height: number;
}

// ── AT Shader Sources (from compiled.vs) ──────────────────────────────────────

/**
 * antimatter.glsl — texture sampling helpers
 * AT uses these in every Antimatter fragment shader.
 */
const ANTIMATTER_HELPERS_GLSL = /* glsl */ `
vec3 getData(sampler2D tex, vec2 uv) {
    return texture(tex, uv).xyz;
}

vec4 getData4(sampler2D tex, vec2 uv) {
    return texture(tex, uv);
}
`;

/**
 * range.glsl — AT's range/crange remapping utilities
 * Ported from compiled.vs. float/vec2/vec3 overloads + clamped variants.
 */
const RANGE_GLSL = /* glsl */ `
float range(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    vec3 sub = vec3(oldValue, newMax, oldMax) - vec3(oldMin, newMin, oldMin);
    return sub.x * sub.y / sub.z + newMin;
}

float crange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    return clamp(range(oldValue, oldMin, oldMax, newMin, newMax), min(newMin, newMax), max(newMin, newMax));
}
`;

/**
 * AntimatterPass.vs — full-screen quad vertex shader for GPGPU passes.
 * AT: `vUv = uv; gl_Position = vec4(position, 1.0);`
 * Adapted to WebGL2 (no three.js builtins: explicit aPosition attribute).
 */
const ANTIMATTER_PASS_VERT = /* glsl */ `#version 300 es
precision highp float;

in vec2 aPosition;
out vec2 vUv;

void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

/**
 * AntimatterCopy.fs — simple texture blit (ping-pong copy).
 * AT: `gl_FragColor = texture2D(tDiffuse, vUv);`
 */
const ANTIMATTER_COPY_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D tDiffuse;
in vec2 vUv;
out vec4 fragColor;

void main() {
    fragColor = texture(tDiffuse, vUv);
}
`;

/**
 * AntimatterSpawn.fs — particle lifecycle: spawn/decay.
 * Ported from AT's AntimatterSpawn.fs.
 * tInput = previous spawn state (ping-pong read).
 * tLife = spawn trigger texture (xyz = new spawn position when life.x > 0.5).
 * tAttribs = per-particle random attributes (w used for decay variance).
 *
 * AT preprocessor directives (#test !window.Metal, #endtest) are resolved
 * for WebGL2 (non-Metal path: use gl_FragCoord for uv).
 */
const ANTIMATTER_SPAWN_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform float uMaxCount;
uniform float uSetup;
uniform float decay;
uniform vec2 decayRandom;
uniform sampler2D tInput;
uniform sampler2D tLife;
uniform sampler2D tAttribs;
uniform float HZ;
uniform float fSize;

in vec2 vUv;
out vec4 fragColor;

${RANGE_GLSL}

void main() {
    vec2 uv = gl_FragCoord.xy / fSize;

    vec4 data = texture(tInput, uv);

    if (uv.x + uv.y * fSize > uMaxCount) {
        fragColor = vec4(9999.0);
        return;
    }

    vec4 life = texture(tLife, uv);
    vec4 random = texture(tAttribs, uv);
    if (life.x > 0.5) {
        data.xyz = life.yzw;
        data.x -= 999.0;
    } else {
        if (data.x < -500.0) {
            data.x = 1.0;
        } else {
            data.x -= 0.005 * decay * crange(random.w, 0.0, 1.0, decayRandom.x, decayRandom.y) * HZ;
        }
    }

    if (uSetup > 0.5) {
        data = vec4(0.0);
    }

    fragColor = data;
}
`;

/**
 * ProtonAntimatterLifecycle.fs — physics update with lifecycle management.
 * Ported from AT's ProtonAntimatterLifecycle.fs + curl noise injection.
 *
 * The AT source uses `//code` and `//requires` placeholder comments that
 * get template-replaced at build time. Here we inline the curl noise physics
 * directly as that's what WorkDetailParticles uses.
 */
const ANTIMATTER_PHYSICS_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D tOrigin;
uniform sampler2D tAttribs;
uniform sampler2D tSpawn;
uniform sampler2D tInput;
uniform float uMaxCount;
uniform float uTime;
uniform float uDelta;
uniform float uCurlScale;
uniform float uCurlSpeed;
uniform float uCurlStrength;
uniform float uOriginStrength;
uniform float fSize;
uniform float HZ;

in vec2 vUv;
out vec4 fragColor;

${RANGE_GLSL}
${ANTIMATTER_HELPERS_GLSL}
${CURL_NOISE_GLSL}

void main() {
    vec3 origin = texture(tOrigin, vUv).rgb;
    vec4 inputData = texture(tInput, vUv);
    vec3 pos = inputData.xyz;
    vec4 random = texture(tAttribs, vUv);
    float data = inputData.w;

    if (vUv.x + vUv.y * fSize > uMaxCount) {
        fragColor = vec4(9999.0);
        return;
    }

    vec4 spawn = texture(tSpawn, vUv);
    float life = spawn.x;

    // Respawn: spawn.x < -500 means particle was just spawned
    if (spawn.x < -500.0) {
        pos = spawn.xyz;
        pos.x += 999.0;
        spawn.x = 1.0;
        fragColor = vec4(pos, data);
        return;
    }

    // Dead particle: park offscreen
    if (spawn.x <= 0.0) {
        pos.x = 9999.0;
        fragColor = vec4(pos, data);
        return;
    }

    // ── Physics: curl noise + origin attraction ──────────────────────────
    // AT WorkDetailParticles pattern:
    //   velocity = curlNoise(pos * scale + time * speed) * strength
    //   pos += velocity * delta
    //   pos = mix(pos, origin, originStrength * delta)

    vec3 curlInput = pos * uCurlScale + vec3(uTime * uCurlSpeed);
    curlInput += random.xyz * 0.5; // per-particle phase offset (AT: random jitter)

    vec3 curl = curlNoise(curlInput);
    vec3 velocity = curl * uCurlStrength;

    // Euler integration, frame-rate normalised
    float dt = uDelta * (1.0 / HZ);
    pos += velocity * dt;

    // Attraction toward origin (spline point or cell centre)
    vec3 toOrigin = origin - pos;
    float dist = length(toOrigin);
    if (dist > 0.001) {
        pos += normalize(toOrigin) * min(dist, uOriginStrength * dt);
    }

    fragColor = vec4(pos, data);
}
`;

/**
 * AntimatterPosition.vs — point-sprite vertex shader.
 * Reads particle position from tPos texture, applies projection.
 * AT: `gl_PointSize = (0.02 * uDPR) * (1000.0 / length(mvPosition.xyz))`
 */
const ANTIMATTER_POSITION_VERT = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D tPos;
uniform sampler2D tSpawn;
uniform mat4 uProjection;
uniform mat4 uModelView;
uniform float uDPR;
uniform float uPointSize;

in vec2 aParticleUv;
out float vLife;
out vec3 vPos;

void main() {
    vec4 decodedPos = texture(tPos, aParticleUv);
    vec3 pos = decodedPos.xyz;
    float spawnLife = texture(tSpawn, aParticleUv).x;

    vLife = clamp(spawnLife, 0.0, 1.0);
    vPos = pos;

    vec4 mvPosition = uModelView * vec4(pos, 1.0);
    gl_PointSize = (uPointSize * uDPR) * (1000.0 / length(mvPosition.xyz));
    gl_Position = uProjection * mvPosition;
}
`;

/**
 * Render fragment — soft circle point sprite with life-based alpha.
 * AT's AntimatterBasicFrag.fs is just `gl_FragColor = vec4(1.0)`.
 * We extend it with soft circle + life fade for visual quality.
 */
const ANTIMATTER_RENDER_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform vec3 uColor;
uniform float uAlpha;

in float vLife;
in vec3 vPos;
out vec4 fragColor;

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r = dot(uv, uv);
    if (r > 1.0) discard;

    float alpha = (1.0 - r) * vLife * uAlpha;
    fragColor = vec4(uColor, alpha);
}
`;

// ── GL Helpers ────────────────────────────────────────────────────────────────

function compileShader(
  gl: WebGL2RenderingContext,
  type: GLenum,
  src: string,
): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s) ?? '';
    gl.deleteShader(s);
    throw new Error(`[Antimatter] Shader compile error:\n${log}\n\nSource:\n${src.slice(0, 500)}`);
  }
  return s;
}

function linkProgram(
  gl: WebGL2RenderingContext,
  vertSrc: string,
  fragSrc: string,
): WebGLProgram {
  const prog = gl.createProgram()!;
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog) ?? '';
    gl.deleteProgram(prog);
    throw new Error(`[Antimatter] Program link error:\n${log}`);
  }
  // Shaders can be detached after linking
  gl.detachShader(prog, vs);
  gl.detachShader(prog, fs);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return prog;
}

/**
 * Create an RGBA32F texture for GPGPU state storage.
 * NEAREST filtering — no interpolation between state pixels.
 */
function createFloat32Texture(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  data?: Float32Array | null,
): WebGLTexture {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(
    gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0,
    gl.RGBA, gl.FLOAT,
    data ?? null,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

/**
 * Create FBO attached to a texture.
 * Validates framebuffer completeness.
 */
function createFBO(
  gl: WebGL2RenderingContext,
  texture: WebGLTexture,
): WebGLFramebuffer {
  const fbo = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error(`[Antimatter] FBO incomplete: 0x${status.toString(16)}`);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return fbo;
}

/**
 * Create a ping-pong FBO pair for double-buffered GPGPU computation.
 * AT pattern: texA/fboA ↔ texB/fboB, phase toggles each frame.
 */
function createFBOPair(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  initialData?: Float32Array | null,
): AntimatterFBOPair {
  const texA = createFloat32Texture(gl, w, h, initialData);
  const texB = createFloat32Texture(gl, w, h, null);
  const fboA = createFBO(gl, texA);
  const fboB = createFBO(gl, texB);
  return { texA, texB, fboA, fboB, phase: 0, width: w, height: h };
}

/** Get current read texture from a ping-pong pair */
function readTex(pair: AntimatterFBOPair): WebGLTexture {
  return pair.phase === 0 ? pair.texA : pair.texB;
}

/** Get current write FBO from a ping-pong pair */
function writeFBO(pair: AntimatterFBOPair): WebGLFramebuffer {
  return pair.phase === 0 ? pair.fboB : pair.fboA;
}

/** Swap phase after a compute pass */
function swapPair(pair: AntimatterFBOPair): void {
  pair.phase = 1 - pair.phase;
}

// ── AT Defaults ──────────────────────────────────────────────────────────────

const AT_DEFAULTS = {
  decay:           0.005,
  decayRandom:     [0.5, 1.5] as [number, number],
  curlScale:       0.5,
  curlSpeed:       0.3,
  curlStrength:    0.02,
  originStrength:  0.05,
  hz:              60,
  dpr:             typeof window !== 'undefined' ? window.devicePixelRatio : 1,
} as const;

// ── AntimatterParticleSystem ─────────────────────────────────────────────────

/**
 * Active Theory Antimatter GPU particle system.
 *
 * FBO ping-pong architecture:
 *   positionPair  — particle world positions (RGBA32F: xyz=pos, w=data)
 *   spawnPair     — lifecycle state (RGBA32F: x=life, yzw=spawn position)
 *   originTex     — static: spawn origin positions (read-only)
 *   attribsTex    — static: per-particle random attributes (read-only)
 *   lifeTex       — dynamic: spawn trigger (upload from CPU when spawning)
 *
 * Each frame:
 *   1. spawnPass:   spawnPair.read → spawnPair.write (lifecycle decay / respawn)
 *   2. physicsPass: positionPair.read + spawn.read → positionPair.write (curl + attract)
 *   3. renderPass:  positionPair.read → screen (point sprites)
 *
 * @example
 * ```ts
 * import { AntimatterParticleSystem } from '$lib/particle/antimatter';
 *
 * const am = new AntimatterParticleSystem({
 *   gl,
 *   particleCount: 16384,
 *   curlScale: 0.5,
 *   curlSpeed: 0.3,
 *   originStrength: 0.05,
 * });
 *
 * // Set spawn origins (e.g. spline points, cell centres)
 * am.setOrigins(originPositions); // Float32Array, RGBA per particle
 *
 * // Per frame:
 * am.update(delta);
 * am.render(projectionMat, modelViewMat);
 * ```
 */
export class AntimatterParticleSystem {
  readonly gl: WebGL2RenderingContext;
  readonly particleCount: number;
  readonly texSize: number;

  readonly uniforms: AntimatterUniforms;

  // ── FBO ping-pong state ──────────────────────────────────────────────────
  private positionPair: AntimatterFBOPair;
  private spawnPair:    AntimatterFBOPair;

  // ── Static textures (read-only after init) ───────────────────────────────
  private originTex:  WebGLTexture;
  private attribsTex: WebGLTexture;
  private lifeTex:    WebGLTexture;

  // ── GL programs ──────────────────────────────────────────────────────────
  private spawnProgram:   WebGLProgram;
  private physicsProgram: WebGLProgram;
  private copyProgram:    WebGLProgram;
  private renderProgram:  WebGLProgram;

  // ── Geometry ──────────────────────────────────────────────────────────────
  private quadVAO:     WebGLVertexArrayObject;
  private particleVAO: WebGLVertexArrayObject;

  private time = 0;
  private _setupFrame = true;

  constructor(config: AntimatterConfig) {
    const { gl, particleCount } = config;
    this.gl = gl;
    this.particleCount = particleCount;

    // Texture side = ceil(sqrt(N)), ensuring perfect square layout
    const side = Math.ceil(Math.sqrt(particleCount));
    this.texSize = side;
    const total = side * side;

    // ── Uniforms (AT parameter defaults) ─────────────────────────────────
    this.uniforms = {
      uTime:           0,
      uDelta:          1,
      uMaxCount:       particleCount,
      decay:           config.decay          ?? AT_DEFAULTS.decay,
      decayRandom:     config.decayRandom    ?? [...AT_DEFAULTS.decayRandom],
      uCurlScale:      config.curlScale      ?? AT_DEFAULTS.curlScale,
      uCurlSpeed:      config.curlSpeed      ?? AT_DEFAULTS.curlSpeed,
      uCurlStrength:   config.curlStrength   ?? AT_DEFAULTS.curlStrength,
      uOriginStrength: config.originStrength  ?? AT_DEFAULTS.originStrength,
      HZ:              config.hz             ?? AT_DEFAULTS.hz,
      uDPR:            config.dpr            ?? AT_DEFAULTS.dpr,
      uSetup:          1.0, // First frame: clear to zero
    };

    // ── EXT_color_buffer_float required for RGBA32F render targets ────────
    const extCBF = gl.getExtension('EXT_color_buffer_float');
    if (!extCBF) {
      console.warn('[Antimatter] EXT_color_buffer_float not available — GPGPU may fail');
    }

    // ── Initialise particle data ─────────────────────────────────────────
    const posData    = new Float32Array(total * 4);
    const originData = new Float32Array(total * 4);
    const attribData = new Float32Array(total * 4);
    const lifeData   = new Float32Array(total * 4);

    for (let i = 0; i < total; i++) {
      // Initial position: scattered randomly in [-1,1]³
      posData[i * 4 + 0] = (Math.random() - 0.5) * 2;
      posData[i * 4 + 1] = (Math.random() - 0.5) * 2;
      posData[i * 4 + 2] = (Math.random() - 0.5) * 2;
      posData[i * 4 + 3] = 0; // data channel

      // Origins: same as initial position (can be overwritten via setOrigins)
      originData[i * 4 + 0] = posData[i * 4 + 0];
      originData[i * 4 + 1] = posData[i * 4 + 1];
      originData[i * 4 + 2] = posData[i * 4 + 2];
      originData[i * 4 + 3] = 1.0;

      // Random attributes (AT: used for decay variance, curl phase offset)
      attribData[i * 4 + 0] = Math.random();
      attribData[i * 4 + 1] = Math.random();
      attribData[i * 4 + 2] = Math.random();
      attribData[i * 4 + 3] = Math.random();

      // Life: x=0 (no spawn trigger initially)
      lifeData[i * 4 + 0] = 0.0;
      lifeData[i * 4 + 1] = 0.0;
      lifeData[i * 4 + 2] = 0.0;
      lifeData[i * 4 + 3] = 0.0;
    }

    // ── Create FBO pairs (ping-pong) ─────────────────────────────────────
    this.positionPair = createFBOPair(gl, side, side, posData);
    // Spawn pair: initialise with life=1 so particles are alive at start
    const spawnInit = new Float32Array(total * 4);
    for (let i = 0; i < total; i++) {
      spawnInit[i * 4 + 0] = Math.random(); // initial life [0,1]
      spawnInit[i * 4 + 1] = 0;
      spawnInit[i * 4 + 2] = 0;
      spawnInit[i * 4 + 3] = 0;
    }
    this.spawnPair = createFBOPair(gl, side, side, spawnInit);

    // ── Static textures ──────────────────────────────────────────────────
    this.originTex  = createFloat32Texture(gl, side, side, originData);
    this.attribsTex = createFloat32Texture(gl, side, side, attribData);
    this.lifeTex    = createFloat32Texture(gl, side, side, lifeData);

    // ── Compile shader programs ──────────────────────────────────────────
    this.spawnProgram   = linkProgram(gl, ANTIMATTER_PASS_VERT, ANTIMATTER_SPAWN_FRAG);
    this.physicsProgram = linkProgram(gl, ANTIMATTER_PASS_VERT,
                            config.physicsShaderSrc ?? ANTIMATTER_PHYSICS_FRAG);
    this.copyProgram    = linkProgram(gl, ANTIMATTER_PASS_VERT, ANTIMATTER_COPY_FRAG);
    this.renderProgram  = linkProgram(gl, ANTIMATTER_POSITION_VERT, ANTIMATTER_RENDER_FRAG);

    // ── Full-screen quad VAO (for GPGPU passes) ──────────────────────────
    const quadVerts = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const quadVBO = gl.createBuffer()!;
    this.quadVAO = gl.createVertexArray()!;
    gl.bindVertexArray(this.quadVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
    gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
    const aPosLoc = gl.getAttribLocation(this.spawnProgram, 'aPosition');
    gl.enableVertexAttribArray(aPosLoc);
    gl.vertexAttribPointer(aPosLoc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // ── Particle UV VAO (for render pass) ────────────────────────────────
    const uvData = new Float32Array(total * 2);
    for (let iy = 0; iy < side; iy++) {
      for (let ix = 0; ix < side; ix++) {
        const idx = (iy * side + ix) * 2;
        uvData[idx + 0] = (ix + 0.5) / side;
        uvData[idx + 1] = (iy + 0.5) / side;
      }
    }
    const uvVBO = gl.createBuffer()!;
    this.particleVAO = gl.createVertexArray()!;
    gl.bindVertexArray(this.particleVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, uvVBO);
    gl.bufferData(gl.ARRAY_BUFFER, uvData, gl.STATIC_DRAW);
    const aUvLoc = gl.getAttribLocation(this.renderProgram, 'aParticleUv');
    gl.enableVertexAttribArray(aUvLoc);
    gl.vertexAttribPointer(aUvLoc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    console.log(
      `[Antimatter] Initialised ${particleCount} particles ` +
      `in ${side}×${side} texture. ` +
      `curlScale=${this.uniforms.uCurlScale} ` +
      `curlSpeed=${this.uniforms.uCurlSpeed} ` +
      `decay=${this.uniforms.decay}`,
    );
  }

  // ── Public API: Update ──────────────────────────────────────────────────

  /**
   * Run one frame of GPU particle simulation.
   * Executes spawn pass → physics pass with FBO ping-pong.
   *
   * @param delta - Time delta in seconds
   */
  update(delta: number): void {
    const gl = this.gl;
    const u = this.uniforms;

    this.time += delta;
    u.uTime = this.time;
    u.uDelta = delta * u.HZ; // AT normalises delta to 60fps base

    // Save GL state that we modify
    const prevViewport = gl.getParameter(gl.VIEWPORT) as Int32Array;

    // ── Pass 1: Spawn lifecycle ──────────────────────────────────────────
    this._runSpawnPass(gl);

    // ── Pass 2: Physics update ───────────────────────────────────────────
    this._runPhysicsPass(gl);

    // Clear setup flag after first frame
    if (this._setupFrame) {
      u.uSetup = 0.0;
      this._setupFrame = false;
    }

    // Restore viewport
    gl.viewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
  }

  /**
   * Spawn pass: update lifecycle (decay / respawn).
   * Reads spawnPair.read → writes spawnPair.write.
   */
  private _runSpawnPass(gl: WebGL2RenderingContext): void {
    const u = this.uniforms;
    const prog = this.spawnProgram;
    gl.useProgram(prog);

    // Uniforms
    this._setF(prog, 'uMaxCount',    u.uMaxCount);
    this._setF(prog, 'uSetup',       u.uSetup);
    this._setF(prog, 'decay',        u.decay);
    this._set2F(prog, 'decayRandom', u.decayRandom[0], u.decayRandom[1]);
    this._setF(prog, 'HZ',           u.uDelta); // AT passes delta-normalised HZ
    this._setF(prog, 'fSize',        this.texSize);

    // Textures
    let texUnit = 0;
    this._bindTex(prog, 'tInput',   readTex(this.spawnPair), texUnit++);
    this._bindTex(prog, 'tLife',    this.lifeTex,             texUnit++);
    this._bindTex(prog, 'tAttribs', this.attribsTex,          texUnit++);

    // Render to write FBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, writeFBO(this.spawnPair));
    gl.viewport(0, 0, this.texSize, this.texSize);

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Swap spawn ping-pong
    swapPair(this.spawnPair);
  }

  /**
   * Physics pass: curl noise + origin attraction.
   * Reads positionPair.read + spawnPair.read → writes positionPair.write.
   */
  private _runPhysicsPass(gl: WebGL2RenderingContext): void {
    const u = this.uniforms;
    const prog = this.physicsProgram;
    gl.useProgram(prog);

    // Uniforms
    this._setF(prog, 'uMaxCount',       u.uMaxCount);
    this._setF(prog, 'uTime',           u.uTime);
    this._setF(prog, 'uDelta',          u.uDelta);
    this._setF(prog, 'uCurlScale',      u.uCurlScale);
    this._setF(prog, 'uCurlSpeed',      u.uCurlSpeed);
    this._setF(prog, 'uCurlStrength',   u.uCurlStrength);
    this._setF(prog, 'uOriginStrength', u.uOriginStrength);
    this._setF(prog, 'fSize',           this.texSize);
    this._setF(prog, 'HZ',              u.HZ);

    // Textures
    let texUnit = 0;
    this._bindTex(prog, 'tInput',   readTex(this.positionPair), texUnit++);
    this._bindTex(prog, 'tOrigin',  this.originTex,              texUnit++);
    this._bindTex(prog, 'tAttribs', this.attribsTex,             texUnit++);
    this._bindTex(prog, 'tSpawn',   readTex(this.spawnPair),     texUnit++);

    // Render to write FBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, writeFBO(this.positionPair));
    gl.viewport(0, 0, this.texSize, this.texSize);

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Swap position ping-pong
    swapPair(this.positionPair);
  }

  // ── Public API: Render ──────────────────────────────────────────────────

  /**
   * Render particles as point sprites using AntimatterPosition.vs.
   * AT: `gl_PointSize = (0.02 * uDPR) * (1000.0 / length(mvPosition.xyz))`
   *
   * @param projection - 4×4 projection matrix (Float32Array, 16 elements)
   * @param modelView  - 4×4 model-view matrix (Float32Array, 16 elements)
   * @param options    - Render overrides
   */
  render(
    projection: Float32Array,
    modelView: Float32Array,
    options: {
      pointSize?: number;
      color?: [number, number, number];
      alpha?: number;
    } = {},
  ): void {
    const gl = this.gl;
    const prog = this.renderProgram;
    gl.useProgram(prog);

    // Matrices
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProjection'), false, projection);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uModelView'), false, modelView);

    // Uniforms
    gl.uniform1f(gl.getUniformLocation(prog, 'uDPR'), this.uniforms.uDPR);
    gl.uniform1f(gl.getUniformLocation(prog, 'uPointSize'), options.pointSize ?? 0.02);
    gl.uniform3fv(
      gl.getUniformLocation(prog, 'uColor'),
      new Float32Array(options.color ?? [0.6, 0.8, 1.0]),
    );
    gl.uniform1f(gl.getUniformLocation(prog, 'uAlpha'), options.alpha ?? 0.8);

    // Textures
    let texUnit = 0;
    this._bindTex(prog, 'tPos',   readTex(this.positionPair), texUnit++);
    this._bindTex(prog, 'tSpawn', readTex(this.spawnPair),    texUnit++);

    // Additive blending (AT standard for particles)
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.depthMask(false);

    gl.bindVertexArray(this.particleVAO);
    gl.drawArrays(gl.POINTS, 0, this.particleCount);
    gl.bindVertexArray(null);

    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }

  // ── Public API: Copy pass ───────────────────────────────────────────────

  /**
   * AntimatterCopy: blit current position texture to an external FBO.
   * Useful for multi-consumer scenarios (e.g. fluid coupling, cell renderer).
   *
   * @param targetFBO - Destination framebuffer (null = default framebuffer)
   * @param w         - Target viewport width
   * @param h         - Target viewport height
   */
  copyPositionTo(targetFBO: WebGLFramebuffer | null, w: number, h: number): void {
    const gl = this.gl;
    const prog = this.copyProgram;
    gl.useProgram(prog);

    this._bindTex(prog, 'tDiffuse', readTex(this.positionPair), 0);

    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO);
    gl.viewport(0, 0, w, h);

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ── Public API: Data upload ─────────────────────────────────────────────

  /**
   * Set particle spawn origins.
   * AT: tOrigin texture, particles attract toward these positions.
   *
   * @param data - Float32Array of RGBA per particle (xyz=origin, w=1).
   *              Length must be texSize*texSize*4.
   */
  setOrigins(data: Float32Array): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.originTex);
    gl.texSubImage2D(
      gl.TEXTURE_2D, 0, 0, 0, this.texSize, this.texSize,
      gl.RGBA, gl.FLOAT, data,
    );
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /**
   * Trigger particle spawns by uploading to the life texture.
   * AT: tLife texture — when tLife.x > 0.5, spawn a particle at tLife.yzw.
   *
   * @param data - Float32Array of RGBA per particle.
   *              x > 0.5 = spawn trigger, yzw = spawn position.
   */
  setSpawnTriggers(data: Float32Array): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.lifeTex);
    gl.texSubImage2D(
      gl.TEXTURE_2D, 0, 0, 0, this.texSize, this.texSize,
      gl.RGBA, gl.FLOAT, data,
    );
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /**
   * Upload new positions directly (bypass physics for manual placement).
   * Writes into the current read texture of the position ping-pong.
   */
  setPositions(data: Float32Array): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, readTex(this.positionPair));
    gl.texSubImage2D(
      gl.TEXTURE_2D, 0, 0, 0, this.texSize, this.texSize,
      gl.RGBA, gl.FLOAT, data,
    );
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ── Public API: Accessors ───────────────────────────────────────────────

  /** Current position texture (for external sampling, e.g. tube/spline renderers) */
  get positionTexture(): WebGLTexture {
    return readTex(this.positionPair);
  }

  /** Current spawn/life texture */
  get spawnTexture(): WebGLTexture {
    return readTex(this.spawnPair);
  }

  /** Access the full position FBO pair for advanced multi-pass integration */
  get positionFBOPair(): AntimatterFBOPair {
    return this.positionPair;
  }

  /** Access the full spawn FBO pair */
  get spawnFBOPair(): AntimatterFBOPair {
    return this.spawnPair;
  }

  /** Set a uniform at runtime (e.g. driven by UIL parameter tweaks) */
  setUniform<K extends keyof AntimatterUniforms>(
    key: K,
    value: AntimatterUniforms[K],
  ): void {
    (this.uniforms as unknown as Record<string, unknown>)[key] = value;
  }

  // ── Cleanup ─────────────────────────────────────────────────────────────

  dispose(): void {
    const gl = this.gl;

    // Delete FBO pairs
    for (const pair of [this.positionPair, this.spawnPair]) {
      gl.deleteTexture(pair.texA);
      gl.deleteTexture(pair.texB);
      gl.deleteFramebuffer(pair.fboA);
      gl.deleteFramebuffer(pair.fboB);
    }

    // Delete static textures
    gl.deleteTexture(this.originTex);
    gl.deleteTexture(this.attribsTex);
    gl.deleteTexture(this.lifeTex);

    // Delete programs
    gl.deleteProgram(this.spawnProgram);
    gl.deleteProgram(this.physicsProgram);
    gl.deleteProgram(this.copyProgram);
    gl.deleteProgram(this.renderProgram);

    // Delete VAOs
    gl.deleteVertexArray(this.quadVAO);
    gl.deleteVertexArray(this.particleVAO);
  }

  // ── Private helpers ─────────────────────────────────────────────────────

  /** Set a float uniform (null-safe) */
  private _setF(prog: WebGLProgram, name: string, value: number): void {
    const loc = this.gl.getUniformLocation(prog, name);
    if (loc !== null) this.gl.uniform1f(loc, value);
  }

  /** Set a vec2 uniform (null-safe) */
  private _set2F(prog: WebGLProgram, name: string, x: number, y: number): void {
    const loc = this.gl.getUniformLocation(prog, name);
    if (loc !== null) this.gl.uniform2f(loc, x, y);
  }

  /** Bind a texture to a named sampler uniform at the given texture unit */
  private _bindTex(
    prog: WebGLProgram,
    name: string,
    tex: WebGLTexture,
    unit: number,
  ): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(gl.getUniformLocation(prog, name), unit);
  }
}

// ── Exported shader sources (for custom pass composition) ────────────────────

export {
  ANTIMATTER_PASS_VERT,
  ANTIMATTER_COPY_FRAG,
  ANTIMATTER_SPAWN_FRAG,
  ANTIMATTER_PHYSICS_FRAG,
  ANTIMATTER_POSITION_VERT,
  ANTIMATTER_RENDER_FRAG,
  ANTIMATTER_HELPERS_GLSL,
  RANGE_GLSL,
};

// ── Utility: createFBOPair (exported for external multi-pass use) ────────────

export { createFBOPair, readTex, writeFBO, swapPair };
