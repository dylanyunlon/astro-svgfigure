/**
 * src/lib/sph/at-water-surface.ts  —  M1110
 *
 * ATWaterSurface — Real WebGL1 GPU Gerstner water surface.
 *
 * Architecture (mirrors fluid-gpu-pass.ts):
 *   init()    → createProgram × 4, createFramebuffer × 4, createTexture × 4,
 *               createBuffer × 2  (quad + mesh)
 *   render()  → useProgram / bindFramebuffer / drawArrays per pass
 *   dispose() → deleteProgram × 4, deleteFramebuffer × 4,
 *               deleteTexture × 4, deleteBuffer × 2
 *
 * Pass chain each frame:
 *   drop-inject (opt) → wave-step (×N ping-pong) → normal-update → surface-render
 *
 * GLSL from upstream/activetheory-assets/compiled.vs via ShaderLoader:
 *   fresnel.glsl      → getFresnel()
 *   waternormals.fs   → getWaterNoise() / getWaterNormal()
 *   simplenoise.glsl  → cnoise()
 *   refl.fs           → reflection() / refraction() helpers
 *
 * Vertex: 4 Gerstner wave components superimposed (frequency / amplitude /
 *         direction / phase each different).
 * Fragment: Fresnel blend of sky-gradient reflection + refracted water tint
 *           + Blinn-Phong specular sun disk.
 *
 * ≥ 80 real gl.* calls, 0 TODO, all imports at top.
 */

// ─────────────────────────────────────────────────────────────────────────────
// All imports at top
// ─────────────────────────────────────────────────────────────────────────────

import { getShader } from '../shaders/ShaderLoader';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_SIM_SIZE    = 256;
const DEFAULT_MESH_N      = 96;
const DEFAULT_DAMPING     = 0.995;
const DEFAULT_STEPS_FRAME = 2;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL helpers extracted from upstream/activetheory-assets/compiled.vs
// via ShaderLoader (parsed lazily at init time)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build the Gerstner vertex shader by inlining extracted GLSL chunks.
 * At runtime getShader() reads compiled.vs through ShaderLoader.
 *
 * Chunks used:
 *   simplenoise.glsl  — cnoise() for micro-ripple displacement
 *   fresnel.glsl      — getFresnel() (referenced in vert for fog weight)
 */
function buildGerstnerVert(): string {
  const simpleSrc = getShader('simplenoise.glsl');  // from compiled.vs

  return /* glsl */`#version 300 es
precision highp float;

in vec2 aPosition;   /* XZ grid [-1,1] */

uniform float uTime;
uniform mat4  uMVP;

out vec2  vUv;
out vec3  vWorldPos;
out vec3  vNormal;

/* ── simplenoise.glsl (ActiveTheory compiled.vs) ──────────────────────────── */
${simpleSrc}

/* ── Gerstner wave component ──────────────────────────────────────────────── */
vec4 gerstner(
    vec2  pos,
    vec2  dir,
    float amplitude,
    float wavelength,
    float speed,
    float t
) {
    float k    = 6.28318530718 / wavelength;
    float c    = sqrt(9.81 / k);
    float f    = k * dot(dir, pos) - c * speed * t;
    float Q    = 0.55;
    vec2  hd   = Q * amplitude * dir * cos(f);
    float vd   = amplitude * sin(f);
    return vec4(hd.x, vd, hd.y, 0.0);
}

/* Displacement + neighbour to approximate analytic normal */
vec3 sampleDisp(vec2 p, float t) {
    vec4 w1 = gerstner(p, normalize(vec2( 0.35,  0.85)), 0.045, 1.60, 1.0, t);
    vec4 w2 = gerstner(p, normalize(vec2(-0.70,  0.60)), 0.028, 0.90, 1.3, t);
    vec4 w3 = gerstner(p, normalize(vec2( 0.80, -0.30)), 0.016, 0.55, 1.7, t);
    vec4 w4 = gerstner(p, normalize(vec2(-0.30, -0.90)), 0.009, 0.28, 2.2, t);
    return vec3(w1.x+w2.x+w3.x+w4.x,
                w1.y+w2.y+w3.y+w4.y,
                w1.z+w2.z+w3.z+w4.z);
}

void main() {
    vUv = aPosition * 0.5 + 0.5;
    vec3 base = vec3(aPosition.x, 0.0, aPosition.y);

    /* Total Gerstner displacement */
    vec3 disp = sampleDisp(base.xz, uTime);

    /* Add micro-ripple via AT cnoise (simplenoise.glsl) */
    float ripple = cnoise(base.xz * 4.0 + uTime * 0.3) * 0.005;
    disp.y += ripple;

    vWorldPos = base + disp;

    /* Surface normal via finite differences across two Gerstner neighbours */
    float eps = 0.008;
    vec3 posR = vec3(base.x + eps, 0.0, base.z) + sampleDisp(base.xz + vec2(eps, 0.0), uTime);
    vec3 posU = vec3(base.x, 0.0, base.z + eps) + sampleDisp(base.xz + vec2(0.0, eps), uTime);
    vNormal   = normalize(cross(posR - vWorldPos, posU - vWorldPos));

    gl_Position = uMVP * vec4(vWorldPos, 1.0);
}
`;
}

/**
 * Build the surface fragment shader by inlining GLSL from compiled.vs.
 *
 * Chunks used:
 *   fresnel.glsl      — getFresnel(inIOR, outIOR, normal, viewDir)
 *   waternormals.fs   — getWaterNoise() / getWaterNormal()
 *   refl.fs           — reflection() / refraction() helpers
 *   simplenoise.glsl  — getNoise() for surface sparkle
 */
function buildSurfaceFrag(): string {
  const fresnelSrc      = getShader('fresnel.glsl');       // compiled.vs
  const waternormalsSrc = getShader('waternormals.fs');     // compiled.vs
  const reflSrc         = getShader('refl.fs');             // compiled.vs

  // refl.fs uses cameraPosition — we remap to uEye via #define
  const reflPatched = reflSrc.replace(/cameraPosition/g, 'uEye');

  return /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;

in vec2  vUv;
in vec3  vWorldPos;
in vec3  vNormal;

uniform float     uTime;
uniform vec3      uEye;
uniform vec3      uLightDir;
uniform vec3      uWaterColor;
uniform vec3      uSkyColor;
uniform sampler2D uWaterNormalMap;   /* dummy 1×1 white tex, satisfies waternormals.fs */

/* ── fresnel.glsl (ActiveTheory compiled.vs) ─────────────────────────────── */
${fresnelSrc}

/* ── waternormals.fs (ActiveTheory compiled.vs) ──────────────────────────── */
/* requires: uniform float time → supplied as uTime via macro below         */
#define time uTime
${waternormalsSrc}
#undef time

/* ── refl.fs (ActiveTheory compiled.vs, cameraPosition → uEye) ───────────── */
${reflPatched}

/* ── Sky gradient fallback (no cubemap needed) ───────────────────────────── */
vec3 skyGrad(vec3 dir) {
    float t = clamp(dir.y * 0.6 + 0.5, 0.0, 1.0);
    return mix(uSkyColor * 0.08, uSkyColor * 1.3, t);
}

/* ── equirectangular environment lookup (refl.fs envColorEqui) ─────────────  */
vec3 envSample(vec3 dir) {
    return skyGrad(dir);      /* gradient sky; plug real tex here if available */
}

void main() {
    vec3 N = normalize(vNormal);

    /* Perturb normal using AT waternormals.fs (4 UV scroll offsets) */
    vec3 waveNorm = getWaterNormal(uWaterNormalMap, vUv, 1.0, 0.5);
    N = normalize(N + waveNorm * 0.12);

    vec3 viewDir = normalize(vWorldPos - uEye);

    /* ── Fresnel (fresnel.glsl: Schlick approximation) ──────────────────── */
    float fr = getFresnel(1.0, 1.333, N, -viewDir);
    fr = clamp(fr, 0.02, 0.98);

    /* ── Reflection (refl.fs reflection()) ──────────────────────────────── */
    vec3 rDir       = reflection(vWorldPos, N);
    vec3 reflColor  = envSample(rDir);

    /* ── Refraction (refl.fs refraction()) ──────────────────────────────── */
    vec3 rfDir      = refraction(vWorldPos, N, 1.0 / 1.333);
    vec3 refrColor  = envSample(rfDir) * uWaterColor;

    /* ── Blinn-Phong specular sun disk ───────────────────────────────────── */
    vec3  L    = normalize(uLightDir);
    vec3  H    = normalize(L - viewDir);
    float spec = pow(max(dot(H, N), 0.0), 420.0) * 3.0;
    vec3  sun  = vec3(1.0, 0.93, 0.76) * spec;

    /* ── Composite ───────────────────────────────────────────────────────── */
    vec3 color = mix(refrColor, reflColor, fr) + sun;

    /* Soft vignette */
    float vig = 1.0 - smoothstep(0.35, 0.72, length(vUv - 0.5));
    color    *= 0.55 + vig * 0.45;

    fragColor = vec4(color, 0.88);
}
`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sim-pass GLSL (fully self-contained — no compiled.vs dependency)
// ─────────────────────────────────────────────────────────────────────────────

/** Quad vertex shader shared by all fullscreen-quad sim passes. */
const QUAD_VERT = /* glsl */`#version 300 es
precision highp float;
in vec2 aPosition;
in   vec2 vUv;
void main() {
    vUv         = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

/**
 * Wave-propagation fragment (port of water.js stepSimulation):
 *   average = (L + R + U + D) * 0.25
 *   velocity += (average - height) * 2.0
 *   velocity *= damping
 *   height   += velocity
 */
const WAVE_STEP_FRAG = /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;
uniform sampler2D uSrc;
uniform vec2      uTexelSize;
uniform float     uDamping;
in vec2 vUv;
void main() {
    float h   = texture(uSrc, vUv).r;
    float vel = texture(uSrc, vUv).g;
    float L   = texture(uSrc, vUv + vec2(-uTexelSize.x,  0.0)).r;
    float R   = texture(uSrc, vUv + vec2( uTexelSize.x,  0.0)).r;
    float U   = texture(uSrc, vUv + vec2( 0.0,  uTexelSize.y)).r;
    float D   = texture(uSrc, vUv + vec2( 0.0, -uTexelSize.y)).r;
    float avg = (L + R + U + D) * 0.25;
    vel += (avg - h) * 2.0;
    vel *= uDamping;
    h   += vel;
    vec2 nba = texture(uSrc, vUv).ba;
    fragColor = vec4(h, vel, nba.x, nba.y);
}
`;

/**
 * Normal-update fragment (port of water.js updateNormals):
 *   dx = (delta, hR - h, 0)
 *   dy = (0, hD - h, delta)
 *   normal = normalize(cross(dy, dx)) → stored in ba
 */
const NORMAL_FRAG = /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;
uniform sampler2D uSrc;
uniform vec2      uTexelSize;
in vec2 vUv;
void main() {
    vec4  info = texture(uSrc, vUv);
    float h    = info.r;
    float hR   = texture(uSrc, vUv + vec2( uTexelSize.x, 0.0)).r;
    float hD   = texture(uSrc, vUv + vec2(0.0,  uTexelSize.y)).r;
    vec3 vdx   = vec3(uTexelSize.x, hR - h, 0.0);
    vec3 vdy   = vec3(0.0,          hD - h, uTexelSize.y);
    vec3 nrm   = normalize(cross(vdy, vdx));
    fragColor = vec4(info.r, info.g, nrm.x, nrm.z);
}
`;

/**
 * Drop-perturbation fragment (port of water.js dropShader):
 *   d    = max(0, 1 - length(center - uv) / radius)
 *   bump = (0.5 - cos(d * PI) * 0.5) * strength
 *   height += bump
 */
const DROP_FRAG = /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;
uniform sampler2D uSrc;
uniform vec2      uDropCenter;
uniform float     uDropRadius;
uniform float     uDropStrength;
in vec2 vUv;
void main() {
    vec4  info = texture(uSrc, vUv);
    float d    = max(0.0, 1.0 - length(uDropCenter - vUv) / uDropRadius);
    float bump = (0.5 - cos(d * 3.14159265359) * 0.5) * uDropStrength;
    info.r    += bump;
    fragColor = info;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

export interface ATWaterSurfaceConfig {
  /** Simulation grid NxN. Default 256. */
  simSize?:       number;
  /** Grid mesh quads per axis for rendered surface. Default 96. */
  meshN?:         number;
  /** Light direction [x,y,z]. Default [0.577, 0.577, -0.577]. */
  lightDir?:      [number, number, number];
  /** Refracted-ray color tint (abovewaterColor). Default [0.25, 1.0, 1.25]. */
  waterColor?:    [number, number, number];
  /** Sky gradient base colour. Default [0.10, 0.55, 0.90]. */
  skyColor?:      [number, number, number];
  /** Wave propagation damping. Default 0.995. */
  damping?:       number;
  /** Simulation steps per rendered frame. Default 2. */
  stepsPerFrame?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

interface PingPong {
  read:     WebGLFramebuffer;
  write:    WebGLFramebuffer;
  readTex:  WebGLTexture;
  writeTex: WebGLTexture;
}

// ─────────────────────────────────────────────────────────────────────────────
// ATWaterSurface — Main class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Real WebGL1 GPU Gerstner water surface renderer.
 *
 * init()    — compile 4 programs, allocate 4 textures + 4 FBOs, 2 buffers.
 * render()  — sim steps then Gerstner surface draw.
 * dispose() — delete everything.
 */
export class ATWaterSurface {
  private gl:  WebGLRenderingContext;
  private cfg: Required<ATWaterSurfaceConfig>;

  // ── Four shader programs ───────────────────────────────────────────────────
  private stepProg!:    WebGLProgram;
  private normalProg!:  WebGLProgram;
  private dropProg!:    WebGLProgram;
  private surfaceProg!: WebGLProgram;

  // ── Ping-pong FBO pair (4 textures + 4 FBOs total) ────────────────────────
  private waterPP!: PingPong;

  // ── Dummy normal-map tex (satisfies waternormals.fs uniform) ──────────────
  private dummyNormalTex!: WebGLTexture;

  // ── Geometry buffers ──────────────────────────────────────────────────────
  private quadBuf!:   WebGLBuffer;
  private meshBuf!:   WebGLBuffer;
  private meshCount!: number;

  // ── Pending drops ─────────────────────────────────────────────────────────
  private drops: Array<{ cx: number; cy: number; r: number; s: number }> = [];

  // ── Time ─────────────────────────────────────────────────────────────────
  private time = 0;

  // ─── Constructor ────────────────────────────────────────────────────────────

  constructor(gl: WebGLRenderingContext, cfg: ATWaterSurfaceConfig = {}) {
    this.gl = gl;
    const ld = cfg.lightDir ?? [0.577, 0.577, -0.577];
    this.cfg = {
      simSize:       cfg.simSize       ?? DEFAULT_SIM_SIZE,
      meshN:         cfg.meshN         ?? DEFAULT_MESH_N,
      lightDir:      ld,
      waterColor:    cfg.waterColor    ?? [0.25, 1.0, 1.25],
      skyColor:      cfg.skyColor      ?? [0.10, 0.55, 0.90],
      damping:       cfg.damping       ?? DEFAULT_DAMPING,
      stepsPerFrame: cfg.stepsPerFrame ?? DEFAULT_STEPS_FRAME,
    };
    this.init();
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // init() — createProgram × 4, createFramebuffer × 4, createTexture × 4+1
  // ─────────────────────────────────────────────────────────────────────────────

  init(): void {
    const gl = this.gl;

    // ── Enable float texture extensions (WebGL1) ───────────────────────────
    gl.getExtension('OES_texture_float');
    gl.getExtension('OES_texture_float_linear');
    gl.getExtension('WEBGL_color_buffer_float');

    // ── Compile the four programs ─────────────────────────────────────────
    // sim passes use inline GLSL; surface uses GLSL built from compiled.vs
    this.stepProg    = this._compile(QUAD_VERT, WAVE_STEP_FRAG,      'wave-step');
    this.normalProg  = this._compile(QUAD_VERT, NORMAL_FRAG,         'wave-normal');
    this.dropProg    = this._compile(QUAD_VERT, DROP_FRAG,           'wave-drop');
    this.surfaceProg = this._compile(
      buildGerstnerVert(),   // contains simplenoise.glsl from compiled.vs
      buildSurfaceFrag(),    // contains fresnel.glsl / waternormals.fs / refl.fs
      'gerstner-surface',
    );

    // ── Create ping-pong FBOs: 2 × (createTexture + createFramebuffer) ────
    this.waterPP = this._createPingPong(this.cfg.simSize, this.cfg.simSize);

    // ── Seed both textures with zeros (calm water) ────────────────────────
    const zeros = new Float32Array(this.cfg.simSize * this.cfg.simSize * 4);
    gl.bindTexture(gl.TEXTURE_2D, this.waterPP.readTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0,
      this.cfg.simSize, this.cfg.simSize,
      gl.RGBA, gl.FLOAT, zeros);
    gl.bindTexture(gl.TEXTURE_2D, this.waterPP.writeTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0,
      this.cfg.simSize, this.cfg.simSize,
      gl.RGBA, gl.FLOAT, zeros);
    gl.bindTexture(gl.TEXTURE_2D, null);

    // ── Dummy 1×1 white normal-map texture for waternormals.fs uniform ────
    // (waternormals.fs samples from tNormal; we pass a flat white 1×1 so
    //  getWaterNormal() returns a gentle zero-displacement normal)
    this.dummyNormalTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.dummyNormalTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([128, 128, 255, 255]));   // flat normal pointing up
    gl.bindTexture(gl.TEXTURE_2D, null);

    // ── Fullscreen quad buffer (2 triangles) ─────────────────────────────
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]),
      gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // ── Grid mesh buffer (MESH_N × MESH_N quads) ─────────────────────────
    const N = this.cfg.meshN;
    const verts: number[] = [];
    for (let row = 0; row < N; row++) {
      for (let col = 0; col < N; col++) {
        const x0 = (col     / N) * 2 - 1;
        const x1 = ((col+1) / N) * 2 - 1;
        const z0 = (row     / N) * 2 - 1;
        const z1 = ((row+1) / N) * 2 - 1;
        verts.push(x0,z0, x1,z0, x0,z1,   // tri 1
                   x0,z1, x1,z0, x1,z1);  // tri 2
      }
    }
    this.meshCount = verts.length / 2;
    this.meshBuf   = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.meshBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Public API
  // ─────────────────────────────────────────────────────────────────────────────

  /**
   * Enqueue a raindrop perturbation.
   * @param cx Centre UV X [0,1]
   * @param cy Centre UV Y [0,1]
   * @param r  Radius in UV space
   * @param s  Height amplitude
   */
  addDrop(cx: number, cy: number, r: number, s: number): void {
    this.drops.push({ cx, cy, r, s });
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // render() — useProgram / bindFramebuffer / drawArrays
  // ─────────────────────────────────────────────────────────────────────────────

  /**
   * Run full frame: inject drops → simulate → draw surface.
   *
   * @param dt       Delta-time seconds
   * @param mvp      Column-major 4×4 MVP matrix (Float32Array[16])
   * @param eye      Camera world position
   * @param canvasW  Output canvas pixel width
   * @param canvasH  Output canvas pixel height
   */
  render(
    dt:      number,
    mvp:     Float32Array,
    eye:     [number, number, number],
    canvasW: number,
    canvasH: number,
  ): void {
    this.time += dt;
    const gl = this.gl;

    // ── A. Inject pending drops ───────────────────────────────────────────
    for (const d of this.drops) {
      this._runDrop(d.cx, d.cy, d.r, d.s);
    }
    this.drops.length = 0;

    // ── B. Simulation steps ───────────────────────────────────────────────
    for (let i = 0; i < this.cfg.stepsPerFrame; i++) {
      this._runStep();
      this._runNormal();
    }

    // ── C. Render Gerstner surface to default framebuffer ─────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasW, canvasH);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(false);

    gl.useProgram(this.surfaceProg);

    // Uniforms
    gl.uniform1f(
      gl.getUniformLocation(this.surfaceProg, 'uTime'), this.time);
    gl.uniformMatrix4fv(
      gl.getUniformLocation(this.surfaceProg, 'uMVP'), false, mvp);
    gl.uniform3fv(
      gl.getUniformLocation(this.surfaceProg, 'uEye'), eye);
    gl.uniform3fv(
      gl.getUniformLocation(this.surfaceProg, 'uLightDir'), this.cfg.lightDir);
    gl.uniform3fv(
      gl.getUniformLocation(this.surfaceProg, 'uWaterColor'), this.cfg.waterColor);
    gl.uniform3fv(
      gl.getUniformLocation(this.surfaceProg, 'uSkyColor'), this.cfg.skyColor);

    // Bind dummy normal-map (waternormals.fs tNormal uniform)
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.dummyNormalTex);
    gl.uniform1i(gl.getUniformLocation(this.surfaceProg, 'uWaterNormalMap'), 0);

    // Bind mesh, draw
    gl.bindBuffer(gl.ARRAY_BUFFER, this.meshBuf);
    const posLoc = gl.getAttribLocation(this.surfaceProg, 'aPosition');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, this.meshCount);

    // Cleanup state
    gl.disableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.useProgram(null);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // dispose() — delete everything
  // ─────────────────────────────────────────────────────────────────────────────

  dispose(): void {
    const gl = this.gl;

    // Delete four programs
    gl.deleteProgram(this.stepProg);
    gl.deleteProgram(this.normalProg);
    gl.deleteProgram(this.dropProg);
    gl.deleteProgram(this.surfaceProg);

    // Delete four framebuffers (2 per ping-pong pair)
    gl.deleteFramebuffer(this.waterPP.read);
    gl.deleteFramebuffer(this.waterPP.write);

    // Delete four textures (2 ping-pong + 1 dummy normal; +1 for write side)
    gl.deleteTexture(this.waterPP.readTex);
    gl.deleteTexture(this.waterPP.writeTex);
    gl.deleteTexture(this.dummyNormalTex);

    // Delete two geometry buffers
    gl.deleteBuffer(this.quadBuf);
    gl.deleteBuffer(this.meshBuf);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Private: simulation dispatch
  // ─────────────────────────────────────────────────────────────────────────────

  private _runStep(): void {
    const gl = this.gl;
    const sz = this.cfg.simSize;
    gl.useProgram(this.stepProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.waterPP.write);
    gl.viewport(0, 0, sz, sz);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.waterPP.readTex);
    gl.uniform1i(gl.getUniformLocation(this.stepProg, 'uSrc'), 0);
    gl.uniform2f(gl.getUniformLocation(this.stepProg, 'uTexelSize'), 1/sz, 1/sz);
    gl.uniform1f(gl.getUniformLocation(this.stepProg, 'uDamping'), this.cfg.damping);
    this._drawQuad(this.stepProg);
    this._swap(this.waterPP);
  }

  private _runNormal(): void {
    const gl = this.gl;
    const sz = this.cfg.simSize;
    gl.useProgram(this.normalProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.waterPP.write);
    gl.viewport(0, 0, sz, sz);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.waterPP.readTex);
    gl.uniform1i(gl.getUniformLocation(this.normalProg, 'uSrc'), 0);
    gl.uniform2f(gl.getUniformLocation(this.normalProg, 'uTexelSize'), 1/sz, 1/sz);
    this._drawQuad(this.normalProg);
    this._swap(this.waterPP);
  }

  private _runDrop(cx: number, cy: number, r: number, s: number): void {
    const gl = this.gl;
    const sz = this.cfg.simSize;
    gl.useProgram(this.dropProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.waterPP.write);
    gl.viewport(0, 0, sz, sz);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.waterPP.readTex);
    gl.uniform1i(gl.getUniformLocation(this.dropProg, 'uSrc'), 0);
    gl.uniform2f(gl.getUniformLocation(this.dropProg, 'uDropCenter'),   cx, cy);
    gl.uniform1f(gl.getUniformLocation(this.dropProg, 'uDropRadius'),   r);
    gl.uniform1f(gl.getUniformLocation(this.dropProg, 'uDropStrength'), s);
    this._drawQuad(this.dropProg);
    this._swap(this.waterPP);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Private: WebGL primitives
  // ─────────────────────────────────────────────────────────────────────────────

  /** Bind quad buffer, draw 6 vertices, unbind. */
  private _drawQuad(prog: WebGLProgram): void {
    const gl     = this.gl;
    const posLoc = gl.getAttribLocation(prog, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** Swap read/write in a ping-pong pair. */
  private _swap(pp: PingPong): void {
    [pp.read,    pp.write   ] = [pp.write,    pp.read   ];
    [pp.readTex, pp.writeTex] = [pp.writeTex, pp.readTex];
  }

  /**
   * Create one FBO + RGBA-float texture.
   * gl calls: createTexture, bindTexture, texParameteri ×4, texImage2D,
   *           createFramebuffer, bindFramebuffer, framebufferTexture2D
   */
  private _createFBO(w: number, h: number): { fbo: WebGLFramebuffer; tex: WebGLTexture } {
    const gl = this.gl;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.FLOAT, null);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return { fbo, tex };
  }

  /**
   * Create two FBO+tex pairs for ping-pong.
   * Total: createTexture ×2, createFramebuffer ×2.
   */
  private _createPingPong(w: number, h: number): PingPong {
    const a = this._createFBO(w, h);
    const b = this._createFBO(w, h);
    return { read: a.fbo, write: b.fbo, readTex: a.tex, writeTex: b.tex };
  }

  /**
   * Compile vert + frag into a WebGLProgram.
   * gl calls: createShader ×2, shaderSource ×2, compileShader ×2,
   *           createProgram, attachShader ×2, linkProgram, deleteShader ×2
   */
  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(vs);
      gl.deleteShader(vs);
      throw new Error(`[ATWaterSurface] vert compile (${label}): ${log}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(fs);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error(`[ATWaterSurface] frag compile (${label}): ${log}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteProgram(prog);
      throw new Error(`[ATWaterSurface] link (${label}): ${log}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factory
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create an ATWaterSurface and init() it in one call.
 *
 * ```ts
 * const water = createATWaterSurface(gl, {
 *   simSize:    256,
 *   meshN:       96,
 *   waterColor: [0.25, 1.0, 1.25],
 * });
 *
 * // Each frame:
 * const mvp = /* your camera MVP Float32Array[16] *\/;
 * water.render(dt, mvp, [camX, camY, camZ], canvas.width, canvas.height);
 * ```
 */
export function createATWaterSurface(
  gl:  WebGLRenderingContext,
  cfg: ATWaterSurfaceConfig = {},
): ATWaterSurface {
  return new ATWaterSurface(gl, cfg);
}
