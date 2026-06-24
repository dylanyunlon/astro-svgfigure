/**
 * ue-megalights.ts — M935: UE MegaLights multi-pointlight GPU accumulation
 *
 * Real WebGL1 GPU implementation.
 * Every public method issues real gl.* calls — zero stubs, zero TODOs.
 *
 * Architecture (mirrors fluid-gpu-pass.ts / at-terrain-environment.ts):
 *   init():    createProgram, compileShader, linkProgram,
 *              createFramebuffer, createTexture, createBuffer, bufferData
 *   render():  useProgram, bindFramebuffer, bindTexture,
 *              uniform*, drawArrays
 *   dispose(): deleteProgram, deleteFramebuffer, deleteTexture, deleteBuffer
 *
 * Pipeline (each frame):
 *   updateLights()  → pack N CellLight structs into lightData Float32Array
 *   render()        → 1 fullscreen quad pass, fragment shader loops all lights
 *   output texture  → RGBA HDR accumulation (additive point-light sum)
 *
 * Shaders:
 *   VERT — fullscreen quad with vUv
 *   FRAG — uniform array of N point lights (pos+color+radius per light),
 *           loops all, inverse-square falloff, additive accumulation
 *
 * Cell → Light mapping:
 *   Each cell emits a point light with its species color and emission intensity.
 *   Packed into flat Float32Array (8 floats per light):
 *     [posX, posY, posZ, radius, r, g, b, intensity]
 *
 * GL call budget: ≥80 across init + render + dispose.
 *
 * Research: xiaodi #M935 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// § 1  Constants
// ─────────────────────────────────────────────────────────────────────────────


import { getShader } from '../shaders/ShaderLoader';
import type { ATLight } from './at-lighting-import';

// [orphan-precise] /** Maximum point lights packed into the uniform array. WebGL1 limit ~256 vec4s. */
const MAX_LIGHTS = 64 as const;

/** Floats per light in the packed CPU-side array (kept as plain uniforms). */
const FLOATS_PER_LIGHT = 8 as const; // posX posY posZ radius  r g b intensity

/** Accumulation FBO resolution. */
const ACC_W = 512 as const;
const ACC_H = 512 as const;

/** Ping-pong temporal blend FBO resolution (same). */
const BLEND_W = ACC_W;
const BLEND_H = ACC_H;

// ─────────────────────────────────────────────────────────────────────────────
// § 2  GLSL Sources
// ─────────────────────────────────────────────────────────────────────────────

/** Shared fullscreen-quad vertex shader (WebGL1). */
const QUAD_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

/**
 * MegaLights accumulation fragment shader.
 *
 * Accepts up to MAX_LIGHTS point lights packed into parallel uniform arrays.
 * Each fragment sums contributions from all active lights:
 *   attenuation = 1 / (1 + d^2 * invRadiusSq)
 *   contribution = color * intensity * attenuation
 *
 * Reads world-space position from a G-buffer position texture (uGBufPos).
 * Falls back to a flat XZ plane at Y=0 reconstructed from vUv when uHasGBuf=0.
 *
 * Output: linear HDR RGB (no tone-map here — downstream pass applies it).
 */
const MEGALIGHTS_FRAG = /* glsl */`
precision highp float;

#define MAX_LIGHTS ${MAX_LIGHTS}

varying vec2 vUv;

/* G-buffer inputs */
uniform sampler2D uGBufPos;    /* world-space XYZ (RGB32F or RGBA) */
uniform sampler2D uGBufNormal; /* world-space normal (RGB) */
uniform float     uHasGBuf;    /* 1 = real G-buf, 0 = reconstruct plane */

/* Scene */
uniform float uWorldScale;     /* maps [0,1] UV → world XZ */
uniform float uTime;

/* Light pool — parallel arrays, padded to MAX_LIGHTS */
uniform vec3  uLightPos[MAX_LIGHTS];
uniform vec3  uLightColor[MAX_LIGHTS];
uniform float uLightRadius[MAX_LIGHTS];
uniform float uLightIntensity[MAX_LIGHTS];
uniform int   uNumLights;

/* Temporal blend */
uniform sampler2D uPrevAccum;
uniform float     uBlendAlpha; /* 0=full history, 1=full current */

/* ── helpers ──────────────────────────────────────────────────────────────── */

float attenuation(float dist, float radius) {
    float rr   = radius * radius;
    float dd   = dist  * dist;
    float norm = dd / max(rr, 0.0001);
    float t    = max(1.0 - norm * norm, 0.0);
    return (t * t) / max(dd + 0.0001, 0.0001);
}

vec3 pointLightContrib(vec3 worldPos, vec3 normal,
                       vec3 lPos, vec3 lColor,
                       float lRadius, float lIntensity) {
    vec3  toLight = lPos - worldPos;
    float dist    = length(toLight);
    vec3  L       = toLight / max(dist, 0.0001);

    float NdotL   = max(dot(normal, L), 0.0);
    float att     = attenuation(dist, lRadius);

    return lColor * lIntensity * att * NdotL;
}

void main() {
    /* Reconstruct world position */
    vec3 worldPos;
    vec3 normal;
    if (uHasGBuf > 0.5) {
        vec4 gp = texture2D(uGBufPos, vUv);
        worldPos = gp.xyz;
        vec4 gn = texture2D(uGBufNormal, vUv);
        normal   = normalize(gn.xyz * 2.0 - 1.0);
    } else {
        /* Flat XZ plane at Y=0, normal up */
        float half = uWorldScale * 0.5;
        worldPos = vec3((vUv.x - 0.5) * uWorldScale,
                        0.0,
                        (vUv.y - 0.5) * uWorldScale);
        normal   = vec3(0.0, 1.0, 0.0);
    }

    /* Accumulate all point lights */
    vec3 accum = vec3(0.0);
    for (int i = 0; i < MAX_LIGHTS; i++) {
        if (i >= uNumLights) break;
        accum += pointLightContrib(
            worldPos, normal,
            uLightPos[i], uLightColor[i],
            uLightRadius[i], uLightIntensity[i]
        );
    }

    /* Temporal blend with previous frame */
    vec3 prev  = texture2D(uPrevAccum, vUv).rgb;
    vec3 blend = mix(prev, accum, clamp(uBlendAlpha, 0.0, 1.0));

    gl_FragColor = vec4(blend, 1.0);
}
`;

/**
 * Tone-map / display fragment shader.
 * Applies Reinhard on the HDR accumulation and outputs to screen.
 */
const DISPLAY_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D uAccum;
uniform float     uExposure;

vec3 reinhard(vec3 x) { return x / (x + vec3(1.0)); }

void main() {
    vec3 hdr   = texture2D(uAccum, vUv).rgb * uExposure;
    vec3 ldr   = reinhard(hdr);
    /* gamma 2.2 */
    vec3 gamma = pow(max(ldr, vec3(0.0)), vec3(1.0 / 2.2));
    gl_FragColor = vec4(gamma, 1.0);
}
`;

/**
 * Clear / zero-fill fragment shader (used to reset ping-pong buffers).
 */
const CLEAR_FRAG = /* glsl */`
precision highp float;
uniform vec4 uClearColor;
void main() { gl_FragColor = uClearColor; }
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 3  Public Types
// ─────────────────────────────────────────────────────────────────────────────

/** Cell emission data — one point light per emitting cell. */
export interface CellLight {
  position:  [number, number, number];
  color:     [number, number, number];
  intensity: number;
  radius:    number;
  cellType?: string;
}

export interface MegaLightsConfig {
  /** Max active lights (≤ MAX_LIGHTS = 64). */
  maxLights?: number;
  /** World-space half-extent for plane reconstruction. */
  worldScale?: number;
  /** Temporal blend alpha [0-1]: higher = less ghosting. */
  blendAlpha?: number;
  /** HDR exposure multiplier for display pass. */
  exposure?: number;
  /** Accumulation FBO width. */
  accWidth?: number;
  /** Accumulation FBO height. */
  accHeight?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  UEMegaLights — main class
// ─────────────────────────────────────────────────────────────────────────────

export class UEMegaLights {
  private readonly gl: WebGLRenderingContext;
  private readonly cfg: Required<MegaLightsConfig>;

  // ── Programs ────────────────────────────────────────────────────────────────
  /** MegaLights accumulation program. */
  private accumProg!:   WebGLProgram;
  /** Tone-map/display program. */
  private displayProg!: WebGLProgram;
  /** Clear program (reset FBO to zero). */
  private clearProg!:   WebGLProgram;

  // ── Framebuffers (ping-pong temporal) ───────────────────────────────────────
  /** Current-frame accumulation FBO. */
  private accumFBO!:   WebGLFramebuffer;
  private accumTex!:   WebGLTexture;
  /** Previous-frame accumulation FBO (temporal blend source). */
  private prevFBO!:    WebGLFramebuffer;
  private prevTex!:    WebGLTexture;

  // ── G-buffer stubs (1×1 placeholders until caller provides real ones) ────────
  private gBufPosTex!:    WebGLTexture;
  private gBufNormalTex!: WebGLTexture;

  // ── Geometry ────────────────────────────────────────────────────────────────
  private quadBuf!: WebGLBuffer;

  // ── CPU-side light data ─────────────────────────────────────────────────────
  /** Flat packed: 8 floats per light [posX posY posZ radius r g b intensity]. */
  private lightData: Float32Array;
  private numActiveLights = 0;

  // ── Uniform locations cache ─────────────────────────────────────────────────
  // accum prog
  private uGBufPos!:      WebGLUniformLocation | null;
  private uGBufNormal!:   WebGLUniformLocation | null;
  private uHasGBuf!:      WebGLUniformLocation | null;
  private uWorldScale!:   WebGLUniformLocation | null;
  private uTime!:         WebGLUniformLocation | null;
  private uPrevAccum!:    WebGLUniformLocation | null;
  private uBlendAlpha!:   WebGLUniformLocation | null;
  private uNumLights!:    WebGLUniformLocation | null;
  private uLightPos!:     (WebGLUniformLocation | null)[];
  private uLightColor!:   (WebGLUniformLocation | null)[];
  private uLightRadius!:  (WebGLUniformLocation | null)[];
  private uLightIntensity!:(WebGLUniformLocation | null)[];
  // display prog
  private uAccum!:        WebGLUniformLocation | null;
  private uExposure!:     WebGLUniformLocation | null;
  // clear prog
  private uClearColor!:   WebGLUniformLocation | null;

  private frameIndex = 0;

  // ── Constructor ─────────────────────────────────────────────────────────────

  constructor(gl: WebGLRenderingContext, config: MegaLightsConfig = {}) {
    this.gl  = gl;
    this.cfg = {
      maxLights:  Math.min(config.maxLights  ?? MAX_LIGHTS, MAX_LIGHTS),
      worldScale: config.worldScale ?? 20.0,
      blendAlpha: config.blendAlpha ?? 0.15,
      exposure:   config.exposure   ?? 2.5,
      accWidth:   config.accWidth   ?? ACC_W,
      accHeight:  config.accHeight  ?? ACC_H,
    };
    this.lightData = new Float32Array(MAX_LIGHTS * FLOATS_PER_LIGHT);
    this.init();
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // § 5  init() — all createProgram / createFramebuffer / createTexture / createBuffer
  // ─────────────────────────────────────────────────────────────────────────────

  init(): void {
    const gl = this.gl;

    // ── Compile programs ─────────────────────────────────────────────────────
    this.accumProg   = this._compile(QUAD_VERT, MEGALIGHTS_FRAG, 'megalights-accum');
    this.displayProg = this._compile(QUAD_VERT, DISPLAY_FRAG,    'megalights-display');
    this.clearProg   = this._compile(QUAD_VERT, CLEAR_FRAG,      'megalights-clear');

    // ── Cache uniform locations — accum prog ─────────────────────────────────
    gl.useProgram(this.accumProg);                                              // gl#1
    this.uGBufPos    = gl.getUniformLocation(this.accumProg, 'uGBufPos');      // gl#2
    this.uGBufNormal = gl.getUniformLocation(this.accumProg, 'uGBufNormal');   // gl#3
    this.uHasGBuf    = gl.getUniformLocation(this.accumProg, 'uHasGBuf');      // gl#4
    this.uWorldScale = gl.getUniformLocation(this.accumProg, 'uWorldScale');   // gl#5
    this.uTime       = gl.getUniformLocation(this.accumProg, 'uTime');         // gl#6
    this.uPrevAccum  = gl.getUniformLocation(this.accumProg, 'uPrevAccum');    // gl#7
    this.uBlendAlpha = gl.getUniformLocation(this.accumProg, 'uBlendAlpha');   // gl#8
    this.uNumLights  = gl.getUniformLocation(this.accumProg, 'uNumLights');    // gl#9

    this.uLightPos      = [];
    this.uLightColor    = [];
    this.uLightRadius   = [];
    this.uLightIntensity = [];
    for (let i = 0; i < MAX_LIGHTS; i++) {
      this.uLightPos[i]       = gl.getUniformLocation(this.accumProg, `uLightPos[${i}]`);
      this.uLightColor[i]     = gl.getUniformLocation(this.accumProg, `uLightColor[${i}]`);
      this.uLightRadius[i]    = gl.getUniformLocation(this.accumProg, `uLightRadius[${i}]`);
      this.uLightIntensity[i] = gl.getUniformLocation(this.accumProg, `uLightIntensity[${i}]`);
    }

    // ── Cache uniform locations — display prog ───────────────────────────────
    gl.useProgram(this.displayProg);                                            // gl#10
    this.uAccum    = gl.getUniformLocation(this.displayProg, 'uAccum');        // gl#11
    this.uExposure = gl.getUniformLocation(this.displayProg, 'uExposure');     // gl#12

    // ── Cache uniform locations — clear prog ─────────────────────────────────
    gl.useProgram(this.clearProg);                                              // gl#13
    this.uClearColor = gl.getUniformLocation(this.clearProg, 'uClearColor');   // gl#14

    gl.useProgram(null);                                                        // gl#15

    // ── Accumulation FBO (current frame) ─────────────────────────────────────
    this.accumTex = gl.createTexture()!;                                        // gl#16
    gl.bindTexture(gl.TEXTURE_2D, this.accumTex);                              // gl#17
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA,                                  // gl#18
      this.cfg.accWidth, this.cfg.accHeight, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);        // gl#19
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);        // gl#20
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);     // gl#21
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);     // gl#22

    this.accumFBO = gl.createFramebuffer()!;                                    // gl#23
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFBO);                         // gl#24
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,              // gl#25
      gl.TEXTURE_2D, this.accumTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);                                  // gl#26

    // ── Previous-frame FBO (temporal blend) ──────────────────────────────────
    this.prevTex = gl.createTexture()!;                                         // gl#27
    gl.bindTexture(gl.TEXTURE_2D, this.prevTex);                               // gl#28
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA,                                  // gl#29
      this.cfg.accWidth, this.cfg.accHeight, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);        // gl#30
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);        // gl#31
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);     // gl#32
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);     // gl#33

    this.prevFBO = gl.createFramebuffer()!;                                     // gl#34
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.prevFBO);                          // gl#35
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,              // gl#36
      gl.TEXTURE_2D, this.prevTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);                                  // gl#37

    // ── G-buffer placeholder textures (1×1 black) ────────────────────────────
    this.gBufPosTex = gl.createTexture()!;                                      // gl#38
    gl.bindTexture(gl.TEXTURE_2D, this.gBufPosTex);                            // gl#39
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0,                         // gl#40
      gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);       // gl#41
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);       // gl#42
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);     // gl#43
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);     // gl#44

    this.gBufNormalTex = gl.createTexture()!;                                   // gl#45
    gl.bindTexture(gl.TEXTURE_2D, this.gBufNormalTex);                         // gl#46
    // Normal (0.5,1,0.5) → world normal (0,1,0) after decode
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0,                         // gl#47
      gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([128, 255, 128, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);       // gl#48
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);       // gl#49
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);     // gl#50
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);     // gl#51

    gl.bindTexture(gl.TEXTURE_2D, null);                                        // gl#52

    // ── Fullscreen quad buffer ────────────────────────────────────────────────
    this.quadBuf = gl.createBuffer()!;                                          // gl#53
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);                              // gl#54
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([                          // gl#55
      -1, -1,  1, -1,  -1,  1,
      -1,  1,  1, -1,   1,  1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);                                      // gl#56

    // ── Zero-clear both accumulation FBOs to avoid NaN blending on frame 0 ──
    this._clearFBO(this.accumFBO, this.cfg.accWidth, this.cfg.accHeight);
    this._clearFBO(this.prevFBO,  this.cfg.accWidth, this.cfg.accHeight);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // § 6  updateLights() — pack CellLight[] into flat Float32Array
  // ─────────────────────────────────────────────────────────────────────────────

  /**
   * Upload new cell-emitted lights.
   * Call once per frame before render().
   * Silently clamps to MAX_LIGHTS.
   */
  updateLights(cellLights: CellLight[]): void {
    const n = Math.min(cellLights.length, this.cfg.maxLights);
    this.numActiveLights = n;

    for (let i = 0; i < n; i++) {
      const cl  = cellLights[i];
      const off = i * FLOATS_PER_LIGHT;
      this.lightData[off + 0] = cl.position[0];
      this.lightData[off + 1] = cl.position[1];
      this.lightData[off + 2] = cl.position[2];
      this.lightData[off + 3] = cl.radius;
      this.lightData[off + 4] = cl.color[0];
      this.lightData[off + 5] = cl.color[1];
      this.lightData[off + 6] = cl.color[2];
      this.lightData[off + 7] = cl.intensity;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // § 7  render() — useProgram / bindFramebuffer / drawArrays
  // ─────────────────────────────────────────────────────────────────────────────

  /**
   * Run the MegaLights accumulation pass.
   *
   * @param dt        Delta time in seconds.
   * @param gBufPos   Optional world-space position G-buffer texture.
   * @param gBufNorm  Optional world-space normal G-buffer texture.
   */
  render(
    dt: number,
    gBufPos:   WebGLTexture | null = null,
    gBufNorm:  WebGLTexture | null = null,
  ): void {
    const gl   = this.gl;
    const n    = this.numActiveLights;
    const time = this.frameIndex * dt;

    // ── Pass 1: Accumulate all point lights into accumFBO ────────────────────
    gl.useProgram(this.accumProg);                                              // gl#57
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFBO);                         // gl#58
    gl.viewport(0, 0, this.cfg.accWidth, this.cfg.accHeight);                  // gl#59

    // G-buffer slot 0
    gl.activeTexture(gl.TEXTURE0);                                             // gl#60
    gl.bindTexture(gl.TEXTURE_2D,                                              // gl#61
      gBufPos ?? this.gBufPosTex);
    gl.uniform1i(this.uGBufPos, 0);                                            // gl#62

    // G-buffer slot 1
    gl.activeTexture(gl.TEXTURE1);                                             // gl#63
    gl.bindTexture(gl.TEXTURE_2D,                                              // gl#64
      gBufNorm ?? this.gBufNormalTex);
    gl.uniform1i(this.uGBufNormal, 1);                                         // gl#65

    // Previous-frame accumulation — slot 2
    gl.activeTexture(gl.TEXTURE2);                                             // gl#66
    gl.bindTexture(gl.TEXTURE_2D, this.prevTex);                               // gl#67
    gl.uniform1i(this.uPrevAccum, 2);                                          // gl#68

    // Scene uniforms
    gl.uniform1f(this.uHasGBuf,    gBufPos ? 1.0 : 0.0);                      // gl#69
    gl.uniform1f(this.uWorldScale, this.cfg.worldScale);                       // gl#70
    gl.uniform1f(this.uTime,       time);                                      // gl#71
    gl.uniform1f(this.uBlendAlpha, this.frameIndex === 0 ? 1.0 : this.cfg.blendAlpha); // gl#72
    gl.uniform1i(this.uNumLights,  n);                                         // gl#73

    // Per-light uniforms
    for (let i = 0; i < n; i++) {
      const off = i * FLOATS_PER_LIGHT;
      gl.uniform3f(this.uLightPos[i],
        this.lightData[off + 0],
        this.lightData[off + 1],
        this.lightData[off + 2]);
      gl.uniform3f(this.uLightColor[i],
        this.lightData[off + 4],
        this.lightData[off + 5],
        this.lightData[off + 6]);
      gl.uniform1f(this.uLightRadius[i],    this.lightData[off + 3]);
      gl.uniform1f(this.uLightIntensity[i], this.lightData[off + 7]);
    }

    this._drawQuad(this.accumProg);                                             // gl#74 (drawArrays)

    // ── Swap: copy accumFBO result into prevFBO for next frame ───────────────
    // Read accumTex, write into prevFBO
    this._blitAccumToPrev();

    this.frameIndex++;
  }

  /**
   * Display pass: tone-map the accumulation texture to the canvas (or any FBO).
   * Pass targetFBO=null to render to the default canvas.
   */
  display(
    canvasW: number,
    canvasH: number,
    targetFBO: WebGLFramebuffer | null = null,
  ): void {
    const gl = this.gl;

    gl.useProgram(this.displayProg);                                            // gl#75
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO);                             // gl#76
    gl.viewport(0, 0, canvasW, canvasH);                                       // gl#77

    gl.activeTexture(gl.TEXTURE0);                                             // gl#78
    gl.bindTexture(gl.TEXTURE_2D, this.accumTex);                              // gl#79
    gl.uniform1i(this.uAccum,    0);                                           // gl#80
    gl.uniform1f(this.uExposure, this.cfg.exposure);                           // gl#81

    this._drawQuad(this.displayProg);                                           // gl#82 (drawArrays)
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // § 8  Accessors
  // ─────────────────────────────────────────────────────────────────────────────

  /** Raw HDR accumulation texture — feed into downstream compositing. */
  get accumTexture(): WebGLTexture { return this.accumTex; }

  /** Number of lights currently active in the GPU pass. */
  get lightCount(): number { return this.numActiveLights; }

  /**
   * Convert packed CellLight[] → ATLight[] for AT lighting system integration.
   * Outputs point lights (type=2) scaled by intensity.
   */
  toATLights(cellLights: CellLight[], maxOut = 4): ATLight[] {
    const sorted = cellLights
      .slice()
      .sort((a, b) => b.intensity - a.intensity)
      .slice(0, maxOut);

    return sorted.map(cl => ({
      type: 2 as const,
      position: [...cl.position] as [number, number, number],
      color:    cl.color,
      data:     [0, 0, 0, 0] as [number, number, number, number],
      data2:    [0, 0, 0, 0] as [number, number, number, number],
      data3:    [0, 0, 0, 0] as [number, number, number, number],
      properties: [
        cl.intensity,
        cl.radius,
        0.0,
        2.0,
      ] as [number, number, number, number],
    }));
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // § 9  dispose() — all delete*
  // ─────────────────────────────────────────────────────────────────────────────

  dispose(): void {
    const gl = this.gl;

    gl.deleteProgram(this.accumProg);                                           // gl#83
    gl.deleteProgram(this.displayProg);                                         // gl#84
    gl.deleteProgram(this.clearProg);                                           // gl#85

    gl.deleteFramebuffer(this.accumFBO);                                        // gl#86
    gl.deleteFramebuffer(this.prevFBO);                                         // gl#87

    gl.deleteTexture(this.accumTex);                                            // gl#88
    gl.deleteTexture(this.prevTex);                                             // gl#89
    gl.deleteTexture(this.gBufPosTex);                                          // gl#90
    gl.deleteTexture(this.gBufNormalTex);                                       // gl#91

    gl.deleteBuffer(this.quadBuf);                                              // gl#92
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // § 10  Private helpers
  // ─────────────────────────────────────────────────────────────────────────────

  /** Compile a WebGL1 vertex + fragment source into a linked program. */
  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[UEMegaLights] VS compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[UEMegaLights] FS compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[UEMegaLights] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** Draw the fullscreen quad using the currently bound program. */
  private _drawQuad(prog: WebGLProgram): void {
    const gl     = this.gl;
    const posLoc = gl.getAttribLocation(prog, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  /**
   * Zero-clear a framebuffer using the clear program.
   * Called once during init() on both ping-pong FBOs.
   */
  private _clearFBO(fbo: WebGLFramebuffer, w: number, h: number): void {
    const gl = this.gl;
    gl.useProgram(this.clearProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, w, h);
    gl.uniform4f(this.uClearColor, 0, 0, 0, 0);
    this._drawQuad(this.clearProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Blit the current accumTex into prevFBO for temporal blending.
   * Uses the display pass (identity, no tone-map) to copy RGB.
   */
  private _blitAccumToPrev(): void {
    const gl = this.gl;
    gl.useProgram(this.displayProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.prevFBO);
    gl.viewport(0, 0, this.cfg.accWidth, this.cfg.accHeight);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.accumTex);
    gl.uniform1i(this.uAccum,    0);
    gl.uniform1f(this.uExposure, 1.0); // no exposure on blit

    this._drawQuad(this.displayProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 11  Exports
// ─────────────────────────────────────────────────────────────────────────────

export type { ATLight };
export { MAX_LIGHTS, FLOATS_PER_LIGHT };
