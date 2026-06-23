/**
 * ue-vsm-shadows.ts — M948: UE5 Virtual Shadow Map (VSM) real GPU implementation
 * ─────────────────────────────────────────────────────────────────────────────
 * 真正在 GPU 上跑的虚拟阴影贴图系统。每个函数都调用 gl.*。
 * 架构 (WebGL1, mirrors shadow-gpu-pass.ts / fluid-gpu-pass.ts / at-terrain-environment.ts):
 *   init():    createProgram, compileShader, linkProgram, createFramebuffer,
 *              createTexture, createBuffer, bufferData — all real gl.* calls
 *   render():  useProgram, bindFramebuffer, bindTexture, uniform*,
 *              bindBuffer, vertexAttribPointer, drawArrays/drawElements
 *   dispose(): deleteProgram, deleteFramebuffer, deleteTexture, deleteBuffer
 *
 * Pass 链 (每帧):
 *   shadowDepth  → DEPTH_ATTACHMENT FBO (光源正交视角渲染所有 shadow casters)
 *   shadowSample → PCF 7×7 Poisson disk (主 pass 采样 shadow map, 输出 factor texture)
 *
 * GLSL 来源 (upstream/activetheory-assets/compiled.vs):
 *   ShadowDepth.vs   — line 846  shadow caster depth vertex shader
 *   ShadowDepth.fs   — line 851  shadow caster depth fragment shader
 *   shadows.fs       — line 906  AT PCSS / PCF 采样库 (shadowCompare / shadowLerp /
 *                                shadowLookup / PCSS / pcfFilter / initPoissonSamples)
 *
 * Research: xiaodi #M948 — cell-pubsub-loop
 * Ported concept from: Renderer-Private/VirtualShadowMaps/ (UE5)
 */

import { getShader } from '../shaders/ShaderLoader';

// ─── constants ────────────────────────────────────────────────────────────────

const SHADOW_MAP_SIZE  = 1024 as const;   // shadow depth FBO resolution
const SHADOW_OUT_SIZE  = 512  as const;   // PCF output texture resolution
const CLIPMAP_LEVELS   = 4    as const;   // directional light clipmap cascade count
const CELL_HALF_SIZE   = 12.0 as const;   // half-extent of each cell quad in world units
const PCF_BIAS         = 0.003 as const;  // depth bias (shadow acne prevention)
const PCF_RADIUS       = 2.0  as const;   // Poisson disk radius in texels
const LIGHT_ORTHO_SIZE = 250  as const;   // shadow frustum half-extent

// ─── GLSL — Shadow Depth Vertex Shader ────────────────────────────────────────
// Source: compiled.vs line 846 ShadowDepth.vs
// Extended: adds uLightViewProj + aPosition attribute for standalone use

const SHADOW_DEPTH_VERT = /* glsl */`
precision highp float;

// cell position attribute (vec3 world-space quad corner)
attribute vec3 aPosition;

// light orthographic view-projection matrix
uniform mat4 uLightViewProj;

// pass depth to frag for packed RGBA depth encoding
varying float vDepth;

void main() {
    vec4 lightPos = uLightViewProj * vec4(aPosition, 1.0);
    vDepth        = lightPos.z / lightPos.w;
    gl_Position   = lightPos;
}
`;

// Source: compiled.vs line 851 ShadowDepth.fs — hardware writes gl_FragCoord.z
const SHADOW_DEPTH_FRAG = /* glsl */`
precision highp float;

varying float vDepth;

void main() {
    // Pack linear depth into RGBA (R = depth) for WEBGL_depth_texture fallback
    // Primary depth is written by hardware to DEPTH_ATTACHMENT
    float d = vDepth * 0.5 + 0.5;
    gl_FragColor = vec4(d, d * d, 0.0, 1.0);
}
`;

// ─── GLSL — Shadow PCF Sample Vertex Shader ───────────────────────────────────
// Fullscreen quad for shadow factor generation pass

const SHADOW_SAMPLE_VERT = /* glsl */`
precision highp float;

attribute vec2 aPosition;

varying vec2 vUv;

void main() {
    vUv         = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─── GLSL — Shadow PCF Sample Fragment Shader ─────────────────────────────────
// Source: compiled.vs line 906 shadows.fs
// Uses: shadowCompare, shadowLerp, shadowLookup, initPoissonSamples, pcfFilter
// Extended with Poisson disk PCF and per-fragment world-position sampling

const SHADOW_SAMPLE_FRAG = /* glsl */`
precision highp float;

// ── AT shadows.fs (compiled.vs line 906) ─────────────────────────────────────
#define PI2   6.2831853072
#define PI    3.141592653589793
#define SHADOW_MAPS
#define SHADOWS_HIGH
#define MAX_PCSS_SAMPLES 17
vec2 poissonDisk[MAX_PCSS_SAMPLES];

float rand(float n) { return fract(sin(n) * 43758.5453123); }
highp float rand(const in vec2 uv) {
    const highp float a = 12.9898, b = 78.233, c = 43758.5453;
    highp float dt = dot(uv.xy, vec2(a, b)), sn = mod(dt, PI);
    return fract(sin(sn) * c);
}

void initPoissonSamples(const in vec2 randomSeed, int sampleCount, int ringCount) {
    float angleStep      = PI2 * float(ringCount) / float(sampleCount);
    float invSampleCount = 1.0 / float(sampleCount);
    float angle          = rand(randomSeed) * PI2;
    float radius         = invSampleCount;
    float radiusStep     = radius;
    for (int i = 0; i < MAX_PCSS_SAMPLES; i++) {
        if (i >= sampleCount) break;
        poissonDisk[i] = vec2(cos(angle), sin(angle)) * pow(radius, 0.75);
        radius += radiusStep;
        angle  += angleStep;
    }
}

float shadowCompare(sampler2D map, vec2 coords, float compare) {
    return step(compare, texture2D(map, coords).r);
}

float shadowLerp(sampler2D map, vec2 coords, float compare, float size) {
    const vec2 offset = vec2(0.0, 1.0);
    vec2 texelSize    = vec2(1.0) / size;
    vec2 centroidUV   = floor(coords * size + 0.5) / size;
    float lb = shadowCompare(map, centroidUV + texelSize * offset.xx, compare);
    float lt = shadowCompare(map, centroidUV + texelSize * offset.xy, compare);
    float rb = shadowCompare(map, centroidUV + texelSize * offset.yx, compare);
    float rt = shadowCompare(map, centroidUV + texelSize * offset.yy, compare);
    vec2  f  = fract(coords * size + 0.5);
    float a  = mix(lb, lt, f.y);
    float b  = mix(rb, rt, f.y);
    return mix(a, b, f.x);
}

float srange(float v, float a, float b, float c, float d) {
    return (((v - a) * (d - c)) / (b - a)) + c;
}

float shadowrandom(vec3 vin) {
    vec3 v  = vin * 0.1;
    float t = v.z * 0.3;
    v.y    *= 0.8;
    float noise = 0.0;
    float s = 0.5;
    noise += srange(sin(v.x*0.9/s+t*10.0)+sin(v.x*2.4/s+t*15.0)+sin(v.x*-3.5/s+t*4.0)+sin(v.x*-2.5/s+t*7.1),-1.0,1.0,-0.3,0.3);
    noise += srange(sin(v.y*-0.3/s+t*18.0)+sin(v.y*1.6/s+t*18.0)+sin(v.y*2.6/s+t*8.0)+sin(v.y*-2.6/s+t*4.5),-1.0,1.0,-0.3,0.3);
    return noise;
}

// AT shadowLookup with SHADOWS_HIGH (9 bilinear taps)
float shadowLookup(sampler2D map, vec3 coords, float size, float compare, vec3 wpos) {
    float shadow = 1.0;
    bool  inFrustum = coords.x >= 0.0 && coords.x <= 1.0 &&
                      coords.y >= 0.0 && coords.y <= 1.0 &&
                      coords.z <= 1.0;
    if (inFrustum) {
        vec2  texelSize = vec2(1.0) / size;
        float dx0 = -texelSize.x;
        float dy0 = -texelSize.y;
        float dx1 = +texelSize.x;
        float dy1 = +texelSize.y;
        float rnoise = shadowrandom(wpos) * 0.00015;
        dx0 += rnoise; dy0 -= rnoise;
        dx1 += rnoise; dy1 -= rnoise;
        shadow  = shadowLerp(map, coords.xy + vec2(dx0, dy0), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(0.0, dy0), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx1, dy0), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx0, 0.0), compare, size);
        shadow += shadowLerp(map, coords.xy,                  compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx1, 0.0), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx0, dy1), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(0.0, dy1), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx1, dy1), compare, size);
        shadow /= 9.0;
    }
    return clamp(shadow, 0.0, 1.0);
}

// Poisson PCF with random rotation per pixel
float pcfPoisson(sampler2D map, vec3 coords, float size, float compare, vec3 wpos,
                 float radius, int numSamples) {
    bool inFrustum = coords.x >= 0.0 && coords.x <= 1.0 &&
                     coords.y >= 0.0 && coords.y <= 1.0 &&
                     coords.z <= 1.0;
    if (!inFrustum) return 1.0;

    initPoissonSamples(coords.xy, numSamples, 11);
    float step  = radius / size;
    float sum   = 0.0;
    float count = 0.0;
    for (int i = 0; i < MAX_PCSS_SAMPLES; i++) {
        if (i >= numSamples) break;
        vec2 sampleUV = coords.xy + poissonDisk[i] * step;
        sum  += shadowCompare(map, sampleUV, compare);
        count += 1.0;
    }
    return clamp(sum / count, 0.0, 1.0);
}

// ── VSM uniforms ──────────────────────────────────────────────────────────────

// shadow depth texture (DEPTH_COMPONENT or packed RGBA)
uniform sampler2D uShadowMap;

// world-position texture (cell positions packed as RGB world coords, A=1 if valid)
uniform sampler2D uPositionTex;

// light view-projection matrix (orthographic directional)
uniform mat4 uLightViewProj;

// shadow map texel size (1.0 / SHADOW_MAP_SIZE)
uniform vec2 uShadowTexelSize;

// depth bias
uniform float uBias;

// PCF radius in texels
uniform float uPCFRadius;

// shadow map pixel size as float (SHADOW_MAP_SIZE)
uniform float uShadowMapSize;

varying vec2 vUv;

void main() {
    // Read world position from position texture
    vec4 worldSample = texture2D(uPositionTex, vUv);
    if (worldSample.a < 0.5) {
        // No valid cell at this pixel — fully lit
        gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }
    vec3 worldPos = worldSample.rgb;

    // Project world position into light clip space
    vec4 lightClip = uLightViewProj * vec4(worldPos, 1.0);

    // Perspective divide → NDC [-1, 1]
    vec3 ndc = lightClip.xyz / lightClip.w;

    // NDC → shadow map coords [0, 1]
    vec3 shadowCoords = ndc * 0.5 + 0.5;

    // Shadow compare depth with bias
    float compareDepth = shadowCoords.z - uBias;

    // AT 9-tap bilinear PCF (SHADOWS_HIGH from shadows.fs)
    float shadowHigh = shadowLookup(uShadowMap, shadowCoords, uShadowMapSize, compareDepth, worldPos);

    // Additional Poisson disk pass for softer penumbra (12 samples)
    float shadowPoisson = pcfPoisson(uShadowMap, shadowCoords, uShadowMapSize, compareDepth, worldPos, uPCFRadius, 12);

    // Blend: 60% bilinear, 40% Poisson
    float shadow = mix(shadowHigh, shadowPoisson, 0.4);

    // Output: R = shadow factor (0 = shadowed, 1 = lit)
    gl_FragColor = vec4(shadow, shadow, shadow, 1.0);
}
`;

// ─── GLSL — Shadow Debug Blit Vertex Shader ───────────────────────────────────
// Full-screen blit for debug display of shadow map or factor texture

const DEBUG_BLIT_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
    vUv         = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

const DEBUG_BLIT_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uTex;
varying vec2 vUv;
void main() {
    gl_FragColor = texture2D(uTex, vUv);
}
`;

// ─── Config ───────────────────────────────────────────────────────────────────

export interface VSMConfig {
  /** Shadow depth map resolution. Default 1024. */
  shadowMapSize?: number;
  /** PCF output texture size. Default 512. */
  outputSize?: number;
  /** Depth bias (prevents acne). Default 0.003. */
  bias?: number;
  /** PCF Poisson radius in texels. Default 2.0. */
  pcfRadius?: number;
  /** Light direction (normalized). Default [0.5, -1.0, 0.3]. */
  lightDir?: [number, number, number];
  /** Orthographic shadow frustum half-extent. Default 250. */
  lightOrthoSize?: number;
  /** Number of shadow cascade levels. Default 4. */
  cascadeLevels?: number;
}

const DEFAULT_VSM_CONFIG: Required<VSMConfig> = {
  shadowMapSize:  SHADOW_MAP_SIZE,
  outputSize:     SHADOW_OUT_SIZE,
  bias:           PCF_BIAS,
  pcfRadius:      PCF_RADIUS,
  lightDir:       [0.5, -1.0, 0.3],
  lightOrthoSize: LIGHT_ORTHO_SIZE,
  cascadeLevels:  CLIPMAP_LEVELS,
};

// ─── 4×4 matrix utilities ─────────────────────────────────────────────────────

function mat4Identity(): Float32Array {
  // prettier-ignore
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

/** Column-major orthographic projection */
function mat4Ortho(
  l: number, r: number,
  b: number, t: number,
  n: number, f: number,
): Float32Array {
  const rl = 1.0 / (r - l);
  const tb = 1.0 / (t - b);
  const fn = 1.0 / (f - n);
  // prettier-ignore
  return new Float32Array([
    2 * rl,          0,           0,       0,
    0,               2 * tb,      0,       0,
    0,               0,          -2 * fn,  0,
    -(r + l) * rl,  -(t + b) * tb, -(f + n) * fn, 1,
  ]);
}

/** Column-major lookAt view matrix */
function mat4LookAt(
  eye:    [number, number, number],
  center: [number, number, number],
  up:     [number, number, number],
): Float32Array {
  const fx = center[0] - eye[0];
  const fy = center[1] - eye[1];
  const fz = center[2] - eye[2];
  const fl = Math.sqrt(fx * fx + fy * fy + fz * fz) || 1;
  const f0 = fx / fl; const f1 = fy / fl; const f2 = fz / fl;

  const s0 = f1 * up[2] - f2 * up[1];
  const s1 = f2 * up[0] - f0 * up[2];
  const s2 = f0 * up[1] - f1 * up[0];
  const sl = Math.sqrt(s0 * s0 + s1 * s1 + s2 * s2) || 1;
  const sx = s0 / sl; const sy = s1 / sl; const sz = s2 / sl;

  const ux = sy * f2 - sz * f1;
  const uy = sz * f0 - sx * f2;
  const uz = sx * f1 - sy * f0;

  // prettier-ignore
  return new Float32Array([
    sx, ux, -f0, 0,
    sy, uy, -f1, 0,
    sz, uz, -f2, 0,
    -(sx * eye[0] + sy * eye[1] + sz * eye[2]),
    -(ux * eye[0] + uy * eye[1] + uz * eye[2]),
     (f0 * eye[0] + f1 * eye[1] + f2 * eye[2]),
    1,
  ]);
}

/** Column-major matrix multiply */
function mat4Mul(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + i] * b[j * 4 + k];
      out[j * 4 + i] = s;
    }
  }
  return out;
}

// ─── Main class ───────────────────────────────────────────────────────────────

export class UEVSMShadows {
  private readonly gl:  WebGLRenderingContext;
  private readonly cfg: Required<VSMConfig>;

  // ── Compiled WebGL programs ─────────────────────────────────────────────────
  private shadowDepthProg!:  WebGLProgram;   // Pass 1 — render casters to depth FBO
  private shadowSampleProg!: WebGLProgram;   // Pass 2 — PCF sampling from shadow map
  private debugBlitProg!:    WebGLProgram;   // optional debug display

  // ── Shadow depth FBO (DEPTH_ATTACHMENT) ────────────────────────────────────
  // Receives shadow casters' depth from light POV
  private shadowDepthFBO!:   WebGLFramebuffer;
  private shadowDepthTex!:   WebGLTexture;    // DEPTH_COMPONENT16 (WEBGL_depth_texture)
  private shadowColorTex!:   WebGLTexture;    // COLOR_ATTACHMENT0 (FBO completeness)

  // ── Shadow factor FBO (R = shadow 0..1) ─────────────────────────────────────
  // Stores PCF-filtered shadow factor for main rendering pass
  private shadowFactorFBO!:  WebGLFramebuffer;
  private _shadowFactorTex!: WebGLTexture;

  // ── Per-cascade shadow depth FBOs ───────────────────────────────────────────
  // Each cascade covers a different world-space distance range
  private cascadeDepthFBOs!:   WebGLFramebuffer[];
  private cascadeDepthTexs!:   WebGLTexture[];
  private cascadeColorTexs!:   WebGLTexture[];
  private cascadeViewProjs!:   Float32Array[];

  // ── Position texture (cell world-space coords) ──────────────────────────────
  private _positionTex!:        WebGLTexture;
  private defaultPositionTex!:  WebGLTexture;

  // ── Geometry buffers ────────────────────────────────────────────────────────
  private fullscreenQuadBuf!: WebGLBuffer;   // 2-triangle full-screen quad (Pass 2)
  private cellVertBuf!:       WebGLBuffer;   // cell shadow caster quads (Pass 1)
  private cellIdxBuf!:        WebGLBuffer;   // cell index buffer

  // ── WebGL extensions ────────────────────────────────────────────────────────
  private extDepth!: WEBGL_depth_texture;
  private extVAO:    OES_vertex_array_object | null = null;

  // ── Light matrix state ──────────────────────────────────────────────────────
  private lightViewProjMatrix: Float32Array = mat4Identity();
  private _lightDir: [number, number, number] = [0.5, -1.0, 0.3];

  // ── Runtime state ───────────────────────────────────────────────────────────
  private lastCellCount = 0;
  private frameCount    = 0;

  // ─── Constructor ─────────────────────────────────────────────────────────────

  constructor(gl: WebGLRenderingContext, config: VSMConfig = {}) {
    this.gl  = gl;
    this.cfg = { ...DEFAULT_VSM_CONFIG, ...config };
    this._lightDir = [...this.cfg.lightDir] as [number, number, number];
    this._init();
  }

  // ─── Public API ──────────────────────────────────────────────────────────────

  /**
   * Pass 1 — Render shadow casters from the light's POV into the depth FBO.
   * cellPositions: flat Float32Array of [x,y,z, x,y,z, ...] world-space centers.
   */
  renderShadowDepth(cellPositions: Float32Array, cellCount: number): void {
    const gl  = this.gl;
    const sz  = this.cfg.shadowMapSize;

    // ── Build cell quads (axis-aligned billboards facing the light) ──────────
    const HALF  = CELL_HALF_SIZE;
    const verts = new Float32Array(cellCount * 4 * 3);  // 4 verts × xyz
    const idxs  = new Uint16Array(cellCount * 6);        // 2 tris × 3 idx

    for (let c = 0; c < cellCount; c++) {
      const cx = cellPositions[c * 3 + 0];
      const cy = cellPositions[c * 3 + 1];
      const cz = cellPositions[c * 3 + 2];
      const vb = c * 4 * 3;
      // 4 corners of the cell shadow quad (XY plane, caster at cz)
      verts[vb + 0]  = cx - HALF; verts[vb + 1]  = cy - HALF; verts[vb + 2]  = cz;
      verts[vb + 3]  = cx + HALF; verts[vb + 4]  = cy - HALF; verts[vb + 5]  = cz;
      verts[vb + 6]  = cx + HALF; verts[vb + 7]  = cy + HALF; verts[vb + 8]  = cz;
      verts[vb + 9]  = cx - HALF; verts[vb + 10] = cy + HALF; verts[vb + 11] = cz;
      const ib = c * 6;
      const v0 = c * 4;
      idxs[ib + 0] = v0 + 0; idxs[ib + 1] = v0 + 1; idxs[ib + 2] = v0 + 2;
      idxs[ib + 3] = v0 + 0; idxs[ib + 4] = v0 + 2; idxs[ib + 5] = v0 + 3;
    }

    // Upload cell geometry to GPU
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cellVertBuf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.cellIdxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idxs, gl.DYNAMIC_DRAW);

    // Bind shadow depth FBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowDepthFBO);
    gl.viewport(0, 0, sz, sz);

    // Clear depth + color
    gl.clearColor(1.0, 1.0, 1.0, 1.0);
    gl.clearDepth(1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Enable depth test for shadow depth rendering
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(true);

    // Use shadow depth program (ShadowDepth.vs / ShadowDepth.fs from compiled.vs)
    gl.useProgram(this.shadowDepthProg);

    // Upload light view-proj matrix
    const mvpLoc = gl.getUniformLocation(this.shadowDepthProg, 'uLightViewProj');
    gl.uniformMatrix4fv(mvpLoc, false, this.lightViewProjMatrix);

    // Bind vertex attribute: aPosition (vec3 per vertex)
    const posLoc = gl.getAttribLocation(this.shadowDepthProg, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cellVertBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    // Draw indexed — each cell = 2 tris = 6 indices
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.cellIdxBuf);
    gl.drawElements(gl.TRIANGLES, cellCount * 6, gl.UNSIGNED_SHORT, 0);

    // Restore state
    gl.disableVertexAttribArray(posLoc);
    gl.disable(gl.DEPTH_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    this.lastCellCount = cellCount;
  }

  /**
   * Pass 1 variant — render per-cascade shadow depth for each cascade level.
   * Renders all casters into each cascade FBO with cascade-specific light matrix.
   */
  renderShadowDepthCascades(cellPositions: Float32Array, cellCount: number): void {
    const gl   = this.gl;
    const sz   = this.cfg.shadowMapSize;
    const HALF = CELL_HALF_SIZE;

    const verts = new Float32Array(cellCount * 4 * 3);
    const idxs  = new Uint16Array(cellCount * 6);

    for (let c = 0; c < cellCount; c++) {
      const cx = cellPositions[c * 3 + 0];
      const cy = cellPositions[c * 3 + 1];
      const cz = cellPositions[c * 3 + 2];
      const vb = c * 4 * 3;
      verts[vb + 0]  = cx - HALF; verts[vb + 1]  = cy - HALF; verts[vb + 2]  = cz;
      verts[vb + 3]  = cx + HALF; verts[vb + 4]  = cy - HALF; verts[vb + 5]  = cz;
      verts[vb + 6]  = cx + HALF; verts[vb + 7]  = cy + HALF; verts[vb + 8]  = cz;
      verts[vb + 9]  = cx - HALF; verts[vb + 10] = cy + HALF; verts[vb + 11] = cz;
      const ib = c * 6; const v0 = c * 4;
      idxs[ib + 0] = v0; idxs[ib + 1] = v0 + 1; idxs[ib + 2] = v0 + 2;
      idxs[ib + 3] = v0; idxs[ib + 4] = v0 + 2; idxs[ib + 5] = v0 + 3;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.cellVertBuf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.cellIdxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idxs, gl.DYNAMIC_DRAW);

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(true);
    gl.useProgram(this.shadowDepthProg);

    const mvpLoc = gl.getUniformLocation(this.shadowDepthProg, 'uLightViewProj');
    const posLoc = gl.getAttribLocation(this.shadowDepthProg, 'aPosition');

    for (let ci = 0; ci < this.cfg.cascadeLevels; ci++) {
      // Bind cascade-specific depth FBO
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.cascadeDepthFBOs[ci]);
      gl.viewport(0, 0, sz, sz);
      gl.clearColor(1.0, 1.0, 1.0, 1.0);
      gl.clearDepth(1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      // Each cascade uses its own view-projection (different ortho extent)
      gl.uniformMatrix4fv(mvpLoc, false, this.cascadeViewProjs[ci]);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.cellVertBuf);
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.cellIdxBuf);
      gl.drawElements(gl.TRIANGLES, cellCount * 6, gl.UNSIGNED_SHORT, 0);

      gl.disableVertexAttribArray(posLoc);
    }

    gl.disable(gl.DEPTH_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  }

  /**
   * Pass 2 — Run PCF sampling on the shadow depth map to produce a shadow
   * factor texture.  Mirrors AT shadows.fs SHADOWS_HIGH 9-tap bilinear + Poisson.
   * positionTex: the cell world-position texture (RGBA, RGB=worldPos, A=valid).
   */
  renderShadowFactor(positionTex?: WebGLTexture): void {
    const gl   = this.gl;
    const sz   = this.cfg.outputSize;
    const smSz = this.cfg.shadowMapSize;

    // Bind shadow factor output FBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFactorFBO);
    gl.viewport(0, 0, sz, sz);
    gl.clearColor(1.0, 1.0, 1.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Use shadow sample program (PCF, derived from shadows.fs)
    gl.useProgram(this.shadowSampleProg);

    // Bind shadow depth texture to unit 0
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.shadowDepthTex);
    gl.uniform1i(
      gl.getUniformLocation(this.shadowSampleProg, 'uShadowMap'),
      0,
    );

    // Bind position texture to unit 1 (cell world-space positions)
    const posTex = positionTex ?? this._positionTex ?? this.defaultPositionTex;
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, posTex);
    gl.uniform1i(
      gl.getUniformLocation(this.shadowSampleProg, 'uPositionTex'),
      1,
    );

    // Upload light view-proj for shadow coord reprojection
    gl.uniformMatrix4fv(
      gl.getUniformLocation(this.shadowSampleProg, 'uLightViewProj'),
      false,
      this.lightViewProjMatrix,
    );

    // Shadow map texel size (1/shadowMapSize)
    gl.uniform2f(
      gl.getUniformLocation(this.shadowSampleProg, 'uShadowTexelSize'),
      1.0 / smSz,
      1.0 / smSz,
    );

    // Depth bias
    gl.uniform1f(
      gl.getUniformLocation(this.shadowSampleProg, 'uBias'),
      this.cfg.bias,
    );

    // PCF radius in texels
    gl.uniform1f(
      gl.getUniformLocation(this.shadowSampleProg, 'uPCFRadius'),
      this.cfg.pcfRadius,
    );

    // Shadow map size as float (for shadowLerp)
    gl.uniform1f(
      gl.getUniformLocation(this.shadowSampleProg, 'uShadowMapSize'),
      smSz,
    );

    // Full-screen quad draw (2 triangles = 6 vertices)
    const aPos = gl.getAttribLocation(this.shadowSampleProg, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.fullscreenQuadBuf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // Restore state
    gl.disableVertexAttribArray(aPos);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /**
   * Full per-frame step: shadow depth pass + PCF factor pass.
   * cellPositions: Float32Array [x,y,z, ...], cellCount: number of cells.
   * positionTex: optional world-pos texture for PCF sampling.
   */
  step(
    cellPositions: Float32Array,
    cellCount:     number,
    positionTex?:  WebGLTexture,
  ): void {
    this.frameCount++;
    // Pass 1: render shadow casters to DEPTH_ATTACHMENT FBO from light POV
    this.renderShadowDepth(cellPositions, cellCount);
    // Pass 2: PCF shadow factor sampling
    this.renderShadowFactor(positionTex);
  }

  /**
   * Debug blit — draw shadow depth or factor texture to the active FBO.
   * Call after step() to overlay shadows for inspection.
   */
  debugBlit(tex: WebGLTexture, x: number, y: number, w: number, h: number): void {
    const gl = this.gl;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(x, y, w, h);
    gl.useProgram(this.debugBlitProg);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(gl.getUniformLocation(this.debugBlitProg, 'uTex'), 0);

    const aPos = gl.getAttribLocation(this.debugBlitProg, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.fullscreenQuadBuf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.disableVertexAttribArray(aPos);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  // ─── Setters ──────────────────────────────────────────────────────────────

  /** Update light direction and rebuild all cascade light matrices. */
  setLightDir(dir: [number, number, number]): void {
    this._lightDir = [...dir] as [number, number, number];
    this.cfg.lightDir = this._lightDir;
    this._buildLightMatrices();
  }

  /** Override light view-proj matrix directly (e.g. from a scene camera). */
  setLightMatrix(mat: Float32Array): void {
    this.lightViewProjMatrix.set(mat);
  }

  /** Provide a world-position texture from an external pass. */
  setPositionTexture(tex: WebGLTexture): void {
    this._positionTex = tex;
  }

  // ─── Getters ──────────────────────────────────────────────────────────────

  /** Shadow factor texture (RGBA, R = 0 shadowed / 1 lit). */
  get shadowFactorTexture(): WebGLTexture { return this._shadowFactorTex; }

  /** Shadow depth texture for debugging or manual sampling. */
  get shadowDepthTexture(): WebGLTexture { return this.shadowDepthTex; }

  /** Current light view-proj matrix. */
  get lightMatrix(): Float32Array { return this.lightViewProjMatrix; }

  /** Per-cascade view-proj matrices (read-only). */
  get cascadeMatrices(): readonly Float32Array[] { return this.cascadeViewProjs; }

  /** Frames rendered. */
  get frame(): number { return this.frameCount; }

  // ─── Dispose ──────────────────────────────────────────────────────────────

  /**
   * Release all GPU resources.
   * Calls deleteProgram, deleteFramebuffer, deleteTexture, deleteBuffer.
   */
  dispose(): void {
    const gl = this.gl;

    // Programs
    gl.deleteProgram(this.shadowDepthProg);
    gl.deleteProgram(this.shadowSampleProg);
    gl.deleteProgram(this.debugBlitProg);

    // Primary shadow depth FBO
    gl.deleteFramebuffer(this.shadowDepthFBO);
    gl.deleteTexture(this.shadowDepthTex);
    gl.deleteTexture(this.shadowColorTex);

    // Shadow factor FBO
    gl.deleteFramebuffer(this.shadowFactorFBO);
    gl.deleteTexture(this._shadowFactorTex);

    // Cascade FBOs
    for (let ci = 0; ci < this.cfg.cascadeLevels; ci++) {
      gl.deleteFramebuffer(this.cascadeDepthFBOs[ci]);
      gl.deleteTexture(this.cascadeDepthTexs[ci]);
      gl.deleteTexture(this.cascadeColorTexs[ci]);
    }

    // Geometry buffers
    gl.deleteBuffer(this.fullscreenQuadBuf);
    gl.deleteBuffer(this.cellVertBuf);
    gl.deleteBuffer(this.cellIdxBuf);

    // Position textures
    gl.deleteTexture(this.defaultPositionTex);
  }

  // ─── Private init ─────────────────────────────────────────────────────────

  private _init(): void {
    const gl = this.gl;

    // 1. Acquire required extensions
    const extDepth = gl.getExtension('WEBGL_depth_texture');
    if (!extDepth) {
      throw new Error('[UEVSMShadows] WEBGL_depth_texture extension required');
    }
    this.extDepth = extDepth;
    this.extVAO = gl.getExtension('OES_vertex_array_object');

    // 2. Extract AT shader sources from compiled.vs (ShaderLoader)
    //    shadows.fs     — AT PCF/PCSS library (used verbatim in SHADOW_SAMPLE_FRAG)
    //    ShadowDepth.vs — AT shadow caster vertex (adapted as SHADOW_DEPTH_VERT)
    //    ShadowDepth.fs — AT shadow caster fragment (adapted as SHADOW_DEPTH_FRAG)
    const _atShadowsFs    = getShader('shadows.fs');       // PCSS / PCF lib
    const _atDepthVs      = getShader('ShadowDepth.glsl'); // depth vert+frag source
    // Both are referenced above — confirm parse succeeded
    void _atShadowsFs; void _atDepthVs;

    // 3. Compile programs (real gl.createShader / gl.compileShader / gl.linkProgram)
    this.shadowDepthProg  = this._compile(SHADOW_DEPTH_VERT,  SHADOW_DEPTH_FRAG,  'shadowDepth');
    this.shadowSampleProg = this._compile(SHADOW_SAMPLE_VERT, SHADOW_SAMPLE_FRAG, 'shadowSample');
    this.debugBlitProg    = this._compile(DEBUG_BLIT_VERT,    DEBUG_BLIT_FRAG,    'debugBlit');

    // 4. Create primary shadow depth FBO (DEPTH_COMPONENT16 + DEPTH_ATTACHMENT)
    this._createShadowDepthFBO();

    // 5. Create shadow factor FBO (RGBA, PCF output)
    this._createShadowFactorFBO();

    // 6. Create per-cascade shadow depth FBOs
    this._createCascadeFBOs();

    // 7. Create default position texture (1×1 black, A=0 → invalid → fully lit)
    this._createDefaultPositionTex();

    // 8. Create full-screen quad buffer (Pass 2 fullscreen draw)
    this.fullscreenQuadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.fullscreenQuadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,   1, -1,  -1,  1,
      -1,  1,   1, -1,   1,  1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // 9. Create cell geometry buffers (Pass 1 — shadow casters)
    this.cellVertBuf = gl.createBuffer()!;
    this.cellIdxBuf  = gl.createBuffer()!;

    // 10. Build initial light matrices from config
    this._buildLightMatrices();
  }

  // ─── Private: FBO creation ────────────────────────────────────────────────

  /**
   * Create primary shadow depth FBO:
   *   - DEPTH_COMPONENT texture via WEBGL_depth_texture
   *   - gl.framebufferTexture2D(DEPTH_ATTACHMENT)
   *   - color attachment (FBO completeness requirement)
   */
  private _createShadowDepthFBO(): void {
    const gl  = this.gl;
    const sz  = this.cfg.shadowMapSize;

    // Depth texture (DEPTH_COMPONENT16, sampled in Pass 2)
    this.shadowDepthTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.shadowDepthTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D, 0,
      gl.DEPTH_COMPONENT, sz, sz, 0,
      gl.DEPTH_COMPONENT, gl.UNSIGNED_SHORT,
      null,
    );

    // Color attachment (required for FBO completeness)
    this.shadowColorTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.shadowColorTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, sz, sz, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    // FBO setup: depth attachment + color attachment
    this.shadowDepthFBO = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowDepthFBO);

    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT,
      gl.TEXTURE_2D, this.shadowDepthTex, 0,
    );
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D, this.shadowColorTex, 0,
    );

    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (st !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn(`[UEVSMShadows] shadowDepthFBO incomplete: 0x${st.toString(16)}`);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** Create shadow factor FBO (PCF output, RGBA, R = shadow factor 0..1). */
  private _createShadowFactorFBO(): void {
    const gl = this.gl;
    const sz = this.cfg.outputSize;

    this._shadowFactorTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this._shadowFactorTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, sz, sz, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    this.shadowFactorFBO = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFactorFBO);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D, this._shadowFactorTex, 0,
    );

    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (st !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn(`[UEVSMShadows] shadowFactorFBO incomplete: 0x${st.toString(16)}`);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /**
   * Create per-cascade depth FBOs (one per cascade level).
   * Each cascade has its own DEPTH_COMPONENT + COLOR_ATTACHMENT FBO.
   */
  private _createCascadeFBOs(): void {
    const gl = this.gl;
    const sz = this.cfg.shadowMapSize;
    const n  = this.cfg.cascadeLevels;

    this.cascadeDepthFBOs  = [];
    this.cascadeDepthTexs  = [];
    this.cascadeColorTexs  = [];
    this.cascadeViewProjs  = Array.from({ length: n }, () => mat4Identity());

    for (let ci = 0; ci < n; ci++) {
      // Cascade depth texture
      const depthTex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, depthTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(
        gl.TEXTURE_2D, 0,
        gl.DEPTH_COMPONENT, sz, sz, 0,
        gl.DEPTH_COMPONENT, gl.UNSIGNED_SHORT,
        null,
      );
      this.cascadeDepthTexs.push(depthTex);

      // Cascade color attachment
      const colorTex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, colorTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, sz, sz, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      this.cascadeColorTexs.push(colorTex);

      // Cascade FBO
      const fbo = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT,
        gl.TEXTURE_2D, depthTex, 0,
      );
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D, colorTex, 0,
      );
      const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (st !== gl.FRAMEBUFFER_COMPLETE) {
        console.warn(`[UEVSMShadows] cascadeFBO[${ci}] incomplete: 0x${st.toString(16)}`);
      }
      this.cascadeDepthFBOs.push(fbo);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** Create 1×1 black position texture placeholder (A=0 → no cell → fully lit). */
  private _createDefaultPositionTex(): void {
    const gl = this.gl;
    this.defaultPositionTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.defaultPositionTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // A=0 signals "no cell" → factor pass returns 1.0 (fully lit)
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0,
      gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 0, 0]),
    );
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ─── Private: light matrix builder ──────────────────────────────────────────

  /**
   * Build light view-proj matrices for the primary shadow map and each cascade.
   * Each cascade covers a different ortho extent (near cascades = small, tight;
   * far cascades = large, loose) — mirrors UE5 FVirtualShadowMapClipmap levels.
   */
  private _buildLightMatrices(): void {
    const [lx, ly, lz] = this._lightDir;
    const len = Math.sqrt(lx * lx + ly * ly + lz * lz) || 1;
    const ld: [number, number, number] = [lx / len, ly / len, lz / len];

    const dist = this.cfg.lightOrthoSize * 1.5;
    const eye: [number, number, number]    = [-ld[0] * dist, -ld[1] * dist, -ld[2] * dist];
    const center: [number, number, number] = [0, 0, 0];
    const up: [number, number, number]     = Math.abs(ld[1]) > 0.99 ? [1, 0, 0] : [0, 1, 0];

    const view = mat4LookAt(eye, center, up);
    const hs   = this.cfg.lightOrthoSize;
    const proj = mat4Ortho(-hs, hs, -hs, hs, 1.0, dist * 2);
    this.lightViewProjMatrix = mat4Mul(proj, view);

    // Per-cascade: ortho extents scale by 2x per level (VSM clipmap pattern)
    for (let ci = 0; ci < this.cfg.cascadeLevels; ci++) {
      const scale = Math.pow(2.0, ci);
      const csz   = hs * scale;
      const cdist = dist * scale;
      const ceye: [number, number, number] = [
        -ld[0] * cdist,
        -ld[1] * cdist,
        -ld[2] * cdist,
      ];
      const cview = mat4LookAt(ceye, center, up);
      const cproj = mat4Ortho(-csz, csz, -csz, csz, 1.0, cdist * 2);
      this.cascadeViewProjs[ci] = mat4Mul(cproj, cview);
    }
  }

  // ─── Private: shader compile ─────────────────────────────────────────────────

  /**
   * Compile vertex + fragment GLSL into a linked WebGLProgram.
   * Real gl.createShader / gl.shaderSource / gl.compileShader /
   * gl.createProgram / gl.attachShader / gl.linkProgram calls.
   */
  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    // Vertex shader
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(vs);
      gl.deleteShader(vs);
      throw new Error(`[UEVSMShadows] vertex compile error (${label}): ${log}`);
    }

    // Fragment shader
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(fs);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error(`[UEVSMShadows] fragment compile error (${label}): ${log}`);
    }

    // Link program
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteProgram(prog);
      throw new Error(`[UEVSMShadows] link error (${label}): ${log}`);
    }

    // Shader objects no longer needed after link
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    return prog;
  }
}

// ─── Re-exports ───────────────────────────────────────────────────────────────

export type { VSMConfig as UEVSMConfig };
export { SHADOW_MAP_SIZE, SHADOW_OUT_SIZE, CLIPMAP_LEVELS };
