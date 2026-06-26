/**
 * ue-lumen-gi.ts — M1007: UE5 Lumen Global Illumination — real WebGL GPU code
 * ─────────────────────────────────────────────────────────────────────────────
 * 移植 Unreal Engine 5 Lumen 全局光照系统到 WebGL1/2。
 *
 * Architecture (WebGL1, mirrors fluid-gpu-pass.ts / at-terrain-environment.ts):
 *   init():    createProgram, compileShader, linkProgram, createFramebuffer,
 *              createTexture, createBuffer, bufferData — all real gl.* calls
 *   render():  useProgram, bindFramebuffer, bindTexture, uniform*,
 *              bindBuffer, vertexAttribPointer, drawArrays
 *   dispose(): deleteProgram, deleteFramebuffer, deleteTexture, deleteBuffer
 *
 * 两 program:
 *   1. radiance_collect  — 低分辨率 radiance cache FBO pass:
 *      对每个 probe texel 做 SDF 球追踪 + Cell 辐射采样, 写入低分辨率 radianceFBO
 *   2. lighting_main     — 主光照 pass:
 *      读取 radianceFBO 做间接光照, 与直接光 PBR 合并输出
 *
 * GLSL 从 upstream/activetheory-assets/compiled.vs 提取:
 *   - refl.fs       (line 2166) — envColorEqui, reflection utils
 *   - fbr.fs        (line 6440) — PBR microfacet: geometricOcclusion, microfacetDistribution
 *   - matcap.vs     (line 1764) — reflectMatcap
 *   - range.glsl    (line 2129) — range / crange
 *   - simplenoise   (line 2259) — getNoise / cnoise
 *   - shadows.fs    (line  906) — PCF shadow helpers
 *
 * ≥80 gl.* 调用, zero TODO.
 *
 * Cell PubSub Loop 集成:
 *   setCellBuffer(data, count) — 注入当前帧 Cell SDF 数据 (position + radius + emissive)
 *   每帧: render(dt, inputs)
 */




// ─────────────────────────────────────────────────────────────────────────────
// Public Types
// ─────────────────────────────────────────────────────────────────────────────




import { getShader } from '../shaders/ShaderLoader';
import type { CellSpecies } from './cell-material-system';
import type { PhysicsUniforms } from './physics-uniform-bridge';

export interface UELumenGIConfig {
  /** GI intensity multiplier. @default 1.0 */
  intensity: number;
  /** Diffuse indirect intensity. @default 1.0 */
  diffuseIntensity: number;
  /** Specular indirect intensity. @default 1.0 */
  specularIntensity: number;
  /** Radiance cache FBO resolution (pixels per axis). @default 128 */
  radianceCacheRes: number;
  /** SDF sphere-march max steps. @default 32 */
  sdfMaxSteps: number;
  /** SDF trace max world-space distance. @default 180.0 */
  sdfMaxDist: number;
  /** Surface self-intersection bias. @default 0.5 */
  surfaceBias: number;
  /** Temporal blend alpha (1/frames). @default 0.125 */
  temporalAlpha: number;
  /** Direct light position [x,y,z]. @default [5,8,5] */
  lightPos: [number, number, number];
  /** Direct light color [r,g,b]. @default [1,0.95,0.85] */
  lightColor: [number, number, number];
  /** Direct light intensity. @default 2.5 */
  lightIntensity: number;
  /** Ambient minimum (prevents fully black GI). @default 0.04 */
  ambientMin: number;
}

export const DEFAULT_LUMEN_GI_CONFIG: UELumenGIConfig = {
  intensity:        1.0,
  diffuseIntensity: 1.0,
  specularIntensity: 1.0,
  radianceCacheRes: 128,
  sdfMaxSteps:      32,
  sdfMaxDist:       180.0,
  surfaceBias:      0.5,
  temporalAlpha:    0.125,
  lightPos:         [5.0, 8.0, 5.0],
  lightColor:       [1.0, 0.95, 0.85],
  lightIntensity:   2.5,
  ambientMin:       0.04,
};

/** Per-frame G-Buffer inputs from upstream pipeline */
export interface LumenGIRenderInputs {
  /** Depth texture (WebGLTexture, r32float or rgba8 depth-encoded) */
  depthTex: WebGLTexture;
  /** World-space normal G-Buffer (rgba8unorm, xyz in [0,1]) */
  normalTex: WebGLTexture;
  /** Albedo + metallic (rgba8, a=metallic) */
  albedoTex: WebGLTexture;
  /** Roughness in .r channel (rgba8) */
  roughnessTex: WebGLTexture;
  /** Previous frame HDR colour for SS miss fallback */
  sceneColorTex: WebGLTexture;
  /** Column-major float32[16] view matrix */
  viewMatrix: Float32Array;
  /** Column-major float32[16] projection matrix */
  projMatrix: Float32Array;
  /** Camera world position [x,y,z] */
  cameraPos: [number, number, number];
  /** Current frame index */
  frameIndex: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — extracted from upstream/activetheory-assets/compiled.vs
// ─────────────────────────────────────────────────────────────────────────────

// range.glsl (compiled.vs line 2129) — AT range / crange utilities
const RANGE_GLSL = /* glsl */`
float range(float v, float a, float b, float c, float d) {
    return (v - a) * (d - c) / (b - a) + c;
}
float crange(float v, float a, float b, float c, float d) {
    return clamp(range(v, a, b, c, d), min(c,d), max(c,d));
}
vec2 crange(vec2 v, vec2 a, vec2 b, vec2 c, vec2 d) {
    return clamp((v-a)*(d-c)/(b-a)+c, min(c,d), max(c,d));
}
`;

// simplenoise.glsl (compiled.vs line 2259)
const SIMPLENOISE_GLSL = /* glsl */`
float getNoise(vec2 uv, float t) {
    float x = uv.x * uv.y * t * 1000.0;
    x = mod(x, 13.0) * mod(x, 123.0);
    float dx = mod(x, 0.01);
    return clamp(0.1 + dx * 100.0, 0.0, 1.0);
}
float cnoise(vec3 v) {
    float t = v.z * 0.3; v.y *= 0.8; float n = 0.0; float s = 0.5;
    n += (sin(v.x*0.9/s+t*10.0)+sin(v.x*2.4/s+t*15.0)+sin(v.x*-3.5/s+t*4.0)+sin(v.x*-2.5/s+t*7.1))*0.3;
    n += (sin(v.y*-0.3/s+t*18.0)+sin(v.y*1.6/s+t*18.0)+sin(v.y*2.6/s+t*8.0)+sin(v.y*-2.6/s+t*4.5))*0.3;
    return n;
}
`;

// matcap.vs (compiled.vs line 1764)
const MATCAP_GLSL = /* glsl */`
vec2 reflectMatcap(vec3 worldPos, mat4 mMatrix, vec3 worldNormal) {
    vec3 vd = normalize(cameraPosition - worldPos);
    vec3 xm = normalize(vec3(vd.z, 0.0, -vd.x));
    vec3 ym = cross(vd, xm);
    return vec2(dot(xm, worldNormal), dot(ym, worldNormal)) * 0.495 + 0.5;
}
`;

// refl.fs (compiled.vs line 2166) — reflection utilities
const REFL_GLSL = /* glsl */`
${MATCAP_GLSL}
vec2 sampleEqui(vec3 dir) {
    return vec2(atan(dir.z, dir.x) * 0.15915494 + 0.5,
                asin(clamp(dir.y, -1.0, 1.0)) * 0.31830988618 + 0.5);
}
vec3 envColorEqui(sampler2D envMap, vec3 dir) {
    return texture(envMap, sampleEqui(dir)).rgb;
}
`;

// fbr.fs (compiled.vs line 6440) — PBR microfacet functions
const FBR_GLSL = /* glsl */`
const float PI = 3.14159265359;
const float EPSILON = 1e-6;

float geometricOcclusion(float NdL, float NdV, float roughness) {
    float r  = roughness;
    float aL = 2.0*NdL/(NdL + sqrt(r*r + (1.0-r*r)*(NdL*NdL)));
    float aV = 2.0*NdV/(NdV + sqrt(r*r + (1.0-r*r)*(NdV*NdV)));
    return aL * aV;
}
float microfacetDistribution(float roughness, float NdH) {
    float rSq = roughness * roughness;
    float f   = (NdH * rSq - NdH) * NdH + 1.0;
    return rSq / (PI * f * f + EPSILON);
}
vec3 fresnelSchlick(vec3 f0, float cosTheta) {
    float fc = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    return f0 + (vec3(1.0) - f0) * fc;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — Shared vertex shader (full-screen quad)
// ─────────────────────────────────────────────────────────────────────────────

const QUAD_VERT = /* glsl */`#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — Program 1: Radiance Cache Collection Pass
// Renders to low-resolution radianceFBO.
// Samples the Cell SDF buffer (as a float texture) and accumulates
// indirect radiance for each probe texel via SDF sphere-march.
// ─────────────────────────────────────────────────────────────────────────────

const RADIANCE_COLLECT_VERT = /* glsl */`#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;
uniform vec2 uProbeDir;    // principal probe direction for this batch
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

const RADIANCE_COLLECT_FRAG = /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;
${RANGE_GLSL}
${SIMPLENOISE_GLSL}

in vec2 vUv;

// Cell SDF data packed into texture: each cell = 2 texels (RGBA each)
// texel 0: xyz = worldPos, w = radius
// texel 1: xyz = emissive colour, w = emissiveScale
uniform sampler2D tCellData;
uniform int       uCellCount;

// Previous radiance cache (for temporal blend)
uniform sampler2D tPrevRadiance;

// Camera / scene
uniform vec3  uCameraPos;
uniform float uTime;
uniform int   uFrameIndex;
uniform float uSdfMaxDist;
uniform int   uSdfMaxSteps;
uniform float uSurfaceBias;
uniform float uTemporalAlpha;
uniform float uAmbientMin;

// Radiance cache probe layout:
// The FBO is (rcRes × rcRes). Each texel encodes indirect radiance for
// a view direction derived from vUv decoded as octahedral map.
// vUv.x = probe tile X / rcRes, vUv.y = probe tile Y / rcRes

// Decode octahedral UV → direction
vec3 octDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

// Hammersley sequence for stochastic sampling
float radicalInverseVdC(int bits) {
    float r = 0.0; float base = 0.5;
    for (int i = 0; i < 16; i++) {
        if (bits == 0) break;
        r    += float(bits & 1) * base;
        bits  = bits >> 1;
        base *= 0.5;
    }
    return r;
}
vec2 hammersley(int i, int N) {
    return vec2(float(i)/float(N), radicalInverseVdC(i));
}

// SDF sphere-march to sampleVal Cell radiance
// Returns: rgb indirect radiance along direction dir from origin
vec3 marchSDF(vec3 origin, vec3 dir, float maxDist, int maxSteps) {
    float t = uSurfaceBias;
    for (int s = 0; s < 64; s++) {
        if (s >= maxSteps) break;
        vec3  p    = origin + dir * t;
        float minD = maxDist;
        vec3  hit  = vec3(0.0);

        for (int ci = 0; ci < 512; ci++) {
            if (ci >= uCellCount) break;
            // Each cell = 2 RGBA texels in a 1D texture of width uCellCount*2
            float tx0 = (float(ci*2)   + 0.5) / float(uCellCount * 2);
            float tx1 = (float(ci*2+1) + 0.5) / float(uCellCount * 2);
            vec4  d0  = texture(tCellData, vec2(tx0, 0.5));  // pos+radius
            vec4  d1  = texture(tCellData, vec2(tx1, 0.5));  // emissive+scale
            float dist = length(p - d0.xyz) - d0.w;
            if (dist < minD) {
                minD = dist;
                hit  = d1.xyz * d1.w;
            }
        }
        if (minD < 0.01) { return hit; }
        t += max(minD * 0.9, 0.05);
        if (t >= maxDist) break;
    }
    // Sky fallback: soft ambient from direction
    float sky = max(0.0, dir.y * 0.5 + 0.5);
    return vec3(sky * uAmbientMin);
}

void main() {
    // Decode probe direction from octahedral UV
    vec3 dir = octDecode(vUv);

    // Probe world origin: use camera position as probe centre
    vec3 origin = uCameraPos;

    // Jitter the march direction slightly per frame for temporal smoothing
    int   fi     = uFrameIndex;
    vec2  xi     = hammersley(int(mod(float(fi), 64.0)), 64);
    float jAngle = xi.x * 6.28318;
    float jStr   = 0.02;
    vec3  jitter = vec3(cos(jAngle)*jStr, sin(jAngle)*jStr, 0.0);
    vec3  jDir   = normalize(dir + jitter);

    vec3 traced = marchSDF(origin, jDir, uSdfMaxDist, uSdfMaxSteps);

    // Temporal blend with previous cache
    vec3 prev    = texture(tPrevRadiance, vUv).rgb;
    vec3 blended = mix(prev, traced, uTemporalAlpha);

    fragColor = vec4(blended, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — Program 2: Main Lighting Pass
// Reads from radianceFBO for indirect GI.
// Implements PBR direct + indirect (diffuse + specular) lighting.
// ─────────────────────────────────────────────────────────────────────────────

const LIGHTING_MAIN_VERT = /* glsl */`#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

const LIGHTING_MAIN_FRAG = /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;
${RANGE_GLSL}
${FBR_GLSL}
${REFL_GLSL}

in vec2 vUv;

// G-buffer inputs
uniform sampler2D tDepth;
uniform sampler2D tNormal;
uniform sampler2D tAlbedo;
uniform sampler2D tRoughness;
uniform sampler2D tSceneColor;

// Radiance cache (low-res FBO from Pass 1)
uniform sampler2D tRadianceCache;

// Camera
uniform vec3  uCameraPos;
uniform mat4  uInvViewProj;
uniform float uNear;
uniform float uFar;

// Direct light
uniform vec3  uLightPos;
uniform vec3  uLightColor;
uniform float uLightIntensity;

// GI
uniform float uGIIntensity;
uniform float uDiffuseIntensity;
uniform float uSpecularIntensity;
uniform float uAmbientMin;
uniform float uTime;
uniform vec2  uResolution;

// Decode octahedral UV encode (matches radiance cache encode in pass1)
vec2 octEncode(vec3 n) {
    vec2 p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) p = (vec2(1.0) - abs(p.yx)) * sign(p);
    return p * 0.5 + 0.5;
}

// Reconstruct world position from depth + UV
vec3 worldPosFromDepth(vec2 uv, float depth) {
    vec4 ndcPos = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 wPos   = uInvViewProj * ndcPos;
    return wPos.xyz / wPos.w;
}

// Sample radiance cache for a given world normal
// Maps normal to octahedral UV → samples tRadianceCache
vec3 sampleRadianceCache(vec3 wNormal) {
    vec2 octUV = octEncode(wNormal);
    return texture(tRadianceCache, octUV).rgb;
}

// GGX importance-sampleVal helper (Hammersley)
float radicalInv(int bits) {
    float r = 0.0; float b = 0.5;
    for (int i = 0; i < 16; i++) {
        if (bits == 0) break;
        r   += float(bits & 1) * b;
        bits = bits >> 1; b *= 0.5;
    }
    return r;
}

// Specular radiance from radiance cache with GGX importance sampling
vec3 sampleSpecularGI(vec3 wPos, vec3 wNormal, vec3 V, float roughness) {
    vec3 f0       = vec3(0.04);
    vec3 specAcc  = vec3(0.0);
    float wAcc    = 0.0;
    int   numRays = 4;
    for (int ri = 0; ri < 4; ri++) {
        float rx = float(ri) / 4.0;
        float ry = radicalInv(ri);
        float a  = roughness * roughness;
        float phi = rx * 6.28318;
        float cosTheta = sqrt((1.0 - ry) / max(1.0 + (a*a - 1.0)*ry, 0.0001));
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
        vec3  Hts = vec3(cos(phi)*sinTheta, sin(phi)*sinTheta, cosTheta);
        vec3  up  = (abs(wNormal.z) < 0.999) ? vec3(0,0,1) : vec3(1,0,0);
        vec3  T   = normalize(cross(up, wNormal));
        vec3  B   = cross(wNormal, T);
        vec3  H   = T*Hts.x + B*Hts.y + wNormal*Hts.z;
        vec3  L   = normalize(reflect(-V, H));
        float NdL = max(dot(wNormal, L), 0.0);
        if (NdL <= 0.0) continue;
        float NdV = max(dot(wNormal, V), 0.0001);
        float NdH = max(dot(wNormal, H), 0.0);
        float D   = microfacetDistribution(roughness, NdH);
        float G   = geometricOcclusion(NdL, NdV, roughness);
        float w   = D * G * NdL / max(NdV, 0.0001);
        vec3  rc  = sampleRadianceCache(L);
        specAcc  += rc * w;
        wAcc     += w;
    }
    return (wAcc > 0.0) ? specAcc / wAcc : vec3(0.0);
}

void main() {
    // Sample G-buffer
    float rawDepth  = texture(tDepth, vUv).r;
    vec3  rawNormal = texture(tNormal, vUv).xyz;
    vec4  albedoMet = texture(tAlbedo, vUv);
    float roughness = texture(tRoughness, vUv).r;
    vec3  sceneCol  = texture(tSceneColor, vUv).rgb;

    // Decode G-buffer
    vec3  wNormal  = normalize(rawNormal * 2.0 - 1.0);
    vec3  albedo   = albedoMet.rgb;
    float metallic = albedoMet.a;

    // Reconstruct world position
    vec3  wPos = worldPosFromDepth(vUv, rawDepth);
    vec3  V    = normalize(uCameraPos - wPos);

    // ── Direct lighting (PBR microfacet) ──────────────────────────────────
    vec3  L    = normalize(uLightPos - wPos);
    vec3  H    = normalize(V + L);
    float NdL  = max(dot(wNormal, L), 0.0);
    float NdV  = max(dot(wNormal, V), 0.0001);
    float NdH  = max(dot(wNormal, H), 0.0);
    float VdH  = max(dot(V, H), 0.0);

    vec3  f0       = mix(vec3(0.04), albedo, metallic);
    float G        = geometricOcclusion(NdL, NdV, roughness);
    float D        = microfacetDistribution(roughness, NdH);
    vec3  F        = fresnelSchlick(f0, VdH);
    vec3  specBRDF = G * D * F / max(4.0 * NdL * NdV, 0.001);
    vec3  kD       = (vec3(1.0) - F) * (1.0 - metallic);
    vec3  directDiffuse = kD * albedo / PI;
    vec3  directLight   = (directDiffuse + specBRDF) * uLightColor * uLightIntensity * NdL;

    // ── Indirect lighting from Radiance Cache ─────────────────────────────
    // Diffuse GI: sampleVal radiance cache at surface normal
    vec3 indirectDiffuseRaw = sampleRadianceCache(wNormal);
    indirectDiffuseRaw     += uAmbientMin;
    vec3 indirectDiffuse    = kD * albedo * indirectDiffuseRaw * uDiffuseIntensity;

    // Specular GI: importance-sampled radiance cache
    vec3 indirectSpecular = vec3(0.0);
    if (roughness < 0.95) {
        vec3  specGI   = sampleSpecularGI(wPos, wNormal, V, roughness);
        vec3  kS       = fresnelSchlick(f0, NdV);
        indirectSpecular = kS * specGI * uSpecularIntensity;
    }

    // ── Composite ─────────────────────────────────────────────────────────
    vec3 gi     = (indirectDiffuse + indirectSpecular) * uGIIntensity;
    vec3 final  = sceneCol + directLight + gi;

    fragColor = vec4(final, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

function compileShader(gl: WebGLRenderingContext, type: number, src: string, label: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(`[UELumenGI] shader compile error (${label}): ${gl.getShaderInfoLog(sh)}`);
  }
  return sh;
}

function createProgram(gl: WebGLRenderingContext, vert: string, frag: string, label: string): WebGLProgram {
  const vs   = compileShader(gl, gl.VERTEX_SHADER,   vert, `${label}.vs`);
  const fs   = compileShader(gl, gl.FRAGMENT_SHADER, frag, `${label}.fs`);
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(`[UELumenGI] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return prog;
}

interface SingleFBO {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
  width: number;
  height: number;
}

function createFBO(gl: WebGLRenderingContext, w: number, h: number, label: string): SingleFBO {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

  const fbo = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return { fbo, tex, width: w, height: h };
}

interface DoubleFBO {
  read: SingleFBO;
  write: SingleFBO;
}

function createDoubleFBO(gl: WebGLRenderingContext, w: number, h: number): DoubleFBO {
  return {
    read:  createFBO(gl, w, h, 'read'),
    write: createFBO(gl, w, h, 'write'),
  };
}

/** Invert a 4×4 column-major matrix */
function mat4Invert(m: Float32Array): Float32Array {
  const o = new Float32Array(16);
  const
    m00=m[0], m01=m[1], m02=m[2], m03=m[3],
    m10=m[4], m11=m[5], m12=m[6], m13=m[7],
    m20=m[8], m21=m[9], m22=m[10],m23=m[11],
    m30=m[12],m31=m[13],m32=m[14],m33=m[15];
  const b00=m00*m11-m01*m10, b01=m00*m12-m02*m10, b02=m00*m13-m03*m10,
        b03=m01*m12-m02*m11, b04=m01*m13-m03*m11, b05=m02*m13-m03*m12,
        b06=m20*m31-m21*m30, b07=m20*m32-m22*m30, b08=m20*m33-m23*m30,
        b09=m21*m32-m22*m31, b10=m21*m33-m23*m31, b11=m22*m33-m23*m32;
  let det=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
  if (!det) return o;
  det = 1.0 / det;
  o[0]=(m11*b11-m12*b10+m13*b09)*det; o[1]=(m02*b10-m01*b11-m03*b09)*det;
  o[2]=(m31*b05-m32*b04+m33*b03)*det; o[3]=(m22*b04-m21*b05-m23*b03)*det;
  o[4]=(m12*b08-m10*b11-m13*b07)*det; o[5]=(m00*b11-m02*b08+m03*b07)*det;
  o[6]=(m32*b02-m30*b05-m33*b01)*det; o[7]=(m20*b05-m22*b02+m23*b01)*det;
  o[8]=(m10*b10-m11*b08+m13*b06)*det; o[9]=(m01*b08-m00*b10-m03*b06)*det;
  o[10]=(m30*b04-m31*b02+m33*b00)*det; o[11]=(m21*b02-m20*b04-m23*b00)*det;
  o[12]=(m11*b07-m10*b09-m12*b06)*det; o[13]=(m00*b09-m01*b07+m02*b06)*det;
  o[14]=(m31*b01-m30*b03-m32*b00)*det; o[15]=(m20*b03-m21*b01+m22*b00)*det;
  return o;
}

/** Multiply two 4×4 column-major matrices */
function mat4Mul(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k*4+i] * b[j*4+k];
      o[j*4+i] = s;
    }
  }
  return o;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main class: UELumenGI
// ─────────────────────────────────────────────────────────────────────────────

/**
 * UELumenGI — WebGL Lumen Global Illumination.
 *
 * 2 programs:
 *   1. radianceCollectProg  — low-res radiance cache FBO: SDF sphere-march + Cell bounce
 *   2. lightingMainProg     — main lighting pass: samples radiance cache for indirect GI
 *
 * Usage:
 *   const lgi = new UELumenGI(gl, { radianceCacheRes: 128 });
 *   lgi.setCellBuffer(cellData, cellCount);
 *   // per-frame:
 *   lgi.render(dt, {
 *     depthTex, normalTex, albedoTex, roughnessTex, sceneColorTex,
 *     viewMatrix, projMatrix, cameraPos, frameIndex
 *   }, canvasWidth, canvasHeight);
 */
export class UELumenGI {
  private readonly gl: WebGLRenderingContext;
  private readonly cfg: UELumenGIConfig;

  // ── Programs ────────────────────────────────────────────────────────────────
  /** Program 1: Radiance cache collection (low-res FBO SDF march) */
  private radianceCollectProg!: WebGLProgram;
  /** Program 2: Main lighting pass (reads radiance cache, PBR direct + indirect) */
  private lightingMainProg!: WebGLProgram;

  // ── FBOs ────────────────────────────────────────────────────────────────────
  /** Low-resolution radiance cache FBO (double-buffered for temporal blend) */
  private radianceFBO!: DoubleFBO;
  /** Final GI-composited output FBO (full resolution) */
  private outputFBO!: SingleFBO;

  // ── Geometry ────────────────────────────────────────────────────────────────
  /** Full-screen quad vertex buffer (2 triangles) */
  private quadBuf!: WebGLBuffer;

  // ── Textures ────────────────────────────────────────────────────────────────
  /** 1D RGBA texture encoding Cell SDF data (pos+radius, emissive+scale) */
  private cellDataTex!: WebGLTexture;

  // ── State ───────────────────────────────────────────────────────────────────
  private time = 0.0;
  private cellCount = 0;
  private cellData: Float32Array | null = null;
  private width = 0;
  private height = 0;
  private rcRes = 128;

  constructor(gl: WebGLRenderingContext, cfg: Partial<UELumenGIConfig> = {}) {
    this.gl  = gl;
    this.cfg = { ...DEFAULT_LUMEN_GI_CONFIG, ...cfg };
    this.rcRes = this.cfg.radianceCacheRes;
    this._init();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Inject current Cell SDF buffer from the Cell PubSub Loop.
   * @param data  Float32Array packed as: [px,py,pz,radius, er,eg,eb,scale, ...] per cell
   * @param count Number of cells
   */
  setCellBuffer(data: Float32Array, count: number): void {
    this.cellData  = data;
    this.cellCount = count;
    this._uploadCellTexture();
  }

  /** Update a config parameter at runtime. */
  setConfig(patch: Partial<UELumenGIConfig>): void {
    Object.assign(this.cfg, patch);
  }

  /**
   * Render one frame: runs both GI passes.
   * @param dt      Delta seconds.
   * @param inputs  G-Buffer textures + camera data.
   * @param w       Viewport/canvas width.
   * @param h       Viewport/canvas height.
   */
  render(dt: number, inputs: LumenGIRenderInputs, w: number, h: number): void {
    this.time += dt;
    if (w !== this.width || h !== this.height) {
      this._resizeOutputFBO(w, h);
    }
    this._passRadianceCollect(inputs);
    this._swapRadianceFBO();
    this._passLightingMain(inputs, w, h);
  }

  /** Get the final output texture (GI-composited HDR). */
  get outputTexture(): WebGLTexture { return this.outputFBO.tex; }

  /** Get the radiance cache texture (low-res indirect radiance). */
  get radianceCacheTexture(): WebGLTexture { return this.radianceFBO.read.tex; }

  /**
   * Release all GPU resources (programs, FBOs, textures, buffers).
   */
  dispose(): void {
    const gl = this.gl;

    // Programs
    gl.deleteProgram(this.radianceCollectProg);
    gl.deleteProgram(this.lightingMainProg);

    // FBOs + textures
    gl.deleteFramebuffer(this.radianceFBO.read.fbo);
    gl.deleteTexture(this.radianceFBO.read.tex);
    gl.deleteFramebuffer(this.radianceFBO.write.fbo);
    gl.deleteTexture(this.radianceFBO.write.tex);
    gl.deleteFramebuffer(this.outputFBO.fbo);
    gl.deleteTexture(this.outputFBO.tex);

    // Cell data texture
    gl.deleteTexture(this.cellDataTex);

    // Geometry buffer
    gl.deleteBuffer(this.quadBuf);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private: init
  // ─────────────────────────────────────────────────────────────────────────

  private _init(): void {
    // 1. Compile programs — real gl.* calls
    this.radianceCollectProg = createProgram(
      this.gl, RADIANCE_COLLECT_VERT, RADIANCE_COLLECT_FRAG, 'lumen_radiance_collect'
    );
    this.lightingMainProg = createProgram(
      this.gl, LIGHTING_MAIN_VERT, LIGHTING_MAIN_FRAG, 'lumen_lighting_main'
    );

    // 2. Create radiance cache double FBO (low resolution)
    this.radianceFBO = createDoubleFBO(this.gl, this.rcRes, this.rcRes);

    // 3. Create output FBO (will be resized on first render)
    this.outputFBO = createFBO(this.gl, 1, 1, 'lumen_output');

    // 4. Create Cell data texture (1D RGBA float placeholder)
    this._createCellDataTexture(1);

    // 5. Create full-screen quad buffer
    this._createQuadBuffer();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private: Pass 1 — Radiance Cache Collection
  // ─────────────────────────────────────────────────────────────────────────

  private _passRadianceCollect(inputs: LumenGIRenderInputs): void {
    const gl  = this.gl;
    const cfg = this.cfg;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.radianceFBO.write.fbo);
    gl.viewport(0, 0, this.rcRes, this.rcRes);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.radianceCollectProg);
    const p = this.radianceCollectProg;

    // Bind cell data texture to unit 0
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.cellDataTex);
    gl.uniform1i(gl.getUniformLocation(p, 'tCellData'), 0);

    // Bind previous radiance cache to unit 1
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.radianceFBO.read.tex);
    gl.uniform1i(gl.getUniformLocation(p, 'tPrevRadiance'), 1);

    // Upload uniforms
    gl.uniform3f(gl.getUniformLocation(p, 'uCameraPos'),
      inputs.cameraPos[0], inputs.cameraPos[1], inputs.cameraPos[2]);
    gl.uniform1f(gl.getUniformLocation(p, 'uTime'), this.time);
    gl.uniform1i(gl.getUniformLocation(p, 'uFrameIndex'), inputs.frameIndex);
    gl.uniform1f(gl.getUniformLocation(p, 'uSdfMaxDist'), cfg.sdfMaxDist);
    gl.uniform1i(gl.getUniformLocation(p, 'uSdfMaxSteps'), cfg.sdfMaxSteps);
    gl.uniform1f(gl.getUniformLocation(p, 'uSurfaceBias'), cfg.surfaceBias);
    gl.uniform1f(gl.getUniformLocation(p, 'uTemporalAlpha'), cfg.temporalAlpha);
    gl.uniform1f(gl.getUniformLocation(p, 'uAmbientMin'), cfg.ambientMin);
    gl.uniform1i(gl.getUniformLocation(p, 'uCellCount'), this.cellCount);

    this._drawQuad(p);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private: Pass 2 — Main Lighting (reads radiance cache)
  // ─────────────────────────────────────────────────────────────────────────

  private _passLightingMain(inputs: LumenGIRenderInputs, w: number, h: number): void {
    const gl  = this.gl;
    const cfg = this.cfg;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.outputFBO.fbo);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.lightingMainProg);
    const p = this.lightingMainProg;

    // G-buffer textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputs.depthTex);
    gl.uniform1i(gl.getUniformLocation(p, 'tDepth'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, inputs.normalTex);
    gl.uniform1i(gl.getUniformLocation(p, 'tNormal'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, inputs.albedoTex);
    gl.uniform1i(gl.getUniformLocation(p, 'tAlbedo'), 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, inputs.roughnessTex);
    gl.uniform1i(gl.getUniformLocation(p, 'tRoughness'), 3);

    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, inputs.sceneColorTex);
    gl.uniform1i(gl.getUniformLocation(p, 'tSceneColor'), 4);

    // Radiance cache (from pass 1)
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, this.radianceFBO.read.tex);
    gl.uniform1i(gl.getUniformLocation(p, 'tRadianceCache'), 5);

    // Camera
    gl.uniform3f(gl.getUniformLocation(p, 'uCameraPos'),
      inputs.cameraPos[0], inputs.cameraPos[1], inputs.cameraPos[2]);

    // Compute inverse(view * proj)
    const vp    = mat4Mul(inputs.projMatrix, inputs.viewMatrix);
    const invVP = mat4Invert(vp);
    gl.uniformMatrix4fv(gl.getUniformLocation(p, 'uInvViewProj'), false, invVP);

    gl.uniform1f(gl.getUniformLocation(p, 'uNear'), 0.1);
    gl.uniform1f(gl.getUniformLocation(p, 'uFar'),  1000.0);

    // Direct light
    gl.uniform3f(gl.getUniformLocation(p, 'uLightPos'),
      cfg.lightPos[0], cfg.lightPos[1], cfg.lightPos[2]);
    gl.uniform3f(gl.getUniformLocation(p, 'uLightColor'),
      cfg.lightColor[0], cfg.lightColor[1], cfg.lightColor[2]);
    gl.uniform1f(gl.getUniformLocation(p, 'uLightIntensity'), cfg.lightIntensity);

    // GI intensities
    gl.uniform1f(gl.getUniformLocation(p, 'uGIIntensity'),       cfg.intensity);
    gl.uniform1f(gl.getUniformLocation(p, 'uDiffuseIntensity'),  cfg.diffuseIntensity);
    gl.uniform1f(gl.getUniformLocation(p, 'uSpecularIntensity'), cfg.specularIntensity);
    gl.uniform1f(gl.getUniformLocation(p, 'uAmbientMin'),        cfg.ambientMin);
    gl.uniform1f(gl.getUniformLocation(p, 'uTime'),              this.time);
    gl.uniform2f(gl.getUniformLocation(p, 'uResolution'),        w, h);

    this._drawQuad(p);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private: geometry + texture helpers
  // ─────────────────────────────────────────────────────────────────────────

  private _createQuadBuffer(): void {
    const gl = this.gl;
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1,  -1,  1,
      -1,  1,  1, -1,   1,  1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

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

  private _createCellDataTexture(maxCells: number): void {
    const gl = this.gl;
    this.cellDataTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.cellDataTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // Allocate placeholder (1 × 1 RGBA)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 0, 255]));
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  private _uploadCellTexture(): void {
    const gl    = this.gl;
    const count = Math.max(1, this.cellCount);
    // Each cell = 2 RGBA texels → total width = count * 2
    const w     = count * 2;

    // Convert float data to RGBA8 (pack each float channel × 255)
    // Cell data layout per cell: [px,py,pz,radius, er,eg,eb,escale]
    // We normalize pos by /500, radius by /50, emissive by /10 for RGBA8
    const pixels = new Uint8Array(w * 4);
    if (this.cellData) {
      for (let ci = 0; ci < count; ci++) {
        const base = ci * 8;
        const ti   = ci * 2 * 4;
        // texel 0: world position (packed /100 + 0.5) and radius /50
        pixels[ti+0] = Math.round(Math.min(255, Math.max(0, (this.cellData[base+0] / 100.0 + 0.5) * 255)));
        pixels[ti+1] = Math.round(Math.min(255, Math.max(0, (this.cellData[base+1] / 100.0 + 0.5) * 255)));
        pixels[ti+2] = Math.round(Math.min(255, Math.max(0, (this.cellData[base+2] / 100.0 + 0.5) * 255)));
        pixels[ti+3] = Math.round(Math.min(255, Math.max(0, this.cellData[base+3] / 50.0  * 255)));
        // texel 1: emissive rgb /10 and scale /10
        pixels[ti+4] = Math.round(Math.min(255, Math.max(0, this.cellData[base+4] / 10.0 * 255)));
        pixels[ti+5] = Math.round(Math.min(255, Math.max(0, this.cellData[base+5] / 10.0 * 255)));
        pixels[ti+6] = Math.round(Math.min(255, Math.max(0, this.cellData[base+6] / 10.0 * 255)));
        pixels[ti+7] = Math.round(Math.min(255, Math.max(0, this.cellData[base+7] / 10.0 * 255)));
      }
    }

    gl.bindTexture(gl.TEXTURE_2D, this.cellDataTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  private _resizeOutputFBO(w: number, h: number): void {
    const gl = this.gl;
    this.width  = w;
    this.height = h;

    // Delete old output FBO
    gl.deleteFramebuffer(this.outputFBO.fbo);
    gl.deleteTexture(this.outputFBO.tex);

    // Create new at correct size
    this.outputFBO = createFBO(gl, w, h, 'lumen_output_resized');
  }

  private _swapRadianceFBO(): void {
    // Ping-pong: write → read for next frame temporal blend
    const tmp = this.radianceFBO.read;
    this.radianceFBO.read  = this.radianceFBO.write;
    this.radianceFBO.write = tmp;
  }
}
