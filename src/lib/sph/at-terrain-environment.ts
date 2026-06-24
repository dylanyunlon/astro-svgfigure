/**
 * at-terrain-environment.ts — M921
 *
 * AT Terrain Environment — real WebGL1 GPU implementation
 * ─────────────────────────────────────────────────────────────────────────────
 * Renders AT's ground floor plane + environment background using
 * FloorShader.glsl and HomeBGShader.glsl extracted from compiled.vs.
 *
 * Architecture (WebGL1, mirrors fluid-gpu-pass.ts / at-antimatter-particles.ts):
 *   init():    createProgram, compileShader, linkProgram, createFramebuffer,
 *              createTexture, createBuffer, bufferData — all real gl.* calls
 *   render():  useProgram, bindFramebuffer, bindTexture, uniform*,
 *              bindBuffer, vertexAttribPointer, drawArrays/drawElements
 *   dispose(): deleteProgram, deleteFramebuffer, deleteTexture, deleteBuffer
 *
 * Shader sources extracted from:
 *   upstream/activetheory-assets/compiled.vs
 *   - FloorShader.glsl  (line 2889) — floor plane FBR material + mirror reflection
 *   - HomeBGShader.glsl (line 3752) — environment background quad
 *   - fbr.vs / fbr.fs   (line 6564/6440) — FBR vertex + PBR fragment
 *   - range.glsl        (line 2129) — AT crange/range utilities
 *   - simplenoise.glsl  (line 2259) — cnoise / getNoise
 *   - transformUV.glsl  (line 2398) — scaleUV / rotateUV
 *   - matcap.vs         (line 1764) — reflectMatcap
 */




// ─── constants ────────────────────────────────────────────────────────────────

/** Floor mesh: tiled plane subdivisions */



import { getShader } from '../shaders/ShaderLoader';

const FLOOR_SEGS_X = 32 as const;
const FLOOR_SEGS_Z = 32 as const;
/** Half-extent of floor plane in world units */
const FLOOR_HALF  = 20.0 as const;

/** Reflection FBO resolution */
const MIRROR_W = 512 as const;
const MIRROR_H = 512 as const;

/** Lightmap bake resolution (1×1 placeholder for offline-baked data) */
const LM_W = 4 as const;
const LM_H = 4 as const;

// ─── GLSL helpers inlined from compiled.vs ────────────────────────────────────

// range.glsl  (compiled.vs line 2131)
const RANGE_GLSL = /* glsl */`
float range(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    vec3 sub = vec3(oldValue, newMax, oldMax) - vec3(oldMin, newMin, oldMin);
    return sub.x * sub.y / sub.z + newMin;
}
vec2 range(vec2 oldValue, vec2 oldMin, vec2 oldMax, vec2 newMin, vec2 newMax) {
    vec2 oldRange = oldMax - oldMin;
    vec2 newRange = newMax - newMin;
    return (oldValue - oldMin) * newRange / oldRange + newMin;
}
float crange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    return clamp(range(oldValue, oldMin, oldMax, newMin, newMax),
                 min(newMin, newMax), max(newMin, newMax));
}
vec2 crange(vec2 oldValue, vec2 oldMin, vec2 oldMax, vec2 newMin, vec2 newMax) {
    return clamp(range(oldValue, oldMin, oldMax, newMin, newMax),
                 min(newMin, newMax), max(newMin, newMax));
}
`;

// simplenoise.glsl  (compiled.vs line 2259)
const SIMPLENOISE_GLSL = /* glsl */`
float getNoise(vec2 uv, float t) {
    float x = uv.x * uv.y * t * 1000.0;
    x = mod(x, 13.0) * mod(x, 123.0);
    float dx = mod(x, 0.01);
    return clamp(0.1 + dx * 100.0, 0.0, 1.0);
}
float cnoise(vec3 v) {
    float t = v.z * 0.3;
    v.y *= 0.8;
    float n = 0.0;
    float s = 0.5;
    n += (sin(v.x*0.9/s+t*10.0)+sin(v.x*2.4/s+t*15.0)+sin(v.x*-3.5/s+t*4.0)+sin(v.x*-2.5/s+t*7.1))*0.3;
    n += (sin(v.y*-0.3/s+t*18.0)+sin(v.y*1.6/s+t*18.0)+sin(v.y*2.6/s+t*8.0)+sin(v.y*-2.6/s+t*4.5))*0.3;
    return n;
}
`;

// transformUV.glsl  (compiled.vs line 2398)
const TRANSFORM_UV_GLSL = /* glsl */`
vec2 scaleUV(vec2 uv, vec2 scale, vec2 origin) {
    return (uv - origin) / scale + origin;
}
vec2 scaleUV(vec2 uv, vec2 scale) {
    return scaleUV(uv, scale, vec2(0.5));
}
vec2 rotateUV(vec2 uv, float r, vec2 origin) {
    float c = cos(r); float s = sin(r);
    vec2 st = uv - origin;
    st = mat2(c, -s, s, c) * st;
    return st + origin;
}
vec2 rotateUV(vec2 uv, float r) { return rotateUV(uv, r, vec2(0.5)); }
`;

// matcap.vs  (compiled.vs line 1764)
const MATCAP_GLSL = /* glsl */`
vec2 reflectMatcap(vec3 worldPos, vec3 worldNormal) {
    vec3 viewDir = normalize(cameraPosition - worldPos);
    vec3 x = normalize(vec3(viewDir.z, 0.0, -viewDir.x));
    vec3 y = cross(viewDir, x);
    return vec2(dot(x, worldNormal), dot(y, worldNormal)) * 0.495 + 0.5;
}
`;

// fbr.vs  (compiled.vs line 6564) — stripped to inline-usable form
const FBR_VERT_BODY = /* glsl */`
void setupFBR(vec3 p0) {
    vNormal      = normalMatrix * normal;
    vWorldNormal = mat3(modelMatrix[0].xyz, modelMatrix[1].xyz, modelMatrix[2].xyz) * normal;
    vUvFBR       = uv;
    vPos         = p0;
    vec4 mPos    = modelMatrix * vec4(p0, 1.0);
    vMPos        = mPos.xyz / mPos.w;
    vEyePos      = vec3(modelViewMatrix * vec4(p0, 1.0));
}
`;

// fbr.fs  (compiled.vs line 6440) — PBR microfacet + matcap material
const FBR_FRAG_BODY = /* glsl */`
${MATCAP_GLSL}

float pcrange(float v, float a, float b, float c, float d) {
    float r = (b - a); float nr = (d - c);
    return clamp((((v - a) * nr) / r) + c, min(d,c), max(d,c));
}

float geometricOcclusion(float NdL, float NdV, float roughness) {
    float r  = roughness;
    float aL = 2.0*NdL/(NdL+sqrt(r*r+(1.0-r*r)*(NdL*NdL)));
    float aV = 2.0*NdV/(NdV+sqrt(r*r+(1.0-r*r)*(NdV*NdV)));
    return aL * aV;
}
float microfacetDistribution(float roughness, float NdH) {
    float rSq = roughness * roughness;
    float f   = (NdH * rSq - NdH) * NdH + 1.0;
    return rSq / (3.14159265 * f * f);
}

vec3 unpackNormalFBR(vec3 eyePos, vec3 surfNorm, sampler2D nMap,
                     float intensity, float scale, vec2 uv) {
    vec3 q0  = dFdx(eyePos); vec3 q1  = dFdy(eyePos);
    vec2 st0 = dFdx(uv);    vec2 st1 = dFdy(uv);
    vec3 N   = normalize(surfNorm);
    vec3 q1perp = cross(q1, N); vec3 q0perp = cross(N, q0);
    vec3 T = q1perp*st0.x + q0perp*st1.x;
    vec3 B = q1perp*st0.y + q0perp*st1.y;
    float det = max(dot(T,T), dot(B,B));
    float sf  = (det == 0.0) ? 0.0 : inversesqrt(det);
    vec3 mapN = texture2D(nMap, uv*scale).xyz*2.0-1.0;
    mapN.xy  *= intensity;
    return normalize(T*(mapN.x*sf) + B*(mapN.y*sf) + N*mapN.z);
}

vec3 getFBR(vec3 baseColor, vec2 uv, vec3 n) {
    vec3 mro      = texture2D(tMRO, uv).rgb;
    float roughness = mro.g;
    vec2 aUV  = reflectMatcap(vMPos, n);
    vec2 bUV  = ((aUV-0.5)*0.5 - vec2(0.1)) + 0.5;
    vec2 mUV  = mix(aUV, bUV, roughness);
    vec3 V    = normalize(cameraPosition - vMPos);
    vec3 L    = normalize(uLight.xyz);
    vec3 H    = normalize((L+V)/2.0);
    float NdL = pcrange(clamp(dot(n,L),0.001,1.0),0.0,1.0,0.4,1.0);
    float NdV = pcrange(clamp(abs(dot(n,V)),0.001,1.0),0.0,1.0,0.4,1.0);
    float NdH = clamp(dot(n,H),0.0,1.0);
    float G   = geometricOcclusion(NdL,NdV,roughness);
    float D   = microfacetDistribution(roughness,NdH);
    vec3 spec = G*D/(4.0*NdL*NdV) * uColor;
    vec3 col  = NdL*spec*uLight.w;
    return ((baseColor * texture2D(tMatcap, mUV).rgb) + col) * mro.b;
}
vec3 getFBR(vec3 baseColor, vec2 uv) {
    vec3 n = unpackNormalFBR(vEyePos, vWorldNormal, tNormal, uNormalStrength, 1.0, uv);
    return getFBR(baseColor, uv, n);
}
vec3 getFBR(vec3 baseColor) { return getFBR(baseColor, vUvFBR); }
`;

// ─── FloorShader.glsl — vertex shader ─────────────────────────────────────────
// Source: compiled.vs line 2889 FloorShader.glsl #!SHADER: Vertex
// Uses: fbr.vs (setupFBR), + adds vUv2 / vMirrorCoord / vWorldPos varyings

const FLOOR_VERT_SRC = /* glsl */`
precision highp float;

// geometry attributes
attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;
attribute vec2 uv2;

// AT standard matrices (set as uniforms here since we're not Three.js)
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;
uniform mat4 uMirrorMatrix;
uniform vec3 cameraPosition;

// FBR varyings
varying vec3 vNormal;
varying vec3 vWorldNormal;
varying vec3 vPos;
varying vec3 vEyePos;
varying vec2 vUvFBR;
varying vec3 vMPos;
// FloorShader extras
varying vec2 vUv2;
varying vec4 vMirrorCoord;
varying vec3 vWorldPos;

${FBR_VERT_BODY}

void main() {
    vec4 worldPos    = modelMatrix * vec4(position, 1.0);
    vMirrorCoord     = uMirrorMatrix * worldPos;
    vWorldPos        = worldPos.xyz;
    setupFBR(position);
    vUv2             = uv2;
    gl_Position      = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// ─── FloorShader.glsl — fragment shader ───────────────────────────────────────
// Source: compiled.vs line 2889 FloorShader.glsl #!SHADER: Fragment
// Requires: fbr.fs, range.glsl, simplenoise.glsl, transformUV.glsl

const FLOOR_FRAG_SRC = /* glsl */`
precision highp float;

// FBR uniforms
uniform sampler2D tMRO;
uniform sampler2D tMatcap;
uniform sampler2D tNormal;
uniform vec4      uLight;
uniform vec3      uColor;
uniform float     uNormalStrength;
// FloorShader-specific uniforms
uniform sampler2D tLightmap;
uniform sampler2D tMirrorReflection;
uniform sampler2D tLightReflection;
uniform mat4      uMirrorMatrix;
uniform float     uMirrorStrength;
uniform float     uDistortStrength;
uniform vec2      uRUVOffset;
uniform float     uRUVScale;
uniform float     time;
uniform vec3      cameraPosition;

// FBR varyings
varying vec3 vNormal;
varying vec3 vWorldNormal;
varying vec3 vPos;
varying vec3 vEyePos;
varying vec2 vUvFBR;
varying vec3 vMPos;
// FloorShader extras
varying vec2 vUv2;
varying vec4 vMirrorCoord;
varying vec3 vWorldPos;

${RANGE_GLSL}
${SIMPLENOISE_GLSL}
${TRANSFORM_UV_GLSL}
${FBR_FRAG_BODY}

void main() {
    // Base FBR colour + noise modulation
    gl_FragColor     = vec4(getFBR(vec3(1.0)), 1.0);
    gl_FragColor.rgb *= crange(getNoise(vUvFBR, time), 0.0, 1.0, 0.5, 1.0);

    vec3 mro     = texture2D(tMRO, vUvFBR).rgb;
    vec3 normTex = texture2D(tNormal, vUvFBR).rgb;

    // Mirror reflection with normal distortion
    vec2 mirrorUV  = vMirrorCoord.xy / vMirrorCoord.w;
    mirrorUV      += crange(normTex.xy, vec2(0.0), vec2(1.0), vec2(-1.0), vec2(1.0))
                   * uDistortStrength;
    float strength = crange(mro.y, 0.6, 0.7, 0.0, 1.0);

    // Simple box-blur approximation: average 5 taps for mirror blur
    vec3 refBlur   = vec3(0.0);
    refBlur       += texture2D(tMirrorReflection, mirrorUV).rgb;
    refBlur       += texture2D(tMirrorReflection, mirrorUV + vec2( 0.003,  0.0  ) * strength).rgb;
    refBlur       += texture2D(tMirrorReflection, mirrorUV + vec2(-0.003,  0.0  ) * strength).rgb;
    refBlur       += texture2D(tMirrorReflection, mirrorUV + vec2( 0.0,    0.003) * strength).rgb;
    refBlur       += texture2D(tMirrorReflection, mirrorUV + vec2( 0.0,   -0.003) * strength).rgb;
    refBlur       /= 5.0;
    gl_FragColor.rgb += refBlur * uMirrorStrength;

    // Lightmap: ao in .r, lighting in .g
    vec3 lightmap  = texture2D(tLightmap, vUv2).rgb;
    float ao       = lightmap.r;
    float lighting = lightmap.g;
    gl_FragColor.rgb *= ao;
    gl_FragColor.rgb += lighting * 0.15;

    // View-skew parallax for light-reflection texture
    vec3 viewDir       = normalize(vWorldPos - cameraPosition);
    vec3 viewProjection = viewDir - dot(viewDir, vWorldNormal) * vWorldNormal;
    float maxViewSkew  = radians(30.0);
    vec2 viewSkew;
    viewSkew.x         = clamp(viewProjection.x / maxViewSkew, -1.0, 1.0);
    viewSkew.y         = -clamp(viewProjection.y / maxViewSkew, -1.0, 1.0);

    vec2 ruv   = scaleUV(vUvFBR, vec2(0.2 * uRUVScale));
    ruv       += uRUVOffset + viewSkew * 0.2;
    gl_FragColor.rgb += texture2D(tLightReflection, ruv).rgb
                      * 0.5
                      * crange(strength, 0.0, 1.0, 0.5, 1.0);
}
`;

// ─── HomeBGShader.glsl — vertex shader ────────────────────────────────────────
// Source: compiled.vs line 3752 HomeBGShader.glsl #!SHADER: Vertex

const HOMEBG_VERT_SRC = /* glsl */`
precision highp float;

attribute vec3 position;
attribute vec2 uv;

uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;

void main() {
    vUv       = uv;
    vPos      = position;
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// ─── HomeBGShader.glsl — fragment shader ──────────────────────────────────────
// Source: compiled.vs line 3752 HomeBGShader.glsl #!SHADER: Fragment
// Renders environment background: tMap sampled at half-UV, height-based fade

const HOMEBG_FRAG_SRC = /* glsl */`
precision highp float;

uniform sampler2D tMap;
uniform float     uAlpha;

varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;

void main() {
    vec2 uv     = vUv;
    uv         *= 0.5;
    vec4 color  = texture2D(tMap, uv);
    color.rgb  *= smoothstep(30.0, 0.0, abs(vWorldPos.y - 5.0)) * 0.1;
    gl_FragColor       = color;
    gl_FragColor.a    *= uAlpha;
}
`;

// ─── Mirror-render pass — simple colour copy vertex ───────────────────────────
// Used to render the scene into the mirror FBO (planar reflection pre-pass)

const MIRROR_VERT_SRC = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
    vUv         = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

const MIRROR_FRAG_SRC = /* glsl */`
precision highp float;
uniform sampler2D tScene;
varying vec2 vUv;
void main() {
    // Flip Y for reflection
    gl_FragColor = texture2D(tScene, vec2(vUv.x, 1.0 - vUv.y));
}
`;

// ─── Config ───────────────────────────────────────────────────────────────────

export interface ATTerrainEnvironmentConfig {
  /** Floor plane half-extent. Default 20. */
  floorHalfExtent?: number;
  /** Mirror reflection strength (0–1). Default 0.4. */
  mirrorStrength?: number;
  /** Normal distortion strength for mirror. Default 0.03. */
  distortStrength?: number;
  /** Light position in world space. Default [2, 4, 3]. */
  lightPos?: [number, number, number];
  /** Light intensity. Default 1.2. */
  lightIntensity?: number;
  /** FBR material colour tint. Default [1, 1, 1]. */
  materialColor?: [number, number, number];
  /** Normal map strength. Default 1.0. */
  normalStrength?: number;
  /** Background quad UV scale. Default 1. */
  bgUvScale?: number;
  /** Background alpha. Default 1. */
  bgAlpha?: number;
  /** Reflection UV scale multiplier. Default 1. */
  ruvScale?: number;
  /** Reflection UV offset. Default [0,0]. */
  ruvOffset?: [number, number];
}

const DEFAULTS: Required<ATTerrainEnvironmentConfig> = {
  floorHalfExtent: FLOOR_HALF,
  mirrorStrength:  0.4,
  distortStrength: 0.03,
  lightPos:        [2.0, 4.0, 3.0],
  lightIntensity:  1.2,
  materialColor:   [1.0, 1.0, 1.0],
  normalStrength:  1.0,
  bgUvScale:       1.0,
  bgAlpha:         1.0,
  ruvScale:        1.0,
  ruvOffset:       [0.0, 0.0],
};

// ─── Main class ───────────────────────────────────────────────────────────────

export class ATTerrainEnvironment {
  private readonly gl: WebGLRenderingContext;
  private readonly cfg: Required<ATTerrainEnvironmentConfig>;

  // ── Programs ────────────────────────────────────────────────────────────────
  private floorProg!:    WebGLProgram;
  private bgProg!:       WebGLProgram;
  private mirrorProg!:   WebGLProgram;

  // ── Floor geometry ──────────────────────────────────────────────────────────
  private floorPosBuf!:  WebGLBuffer;   // vec3 position
  private floorNorBuf!:  WebGLBuffer;   // vec3 normal
  private floorUvBuf!:   WebGLBuffer;   // vec2 uv
  private floorUv2Buf!:  WebGLBuffer;   // vec2 uv2 (lightmap)
  private floorIdxBuf!:  WebGLBuffer;   // uint16 indices
  private floorIndexCount = 0;

  // ── Background quad ─────────────────────────────────────────────────────────
  private bgQuadBuf!:    WebGLBuffer;   // vec2 position (full-screen quad)
  private bgUvBuf!:      WebGLBuffer;   // vec2 uv

  // ── Mirror FBO ──────────────────────────────────────────────────────────────
  private mirrorFBO!:    WebGLFramebuffer;
  private mirrorTex!:    WebGLTexture;
  private mirrorDepth!:  WebGLRenderbuffer;
  private mirrorQuad!:   WebGLBuffer;

  // ── Textures ────────────────────────────────────────────────────────────────
  private tMRO!:              WebGLTexture;  // Metallic/Roughness/AO
  private tMatcap!:           WebGLTexture;  // FBR matcap lookup
  private tNormal!:           WebGLTexture;  // tangent-space normal map
  private tLightmap!:         WebGLTexture;  // pre-baked lightmap (uv2)
  private tMirrorReflection!: WebGLTexture;  // planar mirror (= mirrorTex)
  private tLightReflection!:  WebGLTexture;  // light-reflection shimmer
  private tBGMap!:            WebGLTexture;  // HomeBGShader env texture

  // ── State ───────────────────────────────────────────────────────────────────
  private time = 0.0;
  /** Read-back: pointer to live mirrorTex so the caller can override it. */
  get mirrorTexture(): WebGLTexture { return this.mirrorTex; }

  // ─── Constructor ─────────────────────────────────────────────────────────────

  constructor(gl: WebGLRenderingContext, cfg: ATTerrainEnvironmentConfig = {}) {
    this.gl  = gl;
    this.cfg = { ...DEFAULTS, ...cfg };
    this._init();
  }

  // ─── Public API ──────────────────────────────────────────────────────────────

  /**
   * Render one frame: updates time, renders BG quad then floor plane.
   * @param dt     Delta seconds.
   * @param view   Column-major 4×4 view matrix (Float32Array[16]).
   * @param proj   Column-major 4×4 projection matrix (Float32Array[16]).
   * @param model  Column-major 4×4 model matrix for the floor (optional; identity if omitted).
   * @param camPos Camera world position [x,y,z].
   * @param w      Viewport width.
   * @param h      Viewport height.
   */
  render(
    dt:     number,
    view:   Float32Array,
    proj:   Float32Array,
    model:  Float32Array,
    camPos: [number, number, number],
    w: number, h: number,
  ): void {
    this.time += dt;
    this._renderMirrorPass(view, proj, model, w, h);
    this._renderBG(view, proj, model, camPos, w, h);
    this._renderFloor(view, proj, model, camPos, w, h);
  }

  /**
   * tick() alias — same as render() but matches the interface expected by
   * the cell-pubsub pipeline.
   */
  tick(
    dt: number,
    view: Float32Array,
    proj: Float32Array,
    model: Float32Array,
    camPos: [number, number, number],
    w: number, h: number,
  ): void {
    this.render(dt, view, proj, model, camPos, w, h);
  }

  /**
   * Upload a custom tMRO texture (replaces the default grey 1×1).
   * Must be called after init(). Accepts an HTMLImageElement.
   */
  setMROTexture(img: HTMLImageElement): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tMRO);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /**
   * Upload a custom matcap texture.
   */
  setMatcapTexture(img: HTMLImageElement): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tMatcap);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /**
   * Upload a custom normal map texture.
   */
  setNormalTexture(img: HTMLImageElement): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tNormal);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /**
   * Upload a custom lightmap texture (sampled at uv2 coordinates).
   */
  setLightmapTexture(img: HTMLImageElement): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tLightmap);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /**
   * Upload environment/background map (used by HomeBGShader).
   */
  setBGMapTexture(img: HTMLImageElement): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tBGMap);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /**
   * Release all GPU resources.
   */
  dispose(): void {
    const gl = this.gl;

    // Programs
    gl.deleteProgram(this.floorProg);
    gl.deleteProgram(this.bgProg);
    gl.deleteProgram(this.mirrorProg);

    // Floor geometry buffers
    gl.deleteBuffer(this.floorPosBuf);
    gl.deleteBuffer(this.floorNorBuf);
    gl.deleteBuffer(this.floorUvBuf);
    gl.deleteBuffer(this.floorUv2Buf);
    gl.deleteBuffer(this.floorIdxBuf);

    // BG quad buffers
    gl.deleteBuffer(this.bgQuadBuf);
    gl.deleteBuffer(this.bgUvBuf);

    // Mirror FBO
    gl.deleteFramebuffer(this.mirrorFBO);
    gl.deleteTexture(this.mirrorTex);
    gl.deleteRenderbuffer(this.mirrorDepth);
    gl.deleteBuffer(this.mirrorQuad);

    // Textures
    gl.deleteTexture(this.tMRO);
    gl.deleteTexture(this.tMatcap);
    gl.deleteTexture(this.tNormal);
    gl.deleteTexture(this.tLightmap);
    gl.deleteTexture(this.tLightReflection);
    gl.deleteTexture(this.tBGMap);
    // mirrorTex already deleted above
  }

  // ─── Private: init ───────────────────────────────────────────────────────────

  private _init(): void {
    // 1. compile programs
    this.floorProg  = this._compile(FLOOR_VERT_SRC,   FLOOR_FRAG_SRC,   'floor');
    this.bgProg     = this._compile(HOMEBG_VERT_SRC,  HOMEBG_FRAG_SRC,  'bg');
    this.mirrorProg = this._compile(MIRROR_VERT_SRC,  MIRROR_FRAG_SRC,  'mirror');

    // 2. build geometry
    this._buildFloorMesh();
    this._buildBGQuad();
    this._buildMirrorQuad();

    // 3. create FBOs + textures
    this._createMirrorFBO();
    this._createTextures();
  }

  // ─── Private: compile ────────────────────────────────────────────────────────

  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATTerrainEnv] vert compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATTerrainEnv] frag compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ATTerrainEnv] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  // ─── Private: geometry ───────────────────────────────────────────────────────

  /**
   * Build a subdivided floor plane (FLOOR_SEGS_X × FLOOR_SEGS_Z quads).
   * Interleaved as separate attribute buffers to match AT shader layout.
   * Also generates uv2 as the same as uv (in lieu of a real baked lightmap UV).
   */
  private _buildFloorMesh(): void {
    const gl    = this.gl;
    const segX  = FLOOR_SEGS_X;
    const segZ  = FLOOR_SEGS_Z;
    const half  = this.cfg.floorHalfExtent;

    const vCount = (segX + 1) * (segZ + 1);
    const pos  = new Float32Array(vCount * 3);
    const nor  = new Float32Array(vCount * 3);
    const uvs  = new Float32Array(vCount * 2);
    const uv2  = new Float32Array(vCount * 2);

    let vi = 0;
    for (let iz = 0; iz <= segZ; iz++) {
      const fz = iz / segZ;
      const z  = (fz - 0.5) * 2.0 * half;
      for (let ix = 0; ix <= segX; ix++) {
        const fx    = ix / segX;
        const x     = (fx - 0.5) * 2.0 * half;
        pos[vi*3+0] = x;
        pos[vi*3+1] = 0.0;
        pos[vi*3+2] = z;
        nor[vi*3+0] = 0.0;
        nor[vi*3+1] = 1.0;
        nor[vi*3+2] = 0.0;
        uvs[vi*2+0] = fx;
        uvs[vi*2+1] = fz;
        uv2[vi*2+0] = fx;
        uv2[vi*2+1] = fz;
        vi++;
      }
    }

    const iCount = segX * segZ * 6;
    const idx    = new Uint16Array(iCount);
    let ii = 0;
    for (let iz = 0; iz < segZ; iz++) {
      for (let ix = 0; ix < segX; ix++) {
        const a = iz * (segX+1) + ix;
        const b = a + 1;
        const c = a + (segX+1);
        const d = c + 1;
        idx[ii++] = a; idx[ii++] = b; idx[ii++] = c;
        idx[ii++] = b; idx[ii++] = d; idx[ii++] = c;
      }
    }
    this.floorIndexCount = iCount;

    // upload position
    this.floorPosBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.floorPosBuf);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);

    // upload normals
    this.floorNorBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.floorNorBuf);
    gl.bufferData(gl.ARRAY_BUFFER, nor, gl.STATIC_DRAW);

    // upload UV
    this.floorUvBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.floorUvBuf);
    gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);

    // upload UV2 (lightmap)
    this.floorUv2Buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.floorUv2Buf);
    gl.bufferData(gl.ARRAY_BUFFER, uv2, gl.STATIC_DRAW);

    // upload indices
    this.floorIdxBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.floorIdxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  }

  /**
   * Build background quad: two triangles covering clip space, with UV.
   */
  private _buildBGQuad(): void {
    const gl = this.gl;

    // World-space background quad (large plane behind scene at y = 0..15)
    // Use a world-space quad so HomeBGShader vWorldPos.y works correctly.
    const h = this.cfg.floorHalfExtent * 1.5;
    const bgPos = new Float32Array([
      -h, -h, -h,
       h, -h, -h,
      -h,  h, -h,
       h, -h, -h,
       h,  h, -h,
      -h,  h, -h,
    ]);
    const bgUv = new Float32Array([
      0,0, 1,0, 0,1,
      1,0, 1,1, 0,1,
    ]);

    this.bgQuadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bgQuadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, bgPos, gl.STATIC_DRAW);

    this.bgUvBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bgUvBuf);
    gl.bufferData(gl.ARRAY_BUFFER, bgUv, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /**
   * Build full-screen clip-space quad for mirror blit pass.
   */
  private _buildMirrorQuad(): void {
    const gl = this.gl;
    this.mirrorQuad = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.mirrorQuad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1,-1,  1,-1,  -1, 1,
       1,-1,  1, 1,  -1, 1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  // ─── Private: FBO creation ───────────────────────────────────────────────────

  /**
   * Create the planar mirror reflection FBO.
   * mirrorTex will be sampled as tMirrorReflection in FloorShader.
   */
  private _createMirrorFBO(): void {
    const gl = this.gl;

    // colour texture
    this.mirrorTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.mirrorTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, MIRROR_W, MIRROR_H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // depth renderbuffer
    this.mirrorDepth = gl.createRenderbuffer()!;
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.mirrorDepth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, MIRROR_W, MIRROR_H);

    // framebuffer
    this.mirrorFBO = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.mirrorFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.mirrorTex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.mirrorDepth);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
  }

  // ─── Private: texture creation ───────────────────────────────────────────────

  /**
   * Create all material/environment textures with sensible default data.
   * The caller can replace them with setXxxTexture() after construction.
   */
  private _createTextures(): void {
    const gl = this.gl;

    // tMRO: 1×1 (metallic=0, roughness=0.5, ao=1)
    this.tMRO = this._createTexture1x1(gl, new Uint8Array([0, 128, 255, 255]));

    // tMatcap: 4×4 soft-lit matcap approximation
    this.tMatcap = this._createMatcapDefault(gl);

    // tNormal: flat normal (128,128,255)
    this.tNormal = this._createTexture1x1(gl, new Uint8Array([128, 128, 255, 255]));

    // tLightmap: warm grey (ao=200, lighting=60)
    this.tLightmap = this._createLightmapDefault(gl);

    // tLightReflection: subtle warm shimmer (small gradient texture)
    this.tLightReflection = this._createLightReflectionDefault(gl);

    // tBGMap: deep blue-grey sky gradient
    this.tBGMap = this._createBGMapDefault(gl);

    // tMirrorReflection is the mirrorTex (already created in _createMirrorFBO)
    this.tMirrorReflection = this.mirrorTex;
  }

  /**
   * 1×1 RGBA texture from a 4-byte array.
   */
  private _createTexture1x1(gl: WebGLRenderingContext, data: Uint8Array): WebGLTexture {
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  /**
   * 4×4 soft matcap: bright centre shading off to edges.
   */
  private _createMatcapDefault(gl: WebGLRenderingContext): WebGLTexture {
    const size = 8;
    const data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const nx = (x / (size-1)) * 2.0 - 1.0;
        const ny = (y / (size-1)) * 2.0 - 1.0;
        const d  = Math.sqrt(nx*nx + ny*ny);
        const v  = Math.max(0, 1.0 - d);
        const b  = Math.floor(v * 200 + 55);
        const i  = (y * size + x) * 4;
        data[i]   = b;   // R
        data[i+1] = b;   // G
        data[i+2] = Math.min(255, b + 20);  // B: slightly cooler
        data[i+3] = 255;
      }
    }
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, size, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  /**
   * LM_W×LM_H lightmap: ao=0.85 warm, lighting=0.15.
   */
  private _createLightmapDefault(gl: WebGLRenderingContext): WebGLTexture {
    const data = new Uint8Array(LM_W * LM_H * 4);
    for (let i = 0; i < LM_W * LM_H; i++) {
      data[i*4+0] = 216;  // ao
      data[i*4+1] = 38;   // lighting (indirect)
      data[i*4+2] = 0;
      data[i*4+3] = 255;
    }
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, LM_W, LM_H, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  /**
   * 8×8 light-reflection shimmer: soft white radial gradient.
   */
  private _createLightReflectionDefault(gl: WebGLRenderingContext): WebGLTexture {
    const size = 8;
    const data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const nx = (x / (size-1)) * 2.0 - 1.0;
        const ny = (y / (size-1)) * 2.0 - 1.0;
        const v  = Math.max(0, 1.0 - Math.sqrt(nx*nx + ny*ny));
        const b  = Math.floor(v * 80);
        const i  = (y * size + x) * 4;
        data[i]   = Math.min(255, b + 20);
        data[i+1] = Math.min(255, b + 18);
        data[i+2] = b;
        data[i+3] = 255;
      }
    }
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, size, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  /**
   * 8×8 deep blue-grey sky gradient for HomeBGShader.
   */
  private _createBGMapDefault(gl: WebGLRenderingContext): WebGLTexture {
    const size = 8;
    const data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) {
      const t = y / (size - 1);
      // horizon: warm grey → zenith: cool dark blue
      const r = Math.floor((0.22 - t * 0.12) * 255);
      const g = Math.floor((0.25 - t * 0.10) * 255);
      const b = Math.floor((0.35 + t * 0.15) * 255);
      for (let x = 0; x < size; x++) {
        const i    = (y * size + x) * 4;
        data[i]   = Math.max(0, r);
        data[i+1] = Math.max(0, g);
        data[i+2] = Math.min(255, b);
        data[i+3] = 255;
      }
    }
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, size, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  // ─── Private: render passes ──────────────────────────────────────────────────

  /**
   * Mirror pre-pass: render a Y-flipped copy of the current mirrorTex into
   * the mirror FBO to provide planar reflection for this frame.
   * In a full pipeline the caller would render the scene here first.
   */
  private _renderMirrorPass(
    view: Float32Array, proj: Float32Array, model: Float32Array,
    w: number, h: number,
  ): void {
    const gl = this.gl;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.mirrorFBO);
    gl.viewport(0, 0, MIRROR_W, MIRROR_H);
    gl.clearColor(0.05, 0.05, 0.08, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(this.mirrorProg);

    // Bind mirrorTex as tScene (this is a self-blit that seed-initialises it
    // the first frame; the caller can override by binding an external scene tex)
    const sceneULoc = gl.getUniformLocation(this.mirrorProg, 'tScene');
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tBGMap);  // use BG map as initial fill
    gl.uniform1i(sceneULoc, 0);

    const posLoc = gl.getAttribLocation(this.mirrorProg, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.mirrorQuad);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.disableVertexAttribArray(posLoc);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
  }

  /**
   * Render HomeBGShader environment background quad.
   */
  private _renderBG(
    view: Float32Array, proj: Float32Array, model: Float32Array,
    camPos: [number, number, number],
    w: number, h: number,
  ): void {
    const gl = this.gl;

    gl.useProgram(this.bgProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);

    // matrices
    gl.uniformMatrix4fv(gl.getUniformLocation(this.bgProg, 'projectionMatrix'),    false, proj);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.bgProg, 'modelViewMatrix'),     false, this._multiplyMat4(view, model));
    gl.uniformMatrix4fv(gl.getUniformLocation(this.bgProg, 'modelMatrix'),         false, model);

    // uniforms
    gl.uniform1f(gl.getUniformLocation(this.bgProg, 'uAlpha'), this.cfg.bgAlpha);

    // tMap
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tBGMap);
    gl.uniform1i(gl.getUniformLocation(this.bgProg, 'tMap'), 0);

    // attributes: position
    const posLoc = gl.getAttribLocation(this.bgProg, 'position');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bgQuadBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    // attributes: uv
    const uvLoc = gl.getAttribLocation(this.bgProg, 'uv');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bgUvBuf);
    gl.enableVertexAttribArray(uvLoc);
    gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 0, 0);

    gl.disable(gl.DEPTH_TEST);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.enable(gl.DEPTH_TEST);

    gl.disableVertexAttribArray(posLoc);
    gl.disableVertexAttribArray(uvLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /**
   * Render the floor plane using FloorShader (FBR + mirror reflection + lightmap).
   */
  private _renderFloor(
    view: Float32Array, proj: Float32Array, model: Float32Array,
    camPos: [number, number, number],
    w: number, h: number,
  ): void {
    const gl  = this.gl;
    const cfg = this.cfg;

    gl.useProgram(this.floorProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);

    // ── matrices ──────────────────────────────────────────────────────────────
    const mv       = this._multiplyMat4(view, model);
    const normalM  = this._normalMatrix(mv);
    const mirrorM  = this._buildMirrorMatrix(model);

    gl.uniformMatrix4fv(gl.getUniformLocation(this.floorProg, 'projectionMatrix'),  false, proj);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.floorProg, 'modelViewMatrix'),   false, mv);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.floorProg, 'modelMatrix'),       false, model);
    gl.uniformMatrix3fv(gl.getUniformLocation(this.floorProg, 'normalMatrix'),      false, normalM);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.floorProg, 'uMirrorMatrix'),     false, mirrorM);

    // ── light + material uniforms ─────────────────────────────────────────────
    gl.uniform4f(gl.getUniformLocation(this.floorProg, 'uLight'),
      cfg.lightPos[0], cfg.lightPos[1], cfg.lightPos[2], cfg.lightIntensity);
    gl.uniform3f(gl.getUniformLocation(this.floorProg, 'uColor'),
      cfg.materialColor[0], cfg.materialColor[1], cfg.materialColor[2]);
    gl.uniform1f(gl.getUniformLocation(this.floorProg, 'uNormalStrength'), cfg.normalStrength);
    gl.uniform3f(gl.getUniformLocation(this.floorProg, 'cameraPosition'),  camPos[0], camPos[1], camPos[2]);
    gl.uniform1f(gl.getUniformLocation(this.floorProg, 'time'),            this.time);

    // FloorShader-specific uniforms
    gl.uniform1f(gl.getUniformLocation(this.floorProg, 'uMirrorStrength'),   cfg.mirrorStrength);
    gl.uniform1f(gl.getUniformLocation(this.floorProg, 'uDistortStrength'),  cfg.distortStrength);
    gl.uniform1f(gl.getUniformLocation(this.floorProg, 'uRUVScale'),         cfg.ruvScale);
    gl.uniform2f(gl.getUniformLocation(this.floorProg, 'uRUVOffset'),
      cfg.ruvOffset[0], cfg.ruvOffset[1]);

    // ── bind textures ─────────────────────────────────────────────────────────
    // unit 0: tMRO
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tMRO);
    gl.uniform1i(gl.getUniformLocation(this.floorProg, 'tMRO'), 0);

    // unit 1: tMatcap
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.tMatcap);
    gl.uniform1i(gl.getUniformLocation(this.floorProg, 'tMatcap'), 1);

    // unit 2: tNormal
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.tNormal);
    gl.uniform1i(gl.getUniformLocation(this.floorProg, 'tNormal'), 2);

    // unit 3: tLightmap
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.tLightmap);
    gl.uniform1i(gl.getUniformLocation(this.floorProg, 'tLightmap'), 3);

    // unit 4: tMirrorReflection
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.tMirrorReflection);
    gl.uniform1i(gl.getUniformLocation(this.floorProg, 'tMirrorReflection'), 4);

    // unit 5: tLightReflection
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, this.tLightReflection);
    gl.uniform1i(gl.getUniformLocation(this.floorProg, 'tLightReflection'), 5);

    // ── vertex attributes ────────────────────────────────────────────────────
    const posLoc  = gl.getAttribLocation(this.floorProg, 'position');
    const norLoc  = gl.getAttribLocation(this.floorProg, 'normal');
    const uvLoc   = gl.getAttribLocation(this.floorProg, 'uv');
    const uv2Loc  = gl.getAttribLocation(this.floorProg, 'uv2');

    gl.bindBuffer(gl.ARRAY_BUFFER, this.floorPosBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc,  3, gl.FLOAT, false, 0, 0);

    if (norLoc >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.floorNorBuf);
      gl.enableVertexAttribArray(norLoc);
      gl.vertexAttribPointer(norLoc, 3, gl.FLOAT, false, 0, 0);
    }

    if (uvLoc >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.floorUvBuf);
      gl.enableVertexAttribArray(uvLoc);
      gl.vertexAttribPointer(uvLoc,  2, gl.FLOAT, false, 0, 0);
    }

    if (uv2Loc >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.floorUv2Buf);
      gl.enableVertexAttribArray(uv2Loc);
      gl.vertexAttribPointer(uv2Loc, 2, gl.FLOAT, false, 0, 0);
    }

    // ── draw ─────────────────────────────────────────────────────────────────
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.floorIdxBuf);
    gl.enable(gl.DEPTH_TEST);
    gl.drawElements(gl.TRIANGLES, this.floorIndexCount, gl.UNSIGNED_SHORT, 0);

    // ── cleanup ───────────────────────────────────────────────────────────────
    if (posLoc >= 0) gl.disableVertexAttribArray(posLoc);
    if (norLoc >= 0) gl.disableVertexAttribArray(norLoc);
    if (uvLoc  >= 0) gl.disableVertexAttribArray(uvLoc);
    if (uv2Loc >= 0) gl.disableVertexAttribArray(uv2Loc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  }

  // ─── Private: math helpers ───────────────────────────────────────────────────

  /**
   * Multiply two column-major 4×4 matrices: returns a × b.
   */
  private _multiplyMat4(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 4; col++) {
        let sum = 0;
        for (let k = 0; k < 4; k++) {
          sum += a[k*4 + row] * b[col*4 + k];
        }
        out[col*4 + row] = sum;
      }
    }
    return out;
  }

  /**
   * Extract upper-left 3×3 normal matrix (inverse-transpose) from a 4×4 MV.
   * Returns column-major Float32Array[9].
   */
  private _normalMatrix(mv: Float32Array): Float32Array {
    // For a pure rotation+translation (no non-uniform scale), normalMat = upper 3×3 of mv.
    const n = new Float32Array(9);
    n[0] = mv[0]; n[1] = mv[1]; n[2] = mv[2];
    n[3] = mv[4]; n[4] = mv[5]; n[5] = mv[6];
    n[6] = mv[8]; n[7] = mv[9]; n[8] = mv[10];
    return n;
  }

  /**
   * Build the AT mirror projection matrix from model matrix.
   * This creates a reflection matrix for the ground plane (Y = 0).
   * Returns a column-major Float32Array[16].
   */
  private _buildMirrorMatrix(model: Float32Array): Float32Array {
    // Standard Y=0 plane reflection matrix:
    // [ 1  0  0  0 ]
    // [ 0 -1  0  0 ]
    // [ 0  0  1  0 ]
    // [ 0  0  0  1 ]
    // Then multiply with a simple bias matrix to project into [0,1] NDC for UV lookup.
    const reflect = new Float32Array([
       1, 0, 0, 0,
       0,-1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1,
    ]);
    // bias: NDC → [0,1]
    const bias = new Float32Array([
      0.5, 0, 0, 0,
      0, 0.5, 0, 0,
      0, 0, 0.5, 0,
      0.5, 0.5, 0.5, 1,
    ]);
    return this._multiplyMat4(bias, reflect);
  }
}

// ─── Convenience factory ─────────────────────────────────────────────────────

/**
 * Create an ATTerrainEnvironment with default settings.
 *
 * ```ts
 * const terrain = createATTerrainEnvironment(gl, {
 *   mirrorStrength: 0.5,
 *   lightPos: [3, 6, 2],
 * });
 * // each frame:
 * terrain.render(dt, viewMat, projMat, modelMat, camPos, canvas.width, canvas.height);
 * ```
 */
export function createATTerrainEnvironment(
  gl:  WebGLRenderingContext,
  cfg: ATTerrainEnvironmentConfig = {},
): ATTerrainEnvironment {
  return new ATTerrainEnvironment(gl, cfg);
}
