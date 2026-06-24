/**
 * at-glass-material.ts — M1067: AT Glass PBR Material
 *
 * Implements AT's GlassCubeShader / WorkGlassShader / GlassShaderPBR
 * as a reusable GlassMaterial class.
 *
 * Feature parity with upstream activetheory-assets/compiled.vs:
 *   - Fresnel reflection  (uFresnelPow, uFresnelColor, uFresnelAlpha)
 *   - Screen-space refraction distortion (uRefractionRatio, uDistortStrength)
 *   - Environment map blend — equirectangular cubemap (uEnvBlend)
 *   - Blinn-Phong specular (uShininess, uSpecAdd, uPhongColor, uLightDir)
 *
 * Default parameters sourced from uil-params.json:
 *   GlassCubeShader/GlassCubeShader/Element_0_home_scene/*
 *   WorkGlassShader/WorkGlassShader/*
 *   GlassShaderPBR/GlassShaderPBR/Element_1_glass_test/*
 *
 * Research: M1067 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** RGB tuple [r, g, b] in 0..1 */
type RGB = [number, number, number];

/**
 * All tuneable parameters for GlassMaterial.
 * Names mirror AT's original uniform names exactly so UIL params
 * from uil-params.json can be piped in directly.
 */
export interface GlassMaterialParams {
  // ── Fresnel ────────────────────────────────────────────────────────────────
  /** Fresnel exponent — higher = tighter rim (GlassCube default 1.5, Work default 1) */
  uFresnelPow:   number;
  /** Fresnel rim tint colour RGB — added on top of refraction at grazing angles */
  uFresnelColor: RGB;
  /** Fresnel alpha — how strongly the fresnel rim is applied (0..1) */
  uFresnelAlpha: number;

  // ── Refraction ─────────────────────────────────────────────────────────────
  /** IOR-like ratio fed into GLSL refract() — 0 = no bend, ~0.34 cleanroom glass */
  uRefractionRatio: number;
  /**
   * Screen-space distortion strength.
   * Multiplied by vNormal.xy to offset the tRefraction UV.
   * Negative values flip the distortion (CleanRoom uses −1).
   */
  uDistortStrength: number;

  // ── Environment map ────────────────────────────────────────────────────────
  /**
   * Blend weight for equirectangular environment colour.
   * [0] = direct multiplier, [1] = gamma/power (pass vec2 if needed).
   * Simple float version: 0 = off, 1 = full env.
   */
  uEnvBlend: number;
  /**
   * Attenuation / darkening of the env sample before blending.
   * (WorkGlass default 0.19, GlassCube default 0.5)
   */
  uAttenuation: number;
  /**
   * Reflection scale — multiplier for the mirror / reflection FBO contribution.
   * (WorkGlass default 0, GlassCube default 1)
   */
  uReflectScale: number;

  // ── Blinn-Phong specular ───────────────────────────────────────────────────
  /**
   * Shininess / specular exponent.
   * AT stores this in uSpecAdd[0] (intensity) and uSpecAdd[1] (bias/offset).
   * uShininess drives the pow() call; higher = sharper highlight.
   */
  uShininess:   number;
  /** vec2: [specular intensity multiplier, specular bias] */
  uSpecAdd:     [number, number];
  /** Specular / Phong highlight tint colour */
  uPhongColor:  RGB;
  /** Directional light vector for specular (world space, not normalised here) */
  uLightDir:    [number, number, number];

  // ── Composite / misc ───────────────────────────────────────────────────────
  /** Output alpha (0=fully transparent, 1=opaque) */
  uAlpha: number;
  /** Transparent flag — when true uses premultiplied alpha blending */
  uTransparent: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Default parameter sets (sourced verbatim from uil-params.json)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * GlassCubeShader/GlassCubeShader/Element_0_home_scene
 * The "hero" glass cube on the AT home scene.
 */
export const PARAMS_GLASS_CUBE: GlassMaterialParams = {
  uFresnelPow:      1.5,
  uFresnelColor:    [0.706, 0.878, 0.890],   // #b4e0e3
  uFresnelAlpha:    0,                        // not separately set in GlassCube
  uRefractionRatio: 1.0,
  uDistortStrength: 8.06,
  uEnvBlend:        1.0,
  uAttenuation:     0.5,
  uReflectScale:    1.0,
  uShininess:       80,
  uSpecAdd:         [4.48, 0.0],
  uPhongColor:      [1.0, 1.0, 1.0],         // #ffffff
  uLightDir:        [-15.7, 0.28, 4.5],
  uAlpha:           1.0,
  uTransparent:     true,
};

/**
 * WorkGlassShader/WorkGlassShader — glass panels on the Work page.
 */
export const PARAMS_WORK_GLASS: GlassMaterialParams = {
  uFresnelPow:      1.0,
  uFresnelColor:    [0.914, 0.976, 1.0],     // #e9f9ff
  uFresnelAlpha:    1.0,
  uRefractionRatio: 0.0,
  uDistortStrength: 0.0,                     // not distorting on work glass
  uEnvBlend:        0.0,
  uAttenuation:     0.19,
  uReflectScale:    0.0,
  uShininess:       128,
  uSpecAdd:         [1.53, 0.0],
  uPhongColor:      [1.0, 1.0, 1.0],         // #ffffff
  uLightDir:        [0.0, -1.12, 4.69],
  uAlpha:           1.0,
  uTransparent:     true,
};

/**
 * WorkGlassCubeShader/WorkGlassCubeShader — cube glass in the Work detail.
 */
export const PARAMS_WORK_GLASS_CUBE: GlassMaterialParams = {
  uFresnelPow:      1.0,
  uFresnelColor:    [0.0, 0.0, 0.0],         // #000000
  uFresnelAlpha:    1.0,
  uRefractionRatio: 1.0,
  uDistortStrength: 4.96,
  uEnvBlend:        1.0,
  uAttenuation:     0.01,
  uReflectScale:    0.0,
  uShininess:       64,
  uSpecAdd:         [5.56, 0.98],
  uPhongColor:      [0.0, 0.0, 0.0],         // #000000
  uLightDir:        [0.0, 0.0, 1.0],
  uAlpha:           1.0,
  uTransparent:     true,
};

/**
 * GlassShaderPBR/GlassShaderPBR/Element_1_glass_test
 * The isolated glass PBR test element.
 */
export const PARAMS_GLASS_PBR: GlassMaterialParams = {
  uFresnelPow:      0.3,
  uFresnelColor:    [1.0, 0.0, 0.0],         // #ff0000 (test value)
  uFresnelAlpha:    0.0,
  uRefractionRatio: 0.34,                    // ~CleanRoom ratio
  uDistortStrength: 2.81,
  uEnvBlend:        1.0,                     // uEnv[0]=10 → normalised to 1 here
  uAttenuation:     0.0,
  uReflectScale:    0.0,
  uShininess:       128,
  uSpecAdd:         [1.0, 0.0],
  uPhongColor:      [1.0, 1.0, 1.0],
  uLightDir:        [0.0, 1.0, 1.0],
  uAlpha:           1.0,
  uTransparent:     false,
};

/** Cleanroom glass — tuned from CleanRoomGlass.glsl params */
export const PARAMS_CLEANROOM_GLASS: GlassMaterialParams = {
  uFresnelPow:      -0.03,
  uFresnelColor:    [0.7, 0.85, 1.0],
  uFresnelAlpha:    0.8,
  uRefractionRatio: 0.34,
  uDistortStrength: -1.0,
  uEnvBlend:        0.4,
  uAttenuation:     0.2,
  uReflectScale:    0.6,
  uShininess:       128,
  uSpecAdd:         [1.0, 0.0],
  uPhongColor:      [1.0, 1.0, 1.0],
  uLightDir:        [0.0, 1.0, 2.0],
  uAlpha:           1.0,
  uTransparent:     true,
};

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — Vertex Shader
// Ports refl.vs: computes world-space reflection + refraction vectors
// ─────────────────────────────────────────────────────────────────────────────

const GLASS_MATERIAL_VERT = /* glsl */`
precision highp float;

attribute vec3 aPosition;
attribute vec3 aNormal;
attribute vec2 aUv;

uniform mat4  uProjection;
uniform mat4  uModelView;
uniform mat4  uModel;
uniform mat3  uNormalMatrix;
uniform vec3  uCameraPos;
uniform float uRefractionRatio;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vReflection;
varying vec3 vRefraction;
varying vec3 vPos;
varying vec2 vUv;

// ── refl.vs helpers ──────────────────────────────────────────────────────────
// inverseTransformDirection: transforms a direction back from view → world
vec3 inverseTransformDir(vec3 n, mat4 m) {
    return normalize((vec4(n, 0.0) * m).xyz);
}

vec3 computeReflection(vec4 worldPos) {
    vec3 N = inverseTransformDir(uNormalMatrix * aNormal, uModel);
    vec3 I = normalize(worldPos.xyz - uCameraPos);
    return reflect(I, N);
}

vec3 computeRefraction(vec4 worldPos, float ratio) {
    vec3 N = inverseTransformDir(uNormalMatrix * aNormal, uModel);
    vec3 I = normalize(worldPos.xyz - uCameraPos);
    return refract(I, N, ratio);
}

void main() {
    vec4 worldPos = uModel * vec4(aPosition, 1.0);

    vReflection = computeReflection(worldPos);
    vRefraction = computeRefraction(worldPos, uRefractionRatio);

    vPos      = aPosition;
    vWorldPos = worldPos.xyz;
    vNormal   = uNormalMatrix * aNormal;
    vViewDir  = -vec3(uModelView * vec4(aPosition, 1.0));
    vUv       = aUv;

    gl_Position = uProjection * uModelView * vec4(aPosition, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — Fragment Shader
// Implements:
//   1. Fresnel rim  (getFresnel → Schlick approximation from fresnel.glsl)
//   2. Screen-space refraction distortion  (tRefraction + normal offset)
//   3. Equirectangular environment blend  (envColorEqui from refl.fs)
//   4. Blinn-Phong specular  (uPhongColor, uShininess, uSpecAdd, uLightDir)
// ─────────────────────────────────────────────────────────────────────────────

const GLASS_MATERIAL_FRAG = /* glsl */`
precision highp float;

// ── Varyings (from vertex) ───────────────────────────────────────────────────
varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vReflection;
varying vec3 vRefraction;
varying vec3 vPos;
varying vec2 vUv;

// ── Textures ─────────────────────────────────────────────────────────────────
uniform sampler2D tRefraction;   // screen-space scene behind glass
uniform sampler2D tEnv;          // equirectangular HDRI environment

// ── Resolution (for screen-space UV) ─────────────────────────────────────────
uniform vec2 uResolution;

// ── Fresnel (fresnel.glsl) ───────────────────────────────────────────────────
uniform float uFresnelPow;       // exponent: higher = tighter rim
uniform vec3  uFresnelColor;     // rim tint colour
uniform float uFresnelAlpha;     // rim mix weight (0=ignore, 1=full)

// ── Refraction (refl.vs / screen-space) ──────────────────────────────────────
uniform float uRefractionRatio;  // passed to vertex for refract() vector
uniform float uDistortStrength;  // screen-UV normal-offset multiplier

// ── Environment map ───────────────────────────────────────────────────────────
uniform float uEnvBlend;         // env contribution weight (0..1)
uniform float uAttenuation;      // darkens the env sample
uniform float uReflectScale;     // reflection FBO weight

// ── Blinn-Phong specular (AT: uSpecAdd, uShininess, uPhongColor, uLightDir) ──
uniform float uShininess;        // specular exponent
uniform vec2  uSpecAdd;          // [intensity, bias]
uniform vec3  uPhongColor;       // specular colour tint
uniform vec3  uLightDir;         // directional light (world space)

// ── Composite ─────────────────────────────────────────────────────────────────
uniform float uAlpha;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers — inlined from compiled.vs modules
// ─────────────────────────────────────────────────────────────────────────────

// fresnel.glsl — getFresnel (Schlick approximation)
// AT signature: getFresnel(vec3 normal, vec3 viewDir, float power)
float getFresnel(vec3 N, vec3 V, float power) {
    float cosTheta = clamp(dot(normalize(N), normalize(V)), 0.0, 1.0);
    return pow(1.0 - cosTheta, max(power, 0.001));
}

// refl.fs — envColorEqui: sample an equirectangular env map
// Converts a world-space direction to latitude/longitude UV
vec4 envColorEqui(sampler2D envMap, vec3 dir) {
    vec3 d = normalize(dir);
    float phi   = atan(d.z, d.x);          // azimuth  [-π, π]
    float theta = asin(clamp(d.y, -1.0, 1.0)); // altitude [-π/2, π/2]
    const float PI  = 3.14159265358979;
    const float PI2 = 6.28318530717959;
    vec2 uv;
    uv.x = phi   / PI2 + 0.5;
    uv.y = theta / PI  + 0.5;
    return texture2D(envMap, uv);
}

// Blinn-Phong specular component
// AT: colour.rgb += phongSpec * uPhongColor * uSpecAdd.x
float blinnPhong(vec3 N, vec3 L, vec3 V, float shininess) {
    vec3 H = normalize(L + V);
    return pow(max(dot(N, H), 0.0), shininess);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
void main() {
    vec3  N  = normalize(vNormal);
    vec3  V  = normalize(vViewDir);

    // ── 1. Fresnel coefficient ─────────────────────────────────────────────
    float f = getFresnel(N, V, uFresnelPow);

    // ── 2. Screen-space refraction UV ─────────────────────────────────────
    // AT: ruv = gl_FragCoord.xy / resolution;  ruv += N.xy * uDistortStrength
    vec2 screenUV = gl_FragCoord.xy / uResolution;
    vec2 refrUV   = screenUV + N.xy * uDistortStrength * 0.1;
    refrUV        = clamp(refrUV, 0.001, 0.999);

    // ── 3. Sample refraction texture (scene behind glass) ─────────────────
    vec3 refrColor = texture2D(tRefraction, refrUV).rgb;

    // ── 4. Sample equirectangular environment map (reflection) ─────────────
    // AT: envColorEquiRGB(tEnv, vRefraction, strength, gamma)
    vec3 envRefl    = envColorEqui(tEnv, vReflection).rgb;
    vec3 envRefr    = envColorEqui(tEnv, vRefraction).rgb;

    // Attenuate env samples (AT: uAttenuation darkens the env reflection)
    envRefl *= (1.0 - uAttenuation);
    envRefr *= (1.0 - uAttenuation);

    // ── 5. Compose base glass colour ──────────────────────────────────────
    // AT pattern (WorkItemShader / CleanRoomGlass):
    //   color += envColorEquiRGB(tEnv, vRefraction) * envBlend * 0.08
    //   color.rgb += refractionTex * distortWeight
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);

    // Refraction layer — dominant contribution (scene behind glass)
    color.rgb += refrColor;

    // Environment refraction tint
    color.rgb += envRefr * uEnvBlend * 0.08;

    // Environment reflection layer
    color.rgb += envRefl * uEnvBlend * uReflectScale * f;

    // ── 6. Fresnel rim (uFresnelColor, uFresnelAlpha) ─────────────────────
    // AT: color += mix(vec3(0), uFresnelColor, f * uFresnelAlpha)
    color.rgb += uFresnelColor * f * uFresnelAlpha;

    // ── 7. Blinn-Phong specular ───────────────────────────────────────────
    // AT: phong(N, lightDir, viewDir, shininess) * uSpecAdd.x + uSpecAdd.y
    vec3  L      = normalize(uLightDir);
    float spec   = blinnPhong(N, L, V, max(uShininess, 1.0));
    float specI  = spec * uSpecAdd.x + uSpecAdd.y;
    color.rgb   += uPhongColor * max(specI, 0.0);

    // ── 8. Output alpha ───────────────────────────────────────────────────
    gl_FragColor = vec4(color.rgb, uAlpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GlassMaterial — reusable WebGL1 material class
// ─────────────────────────────────────────────────────────────────────────────

/** Uniform location cache to avoid repeated getUniformLocation() calls */
interface UniformCache {
  uProjection:      WebGLUniformLocation | null;
  uModelView:       WebGLUniformLocation | null;
  uModel:           WebGLUniformLocation | null;
  uNormalMatrix:    WebGLUniformLocation | null;
  uCameraPos:       WebGLUniformLocation | null;
  uResolution:      WebGLUniformLocation | null;
  // textures
  tRefraction:      WebGLUniformLocation | null;
  tEnv:             WebGLUniformLocation | null;
  // fresnel
  uFresnelPow:      WebGLUniformLocation | null;
  uFresnelColor:    WebGLUniformLocation | null;
  uFresnelAlpha:    WebGLUniformLocation | null;
  // refraction
  uRefractionRatio: WebGLUniformLocation | null;
  uDistortStrength: WebGLUniformLocation | null;
  // env
  uEnvBlend:        WebGLUniformLocation | null;
  uAttenuation:     WebGLUniformLocation | null;
  uReflectScale:    WebGLUniformLocation | null;
  // specular
  uShininess:       WebGLUniformLocation | null;
  uSpecAdd:         WebGLUniformLocation | null;
  uPhongColor:      WebGLUniformLocation | null;
  uLightDir:        WebGLUniformLocation | null;
  // composite
  uAlpha:           WebGLUniformLocation | null;
}

/**
 * GlassMaterial — AT PBR glass material, WebGL1.
 *
 * Usage:
 * ```ts
 * const mat = new GlassMaterial(gl, PARAMS_GLASS_CUBE);
 * // set your textures once:
 * mat.setRefractionTexture(sceneFBO);
 * mat.setEnvTexture(hdriTex);
 *
 * // each frame:
 * mat.bind(projection, modelView, model, normalMatrix, cameraPos, [w, h]);
 * gl.drawElements(...);
 * mat.unbind();
 *
 * // live tweak:
 * mat.setParam('uFresnelPow', 2.0);
 * ```
 */
export class GlassMaterial {
  private readonly gl:      WebGLRenderingContext;
  private readonly program: WebGLProgram;
  private readonly uloc:    UniformCache;
  private readonly params:  GlassMaterialParams;

  /** Slot 0 — screen-space refraction FBO texture */
  private refrTex: WebGLTexture | null = null;
  /** Slot 1 — equirectangular HDRI environment texture */
  private envTex:  WebGLTexture | null = null;

  // ── attribute locations ────────────────────────────────────────────────────
  private readonly aPosition: number;
  private readonly aNormal:   number;
  private readonly aUv:       number;

  constructor(
    gl: WebGLRenderingContext,
    params: Partial<GlassMaterialParams> = {},
  ) {
    this.gl     = gl;
    this.params = { ...PARAMS_GLASS_CUBE, ...params };
    this.program = this._compile(GLASS_MATERIAL_VERT, GLASS_MATERIAL_FRAG, 'GlassMaterial');

    // Cache all uniform locations once
    const p = this.program;
    this.uloc = {
      uProjection:      gl.getUniformLocation(p, 'uProjection'),
      uModelView:       gl.getUniformLocation(p, 'uModelView'),
      uModel:           gl.getUniformLocation(p, 'uModel'),
      uNormalMatrix:    gl.getUniformLocation(p, 'uNormalMatrix'),
      uCameraPos:       gl.getUniformLocation(p, 'uCameraPos'),
      uResolution:      gl.getUniformLocation(p, 'uResolution'),
      tRefraction:      gl.getUniformLocation(p, 'tRefraction'),
      tEnv:             gl.getUniformLocation(p, 'tEnv'),
      uFresnelPow:      gl.getUniformLocation(p, 'uFresnelPow'),
      uFresnelColor:    gl.getUniformLocation(p, 'uFresnelColor'),
      uFresnelAlpha:    gl.getUniformLocation(p, 'uFresnelAlpha'),
      uRefractionRatio: gl.getUniformLocation(p, 'uRefractionRatio'),
      uDistortStrength: gl.getUniformLocation(p, 'uDistortStrength'),
      uEnvBlend:        gl.getUniformLocation(p, 'uEnvBlend'),
      uAttenuation:     gl.getUniformLocation(p, 'uAttenuation'),
      uReflectScale:    gl.getUniformLocation(p, 'uReflectScale'),
      uShininess:       gl.getUniformLocation(p, 'uShininess'),
      uSpecAdd:         gl.getUniformLocation(p, 'uSpecAdd'),
      uPhongColor:      gl.getUniformLocation(p, 'uPhongColor'),
      uLightDir:        gl.getUniformLocation(p, 'uLightDir'),
      uAlpha:           gl.getUniformLocation(p, 'uAlpha'),
    };

    this.aPosition = gl.getAttribLocation(p, 'aPosition');
    this.aNormal   = gl.getAttribLocation(p, 'aNormal');
    this.aUv       = gl.getAttribLocation(p, 'aUv');
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /** Attach screen-space refraction FBO texture (tRefraction slot 0) */
  setRefractionTexture(tex: WebGLTexture): void {
    this.refrTex = tex;
  }

  /** Attach equirectangular environment texture (tEnv slot 1) */
  setEnvTexture(tex: WebGLTexture): void {
    this.envTex = tex;
  }

  /**
   * Update a single parameter at runtime (hot-tweakable, no recompile).
   * Key must be a valid GlassMaterialParams field name.
   */
  setParam<K extends keyof GlassMaterialParams>(key: K, value: GlassMaterialParams[K]): void {
    (this.params as GlassMaterialParams)[key] = value;
  }

  /** Apply an entire preset atomically */
  applyPreset(preset: Partial<GlassMaterialParams>): void {
    Object.assign(this.params, preset);
  }

  /**
   * Bind the material and upload all uniforms.
   * Must be called before gl.drawArrays / gl.drawElements.
   *
   * @param projection  4×4 column-major projection matrix (Float32Array)
   * @param modelView   4×4 column-major model-view matrix
   * @param model       4×4 column-major model matrix
   * @param normalMatrix 3×3 column-major normal matrix (mat3)
   * @param cameraPos   camera world-space position [x, y, z]
   * @param resolution  viewport [width, height] in pixels
   */
  bind(
    projection:   Float32Array,
    modelView:    Float32Array,
    model:        Float32Array,
    normalMatrix: Float32Array,
    cameraPos:    [number, number, number],
    resolution:   [number, number],
  ): void {
    const { gl, program, uloc, params } = this;

    gl.useProgram(program);

    // ── Matrices ────────────────────────────────────────────────────────────
    gl.uniformMatrix4fv(uloc.uProjection,   false, projection);
    gl.uniformMatrix4fv(uloc.uModelView,    false, modelView);
    gl.uniformMatrix4fv(uloc.uModel,        false, model);
    gl.uniformMatrix3fv(uloc.uNormalMatrix, false, normalMatrix);
    gl.uniform3f(uloc.uCameraPos, ...cameraPos);
    gl.uniform2f(uloc.uResolution, resolution[0], resolution[1]);

    // ── Textures ────────────────────────────────────────────────────────────
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.refrTex);
    gl.uniform1i(uloc.tRefraction, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.envTex);
    gl.uniform1i(uloc.tEnv, 1);

    // ── Fresnel ─────────────────────────────────────────────────────────────
    gl.uniform1f(uloc.uFresnelPow,   params.uFresnelPow);
    gl.uniform3f(uloc.uFresnelColor, ...params.uFresnelColor);
    gl.uniform1f(uloc.uFresnelAlpha, params.uFresnelAlpha);

    // ── Refraction ───────────────────────────────────────────────────────────
    gl.uniform1f(uloc.uRefractionRatio, params.uRefractionRatio);
    gl.uniform1f(uloc.uDistortStrength, params.uDistortStrength);

    // ── Environment ──────────────────────────────────────────────────────────
    gl.uniform1f(uloc.uEnvBlend,    params.uEnvBlend);
    gl.uniform1f(uloc.uAttenuation, params.uAttenuation);
    gl.uniform1f(uloc.uReflectScale, params.uReflectScale);

    // ── Specular ─────────────────────────────────────────────────────────────
    gl.uniform1f(uloc.uShininess,  params.uShininess);
    gl.uniform2f(uloc.uSpecAdd,    params.uSpecAdd[0], params.uSpecAdd[1]);
    gl.uniform3f(uloc.uPhongColor, ...params.uPhongColor);
    gl.uniform3f(uloc.uLightDir,   ...params.uLightDir);

    // ── Alpha ─────────────────────────────────────────────────────────────────
    gl.uniform1f(uloc.uAlpha, params.uAlpha);

    // ── Blending ─────────────────────────────────────────────────────────────
    if (params.uTransparent) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    }
  }

  /**
   * Bind geometry buffers for a standard mesh with aPosition / aNormal / aUv.
   * Call immediately after bind() and before your draw call.
   */
  bindBuffers(
    positionBuf: WebGLBuffer,
    normalBuf:   WebGLBuffer,
    uvBuf:       WebGLBuffer,
  ): void {
    const { gl } = this;

    if (this.aPosition >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuf);
      gl.enableVertexAttribArray(this.aPosition);
      gl.vertexAttribPointer(this.aPosition, 3, gl.FLOAT, false, 0, 0);
    }

    if (this.aNormal >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, normalBuf);
      gl.enableVertexAttribArray(this.aNormal);
      gl.vertexAttribPointer(this.aNormal, 3, gl.FLOAT, false, 0, 0);
    }

    if (this.aUv >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
      gl.enableVertexAttribArray(this.aUv);
      gl.vertexAttribPointer(this.aUv, 2, gl.FLOAT, false, 0, 0);
    }
  }

  /** Unbind the material after your draw call. */
  unbind(): void {
    const { gl, params } = this;

    if (this.aPosition >= 0) gl.disableVertexAttribArray(this.aPosition);
    if (this.aNormal   >= 0) gl.disableVertexAttribArray(this.aNormal);
    if (this.aUv       >= 0) gl.disableVertexAttribArray(this.aUv);

    if (params.uTransparent) gl.disable(gl.BLEND);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.useProgram(null);
  }

  /** Free GPU resources */
  dispose(): void {
    this.gl.deleteProgram(this.program);
  }

  // ── Getters ────────────────────────────────────────────────────────────────

  get glProgram(): WebGLProgram   { return this.program; }
  get currentParams(): Readonly<GlassMaterialParams> { return this.params; }

  // ── Private ────────────────────────────────────────────────────────────────

  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const { gl } = this;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[GlassMaterial] vert compile error (${label}):\n${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      gl.deleteShader(vs);
      throw new Error(`[GlassMaterial] frag compile error (${label}):\n${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error(`[GlassMaterial] link error (${label}):\n${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// GLSL source exports (for use with AT's NanoGL / shader system)
// ─────────────────────────────────────────────────────────────────────────────

export const GLASS_MATERIAL_VERT_SRC = GLASS_MATERIAL_VERT;
export const GLASS_MATERIAL_FRAG_SRC = GLASS_MATERIAL_FRAG;

// ─────────────────────────────────────────────────────────────────────────────
// Preset map
// ─────────────────────────────────────────────────────────────────────────────

export const GLASS_PRESETS = {
  glassCube:      PARAMS_GLASS_CUBE,
  workGlass:      PARAMS_WORK_GLASS,
  workGlassCube:  PARAMS_WORK_GLASS_CUBE,
  glassPBR:       PARAMS_GLASS_PBR,
  cleanroom:      PARAMS_CLEANROOM_GLASS,
} as const;

export type GlassPresetName = keyof typeof GLASS_PRESETS;
