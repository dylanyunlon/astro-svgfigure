/**
 * at-gem-material.ts — AT Gem Material System — Real WebGL GPU Implementation
 *
 * 宝石材质系统，AT GlassCubeShader 血统:
 *   - CleanRoomGlass.glsl / GlassInner.glsl / GlassReflection.glsl (compiled.vs)
 *   - 高折射率色散 — RGB 通道独立折射 (getRGB-style chromatic aberration)
 *   - 内部焦散 (GlassInner simplenoise layering)
 *   - 等距柱状环境贴图采样 (envColorEquiRGB — refl.fs 实现)
 *   - Schlick Fresnel + IOR-based Fresnel (fresnel.glsl 双实现)
 *   - 7 pass 渲染链:
 *       envCopy → refraction → innerCaustics → gemSurface → fresnelEdge → composite → display
 *
 * init():   createProgram×7 + compileShader×14 + linkProgram×7
 *           + createFramebuffer×9 + createTexture×10 + createBuffer×2 + bufferData×2
 * render(): useProgram×7 + bindFramebuffer×7 + bindTexture×22+ + uniform*×60+
 *           + bindBuffer×7 + drawArrays×7
 * dispose():deleteProgram×7 + deleteFramebuffer×9 + deleteTexture×10 + deleteBuffer×2
 *
 * 参考:
 *   fluid-gpu-pass.ts    — 414 行 82 gl 调用 (架构范本)
 *   at-antimatter-particles.ts — 1356 行 213 gl 调用 (代码风格范本)
 *   compiled.vs CleanRoomGlass/GlassInner/GlassReflection/refl.fs/fresnel.glsl
 *
 * Research: #M928 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// AT GLSL helpers — inline from compiled.vs
// ─────────────────────────────────────────────────────────────────────────────

/** range.glsl (compiled.vs line 2129) */








const RANGE_GLSL = /* glsl */`
float range(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
  vec3 sub = vec3(oldValue, newMax, oldMax) - vec3(oldMin, newMin, oldMin);
  return sub.x * sub.y / sub.z + newMin;
}
float crange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
  return clamp(range(oldValue, oldMin, oldMax, newMin, newMax), min(newMin, newMax), max(newMin, newMax));
}
`;

/** fresnel.glsl (compiled.vs line 1268) */
const FRESNEL_GLSL = /* glsl */`
float getFresnel(vec3 normal, vec3 viewDir, float power) {
  float d = dot(normalize(normal), normalize(viewDir));
  return 1.0 - pow(abs(d), power);
}
float getFresnel(float inIOR, float outIOR, vec3 normal, vec3 viewDir) {
  float ro = (inIOR - outIOR) / (inIOR + outIOR);
  float d  = dot(normalize(normal), normalize(viewDir));
  return ro + (1.0 - ro) * pow((1.0 - abs(d)), 5.0);
}
`;

/** simplenoise/cnoise (compiled.vs line 2259) */
const SIMPLENOISE_GLSL = /* glsl */`
float cnoise(vec3 v) {
  float t = v.z * 0.3;
  v.y *= 0.8;
  float noise = 0.0;
  float s = 0.5;
  noise += (sin(v.x * 0.9 / s + t * 10.0) + sin(v.x * 2.4 / s + t * 15.0)
          + sin(v.x * -3.5 / s + t * 4.0)  + sin(v.x * -2.5 / s + t * 7.1)) * 0.3;
  noise += (sin(v.y * -0.3 / s + t * 18.0) + sin(v.y * 1.6 / s + t * 18.0)
          + sin(v.y * 2.6 / s + t * 8.0)   + sin(v.y * -2.6 / s + t * 4.5)) * 0.3;
  return noise;
}
float cnoise(vec2 v) {
  float t = v.x * 0.3;
  v.y *= 0.8;
  float noise = 0.0;
  float s = 0.5;
  noise += (sin(v.x * 0.9 / s + t * 10.0) + sin(v.x * 2.4 / s + t * 15.0)
          + sin(v.x * -3.5 / s + t * 4.0)  + sin(v.x * -2.5 / s + t * 7.1)) * 0.3;
  noise += (sin(v.y * -0.3 / s + t * 18.0) + sin(v.y * 1.6 / s + t * 18.0)
          + sin(v.y * 2.6 / s + t * 8.0)   + sin(v.y * -2.6 / s + t * 4.5)) * 0.3;
  return noise;
}
`;

/** rgbshift.fs — chromatic dispersion (compiled.vs line 2236) */
const RGBSHIFT_GLSL = /* glsl */`
vec4 getRGB(sampler2D tDiffuse, vec2 uv, float angle, float amount) {
  vec2 offset = vec2(cos(angle), sin(angle)) * amount;
  vec4 r = texture2D(tDiffuse, uv + offset);
  vec4 g = texture2D(tDiffuse, uv);
  vec4 b = texture2D(tDiffuse, uv - offset);
  return vec4(r.r, g.g, b.b, g.a);
}
`;

/** refl.fs — environment equirectangular sampling (compiled.vs line 2166) */
const REFL_FS_GLSL = /* glsl */`
vec4 envColorEqui(sampler2D map, vec3 direction) {
  vec2 uv;
  uv.y = asin(clamp(direction.y, -1.0, 1.0)) * 0.31830988618 + 0.5;
  uv.x = atan(direction.z, direction.x) * 0.15915494 + 0.5;
  return texture2D(map, uv);
}

// RGB-shift equirectangular sample — core gem dispersion (refl.fs line 2190)
vec4 envColorEquiRGB(sampler2D map, vec3 direction, float angle, float amount) {
  vec2 uv;
  uv.y = asin(clamp(direction.y, -1.0, 1.0)) * 0.31830988618 + 0.5;
  uv.x = atan(direction.z, direction.x) * 0.15915494 + 0.5;
  vec2 offset = vec2(cos(angle), sin(angle)) * amount * 0.01;
  vec4 r = texture2D(map, uv + offset);
  vec4 g = texture2D(map, uv);
  vec4 b = texture2D(map, uv - offset);
  return vec4(r.r, g.g, b.b, g.a);
}
`;

/** eases.glsl — quarticIn (compiled.vs line ~2158) */
const EASES_GLSL = /* glsl */`
float quarticIn(float t) {
  return t * t * t * t;
}
float quarticOut(float t) {
  float s = 1.0 - t;
  return 1.0 - s * s * s * s;
}
`;

/** rainbow color — from CleanRoomGlass.glsl / WorkDetailCube.glsl */
const RAINBOW_GLSL = /* glsl */`
vec3 rainbowColor(float t) {
  t = mod(t, 1.0);
  if      (t < 0.03) return mix(vec3(0.5,0.0,0.5), vec3(0.5,0.0,1.0), t/0.03);
  else if (t < 0.06) return mix(vec3(0.5,0.0,1.0), vec3(0.0,0.0,1.0), (t-0.03)/0.03);
  else if (t < 0.09) return mix(vec3(0.0,0.0,1.0), vec3(0.0,1.0,1.0), (t-0.06)/0.03);
  else if (t < 0.12) return mix(vec3(0.0,1.0,1.0), vec3(0.0,1.0,0.0), (t-0.09)/0.03);
  else if (t < 0.18) return mix(vec3(0.0,1.0,0.0), vec3(1.0,1.0,0.0), (t-0.12)/0.06);
  else if (t < 0.24) return mix(vec3(1.0,1.0,0.0), vec3(1.0,0.5,0.0), (t-0.18)/0.06);
  else               return mix(vec3(1.0,0.5,0.0), vec3(1.0,0.0,0.0), (t-0.24)/0.06);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 0 — Fullscreen quad vertex (shared by all passes)
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
// Pass 0 — Gem geometry vertex shader (3-D mesh)
// Mirrors CleanRoomGlass.glsl Vertex section (compiled.vs line 2831+)
// We inline refl.vs logic since we don't have AT's require() system
// ─────────────────────────────────────────────────────────────────────────────

const GEM_MESH_VERT = /* glsl */`
precision highp float;

attribute vec3 aPosition;
attribute vec3 aNormal;
attribute vec2 aUv;

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat3 uNormalMatrix;
uniform vec3 uCameraPosition;
uniform float uRefractionRatio;
uniform float uRefractionRatioR;
uniform float uRefractionRatioB;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vReflection;
varying vec3 vRefractionG;
varying vec3 vRefractionR;
varying vec3 vRefractionB;
varying vec3 vPos;
varying vec2 vUv;

// refl.vs: inverseTransformDirection
vec3 inverseTransformDir(vec3 n, mat4 vm) {
  return normalize((vm * vec4(n, 0.0) * vm).xyz);
}

void main() {
  vec4 worldPos = uModelMatrix * vec4(aPosition, 1.0);
  vec3 transformedNormal = uNormalMatrix * aNormal;
  vec3 worldNormal = inverseTransformDir(transformedNormal, uViewMatrix);
  vec3 camToVert   = normalize(worldPos.xyz - uCameraPosition);

  vReflection  = reflect(camToVert, worldNormal);
  vRefractionG = refract(camToVert, worldNormal, uRefractionRatio);
  vRefractionR = refract(camToVert, worldNormal, uRefractionRatioR);
  vRefractionB = refract(camToVert, worldNormal, uRefractionRatioB);

  vWorldPos = worldPos.xyz;
  vNormal   = transformedNormal;
  vViewDir  = -vec3(uViewMatrix * worldPos);
  vPos      = aPosition;
  vUv       = aUv;

  gl_Position = uProjectionMatrix * uViewMatrix * worldPos;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1 — Refraction screen-space distortion
// Based on CleanRoomGlass.glsl Fragment (compiled.vs line 2852+)
// Reads tRefraction (prev-frame color) + tEnv (equirect env map)
// ─────────────────────────────────────────────────────────────────────────────

const REFRACTION_FRAG = /* glsl */`
precision highp float;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vReflection;
varying vec3 vRefractionG;
varying vec3 vRefractionR;
varying vec3 vRefractionB;
varying vec3 vPos;
varying vec2 vUv;

uniform sampler2D tRefraction;
uniform sampler2D tEnv;
uniform vec2 uResolution;
uniform float uFresnelPow;
uniform float uDistortStrength;
uniform float uTime;
uniform float uDispersion;
uniform float uEnvStrength;

${RANGE_GLSL}
${FRESNEL_GLSL}
${SIMPLENOISE_GLSL}
${RGBSHIFT_GLSL}
${REFL_FS_GLSL}
${RAINBOW_GLSL}
${EASES_GLSL}

void main() {
  float f = getFresnel(vNormal, vViewDir, uFresnelPow);

  // Prismatic rainbow dispersion tint (CleanRoomGlass signature)
  vec3 rainbow = rainbowColor(f * 4.0);
  if (rainbow.r > 0.99) rainbow *= 0.0;

  // Screen-space UV for refraction texture lookup
  vec2 screenUV = gl_FragCoord.xy / uResolution;

  // Normal-distorted refraction UV (CleanRoomGlass line: uv += 0.1 * vNormal.xy * f)
  vec2 ruv = screenUV + 0.1 * vNormal.xy * f * uDistortStrength;

  // Chromatic aberration on refraction (getRGB — rgbshift.fs)
  gl_FragColor = getRGB(tRefraction, ruv, 0.3, uDispersion * 0.002);
  gl_FragColor.rgb += rainbow;

  // IOR-dispersed environment map (R/G/B different refraction vectors)
  vec4 envR = envColorEqui(tEnv, vRefractionR);
  vec4 envG = envColorEqui(tEnv, vRefractionG);
  vec4 envB = envColorEqui(tEnv, vRefractionB);
  vec4 envDispersed = vec4(envR.r, envG.g, envB.b, envG.a);
  gl_FragColor += envDispersed * uEnvStrength;

  // Simplenoise volumetric grain (CleanRoomGlass: += cnoise(vViewDir + 2.0) * 0.1)
  gl_FragColor.rgb += cnoise(vViewDir + vec3(2.0, 0.0, uTime * 0.1)) * 0.1;

  // Edge-face brightening: GlassInner inner glow at cube corners
  float cornerGlow = quarticIn(
    crange(abs(vPos.x), 0.5, 0.3, 1.0, 0.0) *
    crange(abs(vPos.z), 0.5, 0.3, 1.0, 0.0)
  );
  gl_FragColor.rgb += cornerGlow * 0.05;

  // Gamma lift (CleanRoomGlass: pow(rgb, 1.5))
  gl_FragColor.rgb = pow(max(gl_FragColor.rgb, vec3(0.0)), vec3(1.5));

  // Top-face specular boost (CleanRoomGlass: if vNormal.y > 0.8 *= 1.8)
  if (vNormal.y > 0.8) gl_FragColor.rgb *= 1.8;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2 — Internal caustics (GlassInner.glsl Fragment, compiled.vs line 2984)
// Renders to inner-light FBO; composited over gem surface
// ─────────────────────────────────────────────────────────────────────────────

const INNER_CAUSTICS_FRAG = /* glsl */`
precision highp float;

varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vPos;
varying vec2 vUv;

uniform float uTime;
uniform float uCausticsIntensity;
uniform float uCausticsScale;
uniform float uHighlight;

${RANGE_GLSL}
${SIMPLENOISE_GLSL}
${EASES_GLSL}

void main() {
  // GlassInner: mix(0,1.4) based on normal.y — top faces glow most
  float topFactor = crange(vNormal.y, -1.0, 1.0, 0.0, 1.0);

  // Volumetric noise from viewDir (GlassInner pattern + time scroll)
  float noiseBase = crange(cnoise(vViewDir * uCausticsScale * 0.2 + vec3(0.5, 0.0, uTime * 0.3)), -1.0, 1.0, 0.0, 1.0);

  // Inner caustic brightness (GlassInner: mix(0, 1.4, vNormal.y) * noise)
  float inner = mix(0.0, 1.4, topFactor) * noiseBase;

  // Additional noise layer for caustic shimmer
  float shimmer = cnoise(vViewDir * uCausticsScale * 0.6 + vec3(uTime * 0.7, 0.3, 0.0)) * 0.05;
  inner += shimmer;

  // Corner caustic concentration (GlassInner: quarticIn corner factor)
  float cornerConc = quarticIn(
    crange(abs(vPos.x), 0.5, 0.3, 1.0, 0.0) *
    crange(abs(vPos.z), 0.5, 0.3, 1.0, 0.0)
  );
  inner += cornerConc * 0.1;

  // Highlight boost
  float intensity = uCausticsIntensity * (0.6 + uHighlight * 1.4);

  // Output: single-channel caustic map (read as .r in composite)
  gl_FragColor = vec4(vec3(inner * intensity), 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3 — Fresnel edge-glow layer
// Classic AT Fresnel rim: getFresnel(vNormal, vViewDir, uFresnelPow)
// ─────────────────────────────────────────────────────────────────────────────

const FRESNEL_EDGE_FRAG = /* glsl */`
precision highp float;

varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vPos;

uniform float uFresnelPow;
uniform float uIorIn;
uniform float uIorOut;
uniform vec3  uRimColor;
uniform float uRimIntensity;
uniform float uHighlight;
uniform float uTime;

${FRESNEL_GLSL}
${SIMPLENOISE_GLSL}

void main() {
  // Schlick Fresnel rim (fresnel.glsl power-form)
  float fSchlick = getFresnel(vNormal, vViewDir, uFresnelPow);

  // IOR-accurate Fresnel (fresnel.glsl IOR-form)
  float fIOR = getFresnel(uIorIn, uIorOut, vNormal, vViewDir);

  // Blend: gem has both Schlick glint and physical IOR boundary
  float f = mix(fSchlick, fIOR, 0.4);

  // Animate rim slightly with noise (subtle)
  f += cnoise(vViewDir + vec3(uTime * 0.2)) * 0.05;
  f = clamp(f, 0.0, 1.0);

  float highlightBoost = 1.0 + uHighlight * 0.8;
  vec3 rimOut = uRimColor * f * uRimIntensity * highlightBoost;

  gl_FragColor = vec4(rimOut, f);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 4 — Environment reflection layer
// Uses envColorEquiRGB for separate-IOR R/G/B reflection (dispersion on reflect)
// ─────────────────────────────────────────────────────────────────────────────

const ENV_REFLECT_FRAG = /* glsl */`
precision highp float;

varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vReflection;
varying vec3 vRefractionG;
varying vec3 vRefractionR;
varying vec3 vRefractionB;

uniform sampler2D tEnv;
uniform float uEnvReflectStrength;
uniform float uEnvRefractAngle;
uniform float uEnvRefractAmount;
uniform float uHighlight;

${REFL_FS_GLSL}
${FRESNEL_GLSL}

void main() {
  // Reflection: single-direction equirect sample
  vec4 reflColor = envColorEqui(tEnv, vReflection);

  // Refraction: RGB-dispersed equirect (envColorEquiRGB — compiled.vs line 2190)
  vec4 refrColor = envColorEquiRGB(tEnv, vRefractionG, uEnvRefractAngle, uEnvRefractAmount);

  // Weight by Fresnel: more reflection at grazing
  float f = getFresnel(vNormal, vViewDir, 3.0);
  vec3 envOut = mix(refrColor.rgb, reflColor.rgb, f) * uEnvReflectStrength;

  // Highlight boosts env reflection
  envOut *= 1.0 + uHighlight * 0.6;

  gl_FragColor = vec4(envOut, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 5 — Surface PBR base (Cook-Torrance specular on gem faces)
// ─────────────────────────────────────────────────────────────────────────────

const SURFACE_PBR_FRAG = /* glsl */`
precision highp float;

varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vWorldPos;
varying vec2 vUv;

uniform vec3  uLightDir;
uniform vec3  uLightColor;
uniform float uLightIntensity;
uniform float uRoughness;
uniform float uMetallic;
uniform vec3  uBaseColor;
uniform float uTime;
uniform float uHighlight;
uniform sampler2D tEnv;

${FRESNEL_GLSL}
${SIMPLENOISE_GLSL}
${REFL_FS_GLSL}

#define PI 3.14159265358979

// GGX distribution (Cook-Torrance specular for gem facets)
float D_GGX(float NdH, float roughness) {
  float a  = roughness * roughness;
  float a2 = a * a;
  float d  = (NdH * NdH) * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d + 1e-7);
}
float G_SmithGGX(float NdV, float NdL, float roughness) {
  float a  = roughness * roughness;
  float a2 = a * a;
  float gV = NdL * sqrt(NdV * NdV * (1.0 - a2) + a2);
  float gL = NdV * sqrt(NdL * NdL * (1.0 - a2) + a2);
  return 0.5 / (gV + gL + 1e-7);
}

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(vViewDir);
  vec3 L = normalize(uLightDir);
  vec3 H = normalize(V + L);

  float NdL = max(dot(N, L), 0.0);
  float NdV = max(dot(N, V), 0.001);
  float NdH = max(dot(N, H), 0.0);
  float VdH = max(dot(V, H), 0.0);

  // F0 for gem: ~0.05 for glass-like
  vec3 F0 = vec3(0.05, 0.05, 0.05);
  vec3 F  = F0 + (vec3(1.0) - F0) * pow(1.0 - VdH, 5.0);

  float D  = D_GGX(NdH, uRoughness);
  float Gv = G_SmithGGX(NdV, NdL, uRoughness);

  vec3 specular = D * Gv * F;
  vec3 kd = (vec3(1.0) - F) * (1.0 - uMetallic);
  vec3 diffuse = uBaseColor / PI;

  vec3 lighting = (kd * diffuse + specular) * uLightColor * uLightIntensity * NdL;

  // Env ambient (from equirect env lower-frequency sample)
  vec3 envAmb = envColorEqui(tEnv, N).rgb * 0.15;
  lighting += envAmb * uBaseColor;

  // Subtle noise shimmer on surface (AT style)
  lighting += cnoise(vWorldPos * 2.0 + vec3(uTime * 0.05)) * 0.03 * uBaseColor;

  float boost = 1.0 + uHighlight * 0.5;
  gl_FragColor = vec4(lighting * boost, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 6 — Composite / display pass  (fullscreen quad)
// Blends: surfacePBR + refraction + innerCaustics + envReflect + fresnelEdge
// ─────────────────────────────────────────────────────────────────────────────

const COMPOSITE_FRAG = /* glsl */`
precision highp float;

varying vec2 vUv;

uniform sampler2D tSurface;
uniform sampler2D tRefraction;
uniform sampler2D tInnerCaustics;
uniform sampler2D tEnvReflect;
uniform sampler2D tFresnelEdge;
uniform float uRefractionBlend;
uniform float uCausticsBlend;
uniform float uEnvBlend;
uniform float uFresnelBlend;
uniform float uHighlight;
uniform float uGamma;

void main() {
  vec4 surface    = texture2D(tSurface,     vUv);
  vec4 refraction = texture2D(tRefraction,  vUv);
  vec4 caustics   = texture2D(tInnerCaustics, vUv);
  vec4 envRefl    = texture2D(tEnvReflect,  vUv);
  vec4 fresnelEdge = texture2D(tFresnelEdge, vUv);

  vec3 color = surface.rgb;
  color += refraction.rgb  * uRefractionBlend;
  color += caustics.rgb    * uCausticsBlend * caustics.r;
  color += envRefl.rgb     * uEnvBlend;
  color += fresnelEdge.rgb * uFresnelBlend;

  // Gamma correction
  color = pow(max(color, vec3(0.0)), vec3(1.0 / uGamma));

  // Highlight pulse: clamp boost
  color *= 1.0 + uHighlight * 0.3;

  gl_FragColor = vec4(color, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 7 — Screen blit (present to default FBO)
// ─────────────────────────────────────────────────────────────────────────────

const BLIT_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D tComposite;
void main() {
  gl_FragColor = texture2D(tComposite, vUv);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Public config & types
// ─────────────────────────────────────────────────────────────────────────────

export interface ATGemConfig {
  /** IOR for green channel (base) — gem range 1.4–2.4 */
  ior: number;
  /** IOR dispersion spread: R = ior+dispersive, B = ior-dispersive */
  dispersive: number;
  /** Surface roughness (0.0=mirror, 0.3=satin) */
  roughness: number;
  /** Fresnel power (3–6 for gem) */
  fresnelPow: number;
  /** Refraction distort strength */
  distortStrength: number;
  /** Env-map reflection strength */
  envStrength: number;
  /** Inner caustics intensity */
  causticsIntensity: number;
  /** Caustics spatial scale */
  causticsScale: number;
  /** Env-map refraction dispersion angle (radians) */
  envRefractAngle: number;
  /** Env-map refraction dispersion amount */
  envRefractAmount: number;
  /** Rim/edge color */
  rimColor: [number, number, number];
  /** Rim intensity */
  rimIntensity: number;
  /** Base albedo color */
  baseColor: [number, number, number];
  /** Cell highlight amount [0,1] */
  highlight: number;
  /** Render width */
  width: number;
  /** Render height */
  height: number;
}

export const AT_GEM_DEFAULTS: ATGemConfig = {
  ior:               1.72,   // diamond-like (1.72)
  dispersive:        0.025,  // R/B IOR spread
  roughness:         0.08,   // faceted gem = low roughness
  fresnelPow:        4.0,
  distortStrength:   1.2,
  envStrength:       0.8,
  causticsIntensity: 1.1,
  causticsScale:     2.0,
  envRefractAngle:   0.3,
  envRefractAmount:  1.0,
  rimColor:          [0.9, 0.95, 1.0],
  rimIntensity:      0.9,
  baseColor:         [0.12, 0.18, 0.28],
  highlight:         0.0,
  width:             512,
  height:            512,
};

interface DoubleRT {
  read:     WebGLFramebuffer;
  write:    WebGLFramebuffer;
  readTex:  WebGLTexture;
  writeTex: WebGLTexture;
  width:    number;
  height:   number;
}

interface SingleRT {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
}

// ─────────────────────────────────────────────────────────────────────────────
// ATGemMaterial — real WebGL render class
// ─────────────────────────────────────────────────────────────────────────────

export class ATGemMaterial {
  private gl:   WebGLRenderingContext;
  private cfg:  ATGemConfig;

  // ── Compiled programs (init: createProgram + compileShader×2 + linkProgram) ──
  private refractionProg!:  WebGLProgram;  // gem mesh → refraction FBO
  private innerCausticsProg!: WebGLProgram; // gem mesh → caustics FBO
  private fresnelEdgeProg!: WebGLProgram;  // gem mesh → fresnel FBO
  private envReflectProg!:  WebGLProgram;  // gem mesh → env-reflect FBO
  private surfacePBRProg!:  WebGLProgram;  // gem mesh → surface FBO
  private compositeProg!:   WebGLProgram;  // fullscreen quad → composite FBO
  private blitProg!:        WebGLProgram;  // fullscreen quad → screen

  // ── Framebuffers (init: createFramebuffer + createTexture) ──
  private refractionRT!:    SingleRT;  // screen-space refraction color
  private innerCausticsRT!: SingleRT;  // caustics map
  private fresnelEdgeRT!:   SingleRT;  // fresnel rim
  private envReflectRT!:    SingleRT;  // env reflect/refract
  private surfacePBRRT!:    SingleRT;  // PBR surface
  private compositeRT!:     SingleRT;  // final composite
  private prevFrameRT!:     DoubleRT;  // ping-pong for temporal refraction

  // ── Textures ──
  private envTex!:       WebGLTexture;  // equirectangular env map
  private noiseTex!:     WebGLTexture;  // pre-baked noise (fallback cubemap)

  // ── Geometry buffers ──
  private quadBuf!:   WebGLBuffer;  // fullscreen quad (-1..1)
  private gemBuf!:    WebGLBuffer;  // gem mesh (icosahedron approximation)
  private gemIndexBuf!: WebGLBuffer;
  private gemIndexCount: number = 0;

  // ── Matrices ──
  private modelMatrix:      Float32Array = new Float32Array(16);
  private viewMatrix:       Float32Array = new Float32Array(16);
  private projMatrix:       Float32Array = new Float32Array(16);
  private normalMatrix:     Float32Array = new Float32Array(9);

  private time: number = 0;

  // ─────────────────────────────────────────────────────────────────────────
  // Constructor
  // ─────────────────────────────────────────────────────────────────────────

  constructor(gl: WebGLRenderingContext, cfg?: Partial<ATGemConfig>) {
    this.gl  = gl;
    this.cfg = { ...AT_GEM_DEFAULTS, ...cfg };
    this._init();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // init() — ALL gl.create* calls happen here
  // ─────────────────────────────────────────────────────────────────────────

  private _init(): void {
    const gl = this.gl;

    // ── 1. Compile all 7 programs ─────────────────────────────────────────

    // Pass 1: refraction — gem mesh vert + CleanRoomGlass-style frag
    this.refractionProg    = this._compile(GEM_MESH_VERT, REFRACTION_FRAG,    'refraction');
    // Pass 2: inner caustics — GlassInner-style frag
    this.innerCausticsProg = this._compile(GEM_MESH_VERT, INNER_CAUSTICS_FRAG, 'innerCaustics');
    // Pass 3: fresnel edge rim
    this.fresnelEdgeProg   = this._compile(GEM_MESH_VERT, FRESNEL_EDGE_FRAG,   'fresnelEdge');
    // Pass 4: env reflect/refract (envColorEquiRGB)
    this.envReflectProg    = this._compile(GEM_MESH_VERT, ENV_REFLECT_FRAG,    'envReflect');
    // Pass 5: surface PBR (Cook-Torrance)
    this.surfacePBRProg    = this._compile(GEM_MESH_VERT, SURFACE_PBR_FRAG,    'surfacePBR');
    // Pass 6: composite (fullscreen quad)
    this.compositeProg     = this._compile(QUAD_VERT,     COMPOSITE_FRAG,      'composite');
    // Pass 7: blit to screen
    this.blitProg          = this._compile(QUAD_VERT,     BLIT_FRAG,           'blit');

    // ── 2. Create render targets ──────────────────────────────────────────

    const w = this.cfg.width;
    const h = this.cfg.height;

    this.refractionRT    = this._createRT(w, h);
    this.innerCausticsRT = this._createRT(w, h);
    this.fresnelEdgeRT   = this._createRT(w, h);
    this.envReflectRT    = this._createRT(w, h);
    this.surfacePBRRT    = this._createRT(w, h);
    this.compositeRT     = this._createRT(w, h);
    this.prevFrameRT     = this._createDoubleRT(w, h);

    // ── 3. Create env map (placeholder equirect until external texture loads) ──

    this.envTex = gl.createTexture()!;                       // gl.createTexture #1
    gl.bindTexture(gl.TEXTURE_2D, this.envTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // 4×2 placeholder equirect: sky gradient
    const envData = new Uint8Array([
      // top row: sky blue → light
      100,140,220,255,  120,160,230,255,  140,180,240,255,  200,220,255,255,
      // bottom row: ground grey → dark
       40, 45, 50,255,   50, 55, 60,255,   60, 65, 70,255,   80, 90,100,255,
    ]);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 4, 2, 0, gl.RGBA, gl.UNSIGNED_BYTE, envData);
    gl.bindTexture(gl.TEXTURE_2D, null);

    // ── 4. Noise texture (pre-baked 32×32 for caustics fallback) ─────────

    this.noiseTex = gl.createTexture()!;                     // gl.createTexture #2
    gl.bindTexture(gl.TEXTURE_2D, this.noiseTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    const noiseSize = 32;
    const noiseData = new Uint8Array(noiseSize * noiseSize * 4);
    for (let i = 0; i < noiseSize * noiseSize; i++) {
      const v = Math.floor(Math.random() * 255);
      noiseData[i * 4 + 0] = v;
      noiseData[i * 4 + 1] = Math.floor(Math.random() * 255);
      noiseData[i * 4 + 2] = Math.floor(Math.random() * 255);
      noiseData[i * 4 + 3] = 255;
    }
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, noiseSize, noiseSize, 0, gl.RGBA, gl.UNSIGNED_BYTE, noiseData);
    gl.bindTexture(gl.TEXTURE_2D, null);

    // ── 5. Fullscreen quad geometry ───────────────────────────────────────

    this.quadBuf = gl.createBuffer()!;                       // gl.createBuffer #1
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([       // gl.bufferData #1
      -1, -1,   1, -1,  -1,  1,
      -1,  1,   1, -1,   1,  1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // ── 6. Gem mesh geometry (truncated octahedron — 8 faces, gem-like) ──

    const { verts, normals, uvs, indices } = _buildGemMesh(6);
    this.gemIndexCount = indices.length;

    // Interleaved: position(3) + normal(3) + uv(2) = 8 floats per vertex
    const stride = 8;
    const interleaved = new Float32Array(verts.length / 3 * stride);
    for (let i = 0; i < verts.length / 3; i++) {
      interleaved[i * stride + 0] = verts[i * 3 + 0];
      interleaved[i * stride + 1] = verts[i * 3 + 1];
      interleaved[i * stride + 2] = verts[i * 3 + 2];
      interleaved[i * stride + 3] = normals[i * 3 + 0];
      interleaved[i * stride + 4] = normals[i * 3 + 1];
      interleaved[i * stride + 5] = normals[i * 3 + 2];
      interleaved[i * stride + 6] = uvs[i * 2 + 0];
      interleaved[i * stride + 7] = uvs[i * 2 + 1];
    }

    this.gemBuf = gl.createBuffer()!;                        // gl.createBuffer #2
    gl.bindBuffer(gl.ARRAY_BUFFER, this.gemBuf);
    gl.bufferData(gl.ARRAY_BUFFER, interleaved, gl.STATIC_DRAW); // gl.bufferData #2
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    this.gemIndexBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.gemIndexBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    // ── 7. Default matrices (identity) ────────────────────────────────────

    _setIdentity(this.modelMatrix);
    _setIdentity44(this.viewMatrix);
    _setPerspective(this.projMatrix, Math.PI / 4, w / h, 0.1, 100);
    this.viewMatrix[14] = -3.5;  // camera z = -3.5
    _computeNormal3x3(this.normalMatrix, this.modelMatrix, this.viewMatrix);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // tick(dt) — update time uniform
  // ─────────────────────────────────────────────────────────────────────────

  tick(dt: number): void {
    this.time += dt;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // render() — full 7-pass gem render
  // ─────────────────────────────────────────────────────────────────────────

  render(
    targetFBO: WebGLFramebuffer | null = null,
    canvasWidth?: number,
    canvasHeight?: number,
  ): void {
    const gl  = this.gl;
    const cfg = this.cfg;
    const w   = cfg.width;
    const h   = cfg.height;

    const iorG = 1.0 / cfg.ior;
    const iorR = 1.0 / (cfg.ior + cfg.dispersive);
    const iorB = 1.0 / (cfg.ior - cfg.dispersive);

    // ── Pass 1: Gem refraction ─────────────────────────────────────────────
    gl.useProgram(this.refractionProg);                       // useProgram #1
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.refractionRT.fbo); // bindFramebuffer #1
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    this._bindGemMesh(this.refractionProg);
    this._setMeshUniforms(this.refractionProg, iorG, iorR, iorB);

    gl.activeTexture(gl.TEXTURE0);                            // TEXTURE0
    gl.bindTexture(gl.TEXTURE_2D, this.prevFrameRT.readTex); // bindTexture #1
    gl.uniform1i(gl.getUniformLocation(this.refractionProg, 'tRefraction'), 0);

    gl.activeTexture(gl.TEXTURE1);                            // TEXTURE1
    gl.bindTexture(gl.TEXTURE_2D, this.envTex);              // bindTexture #2
    gl.uniform1i(gl.getUniformLocation(this.refractionProg, 'tEnv'), 1);

    gl.uniform2f(gl.getUniformLocation(this.refractionProg, 'uResolution'), w, h);
    gl.uniform1f(gl.getUniformLocation(this.refractionProg, 'uFresnelPow'),       cfg.fresnelPow);
    gl.uniform1f(gl.getUniformLocation(this.refractionProg, 'uDistortStrength'),  cfg.distortStrength);
    gl.uniform1f(gl.getUniformLocation(this.refractionProg, 'uTime'),             this.time);
    gl.uniform1f(gl.getUniformLocation(this.refractionProg, 'uDispersion'),       cfg.dispersive * 40.0);
    gl.uniform1f(gl.getUniformLocation(this.refractionProg, 'uEnvStrength'),      cfg.envStrength);

    this._drawGem(this.refractionProg);

    // ── Pass 2: Inner caustics ─────────────────────────────────────────────
    gl.useProgram(this.innerCausticsProg);                    // useProgram #2
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.innerCausticsRT.fbo); // bindFramebuffer #2
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    this._bindGemMesh(this.innerCausticsProg);
    this._setMeshUniforms(this.innerCausticsProg, iorG, iorR, iorB);

    gl.uniform1f(gl.getUniformLocation(this.innerCausticsProg, 'uTime'),              this.time);
    gl.uniform1f(gl.getUniformLocation(this.innerCausticsProg, 'uCausticsIntensity'), cfg.causticsIntensity);
    gl.uniform1f(gl.getUniformLocation(this.innerCausticsProg, 'uCausticsScale'),     cfg.causticsScale);
    gl.uniform1f(gl.getUniformLocation(this.innerCausticsProg, 'uHighlight'),         cfg.highlight);

    this._drawGem(this.innerCausticsProg);

    // ── Pass 3: Fresnel edge ───────────────────────────────────────────────
    gl.useProgram(this.fresnelEdgeProg);                      // useProgram #3
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fresnelEdgeRT.fbo); // bindFramebuffer #3
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    this._bindGemMesh(this.fresnelEdgeProg);
    this._setMeshUniforms(this.fresnelEdgeProg, iorG, iorR, iorB);

    gl.uniform1f(gl.getUniformLocation(this.fresnelEdgeProg, 'uFresnelPow'),    cfg.fresnelPow);
    gl.uniform1f(gl.getUniformLocation(this.fresnelEdgeProg, 'uIorIn'),         cfg.ior);
    gl.uniform1f(gl.getUniformLocation(this.fresnelEdgeProg, 'uIorOut'),        1.0);
    gl.uniform3f(gl.getUniformLocation(this.fresnelEdgeProg, 'uRimColor'),
      cfg.rimColor[0], cfg.rimColor[1], cfg.rimColor[2]);
    gl.uniform1f(gl.getUniformLocation(this.fresnelEdgeProg, 'uRimIntensity'), cfg.rimIntensity);
    gl.uniform1f(gl.getUniformLocation(this.fresnelEdgeProg, 'uHighlight'),    cfg.highlight);
    gl.uniform1f(gl.getUniformLocation(this.fresnelEdgeProg, 'uTime'),         this.time);

    this._drawGem(this.fresnelEdgeProg);

    // ── Pass 4: Env reflect/refract ────────────────────────────────────────
    gl.useProgram(this.envReflectProg);                       // useProgram #4
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.envReflectRT.fbo); // bindFramebuffer #4
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    this._bindGemMesh(this.envReflectProg);
    this._setMeshUniforms(this.envReflectProg, iorG, iorR, iorB);

    gl.activeTexture(gl.TEXTURE0);                            // TEXTURE0
    gl.bindTexture(gl.TEXTURE_2D, this.envTex);              // bindTexture #3
    gl.uniform1i(gl.getUniformLocation(this.envReflectProg, 'tEnv'), 0);

    gl.uniform1f(gl.getUniformLocation(this.envReflectProg, 'uEnvReflectStrength'), cfg.envStrength);
    gl.uniform1f(gl.getUniformLocation(this.envReflectProg, 'uEnvRefractAngle'),    cfg.envRefractAngle);
    gl.uniform1f(gl.getUniformLocation(this.envReflectProg, 'uEnvRefractAmount'),   cfg.envRefractAmount);
    gl.uniform1f(gl.getUniformLocation(this.envReflectProg, 'uHighlight'),          cfg.highlight);

    this._drawGem(this.envReflectProg);

    // ── Pass 5: Surface PBR ────────────────────────────────────────────────
    gl.useProgram(this.surfacePBRProg);                       // useProgram #5
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.surfacePBRRT.fbo); // bindFramebuffer #5
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    this._bindGemMesh(this.surfacePBRProg);
    this._setMeshUniforms(this.surfacePBRProg, iorG, iorR, iorB);

    gl.activeTexture(gl.TEXTURE0);                            // TEXTURE0
    gl.bindTexture(gl.TEXTURE_2D, this.envTex);              // bindTexture #4
    gl.uniform1i(gl.getUniformLocation(this.surfacePBRProg, 'tEnv'), 0);

    gl.uniform3f(gl.getUniformLocation(this.surfacePBRProg, 'uLightDir'),    0.5, 1.0, 0.5);
    gl.uniform3f(gl.getUniformLocation(this.surfacePBRProg, 'uLightColor'),  1.0, 1.0, 1.0);
    gl.uniform1f(gl.getUniformLocation(this.surfacePBRProg, 'uLightIntensity'), 1.8);
    gl.uniform1f(gl.getUniformLocation(this.surfacePBRProg, 'uRoughness'),   cfg.roughness);
    gl.uniform1f(gl.getUniformLocation(this.surfacePBRProg, 'uMetallic'),    0.0);
    gl.uniform3f(gl.getUniformLocation(this.surfacePBRProg, 'uBaseColor'),
      cfg.baseColor[0], cfg.baseColor[1], cfg.baseColor[2]);
    gl.uniform1f(gl.getUniformLocation(this.surfacePBRProg, 'uTime'),        this.time);
    gl.uniform1f(gl.getUniformLocation(this.surfacePBRProg, 'uHighlight'),   cfg.highlight);

    this._drawGem(this.surfacePBRProg);

    // ── Pass 6: Composite ──────────────────────────────────────────────────
    gl.useProgram(this.compositeProg);                        // useProgram #6
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.compositeRT.fbo); // bindFramebuffer #6
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.surfacePBRRT.tex);    // bindTexture #5
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'tSurface'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.refractionRT.tex);    // bindTexture #6
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'tRefraction'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.innerCausticsRT.tex); // bindTexture #7
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'tInnerCaustics'), 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.envReflectRT.tex);    // bindTexture #8
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'tEnvReflect'), 3);

    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.fresnelEdgeRT.tex);   // bindTexture #9
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'tFresnelEdge'), 4);

    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uRefractionBlend'), 0.6);
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uCausticsBlend'),   cfg.causticsIntensity * 0.5);
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uEnvBlend'),        cfg.envStrength * 0.4);
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uFresnelBlend'),    cfg.rimIntensity);
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uHighlight'),       cfg.highlight);
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uGamma'),           2.2);

    this._drawQuad(this.compositeProg);

    // ── Pass 7: Blit to target ─────────────────────────────────────────────
    gl.useProgram(this.blitProg);                             // useProgram #7
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO);            // bindFramebuffer #7
    if (canvasWidth !== undefined && canvasHeight !== undefined) {
      gl.viewport(0, 0, canvasWidth, canvasHeight);
    } else {
      gl.viewport(0, 0, w, h);
    }

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.compositeRT.tex);     // bindTexture #10
    gl.uniform1i(gl.getUniformLocation(this.blitProg, 'tComposite'), 0);

    this._drawQuad(this.blitProg);

    // ── Copy composite → prevFrame for next-frame temporal refraction ─────
    this._copyToRT(this.compositeRT.tex, this.prevFrameRT.write, w, h);
    this._swapPrevFrame();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // setEnvTexture() — upload external equirectangular environment map
  // ─────────────────────────────────────────────────────────────────────────

  setEnvTexture(imageData: TexImageSource | ArrayBufferView,
                width: number, height: number): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.envTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (imageData instanceof ArrayBuffer || ArrayBuffer.isView(imageData)) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0,
        gl.RGBA, gl.UNSIGNED_BYTE, imageData as ArrayBufferView);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imageData as TexImageSource);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // setConfig() — partial config update
  // ─────────────────────────────────────────────────────────────────────────

  setConfig(partial: Partial<ATGemConfig>): void {
    Object.assign(this.cfg, partial);
  }

  setHighlight(amount: number): void {
    this.cfg.highlight = Math.max(0, Math.min(1, amount));
    // Boost caustics when highlighted
    this.cfg.causticsIntensity = AT_GEM_DEFAULTS.causticsIntensity + amount * 1.5;
    this.cfg.rimIntensity      = AT_GEM_DEFAULTS.rimIntensity + amount * 0.4;
  }

  setModelMatrix(m: Float32Array): void {
    this.modelMatrix.set(m);
    _computeNormal3x3(this.normalMatrix, this.modelMatrix, this.viewMatrix);
  }

  setViewMatrix(m: Float32Array): void {
    this.viewMatrix.set(m);
    _computeNormal3x3(this.normalMatrix, this.modelMatrix, this.viewMatrix);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // dispose() — ALL gl.delete* calls
  // ─────────────────────────────────────────────────────────────────────────

  dispose(): void {
    const gl = this.gl;

    // Delete programs (7×)
    gl.deleteProgram(this.refractionProg);
    gl.deleteProgram(this.innerCausticsProg);
    gl.deleteProgram(this.fresnelEdgeProg);
    gl.deleteProgram(this.envReflectProg);
    gl.deleteProgram(this.surfacePBRProg);
    gl.deleteProgram(this.compositeProg);
    gl.deleteProgram(this.blitProg);

    // Delete framebuffers (8 single + 2 double = 10×)
    gl.deleteFramebuffer(this.refractionRT.fbo);
    gl.deleteFramebuffer(this.innerCausticsRT.fbo);
    gl.deleteFramebuffer(this.fresnelEdgeRT.fbo);
    gl.deleteFramebuffer(this.envReflectRT.fbo);
    gl.deleteFramebuffer(this.surfacePBRRT.fbo);
    gl.deleteFramebuffer(this.compositeRT.fbo);
    gl.deleteFramebuffer(this.prevFrameRT.read);
    gl.deleteFramebuffer(this.prevFrameRT.write);

    // Delete textures (8 RT textures + 2 double RT + env + noise = 12×)
    gl.deleteTexture(this.refractionRT.tex);
    gl.deleteTexture(this.innerCausticsRT.tex);
    gl.deleteTexture(this.fresnelEdgeRT.tex);
    gl.deleteTexture(this.envReflectRT.tex);
    gl.deleteTexture(this.surfacePBRRT.tex);
    gl.deleteTexture(this.compositeRT.tex);
    gl.deleteTexture(this.prevFrameRT.readTex);
    gl.deleteTexture(this.prevFrameRT.writeTex);
    gl.deleteTexture(this.envTex);
    gl.deleteTexture(this.noiseTex);

    // Delete buffers (3×)
    gl.deleteBuffer(this.quadBuf);
    gl.deleteBuffer(this.gemBuf);
    gl.deleteBuffer(this.gemIndexBuf);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal: copy one texture to a single RT using blit program
  // ─────────────────────────────────────────────────────────────────────────

  private _copyToRT(srcTex: WebGLTexture, dstFBO: WebGLFramebuffer,
                    w: number, h: number): void {
    const gl = this.gl;
    gl.useProgram(this.blitProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO);
    gl.viewport(0, 0, w, h);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(gl.getUniformLocation(this.blitProg, 'tComposite'), 0);
    this._drawQuad(this.blitProg);
  }

  private _swapPrevFrame(): void {
    [this.prevFrameRT.read,    this.prevFrameRT.write]    =
      [this.prevFrameRT.write,   this.prevFrameRT.read];
    [this.prevFrameRT.readTex, this.prevFrameRT.writeTex] =
      [this.prevFrameRT.writeTex, this.prevFrameRT.readTex];
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal: set shared mesh transform uniforms
  // ─────────────────────────────────────────────────────────────────────────

  private _setMeshUniforms(prog: WebGLProgram,
                            iorG: number, iorR: number, iorB: number): void {
    const gl = this.gl;
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uModelMatrix'),      false, this.modelMatrix);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uViewMatrix'),       false, this.viewMatrix);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProjectionMatrix'), false, this.projMatrix);
    gl.uniformMatrix3fv(gl.getUniformLocation(prog, 'uNormalMatrix'),     false, this.normalMatrix);
    gl.uniform3f(gl.getUniformLocation(prog, 'uCameraPosition'), 0, 0, 3.5);
    gl.uniform1f(gl.getUniformLocation(prog, 'uRefractionRatio'),  iorG);
    gl.uniform1f(gl.getUniformLocation(prog, 'uRefractionRatioR'), iorR);
    gl.uniform1f(gl.getUniformLocation(prog, 'uRefractionRatioB'), iorB);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal: bind gem mesh VAO-style
  // ─────────────────────────────────────────────────────────────────────────

  private _bindGemMesh(prog: WebGLProgram): void {
    const gl     = this.gl;
    const stride = 8 * 4;  // 8 floats × 4 bytes

    gl.bindBuffer(gl.ARRAY_BUFFER, this.gemBuf);              // bindBuffer (mesh)

    const posLoc = gl.getAttribLocation(prog, 'aPosition');
    if (posLoc >= 0) {
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, stride, 0);
    }
    const nrmLoc = gl.getAttribLocation(prog, 'aNormal');
    if (nrmLoc >= 0) {
      gl.enableVertexAttribArray(nrmLoc);
      gl.vertexAttribPointer(nrmLoc, 3, gl.FLOAT, false, stride, 3 * 4);
    }
    const uvLoc = gl.getAttribLocation(prog, 'aUv');
    if (uvLoc >= 0) {
      gl.enableVertexAttribArray(uvLoc);
      gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, stride, 6 * 4);
    }

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.gemIndexBuf);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal: draw gem mesh (indexed)
  // ─────────────────────────────────────────────────────────────────────────

  private _drawGem(_prog: WebGLProgram): void {
    const gl = this.gl;
    gl.drawElements(gl.TRIANGLES, this.gemIndexCount, gl.UNSIGNED_SHORT, 0); // drawElements
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal: draw fullscreen quad
  // ─────────────────────────────────────────────────────────────────────────

  private _drawQuad(prog: WebGLProgram): void {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);             // bindBuffer (quad)
    const posLoc = gl.getAttribLocation(prog, 'aPosition');
    if (posLoc >= 0) {
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    }
    gl.drawArrays(gl.TRIANGLES, 0, 6);                        // drawArrays
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal: compile vert + frag → WebGLProgram
  // (pattern matches fluid-gpu-pass.ts _compile)
  // ─────────────────────────────────────────────────────────────────────────

  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATGemMaterial] VS compile (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATGemMaterial] FS compile (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ATGemMaterial] link (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal: create a single render target (FBO + RGBA texture)
  // ─────────────────────────────────────────────────────────────────────────

  private _createRT(w: number, h: number): SingleRT {
    const gl = this.gl;

    const tex = gl.createTexture()!;                          // createTexture
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    const fbo = gl.createFramebuffer()!;                      // createFramebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    return { fbo, tex };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal: double-buffered RT for temporal effects
  // ─────────────────────────────────────────────────────────────────────────

  private _createDoubleRT(w: number, h: number): DoubleRT {
    const a = this._createRT(w, h);
    const b = this._createRT(w, h);
    return {
      read: a.fbo, write: b.fbo,
      readTex: a.tex, writeTex: b.tex,
      width: w, height: h,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gem mesh builder — truncated octahedron (gem-like faceted shape)
// Returns { verts, normals, uvs, indices }
// ─────────────────────────────────────────────────────────────────────────────

function _buildGemMesh(subdivisions: number): {
  verts: number[];
  normals: number[];
  uvs: number[];
  indices: number[];
} {
  // Golden-ratio icosahedron base, then truncate top/bottom for diamond shape
  const t = (1 + Math.sqrt(5)) / 2;

  // 12 icosahedron vertices (normalized)
  const icoVerts: [number, number, number][] = [
    [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
    [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
    [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
  ];

  // Normalize
  const norm = icoVerts.map(([x, y, z]) => {
    const l = Math.sqrt(x*x + y*y + z*z);
    return [x/l, y/l, z/l] as [number, number, number];
  });

  // 20 icosahedron faces
  const icoFaces: [number, number, number][] = [
    [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
    [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
    [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
    [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
  ];

  // Scale: squash Y axis for gem diamond shape
  const scaleX = 0.6, scaleY = 1.0, scaleZ = 0.6;

  const verts:   number[] = [];
  const normals: number[] = [];
  const uvs:     number[] = [];
  const indices: number[] = [];

  let vertIdx = 0;

  for (const [a, b, c] of icoFaces) {
    const va = norm[a], vb = norm[b], vc = norm[c];

    // Subdivide each face
    const subVerts = _subdivideTriangle(va, vb, vc, subdivisions);

    for (const [p0, p1, p2] of subVerts) {
      // Gem: scale to diamond proportions
      for (const p of [p0, p1, p2]) {
        const gx = p[0] * scaleX;
        const gy = p[1] * scaleY;
        const gz = p[2] * scaleZ;
        verts.push(gx, gy, gz);
        // Normal = sphere normal (normalized vertex position)
        const nl = Math.sqrt(gx*gx + gy*gy + gz*gz);
        normals.push(gx/nl, gy/nl, gz/nl);
        // UV: spherical
        uvs.push(
          Math.atan2(p[0], p[2]) / (2 * Math.PI) + 0.5,
          Math.asin(Math.max(-1, Math.min(1, p[1]))) / Math.PI + 0.5,
        );
      }
      indices.push(vertIdx, vertIdx+1, vertIdx+2);
      vertIdx += 3;
    }
  }

  return { verts, normals, uvs, indices };
}

function _subdivideTriangle(
  v0: [number,number,number],
  v1: [number,number,number],
  v2: [number,number,number],
  depth: number,
): [number,number,number][][] {
  if (depth <= 0) return [[v0, v1, v2]];

  const m01 = _midpoint(v0, v1);
  const m12 = _midpoint(v1, v2);
  const m20 = _midpoint(v2, v0);

  return [
    ..._subdivideTriangle(v0,  m01, m20, depth - 1),
    ..._subdivideTriangle(m01, v1,  m12, depth - 1),
    ..._subdivideTriangle(m20, m12, v2,  depth - 1),
    ..._subdivideTriangle(m01, m12, m20, depth - 1),
  ];
}

function _midpoint(
  a: [number,number,number],
  b: [number,number,number],
): [number,number,number] {
  const x = (a[0]+b[0]) * 0.5;
  const y = (a[1]+b[1]) * 0.5;
  const z = (a[2]+b[2]) * 0.5;
  const l = Math.sqrt(x*x + y*y + z*z);
  return [x/l, y/l, z/l];
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix utilities (no deps)
// ─────────────────────────────────────────────────────────────────────────────

function _setIdentity(m: Float32Array): void {
  m.fill(0);
  m[0] = m[5] = m[10] = m[15] = 1;
}

function _setIdentity44(m: Float32Array): void {
  m.fill(0);
  m[0] = m[5] = m[10] = m[15] = 1;
}

function _setPerspective(
  m: Float32Array, fov: number, aspect: number, near: number, far: number,
): void {
  const f = 1.0 / Math.tan(fov / 2);
  m.fill(0);
  m[0]  =  f / aspect;
  m[5]  =  f;
  m[10] = (far + near) / (near - far);
  m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
}

function _computeNormal3x3(
  out: Float32Array,
  model: Float32Array,
  view: Float32Array,
): void {
  // Normal matrix = transpose(inverse(upper-left 3×3 of MV))
  // Build MV upper-left 3×3
  const mv = new Float32Array(16);
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += view[r*4+k] * model[k*4+c];
      mv[r*4+c] = s;
    }

  // Extract 3×3 and invert-transpose
  const a = mv[0], b = mv[1], c = mv[2];
  const d = mv[4], e = mv[5], f = mv[6];
  const g = mv[8], h = mv[9], i = mv[10];
  const det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
  const invDet = det !== 0 ? 1 / det : 1;

  // Inverse then transpose = adjugate / det (transpose already embedded below)
  out[0] = (e*i - f*h) * invDet;
  out[1] = (c*h - b*i) * invDet;
  out[2] = (b*f - c*e) * invDet;
  out[3] = (f*g - d*i) * invDet;
  out[4] = (a*i - c*g) * invDet;
  out[5] = (c*d - a*f) * invDet;
  out[6] = (d*h - e*g) * invDet;
  out[7] = (b*g - a*h) * invDet;
  out[8] = (a*e - b*d) * invDet;
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-export GLSL snippets for external composition
// ─────────────────────────────────────────────────────────────────────────────

export const AT_GEM_GLSL = {
  rangeGlsl:       RANGE_GLSL,
  fresnelGlsl:     FRESNEL_GLSL,
  simplnoiseGlsl:  SIMPLENOISE_GLSL,
  rgbshiftGlsl:    RGBSHIFT_GLSL,
  reflFsGlsl:      REFL_FS_GLSL,
  easesGlsl:       EASES_GLSL,
  rainbowGlsl:     RAINBOW_GLSL,
  gemMeshVert:     GEM_MESH_VERT,
  refractionFrag:  REFRACTION_FRAG,
  innerFrag:       INNER_CAUSTICS_FRAG,
  fresnelFrag:     FRESNEL_EDGE_FRAG,
  envReflectFrag:  ENV_REFLECT_FRAG,
  surfacePBRFrag:  SURFACE_PBR_FRAG,
  compositeFrag:   COMPOSITE_FRAG,
} as const;
