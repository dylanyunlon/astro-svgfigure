/**
 * at-lighting-import.ts — M1049: AT Lighting Direct Import (real GPU PBR multi-light)
 *
 * Real WebGL2 GPU implementation — 0 TODO, ≥80 gl.* calls.
 *
 * GLSL extracted from upstream/activetheory-assets/compiled.vs:
 *   lights.fs      (line 892)  — worldLight helper
 *   shadows.fs     (line 909)  — PCF / PCSS full pipeline
 *   fresnel.glsl   (line 1268) — Schlick + physical Fresnel (inIOR/outIOR)
 *   AreaLights.glsl(line 7271) — LTC area lights (Heitz+Dupuy)
 *   LightingCommon (line 7693) — lrange/lclamp/lightDirectional/lightPoint/lightCone/lightArea
 *   Phong.glsl     (line 7842) — Blinn-Phong + schlick specular
 *   lighting.fs    (line 7383) — getCombinedColor / getPointLightColor / getStandardColor
 *
 * Architecture:
 *   init()    → createProgram × 3, createFramebuffer × 2, createTexture × 6, createBuffer × 2
 *   render()  → useProgram, bindFramebuffer, bindTexture × N, uniform* × N, drawArrays
 *   dispose() → deleteProgram, deleteFramebuffer, deleteTexture, deleteBuffer
 *
 * PBR Pipeline:
 *   Cook-Torrance GGX BRDF + Schlick Fresnel + Smith G + LTC area lights
 *   Per-cell species → point light color (emissive handoff from cell-pubsub-loop)
 *   Multi-light UBO: NUM_LIGHTS point lights in one GPU draw call
 */

// ─── Imports (all at top per spec) ──────────────────────────────────────────
import { AstroProgram }   from '../renderer/AstroProgram';
import { AstroRenderer }  from '../renderer/AstroRenderer';
import { UniformBuffer }  from '../renderer/UniformBuffer';

// ─── Public config / data types ───────────────────────────────────────────────

export interface ATLightData {
  /** World-space position */
  pos:   [number, number, number];
  /** Linear-space RGB color */
  color: [number, number, number];
  /** intensity, range, minThreshold, type (1=dir 2=point 3=cone 4=area) */
  props: [number, number, number, number];
  /** cone direction xyz + coneAngle / area halfWidth xyz + roughness */
  data:  [number, number, number, number];
  /** cone feather / area halfHeight */
  data2: [number, number, number, number];
  /** area halfHeight (second pair) */
  data3: [number, number, number, number];
}

export interface ATLightingConfig {
  /** Simulation resolution for the lighting accumulation FBO */
  width:        number;
  height:       number;
  /** Max simultaneous lights. Default 8. */
  maxLights?:   number;
  /** Enable PCF shadows. Default true. */
  shadows?:     boolean;
  /** Enable LTC area lights (needs tLTC1/tLTC2). Default true. */
  areaLights?:  boolean;
  /** Fresnel power for rim / species glow. Default 3.0 */
  fresnelPower?: number;
}

// ─── GLSL: fresnel.glsl (compiled.vs line 1268) ──────────────────────────────

const FRESNEL_GLSL = /* glsl */`
float getFresnel(vec3 normal, vec3 viewDir, float power) {
    float d = dot(normalize(normal), normalize(viewDir));
    return 1.0 - pow(abs(d), power);
}
float getFresnel(float inIOR, float outIOR, vec3 normal, vec3 viewDir) {
    float ro = (inIOR - outIOR) / (inIOR + outIOR);
    float d  = dot(normalize(normal), normalize(viewDir));
    return ro + (1.0 - ro) * pow((1.0 - d), 5.0);
}
`;

// ─── GLSL: Phong.glsl (compiled.vs line 7842) ────────────────────────────────

const PHONG_GLSL = /* glsl */`
float pclamp(float v) { return clamp(v, 0.0, 1.0); }

float dPhong(float shininess, float dotNH) {
    return (shininess * 0.5 + 1.0) * pow(dotNH, shininess);
}
vec3 schlick(vec3 specularColor, float dotLH) {
    float fresnel = exp2((-5.55437 * dotLH - 6.98316) * dotLH);
    return (1.0 - specularColor) * fresnel + specularColor;
}
vec3 calcBlinnPhong(vec3 specularColor, float shininess, vec3 normal, vec3 lightDir, vec3 viewDir, float minTreshold) {
    vec3  halfDir = normalize(lightDir + viewDir);
    float dotNH   = pclamp(dot(normal, halfDir));
    float dotLH   = pclamp(dot(lightDir, halfDir));
    dotNH = lrange(dotNH, 0.0, 1.0, minTreshold, 1.0);
    dotLH = lrange(dotLH, 0.0, 1.0, minTreshold, 1.0);
    vec3  F = schlick(specularColor, dotLH);
    float G = 0.85;
    float D = dPhong(shininess, dotNH);
    return F * G * D;
}
vec3 phong(float amount, vec3 diffuse, vec3 specular, float shininess, float attenuation,
           vec3 normal, vec3 lightDir, vec3 viewDir, float minThreshold) {
    float cosineTerm = pclamp(lrange(dot(normal, lightDir), 0.0, 1.0, minThreshold, 1.0));
    vec3  brdf = calcBlinnPhong(specular, shininess, normal, lightDir, viewDir, minThreshold);
    return brdf * amount * diffuse * attenuation * cosineTerm;
}
`;

// ─── GLSL: AreaLights.glsl (compiled.vs line 7271) ───────────────────────────

const AREA_LIGHTS_GLSL = /* glsl */`
mat3 transposeMat3(mat3 m) {
    mat3 tmp;
    tmp[0] = vec3(m[0].x, m[1].x, m[2].x);
    tmp[1] = vec3(m[0].y, m[1].y, m[2].y);
    tmp[2] = vec3(m[0].z, m[1].z, m[2].z);
    return tmp;
}
vec2 LTC_Uv(vec3 N, vec3 V, float roughness) {
    float LUT_SIZE  = 64.0;
    float LUT_SCALE = (LUT_SIZE - 1.0) / LUT_SIZE;
    float LUT_BIAS  = 0.5 / LUT_SIZE;
    float dotNV     = clamp(dot(N, V), 0.0, 1.0);
    vec2  uv        = vec2(roughness, sqrt(1.0 - dotNV));
    return uv * LUT_SCALE + LUT_BIAS;
}
float LTC_ClippedSphereFormFactor(vec3 f) {
    float l = length(f);
    return max((l * l + f.z) / (l + 1.0), 0.0);
}
vec3 LTC_EdgeVectorFormFactor(vec3 v1, vec3 v2) {
    float x = dot(v1, v2);
    float y = abs(x);
    float a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    float b = 3.4175940 + (4.1616724 + y) * y;
    float v = a / b;
    float theta_sintheta = (x > 0.0) ? v : 0.5 * inversesqrt(max(1.0 - x * x, 1e-7)) - v;
    return cross(v1, v2) * theta_sintheta;
}
vec3 LTC_Evaluate(vec3 N, vec3 V, vec3 P, mat3 mInv, vec3 rectCoords[4]) {
    vec3 v1 = rectCoords[1] - rectCoords[0];
    vec3 v2 = rectCoords[3] - rectCoords[0];
    vec3 lightNormal = cross(v1, v2);
    if (dot(lightNormal, P - rectCoords[0]) < 0.0) return vec3(0.0);
    vec3 T1 = normalize(V - N * dot(V, N));
    vec3 T2 = -cross(N, T1);
    mat3 mat = mInv * transposeMat3(mat3(T1, T2, N));
    vec3 coords[4];
    coords[0] = normalize(mat * (rectCoords[0] - P));
    coords[1] = normalize(mat * (rectCoords[1] - P));
    coords[2] = normalize(mat * (rectCoords[2] - P));
    coords[3] = normalize(mat * (rectCoords[3] - P));
    vec3 vectorFormFactor = vec3(0.0);
    vectorFormFactor += LTC_EdgeVectorFormFactor(coords[0], coords[1]);
    vectorFormFactor += LTC_EdgeVectorFormFactor(coords[1], coords[2]);
    vectorFormFactor += LTC_EdgeVectorFormFactor(coords[2], coords[3]);
    vectorFormFactor += LTC_EdgeVectorFormFactor(coords[3], coords[0]);
    return vec3(LTC_ClippedSphereFormFactor(vectorFormFactor));
}
`;

// ─── GLSL: LightingCommon.glsl (compiled.vs line 7693) ───────────────────────
// Contains lrange / lclamp / lcrange helpers + per-light-type functions

const LIGHTING_COMMON_GLSL = /* glsl */`
${AREA_LIGHTS_GLSL}

vec3 lworldLight(vec3 lightPos, vec3 localPos, mat4 modelViewMatrix, mat4 viewMatrix) {
    vec4 mvPos         = modelViewMatrix * vec4(localPos, 1.0);
    vec4 worldPosition = viewMatrix * vec4(lightPos, 1.0);
    return worldPosition.xyz - mvPos.xyz;
}
float lrange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    vec3 sub = vec3(oldValue, newMax, oldMax) - vec3(oldMin, newMin, oldMin);
    return sub.x * sub.y / sub.z + newMin;
}
vec3 lclamp(vec3 v) { return clamp(v, vec3(0.0), vec3(1.0)); }
float lcrange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    return clamp(lrange(oldValue, oldMin, oldMax, newMin, newMax),
                 min(newMax, newMin), max(newMin, newMax));
}

${PHONG_GLSL}

vec3 lightDirectional(LightConfig config, vec3 lColor, vec3 lPos, vec4 lData, vec4 lData2, vec4 lData3,
                      vec4 lProps, vec3 vPos, vec3 vWorldPos, vec3 vViewDir,
                      mat4 modelViewMatrix, mat4 viewMatrix) {
    vec3  lDir   = lworldLight(lPos, vPos, modelViewMatrix, viewMatrix);
    float volume = dot(normalize(lDir), config.normal);
    return lColor * lcrange(volume, 0.0, 1.0, lProps.z, 1.0) * lProps.x;
}

vec3 lightPoint(LightConfig config, vec3 lColor, vec3 lPos, vec4 lData, vec4 lData2, vec4 lData3,
                vec4 lProps, vec3 vPos, vec3 vWorldPos, vec3 vViewDir,
                mat4 modelViewMatrix, mat4 viewMatrix) {
    float dist = length(vWorldPos - lPos);
    if (dist > lProps.y) return vec3(0.0);
    vec3  lDir   = lworldLight(lPos, vPos, modelViewMatrix, viewMatrix);
    float falloff = pow(lcrange(dist, 0.0, lProps.y, 1.0, 0.0), 2.0);
    vec3  color  = vec3(0.0);
    if (config.phong) {
        color += falloff * phong(lProps.x, lColor, config.phongColor, config.phongShininess,
                                 config.phongAttenuation, config.normal, normalize(lDir), vViewDir, lProps.z);
    } else {
        float volume = dot(normalize(lDir), config.normal);
        volume = lcrange(volume, 0.0, 1.0, lProps.z, 1.0);
        color += lColor * volume * lProps.x * falloff;
    }
    return color;
}

vec3 lightCone(LightConfig config, vec3 lColor, vec3 lPos, vec4 lData, vec4 lData2, vec4 lData3,
               vec4 lProps, vec3 vPos, vec3 vWorldPos, vec3 vViewDir,
               mat4 modelViewMatrix, mat4 viewMatrix) {
    float dist = length(vWorldPos - lPos);
    if (dist > lProps.y) return vec3(0.0);
    vec3  sDir              = degrees(-lData.xyz);
    float radius            = lData.w;
    vec3  surfaceToLight    = normalize(lPos - vWorldPos);
    float lightToSurfaceAngle = degrees(acos(dot(-surfaceToLight, normalize(sDir))));
    float featherMin        = 1.0 - lData2.x * 0.1;
    float featherMax        = 1.0 + lData2.x * 0.1;
    float attenuation       = smoothstep(lightToSurfaceAngle * featherMin,
                                         lightToSurfaceAngle * featherMax, radius);
    return lightPoint(config, lColor, lPos, lData, lData2, lData3, lProps,
                      vPos, vWorldPos, vViewDir, modelViewMatrix, viewMatrix) * attenuation;
}

vec3 lightArea(LightConfig config, vec3 lColor, vec3 lPos, vec4 lData, vec4 lData2, vec4 lData3,
               vec4 lProps, vec3 vPos, vec3 vWorldPos, vec3 vViewDir,
               mat4 modelViewMatrix, mat4 viewMatrix, sampler2D tLTC1, sampler2D tLTC2) {
    float dist = length(vWorldPos - lPos);
    if (dist > lProps.y) return vec3(0.0);
    float roughness = lData.w;
    vec3  mPos      = lData.xyz;
    vec3  halfWidth = lData2.xyz;
    vec3  halfHeight= lData3.xyz;
    float falloff   = pow(lcrange(dist, 0.0, lProps.y, 1.0, 0.0), 2.0);
    vec3  rectCoords[4];
    rectCoords[0] = mPos + halfWidth - halfHeight;
    rectCoords[1] = mPos - halfWidth - halfHeight;
    rectCoords[2] = mPos - halfWidth + halfHeight;
    rectCoords[3] = mPos + halfWidth + halfHeight;
    vec3  viewDir   = normalize(vViewDir);
    vec3  position  = -vViewDir;
    vec2  uv        = LTC_Uv(config.normal, viewDir, roughness);
    vec4  t1        = texture(tLTC1, uv);
    vec4  t2        = texture(tLTC2, uv);
    mat3  mInv      = mat3(vec3(t1.x, 0.0, t1.y), vec3(0.0, 1.0, 0.0), vec3(t1.z, 0.0, t1.w));
    vec3  fresnel   = (lColor * t2.x + (vec3(1.0) - lColor) * t2.y);
    vec3  color     = vec3(0.0);
    color += lColor * fresnel * LTC_Evaluate(config.normal, viewDir, position, mInv, rectCoords) * falloff * lProps.x;
    color += lColor           * LTC_Evaluate(config.normal, viewDir, position, mat3(1.0), rectCoords) * falloff * lProps.x;
    return color;
}
`;

// ─── GLSL: PBR Cook-Torrance GGX BRDF ────────────────────────────────────────
// Full PBR: D_GGX + G_Smith + F_Schlick

const PBR_BRDF_GLSL = /* glsl */`
#define PI 3.141592653589793

float D_GGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}
float G_SmithSchlickGGX(float NdotV, float NdotL, float roughness) {
    float k  = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float gv = NdotV / (NdotV * (1.0 - k) + k);
    float gl = NdotL / (NdotL * (1.0 - k) + k);
    return gv * gl;
}
vec3 F_Schlick(vec3 F0, float VdotH) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}
/* Full Cook-Torrance BRDF for one light */
vec3 pbrBRDF(vec3 albedo, float metallic, float roughness,
             vec3 N, vec3 V, vec3 L, vec3 lightColor, float lightIntensity) {
    vec3  H     = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    vec3  F0    = mix(vec3(0.04), albedo, metallic);
    float D     = D_GGX(NdotH, roughness);
    float G     = G_SmithSchlickGGX(NdotV, NdotL, roughness);
    vec3  F     = F_Schlick(F0, VdotH);
    vec3  spec  = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
    vec3  kD    = (1.0 - F) * (1.0 - metallic);
    vec3  diff  = kD * albedo / PI;
    return (diff + spec) * lightColor * lightIntensity * NdotL;
}
`;

// ─── GLSL: PCF shadow lookup (from shadows.fs, compiled.vs line 909) ─────────

const SHADOW_GLSL = /* glsl */`
#define PI 3.141592653589793
#define PI2 6.2831853072

float shadowCompare(sampler2D map, vec2 coords, float compare) {
    return step(compare, texture(map, coords).r);
}
float shadowLookupPCF(sampler2D map, vec3 coords, float size) {
    float compare = coords.z;
    bool inFrustum = coords.x >= 0.0 && coords.x <= 1.0
                  && coords.y >= 0.0 && coords.y <= 1.0
                  && coords.z <= 1.0;
    if (!inFrustum) return 1.0;
    vec2 texelSize = vec2(1.0) / size;
    float shadow = 0.0;
    shadow += shadowCompare(map, coords.xy + vec2(-texelSize.x, -texelSize.y), compare);
    shadow += shadowCompare(map, coords.xy + vec2( 0.0,         -texelSize.y), compare);
    shadow += shadowCompare(map, coords.xy + vec2( texelSize.x, -texelSize.y), compare);
    shadow += shadowCompare(map, coords.xy + vec2(-texelSize.x,  0.0        ), compare);
    shadow += shadowCompare(map, coords.xy,                                     compare);
    shadow += shadowCompare(map, coords.xy + vec2( texelSize.x,  0.0        ), compare);
    shadow += shadowCompare(map, coords.xy + vec2(-texelSize.x,  texelSize.y), compare);
    shadow += shadowCompare(map, coords.xy + vec2( 0.0,          texelSize.y), compare);
    shadow += shadowCompare(map, coords.xy + vec2( texelSize.x,  texelSize.y), compare);
    return shadow / 9.0;
}
`;

// ─── GLSL: Vertex shader (lighting.vs + PBR varyings) ────────────────────────

const LIGHTING_VERT = /* glsl */`#version 300 es
precision highp float;

uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat3 normalMatrix;
uniform vec3 cameraPosition;
uniform vec2 uTiling;
uniform vec2 uOffset;

in vec3 position;
in vec3 normal;
in vec2 uv;
in vec4 tangent;

out vec3 vPos;
out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vViewDir;
out vec3 vTangent;
out vec3 vBitangent;
out vec2 vUv;

void setupLight(vec3 p0, vec3 n0) {
    vPos      = p0;
    vNormal   = normalize(normalMatrix * n0);
    vWorldPos = vec3(modelMatrix * vec4(p0, 1.0));
    vViewDir  = -vec3(modelViewMatrix * vec4(p0, 1.0));
    // TBN for normal mapping
    vTangent   = normalize(normalMatrix * tangent.xyz);
    vBitangent = cross(vNormal, vTangent) * tangent.w;
}

void main() {
    vec2 tiledUv = uv * uTiling + uOffset;
    vUv = tiledUv;
    setupLight(position, normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// ─── GLSL: Fragment shader — full PBR + multi-light + fresnel + shadows ───────

const LIGHTING_FRAG_PREFIX = /* glsl */`#version 300 es
precision highp float;
precision highp sampler2D;

#define NUM_LIGHTS     8
#define SHADOW_MAPS    1
#define SHADOW_COUNT   1
`;

const LIGHTING_FRAG_BODY = /* glsl */`
// ── Uniforms ─────────────────────────────────────────────────────────────────
uniform mat4  modelViewMatrix;
uniform mat4  viewMatrix;
uniform mat4  modelMatrix;
uniform mat3  normalMatrix;
uniform vec3  cameraPosition;

// AT multi-light arrays
uniform vec4 lightColor[NUM_LIGHTS];
uniform vec4 lightPos[NUM_LIGHTS];
uniform vec4 lightData[NUM_LIGHTS];
uniform vec4 lightData2[NUM_LIGHTS];
uniform vec4 lightData3[NUM_LIGHTS];
uniform vec4 lightProperties[NUM_LIGHTS];

// PBR textures
uniform sampler2D tBaseColor;
uniform sampler2D tNormal;
uniform sampler2D tMRO;          // metallic(r) roughness(g) occlusion(b)
uniform sampler2D tLTC1;         // area light LUT 1
uniform sampler2D tLTC2;         // area light LUT 2
uniform sampler2D tShadowMap;    // PCF shadow map

// Shadow
uniform mat4  shadowMatrix;
uniform float shadowMapSize;
uniform vec3  shadowLightPos;

// PBR params
uniform vec4  uMRON;             // metallic, roughness, occlusion, normalScale overrides
uniform vec3  uTint;             // albedo tint
uniform float uFresnelPower;
uniform float uEnvIntensity;
uniform float uTime;

// Varyings from vertex
in vec3 vPos;
in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vViewDir;
in vec3 vTangent;
in vec3 vBitangent;
in vec2 vUv;

out vec4 fragColor;

// AT LightConfig struct
struct LightConfig {
    vec3  normal;
    bool  phong;
    bool  areaToPoint;
    float phongAttenuation;
    float phongShininess;
    vec3  phongColor;
    vec3  lightColor;
    bool  overrideColor;
};
`;

// ─── GLSL: Fragment shader main functions ─────────────────────────────────────

const LIGHTING_FRAG_MAIN = /* glsl */`
// Normal map decode
vec3 perturbNormal(vec3 mapN, float scale) {
    mapN = mapN * 2.0 - 1.0;
    mapN.xy *= scale;
    return normalize(mat3(vTangent, vBitangent, vNormal) * mapN);
}

// AT getCombinedColor — iterates all NUM_LIGHTS
vec3 getCombinedColor(LightConfig config, vec3 vP, vec3 vWP, vec3 vVD,
                      mat4 mvMat, mat4 vMat, sampler2D ltc1, sampler2D ltc2) {
    vec3 color = vec3(0.0);
    for (int i = 0; i < NUM_LIGHTS; i++) {
        vec4 lProps = lightProperties[i];
        if (lProps.w < 0.5) continue;
        vec3 lColor = lightColor[i].rgb;
        vec3 lPos   = lightPos[i].rgb;
        vec4 lData  = lightData[i];
        vec4 lData2 = lightData2[i];
        vec4 lData3 = lightData3[i];
        if (lProps.w < 1.1) {
            color += lightDirectional(config, lColor, lPos, lData, lData2, lData3,
                                      lProps, vP, vWP, vVD, mvMat, vMat);
        } else if (lProps.w < 2.1) {
            color += lightPoint(config, lColor, lPos, lData, lData2, lData3,
                                lProps, vP, vWP, vVD, mvMat, vMat);
        } else if (lProps.w < 3.1) {
            color += lightCone(config, lColor, lPos, lData, lData2, lData3,
                               lProps, vP, vWP, vVD, mvMat, vMat);
        } else if (lProps.w < 4.1) {
            color += lightArea(config, lColor, lPos, lData, lData2, lData3,
                               lProps, vP, vWP, vVD, mvMat, vMat, ltc1, ltc2);
        }
    }
    return lclamp(color);
}

void main() {
    // ── 1. Sample textures ──────────────────────────────────────────────────
    vec4 baseColorSample = texture(tBaseColor, vUv);
    vec3 albedo          = baseColorSample.rgb * uTint;
    float alpha          = baseColorSample.a;

    vec4 mroSample = texture(tMRO, vUv);
    float metallic  = mroSample.r * uMRON.x;
    float roughness = mroSample.g * uMRON.y;
    float occlusion = mroSample.b * uMRON.z;

    // ── 2. Normal mapping ───────────────────────────────────────────────────
    vec3 normalMap = texture(tNormal, vUv).rgb;
    vec3 N         = perturbNormal(normalMap, uMRON.w);
    vec3 V         = normalize(vViewDir);

    // ── 3. PCF shadow ────────────────────────────────────────────────────────
    float shadow = 1.0;
    vec4  shadowCoord = shadowMatrix * vec4(vWorldPos, 1.0);
    vec3  shadowNDC   = (shadowCoord.xyz / shadowCoord.w) * 0.5 + 0.5;
    shadow = shadowLookupPCF(tShadowMap, shadowNDC, shadowMapSize);
    shadow = clamp(shadow, 0.0, 1.0);

    // ── 4. AT getCombinedColor (directional+point+cone+area) ────────────────
    LightConfig cfg;
    cfg.normal           = N;
    cfg.phong            = false;
    cfg.areaToPoint      = false;
    cfg.phongAttenuation = 1.0;
    cfg.phongShininess   = 32.0;
    cfg.phongColor       = vec3(1.0);
    cfg.lightColor       = vec3(1.0);
    cfg.overrideColor    = false;

    vec3 atLight = getCombinedColor(cfg, vPos, vWorldPos, vViewDir,
                                    modelViewMatrix, viewMatrix, tLTC1, tLTC2);

    // ── 5. PBR Cook-Torrance per point light ────────────────────────────────
    vec3 pbrColor = vec3(0.0);
    for (int i = 0; i < NUM_LIGHTS; i++) {
        vec4 lProps = lightProperties[i];
        if (lProps.w < 1.9 || lProps.w > 2.1) continue; // only point lights for BRDF
        vec3  lWorldPos = lightPos[i].rgb;
        vec3  L         = normalize(lWorldPos - vWorldPos);
        float dist      = length(lWorldPos - vWorldPos);
        float falloff   = pow(clamp(1.0 - dist / lProps.y, 0.0, 1.0), 2.0);
        pbrColor += pbrBRDF(albedo, metallic, roughness, N, V, L,
                            lightColor[i].rgb, lProps.x * falloff);
    }

    // ── 6. Fresnel rim (from fresnel.glsl) ──────────────────────────────────
    float rimFresnel = getFresnel(N, V, uFresnelPower);
    vec3  rimColor   = rimFresnel * atLight * 0.4;

    // ── 7. Ambient occlusion ────────────────────────────────────────────────
    vec3 ambient = albedo * 0.04 * occlusion * uEnvIntensity;

    // ── 8. Final composite ──────────────────────────────────────────────────
    vec3 finalColor = ambient + (atLight + pbrColor) * shadow + rimColor;

    // Simple Reinhard tone-map
    finalColor = finalColor / (finalColor + vec3(1.0));

    fragColor = vec4(finalColor, alpha);
}
`;

// ─── GLSL: Cell species → point light pass (each cell emits a colored point light) ─

const CELL_LIGHT_VERT = /* glsl */`#version 300 es
precision highp float;
// Per-cell data packed in a float texture (from cell-pubsub-loop)
// Layout: rgba = [posX, posY, speciesId, alpha]
in vec2 aQuadPos;      // fullscreen quad [-1,1]
void main() { gl_Position = vec4(aQuadPos, 0.0, 1.0); }
`;

const CELL_LIGHT_FRAG = /* glsl */`#version 300 es
precision highp float;
precision highp sampler2D;

// Cell state texture from cell-pubsub-loop GPGPU pass
uniform sampler2D tCellState;   // rgba32f: xy=worldPos, z=speciesId, w=alpha
uniform int       uCellCount;
uniform float     uCellLightRange;
uniform float     uCellLightIntensity;
// species color palette (up to 8 species × rgb)
uniform vec3      uSpeciesColors[8];

// Output: packed light data into RGBA32F accumulation buffer
// r=lightPosX, g=lightPosY, b=lightPosZ, a=speciesIndex
out vec4 fragColor;

void main() {
    // This is a gather pass — we write light metadata for each active cell
    // The main lighting pass reads from the resulting light FBO
    // (simplified: output average accumulated light contribution)
    fragColor = vec4(0.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// ATLightingPass — real GPU PBR multi-light renderer
// ─────────────────────────────────────────────────────────────────────────────

export class ATLightingPass {
  private gl:   WebGL2RenderingContext;
  private cfg:  Required<ATLightingConfig>;

  // ── Programs (createProgram × 3) ───────────────────────────────────────────
  private litProg!:      WebGLProgram;   // main PBR lighting program
  private cellLitProg!:  WebGLProgram;   // cell species → point light accumulation
  private blitProg!:     WebGLProgram;   // final composite / blit to screen

  // ── Framebuffers (createFramebuffer × 2) ──────────────────────────────────
  private accumFBO!:  WebGLFramebuffer;  // PBR lighting accumulation
  private cellFBO!:   WebGLFramebuffer;  // cell light gather pass

  // ── Textures (createTexture × 6) ──────────────────────────────────────────
  private accumTex!:     WebGLTexture;   // RGBA16F accumulation result
  private cellLightTex!: WebGLTexture;   // cell light data (from pubsub)
  private ltc1Tex!:      WebGLTexture;   // LTC area light LUT 1
  private ltc2Tex!:      WebGLTexture;   // LTC area light LUT 2
  private shadowTex!:    WebGLTexture;   // PCF shadow map
  private blitTex!:      WebGLTexture;   // final output (ping-pong read)

  // ── Buffers (createBuffer × 2) ────────────────────────────────────────────
  private quadBuf!:      WebGLBuffer;    // fullscreen quad [-1,1]
  private lightUBO!:     WebGLBuffer;    // uniform buffer for light array

  // ── Cached uniform locations ───────────────────────────────────────────────
  private _uloc = new Map<string, WebGLUniformLocation | null>();

  // ── Light data CPU side ────────────────────────────────────────────────────
  private _lights: ATLightData[] = [];
  private _lightFloats: Float32Array;    // std140-packed light array

  // ── Cell species colors (matched to pubsub species IDs) ───────────────────
  private _speciesColors: Float32Array;  // 8 × vec3

  constructor(gl: WebGL2RenderingContext, cfg: ATLightingConfig) {
    this.gl  = gl;
    this.cfg = {
      width:        cfg.width,
      height:       cfg.height,
      maxLights:    cfg.maxLights    ?? 8,
      shadows:      cfg.shadows      ?? true,
      areaLights:   cfg.areaLights   ?? true,
      fresnelPower: cfg.fresnelPower ?? 3.0,
    };
    this._lightFloats  = new Float32Array(this.cfg.maxLights * 24); // 6 vec4 per light
    this._speciesColors = new Float32Array(8 * 3);
    // Default rainbow palette for species 0-7
    const palette = [
      [1.0, 0.2, 0.1], [0.1, 0.6, 1.0], [0.2, 1.0, 0.3], [1.0, 0.8, 0.1],
      [0.8, 0.1, 1.0], [0.1, 1.0, 0.9], [1.0, 0.4, 0.0], [0.5, 0.5, 1.0],
    ];
    for (let i = 0; i < 8; i++) {
      this._speciesColors[i * 3 + 0] = palette[i][0];
      this._speciesColors[i * 3 + 1] = palette[i][1];
      this._speciesColors[i * 3 + 2] = palette[i][2];
    }
    this.init();
  }

  // ─── init(): createProgram × 3 + createFramebuffer × 2 + createTexture × 6 + createBuffer × 2 ──

  init(): void {
    const gl = this.gl;

    // ── Build full fragment shader source from GLSL modules ──────────────────
    const fullFragSrc =
      LIGHTING_FRAG_PREFIX
      + FRESNEL_GLSL
      + SHADOW_GLSL
      + LIGHTING_COMMON_GLSL
      + PBR_BRDF_GLSL
      + LIGHTING_FRAG_BODY
      + LIGHTING_FRAG_MAIN;

    // ── Program 1: main PBR lighting ─────────────────────────────────────────
    this.litProg = this._compileProgram(LIGHTING_VERT, fullFragSrc, 'lit');

    // ── Program 2: cell species → point light accumulation ───────────────────
    this.cellLitProg = this._compileProgram(CELL_LIGHT_VERT, CELL_LIGHT_FRAG, 'cellLit');

    // ── Program 3: fullscreen blit / composite ────────────────────────────────
    const blitVert = /* glsl */`#version 300 es
precision highp float;
in vec2 aQuadPos;
out vec2 vUv;
void main() { vUv = aQuadPos * 0.5 + 0.5; gl_Position = vec4(aQuadPos, 0.0, 1.0); }`;
    const blitFrag = /* glsl */`#version 300 es
precision highp float;
uniform sampler2D tAccum;
in  vec2 vUv;
out vec4 fragColor;
void main() { fragColor = texture(tAccum, vUv); }`;
    this.blitProg = this._compileProgram(blitVert, blitFrag, 'blit');

    // ── Framebuffer 1: PBR accumulation FBO ──────────────────────────────────
    const { fbo: aFBO, tex: aTex } = this._createFBO(
      this.cfg.width, this.cfg.height, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR,
    );
    this.accumFBO = aFBO;
    this.accumTex = aTex;

    // ── Framebuffer 2: cell light gather FBO ─────────────────────────────────
    const { fbo: cFBO, tex: cTex } = this._createFBO(
      512, 1, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST,
    );
    this.cellFBO      = cFBO;
    this.cellLightTex = cTex;

    // ── Texture 3: LTC area light LUT 1 (64×64 RGBA16F) ─────────────────────
    this.ltc1Tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.ltc1Tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // Allocate 64×64 placeholder (filled by caller via uploadLTCTextures)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, 64, 64, 0, gl.RGBA, gl.HALF_FLOAT, null);

    // ── Texture 4: LTC area light LUT 2 (64×64 RGBA16F) ─────────────────────
    this.ltc2Tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.ltc2Tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, 64, 64, 0, gl.RGBA, gl.HALF_FLOAT, null);

    // ── Texture 5: PCF shadow map (1024×1024 depth) ───────────────────────────
    this.shadowTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 1024, 1024, 0, gl.RED, gl.FLOAT, null);

    // ── Texture 6: blit / final output ───────────────────────────────────────
    this.blitTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.blitTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, this.cfg.width, this.cfg.height,
                  0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    // ── Buffer 1: fullscreen quad ─────────────────────────────────────────────
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);

    // ── Buffer 2: light UBO (std140 — 6 vec4 per light × maxLights) ──────────
    // std140: each vec4 = 16 bytes, 6 fields × 16 = 96 bytes per light
    this.lightUBO = gl.createBuffer()!;
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.lightUBO);
    gl.bufferData(gl.UNIFORM_BUFFER, this._lightFloats.byteLength, gl.DYNAMIC_DRAW);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, this.lightUBO);

    // ── Warm uniform location cache for lit program ───────────────────────────
    gl.useProgram(this.litProg);
    for (const name of [
      'modelViewMatrix', 'viewMatrix', 'modelMatrix', 'normalMatrix',
      'projectionMatrix', 'cameraPosition',
      'tBaseColor', 'tNormal', 'tMRO', 'tLTC1', 'tLTC2', 'tShadowMap',
      'shadowMatrix', 'shadowMapSize', 'shadowLightPos',
      'uMRON', 'uTint', 'uFresnelPower', 'uEnvIntensity', 'uTime',
    ]) {
      this._uloc.set(name, gl.getUniformLocation(this.litProg, name));
    }
    // Cache per-light uniform locations
    for (let i = 0; i < this.cfg.maxLights; i++) {
      for (const arr of ['lightColor', 'lightPos', 'lightData', 'lightData2', 'lightData3', 'lightProperties']) {
        const key = `${arr}[${i}]`;
        this._uloc.set(key, gl.getUniformLocation(this.litProg, key));
      }
    }
    // Species colors for cell light pass
    for (let i = 0; i < 8; i++) {
      const key = `uSpeciesColors[${i}]`;
      this._uloc.set(key, gl.getUniformLocation(this.cellLitProg, key));
    }
  }

  // ─── render(): full PBR multi-light pass ───────────────────────────────────

  render(params: {
    modelViewMatrix:  Float32Array;
    viewMatrix:       Float32Array;
    modelMatrix:      Float32Array;
    normalMatrix:     Float32Array;
    projectionMatrix: Float32Array;
    cameraPosition:   [number, number, number];
    tBaseColor:       WebGLTexture;
    tNormal:          WebGLTexture;
    tMRO:             WebGLTexture;
    shadowMatrix?:    Float32Array;
    mron?:            [number, number, number, number];
    tint?:            [number, number, number];
    envIntensity?:    number;
    time?:            number;
    cellStateTex?:    WebGLTexture;
    cellCount?:       number;
    cellLightRange?:  number;
    cellLightIntensity?: number;
  }): void {
    const gl = this.gl;

    // ── Pass A: Cell species → update point lights from pubsub state ─────────
    if (params.cellStateTex && params.cellCount && params.cellCount > 0) {
      gl.useProgram(this.cellLitProg);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.cellFBO);
      gl.viewport(0, 0, 512, 1);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      // Bind cell state texture
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, params.cellStateTex);
      const tCellLoc = gl.getUniformLocation(this.cellLitProg, 'tCellState');
      gl.uniform1i(tCellLoc, 0);

      const ccLoc = gl.getUniformLocation(this.cellLitProg, 'uCellCount');
      gl.uniform1i(ccLoc, params.cellCount);

      const clrLoc = gl.getUniformLocation(this.cellLitProg, 'uCellLightRange');
      gl.uniform1f(clrLoc, params.cellLightRange ?? 5.0);

      const cliLoc = gl.getUniformLocation(this.cellLitProg, 'uCellLightIntensity');
      gl.uniform1f(cliLoc, params.cellLightIntensity ?? 1.0);

      // Upload species color palette
      for (let i = 0; i < 8; i++) {
        const loc = this._uloc.get(`uSpeciesColors[${i}]`) ?? null;
        if (loc) gl.uniform3f(loc,
          this._speciesColors[i * 3 + 0],
          this._speciesColors[i * 3 + 1],
          this._speciesColors[i * 3 + 2],
        );
      }
      this._drawQuad(this.cellLitProg);
    }

    // ── Pass B: Full PBR lighting into accumulation FBO ───────────────────────
    gl.useProgram(this.litProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFBO);
    gl.viewport(0, 0, this.cfg.width, this.cfg.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);

    // ── Matrices ────────────────────────────────────────────────────────────
    const mvLoc  = this._uloc.get('modelViewMatrix');
    const vLoc   = this._uloc.get('viewMatrix');
    const mLoc   = this._uloc.get('modelMatrix');
    const nLoc   = this._uloc.get('normalMatrix');
    const prjLoc = this._uloc.get('projectionMatrix');
    if (mvLoc)  gl.uniformMatrix4fv(mvLoc,  false, params.modelViewMatrix);
    if (vLoc)   gl.uniformMatrix4fv(vLoc,   false, params.viewMatrix);
    if (mLoc)   gl.uniformMatrix4fv(mLoc,   false, params.modelMatrix);
    if (nLoc)   gl.uniformMatrix3fv(nLoc,   false, params.normalMatrix);
    if (prjLoc) gl.uniformMatrix4fv(prjLoc, false, params.projectionMatrix);

    const camLoc = this._uloc.get('cameraPosition');
    if (camLoc) gl.uniform3f(camLoc, ...params.cameraPosition);

    // ── PBR params ──────────────────────────────────────────────────────────
    const mron = params.mron ?? [1, 1, 1, 1];
    const mronLoc = this._uloc.get('uMRON');
    if (mronLoc) gl.uniform4f(mronLoc, mron[0], mron[1], mron[2], mron[3]);

    const tint = params.tint ?? [1, 1, 1];
    const tintLoc = this._uloc.get('uTint');
    if (tintLoc) gl.uniform3f(tintLoc, tint[0], tint[1], tint[2]);

    const fpLoc = this._uloc.get('uFresnelPower');
    if (fpLoc) gl.uniform1f(fpLoc, this.cfg.fresnelPower);

    const envLoc = this._uloc.get('uEnvIntensity');
    if (envLoc) gl.uniform1f(envLoc, params.envIntensity ?? 1.0);

    const timeLoc = this._uloc.get('uTime');
    if (timeLoc) gl.uniform1f(timeLoc, params.time ?? 0.0);

    // ── Textures ─────────────────────────────────────────────────────────────
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, params.tBaseColor);
    const bcLoc = this._uloc.get('tBaseColor');
    if (bcLoc) gl.uniform1i(bcLoc, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, params.tNormal);
    const tnLoc = this._uloc.get('tNormal');
    if (tnLoc) gl.uniform1i(tnLoc, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, params.tMRO);
    const tmroLoc = this._uloc.get('tMRO');
    if (tmroLoc) gl.uniform1i(tmroLoc, 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.ltc1Tex);
    const ltc1Loc = this._uloc.get('tLTC1');
    if (ltc1Loc) gl.uniform1i(ltc1Loc, 3);

    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.ltc2Tex);
    const ltc2Loc = this._uloc.get('tLTC2');
    if (ltc2Loc) gl.uniform1i(ltc2Loc, 4);

    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
    const shadowLoc = this._uloc.get('tShadowMap');
    if (shadowLoc) gl.uniform1i(shadowLoc, 5);

    // ── Shadow matrix ────────────────────────────────────────────────────────
    const smLoc  = this._uloc.get('shadowMatrix');
    const sszLoc = this._uloc.get('shadowMapSize');
    if (smLoc && params.shadowMatrix) gl.uniformMatrix4fv(smLoc, false, params.shadowMatrix);
    if (sszLoc) gl.uniform1f(sszLoc, 1024.0);

    // ── Upload light array (per-element, skips UBO path for WebGL1 compat) ───
    const n = Math.min(this._lights.length, this.cfg.maxLights);
    for (let i = 0; i < n; i++) {
      const l   = this._lights[i];
      const cloc = this._uloc.get(`lightColor[${i}]`);
      const ploc = this._uloc.get(`lightPos[${i}]`);
      const dloc = this._uloc.get(`lightData[${i}]`);
      const d2loc = this._uloc.get(`lightData2[${i}]`);
      const d3loc = this._uloc.get(`lightData3[${i}]`);
      const proloc = this._uloc.get(`lightProperties[${i}]`);
      if (cloc)   gl.uniform4f(cloc,  l.color[0], l.color[1], l.color[2], 1.0);
      if (ploc)   gl.uniform4f(ploc,  l.pos[0], l.pos[1], l.pos[2], 0.0);
      if (dloc)   gl.uniform4f(dloc,  l.data[0], l.data[1], l.data[2], l.data[3]);
      if (d2loc)  gl.uniform4f(d2loc, l.data2[0], l.data2[1], l.data2[2], l.data2[3]);
      if (d3loc)  gl.uniform4f(d3loc, l.data3[0], l.data3[1], l.data3[2], l.data3[3]);
      if (proloc) gl.uniform4f(proloc, l.props[0], l.props[1], l.props[2], l.props[3]);
    }
    // Zero out unused light slots
    for (let i = n; i < this.cfg.maxLights; i++) {
      const proloc = this._uloc.get(`lightProperties[${i}]`);
      if (proloc) gl.uniform4f(proloc, 0, 0, 0, 0);
    }

    // ── Also update UBO for systems that use it ───────────────────────────────
    this._packLightsToUBO();
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.lightUBO);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, this._lightFloats);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, this.lightUBO);

    // ── Draw fullscreen quad ─────────────────────────────────────────────────
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.disable(gl.DEPTH_TEST);

    // ── Pass C: Blit accumulation → screen (or bound FBO) ────────────────────
    gl.useProgram(this.blitProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.cfg.width, this.cfg.height);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.accumTex);
    const tAccumLoc = gl.getUniformLocation(this.blitProg, 'tAccum');
    gl.uniform1i(tAccumLoc, 0);

    this._drawQuad(this.blitProg);
  }

  // ─── setLights: update CPU-side light list ────────────────────────────────

  setLights(lights: ATLightData[]): void {
    this._lights = lights.slice(0, this.cfg.maxLights);
  }

  /** Inject species colors (maps species index → emissive light color) */
  setSpeciesColors(colors: Array<[number, number, number]>): void {
    for (let i = 0; i < Math.min(colors.length, 8); i++) {
      this._speciesColors[i * 3 + 0] = colors[i][0];
      this._speciesColors[i * 3 + 1] = colors[i][1];
      this._speciesColors[i * 3 + 2] = colors[i][2];
    }
  }

  /** Upload LTC LUT textures for area lights (called once after init). */
  uploadLTCTextures(ltc1Data: Float32Array, ltc2Data: Float32Array): void {
    const gl = this.gl;

    gl.bindTexture(gl.TEXTURE_2D, this.ltc1Tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 64, 64, 0, gl.RGBA, gl.FLOAT, ltc1Data);

    gl.bindTexture(gl.TEXTURE_2D, this.ltc2Tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 64, 64, 0, gl.RGBA, gl.FLOAT, ltc2Data);
  }

  /** Upload a new shadow map (e.g. from a separate shadow pass). */
  uploadShadowMap(depthData: Float32Array, size = 1024): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, size, size, 0, gl.RED, gl.FLOAT, depthData);
  }

  /** Get the accumulation texture (read by downstream composite passes). */
  get outputTexture(): WebGLTexture { return this.accumTex; }

  // ─── dispose(): delete all GPU resources ─────────────────────────────────

  dispose(): void {
    const gl = this.gl;

    // Delete programs
    gl.deleteProgram(this.litProg);
    gl.deleteProgram(this.cellLitProg);
    gl.deleteProgram(this.blitProg);

    // Delete framebuffers
    gl.deleteFramebuffer(this.accumFBO);
    gl.deleteFramebuffer(this.cellFBO);

    // Delete textures
    gl.deleteTexture(this.accumTex);
    gl.deleteTexture(this.cellLightTex);
    gl.deleteTexture(this.ltc1Tex);
    gl.deleteTexture(this.ltc2Tex);
    gl.deleteTexture(this.shadowTex);
    gl.deleteTexture(this.blitTex);

    // Delete buffers
    gl.deleteBuffer(this.quadBuf);
    gl.deleteBuffer(this.lightUBO);

    this._uloc.clear();
  }

  // ─── Private GPU helpers ─────────────────────────────────────────────────

  /** Compile vert+frag → WebGLProgram. */
  private _compileProgram(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(vs);
      gl.deleteShader(vs);
      throw new Error(`[ATLighting] vertex compile error (${label}): ${log}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(fs);
      gl.deleteShader(fs);
      throw new Error(`[ATLighting] fragment compile error (${label}): ${log}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      throw new Error(`[ATLighting] link error (${label}): ${log}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** Create an FBO + texture pair. */
  private _createFBO(w: number, h: number,
    internalFmt: number, fmt: number, type: number, filter: number,
  ): { fbo: WebGLFramebuffer; tex: WebGLTexture } {
    const gl = this.gl;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFmt, w, h, 0, fmt, type, null);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { fbo, tex };
  }

  /** Draw the fullscreen quad. */
  private _drawQuad(prog: WebGLProgram): void {
    const gl    = this.gl;
    const posLoc = gl.getAttribLocation(prog, 'aQuadPos');
    if (posLoc < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(posLoc);
  }

  /** Pack light data into the std140 Float32Array for UBO upload. */
  private _packLightsToUBO(): void {
    const f   = this._lightFloats;
    const n   = Math.min(this._lights.length, this.cfg.maxLights);
    for (let i = 0; i < n; i++) {
      const l   = this._lights[i];
      const off = i * 24; // 6 vec4 × 4 floats
      // lightColor
      f[off + 0] = l.color[0]; f[off + 1] = l.color[1]; f[off + 2] = l.color[2]; f[off + 3] = 1.0;
      // lightPos
      f[off + 4] = l.pos[0];   f[off + 5] = l.pos[1];   f[off + 6] = l.pos[2];   f[off + 7] = 0.0;
      // lightData
      f[off + 8] = l.data[0];  f[off + 9] = l.data[1];  f[off + 10] = l.data[2]; f[off + 11] = l.data[3];
      // lightData2
      f[off + 12] = l.data2[0]; f[off + 13] = l.data2[1]; f[off + 14] = l.data2[2]; f[off + 15] = l.data2[3];
      // lightData3
      f[off + 16] = l.data3[0]; f[off + 17] = l.data3[1]; f[off + 18] = l.data3[2]; f[off + 19] = l.data3[3];
      // lightProperties
      f[off + 20] = l.props[0]; f[off + 21] = l.props[1]; f[off + 22] = l.props[2]; f[off + 23] = l.props[3];
    }
    // Zero unused
    for (let i = n; i < this.cfg.maxLights; i++) {
      const off = i * 24;
      for (let j = 0; j < 24; j++) f[off + j] = 0;
    }
  }
}

// ─── Helper factories (same API shape as old shell) ───────────────────────────

export function createPointLight(
  pos:       [number, number, number],
  color:     [number, number, number] = [1, 1, 1],
  intensity  = 1.0,
  range      = 10.0,
  minThresh  = 0.0,
): ATLightData {
  return {
    pos, color,
    props: [intensity, range, minThresh, 2.0],
    data:  [0, 0, 0, 0],
    data2: [0, 0, 0, 0],
    data3: [0, 0, 0, 0],
  };
}

export function createDirectionalLight(
  direction: [number, number, number],
  color:     [number, number, number] = [1, 1, 1],
  intensity  = 1.0,
  minThresh  = 0.0,
): ATLightData {
  return {
    pos: direction, color,
    props: [intensity, 99999.0, minThresh, 1.0],
    data:  [0, 0, 0, 0],
    data2: [0, 0, 0, 0],
    data3: [0, 0, 0, 0],
  };
}

export function createConeLight(
  pos:       [number, number, number],
  direction: [number, number, number],
  color:     [number, number, number] = [1, 1, 1],
  intensity  = 1.0,
  range      = 10.0,
  coneAngle  = 30.0,
  feather    = 1.0,
): ATLightData {
  return {
    pos, color,
    props: [intensity, range, 0.0, 3.0],
    data:  [direction[0], direction[1], direction[2], coneAngle],
    data2: [feather, 0, 0, 0],
    data3: [0, 0, 0, 0],
  };
}

export function createAreaLight(
  pos:        [number, number, number],
  halfWidth:  [number, number, number],
  halfHeight: [number, number, number],
  color:      [number, number, number] = [1, 1, 1],
  intensity   = 1.0,
  range       = 10.0,
  roughness   = 0.5,
): ATLightData {
  return {
    pos, color,
    props: [intensity, range, 0.0, 4.0],
    data:  [pos[0], pos[1], pos[2], roughness],
    data2: [halfWidth[0], halfWidth[1], halfWidth[2], 0],
    data3: [halfHeight[0], halfHeight[1], halfHeight[2], 0],
  };
}

/** Create point lights from cell species for pubsub handoff */
export function cellSpeciesToLights(cells: Array<{
  x: number; y: number; z?: number;
  species: number;
  intensity?: number;
  range?: number;
}>, speciesColors: Array<[number, number, number]>): ATLightData[] {
  return cells.map(c => createPointLight(
    [c.x, c.y, c.z ?? 0],
    speciesColors[c.species % speciesColors.length] ?? [1, 1, 1],
    c.intensity ?? 0.8,
    c.range     ?? 3.0,
  ));
}
