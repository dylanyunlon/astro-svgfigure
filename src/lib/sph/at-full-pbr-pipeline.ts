/**
 * at-full-pbr-pipeline.ts — M827: AT Full PBR Pipeline
 *
 * 完整 ActiveTheory PBR 材质管线 WebGL 实现。整合：
 *   lighting.fs  (LightingCommon / AreaLights / Phong) → 4种光源 + LTC面光源
 *   ShadowDepth.fs → PCSS / PCF soft shadows
 *   BasicMirror.fs → planar mirror reflection
 *   matcap*.ktx2 → matcap layer (screen-space normal UV)
 *   env1.ktx2    → IBL equirectangular HDR (RGBM)
 *
 * Cook-Torrance BRDF: GGX NDF + Smith G + Schlick F + IBL + matcap + shadow + mirror
 * GLSL 全部内联 (无外部 #require 依赖)
 *
 * 参考: at-lighting-import.ts (M804) · at-shadow-import.ts (M805) · at-pbr-material.ts (M712)
 *
 * 用法:
 *   const pipe = ATFullPBRPipeline.create(gl, { maxLights: 4, enablePCSS: true });
 *   pipe.setMaterial({ albedo: [0.8,0.6,0.4], roughness: 0.3 });
 *   pipe.drawMesh(vao, ibo, count, matrices, textures, lights, shadows);
 *
 * Research: xiaodi #M827 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Types & Interfaces
// ─────────────────────────────────────────────────────────────────────────────

/** Light type IDs matching AT's lightProperties.w convention. */
export type ATLightType = 1 | 2 | 3 | 4;

/**
 * Single light descriptor — maps directly to AT's per-light uniform arrays.
 *
 *   lightColor[i]      ← color + alpha
 *   lightPos[i]        ← position + unused
 *   lightData[i]       ← directional: unused; point: unused;
 *                         cone: [dirX, dirY, dirZ, coneAngle];
 *                         area: [posX, posY, posZ, roughness]
 *   lightData2[i]      ← cone: [feather, 0, 0, 0]; area: [halfWidthX/Y/Z, 0]
 *   lightData3[i]      ← area: [halfHeightX/Y/Z, 0]
 *   lightProperties[i] ← [intensity, range, minFloor, typeID]
 */
export interface PBRLight {
  /** 1=directional 2=point 3=cone/spot 4=area */
  type: ATLightType;
  /** World-space position (directional: treated as direction) */
  position: [number, number, number];
  /** Linear-space RGB color */
  color: [number, number, number];
  /** Intensity multiplier */
  intensity: number;
  /** Maximum influence radius (unused for directional). Default 20.0 */
  range: number;
  /** Ambient floor (lcrange min). Default 0.0 */
  minFloor: number;
  /** Cone/spot: [dirX, dirY, dirZ, coneAngleDeg]
   *  Area: [posX, posY, posZ, roughness]
   *  Others: [0,0,0,0] */
  data: [number, number, number, number];
  /** Cone: [feather, 0, 0, 0]; Area: [halfWidthX, halfWidthY, halfWidthZ, 0] */
  data2: [number, number, number, number];
  /** Area: [halfHeightX, halfHeightY, halfHeightZ, 0]; Others: [0,0,0,0] */
  data3: [number, number, number, number];
}

/** PBR material parameters. */
export interface PBRMaterialParams {
  /** Albedo tint (linear RGB). Default [1,1,1] */
  albedo: [number, number, number];
  /** Metallic factor 0..1. Default 0.0 */
  metallic: number;
  /** Roughness factor 0..1. Default 0.5 */
  roughness: number;
  /** Ambient occlusion intensity 0..1. Default 1.0 */
  aoIntensity: number;
  /** Normal map intensity. Default 1.0 */
  normalIntensity: number;
  /** IBL environment contribution scale. Default 1.0 */
  envIntensity: number;
  /** Matcap blend weight 0..1. Default 0.0 */
  matcapWeight: number;
  /** Mirror reflection blend 0..1. Default 0.0 */
  mirrorWeight: number;
  /** Emissive color (additive). Default [0,0,0] */
  emissive: [number, number, number];
  /** Emissive intensity. Default 0.0 */
  emissiveIntensity: number;
  /** UV tiling. Default [1,1] */
  tiling: [number, number];
  /** UV offset. Default [0,0] */
  offset: [number, number];
  /** Whether env map uses RGBM encoding. Default true */
  envRGBM: boolean;
  /** Exposure for tone-mapping. Default 1.0 */
  exposure: number;
}

/** Texture bindings for the PBR pipeline. */
export interface PBRTextures {
  /** Albedo / base color */
  baseColor: WebGLTexture | null;
  /** Packed MRO: Metallic(R) Roughness(G) Occlusion(B) */
  mro: WebGLTexture | null;
  /** Tangent-space normal map */
  normal: WebGLTexture | null;
  /** Environment / IBL map (equirectangular, RGBM) */
  env: WebGLTexture | null;
  /** BRDF integration LUT */
  brdfLut: WebGLTexture | null;
  /** Matcap texture (matcap-test.ktx2 or matcap3.ktx2) */
  matcap: WebGLTexture | null;
  /** LTC matrix LUT for area lights (tLTC1) */
  ltc1: WebGLTexture | null;
  /** LTC amplitude LUT for area lights (tLTC2) */
  ltc2: WebGLTexture | null;
  /** Mirror reflection render texture */
  mirrorReflection: WebGLTexture | null;
}

/** Shadow map binding for a single shadow caster. */
export interface PBRShadow {
  map: WebGLTexture;
  /** Shadow view-projection matrix [16] */
  matrix: Float32Array;
  /** Shadow map resolution */
  size: number;
  /** Light world position for self-shadow bias */
  lightPos: [number, number, number];
  /** Depth bias. Default 0.005 */
  bias: number;
}

/** Transform matrices for a draw call. */
export interface PBRMatrices {
  model: Float32Array;
  view: Float32Array;
  projection: Float32Array;
  /** Pre-multiplied modelView = view * model */
  modelView: Float32Array;
  /** Normal matrix = transpose(inverse(modelView)) upper 3×3, flat Float32Array[9] */
  normal: Float32Array;
  cameraPosition: [number, number, number];
}

/** Mirror plane configuration (BasicMirror.vs reprojection). */
export interface MirrorConfig {
  /** Mirror plane texture matrix (world → mirror clip). Float32Array[16] */
  matrix: Float32Array;
}

/** Configuration for ATFullPBRPipeline.create(). */
export interface ATFullPBRPipelineConfig {
  /** Max dynamic lights (NUM_LIGHTS define). Default 4 */
  maxLights?: number;
  /** Max shadow casters (SHADOW_COUNT define). Default 1 */
  maxShadows?: number;
  /** Enable PCSS soft shadows (slower). Default false */
  enablePCSS?: boolean;
  /** Shadow quality: 'low'|'med'|'high'. Default 'med' */
  shadowQuality?: 'low' | 'med' | 'high';
  /** Enable LTC area lights. Requires ltc1/ltc2 textures. Default true */
  enableAreaLights?: boolean;
  /** Enable matcap layer. Default true */
  enableMatcap?: boolean;
  /** Enable planar mirror reflection. Default false */
  enableMirror?: boolean;
  /** Enable IBL environment lighting. Default true */
  enableIBL?: boolean;
  /** WebGL1 compat mode (skip WebGL2 features). Default false */
  webgl1Compat?: boolean;
}

const DEFAULT_CONFIG: Required<ATFullPBRPipelineConfig> = {
  maxLights: 4,
  maxShadows: 1,
  enablePCSS: false,
  shadowQuality: 'med',
  enableAreaLights: true,
  enableMatcap: true,
  enableMirror: false,
  enableIBL: true,
  webgl1Compat: false,
};

const DEFAULT_MATERIAL: PBRMaterialParams = {
  albedo: [1, 1, 1],
  metallic: 0.0,
  roughness: 0.5,
  aoIntensity: 1.0,
  normalIntensity: 1.0,
  envIntensity: 1.0,
  matcapWeight: 0.0,
  mirrorWeight: 0.0,
  emissive: [0, 0, 0],
  emissiveIntensity: 0.0,
  tiling: [1, 1],
  offset: [0, 0],
  envRGBM: true,
  exposure: 1.0,
};

// ─────────────────────────────────────────────────────────────────────────────
// § 2  GLSL: AT LightingCommon (lworldLight / lrange / lclamp / lcrange)
// ─────────────────────────────────────────────────────────────────────────────
// Source: upstream/activetheory-assets/shaders/lighting.fs → {@}LightingCommon.glsl{@}

const GLSL_LIGHTING_COMMON = /* glsl */`
// ── AT LightingCommon.glsl ────────────────────────────────────────────────────
// lworldLight: world light pos → view-space direction vector
vec3 lworldLight(vec3 lp, vec3 lpos, mat4 mv, mat4 vm) {
    return (vm*vec4(lp,1.0)).xyz - (mv*vec4(lpos,1.0)).xyz;
}
float lrange(float v,float a,float b,float c,float d){vec3 s=vec3(v,d,b)-vec3(a,c,a);return s.x*s.y/s.z+c;}
vec3 lclamp(vec3 v){return clamp(v,vec3(0.0),vec3(1.0));}
float lcrange(float v,float a,float b,float c,float d){return clamp(lrange(v,a,b,c,d),min(c,d),max(c,d));}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 3  GLSL: AT Phong (Blinn-Phong + Schlick Fresnel)
// ─────────────────────────────────────────────────────────────────────────────
// Source: AT Phong.glsl (required by LightingCommon → lighting.fs)

const GLSL_PHONG = /* glsl */`
// ── AT Phong.glsl — Blinn-Phong + Schlick Fresnel ─────────────────────────────
float schlick(float c,float F0){float b=1.0-c;return F0+(1.0-F0)*b*b*b*b*b;}
float calcBlinnPhong(vec3 N,vec3 L,vec3 V,float sh){vec3 H=normalize(L+V);return pow(max(dot(N,H),0.0),sh);}
vec3 phong(float intensity,vec3 lC,vec3 pC,float sh,float att,vec3 N,vec3 L,vec3 V,float mf){
    float diff=lcrange(dot(normalize(L),N),0.0,1.0,mf,1.0);
    float F=schlick(max(dot(normalize(V),normalize(L+V)),0.0),0.04);
    return lC*diff*intensity*att + pC*calcBlinnPhong(N,normalize(L),normalize(V),sh)*F*intensity*att;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 4  GLSL: AT AreaLights (LTC — Linearly Transformed Cosines)
// ─────────────────────────────────────────────────────────────────────────────
// Source: AT AreaLights.glsl, referenced by lighting.fs → LightingCommon → AreaLights
// LTC reference: Heitz, Dupuy, Hill, Neubelt 2016

const GLSL_AREA_LIGHTS = /* glsl */`
// ── AT AreaLights.glsl — LTC Area Light Evaluation ───────────────────────────
// Reference: Heitz+Dupuy 2016 "Real-Time Polygonal-Light Shading with LTC"
// LUT textures: tLTC1 = inverse transform matrix, tLTC2 = norm + fresnel

const float LTC_LUT_SIZE  = 64.0;
const float LTC_LUT_SCALE = (LTC_LUT_SIZE - 1.0) / LTC_LUT_SIZE;
const float LTC_LUT_BIAS  = 0.5 / LTC_LUT_SIZE;

// Compute 2D LUT UV from (normal, view, roughness)
// AT AreaLights.glsl: LTC_Uv()
vec2 LTC_Uv(vec3 N, vec3 V, float roughness) {
    const float LUT_SZ = 64.0;
    float theta = acos(clamp(dot(N, V), 0.0, 1.0));
    vec2 uv = vec2(roughness, theta / (0.5 * 3.14159265));
    uv = uv * LTC_LUT_SCALE + LTC_LUT_BIAS;
    return uv;
}

// Clip polygon against hemisphere — AT AreaLights.glsl: LTC_ClipQuadToHorizon
// Sutherland-Hodgman clip of a quad (4 verts) against z=0 plane
void LTC_ClipQuadToHorizon(inout vec3 L[5], out int n) {
    int config = 0;
    if (L[0].z > 0.0) config += 1;
    if (L[1].z > 0.0) config += 2;
    if (L[2].z > 0.0) config += 4;
    if (L[3].z > 0.0) config += 8;
    n = 0;
    if      (config == 1)  { n=3; L[1]=-L[1].z*L[0]+L[0].z*L[1]; L[2]=-L[3].z*L[0]+L[0].z*L[3]; }
    else if (config == 2)  { n=3; L[0]=-L[0].z*L[1]+L[1].z*L[0]; L[2]=-L[2].z*L[1]+L[1].z*L[2]; }
    else if (config == 3)  { n=4; L[2]=-L[2].z*L[1]+L[1].z*L[2]; L[3]=-L[3].z*L[0]+L[0].z*L[3]; }
    else if (config == 4)  { n=3; L[0]=-L[3].z*L[2]+L[2].z*L[3]; L[1]=-L[1].z*L[2]+L[2].z*L[1]; }
    else if (config == 6)  { n=4; L[0]=-L[0].z*L[1]+L[1].z*L[0]; L[3]=-L[3].z*L[2]+L[2].z*L[3]; }
    else if (config == 7)  { n=5; L[4]=-L[3].z*L[0]+L[0].z*L[3]; L[3]=-L[3].z*L[2]+L[2].z*L[3]; }
    else if (config == 8)  { n=3; L[0]=-L[0].z*L[3]+L[3].z*L[0]; L[1]=-L[2].z*L[3]+L[3].z*L[2]; L[2]=L[3]; }
    else if (config == 9)  { n=4; L[1]=-L[1].z*L[0]+L[0].z*L[1]; L[2]=-L[2].z*L[3]+L[3].z*L[2]; }
    else if (config == 11) { n=5; L[4]=L[3]; L[3]=-L[2].z*L[3]+L[3].z*L[2]; L[2]=-L[2].z*L[1]+L[1].z*L[2]; }
    else if (config == 12) { n=4; L[1]=-L[1].z*L[2]+L[2].z*L[1]; L[0]=-L[0].z*L[3]+L[3].z*L[0]; }
    else if (config == 13) { n=5; L[4]=L[3]; L[3]=L[2]; L[2]=-L[1].z*L[2]+L[2].z*L[1]; L[1]=-L[1].z*L[0]+L[0].z*L[1]; }
    else if (config == 14) { n=5; L[4]=-L[0].z*L[3]+L[3].z*L[0]; L[0]=-L[0].z*L[1]+L[1].z*L[0]; }
    else if (config == 15) { n=4; }
    if (n==3) L[3]=L[0];
    if (n==4) L[4]=L[0];
}

// Edge integral — solid-angle accumulator (AT AreaLights.glsl: LTC_IntegrateEdge)
float LTC_IntegrateEdge(vec3 v1, vec3 v2) {
    float x=dot(v1,v2), y=abs(x);
    float v=(0.8543985+(0.4965155+0.0145206*y)*y)/(3.4175940+(4.1616724+y)*y);
    float ts=(x>0.0)?v:0.5*inversesqrt(max(1.0-x*x,1e-7))-v;
    return cross(v1,v2).z*ts;
}

// Evaluate LTC integral for a quad — AT AreaLights.glsl: LTC_Evaluate()
float LTC_Evaluate(vec3 N, vec3 V, vec3 P, mat3 mInv, vec3 points[4]) {
    vec3 T1=normalize(V-N*dot(V,N)), T2=cross(N,T1);
    mat3 basis=transpose(mat3(T1,T2,N));
    vec3 L[5];
    L[0]=mInv*(basis*(points[0]-P)); L[1]=mInv*(basis*(points[1]-P));
    L[2]=mInv*(basis*(points[2]-P)); L[3]=mInv*(basis*(points[3]-P)); L[4]=L[3];
    int n; LTC_ClipQuadToHorizon(L,n);
    if (n==0) return 0.0;
    L[0]=normalize(L[0]); L[1]=normalize(L[1]); L[2]=normalize(L[2]);
    L[3]=normalize(L[3]); L[4]=normalize(L[4]);
    float sum=LTC_IntegrateEdge(L[0],L[1])+LTC_IntegrateEdge(L[1],L[2])+LTC_IntegrateEdge(L[2],L[3]);
    if (n>=4) sum+=LTC_IntegrateEdge(L[3],L[4]);
    if (n==5) sum+=LTC_IntegrateEdge(L[4],L[0]);
    return abs(sum);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 5  GLSL: AT Lighting Functions (4 light types)
// ─────────────────────────────────────────────────────────────────────────────
// Source: lighting.fs — lightDirectional/lightPoint/lightCone/lightArea
// + getCombinedColor / getPointLightColor / getAreaLightColor / getSpotLightColor

function buildLightingFunctions(cfg: Required<ATFullPBRPipelineConfig>): string {
  return /* glsl */`
// ── AT lighting.fs — 4 light type implementations ────────────────────────────

// ── Directional light (type 1) ───────────────────────────────────────────────
// AT lightDirectional: lambertian N·L with ambient floor
vec3 lightDirectional(
    vec3 normal, vec3 lColor, vec3 lPos,
    vec4 lProps,
    vec3 vPos, vec3 vWorldPos, vec3 vViewDir,
    mat4 modelViewMatrix, mat4 viewMatrix
) {
    vec3 lDir = lworldLight(lPos, vPos, modelViewMatrix, viewMatrix);
    float volume = dot(normalize(lDir), normal);
    return lColor * lcrange(volume, 0.0, 1.0, lProps.z, 1.0);
}

// ── Point light (type 2) ─────────────────────────────────────────────────────
// AT lightPoint: inverse-square falloff + optional Phong
vec3 lightPoint(
    vec3 normal, vec3 lColor, vec3 lPos,
    vec4 lData, vec4 lData2, vec4 lData3, vec4 lProps,
    vec3 vPos, vec3 vWorldPos, vec3 vViewDir,
    mat4 modelViewMatrix, mat4 viewMatrix
) {
    float dist = length(vWorldPos - lPos);
    if (dist > lProps.y) return vec3(0.0);

    vec3 lDir = lworldLight(lPos, vPos, modelViewMatrix, viewMatrix);
    float falloff = pow(lcrange(dist, 0.0, lProps.y, 1.0, 0.0), 2.0);

    // Standard diffuse-only path (no Phong in full PBR — BRDF handles specular)
    float volume = dot(normalize(lDir), normal);
    volume = lcrange(volume, 0.0, 1.0, lProps.z, 1.0);
    return lColor * volume * lProps.x * falloff;
}

// ── Cone / Spot light (type 3) ────────────────────────────────────────────────
// AT lightCone: spot with feathered edges
vec3 lightCone(
    vec3 normal, vec3 lColor, vec3 lPos,
    vec4 lData, vec4 lData2, vec4 lData3, vec4 lProps,
    vec3 vPos, vec3 vWorldPos, vec3 vViewDir,
    mat4 modelViewMatrix, mat4 viewMatrix
) {
    float dist = length(vWorldPos - lPos);
    if (dist > lProps.y) return vec3(0.0);

    vec3 sDir = degrees(-lData.xyz);
    float radius = lData.w;
    vec3 surfaceToLight = normalize(lPos - vWorldPos);
    float lightToSurfaceAngle = degrees(acos(dot(-surfaceToLight, normalize(sDir))));

    vec3 nColor = lightPoint(
        normal, lColor, lPos, lData, lData2, lData3, lProps,
        vPos, vWorldPos, vViewDir, modelViewMatrix, viewMatrix
    );

    float featherMin = 1.0 - lData2.x * 0.1;
    float featherMax = 1.0 + lData2.x * 0.1;
    float attenuation = smoothstep(
        lightToSurfaceAngle * featherMin,
        lightToSurfaceAngle * featherMax,
        radius
    );

    return nColor * attenuation;
}

// ── Area light (type 4) — LTC ─────────────────────────────────────────────────
// AT lightArea: Linearly Transformed Cosines — requires tLTC1 / tLTC2
vec3 lightArea(
    vec3 normal, vec3 lColor, vec3 lPos,
    vec4 lData, vec4 lData2, vec4 lData3, vec4 lProps,
    vec3 vPos, vec3 vWorldPos, vec3 vViewDir,
    mat4 modelViewMatrix, mat4 viewMatrix,
    sampler2D tLTC1, sampler2D tLTC2
) {
    float dist = length(vWorldPos - lPos);
    if (dist > lProps.y) return vec3(0.0);

    vec3 normal_ = normal;
    vec3 viewDir = normalize(vViewDir);
    vec3 position = -vViewDir;
    float roughness = lData.w;
    vec3 mPos = lData.xyz;
    vec3 halfWidth  = lData2.xyz;
    vec3 halfHeight = lData3.xyz;

    float falloff = pow(lcrange(dist, 0.0, lProps.y, 1.0, 0.0), 2.0);

    vec3 rectCoords[4];
    rectCoords[0] = mPos + halfWidth - halfHeight;
    rectCoords[1] = mPos - halfWidth - halfHeight;
    rectCoords[2] = mPos - halfWidth + halfHeight;
    rectCoords[3] = mPos + halfWidth + halfHeight;

    vec2 uv = LTC_Uv(normal_, viewDir, roughness);
    vec4 t1 = texture2D(tLTC1, uv);
    vec4 t2 = texture2D(tLTC2, uv);

    mat3 mInv = mat3(
        vec3(t1.x, 0.0, t1.y),
        vec3(0.0,  1.0, 0.0),
        vec3(t1.z, 0.0, t1.w)
    );

    vec3 fresnel = (lColor * t2.x + (vec3(1.0) - lColor) * t2.y);
    vec3 color = lColor * fresnel * LTC_Evaluate(normal_, viewDir, position, mInv, rectCoords) * falloff * lProps.x;
    color += lColor * LTC_Evaluate(normal_, viewDir, position, mat3(1.0), rectCoords) * falloff * lProps.x;

    return color;
}

// ── Combined color dispatcher ─────────────────────────────────────────────────
// AT getCombinedColor: iterate all NUM_LIGHTS and accumulate per-type contribution
vec3 getCombinedLightColor(
    vec3 normal,
    vec3 vPos, vec3 vWorldPos, vec3 vViewDir,
    mat4 modelViewMatrix, mat4 viewMatrix,
    sampler2D tLTC1, sampler2D tLTC2,
    vec4 lightColor[${cfg.maxLights}],
    vec4 lightPos[${cfg.maxLights}],
    vec4 lightData[${cfg.maxLights}],
    vec4 lightData2[${cfg.maxLights}],
    vec4 lightData3[${cfg.maxLights}],
    vec4 lightProperties[${cfg.maxLights}]
) {
    vec3 color = vec3(0.0);

    for (int i = 0; i < ${cfg.maxLights}; i++) {
        vec3 lColor  = lightColor[i].rgb;
        vec3 lPos    = lightPos[i].rgb;
        vec4 lData   = lightData[i];
        vec4 lData2_ = lightData2[i];
        vec4 lData3_ = lightData3[i];
        vec4 lProps  = lightProperties[i];

        // lProps.w encodes light type: 0=disabled, 1=dir, 2=point, 3=cone, 4=area
        if (lProps.w < 1.0) continue;

        if (lProps.w < 1.1) {
            color += lightDirectional(normal, lColor, lPos, lProps,
                vPos, vWorldPos, vViewDir, modelViewMatrix, viewMatrix);
        } else if (lProps.w < 2.1) {
            color += lightPoint(normal, lColor, lPos, lData, lData2_, lData3_, lProps,
                vPos, vWorldPos, vViewDir, modelViewMatrix, viewMatrix);
        } else if (lProps.w < 3.1) {
            color += lightCone(normal, lColor, lPos, lData, lData2_, lData3_, lProps,
                vPos, vWorldPos, vViewDir, modelViewMatrix, viewMatrix);
        } else if (lProps.w < 4.1) {
            color += lightArea(normal, lColor, lPos, lData, lData2_, lData3_, lProps,
                vPos, vWorldPos, vViewDir, modelViewMatrix, viewMatrix, tLTC1, tLTC2);
        }
    }

    return lclamp(color);
}
`;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  GLSL: AT Shadow (PCF / PCSS — from shadows.fs)
// ─────────────────────────────────────────────────────────────────────────────
// Source: AT shadows.fs — PCSShadowConfig / getShadow / getShadowPCSS

function buildShadowGLSL(cfg: Required<ATFullPBRPipelineConfig>): string {
  if (cfg.maxShadows <= 0) {
    return `float getShadowFactor(vec3 pos, vec3 normal) { return 1.0; }`;
  }

  const shadowMapDecls = Array.from({ length: cfg.maxShadows }, (_, i) =>
    `uniform sampler2D shadowMap_${i};`
  ).join('\n');

  const shadowFetch = cfg.maxShadows === 1
    ? `texture2D(shadowMap_0, uv)`
    : Array.from({ length: cfg.maxShadows }, (_, i) =>
        `if (idx == ${i}) return texture2D(shadowMap_${i}, uv);`
      ).join('\n') + `return vec4(1.0);`;

  const qualityDefine = cfg.shadowQuality === 'high'
    ? '#define SHADOWS_HIGH'
    : cfg.shadowQuality === 'med'
    ? '#define SHADOWS_MED'
    : '';

  const pcssBlock = cfg.enablePCSS ? /* glsl */`
// ── AT PCSS (soft shadows, from shadows.fs) ───────────────────────────────────
#define MAX_PCSS_SAMPLES 17
vec2 poissonDisk_pbr[MAX_PCSS_SAMPLES];

void initPoissonSamples_pbr(vec2 seed, int sc, int rc) {
    float PI2 = 6.2831853072;
    float angle = fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453) * PI2;
    float radius = 1.0 / float(sc);
    float radiusStep = radius;
    float angleStep = PI2 * float(rc) / float(sc);
    for (int i = 0; i < MAX_PCSS_SAMPLES; i++) {
        if (i >= sc) break;
        poissonDisk_pbr[i] = vec2(cos(angle), sin(angle)) * pow(radius, 0.75);
        radius += radiusStep; angle += angleStep;
    }
}

float shadowFetchR_pcss(int idx, vec2 uv) {
    return texture2D(shadowMap_0, uv).r;
}

float findBlocker_pbr(int idx, vec2 uv, float zR, float searchR, int sc) {
    float sum = 0.0; int nb = 0;
    for (int i = 0; i < MAX_PCSS_SAMPLES; i++) {
        if (i >= sc) break;
        float d = shadowFetchR_pcss(idx, uv + poissonDisk_pbr[i] * searchR);
        if (d < zR) { sum += d; nb++; }
    }
    return nb == 0 ? -1.0 : sum / float(nb);
}

float pcfFilter_pbr(int idx, vec2 uv, float zR, float fr, int sc) {
    float sum = 0.0;
    for (int i = 0; i < MAX_PCSS_SAMPLES; i++) {
        if (i >= sc) break;
        if (zR <= shadowFetchR_pcss(idx, uv + poissonDisk_pbr[i] * fr)) sum += 1.0;
    }
    for (int i = 0; i < MAX_PCSS_SAMPLES; i++) {
        if (i >= sc) break;
        if (zR <= shadowFetchR_pcss(idx, uv + (-poissonDisk_pbr[i]).yx * fr)) sum += 1.0;
    }
    return sum / (2.0 * float(sc));
}

float PCSS_pbr(int idx, vec3 coords) {
    const float lwSize = 0.3, lfWidth = 6.75, nearP = 6.5;
    const int sc = 10, rc = 11;
    initPoissonSamples_pbr(coords.xy, sc, rc);
    float lSizeUV = lwSize / lfWidth;
    float searchR = lSizeUV * (coords.z - nearP) / coords.z;
    float avgB = findBlocker_pbr(idx, coords.xy, coords.z, searchR, sc);
    if (avgB < 0.0) return 1.0;
    float penumbra = (coords.z - avgB) / avgB;
    float fr = penumbra * lSizeUV * nearP / coords.z;
    return pcfFilter_pbr(idx, coords.xy, coords.z, fr, sc);
}
` : '';

  return /* glsl */`
// ── AT shadows.fs — Shadow System ─────────────────────────────────────────────
${qualityDefine}
${shadowMapDecls}
uniform mat4  uShadowMatrix[${cfg.maxShadows}];
uniform float uShadowSize[${cfg.maxShadows}];
uniform vec3  uShadowLightPos[${cfg.maxShadows}];
uniform float uShadowBias;

// Step compare
float shadowCompare_pbr(float mapDepth, float compare) {
    return step(compare, mapDepth);
}

// Bilinear PCF lerp (AT shadowLerp)
float shadowLerp_pbr(int idx, vec2 coords, float compare, float mapSize) {
    vec2 ts=vec2(1.0)/mapSize, cuv=floor(coords*mapSize+0.5)/mapSize;
    float lb=shadowCompare_pbr(texture2D(shadowMap_0,cuv+ts*vec2(0.0,0.0)).r,compare);
    float lt=shadowCompare_pbr(texture2D(shadowMap_0,cuv+ts*vec2(0.0,1.0)).r,compare);
    float rb=shadowCompare_pbr(texture2D(shadowMap_0,cuv+ts*vec2(1.0,0.0)).r,compare);
    float rt=shadowCompare_pbr(texture2D(shadowMap_0,cuv+ts*vec2(1.0,1.0)).r,compare);
    vec2 f=fract(coords*mapSize+0.5);
    return mix(mix(lb,lt,f.y),mix(rb,rt,f.y),f.x);
}

// Standard PCF lookup with noise jitter — AT shadows.fs
float shadowLookup_pbr(int idx, vec3 coords, float mapSize, float compare, vec3 wpos) {
    bool inFrustum = coords.x >= 0.0 && coords.x <= 1.0 &&
                     coords.y >= 0.0 && coords.y <= 1.0 && coords.z <= 1.0;
    if (!inFrustum) return 1.0;
    vec2 ts = vec2(1.0) / mapSize;
    float t = wpos.z * 0.3;
    float noise = (sin(wpos.x * 0.9/0.5 + t*10.0) + sin(wpos.x * 2.4/0.5 + t*15.0)) * 0.00015 * 0.5;
    float dx0 = -ts.x + noise, dy0 = -ts.y - noise, dx1 = ts.x + noise, dy1 = ts.y - noise;
    float shadow = 1.0;
    #ifdef SHADOWS_HIGH
    shadow  = shadowLerp_pbr(idx, coords.xy+vec2(dx0,dy0), compare, mapSize);
    shadow += shadowLerp_pbr(idx, coords.xy+vec2(0.0,dy0), compare, mapSize);
    shadow += shadowLerp_pbr(idx, coords.xy+vec2(dx1,dy0), compare, mapSize);
    shadow += shadowLerp_pbr(idx, coords.xy+vec2(dx0,0.0), compare, mapSize);
    shadow += shadowLerp_pbr(idx, coords.xy,               compare, mapSize);
    shadow += shadowLerp_pbr(idx, coords.xy+vec2(dx1,0.0), compare, mapSize);
    shadow += shadowLerp_pbr(idx, coords.xy+vec2(dx0,dy1), compare, mapSize);
    shadow += shadowLerp_pbr(idx, coords.xy+vec2(0.0,dy1), compare, mapSize);
    shadow += shadowLerp_pbr(idx, coords.xy+vec2(dx1,dy1), compare, mapSize);
    shadow /= 9.0;
    #elif defined(SHADOWS_MED)
    shadow  = shadowCompare_pbr(texture2D(shadowMap_0, coords.xy+vec2(0.0,dy0)).r, compare);
    shadow += shadowCompare_pbr(texture2D(shadowMap_0, coords.xy+vec2(dx0,0.0)).r, compare);
    shadow += shadowCompare_pbr(texture2D(shadowMap_0, coords.xy).r, compare);
    shadow += shadowCompare_pbr(texture2D(shadowMap_0, coords.xy+vec2(dx1,0.0)).r, compare);
    shadow += shadowCompare_pbr(texture2D(shadowMap_0, coords.xy+vec2(0.0,dy1)).r, compare);
    shadow /= 5.0;
    #else
    shadow = shadowCompare_pbr(texture2D(shadowMap_0, coords.xy).r, compare);
    #endif
    return clamp(shadow, 0.0, 1.0);
}

${pcssBlock}

// Main shadow entry: accumulate over all shadow casters
float getShadowFactor(vec3 wpos, vec3 normal) {
    float shadow = 1.0;
    for (int i = 0; i < ${cfg.maxShadows}; i++) {
        vec4 sc = uShadowMatrix[i] * vec4(wpos, 1.0);
        vec3 coords = (sc.xyz / sc.w) * 0.5 + 0.5;
        float bias = uShadowBias;
        ${cfg.enablePCSS
          ? 'float lookup = PCSS_pbr(i, vec3(coords.xy, coords.z - bias));'
          : 'float lookup = shadowLookup_pbr(i, coords, uShadowSize[i], coords.z - bias, wpos);'}
        shadow *= clamp(lookup, 0.0, 1.0);
    }
    return shadow;
}
`;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 7  GLSL: Cook-Torrance PBR BRDF + IBL + Matcap + Mirror
// ─────────────────────────────────────────────────────────────────────────────

function buildPBRCorGLSL(cfg: Required<ATFullPBRPipelineConfig>): string {
  return /* glsl */`
// ── Cook-Torrance BRDF ─────────────────────────────────────────────────────────
#define PBR_PI 3.14159265358979323846
#define PBR_INV_PI 0.31830988618379067

float D_GGX_pbr(float NdotH, float r) { float a=r*r,a2=a*a,d=(NdotH*NdotH)*(a2-1.0)+1.0; return a2/max(PBR_PI*d*d,1e-7); }
float G_SmithGGX_pbr(float NdotV, float NdotL, float r) {
    float a=r*r,a2=a*a;
    float gV=NdotL*sqrt(NdotV*NdotV*(1.0-a2)+a2), gL=NdotV*sqrt(NdotL*NdotL*(1.0-a2)+a2);
    return 0.5/max(gV+gL,1e-7);
}
vec3 F_Schlick_pbr(vec3 f0, float cosT) { float b=1.0-clamp(cosT,0.0,1.0); float b5=b*b*b*b*b; return f0+(vec3(1.0)-f0)*b5; }
vec3 specularBRDF_pbr(vec3 N, vec3 V, vec3 L, vec3 f0, float r) {
    vec3 H=normalize(V+L);
    float NdotH=clamp(dot(N,H),0.0,1.0), NdotV=clamp(dot(N,V),0.0,1.0);
    float NdotL=clamp(dot(N,L),0.0,1.0), VdotH=clamp(dot(V,H),0.0,1.0);
    return vec3(D_GGX_pbr(NdotH,r)*G_SmithGGX_pbr(NdotV,NdotL,r))*F_Schlick_pbr(f0,VdotH);
}

// ── Normal Map Decoding (AT normalmap.glsl: unpackNormal) ──────────────────────
vec3 unpackNormalMap(vec3 eyePos, vec3 surfNorm, sampler2D nmap, float intensity, float scale, vec2 uv) {
    vec3 q0=dFdx(eyePos), q1=dFdy(eyePos);
    vec2 st0=dFdx(uv), st1=dFdy(uv);
    vec3 N=normalize(surfNorm);
    vec3 T=cross(q1,N)*st0.x+cross(N,q0)*st1.x;
    vec3 B=cross(q1,N)*st0.y+cross(N,q0)*st1.y;
    float det=max(dot(T,T),dot(B,B));
    float sf=(det==0.0)?0.0:inversesqrt(det);
    vec3 mapN=texture2D(nmap,uv*scale).xyz*2.0-1.0;
    mapN.xy*=intensity;
    return normalize(T*(mapN.x*sf)+B*(mapN.y*sf)+N*mapN.z);
}

// ── RGBM + Environment ────────────────────────────────────────────────────────
vec3 decodeRGBM(vec4 c, float maxR) { return c.rgb*(c.a*maxR); }
vec2 sphericalUV(vec3 d) { return vec2(0.5+atan(d.z,d.x)/(2.0*PBR_PI), acos(clamp(d.y,-1.0,1.0))/PBR_PI); }
vec3 sampleEnvMap(sampler2D t, vec3 d, bool rgbm) {
    vec4 s=texture2D(t,sphericalUV(normalize(d)));
    return rgbm ? decodeRGBM(s,6.0) : s.rgb;
}

// ── IBL Contribution (AT pbr.fs: getIBLContribution) ──────────────────────────
vec3 getIBLContribution(vec3 N, vec3 V, vec3 albedo, vec3 f0, float r, float m, float ao,
                        sampler2D envTex, sampler2D brdfLut, float envI, bool rgbm) {
    float NdotV=clamp(dot(N,V),0.0,1.0);
    vec3 diffE=sampleEnvMap(envTex,N,rgbm);
    vec2 specUV=sphericalUV(normalize(reflect(-V,N)));
    specUV.y=clamp(specUV.y+r*0.15,0.0,1.0);
    vec4 sp=texture2D(envTex,specUV);
    vec3 specE=rgbm?decodeRGBM(sp,6.0):sp.rgb;
    vec2 brdf=texture2D(brdfLut,vec2(NdotV,r)).rg;
    vec3 Fenv=f0+(max(vec3(1.0-r),f0)-f0)*pow(1.0-NdotV,5.0);
    vec3 diffI=albedo*(vec3(1.0)-Fenv)*(1.0-m)*diffE;
    vec3 specI=specE*(Fenv*brdf.x+brdf.y);
    return (diffI+specI)*ao*envI;
}

// ── Matcap (AT matcap*.ktx2 — screen-space normal UV) ─────────────────────────
vec3 sampleMatcap(sampler2D t, vec3 normal, mat4 viewMatrix) {
    vec3 nv=normalize(mat3(viewMatrix)*normal);
    vec2 uv=vec2(nv.x,1.0-nv.y)*0.5+0.5;
    return texture2D(t,uv).rgb;
}

// ── Mirror Reflection (AT BasicMirror.fs) ─────────────────────────────────────
vec3 sampleMirrorReflection(sampler2D t, vec4 coord) {
    return texture2D(t,clamp(coord.xy/coord.w,0.001,0.999)).rgb;
}

// ── Tone Mapping + Gamma (AT pbr.fs: Uncharted2) ──────────────────────────────
vec3 _uc2(vec3 x){float A=.15,B=.5,C=.1,D=.2,E=.02,F=.3;return((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;}
vec3 toneMapUncharted2(vec3 c, float e) { return _uc2(c*e)/(_uc2(vec3(11.2))); }
vec3 linearToSRGB(vec3 c) { return pow(max(c,vec3(0.0)),vec3(1.0/2.2)); }
`;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  Full Vertex Shader
// ─────────────────────────────────────────────────────────────────────────────

function buildFullVertexShader(cfg: Required<ATFullPBRPipelineConfig>): string {
  const mirrorVarying = cfg.enableMirror
    ? 'varying vec4 vMirrorCoord;'
    : '';
  const mirrorCalc = cfg.enableMirror
    ? 'vMirrorCoord = uMirrorMatrix * worldPos4;'
    : '';

  return /* glsl */`
precision highp float;

// ── AT Standard Matrices (WebGL1 compat names) ─────────────────────────────────
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat3 normalMatrix;
uniform vec3 cameraPosition;
uniform vec2 uTiling;
uniform vec2 uOffset;

${cfg.enableMirror ? 'uniform mat4 uMirrorMatrix;' : ''}

// ── Attributes ──────────────────────────────────────────────────────────────────
attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;
attribute vec4 tangent;  // optional, for TBN

// ── Varyings ────────────────────────────────────────────────────────────────────
varying vec3 vPos;           // local position (for AT lworldLight)
varying vec3 vWorldPos;      // world-space position
varying vec3 vNormal;        // view-space normal
varying vec3 vWorldNormal;   // world-space normal
varying vec3 vViewDir;       // view direction (world space)
varying vec2 vUv;
varying vec4 vTangent;       // tangent for TBN
${mirrorVarying}

void main() {
    vec3 pos = position;

    // World position
    vec4 worldPos4   = modelMatrix * vec4(pos, 1.0);
    vWorldPos        = worldPos4.xyz;
    vPos             = pos;

    // Normal (view space — used by AT lworldLight)
    vNormal          = normalize(normalMatrix * normal);
    // Normal (world space — used by IBL / matcap)
    vWorldNormal     = normalize(mat3(modelMatrix) * normal);

    // View direction: camera → surface (world space)
    vViewDir         = cameraPosition - vWorldPos;

    // UV with tiling/offset
    vUv              = uv * uTiling + uOffset;
    vTangent         = tangent;

    ${mirrorCalc}

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
`;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 9  Full Fragment Shader
// ─────────────────────────────────────────────────────────────────────────────

function buildFullFragmentShader(cfg: Required<ATFullPBRPipelineConfig>): string {
  const lightingFunctions = buildLightingFunctions(cfg);
  const shadowGLSL = buildShadowGLSL(cfg);
  const pbrCoreGLSL = buildPBRCorGLSL(cfg);

  const mirrorVarying = cfg.enableMirror ? 'varying vec4 vMirrorCoord;' : '';

  const lightUniformArrays = `
uniform vec4 lightColor[${cfg.maxLights}];
uniform vec4 lightPos[${cfg.maxLights}];
uniform vec4 lightData[${cfg.maxLights}];
uniform vec4 lightData2[${cfg.maxLights}];
uniform vec4 lightData3[${cfg.maxLights}];
uniform vec4 lightProperties[${cfg.maxLights}];
`;

  const mainBody = /* glsl */`
void main() {
    // ── UV ───────────────────────────────────────────────────────────────────────
    vec2 uv = vUv;

    // ── Base Color ───────────────────────────────────────────────────────────────
    vec4 baseColorSample = (uUseBaseColorMap > 0.5)
        ? texture2D(tBaseColor, uv)
        : vec4(1.0);
    vec3 albedo = baseColorSample.rgb * uAlbedo;
    float alpha = baseColorSample.a;

    // ── MRO (Metallic / Roughness / Occlusion) ────────────────────────────────────
    vec4 mroSample = (uUseMROMap > 0.5)
        ? texture2D(tMRO, uv)
        : vec4(0.0, 0.5, 1.0, 1.0);
    float metallic  = mroSample.r * uMetallic;
    float roughness = mroSample.g * uRoughness;
    float ao        = mix(1.0, mroSample.b, uAOIntensity);

    // Ensure physically-based bounds
    roughness = clamp(roughness, 0.04, 1.0);
    metallic  = clamp(metallic,  0.0,  1.0);

    // ── F0 (Reflectance at 0°) ───────────────────────────────────────────────────
    // Metallic: F0 = albedo; Dielectric: F0 = 0.04
    vec3 f0 = mix(vec3(0.04), albedo, metallic);

    // ── Normal ───────────────────────────────────────────────────────────────────
    vec3 N;
    if (uUseNormalMap > 0.5) {
        // AT unpackNormal: derivative-based TBN from eyePos + surfNorm
        vec3 eyePos = (modelViewMatrix * vec4(vPos, 1.0)).xyz;
        N = unpackNormalMap(eyePos, vWorldNormal, tNormal, uNormalIntensity, 1.0, uv);
    } else {
        N = normalize(vWorldNormal);
    }

    // View direction (normalized)
    vec3 V = normalize(vViewDir);

    // ── AT Combined Light Color ───────────────────────────────────────────────────
    // Iterate all light types via getCombinedLightColor() (§5)
    vec3 atLightAccum = getCombinedLightColor(
        N, vPos, vWorldPos, vViewDir,
        modelViewMatrix, viewMatrix,
        tLTC1, tLTC2,
        lightColor, lightPos, lightData, lightData2, lightData3, lightProperties
    );

    // ── Cook-Torrance Direct Lighting ─────────────────────────────────────────────
    // For each enabled light, evaluate PBR BRDF and accumulate
    vec3 directLighting = vec3(0.0);
    for (int i = 0; i < ${cfg.maxLights}; i++) {
        vec4 lProps = lightProperties[i];
        if (lProps.w < 1.0) continue;

        vec3  lColor = lightColor[i].rgb;
        vec3  lPos_w = lightPos[i].rgb;
        float lIntensity = lProps.x;

        // Light direction depends on type
        vec3 L;
        float attenuation = 1.0;

        if (lProps.w < 1.1) {
            // Directional — lPos is direction
            L = normalize(lworldLight(lPos_w, vPos, modelViewMatrix, viewMatrix));
            attenuation = 1.0;
        } else {
            // Point / cone / area — lPos is world position
            float dist = length(vWorldPos - lPos_w);
            if (dist > lProps.y) continue;
            attenuation = pow(lcrange(dist, 0.0, lProps.y, 1.0, 0.0), 2.0);
            L = normalize(lPos_w - vWorldPos);
        }

        float NdotL = clamp(dot(N, L), 0.0, 1.0);
        if (NdotL < 0.001) continue;

        // Diffuse: Lambertian (1 - F) * albedo / π * (1 - metallic)
        vec3 H = normalize(V + L);
        vec3 F = F_Schlick_pbr(f0, clamp(dot(V, H), 0.0, 1.0));
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        vec3 diffuse = kD * albedo * PBR_INV_PI;

        // Specular: Cook-Torrance
        vec3 spec = specularBRDF_pbr(N, V, L, f0, roughness);

        directLighting += (diffuse + spec) * lColor * lIntensity * NdotL * attenuation;
    }

    // ── IBL (Environment Lighting) ─────────────────────────────────────────────
    vec3 iblContrib = vec3(0.0);
    ${cfg.enableIBL ? `
    if (uUseEnvMap > 0.5) {
        iblContrib = getIBLContribution(
            N, V, albedo, f0, roughness, metallic, ao,
            tEnv, tBRDFLut, uEnvIntensity, uEnvRGBM > 0.5
        );
    }
    ` : ''}

    // ── Matcap Layer ──────────────────────────────────────────────────────────────
    vec3 matcapContrib = vec3(0.0);
    ${cfg.enableMatcap ? `
    if (uMatcapWeight > 0.001 && uUseMatcap > 0.5) {
        matcapContrib = sampleMatcap(tMatcap, N, viewMatrix) * uMatcapWeight;
    }
    ` : ''}

    // ── Shadow ────────────────────────────────────────────────────────────────────
    float shadowFactor = getShadowFactor(vWorldPos, N);

    // ── Mirror Reflection ─────────────────────────────────────────────────────────
    vec3 mirrorContrib = vec3(0.0);
    ${cfg.enableMirror ? `
    if (uMirrorWeight > 0.001) {
        mirrorContrib = sampleMirrorReflection(tMirrorReflection, vMirrorCoord) * uMirrorWeight;
    }
    ` : ''}

    // ── Combine ───────────────────────────────────────────────────────────────────
    // Direct lighting (shadow-modulated) + IBL + matcap + mirror + emissive
    vec3 color = vec3(0.0);

    // Direct + AT area lights (shadow only on direct)
    color += (directLighting + atLightAccum * albedo) * shadowFactor;

    // IBL (no shadow — indirect lighting)
    color += iblContrib;

    // Matcap overlay
    color += matcapContrib;

    // Mirror reflection
    color += mirrorContrib;

    // Emissive
    color += uEmissive * uEmissiveIntensity;

    // ── Tone Mapping ──────────────────────────────────────────────────────────────
    color = toneMapUncharted2(color, uExposure);

    // ── Gamma ─────────────────────────────────────────────────────────────────────
    color = linearToSRGB(color);

    gl_FragColor = vec4(color, alpha);
}
`;

  return /* glsl */`
precision highp float;
precision highp int;

// ── AT Built-in Matrices ───────────────────────────────────────────────────────
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 viewMatrix;
uniform mat3 normalMatrix;
uniform vec3 cameraPosition;

// ── Varyings ────────────────────────────────────────────────────────────────────
varying vec3 vPos;
varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vWorldNormal;
varying vec3 vViewDir;
varying vec2 vUv;
varying vec4 vTangent;
${mirrorVarying}

// ── PBR Textures ────────────────────────────────────────────────────────────────
uniform sampler2D tBaseColor;
uniform sampler2D tMRO;
uniform sampler2D tNormal;
uniform sampler2D tEnv;
uniform sampler2D tBRDFLut;
uniform sampler2D tMatcap;
uniform sampler2D tLTC1;
uniform sampler2D tLTC2;
${cfg.enableMirror ? 'uniform sampler2D tMirrorReflection;' : ''}

// ── PBR Material Uniforms ───────────────────────────────────────────────────────
uniform vec3  uAlbedo;
uniform float uMetallic;
uniform float uRoughness;
uniform float uAOIntensity;
uniform float uNormalIntensity;
uniform float uEnvIntensity;
uniform float uMatcapWeight;
uniform float uMirrorWeight;
uniform vec3  uEmissive;
uniform float uEmissiveIntensity;
uniform float uExposure;
uniform float uEnvRGBM;

// ── Texture Enable Flags ────────────────────────────────────────────────────────
uniform float uUseBaseColorMap;
uniform float uUseMROMap;
uniform float uUseNormalMap;
uniform float uUseEnvMap;
uniform float uUseMatcap;

// ── AT Light Uniforms ────────────────────────────────────────────────────────────
${lightUniformArrays}

// ═══════════════════════════════════════════════════════════════════════════════
// Inline GLSL Libraries
// ═══════════════════════════════════════════════════════════════════════════════

${GLSL_LIGHTING_COMMON}
${GLSL_PHONG}
${GLSL_AREA_LIGHTS}
${lightingFunctions}
${shadowGLSL}
${pbrCoreGLSL}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

${mainBody}
`;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 10  WebGL Helpers
// ─────────────────────────────────────────────────────────────────────────────

function compileShader(
  gl: WebGLRenderingContext,
  type: number,
  src: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('[ATFullPBRPipeline] gl.createShader failed');
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader) ?? 'unknown';
    gl.deleteShader(shader);
    const typeStr = type === gl.VERTEX_SHADER ? 'vertex' : 'fragment';
    // Annotate the error with line numbers
    const lines = src.split('\n').map((l, i) => `${String(i + 1).padStart(4)}  ${l}`).join('\n');
    throw new Error(`[ATFullPBRPipeline] ${typeStr} shader compile error:\n${info}\n${lines}`);
  }
  return shader;
}

function linkProgram(
  gl: WebGLRenderingContext,
  vertSrc: string,
  fragSrc: string,
): WebGLProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);

  const prog = gl.createProgram();
  if (!prog) throw new Error('[ATFullPBRPipeline] gl.createProgram failed');

  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);

  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(prog) ?? 'unknown';
    gl.deleteProgram(prog);
    throw new Error(`[ATFullPBRPipeline] Program link error:\n${info}`);
  }

  gl.detachShader(prog, vs);
  gl.detachShader(prog, fs);
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  return prog;
}

/** Cached uniform / attribute locations. */
interface ProgramLocations {
  // Matrices
  modelMatrix: WebGLUniformLocation | null;
  modelViewMatrix: WebGLUniformLocation | null;
  viewMatrix: WebGLUniformLocation | null;
  projectionMatrix: WebGLUniformLocation | null;
  normalMatrix: WebGLUniformLocation | null;
  cameraPosition: WebGLUniformLocation | null;
  mirrorMatrix: WebGLUniformLocation | null;
  // Textures
  tBaseColor: WebGLUniformLocation | null;
  tMRO: WebGLUniformLocation | null;
  tNormal: WebGLUniformLocation | null;
  tEnv: WebGLUniformLocation | null;
  tBRDFLut: WebGLUniformLocation | null;
  tMatcap: WebGLUniformLocation | null;
  tLTC1: WebGLUniformLocation | null;
  tLTC2: WebGLUniformLocation | null;
  tMirrorReflection: WebGLUniformLocation | null;
  // Material
  uAlbedo: WebGLUniformLocation | null;
  uMetallic: WebGLUniformLocation | null;
  uRoughness: WebGLUniformLocation | null;
  uAOIntensity: WebGLUniformLocation | null;
  uNormalIntensity: WebGLUniformLocation | null;
  uEnvIntensity: WebGLUniformLocation | null;
  uMatcapWeight: WebGLUniformLocation | null;
  uMirrorWeight: WebGLUniformLocation | null;
  uEmissive: WebGLUniformLocation | null;
  uEmissiveIntensity: WebGLUniformLocation | null;
  uExposure: WebGLUniformLocation | null;
  uEnvRGBM: WebGLUniformLocation | null;
  uTiling: WebGLUniformLocation | null;
  uOffset: WebGLUniformLocation | null;
  // Texture enable flags
  uUseBaseColorMap: WebGLUniformLocation | null;
  uUseMROMap: WebGLUniformLocation | null;
  uUseNormalMap: WebGLUniformLocation | null;
  uUseEnvMap: WebGLUniformLocation | null;
  uUseMatcap: WebGLUniformLocation | null;
  // Shadow
  uShadowBias: WebGLUniformLocation | null;
  shadowMatrix: Array<WebGLUniformLocation | null>;
  shadowSize: Array<WebGLUniformLocation | null>;
  shadowLightPos: Array<WebGLUniformLocation | null>;
  // Lights (per-index)
  lightColor: Array<WebGLUniformLocation | null>;
  lightPos: Array<WebGLUniformLocation | null>;
  lightData: Array<WebGLUniformLocation | null>;
  lightData2: Array<WebGLUniformLocation | null>;
  lightData3: Array<WebGLUniformLocation | null>;
  lightProperties: Array<WebGLUniformLocation | null>;
  // Attributes
  aPosition: number;
  aNormal: number;
  aUV: number;
  aTangent: number;
}

function cacheLocations(
  gl: WebGLRenderingContext,
  prog: WebGLProgram,
  cfg: Required<ATFullPBRPipelineConfig>,
): ProgramLocations {
  const u = (name: string) => gl.getUniformLocation(prog, name);
  const a = (name: string) => gl.getAttribLocation(prog, name);

  const lightColor:   Array<WebGLUniformLocation | null> = [];
  const lightPos:     Array<WebGLUniformLocation | null> = [];
  const lightData:    Array<WebGLUniformLocation | null> = [];
  const lightData2:   Array<WebGLUniformLocation | null> = [];
  const lightData3:   Array<WebGLUniformLocation | null> = [];
  const lightProperties: Array<WebGLUniformLocation | null> = [];

  for (let i = 0; i < cfg.maxLights; i++) {
    lightColor.push(u(`lightColor[${i}]`));
    lightPos.push(u(`lightPos[${i}]`));
    lightData.push(u(`lightData[${i}]`));
    lightData2.push(u(`lightData2[${i}]`));
    lightData3.push(u(`lightData3[${i}]`));
    lightProperties.push(u(`lightProperties[${i}]`));
  }

  const shadowMatrix:   Array<WebGLUniformLocation | null> = [];
  const shadowSize:     Array<WebGLUniformLocation | null> = [];
  const shadowLightPos: Array<WebGLUniformLocation | null> = [];

  for (let i = 0; i < cfg.maxShadows; i++) {
    shadowMatrix.push(u(`uShadowMatrix[${i}]`));
    shadowSize.push(u(`uShadowSize[${i}]`));
    shadowLightPos.push(u(`uShadowLightPos[${i}]`));
  }

  return {
    modelMatrix:       u('modelMatrix'),
    modelViewMatrix:   u('modelViewMatrix'),
    viewMatrix:        u('viewMatrix'),
    projectionMatrix:  u('projectionMatrix'),
    normalMatrix:      u('normalMatrix'),
    cameraPosition:    u('cameraPosition'),
    mirrorMatrix:      u('uMirrorMatrix'),
    tBaseColor:        u('tBaseColor'),
    tMRO:              u('tMRO'),
    tNormal:           u('tNormal'),
    tEnv:              u('tEnv'),
    tBRDFLut:          u('tBRDFLut'),
    tMatcap:           u('tMatcap'),
    tLTC1:             u('tLTC1'),
    tLTC2:             u('tLTC2'),
    tMirrorReflection: u('tMirrorReflection'),
    uAlbedo:           u('uAlbedo'),
    uMetallic:         u('uMetallic'),
    uRoughness:        u('uRoughness'),
    uAOIntensity:      u('uAOIntensity'),
    uNormalIntensity:  u('uNormalIntensity'),
    uEnvIntensity:     u('uEnvIntensity'),
    uMatcapWeight:     u('uMatcapWeight'),
    uMirrorWeight:     u('uMirrorWeight'),
    uEmissive:         u('uEmissive'),
    uEmissiveIntensity:u('uEmissiveIntensity'),
    uExposure:         u('uExposure'),
    uEnvRGBM:          u('uEnvRGBM'),
    uTiling:           u('uTiling'),
    uOffset:           u('uOffset'),
    uUseBaseColorMap:  u('uUseBaseColorMap'),
    uUseMROMap:        u('uUseMROMap'),
    uUseNormalMap:     u('uUseNormalMap'),
    uUseEnvMap:        u('uUseEnvMap'),
    uUseMatcap:        u('uUseMatcap'),
    uShadowBias:       u('uShadowBias'),
    shadowMatrix,
    shadowSize,
    shadowLightPos,
    lightColor,
    lightPos,
    lightData,
    lightData2,
    lightData3,
    lightProperties,
    aPosition: a('position'),
    aNormal:   a('normal'),
    aUV:       a('uv'),
    aTangent:  a('tangent'),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// § 11  Light & Shadow Factory Helpers (re-exported)
// ─────────────────────────────────────────────────────────────────────────────

/** Create a directional light (AT type 1). */
export function makeDirectionalLight(
  direction: [number, number, number],
  color: [number, number, number] = [1, 1, 1],
  intensity = 1.0,
  minFloor = 0.0,
): PBRLight {
  return {
    type: 1,
    position: direction,
    color,
    intensity,
    range: 99999,
    minFloor,
    data:  [0, 0, 0, 0],
    data2: [0, 0, 0, 0],
    data3: [0, 0, 0, 0],
  };
}

/** Create a point light (AT type 2). */
export function makePointLight(
  position: [number, number, number],
  color: [number, number, number] = [1, 1, 1],
  intensity = 1.0,
  range = 20.0,
  minFloor = 0.0,
): PBRLight {
  return {
    type: 2,
    position,
    color,
    intensity,
    range,
    minFloor,
    data:  [0, 0, 0, 0],
    data2: [0, 0, 0, 0],
    data3: [0, 0, 0, 0],
  };
}

/** Create a cone/spot light (AT type 3). */
export function makeConeLight(
  position: [number, number, number],
  direction: [number, number, number],
  color: [number, number, number] = [1, 1, 1],
  intensity = 1.0,
  range = 15.0,
  coneAngleDeg = 30.0,
  feather = 1.0,
  minFloor = 0.0,
): PBRLight {
  return {
    type: 3,
    position,
    color,
    intensity,
    range,
    minFloor,
    data:  [direction[0], direction[1], direction[2], coneAngleDeg],
    data2: [feather, 0, 0, 0],
    data3: [0, 0, 0, 0],
  };
}

/** Create an LTC area light (AT type 4). Requires ltc1/ltc2 textures. */
export function makeAreaLight(
  position: [number, number, number],
  halfWidth: [number, number, number],
  halfHeight: [number, number, number],
  color: [number, number, number] = [1, 1, 1],
  intensity = 1.0,
  range = 15.0,
  roughness = 0.5,
): PBRLight {
  return {
    type: 4,
    position,
    color,
    intensity,
    range,
    minFloor: 0,
    data:  [position[0], position[1], position[2], roughness],
    data2: [halfWidth[0], halfWidth[1], halfWidth[2], 0],
    data3: [halfHeight[0], halfHeight[1], halfHeight[2], 0],
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// § 12  ATFullPBRPipeline — Main Class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * ATFullPBRPipeline
 *
 * Complete PBR material pipeline ported from ActiveTheory's shader system.
 *
 * Integrates:
 *   - 4 light types (directional, point, cone, LTC area) from lighting.fs
 *   - PCSS/PCF shadows from ShadowDepth.fs
 *   - Cook-Torrance BRDF with IBL from env1.ktx2
 *   - Matcap overlay from matcap-test.ktx2 / matcap3.ktx2
 *   - Planar mirror reflection from BasicMirror.fs
 *   - Tone mapping (Uncharted2) + gamma correction
 *
 * All GLSL is self-contained inline (no external shader file loading required).
 */
export class ATFullPBRPipeline {
  // ── Public readonly state ──────────────────────────────────────────────────
  readonly config: Readonly<Required<ATFullPBRPipelineConfig>>;
  readonly vertexSource: string;
  readonly fragmentSource: string;

  // ── Private GL state ───────────────────────────────────────────────────────
  private readonly gl: WebGLRenderingContext;
  private readonly program: WebGLProgram;
  private readonly loc: ProgramLocations;

  /** Texture unit counter — reset each draw call */
  private _texUnit = 0;

  /** Current material params (latches until changed) */
  private _material: PBRMaterialParams = { ...DEFAULT_MATERIAL };

  // ── Private constructor ────────────────────────────────────────────────────

  private constructor(
    gl: WebGLRenderingContext,
    program: WebGLProgram,
    loc: ProgramLocations,
    cfg: Required<ATFullPBRPipelineConfig>,
    vertSrc: string,
    fragSrc: string,
  ) {
    this.gl = gl;
    this.program = program;
    this.loc = loc;
    this.config = cfg;
    this.vertexSource = vertSrc;
    this.fragmentSource = fragSrc;
  }

  // ── Factory ────────────────────────────────────────────────────────────────

  /**
   * Compile and link the full PBR pipeline shaders.
   *
   * May throw if shader compilation fails — check browser console for GLSL errors.
   */
  static create(
    gl: WebGLRenderingContext,
    config?: ATFullPBRPipelineConfig,
  ): ATFullPBRPipeline {
    const cfg: Required<ATFullPBRPipelineConfig> = { ...DEFAULT_CONFIG, ...config };

    const vertSrc = buildFullVertexShader(cfg);
    const fragSrc = buildFullFragmentShader(cfg);

    const program = linkProgram(gl, vertSrc, fragSrc);
    const loc = cacheLocations(gl, program, cfg);

    return new ATFullPBRPipeline(gl, program, loc, cfg, vertSrc, fragSrc);
  }

  // ── Material API ───────────────────────────────────────────────────────────

  /**
   * Update PBR material parameters.
   * Changes are buffered and applied on the next draw call.
   */
  setMaterial(params: Partial<PBRMaterialParams>): this {
    Object.assign(this._material, params);
    return this;
  }

  /** Reset material to AT defaults. */
  resetMaterial(): this {
    this._material = { ...DEFAULT_MATERIAL };
    return this;
  }

  // ── Render API ─────────────────────────────────────────────────────────────

  /**
   * Bind the pipeline program. Call before setting uniforms manually if needed.
   */
  use(): this {
    this.gl.useProgram(this.program);
    this._texUnit = 0;
    return this;
  }

  /**
   * Upload all material uniforms + bind textures + bind lights + bind shadows,
   * then draw the indexed mesh.
   *
   * @param vao — object with bound VBOs for position/normal/uv/tangent attributes
   * @param indexBuffer — WebGL element array buffer
   * @param indexCount — number of indices to draw
   * @param matrices — transform matrices
   * @param textures — PBR texture set
   * @param lights — array of up to maxLights light descriptors
   * @param shadows — array of up to maxShadows shadow map configs
   * @param mirror — optional mirror plane config
   */
  drawMesh(
    vao: {
      position: WebGLBuffer;
      normal?: WebGLBuffer;
      uv?: WebGLBuffer;
      tangent?: WebGLBuffer;
    },
    indexBuffer: WebGLBuffer,
    indexCount: number,
    matrices: PBRMatrices,
    textures: Partial<PBRTextures> = {},
    lights: PBRLight[] = [],
    shadows: PBRShadow[] = [],
    mirror?: MirrorConfig,
  ): void {
    const { gl, loc } = this;

    gl.useProgram(this.program);
    this._texUnit = 0;

    // ── Matrices ──
    this._uploadMatrices(matrices);

    // ── Material ──
    this._uploadMaterial();

    // ── Textures ──
    this._bindTextures(textures);

    // ── Lights ──
    this._uploadLights(lights);

    // ── Shadows ──
    this._uploadShadows(shadows);

    // ── Mirror ──
    if (mirror && this.config.enableMirror && loc.mirrorMatrix) {
      gl.uniformMatrix4fv(loc.mirrorMatrix, false, mirror.matrix);
    }

    // ── Attributes ──
    this._bindAttributes(vao);

    // ── Draw ──
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_SHORT, 0);
  }

  // ── Private: Upload Helpers ────────────────────────────────────────────────

  private _uploadMatrices(m: PBRMatrices): void {
    const { gl, loc } = this;
    if (loc.modelMatrix)      gl.uniformMatrix4fv(loc.modelMatrix, false, m.model);
    if (loc.viewMatrix)       gl.uniformMatrix4fv(loc.viewMatrix, false, m.view);
    if (loc.projectionMatrix) gl.uniformMatrix4fv(loc.projectionMatrix, false, m.projection);
    if (loc.modelViewMatrix)  gl.uniformMatrix4fv(loc.modelViewMatrix, false, m.modelView);
    if (loc.normalMatrix)     gl.uniformMatrix3fv(loc.normalMatrix, false, m.normal);
    if (loc.cameraPosition)   gl.uniform3fv(loc.cameraPosition, m.cameraPosition);
  }

  private _uploadMaterial(): void {
    const { gl, loc } = this;
    const mat = this._material;

    if (loc.uAlbedo)           gl.uniform3fv(loc.uAlbedo, mat.albedo);
    if (loc.uMetallic)         gl.uniform1f(loc.uMetallic, mat.metallic);
    if (loc.uRoughness)        gl.uniform1f(loc.uRoughness, mat.roughness);
    if (loc.uAOIntensity)      gl.uniform1f(loc.uAOIntensity, mat.aoIntensity);
    if (loc.uNormalIntensity)  gl.uniform1f(loc.uNormalIntensity, mat.normalIntensity);
    if (loc.uEnvIntensity)     gl.uniform1f(loc.uEnvIntensity, mat.envIntensity);
    if (loc.uMatcapWeight)     gl.uniform1f(loc.uMatcapWeight, mat.matcapWeight);
    if (loc.uMirrorWeight)     gl.uniform1f(loc.uMirrorWeight, mat.mirrorWeight);
    if (loc.uEmissive)         gl.uniform3fv(loc.uEmissive, mat.emissive);
    if (loc.uEmissiveIntensity)gl.uniform1f(loc.uEmissiveIntensity, mat.emissiveIntensity);
    if (loc.uExposure)         gl.uniform1f(loc.uExposure, mat.exposure);
    if (loc.uEnvRGBM)          gl.uniform1f(loc.uEnvRGBM, mat.envRGBM ? 1.0 : 0.0);
    if (loc.uTiling)           gl.uniform2fv(loc.uTiling, mat.tiling);
    if (loc.uOffset)           gl.uniform2fv(loc.uOffset, mat.offset);
  }

  private _bindTex(loc: WebGLUniformLocation | null, tex: WebGLTexture | null | undefined): void {
    const { gl } = this;
    const unit = this._texUnit++;
    if (loc === null) return;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex ?? null);
    gl.uniform1i(loc, unit);
  }

  private _bindTextures(t: Partial<PBRTextures>): void {
    const { loc, gl } = this;

    this._bindTex(loc.tBaseColor, t.baseColor);
    this._bindTex(loc.tMRO, t.mro);
    this._bindTex(loc.tNormal, t.normal);
    this._bindTex(loc.tEnv, t.env);
    this._bindTex(loc.tBRDFLut, t.brdfLut);
    this._bindTex(loc.tMatcap, t.matcap);
    this._bindTex(loc.tLTC1, t.ltc1);
    this._bindTex(loc.tLTC2, t.ltc2);
    if (this.config.enableMirror) {
      this._bindTex(loc.tMirrorReflection, t.mirrorReflection);
    }

    // Enable flags based on whether textures are provided
    if (loc.uUseBaseColorMap) gl.uniform1f(loc.uUseBaseColorMap, t.baseColor ? 1.0 : 0.0);
    if (loc.uUseMROMap)       gl.uniform1f(loc.uUseMROMap,       t.mro      ? 1.0 : 0.0);
    if (loc.uUseNormalMap)    gl.uniform1f(loc.uUseNormalMap,    t.normal   ? 1.0 : 0.0);
    if (loc.uUseEnvMap)       gl.uniform1f(loc.uUseEnvMap,
      (t.env && this.config.enableIBL) ? 1.0 : 0.0);
    if (loc.uUseMatcap)       gl.uniform1f(loc.uUseMatcap,
      (t.matcap && this.config.enableMatcap) ? 1.0 : 0.0);
  }

  private _uploadLights(lights: PBRLight[]): void {
    const { gl, loc } = this;
    const n = Math.min(lights.length, this.config.maxLights);

    for (let i = 0; i < n; i++) {
      const l = lights[i];
      if (loc.lightColor[i])     gl.uniform4f(loc.lightColor[i]!,     l.color[0],    l.color[1],    l.color[2],    1.0);
      if (loc.lightPos[i])       gl.uniform4f(loc.lightPos[i]!,       l.position[0], l.position[1], l.position[2], 0.0);
      if (loc.lightData[i])      gl.uniform4f(loc.lightData[i]!,      l.data[0],     l.data[1],     l.data[2],     l.data[3]);
      if (loc.lightData2[i])     gl.uniform4f(loc.lightData2[i]!,     l.data2[0],    l.data2[1],    l.data2[2],    l.data2[3]);
      if (loc.lightData3[i])     gl.uniform4f(loc.lightData3[i]!,     l.data3[0],    l.data3[1],    l.data3[2],    l.data3[3]);
      if (loc.lightProperties[i]) {
        gl.uniform4f(
          loc.lightProperties[i]!,
          l.intensity,
          l.range,
          l.minFloor,
          l.type,   // .w = type ID
        );
      }
    }

    // Zero-out unused light slots → disabled (properties.w = 0)
    for (let i = n; i < this.config.maxLights; i++) {
      if (loc.lightProperties[i]) {
        gl.uniform4f(loc.lightProperties[i]!, 0, 0, 0, 0);
      }
    }
  }

  private _uploadShadows(shadows: PBRShadow[]): void {
    const { gl, loc } = this;
    const n = Math.min(shadows.length, this.config.maxShadows);

    // Global bias
    if (loc.uShadowBias) {
      const bias = shadows.length > 0 ? shadows[0].bias : 0.005;
      gl.uniform1f(loc.uShadowBias, bias);
    }

    for (let i = 0; i < n; i++) {
      const s = shadows[i];
      // Bind shadow map texture
      this._bindTex(
        this.gl.getUniformLocation(this.program, `shadowMap_${i}`),
        s.map,
      );
      if (loc.shadowMatrix[i])   gl.uniformMatrix4fv(loc.shadowMatrix[i]!, false, s.matrix);
      if (loc.shadowSize[i])     gl.uniform1f(loc.shadowSize[i]!, s.size);
      if (loc.shadowLightPos[i]) gl.uniform3fv(loc.shadowLightPos[i]!, s.lightPos);
    }
  }

  private _bindAttributes(vao: {
    position: WebGLBuffer;
    normal?: WebGLBuffer;
    uv?: WebGLBuffer;
    tangent?: WebGLBuffer;
  }): void {
    const { gl, loc } = this;

    // Position (required)
    if (loc.aPosition >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, vao.position);
      gl.enableVertexAttribArray(loc.aPosition);
      gl.vertexAttribPointer(loc.aPosition, 3, gl.FLOAT, false, 0, 0);
    }

    // Normal
    if (loc.aNormal >= 0 && vao.normal) {
      gl.bindBuffer(gl.ARRAY_BUFFER, vao.normal);
      gl.enableVertexAttribArray(loc.aNormal);
      gl.vertexAttribPointer(loc.aNormal, 3, gl.FLOAT, false, 0, 0);
    }

    // UV
    if (loc.aUV >= 0 && vao.uv) {
      gl.bindBuffer(gl.ARRAY_BUFFER, vao.uv);
      gl.enableVertexAttribArray(loc.aUV);
      gl.vertexAttribPointer(loc.aUV, 2, gl.FLOAT, false, 0, 0);
    }

    // Tangent (for TBN — optional)
    if (loc.aTangent >= 0 && vao.tangent) {
      gl.bindBuffer(gl.ARRAY_BUFFER, vao.tangent);
      gl.enableVertexAttribArray(loc.aTangent);
      gl.vertexAttribPointer(loc.aTangent, 4, gl.FLOAT, false, 0, 0);
    }
  }

  // ── Convenience: Bind Only (no draw) ──────────────────────────────────────

  /** Bind lights without a full draw call. Useful for multi-pass setups. */
  bindLights(lights: PBRLight[]): this {
    this.gl.useProgram(this.program);
    this._uploadLights(lights);
    return this;
  }

  /** Bind shadow maps without a full draw call. */
  bindShadows(shadows: PBRShadow[]): this {
    this.gl.useProgram(this.program);
    this._uploadShadows(shadows);
    return this;
  }

  /** Bind textures only. Useful when texture set changes but material doesn't. */
  bindTextures(textures: Partial<PBRTextures>): this {
    this.gl.useProgram(this.program);
    this._texUnit = 0;
    this._bindTextures(textures);
    return this;
  }

  /** Set transform matrices directly without a draw call. */
  setMatrices(matrices: PBRMatrices): this {
    this.gl.useProgram(this.program);
    this._uploadMatrices(matrices);
    return this;
  }

  // ── Introspection ──────────────────────────────────────────────────────────

  /** Get the underlying WebGL program. */
  get glProgram(): WebGLProgram {
    return this.program;
  }

  /** Get cached uniform location by name. Useful for custom uniforms. */
  getUniformLocation(name: string): WebGLUniformLocation | null {
    return this.gl.getUniformLocation(this.program, name);
  }

  /** Dump vertex + fragment shader sources (for debugging). */
  dumpShaders(): { vertex: string; fragment: string } {
    return { vertex: this.vertexSource, fragment: this.fragmentSource };
  }

  // ── Dispose ────────────────────────────────────────────────────────────────

  /** Release the WebGL program. */
  dispose(): void {
    this.gl.deleteProgram(this.program);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 13  Convenience Re-exports
// ─────────────────────────────────────────────────────────────────────────────

export { DEFAULT_MATERIAL as DEFAULT_PBR_MATERIAL };
export { DEFAULT_CONFIG as DEFAULT_PBR_PIPELINE_CONFIG };

/** 1×1 white fallback texture (RGBA=255,255,255,255) for tBaseColor / tMRO. */
export function createFallbackTexture(gl: WebGLRenderingContext): WebGLTexture {
  return _mkTex(gl, new Uint8Array([255,255,255,255]));
}

/** 1×1 flat normal-map fallback (RGBA=128,128,255,255 → tangent (0,0,1)). */
export function createFlatNormalTexture(gl: WebGLRenderingContext): WebGLTexture {
  return _mkTex(gl, new Uint8Array([128,128,255,255]));
}

function _mkTex(gl: WebGLRenderingContext, px: Uint8Array): WebGLTexture {
  const t=gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D,t);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,px);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.bindTexture(gl.TEXTURE_2D,null);
  return t;
}

/** Simple 64×64 BRDF integration LUT fallback (analytical approximation of Karis 2013). */
export function createBRDFLutFallback(gl: WebGLRenderingContext, size = 64): WebGLTexture {
  const pixels = new Uint8Array(size * size * 4);
  for (let y=0;y<size;y++) for (let x=0;x<size;x++) {
    const NdotV=(x+0.5)/size, r=(y+0.5)/size, r2=r*r;
    const idx=(y*size+x)*4;
    pixels[idx]  =Math.round((1.0-0.5*r2/(r2+0.33))*255);
    pixels[idx+1]=Math.round((0.04*NdotV*r2/(r2+0.09))*255);
    pixels[idx+2]=0; pixels[idx+3]=255;
  }
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, size, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

/**
 * Build a matrix-math helper for constructing PBRMatrices.
 * Column-major Float32Array[16] (WebGL convention).
 */
export const PBRMath = {
  identity(): Float32Array { const m=new Float32Array(16); m[0]=m[5]=m[10]=m[15]=1; return m; },
  identity3(): Float32Array { const m=new Float32Array(9); m[0]=m[4]=m[8]=1; return m; },

  multiply(a: Float32Array, b: Float32Array): Float32Array {
    const o=new Float32Array(16);
    for (let i=0;i<4;i++) for (let j=0;j<4;j++) { let s=0; for (let k=0;k<4;k++) s+=a[i+k*4]*b[k+j*4]; o[i+j*4]=s; }
    return o;
  },

  perspective(fovY: number, aspect: number, near: number, far: number): Float32Array {
    const o=new Float32Array(16), f=1/Math.tan(fovY*0.5), nf=1/(near-far);
    o[0]=f/aspect; o[5]=f; o[10]=(far+near)*nf; o[11]=-1; o[14]=2*far*near*nf; return o;
  },

  lookAt(ex:number,ey:number,ez:number,cx:number,cy:number,cz:number,ux:number,uy:number,uz:number): Float32Array {
    let fx=cx-ex,fy=cy-ey,fz=cz-ez,len=Math.sqrt(fx*fx+fy*fy+fz*fz);
    if(len>1e-6){fx/=len;fy/=len;fz/=len;}
    let rx=uy*fz-uz*fy,ry=uz*fx-ux*fz,rz=ux*fy-uy*fx;
    len=Math.sqrt(rx*rx+ry*ry+rz*rz); if(len>1e-6){rx/=len;ry/=len;rz/=len;}
    const upx=fy*rz-fz*ry,upy=fz*rx-fx*rz,upz=fx*ry-fy*rx;
    const o=new Float32Array(16);
    o[0]=rx;o[1]=upx;o[2]=-fx; o[4]=ry;o[5]=upy;o[6]=-fy; o[8]=rz;o[9]=upz;o[10]=-fz;
    o[12]=-(rx*ex+ry*ey+rz*ez); o[13]=-(upx*ex+upy*ey+upz*ez); o[14]=(fx*ex+fy*ey+fz*ez); o[15]=1;
    return o;
  },

  /** Upper-left 3×3 normal matrix (transpose-inverse) from modelView Float32Array[16] → Float32Array[9] */
  normalMatrix(mv: Float32Array): Float32Array {
    const a00=mv[0],a01=mv[1],a02=mv[2],a10=mv[4],a11=mv[5],a12=mv[6],a20=mv[8],a21=mv[9],a22=mv[10];
    const b01=a22*a11-a12*a21,b11=-a22*a10+a12*a20,b21=a21*a10-a11*a20;
    const det=a00*b01+a01*b11+a02*b21, id=det===0?0:1/det;
    return new Float32Array([
      b01*id, (-a22*a01+a02*a21)*id, (a12*a01-a02*a11)*id,
      b11*id,  (a22*a00-a02*a20)*id, (-a12*a00+a02*a10)*id,
      b21*id, (-a21*a00+a01*a20)*id, (a11*a00-a01*a10)*id,
    ]);
  },
} as const;
