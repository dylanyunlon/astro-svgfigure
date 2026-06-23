/**
 * PBRMaterial.ts — 物理基础渲染材质
 *
 * 对应 AT 中的 PBR / ATPBR / RoomPBR shader class（15 个材质类中的核心）。
 *
 * AT 参数映射（来自 at_uil_categorized.json material，270 个参数）：
 *   uMRON        [metallic, roughness, occlusionStrength, normalScale]
 *   uEnv         [envDiffuseIntensity, envSpecularIntensity, envOffset?]
 *   uTint        hex color → albedo tint
 *   uTiling      [u, v] UV tiling
 *   uUseLightmap 0/1
 *   uUseLinearOutput 0/1
 *   _tx_tBaseColor   BaseColor/Albedo texture
 *   _tx_tMRO         Metallic-Roughness-Occlusion packed texture (AT 格式)
 *   _tx_tNormal      Normal map
 *   _tx_tEnvDiffuse  Environment diffuse (irradiance) cube / equirectangular
 *   _tx_tEnvSpecular Environment specular (prefiltered) texture
 *   _tx_tLightmap    Optional lightmap
 *
 * BRDF: Cook-Torrance (GGX NDF + Smith G + Schlick F)，与 AT 的 PBR 核心对齐。
 */

import { Material } from './Material';
import type { AstroProgram } from '../AstroProgram';

// ── GLSL sources ─────────────────────────────────────────────────────────────

export const PBR_VERT_SRC = /* glsl */`#version 300 es
precision highp float;

in vec3 aPosition;
in vec3 aNormal;
in vec2 aUv;
in vec4 aTangent;

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat3 uNormalMatrix;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUv;
out mat3 vTBN;

void main() {
  vec4 worldPos = uModelMatrix * vec4(aPosition, 1.0);
  vWorldPos     = worldPos.xyz;
  vNormal       = normalize(uNormalMatrix * aNormal);
  vUv           = aUv;

  // TBN for normal mapping
  vec3 T = normalize(uNormalMatrix * aTangent.xyz);
  vec3 N = vNormal;
  T = normalize(T - dot(T, N) * N);
  vec3 B = cross(N, T) * aTangent.w;
  vTBN = mat3(T, B, N);

  gl_Position = uProjectionMatrix * uViewMatrix * worldPos;
}
`;

export const PBR_FRAG_SRC = /* glsl */`#version 300 es
precision highp float;

// ── AT uniform 命名规范 ───────────────────────────────────────────────────────

// MRON: Metallic · Roughness · Occlusion · NormalScale
uniform vec4  uMRON;          // [metallic, roughness, occlusionStrength, normalScale]
uniform vec3  uEnv;           // [diffuseIntensity, specularIntensity, unused]
uniform vec3  uTint;          // albedo tint (sRGB)
uniform vec2  uTiling;        // UV tiling
uniform float uUseLightmap;
uniform float uUseLinearOutput;

// Textures (AT _tx_ prefix convention)
uniform sampler2D tBaseColor;    // _tx_tBaseColor
uniform sampler2D tMRO;          // _tx_tMRO (Metallic=R, Roughness=G, Occlusion=B)
uniform sampler2D tNormal;       // _tx_tNormal
uniform sampler2D tEnvDiffuse;   // _tx_tEnvDiffuse (equirectangular diffuse)
uniform sampler2D tEnvSpecular;  // _tx_tEnvSpecular (prefiltered specular)
uniform sampler2D tLightmap;     // _tx_tLightmap (optional)

// Camera
uniform vec3  uCameraPos;

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUv;
in mat3 vTBN;

out vec4 fragColor;

// ── Constants ────────────────────────────────────────────────────────────────
const float PI = 3.14159265358979;

// ── Utility ──────────────────────────────────────────────────────────────────

vec3 sRGBToLinear(vec3 c) {
  return pow(max(c, vec3(0.0)), vec3(2.2));
}

vec3 linearToSRGB(vec3 c) {
  return pow(max(c, vec3(0.0)), vec3(1.0 / 2.2));
}

// Equirectangular UV from direction
vec2 dirToEquirect(vec3 dir) {
  float phi   = atan(dir.z, dir.x);
  float theta = acos(clamp(dir.y, -1.0, 1.0));
  return vec2(phi / (2.0 * PI) + 0.5, theta / PI);
}

// ── Cook-Torrance BRDF ───────────────────────────────────────────────────────

// GGX / Trowbridge-Reitz NDF
float D_GGX(float NdotH, float roughness) {
  float a  = roughness * roughness;
  float a2 = a * a;
  float d  = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

// Smith's geometry term (GGX)
float G_SmithGGX(float NdotV, float NdotL, float roughness) {
  float r  = roughness + 1.0;
  float k  = (r * r) / 8.0;
  float gV = NdotV / (NdotV * (1.0 - k) + k);
  float gL = NdotL / (NdotL * (1.0 - k) + k);
  return gV * gL;
}

// Schlick Fresnel
vec3 F_Schlick(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Schlick-Roughness Fresnel (for IBL specular)
vec3 F_SchlickRoughness(float cosTheta, vec3 F0, float roughness) {
  return F0 + (max(vec3(1.0 - roughness), F0) - F0) *
         pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ── IBL (AT env diffuse + specular 对齐) ────────────────────────────────────

vec3 sampleEnvDiffuse(vec3 N) {
  vec2 uv = dirToEquirect(N);
  return sRGBToLinear(texture(tEnvDiffuse, uv).rgb);
}

vec3 sampleEnvSpecular(vec3 R, float roughness) {
  // AT 用 prefiltered mip → 我们用 roughness → lod 近似
  float lod = roughness * 8.0;
  vec2 uv = dirToEquirect(R);
  // WebGL 2 textureLod is available for sampler2D
  return sRGBToLinear(textureLod(tEnvSpecular, uv, lod).rgb);
}

// ── Main ─────────────────────────────────────────────────────────────────────

void main() {
  vec2 uv = vUv * uTiling;

  // ── Sample textures ────────────────────────────────────────────────────────
  vec4  baseColorSample = texture(tBaseColor, uv);
  vec3  mroSample       = texture(tMRO,       uv).rgb;
  vec3  normalSample    = texture(tNormal,    uv).rgb * 2.0 - 1.0;
  vec3  lightmapSample  = texture(tLightmap,  uv).rgb;

  // ── Material params (AT uMRON 格式) ─────────────────────────────────────────
  float metallic       = mroSample.r * uMRON.x;
  float roughness      = clamp(mroSample.g * uMRON.y, 0.04, 1.0);
  float occlusion      = mix(1.0, mroSample.b, uMRON.z);
  float normalScale    = uMRON.w;

  // ── Albedo ────────────────────────────────────────────────────────────────
  vec3 albedo = sRGBToLinear(baseColorSample.rgb) * sRGBToLinear(uTint);
  float alpha  = baseColorSample.a;

  // ── Normal (TBN) ──────────────────────────────────────────────────────────
  vec3 N = normalize(vTBN * (normalSample * vec3(normalScale, normalScale, 1.0)));
  vec3 V = normalize(uCameraPos - vWorldPos);
  vec3 R = reflect(-V, N);

  float NdotV = max(dot(N, V), 0.0001);

  // ── F0 ────────────────────────────────────────────────────────────────────
  vec3 F0 = mix(vec3(0.04), albedo, metallic);

  // ── IBL Diffuse ───────────────────────────────────────────────────────────
  vec3 irradiance = sampleEnvDiffuse(N) * uEnv.x;
  vec3 kS         = F_SchlickRoughness(NdotV, F0, roughness);
  vec3 kD         = (1.0 - kS) * (1.0 - metallic);
  vec3 diffuse    = kD * albedo * irradiance * occlusion;

  // ── IBL Specular ──────────────────────────────────────────────────────────
  vec3 prefilteredColor = sampleEnvSpecular(R, roughness) * uEnv.y;
  // AT approximates split-sum via single prefiltered sample
  vec3 specular = prefilteredColor * F_SchlickRoughness(NdotV, F0, roughness);

  // ── Lightmap (optional) ───────────────────────────────────────────────────
  vec3 lightmapContrib = mix(vec3(1.0), sRGBToLinear(lightmapSample), uUseLightmap);

  // ── Combine ───────────────────────────────────────────────────────────────
  vec3 color = (diffuse + specular) * lightmapContrib;

  // ── Tone / output ─────────────────────────────────────────────────────────
  if (uUseLinearOutput < 0.5) {
    color = linearToSRGB(color);
  }

  fragColor = vec4(color, alpha);
}
`;

// ── PBRMaterial ──────────────────────────────────────────────────────────────

export interface PBRMaterialOptions {
  /** uMRON.x — AT packed MRO texture metallic multiplier [0–1] */
  metallic?: number;
  /** uMRON.y — AT packed MRO texture roughness multiplier [0–1] */
  roughness?: number;
  /** uMRON.z — Ambient occlusion strength */
  ao?: number;
  /** uMRON.w — Normal map scale */
  normalScale?: number;
  /** uEnv — [diffuseIntensity, specularIntensity] */
  envIntensity?: [number, number];
  /** uTint — Albedo tint color (CSS hex or [r,g,b] 0-1) */
  albedo?: string | [number, number, number];
  /** uTiling — UV tiling */
  tiling?: [number, number];
  /** uUseLightmap */
  useLightmap?: boolean;
  /** IOR → used to derive F0 for non-metallic surfaces */
  ior?: number;
}

export class PBRMaterial extends Material {
  // AT shader class names that this material corresponds to
  static readonly AT_CLASS_NAMES = ['PBR', 'ATPBR', 'RoomPBR'] as const;

  // Texture slots (AT _tx_ convention)
  tBaseColor:   WebGLTexture | null = null;
  tMRO:         WebGLTexture | null = null;
  tNormal:      WebGLTexture | null = null;
  tEnvDiffuse:  WebGLTexture | null = null;
  tEnvSpecular: WebGLTexture | null = null;
  tLightmap:    WebGLTexture | null = null;

  constructor(options: PBRMaterialOptions = {}) {
    super('PBRMaterial');

    const {
      metallic     = 1.0,
      roughness    = 0.5,
      ao           = 1.0,
      normalScale  = 1.0,
      envIntensity = [1.0, 1.0],
      albedo       = '#ffffff',
      tiling       = [1, 1],
      useLightmap  = false,
      ior          = 1.5,
    } = options;

    // uMRON: [metallic, roughness, occlusionStrength, normalScale]
    this.setUniform('uMRON', [metallic, roughness, ao, normalScale]);

    // uEnv: [diffuseIntensity, specularIntensity]
    this.setUniform('uEnv', [...envIntensity, 0]);

    // uTint: albedo color (sRGB)
    if (typeof albedo === 'string') {
      this.setUniform('uTint', hexToRGB(albedo));
    } else {
      this.setUniform('uTint', albedo);
    }

    this.setUniform('uTiling',          tiling);
    this.setUniform('uUseLightmap',     useLightmap ? 1 : 0);
    this.setUniform('uUseLinearOutput', 0);

    // IOR → derive dielectric F0: ((ior-1)/(ior+1))^2
    const f0 = Math.pow((ior - 1) / (ior + 1), 2);
    this.setUniform('uIOR', f0); // stored for reference
  }

  /**
   * 从 AT UIL categorized.json 的 material 块中导入 PBR shader 参数。
   * 期望传入已剥离 'ATPBR/ATPBR/Element_N_scene/' 前缀的 flat 参数对象。
   *
   * 示例参数（来自 at_uil_categorized.json）：
   *   uMRON: [1, 1.3, 1, 1]
   *   uEnv:  [1.5, 1]
   *   uTint: '#e5f1ff'
   */
  importFromATPBRParams(params: Record<string, unknown>): this {
    if (params['uMRON'] !== undefined) {
      this.setUniform('uMRON', params['uMRON'] as number[]);
    }
    if (params['uEnv'] !== undefined) {
      const env = params['uEnv'] as number[];
      this.setUniform('uEnv', env.length >= 3 ? env : [...env, 0]);
    }
    if (params['uTint'] !== undefined) {
      const tint = params['uTint'] as string;
      this.setUniform('uTint', hexToRGB(tint));
    }
    if (params['uTiling'] !== undefined) {
      this.setUniform('uTiling', params['uTiling'] as number[]);
    }
    if (params['uUseLightmap'] !== undefined) {
      this.setUniform('uUseLightmap', params['uUseLightmap'] as number);
    }
    if (params['uUseLinearOutput'] !== undefined) {
      this.setUniform('uUseLinearOutput', params['uUseLinearOutput'] as number);
    }
    return this;
  }

  /** GLSL vertex source */
  static get vertexShader(): string { return PBR_VERT_SRC; }

  /** GLSL fragment source (Cook-Torrance BRDF) */
  static get fragmentShader(): string { return PBR_FRAG_SRC; }

  override bind(gl: WebGL2RenderingContext): void {
    if (!this.program) return;

    // Parent handles blending / depth / side / scalar uniforms
    super.bind(gl);

    // Bind textures explicitly with AT slot naming
    let unit = 0;
    const bindTex = (name: string, tex: WebGLTexture | null) => {
      if (!tex || !this.program) return;
      const loc = this.program.uniformLocation(name);
      if (loc === null) return;
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(loc, unit);
      unit++;
    };

    bindTex('tBaseColor',   this.tBaseColor);
    bindTex('tMRO',         this.tMRO);
    bindTex('tNormal',      this.tNormal);
    bindTex('tEnvDiffuse',  this.tEnvDiffuse);
    bindTex('tEnvSpecular', this.tEnvSpecular);
    bindTex('tLightmap',    this.tLightmap);
  }

  override dispose(): void {
    super.dispose();
    this.tBaseColor   = null;
    this.tMRO         = null;
    this.tNormal      = null;
    this.tEnvDiffuse  = null;
    this.tEnvSpecular = null;
    this.tLightmap    = null;
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

/**
 * CSS hex → [r, g, b] in 0-1 sRGB.
 * 支持 '#rrggbb' 和 '#rgb' 格式。
 */
export function hexToRGB(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  if (h.length === 3) {
    return [
      parseInt(h[0] + h[0], 16) / 255,
      parseInt(h[1] + h[1], 16) / 255,
      parseInt(h[2] + h[2], 16) / 255,
    ];
  }
  return [
    parseInt(h.slice(0, 2), 16) / 255,
    parseInt(h.slice(2, 4), 16) / 255,
    parseInt(h.slice(4, 6), 16) / 255,
  ];
}
