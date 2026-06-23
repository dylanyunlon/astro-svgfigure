/**
 * CellMaterial.ts — Cell 专用材质
 *
 * 根据 cell species 选择不同 shader，对应 xiaodi_options_table.json 的参数：
 *   cil-eye    → 家场景 PBR + home particle shader 参数
 *   cil-bolt   → Work 场景 Chain/Spine shader 参数
 *   cil-vector → About 场景 Logo shader 参数
 *   cil-plus   → Contact 场景参数
 *   cil-arrow-right → TreeScene shader 参数
 *   cil-filter  → (default fallback)
 *   ...
 *
 * xiaodi_options_table.json 的 key = species name (cil-eye, cil-bolt…)
 * 每个 species 包含完整的 AT UIL 参数集合，供 cell pubsub loop 使用。
 *
 * 架构：CellMaterial 持有 per-species uniform 快照，在 cell 激活时
 * 通过 activateSpecies() 切换 uniform set，驱动 cell-pubsub-loop 的
 * 渲染分支。
 */

import { Material } from './Material';
import { PBRMaterial, hexToRGB } from './PBRMaterial';
import type { UniformValue } from './Material';

// ── Species type (从 xiaodi_options_table.json keys 提取) ─────────────────────

export type CellSpecies =
  | 'cil-eye'
  | 'cil-bolt'
  | 'cil-vector'
  | 'cil-plus'
  | 'cil-arrow-right'
  | 'cil-filter'
  | 'cil-code'
  | 'cil-layers'
  | 'cil-loop'
  | 'cil-graph'
  | string; // extensible

// ── AT shader class → fragment source 映射 ────────────────────────────────────

/**
 * AT 场景中各 species 使用的 shader class。
 * 来源：xiaodi_options_table.json INPUT_Config_*_shader 字段分析。
 */
export const SPECIES_SHADER_MAP: Record<string, string> = {
  'cil-eye':         'HomeParticleShader',   // home_scene
  'cil-bolt':        'ChainShader',          // Work / SpineShader
  'cil-vector':      'AboutLogoShader',      // About
  'cil-plus':        'HomeLogoShader',       // Contact (tree scene)
  'cil-arrow-right': 'TreeFBR',             // TreeScene
  'cil-filter':      'PBR',                 // CleanRoom default
  'cil-code':        'ATPBR',               // generic PBR
  'cil-layers':      'FloorShader',         // floor
  'cil-loop':        'WallShader',          // wall
  'cil-graph':       'WorkItemShader',      // work item
};

// ── Per-species uniform snapshots (from xiaodi_options_table.json) ────────────

/**
 * 从 xiaodi_options_table.json 提取的关键 AT 参数 per species。
 * 这是 cell-pubsub-loop 的核心数据驱动 uniform 集合。
 */
const SPECIES_DEFAULTS: Record<string, Partial<CellMaterialUniforms>> = {
  'cil-eye': {
    // ATPBR/ATPBR/Element_6_homeScene (platform)
    uMRON:            [1, 1.3, 1, 1],
    uEnv:             [1.5, 1, 0],
    uTint:            hexToRGB('#e5f1ff'),
    uUseLinearOutput: 0,
    // HomeParticleShader
    uFresnelStrength: 0.67,
    uSize:            2.45,
    uVideoBound:      [0.92, 0.92],
    uNormalStrength:  1.0,
  },
  'cil-bolt': {
    // ChainShader + SpineShader (Work scene)
    uMRON:            [1, 1, 1, 1],
    uEnv:             [1, 1, 0],
    uTint:            hexToRGB('#ffffff'),
    uNormalStrength:  0.19,
    uColor:           hexToRGB('#d1fff4'),
    uLight:           [1, 1, 1, 0.4],
    uReflection:      [2.7, 0.85],
    uSize:            8.56,
  },
  'cil-vector': {
    // AboutLogoShader (About scene)
    uMRON:            [1, 0.5, 1, 1],
    uEnv:             [1, 1, 0],
    uTint:            hexToRGB('#ffffff'),
    uNormalStrength:  0.24,
  },
  'cil-plus': {
    // HomeLogoShader / Contact (tree scene)
    uMRON:            [1, 1, 1, 1],
    uEnv:             [1, 1, 0],
    uTint:            hexToRGB('#ffffff'),
    uAlpha:           1.0,
  },
  'cil-arrow-right': {
    // TreeFBR (TreeScene)
    uMRON:            [1, 1, 1, 1],
    uEnv:             [1, 1, 0],
    uTint:            hexToRGB('#8b9de5'),
    uNormalStrength:  1.0,
    uColor:           hexToRGB('#8b9de5'),
    uLight:           [1.08, 0.49, 2.14, 2],
  },
  'cil-filter': {
    // PBR (CleanRoom)
    uMRON:            [1, 0.3, 1, 0.6],
    uEnv:             [1, 1, 0],
    uTint:            hexToRGB('#4f4f4f'),
    uTiling:          [1, 1],
    uUseLightmap:     0,
  },
  'cil-code': {
    // ATPBR generic
    uMRON:            [1, 0.5, 1, 1],
    uEnv:             [1.5, 1, 0],
    uTint:            hexToRGB('#e5f1ff'),
  },
  'cil-layers': {
    // FloorShader (CleanRoom floor)
    uMRON:            [1, 1, 1, 1],
    uColor:           hexToRGB('#454545'),
    uNormalStrength:  1.0,
    uMirrorStrength:  0.55,
    uLight:           [0, 1, -5.57, 0.48],
  },
  'cil-loop': {
    // WallShader (CleanRoom walls)
    uMRON:            [1, 1, 1, 1],
    uColor:           hexToRGB('#000000'),
    uNormalStrength:  1.0,
    uLight:           [-0.51, 0, 1.36, 0.45],
  },
  'cil-graph': {
    // WorkItemShader
    uMRON:            [1, 1, 1, 1],
    uEnv:             [1, 1, 0],
    uTint:            hexToRGB('#ffffff'),
    uDistortStrength: 0,
    uFresnelPow:      1,
    uRefractionRatio: 1,
  },
};

// ── CellMaterialUniforms ──────────────────────────────────────────────────────

/** 所有 cell species 可能使用的 uniform 联合集。 */
export interface CellMaterialUniforms {
  // PBR core (AT uMRON)
  uMRON:             number[];    // [metallic, roughness, occlusion, normalScale]
  uEnv:              number[];    // [diffuse, specular, offset]
  uTint:             number[];    // [r, g, b] sRGB
  uTiling:           number[];    // [u, v]
  uUseLightmap:      number;
  uUseLinearOutput:  number;

  // Surface
  uNormalStrength:   number;
  uColor:            number[];
  uAlpha:            number;
  uLight:            number[];    // AT light vec4

  // Reflection / glass
  uDistortStrength:  number;
  uFresnelPow:       number;
  uFresnelStrength:  number;
  uRefractionRatio:  number;
  uReflection:       number[];
  uFresnelColor:     number[];

  // Particle
  uSize:             number;
  uVideoBound:       number[];

  // Floor / wall
  uMirrorStrength:   number;

  // Generic
  [key: string]:     UniformValue;
}

// ── GLSL fragment sources per species group ───────────────────────────────────

/**
 * cil-eye: HomeParticleShader 风格 (home scene particles)
 * uFresnelStrength, uSize, uVideoBound, tNormal
 */
const FRAG_CIL_EYE = /* glsl */`#version 300 es
precision highp float;

uniform sampler2D tNormal;
uniform vec3  uTint;
uniform float uFresnelStrength;
uniform float uSize;
uniform vec2  uVideoBound;
uniform float uAlpha;
uniform vec3  uCameraPos;

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUv;

out vec4 fragColor;

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(uCameraPos - vWorldPos);
  float fresnel = pow(1.0 - max(dot(N, V), 0.0), uFresnelStrength * 3.0 + 1.0);
  vec3 normalSample = texture(tNormal, vUv).rgb;
  vec3 color = uTint * (0.5 + 0.5 * normalSample) * (1.0 + fresnel);
  fragColor = vec4(color, fresnel * uAlpha);
}
`;

/**
 * cil-bolt: Chain/Spine matcap + normal shader (Work scene)
 */
const FRAG_CIL_BOLT = /* glsl */`#version 300 es
precision highp float;

uniform sampler2D tMatcap;
uniform sampler2D tNormal;
uniform vec3  uColor;
uniform vec4  uLight;
uniform vec2  uReflection;
uniform float uNormalStrength;
uniform vec3  uCameraPos;

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUv;
in mat3 vTBN;

out vec4 fragColor;

void main() {
  vec3 normalTex = texture(tNormal, vUv).rgb * 2.0 - 1.0;
  normalTex.xy  *= uNormalStrength;
  vec3 N  = normalize(vTBN * normalTex);
  vec3 V  = normalize(uCameraPos - vWorldPos);

  // Matcap UV from view-space normal
  vec2 matcapUV = N.xy * 0.5 + 0.5;
  vec3 matcapColor = texture(tMatcap, matcapUV).rgb;

  // Simple phong-like light
  vec3 L        = normalize(uLight.xyz);
  float NdotL   = max(dot(N, L), 0.0);
  float spec    = pow(max(dot(reflect(-L, N), V), 0.0), uReflection.x * 32.0) * uReflection.y;

  vec3 color = uColor * (matcapColor * NdotL * uLight.w + vec3(spec));
  fragColor  = vec4(color, 1.0);
}
`;

/**
 * cil-vector: AboutLogoShader / matcap + normal (About scene)
 */
const FRAG_CIL_VECTOR = /* glsl */`#version 300 es
precision highp float;

uniform sampler2D tMap;
uniform sampler2D tNormal;
uniform float uNormalStrength;
uniform vec3  uCameraPos;

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUv;
in mat3 vTBN;

out vec4 fragColor;

void main() {
  vec3 normalTex = texture(tNormal, vUv).rgb * 2.0 - 1.0;
  normalTex.xy  *= uNormalStrength;
  vec3 N = normalize(vTBN * normalTex);

  vec2 matcapUV  = N.xy * 0.5 + 0.5;
  vec3 color     = texture(tMap, matcapUV).rgb;

  fragColor = vec4(color, 1.0);
}
`;

/**
 * Default / fallback — minimal PBR diffuse
 */
const FRAG_DEFAULT = /* glsl */`#version 300 es
precision highp float;

uniform vec3  uTint;
uniform vec4  uMRON;

in vec3 vNormal;
in vec2 vUv;

out vec4 fragColor;

void main() {
  vec3 N      = normalize(vNormal);
  float light = max(dot(N, normalize(vec3(1.0, 2.0, 1.0))), 0.0) * 0.8 + 0.2;
  vec3 color  = pow(uTint, vec3(2.2)) * light;
  fragColor   = vec4(pow(color, vec3(1.0/2.2)), 1.0);
}
`;

// ── Fragment source selector ──────────────────────────────────────────────────

export function getSpeciesFragSource(species: CellSpecies): string {
  switch (species) {
    case 'cil-eye':         return FRAG_CIL_EYE;
    case 'cil-bolt':        return FRAG_CIL_BOLT;
    case 'cil-vector':
    case 'cil-plus':        return FRAG_CIL_VECTOR;
    default:                return FRAG_DEFAULT;
  }
}

// ── Vertex shader (shared for all cell species) ───────────────────────────────

export const CELL_VERT_SRC = /* glsl */`#version 300 es
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

  vec3 T = normalize(uNormalMatrix * aTangent.xyz);
  vec3 N = vNormal;
  T = normalize(T - dot(T, N) * N);
  vec3 B = cross(N, T) * aTangent.w;
  vTBN   = mat3(T, B, N);

  gl_Position = uProjectionMatrix * uViewMatrix * worldPos;
}
`;

// ── CellMaterial ──────────────────────────────────────────────────────────────

export class CellMaterial extends Material {
  /** 当前激活的 species */
  activeSpecies: CellSpecies = 'cil-eye';

  /** Per-species uniform 快照（从 xiaodi_options_table.json 导入） */
  private _speciesUniforms: Map<string, Map<string, UniformValue>> = new Map();

  /** 对应 AT shader class name */
  get atShaderClass(): string {
    return SPECIES_SHADER_MAP[this.activeSpecies] ?? 'PBR';
  }

  /** GLSL vertex source */
  static get vertexShader(): string { return CELL_VERT_SRC; }

  /** GLSL fragment source for active species */
  get fragmentShader(): string {
    return getSpeciesFragSource(this.activeSpecies);
  }

  constructor(initialSpecies: CellSpecies = 'cil-eye') {
    super('CellMaterial');

    // Pre-populate per-species uniform snapshots from defaults
    for (const [species, defaults] of Object.entries(SPECIES_DEFAULTS)) {
      const map = new Map<string, UniformValue>();
      for (const [k, v] of Object.entries(defaults)) {
        map.set(k, v as UniformValue);
      }
      this._speciesUniforms.set(species, map);
    }

    this.activateSpecies(initialSpecies);
  }

  /**
   * 切换 species → 更新 uniform set。
   * 由 cell-pubsub-loop 在 cell 状态变化时调用。
   */
  activateSpecies(species: CellSpecies): this {
    this.activeSpecies = species;

    // Merge species-specific uniforms into current uniform map
    const specUniforms = this._speciesUniforms.get(species);
    if (specUniforms) {
      for (const [k, v] of specUniforms) {
        this.uniforms.set(k, v);
      }
    }

    return this;
  }

  /**
   * 从 xiaodi_options_table.json 的单个 species 条目导入参数。
   * 只提取 AT material/shader uniform 参数（跳过 CAMERA_, MESH_, INPUT_ 等前缀）。
   */
  importSpeciesParams(species: CellSpecies, params: Record<string, unknown>): this {
    const map = this._speciesUniforms.get(species) ?? new Map<string, UniformValue>();

    for (const [key, value] of Object.entries(params)) {
      // AT UIL parameter key 格式：ShaderClass/ShaderClass/Element_N_scene/uParam
      // 只导入 u* 参数和纯数值数组（跳过 texture 描述符和非 uniform 字段）
      const parts = key.split('/');
      const paramName = parts[parts.length - 1];

      if (!paramName.startsWith('u') && !paramName.startsWith('_tx_')) continue;
      if (typeof value !== 'number' && !Array.isArray(value) && typeof value !== 'boolean') continue;

      // Convert hex color strings embedded in arrays → skip (keep raw)
      map.set(paramName, value as UniformValue);
    }

    this._speciesUniforms.set(species, map);

    // Re-apply if this is the active species
    if (species === this.activeSpecies) {
      this.activateSpecies(species);
    }

    return this;
  }

  /**
   * 返回 PBRMaterial 作为当前 species 的 PBR 代理，
   * 用于需要完整 Cook-Torrance BRDF 的 species (cil-eye, cil-filter…)
   */
  toPBRMaterial(): PBRMaterial {
    const pbr = new PBRMaterial();
    const mron = this.uniforms.get('uMRON') as number[] | undefined;
    if (mron) {
      pbr.setUniform('uMRON', mron);
    }
    const env = this.uniforms.get('uEnv') as number[] | undefined;
    if (env) {
      pbr.setUniform('uEnv', env);
    }
    const tint = this.uniforms.get('uTint') as number[] | undefined;
    if (tint) {
      pbr.setUniform('uTint', tint);
    }
    return pbr;
  }

  override bind(gl: WebGL2RenderingContext): void {
    super.bind(gl);
    // Subclass-specific texture binding can be added per species here
  }

  override dispose(): void {
    super.dispose();
    this._speciesUniforms.clear();
  }
}

// ── Export GLSL sources for external use ─────────────────────────────────────

export { FRAG_CIL_EYE, FRAG_CIL_BOLT, FRAG_CIL_VECTOR, FRAG_DEFAULT };
