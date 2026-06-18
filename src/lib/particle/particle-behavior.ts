/**
 * particle-behavior.ts — Active Theory INPUT_P 粒子配置模式移植
 *
 * 直接移植 AT 的 INPUT_P 粒子系统配置模式，来源:
 *   https://activetheory.net/assets/data/uil.1780406240914.json
 *
 * AT 的 INPUT_P 键结构 (182 keys, 12 particle systems):
 *
 *   INPUT_P_Element_{N}_{SystemName}_config_{field}
 *     config 字段: shader, particleCount, width, height, depth, uniforms,
 *                  type ("lifecycle"), class, output, enablePhysics,
 *                  initialPositions
 *
 *   INPUT_P_Element_{N}_{SystemName}_behavior_{field}
 *     behavior 字段: codeCount (number), data (JSON array of code refs
 *                    like '["code_4","code_1","code_3","code_2"]'),
 *                    uniforms (newline-separated "name: value" pairs)
 *
 *   INPUT_P_Element_{N}_{SystemName}code_{M}_{field}
 *     code 字段: code (GLSL snippet), name, preset, uniforms, file, btn
 *     preset 类型: "curl" | "fluid" | "pointcloud" | "spline" | "planeshape"
 *
 * Uniform 值类型:
 *   number     — "uCurlNoiseSpeed: 1"
 *   color      — "uColor: #ffffff"
 *   array      — "uBox: [1, 1, 1]"
 *   texture T  — "tVelocity: T" (runtime texture reference)
 *   computed C — "uProjMatrix: Cmat4", "tFluid: Csampler2D"
 *
 * Code blocks 内部用 #require(filename) 引用共享 GLSL 库:
 *   #require(curl.glsl)
 *   #require(glscreenprojection.glsl)
 *   #require(luma.fs)
 *   #require(sdfs.glsl)
 *   #require(splineparticles.fs)
 *
 * behavior.data 控制 code block 的执行顺序 (不一定是 code_1, code_2... 的自然序)。
 * 例如 home_scene: ["code_4","code_1","code_3","code_2"]
 * 意味着先执行 code_4 (planeshape)，再 code_1 (curl)，再 code_3 (fluid)，
 * 最后 code_2 (lerp)。所有 code 片段按此顺序拼接进 GPGPU shader 的 main()。
 *
 * 本文件实现:
 *   - ParticleUniformValue   — AT uniform 值类型 (number | color | array | texture | computed)
 *   - ParticleBehaviorCode   — 单个 code block (preset + glsl + uniforms)
 *   - ParticleBehavior       — behavior 层 (执行顺序 + code blocks + shared uniforms)
 *   - ParticleConfig         — 完整粒子系统配置 (config + behavior)
 *   - parseParticleConfig()  — 从 AT INPUT_P flat-key JSON 解析出结构化配置
 *   - compileParticleShader() — 将 behavior code blocks 编译为完整 GLSL, 含 #require 替换
 */

import { CURL_NOISE_GLSL } from './CurlNoise.js';

// ── AT Uniform value types ────────────────────────────────────────────────────

/** AT uniform 值类型标签 */
export type UniformType =
  | 'number'        // "uCurlNoiseSpeed: 1"
  | 'color'         // "uColor: #ffffff"
  | 'array'         // "uBox: [1, 1, 1]"
  | 'texture'       // "tVelocity: T" — runtime texture ref
  | 'computed';     // "uProjMatrix: Cmat4" — computed/connected uniform

/**
 * Parsed uniform value — mirrors AT's UIL uniform declaration format.
 *
 * AT UIL 的 uniform 格式是 newline-separated "name: value"，
 * 其中 value 可以是:
 *   - 数字 → number
 *   - #hex → color (stored as hex string)
 *   - [x,y,z] → array of numbers
 *   - "T" → runtime texture reference
 *   - "Cmat4" | "Csampler2D" → computed uniform (camera matrices, fluid maps)
 */
export interface ParticleUniformValue {
  type: UniformType;
  /** Raw string from AT JSON (for round-tripping) */
  raw: string;
  /** Parsed value: number for 'number', string for 'color'/'texture'/'computed', number[] for 'array' */
  value: number | string | number[];
}

// ── Behavior code block ──────────────────────────────────────────────────────

/**
 * AT behavior preset identifiers — determines the code template category.
 *
 * From the 12 AT particle systems, these are all observed preset values:
 *   curl       — curl noise displacement (#require(curl.glsl))
 *   fluid      — mouse/screen-space fluid interaction (#require(glscreenprojection.glsl))
 *   pointcloud — sample from tPointCloud texture
 *   spline     — spline following (#require(splineparticles.fs))
 *   planeshape — texture-driven plane shape target
 */
export type BehaviorPreset = 'curl' | 'fluid' | 'pointcloud' | 'spline' | 'planeshape' | string;

/**
 * Single behavior code block — one of N GLSL snippets that compose the
 * particle update shader.
 *
 * AT 结构: INPUT_P_Element_{N}_{Sys}code_{M}_{field}
 *   field: code, name, preset, uniforms, file, btn
 */
export interface ParticleBehaviorCode {
  /** Code block id (e.g. "code_1", "code_2") */
  id: string;
  /** Display name (e.g. "Curl Noise", "Mouse Fluid", "Point Cloud") */
  name: string;
  /** Preset category — determines default code and required #require libs */
  preset: BehaviorPreset;
  /** GLSL code snippet — may contain #require(...) directives */
  code: string;
  /** Per-code-block uniform declarations (parsed from AT's "name: value\n..." format) */
  uniforms: Record<string, ParticleUniformValue>;
  /** Optional source file for point cloud data (e.g. "assets/geometry/particles/forest") */
  file?: string;
}

// ── Behavior ─────────────────────────────────────────────────────────────────

/**
 * Particle behavior — the composable GPGPU update logic layer.
 *
 * AT 结构:
 *   behavior_codeCount: number of code blocks
 *   behavior_data: JSON array of code refs in execution order
 *   behavior_uniforms: shared uniforms across all code blocks
 */
export interface ParticleBehavior {
  /** Number of code blocks */
  codeCount: number;
  /**
   * Execution order of code blocks — NOT necessarily sequential.
   * e.g. ["code_4", "code_1", "code_3", "code_2"] means
   * planeshape runs first, then curl, then fluid, then lerp.
   */
  executionOrder: string[];
  /** Shared behavior-level uniforms (applied to all code blocks) */
  uniforms: Record<string, ParticleUniformValue>;
  /** Code blocks indexed by id (e.g. "code_1" → ParticleBehaviorCode) */
  codes: Record<string, ParticleBehaviorCode>;
}

// ── Particle config ──────────────────────────────────────────────────────────

/**
 * AT particle config type — "lifecycle" particles have birth/death cycles.
 * Default (unset) = standard GPGPU particles.
 */
export type ParticleType = 'lifecycle' | 'standard';

/**
 * Complete particle system configuration — mirrors AT's INPUT_P structure.
 *
 * AT 结构: INPUT_P_Element_{N}_{SystemName}_config_{field}
 *
 * Observed across 12 AT particle systems:
 *   Element_0_BodyCores        (lifecycle, CoreParticlesShader)
 *   Element_0_LogoParticle     (LogoParticleShader)
 *   Element_0_ParticleTest     (ParticleTestShader, output=particles)
 *   Element_0_TubesInteraction (lifecycle, TubeShader, class=TubeController)
 *   Element_0_WorkDetailParticles (WorkDetailParticleShader, class=SplineParticles)
 *   Element_0_particleTest     (basic curl test)
 *   Element_19_CleanRoom       (WaterParticles, config only)
 *   Element_19_home_scene      (HomeParticleShader, class=HomeParticles, enablePhysics)
 *   Element_20_TreeScene       (FloatingParticles, config only)
 *   Element_21_TreeScene       (TreeParticleShader)
 *   Element_4_work_page        (WorkPageParticleShader)
 *   Element_6_Work             (FlowerParticleShader)
 */
export interface ParticleConfig {
  /** AT element index (e.g. 0, 4, 19, 21) */
  elementId: number;
  /** System name (e.g. "home_scene", "WorkDetailParticles", "BodyCores") */
  systemName: string;

  // ── Config layer ──────────────────────────────────────────────────────────

  /** Render shader class name (e.g. "HomeParticleShader", "CoreParticlesShader") */
  shader: string;
  /** Total number of particles — may be a static number or expression string */
  particleCount: string;
  /** Spawn volume bounds: [min, max] for each axis */
  width: [number, number];
  height: [number, number];
  depth: [number, number];
  /** Render shader uniforms (not behavior uniforms) */
  configUniforms: Record<string, ParticleUniformValue>;
  /** Particle type: "lifecycle" for birth/death cycle, "standard" otherwise */
  type: ParticleType;
  /** Optional custom controller class (e.g. "HomeParticles", "SplineParticles", "TubeController") */
  class?: string;
  /** Output variable name (e.g. "particles", "tubes") */
  output?: string;
  /** Whether GPGPU physics simulation is enabled */
  enablePhysics?: boolean;
  /** Initial position data source */
  initialPositions?: {
    filename: string;
    prefix: string;
    relative: string;
    src: string;
  };

  // ── Behavior layer ────────────────────────────────────────────────────────

  /** Composable GPGPU update behavior — code blocks + execution order */
  behavior: ParticleBehavior;
}

// ── #require library registry ────────────────────────────────────────────────

/**
 * GLSL library registry for #require() resolution.
 *
 * AT 的 code blocks 使用 #require(filename) 引入共享 GLSL 库。
 * 实际观察到的 5 个 require 目标:
 *   curl.glsl              — curl noise (curlNoise function)
 *   glscreenprojection.glsl — screen-space projection helpers
 *   luma.fs                — luminance extraction
 *   sdfs.glsl              — signed distance functions
 *   splineparticles.fs     — spline particle utilities
 *
 * 用户可通过 registerRequireLib() 注入自定义实现。
 */
const requireRegistry = new Map<string, string>();

// Register the built-in curl.glsl from CurlNoise.ts
requireRegistry.set('curl.glsl', CURL_NOISE_GLSL);

// Stub implementations for other AT require targets — callers should
// register real implementations via registerRequireLib()

requireRegistry.set('glscreenprojection.glsl', /* glsl */`
// glscreenprojection.glsl — AT screen projection helpers (stub)
// Register real implementation via registerRequireLib('glscreenprojection.glsl', src)

vec2 getProjection(vec3 worldPos, mat4 projMatrix) {
  vec4 clip = projMatrix * vec4(worldPos, 1.0);
  return clip.xy / clip.w * 0.5 + 0.5;
}

void applyNormal(inout vec3 v, mat4 normalMatrix) {
  v = (normalMatrix * vec4(v, 0.0)).xyz;
}
`);

requireRegistry.set('luma.fs', /* glsl */`
// luma.fs — AT luminance helper (stub)
// Register real implementation via registerRequireLib('luma.fs', src)

float luma(vec3 color) {
  return dot(color, vec3(0.2126, 0.7152, 0.0722));
}
`);

requireRegistry.set('sdfs.glsl', /* glsl */`
// sdfs.glsl — AT signed distance functions (stub)
// Register real implementation via registerRequireLib('sdfs.glsl', src)

float logo_sdf(vec3 p) {
  return length(p) - 1.0;
}

vec3 logo_norm(vec3 p) {
  return normalize(p);
}
`);

requireRegistry.set('splineparticles.fs', /* glsl */`
// splineparticles.fs — AT spline particle utilities (stub)
// Register real implementation via registerRequireLib('splineparticles.fs', src)

vec3 sRandom;
vec3 sOrigin;

vec3 getSplinePos(float t) {
  return mix(sOrigin, sOrigin + vec3(1.0, 0.0, 0.0), t);
}
`);

/**
 * Register or replace a GLSL library for #require() resolution.
 *
 * @param name - Library filename (e.g. "curl.glsl", "glscreenprojection.glsl")
 * @param source - Full GLSL source to inject at #require(name) sites
 *
 * @example
 * ```ts
 * import screenProjGlsl from './screen-projection.glsl?raw';
 * registerRequireLib('glscreenprojection.glsl', screenProjGlsl);
 * ```
 */
export function registerRequireLib(name: string, source: string): void {
  requireRegistry.set(name, source);
}

/**
 * Get all registered #require library names.
 */
export function getRegisteredLibs(): string[] {
  return Array.from(requireRegistry.keys());
}

// ── Uniform parsing ──────────────────────────────────────────────────────────

/**
 * Parse a single AT uniform value string into a typed ParticleUniformValue.
 *
 * AT uniform format examples:
 *   "1"           → { type: 'number', value: 1 }
 *   "0.1"         → { type: 'number', value: 0.1 }
 *   "#ffffff"     → { type: 'color',  value: '#ffffff' }
 *   "#000"        → { type: 'color',  value: '#000' }
 *   "[1, 1, 1]"   → { type: 'array',  value: [1, 1, 1] }
 *   "[0,0,0]"     → { type: 'array',  value: [0, 0, 0] }
 *   "T"           → { type: 'texture', value: 'T' }
 *   "Cmat4"       → { type: 'computed', value: 'Cmat4' }
 *   "Csampler2D"  → { type: 'computed', value: 'Csampler2D' }
 */
function parseUniformValue(raw: string): ParticleUniformValue {
  const trimmed = raw.trim();

  // Texture reference: exactly "T"
  if (trimmed === 'T') {
    return { type: 'texture', raw: trimmed, value: 'T' };
  }

  // Computed uniform: starts with "C" (Cmat4, Csampler2D)
  if (trimmed.startsWith('C') && trimmed.length > 1 && /^C[a-zA-Z]/.test(trimmed)) {
    return { type: 'computed', raw: trimmed, value: trimmed };
  }

  // Color: starts with "#"
  if (trimmed.startsWith('#')) {
    return { type: 'color', raw: trimmed, value: trimmed };
  }

  // Array: starts with "["
  if (trimmed.startsWith('[')) {
    try {
      const parsed = JSON.parse(trimmed) as number[];
      return { type: 'array', raw: trimmed, value: parsed };
    } catch {
      return { type: 'array', raw: trimmed, value: [] };
    }
  }

  // Number
  const num = parseFloat(trimmed);
  if (!isNaN(num)) {
    return { type: 'number', raw: trimmed, value: num };
  }

  // Fallback: treat as string (expression like "Tests.particleCount()")
  return { type: 'number', raw: trimmed, value: 0 };
}

/**
 * Parse AT's newline-separated uniform block.
 *
 * Format: "name: value\nname: value\n..."
 *
 * Examples from AT:
 *   "uSize: 0.2\nuColor: #ffffff\ntColors: T"
 *   "uCurlNoiseScale: 1\nuCurlTimeScale: 0\nuCurlNoiseSpeed: 0"
 *   "uProjMatrix: Cmat4\ntFluid: Csampler2D\nuMouseStrength: 1"
 */
function parseUniformBlock(block: string): Record<string, ParticleUniformValue> {
  const result: Record<string, ParticleUniformValue> = {};
  if (!block || !block.trim()) return result;

  const lines = block.split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const colonIdx = trimmed.indexOf(':');
    if (colonIdx < 0) continue;

    const name = trimmed.substring(0, colonIdx).trim();
    const value = trimmed.substring(colonIdx + 1).trim();
    if (name) {
      result[name] = parseUniformValue(value);
    }
  }

  return result;
}

/**
 * Parse AT's range string "[min, max]" into a numeric tuple.
 *
 * @param rangeStr - AT format string like "[-1,1]" or "[-0.3,0.3]"
 * @returns [min, max] tuple, defaults to [-1, 1]
 */
function parseRange(rangeStr: string | undefined): [number, number] {
  if (!rangeStr) return [-1, 1];
  try {
    const parsed = JSON.parse(rangeStr) as [number, number];
    if (Array.isArray(parsed) && parsed.length === 2) {
      return [parsed[0], parsed[1]];
    }
  } catch { /* fall through */ }
  return [-1, 1];
}

// ── Parse particle config from AT flat-key JSON ─────────────────────────────

/**
 * Parse a complete ParticleConfig from AT's flat INPUT_P key-value map.
 *
 * AT stores particle system configuration as flat keys in a JSON object:
 *
 *   INPUT_P_Element_0_ParticleTest_config_shader:        "ParticleTestShader"
 *   INPUT_P_Element_0_ParticleTest_config_particleCount: "Tests.particleCount()"
 *   INPUT_P_Element_0_ParticleTest_config_width:         "[-1,1]"
 *   INPUT_P_Element_0_ParticleTest_behavior_data:        '["code_1","code_2"]'
 *   INPUT_P_Element_0_ParticleTestcode_1_code:           "vec3 pointShape = ..."
 *   INPUT_P_Element_0_ParticleTestcode_1_preset:         "pointcloud"
 *   ...
 *
 * This function extracts all keys matching a given system and reconstructs
 * the full typed configuration.
 *
 * @param data      - The full AT UIL JSON object (or a subset with INPUT_P_ keys)
 * @param elementId - Element index (e.g. 0, 4, 19)
 * @param systemName - System name (e.g. "home_scene", "ParticleTest")
 * @returns Fully parsed ParticleConfig
 *
 * @example
 * ```ts
 * const resp = await fetch('https://activetheory.net/assets/data/uil.1780406240914.json');
 * const data = await resp.json();
 * const config = parseParticleConfig(data, 19, 'home_scene');
 * // config.behavior.executionOrder → ["code_4", "code_1", "code_3", "code_2"]
 * // config.shader → "HomeParticleShader"
 * // config.behavior.codes.code_1.preset → "curl"
 * ```
 */
export function parseParticleConfig(
  data: Record<string, unknown>,
  elementId: number,
  systemName: string,
): ParticleConfig {
  const configPrefix = `INPUT_P_Element_${elementId}_${systemName}_config_`;
  const behaviorPrefix = `INPUT_P_Element_${elementId}_${systemName}_behavior_`;
  const codePrefix = `INPUT_P_Element_${elementId}_${systemName}code_`;

  // Helper to get a typed value from the flat map
  const get = (key: string): unknown => data[key];
  const getStr = (key: string, fallback = ''): string => {
    const v = get(key);
    return typeof v === 'string' ? v : fallback;
  };

  // ── Parse config fields ─────────────────────────────────────────────────
  const shader = getStr(configPrefix + 'shader', 'DefaultParticleShader');
  const particleCount = getStr(configPrefix + 'particleCount', '1000');
  const width = parseRange(getStr(configPrefix + 'width'));
  const height = parseRange(getStr(configPrefix + 'height'));
  const depth = parseRange(getStr(configPrefix + 'depth'));
  const configUniforms = parseUniformBlock(getStr(configPrefix + 'uniforms'));
  const typeStr = getStr(configPrefix + 'type');
  const type: ParticleType = typeStr === 'lifecycle' ? 'lifecycle' : 'standard';
  const cls = getStr(configPrefix + 'class') || undefined;
  const output = getStr(configPrefix + 'output') || undefined;
  const enablePhysicsStr = getStr(configPrefix + 'enablePhysics');
  const enablePhysics = enablePhysicsStr === 'true' ? true : undefined;

  // Initial positions (object with filename/prefix/relative/src)
  let initialPositions: ParticleConfig['initialPositions'] | undefined;
  const initPos = get(configPrefix + 'initialPositions');
  if (initPos && typeof initPos === 'object' && !Array.isArray(initPos)) {
    const ip = initPos as Record<string, string>;
    initialPositions = {
      filename: ip.filename ?? '',
      prefix: ip.prefix ?? '',
      relative: ip.relative ?? 'null',
      src: ip.src ?? '',
    };
  }

  // ── Parse behavior fields ──────────────────────────────────────────────
  const codeCountRaw = get(behaviorPrefix + 'codeCount');
  const codeCount = typeof codeCountRaw === 'number' ? codeCountRaw : 0;

  const behaviorDataStr = getStr(behaviorPrefix + 'data', '[]');
  let executionOrder: string[];
  try {
    executionOrder = JSON.parse(behaviorDataStr) as string[];
  } catch {
    executionOrder = [];
  }

  const behaviorUniforms = parseUniformBlock(getStr(behaviorPrefix + 'uniforms'));

  // ── Parse code blocks ──────────────────────────────────────────────────
  const codes: Record<string, ParticleBehaviorCode> = {};

  // Discover code blocks by scanning keys
  const codeIds = new Set<string>();
  for (const key of Object.keys(data)) {
    if (key.startsWith(codePrefix)) {
      const rest = key.substring(codePrefix.length);
      // rest looks like "1_code", "1_name", "2_uniforms", etc.
      const underscoreIdx = rest.indexOf('_');
      if (underscoreIdx > 0) {
        codeIds.add(`code_${rest.substring(0, underscoreIdx)}`);
      }
    }
  }

  for (const codeId of codeIds) {
    // codeId is e.g. "code_1" → numeric part is "1"
    const numPart = codeId.replace('code_', '');
    const codeKeyPrefix = `${codePrefix}${numPart}_`;

    const code = getStr(codeKeyPrefix + 'code');
    const name = getStr(codeKeyPrefix + 'name', codeId);
    const preset = getStr(codeKeyPrefix + 'preset', '') as BehaviorPreset;
    const codeUniforms = parseUniformBlock(getStr(codeKeyPrefix + 'uniforms'));
    const file = getStr(codeKeyPrefix + 'file') || undefined;

    codes[codeId] = { id: codeId, name, preset, code, uniforms: codeUniforms, file };
  }

  // ── Assemble ───────────────────────────────────────────────────────────
  return {
    elementId,
    systemName,
    shader,
    particleCount,
    width,
    height,
    depth,
    configUniforms,
    type,
    class: cls,
    output,
    enablePhysics,
    initialPositions,
    behavior: {
      codeCount,
      executionOrder,
      uniforms: behaviorUniforms,
      codes,
    },
  };
}

// ── Discover all particle systems in AT data ─────────────────────────────────

/**
 * Scan an AT UIL JSON object and return identifiers for all INPUT_P particle systems.
 *
 * @param data - The full AT UIL JSON
 * @returns Array of { elementId, systemName } pairs
 *
 * @example
 * ```ts
 * const systems = discoverParticleSystems(data);
 * // → [{ elementId: 0, systemName: 'BodyCores' },
 * //    { elementId: 0, systemName: 'LogoParticle' },
 * //    { elementId: 19, systemName: 'home_scene' }, ...]
 * ```
 */
export function discoverParticleSystems(
  data: Record<string, unknown>,
): Array<{ elementId: number; systemName: string }> {
  const seen = new Set<string>();
  const results: Array<{ elementId: number; systemName: string }> = [];

  for (const key of Object.keys(data)) {
    if (!key.startsWith('INPUT_P_Element_')) continue;

    const rest = key.substring('INPUT_P_Element_'.length);
    // rest: "0_ParticleTest_config_shader" or "0_ParticleTestcode_1_code"
    const firstUnderscore = rest.indexOf('_');
    if (firstUnderscore < 0) continue;

    const elementIdStr = rest.substring(0, firstUnderscore);
    const elementId = parseInt(elementIdStr, 10);
    if (isNaN(elementId)) continue;

    const afterId = rest.substring(firstUnderscore + 1);

    // Find system name boundary: _config_, _behavior_, or code_ (concatenated)
    let systemName: string;
    const configIdx = afterId.indexOf('_config_');
    const behaviorIdx = afterId.indexOf('_behavior_');
    const codeIdx = afterId.indexOf('code_');

    if (configIdx > 0) {
      systemName = afterId.substring(0, configIdx);
    } else if (behaviorIdx > 0) {
      systemName = afterId.substring(0, behaviorIdx);
    } else if (codeIdx > 0) {
      systemName = afterId.substring(0, codeIdx);
    } else {
      continue;
    }

    const unique = `${elementId}_${systemName}`;
    if (!seen.has(unique)) {
      seen.add(unique);
      results.push({ elementId, systemName });
    }
  }

  return results;
}

// ── #require replacement ─────────────────────────────────────────────────────

/**
 * Pattern matching AT's #require(filename) directive.
 *
 * Examples:
 *   #require(curl.glsl)
 *   #require(glscreenprojection.glsl)
 *   #require(splineparticles.fs)
 */
const REQUIRE_PATTERN = /^\s*#require\(([^)]+)\)\s*$/gm;

/**
 * Replace all #require(filename) directives in GLSL source with the
 * registered library source code.
 *
 * AT's shader compiler resolves #require at build time. This function
 * replicates that behavior using the requireRegistry. Each unique
 * require is only injected once (deduplication across multiple code blocks).
 *
 * @param glsl - GLSL source with #require directives
 * @param alreadyIncluded - Set of already-included library names (for dedup across blocks)
 * @returns GLSL source with #require lines replaced by library code
 * @throws if a required library is not registered
 */
function resolveRequires(
  glsl: string,
  alreadyIncluded: Set<string>,
): string {
  return glsl.replace(REQUIRE_PATTERN, (_match: string, libName: string) => {
    const name = libName.trim();

    // Deduplicate: only inject each library once
    if (alreadyIncluded.has(name)) {
      return `// #require(${name}) — already included above`;
    }

    const source = requireRegistry.get(name);
    if (source === undefined) {
      console.warn(
        `[particle-behavior] Unknown #require("${name}"). ` +
        `Register it via registerRequireLib('${name}', glslSource). ` +
        `Available: ${Array.from(requireRegistry.keys()).join(', ')}`,
      );
      return `// #require(${name}) — NOT FOUND (register via registerRequireLib)`;
    }

    alreadyIncluded.add(name);
    return `// ── #require(${name}) ──\n${source}\n// ── end ${name} ──`;
  });
}

// ── Compile particle shader from config ──────────────────────────────────────

/**
 * GLSL uniform type string for AT uniform types.
 */
function uniformTypeToGlsl(uv: ParticleUniformValue, name: string): string {
  switch (uv.type) {
    case 'number':
      return `uniform float ${name};`;
    case 'color':
      return `uniform vec3 ${name};`;
    case 'array': {
      const arr = uv.value as number[];
      if (arr.length === 2) return `uniform vec2 ${name};`;
      if (arr.length === 3) return `uniform vec3 ${name};`;
      if (arr.length === 4) return `uniform vec4 ${name};`;
      return `uniform float ${name}; // array[${arr.length}]`;
    }
    case 'texture':
      return `uniform sampler2D ${name};`;
    case 'computed': {
      const cv = uv.value as string;
      if (cv === 'Csampler2D') return `uniform sampler2D ${name};`;
      if (cv === 'Cmat4') return `uniform mat4 ${name};`;
      return `uniform float ${name}; // computed: ${cv}`;
    }
    default:
      return `uniform float ${name};`;
  }
}

/**
 * Compile a ParticleConfig's behavior into a complete GLSL fragment shader.
 *
 * The shader follows AT's GPGPU particle update pattern:
 *   1. Read current pos/life from tPosition, velocity from tVelocity
 *   2. Declare AT builtins: pos, origin, target, life, uv, vUv, time, HZ, random
 *   3. Inject #require libraries (deduplicated)
 *   4. Concatenate code blocks in behavior.executionOrder
 *   5. Write updated pos/life to fragColor
 *
 * AT built-in variables available in code blocks:
 *   vec3  pos    — current particle position (read/write)
 *   vec3  origin — initial/spawn position
 *   vec3  target — target position (code blocks accumulate displacement)
 *   float life   — particle life [0,1] (decays, respawn when ≤0)
 *   vec2  uv     — particle UV in state texture
 *   vec2  vUv    — alias for uv (varying)
 *   float time   — elapsed time
 *   float HZ     — frame delta normalized to 60fps (AT convention)
 *   vec4  random — per-particle random values (hash from UV + time)
 *
 * @param config - Parsed ParticleConfig
 * @param options - Optional overrides
 * @returns Complete GLSL fragment shader source
 *
 * @example
 * ```ts
 * const config = parseParticleConfig(data, 19, 'home_scene');
 * const shaderSrc = compileParticleShader(config);
 * // → full #version 300 es fragment shader with curl + fluid + planeshape + lerp
 * ```
 */
export function compileParticleShader(
  config: ParticleConfig,
  options: {
    /** GLSL version header (default: "#version 300 es") */
    version?: string;
    /** Extra uniforms to declare */
    extraUniforms?: Record<string, ParticleUniformValue>;
    /** Extra GLSL code to prepend before main() */
    preamble?: string;
  } = {},
): string {
  const version = options.version ?? '#version 300 es';
  const behavior = config.behavior;

  // Collect all uniforms: config-level + behavior-level + per-code-block
  const allUniforms: Record<string, ParticleUniformValue> = {};

  for (const [name, uv] of Object.entries(config.configUniforms)) {
    allUniforms[name] = uv;
  }
  for (const [name, uv] of Object.entries(behavior.uniforms)) {
    allUniforms[name] = uv;
  }
  for (const codeId of behavior.executionOrder) {
    const code = behavior.codes[codeId];
    if (code) {
      for (const [name, uv] of Object.entries(code.uniforms)) {
        allUniforms[name] = uv;
      }
    }
  }
  if (options.extraUniforms) {
    for (const [name, uv] of Object.entries(options.extraUniforms)) {
      allUniforms[name] = uv;
    }
  }

  // Remove builtins that are already declared in the shader template
  const builtinNames = new Set([
    'tPosition', 'tVelocity', 'tLife',
    'uTime', 'uDelta', 'uSetup',
  ]);

  // Generate uniform declarations
  const uniformDecls: string[] = [];
  for (const [name, uv] of Object.entries(allUniforms)) {
    if (!builtinNames.has(name)) {
      uniformDecls.push(uniformTypeToGlsl(uv, name));
    }
  }

  // Process code blocks in execution order, resolving #require
  const alreadyIncluded = new Set<string>();
  const codeBlocks: string[] = [];

  for (const codeId of behavior.executionOrder) {
    const code = behavior.codes[codeId];
    if (!code) {
      codeBlocks.push(`  // WARNING: ${codeId} not found in behavior.codes`);
      continue;
    }

    const resolvedCode = resolveRequires(code.code, alreadyIncluded);

    // Remove #test / #endtest preprocessor blocks (AT device testing, not needed)
    const cleanedCode = resolvedCode
      .replace(/^\s*#test\b.*$/gm, '// #test (removed)')
      .replace(/^\s*#endtest\b.*$/gm, '// #endtest');

    codeBlocks.push(
      `  // ── ${code.name || code.id} (preset: ${code.preset || 'custom'}) ──\n` +
      cleanedCode
        .split('\n')
        .map((line: string) => '  ' + line)
        .join('\n'),
    );
  }

  // Assemble the full shader
  const shader = `${version}
/**
 * Auto-compiled particle behavior shader
 * System: Element_${config.elementId}_${config.systemName}
 * Shader: ${config.shader}
 * Codes: ${behavior.executionOrder.join(' → ')}
 * Generated by particle-behavior.ts (AT INPUT_P port)
 */
precision highp float;
precision highp sampler2D;

// ── Built-in uniforms ──
uniform sampler2D tPosition;
uniform sampler2D tVelocity;
uniform sampler2D tLife;
uniform float uTime;
uniform float uDelta;
uniform float uSetup;

// ── Config + behavior uniforms ──
${uniformDecls.join('\n')}

in vec2 vUv;
out vec4 fragColor;

// ── AT hash function (per-particle random) ──
vec4 hash44(vec4 p) {
  p = fract(p * vec4(443.897, 441.423, 437.195, 433.813));
  p += dot(p, p.wzxy + 19.19);
  return fract((p.xxyz + p.yzzw) * p.zywx);
}

${options.preamble ? '// ── Custom preamble ──\n' + options.preamble + '\n' : ''}
void main() {
  vec2 uv = vUv;
  vec4 posLife = texture(tPosition, uv);
  vec4 velData = texture(tVelocity, uv);

  // AT built-in variables
  vec3  pos    = posLife.xyz;
  vec3  origin = posLife.xyz;    // snapshot for origin reference
  vec3  target = pos;            // accumulator for target position
  float life   = posLife.w;
  float time   = uTime;
  float HZ     = uDelta;        // AT: delta * 60 (normalised to 60fps)
  vec4  random = hash44(vec4(uv, uTime * 0.01, uTime * 0.017));

  // ── Behavior code blocks (execution order: ${behavior.executionOrder.join(' → ')}) ──

${codeBlocks.join('\n\n')}

  // ── Write output ──
  fragColor = vec4(pos, life);
}
`;

  return shader;
}

// ── Utility: extract all presets used by a config ────────────────────────────

/**
 * Get all behavior preset types used by a particle config.
 *
 * @example
 * ```ts
 * getPresets(homeSceneConfig)
 * // → ['planeshape', 'curl', 'fluid']
 * ```
 */
export function getPresets(config: ParticleConfig): BehaviorPreset[] {
  const presets: BehaviorPreset[] = [];
  for (const codeId of config.behavior.executionOrder) {
    const code = config.behavior.codes[codeId];
    if (code?.preset) {
      presets.push(code.preset);
    }
  }
  return presets;
}

/**
 * Get all #require dependencies for a particle config.
 *
 * Scans all code blocks (in execution order) and extracts #require(name) references.
 *
 * @example
 * ```ts
 * getRequireDependencies(homeSceneConfig)
 * // → ['curl.glsl', 'glscreenprojection.glsl']
 * ```
 */
export function getRequireDependencies(config: ParticleConfig): string[] {
  const deps: string[] = [];
  const seen = new Set<string>();

  for (const codeId of config.behavior.executionOrder) {
    const code = config.behavior.codes[codeId];
    if (!code) continue;

    const matches = code.code.matchAll(/#require\(([^)]+)\)/g);
    for (const match of matches) {
      const name = match[1].trim();
      if (!seen.has(name)) {
        seen.add(name);
        deps.push(name);
      }
    }
  }

  return deps;
}

/**
 * Collect all uniform names and their default values from a particle config.
 *
 * Merges config uniforms, behavior uniforms, and all code block uniforms.
 * Later entries override earlier ones (code > behavior > config).
 *
 * @example
 * ```ts
 * const uniforms = collectUniforms(config);
 * // → { uSize: { type: 'number', value: 1 }, tNormal: { type: 'texture', value: 'T' }, ... }
 * ```
 */
export function collectUniforms(
  config: ParticleConfig,
): Record<string, ParticleUniformValue> {
  const result: Record<string, ParticleUniformValue> = {};

  // Config uniforms (lowest priority)
  for (const [name, uv] of Object.entries(config.configUniforms)) {
    result[name] = uv;
  }

  // Behavior-level uniforms
  for (const [name, uv] of Object.entries(config.behavior.uniforms)) {
    result[name] = uv;
  }

  // Per-code-block uniforms (highest priority, in execution order)
  for (const codeId of config.behavior.executionOrder) {
    const code = config.behavior.codes[codeId];
    if (code) {
      for (const [name, uv] of Object.entries(code.uniforms)) {
        result[name] = uv;
      }
    }
  }

  return result;
}
