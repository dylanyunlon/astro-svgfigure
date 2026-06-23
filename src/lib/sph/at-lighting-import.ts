/**
 * at-lighting-import.ts — M804: AT Lighting Direct Import
 *
 * 直接导入 ActiveTheory 的 lighting.fs (compiled.vs 中 ~9K行 bundle,
 * 经 #require() 解析后含完整 PBR + IBL + Shadow 管线):
 *
 *   lighting.fs
 *     └─ #require(LightingCommon.glsl)
 *          ├─ #require(AreaLights.glsl)    — LTC 面光源 (Heitz+Dupuy)
 *          └─ #require(Phong.glsl)         — Blinn-Phong 镜面 + Schlick
 *     └─ lightDirectional / lightPoint / lightCone / lightArea
 *     └─ getCombinedColor / getStandardColor / getPointLightColor / ...
 *
 *   pbr.fs
 *     └─ Cook-Torrance IBL (env diffuse/specular + BRDF LUT)
 *     └─ RGBM HDR decode + Uncharted2 tone-mapping
 *     └─ getPBR() → full PBR pipeline entry
 *
 *   shadows.fs
 *     └─ PCF / PCSS soft shadows (Poisson disk sampling)
 *     └─ getShadow() / getShadowPCSS() → shadow pipeline entry
 *
 * 本模块功能:
 *   1. 通过 ATShaderLoader 加载 compiled.vs, 递归解析 #require() 依赖
 *   2. 清理 AT 预处理器指令 (#test / #endtest / #pragma unroll_loop)
 *   3. 解析已解析 GLSL 中的 uniform 列表 (sampler2D / float / vec / mat / array)
 *   4. 注入 WebGL2 兼容的 preamble (precision, #define NUM_LIGHTS 等)
 *   5. 通过 AstroProgram 创建 WebGL program, 缓存 uniform location
 *
 * 用法:
 *   const lighting = await ATLightingImport.create(renderer, '/shaders/compiled.vs');
 *   lighting.program.use();
 *   lighting.bindLights(lights);
 *   lighting.bindShadow(shadowMap, shadowMatrix);
 *   lighting.bindPBR(baseColorTex, mroTex, normalTex, envTex);
 *
 * Research: xiaodi #M804 — cell-pubsub-loop
 */

import { ATShaderLoader } from './at-shader-loader.ts';
import { AstroProgram } from '../renderer/AstroProgram.ts';
import type { AstroRenderer } from '../renderer/AstroRenderer.ts';

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Types
// ─────────────────────────────────────────────────────────────────────────────

/** Parsed uniform metadata extracted from resolved GLSL source. */
export interface ATUniformInfo {
  /** GLSL name (e.g. "lightColor", "shadowMatrix") */
  name: string;
  /** GLSL type (e.g. "vec3", "sampler2D", "mat4", "float") */
  type: string;
  /** Array size (0 = scalar, >0 = uniform T name[N]) */
  arraySize: number;
  /** Which shader module declared this (lighting / pbr / shadow / vertex) */
  source: 'lighting' | 'pbr' | 'shadow' | 'vertex' | 'common';
}

/** Light data for a single AT light (matches compiled.vs Lighting.glsl arrays). */
export interface ATLight {
  /** Light type: 1 = directional, 2 = point, 3 = cone/spot, 4 = area */
  type: 1 | 2 | 3 | 4;
  /** Position (world space) or direction for directional */
  position: [number, number, number];
  /** Color (RGB, linear space) */
  color: [number, number, number];
  /**
   * lightData — per-type payload:
   *   directional: unused
   *   point: [0,0,0, unused]
   *   cone: [dirX, dirY, dirZ, coneAngle]
   *   area: [posX, posY, posZ, roughness]
   */
  data: [number, number, number, number];
  /** lightData2 — cone feather / area halfWidth */
  data2: [number, number, number, number];
  /** lightData3 — area halfHeight */
  data3: [number, number, number, number];
  /**
   * lightProperties:
   *   .x = intensity
   *   .y = range (distance cutoff)
   *   .z = min threshold (ambient floor)
   *   .w = light type ID (1.0/2.0/3.0/4.0 — must match .type)
   */
  properties: [number, number, number, number];
}

/** Shadow configuration matching AT's shadow system. */
export interface ATShadowConfig {
  /** Shadow map WebGL texture */
  map: WebGLTexture;
  /** Shadow projection × view matrix (Float32Array[16]) */
  matrix: Float32Array;
  /** Shadow map resolution (width = height) */
  size: number;
  /** Light position for bias calculation */
  lightPos: [number, number, number];
}

/** PBR texture set for AT's IBL pipeline (pbr.fs). */
export interface ATPBRTextures {
  /** Base color / albedo map */
  baseColor: WebGLTexture;
  /** Metallic-Roughness-Occlusion packed texture */
  mro: WebGLTexture;
  /** Tangent-space normal map */
  normal: WebGLTexture;
  /** BRDF integration LUT (2D) */
  lut: WebGLTexture;
  /** IBL diffuse environment (equirectangular) */
  envDiffuse: WebGLTexture;
  /** IBL specular environment (mip-chain equirectangular) */
  envSpecular: WebGLTexture;
  /** Optional: baked lightmap */
  lightmap?: WebGLTexture;
}

/** Configuration for ATLightingImport.create(). */
export interface ATLightingConfig {
  /** Max number of dynamic lights (injected as #define NUM_LIGHTS N). Default: 4 */
  maxLights?: number;
  /** Max shadow casters (injected as #define SHADOW_COUNT N). Default: 1 */
  maxShadows?: number;
  /** Enable PCSS soft shadows. Default: false (standard PCF) */
  enablePCSS?: boolean;
  /** Shadow quality: 'low' | 'med' | 'high'. Default: 'med' */
  shadowQuality?: 'low' | 'med' | 'high';
  /** Enable area lights (requires LTC textures). Default: false */
  enableAreaLights?: boolean;
  /** Custom vertex shader override (GLSL). If null, uses AT lighting.vs */
  customVertex?: string | null;
  /** Custom fragment shader override (GLSL). If null, uses full AT lighting.fs + pbr.fs + shadows.fs */
  customFragment?: string | null;
}

const DEFAULT_CONFIG: Required<ATLightingConfig> = {
  maxLights: 4,
  maxShadows: 1,
  enablePCSS: false,
  shadowQuality: 'med',
  enableAreaLights: false,
  customVertex: null,
  customFragment: null,
};

// ─────────────────────────────────────────────────────────────────────────────
// § 2  AT Preprocessor Cleanup
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Strip AT-specific preprocessor directives that are not valid GLSL:
 *   - #test <condition> ... #endtest  → keep content (non-Metal path)
 *   - #test !!window.Metal ... #endtest → remove content (Metal-only)
 *   - #pragma unroll_loop → remove (WebGL2 doesn't need it)
 *   - #require() → should already be resolved; strip any stale references
 */
function stripATPreprocessor(glsl: string): string {
  let result = glsl;

  // Remove Metal-only blocks: #test !!window.Metal ... #endtest
  result = result.replace(
    /#test\s+!!window\.Metal[\s\S]*?#endtest/g,
    '/* [AT] Metal-only block removed */',
  );

  // Remove Lighting.fallbackAreaToPointTest() conditional
  result = result.replace(
    /#test\s+Lighting\.fallbackAreaToPointTest\(\)[\s\S]*?#endtest/g,
    '/* [AT] fallbackAreaToPoint test removed */',
  );

  // Keep non-Metal blocks: #test !window.Metal ... #endtest → strip markers only
  result = result.replace(/#test\s+!window\.Metal\s*/g, '/* [AT] non-Metal path */\n');
  result = result.replace(/#endtest/g, '/* [AT] /endtest */');

  // Remove any remaining #test ... #endtest blocks (unknown conditions → keep content)
  result = result.replace(/#test\s+[^\n]*/g, '/* [AT] test removed */');

  // Remove #pragma unroll_loop (WebGL2 handles loop unrolling itself)
  result = result.replace(/#pragma\s+unroll_loop\s*/g, '');

  // Remove any stale #require() that wasn't resolved
  result = result.replace(/#require\([^)]*\)/g, '/* [AT] unresolved require */');

  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  Uniform Parser
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Extract all `uniform <type> <name>[<array>];` declarations from resolved GLSL.
 *
 * Handles:
 *   uniform float uHDR;
 *   uniform sampler2D tBaseColor;
 *   uniform vec4 lightColor[NUM_LIGHTS];   → arraySize = NUM_LIGHTS (symbolic)
 *   uniform mat4 shadowMatrix[SHADOW_COUNT];
 *
 * The GLSL struct definitions (like LightConfig) are NOT uniforms — they're
 * stack-local in the fragment shader. We skip struct declarations.
 */
function parseUniforms(glsl: string, source: ATUniformInfo['source']): ATUniformInfo[] {
  const uniforms: ATUniformInfo[] = [];
  // Regex: uniform <type> <name> [ [<size>] ] ;
  // Allows optional whitespace, handles multi-word types (e.g. "sampler2D", "samplerCube")
  const re = /uniform\s+((?:(?:low|medium|high)p\s+)?[\w]+)\s+(\w+)(?:\s*\[\s*(\w+)\s*\])?\s*;/g;

  let m: RegExpExecArray | null;
  const seen = new Set<string>();

  while ((m = re.exec(glsl)) !== null) {
    const type = m[1].replace(/(?:low|medium|high)p\s+/, ''); // strip precision qualifier
    const name = m[2];
    const arraySizeRaw = m[3] ?? '';

    if (seen.has(name)) continue; // dedup (same uniform may appear in #ifdef branches)
    seen.add(name);

    let arraySize = 0;
    if (arraySizeRaw) {
      const parsed = parseInt(arraySizeRaw, 10);
      arraySize = isNaN(parsed) ? -1 : parsed; // -1 = symbolic (NUM_LIGHTS etc.)
    }

    uniforms.push({ name, type, arraySize, source });
  }

  return uniforms;
}

/**
 * Resolve symbolic array sizes (NUM_LIGHTS, SHADOW_COUNT) to concrete numbers.
 */
function resolveArraySizes(
  uniforms: ATUniformInfo[],
  defines: Map<string, number>,
): ATUniformInfo[] {
  return uniforms.map((u) => {
    if (u.arraySize !== -1) return u;
    // Try to match the array size name from the GLSL source
    // This is a heuristic: look for common AT constants
    for (const [key, value] of defines) {
      // The uniform was parsed with arraySize = -1 (symbolic)
      // We need to find which define it maps to — use naming convention
      if (u.name.includes('light') || u.name.includes('Light')) {
        if (key === 'NUM_LIGHTS') return { ...u, arraySize: value };
      }
      if (u.name.includes('shadow') || u.name.includes('Shadow')) {
        if (key === 'SHADOW_COUNT') return { ...u, arraySize: value };
      }
    }
    // Fallback: use MAX define
    return { ...u, arraySize: defines.get('NUM_LIGHTS') ?? 4 };
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  WebGL2 Preamble Builder
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build the GLSL preamble that injects #version, precision, and #defines
 * to make AT's resolved shader WebGL2-compatible.
 */
function buildPreamble(config: Required<ATLightingConfig>, isVertex: boolean): string {
  const lines: string[] = [
    '#version 300 es',
    `precision highp float;`,
    `precision highp int;`,
    `precision highp sampler2D;`,
    '',
    `// ── AT Lighting defines (M804) ──`,
    `#define NUM_LIGHTS ${config.maxLights}`,
    `#define SHADOW_COUNT ${config.maxShadows}`,
  ];

  if (config.maxShadows > 0) {
    lines.push('#define SHADOW_MAPS');
  }

  if (config.shadowQuality === 'med') {
    lines.push('#define SHADOWS_MED');
  } else if (config.shadowQuality === 'high') {
    lines.push('#define SHADOWS_HIGH');
  }

  // WebGL2: varying → in/out
  if (isVertex) {
    lines.push('', '// ── WebGL2 compat ──');
    lines.push('#define varying out');
    lines.push('#define attribute in');
  } else {
    lines.push('', '// ── WebGL2 compat ──');
    lines.push('#define varying in');
    lines.push('#define texture2D texture');
    lines.push('#define gl_FragColor _fragColor');
    lines.push('out vec4 _fragColor;');
  }

  lines.push('');
  return lines.join('\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  Light Uniform Arrays Declaration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * AT's lighting system expects these uniform arrays to be declared globally
 * (they're injected by the AT runtime, not declared in compiled.vs):
 *
 *   uniform vec4 lightColor[NUM_LIGHTS];
 *   uniform vec4 lightPos[NUM_LIGHTS];
 *   uniform vec4 lightData[NUM_LIGHTS];
 *   uniform vec4 lightData2[NUM_LIGHTS];
 *   uniform vec4 lightData3[NUM_LIGHTS];
 *   uniform vec4 lightProperties[NUM_LIGHTS];
 *
 *   uniform sampler2D shadowMap[SHADOW_COUNT];
 *   uniform mat4 shadowMatrix[SHADOW_COUNT];
 *   uniform float shadowSize[SHADOW_COUNT];
 *   uniform vec3 shadowLightPos[SHADOW_COUNT];
 */
function buildLightUniformDecls(config: Required<ATLightingConfig>): string {
  const lines: string[] = [
    '// ── AT light uniform arrays (injected by at-lighting-import) ──',
    `uniform vec4 lightColor[${config.maxLights}];`,
    `uniform vec4 lightPos[${config.maxLights}];`,
    `uniform vec4 lightData[${config.maxLights}];`,
    `uniform vec4 lightData2[${config.maxLights}];`,
    `uniform vec4 lightData3[${config.maxLights}];`,
    `uniform vec4 lightProperties[${config.maxLights}];`,
    '',
  ];

  if (config.maxShadows > 0) {
    lines.push('// ── AT shadow uniforms ──');
    for (let i = 0; i < config.maxShadows; i++) {
      lines.push(`uniform sampler2D shadowMap_${i};`);
    }
    // WebGL2 doesn't support sampler2D arrays indexed dynamically;
    // we declare individual uniforms and a macro wrapper
    lines.push(`uniform mat4 shadowMatrix[${config.maxShadows}];`);
    lines.push(`uniform float shadowSize[${config.maxShadows}];`);
    lines.push(`uniform vec3 shadowLightPos[${config.maxShadows}];`);
    lines.push('');
    // For getShadow() which uses shadowMap[i], provide a getter macro
    // (AT's code uses: texture2D(shadowMap[i], ...))
    // WebGL2 workaround: flatten into individual sampler lookups
    if (config.maxShadows === 1) {
      lines.push('#define shadowMap_FETCH(i, uv) texture(shadowMap_0, uv)');
    } else {
      lines.push('vec4 shadowMap_FETCH(int i, vec2 uv) {');
      for (let i = 0; i < config.maxShadows; i++) {
        lines.push(`  if (i == ${i}) return texture(shadowMap_${i}, uv);`);
      }
      lines.push('  return vec4(1.0);');
      lines.push('}');
    }
    lines.push('');
  }

  return lines.join('\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  Shadow Shader Patching
// ─────────────────────────────────────────────────────────────────────────────

/**
 * AT's shadows.fs uses `texture2D(shadowMap, uv + ...)` where shadowMap is
 * a sampler2D parameter. In WebGL2, we need to replace array-indexed
 * sampler lookups with our flattened macro.
 *
 * Specifically:
 *   texture2D(shadowMap[i], uv) → texture(shadowMap_0, uv)  (for SHADOW_COUNT=1)
 *   or → shadowMap_FETCH(i, uv) (for SHADOW_COUNT>1)
 *
 * Actually, looking at the code more carefully, the shadow functions take
 * `sampler2D map` as a parameter (not an array index), and the loop in
 * getShadow() passes `shadowMap[i]`. We rewrite the loop to use our flattened
 * samplers.
 */
function patchShadowArrayAccess(glsl: string, maxShadows: number): string {
  if (maxShadows <= 0) return glsl;

  // Replace `shadowMap[i]` references in getShadow() loop body with our macro
  // Pattern: shadowLookup(shadowMap[i], ...)  or  shadowLookupPCSS(shadowMap[i], ...)
  let result = glsl;

  // For single shadow: replace shadowMap[0] and shadowMap[i] with shadowMap_0
  if (maxShadows === 1) {
    result = result.replace(/shadowMap\[(\w+)\]/g, 'shadowMap_0');
  } else {
    // For multiple shadows, we can't index sampler arrays in WebGL2
    // The AT code uses `shadowMap[i]` inside an unrolled loop —
    // after unrolling, i becomes a literal. Rewrite known indices:
    for (let i = 0; i < maxShadows; i++) {
      result = result.replace(new RegExp(`shadowMap\\[${i}\\]`, 'g'), `shadowMap_${i}`);
    }
    // Any remaining dynamic `shadowMap[i]` → workaround using if-chain
    // (should be eliminated by loop unrolling, but safety net)
    result = result.replace(
      /shadowMap\[(\w+)\]/g,
      '/* [AT] dynamic shadowMap[$1] — see shadowMap_FETCH */',
    );
  }

  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 7  Loop Unrolling (WebGL2 compatibility)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WebGL2 requires loop bounds to be compile-time constants.
 * AT uses `#pragma unroll_loop` as a hint — we already stripped it.
 * But we also need to ensure `for (int i = 0; i < NUM_LIGHTS; i++)`
 * compiles correctly. Since NUM_LIGHTS is now a #define, this should work.
 *
 * For SHADOW_COUNT loops, same principle applies.
 *
 * This function does a final sanity pass: if there's a loop bound
 * that's still symbolic (not a #define), manually inline it.
 */
function ensureLoopBoundsResolved(glsl: string, defines: Map<string, number>): string {
  let result = glsl;
  for (const [name, value] of defines) {
    // Replace loop conditions like `i < NUM_LIGHTS` where NUM_LIGHTS isn't #defined yet
    // (safety: our preamble already #defines them, but just in case)
    const re = new RegExp(`\\b${name}\\b`, 'g');
    // Only replace if not already in a #define line
    result = result.replace(re, (match) => String(value));
  }
  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  Vertex Shader: AT Standard Lighting + PBR Vertex Setup
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build the combined vertex shader from AT's lighting.vs + pbr.vs.
 *
 * AT vertex pipeline:
 *   1. lighting.vs → setupLight(position, normal) → vPos, vNormal, vWorldPos, vViewDir
 *   2. pbr.vs → setupPBR(position) → vUv, vUv2, vV, vWorldNormal
 *   3. Main: gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0)
 */
function buildVertexShader(
  loader: ATShaderLoader,
  config: Required<ATLightingConfig>,
): string {
  if (config.customVertex) return config.customVertex;

  const lightingVS = stripATPreprocessor(loader.getShader('lighting.vs'));
  const pbrVS = stripATPreprocessor(loader.getShader('pbr.vs'));

  const preamble = buildPreamble(config, true);

  return `${preamble}
// ── AT built-in uniforms (WebGL2) ──
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat3 normalMatrix;
uniform vec3 cameraPosition;
uniform vec2 uTiling;
uniform vec2 uOffset;

// ── Attributes ──
in vec3 position;
in vec3 normal;
in vec2 uv;
in vec2 uv2;

// ── Varyings (shared with fragment) ──
out vec3 vPos;
out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vViewDir;
out vec2 vUv;
out vec2 vUv2;
out vec3 vV;
out vec3 vWorldNormal;

// ── AT lighting.vs (setupLight) ────────────────────────────────────────────
${lightingVS}

// ── AT pbr.vs (setupPBR) ──────────────────────────────────────────────────
${pbrVS}

// ── Main ──────────────────────────────────────────────────────────────────
void main() {
    vec3 pos = position;

    // Setup lighting varyings
    setupLight(pos, normal);

    // Setup PBR varyings
    setupPBR(pos);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
`;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 9  Fragment Shader: Full AT Lighting + PBR + Shadows
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build the combined fragment shader from AT's resolved shader modules.
 *
 * Dependency chain (fully resolved):
 *   lighting.fs
 *     → LightingCommon.glsl
 *       → AreaLights.glsl  (LTC_Evaluate, LTC_Uv, ...)
 *       → Phong.glsl       (calcBlinnPhong, phong, schlick)
 *       → lworldLight, lrange, lclamp, lcrange
 *       → lightDirectional, lightPoint, lightCone, lightArea
 *     → getCombinedColor, getStandardColor, getPointLightColor, ...
 *
 *   pbr.fs
 *     → unpackNormalPBR, SRGBtoLinear, linearToSRGB, RGBMToLinear
 *     → fresnelSphericalGaussianRoughness, sampleSphericalMap
 *     → getIBLContribution, getNormal, getPBR
 *     → Uncharted2 tone-mapping
 *
 *   shadows.fs
 *     → PCSShadowConfig, poissonDisk, PCSS / PCF
 *     → shadowLookup, shadowLookupPCSS
 *     → getShadow, getShadowPCSS
 */
function buildFragmentShader(
  loader: ATShaderLoader,
  config: Required<ATLightingConfig>,
): string {
  if (config.customFragment) return config.customFragment;

  // Get fully-resolved shader sources (all #require deps inlined)
  let lightingFS = loader.getShader('lighting.fs');
  let pbrFS = loader.getShader('pbr.fs');
  let shadowsFS = loader.getShader('shadows.fs');

  // Strip AT preprocessor
  lightingFS = stripATPreprocessor(lightingFS);
  pbrFS = stripATPreprocessor(pbrFS);
  shadowsFS = stripATPreprocessor(shadowsFS);

  // Patch shadow sampler array access for WebGL2
  shadowsFS = patchShadowArrayAccess(shadowsFS, config.maxShadows);

  // Resolve any remaining symbolic constants in loop bounds
  const defines = new Map<string, number>([
    ['NUM_LIGHTS', config.maxLights],
    ['SHADOW_COUNT', config.maxShadows],
  ]);
  lightingFS = ensureLoopBoundsResolved(lightingFS, defines);
  shadowsFS = ensureLoopBoundsResolved(shadowsFS, defines);

  const preamble = buildPreamble(config, false);
  const lightUniforms = buildLightUniformDecls(config);

  return `${preamble}
// ── AT built-in uniforms (WebGL2 fragment) ──
uniform mat4 modelViewMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat3 normalMatrix;
uniform vec3 cameraPosition;

// ── Varyings from vertex ──
in vec3 vPos;
in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vViewDir;
in vec2 vUv;
in vec2 vUv2;
in vec3 vV;
in vec3 vWorldNormal;

// ── LightConfig struct (AT Lighting.glsl) ──
struct LightConfig {
    vec3 normal;
    bool phong;
    bool areaToPoint;
    float phongAttenuation;
    float phongShininess;
    vec3 phongColor;
    vec3 lightColor;
    bool overrideColor;
};

${lightUniforms}

// ── AT lighting.fs (PBR + IBL + all light types) ──────────────────────────
// Resolved dependency chain: AreaLights.glsl → Phong.glsl → LightingCommon.glsl → lighting.fs
// Total: ~800 lines GLSL (area lights + phong + point/dir/cone/area + combined color)

${lightingFS}

// ── AT pbr.fs (Cook-Torrance IBL) ─────────────────────────────────────────
// IBL contribution: env diffuse/specular + BRDF LUT + Fresnel + tone-mapping
// Total: ~210 lines GLSL

${pbrFS}

// ── AT shadows.fs (PCF/PCSS) ──────────────────────────────────────────────
// Poisson disk soft shadows + PCSS blocker search
// Total: ~360 lines GLSL

${shadowsFS}

// ── Main ──────────────────────────────────────────────────────────────────
void main() {
    // PBR base color
    vec4 pbrColor = getPBR();

    // Shadow
    float shadow = getShadow(vPos, vNormal);

    // Combined lighting (all dynamic lights)
    vec3 lightColor_combined = getCombinedColor();

    // Final composite
    vec3 finalColor = pbrColor.rgb * shadow + lightColor_combined;

    _fragColor = vec4(finalColor, pbrColor.a);
}
`;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 10  ATLightingImport — Main Class
// ─────────────────────────────────────────────────────────────────────────────

export class ATLightingImport {
  /** The compiled WebGL program (AstroProgram wrapper). */
  readonly program: AstroProgram;

  /** Full parsed uniform list from the resolved GLSL. */
  readonly uniforms: ReadonlyArray<ATUniformInfo>;

  /** Configuration used to create this instance. */
  readonly config: Readonly<Required<ATLightingConfig>>;

  /** The resolved vertex shader source (for debugging). */
  readonly vertexSource: string;

  /** The resolved fragment shader source (for debugging). */
  readonly fragmentSource: string;

  /** The ATShaderLoader instance (retains all parsed shaders for later queries). */
  readonly loader: ATShaderLoader;

  private readonly _renderer: AstroRenderer;

  // ── Constructor (private — use ATLightingImport.create()) ─────────────────

  private constructor(
    renderer: AstroRenderer,
    program: AstroProgram,
    uniforms: ATUniformInfo[],
    config: Required<ATLightingConfig>,
    vertexSource: string,
    fragmentSource: string,
    loader: ATShaderLoader,
  ) {
    this._renderer = renderer;
    this.program = program;
    this.uniforms = uniforms;
    this.config = config;
    this.vertexSource = vertexSource;
    this.fragmentSource = fragmentSource;
    this.loader = loader;
  }

  // ── Factory ──────────────────────────────────────────────────────────────

  /**
   * Load AT compiled.vs, resolve all shader dependencies, parse uniforms,
   * compile and link a WebGL program.
   *
   * @param renderer — AstroRenderer instance (WebGL context provider)
   * @param compiledVsPath — path to compiled.vs (fetch or fs.readFile)
   * @param config — optional configuration overrides
   */
  static async create(
    renderer: AstroRenderer,
    compiledVsPath: string,
    config?: ATLightingConfig,
  ): Promise<ATLightingImport> {
    const cfg = { ...DEFAULT_CONFIG, ...config };

    // ── 1. Load and parse compiled.vs ──
    const loader = new ATShaderLoader();
    await loader.load(compiledVsPath);

    // ── 2. Build vertex + fragment sources ──
    const vertexSource = buildVertexShader(loader, cfg);
    const fragmentSource = buildFragmentShader(loader, cfg);

    // ── 3. Parse uniforms from both shaders ──
    const vsUniforms = parseUniforms(vertexSource, 'vertex');
    const fsUniforms = parseUniforms(fragmentSource, 'common');

    // Tag uniforms by their AT module origin
    const taggedUniforms = fsUniforms.map((u) => {
      if (u.name.startsWith('light') || u.name === 'tLTC1' || u.name === 'tLTC2') {
        return { ...u, source: 'lighting' as const };
      }
      if (u.name.startsWith('shadow') || u.name.startsWith('Shadow')) {
        return { ...u, source: 'shadow' as const };
      }
      if (
        u.name.startsWith('tBase') ||
        u.name.startsWith('tMRO') ||
        u.name.startsWith('tNormal') ||
        u.name.startsWith('tLUT') ||
        u.name.startsWith('tEnv') ||
        u.name.startsWith('tLight') ||
        u.name.startsWith('uMRON') ||
        u.name.startsWith('uEnv') ||
        u.name.startsWith('uTint') ||
        u.name.startsWith('uTiling') ||
        u.name.startsWith('uOffset') ||
        u.name.startsWith('uHDR') ||
        u.name.startsWith('uUse')
      ) {
        return { ...u, source: 'pbr' as const };
      }
      return u;
    });

    // Merge vertex + fragment uniforms (dedup by name)
    const allUniforms = new Map<string, ATUniformInfo>();
    for (const u of [...vsUniforms, ...taggedUniforms]) {
      if (!allUniforms.has(u.name)) {
        allUniforms.set(u.name, u);
      }
    }

    // Resolve symbolic array sizes
    const defines = new Map<string, number>([
      ['NUM_LIGHTS', cfg.maxLights],
      ['SHADOW_COUNT', cfg.maxShadows],
    ]);
    const resolvedUniforms = resolveArraySizes(
      Array.from(allUniforms.values()),
      defines,
    );

    // ── 4. Compile WebGL program ──
    const program = new AstroProgram(renderer, vertexSource, fragmentSource);

    // ── 5. Warm up uniform location cache ──
    program.use();
    for (const u of resolvedUniforms) {
      if (u.arraySize > 0) {
        // For arrays, cache each element's location
        for (let i = 0; i < u.arraySize; i++) {
          program.uniformLocation(`${u.name}[${i}]`);
        }
      } else {
        program.uniformLocation(u.name);
      }
    }

    return new ATLightingImport(
      renderer,
      program,
      resolvedUniforms,
      cfg,
      vertexSource,
      fragmentSource,
      loader,
    );
  }

  // ── Light Binding ────────────────────────────────────────────────────────

  /**
   * Upload light array data to GPU uniforms.
   *
   * Maps ATLight[] → AT's uniform arrays:
   *   lightColor[i]      ← light.color + [1.0]
   *   lightPos[i]        ← light.position + [0.0]
   *   lightData[i]       ← light.data
   *   lightData2[i]      ← light.data2
   *   lightData3[i]      ← light.data3
   *   lightProperties[i] ← light.properties
   */
  bindLights(lights: ATLight[]): void {
    const p = this.program;
    const n = Math.min(lights.length, this.config.maxLights);

    for (let i = 0; i < n; i++) {
      const l = lights[i];
      p.setVec4(`lightColor[${i}]`, l.color[0], l.color[1], l.color[2], 1.0);
      p.setVec4(`lightPos[${i}]`, l.position[0], l.position[1], l.position[2], 0.0);
      p.setVec4(`lightData[${i}]`, l.data[0], l.data[1], l.data[2], l.data[3]);
      p.setVec4(`lightData2[${i}]`, l.data2[0], l.data2[1], l.data2[2], l.data2[3]);
      p.setVec4(`lightData3[${i}]`, l.data3[0], l.data3[1], l.data3[2], l.data3[3]);
      p.setVec4(
        `lightProperties[${i}]`,
        l.properties[0],
        l.properties[1],
        l.properties[2],
        l.type, // .w = light type (1.0 dir, 2.0 point, 3.0 cone, 4.0 area)
      );
    }

    // Zero out unused light slots (ensure they're disabled: properties.w < 1.0)
    for (let i = n; i < this.config.maxLights; i++) {
      p.setVec4(`lightProperties[${i}]`, 0, 0, 0, 0);
    }
  }

  // ── Shadow Binding ───────────────────────────────────────────────────────

  /**
   * Bind shadow map textures and matrices.
   *
   * @param shadows — array of shadow configs (max: config.maxShadows)
   */
  bindShadows(shadows: ATShadowConfig[]): void {
    const p = this.program;
    const n = Math.min(shadows.length, this.config.maxShadows);

    for (let i = 0; i < n; i++) {
      const s = shadows[i];
      p.setTexture(`shadowMap_${i}`, s.map);
      p.setMatrix4(`shadowMatrix[${i}]`, s.matrix);
      p.setFloat(`shadowSize[${i}]`, s.size);
      p.setVec3(`shadowLightPos[${i}]`, s.lightPos[0], s.lightPos[1], s.lightPos[2]);
    }
  }

  // ── PBR Texture Binding ──────────────────────────────────────────────────

  /**
   * Bind the full PBR texture set for AT's IBL pipeline.
   */
  bindPBR(textures: ATPBRTextures): void {
    const p = this.program;
    p.setTexture('tBaseColor', textures.baseColor);
    p.setTexture('tMRO', textures.mro);
    p.setTexture('tNormal', textures.normal);
    p.setTexture('tLUT', textures.lut);
    p.setTexture('tEnvDiffuse', textures.envDiffuse);
    p.setTexture('tEnvSpecular', textures.envSpecular);
    if (textures.lightmap) {
      p.setTexture('tLightmap', textures.lightmap);
      p.setFloat('uUseLightmap', 1.0);
    } else {
      p.setFloat('uUseLightmap', 0.0);
    }
  }

  /**
   * Set PBR material parameters.
   */
  setPBRParams(params: {
    tint?: [number, number, number];
    tiling?: [number, number];
    offset?: [number, number];
    /** Metallic / Roughness / Occlusion / Normal intensity overrides [M, R, O, N] */
    mron?: [number, number, number, number];
    /** Environment intensity / specular boost / unused */
    env?: [number, number, number];
    /** HDR flag (1.0 = RGBM, 0.0 = sRGB) */
    hdr?: number;
    lightmapIntensity?: number;
    linearOutput?: boolean;
  }): void {
    const p = this.program;
    if (params.tint) p.setVec3('uTint', ...params.tint);
    if (params.tiling) p.setVec2('uTiling', ...params.tiling);
    if (params.offset) p.setVec2('uOffset', ...params.offset);
    if (params.mron) p.setVec4('uMRON', ...params.mron);
    if (params.env) p.setVec3('uEnv', ...params.env);
    if (params.hdr !== undefined) p.setFloat('uHDR', params.hdr);
    if (params.lightmapIntensity !== undefined) {
      p.setFloat('uLightmapIntensity', params.lightmapIntensity);
    }
    if (params.linearOutput !== undefined) {
      p.setFloat('uUseLinearOutput', params.linearOutput ? 1.0 : 0.0);
    }
  }

  // ── Matrix Binding ───────────────────────────────────────────────────────

  /**
   * Set the standard transform matrices (call once per draw call).
   */
  setMatrices(matrices: {
    model: Float32Array;
    view: Float32Array;
    projection: Float32Array;
    modelView: Float32Array;
    normal: Float32Array; // 3×3
    cameraPosition: [number, number, number];
  }): void {
    const p = this.program;
    p.setMatrix4('modelMatrix', matrices.model);
    p.setMatrix4('viewMatrix', matrices.view);
    p.setMatrix4('projectionMatrix', matrices.projection);
    p.setMatrix4('modelViewMatrix', matrices.modelView);
    p.setMatrix3('normalMatrix', matrices.normal);
    p.setVec3('cameraPosition', ...matrices.cameraPosition);
  }

  // ── LTC Textures (Area Lights) ───────────────────────────────────────────

  /**
   * Bind LTC lookup textures for area light rendering.
   * Only needed if config.enableAreaLights === true.
   */
  bindLTCTextures(ltc1: WebGLTexture, ltc2: WebGLTexture): void {
    this.program.setTexture('tLTC1', ltc1);
    this.program.setTexture('tLTC2', ltc2);
  }

  // ── Query Helpers ────────────────────────────────────────────────────────

  /** Get uniform info by name. */
  getUniformInfo(name: string): ATUniformInfo | undefined {
    return this.uniforms.find((u) => u.name === name);
  }

  /** Get all uniforms from a specific AT module. */
  getUniformsBySource(source: ATUniformInfo['source']): ATUniformInfo[] {
    return this.uniforms.filter((u) => u.source === source);
  }

  /** Get the list of all shader names available in the loaded bundle. */
  listShaders(): string[] {
    return this.loader.listShaders();
  }

  /** Get a specific resolved shader by name (for debugging or custom use). */
  getResolvedShader(name: string): string {
    return stripATPreprocessor(this.loader.getShader(name));
  }

  // ── Dispose ──────────────────────────────────────────────────────────────

  dispose(): void {
    this.program.dispose();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 11  Helper Factories
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a default directional light (AT type 1).
 */
export function createDirectionalLight(
  direction: [number, number, number],
  color: [number, number, number] = [1, 1, 1],
  intensity = 1.0,
  minThreshold = 0.0,
): ATLight {
  return {
    type: 1,
    position: direction, // directional light "position" = light direction in AT
    color,
    data: [0, 0, 0, 0],
    data2: [0, 0, 0, 0],
    data3: [0, 0, 0, 0],
    properties: [intensity, 9999, minThreshold, 1.0],
  };
}

/**
 * Create a point light (AT type 2).
 */
export function createPointLight(
  position: [number, number, number],
  color: [number, number, number] = [1, 1, 1],
  intensity = 1.0,
  range = 10.0,
  minThreshold = 0.0,
): ATLight {
  return {
    type: 2,
    position,
    color,
    data: [0, 0, 0, 0],
    data2: [0, 0, 0, 0],
    data3: [0, 0, 0, 0],
    properties: [intensity, range, minThreshold, 2.0],
  };
}

/**
 * Create a spot/cone light (AT type 3).
 */
export function createConeLight(
  position: [number, number, number],
  direction: [number, number, number],
  color: [number, number, number] = [1, 1, 1],
  intensity = 1.0,
  range = 10.0,
  coneAngle = 30.0,
  feather = 1.0,
  minThreshold = 0.0,
): ATLight {
  return {
    type: 3,
    position,
    color,
    data: [direction[0], direction[1], direction[2], coneAngle],
    data2: [feather, 0, 0, 0],
    data3: [0, 0, 0, 0],
    properties: [intensity, range, minThreshold, 3.0],
  };
}

/**
 * Create an area light (AT type 4 — requires LTC textures).
 */
export function createAreaLight(
  position: [number, number, number],
  halfWidth: [number, number, number],
  halfHeight: [number, number, number],
  color: [number, number, number] = [1, 1, 1],
  intensity = 1.0,
  range = 10.0,
  roughness = 0.5,
): ATLight {
  return {
    type: 4,
    position,
    color,
    data: [position[0], position[1], position[2], roughness],
    data2: [halfWidth[0], halfWidth[1], halfWidth[2], 0],
    data3: [halfHeight[0], halfHeight[1], halfHeight[2], 0],
    properties: [intensity, range, 0, 4.0],
  };
}
