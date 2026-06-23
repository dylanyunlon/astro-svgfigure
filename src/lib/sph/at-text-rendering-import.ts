/**
 * at-text-rendering-import.ts — M858: AT Text3D + GLUI Text Rendering Shaders
 * ─────────────────────────────────────────────────────────────────────────────
 * 集成 ActiveTheory 的完整文字渲染系统：
 *
 *   AT 文字渲染层次：
 *
 *   Text3D.glsl         — 3D 世界空间文字，带 per-letter/word/line 动画
 *     ├─ animation attr (letter/word/line index)
 *     ├─ rangeTransition() 驱动逐字母/词/行入场
 *     ├─ cubicOut easing + rotationMatrix 翻转动画
 *     ├─ 流体扰动 (tFluid + tFluidMask) → UV 偏移
 *     └─ MSDF alpha + drawbuffer Color/Refraction 双输出
 *
 *   DefaultText.glsl    — HUD/UI 平面文字，带像素化 → 清晰 transition
 *     ├─ 格子化 UV (gridV 50→500) 实现像素化溶解效果
 *     └─ 颜色波动 (sin(time - worldPos)) + 闪烁
 *
 *   GLUIBatchText.glsl  — GPU instanced GLUI 文字批渲染
 *     ├─ offset/scale/rotation instanced attributes
 *     ├─ lrotationMatrix Z轴旋转 → 支持每字符独立旋转
 *     └─ MSDF alpha × v_uAlpha
 *
 *   GLUIBatch.glsl      — GPU instanced GLUI 通用批渲染 (纹理贴图)
 *     └─ offset/scale/rotation instanced quad，Fragment 输出白色 (由子类覆盖)
 *
 *   GLUIColor.glsl      — GLUI 纯色矩形
 *     └─ uColor + uAlpha，mix(uColor, uvColor, 0.0) → 纯色
 *
 *   GLUIObject.glsl     — GLUI 纹理对象，带时间动画透明度
 *     └─ sin(time * 2.0 + vUv.y * 2.0 - worldPos.x * 0.02) alpha 脉动
 *
 * MSDF (Multi-channel Signed Distance Field) 文字渲染：
 *   - 通过 #require(msdf.glsl) 依赖注入
 *   - msdf(sampler2D, uv) → 利用 fwidth() 反走样 alpha
 *   - strokemsdf() → 描边效果
 *   - 需要 OES_standard_derivatives (WebGL1) 或原生 (WebGL2)
 *
 * 用法 (cell 标签渲染)：
 *   const renderer = ATTextRenderingImport.create(gl, loader);
 *   renderer.parseAll();
 *   const uniforms = renderer.getUniforms('Text3D.glsl');
 *
 *   // 或直接使用 shader 源码：
 *   const src = AT_TEXT_SHADERS['Text3D.glsl'];
 *   // → 传入 ATShaderLoader.parseShader(src) 解析
 *
 * Research: M858 — cell-pubsub-loop
 * Source: upstream/activetheory-assets/compiled.vs
 */

import { ATShaderLoader } from './at-shader-loader.ts';
import type { AstroRenderer } from '../renderer/AstroRenderer.ts';

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Raw GLSL sources (extracted from compiled.vs {@} blocks)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * DefaultText.glsl — AT HUD/UI 平面 MSDF 文字着色器。
 *
 * 特色：像素化溶解 transition (gridV 50→500)，世界坐标颜色波动，高速闪烁。
 * Uniforms: tMap(MSDF atlas), uColor, uAlpha, uMouse
 * Varyings: vUv, vWorldPos
 */
export const DefaultTextGLSL = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;
uniform vec2 uMouse;

#!VARYINGS

varying vec2 vUv;
varying vec3 vWorldPos;

#!SHADER: DefaultText.vs

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
}

#!SHADER: DefaultText.fs

#require(msdf.glsl)

void main() {
    float transition = smoothstep(0.3, 0.8, uAlpha);
    float gridV = mix(50.0, 500.0, transition);
    vec2 gridSize = vec2(gridV*3.0, floor(gridV/(resolution.x/resolution.y)));
    vec2 uv = floor(vUv * gridSize) / gridSize;
    uv += (1.0-transition) * (1.0/gridV) * vec2(0.2, 0.5);
    uv = mix(uv, vUv,transition);

    float alpha = msdf(tMap, uv);
    alpha *= uAlpha;

    vec3 color = uColor;
    color = mix(color, vec3(0.5, 0.5, 1.0), 0.1 + sin(time - vWorldPos.x * 0.01 + vWorldPos.y * 0.005 + alpha * 10.0) * 0.1);

    alpha *= 0.9 + sin(time*40.0) * 0.1 * smoothstep(0.2, 0.15, abs(uAlpha-0.5));

    gl_FragColor = vec4(color, alpha);

}`;

/**
 * Text3D.glsl — AT 3D 世界空间 MSDF 文字着色器。
 *
 * 特色：
 *   - per-letter/word/line 入场动画 (animation.xyz = letter/word/line index)
 *   - rangeTransition + cubicOut easing
 *   - uRotate 轴旋转翻转
 *   - 流体扰动 UV 偏移 (tFluid, tFluidMask)
 *   - #drawbuffer Color / Refraction 双 MRT 输出
 *
 * Uniforms: tMap, uColor, uAlpha, uOpacity, uTranslate, uRotate,
 *           uTransition, uWordCount, uLineCount, uLetterCount,
 *           uByWord, uByLine, uPadding, uBoundingMin, uBoundingMax,
 *           uScrollDelta, uMouse, tFluid, tFluidMask
 * Attributes: animation (vec3: letter/word/line)
 * Varyings: vTrans, vUv, vPos, vWorldPos
 */
export const Text3DGLSL = `#!ATTRIBUTES
attribute vec3 animation;

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;
uniform float uOpacity;
uniform vec3 uTranslate;
uniform vec3 uRotate;
uniform float uTransition;
uniform float uWordCount;
uniform float uLineCount;
uniform float uLetterCount;
uniform float uByWord;
uniform float uByLine;
uniform float uPadding;
uniform vec3 uBoundingMin;
uniform vec3 uBoundingMax;
uniform float uScrollDelta;
uniform vec2 uMouse;
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;

#!VARYINGS
varying float vTrans;
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;

#!SHADER: Vertex

#require(range.glsl)
#require(eases.glsl)
#require(rotation.glsl)
#require(conditionals.glsl)

void main() {
    vUv = uv;
    vTrans = 1.0;

    vec3 pos = position;

    if (uTransition > 0.0 && uTransition < 1.0) {
        float padding = uPadding;
        float letter = (animation.x + 1.0) / uLetterCount;
        float word = (animation.y + 1.0) / uWordCount;
        float line = (animation.z + 1.0) / uLineCount;

        float letterTrans = rangeTransition(uTransition, letter, padding);
        float wordTrans = rangeTransition(uTransition, word, padding);
        float lineTrans = rangeTransition(uTransition, line, padding);

        vTrans = mix(cubicOut(letterTrans), cubicOut(wordTrans), uByWord);
        vTrans = mix(vTrans, cubicOut(lineTrans), uByLine);

        float invTrans = (1.0 - vTrans);
        vec3 nRotate = normalize(uRotate);
        vec3 axisX = vec3(1.0, 0.0, 0.0);
        vec3 axisY = vec3(0.0, 1.0, 0.0);
        vec3 axisZ = vec3(0.0, 0.0, 1.0);
        vec3 axis = mix(axisX, axisY, when_gt(nRotate.y, nRotate.x));
        axis = mix(axis, axisZ, when_gt(nRotate.z, nRotate.x));
        pos = vec3(vec4(position, 1.0) * rotationMatrix(axis, radians(max(max(uRotate.x, uRotate.y), uRotate.z) * invTrans)));
        pos += uTranslate * invTrans;
    }

    vPos = pos;
	vWorldPos = vec3(modelMatrix * vec4(pos, 1.0));

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(msdf.glsl)
#require(simplenoise.glsl)
#require(transformUV.glsl)

vec2 getBoundingUV() {
    vec2 uv;
    uv.x = crange(vPos.x, uBoundingMin.x, uBoundingMax.x, 0.0, 1.0);
    uv.y = crange(vPos.y, uBoundingMin.y, uBoundingMax.y, 0.0, 1.0);
    return uv;
}

void main() {
    vec2 uv = vUv;
    vec2 screenuv = gl_FragCoord.xy / resolution;
    vec2 squareScreenuv = scaleUV(screenuv, vec2(1.0, resolution.x/resolution.y));
    vec2 mouse = scaleUV(vec2(uMouse.x, 1.0-uMouse.y), vec2(1.0, resolution.x/resolution.y));

    mouse += cnoise(screenuv*10.0 + time * 0.2 + length(mouse) * 5.0) * 0.01;

    vec2 fluid = texture2D(tFluid, screenuv).xy;
    float fluidMask = smoothstep(0.0, 1.0, texture2D(tFluidMask, screenuv).r);
    float fluidPush = pow(abs(fluid.x)*0.01, 2.5);
    float fluidEdge = fluidPush * smoothstep(0.0, 0.5, fluidMask) * smoothstep(1.0, 0.8, fluidMask);

    //uv.y -= uScrollDelta * 0.1 * mix(-1.0, 1.0, step(0.05, mod(uv.x, 0.5))) * mod(uv.y, 0.3);
    uv += fluidEdge * 0.1;

    float alpha = msdf(tMap, uv);

    //float noise = 0.5 + smoothstep(-1.0, 1.0, cnoise(vec3(vUv*50.0, time* 0.3))) * 0.5;

    vec4 color = vec4(uColor, alpha * uAlpha * uOpacity * vTrans);

    float mouseLen = (1.0-step(0.1, length(squareScreenuv-mouse)));

    // float lines = sin(screenuv.x * resolution.x * 0.5) * (0.5 + cnoise(screenuv*30.0 + time * 0.2));
    // lines = step(0.2, lines);

    vec2 lineUV = screenuv + fluidPush * 0.1;
    float lines = fract(screenuv.x * 300.0) * fract(screenuv.y * 300.0);
    lines = step(0.7, lines);
    color.a = mix(color.a, lines, fluidEdge);

    #drawbuffer Color gl_FragColor = color;
    #drawbuffer Refraction gl_FragColor = color;
}`;

/**
 * GLUIBatch.glsl — AT GLUI GPU instanced 批渲染着色器 (通用)。
 *
 * 每个实例通过 offset/scale/rotation instanced attributes 定位。
 * Fragment 默认输出 vec4(1.0)——子类覆盖用于特定效果。
 * Attributes: offset(vec3), scale(vec2), rotation(float)
 * Uniforms: tMap, uColor, uAlpha
 */
export const GLUIBatchGLSL = `#!ATTRIBUTES
attribute vec3 offset;
attribute vec2 scale;
attribute float rotation;
//attributes

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;
//varyings

#!SHADER: Vertex

mat4 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
    oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
    oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
    0.0,                                0.0,                                0.0,                                1.0);
}

void main() {
    vUv = uv;
    //vdefines

    vec3 pos = vec3(rotationMatrix(vec3(0.0, 0.0, 1.0), rotation) * vec4(position, 1.0));
    pos.xy *= scale;
    pos.xyz += offset;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment
void main() {
    gl_FragColor = vec4(1.0);
}`;

/**
 * GLUIBatchText.glsl — AT GLUI GPU instanced 批文字渲染着色器。
 *
 * 与 GLUIBatch 相同的 instanced 布局，但 Fragment 使用 MSDF alpha。
 * 支持 //custommain 注入点 (AT 预处理器扩展)。
 * Attributes: offset(vec3), scale(vec2), rotation(float)
 * Fragment: msdf(tMap, vUv) × v_uAlpha
 */
export const GLUIBatchTextGLSL = `#!ATTRIBUTES
attribute vec3 offset;
attribute vec2 scale;
attribute float rotation;
//attributes

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;
//varyings

#!SHADER: Vertex

mat4 lrotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
    oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
    oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
    0.0,                                0.0,                                0.0,                                1.0);
}

void main() {
    vUv = uv;
    //vdefines

    vec3 pos = vec3(lrotationMatrix(vec3(0.0, 0.0, 1.0), rotation) * vec4(position, 1.0));

    //custommain

    pos.xy *= scale;
    pos += offset;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

#require(msdf.glsl)

void main() {
    float alpha = msdf(tMap, vUv);

    gl_FragColor.rgb = v_uColor;
    gl_FragColor.a = alpha * v_uAlpha;
}`;

/**
 * GLUIColor.glsl — AT GLUI 纯色矩形着色器。
 *
 * 最轻量的 GLUI 着色器，用于纯色 UI 矩形。
 * Uniforms: uColor(vec3), uAlpha(float)
 * Note: mix(uColor, uvColor, 0.0) → 输出纯 uColor（第三参数为0预留调试用）
 */
export const GLUIColorGLSL = `#!ATTRIBUTES

#!UNIFORMS
uniform vec3 uColor;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;

#!SHADER: GLUIColor.vs
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: GLUIColor.fs
void main() {
    vec2 uv = vUv;
    vec3 uvColor = vec3(uv, 1.0);
    gl_FragColor = vec4(mix(uColor, uvColor, 0.0), uAlpha);
}`;

/**
 * GLUIObject.glsl — AT GLUI 纹理对象着色器，带时间 alpha 脉动。
 *
 * 用于 GLUI 纹理 UI 元素。alpha = 0.8 + sin(time * 2.0 + ...) * 0.2
 * 产生轻微呼吸感。#require(transformUV.glsl)。
 * Uniforms: tMap(sampler2D), uAlpha(float)
 * Varyings: vUv, vWorldPos
 */
export const GLUIObjectGLSL = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;
varying vec3 vWorldPos;

#!SHADER: GLUIObject.vs
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
}

#!SHADER: GLUIObject.fs

#require(transformUV.glsl)

void main() {
    // float transition = smoothstep(0.0, 0.8, uAlpha);
    // float gridV = mix(20.0, 100.0, transition);
    // vec2 gridSize = vec2(gridV, floor(gridV/(resolution.x/resolution.y)));
    // vec2 uv = floor(vUv * gridSize) / gridSize;
    // uv += (1.0-transition) * (1.0/gridV) * 0.4;
    // uv = mix(uv, vUv,transition);

    vec4 color = texture2D(tMap, vUv);
    color.a *= 0.8 + sin(time * 2.0 + vUv.y * 2.0 - vWorldPos.x * 0.02) * 0.2;
    color.a *= uAlpha;
    gl_FragColor = color;
}`;

// ─────────────────────────────────────────────────────────────────────────────
// § 2  Shader registry
// ─────────────────────────────────────────────────────────────────────────────

/** Map of all AT text/GLUI shader names to their raw GLSL source. */
export const AT_TEXT_SHADERS = {
  'DefaultText.glsl'  : DefaultTextGLSL,
  'Text3D.glsl'       : Text3DGLSL,
  'GLUIBatch.glsl'    : GLUIBatchGLSL,
  'GLUIBatchText.glsl': GLUIBatchTextGLSL,
  'GLUIColor.glsl'    : GLUIColorGLSL,
  'GLUIObject.glsl'   : GLUIObjectGLSL,
} as const;

export type ATTextShaderName = keyof typeof AT_TEXT_SHADERS;

// ─────────────────────────────────────────────────────────────────────────────
// § 3  Types
// ─────────────────────────────────────────────────────────────────────────────

/** Parsed segments of an AT shader file (attributes / uniforms / varyings / vs / fs). */
export interface ATTextShaderParsed {
  name       : ATTextShaderName;
  /** Raw attributes section (after #!ATTRIBUTES) */
  attributes : string;
  /** Raw uniforms section (after #!UNIFORMS) */
  uniforms   : string;
  /** Raw varyings section (after #!VARYINGS) */
  varyings   : string;
  /** Vertex shader body */
  vertex     : string;
  /** Fragment shader body */
  fragment   : string;
  /** All #require() dependencies found in vertex + fragment */
  requires   : string[];
}

/**
 * Runtime uniform descriptor extracted from GLSL source.
 * Used by ATTextRenderingImport to pre-cache WebGL uniform locations.
 */
export interface ATTextUniform {
  name     : string;
  type     : string;
  arraySize: number;  // 0 = scalar
}

/**
 * Instanced attribute descriptor for GLUIBatch / GLUIBatchText.
 */
export interface ATGLUIInstancedAttr {
  name : string;
  type : 'vec2' | 'vec3' | 'float';
  /** Number of float components (2/3/1) */
  size : number;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  Parser
// ─────────────────────────────────────────────────────────────────────────────

const SECTION_RE    = /#!ATTRIBUTES(.*?)(?=#!UNIFORMS|#!VARYINGS|#!SHADER:|$)/s;
const UNIFORMS_RE   = /#!UNIFORMS(.*?)(?=#!VARYINGS|#!SHADER:|$)/s;
const VARYINGS_RE   = /#!VARYINGS(.*?)(?=#!SHADER:|$)/s;
const SHADER_VS_RE  = /#!SHADER:\s*(?:Vertex|DefaultText\.vs|GLUIColor\.vs|GLUIObject\.vs)(.*?)(?=#!SHADER:|$)/s;
const SHADER_FS_RE  = /#!SHADER:\s*(?:Fragment|DefaultText\.fs|GLUIColor\.fs|GLUIObject\.fs)(.*?)$/s;
const REQUIRE_RE    = /#require\(([^)]+)\)/g;
const UNIFORM_RE    = /uniform\s+(\w+)\s+(\w+)(?:\[(\d+)\])?;/g;

/**
 * Parse a raw AT GLSL source into its constituent sections.
 * AT shaders use a custom preprocessor format:
 *   #!ATTRIBUTES  — instanced/per-vertex attributes
 *   #!UNIFORMS    — uniform declarations
 *   #!VARYINGS    — varying declarations
 *   #!SHADER: Name  — shader stage body
 *   #require(x)   — inline shader dependency
 */
export function parseATTextShader(name: ATTextShaderName, src: string): ATTextShaderParsed {
  const getSection = (re: RegExp) => (re.exec(src)?.[1] ?? '').trim();

  const attributes = getSection(SECTION_RE);
  const uniforms   = getSection(UNIFORMS_RE);
  const varyings   = getSection(VARYINGS_RE);
  const vertex     = getSection(SHADER_VS_RE);
  const fragment   = getSection(SHADER_FS_RE);

  const requireSrc = vertex + '\n' + fragment;
  const requires: string[] = [];
  let m: RegExpExecArray | null;
  REQUIRE_RE.lastIndex = 0;
  while ((m = REQUIRE_RE.exec(requireSrc)) !== null) {
    if (!requires.includes(m[1])) requires.push(m[1]);
  }

  return { name, attributes, uniforms, varyings, vertex, fragment, requires };
}

/**
 * Extract uniform list from a raw GLSL uniforms section string.
 */
export function extractUniforms(uniformsSrc: string): ATTextUniform[] {
  const result: ATTextUniform[] = [];
  let m: RegExpExecArray | null;
  UNIFORM_RE.lastIndex = 0;
  while ((m = UNIFORM_RE.exec(uniformsSrc)) !== null) {
    result.push({
      type     : m[1],
      name     : m[2],
      arraySize: m[3] ? parseInt(m[3], 10) : 0,
    });
  }
  return result;
}

/**
 * Extract instanced attributes from GLUIBatch / GLUIBatchText source.
 */
export function extractGLUIInstancedAttrs(src: string): ATGLUIInstancedAttr[] {
  const ATTR_RE = /^attribute\s+(vec2|vec3|float)\s+(\w+);/gm;
  const result: ATGLUIInstancedAttr[] = [];
  let m: RegExpExecArray | null;
  while ((m = ATTR_RE.exec(src)) !== null) {
    const type = m[1] as ATGLUIInstancedAttr['type'];
    const size = type === 'vec3' ? 3 : type === 'vec2' ? 2 : 1;
    result.push({ name: m[2], type, size });
  }
  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  ATTextRenderingImport — runtime integration class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Configuration for a text rendering pass.
 */
export interface ATTextRenderConfig {
  /** Which AT shader to use. */
  shader: ATTextShaderName;
  /** MSDF font atlas WebGL texture (pre-loaded). */
  msdfTexture: WebGLTexture;
  /** Text color in linear RGB. */
  color      : [number, number, number];
  /** Overall alpha (0–1). */
  alpha      : number;
  /** For Text3D: opacity multiplier (combined with alpha). */
  opacity   ?: number;
  /** For Text3D: per-letter/word/line transition progress (0–1). */
  transition?: number;
}

/**
 * ATTextRenderingImport — manages AT text + GLUI shader programs for
 * cell label GPU rendering.
 *
 * Architecture:
 *   compiled.vs → ATShaderLoader.parseBundle() → resolve #require()
 *   → parseATTextShader() → build WebGL programs
 *   → cache uniform locations → render
 *
 * MSDF pipeline:
 *   1. Font atlas generated offline (msdf-bmfont-xml or msdf-atlas-gen)
 *   2. Glyph quads uploaded as instanced geometry (GLUIBatchText)
 *      or as a single mesh (Text3D / DefaultText)
 *   3. Fragment shader: msdf(tMap, uv) → anti-aliased coverage alpha
 *   4. Blend: ONE, ONE_MINUS_SRC_ALPHA (premultiplied alpha)
 */
export class ATTextRenderingImport {
  readonly gl    : WebGL2RenderingContext;
  readonly loader: ATShaderLoader;

  /** Parsed shader descriptors for all 6 text/GLUI shaders. */
  readonly parsed = {} as Record<ATTextShaderName, ATTextShaderParsed>;

  private constructor(gl: WebGL2RenderingContext, loader: ATShaderLoader) {
    this.gl     = gl;
    this.loader = loader;
  }

  /**
   * Parse all 6 shader sources from the embedded literals.
   * Does not require a compiled.vs URL — shaders are inlined above.
   */
  parseAll(): void {
    for (const [name, src] of Object.entries(AT_TEXT_SHADERS) as [ATTextShaderName, string][]) {
      this.parsed[name] = parseATTextShader(name, src);
    }
  }

  /** Get uniform list for a given shader. */
  getUniforms(name: ATTextShaderName): ATTextUniform[] {
    const p = this.parsed[name];
    if (!p) throw new Error(`Shader ${name} not yet parsed — call parseAll() first`);
    return extractUniforms(p.uniforms);
  }

  /** Get instanced attribute layout for GLUIBatch / GLUIBatchText. */
  getGLUIInstancedAttrs(name: 'GLUIBatch.glsl' | 'GLUIBatchText.glsl'): ATGLUIInstancedAttr[] {
    return extractGLUIInstancedAttrs(AT_TEXT_SHADERS[name]);
  }

  /**
   * Summary of all shader dependencies (via #require()) across all shaders.
   * Used to pre-resolve needed utility shaders from compiled.vs.
   */
  getAllRequires(): Map<ATTextShaderName, string[]> {
    const map = new Map<ATTextShaderName, string[]>();
    for (const [name, p] of Object.entries(this.parsed) as [ATTextShaderName, ATTextShaderParsed][]) {
      if (p.requires.length) map.set(name, p.requires);
    }
    return map;
  }

  // ── Factory ────────────────────────────────────────────────────────────────

  /**
   * Create an ATTextRenderingImport instance.
   * Parses all shader sources eagerly.
   */
  static create(gl: WebGL2RenderingContext, loader: ATShaderLoader): ATTextRenderingImport {
    const instance = new ATTextRenderingImport(gl, loader);
    instance.parseAll();
    return instance;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  MSDF utility GLSL — shared across DefaultText, Text3D, GLUIBatchText
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Inline MSDF utility GLSL (mirrors msdf.glsl from compiled.vs).
 *
 * Required by: DefaultText.fs, Text3D.fs, GLUIBatchText.fs
 * Resolved at build time by ATShaderLoader.resolveRequires().
 *
 * Uses fwidth() (requires OES_standard_derivatives in WebGL1;
 * always available in WebGL2).
 */
export const MSDF_GLSL = /* glsl */`
float msdf(vec3 tex, vec2 uv) {
    float signedDist = max(min(tex.r, tex.g), min(max(tex.r, tex.g), tex.b)) - 0.5;
    float d = fwidth(signedDist);
    float alpha = smoothstep(-d, d, signedDist);
    if (alpha < 0.01) discard;
    return alpha;
}

float msdf(sampler2D tMap, vec2 uv) {
    vec3 tex = texture2D(tMap, uv).rgb;
    return msdf(tex, uv);
}

float strokemsdf(sampler2D tMap, vec2 uv, float stroke, float padding) {
    vec3 tex = texture2D(tMap, uv).rgb;
    float signedDist = max(min(tex.r, tex.g), min(max(tex.r, tex.g), tex.b)) - 0.5;
    float t = stroke;
    float alpha = smoothstep(-t, -t + padding, signedDist) * smoothstep(t, t - padding, signedDist);
    return alpha;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 7  Default export
// ─────────────────────────────────────────────────────────────────────────────

export default ATTextRenderingImport;
