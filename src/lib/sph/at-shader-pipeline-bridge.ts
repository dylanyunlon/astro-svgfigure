/**
 * at-shader-pipeline-bridge.ts — M938: 连接 ATShaderLoader → nanogl executor
 *
 * 这是 compiled.vs 字符串到屏幕像素的最后一环:
 *   1. ATShaderLoader.load(compiledVsUrl) → 174 个 shader 拆分 + #require 递归解析
 *   2. ATShaderLoader.getProgram(name)    → { vertex, fragment } 完全解析后的 GLSL
 *   3. compileShader(gl, vert, frag)      → nanogl Program
 *   4. executePasses(gl, passes)          → multi-pass FBO chain → 屏幕
 *
 * Bug fixes (M1218):
 *   - parse() is private on ATShaderLoader; use public load() instead
 *   - _loader.names does not exist; use _loader.listShaders()
 *   - getShader(name) on class blocks returns text with #!SHADER: markers stripped
 *     (parseClassBlock removes them); use getProgram(name) which returns
 *     the correctly assembled { vertex, fragment } pair with preamble prepended
 *   - getATProgram() return type was WebGLProgram; corrected to nanogl Program
 *     so it is directly usable in ShaderPass.program (PBR pass compatible)
 */

import { ATShaderLoader } from './at-shader-loader';
import Program from '../../../upstream/nanogl/src/program';
import { compileShader, executePasses, type ShaderPass } from './nanogl-shader-executor';

let _loader: ATShaderLoader | null = null;

/**
 * initATShaderPipeline — 加载 compiled.vs 并初始化 shader loader。
 * 调用一次, 之后用 getATProgram() 获取编译好的 program。
 *
 * FIX: use loader.load(url) (async, public) instead of
 *      fetch + private parse().
 */
export async function initATShaderPipeline(compiledVsUrl: string): Promise<ATShaderLoader> {
  _loader = new ATShaderLoader();
  await _loader.load(compiledVsUrl);
  console.log(`[AT Pipeline Bridge] Loaded compiled.vs: ${_loader.listShaders().length} shaders parsed`);
  return _loader;
}

/**
 * getATProgram — 获取一个编译好的 nanogl Program。
 * 自动从 ATShaderLoader 获取 resolved GLSL (含所有 #require 依赖),
 * 然后用 nanogl executor 编译。
 *
 * FIX: use loader.getProgram(name) instead of getShader(name) + manual
 *      #!SHADER: regex.  parseClassBlock strips those markers from the stored
 *      text, so the regex in the old code never matched.  getProgram() returns
 *      the correctly assembled { vertex, fragment } pair (preamble prepended,
 *      #require resolved) and is already the right entry-point for class shaders.
 *
 * FIX: return type is now nanogl Program (not WebGLProgram) so the result
 *      plugs directly into ShaderPass.program for PBR / multi-pass use.
 *
 * @param gl   - WebGL context
 * @param name - shader block name (e.g. 'PhysicalMaterial', 'ColorMaterial')
 *               with or without the .glsl suffix
 * @returns compiled nanogl Program + source, or null if not found / not a class shader
 */
export function getATProgram(
  gl: WebGLRenderingContext,
  name: string,
): { program: Program; vertSrc: string; fragSrc: string } | null {
  if (!_loader) {
    console.error('[AT Pipeline Bridge] loader not initialized. Call initATShaderPipeline() first.');
    return null;
  }

  let vertSrc: string;
  let fragSrc: string;

  try {
    // getProgram() handles class blocks (with #!ATTRIBUTES/#!UNIFORMS/#!VARYINGS)
    // and returns the preamble-prefixed, #require-resolved vertex + fragment pair.
    const { vertex, fragment } = _loader.getProgram(name);
    vertSrc = vertex;
    fragSrc = fragment;
  } catch (e) {
    // Not a class shader (library chunk) or name not found — not directly compilable
    console.warn(`[AT Pipeline Bridge] getProgram("${name}") failed:`, (e as Error).message);
    return null;
  }

  try {
    const program = compileShader(gl, vertSrc, fragSrc);
    return { program, vertSrc, fragSrc };
  } catch (e) {
    console.error(`[AT Pipeline Bridge] compile error for "${name}":`, e);
    return null;
  }
}

/**
 * buildATPassChain — 给定一组 shader 名称, 构建 multi-pass FBO 执行链。
 *
 * 典型用法:
 *   const chain = buildATPassChain(gl, [
 *     { name: 'SplatShader', uniforms: { uPoint: [0.5, 0.5], uRadius: 0.01 } },
 *     { name: 'AdvectionShader', uniforms: { uDt: 0.016 } },
 *     { name: 'PressureShader', uniforms: { uPressure: pressureTex } },
 *   ]);
 *   executePasses(gl, chain);
 */
export function buildATPassChain(
  gl: WebGLRenderingContext,
  passes: Array<{ name: string; uniforms?: Record<string, any> }>,
): ShaderPass[] {
  const result: ShaderPass[] = [];

  for (const { name, uniforms } of passes) {
    const compiled = getATProgram(gl, name);
    if (!compiled) {
      console.warn(`[AT Pipeline Bridge] skipping pass "${name}" — not compilable`);
      continue;
    }
    result.push({
      program: compiled.program,
      fbo: null,
      setUniforms: (prog) => {
        for (const [key, val] of Object.entries(uniforms ?? {})) {
          if (typeof prog[key] === 'function') prog[key](val);
        }
      },
    });
  }

  return result;
}

/**
 * getATShaderSource — 获取解析后的 GLSL 源码 (不编译)。
 * 对 class shaders 返回 vertex + fragment 拼接; 对 library shaders 返回原始代码。
 */
export function getATShaderSource(name: string): string | null {
  if (!_loader) return null;
  try {
    const { vertex, fragment } = _loader.getProgram(name);
    return `${vertex}\n// ---\n${fragment}`;
  } catch {
    // Fallback to raw library chunk
    try {
      return _loader.getShader(name);
    } catch {
      return null;
    }
  }
}

/**
 * listATShaders — 列出所有可用的 shader 名称。
 * FIX: use listShaders() method (ATShaderLoader has no .names property).
 */
export function listATShaders(): string[] {
  if (!_loader) return [];
  return _loader.listShaders();
}


// ── Auto-generated export stubs (M1155) ──
