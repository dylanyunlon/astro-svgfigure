/**
 * at-shader-pipeline-bridge.ts — M938: 连接 ATShaderLoader → nanogl executor
 *
 * 这是 compiled.vs 字符串到屏幕像素的最后一环:
 *   1. ATShaderLoader.parse(compiledVsSrc) → 172 个 shader 拆分 + #require 递归解析
 *   2. ATShaderLoader.getShader(name) → 完全解析后的 GLSL 字符串
 *   3. compileShader(gl, vert, frag) → nanogl Program
 *   4. executePasses(gl, passes) → multi-pass FBO chain → 屏幕
 */







import { ATShaderLoader } from './at-shader-loader';
import { compileShader, executePasses, type ShaderPass } from './nanogl-shader-executor';

// [orphan-precise] let _loader: ATShaderLoader | null = null;

// [orphan-precise] /**
// [orphan-precise]  * initATShaderPipeline — 加载 compiled.vs 并初始化 shader loader。
// [orphan-precise]  * 调用一次, 之后用 getATProgram() 获取编译好的 program。
// [orphan-precise]  */
export async function initATShaderPipeline(compiledVsUrl: string): Promise<ATShaderLoader> {
  const resp = await fetch(compiledVsUrl);
  const src = await resp.text();
  _loader = new ATShaderLoader();
  _loader.parse(src);
  console.log(`[AT Pipeline Bridge] Loaded compiled.vs: ${_loader.names.length} shaders parsed`);
  return _loader;
}

/**
 * getATProgram — 获取一个编译好的 WebGL program。
 * 自动从 ATShaderLoader 获取 resolved GLSL (含所有 #require 依赖),
 * 然后用 nanogl executor 编译。
 *
 * @param gl - WebGL context
 * @param name - shader 名称 (如 'PhysicalShader', 'HydraBloom', 'SplatShader')
 * @returns compiled WebGL program, 或 null 如果 shader 不存在
 */
export function getATProgram(
  gl: WebGLRenderingContext,
  name: string,
): { program: WebGLProgram; vertSrc: string; fragSrc: string } | null {
  if (!_loader) {
    console.error('[AT Pipeline Bridge] loader not initialized. Call initATShaderPipeline() first.');
    return null;
  }

  // ATShaderLoader.getShader() 返回 #require 完全递归解析后的 GLSL
  const resolved = _loader.getShader(name);
  if (!resolved) {
    console.warn(`[AT Pipeline Bridge] shader "${name}" not found in compiled.vs`);
    return null;
  }

  // Class shaders 有 #!SHADER: Name.vs / #!SHADER: Name.fs 分隔
  // Library shaders 只是 GLSL 代码片段 (没有 vertex/fragment 分离)
  const vsMatch = resolved.match(/#!SHADER:\s*[\w.]*\.vs\n([\s\S]*?)(?=#!SHADER:|$)/);
  const fsMatch = resolved.match(/#!SHADER:\s*[\w.]*\.fs\n([\s\S]*?)(?=#!SHADER:|$)/);

  if (!vsMatch || !fsMatch) {
    // Library shader — 不能直接编译为 program, 返回原始代码
    // (调用者需要自己组合 vertex + fragment)
    return null;
  }

  const vertSrc = vsMatch[1].trim();
  const fragSrc = fsMatch[1].trim();

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
      uniforms: uniforms ?? {},
    });
  }

  return result;
}

/**
 * getATShaderSource — 获取解析后的 GLSL 源码 (不编译)。
 * 用于需要自定义编译流程的场景。
 */
export function getATShaderSource(name: string): string | null {
  if (!_loader) return null;
  return _loader.getShader(name) ?? null;
}

/**
 * listATShaders — 列出所有可用的 shader 名称。
 */
export function listATShaders(): string[] {
  if (!_loader) return [];
  return _loader.names;
}
