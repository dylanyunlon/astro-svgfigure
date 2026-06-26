/**
 * at-shader-pipeline-bridge.ts — M938 / M1213
 *
 * 连接 compiled.vs → internal shader Map → nanogl executor。
 *
 * compiled.vs 格式:
 *   - 主分隔符: {@}name{@}code{@}name{@}code ...
 *   - Class shader 块 (含 #!ATTRIBUTES) 内部结构:
 *       #!ATTRIBUTES
 *       ...
 *       #!UNIFORMS  uniform vec3 color; ...
 *       #!VARYINGS  varying vec3 vColor; ...
 *       #!SHADER: Name.vs    ...vertex body...
 *       #!SHADER: Name.fs    ...fragment body...
 *
 * M1213 fixes (builds on M1218):
 *   - Bridge maintains own _shaderMap: parses all {@} blocks from compiled.vs
 *   - Class shaders (#!ATTRIBUTES) have vertex/fragment extracted directly
 *   - listATShaders() returns all registered shader canonical names
 *   - Single-block parse failures skip that block (no crash)
 *   - ATShaderLoader used via public load() + getProgram() for #require resolution
 */

import { ATShaderLoader } from './at-shader-loader';
import Program from '../../../upstream/nanogl/src/program';
import { compileShader, executePasses, type ShaderPass } from './nanogl-shader-executor';

// ── Internal data structures ───────────────────────────────────────────────────

interface ParsedBlock {
  /** Raw code from the {@} block */
  rawCode: string;
  /** True if block contains #!ATTRIBUTES (class shader) */
  isClass: boolean;
  /** Extracted vertex GLSL body (class shaders only) */
  vertSrc?: string;
  /** Extracted fragment GLSL body (class shaders only) */
  fragSrc?: string;
}

// ── Module state ───────────────────────────────────────────────────────────────

/** ATShaderLoader instance (for #require recursive resolution via getProgram) */
let _loader: ATShaderLoader | null = null;

/**
 * Bridge-owned shader registry.
 *   key   = canonical name (no .glsl suffix, e.g. 'PhysicalShader', 'blendmodes')
 *   value = ParsedBlock
 */
const _shaderMap: Map<string, ParsedBlock> = new Map();

// ── Internal parsing ──────────────────────────────────────────────────────────

/**
 * Determine if a #!SHADER: marker name is vertex or fragment.
 * Returns 'vertex' | 'fragment' | null.
 */
function subShaderKind(name: string): 'vertex' | 'fragment' | null {
  const lower = name.toLowerCase();
  if (lower.endsWith('.vs') || lower === 'vertex' || lower === 'vertex.vs') return 'vertex';
  if (lower.endsWith('.fs') || lower === 'fragment' || lower === 'fragment.fs') return 'fragment';
  return null;
}

/**
 * Extract vertex and fragment GLSL bodies from a class shader block.
 * Any parse failure returns empty strings (block will be skipped by caller).
 */
function parseSubShaders(code: string): { vertSrc?: string; fragSrc?: string } {
  try {
    const markerRe = /#!SHADER:\s*(\S+)/g;
    const markers: { name: string; bodyStart: number }[] = [];
    let m: RegExpExecArray | null;

    while ((m = markerRe.exec(code)) !== null) {
      markers.push({ name: m[1], bodyStart: m.index + m[0].length });
    }

    let vertSrc: string | undefined;
    let fragSrc: string | undefined;

    for (let j = 0; j < markers.length; j++) {
      const { name, bodyStart } = markers[j];
      const bodyEnd =
        j + 1 < markers.length
          ? markers[j + 1].bodyStart - markers[j + 1].name.length - '#!SHADER: '.length
          : code.length;

      const body = code.slice(bodyStart, bodyEnd).trim();
      const kind = subShaderKind(name);

      if (kind === 'vertex') vertSrc = body;
      else if (kind === 'fragment') fragSrc = body;
    }

    return { vertSrc, fragSrc };
  } catch (_) {
    return {};
  }
}

/**
 * Parse compiled.vs text and populate _shaderMap.
 *
 * Format: {@}name{@}code{@}name{@}code...
 * Single-block failures skip that block without affecting others.
 */
function parseCompiledVs(src: string): void {
  _shaderMap.clear();
  const parts = src.split('{@}');
  // parts[0] = '' (empty before first delimiter)
  let registered = 0;
  let skipped = 0;

  for (let i = 1; i + 1 < parts.length; i += 2) {
    try {
      const rawName = parts[i].trim();
      const code = parts[i + 1] ?? '';

      if (!rawName) continue;

      // Canonical name: strip .glsl suffix
      const canonName = rawName.endsWith('.glsl') ? rawName.slice(0, -5) : rawName;

      const isClass = code.includes('#!ATTRIBUTES');

      if (isClass) {
        const { vertSrc, fragSrc } = parseSubShaders(code);
        _shaderMap.set(canonName, { rawCode: code, isClass: true, vertSrc, fragSrc });
      } else {
        _shaderMap.set(canonName, { rawCode: code, isClass: false });
      }

      registered++;
    } catch (_) {
      skipped++;
    }
  }

  console.log(
    `[AT Pipeline Bridge] compiled.vs parsed: ${registered} shaders registered` +
      (skipped > 0 ? `, ${skipped} blocks skipped` : ''),
  );
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * initATShaderPipeline — load compiled.vs, build bridge shader map,
 * and initialise ATShaderLoader for #require-resolution.
 *
 * M1213: bridge now parses compiled.vs itself (parseCompiledVs) so
 *   listATShaders() / getATProgram() work without ATShaderLoader being ready.
 *   ATShaderLoader is still used for its getProgram() which handles
 *   #require() recursive inlining and preamble assembly.
 *
 * @returns ATShaderLoader instance (for callers that need advanced access)
 */
export async function initATShaderPipeline(compiledVsUrl: string): Promise<ATShaderLoader> {
  _loader = new ATShaderLoader();
  await _loader.load(compiledVsUrl);

  // Also parse compiled.vs ourselves so _shaderMap is populated for
  // fast direct lookups and listATShaders().
  // We re-use the same fetch result via ATShaderLoader's internal state;
  // but since load() is the public API, we do a second fetch here for the
  // bridge map.  compiled.vs is typically cached by the browser after the
  // first request.
  try {
    const resp = await fetch(compiledVsUrl);
    if (resp.ok) {
      const src = await resp.text();
      parseCompiledVs(src);
    }
  } catch (e) {
    console.warn('[AT Pipeline Bridge] bridge map parse failed (non-fatal):', e);
  }

  console.log(
    `[AT Pipeline Bridge] Loaded compiled.vs: ${_loader.listShaders().length} shaders in ATShaderLoader, ` +
      `${_shaderMap.size} in bridge map`,
  );
  return _loader;
}

/**
 * getATProgram — compile a WebGL program from a named AT shader.
 *
 * Lookup order:
 *   1. ATShaderLoader.getProgram(name) — handles #require, preambles
 *   2. Falls back to bridge _shaderMap raw vertex/fragment (no #require resolution)
 *
 * Returns null (never throws) if:
 *   - shader not found
 *   - shader is a library chunk (no vertex+fragment pair)
 *   - compilation fails
 *
 * @param gl   WebGL context
 * @param name shader name with or without .glsl suffix (e.g. 'PhysicalShader')
 */
export function getATProgram(
  gl: WebGLRenderingContext,
  name: string,
): { program: Program; vertSrc: string; fragSrc: string } | null {
  const canonName = name.endsWith('.glsl') ? name.slice(0, -5) : name;

  if (!_loader && _shaderMap.size === 0) {
    console.error('[AT Pipeline Bridge] not initialized. Call initATShaderPipeline() first.');
    return null;
  }

  // ── Path 1: ATShaderLoader (full #require resolution) ──────────────────────
  if (_loader) {
    let vertSrc: string | undefined;
    let fragSrc: string | undefined;

    try {
      const { vertex, fragment } = _loader.getProgram(canonName);
      vertSrc = vertex;
      fragSrc = fragment;
    } catch (_) {
      // Not a class shader or not found — fall through to bridge map
    }

    if (vertSrc && fragSrc) {
      try {
        const program = compileShader(gl, vertSrc, fragSrc);
        return { program, vertSrc, fragSrc };
      } catch (e) {
        console.error(`[AT Pipeline Bridge] compile error for "${canonName}":`, e);
        return null;
      }
    }
  }

  // ── Path 2: bridge _shaderMap (raw GLSL, no #require inlining) ─────────────
  const block = _shaderMap.get(canonName);
  if (!block || !block.isClass) {
    if (_shaderMap.size > 0) {
      console.warn(`[AT Pipeline Bridge] shader "${canonName}" not found or is a library shader`);
    }
    return null;
  }

  const { vertSrc, fragSrc } = block;
  if (!vertSrc || !fragSrc) {
    console.warn(`[AT Pipeline Bridge] shader "${canonName}" missing vertex or fragment sub-shader`);
    return null;
  }

  try {
    const program = compileShader(gl, vertSrc, fragSrc);
    return { program, vertSrc, fragSrc };
  } catch (e) {
    console.error(`[AT Pipeline Bridge] compile error for "${canonName}" (bridge fallback):`, e);
    return null;
  }
}

/**
 * buildATPassChain — build a multi-pass FBO execution chain from shader names.
 *
 *   const chain = buildATPassChain(gl, [
 *     { name: 'SplatShader',    uniforms: { uPoint: [0.5, 0.5], uRadius: 0.01 } },
 *     { name: 'AdvectionShader', uniforms: { uDt: 0.016 } },
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
      name,
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
 * getATShaderSource — return raw GLSL source for a shader (no compilation).
 *
 * For class shaders: returns vertex + fragment concatenated (via ATShaderLoader
 * with #require resolution if available; raw bridge map otherwise).
 * For library shaders: returns the raw code chunk.
 * Returns null if shader not found.
 */
export function getATShaderSource(name: string): string | null {
  const canonName = name.endsWith('.glsl') ? name.slice(0, -5) : name;

  if (_loader) {
    try {
      const { vertex, fragment } = _loader.getProgram(canonName);
      return `${vertex}\n// ---\n${fragment}`;
    } catch (_) {
      // Fallback to raw library chunk
      try {
        return _loader.getShader(canonName);
      } catch (_) {
        // not found in loader
      }
    }
  }

  // Bridge map fallback
  return _shaderMap.get(canonName)?.rawCode ?? null;
}

/**
 * listATShaders — list all registered shader canonical names.
 *
 * Returns names from the bridge _shaderMap (populated by parseCompiledVs).
 * Falls back to ATShaderLoader.listShaders() if bridge map is empty.
 * Returns [] before initATShaderPipeline() is called.
 */
export function listATShaders(): string[] {
  if (_shaderMap.size > 0) {
    return Array.from(_shaderMap.keys()).sort();
  }
  if (_loader) {
    return _loader.listShaders();
  }
  return [];
}


// ── Auto-generated export stubs (M1155) ──
