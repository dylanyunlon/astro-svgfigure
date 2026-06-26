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
 * M1213 fixes:
 *   - Bridge parses compiled.vs {@} blocks directly via _parseCompiledVs()
 *   - Class shaders (#!ATTRIBUTES blocks) have vertex/fragment extracted
 *     and stored in _shaderMap (no dependency on ATShaderLoader being ready)
 *   - ATShaderLoader is loaded in parallel; if it succeeds it is used for
 *     #require-aware resolution (richer GLSL); bridge map is the fallback
 *   - Single-block parse failures skip that block without crashing others
 *   - initATShaderPipeline returns Promise<void> (callers don't need the loader)
 */

import { ATShaderLoader } from './at-shader-loader';
import Program from '../../../upstream/nanogl/src/program';
import { compileShader, executePasses, type ShaderPass } from './nanogl-shader-executor';

// ── Module state ───────────────────────────────────────────────────────────────

/** ATShaderLoader instance — used for #require-aware getProgram() if available */
let _loader: ATShaderLoader | null = null;

/** Bridge-owned registry: baseName (no .glsl) → { vertex, fragment } raw GLSL */
const _shaderMap = new Map<string, { vertex: string; fragment: string }>();

/** Raw block text for library chunks (lookup by name including .glsl suffix) */
const _rawMap = new Map<string, string>();

let _initialized = false;

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * initATShaderPipeline — fetch compiled.vs and register all AT shaders.
 *
 * Two paths run in parallel:
 *   1. _parseCompiledVs(raw) — bridge-owned direct parse (always works)
 *   2. _loader.load(url)     — ATShaderLoader for #require resolution (best-effort)
 *
 * If ATShaderLoader.load() fails, the bridge map is used alone.
 * Individual block parse failures skip that block without affecting others.
 */
export async function initATShaderPipeline(compiledVsUrl: string): Promise<void> {
  let raw: string;

  try {
    const resp = await fetch(compiledVsUrl);
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status} ${resp.statusText} — ${compiledVsUrl}`);
    }
    raw = await resp.text();
  } catch (e) {
    throw new Error(
      `[AT Pipeline Bridge] failed to fetch compiled.vs from "${compiledVsUrl}": ${(e as Error).message}`,
    );
  }

  // Parse {@} blocks into _shaderMap (always)
  _parseCompiledVs(raw);

  // Also initialise ATShaderLoader for #require resolution (best-effort)
  try {
    _loader = new ATShaderLoader();
    await _loader.load(compiledVsUrl);
  } catch (e) {
    console.warn('[AT Pipeline Bridge] ATShaderLoader init failed (bridge map only):', (e as Error).message);
    _loader = null;
  }

  _initialized = true;

  console.log(
    `[AT Pipeline Bridge] compiled.vs parsed: ` +
      `${_shaderMap.size} class shaders, ${_rawMap.size} total entries` +
      (_loader ? `, ATShaderLoader ready (${_loader.listShaders().length} shaders)` : ', ATShaderLoader unavailable'),
  );
}

/**
 * getATProgram — compile a nanogl Program from a named AT shader.
 *
 * Lookup order:
 *   1. ATShaderLoader.getProgram(name)  — #require-resolved, preamble-prefixed
 *   2. Bridge _shaderMap                — direct vertex + fragment (no #require inlining)
 *
 * Returns null (never throws) if shader not found, is a library chunk,
 * or GL compilation fails.
 *
 * @param gl   WebGL context
 * @param name shader block name — e.g. 'PhysicalShader', 'ColorMaterial',
 *             'PBR' — with or without the '.glsl' suffix
 */
export function getATProgram(
  gl: WebGLRenderingContext,
  name: string,
): { program: Program; vertSrc: string; fragSrc: string } | null {
  if (!_initialized) {
    console.error('[AT Pipeline Bridge] not initialized — call initATShaderPipeline() first');
    return null;
  }

  const key = name.endsWith('.glsl') ? name.slice(0, -5) : name;

  let vertSrc: string | undefined;
  let fragSrc: string | undefined;

  // Path 1: ATShaderLoader (handles #require, preambles)
  if (_loader) {
    try {
      const { vertex, fragment } = _loader.getProgram(key);
      vertSrc = vertex;
      fragSrc = fragment;
    } catch (_) {
      // Not a class shader or not found — fall through to bridge map
    }
  }

  // Path 2: Bridge _shaderMap fallback
  if (!vertSrc || !fragSrc) {
    const entry = _shaderMap.get(key);
    if (!entry) {
      console.warn(`[AT Pipeline Bridge] shader "${key}" not found in compiled.vs`);
      return null;
    }
    vertSrc = entry.vertex;
    fragSrc = entry.fragment;
  }

  try {
    const program = compileShader(gl, vertSrc, fragSrc);
    return { program, vertSrc, fragSrc };
  } catch (e) {
    console.error(`[AT Pipeline Bridge] compile error for "${key}":`, e);
    return null;
  }
}

/**
 * buildATPassChain — build a multi-pass FBO execution chain from shader names.
 *
 *   const chain = buildATPassChain(gl, [
 *     { name: 'SplatShader',    uniforms: { uPoint: [0.5,0.5], uRadius: 0.01 } },
 *     { name: 'AdvectionShader', uniforms: { uDt: 0.016 } },
 *   ]);
 *   executePasses(gl, chain);
 */
export function buildATPassChain(
  gl: WebGLRenderingContext,
  passes: Array<{ name: string; uniforms?: Record<string, unknown> }>,
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
          if (typeof (prog as Record<string, unknown>)[key] === 'function') {
            (prog as Record<string, (...args: unknown[]) => void>)[key](val);
          }
        }
      },
    });
  }

  return result;
}

/**
 * getATShaderSource — return GLSL source without compiling.
 * For class shaders: vertex + fragment concatenated.
 * For library chunks: raw GLSL text.
 */
export function getATShaderSource(name: string): string | null {
  if (!_initialized) return null;

  const key = name.endsWith('.glsl') ? name.slice(0, -5) : name;

  // Try ATShaderLoader first for #require-resolved source
  if (_loader) {
    try {
      const { vertex, fragment } = _loader.getProgram(key);
      return `// --- vertex ---\n${vertex}\n// --- fragment ---\n${fragment}`;
    } catch (_) {
      try {
        return _loader.getShader(key);
      } catch (_) {
        // fall through to bridge map
      }
    }
  }

  // Bridge _shaderMap
  const entry = _shaderMap.get(key);
  if (entry) {
    return `// --- vertex ---\n${entry.vertex}\n// --- fragment ---\n${entry.fragment}`;
  }

  // Raw library chunk
  const raw = _rawMap.get(name) ?? _rawMap.get(`${key}.glsl`);
  return raw ?? null;
}

/**
 * listATShaders — list all class shader names (vertex+fragment pairs).
 * Returns names from the bridge _shaderMap (always populated by initATShaderPipeline).
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

// ── Internal parser ───────────────────────────────────────────────────────────

/**
 * _parseCompiledVs — split compiled.vs on {@} and register all shader blocks.
 *
 * Format: file starts with {@}, so split('{@}') gives:
 *   ['', name0, code0, name1, code1, ...]
 *
 * Class blocks (containing #!ATTRIBUTES) have vertex + fragment extracted
 * via _parseClassBlock(). Library blocks are stored in _rawMap only.
 * Individual block failures skip that block and log a warning.
 */
function _parseCompiledVs(raw: string): void {
  _shaderMap.clear();
  _rawMap.clear();

  const parts = raw.split('{@}');

  for (let i = 1; i + 1 < parts.length; i += 2) {
    const blockName = parts[i].trim();
    const blockCode = parts[i + 1];

    if (!blockName) continue;

    // Store raw code for library lookups
    _rawMap.set(blockName, blockCode);

    // Class shaders have #!ATTRIBUTES
    if (blockCode.includes('#!ATTRIBUTES')) {
      try {
        _parseClassBlock(blockName, blockCode);
      } catch (e) {
        console.warn(
          `[AT Pipeline Bridge] skipping block "${blockName}": ${(e as Error).message}`,
        );
        // Other shaders unaffected
      }
    }
  }
}

/**
 * _parseClassBlock — extract vertex + fragment from a {@}Name.glsl{@}...{@} block.
 *
 * Structure:
 *   #!ATTRIBUTES  (may be empty)
 *   #!UNIFORMS    uniform ...;
 *   #!VARYINGS    varying ...;
 *   #!SHADER: Name.vs   void main() { ... }
 *   #!SHADER: Name.fs   void main() { ... }
 *
 * Preamble (ATTRIBUTES + UNIFORMS + VARYINGS) is prepended to both shaders.
 * Result stored under baseName (e.g. 'ColorMaterial' for 'ColorMaterial.glsl').
 */
function _parseClassBlock(blockName: string, code: string): void {
  const baseName = blockName.endsWith('.glsl') ? blockName.slice(0, -5) : blockName;

  // Extract preamble sections
  const attributes = _extractSection(code, '#!ATTRIBUTES', ['#!UNIFORMS', '#!VARYINGS', '#!SHADER:']);
  const uniforms   = _extractSection(code, '#!UNIFORMS',   ['#!VARYINGS', '#!SHADER:']);
  const varyings   = _extractSection(code, '#!VARYINGS',   ['#!SHADER:']);

  const preamble = [attributes, uniforms, varyings]
    .filter((s) => s.trim().length > 0)
    .join('\n');

  // Find all #!SHADER: markers
  const shaderRegex = /#!SHADER:\s*(\S+)/g;
  const markers: Array<{ name: string; headerStart: number; codeStart: number }> = [];
  let m: RegExpExecArray | null;

  while ((m = shaderRegex.exec(code)) !== null) {
    markers.push({ name: m[1], headerStart: m.index, codeStart: m.index + m[0].length });
  }

  if (markers.length < 2) {
    throw new Error(`expected ≥2 #!SHADER: blocks, found ${markers.length}`);
  }

  // Extract each sub-shader body
  const subBodies = new Map<string, string>();

  for (let j = 0; j < markers.length; j++) {
    const start = markers[j].codeStart;
    const end   = j + 1 < markers.length ? markers[j + 1].headerStart : code.length;
    const body  = code.slice(start, end).trim();
    const canonical = _canonicalSubName(markers[j].name, baseName);
    subBodies.set(canonical, body);
    _rawMap.set(canonical, body);
  }

  // Find vertex + fragment keys
  const vsKey = _findSubKey(subBodies, baseName, 'vs');
  const fsKey = _findSubKey(subBodies, baseName, 'fs');

  const vsBody = subBodies.get(vsKey);
  const fsBody = subBodies.get(fsKey);

  if (vsBody === undefined) throw new Error(`vertex sub-shader "${vsKey}" missing`);
  if (fsBody === undefined) throw new Error(`fragment sub-shader "${fsKey}" missing`);

  const vertex   = preamble ? `${preamble}\n${vsBody}` : vsBody;
  const fragment = preamble ? `${preamble}\n${fsBody}` : fsBody;

  _shaderMap.set(baseName, { vertex, fragment });
}

/** Extract text between startMarker and the first of endMarkers. */
function _extractSection(code: string, startMarker: string, endMarkers: string[]): string {
  const startIdx = code.indexOf(startMarker);
  if (startIdx === -1) return '';

  const contentStart = startIdx + startMarker.length;
  let endIdx = code.length;

  for (const em of endMarkers) {
    const idx = code.indexOf(em, contentStart);
    if (idx !== -1 && idx < endIdx) endIdx = idx;
  }

  return code.slice(contentStart, endIdx).trim();
}

/**
 * Canonicalise a #!SHADER: name to a unique key.
 *   "Vertex" | "Vertex.vs"   → "BaseName.vs"
 *   "Fragment" | "Fragment.fs" → "BaseName.fs"
 *   "Foo.vs"                  → "Foo.vs"  (already qualified)
 */
function _canonicalSubName(rawName: string, baseName: string): string {
  const lower = rawName.toLowerCase();
  if (lower === 'vertex'   || lower === 'vertex.vs')   return `${baseName}.vs`;
  if (lower === 'fragment' || lower === 'fragment.fs') return `${baseName}.fs`;
  return rawName;
}

/** Find the canonical key for vs or fs in subBodies. */
function _findSubKey(
  subBodies: Map<string, string>,
  baseName: string,
  type: 'vs' | 'fs',
): string {
  const primary = `${baseName}.${type}`;
  if (subBodies.has(primary)) return primary;

  for (const key of subBodies.keys()) {
    if (key.endsWith(`.${type}`)) return key;
  }

  throw new Error(`no ${type === 'vs' ? 'vertex' : 'fragment'} sub-shader in "${baseName}"`);
}


// ── Auto-generated export stubs (M1155) ──
