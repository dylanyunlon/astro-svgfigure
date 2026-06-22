/**
 * compiler.ts — AT-style shader build-time compiler
 *
 * Reads every .frag / .vert in this directory, resolves two directive forms:
 *
 *   #require(name)          — inline another shader from this directory by
 *                             filename (e.g. #require(colormap.frag)).
 *                             Guards against double-inclusion via a seen-set.
 *
 *   #include lygia/…        — resolve from the project's upstream/lygia tree
 *   #include "lygia/…"      — (both quote-styles accepted).
 *                             Recursively resolves nested includes inside the
 *                             lygia file so the output is fully self-contained.
 *
 * Output format  (compiled.vs) — AT bundle delimiters:
 *
 *   {@ }name{@ }GLSL source...{@ }name2{@ }GLSL source2...
 *
 *   ShaderLoader.ts splits on "{@ }" and maps name → source at runtime.
 *
 * Runtime API (re-exported from index.ts):
 *
 *   getShader(name: string): string
 *     Returns the compiled GLSL for a named shader.  Throws if not found.
 *     Backed by the same compiled.vs bundle via Vite's ?raw import.
 *
 * Usage (build-time, Bun / tsx):
 *
 *   bun src/lib/shaders/compiler.ts
 *   tsx src/lib/shaders/compiler.ts
 *
 * The script writes compiled.vs in-place next to itself.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

// ─── Paths ────────────────────────────────────────────────────────────────────

const SHADERS_DIR = path.resolve(import.meta.dirname ?? __dirname);
const LYGIA_ROOT  = path.resolve(SHADERS_DIR, '../../../upstream/lygia');
const OUTPUT_FILE = path.join(SHADERS_DIR, 'compiled.vs');

// AT bundle delimiter — must match ShaderLoader.ts split pattern
const DELIM = '{@}';

// ─── Lygia inline resolver ────────────────────────────────────────────────────

/**
 * Recursively resolve `#include "lygia/..."` / `#include lygia/...` lines
 * found inside any GLSL source (shader or lygia library file).
 * `seen` prevents re-inlining the same lygia file more than once per
 * top-level shader compilation.
 */
function resolveLygiaIncludes(source: string, seen: Set<string>): string {
  // Matches both:  #include lygia/foo/bar.glsl
  //                #include "lygia/foo/bar.glsl"
  return source.replace(
    /^[ \t]*#include\s+["']?(lygia\/[^\s"']+)["']?[ \t]*$/gm,
    (_match, lygiaPath: string) => {
      const absPath = path.join(LYGIA_ROOT, lygiaPath.replace(/^lygia\//, ''));

      if (seen.has(absPath)) {
        return `// [compiler] already inlined: ${lygiaPath}`;
      }
      seen.add(absPath);

      if (!fs.existsSync(absPath)) {
        console.warn(`  ⚠  lygia not found: ${lygiaPath}`);
        return `// [compiler] missing lygia include: ${lygiaPath}`;
      }

      const lygiaSource = fs.readFileSync(absPath, 'utf8');
      // Recurse so nested lygia #includes are also resolved
      const resolved = resolveLygiaIncludes(lygiaSource, seen);
      return `// ── #include "${lygiaPath}" ──────────────────────────────────────\n${resolved}\n// ── end ${path.basename(lygiaPath)} ──────────────────────────────────────────`;
    },
  );
}

// ─── Local #require resolver ──────────────────────────────────────────────────

/**
 * Resolve `#require(filename)` directives — inline a sibling shader file.
 * `seen` is shared across the compilation of one top-level shader entry so
 * circular/duplicate #require chains are safe.
 */
function resolveRequires(
  source: string,
  seen: Set<string>,
  lygiasSeen: Set<string>,
): string {
  return source.replace(
    /^[ \t]*#require\(([^)]+)\)[ \t]*$/gm,
    (_match, requiredName: string) => {
      const absPath = path.join(SHADERS_DIR, requiredName.trim());

      if (seen.has(absPath)) {
        return `// [compiler] already required: ${requiredName}`;
      }
      seen.add(absPath);

      if (!fs.existsSync(absPath)) {
        console.warn(`  ⚠  #require not found: ${requiredName}`);
        return `// [compiler] missing require: ${requiredName}`;
      }

      const req = fs.readFileSync(absPath, 'utf8');
      // Recursively handle requires + lygia within the required file
      const withLygia   = resolveLygiaIncludes(req, lygiasSeen);
      const withRequire = resolveRequires(withLygia, seen, lygiasSeen);
      return (
        `// ── #require(${requiredName}) ────────────────────────────────────────\n` +
        withRequire +
        `\n// ── end ${requiredName} ──────────────────────────────────────────────`
      );
    },
  );
}

// ─── Compile one shader ───────────────────────────────────────────────────────

function compileShader(filePath: string): string {
  const source = fs.readFileSync(filePath, 'utf8');
  const seenRequires: Set<string> = new Set([filePath]);
  const seenLygias:   Set<string> = new Set();

  const step1 = resolveLygiaIncludes(source, seenLygias);
  const step2 = resolveRequires(step1, seenRequires, seenLygias);
  return step2;
}

// ─── Main build ───────────────────────────────────────────────────────────────

/**
 * Build compiled.vs from all .frag / .vert files in SHADERS_DIR.
 * Files are sorted alphabetically for reproducible output.
 */
export function buildCompiledVS(): void {
  const entries = fs.readdirSync(SHADERS_DIR)
    .filter(f => /\.(frag|vert)$/.test(f))
    .sort();

  const chunks: string[] = [];

  for (const name of entries) {
    const filePath = path.join(SHADERS_DIR, name);
    console.log(`  compiling ${name}…`);
    const compiled = compileShader(filePath);
    chunks.push(`${DELIM}${name}${DELIM}${compiled}`);
  }

  const output = chunks.join('');
  fs.writeFileSync(OUTPUT_FILE, output, 'utf8');
  console.log(`\n✓ compiled.vs written  (${entries.length} shaders, ${output.length} bytes)`);
}

// ─── Runtime lookup (mirrors ShaderLoader.ts) ─────────────────────────────────

/**
 * Lazily parse compiled.vs and return the GLSL source for `name`.
 * Throws if the shader is not found.
 *
 * This function is intended for use in browser/runtime code that imports
 * compiled.vs via Vite's `?raw` suffix.  The build-time script (below)
 * never calls this; it is re-exported from index.ts for consumers.
 */
let _runtimeCache: Map<string, string> | null = null;

export function getShader(name: string): string {
  if (!_runtimeCache) {
    // Dynamic import of compiled.vs via Vite ?raw is handled at call-site;
    // here we expose the parsing logic so tests can inject raw content.
    throw new Error(
      '[compiler] getShader() must be called after injectCompiledVS().\n' +
      'In browser code use ShaderLoader.getShader() instead.',
    );
  }
  const src = _runtimeCache.get(name);
  if (!src) {
    throw new Error(`[compiler] shader "${name}" not found in compiled.vs`);
  }
  return src;
}

/**
 * Populate the runtime cache from a raw compiled.vs string.
 * Call once at app start (or in tests):
 *
 *   import raw from './compiled.vs?raw';
 *   injectCompiledVS(raw);
 */
export function injectCompiledVS(raw: string): void {
  _runtimeCache = new Map();
  const parts = raw.split(DELIM);
  // Layout: [optional-preamble, name1, content1, name2, content2, …]
  for (let i = 1; i < parts.length - 1; i += 2) {
    const name    = parts[i]!.trim();
    const content = parts[i + 1]!;
    _runtimeCache.set(name, content);
  }
}

// ─── CLI entry-point ──────────────────────────────────────────────────────────
// Runs when executed directly:  bun src/lib/shaders/compiler.ts

const isMain =
  typeof process !== 'undefined' &&
  (process.argv[1]?.endsWith('compiler.ts') ||
   process.argv[1]?.endsWith('compiler.js'));

if (isMain) {
  console.log('AT shader compiler — building compiled.vs…\n');
  buildCompiledVS();
}
