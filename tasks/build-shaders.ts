#!/usr/bin/env tsx
/**
 * tasks/build-shaders.ts — Shader build task
 *
 * Reads every .frag / .vert in src/lib/shaders/, resolves:
 *   #include lygia/…   — inlines from upstream/lygia (recursive)
 *   #require(name)     — inlines a sibling shader file (recursive)
 *
 * Writes output to src/lib/shaders/compiled.vs using the AT bundle format:
 *
 *   {@ }name{@ }GLSL source...{@ }name2{@ }GLSL source2...
 *
 * ShaderLoader.ts splits on "{@ }" at runtime to map name → GLSL source.
 *
 * Usage:
 *   tsx tasks/build-shaders.ts
 *   bun tasks/build-shaders.ts
 */

import * as fs   from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

// ─── Paths ────────────────────────────────────────────────────────────────────

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const REPO_ROOT   = path.resolve(__dirname, '..');
const SHADERS_DIR = path.join(REPO_ROOT, 'src', 'lib', 'shaders');
const LYGIA_ROOT  = path.join(REPO_ROOT, 'upstream', 'lygia');
const OUTPUT_FILE = path.join(SHADERS_DIR, 'compiled.vs');

/** AT bundle delimiter — must match ShaderLoader.ts split pattern */
const DELIM = '{@}';

// ─── Lygia #include resolver ──────────────────────────────────────────────────

/**
 * Recursively inline `#include lygia/…` / `#include "lygia/…"` lines.
 * `seen` prevents double-inclusion within one top-level shader compilation.
 */
function resolveLygiaIncludes(source: string, seen: Set<string>): string {
  return source.replace(
    /^[ \t]*#include\s+["']?(lygia\/[^\s"']+)["']?[ \t]*$/gm,
    (_match, lygiaPath: string) => {
      const absPath = path.join(LYGIA_ROOT, lygiaPath.replace(/^lygia\//, ''));

      if (seen.has(absPath)) {
        return `// [build-shaders] already inlined: ${lygiaPath}`;
      }
      seen.add(absPath);

      if (!fs.existsSync(absPath)) {
        console.warn(`  ⚠  lygia not found: ${lygiaPath}`);
        return `// [build-shaders] missing lygia include: ${lygiaPath}`;
      }

      const lygiaSource = fs.readFileSync(absPath, 'utf8');
      const resolved    = resolveLygiaIncludes(lygiaSource, seen);
      return (
        `// ── #include "${lygiaPath}" ──────────────────────────────────────\n` +
        resolved +
        `\n// ── end ${path.basename(lygiaPath)} ──────────────────────────────────────────`
      );
    },
  );
}

// ─── Local #require resolver ──────────────────────────────────────────────────

/**
 * Recursively inline `#require(filename)` directives — sibling shader files.
 * `seenRequires` guards against circular/duplicate requires within one shader.
 */
function resolveRequires(
  source:       string,
  seenRequires: Set<string>,
  seenLygias:   Set<string>,
): string {
  return source.replace(
    /^[ \t]*#require\(([^)]+)\)[ \t]*$/gm,
    (_match, requiredName: string) => {
      const absPath = path.join(SHADERS_DIR, requiredName.trim());

      if (seenRequires.has(absPath)) {
        return `// [build-shaders] already required: ${requiredName}`;
      }
      seenRequires.add(absPath);

      if (!fs.existsSync(absPath)) {
        console.warn(`  ⚠  #require not found: ${requiredName}`);
        return `// [build-shaders] missing require: ${requiredName}`;
      }

      const req         = fs.readFileSync(absPath, 'utf8');
      const withLygia   = resolveLygiaIncludes(req, seenLygias);
      const withRequire = resolveRequires(withLygia, seenRequires, seenLygias);

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
  const source         = fs.readFileSync(filePath, 'utf8');
  const seenRequires   = new Set<string>([filePath]);
  const seenLygias     = new Set<string>();

  const step1 = resolveLygiaIncludes(source, seenLygias);
  const step2 = resolveRequires(step1, seenRequires, seenLygias);
  return step2;
}

// ─── Main build ───────────────────────────────────────────────────────────────

function buildShaders(): void {
  if (!fs.existsSync(SHADERS_DIR)) {
    console.error(`✗ Shaders directory not found: ${SHADERS_DIR}`);
    process.exit(1);
  }

  const entries = fs
    .readdirSync(SHADERS_DIR)
    .filter(f => /\.(frag|vert)$/.test(f))
    .sort();

  if (entries.length === 0) {
    console.warn('⚠  No .frag / .vert files found — compiled.vs will be empty.');
  }

  console.log(`AT shader build — compiling ${entries.length} shader(s) → compiled.vs\n`);

  const chunks: string[] = [];

  for (const name of entries) {
    const filePath = path.join(SHADERS_DIR, name);
    process.stdout.write(`  compiling ${name}… `);
    try {
      const compiled = compileShader(filePath);
      chunks.push(`${DELIM}${name}${DELIM}${compiled}`);
      console.log('✓');
    } catch (err) {
      console.error(`✗ failed`);
      console.error(err);
      process.exit(1);
    }
  }

  const output = chunks.join('');
  fs.writeFileSync(OUTPUT_FILE, output, 'utf8');

  console.log(
    `\n✓ compiled.vs written — ${entries.length} shaders, ${output.length} bytes\n` +
    `  → ${OUTPUT_FILE}`,
  );
}

buildShaders();
