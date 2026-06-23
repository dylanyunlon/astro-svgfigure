/**
 * shader-compile-test.ts
 *
 * Validates shaders from upstream/activetheory-assets/compiled.vs
 *
 * Strategy (headless-gl unavailable in this environment — no GPU / EGL context):
 *   1. Parse compiled.vs with the ShaderLoader split logic ({@} delimiter)
 *   2. Text-level validation for every shader:
 *        a. Not empty
 *        b. Has void main() for .fs / .vs entries that are not utility chunks
 *        c. No unresolved #require directives (only the upstream AT bundle keeps
 *           raw #require lines because AT resolves them at runtime; we flag them
 *           as warnings, not hard failures)
 *   3. GLSL structural checks (precision, bracket balance, return in main)
 *   4. headless-gl attempt for first 20 shaders — graceful fallback if unavailable
 *   5. Printed report: pass / warn / fail per shader + summary
 *
 * Run:
 *   npx tsx tests/shader-compile-test.ts
 */

import * as fs   from 'node:fs';
import * as path from 'node:path';
import * as url  from 'node:url';

// ─── Paths ────────────────────────────────────────────────────────────────────

const __dirname  = path.dirname(url.fileURLToPath(import.meta.url));
const REPO_ROOT  = path.resolve(__dirname, '..');
const COMPILED_VS = path.join(REPO_ROOT, 'upstream', 'activetheory-assets', 'compiled.vs');

// ─── Types ────────────────────────────────────────────────────────────────────

interface ShaderEntry {
  name:    string;
  content: string;
  index:   number;
}

interface ValidationResult {
  name:      string;
  index:     number;
  pass:      boolean;
  warnings:  string[];
  errors:    string[];
  glResult?: 'compiled' | 'failed' | 'skipped';
  glError?:  string;
}

// ─── 1. Parse compiled.vs (mirrors ShaderLoader.ts logic) ─────────────────────

function parseCompiledVS(raw: string): ShaderEntry[] {
  const DELIM  = '{@}';
  const parts  = raw.split(DELIM);
  const shaders: ShaderEntry[] = [];

  // Layout: [optional-preamble, name1, content1, name2, content2, …]
  for (let i = 1; i < parts.length - 1; i += 2) {
    const name    = parts[i]!.trim();
    const content = parts[i + 1]!;
    shaders.push({ name, content, index: shaders.length });
  }

  return shaders;
}

// ─── 2. Build "required-by" map ───────────────────────────────────────────────

function buildRequiredByMap(shaders: ShaderEntry[]): Map<string, string[]> {
  const requiredBy = new Map<string, string[]>();
  const REQUIRE_RE = /^[ \t]*#require\(([^)]+)\)/gm;

  for (const shader of shaders) {
    let m: RegExpExecArray | null;
    REQUIRE_RE.lastIndex = 0;
    while ((m = REQUIRE_RE.exec(shader.content)) !== null) {
      const dep = m[1]!.trim();
      if (!requiredBy.has(dep)) requiredBy.set(dep, []);
      requiredBy.get(dep)!.push(shader.name);
    }
  }

  return requiredBy;
}

// ─── 3. Text-level validation ─────────────────────────────────────────────────

/** Count occurrences of `ch` in `s`. */
function count(s: string, ch: string): number {
  let n = 0, pos = 0;
  while ((pos = s.indexOf(ch, pos)) !== -1) { n++; pos += ch.length; }
  return n;
}

function validateShader(
  shader:     ShaderEntry,
  requiredBy: Map<string, string[]>,
): Omit<ValidationResult, 'glResult' | 'glError'> {
  const { name, content } = shader;
  const errors:   string[] = [];
  const warnings: string[] = [];

  // (a) Not empty
  if (content.trim().length === 0) {
    errors.push('Content is empty');
    return { name, index: shader.index, pass: false, warnings, errors };
  }

  // (b) void main() check for .fs / .vs files
  const isExec = name.endsWith('.fs') || name.endsWith('.vs');
  if (isExec) {
    const hasMain = /void\s+main\s*\(/.test(content);
    const isUtilityChunk = requiredBy.has(name);

    if (!hasMain) {
      if (isUtilityChunk) {
        warnings.push(
          `No void main() — utility chunk (required by: ${requiredBy.get(name)!.slice(0, 2).join(', ')})`,
        );
      } else {
        // Standalone .fs/.vs without void main AND not required by anyone
        // Could still be a partial include used outside this bundle; warn not error
        warnings.push('No void main() and not #required by any other shader in this bundle');
      }
    }
  }

  // (c) Unresolved #require directives
  const REQUIRE_RE = /^[ \t]*#require\(([^)]+)\)/gm;
  const requireMatches: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = REQUIRE_RE.exec(content)) !== null) {
    requireMatches.push(m[1]!.trim());
  }
  if (requireMatches.length > 0) {
    warnings.push(
      `Contains ${requireMatches.length} unresolved #require directive(s): ` +
      requireMatches.slice(0, 3).join(', ') +
      (requireMatches.length > 3 ? ` … (+${requireMatches.length - 3} more)` : ''),
    );
  }

  // (d) GLSL structural checks (only for .fs/.vs with void main)
  if (isExec && /void\s+main\s*\(/.test(content)) {
    // Bracket balance
    const opens  = count(content, '{');
    const closes = count(content, '}');
    if (opens !== closes) {
      errors.push(`Unbalanced braces: ${opens} '{' vs ${closes} '}'`);
    }

    // Precision qualifier — only standard GLSL (no #version 300 es context here,
    // but upstream AT shaders target WebGL 1; check only for fs where it matters)
    if (name.endsWith('.fs') && !requiredBy.has(name)) {
      const hasPrecision = /precision\s+(lowp|mediump|highp)/.test(content);
      if (!hasPrecision) {
        warnings.push('Fragment shader lacks an explicit precision qualifier');
      }
    }
  }

  // (e) Syntax red flags
  if (/\binfinite\b/.test(content)) warnings.push('Contains keyword "infinite" (non-standard GLSL)');
  if (/double\s+\w/.test(content) && !/\/\/.*double/.test(content)) {
    warnings.push('Uses "double" type (unavailable in WebGL 1 GLSL)');
  }

  return {
    name,
    index: shader.index,
    pass:  errors.length === 0,
    warnings,
    errors,
  };
}

// ─── 4. headless-gl compilation attempt ──────────────────────────────────────

/** Wraps a full GLSL source (with void main) for compilation via headless-gl. */
function wrapForGL(name: string, source: string): { type: 'vertex' | 'fragment'; src: string } | null {
  if (name.endsWith('.vs')) return { type: 'vertex',   src: source };
  if (name.endsWith('.fs')) return { type: 'fragment', src: source };
  return null;  // .glsl utility shards not directly compilable
}

interface GLContext {
  createShader(type: number): WebGLShader | null;
  shaderSource(shader: WebGLShader, src: string): void;
  compileShader(shader: WebGLShader): void;
  getShaderParameter(shader: WebGLShader, pname: number): unknown;
  getShaderInfoLog(shader: WebGLShader): string | null;
  deleteShader(shader: WebGLShader): void;
  VERTEX_SHADER:   number;
  FRAGMENT_SHADER: number;
  COMPILE_STATUS:  number;
  destroy(): void;
}

async function tryGLCompile(
  shaders: ShaderEntry[],
  maxShaders = 20,
): Promise<Map<string, { result: 'compiled' | 'failed' | 'skipped'; error?: string }>> {
  const out = new Map<string, { result: 'compiled' | 'failed' | 'skipped'; error?: string }>();

  // Attempt to load headless-gl
  let gl: GLContext | null = null;
  try {
    // Dynamic import — fails gracefully if not available
    const glMod = await import('gl') as { default: (w: number, h: number) => GLContext | null };
    const glFn  = glMod.default ?? glMod;
    if (typeof glFn === 'function') {
      const ctx = (glFn as (w: number, h: number) => GLContext | null)(1, 1);
      if (ctx) {
        gl = ctx;
        console.log('\n✓ headless-gl context obtained — performing GPU compilation checks\n');
      } else {
        console.log('\n⚠  headless-gl loaded but context creation returned null');
        console.log('   (No GPU / EGL context available — text-level validation only)\n');
      }
    }
  } catch {
    console.log('\n⚠  headless-gl not available — text-level validation only\n');
  }

  // Select shaders eligible for GL compilation (.fs / .vs with void main)
  const eligible = shaders
    .filter(s => wrapForGL(s.name, s.content) && /void\s+main\s*\(/.test(s.content))
    .slice(0, maxShaders);

  if (!gl) {
    for (const s of shaders.slice(0, maxShaders)) {
      out.set(s.name, { result: 'skipped' });
    }
    return out;
  }

  for (const s of eligible) {
    const wrapped = wrapForGL(s.name, s.content)!;
    const type = wrapped.type === 'vertex' ? gl.VERTEX_SHADER : gl.FRAGMENT_SHADER;

    const shader = gl.createShader(type);
    if (!shader) { out.set(s.name, { result: 'failed', error: 'createShader returned null' }); continue; }

    gl.shaderSource(shader, wrapped.src);
    gl.compileShader(shader);

    const ok  = gl.getShaderParameter(shader, gl.COMPILE_STATUS) as boolean;
    const log = gl.getShaderInfoLog(shader) ?? '';
    gl.deleteShader(shader);

    if (ok) {
      out.set(s.name, { result: 'compiled' });
    } else {
      out.set(s.name, { result: 'failed', error: log.split('\n')[0] ?? 'compilation error' });
    }
  }

  // Mark non-eligible shaders in first-20 as skipped
  for (const s of shaders.slice(0, maxShaders)) {
    if (!out.has(s.name)) out.set(s.name, { result: 'skipped' });
  }

  try { gl.destroy(); } catch { /* ignore */ }
  return out;
}

// ─── 5. Report ────────────────────────────────────────────────────────────────

const RESET  = '\x1b[0m';
const RED    = '\x1b[31m';
const GREEN  = '\x1b[32m';
const YELLOW = '\x1b[33m';
const CYAN   = '\x1b[36m';
const BOLD   = '\x1b[1m';
const DIM    = '\x1b[2m';

function padRight(s: string, n: number): string {
  return s.length >= n ? s : s + ' '.repeat(n - s.length);
}

function printReport(results: ValidationResult[]): void {
  const HR = '─'.repeat(72);

  console.log(`\n${BOLD}${CYAN}Shader Compilation & Validation Report${RESET}`);
  console.log(`${CYAN}compiled.vs → upstream/activetheory-assets/compiled.vs${RESET}`);
  console.log(HR);

  // Header row
  console.log(
    `${BOLD}${padRight('#',   4)}` +
    `${padRight('Shader Name', 42)}` +
    `${padRight('Text', 6)}` +
    `${padRight('GL', 10)}` +
    `Notes${RESET}`,
  );
  console.log(HR);

  for (const r of results) {
    const textMark  = r.pass ? `${GREEN}PASS${RESET}` : `${RED}FAIL${RESET}`;
    let   glMark: string;

    switch (r.glResult) {
      case 'compiled': glMark = `${GREEN}OK${RESET}`;       break;
      case 'failed':   glMark = `${RED}ERR${RESET}`;        break;
      case 'skipped':  glMark = `${DIM}skip${RESET}`;       break;
      default:         glMark = `${DIM}n/a${RESET}`;
    }

    const noteLines: string[] = [
      ...r.errors.map(e => `${RED}✗ ${e}${RESET}`),
      ...r.warnings.map(w => `${YELLOW}⚠ ${w}${RESET}`),
    ];
    if (r.glError) noteLines.push(`${RED}GL: ${r.glError}${RESET}`);
    if (r.glResult === 'compiled') noteLines.push(`${DIM}GPU compile OK${RESET}`);

    const firstNote = noteLines[0] ?? '';
    console.log(
      `${padRight(String(r.index + 1), 4)}` +
      `${padRight(r.name, 42)}` +
      `${padRight(textMark, 14)}` +   // extra width for escape codes
      `${padRight(glMark, 18)}` +
      firstNote,
    );

    for (let i = 1; i < noteLines.length; i++) {
      console.log(' '.repeat(62) + noteLines[i]);
    }
  }

  console.log(HR);

  // Summary
  const total        = results.length;
  const textPassed   = results.filter(r => r.pass).length;
  const textFailed   = results.filter(r => !r.pass).length;
  const withWarnings = results.filter(r => r.warnings.length > 0).length;
  const glCompiled   = results.filter(r => r.glResult === 'compiled').length;
  const glFailed     = results.filter(r => r.glResult === 'failed').length;
  const glSkipped    = results.filter(r => r.glResult === 'skipped').length;

  console.log(`\n${BOLD}Summary${RESET}`);
  console.log(`  Total shaders parsed : ${total}`);
  console.log(`  ${GREEN}Text-level PASS      : ${textPassed}${RESET}`);
  console.log(`  ${textFailed > 0 ? RED : DIM}Text-level FAIL      : ${textFailed}${textFailed > 0 ? RESET : RESET}`);
  console.log(`  ${YELLOW}With warnings        : ${withWarnings}${RESET}`);
  if (glCompiled + glFailed + glSkipped > 0) {
    console.log(`\n  GPU compile (first 20 eligible shaders):`);
    console.log(`    ${GREEN}Compiled OK : ${glCompiled}${RESET}`);
    console.log(`    ${glFailed > 0 ? RED : DIM}Failed      : ${glFailed}${RESET}`);
    console.log(`    ${DIM}Skipped     : ${glSkipped}${RESET}`);
  }

  console.log('');
  if (textFailed === 0) {
    console.log(`${GREEN}${BOLD}✓ All ${total} shaders pass text-level validation.${RESET}`);
  } else {
    console.log(`${RED}${BOLD}✗ ${textFailed} shader(s) failed text-level validation.${RESET}`);
  }
  console.log('');
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  // 1. Read compiled.vs
  if (!fs.existsSync(COMPILED_VS)) {
    console.error(`${RED}ERROR: compiled.vs not found at:${RESET}\n  ${COMPILED_VS}`);
    process.exit(1);
  }

  const raw = fs.readFileSync(COMPILED_VS, 'utf8');
  console.log(`Read compiled.vs  (${(raw.length / 1024).toFixed(1)} KB)`);

  // 2. Parse
  const shaders = parseCompiledVS(raw);
  console.log(`Parsed ${shaders.length} shaders via {@ } delimiter`);

  // 3. Build required-by map
  const requiredBy = buildRequiredByMap(shaders);

  // 4. Text-level validation
  const textResults = shaders.map(s => validateShader(s, requiredBy));

  // 5. headless-gl attempt (first 20)
  const glResults = await tryGLCompile(shaders, 20);

  // 6. Merge results
  const results: ValidationResult[] = textResults.map(r => {
    const gl = glResults.get(r.name);
    return {
      ...r,
      glResult: gl?.result,
      glError:  gl?.error,
    };
  });

  // 7. Print report
  printReport(results);

  // Exit with non-zero if any hard failures
  const failures = results.filter(r => !r.pass || r.glResult === 'failed');
  process.exit(failures.length > 0 ? 1 : 0);
}

main().catch(err => {
  console.error(`${RED}Unexpected error:${RESET}`, err);
  process.exit(1);
});
