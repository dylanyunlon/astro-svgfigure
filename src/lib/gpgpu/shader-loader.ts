/**
 * shader-loader.ts — AT compiled.vs ShaderPack loader
 *
 * Loads Active Theory's compiled.vs shader pack format:
 *   {at}ShaderName{at}GLSL code{at}NextShaderName{at}...
 *
 * Features:
 *   - load(url)    : fetch compiled.vs, parse into Map<name, rawCode>
 *   - resolve(name): recursively inline #require(xxx.glsl) dependencies
 *   - get(name)    : return fully-resolved shader (cached after first resolve)
 *
 * #require semantics:
 *   A line `#require(foo.glsl)` is replaced with the fully-resolved source of
 *   the shader named "foo.glsl".  Circular dependencies are detected and
 *   skipped so we never loop infinitely.
 *
 * Usage:
 *   const pack = new ShaderPack();
 *   await pack.load('/assets/shaders/compiled.vs');
 *   const src = pack.get('ProtonAntimatter.fs');
 *
 * Alternatively, use the module-level singleton:
 *   import { shaderPack } from './shader-loader.js';
 *   await shaderPack.load('/assets/shaders/compiled.vs');
 *   const src = shaderPack.get('curl.glsl');
 */

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ShaderPackOptions {
  /**
   * Log debug info (parsed shader names, require resolutions).
   * Defaults to false.
   */
  debug?: boolean;
}

// ── ShaderPack ────────────────────────────────────────────────────────────────

export class ShaderPack {
  /** Raw (unresolved) sources keyed by shader name */
  private _raw: Map<string, string> = new Map();
  /** Fully-resolved sources cache keyed by shader name */
  private _resolved: Map<string, string> = new Map();
  /** Whether load() has been called successfully */
  private _loaded = false;
  private _debug: boolean;

  constructor(options: ShaderPackOptions = {}) {
    this._debug = options.debug ?? false;
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Fetch and parse an AT compiled.vs shader pack from `url`.
   * Safe to call multiple times — subsequent calls replace the current pack.
   */
  async load(url: string): Promise<void> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `[ShaderPack] Failed to fetch "${url}": ${response.status} ${response.statusText}`
      );
    }
    const text = await response.text();
    this._parse(text);
    this._loaded = true;
    if (this._debug) {
      console.debug(
        `[ShaderPack] Loaded ${this._raw.size} shaders from "${url}"`
      );
    }
  }

  /**
   * Seed the pack directly from a raw string (e.g. `?raw` Vite import).
   * Useful in unit tests or when the file is bundled at build time.
   */
  loadFromString(raw: string): void {
    this._parse(raw);
    this._loaded = true;
  }

  /**
   * Return the names of all shaders in the pack.
   */
  names(): string[] {
    this._assertLoaded();
    return [...this._raw.keys()];
  }

  /**
   * Resolve `name` — recursively expand all `#require(...)` directives —
   * and return the fully-inlined GLSL source.
   *
   * The result is memoised: calling resolve() a second time for the same
   * name is O(1).
   */
  resolve(name: string): string {
    this._assertLoaded();
    return this._resolve(name, new Set<string>());
  }

  /**
   * Return the fully-resolved shader for `name` (same as resolve() but
   * throws a friendlier error when the shader is absent).
   */
  get(name: string): string {
    this._assertLoaded();
    if (!this._raw.has(name)) {
      throw new Error(
        `[ShaderPack] Shader "${name}" not found. ` +
        `Available: ${[...this._raw.keys()].join(', ')}`
      );
    }
    return this._resolve(name, new Set<string>());
  }

  /**
   * Check whether the pack contains a shader with the given name.
   */
  has(name: string): boolean {
    return this._raw.has(name);
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  /**
   * Parse the AT `{@}name{@}code{@}name2{@}code2...` format.
   *
   * Parts layout after split('{@}'):
   *   index 0      — preamble (empty or ignored)
   *   index 1      — first shader name
   *   index 2      — first shader code
   *   index 3      — second shader name
   *   ...
   */
  private _parse(text: string): void {
    this._raw.clear();
    this._resolved.clear();

    const parts = text.split('{@}');
    // Odd indices = names, even indices (starting at 2) = code
    for (let i = 1; i + 1 < parts.length; i += 2) {
      const name = parts[i].trim();
      const code = parts[i + 1]; // preserve whitespace in GLSL
      if (name) {
        this._raw.set(name, code);
      }
    }

    if (this._debug) {
      console.debug(
        '[ShaderPack] Parsed shaders:',
        [...this._raw.keys()]
      );
    }
  }

  /**
   * Recursive resolver with cycle detection.
   * `stack` tracks the names currently being resolved (the call chain).
   */
  private _resolve(name: string, stack: Set<string>): string {
    // Return memoised result immediately
    const cached = this._resolved.get(name);
    if (cached !== undefined) return cached;

    const raw = this._raw.get(name);
    if (raw === undefined) {
      // Unknown dependency — return a GLSL comment so compilation gives a
      // meaningful error rather than a silent include gap.
      const msg = `/* [ShaderPack] #require "${name}" not found in pack */`;
      console.warn(`[ShaderPack] #require "${name}" not found in pack`);
      return msg;
    }

    // Cycle guard
    if (stack.has(name)) {
      const cycle = [...stack, name].join(' → ');
      console.warn(`[ShaderPack] Circular #require detected: ${cycle}`);
      return `/* [ShaderPack] Circular require skipped: ${name} */`;
    }

    stack = new Set(stack); // copy so siblings don't share state
    stack.add(name);

    // Resolve each line, expanding #require(dep.glsl) directives
    const resolved = raw
      .split('\n')
      .map((line) => {
        const match = line.match(/^[ \t]*#require\(\s*([^\s)]+)\s*\)/);
        if (!match) return line;
        const dep = match[1];
        if (this._debug) {
          console.debug(`[ShaderPack] ${name} requires ${dep}`);
        }
        return this._resolve(dep, stack);
      })
      .join('\n');

    this._resolved.set(name, resolved);
    return resolved;
  }

  private _assertLoaded(): void {
    if (!this._loaded) {
      throw new Error(
        '[ShaderPack] Pack not loaded. Call load(url) or loadFromString() first.'
      );
    }
  }
}

// ── Module-level singleton ────────────────────────────────────────────────────

/**
 * Shared ShaderPack instance.
 *
 * Initialise once at app startup:
 *   await shaderPack.load('/assets/shaders/compiled.vs');
 *
 * Then access anywhere:
 *   const src = shaderPack.get('curl.glsl');
 */
export const shaderPack = new ShaderPack();
