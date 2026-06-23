// ── at-shader-loader.ts ─────────────────────────────────────────────
// Loads Active Theory's compiled.vs bundle and resolves #require() deps.
//
// compiled.vs format (two interleaved section types):
//
//   1. Library shaders — plain GLSL chunks (no #!ATTRIBUTES header):
//      {@}blendmodes.glsl{@}<raw glsl code>{@}nextName{@}...
//
//   2. Class shaders — material definitions with structured sections:
//      {@}ColorMaterial.glsl{@}
//        #!ATTRIBUTES
//        ...
//        #!UNIFORMS
//        uniform vec3 color;
//        ...
//        #!VARYINGS
//        varying vec3 vColor;
//        ...
//        #!SHADER: ColorMaterial.vs
//        void main() { ... }
//        #!SHADER: ColorMaterial.fs
//        void main() { ... }
//      {@}NextBlock.glsl{@}...
//
//   Inside class shaders the #!SHADER: tag names can be:
//     - Fully qualified: "ColorMaterial.vs", "lighting.fs"
//     - Generic:         "Vertex", "Fragment", "Vertex.vs", "Fragment.fs"
//   Generic names get prefixed with the parent block name for uniqueness.
//
//   #require(name) anywhere in shader code means: inline the content of
//   the shader called `name` at that position. Resolved recursively with
//   cycle detection.
// ─────────────────────────────────────────────────────────────────────

export class ATShaderLoader {
  /** All parsed shaders keyed by name */
  private shaders: Map<string, string> = new Map();

  /**
   * For class shaders we also store the structured pieces so callers
   * can grab attributes / uniforms / varyings separately if needed.
   */
  private classBlocks: Map<
    string,
    {
      attributes: string;
      uniforms: string;
      varyings: string;
      subShaders: Map<string, string>; // name → raw code (pre-require)
    }
  > = new Map();

  /** Cache for fully-resolved shader text (after #require inlining) */
  private resolvedCache: Map<string, string> = new Map();

  // ── public API ────────────────────────────────────────────────────

  /**
   * Read compiled.vs from `compiledVsPath` and parse every shader into
   * the internal map.  Can be called multiple times to layer files.
   */
  async load(compiledVsPath: string): Promise<void> {
    // Support both Node (fs) and fetch-based environments
    const raw = await this.readFile(compiledVsPath);
    this.parse(raw);
  }

  /**
   * Return a single shader by exact name, with all #require() deps
   * recursively inlined.
   */
  getShader(name: string): string {
    if (this.resolvedCache.has(name)) {
      return this.resolvedCache.get(name)!;
    }

    const raw = this.shaders.get(name);
    if (raw === undefined) {
      throw new Error(`[ATShaderLoader] shader not found: "${name}"`);
    }

    const resolved = this.resolveRequires(raw, new Set<string>());
    this.resolvedCache.set(name, resolved);
    return resolved;
  }

  /**
   * Return the vertex + fragment pair for a class shader.
   *
   * Resolution order for a class block named "Foo.glsl":
   *   vertex  → "Foo.vs" | "Vertex.vs@Foo" | "Vertex@Foo"
   *   fragment→ "Foo.fs" | "Fragment.fs@Foo"| "Fragment@Foo"
   *
   * Each is returned with its class-level #!UNIFORMS and #!VARYINGS
   * prepended and all #require() deps inlined.
   */
  getProgram(name: string): { vertex: string; fragment: string } {
    // Normalise: accept "Foo", "Foo.glsl"
    const blockName = name.endsWith('.glsl') ? name : `${name}.glsl`;
    const baseName = blockName.replace(/\.glsl$/, '');

    const block = this.classBlocks.get(blockName);
    if (!block) {
      // Fallback: try finding explicit .vs / .fs in the flat map
      const vs = this.tryGetShader(`${baseName}.vs`);
      const fs = this.tryGetShader(`${baseName}.fs`);
      if (vs !== undefined && fs !== undefined) {
        return { vertex: this.getShader(`${baseName}.vs`), fragment: this.getShader(`${baseName}.fs`) };
      }
      throw new Error(`[ATShaderLoader] class block not found: "${blockName}"`);
    }

    // Build the preamble (attributes + uniforms + varyings)
    const preamble = [block.attributes, block.uniforms, block.varyings]
      .filter((s) => s.trim().length > 0)
      .join('\n');

    const vertexName = this.findSubShader(block, baseName, 'vs');
    const fragmentName = this.findSubShader(block, baseName, 'fs');

    const buildFull = (subName: string): string => {
      const rawBody = this.shaders.get(subName);
      if (rawBody === undefined) {
        throw new Error(`[ATShaderLoader] sub-shader "${subName}" not found in block "${blockName}"`);
      }
      const resolved = this.resolveRequires(rawBody, new Set<string>());
      return preamble ? `${preamble}\n${resolved}` : resolved;
    };

    return {
      vertex: buildFull(vertexName),
      fragment: buildFull(fragmentName),
    };
  }

  /** List every shader name in the store. */
  listShaders(): string[] {
    return Array.from(this.shaders.keys()).sort();
  }

  /**
   * Convenience: return the fully-resolved lighting.fs shader.
   * This is AT's core PBR lighting chunk — nearly every material
   * #require()s it.
   */
  getLighting(): string {
    return this.getShader('lighting.fs');
  }

  /**
   * Convenience: return the fully-resolved ShadowDepth.fs shader
   * (AT's shadow-mapping system).
   */
  getShadowDepth(): string {
    return this.getShader('ShadowDepth.fs');
  }

  // ── internals ─────────────────────────────────────────────────────

  /** Read a file as UTF-8 text. */
  private async readFile(path: string): Promise<string> {
    // Node.js
    if (typeof globalThis.process !== 'undefined' && globalThis.process.versions?.node) {
      const fs = await import('fs');
      return fs.readFileSync(path, 'utf-8');
    }
    // Browser / Deno fetch
    const resp = await fetch(path);
    if (!resp.ok) throw new Error(`[ATShaderLoader] fetch failed: ${resp.status} ${path}`);
    return resp.text();
  }

  /**
   * Parse the raw compiled.vs text into `this.shaders` and
   * `this.classBlocks`.
   */
  private parse(raw: string): void {
    // Split on the {@} delimiter.  Result alternates:
    //   ['', name1, code1, name2, code2, ...]
    // The first element is always empty (file starts with {@}).
    const parts = raw.split('{@}');

    for (let i = 1; i + 1 < parts.length; i += 2) {
      const blockName = parts[i].trim();
      const blockCode = parts[i + 1];

      if (this.isClassShader(blockCode)) {
        this.parseClassBlock(blockName, blockCode);
      } else {
        // Library / utility shader — store as-is under its name
        this.shaders.set(blockName, blockCode);
      }
    }
  }

  /** A class shader contains #!ATTRIBUTES (possibly empty). */
  private isClassShader(code: string): boolean {
    return code.includes('#!ATTRIBUTES');
  }

  /**
   * Parse a class block like:
   *   #!ATTRIBUTES
   *   ...
   *   #!UNIFORMS
   *   ...
   *   #!VARYINGS
   *   ...
   *   #!SHADER: Name.vs
   *   ...code...
   *   #!SHADER: Name.fs
   *   ...code...
   */
  private parseClassBlock(blockName: string, code: string): void {
    const baseName = blockName.replace(/\.glsl$/, '');

    // ── extract structured sections ──────────────────────────────
    const attributes = this.extractSection(code, '#!ATTRIBUTES', ['#!UNIFORMS', '#!VARYINGS', '#!SHADER:']);
    const uniforms = this.extractSection(code, '#!UNIFORMS', ['#!VARYINGS', '#!SHADER:']);
    const varyings = this.extractSection(code, '#!VARYINGS', ['#!SHADER:']);

    // ── extract each #!SHADER: sub-block ─────────────────────────
    const subShaders = new Map<string, string>();
    const shaderRegex = /#!SHADER:\s*(\S+)/g;
    const matches: { name: string; start: number }[] = [];
    let m: RegExpExecArray | null;

    while ((m = shaderRegex.exec(code)) !== null) {
      matches.push({ name: m[1], start: m.index + m[0].length });
    }

    for (let j = 0; j < matches.length; j++) {
      const start = matches[j].start;
      const end = j + 1 < matches.length ? matches[j + 1].start - matches[j + 1].name.length - '#!SHADER: '.length : code.length;

      // Extract the code between this #!SHADER marker and the next
      const subCode = code.slice(start, end).replace(/#!SHADER:\s*\S+/g, '').trim();
      let subName = matches[j].name;

      // Normalise generic names → unique keys
      // "Vertex" / "Vertex.vs" → "baseName.vs"
      // "Fragment" / "Fragment.fs" → "baseName.fs"
      const canonical = this.canonicalSubName(subName, baseName);

      subShaders.set(canonical, subCode);
      this.shaders.set(canonical, subCode);
    }

    this.classBlocks.set(blockName, { attributes, uniforms, varyings, subShaders });

    // Also store the full raw block under the .glsl name so
    // #require(Lighting.glsl) can find it if someone refers to
    // the entire block.
    const fullBlock = [attributes, uniforms, varyings, ...Array.from(subShaders.values())]
      .filter((s) => s.trim().length > 0)
      .join('\n');
    this.shaders.set(blockName, fullBlock);
  }

  /**
   * Map sub-shader names to canonical form.
   *
   *  "Vertex"       → "BaseName.vs"
   *  "Vertex.vs"    → "BaseName.vs"
   *  "Fragment"      → "BaseName.fs"
   *  "Fragment.fs"   → "BaseName.fs"
   *  "lighting.vs"   → "lighting.vs"  (already qualified)
   *  "Foo.vs"        → "Foo.vs"       (already qualified, not generic)
   */
  private canonicalSubName(rawName: string, baseName: string): string {
    const lower = rawName.toLowerCase();

    if (lower === 'vertex' || lower === 'vertex.vs') {
      return `${baseName}.vs`;
    }
    if (lower === 'fragment' || lower === 'fragment.fs') {
      return `${baseName}.fs`;
    }
    // Already qualified — return as-is
    return rawName;
  }

  /**
   * Extract text between `startMarker` and the first occurrence of
   * any marker in `endMarkers`.
   */
  private extractSection(code: string, startMarker: string, endMarkers: string[]): string {
    const startIdx = code.indexOf(startMarker);
    if (startIdx === -1) return '';

    const contentStart = startIdx + startMarker.length;

    let endIdx = code.length;
    for (const em of endMarkers) {
      const idx = code.indexOf(em, contentStart);
      if (idx !== -1 && idx < endIdx) {
        endIdx = idx;
      }
    }

    return code.slice(contentStart, endIdx).trim();
  }

  /**
   * Find the vertex or fragment sub-shader name within a class block.
   * `type` is 'vs' or 'fs'.
   */
  private findSubShader(
    block: { subShaders: Map<string, string> },
    baseName: string,
    type: 'vs' | 'fs',
  ): string {
    const primary = `${baseName}.${type}`;
    if (block.subShaders.has(primary)) return primary;

    // Shouldn't happen after canonicalisation, but guard anyway
    for (const key of block.subShaders.keys()) {
      if (key.endsWith(`.${type}`)) return key;
    }

    const kindLabel = type === 'vs' ? 'vertex' : 'fragment';
    throw new Error(`[ATShaderLoader] no ${kindLabel} shader in class block "${baseName}.glsl"`);
  }

  /** Recursively resolve all #require(name) directives. */
  private resolveRequires(code: string, visited: Set<string>): string {
    return code.replace(/#require\(([^)]+)\)/g, (_match, depName: string) => {
      const trimmed = depName.trim();

      if (visited.has(trimmed)) {
        // Cycle detected — emit a comment instead of infinite recursion
        return `/* [ATShaderLoader] circular #require("${trimmed}") skipped */`;
      }

      const depCode = this.shaders.get(trimmed);
      if (depCode === undefined) {
        return `/* [ATShaderLoader] WARNING: unresolved #require("${trimmed}") */`;
      }

      // Check the resolved cache first (dep may have been resolved in
      // another path already)
      if (this.resolvedCache.has(trimmed)) {
        return this.resolvedCache.get(trimmed)!;
      }

      const next = new Set(visited);
      next.add(trimmed);
      const resolved = this.resolveRequires(depCode, next);

      // Cache the fully-resolved dep
      this.resolvedCache.set(trimmed, resolved);
      return resolved;
    });
  }

  /** Safely try to get a shader, returning undefined if not found. */
  private tryGetShader(name: string): string | undefined {
    return this.shaders.get(name);
  }
}
