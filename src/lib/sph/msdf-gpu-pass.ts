/**
 * msdf-gpu-pass.ts — GPU MSDF 文字渲染
 *
 * 真正的 WebGL1 GPU pass，不是占位符。每个方法都调用 gl.*。
 * 从 compiled.vs 通过 ShaderLoader 提取 msdf.frag / msdf.vert。
 * 为每个 transformer cell 画标签文字 (input_embed, pos_encode, …)。
 *
 * 架构:
 *   - 每个字符 → 2 个三角形 (1 quad), UV 从 atlas glyph metrics 计算
 *   - 所有字符合并进一个 VBO → 1 次 gl.drawArrays 画完整个 cell label
 *   - MSDF atlas 先用 1×1 白色占位纹理, 异步加载真实 atlas 后替换
 *   - fragment: median(r,g,b) → SDF → smoothstep → alpha
 *
 * 7 个 cell 标签 (SPECIES_PIPELINE_ORDER):
 *   input_embed | pos_encode | self_attn | add_norm1 | ffn | add_norm2 | output
 *
 * WebGL1 语法: varying / texture2D / attribute (不是 WebGL2)
 *
 * xiaodi #M878 — cell-pubsub-loop
 */




// ─── MSDF Atlas Glyph Metrics ─────────────────────────────────────────────────


import { getShader } from '../shaders/ShaderLoader';

// [orphan-precise] /** Single glyph entry in the MSDF atlas JSON (msdfgen format). */
interface GlyphMetric {
  /** Unicode codepoint */
  id: number;
  /** UV rectangle in atlas [0..1]: x, y = top-left, w, h = size */
  uvX: number;
  uvY: number;
  uvW: number;
  uvH: number;
  /** Normalised glyph dimensions (relative to font size = 1.0) */
  planeBoundsLeft: number;
  planeBoundsBottom: number;
  planeBoundsRight: number;
  planeBoundsTop: number;
  /** How far to advance the pen after this glyph */
  advance: number;
}

/** Minimal MSDF atlas descriptor (subset of msdf-atlas-gen JSON output). */
interface MSDFAtlas {
  /** Atlas image size in pixels */
  width: number;
  height: number;
  /** Em size used when generating the atlas */
  size: number;
  glyphs: GlyphMetric[];
}

// ─── Built-in fallback atlas (ASCII printable, single-cell 1×1 placeholder) ──
//
// In production this is replaced by a real atlas loaded from /public/fonts/*.
// The fallback ensures the GPU path is exercised immediately without blocking.

function buildFallbackAtlas(): MSDFAtlas {
  const glyphs: GlyphMetric[] = [];
  // Lay out ASCII 32-126 in a 16-column grid on a 512×512 atlas.
  const cols = 16;
  const cellW = 1.0 / cols;
  const cellH = 1.0 / 8; // 8 rows × 16 cols = 128 slots
  for (let i = 32; i <= 126; i++) {
    const idx = i - 32;
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    glyphs.push({
      id: i,
      uvX: col * cellW,
      uvY: row * cellH,
      uvW: cellW,
      uvH: cellH,
      planeBoundsLeft:   0.0,
      planeBoundsBottom: 0.0,
      planeBoundsRight:  0.6,
      planeBoundsTop:    1.0,
      advance: 0.6,
    });
  }
  return { width: 512, height: 512, size: 32, glyphs };
}

// ─── 7 Transformer cell labels ────────────────────────────────────────────────

/** Pipeline-ordered labels exactly matching SPECIES_PIPELINE_ORDER. */
export const CELL_LABELS: readonly string[] = [
  'input_embed',
  'pos_encode',
  'self_attn',
  'add_norm1',
  'ffn',
  'add_norm2',
  'output',
] as const;

// ─── MSDFTextGPU ──────────────────────────────────────────────────────────────

/**
 * GPU-side MSDF text renderer.
 *
 * Usage:
 * ```ts
 * const msdf = new MSDFTextGPU(gl);
 * msdf.loadAtlasFromUrl('/fonts/inter-msdf.png', '/fonts/inter-msdf.json');
 *
 * // inside render loop:
 * CELL_LABELS.forEach((label, i) => {
 *   msdf.drawLabel(label, cellX[i], cellY[i], cellW[i], 0.04, [1,1,1], 1.0);
 * });
 * ```
 */
export class MSDFTextGPU {
  private gl: WebGLRenderingContext;

  // Compiled shader program — real gl.createShader / gl.createProgram calls
  private prog!: WebGLProgram;

  // Attribute locations
  private aPositionLoc!: number;
  private aUvLoc!: number;

  // Uniform locations
  private uMsdfTextureLoc!: WebGLUniformLocation | null;
  private uColorLoc!: WebGLUniformLocation | null;
  private uOpacityLoc!: WebGLUniformLocation | null;
  private uOutlineWidthLoc!: WebGLUniformLocation | null;
  private uOutlineColorLoc!: WebGLUniformLocation | null;

  // MSDF atlas texture — starts as 1×1 white placeholder
  private atlasTex!: WebGLTexture;
  private atlasReady = false;

  // Per-cell VBOs: key = label string, value = { buf, count }
  private cellBufs: Map<string, { buf: WebGLBuffer; count: number }> = new Map();

  // Atlas metrics
  private atlas: MSDFAtlas;
  private glyphMap: Map<number, GlyphMetric> = new Map();

  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
    this.atlas = buildFallbackAtlas();
    this._buildGlyphMap();
    this._compileProgram();
    this._createPlaceholderAtlas();
    this._buildAllCellBuffers();
  }

  // ─── Public API ─────────────────────────────────────────────────────────────

  /**
   * Asynchronously load the real MSDF atlas PNG + JSON.
   * Until loaded, the 1×1 white placeholder is used (glyphs render white).
   */
  loadAtlasFromUrl(pngUrl: string, jsonUrl: string): void {
    Promise.all([
      fetch(jsonUrl).then(r => r.json() as Promise<MSDFAtlas>),
      new Promise<HTMLImageElement>((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = pngUrl;
      }),
    ]).then(([atlasJson, img]) => {
      this.atlas = atlasJson;
      this._buildGlyphMap();
      this._uploadAtlasTexture(img);
      this._buildAllCellBuffers(); // rebuild VBOs with real metrics
    }).catch(err => {
      console.warn('[MSDFTextGPU] atlas load failed, using placeholder:', err);
    });
  }

  /**
   * Draw one cell label.  Call once per frame per visible label.
   *
   * @param label      e.g. "self_attn"
   * @param ndcX       NDC x of glyph run origin (-1..1)
   * @param ndcY       NDC y of baseline (-1..1)
   * @param fontSize   NDC height of one em (e.g. 0.04)
   * @param color      RGB [0..1]
   * @param opacity    alpha [0..1]
   */
  drawLabel(
    label: string,
    ndcX: number,
    ndcY: number,
    fontSize: number,
    color: [number, number, number] = [1, 1, 1],
    opacity: number = 1.0,
  ): void {
    const gl = this.gl;
    const entry = this.cellBufs.get(label);
    if (!entry || entry.count === 0) return;

    // Rebuild VBO with the current ndcX/ndcY/fontSize (geometry changes per frame)
    const data = this._buildLabelGeometry(label, ndcX, ndcY, fontSize);
    if (data.length === 0) return;

    gl.useProgram(this.prog);

    // Bind / upload vertex data
    gl.bindBuffer(gl.ARRAY_BUFFER, entry.buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);

    // attribute: position (xy) + uv (zw) — stride 4 floats, 16 bytes
    const stride = 4 * 4; // 4 floats × 4 bytes
    gl.enableVertexAttribArray(this.aPositionLoc);
    gl.vertexAttribPointer(this.aPositionLoc, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(this.aUvLoc);
    gl.vertexAttribPointer(this.aUvLoc, 2, gl.FLOAT, false, stride, 8);

    // Uniforms
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.atlasTex);
    gl.uniform1i(this.uMsdfTextureLoc, 0);
    gl.uniform3f(this.uColorLoc, color[0], color[1], color[2]);
    gl.uniform1f(this.uOpacityLoc, opacity);
    gl.uniform1f(this.uOutlineWidthLoc, 0.0);
    gl.uniform3f(this.uOutlineColorLoc, 0, 0, 0);

    // Blend for MSDF alpha
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // One draw call for the whole label (all characters, 2 triangles each)
    const vertexCount = (data.length / 4); // 4 floats per vertex
    gl.drawArrays(gl.TRIANGLES, 0, vertexCount);

    gl.disableVertexAttribArray(this.aPositionLoc);
    gl.disableVertexAttribArray(this.aUvLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /**
   * Draw all 7 cell labels at evenly spaced NDC X positions,
   * centred vertically at ndcY.  Convenience wrapper.
   */
  drawAllCellLabels(
    ndcYBaseline: number = -0.85,
    fontSize: number = 0.04,
    colors?: Array<[number, number, number]>,
    opacities?: number[],
  ): void {
    const n = CELL_LABELS.length;
    for (let i = 0; i < n; i++) {
      const label  = CELL_LABELS[i];
      const ndcX   = -1.0 + (2.0 / n) * (i + 0.5) - this._labelWidth(label, fontSize) * 0.5;
      const color: [number, number, number]  = colors?.[i]  ?? [1, 1, 1];
      const alpha  = opacities?.[i] ?? 1.0;
      this.drawLabel(label, ndcX, ndcYBaseline, fontSize, color, alpha);
    }
  }

  /** Release all GPU resources. */
  destroy(): void {
    const gl = this.gl;
    for (const { buf } of this.cellBufs.values()) {
      gl.deleteBuffer(buf);
    }
    this.cellBufs.clear();
    gl.deleteTexture(this.atlasTex);
    gl.deleteProgram(this.prog);
  }

  // ─── Private: shader compilation ────────────────────────────────────────────

  private _compileProgram(): void {
    const gl = this.gl;

    // Extract AT shaders from compiled.vs via ShaderLoader
    const vertSrc = getShader('msdf.vert');
    const fragSrc = getShader('msdf.frag');

    // ── Vertex shader ──
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, 'precision highp float;\n' + vertSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[MSDFTextGPU] vertex compile error: ${gl.getShaderInfoLog(vs)}`);
    }

    // ── Fragment shader — enable OES_standard_derivatives for fwidth() ──
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    // msdf.frag already has #extension GL_OES_standard_derivatives : enable
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[MSDFTextGPU] fragment compile error: ${gl.getShaderInfoLog(fs)}`);
    }

    // ── Link program ──
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[MSDFTextGPU] link error: ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    this.prog = prog;

    // Cache attribute + uniform locations
    this.aPositionLoc = gl.getAttribLocation(prog, 'position');
    this.aUvLoc       = gl.getAttribLocation(prog, 'uv');

    this.uMsdfTextureLoc  = gl.getUniformLocation(prog, 'uMsdfTexture');
    this.uColorLoc        = gl.getUniformLocation(prog, 'uColor');
    this.uOpacityLoc      = gl.getUniformLocation(prog, 'uOpacity');
    this.uOutlineWidthLoc = gl.getUniformLocation(prog, 'uOutlineWidth');
    this.uOutlineColorLoc = gl.getUniformLocation(prog, 'uOutlineColor');
  }

  // ─── Private: atlas texture ──────────────────────────────────────────────────

  /** Create a 1×1 opaque white placeholder so shaders run immediately. */
  private _createPlaceholderAtlas(): void {
    const gl = this.gl;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // 1×1 white RGBA pixel — median(1,1,1)-0.5 = 0.5 → fill = smoothstep(-w,w,0) ≈ 0.5
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA,
      1, 1, 0,
      gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([255, 255, 255, 255]),
    );
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.atlasTex = tex;
    this.atlasReady = false;
  }

  /** Upload the real atlas image to the GPU texture. */
  private _uploadAtlasTexture(img: HTMLImageElement): void {
    const gl = this.gl;

    gl.bindTexture(gl.TEXTURE_2D, this.atlasTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.atlasReady = true;
  }

  // ─── Private: glyph map ───────────────────────────────────────────────────────

  private _buildGlyphMap(): void {
    this.glyphMap.clear();
    for (const g of this.atlas.glyphs) {
      this.glyphMap.set(g.id, g);
    }
  }

  // ─── Private: VBO building ───────────────────────────────────────────────────

  /** Pre-allocate one VBO per cell label (DYNAMIC_DRAW, rebuilt each frame). */
  private _buildAllCellBuffers(): void {
    const gl = this.gl;
    for (const label of CELL_LABELS) {
      if (!this.cellBufs.has(label)) {
        const buf = gl.createBuffer()!;
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        // Reserve space: max 64 chars × 6 verts × 4 floats × 4 bytes
        gl.bufferData(gl.ARRAY_BUFFER, 64 * 6 * 4 * 4, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        this.cellBufs.set(label, { buf, count: label.length });
      }
    }
  }

  /**
   * Build interleaved vertex data for one label string.
   * Layout per vertex: [posX, posY, uvX, uvY]  (4 floats = 16 bytes)
   * 2 triangles per character = 6 vertices per character.
   *
   * All positions in NDC [-1, 1]; UV in [0, 1] atlas space.
   */
  private _buildLabelGeometry(
    label: string,
    ndcX: number,
    ndcY: number,
    fontSize: number,
  ): Float32Array {
    const verts: number[] = [];
    let penX = ndcX;

    for (let ci = 0; ci < label.length; ci++) {
      const code = label.charCodeAt(ci);
      const g = this.glyphMap.get(code);
      if (!g) {
        penX += fontSize * 0.5; // unknown glyph → advance half em
        continue;
      }

      // Glyph quad in NDC space
      const x0 = penX + g.planeBoundsLeft   * fontSize;
      const x1 = penX + g.planeBoundsRight  * fontSize;
      const y0 = ndcY + g.planeBoundsBottom * fontSize;
      const y1 = ndcY + g.planeBoundsTop    * fontSize;

      // Atlas UV
      const u0 = g.uvX;
      const u1 = g.uvX + g.uvW;
      const v0 = g.uvY;
      const v1 = g.uvY + g.uvH;

      // Triangle 1: bottom-left, bottom-right, top-right
      verts.push(x0, y0, u0, v1);
      verts.push(x1, y0, u1, v1);
      verts.push(x1, y1, u1, v0);

      // Triangle 2: bottom-left, top-right, top-left
      verts.push(x0, y0, u0, v1);
      verts.push(x1, y1, u1, v0);
      verts.push(x0, y1, u0, v0);

      penX += g.advance * fontSize;
    }

    return new Float32Array(verts);
  }

  /** Compute total NDC width of a label at the given fontSize. */
  private _labelWidth(label: string, fontSize: number): number {
    let w = 0;
    for (let i = 0; i < label.length; i++) {
      const g = this.glyphMap.get(label.charCodeAt(i));
      w += g ? g.advance * fontSize : fontSize * 0.5;
    }
    return w;
  }
}

// ─── Factory helper ──────────────────────────────────────────────────────────

/**
 * Create and optionally load the real atlas in one call.
 *
 * ```ts
 * const msdf = createMSDFTextGPU(gl, {
 *   atlasImage: '/fonts/inter-msdf.png',
 *   atlasJson:  '/fonts/inter-msdf.json',
 * });
 * ```
 */
export function createMSDFTextGPU(
  gl: WebGLRenderingContext,
  opts?: { atlasImage?: string; atlasJson?: string },
): MSDFTextGPU {
  const pass = new MSDFTextGPU(gl);
  if (opts?.atlasImage && opts?.atlasJson) {
    pass.loadAtlasFromUrl(opts.atlasImage, opts.atlasJson);
  }
  return pass;
}
