/**
 * constraint-texture.ts — GPGPU RenderTexture for cell constraint states
 *
 * Active Theory Neon 风格：每个像素 = 一个 cell 的物理约束状态。
 * 纹理布局: sqrt(N) × sqrt(N)，RGBA 编码：
 *   R = dx          (归一化位移 x, [-1,1] → [0,1])
 *   G = dy          (归一化位移 y, [-1,1] → [0,1])
 *   B = force_mag   (力的大小，归一化至 [0,1])
 *   A = converged   (0.0 = 未收敛, 1.0 = 已收敛)
 *
 * 数据来源: channels/physics/force_field.json, cell_registry.json
 *
 * 架构参考:
 *   src/lib/renderers/fluid-fbo.ts   (ping-pong FBO 模式)
 *   src/lib/renderers/antimatter-compute.ts  (双缓冲计算)
 */

// ── Constants ─────────────────────────────────────────────────────────────────

/** 力归一化上限 (px/frame)，超出则 clamp */
const FORCE_MAX = 200.0;

/** 位移归一化范围 (px) */
const DISP_MAX = 500.0;

// ── Types ────────────────────────────────────────────────────────────────────

export interface CellConstraintState {
  cellId: string;
  dx: number;
  dy: number;
  forceMag: number;
  converged: boolean;
}

export interface ConstraintTextureConfig {
  gl: WebGL2RenderingContext;
  /** channels/physics/force_field.json の parsed data */
  forceField: Record<string, { dx: number; dy: number; dz: number }>;
  /** channels/physics/cell_registry.json の parsed data */
  cellRegistry: Record<string, { bbox: { min: number[]; max: number[] }; species: string; z: number; converged?: boolean }>;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** encode [-DISP_MAX, DISP_MAX] → [0, 1] for R/G channel */
function encodeDisp(v: number): number {
  return Math.max(0, Math.min(1, (v + DISP_MAX) / (2 * DISP_MAX)));
}

/** decode [0, 1] → [-DISP_MAX, DISP_MAX] */
function decodeDisp(v: number): number {
  return v * 2 * DISP_MAX - DISP_MAX;
}

/** encode [0, FORCE_MAX] → [0, 1] for B channel */
function encodeForce(v: number): number {
  return Math.max(0, Math.min(1, v / FORCE_MAX));
}

/** next power-of-two >= n, minimum 1 */
function nextPOT(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/** ceil(sqrt(n)) — smallest integer k where k*k >= n */
function ceilSqrt(n: number): number {
  return Math.ceil(Math.sqrt(n));
}

// ── ConstraintTexture ─────────────────────────────────────────────────────────

export class ConstraintTexture {
  readonly gl: WebGL2RenderingContext;

  /** Ordered cell IDs — index determines pixel position in texture */
  readonly cellOrder: string[];

  /** Texture side length (square) */
  readonly texSize: number;

  /** Ping-pong: [0] = current read, [1] = current write */
  private fbos: [WebGLFramebuffer, WebGLFramebuffer];
  private textures: [WebGLTexture, WebGLTexture];
  private _pingPong = 0;

  /** Pixel data cache for CPU readback (lazy allocated) */
  private _readPixelsBuf: Uint8Array | null = null;

  constructor(config: ConstraintTextureConfig) {
    const { gl, forceField, cellRegistry } = config;
    this.gl = gl;

    // ── Build ordered cell list ──────────────────────────────────────────────
    this.cellOrder = Object.keys(cellRegistry);
    const cellCount = this.cellOrder.length;

    // Square texture: side = ceil(sqrt(N)), padded to next POT for safety
    const side = ceilSqrt(cellCount);
    this.texSize = nextPOT(side);

    // ── Build initial pixel data from force_field.json ───────────────────────
    const pixelCount = this.texSize * this.texSize;
    const data = new Uint8Array(pixelCount * 4); // RGBA u8

    for (let i = 0; i < this.cellOrder.length; i++) {
      const cellId = this.cellOrder[i];
      const ff = forceField[cellId];
      const reg = cellRegistry[cellId];

      const dx   = ff?.dx   ?? 0;
      const dy   = ff?.dy   ?? 0;
      const mag  = ff ? Math.hypot(ff.dx, ff.dy) : 0;
      const conv = reg?.converged ?? false;

      const base = i * 4;
      data[base + 0] = Math.round(encodeDisp(dx)    * 255); // R
      data[base + 1] = Math.round(encodeDisp(dy)    * 255); // G
      data[base + 2] = Math.round(encodeForce(mag)  * 255); // B
      data[base + 3] = conv ? 255 : 0;                       // A
    }

    // ── Allocate ping-pong textures + FBOs ───────────────────────────────────
    const [texA, fboA] = this._allocTex(data);
    const [texB, fboB] = this._allocTex(null);

    this.textures = [texA, texB];
    this.fbos     = [fboA, fboB];
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  /** The texture that the constraint shader should READ from (sampler2D uState) */
  get readTexture(): WebGLTexture {
    return this.textures[this._pingPong];
  }

  /** The FBO that the constraint shader should RENDER INTO */
  get writeFBO(): WebGLFramebuffer {
    return this.fbos[1 - this._pingPong];
  }

  /**
   * Swap ping-pong after each compute pass.
   * Call once per frame, after the constraint shader draw call.
   */
  swap(): void {
    this._pingPong = 1 - this._pingPong;
  }

  /**
   * Read pixel at cell index back to CPU.
   * Expensive — only call when the CPU side truly needs the data.
   * Normal path: leave results on GPU and pass readTexture to cell shader.
   */
  readCellState(cellIndex: number): CellConstraintState {
    const { gl, texSize } = this;
    if (!this._readPixelsBuf) {
      this._readPixelsBuf = new Uint8Array(texSize * texSize * 4);
    }

    // Bind the current READ fbo (which is the last WRITTEN result)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos[1 - this._pingPong]);
    gl.readPixels(0, 0, texSize, texSize, gl.RGBA, gl.UNSIGNED_BYTE, this._readPixelsBuf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    const base = cellIndex * 4;
    const r = this._readPixelsBuf[base + 0] / 255;
    const g = this._readPixelsBuf[base + 1] / 255;
    const b = this._readPixelsBuf[base + 2] / 255;
    const a = this._readPixelsBuf[base + 3] / 255;

    return {
      cellId:    this.cellOrder[cellIndex],
      dx:        decodeDisp(r),
      dy:        decodeDisp(g),
      forceMag:  b * FORCE_MAX,
      converged: a > 0.5,
    };
  }

  /**
   * Batch readback — returns all cell states.
   * One gl.readPixels call for the whole texture.
   */
  readAllCellStates(): CellConstraintState[] {
    const { gl, texSize } = this;
    if (!this._readPixelsBuf) {
      this._readPixelsBuf = new Uint8Array(texSize * texSize * 4);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos[1 - this._pingPong]);
    gl.readPixels(0, 0, texSize, texSize, gl.RGBA, gl.UNSIGNED_BYTE, this._readPixelsBuf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return this.cellOrder.map((cellId, i) => {
      const base = i * 4;
      const r = this._readPixelsBuf![base + 0] / 255;
      const g = this._readPixelsBuf![base + 1] / 255;
      const b = this._readPixelsBuf![base + 2] / 255;
      const a = this._readPixelsBuf![base + 3] / 255;
      return {
        cellId,
        dx:       decodeDisp(r),
        dy:       decodeDisp(g),
        forceMag: b * FORCE_MAX,
        converged: a > 0.5,
      };
    });
  }

  /**
   * Upload fresh CPU-side state into the READ texture.
   * Use after a physics epoch reset or when force_field.json is re-loaded.
   */
  uploadStates(states: CellConstraintState[]): void {
    const { gl, texSize } = this;
    const pixelCount = texSize * texSize;
    const data = new Uint8Array(pixelCount * 4);

    for (const s of states) {
      const i = this.cellOrder.indexOf(s.cellId);
      if (i < 0) continue;
      const base = i * 4;
      data[base + 0] = Math.round(encodeDisp(s.dx)       * 255);
      data[base + 1] = Math.round(encodeDisp(s.dy)       * 255);
      data[base + 2] = Math.round(encodeForce(s.forceMag) * 255);
      data[base + 3] = s.converged ? 255 : 0;
    }

    gl.bindTexture(gl.TEXTURE_2D, this.textures[this._pingPong]);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, texSize, texSize,
                     gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** Convert a cell ID to its (u, v) texel centre in [0,1] UV space */
  cellUV(cellId: string): [number, number] {
    const i = this.cellOrder.indexOf(cellId);
    if (i < 0) return [0, 0];
    const x = (i % this.texSize) + 0.5;
    const y = (Math.floor(i / this.texSize)) + 0.5;
    return [x / this.texSize, y / this.texSize];
  }

  /** Pixel (col, row) for a cell index — useful for neighbour sampling in GLSL */
  cellPixelCoord(cellIndex: number): [number, number] {
    return [cellIndex % this.texSize, Math.floor(cellIndex / this.texSize)];
  }

  destroy(): void {
    const { gl } = this;
    for (const tex of this.textures) gl.deleteTexture(tex);
    for (const fbo of this.fbos)     gl.deleteFramebuffer(fbo);
    this._readPixelsBuf = null;
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  private _allocTex(initData: Uint8Array | null): [WebGLTexture, WebGLFramebuffer] {
    const { gl, texSize } = this;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, texSize, texSize, 0,
                  gl.RGBA, gl.UNSIGNED_BYTE, initData);
    gl.bindTexture(gl.TEXTURE_2D, null);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                            gl.TEXTURE_2D, tex, 0);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.error('[ConstraintTexture] FBO incomplete:', status);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return [tex, fbo];
  }
}

// ── Factory: load from channels/physics JSON ──────────────────────────────────

/**
 * Async factory — fetches the two JSON channels and constructs the texture.
 * Call once at app init, after WebGL context is ready.
 */
export async function createConstraintTexture(
  gl: WebGL2RenderingContext,
  opts: {
    forceFieldUrl?:  string;
    cellRegistryUrl?: string;
  } = {}
): Promise<ConstraintTexture> {
  const ffUrl  = opts.forceFieldUrl   ?? '/channels/physics/force_field.json';
  const regUrl = opts.cellRegistryUrl ?? '/channels/physics/cell_registry.json';

  const [ffRes, regRes] = await Promise.all([fetch(ffUrl), fetch(regUrl)]);
  if (!ffRes.ok)  throw new Error(`[ConstraintTexture] fetch failed: ${ffUrl}`);
  if (!regRes.ok) throw new Error(`[ConstraintTexture] fetch failed: ${regUrl}`);

  const forceField    = await ffRes.json();
  const cellRegistry  = await regRes.json();

  return new ConstraintTexture({ gl, forceField, cellRegistry });
}
