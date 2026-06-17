/**
 * cell-env-detect.ts — Browser rendering environment detection + Canvas2D fallback
 *
 * Fuses upstream/pixijs-engine environment-browser patterns to probe WebGL2 and
 * WebGPU availability at runtime, then selects the highest-capability renderer
 * the current environment can sustain:
 *
 *   WebGPU   → GPU compute + advanced render pipeline (future-facing tier)
 *   WebGL2   → Full post-FX pipeline (bloom, DOF, MSDF text, particles)
 *   WebGL1   → Reduced pipeline, no float FBOs, no MRT
 *   Canvas2D → CPU fallback: no GPU effects, PixiJS CanvasRenderer path
 *
 * Integrates with DeviceCapability.ts (performance tier) and pixi-cell-renderer.ts
 * (Application init preference).
 *
 * Key exports:
 *   detectCellEnv()         — async probe, returns CellEnvReport (cached after first call)
 *   CellEnvAdapter          — PixiJS Adapter for the detected environment
 *   buildRendererPreference — maps CellRenderBackend → autoDetectRenderer preference array
 *   createCellCanvas2D      — Canvas2D fallback: draws cell graph without WebGL
 *   CellCanvas2DRenderer    — class-based wrapper for the Canvas2D fallback path
 *
 * Upstream references:
 *   upstream/pixijs-engine/src/environment-browser/BrowserAdapter.ts
 *   upstream/pixijs-engine/src/environment/autoDetectEnvironment.ts
 *   upstream/pixijs-engine/src/rendering/renderers/autoDetectRenderer.ts
 *   upstream/pixijs-engine/src/utils/browser/isWebGLSupported.ts
 *   upstream/pixijs-engine/src/utils/browser/isWebGPUSupported.ts
 *   upstream/pixijs-engine/src/rendering/renderers/canvas/CanvasContextSystem.ts
 *
 * Algorithm:
 *   1. SSR guard — return Canvas2D stub when `document` is absent
 *   2. Probe WebGPU via navigator.gpu.requestAdapter()  (async, cached)
 *   3. Probe WebGL2 via canvas.getContext('webgl2')      (sync, cached)
 *   4. Probe WebGL1 via canvas.getContext('webgl')       (sync, cached)
 *   5. Derive CellRenderBackend enum value
 *   6. Interrogate WebGL2/WebGL1 extensions for feature flags
 *   7. Emit CellEnvReport with backend + capabilities
 *   8. buildRendererPreference converts CellEnvReport → PixiJS preference array
 *      so autoDetectRenderer respects the probe result instead of re-probing
 *
 * Canvas2D fallback draws rounded-rect cells + bezier edges using the native
 * CanvasRenderingContext2D API — no SVG string construction.
 *
 * [CELL-ENV-DETECT] debug prefix.
 */

// ── Backend enum ──────────────────────────────────────────────────────────────

/**
 * Ordered rendering backends from highest to lowest capability.
 * Drives autoDetectRenderer preference and feature-flag gating.
 */
export type CellRenderBackend = 'webgpu' | 'webgl2' | 'webgl1' | 'canvas2d';

// ── Feature flags ─────────────────────────────────────────────────────────────

/** Fine-grained GPU extension flags for the selected backend. */
export interface CellGpuFeatures {
  /** EXT_color_buffer_float — float FBOs for HDR bloom / GPGPU particles */
  colorBufferFloat: boolean;
  /** OES_texture_float_linear — bilinear sampling of float textures */
  floatLinear: boolean;
  /** WEBGL_draw_buffers (WebGL1) or native MRT (WebGL2) — multiple render targets */
  drawBuffers: boolean;
  /** EXT_texture_filter_anisotropic — anisotropic texture filtering */
  anisotropicFilter: boolean;
  /** WEBGL_depth_texture — shadow maps and depth prepass */
  depthTexture: boolean;
  /** ANGLE_instanced_arrays (WebGL1) or native (WebGL2) — GPU instancing */
  instancedArrays: boolean;
  /** OES_vertex_array_object (WebGL1) or native (WebGL2) — VAO support */
  vertexArrayObject: boolean;
  /** Largest usable texture side in pixels (0 = no WebGL) */
  maxTextureSize: number;
  /** Largest usable renderbuffer side in pixels (0 = no WebGL) */
  maxRenderbufferSize: number;
  /** Max simultaneous draw buffer attachments (1 = no MRT) */
  maxDrawBuffers: number;
}

// ── Full environment report ───────────────────────────────────────────────────

/** Complete snapshot of the current browser rendering environment. */
export interface CellEnvReport {
  /** Selected rendering backend */
  backend: CellRenderBackend;
  /** Whether WebGPU is available (navigator.gpu + adapter + device) */
  hasWebGPU: boolean;
  /** Whether WebGL2 context creation succeeded */
  hasWebGL2: boolean;
  /** Whether WebGL1 context creation succeeded (implies WebGL2 unavailable) */
  hasWebGL1: boolean;
  /** GPU extension flags (all false when backend = canvas2d) */
  features: CellGpuFeatures;
  /** True when running server-side (no document/canvas) */
  isSSR: boolean;
  /** True when running inside a Web Worker (no DOM) */
  isWorker: boolean;
  /**
   * Recommended PixiJS autoDetectRenderer preference order.
   * Pass directly as `preference` in AutoDetectOptions.
   */
  pixiPreference: ('webgpu' | 'webgl' | 'canvas')[];
}

// ── Probe cache ───────────────────────────────────────────────────────────────

let _cached: CellEnvReport | null = null;

// ── Null feature set (Canvas2D / SSR) ─────────────────────────────────────────

const _NULL_FEATURES: CellGpuFeatures = {
  colorBufferFloat:   false,
  floatLinear:        false,
  drawBuffers:        false,
  anisotropicFilter:  false,
  depthTexture:       false,
  instancedArrays:    false,
  vertexArrayObject:  false,
  maxTextureSize:     0,
  maxRenderbufferSize: 0,
  maxDrawBuffers:     1,
};

// ── WebGPU async probe ────────────────────────────────────────────────────────

/**
 * Attempt to acquire a GPU adapter + device via the WebGPU API.
 * Returns false if navigator.gpu is absent, the adapter is null, or device
 * request throws — matching upstream isWebGPUSupported logic.
 */
async function _probeWebGPU(): Promise<boolean> {
  if (typeof navigator === 'undefined') return false;
  // navigator.gpu is not in all TypeScript libs — access via bracket notation
  const gpu = (navigator as Record<string, unknown>)['gpu'] as
    | { requestAdapter(opts?: unknown): Promise<{ requestDevice(): Promise<unknown> } | null> }
    | undefined;
  if (!gpu) return false;
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return false;
    await adapter.requestDevice();
    return true;
  } catch {
    return false;
  }
}

// ── WebGL2 sync probe ─────────────────────────────────────────────────────────

/**
 * Create a 1×1 probe canvas and attempt `getContext('webgl2')`.
 * Returns the context on success, null otherwise.
 * The context is explicitly lost to free GPU memory before returning null.
 */
function _probeWebGL2(): WebGL2RenderingContext | null {
  if (typeof document === 'undefined') return null;
  try {
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    const gl = canvas.getContext('webgl2', { stencil: true, failIfMajorPerformanceCaveat: false });
    return gl ?? null;
  } catch {
    return null;
  }
}

// ── WebGL1 sync probe ─────────────────────────────────────────────────────────

/**
 * Probe WebGL1 context (fallback after WebGL2 fails).
 * Mirrors upstream isWebGLSupported: tries 'webgl', then 'experimental-webgl'.
 */
function _probeWebGL1(): WebGLRenderingContext | null {
  if (typeof document === 'undefined') return null;
  try {
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    const gl =
      (canvas.getContext('webgl', { stencil: true }) ??
       canvas.getContext('experimental-webgl', { stencil: true })) as WebGLRenderingContext | null;
    return gl ?? null;
  } catch {
    return null;
  }
}

// ── WebGL2 feature interrogation ──────────────────────────────────────────────

/** Extract all relevant feature flags from a live WebGL2 context. */
function _featuresFromGL2(gl: WebGL2RenderingContext): CellGpuFeatures {
  // EXT_color_buffer_float: required for float-format FBOs (bloom GPGPU)
  const colorBufferFloat  = !!gl.getExtension('EXT_color_buffer_float');
  // OES_texture_float_linear: bilinear on float textures (best-effort)
  const floatLinear       = !!gl.getExtension('OES_texture_float_linear');
  // MRT is core in WebGL2; sanity-check max draw buffers
  const maxDrawBuffers    = gl.getParameter(gl.MAX_DRAW_BUFFERS) as number;
  const drawBuffers       = maxDrawBuffers >= 2;
  // Anisotropic filtering extension
  const anisotropicFilter = !!(
    gl.getExtension('EXT_texture_filter_anisotropic') ??
    gl.getExtension('WEBKIT_EXT_texture_filter_anisotropic') ??
    gl.getExtension('MOZ_EXT_texture_filter_anisotropic')
  );
  // Depth textures (shadow map pass)
  const depthTexture      = !!gl.getExtension('WEBGL_depth_texture');
  // VAO + instancing are core in WebGL2
  const instancedArrays   = true;
  const vertexArrayObject = true;

  const maxTextureSize      = gl.getParameter(gl.MAX_TEXTURE_SIZE)      as number;
  const maxRenderbufferSize = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) as number;

  return {
    colorBufferFloat,
    floatLinear,
    drawBuffers,
    anisotropicFilter,
    depthTexture,
    instancedArrays,
    vertexArrayObject,
    maxTextureSize,
    maxRenderbufferSize,
    maxDrawBuffers,
  };
}

// ── WebGL1 feature interrogation ──────────────────────────────────────────────

/** Extract feature flags from a WebGL1 context. */
function _featuresFromGL1(gl: WebGLRenderingContext): CellGpuFeatures {
  const colorBufferFloat  = !!gl.getExtension('OES_texture_float');
  const floatLinear       = !!gl.getExtension('OES_texture_float_linear');
  const drawBufExt        = gl.getExtension('WEBGL_draw_buffers');
  const drawBuffers       = !!drawBufExt;
  const maxDrawBuffers    = drawBufExt
    ? (gl.getParameter(0x8824 /* MAX_DRAW_BUFFERS_WEBGL */) as number)
    : 1;
  const anisotropicFilter = !!(
    gl.getExtension('EXT_texture_filter_anisotropic') ??
    gl.getExtension('WEBKIT_EXT_texture_filter_anisotropic') ??
    gl.getExtension('MOZ_EXT_texture_filter_anisotropic')
  );
  const depthTexture      = !!gl.getExtension('WEBGL_depth_texture');
  const instancedArrays   = !!gl.getExtension('ANGLE_instanced_arrays');
  const vertexArrayObject = !!gl.getExtension('OES_vertex_array_object');

  const maxTextureSize      = gl.getParameter(gl.MAX_TEXTURE_SIZE)      as number;
  const maxRenderbufferSize = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) as number;

  return {
    colorBufferFloat,
    floatLinear,
    drawBuffers,
    anisotropicFilter,
    depthTexture,
    instancedArrays,
    vertexArrayObject,
    maxTextureSize,
    maxRenderbufferSize,
    maxDrawBuffers,
  };
}

// ── Context cleanup ───────────────────────────────────────────────────────────

/** Return a WebGL context's resources to the GPU after probing. */
function _loseContext(gl: WebGL2RenderingContext | WebGLRenderingContext): void {
  try {
    const ext = gl.getExtension('WEBGL_lose_context');
    ext?.loseContext();
  } catch { /* ignore — context may already be lost */ }
}

// ── Pixi preference builder ───────────────────────────────────────────────────

/**
 * Convert a CellRenderBackend to a PixiJS autoDetectRenderer `preference` array.
 *
 * PixiJS maps 'webgl' → WebGLRenderer (WebGL2 preferring, falling back to WebGL1).
 * We only list backends the probe confirmed available, preventing PixiJS from
 * re-probing and potentially picking a higher tier we already know fails.
 */
export function buildRendererPreference(
  backend: CellRenderBackend
): ('webgpu' | 'webgl' | 'canvas')[] {
  switch (backend) {
    case 'webgpu':  return ['webgpu', 'webgl', 'canvas'];
    case 'webgl2':  return ['webgl', 'canvas'];
    case 'webgl1':  return ['webgl', 'canvas'];
    case 'canvas2d': return ['canvas'];
  }
}

// ── Main probe ────────────────────────────────────────────────────────────────

/**
 * detectCellEnv — async capability probe.
 *
 * Safe to call before Application.init().  Results are cached; subsequent calls
 * return the same object without re-probing.
 *
 * @example
 * ```ts
 * import { detectCellEnv, buildRendererPreference } from '$lib/renderers/cell-env-detect';
 *
 * const env = await detectCellEnv();
 * console.log('[CELL-ENV-DETECT] backend:', env.backend);
 *
 * if (env.backend === 'canvas2d') {
 *   // use CellCanvas2DRenderer — no PixiJS required
 *   const c2d = new CellCanvas2DRenderer(canvas);
 *   c2d.render(cells, edges);
 * } else {
 *   const app = new Application();
 *   await app.init({
 *     canvas,
 *     preference: buildRendererPreference(env.backend),
 *   });
 * }
 * ```
 */
export async function detectCellEnv(): Promise<CellEnvReport> {
  if (_cached) return _cached;

  // ── SSR guard ────────────────────────────────────────────────────────────
  if (typeof document === 'undefined') {
    return (_cached = {
      backend:         'canvas2d',
      hasWebGPU:       false,
      hasWebGL2:       false,
      hasWebGL1:       false,
      features:        { ..._NULL_FEATURES },
      isSSR:           true,
      isWorker:        false,
      pixiPreference:  ['canvas'],
    });
  }

  // ── Worker guard ─────────────────────────────────────────────────────────
  const isWorker = typeof WorkerGlobalScope !== 'undefined' &&
    self instanceof WorkerGlobalScope;

  // ── WebGPU probe (async) ─────────────────────────────────────────────────
  const hasWebGPU = await _probeWebGPU();

  // ── WebGL2 probe (sync) ──────────────────────────────────────────────────
  const gl2 = _probeWebGL2();
  const hasWebGL2 = gl2 !== null;

  if (hasWebGPU) {
    // WebGPU path: still interrogate WebGL2 for feature parity checks,
    // but drive PixiJS toward WebGPU renderer.
    const features = gl2 ? _featuresFromGL2(gl2) : { ..._NULL_FEATURES };
    if (gl2) _loseContext(gl2);

    return (_cached = {
      backend:        'webgpu',
      hasWebGPU:      true,
      hasWebGL2:      hasWebGL2,
      hasWebGL1:      false,
      features,
      isSSR:          false,
      isWorker,
      pixiPreference: buildRendererPreference('webgpu'),
    });
  }

  if (hasWebGL2 && gl2) {
    const features = _featuresFromGL2(gl2);
    _loseContext(gl2);

    return (_cached = {
      backend:        'webgl2',
      hasWebGPU:      false,
      hasWebGL2:      true,
      hasWebGL1:      false,
      features,
      isSSR:          false,
      isWorker,
      pixiPreference: buildRendererPreference('webgl2'),
    });
  }

  // ── WebGL1 probe (sync, fallback) ─────────────────────────────────────────
  const gl1 = _probeWebGL1();
  const hasWebGL1 = gl1 !== null;

  if (hasWebGL1 && gl1) {
    const features = _featuresFromGL1(gl1);
    _loseContext(gl1);

    return (_cached = {
      backend:        'webgl1',
      hasWebGPU:      false,
      hasWebGL2:      false,
      hasWebGL1:      true,
      features,
      isSSR:          false,
      isWorker,
      pixiPreference: buildRendererPreference('webgl1'),
    });
  }

  // ── Canvas2D fallback (no WebGL at all) ───────────────────────────────────
  return (_cached = {
    backend:        'canvas2d',
    hasWebGPU:      false,
    hasWebGL2:      false,
    hasWebGL1:      false,
    features:       { ..._NULL_FEATURES },
    isSSR:          false,
    isWorker,
    pixiPreference: buildRendererPreference('canvas2d'),
  });
}

/** Force-clear the probe cache (useful in tests). */
export function _resetCellEnvCache(): void {
  _cached = null;
}

// ── Canvas2D fallback renderer ────────────────────────────────────────────────

/**
 * Minimal descriptor types for the Canvas2D fallback path.
 * Mirrors the CellDescriptor / EdgeDescriptor shapes in pixi-cell-renderer.ts
 * so callers can share the same data schema.
 */
export interface C2DCellDescriptor {
  cell_id: string;
  species?: string;
  bbox: { x: number; y: number; w: number; h: number };
  fill_color?:   string;
  stroke_color?: string;
  label?:        string;
  opacity?:      number;
  font_size?:    number;
}

export interface C2DEdgeDescriptor {
  from: string;
  to:   string;
  color?: string;
  width?: number;
}

// ── Species → color map for Canvas2D fallback ─────────────────────────────────

const _SPECIES_COLOR: Record<string, string> = {
  'cil-eye':         '#5C6BC0',
  'cil-vector':      '#26A69A',
  'cil-bolt':        '#EF6C00',
  'cil-plus':        '#66BB6A',
  'cil-arrow-right': '#AB47BC',
  'cil-filter':      '#EC407A',
  'cil-code':        '#29B6F6',
  'cil-layers':      '#FFA726',
  'cil-loop':        '#26C6DA',
  'cil-graph':       '#7E57C2',
};

const _FALLBACK_FILL   = '#2a2a2a';
const _FALLBACK_STROKE = '#555555';

// ── Bezier control point helper ───────────────────────────────────────────────

/** Compute S-curve control points between two rectangles. */
function _bezierCtrl(
  x1: number, y1: number,
  x2: number, y2: number
): { cx1: number; cy1: number; cx2: number; cy2: number } {
  const dy = (y2 - y1) * 0.5;
  return { cx1: x1, cy1: y1 + dy, cx2: x2, cy2: y2 - dy };
}

// ── Canvas2D render function ──────────────────────────────────────────────────

/**
 * createCellCanvas2D — one-shot Canvas2D render of a cell graph.
 *
 * Draws rounded-rect cells + S-curve bezier edges directly onto the provided
 * CanvasRenderingContext2D.  Uses PixiJS CanvasRenderer coordinate conventions
 * (top-left origin, CSS pixel units) — no GPU, no SVG strings.
 *
 * @param ctx    - Native 2D rendering context of the target canvas
 * @param cells  - Cell descriptors (bbox in CSS pixels)
 * @param edges  - Edge descriptors (from/to reference cell_id)
 * @param opts   - Optional overrides for background and grid
 */
export function createCellCanvas2D(
  ctx: CanvasRenderingContext2D,
  cells: C2DCellDescriptor[],
  edges: C2DEdgeDescriptor[],
  opts: { background?: string; showGrid?: boolean } = {}
): void {
  const { background = '#111111', showGrid = false } = opts;

  const W = ctx.canvas.width;
  const H = ctx.canvas.height;

  // ── Background ──────────────────────────────────────────────────────────
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, W, H);

  // ── Optional grid ────────────────────────────────────────────────────────
  if (showGrid) {
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    const step = 32;
    for (let x = 0; x < W; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }
    for (let y = 0; y < H; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }
  }

  // ── Build lookup for edge routing ────────────────────────────────────────
  const cellMap = new Map<string, C2DCellDescriptor>();
  for (const cell of cells) cellMap.set(cell.cell_id, cell);

  // ── Edges (drawn first, under cells) ────────────────────────────────────
  for (const edge of edges) {
    const src = cellMap.get(edge.from);
    const dst = cellMap.get(edge.to);
    if (!src || !dst) continue;

    // Exit from bottom-centre of src, enter top-centre of dst
    const x1 = src.bbox.x + src.bbox.w * 0.5;
    const y1 = src.bbox.y + src.bbox.h;
    const x2 = dst.bbox.x + dst.bbox.w * 0.5;
    const y2 = dst.bbox.y;

    const { cx1, cy1, cx2, cy2 } = _bezierCtrl(x1, y1, x2, y2);

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.bezierCurveTo(cx1, cy1, cx2, cy2, x2, y2);
    ctx.strokeStyle = edge.color ?? 'rgba(120,120,200,0.5)';
    ctx.lineWidth   = edge.width ?? 1.5;
    ctx.stroke();
  }

  // ── Cells ────────────────────────────────────────────────────────────────
  for (const cell of cells) {
    const { bbox, fill_color, stroke_color, label, opacity = 1, font_size = 11 } = cell;
    const { x, y, w, h } = bbox;
    const r = Math.min(w, h) * 0.15; // corner radius

    const fill   = fill_color   ?? _SPECIES_COLOR[cell.species ?? ''] ?? _FALLBACK_FILL;
    const stroke = stroke_color ?? _FALLBACK_STROKE;

    ctx.globalAlpha = Math.max(0, Math.min(1, opacity));

    // Rounded rect fill
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    ctx.fillStyle = fill;
    ctx.fill();

    // Rounded rect stroke
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    ctx.strokeStyle = stroke;
    ctx.lineWidth   = 1;
    ctx.stroke();

    // Label text
    if (label) {
      ctx.globalAlpha  = opacity;
      ctx.fillStyle    = '#ffffff';
      ctx.font         = `${font_size}px monospace`;
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      // Clip to cell bbox to prevent overflow
      ctx.save();
      ctx.beginPath();
      ctx.roundRect(x + 2, y + 2, w - 4, h - 4, r);
      ctx.clip();
      ctx.fillText(label, x + w * 0.5, y + h * 0.5, w - 8);
      ctx.restore();
    }

    ctx.globalAlpha = 1;
  }
}

// ── Class-based Canvas2D wrapper ──────────────────────────────────────────────

/**
 * CellCanvas2DRenderer — stateful Canvas2D renderer for the cell graph.
 *
 * Equivalent in API shape to the PixiJS-backed renderCellGraph() in
 * pixi-cell-renderer.ts, but renders via native Canvas2D without any WebGL.
 *
 * Use when detectCellEnv() returns backend === 'canvas2d'.
 *
 * @example
 * ```ts
 * const env = await detectCellEnv();
 * if (env.backend === 'canvas2d') {
 *   const renderer = new CellCanvas2DRenderer(myCanvas, { background: '#0a0a0a' });
 *   renderer.render(cells, edges);
 *   renderer.destroy();
 * }
 * ```
 */
export class CellCanvas2DRenderer {
  private readonly _canvas:  HTMLCanvasElement;
  private readonly _ctx:     CanvasRenderingContext2D;
  private readonly _opts:    { background?: string; showGrid?: boolean };
  private _destroyed = false;

  constructor(
    canvas: HTMLCanvasElement,
    opts: { background?: string; showGrid?: boolean } = {}
  ) {
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error(
        '[CELL-ENV-DETECT] CellCanvas2DRenderer: canvas.getContext("2d") returned null'
      );
    }
    this._canvas = canvas;
    this._ctx    = ctx;
    this._opts   = opts;
  }

  /**
   * Render one frame.  May be called repeatedly (e.g. inside a rAF loop)
   * to animate cells changing position.
   */
  render(cells: C2DCellDescriptor[], edges: C2DEdgeDescriptor[]): void {
    if (this._destroyed) {
      console.warn('[CELL-ENV-DETECT] CellCanvas2DRenderer.render() called after destroy()');
      return;
    }
    createCellCanvas2D(this._ctx, cells, edges, this._opts);
  }

  /** Clear the canvas and release references. */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
  }

  /** The underlying HTMLCanvasElement. */
  get canvas(): HTMLCanvasElement { return this._canvas; }

  /** The live CanvasRenderingContext2D. */
  get context(): CanvasRenderingContext2D { return this._ctx; }
}

// ── CellEnvAdapter — PixiJS Adapter shim for detected environment ─────────────

/**
 * CellEnvAdapter — minimal Adapter implementation drawn from BrowserAdapter.
 *
 * Mirrors upstream/pixijs-engine/src/environment-browser/BrowserAdapter.ts
 * so callers can do:
 *   import { DOMAdapter } from 'upstream/pixijs-engine/src/environment/adapter';
 *   DOMAdapter.set(CellEnvAdapter);
 * before app.init() to ensure PixiJS uses the same environment that
 * detectCellEnv() probed.
 *
 * In SSR / Worker contexts the getWebGLRenderingContext shim returns a no-op
 * guard that prevents PixiJS from accessing the real (absent) global.
 */
export const CellEnvAdapter = {
  createCanvas(width?: number, height?: number): HTMLCanvasElement {
    const canvas = typeof document !== 'undefined'
      ? document.createElement('canvas')
      : ({ width: 0, height: 0 } as unknown as HTMLCanvasElement);
    if (width  !== undefined) canvas.width  = width;
    if (height !== undefined) canvas.height = height;
    return canvas;
  },

  createImage(): HTMLImageElement {
    return typeof Image !== 'undefined'
      ? new Image()
      : ({ src: '' } as unknown as HTMLImageElement);
  },

  getCanvasRenderingContext2D(): typeof CanvasRenderingContext2D {
    return typeof CanvasRenderingContext2D !== 'undefined'
      ? CanvasRenderingContext2D
      : (class {} as unknown as typeof CanvasRenderingContext2D);
  },

  getWebGLRenderingContext(): typeof WebGLRenderingContext {
    return typeof WebGLRenderingContext !== 'undefined'
      ? WebGLRenderingContext
      : (class {} as unknown as typeof WebGLRenderingContext);
  },

  getNavigator(): Navigator {
    return typeof navigator !== 'undefined'
      ? navigator
      : ({ userAgent: 'CellEnvAdapter/SSR', gpu: null } as unknown as Navigator);
  },

  getBaseUrl(): string {
    if (typeof document !== 'undefined') return document.baseURI ?? '';
    if (typeof location !== 'undefined') return location.href;
    return '';
  },

  getFontFaceSet(): FontFaceSet | null {
    return typeof document !== 'undefined' ? document.fonts : null;
  },

  fetch(url: RequestInfo | URL, options?: RequestInit): Promise<Response> {
    return fetch(url as RequestInfo, options);
  },

  parseXML(xml: string): Document {
    if (typeof DOMParser !== 'undefined') {
      return new DOMParser().parseFromString(xml, 'text/xml');
    }
    throw new Error('[CELL-ENV-DETECT] DOMParser not available in this environment');
  },
} as const;
