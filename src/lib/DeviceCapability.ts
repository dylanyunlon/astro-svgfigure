/**
 * DeviceCapability.ts — GPU detection, performance grading, responsive canvas
 *
 * AT: "we factored in device capability and performance to fine-tune the experience"
 *
 * Exports:
 *   DeviceCapability        — singleton: GPU probe + tier classification
 *   ResponsiveCanvas        — ties a <canvas> to window resize, honouring DPR + tier
 *
 * Performance tiers:
 *   HIGH  — WebGL2 + float textures + drawBuffers + maxTexture ≥ 8192
 *             → all post-FX on, render at 2× device-pixel-ratio
 *   MED   — WebGL2 present but limits or extensions missing
 *             → bloom + DOF only, 1× DPR
 *   LOW   — WebGL1 / no float textures / mobile fallback
 *             → no post-processing, reduced cell count
 *
 * Usage:
 *   import { DeviceCapability, ResponsiveCanvas, PerformanceTier } from '$lib/DeviceCapability';
 *
 *   const cap  = DeviceCapability.probe();       // idempotent
 *   const tier = cap.tier;                       // 'HIGH' | 'MED' | 'LOW'
 *
 *   const rc = new ResponsiveCanvas(canvas, cap);
 *   rc.onResize((vp) => { nuke.resize(vp.width, vp.height); });
 *   // … later …
 *   rc.destroy();
 *
 * xiaodi #57
 */

// ── Tier enum ─────────────────────────────────────────────────────────────────

export type PerformanceTier = 'HIGH' | 'MED' | 'LOW';

// ── GPU capability snapshot ───────────────────────────────────────────────────

export interface GPUCapabilities {
  /** WebGL2 context obtained successfully */
  hasWebGL2: boolean;
  /** OES_texture_float (WebGL1) or native float textures (WebGL2) */
  hasFloatTextures: boolean;
  /** WEBGL_draw_buffers (WebGL1) or native MRT (WebGL2) */
  hasDrawBuffers: boolean;
  /** gl.MAX_TEXTURE_SIZE value (0 if GL unavailable) */
  maxTextureSize: number;
  /** gl.MAX_RENDERBUFFER_SIZE value (0 if GL unavailable) */
  maxRenderbufferSize: number;
  /** Derived tier */
  tier: PerformanceTier;
  /** DPR multiplier applied at this tier (2 | 1 | 0.75) */
  dprScale: number;
  /** Suggested max cell count for the cell-pubsub layer */
  maxCells: number;
}

// ── Viewport descriptor emitted by ResponsiveCanvas ──────────────────────────

export interface Viewport {
  /** CSS pixel width of the canvas */
  cssWidth: number;
  /** CSS pixel height of the canvas */
  cssHeight: number;
  /** Physical pixel width (= cssWidth × dpr × dprScale) */
  width: number;
  /** Physical pixel height (= cssHeight × dpr × dprScale) */
  height: number;
  /** Effective device pixel ratio used */
  dpr: number;
}

// ── Singleton probe ───────────────────────────────────────────────────────────

const _TIER_CONFIG: Record<PerformanceTier, { dprScale: number; maxCells: number }> = {
  HIGH: { dprScale: 2,    maxCells: 4096 },
  MED:  { dprScale: 1,    maxCells: 1024 },
  LOW:  { dprScale: 0.75, maxCells: 256  },
};

let _cached: GPUCapabilities | null = null;

export const DeviceCapability = {
  /**
   * Probe the GPU once and cache the result.
   * Safe to call in SSR (returns LOW-tier stub when `document` is absent).
   */
  probe(): GPUCapabilities {
    if (_cached) return _cached;

    if (typeof document === 'undefined') {
      return (_cached = _buildStub('LOW'));
    }

    // ── Try WebGL2 first ────────────────────────────────────────────────────
    const canvas = document.createElement('canvas');
    canvas.width  = 1;
    canvas.height = 1;

    const gl2 = canvas.getContext('webgl2') as WebGL2RenderingContext | null;

    if (gl2) {
      const maxTex = gl2.getParameter(gl2.MAX_TEXTURE_SIZE)         as number;
      const maxRB  = gl2.getParameter(gl2.MAX_RENDERBUFFER_SIZE)    as number;

      // Float textures: WebGL2 supports RGBA16F/RGBA32F natively; verify via
      // EXT_color_buffer_float (needed for float FBOs).
      const hasFloat = !!gl2.getExtension('EXT_color_buffer_float');

      // MRT is core in WebGL2 (gl.drawBuffers exists), but sanity-check limit.
      const maxDraw  = gl2.getParameter(gl2.MAX_DRAW_BUFFERS) as number;
      const hasDrawB = maxDraw >= 2;

      gl2.getExtension('OES_texture_float_linear'); // best-effort, not gating

      const tier = _classify({ hasWebGL2: true, hasFloat, hasDrawB, maxTex });

      _cached = {
        hasWebGL2: true,
        hasFloatTextures: hasFloat,
        hasDrawBuffers:   hasDrawB,
        maxTextureSize:      maxTex,
        maxRenderbufferSize: maxRB,
        tier,
        ..._TIER_CONFIG[tier],
      };
      return _cached;
    }

    // ── Fallback: WebGL1 ────────────────────────────────────────────────────
    const gl1 =
      (canvas.getContext('webgl') ??
       canvas.getContext('experimental-webgl')) as WebGLRenderingContext | null;

    if (gl1) {
      const maxTex = gl1.getParameter(gl1.MAX_TEXTURE_SIZE)         as number;
      const maxRB  = gl1.getParameter(gl1.MAX_RENDERBUFFER_SIZE)    as number;
      const hasFloat = !!gl1.getExtension('OES_texture_float');
      const hasDrawB = !!gl1.getExtension('WEBGL_draw_buffers');

      const tier = _classify({ hasWebGL2: false, hasFloat, hasDrawB, maxTex });

      _cached = {
        hasWebGL2: false,
        hasFloatTextures: hasFloat,
        hasDrawBuffers:   hasDrawB,
        maxTextureSize:      maxTex,
        maxRenderbufferSize: maxRB,
        tier,
        ..._TIER_CONFIG[tier],
      };
      return _cached;
    }

    // ── No WebGL at all ─────────────────────────────────────────────────────
    return (_cached = _buildStub('LOW'));
  },

  /** Force-clear the cache (useful for testing). */
  _reset() { _cached = null; },
};

// ── Internal helpers ──────────────────────────────────────────────────────────

function _classify(p: {
  hasWebGL2: boolean;
  hasFloat:  boolean;
  hasDrawB:  boolean;
  maxTex:    number;
}): PerformanceTier {
  // Mobile heuristic: very small texture budgets → LOW regardless.
  if (p.maxTex < 2048) return 'LOW';

  if (p.hasWebGL2 && p.hasFloat && p.hasDrawB && p.maxTex >= 8192) return 'HIGH';
  if (p.hasWebGL2) return 'MED';

  // WebGL1 with both extensions: MED; otherwise LOW.
  if (p.hasFloat && p.hasDrawB) return 'MED';
  return 'LOW';
}

function _buildStub(tier: PerformanceTier): GPUCapabilities {
  return {
    hasWebGL2:           false,
    hasFloatTextures:    false,
    hasDrawBuffers:      false,
    maxTextureSize:      0,
    maxRenderbufferSize: 0,
    tier,
    ..._TIER_CONFIG[tier],
  };
}

// ── ResponsiveCanvas ──────────────────────────────────────────────────────────

type ResizeCallback = (vp: Viewport) => void;

/**
 * Watches `window.resize` (and `visualViewport.resize` on mobile) and keeps
 * the canvas physical pixel dimensions in sync with its CSS layout size,
 * clamped to the tier DPR scale.
 *
 * Fires all registered callbacks with the new Viewport on each resize.
 *
 * Example:
 * ```ts
 * const rc = new ResponsiveCanvas(myCanvas, DeviceCapability.probe());
 * rc.onResize(({ width, height }) => nuke.resize(width, height));
 * rc.flush(); // apply immediately without waiting for first resize event
 * ```
 */
export class ResponsiveCanvas {
  private readonly _canvas:    HTMLCanvasElement;
  private readonly _cap:       GPUCapabilities;
  private readonly _callbacks: Set<ResizeCallback> = new Set();
  private _raf:   number  = 0;
  private _bound: boolean = false;

  constructor(canvas: HTMLCanvasElement, cap: GPUCapabilities) {
    this._canvas = canvas;
    this._cap    = cap;
    this._attach();
  }

  /** Register a callback invoked on every resize (and on the first `flush()`). */
  onResize(cb: ResizeCallback): this {
    this._callbacks.add(cb);
    return this;
  }

  /** Remove a previously registered callback. */
  offResize(cb: ResizeCallback): this {
    this._callbacks.delete(cb);
    return this;
  }

  /**
   * Immediately compute + apply the current viewport without waiting for a
   * resize event.  Call once after construction to set the initial size.
   */
  flush(): Viewport {
    return this._update();
  }

  /** Remove all event listeners and cancel any pending rAF. */
  destroy(): void {
    cancelAnimationFrame(this._raf);
    window.removeEventListener('resize', this._onResize);
    if (typeof visualViewport !== 'undefined') {
      visualViewport.removeEventListener('resize', this._onResize);
    }
    this._callbacks.clear();
    this._bound = false;
  }

  // ── Private ───────────────────────────────────────────────────────────────

  private _attach(): void {
    if (this._bound) return;
    this._bound = true;
    window.addEventListener('resize', this._onResize, { passive: true });
    // `visualViewport` fires on mobile when the OSK appears/disappears.
    if (typeof visualViewport !== 'undefined') {
      visualViewport.addEventListener('resize', this._onResize, { passive: true });
    }
  }

  /** Debounce via rAF so rapid resize events collapse into one update. */
  private readonly _onResize = (): void => {
    cancelAnimationFrame(this._raf);
    this._raf = requestAnimationFrame(() => this._update());
  };

  private _update(): Viewport {
    const canvas   = this._canvas;
    const { dprScale } = this._cap;

    // CSS layout size (may be 0 before first paint — guard).
    const cssWidth  = canvas.clientWidth  || canvas.offsetWidth  || 1;
    const cssHeight = canvas.clientHeight || canvas.offsetHeight || 1;

    // Device pixel ratio, clamped to sensible range.
    const rawDpr = typeof devicePixelRatio === 'number' ? devicePixelRatio : 1;
    const dpr    = Math.max(0.5, Math.min(rawDpr * dprScale, 4));

    const width  = Math.round(cssWidth  * dpr);
    const height = Math.round(cssHeight * dpr);

    // Only update canvas backing store when dimensions actually changed.
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width  = width;
      canvas.height = height;
    }

    const vp: Viewport = { cssWidth, cssHeight, width, height, dpr };
    for (const cb of this._callbacks) cb(vp);
    return vp;
  }
}

// ── Convenience re-exports ────────────────────────────────────────────────────

/**
 * Quick helper: returns the feature flags the active tier enables.
 *
 * ```ts
 * const { bloom, dof, postProcess } = tierFeatures('HIGH');
 * // → { bloom: true, dof: true, postProcess: true }
 * ```
 */
export function tierFeatures(tier: PerformanceTier) {
  return {
    postProcess: tier !== 'LOW',
    bloom:       tier !== 'LOW',
    dof:         tier !== 'LOW',
    highRes:     tier === 'HIGH',
  } as const;
}
