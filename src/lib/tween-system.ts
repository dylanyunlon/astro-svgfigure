/**
 * tween-system.ts — Animation tween system for cell graph rendering
 *
 * Provides:
 *   TweenManager   — global tween orchestrator, update(dt) per frame
 *                    supports object property tweens, math tweens, frame tweens
 *   VelocityTracker — tracks positional velocity for inertia + motion blur
 *   SplineInterpolation — Catmull-Rom cubic spline for path smoothing
 *
 * Usage in cell graph:
 *   - TweenManager.tween(cell, { x: 200, y: 300 }, 0.5, Easing.easeOut)
 *   - VelocityTracker.track(cellId, x, y) → getVelocity(cellId) → motion blur
 *   - SplineInterpolation.catmullRom(controlPoints, t) → smooth edge routing
 *
 * Upstream reference:
 *   upstream/theatre-js/src/
 *   upstream/animation-editor/
 *   upstream/thing-editor/
 *
 * AT references: TweenManager ×71, VelocityTracker ×8, SplineInterpolation ×35
 */

// ── Easing functions ──────────────────────────────────────────────────────────

export type EasingFn = (t: number) => number;

export const Easing = {
  /** t stays linear */
  linear: (t: number): number => t,

  /** accelerate from zero */
  easeIn: (t: number): number => t * t,

  /** decelerate to zero */
  easeOut: (t: number): number => t * (2 - t),

  /** accelerate then decelerate */
  easeInOut: (t: number): number =>
    t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,

  /** overshoot and settle — back easing */
  back: (t: number): number => {
    const c1 = 1.70158;
    const c3 = c1 + 1;
    return c3 * t * t * t - c1 * t * t;
  },

  /** elastic spring oscillation */
  elastic: (t: number): number => {
    if (t === 0 || t === 1) return t;
    const c4 = (2 * Math.PI) / 3;
    return -(Math.pow(2, 10 * t - 10) * Math.sin((t * 10 - 10.75) * c4));
  },

  /** bounces at the end */
  bounce: (t: number): number => {
    const n1 = 7.5625;
    const d1 = 2.75;
    if (t < 1 / d1) {
      return n1 * t * t;
    } else if (t < 2 / d1) {
      t -= 1.5 / d1;
      return n1 * t * t + 0.75;
    } else if (t < 2.5 / d1) {
      t -= 2.25 / d1;
      return n1 * t * t + 0.9375;
    } else {
      t -= 2.625 / d1;
      return n1 * t * t + 0.984375;
    }
  },
} as const satisfies Record<string, EasingFn>;

// ── Internal tween state ──────────────────────────────────────────────────────

interface TweenEntry {
  /** unique handle for cancellation */
  id: number;
  /** elapsed time in seconds */
  elapsed: number;
  /** total duration in seconds */
  duration: number;
  easing: EasingFn;
  onUpdate: (progress: number) => void;
  onComplete?: () => void;
  /** false once complete or cancelled */
  active: boolean;
}

// ── TweenHandle — returned to caller for cancellation ────────────────────────

export interface TweenHandle {
  /** cancel before completion */
  cancel(): void;
  /** resolves when tween completes or is cancelled */
  promise: Promise<void>;
}

// ── MathTween — animates a standalone number, not an object property ─────────

export interface MathTweenHandle extends TweenHandle {
  /** latest interpolated value */
  readonly value: number;
}

// ── FrameTween — epoch-stepped tween, advances by integer frames ─────────────

export interface FrameTweenHandle {
  /** advance by one epoch step */
  step(): void;
  cancel(): void;
  readonly done: boolean;
  readonly frame: number;
  readonly totalFrames: number;
}

// ── TweenManager ─────────────────────────────────────────────────────────────

/**
 * Global tween manager.
 *
 * Call `TweenManager.update(dt)` once per render frame (dt in seconds).
 * All active tweens are evaluated and completed ones removed automatically.
 *
 * ```ts
 * // Animate a cell's x/y position over 0.4 s with easeOut
 * TweenManager.tween(cell.position, { x: 400, y: 200 }, 0.4, Easing.easeOut);
 *
 * // Animate a standalone number
 * const mt = TweenManager.mathTween(0, 1, 0.6, Easing.easeInOut);
 * // ... later: mt.value gives current interpolated value
 *
 * // Epoch-stepped frame tween (e.g. 30 frames, 1 step per epoch tick)
 * const ft = TweenManager.frameTween(obj, { alpha: 0 }, 30, Easing.easeIn);
 * epochBus.on('tick', () => ft.step());
 * ```
 */
export class TweenManager {
  private static _nextId = 0;
  private static _tweens: Map<number, TweenEntry> = new Map();

  // ── Core update ────────────────────────────────────────────────────────────

  /**
   * Advance all active tweens.
   * @param dt Delta time in seconds (e.g. Ticker.deltaMS / 1000).
   */
  static update(dt: number): void {
    for (const [id, tw] of TweenManager._tweens) {
      if (!tw.active) {
        TweenManager._tweens.delete(id);
        continue;
      }
      tw.elapsed += dt;
      const rawT = Math.min(tw.elapsed / tw.duration, 1);
      const easedT = tw.easing(rawT);
      tw.onUpdate(easedT);
      if (rawT >= 1) {
        tw.active = false;
        TweenManager._tweens.delete(id);
        tw.onComplete?.();
      }
    }
  }

  // ── Object property tween ─────────────────────────────────────────────────

  /**
   * Animate numeric properties on `target` towards `props` over `duration` seconds.
   *
   * @param target   Any object with numeric properties (e.g. `sprite.position`)
   * @param props    Map of property name → target value
   * @param duration Seconds
   * @param easing   Easing function (default: `Easing.linear`)
   */
  static tween<T extends Record<string, number>>(
    target: T,
    props: Partial<T>,
    duration: number,
    easing: EasingFn = Easing.linear,
  ): TweenHandle {
    // snapshot starting values
    const keys = Object.keys(props) as (keyof T)[];
    const startVals: Partial<T> = {} as Partial<T>;
    for (const k of keys) {
      startVals[k] = target[k];
    }

    let resolveFn: () => void;
    const promise = new Promise<void>((res) => { resolveFn = res; });

    const id = ++TweenManager._nextId;
    const entry: TweenEntry = {
      id,
      elapsed: 0,
      duration: Math.max(duration, 0.0001),
      easing,
      onUpdate: (t) => {
        for (const k of keys) {
          const start = startVals[k] as number;
          const end = props[k] as number;
          (target as Record<string, number>)[k as string] = start + (end - start) * t;
        }
      },
      onComplete: () => resolveFn(),
      active: true,
    };
    TweenManager._tweens.set(id, entry);

    return {
      cancel() {
        entry.active = false;
        resolveFn();
      },
      promise,
    };
  }

  // ── MathTween — standalone number ─────────────────────────────────────────

  /**
   * Animate a standalone number from `from` to `to` over `duration` seconds.
   * Access the current value via `handle.value`.
   *
   * Useful for animating scalar uniforms (e.g. bloom intensity, opacity).
   */
  static mathTween(
    from: number,
    to: number,
    duration: number,
    easing: EasingFn = Easing.linear,
    onUpdate?: (value: number) => void,
  ): MathTweenHandle {
    let _value = from;

    let resolveFn: () => void;
    const promise = new Promise<void>((res) => { resolveFn = res; });

    const id = ++TweenManager._nextId;
    const entry: TweenEntry = {
      id,
      elapsed: 0,
      duration: Math.max(duration, 0.0001),
      easing,
      onUpdate: (t) => {
        _value = from + (to - from) * t;
        onUpdate?.(_value);
      },
      onComplete: () => resolveFn(),
      active: true,
    };
    TweenManager._tweens.set(id, entry);

    const handle: MathTweenHandle = {
      get value() { return _value; },
      cancel() {
        entry.active = false;
        resolveFn();
      },
      promise,
    };
    return handle;
  }

  // ── FrameTween — epoch-stepped ────────────────────────────────────────────

  /**
   * Create a tween that advances by `step()` calls rather than wall-clock time.
   * Each `step()` advances one frame; designed for epoch-controlled playback.
   *
   * @param target      Object to animate
   * @param props       Target property values
   * @param totalFrames Total number of steps to reach target
   * @param easing      Easing function
   */
  static frameTween<T extends Record<string, number>>(
    target: T,
    props: Partial<T>,
    totalFrames: number,
    easing: EasingFn = Easing.linear,
    onComplete?: () => void,
  ): FrameTweenHandle {
    const keys = Object.keys(props) as (keyof T)[];
    const startVals: Partial<T> = {} as Partial<T>;
    for (const k of keys) {
      startVals[k] = target[k];
    }

    let _frame = 0;
    let _done = false;
    const frames = Math.max(totalFrames, 1);

    const handle: FrameTweenHandle = {
      step() {
        if (_done) return;
        _frame = Math.min(_frame + 1, frames);
        const rawT = _frame / frames;
        const easedT = easing(rawT);
        for (const k of keys) {
          const start = startVals[k] as number;
          const end = props[k] as number;
          (target as Record<string, number>)[k as string] = start + (end - start) * easedT;
        }
        if (_frame >= frames) {
          _done = true;
          onComplete?.();
        }
      },
      cancel() { _done = true; },
      get done() { return _done; },
      get frame() { return _frame; },
      get totalFrames() { return frames; },
    };
    return handle;
  }

  // ── Utility ───────────────────────────────────────────────────────────────

  /** Cancel all active tweens. Useful on scene teardown. */
  static cancelAll(): void {
    for (const tw of TweenManager._tweens.values()) {
      tw.active = false;
    }
    TweenManager._tweens.clear();
  }

  /** Number of currently active tweens (diagnostic). */
  static get count(): number {
    return TweenManager._tweens.size;
  }
}

// ── VelocityTracker ───────────────────────────────────────────────────────────

export interface Velocity {
  /** pixels per second, x axis */
  vx: number;
  /** pixels per second, y axis */
  vy: number;
  /** |v| magnitude */
  speed: number;
}

interface VelocityEntry {
  x: number;
  y: number;
  /** timestamp of last sample (performance.now ms) */
  ts: number;
  /** exponentially smoothed vx */
  vx: number;
  /** exponentially smoothed vy */
  vy: number;
}

/**
 * VelocityTracker — tracks positional velocity for cells.
 *
 * Call `track(id, x, y)` every frame for each moving cell.
 * Call `getVelocity(id)` to retrieve {vx, vy, speed} for inertia / motion blur.
 *
 * Uses exponential smoothing (α = 0.3) to reduce jitter.
 *
 * ```ts
 * // In render loop:
 * VelocityTracker.track(cell.id, cell.x, cell.y);
 *
 * // When computing motion blur params:
 * const { vx, vy, speed } = VelocityTracker.getVelocity(cell.id);
 * blurFilter.strength = Math.min(speed * BLUR_SCALE, MAX_BLUR);
 * blurFilter.angle = Math.atan2(vy, vx);
 * ```
 */
export class VelocityTracker {
  /** Smoothing factor α ∈ (0, 1]. Higher = faster response, more jitter. */
  static smoothing = 0.3;
  /** Minimum milliseconds between samples to avoid division-by-zero. */
  static minDt = 1;

  private static _entries: Map<string, VelocityEntry> = new Map();

  /**
   * Record the current position for the given tracked id.
   * Should be called once per render frame.
   */
  static track(id: string, x: number, y: number): void {
    const now = performance.now();
    const prev = VelocityTracker._entries.get(id);

    if (!prev) {
      VelocityTracker._entries.set(id, { x, y, ts: now, vx: 0, vy: 0 });
      return;
    }

    const dt = Math.max(now - prev.ts, VelocityTracker.minDt);
    const rawVx = ((x - prev.x) / dt) * 1000; // px/s
    const rawVy = ((y - prev.y) / dt) * 1000;

    const α = VelocityTracker.smoothing;
    prev.vx = prev.vx + α * (rawVx - prev.vx);
    prev.vy = prev.vy + α * (rawVy - prev.vy);
    prev.x = x;
    prev.y = y;
    prev.ts = now;
  }

  /**
   * Return the smoothed velocity for `id`.
   * Returns zero velocity if id has never been tracked.
   */
  static getVelocity(id: string): Velocity {
    const e = VelocityTracker._entries.get(id);
    if (!e) return { vx: 0, vy: 0, speed: 0 };
    const speed = Math.sqrt(e.vx * e.vx + e.vy * e.vy);
    return { vx: e.vx, vy: e.vy, speed };
  }

  /**
   * Remove a tracked entry (e.g. when a cell is destroyed).
   */
  static remove(id: string): void {
    VelocityTracker._entries.delete(id);
  }

  /**
   * Clear all tracked entries (e.g. on scene reset).
   */
  static clear(): void {
    VelocityTracker._entries.clear();
  }

  /** Number of currently tracked objects (diagnostic). */
  static get count(): number {
    return VelocityTracker._entries.size;
  }
}

// ── SplineInterpolation ───────────────────────────────────────────────────────

export interface Vec2 {
  x: number;
  y: number;
}

/**
 * SplineInterpolation — Catmull-Rom cubic spline for smooth paths.
 *
 * Catmull-Rom splines pass through all control points, making them ideal for:
 *   - Cell movement path smoothing (avoids abrupt direction changes)
 *   - Edge routing curve beautification (organic-looking bezier guides)
 *   - Skeleton animation key-frame interpolation
 *
 * ```ts
 * const path: Vec2[] = [
 *   { x: 0, y: 0 }, { x: 100, y: 50 },
 *   { x: 200, y: 20 }, { x: 300, y: 80 },
 * ];
 *
 * // Get position at 30% along the full spline
 * const pos = SplineInterpolation.catmullRom(path, 0.3);
 *
 * // Sample 64 points for rendering a smooth SVG path
 * const pts = SplineInterpolation.sample(path, 64);
 * ```
 */
export class SplineInterpolation {
  /**
   * Catmull-Rom tension parameter α.
   * 0.5 = centripetal (avoids cusps), 0.0 = uniform, 1.0 = chordal.
   */
  static alpha = 0.5;

  // ── Single-segment evaluation ──────────────────────────────────────────────

  /**
   * Evaluate one Catmull-Rom segment defined by four control points.
   * @param p0  Point before segment start
   * @param p1  Segment start (t=0)
   * @param p2  Segment end   (t=1)
   * @param p3  Point after segment end
   * @param t   Parameter ∈ [0, 1]
   */
  static segment(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: number): Vec2 {
    // Centripetal parameterisation knot distances
    const getKnot = (a: Vec2, b: Vec2): number => {
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.sqrt(Math.sqrt(dx * dx + dy * dy));
      return d === 0 ? 0.0001 : d;
    };

    const α = SplineInterpolation.alpha;
    const t0 = 0;
    const t1 = t0 + Math.pow(getKnot(p0, p1), α);
    const t2 = t1 + Math.pow(getKnot(p1, p2), α);
    const t3 = t2 + Math.pow(getKnot(p2, p3), α);

    const tParam = t1 + t * (t2 - t1);

    // De Casteljau interpolation on knot parameterisation
    const lerp = (a: Vec2, b: Vec2, t0: number, t1: number, tv: number): Vec2 => {
      const dt = t1 - t0;
      if (Math.abs(dt) < 1e-10) return { x: a.x, y: a.y };
      const s = (tv - t0) / dt;
      return { x: a.x + (b.x - a.x) * s, y: a.y + (b.y - a.y) * s };
    };

    const a1 = lerp(p0, p1, t0, t1, tParam);
    const a2 = lerp(p1, p2, t1, t2, tParam);
    const a3 = lerp(p2, p3, t2, t3, tParam);

    const b1 = lerp(a1, a2, t0, t2, tParam);
    const b2 = lerp(a2, a3, t1, t3, tParam);

    return lerp(b1, b2, t0, t2, tParam); // NOTE: intentional — t1..t2 window
  }

  // ── Full spline evaluation ─────────────────────────────────────────────────

  /**
   * Evaluate the full Catmull-Rom spline through `points` at global parameter
   * `t ∈ [0, 1]`.
   *
   * For n control points there are (n-1) segments (with phantom end-points
   * reflected from the first and last interior points).
   *
   * Requires at least 2 points; fewer returns the first point.
   *
   * @param points  Array of Vec2 control points (passed through exactly)
   * @param t       Global parameter ∈ [0, 1]
   */
  static catmullRom(points: Vec2[], t: number): Vec2 {
    if (points.length === 0) return { x: 0, y: 0 };
    if (points.length === 1) return { ...points[0] };
    if (points.length === 2) {
      return {
        x: points[0].x + (points[1].x - points[0].x) * t,
        y: points[0].y + (points[1].y - points[0].y) * t,
      };
    }

    // Clamp and map t to segment index
    const clamped = Math.max(0, Math.min(1, t));
    const n = points.length;
    const segments = n - 1;
    const scaled = clamped * segments;
    const segIdx = Math.min(Math.floor(scaled), segments - 1);
    const localT = scaled - segIdx;

    // Phantom control points: reflect first/last interior segment
    const p1 = points[segIdx];
    const p2 = points[segIdx + 1];
    const p0 = segIdx > 0 ? points[segIdx - 1] : {
      x: 2 * p1.x - p2.x,
      y: 2 * p1.y - p2.y,
    };
    const p3 = segIdx + 2 < n ? points[segIdx + 2] : {
      x: 2 * p2.x - p1.x,
      y: 2 * p2.y - p1.y,
    };

    return SplineInterpolation.segment(p0, p1, p2, p3, localT);
  }

  // ── Utility: sample N evenly-spaced points ────────────────────────────────

  /**
   * Sample `count` evenly-spaced points along the spline.
   * Useful for converting the spline to a polyline for SVG `<path d="...">`.
   *
   * @param points  Control points
   * @param count   Number of samples (≥ 2)
   */
  static sample(points: Vec2[], count: number): Vec2[] {
    const n = Math.max(count, 2);
    const result: Vec2[] = [];
    for (let i = 0; i < n; i++) {
      result.push(SplineInterpolation.catmullRom(points, i / (n - 1)));
    }
    return result;
  }

  /**
   * Approximate the arc-length of the spline using `samples` line segments.
   * Useful for computing uniform-speed parameterisation.
   */
  static arcLength(points: Vec2[], samples = 100): number {
    if (points.length < 2) return 0;
    const pts = SplineInterpolation.sample(points, samples);
    let len = 0;
    for (let i = 1; i < pts.length; i++) {
      const dx = pts[i].x - pts[i - 1].x;
      const dy = pts[i].y - pts[i - 1].y;
      len += Math.sqrt(dx * dx + dy * dy);
    }
    return len;
  }

  /**
   * Build an arc-length reparameterisation LUT so that `t` maps uniformly to
   * distance along the curve, eliminating speed variation between segments.
   *
   * Returns a function `(t: number) => Vec2`.
   *
   * @param points  Control points
   * @param lutSize Number of LUT samples (higher = more accurate, default 200)
   */
  static uniformSpeed(
    points: Vec2[],
    lutSize = 200,
  ): (t: number) => Vec2 {
    if (points.length < 2) return () => ({ ...points[0] } ?? { x: 0, y: 0 });

    // Build cumulative length LUT
    const lut: Array<{ t: number; len: number }> = [{ t: 0, len: 0 }];
    let cumLen = 0;
    let prev = SplineInterpolation.catmullRom(points, 0);

    for (let i = 1; i <= lutSize; i++) {
      const rawT = i / lutSize;
      const cur = SplineInterpolation.catmullRom(points, rawT);
      const dx = cur.x - prev.x;
      const dy = cur.y - prev.y;
      cumLen += Math.sqrt(dx * dx + dy * dy);
      lut.push({ t: rawT, len: cumLen });
      prev = cur;
    }

    const totalLen = cumLen;

    return (t: number): Vec2 => {
      const targetLen = Math.max(0, Math.min(1, t)) * totalLen;
      // Binary search LUT
      let lo = 0;
      let hi = lut.length - 1;
      while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        if (lut[mid].len < targetLen) lo = mid;
        else hi = mid;
      }
      const a = lut[lo];
      const b = lut[hi];
      const dLen = b.len - a.len;
      const localT = dLen < 1e-10 ? a.t : a.t + ((targetLen - a.len) / dLen) * (b.t - a.t);
      return SplineInterpolation.catmullRom(points, localT);
    };
  }
}
