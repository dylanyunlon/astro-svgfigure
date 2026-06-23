/**
 * cell-transition.ts — M827: Cell transition animations
 *
 * Smooth transitions when cells change state:
 *   - Fade in/out (opacity tween)
 *   - Scale pulse on spawn/remove
 *   - Position interpolation on layout change
 *   - Color morph when species changes
 */

export type EasingFn = (t: number) => number;

export const easings = {
  linear: (t: number) => t,
  easeOut: (t: number) => 1 - (1 - t) ** 3,
  easeInOut: (t: number) => t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2,
  bounce: (t: number) => {
    const n1 = 7.5625, d1 = 2.75;
    if (t < 1/d1) return n1*t*t;
    if (t < 2/d1) return n1*(t-=1.5/d1)*t+0.75;
    if (t < 2.5/d1) return n1*(t-=2.25/d1)*t+0.9375;
    return n1*(t-=2.625/d1)*t+0.984375;
  },
};

interface TransitionState {
  cellId: string;
  startTime: number;
  duration: number;
  easing: EasingFn;
  from: { x: number; y: number; w: number; h: number; opacity: number; color: string };
  to: { x: number; y: number; w: number; h: number; opacity: number; color: string };
}

function lerpColor(a: string, b: string, t: number): string {
  const pa = parseInt(a.slice(1), 16);
  const pb = parseInt(b.slice(1), 16);
  const r = Math.round(((pa >> 16) & 0xff) * (1-t) + ((pb >> 16) & 0xff) * t);
  const g = Math.round(((pa >> 8) & 0xff) * (1-t) + ((pb >> 8) & 0xff) * t);
  const bl = Math.round((pa & 0xff) * (1-t) + (pb & 0xff) * t);
  return `#${((r << 16) | (g << 8) | bl).toString(16).padStart(6, '0')}`;
}

export class CellTransitionManager {
  private transitions: Map<string, TransitionState> = new Map();

  /** Start a transition for a cell */
  start(cellId: string,
        from: TransitionState['from'],
        to: TransitionState['to'],
        duration: number = 0.4,
        easing: EasingFn = easings.easeOut): void {
    this.transitions.set(cellId, {
      cellId, startTime: performance.now() / 1000,
      duration, easing, from, to,
    });
  }

  /** Spawn animation: scale from 0 + fade in */
  spawn(cellId: string, target: TransitionState['to']): void {
    this.start(cellId,
      { ...target, opacity: 0, w: target.w * 0.3, h: target.h * 0.3 },
      target, 0.5, easings.easeOut);
  }

  /** Remove animation: fade out + scale down */
  remove(cellId: string, current: TransitionState['from']): void {
    this.start(cellId, current,
      { ...current, opacity: 0, w: current.w * 0.3, h: current.h * 0.3 },
      0.35, easings.easeInOut);
  }

  /** Get current interpolated state for a cell, or null if no active transition */
  get(cellId: string, now?: number): {
    x: number; y: number; w: number; h: number;
    opacity: number; color: string; progress: number;
  } | null {
    const tr = this.transitions.get(cellId);
    if (!tr) return null;

    const t = ((now ?? performance.now() / 1000) - tr.startTime) / tr.duration;
    if (t >= 1.0) {
      this.transitions.delete(cellId);
      return { ...tr.to, progress: 1.0 };
    }

    const e = tr.easing(Math.max(0, t));
    return {
      x: tr.from.x + (tr.to.x - tr.from.x) * e,
      y: tr.from.y + (tr.to.y - tr.from.y) * e,
      w: tr.from.w + (tr.to.w - tr.from.w) * e,
      h: tr.from.h + (tr.to.h - tr.from.h) * e,
      opacity: tr.from.opacity + (tr.to.opacity - tr.from.opacity) * e,
      color: lerpColor(tr.from.color, tr.to.color, e),
      progress: t,
    };
  }

  /** Check if any cell is transitioning */
  get isAnimating(): boolean { return this.transitions.size > 0; }
  get activeCount(): number { return this.transitions.size; }
}
