/**
 * edge-particle-bridge.ts — M057: Edge 粒子流 ↔ route.json 桥接
 *
 * 将 channels/edge/*/route.json 贝塞尔控制点数据注入 EdgeParticleSystem，
 * 并在粒子到达 target cell 时触发 GlowFilter pulse。
 *
 * 设计意图：
 *   - 每条 edge 的 route.json points 数组 → 三次贝塞尔 P0-P3
 *   - semanticType 决定粒子颜色：data_flow = #64B5F6, skip_connection = #FFB74D
 *   - 粒子到达终点时触发 target cell GlowFilter outerStrength pulse 0→2→0
 *   - epoch 切换时新增 edge 淡入(alpha 0→0.7)，删除 edge 淡出(alpha 0.7→0)
 *
 * 对接：
 *   - EdgeParticleSystem (M041) — WebGL2 transform feedback 粒子系统
 *   - pixi-cell-renderer.ts — cell Container 上的 GlowFilter 引用
 *   - theatre-epoch-cell-bridge.ts — epoch 切换通知
 */

import type { EdgeRoute, EdgeParticleSystemConfig } from './EdgeParticleSystem';

// ── Route.json → Bezier mapping ─────────────────────────────────────────────

export interface BezierControlPoints {
  p0: { x: number; y: number };
  p1: { x: number; y: number };
  p2: { x: number; y: number };
  p3: { x: number; y: number };
}

/**
 * Convert route.json points array to cubic bezier control points.
 * points[0] = source, points[last] = target
 * If 4+ points: P0=first, P1=second, P2=second-to-last, P3=last
 * If 2 points: straight line with auto-generated control points (1/3 interpolation)
 */
export function routeToBezier(route: EdgeRoute): BezierControlPoints {
  const pts = route.points;
  if (pts.length >= 4) {
    return {
      p0: pts[0],
      p1: pts[1],
      p2: pts[pts.length - 2],
      p3: pts[pts.length - 1],
    };
  }
  if (pts.length >= 2) {
    const p0 = pts[0];
    const p3 = pts[pts.length - 1];
    return {
      p0,
      p1: { x: p0.x + (p3.x - p0.x) / 3, y: p0.y + (p3.y - p0.y) / 3 },
      p2: { x: p0.x + (p3.x - p0.x) * 2 / 3, y: p0.y + (p3.y - p0.y) * 2 / 3 },
      p3,
    };
  }
  // Degenerate: single point
  const p = pts[0] || { x: 0, y: 0 };
  return { p0: p, p1: p, p2: p, p3: p };
}

// ── Semantic type → color ───────────────────────────────────────────────────

const EDGE_COLORS: Record<string, [number, number, number]> = {
  data_flow:       [0.39, 0.71, 0.96],   // #64B5F6
  skip_connection: [1.00, 0.72, 0.30],   // #FFB74D
  residual:        [0.56, 0.93, 0.56],   // #90EE90
  attention:       [0.91, 0.51, 0.98],   // #E882FA
};

export function edgeColor(semanticType?: string): [number, number, number] {
  return EDGE_COLORS[semanticType || 'data_flow'] || EDGE_COLORS.data_flow;
}

// ── Glow pulse trigger ──────────────────────────────────────────────────────

export interface GlowPulseTarget {
  /** PixiJS GlowFilter instance on the target cell Container */
  glowFilter: { outerStrength: number };
  /** Base glow strength to return to after pulse */
  baseStrength: number;
}

const _activePulses = new Map<string, { target: GlowPulseTarget; startTime: number }>();

/**
 * Trigger a glow pulse on target cell when particle arrives.
 * outerStrength: baseStrength → baseStrength+2 → baseStrength over 300ms
 */
export function triggerGlowPulse(cellId: string, target: GlowPulseTarget): void {
  _activePulses.set(cellId, { target, startTime: performance.now() });
}

/**
 * Update active glow pulses. Call each frame from ticker.
 */
export function updateGlowPulses(): void {
  const now = performance.now();
  const DURATION = 300; // ms

  for (const [cellId, pulse] of _activePulses) {
    const elapsed = now - pulse.startTime;
    if (elapsed >= DURATION) {
      pulse.target.glowFilter.outerStrength = pulse.target.baseStrength;
      _activePulses.delete(cellId);
    } else {
      // Triangle wave: 0→1→0 over DURATION
      const t = elapsed / DURATION;
      const intensity = t < 0.5 ? t * 2 : 2 - t * 2;
      pulse.target.glowFilter.outerStrength = pulse.target.baseStrength + intensity * 2;
    }
  }
}

// ── Epoch transition ────────────────────────────────────────────────────────

export interface EdgeTransition {
  added: string[];    // edge_ids appearing in new epoch
  removed: string[];  // edge_ids disappearing
}

/**
 * Compute edge diff between two epoch snapshots.
 */
export function computeEdgeTransition(
  prevEdgeIds: Set<string>,
  nextEdgeIds: Set<string>,
): EdgeTransition {
  const added = [...nextEdgeIds].filter(id => !prevEdgeIds.has(id));
  const removed = [...prevEdgeIds].filter(id => !nextEdgeIds.has(id));
  return { added, removed };
}
