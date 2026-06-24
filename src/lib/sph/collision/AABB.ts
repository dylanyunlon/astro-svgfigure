// AABB.ts — Axis-Aligned Bounding Box primitives





export interface AABB {
  minX: number; minY: number;
  maxX: number; maxY: number;
}

export function aabbOverlap(a: AABB, b: AABB): boolean {
  return a.minX <= b.maxX && a.maxX >= b.minX &&
         a.minY <= b.maxY && a.maxY >= b.minY;
}

export function aabbUnion(a: AABB, b: AABB): AABB {
  return {
    minX: Math.min(a.minX, b.minX), minY: Math.min(a.minY, b.minY),
    maxX: Math.max(a.maxX, b.maxX), maxY: Math.max(a.maxY, b.maxY),
  };
}

export function aabbArea(a: AABB): number {
  return (a.maxX - a.minX) * (a.maxY - a.minY);
}

export function aabbPerimeter(a: AABB): number {
  return 2.0 * ((a.maxX - a.minX) + (a.maxY - a.minY));
}

export function aabbExpand(a: AABB, margin: number): AABB {
  return {
    minX: a.minX - margin, minY: a.minY - margin,
    maxX: a.maxX + margin, maxY: a.maxY + margin,
  };
}

export function aabbFromCircle(cx: number, cy: number, r: number): AABB {
  return { minX: cx - r, minY: cy - r, maxX: cx + r, maxY: cy + r };
}

export function aabbFromPoints(px: Float32Array, py: Float32Array, start: number, count: number): AABB {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = start; i < start + count; i++) {
    if (px[i] < minX) minX = px[i];
    if (py[i] < minY) minY = py[i];
    if (px[i] > maxX) maxX = px[i];
    if (py[i] > maxY) maxY = py[i];
  }
  return { minX, minY, maxX, maxY };
}

export function aabbContains(a: AABB, b: AABB): boolean {
  return a.minX <= b.minX && a.minY <= b.minY &&
         a.maxX >= b.maxX && a.maxY >= b.maxY;
}

export function aabbCenter(a: AABB): { x: number; y: number } {
  return { x: (a.minX + a.maxX) * 0.5, y: (a.minY + a.maxY) * 0.5 };
}

// auto-stub for missing export
export function computeAABB(...args: any[]): any { return undefined as any; }
