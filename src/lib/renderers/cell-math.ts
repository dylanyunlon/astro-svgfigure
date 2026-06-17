// ═══════════════════════════════════════════════════════════════════════════════
// M014: Cell Math — PixiJS Matrix/Point/Rectangle for cell bbox transforms
//
// Mirrors upstream/pixijs-engine/src/maths/ architecture.
// Provides transform utilities for cell coordinate space conversions:
//   - World ↔ Screen coordinate mapping
//   - Cell bbox intersection/containment tests
//   - Matrix composition for nested cell groups
//
// Like activetheory.net: all transforms use PixiJS native math primitives,
// no manual matrix arithmetic.
// ═══════════════════════════════════════════════════════════════════════════════

import { Matrix, Point, Rectangle, type Container } from 'pixi.js';

// ── Cell bbox type (from params.json) ───────────────────────────────────────

export interface CellBBox {
  x: number;
  y: number;
  w: number;
  h: number;
  z?: number;
  rotation?: number;
}

// ── Coordinate transforms ───────────────────────────────────────────────────

/**
 * Build a PixiJS Matrix from cell bbox params.
 * Applies translation, rotation, and optional scale.
 */
export function bboxToMatrix(bbox: CellBBox, scale = 1.0): Matrix {
  const mat = new Matrix();
  const cx = bbox.x + bbox.w / 2;
  const cy = bbox.y + bbox.h / 2;

  mat.translate(-cx, -cy);
  if (bbox.rotation) {
    mat.rotate(bbox.rotation);
  }
  mat.scale(scale, scale);
  mat.translate(cx, cy);

  return mat;
}

/**
 * Convert cell-local coordinates to world coordinates.
 */
export function cellToWorld(
  localPoint: { x: number; y: number },
  bbox: CellBBox,
): Point {
  const mat = bboxToMatrix(bbox);
  return mat.apply(new Point(localPoint.x + bbox.x, localPoint.y + bbox.y));
}

/**
 * Convert world coordinates to cell-local coordinates.
 */
export function worldToCell(
  worldPoint: { x: number; y: number },
  bbox: CellBBox,
): Point {
  const mat = bboxToMatrix(bbox);
  mat.invert();
  return mat.apply(new Point(worldPoint.x, worldPoint.y));
}

// ── Bbox operations ─────────────────────────────────────────────────────────

/**
 * Create a PixiJS Rectangle from cell bbox.
 */
export function bboxToRect(bbox: CellBBox): Rectangle {
  return new Rectangle(bbox.x, bbox.y, bbox.w, bbox.h);
}

/**
 * Test if two cell bboxes overlap.
 */
export function bboxIntersects(a: CellBBox, b: CellBBox): boolean {
  const ra = bboxToRect(a);
  const rb = bboxToRect(b);

  return !(
    ra.x + ra.width < rb.x ||
    rb.x + rb.width < ra.x ||
    ra.y + ra.height < rb.y ||
    rb.y + rb.height < ra.y
  );
}

/**
 * Test if point is inside cell bbox.
 */
export function bboxContainsPoint(bbox: CellBBox, point: { x: number; y: number }): boolean {
  return bboxToRect(bbox).contains(point.x, point.y);
}

/**
 * Compute the union bbox of multiple cells.
 */
export function bboxUnion(bboxes: CellBBox[]): CellBBox {
  if (bboxes.length === 0) return { x: 0, y: 0, w: 0, h: 0 };

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

  for (const b of bboxes) {
    minX = Math.min(minX, b.x);
    minY = Math.min(minY, b.y);
    maxX = Math.max(maxX, b.x + b.w);
    maxY = Math.max(maxY, b.y + b.h);
  }

  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

/**
 * Compute centre point of a cell bbox.
 */
export function bboxCentre(bbox: CellBBox): Point {
  return new Point(bbox.x + bbox.w / 2, bbox.y + bbox.h / 2);
}

/**
 * Apply a PixiJS Container's world transform to a cell bbox.
 * Used when cells are nested inside group containers.
 */
export function transformBBox(bbox: CellBBox, container: Container): CellBBox {
  const wt = container.worldTransform;
  const topLeft = wt.apply(new Point(bbox.x, bbox.y));
  const bottomRight = wt.apply(new Point(bbox.x + bbox.w, bbox.y + bbox.h));

  return {
    x: Math.min(topLeft.x, bottomRight.x),
    y: Math.min(topLeft.y, bottomRight.y),
    w: Math.abs(bottomRight.x - topLeft.x),
    h: Math.abs(bottomRight.y - topLeft.y),
    z: bbox.z,
    rotation: bbox.rotation,
  };
}

/**
 * Lerp between two bboxes (for epoch transition animation).
 */
export function bboxLerp(a: CellBBox, b: CellBBox, t: number): CellBBox {
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    w: a.w + (b.w - a.w) * t,
    h: a.h + (b.h - a.h) * t,
    z: (a.z ?? 0) + ((b.z ?? 0) - (a.z ?? 0)) * t,
    rotation: (a.rotation ?? 0) + ((b.rotation ?? 0) - (a.rotation ?? 0)) * t,
  };
}
