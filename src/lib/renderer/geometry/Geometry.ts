/**
 * Geometry.ts — base geometry container
 *
 * Holds named vertex attributes (position, normal, uv, …) and an optional
 * index buffer.  Mirrors AT's Geometry class (AT bundle ref: 14 classes).
 *
 * The renderer iterates attributes, uploads TypedArrays to GPU buffers,
 * and issues draw calls honouring drawRange.
 */

import { GeometryAttribute } from './GeometryAttribute';

export interface BoundingBox {
  min: [number, number, number];
  max: [number, number, number];
}

export interface DrawRange {
  start: number;
  count: number;
}

export class Geometry {
  /** Named vertex attributes keyed by GLSL attribute name (e.g. "position") */
  attributes: Map<string, GeometryAttribute> = new Map();

  /** Optional element index buffer (Uint16Array or Uint32Array) */
  index: Uint16Array | Uint32Array | null = null;

  /** Explicit vertex count; derived from position attribute if not set */
  vertexCount: number = 0;

  /**
   * Limits which elements/vertices are drawn.
   * start: first index/vertex; count: how many (-1 = all).
   */
  drawRange: DrawRange = { start: 0, count: -1 };

  /** Cached AABB; null until computeBoundingBox() is called */
  boundingBox: BoundingBox | null = null;

  // ── Attribute management ─────────────────────────────────────────────────

  setAttribute(name: string, attribute: GeometryAttribute): this {
    this.attributes.set(name, attribute);
    // Keep vertexCount in sync with the position attribute
    if (name === 'position') {
      this.vertexCount = attribute.count;
    }
    return this;
  }

  getAttribute(name: string): GeometryAttribute | undefined {
    return this.attributes.get(name);
  }

  deleteAttribute(name: string): this {
    this.attributes.delete(name);
    return this;
  }

  hasAttribute(name: string): boolean {
    return this.attributes.has(name);
  }

  // ── Index buffer ─────────────────────────────────────────────────────────

  setIndex(index: Uint16Array | Uint32Array | null): this {
    this.index = index;
    return this;
  }

  // ── Bounding box ─────────────────────────────────────────────────────────

  computeBoundingBox(): BoundingBox {
    const pos = this.getAttribute('position');
    if (!pos) {
      this.boundingBox = { min: [0, 0, 0], max: [0, 0, 0] };
      return this.boundingBox;
    }

    const arr = pos.array as Float32Array;
    const stride = pos.itemSize; // tightly packed assumed
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i < arr.length; i += stride) {
      const x = arr[i];
      const y = arr[i + 1];
      const z = stride >= 3 ? arr[i + 2] : 0;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }

    this.boundingBox = {
      min: [minX, minY, minZ],
      max: [maxX, maxY, maxZ],
    };
    return this.boundingBox;
  }

  // ── Helpers ──────────────────────────────────────────────────────────────

  /** Number of primitives when using index buffer */
  get indexCount(): number {
    return this.index ? this.index.length : 0;
  }

  /** Mark all attributes as needing GPU re-upload */
  markNeedsUpdate(): void {
    for (const attr of this.attributes.values()) {
      attr.needsUpdate = true;
    }
  }

  dispose(): void {
    this.attributes.clear();
    this.index = null;
    this.boundingBox = null;
  }
}
