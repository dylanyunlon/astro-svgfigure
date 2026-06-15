/**
 * CylinderGeometry.ts — open or closed cylinder / cone / frustum
 *
 * Builds:
 *   • lateral surface  (radialSeg quads, one height segment)
 *   • top cap          (fan of triangles around radiusTop)
 *   • bottom cap       (fan of triangles around radiusBottom)
 *
 * Setting radiusTop or radiusBottom to 0 produces a cone.
 * Caps are omitted automatically when the corresponding radius is 0.
 *
 * AT bundle ref: CylinderGeometry
 */

import { Geometry } from './Geometry';
import { GeometryAttribute } from './GeometryAttribute';

export class CylinderGeometry extends Geometry {
  readonly radiusTop: number;
  readonly radiusBottom: number;
  readonly height: number;
  readonly radialSeg: number;

  constructor(
    radiusTop    = 1,
    radiusBottom = 1,
    height       = 1,
    radialSeg    = 32,
  ) {
    super();
    this.radiusTop    = radiusTop;
    this.radiusBottom = radiusBottom;
    this.height       = height;
    this.radialSeg    = Math.max(3, Math.floor(radialSeg));
    this._build();
  }

  private _build(): void {
    const positions: number[] = [];
    const normals:   number[] = [];
    const uvs:       number[] = [];
    const indices:   number[] = [];

    // ── Lateral surface ────────────────────────────────────────────────────
    // Two rings of vertices (bottom + top) with seam duplication for UVs.
    const lateralBase = 0;

    const slope = (this.radiusBottom - this.radiusTop) / this.height;
    const normalY = Math.sin(Math.atan(slope));   // tilt of surface normal
    const normalR = Math.cos(Math.atan(slope));   // radial component

    const hh = this.height / 2;

    for (let ring = 0; ring <= 1; ring++) {
      const t       = ring;                              // 0 = bottom, 1 = top
      const y       = (t - 0.5) * this.height;          // -hh … +hh
      const radius  = this.radiusBottom + t * (this.radiusTop - this.radiusBottom);

      for (let seg = 0; seg <= this.radialSeg; seg++) {
        const theta = (seg / this.radialSeg) * 2 * Math.PI;
        const cos   = Math.cos(theta);
        const sin   = Math.sin(theta);

        positions.push(radius * cos, y, radius * sin);
        normals.push(normalR * cos, normalY, normalR * sin);
        uvs.push(seg / this.radialSeg, ring);
      }
    }

    // Quads between the two rings
    const stride = this.radialSeg + 1;
    for (let seg = 0; seg < this.radialSeg; seg++) {
      const a = lateralBase + seg;
      const b = a + 1;
      const c = a + stride;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }

    // ── Cap helper ─────────────────────────────────────────────────────────
    const buildCap = (isTop: boolean) => {
      const radius = isTop ? this.radiusTop : this.radiusBottom;
      if (radius <= 0) return;

      const y  = isTop ? hh : -hh;
      const ny = isTop ? 1 : -1;

      const centerIdx = positions.length / 3;
      positions.push(0, y, 0);
      normals.push(0, ny, 0);
      uvs.push(0.5, 0.5);

      const fanBase = positions.length / 3;
      for (let seg = 0; seg <= this.radialSeg; seg++) {
        const theta = (seg / this.radialSeg) * 2 * Math.PI;
        const cos   = Math.cos(theta);
        const sin   = Math.sin(theta);

        positions.push(radius * cos, y, radius * sin);
        normals.push(0, ny, 0);
        uvs.push(cos * 0.5 + 0.5, sin * 0.5 * ny + 0.5);
      }

      for (let seg = 0; seg < this.radialSeg; seg++) {
        const a = fanBase + seg;
        const b = fanBase + seg + 1;
        if (isTop) {
          indices.push(centerIdx, b, a);
        } else {
          indices.push(centerIdx, a, b);
        }
      }
    };

    buildCap(true);
    buildCap(false);

    // ── Pack into TypedArrays ──────────────────────────────────────────────
    const vertCount  = positions.length / 3;
    const useUint32  = vertCount > 65535;

    this.setAttribute('position', new GeometryAttribute(new Float32Array(positions), 3));
    this.setAttribute('normal',   new GeometryAttribute(new Float32Array(normals),   3));
    this.setAttribute('uv',       new GeometryAttribute(new Float32Array(uvs),       2));
    this.setIndex(useUint32 ? new Uint32Array(indices) : new Uint16Array(indices));
  }
}
