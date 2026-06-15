/**
 * PlaneGeometry.ts — XZ-plane quad geometry
 *
 * Generates a width × height planar mesh subdivided into segW × segH quads.
 * Normal points +Y.  UV origin (0,0) is at the top-left corner.
 *
 * AT bundle ref: PlaneGeometry
 */

import { Geometry } from './Geometry';
import { GeometryAttribute } from './GeometryAttribute';

export class PlaneGeometry extends Geometry {
  readonly width: number;
  readonly height: number;
  readonly segW: number;
  readonly segH: number;

  constructor(width = 1, height = 1, segW = 1, segH = 1) {
    super();
    this.width = width;
    this.height = height;
    this.segW = Math.max(1, Math.floor(segW));
    this.segH = Math.max(1, Math.floor(segH));
    this._build();
  }

  private _build(): void {
    const cols = this.segW + 1;
    const rows = this.segH + 1;
    const vertCount = cols * rows;

    const positions = new Float32Array(vertCount * 3);
    const normals   = new Float32Array(vertCount * 3);
    const uvs       = new Float32Array(vertCount * 2);

    const hw = this.width / 2;
    const hh = this.height / 2;

    let vi = 0, ni = 0, ui = 0;

    for (let row = 0; row < rows; row++) {
      const t = row / this.segH;
      const z = (t - 0.5) * this.height; // -hh … +hh along Z

      for (let col = 0; col < cols; col++) {
        const s = col / this.segW;
        const x = (s - 0.5) * this.width; // -hw … +hw along X

        positions[vi++] = x;
        positions[vi++] = 0;
        positions[vi++] = z;

        normals[ni++] = 0;
        normals[ni++] = 1;
        normals[ni++] = 0;

        uvs[ui++] = s;
        uvs[ui++] = 1 - t; // flip V so UV origin is top-left
      }
    }

    // Indices — two triangles per quad
    const quadCount = this.segW * this.segH;
    const useUint32 = vertCount > 65535;
    const indices = useUint32
      ? new Uint32Array(quadCount * 6)
      : new Uint16Array(quadCount * 6);

    let ii = 0;
    for (let row = 0; row < this.segH; row++) {
      for (let col = 0; col < this.segW; col++) {
        const a = row * cols + col;
        const b = a + 1;
        const c = a + cols;
        const d = c + 1;
        // CCW winding viewed from +Y
        indices[ii++] = a;
        indices[ii++] = c;
        indices[ii++] = b;
        indices[ii++] = b;
        indices[ii++] = c;
        indices[ii++] = d;
      }
    }

    this.setAttribute('position', new GeometryAttribute(positions, 3));
    this.setAttribute('normal',   new GeometryAttribute(normals,   3));
    this.setAttribute('uv',       new GeometryAttribute(uvs,       2));
    this.setIndex(indices);
  }
}
