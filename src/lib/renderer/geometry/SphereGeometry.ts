/**
 * SphereGeometry.ts — UV sphere (latitude/longitude tessellation)
 *
 * Generates position, normal, uv attributes and an index buffer.
 * Normals are the normalised position vectors (true sphere normals).
 *
 * AT bundle ref: SphereGeometry
 */

import { Geometry } from './Geometry';
import { GeometryAttribute } from './GeometryAttribute';

export class SphereGeometry extends Geometry {
  readonly radius: number;
  readonly widthSegments: number;
  readonly heightSegments: number;

  constructor(radius = 1, widthSegments = 32, heightSegments = 16) {
    super();
    this.radius = radius;
    this.widthSegments  = Math.max(3, Math.floor(widthSegments));
    this.heightSegments = Math.max(2, Math.floor(heightSegments));
    this._build();
  }

  private _build(): void {
    const wSeg = this.widthSegments;
    const hSeg = this.heightSegments;

    // (wSeg + 1) * (hSeg + 1) unique vertices (seam vertices duplicated for correct UVs)
    const vertCount = (wSeg + 1) * (hSeg + 1);

    const positions = new Float32Array(vertCount * 3);
    const normals   = new Float32Array(vertCount * 3);
    const uvs       = new Float32Array(vertCount * 2);

    let vi = 0, ni = 0, ui = 0;

    for (let lat = 0; lat <= hSeg; lat++) {
      const phi = (lat / hSeg) * Math.PI;          // 0 … π  (top → bottom)
      const sinPhi = Math.sin(phi);
      const cosPhi = Math.cos(phi);

      for (let lon = 0; lon <= wSeg; lon++) {
        const theta = (lon / wSeg) * 2 * Math.PI;  // 0 … 2π

        const nx = Math.sin(phi) * Math.cos(theta);
        const ny = cosPhi;
        const nz = sinPhi * Math.sin(theta);

        positions[vi++] = nx * this.radius;
        positions[vi++] = ny * this.radius;
        positions[vi++] = nz * this.radius;

        normals[ni++] = nx;
        normals[ni++] = ny;
        normals[ni++] = nz;

        uvs[ui++] = lon / wSeg;
        uvs[ui++] = 1 - lat / hSeg;
      }
    }

    // Indices
    const quadCount = wSeg * hSeg;
    const useUint32 = vertCount > 65535;
    const indices = useUint32
      ? new Uint32Array(quadCount * 6)
      : new Uint16Array(quadCount * 6);

    let ii = 0;
    const stride = wSeg + 1;

    for (let lat = 0; lat < hSeg; lat++) {
      for (let lon = 0; lon < wSeg; lon++) {
        const a = lat * stride + lon;
        const b = a + 1;
        const c = a + stride;
        const d = c + 1;

        // Skip degenerate triangles at poles
        if (lat !== 0)        { indices[ii++] = a; indices[ii++] = c; indices[ii++] = b; }
        if (lat !== hSeg - 1) { indices[ii++] = b; indices[ii++] = c; indices[ii++] = d; }
      }
    }

    // Trim index buffer in case poles removed some tris
    const trimmedIndices = useUint32
      ? (indices as Uint32Array).slice(0, ii)
      : (indices as Uint16Array).slice(0, ii);

    this.setAttribute('position', new GeometryAttribute(positions, 3));
    this.setAttribute('normal',   new GeometryAttribute(normals,   3));
    this.setAttribute('uv',       new GeometryAttribute(uvs,       2));
    this.setIndex(trimmedIndices);
  }
}
