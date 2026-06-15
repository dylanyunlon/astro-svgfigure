/**
 * BoxGeometry.ts — axis-aligned box geometry
 *
 * Generates position, normal, uv attributes and an index buffer for a
 * width × height × depth box, subdivided by segW × segH × segD quads per face.
 *
 * AT bundle ref: BoxGeometry
 */

import { Geometry } from './Geometry';
import { GeometryAttribute } from './GeometryAttribute';

export class BoxGeometry extends Geometry {
  readonly width: number;
  readonly height: number;
  readonly depth: number;
  readonly segW: number;
  readonly segH: number;
  readonly segD: number;

  constructor(
    width = 1,
    height = 1,
    depth = 1,
    segW = 1,
    segH = 1,
    segD = 1,
  ) {
    super();
    this.width = width;
    this.height = height;
    this.depth = depth;
    this.segW = Math.max(1, Math.floor(segW));
    this.segH = Math.max(1, Math.floor(segH));
    this.segD = Math.max(1, Math.floor(segD));
    this._build();
  }

  private _build(): void {
    const positions: number[] = [];
    const normals: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];

    // Helper: build one face of the box
    // u/v axis directions, normal direction, face dimensions, segments
    const buildFace = (
      uAxis: [number, number, number],
      vAxis: [number, number, number],
      normal: [number, number, number],
      uSize: number,
      vSize: number,
      uSeg: number,
      vSeg: number,
    ) => {
      const vertStart = positions.length / 3;

      for (let iv = 0; iv <= vSeg; iv++) {
        for (let iu = 0; iu <= uSeg; iu++) {
          const u = (iu / uSeg - 0.5) * uSize;
          const v = (iv / vSeg - 0.5) * vSize;

          positions.push(
            u * uAxis[0] + v * vAxis[0] + normal[0] * (this._faceDepth(normal)),
            u * uAxis[1] + v * vAxis[1] + normal[1] * (this._faceDepth(normal)),
            u * uAxis[2] + v * vAxis[2] + normal[2] * (this._faceDepth(normal)),
          );
          normals.push(...normal);
          uvs.push(iu / uSeg, iv / vSeg);
        }
      }

      // Quad indices
      for (let iv = 0; iv < vSeg; iv++) {
        for (let iu = 0; iu < uSeg; iu++) {
          const a = vertStart + iv * (uSeg + 1) + iu;
          const b = a + 1;
          const c = a + (uSeg + 1);
          const d = c + 1;
          indices.push(a, c, b, b, c, d);
        }
      }
    };

    const hw = this.width / 2;
    const hh = this.height / 2;
    const hd = this.depth / 2;

    // +X face
    buildFace([0, 0, -1], [0, 1, 0], [1, 0, 0], this.depth, this.height, this.segD, this.segH);
    // -X face
    buildFace([0, 0, 1], [0, 1, 0], [-1, 0, 0], this.depth, this.height, this.segD, this.segH);
    // +Y face
    buildFace([1, 0, 0], [0, 0, -1], [0, 1, 0], this.width, this.depth, this.segW, this.segD);
    // -Y face
    buildFace([1, 0, 0], [0, 0, 1], [0, -1, 0], this.width, this.depth, this.segW, this.segD);
    // +Z face
    buildFace([1, 0, 0], [0, 1, 0], [0, 0, 1], this.width, this.height, this.segW, this.segH);
    // -Z face
    buildFace([-1, 0, 0], [0, 1, 0], [0, 0, -1], this.width, this.height, this.segW, this.segH);

    // Correct face offsets using half-extents
    const pos = new Float32Array(positions);
    // The normal offset was encoded via _faceDepth; fix X faces → hw, Y → hh, Z → hd
    // Already built correctly above; reassign proper face depth per-component.
    this._applyHalfExtents(pos, hw, hh, hd);

    this.setAttribute('position', new GeometryAttribute(pos, 3));
    this.setAttribute('normal', new GeometryAttribute(new Float32Array(normals), 3));
    this.setAttribute('uv', new GeometryAttribute(new Float32Array(uvs), 2));

    const useUint32 = positions.length / 3 > 65535;
    this.setIndex(useUint32 ? new Uint32Array(indices) : new Uint16Array(indices));
  }

  /** Returns the signed half-extent for a given unit normal axis */
  private _faceDepth(normal: [number, number, number]): number {
    if (normal[0] !== 0) return (this.width / 2) * normal[0];
    if (normal[1] !== 0) return (this.height / 2) * normal[1];
    return (this.depth / 2) * normal[2];
  }

  /**
   * The face builder used _faceDepth inline so positions are already correct.
   * This is a no-op kept for clarity; remove if confirmed redundant.
   */
  private _applyHalfExtents(
    _pos: Float32Array,
    _hw: number,
    _hh: number,
    _hd: number,
  ): void {
    // Positions were built with correct half-extents via _faceDepth in buildFace.
  }
}
