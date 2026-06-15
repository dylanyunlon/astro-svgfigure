/**
 * GeometryAttribute.ts — per-vertex data buffer descriptor
 *
 * Maps to AT's GeometryAttribute: wraps a TypedArray with layout metadata
 * so the renderer can bind it as a WebGL vertex attribute.
 *
 * AT bundle ref: GeometryAttribute (1 of 14 geometry classes)
 */

export type AttributeArray =
  | Float32Array
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array;

export class GeometryAttribute {
  /** Raw typed array holding the data */
  array: AttributeArray;

  /** Number of components per vertex element (e.g. 3 for vec3) */
  itemSize: number;

  /**
   * Stride in bytes between the start of consecutive vertex entries.
   * 0 = tightly packed (stride = itemSize * BYTES_PER_ELEMENT).
   */
  stride: number;

  /** Byte offset from the start of the buffer to the first element */
  offset: number;

  /** Whether integer data should be normalised to [0,1] or [-1,1] */
  normalized: boolean;

  /** Marks the attribute dirty — upload to GPU on next bind */
  needsUpdate: boolean = false;

  constructor(
    array: AttributeArray,
    itemSize: number,
    stride = 0,
    offset = 0,
    normalized = false,
  ) {
    this.array = array;
    this.itemSize = itemSize;
    this.stride = stride;
    this.offset = offset;
    this.normalized = normalized;
  }

  /** Total number of vertex elements stored */
  get count(): number {
    return this.array.length / this.itemSize;
  }

  /** Byte size of one element in the underlying array */
  get bytesPerElement(): number {
    return this.array.BYTES_PER_ELEMENT;
  }

  /** Effective stride in bytes (falls back to tightly-packed if stride === 0) */
  get effectiveStride(): number {
    return this.stride || this.itemSize * this.bytesPerElement;
  }
}
