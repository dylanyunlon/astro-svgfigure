/**
 * draco-geometry-loader.ts — M945: Draco .bin → GPU vertex buffer
 * ─────────────────────────────────────────────────────────────────────────────
 * Loads ActiveTheory Draco-compressed .bin geometry files, decodes them via
 * the draco3d WASM decoder, and uploads the resulting attribute data into
 * WebGL vertex buffer objects (VBOs).
 *
 * AT binary format (custom, no glTF/GLB wrapper):
 *   [2 bytes: ASCII decimal string for JSON length, e.g. "83"]
 *   [8 bytes: zero padding]
 *   [N bytes: UTF-8 JSON header, e.g. {"name":"hexagon_gem","type":0,"attributes":[...]}]
 *   [M bytes: raw Draco-encoded mesh buffer]
 *
 * Example header JSON:
 *   { "name": "hexagon_gem", "type": 0,
 *     "attributes": [["position",7],["normal",7],["uv",7]] }
 *
 * Output VBO layout — interleaved, tightly-packed floats, 32 bytes/vertex:
 *   offset  0: position.xyz  (3 × float32 = 12 bytes)
 *   offset 12: normal.xyz    (3 × float32 = 12 bytes)
 *   offset 24: uv.xy         (2 × float32 =  8 bytes)
 *   stride: 32 bytes
 *
 * Usage (browser, WebGL2):
 *   const loader = new DracoGeometryLoader(gl);
 *   const result = await loader.loadFromURL('/geometry/hexagon_gem.bin');
 *   // result.vbo      — WebGLBuffer (ARRAY_BUFFER, interleaved)
 *   // result.ibo      — WebGLBuffer (ELEMENT_ARRAY_BUFFER, Uint32)
 *   // result.vertexCount
 *   // result.indexCount
 *   // result.stride   — 32 (bytes)
 *
 * Node.js / test usage (no WebGL context, pass gl = null):
 *   const loader = new DracoGeometryLoader(null);
 *   const result = await loader.loadFromURL('file:///path/to/hexagon_gem.bin');
 *   // result.vbo === null, but result.interleavedData / result.positions etc. present
 *
 * All AT geometry names (17 files):
 *   AT_logo, cables, chainlink, cube, flower_spine-128, forest-128,
 *   hexagon_gem, jellyfish, mask, pillars, rock_L, rock_R,
 *   rocky_soil, sand, spine, structure, walls
 *
 * Research: xiaodi #M945 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** AT binary file header decoded from the first section of a .bin file. */








export interface ATBinHeader {
  /** Mesh name, e.g. "hexagon_gem". */
  name: string;
  /**
   * Geometry type flag from the AT pipeline.
   * 0 = triangular mesh (only type currently shipped).
   */
  type: number;
  /**
   * Attribute descriptors: [name, componentType] pairs.
   * componentType 7 = DT_FLOAT32 in Draco's DataType enum.
   */
  attributes: [string, number][];
}

/**
 * Decoded CPU-side geometry data.
 * All arrays are densely-packed with the component counts indicated.
 */
export interface DecodedGeometry {
  /** Flat xyz array — length = vertexCount × 3. */
  positions: Float32Array;
  /** Flat xyz array — length = vertexCount × 3.  Zero-filled if absent. */
  normals: Float32Array;
  /** Flat uv array  — length = vertexCount × 2.  Zero-filled if absent. */
  uvs: Float32Array;
  /** Triangle index list — length = faceCount × 3. */
  indices: Uint32Array;
  vertexCount: number;
  faceCount: number;
  /**
   * Interleaved vertex data ready for gl.bufferData().
   * Layout: [px py pz  nx ny nz  u v] per vertex, all float32.
   * stride = 32 bytes.
   */
  interleavedData: Float32Array;
  /** ATBinHeader parsed from the file. */
  header: ATBinHeader;
}

/**
 * GPU-resident geometry handle returned by loadFromURL() when a valid
 * WebGL2RenderingContext is supplied.  When gl = null the vbo/ibo fields
 * are null but all CPU arrays remain populated.
 */
export interface GPUGeometryHandle {
  /** WebGL ARRAY_BUFFER containing interleaved pos/normal/uv data. */
  vbo: WebGLBuffer | null;
  /** WebGL ELEMENT_ARRAY_BUFFER containing Uint32 triangle indices. */
  ibo: WebGLBuffer | null;
  vertexCount: number;
  indexCount: number;
  /**
   * Vertex stride in bytes.
   * Always 32: (3 pos + 3 normal + 2 uv) × 4 bytes/float.
   */
  stride: number;
  /**
   * Byte offsets within each vertex for use with gl.vertexAttribPointer().
   * offsetPosition = 0, offsetNormal = 12, offsetUV = 24.
   */
  offsets: { position: number; normal: number; uv: number };
  /** CPU geometry kept for CPU-side queries / debugging. */
  geometry: DecodedGeometry;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Number of floats per interleaved vertex (pos3 + normal3 + uv2). */
const FLOATS_PER_VERTEX = 8;

/** Byte stride of the interleaved VBO layout. */
const VERTEX_STRIDE = FLOATS_PER_VERTEX * Float32Array.BYTES_PER_ELEMENT; // 32

/** Byte offsets of each attribute within one interleaved vertex. */
const ATTR_OFFSETS = {
  position: 0,
  normal:   3 * Float32Array.BYTES_PER_ELEMENT,  // 12
  uv:       6 * Float32Array.BYTES_PER_ELEMENT,  // 24
} as const;

/**
 * Known AT geometry asset names (filename without .bin extension).
 * Used by loadByName() as a convenience API.
 */
export const AT_GEOMETRY_NAMES = [
  'AT_logo',
  'cables',
  'chainlink',
  'cube',
  'flower_spine-128',
  'forest-128',
  'hexagon_gem',
  'jellyfish',
  'mask',
  'pillars',
  'rock_L',
  'rock_R',
  'rocky_soil',
  'sand',
  'spine',
  'structure',
  'walls',
] as const;

export type ATGeometryAssetName = (typeof AT_GEOMETRY_NAMES)[number];

// ─────────────────────────────────────────────────────────────────────────────
// Draco WASM module singleton
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Minimal type surface for the Emscripten-generated draco3d decoder module.
 * Only the subset used by DracoGeometryLoader is typed here.
 *
 * @internal
 */
interface DracoDecoderModule {
  // Attribute type constants
  readonly POSITION: number;
  readonly NORMAL: number;
  readonly TEX_COORD: number;
  readonly COLOR: number;
  /**
   * Geometry type constants returned by Decoder.GetEncodedGeometryType().
   * draco3d WASM values: POINT_CLOUD = 0, TRIANGULAR_MESH = 1.
   */
  readonly TRIANGULAR_MESH: number;
  readonly POINT_CLOUD: number;

  Decoder: new () => DracoDecoder;
  DecoderBuffer: new () => DracoDecoderBuffer;
  Mesh: new () => DracoMesh;
  PointCloud: new () => DracoPointCloud;
  DracoFloat32Array: new () => DracoFloat32Array;
  DracoInt32Array: new () => DracoInt32Array;
  destroy(obj: unknown): void;
}

interface DracoDecoder {
  GetEncodedGeometryType(buf: DracoDecoderBuffer): number;
  DecodeBufferToMesh(buf: DracoDecoderBuffer, mesh: DracoMesh): { ok(): boolean; error_msg(): string };
  DecodeBufferToPointCloud(buf: DracoDecoderBuffer, pc: DracoPointCloud): { ok(): boolean; error_msg(): string };
  GetAttributeId(geom: DracoPointCloud, attrType: number): number;
  GetAttribute(geom: DracoPointCloud, id: number): DracoPointAttribute;
  GetAttributeFloatForAllPoints(geom: DracoPointCloud, attr: DracoPointAttribute, out: DracoFloat32Array): boolean;
  GetFaceFromMesh(mesh: DracoMesh, faceIndex: number, out: DracoInt32Array): boolean;
}

interface DracoDecoderBuffer {
  Init(data: Uint8Array, size: number): void;
}

/** Base type for both PointCloud and Mesh geometry objects. */
interface DracoPointCloud {
  num_points(): number;
}

interface DracoMesh extends DracoPointCloud {
  num_faces(): number;
}

interface DracoPointAttribute {
  num_components(): number;
}

interface DracoFloat32Array {
  size(): number;
  GetValue(i: number): number;
}

interface DracoInt32Array {
  GetValue(i: number): number;
}

// Lazy singleton for the draco3d WASM module (resolved via dynamic import / global DracoDecoderModule).
let _dracoModulePromise: Promise<DracoDecoderModule> | null = null;

/**
 * Return (and cache) the Draco decoder WASM module.
 *
 * In browsers this checks `window.DracoDecoderModule` first (injected by the
 * page) then falls back to a dynamic import of the draco_decoder.wasm bundle.
 * In Node.js it requires the `draco3d` npm package.
 */
function getDracoModule(): Promise<DracoDecoderModule> {
  if (_dracoModulePromise) return _dracoModulePromise;

  _dracoModulePromise = (async (): Promise<DracoDecoderModule> => {
    // If the host page already bootstrapped a global decoder module, reuse it.
    if (typeof window !== 'undefined') {
      const global = window as unknown as Record<string, unknown>;
      if (typeof global['DracoDecoderModule'] === 'function') {
        return (global['DracoDecoderModule'] as () => Promise<DracoDecoderModule>)();
      }
    }
    // Fall through to the npm package dynamic import (bundled via Vite).
    const draco3dPkg = await import('draco3d');
    return (draco3dPkg as unknown as { createDecoderModule: (opts: object) => Promise<DracoDecoderModule> })
      .createDecoderModule({});
  })();

  return _dracoModulePromise;
}

// ─────────────────────────────────────────────────────────────────────────────
// DracoGeometryLoader
// ─────────────────────────────────────────────────────────────────────────────

/**
 * DracoGeometryLoader
 *
 * Fetches AT Draco-compressed .bin files, decodes the mesh geometry via the
 * draco3d WASM decoder, and uploads the result into WebGL2 VBOs.
 *
 * The loader is safe to instantiate once and reuse across many load calls.
 * Results are cached by URL; call `clearCache()` to free memory.
 *
 * @example Browser / WebGL2
 * ```ts
 * const loader = new DracoGeometryLoader(gl);
 * const gem = await loader.loadFromURL('/geometry/hexagon_gem.bin');
 *
 * gl.bindBuffer(gl.ARRAY_BUFFER, gem.vbo);
 * gl.vertexAttribPointer(posLoc,    3, gl.FLOAT, false, gem.stride, gem.offsets.position);
 * gl.vertexAttribPointer(normalLoc, 3, gl.FLOAT, false, gem.stride, gem.offsets.normal);
 * gl.vertexAttribPointer(uvLoc,     2, gl.FLOAT, false, gem.stride, gem.offsets.uv);
 *
 * gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gem.ibo);
 * gl.drawElements(gl.TRIANGLES, gem.indexCount, gl.UNSIGNED_INT, 0);
 * ```
 *
 * @example Convenience — load by AT asset name
 * ```ts
 * const gem = await loader.loadByName('hexagon_gem');
 * ```
 */
export class DracoGeometryLoader {
  /**
   * WebGL2 rendering context.  Pass `null` to operate in decode-only mode
   * (no GPU upload; vbo/ibo will be null in the returned handle).
   */
  private gl: WebGL2RenderingContext | null;

  /**
   * Base URL for resolving AT asset names via `loadByName()`.
   * @default '/upstream/activetheory-assets/geometry'
   */
  private basePath: string;

  /** LRU-like in-memory cache: URL → GPUGeometryHandle. */
  private cache = new Map<string, GPUGeometryHandle>();

  constructor(
    gl: WebGL2RenderingContext | null,
    options: { basePath?: string } = {},
  ) {
    this.gl = gl;
    this.basePath = (options.basePath ?? '/upstream/activetheory-assets/geometry')
      .replace(/\/+$/, '');
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Fetch an AT .bin file from `url`, decode it, and upload to GPU.
   *
   * Results are cached by URL so repeated calls are free.
   *
   * @param url  Absolute or relative URL to a .bin Draco file.
   * @returns    GPUGeometryHandle with VBO, IBO, vertex/index counts, stride.
   */
  async loadFromURL(url: string): Promise<GPUGeometryHandle> {
    const cached = this.cache.get(url);
    if (cached) return cached;

    // 1. Fetch
    const buffer = await this._fetch(url);

    // 2. Decode (CPU)
    const geometry = await this._decode(buffer);

    // 3. Upload to GPU
    const handle = this._upload(geometry);

    this.cache.set(url, handle);
    return handle;
  }

  /**
   * Load a geometry by its AT asset name (filename without `.bin`).
   *
   * @example `await loader.loadByName('hexagon_gem')`
   */
  async loadByName(name: ATGeometryAssetName | string): Promise<GPUGeometryHandle> {
    const url = `${this.basePath}/${name}.bin`;
    return this.loadFromURL(url);
  }

  /**
   * Pre-load multiple AT assets in parallel.
   *
   * @param names  List of AT asset names (without `.bin`).
   * @returns      Map keyed by asset name.
   */
  async loadMany(
    names: ReadonlyArray<ATGeometryAssetName | string>,
  ): Promise<Map<string, GPUGeometryHandle>> {
    const entries = await Promise.all(
      names.map(async (name) => [name, await this.loadByName(name)] as const),
    );
    return new Map(entries);
  }

  /**
   * Decode a raw AT .bin ArrayBuffer without fetching.
   * Useful when the caller already has the bytes (e.g. bundled assets, tests).
   *
   * No caching is applied.
   */
  async decodeBuffer(buffer: ArrayBuffer): Promise<GPUGeometryHandle> {
    const geometry = await this._decode(buffer);
    return this._upload(geometry);
  }

  /** Clear the in-memory geometry cache. Does NOT delete GPU buffers. */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Delete all cached GPU buffers and clear the cache.
   * Only meaningful when a GL context was provided.
   */
  dispose(): void {
    if (this.gl) {
      for (const handle of this.cache.values()) {
        if (handle.vbo) this.gl.deleteBuffer(handle.vbo);
        if (handle.ibo) this.gl.deleteBuffer(handle.ibo);
      }
    }
    this.cache.clear();
  }

  // ── Private: fetch ─────────────────────────────────────────────────────────

  private async _fetch(url: string): Promise<ArrayBuffer> {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(
        `[DracoGeometryLoader] fetch failed: ${url} — HTTP ${res.status} ${res.statusText}`,
      );
    }
    return res.arrayBuffer();
  }

  // ── Private: decode ────────────────────────────────────────────────────────

  /**
   * Parse AT .bin header and decode Draco payload → DecodedGeometry.
   *
   * AT binary layout:
   *   bytes  0-1 : ASCII decimal string = JSON header byte length (e.g. "83")
   *   bytes  2-9 : zero padding (8 bytes)
   *   bytes 10.. : UTF-8 JSON header  (jsonLen bytes)
   *   bytes 10+jsonLen.. : raw Draco buffer
   */
  private async _decode(arrayBuffer: ArrayBuffer): Promise<DecodedGeometry> {
    const buf = new Uint8Array(arrayBuffer);

    // ── 1. Parse AT header ──────────────────────────────────────────────────
    // Format: [2 bytes ASCII decimal = JSON length][8 zero bytes][JSON][Draco]
    const jsonLenStr = String.fromCharCode(buf[0], buf[1]);
    const jsonLen = parseInt(jsonLenStr, 10);
    if (isNaN(jsonLen) || jsonLen <= 0 || 10 + jsonLen > buf.length) {
      throw new Error(
        `[DracoGeometryLoader] invalid AT header: jsonLen="${jsonLenStr}" (${jsonLen})`,
      );
    }
    const headerBytes = buf.slice(10, 10 + jsonLen);
    const header: ATBinHeader = JSON.parse(
      new TextDecoder().decode(headerBytes),
    );

    const dracoOffset = 10 + jsonLen;
    const dracoBytes  = buf.slice(dracoOffset);

    // ── 2. Detect geometry type and decode via draco3d ──────────────────────
    // AT header.type: 0 = TRIANGULAR_MESH, 1 = POINT_CLOUD
    // draco3d WASM GetEncodedGeometryType(): 0 = POINT_CLOUD, 1 = TRIANGULAR_MESH
    const dm = await getDracoModule();

    const decoder    = new dm.Decoder();
    const decoderBuf = new dm.DecoderBuffer();
    decoderBuf.Init(dracoBytes, dracoBytes.length);
    const geomType = decoder.GetEncodedGeometryType(decoderBuf);

    let geom: DracoPointCloud | null = null;
    let numVerts: number;
    let numFaces: number;
    let indices: Uint32Array;

    try {
      if (geomType === dm.POINT_CLOUD) {
        // ── Point cloud (e.g. flower_spine-128, forest-128) ─────────────────
        const pc = new dm.PointCloud();
        const status = decoder.DecodeBufferToPointCloud(decoderBuf, pc);
        if (!status.ok()) {
          dm.destroy(pc);
          throw new Error(
            `[DracoGeometryLoader] Draco PointCloud decode error for "${header.name}": ${status.error_msg()}`,
          );
        }
        geom     = pc;
        numVerts = pc.num_points();
        numFaces = 0;
        indices  = new Uint32Array(0); // no face topology for point clouds
      } else {
        // ── Triangular mesh (all other AT assets) ───────────────────────────
        const mesh   = new dm.Mesh();
        const status = decoder.DecodeBufferToMesh(decoderBuf, mesh);
        if (!status.ok()) {
          dm.destroy(mesh);
          throw new Error(
            `[DracoGeometryLoader] Draco Mesh decode error for "${header.name}": ${status.error_msg()}`,
          );
        }
        geom     = mesh;
        numVerts = mesh.num_points();
        numFaces = (mesh as DracoMesh).num_faces();
        indices  = this._extractIndices(dm, decoder, mesh as DracoMesh, numFaces);
      }

      // ── 3. Extract attribute arrays ───────────────────────────────────────
      const positions = this._extractFloat32(dm, decoder, geom, dm.POSITION, numVerts, 3);
      if (!positions) {
        throw new Error(
          `[DracoGeometryLoader] geometry "${header.name}" has no POSITION attribute`,
        );
      }
      const normals = this._extractFloat32(dm, decoder, geom, dm.NORMAL, numVerts, 3)
        ?? new Float32Array(numVerts * 3);
      const uvs = this._extractFloat32(dm, decoder, geom, dm.TEX_COORD, numVerts, 2)
        ?? new Float32Array(numVerts * 2);

      // ── 4. Build interleaved VBO data ─────────────────────────────────────
      // Layout: [px py pz  nx ny nz  u v] per vertex (8 × f32 = 32 bytes)
      const interleavedData = new Float32Array(numVerts * FLOATS_PER_VERTEX);
      for (let v = 0; v < numVerts; v++) {
        const base = v * FLOATS_PER_VERTEX;
        interleavedData[base + 0] = positions[v * 3 + 0];
        interleavedData[base + 1] = positions[v * 3 + 1];
        interleavedData[base + 2] = positions[v * 3 + 2];
        interleavedData[base + 3] = normals[v * 3 + 0];
        interleavedData[base + 4] = normals[v * 3 + 1];
        interleavedData[base + 5] = normals[v * 3 + 2];
        interleavedData[base + 6] = uvs[v * 2 + 0];
        interleavedData[base + 7] = uvs[v * 2 + 1];
      }

      return {
        positions,
        normals,
        uvs,
        indices,
        vertexCount: numVerts,
        faceCount: numFaces,
        interleavedData,
        header,
      };
    } finally {
      if (geom) dm.destroy(geom);
      dm.destroy(decoder);
      dm.destroy(decoderBuf);
    }
  }

  // ── Private: GPU upload ────────────────────────────────────────────────────

  /**
   * Upload decoded geometry to WebGL2 VRAM.
   * Returns null buffers when gl === null (headless / test mode).
   */
  private _upload(geometry: DecodedGeometry): GPUGeometryHandle {
    let vbo: WebGLBuffer | null = null;
    let ibo: WebGLBuffer | null = null;

    if (this.gl) {
      const gl = this.gl;

      // ── VBO: interleaved position/normal/uv ───────────────────────────────
      vbo = gl.createBuffer();
      if (!vbo) throw new Error('[DracoGeometryLoader] gl.createBuffer() returned null (VBO)');
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, geometry.interleavedData, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);

      // ── IBO: triangle indices ─────────────────────────────────────────────
      if (geometry.indices.length > 0) {
        ibo = gl.createBuffer();
        if (!ibo) throw new Error('[DracoGeometryLoader] gl.createBuffer() returned null (IBO)');
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, geometry.indices, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
      }
    }

    return {
      vbo,
      ibo,
      vertexCount: geometry.vertexCount,
      indexCount: geometry.indices.length,
      stride: VERTEX_STRIDE,
      offsets: { ...ATTR_OFFSETS },
      geometry,
    };
  }

  // ── Private: Draco helpers ─────────────────────────────────────────────────

  /** Extract a float32 attribute array, or return null if the attribute is absent. */
  private _extractFloat32(
    dm: DracoDecoderModule,
    decoder: DracoDecoder,
    mesh: DracoPointCloud,
    attrConst: number,
    numVerts: number,
    numComponents: number,
  ): Float32Array | null {
    const id = decoder.GetAttributeId(mesh, attrConst);
    if (id < 0) return null;

    const attr   = decoder.GetAttribute(mesh, id);
    const dracoArr = new dm.DracoFloat32Array();
    try {
      decoder.GetAttributeFloatForAllPoints(mesh, attr, dracoArr);
      const size = dracoArr.size();
      const out  = new Float32Array(size);
      for (let i = 0; i < size; i++) out[i] = dracoArr.GetValue(i);
      return out;
    } finally {
      dm.destroy(dracoArr);
    }
  }

  /** Extract the triangle index buffer from a decoded Draco mesh. */
  private _extractIndices(
    dm: DracoDecoderModule,
    decoder: DracoDecoder,
    mesh: DracoMesh,
    numFaces: number,
  ): Uint32Array {
    const indices   = new Uint32Array(numFaces * 3);
    const faceIdxArr = new dm.DracoInt32Array();
    try {
      for (let f = 0; f < numFaces; f++) {
        decoder.GetFaceFromMesh(mesh, f, faceIdxArr);
        indices[f * 3 + 0] = faceIdxArr.GetValue(0);
        indices[f * 3 + 1] = faceIdxArr.GetValue(1);
        indices[f * 3 + 2] = faceIdxArr.GetValue(2);
      }
    } finally {
      dm.destroy(faceIdxArr);
    }
    return indices;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module-level convenience helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Decode a single AT .bin ArrayBuffer to CPU geometry without any WebGL upload.
 * Useful in Node.js tests, service workers, and other headless environments.
 *
 * @example
 * ```ts
 * const geo = await decodeATBin(fs.readFileSync('hexagon_gem.bin').buffer);
 * console.log(geo.vertexCount, geo.faceCount);
 * ```
 */
export async function decodeATBin(buffer: ArrayBuffer): Promise<DecodedGeometry> {
  const loader = new DracoGeometryLoader(null);
  const handle = await loader.decodeBuffer(buffer);
  return handle.geometry;
}

/**
 * Re-export constants so callers can use them without importing the class.
 */
export { VERTEX_STRIDE as DRACO_VERTEX_STRIDE, ATTR_OFFSETS as DRACO_ATTR_OFFSETS };
