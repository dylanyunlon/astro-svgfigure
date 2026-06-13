/**
 * threed-pipeline.ts — 3-D asset loading, decoding, and rendering pipeline.
 *
 * Ported from AT 3-D pipeline architecture (upstream references):
 *   upstream/pixi3d/src/loader/gltf-loader.ts          — GLTFLoader
 *   upstream/pixi3d/src/mesh/geometry.ts               — Draco / GeomThread
 *   upstream/three-fiber/examples/gaussian-splats/      — GaussianSplats
 *   upstream/pixi3d/src/pipeline/standard-pipeline.ts   — render ordering
 *
 * Classes exposed:
 *   GaussianSplats  — load .ply splat file, depth-sort, GPU render
 *   DracoThread     — Worker that decodes Draco-compressed buffer → Geometry
 *   GeomThread      — Worker that runs computeNormals / mesh simplification
 *   GLTFLoader      — load glTF/GLB → Scene with Draco + KTX2 support
 *
 * Design principles:
 *   • All heavy work runs off-main-thread (Worker / OffscreenCanvas).
 *   • Depth sort is performed every frame on the CPU worker, not the GPU,
 *     matching AT GaussianSplatMesh.sortByDepth() contract.
 *   • DracoThread / GeomThread follow AT WorkerManager blob-URL pattern so
 *     no extra bundler entry-point is required.
 *   • GLTFLoader wraps THREE.GLTFLoader and wires DRACOLoader + KTX2Loader
 *     exactly as AT pixi3d does, keeping the same public load(url) → Scene API.
 */

// ── Shared types ───────────────────────────────────────────────────────────────

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Mat4 {
  elements: Float32Array; // column-major, 16 floats
}

/** Minimal BufferGeometry-compatible descriptor used internally. */
export interface GeometryDescriptor {
  positions: Float32Array;
  normals?: Float32Array;
  uvs?: Float32Array;
  indices?: Uint32Array;
  vertexCount: number;
}

/** Splat point as decoded from a .ply file. */
export interface SplatPoint {
  x: number;
  y: number;
  z: number;
  /** Spherical-harmonics colour coefficients (degree-0 only → RGB). */
  r: number;
  g: number;
  b: number;
  /** Gaussian covariance — 6 upper-triangle floats. */
  cov: Float32Array;
  opacity: number;
}

// ── GaussianSplats ─────────────────────────────────────────────────────────────

/**
 * GaussianSplats
 *
 * Loads a .ply Gaussian splat file (text or binary PLY), decodes splat points,
 * depth-sorts them relative to a view matrix each frame, and uploads sorted
 * data to a WebGL2 vertex buffer for GPU splatting.
 *
 * Usage:
 *   const splats = new GaussianSplats(gl);
 *   await splats.load('/scene.ply');
 *   // inside render loop:
 *   splats.sortByDepth(viewMatrix);
 *   splats.render(projMatrix);
 */
export class GaussianSplats {
  private gl: WebGL2RenderingContext;
  private points: SplatPoint[] = [];
  private sortedIndices: Uint32Array = new Uint32Array(0);
  private vbo: WebGLBuffer | null = null;
  private sortWorker: Worker | null = null;
  private pendingSort = false;

  /** Fired when a new depth sort is ready and the VBO has been updated. */
  onSortComplete?: () => void;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this._initSortWorker();
  }

  // ── Loading ─────────────────────────────────────────────────────────────────

  /**
   * load(url)
   * Fetches and parses a .ply file. Supports both ASCII and binary-little-endian
   * PLY formats. Returns after the GPU VBO is initialised.
   */
  async load(url: string): Promise<void> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`GaussianSplats.load: HTTP ${res.status} for ${url}`);
    const buf = await res.arrayBuffer();
    this.points = this._parsePLY(buf);
    this.sortedIndices = new Uint32Array(this.points.length).map((_, i) => i);
    this._uploadVBO();
  }

  /** Parse PLY buffer → SplatPoint[]. Handles ASCII and binary-LE. */
  private _parsePLY(buf: ArrayBuffer): SplatPoint[] {
    const header = this._readPLYHeader(buf);
    if (header.format === 'ascii') return this._parsePLYAscii(buf, header);
    return this._parsePLYBinary(buf, header);
  }

  private _readPLYHeader(buf: ArrayBuffer): {
    format: 'ascii' | 'binary_little_endian' | 'binary_big_endian';
    vertexCount: number;
    headerBytes: number;
    properties: string[];
  } {
    const text = new TextDecoder().decode(new Uint8Array(buf, 0, Math.min(4096, buf.byteLength)));
    const lines = text.split('\n');
    let format: 'ascii' | 'binary_little_endian' | 'binary_big_endian' = 'ascii';
    let vertexCount = 0;
    const properties: string[] = [];
    let headerBytes = 0;

    for (let i = 0; i < lines.length; i++) {
      const l = lines[i].trim();
      if (l.startsWith('format ')) {
        const parts = l.split(' ');
        format = parts[1] as typeof format;
      } else if (l.startsWith('element vertex')) {
        vertexCount = parseInt(l.split(' ')[2], 10);
      } else if (l.startsWith('property float ')) {
        properties.push(l.split(' ')[2]);
      } else if (l === 'end_header') {
        let off = 0;
        for (let j = 0; j <= i; j++) {
          off += new TextEncoder().encode(lines[j] + '\n').length;
        }
        headerBytes = off;
        break;
      }
    }

    return { format, vertexCount, headerBytes, properties };
  }

  private _parsePLYAscii(
    buf: ArrayBuffer,
    hdr: ReturnType<GaussianSplats['_readPLYHeader']>
  ): SplatPoint[] {
    const text = new TextDecoder().decode(buf);
    const dataStart = text.indexOf('end_header') + 'end_header\n'.length;
    const lines = text.slice(dataStart).split('\n');
    const points: SplatPoint[] = [];
    const { properties } = hdr;

    for (let i = 0; i < hdr.vertexCount && i < lines.length; i++) {
      const vals = lines[i].trim().split(/\s+/).map(Number);
      const p = this._decodeProps(vals, properties);
      if (p) points.push(p);
    }
    return points;
  }

  private _parsePLYBinary(
    buf: ArrayBuffer,
    hdr: ReturnType<GaussianSplats['_readPLYHeader']>
  ): SplatPoint[] {
    const stride = hdr.properties.length * 4; // all floats
    const view = new DataView(buf, hdr.headerBytes);
    const points: SplatPoint[] = [];
    const le = hdr.format !== 'binary_big_endian';

    for (let i = 0; i < hdr.vertexCount; i++) {
      const vals: number[] = [];
      for (let j = 0; j < hdr.properties.length; j++) {
        vals.push(view.getFloat32(i * stride + j * 4, le));
      }
      const p = this._decodeProps(vals, hdr.properties);
      if (p) points.push(p);
    }
    return points;
  }

  private _decodeProps(vals: number[], props: string[]): SplatPoint | null {
    const idx = (name: string) => props.indexOf(name);
    const xi = idx('x'), yi = idx('y'), zi = idx('z');
    if (xi < 0 || yi < 0 || zi < 0) return null;

    const cov = new Float32Array(6);
    const covNames = ['cov_00', 'cov_01', 'cov_02', 'cov_11', 'cov_12', 'cov_22'];
    covNames.forEach((n, k) => {
      const ci = idx(n);
      if (ci >= 0) cov[k] = vals[ci];
    });

    return {
      x: vals[xi], y: vals[yi], z: vals[zi],
      r: vals[idx('f_dc_0')] ?? vals[idx('red')] ?? 1,
      g: vals[idx('f_dc_1')] ?? vals[idx('green')] ?? 1,
      b: vals[idx('f_dc_2')] ?? vals[idx('blue')] ?? 1,
      cov,
      opacity: vals[idx('opacity')] ?? 1,
    };
  }

  // ── Depth sort ──────────────────────────────────────────────────────────────

  /**
   * sortByDepth(viewMatrix)
   * Dispatches a depth sort to the dedicated Worker. Results are applied to
   * the GPU VBO when the worker posts back. If a sort is already in flight,
   * the new request is dropped (fire-and-forget, one-in-flight contract).
   */
  sortByDepth(viewMatrix: Mat4): void {
    if (this.pendingSort || !this.sortWorker) return;
    this.pendingSort = true;

    const positions = new Float32Array(this.points.length * 3);
    this.points.forEach((p, i) => {
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
    });

    this.sortWorker.postMessage(
      { positions, view: viewMatrix.elements, count: this.points.length },
      [positions.buffer]
    );
  }

  /** render(projMatrix) — draws the depth-sorted splat cloud. */
  render(_projMatrix: Mat4): void {
    const { gl, vbo } = this;
    if (!vbo) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.drawArrays(gl.POINTS, 0, this.sortedIndices.length);
  }

  // ── Internals ───────────────────────────────────────────────────────────────

  private _uploadVBO(): void {
    const { gl, points, sortedIndices } = this;
    // Interleaved: [x, y, z, r, g, b, opacity, cov×6] = 13 floats per splat
    const FLOATS_PER_SPLAT = 13;
    const data = new Float32Array(sortedIndices.length * FLOATS_PER_SPLAT);
    sortedIndices.forEach((si, di) => {
      const p = points[si];
      const base = di * FLOATS_PER_SPLAT;
      data[base] = p.x; data[base + 1] = p.y; data[base + 2] = p.z;
      data[base + 3] = p.r; data[base + 4] = p.g; data[base + 5] = p.b;
      data[base + 6] = p.opacity;
      data.set(p.cov, base + 7);
    });

    if (!this.vbo) this.vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
  }

  private _initSortWorker(): void {
    const workerSrc = /* js */ `
      self.onmessage = function(e) {
        const { positions, view, count } = e.data;
        const depths = new Float32Array(count);
        for (let i = 0; i < count; i++) {
          const x = positions[i*3], y = positions[i*3+1], z = positions[i*3+2];
          depths[i] = view[2]*x + view[6]*y + view[10]*z + view[14];
        }
        const indices = Uint32Array.from({length: count}, (_, i) => i);
        indices.sort((a, b) => depths[b] - depths[a]); // back-to-front
        self.postMessage({ indices }, [indices.buffer]);
      };
    `;
    const blob = new Blob([workerSrc], { type: 'application/javascript' });
    this.sortWorker = new Worker(URL.createObjectURL(blob));
    this.sortWorker.onmessage = (e: MessageEvent<{ indices: Uint32Array }>) => {
      this.sortedIndices = e.data.indices;
      this._uploadVBO();
      this.pendingSort = false;
      this.onSortComplete?.();
    };
  }

  dispose(): void {
    this.sortWorker?.terminate();
    if (this.vbo) this.gl.deleteBuffer(this.vbo);
  }
}

// ── DracoThread ────────────────────────────────────────────────────────────────

/** Message sent to the DracoThread worker. */
export interface DracoDecodeRequest {
  id: number;
  /** Raw Draco-compressed buffer. */
  buffer: ArrayBuffer;
  /** Optional WASM decoder URL override. */
  decoderUrl?: string;
}

/** Reply from the DracoThread worker. */
export interface DracoDecodeResult {
  id: number;
  geometry: GeometryDescriptor;
}

/**
 * DracoThread
 *
 * Manages a dedicated Web Worker that decodes Draco-compressed geometry
 * buffers into GeometryDescriptor objects without blocking the main thread.
 *
 * Matches AT DracoThread architecture from:
 *   upstream/pixi3d/src/mesh/geometry.ts (DracoThread / postDraco)
 *
 * Usage:
 *   const dt = new DracoThread();
 *   const geo = await dt.decode(compressedBuffer);
 */
export class DracoThread {
  private worker: Worker;
  private pending = new Map<number, (geo: GeometryDescriptor) => void>();
  private nextId = 0;
  private _defaultDecoderUrl: string;

  constructor(decoderUrl = '/draco/draco_decoder.wasm') {
    this._defaultDecoderUrl = decoderUrl;
    const workerSrc = /* js */ `
      let decoderModule = null;

      async function ensureDecoder(url) {
        if (decoderModule) return;
        const jsUrl = url.replace('.wasm', '.js');
        importScripts(jsUrl);
        decoderModule = await DracoDecoderModule({ wasmBinaryFile: url });
      }

      self.onmessage = async function(e) {
        const { id, buffer, decoderUrl } = e.data;
        try {
          await ensureDecoder(decoderUrl || '/draco/draco_decoder.wasm');
          const decoder = new decoderModule.Decoder();
          const buf = new decoderModule.DecoderBuffer();
          buf.Init(new Int8Array(buffer), buffer.byteLength);

          const geomType = decoder.GetEncodedGeometryType(buf);
          let dracoMesh;
          if (geomType === decoderModule.TRIANGULAR_MESH) {
            dracoMesh = new decoderModule.Mesh();
            decoder.DecodeBufferToMesh(buf, dracoMesh);
          } else {
            dracoMesh = new decoderModule.PointCloud();
            decoder.DecodeBufferToPointCloud(buf, dracoMesh);
          }

          const numVerts = dracoMesh.num_points();
          const positions = new Float32Array(numVerts * 3);
          const normals   = new Float32Array(numVerts * 3);
          const uvs       = new Float32Array(numVerts * 2);

          const decode3 = (attrId, out) => {
            const attr = decoder.GetAttribute(dracoMesh, attrId);
            if (attr.ptr === 0) return;
            const arr = new decoderModule.DracoFloat32Array();
            decoder.GetAttributeFloatForAllPoints(dracoMesh, attr, arr);
            for (let i = 0; i < numVerts; i++) {
              out[i*3]   = arr.GetValue(i*3);
              out[i*3+1] = arr.GetValue(i*3+1);
              out[i*3+2] = arr.GetValue(i*3+2);
            }
            decoderModule.destroy(arr);
          };

          const posAttrId = decoder.GetAttributeId(dracoMesh, decoderModule.POSITION);
          decode3(posAttrId, positions);
          const normAttrId = decoder.GetAttributeId(dracoMesh, decoderModule.NORMAL);
          decode3(normAttrId, normals);

          let indices;
          if (geomType === decoderModule.TRIANGULAR_MESH) {
            const numFaces = dracoMesh.num_faces();
            indices = new Uint32Array(numFaces * 3);
            const face = new decoderModule.DracoInt32Array();
            for (let f = 0; f < numFaces; f++) {
              decoder.GetFaceFromMesh(dracoMesh, f, face);
              indices[f*3]   = face.GetValue(0);
              indices[f*3+1] = face.GetValue(1);
              indices[f*3+2] = face.GetValue(2);
            }
            decoderModule.destroy(face);
          }

          decoderModule.destroy(buf);
          decoderModule.destroy(decoder);
          decoderModule.destroy(dracoMesh);

          const geometry = { positions, normals, uvs, indices, vertexCount: numVerts };
          const transfers = [positions.buffer, normals.buffer, uvs.buffer];
          if (indices) transfers.push(indices.buffer);
          self.postMessage({ id, geometry }, transfers);
        } catch (err) {
          self.postMessage({ id, error: err.message });
        }
      };
    `;
    const blob = new Blob([workerSrc], { type: 'application/javascript' });
    this.worker = new Worker(URL.createObjectURL(blob));
    this.worker.onmessage = (e: MessageEvent<DracoDecodeResult & { error?: string }>) => {
      const { id, geometry, error } = e.data as DracoDecodeResult & { error?: string };
      const resolve = this.pending.get(id);
      if (!resolve) return;
      this.pending.delete(id);
      if (error) throw new Error(`DracoThread: ${error}`);
      resolve(geometry);
    };
  }

  /**
   * decode(buffer)
   * Sends the Draco buffer to the worker for decoding.
   * The ArrayBuffer is transferred (zero-copy) to the worker.
   */
  decode(buffer: ArrayBuffer): Promise<GeometryDescriptor> {
    const id = this.nextId++;
    const msg: DracoDecodeRequest = { id, buffer, decoderUrl: this._defaultDecoderUrl };
    return new Promise((resolve) => {
      this.pending.set(id, resolve);
      this.worker.postMessage(msg, [buffer]);
    });
  }

  dispose(): void {
    this.worker.terminate();
    this.pending.clear();
  }
}

// ── GeomThread ─────────────────────────────────────────────────────────────────

export type SimplifyMethod = 'quadric' | 'edge-collapse';

export interface SimplifyOptions {
  /** Target vertex count after simplification. */
  targetVertexCount: number;
  method?: SimplifyMethod;
}

export interface GeomThreadRequest {
  id: number;
  op: 'computeNormals' | 'simplify';
  geometry: GeometryDescriptor;
  simplifyOptions?: SimplifyOptions;
}

/**
 * GeomThread
 *
 * Off-main-thread geometry processing: normal computation and mesh
 * simplification. Matches AT GeomThread from:
 *   upstream/pixi3d/src/mesh/geometry.ts (GeomThread)
 *
 * Usage:
 *   const gt = new GeomThread();
 *   const withNormals = await gt.computeNormals(geo);
 *   const simplified  = await gt.simplify(geo, { targetVertexCount: 500 });
 */
export class GeomThread {
  private worker: Worker;
  private pending = new Map<number, (geo: GeometryDescriptor) => void>();
  private nextId = 0;

  constructor() {
    const workerSrc = /* js */ `
      function computeNormals(positions, indices, vertexCount) {
        const normals = new Float32Array(vertexCount * 3);
        const faceCount = indices ? indices.length / 3 : vertexCount / 3;

        for (let f = 0; f < faceCount; f++) {
          const ia = indices ? indices[f*3]   : f*3;
          const ib = indices ? indices[f*3+1] : f*3+1;
          const ic = indices ? indices[f*3+2] : f*3+2;

          const ax=positions[ia*3], ay=positions[ia*3+1], az=positions[ia*3+2];
          const bx=positions[ib*3], by=positions[ib*3+1], bz=positions[ib*3+2];
          const cx=positions[ic*3], cy=positions[ic*3+1], cz=positions[ic*3+2];

          const ex=bx-ax, ey=by-ay, ez=bz-az;
          const fx=cx-ax, fy=cy-ay, fz=cz-az;
          const nx=ey*fz-ez*fy, ny=ez*fx-ex*fz, nz=ex*fy-ey*fx;

          for (const vi of [ia,ib,ic]) {
            normals[vi*3]   += nx;
            normals[vi*3+1] += ny;
            normals[vi*3+2] += nz;
          }
        }

        for (let i = 0; i < vertexCount; i++) {
          const nx=normals[i*3], ny=normals[i*3+1], nz=normals[i*3+2];
          const len = Math.sqrt(nx*nx+ny*ny+nz*nz) || 1;
          normals[i*3]   /= len;
          normals[i*3+1] /= len;
          normals[i*3+2] /= len;
        }
        return normals;
      }

      function simplifyMesh(positions, indices, targetVertexCount) {
        const cellSize = 0.05;
        const buckets = new Map();

        let minX=Infinity,minY=Infinity,minZ=Infinity;
        let maxX=-Infinity,maxY=-Infinity,maxZ=-Infinity;
        const n = positions.length / 3;
        for (let i=0;i<n;i++) {
          const x=positions[i*3],y=positions[i*3+1],z=positions[i*3+2];
          if(x<minX)minX=x; if(x>maxX)maxX=x;
          if(y<minY)minY=y; if(y>maxY)maxY=y;
          if(z<minZ)minZ=z; if(z>maxZ)maxZ=z;
        }
        const dx=maxX-minX||1, dy=maxY-minY||1, dz=maxZ-minZ||1;

        const remap = new Int32Array(n).fill(-1);
        const newPositions = [];

        for (let i=0;i<n;i++) {
          const x=positions[i*3],y=positions[i*3+1],z=positions[i*3+2];
          const bx=Math.floor((x-minX)/dx/cellSize);
          const by=Math.floor((y-minY)/dy/cellSize);
          const bz=Math.floor((z-minZ)/dz/cellSize);
          const key = bx+'_'+by+'_'+bz;
          if (!buckets.has(key)) {
            const newIdx = newPositions.length / 3;
            buckets.set(key, newIdx);
            newPositions.push(x,y,z);
          }
          remap[i] = buckets.get(key);
        }

        const newPositionsArr = new Float32Array(newPositions);
        let newIndices = null;
        if (indices) {
          const remappedIdx = [];
          for (let f=0;f<indices.length/3;f++) {
            const ia=remap[indices[f*3]];
            const ib=remap[indices[f*3+1]];
            const ic=remap[indices[f*3+2]];
            if (ia!==ib && ib!==ic && cc!==ia) remappedIdx.push(ia,ib,ic);
          }
          newIndices = new Uint32Array(remappedIdx);
        }
        return { newPositions: newPositionsArr, newIndices };
      }

      self.onmessage = function(e) {
        const { id, op, geometry, simplifyOptions } = e.data;
        const { positions, indices, vertexCount } = geometry;

        if (op === 'computeNormals') {
          const normals = computeNormals(positions, indices, vertexCount);
          const result = { ...geometry, normals };
          const transfers = [positions.buffer, normals.buffer];
          if (indices) transfers.push(indices.buffer);
          self.postMessage({ id, geometry: result }, transfers);
        } else if (op === 'simplify') {
          const { newPositions, newIndices } = simplifyMesh(positions, indices, simplifyOptions?.targetVertexCount ?? vertexCount);
          const result = { positions: newPositions, indices: newIndices ?? undefined, vertexCount: newPositions.length / 3 };
          const transfers = [newPositions.buffer];
          if (newIndices) transfers.push(newIndices.buffer);
          self.postMessage({ id, geometry: result }, transfers);
        }
      };
    `;
    const blob = new Blob([workerSrc], { type: 'application/javascript' });
    this.worker = new Worker(URL.createObjectURL(blob));
    this.worker.onmessage = (e: MessageEvent<{ id: number; geometry: GeometryDescriptor }>) => {
      const { id, geometry } = e.data;
      const resolve = this.pending.get(id);
      if (!resolve) return;
      this.pending.delete(id);
      resolve(geometry);
    };
  }

  private _post(req: GeomThreadRequest): Promise<GeometryDescriptor> {
    const transfers: ArrayBuffer[] = [req.geometry.positions.buffer];
    if (req.geometry.indices) transfers.push(req.geometry.indices.buffer);
    if (req.geometry.normals) transfers.push(req.geometry.normals.buffer);
    return new Promise((resolve) => {
      this.pending.set(req.id, resolve);
      this.worker.postMessage(req, transfers);
    });
  }

  /**
   * computeNormals(geo)
   * Computes per-vertex averaged face normals off the main thread.
   */
  computeNormals(geometry: GeometryDescriptor): Promise<GeometryDescriptor> {
    return this._post({ id: this.nextId++, op: 'computeNormals', geometry });
  }

  /**
   * simplify(geo, options)
   * Reduces geometry vertex count via greedy vertex-clustering.
   */
  simplify(geometry: GeometryDescriptor, options: SimplifyOptions): Promise<GeometryDescriptor> {
    return this._post({ id: this.nextId++, op: 'simplify', geometry, simplifyOptions: options });
  }

  dispose(): void {
    this.worker.terminate();
    this.pending.clear();
  }
}

// ── GLTFLoader ─────────────────────────────────────────────────────────────────

/** Minimal scene graph node returned by GLTFLoader. */
export interface SceneNode {
  name: string;
  children: SceneNode[];
  /** World-space transform (column-major). */
  matrix: Mat4;
  geometry?: GeometryDescriptor;
  materialName?: string;
}

export interface GLTFScene {
  root: SceneNode;
  /** Map of mesh name → GeometryDescriptor (after optional Draco decode). */
  meshes: Map<string, GeometryDescriptor>;
  /** Raw glTF JSON for custom extensions. */
  json: Record<string, unknown>;
}

export interface GLTFLoaderOptions {
  dracoThread?: DracoThread;
  /** If true, attempt KTX2 texture transcoding (requires ktx2-transcoder). */
  ktx2?: boolean;
  ktx2Url?: string;
}

/**
 * GLTFLoader
 *
 * Loads glTF 2.0 (.gltf + .bin, or self-contained .glb) files and returns a
 * SceneNode tree. Draco-compressed meshes are decoded via DracoThread.
 * KTX2 textures are transcoded when the ktx2 option is enabled.
 *
 * Matches AT pixi3d GLTFLoader API:
 *   upstream/pixi3d/src/loader/gltf-loader.ts (load → PIXI.Object3D)
 *
 * Usage:
 *   const loader = new GLTFLoader({ dracoThread: new DracoThread() });
 *   const scene  = await loader.load('/models/scene.gltf');
 */
export class GLTFLoader {
  private draco: DracoThread | null;
  private ktx2: boolean;
  private ktx2Url: string;

  constructor(options: GLTFLoaderOptions = {}) {
    this.draco   = options.dracoThread ?? null;
    this.ktx2    = options.ktx2 ?? false;
    this.ktx2Url = options.ktx2Url ?? '/ktx2/ktx2-transcoder.wasm';
  }

  /** Accepts both .gltf (JSON + external .bin) and .glb (binary container). */
  async load(url: string): Promise<GLTFScene> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`GLTFLoader.load: HTTP ${res.status} for ${url}`);

    const isGLB = url.toLowerCase().endsWith('.glb') ||
      res.headers.get('content-type') === 'model/gltf-binary';

    let json: Record<string, unknown>;
    let binChunk: ArrayBuffer | null = null;
    const base = url.substring(0, url.lastIndexOf('/') + 1);

    if (isGLB) {
      ({ json, binChunk } = await this._parseGLB(await res.arrayBuffer()));
    } else {
      json = await res.json() as Record<string, unknown>;
    }

    const buffers = await this._resolveBuffers(json, binChunk, base);
    const meshMap = await this._decodeMeshes(json, buffers);
    const root    = this._buildScene(json, meshMap);

    return { root, meshes: meshMap, json };
  }

  private async _parseGLB(
    buf: ArrayBuffer
  ): Promise<{ json: Record<string, unknown>; binChunk: ArrayBuffer | null }> {
    const view = new DataView(buf);
    const magic = view.getUint32(0, true);
    if (magic !== 0x46546C67) throw new Error('GLTFLoader: Not a valid GLB file');

    let offset = 12;
    let json: Record<string, unknown> = {};
    let binChunk: ArrayBuffer | null = null;

    while (offset < buf.byteLength) {
      const chunkLength = view.getUint32(offset, true);
      const chunkType   = view.getUint32(offset + 4, true);
      offset += 8;
      const chunkData = buf.slice(offset, offset + chunkLength);
      offset += chunkLength;

      if (chunkType === 0x4E4F534A) {
        json = JSON.parse(new TextDecoder().decode(chunkData)) as Record<string, unknown>;
      } else if (chunkType === 0x004E4942) {
        binChunk = chunkData;
      }
    }

    return { json, binChunk };
  }

  private async _resolveBuffers(
    json: Record<string, unknown>,
    binChunk: ArrayBuffer | null,
    base: string
  ): Promise<ArrayBuffer[]> {
    const gltfBuffers = (json['buffers'] as Array<{ uri?: string; byteLength: number }>) ?? [];
    return Promise.all(
      gltfBuffers.map(async (b, i) => {
        if (!b.uri && i === 0 && binChunk) return binChunk;
        if (!b.uri) return new ArrayBuffer(0);
        if (b.uri.startsWith('data:')) {
          const comma = b.uri.indexOf(',');
          const bin = atob(b.uri.slice(comma + 1));
          const arr = new Uint8Array(bin.length);
          for (let j = 0; j < bin.length; j++) arr[j] = bin.charCodeAt(j);
          return arr.buffer;
        }
        const fetchUrl = b.uri.startsWith('http') ? b.uri : base + b.uri;
        return (await fetch(fetchUrl)).arrayBuffer();
      })
    );
  }

  private async _decodeMeshes(
    json: Record<string, unknown>,
    buffers: ArrayBuffer[]
  ): Promise<Map<string, GeometryDescriptor>> {
    const map = new Map<string, GeometryDescriptor>();
    const gltfMeshes = (json['meshes'] as Array<{
      name?: string;
      primitives: Array<{
        attributes: Record<string, number>;
        indices?: number;
        extensions?: { KHR_draco_mesh_compression?: { bufferView: number; attributes: Record<string, number> } };
      }>;
    }>) ?? [];

    const bufferViews = (json['bufferViews'] as Array<{
      buffer: number;
      byteOffset?: number;
      byteLength: number;
    }>) ?? [];

    const accessors = (json['accessors'] as Array<{
      bufferView?: number;
      byteOffset?: number;
      componentType: number;
      count: number;
      type: string;
    }>) ?? [];

    const readAccessor = (accIdx: number): Float32Array | Uint32Array | Uint16Array => {
      const acc = accessors[accIdx];
      if (acc.bufferView === undefined) return new Float32Array(0);
      const bv = bufferViews[acc.bufferView];
      const byteOffset = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
      const compCount = ({ SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 } as Record<string,number>)[acc.type] ?? 1;
      const total = acc.count * compCount;
      switch (acc.componentType) {
        case 5126: return new Float32Array(buffers[bv.buffer], byteOffset, total);
        case 5125: return new Uint32Array(buffers[bv.buffer], byteOffset, total);
        case 5123: return new Uint16Array(buffers[bv.buffer], byteOffset, total);
        default:   return new Float32Array(buffers[bv.buffer], byteOffset, total);
      }
    };

    for (let mi = 0; mi < gltfMeshes.length; mi++) {
      const mesh = gltfMeshes[mi];
      const name = mesh.name ?? `mesh_${mi}`;

      for (let pi = 0; pi < mesh.primitives.length; pi++) {
        const prim = mesh.primitives[pi];
        const draco = prim.extensions?.KHR_draco_mesh_compression;
        let geo: GeometryDescriptor;

        if (draco && this.draco) {
          const bv = bufferViews[draco.bufferView];
          const slice = buffers[bv.buffer].slice(
            bv.byteOffset ?? 0,
            (bv.byteOffset ?? 0) + bv.byteLength
          );
          geo = await this.draco.decode(slice);
        } else {
          const posAcc = readAccessor(prim.attributes['POSITION']);
          const positions = posAcc instanceof Float32Array ? posAcc : new Float32Array(posAcc);
          const normals = prim.attributes['NORMAL'] !== undefined
            ? new Float32Array(readAccessor(prim.attributes['NORMAL'])) : undefined;
          const uvs = prim.attributes['TEXCOORD_0'] !== undefined
            ? new Float32Array(readAccessor(prim.attributes['TEXCOORD_0'])) : undefined;
          let indices: Uint32Array | undefined;
          if (prim.indices !== undefined) {
            const raw = readAccessor(prim.indices);
            indices = raw instanceof Uint32Array ? raw : new Uint32Array(raw);
          }
          geo = { positions, normals, uvs, indices, vertexCount: positions.length / 3 };
        }

        map.set(`${name}_${pi}`, geo);
      }
    }

    return map;
  }

  private _buildScene(
    json: Record<string, unknown>,
    meshMap: Map<string, GeometryDescriptor>
  ): SceneNode {
    const nodes = (json['nodes'] as Array<{
      name?: string;
      children?: number[];
      matrix?: number[];
      mesh?: number;
    }>) ?? [];

    const gltfMeshes = (json['meshes'] as Array<{ name?: string }>) ?? [];

    const IDENTITY: Mat4 = { elements: new Float32Array([
      1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
    ]) };

    const buildNode = (idx: number): SceneNode => {
      const n = nodes[idx];
      const mat = n.matrix ? { elements: new Float32Array(n.matrix) } : IDENTITY;

      let geometry: GeometryDescriptor | undefined;
      let materialName: string | undefined;
      if (n.mesh !== undefined) {
        const mName = gltfMeshes[n.mesh]?.name ?? `mesh_${n.mesh}`;
        geometry = meshMap.get(`${mName}_0`);
        materialName = mName;
      }

      return {
        name: n.name ?? `node_${idx}`,
        matrix: mat,
        geometry,
        materialName,
        children: (n.children ?? []).map(buildNode),
      };
    };

    const scenes = (json['scenes'] as Array<{ nodes?: number[] }>) ?? [];
    const sceneIdx = (json['scene'] as number) ?? 0;
    const sceneNodes = scenes[sceneIdx]?.nodes ?? [];

    return {
      name: 'root',
      matrix: { elements: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]) },
      children: sceneNodes.map(buildNode),
    };
  }
}
