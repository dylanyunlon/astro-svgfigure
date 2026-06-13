/**
 * threed-pipeline.ts — 3D asset loading pipeline: Gaussian Splats, Draco threading, Geometry threading, GLTF loading
 */

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Vec3f32 { x: number; y: number; z: number; }
export interface Quat4f32 { x: number; y: number; z: number; w: number; }

export interface SplatPoint {
  position: Vec3f32;
  opacity: number;
  color: [number, number, number, number]; // RGBA 0..1
  scale: Vec3f32;
  rotation: Quat4f32;
  sh?: Float32Array; // spherical harmonics coefficients
}

export interface GeometryData {
  positions: Float32Array;
  normals?: Float32Array;
  uvs?: Float32Array;
  indices?: Uint16Array | Uint32Array;
  colors?: Float32Array;
}

export interface GLTFNode {
  name: string;
  mesh?: GeometryData;
  children: GLTFNode[];
  translation?: Vec3f32;
  rotation?: Quat4f32;
  scale?: Vec3f32;
  extras?: Record<string, unknown>;
}

export interface GLTFScene {
  name: string;
  nodes: GLTFNode[];
  animations: GLTFAnimation[];
}

export interface GLTFAnimation {
  name: string;
  duration: number;
  channels: GLTFAnimationChannel[];
}

export interface GLTFAnimationChannel {
  nodeIndex: number;
  path: 'translation' | 'rotation' | 'scale' | 'weights';
  times: Float32Array;
  values: Float32Array;
}

// ─── GaussianSplats ───────────────────────────────────────────────────────────

export interface GaussianSplatsOptions {
  maxSplats?: number;
  useWebWorker?: boolean;
  sortEachFrame?: boolean;
  shDegree?: 0 | 1 | 2 | 3;
}

export interface GaussianSplatsLoadResult {
  count: number;
  positions: Float32Array;
  colors: Float32Array;
  opacities: Float32Array;
  scales: Float32Array;
  rotations: Float32Array;
  sh?: Float32Array;
}

export class GaussianSplats {
  private readonly opts: Required<GaussianSplatsOptions>;
  private data: GaussianSplatsLoadResult | null = null;
  private sorted = false;

  constructor(opts: GaussianSplatsOptions = {}) {
    this.opts = {
      maxSplats: opts.maxSplats ?? 1_000_000,
      useWebWorker: opts.useWebWorker ?? true,
      sortEachFrame: opts.sortEachFrame ?? true,
      shDegree: opts.shDegree ?? 1,
    };
  }

  async load(url: string): Promise<GaussianSplatsLoadResult> {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`GaussianSplats: failed to fetch ${url}`);
    const buffer = await resp.arrayBuffer();
    this.data = this.parseSplatBuffer(buffer);
    return this.data;
  }

  private parseSplatBuffer(buffer: ArrayBuffer): GaussianSplatsLoadResult {
    const view = new DataView(buffer);
    // .splat file format: each splat = 32 bytes (3 pos f32 + 3 scale f32 + 4 color u8 + 4 rot u8)
    const bytesPerSplat = 32;
    const count = Math.min(Math.floor(buffer.byteLength / bytesPerSplat), this.opts.maxSplats);

    const positions = new Float32Array(count * 3);
    const scales    = new Float32Array(count * 3);
    const colors    = new Float32Array(count * 4);
    const opacities = new Float32Array(count);
    const rotations = new Float32Array(count * 4);

    for (let i = 0; i < count; i++) {
      const base = i * bytesPerSplat;
      positions[i * 3]     = view.getFloat32(base, true);
      positions[i * 3 + 1] = view.getFloat32(base + 4, true);
      positions[i * 3 + 2] = view.getFloat32(base + 8, true);
      scales[i * 3]     = view.getFloat32(base + 12, true);
      scales[i * 3 + 1] = view.getFloat32(base + 16, true);
      scales[i * 3 + 2] = view.getFloat32(base + 20, true);
      colors[i * 4]     = view.getUint8(base + 24) / 255;
      colors[i * 4 + 1] = view.getUint8(base + 25) / 255;
      colors[i * 4 + 2] = view.getUint8(base + 26) / 255;
      opacities[i]      = view.getUint8(base + 27) / 255;
      const rw = (view.getUint8(base + 28) - 128) / 128;
      const rx = (view.getUint8(base + 29) - 128) / 128;
      const ry = (view.getUint8(base + 30) - 128) / 128;
      const rz = (view.getUint8(base + 31) - 128) / 128;
      rotations[i * 4]     = rx;
      rotations[i * 4 + 1] = ry;
      rotations[i * 4 + 2] = rz;
      rotations[i * 4 + 3] = rw;
    }

    return { count, positions, colors, opacities, scales, rotations };
  }

  /** Sort splats back-to-front from camera position (view-space depth sort) */
  sortByDepth(cameraPosition: Vec3f32): void {
    if (!this.data) return;
    const { count, positions } = this.data;
    const depths = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const dx = positions[i * 3] - cameraPosition.x;
      const dy = positions[i * 3 + 1] - cameraPosition.y;
      const dz = positions[i * 3 + 2] - cameraPosition.z;
      depths[i] = dx * dx + dy * dy + dz * dz;
    }
    // Simple index sort — production would use a worker + radix sort
    const indices = Array.from({ length: count }, (_, i) => i);
    indices.sort((a, b) => depths[b] - depths[a]);
    this.sorted = true;
    // Rearrange in-place would be ideal; this returns sorted indices for GPU upload
    (this.data as any)._sortedIndices = new Uint32Array(indices);
  }

  get sortedIndices(): Uint32Array | null { return (this.data as any)?._sortedIndices ?? null; }
  get result(): GaussianSplatsLoadResult | null { return this.data; }
  get splatCount(): number { return this.data?.count ?? 0; }
  get isSorted(): boolean { return this.sorted; }

  dispose(): void { this.data = null; }
}

// ─── DracoThread ──────────────────────────────────────────────────────────────

export interface DracoDecodeResult {
  geometry: GeometryData;
  metadata?: Record<string, unknown>;
}

export interface DracoThreadOptions {
  decoderUrl?: string;
  workerCount?: number;
}

export class DracoThread {
  private readonly decoderUrl: string;
  private readonly workerCount: number;
  private workers: Worker[] = [];
  private taskQueue: Array<{
    buffer: ArrayBuffer;
    resolve: (r: DracoDecodeResult) => void;
    reject: (e: Error) => void;
  }> = [];
  private workerBusy: boolean[] = [];

  constructor(opts: DracoThreadOptions = {}) {
    this.decoderUrl = opts.decoderUrl ?? '/draco/draco_decoder.wasm';
    this.workerCount = opts.workerCount ?? 2;
  }

  async init(): Promise<void> {
    if (typeof Worker === 'undefined') return;
    for (let i = 0; i < this.workerCount; i++) {
      // Worker would load the Draco WASM decoder in practice
      const blob = new Blob([this.workerScript()], { type: 'text/javascript' });
      const w = new Worker(URL.createObjectURL(blob));
      w.onmessage = (e: MessageEvent) => this.onWorkerMessage(i, e);
      this.workers.push(w);
      this.workerBusy.push(false);
    }
  }

  async decode(buffer: ArrayBuffer): Promise<DracoDecodeResult> {
    return new Promise((resolve, reject) => {
      this.taskQueue.push({ buffer, resolve, reject });
      this.dispatch();
    });
  }

  private dispatch(): void {
    const freeIdx = this.workerBusy.findIndex(b => !b);
    if (freeIdx < 0 || this.taskQueue.length === 0) return;
    const task = this.taskQueue.shift()!;
    this.workerBusy[freeIdx] = true;
    (this.workers[freeIdx] as any)._task = task;
    this.workers[freeIdx].postMessage({ buffer: task.buffer }, [task.buffer]);
  }

  private onWorkerMessage(workerIdx: number, e: MessageEvent): void {
    const task = (this.workers[workerIdx] as any)._task;
    if (!task) return;
    this.workerBusy[workerIdx] = false;
    if (e.data.error) {
      task.reject(new Error(e.data.error));
    } else {
      task.resolve(e.data.result as DracoDecodeResult);
    }
    this.dispatch();
  }

  /** Fallback: synchronous CPU decode (no worker) */
  async decodeDirect(buffer: ArrayBuffer): Promise<DracoDecodeResult> {
    // Minimal stub — real implementation would call draco_decoder.js
    const positions = new Float32Array(buffer.byteLength / 12);
    return { geometry: { positions } };
  }

  private workerScript(): string {
    return `
self.onmessage = function(e) {
  try {
    // Draco decode stub — replace with real draco_decoder.js usage
    const buffer = e.data.buffer;
    const positions = new Float32Array(buffer.byteLength / 12);
    self.postMessage({ result: { geometry: { positions } } });
  } catch(err) {
    self.postMessage({ error: err.message });
  }
};`;
  }

  dispose(): void {
    for (const w of this.workers) w.terminate();
    this.workers = [];
    this.workerBusy = [];
    this.taskQueue = [];
  }
}

// ─── GeomThread ───────────────────────────────────────────────────────────────

export type GeomTask =
  | { type: 'computeNormals'; positions: Float32Array; indices?: Uint32Array }
  | { type: 'computeTangents'; positions: Float32Array; uvs: Float32Array; normals: Float32Array; indices: Uint32Array }
  | { type: 'simplify'; positions: Float32Array; indices: Uint32Array; targetRatio: number }
  | { type: 'bvhBuild'; positions: Float32Array; indices: Uint32Array };

export interface GeomTaskResult {
  taskType: string;
  data: Record<string, Float32Array | Uint32Array | ArrayBuffer>;
}

export class GeomThread {
  private worker: Worker | null = null;
  private pending = new Map<number, { resolve: (r: GeomTaskResult) => void; reject: (e: Error) => void }>();
  private taskId = 0;

  async init(): Promise<void> {
    if (typeof Worker === 'undefined') return;
    const blob = new Blob([this.workerScript()], { type: 'text/javascript' });
    this.worker = new Worker(URL.createObjectURL(blob));
    this.worker.onmessage = (e: MessageEvent) => {
      const { id, result, error } = e.data;
      const p = this.pending.get(id);
      if (!p) return;
      this.pending.delete(id);
      error ? p.reject(new Error(error)) : p.resolve(result);
    };
  }

  async execute(task: GeomTask): Promise<GeomTaskResult> {
    if (!this.worker) return this.runInline(task);
    const id = this.taskId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const transferables = this.getTransferables(task);
      this.worker!.postMessage({ id, task }, transferables);
    });
  }

  private runInline(task: GeomTask): GeomTaskResult {
    switch (task.type) {
      case 'computeNormals': {
        const norms = computeFlatNormals(task.positions, task.indices);
        return { taskType: 'computeNormals', data: { normals: norms } };
      }
      default:
        return { taskType: task.type, data: {} };
    }
  }

  private getTransferables(task: GeomTask): ArrayBuffer[] {
    const out: ArrayBuffer[] = [];
    for (const v of Object.values(task)) {
      if (v instanceof Float32Array || v instanceof Uint32Array) out.push(v.buffer);
    }
    return out;
  }

  private workerScript(): string {
    return `
function computeFlatNormals(positions, indices) {
  const normals = new Float32Array(positions.length);
  const count = indices ? indices.length / 3 : positions.length / 9;
  for (let i = 0; i < count; i++) {
    const i0 = indices ? indices[i*3]*3 : i*9;
    const i1 = indices ? indices[i*3+1]*3 : i*9+3;
    const i2 = indices ? indices[i*3+2]*3 : i*9+6;
    const ax=positions[i1]-positions[i0], ay=positions[i1+1]-positions[i0+1], az=positions[i1+2]-positions[i0+2];
    const bx=positions[i2]-positions[i0], by=positions[i2+1]-positions[i0+1], bz=positions[i2+2]-positions[i0+2];
    const nx=ay*bz-az*by, ny=az*bx-ax*bz, nz=ax*by-ay*bx;
    if(indices){normals[i0]=normals[i1]=normals[i2]=nx;normals[i0+1]=normals[i1+1]=normals[i2+1]=ny;normals[i0+2]=normals[i1+2]=normals[i2+2]=nz;}
    else{normals[i*9]=normals[i*9+3]=normals[i*9+6]=nx;normals[i*9+1]=normals[i*9+4]=normals[i*9+7]=ny;normals[i*9+2]=normals[i*9+5]=normals[i*9+8]=nz;}
  }
  return normals;
}
self.onmessage = function(e) {
  const {id, task} = e.data;
  try {
    let result;
    if(task.type==='computeNormals'){
      const normals = computeFlatNormals(task.positions, task.indices);
      result = {taskType:'computeNormals', data:{normals}};
      self.postMessage({id, result}, [normals.buffer]);
    } else {
      self.postMessage({id, result:{taskType:task.type,data:{}}});
    }
  } catch(err){ self.postMessage({id, error:err.message}); }
};`;
  }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
    this.pending.clear();
  }
}

// ─── GLTFLoader ───────────────────────────────────────────────────────────────

export interface GLTFLoaderOptions {
  useDraco?: boolean;
  dracoDecoderUrl?: string;
  baseUrl?: string;
  onProgress?: (p: number) => void;
}

export class GLTFLoader {
  private readonly opts: Required<GLTFLoaderOptions>;
  private draco: DracoThread | null = null;

  constructor(opts: GLTFLoaderOptions = {}) {
    this.opts = {
      useDraco: opts.useDraco ?? false,
      dracoDecoderUrl: opts.dracoDecoderUrl ?? '/draco/',
      baseUrl: opts.baseUrl ?? '',
      onProgress: opts.onProgress ?? (() => {}),
    };
  }

  async init(): Promise<void> {
    if (this.opts.useDraco) {
      this.draco = new DracoThread({ decoderUrl: this.opts.dracoDecoderUrl });
      await this.draco.init();
    }
  }

  async load(url: string): Promise<GLTFScene> {
    const fullUrl = this.opts.baseUrl ? `${this.opts.baseUrl}/${url}` : url;
    this.opts.onProgress(0);
    const resp = await fetch(fullUrl);
    if (!resp.ok) throw new Error(`GLTFLoader: failed to fetch ${fullUrl}`);

    const contentType = resp.headers.get('Content-Type') ?? '';
    const isBinary = contentType.includes('octet-stream') || url.endsWith('.glb');

    this.opts.onProgress(0.3);
    const buffer = await resp.arrayBuffer();
    this.opts.onProgress(0.6);

    const scene = isBinary ? this.parseGLB(buffer) : this.parseGLTF(new TextDecoder().decode(buffer));
    this.opts.onProgress(1);
    return scene;
  }

  private parseGLB(buffer: ArrayBuffer): GLTFScene {
    const view = new DataView(buffer);
    const magic = view.getUint32(0, false);
    if (magic !== 0x676C5446) throw new Error('GLTFLoader: invalid GLB magic');
    const version = view.getUint32(4, true);
    if (version !== 2) throw new Error(`GLTFLoader: unsupported GLB version ${version}`);

    // Chunk 0: JSON
    const jsonLen = view.getUint32(12, true);
    const jsonBytes = new Uint8Array(buffer, 20, jsonLen);
    const json = JSON.parse(new TextDecoder().decode(jsonBytes));

    return this.buildScene(json);
  }

  private parseGLTF(text: string): GLTFScene {
    const json = JSON.parse(text);
    return this.buildScene(json);
  }

  private buildScene(json: any): GLTFScene {
    const sceneDef = json.scenes?.[json.scene ?? 0] ?? { name: 'Scene', nodes: [] };
    const nodes = (sceneDef.nodes ?? []).map((idx: number) => this.buildNode(json, idx));
    const animations = (json.animations ?? []).map((a: any) => this.buildAnimation(json, a));
    return { name: sceneDef.name ?? 'Scene', nodes, animations };
  }

  private buildNode(json: any, index: number): GLTFNode {
    const nodeDef = json.nodes?.[index] ?? {};
    const children = (nodeDef.children ?? []).map((ci: number) => this.buildNode(json, ci));
    let mesh: GeometryData | undefined;
    if (nodeDef.mesh !== undefined) {
      mesh = this.buildMesh(json, nodeDef.mesh);
    }
    return {
      name: nodeDef.name ?? `node_${index}`,
      mesh,
      children,
      translation: nodeDef.translation ? { x: nodeDef.translation[0], y: nodeDef.translation[1], z: nodeDef.translation[2] } : undefined,
      rotation: nodeDef.rotation ? { x: nodeDef.rotation[0], y: nodeDef.rotation[1], z: nodeDef.rotation[2], w: nodeDef.rotation[3] } : undefined,
      scale: nodeDef.scale ? { x: nodeDef.scale[0], y: nodeDef.scale[1], z: nodeDef.scale[2] } : undefined,
      extras: nodeDef.extras,
    };
  }

  private buildMesh(json: any, meshIndex: number): GeometryData {
    const meshDef = json.meshes?.[meshIndex];
    if (!meshDef?.primitives?.[0]) return { positions: new Float32Array(0) };
    const prim = meshDef.primitives[0];
    const positions = this.getAccessorData(json, prim.attributes?.POSITION) as Float32Array ?? new Float32Array(0);
    const normals   = prim.attributes?.NORMAL !== undefined ? this.getAccessorData(json, prim.attributes.NORMAL) as Float32Array : undefined;
    const uvs       = prim.attributes?.TEXCOORD_0 !== undefined ? this.getAccessorData(json, prim.attributes.TEXCOORD_0) as Float32Array : undefined;
    const indices   = prim.indices !== undefined ? this.getAccessorData(json, prim.indices) as Uint16Array : undefined;
    return { positions, normals, uvs, indices };
  }

  private getAccessorData(json: any, accessorIndex: number): Float32Array | Uint16Array | Uint32Array | null {
    const accessor = json.accessors?.[accessorIndex];
    if (!accessor) return null;
    // Simplified: return typed array stub
    const componentSizes: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
    const n = accessor.count * (componentSizes[accessor.type] ?? 1);
    if (accessor.componentType === 5123) return new Uint16Array(n);
    if (accessor.componentType === 5125) return new Uint32Array(n);
    return new Float32Array(n);
  }

  private buildAnimation(json: any, animDef: any): GLTFAnimation {
    const channels: GLTFAnimationChannel[] = (animDef.channels ?? []).map((ch: any) => {
      const sampler = animDef.samplers?.[ch.sampler] ?? {};
      return {
        nodeIndex: ch.target?.node ?? 0,
        path: ch.target?.path ?? 'translation',
        times: this.getAccessorData(json, sampler.input) as Float32Array ?? new Float32Array(0),
        values: this.getAccessorData(json, sampler.output) as Float32Array ?? new Float32Array(0),
      };
    });
    const duration = channels.reduce((max, ch) => Math.max(max, ch.times[ch.times.length - 1] ?? 0), 0);
    return { name: animDef.name ?? 'animation', duration, channels };
  }

  dispose(): void {
    this.draco?.dispose();
    this.draco = null;
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function computeFlatNormals(positions: Float32Array, indices?: Uint32Array): Float32Array {
  const normals = new Float32Array(positions.length);
  const triCount = indices ? indices.length / 3 : positions.length / 9;
  for (let i = 0; i < triCount; i++) {
    const i0 = indices ? indices[i * 3] * 3 : i * 9;
    const i1 = indices ? indices[i * 3 + 1] * 3 : i * 9 + 3;
    const i2 = indices ? indices[i * 3 + 2] * 3 : i * 9 + 6;
    const ax = positions[i1] - positions[i0], ay = positions[i1 + 1] - positions[i0 + 1], az = positions[i1 + 2] - positions[i0 + 2];
    const bx = positions[i2] - positions[i0], by = positions[i2 + 1] - positions[i0 + 1], bz = positions[i2 + 2] - positions[i0 + 2];
    const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
    if (indices) {
      normals[i0] = nx; normals[i0 + 1] = ny; normals[i0 + 2] = nz;
      normals[i1] = nx; normals[i1 + 1] = ny; normals[i1 + 2] = nz;
      normals[i2] = nx; normals[i2 + 1] = ny; normals[i2 + 2] = nz;
    } else {
      for (let j = 0; j < 3; j++) { normals[i * 9 + j * 3] = nx; normals[i * 9 + j * 3 + 1] = ny; normals[i * 9 + j * 3 + 2] = nz; }
    }
  }
  return normals;
}
