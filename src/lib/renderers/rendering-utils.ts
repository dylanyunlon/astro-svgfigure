/**
 * renderers/rendering-utils.ts — GPU rendering utilities: screen projection, fullscreen quad,
 * frustum culling, lighting, light volumes, render target pool
 */

import { Container } from 'pixi.js';
import { GlowFilter } from '@pixi/filter-glow';
import { DropShadowFilter } from '@pixi/filter-drop-shadow';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Vec2 { x: number; y: number; }
export interface Vec3 { x: number; y: number; z: number; }
export interface Vec4 { x: number; y: number; z: number; w: number; }
export type Mat4 = Float32Array; // column-major, 16 elements

// ─── ScreenProjection ─────────────────────────────────────────────────────────

export interface ScreenProjectionOptions {
  fov?: number;        // vertical field of view, degrees
  near?: number;
  far?: number;
  width: number;
  height: number;
}

export class ScreenProjection {
  private _projection: Mat4;
  private _view: Mat4;
  private _viewProj: Mat4;
  private opts: Required<ScreenProjectionOptions>;

  constructor(opts: ScreenProjectionOptions) {
    this.opts = { fov: opts.fov ?? 60, near: opts.near ?? 0.1, far: opts.far ?? 1000, ...opts };
    this._projection = new Float32Array(16);
    this._view = mat4Identity();
    this._viewProj = new Float32Array(16);
    this.updateProjection();
  }

  resize(w: number, h: number): void {
    this.opts.width = w;
    this.opts.height = h;
    this.updateProjection();
  }

  setView(mat: Mat4): void {
    this._view = mat;
    this._viewProj = mat4Mul(this._projection, this._view);
  }

  /** Project world-space point to NDC [-1..1] */
  project(p: Vec3): Vec3 {
    const vp = this._viewProj;
    const x = vp[0] * p.x + vp[4] * p.y + vp[8] * p.z + vp[12];
    const y = vp[1] * p.x + vp[5] * p.y + vp[9] * p.z + vp[13];
    const z = vp[2] * p.x + vp[6] * p.y + vp[10] * p.z + vp[14];
    const w = vp[3] * p.x + vp[7] * p.y + vp[11] * p.z + vp[15];
    const invW = 1 / (w || 1e-7);
    return { x: x * invW, y: y * invW, z: z * invW };
  }

  /** Project NDC to screen pixels */
  ndcToScreen(ndc: Vec3): Vec2 {
    return {
      x: (ndc.x + 1) * 0.5 * this.opts.width,
      y: (1 - (ndc.y + 1) * 0.5) * this.opts.height,
    };
  }

  /** Unproject screen point at depth t (0=near, 1=far) back to world */
  unproject(screen: Vec2, depth = 0): Vec3 {
    const ndcX = (screen.x / this.opts.width) * 2 - 1;
    const ndcY = 1 - (screen.y / this.opts.height) * 2;
    const ndcZ = depth * 2 - 1;
    const invVP = mat4Invert(this._viewProj);
    if (!invVP) return { x: 0, y: 0, z: 0 };
    return transformVec3(invVP, { x: ndcX, y: ndcY, z: ndcZ });
  }

  /** World-to-screen (pixels) shortcut */
  worldToScreen(p: Vec3): Vec2 {
    return this.ndcToScreen(this.project(p));
  }

  get projectionMatrix(): Mat4 { return this._projection; }
  get viewMatrix(): Mat4 { return this._view; }
  get viewProjectionMatrix(): Mat4 { return this._viewProj; }
  get aspect(): number { return this.opts.width / this.opts.height; }

  private updateProjection(): void {
    this._projection = mat4Perspective(
      this.opts.fov * Math.PI / 180,
      this.opts.width / this.opts.height,
      this.opts.near,
      this.opts.far,
    );
    this._viewProj = mat4Mul(this._projection, this._view);
  }
}

// ─── ScreenQuad ───────────────────────────────────────────────────────────────

/**
 * Full-screen triangle / quad for post-processing passes.
 * Uses a single oversized triangle to avoid the diagonal seam of two triangles.
 */
export class ScreenQuad {
  private vao: WebGLVertexArrayObject | null = null;
  private vbo: WebGLBuffer | null = null;
  private readonly gl: WebGL2RenderingContext;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.init();
  }

  private init(): void {
    const gl = this.gl;
    // Three vertices forming a clip-space triangle that covers NDC [-1..1]
    const verts = new Float32Array([-1, -1, 3, -1, -1, 3]);
    this.vao = gl.createVertexArray();
    this.vbo = gl.createBuffer();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  draw(): void {
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  dispose(): void {
    const gl = this.gl;
    if (this.vao) gl.deleteVertexArray(this.vao);
    if (this.vbo) gl.deleteBuffer(this.vbo);
    this.vao = null;
    this.vbo = null;
  }
}

// ─── Frustum ──────────────────────────────────────────────────────────────────

export interface FrustumPlane { normal: Vec3; constant: number; }
export interface AABB { min: Vec3; max: Vec3; }
export interface Sphere { center: Vec3; radius: number; }

export class Frustum {
  private planes: FrustumPlane[] = [];

  /** Extract 6 frustum planes from a view-projection matrix (row-major input) */
  setFromViewProjection(vp: Mat4): void {
    this.planes = [
      this.extractPlane(vp, 3, 0, true),   // left
      this.extractPlane(vp, 3, 0, false),  // right
      this.extractPlane(vp, 3, 1, true),   // bottom
      this.extractPlane(vp, 3, 1, false),  // top
      this.extractPlane(vp, 3, 2, true),   // near
      this.extractPlane(vp, 3, 2, false),  // far
    ];
  }

  containsSphere(sphere: Sphere): boolean {
    for (const p of this.planes) {
      if (dot(p.normal, sphere.center) + p.constant < -sphere.radius) return false;
    }
    return true;
  }

  containsAABB(aabb: AABB): boolean {
    for (const p of this.planes) {
      const px = p.normal.x >= 0 ? aabb.max.x : aabb.min.x;
      const py = p.normal.y >= 0 ? aabb.max.y : aabb.min.y;
      const pz = p.normal.z >= 0 ? aabb.max.z : aabb.min.z;
      if (dot(p.normal, { x: px, y: py, z: pz }) + p.constant < 0) return false;
    }
    return true;
  }

  containsPoint(p: Vec3): boolean {
    for (const plane of this.planes) {
      if (dot(plane.normal, p) + plane.constant < 0) return false;
    }
    return true;
  }

  private extractPlane(m: Mat4, row3: number, col: number, negate: boolean): FrustumPlane {
    const sign = negate ? 1 : -1;
    const nx = m[row3] * sign + m[col];
    const ny = m[row3 + 4] * sign + m[col + 4];
    const nz = m[row3 + 8] * sign + m[col + 8];
    const nw = m[row3 + 12] * sign + m[col + 12];
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
    return { normal: { x: nx / len, y: ny / len, z: nz / len }, constant: nw / len };
  }
}

// ─── Lighting ─────────────────────────────────────────────────────────────────

export type LightKind = 'directional' | 'point' | 'spot' | 'ambient';

export interface Light {
  id: string;
  kind: LightKind;
  color: Vec3;
  intensity: number;
  position?: Vec3;
  direction?: Vec3;
  range?: number;      // point / spot
  innerAngle?: number; // spot (radians)
  outerAngle?: number; // spot (radians)
  castShadow?: boolean;
}

export interface LightingUniforms {
  ambientColor: Float32Array;
  lightPositions: Float32Array;
  lightColors: Float32Array;
  lightIntensities: Float32Array;
  lightKinds: Int32Array;
  lightRanges: Float32Array;
  lightCount: number;
}

export class Lighting {
  private lights = new Map<string, Light>();
  private ambient: Vec3 = { x: 0.1, y: 0.1, z: 0.1 };
  private readonly maxLights: number;

  constructor(maxLights = 16) {
    this.maxLights = maxLights;
  }

  add(light: Light): void {
    if (this.lights.size >= this.maxLights) {
      console.warn(`Lighting: max lights (${this.maxLights}) reached`);
      return;
    }
    this.lights.set(light.id, { ...light });
  }

  remove(id: string): void { this.lights.delete(id); }

  update(id: string, patch: Partial<Omit<Light, 'id'>>): void {
    const l = this.lights.get(id);
    if (l) Object.assign(l, patch);
  }

  setAmbient(color: Vec3): void { this.ambient = color; }

  /** Build a flat uniform buffer for upload to a shader */
  buildUniforms(): LightingUniforms {
    const count = this.lights.size;
    const positions = new Float32Array(count * 4);
    const colors = new Float32Array(count * 4);
    const intensities = new Float32Array(count);
    const kinds = new Int32Array(count);
    const ranges = new Float32Array(count);

    const kindMap: Record<LightKind, number> = { ambient: 0, directional: 1, point: 2, spot: 3 };
    let i = 0;
    for (const l of this.lights.values()) {
      const p = l.position ?? l.direction ?? { x: 0, y: 1, z: 0 };
      positions[i * 4] = p.x; positions[i * 4 + 1] = p.y; positions[i * 4 + 2] = p.z;
      positions[i * 4 + 3] = l.kind === 'directional' ? 0 : 1;
      colors[i * 4] = l.color.x; colors[i * 4 + 1] = l.color.y; colors[i * 4 + 2] = l.color.z; colors[i * 4 + 3] = 1;
      intensities[i] = l.intensity;
      kinds[i] = kindMap[l.kind];
      ranges[i] = l.range ?? 100;
      i++;
    }

    return {
      ambientColor: new Float32Array([this.ambient.x, this.ambient.y, this.ambient.z, 1]),
      lightPositions: positions,
      lightColors: colors,
      lightIntensities: intensities,
      lightKinds: kinds,
      lightRanges: ranges,
      lightCount: count,
    };
  }

  get lightList(): Light[] { return [...this.lights.values()]; }
  get count(): number { return this.lights.size; }
}

// ─── LightVolume ──────────────────────────────────────────────────────────────

export interface LightVolumeOptions {
  gl: WebGL2RenderingContext;
  maxLights?: number;
}

/**
 * Tiled / clustered light volume for deferred rendering.
 * Divides screen into tiles and assigns lights to each tile's Z-cluster.
 */
export class LightVolume {
  private readonly gl: WebGL2RenderingContext;
  private readonly maxLights: number;
  private tileCountX = 0;
  private tileCountY = 0;
  private readonly tileSize = 16;
  private clusters: Uint32Array = new Uint32Array(0); // tileX * tileY * maxZ clusters
  private lightBuffer: WebGLBuffer | null = null;
  private clusterBuffer: WebGLBuffer | null = null;

  constructor(opts: LightVolumeOptions) {
    this.gl = opts.gl;
    this.maxLights = opts.maxLights ?? 256;
    this.lightBuffer = this.gl.createBuffer();
    this.clusterBuffer = this.gl.createBuffer();
  }

  resize(w: number, h: number): void {
    this.tileCountX = Math.ceil(w / this.tileSize);
    this.tileCountY = Math.ceil(h / this.tileSize);
    this.clusters = new Uint32Array(this.tileCountX * this.tileCountY * 8); // 8 Z slices
  }

  /** Assign lights to clusters (CPU-side; ideally done in a compute shader) */
  buildClusters(lights: Light[], frustum: Frustum): void {
    this.clusters.fill(0);
    for (let i = 0; i < lights.length && i < this.maxLights; i++) {
      const light = lights[i];
      if (!light.position || light.kind === 'directional') continue;
      // Simple AABB sphere test per cluster tile
      const r = light.range ?? 50;
      const sphere: Sphere = { center: light.position, radius: r };
      if (frustum.containsSphere(sphere)) {
        // Tag all clusters (simplified — real impl would compute tile bounds)
        for (let j = 0; j < this.clusters.length; j++) this.clusters[j]++;
      }
    }
    this.uploadClusters();
  }

  bindClusterBuffer(bindingPoint: number): void {
    const gl = this.gl;
    gl.bindBufferBase(gl.UNIFORM_BUFFER, bindingPoint, this.clusterBuffer);
  }

  private uploadClusters(): void {
    const gl = this.gl;
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.clusterBuffer);
    gl.bufferData(gl.UNIFORM_BUFFER, this.clusters, gl.DYNAMIC_DRAW);
  }

  get tileX(): number { return this.tileCountX; }
  get tileY(): number { return this.tileCountY; }

  dispose(): void {
    const gl = this.gl;
    if (this.lightBuffer) gl.deleteBuffer(this.lightBuffer);
    if (this.clusterBuffer) gl.deleteBuffer(this.clusterBuffer);
    this.lightBuffer = null;
    this.clusterBuffer = null;
  }
}

// ─── RTPool ───────────────────────────────────────────────────────────────────

export interface RTDescriptor {
  width: number;
  height: number;
  internalFormat?: number;  // gl.RGBA8 etc.
  format?: number;          // gl.RGBA
  type?: number;            // gl.UNSIGNED_BYTE
  filter?: number;          // gl.LINEAR
  wrap?: number;            // gl.CLAMP_TO_EDGE
  depth?: boolean;
  stencil?: boolean;
  samples?: number;         // MSAA
}

export interface RenderTarget {
  framebuffer: WebGLFramebuffer;
  texture: WebGLTexture | null;
  depthRenderbuffer: WebGLRenderbuffer | null;
  descriptor: RTDescriptor;
}

/**
 * Render target pool for post-processing ping-pong, shadow maps, etc.
 * Automatically reuses targets of the same size/format.
 */
export class RTPool {
  private readonly gl: WebGL2RenderingContext;
  private free = new Map<string, RenderTarget[]>();
  private inUse = new Set<RenderTarget>();

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
  }

  acquire(desc: RTDescriptor): RenderTarget {
    const key = this.descKey(desc);
    const pool = this.free.get(key);
    if (pool?.length) {
      const rt = pool.pop()!;
      this.inUse.add(rt);
      return rt;
    }
    const rt = this.create(desc);
    this.inUse.add(rt);
    return rt;
  }

  release(rt: RenderTarget): void {
    if (!this.inUse.has(rt)) return;
    this.inUse.delete(rt);
    const key = this.descKey(rt.descriptor);
    let pool = this.free.get(key);
    if (!pool) { pool = []; this.free.set(key, pool); }
    pool.push(rt);
  }

  /** Release all in-use targets and purge the free pool */
  purge(): void {
    for (const rt of [...this.inUse]) this.release(rt);
    for (const pool of this.free.values()) {
      for (const rt of pool) this.destroyRT(rt);
    }
    this.free.clear();
  }

  private create(desc: RTDescriptor): RenderTarget {
    const gl = this.gl;
    const {
      width, height,
      internalFormat = gl.RGBA8,
      format = gl.RGBA,
      type = gl.UNSIGNED_BYTE,
      filter = gl.LINEAR,
      wrap = gl.CLAMP_TO_EDGE,
      depth = false,
      samples = 0,
    } = desc;

    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);

    let tex: WebGLTexture | null = null;
    if (samples <= 1) {
      tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    } else {
      // MSAA renderbuffer
      const colorRB = gl.createRenderbuffer()!;
      gl.bindRenderbuffer(gl.RENDERBUFFER, colorRB);
      (gl as any).renderbufferStorageMultisample(gl.RENDERBUFFER, samples, internalFormat, width, height);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, colorRB);
    }

    let depthRB: WebGLRenderbuffer | null = null;
    if (depth) {
      depthRB = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, depthRB);
      const depthFmt = desc.stencil ? gl.DEPTH24_STENCIL8 : gl.DEPTH_COMPONENT24;
      if (samples > 1) {
        (gl as any).renderbufferStorageMultisample(gl.RENDERBUFFER, samples, depthFmt, width, height);
      } else {
        gl.renderbufferStorage(gl.RENDERBUFFER, depthFmt, width, height);
      }
      const attach = desc.stencil ? gl.DEPTH_STENCIL_ATTACHMENT : gl.DEPTH_ATTACHMENT;
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, attach, gl.RENDERBUFFER, depthRB);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { framebuffer: fb, texture: tex, depthRenderbuffer: depthRB, descriptor: desc };
  }

  private destroyRT(rt: RenderTarget): void {
    const gl = this.gl;
    gl.deleteFramebuffer(rt.framebuffer);
    if (rt.texture) gl.deleteTexture(rt.texture);
    if (rt.depthRenderbuffer) gl.deleteRenderbuffer(rt.depthRenderbuffer);
  }

  private descKey(d: RTDescriptor): string {
    return `${d.width}x${d.height}:${d.internalFormat ?? 0}:${d.depth ?? 0}:${d.samples ?? 0}`;
  }

  get freeCount(): number {
    let n = 0;
    for (const p of this.free.values()) n += p.length;
    return n;
  }

  get inUseCount(): number { return this.inUse.size; }

  dispose(): void {
    this.purge();
    for (const rt of this.inUse) this.destroyRT(rt);
    this.inUse.clear();
  }
}

// ─── Math Helpers ─────────────────────────────────────────────────────────────

function mat4Identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function mat4Perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect; m[5] = f;
  m[10] = (far + near) * nf; m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

function mat4Mul(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + i] * b[j * 4 + k];
      out[j * 4 + i] = sum;
    }
  }
  return out;
}

function mat4Invert(m: Mat4): Mat4 | null {
  const out = new Float32Array(16);
  const a = m;
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return null;
  det = 1 / det;
  out[0]  = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1]  = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2]  = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3]  = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4]  = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5]  = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6]  = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7]  = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8]  = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9]  = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

function transformVec3(m: Mat4, v: Vec3): Vec3 {
  const x = m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12];
  const y = m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13];
  const z = m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14];
  const w = m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15];
  const iw = 1 / (w || 1e-7);
  return { x: x * iw, y: y * iw, z: z * iw };
}

function dot(a: Vec3, b: Vec3): number { return a.x * b.x + a.y * b.y + a.z * b.z; }

// ─── Drawing / Filter Wrappers ───────────────────────────────────────────────

export function drawRoundedRect(
  g: { beginFill(c: number): void; drawRoundedRect(x: number, y: number, w: number, h: number, r: number): void; endFill(): void },
  w: number,
  h: number,
  r: number,
  fill: number,
): void {
  g.beginFill(fill);
  g.drawRoundedRect(0, 0, w, h, r);
  g.endFill();
}

export function wrapWithGlow(child: Container, color: number, strength: number): Container {
  const c = new Container();
  c.addChild(child);
  c.filters = [new GlowFilter({ color, outerStrength: strength })];
  return c;
}

export function wrapWithShadow(child: Container, color: number, blur: number): Container {
  const c = new Container();
  c.addChild(child);
  c.filters = [new DropShadowFilter({ color, blur })];
  return c;
}
