// engine-utils.ts — LinkedList, SnapshotFrame, OptimizationProfiler, CleanRoom, Quaternion, Interpolation

// ---------------------------------------------------------------------------
// LinkedList
// ---------------------------------------------------------------------------

interface ListNode<T> {
  value: T;
  prev: ListNode<T> | null;
  next: ListNode<T> | null;
}

export class LinkedList<T> {
  private _head: ListNode<T> | null = null;
  private _tail: ListNode<T> | null = null;
  private _size = 0;

  get size(): number { return this._size; }
  get head(): T | null { return this._head ? this._head.value : null; }
  get tail(): T | null { return this._tail ? this._tail.value : null; }

  /** Append to the end. Returns the new node (opaque handle for O(1) removal). */
  insert(value: T): ListNode<T> {
    const node: ListNode<T> = { value, prev: this._tail, next: null };
    if (this._tail) {
      this._tail.next = node;
    } else {
      this._head = node;
    }
    this._tail = node;
    this._size++;
    return node;
  }

  /** Prepend to the front. */
  prepend(value: T): ListNode<T> {
    const node: ListNode<T> = { value, prev: null, next: this._head };
    if (this._head) {
      this._head.prev = node;
    } else {
      this._tail = node;
    }
    this._head = node;
    this._size++;
    return node;
  }

  /** Remove a node by reference (O(1)). */
  remove(node: ListNode<T>): void {
    if (node.prev) node.prev.next = node.next;
    else this._head = node.next;
    if (node.next) node.next.prev = node.prev;
    else this._tail = node.prev;
    node.prev = null;
    node.next = null;
    this._size--;
  }

  /** Remove first node whose value matches predicate (O(n)). */
  removeWhere(predicate: (v: T) => boolean): boolean {
    let cur = this._head;
    while (cur) {
      if (predicate(cur.value)) { this.remove(cur); return true; }
      cur = cur.next;
    }
    return false;
  }

  /** Iterate every node in order. */
  forEach(cb: (value: T, index: number) => void): void {
    let cur = this._head;
    let i = 0;
    while (cur) {
      cb(cur.value, i++);
      cur = cur.next;
    }
  }

  toArray(): T[] {
    const arr: T[] = [];
    this.forEach((v) => arr.push(v));
    return arr;
  }

  clear(): void {
    this._head = null;
    this._tail = null;
    this._size = 0;
  }
}

// ---------------------------------------------------------------------------
// SnapshotFrame
// ---------------------------------------------------------------------------

export type Snapshot<T> = Readonly<T>;

export class SnapshotFrame<T extends object> {
  private _snapshots: Array<{ timestamp: number; data: Snapshot<T> }> = [];
  private _maxHistory: number;

  constructor(maxHistory = 64) {
    this._maxHistory = maxHistory;
  }

  /** Take a deep clone snapshot of the current state. */
  capture(state: T): Snapshot<T> {
    const snap = structuredClone(state) as Snapshot<T>;
    this._snapshots.push({ timestamp: performance.now(), data: snap });
    if (this._snapshots.length > this._maxHistory) {
      this._snapshots.shift();
    }
    return snap;
  }

  /** Restore the most recent snapshot, or one at an index from the end (0 = latest). */
  restore(fromEnd = 0): Snapshot<T> | null {
    const idx = this._snapshots.length - 1 - fromEnd;
    return idx >= 0 ? this._snapshots[idx].data : null;
  }

  /**
   * Compute a shallow diff between two snapshots (or the two most recent).
   * Returns an object with only the keys that differ.
   */
  diff(a?: Snapshot<T>, b?: Snapshot<T>): Partial<T> {
    const len = this._snapshots.length;
    const sa = a ?? (len >= 2 ? this._snapshots[len - 2].data : null);
    const sb = b ?? (len >= 1 ? this._snapshots[len - 1].data : null);
    if (!sa || !sb) return {};

    const result: Partial<T> = {};
    const keys = new Set([...Object.keys(sa), ...Object.keys(sb)]) as Set<keyof T>;
    for (const key of keys) {
      if ((sa as T)[key] !== (sb as T)[key]) {
        (result as T)[key] = (sb as T)[key];
      }
    }
    return result;
  }

  get historyLength(): number { return this._snapshots.length; }

  clear(): void { this._snapshots = []; }
}

// ---------------------------------------------------------------------------
// OptimizationProfiler
// ---------------------------------------------------------------------------

export interface MeasureResult {
  name: string;
  duration: number;
  startTime: number;
  endTime: number;
}

export interface ProfileReport {
  measures: MeasureResult[];
  totalDuration: number;
  averageDuration: number;
  minDuration: number;
  maxDuration: number;
}

export const OptimizationProfiler = (() => {
  const _marks = new Map<string, number>();
  const _results: MeasureResult[] = [];

  function mark(name: string): void {
    _marks.set(name, performance.now());
  }

  function measure(startMark: string, endMark?: string): MeasureResult | null {
    const startTime = _marks.get(startMark);
    if (startTime === undefined) {
      console.warn(`[Profiler] No mark found for "${startMark}"`);
      return null;
    }
    const endTime = endMark ? (_marks.get(endMark) ?? performance.now()) : performance.now();
    const result: MeasureResult = {
      name: endMark ? `${startMark}→${endMark}` : startMark,
      duration: endTime - startTime,
      startTime,
      endTime,
    };
    _results.push(result);
    return result;
  }

  function report(filter?: string): ProfileReport {
    const filtered = filter
      ? _results.filter(r => r.name.includes(filter))
      : _results;

    if (filtered.length === 0) {
      return { measures: [], totalDuration: 0, averageDuration: 0, minDuration: 0, maxDuration: 0 };
    }

    const durations = filtered.map(r => r.duration);
    return {
      measures: filtered,
      totalDuration: durations.reduce((a, b) => a + b, 0),
      averageDuration: durations.reduce((a, b) => a + b, 0) / durations.length,
      minDuration: Math.min(...durations),
      maxDuration: Math.max(...durations),
    };
  }

  function clear(): void {
    _marks.clear();
    _results.length = 0;
  }

  return { mark, measure, report, clear };
})();

// ---------------------------------------------------------------------------
// CleanRoom — WebGL state save/restore
// ---------------------------------------------------------------------------

export interface WebGLStateSnapshot {
  program: WebGLProgram | null;
  arrayBuffer: WebGLBuffer | null;
  elementArrayBuffer: WebGLBuffer | null;
  framebuffer: WebGLFramebuffer | null;
  renderbuffer: WebGLRenderbuffer | null;
  viewport: Int32Array;
  blendEnabled: boolean;
  depthTestEnabled: boolean;
  cullFaceEnabled: boolean;
  activeTexture: number;
}

export class CleanRoom {
  private _gl: WebGL2RenderingContext | WebGLRenderingContext;
  private _stack: WebGLStateSnapshot[] = [];

  constructor(gl: WebGL2RenderingContext | WebGLRenderingContext) {
    this._gl = gl;
  }

  /** Save the current WebGL state and optionally reset it to a clean baseline. */
  enter(resetToClean = false): WebGLStateSnapshot {
    const gl = this._gl;
    const snap: WebGLStateSnapshot = {
      program: gl.getParameter(gl.CURRENT_PROGRAM),
      arrayBuffer: gl.getParameter(gl.ARRAY_BUFFER_BINDING),
      elementArrayBuffer: gl.getParameter(gl.ELEMENT_ARRAY_BUFFER_BINDING),
      framebuffer: gl.getParameter(gl.FRAMEBUFFER_BINDING),
      renderbuffer: gl.getParameter(gl.RENDERBUFFER_BINDING),
      viewport: gl.getParameter(gl.VIEWPORT),
      blendEnabled: gl.isEnabled(gl.BLEND),
      depthTestEnabled: gl.isEnabled(gl.DEPTH_TEST),
      cullFaceEnabled: gl.isEnabled(gl.CULL_FACE),
      activeTexture: gl.getParameter(gl.ACTIVE_TEXTURE),
    };
    this._stack.push(snap);

    if (resetToClean) {
      gl.useProgram(null);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindRenderbuffer(gl.RENDERBUFFER, null);
      gl.disable(gl.BLEND);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
    }

    return snap;
  }

  /** Restore the most recently saved WebGL state. */
  exit(): void {
    const snap = this._stack.pop();
    if (!snap) { console.warn("[CleanRoom] exit() called without matching enter()"); return; }
    const gl = this._gl;

    gl.useProgram(snap.program);
    gl.bindBuffer(gl.ARRAY_BUFFER, snap.arrayBuffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, snap.elementArrayBuffer);
    gl.bindFramebuffer(gl.FRAMEBUFFER, snap.framebuffer);
    gl.bindRenderbuffer(gl.RENDERBUFFER, snap.renderbuffer);
    gl.viewport(snap.viewport[0], snap.viewport[1], snap.viewport[2], snap.viewport[3]);
    snap.blendEnabled ? gl.enable(gl.BLEND) : gl.disable(gl.BLEND);
    snap.depthTestEnabled ? gl.enable(gl.DEPTH_TEST) : gl.disable(gl.DEPTH_TEST);
    snap.cullFaceEnabled ? gl.enable(gl.CULL_FACE) : gl.disable(gl.CULL_FACE);
    gl.activeTexture(snap.activeTexture);
  }

  get depth(): number { return this._stack.length; }
}

// ---------------------------------------------------------------------------
// Quaternion
// ---------------------------------------------------------------------------

/** Quaternion in [x, y, z, w] component order. */
export class Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;

  constructor(x = 0, y = 0, z = 0, w = 1) {
    this.x = x; this.y = y; this.z = z; this.w = w;
  }

  static identity(): Quaternion {
    return new Quaternion(0, 0, 0, 1);
  }

  /** Create a quaternion from Euler angles (in radians, XYZ order). */
  static fromEuler(x: number, y: number, z: number): Quaternion {
    const cx = Math.cos(x * 0.5), sx = Math.sin(x * 0.5);
    const cy = Math.cos(y * 0.5), sy = Math.sin(y * 0.5);
    const cz = Math.cos(z * 0.5), sz = Math.sin(z * 0.5);
    return new Quaternion(
      sx * cy * cz + cx * sy * sz,
      cx * sy * cz - sx * cy * sz,
      cx * cy * sz + sx * sy * cz,
      cx * cy * cz - sx * sy * sz,
    );
  }

  /** Hamilton product: this × other. */
  multiply(other: Quaternion): Quaternion {
    const { x: ax, y: ay, z: az, w: aw } = this;
    const { x: bx, y: by, z: bz, w: bw } = other;
    return new Quaternion(
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
    );
  }

  /** Spherical linear interpolation between this and target at t ∈ [0,1]. */
  slerp(target: Quaternion, t: number): Quaternion {
    let dot = this.x * target.x + this.y * target.y + this.z * target.z + this.w * target.w;

    // Ensure shortest path
    const t2 = dot < 0 ? -1 : 1;
    const tx = target.x * t2, ty = target.y * t2, tz = target.z * t2, tw = target.w * t2;
    dot = Math.abs(dot);

    if (dot > 0.9995) {
      // Linear interp for nearly identical quaternions
      return new Quaternion(
        this.x + t * (tx - this.x),
        this.y + t * (ty - this.y),
        this.z + t * (tz - this.z),
        this.w + t * (tw - this.w),
      ).normalize();
    }

    const theta0 = Math.acos(dot);
    const theta = theta0 * t;
    const sinTheta = Math.sin(theta);
    const sinTheta0 = Math.sin(theta0);
    const s0 = Math.cos(theta) - dot * sinTheta / sinTheta0;
    const s1 = sinTheta / sinTheta0;

    return new Quaternion(
      s0 * this.x + s1 * tx,
      s0 * this.y + s1 * ty,
      s0 * this.z + s1 * tz,
      s0 * this.w + s1 * tw,
    );
  }

  normalize(): Quaternion {
    const len = Math.sqrt(this.x ** 2 + this.y ** 2 + this.z ** 2 + this.w ** 2);
    if (len === 0) return Quaternion.identity();
    return new Quaternion(this.x / len, this.y / len, this.z / len, this.w / len);
  }

  conjugate(): Quaternion {
    return new Quaternion(-this.x, -this.y, -this.z, this.w);
  }

  toArray(): [number, number, number, number] {
    return [this.x, this.y, this.z, this.w];
  }

  clone(): Quaternion {
    return new Quaternion(this.x, this.y, this.z, this.w);
  }
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

export const Interpolation = (() => {

  /** Linear interpolation between a and b at t ∈ [0,1]. */
  function linear(a: number, b: number, t: number): number {
    return a + (b - a) * t;
  }

  /** Smoothstep — smooth start and end. */
  function smoothstep(a: number, b: number, t: number): number {
    const x = Math.min(Math.max((t - 0) / (1 - 0), 0), 1);
    const s = x * x * (3 - 2 * x);
    return a + (b - a) * s;
  }

  /**
   * Hermite interpolation between p1 and p2 with tangents m1 and m2 at t ∈ [0,1].
   */
  function hermite(p0: number, m0: number, p1: number, m1: number, t: number): number {
    const t2 = t * t;
    const t3 = t2 * t;
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
  }

  /**
   * Catmull-Rom spline through four control points p0–p3 at t ∈ [0,1].
   * Interpolates between p1 and p2.
   */
  function catmullRom(p0: number, p1: number, p2: number, p3: number, t: number): number {
    const t2 = t * t;
    const t3 = t2 * t;
    return 0.5 * (
      (2 * p1) +
      (-p0 + p2) * t +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
      (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    );
  }

  /**
   * Cubic Bézier through p0–p3 at t ∈ [0,1].
   */
  function bezier(p0: number, p1: number, p2: number, p3: number, t: number): number {
    const mt = 1 - t;
    return (
      mt * mt * mt * p0 +
      3 * mt * mt * t * p1 +
      3 * mt * t * t * p2 +
      t * t * t * p3
    );
  }

  /**
   * Evaluate a cubic Bézier *curve* (2D control points) at parameter t,
   * returning an [x, y] pair.
   */
  function bezier2D(
    p0: [number, number],
    p1: [number, number],
    p2: [number, number],
    p3: [number, number],
    t: number,
  ): [number, number] {
    return [
      bezier(p0[0], p1[0], p2[0], p3[0], t),
      bezier(p0[1], p1[1], p2[1], p3[1], t),
    ];
  }

  /** Clamp a value between min and max. */
  function clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
  }

  /** Remap a value from [inMin, inMax] to [outMin, outMax]. */
  function remap(value: number, inMin: number, inMax: number, outMin: number, outMax: number): number {
    const t = (value - inMin) / (inMax - inMin);
    return outMin + t * (outMax - outMin);
  }

  return { linear, smoothstep, hermite, catmullRom, bezier, bezier2D, clamp, remap };
})();
