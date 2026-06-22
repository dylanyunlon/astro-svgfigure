// src/lib/sph/differential-growth.ts
//
// Differential growth simulation producing organic fractal folds:
// coral reefs, cerebral cortex convolutions, romanesco cauliflower.
//
// Algorithm based on jasonwebb/2d-differential-growth-experiments (★266).
// https://github.com/jasonwebb/2d-differential-growth-experiments
//
// Physics per tick:
//   1. Attraction  – spring force pulling each node toward its two neighbours
//   2. Repulsion   – soft-body pushback from all nearby nodes (spatial hash)
//   3. Alignment   – node smoothly drifts toward midpoint of neighbours
//   4. Random nudge– tiny stochastic perturbation to break symmetry
//   5. Split       – insert midpoint node when edge length > maxEdgeLen
//   6. Prune       – remove node when edge length < minEdgeLen
//
// Output: getPath() → Float32Array [x0,y0, x1,y1, …, x0,y0]
// The last point equals the first, making it a closed line-strip ready for
// WebGPU PRIMITIVE_TOPOLOGY_LINE_STRIP.
//
// Typical usage:
//   const sim = new DifferentialGrowth({ width: 800, height: 600 });
//   sim.initCircle(400, 300, 80, 64);
//   requestAnimationFrame(function loop() {
//     sim.tick();
//     renderLineStrip(sim.getPath());        // Float32Array → WebGPU buffer
//     requestAnimationFrame(loop);
//   });

// ─── Tuning knobs ─────────────────────────────────────────────────────────────

export interface DifferentialGrowthConfig {
  /** Domain width – used for spatial hash bucketing only.  Default 800. */
  width?: number;
  /** Domain height. Default 600. */
  height?: number;

  // ── Force weights ──────────────────────────────────────────────────────────
  /** Spring attraction strength pulling node toward neighbours. Default 0.4 */
  attractionWeight?: number;
  /** Repulsion strength pushing nodes apart. Default 0.8 */
  repulsionWeight?: number;
  /** Alignment weight – drift toward neighbour midpoint. Default 0.45 */
  alignmentWeight?: number;
  /** Amplitude of random noise applied each tick. Default 0.15 */
  noiseStrength?: number;

  // ── Topology thresholds ────────────────────────────────────────────────────
  /** Split edge when length exceeds this value (pixels). Default 12 */
  maxEdgeLen?: number;
  /** Prune edge when length falls below this value (pixels). Default 4 */
  minEdgeLen?: number;
  /** Repulsion radius – nodes further apart are ignored. Default 18 */
  repulsionRadius?: number;

  // ── Integration ────────────────────────────────────────────────────────────
  /** Euler step size. Default 0.5 */
  stepSize?: number;
  /** Maximum nodes before split insertion is suspended. Default 8000 */
  maxNodes?: number;
}

// ─── Internal node ───────────────────────────────────────────────────────────

interface Node {
  x: number;
  y: number;
  // accumulated force for this tick (reset each frame)
  fx: number;
  fy: number;
  // prev / next indices in the circular linked list
  prev: number;
  next: number;
}

// ─── Spatial hash for O(1) neighbourhood queries ─────────────────────────────

class SpatialHash {
  private cellSize: number;
  private cols: number;
  private rows: number;
  private cells: Map<number, number[]>;

  constructor(cellSize: number, width: number, height: number) {
    this.cellSize = cellSize;
    this.cols = Math.ceil(width / cellSize) + 1;
    this.rows = Math.ceil(height / cellSize) + 1;
    this.cells = new Map();
  }

  private key(cx: number, cy: number): number {
    return cy * this.cols + cx;
  }

  private cellOf(x: number, y: number): [number, number] {
    return [
      Math.floor(x / this.cellSize),
      Math.floor(y / this.cellSize),
    ];
  }

  clear(): void {
    this.cells.clear();
  }

  insert(idx: number, x: number, y: number): void {
    const [cx, cy] = this.cellOf(x, y);
    const k = this.key(cx, cy);
    let bucket = this.cells.get(k);
    if (!bucket) {
      bucket = [];
      this.cells.set(k, bucket);
    }
    bucket.push(idx);
  }

  /** Yields all node indices within `radius` of (qx, qy). */
  *query(qx: number, qy: number, radius: number): Generator<number> {
    const r = Math.ceil(radius / this.cellSize);
    const [cx0, cy0] = this.cellOf(qx, qy);
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const k = this.key(cx0 + dx, cy0 + dy);
        const bucket = this.cells.get(k);
        if (bucket) {
          for (const idx of bucket) yield idx;
        }
      }
    }
  }
}

// ─── DifferentialGrowth ───────────────────────────────────────────────────────

export class DifferentialGrowth {
  // resolved config
  private readonly cfg: Required<DifferentialGrowthConfig>;

  // node pool – dense array, gaps marked with sentinel next === -2
  private pool: Node[] = [];
  // head of the circular list (any alive node index)
  private head = -1;
  // count of alive nodes
  private count = 0;

  // spatial hash rebuilt each tick
  private readonly hash: SpatialHash;

  // seeded LCG random for deterministic but varied perturbation
  private seed = 0x9e3779b9;

  constructor(cfg: DifferentialGrowthConfig = {}) {
    this.cfg = {
      width:             cfg.width             ?? 800,
      height:            cfg.height            ?? 600,
      attractionWeight:  cfg.attractionWeight  ?? 0.4,
      repulsionWeight:   cfg.repulsionWeight   ?? 0.8,
      alignmentWeight:   cfg.alignmentWeight   ?? 0.45,
      noiseStrength:     cfg.noiseStrength     ?? 0.15,
      maxEdgeLen:        cfg.maxEdgeLen        ?? 12,
      minEdgeLen:        cfg.minEdgeLen        ?? 4,
      repulsionRadius:   cfg.repulsionRadius   ?? 18,
      stepSize:          cfg.stepSize          ?? 0.5,
      maxNodes:          cfg.maxNodes          ?? 8000,
    };
    this.hash = new SpatialHash(
      this.cfg.repulsionRadius,
      this.cfg.width,
      this.cfg.height,
    );
  }

  // ── LCG random [-1, 1] ─────────────────────────────────────────────────────

  private rand(): number {
    this.seed = (Math.imul(this.seed, 1664525) + 1013904223) >>> 0;
    return (this.seed / 0x80000000) - 1.0;
  }

  // ── Node allocation helpers ────────────────────────────────────────────────

  /** Allocate a new node (recycle pool slot or push). Returns new index. */
  private alloc(x: number, y: number): number {
    // try to find a recycled slot
    for (let i = 0; i < this.pool.length; i++) {
      if (this.pool[i].next === -2) {
        const n = this.pool[i];
        n.x = x; n.y = y; n.fx = 0; n.fy = 0; n.prev = -1; n.next = -1;
        this.count++;
        return i;
      }
    }
    this.pool.push({ x, y, fx: 0, fy: 0, prev: -1, next: -1 });
    this.count++;
    return this.pool.length - 1;
  }

  /** Remove node at index `i` from circular list and free it. */
  private free(i: number): void {
    const n = this.pool[i];
    const p = this.pool[n.prev];
    const nx = this.pool[n.next];
    p.next = n.next;
    nx.prev = n.prev;
    if (this.head === i) this.head = n.next;
    n.next = -2; // sentinel: slot is free
    this.count--;
  }

  // ── Public: add a single node at the end of the curve ─────────────────────

  /** Append a node to the curve (order of calls defines initial topology). */
  addNode(x: number, y: number): void {
    const idx = this.alloc(x, y);
    if (this.head === -1) {
      // first node – self-loop
      this.pool[idx].prev = idx;
      this.pool[idx].next = idx;
      this.head = idx;
    } else {
      // insert before head (i.e. at the "end" of the loop)
      const tail = this.pool[this.head].prev;
      this.pool[tail].next = idx;
      this.pool[idx].prev = tail;
      this.pool[idx].next = this.head;
      this.pool[this.head].prev = idx;
    }
  }

  /** Convenience: initialise a regular polygon as the seed curve. */
  initCircle(cx: number, cy: number, radius: number, segments = 32): void {
    this.pool = [];
    this.head = -1;
    this.count = 0;
    for (let i = 0; i < segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      this.addNode(cx + Math.cos(a) * radius, cy + Math.sin(a) * radius);
    }
  }

  // ── Tick ──────────────────────────────────────────────────────────────────

  tick(): void {
    if (this.head === -1 || this.count < 2) return;

    this._buildHash();
    this._accumForces();
    this._integrate();
    this._splitEdges();
    this._pruneEdges();
  }

  // ── Private: spatial hash rebuild ─────────────────────────────────────────

  private _buildHash(): void {
    this.hash.clear();
    let i = this.head;
    do {
      const n = this.pool[i];
      this.hash.insert(i, n.x, n.y);
      i = n.next;
    } while (i !== this.head);
  }

  // ── Private: force accumulation ───────────────────────────────────────────

  private _accumForces(): void {
    const {
      attractionWeight: wa,
      repulsionWeight:  wr,
      alignmentWeight:  wl,
      noiseStrength:    wn,
      repulsionRadius,
      stepSize,
    } = this.cfg;

    const r2 = repulsionRadius * repulsionRadius;

    let i = this.head;
    do {
      const n   = this.pool[i];
      const prv = this.pool[n.prev];
      const nxt = this.pool[n.next];

      // reset
      n.fx = 0;
      n.fy = 0;

      // ── 1. Spring attraction toward neighbours ──────────────────────────
      n.fx += (prv.x - n.x) * wa;
      n.fy += (prv.y - n.y) * wa;
      n.fx += (nxt.x - n.x) * wa;
      n.fy += (nxt.y - n.y) * wa;

      // ── 2. Alignment toward midpoint of neighbours ──────────────────────
      const midX = (prv.x + nxt.x) * 0.5;
      const midY = (prv.y + nxt.y) * 0.5;
      n.fx += (midX - n.x) * wl;
      n.fy += (midY - n.y) * wl;

      // ── 3. Repulsion from nearby nodes ──────────────────────────────────
      for (const j of this.hash.query(n.x, n.y, repulsionRadius)) {
        if (j === i) continue;
        const other = this.pool[j];
        if (other.next === -2) continue; // freed
        const dx = n.x - other.x;
        const dy = n.y - other.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 1e-6 || d2 >= r2) continue;
        const d   = Math.sqrt(d2);
        const mag = (1.0 - d / repulsionRadius) * wr;
        n.fx += (dx / d) * mag;
        n.fy += (dy / d) * mag;
      }

      // ── 4. Random perturbation ───────────────────────────────────────────
      n.fx += this.rand() * wn;
      n.fy += this.rand() * wn;

      i = n.next;
    } while (i !== this.head);
  }

  // ── Private: Euler integration ────────────────────────────────────────────

  private _integrate(): void {
    const dt = this.cfg.stepSize;
    let i = this.head;
    do {
      const n = this.pool[i];
      n.x += n.fx * dt;
      n.y += n.fy * dt;
      i = n.next;
    } while (i !== this.head);
  }

  // ── Private: edge split ───────────────────────────────────────────────────

  private _splitEdges(): void {
    if (this.count >= this.cfg.maxNodes) return;
    const maxLen2 = this.cfg.maxEdgeLen * this.cfg.maxEdgeLen;

    // collect split candidates first (avoid mutating list during traversal)
    const toSplit: number[] = [];
    let i = this.head;
    do {
      const n   = this.pool[i];
      const nxt = this.pool[n.next];
      const dx  = nxt.x - n.x;
      const dy  = nxt.y - n.y;
      if (dx * dx + dy * dy > maxLen2) toSplit.push(i);
      i = n.next;
    } while (i !== this.head);

    for (const srcIdx of toSplit) {
      if (this.count >= this.cfg.maxNodes) break;
      const src  = this.pool[srcIdx];
      const dstI = src.next;
      const dst  = this.pool[dstI];

      // midpoint with tiny perpendicular nudge to break symmetry
      const mx = (src.x + dst.x) * 0.5;
      const my = (src.y + dst.y) * 0.5;
      const ex = dst.x - src.x;
      const ey = dst.y - src.y;
      const plen = Math.sqrt(ex * ex + ey * ey) || 1;
      const nudge = this.rand() * 0.5; // ±0.5 px perpendicular
      const nx = mx + (-ey / plen) * nudge;
      const ny = my + ( ex / plen) * nudge;

      // allocate and splice into the list: src → new → dst
      const newIdx  = this.alloc(nx, ny);
      const newNode = this.pool[newIdx];
      newNode.prev  = srcIdx;
      newNode.next  = dstI;
      src.next      = newIdx;
      dst.prev      = newIdx;

      // keep head valid
      if (this.head === -1) this.head = newIdx;
    }
  }

  // ── Private: edge prune ───────────────────────────────────────────────────

  private _pruneEdges(): void {
    if (this.count <= 4) return; // keep minimum viable loop
    const minLen2 = this.cfg.minEdgeLen * this.cfg.minEdgeLen;

    const toRemove: number[] = [];
    let i = this.head;
    do {
      const n   = this.pool[i];
      const nxt = this.pool[n.next];
      const dx  = nxt.x - n.x;
      const dy  = nxt.y - n.y;
      if (dx * dx + dy * dy < minLen2) toRemove.push(i);
      i = n.next;
    } while (i !== this.head);

    for (const idx of toRemove) {
      if (this.count <= 4) break;
      if (this.pool[idx].next === -2) continue; // already freed
      this.free(idx);
    }
  }

  // ── Public: output ────────────────────────────────────────────────────────

  /**
   * Returns interleaved [x0,y0, x1,y1, …, x0,y0] in a Float32Array.
   * The closing duplicate makes it a valid closed line-strip for WebGPU:
   *
   *   const arr = sim.getPath();
   *   device.queue.writeBuffer(gpuBuf, 0, arr);
   *   pass.draw(arr.length / 2);   // PRIMITIVE_TOPOLOGY_LINE_STRIP
   */
  getPath(): Float32Array {
    if (this.head === -1) return new Float32Array(0);
    // count alive nodes
    const n = this.count;
    // interleaved xy + closing point = (n + 1) * 2 floats
    const out = new Float32Array((n + 1) * 2);
    let ptr = 0;
    let i = this.head;
    do {
      const node = this.pool[i];
      out[ptr++] = node.x;
      out[ptr++] = node.y;
      i = node.next;
    } while (i !== this.head);
    // close the loop
    out[ptr++] = this.pool[this.head].x;
    out[ptr++] = this.pool[this.head].y;
    return out;
  }

  /** Number of alive nodes. */
  get nodeCount(): number {
    return this.count;
  }

  // ── Serialisation helpers (for debug / save-state) ────────────────────────

  /** Snapshot the current curve as a plain array of {x,y}. */
  snapshot(): Array<{ x: number; y: number }> {
    const result: Array<{ x: number; y: number }> = [];
    if (this.head === -1) return result;
    let i = this.head;
    do {
      const n = this.pool[i];
      result.push({ x: n.x, y: n.y });
      i = n.next;
    } while (i !== this.head);
    return result;
  }

  /** Restore from a snapshot (replaces current state). */
  restore(pts: Array<{ x: number; y: number }>): void {
    this.pool = [];
    this.head = -1;
    this.count = 0;
    for (const { x, y } of pts) this.addNode(x, y);
  }
}
