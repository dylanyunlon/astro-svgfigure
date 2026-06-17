/**
 * SplineEmitter.ts — Spline-based Particle Emitter
 *
 * Active Theory 风格：沿样条线轨迹发射粒子，实现 AT 的 SplineParticleLife class。
 * 粒子沿 Catmull-Rom 样条流动，life cycle 由 uSplineSpeed 控制。
 *
 * M151: CatmullRom + weight-scaled particles
 *   1) 从 points 数组构建 Catmull-Rom 样条（参考 webgl2-particles TF 架构）
 *   2) 弧长参数化表实现均匀采样，N 个粒子沿样条等距分布
 *   3) 粒子 size = edge.weight * baseSizeFactor
 *
 * 参数来源: channels/physics/at_uil_params.json
 *
 *   am_SplineParticleLife_Element_0_WorkDetailParticles:
 *     uSplineSpeed      = [0.82, 1.21]  ← 每个粒子随机speed范围
 *     uStartOffset      = 1             ← 初始沿样条偏移
 *     uStartSpacing     = 0             ← 初始间距
 *     uMaxSDelay        = 0             ← 最大spawn delay
 *     uFlowRange        = [1, 1]        ← 流动范围
 *     uTimeMultiplier   = 0.17          ← 全局时间乘数
 *
 *   INPUT_Element_0_WorkDetailParticlesSplineConfig:
 *     json    = "assets/geometry/work/splines_anim4-SPLINES.json"
 *     infinite = true                   ← 无限循环
 *
 * 架构参考:
 *   channels/physics/at_uil_params.json  INPUT_P_Element_0_WorkDetailParticlescode_1_preset = "spline"
 *   src/lib/gpgpu/constraint-texture.ts (texture init pattern)
 *   ParticleSystem.ts (integration point)
 *   upstream/RESEARCH_122_webgl2_particles_clean.md — TF ping-pong 机制
 */

import { ParticleSystem } from './ParticleSystem.js';

// ── Types ─────────────────────────────────────────────────────────────────────

/** A 3-D control point on the spline */
export interface SplinePoint {
  x: number;
  y: number;
  z: number;
}

/** JSON format from AT's splines_anim4-SPLINES.json */
export interface SplineJSON {
  splines: Array<{
    points: Array<{ x: number; y: number; z: number }>;
  }>;
}

/**
 * Edge descriptor for weight-scaled particle sizing.
 * Each edge owns a spline; its weight controls particle size along that spline.
 */
export interface SplineEdge {
  /** Control points defining the Catmull-Rom spline path */
  points: SplinePoint[];
  /** Edge weight — multiplied by baseSizeFactor to determine particle size */
  weight: number;
}

export interface SplineEmitterConfig {
  /**
   * AT参数: uSplineSpeed = [0.82, 1.21]
   * Each particle gets a random speed in [min, max] along the spline.
   */
  splineSpeed?: [number, number];
  /**
   * AT参数: uTimeMultiplier = 0.17
   * Global time multiplier for all particle travel.
   */
  timeMultiplier?: number;
  /**
   * AT参数: uStartOffset = 1
   * Initial position offset along spline [0,1].
   */
  startOffset?: number;
  /**
   * AT参数: uStartSpacing = 0
   * Additional spacing between particles at spawn.
   */
  startSpacing?: number;
  /**
   * AT参数: uMaxSDelay = 0
   * Maximum random spawn delay in seconds.
   */
  maxSDelay?: number;
  /**
   * AT参数: uFlowRange = [1, 1]
   * Range multiplier for flow direction [min, max].
   */
  flowRange?: [number, number];
  /**
   * AT参数: SplineConfig.infinite = true
   * When true, particles loop back to spline start on completion.
   */
  infinite?: boolean;
  /**
   * AT参数: uThicknessSpeed = 1
   * Speed at which thickness variation animates.
   */
  thicknessSpeed?: number;
  /**
   * Base size factor multiplied by edge.weight to compute per-particle size.
   * size = edge.weight * baseSizeFactor.   Default: 1.0
   */
  baseSizeFactor?: number;
  /**
   * Number of linear segments used to approximate arc length per spline
   * segment during arc-length table construction. Higher = more accurate
   * uniform sampling at the cost of init time. Default: 64
   */
  arcLengthDivisions?: number;
}

export interface SplineParticleState {
  /** Current travel position [0, 1] along spline (arc-length parameterised) */
  travel: number;
  /** Individual speed scalar [uSplineSpeed.min, uSplineSpeed.max] */
  speed: number;
  /** Spawn delay remaining */
  delay: number;
  /** Which spline index this particle follows */
  splineIndex: number;
}

// ── AT parameter defaults ────────────────────────────────────────────────────

const AT_SPLINE_DEFAULTS: Required<SplineEmitterConfig> = {
  splineSpeed:        [0.82, 1.21],  // am_SplineParticleLife uSplineSpeed
  timeMultiplier:     0.17,          // am_SplineParticleLife uTimeMultiplier
  startOffset:        1,             // am_SplineParticleLife uStartOffset
  startSpacing:       0,             // am_SplineParticleLife uStartSpacing
  maxSDelay:          0,             // am_SplineParticleLife uMaxSDelay
  flowRange:          [1, 1],        // am_SplineParticleLife uFlowRange
  infinite:           true,          // SplineConfig.infinite
  thicknessSpeed:     1,             // am_ProtonAntimatter uThicknessSpeed
  baseSizeFactor:     1.0,           // weight→size multiplier
  arcLengthDivisions: 64,            // arc-length LUT resolution per segment
};

// ── Catmull-Rom evaluation ───────────────────────────────────────────────────

function catmullRom(
  p0: SplinePoint, p1: SplinePoint,
  p2: SplinePoint, p3: SplinePoint,
  t: number,
): SplinePoint {
  const t2 = t * t;
  const t3 = t2 * t;
  const f1 = -0.5 * t3 + t2 - 0.5 * t;
  const f2 =  1.5 * t3 - 2.5 * t2 + 1.0;
  const f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
  const f4 =  0.5 * t3 - 0.5 * t2;
  return {
    x: f1*p0.x + f2*p1.x + f3*p2.x + f4*p3.x,
    y: f1*p0.y + f2*p1.y + f3*p2.y + f4*p3.y,
    z: f1*p0.z + f2*p1.z + f3*p2.z + f4*p3.z,
  };
}

/**
 * Evaluate a Catmull-Rom spline at normalised position t ∈ [0, 1].
 * Clamps or wraps depending on `infinite`.
 */
function evalSpline(
  points: SplinePoint[],
  t: number,
  infinite: boolean,
): SplinePoint {
  const n = points.length;
  if (n < 2) return points[0] ?? { x: 0, y: 0, z: 0 };

  const scaled = t * (n - 1);
  const i1     = Math.floor(scaled);
  const localT = scaled - i1;

  const idx = (i: number) =>
    infinite ? ((i % n) + n) % n : Math.max(0, Math.min(n - 1, i));

  return catmullRom(
    points[idx(i1 - 1)],
    points[idx(i1)],
    points[idx(i1 + 1)],
    points[idx(i1 + 2)],
    localT,
  );
}

// ── Arc-length parameterisation ─────────────────────────────────────────────

/**
 * Pre-computed arc-length look-up table for a single spline.
 *
 * Maps uniform arc-length fraction u ∈ [0,1] → parametric t ∈ [0,1]
 * so that equal increments of u produce equal distance steps along the curve.
 *
 * Construction follows the standard approach used by webgl2-particles and
 * THREE.CurvePath: sample the curve at many fine parametric steps, accumulate
 * chord-length, then normalise to [0,1].
 */
interface ArcLengthTable {
  /** Cumulative normalised arc-lengths, length = divisions + 1, range [0, 1] */
  arcLengths: Float32Array;
  /** Total arc length of the spline in world units */
  totalLength: number;
}

/**
 * Build an arc-length LUT for a Catmull-Rom spline.
 *
 * @param points     Control points array
 * @param divisions  Total number of linear segments to approximate the curve
 * @param infinite   Whether the spline wraps
 */
function buildArcLengthTable(
  points: SplinePoint[],
  divisions: number,
  infinite: boolean,
): ArcLengthTable {
  const arcLengths = new Float32Array(divisions + 1);
  arcLengths[0] = 0;

  let prev = evalSpline(points, 0, infinite);
  let cumulative = 0;

  for (let i = 1; i <= divisions; i++) {
    const t = i / divisions;
    const cur = evalSpline(points, t, infinite);
    const dx = cur.x - prev.x;
    const dy = cur.y - prev.y;
    const dz = cur.z - prev.z;
    cumulative += Math.sqrt(dx * dx + dy * dy + dz * dz);
    arcLengths[i] = cumulative;
    prev = cur;
  }

  // Normalise to [0, 1]
  const totalLength = cumulative;
  if (totalLength > 0) {
    for (let i = 1; i <= divisions; i++) {
      arcLengths[i] /= totalLength;
    }
  }

  return { arcLengths, totalLength };
}

/**
 * Map a uniform arc-length fraction u ∈ [0,1] to parametric t ∈ [0,1]
 * using binary search on the LUT, then linear interpolation.
 */
function arcLengthToParametric(u: number, table: ArcLengthTable): number {
  const { arcLengths } = table;
  const n = arcLengths.length; // divisions + 1

  // Edge cases
  if (u <= 0) return 0;
  if (u >= 1) return 1;

  // Binary search for the segment containing u
  let lo = 0;
  let hi = n - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (arcLengths[mid] < u) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  // Linear interpolation within the found segment
  const segLen = arcLengths[hi] - arcLengths[lo];
  const frac = segLen > 0 ? (u - arcLengths[lo]) / segLen : 0;
  return (lo + frac) / (n - 1);
}

// ── SplineEmitter ────────────────────────────────────────────────────────────

export class SplineEmitter {
  private config: Required<SplineEmitterConfig>;
  private splines: SplinePoint[][] = [];
  private arcTables: ArcLengthTable[] = [];
  private edgeWeights: number[] = [];
  private states:  SplineParticleState[] = [];
  private positions: Float32Array;   // flat xyz per particle
  private sizes: Float32Array;       // per-particle size (weight * baseSizeFactor)
  readonly particleCount: number;

  /**
   * @param particleCount  Must match the ParticleSystem this feeds.
   * @param config         AT SplineParticleLife parameters.
   */
  constructor(particleCount: number, config: SplineEmitterConfig = {}) {
    this.particleCount = particleCount;
    this.config = { ...AT_SPLINE_DEFAULTS, ...config };
    this.positions = new Float32Array(particleCount * 3);
    this.sizes = new Float32Array(particleCount);

    console.log(
      `[SplineEmitter] ${particleCount} particles ` +
      `splineSpeed=[${this.config.splineSpeed}] ` +
      `timeMultiplier=${this.config.timeMultiplier} ` +
      `infinite=${this.config.infinite} ` +
      `baseSizeFactor=${this.config.baseSizeFactor}`,
    );
  }

  // ── Spline loading ───────────────────────────────────────────────────────

  /**
   * Load spline geometry from AT's JSON format
   * (assets/geometry/work/splines_anim4-SPLINES.json).
   * All splines receive a default weight of 1.0.
   */
  loadFromJSON(json: SplineJSON): void {
    this.splines = json.splines.map(s => s.points);
    this.edgeWeights = this.splines.map(() => 1.0);
    this._buildArcTables();
    this._initParticleStates();
    console.log(`[SplineEmitter] Loaded ${this.splines.length} splines`);
  }

  /**
   * Set splines programmatically (array of point arrays).
   * All splines receive a default weight of 1.0.
   */
  setSplines(splines: SplinePoint[][]): void {
    this.splines = splines;
    this.edgeWeights = splines.map(() => 1.0);
    this._buildArcTables();
    this._initParticleStates();
  }

  /**
   * Set splines from edge descriptors with per-edge weight.
   * Particles on each edge are sized by edge.weight * baseSizeFactor.
   *
   * Each edge's points array is used to construct a Catmull-Rom spline.
   * Particles are uniformly distributed along each spline via arc-length
   * parameterisation.
   */
  setEdges(edges: SplineEdge[]): void {
    this.splines = edges.map(e => e.points);
    this.edgeWeights = edges.map(e => e.weight);
    this._buildArcTables();
    this._initParticleStates();
    console.log(
      `[SplineEmitter] setEdges: ${edges.length} edges, ` +
      `weights=[${this.edgeWeights.map(w => w.toFixed(2)).join(', ')}]`,
    );
  }

  /**
   * Build arc-length LUTs for all loaded splines.
   * Called automatically by loadFromJSON / setSplines / setEdges.
   */
  private _buildArcTables(): void {
    const { arcLengthDivisions, infinite } = this.config;
    const totalSegments = this.splines.reduce(
      (sum, pts) => sum + Math.max(0, pts.length - 1), 0,
    );
    // Scale divisions by segment count so longer splines get finer resolution
    this.arcTables = this.splines.map(pts => {
      const segments = Math.max(1, pts.length - 1);
      const divs = arcLengthDivisions * segments;
      return buildArcLengthTable(pts, divs, infinite);
    });

    if (totalSegments > 0) {
      const lengths = this.arcTables.map(t => t.totalLength.toFixed(2));
      console.log(
        `[SplineEmitter] Arc-length tables built, ` +
        `lengths=[${lengths.join(', ')}]`,
      );
    }
  }

  private _initParticleStates(): void {
    const { splineSpeed, startOffset, startSpacing, maxSDelay } = this.config;
    const { baseSizeFactor } = this.config;
    const n = this.splines.length;

    this.states = Array.from({ length: this.particleCount }, (_, i) => {
      // AT: uSplineSpeed = [0.82, 1.21] — random speed per particle
      const speed = splineSpeed[0] + Math.random() * (splineSpeed[1] - splineSpeed[0]);

      const splineIndex = n > 0 ? i % n : 0;

      // Uniform distribution: divide particles equally among splines,
      // then space them evenly along each spline via arc-length fraction.
      const particlesOnThisSpline = Math.ceil(this.particleCount / Math.max(1, n));
      const localIdx = Math.floor(i / Math.max(1, n));
      const uniformU = particlesOnThisSpline > 1
        ? localIdx / (particlesOnThisSpline - 1)
        : 0.5;

      // Combine uniform arc-length position with AT offset/spacing
      const travel = ((startOffset * uniformU + i * startSpacing) % 1 + 1) % 1;

      // AT: uMaxSDelay = 0
      const delay = Math.random() * maxSDelay;

      return { travel, speed, delay, splineIndex };
    });

    // Compute per-particle sizes: size = edge.weight * baseSizeFactor
    for (let i = 0; i < this.particleCount; i++) {
      const si = this.states[i].splineIndex;
      const weight = si < this.edgeWeights.length ? this.edgeWeights[si] : 1.0;
      this.sizes[i] = weight * baseSizeFactor;
    }

    // Place particles at their initial uniform positions along each spline
    this._sampleUniformPositions();
  }

  /**
   * Place all particles at their current travel positions using arc-length
   * parameterisation so they are evenly spaced along each spline.
   */
  private _sampleUniformPositions(): void {
    const { infinite } = this.config;

    for (let i = 0; i < this.particleCount; i++) {
      const state = this.states[i];
      const spline = this.splines[state.splineIndex];
      if (!spline || spline.length < 2) continue;

      const table = this.arcTables[state.splineIndex];
      // Map uniform arc-length u → parametric t
      const t = table
        ? arcLengthToParametric(state.travel, table)
        : state.travel;

      const pos = evalSpline(spline, t, infinite);
      const base = i * 3;
      this.positions[base + 0] = pos.x;
      this.positions[base + 1] = pos.y;
      this.positions[base + 2] = pos.z;
    }
  }

  // ── Update ───────────────────────────────────────────────────────────────

  /**
   * Advance all particle positions along their splines.
   * Call this each frame before uploading to GPU texture.
   *
   * Travel is in arc-length space so velocity produces uniform speed
   * regardless of control-point density.
   *
   * @param delta  Frame delta normalised to 60fps (AT's HZ).
   */
  update(delta: number): void {
    if (this.splines.length === 0) return;

    const { timeMultiplier, infinite, flowRange } = this.config;
    const dt = delta * timeMultiplier;

    for (let i = 0; i < this.particleCount; i++) {
      const state = this.states[i];

      // Wait out spawn delay
      if (state.delay > 0) {
        state.delay -= delta * 0.016; // ~seconds
        continue;
      }

      // AT: travel advances by speed * timeMultiplier * flowRange
      const flow = flowRange[0] + Math.random() * (flowRange[1] - flowRange[0]);
      state.travel += state.speed * dt * flow * 0.01;

      if (infinite) {
        // AT: SplineConfig.infinite = true → wrap
        state.travel = state.travel % 1;
      } else {
        if (state.travel >= 1) {
          state.travel = 1;
        }
      }

      const spline = this.splines[state.splineIndex];
      if (!spline || spline.length < 2) continue;

      // Arc-length → parametric mapping for uniform spacing
      const table = this.arcTables[state.splineIndex];
      const t = table
        ? arcLengthToParametric(state.travel, table)
        : state.travel;

      const pos = evalSpline(spline, t, infinite);
      const base = i * 3;
      this.positions[base + 0] = pos.x;
      this.positions[base + 1] = pos.y;
      this.positions[base + 2] = pos.z;
    }
  }

  // ── GPU upload ───────────────────────────────────────────────────────────

  /**
   * Upload current positions to the ParticleSystem's position texture.
   * Call after update() each frame.
   *
   * This replaces the GPGPU-computed positions for particles that are
   * spline-controlled. The AT shader code:
   *   float travel = texture2D(tLife, vUv).z;
   *   vec3 target  = getSplinePos(travel);
   *   pos += (target - pos) * 0.07 * HZ;
   */
  uploadToGPU(
    gl: WebGL2RenderingContext,
    particleSystem: ParticleSystem,
  ): void {
    // Read back current position texture, blend in spline positions,
    // then write back. In production AT uses a dedicated spline texture
    // sampled in the update shader — this is the CPU-side approximation.
    const posTexture = particleSystem.positionTexture;
    const side = Math.ceil(Math.sqrt(this.particleCount));

    // Allocate on first call (cached in production)
    const data = new Float32Array(side * side * 4);

    for (let i = 0; i < this.particleCount; i++) {
      const base = i * 3;
      data[i * 4 + 0] = this.positions[base + 0];
      data[i * 4 + 1] = this.positions[base + 1];
      data[i * 4 + 2] = this.positions[base + 2];
      data[i * 4 + 3] = 1.0 - this.states[i].travel; // life = remaining travel
    }

    gl.bindTexture(gl.TEXTURE_2D, posTexture);
    gl.texSubImage2D(
      gl.TEXTURE_2D, 0,
      0, 0, side, side,
      gl.RGBA, gl.FLOAT, data,
    );
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ── Accessors ────────────────────────────────────────────────────────────

  /** Current particle positions as flat Float32Array [x,y,z, x,y,z, ...] */
  get positionArray(): Float32Array { return this.positions; }

  /** Per-particle sizes: size[i] = edge.weight * baseSizeFactor */
  get sizeArray(): Float32Array { return this.sizes; }

  /** Get size for particle i (edge.weight * baseSizeFactor) */
  getParticleSize(i: number): number { return this.sizes[i] ?? 0; }

  /** Travel progress for particle i ∈ [0, 1] */
  getTravelProgress(i: number): number { return this.states[i]?.travel ?? 0; }

  /** Arc-length of spline at given index (world units) */
  getSplineLength(splineIndex: number): number {
    return this.arcTables[splineIndex]?.totalLength ?? 0;
  }

  /** AT: uSplineSpeed range */
  get splineSpeed(): [number, number] { return this.config.splineSpeed; }

  /** AT: uTimeMultiplier */
  get timeMultiplier(): number { return this.config.timeMultiplier; }

  /** AT: uThicknessSpeed */
  get thicknessSpeed(): number { return this.config.thicknessSpeed; }

  /** Current base size factor */
  get baseSizeFactor(): number { return this.config.baseSizeFactor; }

  /** Update AT params at runtime */
  setSplineSpeed(min: number, max: number): void {
    this.config.splineSpeed = [min, max];
    // Re-randomise speeds for living particles
    for (const state of this.states) {
      state.speed = min + Math.random() * (max - min);
    }
  }

  setTimeMultiplier(v: number): void { this.config.timeMultiplier = v; }
  setThicknessSpeed(v: number): void { this.config.thicknessSpeed = v; }

  /** Update base size factor and recompute all particle sizes */
  setBaseSizeFactor(v: number): void {
    this.config.baseSizeFactor = v;
    for (let i = 0; i < this.particleCount; i++) {
      const si = this.states[i]?.splineIndex ?? 0;
      const weight = si < this.edgeWeights.length ? this.edgeWeights[si] : 1.0;
      this.sizes[i] = weight * v;
    }
  }
}
