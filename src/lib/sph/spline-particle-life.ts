/**
 * spline-particle-life.ts — AT-style SplineParticleLife for Edge Topology
 *
 * M623: Ports Active Theory's SplineParticleLife.fs logic to TypeScript.
 *
 * Each topology edge becomes an emitter. Particles are born at the source cell,
 * flow along the edge's Catmull-Rom spline perturbed by curl noise, and on
 * arrival at the target cell they transition into free SPH particles absorbed
 * by that cell's fluid domain. This models Transformer data-flow:
 *
 *   token → edge spline → attention / FFN cell → SPH processing
 *
 * Lifecycle states (mirrors AT SplineParticleLife.fs):
 *   SPAWN   — particle waits out its uMaxSDelay countdown before moving
 *   FLOW    — particle travels along arc-length-parameterised spline,
 *             with per-particle speed ∈ [uSplineSpeed.min, uSplineSpeed.max]
 *             and curl-noise lateral displacement
 *   DECAY   — travel ≥ 1: alpha fades over uDecayRate frames; particle is
 *             handed to the target cell's SPH domain while still visible
 *   DEAD    — alpha = 0; slot is recycled (respawned or freed)
 *
 * Curl noise perturbation (AT simplenoise.glsl port):
 *   ∇×Ψ(x,y,time) evaluated on the CPU using a 3-D gradient noise function.
 *   The perturbation is applied perpendicular to the spline tangent, keeping
 *   particles roughly on-path while adding organic turbulence.
 *
 * UIL parameters (channels/physics/at_uil_params.json):
 *   uDecayRate        — opacity decay rate after reaching end  [0.01 – 1.0]
 *   uTimeMultiplier   — global time scale                      [0.1 – 2.0]
 *   uFlowRange        — [min, max] flow speed multiplier       e.g. [1, 1]
 *   uSplineSpeed      — [min, max] per-particle speed          e.g. [0.82, 1.21]
 *   uMaxSDelay        — max random spawn delay (seconds)       [0 – 2]
 *   uCurlNoiseScale   — spatial frequency of curl field        [0.5 – 8.0]
 *   uCurlNoiseSpeed   — temporal speed of curl field           [0.5 – 5.0]
 *   uCurlStrength     — lateral displacement amplitude          [0 – 0.3]
 *
 * Integration:
 *   const spl = new SplineParticleLife(edgeSplines, config);
 *   // render loop:
 *   spl.update(deltaSeconds, elapsedSeconds);
 *   // SPH handoff callback fires automatically; renderer reads spl.particles
 *
 * References:
 *   src/lib/particle/SplineEmitter.ts        — arc-length spline base
 *   src/lib/particle/CurlNoise.frag          — GLSL curl noise (ported here)
 *   src/lib/shaders/edge-spline.frag         — edge rendering
 *   src/lib/sph/SPHWorld.ts                  — addFluid() API for handoff
 *   src/lib/sph/curl-flow-field.ts           — GPU curl noise (CPU equiv here)
 *   channels/physics/at_uil_params.json      — UIL parameter source of truth
 */

// ─── Types ────────────────────────────────────────────────────────────────────

/** A 3-D control point */
export interface SplinePoint3 {
  x: number;
  y: number;
  z: number;
}

/**
 * One topology edge with its spline and source/target metadata.
 * Points are in world/domain units (matching SPHWorld domain coords).
 */
export interface EdgeSplineData {
  edgeId:   string;
  sourceId: string;
  targetId: string;
  /** Catmull-Rom control points in SPH domain space */
  points:   SplinePoint3[];
  /** Attention / connectivity weight → controls particle count & size */
  weight:   number;
}

/** Particle lifecycle phase */
export type ParticlePhase = 'spawn' | 'flow' | 'decay' | 'dead';

/** Per-particle CPU state */
export interface SplineParticle {
  /** Index into the EdgeSplineData array */
  edgeIndex:   number;
  /** Arc-length fraction [0, 1] along the spline */
  travel:      number;
  /** Individual speed scalar */
  speed:       number;
  /** Remaining spawn delay (seconds) */
  delay:       number;
  /** Current lifecycle phase */
  phase:       ParticlePhase;
  /** Opacity in [0, 1] — decays to 0 after reaching end */
  alpha:       number;
  /** World-space position (updated by update()) */
  x:           number;
  y:           number;
  z:           number;
  /** Curl-noise lateral offset, perpendicular to tangent */
  noiseOffset: number;
  /** Visual size derived from edge.weight */
  size:        number;
  /** Random seed for curl-noise phase variation */
  seed:        number;
}

/** UIL-driven parameters — all overridable at runtime */
export interface SplineParticleLifeConfig {
  /** [min, max] per-particle speed (AT: [0.82, 1.21]) */
  uSplineSpeed?:     [number, number];
  /** Global time multiplier (AT: 0.17) */
  uTimeMultiplier?:  number;
  /** [min, max] flow range multiplier (AT: [1, 1]) */
  uFlowRange?:       [number, number];
  /** Opacity decay rate per second after reaching end (AT: ~0.6) */
  uDecayRate?:       number;
  /** Max random spawn delay in seconds (AT: 0) */
  uMaxSDelay?:       number;
  /** Curl-noise spatial scale (AT: uCurlNoiseScale 2–5) */
  uCurlNoiseScale?:  number;
  /** Curl-noise temporal speed (AT: uCurlNoiseSpeed 5) */
  uCurlNoiseSpeed?:  number;
  /** Lateral displacement amplitude in domain units */
  uCurlStrength?:    number;
  /** Particles per unit weight on each edge */
  particlesPerUnit?: number;
  /** Arc-length LUT resolution per segment */
  arcLengthDivisions?: number;
  /**
   * Called when a particle enters DECAY and should be injected into
   * the target cell's SPH domain as a free particle.
   *
   * @param edgeId   The topology edge id
   * @param targetId The target cell id
   * @param x        World x in SPH domain units
   * @param y        World y in SPH domain units
   * @param vx       Estimated velocity x
   * @param vy       Estimated velocity y
   * @param species  Particle species tag (from edge weight / species mapping)
   */
  onHandoff?: (
    edgeId: string,
    targetId: string,
    x: number,
    y: number,
    vx: number,
    vy: number,
    species: number,
  ) => void;
}

// ─── Defaults (AT SplineParticleLife.fs source-of-truth values) ──────────────

const DEFAULTS: Required<Omit<SplineParticleLifeConfig, 'onHandoff'>> = {
  uSplineSpeed:        [0.82, 1.21],
  uTimeMultiplier:     0.17,
  uFlowRange:          [1.0, 1.0],
  uDecayRate:          0.6,
  uMaxSDelay:          0.0,
  uCurlNoiseScale:     2.0,
  uCurlNoiseSpeed:     5.0,
  uCurlStrength:       0.04,
  particlesPerUnit:    24,
  arcLengthDivisions:  64,
};

// ─── Catmull-Rom spline math ──────────────────────────────────────────────────

function catmullRom(
  p0: SplinePoint3, p1: SplinePoint3,
  p2: SplinePoint3, p3: SplinePoint3,
  t: number,
): SplinePoint3 {
  const t2 = t * t;
  const t3 = t2 * t;
  const f1 = -0.5 * t3 + t2 - 0.5 * t;
  const f2 =  1.5 * t3 - 2.5 * t2 + 1.0;
  const f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
  const f4 =  0.5 * t3 - 0.5 * t2;
  return {
    x: f1 * p0.x + f2 * p1.x + f3 * p2.x + f4 * p3.x,
    y: f1 * p0.y + f2 * p1.y + f3 * p2.y + f4 * p3.y,
    z: f1 * p0.z + f2 * p1.z + f3 * p2.z + f4 * p3.z,
  };
}

function evalCatmullRom(
  points: SplinePoint3[],
  t: number,
): SplinePoint3 {
  const n = points.length;
  if (n === 0) return { x: 0, y: 0, z: 0 };
  if (n === 1) return { ...points[0] };
  if (n === 2) {
    return {
      x: points[0].x + (points[1].x - points[0].x) * t,
      y: points[0].y + (points[1].y - points[0].y) * t,
      z: points[0].z + (points[1].z - points[0].z) * t,
    };
  }
  const scaled = Math.min(t, 0.9999) * (n - 1);
  const i1 = Math.floor(scaled);
  const localT = scaled - i1;
  const clamp = (i: number) => Math.max(0, Math.min(n - 1, i));
  return catmullRom(
    points[clamp(i1 - 1)],
    points[clamp(i1)],
    points[clamp(i1 + 1)],
    points[clamp(i1 + 2)],
    localT,
  );
}

/** Finite-difference tangent at t, normalised */
function splineTangent(points: SplinePoint3[], t: number): SplinePoint3 {
  const eps = 0.001;
  const a = evalCatmullRom(points, Math.max(0, t - eps));
  const b = evalCatmullRom(points, Math.min(1, t + eps));
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const dz = b.z - a.z;
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-10;
  return { x: dx / len, y: dy / len, z: dz / len };
}

// ─── Arc-length parameterisation ─────────────────────────────────────────────

interface ArcTable {
  /** Cumulative normalised arc-lengths, length = divisions+1, range [0,1] */
  lut: Float32Array;
  totalLength: number;
}

function buildArcTable(points: SplinePoint3[], divisions: number): ArcTable {
  const lut = new Float32Array(divisions + 1);
  lut[0] = 0;
  let prev = evalCatmullRom(points, 0);
  let cum = 0;
  for (let i = 1; i <= divisions; i++) {
    const t = i / divisions;
    const cur = evalCatmullRom(points, t);
    const dx = cur.x - prev.x;
    const dy = cur.y - prev.y;
    const dz = cur.z - prev.z;
    cum += Math.sqrt(dx * dx + dy * dy + dz * dz);
    lut[i] = cum;
    prev = cur;
  }
  const total = cum;
  if (total > 0) for (let i = 1; i <= divisions; i++) lut[i] /= total;
  return { lut, totalLength: total };
}

function arcToParam(u: number, table: ArcTable): number {
  const { lut } = table;
  const n = lut.length;
  if (u <= 0) return 0;
  if (u >= 1) return 1;
  let lo = 0, hi = n - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (lut[mid] < u) lo = mid; else hi = mid;
  }
  const segLen = lut[hi] - lut[lo];
  const frac = segLen > 0 ? (u - lut[lo]) / segLen : 0;
  return (lo + frac) / (n - 1);
}

// ─── CPU curl noise (port of AT simplenoise.glsl / CurlNoise.frag) ───────────
// Uses classic 3-D gradient noise + finite-difference curl.

/** 3-D hash — mirrors AT curl.glsl hash33 */
function hash33(px: number, py: number, pz: number): [number, number, number] {
  let ax = (px * 443.897 + py * 441.423 + pz * 437.195);
  let ay = (px * 441.423 + py * 437.195 + pz * 443.897);
  let az = (px * 437.195 + py * 443.897 + pz * 441.423);
  ax = (ax - Math.floor(ax));
  ay = (ay - Math.floor(ay));
  az = (az - Math.floor(az));
  const dp = ax * ay + ay * az + az * ax + 19.19;
  const hx = (ax + dp) * az;
  const hy = (ay + dp) * ax;
  const hz = (az + dp) * ay;
  return [hx - Math.floor(hx), hy - Math.floor(hy), hz - Math.floor(hz)];
}

/** Gradient noise on [0,1]³ */
function gradNoise3(px: number, py: number, pz: number): number {
  const ix = Math.floor(px), iy = Math.floor(py), iz = Math.floor(pz);
  const fx = px - ix, fy = py - iy, fz = pz - iz;
  // quintic smoothstep
  const ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10);
  const uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10);
  const uz = fz * fz * fz * (fz * (fz * 6 - 15) + 10);

  let result = 0;
  for (let dz = 0; dz <= 1; dz++) {
    for (let dy = 0; dy <= 1; dy++) {
      for (let dx = 0; dx <= 1; dx++) {
        const [gx, gy, gz] = hash33(ix + dx, iy + dy, iz + dz);
        // gradient dot product (map hash to [-1,1] gradients)
        const gdx = gx * 2 - 1;
        const gdy = gy * 2 - 1;
        const gdz = gz * 2 - 1;
        const dot = gdx * (fx - dx) + gdy * (fy - dy) + gdz * (fz - dz);
        // trilinear blend weights
        const wx = dx === 0 ? 1 - ux : ux;
        const wy = dy === 0 ? 1 - uy : uy;
        const wz = dz === 0 ? 1 - uz : uz;
        result += dot * wx * wy * wz;
      }
    }
  }
  return result;
}

const CURL_EPS = 0.01;

/**
 * 2-D curl-noise velocity at (x, y, time).
 * Computes curl of scalar potential ψ(x,y,t) = noise(x,y,t):
 *   F = (∂ψ/∂y, -∂ψ/∂x)
 * This gives a divergence-free planar flow field matching the GPU version
 * in CurlNoise.frag and curl-flow-field.ts.
 */
function curlNoise2D(
  x: number, y: number, time: number,
  scale: number, speed: number,
): [number, number] {
  const sx = x * scale;
  const sy = y * scale;
  const st = time * speed * 0.01;

  const psy_dy = (
    gradNoise3(sx, sy + CURL_EPS, st) -
    gradNoise3(sx, sy - CURL_EPS, st)
  ) / (2 * CURL_EPS);

  const psy_dx = (
    gradNoise3(sx + CURL_EPS, sy, st) -
    gradNoise3(sx - CURL_EPS, sy, st)
  ) / (2 * CURL_EPS);

  return [psy_dy, -psy_dx];
}

// ─── SplineParticleLife ───────────────────────────────────────────────────────

export class SplineParticleLife {
  /** All live particle states (read by renderer each frame) */
  readonly particles: SplineParticle[] = [];

  private edges:   EdgeSplineData[];
  private arcTabs: ArcTable[];
  private cfg:     Required<Omit<SplineParticleLifeConfig, 'onHandoff'>>;
  private onHandoff?: SplineParticleLifeConfig['onHandoff'];

  /** Elapsed simulation time in seconds (drives curl noise) */
  private elapsed = 0;

  /**
   * @param edges   Topology edge splines in SPH domain coordinates
   * @param config  UIL-driven parameters; all optional with AT defaults
   */
  constructor(edges: EdgeSplineData[], config: SplineParticleLifeConfig = {}) {
    this.edges = edges;
    this.cfg = { ...DEFAULTS, ...config } as Required<Omit<SplineParticleLifeConfig, 'onHandoff'>>;
    this.onHandoff = config.onHandoff;

    // Pre-compute arc-length tables for all edge splines
    this.arcTabs = edges.map(e =>
      buildArcTable(e.points, this.cfg.arcLengthDivisions * Math.max(1, e.points.length - 1))
    );

    // Allocate initial particle pool
    this._allocParticles();

    console.log(
      `[SplineParticleLife] ${edges.length} edges, ` +
      `${this.particles.length} particles allocated, ` +
      `uSplineSpeed=[${this.cfg.uSplineSpeed}] ` +
      `uTimeMultiplier=${this.cfg.uTimeMultiplier} ` +
      `uCurlStrength=${this.cfg.uCurlStrength}`
    );
  }

  // ── Particle allocation ───────────────────────────────────────────────────

  private _allocParticles(): void {
    this.particles.length = 0;
    const { uSplineSpeed, uMaxSDelay, particlesPerUnit } = this.cfg;

    for (let ei = 0; ei < this.edges.length; ei++) {
      const edge = this.edges[ei];
      // Number of particles scales with edge weight × particlesPerUnit
      const count = Math.max(1, Math.round(edge.weight * particlesPerUnit));

      for (let pi = 0; pi < count; pi++) {
        const speed =
          uSplineSpeed[0] +
          Math.random() * (uSplineSpeed[1] - uSplineSpeed[0]);

        // Uniformly stagger initial travel so particles are spread across spline
        const travel = pi / count;
        const delay = Math.random() * uMaxSDelay;

        // Evaluate starting position
        const { x, y, z } = this._evalParticlePos(ei, travel, 0, 0);

        this.particles.push({
          edgeIndex:   ei,
          travel,
          speed,
          delay,
          phase:       delay > 0 ? 'spawn' : 'flow',
          alpha:       delay > 0 ? 0 : 1,
          x, y, z,
          noiseOffset: 0,
          size:        edge.weight * 1.0,
          seed:        Math.random() * 1000,
        });
      }
    }
  }

  // ── Update ────────────────────────────────────────────────────────────────

  /**
   * Advance all particles one step.
   *
   * @param deltaSeconds  Wall-clock frame delta (capped to avoid spiral)
   */
  update(deltaSeconds: number): void {
    const dt = Math.min(deltaSeconds, 0.05);
    this.elapsed += dt;

    const {
      uTimeMultiplier, uFlowRange, uDecayRate,
      uCurlNoiseScale, uCurlNoiseSpeed, uCurlStrength,
    } = this.cfg;

    const scaledDt = dt * uTimeMultiplier;

    for (const p of this.particles) {
      switch (p.phase) {
        case 'dead':
          this._respawn(p);
          break;

        case 'spawn':
          p.delay -= dt;
          if (p.delay <= 0) {
            p.delay = 0;
            p.phase = 'flow';
            p.alpha = 1;
          }
          break;

        case 'flow': {
          // AT: travel += speed * timeMultiplier * flowRange * dt
          const flow =
            uFlowRange[0] +
            Math.random() * (uFlowRange[1] - uFlowRange[0]);
          p.travel += p.speed * scaledDt * flow * 0.01;

          // Curl-noise lateral perturbation
          const [cnx, cny] = curlNoise2D(
            p.x + p.seed * 0.1,
            p.y + p.seed * 0.1,
            this.elapsed,
            uCurlNoiseScale,
            uCurlNoiseSpeed,
          );
          // Rotate curl displacement to be perpendicular to tangent
          const edge = this.edges[p.edgeIndex];
          const tan = splineTangent(edge.points, Math.min(p.travel, 0.9999));
          const perpX = -tan.y;
          const perpY =  tan.x;
          const curlMag = (cnx * perpX + cny * perpY) * uCurlStrength;
          p.noiseOffset = curlMag;

          // Update world position
          const { x, y, z } = this._evalParticlePos(
            p.edgeIndex, Math.min(p.travel, 1.0),
            perpX * curlMag, perpY * curlMag,
          );
          p.x = x; p.y = y; p.z = z;

          // Transition to DECAY when spline end reached
          if (p.travel >= 1.0) {
            p.travel = 1.0;
            p.phase = 'decay';
            // Fire SPH handoff immediately as particle arrives
            this._fireHandoff(p);
          }
          break;
        }

        case 'decay':
          p.alpha -= uDecayRate * dt;
          if (p.alpha <= 0) {
            p.alpha = 0;
            p.phase = 'dead';
          }
          break;
      }
    }
  }

  // ── Position evaluation ───────────────────────────────────────────────────

  /**
   * Evaluate particle world position from arc-length travel + curl offset.
   */
  private _evalParticlePos(
    edgeIndex: number,
    travel:    number,
    offX:      number,
    offY:      number,
  ): SplinePoint3 {
    const edge = this.edges[edgeIndex];
    if (!edge || edge.points.length === 0) return { x: 0, y: 0, z: 0 };

    const table = this.arcTabs[edgeIndex];
    const t = table ? arcToParam(Math.min(travel, 1), table) : Math.min(travel, 1);
    const pos = evalCatmullRom(edge.points, t);
    return { x: pos.x + offX, y: pos.y + offY, z: pos.z };
  }

  // ── SPH handoff ───────────────────────────────────────────────────────────

  /**
   * Fire the handoff callback for a particle that just finished its spline.
   * The particle's instantaneous velocity is estimated from the spline tangent.
   */
  private _fireHandoff(p: SplineParticle): void {
    if (!this.onHandoff) return;
    const edge = this.edges[p.edgeIndex];
    const tan = splineTangent(edge.points, 0.999);
    // Velocity estimate: tangent * speed * timeMultiplier (domain units/sec)
    const vScale = p.speed * this.cfg.uTimeMultiplier * 0.01;
    this.onHandoff(
      edge.edgeId,
      edge.targetId,
      p.x, p.y,
      tan.x * vScale,
      tan.y * vScale,
      Math.round(edge.weight), // species tag from weight
    );
  }

  // ── Respawn ───────────────────────────────────────────────────────────────

  /**
   * Recycle a dead particle slot — reset it to the source end of its spline
   * with a new random speed and optional spawn delay.
   */
  private _respawn(p: SplineParticle): void {
    const { uSplineSpeed, uMaxSDelay } = this.cfg;
    p.travel = 0;
    p.speed  = uSplineSpeed[0] + Math.random() * (uSplineSpeed[1] - uSplineSpeed[0]);
    p.delay  = Math.random() * uMaxSDelay;
    p.phase  = p.delay > 0 ? 'spawn' : 'flow';
    p.alpha  = p.delay > 0 ? 0 : 1;
    p.seed   = Math.random() * 1000;
    p.noiseOffset = 0;

    const { x, y, z } = this._evalParticlePos(p.edgeIndex, 0, 0, 0);
    p.x = x; p.y = y; p.z = z;
  }

  // ── Edge / config management ──────────────────────────────────────────────

  /**
   * Replace all edge splines at runtime (e.g. after topology update).
   * Existing particles are cleared and reallocated.
   */
  setEdges(edges: EdgeSplineData[]): void {
    this.edges = edges;
    this.arcTabs = edges.map(e =>
      buildArcTable(
        e.points,
        this.cfg.arcLengthDivisions * Math.max(1, e.points.length - 1),
      )
    );
    this._allocParticles();
    console.log(`[SplineParticleLife] setEdges: ${edges.length} edges, ${this.particles.length} particles`);
  }

  /** Live-update UIL parameters without reallocating particles. */
  setParams(partial: Partial<SplineParticleLifeConfig>): void {
    Object.assign(this.cfg, partial);
  }

  // UIL-named setters (mirrors AT uil_params.json naming convention)
  setSplineSpeed(min: number, max: number): void {
    this.cfg.uSplineSpeed = [min, max];
    for (const p of this.particles) {
      if (p.phase === 'flow' || p.phase === 'spawn') {
        p.speed = min + Math.random() * (max - min);
      }
    }
  }

  setTimeMultiplier(v: number): void  { this.cfg.uTimeMultiplier = v; }
  setDecayRate(v: number): void        { this.cfg.uDecayRate = v; }
  setCurlStrength(v: number): void     { this.cfg.uCurlStrength = v; }
  setCurlNoiseScale(v: number): void   { this.cfg.uCurlNoiseScale = v; }
  setCurlNoiseSpeed(v: number): void   { this.cfg.uCurlNoiseSpeed = v; }
  setFlowRange(min: number, max: number): void { this.cfg.uFlowRange = [min, max]; }

  // ── Accessors ─────────────────────────────────────────────────────────────

  /** Count of non-dead particles currently in FLOW or DECAY */
  get activeCount(): number {
    return this.particles.filter(p => p.phase === 'flow' || p.phase === 'decay').length;
  }

  /** Total particle slot count (including spawning / dead) */
  get totalCount(): number {
    return this.particles.length;
  }

  /** Elapsed simulation seconds */
  get time(): number {
    return this.elapsed;
  }

  /** Edge data array (read-only reference) */
  get edgeData(): readonly EdgeSplineData[] {
    return this.edges;
  }
}

// ─── Factory helpers ──────────────────────────────────────────────────────────

/**
 * Build EdgeSplineData from a flat list of route points in pixel/canvas space,
 * normalising them to SPH domain units.
 *
 * @param edgeId    Edge identifier
 * @param sourceId  Source cell id
 * @param targetId  Target cell id
 * @param points    Control points in canvas-pixel space [{x,y}]
 * @param weight    Edge attention weight
 * @param canvasW   Canvas width in pixels
 * @param canvasH   Canvas height in pixels
 * @param domainW   SPH domain width (world units)
 * @param domainH   SPH domain height (world units)
 */
export function edgeRouteToSplineData(
  edgeId:   string,
  sourceId: string,
  targetId: string,
  points:   Array<{ x: number; y: number }>,
  weight:   number,
  canvasW:  number,
  canvasH:  number,
  domainW:  number,
  domainH:  number,
): EdgeSplineData {
  const scaleX = domainW / canvasW;
  const scaleY = domainH / canvasH;
  return {
    edgeId, sourceId, targetId, weight,
    points: points.map(p => ({
      x: p.x * scaleX,
      y: p.y * scaleY,
      z: 0,
    })),
  };
}

/**
 * Wire a SplineParticleLife instance into an SPHWorld by providing
 * the handoff callback that calls world.addFluid() to inject arriving
 * particles into the target cell's domain.
 *
 * Usage:
 *   const spl = createSplineParticleLifeForSPH(edges, world, config);
 *   // each frame:
 *   spl.update(deltaSeconds);
 */
export function createSplineParticleLifeForSPH(
  edges:   EdgeSplineData[],
  /** addFluid function with signature (x0,y0,x1,y1,spacing,species) */
  addFluid: (x0: number, y0: number, x1: number, y1: number, spacing: number, species: number) => void,
  config:  Omit<SplineParticleLifeConfig, 'onHandoff'> = {},
): SplineParticleLife {
  const HANDOFF_RADIUS = 0.05; // small injection square around landing point

  return new SplineParticleLife(edges, {
    ...config,
    onHandoff: (_edgeId, _targetId, x, y, _vx, _vy, species) => {
      // Inject a tiny fluid patch at the arrival point
      addFluid(
        x - HANDOFF_RADIUS,
        y - HANDOFF_RADIUS,
        x + HANDOFF_RADIUS,
        y + HANDOFF_RADIUS,
        HANDOFF_RADIUS * 0.8, // tight spacing for the micro-patch
        species,
      );
    },
  });
}
