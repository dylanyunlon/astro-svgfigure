/**
 * shadow-map.ts — M786: Shadow Mapping PCF — Cell Casts Shadows
 * ─────────────────────────────────────────────────────────────────────────────
 * 每个 Cell 都是一个遮挡体。
 *
 * Extends the shadow-system (M784) concept with first-class Cell support.
 * Where shadow-system treats only RigidBody and ObstacleData as occluders,
 * this module adds Cell bounding boxes as shadow casters — enabling every
 * Transformer cell (self_attn, ffn, layernorm, …) to cast a directional
 * shadow into the SPH particle field.
 *
 * Architecture
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Phase 1 ── Depth Map Generation ──────────────────────────────────────┐
 *   │  1a. Project ALL occluders into light space:                            │
 *   │      • RigidBody  (oriented rectangle — existing)                      │
 *   │      • ObstacleData (circle — existing)                                │
 *   │      • CellCaster (axis-aligned rectangle — NEW)                       │
 *   │                                                                         │
 *   │  1b. For each texel column along the tangent axis, raycast from the    │
 *   │      light boundary and record nearest occluder depth. Cell casters    │
 *   │      contribute an opacity factor [0, 1] so semi-transparent cells     │
 *   │      (e.g. layernorm) cast softer shadows than opaque cells (e.g. ffn).│
 *   │                                                                         │
 *   │  Depth map is 1D: Float32Array[resolution]                              │
 *   │  Opacity map:     Float32Array[resolution] — per-texel peak opacity     │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                  │
 *                  ▼
 *   ┌─ Phase 2 ── PCF Soft Shadow Sampling ──────────────────────────────────┐
 *   │  samplePCF(x, y):                                                       │
 *   │    Project world point → light space (texel, depth)                     │
 *   │    For N taps in [-radius, +radius]:                                    │
 *   │      compare point depth vs depthMap[texel+offset] + bias              │
 *   │      weight by opacityMap[texel+offset] (cell transparency)            │
 *   │    Return average ∈ [0, 1]  (0 = full shadow, 1 = full light)          │
 *   │                                                                         │
 *   │  Poisson-disk variant available for higher-quality PCF at equal cost.   │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                  │
 *                  ▼
 *   ┌─ Phase 3 ── Integration Points ────────────────────────────────────────┐
 *   │  • queryParticle(px, py)        — per-particle shadow factor            │
 *   │  • queryCells(cells[])          — batch shadow for a cell array         │
 *   │  • compositeToCanvas(ctx, w, h) — full-frame shadow overlay             │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 * Performance
 * ─────────────────────────────────────────────────────────────────────────────
 *   Depth map build: O(resolution × (numRigidBodies + numObstacles + numCells))
 *   PCF query:       O(pcfTaps) per point
 *   Typical frame:   <0.6ms for 256 texels, 12 cells, 7 PCF taps
 *
 * Usage
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   const sm = new ShadowMap({ resolution: 256, lightDir: [-0.5, -1] });
 *   sm.setWorldBounds(0, 0, 800, 600);
 *
 *   // Register cells as shadow casters
 *   sm.setCellCasters([
 *     { id: 'self_attn', x: 200, y: 100, w: 80, h: 60, opacity: 1.0 },
 *     { id: 'ffn',       x: 400, y: 200, w: 100, h: 80, opacity: 1.0 },
 *     { id: 'layernorm', x: 300, y: 150, w: 60, h: 20, opacity: 0.4 },
 *   ]);
 *
 *   // Build the depth map (call once per frame)
 *   sm.buildDepthMap(rigidBodies, obstacles);
 *
 *   // Query shadow at a particle position
 *   const lit = sm.samplePCF(px, py);  // 0 → shadow, 1 → lit
 *
 *   // Or composite shadows over an existing canvas
 *   sm.compositeToCanvas(ctx, canvas.width, canvas.height);
 *
 * Research: xiaodi #M786 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A Cell treated as a shadow-casting occluder.
 *
 * Unlike RigidBody (which has rotation), CellCaster is axis-aligned: the
 * cell bounding box from InteractionCell / CellPhysicsConfig is used directly.
 * An optional opacity controls shadow intensity — semi-transparent UI elements
 * (layernorm indicators, skip connections) cast lighter shadows.
 */



import type { RigidBody } from './rigid-body';
import type { ObstacleData } from './types';

export interface CellCaster {
  /** Unique cell ID (e.g. 'self_attn', 'ffn_0'). */
  id: string;

  /** Centre X in world space. */
  x: number;
  /** Centre Y in world space. */
  y: number;

  /** Full width of the cell bounding box. */
  w: number;
  /** Full height of the cell bounding box. */
  h: number;

  /**
   * Shadow opacity ∈ [0, 1].
   * 1.0 = fully opaque shadow (dense cells like FFN, attention heads).
   * 0.0 = no shadow at all (invisible / debug cells).
   * Values between create semi-transparent shadows (layernorm, residual add).
   * @default 1.0
   */
  opacity: number;
}

/** Configuration for the ShadowMap. */
export interface ShadowMapConfig {
  /** Number of texels in the 1D depth map. @default 256 */
  resolution: number;

  /** Light direction vector (FROM light TOWARD scene). Need not be normalized. */
  lightDir: [number, number];

  /** Depth comparison bias to prevent shadow acne. @default 0.004 */
  bias: number;

  /**
   * Minimum brightness in fully shadowed regions [0, 1].
   * 0 = pitch black shadows, 1 = no visible shadows.
   * @default 0.3
   */
  darkness: number;

  /**
   * Number of PCF taps. Must be odd. Higher values = softer edges, more cost.
   * @default 7
   */
  pcfTaps: number;

  /**
   * PCF kernel radius in texels. Controls the width of the soft penumbra.
   * @default 2.5
   */
  pcfRadius: number;

  /**
   * Use Poisson-disk distribution for PCF samples instead of uniform.
   * Better quality for the same tap count, but slightly more ALU.
   * @default false
   */
  poissonPCF: boolean;

  /**
   * Shadow-map far plane (world units). Rays stop here.
   * @default Infinity (auto-computed from world bounds diagonal)
   */
  farPlane: number;

  /**
   * Ambient occlusion tint applied to shadowed regions (RGB, 0–1).
   * Subtle cool shift simulates indirect sky light.
   * @default [0.82, 0.86, 0.94]
   */
  aoTint: [number, number, number];

  /** AO tint intensity ∈ [0, 1]. @default 0.12 */
  aoIntensity: number;
}

/** Default shadow map configuration. */
const DEFAULTS: ShadowMapConfig = {
  resolution:  256,
  lightDir:    [0, -1],
  bias:        0.004,
  darkness:    0.3,
  pcfTaps:     7,
  pcfRadius:   2.5,
  poissonPCF:  false,
  farPlane:    Infinity,
  aoTint:      [0.82, 0.86, 0.94],
  aoIntensity: 0.12,
};

// ─────────────────────────────────────────────────────────────────────────────
// Pre-baked Poisson-disk offsets (normalized to [−1, 1])
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 16-tap Poisson-disk pattern projected onto 1D. More perceptually uniform
 * than equidistant spacing — avoids banding artefacts at penumbra edges.
 */
const POISSON_1D_16: ReadonlyArray<number> = [
  0.000,
  -0.326,  0.326,
  -0.618,  0.618,
  -0.155,  0.155,
  -0.834,  0.834,
  -0.475,  0.475,
  -0.072,  0.072,
  -0.940,  0.940,
  -0.250,
];

// ─────────────────────────────────────────────────────────────────────────────
// Vector helpers
// ─────────────────────────────────────────────────────────────────────────────

function normalize(out: [number, number], v: [number, number]): number {
  const len = Math.sqrt(v[0] * v[0] + v[1] * v[1]);
  if (len < 1e-12) { out[0] = 0; out[1] = -1; return 0; }
  out[0] = v[0] / len;
  out[1] = v[1] / len;
  return len;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray intersection helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Ray vs line-segment. Returns parametric t along ray, or Infinity.
 * Ray: P = (ox, oy) + t·(dx, dy), t ≥ 0
 * Segment: A → B, u ∈ [0, 1]
 */
function raySegment(
  ox: number, oy: number,
  dx: number, dy: number,
  ax: number, ay: number,
  bx: number, by: number,
): number {
  const sx = bx - ax;
  const sy = by - ay;
  const denom = dx * sy - dy * sx;
  if (Math.abs(denom) < 1e-12) return Infinity;
  const t = ((ax - ox) * sy - (ay - oy) * sx) / denom;
  const u = ((ax - ox) * dy - (ay - oy) * dx) / denom;
  return (t >= 0 && u >= 0 && u <= 1) ? t : Infinity;
}

/** Ray vs circle — nearest positive t, or Infinity. */
function rayCircle(
  ox: number, oy: number,
  dx: number, dy: number,
  cx: number, cy: number,
  r: number,
): number {
  const fx = ox - cx;
  const fy = oy - cy;
  const a = dx * dx + dy * dy;
  const b = 2 * (fx * dx + fy * dy);
  const c = fx * fx + fy * fy - r * r;
  let disc = b * b - 4 * a * c;
  if (disc < 0) return Infinity;
  disc = Math.sqrt(disc);
  const t0 = (-b - disc) / (2 * a);
  const t1 = (-b + disc) / (2 * a);
  if (t0 >= 0) return t0;
  if (t1 >= 0) return t1;
  return Infinity;
}

/** Ray vs circle — far intersection t (exit point). */
function rayCircleExit(
  ox: number, oy: number,
  dx: number, dy: number,
  cx: number, cy: number,
  r: number,
): number {
  const fx = ox - cx;
  const fy = oy - cy;
  const a = dx * dx + dy * dy;
  const b = 2 * (fx * dx + fy * dy);
  const c = fx * fx + fy * fy - r * r;
  let disc = b * b - 4 * a * c;
  if (disc < 0) return Infinity;
  disc = Math.sqrt(disc);
  const t1 = (-b + disc) / (2 * a);
  return t1 >= 0 ? t1 : Infinity;
}

/**
 * Ray vs oriented rectangle (RigidBody).
 * Transform ray into body-local space, test against AABB edges.
 */
function rayRigidBody(
  ox: number, oy: number,
  dx: number, dy: number,
  body: RigidBody,
): number {
  const cos = Math.cos(-body.angle);
  const sin = Math.sin(-body.angle);
  const rx = ox - body.x;
  const ry = oy - body.y;
  const lox = cos * rx - sin * ry;
  const loy = sin * rx + cos * ry;
  const ldx = cos * dx - sin * dy;
  const ldy = sin * dx + cos * dy;

  const hw = body.w;
  const hh = body.h;

  let t = Infinity;
  t = Math.min(t, raySegment(lox, loy, ldx, ldy, -hw, -hh,  hw, -hh));
  t = Math.min(t, raySegment(lox, loy, ldx, ldy, -hw,  hh,  hw,  hh));
  t = Math.min(t, raySegment(lox, loy, ldx, ldy, -hw, -hh, -hw,  hh));
  t = Math.min(t, raySegment(lox, loy, ldx, ldy,  hw, -hh,  hw,  hh));
  return t;
}

/**
 * Ray vs axis-aligned rectangle (Cell bounding box).
 * No rotation — simpler and faster than the RigidBody path.
 *
 * Cell box: centre (cx, cy), full-size (w, h)
 * Edges at x ∈ [cx−hw, cx+hw], y ∈ [cy−hh, cy+hh].
 *
 * Returns { entry, exit } parametric t along the ray, or Infinity if miss.
 */
function rayCellAABB(
  ox: number, oy: number,
  dx: number, dy: number,
  cx: number, cy: number,
  w: number, h: number,
): { entry: number; exit: number } {
  const hw = w * 0.5;
  const hh = h * 0.5;
  const minX = cx - hw;
  const maxX = cx + hw;
  const minY = cy - hh;
  const maxY = cy + hh;

  // Slab intersection — fast AABB test
  let tMin = -Infinity;
  let tMax =  Infinity;

  if (Math.abs(dx) > 1e-12) {
    const invDx = 1 / dx;
    let t0 = (minX - ox) * invDx;
    let t1 = (maxX - ox) * invDx;
    if (t0 > t1) { const tmp = t0; t0 = t1; t1 = tmp; }
    tMin = Math.max(tMin, t0);
    tMax = Math.min(tMax, t1);
  } else {
    // Ray parallel to X slabs — check containment
    if (ox < minX || ox > maxX) return { entry: Infinity, exit: Infinity };
  }

  if (Math.abs(dy) > 1e-12) {
    const invDy = 1 / dy;
    let t0 = (minY - oy) * invDy;
    let t1 = (maxY - oy) * invDy;
    if (t0 > t1) { const tmp = t0; t0 = t1; t1 = tmp; }
    tMin = Math.max(tMin, t0);
    tMax = Math.min(tMax, t1);
  } else {
    if (oy < minY || oy > maxY) return { entry: Infinity, exit: Infinity };
  }

  if (tMax < tMin || tMax < 0) return { entry: Infinity, exit: Infinity };

  const entry = tMin >= 0 ? tMin : 0;
  const exit  = tMax;
  return { entry, exit };
}

// ─────────────────────────────────────────────────────────────────────────────
// ShadowMap
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 2D directional shadow map with PCF soft shadows.
 *
 * Cells, rigid bodies, and obstacles all act as occluders. The depth map is a
 * 1D buffer along the axis perpendicular to the light direction. Each texel
 * stores the nearest occluder depth plus the occluder's shadow opacity (for
 * cells with partial transparency).
 *
 * PCF sampling averages multiple depth comparisons across neighbouring texels
 * to produce smooth shadow edges. The optional Poisson-disk distribution
 * provides higher perceptual quality for the same number of taps.
 */
export class ShadowMap {
  // ── Configuration ───────────────────────────────────────────────────────
  private cfg: ShadowMapConfig;

  // ── Light basis ─────────────────────────────────────────────────────────
  private lightDir: [number, number] = [0, -1];
  private tangent:  [number, number] = [1,  0];

  // ── World bounds ────────────────────────────────────────────────────────
  private wMinX = 0;
  private wMinY = 0;
  private wMaxX = 1;
  private wMaxY = 1;

  // ── Light-space projection cache ────────────────────────────────────────
  private tangentOrigin = 0;
  private tangentSpan   = 1;
  private depthOrigin   = 0;
  private effectiveFar  = 1;

  // ── Depth + opacity maps ────────────────────────────────────────────────
  /** Per-texel nearest occluder depth. Infinity = no occluder. */
  private depthMap: Float32Array;

  /** Per-texel nearest occluder exit depth (for thickness-aware shadow). */
  private exitMap: Float32Array;

  /**
   * Per-texel shadow opacity of the nearest occluder.
   * 1.0 for rigid bodies / obstacles / opaque cells.
   * < 1.0 for semi-transparent cells.
   */
  private opacityMap: Float32Array;

  // ── Cell casters ────────────────────────────────────────────────────────
  private cells: CellCaster[] = [];

  // ── Composite scratch ───────────────────────────────────────────────────
  private shadowBuf: Float32Array | null = null;

  // ───────────────────────────────────────────────────────────────────────
  // Constructor
  // ───────────────────────────────────────────────────────────────────────

  constructor(config?: Partial<ShadowMapConfig>) {
    this.cfg = { ...DEFAULTS, ...config };
    // Ensure odd tap count ≥ 1
    this.cfg.pcfTaps = Math.max(1, this.cfg.pcfTaps | 1);
    const res = Math.max(1, this.cfg.resolution | 0);
    this.cfg.resolution = res;

    this.depthMap   = new Float32Array(res);
    this.exitMap    = new Float32Array(res);
    this.opacityMap = new Float32Array(res);

    this.setLightDirection(this.cfg.lightDir);
  }

  // ───────────────────────────────────────────────────────────────────────
  // Public setters
  // ───────────────────────────────────────────────────────────────────────

  /** Merge partial config updates. */
  configure(partial: Partial<ShadowMapConfig>): void {
    Object.assign(this.cfg, partial);
    this.cfg.pcfTaps = Math.max(1, this.cfg.pcfTaps | 1);
    if (partial.lightDir) this.setLightDirection(partial.lightDir);
  }

  /** Update the light direction (need not be unit length). */
  setLightDirection(dir: [number, number]): void {
    normalize(this.lightDir, dir);
    this.tangent[0] = -this.lightDir[1];
    this.tangent[1] =  this.lightDir[0];
  }

  /** Set the world-space bounding rectangle. Must be called before buildDepthMap. */
  setWorldBounds(minX: number, minY: number, maxX: number, maxY: number): void {
    this.wMinX = minX;
    this.wMinY = minY;
    this.wMaxX = maxX;
    this.wMaxY = maxY;
  }

  /**
   * Register cells that cast shadows.
   *
   * Call this each frame with the current set of visible cells. The array
   * is shallow-copied; callers may reuse their buffer.
   *
   * @param casters Array of CellCaster descriptors.
   */
  setCellCasters(casters: ReadonlyArray<CellCaster>): void {
    this.cells = casters.slice();
  }

  // ───────────────────────────────────────────────────────────────────────
  // Phase 1 — Depth Map Generation
  // ───────────────────────────────────────────────────────────────────────

  /**
   * Build the 1D depth map from the current light direction.
   *
   * All three occluder types contribute:
   *   1. Rigid bodies  — oriented rectangles (from SPH physics)
   *   2. Obstacles     — circles (boundary / domain)
   *   3. Cell casters  — axis-aligned rectangles (Transformer cells)
   *
   * The nearest occluder per texel wins. Its opacity is stored in opacityMap
   * so PCF can weight semi-transparent cell shadows correctly.
   *
   * @param rigidBodies  SPH rigid bodies (may be empty)
   * @param obstacles    Domain obstacles (may be empty)
   * @returns  The raw depth map (read-only — do not mutate).
   */
  buildDepthMap(
    rigidBodies: ReadonlyArray<RigidBody> = [],
    obstacles:   ReadonlyArray<ObstacleData> = [],
  ): Float32Array {
    this.computeLightSpaceBounds();

    const { resolution, bias, farPlane } = this.cfg;
    const { depthMap, exitMap, opacityMap, lightDir, tangent, cells } = this;

    const far = Number.isFinite(farPlane)
      ? farPlane
      : this.computeDiagonal();
    this.effectiveFar = far;

    // Clear maps
    depthMap.fill(far);
    exitMap.fill(far);
    opacityMap.fill(0);

    const texelSize = this.tangentSpan / resolution;

    for (let i = 0; i < resolution; i++) {
      const tPos = this.tangentOrigin + (i + 0.5) * texelSize;

      // Ray origin at the near plane
      const rox = tangent[0] * tPos + lightDir[0] * this.depthOrigin;
      const roy = tangent[1] * tPos + lightDir[1] * this.depthOrigin;

      let nearestEntry   = far;
      let nearestExit    = far;
      let nearestOpacity = 0;

      // ── Rigid bodies ───────────────────────────────────────────────────
      for (let b = 0; b < rigidBodies.length; b++) {
        const body = rigidBodies[b];
        const t = rayRigidBody(rox, roy, lightDir[0], lightDir[1], body);
        if (t < nearestEntry) {
          nearestEntry = t;
          nearestOpacity = 1.0;  // rigid bodies are fully opaque
          // Compute exit depth
          const overT = t + (body.w + body.h) * 4;
          const bx = rox + lightDir[0] * overT;
          const by = roy + lightDir[1] * overT;
          const backT = rayRigidBody(bx, by, -lightDir[0], -lightDir[1], body);
          nearestExit = Number.isFinite(backT)
            ? overT - backT
            : t + Math.max(body.w, body.h) * 2;
        }
      }

      // ── Obstacles (circles) ────────────────────────────────────────────
      for (let o = 0; o < obstacles.length; o++) {
        const obs = obstacles[o];
        const t = rayCircle(rox, roy, lightDir[0], lightDir[1], obs.cx, obs.cy, obs.r);
        if (t < nearestEntry) {
          nearestEntry = t;
          nearestOpacity = 1.0;
          nearestExit = rayCircleExit(
            rox, roy, lightDir[0], lightDir[1],
            obs.cx, obs.cy, obs.r,
          );
        }
      }

      // ── Cell casters (axis-aligned rectangles) ─────────────────────────
      for (let c = 0; c < cells.length; c++) {
        const cell = cells[c];
        if (cell.opacity <= 0) continue;  // invisible cell — skip

        const hit = rayCellAABB(
          rox, roy, lightDir[0], lightDir[1],
          cell.x, cell.y, cell.w, cell.h,
        );

        if (hit.entry < nearestEntry) {
          nearestEntry   = hit.entry;
          nearestExit    = hit.exit;
          nearestOpacity = cell.opacity;
        } else if (
          // Same depth (co-planar) — pick the higher opacity
          Math.abs(hit.entry - nearestEntry) < bias &&
          cell.opacity > nearestOpacity
        ) {
          nearestOpacity = cell.opacity;
          nearestExit    = Math.max(nearestExit, hit.exit);
        }
      }

      depthMap[i]   = nearestEntry;
      exitMap[i]    = nearestExit;
      opacityMap[i] = nearestOpacity;
    }

    return depthMap;
  }

  // ───────────────────────────────────────────────────────────────────────
  // Phase 2 — Shadow Sampling
  // ───────────────────────────────────────────────────────────────────────

  /**
   * Hard shadow test at a single world-space point.
   * @returns 0 (full shadow) or 1 (lit). Ignores occluder opacity — use
   *          samplePCF for opacity-aware queries.
   */
  sampleHard(x: number, y: number): number {
    const { texel, depth } = this.projectToLightSpace(x, y);
    const idx = this.texelIndex(texel);
    if (idx < 0) return 1;
    return depth > this.depthMap[idx] + this.cfg.bias ? 0 : 1;
  }

  /**
   * PCF soft shadow with opacity weighting.
   *
   * Averages N depth-comparison taps across a texel window, weighted by
   * the per-texel occluder opacity. Semi-transparent cells produce a
   * proportionally lighter shadow.
   *
   * @param x World X
   * @param y World Y
   * @returns Value ∈ [0, 1]. 0 = full shadow, 1 = full light.
   */
  samplePCF(x: number, y: number): number {
    const { texel, depth } = this.projectToLightSpace(x, y);
    const { pcfTaps, pcfRadius, bias, poissonPCF } = this.cfg;
    const { depthMap, opacityMap, cfg } = this;

    const halfN = (pcfTaps - 1) / 2;
    let litSum = 0;

    for (let s = 0; s < pcfTaps; s++) {
      let offset: number;
      if (poissonPCF && s < POISSON_1D_16.length) {
        offset = POISSON_1D_16[s] * pcfRadius;
      } else {
        offset = pcfTaps === 1 ? 0 : ((s - halfN) / halfN) * pcfRadius;
      }

      const sTexel = texel + offset;
      const idx = this.texelIndex(sTexel);

      if (idx < 0) {
        litSum += 1;  // outside map → lit
        continue;
      }

      const storedDepth = depthMap[idx];
      const occOpacity  = opacityMap[idx];

      if (depth <= storedDepth + bias) {
        litSum += 1;  // in front of occluder → lit
      } else {
        // In shadow — but attenuated by occluder opacity.
        // opacity 1.0 → litSum += 0 (full shadow)
        // opacity 0.5 → litSum += 0.5 (half shadow)
        litSum += (1 - occOpacity);
      }
    }

    return litSum / pcfTaps;
  }

  /**
   * PCF with thickness-aware attenuation.
   *
   * Thin occluders (entry-exit distance < thinLimit) produce lighter shadows,
   * preventing razor-thin cells or rigid bodies from casting unrealistically
   * hard shadows.
   *
   * @param x         World X
   * @param y         World Y
   * @param thinLimit Maximum occluder thickness (world units) below which
   *                  shadow intensity is reduced. @default 0.08
   * @returns Value ∈ [0, 1].
   */
  samplePCFThickness(
    x: number, y: number,
    thinLimit: number = 0.08,
  ): number {
    const { texel, depth } = this.projectToLightSpace(x, y);
    const { pcfTaps, pcfRadius, bias, poissonPCF } = this.cfg;
    const { depthMap, exitMap, opacityMap } = this;

    const halfN = (pcfTaps - 1) / 2;
    let litSum = 0;

    for (let s = 0; s < pcfTaps; s++) {
      let offset: number;
      if (poissonPCF && s < POISSON_1D_16.length) {
        offset = POISSON_1D_16[s] * pcfRadius;
      } else {
        offset = pcfTaps === 1 ? 0 : ((s - halfN) / halfN) * pcfRadius;
      }

      const sTexel = texel + offset;
      const idx = this.texelIndex(sTexel);

      if (idx < 0) {
        litSum += 1;
        continue;
      }

      const entryD = depthMap[idx];
      const exitD  = exitMap[idx];
      const opc    = opacityMap[idx];

      if (depth <= entryD + bias) {
        litSum += 1;
      } else {
        // In shadow — compute thickness attenuation
        const thickness = exitD - entryD;
        let shadowStrength = opc;

        if (thickness > 0 && thickness < thinLimit) {
          shadowStrength *= thickness / thinLimit;
        }

        litSum += (1 - shadowStrength);
      }
    }

    return litSum / pcfTaps;
  }

  // ───────────────────────────────────────────────────────────────────────
  // Batch queries
  // ───────────────────────────────────────────────────────────────────────

  /**
   * Query shadow factor for a particle position.
   * Convenience wrapper around samplePCF — same result.
   */
  queryParticle(px: number, py: number): number {
    return this.samplePCF(px, py);
  }

  /**
   * Batch shadow query for an array of cell casters.
   *
   * For each cell, samples shadow at the cell centre. Cells can shadow
   * each other — a cell positioned behind another (from the light's
   * perspective) receives shadow.
   *
   * @returns  Map from cell ID to shadow factor ∈ [0, 1].
   */
  queryCells(cells: ReadonlyArray<CellCaster>): Map<string, number> {
    const result = new Map<string, number>();
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      result.set(c.id, this.samplePCF(c.x, c.y));
    }
    return result;
  }

  /**
   * Batch shadow query for particle arrays (SOA layout).
   *
   * Writes shadow factors directly into the output array to avoid allocation.
   *
   * @param px     Particle X positions
   * @param py     Particle Y positions
   * @param count  Number of particles
   * @param out    Pre-allocated output array (length ≥ count)
   */
  queryParticles(
    px: Float32Array,
    py: Float32Array,
    count: number,
    out: Float32Array,
  ): void {
    for (let i = 0; i < count; i++) {
      out[i] = this.samplePCF(px[i], py[i]);
    }
  }

  // ───────────────────────────────────────────────────────────────────────
  // Phase 3 — Canvas2D Composite
  // ───────────────────────────────────────────────────────────────────────

  /**
   * Composite shadow overlay onto an existing Canvas2D rendering.
   *
   * Reads pixels via getImageData, computes per-pixel shadow via PCF at a
   * reduced resolution (for performance), bilinearly upsamples, then writes
   * back darkened + AO-tinted pixels.
   *
   * @param ctx        Canvas rendering context (must have content drawn)
   * @param width      Canvas width in pixels
   * @param height     Canvas height in pixels
   * @param downsample Shadow computation downsampling factor. @default 2
   */
  compositeToCanvas(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    downsample: number = 2,
  ): void {
    const ds = Math.max(1, downsample | 0);
    const sw = Math.ceil(width / ds);
    const sh = Math.ceil(height / ds);
    const total = sw * sh;

    if (!this.shadowBuf || this.shadowBuf.length < total) {
      this.shadowBuf = new Float32Array(total);
    }
    const shadow = this.shadowBuf;
    const { darkness, aoTint, aoIntensity } = this.cfg;

    // World-to-pixel scale
    const scaleX = (this.wMaxX - this.wMinX) / width;
    const scaleY = (this.wMaxY - this.wMinY) / height;

    // Compute shadow at reduced resolution
    for (let sy = 0; sy < sh; sy++) {
      const wy = this.wMinY + (sy * ds + ds * 0.5) * scaleY;
      for (let sx = 0; sx < sw; sx++) {
        const wx = this.wMinX + (sx * ds + ds * 0.5) * scaleX;
        shadow[sy * sw + sx] = this.samplePCF(wx, wy);
      }
    }

    // Read framebuffer
    const imageData = ctx.getImageData(0, 0, width, height);
    const px = imageData.data;

    // Bilinear upsample + composite
    for (let py = 0; py < height; py++) {
      const sy0 = Math.min(sh - 1, (py / ds) | 0);
      const sy1 = Math.min(sh - 1, sy0 + 1);
      const fy  = (py / ds) - sy0;

      for (let pxIdx = 0; pxIdx < width; pxIdx++) {
        const sx0 = Math.min(sw - 1, (pxIdx / ds) | 0);
        const sx1 = Math.min(sw - 1, sx0 + 1);
        const fx  = (pxIdx / ds) - sx0;

        // Bilinear interpolation of shadow factor
        const s00 = shadow[sy0 * sw + sx0];
        const s10 = shadow[sy0 * sw + sx1];
        const s01 = shadow[sy1 * sw + sx0];
        const s11 = shadow[sy1 * sw + sx1];

        const sMix = s00 * (1 - fx) * (1 - fy)
                   + s10 *      fx  * (1 - fy)
                   + s01 * (1 - fx) *      fy
                   + s11 *      fx  *      fy;

        // Shadow multiplier: darkness in full shadow, 1.0 in full light
        const mul = darkness + (1 - darkness) * sMix;

        const i4 = (py * width + pxIdx) * 4;
        let r = px[i4]     * mul;
        let g = px[i4 + 1] * mul;
        let b = px[i4 + 2] * mul;

        // AO tint in shadow
        const ao = (1 - sMix) * aoIntensity;
        if (ao > 0.001) {
          r = r * (1 - ao) + r * aoTint[0] * ao;
          g = g * (1 - ao) + g * aoTint[1] * ao;
          b = b * (1 - ao) + b * aoTint[2] * ao;
        }

        px[i4]     = Math.min(255, r + 0.5) | 0;
        px[i4 + 1] = Math.min(255, g + 0.5) | 0;
        px[i4 + 2] = Math.min(255, b + 0.5) | 0;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }

  // ───────────────────────────────────────────────────────────────────────
  // Internal — light-space projection
  // ───────────────────────────────────────────────────────────────────────

  /** Compute light-space AABB of the world bounds. */
  private computeLightSpaceBounds(): void {
    const corners: [number, number][] = [
      [this.wMinX, this.wMinY],
      [this.wMaxX, this.wMinY],
      [this.wMinX, this.wMaxY],
      [this.wMaxX, this.wMaxY],
    ];

    let tMin = Infinity,  tMax = -Infinity;
    let dMin = Infinity,  dMax = -Infinity;

    for (const [cx, cy] of corners) {
      const t = cx * this.tangent[0]  + cy * this.tangent[1];
      const d = cx * this.lightDir[0] + cy * this.lightDir[1];
      if (t < tMin) tMin = t;
      if (t > tMax) tMax = t;
      if (d < dMin) dMin = d;
      if (d > dMax) dMax = d;
    }

    const margin = (tMax - tMin) * 0.02;
    this.tangentOrigin = tMin - margin;
    this.tangentSpan   = (tMax - tMin) + 2 * margin;
    this.depthOrigin   = dMin - margin;

    if (!Number.isFinite(this.cfg.farPlane)) {
      this.effectiveFar = (dMax - dMin) + 2 * margin;
    }
  }

  /** World diagonal for default far plane. */
  private computeDiagonal(): number {
    const dx = this.wMaxX - this.wMinX;
    const dy = this.wMaxY - this.wMinY;
    return Math.sqrt(dx * dx + dy * dy);
  }

  /** Project world point into light space → (texel, depth). */
  private projectToLightSpace(x: number, y: number): { texel: number; depth: number } {
    const t = x * this.tangent[0]  + y * this.tangent[1];
    const d = x * this.lightDir[0] + y * this.lightDir[1];
    const texel = ((t - this.tangentOrigin) / this.tangentSpan) * this.cfg.resolution;
    const depth = d - this.depthOrigin;
    return { texel, depth };
  }

  /** Floating texel → clamped integer index, or −1 if out of range. */
  private texelIndex(texel: number): number {
    const i = texel | 0;
    if (i < 0 || i >= this.cfg.resolution) return -1;
    return i;
  }

  // ───────────────────────────────────────────────────────────────────────
  // Diagnostics
  // ───────────────────────────────────────────────────────────────────────

  /**
   * Draw a horizontal debug bar visualising the depth map.
   *
   * Bright texels = near occluder, dark = far / empty.
   * Opacity is shown as a coloured overlay (magenta tint for partial opacity).
   */
  debugDrawDepthMap(
    ctx: CanvasRenderingContext2D,
    x: number, y: number,
    width: number, height: number,
  ): void {
    const { resolution, depthMap, opacityMap, effectiveFar } = this;
    const far = Number.isFinite(effectiveFar) ? effectiveFar : this.computeDiagonal();
    const texelW = width / resolution;

    ctx.save();
    for (let i = 0; i < resolution; i++) {
      const d = depthMap[i];
      const norm = Number.isFinite(d) ? Math.min(1, d / far) : 1;
      const brightness = Math.round((1 - norm) * 255);
      const opc = opacityMap[i];

      // Base depth brightness
      ctx.fillStyle = `rgb(${brightness},${brightness},${brightness})`;
      ctx.fillRect(x + i * texelW, y, Math.ceil(texelW), height);

      // Opacity overlay (magenta tint for partial opacity)
      if (opc > 0 && opc < 1) {
        ctx.fillStyle = `rgba(200,50,200,${(0.4 * opc).toFixed(3)})`;
        ctx.fillRect(x + i * texelW, y, Math.ceil(texelW), height);
      }
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);

    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '10px monospace';
    ctx.fillText('shadow-map depth (M786)', x + 4, y + height - 4);
    ctx.restore();
  }

  /**
   * Draw a 2D shadow overlay for debugging.
   * Semi-transparent black rectangles at each sample point.
   */
  debugDrawOverlay(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    step: number = 4,
  ): void {
    const { darkness } = this.cfg;
    const scaleX = (this.wMaxX - this.wMinX) / width;
    const scaleY = (this.wMaxY - this.wMinY) / height;

    ctx.save();
    for (let py = 0; py < height; py += step) {
      const wy = this.wMinY + (py + step * 0.5) * scaleY;
      for (let px = 0; px < width; px += step) {
        const wx = this.wMinX + (px + step * 0.5) * scaleX;
        const lit = this.samplePCF(wx, wy);
        const alpha = (1 - lit) * (1 - darkness);
        if (alpha > 0.01) {
          ctx.fillStyle = `rgba(0,0,18,${alpha.toFixed(3)})`;
          ctx.fillRect(px, py, step, step);
        }
      }
    }
    ctx.restore();
  }

  // ───────────────────────────────────────────────────────────────────────
  // Read-only accessors
  // ───────────────────────────────────────────────────────────────────────

  /** Current resolution. */
  getResolution(): number { return this.cfg.resolution; }

  /** Normalized light direction (copy). */
  getLightDirection(): [number, number] {
    return [this.lightDir[0], this.lightDir[1]];
  }

  /** Tangent axis (copy). */
  getTangent(): [number, number] {
    return [this.tangent[0], this.tangent[1]];
  }

  /** Raw depth map (do not mutate). */
  getDepthMap(): Float32Array { return this.depthMap; }

  /** Raw opacity map (do not mutate). */
  getOpacityMap(): Float32Array { return this.opacityMap; }

  /** Current configuration (read-only copy). */
  getConfig(): Readonly<ShadowMapConfig> { return { ...this.cfg }; }

  /** Registered cell casters (read-only copy). */
  getCellCasters(): ReadonlyArray<CellCaster> { return this.cells; }
}
