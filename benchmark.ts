#!/usr/bin/env tsx
/**
 * benchmark.ts — M753: Dam-break physics benchmark for cell-pubsub-loop.
 *
 * Simulates a "dam break" scenario: a dense column of cells is released
 * into an open canvas and the physics engine (d3-force style link / charge /
 * center forces + tiled collision solver) runs until convergence or frame
 * budget is exhausted.
 *
 * Presets mirror real-world SPH dam-break benchmarks:
 *   SMALL   —   16 cells, 4×4 column   (smoke test)
 *   MEDIUM  —   64 cells, 8×8 column   (CI gate)
 *   LARGE   —  256 cells, 16×16 column  (stress test)
 *   MASSIVE — 1024 cells, 32×32 column  (max_cells limit from FAstroRendererConfig)
 *
 * Usage:
 *   tsx benchmark.ts                  # runs MEDIUM preset, 120 frames
 *   tsx benchmark.ts SMALL  60       # runs SMALL preset, 60 frames
 *   tsx benchmark.ts LARGE  300      # runs LARGE preset, 300 frames
 */

// ─── Types ──────────────────────────────────────────────────────────────────

/** Bounding box for a single cell — mirrors channels/cell/{id}/bbox.json */
interface CellBBox {
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
}

/** Per-cell force accumulator — mirrors physics/force_field.json entries */
interface ForceVec {
  dx: number;
  dy: number;
  dz: number;
}

/** Edge connecting two cells — mirrors topology.json edges */
interface TopoEdge {
  id: string;
  source: string;
  target: string;
}

/** Complete world state fed to the benchmark runner */
interface World {
  canvas: { width: number; height: number };
  cells: Map<string, CellBBox>;
  edges: TopoEdge[];
  forceField: Map<string, ForceVec>;
}

/** Per-frame snapshot recorded during the run */
interface FrameMetric {
  frame: number;
  totalForce: number;
  deltaMax: number;
  collisions: number;
  wallMs: number;
}

/** Final result returned by run() */
interface BenchResult {
  preset: string;
  cellCount: number;
  edgeCount: number;
  framesRun: number;
  convergedAtFrame: number | null;
  totalWallMs: number;
  avgFrameMs: number;
  peakForce: number;
  finalForce: number;
  finalDeltaMax: number;
  peakCollisions: number;
  frames: FrameMetric[];
}

// ─── Dam-break presets ──────────────────────────────────────────────────────

interface DamBreakPreset {
  name: string;
  /** Number of cells along the X axis of the initial column */
  cols: number;
  /** Number of cells along the Y axis of the initial column */
  rows: number;
  /** Individual cell width (px) */
  cellW: number;
  /** Individual cell height (px) */
  cellH: number;
  /** Canvas dimensions */
  canvasW: number;
  canvasH: number;
  /** Gap between cells in the dam column (0 = tightly packed) */
  gap: number;
  /** Column X-offset from canvas left edge */
  damX: number;
  /** Column Y-offset from canvas top edge */
  damY: number;
  /** d3-force parameters */
  alpha: number;
  alphaDecay: number;
  chargeStrength: number;
  linkStrength: number;
  centerStrength: number;
  /** Convergence thresholds (from FAstroRendererConfig / run_loop) */
  forceThreshold: number;
  deltaThreshold: number;
}

const PRESETS: Record<string, DamBreakPreset> = {
  SMALL: {
    name: 'SMALL',
    cols: 4, rows: 4,
    cellW: 40, cellH: 30,
    canvasW: 400, canvasH: 300,
    gap: 2,
    damX: 10, damY: 10,
    alpha: 0.3, alphaDecay: 0.02,
    chargeStrength: -80,
    linkStrength: 0.4,
    centerStrength: 0.05,
    forceThreshold: 1.0,
    deltaThreshold: 0.5,
  },
  MEDIUM: {
    name: 'MEDIUM',
    cols: 8, rows: 8,
    cellW: 40, cellH: 30,
    canvasW: 800, canvasH: 600,
    gap: 2,
    damX: 20, damY: 20,
    alpha: 0.3, alphaDecay: 0.015,
    chargeStrength: -80,
    linkStrength: 0.4,
    centerStrength: 0.05,
    forceThreshold: 1.0,
    deltaThreshold: 0.5,
  },
  LARGE: {
    name: 'LARGE',
    cols: 16, rows: 16,
    cellW: 30, cellH: 25,
    canvasW: 1200, canvasH: 900,
    gap: 1,
    damX: 30, damY: 30,
    alpha: 0.3, alphaDecay: 0.01,
    chargeStrength: -80,
    linkStrength: 0.4,
    centerStrength: 0.05,
    forceThreshold: 1.0,
    deltaThreshold: 0.5,
  },
  MASSIVE: {
    name: 'MASSIVE',
    cols: 32, rows: 32,
    cellW: 20, cellH: 18,
    canvasW: 2000, canvasH: 1600,
    gap: 0,
    damX: 40, damY: 40,
    alpha: 0.3, alphaDecay: 0.008,
    chargeStrength: -80,
    linkStrength: 0.4,
    centerStrength: 0.05,
    forceThreshold: 1.0,
    deltaThreshold: 0.5,
  },
};

// ─── World construction ─────────────────────────────────────────────────────

/**
 * Build the initial dam-break world: a tightly packed column of cells
 * positioned at (damX, damY).  Edges connect each cell to its right
 * and bottom neighbours (grid topology), mimicking the Transformer
 * DAG connectivity pattern at scale.
 */
function createDamBreakWorld(preset: DamBreakPreset): World {
  const cells = new Map<string, CellBBox>();
  const forceField = new Map<string, ForceVec>();
  const edges: TopoEdge[] = [];

  const cellId = (col: number, row: number) => `cell_${row}_${col}`;

  for (let r = 0; r < preset.rows; r++) {
    for (let c = 0; c < preset.cols; c++) {
      const id = cellId(c, r);
      cells.set(id, {
        x: preset.damX + c * (preset.cellW + preset.gap),
        y: preset.damY + r * (preset.cellH + preset.gap),
        w: preset.cellW,
        h: preset.cellH,
        z: 3, // default z-layer, matching project convention
      });
      forceField.set(id, { dx: 0, dy: 0, dz: 0 });

      // Right neighbour edge
      if (c + 1 < preset.cols) {
        edges.push({
          id: `e_${r}_${c}_r`,
          source: id,
          target: cellId(c + 1, r),
        });
      }
      // Bottom neighbour edge
      if (r + 1 < preset.rows) {
        edges.push({
          id: `e_${r}_${c}_d`,
          source: id,
          target: cellId(c, r + 1),
        });
      }
    }
  }

  return {
    canvas: { width: preset.canvasW, height: preset.canvasH },
    cells,
    edges,
    forceField,
  };
}

// ─── Physics forces (TypeScript port of loop_orchestrator.py physics_step) ──

const EPSILON = 1e-6;
const TILE_SIZE = 16;

/** Center force: nudge weighted centroid toward canvas centre. */
function applyCenterForce(
  world: World, alpha: number, strength: number,
): void {
  const { cells, forceField, canvas } = world;
  if (strength <= 0 || cells.size === 0) return;

  let totalMass = 0;
  let sx = 0;
  let sy = 0;
  for (const [id, b] of cells) {
    const mass = Math.max(b.w * b.h, 1);
    const cx = b.x + b.w * 0.5;
    const cy = b.y + b.h * 0.5;
    sx += cx * mass;
    sy += cy * mass;
    totalMass += mass;
  }
  sx /= totalMass;
  sy /= totalMass;

  const targetX = canvas.width * 0.5;
  const targetY = canvas.height * 0.5;
  const shiftX = (sx - targetX) * strength * alpha;
  const shiftY = (sy - targetY) * strength * alpha;

  for (const [id] of cells) {
    const f = forceField.get(id)!;
    f.dx -= shiftX;
    f.dy -= shiftY;
  }
}

/** Many-body charge force: O(N²) all-pairs repulsion (same z only). */
function applyChargeForce(
  world: World, alpha: number, strength: number,
): void {
  if (strength === 0) return;
  const ids = [...world.cells.keys()];
  const n = ids.length;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = world.cells.get(ids[i])!;
      const b = world.cells.get(ids[j])!;
      if (a.z !== b.z) continue;

      let dx = (a.x + a.w * 0.5) - (b.x + b.w * 0.5);
      let dy = (a.y + a.h * 0.5) - (b.y + b.h * 0.5);
      let distSq = dx * dx + dy * dy;
      if (distSq < EPSILON) {
        dx = EPSILON * (0.5 - ((simpleHash(ids[i] + ids[j]) & 0xFF) / 255));
        dy = EPSILON * (0.5 - ((simpleHash(ids[j] + ids[i]) & 0xFF) / 255));
        distSq = dx * dx + dy * dy;
      }
      const distSqClamped = Math.max(distSq, 1.0);
      const fmag = strength * alpha / distSqClamped;
      const fx = dx * fmag;
      const fy = dy * fmag;

      const fa = world.forceField.get(ids[i])!;
      const fb = world.forceField.get(ids[j])!;
      fa.dx += fx; fa.dy += fy;
      fb.dx -= fx; fb.dy -= fy;
    }
  }
}

/** Link force: spring toward target distance for each topology edge. */
function applyLinkForce(
  world: World, alpha: number, strength: number,
): void {
  if (strength <= 0) return;
  for (const edge of world.edges) {
    const sa = world.cells.get(edge.source);
    const sb = world.cells.get(edge.target);
    if (!sa || !sb) continue;

    const ax = sa.x + sa.w * 0.5;
    const ay = sa.y + sa.h * 0.5;
    const bx = sb.x + sb.w * 0.5;
    const by = sb.y + sb.h * 0.5;

    let dx = bx - ax;
    let dy = by - ay;
    const dist = Math.sqrt(dx * dx + dy * dy) || EPSILON;

    // Target distance: mean of the two cells' diagonals
    const diagA = Math.sqrt(sa.w * sa.w + sa.h * sa.h);
    const diagB = Math.sqrt(sb.w * sb.w + sb.h * sb.h);
    const target = (diagA + diagB) * 0.5;

    const delta = (dist - target) / dist;
    const fx = dx * delta * strength * alpha * 0.5;
    const fy = dy * delta * strength * alpha * 0.5;

    const fa = world.forceField.get(edge.source)!;
    const fb = world.forceField.get(edge.target)!;
    fa.dx += fx; fa.dy += fy;
    fb.dx -= fx; fb.dy -= fy;
  }
}

/**
 * Tiled collision solver: O(N·K) spatial-partition overlap detection
 * and repulsion.  Port of _tiled_constraint_solve from loop_orchestrator.py.
 */
function tiledCollisionSolve(world: World): number {
  const { cells, forceField, canvas } = world;
  const cw = Math.max(canvas.width, TILE_SIZE);
  const ch = Math.max(canvas.height, TILE_SIZE);
  const tilesX = Math.ceil(cw / TILE_SIZE);
  const tilesY = Math.ceil(ch / TILE_SIZE);

  // Assign cells to tiles by bbox centre
  const tileMap: string[][] = new Array(tilesX * tilesY);
  for (let i = 0; i < tileMap.length; i++) tileMap[i] = [];

  for (const [id, b] of cells) {
    const cx = b.x + b.w * 0.5;
    const cy = b.y + b.h * 0.5;
    const tx = Math.max(0, Math.min(Math.floor(cx / TILE_SIZE), tilesX - 1));
    const ty = Math.max(0, Math.min(Math.floor(cy / TILE_SIZE), tilesY - 1));
    tileMap[ty * tilesX + tx].push(id);
  }

  const resolved = new Set<string>();
  let collisions = 0;

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      // 3×3 neighbourhood
      const hood: string[] = [];
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = tx + dx;
          const ny = ty + dy;
          if (nx >= 0 && nx < tilesX && ny >= 0 && ny < tilesY) {
            for (const id of tileMap[ny * tilesX + nx]) hood.push(id);
          }
        }
      }

      for (let i = 0; i < hood.length; i++) {
        for (let j = i + 1; j < hood.length; j++) {
          const aid = hood[i] < hood[j] ? hood[i] : hood[j];
          const bid = hood[i] < hood[j] ? hood[j] : hood[i];
          const pairKey = `${aid}|${bid}`;
          if (resolved.has(pairKey)) continue;

          const a = cells.get(hood[i])!;
          const b = cells.get(hood[j])!;
          if (a.z !== b.z) continue;

          const ovX = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
          const ovY = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
          if (ovX <= 0 || ovY <= 0) continue;

          resolved.add(pairKey);
          collisions++;

          const fa = forceField.get(hood[i])!;
          const fb = forceField.get(hood[j])!;
          if (ovX < ovY) {
            const push = ovX / 2 + 5;
            if (a.x < b.x) { fa.dx -= push; fb.dx += push; }
            else            { fa.dx += push; fb.dx -= push; }
          } else {
            const push = ovY / 2 + 5;
            if (a.y < b.y) { fa.dy -= push; fb.dy += push; }
            else            { fa.dy += push; fb.dy -= push; }
          }
        }
      }
    }
  }
  return collisions;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function simpleHash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return h >>> 0;
}

function computeTotalForce(ff: Map<string, ForceVec>): number {
  let total = 0;
  for (const f of ff.values()) {
    total += Math.abs(f.dx) + Math.abs(f.dy);
  }
  return total;
}

function computeDeltaMax(
  prev: Map<string, CellBBox>, cur: Map<string, CellBBox>,
): number {
  let dmax = 0;
  for (const [id, b] of cur) {
    const p = prev.get(id);
    if (!p) continue;
    const d = Math.max(
      Math.abs(b.x - p.x), Math.abs(b.y - p.y),
      Math.abs(b.w - p.w), Math.abs(b.h - p.h),
    );
    if (d > dmax) dmax = d;
  }
  return dmax;
}

function snapshotBBoxes(cells: Map<string, CellBBox>): Map<string, CellBBox> {
  const snap = new Map<string, CellBBox>();
  for (const [id, b] of cells) {
    snap.set(id, { ...b });
  }
  return snap;
}

// ─── Main runner ────────────────────────────────────────────────────────────

/**
 * Run the dam-break benchmark.
 *
 * Each frame:
 *   1. Zero force field
 *   2. Apply center / charge / link forces  (d3-force style)
 *   3. Tiled collision solve                (repulsion)
 *   4. Integrate: cell.x += forceField.dx,  cell.y += forceField.dy
 *   5. Record metrics; check convergence
 *
 * @param world  Initial world state (mutated in place)
 * @param frames Maximum number of frames to simulate
 * @returns      BenchResult with per-frame metrics and summary
 */
function run(world: World, frames: number): BenchResult {
  const presetName = (world as World & { _preset?: string })._preset ?? 'CUSTOM';
  const cellCount = world.cells.size;
  const edgeCount = world.edges.length;

  // Infer preset params or use MEDIUM defaults
  const p = PRESETS[presetName] ?? PRESETS.MEDIUM;

  let alpha = p.alpha;
  const frameMetrics: FrameMetric[] = [];
  let convergedAtFrame: number | null = null;
  let peakForce = 0;
  let peakCollisions = 0;

  let prevBBoxes = snapshotBBoxes(world.cells);
  const t0 = performance.now();

  for (let frame = 0; frame < frames; frame++) {
    const ft0 = performance.now();

    // 1. Zero force field
    for (const f of world.forceField.values()) {
      f.dx = 0; f.dy = 0; f.dz = 0;
    }

    // 2. Apply d3-force trio
    applyCenterForce(world, alpha, p.centerStrength);
    applyChargeForce(world, alpha, p.chargeStrength);
    applyLinkForce(world, alpha, p.linkStrength);

    // 3. Tiled collision solve
    const collisions = tiledCollisionSolve(world);

    // 4. Integrate positions
    for (const [id, b] of world.cells) {
      const f = world.forceField.get(id)!;
      b.x += f.dx;
      b.y += f.dy;
      // Clamp to canvas
      b.x = Math.max(0, Math.min(b.x, world.canvas.width - b.w));
      b.y = Math.max(0, Math.min(b.y, world.canvas.height - b.h));
    }

    // 5. Metrics
    const totalForce = computeTotalForce(world.forceField);
    const deltaMax = computeDeltaMax(prevBBoxes, world.cells);
    const wallMs = performance.now() - ft0;

    if (totalForce > peakForce) peakForce = totalForce;
    if (collisions > peakCollisions) peakCollisions = collisions;

    frameMetrics.push({ frame, totalForce, deltaMax, collisions, wallMs });
    prevBBoxes = snapshotBBoxes(world.cells);

    // 6. Convergence check
    if (
      convergedAtFrame === null &&
      frame >= 5 &&
      totalForce < p.forceThreshold &&
      deltaMax < p.deltaThreshold
    ) {
      convergedAtFrame = frame;
    }

    // 7. Alpha decay (cooling)
    alpha *= (1 - p.alphaDecay);
  }

  const totalWallMs = performance.now() - t0;
  const lastFrame = frameMetrics[frameMetrics.length - 1];

  return {
    preset: presetName,
    cellCount,
    edgeCount,
    framesRun: frames,
    convergedAtFrame,
    totalWallMs,
    avgFrameMs: totalWallMs / frames,
    peakForce,
    finalForce: lastFrame?.totalForce ?? 0,
    finalDeltaMax: lastFrame?.deltaMax ?? 0,
    peakCollisions,
    frames: frameMetrics,
  };
}

// ─── CLI entry ──────────────────────────────────────────────────────────────

function main(): void {
  const args = process.argv.slice(2);
  const presetName = (args[0] ?? 'MEDIUM').toUpperCase();
  const frames = parseInt(args[1] ?? '120', 10);

  const preset = PRESETS[presetName];
  if (!preset) {
    console.error(`Unknown preset: ${presetName}`);
    console.error(`Available: ${Object.keys(PRESETS).join(', ')}`);
    process.exit(1);
  }

  console.log('═'.repeat(60));
  console.log(`M753 Dam-Break Benchmark — ${preset.name}`);
  console.log(`  cells=${preset.cols * preset.rows}  edges≈${2 * preset.cols * preset.rows - preset.cols - preset.rows}  frames=${frames}`);
  console.log(`  canvas=${preset.canvasW}×${preset.canvasH}  cell=${preset.cellW}×${preset.cellH}  gap=${preset.gap}`);
  console.log('═'.repeat(60));

  const world = createDamBreakWorld(preset) as World & { _preset?: string };
  world._preset = preset.name;

  const result = run(world, frames);

  // Summary
  console.log('\n' + '─'.repeat(60));
  console.log('RESULT SUMMARY');
  console.log('─'.repeat(60));
  console.log(`  preset:          ${result.preset}`);
  console.log(`  cells:           ${result.cellCount}`);
  console.log(`  edges:           ${result.edgeCount}`);
  console.log(`  framesRun:       ${result.framesRun}`);
  console.log(`  convergedAt:     ${result.convergedAtFrame ?? 'N/A'}`);
  console.log(`  totalWallMs:     ${result.totalWallMs.toFixed(2)}`);
  console.log(`  avgFrameMs:      ${result.avgFrameMs.toFixed(3)}`);
  console.log(`  peakForce:       ${result.peakForce.toFixed(2)}`);
  console.log(`  finalForce:      ${result.finalForce.toFixed(2)}`);
  console.log(`  finalDeltaMax:   ${result.finalDeltaMax.toFixed(4)}`);
  console.log(`  peakCollisions:  ${result.peakCollisions}`);

  // Frame-by-frame table (every 10th frame + last)
  console.log('\n  FRAME LOG (sampled):');
  console.log(`  ${'frame'.padStart(6)}  ${'force'.padStart(12)}  ${'deltaMax'.padStart(10)}  ${'coll'.padStart(6)}  ${'ms'.padStart(8)}`);
  for (const fm of result.frames) {
    if (fm.frame % 10 === 0 || fm.frame === result.framesRun - 1) {
      console.log(
        `  ${String(fm.frame).padStart(6)}  ` +
        `${fm.totalForce.toFixed(2).padStart(12)}  ` +
        `${fm.deltaMax.toFixed(4).padStart(10)}  ` +
        `${String(fm.collisions).padStart(6)}  ` +
        `${fm.wallMs.toFixed(3).padStart(8)}`
      );
    }
  }
  console.log('═'.repeat(60));
}

// ─── Exports ────────────────────────────────────────────────────────────────

export { run, createDamBreakWorld, PRESETS };
export type { World, BenchResult, CellBBox, ForceVec, TopoEdge, FrameMetric, DamBreakPreset };

main();
