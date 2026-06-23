// src/lib/sph/cell-body-bridge.ts
// Bridges cell_registry.json + species_assignment.json to RigidBody creation.
// Maps the 7 Transformer cells from /api/cells into SPH rigid body parameters.

export interface CellPhysicsConfig {
  id: string;           // e.g. "self_attn"
  x: number; y: number; // center from bbox
  w: number; h: number; // size from bbox
  species: string;      // e.g. "cil-eye"
  mass: number;
  friction: number;
  restitution: number;
  buoyancy: number;
  pinned: boolean;
}

// Species → physics property mapping — loaded at runtime from
// channels/physics/species_physics.json so that tuning values live in one
// place and are shared with the Python pipeline.
let SPECIES_PHYSICS: Record<string, { mass: number; friction: number; restitution: number; buoyancy: number }> = {};

/**
 * Fetch and cache the species physics data.
 * Safe to call multiple times — subsequent calls are no-ops once loaded.
 */
let _loadPromise: Promise<void> | null = null;
export function loadSpeciesPhysics(): Promise<void> {
  if (!_loadPromise) {
    _loadPromise = fetch('/channels/physics/species_physics.json')
      .then((resp) => {
        if (!resp.ok) throw new Error(`Failed to fetch species_physics.json: ${resp.status}`);
        return resp.json();
      })
      .then((data) => {
        SPECIES_PHYSICS = data as Record<string, { mass: number; friction: number; restitution: number; buoyancy: number }>;
      });
  }
  return _loadPromise;
}

// Fallback physics for unknown species
const DEFAULT_PHYSICS = { mass: 75, friction: 0.5, restitution: 0.3, buoyancy: 0.5 };

// Ordered species list for stable numeric index mapping.
// Index 0 is reserved for fluid particles; cell species start at 1.
const SPECIES_ORDER: string[] = [
  'fluid',          // 0 — fluid particles
  'cil-eye',        // 1
  'cil-bolt',       // 2
  'cil-vector',     // 3
  'cil-plus',       // 4
  'cil-arrow-right',// 5
  'cil-filter',     // 6
  'cil-layers',     // 7
  'cil-loop',       // 8
  'cil-code',       // 9
  'cil-graph',      // 10
];

/**
 * Convert a species string to a stable numeric ID used for particle colouring
 * and shader look-ups. Unknown species are bucketed to index 0 (fluid).
 */
export function speciesToIndex(species: string): number {
  const idx = SPECIES_ORDER.indexOf(species);
  return idx >= 0 ? idx : 0;
}

/**
 * Convert an /api/cells response array into fully-resolved CellPhysicsConfig
 * objects ready to be handed to the SPH bridge.
 *
 * Each cell's centre (x, y) and half-extents (w, h) are derived from its
 * axis-aligned bounding box.  Physics properties are looked up by species;
 * unknown species fall back to DEFAULT_PHYSICS.
 *
 * Cells with species "cil-plus" (residual add nodes) are pinned by default
 * because they act as structural anchors in the Transformer diagram.
 */
export function cellsToBodies(
  cells: Array<{ id: string; bbox: { min: number[]; max: number[] }; species: string }>
): CellPhysicsConfig[] {
  return cells.map((cell) => {
    const { id, bbox, species } = cell;

    // Derive centre and dimensions from axis-aligned bbox
    const x = (bbox.min[0] + bbox.max[0]) / 2;
    const y = (bbox.min[1] + bbox.max[1]) / 2;
    const w = Math.abs(bbox.max[0] - bbox.min[0]);
    const h = Math.abs(bbox.max[1] - bbox.min[1]);

    const physics = SPECIES_PHYSICS[species] ?? DEFAULT_PHYSICS;

    // Residual-add nodes are structural anchors → pin them so fluid flows
    // around them without displacing the diagram topology.
    const pinned = species === 'cil-plus';

    return {
      id,
      x,
      y,
      w,
      h,
      species,
      pinned,
      ...physics,
    };
  });
}

/**
 * Initialise all cell bodies inside an SPH world through the generic bridge
 * interface.  The bridge is typed as `any` so this file stays decoupled from
 * a specific SPH implementation (Rapier, custom WASM, etc.).
 *
 * Expected bridge surface:
 *   sphBridge.createRigidBody(config: CellPhysicsConfig): string | number
 *   sphBridge.pinBody?(bodyHandle: string | number): void          // optional
 *
 * The function is async because some bridges initialise asynchronously
 * (e.g. WASM module load) and may return Promises from createRigidBody.
 */
export async function initCellBodies(
  sphBridge: any,
  cells: Array<{ id: string; bbox: { min: number[]; max: number[] }; species: string }>
): Promise<void> {
  // Ensure species physics data is loaded before resolving configs
  await loadSpeciesPhysics();

  if (!sphBridge || typeof sphBridge.createRigidBody !== 'function') {
    throw new Error(
      'cell-body-bridge: sphBridge must expose a createRigidBody(config) method.'
    );
  }

  const configs = cellsToBodies(cells);

  for (const config of configs) {
    // createRigidBody may be sync or async — await covers both cases.
    const handle = await sphBridge.createRigidBody(config);

    if (config.pinned && typeof sphBridge.pinBody === 'function') {
      await sphBridge.pinBody(handle);
    }
  }
}

/**
 * Produce a default set of dam-break fluid blocks that surround the cell
 * layout without overlapping any rigid body.
 *
 * Strategy:
 *  - Left column:  full-height strip to the left of the leftmost cell
 *  - Right column: full-height strip to the right of the rightmost cell
 *  - Top band:     horizontal band above the topmost cell
 *
 * Each block is tagged with a numeric species index (0 = fluid) so the SPH
 * particle emitter can colour them correctly.
 *
 * @param worldW  Total world width  (simulation units)
 * @param worldH  Total world height (simulation units)
 * @param cellBodies  Already-resolved cell configs (used for layout bounds)
 */
export function defaultFluidLayout(
  worldW: number,
  worldH: number,
  cellBodies: CellPhysicsConfig[]
): Array<{ x: number; y: number; w: number; h: number; species: number }> {
  // Fluid species index (always 0)
  const FLUID_SPECIES = 0;

  // Margins so fluid blocks don't kiss the rigid-body edges
  const MARGIN = 8;
  // Minimum block dimension — skip degenerate blocks
  const MIN_DIM = 16;

  if (cellBodies.length === 0) {
    // No cells → single full-world fluid block
    return [{ x: worldW / 2, y: worldH / 2, w: worldW, h: worldH, species: FLUID_SPECIES }];
  }

  // Compute the axis-aligned bounding box of all cell bodies
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const b of cellBodies) {
    minX = Math.min(minX, b.x - b.w / 2);
    minY = Math.min(minY, b.y - b.h / 2);
    maxX = Math.max(maxX, b.x + b.w / 2);
    maxY = Math.max(maxY, b.y + b.h / 2);
  }

  const blocks: Array<{ x: number; y: number; w: number; h: number; species: number }> = [];

  // Left fluid column
  const leftW = minX - MARGIN;
  if (leftW >= MIN_DIM) {
    blocks.push({ x: leftW / 2, y: worldH / 2, w: leftW, h: worldH, species: FLUID_SPECIES });
  }

  // Right fluid column
  const rightStart = maxX + MARGIN;
  const rightW = worldW - rightStart;
  if (rightW >= MIN_DIM) {
    blocks.push({ x: rightStart + rightW / 2, y: worldH / 2, w: rightW, h: worldH, species: FLUID_SPECIES });
  }

  // Top fluid band (spanning the gap between left and right columns)
  const topStart = 0;
  const topH = minY - MARGIN;
  const bandX = minX - MARGIN;
  const bandW = (maxX + MARGIN) - bandX;
  if (topH >= MIN_DIM && bandW >= MIN_DIM) {
    blocks.push({ x: bandX + bandW / 2, y: topH / 2, w: bandW, h: topH, species: FLUID_SPECIES });
  }

  return blocks;
}
