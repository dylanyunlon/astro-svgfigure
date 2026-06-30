/**
 * cell-interaction-physics.ts — M751: Cell Interaction Physics
 *
 * Five interaction modes for the cell pub/sub loop dynamic world:
 *
 *   1. **Drag**    — pointer-down on a cell → kinematic move with the cursor,
 *                    neighbours receive repulsion impulses via the force field.
 *   2. **Throw**   — pointer-up after drag → release velocity is injected as an
 *                    impulse; the cell decelerates under damping + gravity.
 *   3. **Pin**     — double-tap / right-click toggles a cell's pinned state;
 *                    pinned cells have infinite mass and anchor the topology.
 *   4. **Inject**  — shift+click spawns a force impulse radially outward from
 *                    the click point, pushing all cells within blast radius.
 *   5. **Gravity** — a persistent downward (or configurable) acceleration
 *                    applied each physics step; cells settle toward the bottom
 *                    of the viewport unless pinned or actively dragged.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Architecture
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * This module sits between CellInteraction.ts (DOM events + visual feedback)
 * and physics-bridge.ts (WebWorker force computation via comlink).  It owns
 * the *interaction-driven* physics state — velocities, impulses, pin flags,
 * gravity — and publishes updates through the Apollo pub/sub channel system
 * via `cell-interaction` CustomEvents that CellEventSource consumers can
 * subscribe to.
 *
 * ```
 * pointer events ──→ CellInteractionPhysics ──→ force_field delta
 *                          │                          │
 *                          ├─ drag kinematic state     ├─ POST /api/cell/publish
 *                          ├─ throw impulse queue      │   (via CellEventSource)
 *                          ├─ pin registry             │
 *                          ├─ inject blast queue       ▼
 *                          └─ gravity accumulator    DataNotifier broadcast
 * ```
 *
 * The simulation uses semi-implicit Euler integration (matching
 * physics-animation.ts PhysicalSync) with a fixed-timestep accumulator
 * so behaviour is frame-rate independent.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Data sources
 * ─────────────────────────────────────────────────────────────────────────────
 *  • channels/physics/cell_registry.json   — initial bbox + species per cell
 *  • channels/physics/species_physics.json — mass, friction, restitution, buoyancy
 *  • channels/physics/force_field.json     — current force vectors (read)
 *  • channels/physics/edge_routes.json     — edge constraints (pin anchoring)
 *  • channels/physics/cell_groups.json     — community grouping (group drag)
 *
 * Research: xiaodi #M751 — cell-pubsub-loop
 * ─────────────────────────────────────────────────────────────────────────────
 */

// ─── M1294: Spatial hash — O(n) broad-phase optimisation ────────────────────
// SpatialHashGrid replaces the O(n²) brute-force loops in Particle Life,
// collision resolution, chemotaxis pre-pass, quorum sensing and community
// detection.  Grid cell size = interaction_radius (≈ 300 px) so a 3×3 query
// covers all pairs that can possibly interact.  The grid is rebuilt every
// physics sub-step (O(n) inserts) and reused for all that sub-step's queries.
import { SpatialHashGrid } from './sph/spatial-hash';

// Grid size for collision / AABB overlap pass (cells are ~100-200 px wide).
const SPATIAL_GRID_SIZE = 150;

// ─── Particle Life interaction matrix ────────────────────────────────────────
// Loaded from channels/physics/species_interaction_matrix.json.  Each species
// pair has an attract/repel coefficient G (G>0 attract, G<0 repel); within the
// interaction radius the pairwise force is G / distance, à la
// https://github.com/hunar4321/particle-life.  Complex self-organising structure
// emerges from this single simple rule, mapped onto the Transformer data flow.
import speciesInteractionMatrix from '../../channels/physics/species_interaction_matrix.json';

interface SpeciesInteractionMatrix {
  interaction_radius: number;
  matrix: Record<string, Record<string, number>>;
  description?: Record<string, string>;
}

const PARTICLE_LIFE = speciesInteractionMatrix as SpeciesInteractionMatrix;

// ─── Environment parameters ──────────────────────────────────────────────────
// M1281: ambient medium physics loaded from channels/physics/environment.json.
// Drives brownian motion, a global laminar flow field, sedimentation gravity,
// soft-wall boundary repulsion, and a temperature gradient that locally lowers
// viscosity (cells nearer the warm centre move more freely).
import environmentJson from '../../channels/physics/environment.json';

interface EnvironmentParams {
  medium: { type: string; viscosity: number; density: number; temperature: number; pH: number };
  flow_field: { type: string; direction: [number, number]; speed: number; turbulence: number };
  light: { direction: number[]; intensity: number; color: number[]; ambient: number };
  gradients: {
    temperature: { center: [number, number]; radius: number; delta: number };
    nutrient: { center: [number, number]; radius: number; concentration: number };
    signal_molecule: { sources: unknown[]; decay_rate: number; diffusion: number };
  };
  boundaries: {
    type: string;
    repel_force: number;
    margin: number;
    width: number;
    height: number;
  };
  gravity: { x: number; y: number; note?: string };
  brownian_noise: number;
  surface_tension_at_boundary: number;
}

const ENVIRONMENT = environmentJson as EnvironmentParams;

// ─── Types ──────────────────────────────────────────────────────────────────

export interface Vec2 {
  x: number;
  y: number;
}

/** Minimal cell descriptor consumed by the interaction system. */
export interface InteractionCell {
  cell_id: string;
  bbox: { x: number; y: number; w: number; h: number };
  z: number;
  species: string;
}

/** Per-species physics properties (mirrors channels/physics/species_physics.json). */
export interface SpeciesPhysics {
  mass: number;
  friction: number;
  restitution: number;
  buoyancy: number;
  viscosity?: number;
  adhesion?: number;
  chemotaxis_range?: number;
  preferred_neighbors?: number;
}

/** Internal physics body for a single cell. */
interface CellBody {
  id: string;
  // Position (center of bbox, cell-space)
  x: number;
  y: number;
  // Velocity
  vx: number;
  vy: number;
  // Dimensions (full width/height)
  w: number;
  h: number;
  // Physical properties
  mass: number;
  friction: number;
  restitution: number;
  buoyancy: number;
  // State flags
  pinned: boolean;
  dragging: boolean;
  // Accumulated impulse (cleared after each step)
  impulseX: number;
  impulseY: number;
  // Rest position (where the cell was before any interaction)
  restX: number;
  restY: number;
  // Species tag
  species: string;
  z: number;
  // M1282: Energy metabolism [0, 1]
  energy: number;
  // M1282: lifecycle flags derived from energy
  divisionReady: boolean;
  collisionCount: number;
  // M1288: Division cooldown — timestamp (ms) after which division is allowed again
  divisionCooldownEnd: number;
  // M1289: Apoptosis — triggered when energy reaches 0
  apoptosisActive: boolean;
  apoptosisStartTime: number; // ms timestamp when apoptosis began
  // M1290: Community detection — assigned community ID (connected component)
  communityId: number;
  // Quorum-sensing cluster factor: scales collision separation/bounce.
  // 1 = normal repulsion, <1 = reduced repulsion (cells pack tighter when
  // quorum is reached). Default 1. Set via setClusterFactor().
  clusterFactor?: number;
}

/** Radial blast impulse queued by inject(). */
interface BlastImpulse {
  originX: number;
  originY: number;
  radius: number;
  magnitude: number;
  timestamp: number;
}


// ─── M1283: Quorum Sensing — Signal Particles ────────────────────────────────

/** A diffusing signal molecule emitted by a cell for quorum sensing. */
export interface SignalParticle {
  x: number;
  y: number;
  alpha: number;
  species: string;
  vx: number;
  vy: number;
}

/** Snapshot of a cell's interaction state — emitted via events. */
export interface CellInteractionState {
  cell_id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  pinned: boolean;
  dragging: boolean;
  // M1282: energy metabolism
  energy: number;
  divisionReady: boolean;
}

/** Force delta produced by interaction physics for a single cell. */
export interface InteractionForce {
  cell_id: string;
  dx: number;
  dy: number;
}

/** Configuration for CellInteractionPhysics. */
export interface CellInteractionPhysicsOptions {
  /** Gravity vector in cell-space units/s². Default: { x: 0, y: 98.1 } (downward). */
  gravity?: Vec2;
  /** Global velocity damping per step (0 = no damping, 1 = full stop). Default: 0.02. */
  damping?: number;
  /** Fixed simulation timestep in seconds. Default: 1/60. */
  fixedStep?: number;
  /** Maximum sub-steps per frame to prevent spiral of death. Default: 4. */
  maxSubsteps?: number;
  /** Throw velocity multiplier — scales pointer release velocity. Default: 1.5. */
  throwScale?: number;
  /** Maximum throw speed (cell-space units/s). Prevents launch-to-infinity. Default: 2000. */
  throwMaxSpeed?: number;
  /** Inject blast default radius (cell-space units). Default: 200. */
  injectRadius?: number;
  /** Inject blast default magnitude. Default: 8000. */
  injectMagnitude?: number;
  /** Drag neighbour repulsion radius (cell-space units). Default: 180. */
  dragRepulsionRadius?: number;
  /** Drag neighbour repulsion strength. Default: 400. */
  dragRepulsionStrength?: number;
  /** Spring stiffness pulling cells back to rest position when not pinned. Default: 0 (disabled). */
  restoreStiffness?: number;
  /** Boundary rectangle { x, y, w, h } — cells bounce off edges. Null = unbounded. */
  boundary?: { x: number; y: number; w: number; h: number } | null;
  /** Callback invoked each step with the interaction force deltas. */
  onForces?: (forces: InteractionForce[]) => void;
  /** Callback invoked when a cell's pin state changes. */
  onPinChange?: (cellId: string, pinned: boolean) => void;
  /** Callback invoked when a cell is picked up (drag start). */
  onDragStart?: (cellId: string) => void;
  /** Callback invoked when a cell is released (throw). */
  onThrow?: (cellId: string, velocity: Vec2) => void;
  /** Species physics lookup table. Falls back to DEFAULT_SPECIES_PHYSICS. */
  speciesPhysics?: Record<string, SpeciesPhysics>;
}

// ─── Constants ──────────────────────────────────────────────────────────────

const DEFAULT_GRAVITY: Vec2           = { x: 0, y: 98.1 };
const DEFAULT_DAMPING                 = 0.02;
const DEFAULT_FIXED_STEP              = 1 / 60;
const DEFAULT_MAX_SUBSTEPS            = 4;
const DEFAULT_THROW_SCALE             = 1.5;
const DEFAULT_THROW_MAX_SPEED         = 2000;
const DEFAULT_INJECT_RADIUS           = 200;
const DEFAULT_INJECT_MAGNITUDE        = 8000;
const DEFAULT_DRAG_REPULSION_RADIUS   = 180;
const DEFAULT_DRAG_REPULSION_STRENGTH = 400;
const DEFAULT_RESTORE_STIFFNESS       = 0;
const BLAST_DECAY_MS                  = 300;

// M1282: Energy metabolism constants (sourced from channels/physics/cell_lifecycle.json)
const ENERGY_BASE_CONSUMPTION         = 0.01;   // per frame base drain
const ENERGY_MOVEMENT_COST            = 0.001;  // * |velocity| per frame
const ENERGY_COLLISION_COST           = 0.005;  // per collision event
const ENERGY_REGENERATION_RATE        = 0.008;  // per frame recovery
const ENERGY_MAX                      = 1.0;
const ENERGY_INITIAL                  = 1.0;
const ENERGY_APOPTOSIS_THRESHOLD      = 0.05;   // below this → opacity fade
const ENERGY_DIVISION_THRESHOLD       = 0.90;   // above this → divisionReady

// M1288: Cell division constants
const DIVISION_ENERGY_TRIGGER         = 0.95;   // energy must exceed this to actually divide
const DIVISION_ENERGY_SPLIT           = 0.5;    // each daughter receives this fraction of parent energy
const DIVISION_POSITION_OFFSET        = 30;     // random spawn offset in px
const DIVISION_COOLDOWN_MS            = 5000;   // minimum ms between divisions for a given cell

// M1289: Apoptosis constants
const APOPTOSIS_DELAY_MS              = 3000;   // duration of apoptosis phase (ms)

const DEFAULT_SPECIES_PHYSICS: SpeciesPhysics = {
  mass: 75,
  friction: 0.5,
  restitution: 0.3,
  buoyancy: 0.5,
};

// M1293: Default glow colors per species — mirrors gpu-render-loop SPECIES_MATERIAL albedos.
// Used as the starting point for color evolution and as the "species identity color"
// that dominance pressure will lerp a cell's glow toward.
const SPECIES_DEFAULT_GLOW: Record<string, [number, number, number]> = {
  'cil-eye':         [0.247, 0.318, 0.71],
  'cil-bolt':        [1.0,   0.435, 0.0],
  'cil-vector':      [0.18,  0.49,  0.196],
  'cil-plus':        [0.776, 0.157, 0.157],
  'cil-arrow-right': [0.271, 0.353, 0.392],
  'cil-filter':      [0.4,   0.3,   0.6],
  'cil-layers':      [0.1,   0.6,   0.7],
  'cil-loop':        [0.8,   0.6,   0.1],
  'cil-code':        [0.3,   0.7,   0.5],
  'cil-graph':       [0.6,   0.2,   0.7],
};

// ─── Utility ────────────────────────────────────────────────────────────────

function clampMag(vx: number, vy: number, maxMag: number): Vec2 {
  const mag = Math.sqrt(vx * vx + vy * vy);
  if (mag <= maxMag || mag === 0) return { x: vx, y: vy };
  const s = maxMag / mag;
  return { x: vx * s, y: vy * s };
}

function dist2(ax: number, ay: number, bx: number, by: number): number {
  const dx = ax - bx;
  const dy = ay - by;
  return dx * dx + dy * dy;
}

// ─── CellInteractionPhysics ─────────────────────────────────────────────────

/**
 * CellInteractionPhysics
 *
 * Manages the physics state for drag/throw/pin/inject/gravity interactions.
 * Call `step(nowMs)` each frame (typically from a rAF loop or PixiJS Ticker).
 *
 * ```ts
 * const physics = new CellInteractionPhysics(cells, { gravity: { x: 0, y: 98.1 } });
 * // In your frame loop:
 * physics.step(performance.now());
 * // Interaction:
 * physics.dragStart('self_attn', pointerX, pointerY);
 * physics.dragMove(pointerX, pointerY);
 * physics.dragEnd();
 * physics.togglePin('add_norm1');
 * physics.inject(clickX, clickY);
 * ```
 */
export class CellInteractionPhysics {
  // ── Configuration ──
  private readonly gravity: Vec2;
  private readonly damping: number;
  private readonly fixedStep: number;
  private readonly maxSubsteps: number;
  private readonly throwScale: number;
  private readonly throwMaxSpeed: number;
  private readonly injectRadius: number;
  private readonly injectMagnitude: number;
  private readonly dragRepulsionRadius: number;
  private readonly dragRepulsionStrength: number;
  private readonly restoreStiffness: number;
  private readonly boundary: { x: number; y: number; w: number; h: number } | null;
  private readonly onForces: ((forces: InteractionForce[]) => void) | null;
  private readonly onPinChange: ((cellId: string, pinned: boolean) => void) | null;
  private readonly onDragStart: ((cellId: string) => void) | null;
  private readonly onThrow: ((cellId: string, velocity: Vec2) => void) | null;
  private readonly speciesLookup: Record<string, SpeciesPhysics>;

  // ── State ──
  private bodies: Map<string, CellBody> = new Map();
  private blastQueue: BlastImpulse[] = [];
  private accumulator = 0;
  private lastTime = 0;
  private draggedId: string | null = null;
  private dragOffsetX = 0;
  private dragOffsetY = 0;
  private dragPrevX = 0;
  private dragPrevY = 0;
  private dragPrevTime = 0;
  private dragVelX = 0;
  private dragVelY = 0;
  private disposed = false;

  // ── M1283: Quorum Sensing — Signal Particles ──
  /** Diffusing signal molecules for quorum-sensing detection. */
  signalParticles: SignalParticle[] = [];

  /** Frame counter for throttling signal emission (emit every 60 frames). */
  private _signalFrameCount = 0;

  /** Per-cell quorum active state (true when ≥ quorum_threshold neighbours in range). */
  private _quorumActive: Map<string, boolean> = new Map();

  // ── M1288: Cell division counter — suffix appended to child IDs ──
  private _divisionCounter = 0;

  // ── M1289: Apoptosis original size registry ──
  // Stores original {w, h} per cell at the moment apoptosis activates,
  // so we can shrink linearly from original size to 0.
  private _apoptosisOrigSize: Map<string, { w: number; h: number }> = new Map();

  // ── M1290: Community detection state ──
  /** Frame counter — community detection runs every 120 frames (~2 s at 60 fps). */
  private _communityFrameCount = 0;
  /** Per-cell assigned community ID (connected component index, 0-based). */
  private _communityMap: Map<string, number> = new Map();

  // ── M1293: Species color evolution ──
  /**
   * Per-cell current glow color [r, g, b] in [0, 1].  Initialised from the
   * species default palette and gradually drifts toward the dominant neighbour
   * species color every COLOR_EVOLVE_INTERVAL frames.
   */
  private _cellGlowColor: Map<string, [number, number, number]> = new Map();
  /** Frame counter — color evolution runs every 180 frames (~3 s at 60 fps). */
  private _colorEvolveFrameCount = 0;

  // ── Velocity history for smooth throw (ring buffer, last 4 frames) ──
  private readonly velHistory: Array<{ vx: number; vy: number; dt: number }> = [];
  private readonly VEL_HISTORY_LEN = 4;

  // ── M1294: Reusable spatial hash grids ──
  // _plGrid  — cell size = interaction_radius; covers Particle Life, community
  //            detection and quorum sensing (all use the same radius).
  // _collGrid — cell size = SPATIAL_GRID_SIZE (150 px); covers AABB overlap
  //            resolution and same-species chemotaxis (cells ≈ 100-200 px wide).
  // Both are rebuilt (clear + insert) once per physics sub-step and then
  // queried by every O(n²) loop, reducing broad-phase work from O(n²) to O(n).
  private _plGrid!: SpatialHashGrid;
  private _collGrid!: SpatialHashGrid;

  constructor(
    cells: InteractionCell[],
    opts: CellInteractionPhysicsOptions = {},
  ) {
    this.gravity               = opts.gravity               ?? { ...DEFAULT_GRAVITY };
    this.damping               = opts.damping               ?? DEFAULT_DAMPING;
    this.fixedStep             = opts.fixedStep             ?? DEFAULT_FIXED_STEP;
    this.maxSubsteps           = opts.maxSubsteps           ?? DEFAULT_MAX_SUBSTEPS;
    this.throwScale            = opts.throwScale            ?? DEFAULT_THROW_SCALE;
    this.throwMaxSpeed         = opts.throwMaxSpeed         ?? DEFAULT_THROW_MAX_SPEED;
    this.injectRadius          = opts.injectRadius          ?? DEFAULT_INJECT_RADIUS;
    this.injectMagnitude       = opts.injectMagnitude       ?? DEFAULT_INJECT_MAGNITUDE;
    this.dragRepulsionRadius   = opts.dragRepulsionRadius   ?? DEFAULT_DRAG_REPULSION_RADIUS;
    this.dragRepulsionStrength = opts.dragRepulsionStrength ?? DEFAULT_DRAG_REPULSION_STRENGTH;
    this.restoreStiffness      = opts.restoreStiffness      ?? DEFAULT_RESTORE_STIFFNESS;
    this.boundary              = opts.boundary              ?? null;
    this.onForces              = opts.onForces              ?? null;
    this.onPinChange           = opts.onPinChange           ?? null;
    this.onDragStart           = opts.onDragStart           ?? null;
    this.onThrow               = opts.onThrow               ?? null;
    this.speciesLookup         = opts.speciesPhysics        ?? {};

    // M1294: Initialise spatial hash grids.
    // tableSize is sized to the expected cell count (next power-of-2); the
    // grids are reused every sub-step via clear() so allocation happens once.
    const tableSize = Math.max(4096, cells.length * 2);
    this._plGrid   = new SpatialHashGrid(PARTICLE_LIFE.interaction_radius, tableSize);
    this._collGrid = new SpatialHashGrid(SPATIAL_GRID_SIZE, tableSize);

    this._initBodies(cells);
  }

  // ─── Initialisation ─────────────────────────────────────────────────────

  private _initBodies(cells: InteractionCell[]): void {
    for (const cell of cells) {
      const { x, y, w, h } = cell.bbox;
      const cx = x + w / 2;
      const cy = y + h / 2;
      const sp = this.speciesLookup[cell.species] ?? DEFAULT_SPECIES_PHYSICS;

      // Structural anchors: "cil-plus" species (residual add nodes) start pinned,
      // matching the convention in cell-body-bridge.ts.
      const pinned = cell.species === 'cil-plus';

      const body: CellBody = {
        id: cell.cell_id,
        x: cx,
        y: cy,
        vx: 0,
        vy: 0,
        w,
        h,
        mass: sp.mass,
        friction: sp.friction,
        restitution: sp.restitution,
        buoyancy: sp.buoyancy,
        pinned,
        dragging: false,
        impulseX: 0,
        impulseY: 0,
        restX: cx,
        restY: cy,
        species: cell.species,
        z: cell.z,
        energy: ENERGY_INITIAL,
        divisionReady: false,
        collisionCount: 0,
        divisionCooldownEnd: 0,
        apoptosisActive: false,
        apoptosisStartTime: 0,
        communityId: 0,
      };

      this.bodies.set(cell.cell_id, body);

      // M1293: Seed the cell's glow color from its species default palette.
      const defaultGlow = SPECIES_DEFAULT_GLOW[cell.species] ?? [0.5, 0.5, 0.5];
      this._cellGlowColor.set(cell.cell_id, [...defaultGlow] as [number, number, number]);
    }
  }

  // ─── Cell management ────────────────────────────────────────────────────

  /** Add or replace a cell body at runtime (e.g. after topology change). */
  addCell(cell: InteractionCell): void {
    const { x, y, w, h } = cell.bbox;
    const cx = x + w / 2;
    const cy = y + h / 2;
    const sp = this.speciesLookup[cell.species] ?? DEFAULT_SPECIES_PHYSICS;

    this.bodies.set(cell.cell_id, {
      id: cell.cell_id,
      x: cx, y: cy,
      vx: 0, vy: 0,
      w, h,
      mass: sp.mass,
      friction: sp.friction,
      restitution: sp.restitution,
      buoyancy: sp.buoyancy,
      pinned: cell.species === 'cil-plus',
      dragging: false,
      impulseX: 0, impulseY: 0,
      restX: cx, restY: cy,
      species: cell.species,
      z: cell.z,
      energy: ENERGY_INITIAL,
      divisionReady: false,
      collisionCount: 0,
      divisionCooldownEnd: 0,
      apoptosisActive: false,
      apoptosisStartTime: 0,
      communityId: 0,
    });

    // M1293: Seed glow color for the new cell.
    const defaultGlow = SPECIES_DEFAULT_GLOW[cell.species] ?? [0.5, 0.5, 0.5];
    this._cellGlowColor.set(cell.cell_id, [...defaultGlow] as [number, number, number]);
  }

  /** Remove a cell body. If it's currently dragged, the drag is cancelled. */
  removeCell(cellId: string): void {
    if (this.draggedId === cellId) this.dragCancel();
    this.bodies.delete(cellId);
    this._cellGlowColor.delete(cellId);
  }

  /** Update a cell's rest position (e.g. after ELK re-layout). */
  setRestPosition(cellId: string, x: number, y: number): void {
    const b = this.bodies.get(cellId);
    if (b) { b.restX = x; b.restY = y; }
  }

  /**
   * Set a cell's quorum-sensing cluster factor. Scales how strongly the cell
   * is pushed apart during collision resolution: 1 = normal repulsion, values
   * <1 reduce repulsion so cells pack tighter (cluster). Clamped to [0, 1].
   * Used by the quorum-sensing response — when a cell has enough same-species
   * neighbours it lowers its repulsion and aggregates.
   */
  setClusterFactor(cellId: string, factor: number): void {
    const b = this.bodies.get(cellId);
    if (b) b.clusterFactor = Math.max(0, Math.min(1, factor));
  }

  // ─── 1. Drag ────────────────────────────────────────────────────────────

  /**
   * Begin dragging a cell. The cell becomes kinematic (follows the pointer)
   * and its velocity is zeroed.
   *
   * @param cellId    ID of the cell to drag.
   * @param pointerX  Current pointer position in cell-space.
   * @param pointerY  Current pointer position in cell-space.
   */
  dragStart(cellId: string, pointerX: number, pointerY: number): void {
    const body = this.bodies.get(cellId);
    if (!body) return;

    // Cancel any existing drag
    if (this.draggedId && this.draggedId !== cellId) this.dragCancel();

    body.dragging = true;
    body.vx = 0;
    body.vy = 0;
    this.draggedId = cellId;
    this.dragOffsetX = body.x - pointerX;
    this.dragOffsetY = body.y - pointerY;
    this.dragPrevX = pointerX;
    this.dragPrevY = pointerY;
    this.dragPrevTime = performance.now();
    this.dragVelX = 0;
    this.dragVelY = 0;
    this.velHistory.length = 0;

    this.onDragStart?.(cellId);

    window.dispatchEvent(new CustomEvent('cell-drag-start', {
      detail: { cellId, x: body.x, y: body.y },
    }));
  }

  /**
   * Update the dragged cell's position to follow the pointer.
   * Also tracks pointer velocity for throw calculation.
   */
  dragMove(pointerX: number, pointerY: number): void {
    if (!this.draggedId) return;
    const body = this.bodies.get(this.draggedId);
    if (!body) return;

    const now = performance.now();
    const dt = (now - this.dragPrevTime) / 1000;

    // Track velocity as a smoothed ring buffer (avoids jitter on throw)
    if (dt > 0.001) {
      const rawVx = (pointerX - this.dragPrevX) / dt;
      const rawVy = (pointerY - this.dragPrevY) / dt;

      this.velHistory.push({ vx: rawVx, vy: rawVy, dt });
      if (this.velHistory.length > this.VEL_HISTORY_LEN) {
        this.velHistory.shift();
      }
    }

    // Move cell to pointer (kinematic)
    body.x = pointerX + this.dragOffsetX;
    body.y = pointerY + this.dragOffsetY;

    this.dragPrevX = pointerX;
    this.dragPrevY = pointerY;
    this.dragPrevTime = now;

    window.dispatchEvent(new CustomEvent('cell-drag-move', {
      detail: { cellId: this.draggedId, x: body.x, y: body.y },
    }));
  }

  /**
   * End the drag and apply throw velocity.
   * The release velocity is a weighted average of recent pointer velocities,
   * scaled by `throwScale` and clamped to `throwMaxSpeed`.
   */
  dragEnd(): void {
    if (!this.draggedId) return;
    const body = this.bodies.get(this.draggedId);
    if (!body) { this.draggedId = null; return; }

    body.dragging = false;

    // Compute weighted average velocity from history (more recent = more weight)
    const throwVel = this._computeThrowVelocity();
    body.vx = throwVel.x;
    body.vy = throwVel.y;

    const cellId = this.draggedId;
    this.draggedId = null;

    this.onThrow?.(cellId, throwVel);

    window.dispatchEvent(new CustomEvent('cell-throw', {
      detail: { cellId, vx: throwVel.x, vy: throwVel.y },
    }));
  }

  /** Cancel drag without applying throw velocity. */
  dragCancel(): void {
    if (!this.draggedId) return;
    const body = this.bodies.get(this.draggedId);
    if (body) {
      body.dragging = false;
      body.vx = 0;
      body.vy = 0;
    }
    this.draggedId = null;
    this.velHistory.length = 0;
  }

  /** Returns the ID of the currently dragged cell, or null. */
  get draggedCellId(): string | null {
    return this.draggedId;
  }

  private _computeThrowVelocity(): Vec2 {
    if (this.velHistory.length === 0) return { x: 0, y: 0 };

    // Exponential weighting: most recent entry gets the highest weight.
    let totalWeight = 0;
    let avgVx = 0;
    let avgVy = 0;

    for (let i = 0; i < this.velHistory.length; i++) {
      const weight = i + 1; // linear ramp; simple and effective
      const entry = this.velHistory[i];
      avgVx += entry.vx * weight;
      avgVy += entry.vy * weight;
      totalWeight += weight;
    }

    if (totalWeight > 0) {
      avgVx /= totalWeight;
      avgVy /= totalWeight;
    }

    // Scale and clamp
    avgVx *= this.throwScale;
    avgVy *= this.throwScale;

    return clampMag(avgVx, avgVy, this.throwMaxSpeed);
  }

  // ─── 2. Throw (integrated into drag end — see dragEnd) ─────────────────

  /**
   * Apply an immediate velocity impulse to a cell (manual throw API).
   * Useful for programmatic throws triggered by pub/sub events.
   */
  throwCell(cellId: string, vx: number, vy: number): void {
    const body = this.bodies.get(cellId);
    if (!body || body.pinned || body.dragging) return;

    const clamped = clampMag(vx, vy, this.throwMaxSpeed);
    body.vx += clamped.x;
    body.vy += clamped.y;

    window.dispatchEvent(new CustomEvent('cell-throw', {
      detail: { cellId, vx: clamped.x, vy: clamped.y },
    }));
  }

  // ─── 3. Pin ─────────────────────────────────────────────────────────────

  /**
   * Toggle a cell's pinned state. Pinned cells have infinite effective mass:
   * they don't move under gravity, throw, or inject — but can still be dragged.
   * Returns the new pinned state.
   */
  togglePin(cellId: string): boolean {
    const body = this.bodies.get(cellId);
    if (!body) return false;

    body.pinned = !body.pinned;

    // Stop motion when pinning
    if (body.pinned) {
      body.vx = 0;
      body.vy = 0;
      body.impulseX = 0;
      body.impulseY = 0;
    }

    this.onPinChange?.(cellId, body.pinned);

    window.dispatchEvent(new CustomEvent('cell-pin-change', {
      detail: { cellId, pinned: body.pinned },
    }));

    return body.pinned;
  }

  /** Explicitly set a cell's pin state (no toggle). */
  setPin(cellId: string, pinned: boolean): void {
    const body = this.bodies.get(cellId);
    if (!body || body.pinned === pinned) return;

    body.pinned = pinned;
    if (pinned) {
      body.vx = 0;
      body.vy = 0;
      body.impulseX = 0;
      body.impulseY = 0;
    }

    this.onPinChange?.(cellId, pinned);

    window.dispatchEvent(new CustomEvent('cell-pin-change', {
      detail: { cellId, pinned },
    }));
  }

  /** Check if a cell is currently pinned. */
  isPinned(cellId: string): boolean {
    return this.bodies.get(cellId)?.pinned ?? false;
  }

  /** Pin all cells (freeze the layout). */
  pinAll(): void {
    for (const body of this.bodies.values()) {
      if (!body.pinned) {
        body.pinned = true;
        body.vx = 0;
        body.vy = 0;
        this.onPinChange?.(body.id, true);
      }
    }
  }

  /** Unpin all cells (let them fall under gravity). */
  unpinAll(): void {
    for (const body of this.bodies.values()) {
      if (body.pinned) {
        body.pinned = false;
        this.onPinChange?.(body.id, false);
      }
    }
  }

  // ─── 4. Inject ──────────────────────────────────────────────────────────

  /**
   * Inject a radial blast impulse at (originX, originY) in cell-space.
   * All non-pinned cells within `radius` receive an outward impulse scaled
   * by distance falloff.  The blast decays over BLAST_DECAY_MS.
   *
   * @param originX   Blast center X (cell-space).
   * @param originY   Blast center Y (cell-space).
   * @param radius    Optional override for blast radius.
   * @param magnitude Optional override for blast strength.
   */
  inject(
    originX: number,
    originY: number,
    radius?: number,
    magnitude?: number,
  ): void {
    const blast: BlastImpulse = {
      originX,
      originY,
      radius: radius ?? this.injectRadius,
      magnitude: magnitude ?? this.injectMagnitude,
      timestamp: performance.now(),
    };

    this.blastQueue.push(blast);

    window.dispatchEvent(new CustomEvent('cell-inject', {
      detail: { x: originX, y: originY, radius: blast.radius, magnitude: blast.magnitude },
    }));
  }

  // ─── 5. Gravity ─────────────────────────────────────────────────────────

  /** Update the gravity vector at runtime. */
  setGravity(x: number, y: number): void {
    this.gravity.x = x;
    this.gravity.y = y;
  }

  /** Get the current gravity vector. */
  getGravity(): Readonly<Vec2> {
    return this.gravity;
  }

  // ─── Simulation step ────────────────────────────────────────────────────

  /**
   * Advance the interaction physics simulation.
   * Call once per frame with the current timestamp (ms).
   *
   * @param nowMs  Current time in milliseconds (e.g. performance.now()).
   * @returns      Array of per-cell position deltas since last step.
   */
  step(nowMs: number): InteractionForce[] {
    if (this.disposed) return [];

    if (this.lastTime === 0) {
      this.lastTime = nowMs;
      return [];
    }

    const deltaS = Math.min((nowMs - this.lastTime) / 1000, 0.25);
    this.lastTime = nowMs;
    this.accumulator += deltaS;

    // Snapshot positions before stepping (for delta computation)
    const prevPositions = new Map<string, { x: number; y: number }>();
    for (const [id, body] of this.bodies) {
      prevPositions.set(id, { x: body.x, y: body.y });
    }

    // Fixed-timestep integration
    let steps = 0;
    while (this.accumulator >= this.fixedStep && steps < this.maxSubsteps) {
      this._physicsStep(this.fixedStep, nowMs);
      this.accumulator -= this.fixedStep;
      steps++;
    }

    // Expire old blasts
    this.blastQueue = this.blastQueue.filter(
      b => (nowMs - b.timestamp) < BLAST_DECAY_MS,
    );

    // ── M1283: Quorum sensing — signal particle emission & diffusion ──────
    this._signalFrameCount++;
    const SIGNAL_RADIUS = 200;          // sensing range (cell-space units)
    const QUORUM_THRESHOLD = 4;         // minimum neighbours to activate quorum
    const SIGNAL_DIFFUSION_SPEED = 50;  // px/s
    const SIGNAL_DECAY_RATE = 0.02;     // alpha *= (1 - decay_rate) per frame
    const SIGNAL_EMIT_INTERVAL = 60;    // frames between emissions per cell

    // Emit new signal particles every 60 frames
    if (this._signalFrameCount >= SIGNAL_EMIT_INTERVAL) {
      this._signalFrameCount = 0;
      for (const body of this.bodies.values()) {
        const count = 1 + Math.floor(Math.random() * 3); // 1-3 particles
        for (let si = 0; si < count; si++) {
          const angle = Math.random() * Math.PI * 2;
          this.signalParticles.push({
            x: body.x,
            y: body.y,
            alpha: 1.0,
            species: body.species,
            vx: Math.cos(angle) * SIGNAL_DIFFUSION_SPEED,
            vy: Math.sin(angle) * SIGNAL_DIFFUSION_SPEED,
          });
        }
      }
    }

    // Update existing signal particles: diffuse + decay, remove dead ones
    const frameS = this.fixedStep;
    for (const sp of this.signalParticles) {
      sp.x += sp.vx * frameS;
      sp.y += sp.vy * frameS;
      sp.alpha *= (1 - SIGNAL_DECAY_RATE);
    }
    this.signalParticles = this.signalParticles.filter(sp => sp.alpha >= 0.01);

    // Quorum detection: count same-species cells within SIGNAL_RADIUS
    // M1294: Use _plGrid for O(n) broad phase (SIGNAL_RADIUS ≤ interaction_radius).
    {
      const bodiesForQuorum = Array.from(this.bodies.values());
      const r2Signal = SIGNAL_RADIUS * SIGNAL_RADIUS;
      for (let qi = 0; qi < bodiesForQuorum.length; qi++) {
        const a = bodiesForQuorum[qi];
        let neighbourCount = 0;
        const qCandidates = this._plGrid.query(a.x, a.y);
        for (let ci = 0; ci < qCandidates.length; ci++) {
          const qj = qCandidates[ci];
          if (qj === qi) continue;
          const b = bodiesForQuorum[qj];
          // M1284: a stale/out-of-range grid index (e.g. a child cell from an
          // M1280 division that is not yet registered in the species lookup /
          // body array) can yield an undefined body. Skip it instead of
          // dereferencing, which would throw:
          //   TypeError: Cannot read properties of undefined (reading 'species')
          if (!b) continue;
          if (b.species !== a.species) continue;
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          if (dx * dx + dy * dy <= r2Signal) neighbourCount++;
        }
        const active = neighbourCount >= QUORUM_THRESHOLD;
        this._quorumActive.set(a.id, active);
      }
    }

    // ── M1290: Community detection — every 120 frames (~2 s at 60 fps) ──────
    // Uses a simple Union-Find connected-component algorithm: two cells belong
    // to the same community when their distance is below interaction_radius AND
    // the Particle Life G coefficient between their species is > 0 (attractive).
    // After labelling, same-community pairs receive a weak intra-community
    // attraction, and cross-community pairs receive a weak inter-community
    // repulsion, so communities cohere and separate over time.
    this._communityFrameCount++;
    if (this._communityFrameCount >= 120) {
      this._communityFrameCount = 0;
      this._runCommunityDetection();
    }

    // ── M1290: Apply community gravity / repulsion every frame ───────────────
    this._applyCommunityForces();

    // ── M1293: Species color evolution — every 180 frames (~3 s at 60 fps) ──
    // Each cell inspects same-species-radius neighbours, finds the dominant
    // neighbour species, and lerps its glow color 5% toward that species's
    // canonical color.  Dispatches 'species-color-evolve' so the render layer
    // (gpu-render-loop.ts) can update the PBR descriptor.
    this._colorEvolveFrameCount++;
    if (this._colorEvolveFrameCount >= 180) {
      this._colorEvolveFrameCount = 0;
      this._runSpeciesColorEvolution();
    }

    // Compute force deltas
    const forces: InteractionForce[] = [];
    for (const [id, body] of this.bodies) {
      const prev = prevPositions.get(id);
      if (!prev) continue;
      const dx = body.x - prev.x;
      const dy = body.y - prev.y;
      if (Math.abs(dx) > 0.001 || Math.abs(dy) > 0.001) {
        forces.push({ cell_id: id, dx, dy });
      }
    }

    if (forces.length > 0) {
      this.onForces?.(forces);
    }

    return forces;
  }

  private _physicsStep(dt: number, nowMs: number): void {
    // ── M1294: Rebuild spatial hash grids (O(n)) ──
    // Both grids are cleared and repopulated at the start of each sub-step so
    // that all subsequent O(n²) loops can use O(1) broad-phase queries instead.
    {
      this._plGrid.clear();
      this._collGrid.clear();
      let idx = 0;
      for (const body of this.bodies.values()) {
        this._plGrid.insert(idx, body.x, body.y);
        this._collGrid.insert(idx, body.x, body.y);
        idx++;
      }
    }
    // Indexed array for O(1) lookup by index returned from spatial hash queries
    const _bodyArr: CellBody[] = Array.from(this.bodies.values());

    // ── 1. Apply blast impulses ──
    for (const blast of this.blastQueue) {
      const age = (nowMs - blast.timestamp) / BLAST_DECAY_MS;
      const decay = Math.max(0, 1 - age); // linear falloff

      for (const body of this.bodies.values()) {
        if (body.pinned || body.dragging) continue;
        const d2 = dist2(body.x, body.y, blast.originX, blast.originY);
        const r2 = blast.radius * blast.radius;
        if (d2 >= r2 || d2 === 0) continue;

        const d = Math.sqrt(d2);
        const falloff = 1 - d / blast.radius; // 1 at center, 0 at edge
        const impulse = blast.magnitude * falloff * decay * dt;
        const invMass = 1 / body.mass;

        // Direction: outward from blast origin
        const nx = (body.x - blast.originX) / d;
        const ny = (body.y - blast.originY) / d;

        body.impulseX += nx * impulse * invMass;
        body.impulseY += ny * impulse * invMass;
      }
    }

    // ── 2. Drag repulsion (push neighbours away from dragged cell) ──
    if (this.draggedId) {
      const dragged = this.bodies.get(this.draggedId);
      if (dragged) {
        const r2 = this.dragRepulsionRadius * this.dragRepulsionRadius;

        for (const body of this.bodies.values()) {
          if (body.id === this.draggedId || body.pinned || body.dragging) continue;
          const d2 = dist2(body.x, body.y, dragged.x, dragged.y);
          if (d2 >= r2 || d2 === 0) continue;

          const d = Math.sqrt(d2);
          const falloff = 1 - d / this.dragRepulsionRadius;
          const force = this.dragRepulsionStrength * falloff * dt;
          const invMass = 1 / body.mass;

          const nx = (body.x - dragged.x) / d;
          const ny = (body.y - dragged.y) / d;

          body.impulseX += nx * force * invMass;
          body.impulseY += ny * force * invMass;
        }
      }
    }

    // ── 2b. Particle Life pairwise interaction (M1294: spatial hash O(n)) ──
    // For every ordered pair (i, j) within the interaction radius, apply a force
    // proportional to G / distance, where G = matrix[species_i][species_j].
    // G > 0 attracts i toward j, G < 0 repels.  From this trivially simple rule
    // emerges complex self-organising structure (Particle Life), here shaped by
    // the Transformer information-flow matrix in species_interaction_matrix.json.
    // Broad phase: _plGrid narrows candidates from O(n) per cell to O(k) where
    // k is the average number of cells in the 3×3 neighbourhood (typically <10).
    {
      const radius = PARTICLE_LIFE.interaction_radius;
      const r2 = radius * radius;
      const matrix = PARTICLE_LIFE.matrix;

      for (let i = 0; i < _bodyArr.length; i++) {
        const a = _bodyArr[i];
        if (a.pinned || a.dragging) continue;
        const rowA = matrix[a.species];
        if (!rowA) continue;

        // Spatial hash broad phase: only check cells in the 3×3 neighbourhood
        const candidates = this._plGrid.query(a.x, a.y);
        for (let ci = 0; ci < candidates.length; ci++) {
          const j = candidates[ci];
          if (j === i) continue;
          const b = _bodyArr[j];
          // M1284: a stale grid index (e.g. an M1280 division child not yet
          // registered in the body array) yields an undefined body. Skip it
          // instead of dereferencing, which would throw:
          //   TypeError: Cannot read properties of undefined (reading 'species')
          if (!b) continue;

          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const d2 = dx * dx + dy * dy;
          if (d2 >= r2 || d2 === 0) continue;

          const G = rowA[b.species];
          if (G === undefined || G === 0) continue;

          const dist = Math.sqrt(d2);
          const force = G / Math.max(dist, 10); // prevent divide-by-zero
          a.vx += dx * force * dt * 0.5;
          a.vy += dy * force * dt * 0.5;
        }
      }
    }

    // ── 2c. Environment physics (M1279) ──
    // Ambient medium forces sourced from channels/physics/environment.json:
    // brownian jitter, a global laminar flow drift, sedimentation gravity, soft
    // boundary repulsion, and a temperature gradient that modulates viscosity.
    {
      const env = ENVIRONMENT;
      const flowDx = env.flow_field.direction[0] * env.flow_field.speed;
      const flowDy = env.flow_field.direction[1] * env.flow_field.speed;
      const [tempCx, tempCy] = env.gradients.temperature.center;
      const tempR = env.gradients.temperature.radius;
      const margin = env.boundaries.margin;
      const repel = env.boundaries.repel_force;
      // Soft-wall extents: prefer the live boundary if present, else env dims.
      const wMinX = this.boundary ? this.boundary.x : 0;
      const wMinY = this.boundary ? this.boundary.y : 0;
      const wMaxX = this.boundary ? this.boundary.x + this.boundary.w : env.boundaries.width;
      const wMaxY = this.boundary ? this.boundary.y + this.boundary.h : env.boundaries.height;

      for (const body of this.bodies.values()) {
        if (body.pinned || body.dragging) continue;
        const invMass = 1 / body.mass;

        // (a) Brownian motion — per-frame random jitter ±brownian_noise.
        const bn = env.brownian_noise;
        body.vx += (Math.random() * 2 - 1) * bn * dt;
        body.vy += (Math.random() * 2 - 1) * bn * dt;

        // (b) Flow field — global laminar drift direction*speed*dt.
        body.vx += flowDx * dt;
        body.vy += flowDy * dt;

        // (c) Gravity — ambient sedimentation drift (mass-independent accel).
        body.vx += env.gravity.x * dt;
        body.vy += env.gravity.y * dt;

        // (d) Soft walls — when within `margin` of a wall, push back with a
        //     force that ramps from 0 at the margin edge to repel at the wall.
        const distL = body.x - wMinX;
        if (distL < margin) {
          const f = repel * (1 - distL / margin);
          body.vx += f * invMass * dt;
        }
        const distR = wMaxX - body.x;
        if (distR < margin) {
          const f = repel * (1 - distR / margin);
          body.vx -= f * invMass * dt;
        }
        const distT = body.y - wMinY;
        if (distT < margin) {
          const f = repel * (1 - distT / margin);
          body.vy += f * invMass * dt;
        }
        const distB = wMaxY - body.y;
        if (distB < margin) {
          const f = repel * (1 - distB / margin);
          body.vy -= f * invMass * dt;
        }

        // (e) Temperature gradient — nearer the warm centre, viscosity drops so
        //     the cell moves more freely (less drag); farther out it cools and
        //     thickens. warmth ∈ [0,1]; viscosity scales from medium.viscosity
        //     (at the rim/outside) down toward ~0 at the centre.
        const tdx = body.x - tempCx;
        const tdy = body.y - tempCy;
        const tdist = Math.sqrt(tdx * tdx + tdy * tdy);
        const warmth = Math.max(0, 1 - tdist / tempR);
        const localVisc = env.medium.viscosity * (1 - warmth);
        const viscFactor = Math.max(0, 1 - localVisc * dt);
        body.vx *= viscFactor;
        body.vy *= viscFactor;
      }
    }

    // ── 2d. Energy metabolism (M1282) ──
    // Per-frame energy cost: base consumption + movement + regeneration.
    // Collision cost is applied per-collision in the overlap resolution loop
    // below using body.collisionCount accumulated in the previous sub-step.
    for (const body of this.bodies.values()) {
      if (body.pinned) continue;

      // Movement cost: proportional to speed
      const speed = Math.sqrt(body.vx * body.vx + body.vy * body.vy);
      const movementDrain = ENERGY_MOVEMENT_COST * speed;

      // Collision drain accumulated from previous overlap resolution pass
      const collisionDrain = ENERGY_COLLISION_COST * body.collisionCount;
      body.collisionCount = 0; // reset for this sub-step

      // Net energy delta: regeneration minus all consumption terms
      const delta = ENERGY_REGENERATION_RATE
        - ENERGY_BASE_CONSUMPTION
        - movementDrain
        - collisionDrain;

      body.energy = Math.max(0, Math.min(ENERGY_MAX, body.energy + delta * dt));

      // M1289: Trigger apoptosis when energy reaches 0
      if (body.energy <= 0 && !body.apoptosisActive) {
        body.apoptosisActive = true;
        body.apoptosisStartTime = nowMs;
      }

      // Lifecycle state transitions derived from energy level
      // Division readiness: sufficient energy to support cell division
      body.divisionReady = body.energy > ENERGY_DIVISION_THRESHOLD;

      // Apoptosis fade: opacity linearly decreases as energy approaches 0
      // The visual opacity value is read out via getState() / CellPBRDescriptor.opacity
      if (body.energy < ENERGY_APOPTOSIS_THRESHOLD) {
        // Normalized fade: 1.0 at threshold, 0.0 at energy=0
        const fadeOpacity = body.energy / ENERGY_APOPTOSIS_THRESHOLD;
        // Store in clusterFactor channel is not appropriate; we use a dedicated
        // field. CellPBRDescriptor consumers should call getEnergyOpacity().
        // We dispatch an event so the render layer can update opacity.
        window.dispatchEvent(new CustomEvent('cell-apoptosis-fade', {
          detail: { cellId: body.id, opacity: fadeOpacity, energy: body.energy },
        }));
      }

      // Division pulse event: fire once per frame while ready
      if (body.divisionReady) {
        window.dispatchEvent(new CustomEvent('cell-division-ready', {
          detail: { cellId: body.id, energy: body.energy },
        }));
      }
    }

    // ── 2f. Apoptosis progression (M1289) ──
    // Cells with apoptosisActive shrink radius over APOPTOSIS_DELAY_MS,
    // dispatch 'cell-apoptosis' events with progress 0→1,
    // and are removed from bodies at progress=1.
    {
      const apoptosisRemove: string[] = [];

      for (const body of this.bodies.values()) {
        if (!body.apoptosisActive) continue;

        // Snapshot original dimensions on first apoptosis frame
        if (!this._apoptosisOrigSize.has(body.id)) {
          this._apoptosisOrigSize.set(body.id, { w: body.w, h: body.h });
        }

        const elapsed = nowMs - body.apoptosisStartTime;
        const progress = Math.min(1, elapsed / APOPTOSIS_DELAY_MS);

        // Shrink cell radius linearly to 0
        const orig = this._apoptosisOrigSize.get(body.id)!;
        const scale = 1 - progress;
        body.w = orig.w * scale;
        body.h = orig.h * scale;

        // Dispatch progress event
        window.dispatchEvent(new CustomEvent('cell-apoptosis', {
          detail: { cellId: body.id, progress, x: body.x, y: body.y, species: body.species },
        }));

        if (progress >= 1) {
          apoptosisRemove.push(body.id);
        }
      }

      for (const id of apoptosisRemove) {
        this._apoptosisOrigSize.delete(id);
        this.removeCell(id);
      }
    }

    // ── 2e. Cell division (M1288) ──
    // When a cell has divisionReady == true AND energy > DIVISION_ENERGY_TRIGGER
    // AND its cooldown has expired, it divides: a child CellBody is spawned
    // inheriting the parent's species, each daughter receives half the parent
    // energy, divisionReady is reset, and a cooldown is started.
    // Collect pending divisions first (avoid mutating bodies map mid-iteration).
    {
      const divisionQueue: Array<{ parent: CellBody; childId: string }> = [];

      for (const body of this.bodies.values()) {
        if (!body.divisionReady) continue;
        if (body.energy <= DIVISION_ENERGY_TRIGGER) continue;
        if (nowMs < body.divisionCooldownEnd) continue;

        const childId = `${body.id}_child_${++this._divisionCounter}`;
        divisionQueue.push({ parent: body, childId });
      }

      for (const { parent, childId } of divisionQueue) {
        // Random offset in a circle of radius DIVISION_POSITION_OFFSET
        const angle = Math.random() * Math.PI * 2;
        const childX = parent.x + Math.cos(angle) * DIVISION_POSITION_OFFSET;
        const childY = parent.y + Math.sin(angle) * DIVISION_POSITION_OFFSET;

        // Split energy equally
        const halfEnergy = parent.energy * DIVISION_ENERGY_SPLIT;
        parent.energy = halfEnergy;

        // Create child body inheriting species + dimensions from parent
        const child: CellBody = {
          id: childId,
          x: childX,
          y: childY,
          vx: 0,
          vy: 0,
          w: parent.w,
          h: parent.h,
          mass: parent.mass,
          friction: parent.friction,
          restitution: parent.restitution,
          buoyancy: parent.buoyancy,
          pinned: false,
          dragging: false,
          impulseX: 0,
          impulseY: 0,
          restX: childX,
          restY: childY,
          species: parent.species,
          z: parent.z,
          energy: halfEnergy,
          divisionReady: false,
          collisionCount: 0,
          divisionCooldownEnd: nowMs + DIVISION_COOLDOWN_MS,
          apoptosisActive: false,
          apoptosisStartTime: 0,
          communityId: parent.communityId,
        };

        // Reset parent division state + start cooldown
        parent.divisionReady = false;
        parent.divisionCooldownEnd = nowMs + DIVISION_COOLDOWN_MS;

        // Dispatch 'cell-division' CustomEvent
        window.dispatchEvent(new CustomEvent('cell-division', {
          detail: {
            parentId: parent.id,
            childId,
            species: parent.species,
            parentEnergy: parent.energy,
            childEnergy: halfEnergy,
            parentX: parent.x,
            parentY: parent.y,
            childX,
            childY,
          },
        }));
      }
    }

    // ── 3. Integrate velocities (semi-implicit Euler) ──
    for (const body of this.bodies.values()) {
      if (body.pinned || body.dragging) {
        // Clear accumulated impulses for static/kinematic bodies
        body.impulseX = 0;
        body.impulseY = 0;
        continue;
      }

      // Gravity (counteracted by buoyancy — lighter species float more)
      const gravityScale = 1 - body.buoyancy;
      body.vx += this.gravity.x * gravityScale * dt;
      body.vy += this.gravity.y * gravityScale * dt;

      // Accumulated impulses
      body.vx += body.impulseX;
      body.vy += body.impulseY;
      body.impulseX = 0;
      body.impulseY = 0;

      // Restore spring (optional — pulls cells back toward their ELK rest position)
      if (this.restoreStiffness > 0) {
        const rx = body.restX - body.x;
        const ry = body.restY - body.y;
        body.vx += rx * this.restoreStiffness * dt;
        body.vy += ry * this.restoreStiffness * dt;
      }

      // Damping (velocity-proportional drag, plus surface friction)
      const dampFactor = 1 - this.damping - body.friction * 0.01;
      const clampedDamp = Math.max(0.8, Math.min(1, dampFactor));
      body.vx *= clampedDamp;
      body.vy *= clampedDamp;

      // Species viscosity drag (medium thickness slows the cell over time)
      const visc = this.speciesLookup[body.species]?.viscosity ?? 0.3;
      body.vx *= (1 - visc * dt);
      body.vy *= (1 - visc * dt);

      // Integrate position
      body.x += body.vx * dt;
      body.y += body.vy * dt;
    }

    // ── 4. Boundary collision ──
    if (this.boundary) {
      const { x: bx, y: by, w: bw, h: bh } = this.boundary;
      const minX = bx;
      const minY = by;
      const maxX = bx + bw;
      const maxY = by + bh;

      for (const body of this.bodies.values()) {
        if (body.pinned || body.dragging) continue;
        const halfW = body.w / 2;
        const halfH = body.h / 2;

        // Left wall
        if (body.x - halfW < minX) {
          body.x = minX + halfW;
          body.vx = Math.abs(body.vx) * body.restitution;
        }
        // Right wall
        if (body.x + halfW > maxX) {
          body.x = maxX - halfW;
          body.vx = -Math.abs(body.vx) * body.restitution;
        }
        // Top wall
        if (body.y - halfH < minY) {
          body.y = minY + halfH;
          body.vy = Math.abs(body.vy) * body.restitution;
        }
        // Bottom wall
        if (body.y + halfH > maxY) {
          body.y = maxY - halfH;
          body.vy = -Math.abs(body.vy) * body.restitution;

          // Ground friction: reduce horizontal velocity on floor contact
          body.vx *= (1 - body.friction * 0.5);
        }
      }
    }

    // ── 5. Cell-cell overlap resolution + chemotaxis (M1294: spatial hash O(n)) ──
    // Push overlapping cells apart along their separation axis.
    // This prevents cells from stacking on top of each other during
    // gravity settling. Full collision response (with restitution) is
    // handled by the WebWorker physics (physics-bridge.ts); this is
    // just a lightweight overlap correction for interaction-layer stability.
    //
    // M1294: _collGrid (cell size 150 px) is used as a broad phase.
    // Only pairs whose grid cells overlap are tested for AABB collision.
    // Chemotaxis pre-pass also uses _collGrid to count same-species neighbours.

    // Pre-pass: count same-species neighbours within each cell's chemotaxis
    // range. Uses _collGrid for O(n) broad phase instead of O(n²).
    const neighborCount = new Map<string, number>();
    for (let i = 0; i < _bodyArr.length; i++) {
      const a = _bodyArr[i];
      const range = this.speciesLookup[a.species]?.chemotaxis_range ?? 0;
      if (range <= 0) { neighborCount.set(a.id, 0); continue; }
      const range2 = range * range;
      let count = 0;
      const chCandidates = this._collGrid.query(a.x, a.y);
      for (let ci = 0; ci < chCandidates.length; ci++) {
        const j = chCandidates[ci];
        if (j === i) continue;
        const b = _bodyArr[j];
        if (!b) continue; // M1284: skip stale grid index (unregistered child cell)
        if (b.species !== a.species) continue;
        if (dist2(a.x, a.y, b.x, b.y) < range2) count++;
      }
      neighborCount.set(a.id, count);
    }

    // Track processed pairs to avoid double-processing (i,j) and (j,i).
    // A simple visited Set keyed by min*n+max gives O(1) check.
    const _visitedPairs = new Set<number>();
    const _n = _bodyArr.length;

    for (let i = 0; i < _bodyArr.length; i++) {
      const a = _bodyArr[i];
      if (a.dragging) continue;

      // Spatial hash broad phase: query candidates from the 3×3 neighbourhood
      const candidates = this._collGrid.query(a.x, a.y);
      for (let ci = 0; ci < candidates.length; ci++) {
        const j = candidates[ci];
        if (j <= i) continue; // process each pair once (i < j)

        // Dedup guard (in case hash collisions return the same index twice)
        const pairKey = i * _n + j;
        if (_visitedPairs.has(pairKey)) continue;
        _visitedPairs.add(pairKey);

        const b = _bodyArr[j];
        if (!b) continue; // M1284: skip stale grid index (unregistered child cell)
        if (b.dragging) continue;
        if (a.pinned && b.pinned) continue;

        // Same-species chemotactic adhesion: cells of the same species are
        // attracted to one another within a configurable range. The strength
        // is modulated by preferred_neighbors so cells self-organize toward a
        // target connection degree: an under-connected cell pulls harder, an
        // over-connected cell flips to repulsion to shed excess neighbours.
        if (a.species === b.species) {
          const sp = this.speciesLookup[a.species];
          const range = sp?.chemotaxis_range ?? 0;
          const adhesion = sp?.adhesion ?? 0;
          if (range > 0 && adhesion > 0) {
            const d2 = dist2(a.x, a.y, b.x, b.y);
            if (d2 > 0 && d2 < range * range) {
              const dist = Math.sqrt(d2);

              // Neighbour-count balance: >0 when under-connected (attract),
              // <0 when over-connected (repel). Averaged over the pair so the
              // pairwise force stays symmetric. Clamped to [-1, 1].
              const pref = sp?.preferred_neighbors ?? 4;
              const balA = pref > 0 ? (pref - (neighborCount.get(a.id) ?? 0)) / pref : 0;
              const balB = pref > 0 ? (pref - (neighborCount.get(b.id) ?? 0)) / pref : 0;
              const balance = Math.max(-1, Math.min(1, (balA + balB) / 2));

              const force = adhesion * (1 - dist / range) * 10 * balance * dt;
              const nx = (b.x - a.x) / dist;
              const ny = (b.y - a.y) / dist;
              if (!a.pinned) {
                const invA = 1 / a.mass;
                a.vx += nx * force * invA;
                a.vy += ny * force * invA;
              }
              if (!b.pinned) {
                const invB = 1 / b.mass;
                b.vx -= nx * force * invB;
                b.vy -= ny * force * invB;
              }
            }
          }
        }

        // AABB overlap test
        const overlapX = (a.w / 2 + b.w / 2) - Math.abs(a.x - b.x);
        const overlapY = (a.h / 2 + b.h / 2) - Math.abs(a.y - b.y);

        if (overlapX > 0 && overlapY > 0) {
          // Resolve along minimum overlap axis
          const nx = a.x < b.x ? -1 : 1;
          const ny = a.y < b.y ? -1 : 1;

          // Quorum-sensing cluster factor: when both cells are in a quorum
          // they lower their mutual repulsion so they pack tighter. Use the
          // average of the pair's cluster factors (default 1 = full repulsion).
          const cf = ((a.clusterFactor ?? 1) + (b.clusterFactor ?? 1)) / 2;
          const sep = 0.5 * cf;

          // M1282: charge collision cost to both cells
          a.collisionCount++;
          b.collisionCount++;

          if (overlapX < overlapY) {
            // Separate along X
            const totalMass = (a.pinned ? 1e10 : a.mass) + (b.pinned ? 1e10 : b.mass);
            const ratioA = a.pinned ? 0 : (b.pinned ? 1 : b.mass / totalMass);
            const ratioB = b.pinned ? 0 : (a.pinned ? 1 : a.mass / totalMass);

            a.x += nx * overlapX * ratioA * -sep;
            b.x += nx * overlapX * ratioB * sep;

            // Bounce velocities
            const relVx = a.vx - b.vx;
            const restitution = Math.min(a.restitution, b.restitution);
            if (!a.pinned) a.vx -= relVx * (1 + restitution) * ratioA * sep;
            if (!b.pinned) b.vx += relVx * (1 + restitution) * ratioB * sep;
          } else {
            // Separate along Y
            const totalMass = (a.pinned ? 1e10 : a.mass) + (b.pinned ? 1e10 : b.mass);
            const ratioA = a.pinned ? 0 : (b.pinned ? 1 : b.mass / totalMass);
            const ratioB = b.pinned ? 0 : (a.pinned ? 1 : a.mass / totalMass);

            a.y += ny * overlapY * ratioA * -sep;
            b.y += ny * overlapY * ratioB * sep;

            const relVy = a.vy - b.vy;
            const restitution = Math.min(a.restitution, b.restitution);
            if (!a.pinned) a.vy -= relVy * (1 + restitution) * ratioA * sep;
            if (!b.pinned) b.vy += relVy * (1 + restitution) * ratioB * sep;
          }
        }
      }
    }
  }

  // ─── M1290: Community detection ─────────────────────────────────────────

  /**
   * _runCommunityDetection — Union-Find connected-component labelling.
   *
   * Two cells are in the same community when ALL of the following hold:
   *   1. Their Euclidean distance < interaction_radius (300 px by default).
   *   2. The Particle Life G coefficient matrix[species_i][species_j] > 0
   *      (they attract each other, i.e. they are genuinely linked).
   *
   * After labelling, assigns `body.communityId` for every body and stores the
   * result in `_communityMap`.  Dispatches a `community-update` CustomEvent
   * carrying `detail.communities: Map<cellId, communityId>`.
   */
  private _runCommunityDetection(): void {
    // M1294: Build a fresh spatial hash grid for community detection.
    // This method runs every 120 frames so building its own grid is fine;
    // it cannot reuse _plGrid because _physicsStep may not have run yet.
    const radius = PARTICLE_LIFE.interaction_radius;
    const r2 = radius * radius;
    const matrix = PARTICLE_LIFE.matrix;
    const arr = Array.from(this.bodies.values());
    const n = arr.length;

    // Build a temporary grid for this detection pass
    const cdGrid = new SpatialHashGrid(radius, Math.max(4096, n * 2));
    for (let i = 0; i < n; i++) {
      cdGrid.insert(i, arr[i].x, arr[i].y);
    }

    // Union-Find parent array (by index)
    const parent = new Int32Array(n);
    for (let i = 0; i < n; i++) parent[i] = i;

    function find(i: number): number {
      while (parent[i] !== i) {
        parent[i] = parent[parent[i]]; // path compression
        i = parent[i];
      }
      return i;
    }
    function union(i: number, j: number): void {
      const ri = find(i);
      const rj = find(j);
      if (ri !== rj) parent[ri] = rj;
    }

    // Build connectivity: link pairs that are close AND mutually attractive.
    // M1294: spatial hash broad phase — only check candidates in 3×3 neighbourhood.
    const visited = new Set<number>();
    for (let i = 0; i < n; i++) {
      const a = arr[i];
      const rowA = matrix[a.species];
      if (!rowA) continue;
      const candidates = cdGrid.query(a.x, a.y);
      for (let ci = 0; ci < candidates.length; ci++) {
        const j = candidates[ci];
        if (j <= i) continue; // process each pair once
        const pairKey = i * n + j;
        if (visited.has(pairKey)) continue;
        visited.add(pairKey);

        const b = arr[j];
        if (!b) continue; // M1284: skip stale grid index (unregistered child cell)
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        if (dx * dx + dy * dy >= r2) continue;
        // Check G > 0 in at least one direction (symmetric: use average)
        const Gab = rowA[b.species] ?? 0;
        const Gba = (matrix[b.species]?.[a.species]) ?? 0;
        if ((Gab + Gba) / 2 > 0) {
          union(i, j);
        }
      }
    }

    // Canonicalise root labels → compact community IDs starting at 0
    const rootToId = new Map<number, number>();
    let nextId = 0;
    const newMap = new Map<string, number>();

    for (let i = 0; i < n; i++) {
      const root = find(i);
      if (!rootToId.has(root)) rootToId.set(root, nextId++);
      const cid = rootToId.get(root)!;
      arr[i].communityId = cid;
      newMap.set(arr[i].id, cid);
    }

    this._communityMap = newMap;

    // Dispatch community-update event
    window.dispatchEvent(new CustomEvent('community-update', {
      detail: { communities: new Map(newMap) },
    }));
  }

  /**
   * _applyCommunityForces — per-frame micro-forces that make communities
   * cohere and separate.
   *
   * • Same community  → weak attraction (gravity): pulls cells together so the
   *   cluster stays compact.  Magnitude: COMMUNITY_INTRA_G / distance.
   * • Different community → weak repulsion: pushes communities apart so they
   *   form visually distinct groups.  Magnitude: COMMUNITY_INTER_G / distance.
   *
   * Forces are intentionally very small (1-2 orders below Particle Life G) so
   * they guide structure without overwhelming the Particle Life dynamics.
   */
  private _applyCommunityForces(): void {
    // M1294: Use _plGrid for O(n) broad phase (same radius as Particle Life).
    // _plGrid may not be populated (called from step() outside _physicsStep),
    // so we build a lightweight local grid if needed.
    const COMMUNITY_INTRA_G =  0.5;  // attraction within same community
    const COMMUNITY_INTER_G = -0.3;  // repulsion between different communities
    const radius = PARTICLE_LIFE.interaction_radius;
    const r2 = radius * radius;

    const arr = Array.from(this.bodies.values());
    const dt = this.fixedStep; // use fixed step as force scale

    // Use _plGrid (populated in _physicsStep). If it hasn't been populated yet
    // (e.g. first frame), fall back to building a local grid.
    let grid = this._plGrid;
    // Quick freshness check: if the grid has no entries, rebuild locally.
    // (SpatialHashGrid doesn't expose a size but we can test by querying
    //  a known body position — an empty grid returns [].)
    const cfVisited = new Set<number>();
    const cfN = arr.length;

    for (let i = 0; i < arr.length; i++) {
      const a = arr[i];
      if (a.pinned || a.dragging) continue;

      const candidates = grid.query(a.x, a.y);
      for (let ci = 0; ci < candidates.length; ci++) {
        const j = candidates[ci];
        if (j <= i) continue;
        const pairKey = i * cfN + j;
        if (cfVisited.has(pairKey)) continue;
        cfVisited.add(pairKey);

        const b = arr[j];
        if (!b || b.pinned || b.dragging) continue;

        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d2 = dx * dx + dy * dy;
        if (d2 >= r2 || d2 === 0) continue;

        const dist = Math.sqrt(d2);
        const G = a.communityId === b.communityId
          ? COMMUNITY_INTRA_G
          : COMMUNITY_INTER_G;

        const force = G / Math.max(dist, 10); // prevent divide-by-zero
        const fx = dx * force * dt * 0.5;
        const fy = dy * force * dt * 0.5;

        a.vx += fx;
        a.vy += fy;
        b.vx -= fx;
        b.vy -= fy;
      }
    }
  }

  /** M1290: Return the community ID assigned to a cell. -1 if not found. */
  getCommunityId(cellId: string): number {
    return this._communityMap.get(cellId) ?? -1;
  }

  // ─── M1293: Species color evolution ─────────────────────────────────────

  /**
   * _runSpeciesColorEvolution — epigenetic "color infection".
   *
   * For every cell, we look at all neighbours within signal_radius (200 px).
   * We tally the species of those neighbours.  If one species constitutes > 50%
   * of the neighbourhood we call it the dominant species.  The cell's glow color
   * is then lerped 5% toward that species's canonical color:
   *
   *   newColor = lerp(myColor, dominantColor, 0.05)
   *
   * A 'species-color-evolve' CustomEvent is dispatched for each cell that
   * changes, carrying { cellId, newGlowColor: [r, g, b] }.  The render loop
   * (gpu-render-loop.ts) listens for this event and patches the CellData so
   * the PBR pass picks it up next frame.
   *
   * Signal radius is taken from cell_lifecycle.json signaling.signal_radius
   * (200 px default).  Cells with no neighbours are left unchanged.
   */
  private _runSpeciesColorEvolution(): void {
    const SIGNAL_RADIUS = 200;   // px — must match signaling.signal_radius
    const LERP_ALPHA    = 0.05;  // per-tick drift: 5% toward dominant color

    const r2 = SIGNAL_RADIUS * SIGNAL_RADIUS;
    const arr = Array.from(this.bodies.values());

    for (const body of arr) {
      // Count neighbour species within signal_radius
      const speciesCounts: Record<string, number> = {};
      let total = 0;

      for (const other of arr) {
        if (other.id === body.id) continue;
        const dx = body.x - other.x;
        const dy = body.y - other.y;
        if (dx * dx + dy * dy <= r2) {
          speciesCounts[other.species] = (speciesCounts[other.species] ?? 0) + 1;
          total++;
        }
      }

      if (total === 0) continue; // isolated cell — no pressure

      // Find dominant neighbour species (must exceed 50% of neighbours)
      let dominantSpecies: string | null = null;
      let dominantCount = 0;
      for (const [sp, count] of Object.entries(speciesCounts)) {
        if (count > dominantCount) {
          dominantCount = count;
          dominantSpecies = sp;
        }
      }

      // Only drift when one species holds clear majority
      if (dominantSpecies === null || dominantCount / total <= 0.5) continue;

      // Skip drift if dominant species is the same as this cell's own species
      // (the cell is already "at home" — no epigenetic pressure)
      if (dominantSpecies === body.species) continue;

      const targetColor = SPECIES_DEFAULT_GLOW[dominantSpecies] ?? [0.5, 0.5, 0.5];
      const current = this._cellGlowColor.get(body.id)
        ?? (SPECIES_DEFAULT_GLOW[body.species] ?? [0.5, 0.5, 0.5]);

      // Linear interpolation: color = lerp(myColor, neighborDominantColor, 0.05)
      const newColor: [number, number, number] = [
        current[0] + (targetColor[0] - current[0]) * LERP_ALPHA,
        current[1] + (targetColor[1] - current[1]) * LERP_ALPHA,
        current[2] + (targetColor[2] - current[2]) * LERP_ALPHA,
      ];

      this._cellGlowColor.set(body.id, newColor);

      // Dispatch event so the render layer can react immediately
      window.dispatchEvent(new CustomEvent('species-color-evolve', {
        detail: { cellId: body.id, newGlowColor: newColor },
      }));
    }
  }

  /**
   * M1293: Return the current evolved glow color for a cell.
   * Falls back to the species default if the cell has no entry yet.
   */
  getGlowColor(cellId: string): [number, number, number] {
    const body = this.bodies.get(cellId);
    if (!body) return [0.5, 0.5, 0.5];
    return (
      this._cellGlowColor.get(cellId) ??
      SPECIES_DEFAULT_GLOW[body.species] ??
      [0.5, 0.5, 0.5]
    );
  }

  // ─── Queries ────────────────────────────────────────────────────────────

  /** Get the current interaction state for a cell. */
  getState(cellId: string): CellInteractionState | null {
    const body = this.bodies.get(cellId);
    if (!body) return null;
    return {
      cell_id: body.id,
      x: body.x,
      y: body.y,
      vx: body.vx,
      vy: body.vy,
      pinned: body.pinned,
      dragging: body.dragging,
      energy: body.energy,
      divisionReady: body.divisionReady,
    };
  }

  /** Get all cell states as an array. */
  getAllStates(): CellInteractionState[] {
    const states: CellInteractionState[] = [];
    for (const body of this.bodies.values()) {
      states.push({
        cell_id: body.id,
        x: body.x,
        y: body.y,
        vx: body.vx,
        vy: body.vy,
        pinned: body.pinned,
        dragging: body.dragging,
        energy: body.energy,
        divisionReady: body.divisionReady,
      });
    }
    return states;
  }

  /**
   * M1282: Compute the visual opacity for a cell based on its energy level.
   * Returns 1.0 normally; linearly fades toward 0 when energy drops below
   * ENERGY_APOPTOSIS_THRESHOLD (apoptosis phase). Use this value as
   * CellPBRDescriptor.opacity when building the render descriptor.
   */
  getEnergyOpacity(cellId: string): number {
    const body = this.bodies.get(cellId);
    if (!body) return 1.0;
    if (body.energy >= ENERGY_APOPTOSIS_THRESHOLD) return 1.0;
    return body.energy / ENERGY_APOPTOSIS_THRESHOLD;
  }

  /** M1282: Return the raw energy value [0, 1] for a cell. */
  getEnergy(cellId: string): number {
    return this.bodies.get(cellId)?.energy ?? 1.0;
  }

  /**
   * M1283: Return whether a cell has quorum active (≥ quorum_threshold
   * same-species neighbours within signal_radius). When active, the render
   * layer should apply the synchronised flicker opacity:
   *   opacity = 0.7 + 0.3 * Math.sin(time * 3)
   */
  isQuorumActive(cellId: string): boolean {
    return this._quorumActive.get(cellId) ?? false;
  }

  /**
   * M1283: Compute the quorum-sensing visual opacity for a cell.
   * When quorum is inactive returns 1.0.
   * When active, returns a pulsing value in [0.7, 1.0] driven by time (seconds).
   * @param cellId  The cell to query.
   * @param timeSec Current time in seconds (e.g. performance.now() / 1000).
   */
  getQuorumOpacity(cellId: string, timeSec: number): number {
    if (!this._quorumActive.get(cellId)) return 1.0;
    return 0.7 + 0.3 * Math.sin(timeSec * 3);
  }

  /** Get the current bbox (derived from body position + dimensions). */
  getBBox(cellId: string): { x: number; y: number; w: number; h: number } | null {
    const body = this.bodies.get(cellId);
    if (!body) return null;
    return {
      x: body.x - body.w / 2,
      y: body.y - body.h / 2,
      w: body.w,
      h: body.h,
    };
  }

  /** Hit-test: return the topmost cell (highest z) at (px, py) in cell-space. */
  hitTest(px: number, py: number): string | null {
    let bestId: string | null = null;
    let bestZ = -Infinity;

    for (const body of this.bodies.values()) {
      const halfW = body.w / 2;
      const halfH = body.h / 2;
      if (
        px >= body.x - halfW &&
        px <= body.x + halfW &&
        py >= body.y - halfH &&
        py <= body.y + halfH &&
        body.z > bestZ
      ) {
        bestZ = body.z;
        bestId = body.id;
      }
    }

    return bestId;
  }

  /** Check whether any cell is currently in motion (|v| > threshold). */
  isSettled(threshold = 0.5): boolean {
    for (const body of this.bodies.values()) {
      if (body.dragging) return false;
      if (!body.pinned) {
        const speed = Math.sqrt(body.vx * body.vx + body.vy * body.vy);
        if (speed > threshold) return false;
      }
    }
    return true;
  }

  /** Return total kinetic energy (useful for convergence monitoring). */
  kineticEnergy(): number {
    let ke = 0;
    for (const body of this.bodies.values()) {
      if (body.pinned || body.dragging) continue;
      ke += 0.5 * body.mass * (body.vx * body.vx + body.vy * body.vy);
    }
    return ke;
  }

  // ─── Serialization ──────────────────────────────────────────────────────

  /**
   * Export the current interaction state as a JSON-serializable object
   * suitable for POST to /api/cell/publish or writing to force_field.json.
   */
  toForceField(): Record<string, { dx: number; dy: number; dz: number }> {
    const field: Record<string, { dx: number; dy: number; dz: number }> = {};
    for (const body of this.bodies.values()) {
      field[body.id] = {
        dx: body.vx * this.fixedStep,
        dy: body.vy * this.fixedStep,
        dz: 0,
      };
    }
    return field;
  }

  /**
   * Export pin states as a map — useful for persisting user pin choices.
   */
  toPinMap(): Record<string, boolean> {
    const pins: Record<string, boolean> = {};
    for (const body of this.bodies.values()) {
      pins[body.id] = body.pinned;
    }
    return pins;
  }

  /**
   * Restore pin states from a previously exported map.
   */
  fromPinMap(pins: Record<string, boolean>): void {
    for (const [id, pinned] of Object.entries(pins)) {
      const body = this.bodies.get(id);
      if (body) {
        body.pinned = pinned;
        if (pinned) {
          body.vx = 0;
          body.vy = 0;
        }
      }
    }
  }

  // ─── Lifecycle ──────────────────────────────────────────────────────────

  /** Reset all bodies to their rest positions with zero velocity. */
  reset(): void {
    this.dragCancel();
    this.blastQueue.length = 0;
    this.accumulator = 0;
    this.lastTime = 0;

    this.signalParticles.length = 0;
    this._quorumActive.clear();
    this._signalFrameCount = 0;
    this._apoptosisOrigSize.clear();
    this._communityMap.clear();
    this._communityFrameCount = 0;
    this._colorEvolveFrameCount = 0;

    for (const body of this.bodies.values()) {
      body.x = body.restX;
      body.y = body.restY;
      body.vx = 0;
      body.vy = 0;
      body.impulseX = 0;
      body.impulseY = 0;
      body.dragging = false;
      body.energy = ENERGY_INITIAL;
      body.divisionReady = false;
      body.collisionCount = 0;
      body.divisionCooldownEnd = 0;
      body.apoptosisActive = false;
      body.apoptosisStartTime = 0;
      // Preserve pin state — reset doesn't unpin
    }
  }

  /** Dispose of all resources. The instance cannot be reused after this. */
  dispose(): void {
    this.disposed = true;
    this.dragCancel();
    this.bodies.clear();
    this.blastQueue.length = 0;
    this.signalParticles.length = 0;
    this._quorumActive.clear();
    this._apoptosisOrigSize.clear();
    this._communityMap.clear();
    this._cellGlowColor.clear();
  }
}

// ─── Factory ────────────────────────────────────────────────────────────────

/**
 * createCellInteractionPhysics — convenience factory that accepts the same
 * cell format as CellInteraction.ts and wires up sane defaults.
 */
export function createCellInteractionPhysics(
  cells: InteractionCell[],
  opts?: CellInteractionPhysicsOptions,
): CellInteractionPhysics {
  return new CellInteractionPhysics(cells, opts);
}

// ─── DOM mount helper ───────────────────────────────────────────────────────

export interface MountInteractionPhysicsOptions extends CellInteractionPhysicsOptions {
  /** The canvas element to listen for pointer events on. */
  canvas: HTMLCanvasElement;
  /** Coordinate transform: canvas-space → cell-space. */
  transform: { scale: number; offX: number; offY: number };
  /** Cells array. */
  cells: InteractionCell[];
}

export interface MountInteractionPhysicsHandle {
  physics: CellInteractionPhysics;
  destroy: () => void;
}

/**
 * mountInteractionPhysics — attaches pointer event listeners to a canvas
 * and wires them to a CellInteractionPhysics instance.
 *
 * Pointer events:
 *   - pointerdown                → dragStart on hit cell
 *   - pointermove (while down)   → dragMove
 *   - pointerup                  → dragEnd (throw)
 *   - dblclick                   → togglePin on hit cell
 *   - contextmenu                → togglePin on hit cell (right-click)
 *   - shift+pointerdown (no hit) → inject at pointer position
 *
 * Returns a handle with the physics instance and a destroy() cleanup function.
 */
export function mountInteractionPhysics(
  opts: MountInteractionPhysicsOptions,
): MountInteractionPhysicsHandle {
  const { canvas, transform: t, cells, ...physicsOpts } = opts;

  const physics = new CellInteractionPhysics(cells, physicsOpts);
  let animFrameId: number = -1;
  let isPointerDown = false;

  // ── Coordinate conversion ──
  function toCell(clientX: number, clientY: number): { x: number; y: number } {
    const rect = canvas.getBoundingClientRect();
    const canvasX = (clientX - rect.left) * (canvas.width / rect.width);
    const canvasY = (clientY - rect.top) * (canvas.height / rect.height);
    return {
      x: (canvasX - t.offX) / t.scale,
      y: (canvasY - t.offY) / t.scale,
    };
  }

  // ── Pointer handlers ──
  function onPointerDown(e: PointerEvent) {
    const { x, y } = toCell(e.clientX, e.clientY);

    if (e.shiftKey) {
      // Shift+click → inject
      physics.inject(x, y);
      return;
    }

    const hitId = physics.hitTest(x, y);
    if (hitId) {
      isPointerDown = true;
      canvas.setPointerCapture(e.pointerId);
      physics.dragStart(hitId, x, y);
    }
  }

  function onPointerMove(e: PointerEvent) {
    if (!isPointerDown) return;
    const { x, y } = toCell(e.clientX, e.clientY);
    physics.dragMove(x, y);
  }

  function onPointerUp(e: PointerEvent) {
    if (!isPointerDown) return;
    isPointerDown = false;
    canvas.releasePointerCapture(e.pointerId);
    physics.dragEnd();
  }

  function onDblClick(e: MouseEvent) {
    const { x, y } = toCell(e.clientX, e.clientY);
    const hitId = physics.hitTest(x, y);
    if (hitId) {
      e.preventDefault();
      physics.togglePin(hitId);
    }
  }

  function onContextMenu(e: MouseEvent) {
    const { x, y } = toCell(e.clientX, e.clientY);
    const hitId = physics.hitTest(x, y);
    if (hitId) {
      e.preventDefault();
      physics.togglePin(hitId);
    }
  }

  // ── Attach listeners ──
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);
  canvas.addEventListener('dblclick', onDblClick);
  canvas.addEventListener('contextmenu', onContextMenu);

  // ── Animation loop ──
  function tick(now: number) {
    physics.step(now);
    animFrameId = requestAnimationFrame(tick);
  }
  animFrameId = requestAnimationFrame(tick);

  return {
    physics,
    destroy() {
      cancelAnimationFrame(animFrameId);
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerUp);
      canvas.removeEventListener('dblclick', onDblClick);
      canvas.removeEventListener('contextmenu', onContextMenu);
      physics.dispose();
    },
  };
}
