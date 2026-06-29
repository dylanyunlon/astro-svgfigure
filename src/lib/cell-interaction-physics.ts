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
// M1279: ambient medium physics loaded from channels/physics/environment.json.
// Drives brownian motion, a global laminar flow field, sedimentation gravity,
// soft-wall boundary repulsion, and a temperature gradient that locally lowers
// viscosity (cells nearer the warm centre move more freely).
import environmentParams from '../../channels/physics/environment.json';

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

const ENVIRONMENT = environmentParams as EnvironmentParams;

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

/** Snapshot of a cell's interaction state — emitted via events. */
export interface CellInteractionState {
  cell_id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  pinned: boolean;
  dragging: boolean;
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

const DEFAULT_SPECIES_PHYSICS: SpeciesPhysics = {
  mass: 75,
  friction: 0.5,
  restitution: 0.3,
  buoyancy: 0.5,
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

  // ── Velocity history for smooth throw (ring buffer, last 4 frames) ──
  private readonly velHistory: Array<{ vx: number; vy: number; dt: number }> = [];
  private readonly VEL_HISTORY_LEN = 4;

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
      };

      this.bodies.set(cell.cell_id, body);
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
    });
  }

  /** Remove a cell body. If it's currently dragged, the drag is cancelled. */
  removeCell(cellId: string): void {
    if (this.draggedId === cellId) this.dragCancel();
    this.bodies.delete(cellId);
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

    // ── 2b. Particle Life pairwise interaction ──
    // For every ordered pair (i, j) within the interaction radius, apply a force
    // proportional to G / distance, where G = matrix[species_i][species_j].
    // G > 0 attracts i toward j, G < 0 repels.  From this trivially simple rule
    // emerges complex self-organising structure (Particle Life), here shaped by
    // the Transformer information-flow matrix in species_interaction_matrix.json.
    {
      const radius = PARTICLE_LIFE.interaction_radius;
      const r2 = radius * radius;
      const matrix = PARTICLE_LIFE.matrix;
      const plBodies = Array.from(this.bodies.values());

      for (let i = 0; i < plBodies.length; i++) {
        const a = plBodies[i];
        if (a.pinned || a.dragging) continue;
        const rowA = matrix[a.species];
        if (!rowA) continue;

        for (let j = 0; j < plBodies.length; j++) {
          if (i === j) continue;
          const b = plBodies[j];

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

    // ── 5. Simple cell-cell overlap resolution ──
    // Push overlapping cells apart along their separation axis.
    // This prevents cells from stacking on top of each other during
    // gravity settling. Full collision response (with restitution) is
    // handled by the WebWorker physics (physics-bridge.ts); this is
    // just a lightweight overlap correction for interaction-layer stability.
    const bodyArr = Array.from(this.bodies.values());

    // Pre-pass: count same-species neighbours within each cell's chemotaxis
    // range. The count is used below to modulate adhesion so the population
    // self-organizes toward each species' preferred connection degree.
    const neighborCount = new Map<string, number>();
    for (let i = 0; i < bodyArr.length; i++) {
      const a = bodyArr[i];
      const range = this.speciesLookup[a.species]?.chemotaxis_range ?? 0;
      if (range <= 0) { neighborCount.set(a.id, 0); continue; }
      const range2 = range * range;
      let count = 0;
      for (let j = 0; j < bodyArr.length; j++) {
        if (i === j) continue;
        const b = bodyArr[j];
        if (b.species !== a.species) continue;
        if (dist2(a.x, a.y, b.x, b.y) < range2) count++;
      }
      neighborCount.set(a.id, count);
    }

    for (let i = 0; i < bodyArr.length; i++) {
      const a = bodyArr[i];
      if (a.dragging) continue;

      for (let j = i + 1; j < bodyArr.length; j++) {
        const b = bodyArr[j];
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
      });
    }
    return states;
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

    for (const body of this.bodies.values()) {
      body.x = body.restX;
      body.y = body.restY;
      body.vx = 0;
      body.vy = 0;
      body.impulseX = 0;
      body.impulseY = 0;
      body.dragging = false;
      // Preserve pin state — reset doesn't unpin
    }
  }

  /** Dispose of all resources. The instance cannot be reused after this. */
  dispose(): void {
    this.disposed = true;
    this.dragCancel();
    this.bodies.clear();
    this.blastQueue.length = 0;
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
