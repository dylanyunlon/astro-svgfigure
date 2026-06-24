/**
 * adaptive-lod.ts — M779: Adaptive LOD System
 * ─────────────────────────────────────────────────────────────────────────────
 * Dynamically adjusts rendering fidelity per cell and per particle system
 * based on three orthogonal signals:
 *
 *   1. **Distance**       — camera-to-cell distance in world units.
 *   2. **Screen coverage** — fraction of the viewport occupied by a cell's
 *      bounding box (screen-space projected area / viewport area).
 *   3. **FPS budget**      — real-time frame budget from PerformanceBudget
 *      to enforce a 60 fps floor.
 *
 * LOD tiers
 * ─────────────────────────────────────────────────────────────────────────────
 *   LOD 0  (FULL)      Near / large on screen / budget headroom
 *          → Full SDF silhouette + PBR lighting + reaction-diffusion texture
 *          → Full particle count with per-particle physics
 *
 *   LOD 1  (REDUCED)   Mid-distance / moderate screen coverage
 *          → Simplified SDF (lower iteration) + flat shading (no PBR specular)
 *          → Reaction-diffusion replaced by procedural noise approximation
 *          → Particle count reduced to 60 % of nominal
 *
 *   LOD 2  (BILLBOARD) Far / small on screen / budget pressure
 *          → Billboard quad with pre-baked colour + Gaussian soft edge
 *          → No SDF evaluation, no PBR, no RD texture
 *          → Particle count reduced to 20 % (ambient swirl only)
 *
 *   LOD 3  (DOT)       Very far / tiny / severe budget pressure
 *          → Single tinted point sprite (1–4 px)
 *          → Zero particles, zero per-cell shader cost
 *
 * Particle density auto-scaling
 * ─────────────────────────────────────────────────────────────────────────────
 * Independent of per-cell LOD, the total particle budget is governed by
 * viewport density: `effectiveParticles = baseCount × densityScale`, where
 * `densityScale` is the ratio of visible cells to total cells, modulated by
 * the FPS headroom factor.  This avoids GPU overdraw for off-screen emitters
 * while keeping visible fluid rich.
 *
 * Integration points
 * ─────────────────────────────────────────────────────────────────────────────
 *   • PerformanceBudget    — supplies the current tier + FPS; LOD reacts to
 *     tier changes via `onTierChange` and adjusts distance thresholds.
 *   • WorldOrchestrator    — calls `lod.update(camera, cells, dt)` each frame
 *     and reads back per-cell LOD assignments + global particle budget.
 *   • InstancedCellRenderer — reads `CellLODAssignment.renderHint` to select
 *     the shader path (full, reduced, billboard, dot).
 *   • Emitter strategy     — multiplies emission rate by the LOD particle
 *     ratio returned from `getParticleBudgetRatio()`.
 *
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/performance-budget.ts       — Tier / TierConfig / PerformanceBudget
 *   src/lib/sph/world-orchestrator.ts       — Master rAF loop, adaptive tuning
 *   src/lib/sph/instanced-cell-renderer.ts  — CellInstanceDescriptor, CellBBox
 *   src/lib/sph/species-shader-registry.ts  — SdfShape, MaterialType, PatternShader
 *   src/lib/sph/cell-material-system.ts     — PBR + Fresnel + Matcap pipeline
 *
 * Research: xiaodi #M779 — cell-pubsub-loop
 */



// [orphan-import] import {
import type { CellBBox } from './instanced-cell-renderer';
import type { MaterialType, PatternShader } from './species-shader-registry';

// [empty-import] import {
// [empty-import] } from './performance-budget';

// [orphan-precise]   PerformanceBudget,
// [orphan-precise]   getGlobalBudget,
// [orphan] type Tier,
// [orphan] type TierConfig,


// ─────────────────────────────────────────────────────────────────────────────
// LOD Level enum & render hints
// ─────────────────────────────────────────────────────────────────────────────

/** Discrete LOD levels ordered from highest to lowest fidelity. */
export const enum LODLevel {
  /** Full SDF + PBR + reaction-diffusion texture + full particles. */
  FULL      = 0,
  /** Simplified SDF + flat shading + procedural noise + 60 % particles. */
  REDUCED   = 1,
  /** Billboard quad + baked colour + 20 % particles. */
  BILLBOARD = 2,
  /** Single point sprite, zero particles. */
  DOT       = 3,
}

/**
 * Render hint consumed by the instanced cell renderer to select the
 * fragment shader branch for this cell.
 */
export interface CellRenderHint {
  /** Material override for this LOD level. */
  materialType: MaterialType | 'billboard' | 'dot';
  /** Pattern shader override ('none' for LOD ≥ 1). */
  patternShader: PatternShader;
  /** Whether to evaluate the SDF (false → simple quad). */
  useSDF: boolean;
  /** Whether to enable PBR specular + Fresnel. */
  usePBR: boolean;
  /** Whether to sample the reaction-diffusion texture. */
  useRDTexture: boolean;
  /** SDF iteration reduction factor (1.0 = full, 0.5 = halved). */
  sdfIterationScale: number;
  /** Particle count multiplier relative to the cell's nominal count. */
  particleRatio: number;
  /** Billboard opacity modulator (1.0 for LOD 0–1, fade for LOD 2–3). */
  billboardOpacity: number;
  /** Point sprite size in pixels (only used at LOD 3). */
  dotSizePx: number;
}

/**
 * Per-cell LOD assignment produced each frame by `AdaptiveLOD.update()`.
 */
export interface CellLODAssignment {
  /** The cell identifier. */
  cellId: string;
  /** Assigned LOD level. */
  level: LODLevel;
  /** Camera-to-cell distance in world units. */
  distance: number;
  /** Screen-space coverage ratio (0–1). */
  screenCoverage: number;
  /** Complete render hint for the renderer. */
  renderHint: CellRenderHint;
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera abstraction
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Minimal camera interface.  The adaptive LOD system only needs the camera's
 * world-space position and the viewport dimensions to compute distance and
 * screen-space coverage.
 */
export interface LODCamera {
  /** Camera world-space X. */
  x: number;
  /** Camera world-space Y. */
  y: number;
  /** Camera zoom level (1 = default; > 1 = zoomed in). */
  zoom: number;
  /** Viewport width in CSS pixels. */
  viewportW: number;
  /** Viewport height in CSS pixels. */
  viewportH: number;
}

/**
 * Cell descriptor — the minimum information the LOD system needs per cell.
 */
export interface LODCellInput {
  cellId: string;
  bbox: CellBBox;
  /** The cell's original material type (used at LOD 0). */
  materialType: MaterialType;
  /** The cell's original pattern shader (used at LOD 0). */
  patternShader: PatternShader;
  /** Base colour for billboard / dot fallback [r, g, b, a] linear 0–1. */
  baseColor: [number, number, number, number];
  /** Nominal particle count this cell would emit at full LOD. */
  nominalParticleCount: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Distance thresholds per performance tier
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Distance-based LOD transition thresholds in world units.
 * When the performance budget tier drops, the thresholds tighten so that
 * more cells fall to lower LOD levels earlier.
 */
interface LODDistanceThresholds {
  /** Distance beyond which LOD drops from FULL to REDUCED. */
  fullToReduced: number;
  /** Distance beyond which LOD drops from REDUCED to BILLBOARD. */
  reducedToBillboard: number;
  /** Distance beyond which LOD drops from BILLBOARD to DOT. */
  billboardToDot: number;
}

const TIER_DISTANCE_THRESHOLDS: Record<Tier, LODDistanceThresholds> = {
  ULTRA:  { fullToReduced: 5.0,  reducedToBillboard: 12.0, billboardToDot: 25.0 },
  HIGH:   { fullToReduced: 3.5,  reducedToBillboard:  8.0, billboardToDot: 18.0 },
  MEDIUM: { fullToReduced: 2.0,  reducedToBillboard:  5.0, billboardToDot: 12.0 },
  LOW:    { fullToReduced: 1.2,  reducedToBillboard:  3.0, billboardToDot:  7.0 },
};

// ─────────────────────────────────────────────────────────────────────────────
// Screen-coverage thresholds
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Minimum screen-space coverage ratio to remain at a given LOD.
 * If a cell covers less than the threshold, it drops one level.
 */
interface LODScreenThresholds {
  /** Below this coverage → cannot be FULL. */
  minForFull: number;
  /** Below this coverage → cannot be REDUCED. */
  minForReduced: number;
  /** Below this coverage → drop to DOT. */
  minForBillboard: number;
}

const SCREEN_THRESHOLDS: LODScreenThresholds = {
  minForFull:      0.005,   // 0.5 % of viewport
  minForReduced:   0.001,   // 0.1 %
  minForBillboard: 0.0002,  // 0.02 %
};

// ─────────────────────────────────────────────────────────────────────────────
// FPS headroom factor
// ─────────────────────────────────────────────────────────────────────────────

/** Target FPS for the simulation. */
const TARGET_FPS = 60;

/** If smoothed FPS falls below this, the LOD system enters aggressive mode. */
const FPS_AGGRESSIVE_THRESHOLD = 48;

/** If smoothed FPS falls below this, emergency cull to DOT for distant cells. */
const FPS_EMERGENCY_THRESHOLD = 35;

/** Hysteresis frames before upgrading a cell's LOD after FPS recovers. */
const LOD_UPGRADE_COOLDOWN_FRAMES = 30;

// ─────────────────────────────────────────────────────────────────────────────
// Particle budget constants
// ─────────────────────────────────────────────────────────────────────────────

/** Particle ratio per LOD level. */
const LOD_PARTICLE_RATIOS: Record<LODLevel, number> = {
  [LODLevel.FULL]:      1.0,
  [LODLevel.REDUCED]:   0.6,
  [LODLevel.BILLBOARD]: 0.2,
  [LODLevel.DOT]:       0.0,
};

/** Global particle budget scaling per performance tier. */
const TIER_PARTICLE_BUDGET_SCALE: Record<Tier, number> = {
  ULTRA:  1.0,
  HIGH:   0.8,
  MEDIUM: 0.5,
  LOW:    0.25,
};

// ─────────────────────────────────────────────────────────────────────────────
// Render hint presets
// ─────────────────────────────────────────────────────────────────────────────

function makeFullHint(mat: MaterialType, pat: PatternShader): CellRenderHint {
  return {
    materialType:       mat,
    patternShader:      pat,
    useSDF:             true,
    usePBR:             true,
    useRDTexture:       true,
    sdfIterationScale:  1.0,
    particleRatio:      LOD_PARTICLE_RATIOS[LODLevel.FULL],
    billboardOpacity:   1.0,
    dotSizePx:          0,
  };
}

function makeReducedHint(mat: MaterialType): CellRenderHint {
  return {
    materialType:       mat,
    patternShader:      'none',
    useSDF:             true,
    usePBR:             false,
    useRDTexture:       false,
    sdfIterationScale:  0.5,
    particleRatio:      LOD_PARTICLE_RATIOS[LODLevel.REDUCED],
    billboardOpacity:   1.0,
    dotSizePx:          0,
  };
}

function makeBillboardHint(): CellRenderHint {
  return {
    materialType:       'billboard',
    patternShader:      'none',
    useSDF:             false,
    usePBR:             false,
    useRDTexture:       false,
    sdfIterationScale:  0,
    particleRatio:      LOD_PARTICLE_RATIOS[LODLevel.BILLBOARD],
    billboardOpacity:   1.0,
    dotSizePx:          0,
  };
}

function makeDotHint(sizePx: number): CellRenderHint {
  return {
    materialType:       'dot',
    patternShader:      'none',
    useSDF:             false,
    usePBR:             false,
    useRDTexture:       false,
    sdfIterationScale:  0,
    particleRatio:      LOD_PARTICLE_RATIOS[LODLevel.DOT],
    billboardOpacity:   0.6,
    dotSizePx:          sizePx,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// LOD transition smoothing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Per-cell LOD transition state to prevent popping.
 * Tracks the current fractional LOD and smoothly interpolates toward the
 * target discrete LOD over several frames.
 */
interface CellLODState {
  /** Current discrete LOD. */
  currentLevel: LODLevel;
  /** Target discrete LOD computed this frame. */
  targetLevel: LODLevel;
  /** Fractional LOD for smooth transitions (0.0 = FULL, 3.0 = DOT). */
  fractionalLOD: number;
  /** Frames remaining in the upgrade cooldown (prevents oscillation). */
  upgradeCooldown: number;
  /** Last frame number this cell was visible (for aging out stale entries). */
  lastSeenFrame: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaptiveLOD class
// ─────────────────────────────────────────────────────────────────────────────

export interface AdaptiveLODConfig {
  /** Custom distance thresholds (overrides tier-based defaults). */
  distanceOverrides?: Partial<LODDistanceThresholds>;
  /** Custom screen-coverage thresholds. */
  screenOverrides?: Partial<LODScreenThresholds>;
  /** LOD transition smoothing rate (0 = instant snap, 1 = very slow). */
  transitionSmoothing?: number;
  /** Enable viewport-density-based particle auto-scaling. */
  enableParticleDensityScaling?: boolean;
  /** Maximum total particles across all cells (absolute cap). */
  globalParticleCap?: number;
}

export class AdaptiveLOD {
  // ── Performance budget integration ──────────────────────────────────────
  private budget: PerformanceBudget;
  private currentTier: Tier;
  private distThresholds: LODDistanceThresholds;
  private screenThresholds: LODScreenThresholds;

  // ── Per-cell LOD state ──────────────────────────────────────────────────
  private cellStates: Map<string, CellLODState> = new Map();

  // ── Global particle budget ──────────────────────────────────────────────
  private globalParticleBudgetRatio = 1.0;
  private visibleCellCount = 0;
  private totalCellCount   = 0;

  // ── FPS tracking (independent short window for LOD decisions) ───────────
  private fpsRing: Float64Array;
  private fpsRingIdx   = 0;
  private fpsRingFilled = false;
  private smoothedFps   = 60;

  // ── Frame counter ──────────────────────────────────────────────────────
  private frameNumber = 0;

  // ── Config ─────────────────────────────────────────────────────────────
  private transitionRate: number;
  private enableDensityScaling: boolean;
  private globalParticleCap: number;

  // ── Cached assignments (avoids re-allocation each frame) ────────────────
  private assignments: CellLODAssignment[] = [];

  // ── Tier change unsubscribe handle ─────────────────────────────────────
  private unsubTier: (() => void) | null = null;

  // ── Stats ──────────────────────────────────────────────────────────────
  private lodCounts: [number, number, number, number] = [0, 0, 0, 0];

  // ═══════════════════════════════════════════════════════════════════════
  //  Constructor
  // ═══════════════════════════════════════════════════════════════════════

  constructor(config: AdaptiveLODConfig = {}, budget?: PerformanceBudget) {
    this.budget     = budget ?? getGlobalBudget();
    this.currentTier = this.budget.tier;

    // Distance thresholds: start with tier-based, apply overrides
    this.distThresholds = {
      ...TIER_DISTANCE_THRESHOLDS[this.currentTier],
      ...config.distanceOverrides,
    };

    this.screenThresholds = {
      ...SCREEN_THRESHOLDS,
      ...config.screenOverrides,
    };

    this.transitionRate       = config.transitionSmoothing ?? 0.15;
    this.enableDensityScaling = config.enableParticleDensityScaling ?? true;
    this.globalParticleCap    = config.globalParticleCap ?? 50_000;

    this.fpsRing = new Float64Array(20);

    // React to tier changes from the budget system
    this.unsubTier = this.budget.onTierChange((next) => {
      this.currentTier = next;
      this.distThresholds = {
        ...TIER_DISTANCE_THRESHOLDS[next],
        ...(config.distanceOverrides ?? {}),
      };
    });
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  Core update — call once per frame
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Evaluate LOD for every cell, considering distance, screen coverage,
   * and the current FPS budget.
   *
   * @param camera    Current camera state.
   * @param cells     Array of all active cells in the scene.
   * @param deltaMs   Frame delta in milliseconds (for FPS tracking).
   * @returns         Array of per-cell LOD assignments.
   */
  update(
    camera: LODCamera,
    cells: ReadonlyArray<LODCellInput>,
    deltaMs: number,
  ): ReadonlyArray<CellLODAssignment> {
    this.frameNumber++;
    this._updateFps(deltaMs);

    const viewportArea = camera.viewportW * camera.viewportH;
    const fpsHeadroom  = this._computeFpsHeadroomFactor();

    this.totalCellCount   = cells.length;
    this.visibleCellCount = 0;
    this.lodCounts        = [0, 0, 0, 0];

    // Resize the assignments array to avoid GC pressure
    if (this.assignments.length < cells.length) {
      this.assignments.length = cells.length;
    }

    for (let i = 0; i < cells.length; i++) {
      const cell = cells[i];

      // ── Distance computation ─────────────────────────────────────────
      const cx = cell.bbox.x + cell.bbox.w * 0.5;
      const cy = cell.bbox.y + cell.bbox.h * 0.5;
      const dx = cx - camera.x;
      const dy = cy - camera.y;
      const dist = Math.sqrt(dx * dx + dy * dy) / Math.max(camera.zoom, 0.01);

      // ── Screen-space coverage ────────────────────────────────────────
      const projW = (cell.bbox.w * camera.zoom * camera.viewportW) / 10;
      const projH = (cell.bbox.h * camera.zoom * camera.viewportH) / 10;
      const screenArea    = projW * projH;
      const screenCoverage = viewportArea > 0 ? screenArea / viewportArea : 0;

      // ── Determine target LOD level ───────────────────────────────────
      const targetLevel = this._classifyLOD(dist, screenCoverage, fpsHeadroom);

      // ── Per-cell state tracking ──────────────────────────────────────
      let state = this.cellStates.get(cell.cellId);
      if (!state) {
        state = {
          currentLevel:    targetLevel,
          targetLevel,
          fractionalLOD:   targetLevel as number,
          upgradeCooldown: 0,
          lastSeenFrame:   this.frameNumber,
        };
        this.cellStates.set(cell.cellId, state);
      }

      state.targetLevel   = targetLevel;
      state.lastSeenFrame = this.frameNumber;

      // ── Smooth LOD transition ────────────────────────────────────────
      this._smoothTransition(state);

      // ── Build render hint ────────────────────────────────────────────
      const renderHint = this._buildRenderHint(
        state.currentLevel,
        cell.materialType,
        cell.patternShader,
        screenCoverage,
      );

      // ── Write assignment ─────────────────────────────────────────────
      const assignment: CellLODAssignment = {
        cellId:         cell.cellId,
        level:          state.currentLevel,
        distance:       dist,
        screenCoverage,
        renderHint,
      };

      this.assignments[i] = assignment;
      this.visibleCellCount++;
      this.lodCounts[state.currentLevel]++;
    }

    // Trim excess
    this.assignments.length = cells.length;

    // ── Update global particle budget ──────────────────────────────────
    this._updateParticleBudget(cells);

    // ── Age out stale cell states (every 300 frames) ───────────────────
    if (this.frameNumber % 300 === 0) {
      this._purgeStaleStates();
    }

    return this.assignments;
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  Public queries
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Global particle budget ratio [0, 1].
   * Emitter systems multiply their base rate by this value.
   */
  getParticleBudgetRatio(): number {
    return this.globalParticleBudgetRatio;
  }

  /**
   * Effective max particle count for this frame, factoring in the
   * performance tier and viewport density scaling.
   */
  getEffectiveParticleCap(): number {
    return Math.floor(this.globalParticleCap * this.globalParticleBudgetRatio);
  }

  /** Per-LOD-level cell count from the last update. */
  getLODDistribution(): Readonly<[number, number, number, number]> {
    return this.lodCounts;
  }

  /** Current smoothed FPS as seen by the LOD system. */
  getFps(): number {
    return this.smoothedFps;
  }

  /** Snapshot for debugging / HUD overlay. */
  snapshot(): AdaptiveLODSnapshot {
    return {
      tier:                   this.currentTier,
      fps:                    parseFloat(this.smoothedFps.toFixed(1)),
      frameNumber:            this.frameNumber,
      totalCells:             this.totalCellCount,
      visibleCells:           this.visibleCellCount,
      lodDistribution: {
        full:      this.lodCounts[LODLevel.FULL],
        reduced:   this.lodCounts[LODLevel.REDUCED],
        billboard: this.lodCounts[LODLevel.BILLBOARD],
        dot:       this.lodCounts[LODLevel.DOT],
      },
      particleBudgetRatio:    parseFloat(this.globalParticleBudgetRatio.toFixed(3)),
      effectiveParticleCap:   this.getEffectiveParticleCap(),
      distanceThresholds:     { ...this.distThresholds },
    };
  }

  /** Teardown — unsubscribe from budget events and clear state. */
  destroy(): void {
    this.unsubTier?.();
    this.unsubTier = null;
    this.cellStates.clear();
    this.assignments.length = 0;
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  Manual overrides
  // ═══════════════════════════════════════════════════════════════════════

  /** Force a specific LOD level for a cell (e.g. for focus / selection). */
  forceCellLOD(cellId: string, level: LODLevel): void {
    let state = this.cellStates.get(cellId);
    if (!state) {
      state = {
        currentLevel:    level,
        targetLevel:     level,
        fractionalLOD:   level as number,
        upgradeCooldown: 0,
        lastSeenFrame:   this.frameNumber,
      };
      this.cellStates.set(cellId, state);
    } else {
      state.currentLevel  = level;
      state.targetLevel   = level;
      state.fractionalLOD = level as number;
    }
  }

  /** Clear a per-cell LOD override, returning it to automatic management. */
  clearCellLODOverride(cellId: string): void {
    this.cellStates.delete(cellId);
  }

  /** Override distance thresholds at runtime. */
  setDistanceThresholds(overrides: Partial<LODDistanceThresholds>): void {
    Object.assign(this.distThresholds, overrides);
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  Internal — LOD classification
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Classify a cell into a discrete LOD level based on distance,
   * screen coverage, and the current FPS headroom factor.
   */
  private _classifyLOD(
    distance: number,
    screenCoverage: number,
    fpsHeadroom: number,
  ): LODLevel {
    // FPS headroom squeezes the distance thresholds: < 1 pulls them inward
    const dFull      = this.distThresholds.fullToReduced      * fpsHeadroom;
    const dReduced   = this.distThresholds.reducedToBillboard * fpsHeadroom;
    const dBillboard = this.distThresholds.billboardToDot     * fpsHeadroom;

    // Distance-based classification
    let level: LODLevel;
    if (distance < dFull) {
      level = LODLevel.FULL;
    } else if (distance < dReduced) {
      level = LODLevel.REDUCED;
    } else if (distance < dBillboard) {
      level = LODLevel.BILLBOARD;
    } else {
      level = LODLevel.DOT;
    }

    // Screen-coverage can only push the level LOWER (never higher).
    // A very small cell shouldn't get full SDF even if it's close.
    if (screenCoverage < this.screenThresholds.minForBillboard) {
      level = Math.max(level, LODLevel.DOT) as LODLevel;
    } else if (screenCoverage < this.screenThresholds.minForReduced) {
      level = Math.max(level, LODLevel.BILLBOARD) as LODLevel;
    } else if (screenCoverage < this.screenThresholds.minForFull) {
      level = Math.max(level, LODLevel.REDUCED) as LODLevel;
    }

    // Emergency FPS cull: if we're under the emergency threshold,
    // push everything at REDUCED or worse to BILLBOARD/DOT
    if (this.smoothedFps < FPS_EMERGENCY_THRESHOLD) {
      if (level === LODLevel.FULL) {
        level = LODLevel.REDUCED;
      } else if (level === LODLevel.REDUCED) {
        level = LODLevel.BILLBOARD;
      }
    }

    return level;
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  Internal — smooth LOD transitions
  // ═══════════════════════════════════════════════════════════════════════

  private _smoothTransition(state: CellLODState): void {
    const target = state.targetLevel as number;
    const current = state.fractionalLOD;

    // Downgrade (increase LOD number) is immediate to protect frame budget
    if (target > current) {
      state.fractionalLOD    = target;
      state.currentLevel     = state.targetLevel;
      state.upgradeCooldown  = LOD_UPGRADE_COOLDOWN_FRAMES;
      return;
    }

    // Upgrade (decrease LOD number) is gated by cooldown + smoothing
    if (target < current) {
      if (state.upgradeCooldown > 0) {
        state.upgradeCooldown--;
        return; // hold current LOD during cooldown
      }

      // Smooth interpolation toward the target
      const rate = 1.0 - this.transitionRate;
      state.fractionalLOD = current + (target - current) * rate;

      // Snap when close enough
      if (Math.abs(state.fractionalLOD - target) < 0.05) {
        state.fractionalLOD = target;
      }

      // Discrete level follows the rounded fractional value
      state.currentLevel = Math.round(state.fractionalLOD) as LODLevel;
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  Internal — render hint construction
  // ═══════════════════════════════════════════════════════════════════════

  private _buildRenderHint(
    level: LODLevel,
    originalMaterial: MaterialType,
    originalPattern: PatternShader,
    screenCoverage: number,
  ): CellRenderHint {
    switch (level) {
      case LODLevel.FULL:
        return makeFullHint(originalMaterial, originalPattern);

      case LODLevel.REDUCED:
        return makeReducedHint(originalMaterial);

      case LODLevel.BILLBOARD: {
        const hint = makeBillboardHint();
        // Fade billboard opacity for very small cells approaching DOT threshold
        const fadeRange = this.screenThresholds.minForReduced - this.screenThresholds.minForBillboard;
        if (fadeRange > 0) {
          const fadeT = (screenCoverage - this.screenThresholds.minForBillboard) / fadeRange;
          hint.billboardOpacity = 0.4 + 0.6 * Math.min(Math.max(fadeT, 0), 1);
        }
        return hint;
      }

      case LODLevel.DOT: {
        // Dot size based on screen coverage: tiny cells → 1px, larger → 4px
        const sizePx = Math.max(1, Math.min(4, Math.ceil(screenCoverage * 20_000)));
        return makeDotHint(sizePx);
      }

      default:
        return makeFullHint(originalMaterial, originalPattern);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  Internal — FPS tracking
  // ═══════════════════════════════════════════════════════════════════════

  private _updateFps(deltaMs: number): void {
    const clamped = Math.min(Math.max(deltaMs, 1), 200);
    this.fpsRing[this.fpsRingIdx] = clamped;
    this.fpsRingIdx = (this.fpsRingIdx + 1) % this.fpsRing.length;
    if (this.fpsRingIdx === 0) this.fpsRingFilled = true;

    const count = this.fpsRingFilled ? this.fpsRing.length : this.fpsRingIdx;
    if (count === 0) { this.smoothedFps = 60; return; }

    let sum = 0;
    for (let i = 0; i < count; i++) sum += this.fpsRing[i];
    this.smoothedFps = 1000 / (sum / count);
  }

  /**
   * Compute a [0.3, 1.2] headroom factor from the current FPS.
   *   fps >= TARGET_FPS   → 1.0  (full thresholds, may exceed 1.0 for bonus)
   *   fps == AGGRESSIVE   → 0.6  (thresholds halved)
   *   fps <= EMERGENCY    → 0.3  (aggressive culling)
   */
  private _computeFpsHeadroomFactor(): number {
    if (this.smoothedFps >= TARGET_FPS) {
      // Bonus headroom: if we're above 60, slightly relax thresholds
      const bonus = Math.min((this.smoothedFps - TARGET_FPS) / 30, 0.2);
      return 1.0 + bonus;
    }

    // Linear interpolation between emergency and target
    const t = (this.smoothedFps - FPS_EMERGENCY_THRESHOLD)
            / (TARGET_FPS - FPS_EMERGENCY_THRESHOLD);
    const clamped = Math.min(Math.max(t, 0), 1);

    return 0.3 + 0.7 * clamped;
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  Internal — global particle budget
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Compute the global particle budget ratio from three factors:
   *   1. Performance tier scaling (ULTRA=1.0, LOW=0.25)
   *   2. Viewport density (visible / total cells)
   *   3. FPS headroom (exponential damping when dropping below target)
   *
   * The resulting ratio is applied to ALL emitter rates and particle caps.
   */
  private _updateParticleBudget(cells: ReadonlyArray<LODCellInput>): void {
    // Factor 1: Tier-based baseline
    const tierScale = TIER_PARTICLE_BUDGET_SCALE[this.currentTier];

    // Factor 2: Viewport density — only pay for what's visible
    let densityScale = 1.0;
    if (this.enableDensityScaling && this.totalCellCount > 0) {
      // Sum the per-cell particle ratios from their LOD assignments
      let totalDemand = 0;
      let nominalTotal = 0;
      for (let i = 0; i < cells.length; i++) {
        const assignment = this.assignments[i];
        if (!assignment) continue;
        const nominal = cells[i].nominalParticleCount;
        nominalTotal += nominal;
        totalDemand  += nominal * assignment.renderHint.particleRatio;
      }
      densityScale = nominalTotal > 0 ? totalDemand / nominalTotal : 1.0;
    }

    // Factor 3: FPS headroom damping
    let fpsDamping = 1.0;
    if (this.smoothedFps < TARGET_FPS) {
      // Quadratic damping: drops faster as FPS drops
      const ratio = this.smoothedFps / TARGET_FPS;
      fpsDamping = ratio * ratio;
    }

    this.globalParticleBudgetRatio = Math.min(
      1.0,
      Math.max(0.05, tierScale * densityScale * fpsDamping),
    );
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  Internal — stale state cleanup
  // ═══════════════════════════════════════════════════════════════════════

  /** Remove cell LOD states that haven't been seen for 600 frames. */
  private _purgeStaleStates(): void {
    const staleThreshold = this.frameNumber - 600;
    for (const [id, state] of this.cellStates) {
      if (state.lastSeenFrame < staleThreshold) {
        this.cellStates.delete(id);
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Snapshot type
// ─────────────────────────────────────────────────────────────────────────────

export interface AdaptiveLODSnapshot {
  tier: Tier;
  fps: number;
  frameNumber: number;
  totalCells: number;
  visibleCells: number;
  lodDistribution: {
    full: number;
    reduced: number;
    billboard: number;
    dot: number;
  };
  particleBudgetRatio: number;
  effectiveParticleCap: number;
  distanceThresholds: LODDistanceThresholds;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience singleton
// ─────────────────────────────────────────────────────────────────────────────

let _globalLOD: AdaptiveLOD | null = null;

/** Lazily create / return the process-wide adaptive LOD instance. */
export function getGlobalLOD(): AdaptiveLOD {
  if (!_globalLOD) _globalLOD = new AdaptiveLOD();
  return _globalLOD;
}

/** Replace the global LOD (useful in tests). */
export function setGlobalLOD(lod: AdaptiveLOD): void {
  _globalLOD = lod;
}
