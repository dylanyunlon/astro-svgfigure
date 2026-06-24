/**
 * pic-flip-solver.ts — M780: PIC/FLIP Hybrid MAC Grid Solver
 *
 * Implements a 2-D PIC/FLIP fluid simulation on a staggered Marker-And-Cell
 * (MAC) grid.  The solver follows the standard pipeline from Bridson 2015
 * ("Fluid Simulation for Computer Graphics") and the FLIP method of
 * Zhu & Bridson 2005 ("Animating Sand as a Fluid"):
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Pipeline — per time step
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  1. P2G  — Transfer particle velocities to the staggered MAC grid using
 *            bilinear (hat-function) splatting.  Each velocity component lives
 *            on its own half-offset face grid.
 *
 *  2. SAVE — Copy the freshly-transferred grid velocities into a separate
 *            buffer (u_old, v_old) for the FLIP delta computation later.
 *
 *  3. GRAVITY — Apply external body force (gravity) to the v-component grid.
 *
 *  4. PRESSURE POISSON — Enforce incompressibility by solving ∇²p = ρ/Δt · ∇·u
 *            via Gauss-Seidel iteration on the pressure grid.  Then project
 *            the velocity field:  u ← u − (Δt/ρ) ∇p.
 *
 *  5. BOUNDARY — Enforce solid wall boundary conditions on grid faces.
 *
 *  6. G2P  — Interpolate updated grid velocities back to particles.  The
 *            final particle velocity is a blend of PIC (full grid interp)
 *            and FLIP (particle vel + grid delta):
 *
 *              v_particle = (1 − α) · v_PIC + α · (v_particle + Δv_FLIP)
 *
 *            where α = flipRatio (default 0.95).
 *
 *  7. ADVECT — Move particles by their new velocities:  x += Δt · v.
 *             Clamp to domain boundaries.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * MAC grid layout
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  Pressure p[i][j] lives at cell centres:  ((i+0.5)·h, (j+0.5)·h)
 *  Horizontal velocity u[i][j] on vertical faces:  (i·h, (j+0.5)·h)
 *  Vertical velocity v[i][j] on horizontal faces:  ((i+0.5)·h, j·h)
 *
 *  Grid dimensions:  numX × numY cells
 *  u grid:  (numX+1) × numY
 *  v grid:  numX × (numY+1)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Sources / lineage
 * ─────────────────────────────────────────────────────────────────────────────
 *  • Bridson, "Fluid Simulation for Computer Graphics", 2nd ed., 2015
 *  • Zhu & Bridson, "Animating Sand as a Fluid", SIGGRAPH 2005
 *  • src/lib/sph/dfsph-solver.ts       (solver pattern, ~400 lines)
 *  • src/lib/sph/at-navier-stokes.ts   (grid-based fluid, pressure Poisson)
 *  • src/lib/sph/sph-kernels.ts        (SPHConfig interface augmentation)
 *
 * Research: xiaodi #M780 — cell-pubsub-loop
 * ─────────────────────────────────────────────────────────────────────────────
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** Configuration for the PIC/FLIP MAC grid solver. */








export interface PICFLIPConfig {
  /** Domain width in world units. */
  domainW: number;
  /** Domain height in world units. */
  domainH: number;
  /** Number of grid cells along X. */
  numX: number;
  /** Number of grid cells along Y. */
  numY: number;
  /** Time step Δt (seconds). Default 1/120. */
  dt?: number;
  /** Fluid density ρ (kg/m³). Default 1000. */
  density?: number;
  /** Gravitational acceleration (positive = downward). Default 9.81. */
  gravity?: number;
  /** PIC/FLIP blend ratio α. 0 = pure PIC, 1 = pure FLIP. Default 0.95. */
  flipRatio?: number;
  /** Number of Gauss-Seidel pressure iterations. Default 100. */
  pressureIters?: number;
  /** Over-relaxation factor for pressure solve (SOR). Default 1.9. */
  overRelax?: number;
}

/** A single Lagrangian marker particle. */
export interface FLIPParticle {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

/** Staggered MAC grid storing velocity components on faces. */
export interface MACGrid {
  /** Horizontal velocity on vertical faces: (numX+1) × numY. */
  u: Float64Array;
  /** Vertical velocity on horizontal faces: numX × (numY+1). */
  v: Float64Array;
  /** Saved u before pressure project (for FLIP delta). */
  uOld: Float64Array;
  /** Saved v before pressure project (for FLIP delta). */
  vOld: Float64Array;
  /** Accumulated weights for u splatting. */
  uWeight: Float64Array;
  /** Accumulated weights for v splatting. */
  vWeight: Float64Array;
  /** Pressure at cell centres: numX × numY. */
  p: Float64Array;
  /** Cell type: 0 = air, 1 = fluid, 2 = solid. numX × numY. */
  cellType: Uint8Array;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const CELL_AIR   = 0;
const CELL_FLUID = 1;
const CELL_SOLID = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Grid creation
// ─────────────────────────────────────────────────────────────────────────────

/** Allocate a zeroed MAC grid for the given cell dimensions. */
export function createMACGrid(numX: number, numY: number): MACGrid {
  const uLen = (numX + 1) * numY;
  const vLen = numX * (numY + 1);
  const cLen = numX * numY;

  return {
    u:        new Float64Array(uLen),
    v:        new Float64Array(vLen),
    uOld:     new Float64Array(uLen),
    vOld:     new Float64Array(vLen),
    uWeight:  new Float64Array(uLen),
    vWeight:  new Float64Array(vLen),
    p:        new Float64Array(cLen),
    cellType: new Uint8Array(cLen),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers — flat index accessors
// ─────────────────────────────────────────────────────────────────────────────

/** u lives on vertical faces → (numX+1) columns, numY rows. */
function uIdx(i: number, j: number, numX: number): number {
  return i + j * (numX + 1);
}

/** v lives on horizontal faces → numX columns, (numY+1) rows. */
function vIdx(i: number, j: number, numX: number): number {
  return i + j * numX;
}

/** Cell-centred values (pressure, cellType) → numX columns, numY rows. */
function cIdx(i: number, j: number, numX: number): number {
  return i + j * numX;
}

// ─────────────────────────────────────────────────────────────────────────────
// Bilinear interpolation on the staggered grid
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Sample a staggered field using bilinear interpolation.
 *
 * @param field  The flat array (u or v grid).
 * @param fx     Continuous x index in field coordinates (already offset
 *               for the stagger — caller handles the half-cell shift).
 * @param fy     Continuous y index in field coordinates.
 * @param cols   Number of columns in this field's storage layout.
 * @param rows   Number of rows in this field's storage layout.
 */
function sampleField(
  field: Float64Array,
  fx: number,
  fy: number,
  cols: number,
  rows: number,
): number {
  // Clamp to valid range
  const cx = Math.max(0, Math.min(fx, cols - 1.001));
  const cy = Math.max(0, Math.min(fy, rows - 1.001));

  const i0 = Math.floor(cx);
  const j0 = Math.floor(cy);
  const i1 = Math.min(i0 + 1, cols - 1);
  const j1 = Math.min(j0 + 1, rows - 1);

  const sx = cx - i0;
  const sy = cy - j0;

  return (
    field[i0 + j0 * cols] * (1 - sx) * (1 - sy) +
    field[i1 + j0 * cols] * sx       * (1 - sy) +
    field[i0 + j1 * cols] * (1 - sx) * sy       +
    field[i1 + j1 * cols] * sx       * sy
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1 — Particle-to-Grid (P2G) transfer
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Splat particle velocities onto the MAC grid using bilinear (hat-function)
 * weights.  Accumulates weighted sums; caller must normalise afterward.
 */
function particlesToGrid(
  particles: FLIPParticle[],
  grid: MACGrid,
  numX: number,
  numY: number,
  h: number,
): void {
  const invH = 1.0 / h;
  const uCols = numX + 1;
  const uRows = numY;
  const vCols = numX;
  const vRows = numY + 1;

  // Zero accumulators
  grid.u.fill(0);
  grid.v.fill(0);
  grid.uWeight.fill(0);
  grid.vWeight.fill(0);

  for (let pi = 0; pi < particles.length; pi++) {
    const p = particles[pi];

    // ── u-component (vertical faces at integer x, half y) ──
    {
      const fx = p.x * invH;
      const fy = p.y * invH - 0.5;

      const i0 = Math.floor(fx);
      const j0 = Math.floor(fy);

      const sx = fx - i0;
      const sy = fy - j0;

      // Splat to 4 surrounding u-nodes
      const w00 = (1 - sx) * (1 - sy);
      const w10 = sx       * (1 - sy);
      const w01 = (1 - sx) * sy;
      const w11 = sx       * sy;

      if (i0 >= 0 && i0 < uCols && j0 >= 0 && j0 < uRows) {
        const idx = i0 + j0 * uCols;
        grid.u[idx] += w00 * p.vx;
        grid.uWeight[idx] += w00;
      }
      if (i0 + 1 >= 0 && i0 + 1 < uCols && j0 >= 0 && j0 < uRows) {
        const idx = (i0 + 1) + j0 * uCols;
        grid.u[idx] += w10 * p.vx;
        grid.uWeight[idx] += w10;
      }
      if (i0 >= 0 && i0 < uCols && j0 + 1 >= 0 && j0 + 1 < uRows) {
        const idx = i0 + (j0 + 1) * uCols;
        grid.u[idx] += w01 * p.vx;
        grid.uWeight[idx] += w01;
      }
      if (i0 + 1 >= 0 && i0 + 1 < uCols && j0 + 1 >= 0 && j0 + 1 < uRows) {
        const idx = (i0 + 1) + (j0 + 1) * uCols;
        grid.u[idx] += w11 * p.vx;
        grid.uWeight[idx] += w11;
      }
    }

    // ── v-component (horizontal faces at half x, integer y) ──
    {
      const fx = p.x * invH - 0.5;
      const fy = p.y * invH;

      const i0 = Math.floor(fx);
      const j0 = Math.floor(fy);

      const sx = fx - i0;
      const sy = fy - j0;

      const w00 = (1 - sx) * (1 - sy);
      const w10 = sx       * (1 - sy);
      const w01 = (1 - sx) * sy;
      const w11 = sx       * sy;

      if (i0 >= 0 && i0 < vCols && j0 >= 0 && j0 < vRows) {
        const idx = i0 + j0 * vCols;
        grid.v[idx] += w00 * p.vy;
        grid.vWeight[idx] += w00;
      }
      if (i0 + 1 >= 0 && i0 + 1 < vCols && j0 >= 0 && j0 < vRows) {
        const idx = (i0 + 1) + j0 * vCols;
        grid.v[idx] += w10 * p.vy;
        grid.vWeight[idx] += w10;
      }
      if (i0 >= 0 && i0 < vCols && j0 + 1 >= 0 && j0 + 1 < vRows) {
        const idx = i0 + (j0 + 1) * vCols;
        grid.v[idx] += w01 * p.vy;
        grid.vWeight[idx] += w01;
      }
      if (i0 + 1 >= 0 && i0 + 1 < vCols && j0 + 1 >= 0 && j0 + 1 < vRows) {
        const idx = (i0 + 1) + (j0 + 1) * vCols;
        grid.v[idx] += w11 * p.vy;
        grid.vWeight[idx] += w11;
      }
    }
  }

  // Normalise by accumulated weights
  for (let k = 0; k < grid.u.length; k++) {
    if (grid.uWeight[k] > 1e-12) grid.u[k] /= grid.uWeight[k];
    else grid.u[k] = 0;
  }

  for (let k = 0; k < grid.v.length; k++) {
    if (grid.vWeight[k] > 1e-12) grid.v[k] /= grid.vWeight[k];
    else grid.v[k] = 0;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2 — Mark cell types (air / fluid / solid)
// ─────────────────────────────────────────────────────────────────────────────

function markCellTypes(
  particles: FLIPParticle[],
  grid: MACGrid,
  numX: number,
  numY: number,
  h: number,
): void {
  const invH = 1.0 / h;

  // Border cells are solid, interior starts as air
  for (let j = 0; j < numY; j++) {
    for (let i = 0; i < numX; i++) {
      if (i === 0 || i === numX - 1 || j === 0 || j === numY - 1) {
        grid.cellType[cIdx(i, j, numX)] = CELL_SOLID;
      } else {
        grid.cellType[cIdx(i, j, numX)] = CELL_AIR;
      }
    }
  }

  // Mark cells containing particles as fluid
  for (let pi = 0; pi < particles.length; pi++) {
    const p = particles[pi];
    const ci = Math.floor(p.x * invH);
    const cj = Math.floor(p.y * invH);

    const i = Math.max(1, Math.min(ci, numX - 2));
    const j = Math.max(1, Math.min(cj, numY - 2));

    grid.cellType[cIdx(i, j, numX)] = CELL_FLUID;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3 — Apply gravity
// ─────────────────────────────────────────────────────────────────────────────

function applyGravity(grid: MACGrid, numX: number, numY: number, dt: number, gravity: number): void {
  const vCols = numX;
  const vRows = numY + 1;

  for (let j = 1; j < vRows; j++) {
    for (let i = 0; i < vCols; i++) {
      grid.v[vIdx(i, j, numX)] -= gravity * dt;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4 — Pressure Poisson solve + velocity projection
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Solve ∇²p = (ρ/Δt) · ∇·u via Gauss-Seidel with SOR, then subtract the
 * pressure gradient from the velocity field to enforce ∇·u = 0.
 */
function solvePressure(
  grid: MACGrid,
  numX: number,
  numY: number,
  h: number,
  dt: number,
  density: number,
  iters: number,
  overRelax: number,
): void {
  const invH = 1.0 / h;
  const scale = dt / (density * h);

  grid.p.fill(0);

  for (let iter = 0; iter < iters; iter++) {
    for (let j = 1; j < numY - 1; j++) {
      for (let i = 1; i < numX - 1; i++) {
        if (grid.cellType[cIdx(i, j, numX)] !== CELL_FLUID) continue;

        // Count non-solid neighbours and compute divergence
        const sL = grid.cellType[cIdx(i - 1, j, numX)] !== CELL_SOLID ? 1 : 0;
        const sR = grid.cellType[cIdx(i + 1, j, numX)] !== CELL_SOLID ? 1 : 0;
        const sB = grid.cellType[cIdx(i, j - 1, numX)] !== CELL_SOLID ? 1 : 0;
        const sT = grid.cellType[cIdx(i, j + 1, numX)] !== CELL_SOLID ? 1 : 0;
        const sTotal = sL + sR + sB + sT;

        if (sTotal === 0) continue;

        // Velocity divergence at cell centre
        const div =
          grid.u[uIdx(i + 1, j, numX)] - grid.u[uIdx(i, j, numX)] +
          grid.v[vIdx(i, j + 1, numX)] - grid.v[vIdx(i, j, numX)];

        // Neighbour pressure sum (air cells contribute p = 0)
        let pSum = 0;
        if (sL) pSum += grid.p[cIdx(i - 1, j, numX)];
        if (sR) pSum += grid.p[cIdx(i + 1, j, numX)];
        if (sB) pSum += grid.p[cIdx(i, j - 1, numX)];
        if (sT) pSum += grid.p[cIdx(i, j + 1, numX)];

        const pNew = (pSum - div / scale) / sTotal;
        grid.p[cIdx(i, j, numX)] += overRelax * (pNew - grid.p[cIdx(i, j, numX)]);
      }
    }
  }

  // ── Project: subtract pressure gradient from velocity ──
  for (let j = 1; j < numY - 1; j++) {
    for (let i = 1; i < numX - 1; i++) {
      if (grid.cellType[cIdx(i, j, numX)] !== CELL_FLUID) continue;

      const pC = grid.p[cIdx(i, j, numX)];

      // u faces
      if (grid.cellType[cIdx(i - 1, j, numX)] !== CELL_SOLID) {
        grid.u[uIdx(i, j, numX)] -= scale * pC;
      }
      if (grid.cellType[cIdx(i + 1, j, numX)] !== CELL_SOLID) {
        grid.u[uIdx(i + 1, j, numX)] += scale * pC;
      }

      // v faces
      if (grid.cellType[cIdx(i, j - 1, numX)] !== CELL_SOLID) {
        grid.v[vIdx(i, j, numX)] -= scale * pC;
      }
      if (grid.cellType[cIdx(i, j + 1, numX)] !== CELL_SOLID) {
        grid.v[vIdx(i, j + 1, numX)] += scale * pC;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5 — Enforce solid boundary conditions
// ─────────────────────────────────────────────────────────────────────────────

function enforceBoundary(grid: MACGrid, numX: number, numY: number): void {
  // Zero velocity on solid cell faces
  for (let j = 0; j < numY; j++) {
    for (let i = 0; i < numX; i++) {
      if (grid.cellType[cIdx(i, j, numX)] === CELL_SOLID) {
        grid.u[uIdx(i, j, numX)]     = 0;
        grid.u[uIdx(i + 1, j, numX)] = 0;
        grid.v[vIdx(i, j, numX)]     = 0;
        grid.v[vIdx(i, j + 1, numX)] = 0;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 6 — Grid-to-Particle (G2P) transfer with PIC/FLIP blending
// ─────────────────────────────────────────────────────────────────────────────

function gridToParticles(
  particles: FLIPParticle[],
  grid: MACGrid,
  numX: number,
  numY: number,
  h: number,
  flipRatio: number,
): void {
  const invH = 1.0 / h;
  const uCols = numX + 1;
  const uRows = numY;
  const vCols = numX;
  const vRows = numY + 1;

  for (let pi = 0; pi < particles.length; pi++) {
    const p = particles[pi];

    // Interpolate current (post-project) grid velocity → PIC velocity
    const uPIC = sampleField(grid.u, p.x * invH, p.y * invH - 0.5, uCols, uRows);
    const vPIC = sampleField(grid.v, p.x * invH - 0.5, p.y * invH, vCols, vRows);

    // Interpolate old (pre-project) grid velocity → for FLIP delta
    const uOld = sampleField(grid.uOld, p.x * invH, p.y * invH - 0.5, uCols, uRows);
    const vOld = sampleField(grid.vOld, p.x * invH - 0.5, p.y * invH, vCols, vRows);

    // FLIP delta
    const duFLIP = uPIC - uOld;
    const dvFLIP = vPIC - vOld;

    // Blend:  v = (1-α)·PIC + α·(v_particle + Δ)
    p.vx = (1 - flipRatio) * uPIC + flipRatio * (p.vx + duFLIP);
    p.vy = (1 - flipRatio) * vPIC + flipRatio * (p.vy + dvFLIP);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 7 — Advect particles and clamp to domain
// ─────────────────────────────────────────────────────────────────────────────

function advectParticles(
  particles: FLIPParticle[],
  dt: number,
  domainW: number,
  domainH: number,
  h: number,
): void {
  // Keep particles one cell inward from solid boundary
  const lo = h * 1.01;
  const hiX = domainW - h * 1.01;
  const hiY = domainH - h * 1.01;

  for (let pi = 0; pi < particles.length; pi++) {
    const p = particles[pi];

    p.x += dt * p.vx;
    p.y += dt * p.vy;

    // Clamp to interior and zero-out the velocity component on contact
    if (p.x < lo)  { p.x = lo;  p.vx = Math.max(p.vx, 0); }
    if (p.x > hiX) { p.x = hiX; p.vx = Math.min(p.vx, 0); }
    if (p.y < lo)  { p.y = lo;  p.vy = Math.max(p.vy, 0); }
    if (p.y > hiY) { p.y = hiY; p.vy = Math.min(p.vy, 0); }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Full PIC/FLIP time step
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Advance the PIC/FLIP simulation by one time step.
 *
 * 1. P2G transfer (bilinear splat)
 * 2. Save pre-project velocities for FLIP
 * 3. Mark cell types (air / fluid / solid)
 * 4. Apply gravity
 * 5. Pressure Poisson solve + project
 * 6. Enforce boundary conditions
 * 7. G2P transfer (PIC/FLIP blend)
 * 8. Advect particles
 */
export function stepPICFLIP(
  particles: FLIPParticle[],
  grid: MACGrid,
  config: PICFLIPConfig,
): void {
  const dt         = config.dt            ?? 1 / 120;
  const density    = config.density       ?? 1000;
  const gravity    = config.gravity       ?? 9.81;
  const flipRatio  = config.flipRatio     ?? 0.95;
  const pressIters = config.pressureIters ?? 100;
  const overRelax  = config.overRelax     ?? 1.9;
  const { numX, numY, domainW, domainH } = config;
  const h = domainW / numX;

  // 1. P2G
  particlesToGrid(particles, grid, numX, numY, h);

  // 2. Save for FLIP delta
  grid.uOld.set(grid.u);
  grid.vOld.set(grid.v);

  // 3. Mark cell types
  markCellTypes(particles, grid, numX, numY, h);

  // 4. Gravity
  applyGravity(grid, numX, numY, dt, gravity);

  // 5. Pressure solve + project
  solvePressure(grid, numX, numY, h, dt, density, pressIters, overRelax);

  // 6. Boundary
  enforceBoundary(grid, numX, numY);

  // 7. G2P
  gridToParticles(particles, grid, numX, numY, h, flipRatio);

  // 8. Advect
  advectParticles(particles, dt, domainW, domainH, h);
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience — dam-break particle initialiser
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a block of particles filling the left portion of the domain,
 * suitable for a classic dam-break test.
 *
 * @param fillX  Fraction of domain width to fill (default 0.3).
 * @param fillY  Fraction of domain height to fill (default 0.8).
 * @param ppc    Particles per cell per axis (default 2 → 4 per cell).
 */
export function createDamBreak(
  config: PICFLIPConfig,
  fillX = 0.3,
  fillY = 0.8,
  ppc = 2,
): FLIPParticle[] {
  const h = config.domainW / config.numX;
  const spacing = h / ppc;
  const particles: FLIPParticle[] = [];

  const x0 = h * 1.5;
  const y0 = h * 1.5;
  const x1 = config.domainW * fillX;
  const y1 = config.domainH * fillY;

  for (let y = y0; y < y1; y += spacing) {
    for (let x = x0; x < x1; x += spacing) {
      particles.push({ x, y, vx: 0, vy: 0 });
    }
  }

  return particles;
}
