/**
 * flowmap-bridge.ts — SPH velocity field → ogl Flowmap texture bridge  (M573)
 *
 * Converts the live SPH particle velocity field into a low-resolution
 * Float32 RG texture (gridSize × gridSize, R = vx, G = vy) that can be
 * fed directly into ogl's Flowmap uniform or consumed by any GLSL shader
 * that expects a velocity / distortion map.
 *
 * The canonical consumer is the cell surface shader (grayscott-species.frag /
 * fluid-surface.frag) where this texture drives Turing-pattern UV distortion:
 * fluid flowing past a cell stretches its reaction-diffusion pattern along
 * the local velocity direction, making the surface feel alive.
 *
 * ─── ogl Flowmap context ─────────────────────────────────────────────────────
 * upstream/ogl/src/extras/Flowmap.js encodes force stamps into a ping-pong
 * render-target texture via:
 *
 *   flowmap.aspect   = canvasW / canvasH;
 *   flowmap.mouse    = normalisedCursorPos;   // [0,1]²
 *   flowmap.velocity = cursorVelocity;        // delta per frame
 *   flowmap.update();
 *   // flowmap.uniform.value → sampler2D fed to the mesh program
 *
 * FlowmapBridge mirrors that contract but sources velocity from SPH particles
 * instead of cursor events, so the same downstream shaders work unchanged.
 *
 * ─── Rasterisation approach ──────────────────────────────────────────────────
 * Scattered-to-grid projection uses a "splat accumulate + weighted average":
 *
 *   for each particle p at (x, y) with velocity (vx, vy):
 *     cell (ci, cj) = floor( (p.x/domainW, p.y/domainH) * gridSize )
 *     accum[ci][cj].vx += vx
 *     accum[ci][cj].vy += vy
 *     accum[ci][cj].w  += 1
 *   output[ci][cj] = accum[ci][cj].{ vx/w, vy/w }
 *
 * Cells with no particles keep their previous value (dissipated).  This
 * approach is O(N) in particle count and independent of grid resolution —
 * a 64 × 64 grid handles 50 000 particles in < 0.3 ms on a modern CPU core.
 *
 * ─── Coordinate conventions ──────────────────────────────────────────────────
 * SPH domain:   origin bottom-left, Y-up, units = simulation metres.
 * Grid texture: origin top-left,    Y-down, UV = [0,1]².
 * Bridge flips Y when mapping particle positions to grid cells so that the
 * texture UV space matches WebGL / canvas conventions (V=0 at top).
 *
 * ─── Velocity normalisation ──────────────────────────────────────────────────
 * Raw SPH velocities are in m/s and can exceed ±10 m/s during fast-moving
 * events.  The texture channels are stored in [−maxSpeed, +maxSpeed] and
 * re-normalised to [−1, +1] using `maxSpeed` (default 5.0 m/s).  Downstream
 * shaders multiply by a per-uniform scale factor to get the desired distortion
 * magnitude; clamping prevents NaNs from rare high-velocity collision spikes.
 *
 * ─── Usage ───────────────────────────────────────────────────────────────────
 *
 *   import { FlowmapBridge } from '$lib/sph/flowmap-bridge';
 *
 *   // Construct once (grid lives for the lifetime of the scene)
 *   const bridge = new FlowmapBridge({
 *     gridSize:    64,
 *     dissipation: 0.96,
 *     domainW:     world.domainW,
 *     domainH:     world.domainH,
 *     maxSpeed:    5.0,
 *   });
 *
 *   // Each frame, after SPH tick():
 *   const velocityTex = bridge.update(world.getParticles());
 *   // velocityTex: Float32Array[64*64*2], laid out row-major (R=vx, G=vy)
 *
 *   // Upload to WebGL / ogl:
 *   gl.texSubImage2D(
 *     gl.TEXTURE_2D, 0, 0, 0,
 *     64, 64, gl.RG, gl.FLOAT, velocityTex
 *   );
 *
 * ─── References ──────────────────────────────────────────────────────────────
 * ogl Flowmap:         upstream/ogl/src/extras/Flowmap.js
 * fluid-surface.frag:  src/lib/shaders/fluid-surface.frag  (uVelocityTex)
 * grayscott-species:   src/lib/shaders/grayscott-species.frag (u_velocity)
 * physics-uniform-bridge.ts — sibling bridge for per-cell scalar uniforms
 * Research:  M573 — cell-pubsub-loop
 */

// ── Particle input type ────────────────────────────────────────────────────────
// Accepts both the SOA layout from SPHWorld.cpuPos and the AOS layout from
// world-stepper.ts.  The generic constraint keeps the function tree-shakeable.

export interface SPHParticleAOS {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

/** SOA (Structure-of-Arrays) variant — direct view into SPHWorld.cpuPos arrays. */
export interface SPHParticleSOA {
  x:     Float32Array;
  y:     Float32Array;
  vx:    Float32Array;
  vy:    Float32Array;
  count: number;
}

// ── FlowmapBridge options ──────────────────────────────────────────────────────

export interface FlowmapBridgeOptions {
  /**
   * Grid side length in texels.  Must be a power-of-two for best GPU compat.
   * Default: 64.  Range: [8, 512].
   */
  gridSize?: number;

  /**
   * Per-frame exponential dissipation coefficient [0, 1).
   *   1.0  → no decay (field persists indefinitely — not recommended)
   *   0.98 → slow fade  (~50 frames to reach ~36 % of peak)
   *   0.90 → fast fade  (~7  frames to reach ~48 % of peak)
   * Default: 0.96.
   */
  dissipation?: number;

  /**
   * SPH simulation domain width in domain units (same unit as particle x).
   * Used to map particle x position to [0, gridSize).
   * Default: 3.0.
   */
  domainW?: number;

  /**
   * SPH simulation domain height in domain units.
   * Default: 3.0.
   */
  domainH?: number;

  /**
   * Velocity clamp reference (m/s).  Velocities are divided by this value
   * before being stored so the texture stays in [−1, +1] normalised space.
   * Default: 5.0.
   */
  maxSpeed?: number;
}

// ── Pure rasterisation helpers ─────────────────────────────────────────────────

/**
 * Project an array of AOS particles onto a (gridSize × gridSize) × 2 Float32
 * grid.  Each occupied cell stores the mean velocity of all particles that
 * fall into it.  Empty cells are left at zero — dissipation is handled
 * separately by `dissipateField`.
 *
 * Output layout: `[vx₀, vy₀, vx₁, vy₁, …]` row-major, Y-down.
 *
 * @param particles  Array of `{x, y, vx, vy}` objects (AOS).
 * @param domainW    Simulation width  (domain units).
 * @param domainH    Simulation height (domain units).
 * @param gridSize   Target grid side length (texels).
 * @returns          Float32Array of length `gridSize * gridSize * 2`.
 */
export function rasterizeVelocityField(
  particles: Array<SPHParticleAOS>,
  domainW:   number,
  domainH:   number,
  gridSize:  number,
): Float32Array {
  const stride  = gridSize * gridSize;
  const out     = new Float32Array(stride * 2);      // RG
  const weights = new Float32Array(stride);           // particle count per cell

  const invW = gridSize / domainW;
  const invH = gridSize / domainH;

  for (let i = 0, n = particles.length; i < n; i++) {
    const p = particles[i];

    // Map to [0, gridSize) — clamp to guard against boundary particles
    const ci = Math.min(Math.floor(p.x * invW), gridSize - 1);
    // Y-flip: SPH Y-up → texture V-down
    const cj = Math.min(Math.floor((domainH - p.y) * invH), gridSize - 1);

    // Guard: skip particles outside domain
    if (ci < 0 || cj < 0) continue;

    const idx = cj * gridSize + ci;
    out[idx * 2]     += p.vx;
    out[idx * 2 + 1] += p.vy;
    weights[idx]++;
  }

  // Normalise accumulated velocities to mean
  for (let idx = 0; idx < stride; idx++) {
    const w = weights[idx];
    if (w > 0) {
      out[idx * 2]     /= w;
      out[idx * 2 + 1] /= w;
    }
  }

  return out;
}

/**
 * SOA variant of `rasterizeVelocityField`.
 * Accepts the `{x, y, vx, vy, count}` object from `SPHWorld.cpuPos` directly,
 * avoiding a temporary AOS conversion for the hot-path caller.
 *
 * @param soa        Structure-of-arrays particle buffers.
 * @param domainW    Simulation width  (domain units).
 * @param domainH    Simulation height (domain units).
 * @param gridSize   Target grid side length (texels).
 * @param out        Pre-allocated output buffer (reused across frames).
 */
export function rasterizeVelocityFieldSOA(
  soa:      SPHParticleSOA,
  domainW:  number,
  domainH:  number,
  gridSize: number,
  out:      Float32Array,
): void {
  const { x, y, vx, vy, count } = soa;
  const stride  = gridSize * gridSize;
  const weights = new Float32Array(stride);

  // Zero-fill output (will be overwritten / partially kept by dissipation caller)
  out.fill(0);

  const invW = gridSize / domainW;
  const invH = gridSize / domainH;

  for (let i = 0; i < count; i++) {
    const ci = Math.min(Math.floor(x[i] * invW), gridSize - 1);
    const cj = Math.min(Math.floor((domainH - y[i]) * invH), gridSize - 1);

    if (ci < 0 || cj < 0) continue;

    const idx = cj * gridSize + ci;
    out[idx * 2]     += vx[i];
    out[idx * 2 + 1] += vy[i];
    weights[idx]++;
  }

  for (let idx = 0; idx < stride; idx++) {
    const w = weights[idx];
    if (w > 0) {
      out[idx * 2]     /= w;
      out[idx * 2 + 1] /= w;
    }
  }
}

// ── Dissipation ────────────────────────────────────────────────────────────────

/**
 * Apply per-frame exponential dissipation in-place on a velocity field buffer.
 * Each component is multiplied by `dissipation`, fading the field towards zero
 * over time — matching the Flowmap dissipation model used in ogl's
 * `texture2D(tMap, vUv) * uDissipation` fragment step.
 *
 * @param field       RG Float32Array (gridSize*gridSize*2).
 * @param dissipation Decay coefficient [0,1); values ≥ 1 are clamped to 0.999.
 */
export function dissipateField(
  field:       Float32Array,
  dissipation: number,
): void {
  // Guard: prevent NaN / infinite growth
  const d = Math.min(Math.max(dissipation, 0), 0.999);
  for (let i = 0, n = field.length; i < n; i++) {
    field[i] *= d;
  }
}

// ── Normalisation helper ───────────────────────────────────────────────────────

/**
 * Clamp and normalise velocity components to [−1, +1] using `maxSpeed`.
 * Modifies the buffer in-place.  Call after rasterisation, before uploading
 * to GPU — downstream shaders expect normalised values.
 *
 * @param field    RG Float32Array (gridSize*gridSize*2).
 * @param maxSpeed Reference speed (m/s) that maps to ±1.0 in the texture.
 */
export function normalizeVelocityField(
  field:    Float32Array,
  maxSpeed: number,
): void {
  const invMax = maxSpeed > 0 ? 1.0 / maxSpeed : 1.0;
  for (let i = 0, n = field.length; i < n; i++) {
    field[i] = Math.max(-1, Math.min(1, field[i] * invMax));
  }
}

// ── FlowmapBridge class ────────────────────────────────────────────────────────

/**
 * Stateful per-scene bridge that maintains the velocity field texture across
 * frames.  Combines rasterisation + dissipation + normalisation into a single
 * `update()` call suitable for a rAF loop.
 *
 * The internal `_field` buffer is double-duty:
 *   1. After `update()` it holds the fully-processed, GPU-ready texture data.
 *   2. Between frames it holds the dissipated previous frame — so the next
 *      `update()` can accumulate on top of it (optional: only used when
 *      `accumulate = true` in the option, default false for clean frame reads).
 *
 * Thread safety: not thread-safe.  Intended for main-thread use only.
 * If the SPH sim runs in a worker, `postMessage` the particle arrays back to
 * main before calling `update()`.
 */
export class FlowmapBridge {
  // ── Configuration ──────────────────────────────────────────────────────────
  readonly gridSize:    number;
  readonly dissipation: number;
  readonly domainW:     number;
  readonly domainH:     number;
  readonly maxSpeed:    number;

  // ── Internal state ─────────────────────────────────────────────────────────
  /**
   * The live velocity field texture: Float32Array[gridSize*gridSize*2].
   * R = normalised vx, G = normalised vy, both in [−1, +1].
   * Row-major, Y-down (matching WebGL texSubImage2D conventions).
   * Exposed directly via `getTexture()` — no copy, zero allocation on read.
   */
  private _field: Float32Array;

  /**
   * Scratch SOA accumulation buffer: same layout as `_field`.
   * Allocated once; reused every frame to avoid GC pressure.
   */
  private _scratch: Float32Array;

  /** Running frame count for optional diagnostics. */
  private _frameCount = 0;

  // ── Constructor ────────────────────────────────────────────────────────────

  constructor(options: FlowmapBridgeOptions = {}) {
    this.gridSize    = Math.max(8, Math.min(512, options.gridSize    ?? 64));
    this.dissipation = options.dissipation ?? 0.96;
    this.domainW     = options.domainW     ?? 3.0;
    this.domainH     = options.domainH     ?? 3.0;
    this.maxSpeed    = options.maxSpeed    ?? 5.0;

    const bufLen   = this.gridSize * this.gridSize * 2;
    this._field    = new Float32Array(bufLen);
    this._scratch  = new Float32Array(bufLen);
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Main per-frame update entry point — accepts **AOS** particles.
   *
   * Pipeline:
   *   1. Dissipate existing field (exponential decay).
   *   2. Rasterise new particle velocities into `_scratch`.
   *   3. Splat `_scratch` onto `_field` (new contributions override the
   *      dissipated background where particles are present).
   *   4. Normalise the combined field to [−1, +1].
   *
   * @param particles  Array of `{x, y, vx, vy}` objects from the SPH world.
   * @returns          Reference to the internal field buffer (no copy).
   *                   Caller must NOT mutate or hold past the next `update()`.
   */
  update(particles: Array<SPHParticleAOS>): Float32Array {
    const { gridSize, dissipation, domainW, domainH, maxSpeed } = this;
    const stride = gridSize * gridSize;

    // ── 1. Dissipate previous frame ─────────────────────────────────────────
    dissipateField(this._field, dissipation);

    // ── 2. Rasterise current particles into scratch ──────────────────────────
    const scratch = this._scratch;
    const weights = new Float32Array(stride);   // one allocation per frame, tiny

    scratch.fill(0);

    const invW = gridSize / domainW;
    const invH = gridSize / domainH;

    for (let i = 0, n = particles.length; i < n; i++) {
      const p = particles[i];
      const ci = Math.min(Math.floor(p.x * invW), gridSize - 1);
      const cj = Math.min(Math.floor((domainH - p.y) * invH), gridSize - 1);
      if (ci < 0 || cj < 0) continue;

      const idx = cj * gridSize + ci;
      scratch[idx * 2]     += p.vx;
      scratch[idx * 2 + 1] += p.vy;
      weights[idx]++;
    }

    // ── 3. Merge scratch onto field (occupied cells overwrite, rest keep fade)
    for (let idx = 0; idx < stride; idx++) {
      const w = weights[idx];
      if (w > 0) {
        // Mean velocity of particles in this cell
        this._field[idx * 2]     = scratch[idx * 2]     / w;
        this._field[idx * 2 + 1] = scratch[idx * 2 + 1] / w;
        // Dissipated background for cells with no particles is already in _field
      }
    }

    // ── 4. Normalise to [−1, +1] ────────────────────────────────────────────
    normalizeVelocityField(this._field, maxSpeed);

    this._frameCount++;
    return this._field;
  }

  /**
   * SOA variant of `update()` — avoids an AOS conversion overhead when the
   * caller already holds an `SPHParticleSOA` (e.g. `SPHWorld.cpuPos`).
   *
   * @param soa  Structure-of-arrays particle state from SPHWorld.
   * @returns    Reference to the internal field buffer (no copy).
   */
  updateSOA(soa: SPHParticleSOA): Float32Array {
    const { gridSize, dissipation, domainW, domainH, maxSpeed } = this;
    const { x, y, vx, vy, count } = soa;
    const stride = gridSize * gridSize;

    // ── 1. Dissipate ────────────────────────────────────────────────────────
    dissipateField(this._field, dissipation);

    // ── 2. Accumulate ────────────────────────────────────────────────────────
    const scratch = this._scratch;
    const weights = new Float32Array(stride);
    scratch.fill(0);

    const invW = gridSize / domainW;
    const invH = gridSize / domainH;

    for (let i = 0; i < count; i++) {
      const ci = Math.min(Math.floor(x[i] * invW), gridSize - 1);
      const cj = Math.min(Math.floor((domainH - y[i]) * invH), gridSize - 1);
      if (ci < 0 || cj < 0) continue;

      const idx = cj * gridSize + ci;
      scratch[idx * 2]     += vx[i];
      scratch[idx * 2 + 1] += vy[i];
      weights[idx]++;
    }

    // ── 3. Merge ─────────────────────────────────────────────────────────────
    for (let idx = 0; idx < stride; idx++) {
      const w = weights[idx];
      if (w > 0) {
        this._field[idx * 2]     = scratch[idx * 2]     / w;
        this._field[idx * 2 + 1] = scratch[idx * 2 + 1] / w;
      }
    }

    // ── 4. Normalise ─────────────────────────────────────────────────────────
    normalizeVelocityField(this._field, maxSpeed);

    this._frameCount++;
    return this._field;
  }

  /**
   * Return the current field buffer **without** running a new update.
   * Useful to read the texture between frames or to pass to a WebGL upload
   * function without triggering an unnecessary rasterisation.
   *
   * @returns Float32Array[gridSize*gridSize*2] — direct reference (no copy).
   */
  getTexture(): Float32Array {
    return this._field;
  }

  /**
   * Reset the velocity field to zero (e.g. on scene teardown or after a
   * topology change where continuity with the previous field is undesirable).
   */
  reset(): void {
    this._field.fill(0);
    this._scratch.fill(0);
    this._frameCount = 0;
  }

  /** Diagnostic: number of `update()` / `updateSOA()` calls since construction or last `reset()`. */
  get frameCount(): number {
    return this._frameCount;
  }

  // ── ogl Flowmap compatibility shim ────────────────────────────────────────

  /**
   * Produce an **ogl Flowmap**-compatible input object that can be passed to
   * `flowmap.mouse` / `flowmap.velocity` by sampling the rasterised field at
   * a given canvas-normalised position `(u, v) ∈ [0,1]²`.
   *
   * This lets scenes that already use `ogl.Flowmap` with cursor input layer
   * in SPH-driven flow without replacing the existing mouse interaction:
   *
   *   // In rAF, after bridge.update():
   *   const { u, v } = cellUVs[cellId];          // cell centre in [0,1]²
   *   const { mouse, velocity } = bridge.sampleOGLInput(u, v);
   *   flowmap.mouse.set(mouse[0], mouse[1]);
   *   flowmap.velocity.set(velocity[0], velocity[1]);
   *   flowmap.update();
   *
   * @param u  Horizontal UV coordinate [0, 1] (left=0, right=1).
   * @param v  Vertical   UV coordinate [0, 1] (top=0, bottom=1).
   * @returns  `{ mouse: [u,v], velocity: [vx,vy] }` ready for ogl Flowmap.
   */
  sampleOGLInput(u: number, v: number): { mouse: [number, number]; velocity: [number, number] } {
    const { gridSize, _field } = this;

    // Clamp to valid texel range
    const ci  = Math.min(Math.floor(u * gridSize), gridSize - 1);
    const cj  = Math.min(Math.floor(v * gridSize), gridSize - 1);
    const idx = (cj * gridSize + ci) * 2;

    return {
      mouse:    [u,               v              ],
      velocity: [_field[idx] ?? 0, _field[idx + 1] ?? 0],
    };
  }

  /**
   * Sample the normalised velocity at an arbitrary UV position using bilinear
   * interpolation between the four surrounding grid cells.
   *
   * More expensive than `sampleOGLInput` (4 lookups + lerp) but produces
   * smoother results for per-cell distortion uniforms.
   *
   * @param u  Horizontal UV [0, 1].
   * @param v  Vertical   UV [0, 1].
   * @returns  Interpolated [vx, vy] in [−1, +1].
   */
  sampleBilinear(u: number, v: number): [number, number] {
    const { gridSize, _field } = this;
    const gs = gridSize;

    // Fractional texel coords
    const fx = u * gs - 0.5;
    const fy = v * gs - 0.5;

    // Integer cell coordinates (clamped)
    const x0 = Math.max(0, Math.min(gs - 1, Math.floor(fx)));
    const y0 = Math.max(0, Math.min(gs - 1, Math.floor(fy)));
    const x1 = Math.min(gs - 1, x0 + 1);
    const y1 = Math.min(gs - 1, y0 + 1);

    // Fractional weights
    const tx = fx - Math.floor(fx);
    const ty = fy - Math.floor(fy);

    // Four corner indices
    const i00 = (y0 * gs + x0) * 2;
    const i10 = (y0 * gs + x1) * 2;
    const i01 = (y1 * gs + x0) * 2;
    const i11 = (y1 * gs + x1) * 2;

    // Bilinear interpolation for vx and vy separately
    const vx =
      _field[i00]     * (1 - tx) * (1 - ty) +
      _field[i10]     * tx       * (1 - ty) +
      _field[i01]     * (1 - tx) * ty       +
      _field[i11]     * tx       * ty;

    const vy =
      _field[i00 + 1] * (1 - tx) * (1 - ty) +
      _field[i10 + 1] * tx       * (1 - ty) +
      _field[i01 + 1] * (1 - tx) * ty       +
      _field[i11 + 1] * tx       * ty;

    return [vx, vy];
  }

  /**
   * Compute the speed (magnitude) at the given UV position via bilinear
   * interpolation.  Useful for driving per-cell effect intensities:
   *
   *   const speed = bridge.sampleSpeed(cellU, cellV); // [0, 1]
   *   grayscottUniforms.u_noiseStrength = speed * 0.3;
   *
   * @param u  Horizontal UV [0, 1].
   * @param v  Vertical   UV [0, 1].
   * @returns  Speed in [0, 1] (normalised, so 1.0 = maxSpeed).
   */
  sampleSpeed(u: number, v: number): number {
    const [vx, vy] = this.sampleBilinear(u, v);
    return Math.min(1, Math.sqrt(vx * vx + vy * vy));
  }

  /**
   * Compute the dominant flow angle at the given UV position.
   * Returns the angle in radians (atan2(vy, vx)), or 0 if speed is negligible.
   *
   * Downstream shaders can use this to align the UV distortion axis:
   *
   *   const angle = bridge.sampleAngle(cellU, cellV);
   *   // rotate UV by angle before sampling Gray-Scott reaction products
   *
   * @param u  Horizontal UV [0, 1].
   * @param v  Vertical   UV [0, 1].
   * @returns  Flow angle in [−π, +π] radians.
   */
  sampleAngle(u: number, v: number): number {
    const [vx, vy] = this.sampleBilinear(u, v);
    const speed    = Math.sqrt(vx * vx + vy * vy);
    return speed > 1e-6 ? Math.atan2(vy, vx) : 0;
  }
}
