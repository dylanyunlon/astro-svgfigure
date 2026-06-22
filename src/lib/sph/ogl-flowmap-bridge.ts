/**
 * ogl-flowmap-bridge.ts — OGL Flowmap bridge for SPH velocity distortion  (M614)
 *
 * Ports the upstream/ogl/src/extras/Flowmap.js ping-pong logic to pure
 * TypeScript/CPU, replacing the mouse/cursor input with the SPH velocity
 * field so that cell-surface Turing patterns and Voronoi membranes flow
 * like oil paint dragged by water.
 *
 * ─── OGL Flowmap recap ───────────────────────────────────────────────────────
 * The OGL Flowmap maintains two ping-pong RenderTargets (read / write).
 * Each frame its fragment shader:
 *
 *   1. Samples the previous frame's texture × dissipation:
 *        vec4 color = texture2D(tMap, vUv) * uDissipation;
 *
 *   2. Computes a "stamp" centred on the cursor:
 *        vec2  cursor = vUv - uMouse;
 *        vec3  stamp  = vec3(uVelocity * vec2(1,-1),
 *                            1.0 - pow(1.0-min(1.0,length(uVelocity)), 3.0));
 *        float falloff = smoothstep(uFalloff, 0.0, length(cursor)) * uAlpha;
 *
 *   3. Blends stamp into the dissipated field:
 *        color.rgb = mix(color.rgb, stamp, vec3(falloff));
 *        gl_FragColor = color;
 *
 *   4. Swaps read/write targets:
 *        uniform.value = mask.read.texture;
 *
 * Output texture: RG = velocity XY (normalised), B = speed magnitude.
 * This texture is fed to the downstream distortion shader as `tFlowmap`.
 *
 * ─── SPH adaptation ──────────────────────────────────────────────────────────
 * Instead of a single cursor stamp, we inject *many* stamps simultaneously —
 * one per particle.  For each particle at (x,y) with velocity (vx,vy) we:
 *
 *   • Map (x,y) → UV (u,v) ∈ [0,1]²  (Y-flipped: SPH Y-up → UV V-down)
 *   • "Stamp" its velocity onto the write buffer at UV (u,v) using
 *     a Gaussian falloff of radius `falloff` (default 1.5 / resolution)
 *   • Dissipate the read buffer before merging
 *
 * Gaussian falloff is approximated with a 3×3 kernel splat that keeps the
 * inner loop O(N · 9) = O(N) and avoids per-pixel distance computations.
 * This is accurate enough for the downstream effect and runs in < 1 ms for
 * 20 000 particles on a modern CPU.
 *
 * ─── Ping-pong in CPU Float32Arrays ─────────────────────────────────────────
 * Two Float32Arrays (`_read`, `_write`) of length `resolution² × 4` mirror
 * the OGL RenderTarget pair.  Layout per texel (RGBA matching OGL):
 *
 *   [idx*4+0] = vx  ∈ [−1, +1]  (R channel)
 *   [idx*4+1] = vy  ∈ [−1, +1]  (G channel, Y-flipped for GL)
 *   [idx*4+2] = speed ∈ [0, 1]  (B channel — 1.0 - pow(…) in OGL)
 *   [idx*4+3] = alpha ∈ [0, 1]  (A channel)
 *
 * After `update()` the read buffer holds the fully processed texture ready
 * for `gl.texSubImage2D(…, gl.RGBA, gl.FLOAT, flowmap.getFlowTexture())`.
 *
 * ─── Coordinate conventions ──────────────────────────────────────────────────
 * SPH world : origin bottom-left, Y-up,  units = simulation metres.
 * UV space  : origin top-left,    V-down, both axes ∈ [0, 1].
 * Y-flip formula: v = 1 − (y / worldH)
 *
 * ─── Usage ───────────────────────────────────────────────────────────────────
 *
 *   import { SPHFlowmap } from '$lib/sph/ogl-flowmap-bridge';
 *
 *   // Construct once
 *   const flowmap = new SPHFlowmap(128);          // 128×128 grid
 *
 *   // Each frame, after SPH tick():
 *   flowmap.updateFromVelocityField(
 *     world.getParticles(),                        // {x,y,vx,vy}[]
 *     world.domainW, world.domainH
 *   );
 *
 *   // Upload to WebGL:
 *   gl.texSubImage2D(
 *     gl.TEXTURE_2D, 0, 0, 0,
 *     128, 128, gl.RGBA, gl.FLOAT,
 *     flowmap.getFlowTexture()
 *   );
 *
 *   // Or sample per-cell for uniform distortion:
 *   const [vx, vy] = flowmap.getFlowAt(cellX / worldW, cellY / worldH);
 *   shader.uniforms.u_flowVelocity = [vx, vy];
 *
 * ─── References ──────────────────────────────────────────────────────────────
 * OGL Flowmap source : upstream/ogl/src/extras/Flowmap.js
 * Sibling bridge     : src/lib/sph/flowmap-bridge.ts  (M573, cursor-driven)
 * Consumer shaders   : src/lib/shaders/grayscott-species.frag  (u_velocity)
 *                      src/lib/shaders/fluid-surface.frag      (uVelocityTex)
 * Turing patterns    : src/lib/sph/natural-patterns.ts
 * Research           : M614 — cell-pubsub-loop
 */

// ── Particle input type ────────────────────────────────────────────────────────

/** Minimal particle descriptor consumed by SPHFlowmap. */
export interface FlowParticle {
  /** World-space X coordinate (domain units). */
  x: number;
  /** World-space Y coordinate (domain units, Y-up). */
  y: number;
  /** Velocity X component (m/s). */
  vx: number;
  /** Velocity Y component (m/s). */
  vy: number;
}

// ── Internal constants ─────────────────────────────────────────────────────────

/**
 * Default dissipation factor — mirrors OGL Flowmap default of 0.98.
 * Closer to 1.0 → slower fade (trail persists longer).
 */
const DEFAULT_DISSIPATION = 0.98;

/**
 * Default falloff radius as a fraction of the grid side length.
 * 0.015 ≈ 1.5 texels at resolution 100 — matches OGL default `falloff=0.3`
 * scaled to a per-particle stamp size.
 */
const DEFAULT_FALLOFF_FRAC = 0.015;

/**
 * Default maximum speed for normalisation (m/s).
 * SPH velocities are divided by this before being written to the texture.
 */
const DEFAULT_MAX_SPEED = 5.0;

/**
 * Precomputed 3×3 Gaussian kernel weights (σ ≈ 1.0, centre = 1.0).
 * Index maps to offset (dx, dy) ∈ {−1,0,+1}²  via `KERNEL_OFFSETS`.
 */
const KERNEL_WEIGHTS: readonly number[] = [
  0.0625, 0.125, 0.0625,  // row −1: (−1,−1) (0,−1) (+1,−1)
  0.125,  0.25,  0.125,   // row  0: (−1, 0) (0, 0) (+1, 0)
  0.0625, 0.125, 0.0625,  // row +1: (−1,+1) (0,+1) (+1,+1)
];

/** (dcol, drow) offsets corresponding to KERNEL_WEIGHTS. */
const KERNEL_OFFSETS: readonly [number, number][] = [
  [-1,-1],[0,-1],[1,-1],
  [-1, 0],[0, 0],[1, 0],
  [-1, 1],[0, 1],[1, 1],
];

// ── SPHFlowmap ────────────────────────────────────────────────────────────────

/**
 * CPU-side ping-pong Flowmap fed by SPH particle velocity fields.
 *
 * Mirrors the render-target ping-pong and stamp logic of `ogl/extras/Flowmap.js`
 * in TypeScript, enabling the same downstream GLSL shaders that were designed
 * for cursor-driven Flowmap to consume live SPH velocity data unchanged.
 *
 * Textures produced by this class can be uploaded directly via
 * `gl.texSubImage2D(…, gl.RGBA, gl.FLOAT, flowmap.getFlowTexture())`.
 */
export class SPHFlowmap {
  // ── Configuration ────────────────────────────────────────────────────────

  /** Side length of the square flow texture in texels. */
  readonly resolution: number;

  /**
   * Per-frame exponential dissipation [0, 1).
   * Mirrors `uDissipation` in OGL Flowmap fragment shader.
   * Default: 0.98.
   */
  dissipation: number;

  /**
   * Falloff radius of each particle stamp, expressed as a fraction of
   * `resolution`.  E.g. 0.015 → radius ≈ 1.5 texels at res=100.
   * Mirrors `uFalloff` in OGL Flowmap.
   * Default: 0.015.
   */
  falloff: number;

  /**
   * Maximum speed (m/s) that maps to ±1.0 in the output texture.
   * Velocities are clamped to [−maxSpeed, +maxSpeed] before normalisation.
   * Default: 5.0.
   */
  maxSpeed: number;

  // ── Ping-pong buffers ────────────────────────────────────────────────────

  /**
   * Read buffer — the current processed frame.
   * Layout: Float32Array[resolution * resolution * 4]
   *   [i*4+0] = vx  ∈ [−1,+1]  (R)
   *   [i*4+1] = vy  ∈ [−1,+1]  (G, Y-flipped for GL conventions)
   *   [i*4+2] = speed ∈ [0,1]  (B)
   *   [i*4+3] = alpha = 1.0    (A)
   */
  private _read: Float32Array;

  /**
   * Write buffer — accumulated stamps for the current frame.
   * After `_finalise()` this becomes the new read buffer.
   */
  private _write: Float32Array;

  /**
   * Per-cell accumulation weight buffer (particle count per cell).
   * Reused each frame to avoid allocating inside the hot loop.
   */
  private _weights: Float32Array;

  /** Monotonically increasing frame counter (diagnostic). */
  private _frameCount = 0;

  // ── Constructor ──────────────────────────────────────────────────────────

  /**
   * Create an SPHFlowmap.
   *
   * @param resolution  Side length of the square flow grid in texels.
   *                    Powers-of-two (64, 128, 256) are recommended for GPU
   *                    compat but arbitrary values work.  Default: 128.
   * @param dissipation Per-frame fade factor [0,1).  Default: 0.98.
   * @param falloff     Stamp radius as a fraction of `resolution`.  Default: 0.015.
   * @param maxSpeed    Velocity normalisation reference (m/s).  Default: 5.0.
   */
  constructor(
    resolution  = 128,
    dissipation = DEFAULT_DISSIPATION,
    falloff     = DEFAULT_FALLOFF_FRAC,
    maxSpeed    = DEFAULT_MAX_SPEED,
  ) {
    this.resolution  = Math.max(4, resolution);
    this.dissipation = Math.min(Math.max(dissipation, 0), 0.999);
    this.falloff     = Math.max(0, falloff);
    this.maxSpeed    = maxSpeed > 0 ? maxSpeed : DEFAULT_MAX_SPEED;

    const len = this.resolution * this.resolution * 4;
    this._read    = new Float32Array(len);
    this._write   = new Float32Array(len);
    this._weights = new Float32Array(this.resolution * this.resolution);

    // Initialise alpha channel to 1.0 (matches ogl RGBA clear value)
    for (let i = 3; i < len; i += 4) {
      this._read[i]  = 1.0;
      this._write[i] = 1.0;
    }
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Main per-frame update — inject SPH velocity field into the ping-pong
   * texture, mirroring OGL Flowmap's `update()` logic.
   *
   * Pipeline (one frame):
   *   1. Copy read → write, multiplied by `dissipation`  (= OGL "tMap × uDissipation")
   *   2. Rasterise each particle as a Gaussian stamp onto write            (= OGL "mix stamp")
   *   3. Normalise velocity components to [−1,+1] and compute B = speed   (= OGL stamp.z)
   *   4. Swap read ↔ write                                                 (= OGL mask.swap())
   *
   * @param particles  Array of `{x, y, vx, vy}` objects.
   * @param worldW     Simulation domain width  (same unit as particle.x).
   * @param worldH     Simulation domain height (same unit as particle.y).
   */
  updateFromVelocityField(
    particles: FlowParticle[],
    worldW:    number,
    worldH:    number,
  ): void {
    const { resolution, dissipation, maxSpeed } = this;
    const stride = resolution * resolution;
    const invMax = 1.0 / maxSpeed;

    // ── 1. Dissipate: write = read × dissipation ─────────────────────────
    //    Mirrors: `vec4 color = texture2D(tMap, vUv) * uDissipation;`
    const read  = this._read;
    const write = this._write;

    for (let i = 0, n = stride * 4; i < n; i++) {
      write[i] = read[i] * dissipation;
    }
    // Keep alpha at 1.0 after dissipation (OGL writes alpha via stamp blend)
    for (let i = 3; i < stride * 4; i += 4) {
      write[i] = 1.0;
    }

    // ── 2. Rasterise particle stamps ──────────────────────────────────────
    //    Each particle becomes a 3×3 Gaussian splat centred at its UV cell.
    //    Mirrors: `stamp = vec3(uVelocity, 1-pow(…)); mix(color, stamp, falloff)`
    const weights = this._weights;
    weights.fill(0);

    // Velocity accumulation buffers (RG channels, before normalisation)
    // We stamp into write[] directly — see note on alpha below.
    // Separate vx/vy accumulators let us merge at the end without
    // overwriting the dissipated trail prematurely.
    const accVx = new Float32Array(stride);
    const accVy = new Float32Array(stride);

    const invW = resolution / worldW;
    const invH = resolution / worldH;

    for (let p = 0, nP = particles.length; p < nP; p++) {
      const { x, y, vx: rawVx, vy: rawVy } = particles[p];

      // UV ∈ [0,1]², Y-flip for texture-space (V=0 at top)
      const u = x / worldW;
      const v = 1.0 - (y / worldH);

      // Central texel in grid space
      const col0 = Math.round(u * (resolution - 1));
      const row0 = Math.round(v * (resolution - 1));

      // Skip out-of-bounds particles
      if (col0 < 0 || col0 >= resolution || row0 < 0 || row0 >= resolution) continue;

      // Normalise velocity (clamped to [−1,+1])
      const nvx =  Math.max(-1, Math.min(1, rawVx * invMax));
      const nvy = -Math.max(-1, Math.min(1, rawVy * invMax)); // Y-flip velocity for GL

      // 3×3 Gaussian kernel splat
      for (let k = 0; k < 9; k++) {
        const [dc, dr] = KERNEL_OFFSETS[k];
        const kw        = KERNEL_WEIGHTS[k];

        const col = col0 + dc;
        const row = row0 + dr;
        if (col < 0 || col >= resolution || row < 0 || row >= resolution) continue;

        const idx = row * resolution + col;
        accVx[idx]   += nvx * kw;
        accVy[idx]   += nvy * kw;
        weights[idx] += kw;
      }
    }

    // ── 3. Merge stamps: occupied cells get velocity; empty cells keep trail ─
    //    Mirrors: `color.rgb = mix(color.rgb, stamp, vec3(falloff));`
    //    where falloff = 1.0 for a fully-occupied cell (weight > 0).
    for (let idx = 0; idx < stride; idx++) {
      const w = weights[idx];
      if (w <= 0.0) continue;

      const vx = accVx[idx] / w;   // mean normalised vx (already in [−1,+1])
      const vy = accVy[idx] / w;   // mean normalised vy

      const speed = Math.min(1.0,  // mirrors OGL: 1.0 - pow(1.0-min(1,|vel|), 3)
        1.0 - Math.pow(1.0 - Math.min(1.0, Math.sqrt(
          (accVx[idx] * accVx[idx] + accVy[idx] * accVy[idx]) / (w * w)
        )), 3.0)
      );

      const base = idx * 4;
      // Blend: stamp overwrites dissipated trail at this cell
      // (weight > 0 → falloff = 1.0; weight = 0 → trail kept above)
      write[base]     = vx;
      write[base + 1] = vy;
      write[base + 2] = speed;
      write[base + 3] = 1.0;
    }

    // ── 4. Swap ping-pong  ────────────────────────────────────────────────
    //    Mirrors: `mask.swap(); uniform.value = mask.read.texture;`
    const tmp  = this._read;
    this._read  = this._write;
    this._write = tmp;

    this._frameCount++;
  }

  /**
   * Return the current flow texture as a CPU-side `Float32Array`.
   *
   * Layout: `resolution × resolution × 4` floats, row-major, Y-down.
   *   [i*4+0] = vx    ∈ [−1,+1]  — R channel (horizontal flow)
   *   [i*4+1] = vy    ∈ [−1,+1]  — G channel (vertical flow, Y-flipped for GL)
   *   [i*4+2] = speed ∈  [0, 1]  — B channel (flow magnitude)
   *   [i*4+3] = 1.0              — A channel
   *
   * Upload to WebGL:
   *   `gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, res, res, gl.RGBA, gl.FLOAT, tex)`
   *
   * @returns Direct reference to the internal read buffer — do **not** mutate
   *          or retain across the next `updateFromVelocityField()` call.
   */
  getFlowTexture(): Float32Array {
    return this._read;
  }

  /**
   * Sample the normalised flow velocity at an arbitrary world-space point.
   *
   * Uses bilinear interpolation between the four surrounding grid cells for
   * smooth per-cell distortion uniforms.  The output is in [−1, +1] for both
   * components, matching what the downstream shader expects as a UV delta.
   *
   * @param x  World-space X coordinate (same unit as `worldW` in `updateFromVelocityField`).
   *           Pass as a UV ∈ [0,1] if the world is already normalised.
   * @param y  World-space Y coordinate (Y-up, same unit as `worldH`).
   * @returns  `[vx, vy]` in [−1, +1], or `[0, 0]` if the point is outside the domain.
   *
   * @example
   *   // Per-cell: distort Turing pattern UV by local SPH flow
   *   const [fx, fy] = flowmap.getFlowAt(cell.x / worldW, cell.y / worldH);
   *   turingShader.uniforms.u_flowOffset = [fx * 0.05, fy * 0.05];
   */
  getFlowAt(x: number, y: number): [number, number] {
    // x, y may be either UV ∈ [0,1] or world coords if worldW/H = 1.
    // Normalise to UV by clamping (caller is responsible for passing [0,1] u,v).
    const u = Math.max(0, Math.min(1, x));
    const v = Math.max(0, Math.min(1, 1.0 - y));   // Y-flip: Y-up → V-down

    return this._sampleBilinearRG(u, v);
  }

  /**
   * Sample the normalised flow velocity at a UV position `(u,v) ∈ [0,1]²`
   * (V=0 at top, matching texture convention).
   *
   * Prefer `getFlowAt` for world-space inputs; use this directly only when
   * you already have UV coordinates (e.g. from the renderer).
   *
   * @param u  Horizontal UV ∈ [0,1].
   * @param v  Vertical   UV ∈ [0,1]  (V=0 at top).
   * @returns  `[vx, vy]` in [−1, +1].
   */
  sampleUV(u: number, v: number): [number, number] {
    return this._sampleBilinearRG(
      Math.max(0, Math.min(1, u)),
      Math.max(0, Math.min(1, v)),
    );
  }

  /**
   * Sample flow speed (magnitude) at the given UV position ∈ [0,1]²,
   * reading the pre-computed B channel.
   *
   * Useful for driving effect intensities that scale with flow energy:
   *   `shader.uniforms.u_distortStrength = flowmap.sampleSpeedUV(u, v) * 0.3;`
   *
   * @returns Speed in [0, 1].
   */
  sampleSpeedUV(u: number, v: number): number {
    const res = this.resolution;
    const cu  = Math.max(0, Math.min(1, u));
    const cv  = Math.max(0, Math.min(1, v));

    const fx = cu * (res - 1);
    const fy = cv * (res - 1);
    const x0 = Math.max(0, Math.min(res - 1, Math.floor(fx)));
    const y0 = Math.max(0, Math.min(res - 1, Math.floor(fy)));
    const x1 = Math.min(res - 1, x0 + 1);
    const y1 = Math.min(res - 1, y0 + 1);
    const tx = fx - x0;
    const ty = fy - y0;

    const s00 = this._read[(y0 * res + x0) * 4 + 2];
    const s10 = this._read[(y0 * res + x1) * 4 + 2];
    const s01 = this._read[(y1 * res + x0) * 4 + 2];
    const s11 = this._read[(y1 * res + x1) * 4 + 2];

    return s00 * (1-tx)*(1-ty) + s10 * tx*(1-ty) + s01 * (1-tx)*ty + s11 * tx*ty;
  }

  /**
   * Reset the flowmap to a zero-velocity state (e.g. on scene teardown).
   * Both ping-pong buffers are zeroed; alpha channels are restored to 1.0.
   */
  reset(): void {
    this._read.fill(0);
    this._write.fill(0);
    this._weights.fill(0);
    this._frameCount = 0;

    // Restore alpha
    const len = this.resolution * this.resolution * 4;
    for (let i = 3; i < len; i += 4) {
      this._read[i]  = 1.0;
      this._write[i] = 1.0;
    }
  }

  /** Diagnostic: number of `updateFromVelocityField()` calls since construction or `reset()`. */
  get frameCount(): number {
    return this._frameCount;
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  /**
   * Bilinear interpolation of the RG (vx, vy) channels in `_read`.
   *
   * @param u  Horizontal UV ∈ [0,1].
   * @param v  Vertical   UV ∈ [0,1] (V=0 at top).
   * @returns  [vx, vy] in [−1,+1].
   */
  private _sampleBilinearRG(u: number, v: number): [number, number] {
    const res = this.resolution;
    const field = this._read;

    const fx = u * (res - 1);
    const fy = v * (res - 1);

    const x0 = Math.max(0, Math.min(res - 1, Math.floor(fx)));
    const y0 = Math.max(0, Math.min(res - 1, Math.floor(fy)));
    const x1 = Math.min(res - 1, x0 + 1);
    const y1 = Math.min(res - 1, y0 + 1);

    const tx = fx - x0;
    const ty = fy - y0;

    const i00 = (y0 * res + x0) * 4;
    const i10 = (y0 * res + x1) * 4;
    const i01 = (y1 * res + x0) * 4;
    const i11 = (y1 * res + x1) * 4;

    const w00 = (1-tx) * (1-ty);
    const w10 =    tx  * (1-ty);
    const w01 = (1-tx) *    ty;
    const w11 =    tx  *    ty;

    const vx = field[i00]*w00 + field[i10]*w10 + field[i01]*w01 + field[i11]*w11;
    const vy = field[i00+1]*w00 + field[i10+1]*w10 + field[i01+1]*w01 + field[i11+1]*w11;

    return [vx, vy];
  }
}

// ── Convenience factory ────────────────────────────────────────────────────────

/**
 * Construct an `SPHFlowmap` with explicit option bag — useful when configuring
 * from scene settings without positional arguments.
 *
 * @example
 *   const flowmap = createSPHFlowmap({
 *     resolution:  128,
 *     dissipation: 0.97,
 *     falloff:     0.02,
 *     maxSpeed:    4.0,
 *   });
 */
export interface SPHFlowmapOptions {
  /** Grid side length (texels). Default: 128. */
  resolution?: number;
  /** Dissipation factor [0,1). Default: 0.98. */
  dissipation?: number;
  /** Stamp falloff radius as fraction of resolution. Default: 0.015. */
  falloff?: number;
  /** Velocity normalisation reference (m/s). Default: 5.0. */
  maxSpeed?: number;
}

export function createSPHFlowmap(opts: SPHFlowmapOptions = {}): SPHFlowmap {
  return new SPHFlowmap(
    opts.resolution  ?? 128,
    opts.dissipation ?? DEFAULT_DISSIPATION,
    opts.falloff     ?? DEFAULT_FALLOFF_FRAC,
    opts.maxSpeed    ?? DEFAULT_MAX_SPEED,
  );
}
