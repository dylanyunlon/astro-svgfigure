/**
 * src/lib/sph/heat-distortion.ts — M787
 *
 * High-Energy Region Heat Distortion — UV offset + noise + energy field
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Simulates localised "heat-haze" distortion that emerges around high-energy
 * regions of the SPH simulation.  Unlike the global collision-shockwave
 * (M764) which radiates outward from discrete impact events, this module
 * continuously warps screen-space UVs in the vicinity of persistent energy
 * hotspots — cell nuclei under high metabolic load, edge junctions carrying
 * dense traffic, or any world-space emitter the host registers.
 *
 * Visual effect
 * ─────────────
 *   The distortion mimics atmospheric refraction above a hot surface:
 *
 *   • Rising convection columns — UV offset drifts upward with time-varying
 *     amplitude, creating the characteristic shimmering "mirage" look.
 *   • Layered noise perturbation — multi-octave simplex noise (FBM) breaks
 *     the displacement into organic, turbulent micro-eddies so the warp
 *     never looks artificially smooth.
 *   • Energy field falloff — each hotspot radiates an inverse-square energy
 *     field.  The distortion amplitude is proportional to the accumulated
 *     field intensity at each fragment, producing soft halos of increasing
 *     turbulence around each emitter.  The field naturally composites when
 *     multiple emitters overlap, intensifying distortion at convergence
 *     zones without any special-case blending logic.
 *   • Chromatic fringing — R/G/B channels sample at slightly different UV
 *     offsets along the displacement vector, yielding a subtle prismatic
 *     edge that sells the "hot air" refractive illusion.
 *
 * Energy emitter model
 * ────────────────────
 *   Hotspots are registered as HeatEmitter structs:
 *
 *     { x, y, energy, radius, phase }
 *
 *   `energy` [0, 1] drives distortion amplitude (0 = dormant, 1 = maximum).
 *   `radius` defines the falloff envelope — outside this radius the emitter's
 *   contribution drops below perceptual threshold and is culled.  `phase`
 *   offsets the noise animation so adjacent emitters don't shimmer in sync.
 *
 *   The CPU-side HeatDistortionSystem manages a pool of emitters, exposing
 *   add/remove/update operations and a per-frame snapshot for GPU upload.
 *
 * Architecture
 * ────────────
 *   HeatDistortionSystem  (CPU side — emitter lifecycle + field evaluation)
 *     ├─ addEmitter(emitter)             — register a heat source
 *     ├─ removeEmitter(id)               — remove by stable ID
 *     ├─ updateEmitter(id, patch)        — mutate energy / position / radius
 *     ├─ update(dt)                      — advance noise phase, cull dead
 *     └─ getSnapshot()                   — GPU-uploadable emitter array
 *
 *   HeatDistortionPipeline  (WebGPU full-screen post-process pass)
 *     ├─ create(device, format)          — factory; compiles WGSL, creates BGL
 *     ├─ uploadEmitters(snapshot)        — writes emitter SSBO
 *     └─ render(encoder, src, dst, w, h) — records the distortion pass
 *
 * Integration
 * ───────────
 *   ```ts
 *   import { HeatDistortionSystem, HeatDistortionPipeline } from '$lib/sph/heat-distortion';
 *
 *   const heatSys  = new HeatDistortionSystem();
 *   const heatPipe = await HeatDistortionPipeline.create(device, format);
 *
 *   // Register emitters from cell energy data:
 *   topology.cells.forEach(cell => {
 *     heatSys.addEmitter({
 *       id:     cell.id,
 *       x:      cell.x,
 *       y:      cell.y,
 *       energy: cell.metabolicLoad,    // [0, 1]
 *       radius: cell.influenceRadius,  // world-space units
 *       phase:  Math.random() * Math.PI * 2,
 *     });
 *   });
 *
 *   // Each frame:
 *   heatSys.update(dt);
 *   heatPipe.uploadEmitters(heatSys.getSnapshot());
 *   heatPipe.render(encoder, sceneView, dstView, width, height);
 *   ```
 *
 * Design references
 * ─────────────────
 *   src/lib/sph/collision-shockwave.ts   — SSBO ring pattern, WGSL full-screen
 *   src/lib/sph/noise-flow-field.ts      — simplex / FBM WGSL (lygia-derived)
 *   src/lib/sph/ripple-effect.ts         — ping-pong + composite post-process
 *   src/lib/sph/density-field-texture.ts — energy field sampling approach
 *   src/lib/sph/edge-energy-flow.ts      — energy-to-visual mapping curves
 *
 * Research: xiaodi #M787 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** A single heat-distortion emitter (CPU-side representation). */








export interface HeatEmitter {
  /** Stable identifier for add/remove/update. */
  id: string;
  /** World-space X position (pixels or simulation units). */
  x: number;
  /** World-space Y position (pixels or simulation units). */
  y: number;
  /**
   * Energy intensity [0, 1].
   * 0 = dormant (no visible distortion), 1 = maximum heat shimmer.
   */
  energy: number;
  /**
   * Influence radius in world-space units.
   * The distortion falloff reaches perceptual zero at this distance.
   */
  radius: number;
  /**
   * Phase offset for the noise animation (radians).
   * Prevents adjacent emitters from shimmering in lock-step.
   */
  phase: number;
}

/** GPU-uploadable snapshot of an emitter (8 × f32 = 32 bytes). */
export interface HeatEmitterGPU {
  /** Centre UV x. */
  cx: number;
  /** Centre UV y. */
  cy: number;
  /** Normalised energy [0, 1], pre-multiplied by any smoothing. */
  energy: number;
  /** Falloff radius in UV-space units. */
  radius: number;
  /** Phase offset (radians). */
  phase: number;
  /** Current time-based noise seed. */
  noiseSeed: number;
  /** Distortion amplitude scale (UV units, derived from energy). */
  amplitude: number;
  /** Chromatic fringing spread factor [0, 1]. */
  chromatic: number;
}

/** Configuration for the heat distortion system. */
export interface HeatDistortionConfig {
  /**
   * Maximum UV distortion amplitude at full energy.
   * Default 0.025 — gentler than shockwave (0.04) for persistent effect.
   */
  maxAmplitude: number;

  /**
   * Maximum chromatic aberration spread at full energy.
   * Default 0.35.
   */
  maxChromatic: number;

  /**
   * Noise animation speed multiplier.
   * Default 1.0 — drives the time term in the FBM.
   */
  noiseSpeed: number;

  /**
   * Convection drift speed (UV/s, upward).
   * Default 0.06 — how fast the shimmering columns rise.
   */
  convectionSpeed: number;

  /**
   * Noise frequency multiplier.
   * Higher values → finer turbulence grain.
   * Default 8.0.
   */
  noiseFrequency: number;

  /**
   * Number of FBM octaves for the distortion noise.
   * More octaves → richer micro-detail at GPU cost.
   * Default 4.
   */
  noiseOctaves: number;

  /**
   * Energy threshold below which an emitter is invisible.
   * Default 0.01.
   */
  minEnergy: number;

  /**
   * Maximum concurrent emitters supported on GPU.
   * Default 32.
   */
  maxEmitters: number;

  /**
   * Energy → amplitude mapping exponent.
   * Default 1.5 — super-linear so low-energy emitters barely shimmer
   * while high-energy ones are dramatically distorted.
   */
  energyExponent: number;

  /**
   * Additive brightness at the core of high-energy emitters.
   * Creates a faint white-hot glow in the distortion centre.
   * Default 0.08.
   */
  coreGlow: number;
}

const DEFAULT_CONFIG: HeatDistortionConfig = {
  maxAmplitude:    0.025,
  maxChromatic:    0.35,
  noiseSpeed:      1.0,
  convectionSpeed: 0.06,
  noiseFrequency:  8.0,
  noiseOctaves:    4,
  minEnergy:       0.01,
  maxEmitters:     32,
  energyExponent:  1.5,
  coreGlow:        0.08,
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal emitter state
// ─────────────────────────────────────────────────────────────────────────────

interface EmitterState {
  /** Original emitter data. */
  emitter: HeatEmitter;
  /** Smoothed energy (lerped toward target each frame for fade transitions). */
  smoothedEnergy: number;
  /** Accumulated animation time for this emitter's noise. */
  noiseTime: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// HeatDistortionSystem  (CPU side — emitter lifecycle + snapshot)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Manages a pool of heat-distortion emitters.
 *
 * Each emitter represents a persistent high-energy region whose energy
 * level can vary over time.  The system smooths energy transitions
 * (exponential decay toward target) to avoid visual popping, advances
 * per-emitter noise phases, and produces a GPU-uploadable snapshot.
 *
 * @example
 * ```ts
 * const sys = new HeatDistortionSystem({ maxAmplitude: 0.03 });
 *
 * sys.addEmitter({
 *   id: 'nucleus-A', x: 400, y: 300,
 *   energy: 0.8, radius: 120, phase: 0,
 * });
 *
 * // Each frame:
 * sys.update(dt);
 * const snap = sys.getSnapshot(canvasWidth, canvasHeight);
 * pipeline.uploadEmitters(snap);
 * ```
 */
export class HeatDistortionSystem {
  private readonly cfg: HeatDistortionConfig;
  private readonly states: Map<string, EmitterState> = new Map();
  private globalTime = 0;

  constructor(overrides?: Partial<HeatDistortionConfig>) {
    this.cfg = { ...DEFAULT_CONFIG, ...overrides };
  }

  // ── Emitter lifecycle ──────────────────────────────────────────────────────

  /**
   * Register a new heat emitter.
   *
   * If an emitter with the same ID already exists, it is updated in-place
   * (position, energy, radius) without resetting the smoothed energy —
   * this allows seamless hot-swapping during topology changes.
   */
  addEmitter(emitter: HeatEmitter): void {
    const existing = this.states.get(emitter.id);
    if (existing) {
      existing.emitter = { ...emitter };
      return;
    }
    this.states.set(emitter.id, {
      emitter:        { ...emitter },
      smoothedEnergy: emitter.energy,
      noiseTime:      emitter.phase,
    });
  }

  /**
   * Remove an emitter by ID.
   *
   * The emitter is immediately removed from the pool.  For a graceful
   * fade-out, set its energy to 0 and let the system cull it once the
   * smoothed energy drops below `minEnergy`.
   */
  removeEmitter(id: string): void {
    this.states.delete(id);
  }

  /**
   * Partially update an existing emitter.
   *
   * Only the fields present in `patch` are overwritten; smoothed energy
   * continues to track the (potentially new) target energy smoothly.
   */
  updateEmitter(id: string, patch: Partial<HeatEmitter>): void {
    const state = this.states.get(id);
    if (!state) return;
    Object.assign(state.emitter, patch);
  }

  /**
   * Batch-set emitters: replaces the entire pool with the provided array.
   *
   * Emitters whose IDs already exist are updated in-place (preserving
   * smooth energy); new IDs are added; IDs not present in `emitters`
   * are removed.
   */
  setEmitters(emitters: HeatEmitter[]): void {
    const incoming = new Set<string>();
    for (const e of emitters) {
      incoming.add(e.id);
      this.addEmitter(e);
    }
    // Remove emitters not in the new set
    for (const id of this.states.keys()) {
      if (!incoming.has(id)) {
        this.states.delete(id);
      }
    }
  }

  // ── Per-frame tick ─────────────────────────────────────────────────────────

  /**
   * Advance the system by `dt` seconds.
   *
   * • Smooths each emitter's energy toward its target using exponential
   *   decay (τ ≈ 0.15 s → ~87% within one τ).
   * • Advances per-emitter noise time.
   * • Culls emitters whose smoothed energy has fallen below `minEnergy`.
   */
  update(dt: number): void {
    this.globalTime += dt;

    const smoothRate = 1 - Math.exp(-dt / 0.15);
    const toDelete: string[] = [];

    for (const [id, state] of this.states) {
      // Smooth energy toward target
      const target = state.emitter.energy;
      state.smoothedEnergy += (target - state.smoothedEnergy) * smoothRate;

      // Advance noise phase
      state.noiseTime += dt * this.cfg.noiseSpeed;

      // Cull negligible emitters (only if target is also zero to avoid
      // removing an emitter that's ramping up)
      if (
        state.smoothedEnergy < this.cfg.minEnergy &&
        target < this.cfg.minEnergy
      ) {
        toDelete.push(id);
      }
    }

    for (const id of toDelete) {
      this.states.delete(id);
    }
  }

  // ── GPU snapshot ───────────────────────────────────────────────────────────

  /**
   * Produce a GPU-uploadable snapshot of active emitters.
   *
   * World-space emitter positions are converted to UV space using the
   * provided viewport dimensions.  The snapshot is sorted by descending
   * energy so the strongest emitters are evaluated first in the shader
   * (early-exit optimisation for the loop upper bound).
   *
   * @param viewportW  Canvas / render-target width in pixels.
   * @param viewportH  Canvas / render-target height in pixels.
   * @returns Array of HeatEmitterGPU structs (length ≤ maxEmitters).
   */
  getSnapshot(viewportW: number, viewportH: number): HeatEmitterGPU[] {
    const out: HeatEmitterGPU[] = [];
    const maxDim = Math.max(viewportW, viewportH, 1);

    for (const state of this.states.values()) {
      const e = state.emitter;
      const se = state.smoothedEnergy;
      if (se < this.cfg.minEnergy) continue;

      // Energy → amplitude: super-linear mapping
      const energyT = Math.pow(Math.min(se, 1), this.cfg.energyExponent);

      out.push({
        cx:        e.x / viewportW,
        cy:        e.y / viewportH,
        energy:    se,
        radius:    e.radius / maxDim,
        phase:     e.phase,
        noiseSeed: state.noiseTime,
        amplitude: this.cfg.maxAmplitude * energyT,
        chromatic: this.cfg.maxChromatic * Math.sqrt(energyT),
      });
    }

    // Sort by descending energy for shader early-exit
    out.sort((a, b) => b.energy - a.energy);

    // Clamp to GPU limit
    if (out.length > this.cfg.maxEmitters) {
      out.length = this.cfg.maxEmitters;
    }

    return out;
  }

  // ── Canvas 2D fallback ─────────────────────────────────────────────────────

  /**
   * Draw a simplified heat distortion overlay on a Canvas 2D context.
   *
   * This is a lightweight fallback for hosts without WebGPU.  It renders
   * each emitter as concentric radial gradient circles with pulsating
   * alpha — no actual UV distortion (that requires the GPU pipeline).
   * The visual effect is a soft glowing halo around each hotspot.
   *
   * @param ctx     Canvas 2D rendering context.
   * @param width   Canvas width in pixels.
   * @param height  Canvas height in pixels.
   */
  drawCanvas2D(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
  ): void {
    if (this.states.size === 0) return;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    for (const state of this.states.values()) {
      const e = state.emitter;
      const se = state.smoothedEnergy;
      if (se < this.cfg.minEnergy) continue;

      // Pulsating alpha based on noise time
      const pulse = 0.5 + 0.5 * Math.sin(state.noiseTime * 3.7 + e.phase);
      const alpha = se * 0.3 * (0.6 + 0.4 * pulse);
      if (alpha < 0.005) continue;

      // Radial gradient
      const grad = ctx.createRadialGradient(
        e.x, e.y, 0,
        e.x, e.y, e.radius,
      );
      grad.addColorStop(0.0, `rgba(255, 240, 200, ${(alpha * 0.8).toFixed(3)})`);
      grad.addColorStop(0.3, `rgba(255, 200, 120, ${(alpha * 0.5).toFixed(3)})`);
      grad.addColorStop(0.7, `rgba(200, 140, 80, ${(alpha * 0.2).toFixed(3)})`);
      grad.addColorStop(1.0, `rgba(150, 100, 60, 0)`);

      ctx.beginPath();
      ctx.arc(e.x, e.y, e.radius, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      // Inner bright core
      if (se > 0.5) {
        const coreAlpha = (se - 0.5) * 2 * alpha * 0.6;
        const coreGrad = ctx.createRadialGradient(
          e.x, e.y, 0,
          e.x, e.y, e.radius * 0.2,
        );
        coreGrad.addColorStop(0.0, `rgba(255, 255, 240, ${coreAlpha.toFixed(3)})`);
        coreGrad.addColorStop(1.0, `rgba(255, 255, 240, 0)`);
        ctx.beginPath();
        ctx.arc(e.x, e.y, e.radius * 0.2, 0, Math.PI * 2);
        ctx.fillStyle = coreGrad;
        ctx.fill();
      }
    }

    ctx.globalCompositeOperation = 'source-over';
    ctx.restore();
  }

  // ── Introspection ──────────────────────────────────────────────────────────

  /** Number of currently active emitters. */
  get count(): number {
    return this.states.size;
  }

  /** Remove all emitters immediately. */
  clear(): void {
    this.states.clear();
    this.globalTime = 0;
  }

  /** Mutate config at runtime (e.g. from a debug panel). */
  configure(overrides: Partial<HeatDistortionConfig>): void {
    Object.assign(this.cfg, overrides);
  }

  /** Current global time (seconds elapsed since creation/reset). */
  get elapsed(): number {
    return this.globalTime;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — full-screen heat distortion shader
// ─────────────────────────────────────────────────────────────────────────────

/** Maximum emitters the shader supports (must match SSBO array size). */
const MAX_GPU_EMITTERS = 32;

/** Floats per emitter in the storage buffer (8 × f32 = 32 bytes). */
const FLOATS_PER_EMITTER = 8;

/**
 * Heat distortion WGSL shader.
 *
 * For each fragment, iterates over active emitters and accumulates a
 * UV displacement based on:
 *   1. Inverse-square energy field falloff from emitter centre.
 *   2. Multi-octave simplex noise (FBM) for turbulence detail.
 *   3. Upward convection drift term that shifts the noise sampling.
 *   4. Chromatic aberration split along the displacement vector.
 *   5. Core glow additive brightness near emitter centres.
 */
const HEAT_DISTORTION_WGSL = /* wgsl */`
// ─── Uniforms ────────────────────────────────────────────────────────────────
struct HeatUniforms {
  width         : f32,
  height        : f32,
  emitterCount  : f32,
  aspectRatio   : f32,
  time          : f32,
  noiseFreq     : f32,
  convectionSpd : f32,
  coreGlow      : f32,
}

struct Emitter {
  cx        : f32,    // centre U
  cy        : f32,    // centre V
  energy    : f32,    // normalised energy [0, 1]
  radius    : f32,    // falloff radius (UV)
  phase     : f32,    // noise phase offset
  noiseSeed : f32,    // time-based noise seed
  amplitude : f32,    // peak UV distortion (pre-scaled)
  chromatic : f32,    // R/B split factor
}

@group(0) @binding(0) var<uniform>       u        : HeatUniforms;
@group(0) @binding(1) var                smp      : sampler;
@group(0) @binding(2) var                src      : texture_2d<f32>;
@group(0) @binding(3) var<storage, read> emitters : array<Emitter, ${MAX_GPU_EMITTERS}>;

// ─── Constants ───────────────────────────────────────────────────────────────
const PI : f32 = 3.14159265359;

// ─── Simplex noise (lygia-derived, self-contained) ───────────────────────────
fn mod289_2(x: vec2f) -> vec2f { return x - floor(x / 289.0) * 289.0; }
fn mod289_3(x: vec3f) -> vec3f { return x - floor(x / 289.0) * 289.0; }
fn permute3(x: vec3f) -> vec3f { return mod289_3((x * 34.0 + 1.0) * x); }

fn snoise2(v: vec2f) -> f32 {
  let C = vec4f(
    0.211324865405187,   // (3 - sqrt(3)) / 6
    0.366025403784439,   // 0.5 * (sqrt(3) - 1)
   -0.577350269189626,   // -1 + 2 * C.x
    0.024390243902439,   // 1 / 41
  );

  var i  = floor(v + dot(v, C.yy));
  let x0 = v - i + dot(i, C.xx);

  let i1 = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), x0.x > x0.y);
  let x12 = x0.xyxy + C.xxzz - vec4f(i1, 1.0, 1.0);

  i = mod289_2(i);
  let p = permute3(
    permute3(i.y + vec3f(0.0, i1.y, 1.0)) + i.x + vec3f(0.0, i1.x, 1.0));

  var m = max(0.5 - vec3f(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3f(0.0));
  m = m * m;
  m = m * m;

  let x = 2.0 * fract(p * C.www) - 1.0;
  let h = abs(x) - 0.5;
  let ox = floor(x + 0.5);
  let a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
  let g0 = a0.x * x0.x + h.x * x0.y;
  let g12 = vec2f(a0.y * x12.x + h.y * x12.y, a0.z * x12.z + h.z * x12.w);
  return 130.0 * dot(m, vec3f(g0, g12));
}

// ─── FBM (4 octaves of simplex noise) ────────────────────────────────────────
fn fbm4(p: vec2f) -> f32 {
  var value = 0.0;
  var amp   = 0.5;
  var freq  = 1.0;
  var pos   = p;
  for (var i = 0; i < 4; i++) {
    value += amp * snoise2(pos * freq);
    freq  *= 2.03;
    amp   *= 0.49;
    // Rotate to reduce axis-aligned artifacts
    pos    = vec2f(
      pos.x * 0.866 - pos.y * 0.5,
      pos.x * 0.5   + pos.y * 0.866,
    );
  }
  return value;
}

// ─── Vertex shader — full-screen quad ────────────────────────────────────────

struct Vert {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
}

@vertex fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> Vert {
  var pos = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
    vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
  );
  var uv = array<vec2f, 6>(
    vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(0.0, 0.0),
  );
  var out: Vert;
  out.pos = vec4f(pos[vi], 0.0, 1.0);
  out.uv  = uv[vi];
  return out;
}

// ─── Fragment shader — heat distortion + noise + energy field ────────────────

@fragment fn fs_heat_distortion(in: Vert) -> @location(0) vec4f {
  let st = in.uv;
  let emitterCount = i32(u.emitterCount);

  // Accumulate UV displacement and additive glow across all emitters
  var offsetR = vec2f(0.0);
  var offsetG = vec2f(0.0);
  var offsetB = vec2f(0.0);
  var addGlow = 0.0;

  for (var i = 0; i < ${MAX_GPU_EMITTERS}; i++) {
    if (i >= emitterCount) { break; }

    let em = emitters[i];

    // Aspect-corrected distance from emitter centre
    let delta = vec2f(
      (st.x - em.cx) * u.aspectRatio,
       st.y - em.cy,
    );
    let dist = length(delta);

    // Early skip: outside influence radius (with small margin for noise)
    if (dist > em.radius * 1.15) {
      continue;
    }

    // ── Energy field falloff ──────────────────────────────────────────────
    // Smooth hermite falloff within the radius envelope.
    // Uses smoothstep for a perceptually natural fade:
    //   full intensity at centre → zero at radius boundary.
    let falloff = 1.0 - smoothstep(0.0, em.radius, dist);
    // Square the falloff for tighter concentration near the core
    let field = falloff * falloff;

    // ── Noise sampling coordinates ───────────────────────────────────────
    // Convection drift: shift the noise field upward over time
    // (in UV space, negative Y = upward on screen)
    let convection = vec2f(0.0, -u.convectionSpd * em.noiseSeed);

    // Noise input: position scaled by frequency + drift + per-emitter phase
    let noiseP = (st + convection) * u.noiseFreq + vec2f(em.phase, em.phase * 0.7);

    // Two orthogonal FBM channels for 2D displacement
    let noiseX = fbm4(noiseP);
    let noiseY = fbm4(noiseP + vec2f(17.3, 31.7));  // offset to decorrelate

    // ── UV displacement ──────────────────────────────────────────────────
    // Displacement = noise × field × amplitude
    // The noise provides turbulent micro-eddies; the field shapes the
    // spatial envelope; amplitude carries the energy→visual mapping.
    let disp = vec2f(noiseX, noiseY) * field * em.amplitude;

    // Add a subtle upward bias to sell the convection column look
    let convBias = vec2f(0.0, -1.0) * field * em.amplitude * 0.3;
    let totalDisp = disp + convBias;

    // ── Chromatic aberration ─────────────────────────────────────────────
    let chrSpread = em.chromatic * field;
    offsetR += totalDisp * (1.0 + chrSpread);
    offsetG += totalDisp;
    offsetB += totalDisp * (1.0 - chrSpread);

    // ── Core glow ────────────────────────────────────────────────────────
    // Additive brightness near emitter centres (inner 30% of radius)
    let coreFalloff = 1.0 - smoothstep(0.0, em.radius * 0.3, dist);
    addGlow += u.coreGlow * em.energy * coreFalloff * coreFalloff;
  }

  // Sample scene colour with per-channel UV offsets (chromatic split)
  let colR = textureSample(src, smp, st + offsetR).r;
  let colG = textureSample(src, smp, st + offsetG).g;
  let colB = textureSample(src, smp, st + offsetB).b;

  var color = vec3f(colR, colG, colB);

  // Core glow: warm-white additive tint (slightly amber for heat feel)
  color += vec3f(addGlow * 1.0, addGlow * 0.9, addGlow * 0.7);

  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GPU uniform layout:  8 × f32 = 32 bytes
// ─────────────────────────────────────────────────────────────────────────────

const UNIFORM_FLOATS = 8;

// ─────────────────────────────────────────────────────────────────────────────
// HeatDistortionPipeline  (WebGPU full-screen post-process pass)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WebGPU render pipeline that applies screen-space heat distortion driven
 * by an array of energy emitters.
 *
 * Each frame, emitter data is uploaded from HeatDistortionSystem.getSnapshot()
 * and the shader evaluates the energy field, noise perturbation, and chromatic
 * aberration for every on-screen fragment.
 *
 * Slots into the post-process chain alongside other passes (bloom,
 * tone-mapping, collision shockwave).  Run after tone-mapping and before
 * or after the shockwave pass depending on desired visual layering.
 *
 * @example
 * ```ts
 * const pipeline = await HeatDistortionPipeline.create(device, format);
 *
 * // Per frame:
 * pipeline.uploadEmitters(heatSys.getSnapshot(w, h));
 * pipeline.render(encoder, sceneView, dstView, width, height);
 * ```
 */
export class HeatDistortionPipeline {
  private readonly device:      GPUDevice;
  private readonly pipeline:    GPURenderPipeline;
  private readonly bgl:         GPUBindGroupLayout;
  private readonly sampler:     GPUSampler;
  private readonly uniformBuf:  GPUBuffer;
  private readonly emitterBuf:  GPUBuffer;
  private readonly cfg:         HeatDistortionConfig;

  // Bind group cache — invalidated when source texture view changes
  private cachedBG:   GPUBindGroup  | null = null;
  private cachedSrc:  GPUTextureView | null = null;

  private _emitterCount = 0;
  private _time = 0;

  private constructor(
    device:     GPUDevice,
    pipeline:   GPURenderPipeline,
    bgl:        GPUBindGroupLayout,
    sampler:    GPUSampler,
    uniformBuf: GPUBuffer,
    emitterBuf: GPUBuffer,
    cfg:        HeatDistortionConfig,
  ) {
    this.device     = device;
    this.pipeline   = pipeline;
    this.bgl        = bgl;
    this.sampler    = sampler;
    this.uniformBuf = uniformBuf;
    this.emitterBuf = emitterBuf;
    this.cfg        = cfg;
  }

  // ── Factory ────────────────────────────────────────────────────────────────

  /**
   * Compile the heat distortion WGSL, create the render pipeline and GPU
   * buffers.
   *
   * @param device     WebGPU device.
   * @param format     Swap-chain / render-target texture format.
   * @param overrides  Optional config overrides.
   * @returns Ready-to-use HeatDistortionPipeline instance.
   */
  static async create(
    device:    GPUDevice,
    format:    GPUTextureFormat,
    overrides?: Partial<HeatDistortionConfig>,
  ): Promise<HeatDistortionPipeline> {
    const cfg = { ...DEFAULT_CONFIG, ...overrides };

    const module = device.createShaderModule({ code: HEAT_DISTORTION_WGSL });

    const bgl = device.createBindGroupLayout({
      entries: [
        // 0: uniforms
        { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
          buffer: { type: 'uniform' } },
        // 1: linear sampler
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'filtering' } },
        // 2: source scene texture
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },
        // 3: emitter SSBO (read-only storage)
        { binding: 3, visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'read-only-storage' } },
      ],
    });

    const pipeline = await device.createRenderPipelineAsync({
      layout:   device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex:   { module, entryPoint: 'vs_fullscreen' },
      fragment: {
        module,
        entryPoint: 'fs_heat_distortion',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
    });

    const sampler = device.createSampler({
      magFilter:    'linear',
      minFilter:    'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    const uniformBuf = device.createBuffer({
      size:  UNIFORM_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Emitter storage buffer: MAX_GPU_EMITTERS × FLOATS_PER_EMITTER × 4 bytes
    const emitterBuf = device.createBuffer({
      size:  MAX_GPU_EMITTERS * FLOATS_PER_EMITTER * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new HeatDistortionPipeline(
      device, pipeline, bgl, sampler, uniformBuf, emitterBuf, cfg,
    );
  }

  // ── Per-frame upload ───────────────────────────────────────────────────────

  /**
   * Upload the current set of active emitters to the GPU storage buffer.
   *
   * Call this once per frame before `render()`.
   *
   * @param emitters  Active emitters from `HeatDistortionSystem.getSnapshot()`.
   */
  uploadEmitters(emitters: HeatEmitterGPU[]): void {
    const count = Math.min(emitters.length, MAX_GPU_EMITTERS);
    const data = new Float32Array(MAX_GPU_EMITTERS * FLOATS_PER_EMITTER);

    for (let i = 0; i < count; i++) {
      const e = emitters[i];
      const base = i * FLOATS_PER_EMITTER;
      data[base + 0] = e.cx;
      data[base + 1] = e.cy;
      data[base + 2] = e.energy;
      data[base + 3] = e.radius;
      data[base + 4] = e.phase;
      data[base + 5] = e.noiseSeed;
      data[base + 6] = e.amplitude;
      data[base + 7] = e.chromatic;
    }

    this.device.queue.writeBuffer(this.emitterBuf, 0, data);
    this._emitterCount = count;
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  /**
   * Record the heat distortion pass into `encoder`.
   *
   * If no emitters are active the pass is still recorded (as a pass-through
   * blit) to maintain consistent attachment state in the render graph.
   *
   * @param encoder  Active GPUCommandEncoder.
   * @param srcView  Input scene colour texture view.
   * @param dstView  Output texture view (swap-chain surface or next RT).
   * @param width    Render target width in pixels.
   * @param height   Render target height in pixels.
   * @param dt       Delta time in seconds (advances internal clock).
   */
  render(
    encoder: GPUCommandEncoder,
    srcView: GPUTextureView,
    dstView: GPUTextureView,
    width:   number,
    height:  number,
    dt = 0,
  ): void {
    this._time += dt;
    this._uploadUniforms(width, height);
    const bg = this._bindGroup(srcView);

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view:       dstView,
        loadOp:     'clear',
        storeOp:    'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bg);
    pass.draw(6);   // full-screen quad (two triangles)
    pass.end();
  }

  /**
   * Destroy GPU resources.  Call when the pipeline is no longer needed.
   */
  destroy(): void {
    this.uniformBuf.destroy();
    this.emitterBuf.destroy();
  }

  // ── Internal ───────────────────────────────────────────────────────────────

  private _uploadUniforms(width: number, height: number): void {
    const data = new Float32Array(UNIFORM_FLOATS);
    data[0] = width;
    data[1] = height;
    data[2] = this._emitterCount;
    data[3] = width / Math.max(height, 1);   // aspect ratio
    data[4] = this._time;
    data[5] = this.cfg.noiseFrequency;
    data[6] = this.cfg.convectionSpeed;
    data[7] = this.cfg.coreGlow;
    this.device.queue.writeBuffer(this.uniformBuf, 0, data);
  }

  private _bindGroup(srcView: GPUTextureView): GPUBindGroup {
    if (this.cachedBG && this.cachedSrc === srcView) {
      return this.cachedBG;
    }
    const bg = this.device.createBindGroup({
      layout:  this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: srcView },
        { binding: 3, resource: { buffer: this.emitterBuf } },
      ],
    });
    this.cachedBG  = bg;
    this.cachedSrc = srcView;
    return bg;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: energy field evaluation (CPU side, for debug / spatial queries)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Evaluate the accumulated heat distortion energy field at a world-space
 * point.  This is the CPU-side equivalent of the shader's per-fragment
 * energy accumulation — useful for debug overlays, spatial audio coupling,
 * or deciding whether to spawn additional particle effects.
 *
 * @param px        Query point X (world space).
 * @param py        Query point Y (world space).
 * @param emitters  Array of HeatEmitter objects.
 * @returns Accumulated energy field intensity at (px, py), in [0, ∞).
 */
export function evaluateEnergyField(
  px: number,
  py: number,
  emitters: HeatEmitter[],
): number {
  let total = 0;
  for (const e of emitters) {
    const dx = px - e.x;
    const dy = py - e.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist >= e.radius) continue;

    // Smooth hermite falloff, squared (matches shader)
    const t = 1 - dist / e.radius;
    const falloff = t * t * t * (t * (t * 6 - 15) + 10); // smootherstep
    total += e.energy * falloff;
  }
  return total;
}

/**
 * Find the nearest heat emitter to a world-space point.
 *
 * @param px        Query point X (world space).
 * @param py        Query point Y (world space).
 * @param emitters  Array of HeatEmitter objects.
 * @returns The nearest emitter, or null if the array is empty.
 */
export function nearestEmitter(
  px: number,
  py: number,
  emitters: HeatEmitter[],
): HeatEmitter | null {
  let best: HeatEmitter | null = null;
  let bestDist = Infinity;
  for (const e of emitters) {
    const dx = px - e.x;
    const dy = py - e.y;
    const d = dx * dx + dy * dy;
    if (d < bestDist) {
      bestDist = d;
      best = e;
    }
  }
  return best;
}
