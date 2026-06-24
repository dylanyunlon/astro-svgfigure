/**
 * src/lib/sph/collision-shockwave.ts — M764
 *
 * Collision Shockwave — screen-space UV distortion expanding ring
 * ─────────────────────────────────────────────────────────────────────────────
 * When two rigid bodies collide, the system emits an expanding distortion
 * ring centred at the contact point.  The shockwave displaces UV coordinates
 * in a narrow radial band around the ring front, producing a refractive
 * "heat-haze" look.  Chromatic aberration splits the R/G/B sample offsets
 * along the radial direction for a prismatic edge, and the entire effect's
 * amplitude is proportional to the collision impulse so gentle taps barely
 * shimmer while heavy impacts produce a dramatic full-screen ripple.
 *
 * Architecture
 * ────────────
 *   CollisionShockwaveSystem  (CPU side, manages active ring instances)
 *     ├─ emit(contact, impulse)       — spawn a new expanding ring
 *     ├─ update(dt)                   — advance all rings, cull dead ones
 *     └─ getActiveRings()             — snapshot for the GPU pass
 *
 *   CollisionShockwavePipeline  (WebGPU full-screen post-process pass)
 *     ├─ create(device, format)       — factory; compiles WGSL, creates BGL
 *     ├─ uploadRings(rings)           — writes ring SSBO from CPU snapshot
 *     └─ render(encoder, src, dst, w, h)  — records the distortion pass
 *
 * The pipeline reads the scene colour texture, offsets UVs per-ring, and
 * writes the distorted result to the destination.  It slots into the
 * post-process chain after tone-mapping and before final blit, or can be
 * run standalone.
 *
 * UV distortion model
 * ───────────────────
 *   For each fragment, for each active ring:
 *     1. Compute radial distance `d` from the ring centre (in UV space).
 *     2. Ring front sits at `radius`; the distortion band has width `thickness`.
 *     3. Inside the band:  offset = radialDir × amplitude × bandProfile(d)
 *        where bandProfile = sin(π · bandT) — strongest at the ring front,
 *        zero at the leading/trailing edges.
 *     4. Chromatic aberration: R samples at offset × (1 + chromaticSpread),
 *        G at offset × 1.0, B at offset × (1 − chromaticSpread).
 *     5. Amplitude = baseAmplitude × impulseT × (1 − travel²)
 *        so the ring fades as it expands.
 *
 * Impulse → visual mapping
 * ────────────────────────
 *   impulse  →  t = clamp(impulse × impulseScale, 0, 1)
 *   amplitude = maxAmplitude × t                (stronger hit = bigger warp)
 *   chromatic = maxChromatic × √t               (prismatic split)
 *   speed     = baseSpeed × (0.7 + 0.3t)        (heavier hits expand faster)
 *   thickness = baseThickness × (0.8 + 0.2t)    (wider band at high impulse)
 *
 * Integration with existing collision pipeline
 * ─────────────────────────────────────────────
 *   const shockwave = new CollisionShockwaveSystem();
 *   const pipeline  = await CollisionShockwavePipeline.create(device, format);
 *
 *   dispatcher.onCollisionEnter((evt) => {
 *     if (!evt.contact) return;
 *     const impulse = evt.contact.depth * 120;
 *     shockwave.emit(evt.contact, impulse);
 *   });
 *
 *   // Each frame:
 *   shockwave.update(dt);
 *   pipeline.uploadRings(shockwave.getActiveRings());
 *   pipeline.render(encoder, sceneView, dstView, width, height);
 *
 * Design references
 * ─────────────────
 *   src/lib/sph/post-process.ts            — full-screen blit pattern, WGSL
 *   src/lib/sph/ripple-effect.ts           — wave propagation + composite
 *   src/lib/sph/collision-fx-system.ts     — impulse→visual curve, lygia RNG
 *   src/lib/sph/contact-sparks.ts          — impulseScale convention
 *   src/lib/sph/collision/CollisionEvents.ts — CollisionContactInfo types
 */


import type {
} from './collision/CollisionEvents';
import type { CollisionEventDispatcher } from './collision/CollisionEvents';

// [orphan3]   CollisionContactInfo,
// [orphan3]   CollisionEvent,

// ─────────────────────────────────────────────────────────────────────────────
// Lygia random port (shared convention with contact-sparks / collision-fx)
// ─────────────────────────────────────────────────────────────────────────────

const SCALE_X = 0.1031;
const SCALE_Y = 0.1030;
const SCALE_Z = 0.0973;

function fract(x: number): number {
  return x - Math.floor(x);
}

function lygiaRandom(p: number): number {
  let x = fract(p * SCALE_X);
  x *= x + 33.33;
  x *= x + x;
  return fract(x);
}

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface Vec2 {
  x: number;
  y: number;
}

/** Configuration for the collision shockwave system. */
export interface ShockwaveConfig {
  /**
   * Impulse scale: maps raw impulse magnitude to normalised [0, 1].
   * Default 0.008 — tune to match your velocity / depth units.
   */
  impulseScale: number;

  /**
   * Maximum UV distortion amplitude at full impulse.
   * Expressed in UV-space units (0.04 ≈ 4% of screen width).
   * Default 0.04.
   */
  maxAmplitude: number;

  /**
   * Maximum chromatic aberration spread at full impulse.
   * Fraction of distortion offset applied as R/B split.
   * Default 0.6 — R samples at offset × 1.6, B at offset × 0.4.
   */
  maxChromatic: number;

  /**
   * Ring expansion speed in UV-space units per second.
   * Default 0.8 (crosses half the screen in ~0.625 s).
   */
  baseSpeed: number;

  /**
   * Radial band thickness in UV-space units.
   * The distortion is strongest at the ring front and tapers
   * over this thickness behind the wavefront.
   * Default 0.12.
   */
  baseThickness: number;

  /**
   * Maximum ring lifetime in seconds.
   * Rings are culled once their radius exceeds the visible area
   * or lifetime expires, whichever comes first.
   * Default 1.2.
   */
  maxLifetime: number;

  /**
   * Maximum number of concurrent shockwave rings.
   * Oldest ring is culled when this limit is reached.
   * Default 8.
   */
  maxRings: number;

  /**
   * Minimum impulse threshold (normalised).
   * Collisions weaker than this are ignored.
   * Default 0.02.
   */
  minThreshold: number;

  /**
   * Brightness boost at ring front (additive white flash).
   * Impulse-proportional: final = brightnessBoost × impulseT × bandProfile.
   * Default 0.15.
   */
  brightnessBoost: number;
}

const DEFAULT_CONFIG: ShockwaveConfig = {
  impulseScale:    0.008,
  maxAmplitude:    0.04,
  maxChromatic:    0.6,
  baseSpeed:       0.8,
  baseThickness:   0.12,
  maxLifetime:     1.2,
  maxRings:        8,
  minThreshold:    0.02,
  brightnessBoost: 0.15,
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal ring representation
// ─────────────────────────────────────────────────────────────────────────────

/** A single expanding shockwave ring (CPU-side state). */
interface ShockwaveRing {
  /** Centre in UV space [0, 1]². */
  cx: number;
  cy: number;
  /** Current ring radius in UV-space units. */
  radius: number;
  /** Expansion speed (UV/s). */
  speed: number;
  /** Band thickness (UV units). */
  thickness: number;
  /** Peak distortion amplitude (UV units). */
  amplitude: number;
  /** Chromatic aberration spread factor [0, 1]. */
  chromatic: number;
  /** Additive brightness at ring front. */
  brightness: number;
  /** Remaining lifetime (seconds). */
  life: number;
  /** Total lifetime for fade computation. */
  maxLife: number;
  /** Impulse-normalised intensity [0, 1]. */
  impulseT: number;
  /** Per-ring random seed for subtle phase offsets. */
  seed: number;
}

/**
 * GPU-uploadable ring snapshot.
 * Packed as 8 floats per ring for the WGSL storage buffer.
 */
export interface ShockwaveRingGPU {
  /** Centre UV. */
  cx: number;
  cy: number;
  /** Current radius (UV). */
  radius: number;
  /** Band thickness (UV). */
  thickness: number;
  /** Distortion amplitude (UV), already decay-scaled. */
  amplitude: number;
  /** Chromatic spread factor. */
  chromatic: number;
  /** Additive brightness. */
  brightness: number;
  /** Padding / seed for noise in shader. */
  seed: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// CollisionShockwaveSystem  (CPU side — ring lifecycle)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Manages expanding UV-distortion shockwave rings spawned at collision
 * contact points.  Impulse magnitude drives amplitude, chromatic
 * aberration, expansion speed, and ring thickness.
 *
 * @example
 * ```ts
 * const sw = new CollisionShockwaveSystem({ impulseScale: 0.01 });
 *
 * dispatcher.onCollisionEnter((evt) => {
 *   if (!evt.contact) return;
 *   sw.emit(evt.contact, evt.contact.depth * 120);
 * });
 *
 * // Animation loop:
 * sw.update(dt);
 * const rings = sw.getActiveRings();
 * // → feed rings to CollisionShockwavePipeline.uploadRings()
 * ```
 */
export class CollisionShockwaveSystem {
  private rings: ShockwaveRing[] = [];
  private cfg: ShockwaveConfig;
  private _frame = 0;

  constructor(config: Partial<ShockwaveConfig> = {}) {
    this.cfg = { ...DEFAULT_CONFIG, ...config };
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Emit a shockwave ring at a collision contact point.
   *
   * The contact position is mapped to UV space using `viewport` dimensions.
   * If no viewport is provided the raw world-space position is normalised
   * by a default 1920×1080 assumption — callers should supply the actual
   * canvas dimensions for accurate placement.
   *
   * @param contact   CollisionContactInfo from the narrow phase.
   * @param impulse   Raw impulse magnitude.
   * @param viewport  Canvas / render-target dimensions for world→UV mapping.
   */
  emit(
    contact: CollisionContactInfo,
    impulse: number,
    viewport: { width: number; height: number } = { width: 1920, height: 1080 },
  ): void {
    const cfg = this.cfg;

    // Map impulse → normalised [0, 1]
    const t = Math.min(impulse * cfg.impulseScale, 1.0);
    if (t < cfg.minThreshold) return;

    // Cull oldest ring if at capacity
    if (this.rings.length >= cfg.maxRings) {
      // Find ring with least remaining life and remove it
      let minLifeIdx = 0;
      for (let i = 1; i < this.rings.length; i++) {
        if (this.rings[i].life < this.rings[minLifeIdx].life) {
          minLifeIdx = i;
        }
      }
      this.rings.splice(minLifeIdx, 1);
    }

    // Contact midpoint → UV space
    const mx = (contact.pointA.x + contact.pointB.x) * 0.5;
    const my = (contact.pointA.y + contact.pointB.y) * 0.5;
    const uvX = mx / viewport.width;
    const uvY = my / viewport.height;

    // Impulse-driven parameters
    const amplitude  = cfg.maxAmplitude * t;
    const chromatic  = cfg.maxChromatic * Math.sqrt(t);
    const speed      = cfg.baseSpeed * (0.7 + 0.3 * t);
    const thickness  = cfg.baseThickness * (0.8 + 0.2 * t);
    const brightness = cfg.brightnessBoost * t;
    const lifetime   = cfg.maxLifetime * (0.6 + 0.4 * t);

    const seed = lygiaRandom(this._frame * 7.31 + impulse * 0.13);

    this.rings.push({
      cx: uvX,
      cy: uvY,
      radius: 0,
      speed,
      thickness,
      amplitude,
      chromatic,
      brightness,
      life: lifetime,
      maxLife: lifetime,
      impulseT: t,
      seed,
    });

    this._frame++;
  }

  /**
   * Convenience: emit from a CollisionEvent directly.
   */
  emitFromEvent(
    event: CollisionEvent,
    impulse: number,
    viewport?: { width: number; height: number },
  ): void {
    if (!event.contact) return;
    this.emit(event.contact, impulse, viewport);
  }

  /**
   * Convenience: emit using a screen-space UV point directly.
   *
   * @param uvCenter  Centre of the shockwave in UV [0, 1]².
   * @param impulse   Raw impulse magnitude.
   */
  emitAtUV(uvCenter: Vec2, impulse: number): void {
    const cfg = this.cfg;
    const t = Math.min(impulse * cfg.impulseScale, 1.0);
    if (t < cfg.minThreshold) return;

    if (this.rings.length >= cfg.maxRings) {
      let minLifeIdx = 0;
      for (let i = 1; i < this.rings.length; i++) {
        if (this.rings[i].life < this.rings[minLifeIdx].life) {
          minLifeIdx = i;
        }
      }
      this.rings.splice(minLifeIdx, 1);
    }

    const amplitude  = cfg.maxAmplitude * t;
    const chromatic  = cfg.maxChromatic * Math.sqrt(t);
    const speed      = cfg.baseSpeed * (0.7 + 0.3 * t);
    const thickness  = cfg.baseThickness * (0.8 + 0.2 * t);
    const brightness = cfg.brightnessBoost * t;
    const lifetime   = cfg.maxLifetime * (0.6 + 0.4 * t);
    const seed = lygiaRandom(this._frame * 7.31 + impulse * 0.13);

    this.rings.push({
      cx: uvCenter.x,
      cy: uvCenter.y,
      radius: 0,
      speed,
      thickness,
      amplitude,
      chromatic,
      brightness,
      life: lifetime,
      maxLife: lifetime,
      impulseT: t,
      seed,
    });

    this._frame++;
  }

  /**
   * Advance all shockwave rings by `dt` seconds.
   * Expands radii and culls expired or off-screen rings.
   */
  update(dt: number): void {
    for (let i = this.rings.length - 1; i >= 0; i--) {
      const r = this.rings[i];

      // Expand
      r.radius += r.speed * dt;

      // Age
      r.life -= dt;

      // Cull: dead or fully off-screen (radius - thickness > √2 ≈ 1.42)
      if (r.life <= 0 || (r.radius - r.thickness) > 1.5) {
        const last = this.rings.length - 1;
        if (i !== last) {
          this.rings[i] = this.rings[last];
        }
        this.rings.pop();
      }
    }
  }

  /**
   * Snapshot active rings into GPU-uploadable format.
   *
   * Amplitude is pre-multiplied by the time-based decay curve
   * (1 − travel²) so the shader only needs the final amplitude.
   *
   * @returns Array of ShockwaveRingGPU structs, sorted by spawn order.
   */
  getActiveRings(): ShockwaveRingGPU[] {
    const out: ShockwaveRingGPU[] = [];
    for (const r of this.rings) {
      const travel = Math.max(0, 1 - r.life / r.maxLife);
      // Decay: (1 − travel²) — strong near spawn, fading outward
      const decay = Math.max(0, 1 - travel * travel);

      out.push({
        cx:         r.cx,
        cy:         r.cy,
        radius:     r.radius,
        thickness:  r.thickness,
        amplitude:  r.amplitude * decay,
        chromatic:  r.chromatic * decay,
        brightness: r.brightness * decay,
        seed:       r.seed,
      });
    }
    return out;
  }

  // ── Canvas 2D fallback ─────────────────────────────────────────────────────

  /**
   * Draw a simplified shockwave ring overlay on a Canvas 2D context.
   *
   * This is a lightweight fallback for hosts without WebGPU.  It draws
   * each ring as a stroked arc with gaussian-edge alpha — no actual UV
   * distortion or chromatic aberration (those require the GPU pipeline).
   * The visual effect is a bright expanding ring that fades outward.
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
    if (this.rings.length === 0) return;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    for (const r of this.rings) {
      const travel = Math.max(0, 1 - r.life / r.maxLife);
      const decay  = Math.max(0, 1 - travel * travel);
      if (decay < 0.01) continue;

      // Convert UV to pixel coords
      const px = r.cx * width;
      const py = r.cy * height;
      const radiusPx = r.radius * Math.max(width, height);
      const thickPx  = r.thickness * Math.max(width, height);

      // Ring alpha: sin(π·bandT) peak at the wavefront
      const alpha = decay * r.impulseT * 0.6;
      if (alpha < 0.005) continue;

      // Outer glow ring
      ctx.beginPath();
      ctx.arc(px, py, Math.max(0, radiusPx), 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(200, 220, 255, ${alpha.toFixed(3)})`;
      ctx.lineWidth = thickPx * 0.8;
      ctx.stroke();

      // Inner bright edge (narrower, brighter)
      ctx.beginPath();
      ctx.arc(px, py, Math.max(0, radiusPx), 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255, 255, 255, ${(alpha * 0.7).toFixed(3)})`;
      ctx.lineWidth = thickPx * 0.25;
      ctx.stroke();
    }

    ctx.globalCompositeOperation = 'source-over';
    ctx.restore();
  }

  // ── Wiring helper ──────────────────────────────────────────────────────────

  /**
   * Subscribe this system to a CollisionEventDispatcher's enter events.
   *
   * @param dispatcher       The world's CollisionEventDispatcher instance.
   * @param depthMultiplier  Scales contact.depth to impulse.  Default 120.
   * @param viewport         Canvas dimensions for world→UV mapping.
   * @returns Unsubscribe function.
   */
  subscribe(
    dispatcher: CollisionEventDispatcher,
    depthMultiplier = 120,
    viewport?: { width: number; height: number },
  ): () => void {
    return dispatcher.onCollisionEnter((evt: CollisionEvent) => {
      if (!evt.contact) return;
      const impulse = evt.contact.depth * depthMultiplier;
      this.emit(evt.contact, impulse, viewport);
    });
  }

  // ── Introspection ──────────────────────────────────────────────────────────

  /** Number of currently active shockwave rings. */
  get count(): number {
    return this.rings.length;
  }

  /** Remove all active rings immediately. */
  clear(): void {
    this.rings.length = 0;
  }

  /** Mutate config at runtime (e.g. from a debug panel). */
  configure(overrides: Partial<ShockwaveConfig>): void {
    Object.assign(this.cfg, overrides);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — full-screen shockwave distortion shader
// ─────────────────────────────────────────────────────────────────────────────

/** Maximum rings the shader supports (must match SSBO array size). */
const MAX_GPU_RINGS = 16;

/** Floats per ring in the storage buffer (8 × f32 = 32 bytes). */
const FLOATS_PER_RING = 8;

const SHOCKWAVE_WGSL = /* wgsl */`
// ─── Uniforms ────────────────────────────────────────────────────────────────
struct ShockwaveUniforms {
  width      : f32,
  height     : f32,
  ringCount  : f32,
  aspectRatio: f32,
}

struct Ring {
  cx        : f32,    // centre U
  cy        : f32,    // centre V
  radius    : f32,    // current expansion radius (UV)
  thickness : f32,    // distortion band width (UV)
  amplitude : f32,    // peak UV offset (pre-decayed)
  chromatic : f32,    // R/B split factor
  brightness: f32,    // additive white flash
  seed      : f32,    // per-ring noise seed
}

@group(0) @binding(0) var<uniform>       u     : ShockwaveUniforms;
@group(0) @binding(1) var                smp   : sampler;
@group(0) @binding(2) var                src   : texture_2d<f32>;
@group(0) @binding(3) var<storage, read> rings : array<Ring, ${MAX_GPU_RINGS}>;

// ─── Constants ───────────────────────────────────────────────────────────────
const PI : f32 = 3.14159265359;

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

// ─── Fragment shader — shockwave distortion + chromatic aberration ───────────

@fragment fn fs_shockwave(in: Vert) -> @location(0) vec4f {
  let st = in.uv;
  let ringCount = i32(u.ringCount);

  // Accumulate UV offset across all active rings
  var offsetR = vec2f(0.0);
  var offsetG = vec2f(0.0);
  var offsetB = vec2f(0.0);
  var addBright = 0.0;

  for (var i = 0; i < ${MAX_GPU_RINGS}; i++) {
    if (i >= ringCount) { break; }

    let ring = rings[i];

    // Aspect-corrected distance from ring centre
    let delta = vec2f(
      (st.x - ring.cx) * u.aspectRatio,
       st.y - ring.cy,
    );
    let dist = length(delta);

    // Radial direction (avoid division by zero)
    let radDir = select(delta / dist, vec2f(0.0), dist < 0.0001);

    // Distance from the ring front (positive = inside ring, behind wavefront)
    let fromFront = ring.radius - dist;

    // Band membership: [0, thickness] behind the wavefront
    if (fromFront < 0.0 || fromFront > ring.thickness) {
      continue;
    }

    // Band profile: sin(π · t) where t = fromFront / thickness
    // Peaks at the wavefront (t≈0 → just entered band), smooth taper
    let bandT = fromFront / ring.thickness;
    let profile = sin(PI * (1.0 - bandT));

    // UV displacement along radial direction
    // Un-correct aspect ratio for the actual UV offset
    let uvDir = vec2f(radDir.x / u.aspectRatio, radDir.y);
    let disp = uvDir * ring.amplitude * profile;

    // Chromatic aberration: split R and B along the radial
    let chrSpread = ring.chromatic * profile;
    offsetR += disp * (1.0 + chrSpread);
    offsetG += disp;
    offsetB += disp * (1.0 - chrSpread);

    // Additive brightness flash at wavefront
    addBright += ring.brightness * profile * profile;
  }

  // Sample scene colour with per-channel UV offsets
  let colR = textureSample(src, smp, st + offsetR).r;
  let colG = textureSample(src, smp, st + offsetG).g;
  let colB = textureSample(src, smp, st + offsetB).b;

  var color = vec3f(colR, colG, colB);

  // Add brightness flash (white-ish tint for energy feel)
  color += vec3f(addBright * 0.9, addBright * 0.95, addBright);

  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GPU uniform layout:  4 × f32 = 16 bytes
// ─────────────────────────────────────────────────────────────────────────────

const UNIFORM_FLOATS = 4;

// ─────────────────────────────────────────────────────────────────────────────
// CollisionShockwavePipeline  (WebGPU full-screen post-process pass)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WebGPU render pipeline that applies screen-space shockwave distortion
 * with chromatic aberration.
 *
 * Reads an array of active rings (uploaded each frame from
 * CollisionShockwaveSystem.getActiveRings()) and displaces UV coordinates
 * in a narrow radial band around each ring's expanding wavefront.
 *
 * Slots into the post-process chain alongside PostProcessPipeline —
 * run this pass after tone-mapping and before final present.
 *
 * @example
 * ```ts
 * const pipeline = await CollisionShockwavePipeline.create(device, format);
 *
 * // Per frame:
 * pipeline.uploadRings(shockwaveSystem.getActiveRings());
 * pipeline.render(encoder, sceneView, dstView, width, height);
 * ```
 */
export class CollisionShockwavePipeline {
  private readonly device:     GPUDevice;
  private readonly pipeline:   GPURenderPipeline;
  private readonly bgl:        GPUBindGroupLayout;
  private readonly sampler:    GPUSampler;
  private readonly uniformBuf: GPUBuffer;
  private readonly ringBuf:    GPUBuffer;

  // Bind group cache — invalidated when source texture view changes
  private cachedBG:  GPUBindGroup | null = null;
  private cachedSrc: GPUTextureView | null = null;

  private constructor(
    device:     GPUDevice,
    pipeline:   GPURenderPipeline,
    bgl:        GPUBindGroupLayout,
    sampler:    GPUSampler,
    uniformBuf: GPUBuffer,
    ringBuf:    GPUBuffer,
  ) {
    this.device     = device;
    this.pipeline   = pipeline;
    this.bgl        = bgl;
    this.sampler    = sampler;
    this.uniformBuf = uniformBuf;
    this.ringBuf    = ringBuf;
  }

  // ── Factory ────────────────────────────────────────────────────────────────

  /**
   * Compile the shockwave WGSL, create the render pipeline and GPU buffers.
   *
   * @param device  WebGPU device.
   * @param format  Swap-chain / render-target texture format.
   * @returns Ready-to-use CollisionShockwavePipeline instance.
   */
  static async create(
    device: GPUDevice,
    format: GPUTextureFormat,
  ): Promise<CollisionShockwavePipeline> {
    const module = device.createShaderModule({ code: SHOCKWAVE_WGSL });

    const bgl = device.createBindGroupLayout({
      entries: [
        // 0: uniforms (width, height, ringCount, aspectRatio)
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        // 1: linear sampler
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        // 2: source scene texture
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        // 3: ring SSBO (read-only storage)
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });

    const pipeline = await device.createRenderPipelineAsync({
      layout:   device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex:   { module, entryPoint: 'vs_fullscreen' },
      fragment: {
        module,
        entryPoint: 'fs_shockwave',
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

    // Ring storage buffer: MAX_GPU_RINGS × FLOATS_PER_RING × 4 bytes
    const ringBuf = device.createBuffer({
      size:  MAX_GPU_RINGS * FLOATS_PER_RING * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new CollisionShockwavePipeline(
      device, pipeline, bgl, sampler, uniformBuf, ringBuf,
    );
  }

  // ── Per-frame upload ───────────────────────────────────────────────────────

  /**
   * Upload the current set of active rings to the GPU storage buffer.
   *
   * Call this once per frame before `render()`.
   *
   * @param rings  Active rings from `CollisionShockwaveSystem.getActiveRings()`.
   */
  uploadRings(rings: ShockwaveRingGPU[]): void {
    const count = Math.min(rings.length, MAX_GPU_RINGS);
    const data = new Float32Array(MAX_GPU_RINGS * FLOATS_PER_RING);

    for (let i = 0; i < count; i++) {
      const r = rings[i];
      const base = i * FLOATS_PER_RING;
      data[base + 0] = r.cx;
      data[base + 1] = r.cy;
      data[base + 2] = r.radius;
      data[base + 3] = r.thickness;
      data[base + 4] = r.amplitude;
      data[base + 5] = r.chromatic;
      data[base + 6] = r.brightness;
      data[base + 7] = r.seed;
    }

    this.device.queue.writeBuffer(this.ringBuf, 0, data);
    // Ring count is written in the uniform buffer during render()
    this._ringCount = count;
  }

  private _ringCount = 0;

  // ── Render ─────────────────────────────────────────────────────────────────

  /**
   * Record the shockwave distortion pass into `encoder`.
   *
   * If no rings are active the pass is still recorded (as a pass-through
   * blit) to maintain consistent attachment state in the render graph.
   *
   * @param encoder  Active GPUCommandEncoder.
   * @param srcView  Input scene colour texture view.
   * @param dstView  Output texture view (swap-chain surface or next RT).
   * @param width    Render target width in pixels.
   * @param height   Render target height in pixels.
   */
  render(
    encoder: GPUCommandEncoder,
    srcView: GPUTextureView,
    dstView: GPUTextureView,
    width:   number,
    height:  number,
  ): void {
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
    pass.draw(6);   // two triangles = full-screen quad
    pass.end();
  }

  /**
   * Destroy GPU resources.  Call when the pipeline is no longer needed.
   */
  destroy(): void {
    this.uniformBuf.destroy();
    this.ringBuf.destroy();
  }

  // ── Internal ───────────────────────────────────────────────────────────────

  private _uploadUniforms(width: number, height: number): void {
    const data = new Float32Array(UNIFORM_FLOATS);
    data[0] = width;
    data[1] = height;
    data[2] = this._ringCount;
    data[3] = width / Math.max(height, 1);   // aspect ratio
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
        { binding: 3, resource: { buffer: this.ringBuf } },
      ],
    });
    this.cachedBG  = bg;
    this.cachedSrc = srcView;
    return bg;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: impulse estimation (mirrors collision-fx-system.ts convention)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Estimate a proxy impulse magnitude from collision contact geometry.
 *
 * Prefer the actual rigid-body solver impulse `j` when available.
 * This helper covers cases where only CollisionContactInfo is accessible.
 *
 * @param contact    Contact info from the narrow phase or CollisionEvent.
 * @param relSpeed   Relative speed at the contact point (px/s).
 * @param massScale  Tuning constant for visual intensity.
 * @returns Impulse proxy suitable for `CollisionShockwaveSystem.emit`.
 */
export function estimateShockwaveImpulse(
  contact: CollisionContactInfo,
  relSpeed = 0,
  massScale = 1,
): number {
  return (contact.depth * 100 + relSpeed * 0.6) * massScale;
}
