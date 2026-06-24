/**
 * physics-render-bridge.ts — M781: SPH → Render Bridge
 * ─────────────────────────────────────────────────────────────────────────────
 * Bridges the live SPH physics simulation to every rendering subsystem in a
 * single, frame-coherent data pump.  Instead of each renderer independently
 * sampling the physics state (leading to one-frame skew between passes),
 * this bridge captures a consistent snapshot at the start of each frame and
 * fans it out to all consumers.
 *
 * Data flow
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   SPHWorld / World (physics)
 *        │
 *        │  capture()
 *        ▼
 *   PhysicsRenderBridge
 *        │
 *        ├──→  ParticleRenderer         (positions, velocities, species)
 *        ├──→  InstancedCellRenderer     (rigid body transforms + PhysicsUniforms)
 *        ├──→  EdgeFlowRenderer          (contact forces → edge flow rate)
 *        ├──→  CollisionFXSystem         (contact manifolds → sparks/shockwaves)
 *        ├──→  CurlAuraRenderer          (density field → aura intensity)
 *        ├──→  DensityFieldTexture       (particle density → GPU texture)
 *        ├──→  VelocityFieldTexture      (particle velocity → GPU texture)
 *        ├──→  VFXTimelinePlayer         (collision events → VFX triggers)
 *        └──→  SSE broadcast             (physics_step / physics_collision)
 *
 * The bridge operates in two modes:
 *
 *   1. **GPU mode** — reads directly from GPUBufferSet (WebGPU path).
 *      Positions and velocities are already on the GPU; the bridge only
 *      performs the CPU-side rigid-body / contact readback and dispatches
 *      GPU buffer references to renderers that can bind them directly.
 *
 *   2. **CPU mode** — reads from the World (world-stepper.ts) particle
 *      arrays.  Used by the Canvas2D debug renderer and the SSE broadcast
 *      path.  Particle data is packed into typed arrays for efficient
 *      iteration.
 *
 * Frame protocol
 * ─────────────────────────────────────────────────────────────────────────────
 *   const bridge = new PhysicsRenderBridge(options);
 *
 *   // per frame:
 *   bridge.capture(world, dt);         // snapshot physics state
 *   bridge.dispatchToRenderers();      // fan out to all registered consumers
 *   bridge.dispatchCollisionFX();      // trigger VFX for new contacts
 *   bridge.broadcastSSE();             // fire SSE events (rate-limited)
 *
 * ── References ─────────────────────────────────────────────────────────────
 *   src/lib/sph/SPHWorld.ts              — WebGPU SPH simulation
 *   src/lib/sph/world-stepper.ts         — CPU SPH World
 *   src/lib/sph/physics-uniform-bridge.ts — per-body uniform sampling
 *   src/lib/sph/sph-bridge.ts            — worker bridge (frame snapshot)
 *   src/lib/sph/instanced-cell-renderer.ts — GPU instanced cell draw
 *   src/lib/sph/collision-fx-system.ts   — collision VFX dispatcher
 *   src/lib/sph/vfx-timeline.ts          — VFX event sequencer
 *   backend/sse_physics_bridge.py        — SSE broadcast (Python side)
 *
 * Research: xiaodi #M781 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

<<<<<<< HEAD
// [orphan-precise] /**
// [orphan-precise]  * Minimal physics world interface — duck-typed to accept both SPHWorld
// [orphan-precise]  * (WebGPU) and World (CPU world-stepper.ts) without tight coupling.
// [orphan-precise]  */
=======
/**
 * Minimal physics world interface — duck-typed to accept both SPHWorld
 * (WebGPU) and World (CPU world-stepper.ts) without tight coupling.
 */



import type { ParticleData, GPUBufferSet, SimParams } from './types';
import type { PhysicsUniforms } from './physics-uniform-bridge';
import { samplePhysicsForBody } from './physics-uniform-bridge';
import type { SPHFrameSnapshot } from './sph-bridge';
import type { CollisionFXSystem } from './collision-fx-system';
import type { VFXTimelinePlayer, VFXEventKind } from './vfx-timeline';
import type { DensityFieldTexture } from './density-field-texture';
import type { VelocityFieldTexture } from './velocity-field-texture';

>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
export interface PhysicsWorldView {
  /** Fluid particle arrays (CPU-side mirror for readback). */
  readonly particles: ParticleData | ReadonlyArray<{
    x: number; y: number;
    vx: number; vy: number;
    density: number; pressure: number;
    mass: number; species: number;
  }>;

  /** Rigid bodies currently in the simulation. */
  readonly rigidBodies: ReadonlyArray<{
    id: number | string;
    x: number; y: number;
    angle: number;
    vx: number; vy: number;
    omega: number;
    invMass: number;
  }>;

  /** Active contact points from the last physics step. */
  readonly contacts: ReadonlyArray<{
    x: number; y: number;
    nx: number; ny: number;
    depth: number;
    bodyA: number; bodyB: number;
    impulse?: number;
  }>;

  /** Simulation elapsed time (seconds). */
  readonly time: number;

  /** Domain dimensions. */
  readonly domainW: number;
  readonly domainH: number;

  /** Simulation parameters. */
  readonly params: SimParams;
}

/** A rigid body's per-frame render state. */
export interface RigidBodyRenderState {
  id: string;
  x: number;
  y: number;
  angle: number;
  vx: number;
  vy: number;
  omega: number;
  species: number;
  pinned: boolean;
  uniforms: PhysicsUniforms;
}

/** Contact event for VFX dispatch. */
export interface ContactEvent {
  x: number;
  y: number;
  nx: number;
  ny: number;
  depth: number;
  impulse: number;
  bodyA: string;
  bodyB: string;
  isNew: boolean;
}

/** Aggregate physics stats for HUD / debug overlay. */
export interface PhysicsFrameStats {
  particleCount: number;
  rigidBodyCount: number;
  contactCount: number;
  avgDensity: number;
  maxVelocity: number;
  kineticEnergy: number;
  simTime: number;
  dt: number;
  frameIndex: number;
}

/** Per-frame snapshot captured by the bridge. */
export interface PhysicsRenderSnapshot {
  /** Monotonically increasing frame counter. */
  frameIndex: number;

  /** Delta time for this frame (seconds). */
  dt: number;

  /** Packed particle positions — interleaved [x0, y0, x1, y1, ...]. */
  particlePositions: Float32Array;

  /** Packed particle velocities — interleaved [vx0, vy0, vx1, vy1, ...]. */
  particleVelocities: Float32Array;

  /** Particle species tags. */
  particleSpecies: Uint32Array;

  /** Number of active particles. */
  particleCount: number;

  /** Per-rigid-body render state (transforms + physics uniforms). */
  bodies: RigidBodyRenderState[];

  /** Active contact events (with new-contact detection). */
  contacts: ContactEvent[];

  /** Aggregate frame statistics. */
  stats: PhysicsFrameStats;

  /** Reference to GPU buffers when in GPU mode (null for CPU mode). */
  gpuBuffers: GPUBufferSet | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Renderer consumer interface
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Any rendering subsystem that wants to receive physics snapshots must
 * implement this interface and register itself with the bridge.
 */
export interface PhysicsRenderConsumer {
  /**
   * Called once per frame with the consistent physics snapshot.
   * Implementations should NOT perform their own physics sampling;
   * all data is pre-captured in the snapshot.
   */
  onPhysicsFrame(snapshot: PhysicsRenderSnapshot): void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

export interface PhysicsRenderBridgeOptions {
  /** Canvas pixel dimensions (for domain→pixel coordinate conversion). */
  canvasW: number;
  canvasH: number;

  /** Sampling radius for per-body physics uniform extraction (world units). */
  sampleRadius: number;

  /** SSE broadcast rate limit (milliseconds). 0 = disabled. */
  sseBroadcastIntervalMs: number;

  /** Minimum impulse magnitude to trigger collision VFX. */
  collisionImpulseThreshold: number;

  /** Maximum contacts to process per frame (performance cap). */
  maxContactsPerFrame: number;

  /** GPU buffers reference (null for CPU-only mode). */
  gpuBuffers: GPUBufferSet | null;

  /** Species ID resolver — maps rigid body id → species string. */
  speciesResolver: ((bodyId: string) => number) | null;
}

const DEFAULT_OPTIONS: PhysicsRenderBridgeOptions = {
  canvasW: 1280,
  canvasH: 720,
  sampleRadius: 0.3,
  sseBroadcastIntervalMs: 100,
  collisionImpulseThreshold: 0.5,
  maxContactsPerFrame: 64,
  gpuBuffers: null,
  speciesResolver: null,
};

// ─────────────────────────────────────────────────────────────────────────────
// Contact tracking (for new-contact detection)
// ─────────────────────────────────────────────────────────────────────────────

/** Generates a stable key for a contact pair (order-independent). */
function contactPairKey(a: number | string, b: number | string): string {
  const sa = String(a);
  const sb = String(b);
  return sa < sb ? `${sa}:${sb}` : `${sb}:${sa}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// PhysicsRenderBridge
// ─────────────────────────────────────────────────────────────────────────────

export class PhysicsRenderBridge {
  private opts: PhysicsRenderBridgeOptions;
  private consumers: Set<PhysicsRenderConsumer> = new Set();
  private collisionFX: CollisionFXSystem | null = null;
  private vfxPlayer: VFXTimelinePlayer | null = null;
  private densityField: DensityFieldTexture | null = null;
  private velocityField: VelocityFieldTexture | null = null;

  // ── Internal state ──────────────────────────
  private frameIndex = 0;
  private lastSnapshot: PhysicsRenderSnapshot | null = null;

  // ── Contact tracking ────────────────────────
  /** Set of contact pair keys active in the previous frame. */
  private prevContactKeys: Set<string> = new Set();

  // ── SSE rate limiter ────────────────────────
  private lastSSEBroadcast = 0;

  // ── Reusable typed arrays (avoid per-frame allocation) ──
  private posBuffer: Float32Array;
  private velBuffer: Float32Array;
  private speciesBuffer: Uint32Array;
  private maxParticleCapacity: number;

  // ────────────────────────────────────────────
  constructor(options: Partial<PhysicsRenderBridgeOptions> = {}) {
    this.opts = { ...DEFAULT_OPTIONS, ...options };

    // Pre-allocate buffers for up to 50k particles (resizable)
    this.maxParticleCapacity = 50_000;
    this.posBuffer     = new Float32Array(this.maxParticleCapacity * 2);
    this.velBuffer     = new Float32Array(this.maxParticleCapacity * 2);
    this.speciesBuffer = new Uint32Array(this.maxParticleCapacity);
  }

  // ────────────────────────────────────────────
  // Registration
  // ────────────────────────────────────────────

  /** Register a rendering consumer to receive physics snapshots. */
  addConsumer(consumer: PhysicsRenderConsumer): void {
    this.consumers.add(consumer);
  }

  /** Unregister a rendering consumer. */
  removeConsumer(consumer: PhysicsRenderConsumer): void {
    this.consumers.delete(consumer);
  }

  /** Attach the collision VFX system (sparks, shockwaves). */
  setCollisionFX(fx: CollisionFXSystem): void {
    this.collisionFX = fx;
  }

  /** Attach the VFX timeline player for event-driven effects. */
  setVFXPlayer(player: VFXTimelinePlayer): void {
    this.vfxPlayer = player;
  }

  /** Attach the density field texture for GPU field rasterization. */
  setDensityField(field: DensityFieldTexture): void {
    this.densityField = field;
  }

  /** Attach the velocity field texture for GPU velocity rasterization. */
  setVelocityField(field: VelocityFieldTexture): void {
    this.velocityField = field;
  }

  /** Update canvas dimensions (e.g. on resize). */
  updateCanvasDimensions(w: number, h: number): void {
    this.opts.canvasW = w;
    this.opts.canvasH = h;
  }

  /** Update GPU buffer references (e.g. after buffer reallocation). */
  updateGPUBuffers(bufs: GPUBufferSet | null): void {
    this.opts.gpuBuffers = bufs;
  }

  // ────────────────────────────────────────────
  // Snapshot capture
  // ────────────────────────────────────────────

  /**
   * Capture a frame-coherent physics snapshot from the live world.
   *
   * Call this once at the start of each frame, BEFORE any renderer
   * accesses physics data.  All subsequent dispatchToRenderers() /
   * dispatchCollisionFX() / broadcastSSE() calls use this snapshot.
   *
   * @param world  The physics world (SPHWorld or World from world-stepper)
   * @param dt     Frame delta time in seconds
   */
  capture(world: PhysicsWorldView, dt: number): PhysicsRenderSnapshot {
    this.frameIndex++;

    // ── 1. Pack particle data ──────────────────────────────────────────
    const particleCount = this._extractParticleCount(world);
    this._ensureBufferCapacity(particleCount);
    this._packParticles(world, particleCount);

    // ── 2. Extract rigid body state + per-body physics uniforms ───────
    const bodies = this._extractBodies(world);

    // ── 3. Process contacts with new-contact detection ────────────────
    const contacts = this._extractContacts(world);

    // ── 4. Compute aggregate stats ───────────────────────────────────
    const stats = this._computeStats(world, particleCount, bodies, contacts, dt);

    // ── 5. Assemble snapshot ─────────────────────────────────────────
    const snapshot: PhysicsRenderSnapshot = {
      frameIndex: this.frameIndex,
      dt,
      particlePositions: this.posBuffer,
      particleVelocities: this.velBuffer,
      particleSpecies: this.speciesBuffer,
      particleCount,
      bodies,
      contacts,
      stats,
      gpuBuffers: this.opts.gpuBuffers,
    };

    this.lastSnapshot = snapshot;
    return snapshot;
  }

  // ────────────────────────────────────────────
  // Dispatch
  // ────────────────────────────────────────────

  /**
   * Fan out the captured snapshot to all registered rendering consumers.
   *
   * Each consumer receives the same snapshot reference — mutations by
   * one consumer are visible to later ones (intentional: allows a
   * consumer to annotate the snapshot with derived data for downstream
   * consumers, e.g. CurlAura → VFX).
   */
  dispatchToRenderers(): void {
    if (!this.lastSnapshot) return;

    for (const consumer of this.consumers) {
      try {
        consumer.onPhysicsFrame(this.lastSnapshot);
      } catch (err) {
        console.warn('[PhysicsRenderBridge] consumer error:', err);
      }
    }

    // ── Feed density field texture ────────────────────────────────────
    if (this.densityField && this.lastSnapshot.particleCount > 0) {
      try {
        const snap = this.lastSnapshot;
        this.densityField.update({
          positions: snap.particlePositions,
          count: snap.particleCount,
          domainW: snap.stats.dt > 0 ? this.opts.canvasW : this.opts.canvasW,
          domainH: this.opts.canvasH,
        });
      } catch (err) {
        console.warn('[PhysicsRenderBridge] densityField error:', err);
      }
    }

    // ── Feed velocity field texture ──────────────────────────────────
    if (this.velocityField && this.lastSnapshot.particleCount > 0) {
      try {
        const snap = this.lastSnapshot;
        this.velocityField.update({
          positions: snap.particlePositions,
          velocities: snap.particleVelocities,
          count: snap.particleCount,
          domainW: this.opts.canvasW,
          domainH: this.opts.canvasH,
        });
      } catch (err) {
        console.warn('[PhysicsRenderBridge] velocityField error:', err);
      }
    }
  }

  /**
   * Dispatch collision VFX for newly-detected contacts.
   *
   * Only contacts with `isNew === true` and impulse above the threshold
   * trigger VFX events.  This prevents repeated sparks/shockwaves on
   * persistent resting contacts.
   */
  dispatchCollisionFX(): void {
    if (!this.lastSnapshot) return;

    const threshold = this.opts.collisionImpulseThreshold;

    for (const contact of this.lastSnapshot.contacts) {
      if (!contact.isNew) continue;
      if (contact.impulse < threshold) continue;

      // ── CollisionFXSystem (sparks + shockwave rings) ───────────────
      if (this.collisionFX) {
        try {
          this.collisionFX.onCollision({
            x: contact.x,
            y: contact.y,
            nx: contact.nx,
            ny: contact.ny,
            impulse: contact.impulse,
          });
        } catch (err) {
          console.warn('[PhysicsRenderBridge] collisionFX error:', err);
        }
      }

      // ── VFX timeline (shockwave + bloom spike + screen flash) ─────
      if (this.vfxPlayer) {
        try {
          const intensity = Math.min(contact.impulse / 10, 1);
          this.vfxPlayer.fire('shockwave' as VFXEventKind, {
            x: contact.x,
            y: contact.y,
            intensity,
          });
          if (intensity > 0.5) {
            this.vfxPlayer.fire('bloom_spike' as VFXEventKind, {
              intensity: intensity * 0.6,
              duration: 0.15,
            });
          }
        } catch (err) {
          console.warn('[PhysicsRenderBridge] vfxPlayer error:', err);
        }
      }
    }
  }

  /**
   * Broadcast the current physics state via SSE, subject to rate limiting.
   *
   * Uses a lightweight fetch POST to the server endpoint which then
   * distributes via the DataNotifier → SSE pipeline.
   *
   * @returns true if a broadcast was sent, false if rate-limited or skipped.
   */
  broadcastSSE(): boolean {
    if (!this.lastSnapshot) return false;
    if (this.opts.sseBroadcastIntervalMs <= 0) return false;

    const now = performance.now();
    if (now - this.lastSSEBroadcast < this.opts.sseBroadcastIntervalMs) {
      return false;
    }
    this.lastSSEBroadcast = now;

    const snap = this.lastSnapshot;

    // Build a compact force_field representation for the SSE payload
    const forceField: Record<string, { dx: number; dy: number; dz: number }> = {};
    for (const body of snap.bodies) {
      const [vx, vy] = body.uniforms.u_velocity;
      forceField[body.id] = { dx: vx, dy: vy, dz: 0 };
    }

    // Collision payload
    const collisions = snap.contacts
      .filter((c) => c.isNew)
      .map((c) => ({
        bodyA: c.bodyA,
        bodyB: c.bodyB,
        normal: { x: c.nx, y: c.ny },
        depth: c.depth,
        impulse: c.impulse,
      }));

    // Non-blocking fire-and-forget POST
    this._postSSE('physics_step', {
      epoch: snap.frameIndex,
      force_field: forceField,
      converged: snap.stats.maxVelocity < 0.01,
      stats: {
        particleCount: snap.stats.particleCount,
        avgDensity: snap.stats.avgDensity,
        maxVelocity: snap.stats.maxVelocity,
        kineticEnergy: snap.stats.kineticEnergy,
      },
    });

    if (collisions.length > 0) {
      this._postSSE('physics_collision', {
        collisions,
        count: collisions.length,
      });
    }

    return true;
  }

  // ────────────────────────────────────────────
  // Accessors
  // ────────────────────────────────────────────

  /** Return the most recent snapshot (null before first capture). */
  getLastSnapshot(): PhysicsRenderSnapshot | null {
    return this.lastSnapshot;
  }

  /** Return aggregate stats from the most recent frame. */
  getStats(): PhysicsFrameStats | null {
    return this.lastSnapshot?.stats ?? null;
  }

  /**
   * Convert a physics-domain position to canvas-pixel coordinates.
   *
   * Used by renderers that operate in pixel space (PixiJS, Canvas2D)
   * rather than normalised domain coordinates (WebGPU shaders).
   */
  domainToPixel(x: number, y: number, domainW: number, domainH: number): [number, number] {
    return [
      (x / domainW) * this.opts.canvasW,
      (y / domainH) * this.opts.canvasH,
    ];
  }

  /**
   * Convert canvas-pixel coordinates back to physics-domain position.
   */
  pixelToDomain(px: number, py: number, domainW: number, domainH: number): [number, number] {
    return [
      (px / this.opts.canvasW) * domainW,
      (py / this.opts.canvasH) * domainH,
    ];
  }

  // ────────────────────────────────────────────
  // Cleanup
  // ────────────────────────────────────────────

  /** Release all references. Safe to call multiple times. */
  destroy(): void {
    this.consumers.clear();
    this.collisionFX = null;
    this.vfxPlayer = null;
    this.densityField = null;
    this.velocityField = null;
    this.lastSnapshot = null;
    this.prevContactKeys.clear();
  }

  // ────────────────────────────────────────────
  // Private: particle extraction
  // ────────────────────────────────────────────

  private _extractParticleCount(world: PhysicsWorldView): number {
    const p = world.particles;
    if ('count' in p && typeof p.count === 'number') {
      // ParticleData (SoA from SPHWorld)
      return (p as ParticleData).count;
    }
    // AoS from world-stepper
    return (p as ReadonlyArray<unknown>).length;
  }

  private _ensureBufferCapacity(count: number): void {
    if (count <= this.maxParticleCapacity) return;

    // Grow by 2× to amortise reallocations
    this.maxParticleCapacity = Math.max(count, this.maxParticleCapacity * 2);
    this.posBuffer     = new Float32Array(this.maxParticleCapacity * 2);
    this.velBuffer     = new Float32Array(this.maxParticleCapacity * 2);
    this.speciesBuffer = new Uint32Array(this.maxParticleCapacity);
  }

  private _packParticles(world: PhysicsWorldView, count: number): void {
    const p = world.particles;

    if ('x' in p && p.x instanceof Float32Array) {
      // SoA layout (SPHWorld CPU mirror)
      const soa = p as ParticleData;
      for (let i = 0; i < count; i++) {
        this.posBuffer[i * 2]     = soa.x[i];
        this.posBuffer[i * 2 + 1] = soa.y[i];
        this.velBuffer[i * 2]     = soa.vx[i];
        this.velBuffer[i * 2 + 1] = soa.vy[i];
        this.speciesBuffer[i]     = soa.species[i];
      }
    } else {
      // AoS layout (world-stepper)
      const aos = p as ReadonlyArray<{
        x: number; y: number;
        vx: number; vy: number;
        species: number;
      }>;
      for (let i = 0; i < count; i++) {
        const pt = aos[i];
        this.posBuffer[i * 2]     = pt.x;
        this.posBuffer[i * 2 + 1] = pt.y;
        this.velBuffer[i * 2]     = pt.vx;
        this.velBuffer[i * 2 + 1] = pt.vy;
        this.speciesBuffer[i]     = pt.species;
      }
    }
  }

  // ────────────────────────────────────────────
  // Private: rigid body extraction
  // ────────────────────────────────────────────

  private _extractBodies(world: PhysicsWorldView): RigidBodyRenderState[] {
    const bodies: RigidBodyRenderState[] = [];
    const resolver = this.opts.speciesResolver;

    for (const rb of world.rigidBodies) {
      const id = String(rb.id);
      const speciesIdx = resolver ? resolver(id) : 0;
      const pinned = rb.invMass === 0;

      // Sample per-body physics uniforms from the fluid neighbourhood.
      // We duck-type the world to samplePhysicsForBody's expected shape;
      // if the world doesn't match we fall back to zero uniforms.
      let uniforms: PhysicsUniforms;
      try {
        uniforms = samplePhysicsForBody(
          world as any,
          id,
          this.opts.sampleRadius,
        );
      } catch {
        uniforms = {
          u_density: 0,
          u_velocity: [0, 0],
          u_pressure: 0,
          u_contactCount: 0,
          u_neighborCount: 0,
          u_kineticEnergy: 0,
          u_vorticity: 0,
          u_time: world.time,
        };
      }

      bodies.push({
        id,
        x: rb.x,
        y: rb.y,
        angle: rb.angle,
        vx: rb.vx,
        vy: rb.vy,
        omega: rb.omega,
        species: speciesIdx,
        pinned,
        uniforms,
      });
    }

    return bodies;
  }

  // ────────────────────────────────────────────
  // Private: contact extraction + new-contact detection
  // ────────────────────────────────────────────

  private _extractContacts(world: PhysicsWorldView): ContactEvent[] {
    const contacts: ContactEvent[] = [];
    const currentKeys = new Set<string>();

    const maxContacts = Math.min(
      world.contacts.length,
      this.opts.maxContactsPerFrame,
    );

    for (let i = 0; i < maxContacts; i++) {
      const c = world.contacts[i];
      const key = contactPairKey(c.bodyA, c.bodyB);
      currentKeys.add(key);

      const isNew = !this.prevContactKeys.has(key);
      const impulse = c.impulse ?? Math.abs(c.depth) * 100;

      contacts.push({
        x: c.x,
        y: c.y,
        nx: c.nx,
        ny: c.ny,
        depth: c.depth,
        impulse,
        bodyA: String(c.bodyA),
        bodyB: String(c.bodyB),
        isNew,
      });
    }

    // Update tracking set for next frame
    this.prevContactKeys = currentKeys;

    return contacts;
  }

  // ────────────────────────────────────────────
  // Private: stats computation
  // ────────────────────────────────────────────

  private _computeStats(
    world: PhysicsWorldView,
    particleCount: number,
    bodies: RigidBodyRenderState[],
    contacts: ContactEvent[],
    dt: number,
  ): PhysicsFrameStats {
    let sumDensity = 0;
    let maxVelocity = 0;
    let kineticEnergy = 0;

    for (let i = 0; i < particleCount; i++) {
      const vx = this.velBuffer[i * 2];
      const vy = this.velBuffer[i * 2 + 1];
      const speed = Math.sqrt(vx * vx + vy * vy);
      if (speed > maxVelocity) maxVelocity = speed;
      kineticEnergy += 0.5 * speed * speed;
    }

    // Density from body uniforms (averaged over sampled bodies)
    if (bodies.length > 0) {
      for (const b of bodies) {
        sumDensity += b.uniforms.u_density;
      }
      sumDensity /= bodies.length;
    }

    return {
      particleCount,
      rigidBodyCount: bodies.length,
      contactCount: contacts.length,
      avgDensity: sumDensity,
      maxVelocity,
      kineticEnergy,
      simTime: world.time,
      dt,
      frameIndex: this.frameIndex,
    };
  }

  // ────────────────────────────────────────────
  // Private: SSE broadcast
  // ────────────────────────────────────────────

  private _postSSE(event: string, data: unknown): void {
    try {
      fetch('/api/cell/publish', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ event, data }),
      }).catch(() => {
        // SSE broadcast is best-effort; swallow fetch errors silently
      });
    } catch {
      // Guard against synchronous throw from JSON.stringify on circular refs
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adapter: wrap SPHFrameSnapshot from sph-bridge worker as PhysicsWorldView
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Wraps an `SPHFrameSnapshot` (returned by the SPH Web Worker via
 * sph-bridge.ts) into a `PhysicsWorldView` so it can be fed directly
 * to `PhysicsRenderBridge.capture()`.
 *
 * This adapter is used when the physics simulation runs in a dedicated
 * Worker thread and the main thread receives snapshots via Comlink.
 */
export function snapshotToWorldView(
  snapshot: SPHFrameSnapshot,
  domainW: number,
  domainH: number,
  params: SimParams,
): PhysicsWorldView {
  // The snapshot's .particles is a packed Float64Array:
  // [x0, y0, vx0, vy0, density0, pressure0, mass0, species0, x1, ...]
  const STRIDE = 8;
  const count = snapshot.particleCount;

  const particles: Array<{
    x: number; y: number;
    vx: number; vy: number;
    density: number; pressure: number;
    mass: number; species: number;
  }> = [];

  for (let i = 0; i < count; i++) {
    const off = i * STRIDE;
    particles.push({
      x:        snapshot.particles[off + 0],
      y:        snapshot.particles[off + 1],
      vx:       snapshot.particles[off + 2],
      vy:       snapshot.particles[off + 3],
      density:  snapshot.particles[off + 4],
      pressure: snapshot.particles[off + 5],
      mass:     snapshot.particles[off + 6],
      species:  snapshot.particles[off + 7],
    });
  }

  const rigidBodies = snapshot.rigidBodies.map((rb) => ({
    id: rb.id,
    x: rb.x,
    y: rb.y,
    angle: rb.angle ?? 0,
    vx: 0,
    vy: 0,
    omega: 0,
    invMass: rb.pinned ? 0 : 1,
  }));

  const contacts = snapshot.contacts.map((c) => ({
    x: c.x,
    y: c.y,
    nx: c.nx,
    ny: c.ny,
    depth: c.depth,
    bodyA: 0,
    bodyB: 0,
    impulse: Math.abs(c.depth) * 100,
  }));

  return {
    particles,
    rigidBodies,
    contacts,
    time: performance.now() * 0.001,
    domainW,
    domainH,
    params,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Convenience factory that creates a PhysicsRenderBridge with sensible
 * defaults for the AT cell-pubsub-loop pipeline.
 */
export function createPhysicsRenderBridge(
  canvas: HTMLCanvasElement,
  gpuBuffers?: GPUBufferSet | null,
  options?: Partial<PhysicsRenderBridgeOptions>,
): PhysicsRenderBridge {
  return new PhysicsRenderBridge({
    canvasW: canvas.width,
    canvasH: canvas.height,
    gpuBuffers: gpuBuffers ?? null,
    ...options,
  });
}
