/**
 * physarum-edge-bridge.ts — M742
 *
 * Bridge between PhysarumSimulation (GPU trail map) and EdgeFlowRenderer
 * (CPU Canvas2D particle system).
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * The Physarum slime-mould simulation (physarum-sim.ts) produces an r32float
 * trail-map texture on the GPU — a pheromone concentration field that evolves
 * emergent vascular networks.  The EdgeFlowRenderer (edge-flow-renderer.ts)
 * animates particles along topology edges via Catmull-Rom splines.
 *
 * This bridge connects the two systems:
 *
 *   1. **Trail readback** — copies the GPU trail texture to a CPU-side
 *      Float32Array at a configurable frequency (default: every 4 frames)
 *      to amortise the GPU→CPU transfer cost.
 *
 *   2. **Pheromone sampling** — provides a `sampleTrail(x, y)` function that
 *      edge-flow particles call during their FLOW phase to query local
 *      pheromone concentration in domain coordinates.
 *
 *   3. **Speed modulation** — particles travelling through high-pheromone
 *      regions accelerate (chemotaxis attraction), while low-pheromone
 *      regions slow them down.  The modulation curve is:
 *
 *        effectiveSpeed = baseSpeed × lerp(minFactor, maxFactor, saturate(trail / trailCeil))
 *
 *   4. **Drift bias** — optionally nudges particle headings toward the local
 *      pheromone gradient (finite-difference on the readback grid), pulling
 *      edge-flow particles toward nearby Physarum veins.
 *
 *   5. **Deposit feedback** — when edge-flow particles arrive at their target
 *      cell, the bridge can inject a pheromone deposit into the Physarum
 *      simulation's trail map via a small GPU compute pass, creating a
 *      bidirectional coupling: edges feed the mould, the mould shapes edges.
 *
 * ─── AT context ──────────────────────────────────────────────────────────────
 * ActiveTheory's visual pipeline frequently couples simulation layers through
 * lightweight bridge modules that read back a low-frequency snapshot of one
 * GPU pass to modulate a second system (cf. flowmap-bridge.ts, ocean-bridge.ts).
 * This bridge follows the same pattern: async readback → CPU sample → modulate.
 *
 * ─── Usage ───────────────────────────────────────────────────────────────────
 *
 *   import { PhysarumEdgeBridge } from '$lib/sph/physarum-edge-bridge';
 *   import { PhysarumSimulation } from '$lib/sph/physarum-sim';
 *   import { EdgeFlowRenderer }  from '$lib/sph/edge-flow-renderer';
 *
 *   const physarum = await PhysarumSimulation.create(device, 512, 512, 500_000);
 *   const flow     = new EdgeFlowRenderer(ctx, { edges });
 *
 *   const bridge = new PhysarumEdgeBridge(device, physarum, {
 *     domainWidth:  canvasWidth,
 *     domainHeight: canvasHeight,
 *   });
 *
 *   // In render loop:
 *   bridge.readback(frameCount);                   // async GPU→CPU (throttled)
 *   const speedMod = bridge.speedModulator();       // returns (x, y, baseSpeed) => modulated speed
 *   // pass speedMod into custom EdgeFlowRenderer update logic
 *
 * ─── References ──────────────────────────────────────────────────────────────
 *   src/lib/sph/physarum-sim.ts       — GPU Physarum simulation (Jones 2010)
 *   src/lib/sph/edge-flow-renderer.ts — CPU edge-flow particle system
 *   src/lib/sph/flowmap-bridge.ts     — analogous GPU→CPU bridge pattern
 */

import type { PhysarumSimulation, PhysarumParams } from './physarum-sim';

// ─── Configuration ────────────────────────────────────────────────────────────

export interface PhysarumEdgeBridgeConfig {
  /** Domain width in canvas/world units (maps trail texture X to world X). */
  domainWidth:    number;
  /** Domain height in canvas/world units (maps trail texture Y to world Y). */
  domainHeight:   number;
  /** Read back trail texture every N frames (default 4). */
  readbackInterval?: number;
  /** Pheromone concentration ceiling for speed modulation saturation (default 50). */
  trailCeil?:     number;
  /** Minimum speed factor at zero pheromone (default 0.6). */
  minSpeedFactor?: number;
  /** Maximum speed factor at saturated pheromone (default 1.8). */
  maxSpeedFactor?: number;
  /** Gradient drift strength — 0 disables drift bias (default 0.15). */
  driftStrength?: number;
  /** Enable deposit feedback from edge arrivals (default false). */
  depositFeedback?: boolean;
  /** Deposit amount per arrival event (default 20). */
  depositAmount?: number;
}

const DEFAULTS = {
  readbackInterval: 4,
  trailCeil:        50.0,
  minSpeedFactor:   0.6,
  maxSpeedFactor:   1.8,
  driftStrength:    0.15,
  depositFeedback:  false,
  depositAmount:    20.0,
} as const;

// ─── Types ────────────────────────────────────────────────────────────────────

/** Speed modulation function: given world position and base speed, returns modulated speed. */
export type SpeedModulatorFn = (x: number, y: number, baseSpeed: number) => number;

/** Drift vector at a world position (normalised pheromone gradient). */
export interface DriftVector {
  dx: number;
  dy: number;
  /** Pheromone concentration at the sample point (unnormalised). */
  concentration: number;
}

/** Pending deposit from edge-flow arrival. */
interface PendingDeposit {
  /** Trail-space X coordinate. */
  tx: number;
  /** Trail-space Y coordinate. */
  ty: number;
  /** Deposit amount. */
  amount: number;
}

// ─── PhysarumEdgeBridge ───────────────────────────────────────────────────────

export class PhysarumEdgeBridge {
  private readonly device:     GPUDevice;
  private readonly sim:        PhysarumSimulation;
  private readonly config:     Required<PhysarumEdgeBridgeConfig>;

  // Trail readback state
  private trailData:           Float32Array | null = null;
  private trailWidth  = 0;
  private trailHeight = 0;
  private lastReadbackFrame = -Infinity;

  // GPU readback resources (lazily created)
  private readbackBuf:   GPUBuffer | null = null;
  private stagingBuf:    GPUBuffer | null = null;
  private readbackReady  = true;

  // Deposit feedback queue
  private pendingDeposits: PendingDeposit[] = [];

  constructor(
    device: GPUDevice,
    sim:    PhysarumSimulation,
    config: PhysarumEdgeBridgeConfig,
  ) {
    this.device = device;
    this.sim    = sim;
    this.config = {
      ...DEFAULTS,
      ...config,
    };
    this.trailWidth  = sim.width;
    this.trailHeight = sim.height;
  }

  // ─── Public API ─────────────────────────────────────────────────────────────

  /**
   * Trigger an async GPU→CPU readback of the physarum trail texture.
   * Throttled to every `readbackInterval` frames.  Non-blocking — the
   * previous readback's data remains available while a new one is in flight.
   *
   * @param frameCount  Current frame number (monotonically increasing).
   */
  async readback(frameCount: number): Promise<void> {
    if (frameCount - this.lastReadbackFrame < this.config.readbackInterval) return;
    if (!this.readbackReady) return;

    this.lastReadbackFrame = frameCount;
    this.readbackReady     = false;

    try {
      await this._performReadback();
    } finally {
      this.readbackReady = true;
    }
  }

  /**
   * Sample pheromone concentration at a world-space position.
   * Returns 0 if no readback data is available yet.
   */
  sampleTrail(worldX: number, worldY: number): number {
    if (!this.trailData) return 0;

    const { domainWidth, domainHeight } = this.config;
    const u = worldX / domainWidth;
    const v = worldY / domainHeight;

    return this._sampleUV(u, v);
  }

  /**
   * Compute the pheromone gradient (drift direction) at a world-space position.
   * Uses central finite differences on the readback grid.
   */
  sampleDrift(worldX: number, worldY: number): DriftVector {
    if (!this.trailData || this.config.driftStrength <= 0) {
      return { dx: 0, dy: 0, concentration: 0 };
    }

    const { domainWidth, domainHeight } = this.config;
    const u = worldX / domainWidth;
    const v = worldY / domainHeight;

    // Finite difference step in UV space (one texel)
    const du = 1.0 / this.trailWidth;
    const dv = 1.0 / this.trailHeight;

    const cx = this._sampleUV(u, v);
    const gx = this._sampleUV(u + du, v) - this._sampleUV(u - du, v);
    const gy = this._sampleUV(u, v + dv) - this._sampleUV(u, v - dv);

    const len = Math.sqrt(gx * gx + gy * gy);
    if (len < 1e-8) return { dx: 0, dy: 0, concentration: cx };

    const s = this.config.driftStrength;
    return {
      dx: (gx / len) * s,
      dy: (gy / len) * s,
      concentration: cx,
    };
  }

  /**
   * Return a speed modulation function suitable for patching into
   * EdgeFlowRenderer's particle update loop.
   *
   * The returned function maps (worldX, worldY, baseSpeed) → modulatedSpeed.
   */
  speedModulator(): SpeedModulatorFn {
    const { trailCeil, minSpeedFactor, maxSpeedFactor } = this.config;

    return (x: number, y: number, baseSpeed: number): number => {
      const c = this.sampleTrail(x, y);
      const t = Math.min(c / trailCeil, 1.0);
      const factor = minSpeedFactor + t * (maxSpeedFactor - minSpeedFactor);
      return baseSpeed * factor;
    };
  }

  /**
   * Queue a pheromone deposit at a world-space position (called on edge-flow
   * particle arrival).  Deposits are batched and flushed to the GPU on the
   * next `flushDeposits()` call.
   */
  queueDeposit(worldX: number, worldY: number, amount?: number): void {
    if (!this.config.depositFeedback) return;

    const { domainWidth, domainHeight } = this.config;
    const tx = Math.floor((worldX / domainWidth)  * this.trailWidth)  % this.trailWidth;
    const ty = Math.floor((worldY / domainHeight) * this.trailHeight) % this.trailHeight;

    this.pendingDeposits.push({
      tx: ((tx % this.trailWidth)  + this.trailWidth)  % this.trailWidth,
      ty: ((ty % this.trailHeight) + this.trailHeight) % this.trailHeight,
      amount: amount ?? this.config.depositAmount,
    });
  }

  /**
   * Flush pending deposits to the GPU trail map.
   * Call once per frame after all edge-flow arrivals have been queued.
   *
   * Implementation: writes deposit data into the trail texture via
   * writeTexture (CPU→GPU copy of modified pixels).  This is simpler
   * than a compute pass and sufficient for the low arrival rate
   * (typically < 50 deposits/frame).
   */
  flushDeposits(): void {
    if (this.pendingDeposits.length === 0) return;
    if (!this.trailData) {
      this.pendingDeposits.length = 0;
      return;
    }

    // Accumulate deposits into a sparse set of pixel writes
    const depositMap = new Map<number, number>();
    for (const d of this.pendingDeposits) {
      const key = d.ty * this.trailWidth + d.tx;
      depositMap.set(key, (depositMap.get(key) ?? 0) + d.amount);
    }
    this.pendingDeposits.length = 0;

    // Write each modified pixel to the trail texture
    const trailTex = this.sim.getTrailTexture();
    for (const [key, amount] of depositMap) {
      const tx = key % this.trailWidth;
      const ty = Math.floor(key / this.trailWidth);

      // Read current value from CPU cache, add deposit
      const current = this.trailData[key] ?? 0;
      const updated = current + amount;

      // Update CPU cache
      this.trailData[key] = updated;

      // Write single pixel to GPU
      const pixel = new Float32Array([updated]);
      this.device.queue.writeTexture(
        { texture: trailTex, origin: { x: tx, y: ty } },
        pixel.buffer,
        { bytesPerRow: 4 },
        { width: 1, height: 1 },
      );
    }
  }

  /**
   * Whether readback data is available (at least one readback has completed).
   */
  get hasData(): boolean {
    return this.trailData !== null;
  }

  /**
   * Raw trail data (read-only view).  Returns null if no readback yet.
   */
  get rawTrailData(): Float32Array | null {
    return this.trailData;
  }

  /**
   * Release GPU readback resources.
   */
  destroy(): void {
    this.readbackBuf?.destroy();
    this.stagingBuf?.destroy();
    this.readbackBuf = null;
    this.stagingBuf  = null;
    this.trailData   = null;
  }

  // ─── Private: GPU readback ──────────────────────────────────────────────────

  private async _performReadback(): Promise<void> {
    const { device, sim } = this;
    const w = this.trailWidth;
    const h = this.trailHeight;

    // Bytes per row must be aligned to 256 for copyTextureToBuffer
    const bytesPerPixel = 4; // r32float
    const rawBytesPerRow = w * bytesPerPixel;
    const alignedBytesPerRow = Math.ceil(rawBytesPerRow / 256) * 256;
    const totalBytes = alignedBytesPerRow * h;

    // Lazily create readback buffer
    if (!this.readbackBuf || this.readbackBuf.size < totalBytes) {
      this.readbackBuf?.destroy();
      this.stagingBuf?.destroy();

      this.readbackBuf = device.createBuffer({
        label: 'physarum-edge-readback',
        size:  totalBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }

    // Copy trail texture → readback buffer
    const enc = device.createCommandEncoder({ label: 'physarum-readback-copy' });
    enc.copyTextureToBuffer(
      { texture: sim.getTrailTexture() },
      { buffer: this.readbackBuf, bytesPerRow: alignedBytesPerRow, rowsPerImage: h },
      { width: w, height: h },
    );
    device.queue.submit([enc.finish()]);

    // Map and copy to CPU
    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    const mapped = new Float32Array(this.readbackBuf.getMappedRange());

    // Compact from aligned rows to dense array
    if (!this.trailData || this.trailData.length !== w * h) {
      this.trailData = new Float32Array(w * h);
    }

    const srcStride = alignedBytesPerRow / bytesPerPixel;
    for (let row = 0; row < h; row++) {
      const srcOffset = row * srcStride;
      const dstOffset = row * w;
      this.trailData.set(mapped.subarray(srcOffset, srcOffset + w), dstOffset);
    }

    this.readbackBuf.unmap();
  }

  // ─── Private: UV sampling ───────────────────────────────────────────────────

  /**
   * Sample the CPU trail cache at normalised UV coordinates [0,1]².
   * Bilinear interpolation with toroidal wrapping.
   */
  private _sampleUV(u: number, v: number): number {
    if (!this.trailData) return 0;

    const w = this.trailWidth;
    const h = this.trailHeight;

    // Wrap to [0,1)
    const wu = ((u % 1) + 1) % 1;
    const wv = ((v % 1) + 1) % 1;

    const fx = wu * w - 0.5;
    const fy = wv * h - 0.5;

    const ix0 = ((Math.floor(fx) % w) + w) % w;
    const iy0 = ((Math.floor(fy) % h) + h) % h;
    const ix1 = (ix0 + 1) % w;
    const iy1 = (iy0 + 1) % h;

    const tx = fx - Math.floor(fx);
    const ty = fy - Math.floor(fy);

    const c00 = this.trailData[iy0 * w + ix0];
    const c10 = this.trailData[iy0 * w + ix1];
    const c01 = this.trailData[iy1 * w + ix0];
    const c11 = this.trailData[iy1 * w + ix1];

    // Bilinear
    const top    = c00 + tx * (c10 - c00);
    const bottom = c01 + tx * (c11 - c01);
    return top + ty * (bottom - top);
  }
}

// ─── Factory helper ───────────────────────────────────────────────────────────

/**
 * Create a PhysarumEdgeBridge and wire its deposit feedback to an
 * EdgeFlowRenderer's onArrival callback.
 *
 * Returns { bridge, onArrival } — pass `onArrival` into EdgeFlowRendererConfig.
 *
 * @example
 * ```ts
 * const { bridge, onArrival } = createPhysarumEdgeBridge(device, sim, {
 *   domainWidth: 800,
 *   domainHeight: 600,
 *   depositFeedback: true,
 * });
 *
 * const flow = new EdgeFlowRenderer(ctx, {
 *   edges,
 *   onArrival,
 * });
 * ```
 */
export function createPhysarumEdgeBridge(
  device: GPUDevice,
  sim:    PhysarumSimulation,
  config: PhysarumEdgeBridgeConfig & { depositFeedback: true },
): { bridge: PhysarumEdgeBridge; onArrival: (edgeId: string, targetId: string, x: number, y: number) => void };
export function createPhysarumEdgeBridge(
  device: GPUDevice,
  sim:    PhysarumSimulation,
  config: PhysarumEdgeBridgeConfig,
): { bridge: PhysarumEdgeBridge; onArrival: (edgeId: string, targetId: string, x: number, y: number) => void };
export function createPhysarumEdgeBridge(
  device: GPUDevice,
  sim:    PhysarumSimulation,
  config: PhysarumEdgeBridgeConfig,
): { bridge: PhysarumEdgeBridge; onArrival: (edgeId: string, targetId: string, x: number, y: number) => void } {
  const bridge = new PhysarumEdgeBridge(device, sim, config);

  const onArrival = (_edgeId: string, _targetId: string, x: number, y: number): void => {
    bridge.queueDeposit(x, y);
  };

  return { bridge, onArrival };
}

// ─── Defaults re-export ───────────────────────────────────────────────────────

export const PHYSARUM_EDGE_BRIDGE_DEFAULTS = {
  readbackInterval: DEFAULTS.readbackInterval,
  trailCeil:        DEFAULTS.trailCeil,
  minSpeedFactor:   DEFAULTS.minSpeedFactor,
  maxSpeedFactor:   DEFAULTS.maxSpeedFactor,
  driftStrength:    DEFAULTS.driftStrength,
  depositAmount:    DEFAULTS.depositAmount,
} as const;
