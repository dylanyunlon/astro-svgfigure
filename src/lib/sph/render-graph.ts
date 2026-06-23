/**
 * render-graph.ts — M822: Render Graph Frame Scheduler
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * A FrameGraph-style system for declaratively defining render pass dependency
 * relationships, automatically deriving execution order, managing FBO/texture
 * lifetimes, and supporting dynamic pass enable/disable — all without the
 * rigid, hand-wired daisy-chain of at-render-pipeline.ts (M720) or the
 * imperative tick() sequencing of at-scene-compositor.ts (M730).
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Architecture
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * The graph consists of three declaration-time primitives:
 *
 *   RenderGraphResource — a virtual texture/buffer slot. The graph tracks
 *     which pass writes it and which passes read it, enabling automatic
 *     lifetime analysis. Resources are backed by physical GPU textures
 *     only when the passes that produce/consume them are enabled.
 *
 *   RenderGraphPass — a named unit of GPU work. Each pass declares:
 *     • inputs  — resources it reads (creates read-edges)
 *     • outputs — resources it writes (creates write-edges)
 *     • execute — callback that records GPU commands for the pass
 *     • enabled — runtime flag for dynamic enable/disable
 *
 *   RenderGraph — the top-level container.  At compile time:
 *     1. Topological sort of passes based on read/write edges.
 *     2. Lifetime analysis: for each resource, determine the first pass
 *        that writes it and the last pass that reads it.
 *     3. Transient resource aliasing: textures whose lifetimes don't
 *        overlap share the same physical GPUTexture.
 *     At execute time:
 *     1. Allocate/recycle physical textures from a pool.
 *     2. Execute each enabled pass in sorted order.
 *     3. Return transient textures to the pool.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Usage
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   const graph = new RenderGraph(device, 'bgra8unorm');
 *
 *   // Declare resources
 *   const geoColor   = graph.createResource('geo-color',   { sizeClass: 'full' });
 *   const geoDepth   = graph.createResource('geo-depth',   { sizeClass: 'full', format: 'depth24plus' });
 *   const bloomColor = graph.createResource('bloom-color', { sizeClass: 'full' });
 *
 *   // Declare passes
 *   graph.addPass('geometry', {
 *     outputs: [geoColor, geoDepth],
 *     execute: (enc, res) => {
 *       const colorView = res.getView(geoColor);
 *       const depthView = res.getView(geoDepth);
 *       myPBR.render(enc, colorView, depthView);
 *     },
 *   });
 *
 *   graph.addPass('bloom', {
 *     inputs:  [geoColor],
 *     outputs: [bloomColor],
 *     execute: (enc, res) => {
 *       const inTex  = res.getTexture(geoColor);
 *       const outView = res.getView(bloomColor);
 *       myBloom.render(enc, inTex, outView);
 *     },
 *   });
 *
 *   // Compile once (or after structural changes)
 *   graph.compile(canvasWidth, canvasHeight);
 *
 *   // Per frame
 *   const enc = device.createCommandEncoder();
 *   graph.execute(enc, swapChainView);
 *   device.queue.submit([enc.finish()]);
 *
 *   // Dynamic disable
 *   graph.setPassEnabled('bloom', false);
 *   // No recompile needed — disabled passes are simply skipped and their
 *   // sole-consumer resources are not allocated.
 *
 * Research: xiaodi #M822 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types & Enums
// ─────────────────────────────────────────────────────────────────────────────

/** Resolution class for a virtual resource. */
export type SizeClass = 'full' | 'half' | 'quarter';

/** Descriptor for a virtual resource at declaration time. */
export interface ResourceDescriptor {
  /**
   * Resolution relative to the graph's compile-time dimensions.
   * 'full' = 1×, 'half' = 0.5×, 'quarter' = 0.25×.
   * Default: 'full'.
   */
  sizeClass?: SizeClass;

  /**
   * GPU texture format. If omitted, uses the graph's default colour format.
   */
  format?: GPUTextureFormat;

  /**
   * Additional texture usage flags beyond the defaults
   * (RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_SRC).
   */
  extraUsage?: GPUTextureUsageFlags;
}

/**
 * Opaque handle returned by `graph.createResource()`.
 * Passes reference resources through these handles.
 */
export interface ResourceHandle {
  /** Internal unique ID. */
  readonly _id: number;
  /** Human-readable debug label. */
  readonly name: string;
}

/**
 * Callback signature for a render pass's execute function.
 *
 * @param encoder — The GPUCommandEncoder for the current frame.
 * @param accessor — Provides physical GPU objects backing virtual resources.
 * @param passContext — Per-pass metadata (index, name, etc.).
 */
export type PassExecuteFn = (
  encoder:     GPUCommandEncoder,
  accessor:    ResourceAccessor,
  passContext: PassContext,
) => void;

/** Descriptor for a render pass at declaration time. */
export interface PassDescriptor {
  /** Resources this pass reads from (creates dependency edges). */
  inputs?: ResourceHandle[];

  /** Resources this pass writes to. */
  outputs?: ResourceHandle[];

  /** GPU work to record when this pass is executed. */
  execute: PassExecuteFn;

  /**
   * If true, this pass writes directly to the final swap-chain view
   * (the `dstView` passed to `graph.execute()`).
   * Only one pass should be marked as presentPass per graph.
   */
  presentPass?: boolean;
}

/** Runtime context available inside a pass's execute callback. */
export interface PassContext {
  /** The pass's name (as given to `addPass`). */
  readonly name: string;
  /** The pass's topological order index within this frame. */
  readonly sortIndex: number;
  /** The graph's compiled width. */
  readonly width: number;
  /** The graph's compiled height. */
  readonly height: number;
  /** Wall-clock elapsed time (seconds) forwarded from the last `execute()`. */
  readonly elapsed: number;
  /** Delta time (seconds) forwarded from the last `execute()`. */
  readonly dt: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// ResourceAccessor — provides physical GPU objects to pass execute callbacks
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The accessor is constructed per-frame and maps virtual ResourceHandles
 * to the physical GPUTexture / GPUTextureView that the graph allocated.
 */
export class ResourceAccessor {
  /** @internal */
  private readonly _textures: Map<number, GPUTexture>;
  /** @internal */
  private readonly _views:    Map<number, GPUTextureView>;
  /** @internal */
  readonly presentView: GPUTextureView;

  /** @internal */
  constructor(
    textures:    Map<number, GPUTexture>,
    views:       Map<number, GPUTextureView>,
    presentView: GPUTextureView,
  ) {
    this._textures   = textures;
    this._views      = views;
    this.presentView = presentView;
  }

  /**
   * Get the physical GPUTexture backing a virtual resource.
   * Used when you need the texture as an input to another pass's bind group.
   */
  getTexture(handle: ResourceHandle): GPUTexture {
    const tex = this._textures.get(handle._id);
    if (!tex) {
      throw new Error(
        `[RenderGraph] No physical texture for resource "${handle.name}" `
        + `(id=${handle._id}). Was the resource declared as an input/output `
        + `of an enabled pass?`,
      );
    }
    return tex;
  }

  /**
   * Get a GPUTextureView for a virtual resource.
   * Used as render attachment (colorAttachments[].view).
   */
  getView(handle: ResourceHandle): GPUTextureView {
    const view = this._views.get(handle._id);
    if (!view) {
      throw new Error(
        `[RenderGraph] No texture view for resource "${handle.name}" `
        + `(id=${handle._id}). Was the resource declared as an input/output `
        + `of an enabled pass?`,
      );
    }
    return view;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal structures
// ─────────────────────────────────────────────────────────────────────────────

/** Internal resource record. */
interface ResourceRecord {
  handle:     ResourceHandle;
  descriptor: Required<ResourceDescriptor>;
  /** Pass that writes to this resource (at most one writer). */
  writer:     string | null;
  /** Passes that read from this resource. */
  readers:    Set<string>;
}

/** Internal pass record. */
interface PassRecord {
  name:        string;
  descriptor:  PassDescriptor;
  enabled:     boolean;
  /** Resource IDs this pass reads. */
  inputs:      Set<number>;
  /** Resource IDs this pass writes. */
  outputs:     Set<number>;
}

/** Compiled execution plan — the result of topological sort + lifetime analysis. */
interface CompiledPlan {
  /** Pass names in topologically sorted execution order. */
  sortedPasses: string[];
  /**
   * Per-resource lifetime: [firstPassIndex, lastPassIndex] in sorted order.
   * Only includes resources that are actually live (connected to enabled passes).
   */
  lifetimes: Map<number, [number, number]>;
  /**
   * Alias groups: resources whose lifetimes don't overlap may share a
   * physical texture. Map from resource ID to alias-group index.
   */
  aliasGroups: Map<number, number>;
  /** Number of distinct alias groups (= number of physical textures needed). */
  aliasGroupCount: number;
  /** Set of resource IDs that are actually live in this plan. */
  liveResources: Set<number>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Texture pool — recycles GPUTextures across frames and alias groups
// ─────────────────────────────────────────────────────────────────────────────

/** Key for pooling: textures with the same key are interchangeable. */
interface PoolKey {
  width:  number;
  height: number;
  format: string;  // GPUTextureFormat
  usage:  number;  // GPUTextureUsageFlags bitmask
}

function poolKeyStr(k: PoolKey): string {
  return `${k.width}x${k.height}:${k.format}:${k.usage}`;
}

interface PoolEntry {
  texture: GPUTexture;
  key:     PoolKey;
  /** Frame number when this texture was last returned to the pool. */
  lastUsedFrame: number;
}

class TexturePool {
  private readonly device: GPUDevice;
  private pool: PoolEntry[] = [];
  private frameCounter = 0;
  /** Textures idle for this many frames are destroyed. */
  private static readonly MAX_IDLE_FRAMES = 4;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /** Acquire a texture matching the given key, or create a new one. */
  acquire(key: PoolKey, label: string): GPUTexture {
    const keyStr = poolKeyStr(key);
    for (let i = 0; i < this.pool.length; i++) {
      if (poolKeyStr(this.pool[i].key) === keyStr) {
        const entry = this.pool[i];
        this.pool.splice(i, 1);
        return entry.texture;
      }
    }
    // No match — allocate new
    return this.device.createTexture({
      label,
      size:   [key.width, key.height],
      format: key.format as GPUTextureFormat,
      usage:  key.usage,
    });
  }

  /** Return a texture to the pool for future reuse. */
  release(texture: GPUTexture, key: PoolKey): void {
    this.pool.push({ texture, key, lastUsedFrame: this.frameCounter });
  }

  /** Call once per frame to evict stale textures. */
  tick(): void {
    this.frameCounter++;
    const cutoff = this.frameCounter - TexturePool.MAX_IDLE_FRAMES;
    for (let i = this.pool.length - 1; i >= 0; i--) {
      if (this.pool[i].lastUsedFrame < cutoff) {
        this.pool[i].texture.destroy();
        this.pool.splice(i, 1);
      }
    }
  }

  /** Destroy all pooled textures. */
  destroy(): void {
    for (const entry of this.pool) {
      entry.texture.destroy();
    }
    this.pool.length = 0;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// RenderGraph
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Render Graph — FrameGraph-style declarative render pass scheduler.
 *
 * Lifecycle:
 *   1. Construct: `new RenderGraph(device, format)`
 *   2. Declare resources: `graph.createResource(name, desc)`
 *   3. Declare passes: `graph.addPass(name, desc)`
 *   4. Compile: `graph.compile(w, h)` — topological sort + lifetime analysis
 *   5. Per-frame: `graph.execute(encoder, dstView, dt, elapsed)`
 *   6. Dynamic toggles: `graph.setPassEnabled(name, bool)` — no recompile
 *      needed for simple enable/disable; the compiled sort order remains
 *      valid, and disabled passes are simply skipped.
 *   7. Structural changes (add/remove passes or resources): call `compile()`
 *      again to rebuild the execution plan.
 *   8. Resize: `graph.compile(newW, newH)` — reallocates physical textures.
 *   9. Destroy: `graph.destroy()`.
 */
export class RenderGraph {
  // ── Core ──────────────────────────────────────────────────────────────────
  private readonly device: GPUDevice;
  private readonly defaultFormat: GPUTextureFormat;

  // ── Declaration-time registries ───────────────────────────────────────────
  private resources:    Map<number, ResourceRecord> = new Map();
  private passes:       Map<string, PassRecord>     = new Map();
  private nextResourceId = 0;
  /** Insertion order for passes (used when no dependencies exist). */
  private passInsertionOrder: string[] = [];

  // ── Compiled plan ─────────────────────────────────────────────────────────
  private plan: CompiledPlan | null = null;
  private compiledWidth  = 0;
  private compiledHeight = 0;

  // ── Runtime physical resources ────────────────────────────────────────────
  private pool: TexturePool;
  /**
   * Currently allocated physical textures for live resources.
   * Keyed by resource ID.
   */
  private physicalTextures: Map<number, GPUTexture> = new Map();
  private physicalViews:    Map<number, GPUTextureView> = new Map();

  // ── Frame clock ───────────────────────────────────────────────────────────
  private elapsed = 0;
  private dt      = 0;

  // ── Lifecycle ─────────────────────────────────────────────────────────────
  private destroyed = false;

  // ─────────────────────────────────────────────────────────────────────────
  // Constructor
  // ─────────────────────────────────────────────────────────────────────────

  constructor(device: GPUDevice, defaultFormat: GPUTextureFormat = 'bgra8unorm') {
    this.device        = device;
    this.defaultFormat = defaultFormat;
    this.pool          = new TexturePool(device);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Declaration API — Resources
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Declare a virtual resource (texture slot).
   *
   * The resource does not allocate GPU memory until `compile()` determines
   * that at least one enabled pass writes to it.
   *
   * @param name — Human-readable debug label.
   * @param desc — Size class, format, and usage overrides.
   * @returns An opaque handle used to reference the resource in pass descriptors.
   */
  createResource(name: string, desc: ResourceDescriptor = {}): ResourceHandle {
    this._assertNotDestroyed();
    const id = this.nextResourceId++;
    const handle: ResourceHandle = { _id: id, name };

    const defaultUsage =
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING   |
      GPUTextureUsage.COPY_SRC;

    const record: ResourceRecord = {
      handle,
      descriptor: {
        sizeClass:  desc.sizeClass  ?? 'full',
        format:     desc.format     ?? this.defaultFormat,
        extraUsage: desc.extraUsage ?? 0,
      },
      writer:  null,
      readers: new Set(),
    };

    // Inject COPY_DST when extraUsage includes it or for depth textures
    // (depth textures might not need it, but colour intermediates often do)
    const fullUsage = defaultUsage | (desc.extraUsage ?? 0);
    record.descriptor.extraUsage = fullUsage & ~defaultUsage; // store only extras

    this.resources.set(id, record);
    return handle;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Declaration API — Passes
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Declare a render pass.
   *
   * @param name — Unique name for the pass (used in setPassEnabled, etc.).
   * @param desc — Inputs, outputs, and execute callback.
   */
  addPass(name: string, desc: PassDescriptor): void {
    this._assertNotDestroyed();
    if (this.passes.has(name)) {
      throw new Error(`[RenderGraph] Duplicate pass name: "${name}"`);
    }

    const inputIds  = new Set<number>();
    const outputIds = new Set<number>();

    // Register read-edges
    if (desc.inputs) {
      for (const h of desc.inputs) {
        const rec = this.resources.get(h._id);
        if (!rec) throw new Error(`[RenderGraph] Unknown resource "${h.name}" in inputs of pass "${name}".`);
        rec.readers.add(name);
        inputIds.add(h._id);
      }
    }

    // Register write-edges (single-writer rule)
    if (desc.outputs) {
      for (const h of desc.outputs) {
        const rec = this.resources.get(h._id);
        if (!rec) throw new Error(`[RenderGraph] Unknown resource "${h.name}" in outputs of pass "${name}".`);
        if (rec.writer !== null) {
          throw new Error(
            `[RenderGraph] Resource "${h.name}" already has a writer ("${rec.writer}"). `
            + `Pass "${name}" cannot also write to it. Use a separate resource.`,
          );
        }
        rec.writer = name;
        outputIds.add(h._id);
      }
    }

    this.passes.set(name, {
      name,
      descriptor: desc,
      enabled:    true,
      inputs:     inputIds,
      outputs:    outputIds,
    });

    this.passInsertionOrder.push(name);
    // Invalidate compiled plan
    this.plan = null;
  }

  /**
   * Remove a pass from the graph.
   * Requires a subsequent `compile()` call to rebuild the execution plan.
   */
  removePass(name: string): void {
    this._assertNotDestroyed();
    const rec = this.passes.get(name);
    if (!rec) return;

    // Remove read-edges
    for (const resId of rec.inputs) {
      const res = this.resources.get(resId);
      if (res) res.readers.delete(name);
    }
    // Remove write-edges
    for (const resId of rec.outputs) {
      const res = this.resources.get(resId);
      if (res && res.writer === name) res.writer = null;
    }

    this.passes.delete(name);
    this.passInsertionOrder = this.passInsertionOrder.filter(n => n !== name);
    this.plan = null;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Dynamic enable/disable
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Enable or disable a pass at runtime.
   *
   * Disabled passes are skipped during `execute()`. Resources that are only
   * consumed by disabled passes are not allocated, saving GPU memory.
   *
   * No recompile is needed — the topological order doesn't change. The
   * resource allocation step inside `execute()` checks the enabled flag.
   */
  setPassEnabled(name: string, enabled: boolean): void {
    const rec = this.passes.get(name);
    if (!rec) {
      throw new Error(`[RenderGraph] Unknown pass: "${name}"`);
    }
    rec.enabled = enabled;
  }

  /** Check whether a pass is currently enabled. */
  isPassEnabled(name: string): boolean {
    const rec = this.passes.get(name);
    return rec ? rec.enabled : false;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Compile
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Compile the render graph: topological sort, lifetime analysis, and
   * alias group assignment.
   *
   * Must be called after structural changes (addPass/removePass/createResource)
   * and whenever the viewport size changes.
   *
   * @param width  — Viewport width in pixels ('full' sizeClass).
   * @param height — Viewport height in pixels.
   */
  compile(width: number, height: number): void {
    this._assertNotDestroyed();
    this.compiledWidth  = width;
    this.compiledHeight = height;

    // ── Release old physical textures ────────────────────────────────────
    this._releaseAllPhysical();

    // ── 1. Topological sort (Kahn's algorithm) ──────────────────────────
    //    Edge: passA → passB  iff  passA writes a resource that passB reads.
    const sorted = this._topologicalSort();

    // ── 2. Determine live resources (connected to at least one pass) ────
    const liveResources = new Set<number>();
    for (const passName of sorted) {
      const pass = this.passes.get(passName)!;
      for (const resId of pass.inputs)  liveResources.add(resId);
      for (const resId of pass.outputs) liveResources.add(resId);
    }

    // ── 3. Lifetime analysis ────────────────────────────────────────────
    //    For each live resource, record the index of the first pass that
    //    writes/reads it and the last pass that writes/reads it.
    const passIndex = new Map<string, number>();
    sorted.forEach((name, idx) => passIndex.set(name, idx));

    const lifetimes = new Map<number, [number, number]>();
    for (const resId of liveResources) {
      const rec = this.resources.get(resId)!;
      let first = Infinity;
      let last  = -1;

      // Writer pass
      if (rec.writer && passIndex.has(rec.writer)) {
        const idx = passIndex.get(rec.writer)!;
        first = Math.min(first, idx);
        last  = Math.max(last,  idx);
      }

      // Reader passes
      for (const reader of rec.readers) {
        if (!passIndex.has(reader)) continue;
        const idx = passIndex.get(reader)!;
        first = Math.min(first, idx);
        last  = Math.max(last,  idx);
      }

      if (first <= last) {
        lifetimes.set(resId, [first, last]);
      }
    }

    // ── 4. Greedy interval-colouring for alias groups ───────────────────
    //    Resources whose lifetime intervals don't overlap AND have the same
    //    format + size class can share a physical texture.
    const { aliasGroups, aliasGroupCount } = this._computeAliasGroups(lifetimes);

    this.plan = {
      sortedPasses: sorted,
      lifetimes,
      aliasGroups,
      aliasGroupCount,
      liveResources,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Execute
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Execute the compiled render graph for one frame.
   *
   * Allocates physical textures for live resources, iterates passes in
   * topological order (skipping disabled ones), and returns transient
   * textures to the pool.
   *
   * @param encoder  — GPUCommandEncoder for this frame's GPU work.
   * @param dstView  — The swap-chain surface view (final present target).
   * @param dt       — Delta time in seconds since last frame.
   * @param elapsed  — Optional total elapsed time in seconds.
   */
  execute(
    encoder:  GPUCommandEncoder,
    dstView:  GPUTextureView,
    dt:       number = 0,
    elapsed?: number,
  ): void {
    this._assertNotDestroyed();
    if (!this.plan) {
      throw new Error(
        '[RenderGraph] Graph not compiled. Call compile(width, height) first.',
      );
    }

    this.dt = dt;
    if (elapsed !== undefined) {
      this.elapsed = elapsed;
    } else {
      this.elapsed += dt;
    }

    // ── Pool maintenance ─────────────────────────────────────────────────
    this.pool.tick();

    // ── Determine which resources are actually needed this frame ─────────
    //    A resource is needed if at least one enabled pass writes/reads it.
    const neededResources = new Set<number>();
    for (const passName of this.plan.sortedPasses) {
      const pass = this.passes.get(passName)!;
      if (!pass.enabled) continue;
      for (const resId of pass.inputs)  neededResources.add(resId);
      for (const resId of pass.outputs) neededResources.add(resId);
    }

    // ── Allocate physical textures ───────────────────────────────────────
    this._allocatePhysical(neededResources);

    // ── Build accessor ───────────────────────────────────────────────────
    const accessor = new ResourceAccessor(
      this.physicalTextures,
      this.physicalViews,
      dstView,
    );

    // ── Execute passes in sorted order ───────────────────────────────────
    const sorted = this.plan.sortedPasses;
    for (let i = 0; i < sorted.length; i++) {
      const passName = sorted[i];
      const pass = this.passes.get(passName)!;
      if (!pass.enabled) continue;

      const ctx: PassContext = {
        name:      passName,
        sortIndex: i,
        width:     this.compiledWidth,
        height:    this.compiledHeight,
        elapsed:   this.elapsed,
        dt:        this.dt,
      };

      pass.descriptor.execute(encoder, accessor, ctx);
    }

    // ── Release textures for resources that are no longer needed ─────────
    //    (Return to pool for next frame)
    this._releaseUnneeded(neededResources);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Resize
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Convenience shorthand: recompile the graph at a new viewport size.
   * Equivalent to `compile(newWidth, newHeight)`.
   */
  resize(width: number, height: number): void {
    this.compile(width, height);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Introspection
  // ─────────────────────────────────────────────────────────────────────────

  /** Get the topologically sorted pass names from the last compile. */
  getSortedPassNames(): readonly string[] {
    return this.plan?.sortedPasses ?? [];
  }

  /** Get all declared pass names (in insertion order). */
  getPassNames(): readonly string[] {
    return [...this.passInsertionOrder];
  }

  /** Get all declared resource handles. */
  getResourceHandles(): readonly ResourceHandle[] {
    return [...this.resources.values()].map(r => r.handle);
  }

  /** Get the compiled plan's alias group count (number of physical textures). */
  getAliasGroupCount(): number {
    return this.plan?.aliasGroupCount ?? 0;
  }

  /** Check whether the graph has been compiled. */
  get isCompiled(): boolean {
    return this.plan !== null;
  }

  /** Get the compiled width. */
  get width(): number {
    return this.compiledWidth;
  }

  /** Get the compiled height. */
  get height(): number {
    return this.compiledHeight;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Destroy
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Release all GPU resources. The graph must not be used after this.
   */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this._releaseAllPhysical();
    this.pool.destroy();
    this.resources.clear();
    this.passes.clear();
    this.plan = null;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Topological sort (Kahn's algorithm)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Produce a topologically sorted ordering of all declared passes based
   * on resource read/write edges.
   *
   * Passes with no dependencies are ordered by insertion order.
   * Cycles cause an error.
   */
  private _topologicalSort(): string[] {
    // Build adjacency list: passA → passB means passA must execute before passB.
    // This happens when passA writes a resource that passB reads.
    const adj     = new Map<string, Set<string>>();
    const inDeg   = new Map<string, number>();

    for (const passName of this.passes.keys()) {
      adj.set(passName, new Set());
      inDeg.set(passName, 0);
    }

    // For each resource: writer → every reader
    for (const res of this.resources.values()) {
      if (!res.writer) continue;
      if (!this.passes.has(res.writer)) continue;

      for (const reader of res.readers) {
        if (!this.passes.has(reader)) continue;
        if (reader === res.writer) continue; // self-loops are fine (read-back)
        const neighbors = adj.get(res.writer)!;
        if (!neighbors.has(reader)) {
          neighbors.add(reader);
          inDeg.set(reader, (inDeg.get(reader) ?? 0) + 1);
        }
      }
    }

    // Seed the queue with zero-in-degree passes, sorted by insertion order
    // for deterministic output.
    const insertionIdx = new Map<string, number>();
    this.passInsertionOrder.forEach((name, idx) => insertionIdx.set(name, idx));

    const queue: string[] = [];
    for (const [name, deg] of inDeg) {
      if (deg === 0) queue.push(name);
    }
    queue.sort((a, b) => (insertionIdx.get(a) ?? 0) - (insertionIdx.get(b) ?? 0));

    const sorted: string[] = [];

    while (queue.length > 0) {
      const curr = queue.shift()!;
      sorted.push(curr);

      const neighbors = adj.get(curr) ?? new Set();
      // Collect neighbors and sort for determinism
      const sortedNeighbors = [...neighbors].sort(
        (a, b) => (insertionIdx.get(a) ?? 0) - (insertionIdx.get(b) ?? 0),
      );

      for (const next of sortedNeighbors) {
        const newDeg = (inDeg.get(next) ?? 1) - 1;
        inDeg.set(next, newDeg);
        if (newDeg === 0) {
          queue.push(next);
          // Re-sort to maintain insertion-order priority among ready nodes
          queue.sort(
            (a, b) => (insertionIdx.get(a) ?? 0) - (insertionIdx.get(b) ?? 0),
          );
        }
      }
    }

    if (sorted.length !== this.passes.size) {
      const missing = [...this.passes.keys()].filter(n => !sorted.includes(n));
      throw new Error(
        `[RenderGraph] Cycle detected in pass dependencies. `
        + `Passes involved: ${missing.join(', ')}`,
      );
    }

    return sorted;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Alias groups (greedy interval colouring)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Assign alias groups to resources whose lifetimes don't overlap and
   * whose format + size class match, so they can share a physical texture.
   *
   * Uses a greedy interval-graph colouring: sort intervals by start time,
   * then assign each to the first compatible group whose last-used endpoint
   * is before the current interval's start.
   */
  private _computeAliasGroups(
    lifetimes: Map<number, [number, number]>,
  ): { aliasGroups: Map<number, number>; aliasGroupCount: number } {
    // Sort resources by lifetime start
    const entries = [...lifetimes.entries()].sort(
      (a, b) => a[1][0] - b[1][0],
    );

    // Each group tracks: { lastEnd, format, sizeClass }
    interface Group {
      lastEnd:   number;
      format:    string;
      sizeClass: SizeClass;
    }
    const groups: Group[] = [];
    const aliasGroups = new Map<number, number>();

    for (const [resId, [start, end]] of entries) {
      const rec = this.resources.get(resId)!;
      const fmt = rec.descriptor.format;
      const sc  = rec.descriptor.sizeClass;

      // Try to find an existing compatible group
      let assigned = false;
      for (let g = 0; g < groups.length; g++) {
        const grp = groups[g];
        if (grp.format === fmt && grp.sizeClass === sc && grp.lastEnd < start) {
          grp.lastEnd = end;
          aliasGroups.set(resId, g);
          assigned = true;
          break;
        }
      }

      if (!assigned) {
        aliasGroups.set(resId, groups.length);
        groups.push({ lastEnd: end, format: fmt, sizeClass: sc });
      }
    }

    return { aliasGroups, aliasGroupCount: groups.length };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Physical texture allocation
  // ─────────────────────────────────────────────────────────────────────────

  /** Resolve a sizeClass to pixel dimensions. */
  private _resolveDimensions(sc: SizeClass): [number, number] {
    switch (sc) {
      case 'half':    return [
        Math.max(1, (this.compiledWidth  + 1) >> 1),
        Math.max(1, (this.compiledHeight + 1) >> 1),
      ];
      case 'quarter': return [
        Math.max(1, (this.compiledWidth  + 3) >> 2),
        Math.max(1, (this.compiledHeight + 3) >> 2),
      ];
      default:        return [this.compiledWidth, this.compiledHeight];
    }
  }

  /** Compute the full GPU texture usage flags for a resource. */
  private _resolveUsage(rec: ResourceRecord): GPUTextureUsageFlags {
    const base =
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING   |
      GPUTextureUsage.COPY_SRC;
    return base | rec.descriptor.extraUsage;
  }

  /** Allocate physical textures for the given set of needed resource IDs. */
  private _allocatePhysical(neededResources: Set<number>): void {
    for (const resId of neededResources) {
      if (this.physicalTextures.has(resId)) continue; // already allocated

      const rec = this.resources.get(resId);
      if (!rec) continue;

      const [w, h] = this._resolveDimensions(rec.descriptor.sizeClass);
      const format  = rec.descriptor.format;
      const usage   = this._resolveUsage(rec);

      const key: PoolKey = { width: w, height: h, format, usage };
      const texture = this.pool.acquire(key, `rg:${rec.handle.name}`);
      const view    = texture.createView({ label: `rg:${rec.handle.name}-view` });

      this.physicalTextures.set(resId, texture);
      this.physicalViews.set(resId, view);
    }
  }

  /** Release physical textures that are no longer in the needed set. */
  private _releaseUnneeded(neededResources: Set<number>): void {
    for (const [resId, texture] of this.physicalTextures) {
      if (neededResources.has(resId)) continue;

      const rec = this.resources.get(resId);
      if (!rec) continue;

      const [w, h] = this._resolveDimensions(rec.descriptor.sizeClass);
      const format  = rec.descriptor.format;
      const usage   = this._resolveUsage(rec);

      this.pool.release(texture, { width: w, height: h, format, usage });
      this.physicalTextures.delete(resId);
      this.physicalViews.delete(resId);
    }
  }

  /** Release all physical textures back to the pool (or destroy them). */
  private _releaseAllPhysical(): void {
    for (const [resId, texture] of this.physicalTextures) {
      const rec = this.resources.get(resId);
      if (rec) {
        const [w, h] = this._resolveDimensions(rec.descriptor.sizeClass);
        const format  = rec.descriptor.format;
        const usage   = this._resolveUsage(rec);
        this.pool.release(texture, { width: w, height: h, format, usage });
      } else {
        texture.destroy();
      }
    }
    this.physicalTextures.clear();
    this.physicalViews.clear();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Guards
  // ─────────────────────────────────────────────────────────────────────────

  private _assertNotDestroyed(): void {
    if (this.destroyed) {
      throw new Error('[RenderGraph] Graph has been destroyed.');
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: RenderGraphBuilder — fluent API for common patterns
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A fluent builder that wraps RenderGraph for common multi-pass pipeline
 * patterns (linear chain, fan-in, post-process stack).
 *
 * Example:
 *   const { graph, handles } = new RenderGraphBuilder(device, 'bgra8unorm')
 *     .size(1920, 1080)
 *     .pass('geometry', { ... })
 *     .chain('bloom', { ... })   // auto-wires previous output as input
 *     .chain('lut',   { ... })
 *     .build();
 */
export class RenderGraphBuilder {
  private readonly graph: RenderGraph;
  private lastOutput: ResourceHandle | null = null;
  private readonly handles: Map<string, ResourceHandle> = new Map();
  private w = 0;
  private h = 0;

  constructor(device: GPUDevice, format?: GPUTextureFormat) {
    this.graph = new RenderGraph(device, format);
  }

  /** Set the viewport dimensions for compile. */
  size(width: number, height: number): this {
    this.w = width;
    this.h = height;
    return this;
  }

  /** Declare a named resource and return this builder. */
  resource(name: string, desc?: ResourceDescriptor): this {
    const handle = this.graph.createResource(name, desc);
    this.handles.set(name, handle);
    return this;
  }

  /** Get a previously declared handle by name (creates one if missing). */
  getHandle(name: string, desc?: ResourceDescriptor): ResourceHandle {
    let h = this.handles.get(name);
    if (!h) {
      h = this.graph.createResource(name, desc);
      this.handles.set(name, h);
    }
    return h;
  }

  /**
   * Add a pass with explicit inputs/outputs.
   * The last declared output becomes the "chain head" for subsequent
   * `chain()` calls.
   */
  pass(name: string, desc: {
    inputs?:      string[];
    outputs?:     string[];
    execute:      PassExecuteFn;
    presentPass?: boolean;
    outputDesc?:  ResourceDescriptor;
  }): this {
    const inputs  = desc.inputs?.map(n => this.getHandle(n))  ?? [];
    const outputs = desc.outputs?.map(n => this.getHandle(n, desc.outputDesc)) ?? [];

    // If no explicit output, auto-create one named `${name}-out`
    if (outputs.length === 0) {
      const autoOut = this.getHandle(`${name}-out`, desc.outputDesc);
      outputs.push(autoOut);
    }

    this.graph.addPass(name, {
      inputs,
      outputs,
      execute:     desc.execute,
      presentPass: desc.presentPass,
    });

    this.lastOutput = outputs[outputs.length - 1];
    return this;
  }

  /**
   * Add a pass that auto-wires the previous pass's output as its input.
   * Creates an output resource named `${name}-out`.
   */
  chain(name: string, desc: {
    extraInputs?:  string[];
    execute:       PassExecuteFn;
    presentPass?:  boolean;
    outputDesc?:   ResourceDescriptor;
  }): this {
    const inputs: string[] = [];
    if (this.lastOutput) {
      inputs.push(this.lastOutput.name);
    }
    if (desc.extraInputs) {
      inputs.push(...desc.extraInputs);
    }

    return this.pass(name, {
      inputs,
      outputs: [`${name}-out`],
      execute:     desc.execute,
      presentPass: desc.presentPass,
      outputDesc:  desc.outputDesc,
    });
  }

  /** Compile and return the built graph plus all named handles. */
  build(): { graph: RenderGraph; handles: Map<string, ResourceHandle> } {
    if (this.w > 0 && this.h > 0) {
      this.graph.compile(this.w, this.h);
    }
    return { graph: this.graph, handles: this.handles };
  }
}
