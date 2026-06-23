/**
 * at-world-integrator.ts — M833: AT World Integrator
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Unifies every AT rendering module into a single, graph-driven world
 * coordinator.  Where at-scene-compositor.ts (M730) hard-wires a linear
 * FBO daisy-chain and at-render-pipeline.ts (M720) bakes a fixed pass
 * sequence, the World Integrator delegates execution ordering to the
 * RenderGraph (M822) while providing:
 *
 *   • Automatic AT module registration — each module becomes a typed
 *     ATWorldNode with declared z-depth, resource inputs/outputs, and a
 *     standardised init / tick / dispose lifecycle.
 *
 *   • Z-depth sorted rendering — nodes are bucketed into ordered layers
 *     (Background → Fluid → Geometry → Particle → Surface → Volumetric →
 *     PostProcess → Composite → HUD) and within each layer sorted by the
 *     node's z-depth value.  The sorted order is projected onto the
 *     RenderGraph as pass dependencies so that the topological sort
 *     respects painter's-algorithm ordering.
 *
 *   • Unified init / tick / dispose — a single async init() boots every
 *     registered module; tick(dt) advances physics, compute, and render
 *     passes in graph order; dispose() tears everything down.
 *
 *   • Dynamic add / remove / enable / disable of modules at runtime
 *     without a full graph recompile (enable/disable) or with a
 *     lightweight incremental recompile (add/remove).
 *
 *   • Cell management — addCell / removeCell delegated to the appropriate
 *     material modules (PBR, Matcap, Gem) with per-cell z-depth offsets
 *     for correct front-to-back or back-to-front compositing.
 *
 *   • SPH physics coupling — optional SPHWorldView drives material
 *     modulation, Navier-Stokes dye injection, and particle emission
 *     rates, exactly as in ATSceneCompositor but now expressed as
 *     graph node data-flow rather than imperative sequencing.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Architecture
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * ATWorldIntegrator owns:
 *   ┌─────────────────────┐      ┌────────────────┐
 *   │  ATWorldNode[]      │─────▶│  RenderGraph    │
 *   │  (sorted by zDepth) │      │  (M822)         │
 *   └─────────────────────┘      └────────────────┘
 *          │                             │
 *          │  per-node init/tick/dispose  │  per-pass execute()
 *          ▼                             ▼
 *   ┌────────────┐  ┌────────────┐  ┌────────────┐
 *   │ NS Fluid   │  │ Flower     │  │ Bloom      │  ...
 *   │ (compute)  │  │ (compute+  │  │ (render)   │
 *   │            │  │  render)   │  │            │
 *   └────────────┘  └────────────┘  └────────────┘
 *
 * Each ATWorldNode wraps one AT module and exposes:
 *   • nodeId       — unique string identifier
 *   • layer        — RenderLayer enum bucket
 *   • zDepth       — numeric sort key within the layer (lower = farther)
 *   • resources    — declared input/output ResourceHandle IDs
 *   • init()       — one-time GPU resource allocation
 *   • tick()       — per-frame update (compute dispatches, uniform uploads)
 *   • render()     — per-frame render pass recording
 *   • dispose()    — GPU resource teardown
 *   • enabled      — runtime toggle (maps to graph.setPassEnabled)
 *
 * The integrator maps each node to one or two RenderGraph passes
 * (compute + render, or render-only) and wires resource handles
 * automatically based on the node's declared inputs/outputs.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Usage
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   const world = new ATWorldIntegrator();
 *   await world.init(device, canvas, {
 *     modules: ['navier-stokes', 'flower-particle', 'spline-particle',
 *               'water-surface', 'volumetric-light', 'bloom', 'pbr-material'],
 *   });
 *
 *   world.addCell('cell-0', 'attention', { x: 0, y: 0, w: 100, h: 100 });
 *
 *   function frame(dt: number) {
 *     world.tick(dt, sphWorld);
 *     requestAnimationFrame(() => frame(1/60));
 *   }
 *   frame(1/60);
 *
 *   // dynamic toggle:
 *   world.setModuleEnabled('bloom', false);
 *
 *   // cleanup:
 *   world.dispose();
 *
 * Research: xiaodi #M833 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Imports — AT modules
// ─────────────────────────────────────────────────────────────────────────────

import {
  ATBloomPostProcess,
  type ATBloomParams,
} from './at-bloom-postprocess.ts';

import {
  ATFlowerParticleRenderer,
  type FlowerEdgeSpline,
} from './at-flower-particle.ts';

import {
  ATGemMaterial,
} from './at-gem-material.ts';

import {
  ATSplineParticleLife,
  type EdgeSpline,
} from './at-spline-particle.ts';

import {
  ATWaterSurface,
  type ATWaterSurfaceConfig,
} from './at-water-surface.ts';

import {
  NavierStokesFluid,
  type NavierStokesSplat,
} from './at-navier-stokes.ts';

import {
  ATVolumetricLight,
  type ATVolumetricLightParams,
} from './at-volumetric-light.ts';

import {
  ATPBRMaterial,
  ATMatcapFresnel,
  type PBRParams,
  type MatcapParams,
} from './at-pbr-material.ts';

import {
  ParticleCompositor,
  LayerType,
  type LayerDescriptor,
} from './particle-compositor.ts';

import {
  getSpeciesShaderConfig,
  type SpeciesShaderConfig,
  type MaterialType,
} from './species-shader-registry.ts';

import {
  getCellMaterial,
  type CellSpecies,
  type SpeciesMaterialDef,
} from './cell-material-system.ts';

// ─────────────────────────────────────────────────────────────────────────────
// Imports — Render Graph (M822)
// ─────────────────────────────────────────────────────────────────────────────

import {
  RenderGraph,
  type ResourceHandle,
  type ResourceDescriptor,
  type PassExecuteFn,
  type PassContext,
  type ResourceAccessor,
} from './render-graph.ts';

// ─────────────────────────────────────────────────────────────────────────────
// Render Layer Enum — ordered buckets for z-depth sorting
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Render layers define the major ordering buckets for the compositing
 * pipeline.  Within each layer, individual nodes are sorted by their
 * numeric zDepth value (ascending = farther from camera first).
 *
 * The layer ordering implements a standard deferred-style pipeline:
 *   Background → Fluid (simulation) → Geometry (opaque) → Particle →
 *   Surface (translucent) → Volumetric → PostProcess → Composite → HUD
 */
export const enum RenderLayer {
  /** Sky, environment maps, far-field background. */
  Background   = 0,
  /** Fluid simulation compute passes (Navier-Stokes). */
  Fluid        = 100,
  /** Opaque geometry: PBR materials, gem materials. */
  Geometry     = 200,
  /** Particle compute + render: flower, spline, sparks. */
  Particle     = 300,
  /** Translucent surfaces: water, glass, jellyfish. */
  Surface      = 400,
  /** Screen-space volumetric effects: god rays, fog. */
  Volumetric   = 500,
  /** Full-screen post-processing: bloom, FXAA, LUT. */
  PostProcess  = 600,
  /** Final composite to swap-chain. */
  Composite    = 700,
  /** Overlay / HUD elements (debug, UI). */
  HUD          = 800,
}

// ─────────────────────────────────────────────────────────────────────────────
// ATWorldNode — the uniform interface every AT module adapter implements
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Lifecycle state machine for a world node.
 *
 *   Created → Initialising → Ready → Disposed
 *                              ↕
 *                           Disabled
 */
export const enum NodeState {
  Created       = 0,
  Initialising  = 1,
  Ready         = 2,
  Disabled      = 3,
  Disposed      = 4,
}

/**
 * Describes which graph resources a node reads from and writes to.
 * Resource names are resolved to ResourceHandle IDs during graph wiring.
 */
export interface NodeResourceDecl {
  /** Resource names this node reads (creates graph input edges). */
  inputs: string[];
  /** Resource names this node writes (creates graph output edges). */
  outputs: string[];
  /** Resource descriptors for any outputs this node creates. */
  outputDescs?: Record<string, ResourceDescriptor>;
}

/**
 * A world node is the adapter between an AT rendering module and the
 * RenderGraph + WorldIntegrator lifecycle.
 *
 * Implementations are created by the factory functions in the
 * ModuleFactory registry below.
 */
export interface ATWorldNode {
  /** Unique identifier within the world. */
  readonly nodeId: string;

  /**
   * Render layer bucket.  Determines coarse execution ordering.
   * Nodes in lower-numbered layers execute first.
   */
  readonly layer: RenderLayer;

  /**
   * Z-depth sort key within the layer.
   * Lower values are farther from the camera (drawn first in a
   * painter's-algorithm back-to-front pass).
   * For compute-only nodes this is advisory and used for ordering
   * compute dispatches within the same layer.
   */
  zDepth: number;

  /**
   * Declared resource dependencies.
   * Evaluated once during graph wiring; structural changes require
   * a graph recompile.
   */
  readonly resources: NodeResourceDecl;

  /** Current lifecycle state. */
  state: NodeState;

  /**
   * One-time GPU resource allocation.
   * Called by the integrator during init() in dependency order.
   *
   * @param device — WebGPU device.
   * @param format — Swap-chain texture format.
   * @param width  — Viewport width.
   * @param height — Viewport height.
   */
  init(
    device: GPUDevice,
    format: GPUTextureFormat,
    width:  number,
    height: number,
  ): Promise<void>;

  /**
   * Per-frame update: advance simulation, upload uniforms, dispatch
   * compute passes.  Called before render() in the same frame.
   *
   * @param encoder — GPUCommandEncoder for this frame.
   * @param dt      — Delta time in seconds.
   * @param elapsed — Total elapsed time in seconds.
   */
  tick(
    encoder: GPUCommandEncoder,
    dt:      number,
    elapsed: number,
  ): void;

  /**
   * Per-frame render pass recording.  Called after tick() in the same
   * frame, within the graph's sorted execution order.
   *
   * @param encoder  — GPUCommandEncoder.
   * @param accessor — Provides physical GPU textures for virtual resources.
   * @param ctx      — Per-pass metadata (dimensions, timing).
   */
  render(
    encoder:  GPUCommandEncoder,
    accessor: ResourceAccessor,
    ctx:      PassContext,
  ): void;

  /**
   * Handle viewport resize.  Called before the next frame's tick/render
   * when the integrator detects a size change.
   *
   * @param width  — New viewport width.
   * @param height — New viewport height.
   */
  resize(width: number, height: number): void;

  /**
   * Release all GPU resources.  The node must not be used after this.
   */
  dispose(): void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Module name registry — canonical string IDs for built-in AT modules
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Canonical module identifiers accepted by the integrator's module
 * configuration.  Each ID maps to exactly one AT rendering module class.
 */
export type ATModuleName =
  | 'navier-stokes'
  | 'flower-particle'
  | 'spline-particle'
  | 'water-surface'
  | 'volumetric-light'
  | 'bloom'
  | 'pbr-material'
  | 'gem-material'
  | 'particle-compositor'
  | 'scene-clear';

// ─────────────────────────────────────────────────────────────────────────────
// Built-in resource names — well-known texture slots shared between modules
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Standard resource names used by the built-in module adapters.
 * External code can reference these when adding custom nodes that need
 * to read from or write to the same intermediate buffers.
 */
export const ATResource = {
  /** Main scene colour buffer (opaque geometry + particles). */
  SCENE_COLOR:     'at-scene-color',
  /** Main scene depth buffer. */
  SCENE_DEPTH:     'at-scene-depth',
  /** Navier-Stokes dye output (RGBA). */
  FLUID_DYE:       'at-fluid-dye',
  /** Flower particle tPos texture. */
  FLOWER_TPOS:     'at-flower-tpos',
  /** Spline particle tPos texture. */
  SPLINE_TPOS:     'at-spline-tpos',
  /** Water surface colour output. */
  WATER_COLOR:     'at-water-color',
  /** Water surface depth output. */
  WATER_DEPTH:     'at-water-depth',
  /** Volumetric light composited output. */
  VL_COLOR:        'at-vl-color',
  /** Bloom composited output (final before swap-chain). */
  BLOOM_COLOR:     'at-bloom-color',
  /** Particle compositor alpha layer output. */
  PARTICLE_ALPHA:  'at-particle-alpha',
  /** Particle compositor glow layer output. */
  PARTICLE_GLOW:   'at-particle-glow',
  /** Gem material colour output. */
  GEM_COLOR:       'at-gem-color',
} as const;

export type ATResourceName = (typeof ATResource)[keyof typeof ATResource];

// ─────────────────────────────────────────────────────────────────────────────
// Cell Entry — per-cell rendering resources (same as ATSceneCompositor)
// ─────────────────────────────────────────────────────────────────────────────

/** Axis-aligned bounding box for a cell in SPH domain units. */
export interface CellBBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Minimal read-only view of SPH world state consumed by the integrator. */
export interface SPHWorldView {
  readonly x:       Float32Array;
  readonly y:       Float32Array;
  readonly vx:      Float32Array;
  readonly vy:      Float32Array;
  readonly species: Uint32Array;
  readonly count:   number;
}

/**
 * Internal bookkeeping for a registered cell's rendering resources.
 */
interface CellEntry {
  cellId:      string;
  species:     string;
  bbox:        CellBBox;
  zDepth:      number;
  shaderCfg:   SpeciesShaderConfig;
  materialDef: SpeciesMaterialDef | null;
  pbrMat:      ATPBRMaterial   | null;
  matcapMat:   ATMatcapFresnel | null;
  gemMat:      ATGemMaterial   | null;
  /** Whether this cell is in "highlight" mode (uses gem material). */
  highlighted: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// World Integrator Configuration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Configuration object for ATWorldIntegrator.init().
 * All fields are optional — sensible defaults are provided.
 */
export interface ATWorldIntegratorConfig {
  /** Texture format for swap-chain / render targets.  Default 'bgra8unorm'. */
  format?: GPUTextureFormat;

  /**
   * Which built-in modules to initialise.
   * Default: all modules.
   */
  modules?: ATModuleName[];

  /** Flower particle edge splines for the initial scene. */
  flowerEdges?: FlowerEdgeSpline[];

  /** Spline particle edge splines for the initial scene. */
  splineEdges?: EdgeSpline[];

  /** Water surface configuration overrides. */
  waterConfig?: ATWaterSurfaceConfig;

  /** Bloom post-process parameter overrides. */
  bloomParams?: ATBloomParams;

  /** Volumetric light parameter overrides. */
  vlParams?: ATVolumetricLightParams;

  /** Base z-depth offsets per render layer (for fine-tuning ordering). */
  layerDepthOffsets?: Partial<Record<RenderLayer, number>>;

  /** Maximum number of cells the integrator can manage. */
  maxCells?: number;

  /**
   * If true, the integrator skips the graph-based execution and falls back
   * to a linear pipeline order (useful for debugging).
   */
  linearFallback?: boolean;

  /**
   * Custom nodes to register alongside the built-in modules.
   * They participate in the same z-depth sorting and graph compilation.
   */
  customNodes?: ATWorldNode[];
}

/** Default module list — all built-in modules enabled. */
const DEFAULT_MODULES: ATModuleName[] = [
  'scene-clear',
  'navier-stokes',
  'flower-particle',
  'spline-particle',
  'particle-compositor',
  'pbr-material',
  'water-surface',
  'volumetric-light',
  'bloom',
];

// ─────────────────────────────────────────────────────────────────────────────
// Z-depth constants — default layer base depths
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Default z-depth base values for built-in modules.
 * These provide sensible painter's-algorithm ordering out of the box.
 * Lower values = farther from camera = drawn first.
 */
const MODULE_Z_DEPTHS: Record<ATModuleName, number> = {
  'scene-clear':         -1000,
  'navier-stokes':       -500,
  'flower-particle':      100,
  'spline-particle':      110,
  'particle-compositor':  150,
  'pbr-material':         200,
  'gem-material':         210,
  'water-surface':        400,
  'volumetric-light':     500,
  'bloom':                600,
};

// ─────────────────────────────────────────────────────────────────────────────
// FBO helper — off-screen render target pair (matches ATSceneCompositor)
// ─────────────────────────────────────────────────────────────────────────────

interface FBO {
  color:     GPUTexture;
  colorView: GPUTextureView;
  depth:     GPUTexture;
  depthView: GPUTextureView;
}

function createFBO(
  device: GPUDevice,
  w:      number,
  h:      number,
  format: GPUTextureFormat,
  label:  string,
): FBO {
  const color = device.createTexture({
    label:  `${label}-color`,
    size:   [w, h],
    format,
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING   |
      GPUTextureUsage.COPY_SRC          |
      GPUTextureUsage.COPY_DST,
  });
  const depth = device.createTexture({
    label:  `${label}-depth`,
    size:   [w, h],
    format: 'depth24plus',
    usage:  GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return {
    color,
    colorView: color.createView(),
    depth,
    depthView: depth.createView(),
  };
}

function destroyFBO(fbo: FBO): void {
  fbo.color.destroy();
  fbo.depth.destroy();
}

// ─────────────────────────────────────────────────────────────────────────────
// Sorted Node List — z-depth comparator utilities
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compare two nodes for rendering order.
 * Primary sort: layer (ascending).
 * Secondary sort: zDepth (ascending — lower depth = farther = drawn first).
 * Tertiary sort: nodeId (lexicographic, for determinism).
 */
function compareNodes(a: ATWorldNode, b: ATWorldNode): number {
  if (a.layer !== b.layer) return a.layer - b.layer;
  if (a.zDepth !== b.zDepth) return a.zDepth - b.zDepth;
  return a.nodeId < b.nodeId ? -1 : a.nodeId > b.nodeId ? 1 : 0;
}

/**
 * Insert a node into a sorted array, maintaining sort order.
 * Uses binary search for O(log n) insertion.
 */
function sortedInsert(arr: ATWorldNode[], node: ATWorldNode): void {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (compareNodes(arr[mid], node) <= 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  arr.splice(lo, 0, node);
}

/**
 * Remove a node from a sorted array by nodeId.
 * Returns true if found and removed.
 */
function sortedRemove(arr: ATWorldNode[], nodeId: string): boolean {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i].nodeId === nodeId) {
      arr.splice(i, 1);
      return true;
    }
  }
  return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in Node Adapters
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Scene clear node — clears the main scene FBO to transparent black.
 * Always runs first (Background layer, lowest z-depth).
 */
class SceneClearNode implements ATWorldNode {
  readonly nodeId = 'scene-clear';
  readonly layer  = RenderLayer.Background;
  zDepth          = -1000;
  state           = NodeState.Created;

  readonly resources: NodeResourceDecl = {
    inputs:  [],
    outputs: [ATResource.SCENE_COLOR, ATResource.SCENE_DEPTH],
  };

  private fbo: FBO | null = null;
  private device: GPUDevice | null = null;
  private format: GPUTextureFormat = 'bgra8unorm';
  private width  = 0;
  private height = 0;

  get sceneFBO(): FBO | null { return this.fbo; }

  async init(
    device: GPUDevice, format: GPUTextureFormat,
    width: number, height: number,
  ): Promise<void> {
    this.device = device;
    this.format = format;
    this.width  = width;
    this.height = height;
    this.fbo    = createFBO(device, width, height, format, 'at-world-scene');
    this.state  = NodeState.Ready;
  }

  tick(): void { /* no-op */ }

  render(
    encoder: GPUCommandEncoder,
    _accessor: ResourceAccessor,
    _ctx: PassContext,
  ): void {
    if (!this.fbo) return;
    const pass = encoder.beginRenderPass({
      label: 'at-world-clear',
      colorAttachments: [{
        view:       this.fbo.colorView,
        loadOp:     'clear',
        storeOp:    'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
      depthStencilAttachment: {
        view:            this.fbo.depthView,
        depthLoadOp:     'clear',
        depthStoreOp:    'store',
        depthClearValue: 1.0,
      },
    });
    pass.end();
  }

  resize(width: number, height: number): void {
    if (!this.device || !this.fbo) return;
    if (this.width === width && this.height === height) return;
    this.width  = width;
    this.height = height;
    destroyFBO(this.fbo);
    this.fbo = createFBO(this.device, width, height, this.format, 'at-world-scene');
  }

  dispose(): void {
    if (this.fbo) {
      destroyFBO(this.fbo);
      this.fbo = null;
    }
    this.state = NodeState.Disposed;
  }
}

/**
 * Navier-Stokes fluid simulation node.
 * Runs as a compute-only pass in the Fluid layer.
 */
class NavierStokesNode implements ATWorldNode {
  readonly nodeId = 'navier-stokes';
  readonly layer  = RenderLayer.Fluid;
  zDepth          = -500;
  state           = NodeState.Created;

  readonly resources: NodeResourceDecl = {
    inputs:  [],
    outputs: [ATResource.FLUID_DYE],
  };

  private fluid: NavierStokesFluid | null = null;
  private pendingSplats: NavierStokesSplat[] = [];

  get nsFluid(): NavierStokesFluid | null { return this.fluid; }

  async init(device: GPUDevice): Promise<void> {
    this.fluid = new NavierStokesFluid(device);
    this.state = NodeState.Ready;
  }

  queueSplat(splat: NavierStokesSplat): void {
    this.pendingSplats.push(splat);
  }

  tick(encoder: GPUCommandEncoder): void {
    if (!this.fluid) return;
    for (const s of this.pendingSplats) {
      this.fluid.splat(encoder, s);
    }
    this.pendingSplats.length = 0;
    this.fluid.step(encoder);
  }

  render(): void { /* compute-only node, work done in tick() */ }

  resize(): void { /* NS grid is fixed resolution */ }

  dispose(): void {
    this.fluid?.destroy();
    this.fluid = null;
    this.pendingSplats.length = 0;
    this.state = NodeState.Disposed;
  }
}

/**
 * Flower particle system node — compute dispatch + instanced render.
 */
class FlowerParticleNode implements ATWorldNode {
  readonly nodeId = 'flower-particle';
  readonly layer  = RenderLayer.Particle;
  zDepth          = 100;
  state           = NodeState.Created;

  readonly resources: NodeResourceDecl = {
    inputs:  [ATResource.SCENE_COLOR, ATResource.SCENE_DEPTH],
    outputs: [ATResource.FLOWER_TPOS],
  };

  private renderer: ATFlowerParticleRenderer | null = null;
  private edges: FlowerEdgeSpline[];
  private sceneFBORef: (() => FBO | null) | null = null;

  constructor(edges: FlowerEdgeSpline[] = []) {
    this.edges = edges;
  }

  setSceneFBOProvider(provider: () => FBO | null): void {
    this.sceneFBORef = provider;
  }

  async init(
    device: GPUDevice, _format: GPUTextureFormat,
    _width: number, _height: number,
  ): Promise<void> {
    // The flower renderer needs the canvas for sizing, but we only
    // need a minimal stub — actual dimensions come from the FBO.
    const canvas = document.createElement('canvas');
    canvas.width  = _width;
    canvas.height = _height;
    this.renderer = new ATFlowerParticleRenderer(device, canvas, this.edges, {});
    await this.renderer.build();
    this.state = NodeState.Ready;
  }

  tick(encoder: GPUCommandEncoder, dt: number, elapsed: number): void {
    if (!this.renderer?.isBuilt) return;
    this.renderer.update(encoder, elapsed, dt);
  }

  render(encoder: GPUCommandEncoder): void {
    if (!this.renderer?.isBuilt) return;
    const fbo = this.sceneFBORef?.();
    if (!fbo) return;
    this.renderer.render(encoder, fbo.colorView, fbo.depthView);
  }

  async setEdges(edges: FlowerEdgeSpline[]): Promise<void> {
    this.edges = edges;
    if (this.renderer) {
      await this.renderer.setEdges(edges);
    }
  }

  resize(): void { /* Flower renderer is resolution-independent */ }

  dispose(): void {
    this.renderer?.destroy();
    this.renderer = null;
    this.state = NodeState.Disposed;
  }
}

/**
 * Spline particle system node — compute dispatch + instanced render.
 */
class SplineParticleNode implements ATWorldNode {
  readonly nodeId = 'spline-particle';
  readonly layer  = RenderLayer.Particle;
  zDepth          = 110;
  state           = NodeState.Created;

  readonly resources: NodeResourceDecl = {
    inputs:  [ATResource.SCENE_COLOR, ATResource.SCENE_DEPTH],
    outputs: [ATResource.SPLINE_TPOS],
  };

  private renderer: ATSplineParticleLife | null = null;
  private edges: EdgeSpline[];
  private sceneFBORef: (() => FBO | null) | null = null;

  constructor(edges: EdgeSpline[] = []) {
    this.edges = edges;
  }

  setSceneFBOProvider(provider: () => FBO | null): void {
    this.sceneFBORef = provider;
  }

  async init(
    device: GPUDevice, _format: GPUTextureFormat,
    _width: number, _height: number,
  ): Promise<void> {
    const canvas = document.createElement('canvas');
    canvas.width  = _width;
    canvas.height = _height;
    this.renderer = new ATSplineParticleLife(device, canvas, this.edges, {});
    await this.renderer.build();
    this.state = NodeState.Ready;
  }

  tick(encoder: GPUCommandEncoder, dt: number, elapsed: number): void {
    if (!this.renderer?.isBuilt) return;
    this.renderer.update(encoder, elapsed, dt);
  }

  render(encoder: GPUCommandEncoder): void {
    if (!this.renderer?.isBuilt) return;
    const fbo = this.sceneFBORef?.();
    if (!fbo) return;
    this.renderer.render(encoder, fbo.colorView, fbo.depthView);
  }

  async setEdges(edges: EdgeSpline[]): Promise<void> {
    this.edges = edges;
    if (this.renderer) {
      await this.renderer.setEdges(edges);
    }
  }

  get splineRenderer(): ATSplineParticleLife | null { return this.renderer; }

  resize(): void { /* Spline renderer is resolution-independent */ }

  dispose(): void {
    this.renderer?.destroy();
    this.renderer = null;
    this.state = NodeState.Disposed;
  }
}

/**
 * Particle compositor node — depth-sorts and composites particle layers.
 */
class ParticleCompositorNode implements ATWorldNode {
  readonly nodeId = 'particle-compositor';
  readonly layer  = RenderLayer.Particle;
  zDepth          = 150;
  state           = NodeState.Created;

  readonly resources: NodeResourceDecl = {
    inputs:  [ATResource.FLOWER_TPOS, ATResource.SPLINE_TPOS, ATResource.SCENE_COLOR],
    outputs: [ATResource.PARTICLE_ALPHA, ATResource.PARTICLE_GLOW],
  };

  private compositor: ParticleCompositor | null = null;
  private sceneFBORef: (() => FBO | null) | null = null;

  setSceneFBOProvider(provider: () => FBO | null): void {
    this.sceneFBORef = provider;
  }

  async init(
    device: GPUDevice, _format: GPUTextureFormat,
    _width: number, _height: number,
  ): Promise<void> {
    const canvas = document.createElement('canvas');
    canvas.width  = _width;
    canvas.height = _height;
    this.compositor = new ParticleCompositor(device, canvas);
    await this.compositor.build();
    this.state = NodeState.Ready;
  }

  tick(): void { /* compositor has no per-frame update */ }

  render(encoder: GPUCommandEncoder): void {
    if (!this.compositor) return;
    const fbo = this.sceneFBORef?.();
    if (!fbo) return;
    this.compositor.sort(encoder);
    this.compositor.renderAlpha(encoder, fbo.colorView);
    this.compositor.renderGlow(encoder, fbo.colorView);
  }

  resize(): void { /* Compositor is resolution-independent */ }

  dispose(): void {
    this.compositor?.destroy();
    this.compositor = null;
    this.state = NodeState.Disposed;
  }
}

/**
 * PBR material node — manages per-cell material rendering.
 * The actual cell material instances are managed by ATWorldIntegrator.
 * This node just provides the graph slot and render callback.
 */
class PBRMaterialNode implements ATWorldNode {
  readonly nodeId = 'pbr-material';
  readonly layer  = RenderLayer.Geometry;
  zDepth          = 200;
  state           = NodeState.Created;

  readonly resources: NodeResourceDecl = {
    inputs:  [ATResource.SCENE_COLOR, ATResource.SCENE_DEPTH],
    outputs: [],
  };

  /** Cell render callback — set by the integrator. */
  onRenderCells:
    ((encoder: GPUCommandEncoder, colorView: GPUTextureView, depthView: GPUTextureView, dt: number) => void) | null
    = null;

  private lastDt = 0;

  async init(): Promise<void> {
    this.state = NodeState.Ready;
  }

  tick(_encoder: GPUCommandEncoder, dt: number): void {
    this.lastDt = dt;
  }

  render(encoder: GPUCommandEncoder): void {
    // Cell rendering is delegated to the integrator's cell registry.
    // The integrator sets onRenderCells before each frame.
    this.onRenderCells?.(
      encoder,
      null as unknown as GPUTextureView,   // replaced by integrator
      null as unknown as GPUTextureView,   // replaced by integrator
      this.lastDt,
    );
  }

  resize(): void { /* Materials are resolution-independent */ }

  dispose(): void {
    this.onRenderCells = null;
    this.state = NodeState.Disposed;
  }
}

/**
 * Gem material node — manages per-cell gem (highlight) material rendering.
 */
class GemMaterialNode implements ATWorldNode {
  readonly nodeId = 'gem-material';
  readonly layer  = RenderLayer.Geometry;
  zDepth          = 210;
  state           = NodeState.Created;

  readonly resources: NodeResourceDecl = {
    inputs:  [ATResource.SCENE_COLOR, ATResource.SCENE_DEPTH],
    outputs: [ATResource.GEM_COLOR],
  };

  onRenderGemCells:
    ((encoder: GPUCommandEncoder, colorView: GPUTextureView, depthView: GPUTextureView, dt: number) => void) | null
    = null;

  private lastDt = 0;

  async init(): Promise<void> {
    this.state = NodeState.Ready;
  }

  tick(_encoder: GPUCommandEncoder, dt: number): void {
    this.lastDt = dt;
  }

  render(encoder: GPUCommandEncoder): void {
    this.onRenderGemCells?.(
      encoder,
      null as unknown as GPUTextureView,
      null as unknown as GPUTextureView,
      this.lastDt,
    );
  }

  resize(): void { }

  dispose(): void {
    this.onRenderGemCells = null;
    this.state = NodeState.Disposed;
  }
}

/**
 * Water surface node — wave simulation + mesh render + particle overlay.
 */
class WaterSurfaceNode implements ATWorldNode {
  readonly nodeId = 'water-surface';
  readonly layer  = RenderLayer.Surface;
  zDepth          = 400;
  state           = NodeState.Created;

  readonly resources: NodeResourceDecl = {
    inputs:  [ATResource.SCENE_COLOR],
    outputs: [ATResource.WATER_COLOR, ATResource.WATER_DEPTH],
  };

  private water: ATWaterSurface | null = null;
  private cfg: ATWaterSurfaceConfig;
  private fbo: FBO | null = null;
  private sceneUniformBuf: GPUBuffer | null = null;
  private device: GPUDevice | null = null;
  private format: GPUTextureFormat = 'bgra8unorm';
  private width  = 0;
  private height = 0;
  private sceneFBORef: (() => FBO | null) | null = null;

  constructor(cfg: ATWaterSurfaceConfig = {}) {
    this.cfg = cfg;
  }

  setSceneFBOProvider(provider: () => FBO | null): void {
    this.sceneFBORef = provider;
  }

  get waterFBO(): FBO | null { return this.fbo; }

  async init(
    device: GPUDevice, format: GPUTextureFormat,
    width: number, height: number,
  ): Promise<void> {
    this.device = device;
    this.format = format;
    this.width  = width;
    this.height = height;

    this.water = new ATWaterSurface(device, format, this.cfg);
    await this.water.build();

    this.fbo = createFBO(device, width, height, format, 'at-world-water');

    this.sceneUniformBuf = device.createBuffer({
      label: 'at-world-scene-uniforms',
      size:  144,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.state = NodeState.Ready;
  }

  addDrop(x: number, y: number, radius: number, strength: number): void {
    this.water?.addDrop(x, y, radius, strength);
  }

  tick(encoder: GPUCommandEncoder, _dt: number, elapsed: number): void {
    if (!this.water || !this.fbo) return;

    // Copy scene FBO to water FBO as base
    const sceneFBO = this.sceneFBORef?.();
    if (sceneFBO) {
      encoder.copyTextureToTexture(
        { texture: sceneFBO.color },
        { texture: this.fbo.color },
        [this.width, this.height],
      );
    }

    this.water.tick(encoder, elapsed);
  }

  render(encoder: GPUCommandEncoder): void {
    if (!this.water || !this.fbo || !this.sceneUniformBuf) return;
    this.water.renderPass(
      encoder,
      this.fbo.colorView,
      this.fbo.depthView,
      this.sceneUniformBuf,
    );
  }

  resize(width: number, height: number): void {
    if (!this.device || !this.fbo) return;
    if (this.width === width && this.height === height) return;
    this.width  = width;
    this.height = height;
    destroyFBO(this.fbo);
    this.fbo = createFBO(this.device, width, height, this.format, 'at-world-water');
  }

  dispose(): void {
    this.water?.destroy();
    this.water = null;
    if (this.fbo) {
      destroyFBO(this.fbo);
      this.fbo = null;
    }
    this.sceneUniformBuf?.destroy();
    this.sceneUniformBuf = null;
    this.state = NodeState.Disposed;
  }
}

/**
 * Volumetric light node — screen-space god rays.
 */
class VolumetricLightNode implements ATWorldNode {
  readonly nodeId = 'volumetric-light';
  readonly layer  = RenderLayer.Volumetric;
  zDepth          = 500;
  state           = NodeState.Created;

  readonly resources: NodeResourceDecl = {
    inputs:  [ATResource.WATER_COLOR],
    outputs: [ATResource.VL_COLOR],
  };

  private vlight: ATVolumetricLight | null = null;
  private vlParams: ATVolumetricLightParams;
  private fbo: FBO | null = null;
  private device: GPUDevice | null = null;
  private format: GPUTextureFormat = 'bgra8unorm';
  private width  = 0;
  private height = 0;
  private inputFBORef: (() => FBO | null) | null = null;

  constructor(params: ATVolumetricLightParams = {}) {
    this.vlParams = { ...params };
  }

  setInputFBOProvider(provider: () => FBO | null): void {
    this.inputFBORef = provider;
  }

  get vlFBO(): FBO | null { return this.fbo; }

  async init(
    device: GPUDevice, format: GPUTextureFormat,
    width: number, height: number,
  ): Promise<void> {
    this.device = device;
    this.format = format;
    this.width  = width;
    this.height = height;

    this.vlight = await ATVolumetricLight.create(device, format, width, height);
    if (Object.keys(this.vlParams).length > 0) {
      this.vlight.setParams(this.vlParams);
    }

    this.fbo = createFBO(device, width, height, format, 'at-world-vl');
    this.state = NodeState.Ready;
  }

  setParams(p: ATVolumetricLightParams): void {
    Object.assign(this.vlParams, p);
    this.vlight?.setParams(p);
  }

  tick(_encoder: GPUCommandEncoder, dt: number): void {
    this.vlight?.tick(dt);
  }

  render(encoder: GPUCommandEncoder): void {
    if (!this.vlight || !this.fbo) return;
    const inputFBO = this.inputFBORef?.();
    if (!inputFBO) return;
    this.vlight.render(encoder, inputFBO.color, this.fbo.colorView);
  }

  resize(width: number, height: number): void {
    if (!this.device || !this.fbo) return;
    if (this.width === width && this.height === height) return;
    this.width  = width;
    this.height = height;
    destroyFBO(this.fbo);
    this.fbo = createFBO(this.device, width, height, this.format, 'at-world-vl');
    this._recreateVL(width, height);
  }

  private async _recreateVL(w: number, h: number): Promise<void> {
    if (!this.device) return;
    this.vlight?.destroy();
    this.vlight = await ATVolumetricLight.create(this.device, this.format, w, h);
    if (Object.keys(this.vlParams).length > 0) {
      this.vlight.setParams(this.vlParams);
    }
  }

  dispose(): void {
    this.vlight?.destroy();
    this.vlight = null;
    if (this.fbo) {
      destroyFBO(this.fbo);
      this.fbo = null;
    }
    this.state = NodeState.Disposed;
  }
}

/**
 * Bloom post-process node — bright extract → gaussian blur → composite.
 */
class BloomNode implements ATWorldNode {
  readonly nodeId = 'bloom';
  readonly layer  = RenderLayer.PostProcess;
  zDepth          = 600;
  state           = NodeState.Created;

  readonly resources: NodeResourceDecl = {
    inputs:  [ATResource.VL_COLOR],
    outputs: [ATResource.BLOOM_COLOR],
  };

  private bloom: ATBloomPostProcess | null = null;
  private bloomParams: ATBloomParams;
  private device: GPUDevice | null = null;
  private format: GPUTextureFormat = 'bgra8unorm';
  private width  = 0;
  private height = 0;
  private inputFBORef: (() => FBO | null) | null = null;

  constructor(params: ATBloomParams = {}) {
    this.bloomParams = { ...params };
  }

  setInputFBOProvider(provider: () => FBO | null): void {
    this.inputFBORef = provider;
  }

  async init(
    device: GPUDevice, format: GPUTextureFormat,
    width: number, height: number,
  ): Promise<void> {
    this.device = device;
    this.format = format;
    this.width  = width;
    this.height = height;

    this.bloom = await ATBloomPostProcess.create(device, format, width, height);
    if (Object.keys(this.bloomParams).length > 0) {
      this.bloom.setParams(this.bloomParams);
    }

    this.state = NodeState.Ready;
  }

  setParams(p: ATBloomParams): void {
    Object.assign(this.bloomParams, p);
    this.bloom?.setParams(p);
  }

  tick(): void { /* bloom has no per-frame update */ }

  render(
    encoder: GPUCommandEncoder,
    _accessor: ResourceAccessor,
    _ctx: PassContext,
    dstView?: GPUTextureView,
  ): void {
    if (!this.bloom) return;
    const inputFBO = this.inputFBORef?.();
    if (!inputFBO || !dstView) return;
    this.bloom.render(encoder, inputFBO.color, dstView);
  }

  resize(width: number, height: number): void {
    if (this.width === width && this.height === height) return;
    this.width  = width;
    this.height = height;
    this._recreateBloom(width, height);
  }

  private async _recreateBloom(w: number, h: number): Promise<void> {
    if (!this.device) return;
    this.bloom?.destroy();
    this.bloom = await ATBloomPostProcess.create(this.device, this.format, w, h);
    if (Object.keys(this.bloomParams).length > 0) {
      this.bloom.setParams(this.bloomParams);
    }
  }

  dispose(): void {
    this.bloom?.destroy();
    this.bloom = null;
    this.state = NodeState.Disposed;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module Factory — creates built-in node adapters by name
// ─────────────────────────────────────────────────────────────────────────────

type NodeFactory = (cfg: ATWorldIntegratorConfig) => ATWorldNode;

const MODULE_FACTORIES: Record<ATModuleName, NodeFactory> = {
  'scene-clear':         ()    => new SceneClearNode(),
  'navier-stokes':       ()    => new NavierStokesNode(),
  'flower-particle':     (cfg) => new FlowerParticleNode(cfg.flowerEdges ?? []),
  'spline-particle':     (cfg) => new SplineParticleNode(cfg.splineEdges ?? []),
  'particle-compositor': ()    => new ParticleCompositorNode(),
  'pbr-material':        ()    => new PBRMaterialNode(),
  'gem-material':        ()    => new GemMaterialNode(),
  'water-surface':       (cfg) => new WaterSurfaceNode(cfg.waterConfig ?? {}),
  'volumetric-light':    (cfg) => new VolumetricLightNode(cfg.vlParams ?? {}),
  'bloom':               (cfg) => new BloomNode(cfg.bloomParams ?? {}),
};

// ─────────────────────────────────────────────────────────────────────────────
// Frame Statistics
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Per-frame diagnostic counters exposed by the integrator.
 */
export interface WorldFrameStats {
  /** Number of nodes that executed their tick() this frame. */
  tickedNodes: number;
  /** Number of nodes that executed their render() this frame. */
  renderedNodes: number;
  /** Number of nodes currently disabled. */
  disabledNodes: number;
  /** Total registered cells. */
  cellCount: number;
  /** Total elapsed time in seconds. */
  elapsed: number;
  /** Last frame's delta time in seconds. */
  dt: number;
  /** Number of render graph passes in the compiled plan. */
  graphPassCount: number;
  /** Number of render graph alias groups (physical textures). */
  graphAliasGroups: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// ATWorldIntegrator — the main exported class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * AT World Integrator — M833
 *
 * Coordinates all AT rendering modules through the RenderGraph (M822),
 * providing unified lifecycle management, z-depth sorted rendering, and
 * dynamic module enable/disable.
 *
 * Lifecycle:
 *   1. `new ATWorldIntegrator()`
 *   2. `await integrator.init(device, canvas, config)`
 *   3. Per frame: `integrator.tick(dt, sphWorld)`
 *   4. Dynamic: `integrator.setModuleEnabled(name, bool)`
 *   5. Resize:  `integrator.resize(w, h)`
 *   6. Cleanup: `integrator.dispose()`
 */
export class ATWorldIntegrator {
  // ── Core WebGPU ────────────────────────────────────────────────────────────
  private device!:  GPUDevice;
  private canvas!:  HTMLCanvasElement;
  private ctx!:     GPUCanvasContext;
  private format:   GPUTextureFormat = 'bgra8unorm';

  // ── Dimensions ─────────────────────────────────────────────────────────────
  private width  = 0;
  private height = 0;

  // ── Render Graph (M822) ────────────────────────────────────────────────────
  private graph: RenderGraph | null = null;
  private graphResources: Map<string, ResourceHandle> = new Map();

  // ── Node registry — sorted by z-depth ──────────────────────────────────────
  private nodes: ATWorldNode[] = [];
  private nodeMap: Map<string, ATWorldNode> = new Map();

  // ── Built-in node references (typed accessors for internal wiring) ─────────
  private sceneClearNode:    SceneClearNode         | null = null;
  private nsNode:            NavierStokesNode       | null = null;
  private flowerNode:        FlowerParticleNode     | null = null;
  private splineNode:        SplineParticleNode     | null = null;
  private particleCompNode:  ParticleCompositorNode | null = null;
  private pbrNode:           PBRMaterialNode        | null = null;
  private gemNode:           GemMaterialNode        | null = null;
  private waterNode:         WaterSurfaceNode       | null = null;
  private vlNode:            VolumetricLightNode    | null = null;
  private bloomNode:         BloomNode              | null = null;

  // ── Cell registry ──────────────────────────────────────────────────────────
  private cells: Map<string, CellEntry> = new Map();
  private cellZDepthBase = 200;
  private cellZDepthStep = 0.1;
  private nextCellZOffset = 0;

  // ── Pending NS splats ──────────────────────────────────────────────────────
  private pendingSplats: NavierStokesSplat[] = [];

  // ── Frame clock ────────────────────────────────────────────────────────────
  private elapsed = 0;
  private lastDt  = 0;

  // ── Frame statistics ───────────────────────────────────────────────────────
  private stats: WorldFrameStats = {
    tickedNodes:    0,
    renderedNodes:  0,
    disabledNodes:  0,
    cellCount:      0,
    elapsed:        0,
    dt:             0,
    graphPassCount: 0,
    graphAliasGroups: 0,
  };

  // ── Configuration ──────────────────────────────────────────────────────────
  private config: ATWorldIntegratorConfig = {};
  private linearFallback = false;

  // ── Lifecycle ──────────────────────────────────────────────────────────────
  private initialised = false;
  private disposed    = false;

  // ─────────────────────────────────────────────────────────────────────────
  // init(device, canvas, config)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Initialise the world integrator.
   *
   * Creates all requested AT module nodes, wires them into the RenderGraph,
   * and performs one-time GPU resource allocation for each module.
   *
   * @param device — WebGPU device (already initialised).
   * @param canvas — Target HTMLCanvasElement.
   * @param cfg    — Configuration overrides.
   */
  async init(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    cfg:    ATWorldIntegratorConfig = {},
  ): Promise<void> {
    if (this.initialised) return;

    this.device = device;
    this.canvas = canvas;
    this.config = cfg;
    this.format = cfg.format ?? 'bgra8unorm';
    this.width  = canvas.width;
    this.height = canvas.height;
    this.linearFallback = cfg.linearFallback ?? false;

    // ── Canvas context ────────────────────────────────────────────────────
    this.ctx = canvas.getContext('webgpu') as GPUCanvasContext;
    this.ctx.configure({
      device,
      format: this.format,
      alphaMode: 'premultiplied',
    });

    // ── Create built-in module nodes ──────────────────────────────────────
    const moduleNames = cfg.modules ?? [...DEFAULT_MODULES];

    for (const name of moduleNames) {
      const factory = MODULE_FACTORIES[name];
      if (!factory) {
        console.warn(`[ATWorldIntegrator] Unknown module: "${name}", skipping.`);
        continue;
      }
      const node = factory(cfg);

      // Apply layer depth offsets
      if (cfg.layerDepthOffsets) {
        const offset = cfg.layerDepthOffsets[node.layer as RenderLayer];
        if (offset !== undefined) {
          node.zDepth += offset;
        }
      }

      this._registerNode(node);
    }

    // ── Register custom nodes ─────────────────────────────────────────────
    if (cfg.customNodes) {
      for (const node of cfg.customNodes) {
        this._registerNode(node);
      }
    }

    // ── Wire inter-node FBO references ────────────────────────────────────
    this._wireInternalReferences();

    // ── Initialise all nodes in sorted order ──────────────────────────────
    for (const node of this.nodes) {
      try {
        node.state = NodeState.Initialising;
        await node.init(device, this.format, this.width, this.height);
        node.state = NodeState.Ready;
      } catch (err) {
        console.error(
          `[ATWorldIntegrator] Failed to init node "${node.nodeId}":`, err,
        );
        node.state = NodeState.Disposed;
      }
    }

    // ── Build the RenderGraph ─────────────────────────────────────────────
    if (!this.linearFallback) {
      this._buildRenderGraph();
    }

    // ── Wire PBR material node render callbacks ───────────────────────────
    this._wireMaterialCallbacks();

    this.initialised = true;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // tick(dt, sphWorld) — per-frame update + render
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Advance all modules by one frame and present to the canvas.
   *
   * The tick method:
   *   1. Updates the frame clock.
   *   2. Processes SPH physics coupling (if sphWorld provided).
   *   3. Iterates all enabled nodes in z-depth order, calling tick().
   *   4. Iterates all enabled nodes in z-depth order, calling render().
   *   5. Submits the GPU command buffer.
   *   6. Updates frame statistics.
   *
   * @param dt       — Delta time in seconds (e.g. 1/60).
   * @param sphWorld — Optional SPH world state for physics coupling.
   */
  tick(dt: number, sphWorld: SPHWorldView | null = null): void {
    if (this.disposed || !this.initialised) return;

    this.elapsed += dt;
    this.lastDt   = dt;

    // ── Acquire swap-chain texture ────────────────────────────────────────
    const swapTex = this.ctx.getCurrentTexture();
    const dstView = swapTex.createView();

    const encoder = this.device.createCommandEncoder({
      label: 'at-world-integrator-frame',
    });

    // ── Phase 1: SPH physics coupling ─────────────────────────────────────
    if (sphWorld && sphWorld.count > 0) {
      this._updateCellMaterialsFromPhysics(sphWorld, dt);
      this._injectNSSplats(encoder, sphWorld);
    }

    // ── Phase 2: Process pending splats ───────────────────────────────────
    if (this.nsNode && this.nsNode.state === NodeState.Ready) {
      for (const s of this.pendingSplats) {
        this.nsNode.queueSplat(s);
      }
      this.pendingSplats.length = 0;
    }

    // ── Phase 3: Tick + Render all nodes in z-depth order ─────────────────
    let tickCount   = 0;
    let renderCount = 0;
    let disabledCount = 0;

    for (const node of this.nodes) {
      if (node.state === NodeState.Disposed) continue;
      if (node.state === NodeState.Disabled) {
        disabledCount++;
        continue;
      }
      if (node.state !== NodeState.Ready) continue;

      // Tick phase
      node.tick(encoder, dt, this.elapsed);
      tickCount++;
    }

    for (const node of this.nodes) {
      if (node.state !== NodeState.Ready) {
        if (node.state === NodeState.Disabled) disabledCount++;
        continue;
      }

      // Special handling for bloom — needs dstView
      if (node.nodeId === 'bloom') {
        (node as BloomNode).render(encoder, null as any, null as any, dstView);
      } else {
        node.render(encoder, null as any, null as any);
      }
      renderCount++;
    }

    // ── Phase 4: Fallback blit if bloom is disabled ───────────────────────
    if (!this.bloomNode || this.bloomNode.state !== NodeState.Ready) {
      this._blitFinalToSwapChain(encoder, dstView);
    }

    // ── Phase 5: Submit ───────────────────────────────────────────────────
    this.device.queue.submit([encoder.finish()]);

    // ── Phase 6: Async readbacks ──────────────────────────────────────────
    if (this.splineNode?.splineRenderer?.isBuilt) {
      this.splineNode.splineRenderer.scheduleHandoffReadback().catch(() => {});
    }

    // ── Update stats ──────────────────────────────────────────────────────
    this.stats.tickedNodes    = tickCount;
    this.stats.renderedNodes  = renderCount;
    this.stats.disabledNodes  = disabledCount;
    this.stats.cellCount      = this.cells.size;
    this.stats.elapsed        = this.elapsed;
    this.stats.dt             = dt;
    this.stats.graphPassCount = this.graph?.getSortedPassNames().length ?? 0;
    this.stats.graphAliasGroups = this.graph?.getAliasGroupCount() ?? 0;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Cell Management
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Register a cell and create its per-species rendering material.
   *
   * Each cell receives a unique z-depth offset within the Geometry layer
   * so that overlapping cells composite correctly in painter's order.
   *
   * The species string is looked up in:
   *   1. species-shader-registry → SpeciesShaderConfig
   *   2. cell-material-system    → SpeciesMaterialDef
   *
   * @param cellId  — Unique cell identifier (e.g. 'cell-0').
   * @param species — Species string (e.g. 'attention', 'ffn').
   * @param bbox    — Bounding box in SPH domain units.
   */
  async addCell(
    cellId:  string,
    species: string,
    bbox:    CellBBox,
  ): Promise<void> {
    if (this.disposed || !this.initialised) return;
    if (this.cells.has(cellId)) return;

    // ── Z-depth assignment ────────────────────────────────────────────────
    const zDepth = this.cellZDepthBase + this.nextCellZOffset * this.cellZDepthStep;
    this.nextCellZOffset++;

    // ── Registry lookups ──────────────────────────────────────────────────
    let shaderCfg: SpeciesShaderConfig;
    try {
      shaderCfg = getSpeciesShaderConfig(species);
    } catch {
      shaderCfg = getSpeciesShaderConfig('cil-eye');
    }

    let materialDef: SpeciesMaterialDef | null = null;
    try {
      materialDef = getCellMaterial(species as CellSpecies);
    } catch { /* not a CellSpecies — fine */ }

    // ── Create GPU materials ──────────────────────────────────────────────
    let pbrMat:    ATPBRMaterial   | null = null;
    let matcapMat: ATMatcapFresnel | null = null;
    let gemMat:    ATGemMaterial   | null = null;

    const matType: MaterialType = materialDef?.materialType ?? shaderCfg.materialType;

    if (matType === 'matcap') {
      matcapMat = await ATMatcapFresnel.create(this.device, this.format);
      if (materialDef?.matcapParams) {
        matcapMat.setParams(materialDef.matcapParams);
      }
    } else {
      pbrMat = await ATPBRMaterial.create(this.device, this.format);
      const pbrParams: Partial<PBRParams> =
        materialDef?.pbrParams ??
        shaderCfg.materialParams as Partial<PBRParams>;
      if (pbrParams) {
        pbrMat.setParams(pbrParams);
      }
    }

    // Pre-create gem material for potential highlight mode
    try {
      gemMat = await ATGemMaterial.create(this.device, this.format);
    } catch {
      // Gem material texture loading may fail — non-critical
    }

    const entry: CellEntry = {
      cellId,
      species,
      bbox,
      zDepth,
      shaderCfg,
      materialDef,
      pbrMat,
      matcapMat,
      gemMat,
      highlighted: false,
    };

    this.cells.set(cellId, entry);
  }

  /**
   * Remove a cell and destroy its GPU material resources.
   */
  removeCell(cellId: string): void {
    const entry = this.cells.get(cellId);
    if (!entry) return;

    entry.pbrMat?.destroy();
    entry.matcapMat?.destroy();
    entry.gemMat?.destroy();
    this.cells.delete(cellId);
  }

  /**
   * Toggle a cell's highlight state (switches to gem material rendering).
   */
  setCellHighlight(cellId: string, highlighted: boolean): void {
    const entry = this.cells.get(cellId);
    if (!entry) return;
    entry.highlighted = highlighted;
  }

  /**
   * Update a cell's bounding box (e.g. after physics simulation).
   */
  updateCellBBox(cellId: string, bbox: CellBBox): void {
    const entry = this.cells.get(cellId);
    if (entry) entry.bbox = bbox;
  }

  /**
   * Get the z-depth value assigned to a cell.
   */
  getCellZDepth(cellId: string): number {
    return this.cells.get(cellId)?.zDepth ?? 0;
  }

  /**
   * Iterate all registered cell IDs.
   */
  cellIds(): IterableIterator<string> {
    return this.cells.keys();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Module Enable / Disable
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Enable or disable a module at runtime.
   *
   * Disabled modules skip both tick() and render() but retain their GPU
   * resources.  This is a lightweight operation that does not require a
   * graph recompile.
   *
   * @param nameOrId — Module name (e.g. 'bloom') or node ID.
   * @param enabled  — Whether to enable (true) or disable (false).
   */
  setModuleEnabled(nameOrId: string, enabled: boolean): void {
    const node = this.nodeMap.get(nameOrId);
    if (!node) {
      console.warn(`[ATWorldIntegrator] Unknown module: "${nameOrId}"`);
      return;
    }

    if (enabled) {
      if (node.state === NodeState.Disabled) {
        node.state = NodeState.Ready;
      }
    } else {
      if (node.state === NodeState.Ready) {
        node.state = NodeState.Disabled;
      }
    }

    // Also update the render graph pass if present
    if (this.graph) {
      try {
        this.graph.setPassEnabled(node.nodeId, enabled);
      } catch {
        // Pass may not exist in graph (compute-only nodes)
      }
    }
  }

  /**
   * Check whether a module is currently enabled.
   */
  isModuleEnabled(nameOrId: string): boolean {
    const node = this.nodeMap.get(nameOrId);
    return node?.state === NodeState.Ready;
  }

  /**
   * Get the list of all registered module/node IDs.
   */
  getModuleIds(): string[] {
    return this.nodes.map(n => n.nodeId);
  }

  /**
   * Get the sorted node list (read-only snapshot).
   */
  getSortedNodes(): readonly ATWorldNode[] {
    return [...this.nodes];
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Dynamic Node Management
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Add a custom node to the world at runtime.
   *
   * The node is initialised immediately and inserted into the sorted
   * node list.  A graph recompile is triggered to incorporate the new
   * node's resource declarations.
   *
   * @param node — A fully constructed ATWorldNode implementation.
   */
  async addNode(node: ATWorldNode): Promise<void> {
    if (this.disposed) return;
    if (this.nodeMap.has(node.nodeId)) {
      console.warn(`[ATWorldIntegrator] Node "${node.nodeId}" already registered.`);
      return;
    }

    this._registerNode(node);

    // Initialise the new node
    try {
      node.state = NodeState.Initialising;
      await node.init(this.device, this.format, this.width, this.height);
      node.state = NodeState.Ready;
    } catch (err) {
      console.error(
        `[ATWorldIntegrator] Failed to init node "${node.nodeId}":`, err,
      );
      node.state = NodeState.Disposed;
    }

    // Recompile graph to include new node
    if (!this.linearFallback && this.graph) {
      this._buildRenderGraph();
    }
  }

  /**
   * Remove a node from the world.
   *
   * The node is disposed and removed from the sorted list.
   * A graph recompile is triggered.
   *
   * @param nodeId — The node's unique identifier.
   */
  removeNode(nodeId: string): void {
    const node = this.nodeMap.get(nodeId);
    if (!node) return;

    node.dispose();
    sortedRemove(this.nodes, nodeId);
    this.nodeMap.delete(nodeId);

    // Recompile graph
    if (!this.linearFallback && this.graph) {
      this._buildRenderGraph();
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Configuration API
  // ─────────────────────────────────────────────────────────────────────────

  /** Update bloom post-process parameters. */
  setBloomParams(p: ATBloomParams): void {
    this.bloomNode?.setParams(p);
  }

  /** Update volumetric light parameters. */
  setVolumetricLightParams(p: ATVolumetricLightParams): void {
    this.vlNode?.setParams(p);
  }

  /** Queue a Navier-Stokes splat (e.g. from mouse/touch). */
  queueSplat(splat: NavierStokesSplat): void {
    if (this.nsNode && this.nsNode.state === NodeState.Ready) {
      this.nsNode.queueSplat(splat);
    } else {
      this.pendingSplats.push(splat);
    }
  }

  /** Add a drop to the water surface. */
  addWaterDrop(x: number, y: number, radius: number, strength: number): void {
    this.waterNode?.addDrop(x, y, radius, strength);
  }

  /**
   * Replace flower particle edge splines at runtime.
   */
  async setFlowerEdges(edges: FlowerEdgeSpline[]): Promise<void> {
    if (this.flowerNode) {
      await this.flowerNode.setEdges(edges);
    }
  }

  /**
   * Replace spline particle edge splines at runtime.
   */
  async setSplineEdges(edges: EdgeSpline[]): Promise<void> {
    if (this.splineNode) {
      await this.splineNode.setEdges(edges);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Resize
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Handle canvas resize.  Notifies all nodes and reallocates
   * size-dependent resources.
   *
   * @param w — New width in pixels.
   * @param h — New height in pixels.
   */
  resize(w: number, h: number): void {
    if (this.disposed || !this.initialised) return;
    if (this.width === w && this.height === h) return;

    this.width  = w;
    this.height = h;

    // Reconfigure canvas context
    this.ctx.configure({
      device:    this.device,
      format:    this.format,
      alphaMode: 'premultiplied',
    });

    // Notify all nodes
    for (const node of this.nodes) {
      if (node.state === NodeState.Ready || node.state === NodeState.Disabled) {
        node.resize(w, h);
      }
    }

    // Recompile graph at new dimensions
    if (!this.linearFallback && this.graph) {
      this.graph.compile(w, h);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Dispose
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Release all GPU resources.
   * The integrator must not be used after this.
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;

    // Dispose cells
    for (const entry of this.cells.values()) {
      entry.pbrMat?.destroy();
      entry.matcapMat?.destroy();
      entry.gemMat?.destroy();
    }
    this.cells.clear();

    // Dispose nodes in reverse order (dependents before dependencies)
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const node = this.nodes[i];
      if (node.state !== NodeState.Disposed) {
        try {
          node.dispose();
        } catch (err) {
          console.error(
            `[ATWorldIntegrator] Error disposing node "${node.nodeId}":`, err,
          );
        }
      }
    }
    this.nodes.length = 0;
    this.nodeMap.clear();

    // Destroy graph
    this.graph?.destroy();
    this.graph = null;
    this.graphResources.clear();

    // Clear typed references
    this.sceneClearNode   = null;
    this.nsNode           = null;
    this.flowerNode       = null;
    this.splineNode       = null;
    this.particleCompNode = null;
    this.pbrNode          = null;
    this.gemNode          = null;
    this.waterNode        = null;
    this.vlNode           = null;
    this.bloomNode        = null;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Accessors
  // ─────────────────────────────────────────────────────────────────────────

  get isInitialised(): boolean { return this.initialised; }
  get isDisposed():    boolean { return this.disposed; }
  get cellCount():     number  { return this.cells.size; }
  get nodeCount():     number  { return this.nodes.length; }
  get elapsedTime():   number  { return this.elapsed; }
  get frameStats():    Readonly<WorldFrameStats> { return this.stats; }

  /** Get the current render graph (null if linearFallback or disposed). */
  get renderGraph(): RenderGraph | null { return this.graph; }

  /** Get a node by ID (typed or custom). */
  getNode(nodeId: string): ATWorldNode | undefined {
    return this.nodeMap.get(nodeId);
  }

  /** Get a typed reference to a built-in node. */
  getTypedNode<T extends ATWorldNode>(nodeId: string): T | null {
    return (this.nodeMap.get(nodeId) as T) ?? null;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Node Registration
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Register a node: add to sorted list and node map, store typed references.
   */
  private _registerNode(node: ATWorldNode): void {
    if (this.nodeMap.has(node.nodeId)) {
      console.warn(`[ATWorldIntegrator] Duplicate node ID: "${node.nodeId}"`);
      return;
    }

    sortedInsert(this.nodes, node);
    this.nodeMap.set(node.nodeId, node);

    // Cache typed references for built-in nodes
    if (node instanceof SceneClearNode)        this.sceneClearNode   = node;
    if (node instanceof NavierStokesNode)      this.nsNode           = node;
    if (node instanceof FlowerParticleNode)    this.flowerNode       = node;
    if (node instanceof SplineParticleNode)    this.splineNode       = node;
    if (node instanceof ParticleCompositorNode) this.particleCompNode = node;
    if (node instanceof PBRMaterialNode)       this.pbrNode          = node;
    if (node instanceof GemMaterialNode)       this.gemNode          = node;
    if (node instanceof WaterSurfaceNode)      this.waterNode        = node;
    if (node instanceof VolumetricLightNode)   this.vlNode           = node;
    if (node instanceof BloomNode)             this.bloomNode        = node;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Internal FBO Wiring
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Wire inter-node FBO references so downstream nodes can read upstream
   * outputs without going through the RenderGraph resource system.
   *
   * This direct-wiring approach is used because many AT modules expect
   * concrete GPUTextureView / GPUTexture handles rather than virtual
   * resource handles.  The graph still tracks the dependencies for
   * ordering purposes.
   */
  private _wireInternalReferences(): void {
    const getSceneFBO = () => this.sceneClearNode?.sceneFBO ?? null;
    const getWaterFBO = () => this.waterNode?.waterFBO ?? null;
    const getVLFBO    = () => this.vlNode?.vlFBO ?? null;

    // Flower, spline, particle compositor → scene FBO
    this.flowerNode?.setSceneFBOProvider(getSceneFBO);
    this.splineNode?.setSceneFBOProvider(getSceneFBO);
    this.particleCompNode?.setSceneFBOProvider(getSceneFBO);

    // Water surface → scene FBO (copies scene as base)
    this.waterNode?.setSceneFBOProvider(getSceneFBO);

    // Volumetric light → water FBO (or scene FBO if water disabled)
    this.vlNode?.setInputFBOProvider(() => {
      if (this.waterNode && this.waterNode.state === NodeState.Ready) {
        return getWaterFBO();
      }
      return getSceneFBO();
    });

    // Bloom → VL FBO (or water FBO if VL disabled, or scene FBO)
    this.bloomNode?.setInputFBOProvider(() => {
      if (this.vlNode && this.vlNode.state === NodeState.Ready) {
        return getVLFBO();
      }
      if (this.waterNode && this.waterNode.state === NodeState.Ready) {
        return getWaterFBO();
      }
      return getSceneFBO();
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Material Render Callbacks
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Wire the PBR and Gem material nodes' render callbacks to the cell
   * registry.  These callbacks are invoked during the render phase to
   * draw all registered cells with their per-species materials.
   */
  private _wireMaterialCallbacks(): void {
    if (this.pbrNode) {
      this.pbrNode.onRenderCells = (encoder, _cv, _dv, dt) => {
        this._renderCellMaterials(encoder, dt);
      };
    }

    if (this.gemNode) {
      this.gemNode.onRenderGemCells = (encoder, _cv, _dv, dt) => {
        this._renderGemCellMaterials(encoder, dt);
      };
    }
  }

  /**
   * Render all non-highlighted cell materials (PBR / Matcap).
   * Cells are rendered in z-depth order (ascending = back to front).
   */
  private _renderCellMaterials(encoder: GPUCommandEncoder, dt: number): void {
    const fbo = this.sceneClearNode?.sceneFBO;
    if (!fbo) return;

    // Collect and sort cells by z-depth
    const sortedCells = [...this.cells.values()]
      .filter(c => !c.highlighted)
      .sort((a, b) => a.zDepth - b.zDepth);

    for (const entry of sortedCells) {
      if (entry.pbrMat) {
        entry.pbrMat.tick(dt);
        entry.pbrMat.render(encoder, fbo.colorView, fbo.depthView);
      } else if (entry.matcapMat) {
        entry.matcapMat.tick(dt);
        entry.matcapMat.render(encoder, fbo.colorView);
      }
    }
  }

  /**
   * Render all highlighted cell materials (Gem).
   * Gems are rendered after regular PBR materials for correct layering.
   */
  private _renderGemCellMaterials(encoder: GPUCommandEncoder, dt: number): void {
    const fbo = this.sceneClearNode?.sceneFBO;
    if (!fbo) return;

    const sortedCells = [...this.cells.values()]
      .filter(c => c.highlighted && c.gemMat)
      .sort((a, b) => a.zDepth - b.zDepth);

    for (const entry of sortedCells) {
      if (entry.gemMat) {
        entry.gemMat.tick(dt);
        entry.gemMat.render(encoder, fbo.colorView, fbo.depthView);
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Render Graph Construction
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Build (or rebuild) the RenderGraph from the current node set.
   *
   * Each enabled node is mapped to a graph pass with:
   *   - inputs:  the node's declared input resource handles
   *   - outputs: the node's declared output resource handles
   *   - execute: a wrapper that calls the node's render() method
   *
   * The graph's topological sort + the z-depth sorted pass insertion
   * order ensure correct execution ordering even when dependency edges
   * create partial orderings.
   */
  private _buildRenderGraph(): void {
    // Destroy old graph
    this.graph?.destroy();
    this.graphResources.clear();

    this.graph = new RenderGraph(this.device, this.format);

    // ── Declare all resources referenced by any node ──────────────────────
    const allResourceNames = new Set<string>();
    for (const node of this.nodes) {
      for (const r of node.resources.inputs)  allResourceNames.add(r);
      for (const r of node.resources.outputs) allResourceNames.add(r);
    }

    for (const name of allResourceNames) {
      // Find if any node declares a custom descriptor for this resource
      let desc: ResourceDescriptor = {};
      for (const node of this.nodes) {
        if (node.resources.outputDescs?.[name]) {
          desc = node.resources.outputDescs[name];
          break;
        }
      }

      // Depth resources use depth format
      if (name.includes('depth') || name.includes('DEPTH')) {
        desc = { ...desc, format: 'depth24plus' };
      }

      const handle = this.graph.createResource(name, desc);
      this.graphResources.set(name, handle);
    }

    // ── Add passes in z-depth sorted order ────────────────────────────────
    // Nodes are already sorted; adding them in order ensures the graph's
    // insertion-order tiebreaker respects z-depth within the same
    // dependency tier.
    for (const node of this.nodes) {
      if (node.state === NodeState.Disposed) continue;

      const inputHandles: ResourceHandle[] = [];
      const outputHandles: ResourceHandle[] = [];

      for (const name of node.resources.inputs) {
        const h = this.graphResources.get(name);
        if (h) inputHandles.push(h);
      }
      for (const name of node.resources.outputs) {
        const h = this.graphResources.get(name);
        if (h) outputHandles.push(h);
      }

      // Create the graph pass
      try {
        this.graph.addPass(node.nodeId, {
          inputs:  inputHandles.length  > 0 ? inputHandles  : undefined,
          outputs: outputHandles.length > 0 ? outputHandles : undefined,
          execute: (enc, accessor, ctx) => {
            if (node.state === NodeState.Ready) {
              node.render(enc, accessor, ctx);
            }
          },
        });

        // Set initial enabled state
        this.graph.setPassEnabled(
          node.nodeId,
          node.state === NodeState.Ready,
        );
      } catch (err) {
        // Resource may already have a writer from another node
        // (e.g. two nodes writing the same resource is illegal).
        // Log and skip — the node will still tick() but won't be
        // part of the graph's compiled execution plan.
        console.warn(
          `[ATWorldIntegrator] Failed to add graph pass for "${node.nodeId}":`,
          err,
        );
      }
    }

    // ── Compile the graph ─────────────────────────────────────────────────
    try {
      this.graph.compile(this.width, this.height);
    } catch (err) {
      console.error('[ATWorldIntegrator] Graph compilation failed:', err);
      // Fall back to linear execution
      this.graph.destroy();
      this.graph = null;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Final Blit Fallback
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * When bloom is disabled, blit the last active FBO's colour texture
   * directly to the swap-chain surface.
   *
   * The fallback cascade is: VL FBO → Water FBO → Scene FBO.
   */
  private _blitFinalToSwapChain(
    encoder: GPUCommandEncoder,
    dstView: GPUTextureView,
  ): void {
    let srcTexture: GPUTexture | null = null;

    if (this.vlNode && this.vlNode.state === NodeState.Ready && this.vlNode.vlFBO) {
      srcTexture = this.vlNode.vlFBO.color;
    } else if (this.waterNode && this.waterNode.state === NodeState.Ready && this.waterNode.waterFBO) {
      srcTexture = this.waterNode.waterFBO.color;
    } else if (this.sceneClearNode?.sceneFBO) {
      srcTexture = this.sceneClearNode.sceneFBO.color;
    }

    if (!srcTexture) return;

    const swapTex = this.ctx.getCurrentTexture();
    encoder.copyTextureToTexture(
      { texture: srcTexture },
      { texture: swapTex },
      [this.width, this.height],
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Physics-Driven Material Updates
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Derive per-cell aggregate physics from SPH state and modulate
   * material parameters.
   *
   * For each cell, particles within the cell's bounding box are scanned
   * to compute average velocity magnitude (kinetic energy proxy) and
   * particle density.  These drive:
   *   - PBR: roughness ↑ with KE, iridescence ↑ with KE
   *   - Matcap: Fresnel power ↑ with density
   *   - Gem: caustics intensity ↑ with KE, dispersiveness ↑ with speed
   *
   * The scan is O(cells × particles) per frame; for large particle counts
   * consider spatial hashing (ATSpatialHashGrid) to reduce to O(cells × k).
   */
  private _updateCellMaterialsFromPhysics(
    sph: SPHWorldView,
    _dt: number,
  ): void {
    for (const entry of this.cells.values()) {
      const { bbox, pbrMat, matcapMat, gemMat, highlighted } = entry;
      if (!pbrMat && !matcapMat && !(highlighted && gemMat)) continue;

      // Aggregate particles within the cell bbox
      let sumVelSq = 0;
      let count    = 0;

      for (let i = 0; i < sph.count; i++) {
        const px = sph.x[i];
        const py = sph.y[i];
        if (
          px >= bbox.x && px <= bbox.x + bbox.w &&
          py >= bbox.y && py <= bbox.y + bbox.h
        ) {
          const vx = sph.vx[i];
          const vy = sph.vy[i];
          sumVelSq += vx * vx + vy * vy;
          count++;
        }
      }

      if (count === 0) continue;

      // Normalised kinetic energy proxy [0, 1]
      const avgKE = Math.min(1.0, (sumVelSq / count) * 2.0);
      // Density proxy [0, 1]
      const area    = Math.max(bbox.w * bbox.h, 0.001);
      const density = Math.min(1.0, count / (area * 500));

      if (pbrMat && !highlighted) {
        pbrMat.setParams({
          roughness:    0.15 + avgKE * 0.45,
          iridStrength: 0.3  + avgKE * 0.7,
        } as Partial<PBRParams>);
      }

      if (matcapMat && !highlighted) {
        matcapMat.setParams({
          fresnelPower: 2.0 + density * 4.0,
        } as Partial<MatcapParams>);
      }

      if (gemMat && highlighted) {
        gemMat.setParams({
          causticsIntensity: 0.5 + avgKE * 1.5,
          dispersiveness:    0.15 + avgKE * 0.25,
        });
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Navier-Stokes Splat Injection
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Inject NS dye splats from SPH cell-centre velocities.
   * One splat per cell with non-trivial average velocity.
   */
  private _injectNSSplats(
    encoder: GPUCommandEncoder,
    sph: SPHWorldView,
  ): void {
    if (!this.nsNode || this.nsNode.state !== NodeState.Ready) return;
    const fluid = this.nsNode.nsFluid;
    if (!fluid) return;

    for (const entry of this.cells.values()) {
      const { bbox } = entry;
      let sumVx = 0;
      let sumVy = 0;
      let count = 0;

      for (let i = 0; i < sph.count; i++) {
        const px = sph.x[i];
        const py = sph.y[i];
        if (
          px >= bbox.x && px <= bbox.x + bbox.w &&
          py >= bbox.y && py <= bbox.y + bbox.h
        ) {
          sumVx += sph.vx[i];
          sumVy += sph.vy[i];
          count++;
        }
      }

      if (count < 5) continue;

      const avgVx = sumVx / count;
      const avgVy = sumVy / count;
      const speed = Math.sqrt(avgVx * avgVx + avgVy * avgVy);
      if (speed < 0.01) continue;

      // Cell centre normalised to [0, 1]
      const cx = (bbox.x + bbox.w * 0.5) / (this.width || 1);
      const cy = (bbox.y + bbox.h * 0.5) / (this.height || 1);

      // Dye colour from species shader config
      const mp = entry.shaderCfg.materialParams;
      const dyeR = (mp as any).albedo?.[0] ?? 0.5;
      const dyeG = (mp as any).albedo?.[1] ?? 0.5;
      const dyeB = (mp as any).albedo?.[2] ?? 0.8;

      fluid.splat(encoder, {
        x:  cx,
        y:  cy,
        vx: avgVx * 0.5,
        vy: avgVy * 0.5,
        color: [dyeR, dyeG, dyeB],
      });
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Z-Depth Sorting Utilities (public for external tooling)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Re-sort all nodes by z-depth.  Call this after changing a node's zDepth
   * property to restore the sorted invariant.
   *
   * This is an O(n log n) operation and should be called sparingly (e.g.
   * after a batch of z-depth changes, not per-frame).
   */
  resortNodes(): void {
    this.nodes.sort(compareNodes);
  }

  /**
   * Get the z-depth rendering order as a flat list of node IDs.
   * Useful for debugging and visualization.
   */
  getZDepthOrder(): string[] {
    return this.nodes
      .filter(n => n.state === NodeState.Ready)
      .map(n => n.nodeId);
  }

  /**
   * Get the z-depth rendering order grouped by layer.
   * Returns a map from layer number to sorted node IDs.
   */
  getLayerOrder(): Map<RenderLayer, string[]> {
    const result = new Map<RenderLayer, string[]>();
    for (const node of this.nodes) {
      if (node.state !== NodeState.Ready && node.state !== NodeState.Disabled) continue;
      const layer = node.layer as RenderLayer;
      if (!result.has(layer)) {
        result.set(layer, []);
      }
      result.get(layer)!.push(node.nodeId);
    }
    return result;
  }

  /**
   * Get a debug description of the current world state.
   * Returns a multi-line string with node states, z-depths, and cell info.
   */
  getDebugInfo(): string {
    const lines: string[] = [];
    lines.push(`[ATWorldIntegrator] State: ${this.initialised ? 'INIT' : 'UNINIT'} | Nodes: ${this.nodes.length} | Cells: ${this.cells.size}`);
    lines.push(`  Elapsed: ${this.elapsed.toFixed(3)}s | Last dt: ${this.lastDt.toFixed(4)}s`);
    lines.push(`  Dimensions: ${this.width}×${this.height}`);
    lines.push(`  Graph: ${this.graph ? 'compiled' : 'none'} | Linear fallback: ${this.linearFallback}`);
    lines.push('');
    lines.push('  Nodes (sorted by z-depth):');

    const layerNames: Record<number, string> = {
      [RenderLayer.Background]:  'BG',
      [RenderLayer.Fluid]:       'FL',
      [RenderLayer.Geometry]:    'GE',
      [RenderLayer.Particle]:    'PA',
      [RenderLayer.Surface]:     'SU',
      [RenderLayer.Volumetric]:  'VO',
      [RenderLayer.PostProcess]: 'PP',
      [RenderLayer.Composite]:   'CO',
      [RenderLayer.HUD]:         'HU',
    };

    const stateNames: Record<number, string> = {
      [NodeState.Created]:      'CREATED',
      [NodeState.Initialising]: 'INIT...',
      [NodeState.Ready]:        'READY',
      [NodeState.Disabled]:     'DISABLED',
      [NodeState.Disposed]:     'DISPOSED',
    };

    for (const node of this.nodes) {
      const lname = layerNames[node.layer] ?? `L${node.layer}`;
      const sname = stateNames[node.state] ?? `S${node.state}`;
      const ins  = node.resources.inputs.join(',') || '(none)';
      const outs = node.resources.outputs.join(',') || '(none)';
      lines.push(`    [${lname}] z=${node.zDepth.toFixed(1).padStart(8)} ${sname.padEnd(8)} ${node.nodeId.padEnd(24)} in:[${ins}] out:[${outs}]`);
    }

    if (this.cells.size > 0) {
      lines.push('');
      lines.push('  Cells (sorted by z-depth):');
      const sortedCells = [...this.cells.values()].sort((a, b) => a.zDepth - b.zDepth);
      for (const cell of sortedCells) {
        const matType = cell.highlighted ? 'GEM' : (cell.pbrMat ? 'PBR' : 'MAT');
        lines.push(`    z=${cell.zDepth.toFixed(2).padStart(8)} ${matType.padEnd(4)} ${cell.cellId.padEnd(16)} species=${cell.species} bbox=(${cell.bbox.x},${cell.bbox.y},${cell.bbox.w},${cell.bbox.h})`);
      }
    }

    return lines.join('\n');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports for consumer convenience
// ─────────────────────────────────────────────────────────────────────────────

export {
  RenderGraph,
  type ResourceHandle,
  type ResourceDescriptor,
  type PassExecuteFn,
  type PassContext,
  type ResourceAccessor,
} from './render-graph.ts';

export type {
  ATBloomParams,
} from './at-bloom-postprocess.ts';

export type {
  ATVolumetricLightParams,
} from './at-volumetric-light.ts';

export type {
  FlowerEdgeSpline,
} from './at-flower-particle.ts';

export type {
  EdgeSpline,
} from './at-spline-particle.ts';

export type {
  ATWaterSurfaceConfig,
} from './at-water-surface.ts';

export type {
  NavierStokesSplat,
} from './at-navier-stokes.ts';

export type {
  PBRParams,
  MatcapParams,
} from './at-pbr-material.ts';

export type {
  SpeciesShaderConfig,
  MaterialType,
} from './species-shader-registry.ts';

export type {
  CellSpecies,
  SpeciesMaterialDef,
} from './cell-material-system.ts';

export type {
  LayerDescriptor,
} from './particle-compositor.ts';
