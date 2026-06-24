/**
 * particle-compositor.ts — M767
 *
 * Unified Particle Render Compositor
 * ─────────────────────────────────────────────────────────────────────────────
 * Single WebGPU render compositor that consolidates all AT particle renderers
 * into one coherent draw sequence with:
 *
 *   ① Depth sorting  — GPU radix sort (O(n) two-pass 16-bit LSB) of all
 *                      active particles by configurable depth key (travel,
 *                      worldY, worldZ, or manual).  Automatic fallback to
 *                      bitonic sort for small counts (<256).
 *
 *   ② Alpha blending — per-layer blend modes: premultiplied (default),
 *                      straight, additive, or custom GPUBlendState.  Drawn
 *                      in sorted order for correct transparency compositing.
 *
 *   ③ Additive glow  — variable-sigma Gaussian kernel per layer with energy
 *                      conservation (maxGlowEnergy cap per layer and global).
 *                      Produces AT-authentic bloom halos without a full
 *                      post-process bloom chain.
 *
 *   ④ Instanced draw — indirect draw via GPUBuffer (drawIndirect) avoiding
 *                      CPU round-trips.  One indirect call per visible layer.
 *
 * Layer types (6):
 *   Flower   — ATFlowerParticleRenderer, matcap shading, spiral motion
 *   Spline   — ATSplineParticleLife, soft-disc SDF, sin(π·travel) fade
 *   EdgeFlow — EdgeFlowRenderer, QoS-tinted spline flow particles
 *   CurlAura — CurlAuraRenderer, curl-noise aura halos (WebGL2 bridge)
 *   Sparks   — ContactSparkSystem, collision contact spark particles
 *   Custom   — user-defined tPos-backed layer with arbitrary shader
 *
 * Architecture
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ParticleCompositor
 *     ├─ LayerDescriptor[]        — registered tPos textures + metadata
 *     ├─ SORT pass (compute)      — radix / bitonic / none (auto-select)
 *     │   ├─ KEY_EXTRACT          — read tPos → (key, index) pairs
 *     │   ├─ RADIX_HISTOGRAM      — 256-bin histogram per workgroup
 *     │   ├─ RADIX_PREFIX_SUM     — exclusive prefix sum over bins
 *     │   └─ RADIX_SCATTER        — scatter pairs into sorted order
 *     ├─ ALPHA pass (render)      — sorted instanced quads, per-layer blend
 *     ├─ GLOW pass (render)       — sorted instanced quads, additive blend
 *     ├─ FILL_INDIRECT (compute)  — populate drawIndirect buffers
 *     └─ compositor uniforms buf  — shared sort+render params (80 B)
 *
 * Per-frame sequence (encode into one GPUCommandEncoder):
 *   compositor.update(enc, elapsed)      — refresh uniforms + indirect
 *   compositor.sort(enc)                 — rebuild sorted-index buffer
 *   compositor.renderAlpha(enc, view)    — alpha-blended layer
 *   compositor.renderGlow(enc, view)     — additive glow layer
 *
 * Integration:
 * ─────────────────────────────────────────────────────────────────────────────
 * ```ts
 * import { ParticleCompositor, LayerType } from '$lib/sph/particle-compositor';
 *
 * const compositor = new ParticleCompositor(device, canvas);
 *
 * compositor.addLayer({
 *   id:         'flower',
 *   type:       LayerType.Flower,
 *   tPosView:   flowerRenderer.tPosView,
 *   texW:       256,
 *   texH:       128,
 *   particleCount: flowerRenderer.activeParticleCount,
 *   uSize:      0.025,
 *   glowScale:  2.2,
 *   glowAlpha:  0.35,
 *   glowSigma:  4.0,
 *   colorTint:  { r: 1, g: 0.78, b: 0.55 },
 * });
 *
 * await compositor.build();
 *
 * // render loop:
 * const enc = device.createCommandEncoder();
 * compositor.composite(enc, swapchainView, elapsed);
 * device.queue.submit([enc.finish()]);
 * ```
 *
 * Sort strategy
 * ─────────────────────────────────────────────────────────────────────────────
 * Radix sort (two-pass 16-bit LSB then MSB) is O(n) and significantly faster
 * than bitonic O(n log²n) for large particle counts.  For counts below 256,
 * bitonic is cheaper due to radix overhead — the compositor auto-selects
 * unless overridden.
 *
 * Sort key modes:
 *   'travel' — tPos.b (travel fraction) as depth key (default, cheap)
 *   'worldY' — tPos.g (world Y) as depth key (good for isometric/2-D)
 *   'worldZ' — reserved for 3-D scenes
 *   'manual' — external sort key buffer, user-populated
 *
 * References
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/at-flower-particle.ts    — FlowerParticleShader WebGPU port
 *   src/lib/sph/at-spline-particle.ts    — SplineParticleLife WebGPU port
 *   src/lib/sph/edge-flow-renderer.ts    — EdgeFlowRenderer M742
 *   src/lib/sph/curl-aura.ts             — CurlAuraRenderer M749
 *   src/lib/sph/contact-sparks.ts        — ContactSparkSystem M587
 *   src/lib/sph/ParticleRenderer.ts      — instanced quad helpers
 *   src/lib/sph/at-bloom-postprocess.ts  — full bloom chain (heavier alt)
 */

// ─── Constants ────────────────────────────────────────────────────────────────

/** Workgroup size for all compute passes. Keep ≤256 for broad compatibility. */








const WG = 64 as const;

/**
 * Sort buffer is capped at this maximum particle count.
 * Both radix and bitonic allocate buffers proportional to this.
 */
const MAX_SORT_PARTICLES = 131072 as const;

/** Byte stride of one (key, index) pair in the sort buffer. */
const SORT_PAIR_STRIDE = 8 as const;  // 2 × u32

/** Compositor uniform buffer byte size (20 f32 slots = 80 bytes). */
const COMPOSITOR_UNIFORMS_BYTES = 80 as const;

/** Number of radix bins per pass (8-bit radix → 256 bins). */
const RADIX_BINS = 256 as const;

/** Threshold below which bitonic sort is preferred over radix. */
const RADIX_THRESHOLD = 256 as const;

/** Bytes per indirect draw call: 4 × u32 (vertexCount, instanceCount, firstVertex, firstInstance). */
const INDIRECT_DRAW_BYTES = 16 as const;

// ─── Public types ─────────────────────────────────────────────────────────────

/** Distinguishes particle system types so the VS reads the correct tPos layout. */
export const enum LayerType {
  /** ATFlowerParticleRenderer — matcap shading, spiral motion. */
  Flower   = 0,
  /** ATSplineParticleLife — soft-disc SDF, sin(π·travel) fade. */
  Spline   = 1,
  /** EdgeFlowRenderer — QoS-tinted spline flow particles. */
  EdgeFlow = 2,
  /** CurlAuraRenderer — curl-noise aura halos (WebGL2 bridge). */
  CurlAura = 3,
  /** ContactSparkSystem — collision contact sparks. */
  Sparks   = 4,
  /** User-defined tPos-backed layer with arbitrary shader. */
  Custom   = 5,
}

/** Simple RGB colour triple (0–1 range per channel). */
export interface ColorRGB {
  r: number;
  g: number;
  b: number;
}

/** Sort key extraction mode. */
export type SortKeyMode = 'travel' | 'worldY' | 'worldZ' | 'manual';

/** Sort algorithm selection. */
export type SortAlgorithm = 'auto' | 'radix' | 'bitonic' | 'none';

/** Per-layer blend mode presets. */
export type BlendPreset = 'premultiplied' | 'straight' | 'additive' | 'custom';

/**
 * A registered tPos-backed particle layer fed to the compositor.
 * The tPos texture must be readable (TEXTURE_BINDING usage) and alive for
 * the full lifetime of the compositor.
 */
export interface LayerDescriptor {
  /** Unique identifier (used in error messages and debug). */
  id: string;
  /** Which AT particle system produced this tPos texture. */
  type: LayerType;
  /**
   * tPos texture view (rgba32float, W×H).
   * .r = worldX  .g = worldY  .b = travel  .a = alpha
   */
  tPosView: any /*GPUTextureView*/;
  /** tPos texture width (= 256 for most AT systems). */
  texW: number;
  /** tPos texture height (= 128 for most AT systems → 32768 slots). */
  texH: number;
  /** Number of active particle slots (≤ texW × texH). */
  particleCount: number;
  /**
   * Particle quad half-size in domain units for the alpha pass.
   * Matches the renderer's uSize (Flower: 0.025, Spline: 0.012).
   */
  uSize: number;
  /**
   * Glow quad scale multiplier relative to uSize (e.g. 2.2 → glow quad is
   * 2.2× larger than the alpha quad). AT-style halos need 1.5–3.0.
   */
  glowScale?: number;
  /**
   * Peak glow alpha at particle centre (0–1). AT bloom headroom ~0.2–0.4.
   * Additive blending means bright clusters self-accumulate naturally.
   */
  glowAlpha?: number;
  /**
   * Gaussian glow kernel width (σ). Higher = softer, wider glow.
   * Default: 4.0 (matches AT bloom radius).
   */
  glowSigma?: number;
  /**
   * Maximum accumulated glow energy for this layer (energy conservation).
   * Prevents overlapping glow from blowing out to white.
   * Default: 1.0 (no cap).
   */
  maxGlowEnergy?: number;
  /**
   * Per-layer colour tint applied in the fragment shader.
   * Default: white {r:1, g:1, b:1} (no tint).
   */
  colorTint?: ColorRGB;
  /**
   * Per-layer blend mode for the alpha pass.
   * Default: 'premultiplied'.
   */
  blendMode?: BlendPreset;
  /**
   * Custom GPU blend state (only used when blendMode === 'custom').
   */
  customBlend?: GPUBlendState;
  /**
   * Draw-order priority within the compositor.
   * Lower = drawn first (further back). Default 0.
   */
  zOrder?: number;
  /**
   * Whether this layer is initially hidden.
   * Hidden layers are not drawn but retain their sort buffer allocation.
   * Default: false.
   */
  hidden?: boolean;
  /**
   * Skip depth sorting for this layer (e.g. fully additive layers
   * where draw order doesn't matter).
   * Default: false.
   */
  skipSort?: boolean;
  /**
   * Sort key mode override for this layer.
   * If not set, uses the compositor-level sortKey.
   */
  sortKeyMode?: SortKeyMode;
}

/** Compositor-level per-frame configuration. */
export interface CompositorConfig {
  /** Background clear colour applied before the alpha pass. Default: transparent. */
  clearColor?: GPUColorDict;
  /**
   * Whether to run the glow pass at all.
   * Disable on low-end hardware or when a separate bloom chain is used.
   * Default: true.
   */
  enableGlow?: boolean;
  /**
   * Global glow intensity multiplier (0–1). Scales all layer glow.
   * Default: 1.0.
   */
  globalGlowIntensity?: number;
  /**
   * Maximum accumulated glow energy across ALL layers (global energy cap).
   * Default: 2.0.
   */
  maxGlobalGlowEnergy?: number;
  /**
   * Sort algorithm selection.
   * 'auto' — radix for counts ≥ 256, bitonic otherwise.
   * 'radix' / 'bitonic' / 'none' — force specific algorithm.
   * Default: 'auto'.
   */
  sortAlgorithm?: SortAlgorithm;
  /**
   * Sort depth key mode (global default, overridable per-layer).
   * Default: 'travel'.
   */
  sortKey?: SortKeyMode;
}

/** Runtime diagnostics snapshot. */
export interface CompositorDiagnostics {
  /** Which sort algorithm was actually used this frame. */
  sortAlgorithm: 'radix' | 'bitonic' | 'none';
  /** Number of visible (non-hidden) layers. */
  activeLayers: number;
  /** Total active particles across visible layers. */
  activeParticles: number;
  /** Total sort buffer slots (pow2 rounded). */
  sortBufferSlots: number;
}

// ─── Internal layout types (not exported) ─────────────────────────────────────

/** One GPU-side layer descriptor stored in the layer metadata buffer. */
interface GPULayerMeta {
  layerOffset:    number;  // particle slot base in the combined sort buffer
  particleCount:  number;
  sortedCount:    number;  // rounded up to pow2 for bitonic sort
  texW:           number;
  texH:           number;
  uSize:          number;
  glowScale:      number;
  glowAlpha:      number;
  glowSigma:      number;
  maxGlowEnergy:  number;
  colorTint:      ColorRGB;
  blendMode:      BlendPreset;
  customBlend?:   GPUBlendState;
  type:           LayerType;
  tPosView:       GPUTextureView;
  id:             string;
  hidden:         boolean;
  skipSort:       boolean;
  sortKeyMode:    SortKeyMode;
}

// ─────────────────────────────────────────────────────────────────────────────
// Blend state presets
// ─────────────────────────────────────────────────────────────────────────────

function blendStateForPreset(preset: BlendPreset, custom?: GPUBlendState): GPUBlendState {
  switch (preset) {
    case 'premultiplied':
      return {
        color: { srcFactor: 'src-alpha',  dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one',        dstFactor: 'one-minus-src-alpha', operation: 'add' },
      };
    case 'straight':
      return {
        color: { srcFactor: 'src-alpha',  dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'src-alpha',  dstFactor: 'one-minus-src-alpha', operation: 'add' },
      };
    case 'additive':
      return {
        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
        alpha: { srcFactor: 'one',       dstFactor: 'one', operation: 'add' },
      };
    case 'custom':
      return custom ?? blendStateForPreset('premultiplied');
    default:
      return blendStateForPreset('premultiplied');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Key extraction compute pass
// Reads each tPos texture and writes (key, originalIndex) into sortBuf.
// Supports 4 sort key modes via uniform switch.
// ─────────────────────────────────────────────────────────────────────────────

const KEY_EXTRACT_WGSL = /* wgsl */`
struct CompUniforms {
  layerOffset   : u32,   // first slot index of this layer in sort buffer
  particleCount : u32,   // number of active slots in this layer
  texW          : u32,   // tPos texture width
  texH          : u32,   // tPos texture height
  sortKeyMode   : u32,   // 0=travel, 1=worldY, 2=worldZ, 3=manual
  _pad0         : u32,
  _pad1         : u32,
  _pad2         : u32,
}

@group(0) @binding(0) var<uniform>             uni     : CompUniforms;
@group(1) @binding(0) var                      tPos    : texture_2d<f32>;
@group(1) @binding(1) var<storage, read_write> sortBuf : array<u32>;

// Convert a float to an unsigned int preserving sort order:
//   positive floats: flip only sign bit → maintains relative order
//   negative floats: flip all bits → reverses into correct order
fn floatToKey(f: f32) -> u32 {
  let bits = bitcast<u32>(f);
  let mask = select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
  return bits ^ mask;
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= uni.particleCount) { return; }

  let texX = i32(slot % uni.texW);
  let texY = i32(slot / uni.texW);
  let p    = textureLoad(tPos, vec2<i32>(texX, texY), 0);

  // Choose sort key based on mode:
  //   0 = travel (.b), 1 = worldY (.g), 2 = worldZ (reserved, use .b),
  //   3 = manual (use .a — external key in alpha channel)
  var rawKey: f32;
  switch uni.sortKeyMode {
    case 0u: { rawKey = p.b; }
    case 1u: { rawKey = p.g; }
    case 2u: { rawKey = p.b; }  // worldZ: reserved, alias to travel
    case 3u: { rawKey = p.a; }  // manual: key from alpha channel
    default: { rawKey = p.b; }
  }

  // Dead/invisible particles (alpha ≤ 0) sort to the back
  let alive  = select(0.0, rawKey, p.a > 0.004);
  let key    = floatToKey(alive);

  let globalSlot = uni.layerOffset + slot;
  sortBuf[globalSlot * 2u + 0u] = key;
  sortBuf[globalSlot * 2u + 1u] = globalSlot;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Bitonic sort compute pass
// Single-pass bitonic merge network step.  Call once per (step, subStep) pair.
// Full sort requires log2(N) * (log2(N)+1) / 2 dispatch calls.
// Used when particle count < RADIX_THRESHOLD or forced via config.
// ─────────────────────────────────────────────────────────────────────────────

const BITONIC_SORT_WGSL = /* wgsl */`
struct SortParams {
  k : u32,   // current merge block size (power of two)
  j : u32,   // sub-step comparator distance
  n : u32,   // total element count (rounded to pow2)
  _pad: u32,
}

@group(0) @binding(0) var<uniform>             params  : SortParams;
@group(0) @binding(1) var<storage, read_write> sortBuf : array<u32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n / 2u) { return; }

  let ixj = i ^ params.j;
  if (ixj <= i) { return; }

  let ai = i   * 2u;   // key offset for element i
  let bi = ixj * 2u;   // key offset for element ixj

  let ka = sortBuf[ai];
  let kb = sortBuf[bi];

  // Ascending sort (back-to-front → smallest depth key first)
  let shouldSwap = (ka > kb);

  if (shouldSwap) {
    let ia = sortBuf[ai + 1u];
    let ib = sortBuf[bi + 1u];
    sortBuf[ai]      = kb;
    sortBuf[ai + 1u] = ib;
    sortBuf[bi]      = ka;
    sortBuf[bi + 1u] = ia;
  }
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Radix sort: histogram pass
// Counts occurrences of each 8-bit radix digit within each workgroup.
// Two passes total: LSB 8 bits, then MSB 8 bits.
// ─────────────────────────────────────────────────────────────────────────────

const RADIX_HISTOGRAM_WGSL = /* wgsl */`
struct RadixParams {
  n        : u32,   // total pair count
  shift    : u32,   // bit shift for current pass (0 = LSB, 8 = MSB)
  numWGs   : u32,   // total workgroups dispatched
  _pad     : u32,
}

@group(0) @binding(0) var<uniform>             params : RadixParams;
@group(0) @binding(1) var<storage, read>       keys   : array<u32>;
@group(0) @binding(2) var<storage, read_write> histo  : array<atomic<u32>>;

var<workgroup> localHisto: array<atomic<u32>, 256>;

@compute @workgroup_size(${WG})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id)         wgid: vec3<u32>,
  @builtin(local_invocation_id)  lid: vec3<u32>,
) {
  // Clear local histogram bins (each thread clears WG_SIZE / 256 bins)
  for (var b = lid.x; b < 256u; b += ${WG}u) {
    atomicStore(&localHisto[b], 0u);
  }
  workgroupBarrier();

  // Tally keys in this workgroup
  let idx = gid.x;
  if (idx < params.n) {
    let key  = keys[idx * 2u];  // key is at even indices in (key,index) pairs
    let digit = (key >> params.shift) & 0xFFu;
    atomicAdd(&localHisto[digit], 1u);
  }
  workgroupBarrier();

  // Write local histogram to global histogram: histo[wgid * 256 + bin]
  for (var b = lid.x; b < 256u; b += ${WG}u) {
    let count = atomicLoad(&localHisto[b]);
    if (count > 0u) {
      atomicAdd(&histo[b * params.numWGs + wgid.x], count);
    }
  }
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Radix sort: prefix sum pass
// Exclusive prefix sum over the flattened histogram (256 × numWGs entries).
// Simple sequential scan within a single workgroup — fine for counts ≤ 64K.
// ─────────────────────────────────────────────────────────────────────────────

const RADIX_PREFIX_SUM_WGSL = /* wgsl */`
struct PrefixParams {
  totalBins : u32,   // 256 * numWGs
  _pad0     : u32,
  _pad1     : u32,
  _pad2     : u32,
}

@group(0) @binding(0) var<uniform>             params : PrefixParams;
@group(0) @binding(1) var<storage, read_write> histo  : array<u32>;

// Blelloch-style prefix sum within a single workgroup for small arrays.
// For larger arrays (> WG), we do multiple sequential iterations.
@compute @workgroup_size(1)
fn main() {
  var sum = 0u;
  for (var i = 0u; i < params.totalBins; i++) {
    let val = histo[i];
    histo[i] = sum;
    sum += val;
  }
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Radix sort: scatter pass
// Reads (key, index) pairs, extracts radix digit, looks up prefix-sum offset,
// and scatters into the output buffer in sorted order.
// ─────────────────────────────────────────────────────────────────────────────

const RADIX_SCATTER_WGSL = /* wgsl */`
struct RadixParams {
  n        : u32,
  shift    : u32,
  numWGs   : u32,
  _pad     : u32,
}

@group(0) @binding(0) var<uniform>             params  : RadixParams;
@group(0) @binding(1) var<storage, read>       src     : array<u32>;
@group(0) @binding(2) var<storage, read_write> dst     : array<u32>;
@group(0) @binding(3) var<storage, read_write> offsets : array<atomic<u32>>;

@compute @workgroup_size(${WG})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id)         wgid: vec3<u32>,
) {
  let idx = gid.x;
  if (idx >= params.n) { return; }

  let key   = src[idx * 2u];
  let val   = src[idx * 2u + 1u];
  let digit = (key >> params.shift) & 0xFFu;

  // Atomic increment of the global offset for this (digit, workgroup) bin
  let binIdx = digit * params.numWGs + wgid.x;
  let dstIdx = atomicAdd(&offsets[binIdx], 1u);

  dst[dstIdx * 2u]      = key;
  dst[dstIdx * 2u + 1u] = val;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Fill indirect draw buffer (compute)
// Populates drawIndirect params for each visible layer.
// ─────────────────────────────────────────────────────────────────────────────

const FILL_INDIRECT_WGSL = /* wgsl */`
struct IndirectParams {
  vertexCount   : u32,   // always 6 (two triangles per quad)
  instanceCount : u32,   // particleCount for this layer
  firstVertex   : u32,   // 0
  firstInstance : u32,   // 0
}

struct FillUniforms {
  particleCount : u32,
  _pad0         : u32,
  _pad1         : u32,
  _pad2         : u32,
}

@group(0) @binding(0) var<uniform>             uni  : FillUniforms;
@group(0) @binding(1) var<storage, read_write> indirect : IndirectParams;

@compute @workgroup_size(1)
fn main() {
  indirect.vertexCount   = 6u;
  indirect.instanceCount = uni.particleCount;
  indirect.firstVertex   = 0u;
  indirect.firstInstance = 0u;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Alpha pass vertex shader
// Reads particle position from tPos via the sort-remapped index.
// Per-layer type size attenuation + colour tint varying.
// Uniform struct expanded to 20 f32 slots (80 bytes).
// ─────────────────────────────────────────────────────────────────────────────

const ALPHA_VERTEX_WGSL = /* wgsl */`
struct AlphaUniforms {
  domainW       : f32,    // 0
  domainH       : f32,    // 1
  scaleX        : f32,    // 2 = 2 / domainW
  scaleY        : f32,    // 3 = 2 / domainH
  uSize         : f32,    // 4
  glowScale     : f32,    // 5
  glowAlpha     : f32,    // 6
  glowSigma     : f32,    // 7
  maxGlowEnergy : f32,    // 8
  tintR         : f32,    // 9
  tintG         : f32,    // 10
  tintB         : f32,    // 11
  layerOffset   : u32,    // 12
  particleCount : u32,    // 13
  texW          : u32,    // 14
  texH          : u32,    // 15
  layerType     : u32,    // 16
  globalGlow    : f32,    // 17 global glow intensity
  _pad0         : u32,    // 18
  _pad1         : u32,    // 19
}

@group(0) @binding(0) var<uniform>       uni     : AlphaUniforms;
@group(0) @binding(1) var                tPos    : texture_2d<f32>;
@group(0) @binding(2) var                sSampler: sampler;
@group(0) @binding(3) var<storage, read> sortBuf : array<u32>;

struct AlphaVertOut {
  @builtin(position) pos     : vec4f,
  @location(0)       vUv     : vec2f,
  @location(1)       vAlpha  : f32,
  @location(2)       vTravel : f32,
  @location(3)       vTint   : vec3f,
}

var<private> QUAD: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
  vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
);

@vertex fn vs_alpha(
  @builtin(vertex_index)   vi : u32,
  @builtin(instance_index) ii : u32,
) -> AlphaVertOut {
  let globalSlot = uni.layerOffset + ii;
  let remapped   = sortBuf[globalSlot * 2u + 1u];
  let localSlot  = remapped - uni.layerOffset;

  let texX = i32(localSlot % uni.texW);
  let texY = i32(localSlot / uni.texW);
  let p    = textureLoad(tPos, vec2<i32>(texX, texY), 0);

  let worldX = p.r;
  let worldY = p.g;
  let travel = p.b;
  let alpha  = p.a;

  let alive = select(0.0, 1.0, alpha > 0.004);

  // Size attenuation per layer type
  var sizeScale = 1.0;
  switch uni.layerType {
    case 0u: { // Flower — travel² decay
      sizeScale = clamp(1.0 - travel * travel, 0.0, 1.0);
    }
    case 1u: { // Spline — travel² decay
      sizeScale = clamp(1.0 - travel * travel, 0.0, 1.0);
    }
    case 2u: { // EdgeFlow — linear decay with minimum
      sizeScale = clamp(1.0 - travel * 0.5, 0.3, 1.0);
    }
    case 3u: { // CurlAura — constant (aura rings don't shrink)
      sizeScale = 1.0;
    }
    case 4u: { // Sparks — rapid cubic decay
      let t3 = travel * travel * travel;
      sizeScale = clamp(1.0 - t3, 0.0, 1.0);
    }
    case 5u: { // Custom — travel² decay (same as Flower default)
      sizeScale = clamp(1.0 - travel * travel, 0.0, 1.0);
    }
    default: {
      sizeScale = clamp(1.0 - travel * travel, 0.0, 1.0);
    }
  }

  let halfSize = uni.uSize * sizeScale * 0.5;

  let qUv  = QUAD[vi];
  let ndcX = worldX * uni.scaleX - 1.0 + qUv.x * halfSize * uni.scaleX;
  let ndcY = worldY * uni.scaleY - 1.0 + qUv.y * halfSize * uni.scaleY;

  var out: AlphaVertOut;
  out.pos     = vec4f(ndcX * alive, ndcY * alive, 0.0, 1.0);
  out.vUv     = qUv;
  out.vAlpha  = alpha;
  out.vTravel = travel;
  out.vTint   = vec3f(uni.tintR, uni.tintG, uni.tintB);
  return out;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Alpha pass fragment shader
// Soft-disc SDF circle with sin(π·travel) fade and per-layer colour tint.
// ─────────────────────────────────────────────────────────────────────────────

const ALPHA_FRAGMENT_WGSL = /* wgsl */`
struct AlphaUniforms {
  domainW       : f32,
  domainH       : f32,
  scaleX        : f32,
  scaleY        : f32,
  uSize         : f32,
  glowScale     : f32,
  glowAlpha     : f32,
  glowSigma     : f32,
  maxGlowEnergy : f32,
  tintR         : f32,
  tintG         : f32,
  tintB         : f32,
  layerOffset   : u32,
  particleCount : u32,
  texW          : u32,
  texH          : u32,
  layerType     : u32,
  globalGlow    : f32,
  _pad0         : u32,
  _pad1         : u32,
}

@group(0) @binding(0) var<uniform> uni : AlphaUniforms;

struct AlphaFragIn {
  @location(0) vUv     : vec2f,
  @location(1) vAlpha  : f32,
  @location(2) vTravel : f32,
  @location(3) vTint   : vec3f,
}

@fragment fn fs_alpha(in: AlphaFragIn) -> @location(0) vec4f {
  let r2 = dot(in.vUv, in.vUv);
  if (r2 > 1.0) { discard; }

  // Soft edge falloff to avoid hard-clipped discs
  let edge = 1.0 - smoothstep(0.64, 1.0, r2);

  // AT sin(π·travel) alpha fade
  let fade = sin(3.14159265 * clamp(in.vTravel, 0.0, 1.0));

  // Core colour: warm centre tinted by per-layer colour
  let coreWhite = mix(vec3f(1.0, 0.78, 0.55), vec3f(1.0, 1.0, 1.0), 1.0 - r2);
  let coreColor = coreWhite * in.vTint;

  let finalA = in.vAlpha * fade * edge;
  return vec4f(coreColor * finalA, finalA);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Glow pass vertex shader
// Same as alpha VS but quad is scaled by glowScale.
// Includes per-layer tint varying.
// ─────────────────────────────────────────────────────────────────────────────

const GLOW_VERTEX_WGSL = /* wgsl */`
struct AlphaUniforms {
  domainW       : f32,
  domainH       : f32,
  scaleX        : f32,
  scaleY        : f32,
  uSize         : f32,
  glowScale     : f32,
  glowAlpha     : f32,
  glowSigma     : f32,
  maxGlowEnergy : f32,
  tintR         : f32,
  tintG         : f32,
  tintB         : f32,
  layerOffset   : u32,
  particleCount : u32,
  texW          : u32,
  texH          : u32,
  layerType     : u32,
  globalGlow    : f32,
  _pad0         : u32,
  _pad1         : u32,
}

@group(0) @binding(0) var<uniform>       uni     : AlphaUniforms;
@group(0) @binding(1) var                tPos    : texture_2d<f32>;
@group(0) @binding(2) var                sSampler: sampler;
@group(0) @binding(3) var<storage, read> sortBuf : array<u32>;

struct GlowVertOut {
  @builtin(position) pos     : vec4f,
  @location(0)       vUv     : vec2f,
  @location(1)       vAlpha  : f32,
  @location(2)       vTravel : f32,
  @location(3)       vTint   : vec3f,
}

var<private> QUAD: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
  vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
);

@vertex fn vs_glow(
  @builtin(vertex_index)   vi : u32,
  @builtin(instance_index) ii : u32,
) -> GlowVertOut {
  let globalSlot = uni.layerOffset + ii;
  let remapped   = sortBuf[globalSlot * 2u + 1u];
  let localSlot  = remapped - uni.layerOffset;

  let texX = i32(localSlot % uni.texW);
  let texY = i32(localSlot / uni.texW);
  let p    = textureLoad(tPos, vec2<i32>(texX, texY), 0);

  let worldX = p.r;
  let worldY = p.g;
  let travel = p.b;
  let alpha  = p.a;

  let alive = select(0.0, 1.0, alpha > 0.004);

  // Size attenuation per layer type (same as alpha VS)
  var sizeScale = 1.0;
  switch uni.layerType {
    case 0u: { sizeScale = clamp(1.0 - travel * travel, 0.0, 1.0); }
    case 1u: { sizeScale = clamp(1.0 - travel * travel, 0.0, 1.0); }
    case 2u: { sizeScale = clamp(1.0 - travel * 0.5, 0.3, 1.0); }
    case 3u: { sizeScale = 1.0; }
    case 4u: { let t3 = travel * travel * travel; sizeScale = clamp(1.0 - t3, 0.0, 1.0); }
    case 5u: { sizeScale = clamp(1.0 - travel * travel, 0.0, 1.0); }
    default: { sizeScale = clamp(1.0 - travel * travel, 0.0, 1.0); }
  }

  // Glow quad is scaled up relative to the alpha quad
  let halfSize = uni.uSize * uni.glowScale * sizeScale * 0.5;

  let qUv  = QUAD[vi];
  let ndcX = worldX * uni.scaleX - 1.0 + qUv.x * halfSize * uni.scaleX;
  let ndcY = worldY * uni.scaleY - 1.0 + qUv.y * halfSize * uni.scaleY;

  var out: GlowVertOut;
  out.pos     = vec4f(ndcX * alive, ndcY * alive, 0.0, 1.0);
  out.vUv     = qUv;
  out.vAlpha  = alpha;
  out.vTravel = travel;
  out.vTint   = vec3f(uni.tintR, uni.tintG, uni.tintB);
  return out;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Glow pass fragment shader
// Variable-sigma Gaussian kernel with energy conservation and per-layer tint.
// ─────────────────────────────────────────────────────────────────────────────

const GLOW_FRAGMENT_WGSL = /* wgsl */`
struct AlphaUniforms {
  domainW       : f32,
  domainH       : f32,
  scaleX        : f32,
  scaleY        : f32,
  uSize         : f32,
  glowScale     : f32,
  glowAlpha     : f32,
  glowSigma     : f32,
  maxGlowEnergy : f32,
  tintR         : f32,
  tintG         : f32,
  tintB         : f32,
  layerOffset   : u32,
  particleCount : u32,
  texW          : u32,
  texH          : u32,
  layerType     : u32,
  globalGlow    : f32,
  _pad0         : u32,
  _pad1         : u32,
}

@group(0) @binding(0) var<uniform> uni : AlphaUniforms;

struct GlowFragIn {
  @location(0) vUv     : vec2f,
  @location(1) vAlpha  : f32,
  @location(2) vTravel : f32,
  @location(3) vTint   : vec3f,
}

@fragment fn fs_glow(in: GlowFragIn) -> @location(0) vec4f {
  let r2 = dot(in.vUv, in.vUv);
  if (r2 > 1.0) { discard; }

  // Variable-sigma Gaussian kernel: exp(-sigma * r²)
  let sigma   = uni.glowSigma;
  let gauss   = exp(-sigma * r2);

  // AT sin(π·travel) fade for temporal coherence
  let fade    = sin(3.14159265 * clamp(in.vTravel, 0.0, 1.0));

  // Glow colour: warm halo tinted by per-layer colour
  let glowBase = vec3f(1.0, 0.92, 0.70);
  let glowTint = glowBase * in.vTint;

  // Apply global glow intensity
  let glowIntensity = uni.glowAlpha * uni.globalGlow;

  // Compute raw glow energy
  let rawEnergy = glowIntensity * in.vAlpha * fade * gauss;

  // Energy conservation: clamp to maxGlowEnergy
  let finalA = min(rawEnergy, uni.maxGlowEnergy);

  return vec4f(glowTint * finalA, finalA);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// CPU helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Round x up to the nearest power of two. */
function nextPow2(x: number): number {
  if (x <= 1) return 1;
  let n = 1;
  while (n < x) n <<= 1;
  return n;
}

/** Map SortKeyMode string to integer for WGSL uniform. */
function sortKeyModeToInt(mode: SortKeyMode): number {
  switch (mode) {
    case 'travel': return 0;
    case 'worldY': return 1;
    case 'worldZ': return 2;
    case 'manual': return 3;
    default:       return 0;
  }
}

/** Build the per-layer uniform data for alpha / glow passes (20 f32 slots = 80 bytes). */
function buildLayerUniforms(
  meta:              GPULayerMeta,
  domainW:           number,
  domainH:           number,
  globalGlowIntensity: number,
): Float32Array {
  const data = new Float32Array(20);
  data[0]  = domainW;                            // domainW
  data[1]  = domainH;                            // domainH
  data[2]  = 2.0 / domainW;                      // scaleX
  data[3]  = 2.0 / domainH;                      // scaleY
  data[4]  = meta.uSize;                          // uSize
  data[5]  = meta.glowScale;                      // glowScale
  data[6]  = meta.glowAlpha;                      // glowAlpha
  data[7]  = meta.glowSigma;                      // glowSigma
  data[8]  = meta.maxGlowEnergy;                  // maxGlowEnergy
  data[9]  = meta.colorTint.r;                    // tintR
  data[10] = meta.colorTint.g;                    // tintG
  data[11] = meta.colorTint.b;                    // tintB

  const u32 = new Uint32Array(data.buffer);
  u32[12] = meta.layerOffset;                     // layerOffset
  u32[13] = meta.particleCount;                   // particleCount
  u32[14] = meta.texW;                            // texW
  u32[15] = meta.texH;                            // texH
  u32[16] = meta.type;                            // layerType

  data[17] = globalGlowIntensity;                 // globalGlow
  u32[18] = 0;                                    // _pad0
  u32[19] = 0;                                    // _pad1

  return data;
}

// ─────────────────────────────────────────────────────────────────────────────
// ParticleCompositor — Main class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * ParticleCompositor
 *
 * Unified compositor for multiple AT particle layers.  After calling build(),
 * drive it each frame with:
 *
 *   compositor.update(enc, elapsed)       — refresh uniforms + indirect
 *   compositor.sort(enc)                  — GPU sort (radix / bitonic / none)
 *   compositor.renderAlpha(enc, view)     — alpha-blended layer
 *   compositor.renderGlow(enc, view)      — additive glow layer
 *
 * Or use the one-shot convenience:
 *   compositor.composite(enc, view, elapsed)
 */
export class ParticleCompositor {
  private readonly device: any /*GPUDevice*/;
  private readonly canvas: HTMLCanvasElement;

  private layers: LayerDescriptor[]   = [];
  private gpuMeta: GPULayerMeta[]     = [];

  private cfg: Required<CompositorConfig> = {
    clearColor:           { r: 0, g: 0, b: 0, a: 0 },
    enableGlow:           true,
    globalGlowIntensity:  1.0,
    maxGlobalGlowEnergy:  2.0,
    sortAlgorithm:        'auto',
    sortKey:              'travel',
  };

  // ── Combined sort buffers ───────────────────────────────────────────────────
  /** Interleaved (key u32, index u32) × totalSortSlots. */
  private sortBufA!:     GPUBuffer;   // primary sort buffer
  private sortBufB!:     GPUBuffer;   // secondary (for radix ping-pong)
  private totalSlots  = 0;
  private sortedCount = 0;   // rounded to pow2

  // ── Key extraction pipeline ──────────────────────────────────────────────────
  private keyPipeline!:  GPUComputePipeline;
  private keyUniBufs:    GPUBuffer[]    = [];
  private keyBGs:        GPUBindGroup[] = [];

  // ── Bitonic sort pipeline ────────────────────────────────────────────────────
  private bitonicPipeline!: any /*GPUComputePipeline*/;
  private bitonicUniBuf!:   GPUBuffer;
  private bitonicBG!:       GPUBindGroup;

  // ── Radix sort pipelines ─────────────────────────────────────────────────────
  private radixHistoPipeline!:   GPUComputePipeline;
  private radixPrefixPipeline!:  GPUComputePipeline;
  private radixScatterPipeline!: any /*GPUComputePipeline*/;
  private radixParamsBuf!:       GPUBuffer;
  private radixHistoBuf!:        GPUBuffer;
  private prefixParamsBuf!:      GPUBuffer;
  // Bind groups created per-frame since they depend on ping-pong buffers
  private radixHistoBGLayout!:   GPUBindGroupLayout;
  private radixPrefixBGLayout!:  GPUBindGroupLayout;
  private radixScatterBGLayout!: GPUBindGroupLayout;

  // ── Fill indirect pipeline ───────────────────────────────────────────────────
  private fillIndirectPipeline!: any /*GPUComputePipeline*/;
  private indirectBufs:          GPUBuffer[]    = [];
  private fillIndirectUniBufs:   GPUBuffer[]    = [];
  private fillIndirectBGs:       GPUBindGroup[] = [];

  // ── Alpha / Glow render pipelines ────────────────────────────────────────────
  /** One alpha pipeline per unique blend mode. */
  private alphaPipelines:  Map<string, GPURenderPipeline> = new Map();
  private glowPipeline!:   GPURenderPipeline;
  private sampler!:        GPUSampler;

  /** Per-layer render bind groups for alpha pass. */
  private alphaBGs:       GPUBindGroup[] = [];
  /** Per-layer render bind groups for glow pass. */
  private glowBGs:        GPUBindGroup[] = [];
  /** Per-layer uniform buffers (shared between alpha and glow BGs). */
  private renderUniBufs:  GPUBuffer[]    = [];

  private built = false;

  /** Runtime diagnostics — updated each frame in update(). */
  private _diag: CompositorDiagnostics = {
    sortAlgorithm:   'none',
    activeLayers:    0,
    activeParticles: 0,
    sortBufferSlots: 0,
  };

  constructor(
    device: any /*GPUDevice*/,
    canvas: HTMLCanvasElement,
    config: Partial<CompositorConfig> = {},
  ) {
    this.device = device;
    this.canvas = canvas;
    Object.assign(this.cfg, config);
  }

  // ── Layer registration ───────────────────────────────────────────────────────

  /**
   * Register a tPos-backed particle layer.
   * Call before build(). Layers are drawn in ascending zOrder order.
   */
  addLayer(desc: LayerDescriptor): void {
    this.layers.push(desc);
    if (this.built) {
      console.warn('[ParticleCompositor] addLayer() called after build(); call build() again.');
    }
  }

  /** Remove a previously registered layer by id. Requires rebuild. */
  removeLayer(id: string): void {
    this.layers = this.layers.filter(l => l.id !== id);
    if (this.built) {
      console.warn('[ParticleCompositor] removeLayer() called after build(); call build() again.');
    }
  }

  // ── Build ────────────────────────────────────────────────────────────────────

  /**
   * Build all GPU pipelines, buffers, and bind groups.
   * Must be called after all layers are registered, and again after any
   * addLayer / removeLayer call.
   */
  async build(): Promise<void> {
    if (this.built) this._destroy();

    const { device } = this;
    const fmt = navigator.gpu.getPreferredCanvasFormat();

    // Sort layers by zOrder
    const sortedLayers = [...this.layers].sort(
      (a, b) => (a.zOrder ?? 0) - (b.zOrder ?? 0),
    );

    // ── Compute GPU metadata and slot offsets ───────────────────────────────────
    this.gpuMeta = [];
    let offset = 0;
    for (const l of sortedLayers) {
      const sorted = nextPow2(l.particleCount);
      this.gpuMeta.push({
        id:            l.id,
        type:          l.type,
        tPosView:      l.tPosView,
        texW:          l.texW,
        texH:          l.texH,
        particleCount: l.particleCount,
        sortedCount:   sorted,
        layerOffset:   offset,
        uSize:         l.uSize,
        glowScale:     l.glowScale    ?? 2.0,
        glowAlpha:     l.glowAlpha    ?? 0.25,
        glowSigma:     l.glowSigma    ?? 4.0,
        maxGlowEnergy: l.maxGlowEnergy ?? 1.0,
        colorTint:     l.colorTint     ?? { r: 1, g: 1, b: 1 },
        blendMode:     l.blendMode     ?? 'premultiplied',
        customBlend:   l.customBlend,
        hidden:        l.hidden        ?? false,
        skipSort:      l.skipSort      ?? false,
        sortKeyMode:   l.sortKeyMode   ?? this.cfg.sortKey,
      });
      offset += sorted;
    }
    this.totalSlots  = offset;
    this.sortedCount = nextPow2(offset);

    if (this.sortedCount > MAX_SORT_PARTICLES) {
      console.warn(
        `[ParticleCompositor] Combined particle count ${this.sortedCount} ` +
        `exceeds MAX_SORT_PARTICLES ${MAX_SORT_PARTICLES}; clamping.`,
      );
      this.sortedCount = MAX_SORT_PARTICLES;
    }

    // ── Sort buffers (A = primary, B = ping-pong for radix) ─────────────────────
    const sortBufSize = this.sortedCount * SORT_PAIR_STRIDE;
    this.sortBufA = device.createBuffer({
      size:  sortBufSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Uint32Array(this.sortBufA.getMappedRange()).fill(0xFFFFFFFF);
    this.sortBufA.unmap();

    this.sortBufB = device.createBuffer({
      size:  sortBufSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // ── Sampler (nearest — tPos is a data texture) ─────────────────────────────
    this.sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });

    // ── Key extraction pipeline ─────────────────────────────────────────────────
    const keyMod = device.createShaderModule({ code: KEY_EXTRACT_WGSL });
    this.keyPipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: keyMod, entryPoint: 'main' },
    });

    this.keyUniBufs = [];
    this.keyBGs     = [];
    for (const meta of this.gpuMeta) {
      const keyUni = device.createBuffer({
        size:  32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this._writeKeyUniforms(keyUni, meta);

      const bg0 = device.createBindGroup({
        layout: this.keyPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: keyUni } }],
      });
      const bg1 = device.createBindGroup({
        layout: this.keyPipeline.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: meta.tPosView },
          { binding: 1, resource: { buffer: this.sortBufA } },
        ],
      });
      this.keyUniBufs.push(keyUni);
      this.keyBGs.push(bg0, bg1);
    }

    // ── Bitonic sort pipeline ───────────────────────────────────────────────────
    const bitonicMod = device.createShaderModule({ code: BITONIC_SORT_WGSL });
    this.bitonicPipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: bitonicMod, entryPoint: 'main' },
    });
    this.bitonicUniBuf = device.createBuffer({
      size:  16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.bitonicBG = device.createBindGroup({
      layout: this.bitonicPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bitonicUniBuf } },
        { binding: 1, resource: { buffer: this.sortBufA } },
      ],
    });

    // ── Radix sort pipelines ────────────────────────────────────────────────────
    const radixHistoMod   = device.createShaderModule({ code: RADIX_HISTOGRAM_WGSL });
    const radixPrefixMod  = device.createShaderModule({ code: RADIX_PREFIX_SUM_WGSL });
    const radixScatterMod = device.createShaderModule({ code: RADIX_SCATTER_WGSL });

    this.radixHistoPipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: radixHistoMod, entryPoint: 'main' },
    });
    this.radixPrefixPipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: radixPrefixMod, entryPoint: 'main' },
    });
    this.radixScatterPipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: radixScatterMod, entryPoint: 'main' },
    });

    // Cache bind group layouts for per-frame BG creation
    this.radixHistoBGLayout   = this.radixHistoPipeline.getBindGroupLayout(0);
    this.radixPrefixBGLayout  = this.radixPrefixPipeline.getBindGroupLayout(0);
    this.radixScatterBGLayout = this.radixScatterPipeline.getBindGroupLayout(0);

    // Radix parameter buffers
    this.radixParamsBuf = device.createBuffer({
      size:  16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.prefixParamsBuf = device.createBuffer({
      size:  16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Histogram buffer: 256 bins × numWGs
    const numWGs = Math.ceil(this.sortedCount / WG);
    const histoSize = RADIX_BINS * numWGs * 4;
    this.radixHistoBuf = device.createBuffer({
      size:  Math.max(histoSize, 256),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // ── Fill indirect pipeline ──────────────────────────────────────────────────
    const fillMod = device.createShaderModule({ code: FILL_INDIRECT_WGSL });
    this.fillIndirectPipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: fillMod, entryPoint: 'main' },
    });

    this.indirectBufs        = [];
    this.fillIndirectUniBufs = [];
    this.fillIndirectBGs     = [];

    for (const meta of this.gpuMeta) {
      const indirectBuf = device.createBuffer({
        size:  INDIRECT_DRAW_BYTES,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const fillUni = device.createBuffer({
        size:  16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const fillBG = device.createBindGroup({
        layout: this.fillIndirectPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fillUni } },
          { binding: 1, resource: { buffer: indirectBuf } },
        ],
      });
      this.indirectBufs.push(indirectBuf);
      this.fillIndirectUniBufs.push(fillUni);
      this.fillIndirectBGs.push(fillBG);

      // Write initial particle count
      const fillData = new Uint32Array([meta.particleCount, 0, 0, 0]);
      device.queue.writeBuffer(fillUni, 0, fillData);
    }

    // ── Collect unique blend modes for alpha pipeline variants ──────────────────
    const blendKeys = new Set<string>();
    for (const meta of this.gpuMeta) {
      blendKeys.add(this._blendKey(meta));
    }

    const alphaVsMod = device.createShaderModule({ code: ALPHA_VERTEX_WGSL });
    const alphaFsMod = device.createShaderModule({ code: ALPHA_FRAGMENT_WGSL });

    this.alphaPipelines = new Map();
    for (const bk of blendKeys) {
      const blendState = this._blendStateFromKey(bk);
      const pipeline = device.createRenderPipeline({
        layout:   'auto',
        vertex:   { module: alphaVsMod, entryPoint: 'vs_alpha' },
        fragment: {
          module: alphaFsMod, entryPoint: 'fs_alpha',
          targets: [{ format: fmt, blend: blendState }],
        },
        primitive: { topology: 'triangle-list' },
      });
      this.alphaPipelines.set(bk, pipeline);
    }

    // ── Glow render pipeline (always additive) ─────────────────────────────────
    const glowVsMod = device.createShaderModule({ code: GLOW_VERTEX_WGSL });
    const glowFsMod = device.createShaderModule({ code: GLOW_FRAGMENT_WGSL });
    this.glowPipeline = device.createRenderPipeline({
      layout:   'auto',
      vertex:   { module: glowVsMod, entryPoint: 'vs_glow' },
      fragment: {
        module: glowFsMod, entryPoint: 'fs_glow',
        targets: [{
          format: fmt,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // ── Per-layer render bind groups ───────────────────────────────────────────
    this.renderUniBufs = [];
    this.alphaBGs      = [];
    this.glowBGs       = [];

    for (let i = 0; i < this.gpuMeta.length; i++) {
      const meta     = this.gpuMeta[i];
      const unifData = buildLayerUniforms(meta, this.canvas.width || 1, this.canvas.height || 1, this.cfg.globalGlowIntensity);
      const uniBuf   = device.createBuffer({
        size:  unifData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(uniBuf, 0, unifData);
      this.renderUniBufs.push(uniBuf);

      // Get the correct alpha pipeline for this layer's blend mode
      const bk = this._blendKey(meta);
      const alphaPipe = this.alphaPipelines.get(bk)!;

      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: uniBuf } },
        { binding: 1, resource: meta.tPosView },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: { buffer: this.sortBufA } },
      ];
      this.alphaBGs.push(device.createBindGroup({
        layout: alphaPipe.getBindGroupLayout(0),
        entries,
      }));
      this.glowBGs.push(device.createBindGroup({
        layout: this.glowPipeline.getBindGroupLayout(0),
        entries,
      }));
    }

    this.built = true;
    console.log(
      `[ParticleCompositor] M767 built: ${this.gpuMeta.length} layers, ` +
      `${this.totalSlots} total slots, sort buffer = ${this.sortedCount} × 8 B, ` +
      `blend variants = ${this.alphaPipelines.size}`,
    );
  }

  // ── Per-frame API ─────────────────────────────────────────────────────────────

  /**
   * Update compositor uniform buffers (domain size may change on resize).
   * Fills indirect draw buffers for each visible layer.
   */
  update(_enc: any /*GPUCommandEncoder*/, _elapsed: number): void {
    if (!this.built) return;
    const { device, canvas, gpuMeta, renderUniBufs } = this;
    const dw = canvas.width  || 1;
    const dh = canvas.height || 1;

    let activeLayers = 0;
    let activeParticles = 0;

    for (let i = 0; i < gpuMeta.length; i++) {
      const meta     = gpuMeta[i];
      const unifData = buildLayerUniforms(meta, dw, dh, this.cfg.globalGlowIntensity);
      device.queue.writeBuffer(renderUniBufs[i], 0, unifData);

      // Update indirect draw buffer
      const fillData = new Uint32Array([
        meta.hidden ? 0 : meta.particleCount, 0, 0, 0,
      ]);
      device.queue.writeBuffer(this.fillIndirectUniBufs[i], 0, fillData);

      if (!meta.hidden) {
        activeLayers++;
        activeParticles += meta.particleCount;
      }
    }

    // Update diagnostics
    this._diag.activeLayers    = activeLayers;
    this._diag.activeParticles = activeParticles;
    this._diag.sortBufferSlots = this.sortedCount;
  }

  /**
   * Encode the depth sort sequence (key extract + radix/bitonic sort) into enc.
   * Must be called before renderAlpha / renderGlow.
   */
  sort(enc: any /*GPUCommandEncoder*/): void {
    if (!this.built) return;

    const algo = this._chooseSortAlgorithm();
    this._diag.sortAlgorithm = algo;

    if (algo === 'none') return;

    // ── Phase 1: extract keys from each layer's tPos ───────────────────────────
    const keyPass = enc.beginComputePass({ label: 'compositor:keyExtract' });
    keyPass.setPipeline(this.keyPipeline);

    for (let i = 0; i < this.gpuMeta.length; i++) {
      const meta = this.gpuMeta[i];
      if (meta.hidden || meta.skipSort) continue;
      const wgs = Math.ceil(meta.particleCount / WG);
      keyPass.setBindGroup(0, this.keyBGs[i * 2]);
      keyPass.setBindGroup(1, this.keyBGs[i * 2 + 1]);
      keyPass.dispatchWorkgroups(wgs);
    }
    keyPass.end();

    // ── Phase 2: sort ──────────────────────────────────────────────────────────
    if (this.sortedCount <= 1) return;

    if (algo === 'bitonic') {
      this._sortBitonic(enc);
    } else {
      this._sortRadix(enc);
    }
  }

  /**
   * Encode the alpha-blended render pass for all visible layers.
   */
  renderAlpha(
    enc:       GPUCommandEncoder,
    colorView: any /*GPUTextureView*/,
    loadOp:    GPULoadOp = 'clear',
  ): void {
    if (!this.built || this.gpuMeta.length === 0) return;

    const pass = enc.beginRenderPass({
      label:            'compositor:alpha',
      colorAttachments: [{
        view:       colorView,
        loadOp,
        storeOp:    'store',
        clearValue: this.cfg.clearColor,
      }],
    });

    for (let i = 0; i < this.gpuMeta.length; i++) {
      const meta = this.gpuMeta[i];
      if (meta.hidden) continue;

      const bk   = this._blendKey(meta);
      const pipe = this.alphaPipelines.get(bk)!;
      pass.setPipeline(pipe);
      pass.setBindGroup(0, this.alphaBGs[i]);
      // Use indirect draw buffer
      pass.drawIndirect(this.indirectBufs[i], 0);
    }
    pass.end();
  }

  /**
   * Encode the additive glow render pass for all visible layers.
   * Must be called after renderAlpha (uses 'load' to preserve alpha-pass output).
   */
  renderGlow(enc: any /*GPUCommandEncoder*/, colorView: any /*GPUTextureView*/): void {
    if (!this.built || !this.cfg.enableGlow || this.gpuMeta.length === 0) return;

    const pass = enc.beginRenderPass({
      label:            'compositor:glow',
      colorAttachments: [{
        view:    colorView,
        loadOp:  'load',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(this.glowPipeline);

    for (let i = 0; i < this.gpuMeta.length; i++) {
      const meta = this.gpuMeta[i];
      if (meta.hidden) continue;
      pass.setBindGroup(0, this.glowBGs[i]);
      pass.drawIndirect(this.indirectBufs[i], 0);
    }
    pass.end();
  }

  /**
   * Convenience method: sort + renderAlpha + renderGlow in one call.
   */
  composite(
    enc:     GPUCommandEncoder,
    view:    GPUTextureView,
    elapsed: number,
    loadOp:  GPULoadOp = 'clear',
  ): void {
    this.update(enc, elapsed);
    this._fillIndirect(enc);
    this.sort(enc);
    this.renderAlpha(enc, view, loadOp);
    this.renderGlow(enc, view);
  }

  // ── Accessors ─────────────────────────────────────────────────────────────────

  get layerCount():         number  { return this.gpuMeta.length; }
  get totalParticleSlots(): number  { return this.totalSlots; }
  get isBuilt():            boolean { return this.built; }

  /** Get runtime diagnostics snapshot. */
  get diagnostics(): CompositorDiagnostics {
    return { ...this._diag };
  }

  // ── Live config updates (no rebuild required) ────────────────────────────────

  /** Toggle the glow pass at runtime. */
  setGlowEnabled(enabled: boolean): void {
    this.cfg.enableGlow = enabled;
  }

  /** Set global glow intensity multiplier (0–1). */
  setGlobalGlowIntensity(intensity: number): void {
    this.cfg.globalGlowIntensity = Math.max(0, Math.min(1, intensity));
  }

  /** Set glow sigma for a specific layer by id at runtime. */
  setLayerGlowSigma(id: string, sigma: number): void {
    const meta = this._findMeta(id);
    if (meta) meta.glowSigma = sigma;
  }

  /** Set colour tint for a specific layer by id at runtime. */
  setLayerColorTint(id: string, tint: ColorRGB): void {
    const meta = this._findMeta(id);
    if (meta) meta.colorTint = { ...tint };
  }

  /** Set maxGlowEnergy for a specific layer by id at runtime. */
  setLayerMaxGlowEnergy(id: string, energy: number): void {
    const meta = this._findMeta(id);
    if (meta) meta.maxGlowEnergy = energy;
  }

  /** Update glowAlpha for a specific layer by id at runtime. */
  setLayerGlowAlpha(id: string, alpha: number): void {
    const meta = this._findMeta(id);
    if (meta) meta.glowAlpha = alpha;
  }

  /** Update uSize for a specific layer by id at runtime. */
  setLayerSize(id: string, uSize: number): void {
    const meta = this._findMeta(id);
    if (meta) meta.uSize = uSize;
  }

  /** Toggle layer visibility at runtime. Hidden layers skip draw. */
  setLayerVisible(id: string, visible: boolean): void {
    const meta = this._findMeta(id);
    if (meta) meta.hidden = !visible;
  }

  /** Update active particle count for a layer (e.g. after emitter changes). */
  updateLayerParticleCount(id: string, count: number): void {
    const meta = this._findMeta(id);
    if (meta) meta.particleCount = count;
  }

  /** Hot-swap the tPosView for a layer (e.g. double-buffer flip). */
  updateLayerTexture(id: string, tPosView: any /*GPUTextureView*/): void {
    const idx = this.gpuMeta.findIndex(m => m.id === id);
    if (idx < 0) return;
    this.gpuMeta[idx].tPosView = tPosView;
    // Rebuild bind groups for this layer
    if (this.built) {
      this._rebuildLayerBindGroups(idx);
    }
  }

  /** Set sort algorithm override. */
  setSortAlgorithm(algo: SortAlgorithm): void {
    this.cfg.sortAlgorithm = algo;
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────────

  destroy(): void { this._destroy(); }

  // ── Private: sort implementations ─────────────────────────────────────────────

  private _chooseSortAlgorithm(): 'radix' | 'bitonic' | 'none' {
    const algo = this.cfg.sortAlgorithm;
    if (algo === 'none') return 'none';
    if (algo === 'radix')  return 'radix';
    if (algo === 'bitonic') return 'bitonic';
    // 'auto' — radix for large counts, bitonic for small
    return this.sortedCount >= RADIX_THRESHOLD ? 'radix' : 'bitonic';
  }

  private _sortBitonic(enc: any /*GPUCommandEncoder*/): void {
    const sortPass = enc.beginComputePass({ label: 'compositor:bitonicSort' });
    sortPass.setPipeline(this.bitonicPipeline);
    sortPass.setBindGroup(0, this.bitonicBG);

    const n   = this.sortedCount;
    const wgs = Math.ceil(n / 2 / WG);

    for (let k = 2; k <= n; k <<= 1) {
      for (let j = k >> 1; j >= 1; j >>= 1) {
        const params = new Uint32Array([k, j, n, 0]);
        this.device.queue.writeBuffer(this.bitonicUniBuf, 0, params);
        sortPass.dispatchWorkgroups(wgs);
      }
    }
    sortPass.end();
  }

  private _sortRadix(enc: any /*GPUCommandEncoder*/): void {
    const n     = this.sortedCount;
    const numWG = Math.ceil(n / WG);
    const { device } = this;

    // Two radix passes: LSB 8 bits (shift=0), then next 8 bits (shift=8)
    // For 16-bit radix coverage (handles typical float-key distributions)
    const shifts = [0, 8];

    let srcBuf = this.sortBufA;
    let dstBuf = this.sortBufB;

    for (const shift of shifts) {
      // Clear histogram buffer
      device.queue.writeBuffer(
        this.radixHistoBuf, 0,
        new Uint32Array(RADIX_BINS * numWG).fill(0),
      );

      // Write radix params
      const radixParams = new Uint32Array([n, shift, numWG, 0]);
      device.queue.writeBuffer(this.radixParamsBuf, 0, radixParams);

      // Histogram pass
      const histoBG = device.createBindGroup({
        layout: this.radixHistoBGLayout,
        entries: [
          { binding: 0, resource: { buffer: this.radixParamsBuf } },
          { binding: 1, resource: { buffer: srcBuf } },
          { binding: 2, resource: { buffer: this.radixHistoBuf } },
        ],
      });
      const histoPass = enc.beginComputePass({ label: `compositor:radixHisto:${shift}` });
      histoPass.setPipeline(this.radixHistoPipeline);
      histoPass.setBindGroup(0, histoBG);
      histoPass.dispatchWorkgroups(numWG);
      histoPass.end();

      // Prefix sum pass
      const totalBins = RADIX_BINS * numWG;
      const prefixParams = new Uint32Array([totalBins, 0, 0, 0]);
      device.queue.writeBuffer(this.prefixParamsBuf, 0, prefixParams);

      const prefixBG = device.createBindGroup({
        layout: this.radixPrefixBGLayout,
        entries: [
          { binding: 0, resource: { buffer: this.prefixParamsBuf } },
          { binding: 1, resource: { buffer: this.radixHistoBuf } },
        ],
      });
      const prefixPass = enc.beginComputePass({ label: `compositor:radixPrefix:${shift}` });
      prefixPass.setPipeline(this.radixPrefixPipeline);
      prefixPass.setBindGroup(0, prefixBG);
      prefixPass.dispatchWorkgroups(1);
      prefixPass.end();

      // Scatter pass
      const scatterBG = device.createBindGroup({
        layout: this.radixScatterBGLayout,
        entries: [
          { binding: 0, resource: { buffer: this.radixParamsBuf } },
          { binding: 1, resource: { buffer: srcBuf } },
          { binding: 2, resource: { buffer: dstBuf } },
          { binding: 3, resource: { buffer: this.radixHistoBuf } },
        ],
      });
      const scatterPass = enc.beginComputePass({ label: `compositor:radixScatter:${shift}` });
      scatterPass.setPipeline(this.radixScatterPipeline);
      scatterPass.setBindGroup(0, scatterBG);
      scatterPass.dispatchWorkgroups(numWG);
      scatterPass.end();

      // Ping-pong: dst becomes src for next pass
      const tmp = srcBuf;
      srcBuf = dstBuf;
      dstBuf = tmp;
    }

    // After two passes, sorted data is in srcBuf.
    // If srcBuf !== sortBufA, copy result back to sortBufA (where render BGs point)
    if (srcBuf !== this.sortBufA) {
      enc.copyBufferToBuffer(srcBuf, 0, this.sortBufA, 0, n * SORT_PAIR_STRIDE);
    }
  }

  private _fillIndirect(enc: any /*GPUCommandEncoder*/): void {
    const fillPass = enc.beginComputePass({ label: 'compositor:fillIndirect' });
    fillPass.setPipeline(this.fillIndirectPipeline);
    for (let i = 0; i < this.gpuMeta.length; i++) {
      fillPass.setBindGroup(0, this.fillIndirectBGs[i]);
      fillPass.dispatchWorkgroups(1);
    }
    fillPass.end();
  }

  // ── Private: helpers ──────────────────────────────────────────────────────────

  private _findMeta(id: string): GPULayerMeta | undefined {
    return this.gpuMeta.find(m => m.id === id);
  }

  private _blendKey(meta: GPULayerMeta): string {
    if (meta.blendMode === 'custom' && meta.customBlend) {
      return 'custom:' + JSON.stringify(meta.customBlend);
    }
    return meta.blendMode;
  }

  private _blendStateFromKey(key: string): GPUBlendState {
    if (key.startsWith('custom:')) {
      try { return JSON.parse(key.substring(7)); }
      catch { return blendStateForPreset('premultiplied'); }
    }
    return blendStateForPreset(key as BlendPreset);
  }

  private _writeKeyUniforms(buf: any /*GPUBuffer*/, meta: GPULayerMeta): void {
    const data = new Uint32Array(8);
    data[0] = meta.layerOffset;
    data[1] = meta.particleCount;
    data[2] = meta.texW;
    data[3] = meta.texH;
    data[4] = sortKeyModeToInt(meta.sortKeyMode);
    this.device.queue.writeBuffer(buf, 0, data);
  }

  private _rebuildLayerBindGroups(idx: number): void {
    const { device } = this;
    const meta = this.gpuMeta[idx];

    // Rebuild key extraction BG1 (tPos binding changed)
    const bg1 = device.createBindGroup({
      layout: this.keyPipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: meta.tPosView },
        { binding: 1, resource: { buffer: this.sortBufA } },
      ],
    });
    this.keyBGs[idx * 2 + 1] = bg1;

    // Rebuild render bind groups
    const bk       = this._blendKey(meta);
    const alphaPipe = this.alphaPipelines.get(bk)!;
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.renderUniBufs[idx] } },
      { binding: 1, resource: meta.tPosView },
      { binding: 2, resource: this.sampler },
      { binding: 3, resource: { buffer: this.sortBufA } },
    ];
    this.alphaBGs[idx] = device.createBindGroup({
      layout: alphaPipe.getBindGroupLayout(0),
      entries,
    });
    this.glowBGs[idx] = device.createBindGroup({
      layout: this.glowPipeline.getBindGroupLayout(0),
      entries,
    });
  }

  private _destroy(): void {
    if (!this.built) return;
    this.sortBufA?.destroy();
    this.sortBufB?.destroy();
    this.bitonicUniBuf?.destroy();
    this.radixParamsBuf?.destroy();
    this.radixHistoBuf?.destroy();
    this.prefixParamsBuf?.destroy();
    for (const b of this.keyUniBufs) b.destroy();
    for (const b of this.renderUniBufs) b.destroy();
    for (const b of this.indirectBufs) b.destroy();
    for (const b of this.fillIndirectUniBufs) b.destroy();
    this.keyUniBufs          = [];
    this.keyBGs              = [];
    this.alphaBGs            = [];
    this.glowBGs             = [];
    this.renderUniBufs       = [];
    this.indirectBufs        = [];
    this.fillIndirectUniBufs = [];
    this.fillIndirectBGs     = [];
    this.gpuMeta             = [];
    this.alphaPipelines.clear();
    this.built = false;
  }
}

// ─── Factory helpers ──────────────────────────────────────────────────────────

/**
 * createCompositorForATRenderers
 *
 * Convenience factory: build a ParticleCompositor pre-wired for AT particle
 * renderers.  Supports flower, spline, edgeFlow, and sparks layers.  All
 * renderers must be fully built (`.build()` awaited) before calling this.
 *
 * @example
 * ```ts
 * import { createCompositorForATRenderers } from '$lib/sph/particle-compositor';
 *
 * const compositor = await createCompositorForATRenderers(device, canvas, {
 *   flower:   { renderer: flowerRenderer, glowScale: 2.2, glowAlpha: 0.30 },
 *   spline:   { renderer: splineLife,     glowScale: 1.6, glowAlpha: 0.18 },
 *   edgeFlow: { renderer: edgeFlowRenderer, glowScale: 1.8, glowAlpha: 0.22 },
 *   sparks:   { renderer: sparkSystem,    glowScale: 3.0, glowAlpha: 0.40, skipSort: true },
 * });
 *
 * // render loop:
 * compositor.composite(enc, swapchainView, elapsed);
 * ```
 */
export async function createCompositorForATRenderers(
  device:  GPUDevice,
  canvas:  HTMLCanvasElement,
  layers: {
    flower?: {
      renderer:   { tPosView: any /*GPUTextureView*/; activeParticleCount: number };
      glowScale?: number;
      glowAlpha?: number;
      glowSigma?: number;
      colorTint?: ColorRGB;
      zOrder?:    number;
    };
    spline?: {
      renderer:   { tPosView: any /*GPUTextureView*/; particleSlots: number };
      glowScale?: number;
      glowAlpha?: number;
      glowSigma?: number;
      colorTint?: ColorRGB;
      zOrder?:    number;
    };
    edgeFlow?: {
      renderer:   { tPosView?: any /*GPUTextureView*/; totalSlots: number };
      glowScale?: number;
      glowAlpha?: number;
      glowSigma?: number;
      colorTint?: ColorRGB;
      zOrder?:    number;
    };
    sparks?: {
      renderer:   { tPosView: any /*GPUTextureView*/; particleCount: number };
      glowScale?: number;
      glowAlpha?: number;
      glowSigma?: number;
      colorTint?: ColorRGB;
      zOrder?:    number;
      skipSort?:  boolean;
    };
  },
  config?: Partial<CompositorConfig>,
): Promise<ParticleCompositor> {
  const compositor = new ParticleCompositor(device, canvas, config);

  if (layers.flower) {
    const { renderer, glowScale, glowAlpha, glowSigma, colorTint, zOrder } = layers.flower;
    compositor.addLayer({
      id:            'flower',
      type:          LayerType.Flower,
      tPosView:      renderer.tPosView,
      texW:          256,
      texH:          128,
      particleCount: renderer.activeParticleCount,
      uSize:         0.025,
      glowScale:     glowScale ?? 2.2,
      glowAlpha:     glowAlpha ?? 0.30,
      glowSigma:     glowSigma ?? 4.0,
      colorTint:     colorTint ?? { r: 1, g: 0.78, b: 0.55 },
      zOrder:        zOrder    ?? 0,
    });
  }

  if (layers.spline) {
    const { renderer, glowScale, glowAlpha, glowSigma, colorTint, zOrder } = layers.spline;
    compositor.addLayer({
      id:            'spline',
      type:          LayerType.Spline,
      tPosView:      renderer.tPosView,
      texW:          256,
      texH:          128,
      particleCount: renderer.particleSlots,
      uSize:         0.012,
      glowScale:     glowScale ?? 1.6,
      glowAlpha:     glowAlpha ?? 0.18,
      glowSigma:     glowSigma ?? 4.0,
      colorTint:     colorTint ?? { r: 1, g: 1, b: 1 },
      zOrder:        zOrder    ?? 1,
    });
  }

  if (layers.edgeFlow) {
    const { renderer, glowScale, glowAlpha, glowSigma, colorTint, zOrder } = layers.edgeFlow;
    if (renderer.tPosView) {
      compositor.addLayer({
        id:            'edgeFlow',
        type:          LayerType.EdgeFlow,
        tPosView:      renderer.tPosView,
        texW:          256,
        texH:          128,
        particleCount: renderer.totalSlots,
        uSize:         0.015,
        glowScale:     glowScale ?? 1.8,
        glowAlpha:     glowAlpha ?? 0.22,
        glowSigma:     glowSigma ?? 3.5,
        colorTint:     colorTint ?? { r: 0.7, g: 0.85, b: 1.0 },
        zOrder:        zOrder    ?? 2,
      });
    }
  }

  if (layers.sparks) {
    const { renderer, glowScale, glowAlpha, glowSigma, colorTint, zOrder, skipSort } = layers.sparks;
    compositor.addLayer({
      id:            'sparks',
      type:          LayerType.Sparks,
      tPosView:      renderer.tPosView,
      texW:          256,
      texH:          128,
      particleCount: renderer.particleCount,
      uSize:         0.008,
      glowScale:     glowScale ?? 3.0,
      glowAlpha:     glowAlpha ?? 0.40,
      glowSigma:     glowSigma ?? 6.0,
      colorTint:     colorTint ?? { r: 1, g: 0.9, b: 0.6 },
      blendMode:     'additive',
      skipSort:      skipSort   ?? true,
      zOrder:        zOrder     ?? 3,
    });
  }

  await compositor.build();
  return compositor;
}

// ─── Defaults re-export ───────────────────────────────────────────────────────

export const COMPOSITOR_DEFAULTS = {
  maxSortParticles: MAX_SORT_PARTICLES,
  sortPairStride:   SORT_PAIR_STRIDE,
  workgroupSize:    WG,
  radixBins:        RADIX_BINS,
  radixThreshold:   RADIX_THRESHOLD,
  uniformBytes:     COMPOSITOR_UNIFORMS_BYTES,
} as const;
