/**
 * particle-compositor.ts — M718
 *
 * Unified Particle Render Compositor
 * ─────────────────────────────────────────────────────────────────────────────
 * Single WebGPU render compositor that consolidates all AT particle renderers
 * (ATFlowerParticleRenderer and ATSplineParticleLife) into one coherent draw
 * sequence with:
 *
 *   ① Depth sorting  — GPU radix sort of all active particles by depth key
 *                      (world Z, or screen-Z proxy for 2-D: travel fraction).
 *                      Ensures correct back-to-front alpha blending order.
 *
 *   ② Alpha blending — standard src-alpha / one-minus-src-alpha, drawn in
 *                      sorted order so overlapping semi-transparent particles
 *                      composite correctly.
 *
 *   ③ Additive glow  — a second render pass over the same sorted draw list,
 *                      using additive blending (src=one, dst=one) with a
 *                      Gaussian-weighted radial kernel.  Produces AT-authentic
 *                      bloom-style halo around dense particle clusters without
 *                      a full post-process bloom chain.
 *
 *   ④ Instanced draw — one indirect-draw call per layer type (flower / spline)
 *                      sourcing positions from their tPos textures.  The sort
 *                      produces a reorder-index buffer consumed by the VS via
 *                      @builtin(instance_index) → sortedIndex[ii] → tPos texel.
 *
 * Architecture
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ParticleCompositor
 *     ├─ LayerDescriptor[]        — registered tPos textures + metadata
 *     ├─ SORT pass (compute)      — bionic / bitonic sort on depth key buffer
 *     ├─ ALPHA pass (render)      — sorted instanced quads, standard blending
 *     ├─ GLOW pass (render)       — sorted instanced quads, additive blending
 *     └─ composite uniforms buf   — shared sort+render params
 *
 * Per-frame sequence (encode into one GPUCommandEncoder):
 *   compositor.sort(enc)    — rebuild sorted-index buffer
 *   compositor.renderAlpha(enc, colorView)  — standard alpha layer
 *   compositor.renderGlow(enc, colorView)   — additive glow layer
 *
 * Integration:
 * ─────────────────────────────────────────────────────────────────────────────
 * ```ts
 * import { ParticleCompositor, LayerType } from '$lib/sph/particle-compositor';
 *
 * const compositor = new ParticleCompositor(device, canvas);
 *
 * // Register particle layers (after each renderer.build())
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
 * });
 * compositor.addLayer({
 *   id:         'spline',
 *   type:       LayerType.Spline,
 *   tPosView:   splineLife.tPosView,
 *   texW:       256,
 *   texH:       128,
 *   particleCount: splineLife.particleSlots,
 *   uSize:      0.012,
 *   glowScale:  1.6,
 *   glowAlpha:  0.20,
 * });
 *
 * await compositor.build();
 *
 * // render loop:
 * const enc = device.createCommandEncoder();
 * compositor.update(enc, elapsed);   // update sort key buffer + uniforms
 * compositor.sort(enc);              // bitonic sort pass
 * compositor.renderAlpha(enc, swapchainView);
 * compositor.renderGlow(enc, swapchainView);
 * device.queue.submit([enc.finish()]);
 * ```
 *
 * Sort strategy
 * ─────────────────────────────────────────────────────────────────────────────
 * For 2-D AT scenes, "depth" is encoded as a 32-bit float key derived from the
 * tPos texture (travel fraction as a proxy for layering priority — later-travel
 * particles are drawn first so early particles appear on top).  The sort is a
 * GPU bitonic sort implemented in a compute shader operating on (key, index)
 * pairs stored in a single interleaved u32 buffer:
 *   sortBuf[2*i+0] = floatToOrderedUint(key[i])   — sort key
 *   sortBuf[2*i+1] = i                             — original slot index
 * After sort, the vertex shader reads sortBuf[2*ii+1] as the remapped instance.
 *
 * References
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/at-flower-particle.ts    — FlowerParticleShader WebGPU port
 *   src/lib/sph/at-spline-particle.ts    — SplineParticleLife WebGPU port
 *   src/lib/sph/ParticleRenderer.ts      — instanced quad helpers
 *   src/lib/sph/at-bloom-postprocess.ts  — full bloom chain (heavier alternative)
 */

// ─── Constants ────────────────────────────────────────────────────────────────

/** Workgroup size for all compute passes. Keep ≤256 for broad compatibility. */
const WG = 64 as const;

/**
 * Bitonic sort requires the array length to be a power of two.
 * We round up each layer's particleCount to the next power of two.
 * Combined sort buffer is capped at this maximum.
 */
const MAX_SORT_PARTICLES = 65536 as const;

/** Byte stride of one (key, index) pair in the sort buffer. */
const SORT_PAIR_STRIDE = 8 as const;  // 2 × u32

/** Compositor uniform buffer byte size. */
const COMPOSITOR_UNIFORMS_BYTES = 96 as const;

// ─── Public types ─────────────────────────────────────────────────────────────

/** Distinguishes particle system types so the VS reads the correct tPos layout. */
export const enum LayerType {
  /** ATFlowerParticleRenderer — matcap shading, spiral motion. */
  Flower = 0,
  /** ATSplineParticleLife     — soft-disc SDF, sin(π·travel) fade. */
  Spline = 1,
}

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
  tPosView: GPUTextureView;
  /** tPos texture width (= 256 for both AT systems). */
  texW: number;
  /** tPos texture height (= 128 for both AT systems → 32768 slots). */
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
   * Draw-order priority within the compositor.
   * Lower = drawn first (further back). Default 0.
   */
  zOrder?: number;
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
   * Whether to sort particles before drawing.
   * Disable if all particles are fully opaque or sorting is too expensive.
   * Default: true.
   */
  enableSort?: boolean;
  /**
   * Sort depth key mode.
   * 'travel' — use tPos.b (travel fraction) as depth key (default, cheap).
   * 'worldY' — use tPos.g (world Y) as depth key (good for isometric/2-D).
   */
  sortKey?: 'travel' | 'worldY';
}

// ─── Internal layout types (not exported) ─────────────────────────────────────

/** One GPU-side layer descriptor stored in the layer metadata buffer. */
interface GPULayerMeta {
  layerOffset: number;  // particle slot base in the combined sort buffer
  particleCount: number;
  sortedCount: number;  // rounded up to pow2 for bitonic sort
  texW: number;
  texH: number;
  uSize: number;
  glowScale: number;
  glowAlpha: number;
  type: LayerType;
  tPosView: GPUTextureView;
  id: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Key extraction compute pass
// Reads each tPos texture and writes (key, originalIndex) into sortBuf.
// ─────────────────────────────────────────────────────────────────────────────

const KEY_EXTRACT_WGSL = /* wgsl */`
struct CompUniforms {
  layerOffset   : u32,   // first slot index of this layer in sort buffer
  particleCount : u32,   // number of active slots in this layer
  texW          : u32,   // tPos texture width
  texH          : u32,   // tPos texture height
  sortKeyMode   : u32,   // 0 = travel, 1 = worldY
  _pad0         : u32,
  _pad1         : u32,
  _pad2         : u32,
}

@group(0) @binding(0) var<uniform>             uni     : CompUniforms;
@group(1) @binding(0) var                      tPos    : texture_2d<f32>;
@group(1) @binding(1) var<storage, read_write> sortBuf : array<u32>;

// Convert a float to an unsigned int preserving sort order:
//   positive floats: just reinterpret bits (MSB=0, maintained order)
//   negative floats: flip all bits
fn floatToKey(f: f32) -> u32 {
  let bits = bitcast<u32>(f);
  // If sign bit set → negative float: flip all bits; else flip only sign bit
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

  // Choose sort key based on mode
  let rawKey = select(p.g, p.b, uni.sortKeyMode == 0u);

  // Dead/invisible particles (alpha ≤ 0) sort to the back
  let alive  = select(0.0, rawKey, p.a > 0.004);
  let key    = floatToKey(alive);

  let globalSlot = uni.layerOffset + slot;
  // Pack (key, globalSlot) into sortBuf
  sortBuf[globalSlot * 2u + 0u] = key;
  sortBuf[globalSlot * 2u + 1u] = globalSlot;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Bitonic sort compute pass
// Single-pass bitonic merge network step.  Call once per (step, subStep) pair.
// Full sort requires log2(N) * (log2(N)+1) / 2 dispatch calls.
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

  // Both indices are valid
  let ai = i   * 2u;   // key offset for element i
  let bi = ixj * 2u;   // key offset for element ixj

  let ka = sortBuf[ai];
  let kb = sortBuf[bi];

  // Ascending sort when (i & k) == 0, descending otherwise
  // We always sort ascending (back-to-front → smallest depth key first)
  let shouldSwap = (ka > kb);

  if (shouldSwap) {
    // Swap both key and index
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
// WGSL — Alpha pass vertex shader
// Reads particle position from tPos via the sort-remapped index.
// ─────────────────────────────────────────────────────────────────────────────

const ALPHA_VERTEX_WGSL = /* wgsl */`
struct AlphaUniforms {
  domainW      : f32,
  domainH      : f32,
  scaleX       : f32,   // 2 / domainW
  scaleY       : f32,   // 2 / domainH
  uSize        : f32,   // quad half-size (domain units)
  glowScale    : f32,   // glow quad multiplier (not used in alpha VS)
  glowAlpha    : f32,   // glow peak alpha (not used in alpha VS)
  layerOffset  : u32,   // base slot in sortBuf for this layer
  particleCount: u32,
  texW         : u32,
  texH         : u32,
  layerType    : u32,   // 0=Flower, 1=Spline
}

@group(0) @binding(0) var<uniform>       uni     : AlphaUniforms;
@group(0) @binding(1) var                tPos    : texture_2d<f32>;
@group(0) @binding(2) var                sSampler: sampler;
@group(0) @binding(3) var<storage, read> sortBuf : array<u32>;

struct AlphaVertOut {
  @builtin(position) pos    : vec4f,
  @location(0)       vUv    : vec2f,
  @location(1)       vAlpha : f32,
  @location(2)       vTravel: f32,
}

var<private> QUAD: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
  vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
);

@vertex fn vs_alpha(
  @builtin(vertex_index)   vi : u32,
  @builtin(instance_index) ii : u32,
) -> AlphaVertOut {
  // Map ii → sorted global slot → layer-local slot
  let globalSlot = uni.layerOffset + ii;
  let remapped   = sortBuf[globalSlot * 2u + 1u];  // sort-order remapped index
  let localSlot  = remapped - uni.layerOffset;       // convert back to layer-local

  let texX = i32(localSlot % uni.texW);
  let texY = i32(localSlot / uni.texW);
  let p    = textureLoad(tPos, vec2<i32>(texX, texY), 0);

  let worldX = p.r;
  let worldY = p.g;
  let travel = p.b;
  let alpha  = p.a;

  // Invisible → degenerate quad
  let alive       = select(0.0, 1.0, alpha > 0.004);

  // AT vScale: uSize * (1 - travel²)
  let travelDecay = clamp(1.0 - travel * travel, 0.0, 1.0);
  let halfSize    = uni.uSize * travelDecay * 0.5;

  let qUv  = QUAD[vi];
  let ndcX = worldX * uni.scaleX - 1.0 + qUv.x * halfSize * uni.scaleX;
  let ndcY = worldY * uni.scaleY - 1.0 + qUv.y * halfSize * uni.scaleY;

  var out: AlphaVertOut;
  out.pos     = vec4f(ndcX * alive, ndcY * alive, 0.0, 1.0);
  out.vUv     = qUv;
  out.vAlpha  = alpha;
  out.vTravel = travel;
  return out;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Alpha pass fragment shader
// Soft-disc SDF circle with sin(π·travel) fade (unified AT behaviour).
// ─────────────────────────────────────────────────────────────────────────────

const ALPHA_FRAGMENT_WGSL = /* wgsl */`
struct AlphaUniforms {
  domainW      : f32,
  domainH      : f32,
  scaleX       : f32,
  scaleY       : f32,
  uSize        : f32,
  glowScale    : f32,
  glowAlpha    : f32,
  layerOffset  : u32,
  particleCount: u32,
  texW         : u32,
  texH         : u32,
  layerType    : u32,
}

@group(0) @binding(0) var<uniform> uni : AlphaUniforms;

struct AlphaFragIn {
  @location(0) vUv    : vec2f,
  @location(1) vAlpha : f32,
  @location(2) vTravel: f32,
}

@fragment fn fs_alpha(in: AlphaFragIn) -> @location(0) vec4f {
  let r2 = dot(in.vUv, in.vUv);
  if (r2 > 1.0) { discard; }

  // Soft edge falloff to avoid hard-clipped discs
  let edge   = 1.0 - smoothstep(0.64, 1.0, r2);

  // AT sin(π·travel) alpha fade
  let fade   = sin(3.14159265 * clamp(in.vTravel, 0.0, 1.0));

  // Warm core tint (AT matcap3 approximation — centre bright-white, edge amber)
  let coreTint = mix(vec3f(1.0, 0.78, 0.55), vec3f(1.0, 1.0, 1.0), 1.0 - r2);

  let finalA = in.vAlpha * fade * edge;
  return vec4f(coreTint * finalA, finalA);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Glow pass vertex shader
// Same as alpha VS but quad is scaled by glowScale.
// ─────────────────────────────────────────────────────────────────────────────

const GLOW_VERTEX_WGSL = /* wgsl */`
struct AlphaUniforms {
  domainW      : f32,
  domainH      : f32,
  scaleX       : f32,
  scaleY       : f32,
  uSize        : f32,
  glowScale    : f32,
  glowAlpha    : f32,
  layerOffset  : u32,
  particleCount: u32,
  texW         : u32,
  texH         : u32,
  layerType    : u32,
}

@group(0) @binding(0) var<uniform>       uni     : AlphaUniforms;
@group(0) @binding(1) var                tPos    : texture_2d<f32>;
@group(0) @binding(2) var                sSampler: sampler;
@group(0) @binding(3) var<storage, read> sortBuf : array<u32>;

struct GlowVertOut {
  @builtin(position) pos    : vec4f,
  @location(0)       vUv    : vec2f,
  @location(1)       vAlpha : f32,
  @location(2)       vTravel: f32,
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

  let alive       = select(0.0, 1.0, alpha > 0.004);
  let travelDecay = clamp(1.0 - travel * travel, 0.0, 1.0);
  // Glow quad is scaled up relative to the alpha quad
  let halfSize    = uni.uSize * uni.glowScale * travelDecay * 0.5;

  let qUv  = QUAD[vi];
  let ndcX = worldX * uni.scaleX - 1.0 + qUv.x * halfSize * uni.scaleX;
  let ndcY = worldY * uni.scaleY - 1.0 + qUv.y * halfSize * uni.scaleY;

  var out: GlowVertOut;
  out.pos     = vec4f(ndcX * alive, ndcY * alive, 0.0, 1.0);
  out.vUv     = qUv;
  out.vAlpha  = alpha;
  out.vTravel = travel;
  return out;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Glow pass fragment shader
// Gaussian radial kernel with additive blending for AT-style bloom halo.
// ─────────────────────────────────────────────────────────────────────────────

const GLOW_FRAGMENT_WGSL = /* wgsl */`
struct AlphaUniforms {
  domainW      : f32,
  domainH      : f32,
  scaleX       : f32,
  scaleY       : f32,
  uSize        : f32,
  glowScale    : f32,
  glowAlpha    : f32,
  layerOffset  : u32,
  particleCount: u32,
  texW         : u32,
  texH         : u32,
  layerType    : u32,
}

@group(0) @binding(0) var<uniform> uni : AlphaUniforms;

struct GlowFragIn {
  @location(0) vUv    : vec2f,
  @location(1) vAlpha : f32,
  @location(2) vTravel: f32,
}

@fragment fn fs_glow(in: GlowFragIn) -> @location(0) vec4f {
  let r2 = dot(in.vUv, in.vUv);
  if (r2 > 1.0) { discard; }

  // Gaussian kernel: exp(-k·r²) — concentrated inner glow
  let k       = 4.0;
  let gauss   = exp(-k * r2);

  // AT sin(π·travel) fade for temporal coherence
  let fade    = sin(3.14159265 * clamp(in.vTravel, 0.0, 1.0));

  // Warm halo tint (slightly shifted toward yellow-gold for bloom feel)
  let glowTint = vec3f(1.0, 0.92, 0.70);

  let finalA  = uni.glowAlpha * in.vAlpha * fade * gauss;
  return vec4f(glowTint * finalA, finalA);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// CPU helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Round x up to the nearest power of two. */
function nextPow2(x: number): number {
  let n = 1;
  while (n < x) n <<= 1;
  return n;
}

/** Build the per-layer uniform data for alpha / glow passes. */
function buildLayerUniforms(
  meta:    GPULayerMeta,
  domainW: number,
  domainH: number,
): Float32Array {
  // 12 slots — 8 f32 + 4 u32 aliased in the same Float32Array
  const data = new Float32Array(12);
  data[0]  = domainW;
  data[1]  = domainH;
  data[2]  = 2.0 / domainW;
  data[3]  = 2.0 / domainH;
  data[4]  = meta.uSize;
  data[5]  = meta.glowScale;
  data[6]  = meta.glowAlpha;
  const u32 = new Uint32Array(data.buffer);
  u32[7]   = meta.layerOffset;
  u32[8]   = meta.particleCount;
  u32[9]   = meta.texW;
  u32[10]  = meta.texH;
  u32[11]  = meta.type;
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
 *   compositor.update(enc, elapsed)       — refresh uniforms
 *   compositor.sort(enc)                  — GPU bitonic sort
 *   compositor.renderAlpha(enc, view)     — alpha-blended layer
 *   compositor.renderGlow(enc, view)      — additive glow layer
 */
export class ParticleCompositor {
  private readonly device: GPUDevice;
  private readonly canvas: HTMLCanvasElement;

  private layers: LayerDescriptor[]   = [];
  private gpuMeta: GPULayerMeta[]     = [];

  private cfg: Required<CompositorConfig> = {
    clearColor:  { r: 0, g: 0, b: 0, a: 0 },
    enableGlow:  true,
    enableSort:  true,
    sortKey:     'travel',
  };

  // ── Combined sort buffer ─────────────────────────────────────────────────────
  /** Interleaved (key u32, index u32) × totalSortSlots. */
  private sortBuf!:      GPUBuffer;
  private totalSlots  = 0;
  private sortedCount = 0;   // rounded to pow2

  // ── Key extraction pipeline ──────────────────────────────────────────────────
  private keyPipeline!:  GPUComputePipeline;

  /** Per-layer uniform buffers for key extraction. */
  private keyUniBufs:    GPUBuffer[]   = [];
  /** Per-layer bind groups (BG0: keyUni, BG1: tPos+sortBuf). */
  private keyBGs:        GPUBindGroup[] = [];

  // ── Bitonic sort pipeline ────────────────────────────────────────────────────
  private sortPipeline!: GPUComputePipeline;
  private sortUniBuf!:   GPUBuffer;
  private sortBG!:       GPUBindGroup;

  // ── Alpha / Glow render pipelines ────────────────────────────────────────────
  private alphaPipeline!: GPURenderPipeline;
  private glowPipeline!:  GPURenderPipeline;
  private sampler!:       GPUSampler;

  /** Per-layer render bind groups for alpha pass. */
  private alphaBGs:       GPUBindGroup[] = [];
  /** Per-layer render bind groups for glow pass. */
  private glowBGs:        GPUBindGroup[] = [];
  /** Per-layer uniform buffers (shared between alpha and glow BGs). */
  private renderUniBufs:  GPUBuffer[]    = [];

  private built = false;

  constructor(
    device: GPUDevice,
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
    // Trigger rebuild if already built
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
        glowScale:     l.glowScale ?? 2.0,
        glowAlpha:     l.glowAlpha ?? 0.25,
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

    // ── Sort buffer ─────────────────────────────────────────────────────────────
    this.sortBuf = device.createBuffer({
      size:  this.sortedCount * SORT_PAIR_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    // Initialise all keys to 0xFFFFFFFF (dead particles sort to back)
    new Uint32Array(this.sortBuf.getMappedRange()).fill(0xFFFFFFFF);
    this.sortBuf.unmap();

    // ── Sampler (nearest — tPos is a data texture) ─────────────────────────────
    this.sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });

    // ── Key extraction pipeline ─────────────────────────────────────────────────
    const keyMod = device.createShaderModule({ code: KEY_EXTRACT_WGSL });
    this.keyPipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: keyMod, entryPoint: 'main' },
    });

    // Per-layer key-extraction resources
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
          { binding: 1, resource: { buffer: this.sortBuf } },
        ],
      });
      this.keyUniBufs.push(keyUni);
      this.keyBGs.push(bg0, bg1);  // store alternating BG0, BG1
    }

    // ── Bitonic sort pipeline ───────────────────────────────────────────────────
    const sortMod = device.createShaderModule({ code: BITONIC_SORT_WGSL });
    this.sortPipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: sortMod, entryPoint: 'main' },
    });
    this.sortUniBuf = device.createBuffer({
      size:  16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.sortBG = device.createBindGroup({
      layout: this.sortPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sortUniBuf } },
        { binding: 1, resource: { buffer: this.sortBuf } },
      ],
    });

    // ── Alpha render pipeline ──────────────────────────────────────────────────
    const alphaVsMod = device.createShaderModule({ code: ALPHA_VERTEX_WGSL });
    const alphaFsMod = device.createShaderModule({ code: ALPHA_FRAGMENT_WGSL });
    this.alphaPipeline = device.createRenderPipeline({
      layout:   'auto',
      vertex:   { module: alphaVsMod, entryPoint: 'vs_alpha' },
      fragment: {
        module: alphaFsMod, entryPoint: 'fs_alpha',
        targets: [{
          format: fmt,
          blend: {
            // Standard premultiplied alpha blending
            color: {
              srcFactor: 'src-alpha',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // ── Glow render pipeline ───────────────────────────────────────────────────
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
            // Additive blending — bright clusters self-accumulate
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

    for (const meta of this.gpuMeta) {
      const unifData = buildLayerUniforms(meta, this.canvas.width || 1, this.canvas.height || 1);
      const uniBuf   = device.createBuffer({
        size:  unifData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(uniBuf, 0, unifData);
      this.renderUniBufs.push(uniBuf);

      const alphaEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: uniBuf } },
        { binding: 1, resource: meta.tPosView },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: { buffer: this.sortBuf } },
      ];
      this.alphaBGs.push(device.createBindGroup({
        layout: this.alphaPipeline.getBindGroupLayout(0),
        entries: alphaEntries,
      }));
      this.glowBGs.push(device.createBindGroup({
        layout: this.glowPipeline.getBindGroupLayout(0),
        entries: alphaEntries,   // same layout, different pipeline
      }));
    }

    this.built = true;
    console.log(
      `[ParticleCompositor] built: ${this.gpuMeta.length} layers, ` +
      `${this.totalSlots} total slots, sort buffer = ${this.sortedCount} × 8 B`,
    );
  }

  // ── Per-frame API ─────────────────────────────────────────────────────────────

  /**
   * Update compositor uniform buffers (domain size may change on resize).
   * Also resets the sort buffer keys to 0xFFFFFFFF so dead-particle sentinels
   * remain valid even if a layer shrank since last build.
   *
   * @param _enc      — GPUCommandEncoder (not currently used; reserved)
   * @param _elapsed  — total elapsed seconds (reserved for future time-based effects)
   */
  update(_enc: GPUCommandEncoder, _elapsed: number): void {
    if (!this.built) return;
    const { device, canvas, gpuMeta, renderUniBufs } = this;
    const dw = canvas.width  || 1;
    const dh = canvas.height || 1;

    for (let i = 0; i < gpuMeta.length; i++) {
      const unifData = buildLayerUniforms(gpuMeta[i], dw, dh);
      device.queue.writeBuffer(renderUniBufs[i], 0, unifData);
    }
  }

  /**
   * Encode the depth sort sequence (key extract + bitonic sort) into enc.
   * Must be called before renderAlpha / renderGlow.
   */
  sort(enc: GPUCommandEncoder): void {
    if (!this.built || !this.cfg.enableSort) return;

    // ── Phase 1: extract keys from each layer's tPos ───────────────────────────
    const keyPass = enc.beginComputePass({ label: 'compositor:keyExtract' });
    keyPass.setPipeline(this.keyPipeline);

    for (let i = 0; i < this.gpuMeta.length; i++) {
      const meta = this.gpuMeta[i];
      const wgs  = Math.ceil(meta.particleCount / WG);
      keyPass.setBindGroup(0, this.keyBGs[i * 2]);      // BG0: keyUni
      keyPass.setBindGroup(1, this.keyBGs[i * 2 + 1]);  // BG1: tPos + sortBuf
      keyPass.dispatchWorkgroups(wgs);
    }
    keyPass.end();

    // ── Phase 2: bitonic sort ──────────────────────────────────────────────────
    if (this.sortedCount <= 1) return;

    const sortPass = enc.beginComputePass({ label: 'compositor:bitonicSort' });
    sortPass.setPipeline(this.sortPipeline);
    sortPass.setBindGroup(0, this.sortBG);

    const n    = this.sortedCount;
    const wgs  = Math.ceil(n / 2 / WG);

    // Full bitonic sort: log2(n) major steps, each with j sub-steps
    for (let k = 2; k <= n; k <<= 1) {
      for (let j = k >> 1; j >= 1; j >>= 1) {
        const params = new Uint32Array([k, j, n, 0]);
        this.device.queue.writeBuffer(this.sortUniBuf, 0, params);
        sortPass.dispatchWorkgroups(wgs);
      }
    }
    sortPass.end();
  }

  /**
   * Encode the alpha-blended render pass for all layers.
   *
   * @param enc         GPUCommandEncoder
   * @param colorView   Swapchain / render target texture view
   * @param loadOp      'clear' (default, first pass) or 'load' (compositing on top)
   */
  renderAlpha(
    enc:       GPUCommandEncoder,
    colorView: GPUTextureView,
    loadOp:    GPULoadOp = 'clear',
  ): void {
    if (!this.built || this.gpuMeta.length === 0) return;

    const pass = enc.beginRenderPass({
      label:                'compositor:alpha',
      colorAttachments: [{
        view:       colorView,
        loadOp,
        storeOp:    'store',
        clearValue: this.cfg.clearColor,
      }],
    });
    pass.setPipeline(this.alphaPipeline);

    for (let i = 0; i < this.gpuMeta.length; i++) {
      const meta = this.gpuMeta[i];
      pass.setBindGroup(0, this.alphaBGs[i]);
      // 6 vertices per quad (2 triangles), particleCount instances
      pass.draw(6, meta.particleCount, 0, 0);
    }
    pass.end();
  }

  /**
   * Encode the additive glow render pass for all layers.
   * Must be called after renderAlpha (uses 'load' to preserve alpha-pass output).
   *
   * @param enc         GPUCommandEncoder
   * @param colorView   Same swapchain / render target as renderAlpha
   */
  renderGlow(enc: GPUCommandEncoder, colorView: GPUTextureView): void {
    if (!this.built || !this.cfg.enableGlow || this.gpuMeta.length === 0) return;

    const pass = enc.beginRenderPass({
      label:            'compositor:glow',
      colorAttachments: [{
        view:    colorView,
        loadOp:  'load',   // composite on top of alpha pass
        storeOp: 'store',
      }],
    });
    pass.setPipeline(this.glowPipeline);

    for (let i = 0; i < this.gpuMeta.length; i++) {
      const meta = this.gpuMeta[i];
      pass.setBindGroup(0, this.glowBGs[i]);
      pass.draw(6, meta.particleCount, 0, 0);
    }
    pass.end();
  }

  /**
   * Convenience method: sort + renderAlpha + renderGlow in one call.
   * For callers that want a single-frame composite without manual sequencing.
   *
   * @param enc       open GPUCommandEncoder
   * @param view      swapchain texture view
   * @param elapsed   total elapsed seconds
   * @param loadOp    clear (default) or load
   */
  composite(
    enc:     GPUCommandEncoder,
    view:    GPUTextureView,
    elapsed: number,
    loadOp:  GPULoadOp = 'clear',
  ): void {
    this.update(enc, elapsed);
    this.sort(enc);
    this.renderAlpha(enc, view, loadOp);
    this.renderGlow(enc, view);
  }

  // ── Accessors ─────────────────────────────────────────────────────────────────

  get layerCount():  number  { return this.gpuMeta.length; }
  get totalParticleSlots(): number { return this.totalSlots; }
  get isBuilt():     boolean { return this.built; }

  // ── Live config updates ───────────────────────────────────────────────────────

  /**
   * Toggle the glow pass at runtime without rebuilding pipelines.
   * Useful for performance headroom management.
   */
  setGlowEnabled(enabled: boolean): void {
    this.cfg.enableGlow = enabled;
  }

  /**
   * Toggle depth sorting at runtime.
   * Disabling avoids the compute sort overhead (saves ~0.2 ms for 32 k particles).
   */
  setSortEnabled(enabled: boolean): void {
    this.cfg.enableSort = enabled;
  }

  /**
   * Update glowAlpha for a specific layer by id at runtime.
   * The uniform buffer is updated immediately; no rebuild required.
   */
  setLayerGlowAlpha(id: string, alpha: number): void {
    const idx = this.gpuMeta.findIndex(m => m.id === id);
    if (idx < 0) return;
    this.gpuMeta[idx].glowAlpha = alpha;
    // writeBuffer on next update() call via buildLayerUniforms
  }

  /**
   * Update uSize for a specific layer by id at runtime.
   * Takes effect on the next update() call.
   */
  setLayerSize(id: string, uSize: number): void {
    const idx = this.gpuMeta.findIndex(m => m.id === id);
    if (idx < 0) return;
    this.gpuMeta[idx].uSize = uSize;
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────────

  destroy(): void { this._destroy(); }

  // ── Private helpers ───────────────────────────────────────────────────────────

  private _writeKeyUniforms(buf: GPUBuffer, meta: GPULayerMeta): void {
    const data = new Uint32Array(8);
    data[0] = meta.layerOffset;
    data[1] = meta.particleCount;
    data[2] = meta.texW;
    data[3] = meta.texH;
    data[4] = this.cfg.sortKey === 'travel' ? 0 : 1;
    this.device.queue.writeBuffer(buf, 0, data);
  }

  private _destroy(): void {
    if (!this.built) return;
    this.sortBuf?.destroy();
    this.sortUniBuf?.destroy();
    for (const b of this.keyUniBufs) b.destroy();
    for (const b of this.renderUniBufs) b.destroy();
    this.keyUniBufs  = [];
    this.keyBGs      = [];
    this.alphaBGs    = [];
    this.glowBGs     = [];
    this.renderUniBufs = [];
    this.gpuMeta     = [];
    this.built       = false;
  }
}

// ─── Factory helpers ──────────────────────────────────────────────────────────

/**
 * createCompositorForATRenderers
 *
 * Convenience factory: build a ParticleCompositor pre-wired for one
 * ATFlowerParticleRenderer and one ATSplineParticleLife.  Both renderers
 * must be fully built (`.build()` awaited) before calling this.
 *
 * @example
 * ```ts
 * import { createCompositorForATRenderers } from '$lib/sph/particle-compositor';
 *
 * const compositor = await createCompositorForATRenderers(device, canvas, {
 *   flower: { renderer: flowerRenderer, glowScale: 2.2, glowAlpha: 0.30 },
 *   spline: { renderer: splineLife,     glowScale: 1.6, glowAlpha: 0.18 },
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
      renderer: {
        tPosView:           GPUTextureView;
        activeParticleCount: number;
      };
      glowScale?: number;
      glowAlpha?: number;
      zOrder?:   number;
    };
    spline?: {
      renderer: {
        tPosView:      GPUTextureView;
        particleSlots: number;
      };
      glowScale?: number;
      glowAlpha?: number;
      zOrder?:   number;
    };
  },
  config?: Partial<CompositorConfig>,
): Promise<ParticleCompositor> {
  const compositor = new ParticleCompositor(device, canvas, config);

  if (layers.flower) {
    const { renderer, glowScale, glowAlpha, zOrder } = layers.flower;
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
      zOrder:        zOrder    ?? 0,
    });
  }

  if (layers.spline) {
    const { renderer, glowScale, glowAlpha, zOrder } = layers.spline;
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
      zOrder:        zOrder    ?? 1,
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
} as const;
