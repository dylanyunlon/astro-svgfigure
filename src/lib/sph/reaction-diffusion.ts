/**
 * reaction-diffusion.ts
 *
 * Gray-Scott Reaction-Diffusion — WebGPU compute pipeline
 *
 * M743: Per-species (f,k) parameters
 * ─────────────────────────────────────────────────────────────────────────────
 * Each texel in the simulation grid can belong to a different species, with
 * its own (f, k, Du, Dv) parameter set.  This enables heterogeneous
 * reaction-diffusion where different cell types produce distinct morphologies
 * (coral, mitosis, worms, etc.) side by side on the same grid.
 *
 * 架构设计：
 *   • 双缓冲 ping-pong：两张 rgba32float storage texture
 *       ch R = u (activator,  chemical U)
 *       ch G = v (inhibitor,  chemical V)
 *       ch B = species index (float-encoded, preserved across steps)
 *       ch A = reserved (1)
 *   • species map texture (r32uint)：每个 texel 存储 species index (0–MAX_SPECIES)
 *   • parameter LUT buffer：storage buffer, MAX_SPECIES 条 (f, k, Du, Dv)
 *   • 每次 step() 在 GPU 上执行 N 次 GS compute pass（默认 8 次/帧）
 *   • 最终输出纹理可直接绑定到渲染 pipeline
 *   • parameterSpace(name) 按 Munafo/Pearson/Karl Sims 参数映射返回 (f, k)
 *   • speciesGrayScottMap() 将 cil-* species 映射到各自的 GS pattern
 *
 * Gray-Scott 方程（每步 Δt=1.0）：
 *   du/dt = Du·∇²u − u·v² + f·(1−u)
 *   dv/dt = Dv·∇²v + u·v² − (f+k)·v
 *
 * 扩散系数（规范值，源自 Pearson 1993）：
 *   Du = 0.2097,  Dv = 0.1050   (Du/Dv ≈ 2 : 1)
 *
 * 参考：
 *   Pearson, J.E. (1993) Complex Patterns in a Simple System. Science 261.
 *   Munafo, R. — mrob.com/pub/comp/xmorphia  (Pearson extended classes)
 *   Karl Sims — karlsims.com/rd.html  (coral f=0.0545 k=0.062; mitosis f=0.0367 k=0.0649)
 *   Shader: grayscott-species.frag — M550, cell-pubsub-loop branch
 */

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Storage texture format for ping-pong buffers (R=u, G=v, B=speciesIdx, A=1). */








const RD_TEX_FORMAT: GPUTextureFormat = 'rgba32float';

/** Species map texture format — single-channel unsigned integer per texel. */
const SPECIES_TEX_FORMAT: GPUTextureFormat = 'r32uint';

/** Compute workgroup tile size (8×8 = 64 threads per workgroup). */
const RD_WG = 8;

/** Default simulation grid resolution. */
export const RD_DEFAULT_SIZE = 256;

/** Default number of Gray-Scott substeps per animation frame. */
export const RD_DEFAULT_SUBSTEPS = 8;

/**
 * Maximum number of distinct species in the parameter LUT.
 *
 * Index 0 = fluid background (uses fallback uniform params).
 * Indices 1–10 = cil-eye … cil-graph (matching cell-body-bridge SPECIES_ORDER).
 * Indices 11–15 = reserved for future species.
 */
export const RD_MAX_SPECIES = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Named pattern species from the Pearson/Munafo Gray-Scott parameter space.
 *
 * Values verified against:
 *   • Karl Sims — karlsims.com/rd.html
 *   • Munafo — mrob.com/pub/comp/xmorphia/pearson-classes.html
 */
export type GrayScottSpecies =
  | 'coral'
  | 'mitosis'
  | 'dots'
  | 'stripes'
  | 'labyrinth'
  | 'worms'
  | 'spirals'
  | 'bubbles'
  | 'maze'
  | 'chaos';

/** Feed/kill parameter pair for the Gray-Scott model. */
export interface GrayScottParams {
  /** Feed rate f (controls u replenishment). */
  f: number;
  /** Kill rate k (controls v removal). */
  k: number;
  /** Diffusion rate for u (activator). Default 0.2097. */
  Du?: number;
  /** Diffusion rate for v (inhibitor). Default 0.1050. */
  Dv?: number;
}

/** Configuration for ReactionDiffusionSim constructor. */
export interface RDSimConfig {
  /** Grid width in cells. Default: RD_DEFAULT_SIZE. */
  width?: number;
  /** Grid height in cells. Default: RD_DEFAULT_SIZE. */
  height?: number;
  /** Gray-Scott (f, k, Du, Dv) fallback parameters. Default: coral preset. */
  params?: GrayScottParams;
  /** GS substeps executed per step() call. Default: RD_DEFAULT_SUBSTEPS. */
  substeps?: number;
  /**
   * Enable per-species heterogeneous (f,k) parameters.
   * When true, the compute shader reads species indices from a species map
   * texture and looks up per-species (f,k,Du,Dv) from a storage buffer LUT.
   * Default: false (single-species uniform mode for backward compatibility).
   */
  perSpecies?: boolean;
}

/**
 * Per-species Gray-Scott parameter entry for the LUT buffer.
 *
 * Maps a cell-body-bridge species index (0=fluid, 1=cil-eye, …, 10=cil-graph)
 * to its canonical Gray-Scott (f, k, Du, Dv) parameter set.
 */
export interface SpeciesParamEntry {
  /** Species index (0–RD_MAX_SPECIES-1). Matches cell-body-bridge SPECIES_ORDER. */
  index: number;
  /** Gray-Scott parameters for this species. */
  params: Required<GrayScottParams>;
}

/**
 * Rectangular region on the grid to assign a species index.
 * Used with `setSpeciesRegion()` to paint species zones.
 */
export interface SpeciesRegion {
  /** Species index to paint into this region. */
  speciesIndex: number;
  /** Left column (inclusive). */
  x: number;
  /** Top row (inclusive). */
  y: number;
  /** Region width in texels. */
  w: number;
  /** Region height in texels. */
  h: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameter space — Munafo / Pearson / Karl Sims canonical values
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Map a named species to its canonical Gray-Scott (f, k) parameters.
 *
 * Parameter sources:
 *   coral     — f=0.0545, k=0.0620  Karl Sims "coral growth" demo
 *   mitosis   — f=0.0367, k=0.0649  Karl Sims "mitosis" demo
 *   dots      — f=0.0350, k=0.0650  Pearson type λ (soliton dots)
 *   stripes   — f=0.0600, k=0.0630  Pearson type κ (labyrinthine)
 *   labyrinth — f=0.0300, k=0.0570  Pearson type δ (Turing stripes)
 *   worms     — f=0.0780, k=0.0610  Pearson type μ (worm/stripe mix)
 *   spirals   — f=0.0100, k=0.0350  Munafo type α (spiral chaos)
 *   bubbles   — f=0.0900, k=0.0590  xmorphia "soap-bubbles"
 *   maze      — f=0.0220, k=0.0510  Pearson type γ (worm maze)
 *   chaos     — f=0.0260, k=0.0510  Pearson type β/γ boundary
 */
export function parameterSpace(species: GrayScottSpecies): GrayScottParams {
  // Canonical Du/Dv from Pearson 1993 (rescaled to ≤1 range for numerical stability)
  const Du = 0.2097;
  const Dv = 0.1050;

  switch (species) {
    // ── Pearson κ — coral / branching loops growing from worm tips
    //    Karl Sims "coral growth" reference: f=0.0545, k=0.062
    case 'coral':
      return { f: 0.0545, k: 0.0620, Du, Dv };

    // ── Pearson λ — mitosis / solitons that grow then divide
    //    Karl Sims "mitosis" reference: f=0.0367, k=0.0649
    case 'mitosis':
      return { f: 0.0367, k: 0.0649, Du, Dv };

    // ── Pearson λ edge — isolated soliton spots, hexagonal packing
    //    xmorphia F350/k650; "dots" in Karl Sims parameter map
    case 'dots':
      return { f: 0.0350, k: 0.0650, Du, Dv };

    // ── Pearson κ — labyrinthine stripe maze (fingerprint / hedgerow)
    //    xmorphia F600/k630 (coral→maze evolution after 250 000 tu)
    case 'stripes':
      return { f: 0.0600, k: 0.0630, Du, Dv };

    // ── Pearson δ — Turing instability: stationary negative-spot hexarray
    //    xmorphia F300/k550
    case 'labyrinth':
      return { f: 0.0300, k: 0.0570, Du, Dv };

    // ── Pearson μ — worms/filaments that elongate without branching
    //    xmorphia high-F stripe zone F780/k610
    case 'worms':
      return { f: 0.0780, k: 0.0610, Du, Dv };

    // ── Munafo type α — spatiotemporal chaos, spirals / wavelets
    //    xmorphia F100/k470
    case 'spirals':
      return { f: 0.0100, k: 0.0350, Du, Dv };

    // ── Munafo high-F "soap bubbles" — negatons in red sea
    //    xmorphia F900/k590 "soap-bubbles" (Karl Sims F=0.090, k=0.059)
    case 'bubbles':
      return { f: 0.0900, k: 0.0590, Du, Dv };

    // ── Pearson γ — worm maze with endless grain-boundary instability
    //    xmorphia F220/k510
    case 'maze':
      return { f: 0.0220, k: 0.0510, Du, Dv };

    // ── Pearson β/γ boundary — localised spatiotemporal chaos
    //    xmorphia F260/k510
    case 'chaos':
      return { f: 0.0260, k: 0.0510, Du, Dv };

    default: {
      // Exhaustiveness guard — TypeScript will error if a case is missing
      const _exhaustive: never = species;
      return { f: 0.0545, k: 0.0620, Du, Dv };
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Species → Gray-Scott mapping
// ─────────────────────────────────────────────────────────────────────────────
//
// Maps each cil-* cell species to the Gray-Scott morphology that best
// represents its computational role in the Transformer architecture.
//
// Design rationale (species → pattern → role analogy):
//
//   cil-eye         → coral     Attention heads: branching, self-similar expansion
//                                from query/key dot products — coral growth echoes
//                                the recursive fan-out of multi-head attention.
//
//   cil-bolt        → spirals   FFN/MLP: rotating activation spirals mirror the
//                                nonlinear mixing of GELU/SiLU activation functions
//                                in feed-forward layers.
//
//   cil-vector      → stripes   Embedding: parallel stripe channels represent the
//                                orderly positional encoding frequencies (sin/cos
//                                bands) laid across the embedding dimension.
//
//   cil-plus        → dots      Add & LayerNorm: stable soliton dots represent
//                                the residual connection's role as a fixed-point
//                                stabiliser in the skip-connection stream.
//
//   cil-arrow-right → worms     Skip/Flow: elongating worms without branching
//                                visualise the directed, non-bifurcating information
//                                highway of residual skip connections.
//
//   cil-filter      → maze      Selection gate: connected maze networks represent
//                                the complex masking topology of attention masks
//                                and gating mechanisms.
//
//   cil-code        → mitosis   Output/Token: solitons that grow and divide mirror
//                                token prediction — logit distributions splitting
//                                into discrete output symbols.
//
//   cil-layers      → bubbles   Layer stack: large smooth blobs with halos represent
//                                the hierarchical depth of stacked Transformer layers,
//                                each encapsulating the one below.
//
//   cil-loop        → chaos     Loop/Ctrl: spatiotemporal chaos at the β/γ boundary
//                                represents the feedback oscillation of recurrent
//                                control loops (adaptive compute, early exit).
//
//   cil-graph       → labyrinth Topology: Turing instability hexagonal arrays mirror
//                                the regular lattice of graph-structured attention
//                                topologies (sparse / local / linear attention).
//
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Canonical mapping from cil-* cell species to Gray-Scott pattern species.
 *
 * Keys match cell-body-bridge SPECIES_ORDER (index 0 = 'fluid').
 * Each cell species gets a distinct Gray-Scott morphology tuned to its
 * Transformer architectural role.
 */
export const SPECIES_GRAYSCOTT_MAP: Readonly<Record<string, GrayScottSpecies>> = {
  'fluid':           'coral',       // background fluid — default coral
  'cil-eye':         'coral',       // attention — branching coral growth
  'cil-bolt':        'spirals',     // FFN — rotating spiral waves
  'cil-vector':      'stripes',     // embedding — parallel stripe channels
  'cil-plus':        'dots',        // add & norm — stable soliton dots
  'cil-arrow-right': 'worms',       // skip connection — elongating worms
  'cil-filter':      'maze',        // selection gate — connected maze
  'cil-code':        'mitosis',     // output — dividing solitons
  'cil-layers':      'bubbles',     // layer stack — encapsulating blobs
  'cil-loop':        'chaos',       // loop control — spatiotemporal chaos
  'cil-graph':       'labyrinth',   // topology — hexagonal Turing lattice
};

/**
 * Resolve the full (f, k, Du, Dv) parameters for a cil-* species string.
 *
 * Looks up the species in SPECIES_GRAYSCOTT_MAP, then returns the canonical
 * Gray-Scott parameters from the Pearson/Munafo parameter space.
 *
 * @param species  Cell species string (e.g. 'cil-eye'). Unknown species
 *                 fall back to 'coral'.
 * @returns        Full { f, k, Du, Dv } parameter set.
 */
export function speciesGrayScottParams(species: string): Required<GrayScottParams> {
  const gsSpecies = SPECIES_GRAYSCOTT_MAP[species] ?? 'coral';
  const p = parameterSpace(gsSpecies);
  return { f: p.f, k: p.k, Du: p.Du ?? 0.2097, Dv: p.Dv ?? 0.1050 };
}

/**
 * Build the default per-species parameter LUT matching cell-body-bridge
 * SPECIES_ORDER indices.
 *
 * Returns an array of RD_MAX_SPECIES entries, each with { f, k, Du, Dv }.
 * Indices 0–10 are populated from SPECIES_GRAYSCOTT_MAP; indices 11–15 are
 * filled with the coral fallback.
 */
export function buildDefaultSpeciesLUT(): Required<GrayScottParams>[] {
  const SPECIES_ORDER = [
    'fluid',           // 0
    'cil-eye',         // 1
    'cil-bolt',        // 2
    'cil-vector',      // 3
    'cil-plus',        // 4
    'cil-arrow-right', // 5
    'cil-filter',      // 6
    'cil-layers',      // 7
    'cil-loop',        // 8
    'cil-code',        // 9
    'cil-graph',       // 10
  ];

  const fallback = speciesGrayScottParams('fluid');
  const lut: Required<GrayScottParams>[] = [];

  for (let i = 0; i < RD_MAX_SPECIES; i++) {
    if (i < SPECIES_ORDER.length) {
      lut.push(speciesGrayScottParams(SPECIES_ORDER[i]));
    } else {
      lut.push({ ...fallback });
    }
  }

  return lut;
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Gray-Scott compute shader (per-species variant)
// ─────────────────────────────────────────────────────────────────────────────
//
// Extends the original single-species shader with:
//   • @group(2) @binding(0)  speciesMap  : texture_2d<u32>   — per-texel species index
//   • @group(2) @binding(1)  speciesLUT  : storage buffer    — (f,k,Du,Dv) × MAX_SPECIES
//
// The compute step reads the species index for each texel, fetches the
// corresponding (f,k,Du,Dv) from the LUT, and applies per-texel GS reaction.
//
// The species index is stored in the B channel of the ping-pong textures
// (as a float-encoded integer) so it survives across substeps without needing
// to re-read the species map every iteration.
//
// ─────────────────────────────────────────────────────────────────────────────

const GS_COMPUTE_SHADER_PER_SPECIES = /* wgsl */`

// ── Constants ───────────────────────────────────────────────────────────────
const MAX_SPECIES: u32 = ${RD_MAX_SPECIES}u;

// ── Uniforms ────────────────────────────────────────────────────────────────
struct GSUniforms {
  f      : f32,   // fallback feed rate (used when species index = 0)
  k      : f32,   // fallback kill rate
  Du     : f32,   // fallback diffusion rate, activator u
  Dv     : f32,   // fallback diffusion rate, inhibitor v
  width  : u32,
  height : u32,
  _pad0  : u32,
  _pad1  : u32,
}
@group(0) @binding(0) var<uniform> u_gs : GSUniforms;

// ── Ping-pong storage textures ───────────────────────────────────────────────
@group(1) @binding(0) var readTex  : texture_2d<f32>;
@group(1) @binding(1) var writeTex : texture_storage_2d<rgba32float, write>;

// ── Per-species data ─────────────────────────────────────────────────────────
// Species map: R channel = species index (0–MAX_SPECIES-1) as u32.
@group(2) @binding(0) var speciesMap : texture_2d<u32>;

// Species parameter LUT: array of vec4f, each = (f, k, Du, Dv).
// Indexed by species index.  Padded to MAX_SPECIES entries.
struct SpeciesLUT {
  entries : array<vec4f, ${RD_MAX_SPECIES}>,
}
@group(2) @binding(1) var<storage, read> speciesLUT : SpeciesLUT;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn loadUV(coord: vec2i) -> vec3f {
  // Clamp-to-edge boundary condition (Neumann: zero-flux)
  let c = clamp(coord, vec2i(0, 0),
                vec2i(i32(u_gs.width) - 1, i32(u_gs.height) - 1));
  let s = textureLoad(readTex, c, 0);
  return s.rgb;   // R=u, G=v, B=speciesIdx(float)
}

fn lookupParams(speciesIdx: u32) -> vec4f {
  // Clamp to valid range to prevent out-of-bounds
  let idx = min(speciesIdx, MAX_SPECIES - 1u);
  return speciesLUT.entries[idx];
}

// ── Gray-Scott compute step (per-species) ────────────────────────────────────
@compute @workgroup_size(${RD_WG}, ${RD_WG})
fn gs_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let st = vec2i(i32(gid.x), i32(gid.y));

  // Bounds check
  if (gid.x >= u_gs.width || gid.y >= u_gs.height) { return; }

  // ── Read species index from the species map texture ───────────────────────
  let speciesIdx = textureLoad(speciesMap, st, 0).r;

  // ── Look up per-species (f, k, Du, Dv) from the LUT ──────────────────────
  let sp = lookupParams(speciesIdx);
  let sp_f  = sp.x;   // feed rate
  let sp_k  = sp.y;   // kill rate
  let sp_Du = sp.z;   // diffusion rate u
  let sp_Dv = sp.w;   // diffusion rate v

  // ── Sample 3×3 neighbourhood ──────────────────────────────────────────────
  // Pearson isotropic Laplacian kernel:
  //   [ 0.05  0.20  0.05 ]
  //   [ 0.20 -1.00  0.20 ]
  //   [ 0.05  0.20  0.05 ]
  let c  = loadUV(st);
  let n  = loadUV(st + vec2i( 0,  1));
  let s  = loadUV(st + vec2i( 0, -1));
  let e  = loadUV(st + vec2i( 1,  0));
  let w  = loadUV(st + vec2i(-1,  0));
  let ne = loadUV(st + vec2i( 1,  1));
  let nw = loadUV(st + vec2i(-1,  1));
  let se = loadUV(st + vec2i( 1, -1));
  let sw = loadUV(st + vec2i(-1, -1));

  // ── Discrete Laplacian (only on u,v channels) ────────────────────────────
  let lap = 0.20 * (n.xy + s.xy + e.xy + w.xy)
          + 0.05 * (ne.xy + nw.xy + se.xy + sw.xy)
          - 1.00 * c.xy;

  // ── Gray-Scott reaction with per-species parameters ───────────────────────
  let uu   = c.x;
  let vv   = c.y;
  let uvv  = uu * vv * vv;

  // du/dt = Du·∇²u − u·v² + f·(1−u)
  let du   = sp_Du * lap.x - uvv + sp_f * (1.0 - uu);
  // dv/dt = Dv·∇²v + u·v² − (f+k)·v
  let dv   = sp_Dv * lap.y + uvv - (sp_f + sp_k) * vv;

  // ── Euler integration (Δt = 1.0) ─────────────────────────────────────────
  let newU = clamp(uu + du, 0.0, 1.0);
  let newV = clamp(vv + dv, 0.0, 1.0);

  // B channel: preserve species index as float for downstream render reads
  textureStore(writeTex, st, vec4f(newU, newV, f32(speciesIdx), 1.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Gray-Scott compute shader (original single-species)
// ─────────────────────────────────────────────────────────────────────────────
//
// Kept for backward compatibility when perSpecies=false.
// Uses two rgba32float storage textures for ping-pong double-buffering.
//   readTex  — current chemical concentrations (read-only)
//   writeTex — next chemical concentrations (write-only)
//
// Laplacian kernel: Pearson 3×3 isotropic weights
//   edge neighbours (N/S/E/W): weight  0.20
//   corner neighbours:          weight  0.05
//   centre:                     weight −1.00
//   Σ = 4×0.20 + 4×0.05 − 1.00 = 0.00  (conservation)
//
// Gray-Scott step (Euler, Δt=1.0):
//   uvv  = u * v * v
//   u′  = u + Du·∇²u − uvv + f·(1−u)
//   v′  = v + Dv·∇²v + uvv − (f+k)·v
//
// ─────────────────────────────────────────────────────────────────────────────

const GS_COMPUTE_SHADER = /* wgsl */`

// ── Uniforms ────────────────────────────────────────────────────────────────
struct GSUniforms {
  f    : f32,   // feed rate
  k    : f32,   // kill rate
  Du   : f32,   // diffusion rate, activator u
  Dv   : f32,   // diffusion rate, inhibitor v
  // grid dims — used for boundary clamping
  width  : u32,
  height : u32,
  _pad0  : u32,
  _pad1  : u32,
}
@group(0) @binding(0) var<uniform> u_gs : GSUniforms;

// ── Ping-pong storage textures ───────────────────────────────────────────────
@group(1) @binding(0) var readTex  : texture_2d<f32>;
@group(1) @binding(1) var writeTex : texture_storage_2d<rgba32float, write>;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn loadUV(coord: vec2i) -> vec2f {
  // Clamp-to-edge boundary condition (Neumann: zero-flux)
  let c = clamp(coord, vec2i(0, 0),
                vec2i(i32(u_gs.width) - 1, i32(u_gs.height) - 1));
  let s = textureLoad(readTex, c, 0);
  return s.rg;   // R=u, G=v
}

// ── Gray-Scott compute step ──────────────────────────────────────────────────
@compute @workgroup_size(${RD_WG}, ${RD_WG})
fn gs_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let st = vec2i(i32(gid.x), i32(gid.y));

  // Bounds check
  if (gid.x >= u_gs.width || gid.y >= u_gs.height) { return; }

  // ── Sample 3×3 neighbourhood ──────────────────────────────────────────────
  // Pearson isotropic Laplacian kernel:
  //   [ 0.05  0.20  0.05 ]
  //   [ 0.20 -1.00  0.20 ]
  //   [ 0.05  0.20  0.05 ]
  let c  = loadUV(st);
  let n  = loadUV(st + vec2i( 0,  1));
  let s  = loadUV(st + vec2i( 0, -1));
  let e  = loadUV(st + vec2i( 1,  0));
  let w  = loadUV(st + vec2i(-1,  0));
  let ne = loadUV(st + vec2i( 1,  1));
  let nw = loadUV(st + vec2i(-1,  1));
  let se = loadUV(st + vec2i( 1, -1));
  let sw = loadUV(st + vec2i(-1, -1));

  // ── Discrete Laplacian ────────────────────────────────────────────────────
  let lap = 0.20 * (n + s + e + w)
          + 0.05 * (ne + nw + se + sw)
          - 1.00 * c;

  // ── Gray-Scott reaction ───────────────────────────────────────────────────
  let uu   = c.x;
  let vv   = c.y;
  let uvv  = uu * vv * vv;

  // du/dt = Du·∇²u − u·v² + f·(1−u)
  let du   = u_gs.Du * lap.x - uvv + u_gs.f * (1.0 - uu);
  // dv/dt = Dv·∇²v + u·v² − (f+k)·v
  let dv   = u_gs.Dv * lap.y + uvv - (u_gs.f + u_gs.k) * vv;

  // ── Euler integration (Δt = 1.0) ─────────────────────────────────────────
  let newU = clamp(uu + du, 0.0, 1.0);
  let newV = clamp(vv + dv, 0.0, 1.0);

  textureStore(writeTex, st, vec4f(newU, newV, 0.0, 1.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// ReactionDiffusionSim
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WebGPU compute-based Gray-Scott reaction-diffusion simulator.
 *
 * Supports two modes:
 *
 * 1. **Uniform mode** (default, `perSpecies: false`):
 *    Single global (f, k, Du, Dv) applied to every texel. Backward-compatible
 *    with the original API — `setParams()` / `setSpecies()` control the whole grid.
 *
 * 2. **Per-species mode** (`perSpecies: true`):
 *    Each texel reads a species index from a species map texture and looks up
 *    its own (f, k, Du, Dv) from a storage buffer LUT.  Different regions of
 *    the grid can exhibit different Gray-Scott morphologies simultaneously.
 *
 * Usage (uniform mode — original API):
 * ```ts
 * const sim = new ReactionDiffusionSim(device, { width: 512, height: 512 });
 * await sim.init();
 * sim.setParams(parameterSpace('coral'));
 * ```
 *
 * Usage (per-species mode):
 * ```ts
 * const sim = new ReactionDiffusionSim(device, {
 *   width: 512, height: 512,
 *   perSpecies: true,
 * });
 * await sim.init();
 *
 * // Paint species regions onto the grid
 * sim.setSpeciesRegion([
 *   { speciesIndex: 1, x: 0,   y: 0,   w: 256, h: 256 },  // cil-eye → coral
 *   { speciesIndex: 2, x: 256, y: 0,   w: 256, h: 256 },  // cil-bolt → spirals
 *   { speciesIndex: 3, x: 0,   y: 256, w: 256, h: 256 },  // cil-vector → stripes
 *   { speciesIndex: 4, x: 256, y: 256, w: 256, h: 256 },  // cil-plus → dots
 * ]);
 *
 * // Optionally override a specific species' params:
 * sim.setSpeciesLUTEntry(1, { f: 0.06, k: 0.062, Du: 0.2097, Dv: 0.1050 });
 * ```
 */
export class ReactionDiffusionSim {

  private readonly device:      GPUDevice;
  private readonly width:       number;
  private readonly height:      number;
  private readonly substeps:    number;
  private readonly perSpecies:  boolean;

  // Ping-pong textures: A ↔ B
  private texA!: GPUTexture;
  private texB!: GPUTexture;

  // Uniform buffer: GSUniforms (32 bytes — 8 × f32/u32)
  private uniformBuf!: any /*GPUBuffer*/;

  // Pipeline & bind-group layouts
  private pipeline!:     GPUComputePipeline;
  private uniformBGL!:   GPUBindGroupLayout;
  private ppBGL!:        GPUBindGroupLayout;

  // Prebuilt bind-groups for both ping-pong directions
  private uniformBG!:   GPUBindGroup;
  private bgAtoB!:      GPUBindGroup;  // read A → write B
  private bgBtoA!:      GPUBindGroup;  // read B → write A

  // ── Per-species resources (only allocated when perSpecies=true) ──────────
  private speciesTex!:      GPUTexture;     // r32uint species map
  private speciesLUTBuf!:   GPUBuffer;      // storage buffer: vec4f × MAX_SPECIES
  private speciesBGL!:      GPUBindGroupLayout;
  private speciesBG!:       GPUBindGroup;

  /** CPU-side copy of the species parameter LUT. */
  private speciesLUT: Required<GrayScottParams>[] = [];

  /** CPU-side copy of the species map (Uint32, one per texel). */
  private speciesMapData!: Uint32Array;

  /** Current simulation parameters (uniform mode fallback). */
  private params: Required<GrayScottParams>;

  /** Total number of step() calls executed. */
  private frameIndex = 0;

  /** Whether init() has completed. */
  private ready = false;

  constructor(device: any /*GPUDevice*/, cfg: RDSimConfig = {}) {
    this.device     = device;
    this.width      = cfg.width      ?? RD_DEFAULT_SIZE;
    this.height     = cfg.height     ?? RD_DEFAULT_SIZE;
    this.substeps   = cfg.substeps   ?? RD_DEFAULT_SUBSTEPS;
    this.perSpecies = cfg.perSpecies ?? false;

    const base = parameterSpace('coral');
    const p    = cfg.params ?? base;
    this.params = {
      f:  p.f,
      k:  p.k,
      Du: p.Du ?? base.Du!,
      Dv: p.Dv ?? base.Dv!,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Init
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Compile the compute pipeline, allocate textures, and seed the grid.
   *
   * Must be called once (and awaited) before the first step().
   */
  async init(): Promise<void> {
    if (this.ready) return;

    this._createTextures();
    this._createUniformBuffer();

    if (this.perSpecies) {
      this._createSpeciesResources();
    }

    await this._createPipeline();
    this._createBindGroups();
    this._seedGrid();

    this.ready = true;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Per-frame API
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Execute `substeps` Gray-Scott compute passes on the GPU.
   *
   * The encoder is NOT submitted here — caller must submit at end of frame.
   *
   * @param encoder  Current frame's GPUCommandEncoder.
   */
  step(encoder: any /*GPUCommandEncoder*/): void {
    if (!this.ready) {
      throw new Error('ReactionDiffusionSim.init() must be awaited before step()');
    }

    this._writeUniforms();

    for (let i = 0; i < this.substeps; i++) {
      const pass = encoder.beginComputePass({
        label: `gs-step-${this.frameIndex * this.substeps + i}`,
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.uniformBG);
      // Even i → A→B, odd i → B→A  (ping-pong within the same frame)
      pass.setBindGroup(1, (i & 1) === 0 ? this.bgAtoB : this.bgBtoA);

      if (this.perSpecies) {
        pass.setBindGroup(2, this.speciesBG);
      }

      const wgX = Math.ceil(this.width  / RD_WG);
      const wgY = Math.ceil(this.height / RD_WG);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    }

    this.frameIndex++;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Parameter control (uniform mode)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Set Gray-Scott parameters directly (uniform mode fallback).
   *
   * In per-species mode this sets the fallback params for species index 0.
   * Changes take effect on the next step() call (GPU uniform is re-uploaded).
   */
  setParams(p: GrayScottParams): void {
    this.params.f  = p.f;
    this.params.k  = p.k;
    if (p.Du !== undefined) this.params.Du = p.Du;
    if (p.Dv !== undefined) this.params.Dv = p.Dv;
  }

  /**
   * Set parameters from a named species via the Munafo/Pearson parameter space.
   *
   * Equivalent to `sim.setParams(parameterSpace(name))`.
   */
  setSpecies(name: GrayScottSpecies): void {
    this.setParams(parameterSpace(name));
  }

  /** Read current (f, k, Du, Dv) values. */
  get currentParams(): Readonly<Required<GrayScottParams>> {
    return { ...this.params };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Per-species parameter control
  // ─────────────────────────────────────────────────────────────────────────

  /** Whether this simulator is in per-species mode. */
  get isPerSpecies(): boolean {
    return this.perSpecies;
  }

  /**
   * Update a single entry in the per-species parameter LUT.
   *
   * @param index   Species index (0–RD_MAX_SPECIES-1).
   * @param params  Full (f, k, Du, Dv) parameter set.
   * @throws        If `perSpecies` is false or index is out of range.
   */
  setSpeciesLUTEntry(index: number, params: Required<GrayScottParams>): void {
    if (!this.perSpecies) {
      throw new Error('setSpeciesLUTEntry() requires perSpecies=true');
    }
    if (index < 0 || index >= RD_MAX_SPECIES) {
      throw new RangeError(`Species index ${index} out of range [0, ${RD_MAX_SPECIES})`);
    }

    this.speciesLUT[index] = { ...params };
    this._writeSpeciesLUT();
  }

  /**
   * Replace the entire per-species parameter LUT.
   *
   * @param lut  Array of RD_MAX_SPECIES entries. Missing entries are filled
   *             with the coral fallback.
   * @throws     If `perSpecies` is false.
   */
  setSpeciesLUT(lut: Required<GrayScottParams>[]): void {
    if (!this.perSpecies) {
      throw new Error('setSpeciesLUT() requires perSpecies=true');
    }

    const fallback = speciesGrayScottParams('fluid');
    for (let i = 0; i < RD_MAX_SPECIES; i++) {
      this.speciesLUT[i] = lut[i] ? { ...lut[i] } : { ...fallback };
    }
    this._writeSpeciesLUT();
  }

  /**
   * Read the current per-species parameter LUT (CPU-side copy).
   * Returns a frozen shallow copy.
   */
  get currentSpeciesLUT(): ReadonlyArray<Readonly<Required<GrayScottParams>>> {
    return [...this.speciesLUT];
  }

  /**
   * Paint rectangular regions of the species map with species indices.
   *
   * Each region's `speciesIndex` determines which LUT entry (and therefore
   * which GS morphology) governs that area of the grid.
   *
   * @param regions  Array of SpeciesRegion descriptors.
   * @throws         If `perSpecies` is false.
   */
  setSpeciesRegion(regions: SpeciesRegion[]): void {
    if (!this.perSpecies) {
      throw new Error('setSpeciesRegion() requires perSpecies=true');
    }
    if (!this.ready) {
      throw new Error('init() must be awaited before setSpeciesRegion()');
    }

    for (const r of regions) {
      const si = Math.max(0, Math.min(r.speciesIndex, RD_MAX_SPECIES - 1));
      for (let row = r.y; row < r.y + r.h; row++) {
        for (let col = r.x; col < r.x + r.w; col++) {
          if (col >= 0 && col < this.width && row >= 0 && row < this.height) {
            this.speciesMapData[row * this.width + col] = si;
          }
        }
      }
    }

    this._uploadSpeciesMap();
  }

  /**
   * Write a raw species map array directly.
   *
   * @param data  Uint32Array of length width × height. Each value is a species index.
   * @throws      If `perSpecies` is false or data has wrong length.
   */
  setSpeciesMapRaw(data: Uint32Array): void {
    if (!this.perSpecies) {
      throw new Error('setSpeciesMapRaw() requires perSpecies=true');
    }
    const expected = this.width * this.height;
    if (data.length !== expected) {
      throw new Error(`Species map data length ${data.length} !== grid size ${expected}`);
    }

    this.speciesMapData.set(data);
    this._uploadSpeciesMap();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Accessors
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * The GPU texture that contains the *latest* simulation output.
   *
   * After substeps iterations per step():
   *   If substeps is even → last write was to texB (even i final = B).
   *   If substeps is odd  → last write was to texA (odd  i final = A).
   *
   * Channel layout: R=u (activator), G=v (inhibitor), B=speciesIdx(float), A=1.
   */
  get outputTexture(): GPUTexture {
    // After N substeps (0-indexed): last write was to
    //   substeps even → texB  (final i = substeps-1 is odd → wrote A)
    //   substeps odd  → texA  (final i = substeps-1 is even → wrote B)
    // Track based on (frameIndex * substeps) parity.
    const totalSteps = this.frameIndex * this.substeps;
    return (totalSteps & 1) === 0 ? this.texA : this.texB;
  }

  /** The species map GPU texture (r32uint). Only available in perSpecies mode. */
  get speciesMapTexture(): GPUTexture {
    if (!this.perSpecies) {
      throw new Error('speciesMapTexture requires perSpecies=true');
    }
    return this.speciesTex;
  }

  /** Grid width in cells. */
  get gridWidth(): number  { return this.width;  }
  /** Grid height in cells. */
  get gridHeight(): number { return this.height; }
  /** Total frames stepped so far. */
  get frame(): number      { return this.frameIndex; }

  // ─────────────────────────────────────────────────────────────────────────
  // Seeding
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Reset and re-seed the simulation grid.
   *
   * Initial condition (Pearson 1993):
   *   Most cells: u=1, v=0  (all-A steady state)
   *   Central square (20% of grid): u=0.5, v=0.25  (perturbation seed)
   *   + small uniform noise on the seed region to break symmetry
   *
   * In per-species mode, each species region gets its own seed point
   * at its centroid, enabling independent pattern nucleation.
   */
  resetSeed(): void {
    if (!this.ready) return;
    this._seedGrid();
    this.frameIndex = 0;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private helpers
  // ─────────────────────────────────────────────────────────────────────────

  private _createTextures(): void {
    const desc: GPUTextureDescriptor = {
      size:   { width: this.width, height: this.height },
      format: RD_TEX_FORMAT,
      usage:  GPUTextureUsage.TEXTURE_BINDING   // sampled read
            | GPUTextureUsage.STORAGE_BINDING   // storage write
            | GPUTextureUsage.COPY_DST          // CPU upload (seed)
            | GPUTextureUsage.COPY_SRC,         // readback (optional)
    };
    this.texA = this.device.createTexture({ ...desc, label: 'rd-tex-A' });
    this.texB = this.device.createTexture({ ...desc, label: 'rd-tex-B' });
  }

  private _createUniformBuffer(): void {
    // GSUniforms: 8 × 4 bytes = 32 bytes
    this.uniformBuf = this.device.createBuffer({
      label:  'rd-uniform',
      size:   32,
      usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._writeUniforms();
  }

  private _writeUniforms(): void {
    const f32 = new Float32Array(8);
    const u32 = new Uint32Array(f32.buffer);
    f32[0] = this.params.f;
    f32[1] = this.params.k;
    f32[2] = this.params.Du;
    f32[3] = this.params.Dv;
    u32[4] = this.width;
    u32[5] = this.height;
    u32[6] = 0;   // _pad0
    u32[7] = 0;   // _pad1
    this.device.queue.writeBuffer(this.uniformBuf, 0, f32);
  }

  // ── Per-species resource creation ─────────────────────────────────────────

  private _createSpeciesResources(): void {
    // Species map texture: r32uint, one species index per texel
    this.speciesTex = this.device.createTexture({
      label:  'rd-species-map',
      size:   { width: this.width, height: this.height },
      format: SPECIES_TEX_FORMAT,
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // CPU-side species map (all zeros = fluid/fallback)
    this.speciesMapData = new Uint32Array(this.width * this.height);

    // Species parameter LUT: vec4f × MAX_SPECIES = 16 × 16 = 256 bytes
    const lutSize = RD_MAX_SPECIES * 4 * 4;  // MAX_SPECIES × vec4f (4 × f32)
    this.speciesLUTBuf = this.device.createBuffer({
      label:  'rd-species-lut',
      size:   lutSize,
      usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Populate CPU-side LUT with default species→GS mapping
    this.speciesLUT = buildDefaultSpeciesLUT();
    this._writeSpeciesLUT();
  }

  private _writeSpeciesLUT(): void {
    // Pack LUT into a flat Float32Array: [f, k, Du, Dv] × MAX_SPECIES
    const data = new Float32Array(RD_MAX_SPECIES * 4);
    for (let i = 0; i < RD_MAX_SPECIES; i++) {
      const p = this.speciesLUT[i];
      data[i * 4 + 0] = p.f;
      data[i * 4 + 1] = p.k;
      data[i * 4 + 2] = p.Du;
      data[i * 4 + 3] = p.Dv;
    }
    this.device.queue.writeBuffer(this.speciesLUTBuf, 0, data);
  }

  private _uploadSpeciesMap(): void {
    // r32uint: 4 bytes per texel
    this.device.queue.writeTexture(
      { texture: this.speciesTex },
      this.speciesMapData,
      { bytesPerRow: this.width * 4, rowsPerImage: this.height },
      { width: this.width, height: this.height },
    );
  }

  // ── Pipeline creation ─────────────────────────────────────────────────────

  private async _createPipeline(): Promise<void> {
    const shaderCode = this.perSpecies ? GS_COMPUTE_SHADER_PER_SPECIES : GS_COMPUTE_SHADER;

    const shaderModule = this.device.createShaderModule({
      label: this.perSpecies ? 'rd-gs-shader-per-species' : 'rd-gs-shader',
      code:  shaderCode,
    });

    // BGL 0: uniform buffer
    this.uniformBGL = this.device.createBindGroupLayout({
      label:   'rd-uniform-bgl',
      entries: [
        {
          binding:    0,
          visibility: GPUShaderStage.COMPUTE,
          buffer:     { type: 'uniform' },
        },
      ],
    });

    // BGL 1: ping-pong (read texture + write storage texture)
    this.ppBGL = this.device.createBindGroupLayout({
      label:   'rd-pingpong-bgl',
      entries: [
        {
          binding:    0,
          visibility: GPUShaderStage.COMPUTE,
          texture:    { sampleType: 'unfilterable-float' },
        },
        {
          binding:        1,
          visibility:     GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: RD_TEX_FORMAT },
        },
      ],
    });

    const bindGroupLayouts: GPUBindGroupLayout[] = [this.uniformBGL, this.ppBGL];

    // BGL 2: per-species data (species map texture + LUT buffer)
    if (this.perSpecies) {
      this.speciesBGL = this.device.createBindGroupLayout({
        label:   'rd-species-bgl',
        entries: [
          {
            binding:    0,
            visibility: GPUShaderStage.COMPUTE,
            texture:    { sampleType: 'uint' },
          },
          {
            binding:    1,
            visibility: GPUShaderStage.COMPUTE,
            buffer:     { type: 'read-only-storage' },
          },
        ],
      });
      bindGroupLayouts.push(this.speciesBGL);
    }

    const layout = this.device.createPipelineLayout({
      label:            'rd-pipeline-layout',
      bindGroupLayouts,
    });

    this.pipeline = await this.device.createComputePipelineAsync({
      label:   this.perSpecies ? 'rd-gs-pipeline-per-species' : 'rd-gs-pipeline',
      layout,
      compute: { module: shaderModule, entryPoint: 'gs_step' },
    });
  }

  private _createBindGroups(): void {
    // Uniform bind-group (static — same for both ping-pong directions)
    this.uniformBG = this.device.createBindGroup({
      label:   'rd-uniform-bg',
      layout:  this.uniformBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
      ],
    });

    // A→B: read texA, write texB
    this.bgAtoB = this.device.createBindGroup({
      label:   'rd-bg-AtoB',
      layout:  this.ppBGL,
      entries: [
        { binding: 0, resource: this.texA.createView() },
        { binding: 1, resource: this.texB.createView() },
      ],
    });

    // B→A: read texB, write texA
    this.bgBtoA = this.device.createBindGroup({
      label:   'rd-bg-BtoA',
      layout:  this.ppBGL,
      entries: [
        { binding: 0, resource: this.texB.createView() },
        { binding: 1, resource: this.texA.createView() },
      ],
    });

    // Per-species bind-group
    if (this.perSpecies) {
      this.speciesBG = this.device.createBindGroup({
        label:   'rd-species-bg',
        layout:  this.speciesBGL,
        entries: [
          { binding: 0, resource: this.speciesTex.createView() },
          { binding: 1, resource: { buffer: this.speciesLUTBuf } },
        ],
      });
    }
  }

  private _seedGrid(): void {
    const w = this.width;
    const h = this.height;
    // RGBA32Float — 4 channels × 4 bytes each
    const data = new Float32Array(w * h * 4);

    // Background state: u=1, v=0  (stable "all-A" fixed point)
    for (let i = 0; i < w * h; i++) {
      data[i * 4 + 0] = 1.0;  // u
      data[i * 4 + 1] = 0.0;  // v
      data[i * 4 + 2] = 0.0;  // species index (float) — 0 = fluid
      data[i * 4 + 3] = 1.0;
    }

    if (this.perSpecies) {
      // ── Per-species seeding ──────────────────────────────────────────────
      // Identify connected regions in the species map and seed each one
      // at its centroid.  This ensures every species zone gets an independent
      // nucleation site, even if zones are scattered across the grid.
      this._seedPerSpeciesRegions(data);
    } else {
      // ── Single-species seeding (original Pearson 1993 IC) ────────────────
      // Central square: u=0.5, v=0.25 + small noise to break symmetry
      const cx   = Math.floor(w * 0.5);
      const cy   = Math.floor(h * 0.5);
      const half = Math.floor(Math.min(w, h) * 0.10);  // 10% of grid

      for (let dy = -half; dy <= half; dy++) {
        for (let dx = -half; dx <= half; dx++) {
          const px = cx + dx;
          const py = cy + dy;
          if (px < 0 || px >= w || py < 0 || py >= h) continue;

          const idx = (py * w + px) * 4;
          const noise = (Math.random() - 0.5) * 0.05;
          data[idx + 0] = 0.50 + noise;   // u
          data[idx + 1] = 0.25 + noise;   // v
        }
      }
    }

    // Upload to texA; texB starts as all-zero (GPU default)
    this.device.queue.writeTexture(
      { texture: this.texA },
      data,
      { bytesPerRow: w * 4 * 4, rowsPerImage: h },
      { width: w, height: h },
    );

    // Ensure texB has a defined initial state too (copy texA → texB)
    const enc = this.device.createCommandEncoder({ label: 'rd-seed-copy' });
    enc.copyTextureToTexture(
      { texture: this.texA },
      { texture: this.texB },
      { width: w, height: h },
    );
    this.device.queue.submit([enc.finish()]);

    // Reset frame counter so outputTexture tracks correctly
    this.frameIndex = 0;
  }

  /**
   * Seed each species region independently.
   *
   * Scans the species map to find the axis-aligned bounding box of each
   * species index, then places a Pearson 1993-style seed square at the
   * centroid of each region.  The seed size is proportional to the region's
   * smaller extent (10% of min(regionW, regionH), minimum 2 texels).
   *
   * The species index is written into the B channel of each texel so the
   * per-species compute shader can verify/preserve it across substeps.
   */
  private _seedPerSpeciesRegions(data: Float32Array): void {
    const w = this.width;
    const h = this.height;

    // Build per-species bounding boxes from the species map
    const bboxes: Map<number, { minX: number; minY: number; maxX: number; maxY: number }> = new Map();

    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        const si = this.speciesMapData[row * w + col];
        // Write species index into B channel for all texels
        data[(row * w + col) * 4 + 2] = si;

        if (si === 0) continue; // fluid background: no dedicated seed

        let bb = bboxes.get(si);
        if (!bb) {
          bb = { minX: col, minY: row, maxX: col, maxY: row };
          bboxes.set(si, bb);
        } else {
          bb.minX = Math.min(bb.minX, col);
          bb.minY = Math.min(bb.minY, row);
          bb.maxX = Math.max(bb.maxX, col);
          bb.maxY = Math.max(bb.maxY, row);
        }
      }
    }

    // Seed each species region at its centroid
    for (const [si, bb] of bboxes) {
      const regionW = bb.maxX - bb.minX + 1;
      const regionH = bb.maxY - bb.minY + 1;
      const cx = Math.floor((bb.minX + bb.maxX) / 2);
      const cy = Math.floor((bb.minY + bb.maxY) / 2);
      const seedHalf = Math.max(2, Math.floor(Math.min(regionW, regionH) * 0.10));

      for (let dy = -seedHalf; dy <= seedHalf; dy++) {
        for (let dx = -seedHalf; dx <= seedHalf; dx++) {
          const px = cx + dx;
          const py = cy + dy;
          if (px < 0 || px >= w || py < 0 || py >= h) continue;
          // Only seed within this species' own territory
          if (this.speciesMapData[py * w + px] !== si) continue;

          const idx = (py * w + px) * 4;
          const noise = (Math.random() - 0.5) * 0.05;
          data[idx + 0] = 0.50 + noise;   // u
          data[idx + 1] = 0.25 + noise;   // v
          // B channel already set to si above
        }
      }
    }

    // Also seed species 0 (fluid) regions if they have area — use grid centre fallback
    const cx0   = Math.floor(w * 0.5);
    const cy0   = Math.floor(h * 0.5);
    const half0 = Math.floor(Math.min(w, h) * 0.05);  // smaller seed for fluid

    for (let dy = -half0; dy <= half0; dy++) {
      for (let dx = -half0; dx <= half0; dx++) {
        const px = cx0 + dx;
        const py = cy0 + dy;
        if (px < 0 || px >= w || py < 0 || py >= h) continue;
        if (this.speciesMapData[py * w + px] !== 0) continue;

        const idx = (py * w + px) * 4;
        const noise = (Math.random() - 0.5) * 0.05;
        data[idx + 0] = 0.50 + noise;
        data[idx + 1] = 0.25 + noise;
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Destroy
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Release all WebGPU resources held by this simulator.
   * The instance must not be used after destroy().
   */
  destroy(): void {
    this.texA.destroy();
    this.texB.destroy();
    this.uniformBuf.destroy();

    if (this.perSpecies) {
      this.speciesTex.destroy();
      this.speciesLUTBuf.destroy();
    }

    this.ready = false;
  }
}
