/**
 * gpu-particle-sort.ts — M769
 * ─────────────────────────────────────────────────────────────────────────────
 * WebGPU radix sort for particle depth ordering.
 *
 * Replaces the O(n log²n) bitonic sort in particle-compositor.ts with an
 * O(n·k) four-pass LSD (Least Significant Digit) radix sort, where k = 8
 * passes of 4-bit digits yield a full 32-bit key sort.  For large particle
 * counts (>16 k) radix sort is substantially faster than bitonic on GPU
 * because it performs a fixed number of passes independent of N, and each
 * pass is fully parallelised via workgroup-local prefix sums.
 *
 * Algorithm — per 4-bit digit (8 passes total, LSB first):
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   Pass 1: HISTOGRAM
 *     Each workgroup processes a tile of particles and tallies local
 *     16-bucket histograms.  Workgroup histograms are written to a global
 *     histogram buffer (WG × 16 u32s).
 *
 *   Pass 2: PREFIX SUM
 *     Column-major exclusive prefix sum over the global histogram table
 *     so that each workgroup knows the global scatter offset for each
 *     digit bucket.
 *
 *   Pass 3: SCATTER
 *     Each particle reads its digit, looks up its global offset from the
 *     prefix-summed histogram, and scatters (key, index) into the output
 *     buffer at that position.  A workgroup-local rank is computed via a
 *     shared-memory exclusive scan within each digit bucket.
 *
 *   Pass 4: FLIP
 *     Ping-pong swap: output becomes input for the next digit pass.
 *
 * Key encoding
 * ─────────────────────────────────────────────────────────────────────────────
 * Depth values (f32) are converted to sort-order-preserving u32 keys via
 * `floatToOrderedUint`:
 *   - positive floats: flip sign bit (MSB 0→1, preserves magnitude order)
 *   - negative floats: flip all bits (reverses magnitude → correct order)
 * This is the same encoding used in particle-compositor.ts.
 *
 * Integration
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * ```ts
 * import { GPUParticleRadixSort } from '$lib/sph/gpu-particle-sort';
 *
 * const sorter = new GPUParticleRadixSort(device, maxParticles);
 * await sorter.build();
 *
 * // Each frame — provide a buffer of (key, index) u32 pairs:
 * const enc = device.createCommandEncoder();
 * sorter.sort(enc, particleCount);
 * device.queue.submit([enc.finish()]);
 *
 * // Read the sorted index buffer for instanced draw reordering:
 * const sortedBuf = sorter.sortedBuffer;  // storage buffer, (key, idx) pairs
 * ```
 *
 * The sorted buffer can be consumed directly by the compositor's vertex
 * shader via `sortBuf[2*ii+1]` — same layout as particle-compositor.ts.
 *
 * References
 * ─────────────────────────────────────────────────────────────────────────────
 *   - Merrill & Grimshaw, "High Performance and Scalable Radix Sorting"
 *   - Harada & Howes, "Introduction to GPU Radix Sort" (GPU Gems 3 ch.39)
 *   - src/lib/sph/particle-compositor.ts   (bitonic sort baseline)
 *   - src/lib/sph/webgpu-sph-compute.ts    (WebGPU compute patterns)
 *   - src/lib/sph/SPHGPUOrchestrator.ts    (prefix-sum reference)
 */

// ─── Constants ───────────────────────────────────────────────────────────────

/** Workgroup size for histogram and scatter passes. */








const WG_SIZE = 256 as const;

/** Radix bits per pass — 4-bit digits → 16 buckets. */
const RADIX_BITS = 4 as const;

/** Number of buckets per digit pass. */
const RADIX_BUCKETS = (1 << RADIX_BITS) as 16;

/** Total digit passes for a full 32-bit key sort. */
const NUM_PASSES = (32 / RADIX_BITS) as 8;

/** Byte stride of one (key, index) pair. */
const PAIR_STRIDE = 8 as const; // 2 × u32

/** Maximum supported particle count (must be a multiple of WG_SIZE). */
const MAX_PARTICLES = 1 << 20; // 1M particles

// ─── WGSL Shaders ────────────────────────────────────────────────────────────

// ─── Shared: float-to-key conversion (identical to particle-compositor.ts) ───

const FLOAT_TO_KEY_FN = /* wgsl */ `
// Convert a float to an unsigned int preserving sort order:
//   positive floats: flip sign bit (MSB 0→1)
//   negative floats: flip all bits (reverses magnitude ordering)
fn floatToOrderedUint(f: f32) -> u32 {
  let bits = bitcast<u32>(f);
  let mask = select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
  return bits ^ mask;
}

// Inverse: convert sorted u32 key back to original float.
fn orderedUintToFloat(u: u32) -> f32 {
  let mask = select(0x80000000u, 0xFFFFFFFFu, (u & 0x80000000u) == 0u);
  return bitcast<f32>(u ^ mask);
}
`;

// ─── Pass 1: Histogram ───────────────────────────────────────────────────────
//
// Each workgroup processes WG_SIZE elements and produces a local 16-bucket
// histogram.  The global histogram buffer has shape [numWorkgroups × 16].
// Element histBuf[wg * 16 + digit] = count of items in workgroup `wg`
// whose current 4-bit digit equals `digit`.

const HISTOGRAM_WGSL = /* wgsl */ `
struct RadixUniforms {
  count     : u32,   // number of active particles
  numGroups : u32,   // number of workgroups dispatched
  digitShift: u32,   // current digit bit offset (0, 4, 8, …, 28)
  _pad      : u32,
}

@group(0) @binding(0) var<uniform>             params  : RadixUniforms;
@group(0) @binding(1) var<storage, read>       keysIn  : array<u32>;
@group(0) @binding(2) var<storage, read_write> histBuf : array<atomic<u32>>;

var<workgroup> localHist: array<atomic<u32>, ${RADIX_BUCKETS}>;

@compute @workgroup_size(${WG_SIZE})
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>,
  @builtin(workgroup_id)         wid : vec3<u32>,
) {
  // Clear local histogram
  if (lid.x < ${RADIX_BUCKETS}u) {
    atomicStore(&localHist[lid.x], 0u);
  }
  workgroupBarrier();

  // Tally
  let idx = gid.x;
  if (idx < params.count) {
    let key   = keysIn[idx * 2u];  // key is at even indices, index at odd
    let digit = (key >> params.digitShift) & ${RADIX_BUCKETS - 1}u;
    atomicAdd(&localHist[digit], 1u);
  }
  workgroupBarrier();

  // Write local histogram to global buffer — column-major layout
  // histBuf[digit * numGroups + wid.x] = localHist[digit]
  // Column-major enables a simple single-workgroup prefix sum per column.
  if (lid.x < ${RADIX_BUCKETS}u) {
    let val = atomicLoad(&localHist[lid.x]);
    histBuf[lid.x * params.numGroups + wid.x] = val;
  }
}
`;

// ─── Pass 2: Global prefix sum over the histogram ────────────────────────────
//
// The histogram buffer has numGroups × 16 entries in column-major layout.
// We perform an exclusive prefix sum over the entire flattened array
// (length = 16 × numGroups) so that each entry becomes the global scatter
// offset for that (digit, workgroup) cell.
//
// For simplicity and correctness at moderate workgroup counts (≤4096),
// we use a two-level Blelloch scan: workgroup-local scan + block-sum
// propagation.  For ≤ 512 elements a single-workgroup scan suffices.

const PREFIX_SUM_WGSL = /* wgsl */ `
struct ScanUniforms {
  n : u32,      // total number of elements to scan
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}

@group(0) @binding(0) var<uniform>             scanParams : ScanUniforms;
@group(0) @binding(1) var<storage, read_write> data       : array<u32>;
@group(0) @binding(2) var<storage, read_write> blockSums  : array<u32>;

var<workgroup> temp: array<u32, ${WG_SIZE * 2}>;

@compute @workgroup_size(${WG_SIZE})
fn scan(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>,
  @builtin(workgroup_id)         wid : vec3<u32>,
) {
  let thid = lid.x;
  let blockSize = ${WG_SIZE * 2}u;
  let base = wid.x * blockSize;

  // Load two elements per thread into shared memory
  let ai = base + thid;
  let bi = base + thid + ${WG_SIZE}u;
  temp[thid]              = select(0u, data[ai], ai < scanParams.n);
  temp[thid + ${WG_SIZE}u] = select(0u, data[bi], bi < scanParams.n);
  workgroupBarrier();

  // Up-sweep (reduce)
  var offset = 1u;
  var d = ${WG_SIZE}u;
  loop {
    if (d == 0u) { break; }
    workgroupBarrier();
    if (thid < d) {
      let ai2 = offset * (2u * thid + 1u) - 1u;
      let bi2 = offset * (2u * thid + 2u) - 1u;
      temp[bi2] += temp[ai2];
    }
    offset *= 2u;
    d /= 2u;
  }

  // Store block sum and clear last element
  if (thid == 0u) {
    blockSums[wid.x] = temp[blockSize - 1u];
    temp[blockSize - 1u] = 0u;
  }
  workgroupBarrier();

  // Down-sweep
  d = 1u;
  loop {
    if (offset == 0u) { break; }
    workgroupBarrier();
    if (thid < d) {
      let ai2 = offset * (2u * thid + 1u) - 1u;
      let bi2 = offset * (2u * thid + 2u) - 1u;
      let t   = temp[ai2];
      temp[ai2] = temp[bi2];
      temp[bi2] += t;
    }
    offset /= 2u;
    d *= 2u;
  }
  workgroupBarrier();

  // Write back
  if (ai < scanParams.n) { data[ai] = temp[thid]; }
  if (bi < scanParams.n) { data[bi] = temp[thid + ${WG_SIZE}u]; }
}
`;

// Add-block-sums shader: after scanning block sums, propagate offsets back.

const ADD_BLOCK_SUMS_WGSL = /* wgsl */ `
struct ScanUniforms {
  n : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}

@group(0) @binding(0) var<uniform>             scanParams : ScanUniforms;
@group(0) @binding(1) var<storage, read_write> data       : array<u32>;
@group(0) @binding(2) var<storage, read>       blockSums  : array<u32>;

@compute @workgroup_size(${WG_SIZE})
fn addBlockSums(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(workgroup_id)         wid : vec3<u32>,
) {
  if (wid.x == 0u) { return; } // first block needs no offset
  let idx = gid.x;
  if (idx >= scanParams.n) { return; }
  // Also add the element at idx + WG_SIZE (two elements per thread in scan)
  data[idx] += blockSums[wid.x];
}
`;

// ─── Pass 3: Scatter ─────────────────────────────────────────────────────────
//
// Each workgroup re-reads its tile of (key, index) pairs, extracts the
// current digit, computes a workgroup-local rank via shared-memory
// exclusive scan per bucket, reads the global offset from the prefix-
// summed histogram, and writes the pair to the output buffer at position
// globalOffset + localRank.

const SCATTER_WGSL = /* wgsl */ `
struct RadixUniforms {
  count     : u32,
  numGroups : u32,
  digitShift: u32,
  _pad      : u32,
}

@group(0) @binding(0) var<uniform>             params  : RadixUniforms;
@group(0) @binding(1) var<storage, read>       pairsIn : array<u32>;
@group(0) @binding(2) var<storage, read_write> pairsOut: array<u32>;
@group(0) @binding(3) var<storage, read>       histBuf : array<u32>;

// Shared memory for local ranking
var<workgroup> localDigits: array<u32, ${WG_SIZE}>;
var<workgroup> localOffsets: array<u32, ${WG_SIZE}>;
var<workgroup> bucketBase: array<u32, ${RADIX_BUCKETS}>;

@compute @workgroup_size(${WG_SIZE})
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>,
  @builtin(workgroup_id)         wid : vec3<u32>,
) {
  let globalIdx = gid.x;
  let inBounds  = globalIdx < params.count;

  // Read this thread's key and compute its digit
  var key   = 0xFFFFFFFFu;  // dead particles → max key (sort to back)
  var val   = 0u;
  var digit = ${RADIX_BUCKETS - 1}u;

  if (inBounds) {
    key   = pairsIn[globalIdx * 2u];
    val   = pairsIn[globalIdx * 2u + 1u];
    digit = (key >> params.digitShift) & ${RADIX_BUCKETS - 1}u;
  }
  localDigits[lid.x] = digit;
  workgroupBarrier();

  // Load global prefix-sum offsets for this workgroup's buckets
  if (lid.x < ${RADIX_BUCKETS}u) {
    bucketBase[lid.x] = histBuf[lid.x * params.numGroups + wid.x];
  }
  workgroupBarrier();

  // Compute local rank: count how many threads before me in this
  // workgroup have the same digit.  A simple serial scan over the
  // workgroup is cheap at WG_SIZE=256 and avoids complex multi-bucket
  // shared scans.
  var rank = 0u;
  for (var i = 0u; i < lid.x; i++) {
    if (localDigits[i] == digit) {
      rank++;
    }
  }
  localOffsets[lid.x] = rank;
  workgroupBarrier();

  // Scatter to output
  if (inBounds) {
    let dst = bucketBase[digit] + rank;
    pairsOut[dst * 2u]      = key;
    pairsOut[dst * 2u + 1u] = val;
  }
}
`;

// ─── Key extraction from depth values ────────────────────────────────────────
//
// Extracts sort keys from a tPos texture or a raw depth buffer and writes
// (key, originalIndex) pairs into the sort input buffer.
// This mirrors the key extraction in particle-compositor.ts.

const KEY_EXTRACT_TEXTURE_WGSL = /* wgsl */ `
struct KeyExtractUniforms {
  particleCount : u32,
  texW          : u32,
  texH          : u32,
  sortKeyMode   : u32,  // 0 = travel (.b), 1 = worldY (.g)
}

@group(0) @binding(0) var<uniform>             params  : KeyExtractUniforms;
@group(1) @binding(0) var                      tPos    : texture_2d<f32>;
@group(1) @binding(1) var<storage, read_write> pairsOut: array<u32>;

${FLOAT_TO_KEY_FN}

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= params.particleCount) { return; }

  let texX = i32(slot % params.texW);
  let texY = i32(slot / params.texW);
  let p    = textureLoad(tPos, vec2<i32>(texX, texY), 0);

  // Choose depth key based on mode
  let rawKey = select(p.g, p.b, params.sortKeyMode == 0u);

  // Dead / invisible particles sort to the back (key = max)
  let alive = p.a > 0.004;
  let key   = select(0xFFFFFFFFu, floatToOrderedUint(rawKey), alive);

  pairsOut[slot * 2u]      = key;
  pairsOut[slot * 2u + 1u] = slot;
}
`;

// Key extraction from a raw f32 depth buffer (for 3-D scenes / z-buffer proxy)

const KEY_EXTRACT_BUFFER_WGSL = /* wgsl */ `
struct KeyExtractUniforms {
  particleCount : u32,
  _pad0         : u32,
  _pad1         : u32,
  _pad2         : u32,
}

@group(0) @binding(0) var<uniform>             params   : KeyExtractUniforms;
@group(0) @binding(1) var<storage, read>       depthBuf : array<f32>;
@group(0) @binding(2) var<storage, read_write> pairsOut : array<u32>;

${FLOAT_TO_KEY_FN}

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.particleCount) { return; }

  let depth = depthBuf[idx];
  let key   = floatToOrderedUint(depth);

  pairsOut[idx * 2u]      = key;
  pairsOut[idx * 2u + 1u] = idx;
}
`;

// ─── Public types ────────────────────────────────────────────────────────────

/** Depth key source mode. */
export type SortKeyMode = 'travel' | 'worldY' | 'buffer';

/** Configuration for building the radix sorter. */
export interface RadixSortConfig {
  /** Maximum number of particles the sorter can handle. Default 65536. */
  maxParticles?: number;
  /** Initial sort key mode.  Default 'travel'. */
  sortKeyMode?: SortKeyMode;
}

/** Metrics from the most recent sort pass (for profiling). */
export interface RadixSortMetrics {
  /** Number of particles actually sorted in the last call. */
  particleCount: number;
  /** Number of radix digit passes executed (always 8 for 32-bit keys). */
  digitPasses: number;
  /** Number of compute workgroups dispatched per pass. */
  workgroupsPerPass: number;
  /** Total histogram entries (numGroups × 16). */
  histogramSize: number;
}

// ─── Helper ──────────────────────────────────────────────────────────────────

/** Round up to the next multiple of `align`. */
function alignUp(n: number, align: number): number {
  return Math.ceil(n / align) * align;
}

// ─── GPUParticleRadixSort ────────────────────────────────────────────────────

/**
 * GPUParticleRadixSort
 *
 * Self-contained WebGPU radix sort for (key, index) u32 pairs.
 * After `build()`, call `sort(enc, count)` each frame to produce a
 * sorted pair buffer suitable for instanced-draw reordering.
 *
 * The sort is stable: equal keys preserve their original relative order.
 */
export class GPUParticleRadixSort {
  private readonly device: GPUDevice;
  private readonly maxParticles: number;

  // ── Ping-pong pair buffers ────────────────────────────────────────────────
  /** Buffer A: (key, index) pairs — input for even digit passes. */
  private pairBufA!: GPUBuffer;
  /** Buffer B: (key, index) pairs — output for even digit passes. */
  private pairBufB!: GPUBuffer;

  // ── Histogram buffer ──────────────────────────────────────────────────────
  private histBuf!: GPUBuffer;
  /** Maximum number of workgroups across any dispatch. */
  private maxWorkgroups: number;
  /** Maximum histogram size = maxWorkgroups × RADIX_BUCKETS. */
  private maxHistSize: number;

  // ── Prefix-sum buffers ────────────────────────────────────────────────────
  private scanBlockSumBuf!: GPUBuffer;

  // ── Uniform buffers ───────────────────────────────────────────────────────
  private radixUniformBuf!: GPUBuffer;
  private scanUniformBuf!:  GPUBuffer;

  // ── Pipelines ─────────────────────────────────────────────────────────────
  private histogramPipeline!:   GPUComputePipeline;
  private scanPipeline!:        GPUComputePipeline;
  private addBlockSumPipeline!: GPUComputePipeline;
  private scatterPipeline!:     GPUComputePipeline;

  // ── Key extraction pipelines ──────────────────────────────────────────────
  private keyExtractTexPipeline!:  GPUComputePipeline;
  private keyExtractBufPipeline!:  GPUComputePipeline;
  private keyExtractUniBuf!:       GPUBuffer;

  // ── Bind group layouts ────────────────────────────────────────────────────
  private histBGL!:        GPUBindGroupLayout;
  private scanBGL!:        GPUBindGroupLayout;
  private scatterBGL!:     GPUBindGroupLayout;
  private keyExtTexBGL0!:  GPUBindGroupLayout;
  private keyExtTexBGL1!:  GPUBindGroupLayout;
  private keyExtBufBGL!:   GPUBindGroupLayout;

  // ── State ─────────────────────────────────────────────────────────────────
  private built = false;
  private lastMetrics: RadixSortMetrics = {
    particleCount: 0,
    digitPasses: NUM_PASSES,
    workgroupsPerPass: 0,
    histogramSize: 0,
  };

  /** Current sort key mode. */
  sortKeyMode: SortKeyMode = 'travel';

  constructor(device: GPUDevice, config: RadixSortConfig = {}) {
    this.device       = device;
    this.maxParticles = Math.min(config.maxParticles ?? 65536, MAX_PARTICLES);
    this.sortKeyMode  = config.sortKeyMode ?? 'travel';

    this.maxWorkgroups = Math.ceil(this.maxParticles / WG_SIZE);
    this.maxHistSize   = this.maxWorkgroups * RADIX_BUCKETS;
  }

  // ═══════════════════════════════════════════════════════════════════
  // Build
  // ═══════════════════════════════════════════════════════════════════

  async build(): Promise<void> {
    if (this.built) this.destroy();

    const dev = this.device;

    // ── Buffers ──────────────────────────────────────────────────────────────

    const pairBytes = this.maxParticles * PAIR_STRIDE;

    this.pairBufA = dev.createBuffer({
      label: 'radix-sort:pairA',
      size:  pairBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.pairBufB = dev.createBuffer({
      label: 'radix-sort:pairB',
      size:  pairBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Histogram buffer — atomics in histogram pass, read in scatter pass
    this.histBuf = dev.createBuffer({
      label: 'radix-sort:histogram',
      size:  this.maxHistSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Block sums for prefix-sum scan
    const scanBlockCount = Math.ceil(this.maxHistSize / (WG_SIZE * 2));
    this.scanBlockSumBuf = dev.createBuffer({
      label: 'radix-sort:scan-block-sums',
      size:  Math.max(scanBlockCount * 4, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Uniform buffers
    this.radixUniformBuf = dev.createBuffer({
      label: 'radix-sort:radix-uniforms',
      size:  16, // 4 × u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.scanUniformBuf = dev.createBuffer({
      label: 'radix-sort:scan-uniforms',
      size:  16, // 4 × u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.keyExtractUniBuf = dev.createBuffer({
      label: 'radix-sort:key-extract-uniforms',
      size:  16, // 4 × u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ── Bind group layouts ───────────────────────────────────────────────────

    // Histogram BGL: [uniform, storage<read>(keysIn), storage<rw>(histBuf)]
    this.histBGL = dev.createBindGroupLayout({
      label: 'radix-sort:hist-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    // Scan BGL: [uniform, storage<rw>(data), storage<rw>(blockSums)]
    this.scanBGL = dev.createBindGroupLayout({
      label: 'radix-sort:scan-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    // Scatter BGL: [uniform, storage<read>(pairsIn), storage<rw>(pairsOut), storage<read>(histBuf)]
    this.scatterBGL = dev.createBindGroupLayout({
      label: 'radix-sort:scatter-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });

    // Key extract (texture) BGL0: [uniform]
    this.keyExtTexBGL0 = dev.createBindGroupLayout({
      label: 'radix-sort:key-ext-tex-bgl0',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    // Key extract (texture) BGL1: [texture, storage<rw>(pairsOut)]
    this.keyExtTexBGL1 = dev.createBindGroupLayout({
      label: 'radix-sort:key-ext-tex-bgl1',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    // Key extract (buffer) BGL: [uniform, storage<read>(depthBuf), storage<rw>(pairsOut)]
    this.keyExtBufBGL = dev.createBindGroupLayout({
      label: 'radix-sort:key-ext-buf-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    // ── Pipelines ────────────────────────────────────────────────────────────

    const histModule = dev.createShaderModule({
      label: 'radix-sort:histogram-module',
      code:  HISTOGRAM_WGSL,
    });
    this.histogramPipeline = dev.createComputePipeline({
      label:   'radix-sort:histogram-pipeline',
      layout:  dev.createPipelineLayout({ bindGroupLayouts: [this.histBGL] }),
      compute: { module: histModule, entryPoint: 'main' },
    });

    const scanModule = dev.createShaderModule({
      label: 'radix-sort:scan-module',
      code:  PREFIX_SUM_WGSL,
    });
    this.scanPipeline = dev.createComputePipeline({
      label:   'radix-sort:scan-pipeline',
      layout:  dev.createPipelineLayout({ bindGroupLayouts: [this.scanBGL] }),
      compute: { module: scanModule, entryPoint: 'scan' },
    });

    const addBlockModule = dev.createShaderModule({
      label: 'radix-sort:add-block-sums-module',
      code:  ADD_BLOCK_SUMS_WGSL,
    });
    this.addBlockSumPipeline = dev.createComputePipeline({
      label:   'radix-sort:add-block-sums-pipeline',
      layout:  dev.createPipelineLayout({ bindGroupLayouts: [this.scanBGL] }),
      compute: { module: addBlockModule, entryPoint: 'addBlockSums' },
    });

    const scatterModule = dev.createShaderModule({
      label: 'radix-sort:scatter-module',
      code:  SCATTER_WGSL,
    });
    this.scatterPipeline = dev.createComputePipeline({
      label:   'radix-sort:scatter-pipeline',
      layout:  dev.createPipelineLayout({ bindGroupLayouts: [this.scatterBGL] }),
      compute: { module: scatterModule, entryPoint: 'main' },
    });

    // Key extraction pipelines
    const keyExtTexModule = dev.createShaderModule({
      label: 'radix-sort:key-extract-tex-module',
      code:  KEY_EXTRACT_TEXTURE_WGSL,
    });
    this.keyExtractTexPipeline = dev.createComputePipeline({
      label:   'radix-sort:key-extract-tex-pipeline',
      layout:  dev.createPipelineLayout({
        bindGroupLayouts: [this.keyExtTexBGL0, this.keyExtTexBGL1],
      }),
      compute: { module: keyExtTexModule, entryPoint: 'main' },
    });

    const keyExtBufModule = dev.createShaderModule({
      label: 'radix-sort:key-extract-buf-module',
      code:  KEY_EXTRACT_BUFFER_WGSL,
    });
    this.keyExtractBufPipeline = dev.createComputePipeline({
      label:   'radix-sort:key-extract-buf-pipeline',
      layout:  dev.createPipelineLayout({ bindGroupLayouts: [this.keyExtBufBGL] }),
      compute: { module: keyExtBufModule, entryPoint: 'main' },
    });

    this.built = true;
  }

  // ═══════════════════════════════════════════════════════════════════
  // Key extraction — write (key, index) pairs into pairBufA
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Extract depth keys from a tPos texture and write (key, index) pairs
   * into the sort input buffer.
   *
   * @param enc       open command encoder
   * @param tPosView  tPos texture view (rgba32float)
   * @param count     number of active particles
   * @param texW      tPos width
   * @param texH      tPos height
   * @param mode      'travel' or 'worldY'
   */
  extractKeysFromTexture(
    enc:      GPUCommandEncoder,
    tPosView: GPUTextureView,
    count:    number,
    texW:     number,
    texH:     number,
    mode:     'travel' | 'worldY' = 'travel',
  ): void {
    if (!this.built) return;
    const dev = this.device;
    const n   = Math.min(count, this.maxParticles);
    const wg  = Math.ceil(n / WG_SIZE);

    // Write uniforms
    const uniData = new Uint32Array([n, texW, texH, mode === 'travel' ? 0 : 1]);
    dev.queue.writeBuffer(this.keyExtractUniBuf, 0, uniData);

    // Build bind groups
    const bg0 = dev.createBindGroup({
      layout:  this.keyExtTexBGL0,
      entries: [{ binding: 0, resource: { buffer: this.keyExtractUniBuf } }],
    });
    const bg1 = dev.createBindGroup({
      layout:  this.keyExtTexBGL1,
      entries: [
        { binding: 0, resource: tPosView },
        { binding: 1, resource: { buffer: this.pairBufA } },
      ],
    });

    const pass = enc.beginComputePass({ label: 'radix-sort:key-extract-tex' });
    pass.setPipeline(this.keyExtractTexPipeline);
    pass.setBindGroup(0, bg0);
    pass.setBindGroup(1, bg1);
    pass.dispatchWorkgroups(wg);
    pass.end();
  }

  /**
   * Extract depth keys from a raw f32 depth buffer and write (key, index)
   * pairs into the sort input buffer.
   *
   * @param enc       open command encoder
   * @param depthBuf  GPUBuffer containing one f32 per particle
   * @param count     number of active particles
   */
  extractKeysFromBuffer(
    enc:      GPUCommandEncoder,
    depthBuf: GPUBuffer,
    count:    number,
  ): void {
    if (!this.built) return;
    const dev = this.device;
    const n   = Math.min(count, this.maxParticles);
    const wg  = Math.ceil(n / WG_SIZE);

    // Write uniforms
    const uniData = new Uint32Array([n, 0, 0, 0]);
    dev.queue.writeBuffer(this.keyExtractUniBuf, 0, uniData);

    const bg = dev.createBindGroup({
      layout:  this.keyExtBufBGL,
      entries: [
        { binding: 0, resource: { buffer: this.keyExtractUniBuf } },
        { binding: 1, resource: { buffer: depthBuf } },
        { binding: 2, resource: { buffer: this.pairBufA } },
      ],
    });

    const pass = enc.beginComputePass({ label: 'radix-sort:key-extract-buf' });
    pass.setPipeline(this.keyExtractBufPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wg);
    pass.end();
  }

  // ═══════════════════════════════════════════════════════════════════
  // Sort — 8-pass LSD radix sort on pairBufA ↔ pairBufB
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Execute the full radix sort on the (key, index) pairs currently in
   * pairBufA.  After the call, `sortedBuffer` points to the buffer
   * containing the sorted result.
   *
   * Keys must already be written (via extractKeysFrom* or directly).
   *
   * @param enc   open command encoder (all passes are recorded sequentially)
   * @param count number of active particles to sort
   */
  sort(enc: GPUCommandEncoder, count: number): void {
    if (!this.built) return;

    const dev       = this.device;
    const n         = Math.min(count, this.maxParticles);
    const numGroups = Math.ceil(n / WG_SIZE);

    this.lastMetrics = {
      particleCount:     n,
      digitPasses:       NUM_PASSES,
      workgroupsPerPass: numGroups,
      histogramSize:     numGroups * RADIX_BUCKETS,
    };

    const histSize = numGroups * RADIX_BUCKETS;

    // Ping-pong references
    let srcBuf = this.pairBufA;
    let dstBuf = this.pairBufB;

    for (let pass = 0; pass < NUM_PASSES; pass++) {
      const digitShift = pass * RADIX_BITS;

      // ── Write radix uniforms ──────────────────────────────────────────
      const radixUni = new Uint32Array([n, numGroups, digitShift, 0]);
      dev.queue.writeBuffer(this.radixUniformBuf, 0, radixUni);

      // ── Clear histogram buffer ────────────────────────────────────────
      // We clear the entire maxHistSize region to avoid stale data
      dev.queue.writeBuffer(
        this.histBuf, 0,
        new Uint32Array(this.maxHistSize),
      );

      // ── Pass 1: Histogram ─────────────────────────────────────────────
      {
        const bg = dev.createBindGroup({
          layout:  this.histBGL,
          entries: [
            { binding: 0, resource: { buffer: this.radixUniformBuf } },
            { binding: 1, resource: { buffer: srcBuf } },
            { binding: 2, resource: { buffer: this.histBuf } },
          ],
        });
        const cp = enc.beginComputePass({ label: `radix:hist-${pass}` });
        cp.setPipeline(this.histogramPipeline);
        cp.setBindGroup(0, bg);
        cp.dispatchWorkgroups(numGroups);
        cp.end();
      }

      // ── Pass 2: Prefix sum over histogram ─────────────────────────────
      this.encodePrefixSum(enc, histSize, pass);

      // ── Pass 3: Scatter ───────────────────────────────────────────────
      {
        const bg = dev.createBindGroup({
          layout:  this.scatterBGL,
          entries: [
            { binding: 0, resource: { buffer: this.radixUniformBuf } },
            { binding: 1, resource: { buffer: srcBuf } },
            { binding: 2, resource: { buffer: dstBuf } },
            { binding: 3, resource: { buffer: this.histBuf } },
          ],
        });
        const cp = enc.beginComputePass({ label: `radix:scatter-${pass}` });
        cp.setPipeline(this.scatterPipeline);
        cp.setBindGroup(0, bg);
        cp.dispatchWorkgroups(numGroups);
        cp.end();
      }

      // ── Pass 4: Flip (ping-pong) ─────────────────────────────────────
      const tmp = srcBuf;
      srcBuf = dstBuf;
      dstBuf = tmp;
    }

    // After 8 passes (even number), the result is back in pairBufA.
    // If NUM_PASSES were odd, we'd need a final copy.  Since 8 is even,
    // srcBuf === pairBufA after the loop.
    this._sortedBuf = srcBuf;
  }

  /**
   * Encode a Blelloch exclusive prefix sum over `this.histBuf[0..n-1]`.
   * Uses a two-level scan for n > 2 × WG_SIZE.
   */
  private encodePrefixSum(
    enc:   GPUCommandEncoder,
    n:     number,
    label: number,
  ): void {
    const dev        = this.device;
    const blockSize  = WG_SIZE * 2;
    const numBlocks  = Math.ceil(n / blockSize);

    // Write scan uniforms
    const scanUni = new Uint32Array([n, 0, 0, 0]);
    dev.queue.writeBuffer(this.scanUniformBuf, 0, scanUni);

    // Clear block sums
    dev.queue.writeBuffer(
      this.scanBlockSumBuf, 0,
      new Uint32Array(Math.max(numBlocks, 1)),
    );

    // Phase A: scan each block
    {
      const bg = dev.createBindGroup({
        layout:  this.scanBGL,
        entries: [
          { binding: 0, resource: { buffer: this.scanUniformBuf } },
          { binding: 1, resource: { buffer: this.histBuf } },
          { binding: 2, resource: { buffer: this.scanBlockSumBuf } },
        ],
      });
      const cp = enc.beginComputePass({ label: `radix:scan-A-${label}` });
      cp.setPipeline(this.scanPipeline);
      cp.setBindGroup(0, bg);
      cp.dispatchWorkgroups(numBlocks);
      cp.end();
    }

    // Phase B: scan block sums (single workgroup) if more than one block
    if (numBlocks > 1) {
      // Write scan uniforms for block sums
      const blockScanUni = new Uint32Array([numBlocks, 0, 0, 0]);
      dev.queue.writeBuffer(this.scanUniformBuf, 0, blockScanUni);

      // Create a small temp buffer for block-of-block sums
      const bobBuf = dev.createBuffer({
        label: `radix:scan-bob-${label}`,
        size:  16, // single block → 1 block sum
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      dev.queue.writeBuffer(bobBuf, 0, new Uint32Array([0, 0, 0, 0]));

      {
        const bg = dev.createBindGroup({
          layout:  this.scanBGL,
          entries: [
            { binding: 0, resource: { buffer: this.scanUniformBuf } },
            { binding: 1, resource: { buffer: this.scanBlockSumBuf } },
            { binding: 2, resource: { buffer: bobBuf } },
          ],
        });
        const cp = enc.beginComputePass({ label: `radix:scan-B-${label}` });
        cp.setPipeline(this.scanPipeline);
        cp.setBindGroup(0, bg);
        cp.dispatchWorkgroups(1);
        cp.end();
      }

      // Restore original n for the add-block pass
      dev.queue.writeBuffer(this.scanUniformBuf, 0, scanUni);

      // Phase C: add block sums back
      {
        const bg = dev.createBindGroup({
          layout:  this.scanBGL,
          entries: [
            { binding: 0, resource: { buffer: this.scanUniformBuf } },
            { binding: 1, resource: { buffer: this.histBuf } },
            { binding: 2, resource: { buffer: this.scanBlockSumBuf } },
          ],
        });
        const cp = enc.beginComputePass({ label: `radix:scan-C-${label}` });
        cp.setPipeline(this.addBlockSumPipeline);
        cp.setBindGroup(0, bg);
        cp.dispatchWorkgroups(numBlocks);
        cp.end();
      }

      // Schedule temp buffer cleanup
      Promise.resolve().then(() => bobBuf.destroy());
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // Convenience: extract + sort in one call
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Full pipeline: extract keys from a tPos texture and sort.
   * The sorted (key, index) pairs are available in `sortedBuffer`
   * after GPU submission.
   */
  extractAndSort(
    enc:      GPUCommandEncoder,
    tPosView: GPUTextureView,
    count:    number,
    texW:     number,
    texH:     number,
    mode?:    'travel' | 'worldY',
  ): void {
    this.extractKeysFromTexture(enc, tPosView, count, texW, texH, mode ?? (this.sortKeyMode as 'travel' | 'worldY'));
    this.sort(enc, count);
  }

  /**
   * Full pipeline: extract keys from a depth buffer and sort.
   */
  extractAndSortFromBuffer(
    enc:      GPUCommandEncoder,
    depthBuf: GPUBuffer,
    count:    number,
  ): void {
    this.extractKeysFromBuffer(enc, depthBuf, count);
    this.sort(enc, count);
  }

  // ═══════════════════════════════════════════════════════════════════
  // Accessors
  // ═══════════════════════════════════════════════════════════════════

  /** After sort(), returns the buffer holding sorted (key, index) pairs. */
  private _sortedBuf: GPUBuffer | null = null;

  get sortedBuffer(): GPUBuffer {
    return this._sortedBuf ?? this.pairBufA;
  }

  /**
   * The input pair buffer (pairBufA).
   * Write (key, index) pairs here directly if not using extractKeysFrom*.
   * Layout: for particle i, `buf[2*i+0]` = key (u32), `buf[2*i+1]` = index (u32).
   */
  get inputBuffer(): GPUBuffer {
    return this.pairBufA;
  }

  /** True if build() has been called and resources are live. */
  get isBuilt(): boolean {
    return this.built;
  }

  /** Metrics from the most recent sort() call. */
  get metrics(): Readonly<RadixSortMetrics> {
    return this.lastMetrics;
  }

  /** Maximum particles this instance supports. */
  get capacity(): number {
    return this.maxParticles;
  }

  // ═══════════════════════════════════════════════════════════════════
  // CPU-side key encoding (for upload / debug)
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Encode a float depth value as a sort-order-preserving u32 key.
   * Matches the GPU's `floatToOrderedUint` exactly.
   */
  static floatToKey(f: number): number {
    const buf = new ArrayBuffer(4);
    new Float32Array(buf)[0] = f;
    let bits = new Uint32Array(buf)[0];
    const mask = (bits & 0x80000000) !== 0 ? 0xFFFFFFFF : 0x80000000;
    return (bits ^ mask) >>> 0;
  }

  /**
   * Decode a sort key back to the original float value.
   */
  static keyToFloat(u: number): number {
    const mask = (u & 0x80000000) === 0 ? 0xFFFFFFFF : 0x80000000;
    const bits = (u ^ mask) >>> 0;
    const buf = new ArrayBuffer(4);
    new Uint32Array(buf)[0] = bits;
    return new Float32Array(buf)[0];
  }

  /**
   * Pack an array of (depth, originalIndex) pairs into a Uint32Array
   * suitable for upload to `inputBuffer`.
   *
   * @param depths  f32 depth values per particle
   * @param count   number of active particles (≤ depths.length)
   * @returns       Uint32Array of length count×2 with (key, index) pairs
   */
  static packDepthPairs(depths: Float32Array, count?: number): Uint32Array {
    const n   = count ?? depths.length;
    const out = new Uint32Array(n * 2);
    for (let i = 0; i < n; i++) {
      out[i * 2]     = GPUParticleRadixSort.floatToKey(depths[i]);
      out[i * 2 + 1] = i;
    }
    return out;
  }

  // ═══════════════════════════════════════════════════════════════════
  // Cleanup
  // ═══════════════════════════════════════════════════════════════════

  destroy(): void {
    if (!this.built) return;

    this.pairBufA.destroy();
    this.pairBufB.destroy();
    this.histBuf.destroy();
    this.scanBlockSumBuf.destroy();
    this.radixUniformBuf.destroy();
    this.scanUniformBuf.destroy();
    this.keyExtractUniBuf.destroy();

    this._sortedBuf = null;
    this.built = false;
  }
}

// ─── Exported constants ──────────────────────────────────────────────────────

export const RADIX_SORT_CONSTANTS = {
  WG_SIZE,
  RADIX_BITS,
  RADIX_BUCKETS,
  NUM_PASSES,
  PAIR_STRIDE,
  MAX_PARTICLES,
} as const;
