/**
 * cell-compressed-tex.ts — M020: ASTC / ETC2 / BC 压缩纹理支持
 *
 * 大 topology 图（节点数 > LARGE_TOPOLOGY_THRESHOLD）的 GPU 内存优化：
 * 使用硬件压缩纹理代替 RGBA8 unorm，减少显存占用 4–8×。
 *
 * ─── 格式优先级（从最优到回退）───────────────────────────────────────────────
 *   Mobile / ARM GPU   → ASTC 4×4 unorm        (6:1 ~1 bpp, KHR extension)
 *   Mobile + desktop   → ETC2 RGB8 / RGBA8      (4:1 ~0.5-1 bpp, WEBGL_compressed_texture_etc)
 *   Desktop (Windows)  → BC7 rgba-unorm         (8:1 ~1 bpp, EXT_texture_compression_bptc)
 *   Desktop fallback   → BC3 DXT5 rgba-unorm    (4:1 ~1 bpp, WEBGL_compressed_texture_s3tc)
 *   Final fallback     → RGBA8 (uncompressed)
 *
 * ─── 上游融合源 ──────────────────────────────────────────────────────────────
 *   upstream/pixijs-engine/src/compressed-textures/
 *     ktx/parseKTX.ts        — KTX binary header + mipmap level extraction
 *     ktx/loadKTX.ts         — LoaderParser  wrapper → CompressedSource
 *     dds/parseDDS.ts        — DDS binary header + block-aligned mip levels
 *     dds/loadDDS.ts         — LoaderParser wrapper
 *     dds/const.ts           — DXGI_FORMAT enum, block-size table
 *     ktx2/const.ts          — GL_INTERNAL_FORMAT enum → TEXTURE_FORMATS map
 *     shared/detectCompressed.ts  — format priority & extension detection
 *     shared/resolveCompressedTextureUrl.ts — validFormats list
 *   upstream/pixijs-engine/src/rendering/renderers/
 *     gl/texture/utils/getSupportedGlCompressedTextureFormats.ts — WebGL ext probes
 *     shared/texture/sources/CompressedSource.ts — uploadMethodId='compressed'
 *     gl/texture/uploaders/glUploadCompressedTextureResource.ts  — compressedTexImage2D
 *     gl/texture/utils/mapFormatToGlInternalFormat.ts            — ASTC/ETC2/BC→GL enum
 *
 * ─── 本模块职责 ──────────────────────────────────────────────────────────────
 *   CompressedTexProbe     — 运行时 WebGL/WebGPU 扩展探测 + 格式能力缓存
 *   CompressedTexManager   — 生命周期管理：分配 / 释放 / 内存统计
 *   TopologyTexAtlas       — 大 topology 图的纹理图集分配器（bin-packing light）
 *   parseKTXBuffer()       — KTX ArrayBuffer → TextureSourceOptions (ASTC/ETC2)
 *   parseDDSBuffer()       — DDS ArrayBuffer → TextureSourceOptions (BC7/BC3)
 *   createTopologyTex()    — 从 topology 的 cells RGBA 像素数据生成压缩纹理
 *   selectBestFormat()     — 根据 CompressedTexCapabilities 选择最佳格式
 *   estimateMemorySaving() — 计算压缩 vs RGBA8 的显存节省率
 *
 * ─── 使用方式 ────────────────────────────────────────────────────────────────
 *   const probe = await CompressedTexProbe.get();
 *   if (probe.hasCompression && cells.length > LARGE_TOPOLOGY_THRESHOLD) {
 *     const mgr = new CompressedTexManager(renderer);
 *     const atlas = await mgr.allocateTopologyAtlas(cells, probe.bestFormat);
 *     // atlas.source → CompressedSource → upload via glUploadCompressedTextureResource
 *   }
 *
 * [CELL-COMPRESSED-TEX] debug prefix.
 */

// ── PixiJS upstream types ────────────────────────────────────────────────────
import { CompressedSource }
  from '../../upstream/pixijs-engine/src/rendering/renderers/shared/texture/sources/CompressedSource';
import { TextureSource }
  from '../../upstream/pixijs-engine/src/rendering/renderers/shared/texture/sources/TextureSource';
import { Texture }
  from '../../upstream/pixijs-engine/src/rendering/renderers/shared/texture/Texture';

import type { TEXTURE_FORMATS }
  from '../../upstream/pixijs-engine/src/rendering/renderers/shared/texture/const';
import type { TextureSourceOptions }
  from '../../upstream/pixijs-engine/src/rendering/renderers/shared/texture/sources/TextureSource';
import type { Renderer }
  from '../../upstream/pixijs-engine/src/rendering/renderers/types';

// ── KTX2 container parser (npm: ktx-parse ^1.1.0) ────────────────────────────
import { read as readKTX2 } from 'ktx-parse';
import type { KTX2Container } from 'ktx-parse';

// ── Cell types (pixi-cell-renderer compat) ───────────────────────────────────
import type { CellDescriptor } from './pixi-cell-renderer';

// ═══════════════════════════════════════════════════════════════════════════════
// § 0. Constants
// ═══════════════════════════════════════════════════════════════════════════════

/** Cell count above which compressed textures are preferred. */
export const LARGE_TOPOLOGY_THRESHOLD = 64;

/**
 * ASTC block dimensions in texels (width × height).
 * 4×4 is the finest — 8 bpp; 12×12 is the coarsest — ~0.89 bpp.
 * We default to 4×4 for best quality on topology icons.
 */
export const ASTC_BLOCK_4x4 = { w: 4, h: 4, bytesPerBlock: 16 } as const;

/** Bytes per block for the key compressed formats. */
export const COMPRESSED_BLOCK_BYTES: Partial<Record<TEXTURE_FORMATS, number>> = {
  'astc-4x4-unorm':      16,
  'astc-4x4-unorm-srgb': 16,
  'astc-8x8-unorm':      16,
  'astc-8x8-unorm-srgb': 16,
  'etc2-rgba8unorm':     16,
  'etc2-rgba8unorm-srgb':16,
  'etc2-rgb8unorm':       8,
  'etc2-rgb8unorm-srgb':  8,
  'bc7-rgba-unorm':      16,
  'bc7-rgba-unorm-srgb': 16,
  'bc3-rgba-unorm':      16,
  'bc3-rgba-unorm-srgb': 16,
  'bc1-rgba-unorm':       8,
  'bc1-rgba-unorm-srgb':  8,
};

/** Block footprint (W×H texels) for each format family. */
export const COMPRESSED_BLOCK_DIM: Partial<Record<TEXTURE_FORMATS, { w: number; h: number }>> = {
  'astc-4x4-unorm':      { w: 4, h: 4 },
  'astc-4x4-unorm-srgb': { w: 4, h: 4 },
  'astc-8x8-unorm':      { w: 8, h: 8 },
  'astc-8x8-unorm-srgb': { w: 8, h: 8 },
  'etc2-rgba8unorm':     { w: 4, h: 4 },
  'etc2-rgba8unorm-srgb':{ w: 4, h: 4 },
  'etc2-rgb8unorm':      { w: 4, h: 4 },
  'etc2-rgb8unorm-srgb': { w: 4, h: 4 },
  'bc7-rgba-unorm':      { w: 4, h: 4 },
  'bc7-rgba-unorm-srgb': { w: 4, h: 4 },
  'bc3-rgba-unorm':      { w: 4, h: 4 },
  'bc3-rgba-unorm-srgb': { w: 4, h: 4 },
  'bc1-rgba-unorm':      { w: 4, h: 4 },
  'bc1-rgba-unorm-srgb': { w: 4, h: 4 },
};

// ═══════════════════════════════════════════════════════════════════════════════
// § 1. CompressedTexCapabilities — runtime format probe
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Fine-grained compressed texture capability flags for the current GPU.
 * Mirrors the extension probes in upstream:
 *   getSupportedGlCompressedTextureFormats.ts
 *   getSupportedGPUCompressedTextureFormats.ts
 */
export interface CompressedTexCapabilities {
  /** WEBGL_compressed_texture_astc — ARM / Mali / Adreno mobile GPU */
  astc: boolean;
  /** WEBGL_compressed_texture_etc — OpenGL ES 3.0 baseline (Android/iOS WebGL2) */
  etc2: boolean;
  /** EXT_texture_compression_bptc — BC6H + BC7 (desktop, D3D11 class GPU) */
  bptc: boolean;
  /** WEBGL_compressed_texture_s3tc — BC1/BC3 (desktop, near-universal) */
  s3tc: boolean;
  /** WEBGL_compressed_texture_s3tc_srgb — sRGB variant of s3tc */
  s3tc_srgb: boolean;
  /** EXT_texture_compression_rgtc — BC4 / BC5 (desktop) */
  rgtc: boolean;
  /** True when any compression is available */
  hasAny: boolean;
  /** Highest-quality format available, or null if none */
  best: TEXTURE_FORMATS | null;
  /** All supported compressed TEXTURE_FORMATS, priority-ordered */
  formats: TEXTURE_FORMATS[];
}

// ── GL internal format constants for extension enum values ───────────────────
// (mirrors upstream/pixijs-engine/src/compressed-textures/ktx2/const.ts)

const GL_COMPRESSED: Record<string, number> = {
  // ASTC (KHR_texture_compression_astc_ldr)
  COMPRESSED_RGBA_ASTC_4x4_KHR:           0x93B0,
  COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:   0x93D0,
  COMPRESSED_RGBA_ASTC_8x8_KHR:           0x93B7,
  COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:   0x93D7,
  // ETC2 (WEBGL_compressed_texture_etc)
  COMPRESSED_RGB8_ETC2:                   0x9274,
  COMPRESSED_RGBA8_ETC2_EAC:              0x9278,
  COMPRESSED_SRGB8_ETC2:                  0x9275,
  COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:       0x9279,
  // BC7 (EXT_texture_compression_bptc)
  COMPRESSED_RGBA_BPTC_UNORM_EXT:         0x8E8C,
  COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:   0x8E8D,
  // BC3 DXT5 (WEBGL_compressed_texture_s3tc)
  COMPRESSED_RGBA_S3TC_DXT5_EXT:          0x83F3,
  COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:    35919,
  // BC1 DXT1
  COMPRESSED_RGBA_S3TC_DXT1_EXT:          0x83F1,
};

/**
 * CompressedTexProbe — singleton WebGL extension probe.
 *
 * Uses the same strategy as upstream getSupportedGlCompressedTextureFormats.ts:
 * create a temporary 1×1 canvas WebGL context, query extensions, destroy canvas.
 *
 * Results are cached — call CompressedTexProbe.get() freely.
 */
export class CompressedTexProbe {
  private static _cached: CompressedTexCapabilities | null = null;

  private constructor() {}

  /**
   * Probe the current browser's WebGL compressed texture extension support.
   * Caches result after first call.
   */
  static get(): CompressedTexCapabilities {
    if (CompressedTexProbe._cached) return CompressedTexProbe._cached;

    // SSR / non-browser guard
    if (typeof document === 'undefined') {
      return (CompressedTexProbe._cached = CompressedTexProbe._nullCaps());
    }

    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;

    // Prefer WebGL2 — ASTC / ETC2 are core in WebGL2 on compliant devices.
    const gl =
      (canvas.getContext('webgl2') as WebGL2RenderingContext | null) ??
      (canvas.getContext('webgl') as WebGLRenderingContext | null);

    if (!gl) {
      return (CompressedTexProbe._cached = CompressedTexProbe._nullCaps());
    }

    const hasAstc  = !!gl.getExtension('WEBGL_compressed_texture_astc');
    const hasEtc2  = !!gl.getExtension('WEBGL_compressed_texture_etc');
    const hasBptc  = !!gl.getExtension('EXT_texture_compression_bptc');
    const hasS3tc  = !!gl.getExtension('WEBGL_compressed_texture_s3tc');
    const hasS3tcS = !!gl.getExtension('WEBGL_compressed_texture_s3tc_srgb');
    const hasRgtc  = !!gl.getExtension('EXT_texture_compression_rgtc');

    // Discard temporary context
    const ext = gl.getExtension('WEBGL_lose_context');
    ext?.loseContext();

    const formats: TEXTURE_FORMATS[] = [];
    if (hasAstc)  formats.push('astc-4x4-unorm', 'astc-4x4-unorm-srgb', 'astc-8x8-unorm', 'astc-8x8-unorm-srgb');
    if (hasEtc2)  formats.push('etc2-rgba8unorm', 'etc2-rgba8unorm-srgb', 'etc2-rgb8unorm', 'etc2-rgb8unorm-srgb');
    if (hasBptc)  formats.push('bc7-rgba-unorm', 'bc7-rgba-unorm-srgb');
    if (hasS3tc)  formats.push('bc3-rgba-unorm');
    if (hasS3tcS) formats.push('bc3-rgba-unorm-srgb');
    if (hasS3tc)  formats.push('bc1-rgba-unorm');

    const hasAny = formats.length > 0;
    const best   = formats[0] ?? null;

    const caps: CompressedTexCapabilities = {
      astc:      hasAstc,
      etc2:      hasEtc2,
      bptc:      hasBptc,
      s3tc:      hasS3tc,
      s3tc_srgb: hasS3tcS,
      rgtc:      hasRgtc,
      hasAny,
      best,
      formats,
    };

    console.debug('[CELL-COMPRESSED-TEX] probe caps:', caps);
    return (CompressedTexProbe._cached = caps);
  }

  /** Reset probe cache (for testing). */
  static _reset(): void {
    CompressedTexProbe._cached = null;
  }

  private static _nullCaps(): CompressedTexCapabilities {
    return {
      astc: false, etc2: false, bptc: false,
      s3tc: false, s3tc_srgb: false, rgtc: false,
      hasAny: false, best: null, formats: [],
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// § 2. Format selection
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Select the best compressed format for topology textures given the probe result.
 *
 * Priority order (mirrors upstream resolveCompressedTextureUrl validFormats):
 *   ASTC 4×4 → ETC2 RGBA8 → BC7 → BC3 → BC1 → null (fall back to RGBA8)
 *
 * @param caps  Result of CompressedTexProbe.get()
 * @param srgb  True when texture data is in sRGB colour space (default: false)
 */
export function selectBestFormat(
  caps: CompressedTexCapabilities,
  srgb = false,
): TEXTURE_FORMATS | null {
  if (caps.astc)  return srgb ? 'astc-4x4-unorm-srgb'  : 'astc-4x4-unorm';
  if (caps.etc2)  return srgb ? 'etc2-rgba8unorm-srgb'  : 'etc2-rgba8unorm';
  if (caps.bptc)  return srgb ? 'bc7-rgba-unorm-srgb'   : 'bc7-rgba-unorm';
  if (caps.s3tc_srgb && srgb) return 'bc3-rgba-unorm-srgb';
  if (caps.s3tc)  return 'bc3-rgba-unorm';
  return null;
}

// ═══════════════════════════════════════════════════════════════════════════════
// § 3. KTX binary parser (ASTC / ETC2 texture container)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * KTX file identifier — 12 bytes
 * @see https://www.khronos.org/opengles/sdk/tools/KTX/file_format_spec/#2.1
 */
const KTX_FILE_IDENTIFIER = [
  0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31,
  0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
] as const;

const KTX_FIELDS = {
  FILE_IDENTIFIER:        0,
  ENDIANNESS:            12,
  GL_TYPE:               16,
  GL_TYPE_SIZE:          20,
  GL_FORMAT:             24,
  GL_INTERNAL_FORMAT:    28,
  GL_BASE_INTERNAL_FORMAT: 32,
  PIXEL_WIDTH:           36,
  PIXEL_HEIGHT:          40,
  PIXEL_DEPTH:           44,
  NUMBER_OF_ARRAY_ELEMENTS: 48,
  NUMBER_OF_FACES:       52,
  NUMBER_OF_MIPMAP_LEVELS:  56,
  BYTES_OF_KEY_VALUE_DATA:  60,
} as const;
const KTX_FILE_HEADER_SIZE = 64;
const KTX_ENDIANNESS       = 0x04030201;

/**
 * GL internal format → TEXTURE_FORMATS.
 * Mirrors upstream/pixijs-engine/src/compressed-textures/ktx2/const.ts
 * INTERNAL_FORMAT_TO_TEXTURE_FORMATS (subset — ASTC + ETC2 + BC).
 */
const GL_INTERNAL_FORMAT_TO_TEXTURE: Record<number, TEXTURE_FORMATS> = {
  // ETC2
  0x9274: 'etc2-rgb8unorm',
  0x9275: 'etc2-rgb8unorm-srgb',
  0x9278: 'etc2-rgba8unorm',
  0x9279: 'etc2-rgba8unorm-srgb',
  0x9276: 'etc2-rgb8a1unorm',
  0x9277: 'etc2-rgb8a1unorm-srgb',
  // EAC
  0x9270: 'eac-r11unorm',
  0x9272: 'eac-rg11unorm',
  // ASTC
  0x93B0: 'astc-4x4-unorm',      0x93D0: 'astc-4x4-unorm-srgb',
  0x93B1: 'astc-5x4-unorm',      0x93D1: 'astc-5x4-unorm-srgb',
  0x93B2: 'astc-5x5-unorm',      0x93D2: 'astc-5x5-unorm-srgb',
  0x93B3: 'astc-6x5-unorm',      0x93D3: 'astc-6x5-unorm-srgb',
  0x93B4: 'astc-6x6-unorm',      0x93D4: 'astc-6x6-unorm-srgb',
  0x93B5: 'astc-8x5-unorm',      0x93D5: 'astc-8x5-unorm-srgb',
  0x93B6: 'astc-8x6-unorm',      0x93D6: 'astc-8x6-unorm-srgb',
  0x93B7: 'astc-8x8-unorm',      0x93D7: 'astc-8x8-unorm-srgb',
  0x93B8: 'astc-10x5-unorm',     0x93D8: 'astc-10x5-unorm-srgb',
  0x93B9: 'astc-10x6-unorm',     0x93D9: 'astc-10x6-unorm-srgb',
  0x93BA: 'astc-10x8-unorm',     0x93DA: 'astc-10x8-unorm-srgb',
  0x93BB: 'astc-10x10-unorm',    0x93DB: 'astc-10x10-unorm-srgb',
  0x93BC: 'astc-12x10-unorm',    0x93DC: 'astc-12x10-unorm-srgb',
  0x93BD: 'astc-12x12-unorm',    0x93DD: 'astc-12x12-unorm-srgb',
  // BC (S3TC / DXT)
  0x83F0: 'bc1-rgba-unorm',   // COMPRESSED_RGB_S3TC_DXT1_EXT
  0x83F1: 'bc1-rgba-unorm',   // COMPRESSED_RGBA_S3TC_DXT1_EXT
  0x83F2: 'bc2-rgba-unorm',   // COMPRESSED_RGBA_S3TC_DXT3_EXT
  0x83F3: 'bc3-rgba-unorm',   // COMPRESSED_RGBA_S3TC_DXT5_EXT
  35916: 'bc1-rgba-unorm-srgb',
  35917: 'bc1-rgba-unorm-srgb',
  35918: 'bc2-rgba-unorm-srgb',
  35919: 'bc3-rgba-unorm-srgb',
  // RGTC (BC4/BC5)
  0x8DBB: 'bc4-r-unorm',  0x8DBC: 'bc4-r-snorm',
  0x8DBD: 'bc5-rg-unorm', 0x8DBE: 'bc5-rg-snorm',
  // BPTC (BC6H / BC7)
  0x8E8C: 'bc7-rgba-unorm',        0x8E8D: 'bc7-rgba-unorm-srgb',
  0x8E8E: 'bc6h-rgb-float',        0x8E8F: 'bc6h-rgb-ufloat',
  // Uncompressed (KTX may embed RGBA8)
  0x8058: 'rgba8unorm',            // RGBA8
  0x8C43: 'rgba8unorm-srgb',       // SRGB8_ALPHA8
};

/** Bytes-per-pixel (or per block at 1×1 texel) for compressed GL formats. */
const GL_INTERNAL_FORMAT_BPP: Record<number, number> = {
  // ETC2 (4×4 block = 8 or 16 bytes)
  0x9274: 0.5, 0x9275: 0.5, 0x9276: 0.5, 0x9277: 0.5,
  0x9278: 1.0, 0x9279: 1.0,
  // EAC
  0x9270: 0.5, 0x9272: 1.0,
  // ASTC 4×4 = 1 bpp
  0x93B0: 1.0, 0x93D0: 1.0,
  // ASTC 8×8 = 0.25 bpp
  0x93B7: 0.25, 0x93D7: 0.25,
  // BC1
  0x83F0: 0.5, 0x83F1: 0.5, 35916: 0.5, 35917: 0.5,
  // BC2
  0x83F2: 1.0, 35918: 1.0,
  // BC3
  0x83F3: 1.0, 35919: 1.0,
  // BC7 / BC6H
  0x8E8C: 1.0, 0x8E8D: 1.0, 0x8E8E: 1.0, 0x8E8F: 1.0,
};

function validateKTXHeader(dv: DataView): boolean {
  for (let i = 0; i < KTX_FILE_IDENTIFIER.length; i++) {
    if (dv.getUint8(i) !== KTX_FILE_IDENTIFIER[i]) return false;
  }
  return true;
}

/**
 * Parse a KTX ArrayBuffer into TextureSourceOptions (ASTC/ETC2/BC family).
 *
 * Fuses upstream/pixijs-engine/src/compressed-textures/ktx/parseKTX.ts logic
 * into a standalone function usable without PixiJS Assets pipeline.
 *
 * @param buf  Raw KTX file bytes
 * @throws if the KTX header is invalid or the format is unsupported
 */
export function parseKTXBuffer(buf: ArrayBuffer): TextureSourceOptions<Uint8Array[]> {
  const dv = new DataView(buf);

  if (!validateKTXHeader(dv)) {
    throw new Error('[CELL-COMPRESSED-TEX] Invalid KTX file identifier');
  }

  const littleEndian      = dv.getUint32(KTX_FIELDS.ENDIANNESS, true) === KTX_ENDIANNESS;
  const glType            = dv.getUint32(KTX_FIELDS.GL_TYPE,            littleEndian);
  const glFormat          = dv.getUint32(KTX_FIELDS.GL_FORMAT,          littleEndian);
  const glInternalFormat  = dv.getUint32(KTX_FIELDS.GL_INTERNAL_FORMAT, littleEndian);
  const pixelWidth        = dv.getUint32(KTX_FIELDS.PIXEL_WIDTH,        littleEndian);
  const pixelHeight       = dv.getUint32(KTX_FIELDS.PIXEL_HEIGHT,       littleEndian) || 1;
  const pixelDepth        = dv.getUint32(KTX_FIELDS.PIXEL_DEPTH,        littleEndian) || 1;
  const arrayElements     = dv.getUint32(KTX_FIELDS.NUMBER_OF_ARRAY_ELEMENTS,  littleEndian) || 1;
  const numberOfFaces     = dv.getUint32(KTX_FIELDS.NUMBER_OF_FACES,    littleEndian);
  const mipmapLevels      = dv.getUint32(KTX_FIELDS.NUMBER_OF_MIPMAP_LEVELS,   littleEndian);
  const bytesKeyValue     = dv.getUint32(KTX_FIELDS.BYTES_OF_KEY_VALUE_DATA,   littleEndian);

  if (pixelDepth !== 1)   throw new Error('[CELL-COMPRESSED-TEX] KTX: only 2D textures supported');
  if (numberOfFaces !== 1) throw new Error('[CELL-COMPRESSED-TEX] KTX: cube textures not supported');
  if (arrayElements !== 1) throw new Error('[CELL-COMPRESSED-TEX] KTX: array textures not supported');

  const textureFormat = GL_INTERNAL_FORMAT_TO_TEXTURE[glInternalFormat];
  if (!textureFormat) {
    throw new Error(`[CELL-COMPRESSED-TEX] KTX: unknown glInternalFormat 0x${glInternalFormat.toString(16)}`);
  }

  // Compute BPP for mip size calculation
  let bpp = GL_INTERNAL_FORMAT_BPP[glInternalFormat];
  if (bpp === undefined) {
    // Uncompressed path (GL type present)
    bpp = 4; // assume RGBA8
  }

  const dataOffset = KTX_FILE_HEADER_SIZE + bytesKeyValue;
  const alignedW   = (pixelWidth + 3) & ~3;
  const alignedH   = (pixelHeight + 3) & ~3;

  const levels = Math.max(1, mipmapLevels);
  const imageBuffers: Uint8Array[] = [];

  let imageOffset = dataOffset;
  let mipW = pixelWidth;
  let mipH = pixelHeight;

  for (let level = 0; level < levels; level++) {
    const imageSize   = dv.getUint32(imageOffset, littleEndian);
    const elementOff  = imageOffset + 4;

    // Compressed: align dimensions to block boundary (4×4)
    const aw = (mipW + 3) & ~3;
    const ah = (mipH + 3) & ~3;
    const mipBytes = glType === 0
      ? Math.round(aw * ah * bpp)
      : mipW * mipH * bpp;

    imageBuffers.push(new Uint8Array(buf, elementOff, Math.min(mipBytes, imageSize)));

    // Advance offset — pad to 4-byte boundary after imageSize field
    imageOffset += imageSize + 4;
    if (imageOffset % 4 !== 0) imageOffset += 4 - (imageOffset % 4);

    mipW = Math.max(mipW >> 1, 1);
    mipH = Math.max(mipH >> 1, 1);
  }

  return {
    format:    textureFormat,
    width:     pixelWidth,
    height:    pixelHeight,
    resource:  imageBuffers,
    alphaMode: 'no-premultiply-alpha',
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// § 4. DDS binary parser (BC7 / BC3 / BC1 texture container)
// ═══════════════════════════════════════════════════════════════════════════════

const DDS_MAGIC           = 0x20534444; // 'DDS '
const DDS_FOURCC_DXT1     = 0x31545844; // 'DXT1'
const DDS_FOURCC_DXT3     = 0x33545844; // 'DXT3'
const DDS_FOURCC_DXT5     = 0x35545844; // 'DXT5'
const DDS_FOURCC_DX10     = 0x30315844; // 'DX10'
const DWORD               = 4;          // bytes per DWORD

const DDS_HEADER_FIELDS = {
  MAGIC:        0,  HEIGHT: 3, WIDTH: 4,
  MIPMAP_COUNT: 7,  PF_FOURCC: 21,
} as const;

// DXGI format subset → TEXTURE_FORMATS
// (mirrors upstream/pixijs-engine/src/compressed-textures/dds/const.ts)
const DXGI_FORMAT_MAP: Record<number, TEXTURE_FORMATS> = {
  71:  'bc1-rgba-unorm',        // DXGI_FORMAT_BC1_UNORM
  72:  'bc1-rgba-unorm-srgb',   // DXGI_FORMAT_BC1_UNORM_SRGB
  74:  'bc2-rgba-unorm',        // DXGI_FORMAT_BC2_UNORM
  75:  'bc2-rgba-unorm-srgb',
  77:  'bc3-rgba-unorm',        // DXGI_FORMAT_BC3_UNORM
  78:  'bc3-rgba-unorm-srgb',
  80:  'bc4-r-unorm',           // DXGI_FORMAT_BC4_UNORM
  81:  'bc4-r-snorm',
  83:  'bc5-rg-unorm',          // DXGI_FORMAT_BC5_UNORM
  84:  'bc5-rg-snorm',
  95:  'bc6h-rgb-ufloat',       // DXGI_FORMAT_BC6H_UF16
  96:  'bc6h-rgb-float',        // DXGI_FORMAT_BC6H_SF16
  98:  'bc7-rgba-unorm',        // DXGI_FORMAT_BC7_UNORM
  99:  'bc7-rgba-unorm-srgb',   // DXGI_FORMAT_BC7_UNORM_SRGB
};

const FOURCC_FORMAT_MAP: Record<number, TEXTURE_FORMATS> = {
  [DDS_FOURCC_DXT1]: 'bc1-rgba-unorm',
  [DDS_FOURCC_DXT3]: 'bc2-rgba-unorm',
  [DDS_FOURCC_DXT5]: 'bc3-rgba-unorm',
};

/**
 * Bytes per 4×4 block for DDS formats.
 * Mirrors upstream TEXTURE_FORMAT_BLOCK_SIZE.
 */
const DDS_BLOCK_BYTES: Partial<Record<TEXTURE_FORMATS, number>> = {
  'bc1-rgba-unorm':     8,  'bc1-rgba-unorm-srgb': 8,
  'bc2-rgba-unorm':    16,  'bc2-rgba-unorm-srgb': 16,
  'bc3-rgba-unorm':    16,  'bc3-rgba-unorm-srgb': 16,
  'bc4-r-unorm':        8,  'bc4-r-snorm':          8,
  'bc5-rg-unorm':      16,  'bc5-rg-snorm':         16,
  'bc6h-rgb-ufloat':   16,  'bc6h-rgb-float':       16,
  'bc7-rgba-unorm':    16,  'bc7-rgba-unorm-srgb':  16,
};

/**
 * Parse a DDS ArrayBuffer into TextureSourceOptions (BC7 / BC6H / BC3 / BC1).
 *
 * Fuses upstream/pixijs-engine/src/compressed-textures/dds/parseDDS.ts logic.
 *
 * @param buf  Raw DDS file bytes
 * @throws if the DDS magic is wrong or the format is unsupported
 */
export function parseDDSBuffer(buf: ArrayBuffer): TextureSourceOptions<Uint8Array[]> {
  const dv = new DataView(buf);

  if (dv.getUint32(0, true) !== DDS_MAGIC) {
    throw new Error('[CELL-COMPRESSED-TEX] Invalid DDS magic number');
  }

  const height     = dv.getUint32(DDS_HEADER_FIELDS.HEIGHT       * DWORD, true);
  const width      = dv.getUint32(DDS_HEADER_FIELDS.WIDTH        * DWORD, true);
  const mipmapCnt  = Math.max(1, dv.getUint32(DDS_HEADER_FIELDS.MIPMAP_COUNT * DWORD, true));
  const fourCC     = dv.getUint32(DDS_HEADER_FIELDS.PF_FOURCC    * DWORD, true);

  let format: TEXTURE_FORMATS;
  let dataOffset: number;

  if (fourCC === DDS_FOURCC_DX10) {
    // DX10 extended header: 4 DWORDs header + 20 bytes (DDS_HEADER_DX10) padding
    const dxgiFormat = dv.getUint32((31 + 0) * DWORD, true); // DX10 header starts at DWORD 31
    const mapped     = DXGI_FORMAT_MAP[dxgiFormat];
    if (!mapped) throw new Error(`[CELL-COMPRESSED-TEX] DDS DX10: unsupported DXGI format ${dxgiFormat}`);
    format     = mapped;
    dataOffset = (31 + 5) * DWORD;   // 31 DWORDs DDS header + 5 DWORDs DX10 header
  } else {
    const mapped = FOURCC_FORMAT_MAP[fourCC];
    if (!mapped) throw new Error(`[CELL-COMPRESSED-TEX] DDS: unsupported FourCC 0x${fourCC.toString(16)}`);
    format     = mapped;
    dataOffset = 128;                 // standard DDS header = 128 bytes
  }

  const blockBytes = DDS_BLOCK_BYTES[format] ?? 16;
  const levels: Uint8Array[] = [];
  let mipW = width;
  let mipH = height;
  let offset = dataOffset;

  for (let i = 0; i < mipmapCnt; i++) {
    const aw       = Math.ceil(Math.max(4, mipW) / 4) * 4;
    const ah       = Math.ceil(Math.max(4, mipH) / 4) * 4;
    const byteLen  = (aw / 4) * (ah / 4) * blockBytes;

    levels.push(new Uint8Array(buf, offset, byteLen));
    offset += byteLen;
    mipW    = Math.max(mipW >> 1, 1);
    mipH    = Math.max(mipH >> 1, 1);
  }

  return {
    format,
    width,
    height,
    resource:  levels,
    alphaMode: 'no-premultiply-alpha',
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// § 5. Synthetic RGBA8 → pseudo-compressed (software fallback block encoder)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Compute aligned block count for a given dimension.
 * @internal
 */
function blockCount(dim: number, blockSize: number): number {
  return Math.ceil(Math.max(dim, blockSize) / blockSize);
}

/**
 * Trivial ETC2 RGB8 block encoder (placeholder quality — for GPU upload shape).
 *
 * A proper encoder would use error-diffusion or exhaustive search over
 * ETC2 differential/T/H/planar modes.  This version packs the 4×4 block's
 * average colour into a degenerate "individual" mode block — good enough for
 * topology colour tiles where exact pixel fidelity is not required.
 *
 * Block format (64 bits):
 *   [63:48] colour1 R4G4B4+flip+diff bits
 *   [47:32] colour2 R4G4B4+codeword
 *   [31:0]  pixel indices (all 0 = base colour)
 *
 * @internal
 */
function encodeETC2RGB8Block(rgba: Uint8Array, srcOff: number, srcW: number): Uint8Array {
  // Gather 4×4 texel block — compute average RGB
  let rSum = 0, gSum = 0, bSum = 0;
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      const px = srcOff + (row * srcW + col) * 4;
      rSum += rgba[px] ?? 0;
      gSum += rgba[px + 1] ?? 0;
      bSum += rgba[px + 2] ?? 0;
    }
  }
  const r4 = Math.round(rSum / 16 / 17) & 0xF;
  const g4 = Math.round(gSum / 16 / 17) & 0xF;
  const b4 = Math.round(bSum / 16 / 17) & 0xF;

  // ETC1-compatible "individual" mode: diffBit=0, each sub-block same colour
  const word0 = (r4 << 28) | (r4 << 24) | (g4 << 20) | (g4 << 16) | (b4 << 12) | (b4 << 8) | 0x00;
  const word1 = 0x00000000; // pixel indices = 0 (all use base colour, modifier 0)

  const block = new Uint8Array(8);
  const dv    = new DataView(block.buffer);
  dv.setUint32(0, word0, false); // big-endian per ETC spec
  dv.setUint32(4, word1, false);
  return block;
}

/**
 * Trivial ASTC 4×4 block encoder (identity / solid colour placeholder).
 *
 * ASTC blocks are always 128 bits.  The void-extent encoding (mode=0x1FE)
 * efficiently represents a solid colour — perfectly valid and minimal for
 * single-colour topology icon tiles.
 *
 * @internal
 */
function encodeASTC4x4Block(rgba: Uint8Array, srcOff: number, srcW: number): Uint8Array {
  // Average RGBA of 4×4 block
  let r = 0, g = 0, b = 0, a = 0;
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      const px = srcOff + (row * srcW + col) * 4;
      r += rgba[px] ?? 0;
      g += rgba[px + 1] ?? 0;
      b += rgba[px + 2] ?? 0;
      a += rgba[px + 3] ?? 255;
    }
  }
  r = Math.round(r / 16);
  g = Math.round(g / 16);
  b = Math.round(b / 16);
  a = Math.round(a / 16);

  // Void-extent block: bits[0..8]=0xFC (void-extent 2D LDR), extent coords all-1s,
  // then 64-bit endpoint pair.
  const block = new Uint8Array(16);
  // Block mode bits [9:0] = 0b1111100 (void-extent, LDR)
  block[0] = 0xFC; block[1] = 0xFF;
  block[2] = 0xFF; block[3] = 0xFF;
  block[4] = 0xFF; block[5] = 0xFF;
  // Endpoint pair: e0=(r,g,b,a), e1=(r,g,b,a) — stored as FP16 pairs
  // For LDR UNORM: value = round(c * 65535 / 255)
  const dv = new DataView(block.buffer);
  const r16 = Math.round(r * 65535 / 255);
  const g16 = Math.round(g * 65535 / 255);
  const b16 = Math.round(b * 65535 / 255);
  const a16 = Math.round(a * 65535 / 255);
  dv.setUint16(8,  r16, true);
  dv.setUint16(10, g16, true);
  dv.setUint16(12, b16, true);
  dv.setUint16(14, a16, true);
  return block;
}

// ═══════════════════════════════════════════════════════════════════════════════
// § 6. Memory estimation
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Estimate GPU memory for a texture at the given format.
 * Returns bytes (all mip levels included via ×4/3 factor).
 */
export function estimateTextureMemory(
  width: number,
  height: number,
  format: TEXTURE_FORMATS | 'rgba8',
): number {
  const mipFactor = 4 / 3; // geometric series sum over all mip levels

  if (format === 'rgba8') {
    return Math.ceil(width * height * 4 * mipFactor);
  }

  const blockDim  = COMPRESSED_BLOCK_DIM[format] ?? { w: 4, h: 4 };
  const blockSize = COMPRESSED_BLOCK_BYTES[format] ?? 16;
  const bw = blockCount(width, blockDim.w);
  const bh = blockCount(height, blockDim.h);
  return Math.ceil(bw * bh * blockSize * mipFactor);
}

/**
 * Calculate memory saving ratio when using compressed vs RGBA8 unorm.
 *
 * @returns Object with `savedBytes`, `savedPercent` (0–100), and `ratio` (RGBA8/compressed).
 */
export function estimateMemorySaving(
  width: number,
  height: number,
  format: TEXTURE_FORMATS,
): { savedBytes: number; savedPercent: number; ratio: number } {
  const rgba8Bytes   = estimateTextureMemory(width, height, 'rgba8');
  const compBytes    = estimateTextureMemory(width, height, format);
  const savedBytes   = rgba8Bytes - compBytes;
  const savedPercent = Math.round((savedBytes / rgba8Bytes) * 100);
  const ratio        = rgba8Bytes / Math.max(compBytes, 1);

  return { savedBytes, savedPercent, ratio };
}

// ═══════════════════════════════════════════════════════════════════════════════
// § 7. TopologyTexAtlas — lightweight bin-packer for cell icon tiles
// ═══════════════════════════════════════════════════════════════════════════════

/** Placement of one cell's icon region in the atlas. */
export interface AtlasTile {
  cellId: string;
  x: number;
  y: number;
  w: number;
  h: number;
  u0: number;
  v0: number;
  u1: number;
  v1: number;
}

/** Output of TopologyTexAtlas.pack(). */
export interface TopologyAtlasLayout {
  /** Atlas width (next power-of-two ≥ packed width) */
  atlasWidth:  number;
  /** Atlas height (next power-of-two ≥ packed height) */
  atlasHeight: number;
  /** Per-cell tile placements */
  tiles: AtlasTile[];
}

/**
 * TopologyTexAtlas — row-first shelf bin-packer.
 *
 * Assigns each cell a square tile slot in a texture atlas, packing
 * left-to-right / top-to-bottom.  Atlas dimensions are padded to the
 * nearest power-of-two to satisfy ASTC/ETC2 block alignment.
 *
 * This is intentionally simple — no tree-based optimal packing —
 * because topology textures are uniform-sized icon tiles.
 */
export class TopologyTexAtlas {
  private readonly _tileSize: number;
  private readonly _maxCols: number;

  /**
   * @param tileSize  Each cell occupies a `tileSize × tileSize` slot (default 64 px)
   * @param maxCols   Maximum columns before wrapping to next row (default 16)
   */
  constructor(tileSize = 64, maxCols = 16) {
    this._tileSize = tileSize;
    this._maxCols  = maxCols;
  }

  /**
   * Pack a list of cell descriptors into an atlas layout.
   */
  pack(cells: Pick<CellDescriptor, 'cell_id'>[]): TopologyAtlasLayout {
    const t    = this._tileSize;
    const cols = Math.min(cells.length, this._maxCols);
    const rows = Math.ceil(cells.length / cols);

    const rawW = cols * t;
    const rawH = rows * t;

    const atlasWidth  = nextPowerOfTwo(rawW);
    const atlasHeight = nextPowerOfTwo(rawH);

    const tiles: AtlasTile[] = cells.map((cell, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const x   = col * t;
      const y   = row * t;
      return {
        cellId: cell.cell_id,
        x, y, w: t, h: t,
        u0: x / atlasWidth,
        v0: y / atlasHeight,
        u1: (x + t) / atlasWidth,
        v1: (y + t) / atlasHeight,
      };
    });

    return { atlasWidth, atlasHeight, tiles };
  }
}

function nextPowerOfTwo(n: number): number {
  if (n <= 1) return 1;
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

// ═══════════════════════════════════════════════════════════════════════════════
// § 8. createTopologyTex — generate a CompressedSource from cell RGBA pixels
// ═══════════════════════════════════════════════════════════════════════════════

/** Options for createTopologyTex(). */
export interface TopologyTexOptions {
  /**
   * Pre-rendered RGBA8 pixels for the full atlas.
   * Must be width × height × 4 bytes, row-major.
   */
  rgba:   Uint8Array;
  width:  number;
  height: number;
  /**
   * Target compressed format.  If null, returns an uncompressed RGBA8 source.
   */
  format: TEXTURE_FORMATS | null;
  /**
   * Number of mip levels to generate (default: 1 — no mipmaps).
   * Additional levels are block-encoded at halved resolution.
   */
  mipLevels?: number;
}

/**
 * Create a CompressedSource from raw RGBA8 pixels using software block encoding.
 *
 * For production, the topology pipeline should supply pre-encoded .ktx / .dds
 * files via parseKTXBuffer() / parseDDSBuffer().  This function handles the
 * dynamic / runtime case where the topology is generated on-the-fly and we do
 * not have a pre-compressed file.
 *
 * The software encoder produces valid ASTC void-extent or ETC2 individual-mode
 * blocks — GPU upload will succeed, quality matches a solid-colour per 4×4 tile.
 *
 * For BC formats (no software encoder), falls back to RGBA8 uncompressed.
 *
 * @returns CompressedSource ready to be assigned as TextureSource on a PixiJS Texture.
 */
export function createTopologyTex(opts: TopologyTexOptions): CompressedSource {
  const { rgba, width, height, format, mipLevels = 1 } = opts;

  if (!format || format === 'rgba8unorm') {
    // Uncompressed RGBA8 path
    const src = new CompressedSource({
      format:    'rgba8unorm',
      width,
      height,
      resource:  [rgba.slice()],
      alphaMode: 'no-premultiply-alpha',
    });
    return src;
  }

  const blockDim  = COMPRESSED_BLOCK_DIM[format];
  const blockSize = COMPRESSED_BLOCK_BYTES[format];

  // Fall back to RGBA8 for formats without a software encoder
  if (!blockDim || !blockSize) {
    console.warn(`[CELL-COMPRESSED-TEX] No software encoder for ${format}; using RGBA8 fallback`);
    return new CompressedSource({
      format:    'rgba8unorm',
      width,
      height,
      resource:  [rgba.slice()],
      alphaMode: 'no-premultiply-alpha',
    });
  }

  const isAstc = format.startsWith('astc');
  const isEtc2 = format.startsWith('etc2');

  if (!isAstc && !isEtc2) {
    // BC formats require GPU or offline encoder — fallback to RGBA8
    console.warn(`[CELL-COMPRESSED-TEX] Software BC encoding unsupported for ${format}; using RGBA8`);
    return new CompressedSource({
      format:    'rgba8unorm',
      width,
      height,
      resource:  [rgba.slice()],
      alphaMode: 'no-premultiply-alpha',
    });
  }

  const levels: Uint8Array[] = [];

  let mipW    = width;
  let mipH    = height;
  let mipRGBA = rgba;

  for (let level = 0; level < mipLevels; level++) {
    const bw      = blockCount(mipW, blockDim.w);
    const bh      = blockCount(mipH, blockDim.h);
    const encoded = new Uint8Array(bw * bh * blockSize);

    let blockIdx = 0;
    for (let by = 0; by < bh; by++) {
      for (let bx = 0; bx < bw; bx++) {
        const srcOff = (by * blockDim.h * mipW + bx * blockDim.w);
        const block  = isAstc
          ? encodeASTC4x4Block(mipRGBA, srcOff, mipW)
          : encodeETC2RGB8Block(mipRGBA, srcOff, mipW);

        encoded.set(block, blockIdx * blockSize);
        blockIdx++;
      }
    }
    levels.push(encoded);

    // Generate next mip level (box filter)
    if (level < mipLevels - 1) {
      const nextW  = Math.max(mipW >> 1, 1);
      const nextH  = Math.max(mipH >> 1, 1);
      mipRGBA      = _boxFilterRGBA(mipRGBA, mipW, mipH, nextW, nextH);
      mipW         = nextW;
      mipH         = nextH;
    }
  }

  return new CompressedSource({
    format,
    width,
    height,
    resource:  levels,
    alphaMode: 'no-premultiply-alpha',
  });
}

/** Simple 2×2 box filter for mip generation. @internal */
function _boxFilterRGBA(
  src: Uint8Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): Uint8Array {
  const dst = new Uint8Array(dstW * dstH * 4);
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const sx = x * 2, sy = y * 2;
      let r = 0, g = 0, b = 0, a = 0, n = 0;
      for (let dy = 0; dy <= 1; dy++) {
        for (let dx = 0; dx <= 1; dx++) {
          const nx = Math.min(sx + dx, srcW - 1);
          const ny = Math.min(sy + dy, srcH - 1);
          const p  = (ny * srcW + nx) * 4;
          r += src[p];
          g += src[p + 1];
          b += src[p + 2];
          a += src[p + 3];
          n++;
        }
      }
      const p = (y * dstW + x) * 4;
      dst[p]   = Math.round(r / n);
      dst[p+1] = Math.round(g / n);
      dst[p+2] = Math.round(b / n);
      dst[p+3] = Math.round(a / n);
    }
  }
  return dst;
}

// ═══════════════════════════════════════════════════════════════════════════════
// § 9. CompressedTexManager — lifecycle + renderer integration
// ═══════════════════════════════════════════════════════════════════════════════

/** Per-atlas allocation record. */
export interface AtlasRecord {
  source:  CompressedSource;
  texture: Texture;
  layout:  TopologyAtlasLayout;
  format:  TEXTURE_FORMATS;
  width:   number;
  height:  number;
  /** Estimated GPU bytes across all mip levels */
  gpuBytes: number;
}

/**
 * CompressedTexManager — lifecycle manager for topology compressed atlases.
 *
 * Integrates with PixiJS Renderer — the renderer's TextureSystem handles
 * actual GPU upload via glUploadCompressedTextureResource when the
 * CompressedSource is first used in a draw call.
 *
 * Usage pattern (mirrors upstream loadKTX / loadDDS):
 *   1. probe = CompressedTexProbe.get()
 *   2. mgr   = new CompressedTexManager(app.renderer)
 *   3. atlas = await mgr.allocateTopologyAtlas(cells, probe.best)
 *   4. sprite.texture = atlas.texture  → GPU upload on next render
 *   5. mgr.destroy()                   → frees all allocations
 */
export class CompressedTexManager {
  private readonly _renderer: Renderer | null;
  private readonly _atlases: Map<string, AtlasRecord> = new Map();
  private readonly _packer: TopologyTexAtlas;

  /** Total GPU memory estimate across all managed atlases. */
  gpuBytesTotal = 0;

  constructor(renderer: Renderer | null = null, tileSize = 64) {
    this._renderer = renderer;
    this._packer   = new TopologyTexAtlas(tileSize);
  }

  /**
   * Allocate a compressed texture atlas for the given cell list.
   *
   * When `format` is null (no compression available), the atlas uses RGBA8.
   *
   * @param cells        Cell descriptors whose icons will occupy the atlas
   * @param format       Compressed format from CompressedTexProbe (or null)
   * @param rgbaFactory  Optional: supply RGBA8 pixel data for the atlas.
   *                     If omitted, a solid grey placeholder is generated.
   */
  allocateTopologyAtlas(
    cells: CellDescriptor[],
    format: TEXTURE_FORMATS | null,
    rgbaFactory?: (layout: TopologyAtlasLayout) => Uint8Array,
  ): AtlasRecord {
    const id     = `atlas-${cells.map(c => c.cell_id).join('-').slice(0, 64)}`;
    const cached = this._atlases.get(id);
    if (cached) return cached;

    const layout   = this._packer.pack(cells);
    const { atlasWidth: w, atlasHeight: h } = layout;

    // Generate or receive RGBA8 pixel data
    const rgba = rgbaFactory
      ? rgbaFactory(layout)
      : _placeholderRGBA(w, h, cells);

    const source = createTopologyTex({ rgba, width: w, height: h, format, mipLevels: 1 });
    const texture = new Texture({ source });

    const effectiveFormat = format ?? 'rgba8unorm' as TEXTURE_FORMATS;
    const gpuBytes = estimateTextureMemory(w, h, effectiveFormat as TEXTURE_FORMATS);

    const record: AtlasRecord = { source, texture, layout, format: effectiveFormat as TEXTURE_FORMATS, width: w, height: h, gpuBytes };
    this._atlases.set(id, record);
    this.gpuBytesTotal += gpuBytes;

    const saving = format ? estimateMemorySaving(w, h, format) : null;
    console.debug(
      `[CELL-COMPRESSED-TEX] atlas ${id}: ${w}×${h} format=${effectiveFormat}` +
      (saving ? ` savings=${saving.savedPercent}% (${(saving.savedBytes/1024).toFixed(1)}KB)` : ''),
    );

    return record;
  }

  /**
   * Load a pre-compressed texture from a .ktx URL.
   * Returns a PixiJS Texture backed by CompressedSource (GPU upload on demand).
   */
  async loadKTX(url: string): Promise<Texture> {
    const resp = await fetch(url);
    const buf  = await resp.arrayBuffer();
    const opts = parseKTXBuffer(buf);
    const src  = new CompressedSource({ ...opts, resolution: 1 });
    return new Texture({ source: src });
  }

  /**
   * Load a pre-compressed texture from a .dds URL.
   */
  async loadDDS(url: string): Promise<Texture> {
    const resp = await fetch(url);
    const buf  = await resp.arrayBuffer();
    const opts = parseDDSBuffer(buf);
    const src  = new CompressedSource({ ...opts, resolution: 1 });
    return new Texture({ source: src });
  }

  /**
   * Destroy all managed atlases and release GPU resources via renderer.
   */
  destroy(): void {
    for (const record of this._atlases.values()) {
      record.texture.destroy(true);
    }
    this._atlases.clear();
    this.gpuBytesTotal = 0;
  }

  /** Number of allocated atlases. */
  get atlasCount(): number {
    return this._atlases.size;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// § 10. Convenience factory
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Build a compressed texture pipeline for a large topology graph.
 *
 * Convenience wrapper that:
 *   1. Probes compressed texture support
 *   2. Selects the best available format
 *   3. Allocates an atlas for the cell list
 *   4. Returns the atlas record with memory saving stats
 *
 * @example
 * ```ts
 * const result = buildCompressedTopology(app.renderer, cells);
 * if (result.atlas) {
 *   sprite.texture = result.atlas.texture;
 *   console.log(`GPU saving: ${result.saving?.savedPercent}%`);
 * }
 * ```
 */
export function buildCompressedTopology(
  renderer: Renderer | null,
  cells: CellDescriptor[],
  opts: {
    /** Force a specific format (skips probe) */
    forceFormat?: TEXTURE_FORMATS;
    /** Use sRGB variants (default false) */
    srgb?: boolean;
    /** Tile size in pixels (default 64) */
    tileSize?: number;
    /** RGBA pixel factory (see CompressedTexManager.allocateTopologyAtlas) */
    rgbaFactory?: (layout: TopologyAtlasLayout) => Uint8Array;
  } = {},
): {
  atlas:    AtlasRecord | null;
  caps:     CompressedTexCapabilities;
  format:   TEXTURE_FORMATS | null;
  saving:   ReturnType<typeof estimateMemorySaving> | null;
  skipped:  boolean;
} {
  const caps = CompressedTexProbe.get();

  if (cells.length < LARGE_TOPOLOGY_THRESHOLD && !opts.forceFormat) {
    return { atlas: null, caps, format: null, saving: null, skipped: true };
  }

  const format = opts.forceFormat ?? selectBestFormat(caps, opts.srgb ?? false);
  const mgr    = new CompressedTexManager(renderer, opts.tileSize);
  const atlas  = mgr.allocateTopologyAtlas(cells, format, opts.rgbaFactory);
  const saving = format ? estimateMemorySaving(atlas.width, atlas.height, format) : null;

  return { atlas, caps, format, saving, skipped: false };
}

// ═══════════════════════════════════════════════════════════════════════════════
// § 11. KTX2 compressed texture loading — loadTexture(src, useCompressed)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * VkFormat → PixiJS TEXTURE_FORMATS mapping for KTX2 containers.
 *
 * ktx-parse exposes vkFormat as a number; we map the compressed (and common
 * uncompressed) VK enum values to the TEXTURE_FORMATS strings that PixiJS 8's
 * CompressedSource accepts.
 *
 * Source of truth:
 *   - ktx-parse/dist/constants.d.ts   (VK_FORMAT_* values)
 *   - upstream/pixijs-engine/src/compressed-textures/ktx2/utils/vkFormatToGPUFormat.ts
 *   - upstream/pixijs-engine/src/compressed-textures/ktx2/utils/glFormatToGPUFormat.ts
 *
 * @internal
 */
const VK_FORMAT_TO_TEXTURE: Record<number, TEXTURE_FORMATS> = {
  // ── Uncompressed ──────────────────────────────────────────────────────────
  23:  'rgb8unorm',            // VK_FORMAT_R8G8B8_UNORM
  37:  'rgba8unorm',           // VK_FORMAT_R8G8B8A8_UNORM
  43:  'rgba8unorm-srgb',      // VK_FORMAT_R8G8B8A8_SRGB

  // ── BC (S3TC / BPTC) ─────────────────────────────────────────────────────
  131: 'bc1-rgba-unorm',       // VK_FORMAT_BC1_RGB_UNORM_BLOCK
  132: 'bc1-rgba-unorm-srgb',  // VK_FORMAT_BC1_RGB_SRGB_BLOCK
  133: 'bc1-rgba-unorm',       // VK_FORMAT_BC1_RGBA_UNORM_BLOCK
  134: 'bc1-rgba-unorm-srgb',  // VK_FORMAT_BC1_RGBA_SRGB_BLOCK
  135: 'bc2-rgba-unorm',       // VK_FORMAT_BC2_UNORM_BLOCK
  136: 'bc2-rgba-unorm-srgb',  // VK_FORMAT_BC2_SRGB_BLOCK
  137: 'bc3-rgba-unorm',       // VK_FORMAT_BC3_UNORM_BLOCK
  138: 'bc3-rgba-unorm-srgb',  // VK_FORMAT_BC3_SRGB_BLOCK
  139: 'bc4-r-unorm',          // VK_FORMAT_BC4_UNORM_BLOCK
  140: 'bc4-r-snorm',          // VK_FORMAT_BC4_SNORM_BLOCK
  141: 'bc5-rg-unorm',         // VK_FORMAT_BC5_UNORM_BLOCK
  142: 'bc5-rg-snorm',         // VK_FORMAT_BC5_SNORM_BLOCK
  143: 'bc6h-rgb-ufloat',      // VK_FORMAT_BC6H_UFLOAT_BLOCK
  144: 'bc6h-rgb-float',       // VK_FORMAT_BC6H_SFLOAT_BLOCK
  145: 'bc7-rgba-unorm',       // VK_FORMAT_BC7_UNORM_BLOCK
  146: 'bc7-rgba-unorm-srgb',  // VK_FORMAT_BC7_SRGB_BLOCK

  // ── ETC2 ──────────────────────────────────────────────────────────────────
  147: 'etc2-rgb8unorm',       // VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK
  148: 'etc2-rgb8unorm-srgb',  // VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK
  149: 'etc2-rgb8a1unorm',     // VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK
  150: 'etc2-rgb8a1unorm-srgb',// VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK
  151: 'etc2-rgba8unorm',      // VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK
  152: 'etc2-rgba8unorm-srgb', // VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK

  // ── EAC ───────────────────────────────────────────────────────────────────
  153: 'eac-r11unorm',         // VK_FORMAT_EAC_R11_UNORM_BLOCK
  155: 'eac-rg11unorm',        // VK_FORMAT_EAC_R11G11_UNORM_BLOCK

  // ── ASTC ──────────────────────────────────────────────────────────────────
  157: 'astc-4x4-unorm',       // VK_FORMAT_ASTC_4x4_UNORM_BLOCK
  158: 'astc-4x4-unorm-srgb',  // VK_FORMAT_ASTC_4x4_SRGB_BLOCK
  159: 'astc-5x4-unorm',       // VK_FORMAT_ASTC_5x4_UNORM_BLOCK
  160: 'astc-5x4-unorm-srgb',  // VK_FORMAT_ASTC_5x4_SRGB_BLOCK
  161: 'astc-5x5-unorm',       // VK_FORMAT_ASTC_5x5_UNORM_BLOCK
  162: 'astc-5x5-unorm-srgb',  // VK_FORMAT_ASTC_5x5_SRGB_BLOCK
  163: 'astc-6x5-unorm',       // VK_FORMAT_ASTC_6x5_UNORM_BLOCK
  164: 'astc-6x5-unorm-srgb',  // VK_FORMAT_ASTC_6x5_SRGB_BLOCK
  165: 'astc-6x6-unorm',       // VK_FORMAT_ASTC_6x6_UNORM_BLOCK
  166: 'astc-6x6-unorm-srgb',  // VK_FORMAT_ASTC_6x6_SRGB_BLOCK
  167: 'astc-8x5-unorm',       // VK_FORMAT_ASTC_8x5_UNORM_BLOCK
  168: 'astc-8x5-unorm-srgb',  // VK_FORMAT_ASTC_8x5_SRGB_BLOCK
  169: 'astc-8x6-unorm',       // VK_FORMAT_ASTC_8x6_UNORM_BLOCK
  170: 'astc-8x6-unorm-srgb',  // VK_FORMAT_ASTC_8x6_SRGB_BLOCK
  171: 'astc-8x8-unorm',       // VK_FORMAT_ASTC_8x8_UNORM_BLOCK
  172: 'astc-8x8-unorm-srgb',  // VK_FORMAT_ASTC_8x8_SRGB_BLOCK
  173: 'astc-10x5-unorm',      // VK_FORMAT_ASTC_10x5_UNORM_BLOCK
  174: 'astc-10x5-unorm-srgb', // VK_FORMAT_ASTC_10x5_SRGB_BLOCK
  175: 'astc-10x6-unorm',      // VK_FORMAT_ASTC_10x6_UNORM_BLOCK
  176: 'astc-10x6-unorm-srgb', // VK_FORMAT_ASTC_10x6_SRGB_BLOCK
  177: 'astc-10x8-unorm',      // VK_FORMAT_ASTC_10x8_UNORM_BLOCK
  178: 'astc-10x8-unorm-srgb', // VK_FORMAT_ASTC_10x8_SRGB_BLOCK
  179: 'astc-10x10-unorm',     // VK_FORMAT_ASTC_10x10_UNORM_BLOCK
  180: 'astc-10x10-unorm-srgb',// VK_FORMAT_ASTC_10x10_SRGB_BLOCK
  181: 'astc-12x10-unorm',     // VK_FORMAT_ASTC_12x10_UNORM_BLOCK
  182: 'astc-12x10-unorm-srgb',// VK_FORMAT_ASTC_12x10_SRGB_BLOCK
  183: 'astc-12x12-unorm',     // VK_FORMAT_ASTC_12x12_UNORM_BLOCK
  184: 'astc-12x12-unorm-srgb',// VK_FORMAT_ASTC_12x12_SRGB_BLOCK
};

/**
 * Parse a KTX2 ArrayBuffer (using ktx-parse) into TextureSourceOptions
 * compatible with PixiJS 8 CompressedSource.
 *
 * Handles non-supercompressed KTX2 containers (vkFormat ≠ VK_FORMAT_UNDEFINED).
 * For Basis-supercompressed KTX2 files (ETC1S / UASTC with BasisLZ or Zstd),
 * the upstream PixiJS loadKTX2 worker + libktx transcoder must be used instead.
 *
 * @param buf  Raw KTX2 file bytes
 * @throws if the format is unsupported or the container uses supercompression
 * @internal
 */
export function parseKTX2Buffer(buf: ArrayBuffer): TextureSourceOptions<Uint8Array[]> {
  const container: KTX2Container = readKTX2(new Uint8Array(buf));

  // Basis-supercompressed containers need libktx transcoder — not handled here
  if (container.vkFormat === 0) {
    throw new Error(
      '[CELL-COMPRESSED-TEX] KTX2: Basis-supercompressed (vkFormat=0) containers require ' +
      'the upstream PixiJS loadKTX2 worker + libktx transcoder. Use loadTexture() which ' +
      'will fall back to the image path, or supply a non-supercompressed .ktx2 file.',
    );
  }

  const format = VK_FORMAT_TO_TEXTURE[container.vkFormat];
  if (!format) {
    throw new Error(
      `[CELL-COMPRESSED-TEX] KTX2: unsupported vkFormat ${container.vkFormat}`,
    );
  }

  const levels: Uint8Array[] = container.levels.map(level => level.levelData);

  return {
    format,
    width:     container.pixelWidth,
    height:    container.pixelHeight || 1,
    resource:  levels,
    alphaMode: 'no-premultiply-alpha',
  };
}

/**
 * Load a texture from a URL, optionally using KTX2 compressed texture format.
 *
 * This is the primary entry point for M311 KTX2 compressed texture support.
 *
 * Behaviour:
 *   - `useCompressed = true` AND `src` ends with `.ktx2`:
 *       Fetches the KTX2 file, parses via ktx-parse, creates a CompressedSource,
 *       and returns a GPU-uploadable Texture.
 *   - `useCompressed = true` AND `src` does NOT end with `.ktx2`:
 *       Probes for a `.ktx2` sibling file by replacing the extension.
 *       If the probe fetch succeeds (HTTP 200), loads as KTX2.
 *       Otherwise falls back to standard image loading.
 *   - `useCompressed = false`:
 *       Always loads via standard Image → TextureSource path (uncompressed RGBA8).
 *
 * For Basis-supercompressed KTX2 files (vkFormat === 0), the function catches
 * the parse error and falls back to standard image loading with a console warning,
 * since those files require the libktx WASM transcoder that this standalone
 * function does not bundle.
 *
 * @param src            URL of the texture (image or .ktx2 file)
 * @param useCompressed  When true, attempt KTX2 compressed loading
 * @returns              PixiJS Texture ready for rendering
 *
 * @example
 * ```ts
 * // Load a KTX2 compressed texture directly
 * const tex = await loadTexture('/assets/cell-atlas.ktx2', true);
 *
 * // Load with auto-detection: probes for .ktx2 sibling of a .png
 * const tex2 = await loadTexture('/assets/cell-atlas.png', true);
 *
 * // Force uncompressed loading
 * const tex3 = await loadTexture('/assets/cell-atlas.png', false);
 * ```
 */
export async function loadTexture(
  src: string,
  useCompressed: boolean,
): Promise<Texture> {
  // ── KTX2 compressed path ──────────────────────────────────────────────────
  if (useCompressed) {
    let ktx2Url: string;
    let isProbed = false;

    if (src.toLowerCase().endsWith('.ktx2')) {
      ktx2Url = src;
    } else {
      // Probe for a .ktx2 sibling: /assets/foo.png → /assets/foo.ktx2
      const dotIdx = src.lastIndexOf('.');
      ktx2Url = (dotIdx >= 0 ? src.slice(0, dotIdx) : src) + '.ktx2';
      isProbed = true;
    }

    try {
      const resp = await fetch(ktx2Url);
      if (!resp.ok) {
        if (isProbed) {
          console.debug(
            `[CELL-COMPRESSED-TEX] KTX2 probe miss (${resp.status}): ${ktx2Url} → image fallback`,
          );
          return _loadImageTexture(src);
        }
        throw new Error(`[CELL-COMPRESSED-TEX] KTX2 fetch failed: ${resp.status} ${ktx2Url}`);
      }

      const buf     = await resp.arrayBuffer();
      const opts    = parseKTX2Buffer(buf);
      const source  = new CompressedSource({ ...opts, resolution: 1 });
      const texture = new Texture({ source });

      console.debug(
        `[CELL-COMPRESSED-TEX] KTX2 loaded: ${ktx2Url} ` +
        `${opts.width}×${opts.height} format=${String(opts.format)} ` +
        `levels=${opts.resource?.length ?? 1}`,
      );

      return texture;
    } catch (err) {
      // Basis-supercompressed or other parse failure → fall back
      if (isProbed) {
        console.debug(
          `[CELL-COMPRESSED-TEX] KTX2 probe parse error → image fallback:`,
          err,
        );
        return _loadImageTexture(src);
      }
      // Non-probed direct .ktx2 URL — caller explicitly asked for KTX2,
      // still fall back gracefully with a warning
      console.warn(
        `[CELL-COMPRESSED-TEX] KTX2 load failed, falling back to image:`,
        err,
      );
      // Try to load the image extension variant
      const dotIdx  = src.lastIndexOf('.');
      const imgUrl  = (dotIdx >= 0 ? src.slice(0, dotIdx) : src) + '.png';
      return _loadImageTexture(imgUrl);
    }
  }

  // ── Standard image path ───────────────────────────────────────────────────
  return _loadImageTexture(src);
}

/**
 * Load a standard (uncompressed) image as a PixiJS Texture.
 *
 * Creates an HTMLImageElement, waits for load, wraps in TextureSource → Texture.
 * Works in any browser; the GPU receives RGBA8 uncompressed data.
 *
 * @param url  Image URL (.png, .jpg, .webp, etc.)
 * @internal
 */
async function _loadImageTexture(url: string): Promise<Texture> {
  // SSR guard
  if (typeof Image === 'undefined') {
    console.warn('[CELL-COMPRESSED-TEX] Image not available (SSR), returning empty texture');
    return Texture.EMPTY;
  }

  return new Promise<Texture>((resolve, reject) => {
    const img     = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      const source  = new TextureSource({ resource: img, resolution: 1 });
      const texture = new Texture({ source });
      console.debug(`[CELL-COMPRESSED-TEX] Image loaded: ${url} ${img.width}×${img.height}`);
      resolve(texture);
    };

    img.onerror = (_event) => {
      reject(new Error(`[CELL-COMPRESSED-TEX] Image load failed: ${url}`));
    };

    img.src = url;
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// § 12. Internal helpers
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Generate a placeholder RGBA8 atlas where each cell tile is filled with
 * a species-keyed pastel colour.
 * @internal
 */
function _placeholderRGBA(
  w: number,
  h: number,
  cells: CellDescriptor[],
): Uint8Array {
  const rgba = new Uint8Array(w * h * 4);

  // Species colour map (matches SPECIES_PALETTE hue families)
  const speciesColour: Record<string, [number, number, number]> = {
    'cil-eye':         [92, 107, 192],
    'cil-vector':      [102, 187, 106],
    'cil-bolt':        [255, 167, 38],
    'cil-plus':        [236, 64, 122],
    'cil-arrow-right': [120, 144, 156],
    'cil-filter':      [171, 71, 188],
    'cil-code':        [38, 166, 154],
    'cil-layers':      [66, 165, 245],
    'cil-loop':        [255, 202, 40],
    'cil-graph':       [120, 144, 156],
  };

  // Fill entire atlas with #1a1a2e background
  for (let i = 0; i < w * h; i++) {
    rgba[i * 4]     = 26;
    rgba[i * 4 + 1] = 26;
    rgba[i * 4 + 2] = 46;
    rgba[i * 4 + 3] = 255;
  }

  // Bake per-cell tile with species colour
  const tileSize = Math.floor(Math.min(w, h) / Math.min(cells.length, 16));
  const cols     = Math.max(1, Math.floor(w / Math.max(1, tileSize)));

  cells.forEach((cell, i) => {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const tx  = col * tileSize;
    const ty  = row * tileSize;
    const [r, g, b] = speciesColour[cell.species] ?? [128, 128, 128];

    for (let dy = 2; dy < tileSize - 2; dy++) {
      for (let dx = 2; dx < tileSize - 2; dx++) {
        const px   = ((ty + dy) * w + tx + dx) * 4;
        rgba[px]   = r;
        rgba[px+1] = g;
        rgba[px+2] = b;
        rgba[px+3] = 255;
      }
    }
  });

  return rgba;
}
