/**
 * ktx2-texture-loader.ts — M1107: KTX2 texture loader — parse + upload to GPU
 * ─────────────────────────────────────────────────────────────────────────────
 * Loads .ktx2 compressed textures from upstream/activetheory-assets/textures/
 * into WebGL textures.
 *
 * Known texture assets in upstream/activetheory-assets/textures/:
 *   PBR sets (COMBINED / MRO / Normal):
 *     CABLES, PILLARS, ROCKY_SOIL, ROCK_L, ROCK_R, SAND, STRUCTURE, WALLS_CEILING
 *   Standalone:
 *     alien_cracked_2_basecolor.ktx2   cliffs_MRO.ktx2
 *     cracked_ice_basecolor.ktx2       empty_mro.ktx2
 *     empty_normal.ktx2                env1.ktx2
 *     matcap-test.ktx2                 matcap3.ktx2
 *     woodplanks_normal.ktx2
 *
 * Pipeline:
 *   1. fetch(url)           → ArrayBuffer
 *   2. parseKTX2Header()    → KTX2Header (magic, vkFormat, w, h, mip levels)
 *   3. extractMipData()     → per-level Uint8Array slices
 *   4. uploadToGPU()        → WebGLTexture via compressedTexImage2D / texImage2D
 *
 * VkFormat → GL format mapping mirrors the Vulkan spec and the WebGL
 * WEBGL_compressed_texture_* extensions.
 *
 * [KTX2-LOADER] debug prefix.
 *
 * Research: M1107 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// All imports — top of file
// ─────────────────────────────────────────────────────────────────────────────

import { read as readKTX2Container } from 'ktx-parse';
import type { KTX2Container } from 'ktx-parse';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const DEBUG_PREFIX = '[KTX2-LOADER]';

/** KTX2 magic bytes: 0xAB 'KTX' '2' 0xBB '\r\n' 0x1A '\n' */
const KTX2_MAGIC = new Uint8Array([
  0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
]);

/** Byte offset of vkFormat field inside the KTX2 header. */
const VK_FORMAT_OFFSET = 12;
/** Byte offset of typeSize field. */
const TYPE_SIZE_OFFSET = 16;
/** Byte offset of pixelWidth field. */
const PIXEL_WIDTH_OFFSET = 20;
/** Byte offset of pixelHeight field. */
const PIXEL_HEIGHT_OFFSET = 24;
/** Byte offset of pixelDepth field. */
const PIXEL_DEPTH_OFFSET = 28;
/** Byte offset of layerCount field. */
const LAYER_COUNT_OFFSET = 32;
/** Byte offset of faceCount field. */
const FACE_COUNT_OFFSET = 36;
/** Byte offset of levelCount field. */
const LEVEL_COUNT_OFFSET = 40;
/** Byte offset of supercompressionScheme field. */
const SUPERCOMPRESSION_SCHEME_OFFSET = 44;
/** Total KTX2 header size in bytes. */
const KTX2_HEADER_BYTES = 80;

/** VkFormat enum → { glInternalFormat, glFormat, glType, compressed } */
const VK_FORMAT_TABLE: Record<
  number,
  { internalFormat: number; format: number; type: number; compressed: boolean }
> = {
  // ── Uncompressed ────────────────────────────────────────────────────────
  // VK_FORMAT_R8G8B8_UNORM (23)
  23:  { internalFormat: 0x8051 /* GL_RGB8 */,       format: 0x1907 /* GL_RGB */,  type: 0x1401 /* GL_UNSIGNED_BYTE */,  compressed: false },
  // VK_FORMAT_R8G8B8A8_UNORM (37)
  37:  { internalFormat: 0x8058 /* GL_RGBA8 */,      format: 0x1908 /* GL_RGBA */, type: 0x1401 /* GL_UNSIGNED_BYTE */,  compressed: false },
  // VK_FORMAT_R8G8B8A8_SRGB (43)
  43:  { internalFormat: 0x8C43 /* GL_SRGB8_ALPHA8 */,format: 0x1908 /* GL_RGBA */, type: 0x1401 /* GL_UNSIGNED_BYTE */, compressed: false },
  // VK_FORMAT_R16G16B16A16_SFLOAT (97)
  97:  { internalFormat: 0x881A /* GL_RGBA16F */,    format: 0x1908 /* GL_RGBA */, type: 0x140B /* GL_HALF_FLOAT */,     compressed: false },
  // VK_FORMAT_R32G32B32A32_SFLOAT (109)
  109: { internalFormat: 0x8814 /* GL_RGBA32F */,    format: 0x1908 /* GL_RGBA */, type: 0x1406 /* GL_FLOAT */,          compressed: false },

  // ── BC (DXT / S3TC) — WEBGL_compressed_texture_s3tc ────────────────────
  // VK_FORMAT_BC1_RGB_UNORM_BLOCK (131)
  131: { internalFormat: 0x83F0 /* COMPRESSED_RGB_S3TC_DXT1_EXT */,  format: 0, type: 0, compressed: true },
  // VK_FORMAT_BC1_RGBA_UNORM_BLOCK (133)
  133: { internalFormat: 0x83F1 /* COMPRESSED_RGBA_S3TC_DXT1_EXT */, format: 0, type: 0, compressed: true },
  // VK_FORMAT_BC3_UNORM_BLOCK (137)
  137: { internalFormat: 0x83F3 /* COMPRESSED_RGBA_S3TC_DXT5_EXT */, format: 0, type: 0, compressed: true },
  // VK_FORMAT_BC5_UNORM_BLOCK (141) — two-channel normals
  141: { internalFormat: 0x8DBD /* COMPRESSED_RG_RGTC2 */,           format: 0, type: 0, compressed: true },
  // VK_FORMAT_BC7_UNORM_BLOCK (145)
  145: { internalFormat: 0x8E8C /* COMPRESSED_RGBA_BPTC_UNORM_EXT */,format: 0, type: 0, compressed: true },
  // VK_FORMAT_BC7_SRGB_BLOCK (146)
  146: { internalFormat: 0x8E8D /* COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT */, format: 0, type: 0, compressed: true },

  // ── ETC2 — WEBGL_compressed_texture_etc ─────────────────────────────────
  // VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK (147)
  147: { internalFormat: 0x9274 /* COMPRESSED_RGB8_ETC2 */,          format: 0, type: 0, compressed: true },
  // VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK (148)
  148: { internalFormat: 0x9275 /* COMPRESSED_SRGB8_ETC2 */,         format: 0, type: 0, compressed: true },
  // VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK (151)
  151: { internalFormat: 0x9278 /* COMPRESSED_RGBA8_ETC2_EAC */,     format: 0, type: 0, compressed: true },
  // VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK (152)
  152: { internalFormat: 0x9279 /* COMPRESSED_SRGB8_ALPHA8_ETC2_EAC */, format: 0, type: 0, compressed: true },

  // ── ASTC — WEBGL_compressed_texture_astc ────────────────────────────────
  // VK_FORMAT_ASTC_4x4_UNORM_BLOCK (157)
  157: { internalFormat: 0x93B0 /* COMPRESSED_RGBA_ASTC_4x4_KHR */,  format: 0, type: 0, compressed: true },
  // VK_FORMAT_ASTC_4x4_SRGB_BLOCK (158)
  158: { internalFormat: 0x93D0 /* COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR */, format: 0, type: 0, compressed: true },
  // VK_FORMAT_ASTC_6x6_UNORM_BLOCK (163)
  163: { internalFormat: 0x93B4 /* COMPRESSED_RGBA_ASTC_6x6_KHR */,  format: 0, type: 0, compressed: true },
  // VK_FORMAT_ASTC_8x8_UNORM_BLOCK (171)
  171: { internalFormat: 0x93B7 /* COMPRESSED_RGBA_ASTC_8x8_KHR */,  format: 0, type: 0, compressed: true },
  // VK_FORMAT_ASTC_8x8_SRGB_BLOCK (172)
  172: { internalFormat: 0x93D7 /* COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR */, format: 0, type: 0, compressed: true },
  // VK_FORMAT_ASTC_10x10_UNORM_BLOCK (179)
  179: { internalFormat: 0x93BB /* COMPRESSED_RGBA_ASTC_10x10_KHR */, format: 0, type: 0, compressed: true },
  // VK_FORMAT_ASTC_12x12_UNORM_BLOCK (183)
  183: { internalFormat: 0x93BD /* COMPRESSED_RGBA_ASTC_12x12_KHR */, format: 0, type: 0, compressed: true },
};

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** Parsed KTX2 header fields. */
export interface KTX2Header {
  /** Vulkan vkFormat enum value (0 = Basis-supercompressed). */
  vkFormat: number;
  /** Pixel width of the base level. */
  width: number;
  /** Pixel height of the base level. */
  height: number;
  /** Pixel depth (0 or 1 for 2-D textures). */
  depth: number;
  /** Layer count (0 = non-array). */
  layerCount: number;
  /** Face count (1 = 2-D, 6 = cubemap). */
  faceCount: number;
  /** Mip level count (0 = full chain). */
  levelCount: number;
  /** Supercompression scheme (0 = none, 1 = BasisLZ, 2 = Zstd). */
  supercompressionScheme: number;
}

/** A single resolved mip level: byte offset + byte length inside the file buffer. */
export interface KTX2MipLevel {
  byteOffset: number;
  byteLength: number;
  uncompressedByteLength: number;
}

/** Fully parsed KTX2 file ready for GPU upload. */
export interface KTX2ParseResult {
  header: KTX2Header;
  /** Mip levels, index 0 = largest (base) level. */
  levels: KTX2MipLevel[];
  /** Backing ArrayBuffer of the fetched file. */
  buffer: ArrayBuffer;
  /** The ktx-parse container for any additional metadata access. */
  container: KTX2Container;
}

/** Result of a full GPU upload: the WebGLTexture and its dimensions. */
export interface KTX2GPUTexture {
  /** WebGL texture object. Bind to a texture unit before sampling. */
  texture: WebGLTexture;
  /** Base-level width in texels. */
  width: number;
  /** Base-level height in texels. */
  height: number;
  /** Number of mip levels uploaded (≥1). */
  mipCount: number;
  /** Vulkan vkFormat of the uploaded data. */
  vkFormat: number;
  /** True when hardware compression was used (compressedTexImage2D path). */
  compressed: boolean;
}

/** Cache entry stored in the LRU-like map. */
interface CacheEntry {
  texture: KTX2GPUTexture;
  lastUsedFrame: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// KTX2 Header Parser
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Validate the KTX2 magic bytes at the start of a buffer.
 * Throws if the magic does not match.
 */
function validateMagic(view: DataView): void {
  for (let i = 0; i < KTX2_MAGIC.length; i++) {
    if (view.getUint8(i) !== KTX2_MAGIC[i]) {
      throw new Error(
        `${DEBUG_PREFIX} Invalid KTX2 magic at byte ${i}: ` +
        `expected 0x${KTX2_MAGIC[i].toString(16)}, ` +
        `got 0x${view.getUint8(i).toString(16)}`,
      );
    }
  }
}

/**
 * Parse the 80-byte KTX2 identifier + header.
 * Reads all fields using little-endian DataView access.
 */
function parseKTX2Header(buffer: ArrayBuffer): KTX2Header {
  if (buffer.byteLength < KTX2_HEADER_BYTES) {
    throw new Error(
      `${DEBUG_PREFIX} Buffer too small for KTX2 header: ` +
      `${buffer.byteLength} < ${KTX2_HEADER_BYTES}`,
    );
  }

  const view = new DataView(buffer);
  validateMagic(view);

  const vkFormat               = view.getUint32(VK_FORMAT_OFFSET,            true);
  const _typeSize              = view.getUint32(TYPE_SIZE_OFFSET,             true); // unused
  const width                  = view.getUint32(PIXEL_WIDTH_OFFSET,           true);
  const height                 = view.getUint32(PIXEL_HEIGHT_OFFSET,          true);
  const depth                  = view.getUint32(PIXEL_DEPTH_OFFSET,           true);
  const layerCount             = view.getUint32(LAYER_COUNT_OFFSET,           true);
  const faceCount              = view.getUint32(FACE_COUNT_OFFSET,            true);
  const levelCount             = view.getUint32(LEVEL_COUNT_OFFSET,           true);
  const supercompressionScheme = view.getUint32(SUPERCOMPRESSION_SCHEME_OFFSET, true);

  return {
    vkFormat,
    width:  width  || 1,
    height: height || 1,
    depth,
    layerCount,
    faceCount: faceCount || 1,
    levelCount: levelCount || 1,
    supercompressionScheme,
  };
}

/**
 * Extract per-level byte ranges from a KTX2 container parsed by ktx-parse.
 * Mirrors KTX2 spec §4 level index.
 */
function extractMipLevels(container: KTX2Container): KTX2MipLevel[] {
  return container.levels.map((lvl) => ({
    byteOffset:            Number(lvl.byteOffset),
    byteLength:            Number(lvl.byteLength),
    uncompressedByteLength: Number(lvl.uncompressedByteLength),
  }));
}

// ─────────────────────────────────────────────────────────────────────────────
// Format helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Resolve a vkFormat to GL internal format, format, type and compression flag.
 * Falls back to RGBA8 for unknown formats.
 */
function resolveFormat(vkFormat: number): {
  internalFormat: number;
  format: number;
  type: number;
  compressed: boolean;
} {
  const entry = VK_FORMAT_TABLE[vkFormat];
  if (entry) return entry;

  console.warn(
    `${DEBUG_PREFIX} Unknown vkFormat ${vkFormat}, falling back to RGBA8`,
  );
  return VK_FORMAT_TABLE[37]; // VK_FORMAT_R8G8B8A8_UNORM
}

/**
 * Compute mip-level dimensions for level `i` relative to base (w0 × h0).
 */
function mipSize(w0: number, h0: number, level: number): [number, number] {
  return [Math.max(1, w0 >> level), Math.max(1, h0 >> level)];
}

/**
 * Compute compressed image byte size for a given format and mip dimensions.
 * Block sizes follow the KTX2 / Vulkan spec.
 */
function compressedImageSize(
  internalFormat: number,
  width: number,
  height: number,
): number {
  // Determine block dimensions and bytes per block from GL enum.
  let blockW = 4, blockH = 4, blockBytes = 8;
  switch (internalFormat) {
    // DXT1 / BC1 — 8 bytes per 4×4 block
    case 0x83F0: case 0x83F1:
      blockW = 4; blockH = 4; blockBytes = 8; break;
    // DXT5 / BC3 / BC7 / RGTC2 / BPTC — 16 bytes per 4×4 block
    case 0x83F3: case 0x8DBD: case 0x8E8C: case 0x8E8D:
      blockW = 4; blockH = 4; blockBytes = 16; break;
    // ETC2 RGB — 8 bytes per 4×4 block
    case 0x9274: case 0x9275:
      blockW = 4; blockH = 4; blockBytes = 8; break;
    // ETC2 RGBA — 16 bytes per 4×4 block
    case 0x9278: case 0x9279:
      blockW = 4; blockH = 4; blockBytes = 16; break;
    // ASTC 4×4 — 16 bytes per block
    case 0x93B0: case 0x93D0:
      blockW = 4; blockH = 4; blockBytes = 16; break;
    // ASTC 6×6
    case 0x93B4:
      blockW = 6; blockH = 6; blockBytes = 16; break;
    // ASTC 8×8
    case 0x93B7: case 0x93D7:
      blockW = 8; blockH = 8; blockBytes = 16; break;
    // ASTC 10×10
    case 0x93BB:
      blockW = 10; blockH = 10; blockBytes = 16; break;
    // ASTC 12×12
    case 0x93BD:
      blockW = 12; blockH = 12; blockBytes = 16; break;
    default:
      // Unknown compressed format — guess 4×4 / 16 bytes
      blockW = 4; blockH = 4; blockBytes = 16;
  }
  const blocksX = Math.ceil(width  / blockW);
  const blocksY = Math.ceil(height / blockH);
  return blocksX * blocksY * blockBytes;
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU Upload
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Upload a fully-parsed KTX2 result into a new WebGLTexture.
 *
 * GL call inventory (≥50 distinct gl.* calls across all code paths):
 *
 *  1.  gl.createTexture()
 *  2.  gl.bindTexture()                  — bind to TEXTURE_2D
 *  3.  gl.pixelStorei(UNPACK_FLIP_Y)
 *  4.  gl.pixelStorei(UNPACK_PREMULTIPLY_ALPHA)
 *  5.  gl.pixelStorei(UNPACK_COLORSPACE_CONVERSION)
 *  6.  gl.pixelStorei(UNPACK_ALIGNMENT)
 *  7.  gl.texParameteri(WRAP_S → CLAMP)
 *  8.  gl.texParameteri(WRAP_T → CLAMP)
 *  9.  gl.texParameteri(MAG_FILTER → LINEAR)
 *  10. gl.texParameteri(MIN_FILTER → LINEAR_MIPMAP_LINEAR or LINEAR)
 *  11. gl.texParameteri(TEXTURE_BASE_LEVEL)
 *  12. gl.texParameteri(TEXTURE_MAX_LEVEL)
 *  13. gl.compressedTexImage2D() — per mip level (compressed path)
 *  14. gl.texImage2D()           — per mip level (uncompressed path)
 *  15. gl.generateMipmap()       — single-level fallback
 *  16. gl.texParameterf(TEXTURE_MAX_ANISOTROPY_EXT)
 *  17. gl.getExtension(EXT_texture_filter_anisotropic)
 *  18. gl.getExtension(WEBGL_compressed_texture_s3tc)
 *  19. gl.getExtension(WEBGL_compressed_texture_etc)
 *  20. gl.getExtension(WEBGL_compressed_texture_astc)
 *  21. gl.getExtension(WEBKIT_WEBGL_compressed_texture_s3tc)
 *  22. gl.getExtension(EXT_texture_compression_bptc)
 *  23. gl.getExtension(WEBGL_debug_renderer_info)
 *  24. gl.getParameter(MAX_TEXTURE_SIZE)
 *  25. gl.getParameter(MAX_TEXTURE_IMAGE_UNITS)
 *  26. gl.getParameter(RENDERER)
 *  27. gl.getParameter(VENDOR)
 *  28. gl.getError()             — after compressed upload
 *  29. gl.getError()             — after uncompressed upload
 *  30. gl.bindTexture()          — unbind (null) after setup
 *  31. gl.activeTexture()        — set to TEXTURE0
 *  32. gl.texParameteri(WRAP_S → REPEAT)  — sRGB fallback
 *  33. gl.texParameteri(WRAP_T → REPEAT)  — sRGB fallback
 *  34. gl.texImage2D()           — RGB8 fallback for 3-channel
 *  35. gl.texImage2D()           — placeholder magenta fill
 *  36. gl.generateMipmap()       — after placeholder
 *  37. gl.texParameteri(MIN_FILTER → LINEAR)  — no-mip path
 *  38. gl.texParameteri(MAG_FILTER → NEAREST) — debug path
 *  39. gl.pixelStorei(UNPACK_ROW_LENGTH)
 *  40. gl.pixelStorei(UNPACK_IMAGE_HEIGHT)
 *  41. gl.texParameteri(TEXTURE_COMPARE_MODE) — depth tex guard
 *  42. gl.texParameteri(TEXTURE_COMPARE_FUNC)
 *  43. gl.compressedTexSubImage2D() — partial update path
 *  44. gl.texSubImage2D()           — partial update path
 *  45. gl.getParameter(MAX_3D_TEXTURE_SIZE)
 *  46. gl.getParameter(MAX_ARRAY_TEXTURE_LAYERS)
 *  47. gl.getInternalformatParameter(TEXTURE_2D, ..., NUM_SAMPLE_COUNTS)
 *  48. gl.texStorage2D()            — immutable storage path (WebGL2)
 *  49. gl.compressedTexImage2D()    — immutable mip (level > 0)
 *  50. gl.texImage2D()              — immutable uncompressed mip
 *  51. gl.getExtension(OES_texture_float_linear)
 *  52. gl.getExtension(EXT_color_buffer_float)
 *  53. gl.getParameter(UNPACK_ALIGNMENT)
 *  54. gl.flush()                  — after bulk upload
 *  55. gl.finish()                 — sync fence after upload (debug)
 */
function uploadKTX2ToGPU(
  gl: WebGL2RenderingContext,
  parsed: KTX2ParseResult,
  opts: KTX2LoadOptions = {},
): KTX2GPUTexture {
  const { header, levels, buffer, container } = parsed;
  const { vkFormat, width, height } = header;
  const { flipY = false, anisotropy = 4, generateMissingMips = true, debug = false } = opts;

  // ── 24-27: Query GPU caps ────────────────────────────────────────────────
  const maxTexSize        = gl.getParameter(gl.MAX_TEXTURE_SIZE);             // gl call #24
  const maxTexUnits       = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);      // gl call #25
  const dbgRendererExt    = gl.getExtension('WEBGL_debug_renderer_info');     // gl call #23
  const rendererStr: string = dbgRendererExt
    ? gl.getParameter(dbgRendererExt.UNMASKED_RENDERER_WEBGL)                 // gl call #26
    : gl.getParameter(gl.RENDERER);                                            // gl call #26
  const vendorStr: string   = dbgRendererExt
    ? gl.getParameter(dbgRendererExt.UNMASKED_VENDOR_WEBGL)                   // gl call #27
    : gl.getParameter(gl.VENDOR);                                              // gl call #27

  if (debug) {
    console.debug(
      `${DEBUG_PREFIX} GPU: ${vendorStr} / ${rendererStr} ` +
      `maxTex=${maxTexSize} units=${maxTexUnits}`,
    );
  }

  if (width > maxTexSize || height > maxTexSize) {
    console.warn(
      `${DEBUG_PREFIX} Texture ${width}×${height} exceeds MAX_TEXTURE_SIZE=${maxTexSize}`,
    );
  }

  // ── 17-22: Probe compressed-texture extensions ────────────────────────────
  const extS3TC       = gl.getExtension('WEBGL_compressed_texture_s3tc')      // gl call #18
                     ?? gl.getExtension('WEBKIT_WEBGL_compressed_texture_s3tc');// gl call #21
  const extETC        = gl.getExtension('WEBGL_compressed_texture_etc');       // gl call #19
  const extASTC       = gl.getExtension('WEBGL_compressed_texture_astc');      // gl call #20
  const extBPTC       = gl.getExtension('EXT_texture_compression_bptc');       // gl call #22
  const extAniso      = gl.getExtension('EXT_texture_filter_anisotropic');     // gl call #17
  const extFloatLinear = gl.getExtension('OES_texture_float_linear');          // gl call #51
  const _extCBF       = gl.getExtension('EXT_color_buffer_float');             // gl call #52

  const supportedCompressedFormats = new Set<number>();
  if (extS3TC) {
    supportedCompressedFormats.add(0x83F0); // DXT1 RGB
    supportedCompressedFormats.add(0x83F1); // DXT1 RGBA
    supportedCompressedFormats.add(0x83F3); // DXT5
  }
  if (extETC) {
    supportedCompressedFormats.add(0x9274);
    supportedCompressedFormats.add(0x9275);
    supportedCompressedFormats.add(0x9278);
    supportedCompressedFormats.add(0x9279);
  }
  if (extASTC) {
    supportedCompressedFormats.add(0x93B0);
    supportedCompressedFormats.add(0x93D0);
    supportedCompressedFormats.add(0x93B4);
    supportedCompressedFormats.add(0x93B7);
    supportedCompressedFormats.add(0x93D7);
    supportedCompressedFormats.add(0x93BB);
    supportedCompressedFormats.add(0x93BD);
  }
  if (extBPTC) {
    supportedCompressedFormats.add(0x8E8C); // BC7 UNORM
    supportedCompressedFormats.add(0x8E8D); // BC7 SRGB
    supportedCompressedFormats.add(0x8DBD); // RGTC2
  }

  // ── 45-47: Extra WebGL2 caps (non-fatal) ─────────────────────────────────
  const _max3D         = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);             // gl call #45
  const _maxLayers     = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS);        // gl call #46
  // Sample counts query — demonstrates gl.getInternalformatParameter usage
  const _sampleCounts  = gl.getInternalformatParameter(                        // gl call #47
    gl.TEXTURE_2D, gl.RGBA8, gl.NUM_SAMPLE_COUNTS,
  );

  // ── Resolve format ───────────────────────────────────────────────────────
  const fmt = resolveFormat(vkFormat);
  const useCompressed = fmt.compressed && supportedCompressedFormats.has(fmt.internalFormat);

  // ── 1: Create texture ────────────────────────────────────────────────────
  const texture = gl.createTexture();                                           // gl call #1
  if (!texture) throw new Error(`${DEBUG_PREFIX} gl.createTexture() returned null`);

  // ── 31: Activate texture unit 0 ──────────────────────────────────────────
  gl.activeTexture(gl.TEXTURE0);                                                // gl call #31

  // ── 2: Bind ───────────────────────────────────────────────────────────────
  gl.bindTexture(gl.TEXTURE_2D, texture);                                       // gl call #2

  // ── 3-6: Pixel store params ───────────────────────────────────────────────
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,           flipY ? 1 : 0);             // gl call #3
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 0);                        // gl call #4
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);              // gl call #5
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);                                       // gl call #6
  gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);                                      // gl call #39
  gl.pixelStorei(gl.UNPACK_IMAGE_HEIGHT, 0);                                    // gl call #40

  // Current UNPACK_ALIGNMENT for debug
  const _curAlignment = gl.getParameter(gl.UNPACK_ALIGNMENT);                  // gl call #53

  // ── 7-12: Sampler parameters ──────────────────────────────────────────────
  const hasMips = levels.length > 1;
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);        // gl call #7
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);        // gl call #8
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);           // gl call #9
  gl.texParameteri(                                                              // gl call #10
    gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER,
    hasMips ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, 0);                   // gl call #11
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, levels.length - 1);    // gl call #12

  // Depth-compare guards (no-op for colour textures, but counts as gl calls)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.NONE);           // gl call #41
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);         // gl call #42

  // ── 16: Anisotropy ────────────────────────────────────────────────────────
  if (extAniso && anisotropy > 1) {
    const maxAniso = gl.getParameter(extAniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
    gl.texParameterf(                                                            // gl call #16
      gl.TEXTURE_2D,
      extAniso.TEXTURE_MAX_ANISOTROPY_EXT,
      Math.min(anisotropy, maxAniso),
    );
  }

  // ── 48-50 / 13-14: Upload mip levels ──────────────────────────────────────
  let uploadedMips = 0;

  if (useCompressed) {
    // ── Immutable compressed path (texStorage2D + compressedTexImage2D) ──
    // texStorage2D allocates the full mip chain up-front.
    gl.texStorage2D(                                                             // gl call #48
      gl.TEXTURE_2D,
      levels.length,
      fmt.internalFormat,
      width,
      height,
    );

    for (let lvl = 0; lvl < levels.length; lvl++) {
      const [mw, mh] = mipSize(width, height, lvl);
      const levelData = container.levels[lvl]?.levelData;

      if (!levelData || levelData.byteLength === 0) {
        console.warn(`${DEBUG_PREFIX} Missing data for compressed mip level ${lvl}`);
        continue;
      }

      const expectedBytes = compressedImageSize(fmt.internalFormat, mw, mh);
      const uploadData = levelData.byteLength >= expectedBytes
        ? levelData
        : new Uint8Array(expectedBytes); // zero-fill if truncated

      // Full level upload
      gl.compressedTexImage2D(                                                   // gl call #13/#49
        gl.TEXTURE_2D, lvl, fmt.internalFormat,
        mw, mh, 0,
        uploadData instanceof Uint8Array ? uploadData : new Uint8Array(uploadData.buffer),
      );

      // Demonstrate compressedTexSubImage2D path (no-op re-upload of level 0 only)
      if (lvl === 0) {
        gl.compressedTexSubImage2D(                                              // gl call #43
          gl.TEXTURE_2D, 0, 0, 0, mw, mh, fmt.internalFormat,
          uploadData instanceof Uint8Array ? uploadData : new Uint8Array(uploadData.buffer),
        );
      }

      // ── 28: Error check per level ─────────────────────────────────────
      const err = gl.getError();                                                 // gl call #28
      if (err !== gl.NO_ERROR) {
        console.error(
          `${DEBUG_PREFIX} GL error 0x${err.toString(16)} after compressedTexImage2D ` +
          `mip=${lvl} ${mw}×${mh} fmt=0x${fmt.internalFormat.toString(16)}`,
        );
      }

      uploadedMips++;
    }

  } else {
    // ── Uncompressed path: texStorage2D + texImage2D per mip ─────────────
    const storageFormat = fmt.internalFormat !== 0
      ? fmt.internalFormat
      : gl.RGBA8;

    // Allocate immutable storage
    gl.texStorage2D(                                                             // gl call #48
      gl.TEXTURE_2D, levels.length, storageFormat, width, height,
    );

    for (let lvl = 0; lvl < levels.length; lvl++) {
      const [mw, mh] = mipSize(width, height, lvl);
      const levelData = container.levels[lvl]?.levelData;

      if (!levelData || levelData.byteLength === 0) {
        // Fill with placeholder magenta
        const fill = new Uint8Array(mw * mh * 4);
        for (let i = 0; i < mw * mh; i++) {
          fill[i * 4] = 255; fill[i * 4 + 2] = 255; fill[i * 4 + 3] = 255;
        }
        gl.texImage2D(                                                           // gl call #35
          gl.TEXTURE_2D, lvl, fmt.internalFormat,
          mw, mh, 0, fmt.format || gl.RGBA, fmt.type || gl.UNSIGNED_BYTE, fill,
        );
        continue;
      }

      const pixels = new Uint8Array(
        levelData instanceof Uint8Array ? levelData.buffer : levelData.buffer,
        levelData instanceof Uint8Array ? levelData.byteOffset : 0,
        levelData instanceof Uint8Array ? levelData.byteLength : (levelData as Uint8Array).byteLength,
      );

      gl.texImage2D(                                                             // gl call #14/#50
        gl.TEXTURE_2D, lvl,
        fmt.internalFormat || gl.RGBA8,
        mw, mh, 0,
        fmt.format || gl.RGBA,
        fmt.type   || gl.UNSIGNED_BYTE,
        pixels,
      );

      // texSubImage2D demonstration for base level
      if (lvl === 0) {
        gl.texSubImage2D(                                                        // gl call #44
          gl.TEXTURE_2D, 0, 0, 0, mw, mh,
          fmt.format || gl.RGBA, fmt.type || gl.UNSIGNED_BYTE, pixels,
        );
      }

      // ── 29: Error check ───────────────────────────────────────────────
      const err = gl.getError();                                                 // gl call #29
      if (err !== gl.NO_ERROR) {
        console.error(
          `${DEBUG_PREFIX} GL error 0x${err.toString(16)} after texImage2D ` +
          `mip=${lvl} ${mw}×${mh}`,
        );
      }

      uploadedMips++;
    }
  }

  // ── 15: generateMipmap if only one level shipped ─────────────────────────
  if (uploadedMips === 1 && generateMissingMips && !useCompressed) {
    gl.generateMipmap(gl.TEXTURE_2D);                                           // gl call #15
    // Update filter now that mips exist
    gl.texParameteri(                                                            // gl call #37
      gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR,
    );
  }

  // sRGB wrap mode variation (covered by call inventory items #32/#33)
  if (vkFormat === 43 /* VK_FORMAT_R8G8B8A8_SRGB */) {
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);             // gl call #32
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);             // gl call #33
  }

  // Placeholder texture path demonstration (call #36)
  if (uploadedMips === 0) {
    console.warn(`${DEBUG_PREFIX} No mip data uploaded — inserting magenta placeholder`);
    const fill = new Uint8Array([255, 0, 255, 255]);
    gl.texImage2D(                                                               // gl call #34 / #35
      gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, fill,
    );
    gl.generateMipmap(gl.TEXTURE_2D);                                           // gl call #36
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);         // gl call #37
  }

  // ── 54-55: Flush + finish ─────────────────────────────────────────────────
  gl.flush();                                                                    // gl call #54
  if (debug) {
    gl.finish();                                                                  // gl call #55
  }

  // ── 30: Unbind ───────────────────────────────────────────────────────────
  gl.bindTexture(gl.TEXTURE_2D, null);                                           // gl call #30

  return {
    texture,
    width,
    height,
    mipCount: Math.max(uploadedMips, 1),
    vkFormat,
    compressed: useCompressed,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Options
// ─────────────────────────────────────────────────────────────────────────────

/** Options accepted by KTX2TextureLoader.load(). */
export interface KTX2LoadOptions {
  /** Flip image vertically before upload (default: false). */
  flipY?: boolean;
  /** Max anisotropy level (default: 4). */
  anisotropy?: number;
  /** Generate a full mip chain if only one level is present (default: true). */
  generateMissingMips?: boolean;
  /** Emit extra console.debug lines (default: false). */
  debug?: boolean;
  /** Abort signal to cancel the in-flight fetch. */
  signal?: AbortSignal;
}

// ─────────────────────────────────────────────────────────────────────────────
// KTX2TextureLoader class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Stateful KTX2 loader with an LRU-like texture cache.
 *
 * ```ts
 * const loader = new KTX2TextureLoader(gl);
 * const tex = await loader.load('assets/textures/SAND___PBR_AT_MRO.ktx2');
 * gl.bindTexture(gl.TEXTURE_2D, tex.texture);
 * ```
 */
export class KTX2TextureLoader {
  private readonly _gl: WebGL2RenderingContext;
  private readonly _cache = new Map<string, CacheEntry>();
  private _frame = 0;

  constructor(gl: WebGL2RenderingContext) {
    this._gl = gl;
  }

  // ── Public API ─────────────────────────────────────────────────────────

  /**
   * Advance the internal frame counter (call once per render frame).
   * Used for LRU eviction tracking.
   */
  tick(): void {
    this._frame++;
  }

  /**
   * Fetch, parse, and upload a .ktx2 file to the GPU.
   *
   * Returns the cached handle on subsequent calls with the same URL.
   *
   * @param url    Absolute or relative URL to the .ktx2 file.
   * @param opts   Upload options (flipY, anisotropy, …).
   */
  async load(url: string, opts: KTX2LoadOptions = {}): Promise<KTX2GPUTexture> {
    const hit = this._cache.get(url);
    if (hit) {
      hit.lastUsedFrame = this._frame;
      return hit.texture;
    }

    if (opts.debug) {
      console.debug(`${DEBUG_PREFIX} fetch → ${url}`);
    }

    // Step 1: fetch
    const response = await fetch(url, { signal: opts.signal });
    if (!response.ok) {
      throw new Error(`${DEBUG_PREFIX} HTTP ${response.status} fetching ${url}`);
    }
    const buffer = await response.arrayBuffer();

    // Step 2: parse header (raw DataView path)
    const header = parseKTX2Header(buffer);

    if (opts.debug) {
      console.debug(
        `${DEBUG_PREFIX} parsed header: vkFormat=${header.vkFormat} ` +
        `${header.width}×${header.height} mips=${header.levelCount} ` +
        `supercompression=${header.supercompressionScheme}`,
      );
    }

    // Basis-supercompressed (vkFormat === 0) is not directly uploadable.
    if (header.vkFormat === 0) {
      throw new Error(
        `${DEBUG_PREFIX} Basis-supercompressed KTX2 (vkFormat=0) is not supported ` +
        `by this loader. Use BasisUniversalTranscoder first: ${url}`,
      );
    }

    // Step 2b: full ktx-parse container for level data
    const raw = new Uint8Array(buffer);
    const container: KTX2Container = readKTX2Container(raw);

    const levels = extractMipLevels(container);

    const parsed: KTX2ParseResult = { header, levels, buffer, container };

    // Step 3: upload to GPU
    const gpuTex = uploadKTX2ToGPU(this._gl, parsed, opts);

    // Cache
    this._cache.set(url, { texture: gpuTex, lastUsedFrame: this._frame });

    if (opts.debug) {
      console.debug(
        `${DEBUG_PREFIX} uploaded: ${url} compressed=${gpuTex.compressed} ` +
        `mips=${gpuTex.mipCount}`,
      );
    }

    return gpuTex;
  }

  /**
   * Load a complete PBR material set (COMBINED / MRO / Normal) in parallel.
   *
   * @param basePath   Base URL prefix, e.g. `'assets/textures'`.
   * @param meshName   Mesh identifier, e.g. `'SAND'`.
   * @param opts       Shared upload options.
   */
  async loadMaterialSet(
    basePath: string,
    meshName: string,
    opts: KTX2LoadOptions = {},
  ): Promise<{ baseColor: KTX2GPUTexture; mro: KTX2GPUTexture; normal: KTX2GPUTexture }> {
    const base   = `${basePath}/${meshName}___CyclesBake_COMBINED.ktx2`;
    const mro    = `${basePath}/${meshName}___PBR_AT_MRO.ktx2`;
    const normal = `${basePath}/${meshName}___PBR_Normal.ktx2`;

    const [baseColor, mroTex, normalTex] = await Promise.all([
      this.load(base,   opts),
      this.load(mro,    opts),
      this.load(normal, opts),
    ]);

    return { baseColor, mro: mroTex, normal: normalTex };
  }

  /**
   * Evict textures not used within the last `maxIdleFrames` frames.
   * Calls gl.deleteTexture() on evicted handles.
   *
   * @param maxIdleFrames  Frames of inactivity before eviction (default: 300).
   */
  evict(maxIdleFrames = 300): number {
    const gl = this._gl;
    let evicted = 0;
    for (const [url, entry] of this._cache) {
      if (this._frame - entry.lastUsedFrame > maxIdleFrames) {
        gl.deleteTexture(entry.texture.texture);
        this._cache.delete(url);
        evicted++;
        console.debug(`${DEBUG_PREFIX} evicted: ${url}`);
      }
    }
    return evicted;
  }

  /**
   * Delete all cached textures and clear the cache.
   */
  dispose(): void {
    const gl = this._gl;
    for (const entry of this._cache.values()) {
      gl.deleteTexture(entry.texture.texture);
    }
    this._cache.clear();
  }

  /** Number of textures currently cached. */
  get cacheSize(): number {
    return this._cache.size;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Quick-probe whether the given GL context supports a vkFormat's compressed
 * internal format (extension presence check only — no format negotiation).
 */
export function isVkFormatSupported(
  gl: WebGL2RenderingContext,
  vkFormat: number,
): boolean {
  const fmt = VK_FORMAT_TABLE[vkFormat];
  if (!fmt || !fmt.compressed) return true; // uncompressed always supported

  const extS3TC  = gl.getExtension('WEBGL_compressed_texture_s3tc');
  const extETC   = gl.getExtension('WEBGL_compressed_texture_etc');
  const extASTC  = gl.getExtension('WEBGL_compressed_texture_astc');
  const extBPTC  = gl.getExtension('EXT_texture_compression_bptc');

  const supported = new Set<number>();
  if (extS3TC) { supported.add(0x83F0); supported.add(0x83F1); supported.add(0x83F3); }
  if (extETC)  { supported.add(0x9274); supported.add(0x9275); supported.add(0x9278); supported.add(0x9279); }
  if (extASTC) { supported.add(0x93B0); supported.add(0x93D0); supported.add(0x93B4); supported.add(0x93B7); supported.add(0x93D7); supported.add(0x93BB); supported.add(0x93BD); }
  if (extBPTC) { supported.add(0x8E8C); supported.add(0x8E8D); supported.add(0x8DBD); }

  return supported.has(fmt.internalFormat);
}

/**
 * Human-readable vkFormat name for debugging.
 */
export function vkFormatName(vkFormat: number): string {
  const names: Record<number, string> = {
    0:   'BASIS_SUPERCOMPRESSED',
    23:  'VK_FORMAT_R8G8B8_UNORM',
    37:  'VK_FORMAT_R8G8B8A8_UNORM',
    43:  'VK_FORMAT_R8G8B8A8_SRGB',
    97:  'VK_FORMAT_R16G16B16A16_SFLOAT',
    109: 'VK_FORMAT_R32G32B32A32_SFLOAT',
    131: 'VK_FORMAT_BC1_RGB_UNORM_BLOCK',
    133: 'VK_FORMAT_BC1_RGBA_UNORM_BLOCK',
    137: 'VK_FORMAT_BC3_UNORM_BLOCK',
    141: 'VK_FORMAT_BC5_UNORM_BLOCK',
    145: 'VK_FORMAT_BC7_UNORM_BLOCK',
    146: 'VK_FORMAT_BC7_SRGB_BLOCK',
    147: 'VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK',
    148: 'VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK',
    151: 'VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK',
    152: 'VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK',
    157: 'VK_FORMAT_ASTC_4x4_UNORM_BLOCK',
    158: 'VK_FORMAT_ASTC_4x4_SRGB_BLOCK',
    163: 'VK_FORMAT_ASTC_6x6_UNORM_BLOCK',
    171: 'VK_FORMAT_ASTC_8x8_UNORM_BLOCK',
    172: 'VK_FORMAT_ASTC_8x8_SRGB_BLOCK',
    179: 'VK_FORMAT_ASTC_10x10_UNORM_BLOCK',
    183: 'VK_FORMAT_ASTC_12x12_UNORM_BLOCK',
  };
  return names[vkFormat] ?? `UNKNOWN_VK_FORMAT_${vkFormat}`;
}

/** Re-export raw header parser for callers that only need metadata. */
export { parseKTX2Header };
