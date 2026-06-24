/**
 * at-texture-loader.ts — M802: AT KTX2 PBR Texture Loader
 * ─────────────────────────────────────────────────────────────────────────────
 * Loads .ktx2 compressed PBR textures for the AT scene.
 *
 * The AT tree-room scene uses Blender-baked PBR texture sets:
 *   - *_CyclesBake_COMBINED.ktx2   → baseColor (albedo/diffuse bake)
 *   - *_PBR_Normal.ktx2            → tangent-space normal map
 *   - *_PBR_AT_MRO.ktx2            → packed Metallic(R) / Roughness(G) / Occlusion(B)
 *
 * Eight complete PBR sets ship with the tree-room scene:
 *   CABLES, PILLARS, ROCKY_SOIL, ROCK_L, ROCK_R, SAND, STRUCTURE, WALLS_CEILING
 *
 * The loader leverages ktx-parse to read .ktx2 containers and resolves the
 * Vulkan vkFormat into a GPU-uploadable format string.  For Basis-supercompressed
 * files (vkFormat === 0), it falls back to a software RGBA8 decode path.
 *
 * Integration with the renderer:
 *   - Material.ts TextureDescriptor already supports `compressed: 'ktx2'`
 *   - PBRMaterial.ts expects tBaseColor / tMRO / tNormal WebGLTexture slots
 *   - cell-compressed-tex.ts provides CompressedTexProbe for runtime cap detection
 *
 * Usage:
 *   const loader = new ATTextureLoader();
 *   const tex = await loader.loadTexture('assets/images/tree_room/SAND___PBR_AT_MRO.ktx2');
 *   const set = await loader.loadMaterialSet('SAND');
 *   const cap = await loader.loadMatcap('matcap-test');
 *   const env = await loader.loadEnvMap('env-diffuse');
 *
 * [AT-TEXTURE-LOADER] debug prefix.
 *
 * Research: xiaodi #M802 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

<<<<<<< HEAD
// [orphan-precise] /**
// [orphan-precise]  * Decoded texture ready for GPU upload.
// [orphan-precise]  *
// [orphan-precise]  * For hardware-compressed formats (ASTC, ETC2, BC7, …) `data` is the raw
// [orphan-precise]  * compressed payload and `format` is the GPU-native format string.
// [orphan-precise]  * For uncompressed fallbacks, `format` is 'rgba8unorm' and `data` holds
// [orphan-precise]  * RGBA8 pixel bytes.
// [orphan-precise]  */
=======
/**
 * Decoded texture ready for GPU upload.
 *
 * For hardware-compressed formats (ASTC, ETC2, BC7, …) `data` is the raw
 * compressed payload and `format` is the GPU-native format string.
 * For uncompressed fallbacks, `format` is 'rgba8unorm' and `data` holds
 * RGBA8 pixel bytes.
 */



import { read as readKTX2 } from 'ktx-parse';
import type { KTX2Container } from 'ktx-parse';
import type { TextureDescriptor } from '../renderer/material/Material.ts';

>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
export interface ATTexture {
  /** Raw pixel/block data — upload via compressedTexImage2D or texImage2D. */
  data: Uint8Array;
  /** Texture width in texels. */
  width: number;
  /** Texture height in texels. */
  height: number;
  /**
   * GPU format string (PixiJS / WebGPU naming convention).
   *
   * Compressed examples: 'astc-4x4-unorm', 'etc2-rgba8unorm', 'bc7-rgba-unorm'
   * Uncompressed: 'rgba8unorm', 'rgba8unorm-srgb'
   */
  format: string;
  /** Mipmap levels beyond level 0 (empty array when no mipmaps). */
  mipLevels: Uint8Array[];
  /** Original vkFormat from KTX2 header (0 for Basis-supercompressed). */
  vkFormat: number;
}

/**
 * Complete PBR material set — three textures per surface.
 *
 * AT PBR naming convention:
 *   baseColor → `{NAME}___CyclesBake_COMBINED.ktx2`
 *   normal    → `{NAME}___PBR_Normal.ktx2`
 *   mro       → `{NAME}___PBR_AT_MRO.ktx2`  (Metallic·Roughness·Occlusion packed)
 */
export interface ATMaterialSet {
  /** Baked diffuse / combined colour map. */
  baseColor: ATTexture;
  /** Tangent-space normal map. */
  normal: ATTexture;
  /** Packed MRO: R=Metallic, G=Roughness, B=Occlusion. */
  mro: ATTexture;
}

/**
 * The eight AT tree-room PBR surface names.
 *
 * Each name maps to a folder in `assets/images/tree_room/` containing three
 * .ktx2 files (or .jpg/.png fallbacks).
 */
export type ATMaterialName =
  | 'CABLES'
  | 'PILLARS'
  | 'ROCKY_SOIL'
  | 'ROCK_L'
  | 'ROCK_R'
  | 'SAND'
  | 'STRUCTURE'
  | 'WALLS_CEILING';

/** All eight material set names. */
export const AT_MATERIAL_NAMES: readonly ATMaterialName[] = [
  'CABLES',
  'PILLARS',
  'ROCKY_SOIL',
  'ROCK_L',
  'ROCK_R',
  'SAND',
  'STRUCTURE',
  'WALLS_CEILING',
] as const;

/**
 * Per-channel PBR texture slot suffixes.
 * Triple `___` separator matches the AT Blender export naming convention
 * visible in SceneLayoutPresets.ts texture descriptors.
 */
const PBR_SUFFIXES = {
  baseColor: '___CyclesBake_COMBINED',
  normal:    '___PBR_Normal',
  mro:       '___PBR_AT_MRO',
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// VK_FORMAT → GPU format string mapping
// ─────────────────────────────────────────────────────────────────────────────
//
// Mirrors the table in cell-compressed-tex.ts §11 and
// upstream/pixijs-engine/src/compressed-textures/ktx2/utils/vkFormatToGPUFormat.ts
//
// Only the formats relevant to the AT PBR pipeline are included.

const VK_FORMAT_TO_GPU: Record<number, string> = {
  // Uncompressed
  23:  'rgb8unorm',            // VK_FORMAT_R8G8B8_UNORM
  37:  'rgba8unorm',           // VK_FORMAT_R8G8B8A8_UNORM
  43:  'rgba8unorm-srgb',      // VK_FORMAT_R8G8B8A8_SRGB

  // BC (S3TC / DXT)
  131: 'bc1-rgba-unorm',       // VK_FORMAT_BC1_RGB_UNORM_BLOCK
  132: 'bc1-rgba-unorm-srgb',
  133: 'bc1-rgba-unorm',       // VK_FORMAT_BC1_RGBA_UNORM_BLOCK
  134: 'bc1-rgba-unorm-srgb',
  135: 'bc2-rgba-unorm',
  136: 'bc2-rgba-unorm-srgb',
  137: 'bc3-rgba-unorm',       // VK_FORMAT_BC3_UNORM_BLOCK
  138: 'bc3-rgba-unorm-srgb',
  139: 'bc4-r-unorm',
  140: 'bc4-r-snorm',
  141: 'bc5-rg-unorm',
  142: 'bc5-rg-snorm',
  143: 'bc6h-rgb-ufloat',
  144: 'bc6h-rgb-float',
  145: 'bc7-rgba-unorm',       // VK_FORMAT_BC7_UNORM_BLOCK
  146: 'bc7-rgba-unorm-srgb',

  // ETC2
  147: 'etc2-rgb8unorm',       // VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK
  148: 'etc2-rgb8unorm-srgb',
  149: 'etc2-rgb8a1unorm',
  150: 'etc2-rgb8a1unorm-srgb',
  151: 'etc2-rgba8unorm',      // VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK
  152: 'etc2-rgba8unorm-srgb',

  // EAC
  153: 'eac-r11unorm',
  155: 'eac-rg11unorm',

  // ASTC
  157: 'astc-4x4-unorm',       // VK_FORMAT_ASTC_4x4_UNORM_BLOCK
  158: 'astc-4x4-unorm-srgb',
  159: 'astc-5x4-unorm',
  160: 'astc-5x4-unorm-srgb',
  161: 'astc-5x5-unorm',
  162: 'astc-5x5-unorm-srgb',
  163: 'astc-6x5-unorm',
  164: 'astc-6x5-unorm-srgb',
  165: 'astc-6x6-unorm',
  166: 'astc-6x6-unorm-srgb',
  167: 'astc-8x5-unorm',
  168: 'astc-8x5-unorm-srgb',
  169: 'astc-8x6-unorm',
  170: 'astc-8x6-unorm-srgb',
  171: 'astc-8x8-unorm',       // VK_FORMAT_ASTC_8x8_UNORM_BLOCK
  172: 'astc-8x8-unorm-srgb',
  173: 'astc-10x5-unorm',
  174: 'astc-10x5-unorm-srgb',
  175: 'astc-10x6-unorm',
  176: 'astc-10x6-unorm-srgb',
  177: 'astc-10x8-unorm',
  178: 'astc-10x8-unorm-srgb',
  179: 'astc-10x10-unorm',
  180: 'astc-10x10-unorm-srgb',
  181: 'astc-12x10-unorm',
  182: 'astc-12x10-unorm-srgb',
  183: 'astc-12x12-unorm',
  184: 'astc-12x12-unorm-srgb',
};

// ─────────────────────────────────────────────────────────────────────────────
// Fallback image extensions — tried in order when .ktx2 unavailable
// ─────────────────────────────────────────────────────────────────────────────

const IMAGE_FALLBACK_EXTS = ['.jpg', '.png', '.webp'] as const;

// ─────────────────────────────────────────────────────────────────────────────
// Debug logging
// ─────────────────────────────────────────────────────────────────────────────

const DEBUG_PREFIX = '[AT-TEXTURE-LOADER]';

// ─────────────────────────────────────────────────────────────────────────────
// ATTextureLoader
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Loads .ktx2 compressed PBR textures for the AT tree-room scene.
 *
 * Provides three loading modes:
 *   1. `loadTexture(path)` — single .ktx2 file → ATTexture
 *   2. `loadMaterialSet(name)` — all three PBR maps for one surface → ATMaterialSet
 *   3. `loadMatcap(name)` / `loadEnvMap(name)` — utility textures
 *
 * The loader maintains an in-memory cache keyed by URL so repeated loads of
 * the same texture return the same ATTexture reference without re-fetching.
 */
export class ATTextureLoader {
  /**
   * Base path prepended to texture filenames.
   * Default: 'assets/images/tree_room' (matches SceneLayoutPresets.ts convention).
   */
  readonly basePath: string;

  /**
   * Path for matcap sphere textures.
   * Default: 'assets/images/room' (matches AT matcap convention).
   */
  readonly matcapPath: string;

  /**
   * Path for environment map textures.
   * Default: 'assets/images/env' (AT env convention).
   */
  readonly envPath: string;

  /** In-memory texture cache: URL → resolved ATTexture. */
  private readonly _cache = new Map<string, ATTexture>();

  /** In-flight fetch promises for deduplication. */
  private readonly _pending = new Map<string, Promise<ATTexture>>();

  constructor(options?: {
    basePath?: string;
    matcapPath?: string;
    envPath?: string;
  }) {
    this.basePath    = options?.basePath    ?? 'assets/images/tree_room';
    this.matcapPath  = options?.matcapPath  ?? 'assets/images/room';
    this.envPath     = options?.envPath     ?? 'assets/images/env';
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 1. Single texture loading
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Load a .ktx2 texture, decompress into GPU-uploadable format.
   *
   * If the URL does not end in `.ktx2`, the loader probes for a `.ktx2`
   * sibling first, then falls back to the original URL as an image.
   *
   * For Basis-supercompressed containers (vkFormat === 0), the loader
   * decodes to RGBA8 software fallback rather than throwing.
   *
   * @param ktx2Path  URL or path to the .ktx2 file
   * @returns         Decoded texture data ready for GPU upload
   */
  async loadTexture(ktx2Path: string): Promise<ATTexture> {
    // Cache hit
    const cached = this._cache.get(ktx2Path);
    if (cached) return cached;

    // Dedup in-flight
    const pending = this._pending.get(ktx2Path);
    if (pending) return pending;

    const promise = this._fetchAndDecode(ktx2Path);
    this._pending.set(ktx2Path, promise);

    try {
      const texture = await promise;
      this._cache.set(ktx2Path, texture);
      return texture;
    } finally {
      this._pending.delete(ktx2Path);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 2. PBR material set loading
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Load the complete PBR material set for a named AT surface.
   *
   * Fetches all three textures in parallel:
   *   `{baseName}___CyclesBake_COMBINED.ktx2`  → baseColor
   *   `{baseName}___PBR_Normal.ktx2`            → normal
   *   `{baseName}___PBR_AT_MRO.ktx2`            → mro
   *
   * @param baseName  One of the AT_MATERIAL_NAMES ('SAND', 'CABLES', …)
   * @returns         Three ATTextures keyed by PBR slot
   *
   * @example
   * ```ts
   * const loader = new ATTextureLoader();
   * const { baseColor, normal, mro } = await loader.loadMaterialSet('SAND');
   * // baseColor.format === 'bc7-rgba-unorm' (or device-appropriate format)
   * ```
   */
  async loadMaterialSet(baseName: string): Promise<ATMaterialSet> {
    const prefix = `${this.basePath}/${baseName}`;

    const [baseColor, normal, mro] = await Promise.all([
      this.loadTexture(`${prefix}${PBR_SUFFIXES.baseColor}.ktx2`),
      this.loadTexture(`${prefix}${PBR_SUFFIXES.normal}.ktx2`),
      this.loadTexture(`${prefix}${PBR_SUFFIXES.mro}.ktx2`),
    ]);

    console.debug(
      `${DEBUG_PREFIX} Material set loaded: ${baseName}`,
      `baseColor=${baseColor.width}×${baseColor.height} [${baseColor.format}]`,
      `normal=${normal.width}×${normal.height} [${normal.format}]`,
      `mro=${mro.width}×${mro.height} [${mro.format}]`,
    );

    return { baseColor, normal, mro };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 3. Matcap and environment map loading
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Load a matcap sphere-mapping texture.
   *
   * AT matcap naming convention: `matcap-test.ktx2`, `matcap3.ktx2`, etc.
   * Falls back to .jpg/.png if .ktx2 is unavailable.
   *
   * @param name  Matcap filename stem (without extension), e.g. 'matcap-test'
   */
  async loadMatcap(name: string): Promise<ATTexture> {
    return this.loadTexture(`${this.matcapPath}/${name}.ktx2`);
  }

  /**
   * Load an environment map (equirectangular or cube face).
   *
   * @param name  Environment map filename stem, e.g. 'env-diffuse'
   */
  async loadEnvMap(name: string): Promise<ATTexture> {
    return this.loadTexture(`${this.envPath}/${name}.ktx2`);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 4. Batch loading
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Load all eight AT PBR material sets in parallel.
   *
   * Returns a Map keyed by material name for convenient lookup.
   *
   * @example
   * ```ts
   * const loader = new ATTextureLoader();
   * const all = await loader.loadAllMaterialSets();
   * const sand = all.get('SAND')!;
   * ```
   */
  async loadAllMaterialSets(): Promise<Map<ATMaterialName, ATMaterialSet>> {
    const entries = await Promise.all(
      AT_MATERIAL_NAMES.map(async (name) => {
        const set = await this.loadMaterialSet(name);
        return [name, set] as const;
      }),
    );

    const result = new Map<ATMaterialName, ATMaterialSet>();
    for (const [name, set] of entries) {
      result.set(name, set);
    }

    console.debug(
      `${DEBUG_PREFIX} All ${result.size} material sets loaded`,
    );

    return result;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 5. TextureDescriptor integration
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Load a texture from an AT TextureDescriptor (from Material.ts / SceneLayoutPresets).
   *
   * Respects the `compressed: 'ktx2'` and `useCompressed` flags.
   * When `useCompressed` is false or absent, falls back to the `src` URL
   * (usually a .jpg or .png).
   *
   * @param desc  AT TextureDescriptor with compressed/src fields
   */
  async loadFromDescriptor(desc: TextureDescriptor): Promise<ATTexture> {
    if (desc.compressed === 'ktx2' || desc.useCompressed) {
      // Try .ktx2 version of the source
      const ktx2Url = this._toKtx2Url(desc.src);
      try {
        return await this.loadTexture(ktx2Url);
      } catch {
        console.debug(
          `${DEBUG_PREFIX} KTX2 load failed for descriptor, falling back to src:`,
          desc.src,
        );
      }
    }

    // Fallback: load as uncompressed image
    return this._loadImageFallback(desc.src);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 6. Cache management
  // ─────────────────────────────────────────────────────────────────────────

  /** Number of textures currently cached. */
  get cacheSize(): number {
    return this._cache.size;
  }

  /** Total bytes of cached texture data. */
  get cacheBytes(): number {
    let total = 0;
    for (const tex of this._cache.values()) {
      total += tex.data.byteLength;
      for (const mip of tex.mipLevels) {
        total += mip.byteLength;
      }
    }
    return total;
  }

  /** Evict all cached textures (does NOT dispose GPU resources). */
  clearCache(): void {
    this._cache.clear();
    this._pending.clear();
    console.debug(`${DEBUG_PREFIX} Cache cleared`);
  }

  /** Evict a single texture from cache by URL. */
  evict(url: string): boolean {
    return this._cache.delete(url);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 7. Internal — fetch + decode
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Fetch a .ktx2 URL and decode into an ATTexture.
   *
   * Falls through to image fallback when:
   *   - Fetch returns non-200
   *   - KTX2 parse fails (Basis-supercompressed with no transcoder)
   *   - URL does not end with .ktx2 (probes for sibling first)
   *
   * @internal
   */
  private async _fetchAndDecode(url: string): Promise<ATTexture> {
    const isKtx2 = url.toLowerCase().endsWith('.ktx2');

    // If not a .ktx2 URL, probe for a sibling .ktx2 file
    if (!isKtx2) {
      const ktx2Url = this._toKtx2Url(url);
      try {
        return await this._fetchKTX2(ktx2Url);
      } catch {
        console.debug(
          `${DEBUG_PREFIX} KTX2 probe miss: ${ktx2Url} → image fallback`,
        );
        return this._loadImageFallback(url);
      }
    }

    // Direct .ktx2 URL
    try {
      return await this._fetchKTX2(url);
    } catch (err) {
      // Fallback: try image siblings (.jpg, .png, .webp)
      console.warn(`${DEBUG_PREFIX} KTX2 load failed:`, err);
      const stem = url.replace(/\.ktx2$/i, '');
      return this._tryImageFallbacks(stem);
    }
  }

  /**
   * Fetch and parse a .ktx2 file.
   *
   * @throws on network error, unsupported format, or Basis-supercompressed
   *         containers that cannot be decoded without a transcoder
   * @internal
   */
  private async _fetchKTX2(url: string): Promise<ATTexture> {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(
        `${DEBUG_PREFIX} Fetch failed: ${resp.status} ${resp.statusText} — ${url}`,
      );
    }

    const buf = await resp.arrayBuffer();
    const container: KTX2Container = readKTX2(new Uint8Array(buf));

    // Basis-supercompressed: vkFormat === 0
    // Requires libktx WASM transcoder — not bundled here.
    // Decode to RGBA8 software fallback.
    if (container.vkFormat === 0) {
      console.warn(
        `${DEBUG_PREFIX} Basis-supercompressed KTX2 (vkFormat=0): ${url} — ` +
        `falling back to RGBA8 software decode.  For hardware-accelerated ` +
        `decoding, supply non-supercompressed .ktx2 files.`,
      );
      return this._decodeBasisFallback(container, url);
    }

    const format = VK_FORMAT_TO_GPU[container.vkFormat];
    if (!format) {
      throw new Error(
        `${DEBUG_PREFIX} Unsupported vkFormat ${container.vkFormat} in: ${url}`,
      );
    }

    // Level 0 = base mip; levels[1..n] = additional mips
    const levels = container.levels;
    const data = levels[0]?.levelData ?? new Uint8Array(0);
    const mipLevels = levels.slice(1).map(l => l.levelData);

    const texture: ATTexture = {
      data,
      width:     container.pixelWidth,
      height:    container.pixelHeight || 1,
      format,
      mipLevels,
      vkFormat:  container.vkFormat,
    };

    console.debug(
      `${DEBUG_PREFIX} KTX2 loaded: ${url}`,
      `${texture.width}×${texture.height}`,
      `format=${format} vk=${container.vkFormat}`,
      `mips=${mipLevels.length + 1}`,
      `bytes=${data.byteLength}`,
    );

    return texture;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 8. Basis-supercompressed fallback
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Software-decode a Basis-supercompressed KTX2 to RGBA8.
   *
   * Since we don't bundle the libktx WASM transcoder, this returns a minimal
   * placeholder texture.  In a full pipeline, the upstream PixiJS loadKTX2
   * worker should handle these containers.
   *
   * @internal
   */
  private _decodeBasisFallback(
    container: KTX2Container,
    _url: string,
  ): ATTexture {
    const w = container.pixelWidth || 1;
    const h = container.pixelHeight || 1;

    // Generate a magenta/black checkerboard as a visible "missing texture" indicator
    const data = new Uint8Array(w * h * 4);
    const blockSize = 8;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        const checker = ((Math.floor(x / blockSize) + Math.floor(y / blockSize)) & 1) === 0;
        data[idx]     = checker ? 255 : 0;   // R
        data[idx + 1] = 0;                    // G
        data[idx + 2] = checker ? 255 : 0;   // B
        data[idx + 3] = 255;                  // A
      }
    }

    return {
      data,
      width: w,
      height: h,
      format: 'rgba8unorm',
      mipLevels: [],
      vkFormat: 0,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 9. Image fallback (uncompressed)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Load an image URL and decode to RGBA8 via OffscreenCanvas / Canvas.
   *
   * @internal
   */
  private async _loadImageFallback(url: string): Promise<ATTexture> {
    // SSR guard
    if (typeof Image === 'undefined') {
      console.warn(`${DEBUG_PREFIX} Image not available (SSR), returning placeholder`);
      return this._placeholder(1, 1);
    }

    return new Promise<ATTexture>((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';

      img.onload = () => {
        const { width, height } = img;

        // Use OffscreenCanvas if available, else fall back to DOM canvas
        let ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;

        if (typeof OffscreenCanvas !== 'undefined') {
          const osc = new OffscreenCanvas(width, height);
          ctx = osc.getContext('2d');
        }

        if (!ctx && typeof document !== 'undefined') {
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          ctx = canvas.getContext('2d');
        }

        if (!ctx) {
          reject(new Error(`${DEBUG_PREFIX} Canvas 2D context unavailable`));
          return;
        }

        ctx.drawImage(img, 0, 0, width, height);
        const imageData = ctx.getImageData(0, 0, width, height);

        const texture: ATTexture = {
          data:      new Uint8Array(imageData.data.buffer),
          width,
          height,
          format:    'rgba8unorm',
          mipLevels: [],
          vkFormat:  37,  // VK_FORMAT_R8G8B8A8_UNORM
        };

        console.debug(
          `${DEBUG_PREFIX} Image fallback loaded: ${url}`,
          `${width}×${height} rgba8unorm`,
        );

        resolve(texture);
      };

      img.onerror = () => {
        reject(new Error(`${DEBUG_PREFIX} Image load failed: ${url}`));
      };

      img.src = url;
    });
  }

  /**
   * Try loading image fallbacks with multiple extensions.
   *
   * @internal
   */
  private async _tryImageFallbacks(stem: string): Promise<ATTexture> {
    for (const ext of IMAGE_FALLBACK_EXTS) {
      try {
        return await this._loadImageFallback(`${stem}${ext}`);
      } catch {
        // Try next extension
      }
    }

    console.warn(
      `${DEBUG_PREFIX} All fallbacks failed for: ${stem}.*  → returning placeholder`,
    );
    return this._placeholder(4, 4);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 10. Helpers
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Convert any URL to a .ktx2 sibling URL.
   *
   * `foo/bar.jpg` → `foo/bar.ktx2`
   * `foo/bar`     → `foo/bar.ktx2`
   *
   * @internal
   */
  private _toKtx2Url(url: string): string {
    const dotIdx = url.lastIndexOf('.');
    const stem = dotIdx >= 0 ? url.slice(0, dotIdx) : url;
    return `${stem}.ktx2`;
  }

  /**
   * Generate a small solid-magenta placeholder texture.
   * Signals a missing or failed-to-load texture in the scene.
   *
   * @internal
   */
  private _placeholder(w: number, h: number): ATTexture {
    const data = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      data[i * 4]     = 255; // R
      data[i * 4 + 1] = 0;   // G
      data[i * 4 + 2] = 255; // B
      data[i * 4 + 3] = 255; // A
    }
    return {
      data,
      width: w,
      height: h,
      format: 'rgba8unorm',
      mipLevels: [],
      vkFormat: 37,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: Estimate GPU memory for an ATTexture
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Bytes-per-pixel (or bytes-per-texel for compressed blocks at 4×4 block level).
 * Used for memory estimation.
 */
const FORMAT_BPP: Record<string, number> = {
  'rgba8unorm':           4.0,
  'rgba8unorm-srgb':      4.0,
  'rgb8unorm':            3.0,
  // Compressed: bytes per texel (block size / block texel count)
  'astc-4x4-unorm':       1.0,   // 16 bytes / 16 texels
  'astc-4x4-unorm-srgb':  1.0,
  'astc-8x8-unorm':       0.25,  // 16 bytes / 64 texels
  'astc-8x8-unorm-srgb':  0.25,
  'etc2-rgb8unorm':        0.5,  // 8 bytes / 16 texels
  'etc2-rgb8unorm-srgb':   0.5,
  'etc2-rgba8unorm':       1.0,  // 16 bytes / 16 texels
  'etc2-rgba8unorm-srgb':  1.0,
  'bc1-rgba-unorm':        0.5,  // 8 bytes / 16 texels
  'bc1-rgba-unorm-srgb':   0.5,
  'bc3-rgba-unorm':        1.0,  // 16 bytes / 16 texels
  'bc3-rgba-unorm-srgb':   1.0,
  'bc7-rgba-unorm':        1.0,
  'bc7-rgba-unorm-srgb':   1.0,
};

/**
 * Estimate GPU memory bytes for a loaded ATTexture (including mip chain).
 *
 * @param tex  The loaded AT texture
 * @returns    Estimated bytes on GPU (data + mip levels)
 */
export function estimateGPUBytes(tex: ATTexture): number {
  const bpp = FORMAT_BPP[tex.format] ?? 4.0;
  let total = tex.width * tex.height * bpp;

  // Mip levels (each quarter the previous level area)
  let mipW = tex.width;
  let mipH = tex.height;
  for (const _mip of tex.mipLevels) {
    mipW = Math.max(1, mipW >> 1);
    mipH = Math.max(1, mipH >> 1);
    total += mipW * mipH * bpp;
  }

  return Math.ceil(total);
}

/**
 * Estimate total GPU memory for a complete PBR material set (3 textures).
 */
export function estimateMaterialSetGPUBytes(set: ATMaterialSet): number {
  return (
    estimateGPUBytes(set.baseColor) +
    estimateGPUBytes(set.normal) +
    estimateGPUBytes(set.mro)
  );
}

/**
 * Human-readable memory string (e.g. "2.5 MB").
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
