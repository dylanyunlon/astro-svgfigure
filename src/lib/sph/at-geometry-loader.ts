/**
 * at-geometry-loader.ts — Load AT .bin Draco-compressed geometry files.
 *
 * Upstream reference:
 *   upstream/activetheory-assets/geometry/*.bin
 *
 * Each .bin is a raw Draco-compressed buffer (no glTF/GLB wrapper).
 * DracoThread (from threed-pipeline.ts) decodes them off-main-thread
 * into interleaved position/normal/uv/index arrays.
 *
 * Asset manifest — cell-pubsub-loop role mapping:
 *   jellyfish.bin          (24K)  — organic cell body shape
 *   flower_spine-128.bin  (133K)  — edge particle-flow spine
 *   cables.bin             (87K)  — inter-cell data connections
 *   structure.bin         (145K)  — scene structural scaffolding
 *   spine.bin              (15K)  — spline backbone
 *   hexagon_gem.bin       (501B)  — cell icon gem
 */




// ── Types ──────────────────────────────────────────────────────────────────────

<<<<<<< HEAD
// [orphan-precise] /** Decoded AT geometry with guaranteed attribute arrays. */
=======
/** Decoded AT geometry with guaranteed attribute arrays. */



import { DracoThread } from '../threed-pipeline';
import type { GeometryDescriptor } from '../threed-pipeline';

>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
export interface ATGeometry {
  positions: Float32Array;   // xyz, stride 3
  normals:   Float32Array;   // xyz, stride 3
  uvs:       Float32Array;   // uv,  stride 2
  indices:   Uint32Array;
  vertexCount: number;
  indexCount:  number;
}

/** Known AT geometry asset names (filename without extension). */
export type ATGeometryName =
  | 'jellyfish'
  | 'flower_spine-128'
  | 'cables'
  | 'structure'
  | 'spine'
  | 'hexagon_gem';

/** Configuration for the loader. */
export interface ATGeometryLoaderOptions {
  /**
   * Base path to the geometry directory.
   * @default '/upstream/activetheory-assets/geometry'
   */
  basePath?: string;

  /** Pre-existing DracoThread instance to reuse. */
  dracoThread?: DracoThread;

  /** Draco WASM decoder URL passed to DracoThread if creating a new one. */
  decoderUrl?: string;
}

// ── Asset manifest ─────────────────────────────────────────────────────────────

interface AssetEntry {
  file: string;
  /** Approximate file size in bytes (for progress reporting). */
  sizeHint: number;
  /** Semantic role within the cell-pubsub-loop visualisation. */
  role: string;
}

const ASSET_MANIFEST: Record<ATGeometryName, AssetEntry> = {
  jellyfish:           { file: 'jellyfish.bin',          sizeHint: 24_517,  role: 'cell organic body' },
  'flower_spine-128':  { file: 'flower_spine-128.bin',   sizeHint: 133_260, role: 'edge particle-flow spine' },
  cables:              { file: 'cables.bin',             sizeHint: 87_527,  role: 'inter-cell data connection' },
  structure:           { file: 'structure.bin',          sizeHint: 145_133, role: 'scene scaffolding' },
  spine:               { file: 'spine.bin',              sizeHint: 15_117,  role: 'spline backbone' },
  hexagon_gem:         { file: 'hexagon_gem.bin',        sizeHint: 501,     role: 'cell icon gem' },
};

// ── ATGeometryLoader ───────────────────────────────────────────────────────────

/**
 * ATGeometryLoader
 *
 * Fetches raw Draco .bin files from the AT geometry asset directory and
 * decodes them via DracoThread into ATGeometry descriptors.
 *
 * Usage:
 *   const loader = new ATGeometryLoader();
 *   const jelly  = await loader.loadGeometry('jellyfish');
 *   const all    = await loader.loadAll();
 */
export class ATGeometryLoader {
  private basePath: string;
  private draco: DracoThread;
  private ownsThread: boolean;
  private cache = new Map<string, ATGeometry>();

  constructor(options: ATGeometryLoaderOptions = {}) {
    this.basePath = (options.basePath ?? '/upstream/activetheory-assets/geometry')
      .replace(/\/+$/, '');

    if (options.dracoThread) {
      this.draco = options.dracoThread;
      this.ownsThread = false;
    } else {
      this.draco = new DracoThread(options.decoderUrl);
      this.ownsThread = true;
    }
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Load a single .bin geometry file and decode it via DracoThread.
   *
   * Accepts either a known ATGeometryName (`'jellyfish'`) or a raw path
   * (`'/custom/path/model.bin'`). Known names resolve through the manifest;
   * arbitrary paths are fetched directly.
   *
   * Results are cached — subsequent calls for the same key return the
   * cached ATGeometry without re-fetching or re-decoding.
   */
  async loadGeometry(binPath: string): Promise<ATGeometry> {
    // Resolve manifest name → full URL
    const cacheKey = binPath;
    const cached = this.cache.get(cacheKey);
    if (cached) return cached;

    const url = this.resolveUrl(binPath);

    // Fetch the raw Draco buffer
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(
        `ATGeometryLoader: failed to fetch ${url} — HTTP ${res.status}`
      );
    }
    const buffer = await res.arrayBuffer();

    // Decode via DracoThread (off-main-thread)
    const descriptor: GeometryDescriptor = await this.draco.decode(buffer);

    // Normalise into ATGeometry — guarantee all attributes exist
    const geo = this.descriptorToATGeometry(descriptor);
    this.cache.set(cacheKey, geo);
    return geo;
  }

  /**
   * Pre-load every geometry in the AT asset manifest.
   * All fetches run in parallel; Draco decoding is serialised through
   * the single DracoThread worker (requests queue internally).
   *
   * Returns a Map keyed by ATGeometryName.
   */
  async loadAll(): Promise<Map<string, ATGeometry>> {
    const names = Object.keys(ASSET_MANIFEST) as ATGeometryName[];
    const entries = await Promise.all(
      names.map(async (name) => {
        const geo = await this.loadGeometry(name);
        return [name, geo] as const;
      })
    );

    const result = new Map<string, ATGeometry>();
    for (const [name, geo] of entries) {
      result.set(name, geo);
    }
    return result;
  }

  /** Return the known asset manifest (readonly). */
  get manifest(): Readonly<Record<ATGeometryName, AssetEntry>> {
    return ASSET_MANIFEST;
  }

  /** Clear the in-memory geometry cache. */
  clearCache(): void {
    this.cache.clear();
  }

  /** Dispose the loader. Terminates the DracoThread if we own it. */
  dispose(): void {
    this.cache.clear();
    if (this.ownsThread) {
      this.draco.dispose();
    }
  }

  // ── Internals ──────────────────────────────────────────────────────────────

  /**
   * Resolve a geometry key to a fetchable URL.
   * Known manifest names map to `basePath/filename.bin`.
   * Anything else is treated as a literal path.
   */
  private resolveUrl(binPath: string): string {
    const entry = ASSET_MANIFEST[binPath as ATGeometryName];
    if (entry) {
      return `${this.basePath}/${entry.file}`;
    }
    // Treat as a raw path — if it doesn't start with '/' or 'http',
    // resolve relative to basePath
    if (binPath.startsWith('/') || binPath.startsWith('http')) {
      return binPath;
    }
    return `${this.basePath}/${binPath}`;
  }

  /**
   * Convert a GeometryDescriptor (from DracoThread) into a fully-typed
   * ATGeometry, filling in missing attributes with zero-initialised arrays.
   */
  private descriptorToATGeometry(d: GeometryDescriptor): ATGeometry {
    const vertexCount = d.vertexCount;

    const positions = d.positions;
    const normals   = d.normals  ?? new Float32Array(vertexCount * 3);
    const uvs       = d.uvs      ?? new Float32Array(vertexCount * 2);
    const indices   = d.indices  ?? new Uint32Array(0);

    return {
      positions,
      normals,
      uvs,
      indices,
      vertexCount,
      indexCount: indices.length,
    };
  }
}
