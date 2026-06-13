// asset-pipeline.ts — AssetLoader, AssetList, CMSData, Config

// ---------------------------------------------------------------------------
// AssetLoader
// ---------------------------------------------------------------------------

export type AssetType = "image" | "audio" | "video" | "json" | "text" | "binary" | "blob";

export interface AssetEntry {
  url: string;
  type: AssetType;
}

export interface LoadedAsset<T = unknown> {
  url: string;
  type: AssetType;
  data: T;
}

export type ProgressCallback = (loaded: number, total: number, url: string) => void;

export class AssetLoader {
  private _cache = new Map<string, LoadedAsset<unknown>>();
  private _progressCbs: ProgressCallback[] = [];

  /** Register a progress listener. Returns an unsubscribe fn. */
  onProgress(cb: ProgressCallback): () => void {
    this._progressCbs.push(cb);
    return () => { this._progressCbs = this._progressCbs.filter(c => c !== cb); };
  }

  private _notifyProgress(loaded: number, total: number, url: string): void {
    for (const cb of this._progressCbs) cb(loaded, total, url);
  }

  /** Load a single asset, using the internal cache. */
  async load<T = unknown>(url: string, type: AssetType): Promise<LoadedAsset<T>> {
    if (this._cache.has(url)) {
      return this._cache.get(url) as LoadedAsset<T>;
    }

    const data = await this._fetch<T>(url, type);
    const entry: LoadedAsset<T> = { url, type, data };
    this._cache.set(url, entry);
    return entry;
  }

  /** Load multiple assets in parallel with aggregate progress. */
  async loadAll<T = unknown>(entries: AssetEntry[]): Promise<LoadedAsset<T>[]> {
    const total = entries.length;
    let loaded = 0;
    const results = await Promise.all(
      entries.map(async (e) => {
        const asset = await this.load<T>(e.url, e.type);
        loaded++;
        this._notifyProgress(loaded, total, e.url);
        return asset;
      })
    );
    return results;
  }

  /** Check whether a URL is already cached. */
  isCached(url: string): boolean {
    return this._cache.has(url);
  }

  /** Evict a single entry from the cache. */
  evict(url: string): void {
    this._cache.delete(url);
  }

  /** Clear the entire cache. */
  clearCache(): void {
    this._cache.clear();
  }

  private async _fetch<T>(url: string, type: AssetType): Promise<T> {
    if (type === "image") {
      return await new Promise<T>((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img as unknown as T);
        img.onerror = (e) => reject(new Error(`Failed to load image: ${url} — ${e}`));
        img.src = url;
      });
    }

    const res = await fetch(url);
    if (!res.ok) throw new Error(`[AssetLoader] HTTP ${res.status} for ${url}`);

    switch (type) {
      case "json":   return (await res.json()) as T;
      case "text":   return (await res.text()) as unknown as T;
      case "blob":   return (await res.blob()) as unknown as T;
      case "binary": return (await res.arrayBuffer()) as unknown as T;
      case "audio":
      case "video":  return (await res.blob()) as unknown as T;
      default:       return (await res.text()) as unknown as T;
    }
  }
}

// Singleton instance
export const assetLoader = new AssetLoader();

// ---------------------------------------------------------------------------
// AssetList
// ---------------------------------------------------------------------------

export interface AssetDescriptor {
  id: string;
  url: string;
  type: AssetType;
  meta?: Record<string, unknown>;
}

export class AssetList {
  private _registry = new Map<string, AssetDescriptor>();

  /** Register an asset by id. */
  register(descriptor: AssetDescriptor): void {
    this._registry.set(descriptor.id, descriptor);
  }

  /** Register multiple assets at once. */
  registerAll(descriptors: AssetDescriptor[]): void {
    for (const d of descriptors) this.register(d);
  }

  /** Get all registered descriptors. */
  getAll(): AssetDescriptor[] {
    return Array.from(this._registry.values());
  }

  /** Resolve a registered id to its URL, or return the string as-is if it looks like a URL. */
  resolve(idOrUrl: string): string {
    if (this._registry.has(idOrUrl)) {
      return this._registry.get(idOrUrl)!.url;
    }
    if (idOrUrl.startsWith("http") || idOrUrl.startsWith("/") || idOrUrl.startsWith("./")) {
      return idOrUrl;
    }
    throw new Error(`[AssetList] Cannot resolve "${idOrUrl}" — not registered and not a URL.`);
  }

  /** Get a descriptor by id. */
  get(id: string): AssetDescriptor | undefined {
    return this._registry.get(id);
  }

  has(id: string): boolean {
    return this._registry.has(id);
  }

  unregister(id: string): void {
    this._registry.delete(id);
  }
}

// Singleton instance
export const assetList = new AssetList();

// ---------------------------------------------------------------------------
// CMSData — fetch + cache with TTL
// ---------------------------------------------------------------------------

export interface CMSOptions {
  ttl?: number; // milliseconds, default 60_000
  headers?: Record<string, string>;
}

interface CMSCacheEntry<T> {
  data: T;
  expires: number;
}

export class CMSData {
  private _cache = new Map<string, CMSCacheEntry<unknown>>();
  private _ttl: number;
  private _headers: Record<string, string>;

  constructor(opts: CMSOptions = {}) {
    this._ttl = opts.ttl ?? 60_000;
    this._headers = opts.headers ?? {};
  }

  /**
   * Fetch from a CMS endpoint. Results are cached for `ttl` ms.
   * Pass `force: true` to bypass the cache.
   */
  async fetch<T = unknown>(endpoint: string, force = false): Promise<T> {
    if (!force && this._cache.has(endpoint)) {
      const entry = this._cache.get(endpoint) as CMSCacheEntry<T>;
      if (Date.now() < entry.expires) {
        return entry.data;
      }
    }

    const res = await fetch(endpoint, { headers: this._headers });
    if (!res.ok) throw new Error(`[CMSData] HTTP ${res.status} for ${endpoint}`);
    const data = (await res.json()) as T;

    this._cache.set(endpoint, { data, expires: Date.now() + this._ttl });
    return data;
  }

  /** Invalidate a specific endpoint. */
  invalidate(endpoint: string): void {
    this._cache.delete(endpoint);
  }

  /** Clear all cached responses. */
  clearCache(): void {
    this._cache.clear();
  }

  /** Update the TTL for future fetches. */
  setTTL(ms: number): void {
    this._ttl = ms;
  }
}

// Default singleton
export const cmsData = new CMSData();

// ---------------------------------------------------------------------------
// Config — URL param reading + runtime modification
// ---------------------------------------------------------------------------

export const Config = (() => {
  const _overrides = new Map<string, string>();

  /** Read a value from URL search params, with optional runtime override and fallback. */
  function get(key: string, fallback?: string): string | undefined {
    if (_overrides.has(key)) return _overrides.get(key);
    if (typeof location !== "undefined") {
      const params = new URLSearchParams(location.search);
      if (params.has(key)) return params.get(key)!;
    }
    return fallback;
  }

  /** Returns the value coerced to boolean ("1" | "true" | "yes" → true). */
  function getBool(key: string, fallback = false): boolean {
    const v = get(key);
    if (v === undefined) return fallback;
    return ["1", "true", "yes", "on"].includes(v.toLowerCase());
  }

  /** Returns the value coerced to a number, or the fallback. */
  function getNumber(key: string, fallback = 0): number {
    const v = get(key);
    if (v === undefined) return fallback;
    const n = parseFloat(v);
    return isNaN(n) ? fallback : n;
  }

  /** Set a runtime override that takes precedence over URL params. */
  function set(key: string, value: string): void {
    _overrides.set(key, value);
  }

  /** Remove a runtime override (URL param will be used again). */
  function remove(key: string): void {
    _overrides.delete(key);
  }

  /** Dump all effective values (overrides + URL params). */
  function dump(): Record<string, string> {
    const result: Record<string, string> = {};
    if (typeof location !== "undefined") {
      const params = new URLSearchParams(location.search);
      for (const [k, v] of params.entries()) result[k] = v;
    }
    for (const [k, v] of _overrides.entries()) result[k] = v;
    return result;
  }

  return { get, getBool, getNumber, set, remove, dump };
})();
