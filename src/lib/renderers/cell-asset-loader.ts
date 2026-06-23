/**
 * cell-asset-loader.ts — M827: Asset preloader for species SDF + MSDF
 *
 * Preloads species SDF shader textures and MSDF font atlases.
 * Provides progress callbacks and retry logic.
 */

export interface AssetManifest {
  sdfShaders: string[];      // e.g. ['cil-eye.frag', 'cil-bolt.frag', ...]
  msdfAtlases: string[];     // e.g. ['inter-msdf.png', 'inter-msdf.json']
  textures: string[];        // additional textures
}

export type ProgressCallback = (loaded: number, total: number, current: string) => void;

interface LoadTask {
  url: string;
  type: 'shader' | 'texture' | 'json';
  retries: number;
}

const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1000;

async function fetchWithRetry(url: string, retries: number = MAX_RETRIES): Promise<Response> {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return resp;
    } catch (err) {
      if (attempt === retries) throw err;
      await new Promise(r => setTimeout(r, RETRY_DELAY_MS * (attempt + 1)));
    }
  }
  throw new Error('unreachable');
}

export class CellAssetLoader {
  private loaded: Map<string, any> = new Map();
  private loading: Set<string> = new Set();
  private failed: Set<string> = new Set();

  /**
   * Preload all assets from manifest.
   * Calls onProgress for each loaded asset.
   * Returns when all assets are loaded or failed.
   */
  async preload(manifest: AssetManifest, onProgress?: ProgressCallback): Promise<{
    loaded: number; failed: string[];
  }> {
    const tasks: LoadTask[] = [
      ...manifest.sdfShaders.map(u => ({ url: u, type: 'shader' as const, retries: MAX_RETRIES })),
      ...manifest.msdfAtlases.map(u => ({ url: u, type: u.endsWith('.json') ? 'json' as const : 'texture' as const, retries: MAX_RETRIES })),
      ...manifest.textures.map(u => ({ url: u, type: 'texture' as const, retries: MAX_RETRIES })),
    ];

    const total = tasks.length;
    let loadedCount = 0;

    const promises = tasks.map(async (task) => {
      if (this.loaded.has(task.url)) {
        loadedCount++;
        onProgress?.(loadedCount, total, task.url);
        return;
      }

      this.loading.add(task.url);
      try {
        const resp = await fetchWithRetry(task.url, task.retries);
        let data: any;
        if (task.type === 'json') {
          data = await resp.json();
        } else if (task.type === 'shader') {
          data = await resp.text();
        } else {
          data = await resp.blob();
        }
        this.loaded.set(task.url, data);
        loadedCount++;
        onProgress?.(loadedCount, total, task.url);
      } catch {
        this.failed.add(task.url);
      } finally {
        this.loading.delete(task.url);
      }
    });

    await Promise.allSettled(promises);
    return { loaded: loadedCount, failed: [...this.failed] };
  }

  get(url: string): any | undefined {
    return this.loaded.get(url);
  }

  get stats() {
    return {
      loaded: this.loaded.size,
      loading: this.loading.size,
      failed: this.failed.size,
    };
  }
}
