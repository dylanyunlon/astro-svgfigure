/**
 * ProgramFactory.ts — 声明式 shader program 工厂
 *
 * Material → AstroProgram 的编译适配层。
 * 解决 cell-pubsub-loop 场景下频繁 species 切换导致的 shader 重编译问题:
 *
 *   cell_update event → species 变化 → 新 fragment source → 重编译?
 *
 * ProgramFactory 维护一个 (vertHash, fragHash) → AstroProgram 的缓存池,
 * 使得同一 species 的多个 cell 共享同一个编译好的 program。
 *
 * 生命周期:
 *   - get()      命中缓存或编译新 program
 *   - release()  引用计数 -1, 归零时自动 dispose
 *   - flush()    清空全部缓存（场景切换时）
 *
 * Usage:
 *   const factory = new ProgramFactory(renderer);
 *   const program = factory.get(vertSrc, fragSrc);
 *   // ... render ...
 *   factory.release(program);
 */

import { AstroRenderer } from '../AstroRenderer.js';
import { AstroProgram } from '../AstroProgram.js';

// ── CacheEntry ──────────────────────────────────────────────────────────────

interface CacheEntry {
  program: AstroProgram;
  refCount: number;
  vertHash: number;
  fragHash: number;
}

// ── ProgramFactory ──────────────────────────────────────────────────────────

export class ProgramFactory {
  private _renderer: AstroRenderer;
  /** key = combined hash string, value = CacheEntry */
  private _cache = new Map<string, CacheEntry>();

  constructor(renderer: AstroRenderer) {
    this._renderer = renderer;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Get (or compile) a shader program for the given GLSL sources.
   * Increments the reference count — call release() when done.
   */
  get(vertSrc: string, fragSrc: string): AstroProgram {
    const key = _cacheKey(vertSrc, fragSrc);
    let entry = this._cache.get(key);

    if (entry) {
      entry.refCount++;
      return entry.program;
    }

    // Compile new program
    const program = new AstroProgram(this._renderer, vertSrc, fragSrc);

    entry = {
      program,
      refCount: 1,
      vertHash: _djb2(vertSrc),
      fragHash: _djb2(fragSrc),
    };
    this._cache.set(key, entry);

    return program;
  }

  /**
   * Decrement the reference count for a program.
   * When the count reaches zero the GPU resources are freed.
   * Returns true if the program was disposed.
   */
  release(program: AstroProgram): boolean {
    for (const [key, entry] of this._cache) {
      if (entry.program === program) {
        entry.refCount--;
        if (entry.refCount <= 0) {
          entry.program.dispose();
          this._cache.delete(key);
          return true;
        }
        return false;
      }
    }
    // Program not in cache — caller manages it externally
    return false;
  }

  /**
   * Check if a program compiled from these sources already exists.
   */
  has(vertSrc: string, fragSrc: string): boolean {
    return this._cache.has(_cacheKey(vertSrc, fragSrc));
  }

  /**
   * Get the current reference count for a program.
   * Returns 0 if not cached.
   */
  refCount(program: AstroProgram): number {
    for (const entry of this._cache.values()) {
      if (entry.program === program) return entry.refCount;
    }
    return 0;
  }

  /** Number of unique programs in the cache. */
  get size(): number {
    return this._cache.size;
  }

  /**
   * Dispose all cached programs and clear the cache.
   * Use on scene transition or full cleanup.
   */
  flush(): void {
    for (const entry of this._cache.values()) {
      entry.program.dispose();
    }
    this._cache.clear();
  }

  /**
   * Dispose programs with zero references.
   * Use as a periodic GC pass to reclaim GPU memory from unused species shaders.
   */
  gc(): number {
    let freed = 0;
    for (const [key, entry] of this._cache) {
      if (entry.refCount <= 0) {
        entry.program.dispose();
        this._cache.delete(key);
        freed++;
      }
    }
    return freed;
  }

  /**
   * Debug: list all cached entries with their ref counts.
   */
  dump(): Array<{ key: string; refCount: number; vertHash: number; fragHash: number }> {
    const out: Array<{ key: string; refCount: number; vertHash: number; fragHash: number }> = [];
    for (const [key, entry] of this._cache) {
      out.push({
        key,
        refCount: entry.refCount,
        vertHash: entry.vertHash,
        fragHash: entry.fragHash,
      });
    }
    return out;
  }

  /** Dispose = flush. */
  dispose(): void {
    this.flush();
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/** djb2 hash → unsigned 32-bit integer */
function _djb2(str: string): number {
  let h = 5381;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) + h + str.charCodeAt(i)) | 0;
  }
  return h >>> 0;
}

/** Combined cache key from two source strings. */
function _cacheKey(vert: string, frag: string): string {
  return `${_djb2(vert).toString(36)}_${_djb2(frag).toString(36)}`;
}
