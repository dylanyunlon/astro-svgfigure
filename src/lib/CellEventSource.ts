/**
 * CellEventSource.ts — Apollo DataNotifier → SSE → CellInstanceManager
 *
 * Connects to GET /api/cell-events (text/event-stream) and applies every
 * incoming `cell_update` event to the live CellInstanceManager so the
 * renderer reflects pipeline state changes in real-time without polling.
 *
 * Usage:
 *   import { CellEventSource } from '$lib/CellEventSource';
 *   import type { CellInstanceManager } from '$lib/renderer/CellInstanceManager';
 *
 *   // after mgr is constructed and initial loadFromDescriptors() called:
 *   const src = new CellEventSource(mgr);
 *   src.connect();
 *   // later:
 *   src.disconnect();
 *
 * SSE event format (from server.py api_cell_events):
 *   event: cell_update
 *   data: { "cell_id": "self_attn", "params": { ...CellParamsJson... } }
 *
 * Reconnect strategy:
 *   The browser's native EventSource already reconnects automatically after
 *   network errors.  We add an exponential-backoff wrapper that recreates the
 *   EventSource after repeated onerror events (e.g. server restart) so the
 *   frontend never requires a manual page refresh during a long training run.
 *
 *   Backoff sequence: 1s → 2s → 4s → 8s → 16s → 30s (capped).
 */

import type { CellInstanceManager, CellParamsJson } from './renderer/CellInstanceManager';

// ── Types ────────────────────────────────────────────────────────────────────

export interface CellUpdateEvent {
  cell_id: string;
  params: CellParamsJson;
}

export type CellEventListener = (event: CellUpdateEvent) => void;

// ── Constants ────────────────────────────────────────────────────────────────

const SSE_URL = '/api/cell-events';
const BACKOFF_INITIAL_MS = 1_000;
const BACKOFF_MAX_MS = 30_000;
const ERROR_THRESHOLD = 3;   // consecutive errors before exponential backoff kicks in

// ── CellEventSource ──────────────────────────────────────────────────────────

export class CellEventSource {
  private _mgr: CellInstanceManager | null;
  private _es: EventSource | null = null;
  private _connected = false;
  private _destroyed = false;

  /** Extra listeners beyond the CellInstanceManager update. */
  private _listeners: Set<CellEventListener> = new Set();

  private _errorCount = 0;
  private _backoffMs = BACKOFF_INITIAL_MS;
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  /**
   * @param mgr  The CellInstanceManager to update on each event.
   *             Pass null if you only want to use the listener API.
   */
  constructor(mgr: CellInstanceManager | null = null) {
    this._mgr = mgr;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  get connected(): boolean {
    return this._connected;
  }

  /**
   * Open the SSE connection.  Safe to call multiple times — no-ops if already
   * connected.  Throws if disconnect() has been called.
   */
  connect(): void {
    if (this._destroyed) throw new Error('CellEventSource: already destroyed');
    if (this._es) return;
    this._open();
  }

  /**
   * Close the SSE connection permanently.  The instance cannot be reused.
   */
  disconnect(): void {
    this._destroyed = true;
    this._clearReconnect();
    this._close();
  }

  /**
   * Register an additional callback invoked for every cell_update event,
   * alongside the automatic CellInstanceManager.updateCell() call.
   */
  addListener(fn: CellEventListener): void {
    this._listeners.add(fn);
  }

  removeListener(fn: CellEventListener): void {
    this._listeners.delete(fn);
  }

  /**
   * Swap out the managed CellInstanceManager at runtime (e.g. after a full
   * scene reload rebuilds the manager).
   */
  setManager(mgr: CellInstanceManager | null): void {
    this._mgr = mgr;
  }

  // ── Private — EventSource lifecycle ────────────────────────────────────────

  private _open(): void {
    if (typeof EventSource === 'undefined') {
      console.warn('[CellEventSource] EventSource not available in this environment');
      return;
    }

    const es = new EventSource(SSE_URL);
    this._es = es;

    es.addEventListener('cell_update', (ev: MessageEvent) => {
      this._onCellUpdate(ev);
    });

    es.onopen = () => {
      this._connected = true;
      this._errorCount = 0;
      this._backoffMs = BACKOFF_INITIAL_MS;
      console.info('[CellEventSource] connected →', SSE_URL);
    };

    es.onerror = () => {
      this._connected = false;
      this._errorCount++;

      if (this._destroyed) return;

      // Native EventSource will attempt its own reconnect; we layer exponential
      // backoff on top for the case where the server is completely unreachable
      // and the native retry loop floods the console.
      if (this._errorCount >= ERROR_THRESHOLD) {
        // Suppress the native retry by closing and reopening ourselves.
        this._close();
        const delay = Math.min(this._backoffMs, BACKOFF_MAX_MS);
        console.warn(
          `[CellEventSource] error #${this._errorCount} — reconnecting in ${delay}ms`,
        );
        this._reconnectTimer = setTimeout(() => {
          if (!this._destroyed) {
            this._backoffMs = Math.min(this._backoffMs * 2, BACKOFF_MAX_MS);
            this._open();
          }
        }, delay);
      }
    };
  }

  private _close(): void {
    if (this._es) {
      this._es.close();
      this._es = null;
    }
    this._connected = false;
  }

  private _clearReconnect(): void {
    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
  }

  // ── Private — event handling ────────────────────────────────────────────────

  private _onCellUpdate(ev: MessageEvent): void {
    let update: CellUpdateEvent;
    try {
      update = JSON.parse(ev.data) as CellUpdateEvent;
    } catch (e) {
      console.error('[CellEventSource] failed to parse cell_update payload:', ev.data, e);
      return;
    }

    const { cell_id, params } = update;

    // ── 1. Update CellInstanceManager ─────────────────────────────────────
    if (this._mgr) {
      try {
        this._mgr.updateCell(cell_id, {
          bbox:        params.bbox,
          opacity:     params.opacity,
          fill_color:  params.fill_color,
        });
      } catch (e) {
        console.warn(`[CellEventSource] updateCell("${cell_id}") failed:`, e);
      }
    }

    // ── 2. Notify extra listeners ──────────────────────────────────────────
    for (const fn of this._listeners) {
      try {
        fn(update);
      } catch (e) {
        console.error('[CellEventSource] listener threw:', e);
      }
    }
  }
}

// ── Singleton convenience ─────────────────────────────────────────────────────
// Many components may want to share the same EventSource rather than each
// opening their own connection.  The module-level singleton is lazily created
// and never auto-connects; call .connect() explicitly after wiring up a mgr.

let _singleton: CellEventSource | null = null;

export function getCellEventSource(): CellEventSource {
  if (!_singleton || (_singleton as any)._destroyed) {
    _singleton = new CellEventSource();
  }
  return _singleton;
}
