/**
 * src/lib/sph/collision/CollisionEvents.ts
 *
 * Collision Event System for the SPH rigid-body pipeline.
 *
 * Provides three lifecycle callbacks per body-pair:
 *  - onCollisionEnter  — fired the first frame two bodies overlap
 *  - onCollisionStay   — fired every subsequent frame they remain overlapping
 *  - onCollisionExit   — fired the first frame they stop overlapping
 *
 * Architecture
 * ────────────
 *  CollisionEventDispatcher
 *    • Holds a "previous-frame" contact set (Set<string> of pairKeys).
 *    • Each step() receives the current-frame contact pairs.
 *    • Compares prev vs current → emits Enter / Stay / Exit events into a
 *      typed EventQueue<CollisionEvent>.
 *    • Registered callback listeners are invoked synchronously after the diff.
 *    • The queue can also be drained manually (e.g. in a game update loop).
 *
 * Usage
 * ─────
 *   const dispatcher = new CollisionEventDispatcher();
 *
 *   dispatcher.onCollisionEnter((e) => console.log('enter', e.bodyA, e.bodyB));
 *   dispatcher.onCollisionStay ((e) => console.log('stay',  e.bodyA, e.bodyB));
 *   dispatcher.onCollisionExit ((e) => console.log('exit',  e.bodyA, e.bodyB));
 *
 *   // Inside your step loop — pass the active contact pairs this frame:
 *   dispatcher.update(currentContacts);          // emits events immediately
 *
 *   // Or drain the event queue manually:
 *   dispatcher.drainQueue((event) => handle(event));
 */

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Public types
// ─────────────────────────────────────────────────────────────────────────────

/** Phase of the collision lifecycle. */
export type CollisionPhase = 'enter' | 'stay' | 'exit';

/** Snapshot of contact information carried with every event. */
export interface CollisionContactInfo {
  /** Contact normal pointing from bodyA toward bodyB. */
  normal: { x: number; y: number };
  /** Penetration depth (> 0 means overlapping). */
  depth: number;
  /** World-space contact point on bodyA surface. */
  pointA: { x: number; y: number };
  /** World-space contact point on bodyB surface. */
  pointB: { x: number; y: number };
}

/** A single collision event. */
export interface CollisionEvent {
  /** Lifecycle phase. */
  phase: CollisionPhase;
  /** ID of the first body in the pair (always the smaller id). */
  bodyA: number;
  /** ID of the second body in the pair. */
  bodyB: number;
  /** Contact details — available for Enter and Stay; null for Exit. */
  contact: CollisionContactInfo | null;
  /** Simulation timestamp (seconds) when the event was generated. */
  time: number;
}

/** Minimal contact-pair description fed into the dispatcher each frame. */
export interface ActiveContactPair {
  bodyA: number;
  bodyB: number;
  contact: CollisionContactInfo;
}

/** Callback type for collision events. */
export type CollisionCallback = (event: CollisionEvent) => void;

// ─────────────────────────────────────────────────────────────────────────────
// 2.  EventQueue  (generic, reusable)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A simple FIFO queue for typed events.
 * Events are pushed each frame and can be drained manually or via listeners.
 */
export class EventQueue<T> {
  private _items: T[] = [];

  /** Add an event to the back of the queue. */
  push(item: T): void {
    this._items.push(item);
  }

  /**
   * Drain all queued events by invoking `cb` for each, then clear the queue.
   * Drain is destructive — events are consumed.
   */
  drain(cb: (item: T) => void): void {
    for (const item of this._items) cb(item);
    this._items.length = 0;
  }

  /**
   * Peek at all pending events without consuming them.
   * Returns a read-only snapshot.
   */
  peek(): readonly T[] {
    return this._items;
  }

  /** Number of pending events. */
  get size(): number {
    return this._items.length;
  }

  /** Discard all pending events without processing them. */
  clear(): void {
    this._items.length = 0;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  CollisionCache  (prev-frame vs current-frame set diffing)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Canonical ordering: always store pair as (min(a,b), max(a,b)) so
 * (A,B) and (B,A) map to the same key.
 */
export function makePairKey(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

/**
 * Stores the set of active contact pairs from the previous simulation frame,
 * keyed by canonical pair string.  Used by the dispatcher to diff prev vs
 * current and derive Enter / Stay / Exit phases.
 */
export class CollisionCache {
  /** Map: pairKey → contact info snapshot from the last frame. */
  private _prev: Map<string, CollisionContactInfo> = new Map();

  /** Swap the "previous" cache with a freshly-computed current set. */
  update(current: Map<string, CollisionContactInfo>): {
    entered: string[];
    stayed: string[];
    exited: string[];
  } {
    const entered: string[] = [];
    const stayed: string[] = [];
    const exited: string[] = [];

    // Entries in current but not in prev → Enter
    // Entries in both              → Stay
    for (const key of current.keys()) {
      if (this._prev.has(key)) {
        stayed.push(key);
      } else {
        entered.push(key);
      }
    }

    // Entries in prev but not in current → Exit
    for (const key of this._prev.keys()) {
      if (!current.has(key)) {
        exited.push(key);
      }
    }

    // Replace prev with current
    this._prev = current;

    return { entered, stayed, exited };
  }

  /** Number of pairs currently active (from previous frame). */
  get size(): number {
    return this._prev.size;
  }

  /** Check if a specific pair was active last frame. */
  has(a: number, b: number): boolean {
    return this._prev.has(makePairKey(a, b));
  }

  /** Retrieve the contact info for a pair (from last frame), if present. */
  getContact(a: number, b: number): CollisionContactInfo | undefined {
    return this._prev.get(makePairKey(a, b));
  }

  /**
   * Remove all pairs that involve `bodyId` from the cache.
   * Call this when a body is removed from the world so it won't generate
   * spurious Exit events for a body that no longer exists.
   */
  evictBody(bodyId: number): void {
    const toDelete: string[] = [];
    for (const key of this._prev.keys()) {
      const sep = key.indexOf(':');
      const a = parseInt(key.slice(0, sep), 10);
      const b = parseInt(key.slice(sep + 1), 10);
      if (a === bodyId || b === bodyId) toDelete.push(key);
    }
    for (const key of toDelete) this._prev.delete(key);
  }

  /** Wipe the cache (use when resetting or removing bodies). */
  clear(): void {
    this._prev.clear();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  CollisionEventDispatcher  (main public class)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * CollisionEventDispatcher
 *
 * Integrate into a rigid-body world by calling `update(contacts, time)` once
 * per simulation step, after the narrow phase has resolved contacts.
 *
 * The dispatcher:
 *  1. Diffs current contacts against the previous-frame cache.
 *  2. Classifies each pair as Enter / Stay / Exit.
 *  3. Pushes CollisionEvent objects onto the internal EventQueue.
 *  4. Immediately fires any registered per-phase callbacks.
 */
export class CollisionEventDispatcher {
  // ── Internal state ─────────────────────────────────────────────────────────
  private _cache   = new CollisionCache();
  private _queue   = new EventQueue<CollisionEvent>();

  // ── Per-phase callback lists ───────────────────────────────────────────────
  private _enterCbs: CollisionCallback[] = [];
  private _stayCbs:  CollisionCallback[] = [];
  private _exitCbs:  CollisionCallback[] = [];

  // ── Callback registration ──────────────────────────────────────────────────

  /**
   * Register a listener for Enter events (first frame of contact).
   * Returns an unsubscribe function.
   */
  onCollisionEnter(cb: CollisionCallback): () => void {
    this._enterCbs.push(cb);
    return () => {
      this._enterCbs = this._enterCbs.filter(fn => fn !== cb);
    };
  }

  /**
   * Register a listener for Stay events (contact persists across frames).
   * Returns an unsubscribe function.
   */
  onCollisionStay(cb: CollisionCallback): () => void {
    this._stayCbs.push(cb);
    return () => {
      this._stayCbs = this._stayCbs.filter(fn => fn !== cb);
    };
  }

  /**
   * Register a listener for Exit events (last frame of contact).
   * Returns an unsubscribe function.
   */
  onCollisionExit(cb: CollisionCallback): () => void {
    this._exitCbs.push(cb);
    return () => {
      this._exitCbs = this._exitCbs.filter(fn => fn !== cb);
    };
  }

  // ── Main update — call once per simulation step ───────────────────────────

  /**
   * Compute Enter / Stay / Exit events from the current-frame contacts.
   *
   * @param contacts  All active contact pairs this frame (from narrow phase).
   * @param time      Current simulation time in seconds (monotone).
   */
  update(contacts: ActiveContactPair[], time: number): void {
    // Build current-frame map
    const current = new Map<string, CollisionContactInfo>();
    for (const pair of contacts) {
      const key = makePairKey(pair.bodyA, pair.bodyB);
      current.set(key, pair.contact);
    }

    // Diff against previous frame
    const { entered, stayed, exited } = this._cache.update(current);

    // Helper: parse a pairKey back into [bodyA, bodyB]
    const parseKey = (key: string): [number, number] => {
      const idx = key.indexOf(':');
      return [parseInt(key.slice(0, idx), 10), parseInt(key.slice(idx + 1), 10)];
    };

    // Emit Enter events
    for (const key of entered) {
      const [a, b] = parseKey(key);
      const event: CollisionEvent = {
        phase: 'enter',
        bodyA: a,
        bodyB: b,
        contact: current.get(key) ?? null,
        time,
      };
      this._queue.push(event);
      for (const cb of this._enterCbs) cb(event);
    }

    // Emit Stay events
    for (const key of stayed) {
      const [a, b] = parseKey(key);
      const event: CollisionEvent = {
        phase: 'stay',
        bodyA: a,
        bodyB: b,
        contact: current.get(key) ?? null,
        time,
      };
      this._queue.push(event);
      for (const cb of this._stayCbs) cb(event);
    }

    // Emit Exit events (contact info not available — use null)
    for (const key of exited) {
      const [a, b] = parseKey(key);
      const event: CollisionEvent = {
        phase: 'exit',
        bodyA: a,
        bodyB: b,
        contact: null,
        time,
      };
      this._queue.push(event);
      for (const cb of this._exitCbs) cb(event);
    }
  }

  // ── Queue access ──────────────────────────────────────────────────────────

  /**
   * Drain all pending events from the queue, invoking `cb` for each.
   * Useful for polling-style consumers who prefer not to use callbacks.
   */
  drainQueue(cb: (event: CollisionEvent) => void): void {
    this._queue.drain(cb);
  }

  /**
   * Peek at all pending events without consuming them.
   */
  peekQueue(): readonly CollisionEvent[] {
    return this._queue.peek();
  }

  /** Number of pending unprocessed events in the queue. */
  get pendingEventCount(): number {
    return this._queue.size;
  }

  // ── Cache introspection ───────────────────────────────────────────────────

  /** Number of currently active (staying) contact pairs. */
  get activeContactCount(): number {
    return this._cache.size;
  }

  /** Returns true if bodies `a` and `b` were in contact last frame. */
  isContacting(a: number, b: number): boolean {
    return this._cache.has(a, b);
  }

  /**
   * Reset the dispatcher: clear the cache, queue, and all registered callbacks.
   * Use when the world is re-initialized.
   */
  reset(): void {
    this._cache.clear();
    this._queue.clear();
    this._enterCbs.length = 0;
    this._stayCbs.length  = 0;
    this._exitCbs.length  = 0;
  }

  /**
   * Evict a body from the cache (call when `removeBody` is used).
   * Any pairs involving `bodyId` will be treated as new Enter events
   * if that body is re-added later.
   */
  evictBody(bodyId: number): void {
    // We rebuild the cache without pairs that mention bodyId.
    // CollisionCache doesn't expose per-body removal, so we do a full rebuild
    // from its current peek — but since CollisionCache is private-data we
    // instead signal via a cache-invalidation helper.
    this._cache.evictBody(bodyId);
  }
}


