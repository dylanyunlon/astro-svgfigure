/**
 * epoch-physics-recorder.ts — Record & play back epoch-level physics snapshots  (M768)
 *
 * Captures a binary snapshot of the SPH World at the end of each epoch using
 * the compact binary format from `world-serializer`.  The resulting `Recording`
 * object stores an ordered timeline of snapshots that can be played back at any
 * normalised time `t ∈ [0, 1]` — yielding the nearest epoch's reconstructed
 * World state.
 *
 * Design notes
 * ────────────
 *  • Snapshots are stored as raw `ArrayBuffer`s (the output of `serializeWorld`)
 *    rather than live `World` objects.  This keeps memory flat and avoids holding
 *    references to solver / spatial-hash internals that would prevent GC.
 *
 *  • `playback(t)` deserializes on the fly.  For tight playback loops the caller
 *    should cache the returned `World` and only call `playback` again when the
 *    epoch index changes (use `epochAt(t)` to check).
 *
 *  • The recorder is intentionally stateless w.r.t. the simulation loop: it does
 *    not subscribe to SSE or drive the stepper.  The owner of the epoch loop
 *    calls `onEpochComplete(epochNum, world)` at the appropriate time.
 *
 * Usage
 * ─────
 *   import { EpochPhysicsRecorder } from '$lib/sph/epoch-physics-recorder';
 *
 *   const recorder = new EpochPhysicsRecorder();
 *   recorder.startRecording();
 *
 *   // inside your epoch loop:
 *   recorder.onEpochComplete(epochNum, world);
 *
 *   const recording = recorder.stopRecording();
 *   const midWorld  = recording.playback(0.5);   // half-way through
 *   const epochs    = recording.epochCount;       // total captured epochs
 */

import type { World } from './world-stepper';
import { serializeWorld, deserializeWorld } from './world-serializer';

// ─── EpochSnapshot ──────────────────────────────────────────────────────────

/** A single epoch's captured state. */
export interface EpochSnapshot {
  /** The epoch number as reported by the caller (monotonically increasing). */
  epoch: number;
  /** Wall-clock timestamp (ms) at capture time. */
  capturedAt: number;
  /** Binary-serialized World produced by `serializeWorld`. */
  buffer: ArrayBuffer;
}

// ─── Recording ──────────────────────────────────────────────────────────────

/**
 * Immutable recording produced by `EpochPhysicsRecorder.stopRecording()`.
 *
 * Contains an ordered sequence of epoch snapshots and provides time-based
 * playback via `playback(t)`.
 */
export class Recording {
  /** Ordered epoch snapshots (ascending by epoch number). */
  readonly snapshots: ReadonlyArray<EpochSnapshot>;

  /** Wall-clock timestamp (ms) when the recording started. */
  readonly startedAt: number;

  /** Wall-clock timestamp (ms) when the recording ended. */
  readonly endedAt: number;

  constructor(
    snapshots: EpochSnapshot[],
    startedAt: number,
    endedAt: number,
  ) {
    this.snapshots = Object.freeze([...snapshots]);
    this.startedAt = startedAt;
    this.endedAt   = endedAt;
  }

  // ── Accessors ───────────────────────────────────────────────────────────

  /** Number of captured epochs. */
  get epochCount(): number {
    return this.snapshots.length;
  }

  /** Total wall-clock duration of the recording in milliseconds. */
  get durationMs(): number {
    return this.endedAt - this.startedAt;
  }

  /** `true` when the recording contains zero snapshots. */
  get isEmpty(): boolean {
    return this.snapshots.length === 0;
  }

  /** Epoch number of the first snapshot, or `undefined` if empty. */
  get firstEpoch(): number | undefined {
    return this.snapshots.length > 0 ? this.snapshots[0].epoch : undefined;
  }

  /** Epoch number of the last snapshot, or `undefined` if empty. */
  get lastEpoch(): number | undefined {
    return this.snapshots.length > 0
      ? this.snapshots[this.snapshots.length - 1].epoch
      : undefined;
  }

  // ── Playback ────────────────────────────────────────────────────────────

  /**
   * Return the epoch index (into `snapshots`) that corresponds to a
   * normalised time value `t ∈ [0, 1]`.
   *
   * - `t = 0`  → first epoch (index 0)
   * - `t = 1`  → last epoch  (index epochCount − 1)
   * - values in between are linearly mapped and floored to the nearest
   *   epoch boundary.
   *
   * Returns `−1` if the recording is empty.
   */
  epochAt(t: number): number {
    const n = this.snapshots.length;
    if (n === 0) return -1;
    if (n === 1) return 0;

    const clamped = Math.max(0, Math.min(1, t));
    const index = Math.round(clamped * (n - 1));
    return Math.max(0, Math.min(n - 1, index));
  }

  /**
   * Deserialize and return the `World` at normalised time `t ∈ [0, 1]`.
   *
   * The returned `World` has fully populated particles and rigid bodies but
   * transient solver state is zeroed — see `deserializeWorld` in
   * `world-serializer.ts` for details.
   *
   * Throws if the recording is empty.
   */
  playback(t: number): World {
    if (this.snapshots.length === 0) {
      throw new Error('[Recording] Cannot play back an empty recording');
    }
    const idx = this.epochAt(t);
    return deserializeWorld(this.snapshots[idx].buffer);
  }

  /**
   * Deserialize the snapshot at a specific epoch index.
   *
   * Throws if `index` is out of range.
   */
  playbackAtIndex(index: number): World {
    if (index < 0 || index >= this.snapshots.length) {
      throw new RangeError(
        `[Recording] Index ${index} out of range [0, ${this.snapshots.length - 1}]`,
      );
    }
    return deserializeWorld(this.snapshots[index].buffer);
  }

  /**
   * Deserialize the snapshot whose `epoch` field matches the given number.
   *
   * Returns `null` if no such epoch was recorded.
   */
  playbackAtEpoch(epoch: number): World | null {
    const snap = this.snapshots.find((s) => s.epoch === epoch);
    return snap ? deserializeWorld(snap.buffer) : null;
  }

  // ── Aggregate stats ─────────────────────────────────────────────────────

  /** Total bytes consumed by all snapshot buffers. */
  get totalBytes(): number {
    let sum = 0;
    for (const s of this.snapshots) sum += s.buffer.byteLength;
    return sum;
  }
}

// ─── EpochPhysicsRecorder ───────────────────────────────────────────────────

/**
 * Records epoch-level physics world snapshots.
 *
 * Lifecycle: `startRecording()` → N × `onEpochComplete(…)` → `stopRecording()`.
 *
 * The recorder can be reused: calling `startRecording()` again after
 * `stopRecording()` begins a fresh capture.
 */
export class EpochPhysicsRecorder {
  private _recording  = false;
  private _snapshots: EpochSnapshot[] = [];
  private _startedAt  = 0;

  // ── Public API ──────────────────────────────────────────────────────────

  /** Whether the recorder is currently capturing. */
  get isRecording(): boolean {
    return this._recording;
  }

  /** Number of snapshots captured so far in the current (or last) session. */
  get capturedCount(): number {
    return this._snapshots.length;
  }

  /**
   * Begin a new recording session.
   *
   * Any previously buffered snapshots are discarded.  If already recording,
   * calling `startRecording()` again restarts from scratch.
   */
  startRecording(): void {
    this._snapshots = [];
    this._startedAt = Date.now();
    this._recording = true;
  }

  /**
   * Capture a world snapshot for the given epoch.
   *
   * Should be called once at the end of each epoch.  If the recorder is not
   * in recording mode the call is silently ignored, making it safe to leave
   * the hook wired up even when recording is off.
   *
   * @param epochNum  The epoch number (caller-defined, should be monotonic).
   * @param world     The current `World` whose physical state will be
   *                  serialized via `serializeWorld`.
   */
  onEpochComplete(epochNum: number, world: World): void {
    if (!this._recording) return;

    const buffer = serializeWorld(world);
    this._snapshots.push({
      epoch:      epochNum,
      capturedAt: Date.now(),
      buffer,
    });
  }

  /**
   * Finalise the recording and return an immutable `Recording` object.
   *
   * The recorder transitions out of recording mode.  The internal snapshot
   * buffer is transferred to the `Recording` (the recorder's buffer is
   * cleared), so the `Recording` solely owns the data.
   *
   * Throws if the recorder is not currently recording.
   */
  stopRecording(): Recording {
    if (!this._recording) {
      throw new Error(
        '[EpochPhysicsRecorder] stopRecording() called while not recording',
      );
    }

    this._recording = false;
    const endedAt = Date.now();

    const recording = new Recording(
      this._snapshots,
      this._startedAt,
      endedAt,
    );

    // Transfer ownership — clear local buffer.
    this._snapshots = [];
    return recording;
  }
}
