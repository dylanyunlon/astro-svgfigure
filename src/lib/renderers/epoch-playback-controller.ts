/**
 * epoch-playback-controller.ts — M069: Theatre.js sequence playback 控制
 *
 * 暴露 play/pause/seekTo/setRate API，驱动 epoch 时间线动画。
 * 对接 EpochTimeline 的 Theatre.js sequence + EpochCellBridge 的 onFrame。
 *
 * UI 控制条由 pipeline/index.astro 渲染，本模块只提供逻辑层。
 */

export type PlaybackState = 'playing' | 'paused' | 'stopped';
export type PlaybackRate = 0.25 | 0.5 | 1 | 2 | 4;

export interface PlaybackCallbacks {
  /** Called when playback state changes */
  onStateChange?: (state: PlaybackState) => void;
  /** Called each frame with current epoch position (float, e.g. 2.7 = between epoch 2 and 3) */
  onPositionChange?: (position: number) => void;
  /** Called when playback reaches the last epoch */
  onComplete?: () => void;
}

export class EpochPlaybackController {
  private _state: PlaybackState = 'stopped';
  private _rate: PlaybackRate = 1;
  private _position = 0;
  private _maxEpoch = 0;
  private _callbacks: PlaybackCallbacks;

  /** Theatre.js sequence reference — injected by EpochTimeline */
  private _sequence: {
    play: (opts: { range?: [number, number]; rate?: number; iterationCount?: number }) => Promise<boolean>;
    pause: () => void;
    position: number;
  } | null = null;

  /** rAF handle for position tracking */
  private _rafId: number | null = null;

  /** Loop playback — when true, rewind to 0 instead of stopping at end */
  private _loop = false;

  /** Speed presets for cycleSpeed() */
  private _presets: PlaybackRate[] = [0.25, 0.5, 1, 2, 4];

  /** Registered epoch-change listeners */
  private _epochCb: Array<(epoch: number) => void> = [];

  /** Last integer epoch emitted, for dedup in onEpochChange */
  private _lastEpochInt = -1;

  constructor(maxEpoch: number, callbacks: PlaybackCallbacks = {}) {
    this._maxEpoch = maxEpoch;
    this._callbacks = callbacks;
  }

  /** Connect to Theatre.js sequence (called by EpochTimeline after init) */
  connectSequence(seq: typeof this._sequence): void {
    this._sequence = seq;
  }

  get state(): PlaybackState { return this._state; }
  get rate(): PlaybackRate { return this._rate; }
  get position(): number { return this._position; }
  get maxEpoch(): number { return this._maxEpoch; }
  get loop(): boolean { return this._loop; }

  // ── Playback controls ───────────────────────────────────────────────────

  async play(): Promise<void> {
    if (!this._sequence || this._state === 'playing') return;

    this._state = 'playing';
    this._callbacks.onStateChange?.('playing');
    this._startPositionTracking();

    const completed = await this._sequence.play({
      range: [this._position, this._maxEpoch],
      rate: this._rate,
      iterationCount: 1,
    });

    if (completed) {
      this._state = 'stopped';
      this._callbacks.onComplete?.();
      this._callbacks.onStateChange?.('stopped');
      this._stopPositionTracking();
    }
  }

  pause(): void {
    if (!this._sequence || this._state !== 'playing') return;
    this._sequence.pause();
    this._state = 'paused';
    this._callbacks.onStateChange?.('paused');
    this._stopPositionTracking();
  }

  /** Toggle play/pause */
  togglePlayPause(): void {
    if (this._state === 'playing') this.pause();
    else this.play();
  }

  /** Seek to a specific epoch position (can be fractional) */
  seekTo(epoch: number): void {
    const clamped = Math.max(0, Math.min(this._maxEpoch, epoch));
    this._position = clamped;
    if (this._sequence) {
      this._sequence.position = clamped;
    }
    this._callbacks.onPositionChange?.(clamped);
  }

  /** Set playback speed */
  setRate(rate: PlaybackRate): void {
    this._rate = rate;
    // If currently playing, restart at new rate
    if (this._state === 'playing') {
      this.pause();
      this.play();
    }
  }

  /** Step forward/backward by one epoch */
  stepForward(): void { this.seekTo(Math.floor(this._position) + 1); }
  stepBackward(): void { this.seekTo(Math.ceil(this._position) - 1); }

  /** Reset to start */
  reset(): void {
    this.pause();
    this.seekTo(0);
    this._state = 'stopped';
    this._callbacks.onStateChange?.('stopped');
  }

  /** Update max epoch count (e.g. when new epochs are generated) */
  setMaxEpoch(n: number): void { this._maxEpoch = n; }

  /** Enable / disable loop playback */
  setLoop(v: boolean): void { this._loop = v; }

  /** Cycle through speed presets: 0.25 → 0.5 → 1 → 2 → 4 → 0.25 … */
  cycleSpeed(): PlaybackRate {
    const idx = this._presets.indexOf(this._rate);
    const next = this._presets[(idx + 1) % this._presets.length];
    this.setRate(next);
    return next;
  }

  /** Register a callback fired whenever the integer epoch changes */
  onEpochChange(cb: (epoch: number) => void): () => void {
    this._epochCb.push(cb);
    return () => {
      this._epochCb = this._epochCb.filter((fn) => fn !== cb);
    };
  }

  // ── Position tracking ─────────────────────────────────────────────────

  private _startPositionTracking(): void {
    const tick = () => {
      if (this._sequence && this._state === 'playing') {
        this._position = this._sequence.position;
        this._callbacks.onPositionChange?.(this._position);

        // ── Epoch-change detection ──────────────────────────────────
        const currentInt = Math.floor(this._position);
        if (currentInt !== this._lastEpochInt) {
          this._lastEpochInt = currentInt;
          for (const cb of this._epochCb) cb(currentInt);
        }

        // ── End-of-timeline: loop or pause ──────────────────────────
        if (this._position >= this._maxEpoch) {
          if (this._loop) {
            // Rewind to 0, fire epoch-change, keep playing
            this.seekTo(0);
            this._lastEpochInt = 0;
            for (const cb of this._epochCb) cb(0);
            // Restart playback from beginning
            this._sequence.pause();
            this._sequence.play({
              range: [0, this._maxEpoch],
              rate: this._rate,
              iterationCount: 1,
            });
          } else {
            this._sequence.pause();
            this._state = 'paused';
            this._callbacks.onComplete?.();
            this._callbacks.onStateChange?.('paused');
            this._stopPositionTracking();
            return;
          }
        }

        this._rafId = requestAnimationFrame(tick);
      }
    };
    this._rafId = requestAnimationFrame(tick);
  }

  private _stopPositionTracking(): void {
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
    // Sync final position
    if (this._sequence) {
      this._position = this._sequence.position;
    }
  }

  /** Cleanup */
  dispose(): void {
    this._stopPositionTracking();
    this._sequence = null;
    this._epochCb = [];
  }
}
