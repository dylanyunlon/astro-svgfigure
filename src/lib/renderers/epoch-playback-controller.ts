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

  // ── Position tracking ─────────────────────────────────────────────────

  private _startPositionTracking(): void {
    const tick = () => {
      if (this._sequence && this._state === 'playing') {
        this._position = this._sequence.position;
        this._callbacks.onPositionChange?.(this._position);
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
  }
}
