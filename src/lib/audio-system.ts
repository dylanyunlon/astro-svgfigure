/**
 * audio-system.ts — Spatial audio, SFX, and speech input
 *
 * Covers:
 *   SFXController       — Web Audio API sfx pool with pitch/volume/pan
 *   ResonanceAudioScene — Google Resonance Audio wrapper (HRTF spatial audio)
 *   SpeechInputManager  — Web Speech API SpeechRecognition wrapper
 *
 * All classes are tree-shakeable; import only what you need.
 */

// ─── Types ────────────────────────────────────────────────────────────────────

export interface SFXOptions {
  volume?:     number; // 0–1, default 1
  pitch?:      number; // semitones, default 0
  pan?:        number; // -1 to 1 (StereoPanner), default 0
  loop?:       boolean;
  fadeIn?:     number; // seconds
  fadeOut?:    number; // seconds
  spatial?:    boolean;
  position?:   [number, number, number]; // world XYZ for spatial
}

export interface SFXHandle {
  id:     number;
  stop:   (fadeOut?: number) => void;
  setVolume: (v: number, rampTime?: number) => void;
  setPitch:  (semitones: number) => void;
}

export interface ResonanceSourceOptions {
  minDistance?:   number;
  maxDistance?:   number;
  rolloff?:       'logarithmic' | 'linear' | 'none';
  directivity?:   number; // 0 = omni, 1 = cardioid
  occluder?:      boolean;
}

export interface SpeechGrammar {
  phrases: string[];
  weight?: number;
}

export interface SpeechResult {
  transcript: string;
  confidence: number;
  isFinal:    boolean;
}

// ─── SFXController ───────────────────────────────────────────────────────────

interface PoolEntry {
  id:           number;
  source:       AudioBufferSourceNode;
  gainNode:     GainNode;
  pannerNode:   StereoPannerNode | PannerNode;
  playbackRate: AudioParam;
  startedAt:    number;
  loop:         boolean;
  active:       boolean;
}

export class SFXController {
  private _ctx:       AudioContext;
  private _masterGain:GainNode;
  private _pool:      Map<number, PoolEntry> = new Map();
  private _buffers:   Map<string, AudioBuffer> = new Map();
  private _nextId:    number = 1;
  private _muted:     boolean = false;

  constructor(ctx?: AudioContext) {
    this._ctx = ctx ?? new AudioContext();
    this._masterGain = this._ctx.createGain();
    this._masterGain.connect(this._ctx.destination);
  }

  get context(): AudioContext { return this._ctx; }
  get masterGain(): GainNode  { return this._masterGain; }
  get muted(): boolean        { return this._muted; }

  async loadBuffer(key: string, url: string): Promise<AudioBuffer> {
    if (this._buffers.has(key)) return this._buffers.get(key)!;
    const res = await fetch(url);
    const ab  = await res.arrayBuffer();
    const buf = await this._ctx.decodeAudioData(ab);
    this._buffers.set(key, buf);
    return buf;
  }

  registerBuffer(key: string, buffer: AudioBuffer): void {
    this._buffers.set(key, buffer);
  }

  play(key: string, opts: SFXOptions = {}): SFXHandle {
    const buf = this._buffers.get(key);
    if (!buf) throw new Error(`[SFXController] Buffer not loaded: ${key}`);

    const {
      volume   = 1,
      pitch    = 0,
      pan      = 0,
      loop     = false,
      fadeIn   = 0,
      spatial  = false,
      position = [0, 0, 0],
    } = opts;

    const id = this._nextId++;

    // Gain
    const gainNode = this._ctx.createGain();
    gainNode.gain.setValueAtTime(fadeIn > 0 ? 0.0001 : volume, this._ctx.currentTime);
    if (fadeIn > 0) gainNode.gain.linearRampToValueAtTime(volume, this._ctx.currentTime + fadeIn);

    // Panner
    let pannerNode: StereoPannerNode | PannerNode;
    if (spatial) {
      const panner = this._ctx.createPanner();
      panner.panningModel  = 'HRTF';
      panner.distanceModel = 'inverse';
      panner.setPosition(...position);
      pannerNode = panner;
    } else {
      const stereo = this._ctx.createStereoPanner();
      stereo.pan.value = Math.max(-1, Math.min(1, pan));
      pannerNode = stereo;
    }

    // Source
    const source = this._ctx.createBufferSource();
    source.buffer = buf;
    source.loop   = loop;
    source.playbackRate.value = Math.pow(2, pitch / 12);

    source.connect(gainNode);
    gainNode.connect(pannerNode);
    pannerNode.connect(this._masterGain);

    if (this._muted) gainNode.gain.value = 0;

    source.start();
    source.onended = () => {
      const entry = this._pool.get(id);
      if (entry) entry.active = false;
    };

    const entry: PoolEntry = {
      id, source, gainNode, pannerNode,
      playbackRate: source.playbackRate,
      startedAt: this._ctx.currentTime,
      loop, active: true,
    };
    this._pool.set(id, entry);

    return {
      id,
      stop: (fadeOut = 0) => this._stopEntry(id, fadeOut),
      setVolume: (v, ramp = 0.05) => {
        const e = this._pool.get(id);
        if (!e) return;
        e.gainNode.gain.linearRampToValueAtTime(v, this._ctx.currentTime + ramp);
      },
      setPitch: (semitones) => {
        const e = this._pool.get(id);
        if (e) e.source.playbackRate.setValueAtTime(Math.pow(2, semitones / 12), this._ctx.currentTime);
      },
    };
  }

  private _stopEntry(id: number, fadeOut: number): void {
    const entry = this._pool.get(id);
    if (!entry || !entry.active) return;
    const now = this._ctx.currentTime;
    if (fadeOut > 0) {
      entry.gainNode.gain.linearRampToValueAtTime(0.0001, now + fadeOut);
      entry.source.stop(now + fadeOut);
    } else {
      entry.source.stop();
    }
    entry.active = false;
    this._pool.delete(id);
  }

  stopAll(fadeOut = 0): void {
    for (const [id] of this._pool) this._stopEntry(id, fadeOut);
  }

  setMasterVolume(v: number, ramp = 0.05): void {
    this._masterGain.gain.linearRampToValueAtTime(
      Math.max(0, Math.min(1, v)),
      this._ctx.currentTime + ramp
    );
  }

  mute():   void { this._muted = true;  this.setMasterVolume(0); }
  unmute(): void { this._muted = false; this.setMasterVolume(1); }

  /** GC inactive pool entries. */
  prune(): void {
    for (const [id, entry] of this._pool) {
      if (!entry.active) this._pool.delete(id);
    }
  }

  async resume(): Promise<void> {
    if (this._ctx.state === 'suspended') await this._ctx.resume();
  }

  dispose(): void {
    this.stopAll(0);
    this._ctx.close();
  }
}

// ─── ResonanceAudioScene ─────────────────────────────────────────────────────

/**
 * Thin wrapper over the Google Resonance Audio SDK.
 * Resonance must be loaded as a UMD global (`ResonanceAudio`) or imported
 * separately: https://github.com/resonance-audio/resonance-audio-web-sdk
 *
 * This wrapper is designed for tree-shaking and avoids hard-importing the
 * Resonance SDK so the file compiles without the optional dependency.
 */

declare const ResonanceAudio: any; // provided by the Resonance SDK

export interface ResonanceRoomDimensions {
  width:  number;
  height: number;
  depth:  number;
}

export interface ResonanceMaterials {
  left:     string;
  right:    string;
  front:    string;
  back:     string;
  up:       string;
  down:     string;
}

export interface ResonanceSceneOptions {
  ambisonicOrder?:    1 | 3;
  listenerPosition?:  [number, number, number];
  listenerOrientation?:[number, number, number, number]; // quaternion
  room?:              ResonanceRoomDimensions;
  materials?:         Partial<ResonanceMaterials>;
}

export interface ResonanceSourceHandle {
  id:        number;
  setPosition(x: number, y: number, z: number): void;
  setOrientation(qx: number, qy: number, qz: number, qw: number): void;
  setVolume(gain: number): void;
  connectMediaElement(el: HTMLMediaElement): void;
  connectAudioNode(node: AudioNode): AudioNode; // returns resonance input node
  disconnect(): void;
}

export class ResonanceAudioScene {
  private _ctx:     AudioContext;
  private _scene:   any; // ResonanceAudio scene instance
  private _sources: Map<number, { source: any; gain: GainNode }> = new Map();
  private _nextId:  number = 1;
  private _ready:   boolean = false;

  constructor(ctx: AudioContext, opts: ResonanceSceneOptions = {}) {
    this._ctx = ctx;

    if (typeof ResonanceAudio === 'undefined') {
      console.warn('[ResonanceAudioScene] ResonanceAudio SDK not found. Falling back to no-op.');
      return;
    }

    const sceneOpts: Record<string, unknown> = {
      ambisonicOrder: opts.ambisonicOrder ?? 1,
    };

    if (opts.room) {
      sceneOpts.dimensions = opts.room;
      const defaultMat = 'acoustic-ceiling-tiles';
      const m = opts.materials ?? {};
      sceneOpts.materials = {
        left: m.left ?? defaultMat, right: m.right ?? defaultMat,
        front: m.front ?? defaultMat, back: m.back ?? defaultMat,
        up: m.up ?? defaultMat, down: m.down ?? defaultMat,
      };
    }

    this._scene = new ResonanceAudio(ctx, sceneOpts);
    this._scene.output.connect(ctx.destination);
    this._ready = true;

    if (opts.listenerPosition) {
      const [x, y, z] = opts.listenerPosition;
      this._scene.setListenerPosition(x, y, z);
    }
    if (opts.listenerOrientation) {
      const [qx, qy, qz, qw] = opts.listenerOrientation;
      this._scene.setListenerOrientationQuaternion(qx, qy, qz, qw);
    }
  }

  get isReady(): boolean { return this._ready; }
  get context(): AudioContext { return this._ctx; }

  setListenerPosition(x: number, y: number, z: number): void {
    this._scene?.setListenerPosition(x, y, z);
  }

  setListenerOrientation(qx: number, qy: number, qz: number, qw: number): void {
    this._scene?.setListenerOrientationQuaternion(qx, qy, qz, qw);
  }

  setListenerFromMatrix(m: Float32Array): void {
    // Column-major 4×4; extract position and forward/up
    if (!this._scene) return;
    this._scene.setListenerPosition(m[12], m[13], m[14]);
    // forward = -Z column, up = +Y column
    this._scene.setListenerFromMatrix({ elements: Array.from(m) });
  }

  createSource(opts: ResonanceSourceOptions = {}): ResonanceSourceHandle {
    const id = this._nextId++;
    const gain = this._ctx.createGain();
    gain.gain.value = 1;

    if (!this._ready) {
      // No-op handle
      return {
        id,
        setPosition: () => {},
        setOrientation: () => {},
        setVolume: (g) => { gain.gain.value = g; },
        connectMediaElement: () => {},
        connectAudioNode: (node) => node,
        disconnect: () => {},
      };
    }

    const srcOpts: Record<string, unknown> = {};
    if (opts.minDistance !== undefined) srcOpts.minDistance = opts.minDistance;
    if (opts.maxDistance !== undefined) srcOpts.maxDistance = opts.maxDistance;
    if (opts.rolloff)     srcOpts.rolloff = opts.rolloff;
    if (opts.directivity !== undefined) srcOpts.directivity = opts.directivity;

    const resonanceSource = this._scene.createSource(srcOpts);
    this._sources.set(id, { source: resonanceSource, gain });

    const handle: ResonanceSourceHandle = {
      id,
      setPosition: (x, y, z) => resonanceSource.setPosition(x, y, z),
      setOrientation: (qx, qy, qz, qw) => resonanceSource.setOrientationQuaternion(qx, qy, qz, qw),
      setVolume: (g) => { gain.gain.value = g; },
      connectMediaElement: (el) => {
        const srcNode = this._ctx.createMediaElementSource(el);
        srcNode.connect(gain);
        gain.connect(resonanceSource.input);
      },
      connectAudioNode: (node) => {
        node.connect(gain);
        gain.connect(resonanceSource.input);
        return resonanceSource.input as AudioNode;
      },
      disconnect: () => {
        gain.disconnect();
        this._sources.delete(id);
      },
    };

    return handle;
  }

  setRoomProperties(dimensions: ResonanceRoomDimensions, materials: Partial<ResonanceMaterials>): void {
    if (!this._scene) return;
    const defaultMat = 'acoustic-ceiling-tiles';
    this._scene.setRoomProperties(dimensions, {
      left:  materials.left  ?? defaultMat,
      right: materials.right ?? defaultMat,
      front: materials.front ?? defaultMat,
      back:  materials.back  ?? defaultMat,
      up:    materials.up    ?? defaultMat,
      down:  materials.down  ?? defaultMat,
    });
  }

  dispose(): void {
    for (const [, { gain }] of this._sources) gain.disconnect();
    this._sources.clear();
    this._scene?.output?.disconnect?.();
  }
}

// ─── SpeechInputManager ───────────────────────────────────────────────────────

export interface SpeechInputOptions {
  lang?:         string;   // BCP-47, default 'en-US'
  continuous?:   boolean;
  interimResults?:boolean;
  maxAlternatives?:number;
  grammars?:     SpeechGrammar[];
}

type SpeechEventType = 'result' | 'start' | 'end' | 'error' | 'nomatch';

export class SpeechInputManager extends EventTarget {
  private _recognition: SpeechRecognition | null = null;
  private _opts:        SpeechInputOptions;
  private _running:     boolean = false;
  private _supported:   boolean = false;
  private _transcript:  string = '';

  constructor(opts: SpeechInputOptions = {}) {
    super();
    this._opts = opts;

    const SRClass = (typeof window !== 'undefined')
      ? (window.SpeechRecognition ?? (window as any).webkitSpeechRecognition ?? null)
      : null;

    if (!SRClass) {
      console.warn('[SpeechInputManager] SpeechRecognition not available in this browser.');
      return;
    }

    this._recognition = new SRClass() as SpeechRecognition;
    this._supported   = true;
    this._configure();
  }

  get isSupported(): boolean { return this._supported; }
  get isRunning():   boolean { return this._running; }
  get lastTranscript(): string { return this._transcript; }

  private _configure(): void {
    const r = this._recognition!;
    r.lang              = this._opts.lang             ?? 'en-US';
    r.continuous        = this._opts.continuous        ?? false;
    r.interimResults    = this._opts.interimResults    ?? true;
    r.maxAlternatives   = this._opts.maxAlternatives   ?? 1;

    // Grammar list
    if (this._opts.grammars?.length && typeof SpeechGrammarList !== 'undefined') {
      const list = new SpeechGrammarList();
      for (const g of this._opts.grammars) {
        const jsgf = `#JSGF V1.0; grammar phrases; public <phrase> = ${g.phrases.join(' | ')};`;
        list.addFromString(jsgf, g.weight ?? 1);
      }
      r.grammars = list;
    }

    r.onstart = () => {
      this._running = true;
      this.dispatchEvent(new Event('start'));
    };

    r.onend = () => {
      this._running = false;
      this.dispatchEvent(new Event('end'));
      // Auto-restart in continuous mode if not explicitly stopped
      if (this._opts.continuous) {
        try { r.start(); } catch { /* already stopped */ }
      }
    };

    r.onerror = (ev) => {
      this._running = false;
      this.dispatchEvent(new CustomEvent('error', { detail: { error: ev.error, message: ev.message } }));
    };

    r.onnomatch = () => {
      this.dispatchEvent(new Event('nomatch'));
    };

    r.onresult = (ev: SpeechRecognitionEvent) => {
      let interim = '';
      let finalText = '';

      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const alt = ev.results[i][0];
        if (ev.results[i].isFinal) {
          finalText += alt.transcript;
        } else {
          interim += alt.transcript;
        }
      }

      if (finalText) this._transcript = finalText.trim();

      const result: SpeechResult = {
        transcript: finalText || interim,
        confidence: ev.results[ev.resultIndex]?.[0]?.confidence ?? 0,
        isFinal:    !!finalText,
      };

      this.dispatchEvent(new CustomEvent('result', { detail: result }));
    };
  }

  start(): void {
    if (!this._supported || this._running) return;
    try {
      this._recognition!.start();
    } catch (e) {
      console.warn('[SpeechInputManager] start() failed:', e);
    }
  }

  stop(): void {
    if (!this._supported || !this._running) return;
    // Disable auto-restart before calling stop
    const wasContinuous = this._opts.continuous;
    this._opts.continuous = false;
    this._recognition!.stop();
    this._opts.continuous = wasContinuous;
  }

  abort(): void {
    if (!this._supported) return;
    this._opts.continuous = false;
    this._recognition!.abort();
    this._running = false;
  }

  on<T = unknown>(event: SpeechEventType, handler: (data: T) => void): this {
    this.addEventListener(event, (e) => handler((e as CustomEvent<T>).detail ?? e as unknown as T));
    return this;
  }

  /** Convenience: returns a promise that resolves on the next final transcript. */
  listenOnce(timeoutMs = 10000): Promise<SpeechResult> {
    return new Promise((resolve, reject) => {
      if (!this._supported) { reject(new Error('SpeechRecognition not supported')); return; }
      const timer = setTimeout(() => { this.abort(); reject(new Error('Speech timeout')); }, timeoutMs);

      const handler = (e: Event) => {
        const result = (e as CustomEvent<SpeechResult>).detail;
        if (result.isFinal) {
          clearTimeout(timer);
          this.removeEventListener('result', handler);
          this.stop();
          resolve(result);
        }
      };
      this.addEventListener('result', handler);
      this.start();
    });
  }

  dispose(): void {
    this.abort();
    this._recognition = null;
  }
}

// ─── Convenience factory ──────────────────────────────────────────────────────

export interface AudioSystemBundle {
  ctx:       AudioContext;
  sfx:       SFXController;
  resonance: ResonanceAudioScene;
  speech:    SpeechInputManager;
}

export function createAudioSystem(
  resonanceOpts: ResonanceSceneOptions = {},
  speechOpts:    SpeechInputOptions    = {},
): AudioSystemBundle {
  const ctx       = new AudioContext();
  const sfx       = new SFXController(ctx);
  const resonance = new ResonanceAudioScene(ctx, resonanceOpts);
  const speech    = new SpeechInputManager(speechOpts);
  return { ctx, sfx, resonance, speech };
}
