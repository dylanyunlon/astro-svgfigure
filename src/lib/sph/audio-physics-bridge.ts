/**
 * src/lib/sph/audio-physics-bridge.ts
 *
 * Audio-Physics Bridge — M748
 * ────────────────────────────
 * Translates physics events produced by the SPH + rigid-body pipeline into
 * real-time synthesised sound via Tone.js.
 *
 * Three event categories are sonified:
 *
 *   1. **Collision contacts** → short percussive *ping*.
 *      Volume scales with penetration depth; pitch is determined by the
 *      species of the colliding body (each species has its own oscillator
 *      waveform and base frequency).
 *
 *   2. **Fluid density waves** → low-frequency *rumble*.
 *      A slowly modulated sub-bass whose frequency tracks the rate of
 *      change in the neighbourhood-averaged SPH density field.
 *
 *   3. **Emitter jets** → continuous filtered *hiss* (white noise).
 *      Each active emitter contributes band-passed noise whose gain
 *      follows the emission rate multiplier.
 *
 * Species → timbre mapping
 * ────────────────────────
 *   cil-eye   → bell tone   (sine oscillator,  base 880 Hz)
 *   cil-bolt  → sharp zap   (sawtooth,         base 1320 Hz)
 *   cil-plus  → soft thud   (triangle,          base 220 Hz)
 *   (others)  → neutral tap (square,            base 440 Hz)
 *
 * Integration
 * ───────────
 *   import { AudioPhysicsBridge } from './audio-physics-bridge';
 *
 *   const audio = new AudioPhysicsBridge();
 *
 *   // Inside your frame loop:
 *   audio.update(world, manifolds);
 *
 *   // Volume / mute controls:
 *   audio.setVolume(0.5);
 *   audio.mute();
 *   audio.unmute();
 *
 *   // Clean-up:
 *   audio.dispose();
 */

import * as Tone from 'tone';

import type { ContactManifold, ContactPoint } from './collision/contact-manifold';
import type { World, Particle, Emitter } from './world-stepper';

// ─────────────────────────────────────────────────────────────────────────────
// Species → timbre configuration
// ─────────────────────────────────────────────────────────────────────────────

interface SpeciesTimbre {
  /** Oscillator waveform type. */
  waveform: OscillatorType;
  /** Base MIDI-style frequency (Hz) for the collision ping. */
  baseFreq: number;
  /** Attack time in seconds for the ping envelope. */
  attack: number;
  /** Decay time in seconds for the ping envelope. */
  decay: number;
}

const SPECIES_TIMBRES: Record<string, SpeciesTimbre> = {
  'cil-eye': {
    waveform: 'sine',
    baseFreq: 880,
    attack: 0.002,
    decay: 0.15,
  },
  'cil-bolt': {
    waveform: 'sawtooth',
    baseFreq: 1320,
    attack: 0.001,
    decay: 0.08,
  },
  'cil-plus': {
    waveform: 'triangle',
    baseFreq: 220,
    attack: 0.005,
    decay: 0.25,
  },
};

const DEFAULT_TIMBRE: SpeciesTimbre = {
  waveform: 'square',
  baseFreq: 440,
  attack: 0.003,
  decay: 0.12,
};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Maximum simultaneous collision pings per frame to avoid audio blowout. */
const MAX_PINGS_PER_FRAME = 8;

/** Minimum penetration depth to trigger a ping (filters micro-contacts). */
const MIN_DEPTH_THRESHOLD = 0.001;

/** Smoothing factor for the density-wave EMA (0 = no smoothing, 1 = frozen). */
const DENSITY_EMA_ALPHA = 0.85;

/** Maximum rumble frequency (Hz). */
const RUMBLE_MAX_FREQ = 80;

/** Minimum rumble frequency (Hz). */
const RUMBLE_MIN_FREQ = 20;

/** Noise filter centre frequency for emitter hiss. */
const HISS_FILTER_FREQ = 4000;

/** Noise filter Q for emitter hiss. */
const HISS_FILTER_Q = 1.2;

// ─────────────────────────────────────────────────────────────────────────────
// Pooled ping voice — reusable synth for collision sounds
// ─────────────────────────────────────────────────────────────────────────────

interface PingVoice {
  osc: Tone.Oscillator;
  env: Tone.AmplitudeEnvelope;
  gain: Tone.Gain;
  busy: boolean;
}

function createPingVoice(destination: Tone.ToneAudioNode): PingVoice {
  const env = new Tone.AmplitudeEnvelope({
    attack: 0.002,
    decay: 0.15,
    sustain: 0,
    release: 0.05,
  });

  const gain = new Tone.Gain(0);
  const osc = new Tone.Oscillator({ frequency: 440, type: 'sine' });

  osc.connect(env);
  env.connect(gain);
  gain.connect(destination);
  osc.start();

  return { osc, env, gain, busy: false };
}

// ─────────────────────────────────────────────────────────────────────────────
// AudioPhysicsBridge
// ─────────────────────────────────────────────────────────────────────────────

export class AudioPhysicsBridge {
  // ── Master output chain ──────────────────────────────────
  private masterGain: Tone.Gain;
  private limiter: Tone.Limiter;

  // ── Collision ping pool ──────────────────────────────────
  private pingPool: PingVoice[] = [];
  private readonly pingPoolSize = MAX_PINGS_PER_FRAME;

  // ── Density rumble ───────────────────────────────────────
  private rumbleOsc: Tone.Oscillator;
  private rumbleGain: Tone.Gain;
  private rumbleFilter: Tone.Filter;
  private prevAvgDensity = 0;
  private densityEma = 0;

  // ── Emitter hiss ─────────────────────────────────────────
  private hissNoise: Tone.Noise;
  private hissFilter: Tone.BiquadFilter;
  private hissGain: Tone.Gain;

  // ── State ────────────────────────────────────────────────
  private _muted = false;
  private _volume = 0.5; // 0 … 1
  private _disposed = false;
  private _started = false;

  // ──────────────────────────────────────────────────────────
  constructor() {
    // Master output: gain → limiter → destination
    this.limiter = new Tone.Limiter(-3);
    this.masterGain = new Tone.Gain(this._volume);
    this.masterGain.connect(this.limiter);
    this.limiter.toDestination();

    // ── Ping voice pool ────────────────────────────────────
    for (let i = 0; i < this.pingPoolSize; i++) {
      this.pingPool.push(createPingVoice(this.masterGain));
    }

    // ── Density rumble oscillator ──────────────────────────
    this.rumbleFilter = new Tone.Filter({
      frequency: 120,
      type: 'lowpass',
      rolloff: -24,
    });
    this.rumbleGain = new Tone.Gain(0);
    this.rumbleOsc = new Tone.Oscillator({
      frequency: RUMBLE_MIN_FREQ,
      type: 'sine',
    });
    this.rumbleOsc.connect(this.rumbleFilter);
    this.rumbleFilter.connect(this.rumbleGain);
    this.rumbleGain.connect(this.masterGain);
    this.rumbleOsc.start();

    // ── Emitter hiss (white noise → bandpass → gain) ───────
    this.hissFilter = new Tone.BiquadFilter({
      frequency: HISS_FILTER_FREQ,
      type: 'bandpass',
      Q: HISS_FILTER_Q,
    });
    this.hissGain = new Tone.Gain(0);
    this.hissNoise = new Tone.Noise('white');
    this.hissNoise.connect(this.hissFilter);
    this.hissFilter.connect(this.hissGain);
    this.hissGain.connect(this.masterGain);
    this.hissNoise.start();
  }

  // ──────────────────────────────────────────────────────────
  // Public API
  // ──────────────────────────────────────────────────────────

  /**
   * Call once per frame.  Reads the current physics state and triggers
   * the appropriate audio events.
   *
   * @param world     The SPH world (particles, emitters, config).
   * @param contacts  Contact manifolds produced by the collision pipeline
   *                  this frame (the return value of `CollisionWorld.step()`).
   */
  update(world: World, contacts: ContactManifold[]): void {
    if (this._disposed || this._muted) return;

    // Tone.js requires a user-gesture before the AudioContext can run.
    // We lazily resume on the first update call after construction.
    this.ensureStarted();

    this.processContacts(contacts, world);
    this.processDensityWave(world);
    this.processEmitters(world);
  }

  /** Set master volume (0 = silent, 1 = full). */
  setVolume(v: number): void {
    this._volume = Math.max(0, Math.min(1, v));
    if (!this._muted) {
      this.masterGain.gain.rampTo(this._volume, 0.05);
    }
  }

  /** Mute all audio output (retains volume setting for unmute). */
  mute(): void {
    this._muted = true;
    this.masterGain.gain.rampTo(0, 0.05);
  }

  /** Restore audio to the previously set volume. */
  unmute(): void {
    this._muted = false;
    this.masterGain.gain.rampTo(this._volume, 0.05);
  }

  /** True when muted. */
  get isMuted(): boolean {
    return this._muted;
  }

  /** Release all Tone.js resources. Safe to call multiple times. */
  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;

    for (const v of this.pingPool) {
      v.osc.stop();
      v.osc.dispose();
      v.env.dispose();
      v.gain.dispose();
    }
    this.pingPool.length = 0;

    this.rumbleOsc.stop();
    this.rumbleOsc.dispose();
    this.rumbleFilter.dispose();
    this.rumbleGain.dispose();

    this.hissNoise.stop();
    this.hissNoise.dispose();
    this.hissFilter.dispose();
    this.hissGain.dispose();

    this.limiter.dispose();
    this.masterGain.dispose();
  }

  // ──────────────────────────────────────────────────────────
  // Internal: audio context bootstrap
  // ──────────────────────────────────────────────────────────

  private ensureStarted(): void {
    if (this._started) return;
    if (Tone.getContext().state !== 'running') {
      // Cannot resume without gesture — the first user interaction will
      // unblock the context and subsequent update() calls will produce sound.
      Tone.start().catch(() => {});
      return;
    }
    this._started = true;
  }

  // ──────────────────────────────────────────────────────────
  // 1. Collision contacts → pings
  // ──────────────────────────────────────────────────────────

  private processContacts(manifolds: ContactManifold[], world: World): void {
    if (manifolds.length === 0) return;

    // Collect candidate contact points sorted by depth (loudest first).
    const candidates: Array<{ point: ContactPoint; species: string }> = [];

    for (const manifold of manifolds) {
      // Resolve species from the body IDs via the world particle list.
      const species = this.resolveSpecies(manifold.bodyA.id, world);

      for (const pt of manifold.points) {
        if (pt.depth < MIN_DEPTH_THRESHOLD) continue;
        candidates.push({ point: pt, species });
      }
    }

    // Sort descending by penetration depth so the loudest contacts
    // win the limited voice pool.
    candidates.sort((a, b) => b.point.depth - a.point.depth);

    const count = Math.min(candidates.length, MAX_PINGS_PER_FRAME);
    for (let i = 0; i < count; i++) {
      this.triggerPing(candidates[i].point, candidates[i].species);
    }
  }

  /**
   * Resolve the species string for a body id by scanning the world's
   * particle list.  Falls back to empty string (which maps to the
   * default timbre).
   */
  private resolveSpecies(bodyId: number, world: World): string {
    // Rigid bodies in the world carry species on their originating
    // particle (same id).  A linear scan is acceptable because the
    // body count is typically < 50 and this runs at most once per
    // manifold per frame.
    for (const p of world.particles) {
      if (p.id === bodyId) return p.species;
    }
    return '';
  }

  /** Fire a single percussive ping on a free voice from the pool. */
  private triggerPing(pt: ContactPoint, species: string): void {
    const voice = this.acquireVoice();
    if (!voice) return;

    const timbre = SPECIES_TIMBRES[species] ?? DEFAULT_TIMBRE;

    // Pitch: base frequency ± slight variation from contact position
    // to avoid identical tones stacking unnaturally.
    const pitchJitter = 1.0 + (pt.x * 0.0013 + pt.y * 0.0007) % 0.06;
    const freq = timbre.baseFreq * pitchJitter;

    // Volume: proportional to penetration depth, clamped to [0, 1].
    const depthGain = Math.min(1.0, pt.depth * 15);

    voice.osc.type = timbre.waveform;
    voice.osc.frequency.setValueAtTime(freq, Tone.now());
    voice.gain.gain.setValueAtTime(depthGain * this._volume, Tone.now());

    voice.env.attack = timbre.attack;
    voice.env.decay = timbre.decay;
    voice.env.triggerAttackRelease(timbre.attack + timbre.decay);

    // Mark voice as free after the envelope completes.
    const releaseMs = (timbre.attack + timbre.decay + 0.05) * 1000;
    setTimeout(() => {
      voice.busy = false;
    }, releaseMs);
  }

  /** Grab a non-busy voice from the pool, or null if all are occupied. */
  private acquireVoice(): PingVoice | null {
    for (const v of this.pingPool) {
      if (!v.busy) {
        v.busy = true;
        return v;
      }
    }
    return null;
  }

  // ──────────────────────────────────────────────────────────
  // 2. Fluid density wave → rumble
  // ──────────────────────────────────────────────────────────

  private processDensityWave(world: World): void {
    if (world.particles.length === 0) {
      this.rumbleGain.gain.rampTo(0, 0.1);
      return;
    }

    // Compute average density across all live particles.
    let totalDensity = 0;
    for (const p of world.particles) {
      totalDensity += p.density;
    }
    const avgDensity = totalDensity / world.particles.length;

    // Rate of change (finite difference between frames).
    const deltaRho = Math.abs(avgDensity - this.prevAvgDensity);
    this.prevAvgDensity = avgDensity;

    // EMA-smoothed rate of change to avoid jitter.
    this.densityEma =
      DENSITY_EMA_ALPHA * this.densityEma +
      (1 - DENSITY_EMA_ALPHA) * deltaRho;

    // Map smoothed delta to rumble frequency (higher change = higher freq).
    const t = Math.min(1.0, this.densityEma * 5);
    const rumbleFreq = RUMBLE_MIN_FREQ + t * (RUMBLE_MAX_FREQ - RUMBLE_MIN_FREQ);
    this.rumbleOsc.frequency.rampTo(rumbleFreq, 0.1);

    // Rumble gain proportional to density change magnitude.
    const rumbleVol = Math.min(0.35, t * 0.5);
    this.rumbleGain.gain.rampTo(rumbleVol, 0.1);
  }

  // ──────────────────────────────────────────────────────────
  // 3. Emitter jets → hiss
  // ──────────────────────────────────────────────────────────

  private processEmitters(world: World): void {
    if (!world.emitters || world.emitters.length === 0) {
      this.hissGain.gain.rampTo(0, 0.15);
      return;
    }

    // Aggregate emission intensity from all active emitters.
    let totalRate = 0;
    for (const em of world.emitters) {
      totalRate += em.rate;
    }

    // Normalise so that a single moderate-rate emitter (~200 p/s)
    // produces a comfortable hiss level.
    const normRate = Math.min(1.0, totalRate / 800);
    this.hissGain.gain.rampTo(normRate * 0.2, 0.1);

    // Shift filter frequency slightly with total rate to add variation.
    const filterFreq = HISS_FILTER_FREQ + normRate * 1500;
    this.hissFilter.frequency.rampTo(filterFreq, 0.15);
  }
}
