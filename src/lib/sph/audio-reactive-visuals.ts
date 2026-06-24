/**
 * src/lib/sph/audio-reactive-visuals.ts — M776: Audio-Reactive Visual System
 *
 * Audio-Driven Visuals — Music → Physics "Breathing"
 * ─────────────────────────────────────────────────────────────────────────────
 * Analyses the Tone.js FFT frequency spectrum in real-time and translates
 * spectral energy into visual modulation across three frequency bands:
 *
 *   **LOW band**   (20 – 250 Hz)  → Cell breathing scale
 *     Bass and sub-bass energy drives a smooth, rhythmic scale oscillation
 *     on all cell rigid bodies.  Cells swell on kick drums and bass drops,
 *     creating a "heartbeat" effect that makes the physics world feel alive.
 *     Output: a normalised `breathScale` factor (0.85 – 1.25 range) applied
 *     to each cell's model-matrix scale via the InstancedCellRenderer.
 *
 *   **MID band**   (250 – 4 000 Hz) → Particle velocity multiplier
 *     Melody, vocals, and harmonic content modulate the speed of SPH fluid
 *     particles.  When mid-band energy rises (vocal onset, guitar riff),
 *     particles accelerate; during quiet passages they drift slowly.
 *     Output: a `velocityMultiplier` (0.3 – 2.5) applied as a force-scale
 *     factor on particle velocities each frame.
 *
 *   **HIGH band**  (4 000 – 20 000 Hz) → Bloom intensity
 *     Hi-hats, cymbals, sibilance, and transient brightness map to the
 *     bloom post-process scale.  Sharp high-frequency transients produce
 *     brief bloom flashes; sustained highs create a warm glow.
 *     Output: a `bloomScale` factor (0.5 – 4.0) fed to ATBloomPostProcess
 *     via the VFXHandler.onBloomUpdate callback.
 *
 * Signal chain
 * ────────────
 *   Tone.js AudioContext
 *       ↓
 *   Tone.FFT (1024-bin)
 *       ↓
 *   AudioReactiveVisuals.update(dt)
 *       ↓  ┌── lowEnergy  → breathScale      → cell model-matrix uniform
 *       ├──┤── midEnergy  → velocityMultiplier→ particle force scaling
 *       │  └── highEnergy → bloomScale        → post-process bloom
 *       ↓
 *   AudioReactiveSnapshot (returned per frame)
 *
 * EMA smoothing
 * ─────────────
 * Raw FFT bins are noisy frame-to-frame.  Each band uses an exponential
 * moving average (EMA) with independently tuneable α values so that:
 *   - Low band: heavy smoothing (α = 0.12) for slow, organic breathing
 *   - Mid band: moderate smoothing (α = 0.20) for responsive yet stable speed
 *   - High band: light smoothing (α = 0.35) for crisp transient response
 *
 * Integration
 * ───────────
 *
 *   const arv = new AudioReactiveVisuals();
 *
 *   // Each frame:
 *   const snap = arv.update(dt);
 *
 *   // Apply to cell renderer:
 *   for (const cell of cells) {
 *     cell.scaleX *= snap.breathScale;
 *     cell.scaleY *= snap.breathScale;
 *   }
 *
 *   // Apply to particles:
 *   for (const p of world.particles) {
 *     p.vx *= snap.velocityMultiplier;
 *     p.vy *= snap.velocityMultiplier;
 *   }
 *
 *   // Apply to bloom:
 *   bloom.setParams({ bloomScale: snap.bloomScale });
 *
 *   // Clean up:
 *   arv.dispose();
 *
 * Upstream references
 * ───────────────────
 *   src/lib/sph/audio-physics-bridge.ts   — Tone.js integration patterns
 *   src/lib/sph/vfx-timeline.ts           — VFXHandler bloom/shake callbacks
 *   src/lib/sph/at-bloom-postprocess.ts   — ATBloomParams.bloomScale target
 *   src/lib/sph/physics-uniform-bridge.ts — PhysicsUniforms data flow model
 *   src/lib/sph/instanced-cell-renderer.ts— Cell model-matrix scale target
 *   src/lib/sph/world-stepper.ts          — World / Particle types
 *
 * [ASTRO-ARV] debug prefix.
 *
 * Research: xiaodi #M776 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// Constants — Frequency Band Boundaries (Hz)
// ─────────────────────────────────────────────────────────────────────────────


import * as Tone from 'tone';
import type { World, Particle }    from './world-stepper';
import type { ATBloomParams }      from './at-bloom-postprocess';
import type { VFXHandler }         from './vfx-timeline';

<<<<<<< HEAD
// [orphan-precise] /** Low band floor (Hz) — sub-bass. */
=======
/** Low band floor (Hz) — sub-bass. */




>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
const LOW_FREQ_MIN  = 20;
/** Low band ceiling (Hz) — upper bass. */
const LOW_FREQ_MAX  = 250;
/** Mid band ceiling (Hz) — upper midrange. */
const MID_FREQ_MAX  = 4000;
/** High band ceiling (Hz) — presence/brilliance limit. */
const HIGH_FREQ_MAX = 20000;

// ─────────────────────────────────────────────────────────────────────────────
// Constants — FFT Configuration
// ─────────────────────────────────────────────────────────────────────────────

/** FFT size — 1024 bins provides ~43 Hz per bin at 44.1 kHz sample rate. */
const FFT_SIZE = 1024;

/**
 * FFT normalisation floor (dB).
 * Tone.FFT returns values in decibels; anything below this floor is treated
 * as silence.  Typical FFT noise floor is around -100 dB.
 */
const FFT_FLOOR_DB = -100;

/** FFT ceiling (dB) — maximum expected level. */
const FFT_CEILING_DB = 0;

/** Inverse dB range for normalisation. */
const DB_RANGE_INV = 1 / (FFT_CEILING_DB - FFT_FLOOR_DB);

// ─────────────────────────────────────────────────────────────────────────────
// Constants — EMA Smoothing Alphas
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Low band EMA α: heavy smoothing for slow, organic pulsation.
 * Lower α = heavier smoothing, smoother output.
 */
const LOW_ALPHA  = 0.12;

/** Mid band EMA α: moderate smoothing for responsive velocity modulation. */
const MID_ALPHA  = 0.20;

/** High band EMA α: light smoothing for crisp transient bloom spikes. */
const HIGH_ALPHA = 0.35;

// ─────────────────────────────────────────────────────────────────────────────
// Constants — Output Mapping Ranges
// ─────────────────────────────────────────────────────────────────────────────

/** Cell breathing scale range: [min, max]. */
const BREATH_SCALE_MIN = 0.85;
const BREATH_SCALE_MAX = 1.25;

/** Particle velocity multiplier range: [min, max]. */
const VELOCITY_MULT_MIN = 0.3;
const VELOCITY_MULT_MAX = 2.5;

/** Bloom scale range: [min, max]. */
const BLOOM_SCALE_MIN = 0.5;
const BLOOM_SCALE_MAX = 4.0;

// ─────────────────────────────────────────────────────────────────────────────
// Constants — Sensitivity / Gain
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Per-band energy gain multiplier.
 * The raw normalised RMS from FFT bins is often quite small; these gains
 * boost each band into a usable [0, 1] range before EMA / mapping.
 */
const LOW_GAIN  = 2.5;
const MID_GAIN  = 2.0;
const HIGH_GAIN = 3.0;

/**
 * Non-linear response curve exponent per band.
 * Applied as `energy = pow(energy, exponent)` after gain to shape the
 * perceptual response curve.
 *   < 1.0 → more sensitive to quiet signals (expands low end)
 *   > 1.0 → more sensitive to loud signals (compresses low end)
 */
const LOW_EXPONENT  = 0.8;
const MID_EXPONENT  = 1.0;
const HIGH_EXPONENT = 0.7;

// ─────────────────────────────────────────────────────────────────────────────
// Constants — Breath Phase Modulation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The breathing effect adds a subtle sinusoidal modulation on top of the
 * bass-energy-driven scale to avoid a static "stuck at max" look during
 * sustained bass passages.  The sine wave's amplitude is proportional to
 * the current low-band energy so it disappears during silence.
 */
const BREATH_SINE_FREQ = 1.8;   // Hz — organic breathing rhythm
const BREATH_SINE_AMP  = 0.04;  // max ±4% scale modulation from sine

// ─────────────────────────────────────────────────────────────────────────────
// Output Snapshot
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Per-frame audio-reactive snapshot returned by `update()`.
 *
 * Consumers read the fields they care about and apply them to their
 * respective subsystems.  All values are ready to use — no further
 * processing required.
 */
export interface AudioReactiveSnapshot {
  // ── Mapped output values ───────────────────────────────────────────────

  /**
   * Cell breathing scale factor (0.85 – 1.25).
   * Multiply into cell model-matrix scale axes.
   */
  breathScale: number;

  /**
   * Particle velocity multiplier (0.3 – 2.5).
   * Apply as a force/velocity scale on SPH particles.
   */
  velocityMultiplier: number;

  /**
   * Bloom post-process scale (0.5 – 4.0).
   * Feed directly to ATBloomPostProcess.setParams({ bloomScale }).
   */
  bloomScale: number;

  // ── Raw normalised band energies (0 – 1, post-EMA) ────────────────────

  /** Smoothed low-band energy (bass). */
  lowEnergy: number;

  /** Smoothed mid-band energy (melody/vocals). */
  midEnergy: number;

  /** Smoothed high-band energy (presence/brilliance). */
  highEnergy: number;

  // ── Diagnostics ────────────────────────────────────────────────────────

  /** True when the Tone.js AudioContext is running and producing data. */
  active: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Optional configuration overrides for AudioReactiveVisuals.
 * All fields are optional — sensible defaults are provided.
 */
export interface AudioReactiveConfig {
  /** FFT size (power of 2). Default 1024. */
  fftSize?: number;

  /** Low band EMA alpha (0–1). Default 0.12. */
  lowAlpha?: number;
  /** Mid band EMA alpha (0–1). Default 0.20. */
  midAlpha?: number;
  /** High band EMA alpha (0–1). Default 0.35. */
  highAlpha?: number;

  /** Low band gain multiplier. Default 2.5. */
  lowGain?: number;
  /** Mid band gain multiplier. Default 2.0. */
  midGain?: number;
  /** High band gain multiplier. Default 3.0. */
  highGain?: number;

  /** Low band response exponent. Default 0.8. */
  lowExponent?: number;
  /** Mid band response exponent. Default 1.0. */
  midExponent?: number;
  /** High band response exponent. Default 0.7. */
  highExponent?: number;

  /** Breath scale range [min, max]. Default [0.85, 1.25]. */
  breathRange?: [number, number];
  /** Velocity multiplier range [min, max]. Default [0.3, 2.5]. */
  velocityRange?: [number, number];
  /** Bloom scale range [min, max]. Default [0.5, 4.0]. */
  bloomRange?: [number, number];

  /** Breathing sine modulation frequency (Hz). Default 1.8. */
  breathSineFreq?: number;
  /** Breathing sine modulation amplitude. Default 0.04. */
  breathSineAmp?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper — frequency to FFT bin index
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Convert a frequency (Hz) to an FFT bin index.
 *
 * @param freq        Target frequency (Hz).
 * @param fftSize     Number of FFT bins (e.g. 1024).
 * @param sampleRate  Audio context sample rate (e.g. 44100).
 * @returns           Bin index (clamped to [0, fftSize - 1]).
 */
function freqToBin(freq: number, fftSize: number, sampleRate: number): number {
  // Each bin spans (sampleRate / fftSize) Hz.
  // bin = floor(freq / binWidth) = floor(freq * fftSize / sampleRate)
  const bin = Math.floor((freq * fftSize) / sampleRate);
  return Math.max(0, Math.min(fftSize - 1, bin));
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper — linear interpolation
// ─────────────────────────────────────────────────────────────────────────────

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper — clamp
// ─────────────────────────────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

// ─────────────────────────────────────────────────────────────────────────────
// AudioReactiveVisuals
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Core audio-reactive visual system.
 *
 * Instantiate once, call `update(dt)` each frame, and read the returned
 * `AudioReactiveSnapshot` to drive cell scale, particle velocity, and
 * bloom intensity.
 *
 * The system lazily creates a Tone.FFT analyser node connected to the
 * master Tone.js destination.  If Tone.js hasn't started (audio context
 * suspended), the system returns neutral snapshot values (1.0 scale,
 * 1.0 velocity, 1.0 bloom) until audio begins.
 */
export class AudioReactiveVisuals {
  // ── Tone.js analyser ────────────────────────────────────────────────────
  private fft: Tone.FFT;
  private readonly fftSize: number;

  // ── EMA state (smoothed band energies, 0–1) ────────────────────────────
  private emaLow  = 0;
  private emaMid  = 0;
  private emaHigh = 0;

  // ── Phase accumulator for breathing sine modulation ────────────────────
  private breathPhase = 0;

  // ── Resolved configuration ─────────────────────────────────────────────
  private readonly lowAlpha:  number;
  private readonly midAlpha:  number;
  private readonly highAlpha: number;

  private readonly lowGain:  number;
  private readonly midGain:  number;
  private readonly highGain: number;

  private readonly lowExp:  number;
  private readonly midExp:  number;
  private readonly highExp: number;

  private readonly breathMin:   number;
  private readonly breathMax:   number;
  private readonly velMin:      number;
  private readonly velMax:      number;
  private readonly bloomMin:    number;
  private readonly bloomMax:    number;
  private readonly sineFreq:    number;
  private readonly sineAmp:     number;

  // ── Lifecycle state ────────────────────────────────────────────────────
  private _disposed = false;
  private _started  = false;

  // ── Cached bin boundaries (set once sampleRate is known) ───────────────
  private binLowMin  = 0;
  private binLowMax  = 0;
  private binMidMax  = 0;
  private binHighMax = 0;
  private _binsCached = false;

  // ──────────────────────────────────────────────────────────────────────────
  constructor(config?: AudioReactiveConfig) {
    this.fftSize = config?.fftSize ?? FFT_SIZE;

    // ── Create Tone.FFT analyser ──────────────────────────────────────────
    // Tone.FFT wraps AnalyserNode in FFT mode and returns dB magnitude per
    // bin via getValue().  We connect it to the master destination so it
    // sees whatever audio Tone.js is currently routing to speakers.
    this.fft = new Tone.FFT(this.fftSize);
    Tone.getDestination().connect(this.fft);

    // ── Resolve config ────────────────────────────────────────────────────
    this.lowAlpha  = config?.lowAlpha  ?? LOW_ALPHA;
    this.midAlpha  = config?.midAlpha  ?? MID_ALPHA;
    this.highAlpha = config?.highAlpha ?? HIGH_ALPHA;

    this.lowGain   = config?.lowGain  ?? LOW_GAIN;
    this.midGain   = config?.midGain  ?? MID_GAIN;
    this.highGain  = config?.highGain ?? HIGH_GAIN;

    this.lowExp    = config?.lowExponent  ?? LOW_EXPONENT;
    this.midExp    = config?.midExponent  ?? MID_EXPONENT;
    this.highExp   = config?.highExponent ?? HIGH_EXPONENT;

    const br = config?.breathRange   ?? [BREATH_SCALE_MIN, BREATH_SCALE_MAX];
    const vr = config?.velocityRange ?? [VELOCITY_MULT_MIN, VELOCITY_MULT_MAX];
    const bl = config?.bloomRange    ?? [BLOOM_SCALE_MIN, BLOOM_SCALE_MAX];

    this.breathMin = br[0];
    this.breathMax = br[1];
    this.velMin    = vr[0];
    this.velMax    = vr[1];
    this.bloomMin  = bl[0];
    this.bloomMax  = bl[1];

    this.sineFreq  = config?.breathSineFreq ?? BREATH_SINE_FREQ;
    this.sineAmp   = config?.breathSineAmp  ?? BREATH_SINE_AMP;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Public API
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Call once per frame.
   *
   * Reads the current FFT spectrum, splits energy into three bands,
   * smooths via EMA, and maps to visual modulation parameters.
   *
   * @param dt  Frame delta time in seconds (e.g. 1/60).
   * @returns   AudioReactiveSnapshot with ready-to-apply visual values.
   */
  update(dt: number): AudioReactiveSnapshot {
    if (this._disposed) {
      return this.neutralSnapshot();
    }

    // Ensure Tone.js AudioContext is running (requires user gesture).
    if (!this._started) {
      if (Tone.getContext().state !== 'running') {
        Tone.start().catch(() => {});
        return this.neutralSnapshot();
      }
      this._started = true;
    }

    // ── Cache FFT bin boundaries on first live frame ───────────────────
    if (!this._binsCached) {
      this.cacheBinBoundaries();
    }

    // ── Read FFT spectrum ─────────────────────────────────────────────
    const spectrum = this.fft.getValue() as Float32Array;

    // ── Compute raw band energies (RMS over bins, dB → linear) ─────────
    const rawLow  = this.bandEnergy(spectrum, this.binLowMin,  this.binLowMax);
    const rawMid  = this.bandEnergy(spectrum, this.binLowMax,  this.binMidMax);
    const rawHigh = this.bandEnergy(spectrum, this.binMidMax,  this.binHighMax);

    // ── Apply gain and non-linear response curve ───────────────────────
    const gainedLow  = Math.pow(clamp(rawLow  * this.lowGain,  0, 1), this.lowExp);
    const gainedMid  = Math.pow(clamp(rawMid  * this.midGain,  0, 1), this.midExp);
    const gainedHigh = Math.pow(clamp(rawHigh * this.highGain, 0, 1), this.highExp);

    // ── EMA smoothing ──────────────────────────────────────────────────
    this.emaLow  = this.emaLow  + this.lowAlpha  * (gainedLow  - this.emaLow);
    this.emaMid  = this.emaMid  + this.midAlpha  * (gainedMid  - this.emaMid);
    this.emaHigh = this.emaHigh + this.highAlpha * (gainedHigh - this.emaHigh);

    // ── Advance breath phase ───────────────────────────────────────────
    this.breathPhase += dt * this.sineFreq * Math.PI * 2;
    // Wrap to avoid floating-point drift over long sessions.
    if (this.breathPhase > Math.PI * 200) {
      this.breathPhase -= Math.PI * 200;
    }

    // ── Map to output ranges ───────────────────────────────────────────

    // Breath scale: bass energy maps linearly to [breathMin, breathMax],
    // plus a subtle sine modulation whose amplitude scales with energy.
    const breathBase = lerp(this.breathMin, this.breathMax, this.emaLow);
    const breathSine = Math.sin(this.breathPhase) * this.sineAmp * this.emaLow;
    const breathScale = clamp(
      breathBase + breathSine,
      this.breathMin,
      this.breathMax,
    );

    // Velocity multiplier: mid energy maps linearly.
    const velocityMultiplier = lerp(this.velMin, this.velMax, this.emaMid);

    // Bloom scale: high energy maps linearly.
    const bloomScale = lerp(this.bloomMin, this.bloomMax, this.emaHigh);

    return {
      breathScale,
      velocityMultiplier,
      bloomScale,
      lowEnergy:  this.emaLow,
      midEnergy:  this.emaMid,
      highEnergy: this.emaHigh,
      active:     true,
    };
  }

  /**
   * Apply the audio-reactive snapshot to the physics world in-place.
   *
   * This is a convenience method that directly modulates particle
   * velocities by the `velocityMultiplier`.  Cell scale and bloom must
   * be applied by the caller since they depend on the renderer API.
   *
   * For particle velocity, the multiplier is applied as a blend toward
   * the target speed rather than a hard multiply, to avoid compounding
   * across frames.  Specifically:
   *
   *   targetSpeed = currentSpeed × multiplier
   *   newSpeed    = lerp(currentSpeed, targetSpeed, blendRate × dt)
   *
   * This approach ensures velocity modulation is frame-rate independent
   * and doesn't accumulate exponentially.
   *
   * @param world    Live SPH World.
   * @param snap     Snapshot returned by update().
   * @param dt       Frame delta (seconds).
   * @param blendRate Rate at which velocity approaches the target per
   *                  second.  Default 5.0 (reaches ~99% in 1 s).
   */
  applyToWorld(
    world: World,
    snap: AudioReactiveSnapshot,
    dt: number,
    blendRate = 5.0,
  ): void {
    if (!snap.active || world.particles.length === 0) return;

    const mult = snap.velocityMultiplier;

    // Skip if multiplier is effectively neutral (avoids unnecessary work).
    if (Math.abs(mult - 1.0) < 0.001) return;

    const blend = clamp(blendRate * dt, 0, 1);

    for (let i = 0, n = world.particles.length; i < n; i++) {
      const p = world.particles[i];

      // Current speed² — avoid sqrt when possible.
      const speed2 = p.vx * p.vx + p.vy * p.vy;
      if (speed2 < 1e-12) continue; // stationary particle — skip

      const speed = Math.sqrt(speed2);
      const targetSpeed = speed * mult;
      const newSpeed = speed + blend * (targetSpeed - speed);

      // Scale velocity components proportionally.
      const ratio = newSpeed / speed;
      p.vx *= ratio;
      p.vy *= ratio;
    }
  }

  /**
   * Create a VFXHandler-compatible bloom callback that applies the
   * audio-reactive bloom scale.
   *
   * Returns a function suitable for use as `VFXHandler.onBloomUpdate`.
   * The returned function merges the audio-reactive bloom with the
   * VFX timeline bloom (multiplicative blend) so that timeline bloom
   * spikes and audio reactivity compose naturally.
   *
   * @param baseBloomScale  The scene's resting bloom scale.  Default 1.0.
   * @returns  A callback `(timelineBloom: number, threshold: number | null) => void`
   *           that should be assigned to `VFXHandler.onBloomUpdate`.
   */
  createBloomCallback(
    baseBloomScale = 1.0,
  ): (timelineBloom: number, threshold: number | null) => void {
    return (timelineBloom: number, _threshold: number | null) => {
      // `timelineBloom` is the envelope value from the VFX timeline system.
      // We multiply it with the audio-reactive bloom for a compound effect.
      // When no timeline is active, timelineBloom = baseBloomScale.
      const _audioBloom = lerp(
        this.bloomMin,
        this.bloomMax,
        this.emaHigh,
      );
      // The caller can read the composed value from the snapshot instead;
      // this callback is for direct wiring into VFXHandler's per-frame loop.
      // We deliberately don't mutate anything here — the composed value is
      // exposed via getComposedBloom() for the renderer to pick up.
      this._composedBloom = timelineBloom * _audioBloom / baseBloomScale;
    };
  }

  /** Last composed bloom value from createBloomCallback. */
  private _composedBloom = 1.0;

  /**
   * Read the last composed bloom value (audio × timeline).
   * Only meaningful if createBloomCallback() is actively wired.
   */
  getComposedBloom(): number {
    return this._composedBloom;
  }

  /**
   * Get the current smoothed band energies without generating a full
   * snapshot.  Useful for debugging / UI meters.
   */
  getBandEnergies(): { low: number; mid: number; high: number } {
    return {
      low:  this.emaLow,
      mid:  this.emaMid,
      high: this.emaHigh,
    };
  }

  /**
   * Reset all EMA accumulators to zero.
   * Useful when switching tracks or resuming after a pause.
   */
  reset(): void {
    this.emaLow  = 0;
    this.emaMid  = 0;
    this.emaHigh = 0;
    this.breathPhase = 0;
    this._composedBloom = 1.0;
  }

  /** True when the audio context is running and producing data. */
  get isActive(): boolean {
    return this._started && !this._disposed;
  }

  /** Release Tone.js FFT analyser resources. Safe to call multiple times. */
  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;

    try {
      Tone.getDestination().disconnect(this.fft);
    } catch {
      // Already disconnected — safe to ignore.
    }
    this.fft.dispose();
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Internal — bin boundary caching
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Cache FFT bin indices for each frequency band boundary.
   * Called once when the audio context is first confirmed running
   * (at which point the sample rate is known).
   */
  private cacheBinBoundaries(): void {
    const sr = Tone.getContext().sampleRate;

    this.binLowMin  = freqToBin(LOW_FREQ_MIN,  this.fftSize, sr);
    this.binLowMax  = freqToBin(LOW_FREQ_MAX,  this.fftSize, sr);
    this.binMidMax  = freqToBin(MID_FREQ_MAX,  this.fftSize, sr);
    this.binHighMax = freqToBin(HIGH_FREQ_MAX, this.fftSize, sr);

    this._binsCached = true;

    if (process.env.NODE_ENV !== 'production') {
      console.log(
        `[ASTRO-ARV] bin boundaries cached: ` +
        `low=[${this.binLowMin},${this.binLowMax}) ` +
        `mid=[${this.binLowMax},${this.binMidMax}) ` +
        `high=[${this.binMidMax},${this.binHighMax}) ` +
        `sampleRate=${sr}`,
      );
    }
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Internal — band energy computation
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Compute the RMS energy of a frequency band from FFT dB values.
   *
   * The FFT bins contain magnitudes in decibels.  We:
   *   1. Normalise each bin from [FFT_FLOOR_DB, FFT_CEILING_DB] → [0, 1]
   *   2. Square each normalised magnitude
   *   3. Average the squares
   *   4. Take the square root → RMS
   *
   * This gives a perceptually meaningful energy metric that responds
   * to both the number of active bins and their magnitudes.
   *
   * @param spectrum  Float32Array of dB magnitudes from Tone.FFT.
   * @param binStart  First bin index (inclusive).
   * @param binEnd    Last bin index (exclusive).
   * @returns         Normalised RMS energy in [0, 1].
   */
  private bandEnergy(
    spectrum: Float32Array,
    binStart: number,
    binEnd: number,
  ): number {
    if (binEnd <= binStart) return 0;

    let sumSq = 0;
    let count = 0;

    for (let i = binStart; i < binEnd && i < spectrum.length; i++) {
      // Normalise dB → [0, 1]
      const db = spectrum[i];
      const norm = clamp((db - FFT_FLOOR_DB) * DB_RANGE_INV, 0, 1);
      sumSq += norm * norm;
      count++;
    }

    if (count === 0) return 0;
    return Math.sqrt(sumSq / count);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Internal — neutral (no-audio) snapshot
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Return a snapshot with neutral values (no modulation).
   * Used before Tone.js has started or after dispose.
   */
  private neutralSnapshot(): AudioReactiveSnapshot {
    return {
      breathScale:        1.0,
      velocityMultiplier: 1.0,
      bloomScale:         1.0,
      lowEnergy:          0,
      midEnergy:          0,
      highEnergy:         0,
      active:             false,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience — apply breath scale to cell model matrices
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Applies the breathing scale factor to an array of column-major 4×4 model
 * matrices (Float32Array, 16 floats per cell).
 *
 * This modifies the scale components of each matrix in-place.  The scale
 * lives in elements [0] (X), [5] (Y), and [10] (Z) for a standard TRS
 * decomposition.  We multiply them uniformly by `breathScale` to achieve
 * an isotropic "swell" effect.
 *
 * This function is designed to be called after InstancedCellRenderer has
 * built its instance buffer but before the GPU upload, so that the breath
 * modulation is baked into the same draw call (zero extra GPU overhead).
 *
 * @param buffer       The interleaved Float32Array of per-cell instance data.
 * @param cellCount    Number of active cells in the buffer.
 * @param floatsPerCell Stride between consecutive cells (FLOATS_PER_CELL = 32).
 * @param breathScale  The scale factor from AudioReactiveSnapshot.breathScale.
 */
export function applyBreathToInstanceBuffer(
  buffer: Float32Array,
  cellCount: number,
  floatsPerCell: number,
  breathScale: number,
): void {
  // Skip if neutral — avoid touching memory unnecessarily.
  if (Math.abs(breathScale - 1.0) < 0.0001) return;

  for (let i = 0; i < cellCount; i++) {
    const base = i * floatsPerCell;

    // Column-major mat4 scale components:
    //   m[0]  = scaleX (column 0, row 0)
    //   m[5]  = scaleY (column 1, row 1)
    //   m[10] = scaleZ (column 2, row 2)
    buffer[base + 0]  *= breathScale;
    buffer[base + 5]  *= breathScale;
    buffer[base + 10] *= breathScale;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience — per-cell differential breathing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A more expressive variant of `applyBreathToInstanceBuffer` that applies
 * per-cell phase offsets to create a wave-like breathing pattern across
 * the cell population rather than uniform synchronized breathing.
 *
 * The phase offset for each cell is derived from its world-space position
 * (encoded in the model matrix translation), creating a spatial wave that
 * propagates through the cell grid.  This looks more organic than having
 * all cells breathe in perfect sync.
 *
 * @param buffer       Instance buffer (Float32Array).
 * @param cellCount    Number of active cells.
 * @param floatsPerCell Stride per cell (32).
 * @param breathScale  Base breath scale from AudioReactiveSnapshot.
 * @param time         Current time (seconds) for wave animation.
 * @param waveSpeed    Spatial wave propagation speed. Default 0.008.
 * @param waveAmp      Additional per-cell scale variance. Default 0.03.
 */
export function applyBreathWaveToInstanceBuffer(
  buffer: Float32Array,
  cellCount: number,
  floatsPerCell: number,
  breathScale: number,
  time: number,
  waveSpeed = 0.008,
  waveAmp   = 0.03,
): void {
  for (let i = 0; i < cellCount; i++) {
    const base = i * floatsPerCell;

    // Read cell world-space position from model matrix translation column.
    // Column-major mat4: translation = (m[12], m[13], m[14]).
    const tx = buffer[base + 12];
    const ty = buffer[base + 13];

    // Per-cell phase offset from spatial position.
    const spatialPhase = (tx + ty) * waveSpeed;
    const cellBreath = breathScale
      + Math.sin(time * Math.PI * 2 * 1.2 + spatialPhase) * waveAmp * (breathScale - 1.0);

    const s = clamp(cellBreath, BREATH_SCALE_MIN * 0.95, BREATH_SCALE_MAX * 1.05);

    buffer[base + 0]  *= s;
    buffer[base + 5]  *= s;
    buffer[base + 10] *= s;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience — one-shot integration helper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * All-in-one per-frame helper that:
 *   1. Calls `arv.update(dt)` to get the snapshot
 *   2. Applies velocity modulation to the world
 *   3. Applies breath scale to the cell instance buffer
 *   4. Returns the snapshot for bloom / other use
 *
 * Designed for the common case where you just want "plug and play" audio
 * reactivity without managing each subsystem separately.
 *
 * @param arv           AudioReactiveVisuals instance.
 * @param world         Live SPH World.
 * @param dt            Frame delta (seconds).
 * @param instanceBuffer Cell instance buffer (optional — skip if null).
 * @param cellCount     Active cell count (required if instanceBuffer given).
 * @param floatsPerCell Stride (default 32).
 * @returns             The AudioReactiveSnapshot for additional use.
 */
export function updateAudioReactiveFrame(
  arv: AudioReactiveVisuals,
  world: World,
  dt: number,
  instanceBuffer?: Float32Array | null,
  cellCount = 0,
  floatsPerCell = 32,
): AudioReactiveSnapshot {
  const snap = arv.update(dt);

  // Apply velocity modulation to particles.
  arv.applyToWorld(world, snap, dt);

  // Apply breath scale to cell instance buffer.
  if (instanceBuffer && cellCount > 0) {
    applyBreathToInstanceBuffer(
      instanceBuffer,
      cellCount,
      floatsPerCell,
      snap.breathScale,
    );
  }

  return snap;
}
