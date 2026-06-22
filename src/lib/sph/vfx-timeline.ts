/**
 * src/lib/sph/vfx-timeline.ts — M768: VFX Timeline Sequencer
 *
 * Visual Effect Timeline System
 * ─────────────────────────────────────────────────────────────────────────────
 * Orchestrates a sequence of VFX events — shockwave → bloom spike → particle
 * burst → screen flash — arranged on a time axis and played in order.  Used
 * for dramatic, multi-layered feedback on collision contacts, topology changes
 * (cell merge/split/death), epoch transitions, and player interactions.
 *
 * Design philosophy
 * ─────────────────
 * Real-time VFX in games and film rarely consist of a single effect.  A heavy
 * collision might produce:
 *
 *   t = 0.00 s  —  shockwave ring expands from contact point
 *   t = 0.03 s  —  bloom intensity spikes (bright flash at contact)
 *   t = 0.06 s  —  particle burst radiates outward (petals / sparks)
 *   t = 0.12 s  —  screen flash white overlay (optional, heavy impacts)
 *   t = 0.20 s  —  camera shake (if camera controller is wired)
 *
 * This module lets you describe such a choreography declaratively as a
 * VFXTimeline — an ordered list of VFXKeyframes — and play it with a single
 * `play()` call.  The VFXTimelinePlayer advances the timeline each frame,
 * firing callbacks when each keyframe's time is reached.
 *
 * Integration points
 * ──────────────────
 *   - CollisionFXSystem (collision-fx-system.ts)   — flower petal bursts
 *   - ContactSparkSystem (contact-sparks.ts)        — spark particle bursts
 *   - ATBloomPostProcess (at-bloom-postprocess.ts)  — bloom intensity spikes
 *   - RippleEffect (ripple-effect.ts)               — shockwave ring
 *   - TransitionSystem (transition-system.ts)        — cell transitions
 *   - CollisionEventDispatcher (collision/CollisionEvents.ts) — trigger source
 *
 * Each subsystem remains fully independent.  The timeline simply invokes
 * their public APIs at the right time.  No subsystem needs to know about
 * the timeline — it just receives normal `emit()`, `setParams()`, or
 * `start()` calls.
 *
 * Performance
 * ───────────
 * The player maintains a sorted playhead index into the keyframe array,
 * so per-frame cost is O(k) where k is the number of keyframes firing
 * this frame (typically 0–2).  Inactive timelines cost nothing.
 *
 * Upstream references:
 *   upstream/theatre-js — keyframe sequencing, sheet/sequence model
 *   upstream/animation-editor — node-based timeline graph
 *   src/lib/tween-system.ts — Easing functions
 *   src/lib/sph/transition-system.ts — TransitionSystem stagger pattern
 *
 * [ASTRO-VFX-TIMELINE] debug prefix.
 */

import { Easing, type EasingFn } from '../tween-system';
import type { CollisionContactInfo, CollisionEvent } from './collision/CollisionEvents';
import type { CollisionEventDispatcher }              from './collision/CollisionEvents';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Default timeline duration cap — prevents runaway timelines (seconds) */
const MAX_TIMELINE_DURATION = 5.0;

/** Maximum number of concurrent playing timelines per player instance */
const MAX_CONCURRENT_TIMELINES = 32;

/** Minimum interval between auto-triggered timelines (seconds) */
const AUTO_TRIGGER_COOLDOWN = 0.08;

// ─────────────────────────────────────────────────────────────────────────────
// VFX Event Types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The built-in VFX event kinds.
 *
 *   shockwave      — expanding ring distortion from an origin point
 *   bloom_spike    — temporary bloom intensity increase
 *   particle_burst — radial particle emission (sparks or petals)
 *   screen_flash   — full-screen additive colour overlay
 *   camera_shake   — camera displacement oscillation
 *   custom         — user-defined callback, for anything not covered above
 */
export type VFXEventKind =
  | 'shockwave'
  | 'bloom_spike'
  | 'particle_burst'
  | 'screen_flash'
  | 'camera_shake'
  | 'custom';

// ─────────────────────────────────────────────────────────────────────────────
// VFX Event Parameter Interfaces
// ─────────────────────────────────────────────────────────────────────────────

/** Parameters for a shockwave ring effect. */
export interface ShockwaveParams {
  /** World-space origin of the shockwave ring. */
  originX: number;
  originY: number;
  /** Maximum radius the ring expands to (domain units). Default 200. */
  maxRadius?: number;
  /** Ring expansion duration (seconds). Default 0.4. */
  expandDuration?: number;
  /** Ring thickness as a fraction of maxRadius. Default 0.12. */
  thickness?: number;
  /** Peak distortion amplitude (pixels of UV offset). Default 8. */
  amplitude?: number;
}

/** Parameters for a bloom intensity spike. */
export interface BloomSpikeParams {
  /** Peak bloom scale multiplier (applied on top of base). Default 3.0. */
  peakScale?: number;
  /** Attack time — seconds to ramp from base to peak. Default 0.03. */
  attack?: number;
  /** Decay time — seconds to ramp from peak back to base. Default 0.25. */
  decay?: number;
  /** Easing for the decay phase. Default easeOut. */
  decayEasing?: EasingFn;
  /** Optional: threshold override during spike. */
  thresholdOverride?: number;
}

/** Parameters for a radial particle burst. */
export interface ParticleBurstParams {
  /** World-space burst origin. */
  originX: number;
  originY: number;
  /** Number of particles to emit. Default 32. */
  count?: number;
  /** Burst speed (units/s). Default 160. */
  speed?: number;
  /** Speed randomness jitter (units/s). Default 80. */
  speedJitter?: number;
  /**
   * Burst style — 'sparks' routes to ContactSparkSystem,
   * 'petals' routes to CollisionFXSystem.
   * Default 'sparks'.
   */
  style?: 'sparks' | 'petals';
  /** Impulse magnitude fed to the particle system. Default 100. */
  impulse?: number;
  /** Contact normal direction (used for directional bursts). */
  normalX?: number;
  normalY?: number;
}

/** Parameters for a full-screen flash overlay. */
export interface ScreenFlashParams {
  /** Flash colour [r, g, b] in 0..1. Default [1, 1, 1] (white). */
  color?: [number, number, number];
  /** Peak opacity. Default 0.6. */
  peakOpacity?: number;
  /** Attack time (seconds). Default 0.02. */
  attack?: number;
  /** Decay time (seconds). Default 0.2. */
  decay?: number;
  /** Easing for the decay phase. Default easeOut. */
  decayEasing?: EasingFn;
}

/** Parameters for a camera shake effect. */
export interface CameraShakeParams {
  /** Peak displacement (pixels). Default 6. */
  amplitude?: number;
  /** Shake frequency (Hz). Default 30. */
  frequency?: number;
  /** Duration of the shake (seconds). Default 0.25. */
  duration?: number;
  /** Damping — how quickly shake decays (0=instant, 1=linear). Default 0.8. */
  damping?: number;
}

/** Parameters for a user-defined custom event. */
export interface CustomVFXParams {
  /** Arbitrary payload passed to the handler callback. */
  [key: string]: unknown;
}

/** Union of all VFX parameter types, discriminated by event kind. */
export type VFXEventParams =
  | { kind: 'shockwave';      params: ShockwaveParams }
  | { kind: 'bloom_spike';    params: BloomSpikeParams }
  | { kind: 'particle_burst'; params: ParticleBurstParams }
  | { kind: 'screen_flash';   params: ScreenFlashParams }
  | { kind: 'camera_shake';   params: CameraShakeParams }
  | { kind: 'custom';         params: CustomVFXParams };

// ─────────────────────────────────────────────────────────────────────────────
// VFX Keyframe
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A single keyframe on the VFX timeline.
 *
 * Keyframes are sorted by `time` and executed sequentially by the player.
 * Multiple keyframes can share the same time for simultaneous effects.
 */
export interface VFXKeyframe {
  /** Time offset from timeline start (seconds). Must be ≥ 0. */
  time: number;

  /** Which VFX event kind to fire. */
  kind: VFXEventKind;

  /** Parameters specific to the event kind. */
  params: ShockwaveParams
        | BloomSpikeParams
        | ParticleBurstParams
        | ScreenFlashParams
        | CameraShakeParams
        | CustomVFXParams;

  /**
   * Optional label for debugging / logging.
   * Printed in [ASTRO-VFX-TIMELINE] console messages.
   */
  label?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// VFX Timeline
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A complete VFX timeline — an ordered choreography of effects.
 *
 * Construct one from keyframes, then pass it to VFXTimelinePlayer.play().
 */
export interface VFXTimeline {
  /** Human-readable name for logging / debug. */
  name: string;

  /** Ordered list of keyframes (sorted by time ascending). */
  keyframes: VFXKeyframe[];

  /**
   * Optional: global intensity multiplier applied to all keyframe params.
   * Lets you scale an entire timeline up/down based on impulse magnitude.
   * Default 1.0.
   */
  intensity?: number;

  /**
   * Optional: global time scale.  Values > 1 speed up, < 1 slow down.
   * Default 1.0.
   */
  timeScale?: number;

  /**
   * If true, the timeline loops (wraps playhead back to 0 after completion).
   * Default false.
   */
  loop?: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Active timeline instance (internal)
// ─────────────────────────────────────────────────────────────────────────────

interface ActiveTimeline {
  /** Unique playback ID. */
  id: number;
  /** The timeline definition being played. */
  timeline: VFXTimeline;
  /** Current playhead position in seconds (accounts for timeScale). */
  playhead: number;
  /** Index of the next keyframe to fire. */
  nextIdx: number;
  /** Resolved intensity (timeline.intensity ?? 1). */
  intensity: number;
  /** Resolved timeScale (timeline.timeScale ?? 1). */
  timeScale: number;
  /** Total duration = last keyframe time / timeScale. */
  duration: number;
  /** True if still playing. */
  active: boolean;
  /** Optional completion callback. */
  onComplete: (() => void) | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Active continuous effects (internal state for envelope-driven effects)
// ─────────────────────────────────────────────────────────────────────────────

/** A bloom spike that ramps up then decays — tracked across frames. */
interface ActiveBloomSpike {
  startTime: number;
  attack: number;
  decay: number;
  peakScale: number;
  decayEasing: EasingFn;
  thresholdOverride: number | null;
  baseBloomScale: number;
}

/** A screen flash that ramps up then decays. */
interface ActiveScreenFlash {
  startTime: number;
  attack: number;
  decay: number;
  peakOpacity: number;
  decayEasing: EasingFn;
  color: [number, number, number];
}

/** A camera shake that oscillates and decays. */
interface ActiveCameraShake {
  startTime: number;
  amplitude: number;
  frequency: number;
  duration: number;
  damping: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// VFX Handler Interface
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Adapter interface — the player doesn't call subsystems directly.
 * Instead, you wire up a VFXHandler that bridges to your actual VFX
 * subsystems (RippleEffect, ATBloomPostProcess, ContactSparkSystem, etc.).
 *
 * All methods are optional — implement only the ones your scene uses.
 */
export interface VFXHandler {
  /**
   * Fire a shockwave ring at the given origin.
   * Typically bridges to RippleEffect.addImpulse() or a custom shader.
   */
  onShockwave?(params: ShockwaveParams, intensity: number): void;

  /**
   * Trigger a bloom intensity spike.
   * Typically bridges to ATBloomPostProcess.setParams({ bloomScale }).
   * The handler receives the computed bloom envelope value each frame
   * via onBloomUpdate() instead.
   */
  onBloomSpike?(params: BloomSpikeParams, intensity: number): void;

  /**
   * Fire a particle burst.
   * Typically bridges to ContactSparkSystem.emit() or CollisionFXSystem.emit().
   */
  onParticleBurst?(params: ParticleBurstParams, intensity: number): void;

  /**
   * Flash the screen with an overlay.
   * Typically draws a full-screen quad with additive blending.
   */
  onScreenFlash?(params: ScreenFlashParams, intensity: number): void;

  /**
   * Start a camera shake.
   * Typically bridges to CameraController shake methods.
   */
  onCameraShake?(params: CameraShakeParams, intensity: number): void;

  /**
   * Fire a custom user-defined event.
   */
  onCustom?(params: CustomVFXParams, intensity: number): void;

  // ── Continuous envelope callbacks (called every frame while active) ──

  /**
   * Called every frame while a bloom spike envelope is active.
   * @param bloomScale  The current computed bloom scale value.
   * @param threshold   If a threshold override was set, the value; else null.
   */
  onBloomUpdate?(bloomScale: number, threshold: number | null): void;

  /**
   * Called every frame while a screen flash is active.
   * @param r, g, b  Flash colour (0..1).
   * @param alpha     Current flash opacity (0..1).
   */
  onFlashUpdate?(r: number, g: number, b: number, alpha: number): void;

  /**
   * Called every frame while a camera shake is active.
   * @param dx, dy  Displacement to apply to the camera (pixels).
   */
  onShakeUpdate?(dx: number, dy: number): void;
}

// ─────────────────────────────────────────────────────────────────────────────
// VFX Timeline Player
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The core runtime — drives VFX timelines forward each frame.
 *
 * Usage:
 *
 *   const player = new VFXTimelinePlayer(handler);
 *
 *   // On collision:
 *   player.play(VFX_PRESETS.heavyCollision(cx, cy), {
 *     intensity: impulse / 100,
 *     onComplete: () => console.log('VFX done'),
 *   });
 *
 *   // Each frame:
 *   player.update(dt);
 *
 * The player manages multiple concurrent timelines, fires keyframe
 * callbacks through the VFXHandler, and maintains envelope state for
 * bloom spikes, screen flashes, and camera shakes.
 */
export class VFXTimelinePlayer {
  private handler: VFXHandler;
  private actives: ActiveTimeline[] = [];
  private nextId = 1;

  // ── Continuous effect state pools ──
  private bloomSpikes: ActiveBloomSpike[] = [];
  private screenFlashes: ActiveScreenFlash[] = [];
  private cameraShakes: ActiveCameraShake[] = [];

  /** Monotonic clock — advanced by update(dt). */
  private clock = 0;

  /** Cooldown tracker for auto-trigger deduplication. */
  private lastTriggerTime = -Infinity;

  constructor(handler: VFXHandler) {
    this.handler = handler;
  }

  // ── Playback ────────────────────────────────────────────────────────────

  /**
   * Start playing a VFX timeline.
   *
   * @param timeline   The timeline to play.
   * @param options    Optional overrides for intensity, timeScale, callback.
   * @returns  A unique playback ID (can be passed to cancel()).
   */
  play(
    timeline: VFXTimeline,
    options?: {
      intensity?: number;
      timeScale?: number;
      onComplete?: () => void;
    },
  ): number {
    // Enforce concurrency limit
    if (this.actives.length >= MAX_CONCURRENT_TIMELINES) {
      // Remove the oldest completed or least-progressed timeline
      this._evictOldest();
    }

    // Sort keyframes by time (defensive — presets should already be sorted)
    const sorted = [...timeline.keyframes].sort((a, b) => a.time - b.time);

    const intensity = (options?.intensity ?? 1) * (timeline.intensity ?? 1);
    const timeScale = (options?.timeScale ?? 1) * (timeline.timeScale ?? 1);
    const lastTime  = sorted.length > 0 ? sorted[sorted.length - 1].time : 0;
    const duration  = Math.min(lastTime / Math.max(timeScale, 0.001), MAX_TIMELINE_DURATION);

    const id = this.nextId++;
    const active: ActiveTimeline = {
      id,
      timeline: { ...timeline, keyframes: sorted },
      playhead: 0,
      nextIdx: 0,
      intensity,
      timeScale,
      duration,
      active: true,
      onComplete: options?.onComplete ?? null,
    };

    this.actives.push(active);

    if (process.env.NODE_ENV !== 'production') {
      console.log(
        `[ASTRO-VFX-TIMELINE] play "${timeline.name}" id=${id} ` +
        `keyframes=${sorted.length} intensity=${intensity.toFixed(2)} ` +
        `timeScale=${timeScale.toFixed(2)} duration=${duration.toFixed(3)}s`,
      );
    }

    return id;
  }

  /**
   * Cancel a playing timeline by its playback ID.
   * Returns true if found and cancelled.
   */
  cancel(playbackId: number): boolean {
    for (let i = 0; i < this.actives.length; i++) {
      if (this.actives[i].id === playbackId) {
        this.actives[i].active = false;
        return true;
      }
    }
    return false;
  }

  /** Cancel all active timelines and clear continuous effects. */
  cancelAll(): void {
    for (const a of this.actives) a.active = false;
    this.actives.length = 0;
    this.bloomSpikes.length = 0;
    this.screenFlashes.length = 0;
    this.cameraShakes.length = 0;
  }

  // ── Frame update ────────────────────────────────────────────────────────

  /**
   * Advance all active timelines by `dt` seconds.
   * Call once per animation frame.
   *
   * @param dt  Delta time in seconds since last frame.
   */
  update(dt: number): void {
    this.clock += dt;

    // ── Advance timelines and fire keyframes ──
    for (let i = this.actives.length - 1; i >= 0; i--) {
      const a = this.actives[i];
      if (!a.active) {
        this._removeActive(i);
        continue;
      }

      a.playhead += dt * a.timeScale;
      const kfs = a.timeline.keyframes;

      // Fire all keyframes whose time has been reached
      while (a.nextIdx < kfs.length && kfs[a.nextIdx].time <= a.playhead) {
        this._fireKeyframe(kfs[a.nextIdx], a.intensity);
        a.nextIdx++;
      }

      // Check completion
      if (a.nextIdx >= kfs.length && a.playhead >= a.duration) {
        if (a.timeline.loop) {
          // Wrap around
          a.playhead = 0;
          a.nextIdx = 0;
        } else {
          a.active = false;
          if (a.onComplete) {
            try { a.onComplete(); } catch (e) {
              console.warn('[ASTRO-VFX-TIMELINE] onComplete error:', e);
            }
          }
          this._removeActive(i);
        }
      }
    }

    // ── Update continuous envelopes ──
    this._updateBloomSpikes(dt);
    this._updateScreenFlashes(dt);
    this._updateCameraShakes(dt);
  }

  // ── Fire individual keyframes ───────────────────────────────────────────

  private _fireKeyframe(kf: VFXKeyframe, intensity: number): void {
    if (process.env.NODE_ENV !== 'production' && kf.label) {
      console.log(
        `[ASTRO-VFX-TIMELINE] fire "${kf.label}" kind=${kf.kind} ` +
        `t=${kf.time.toFixed(3)}s intensity=${intensity.toFixed(2)}`,
      );
    }

    switch (kf.kind) {
      case 'shockwave':
        this.handler.onShockwave?.(kf.params as ShockwaveParams, intensity);
        break;

      case 'bloom_spike': {
        const bp = kf.params as BloomSpikeParams;
        this.handler.onBloomSpike?.(bp, intensity);
        // Register continuous envelope
        this.bloomSpikes.push({
          startTime: this.clock,
          attack: bp.attack ?? 0.03,
          decay: bp.decay ?? 0.25,
          peakScale: (bp.peakScale ?? 3.0) * intensity,
          decayEasing: bp.decayEasing ?? Easing.easeOut,
          thresholdOverride: bp.thresholdOverride ?? null,
          baseBloomScale: 1.0,
        });
        break;
      }

      case 'particle_burst':
        this.handler.onParticleBurst?.(kf.params as ParticleBurstParams, intensity);
        break;

      case 'screen_flash': {
        const fp = kf.params as ScreenFlashParams;
        this.handler.onScreenFlash?.(fp, intensity);
        // Register continuous envelope
        this.screenFlashes.push({
          startTime: this.clock,
          attack: fp.attack ?? 0.02,
          decay: fp.decay ?? 0.2,
          peakOpacity: (fp.peakOpacity ?? 0.6) * Math.min(intensity, 1.5),
          decayEasing: fp.decayEasing ?? Easing.easeOut,
          color: fp.color ?? [1, 1, 1],
        });
        break;
      }

      case 'camera_shake': {
        const sp = kf.params as CameraShakeParams;
        this.handler.onCameraShake?.(sp, intensity);
        // Register continuous envelope
        this.cameraShakes.push({
          startTime: this.clock,
          amplitude: (sp.amplitude ?? 6) * intensity,
          frequency: sp.frequency ?? 30,
          duration: sp.duration ?? 0.25,
          damping: sp.damping ?? 0.8,
        });
        break;
      }

      case 'custom':
        this.handler.onCustom?.(kf.params as CustomVFXParams, intensity);
        break;
    }
  }

  // ── Continuous envelope updates ─────────────────────────────────────────

  private _updateBloomSpikes(_dt: number): void {
    if (this.bloomSpikes.length === 0) return;

    let maxScale = 1.0;
    let thresholdOverride: number | null = null;

    for (let i = this.bloomSpikes.length - 1; i >= 0; i--) {
      const spike = this.bloomSpikes[i];
      const elapsed = this.clock - spike.startTime;
      const totalDur = spike.attack + spike.decay;

      if (elapsed >= totalDur) {
        // Remove expired spike
        const last = this.bloomSpikes.length - 1;
        if (i !== last) this.bloomSpikes[i] = this.bloomSpikes[last];
        this.bloomSpikes.pop();
        continue;
      }

      let value: number;
      if (elapsed < spike.attack) {
        // Attack phase — linear ramp to peak
        const t = spike.attack > 0 ? elapsed / spike.attack : 1;
        value = spike.baseBloomScale + (spike.peakScale - spike.baseBloomScale) * t;
      } else {
        // Decay phase — eased ramp from peak back to base
        const decayElapsed = elapsed - spike.attack;
        const t = spike.decay > 0 ? decayElapsed / spike.decay : 1;
        const eased = spike.decayEasing(Math.min(t, 1));
        value = spike.peakScale + (spike.baseBloomScale - spike.peakScale) * eased;
      }

      if (value > maxScale) maxScale = value;
      if (spike.thresholdOverride !== null) {
        thresholdOverride = spike.thresholdOverride;
      }
    }

    this.handler.onBloomUpdate?.(maxScale, thresholdOverride);
  }

  private _updateScreenFlashes(_dt: number): void {
    if (this.screenFlashes.length === 0) return;

    // Composite all active flashes (additive)
    let totalR = 0, totalG = 0, totalB = 0, totalA = 0;

    for (let i = this.screenFlashes.length - 1; i >= 0; i--) {
      const flash = this.screenFlashes[i];
      const elapsed = this.clock - flash.startTime;
      const totalDur = flash.attack + flash.decay;

      if (elapsed >= totalDur) {
        const last = this.screenFlashes.length - 1;
        if (i !== last) this.screenFlashes[i] = this.screenFlashes[last];
        this.screenFlashes.pop();
        continue;
      }

      let alpha: number;
      if (elapsed < flash.attack) {
        const t = flash.attack > 0 ? elapsed / flash.attack : 1;
        alpha = flash.peakOpacity * t;
      } else {
        const decayElapsed = elapsed - flash.attack;
        const t = flash.decay > 0 ? decayElapsed / flash.decay : 1;
        const eased = flash.decayEasing(Math.min(t, 1));
        alpha = flash.peakOpacity * (1 - eased);
      }

      totalR += flash.color[0] * alpha;
      totalG += flash.color[1] * alpha;
      totalB += flash.color[2] * alpha;
      totalA += alpha;
    }

    if (totalA > 0) {
      // Clamp composited values
      const a = Math.min(totalA, 1);
      this.handler.onFlashUpdate?.(
        Math.min(totalR, 1),
        Math.min(totalG, 1),
        Math.min(totalB, 1),
        a,
      );
    }
  }

  private _updateCameraShakes(_dt: number): void {
    if (this.cameraShakes.length === 0) return;

    // Combine all active shakes (sum displacements)
    let totalDx = 0, totalDy = 0;

    for (let i = this.cameraShakes.length - 1; i >= 0; i--) {
      const shake = this.cameraShakes[i];
      const elapsed = this.clock - shake.startTime;

      if (elapsed >= shake.duration) {
        const last = this.cameraShakes.length - 1;
        if (i !== last) this.cameraShakes[i] = this.cameraShakes[last];
        this.cameraShakes.pop();
        continue;
      }

      // Decay envelope: linear or exponential depending on damping
      const t = elapsed / shake.duration;
      const envelope = Math.pow(1 - t, 1 / Math.max(shake.damping, 0.01));
      const amp = shake.amplitude * envelope;

      // Oscillating displacement using sin/cos at different frequencies
      // for uncorrelated X/Y motion
      const phase = elapsed * shake.frequency * Math.PI * 2;
      const dx = amp * Math.sin(phase);
      const dy = amp * Math.cos(phase * 1.37 + 0.5); // offset phase for Y

      totalDx += dx;
      totalDy += dy;
    }

    if (Math.abs(totalDx) > 0.01 || Math.abs(totalDy) > 0.01) {
      this.handler.onShakeUpdate?.(totalDx, totalDy);
    }
  }

  // ── Helpers ─────────────────────────────────────────────────────────────

  private _removeActive(idx: number): void {
    const last = this.actives.length - 1;
    if (idx !== last) this.actives[idx] = this.actives[last];
    this.actives.pop();
  }

  private _evictOldest(): void {
    // Find the timeline with the smallest remaining time
    let minRemaining = Infinity;
    let minIdx = 0;
    for (let i = 0; i < this.actives.length; i++) {
      const remaining = this.actives[i].duration - this.actives[i].playhead;
      if (remaining < minRemaining) {
        minRemaining = remaining;
        minIdx = i;
      }
    }
    this.actives[minIdx].active = false;
    this._removeActive(minIdx);
  }

  // ── Introspection ───────────────────────────────────────────────────────

  /** Number of currently playing timelines. */
  get playingCount(): number {
    return this.actives.length;
  }

  /** True if any timeline, bloom spike, screen flash, or shake is active. */
  get isActive(): boolean {
    return this.actives.length > 0
        || this.bloomSpikes.length > 0
        || this.screenFlashes.length > 0
        || this.cameraShakes.length > 0;
  }

  /** Current internal clock value (seconds). */
  get currentTime(): number {
    return this.clock;
  }

  /** Number of active continuous bloom spike envelopes. */
  get activeBloomCount(): number {
    return this.bloomSpikes.length;
  }

  /** Number of active screen flash envelopes. */
  get activeFlashCount(): number {
    return this.screenFlashes.length;
  }

  /** Number of active camera shake envelopes. */
  get activeShakeCount(): number {
    return this.cameraShakes.length;
  }

  /** Reset the player to a clean state. */
  reset(): void {
    this.cancelAll();
    this.clock = 0;
    this.lastTriggerTime = -Infinity;
    this.nextId = 1;
  }

  /** Destroy the player. */
  destroy(): void {
    this.reset();
    console.log('[ASTRO-VFX-TIMELINE] player destroyed');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Timeline Builder (fluent API)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fluent builder for constructing VFXTimeline instances.
 *
 * Usage:
 *   const tl = VFXTimelineBuilder.create('my-collision')
 *     .shockwave(0.00, { originX: cx, originY: cy })
 *     .bloomSpike(0.03, { peakScale: 2.5 })
 *     .particleBurst(0.06, { originX: cx, originY: cy, count: 40 })
 *     .screenFlash(0.12, { peakOpacity: 0.3 })
 *     .build();
 */
export class VFXTimelineBuilder {
  private _name: string;
  private _keyframes: VFXKeyframe[] = [];
  private _intensity = 1.0;
  private _timeScale = 1.0;
  private _loop = false;

  private constructor(name: string) {
    this._name = name;
  }

  /** Create a new builder. */
  static create(name: string): VFXTimelineBuilder {
    return new VFXTimelineBuilder(name);
  }

  /** Set the global intensity multiplier. */
  intensity(value: number): this {
    this._intensity = value;
    return this;
  }

  /** Set the global time scale. */
  timeScale(value: number): this {
    this._timeScale = value;
    return this;
  }

  /** Enable looping. */
  loop(enabled = true): this {
    this._loop = enabled;
    return this;
  }

  /** Add a shockwave keyframe. */
  shockwave(time: number, params: ShockwaveParams, label?: string): this {
    this._keyframes.push({ time, kind: 'shockwave', params, label: label ?? 'shockwave' });
    return this;
  }

  /** Add a bloom spike keyframe. */
  bloomSpike(time: number, params: BloomSpikeParams = {}, label?: string): this {
    this._keyframes.push({ time, kind: 'bloom_spike', params, label: label ?? 'bloom-spike' });
    return this;
  }

  /** Add a particle burst keyframe. */
  particleBurst(time: number, params: ParticleBurstParams, label?: string): this {
    this._keyframes.push({ time, kind: 'particle_burst', params, label: label ?? 'particle-burst' });
    return this;
  }

  /** Add a screen flash keyframe. */
  screenFlash(time: number, params: ScreenFlashParams = {}, label?: string): this {
    this._keyframes.push({ time, kind: 'screen_flash', params, label: label ?? 'screen-flash' });
    return this;
  }

  /** Add a camera shake keyframe. */
  cameraShake(time: number, params: CameraShakeParams = {}, label?: string): this {
    this._keyframes.push({ time, kind: 'camera_shake', params, label: label ?? 'camera-shake' });
    return this;
  }

  /** Add a custom event keyframe. */
  custom(time: number, params: CustomVFXParams, label?: string): this {
    this._keyframes.push({ time, kind: 'custom', params, label: label ?? 'custom' });
    return this;
  }

  /** Build the final VFXTimeline. */
  build(): VFXTimeline {
    return {
      name: this._name,
      keyframes: [...this._keyframes].sort((a, b) => a.time - b.time),
      intensity: this._intensity,
      timeScale: this._timeScale,
      loop: this._loop,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preset Timelines
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Pre-built VFX timeline presets for common game/visualisation events.
 *
 * Each factory accepts the world-space origin and returns a ready-to-play
 * VFXTimeline.  Pass to VFXTimelinePlayer.play() with optional intensity
 * override.
 *
 * Usage:
 *   player.play(VFX_PRESETS.heavyCollision(cx, cy), { intensity: impulse / 80 });
 */
export const VFX_PRESETS = {

  /**
   * Heavy collision — full dramatic sequence.
   *
   *   t=0.00  shockwave ring
   *   t=0.02  bloom spike (bright flash)
   *   t=0.05  particle burst (sparks)
   *   t=0.05  particle burst (petals, overlapping)
   *   t=0.08  screen flash (white)
   *   t=0.10  camera shake
   */
  heavyCollision(originX: number, originY: number): VFXTimeline {
    return VFXTimelineBuilder.create('heavy-collision')
      .shockwave(0.00, {
        originX, originY,
        maxRadius: 250,
        expandDuration: 0.45,
        thickness: 0.10,
        amplitude: 10,
      }, 'shockwave-ring')
      .bloomSpike(0.02, {
        peakScale: 4.0,
        attack: 0.02,
        decay: 0.30,
        decayEasing: Easing.easeOut,
      }, 'bloom-flash')
      .particleBurst(0.05, {
        originX, originY,
        count: 48,
        speed: 200,
        speedJitter: 100,
        style: 'sparks',
        impulse: 150,
      }, 'spark-burst')
      .particleBurst(0.05, {
        originX, originY,
        count: 24,
        speed: 120,
        speedJitter: 60,
        style: 'petals',
        impulse: 120,
      }, 'petal-burst')
      .screenFlash(0.08, {
        color: [1, 0.95, 0.9],
        peakOpacity: 0.5,
        attack: 0.015,
        decay: 0.18,
      }, 'white-flash')
      .cameraShake(0.10, {
        amplitude: 8,
        frequency: 28,
        duration: 0.3,
        damping: 0.75,
      }, 'impact-shake')
      .build();
  },

  /**
   * Light collision — subtle feedback.
   *
   *   t=0.00  small shockwave
   *   t=0.02  gentle bloom spike
   *   t=0.04  small spark burst
   */
  lightCollision(originX: number, originY: number): VFXTimeline {
    return VFXTimelineBuilder.create('light-collision')
      .shockwave(0.00, {
        originX, originY,
        maxRadius: 100,
        expandDuration: 0.3,
        thickness: 0.15,
        amplitude: 4,
      }, 'shockwave-small')
      .bloomSpike(0.02, {
        peakScale: 1.8,
        attack: 0.02,
        decay: 0.15,
      }, 'bloom-gentle')
      .particleBurst(0.04, {
        originX, originY,
        count: 12,
        speed: 100,
        speedJitter: 50,
        style: 'sparks',
        impulse: 40,
      }, 'sparks-small')
      .build();
  },

  /**
   * Cell merge — two cells fusing together.
   *
   *   t=0.00  bloom spike (warm glow)
   *   t=0.03  shockwave (inward, smaller)
   *   t=0.06  petal burst (celebratory)
   *   t=0.10  gentle screen flash (warm tone)
   */
  cellMerge(originX: number, originY: number): VFXTimeline {
    return VFXTimelineBuilder.create('cell-merge')
      .bloomSpike(0.00, {
        peakScale: 2.5,
        attack: 0.05,
        decay: 0.35,
        decayEasing: Easing.easeInOut,
      }, 'merge-glow')
      .shockwave(0.03, {
        originX, originY,
        maxRadius: 120,
        expandDuration: 0.35,
        thickness: 0.18,
        amplitude: 5,
      }, 'merge-wave')
      .particleBurst(0.06, {
        originX, originY,
        count: 36,
        speed: 90,
        speedJitter: 40,
        style: 'petals',
        impulse: 80,
      }, 'merge-petals')
      .screenFlash(0.10, {
        color: [1, 0.85, 0.6],
        peakOpacity: 0.2,
        attack: 0.03,
        decay: 0.25,
      }, 'merge-flash')
      .build();
  },

  /**
   * Cell split / mitosis — a cell dividing.
   *
   *   t=0.00  sharp bloom spike
   *   t=0.02  shockwave (fast, tight)
   *   t=0.04  dual particle bursts (opposing directions)
   *   t=0.06  camera shake (brief)
   */
  cellSplit(originX: number, originY: number): VFXTimeline {
    return VFXTimelineBuilder.create('cell-split')
      .bloomSpike(0.00, {
        peakScale: 3.5,
        attack: 0.01,
        decay: 0.20,
      }, 'split-flash')
      .shockwave(0.02, {
        originX, originY,
        maxRadius: 150,
        expandDuration: 0.25,
        thickness: 0.08,
        amplitude: 7,
      }, 'split-wave')
      .particleBurst(0.04, {
        originX, originY,
        count: 20,
        speed: 180,
        speedJitter: 70,
        style: 'sparks',
        impulse: 100,
        normalX: -1, normalY: 0,
      }, 'split-sparks-left')
      .particleBurst(0.04, {
        originX, originY,
        count: 20,
        speed: 180,
        speedJitter: 70,
        style: 'sparks',
        impulse: 100,
        normalX: 1, normalY: 0,
      }, 'split-sparks-right')
      .cameraShake(0.06, {
        amplitude: 4,
        frequency: 35,
        duration: 0.15,
        damping: 0.9,
      }, 'split-shake')
      .build();
  },

  /**
   * Cell death / dissolution — a cell being removed.
   *
   *   t=0.00  bloom spike (cold blue)
   *   t=0.03  shockwave (slow, large)
   *   t=0.06  petal scatter (outward dissolution)
   *   t=0.15  screen flash (dim)
   */
  cellDeath(originX: number, originY: number): VFXTimeline {
    return VFXTimelineBuilder.create('cell-death')
      .bloomSpike(0.00, {
        peakScale: 2.0,
        attack: 0.03,
        decay: 0.40,
        decayEasing: Easing.easeOut,
      }, 'death-glow')
      .shockwave(0.03, {
        originX, originY,
        maxRadius: 180,
        expandDuration: 0.5,
        thickness: 0.20,
        amplitude: 6,
      }, 'death-wave')
      .particleBurst(0.06, {
        originX, originY,
        count: 48,
        speed: 60,
        speedJitter: 30,
        style: 'petals',
        impulse: 60,
      }, 'death-dissolve')
      .screenFlash(0.15, {
        color: [0.6, 0.7, 1.0],
        peakOpacity: 0.15,
        attack: 0.04,
        decay: 0.30,
      }, 'death-flash')
      .build();
  },

  /**
   * Epoch transition — global dramatic punctuation.
   *
   *   t=0.00  screen flash (warm gold)
   *   t=0.05  bloom spike (broad)
   *   t=0.10  camera shake (gentle, prolonged)
   */
  epochTransition(): VFXTimeline {
    return VFXTimelineBuilder.create('epoch-transition')
      .screenFlash(0.00, {
        color: [1, 0.9, 0.7],
        peakOpacity: 0.35,
        attack: 0.04,
        decay: 0.40,
        decayEasing: Easing.easeInOut,
      }, 'epoch-flash')
      .bloomSpike(0.05, {
        peakScale: 2.0,
        attack: 0.06,
        decay: 0.50,
        decayEasing: Easing.easeInOut,
      }, 'epoch-bloom')
      .cameraShake(0.10, {
        amplitude: 3,
        frequency: 12,
        duration: 0.5,
        damping: 0.6,
      }, 'epoch-shake')
      .build();
  },

  /**
   * Topology change — graph structure mutation (edge add/remove).
   *
   *   t=0.00  shockwave (small)
   *   t=0.02  bloom spike
   *   t=0.05  spark burst
   */
  topologyChange(originX: number, originY: number): VFXTimeline {
    return VFXTimelineBuilder.create('topology-change')
      .shockwave(0.00, {
        originX, originY,
        maxRadius: 80,
        expandDuration: 0.25,
        thickness: 0.14,
        amplitude: 3,
      }, 'topo-wave')
      .bloomSpike(0.02, {
        peakScale: 1.5,
        attack: 0.02,
        decay: 0.12,
      }, 'topo-bloom')
      .particleBurst(0.05, {
        originX, originY,
        count: 8,
        speed: 80,
        speedJitter: 40,
        style: 'sparks',
        impulse: 30,
      }, 'topo-sparks')
      .build();
  },

} as const;

// ─────────────────────────────────────────────────────────────────────────────
// Collision Event → Timeline Wiring Helper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Automatically maps collision events to VFX timeline presets based on
 * impulse magnitude.
 *
 * Light impacts (< lightThreshold)     → lightCollision preset
 * Heavy impacts (≥ heavyThreshold)     → heavyCollision preset
 * Medium impacts (between)             → lightCollision with scaled intensity
 *
 * Usage:
 *   const unsub = wireCollisionVFX(player, dispatcher, {
 *     impulseScale: 0.01,
 *     heavyThreshold: 0.6,
 *   });
 *
 * @returns Unsubscribe function.
 */
export interface CollisionVFXWiringConfig {
  /**
   * Maps raw impulse to normalised [0, 1].
   * normalised = clamp(rawImpulse × impulseScale, 0, 1).
   * Default 0.008.
   */
  impulseScale?: number;

  /** Normalised threshold above which heavyCollision is triggered. Default 0.6. */
  heavyThreshold?: number;

  /** Normalised threshold below which no VFX plays at all. Default 0.05. */
  minThreshold?: number;

  /** Depth multiplier for estimating impulse from contact depth. Default 120. */
  depthMultiplier?: number;
}

export function wireCollisionVFX(
  player: VFXTimelinePlayer,
  dispatcher: CollisionEventDispatcher,
  config: CollisionVFXWiringConfig = {},
): () => void {
  const impulseScale    = config.impulseScale    ?? 0.008;
  const heavyThreshold  = config.heavyThreshold  ?? 0.6;
  const minThreshold    = config.minThreshold    ?? 0.05;
  const depthMultiplier = config.depthMultiplier ?? 120;

  let lastTrigger = -Infinity;

  return dispatcher.onCollisionEnter((evt: CollisionEvent) => {
    if (!evt.contact) return;

    const rawImpulse = evt.contact.depth * depthMultiplier;
    const norm = Math.min(Math.max(rawImpulse * impulseScale, 0), 1);

    if (norm < minThreshold) return;

    // Cooldown — avoid VFX spam from rapid collision chatter
    if (evt.time - lastTrigger < AUTO_TRIGGER_COOLDOWN) return;
    lastTrigger = evt.time;

    const cx = (evt.contact.pointA.x + evt.contact.pointB.x) * 0.5;
    const cy = (evt.contact.pointA.y + evt.contact.pointB.y) * 0.5;

    if (norm >= heavyThreshold) {
      player.play(VFX_PRESETS.heavyCollision(cx, cy), {
        intensity: norm,
      });
    } else {
      player.play(VFX_PRESETS.lightCollision(cx, cy), {
        intensity: norm / heavyThreshold,
      });
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Canvas 2D Screen Flash Renderer (convenience)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A minimal Canvas 2D screen flash renderer that can be wired as the
 * `onFlashUpdate` callback.
 *
 * Usage:
 *   const flashRenderer = new CanvasScreenFlash(canvas);
 *   const handler: VFXHandler = {
 *     onFlashUpdate: (r, g, b, a) => flashRenderer.draw(r, g, b, a),
 *   };
 */
export class CanvasScreenFlash {
  private ctx: CanvasRenderingContext2D | null;
  private width: number;
  private height: number;

  constructor(canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d');
    this.width = canvas.width;
    this.height = canvas.height;
  }

  /** Call from onFlashUpdate. */
  draw(r: number, g: number, b: number, alpha: number): void {
    if (!this.ctx || alpha < 0.001) return;

    const ctx = this.ctx;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = alpha;
    ctx.fillStyle = `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;
    ctx.fillRect(0, 0, this.width, this.height);
    ctx.restore();
  }

  /** Update dimensions (call on resize). */
  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global singleton (optional convenience)
// ─────────────────────────────────────────────────────────────────────────────

let _globalVFXPlayer: VFXTimelinePlayer | null = null;

/**
 * Get or create the global VFXTimelinePlayer singleton.
 * Requires a handler to be provided on first call.
 */
export function getGlobalVFXPlayer(handler?: VFXHandler): VFXTimelinePlayer {
  if (!_globalVFXPlayer) {
    if (!handler) {
      throw new Error(
        '[ASTRO-VFX-TIMELINE] getGlobalVFXPlayer() requires a handler on first call',
      );
    }
    _globalVFXPlayer = new VFXTimelinePlayer(handler);
    console.log('[ASTRO-VFX-TIMELINE] global singleton created');
  }
  return _globalVFXPlayer;
}

/**
 * Replace the global VFXTimelinePlayer singleton (e.g. for testing).
 */
export function setGlobalVFXPlayer(player: VFXTimelinePlayer | null): void {
  if (_globalVFXPlayer && _globalVFXPlayer !== player) {
    _globalVFXPlayer.destroy();
  }
  _globalVFXPlayer = player;
}
