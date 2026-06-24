/**
 * src/lib/sph/cinematic-camera.ts — M789: Cinematic Camera System
 * ─────────────────────────────────────────────────────────────────────────────
 * A full-featured cinematic camera controller for the SPH world, providing
 * smooth follow, orbit, dolly zoom (Hitchcock vertigo effect), procedural
 * shake, spline-path playback, and auto-framing of bounding-box targets.
 *
 * Design philosophy
 * ─────────────────
 * Film and game cameras are not simply "placed" — they are choreographed.
 * A good camera conveys narrative weight: a slow dolly-in builds tension,
 * a sudden shake sells impact, an orbit reveals structure.  This module
 * treats the camera as a first-class animation subject with layered
 * behaviours that compose additively:
 *
 *   Layer 0  —  Base transform (position + orientation)
 *   Layer 1  —  Smooth follow (lerp toward a target entity / point)
 *   Layer 2  —  Orbit (rotation around the focus point)
 *   Layer 3  —  Dolly zoom (simultaneous translate + FOV to keep subject size)
 *   Layer 4  —  Spline path (position follows a Catmull-Rom spline)
 *   Layer 5  —  Auto-framing (compute best position/zoom for a bounding box)
 *   Layer 6  —  Procedural shake (Perlin-seeded additive offset)
 *
 * Each layer writes into an additive accumulator.  Layers can be enabled
 * or disabled independently, blended with per-layer weights, and
 * transitioned smoothly via crossfade.
 *
 * Integration points
 * ──────────────────
 *   - CameraController (src/lib/CameraController.ts) — 2D gaze camera, can
 *     feed normalised mouse into orbit yaw/pitch
 *   - ATSceneCompositor (at-scene-compositor.ts)      — provides cell BBox
 *   - VFXTimeline (vfx-timeline.ts)                   — triggers shake events
 *   - TransitionSystem (transition-system.ts)          — epoch camera moves
 *   - WorldOrchestrator (world-orchestrator.ts)        — scene preset switching
 *   - TweenSystem (tween-system.ts)                   — Easing, SplineInterpolation
 *   - Math library (math/)                            — Vec3, Quat, Mat4, Box3
 *
 * Upstream references:
 *   upstream/theatre-js     — keyframe sequencing, sheet model
 *   upstream/animation-editor — node-based camera graph
 *   upstream/unreal-renderer — UE5 cinematic camera concepts
 *
 * [ASTRO-CINECAM] debug prefix.
 */

import { Vec3 }  from '../math/Vec3';
import { Quat }  from '../math/Quat';
import { Box3 }  from '../math/Box3';
import { Easing, type EasingFn } from '../tween-system';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// [orphan-precise] /** Default smooth-follow lerp speed (frame-rate-independent) */
const DEFAULT_FOLLOW_SPEED    = 0.08;

/** Default orbit angular speed (radians / second for auto-orbit) */
const DEFAULT_ORBIT_SPEED     = 0.0;

/** Default field-of-view in degrees */
const DEFAULT_FOV             = 60.0;

/** Minimum FOV clamp for dolly zoom (degrees) */
const MIN_FOV                 = 5.0;

/** Maximum FOV clamp for dolly zoom (degrees) */
const MAX_FOV                 = 150.0;

/** Minimum camera distance (prevents clipping into target) */
const MIN_DISTANCE            = 0.1;

/** Maximum camera distance */
const MAX_DISTANCE            = 10000.0;

/** Shake decay rate — higher = faster falloff */
const SHAKE_DECAY_RATE        = 5.0;

/** Auto-framing padding multiplier (1.2 = 20% border) */
const AUTOFRAME_PADDING       = 1.2;

/** Spline path default playback speed (normalised t per second) */
const SPLINE_DEFAULT_SPEED    = 0.1;

/** Maximum number of spline control points */
const MAX_SPLINE_POINTS       = 256;

/** Epsilon for float comparisons */
const EPSILON                 = 1e-6;

/** Number of Perlin octaves for shake noise */
const SHAKE_OCTAVES           = 3;

// ─────────────────────────────────────────────────────────────────────────────
// Types & Interfaces
// ─────────────────────────────────────────────────────────────────────────────

/** Camera operation mode — determines which layers are active by default. */
export type CameraMode =
  | 'free'        // manual position/rotation, no auto-behaviour
  | 'follow'      // smooth follow a target
  | 'orbit'       // orbit around a focus point
  | 'dolly'       // dolly zoom (vertigo effect)
  | 'path'        // spline path playback
  | 'autoframe';  // auto-frame a bounding region

/** Shake profile — different trauma patterns. */
export type ShakeProfile =
  | 'impact'      // sharp, short
  | 'rumble'      // low-frequency sustained
  | 'handheld'    // subtle organic drift
  | 'explosion';  // heavy, multi-axis

/** Spline path loop behaviour. */
export type PathLoopMode =
  | 'once'        // play once, stop at end
  | 'loop'        // restart from beginning
  | 'pingpong'    // reverse at endpoints
  | 'hold';       // play once, hold last frame

/**
 * Smooth-follow configuration.
 */
export interface FollowConfig {
  /** Target position to follow */
  target:        Vec3;
  /** Interpolation speed (0..1 per frame at 60fps) */
  speed:         number;
  /** Offset from target in local space (e.g., behind and above) */
  offset:        Vec3;
  /** Optional look-at bias — blend between following direction and target */
  lookAtBias:    number;
  /** Dead zone radius — don't follow if within this distance */
  deadZone:      number;
  /** Lead factor — predict target motion based on velocity */
  leadFactor:    number;
}

/**
 * Orbit configuration.
 */
export interface OrbitConfig {
  /** Centre point to orbit around */
  centre:        Vec3;
  /** Orbit distance from centre */
  distance:      number;
  /** Current yaw angle (radians) — around Y axis */
  yaw:           number;
  /** Current pitch angle (radians) — around X axis */
  pitch:         number;
  /** Auto-rotation speed (radians/sec), 0 = manual only */
  autoSpeed:     number;
  /** Pitch clamp: [min, max] in radians */
  pitchClamp:    [number, number];
  /** Mouse/touch sensitivity multiplier */
  sensitivity:   number;
  /** Inertia — how much angular velocity carries after input stops */
  inertia:       number;
}

/**
 * Dolly zoom (vertigo / Hitchcock effect) configuration.
 *
 * The camera simultaneously moves toward/away from the subject while
 * adjusting FOV so the subject maintains the same apparent size.
 * The background stretches/compresses dramatically.
 */
export interface DollyZoomConfig {
  /** Subject position (the point that maintains apparent size) */
  subject:       Vec3;
  /** Start distance from subject */
  startDistance:  number;
  /** End distance from subject */
  endDistance:    number;
  /** Duration of the dolly zoom in seconds */
  duration:      number;
  /** Easing function */
  easing:        EasingFn;
  /** Whether to auto-reverse after completion */
  autoReverse:   boolean;
}

/**
 * Shake event — a single trauma impulse that decays over time.
 */
export interface ShakeEvent {
  /** Remaining trauma amount (0..1) */
  trauma:        number;
  /** Shake frequency (Hz) */
  frequency:     number;
  /** Translational amplitude (world units) */
  amplitude:     Vec3;
  /** Rotational amplitude (radians) */
  rotAmplitude:  Vec3;
  /** Decay rate override (0 = use global default) */
  decayRate:     number;
  /** Noise seed for this event */
  seed:          number;
  /** Shake profile */
  profile:       ShakeProfile;
}

/**
 * Spline path control point with optional timing / easing.
 */
export interface PathPoint {
  /** Position on the path */
  position:      Vec3;
  /** Optional look-at target at this point (null = forward along spline) */
  lookAt:        Vec3 | null;
  /** Optional FOV at this point (null = keep current) */
  fov:           number | null;
  /** Optional easing to reach this point from the previous */
  easing:        EasingFn | null;
  /** Optional speed multiplier at this segment (1.0 = normal) */
  speedMult:     number;
}

/**
 * Auto-framing configuration.
 */
export interface AutoFrameConfig {
  /** Bounding box of the region to frame */
  bounds:        Box3;
  /** Padding multiplier (1.0 = tight, 1.5 = generous) */
  padding:       number;
  /** Transition speed (lerp rate) */
  speed:         number;
  /** Preferred viewing angle (direction camera looks from) */
  viewDirection: Vec3;
  /** Minimum distance from bounds centre */
  minDistance:    number;
  /** Maximum distance from bounds centre */
  maxDistance:    number;
}

/**
 * Complete camera state snapshot — serialisable for presets / undo.
 */
export interface CameraSnapshot {
  position:      [number, number, number];
  rotation:      [number, number, number, number]; // quaternion xyzw
  fov:           number;
  near:          number;
  far:           number;
  mode:          CameraMode;
  timestamp:     number;
}

/**
 * Camera transition request — animate between two snapshots.
 */
export interface CameraTransition {
  from:          CameraSnapshot;
  to:            CameraSnapshot;
  duration:      number;
  easing:        EasingFn;
  elapsed:       number;
  onComplete:    (() => void) | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Noise utility (simplified Perlin for shake)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Simple 1-D value noise with linear interpolation.
 * Not cryptographic — just smooth, deterministic randomness for shake.
 */
function hashNoise(n: number): number {
  // Integer hash → float in [0, 1)
  let x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  x -= Math.floor(x);
  return x;
}

function smoothNoise1D(t: number, seed: number): number {
  const i  = Math.floor(t);
  const f  = t - i;
  const s  = f * f * (3 - 2 * f); // smoothstep
  const a  = hashNoise(i + seed);
  const b  = hashNoise(i + 1 + seed);
  return a + (b - a) * s;
}

/**
 * Multi-octave noise for shake — returns value in [-1, 1].
 */
function fbmNoise(t: number, seed: number, octaves: number = SHAKE_OCTAVES): number {
  let value  = 0;
  let amp    = 1.0;
  let freq   = 1.0;
  let maxAmp = 0;
  for (let i = 0; i < octaves; i++) {
    value  += smoothNoise1D(t * freq, seed + i * 1000) * amp;
    maxAmp += amp;
    amp    *= 0.5;
    freq   *= 2.0;
  }
  return (value / maxAmp) * 2 - 1; // remap [0, 1] → [-1, 1]
}

// ─────────────────────────────────────────────────────────────────────────────
// Shake profile presets
// ─────────────────────────────────────────────────────────────────────────────

const SHAKE_PROFILES: Record<ShakeProfile, {
  frequency: number;
  amplitude: Vec3;
  rotAmplitude: Vec3;
  decayRate: number;
}> = {
  impact: {
    frequency:    25,
    amplitude:    new Vec3(0.3, 0.3, 0.1),
    rotAmplitude: new Vec3(0.02, 0.02, 0.01),
    decayRate:    8.0,
  },
  rumble: {
    frequency:    8,
    amplitude:    new Vec3(0.15, 0.1, 0.05),
    rotAmplitude: new Vec3(0.005, 0.005, 0.003),
    decayRate:    2.0,
  },
  handheld: {
    frequency:    3,
    amplitude:    new Vec3(0.05, 0.04, 0.02),
    rotAmplitude: new Vec3(0.003, 0.002, 0.001),
    decayRate:    0.5,
  },
  explosion: {
    frequency:    30,
    amplitude:    new Vec3(1.0, 0.8, 0.5),
    rotAmplitude: new Vec3(0.05, 0.04, 0.03),
    decayRate:    4.0,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Default configurations
// ─────────────────────────────────────────────────────────────────────────────

function defaultFollowConfig(): FollowConfig {
  return {
    target:     Vec3.zero(),
    speed:      DEFAULT_FOLLOW_SPEED,
    offset:     new Vec3(0, 5, 15),
    lookAtBias: 1.0,
    deadZone:   0.0,
    leadFactor: 0.0,
  };
}

function defaultOrbitConfig(): OrbitConfig {
  return {
    centre:      Vec3.zero(),
    distance:    20,
    yaw:         0,
    pitch:       -0.3,
    autoSpeed:   DEFAULT_ORBIT_SPEED,
    pitchClamp:  [-Math.PI * 0.45, Math.PI * 0.45],
    sensitivity: 0.005,
    inertia:     0.92,
  };
}

function defaultDollyZoomConfig(): DollyZoomConfig {
  return {
    subject:       Vec3.zero(),
    startDistance:  20,
    endDistance:    5,
    duration:       2.0,
    easing:        Easing.easeInOut,
    autoReverse:   false,
  };
}

function defaultAutoFrameConfig(): AutoFrameConfig {
  return {
    bounds:        new Box3(new Vec3(-10, -10, -10), new Vec3(10, 10, 10)),
    padding:       AUTOFRAME_PADDING,
    speed:         0.05,
    viewDirection: new Vec3(0, 0.3, 1).normalize(),
    minDistance:    5,
    maxDistance:    500,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// CinematicCamera
// ─────────────────────────────────────────────────────────────────────────────

/**
 * CinematicCamera — layered camera behaviours for cinematic scene control.
 *
 * Usage:
 * ```ts
 * const cam = new CinematicCamera();
 * cam.setMode('follow');
 * cam.follow.target = playerPosition;
 *
 * // Each frame:
 * cam.update(dt);
 * const { position, rotation, fov } = cam.getState();
 * ```
 *
 * Shake can be triggered at any time, regardless of mode:
 * ```ts
 * cam.addShake('impact', 0.8); // 80% trauma
 * ```
 *
 * Spline path playback:
 * ```ts
 * cam.setPath(controlPoints, 'loop', 0.15);
 * cam.setMode('path');
 * cam.playPath();
 * ```
 */
export class CinematicCamera {

  // ── Core state ──────────────────────────────────────────────────────────
  private _position:       Vec3  = new Vec3(0, 5, 20);
  private _rotation:       Quat  = Quat.identity();
  private _fov:            number = DEFAULT_FOV;
  private _near:           number = 0.1;
  private _far:            number = 5000;
  private _mode:           CameraMode = 'free';

  // ── Layer configs (public for direct mutation) ──────────────────────────
  readonly follow:         FollowConfig     = defaultFollowConfig();
  readonly orbit:          OrbitConfig      = defaultOrbitConfig();
  readonly dollyZoom:      DollyZoomConfig  = defaultDollyZoomConfig();
  readonly autoFrame:      AutoFrameConfig  = defaultAutoFrameConfig();

  // ── Orbit angular velocity (for inertia) ────────────────────────────────
  private _orbitYawVel:    number = 0;
  private _orbitPitchVel:  number = 0;

  // ── Dolly zoom runtime ──────────────────────────────────────────────────
  private _dollyElapsed:   number = 0;
  private _dollyActive:    boolean = false;
  private _dollyReversing: boolean = false;
  private _dollyStartFov:  number = DEFAULT_FOV;

  // ── Shake system ────────────────────────────────────────────────────────
  private _shakeEvents:    ShakeEvent[] = [];
  private _shakeOffset:    Vec3  = Vec3.zero();
  private _shakeRotOffset: Vec3  = Vec3.zero();
  private _shakeTime:      number = 0;

  // ── Spline path ─────────────────────────────────────────────────────────
  private _pathPoints:     PathPoint[] = [];
  private _pathT:          number = 0;
  private _pathSpeed:      number = SPLINE_DEFAULT_SPEED;
  private _pathLoop:       PathLoopMode = 'once';
  private _pathPlaying:    boolean = false;
  private _pathDirection:  1 | -1 = 1; // for pingpong

  // ── Transition ──────────────────────────────────────────────────────────
  private _transition:     CameraTransition | null = null;

  // ── Follow velocity tracking (for lead prediction) ──────────────────────
  private _prevTarget:     Vec3  = Vec3.zero();
  private _targetVelocity: Vec3  = Vec3.zero();

  // ── Debug ───────────────────────────────────────────────────────────────
  private _debugEnabled:   boolean = false;

  constructor(config?: Partial<{
    position: Vec3;
    rotation: Quat;
    fov:      number;
    near:     number;
    far:      number;
    mode:     CameraMode;
  }>) {
    if (config) {
      if (config.position) this._position.copy(config.position);
      if (config.rotation) this._rotation.copy(config.rotation);
      if (config.fov  !== undefined) this._fov  = config.fov;
      if (config.near !== undefined) this._near = config.near;
      if (config.far  !== undefined) this._far  = config.far;
      if (config.mode) this._mode = config.mode;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public — Mode management
  // ─────────────────────────────────────────────────────────────────────────

  /** Get current camera mode. */
  getMode(): CameraMode { return this._mode; }

  /**
   * Set the camera mode.  Clears mode-specific runtime state on change.
   */
  setMode(mode: CameraMode): void {
    if (mode === this._mode) return;
    const prev = this._mode;
    this._mode = mode;

    // Reset mode-specific runtime on switch
    if (mode === 'dolly') {
      this._dollyElapsed   = 0;
      this._dollyActive    = true;
      this._dollyReversing = false;
      this._dollyStartFov  = this._fov;
    }
    if (mode !== 'path') {
      this._pathPlaying = false;
    }

    this._debug(`mode: ${prev} → ${mode}`);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public — Core state access
  // ─────────────────────────────────────────────────────────────────────────

  get position(): Vec3  { return this._position; }
  get rotation(): Quat  { return this._rotation; }
  get fov():      number { return this._fov; }
  get near():     number { return this._near; }
  get far():      number { return this._far; }

  set fov(v: number)  { this._fov  = clamp(v, MIN_FOV, MAX_FOV); }
  set near(v: number) { this._near = Math.max(EPSILON, v); }
  set far(v: number)  { this._far  = Math.max(this._near + EPSILON, v); }

  /** Get a serialisable snapshot of the current camera state. */
  getSnapshot(): CameraSnapshot {
    return {
      position:  this._position.toArray(),
      rotation:  this._rotation.toArray(),
      fov:       this._fov,
      near:      this._near,
      far:       this._far,
      mode:      this._mode,
      timestamp: performance.now(),
    };
  }

  /** Restore camera from a snapshot. */
  applySnapshot(snap: CameraSnapshot): void {
    this._position.set(snap.position[0], snap.position[1], snap.position[2]);
    this._rotation.set(snap.rotation[0], snap.rotation[1], snap.rotation[2], snap.rotation[3]);
    this._fov  = snap.fov;
    this._near = snap.near;
    this._far  = snap.far;
    this._mode = snap.mode;
  }

  /**
   * Compose a final state object (position + rotation + fov) suitable
   * for setting up a view matrix.  This is the output the renderer reads.
   */
  getState(): { position: Vec3; rotation: Quat; fov: number; near: number; far: number } {
    return {
      position: this._position.add(this._shakeOffset),
      rotation: applyRotationalShake(this._rotation, this._shakeRotOffset),
      fov:      this._fov,
      near:     this._near,
      far:      this._far,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public — Smooth Follow
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Convenience — set follow target position each frame.
   * (Alternatively, mutate `camera.follow.target` directly.)
   */
  setFollowTarget(target: Vec3): void {
    this.follow.target.copy(target);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public — Orbit
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Apply orbit input (e.g. from mouse drag).
   * @param deltaYaw   — horizontal rotation delta (radians)
   * @param deltaPitch — vertical rotation delta (radians)
   */
  orbitInput(deltaYaw: number, deltaPitch: number): void {
    this._orbitYawVel   += deltaYaw   * this.orbit.sensitivity;
    this._orbitPitchVel += deltaPitch  * this.orbit.sensitivity;
  }

  /**
   * Set orbit distance (zoom).
   */
  setOrbitDistance(distance: number): void {
    this.orbit.distance = clamp(distance, MIN_DISTANCE, MAX_DISTANCE);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public — Dolly Zoom
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Start a dolly zoom effect.  Call setMode('dolly') first, or this
   * will switch to dolly mode automatically.
   */
  startDollyZoom(config?: Partial<DollyZoomConfig>): void {
    if (config) {
      if (config.subject)       this.dollyZoom.subject.copy(config.subject);
      if (config.startDistance !== undefined) this.dollyZoom.startDistance = config.startDistance;
      if (config.endDistance   !== undefined) this.dollyZoom.endDistance   = config.endDistance;
      if (config.duration      !== undefined) this.dollyZoom.duration     = config.duration;
      if (config.easing)        this.dollyZoom.easing = config.easing;
      if (config.autoReverse !== undefined)  this.dollyZoom.autoReverse  = config.autoReverse;
    }

    this._dollyElapsed   = 0;
    this._dollyActive    = true;
    this._dollyReversing = false;
    this._dollyStartFov  = this._fov;

    if (this._mode !== 'dolly') this.setMode('dolly');

    this._debug(
      `dollyZoom start: dist ${this.dollyZoom.startDistance}→${this.dollyZoom.endDistance} ` +
      `over ${this.dollyZoom.duration}s`,
    );
  }

  /** Whether a dolly zoom is currently playing. */
  isDollyActive(): boolean { return this._dollyActive; }

  // ─────────────────────────────────────────────────────────────────────────
  // Public — Shake
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Add a shake impulse.  Shake is additive and mode-independent — it
   * overlays on top of whatever the active camera mode produces.
   *
   * @param profile  — one of the built-in shake profiles
   * @param trauma   — initial trauma in [0, 1]
   * @param overrides — optional per-field overrides
   */
  addShake(
    profile:   ShakeProfile = 'impact',
    trauma:    number = 0.5,
    overrides?: Partial<Pick<ShakeEvent, 'frequency' | 'amplitude' | 'rotAmplitude' | 'decayRate'>>,
  ): void {
    const preset = SHAKE_PROFILES[profile];

    const event: ShakeEvent = {
      trauma:       clamp(trauma, 0, 1),
      frequency:    overrides?.frequency    ?? preset.frequency,
      amplitude:    overrides?.amplitude    ?? preset.amplitude.clone(),
      rotAmplitude: overrides?.rotAmplitude ?? preset.rotAmplitude.clone(),
      decayRate:    overrides?.decayRate    ?? preset.decayRate,
      seed:         Math.random() * 10000,
      profile,
    };

    this._shakeEvents.push(event);
    this._debug(`shake added: profile=${profile} trauma=${trauma.toFixed(2)}`);
  }

  /**
   * Set sustained shake (e.g. for rumble that persists while a condition
   * is active).  Unlike addShake, this does not decay — call clearShake()
   * or set trauma to 0 to stop.
   */
  setSustainedShake(
    profile: ShakeProfile = 'rumble',
    trauma:  number = 0.3,
  ): void {
    this.addShake(profile, trauma, { decayRate: 0 });
  }

  /** Clear all shake events. */
  clearShake(): void {
    this._shakeEvents.length = 0;
    this._shakeOffset.set(0, 0, 0);
    this._shakeRotOffset.set(0, 0, 0);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public — Spline Path
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Set the spline path from an array of control points.
   *
   * @param points   — ordered path control points
   * @param loopMode — what to do when the path ends
   * @param speed    — normalised t advancement per second
   */
  setPath(points: PathPoint[], loopMode: PathLoopMode = 'once', speed: number = SPLINE_DEFAULT_SPEED): void {
    if (points.length < 2) {
      console.warn('[ASTRO-CINECAM] Path requires at least 2 control points');
      return;
    }
    if (points.length > MAX_SPLINE_POINTS) {
      console.warn(`[ASTRO-CINECAM] Path clamped to ${MAX_SPLINE_POINTS} points`);
      points = points.slice(0, MAX_SPLINE_POINTS);
    }

    this._pathPoints    = points;
    this._pathT         = 0;
    this._pathSpeed     = speed;
    this._pathLoop      = loopMode;
    this._pathPlaying   = false;
    this._pathDirection = 1;
  }

  /** Start path playback.  Requires setPath to have been called first. */
  playPath(): void {
    if (this._pathPoints.length < 2) {
      console.warn('[ASTRO-CINECAM] No path set — call setPath() first');
      return;
    }
    this._pathPlaying   = true;
    this._pathDirection = 1;
    if (this._mode !== 'path') this.setMode('path');
    this._debug('path playback started');
  }

  /** Pause path playback (can be resumed). */
  pausePath(): void {
    this._pathPlaying = false;
  }

  /** Resume path playback. */
  resumePath(): void {
    if (this._pathPoints.length >= 2) this._pathPlaying = true;
  }

  /** Reset path to beginning. */
  resetPath(): void {
    this._pathT         = 0;
    this._pathDirection = 1;
  }

  /** Whether the path is currently playing. */
  isPathPlaying(): boolean { return this._pathPlaying; }

  /** Current normalised path position [0, 1]. */
  getPathProgress(): number { return this._pathT; }

  // ─────────────────────────────────────────────────────────────────────────
  // Public — Auto-Framing
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Set the bounding box to auto-frame.  Call each frame if the region
   * is dynamic (e.g. moving cell cluster).
   */
  setAutoFrameBounds(bounds: Box3, padding?: number): void {
    this.autoFrame.bounds = bounds;
    if (padding !== undefined) this.autoFrame.padding = padding;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public — Transitions
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Smoothly transition from the current state to a target snapshot.
   */
  transitionTo(
    target:     CameraSnapshot,
    duration:   number = 1.0,
    easing:     EasingFn = Easing.easeInOut,
    onComplete?: () => void,
  ): void {
    this._transition = {
      from:       this.getSnapshot(),
      to:         target,
      duration:   Math.max(EPSILON, duration),
      easing,
      elapsed:    0,
      onComplete: onComplete ?? null,
    };
    this._debug(`transition started: ${duration}s`);
  }

  /** Cancel an in-progress transition. */
  cancelTransition(): void {
    this._transition = null;
  }

  /** Whether a transition is currently playing. */
  isTransitioning(): boolean { return this._transition !== null; }

  // ─────────────────────────────────────────────────────────────────────────
  // Public — Debug
  // ─────────────────────────────────────────────────────────────────────────

  /** Enable/disable debug logging. */
  setDebug(enabled: boolean): void { this._debugEnabled = enabled; }

  // ─────────────────────────────────────────────────────────────────────────
  // Public — Main update
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Advance the camera by `dt` seconds.  Call once per frame.
   *
   * Processing order:
   *   1. Transition override (if active, it drives everything)
   *   2. Mode-specific layer (follow / orbit / dolly / path / autoframe)
   *   3. Shake overlay (always, if events exist)
   */
  update(dt: number): void {
    // Clamp dt to prevent spiralling after tab-switch or debugger pause.
    dt = Math.min(dt, 0.1);

    // ── 1. Transition override ──────────────────────────────────────────
    if (this._transition) {
      this._updateTransition(dt);
      // Shake still applies on top of transitions
      this._updateShake(dt);
      return;
    }

    // ── 2. Mode-specific update ─────────────────────────────────────────
    switch (this._mode) {
      case 'follow':    this._updateFollow(dt);    break;
      case 'orbit':     this._updateOrbit(dt);     break;
      case 'dolly':     this._updateDollyZoom(dt); break;
      case 'path':      this._updatePath(dt);      break;
      case 'autoframe': this._updateAutoFrame(dt); break;
      case 'free':      /* no auto-behaviour */     break;
    }

    // ── 3. Shake overlay ────────────────────────────────────────────────
    this._updateShake(dt);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Smooth Follow
  // ─────────────────────────────────────────────────────────────────────────

  private _updateFollow(dt: number): void {
    const f = this.follow;

    // Track target velocity for lead prediction
    this._targetVelocity = f.target.sub(this._prevTarget).mul(1 / Math.max(dt, EPSILON));
    this._prevTarget.copy(f.target);

    // Compute desired position: target + offset + lead
    const lead    = this._targetVelocity.mul(f.leadFactor * dt * 10);
    const desired = f.target.add(f.offset).add(lead);

    // Dead zone — don't move if within threshold
    const dist = this._position.distanceTo(desired);
    if (dist < f.deadZone) return;

    // Frame-rate-independent lerp: alpha = 1 - (1 - speed)^(dt * 60)
    const alpha = 1 - Math.pow(1 - f.speed, dt * 60);
    this._position.lerpSelf(desired, alpha);

    // Look-at: compute rotation toward target (blended with bias)
    if (f.lookAtBias > EPSILON) {
      const lookTarget = f.target.add(lead.mul(0.5));
      const desiredRot = lookAtRotation(this._position, lookTarget);
      this._rotation   = Quat.slerp(this._rotation, desiredRot, alpha * f.lookAtBias);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Orbit
  // ─────────────────────────────────────────────────────────────────────────

  private _updateOrbit(dt: number): void {
    const o = this.orbit;

    // Apply auto-rotation
    if (Math.abs(o.autoSpeed) > EPSILON) {
      this._orbitYawVel += o.autoSpeed * dt;
    }

    // Integrate angular velocity
    o.yaw   += this._orbitYawVel;
    o.pitch += this._orbitPitchVel;

    // Clamp pitch
    o.pitch = clamp(o.pitch, o.pitchClamp[0], o.pitchClamp[1]);

    // Apply inertia decay
    this._orbitYawVel   *= Math.pow(o.inertia, dt * 60);
    this._orbitPitchVel *= Math.pow(o.inertia, dt * 60);

    // Stop tiny residual motion
    if (Math.abs(this._orbitYawVel)   < 1e-6) this._orbitYawVel   = 0;
    if (Math.abs(this._orbitPitchVel) < 1e-6) this._orbitPitchVel = 0;

    // Compute camera position on the orbit sphere
    const cosPitch = Math.cos(o.pitch);
    const offset   = new Vec3(
      Math.sin(o.yaw) * cosPitch * o.distance,
      Math.sin(o.pitch) * o.distance,
      Math.cos(o.yaw) * cosPitch * o.distance,
    );

    this._position.copy(o.centre.add(offset));
    this._rotation = lookAtRotation(this._position, o.centre);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Dolly Zoom
  // ─────────────────────────────────────────────────────────────────────────

  private _updateDollyZoom(dt: number): void {
    if (!this._dollyActive) return;

    const d = this.dollyZoom;
    this._dollyElapsed += dt;

    let t = clamp(this._dollyElapsed / d.duration, 0, 1);
    t = d.easing(t);

    if (this._dollyReversing) t = 1 - t;

    // Interpolate distance
    const currentDist = d.startDistance + (d.endDistance - d.startDistance) * t;

    // Direction from subject to camera (maintain current view direction)
    const dir = this._position.sub(d.subject);
    const dirLen = dir.length();
    const normDir = dirLen > EPSILON ? dir.mul(1 / dirLen) : new Vec3(0, 0, 1);

    // Move camera to new distance along the view direction
    this._position = d.subject.add(normDir.mul(currentDist));

    // Adjust FOV to keep the subject at the same apparent size.
    // apparent_size ∝ real_size / distance  ∝  tan(fov/2) (for perspective)
    // So: tan(newFov/2) / tan(startFov/2) = startDist / newDist
    // → newFov = 2 * atan( tan(startFov/2) * startDist / currentDist )
    const startFovRad = degToRad(this._dollyStartFov);
    const newFovRad   = 2 * Math.atan(
      Math.tan(startFovRad / 2) * d.startDistance / Math.max(currentDist, EPSILON),
    );
    this._fov = clamp(radToDeg(newFovRad), MIN_FOV, MAX_FOV);

    // Look at subject
    this._rotation = lookAtRotation(this._position, d.subject);

    // Check completion
    if (this._dollyElapsed >= d.duration) {
      if (d.autoReverse && !this._dollyReversing) {
        this._dollyElapsed   = 0;
        this._dollyReversing = true;
        this._debug('dollyZoom: reversing');
      } else {
        this._dollyActive = false;
        this._debug('dollyZoom: complete');
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Shake
  // ─────────────────────────────────────────────────────────────────────────

  private _updateShake(dt: number): void {
    this._shakeTime += dt;

    if (this._shakeEvents.length === 0) {
      this._shakeOffset.set(0, 0, 0);
      this._shakeRotOffset.set(0, 0, 0);
      return;
    }

    let totalOffsetX = 0, totalOffsetY = 0, totalOffsetZ = 0;
    let totalRotX    = 0, totalRotY    = 0, totalRotZ    = 0;

    // Process events in reverse so we can splice cleanly
    for (let i = this._shakeEvents.length - 1; i >= 0; i--) {
      const ev = this._shakeEvents[i];

      // Decay trauma (unless sustained — decayRate === 0)
      if (ev.decayRate > 0) {
        ev.trauma -= ev.decayRate * dt;
      }

      if (ev.trauma <= 0) {
        this._shakeEvents.splice(i, 1);
        continue;
      }

      // Trauma² gives a more pleasing response curve (small hits are subtle,
      // large hits are dramatic — this is the standard GDC "juicing" technique).
      const t2   = ev.trauma * ev.trauma;
      const time = this._shakeTime * ev.frequency;

      // Per-axis noise (different seeds per axis for decorrelation)
      const nx = fbmNoise(time, ev.seed + 0);
      const ny = fbmNoise(time, ev.seed + 100);
      const nz = fbmNoise(time, ev.seed + 200);

      totalOffsetX += nx * ev.amplitude.x * t2;
      totalOffsetY += ny * ev.amplitude.y * t2;
      totalOffsetZ += nz * ev.amplitude.z * t2;

      const nrx = fbmNoise(time, ev.seed + 300);
      const nry = fbmNoise(time, ev.seed + 400);
      const nrz = fbmNoise(time, ev.seed + 500);

      totalRotX += nrx * ev.rotAmplitude.x * t2;
      totalRotY += nry * ev.rotAmplitude.y * t2;
      totalRotZ += nrz * ev.rotAmplitude.z * t2;
    }

    this._shakeOffset.set(totalOffsetX, totalOffsetY, totalOffsetZ);
    this._shakeRotOffset.set(totalRotX, totalRotY, totalRotZ);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Spline Path
  // ─────────────────────────────────────────────────────────────────────────

  private _updatePath(dt: number): void {
    if (!this._pathPlaying || this._pathPoints.length < 2) return;

    const pts = this._pathPoints;
    const n   = pts.length;

    // Get speed multiplier at current segment
    const segIdx   = Math.min(Math.floor(this._pathT * (n - 1)), n - 2);
    const speedMul = pts[segIdx].speedMult;

    // Advance t
    this._pathT += this._pathDirection * this._pathSpeed * speedMul * dt;

    // Handle end-of-path
    if (this._pathT >= 1.0) {
      switch (this._pathLoop) {
        case 'once':
          this._pathT       = 1.0;
          this._pathPlaying = false;
          this._debug('path: finished (once)');
          break;
        case 'loop':
          this._pathT -= 1.0;
          break;
        case 'pingpong':
          this._pathT         = 1.0;
          this._pathDirection = -1;
          break;
        case 'hold':
          this._pathT       = 1.0;
          this._pathPlaying = false;
          break;
      }
    } else if (this._pathT <= 0.0) {
      // Only reachable in pingpong reverse
      this._pathT         = 0.0;
      this._pathDirection = 1;
    }

    this._pathT = clamp(this._pathT, 0, 1);

    // Evaluate spline position via Catmull-Rom interpolation
    const pos = evaluateCatmullRom3D(
      pts.map(p => p.position),
      this._pathT,
    );

    // Determine look-at target
    // Check current segment's lookAt, fall back to looking along spline tangent
    const currentPoint = pts[segIdx];
    let lookTarget: Vec3;

    if (currentPoint.lookAt) {
      // Lerp between current and next segment's lookAt for smooth transition
      const nextIdx    = Math.min(segIdx + 1, n - 1);
      const nextPoint  = pts[nextIdx];
      const localT     = (this._pathT * (n - 1)) - segIdx;

      if (nextPoint.lookAt) {
        lookTarget = currentPoint.lookAt.lerp(nextPoint.lookAt, localT);
      } else {
        lookTarget = currentPoint.lookAt;
      }
    } else {
      // Look along spline tangent — sample a point slightly ahead
      const aheadT = Math.min(this._pathT + 0.01, 1.0);
      lookTarget   = evaluateCatmullRom3D(pts.map(p => p.position), aheadT);
    }

    // Smooth-set position and rotation
    const alpha = 1 - Math.pow(0.05, dt * 60);
    this._position.lerpSelf(pos, alpha);
    this._rotation = Quat.slerp(
      this._rotation,
      lookAtRotation(this._position, lookTarget),
      alpha,
    );

    // Optional per-point FOV
    if (currentPoint.fov !== null) {
      const nextIdx   = Math.min(segIdx + 1, n - 1);
      const nextFov   = pts[nextIdx].fov ?? currentPoint.fov;
      const localT    = (this._pathT * (n - 1)) - segIdx;
      const easedT    = currentPoint.easing ? currentPoint.easing(localT) : localT;
      const targetFov = currentPoint.fov + (nextFov - currentPoint.fov) * easedT;
      this._fov       = clamp(targetFov, MIN_FOV, MAX_FOV);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Auto-Framing
  // ─────────────────────────────────────────────────────────────────────────

  private _updateAutoFrame(dt: number): void {
    const af     = this.autoFrame;
    const bounds = af.bounds;

    // Compute bounds centre and extent
    const centre = new Vec3(
      (bounds.min.x + bounds.max.x) * 0.5,
      (bounds.min.y + bounds.max.y) * 0.5,
      (bounds.min.z + bounds.max.z) * 0.5,
    );
    const extent = new Vec3(
      (bounds.max.x - bounds.min.x) * 0.5 * af.padding,
      (bounds.max.y - bounds.min.y) * 0.5 * af.padding,
      (bounds.max.z - bounds.min.z) * 0.5 * af.padding,
    );

    // Compute required distance to fit the bounds in view.
    // Use the largest extent axis and the vertical FOV to determine distance.
    const maxExtent  = Math.max(extent.x, extent.y, extent.z, 0.1);
    const fovRad     = degToRad(this._fov);
    const halfFovTan = Math.tan(fovRad * 0.5);

    // distance = maxExtent / tan(fov/2), with aspect correction
    let distance = maxExtent / halfFovTan;
    distance = clamp(distance, af.minDistance, af.maxDistance);

    // Desired camera position: centre + viewDirection * distance
    const desired = centre.add(af.viewDirection.normalize().mul(distance));

    // Lerp toward desired
    const alpha = 1 - Math.pow(1 - af.speed, dt * 60);
    this._position.lerpSelf(desired, alpha);
    this._rotation = Quat.slerp(
      this._rotation,
      lookAtRotation(this._position, centre),
      alpha,
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Transition
  // ─────────────────────────────────────────────────────────────────────────

  private _updateTransition(dt: number): void {
    const tr = this._transition!;
    tr.elapsed += dt;

    const rawT   = clamp(tr.elapsed / tr.duration, 0, 1);
    const easedT = tr.easing(rawT);

    // Interpolate position
    const fromPos = Vec3.fromArray(tr.from.position);
    const toPos   = Vec3.fromArray(tr.to.position);
    this._position = fromPos.lerp(toPos, easedT);

    // Interpolate rotation (slerp)
    const fromRot = new Quat(tr.from.rotation[0], tr.from.rotation[1], tr.from.rotation[2], tr.from.rotation[3]);
    const toRot   = new Quat(tr.to.rotation[0], tr.to.rotation[1], tr.to.rotation[2], tr.to.rotation[3]);
    this._rotation = Quat.slerp(fromRot, toRot, easedT);

    // Interpolate FOV
    this._fov = tr.from.fov + (tr.to.fov - tr.from.fov) * easedT;

    // Check completion
    if (rawT >= 1.0) {
      const cb = tr.onComplete;
      this._transition = null;
      this._mode = tr.to.mode;
      this._debug('transition: complete');
      if (cb) cb();
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Debug
  // ─────────────────────────────────────────────────────────────────────────

  private _debug(msg: string): void {
    if (this._debugEnabled) {
      console.debug(`[ASTRO-CINECAM] ${msg}`);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public — Cleanup
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Reset all state to defaults.  Does not destroy the instance —
   * it can be reused after reset.
   */
  reset(): void {
    this._position.set(0, 5, 20);
    this._rotation = Quat.identity();
    this._fov      = DEFAULT_FOV;
    this._mode     = 'free';

    Object.assign(this.follow,    defaultFollowConfig());
    Object.assign(this.orbit,     defaultOrbitConfig());
    Object.assign(this.dollyZoom, defaultDollyZoomConfig());
    Object.assign(this.autoFrame, defaultAutoFrameConfig());

    this._orbitYawVel    = 0;
    this._orbitPitchVel  = 0;
    this._dollyElapsed   = 0;
    this._dollyActive    = false;
    this._dollyReversing = false;
    this._dollyStartFov  = DEFAULT_FOV;

    this.clearShake();

    this._pathPoints  = [];
    this._pathT       = 0;
    this._pathPlaying = false;

    this._transition  = null;
    this._prevTarget  = Vec3.zero();
    this._targetVelocity = Vec3.zero();

    this._debug('camera reset');
  }

  /**
   * Destroy the camera instance — release references for GC.
   */
  destroy(): void {
    this.clearShake();
    this._pathPoints = [];
    this._transition = null;
    this._debug('camera destroyed');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper — Path point factory
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Convenience factory for creating PathPoint objects with sensible defaults.
 */
export function pathPoint(
  position: Vec3,
  options?: Partial<Omit<PathPoint, 'position'>>,
): PathPoint {
  return {
    position,
    lookAt:    options?.lookAt    ?? null,
    fov:       options?.fov      ?? null,
    easing:    options?.easing   ?? null,
    speedMult: options?.speedMult ?? 1.0,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera presets
// ─────────────────────────────────────────────────────────────────────────────

/** Pre-built camera configurations for common cinematic shots. */
export const CINEMATIC_PRESETS = {

  /** Wide establishing shot — high angle, slow orbit */
  establishing: (): Partial<CameraSnapshot> & { orbitConfig?: Partial<OrbitConfig> } => ({
    position: [0, 40, 60] as [number, number, number],
    fov: 55,
    orbitConfig: {
      distance:  70,
      pitch:     -0.5,
      autoSpeed: 0.05,
    },
  }),

  /** Close-up — tight framing, shallow DOF feel */
  closeUp: (): Partial<CameraSnapshot> & { followConfig?: Partial<FollowConfig> } => ({
    fov: 35,
    followConfig: {
      offset:     new Vec3(0, 1, 4),
      speed:      0.15,
      lookAtBias: 1.0,
      deadZone:   0.5,
    },
  }),

  /** Over-the-shoulder — offset follow with parallax */
  overShoulder: (): Partial<CameraSnapshot> & { followConfig?: Partial<FollowConfig> } => ({
    fov: 50,
    followConfig: {
      offset:     new Vec3(3, 2, 6),
      speed:      0.1,
      lookAtBias: 0.8,
      leadFactor: 0.3,
    },
  }),

  /** Bird's eye — top-down view */
  birdsEye: (): Partial<CameraSnapshot> => ({
    position: [0, 80, 0.1] as [number, number, number],
    fov: 40,
  }),

  /** Dutch angle — tilted for dramatic tension */
  dutchAngle: (): Partial<CameraSnapshot> => ({
    rotation: Quat.fromEuler(0, 0, 0.2).toArray() as [number, number, number, number],
    fov: 45,
  }),

  /** Vertigo — dolly zoom for maximum disorientation */
  vertigo: (): Partial<DollyZoomConfig> => ({
    startDistance: 30,
    endDistance:   6,
    duration:      3.0,
    easing:        Easing.easeInOut,
    autoReverse:   true,
  }),
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions (module-private)
// ─────────────────────────────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function degToRad(deg: number): number { return deg * (Math.PI / 180); }
function radToDeg(rad: number): number { return rad * (180 / Math.PI); }

/**
 * Compute a look-at quaternion: camera at `eye` looking toward `target`,
 * with world-up = (0, 1, 0).
 */
function lookAtRotation(eye: Vec3, target: Vec3): Quat {
  const forward = target.sub(eye);
  const len     = forward.length();
  if (len < EPSILON) return Quat.identity();

  forward.mulSelf(1 / len); // normalise in-place (forward is a new vec)

  const worldUp = new Vec3(0, 1, 0);

  // If forward is nearly parallel to world-up, use a fallback right vector
  let right: Vec3;
  if (Math.abs(forward.dot(worldUp)) > 0.999) {
    right = new Vec3(1, 0, 0);
  } else {
    right = worldUp.cross(forward).normalize();
  }

  const up = forward.cross(right).normalize();

  // Build rotation matrix columns → quaternion
  // Column-major: right = col0, up = col1, forward = col2
  const m00 = right.x,   m01 = up.x,   m02 = forward.x;
  const m10 = right.y,   m11 = up.y,   m12 = forward.y;
  const m20 = right.z,   m21 = up.z,   m22 = forward.z;

  const trace = m00 + m11 + m22;
  const q     = new Quat();

  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1.0);
    q.w = 0.25 / s;
    q.x = (m21 - m12) * s;
    q.y = (m02 - m20) * s;
    q.z = (m10 - m01) * s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2.0 * Math.sqrt(1.0 + m00 - m11 - m22);
    q.w = (m21 - m12) / s;
    q.x = 0.25 * s;
    q.y = (m01 + m10) / s;
    q.z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = 2.0 * Math.sqrt(1.0 + m11 - m00 - m22);
    q.w = (m02 - m20) / s;
    q.x = (m01 + m10) / s;
    q.y = 0.25 * s;
    q.z = (m12 + m21) / s;
  } else {
    const s = 2.0 * Math.sqrt(1.0 + m22 - m00 - m11);
    q.w = (m10 - m01) / s;
    q.x = (m02 + m20) / s;
    q.y = (m12 + m21) / s;
    q.z = 0.25 * s;
  }

  return q.normalize();
}

/**
 * Apply small Euler-angle rotational shake to a quaternion.
 */
function applyRotationalShake(base: Quat, euler: Vec3): Quat {
  if (euler.lengthSq() < EPSILON * EPSILON) return base;
  const shakeQ = Quat.fromEuler(euler.x, euler.y, euler.z);
  return Quat.multiply(base, shakeQ).normalize();
}

/**
 * Evaluate a Catmull-Rom spline through 3D points at parameter t ∈ [0, 1].
 *
 * Uses centripetal parameterisation (alpha = 0.5) to avoid cusps and
 * self-intersections that occur with uniform Catmull-Rom.
 */
function evaluateCatmullRom3D(points: Vec3[], t: number): Vec3 {
  const n = points.length;
  if (n === 0) return Vec3.zero();
  if (n === 1) return points[0].clone();

  t = clamp(t, 0, 1);
  const totalSegments = n - 1;
  const scaledT       = t * totalSegments;
  const seg           = Math.min(Math.floor(scaledT), totalSegments - 1);
  const localT        = scaledT - seg;

  // Four control points: p0, p1, p2, p3 (clamped at boundaries)
  const i0 = Math.max(seg - 1, 0);
  const i1 = seg;
  const i2 = Math.min(seg + 1, n - 1);
  const i3 = Math.min(seg + 2, n - 1);

  const p0 = points[i0];
  const p1 = points[i1];
  const p2 = points[i2];
  const p3 = points[i3];

  // Catmull-Rom basis matrix evaluation
  const t2 = localT * localT;
  const t3 = t2 * localT;

  // Standard Catmull-Rom (tension = 0.5)
  const x = 0.5 * (
    (2 * p1.x) +
    (-p0.x + p2.x) * localT +
    (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
    (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3
  );
  const y = 0.5 * (
    (2 * p1.y) +
    (-p0.y + p2.y) * localT +
    (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
    (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3
  );
  const z = 0.5 * (
    (2 * p1.z) +
    (-p0.z + p2.z) * localT +
    (2 * p0.z - 5 * p1.z + 4 * p2.z - p3.z) * t2 +
    (-p0.z + 3 * p1.z - 3 * p2.z + p3.z) * t3
  );

  return new Vec3(x, y, z);
}

// ─────────────────────────────────────────────────────────────────────────────
// Default export
// ─────────────────────────────────────────────────────────────────────────────

export default CinematicCamera;
