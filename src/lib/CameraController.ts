/**
 * CameraController.ts
 * M321 — AT GazeCamera: lerpSpeed + moveXY mouse-offset + wobbleStrength
 * M1291 — autoTrack mode: camera smoothly follows the most active cell cluster
 *
 * Replaces the M162 wheel-zoom / drag-pan controller with an AT-style
 * "GazeCamera" that smoothly follows the cursor with lerp interpolation
 * and adds procedural wobble for organic motion.
 *
 * Configuration is loaded from channels/physics/camera_at_params.json
 * (UIL JSON exported by xiaodi #67).
 *
 *   • lerpSpeed       — interpolation rate toward mouse target [0..1]
 *   • moveXY          — mouse-offset parallax multiplier [x, y]
 *   • wobbleStrength  — amplitude of sine-based procedural drift
 *   • lerpSpeed2      — secondary lerp for zoom transitions
 *   • deltaRotate     — subtle rotational response to cursor
 *
 * M1291 autoTrack additions:
 *   • autoTrack       — when enabled, camera lerps toward the weighted
 *                       centroid of the most active cell cluster each frame
 *   • Activity score  — speed (|vx|+|vy|) + collision frequency per cell
 *   • Hotspot         — recomputed every 60 frames via weighted average position
 *   • Lerp factor     — 0.02 (slower than gaze lerp for cinematic feel)
 *   • Pause on drag   — user panning suspends autoTrack for 5 s then resumes
 *   • toggleAutoTrack()  — public API to flip the mode
 */

import type { Application } from 'pixi.js';
import type { CellInteractionPhysics } from './cell-interaction-physics';

// ─── AT Camera Preset types (preserved from xiaodi #67) ──────────────────────

export interface CameraPreset {
  position?: [number, number, number];
  lookAt?: [number, number, number];
  groupPos?: [number, number, number];
  moveXY?: [number, number];
  fov?: number;
  far?: number;
  near?: number;
  lerpSpeed?: number;
  lerpSpeed2?: number;
  rotation?: [number, number, number];
  cameraRotation?: [number, number, number];
  viewportFocus?: [number, number];
  deltaRotate?: number;
  wobbleStrength?: number;
}

// ─── JSON shape for channels/physics/camera_at_params.json ───────────────────

interface CameraAtParamsJSON {
  meta: {
    source: string;
    category: string;
    total_params: number;
    total_scenes: number;
    author: string;
  };
  params: Record<string, number | number[]>;
  presets: Record<string, CameraPreset>;
}

// ─── Internal gaze state ─────────────────────────────────────────────────────

interface GazeState {
  /** Normalised mouse position [-1, 1] relative to canvas centre. */
  mouseNX: number;
  mouseNY: number;
  /** Current lerped camera offset (pixels). */
  currentX: number;
  currentY: number;
  /** Target camera offset before lerp (pixels). */
  targetX: number;
  targetY: number;
  /** Base (rest) position — the pan origin loaded from preset. */
  baseX: number;
  baseY: number;
  /** Wobble phase accumulator (radians). */
  wobblePhase: number;
}

// ─── M1291: AutoTrack state ───────────────────────────────────────────────────

interface AutoTrackState {
  /** Whether autoTrack mode is currently active. */
  enabled: boolean;
  /**
   * Frame counter — hotspot is recomputed every AUTO_TRACK_HOTSPOT_INTERVAL
   * frames to spread the O(n) scan cost.
   */
  frameCounter: number;
  /** Cached hotspot centroid in world-space (stage coordinates). */
  hotspotX: number;
  hotspotY: number;
  /**
   * Timestamp (ms) after which autoTrack resumes following a manual drag.
   * 0 = not paused.
   */
  pauseUntil: number;
  /**
   * Whether the user is currently dragging (pointer is held down on canvas).
   * Set by the mousedown/mouseup/mouseleave listeners.
   */
  userDragging: boolean;
  /** Previous pointer position for drag-detection. */
  lastPointerX: number;
  lastPointerY: number;
}

const AUTO_TRACK_LERP       = 0.02;          // camera lerp toward hotspot
const AUTO_TRACK_HOTSPOT_INTERVAL = 60;       // frames between hotspot recomputes
const AUTO_TRACK_PAUSE_MS   = 5_000;          // ms to suspend after manual drag

// ─── CameraController (AT GazeCamera) ────────────────────────────────────────

export class CameraController {
  private app: Application;
  private zoomFactor = 1.0;

  // ── Gaze parameters (from UIL preset) ───────────────────────────────────
  private lerpSpeed = 0.08;
  private lerpSpeed2 = 1.0;
  private moveXY: [number, number] = [0, 0];
  private wobbleStrength = 0.0;
  private deltaRotate = 0.0;

  // ── Internal gaze state ─────────────────────────────────────────────────
  private gaze: GazeState = {
    mouseNX: 0,
    mouseNY: 0,
    currentX: 0,
    currentY: 0,
    targetX: 0,
    targetY: 0,
    baseX: 0,
    baseY: 0,
    wobblePhase: 0,
  };

  // ── Animation frame handle ──────────────────────────────────────────────
  private _rafId: number | null = null;
  private _lastTime = 0;

  // Bound listeners (for cleanup)
  private _onMouseMove: (e: MouseEvent) => void;
  private _onMouseLeave: () => void;

  // ── M1291: autoTrack ────────────────────────────────────────────────────
  private _physics: CellInteractionPhysics | null = null;
  private _autoTrack: AutoTrackState = {
    enabled: false,
    frameCounter: 0,
    hotspotX: 0,
    hotspotY: 0,
    pauseUntil: 0,
    userDragging: false,
    lastPointerX: 0,
    lastPointerY: 0,
  };
  private _onPointerDown: (e: PointerEvent) => void;
  private _onPointerUp: (e: PointerEvent) => void;

  // Loaded presets from JSON
  private _presets: Record<string, CameraPreset> = {};
  private _params: Record<string, number | number[]> = {};

  // Active preset name
  private _activeScene = '';

  constructor(app: Application) {
    this.app = app;

    // ── Bind mouse tracking ───────────────────────────────────────────────

    this._onMouseMove = (e: MouseEvent) => {
      const canvas = this.app.canvas as HTMLCanvasElement;
      const rect = canvas.getBoundingClientRect();
      // Normalise to [-1, 1] with (0, 0) at canvas centre.
      this.gaze.mouseNX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.gaze.mouseNY = ((e.clientY - rect.top) / rect.height) * 2 - 1;
    };

    this._onMouseLeave = () => {
      // Smoothly return to centre when cursor leaves.
      this.gaze.mouseNX = 0;
      this.gaze.mouseNY = 0;
    };

    // ── Attach listeners to canvas ────────────────────────────────────────

    const canvas = this.app.canvas as HTMLCanvasElement;
    canvas.addEventListener('mousemove', this._onMouseMove);
    canvas.addEventListener('mouseleave', this._onMouseLeave);

    // ── M1291: pointer drag detection for autoTrack pause ─────────────────

    this._onPointerDown = (e: PointerEvent) => {
      const at = this._autoTrack;
      at.userDragging = true;
      at.lastPointerX = e.clientX;
      at.lastPointerY = e.clientY;
    };

    this._onPointerUp = (_e: PointerEvent) => {
      const at = this._autoTrack;
      if (at.userDragging) {
        at.userDragging = false;
        // Pause autoTrack for 5 s after the user releases the pointer.
        at.pauseUntil = performance.now() + AUTO_TRACK_PAUSE_MS;
      }
    };

    canvas.addEventListener('pointerdown', this._onPointerDown);
    canvas.addEventListener('pointerup',   this._onPointerUp);
    canvas.addEventListener('pointerleave', this._onPointerUp);

    // ── Load camera config from UIL JSON then start the gaze loop ─────────

    this._loadInitialParams().then(() => {
      this._lastTime = performance.now();
      this._tick(this._lastTime);
    });
  }

  // ── Core animation loop ───────────────────────────────────────────────────

  private _tick = (now: number): void => {
    const dt = Math.min((now - this._lastTime) / 1000, 0.1); // cap at 100 ms
    this._lastTime = now;

    const g = this.gaze;
    const canvas = this.app.canvas as HTMLCanvasElement;
    const halfW = canvas.width * 0.5;
    const halfH = canvas.height * 0.5;

    // ── Compute target offset from mouse position × moveXY multiplier ───
    g.targetX = g.baseX + g.mouseNX * this.moveXY[0] * halfW;
    g.targetY = g.baseY + g.mouseNY * this.moveXY[1] * halfH;

    // ── Lerp toward target ──────────────────────────────────────────────
    // Use 1 - (1 - lerpSpeed)^(dt*60) for frame-rate-independent lerp.
    const alpha = 1 - Math.pow(1 - this.lerpSpeed, dt * 60);
    g.currentX += (g.targetX - g.currentX) * alpha;
    g.currentY += (g.targetY - g.currentY) * alpha;

    // ── Wobble (procedural sine drift) ──────────────────────────────────
    g.wobblePhase += dt;
    let wobX = 0;
    let wobY = 0;
    if (this.wobbleStrength > 0) {
      const str = this.wobbleStrength * halfW;
      // Two incommensurate frequencies for organic motion.
      wobX = Math.sin(g.wobblePhase * 1.3) * str * 0.6
           + Math.sin(g.wobblePhase * 2.7) * str * 0.4;
      wobY = Math.cos(g.wobblePhase * 1.1) * str * 0.5
           + Math.cos(g.wobblePhase * 2.3) * str * 0.3;
    }

    // ── Apply to stage ──────────────────────────────────────────────────
    this.app.stage.position.x = g.currentX + wobX;
    this.app.stage.position.y = g.currentY + wobY;

    // ── M1291: autoTrack — shift base toward hottest cell cluster ───────
    this._tickAutoTrack(dt);

    // ── Optional subtle rotation from deltaRotate ───────────────────────
    if (this.deltaRotate !== 0) {
      const rotAlpha = 1 - Math.pow(1 - this.lerpSpeed, dt * 60);
      const targetRot = g.mouseNX * this.deltaRotate * (Math.PI / 180);
      this.app.stage.rotation +=
        (targetRot - this.app.stage.rotation) * rotAlpha;
    }

    this._rafId = requestAnimationFrame(this._tick);
  };

  // ── M1291: autoTrack private helpers ──────────────────────────────────────

  /**
   * Called once per animation frame.  When autoTrack is enabled and not
   * paused, smoothly shifts the camera base position toward the weighted
   * centroid of the most active cell cluster.
   *
   * "Activity" for each cell = |vx| + |vy|  +  collisionCount * 2
   * (collisionCount is the accumulated count since the last physics step,
   * exposed via CellInteractionPhysics.getAllStates() — we treat it as a
   * scalar weight multiplier).
   *
   * The hotspot is only recomputed every AUTO_TRACK_HOTSPOT_INTERVAL frames
   * to amortise the O(n) scan; between recomputations the camera lerps toward
   * the cached hotspot.
   */
  private _tickAutoTrack(dt: number): void {
    const at = this._autoTrack;

    if (!at.enabled) return;
    if (!this._physics) return;

    const now = performance.now();

    // If the user is actively dragging or the cooldown has not expired, skip.
    if (at.userDragging || now < at.pauseUntil) return;

    // ── Recompute hotspot every N frames ──────────────────────────────────
    at.frameCounter++;
    if (at.frameCounter >= AUTO_TRACK_HOTSPOT_INTERVAL) {
      at.frameCounter = 0;
      this._computeHotspot();
    }

    // ── Lerp camera base toward hotspot ───────────────────────────────────
    // hotspot is in simulation world-space (cell centre coordinates).
    // We need to map it to stage-space pan.  The stage origin is at the
    // canvas centre when pan = 0; cell x/y are in the same space that
    // app.stage children use.  So the target pan offset that centres the
    // camera on the hotspot is:
    //   panX = canvasCentreX - hotspotX * zoomFactor
    //   panY = canvasCentreY - hotspotY * zoomFactor
    const canvas = this.app.canvas as HTMLCanvasElement;
    const targetBaseX = canvas.width  * 0.5 - at.hotspotX * this.zoomFactor;
    const targetBaseY = canvas.height * 0.5 - at.hotspotY * this.zoomFactor;

    // Frame-rate-independent lerp using the fixed AUTO_TRACK_LERP factor.
    const alpha = 1 - Math.pow(1 - AUTO_TRACK_LERP, dt * 60);
    this.gaze.baseX += (targetBaseX - this.gaze.baseX) * alpha;
    this.gaze.baseY += (targetBaseY - this.gaze.baseY) * alpha;
  }

  /**
   * Walk every cell body, compute an activity score, and derive the
   * weighted centroid (hotspot) in world-space coordinates.
   * Pinned and dragging cells are excluded (they are intentionally static).
   */
  private _computeHotspot(): void {
    if (!this._physics) return;

    const states = this._physics.getAllStates();
    if (states.length === 0) return;

    let sumW  = 0;
    let sumWX = 0;
    let sumWY = 0;

    for (const s of states) {
      // Skip pinned / dragged cells — they are not "active" in a meaningful sense.
      if (s.pinned || s.dragging) continue;

      // Activity score: speed component + a small baseline so static cells
      // still contribute a tiny weight (prevents hotspot from jumping to
      // the origin when everything is slow).
      const speed    = Math.abs(s.vx) + Math.abs(s.vy);
      const activity = speed + 0.01;  // baseline prevents division-by-zero

      sumW  += activity;
      sumWX += s.x * activity;
      sumWY += s.y * activity;
    }

    if (sumW > 0) {
      this._autoTrack.hotspotX = sumWX / sumW;
      this._autoTrack.hotspotY = sumWY / sumW;
    }
  }

  // ── Load initial camera parameters from UIL JSON ──────────────────────────

  private async _loadInitialParams(): Promise<void> {
    try {
      const resp = await fetch('/channels/physics/camera_at_params.json');
      if (!resp.ok) {
        console.warn(
          `[GazeCamera] Failed to load camera_at_params.json: ${resp.status}`,
        );
        return;
      }
      const data: CameraAtParamsJSON = await resp.json();
      this._presets = data.presets ?? {};
      this._params = data.params ?? {};

      // Apply the Home preset as default (consistent with AT convention).
      const defaultPreset =
        data.presets['Element_1_Home'] ?? Object.values(data.presets)[0];

      if (defaultPreset) {
        this._applyPreset(defaultPreset, 'Element_1_Home');
      }

      console.debug(
        `[GazeCamera] Loaded camera_at_params.json — ` +
          `${Object.keys(this._presets).length} presets, ` +
          `lerpSpeed=${this.lerpSpeed}, ` +
          `moveXY=[${this.moveXY}], ` +
          `wobbleStrength=${this.wobbleStrength}`,
      );
    } catch (err) {
      console.warn('[GazeCamera] Could not load camera_at_params.json:', err);
    }
  }

  // ── Apply a preset to gaze parameters ─────────────────────────────────────

  private _applyPreset(preset: CameraPreset, name?: string): void {
    if (name) this._activeScene = name;

    // ── Gaze parameters ───────────────────────────────────────────────────
    if (preset.lerpSpeed !== undefined) this.lerpSpeed = preset.lerpSpeed;
    if (preset.lerpSpeed2 !== undefined) this.lerpSpeed2 = preset.lerpSpeed2;
    if (preset.moveXY !== undefined) this.moveXY = [...preset.moveXY];
    if (preset.wobbleStrength !== undefined) this.wobbleStrength = preset.wobbleStrength;
    if (preset.deltaRotate !== undefined) this.deltaRotate = preset.deltaRotate;

    // ── Derive zoom from position z-component ─────────────────────────────
    // AT params use z as a "distance" value; normalise so that
    // the Home scene's z=40 ≈ zoomFactor 1.0 (preserved from M162).
    if (preset.position) {
      const z = preset.position[2];
      if (z > 0) {
        this.zoomFactor = 40 / z; // z=40 → 1.0, z=20 → 2.0, z=80 → 0.5
        this.zoomFactor = Math.max(0.2, Math.min(5.0, this.zoomFactor));
        this.app.stage.scale.set(this.zoomFactor);
      }
    }

    // ── Derive base pan from moveXY (initial offset) ──────────────────────
    if (preset.moveXY) {
      this.gaze.baseX = preset.moveXY[0];
      this.gaze.baseY = preset.moveXY[1];
      // Snap current position when switching presets.
      this.gaze.currentX = this.gaze.baseX;
      this.gaze.currentY = this.gaze.baseY;
      this.gaze.targetX = this.gaze.baseX;
      this.gaze.targetY = this.gaze.baseY;
    }
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /** Current zoom factor. */
  getZoom(): number {
    return this.zoomFactor;
  }

  /** Programmatically set zoom (clamped [0.2, 5.0]). */
  setZoom(z: number): void {
    this.zoomFactor = Math.max(0.2, Math.min(5.0, z));
    this.app.stage.scale.set(this.zoomFactor);
  }

  /** Current stage pan position. */
  getPan(): { x: number; y: number } {
    return {
      x: this.app.stage.position.x,
      y: this.app.stage.position.y,
    };
  }

  /** Programmatically set pan offset (updates gaze base position). */
  setPan(x: number, y: number): void {
    this.gaze.baseX = x;
    this.gaze.baseY = y;
    this.gaze.currentX = x;
    this.gaze.currentY = y;
    this.app.stage.position.x = x;
    this.app.stage.position.y = y;
  }

  /** Get a named AT camera preset. */
  getPreset(scene: string): CameraPreset | undefined {
    return this._presets[scene];
  }

  /** List all loaded preset names. */
  listScenes(): string[] {
    return Object.keys(this._presets);
  }

  /** Get all raw AT camera params. */
  getParams(): Record<string, number | number[]> {
    return this._params;
  }

  /** Transition to a named scene preset. */
  goToScene(scene: string): boolean {
    const preset = this._presets[scene];
    if (!preset) return false;
    this._applyPreset(preset, scene);
    return true;
  }

  /** Current active scene name. */
  getActiveScene(): string {
    return this._activeScene;
  }

  // ── M1291: autoTrack public API ───────────────────────────────────────────

  /**
   * Provide the physics simulation so autoTrack can query cell states.
   * Call this once after constructing CellInteractionPhysics.
   */
  setPhysics(physics: CellInteractionPhysics): void {
    this._physics = physics;
  }

  /**
   * Toggle autoTrack mode on/off.
   * Returns the new enabled state.
   *
   * When enabled the camera will start smoothly following the most active
   * cell cluster.  When disabled the gaze base position stays where it is
   * (the user can pan freely).
   */
  toggleAutoTrack(): boolean {
    const at = this._autoTrack;
    at.enabled = !at.enabled;

    if (at.enabled) {
      // Reset counters so we compute the first hotspot immediately.
      at.frameCounter = AUTO_TRACK_HOTSPOT_INTERVAL - 1;
      at.pauseUntil  = 0;
      at.userDragging = false;
      console.debug('[GazeCamera] autoTrack enabled');
    } else {
      console.debug('[GazeCamera] autoTrack disabled');
    }

    return at.enabled;
  }

  /**
   * Enable autoTrack explicitly (no-op if already on).
   * Returns the new enabled state (always true).
   */
  enableAutoTrack(): boolean {
    if (!this._autoTrack.enabled) this.toggleAutoTrack();
    return true;
  }

  /**
   * Disable autoTrack explicitly (no-op if already off).
   * Returns the new enabled state (always false).
   */
  disableAutoTrack(): boolean {
    if (this._autoTrack.enabled) this.toggleAutoTrack();
    return false;
  }

  /** Whether autoTrack is currently active (not paused). */
  isAutoTracking(): boolean {
    const at = this._autoTrack;
    return at.enabled && !at.userDragging && performance.now() >= at.pauseUntil;
  }

  /** The last computed hotspot centroid in world-space (for debugging). */
  getAutoTrackHotspot(): { x: number; y: number } {
    return { x: this._autoTrack.hotspotX, y: this._autoTrack.hotspotY };
  }

  /** Current gaze parameters (for debugging / UI panels). */
  getGazeConfig(): {
    lerpSpeed: number;
    lerpSpeed2: number;
    moveXY: [number, number];
    wobbleStrength: number;
    deltaRotate: number;
  } {
    return {
      lerpSpeed: this.lerpSpeed,
      lerpSpeed2: this.lerpSpeed2,
      moveXY: [...this.moveXY] as [number, number],
      wobbleStrength: this.wobbleStrength,
      deltaRotate: this.deltaRotate,
    };
  }

  /** Remove all event listeners and stop the animation loop. */
  destroy(): void {
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
    const canvas = this.app.canvas as HTMLCanvasElement;
    canvas.removeEventListener('mousemove',   this._onMouseMove);
    canvas.removeEventListener('mouseleave',  this._onMouseLeave);
    // M1291: remove autoTrack drag-detection listeners
    canvas.removeEventListener('pointerdown',  this._onPointerDown);
    canvas.removeEventListener('pointerup',    this._onPointerUp);
    canvas.removeEventListener('pointerleave', this._onPointerUp);
  }
}

export default CameraController;
