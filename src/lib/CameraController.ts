/**
 * CameraController.ts
 * M321 — AT GazeCamera: lerpSpeed + moveXY mouse-offset + wobbleStrength
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
 */

import type { Application } from 'pixi.js';

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

    // ── Optional subtle rotation from deltaRotate ───────────────────────
    if (this.deltaRotate !== 0) {
      const rotAlpha = 1 - Math.pow(1 - this.lerpSpeed, dt * 60);
      const targetRot = g.mouseNX * this.deltaRotate * (Math.PI / 180);
      this.app.stage.rotation +=
        (targetRot - this.app.stage.rotation) * rotAlpha;
    }

    this._rafId = requestAnimationFrame(this._tick);
  };

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
    canvas.removeEventListener('mousemove', this._onMouseMove);
    canvas.removeEventListener('mouseleave', this._onMouseLeave);
  }
}

export default CameraController;
