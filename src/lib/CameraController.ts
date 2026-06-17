/**
 * CameraController.ts
 * M162 — PixiJS Camera Controller: wheel zoom + drag pan + camera_at_params.json
 *
 * Wraps a PixiJS Application to provide:
 *   • wheel zoom    — scroll to scale stage, clamped [0.2, 5.0]
 *   • drag pan      — pointerdown/move/up to translate stage
 *   • initial state — zoom & pan loaded from channels/physics/camera_at_params.json
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

// ─── CameraController ────────────────────────────────────────────────────────

export class CameraController {
  private app: Application;
  private zoomFactor = 1.0;

  // Drag state
  private dragging = false;
  private startX = 0;
  private startY = 0;
  private stageStartX = 0;
  private stageStartY = 0;

  // Bound listeners (for cleanup)
  private _onWheel: (e: WheelEvent) => void;
  private _onPointerDown: (e: PointerEvent) => void;
  private _onPointerMove: (e: PointerEvent) => void;
  private _onPointerUp: (e: PointerEvent) => void;

  // Loaded presets from JSON
  private _presets: Record<string, CameraPreset> = {};
  private _params: Record<string, number | number[]> = {};

  constructor(app: Application) {
    this.app = app;

    // ── Bind event handlers ────────────────────────────────────────────────

    this._onWheel = (e: WheelEvent) => {
      e.preventDefault();
      this.zoomFactor *= 1 - e.deltaY * 0.001;
      this.zoomFactor = Math.max(0.2, Math.min(5.0, this.zoomFactor));
      this.app.stage.scale.set(this.zoomFactor);
    };

    this._onPointerDown = (e: PointerEvent) => {
      this.dragging = true;
      this.startX = e.clientX;
      this.startY = e.clientY;
      this.stageStartX = this.app.stage.position.x;
      this.stageStartY = this.app.stage.position.y;
    };

    this._onPointerMove = (e: PointerEvent) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.startX;
      const dy = e.clientY - this.startY;
      this.app.stage.position.x = this.stageStartX + dx;
      this.app.stage.position.y = this.stageStartY + dy;
    };

    this._onPointerUp = (_e: PointerEvent) => {
      this.dragging = false;
    };

    // ── Attach listeners to canvas ─────────────────────────────────────────

    const canvas = this.app.canvas as HTMLCanvasElement;

    canvas.addEventListener('wheel', this._onWheel, { passive: false });
    canvas.addEventListener('pointerdown', this._onPointerDown);
    canvas.addEventListener('pointermove', this._onPointerMove);
    canvas.addEventListener('pointerup', this._onPointerUp);
    canvas.addEventListener('pointerleave', this._onPointerUp);

    // ── Load initial zoom/pan from camera_at_params.json ───────────────────

    this._loadInitialParams();
  }

  // ── Load initial camera parameters from JSON ─────────────────────────────

  private async _loadInitialParams(): Promise<void> {
    try {
      const resp = await fetch('/channels/physics/camera_at_params.json');
      if (!resp.ok) {
        console.warn(
          `[CameraController] Failed to load camera_at_params.json: ${resp.status}`,
        );
        return;
      }
      const data: CameraAtParamsJSON = await resp.json();
      this._presets = data.presets ?? {};
      this._params = data.params ?? {};

      // Use the first preset with both position and moveXY as initial state.
      // position[2] (z-depth) maps to zoom; moveXY maps to pan offset.
      const defaultPreset =
        data.presets['Element_1_Home'] ?? Object.values(data.presets)[0];

      if (defaultPreset) {
        // Derive initial zoom from position z-component.
        // The AT params use z as a "distance" value; normalise so that
        // the Home scene's z=40 ≈ zoomFactor 1.0 (sensible default).
        if (defaultPreset.position) {
          const z = defaultPreset.position[2];
          if (z > 0) {
            this.zoomFactor = 40 / z; // z=40 → 1.0, z=20 → 2.0, z=80 → 0.5
            this.zoomFactor = Math.max(0.2, Math.min(5.0, this.zoomFactor));
            this.app.stage.scale.set(this.zoomFactor);
          }
        }

        // Derive initial pan from moveXY.
        if (defaultPreset.moveXY) {
          this.app.stage.position.x = defaultPreset.moveXY[0];
          this.app.stage.position.y = defaultPreset.moveXY[1];
        }
      }

      console.debug(
        `[CameraController] Loaded camera_at_params.json — ` +
          `${Object.keys(this._presets).length} presets, ` +
          `zoom=${this.zoomFactor.toFixed(2)}, ` +
          `pan=(${this.app.stage.position.x}, ${this.app.stage.position.y})`,
      );
    } catch (err) {
      console.warn('[CameraController] Could not load camera_at_params.json:', err);
    }
  }

  // ── Public API ───────────────────────────────────────────────────────────

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

  /** Programmatically set pan offset. */
  setPan(x: number, y: number): void {
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

  /** Remove all event listeners. Call when the controller is no longer needed. */
  destroy(): void {
    const canvas = this.app.canvas as HTMLCanvasElement;
    canvas.removeEventListener('wheel', this._onWheel);
    canvas.removeEventListener('pointerdown', this._onPointerDown);
    canvas.removeEventListener('pointermove', this._onPointerMove);
    canvas.removeEventListener('pointerup', this._onPointerUp);
    canvas.removeEventListener('pointerleave', this._onPointerUp);
  }
}

export default CameraController;
