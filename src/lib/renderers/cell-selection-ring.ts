/**
 * cell-selection-ring.ts — M827: Selection ring with species-colored pulse
 *
 * Draws a glowing animated ring around the selected/hovered cell.
 * Ring color follows the cell's species primary_color.
 * Pulse animation: radius + alpha oscillate on a sine wave.
 */

export interface SelectionRingConfig {
  baseRadius: number;
  pulseAmplitude: number;   // how much the ring expands (px)
  pulseSpeed: number;       // cycles per second
  strokeWidth: number;
  glowRadius: number;
  color: string;            // species primary_color hex
}

const DEFAULT_CONFIG: SelectionRingConfig = {
  baseRadius: 0,       // computed from cell bbox
  pulseAmplitude: 6,
  pulseSpeed: 1.5,
  strokeWidth: 2.5,
  glowRadius: 12,
  color: '#3F51B5',
};

export class CellSelectionRing {
  private config: SelectionRingConfig;
  private phase: number = 0;
  private active: boolean = false;
  private targetCellId: string | null = null;
  private locked: boolean = false;  // click locks, hover doesn't

  constructor(config: Partial<SelectionRingConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /** Show ring on hover — won't override a locked selection */
  hover(cellId: string, bbox: { x: number; y: number; w: number; h: number },
        speciesColor?: string): void {
    if (this.locked && this.targetCellId !== cellId) return;
    this.targetCellId = cellId;
    this.active = true;
    this.config.baseRadius = Math.max(bbox.w, bbox.h) / 2 + 8;
    if (speciesColor) this.config.color = speciesColor;
  }

  /** Lock ring on click */
  select(cellId: string, bbox: { x: number; y: number; w: number; h: number },
         speciesColor?: string): void {
    this.targetCellId = cellId;
    this.active = true;
    this.locked = true;
    this.config.baseRadius = Math.max(bbox.w, bbox.h) / 2 + 8;
    if (speciesColor) this.config.color = speciesColor;
  }

  /** Dismiss ring */
  deselect(): void {
    this.active = false;
    this.locked = false;
    this.targetCellId = null;
  }

  /** Called each frame — returns draw params or null */
  update(dt: number): {
    cx: number; cy: number; radius: number;
    alpha: number; color: string; strokeWidth: number; glowRadius: number;
  } | null {
    if (!this.active) return null;

    this.phase += dt * this.config.pulseSpeed * Math.PI * 2;
    const pulse = Math.sin(this.phase);

    const radius = this.config.baseRadius + pulse * this.config.pulseAmplitude;
    const alpha = 0.6 + 0.3 * (0.5 + 0.5 * pulse);  // 0.45 — 0.9

    return {
      cx: 0, cy: 0,  // caller positions relative to cell center
      radius,
      alpha,
      color: this.config.color,
      strokeWidth: this.config.strokeWidth,
      glowRadius: this.config.glowRadius + pulse * 4,
    };
  }

  get isActive(): boolean { return this.active; }
  get isLocked(): boolean { return this.locked; }
  get cellId(): string | null { return this.targetCellId; }
}
