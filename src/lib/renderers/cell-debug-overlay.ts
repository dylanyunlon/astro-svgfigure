/**
 * cell-debug-overlay.ts — M827: Debug overlay for cells
 *
 * When debug mode is active (F12 toggle), renders diagnostic info
 * above each cell: cell_id, species, bbox, z-layer, epoch, delta.
 */

export interface CellDebugInfo {
  cell_id: string;
  species: string;
  x: number; y: number; w: number; h: number;
  z: number;
  epoch: number;
  delta: number;
  opacity: number;
  lod?: number;
}

export class CellDebugOverlay {
  private enabled: boolean = false;
  private fontSize: number = 10;
  private padding: number = 4;
  private bgColor: string = 'rgba(0,0,0,0.75)';
  private textColor: string = '#00FF88';

  constructor() {
    // Listen for F12 toggle
    if (typeof window !== 'undefined') {
      window.addEventListener('keydown', (e) => {
        if (e.key === 'F12' && e.shiftKey) {
          this.toggle();
          e.preventDefault();
        }
      });
    }
  }

  toggle(): void {
    this.enabled = !this.enabled;
  }

  get isEnabled(): boolean { return this.enabled; }
  set isEnabled(v: boolean) { this.enabled = v; }

  /**
   * Generate overlay text lines for a cell.
   * Returns array of formatted strings to render above the cell.
   */
  getOverlayLines(info: CellDebugInfo): string[] {
    if (!this.enabled) return [];
    return [
      `${info.cell_id} [${info.species}]`,
      `bbox: ${info.x.toFixed(0)},${info.y.toFixed(0)} ${info.w}×${info.h}`,
      `z:${info.z} ep:${info.epoch} Δ:${info.delta.toFixed(2)} α:${info.opacity.toFixed(2)}`,
      ...(info.lod !== undefined ? [`LOD:${info.lod}`] : []),
    ];
  }

  /**
   * Generate draw params for the overlay background + text.
   * Caller renders with PixiJS Text or BitmapText.
   */
  getDrawParams(info: CellDebugInfo): {
    x: number; y: number; lines: string[];
    fontSize: number; bgColor: string; textColor: string; padding: number;
  } | null {
    if (!this.enabled) return null;
    const lines = this.getOverlayLines(info);
    return {
      x: info.x,
      y: info.y - (lines.length * (this.fontSize + 2) + this.padding * 2) - 4,
      lines, fontSize: this.fontSize,
      bgColor: this.bgColor, textColor: this.textColor,
      padding: this.padding,
    };
  }
}
