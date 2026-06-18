/**
 * cell-minimap.ts — M282: 右下角小地图
 *
 * createMinimap(stage, cw, ch) → Container
 *   在 canvas 右下角放置一个半透明背景矩形 + cell dots 缩略图。
 *   返回的 Container 包含一个 update(cells, tw, th) 方法，
 *   接收最新 cell 列表和世界总尺寸，重绘 dot 点。
 *
 * 设计：
 *   - 固定尺寸 160×100，右下偏移 margin 12
 *   - 暗色半透明背景 + 1px 边框
 *   - 每个 cell 画为 3px 圆点，颜色按 species 映射
 *   - cell 世界坐标按 (tw, th) → (mapW, mapH) 缩放
 */

import { Container } from '../../upstream/pixijs-engine/src/scene/container/Container';
import { Graphics }  from '../../upstream/pixijs-engine/src/scene/graphics/shared/Graphics';

// ── Species dot colours (matches pixi-cell-renderer SPECIES_COLOURS) ────────

const DOT_COLOURS: Record<string, number> = {
  'cil-eye':         0x5C6BC0,
  'cil-vector':      0x66BB6A,
  'cil-bolt':        0xFFA726,
  'cil-plus':        0xEC407A,
  'cil-arrow-right': 0x78909C,
  'cil-filter':      0xAB47BC,
  'cil-code':        0x26A69A,
  'cil-layers':      0x42A5F5,
  'cil-loop':        0xFFCA28,
  'cil-graph':       0x78909C,
};

const DEFAULT_DOT_COLOUR = 0x90A4AE;

// ── Minimap configuration ───────────────────────────────────────────────────

const MAP_W      = 160;
const MAP_H      = 100;
const MARGIN     = 12;
const PAD        = 4;
const DOT_RADIUS = 3;
const BG_COLOUR  = 0x1a1a2e;
const BG_ALPHA   = 0.75;
const BORDER_COL = 0x444466;

// ── Cell shape expected by update() ─────────────────────────────────────────

export interface MinimapCell {
  species: string;
  bbox: { x: number; y: number; w: number; h: number };
}

// ── Public API ──────────────────────────────────────────────────────────────

export interface MinimapHandle {
  container: Container;
  /** Redraw cell dots for the current world state. */
  update(cells: MinimapCell[], totalWorldW: number, totalWorldH: number): void;
}

/**
 * Create a minimap overlay anchored to the bottom-right corner of the canvas.
 *
 * @param stage  - root stage Container (minimap is added as direct child)
 * @param cw     - canvas / screen width  (px)
 * @param ch     - canvas / screen height (px)
 * @returns MinimapHandle with the container and an update function
 */
export function createMinimap(
  stage: Container,
  cw: number,
  ch: number,
): MinimapHandle {
  const root = new Container();
  root.label = 'minimap';

  // Position at bottom-right
  root.position.set(cw - MAP_W - MARGIN, ch - MAP_H - MARGIN);

  // Background rect
  const bg = new Graphics();
  bg.roundRect(0, 0, MAP_W, MAP_H, 4);
  bg.fill({ color: BG_COLOUR, alpha: BG_ALPHA });
  bg.stroke({ color: BORDER_COL, width: 1, alpha: 0.6 });
  root.addChild(bg);

  // Dots layer (redrawn on each update)
  const dots = new Graphics();
  root.addChild(dots);

  stage.addChild(root);

  // ── update ────────────────────────────────────────────────────────────────

  function update(
    cells: MinimapCell[],
    totalWorldW: number,
    totalWorldH: number,
  ): void {
    dots.clear();

    if (totalWorldW <= 0 || totalWorldH <= 0 || cells.length === 0) return;

    // Usable area inside padding
    const innerW = MAP_W - PAD * 2;
    const innerH = MAP_H - PAD * 2;

    const scaleX = innerW / totalWorldW;
    const scaleY = innerH / totalWorldH;

    for (const cell of cells) {
      const cx = PAD + (cell.bbox.x + cell.bbox.w * 0.5) * scaleX;
      const cy = PAD + (cell.bbox.y + cell.bbox.h * 0.5) * scaleY;
      const col = DOT_COLOURS[cell.species] ?? DEFAULT_DOT_COLOUR;

      dots.circle(cx, cy, DOT_RADIUS);
      dots.fill({ color: col, alpha: 0.9 });
    }
  }

  return { container: root, update };
}
