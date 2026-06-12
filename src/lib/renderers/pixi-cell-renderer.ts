/**
 * pixi-cell-renderer.ts — Method 1: PixiJS Graphics cell renderer
 *
 * Cell 只输出参数（species、bbox、z），PixiJS 负责所有视觉：
 * - 圆角矩形 (Graphics.roundRect)
 * - Species 内部图案 (procedural Graphics draw calls)
 * - 每个 cell 背后的 bloom glow（BlurFilter + additive blend）
 * - 贝塞尔曲线连线 (Graphics.bezierCurveTo)
 * - Anti-aliasing（GPU 子像素平滑）
 *
 * 渲染质量完全由 PixiJS 管线决定，跟 LLM 代码生成能力无关。
 *
 * Upstream reference:
 *   upstream/pixijs-engine/src/scene/graphics/shared/Graphics.ts
 *   upstream/pixijs-engine/src/filters/defaults/blur/
 *   skills/pixijs/pixijs-filters/SKILL.md
 *   skills/pixijs/pixijs-graphics/SKILL.md
 */

import {
  Application,
  Container,
  Graphics,
  Text,
  TextStyle,
  BlurFilter,
  ColorMatrixFilter,
  FederatedPointerEvent,
} from 'pixi.js';

// ── Cell descriptor — this is ALL the LLM needs to produce ──────────────────

export interface CellDescriptor {
  cell_id: string;
  label: string;
  species: string;
  bbox: { x: number; y: number; w: number; h: number };
  z: number;
  topology: {
    incoming_edges: string[];
    outgoing_edges: string[];
  };
}

export interface EdgeDescriptor {
  id: string;
  source: string;
  target: string;
  type: 'normal' | 'skip_connection';
}

// ── Species colour palette ──────────────────────────────────────────────────

const SPECIES_COLOURS: Record<string, { fill: number; stroke: number; glow: number }> = {
  'cil-eye':         { fill: 0x5C6BC0, stroke: 0x3949AB, glow: 0x7986CB },
  'cil-vector':      { fill: 0x66BB6A, stroke: 0x388E3C, glow: 0x81C784 },
  'cil-bolt':        { fill: 0xFFA726, stroke: 0xF57C00, glow: 0xFFCC80 },
  'cil-plus':        { fill: 0xEC407A, stroke: 0xC62828, glow: 0xF48FB1 },
  'cil-arrow-right': { fill: 0x78909C, stroke: 0x455A64, glow: 0xB0BEC5 },
  'cil-filter':      { fill: 0xAB47BC, stroke: 0x7B1FA2, glow: 0xCE93D8 },
  'cil-code':        { fill: 0x26A69A, stroke: 0x00796B, glow: 0x80CBC4 },
  'cil-layers':      { fill: 0x42A5F5, stroke: 0x1565C0, glow: 0x90CAF9 },
  'cil-loop':        { fill: 0xFFCA28, stroke: 0xF9A825, glow: 0xFFE082 },
  'cil-graph':       { fill: 0x78909C, stroke: 0x37474F, glow: 0xB0BEC5 },
};

function getColours(species: string) {
  return SPECIES_COLOURS[species] ?? { fill: 0x90A4AE, stroke: 0x607D8B, glow: 0xB0BEC5 };
}

// ── Species pattern drawers ─────────────────────────────────────────────────
// Each draws a procedural pattern INSIDE the cell's rounded rect.
// GPU does all anti-aliasing — these are just Graphics draw calls.

type PatternDrawer = (g: Graphics, w: number, h: number, col: number) => void;

const SPECIES_PATTERNS: Record<string, PatternDrawer> = {
  'cil-eye': (g, w, h, col) => {
    // Radial heatmap: concentric circles
    const cx = w / 2, cy = h / 2, r = Math.min(w, h) * 0.35;
    for (let i = 3; i >= 1; i--) {
      g.circle(cx, cy, r * (i / 3));
      g.fill({ color: col, alpha: 0.08 * i });
    }
    // Pupil
    g.circle(cx, cy, r * 0.15);
    g.fill({ color: col, alpha: 0.5 });
  },

  'cil-vector': (g, w, h, col) => {
    // Gradient arrow
    const pad = 8;
    g.moveTo(pad, h / 2);
    g.lineTo(w - pad * 3, h / 2);
    g.moveTo(w - pad * 3, h / 2);
    g.lineTo(w - pad * 4, h / 2 - 6);
    g.moveTo(w - pad * 3, h / 2);
    g.lineTo(w - pad * 4, h / 2 + 6);
    g.stroke({ color: col, width: 1.5, alpha: 0.4 });
  },

  'cil-bolt': (g, w, h, col) => {
    // Zigzag activation
    const n = 5, dy = h / (n + 1), amp = w * 0.15;
    g.moveTo(w / 2, 6);
    for (let i = 1; i <= n; i++) {
      const x = w / 2 + (i % 2 === 1 ? amp : -amp);
      g.lineTo(x, dy * i + 6);
    }
    g.lineTo(w / 2, h - 6);
    g.stroke({ color: col, width: 1.5, alpha: 0.35 });
  },

  'cil-plus': (g, w, h, col) => {
    // Cross grid
    const cx = w / 2, cy = h / 2, arm = Math.min(w, h) * 0.3;
    g.moveTo(cx - arm, cy); g.lineTo(cx + arm, cy);
    g.moveTo(cx, cy - arm); g.lineTo(cx, cy + arm);
    g.stroke({ color: col, width: 2, alpha: 0.3 });
  },

  'cil-arrow-right': (g, w, h, col) => {
    // Chevron
    const cx = w / 2, cy = h / 2, sz = Math.min(w, h) * 0.25;
    g.moveTo(cx - sz, cy - sz);
    g.lineTo(cx + sz * 0.5, cy);
    g.lineTo(cx - sz, cy + sz);
    g.stroke({ color: col, width: 2, alpha: 0.4 });
  },

  'cil-filter': (g, w, h, col) => {
    // 3x3 grid
    const pad = 10, gw = (w - pad * 2) / 3, gh = (h - pad * 2) / 3;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        g.rect(pad + c * gw + 1, pad + r * gh + 1, gw - 2, gh - 2);
        g.stroke({ color: col, width: 0.8, alpha: 0.2 });
      }
    }
  },

  'cil-code': (g, w, h, col) => {
    // Curly braces { }
    const bx = 12, by = 8;
    g.moveTo(bx, by); g.lineTo(bx - 4, h / 2); g.lineTo(bx, h - by);
    g.moveTo(w - bx, by); g.lineTo(w - bx + 4, h / 2); g.lineTo(w - bx, h - by);
    g.stroke({ color: col, width: 1.5, alpha: 0.3 });
  },

  'cil-layers': (g, w, h, col) => {
    // Stacked rects
    for (let i = 0; i < 3; i++) {
      const off = i * 4;
      g.roundRect(6 + off, 6 + off, w - 12 - off * 2, h - 12 - off * 2, 3);
      g.stroke({ color: col, width: 1, alpha: 0.15 + i * 0.1 });
    }
  },

  'cil-loop': (g, w, h, col) => {
    // Arc arrow
    const cx = w / 2, cy = h / 2, r = Math.min(w, h) * 0.3;
    g.arc(cx, cy, r, -Math.PI * 0.8, Math.PI * 0.5);
    g.stroke({ color: col, width: 1.5, alpha: 0.4 });
    // Arrowhead at end of arc
    const ax = cx + r * Math.cos(Math.PI * 0.5);
    const ay = cy + r * Math.sin(Math.PI * 0.5);
    g.moveTo(ax - 4, ay - 4); g.lineTo(ax, ay); g.lineTo(ax + 4, ay - 4);
    g.stroke({ color: col, width: 1.5, alpha: 0.4 });
  },

  'cil-graph': (g, w, h, col) => {
    // Scatter dots + lines
    const pts = [[w*0.25, h*0.3], [w*0.6, h*0.25], [w*0.75, h*0.6], [w*0.35, h*0.7]];
    for (const [x, y] of pts) {
      g.circle(x, y, 3);
      g.fill({ color: col, alpha: 0.35 });
    }
    g.moveTo(pts[0][0], pts[0][1]); g.lineTo(pts[1][0], pts[1][1]);
    g.lineTo(pts[2][0], pts[2][1]); g.lineTo(pts[3][0], pts[3][1]);
    g.lineTo(pts[0][0], pts[0][1]);
    g.stroke({ color: col, width: 1, alpha: 0.2 });
  },
};

// ── Bloom glow factory ──────────────────────────────────────────────────────

function createGlowSprite(w: number, h: number, glowColor: number): Graphics {
  const glow = new Graphics();
  const pad = 20; // glow extends beyond cell bbox
  glow.roundRect(-pad, -pad, w + pad * 2, h + pad * 2, 12);
  glow.fill({ color: glowColor, alpha: 0.25 });

  // BlurFilter for bloom — GPU handles all the convolution
  const blur = new BlurFilter({ strength: 12, quality: 4 });
  glow.filters = [blur];

  return glow;
}

// ── Cell container builder ──────────────────────────────────────────────────

function buildCellContainer(desc: CellDescriptor): Container {
  const { bbox, species, label, z } = desc;
  const { w, h } = bbox;
  const cols = getColours(species);

  const container = new Container();
  container.position.set(bbox.x, bbox.y);
  container.zIndex = z;

  // 1. Bloom glow (behind everything)
  const glow = createGlowSprite(w, h, cols.glow);
  container.addChild(glow);

  // 2. Cell body — rounded rect with fill + stroke
  const body = new Graphics();
  body.roundRect(0, 0, w, h, 8);
  body.fill({ color: cols.fill, alpha: 0.9 });
  body.roundRect(0, 0, w, h, 8);
  body.stroke({ color: cols.stroke, width: 1.5, alpha: 0.8 });
  container.addChild(body);

  // 3. Species pattern overlay
  const pattern = new Graphics();
  const drawer = SPECIES_PATTERNS[species] ?? SPECIES_PATTERNS['cil-code'];
  drawer(pattern, w, h, cols.stroke);
  container.addChild(pattern);

  // 4. Label text
  const style = new TextStyle({
    fontFamily: 'Inter, system-ui, sans-serif',
    fontSize: 11,
    fill: 0xFFFFFF,
    fontWeight: '500',
  });
  const txt = new Text({ text: label, style });
  txt.anchor.set(0.5);
  txt.position.set(w / 2, h / 2);
  container.addChild(txt);

  return container;
}

// ── Edge renderer ───────────────────────────────────────────────────────────

function drawEdges(
  g: Graphics,
  edges: EdgeDescriptor[],
  cellMap: Map<string, CellDescriptor>,
): void {
  for (const edge of edges) {
    const src = cellMap.get(edge.source);
    const tgt = cellMap.get(edge.target);
    if (!src || !tgt) continue;

    const sx = src.bbox.x + src.bbox.w / 2;
    const sy = src.bbox.y + src.bbox.h;       // bottom center
    const tx = tgt.bbox.x + tgt.bbox.w / 2;
    const ty = tgt.bbox.y;                     // top center

    if (edge.type === 'skip_connection') {
      // Bezier curve — skip connection
      const cx = Math.max(sx, tx) + 80;
      g.moveTo(sx, sy);
      g.bezierCurveTo(cx, sy, cx, ty, tx, ty);
      g.stroke({
        color: 0x4CAF50,
        width: 2,
        alpha: 0.6,
      });
    } else {
      // Straight line with slight curve for aesthetics
      const mid_y = (sy + ty) / 2;
      g.moveTo(sx, sy);
      g.bezierCurveTo(sx, mid_y, tx, mid_y, tx, ty);
      g.stroke({
        color: 0x999999,
        width: 1.5,
        alpha: 0.5,
      });
    }

    // Arrowhead at target
    const angle = Math.atan2(ty - sy, tx - sx);
    const arrLen = 8;
    g.moveTo(tx - arrLen * Math.cos(angle - 0.4), ty - arrLen * Math.sin(angle - 0.4));
    g.lineTo(tx, ty);
    g.lineTo(tx - arrLen * Math.cos(angle + 0.4), ty - arrLen * Math.sin(angle + 0.4));
    g.stroke({ color: 0x999999, width: 1.5, alpha: 0.5 });
  }
}

// ── Main renderer ───────────────────────────────────────────────────────────

export async function renderCellGraph(
  canvas: HTMLCanvasElement,
  cells: CellDescriptor[],
  edges: EdgeDescriptor[],
): Promise<Application> {
  const app = new Application();
  await app.init({
    canvas,
    width: canvas.width,
    height: canvas.height,
    backgroundColor: 0x1A1A2E,
    antialias: true,            // GPU sub-pixel smoothing
    resolution: window.devicePixelRatio || 1,
    autoDensity: true,
  });

  // Enable z-index sorting
  app.stage.sortableChildren = true;

  // Build cell map
  const cellMap = new Map<string, CellDescriptor>();
  for (const c of cells) cellMap.set(c.cell_id, c);

  // Draw edges first (behind cells)
  const edgeLayer = new Graphics();
  edgeLayer.zIndex = 0;
  drawEdges(edgeLayer, edges, cellMap);
  app.stage.addChild(edgeLayer);

  // Draw cells
  for (const cell of cells) {
    const container = buildCellContainer(cell);
    app.stage.addChild(container);
  }

  return app;
}
