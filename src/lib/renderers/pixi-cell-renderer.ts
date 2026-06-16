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
 * Live Poll 模式 (pollCellChannels):
 * - 每 500ms fetch /api/cells 拉取最新 CellDescriptor[]
 * - 位置变化 → lerp 平滑过渡 (alpha += (target - current) * 0.1)
 * - 新 cell fade in (alpha 0→1)，消失 cell fade out (alpha 1→0) 后销毁
 * - edge layer 每帧跟着 cell 当前位置实时重绘
 *
 * Upstream reference:
 *   upstream/pixijs-engine/src/scene/graphics/shared/Graphics.ts
 *   upstream/pixijs-engine/src/filters/defaults/blur/
 *   skills/pixijs/pixijs-filters/SKILL.md
 *   skills/pixijs/pixijs-graphics/SKILL.md
 */

import { Application } from '../../upstream/pixijs-engine/src/app/Application';
import { Container } from '../../upstream/pixijs-engine/src/scene/container/Container';
import { Graphics } from '../../upstream/pixijs-engine/src/scene/graphics/shared/Graphics';
import { Text } from '../../upstream/pixijs-engine/src/scene/text/Text';
import { TextStyle } from '../../upstream/pixijs-engine/src/scene/text/TextStyle';
import { Ticker } from '../../upstream/pixijs-engine/src/ticker/Ticker';
import { AdvancedBloomFilter } from '../../../upstream/pixijs-filters/src/advanced-bloom';
import { DropShadowFilter } from '../../../upstream/pixijs-filters/src/drop-shadow';

// ── params.json schema (M152 output) ───────────────────────────────────────

export interface ParamsJson {
  cell_id: string;
  species: string;
  bbox: { x: number; y: number; w: number; h: number; z?: number };
  z?: number;
  /** crowding_opacity — sets container.alpha */
  opacity?: number;
  /** hex string e.g. "#3F51B5" */
  fill_color?: string;
  /** hex string e.g. "#3F51B5" */
  stroke_color?: string;
  label?: string;
  font_size?: number;
  shadow?: {
    dx: number;
    dy: number;
    blur: number;
    opacity: number;
  };
  species_params?: Record<string, unknown>;
  epoch?: number;
}

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
  /** Optional params.json overlay (M152 output) — merged at build time or runtime */
  params?: ParamsJson;
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

function hexToNum(hex: string): number {
  return parseInt(hex.replace('#', ''), 16);
}

function getColours(
  species: string,
  fillOverride?: string,
  strokeOverride?: string,
): { fill: number; stroke: number; glow: number } {
  const base = SPECIES_COLOURS[species] ?? { fill: 0x90A4AE, stroke: 0x607D8B, glow: 0xB0BEC5 };
  return {
    fill:   fillOverride   ? hexToNum(fillOverride)   : base.fill,
    stroke: strokeOverride ? hexToNum(strokeOverride) : base.stroke,
    glow:   base.glow,
  };
}

// ── Species pattern drawers ─────────────────────────────────────────────────

type PatternDrawer = (g: Graphics, w: number, h: number, col: number, sp?: Record<string, unknown>) => void;

const SPECIES_PATTERNS: Record<string, PatternDrawer> = {
  'cil-eye': (g, w, h, col, sp) => {
    const cx = w / 2, cy = h / 2;
    const rOuter = (sp?.r_outer as number | undefined) ?? Math.min(w, h) * 0.35;
    const ringCount = (sp?.ring_count as number | undefined) ?? 3;
    const pupilRadius = (sp?.pupil_radius as number | undefined) ?? rOuter * 0.15;
    const rInnerRatio = (sp?.r_inner_ratio as number | undefined) ?? 0.3;
    for (let i = ringCount; i >= 1; i--) {
      g.circle(cx, cy, rOuter * (i / ringCount));
      g.fill({ color: col, alpha: 0.08 * i * (3 / ringCount) });
    }
    g.circle(cx, cy, pupilRadius);
    g.fill({ color: col, alpha: 0.5 });
    // draw iris line using r_inner_ratio
    g.circle(cx, cy, rOuter * rInnerRatio);
    g.stroke({ color: col, width: 0.8, alpha: 0.25 });
  },

  'cil-vector': (g, w, h, col, sp) => {
    const pad = 8;
    const arrowCount = (sp?.arrow_count as number | undefined) ?? 1;
    const arrowLength = (sp?.arrow_length as number | undefined) ?? (w - pad * 4);
    const angleSpread = (sp?.angle_spread as number | undefined) ?? 0;
    const cx = w / 2, cy = h / 2;
    const startX = cx - arrowLength / 2;
    for (let i = 0; i < arrowCount; i++) {
      const angle = arrowCount > 1 ? angleSpread * (i / (arrowCount - 1) - 0.5) : 0;
      const endX = startX + arrowLength * Math.cos(angle);
      const endY = cy + arrowLength * Math.sin(angle);
      g.moveTo(startX, cy);
      g.lineTo(endX, endY);
      g.moveTo(endX, endY);
      g.lineTo(endX - 6 * Math.cos(angle - 0.4), endY - 6 * Math.sin(angle - 0.4));
      g.moveTo(endX, endY);
      g.lineTo(endX - 6 * Math.cos(angle + 0.4), endY - 6 * Math.sin(angle + 0.4));
      g.stroke({ color: col, width: 1.5, alpha: 0.4 });
    }
  },

  'cil-bolt': (g, w, h, col, sp) => {
    const n = (sp?.zigzag_segments as number | undefined) ?? (sp?.num_rays as number | undefined) ?? 5;
    const dy = h / (n + 1), amp = w * 0.15;
    g.moveTo(w / 2, 6);
    for (let i = 1; i <= n; i++) {
      const x = w / 2 + (i % 2 === 1 ? amp : -amp);
      g.lineTo(x, dy * i + 6);
    }
    g.lineTo(w / 2, h - 6);
    g.stroke({ color: col, width: 1.5, alpha: 0.35 });
  },

  'cil-plus': (g, w, h, col, sp) => {
    const cx = w / 2, cy = h / 2;
    const arm = (sp?.arm_length as number | undefined) ?? Math.min(w, h) * 0.3;
    const strokeWidth = (sp?.stroke_width as number | undefined) ?? 2;
    const dashCorners = (sp?.dash_corners as boolean | undefined) ?? false;
    g.moveTo(cx - arm, cy); g.lineTo(cx + arm, cy);
    g.moveTo(cx, cy - arm); g.lineTo(cx, cy + arm);
    g.stroke({ color: col, width: strokeWidth, alpha: 0.3 });
    if (dashCorners) {
      const d = arm * 0.6;
      for (const [ox, oy] of [[-d,-d],[d,-d],[d,d],[-d,d]]) {
        g.circle(cx + ox, cy + oy, 1.5);
        g.fill({ color: col, alpha: 0.2 });
      }
    }
  },

  'cil-arrow-right': (g, w, h, col, sp) => {
    const cx = w / 2, cy = h / 2, sz = Math.min(w, h) * 0.25;
    g.moveTo(cx - sz, cy - sz);
    g.lineTo(cx + sz * 0.5, cy);
    g.lineTo(cx - sz, cy + sz);
    g.stroke({ color: col, width: 2, alpha: 0.4 });
  },

  'cil-filter': (g, w, h, col, sp) => {
    const pad = 10, gw = (w - pad * 2) / 3, gh = (h - pad * 2) / 3;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        g.rect(pad + c * gw + 1, pad + r * gh + 1, gw - 2, gh - 2);
        g.stroke({ color: col, width: 0.8, alpha: 0.2 });
      }
    }
  },

  'cil-code': (g, w, h, col, sp) => {
    const bx = 12, by = 8;
    g.moveTo(bx, by); g.lineTo(bx - 4, h / 2); g.lineTo(bx, h - by);
    g.moveTo(w - bx, by); g.lineTo(w - bx + 4, h / 2); g.lineTo(w - bx, h - by);
    g.stroke({ color: col, width: 1.5, alpha: 0.3 });
  },

  'cil-layers': (g, w, h, col, sp) => {
    for (let i = 0; i < 3; i++) {
      const off = i * 4;
      g.roundRect(6 + off, 6 + off, w - 12 - off * 2, h - 12 - off * 2, 3);
      g.stroke({ color: col, width: 1, alpha: 0.15 + i * 0.1 });
    }
  },

  'cil-loop': (g, w, h, col, sp) => {
    const cx = w / 2, cy = h / 2, r = Math.min(w, h) * 0.3;
    g.arc(cx, cy, r, -Math.PI * 0.8, Math.PI * 0.5);
    g.stroke({ color: col, width: 1.5, alpha: 0.4 });
    const ax = cx + r * Math.cos(Math.PI * 0.5);
    const ay = cy + r * Math.sin(Math.PI * 0.5);
    g.moveTo(ax - 4, ay - 4); g.lineTo(ax, ay); g.lineTo(ax + 4, ay - 4);
    g.stroke({ color: col, width: 1.5, alpha: 0.4 });
  },

  'cil-graph': (g, w, h, col, sp) => {
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

const SPECIES_BLOOM_SCALE: Record<string, number> = {
  'cil-eye':  2.0,
  'cil-bolt': 1.8,
  'cil-code': 0.8,
};

function createGlowSprite(w: number, h: number, glowColor: number, species: string): Graphics {
  const glow = new Graphics();
  const pad = 20;
  glow.roundRect(-pad, -pad, w + pad * 2, h + pad * 2, 12);
  glow.fill({ color: glowColor, alpha: 0.25 });
  const bloomScale = SPECIES_BLOOM_SCALE[species] ?? 1.5;
  const bloom = new AdvancedBloomFilter({ bloomScale, threshold: 0.5, blur: 8, quality: 4 });
  glow.filters = [bloom];
  return glow;
}

// ── Cell container builder ──────────────────────────────────────────────────

function buildCellContainer(desc: CellDescriptor): Container {
  const { bbox, species, label, z } = desc;
  const { w, h } = bbox;

  // ── Resolve params.json overrides ──────────────────────────────────────
  const p = desc.params;
  const cols = getColours(species, p?.fill_color, p?.stroke_color);
  const speciesParams = p?.species_params;
  const crowdingOpacity = (p?.opacity !== undefined) ? Math.max(0.15, Math.min(1, p.opacity)) : 1;
  const fontSize = p?.font_size ?? 11;

  const container = new Container();
  container.position.set(bbox.x, bbox.y);
  container.zIndex = z;

  const glow = createGlowSprite(w, h, cols.glow, species);
  container.addChild(glow);

  const body = new Graphics();
  body.roundRect(0, 0, w, h, 8);
  body.fill({ color: cols.fill, alpha: 0.9 });
  body.roundRect(0, 0, w, h, 8);
  body.stroke({ color: cols.stroke, width: 1.5, alpha: 0.8 });

  // ── DropShadowFilter from params.json shadow field ──────────────────────
  if (p?.shadow) {
    const { dx, dy, blur, opacity } = p.shadow;
    const dropShadow = new DropShadowFilter({
      offset: { x: dx, y: dy },
      blur,
      alpha: opacity,
      color: 0x000000,
      quality: 4,
    });
    // Compose with bloom on the body: [dropShadow, ...existing]
    body.filters = [dropShadow];
  }

  container.addChild(body);

  const pattern = new Graphics();
  const drawer = SPECIES_PATTERNS[species] ?? SPECIES_PATTERNS['cil-code'];
  drawer(pattern, w, h, cols.stroke, speciesParams);
  container.addChild(pattern);

  const style = new TextStyle({
    fontFamily: 'Inter, system-ui, sans-serif',
    fontSize,
    fill: 0xFFFFFF,
    fontWeight: '500',
  });
  const txt = new Text({ text: label, style });
  txt.anchor.set(0.5);
  txt.position.set(w / 2, h / 2);
  container.addChild(txt);

  // ── crowding_opacity: applies after fade-in completes ──────────────────
  // Store as userData so the poll loop can respect it when fading in
  (container as any).__targetAlpha = crowdingOpacity;
  container.alpha = 0; // caller sets to 0 for fade-in; will stop at __targetAlpha

  return container;
}

// ── Edge renderer ───────────────────────────────────────────────────────────

function drawEdges(
  g: Graphics,
  edges: EdgeDescriptor[],
  cellMap: Map<string, { x: number; y: number; w: number; h: number }>,
): void {
  g.clear();
  for (const edge of edges) {
    const src = cellMap.get(edge.source);
    const tgt = cellMap.get(edge.target);
    if (!src || !tgt) continue;

    const sx = src.x + src.w / 2;
    const sy = src.y + src.h;
    const tx = tgt.x + tgt.w / 2;
    const ty = tgt.y;

    if (edge.type === 'skip_connection') {
      const cx = Math.max(sx, tx) + 80;
      g.moveTo(sx, sy);
      g.bezierCurveTo(cx, sy, cx, ty, tx, ty);
      g.stroke({ color: 0x4CAF50, width: 2, alpha: 0.6 });
    } else {
      const mid_y = (sy + ty) / 2;
      g.moveTo(sx, sy);
      g.bezierCurveTo(sx, mid_y, tx, mid_y, tx, ty);
      g.stroke({ color: 0x999999, width: 1.5, alpha: 0.5 });
    }

    const angle = Math.atan2(ty - sy, tx - sx);
    const arrLen = 8;
    g.moveTo(tx - arrLen * Math.cos(angle - 0.4), ty - arrLen * Math.sin(angle - 0.4));
    g.lineTo(tx, ty);
    g.lineTo(tx - arrLen * Math.cos(angle + 0.4), ty - arrLen * Math.sin(angle + 0.4));
    g.stroke({ color: 0x999999, width: 1.5, alpha: 0.5 });
  }
}

// ── Live cell state (used by poll loop) ────────────────────────────────────

interface LiveCell {
  desc: CellDescriptor;
  /** current rendered position (lerp target) */
  curX: number;
  curY: number;
  /** target position from latest poll */
  tgtX: number;
  tgtY: number;
  container: Container;
  /** fade direction: +1 = fading in, -1 = fading out, 0 = stable */
  fadeDir: 0 | 1 | -1;
}

const LERP_FACTOR  = 0.1;   // position lerp per frame
const FADE_SPEED   = 0.05;  // alpha change per frame

// ── pollCellChannels ────────────────────────────────────────────────────────

/**
 * pollCellChannels — starts a 500ms polling loop against /api/cells.
 *
 * Behaviour:
 *   1. Every 500ms fetch /api/cells → CellDescriptor[]
 *   2. New cells: spawn container at alpha=0, fade in to 1
 *   3. Removed cells: fade out to 0, then destroy
 *   4. Existing cells: lerp position toward new bbox (alpha += (target-current)*0.1)
 *   5. Edge layer redraws every frame based on current live positions
 *
 * @param app        Running PixiJS Application
 * @param edges      EdgeDescriptor[] (static topology — edges don't change)
 * @param edgeLayer  Graphics node dedicated to edge drawing
 * @returns          stop() to cancel polling + animation
 */
export function pollCellChannels(
  app: Application,
  edges: EdgeDescriptor[],
  edgeLayer: Graphics,
): () => void {
  // Map of live cells keyed by cell_id
  const live = new Map<string, LiveCell>();

  let pollHandle: ReturnType<typeof setInterval> | null = null;
  let stopped = false;

  // ── Per-frame tick: lerp positions + fade + redraw edges ───────────────
  function tick(_ticker: Ticker): void {
    if (stopped) return;

    // Build bbox snapshot for edge drawing
    const bboxSnap = new Map<string, { x: number; y: number; w: number; h: number }>();

    for (const [id, lc] of live) {
      // Lerp position
      lc.curX += (lc.tgtX - lc.curX) * LERP_FACTOR;
      lc.curY += (lc.tgtY - lc.curY) * LERP_FACTOR;
      lc.container.position.set(lc.curX, lc.curY);

      // Fade
      const targetAlpha: number = (lc.container as any).__targetAlpha ?? 1;
      if (lc.fadeDir === 1) {
        lc.container.alpha = Math.min(targetAlpha, lc.container.alpha + FADE_SPEED);
        if (lc.container.alpha >= targetAlpha) lc.fadeDir = 0;
      } else if (lc.fadeDir === -1) {
        lc.container.alpha = Math.max(0, lc.container.alpha - FADE_SPEED);
        if (lc.container.alpha <= 0) {
          // Fully faded — destroy and remove
          app.stage.removeChild(lc.container);
          lc.container.destroy({ children: true });
          live.delete(id);
          continue;
        }
      }

      // Record current rendered bbox for edge drawing
      const { w, h } = lc.desc.bbox;
      bboxSnap.set(id, { x: lc.curX, y: lc.curY, w, h });
    }

    // Redraw edges at current (lerped) positions
    drawEdges(edgeLayer, edges, bboxSnap);
  }

  app.ticker.add(tick);

  // ── Poll loop: fetch /api/cells every 500ms ────────────────────────────
  async function fetchAndReconcile(): Promise<void> {
    if (stopped) return;
    try {
      const res = await fetch('/api/cells');
      if (!res.ok) return;
      const incoming: CellDescriptor[] = await res.json();

      const seen = new Set<string>();

      for (const desc of incoming) {
        seen.add(desc.cell_id);
        const tgtX = desc.bbox.x;
        const tgtY = desc.bbox.y;

        if (live.has(desc.cell_id)) {
          // Existing cell — update target position
          const lc = live.get(desc.cell_id)!;
          lc.tgtX = tgtX;
          lc.tgtY = tgtY;
          // Also update desc so edge dimensions stay correct
          lc.desc = desc;
          // Cancel any ongoing fade-out if cell reappears
          if (lc.fadeDir === -1) lc.fadeDir = 1;
        } else {
          // New cell — spawn at target, alpha=0, fade in
          const container = buildCellContainer(desc);
          container.alpha = 0;
          app.stage.addChild(container);

          const lc: LiveCell = {
            desc,
            curX: tgtX,
            curY: tgtY,
            tgtX,
            tgtY,
            container,
            fadeDir: 1,
          };
          live.set(desc.cell_id, lc);
        }
      }

      // Cells in live but NOT in incoming → fade out
      for (const [id, lc] of live) {
        if (!seen.has(id) && lc.fadeDir !== -1) {
          lc.fadeDir = -1;
        }
      }
    } catch (err) {
      console.warn('[pollCellChannels] fetch error:', err);
    }
  }

  // Initial fetch, then schedule
  fetchAndReconcile();
  pollHandle = setInterval(fetchAndReconcile, 500);

  // ── Return stop handle ─────────────────────────────────────────────────
  return () => {
    stopped = true;
    if (pollHandle !== null) clearInterval(pollHandle);
    app.ticker.remove(tick);
  };
}

// ── Main renderer (static, one-shot) ───────────────────────────────────────

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
    antialias: true,
    resolution: window.devicePixelRatio || 1,
    autoDensity: true,
  });

  app.stage.sortableChildren = true;

  const cellMap = new Map<string, CellDescriptor>();
  for (const c of cells) cellMap.set(c.cell_id, c);

  // Build bbox map for edge drawing
  const bboxMap = new Map<string, { x: number; y: number; w: number; h: number }>();
  for (const c of cells) bboxMap.set(c.cell_id, c.bbox);

  // Draw edges first (behind cells)
  const edgeLayer = new Graphics();
  edgeLayer.zIndex = 0;
  drawEdges(edgeLayer, edges, bboxMap);
  app.stage.addChild(edgeLayer);

  // Draw cells
  for (const cell of cells) {
    const container = buildCellContainer(cell);
    // In static mode: skip fade-in, go straight to target alpha
    container.alpha = (container as any).__targetAlpha ?? 1;
    app.stage.addChild(container);
  }

  return app;
}

// ── Live poll renderer (uses pollCellChannels) ──────────────────────────────

/**
 * renderCellGraphLive — initialise a PixiJS canvas in live-poll mode.
 *
 * No initial cells are rendered; the poll loop populates the stage.
 * Returns both the Application and a stop() handle.
 */
export async function renderCellGraphLive(
  canvas: HTMLCanvasElement,
  edges: EdgeDescriptor[],
): Promise<{ app: Application; stop: () => void }> {
  const app = new Application();
  await app.init({
    canvas,
    width: canvas.width,
    height: canvas.height,
    backgroundColor: 0x1A1A2E,
    antialias: true,
    resolution: window.devicePixelRatio || 1,
    autoDensity: true,
  });

  app.stage.sortableChildren = true;

  const edgeLayer = new Graphics();
  edgeLayer.zIndex = 0;
  app.stage.addChild(edgeLayer);

  const stop = pollCellChannels(app, edges, edgeLayer);

  return { app, stop };
}

// ── params.json loader helper ───────────────────────────────────────────────

/**
 * mergeParamsJson — fetches a params.json file for a given cell_id and
 * merges it into a CellDescriptor so that fill_color, stroke_color, shadow,
 * opacity, and species_params are all available to buildCellContainer.
 *
 * Usage:
 *   const cell = await mergeParamsJson(desc, '/channels/cell/self_attn/params.json');
 */
export async function mergeParamsJson(
  desc: CellDescriptor,
  paramsUrl: string,
): Promise<CellDescriptor> {
  try {
    const res = await fetch(paramsUrl);
    if (!res.ok) return desc;
    const p: ParamsJson = await res.json();
    return { ...desc, params: p };
  } catch {
    return desc;
  }
}
