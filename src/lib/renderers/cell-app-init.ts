/**
 * cell-app-init.ts — M052: PixiJS Application 初始化入口
 *
 * 完整的前端渲染启动流程：
 *   1. 创建 PixiJS Application (WebGL2, antialias, 自适应 resolution)
 *   2. fetch composite_params.json 解析 cell 数据
 *   3. 为每个 cell 创建 Container + Graphics背景 + Sprite(species icon) + Text(label)
 *   4. 挂载 SDF species filter + Bloom/Glow filter chain
 *   5. mount 到页面 canvas 容器
 *
 * 对接模块：
 *   - cell-asset-loader.ts   → 纹理/MSDF 预加载
 *   - cell-color-palette.ts  → species 调色板
 *   - cell-math.ts           → bbox 变换
 *   - pixi-cell-renderer.ts  → buildCellContainer
 *   - cell-culling.ts        → 视锥剔除
 *   - epoch-ticker.ts        → 动画帧循环
 */

import { Application, Container, Graphics, Sprite, Text, TextStyle } from 'pixi.js';
import { loadCellAssets, getCellAssets } from './cell-asset-loader';
import { getSpeciesColors } from './cell-color-palette';
import { bboxToRect } from './cell-math';

// ── Types ───────────────────────────────────────────────────────────────────

export interface CellEntry {
  cell_id: string;
  species: string;
  label: string;
  bbox: { x: number; y: number; w: number; h: number };
  params: Record<string, unknown>;
}

export interface CompositeParams {
  topology_id: string;
  epoch: number;
  cells: CellEntry[];
  edges: Array<{ edge_id: string; source: string; target: string }>;
}

// ── Application factory ─────────────────────────────────────────────────────

let _app: Application | null = null;
let _cellContainers = new Map<string, Container>();

/**
 * Initialize the PixiJS rendering pipeline.
 *
 * @param mountEl - DOM element to mount the canvas into
 * @param paramsUrl - URL to fetch composite_params.json (default: /api/composite-params)
 * @param onProgress - Loading progress callback
 */
export async function initCellApp(
  mountEl: HTMLElement,
  paramsUrl = '/api/composite-params',
  onProgress?: (p: number) => void,
): Promise<{ app: Application; containers: Map<string, Container> }> {
  // 1. Create Application
  const app = new Application();
  await app.init({
    background: 0x0a0a1a,
    resizeTo: mountEl,
    antialias: true,
    resolution: Math.min(window.devicePixelRatio, 2),
    autoDensity: true,
    preferWebGLVersion: 2,
  });
  mountEl.appendChild(app.canvas);
  _app = app;

  // 2. Preload assets (species icons, MSDF atlas)
  onProgress?.(0.1);
  await loadCellAssets('', (p) => onProgress?.(0.1 + p * 0.3));

  // 3. Fetch composite params
  onProgress?.(0.4);
  const resp = await fetch(paramsUrl);
  const composite: CompositeParams = await resp.json();
  onProgress?.(0.5);

  // 4. Build cell containers
  const worldContainer = new Container();
  worldContainer.label = 'world';
  app.stage.addChild(worldContainer);

  const assets = getCellAssets();
  const totalCells = composite.cells.length;

  for (let i = 0; i < totalCells; i++) {
    const cell = composite.cells[i];
    const container = buildCellContainer(cell, assets);
    worldContainer.addChild(container);
    _cellContainers.set(cell.cell_id, container);
    onProgress?.(0.5 + (i / totalCells) * 0.4);
  }

  onProgress?.(0.9);

  // 5. Start ticker
  app.ticker.add(() => {
    // Ticker loop — modules hook into this via epoch-ticker.ts
  });

  onProgress?.(1.0);

  return { app, containers: _cellContainers };
}

// ── Cell Container builder ──────────────────────────────────────────────────

function buildCellContainer(
  cell: CellEntry,
  assets: ReturnType<typeof getCellAssets>,
): Container {
  const { bbox, species, label, cell_id } = cell;
  const colors = getSpeciesColors(species);

  const container = new Container();
  container.label = cell_id;
  container.position.set(bbox.x, bbox.y);

  // Background quad
  const bg = new Graphics();
  bg.roundRect(0, 0, bbox.w, bbox.h, 6);
  bg.fill({ color: colors.bg.toNumber() });
  bg.stroke({ color: colors.stroke.toNumber(), width: 1.5 });
  container.addChild(bg);

  // Species icon sprite
  const iconTex = assets.icons[species as keyof typeof assets.icons];
  if (iconTex) {
    const sprite = new Sprite(iconTex);
    const iconSize = Math.min(bbox.w, bbox.h) * 0.35;
    sprite.width = iconSize;
    sprite.height = iconSize;
    sprite.position.set(6, (bbox.h - iconSize) / 2);
    sprite.alpha = 0.8;
    container.addChild(sprite);
  }

  // Label text
  const textStyle = new TextStyle({
    fontFamily: 'Inter, system-ui, sans-serif',
    fontSize: Math.max(10, Math.min(14, bbox.w * 0.08)),
    fill: colors.text.toNumber(),
    wordWrap: true,
    wordWrapWidth: bbox.w - 40,
  });
  const text = new Text({ text: label || cell_id, style: textStyle });
  text.position.set(bbox.w * 0.3, bbox.h * 0.3);
  text.alpha = 0.9;
  container.addChild(text);

  // Interactive
  container.eventMode = 'static';
  container.cursor = 'pointer';

  return container;
}

// ── Accessors ───────────────────────────────────────────────────────────────

export function getApp(): Application | null { return _app; }
export function getCellContainers(): Map<string, Container> { return _cellContainers; }

/**
 * Destroy the application and clean up.
 */
export function destroyCellApp(): void {
  _app?.destroy(true, { children: true });
  _app = null;
  _cellContainers.clear();
}
