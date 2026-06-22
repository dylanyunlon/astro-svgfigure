/**
 * cell-label-renderer.ts — MSDF 文字标签渲染器
 *
 * 职责:
 *   把每个 cell 的 label 字符串渲染为 GPU MSDF 文字，对齐到 cell bbox 中心下方。
 *   替代 PixiJS Text 的逐对象 canvas-texture 方案（500 labels = 500 draw calls）
 *   为单次或少量 draw call 的 instanced MSDF 渲染（500 labels ≈ 1-5 draw calls）。
 *
 * 架构:
 *   CellLabelRenderer
 *     ├── MSDFAtlas        共享字体 atlas（JSON metrics + PNG texture）
 *     ├── GLText           instanced MSDF quad 渲染（来自 gl-text.ts）
 *     ├── CellParamsJson[] 数据源（来自 CellInstanceManager 或 pubsub 事件）
 *     └── CellEventSource  可选 SSE 实时更新桥接
 *
 * 渲染流水线:
 *   1. init()           → loadMSDFAtlas() 加载字体 atlas
 *   2. syncFromCells()  → 遍历 CellParamsJson[] 注册 / 更新所有 label
 *   3. draw()           → GLText.draw(projection) 一次或几次 draw call 输出所有文字
 *   4. onCellUpdate()   → 接收 pubsub 增量更新，只改变单个 label 位置/文字/颜色
 *
 * 对齐策略:
 *   label 水平居中于 cell bbox, 垂直位于 bbox 底边下方 labelOffsetY 处。
 *   当 camera zoom 低于 minZoomForLabels 时整体隐藏（LOD 优化）。
 *
 * 颜色策略:
 *   默认继承 cell fill_color（解析 hex → Color）。
 *   可通过 setLabelColor() 单独覆盖。
 *
 * 依赖:
 *   gl-text.ts           MSDFAtlas, GLText, GLTextGeometry, loadMSDFAtlas,
 *                         makeOrthoProjection
 *   color-utils.ts        Color
 *   CellInstanceManager   CellParamsJson, CellBBox
 *   CellEventSource       CellUpdateEvent (pubsub)
 *
 * Author: dylanyunlon <dogechat@163.com>
 */

import {
  MSDFAtlas,
  GLText,
  loadMSDFAtlas,
  makeOrthoProjection,
} from './gl-text';
import type {
  GLTextGeometryOptions,
} from './gl-text';
import { Color } from '../color-utils';
import type { CellParamsJson, CellBBox } from '../renderer/CellInstanceManager';
import type { CellUpdateEvent } from '../CellEventSource';

// ── 配置类型 ──────────────────────────────────────────────────────────────────

/** CellLabelRenderer 初始化选项 */
export interface CellLabelRendererOptions {
  /** MSDF atlas JSON url (默认 '/fonts/inter-medium-msdf.json') */
  atlasJsonUrl?: string;
  /** MSDF atlas PNG url (默认 '/fonts/inter-medium-msdf.png') */
  atlasImageUrl?: string;
  /** 默认字号 (px, 默认 12) */
  fontSize?: number;
  /** label 在 cell bbox 底边下方偏移 (px, 默认 6) */
  labelOffsetY?: number;
  /** 字间距补偿 (px, 默认 0) */
  letterSpacing?: number;
  /** 对齐方式 (默认 'center') */
  align?: 'left' | 'center' | 'right';
  /** 低于此 zoom 隐藏所有 label (默认 0.3) */
  minZoomForLabels?: number;
  /** msdfgen pxRange (默认 4) */
  pxRange?: number;
  /** 默认 label 颜色 (当 cell 无 fill_color 时使用, 默认 '#e0e0e0') */
  defaultColor?: string;
  /** label 不透明度 (默认 1.0) */
  opacity?: number;
}

/** 渲染时传入的相机/视口参数 */
export interface CellLabelDrawParams {
  /** 视口宽度 (px) */
  viewportWidth: number;
  /** 视口高度 (px) */
  viewportHeight: number;
  /** 相机偏移 X (px, 默认 0) */
  cameraX?: number;
  /** 相机偏移 Y (px, 默认 0) */
  cameraY?: number;
  /** 相机缩放 (默认 1.0) */
  zoom?: number;
}

/** 内部 label 记录 */
interface LabelRecord {
  cellId: string;
  text: string;
  /** label 世界坐标 (cell bbox 中心底边 + offset) */
  x: number;
  y: number;
  color: Color;
  fontSize: number;
  visible: boolean;
}

// ── CellLabelRenderer ─────────────────────────────────────────────────────────

/**
 * CellLabelRenderer — 为所有 cell 提供 MSDF 文字标签渲染.
 *
 * @example
 *   const renderer = new CellLabelRenderer(gl);
 *   await renderer.init();
 *   renderer.syncFromCells(cellDescriptors);
 *
 *   // render loop
 *   function frame() {
 *     renderer.draw({
 *       viewportWidth: canvas.width,
 *       viewportHeight: canvas.height,
 *       cameraX: cam.x,
 *       cameraY: cam.y,
 *       zoom: cam.zoom,
 *     });
 *   }
 *
 *   // pubsub 增量更新
 *   cellEventSource.addListener(ev => renderer.onCellUpdate(ev));
 */
export class CellLabelRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly opts: Required<CellLabelRendererOptions>;

  /** 已加载的 MSDF atlas (init 后可用) */
  private atlas: MSDFAtlas | null = null;
  /** instanced MSDF 渲染器 */
  private glText: GLText | null = null;

  /** cellId → LabelRecord */
  private labels = new Map<string, LabelRecord>();

  /** 初始化状态 */
  private _inited = false;
  /** 已销毁标记 */
  private _destroyed = false;

  constructor(gl: WebGL2RenderingContext, opts: CellLabelRendererOptions = {}) {
    this.gl = gl;
    this.opts = {
      atlasJsonUrl:     opts.atlasJsonUrl     ?? '/fonts/inter-medium-msdf.json',
      atlasImageUrl:    opts.atlasImageUrl    ?? '/fonts/inter-medium-msdf.png',
      fontSize:         opts.fontSize         ?? 12,
      labelOffsetY:     opts.labelOffsetY     ?? 6,
      letterSpacing:    opts.letterSpacing    ?? 0,
      align:            opts.align            ?? 'center',
      minZoomForLabels: opts.minZoomForLabels ?? 0.3,
      pxRange:          opts.pxRange          ?? 4,
      defaultColor:     opts.defaultColor     ?? '#e0e0e0',
      opacity:          opts.opacity          ?? 1.0,
    };
  }

  // ── 初始化 ────────────────────────────────────────────────────────────────

  /**
   * init — 异步加载 MSDF atlas 并创建 GLText 渲染器.
   *
   * 必须在首次 draw() 之前调用. 幂等——重复调用无副作用.
   */
  async init(
    jsonUrl?: string,
    imageUrl?: string,
  ): Promise<void> {
    if (this._inited || this._destroyed) return;

    const jUrl = jsonUrl  ?? this.opts.atlasJsonUrl;
    const iUrl = imageUrl ?? this.opts.atlasImageUrl;

    this.atlas = await loadMSDFAtlas(this.gl, jUrl, iUrl);
    this.glText = new GLText(this.gl, this.atlas);
    this._inited = true;
  }

  /** 是否已完成初始化 */
  get inited(): boolean {
    return this._inited;
  }

  // ── 数据同步 (全量) ───────────────────────────────────────────────────────

  /**
   * syncFromCells — 从 CellParamsJson[] 全量同步所有 label.
   *
   * 清除旧 label, 根据每个 cell 的 label / bbox / fill_color 重建.
   * 适用于初始化和 epoch 切换时的全量刷新.
   *
   * @param cells  CellParamsJson 数组 (来自 CellInstanceManager 或 topology JSON)
   */
  syncFromCells(cells: CellParamsJson[]): void {
    if (!this._inited || !this.glText) {
      console.warn('[CellLabelRenderer] syncFromCells called before init()');
      return;
    }

    // 清除旧 label
    for (const id of this.labels.keys()) {
      this.glText.removeLabel(id);
    }
    this.labels.clear();

    // 注册新 label
    for (const cell of cells) {
      if (!cell.label) continue;

      const pos = this._bboxToLabelPos(cell.bbox);
      const color = this._resolveColor(cell.fill_color);
      const fontSize = (cell as any).font_size ?? this.opts.fontSize;

      const record: LabelRecord = {
        cellId: cell.cell_id,
        text: cell.label,
        x: pos.x,
        y: pos.y,
        color,
        fontSize,
        visible: true,
      };
      this.labels.set(cell.cell_id, record);

      this.glText.addLabel(
        cell.cell_id,
        cell.label,
        { x: pos.x, y: pos.y },
        color,
        this._buildGeomOpts(fontSize),
      );
    }
  }

  // ── 增量更新 (pubsub) ─────────────────────────────────────────────────────

  /**
   * onCellUpdate — 处理 CellEventSource pubsub 事件.
   *
   * 当 cell 位置/颜色/文字变化时, 仅更新受影响的单个 label,
   * 避免全量重建. 高频调用安全.
   *
   * @param event  CellUpdateEvent (来自 SSE cell_update 事件)
   */
  onCellUpdate(event: CellUpdateEvent): void {
    if (!this._inited || !this.glText) return;

    const { cell_id, params } = event;
    const existing = this.labels.get(cell_id);

    // 如果更新后的 cell 无 label, 移除
    if (!params.label) {
      if (existing) {
        this.glText.removeLabel(cell_id);
        this.labels.delete(cell_id);
      }
      return;
    }

    // 计算新位置
    const pos = params.bbox
      ? this._bboxToLabelPos(params.bbox)
      : existing
        ? { x: existing.x, y: existing.y }
        : { x: 0, y: 0 };

    const color = params.fill_color
      ? this._resolveColor(params.fill_color)
      : existing?.color ?? Color.fromHex(this.opts.defaultColor);

    const fontSize = (params as any).font_size ?? existing?.fontSize ?? this.opts.fontSize;

    if (existing) {
      // 更新位置
      if (params.bbox) {
        existing.x = pos.x;
        existing.y = pos.y;
        this.glText.setPosition(cell_id, pos.x, pos.y);
      }

      // 更新颜色
      if (params.fill_color) {
        existing.color = color;
        this.glText.setColor(cell_id, color);
      }

      // 更新文字内容 (需要重新 layout)
      if (params.label !== existing.text) {
        existing.text = params.label;
        this.glText.setText(cell_id, params.label);
      }
    } else {
      // 新 label
      const record: LabelRecord = {
        cellId: cell_id,
        text: params.label,
        x: pos.x,
        y: pos.y,
        color,
        fontSize,
        visible: true,
      };
      this.labels.set(cell_id, record);

      this.glText.addLabel(
        cell_id,
        params.label,
        { x: pos.x, y: pos.y },
        color,
        this._buildGeomOpts(fontSize),
      );
    }
  }

  // ── 单 label 操控 API ──────────────────────────────────────────────────────

  /** 更新单个 label 的世界坐标 (高频安全, 不触发 layout) */
  setLabelPosition(cellId: string, x: number, y: number): void {
    const record = this.labels.get(cellId);
    if (!record || !this.glText) return;
    record.x = x;
    record.y = y;
    this.glText.setPosition(cellId, x, y);
  }

  /** 覆盖单个 label 颜色 */
  setLabelColor(cellId: string, color: Color): void {
    const record = this.labels.get(cellId);
    if (!record || !this.glText) return;
    record.color = color;
    this.glText.setColor(cellId, color);
  }

  /** 设置单个 label 可见性 */
  setLabelVisible(cellId: string, visible: boolean): void {
    const record = this.labels.get(cellId);
    if (!record || !this.glText) return;
    record.visible = visible;
    this.glText.setVisible(cellId, visible);
  }

  /** 更新单个 label 文字 (触发 geometry layout 重建) */
  setLabelText(cellId: string, text: string): void {
    const record = this.labels.get(cellId);
    if (!record || !this.glText) return;
    record.text = text;
    this.glText.setText(cellId, text);
  }

  /** 移除单个 label */
  removeLabel(cellId: string): void {
    if (!this.glText) return;
    this.glText.removeLabel(cellId);
    this.labels.delete(cellId);
  }

  // ── 批量位置同步 (物理引擎回调) ─────────────────────────────────────────────

  /**
   * syncPositions — 批量更新 label 位置.
   *
   * 适用于物理引擎每帧推送 cell 新位置时, 批量对齐所有 label.
   * 比逐个调用 setLabelPosition 更紧凑.
   *
   * @param positions  cellId → { x, y, w, h } 的映射 (或 CellBBox)
   */
  syncPositions(positions: Map<string, CellBBox> | Record<string, CellBBox>): void {
    if (!this.glText) return;

    const entries = positions instanceof Map
      ? positions.entries()
      : Object.entries(positions);

    for (const [cellId, bbox] of entries) {
      const record = this.labels.get(cellId);
      if (!record) continue;

      const pos = this._bboxToLabelPos(bbox);
      record.x = pos.x;
      record.y = pos.y;
      this.glText.setPosition(cellId, pos.x, pos.y);
    }
  }

  // ── 渲染 ─────────────────────────────────────────────────────────────────

  /**
   * draw — 渲染所有可见 label.
   *
   * 调用 GLText.draw() 将所有字符以 instanced MSDF quad 方式提交到 GPU.
   *
   * @param params  视口/相机参数
   */
  draw(params: CellLabelDrawParams): void {
    if (!this._inited || !this.glText || this._destroyed) return;

    const zoom = params.zoom ?? 1.0;

    // LOD: zoom 过低时跳过所有 label 渲染
    if (zoom < this.opts.minZoomForLabels) return;

    const { viewportWidth, viewportHeight } = params;
    const cx = params.cameraX ?? 0;
    const cy = params.cameraY ?? 0;

    // 构建正交投影 (camera offset + zoom)
    const halfW = (viewportWidth / 2) / zoom;
    const halfH = (viewportHeight / 2) / zoom;
    const projection = makeOrthoProjection(
      cx - halfW,  cx + halfW,   // left, right
      cy - halfH,  cy + halfH,   // top, bottom
    );

    this.glText.draw(projection, this.opts.pxRange);
  }

  /**
   * drawSingleCall — 使用合并 instance buffer 的单次 draw call 模式.
   *
   * 所有 label 坐标已 baked 到 instance 数据, 不区分颜色.
   * 适用于大规模 topology（>200 cells）的极限性能场景.
   */
  drawSingleCall(params: CellLabelDrawParams): void {
    if (!this._inited || !this.glText || this._destroyed) return;

    const zoom = params.zoom ?? 1.0;
    if (zoom < this.opts.minZoomForLabels) return;

    const { viewportWidth, viewportHeight } = params;
    const cx = params.cameraX ?? 0;
    const cy = params.cameraY ?? 0;

    const halfW = (viewportWidth / 2) / zoom;
    const halfH = (viewportHeight / 2) / zoom;
    const projection = makeOrthoProjection(
      cx - halfW,  cx + halfW,
      cy - halfH,  cy + halfH,
    );

    this.glText.drawSingleCall(projection, this.opts.pxRange);
  }

  // ── 查询 ──────────────────────────────────────────────────────────────────

  /** label 总数 */
  get labelCount(): number {
    return this.labels.size;
  }

  /** 获取单个 label 记录 (调试) */
  getLabel(cellId: string): Readonly<LabelRecord> | undefined {
    return this.labels.get(cellId);
  }

  /** 列出所有 cellId */
  get cellIds(): string[] {
    return [...this.labels.keys()];
  }

  /** MSDF atlas 引用 (调试, init 后可用) */
  get msdfAtlas(): MSDFAtlas | null {
    return this.atlas;
  }

  // ── 生命周期 ──────────────────────────────────────────────────────────────

  /**
   * destroy — 释放所有 WebGL 资源.
   *
   * 调用后实例不可再使用.
   */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;

    if (this.glText) {
      this.glText.destroy();
      this.glText = null;
    }

    this.atlas = null;
    this.labels.clear();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  /**
   * 将 cell bbox 转换为 label 锚点坐标.
   *
   * 水平居中于 bbox, 垂直在 bbox 底边下方 labelOffsetY.
   */
  private _bboxToLabelPos(bbox: CellBBox): { x: number; y: number } {
    return {
      x: bbox.x + bbox.w / 2,
      y: bbox.y + bbox.h + this.opts.labelOffsetY,
    };
  }

  /**
   * 解析 hex 颜色字符串为 Color, 带 opacity 调制.
   */
  private _resolveColor(hex?: string): Color {
    const base = hex ? Color.fromHex(hex) : Color.fromHex(this.opts.defaultColor);
    if (this.opts.opacity < 1.0) {
      return new Color(base.r, base.g, base.b, base.a * this.opts.opacity);
    }
    return base;
  }

  /**
   * 构建 GLTextGeometryOptions.
   */
  private _buildGeomOpts(fontSize: number): GLTextGeometryOptions {
    return {
      fontSize,
      letterSpacing: this.opts.letterSpacing,
      align: this.opts.align,
    };
  }
}

// ── 工厂函数 ──────────────────────────────────────────────────────────────────

/**
 * createCellLabelRenderer — 一步创建并初始化 CellLabelRenderer.
 *
 * @example
 *   const labelRenderer = await createCellLabelRenderer(gl, cells, {
 *     fontSize: 11,
 *     labelOffsetY: 8,
 *   });
 *
 *   // render loop
 *   labelRenderer.draw({ viewportWidth: w, viewportHeight: h, zoom });
 *
 * @param gl     WebGL2 context
 * @param cells  初始 cell 数据 (可选, 传入则自动 syncFromCells)
 * @param opts   配置选项
 */
export async function createCellLabelRenderer(
  gl: WebGL2RenderingContext,
  cells?: CellParamsJson[],
  opts?: CellLabelRendererOptions,
): Promise<CellLabelRenderer> {
  const renderer = new CellLabelRenderer(gl, opts);
  await renderer.init();
  if (cells && cells.length > 0) {
    renderer.syncFromCells(cells);
  }
  return renderer;
}

// ── Pubsub 桥接 ──────────────────────────────────────────────────────────────

/**
 * CellLabelPubSubBridge — 将 CellEventSource 自动接线到 CellLabelRenderer.
 *
 * 监听 CellEventSource 的 cell_update 事件, 自动调用 onCellUpdate().
 * 同时处理 label 位置/文字/颜色的增量更新, 无需外部手动中转.
 *
 * @example
 *   import { getCellEventSource } from '$lib/CellEventSource';
 *
 *   const bridge = new CellLabelPubSubBridge(labelRenderer);
 *   bridge.attach(getCellEventSource());
 *
 *   // 断开:
 *   bridge.detach();
 */
export class CellLabelPubSubBridge {
  private renderer: CellLabelRenderer;
  private _listener: ((event: CellUpdateEvent) => void) | null = null;

  /** 事件源引用 (attach 后可用) */
  private _source: { addListener: (fn: any) => void; removeListener: (fn: any) => void } | null = null;

  constructor(renderer: CellLabelRenderer) {
    this.renderer = renderer;
  }

  /**
   * attach — 绑定到 CellEventSource (或任何有 addListener/removeListener 的对象).
   */
  attach(source: { addListener: (fn: (event: CellUpdateEvent) => void) => void; removeListener: (fn: (event: CellUpdateEvent) => void) => void }): void {
    this.detach();

    this._listener = (event: CellUpdateEvent) => {
      this.renderer.onCellUpdate(event);
    };

    this._source = source;
    source.addListener(this._listener);
  }

  /**
   * detach — 解除绑定, 停止监听 pubsub 事件.
   */
  detach(): void {
    if (this._listener && this._source) {
      this._source.removeListener(this._listener);
    }
    this._listener = null;
    this._source = null;
  }
}

// ── 辅助: 从 topology JSON 提取 label 数据 ──────────────────────────────────

/**
 * extractLabelsFromTopology — 从 composite_params.json 提取用于 label 渲染的数据.
 *
 * composite_params.json 的 cells 数组可能包含或不包含 label 字段.
 * 此函数过滤出带 label 的 cell, 返回 CellParamsJson 子集.
 *
 * @param topologyUrl  composite_params.json 的 URL
 * @returns 过滤后的 CellParamsJson[] (仅含有 label 的 cell)
 */
export async function extractLabelsFromTopology(
  topologyUrl: string,
): Promise<CellParamsJson[]> {
  const res = await fetch(topologyUrl);
  if (!res.ok) throw new Error(`[extractLabelsFromTopology] fetch failed: ${res.status} ${topologyUrl}`);
  const data = await res.json();

  // composite_params.json 结构: { cells: CellParamsJson[], edges: ... }
  const cells: CellParamsJson[] = data.cells ?? data;
  return cells.filter(c => !!c.label);
}
