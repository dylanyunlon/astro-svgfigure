/**
 * glui-system.ts — WebGL/PixiJS canvas-rendered UI (AT GLUI, 135 次引用)
 *
 * 架构:
 *   GLUIObject      base class — position / size / opacity / visible / interactive
 *   GLUIElement     PixiJS Graphics 画 rect / rounded-rect / circle
 *   GLUIText        PixiJS Text 在 canvas 内渲染文字
 *   GLUIBatch       批量渲染优化 — dirty-flag + 合并 draw call
 *   GLUIStage       UI 根容器 — hit testing + event dispatch
 *
 * 设计原则 (AT GLUI 同源):
 *   - 所有 UI 渲染在 WebGL canvas 内完成, 零 DOM 元素
 *   - 事件通过 PixiJS FederatedPointerEvent 系统分发
 *   - 支持 pointer-down/up/move/over/out/click 全事件链
 *   - GLUIBatch 把同层 GLUIElement 合并为单一 Graphics draw call
 *   - GLUIStage 提供 hit testing: topmost interactive 优先响应
 *
 * 对比 DOM UI:
 *   DOM             每个元素独立 layout + paint pass, z-index 复杂
 *   GLUI            全部在同一 canvas compositing pass, GPU 直接合成
 *
 * 依赖:
 *   upstream/pixijs-engine  Graphics, Text, Container, FederatedPointerEvent
 *   upstream/pixijs-ui      Button/Slider API 设计参考
 *
 * Author: dylanyunlon <dogechat@163.com>
 */

import { Container }             from '../../upstream/pixijs-engine/src/scene/container/Container';
import { Graphics }              from '../../upstream/pixijs-engine/src/scene/graphics/shared/Graphics';
import { Text }                  from '../../upstream/pixijs-engine/src/scene/text/Text';
import { TextStyle }             from '../../upstream/pixijs-engine/src/scene/text/TextStyle';
import type { FederatedPointerEvent } from '../../upstream/pixijs-engine/src/events/FederatedPointerEvent';

// ── 公共类型 ──────────────────────────────────────────────────────────────────

/** 2D 位置 */
export interface GLUIPoint {
  x: number;
  y: number;
}

/** 2D 尺寸 */
export interface GLUISize {
  width:  number;
  height: number;
}

/** RGBA 颜色 (0xRRGGBB, alpha 0-1) */
export interface GLUIColor {
  hex:   number;
  alpha: number;
}

/** 事件回调签名 (对齐 pixijs-ui Button.onPress) */
export type GLUIPointerHandler = (obj: GLUIObject, e: FederatedPointerEvent) => void;

/** 支持的事件类型 */
export type GLUIEventType =
  | 'pointerdown'
  | 'pointerup'
  | 'pointermove'
  | 'pointerover'
  | 'pointerout'
  | 'click';

// ── GLUIObject — base class ───────────────────────────────────────────────────

/**
 * GLUIObject — 所有 GLUI 组件的基类.
 *
 * 对应 AT `GLUIObject` (135 次引用的根节点).
 * 持有 PixiJS Container 作为场景树节点,
 * 统一管理 position / size / opacity / visible / interactive 五个核心属性.
 *
 * @example
 *   // 子类继承
 *   class MyWidget extends GLUIObject {
 *     protected _container: Container;
 *     constructor() {
 *       super();
 *       this._container = new Container();
 *     }
 *     get view() { return this._container; }
 *   }
 */
export abstract class GLUIObject {
  /** 唯一标识符 (AT GLUIObject.id) */
  readonly id: string;

  /** 内部 PixiJS 场景节点 */
  protected abstract readonly _container: Container;

  /** 位置 (px, 相对父容器) */
  protected _x = 0;
  protected _y = 0;

  /** 尺寸 (px) */
  protected _width  = 0;
  protected _height = 0;

  /** 透明度 0-1 */
  protected _opacity = 1;

  /** 是否可见 */
  protected _visible = true;

  /** 是否响应指针事件 (对齐 pixijs-ui Button.enabled) */
  protected _interactive = false;

  /** 事件监听器 Map */
  private readonly _listeners = new Map<GLUIEventType, Set<GLUIPointerHandler>>();

  /** 父节点引用 (由 GLUIStage 或 GLUIObject.addChild 写入) */
  _parent: GLUIObject | null = null;

  constructor(id?: string) {
    this.id = id ?? `glui-${Math.random().toString(36).slice(2, 9)}`;
  }

  // ── 场景树代理 ──────────────────────────────────────────────────────────────

  /** 返回底层 PixiJS Container (供 GLUIStage/GLUIBatch 使用) */
  abstract get view(): Container;

  // ── position ────────────────────────────────────────────────────────────────

  get x(): number { return this._x; }
  set x(v: number) {
    this._x = v;
    this._container.x = v;
  }

  get y(): number { return this._y; }
  set y(v: number) {
    this._y = v;
    this._container.y = v;
  }

  /** 一次性设置 x/y (对齐 Container.position.set) */
  setPosition(x: number, y: number): this {
    this.x = x;
    this.y = y;
    return this;
  }

  // ── size ────────────────────────────────────────────────────────────────────

  get width(): number  { return this._width; }
  get height(): number { return this._height; }

  /** 调整尺寸 (子类可 override 触发重绘) */
  setSize(width: number, height: number): this {
    this._width  = width;
    this._height = height;
    return this;
  }

  // ── opacity ─────────────────────────────────────────────────────────────────

  get opacity(): number { return this._opacity; }
  set opacity(v: number) {
    this._opacity = Math.max(0, Math.min(1, v));
    this._container.alpha = this._opacity;
  }

  // ── visible ─────────────────────────────────────────────────────────────────

  get visible(): boolean { return this._visible; }
  set visible(v: boolean) {
    this._visible = v;
    this._container.visible = v;
  }

  /** 显示/隐藏便捷方法 (对齐 AT GLUIObject.show/hide) */
  show(): this { this.visible = true;  return this; }
  hide(): this { this.visible = false; return this; }

  // ── interactive ─────────────────────────────────────────────────────────────

  get interactive(): boolean { return this._interactive; }
  set interactive(v: boolean) {
    this._interactive = v;
    // 对齐 pixijs-ui Button.enabled 的 eventMode 模式
    this._container.eventMode = v ? 'static' : 'passive';
    this._container.cursor    = v ? 'pointer' : 'default';
  }

  // ── 事件系统 (对齐 pixijs-ui Button.onPress / Signal pattern) ───────────────

  /**
   * 添加事件监听器.
   *
   * @example
   *   btn.on('click', (obj, e) => console.log('clicked', obj.id));
   */
  on(type: GLUIEventType, handler: GLUIPointerHandler): this {
    if (!this._listeners.has(type)) {
      this._listeners.set(type, new Set());
    }
    this._listeners.get(type)!.add(handler);

    // 同步到 PixiJS 事件系统 (当 interactive=true 时生效)
    if (this._interactive) {
      this._container.on(type as any, (e: FederatedPointerEvent) =>
        this._dispatch(type, e),
      );
    }

    return this;
  }

  /** 移除事件监听器 */
  off(type: GLUIEventType, handler: GLUIPointerHandler): this {
    this._listeners.get(type)?.delete(handler);
    return this;
  }

  /** 内部分发 (供 GLUIStage hit-testing 调用) */
  _dispatch(type: GLUIEventType, e: FederatedPointerEvent): void {
    const handlers = this._listeners.get(type);
    if (!handlers) return;
    for (const h of handlers) h(this, e);
  }

  // ── 子节点管理 ──────────────────────────────────────────────────────────────

  /**
   * 添加子 GLUIObject.
   * 代理到底层 Container.addChild, 同时维护 _parent 链.
   */
  addChild(child: GLUIObject): this {
    this._container.addChild(child.view);
    child._parent = this;
    return this;
  }

  removeChild(child: GLUIObject): this {
    this._container.removeChild(child.view);
    child._parent = null;
    return this;
  }

  // ── 销毁 ────────────────────────────────────────────────────────────────────

  destroy(): void {
    this._listeners.clear();
    this._container.destroy({ children: true });
  }
}

// ── GLUIElement — PixiJS Graphics 绘制基本形状 ────────────────────────────────

/**
 * GLUIElement — 用 PixiJS Graphics 在 canvas 内绘制 UI 形状.
 *
 * 支持 rect / rounded-rect / circle 三种基元,
 * 提供 fill + stroke 独立控制.
 *
 * 对应 AT GLUIElement 的 shape-based rendering 分支.
 *
 * @example
 *   const panel = new GLUIElement('panel');
 *   panel
 *     .setPosition(10, 10)
 *     .setSize(200, 100)
 *     .drawRoundedRect({ hex: 0x1a1a2e, alpha: 0.9 }, 12)
 *     .setStroke({ hex: 0x7986CB, alpha: 1 }, 1.5);
 */
export class GLUIElement extends GLUIObject {
  protected readonly _container: Container;
  private readonly _graphics:  Graphics;

  /** 最后绘制的形状类型 */
  private _shapeKind: 'rect' | 'roundedRect' | 'circle' | null = null;

  /** 当前 fill 颜色 */
  private _fill:   GLUIColor | null = null;
  /** 当前 stroke 颜色 + 粗细 */
  private _stroke: (GLUIColor & { lineWidth: number }) | null = null;
  /** rounded-rect 圆角半径 */
  private _radius = 0;

  constructor(id?: string) {
    super(id);
    this._graphics  = new Graphics();
    this._container = new Container();
    this._container.addChild(this._graphics);
  }

  override get view(): Container {
    return this._container;
  }

  // ── setSize override — 触发重绘 ─────────────────────────────────────────────

  override setSize(width: number, height: number): this {
    super.setSize(width, height);
    this._redraw();
    return this;
  }

  // ── 形状绘制 API ─────────────────────────────────────────────────────────────

  /**
   * 绘制矩形 (AT GLUIElement.drawRect).
   *
   * @param fill    填充颜色
   */
  drawRect(fill: GLUIColor): this {
    this._shapeKind = 'rect';
    this._fill = fill;
    this._redraw();
    return this;
  }

  /**
   * 绘制圆角矩形 (AT GLUIElement.drawRoundedRect).
   *
   * @param fill    填充颜色
   * @param radius  圆角半径 (px), 默认 8
   */
  drawRoundedRect(fill: GLUIColor, radius = 8): this {
    this._shapeKind = 'roundedRect';
    this._fill   = fill;
    this._radius = radius;
    this._redraw();
    return this;
  }

  /**
   * 绘制圆 (AT GLUIElement.drawCircle).
   * 圆心为 (width/2, height/2), 半径为 min(w,h)/2.
   *
   * @param fill  填充颜色
   */
  drawCircle(fill: GLUIColor): this {
    this._shapeKind = 'circle';
    this._fill = fill;
    this._redraw();
    return this;
  }

  /**
   * 设置描边 (对齐 PixiJS Graphics.stroke).
   *
   * @param color     描边颜色
   * @param lineWidth 线宽 (px)
   */
  setStroke(color: GLUIColor, lineWidth = 1): this {
    this._stroke = { ...color, lineWidth };
    this._redraw();
    return this;
  }

  /** 清除描边 */
  clearStroke(): this {
    this._stroke = null;
    this._redraw();
    return this;
  }

  // ── 内部重绘 ─────────────────────────────────────────────────────────────────

  private _redraw(): void {
    if (!this._shapeKind || !this._fill) return;

    const g = this._graphics;
    const { _width: w, _height: h } = this;

    g.clear();

    // fill
    g.fillStyle = { color: this._fill.hex, alpha: this._fill.alpha };

    // stroke (PixiJS v8 stroke API)
    if (this._stroke) {
      g.strokeStyle = {
        color: this._stroke.hex,
        alpha: this._stroke.alpha,
        width: this._stroke.lineWidth,
      };
    }

    switch (this._shapeKind) {
      case 'rect':
        this._stroke
          ? g.rect(0, 0, w, h).fill().stroke()
          : g.rect(0, 0, w, h).fill();
        break;

      case 'roundedRect':
        this._stroke
          ? g.roundRect(0, 0, w, h, this._radius).fill().stroke()
          : g.roundRect(0, 0, w, h, this._radius).fill();
        break;

      case 'circle': {
        const r  = Math.min(w, h) / 2;
        const cx = w / 2;
        const cy = h / 2;
        this._stroke
          ? g.circle(cx, cy, r).fill().stroke()
          : g.circle(cx, cy, r).fill();
        break;
      }
    }
  }
}

// ── GLUIText — PixiJS Text canvas 渲染 ────────────────────────────────────────

/**
 * GLUIText — 用 PixiJS Text 在 canvas 内渲染文字.
 *
 * 使用 PixiJS BitmapText/Text (GPU-accelerated canvas 字体),
 * 与 DOM 文字完全隔离.
 *
 * 对应 AT GLUIText (差异: AT 用 MSDF, 此处用 PixiJS Text 集成路径).
 *
 * @example
 *   const label = new GLUIText('label');
 *   label
 *     .setPosition(8, 8)
 *     .setText('Hello GLUI', {
 *       fontFamily: 'Inter',
 *       fontSize:   14,
 *       fill:       0xffffff,
 *       align:      'left',
 *     });
 */

export interface GLUITextOptions {
  fontFamily?: string;
  fontSize?:   number;
  fontWeight?: string;
  fill?:       number;
  alpha?:      number;
  align?:      'left' | 'center' | 'right';
  wordWrap?:   boolean;
  wordWrapWidth?: number;
  lineHeight?: number;
  letterSpacing?: number;
}

export class GLUIText extends GLUIObject {
  protected readonly _container: Container;
  private _textNode:  Text;
  private _content = '';

  constructor(id?: string) {
    super(id);
    this._textNode  = new Text({ text: '' });
    this._container = new Container();
    this._container.addChild(this._textNode);
  }

  override get view(): Container {
    return this._container;
  }

  // ── 文字内容 ─────────────────────────────────────────────────────────────────

  /** 获取当前文字内容 */
  get text(): string { return this._content; }

  /**
   * 设置文字内容 + 样式 (AT GLUIText.setText).
   *
   * @param content  显示文字
   * @param opts     样式覆盖 (可选, 只需传差异项)
   */
  setText(content: string, opts?: GLUITextOptions): this {
    this._content = content;
    this._textNode.text = content;
    if (opts) this.setStyle(opts);
    return this;
  }

  /**
   * 仅更新样式, 不更改文字 (对齐 PixiJS Text.style setter).
   */
  setStyle(opts: GLUITextOptions): this {
    const style: Partial<TextStyle> = {};

    if (opts.fontFamily   !== undefined) style.fontFamily    = opts.fontFamily;
    if (opts.fontSize     !== undefined) style.fontSize      = opts.fontSize;
    if (opts.fontWeight   !== undefined) style.fontWeight    = opts.fontWeight;
    if (opts.fill         !== undefined) style.fill          = opts.fill;
    if (opts.align        !== undefined) style.align         = opts.align;
    if (opts.wordWrap     !== undefined) style.wordWrap      = opts.wordWrap;
    if (opts.wordWrapWidth !== undefined) style.wordWrapWidth = opts.wordWrapWidth;
    if (opts.lineHeight   !== undefined) style.lineHeight    = opts.lineHeight;
    if (opts.letterSpacing !== undefined) style.letterSpacing = opts.letterSpacing;

    Object.assign(this._textNode.style, style);

    if (opts.alpha !== undefined) {
      this._textNode.alpha = opts.alpha;
    }

    return this;
  }

  /**
   * 测量文字包围盒 (px).
   * 返回 PixiJS 渲染后的实际宽高.
   */
  measureText(): GLUISize {
    return {
      width:  this._textNode.width,
      height: this._textNode.height,
    };
  }
}

// ── GLUIBatch — 批量渲染优化 ──────────────────────────────────────────────────

/**
 * GLUIBatch — 把同层多个 GLUIElement 合并为单一 Graphics pass.
 *
 * AT GLUIBatch 原理:
 *   1. 收集所有子 GLUIElement 的绘制指令
 *   2. 合并进一个 Graphics 实例 → 单 draw call
 *   3. 通过 dirty-flag 避免每帧重建 (仅在子元素变更时重建)
 *
 * 使用场景: 大量静态背景面板 (如 cell 背景 300+ 个 card)
 *
 * @example
 *   const batch = new GLUIBatch('bg-cards');
 *   for (const cell of cells) {
 *     batch.addRect(cell.x, cell.y, cell.w, cell.h, { hex: 0x1a1a2e, alpha: 0.8 }, 8);
 *   }
 *   stage.addObject(batch);
 *   batch.flush(); // 合并并提交 GPU
 */

export interface BatchRectEntry {
  x:      number;
  y:      number;
  width:  number;
  height: number;
  fill:   GLUIColor;
  radius: number;
  stroke?: GLUIColor & { lineWidth: number };
}

export interface BatchCircleEntry {
  cx:     number;
  cy:     number;
  radius: number;
  fill:   GLUIColor;
  stroke?: GLUIColor & { lineWidth: number };
}

export type BatchEntry =
  | ({ kind: 'rect'   } & BatchRectEntry)
  | ({ kind: 'circle' } & BatchCircleEntry);

export class GLUIBatch extends GLUIObject {
  protected readonly _container: Container;
  private readonly _graphics: Graphics;

  private _entries: BatchEntry[] = [];
  private _dirty   = false;

  /** 已提交的 entry 数量 (用于统计 draw call 节省量) */
  private _lastFlushCount = 0;

  constructor(id?: string) {
    super(id);
    this._graphics  = new Graphics();
    this._container = new Container();
    this._container.addChild(this._graphics);
  }

  override get view(): Container {
    return this._container;
  }

  // ── 批量录入 API ─────────────────────────────────────────────────────────────

  /**
   * 录入一个矩形绘制指令.
   * 不立即绘制, 在 flush() 时统一提交.
   */
  addRect(
    x: number, y: number, width: number, height: number,
    fill: GLUIColor,
    radius = 0,
    stroke?: GLUIColor & { lineWidth: number },
  ): this {
    this._entries.push({ kind: 'rect', x, y, width, height, fill, radius, stroke });
    this._dirty = true;
    return this;
  }

  /**
   * 录入一个圆形绘制指令.
   */
  addCircle(
    cx: number, cy: number, radius: number,
    fill: GLUIColor,
    stroke?: GLUIColor & { lineWidth: number },
  ): this {
    this._entries.push({ kind: 'circle', cx, cy, radius, fill, stroke });
    this._dirty = true;
    return this;
  }

  /** 清空所有录入 (不重绘) */
  clearEntries(): this {
    this._entries = [];
    this._dirty   = true;
    return this;
  }

  /**
   * flush — 把所有录入合并为单一 Graphics draw call.
   *
   * AT GLUIBatch.flush() 对应.
   * 仅在 dirty 时执行实际 GPU 提交.
   */
  flush(): this {
    if (!this._dirty) return this;

    const g = this._graphics;
    g.clear();

    for (const entry of this._entries) {
      if (entry.kind === 'rect') {
        g.fillStyle = { color: entry.fill.hex, alpha: entry.fill.alpha };
        if (entry.stroke) {
          g.strokeStyle = {
            color: entry.stroke.hex,
            alpha: entry.stroke.alpha,
            width: entry.stroke.lineWidth,
          };
        }
        if (entry.radius > 0) {
          entry.stroke
            ? g.roundRect(entry.x, entry.y, entry.width, entry.height, entry.radius).fill().stroke()
            : g.roundRect(entry.x, entry.y, entry.width, entry.height, entry.radius).fill();
        } else {
          entry.stroke
            ? g.rect(entry.x, entry.y, entry.width, entry.height).fill().stroke()
            : g.rect(entry.x, entry.y, entry.width, entry.height).fill();
        }
      } else {
        // circle
        g.fillStyle = { color: entry.fill.hex, alpha: entry.fill.alpha };
        if (entry.stroke) {
          g.strokeStyle = {
            color: entry.stroke.hex,
            alpha: entry.stroke.alpha,
            width: entry.stroke.lineWidth,
          };
        }
        entry.stroke
          ? g.circle(entry.cx, entry.cy, entry.radius).fill().stroke()
          : g.circle(entry.cx, entry.cy, entry.radius).fill();
      }
    }

    this._lastFlushCount = this._entries.length;
    this._dirty = false;
    return this;
  }

  /** 返回最后一次 flush 的 entry 数量 (= 节省的 draw calls - 1) */
  get flushedCount(): number { return this._lastFlushCount; }

  /** dirty 状态 (供外部调度器查询) */
  get dirty(): boolean { return this._dirty; }
}

// ── GLUIStage — UI 根容器 + hit testing + event dispatch ──────────────────────

/**
 * GLUIStage — GLUI 场景树的根容器.
 *
 * 职责:
 *   1. 持有所有 GLUIObject 的 PixiJS scene graph
 *   2. hit testing: 从顶层 (z最大) 向底层遍历，找到第一个 interactive 对象
 *   3. event dispatch: 把 PixiJS FederatedPointerEvent 路由到命中对象
 *   4. resize: 跟随 canvas 尺寸更新投影
 *
 * 对应 AT GLUIStage (差异: AT 直接操作 WebGL context,
 *                           此处委托 PixiJS Application 管理)
 *
 * @example
 *   const app   = new Application();
 *   await app.init({ width: 800, height: 600 });
 *
 *   const stage = new GLUIStage('root', app.stage, 800, 600);
 *
 *   const panel = new GLUIElement('panel');
 *   panel.drawRoundedRect({ hex: 0x1a1a2e, alpha: 0.9 }, 12)
 *        .setPosition(50, 50).setSize(200, 100);
 *   panel.interactive = true;
 *   panel.on('click', (obj, e) => console.log('panel clicked'));
 *
 *   stage.addObject(panel);
 */
export class GLUIStage {
  /** 场景根节点 (挂载到 PixiJS Application.stage) */
  readonly root: Container;

  /** 舞台宽度 (px) */
  private _stageWidth:  number;
  /** 舞台高度 (px) */
  private _stageHeight: number;

  /**
   * 注册的 GLUIObject 列表 (按 z-index 升序, 最后添加的在最顶层).
   * hit testing 从末尾向前遍历.
   */
  private _objects: GLUIObject[] = [];

  /** 当前 hover 对象 (用于 pointerout 触发) */
  private _hoverTarget: GLUIObject | null = null;

  constructor(id: string, pixiStageRoot: Container, width: number, height: number) {
    this._stageWidth  = width;
    this._stageHeight = height;

    // 创建 GLUI 专用子容器 (隔离 z-order)
    this.root = new Container();
    this.root.label = `GLUIStage:${id}`;
    pixiStageRoot.addChild(this.root);

    // 注册 stage-level 指针事件 (捕获阶段)
    this.root.eventMode = 'static';
    this.root.hitArea   = { contains: () => true } as any;

    this.root.on('pointermove',  (e: FederatedPointerEvent) => this._onPointerMove(e));
    this.root.on('pointerdown',  (e: FederatedPointerEvent) => this._onPointerDown(e));
    this.root.on('pointerup',    (e: FederatedPointerEvent) => this._onPointerUp(e));
    this.root.on('pointerupoutside', (e: FederatedPointerEvent) => this._onPointerUp(e));
  }

  // ── 对象管理 ─────────────────────────────────────────────────────────────────

  /**
   * 添加 GLUIObject 到舞台.
   * 后添加的对象渲染在上层 (z 更高).
   */
  addObject(obj: GLUIObject): this {
    this._objects.push(obj);
    this.root.addChild(obj.view);
    obj._parent = null; // stage is conceptual root
    return this;
  }

  /** 移除 GLUIObject */
  removeObject(obj: GLUIObject): this {
    const idx = this._objects.indexOf(obj);
    if (idx >= 0) {
      this._objects.splice(idx, 1);
      this.root.removeChild(obj.view);
    }
    return this;
  }

  /** 按 id 查找 GLUIObject */
  findById(id: string): GLUIObject | undefined {
    return this._objects.find(o => o.id === id);
  }

  /** 所有已注册对象的快照 */
  get objects(): readonly GLUIObject[] {
    return this._objects;
  }

  // ── resize ──────────────────────────────────────────────────────────────────

  /**
   * 更新舞台尺寸 (canvas resize 时调用).
   * 对齐 AT GLUIStage.resize().
   */
  resize(width: number, height: number): void {
    this._stageWidth  = width;
    this._stageHeight = height;
    // hitArea 跟随更新
    this.root.hitArea = {
      contains: (px: number, py: number) =>
        px >= 0 && py >= 0 && px <= width && py <= height,
    } as any;
  }

  get stageWidth():  number { return this._stageWidth; }
  get stageHeight(): number { return this._stageHeight; }

  // ── hit testing ─────────────────────────────────────────────────────────────

  /**
   * hitTest(x, y) — 在所有 interactive 对象中找到最顶层命中者.
   *
   * AT GLUIStage.hitTest() 对应.
   * 遍历顺序: 从 z 最高 (列表末尾) 到 z 最低 (列表开头).
   *
   * @returns 命中的 GLUIObject, 未命中返回 null
   */
  hitTest(x: number, y: number): GLUIObject | null {
    for (let i = this._objects.length - 1; i >= 0; i--) {
      const obj = this._objects[i];
      if (!obj.interactive || !obj.visible) continue;

      // 简单 AABB hit test (世界坐标)
      const ox = obj.x;
      const oy = obj.y;
      if (
        x >= ox &&
        x <= ox + obj.width &&
        y >= oy &&
        y <= oy + obj.height
      ) {
        return obj;
      }
    }
    return null;
  }

  // ── 内部事件路由 ─────────────────────────────────────────────────────────────

  private _onPointerMove(e: FederatedPointerEvent): void {
    const { x, y } = e.global;
    const hit = this.hitTest(x, y);

    // pointerout / pointerover 转换
    if (hit !== this._hoverTarget) {
      if (this._hoverTarget) {
        this._hoverTarget._dispatch('pointerout', e);
      }
      if (hit) {
        hit._dispatch('pointerover', e);
      }
      this._hoverTarget = hit;
    }

    // pointermove 传给命中目标
    if (hit) {
      hit._dispatch('pointermove', e);
    }
  }

  private _onPointerDown(e: FederatedPointerEvent): void {
    const { x, y } = e.global;
    const hit = this.hitTest(x, y);
    if (hit) {
      hit._dispatch('pointerdown', e);
    }
  }

  private _onPointerUp(e: FederatedPointerEvent): void {
    const { x, y } = e.global;
    const hit = this.hitTest(x, y);
    if (hit) {
      hit._dispatch('pointerup', e);
      hit._dispatch('click',     e);
    }
  }

  // ── 调试工具 ─────────────────────────────────────────────────────────────────

  /**
   * debugDrawHitAreas — 在所有 interactive 对象下方绘制半透明命中区域轮廓.
   * 仅用于开发调试, 生产构建应移除.
   */
  debugDrawHitAreas(g: Graphics): void {
    g.clear();
    for (const obj of this._objects) {
      if (!obj.interactive) continue;
      g.strokeStyle = { color: 0x00ff88, alpha: 0.6, width: 1 };
      g.rect(obj.x, obj.y, obj.width, obj.height).stroke();
    }
  }

  // ── 销毁 ────────────────────────────────────────────────────────────────────

  destroy(): void {
    for (const obj of this._objects) obj.destroy();
    this._objects = [];
    this._hoverTarget = null;
    this.root.destroy({ children: true });
  }
}

// ── 工厂函数 (便捷 API, 对齐 pixijs-ui FancyButton / Slider 风格) ────────────

/**
 * createGLUIButton — 创建一个可点击的圆角矩形按钮.
 *
 * 对齐 upstream/pixijs-ui Button + ButtonContainer API.
 *
 * @example
 *   const btn = createGLUIButton('save-btn', 'Save', {
 *     width:   120,
 *     height:  36,
 *     bgColor: { hex: 0x5C6BC0, alpha: 1 },
 *     radius:  8,
 *   });
 *   btn.on('click', () => handleSave());
 *   stage.addObject(btn);
 */
export interface GLUIButtonOptions {
  width?:    number;
  height?:   number;
  bgColor?:  GLUIColor;
  textColor?: number;
  fontSize?:  number;
  radius?:    number;
  disabled?:  boolean;
}

export function createGLUIButton(
  id:    string,
  label: string,
  opts:  GLUIButtonOptions = {},
): GLUIElement {
  const w  = opts.width    ?? 120;
  const h  = opts.height   ?? 36;
  const bg = opts.bgColor  ?? { hex: 0x5C6BC0, alpha: 1 };
  const r  = opts.radius   ?? 8;

  const btn = new GLUIElement(id);
  btn.setSize(w, h).drawRoundedRect(bg, r);
  btn.interactive = !(opts.disabled ?? false);

  // 悬停高亮 (对齐 pixijs-ui FancyButton hover state)
  btn.on('pointerover', (obj) => {
    (obj as GLUIElement).drawRoundedRect(
      { hex: lighten(bg.hex, 0.15), alpha: bg.alpha }, r,
    );
  });
  btn.on('pointerout', (obj) => {
    (obj as GLUIElement).drawRoundedRect(bg, r);
  });
  btn.on('pointerdown', (obj) => {
    (obj as GLUIElement).drawRoundedRect(
      { hex: darken(bg.hex, 0.15), alpha: bg.alpha }, r,
    );
  });
  btn.on('pointerup', (obj) => {
    (obj as GLUIElement).drawRoundedRect(bg, r);
  });

  // 嵌入文字标签
  const txt = new GLUIText(`${id}-label`);
  txt.setText(label, {
    fontFamily: 'Inter, system-ui, sans-serif',
    fontSize:   opts.fontSize  ?? 13,
    fill:       opts.textColor ?? 0xffffff,
    align:      'center',
  });

  // 文字居中 (近似值, 完整版应 measureText 后计算)
  txt.setPosition(w / 2 - 30, h / 2 - 8);
  btn.addChild(txt);

  return btn;
}

/**
 * createGLUISlider — 创建水平滑块.
 *
 * 对齐 upstream/pixijs-ui SliderBase API (min/max/value/onChange).
 *
 * @example
 *   const slider = createGLUISlider('opacity-slider', {
 *     min: 0, max: 1, value: 0.8,
 *     onChange: (v) => setOpacity(v),
 *   });
 *   stage.addObject(slider);
 */
export interface GLUISliderOptions {
  width?:     number;
  height?:    number;
  min?:       number;
  max?:       number;
  value?:     number;
  trackColor?: GLUIColor;
  fillColor?:  GLUIColor;
  thumbColor?: GLUIColor;
  onChange?:  (value: number) => void;
}

export interface GLUISliderHandle {
  /** 对应 SliderBase.value */
  value: number;
  /** 读取当前 normalized value (0-1) */
  normalized: number;
  /** 销毁 */
  destroy(): void;
}

export function createGLUISlider(
  id:   string,
  opts: GLUISliderOptions = {},
): { track: GLUIElement; thumb: GLUIElement; handle: GLUISliderHandle } {
  const W    = opts.width  ?? 200;
  const H    = opts.height ?? 6;
  const min  = opts.min    ?? 0;
  const max  = opts.max    ?? 1;
  const init = Math.max(min, Math.min(max, opts.value ?? min));

  const trackColor = opts.trackColor ?? { hex: 0x37474F, alpha: 1 };
  const fillColor  = opts.fillColor  ?? { hex: 0x5C6BC0, alpha: 1 };
  const thumbColor = opts.thumbColor ?? { hex: 0xffffff, alpha: 1 };

  const THUMB_R = 8;

  // 轨道背景
  const track = new GLUIElement(`${id}-track`);
  track.setSize(W, H).drawRoundedRect(trackColor, H / 2);

  // 填充 (左侧进度)
  const fill = new GLUIElement(`${id}-fill`);
  fill.setSize(1, H).drawRoundedRect(fillColor, H / 2);
  track.addChild(fill);

  // 拇指滑块
  const thumb = new GLUIElement(`${id}-thumb`);
  thumb.setSize(THUMB_R * 2, THUMB_R * 2)
       .drawCircle(thumbColor)
       .setPosition(-THUMB_R, H / 2 - THUMB_R);
  thumb.interactive = true;
  track.addChild(thumb);

  // 轨道本身也可点击跳转
  track.interactive = true;

  // 状态
  let currentValue = init;
  let dragging     = false;

  function normalize(v: number): number {
    return (v - min) / (max - min);
  }

  function updateVisual(v: number): void {
    const pct = normalize(v);
    const px  = pct * W;
    fill.setSize(Math.max(0, px), H);
    thumb.setPosition(px - THUMB_R, H / 2 - THUMB_R);
  }

  function applyValue(v: number): void {
    currentValue = Math.max(min, Math.min(max, v));
    updateVisual(currentValue);
    opts.onChange?.(currentValue);
  }

  // 点击轨道直接跳转 (对齐 SliderBase click behavior)
  track.on('pointerdown', (_obj, e) => {
    dragging = true;
    const localX = e.global.x - track.x;
    applyValue(min + (localX / W) * (max - min));
  });

  // 拇指拖拽 (对齐 SliderBase dragging logic)
  thumb.on('pointerdown', (_obj, _e) => { dragging = true; });

  // pointermove 在 track 上监听拖拽 (GLUIStage 会路由过来)
  track.on('pointermove', (_obj, e) => {
    if (!dragging) return;
    const localX = e.global.x - track.x;
    applyValue(min + (localX / W) * (max - min));
  });

  track.on('pointerup',  () => { dragging = false; });
  thumb.on('pointerup',  () => { dragging = false; });
  track.on('pointerout', () => { dragging = false; });

  // 初始化位置
  updateVisual(init);

  const handle: GLUISliderHandle = {
    get value()      { return currentValue; },
    get normalized() { return normalize(currentValue); },
    destroy()        { track.destroy(); },
  };

  return { track, thumb, handle };
}

// ── 颜色辅助 (内部) ───────────────────────────────────────────────────────────

/** 亮化 hex 颜色 (factor 0-1) */
function lighten(hex: number, factor: number): number {
  const r = Math.min(255, ((hex >> 16) & 0xff) + Math.round(255 * factor));
  const g = Math.min(255, ((hex >>  8) & 0xff) + Math.round(255 * factor));
  const b = Math.min(255, ( hex        & 0xff) + Math.round(255 * factor));
  return (r << 16) | (g << 8) | b;
}

/** 暗化 hex 颜色 (factor 0-1) */
function darken(hex: number, factor: number): number {
  const r = Math.max(0, ((hex >> 16) & 0xff) - Math.round(255 * factor));
  const g = Math.max(0, ((hex >>  8) & 0xff) - Math.round(255 * factor));
  const b = Math.max(0, ( hex        & 0xff) - Math.round(255 * factor));
  return (r << 16) | (g << 8) | b;
}
