// interaction.ts — Drag & Drop, Scroll, Keyboard, ContextMenu, UserInput normalization

// ---------------------------------------------------------------------------
// Shared point type
// ---------------------------------------------------------------------------

export interface Point {
  x: number;
  y: number;
}

// ---------------------------------------------------------------------------
// DragAndDrop
// ---------------------------------------------------------------------------

export type DragEventCallback = (point: Point, event: PointerEvent) => void;

export interface DragAndDropOptions {
  threshold?: number; // px before drag is recognized
  onDragStart?: DragEventCallback;
  onDrag?: DragEventCallback;
  onDragEnd?: DragEventCallback;
}

export class DragAndDrop {
  private _el: HTMLElement;
  private _opts: Required<DragAndDropOptions>;
  private _enabled = false;
  private _dragging = false;
  private _startPoint: Point = { x: 0, y: 0 };
  private _handlers: Record<string, EventListener> = {};

  constructor(el: HTMLElement, opts: DragAndDropOptions = {}) {
    this._el = el;
    this._opts = {
      threshold: opts.threshold ?? 4,
      onDragStart: opts.onDragStart ?? (() => {}),
      onDrag: opts.onDrag ?? (() => {}),
      onDragEnd: opts.onDragEnd ?? (() => {}),
    };
  }

  enable(): void {
    if (this._enabled) return;
    this._enabled = true;

    const el = this._el;

    this._handlers.pointerdown = ((e: PointerEvent) => {
      this._dragging = false;
      this._startPoint = { x: e.clientX, y: e.clientY };
      el.setPointerCapture(e.pointerId);
    }) as EventListener;

    this._handlers.pointermove = ((e: PointerEvent) => {
      if (!el.hasPointerCapture(e.pointerId)) return;
      const pt: Point = { x: e.clientX, y: e.clientY };
      const dx = pt.x - this._startPoint.x;
      const dy = pt.y - this._startPoint.y;

      if (!this._dragging && Math.hypot(dx, dy) >= this._opts.threshold) {
        this._dragging = true;
        this.onDragStart(this._startPoint, e as PointerEvent);
      }
      if (this._dragging) {
        this.onDrag(pt, e as PointerEvent);
      }
    }) as EventListener;

    this._handlers.pointerup = ((e: PointerEvent) => {
      if (this._dragging) {
        this.onDragEnd({ x: e.clientX, y: e.clientY }, e as PointerEvent);
        this._dragging = false;
      }
    }) as EventListener;

    for (const [evt, handler] of Object.entries(this._handlers)) {
      el.addEventListener(evt, handler);
    }
  }

  disable(): void {
    if (!this._enabled) return;
    this._enabled = false;
    for (const [evt, handler] of Object.entries(this._handlers)) {
      this._el.removeEventListener(evt, handler);
    }
    this._handlers = {};
  }

  onDragStart(pt: Point, e: PointerEvent): void {
    this._opts.onDragStart(pt, e);
  }

  onDrag(pt: Point, e: PointerEvent): void {
    this._opts.onDrag(pt, e);
  }

  onDragEnd(pt: Point, e: PointerEvent): void {
    this._opts.onDragEnd(pt, e);
  }

  get isDragging(): boolean { return this._dragging; }
  get isEnabled(): boolean { return this._enabled; }
}

// ---------------------------------------------------------------------------
// ScrollController  (inertia + rubber-band)
// ---------------------------------------------------------------------------

export interface ScrollControllerOptions {
  friction?: number;       // 0-1, default 0.92
  rubberBand?: number;     // elasticity factor, default 0.3
  minBound?: number;
  maxBound?: number;
}

export class ScrollController {
  private _pos = 0;
  private _vel = 0;
  private _friction: number;
  private _rubberBand: number;
  private _minBound: number;
  private _maxBound: number;
  private _rafId = 0;
  private _running = false;

  constructor(opts: ScrollControllerOptions = {}) {
    this._friction = opts.friction ?? 0.92;
    this._rubberBand = opts.rubberBand ?? 0.3;
    this._minBound = opts.minBound ?? 0;
    this._maxBound = opts.maxBound ?? Infinity;
  }

  /** Add velocity (e.g. from a flick gesture). */
  addVelocity(v: number): void {
    this._vel += v;
    this._ensureRunning();
  }

  /** Jump to a position with no animation. */
  setPosition(pos: number): void {
    this._pos = pos;
    this._vel = 0;
  }

  /** Scroll by delta immediately (for wheel events). */
  scrollBy(delta: number): void {
    this._vel += delta;
    this._ensureRunning();
  }

  get position(): number { return this._pos; }
  get velocity(): number { return this._vel; }

  private _ensureRunning(): void {
    if (this._running) return;
    this._running = true;
    this._tick();
  }

  private _tick = (): void => {
    // Rubber-band pull-back when out of bounds
    if (this._pos < this._minBound) {
      const over = this._minBound - this._pos;
      this._vel += over * this._rubberBand;
    } else if (this._pos > this._maxBound) {
      const over = this._pos - this._maxBound;
      this._vel -= over * this._rubberBand;
    }

    this._vel *= this._friction;
    this._pos += this._vel;

    if (Math.abs(this._vel) < 0.01) {
      // Snap to bounds
      this._pos = Math.min(Math.max(this._pos, this._minBound), this._maxBound);
      this._vel = 0;
      this._running = false;
      return;
    }

    this._rafId = requestAnimationFrame(this._tick);
  };

  stop(): void {
    cancelAnimationFrame(this._rafId);
    this._running = false;
    this._vel = 0;
  }

  destroy(): void {
    this.stop();
  }
}

// ---------------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------------

export type KeyCombo = string; // e.g. "ctrl+shift+z", "alt+f4", "escape"
export type KeyCallback = (event: KeyboardEvent) => void;

export const Keyboard = (() => {
  const _bindings = new Map<string, Set<KeyCallback>>();
  let _listening = false;

  function _normalizeCombo(combo: KeyCombo): string {
    return combo
      .toLowerCase()
      .split("+")
      .map(p => p.trim())
      .sort((a, b) => {
        // Modifier keys first, then the actual key
        const order = ["ctrl", "alt", "shift", "meta"];
        const ai = order.indexOf(a);
        const bi = order.indexOf(b);
        if (ai !== -1 && bi !== -1) return ai - bi;
        if (ai !== -1) return -1;
        if (bi !== -1) return 1;
        return 0;
      })
      .join("+");
  }

  function _comboFromEvent(e: KeyboardEvent): string {
    const parts: string[] = [];
    if (e.ctrlKey) parts.push("ctrl");
    if (e.altKey) parts.push("alt");
    if (e.shiftKey) parts.push("shift");
    if (e.metaKey) parts.push("meta");
    const key = e.key.toLowerCase();
    if (!["control", "alt", "shift", "meta"].includes(key)) parts.push(key);
    return parts.join("+");
  }

  function _onKeyDown(e: KeyboardEvent): void {
    const combo = _comboFromEvent(e);
    const cbs = _bindings.get(combo);
    if (cbs) {
      for (const cb of cbs) cb(e);
    }
  }

  function _startListening(): void {
    if (_listening) return;
    _listening = true;
    window.addEventListener("keydown", _onKeyDown);
  }

  function bind(combo: KeyCombo, cb: KeyCallback): () => void {
    _startListening();
    const key = _normalizeCombo(combo);
    if (!_bindings.has(key)) _bindings.set(key, new Set());
    _bindings.get(key)!.add(cb);
    return () => unbind(combo, cb);
  }

  function unbind(combo: KeyCombo, cb: KeyCallback): void {
    const key = _normalizeCombo(combo);
    _bindings.get(key)?.delete(cb);
  }

  function unbindAll(combo?: KeyCombo): void {
    if (combo) {
      _bindings.delete(_normalizeCombo(combo));
    } else {
      _bindings.clear();
    }
  }

  return { bind, unbind, unbindAll };
})();

// ---------------------------------------------------------------------------
// ContextMenu
// ---------------------------------------------------------------------------

export interface ContextMenuItem {
  id: string;
  label: string;
  action?: () => void;
  disabled?: boolean;
  separator?: boolean;
}

export const ContextMenu = (() => {
  let _menuEl: HTMLElement | null = null;
  let _items: ContextMenuItem[] = [];
  let _visible = false;

  function _ensureEl(): HTMLElement {
    if (_menuEl) return _menuEl;
    const el = document.createElement("ul");
    el.className = "context-menu";
    Object.assign(el.style, {
      position: "fixed",
      zIndex: "9999",
      display: "none",
      margin: "0",
      padding: "4px 0",
      listStyle: "none",
      background: "#fff",
      border: "1px solid rgba(0,0,0,0.15)",
      borderRadius: "6px",
      boxShadow: "0 4px 16px rgba(0,0,0,0.15)",
      minWidth: "160px",
    });
    document.body.appendChild(el);
    _menuEl = el;

    document.addEventListener("pointerdown", (e) => {
      if (_menuEl && !_menuEl.contains(e.target as Node)) hide();
    });

    return el;
  }

  function show(x: number, y: number, items?: ContextMenuItem[]): void {
    if (typeof document === "undefined") return;
    if (items) _items = items;
    const el = _ensureEl();

    // Rebuild list
    el.innerHTML = "";
    for (const item of _items) {
      const li = document.createElement("li");
      if (item.separator) {
        li.style.cssText = "height:1px;background:rgba(0,0,0,0.1);margin:4px 0";
      } else {
        li.textContent = item.label;
        Object.assign(li.style, {
          padding: "6px 16px",
          cursor: item.disabled ? "not-allowed" : "pointer",
          opacity: item.disabled ? "0.4" : "1",
          userSelect: "none",
        });
        if (!item.disabled && item.action) {
          li.addEventListener("click", () => { item.action!(); hide(); });
        }
      }
      el.appendChild(li);
    }

    el.style.display = "block";
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
    _visible = true;
  }

  function hide(): void {
    if (_menuEl) _menuEl.style.display = "none";
    _visible = false;
  }

  function setItems(items: ContextMenuItem[]): void {
    _items = items;
  }

  return {
    show,
    hide,
    setItems,
    get isVisible() { return _visible; },
    get items() { return _items; },
  };
})();

// ---------------------------------------------------------------------------
// UserInput — normalize mouse / touch / pen events
// ---------------------------------------------------------------------------

export type PointerKind = "mouse" | "touch" | "pen" | "unknown";

export interface NormalizedPointer {
  kind: PointerKind;
  pointerId: number;
  x: number;
  y: number;
  pressure: number;
  tiltX: number;
  tiltY: number;
  buttons: number;
  timestamp: number;
}

export const UserInput = (() => {
  function normalize(e: PointerEvent | MouseEvent | TouchEvent): NormalizedPointer[] {
    const ts = e.timeStamp;

    // PointerEvent — covers mouse, touch, pen via unified API
    if (window.PointerEvent && e instanceof PointerEvent) {
      return [{
        kind: (e.pointerType as PointerKind) ?? "unknown",
        pointerId: e.pointerId,
        x: e.clientX,
        y: e.clientY,
        pressure: e.pressure,
        tiltX: e.tiltX,
        tiltY: e.tiltY,
        buttons: e.buttons,
        timestamp: ts,
      }];
    }

    // TouchEvent fallback
    if (typeof TouchEvent !== "undefined" && e instanceof TouchEvent) {
      const touches = Array.from(e.changedTouches);
      return touches.map((t) => ({
        kind: "touch" as PointerKind,
        pointerId: t.identifier,
        x: t.clientX,
        y: t.clientY,
        pressure: t.force,
        tiltX: 0,
        tiltY: 0,
        buttons: 1,
        timestamp: ts,
      }));
    }

    // MouseEvent fallback
    const me = e as MouseEvent;
    return [{
      kind: "mouse",
      pointerId: 0,
      x: me.clientX,
      y: me.clientY,
      pressure: me.buttons > 0 ? 0.5 : 0,
      tiltX: 0,
      tiltY: 0,
      buttons: me.buttons,
      timestamp: ts,
    }];
  }

  function isPrimary(p: NormalizedPointer): boolean {
    return p.pointerId === 0 || p.kind === "mouse";
  }

  return { normalize, isPrimary };
})();
