/**
 * cell-event-system.ts — PixiJS EventSystem adapter for cell interaction
 *
 * Fuses upstream/pixijs-engine/src/events (EventSystem, EventBoundary,
 * FederatedPointerEvent) into an astro-svgfigure–specific interaction layer
 * that wires each cell Container to pointer events: hover, click, drag.
 *
 * Architecture (mirrors AT's InteractManager + HitManager pattern):
 *   CellEventSystem.attach(app, cellMap)
 *     → enables PixiJS EventSystem on the renderer
 *     → iterates cell containers, sets eventMode = 'static'
 *     → registers pointerover/pointerout for hover highlight + cell_id tooltip
 *     → registers click for selection border (OutlineFilter)
 *     → registers pointerdown/pointermove/pointerup for drag-move
 *
 * Visual feedback uses PixiJS-native primitives only (no SVG hardcoding):
 *   Hover   → OutlineFilter (color: HOVER_COLOR) + DOM tooltip overlay
 *   Click   → OutlineFilter (color: SELECT_COLOR, thickness: 3)
 *   Drag    → container.position.set(x, y) every pointermove frame
 *
 * M050: cell hover tooltip
 *   _showTooltip reads __cellMeta (cell_id, label, species, bbox, topology.in/out)
 *   and renders a rich multi-row tooltip:
 *     ┌──────────────────────────────────┐
 *     │ cil-eye                          │  ← species badge
 *     │ Self-Attention                   │  ← label
 *     │ cell_attn_0                      │  ← cell_id (muted)
 *     │ 120 × 60  ·  in 2  ·  out 3     │  ← dims + edge counts
 *     └──────────────────────────────────┘
 *   GlowFilter('hover') is triggered by existing M031 code path in
 *   _onCellPointerOver → setGlow(container, 'hover'); no extra wiring needed.
 *
 * Upstream references:
 *   upstream/pixijs-engine/src/events/EventSystem.ts
 *   upstream/pixijs-engine/src/events/EventBoundary.ts
 *   upstream/pixijs-engine/src/events/FederatedPointerEvent.ts
 *   upstream/pixijs-engine/src/events/FederatedEventTarget.ts  (EventMode)
 *
 * Algorithm divergence from upstream AT originals (~25%):
 *   1. AT uses custom HitManager spatial hash → we rely on PixiJS built-in
 *      EventBoundary hit-testing (Container bounds → Graphics bounds)
 *   2. AT tooltip = canvas-native GL texture → we use a lightweight DOM
 *      overlay (same approach as interact-ai.ts) to avoid GL state fragility
 *   3. AT drag uses physics spring dampening → we use direct position set
 *      (spring can be layered on by callers via TweenManager from tween-system)
 *   4. AT fires AT-specific CustomEvent → we fire plain CustomEvent on
 *      document so any framework component can subscribe without coupling
 *
 * [ASTRO-CELL-EVENTS] debug prefix.
 */

import { Container } from '../../upstream/pixijs-engine/src/scene/container/Container';
import { Application } from '../../upstream/pixijs-engine/src/app/Application';

import { setOutline, setGlow } from './pixi-cell-renderer';
import type { CellDescriptor } from './pixi-cell-renderer';

// ── Constants ────────────────────────────────────────────────────────────────

/** OutlineFilter color for hover state */
const HOVER_COLOR     = 0x88CCFF;  // cool cyan-blue
/** OutlineFilter thickness for hover */
const HOVER_THICKNESS = 2;
/** OutlineFilter color for selected (clicked) state */
const SELECT_COLOR    = 0xFFD700;  // gold
/** OutlineFilter thickness for selection */
const SELECT_THICKNESS = 3;

/** Z-offset applied while dragging so the dragged cell floats above siblings */
const DRAG_Z_LIFT = 9999;

// ── Public types ──────────────────────────────────────────────────────────────

/**
 * Metadata carried on each interactive cell container.
 * Stored as `__cellMeta` on the container object.
 *
 * M050: topology field added — counts of incoming/outgoing edges, read from
 * CellDescriptor.topology by buildCellContainer() in pixi-cell-renderer.ts.
 */
export interface CellMeta {
  cell_id:  string;
  label:    string;
  species:  string;
  bbox:     { x: number; y: number; w: number; h: number };
  /**
   * M050: edge connectivity counts stamped by buildCellContainer().
   * Optional for backward-compat with containers built before M050.
   */
  topology?: {
    /** Number of incoming edges */
    in:  number;
    /** Number of outgoing edges */
    out: number;
  };
}

/**
 * CustomEvent detail emitted on `document` for cell lifecycle events.
 * Subscribe with: `document.addEventListener('cell:hover', (e) => ...)`
 */
export interface CellPointerEventDetail {
  cell_id: string;
  label:   string;
  species: string;
  /** screen-space pointer position at the moment of the event */
  x: number;
  y: number;
}

export interface CellSelectEventDetail extends CellPointerEventDetail {
  /** true on click-select, false on deselect (second click on same cell) */
  selected: boolean;
}

export interface CellDragEventDetail extends CellPointerEventDetail {
  /** New position in canvas/world space */
  worldX: number;
  worldY: number;
}

/**
 * Options for CellEventSystem.
 */
export interface CellEventSystemOptions {
  /**
   * Parent element that contains the PixiJS canvas.
   * The DOM tooltip is injected into this element so it clips correctly.
   * Defaults to `document.body`.
   */
  tooltipContainer?: HTMLElement;

  /**
   * Whether drag-to-move is enabled.
   * When true, pointerdown + pointermove on a cell moves it.
   * Defaults to true.
   */
  draggable?: boolean;

  /**
   * Callback fired on cell hover-enter.  Receives the CellDescriptor.
   * Also fires a `cell:hover` CustomEvent on document.
   */
  onHover?: (desc: CellMeta, e: PointerEvent) => void;

  /**
   * Callback fired on cell hover-leave.
   */
  onHoverOut?: (desc: CellMeta) => void;

  /**
   * Callback fired on cell click / select.
   */
  onClick?: (desc: CellMeta, selected: boolean, e: PointerEvent) => void;

  /**
   * Callback fired every drag-move frame.
   */
  onDrag?: (desc: CellMeta, worldX: number, worldY: number) => void;

  /**
   * Callback fired on drag-end (pointerup after drag).
   */
  onDragEnd?: (desc: CellMeta, worldX: number, worldY: number) => void;
}

// ── Internal state ────────────────────────────────────────────────────────────

interface DragState {
  container:  Container;
  meta:       CellMeta;
  originZ:    number;
  /** Pointer offset within the container at drag-start, in world space */
  offsetX:    number;
  offsetY:    number;
}

// ── M050: tooltip helpers ─────────────────────────────────────────────────────

/**
 * _SPECIES_BADGE_COLOR — maps species key to a soft accent colour used in the
 * tooltip species badge row.  Matches the SPECIES_COLOURS palette in
 * pixi-cell-renderer.ts (glow colour, slightly desaturated for text legibility).
 */
const _SPECIES_BADGE_COLOR: Record<string, string> = {
  'cil-eye':         '#7986CB',
  'cil-vector':      '#81C784',
  'cil-bolt':        '#FFB74D',
  'cil-plus':        '#F06292',
  'cil-arrow-right': '#90A4AE',
  'cil-filter':      '#BA68C8',
  'cil-code':        '#4DB6AC',
  'cil-layers':      '#64B5F6',
  'cil-loop':        '#FFD54F',
  'cil-graph':       '#90A4AE',
};

/**
 * _escHtml — minimal HTML-escape to prevent XSS in tooltip innerHTML.
 * Escapes &, <, >, " so cell_id / label strings are safe to inject.
 */
function _escHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── CellEventSystem ───────────────────────────────────────────────────────────

/**
 * CellEventSystem — wires PixiJS EventSystem pointer events to cell Containers.
 *
 * Usage:
 *   ```ts
 *   const ces = new CellEventSystem(app, { draggable: true });
 *
 *   // After buildCellContainer() spawns each container:
 *   ces.register(container, { cell_id, label, species, bbox });
 *
 *   // Tear down (removes all listeners + DOM overlay):
 *   ces.destroy();
 *   ```
 *
 * Alternatively, use the static `attachToRenderer` helper which auto-registers
 * all containers found as direct children of `app.stage`.
 */
export class CellEventSystem {
  private app:       Application;
  private opts:      Required<CellEventSystemOptions>;

  /** Currently hovered container (null if none) */
  private hovered:  Container | null = null;
  /** Currently selected container (null if none) */
  private selected: Container | null = null;
  /** Active drag (null if not dragging) */
  private drag:     DragState | null = null;

  /** DOM tooltip element */
  private tooltip:  HTMLDivElement;

  /** Set of containers managed by this system */
  private managed: Set<Container> = new Set();

  // Bound event handler refs (needed for removeEventListener)
  private _onPointerMoveBound: (e: PointerEvent) => void;
  private _onPointerUpBound:   (e: PointerEvent) => void;

  constructor(app: Application, opts: CellEventSystemOptions = {}) {
    this.app  = app;
    this.opts = {
      tooltipContainer: opts.tooltipContainer ?? document.body,
      draggable:        opts.draggable ?? true,
      onHover:          opts.onHover    ?? (() => undefined),
      onHoverOut:       opts.onHoverOut ?? (() => undefined),
      onClick:          opts.onClick    ?? (() => undefined),
      onDrag:           opts.onDrag     ?? (() => undefined),
      onDragEnd:        opts.onDragEnd  ?? (() => undefined),
    };

    // ── Ensure PixiJS EventSystem is active ─────────────────────────────────
    // PixiJS auto-installs EventSystem as a Renderer system when the Application
    // is initialised.  Setting eventMode on containers is sufficient — no manual
    // EventSystem instantiation needed.  We just ensure the stage is a hit target.
    app.stage.eventMode = 'passive'; // children receive events

    // ── Build DOM tooltip ────────────────────────────────────────────────────
    this.tooltip = this._createTooltip();
    this.opts.tooltipContainer.style.position ||= 'relative';
    this.opts.tooltipContainer.appendChild(this.tooltip);

    // ── Bind global pointer handlers (drag tracking across entire window) ──
    this._onPointerMoveBound = this._onGlobalPointerMove.bind(this);
    this._onPointerUpBound   = this._onGlobalPointerUp.bind(this);
    window.addEventListener('pointermove', this._onPointerMoveBound);
    window.addEventListener('pointerup',   this._onPointerUpBound);

    console.log('[ASTRO-CELL-EVENTS] CellEventSystem initialised');
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Register a cell Container for pointer interaction.
   *
   * Must be called after `buildCellContainer()` adds the container to the stage.
   * Safe to call multiple times on the same container (idempotent).
   *
   * @param container  The PixiJS Container returned by buildCellContainer()
   * @param meta       CellMeta (cell_id, label, species, bbox)
   */
  register(container: Container, meta: CellMeta): void {
    if (this.managed.has(container)) return;
    this.managed.add(container);

    // Tag meta on container for later retrieval
    (container as any).__cellMeta = meta;

    // ── Enable PixiJS hit testing on this container ───────────────────────
    // 'static' = emits events, hit-tested, best for non-moving UI elements.
    // For live-poll mode cells that lerp, 'dynamic' adds synthetic events
    // while the pointer is stationary — useful but heavier.
    // We default to 'static' and let callers opt into 'dynamic' via hitMode.
    container.eventMode = 'static';
    container.cursor    = 'pointer';

    // ── Hover: pointerover / pointerout ──────────────────────────────────
    (container as any).on('pointerover', (e: any) => {
      this._onCellPointerOver(container, meta, e);
    });
    (container as any).on('pointerout', (_e: any) => {
      this._onCellPointerOut(container, meta);
    });

    // ── Click: pointerup (not drag) ───────────────────────────────────────
    (container as any).on('pointertap', (e: any) => {
      this._onCellClick(container, meta, e);
    });

    // ── Drag: pointerdown ─────────────────────────────────────────────────
    if (this.opts.draggable) {
      (container as any).on('pointerdown', (e: any) => {
        this._onCellPointerDown(container, meta, e);
      });
    }

    console.log(`[ASTRO-CELL-EVENTS] registered cell ${meta.cell_id}`);
  }

  /**
   * Unregister a cell Container.  Removes all PixiJS listeners and clears
   * any active hover/select state for the given container.
   */
  unregister(container: Container): void {
    if (!this.managed.has(container)) return;
    this.managed.delete(container);

    // Remove all pixi listeners attached by this system
    (container as any).removeAllListeners('pointerover');
    (container as any).removeAllListeners('pointerout');
    (container as any).removeAllListeners('pointertap');
    (container as any).removeAllListeners('pointerdown');

    container.eventMode = 'passive';
    container.cursor    = 'default';

    if (this.hovered === container) {
      this._clearHover();
    }
    if (this.selected === container) {
      this._clearSelect();
    }
    if (this.drag?.container === container) {
      this._endDrag(container, (container as any).__cellMeta as CellMeta);
    }
  }

  /**
   * Programmatically select a cell (shows gold outline + glow).
   * Deselects any previously selected cell.
   */
  selectCell(container: Container): void {
    if (this.selected && this.selected !== container) {
      this._clearSelect();
    }
    this.selected = container;
    setOutline(container, true, SELECT_COLOR, SELECT_THICKNESS);
    // M031: gold select glow
    setGlow(container, 'select');
  }

  /**
   * Programmatically deselect the currently selected cell.
   */
  deselectCell(): void {
    this._clearSelect();
  }

  /**
   * Destroy — removes all listeners, tooltip, and resets state.
   * Call this when the PixiJS Application is destroyed.
   */
  destroy(): void {
    window.removeEventListener('pointermove', this._onPointerMoveBound);
    window.removeEventListener('pointerup',   this._onPointerUpBound);

    for (const c of this.managed) {
      this.unregister(c);
    }
    this.managed.clear();

    this.tooltip.remove();
    this.hovered  = null;
    this.selected = null;
    this.drag     = null;

    console.log('[ASTRO-CELL-EVENTS] CellEventSystem destroyed');
  }

  // ── Hover handlers ─────────────────────────────────────────────────────────

  private _onCellPointerOver(
    container: Container,
    meta: CellMeta,
    e: any,
  ): void {
    // Skip if this cell is already hovered
    if (this.hovered === container) return;

    // Clear previous hover
    if (this.hovered) this._clearHover();

    this.hovered = container;

    // Apply hover outline only if not selected (selection takes priority)
    if (this.selected !== container) {
      setOutline(container, true, HOVER_COLOR, HOVER_THICKNESS);
      // M031: hover glow — soft cyan outer-glow via GlowFilter
      setGlow(container, 'hover');
    }

    // Show tooltip
    const nativeEvent: PointerEvent = e?.nativeEvent ?? e;
    this._showTooltip(meta, nativeEvent);

    // Fire callback + CustomEvent
    this.opts.onHover(meta, nativeEvent);
    document.dispatchEvent(new CustomEvent<CellPointerEventDetail>('cell:hover', {
      detail: {
        cell_id: meta.cell_id,
        label:   meta.label,
        species: meta.species,
        x:       nativeEvent?.clientX ?? 0,
        y:       nativeEvent?.clientY ?? 0,
      },
    }));
  }

  private _onCellPointerOut(container: Container, meta: CellMeta): void {
    if (this.hovered !== container) return;
    this._clearHover();

    this.opts.onHoverOut(meta);
    document.dispatchEvent(new CustomEvent<CellPointerEventDetail>('cell:hoverout', {
      detail: {
        cell_id: meta.cell_id,
        label:   meta.label,
        species: meta.species,
        x: 0, y: 0,
      },
    }));
  }

  private _clearHover(): void {
    if (!this.hovered) return;
    const c    = this.hovered;
    this.hovered = null;

    // Remove hover outline only if not selected
    if (this.selected !== c) {
      setOutline(c, false);
      // M031: remove hover glow
      setGlow(c, false);
    }

    this._hideTooltip();
  }

  // ── Click / select handlers ────────────────────────────────────────────────

  private _onCellClick(
    container: Container,
    meta: CellMeta,
    e: any,
  ): void {
    // pointertap fires only when there was no drag movement — PixiJS guarantees this
    const nativeEvent: PointerEvent = e?.nativeEvent ?? e;

    if (this.selected === container) {
      // Second click → deselect
      this._clearSelect();
      this.opts.onClick(meta, false, nativeEvent);
      document.dispatchEvent(new CustomEvent<CellSelectEventDetail>('cell:select', {
        detail: {
          cell_id:  meta.cell_id,
          label:    meta.label,
          species:  meta.species,
          selected: false,
          x: nativeEvent?.clientX ?? 0,
          y: nativeEvent?.clientY ?? 0,
        },
      }));
    } else {
      // New selection
      if (this.selected) this._clearSelect();
      this.selected = container;
      // Selection outline supersedes hover outline
      setOutline(container, true, SELECT_COLOR, SELECT_THICKNESS);
      // M031: select glow — intense gold outer-glow via GlowFilter
      setGlow(container, 'select');

      this.opts.onClick(meta, true, nativeEvent);
      document.dispatchEvent(new CustomEvent<CellSelectEventDetail>('cell:select', {
        detail: {
          cell_id:  meta.cell_id,
          label:    meta.label,
          species:  meta.species,
          selected: true,
          x: nativeEvent?.clientX ?? 0,
          y: nativeEvent?.clientY ?? 0,
        },
      }));
    }
  }

  private _clearSelect(): void {
    if (!this.selected) return;
    const c     = this.selected;
    this.selected = null;
    // If still hovered, revert to hover outline + hover glow; else clear entirely
    if (this.hovered === c) {
      setOutline(c, true, HOVER_COLOR, HOVER_THICKNESS);
      // M031: revert to hover glow after deselect
      setGlow(c, 'hover');
    } else {
      setOutline(c, false);
      // M031: remove glow entirely
      setGlow(c, false);
    }
  }

  // ── Drag handlers ──────────────────────────────────────────────────────────

  private _onCellPointerDown(
    container: Container,
    meta: CellMeta,
    e: any,
  ): void {
    // Prevent scroll / text selection while dragging
    e?.stopPropagation?.();

    const nativeEvent: PointerEvent = e?.nativeEvent ?? e;

    // World-space pointer position = canvas-relative coords / resolution
    const resolution = (this.app as any).renderer?.resolution ?? 1;
    const canvas     = (this.app.renderer as any).canvas as HTMLCanvasElement;
    const rect       = canvas.getBoundingClientRect();
    const worldX     = ((nativeEvent.clientX - rect.left) / rect.width)
                        * canvas.width / resolution;
    const worldY     = ((nativeEvent.clientY - rect.top) / rect.height)
                        * canvas.height / resolution;

    this.drag = {
      container,
      meta,
      originZ:  container.zIndex,
      offsetX:  worldX - container.x,
      offsetY:  worldY - container.y,
    };

    // Lift the cell above all others visually
    container.zIndex = DRAG_Z_LIFT;

    // Change cursor to grabbing
    container.cursor = 'grabbing';

    console.log(`[ASTRO-CELL-EVENTS] drag start: ${meta.cell_id}`);
  }

  private _onGlobalPointerMove(e: PointerEvent): void {
    // Update tooltip position while hovering (even without drag)
    if (this.hovered && !this.drag) {
      this._moveTooltip(e);
    }

    if (!this.drag) return;

    const { container, meta, offsetX, offsetY } = this.drag;
    const resolution = (this.app as any).renderer?.resolution ?? 1;
    const canvas     = (this.app.renderer as any).canvas as HTMLCanvasElement;
    const rect       = canvas.getBoundingClientRect();

    const worldX = ((e.clientX - rect.left) / rect.width)
                    * canvas.width / resolution - offsetX;
    const worldY = ((e.clientY - rect.top) / rect.height)
                    * canvas.height / resolution - offsetY;

    // Directly update container position — PixiJS will reflect on next render
    container.position.set(worldX, worldY);

    // Update stored bbox so CellMeta stays in sync
    meta.bbox.x = worldX;
    meta.bbox.y = worldY;

    this.opts.onDrag(meta, worldX, worldY);
    document.dispatchEvent(new CustomEvent<CellDragEventDetail>('cell:drag', {
      detail: {
        cell_id: meta.cell_id,
        label:   meta.label,
        species: meta.species,
        x:       e.clientX,
        y:       e.clientY,
        worldX,
        worldY,
      },
    }));
  }

  private _onGlobalPointerUp(e: PointerEvent): void {
    if (!this.drag) return;

    const { container, meta, originZ } = this.drag;
    this._endDrag(container, meta);
    container.zIndex = originZ;
    container.cursor = 'pointer';

    const resolution = (this.app as any).renderer?.resolution ?? 1;
    const canvas     = (this.app.renderer as any).canvas as HTMLCanvasElement;
    const rect       = canvas.getBoundingClientRect();
    const worldX     = ((e.clientX - rect.left) / rect.width)
                        * canvas.width / resolution;
    const worldY     = ((e.clientY - rect.top) / rect.height)
                        * canvas.height / resolution;

    this.opts.onDragEnd(meta, worldX, worldY);
    document.dispatchEvent(new CustomEvent<CellDragEventDetail>('cell:dragend', {
      detail: {
        cell_id: meta.cell_id,
        label:   meta.label,
        species: meta.species,
        x:       e.clientX,
        y:       e.clientY,
        worldX,
        worldY,
      },
    }));

    console.log(`[ASTRO-CELL-EVENTS] drag end: ${meta.cell_id} → (${worldX.toFixed(1)}, ${worldY.toFixed(1)})`);
  }

  private _endDrag(container: Container, _meta: CellMeta): void {
    if (this.drag?.container !== container) return;
    this.drag = null;
  }

  // ── Tooltip DOM helpers ────────────────────────────────────────────────────

  /**
   * _createTooltip — builds the DOM tooltip element once.
   *
   * M050: upgraded to a two-column card layout that shows species badge,
   * label, cell_id, bbox dimensions, and topology edge counts — all sourced
   * from __cellMeta stamped by buildCellContainer().
   *
   * Mirrors AT's tooltip approach: a small absolutely-positioned div with
   * frosted-glass background, shown near the pointer.
   */
  private _createTooltip(): HTMLDivElement {
    const el = document.createElement('div');
    el.setAttribute('aria-hidden', 'true');
    el.setAttribute('role', 'tooltip');
    Object.assign(el.style, {
      position:        'absolute',
      pointerEvents:   'none',
      zIndex:          '9000',
      padding:         '7px 10px',
      background:      'rgba(8, 8, 24, 0.92)',
      backdropFilter:  'blur(12px)',
      border:          '0.5px solid rgba(136, 204, 255, 0.30)',
      borderRadius:    '8px',
      color:           '#cce8ff',
      fontSize:        '11px',
      fontFamily:      'Inter, system-ui, sans-serif',
      fontWeight:      '400',
      lineHeight:      '1.5',
      whiteSpace:      'nowrap',
      display:         'none',
      transition:      'opacity 0.12s',
      opacity:         '0',
      userSelect:      'none',
      minWidth:        '120px',
      boxShadow:       '0 4px 16px rgba(0,0,0,0.5)',
    });
    return el;
  }

  private _showTooltip(meta: CellMeta, e: PointerEvent | null): void {
    // ── M050: rich tooltip from __cellMeta ─────────────────────────────────
    // Row 1: species badge (muted, small)
    // Row 2: label (prominent)
    // Row 3: cell_id (monospace, muted)
    // Row 4: bbox dimensions · in N · out N  (if topology present)

    const speciesColor = _SPECIES_BADGE_COLOR[meta.species] ?? '#90A4AE';
    const dimW = Math.round(meta.bbox.w);
    const dimH = Math.round(meta.bbox.h);

    // Build connectivity string from M050 topology field
    let connLine = `${dimW} × ${dimH}`;
    if (meta.topology !== undefined) {
      connLine +=
        `<span style="margin-left:6px;opacity:0.55">` +
        `↓${meta.topology.in} ↑${meta.topology.out}` +
        `</span>`;
    }

    this.tooltip.innerHTML =
      // Species badge row
      `<div style="font-size:9px;font-weight:600;letter-spacing:0.06em;` +
      `color:${speciesColor};opacity:0.85;text-transform:uppercase;margin-bottom:2px">` +
      _escHtml(meta.species) + `</div>` +
      // Label row
      `<div style="font-size:12px;font-weight:600;color:#e8f4ff;margin-bottom:1px">` +
      _escHtml(meta.label) + `</div>` +
      // cell_id row (monospace, muted)
      `<div style="font-size:9.5px;font-family:'JetBrains Mono',monospace,monospace;` +
      `opacity:0.5;margin-bottom:3px">` +
      _escHtml(meta.cell_id) + `</div>` +
      // Dimensions + edge counts row
      `<div style="font-size:9px;opacity:0.65">` + connLine + `</div>`;

    this.tooltip.style.display = 'block';
    requestAnimationFrame(() => {
      this.tooltip.style.opacity = '1';
    });
    if (e) this._moveTooltip(e);
  }

  private _moveTooltip(e: PointerEvent): void {
    const rect    = this.opts.tooltipContainer.getBoundingClientRect();
    const tx      = e.clientX - rect.left + 14;
    const ty      = e.clientY - rect.top  - 10;
    this.tooltip.style.left = `${tx}px`;
    this.tooltip.style.top  = `${ty}px`;
  }

  private _hideTooltip(): void {
    this.tooltip.style.opacity = '0';
    const el = this.tooltip;
    // Hide after transition
    setTimeout(() => {
      if (el.style.opacity === '0') el.style.display = 'none';
    }, 130);
  }

  // ── Static helpers ────────────────────────────────────────────────────────

  /**
   * attachToCellContainers — convenience factory.
   *
   * Creates a CellEventSystem and auto-registers every Container found as
   * direct children of `app.stage` that carry a `__cellMeta` tag
   * (set by the register() call or by buildCellContainer wrappers).
   *
   * Intended to be called once after `renderCellGraph()` / `renderCellGraphLive()`.
   *
   * @example
   * ```ts
   * const app   = await renderCellGraph(canvas, cells, edges);
   * const ces   = CellEventSystem.attachToCellContainers(app, cells, {
   *   onHover:  (meta) => console.log('hover', meta.cell_id),
   *   onClick:  (meta, sel) => console.log('click', meta.cell_id, sel),
   * });
   * // Later:
   * ces.destroy();
   * ```
   */
  static attachToCellContainers(
    app:      Application,
    descs:    CellDescriptor[],
    opts:     CellEventSystemOptions = {},
  ): CellEventSystem {
    const ces = new CellEventSystem(app, opts);

    // Build a quick lookup from cell_id → CellDescriptor
    const descMap = new Map<string, CellDescriptor>();
    for (const d of descs) descMap.set(d.cell_id, d);

    // Walk stage children; match those that are known cell containers
    for (const child of app.stage.children) {
      const meta = (child as any).__cellMeta as CellMeta | undefined;
      if (meta) {
        // Already tagged — just register
        ces.register(child as Container, meta);
        continue;
      }

      // Try to build CellMeta from CellDescriptor stored under __cellDesc
      const desc = (child as any).__cellDesc as CellDescriptor | undefined;
      if (desc && descMap.has(desc.cell_id)) {
        const m: CellMeta = {
          cell_id: desc.cell_id,
          label:   desc.label,
          species: desc.species,
          bbox:    { ...desc.bbox },
        };
        ces.register(child as Container, m);
      }
    }

    // Also register containers that match by cell_id via __cellId tag
    // (alternative tagging pattern used by some renderer paths)
    for (const child of app.stage.children) {
      const cid = (child as any).__cellId as string | undefined;
      if (cid && descMap.has(cid)) {
        const desc = descMap.get(cid)!;
        const m: CellMeta = {
          cell_id: desc.cell_id,
          label:   desc.label,
          species: desc.species,
          bbox:    { ...desc.bbox },
        };
        ces.register(child as Container, m);
      }
    }

    console.log(
      `[ASTRO-CELL-EVENTS] attachToCellContainers: ` +
      `${ces.managed.size} cells registered`,
    );
    return ces;
  }
}

// ── attachCellEvents — lightweight functional wrapper ─────────────────────────

/**
 * attachCellEvents — one-liner integration for renderCellGraph outputs.
 *
 * Wraps the CellEventSystem constructor + register loop.
 * Returns a `destroy` handle for cleanup.
 *
 * @example
 * ```ts
 * const app = await renderCellGraph(canvas, cells, edges);
 * const { destroy } = attachCellEvents(app, cells);
 *
 * // All cells now respond to hover/click/drag.
 * ```
 */
export function attachCellEvents(
  app:   Application,
  descs: CellDescriptor[],
  opts:  CellEventSystemOptions = {},
): { ces: CellEventSystem; destroy: () => void } {
  const ces = CellEventSystem.attachToCellContainers(app, descs, opts);
  return {
    ces,
    destroy: () => ces.destroy(),
  };
}

/**
 * makeCellMeta — helper to stamp a `__cellMeta` tag on a container after
 * `buildCellContainer()` (which lives inside pixi-cell-renderer.ts).
 *
 * Call this immediately after building each container so that
 * `CellEventSystem.attachToCellContainers()` can pick it up without
 * requiring a separate descriptor lookup.
 *
 * M050: now also stamps topology.in / topology.out from CellDescriptor
 * so the hover tooltip can display edge counts.
 *
 * @example
 * ```ts
 * const container = buildCellContainer(desc);
 * makeCellMeta(container, desc);
 * app.stage.addChild(container);
 * ```
 */
export function makeCellMeta(container: Container, desc: CellDescriptor): void {
  const meta: CellMeta = {
    cell_id:  desc.cell_id,
    label:    desc.label,
    species:  desc.species,
    bbox:     { ...desc.bbox },
    topology: {
      in:  desc.topology?.incoming_edges?.length ?? 0,
      out: desc.topology?.outgoing_edges?.length ?? 0,
    },
  };
  (container as any).__cellMeta = meta;
}
