/**
 * cell-a11y.ts — Cell Container ARIA labels + keyboard Tab navigation
 *
 * Fuses upstream/pixijs-engine/src/accessibility (AccessibilitySystem,
 * accessibilityTarget, AccessibleOptions) into the cell rendering pipeline:
 *   - Each cell Container gets accessible = true with ARIA label derived from
 *     cell_id and species: "<species>: <cell_id>"
 *   - accessibleTitle   = species (short screen-reader prefix)
 *   - accessibleHint    = full aria-label: "<species>: <cell_id>"
 *   - accessibleType    = 'button' (default — Enter/Space triggers click)
 *   - tabIndex          = ordered by cell z-index (lower z → earlier in Tab flow)
 *   - eventMode         = 'static' (required for tabIndex to propagate to shadow div)
 *
 * Integration points:
 *   1. applyCellA11y(container, meta)      — stamp a11y on a single Container
 *   2. attachCellA11y(app, descs, opts)    — post-render bulk setup helper
 *   3. CellA11yManager                     — class for lifecycle management
 *
 * Upstream references:
 *   upstream/pixijs-engine/src/accessibility/AccessibilitySystem.ts
 *   upstream/pixijs-engine/src/accessibility/accessibilityTarget.ts
 *   upstream/pixijs-engine/src/accessibility/__docs__/accessibility.md
 *   skills/pixijs/pixijs-accessibility/SKILL.md
 *
 * Algorithm notes:
 *   - We do NOT manually create DOM divs — that is AccessibilitySystem's job.
 *     We only set the Container properties it reads: accessible, accessibleTitle,
 *     accessibleHint, accessibleType, accessiblePointerEvents, tabIndex, eventMode.
 *   - AccessibilitySystem activates on first Tab keypress (default) or
 *     immediately when enabledByDefault: true is passed to Application.init().
 *   - We call app.renderer.accessibility.setAccessibilityEnabled(true) if
 *     enableOnInit is true (matches enabledByDefault behaviour from the outside).
 *   - Tab order: cells sorted by z ascending, then by cell_id lexicographically
 *     for stability across re-renders. tabIndex values are 1-based consecutive
 *     integers so browsers honour the declared order.
 *   - keyboard Enter/Space → shadow div click → PixiJS dispatches 'pointertap'
 *     on the Container → CellEventSystem.onClick fires (if wired).
 *
 * [ASTRO-CELL-A11Y] debug prefix.
 */

import { Application } from '../../../upstream/pixijs-engine/src/app/Application';
import { Container } from '../../../upstream/pixijs-engine/src/scene/container/Container';

import type { CellMeta } from './cell-event-system';
import type { CellDescriptor } from './pixi-cell-renderer';

// ── Species → human-readable label map ────────────────────────────────────────
// Maps CIL icon species identifiers to descriptive English names used in ARIA
// labels.  Screen readers announce these to visually-impaired users.

const SPECIES_LABEL: Record<string, string> = {
  'cil-eye':         'Attention',
  'cil-vector':      'Vector transform',
  'cil-bolt':        'Activation',
  'cil-plus':        'Addition',
  'cil-arrow-right': 'Output gate',
  'cil-filter':      'Filter',
  'cil-code':        'Code block',
  'cil-layers':      'Layers',
  'cil-loop':        'Loop',
  'cil-graph':       'Graph',
};

/**
 * buildAriaLabel — canonical aria-label string for a cell.
 *
 * Format: "<species readable> — <cell_id>"
 * Example: "Attention — self_attn_q"
 *
 * Screen readers read this verbatim; keep it terse and descriptive.
 */
export function buildAriaLabel(cell_id: string, species: string): string {
  const speciesName = SPECIES_LABEL[species] ?? species.replace('cil-', '');
  return `${speciesName} — ${cell_id}`;
}

/**
 * buildAriaTitle — short title (announced first in some screen readers).
 *
 * Format: "<species readable>"
 * Example: "Attention"
 */
export function buildAriaTitle(species: string): string {
  return SPECIES_LABEL[species] ?? species.replace('cil-', '');
}

// ── applyCellA11y — stamp accessibility properties on one Container ────────────

/**
 * applyCellA11y — mutate one cell Container to be keyboard/screen-reader
 * accessible via the PixiJS AccessibilitySystem.
 *
 * Must be called after buildCellContainer() so the Container exists.
 * The tabIndex parameter sets the keyboard Tab order (1-based integer).
 * eventMode is set to 'static' so PixiJS forwards tabIndex to the shadow div.
 *
 * @param container  The cell Container returned by buildCellContainer()
 * @param meta       CellMeta (cell_id, species) — typically stored as __cellMeta
 * @param tabIndex   1-based Tab order position
 *
 * @example
 * ```ts
 * const container = buildCellContainer(desc);
 * applyCellA11y(container, { cell_id: desc.cell_id, species: desc.species, ... }, 1);
 * app.stage.addChild(container);
 * ```
 */
export function applyCellA11y(
  container: Container,
  meta: Pick<CellMeta, 'cell_id' | 'species'>,
  tabIndex: number,
): void {
  // ── Core accessible flag — triggers AccessibilitySystem to create shadow div ─
  container.accessible = true;

  // ── ARIA label via accessibleHint → sets aria-label on the shadow div ─────
  container.accessibleHint = buildAriaLabel(meta.cell_id, meta.species);

  // ── accessibleTitle → sets title="" on shadow div (tooltip / AT prefix) ───
  container.accessibleTitle = buildAriaTitle(meta.species);

  // ── Shadow div type: button — Enter/Space will dispatch 'click' + 'pointertap'
  container.accessibleType = 'button';

  // ── pointer-events: auto so keyboard-focus on the div can be clicked ──────
  container.accessiblePointerEvents = 'auto';

  // ── tabIndex: positive integer, 1-based, controls Tab traversal order ─────
  // eventMode must be 'static' or 'dynamic' for the value to propagate to
  // the shadow div (see AccessibilitySystem._addChild tabIndex block).
  container.eventMode = 'static';
  container.tabIndex  = tabIndex;

  // ── Store on container for debugging / CellEventSystem introspection ──────
  (container as any).__a11yLabel    = container.accessibleHint;
  (container as any).__a11yTabIndex = tabIndex;

  console.log(
    `[ASTRO-CELL-A11Y] ${meta.cell_id} → tabIndex=${tabIndex}  "${container.accessibleHint}"`,
  );
}

// ── Tab order comparator ───────────────────────────────────────────────────────

/**
 * cellTabOrder — sort key for a CellDescriptor or CellMeta+z bundle.
 *
 * Primary:   z ascending (lower z-layer → earlier in Tab flow, i.e. "back-to-front" reading order)
 * Secondary: cell_id lexicographic (stable across re-renders)
 */
function cellTabOrderKey(cell_id: string, z: number): string {
  // Zero-pad z to 6 digits for lexicographic string sort
  return `${String(z).padStart(6, '0')}_${cell_id}`;
}

// ── CellA11yOptions ────────────────────────────────────────────────────────────

export interface CellA11yOptions {
  /**
   * When true, immediately enables the AccessibilitySystem so keyboard/screen-reader
   * overlay is active without waiting for the first Tab keypress.
   * Mirrors AccessibilityOptions.enabledByDefault.
   * @default false
   */
  enableOnInit?: boolean;

  /**
   * When true, the AccessibilitySystem stays active even after mouse movement.
   * Useful in debug/demo environments where you want to verify the shadow divs.
   * Mirrors AccessibilityOptions.deactivateOnMouseMove = false.
   * @default false
   */
  persistOnMouseMove?: boolean;

  /**
   * Custom ARIA label builder.  Override to localise or customise the label format.
   * Receives cell_id and species; return the full aria-label string.
   */
  buildLabel?: (cell_id: string, species: string) => string;
}

// ── CellA11yManager ────────────────────────────────────────────────────────────

/**
 * CellA11yManager — lifecycle manager for cell ARIA labelling + Tab navigation.
 *
 * Wraps the bulk applyCellA11y logic, tracks registered containers, and
 * provides an update() path for live-poll scenarios where cells are spawned
 * and destroyed dynamically.
 *
 * Usage:
 * ```ts
 * const a11y = new CellA11yManager(app, { enableOnInit: true });
 *
 * // After renderCellGraph():
 * a11y.registerAll(cells);        // bulk register from CellDescriptor[]
 *
 * // Or per-container:
 * a11y.register(container, meta, zIndex);
 *
 * // When a cell is removed:
 * a11y.unregister(cell_id);
 *
 * // Cleanup:
 * a11y.destroy();
 * ```
 */
export class CellA11yManager {
  private app:  Application;
  private opts: Required<CellA11yOptions>;

  /** Map from cell_id → registered Container */
  private containers: Map<string, Container> = new Map();

  constructor(app: Application, opts: CellA11yOptions = {}) {
    this.app  = app;
    this.opts = {
      enableOnInit:       opts.enableOnInit       ?? false,
      persistOnMouseMove: opts.persistOnMouseMove ?? false,
      buildLabel:         opts.buildLabel         ?? buildAriaLabel,
    };

    // ── Optionally activate AccessibilitySystem immediately ────────────────
    if (this.opts.enableOnInit) {
      // Attempt to enable immediately; falls back gracefully if renderer has no
      // accessibility system (e.g. unit tests with headless renderer).
      try {
        (this.app.renderer as any).accessibility?.setAccessibilityEnabled(true);
      } catch {
        // accessibility system absent — silently skip
      }
    }

    // ── Optionally persist overlay across mouse moves ──────────────────────
    // AccessibilitySystem exposes no direct setter for deactivateOnMouseMove after
    // init, but we can override the private handler reference at runtime.
    if (this.opts.persistOnMouseMove) {
      try {
        const sys = (this.app.renderer as any).accessibility;
        if (sys) {
          // Patch: replace the bound onMouseMove so it no longer calls _deactivate.
          // Accesses private members — acceptable here since this is a debug/demo option.
          sys._boundOnMouseMove = () => { /* no-op: do not deactivate on mouse move */ };
        }
      } catch {
        // silently ignore if accessor not available
      }
    }

    console.log(
      `[ASTRO-CELL-A11Y] CellA11yManager init  enableOnInit=${this.opts.enableOnInit}`,
    );
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * register — apply ARIA attributes to one container.
   *
   * @param container  The PixiJS Container to tag.
   * @param meta       CellMeta carrying cell_id and species.
   * @param z          z-index used for Tab-order sorting (pass desc.z).
   */
  register(container: Container, meta: Pick<CellMeta, 'cell_id' | 'species'>, z: number): void {
    // Compute tabIndex by re-sorting entire set (so order stays consistent
    // even when register is called incrementally).
    this.containers.set(meta.cell_id, container);
    // Stamp a temporary tabIndex; _reindex() will fix all indices after bulk load.
    applyCellA11y(container, meta, 0);
    // Store z on container for reindex
    (container as any).__a11yZ = z;
    this._reindex();
  }

  /**
   * registerAll — bulk registration from a CellDescriptor array.
   *
   * Locates containers on app.stage via __cellMeta stamp (set by buildCellContainer).
   * Call once after renderCellGraph() has added all containers to the stage.
   *
   * @param descs  Array of CellDescriptors (same array passed to renderCellGraph).
   */
  registerAll(descs: CellDescriptor[]): void {
    // Walk stage children and match to descriptors
    for (const child of this.app.stage.children) {
      const meta = (child as any).__cellMeta as CellMeta | undefined;
      if (!meta) continue;

      const desc = descs.find((d) => d.cell_id === meta.cell_id);
      if (!desc) continue;

      this.containers.set(meta.cell_id, child as Container);
      (child as any).__a11yZ = desc.z ?? 0;
    }

    // Apply ARIA + compute correct Tab order in one pass
    this._reindex();

    console.log(
      `[ASTRO-CELL-A11Y] registerAll: ${this.containers.size} containers registered`,
    );
  }

  /**
   * unregister — remove a11y tracking for a cell (called when a cell is destroyed).
   */
  unregister(cell_id: string): void {
    const container = this.containers.get(cell_id);
    if (container) {
      container.accessible = false;
      // Clearing accessible will cause AccessibilitySystem to remove the shadow div
      // on the next postrender cycle.
    }
    this.containers.delete(cell_id);
    this._reindex();
  }

  /**
   * update — re-apply ARIA labels after a bulk cell state change (e.g. poll cycle).
   *
   * Call this when the live poll loop spawns or removes cells so that Tab order
   * stays correct and newly-added containers get their labels.
   *
   * @param descs  Latest CellDescriptor array from the poll response.
   */
  update(descs: CellDescriptor[]): void {
    const seen = new Set<string>();

    for (const child of this.app.stage.children) {
      const meta = (child as any).__cellMeta as CellMeta | undefined;
      if (!meta) continue;

      const desc = descs.find((d) => d.cell_id === meta.cell_id);
      if (!desc) continue;

      seen.add(meta.cell_id);

      if (!this.containers.has(meta.cell_id)) {
        // New container — register it
        this.containers.set(meta.cell_id, child as Container);
        (child as any).__a11yZ = desc.z ?? 0;
      }
    }

    // Unregister stale containers
    for (const [id] of this.containers) {
      if (!seen.has(id)) this.unregister(id);
    }

    this._reindex();
  }

  /**
   * destroy — clean up (does not destroy the underlying containers).
   */
  destroy(): void {
    for (const [, container] of this.containers) {
      container.accessible = false;
    }
    this.containers.clear();
    console.log('[ASTRO-CELL-A11Y] CellA11yManager destroyed');
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  /**
   * _reindex — recompute Tab order for all registered containers and apply.
   *
   * Sort order: z ascending, then cell_id lexicographic.
   * Assigns consecutive 1-based tabIndex values.
   */
  private _reindex(): void {
    // Build (cell_id, z, container) triples
    const entries: Array<{ cell_id: string; z: number; container: Container }> = [];

    for (const [cell_id, container] of this.containers) {
      const z: number = (container as any).__a11yZ ?? 0;
      entries.push({ cell_id, z, container });
    }

    // Sort by z then cell_id
    entries.sort((a, b) => {
      const ka = cellTabOrderKey(a.cell_id, a.z);
      const kb = cellTabOrderKey(b.cell_id, b.z);
      return ka < kb ? -1 : ka > kb ? 1 : 0;
    });

    // Apply 1-based tabIndex; re-apply ARIA labels using custom builder if set
    for (let i = 0; i < entries.length; i++) {
      const { cell_id, container } = entries[i];
      const meta = (container as any).__cellMeta as CellMeta | undefined;
      if (!meta) continue;

      const tabIdx = i + 1;
      const label  = this.opts.buildLabel(meta.cell_id, meta.species);

      // Only mutate if values have changed to avoid unnecessary AccessibilitySystem updates
      if (
        container.tabIndex       !== tabIdx ||
        container.accessibleHint !== label
      ) {
        container.tabIndex       = tabIdx;
        container.accessibleHint = label;
        container.accessibleTitle = buildAriaTitle(meta.species);
        (container as any).__a11yLabel    = label;
        (container as any).__a11yTabIndex = tabIdx;
      }
    }
  }
}

// ── attachCellA11y — one-liner integration ────────────────────────────────────

/**
 * attachCellA11y — convenience wrapper.
 *
 * Creates a CellA11yManager, calls registerAll(descs), and optionally enables
 * the AccessibilitySystem immediately.  Returns the manager for lifecycle control.
 *
 * Designed to be called right after renderCellGraph() or
 * CellEventSystem.attachToCellContainers() so all containers are in the stage.
 *
 * @example
 * ```ts
 * const app  = await renderCellGraph(canvas, cells, edges);
 * const a11y = attachCellA11y(app, cells, { enableOnInit: true });
 *
 * // Tab key now navigates between cells; screen readers announce
 * // "Attention — self_attn_q", "Activation — ffn_0", etc.
 *
 * // Cleanup:
 * a11y.destroy();
 * ```
 */
export function attachCellA11y(
  app:   Application,
  descs: CellDescriptor[],
  opts:  CellA11yOptions = {},
): CellA11yManager {
  const mgr = new CellA11yManager(app, opts);
  mgr.registerAll(descs);
  return mgr;
}

// ── Standalone helper: applyA11yToContainer ────────────────────────────────────

/**
 * applyA11yToContainer — apply a11y to a container given a CellDescriptor,
 * without requiring a CellA11yManager instance.
 *
 * Useful when you build containers one-by-one and want to stamp ARIA inline:
 *
 * @example
 * ```ts
 * const container = buildCellContainer(desc);
 * applyA11yToContainer(container, desc, 1);
 * app.stage.addChild(container);
 * ```
 */
export function applyA11yToContainer(
  container: Container,
  desc: CellDescriptor,
  tabIndex: number,
): void {
  applyCellA11y(
    container,
    { cell_id: desc.cell_id, species: desc.species },
    tabIndex,
  );
}
