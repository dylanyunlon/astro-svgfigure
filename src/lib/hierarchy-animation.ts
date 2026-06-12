/**
 * hierarchy-animation.ts — Parent-child cell hierarchy & compound node animation
 *
 * Provides:
 *   HierarchyAnimation — manages parent-child relationships between cells
 *                        compound node children follow parent cell movement
 *                        setParent / removeParent / updateHierarchy
 *
 * Design:
 *   - Each cell may have at most one parent.
 *   - A parent may have N children (compound node semantics).
 *   - When a parent cell moves by (Δx, Δy), every descendant is shifted by
 *     the same delta so that the compound node behaves as a rigid body.
 *   - updateHierarchy(dt) integrates pending deltas accumulated via
 *     notifyMove() and flushes them to all registered position stores.
 *   - The class is intentionally renderer-agnostic: callers inject a
 *     PositionStore callback; HierarchyAnimation never touches the DOM or GPU.
 *
 * Upstream references:
 *   upstream/thing-editor/src/engine/lib/hierarchy-utils.ts
 *   upstream/animation-editor/src/scene/compound-node.ts
 *
 * AT references: HierarchyAnimation ×14, setParent ×22, compound node ×9
 */

// ── Public types ──────────────────────────────────────────────────────────────

/** Minimal position record used by all callbacks. */
export interface CellPos {
  x: number;
  y: number;
}

/**
 * Callback invoked by HierarchyAnimation whenever a cell's position
 * must be updated.  The renderer layer should write these values into
 * its display-object / descriptor store.
 */
export type PositionSetter = (cellId: string, pos: CellPos) => void;

/** Options forwarded to the HierarchyAnimation constructor. */
export interface HierarchyAnimationOptions {
  /**
   * Called once per cell whose position changed during updateHierarchy().
   * If omitted the caller must poll getDirty() instead.
   */
  onPositionChange?: PositionSetter;

  /**
   * Maximum depth when propagating deltas downward.
   * Guards against infinite loops caused by cyclic parent assignments.
   * Default: 64.
   */
  maxDepth?: number;
}

/** A snapshot of a cell's current logical position. */
export interface CellState {
  id: string;
  pos: CellPos;
  parentId: string | null;
  /** Ordered list of direct child IDs. */
  childIds: string[];
}

// ── Internal bookkeeping types ────────────────────────────────────────────────

interface InternalCell {
  id: string;
  pos: CellPos;
  parentId: string | null;
  childIds: Set<string>;
  /** Accumulated positional delta not yet flushed to children. */
  pendingDx: number;
  pendingDy: number;
  /** Marks the cell dirty until the next updateHierarchy() flush. */
  dirty: boolean;
}

// ── HierarchyAnimation ────────────────────────────────────────────────────────

/**
 * HierarchyAnimation
 *
 * Manages a forest of cell nodes.  Call register() for every cell you want
 * to track, setParent() to wire up compound-node relationships, notifyMove()
 * whenever your interaction layer moves a cell, and updateHierarchy(dt) once
 * per animation frame to propagate deltas to children.
 *
 * @example
 * ```ts
 * const ha = new HierarchyAnimation({
 *   onPositionChange: (id, pos) => scene.setCellPos(id, pos),
 * });
 * ha.register('group-A', { x: 100, y: 200 });
 * ha.register('child-1', { x: 110, y: 210 });
 * ha.register('child-2', { x: 130, y: 230 });
 * ha.setParent('child-1', 'group-A');
 * ha.setParent('child-2', 'group-A');
 *
 * // User drags group-A by (+50, +30):
 * ha.notifyMove('group-A', 50, 30);
 *
 * // In your animation loop:
 * ha.updateHierarchy(dt);  // child-1 and child-2 are also shifted
 * ```
 */
export class HierarchyAnimation {
  private readonly cells = new Map<string, InternalCell>();
  private readonly onPositionChange: PositionSetter | undefined;
  private readonly maxDepth: number;

  /** IDs that became dirty during the last updateHierarchy() call. */
  private lastDirtySet: Set<string> = new Set();

  constructor(options: HierarchyAnimationOptions = {}) {
    this.onPositionChange = options.onPositionChange;
    this.maxDepth = options.maxDepth ?? 64;
  }

  // ── Registration ────────────────────────────────────────────────────────────

  /**
   * Register a cell so HierarchyAnimation can track it.
   * Idempotent — calling register() on an already-registered cell updates
   * its initial position but preserves existing parent/child relationships.
   */
  register(cellId: string, initialPos: CellPos): void {
    if (this.cells.has(cellId)) {
      const cell = this.cells.get(cellId)!;
      cell.pos = { ...initialPos };
      return;
    }
    this.cells.set(cellId, {
      id: cellId,
      pos: { ...initialPos },
      parentId: null,
      childIds: new Set(),
      pendingDx: 0,
      pendingDy: 0,
      dirty: false,
    });
  }

  /**
   * Unregister a cell.  Its children are reparented to the cell's parent
   * (or become root nodes if the cell was a root).
   */
  unregister(cellId: string): void {
    const cell = this.cells.get(cellId);
    if (!cell) return;

    // Detach from parent
    if (cell.parentId) {
      const parent = this.cells.get(cell.parentId);
      parent?.childIds.delete(cellId);
    }

    // Reparent children
    for (const childId of cell.childIds) {
      const child = this.cells.get(childId);
      if (child) {
        child.parentId = cell.parentId; // null → becomes root
        if (cell.parentId) {
          const grandparent = this.cells.get(cell.parentId);
          grandparent?.childIds.add(childId);
        }
      }
    }

    this.cells.delete(cellId);
  }

  // ── Hierarchy management ────────────────────────────────────────────────────

  /**
   * Make `childId` a direct child of `parentId`.
   *
   * Automatically removes any previous parent relationship for `childId`.
   * Rejects the assignment if it would create a cycle and throws a
   * RangeError in that case.
   *
   * @param childId  ID of the cell to re-parent.
   * @param parentId ID of the new parent cell.
   */
  setParent(childId: string, parentId: string): void {
    if (childId === parentId) {
      throw new RangeError(`[HierarchyAnimation] setParent: cell "${childId}" cannot be its own parent`);
    }

    const child = this._requireCell(childId);
    const parent = this._requireCell(parentId);

    // Cycle detection: parentId must not be a descendant of childId
    if (this._isDescendant(parentId, childId)) {
      throw new RangeError(
        `[HierarchyAnimation] setParent: assigning "${parentId}" as parent of "${childId}" would create a cycle`
      );
    }

    // Remove from current parent
    if (child.parentId) {
      const oldParent = this.cells.get(child.parentId);
      oldParent?.childIds.delete(childId);
    }

    child.parentId = parentId;
    parent.childIds.add(childId);
  }

  /**
   * Detach `childId` from its current parent, making it a root node.
   * No-op if the cell has no parent.
   */
  removeParent(childId: string): void {
    const child = this._requireCell(childId);
    if (!child.parentId) return;

    const parent = this.cells.get(child.parentId);
    parent?.childIds.delete(childId);
    child.parentId = null;
  }

  /**
   * Re-build all parent/child links from a flat descriptor array.
   * Useful when loading a saved graph or receiving a full layout update
   * from the ELK layout engine.
   *
   * Cells not present in `descriptors` keep their existing relationships.
   *
   * @param descriptors  Array of { id, parentId?, pos? }.
   */
  updateHierarchyFromDescriptors(
    descriptors: Array<{ id: string; parentId?: string | null; pos?: CellPos }>
  ): void {
    // First pass: ensure all cells are registered
    for (const d of descriptors) {
      if (!this.cells.has(d.id)) {
        this.register(d.id, d.pos ?? { x: 0, y: 0 });
      } else if (d.pos) {
        this.cells.get(d.id)!.pos = { ...d.pos };
      }
    }

    // Second pass: wire parents
    for (const d of descriptors) {
      if (d.parentId) {
        try {
          this.setParent(d.id, d.parentId);
        } catch (err) {
          console.warn(`[HierarchyAnimation] updateHierarchyFromDescriptors: ${(err as Error).message}`);
        }
      } else {
        // Explicit null/undefined means root
        this.cells.get(d.id)!.parentId = null;
      }
    }
  }

  // ── Movement notification ───────────────────────────────────────────────────

  /**
   * Record that a cell has been moved by the given delta.
   * Call this from your drag handler or animation system.
   * Children are NOT updated immediately — call updateHierarchy(dt) to flush.
   *
   * @param cellId  Cell that was moved.
   * @param dx      Horizontal displacement in local (canvas) pixels.
   * @param dy      Vertical displacement in local (canvas) pixels.
   */
  notifyMove(cellId: string, dx: number, dy: number): void {
    const cell = this._requireCell(cellId);
    cell.pos.x += dx;
    cell.pos.y += dy;
    cell.pendingDx += dx;
    cell.pendingDy += dy;
    cell.dirty = true;
  }

  /**
   * Teleport a cell to an absolute position.
   * Computes the required delta and delegates to notifyMove().
   */
  setPosition(cellId: string, x: number, y: number): void {
    const cell = this._requireCell(cellId);
    const dx = x - cell.pos.x;
    const dy = y - cell.pos.y;
    this.notifyMove(cellId, dx, dy);
  }

  // ── Per-frame update ────────────────────────────────────────────────────────

  /**
   * Propagate all pending movement deltas down the hierarchy and invoke
   * `onPositionChange` for every affected cell.
   *
   * Should be called once per animation frame (inside requestAnimationFrame
   * or a Ticker callback).
   *
   * @param _dt  Frame delta-time in seconds (reserved for future spring/damping).
   */
  updateHierarchy(_dt = 0.016): void {
    this.lastDirtySet.clear();

    for (const cell of this.cells.values()) {
      if (!cell.dirty && (cell.pendingDx !== 0 || cell.pendingDy !== 0)) {
        cell.dirty = true;
      }

      if (cell.dirty && (cell.pendingDx !== 0 || cell.pendingDy !== 0)) {
        this._propagateDelta(cell.id, cell.pendingDx, cell.pendingDy, 0);
        cell.pendingDx = 0;
        cell.pendingDy = 0;
      }
    }

    // Fire onPositionChange for all dirty cells
    for (const cell of this.cells.values()) {
      if (cell.dirty) {
        this.lastDirtySet.add(cell.id);
        this.onPositionChange?.(cell.id, { ...cell.pos });
        cell.dirty = false;
      }
    }
  }

  // ── Query helpers ────────────────────────────────────────────────────────────

  /** Returns the current position of a registered cell, or null. */
  getPosition(cellId: string): CellPos | null {
    const cell = this.cells.get(cellId);
    return cell ? { ...cell.pos } : null;
  }

  /** Returns a read-only snapshot of a cell's state. */
  getCellState(cellId: string): CellState | null {
    const cell = this.cells.get(cellId);
    if (!cell) return null;
    return {
      id: cell.id,
      pos: { ...cell.pos },
      parentId: cell.parentId,
      childIds: [...cell.childIds],
    };
  }

  /** Returns all direct children of the given cell. */
  getChildren(cellId: string): string[] {
    return [...(this.cells.get(cellId)?.childIds ?? [])];
  }

  /** Returns all registered root cell IDs (cells without a parent). */
  getRoots(): string[] {
    const roots: string[] = [];
    for (const cell of this.cells.values()) {
      if (!cell.parentId) roots.push(cell.id);
    }
    return roots;
  }

  /**
   * Returns all descendant IDs of `cellId` in BFS order.
   * Does NOT include `cellId` itself.
   */
  getDescendants(cellId: string): string[] {
    const result: string[] = [];
    const queue: string[] = [...(this.cells.get(cellId)?.childIds ?? [])];
    const seen = new Set<string>();
    while (queue.length) {
      const id = queue.shift()!;
      if (seen.has(id)) continue;
      seen.add(id);
      result.push(id);
      const cell = this.cells.get(id);
      if (cell) {
        for (const cid of cell.childIds) queue.push(cid);
      }
    }
    return result;
  }

  /** Returns the set of cell IDs that were dirtied during the last updateHierarchy() call. */
  getDirty(): ReadonlySet<string> {
    return this.lastDirtySet;
  }

  /** Total number of registered cells. */
  get size(): number {
    return this.cells.size;
  }

  // ── Internals ────────────────────────────────────────────────────────────────

  private _requireCell(id: string): InternalCell {
    const cell = this.cells.get(id);
    if (!cell) {
      throw new ReferenceError(`[HierarchyAnimation] cell "${id}" is not registered`);
    }
    return cell;
  }

  /**
   * Recursively push a (dx, dy) delta to all children of `cellId`.
   * The parent's own position is already updated by notifyMove(); here we
   * only touch the children.
   */
  private _propagateDelta(cellId: string, dx: number, dy: number, depth: number): void {
    if (depth > this.maxDepth) {
      console.error(`[HierarchyAnimation] _propagateDelta exceeded maxDepth=${this.maxDepth} at "${cellId}" — possible cycle`);
      return;
    }
    const cell = this.cells.get(cellId);
    if (!cell) return;

    for (const childId of cell.childIds) {
      const child = this.cells.get(childId);
      if (!child) continue;
      child.pos.x += dx;
      child.pos.y += dy;
      child.dirty = true;
      // Recurse: children of this child also need to move
      this._propagateDelta(childId, dx, dy, depth + 1);
    }
  }

  /**
   * Returns true if `candidateAncestor` is an ancestor (or equal) of
   * `cellId` in the current hierarchy.
   */
  private _isDescendant(cellId: string, candidateAncestor: string): boolean {
    let current: string | null = cellId;
    const visited = new Set<string>();
    while (current !== null) {
      if (visited.has(current)) break; // cycle guard
      visited.add(current);
      if (current === candidateAncestor) return true;
      current = this.cells.get(current)?.parentId ?? null;
    }
    return false;
  }
}
