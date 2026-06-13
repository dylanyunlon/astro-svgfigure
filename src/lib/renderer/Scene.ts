/**
 * Scene.ts — base scene graph node
 *
 * AT BaseScene (53 refs in FXScene family): maintains an ordered list of
 * renderable children and delegates draw calls down the hierarchy.
 *
 * Design goals:
 *   - Framework-agnostic: no dependency on PixiJS, Three.js, or Canvas2D.
 *     Children implement the Renderable interface.
 *   - Deterministic render order via zIndex, with stable sort for equal values.
 *   - Camera transform applied before traversal so children render in
 *     view-space coordinates.
 *
 * References:
 *   AT BaseScene → FXScene → composite pipeline
 *   src/lib/fx-scene.ts  (Canvas2D layer equivalent)
 *   src/lib/renderers/nuke-pipeline.ts (pass ordering pattern)
 */

// ── Renderable interface ─────────────────────────────────────────────────────

/**
 * Anything that can be drawn by a Scene.
 * Implement this interface to add custom draw logic (meshes, sprites, etc.).
 */
export interface Renderable {
  /** Render order — lower draws first (behind).  Default: 0. */
  zIndex: number;
  /** Skip draw when false.  Default: true. */
  visible: boolean;
  /**
   * Called once per frame by Scene.render().
   * @param gl      The WebGL2 context for the active render target.
   * @param scene   The owning Scene (gives access to camera, uniforms, etc.).
   */
  draw(gl: WebGL2RenderingContext, scene: Scene): void;
  /** Optional teardown. Called by Scene.remove() if present. */
  destroy?(): void;
}

// ── Camera ───────────────────────────────────────────────────────────────────

/**
 * Minimal orthographic/perspective camera descriptor.
 * FXScene and derived classes can extend this with projection matrices.
 */
export interface Camera {
  /** Left edge of the view frustum (orthographic). */
  left: number;
  /** Right edge. */
  right: number;
  /** Top edge. */
  top: number;
  /** Bottom edge. */
  bottom: number;
  /** Near clip plane. */
  near: number;
  /** Far clip plane. */
  far: number;
}

/** Create a default identity-like orthographic camera covering [0,w] × [0,h]. */
export function makeOrthoCameraForSize(width: number, height: number): Camera {
  return { left: 0, right: width, top: 0, bottom: height, near: -1, far: 1 };
}

// ── Scene ────────────────────────────────────────────────────────────────────

export interface SceneOptions {
  /** Initial camera.  A default ortho camera is created if omitted. */
  camera?: Camera;
  /** Canvas width used for default camera and viewport. */
  width?: number;
  /** Canvas height used for default camera and viewport. */
  height?: number;
}

export class Scene {
  /** Ordered child list — sort by zIndex before rendering. */
  protected _children: Renderable[] = [];
  /** Sorted snapshot rebuilt lazily when _dirty is true. */
  private _sorted: Renderable[] = [];
  private _dirty = false;

  camera: Camera;
  width: number;
  height: number;

  constructor(opts: SceneOptions = {}) {
    this.width  = opts.width  ?? 1;
    this.height = opts.height ?? 1;
    this.camera = opts.camera ?? makeOrthoCameraForSize(this.width, this.height);
  }

  // ── Child management ─────────────────────────────────────────────────────

  /** Add a renderable child.  Duplicate adds are silently ignored. */
  add(child: Renderable): this {
    if (!this._children.includes(child)) {
      this._children.push(child);
      this._dirty = true;
    }
    return this;
  }

  /** Remove a child.  Calls child.destroy() if defined.  No-op if absent. */
  remove(child: Renderable): this {
    const idx = this._children.indexOf(child);
    if (idx !== -1) {
      this._children.splice(idx, 1);
      this._dirty = true;
      child.destroy?.();
    }
    return this;
  }

  /** Remove and destroy all children. */
  clear(): this {
    for (const child of this._children) child.destroy?.();
    this._children = [];
    this._sorted   = [];
    this._dirty    = false;
    return this;
  }

  /** True if child is in this scene. */
  has(child: Renderable): boolean {
    return this._children.includes(child);
  }

  get children(): readonly Renderable[] {
    return this._children;
  }

  // ── Sorted traversal ─────────────────────────────────────────────────────

  /** Returns children sorted by zIndex (stable). */
  private _getSorted(): Renderable[] {
    if (this._dirty) {
      // Stable sort: use index as tiebreaker
      this._sorted = this._children
        .map((c, i) => ({ c, i }))
        .sort((a, b) => a.c.zIndex - b.c.zIndex || a.i - b.i)
        .map(({ c }) => c);
      this._dirty = false;
    }
    return this._sorted;
  }

  // ── Render ───────────────────────────────────────────────────────────────

  /**
   * Traverse and draw all visible children in zIndex order.
   *
   * Subclasses (FXScene) override this to bind their RenderTarget first,
   * then call super.render() for the traversal.
   *
   * @param gl  Active WebGL2 context.  May be rendering to canvas or an FBO.
   */
  render(gl: WebGL2RenderingContext): void {
    for (const child of this._getSorted()) {
      if (child.visible) {
        child.draw(gl, this);
      }
    }
  }

  // ── Resize ───────────────────────────────────────────────────────────────

  /**
   * Update logical size and reset the default camera.
   * Called by FXScene.resize() after resizing the RenderTarget.
   */
  resize(width: number, height: number): void {
    this.width  = width;
    this.height = height;
    this.camera = makeOrthoCameraForSize(width, height);
  }

  // ── Teardown ─────────────────────────────────────────────────────────────

  /** Destroy all children.  Does NOT delete GL resources (no gl ref here). */
  destroy(): void {
    this.clear();
  }
}
