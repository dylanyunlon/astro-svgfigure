/**
 * cell-culling.ts — M017: viewport frustum skip for offscreen cells
 *
 * 将 upstream/pixijs-engine/src/culling/Culler 的思路融合进 cell 渲染管线：
 * 当 cell Container 的 bbox 完全在可视 viewport 之外时，
 *   container.visible = false  — PixiJS 跳过整棵子树的渲染 + transform 更新
 *   → filters（bloom / godray / glitch / motionBlur）不被评估，GPU 开销清零
 *
 * 可视区域来源：
 *   1. 直接传 PixiJS Application.screen（最简单，canvas 坐标）
 *   2. 传 PixiViewport（x / y / width / height）——适合 pixi-viewport 缩放/平移场景
 *   3. CameraController 的 stage 世界坐标 viewport（见 getViewportFromCamera()）
 *
 * 对接方式：在 pixi-cell-renderer.ts ticker loop 里调用一行：
 *   cullCells(live, app.screen);
 *
 * Upstream reference:
 *   upstream/pixijs-engine/src/culling/Culler.ts  — Culler._cullRecursive / culled bit
 *   upstream/pixijs-engine/src/culling/CullerPlugin.ts — 同 tick 内调用模式
 *   upstream/pixijs-engine/src/scene/container/Container.ts — visible / culled setter
 */

import type { Container } from '../../upstream/pixijs-engine/src/scene/container/Container';

// ── Public types ──────────────────────────────────────────────────────────────

/**
 * Axis-aligned viewport rectangle used as the frustum for 2-D culling.
 * Mirrors the shape of PixiJS Renderer.screen / Rectangle.
 */
export interface CellViewport {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Cell bbox as produced by the poll loop (world / stage coordinates).
 * Matches CellDescriptor.bbox — { x, y, w, h }.
 */
export interface CellBbox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Minimal live-cell interface consumed by CellCuller.
 * Works with the LiveCell shape inside pixi-cell-renderer without importing it.
 */
export interface CullableLiveCell {
  /** Current rendered X (lerped, stage-space). */
  curX: number;
  /** Current rendered Y (lerped, stage-space). */
  curY: number;
  /** Bbox dimensions from the latest descriptor. */
  desc: { bbox: CellBbox };
  /** The PixiJS Container to show/hide. */
  container: Container;
}

// ── CellCuller ────────────────────────────────────────────────────────────────

/**
 * CellCuller
 *
 * Stateless frustum-test engine for 2-D cell containers.
 *
 * The canonical usage is via the free function `cullCells()`, but the class
 * is exported for callers that want to tune the margin or reuse an instance.
 *
 * Culling algorithm (axis-aligned, stage-space):
 *   A cell is invisible when ANY of the four interval tests fails:
 *     cellMaxX  <= vp.x                (cell fully left of viewport)
 *     cellMinX  >= vp.x + vp.width     (cell fully right)
 *     cellMaxY  <= vp.y                (cell fully above)
 *     cellMinY  >= vp.y + vp.height    (cell fully below)
 *
 * This is equivalent to the Culler._cullRecursive AABB intersection test used
 * by upstream/pixijs-engine Culler when cullArea is NOT set:
 *   container.culled = bounds.x >= view.x + view.width || …
 *
 * @example
 * ```ts
 * const culler = new CellCuller({ margin: 80 });
 * // Inside the Ticker callback:
 * culler.cull(live, app.screen);
 * ```
 */
export class CellCuller {
  /**
   * Extra pixels to pad the viewport on every side before the frustum test.
   * A positive margin keeps cells visible slightly before they enter the
   * screen, preventing a 1-frame pop-in at the edge.
   * Default: 64 px — matches the typical bloom glow radius so glow halos
   * never disappear before the cell body.
   */
  margin: number;

  /** Total cells tested this frame (diagnostic). */
  testedThisFrame = 0;
  /** Total cells culled (set invisible) this frame (diagnostic). */
  culledThisFrame = 0;

  constructor(opts: { margin?: number } = {}) {
    this.margin = opts.margin ?? 64;
  }

  /**
   * cull — runs the 2-D frustum test for every live cell and sets
   * `container.visible` accordingly.
   *
   * Cells that are fading out (alpha approaching 0) are never forced
   * invisible by culling — we let the fade-out animation complete so
   * the alpha reaches 0 before the container is destroyed.
   *
   * @param cells    Map or iterable of CullableLiveCell entries.
   * @param viewport The visible area in stage/world coordinates.
   */
  cull(
    cells: Map<string, CullableLiveCell> | Iterable<CullableLiveCell>,
    viewport: CellViewport,
  ): void {
    this.testedThisFrame = 0;
    this.culledThisFrame = 0;

    const m = this.margin;
    const vpMinX = viewport.x - m;
    const vpMinY = viewport.y - m;
    const vpMaxX = viewport.x + viewport.width  + m;
    const vpMaxY = viewport.y + viewport.height + m;

    const iter: Iterable<CullableLiveCell> =
      cells instanceof Map
        ? (cells as Map<string, CullableLiveCell>).values()
        : (cells as Iterable<CullableLiveCell>);

    for (const lc of iter) {
      this.testedThisFrame++;

      // ── Determine cell AABB in stage space ─────────────────────────────
      const cellMinX = lc.curX;
      const cellMinY = lc.curY;
      const cellMaxX = lc.curX + lc.desc.bbox.w;
      const cellMaxY = lc.curY + lc.desc.bbox.h;

      // ── Frustum test (AABB vs expanded viewport) ────────────────────────
      const offscreen =
        cellMaxX <= vpMinX ||
        cellMinX >= vpMaxX ||
        cellMaxY <= vpMinY ||
        cellMinY >= vpMaxY;

      if (offscreen) {
        // Only force invisible if the cell is not mid-fade-out.
        // Fade-out cells (alpha decreasing) must stay visible so the alpha
        // animation reaches 0 and the poll loop can destroy them.
        const fadeDir = (lc.container as any).__fadeDir as number | undefined;
        if (fadeDir !== -1) {
          lc.container.visible = false;
          this.culledThisFrame++;
        }
      } else {
        // Restore visibility if the cell re-enters the viewport.
        if (!lc.container.visible) {
          lc.container.visible = true;
        }
      }
    }
  }

  /**
   * reset — force all cells visible (e.g. after a viewport jump or when
   * disabling culling at runtime).
   */
  reset(cells: Map<string, CullableLiveCell> | Iterable<CullableLiveCell>): void {
    const iter: Iterable<CullableLiveCell> =
      cells instanceof Map
        ? (cells as Map<string, CullableLiveCell>).values()
        : (cells as Iterable<CullableLiveCell>);
    for (const lc of iter) {
      lc.container.visible = true;
    }
  }

  /**
   * isVisible — single-cell frustum test without mutating visible.
   * Useful for pre-flight checks (e.g. skip building a filter chain for a
   * cell that is already off-screen at spawn time).
   */
  isVisible(lc: CullableLiveCell, viewport: CellViewport): boolean {
    const m = this.margin;
    const cellMaxX = lc.curX + lc.desc.bbox.w;
    const cellMaxY = lc.curY + lc.desc.bbox.h;
    return !(
      cellMaxX <= viewport.x - m ||
      lc.curX  >= viewport.x + viewport.width  + m ||
      cellMaxY <= viewport.y - m ||
      lc.curY  >= viewport.y + viewport.height + m
    );
  }

  /**
   * stats — returns a copy of last-frame diagnostic counters.
   */
  stats(): { tested: number; culled: number; visible: number } {
    return {
      tested:  this.testedThisFrame,
      culled:  this.culledThisFrame,
      visible: this.testedThisFrame - this.culledThisFrame,
    };
  }
}

// ── Shared singleton ─────────────────────────────────────────────────────────

/**
 * Shared CellCuller instance (64 px margin by default).
 * Mirrors the Culler.shared pattern from upstream/pixijs-engine Culler.
 *
 * @example
 * ```ts
 * import { sharedCellCuller } from './cell-culling';
 * // In ticker:
 * sharedCellCuller.cull(live, app.screen);
 * ```
 */
export const sharedCellCuller = new CellCuller();

// ── Free function API ─────────────────────────────────────────────────────────

/**
 * cullCells — convenience wrapper around `sharedCellCuller.cull()`.
 *
 * Drop this call into the per-frame tick function inside pixi-cell-renderer.ts:
 *
 * ```ts
 * // Inside pollCellChannels tick():
 * cullCells(live, app.screen);
 * ```
 *
 * @param cells    Map<string, LiveCell> from the poll loop.
 * @param viewport PixiJS Application.screen (or any CellViewport).
 * @param margin   Optional per-call margin override (px).
 */
export function cullCells(
  cells: Map<string, CullableLiveCell> | Iterable<CullableLiveCell>,
  viewport: CellViewport,
  margin?: number,
): void {
  const prev = sharedCellCuller.margin;
  if (margin !== undefined) sharedCellCuller.margin = margin;
  sharedCellCuller.cull(cells, viewport);
  if (margin !== undefined) sharedCellCuller.margin = prev;
}

// ── Viewport helpers ──────────────────────────────────────────────────────────

/**
 * viewportFromStageTransform — derives the world-space visible rectangle when
 * the PixiJS stage itself has been panned/scaled (e.g. via pixi-viewport or
 * manual stage.position / stage.scale manipulation).
 *
 * Inputs:
 *   screenWidth, screenHeight — canvas pixel dimensions
 *   stageX, stageY            — stage.position.x/y (pan offset)
 *   stageScale                — stage.scale.x (uniform zoom factor)
 *
 * The returned CellViewport is in world/stage-child coordinates so cell bbox
 * positions (which live in stage-child space) can be compared directly.
 *
 * @example
 * ```ts
 * // When using pixi-viewport or manual stage scale:
 * const vp = viewportFromStageTransform(
 *   app.screen.width, app.screen.height,
 *   app.stage.position.x, app.stage.position.y,
 *   app.stage.scale.x,
 * );
 * cullCells(live, vp);
 * ```
 */
export function viewportFromStageTransform(
  screenWidth: number,
  screenHeight: number,
  stageX: number,
  stageY: number,
  stageScale: number,
): CellViewport {
  const invScale = 1 / (stageScale || 1);
  return {
    x:      (-stageX) * invScale,
    y:      (-stageY) * invScale,
    width:  screenWidth  * invScale,
    height: screenHeight * invScale,
  };
}

/**
 * viewportFromCamera — maps an OGLCameraController's position + zoom into a
 * 2-D stage-space viewport rectangle.
 *
 * The PixiJS cell graph lives entirely in 2-D stage coordinates (cell bbox x/y).
 * The OGL camera views the scene from Z = cameraZ looking at the XY plane.
 * Given a known worldScale (cell-px → world-unit conversion factor used by
 * CameraController.focusOnCell), we can back-calculate the visible XY slab:
 *
 *   halfH_world = tan(fovRad/2) × cameraZ
 *   halfW_world = halfH_world × aspect
 *   cell_px per world unit = 1 / worldScale
 *
 * The result is a CellViewport in stage/cell-px coordinates centered on the
 * camera's orbit target XY, suitable for direct use with cullCells().
 *
 * @param cameraZ      Camera distance from XY plane (orbit radius).
 * @param targetX      Camera orbit target X in world units.
 * @param targetY      Camera orbit target Y in world units.
 * @param fovDeg       Camera vertical field of view in degrees.
 * @param aspect       Viewport aspect ratio (width / height).
 * @param worldScale   CameraController worldScale option (cell-px → world-unit).
 *                     Typically 0.01 (default in CameraController).
 *
 * @example
 * ```ts
 * import { viewportFromCamera } from './cell-culling';
 * // Inside the rAF / ticker:
 * const vp = viewportFromCamera(
 *   ctrl.camera.position.z,
 *   ctrl.orbit.target.x,
 *   ctrl.orbit.target.y,
 *   ctrl.camera.fov,
 *   canvas.width / canvas.height,
 *   0.01, // worldScale
 * );
 * cullCells(live, vp);
 * ```
 */
export function viewportFromCamera(
  cameraZ: number,
  targetX: number,
  targetY: number,
  fovDeg: number,
  aspect: number,
  worldScale: number,
): CellViewport {
  const fovRad   = (fovDeg * Math.PI) / 180;
  const halfH_w  = Math.tan(fovRad * 0.5) * Math.abs(cameraZ);
  const halfW_w  = halfH_w * aspect;

  // Convert world-unit half-extents to cell-px
  const invScale = 1 / (worldScale || 0.01);
  const halfH_px = halfH_w * invScale;
  const halfW_px = halfW_w * invScale;

  // Camera orbit target in cell-px
  const cx = targetX * invScale;
  const cy = targetY * invScale;

  return {
    x:      cx - halfW_px,
    y:      cy - halfH_px,
    width:  halfW_px * 2,
    height: halfH_px * 2,
  };
}

// ── Integration helper: attach culling to an existing ticker ─────────────────

/**
 * attachCullingToTicker — registers a culling pass on a PixiJS Application's
 * ticker that runs before the render frame.
 *
 * This is the "batteries included" integration path.  It adds a high-priority
 * ticker callback that calls `sharedCellCuller.cull()` every frame using the
 * current `app.screen` rectangle.  The live-cell map is polled via the
 * getter `getCells()` so the callback stays valid across poll-loop rebuilds.
 *
 * Priority is set to UPDATE_PRIORITY.HIGH (100) so culling runs before the
 * normal render callbacks (NORMAL = 0) — mirroring the CullerPlugin pattern.
 *
 * @param app       PixiJS Application instance.
 * @param getCells  Accessor that returns the current live-cell Map each frame.
 * @param opts      Optional: margin, enabled flag.
 * @returns         stop() function that removes the ticker callback.
 *
 * @example
 * ```ts
 * import { attachCullingToTicker } from './cell-culling';
 *
 * // After renderCellGraphLive():
 * const { app, stop: stopPoll } = await renderCellGraphLive(canvas, edges);
 * const stopCull = attachCullingToTicker(app, () => liveMap);
 *
 * // On cleanup:
 * stopCull();
 * stopPoll();
 * ```
 */
export function attachCullingToTicker(
  app: { ticker: { add: (cb: (t: any) => void, ctx?: any, priority?: number) => void; remove: (cb: (t: any) => void) => void }; screen: CellViewport },
  getCells: () => Map<string, CullableLiveCell> | Iterable<CullableLiveCell>,
  opts: { margin?: number; enabled?: boolean } = {},
): () => void {
  let enabled = opts.enabled ?? true;
  if (opts.margin !== undefined) sharedCellCuller.margin = opts.margin;

  function tick(): void {
    if (!enabled) return;
    sharedCellCuller.cull(getCells(), app.screen);
  }

  // Priority 100 = HIGH, mirrors upstream CullerPlugin tick priority
  app.ticker.add(tick, undefined, 100);

  return function stop() {
    enabled = false;
    app.ticker.remove(tick);
  };
}
