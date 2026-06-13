/**
 * OcclusionQuery.ts — GPU遮挡查询，跳过不可见 cell
 *
 * AT Renderer.useOcclusionQuery 对标实现。
 *
 * 策略（两帧延迟 / two-frame-lag）:
 *   Frame N  : 用 N-1 帧的 query 结果过滤 cell → 只渲染可见 cell
 *              同时为所有 cell 发起本帧新的 bbox query
 *   Frame N+1: 读取 Frame N 的 query 结果 → 更新 visibilityMap
 *
 * 这样永远不阻塞 GPU pipeline — gl.getQueryParameter 只在结果已可用时调用，
 * 绝不 stall。
 *
 * WebGL2 API 用到:
 *   gl.createQuery / gl.deleteQuery
 *   gl.beginQuery(gl.ANY_SAMPLES_PASSED_CONSERVATIVE, query)
 *   gl.endQuery(gl.ANY_SAMPLES_PASSED_CONSERVATIVE)
 *   gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE)
 *   gl.getQueryParameter(query, gl.QUERY_RESULT)
 *
 * 与 CellInstanceManager 集成:
 *   const oq = new OcclusionQueryManager(gl, cellManager);
 *   // 每帧:
 *   oq.beginFrame();
 *   oq.issueQueries(viewMat, projMat);   // bbox depth-only pass
 *   cellManager.drawVisible(oq.visibilityMap, viewMat, projMat); // 正常渲染
 *   oq.endFrame();
 */

import type { CellInstanceManager, CellParamsJson, CellBBox } from './CellInstanceManager';

// ── Shaders (bbox depth-only) ─────────────────────────────────────────────────

/**
 * Minimal vertex shader — projects the 8 corners of a 2D bbox as a screen-
 * aligned quad.  We pass bbox as uniforms (cx, cy, w, h, z) and use
 * gl_VertexID 0-5 (two triangles) to reconstruct corners without a VBO.
 */
const BBOX_VERT = /* glsl */ `#version 300 es
precision highp float;

// bbox in world space
uniform float u_cx;
uniform float u_cy;
uniform float u_w;
uniform float u_h;
uniform float u_z;          // depth layer (0 = front)

uniform mat4 u_view;
uniform mat4 u_projection;

// Quad corners from gl_VertexID (0-5, two CCW triangles)
// 0: TL  1: TR  2: BL  3: TR  4: BR  5: BL
const vec2 CORNERS[6] = vec2[6](
  vec2(-0.5,  0.5),
  vec2( 0.5,  0.5),
  vec2(-0.5, -0.5),
  vec2( 0.5,  0.5),
  vec2( 0.5, -0.5),
  vec2(-0.5, -0.5)
);

void main() {
  vec2 corner = CORNERS[gl_VertexID];
  vec3 world  = vec3(u_cx + corner.x * u_w,
                     u_cy + corner.y * u_h,
                     u_z);
  gl_Position = u_projection * u_view * vec4(world, 1.0);
}
`;

/** Fragment shader outputs nothing — depth write only */
const BBOX_FRAG = /* glsl */ `#version 300 es
precision lowp float;
void main() { /* depth-only pass, no colour output */ }
`;

// ── Types ─────────────────────────────────────────────────────────────────────

export interface OcclusionQueryOptions {
  /**
   * Use ANY_SAMPLES_PASSED_CONSERVATIVE (default) for better GPU performance.
   * Set false to use ANY_SAMPLES_PASSED (more accurate, slower on some GPUs).
   */
  conservative?: boolean;

  /**
   * Extra margin factor applied to the bbox before issuing a query.
   * 1.0 = exact bbox, 1.1 = 10 % larger.  Helps avoid false negatives for
   * cells that are only partially visible.
   * Default: 1.05
   */
  bboxPadding?: number;
}

/** Per-cell query state for a single frame */
interface CellQueryState {
  cellId:  string;
  query:   WebGLQuery;
  pending: boolean;   // query has been issued, result not yet read
}

/** Public visibility map: cellId → true (visible) | false (occluded) */
export type VisibilityMap = Map<string, boolean>;

// ── OcclusionQueryManager ─────────────────────────────────────────────────────

export class OcclusionQueryManager {
  private gl:      WebGL2RenderingContext;
  private manager: CellInstanceManager;

  // Compiled bbox shader program + uniform locations
  private program:    WebGLProgram;
  private uCx:        WebGLUniformLocation;
  private uCy:        WebGLUniformLocation;
  private uW:         WebGLUniformLocation;
  private uH:         WebGLUniformLocation;
  private uZ:         WebGLUniformLocation;
  private uView:      WebGLUniformLocation;
  private uProj:      WebGLUniformLocation;

  // Two-frame-lag ring buffer: index 0 and 1 alternate each frame
  private _pendingQueries: Array<CellQueryState[]> = [[], []];
  private _frameSlot = 0;  // which slot we WRITE to this frame (0 or 1)

  // The visibility map that the caller reads
  readonly visibilityMap: VisibilityMap = new Map();

  private _conservative: boolean;
  private _bboxPadding:  number;

  // Pool of recycled WebGLQuery objects to avoid churn
  private _queryPool: WebGLQuery[] = [];

  // Cached list of all cells (rebuilt when manager's cell list changes)
  private _cells: CellParamsJson[] = [];

  constructor(
    gl: WebGL2RenderingContext,
    manager: CellInstanceManager,
    opts: OcclusionQueryOptions = {},
  ) {
    this.gl      = gl;
    this.manager = manager;
    this._conservative = opts.conservative ?? true;
    this._bboxPadding  = opts.bboxPadding  ?? 1.05;

    // Compile depth-only bbox program
    this.program = this._compileProgram();

    // Resolve uniform locations
    const p = this.program;
    this.uCx   = this._loc(p, 'u_cx');
    this.uCy   = this._loc(p, 'u_cy');
    this.uW    = this._loc(p, 'u_w');
    this.uH    = this._loc(p, 'u_h');
    this.uZ    = this._loc(p, 'u_z');
    this.uView = this._loc(p, 'u_view');
    this.uProj = this._loc(p, 'u_projection');
  }

  // ── Public frame lifecycle ────────────────────────────────────────────────

  /**
   * Call at the start of each frame — resolves last frame's pending queries
   * and updates visibilityMap.
   *
   * Must be called BEFORE issueQueries().
   */
  beginFrame(): void {
    // The slot we READ is the *other* slot (the one we wrote last frame)
    const readSlot = this._frameSlot ^ 1;
    this._resolvePendingSlot(readSlot);
  }

  /**
   * Render every cell's bbox as a depth-only proxy and issue an occlusion
   * query for each.
   *
   * Call this AFTER your depth pre-pass so the depth buffer already contains
   * occluder geometry.
   *
   * @param view       4×4 column-major view matrix (Float32Array[16])
   * @param projection 4×4 column-major projection matrix (Float32Array[16])
   */
  issueQueries(view: Float32Array, projection: Float32Array): void {
    const gl = this.gl;
    const writeSlot = this._frameSlot;

    // Return any queries from this slot to the pool before reuse
    this._recycleSlot(writeSlot);

    // Refresh cell list (cheap — just iterates groups Map)
    this._refreshCellList();

    // Set up GL state for depth-only pass
    const prevProgram = gl.getParameter(gl.CURRENT_PROGRAM) as WebGLProgram | null;
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uView, false, view);
    gl.uniformMatrix4fv(this.uProj, false, projection);

    // Disable colour writes — depth only
    gl.colorMask(false, false, false, false);
    gl.depthMask(false);       // don't write depth for the probe quads
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);

    const target = this._conservative
      ? (gl as any).ANY_SAMPLES_PASSED_CONSERVATIVE
      : (gl as any).ANY_SAMPLES_PASSED;

    const newPending: CellQueryState[] = [];

    for (const cell of this._cells) {
      const query = this._acquireQuery();
      const bbox  = cell.bbox;
      const pad   = this._bboxPadding;

      gl.uniform1f(this.uCx, bbox.x + bbox.w / 2);
      gl.uniform1f(this.uCy, bbox.y + bbox.h / 2);
      gl.uniform1f(this.uW,  bbox.w * pad);
      gl.uniform1f(this.uH,  bbox.h * pad);
      gl.uniform1f(this.uZ,  bbox.z ?? 0.0);

      gl.beginQuery(target, query);
      // Draw 6 vertices (two triangles) with no VBO — gl_VertexID only
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.endQuery(target);

      newPending.push({ cellId: cell.cell_id, query, pending: true });

      // Seed visibility as true on first encounter (conservative default)
      if (!this.visibilityMap.has(cell.cell_id)) {
        this.visibilityMap.set(cell.cell_id, true);
      }
    }

    this._pendingQueries[writeSlot] = newPending;

    // Restore GL state
    gl.colorMask(true, true, true, true);
    gl.depthMask(true);
    gl.useProgram(prevProgram);

    // Advance slot for next frame
    this._frameSlot = writeSlot ^ 1;
  }

  /**
   * Optionally call at the end of the frame for bookkeeping.
   * Not strictly required — beginFrame() handles cross-frame resolution.
   */
  endFrame(): void {
    // no-op for now; hook available for future instrumentation
  }

  // ── Visibility helpers ────────────────────────────────────────────────────

  /**
   * Returns true if the cell should be drawn this frame.
   * Defaults to true for cells with no query result yet (first frame).
   */
  isVisible(cellId: string): boolean {
    return this.visibilityMap.get(cellId) ?? true;
  }

  /** List of cell IDs currently marked as visible */
  get visibleCellIds(): string[] {
    return [...this.visibilityMap.entries()]
      .filter(([, v]) => v)
      .map(([k]) => k);
  }

  /** Number of cells skipped (occluded) last frame */
  get occludedCount(): number {
    let n = 0;
    for (const v of this.visibilityMap.values()) if (!v) n++;
    return n;
  }

  // ── Dispose ───────────────────────────────────────────────────────────────

  dispose(): void {
    const gl = this.gl;

    for (const slot of this._pendingQueries) {
      for (const s of slot) gl.deleteQuery(s.query);
    }
    for (const q of this._queryPool) gl.deleteQuery(q);

    this._queryPool.length = 0;
    this._pendingQueries   = [[], []];
    this.visibilityMap.clear();

    gl.deleteProgram(this.program);
  }

  // ── Private ───────────────────────────────────────────────────────────────

  /** Read back query results for `slot`.  Non-blocking — skips unavailable queries. */
  private _resolvePendingSlot(slot: number): void {
    const gl      = this.gl;
    const pending = this._pendingQueries[slot];

    for (const state of pending) {
      if (!state.pending) continue;

      const available = gl.getQueryParameter(state.query, gl.QUERY_RESULT_AVAILABLE) as boolean;
      if (!available) {
        // GPU hasn't finished yet — keep last known visibility (conservative)
        continue;
      }

      const anySamples = gl.getQueryParameter(state.query, gl.QUERY_RESULT) as number;
      this.visibilityMap.set(state.cellId, anySamples !== 0);
      state.pending = false;
    }
  }

  /** Return all queries in slot back to the pool for reuse */
  private _recycleSlot(slot: number): void {
    for (const s of this._pendingQueries[slot]) {
      this._queryPool.push(s.query);
    }
    this._pendingQueries[slot] = [];
  }

  /** Acquire a query from pool or create a new one */
  private _acquireQuery(): WebGLQuery {
    return this._queryPool.pop() ?? this.gl.createQuery()!;
  }

  /** Flatten all cells from CellInstanceManager groups */
  private _refreshCellList(): void {
    this._cells = [];
    // Access the manager's groups via the public cells property on each group
    for (const species of (this.manager as any).groups.values()) {
      this._cells.push(...(species as { cells: CellParamsJson[] }).cells);
    }
  }

  private _compileProgram(): WebGLProgram {
    const gl   = this.gl;
    const vert = this._compileShader(gl.VERTEX_SHADER,   BBOX_VERT);
    const frag = this._compileShader(gl.FRAGMENT_SHADER, BBOX_FRAG);

    const program = gl.createProgram()!;
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);
    gl.deleteShader(vert);
    gl.deleteShader(frag);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program) ?? 'unknown';
      gl.deleteProgram(program);
      throw new Error(`[OcclusionQuery] Program link error:\n${log}`);
    }
    return program;
  }

  private _compileShader(type: number, src: string): WebGLShader {
    const gl     = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader) ?? 'unknown';
      gl.deleteShader(shader);
      throw new Error(`[OcclusionQuery] Shader compile error:\n${log}`);
    }
    return shader;
  }

  private _loc(program: WebGLProgram, name: string): WebGLUniformLocation {
    const loc = this.gl.getUniformLocation(program, name);
    if (!loc) throw new Error(`[OcclusionQuery] Uniform not found: ${name}`);
    return loc;
  }
}

// ── CellInstanceManager integration patch ────────────────────────────────────
//
// Augment CellInstanceManager with drawVisible() so callers can filter by
// the OcclusionQueryManager's visibility map without modifying the base class.
//
// Usage:
//   import { patchCellInstanceManagerWithOcclusion } from './OcclusionQuery';
//   patchCellInstanceManagerWithOcclusion();
//   // then: cellManager.drawVisible(oq.visibilityMap, view, proj);

declare module './CellInstanceManager' {
  interface CellInstanceManager {
    /**
     * Like draw(), but skips any species group whose cells are ALL occluded,
     * and within partially-visible groups adjusts the instance count so only
     * visible instances are submitted.
     *
     * NOTE: this reorders cells in-place if necessary; call after issueQueries()
     * but before the next loadFromDescriptors() call.
     */
    drawVisible(
      visibilityMap: VisibilityMap,
      view?: Float32Array,
      projection?: Float32Array,
    ): void;
  }
}

/**
 * One-time monkey-patch that adds drawVisible() to CellInstanceManager.
 * Call once at app startup (idempotent).
 */
export function patchCellInstanceManagerWithOcclusion(): void {
  const proto = (
    // dynamic import avoids circular dependency at module parse time
    require('./CellInstanceManager').CellInstanceManager as { prototype: any }
  ).prototype;

  if (proto.drawVisible) return; // already patched

  proto.drawVisible = function (
    this: any,
    visibilityMap: VisibilityMap,
    view?: Float32Array,
    projection?: Float32Array,
  ): void {
    for (const group of (this.groups as Map<string, any>).values()) {
      const cells: CellParamsJson[] = group.cells;

      // Partition: visible cells first, occluded last
      const visibleIdx: number[] = [];
      for (let i = 0; i < cells.length; i++) {
        if (visibilityMap.get(cells[i].cell_id) !== false) {
          visibleIdx.push(i);
        }
      }

      if (visibleIdx.length === 0) continue; // entire species occluded

      // Temporarily reduce instance count to only visible cells
      const originalCount = group.mesh.instanceCount as number;
      if (visibleIdx.length < originalCount) {
        // Compact visible instances to the front of the buffer
        const savedInstances: any[] = cells.map((_, i) =>
          (group.mesh as any)._instanceData
            ? (group.mesh as any)._instanceData[i]
            : null
        );

        // Re-upload only visible slice
        for (let vi = 0; vi < visibleIdx.length; vi++) {
          const src = visibleIdx[vi];
          if (savedInstances[src]) {
            (group.mesh as any).setInstanceAttribute(vi, savedInstances[src]);
          }
        }
        group.mesh.setInstanceCount(visibleIdx.length);
        group.mesh.upload();
        group.mesh.draw(view, projection);

        // Restore full buffer
        group.mesh.setInstanceCount(originalCount);
        for (let i = 0; i < originalCount; i++) {
          if (savedInstances[i]) {
            (group.mesh as any).setInstanceAttribute(i, savedInstances[i]);
          }
        }
        group.mesh.upload();
      } else {
        // All visible — normal draw
        group.mesh.draw(view, projection);
      }
    }
  };
}
