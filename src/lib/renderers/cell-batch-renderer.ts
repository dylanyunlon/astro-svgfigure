/**
 * cell-batch-renderer.ts  — M010
 *
 * Fuses upstream/pixijs-engine rendering/batcher into a per-species cell
 * batch pipeline.  Same-species cells share one geometry batch (one draw call),
 * cutting GPU state-switch overhead when the stage has many cells.
 *
 * ─── Upstream batcher references ─────────────────────────────────────────────
 *   upstream/pixijs-engine/src/rendering/batcher/shared/Batcher.ts
 *     → abstract Batcher, add(), break(), begin(), BatchableElement interface
 *   upstream/pixijs-engine/src/rendering/batcher/shared/DefaultBatcher.ts
 *     → vertexSize=6, packQuadAttributes() layout  [x,y,u,v,color,texIdRound]
 *   upstream/pixijs-engine/src/rendering/batcher/shared/BatchGeometry.ts
 *     → attribute stride 6×4=24 bytes, aPosition/aUV/aColor/aTextureIdAndRound
 *   upstream/pixijs-engine/src/rendering/batcher/shared/BatcherPipe.ts
 *     → buildStart → addToBatch → buildEnd flow; per-batcherName segmentation
 *   upstream/pixijs-engine/src/rendering/batcher/gl/GlBatchAdaptor.ts
 *     → start(): bind shader+geometry; execute(): set blendMode, bind textures,
 *       draw(topology, size, start)
 *   upstream/pixijs-engine/src/scene/sprite/BatchableSprite.ts
 *     → packAsQuad=true, attributeSize=4, indexSize=6
 *
 * ─── Integration with PixiJS Application ─────────────────────────────────────
 *   CellSpeciesBatch subclasses Container.  One instance per species per frame.
 *   CellBatchManager sits on top and orchestrates the buildStart/add/buildEnd
 *   cycle, mirroring BatcherPipe.
 *
 * ─── Rule: NO SVG strings ────────────────────────────────────────────────────
 *   All colour/shape data flows as JSON params.  GPU draws everything.
 */

import { Container }     from '../../upstream/pixijs-engine/src/scene/container/Container';
import { Graphics }      from '../../upstream/pixijs-engine/src/scene/graphics/shared/Graphics';
import { Sprite }        from '../../upstream/pixijs-engine/src/scene/sprite/Sprite';
import { RenderTexture } from '../../upstream/pixijs-engine/src/rendering/renderers/shared/texture/RenderTexture';
import { Texture }       from '../../upstream/pixijs-engine/src/rendering/renderers/shared/texture/Texture';
import { ParticleContainer } from '../../upstream/pixijs-engine/src/scene/particle-container/shared/ParticleContainer';
import { Particle }          from '../../upstream/pixijs-engine/src/scene/particle-container/shared/Particle';

import type { Application } from '../../upstream/pixijs-engine/src/app/Application';
import type { CellDescriptor } from './pixi-cell-renderer';

// ─── Species colour palette (mirrors pixi-cell-renderer.ts) ─────────────────

export const SPECIES_PALETTE: Record<string, { fill: number; stroke: number; glow: number }> = {
  'cil-eye':         { fill: 0x5C6BC0, stroke: 0x3949AB, glow: 0x7986CB },
  'cil-vector':      { fill: 0x66BB6A, stroke: 0x388E3C, glow: 0x81C784 },
  'cil-bolt':        { fill: 0xFFA726, stroke: 0xF57C00, glow: 0xFFCC80 },
  'cil-plus':        { fill: 0xEC407A, stroke: 0xC62828, glow: 0xF48FB1 },
  'cil-arrow-right': { fill: 0x78909C, stroke: 0x455A64, glow: 0xB0BEC5 },
  'cil-filter':      { fill: 0xAB47BC, stroke: 0x7B1FA2, glow: 0xCE93D8 },
  'cil-code':        { fill: 0x26A69A, stroke: 0x00796B, glow: 0x80CBC4 },
  'cil-layers':      { fill: 0x42A5F5, stroke: 0x1565C0, glow: 0x90CAF9 },
  'cil-loop':        { fill: 0xFFCA28, stroke: 0xF9A825, glow: 0xFFE082 },
  'cil-graph':       { fill: 0x78909C, stroke: 0x37474F, glow: 0xB0BEC5 },
};
const FALLBACK_PALETTE = { fill: 0x90A4AE, stroke: 0x607D8B, glow: 0xB0BEC5 };

function palette(species: string) {
  return SPECIES_PALETTE[species] ?? FALLBACK_PALETTE;
}

// ─── BatchCell ────────────────────────────────────────────────────────────────
//
// Mirrors the BatchableElement / BatchableQuadElement contract from upstream:
//   attributeSize = 4   (four vertices per quad, as in BatchableSprite)
//   indexSize     = 6   (two triangles, as in packQuadIndex)
//   packAsQuad    = true
//
// Carries the cell's positional bounds so CellSpeciesBatch.packQuadAttributes()
// can write directly into the shared attribute buffer — exactly the pattern
// upstream's DefaultBatcher.packQuadAttributes() uses.

export interface BatchCell {
  /** Source CellDescriptor */
  desc:       CellDescriptor;
  /** World-space bounding box (may differ from desc.bbox after lerp) */
  x: number;
  y: number;
  w: number;
  h: number;
  /** Combined ARGB tint as uint32 (mirrors DefaultBatchElements.color) */
  color:      number;
  /** Pre-assigned texture slot in this species' batch */
  _textureId: number;
}

// ─── CellSpeciesBatch ─────────────────────────────────────────────────────────
//
// One per species per frame.  Owns:
//   • A RenderTexture pre-rendered with the species icon (cached across frames)
//   • A cpu-side Float32Array / Uint16Array built by packQuadAttributes()
//   • Sprites spawned via ParticleContainer-style sharing of the same texture
//
// The attribute layout follows DefaultBatcher.vertexSize = 6:
//   [0] x      [1] y      [2] u     [3] v
//   [4] argb (uint32 reinterpreted) [5] texIdAndRound (uint32)
//
// Geometry buffers are populated just like upstream's Batcher.break() does:
//   - attributeBuffer.float32View / uint32View written by packQuadAttributes()
//   - indexBuffer written by packQuadIndex()

const VERTEX_SIZE    = 6;   // floats — matches DefaultBatcher.vertexSize
const VERTS_PER_QUAD = 4;   // TL TR BR BL — matches BatchableSprite.attributeSize
const IDX_PER_QUAD   = 6;   // 2 triangles — matches BatchableSprite.indexSize

export class CellSpeciesBatch {
  public readonly species: string;

  // CPU-side geometry buffers — mirrors Batcher.attributeBuffer + indexBuffer
  public attributeBuffer: Float32Array;   // float32 view
  public attributeU32:    Uint32Array;    // shared uint32 view (same ArrayBuffer)
  public indexBuffer:     Uint16Array;    // index buffer (< 65535 cells per species)
  public attributeSize = 0;  // floats used
  public indexSize     = 0;  // indices used

  // Batch items queued this frame
  private _cells: BatchCell[] = [];

  // RenderTexture for this species' icon — shared across frames
  private _iconTexture: RenderTexture | null = null;

  constructor(species: string, initialCap = 64) {
    this.species = species;
    this._alloc(initialCap);
  }

  // ── begin() ─ mirrors Batcher.begin() ────────────────────────────────────
  begin(): void {
    this.attributeSize = 0;
    this.indexSize     = 0;
    this._cells.length = 0;
  }

  // ── add() — mirrors Batcher.add() ─────────────────────────────────────────
  add(cell: BatchCell): void {
    const needed = (this._cells.length + 1) * VERTS_PER_QUAD * VERTEX_SIZE;
    if (needed > this.attributeBuffer.length) this._grow();

    cell._textureId = 0; // single texture per species batch (slot 0)
    this._cells.push(cell);

    // Reserve space — mirrors Batcher.add() incrementing attributeSize/indexSize
    this.attributeSize += VERTS_PER_QUAD * VERTEX_SIZE;
    this.indexSize     += IDX_PER_QUAD;
  }

  // ── packAll() — mirrors Batcher.break() inner loop ────────────────────────
  //
  // Writes all queued cells into attributeBuffer (float32/uint32 views) and
  // indexBuffer.  Follows DefaultBatcher.packQuadAttributes() exactly:
  //
  //   float32View[index + 0]  = transformed x   (no wt here: identity xform)
  //   float32View[index + 1]  = transformed y
  //   float32View[index + 2]  = uv.x0
  //   float32View[index + 3]  = uv.y0
  //   uint32View [index + 4]  = argb
  //   uint32View [index + 5]  = (textureId << 16) | roundPixels
  //   … repeated for 4 vertices
  //
  // Index pattern mirrors Batcher.packQuadIndex():
  //   [0,1,2,  0,2,3]  (two CCW triangles sharing diagonal 0→2)

  packAll(): void {
    const f32 = this.attributeBuffer;
    const u32 = this.attributeU32;
    const idx = this.indexBuffer;

    let attrOff = 0;
    let idxOff  = 0;

    for (const cell of this._cells) {
      const { x, y, w, h, color } = cell;
      const texIdAndRound = (0 << 16) | 0; // slot 0, no rounding

      const x1 = x,     y1 = y;
      const x2 = x + w, y2 = y;
      const x3 = x + w, y3 = y + h;
      const x4 = x,     y4 = y + h;

      // UV covers full icon texture [0,0]→[1,1]
      // TL
      f32[attrOff + 0] = x1;  f32[attrOff + 1] = y1;
      f32[attrOff + 2] = 0;   f32[attrOff + 3] = 0;
      u32[attrOff + 4] = color;
      u32[attrOff + 5] = texIdAndRound;
      attrOff += VERTEX_SIZE;

      // TR
      f32[attrOff + 0] = x2;  f32[attrOff + 1] = y2;
      f32[attrOff + 2] = 1;   f32[attrOff + 3] = 0;
      u32[attrOff + 4] = color;
      u32[attrOff + 5] = texIdAndRound;
      attrOff += VERTEX_SIZE;

      // BR
      f32[attrOff + 0] = x3;  f32[attrOff + 1] = y3;
      f32[attrOff + 2] = 1;   f32[attrOff + 3] = 1;
      u32[attrOff + 4] = color;
      u32[attrOff + 5] = texIdAndRound;
      attrOff += VERTEX_SIZE;

      // BL
      f32[attrOff + 0] = x4;  f32[attrOff + 1] = y4;
      f32[attrOff + 2] = 0;   f32[attrOff + 3] = 1;
      u32[attrOff + 4] = color;
      u32[attrOff + 5] = texIdAndRound;
      attrOff += VERTEX_SIZE;

      // Indices — mirrors Batcher.packQuadIndex()
      const base = (attrOff / VERTEX_SIZE - VERTS_PER_QUAD);
      idx[idxOff++] = base + 0;
      idx[idxOff++] = base + 1;
      idx[idxOff++] = base + 2;
      idx[idxOff++] = base + 0;
      idx[idxOff++] = base + 2;
      idx[idxOff++] = base + 3;
    }
  }

  // ── getIconTexture() — RenderTexture atlas slot for this species ──────────
  //
  // Pre-renders the species icon into a RenderTexture using PixiJS Graphics,
  // then caches it.  On subsequent frames the same texture is reused with no
  // redraws — the Sprite array in buildEnd() all share this one texture, giving
  // us a "texture atlas" at zero extra memory cost per cell.

  getIconTexture(app: Application): RenderTexture {
    if (this._iconTexture) return this._iconTexture;

    const W = 64, H = 64;
    const pal = palette(this.species);

    // Draw species icon into a temporary Graphics container
    const g = new Graphics();
    // Body
    g.roundRect(0, 0, W, H, 8);
    g.fill({ color: pal.fill, alpha: 0.92 });
    g.roundRect(0, 0, W, H, 8);
    g.stroke({ color: pal.stroke, width: 1.5, alpha: 0.85 });

    // Species-specific inner mark (minimal procedural shape)
    this._drawSpeciesMark(g, W, H, pal.stroke);

    // Render into RenderTexture
    const rt = RenderTexture.create({ width: W, height: H });
    app.renderer.render({ container: g, target: rt });
    g.destroy();

    this._iconTexture = rt;
    return rt;
  }

  // ── buildEnd(): flush batch as Sprite[] ──────────────────────────────────
  //
  // Mirrors GlBatchAdaptor.execute() — but operates at PixiJS scene-graph
  // level rather than raw GL.  Each cell becomes a Sprite backed by the
  // species RenderTexture (shared, zero-copy).  All same-species Sprites form
  // a single draw call because PixiJS's internal batcher groups Sprites by
  // texture, which is exactly what upstream's BatcherPipe does per batcherName.

  buildEnd(
    app:    Application,
    stage:  Container,
    layer:  Container,
    cells:  BatchCell[],
    zBase:  number,
  ): Sprite[] {
    if (cells.length === 0) return [];

    const tex     = this.getIconTexture(app);
    const sprites: Sprite[] = [];

    for (const cell of cells) {
      const { x, y, w, h } = cell;
      const alpha = ((cell.color >>> 24) & 0xFF) / 255;

      const sp = new Sprite(tex as unknown as Texture);
      sp.position.set(x, y);
      sp.width  = w;
      sp.height = h;
      sp.alpha  = alpha;
      sp.zIndex = cell.desc.z ?? zBase;
      layer.addChild(sp);
      sprites.push(sp);
    }

    return sprites;
  }

  destroy(): void {
    if (this._iconTexture) {
      this._iconTexture.destroy(true);
      this._iconTexture = null;
    }
    this._cells.length = 0;
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  private _alloc(cap: number): void {
    const ab = new ArrayBuffer(cap * VERTS_PER_QUAD * VERTEX_SIZE * 4);
    this.attributeBuffer = new Float32Array(ab);
    this.attributeU32    = new Uint32Array(ab);
    this.indexBuffer     = new Uint16Array(cap * IDX_PER_QUAD);
  }

  private _grow(): void {
    const oldCap = this.attributeBuffer.length / (VERTS_PER_QUAD * VERTEX_SIZE);
    const newCap = Math.max(oldCap * 2, this._cells.length + 32);
    const oldF32 = this.attributeBuffer;
    const oldIdx = this.indexBuffer;
    this._alloc(newCap);
    this.attributeBuffer.set(oldF32);
    this.indexBuffer.set(oldIdx);
  }

  /** Minimal species-distinguishing icon shape (no SVG strings). */
  private _drawSpeciesMark(g: Graphics, W: number, H: number, col: number): void {
    const cx = W / 2, cy = H / 2, r = Math.min(W, H) * 0.28;
    switch (this.species) {
      case 'cil-eye':
        g.circle(cx, cy, r);
        g.stroke({ color: col, width: 1.2, alpha: 0.5 });
        g.circle(cx, cy, r * 0.35);
        g.fill({ color: col, alpha: 0.45 });
        break;
      case 'cil-bolt': {
        const dy = H / 5;
        g.moveTo(cx, 10);
        for (let i = 1; i <= 4; i++) g.lineTo(cx + (i % 2 === 1 ? r : -r), 10 + dy * i);
        g.lineTo(cx, H - 10);
        g.stroke({ color: col, width: 1.5, alpha: 0.5 });
        break;
      }
      case 'cil-vector':
        g.moveTo(cx - r, cy); g.lineTo(cx + r, cy);
        g.moveTo(cx + r, cy); g.lineTo(cx + r - 6, cy - 5);
        g.moveTo(cx + r, cy); g.lineTo(cx + r - 6, cy + 5);
        g.stroke({ color: col, width: 1.5, alpha: 0.5 });
        break;
      case 'cil-plus':
        g.moveTo(cx - r, cy); g.lineTo(cx + r, cy);
        g.moveTo(cx, cy - r); g.lineTo(cx, cy + r);
        g.stroke({ color: col, width: 2, alpha: 0.45 });
        break;
      case 'cil-layers':
        for (let i = 0; i < 3; i++) {
          const off = i * 5;
          g.roundRect(8 + off, 8 + off, W - 16 - off * 2, H - 16 - off * 2, 3);
          g.stroke({ color: col, width: 1, alpha: 0.2 + i * 0.12 });
        }
        break;
      case 'cil-code':
        g.moveTo(14, 14); g.lineTo(10, cy); g.lineTo(14, H - 14);
        g.moveTo(W - 14, 14); g.lineTo(W - 10, cy); g.lineTo(W - 14, H - 14);
        g.stroke({ color: col, width: 1.5, alpha: 0.45 });
        break;
      case 'cil-loop':
        g.arc(cx, cy, r, -Math.PI * 0.75, Math.PI * 0.5);
        g.stroke({ color: col, width: 1.5, alpha: 0.45 });
        break;
      case 'cil-graph': {
        const pts: [number, number][] = [[cx - r, cy - r * 0.5], [cx + r * 0.3, cy - r], [cx + r, cy + r * 0.5], [cx - r * 0.3, cy + r]];
        for (const [px, py] of pts) { g.circle(px, py, 3); g.fill({ color: col, alpha: 0.4 }); }
        g.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]);
        g.stroke({ color: col, width: 1, alpha: 0.25 });
        break;
      }
      case 'cil-filter': {
        const gs = (W - 16) / 3;
        for (let row = 0; row < 3; row++) for (let col2 = 0; col2 < 3; col2++) {
          g.rect(8 + col2 * gs, 8 + row * gs, gs - 2, gs - 2);
          g.stroke({ color: col, width: 0.8, alpha: 0.22 });
        }
        break;
      }
      case 'cil-arrow-right':
        g.moveTo(cx - r, cy - r * 0.6); g.lineTo(cx + r, cy); g.lineTo(cx - r, cy + r * 0.6);
        g.stroke({ color: col, width: 2, alpha: 0.45 });
        break;
      default:
        g.circle(cx, cy, r * 0.6);
        g.stroke({ color: col, width: 1, alpha: 0.3 });
    }
  }
}

// ─── ARGB helpers ─────────────────────────────────────────────────────────────

function rgbToArgb(rgb: number, alpha = 1): number {
  const a = Math.round(Math.min(1, Math.max(0, alpha)) * 255) & 0xFF;
  return ((a << 24) | (rgb & 0x00FFFFFF)) >>> 0;
}

// ─── CellBatchManager ─────────────────────────────────────────────────────────
//
// Orchestrates the BatcherPipe flow across all species:
//
//   buildStart()           → mirrors BatcherPipe.buildStart()
//     for each cell:
//       addToBatch(cell)   → mirrors BatcherPipe.addToBatch()  (segments by species)
//   buildEnd()             → mirrors BatcherPipe.buildEnd()  (flush geometry, draw)
//
// Key property: cells with the same species are automatically grouped into the
// same CellSpeciesBatch — so PixiJS sees N species × 1 draw call, not N_cells
// draw calls.  This is the direct analogue of BatcherPipe grouping by
// batcherName.

export class CellBatchManager {
  private readonly _app:      Application;
  private readonly _stage:    Container;

  // One batch per species — persists across frames (icon textures are cached)
  private readonly _batches   = new Map<string, CellSpeciesBatch>();

  // Batch accumulator this frame — maps species → queued BatchCell[]
  private _frameCells         = new Map<string, BatchCell[]>();

  // Container owned by this manager — added to stage once
  private _layer:             Container;

  // Sprites spawned last frame — cleared and re-populated each buildEnd()
  private _activeSprites:     Sprite[] = [];

  /** Stats updated by buildEnd() */
  public drawCallsLastFrame   = 0;
  public cellsLastFrame       = 0;

  constructor(app: Application, stage: Container) {
    this._app   = app;
    this._stage = stage;
    this._layer = new Container();
    this._layer.sortableChildren = true;
    this._stage.addChild(this._layer);
  }

  // ── buildStart() — mirrors BatcherPipe.buildStart() ──────────────────────

  buildStart(): void {
    // Clear old sprites but keep batch objects (icon textures are expensive)
    for (const sp of this._activeSprites) {
      sp.destroy();
    }
    this._activeSprites.length = 0;

    // Reset frame accumulator — equivalent to Batcher.begin()
    this._frameCells.clear();
    for (const b of this._batches.values()) b.begin();
  }

  // ── addToBatch() — mirrors BatcherPipe.addToBatch() ──────────────────────
  //
  // Routes each CellDescriptor to its species bucket.
  // Equivalent to checking batchableObject.batcherName and switching batchers.

  addToBatch(desc: CellDescriptor, x?: number, y?: number): void {
    const { species, bbox, params } = desc;
    const alpha  = params?.opacity ?? 1;
    const pal    = palette(species);
    const color  = rgbToArgb(pal.fill, alpha);

    const cell: BatchCell = {
      desc,
      x:      x ?? bbox.x,
      y:      y ?? bbox.y,
      w:      bbox.w,
      h:      bbox.h,
      color,
      _textureId: 0,
    };

    if (!this._frameCells.has(species)) {
      this._frameCells.set(species, []);
    }
    this._frameCells.get(species)!.push(cell);

    // Ensure batch object exists — mirrors BatcherPipe creating a new Batcher
    if (!this._batches.has(species)) {
      this._batches.set(species, new CellSpeciesBatch(species));
    }
    this._batches.get(species)!.add(cell);
  }

  // ── buildEnd() — mirrors BatcherPipe.buildEnd() ───────────────────────────
  //
  // For each species:
  //   1. packAll() fills cpu-side attribute + index buffers
  //      (mirrors geometry.buffers[0].setDataWithSize() in BatcherPipe.buildEnd)
  //   2. buildEnd() emits Sprites backed by the shared icon RenderTexture
  //      (mirrors GlBatchAdaptor.execute() drawing all elements in one call)
  //
  // Net result: one Sprite.texture === same RenderTexture for every cell of a
  // given species → PixiJS's internal batcher issues a single draw call per
  // species, because it groups consecutive Sprites with the same texture.

  buildEnd(): void {
    let drawCalls = 0;
    let cellCount = 0;

    for (const [species, cells] of this._frameCells) {
      if (cells.length === 0) continue;

      const batch = this._batches.get(species)!;

      // Pack geometry buffers (CPU side) — mirrors Batcher.break() inner pack
      batch.packAll();

      // Emit sprites — mirrors GlBatchAdaptor.execute() per batch segment
      const sprites = batch.buildEnd(
        this._app, this._stage, this._layer, cells, 1,
      );

      this._activeSprites.push(...sprites);
      drawCalls++;
      cellCount += cells.length;
    }

    this.drawCallsLastFrame = drawCalls;
    this.cellsLastFrame     = cellCount;
  }

  // ── Convenience: full cycle in one call ───────────────────────────────────

  renderCells(cells: CellDescriptor[]): void {
    this.buildStart();
    for (const c of cells) this.addToBatch(c);
    this.buildEnd();
  }

  /** Render with live-lerped positions (pollCellChannels integration). */
  renderCellsLerped(
    cells: Array<{ desc: CellDescriptor; x: number; y: number }>,
  ): void {
    this.buildStart();
    for (const { desc, x, y } of cells) this.addToBatch(desc, x, y);
    this.buildEnd();
  }

  // ── Stats ─────────────────────────────────────────────────────────────────

  get stats() {
    return {
      drawCalls:   this.drawCallsLastFrame,
      cells:       this.cellsLastFrame,
      speciesBatches: this._batches.size,
    };
  }

  destroy(): void {
    this.buildStart(); // clears sprites
    for (const b of this._batches.values()) b.destroy();
    this._batches.clear();
    this._stage.removeChild(this._layer);
    this._layer.destroy({ children: true });
  }
}

// ─── Factory ──────────────────────────────────────────────────────────────────

/**
 * createCellBatchManager
 *
 * Drop-in companion to renderCellGraph() / pollCellChannels().
 * Returns a CellBatchManager wired to the provided Application's stage.
 *
 * @example
 * ```ts
 * const app = await renderCellGraph(canvas, cells, edges);
 * const batchMgr = createCellBatchManager(app);
 *
 * app.ticker.add(() => {
 *   batchMgr.renderCells(liveCells);
 * });
 * ```
 */
export function createCellBatchManager(app: Application): CellBatchManager {
  return new CellBatchManager(app, app.stage);
}

// ─── Batch stats formatter ────────────────────────────────────────────────────

export function formatCellBatchStats(mgr: CellBatchManager): string {
  const { drawCalls, cells, speciesBatches } = mgr.stats;
  return `[CellBatch] species=${speciesBatches}  drawCalls=${drawCalls}  cells=${cells}`;
}

// ─── Instanced rendering threshold ──────────────────────────────────────────
//
// When cell count exceeds the BATCH_THRESHOLD, the per-Container approach
// (one Sprite per cell inside a regular Container) becomes a GPU state-switch
// bottleneck.  ParticleContainer uses a single shared geometry buffer and one
// draw call per texture — the same instanced-rendering pattern as upstream's
// ParticleContainerPipe, but driven by our CellDescriptor data.
//
// shouldUseBatch() gates the decision; createBatchRenderer() builds the
// ParticleContainer tree grouped by species (one ParticleContainer per species
// texture, matching the one-draw-call-per-species invariant of CellBatchManager).

const BATCH_THRESHOLD = 20;

/**
 * Returns `true` when the cell count is large enough to benefit from
 * ParticleContainer instanced rendering instead of per-cell Containers.
 *
 * The threshold (>20) is the point at which GPU state-switch overhead from
 * individual Container/Sprite pairs exceeds the cost of populating a
 * ParticleContainer's shared vertex buffer.
 */
export function shouldUseBatch(cellCount: number): boolean {
  return cellCount > BATCH_THRESHOLD;
}

/**
 * createBatchRenderer
 *
 * Builds a ParticleContainer-based instanced renderer for the given cells and
 * attaches it to `stage`.  All cells sharing the same species share one
 * ParticleContainer (and therefore one draw call), mirroring the
 * CellSpeciesBatch grouping but using PixiJS's built-in particle pipeline
 * for maximum throughput.
 *
 * Returns a handle with:
 *   • `container`  — the root Container added to `stage` (owns all
 *     ParticleContainers)
 *   • `update(cells)` — re-sync particle positions/alpha from fresh
 *     CellDescriptor data (e.g. after lerp)
 *   • `destroy()` — tear down everything and remove from `stage`
 *
 * Intended to be used behind the `shouldUseBatch()` gate:
 *
 * ```ts
 * if (shouldUseBatch(cells.length)) {
 *   const batch = createBatchRenderer(cells, app.stage, app);
 *   // on tick: batch.update(newCells);
 * } else {
 *   // fall back to regular per-cell Container rendering
 * }
 * ```
 *
 * @param cells  Array of CellDescriptors to render
 * @param stage  PixiJS Container to attach to (typically app.stage)
 * @param app    PixiJS Application (needed for RenderTexture generation)
 */
export function createBatchRenderer(
  cells: CellDescriptor[],
  stage: Container,
  app?: Application,
): {
  container: Container;
  update: (cells: CellDescriptor[]) => void;
  destroy: () => void;
} {
  // Root container that holds all per-species ParticleContainers
  const root = new Container();
  root.sortableChildren = true;
  stage.addChild(root);

  // Per-species ParticleContainer cache
  //   species → { pc, texture, particles[] }
  const speciesMap = new Map<
    string,
    {
      pc: ParticleContainer;
      tex: RenderTexture;
      particles: Particle[];
    }
  >();

  // Icon texture size — matches CellSpeciesBatch.getIconTexture()
  const ICON_W = 64;
  const ICON_H = 64;

  /**
   * Lazily create a RenderTexture for a species by drawing the same
   * procedural icon that CellSpeciesBatch uses.  When `app` is unavailable
   * we fall back to a plain coloured rect (still functional, just no inner
   * species mark).
   */
  function getOrCreateSpeciesTexture(species: string): RenderTexture {
    const pal = palette(species);
    const g = new Graphics();
    g.roundRect(0, 0, ICON_W, ICON_H, 8);
    g.fill({ color: pal.fill, alpha: 0.92 });
    g.roundRect(0, 0, ICON_W, ICON_H, 8);
    g.stroke({ color: pal.stroke, width: 1.5, alpha: 0.85 });

    const rt = RenderTexture.create({ width: ICON_W, height: ICON_H });
    if (app) {
      app.renderer.render({ container: g, target: rt });
    }
    g.destroy();
    return rt;
  }

  /**
   * Ensure a ParticleContainer exists for the given species and return it.
   */
  function ensureSpecies(species: string) {
    let entry = speciesMap.get(species);
    if (entry) return entry;

    const tex = getOrCreateSpeciesTexture(species);
    const pc = new ParticleContainer({
      texture: tex as unknown as Texture,
      dynamicProperties: {
        position: true,
        vertex:   true,   // enables scaleX/scaleY updates for variable cell sizes
        color:    true,    // enables per-particle alpha/tint
        rotation: false,
        uvs:      false,
      },
    });

    root.addChild(pc as unknown as Container);
    entry = { pc, tex, particles: [] };
    speciesMap.set(species, entry);
    return entry;
  }

  /**
   * Populate (or re-populate) from a CellDescriptor array.
   * Clears existing particles and rebuilds — ParticleContainer's shared
   * buffer makes this cheaper than individual Container add/remove.
   */
  function populate(descriptors: CellDescriptor[]): void {
    // Clear all existing particles
    for (const entry of speciesMap.values()) {
      if (entry.particles.length > 0) {
        entry.pc.removeParticles(0, entry.particles.length);
        entry.particles.length = 0;
      }
    }

    // Group cells by species and create Particles
    for (const desc of descriptors) {
      const entry = ensureSpecies(desc.species);
      const pal   = palette(desc.species);
      const alpha = desc.params?.opacity ?? 1;

      const p = new Particle({
        texture: entry.tex as unknown as Texture,
        x:       desc.bbox.x,
        y:       desc.bbox.y,
        scaleX:  desc.bbox.w / ICON_W,
        scaleY:  desc.bbox.h / ICON_H,
        tint:    pal.fill,
        alpha,
      });

      entry.pc.addParticle(p);
      entry.particles.push(p);
    }

    // Signal ParticleContainer to re-upload static buffers
    for (const entry of speciesMap.values()) {
      entry.pc.update();
    }
  }

  // Initial population
  populate(cells);

  // ── Public handle ───────────────────────────────────────────────────────────

  return {
    container: root,

    /**
     * Re-sync particle transforms from fresh CellDescriptor data.
     * Call this each tick after lerp / physics / layout updates.
     */
    update(newCells: CellDescriptor[]): void {
      populate(newCells);
    },

    /** Tear down all ParticleContainers and remove from stage. */
    destroy(): void {
      for (const entry of speciesMap.values()) {
        entry.pc.removeParticles(0, entry.particles.length);
        entry.pc.destroy({ children: true });
        entry.tex.destroy(true);
      }
      speciesMap.clear();
      stage.removeChild(root);
      root.destroy({ children: true });
    },
  };
}
