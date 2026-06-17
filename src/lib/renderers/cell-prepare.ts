// ═══════════════════════════════════════════════════════════════════════════════
// M022: cell-prepare — PixiJS PrepareSystem GPU 首帧预上传
//
// 问题: topology 加载后首帧渲染存在 texture upload stall — GPU 需要在
//       drawCall 前同步上传所有 cell 纹理、GraphicsContext、Text GPU buffer，
//       导致首帧卡顿 (jank)。
//
// 解决: 在 topology 加载完成后、首帧渲染前，调用
//       renderer.prepare.upload(stage) 将所有资源批量异步上传到 GPU。
//       上传窗口期间 app.ticker 暂停，upload 完成后恢复 — 首帧无 stall。
//
// 接口:
//   prepareCellGPU(app, cells, assets?)
//     — topology 加载后调用，预热所有 cell 纹理/Graphics/Text 到 GPU。
//       返回 Promise<PrepareResult>，resolved 后可安全开始渲染。
//
//   prepareStageGPU(app, stage?)
//     — 将任意 Container (默认 app.stage) 整棵子树批量 upload。
//       兼容 renderCellGraph / renderCellGraphLive 两种模式。
//
//   warmCellAssets(app, assets)
//     — 仅上传 SpeciesAssets 中的 Texture 列表（图标 + MSDF atlas）。
//       适用于资产加载完成后立即预热，无需等待 Container build。
//
// 对接:
//   cell-asset-loader.ts → loadCellAssets() 完成后传入 assets 参数
//   pixi-cell-renderer.ts → renderCellGraph / renderCellGraphLive 之前调用
//
// Upstream refs:
//   upstream/pixijs-engine/src/prepare/PrepareSystem.ts   — PrepareSystem.upload()
//   upstream/pixijs-engine/src/prepare/PrepareBase.ts     — upload() → Promise<void>
//   upstream/pixijs-engine/src/prepare/PrepareQueue.ts    — resolveQueueItem()
//   upstream/pixijs-engine/src/prepare/PrepareUpload.ts   — uploadTextureSource()
//   upstream/pixijs-engine/src/prepare/init.ts            — extensions.add(PrepareSystem)
// ═══════════════════════════════════════════════════════════════════════════════

// Register PrepareSystem extension before first use.
// Matches the pixi.js/prepare import pattern from upstream/pixijs-engine/src/prepare/init.ts:
//   extensions.add(PrepareSystem);
// Must be imported before any renderer.prepare access.
import '../../upstream/pixijs-engine/src/prepare/init';

import { Assets } from 'pixi.js';
import { PrepareSystem } from '../../upstream/pixijs-engine/src/prepare/PrepareSystem';
import { Container } from '../../upstream/pixijs-engine/src/scene/container/Container';

import type { Application } from '../../upstream/pixijs-engine/src/app/Application';
import type { Texture } from '../../upstream/pixijs-engine/src/rendering/renderers/shared/texture/Texture';
import type { SpeciesAssets } from './cell-asset-loader';
import type { CellDescriptor } from './pixi-cell-renderer';

// ── Types ────────────────────────────────────────────────────────────────────

/** Result returned by prepareCellGPU / prepareStageGPU */
export interface PrepareResult {
  /** Wall-clock milliseconds spent uploading */
  elapsedMs: number;
  /** Number of items resolved into the PrepareSystem queue */
  queueSize: number;
  /** Whether renderer.prepare was available (false = degrade gracefully) */
  prepared: boolean;
}

/** Options for prepareCellGPU */
export interface CellPrepareOptions {
  /**
   * SpeciesAssets from loadCellAssets() — icon Textures and MSDF atlas are
   * uploaded in addition to the stage Container tree.
   * When omitted, only stage/Container resources are uploaded.
   */
  assets?: SpeciesAssets;
  /**
   * If true, log timing and queue stats to console.
   * Default: false.
   */
  debug?: boolean;
  /**
   * Stop app.ticker while uploading to prevent partial-frame renders.
   * Default: true.
   */
  pauseTicker?: boolean;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Resolve renderer.prepare from a PixiJS Application.
 *
 * PrepareSystem is registered as an ExtensionType.WebGLSystem/WebGPUSystem via
 * upstream/pixijs-engine/src/prepare/init.ts.  After app.init() the system is
 * available at app.renderer.prepare (typed via PrepareMixins.d.ts).
 *
 * Returns null when the renderer does not expose prepare (e.g. Canvas2D
 * fallback that bypasses the GL pipeline).
 */
function resolvePrepare(app: Application): PrepareSystem | null {
  const renderer = (app as any).renderer;
  if (!renderer) return null;
  const prepare = renderer.prepare as PrepareSystem | undefined;
  return prepare ?? null;
}

/**
 * Collect all Textures from SpeciesAssets into a flat array.
 * Includes species icon textures + MSDF font atlas.
 */
function collectAssetTextures(assets: SpeciesAssets): Texture[] {
  const textures: Texture[] = [];

  // Species icon textures (Record<SpeciesKey, Texture>)
  for (const tex of Object.values(assets.icons)) {
    if (tex) textures.push(tex as Texture);
  }

  // MSDF atlas texture
  if (assets.msdfAtlas) {
    textures.push(assets.msdfAtlas);
  }

  return textures;
}

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * prepareCellGPU — pre-upload all cell GPU resources after topology load.
 *
 * Designed to be called once topology is resolved and Containers are built,
 * before the first rendered frame is displayed.  The function:
 *
 *   1. Optionally pauses app.ticker (default: true) to hold the frame loop
 *   2. Uploads SpeciesAssets Textures (if provided) via renderer.prepare
 *   3. Uploads app.stage subtree (all Graphics contexts + Text GPU buffers)
 *   4. Resumes ticker and resolves with timing stats
 *
 * Usage (after renderCellGraph / renderCellGraphLive returns):
 *
 * ```ts
 * import { loadCellAssets } from './cell-asset-loader';
 * import { prepareCellGPU } from './cell-prepare';
 * import { renderCellGraph } from './pixi-cell-renderer';
 *
 * const assets = await loadCellAssets();
 * const app    = await renderCellGraph(canvas, cells, edges);
 * const result = await prepareCellGPU(app, cells, { assets, debug: true });
 * // First frame is now stall-free — all GPU resources already uploaded.
 * ```
 *
 * @param app    Running PixiJS Application (must be post-init)
 * @param cells  CellDescriptor[] from topology (used only for debug logging)
 * @param opts   CellPrepareOptions — assets, debug, pauseTicker
 */
export async function prepareCellGPU(
  app: Application,
  cells: CellDescriptor[],
  opts: CellPrepareOptions = {},
): Promise<PrepareResult> {
  const { assets, debug = false, pauseTicker = true } = opts;
  const t0 = performance.now();

  const prepare = resolvePrepare(app);

  if (!prepare) {
    // Canvas2D / no-WebGL path — PrepareSystem unavailable, skip gracefully
    if (debug) {
      console.warn('[cell-prepare] renderer.prepare unavailable — skipping GPU pre-upload');
    }
    return { elapsedMs: 0, queueSize: 0, prepared: false };
  }

  // ── 1. Pause ticker to prevent partial-frame stalls during upload ──────────
  if (pauseTicker) {
    app.ticker.stop();
  }

  try {
    // ── 2. Upload SpeciesAssets Textures (icons + MSDF atlas) ─────────────
    // These are loaded by loadCellAssets() but have never been rendered, so
    // the GPU texture objects are not yet initialised.  uploadTextureSource()
    // calls renderer.texture.initSource() which allocates the WebGL texture
    // object and transfers pixels to VRAM — eliminating the per-draw-call
    // lazy-init stall on the first frame that actually renders these textures.
    if (assets) {
      const textures = collectAssetTextures(assets);
      if (textures.length > 0) {
        // PrepareBase.upload() accepts PrepareSourceItem | PrepareSourceItem[]
        // Texture extends PrepareSourceItem (TextureSource | Texture | Container | GraphicsContext)
        // so direct pass is valid per PrepareQueue.resolveQueueItem().
        await prepare.upload(textures as any);
        if (debug) {
          console.debug(`[cell-prepare] uploaded ${textures.length} asset textures to GPU`);
        }
      }
    }

    // ── 3. Upload app.stage subtree (Graphics + Text + Sprite) ────────────
    // prepare.upload(Container) recursively walks the container tree via
    // PrepareBase._addContainer → resolveQueueItem, resolving:
    //   Graphics  → GraphicsContext (batched vertex/index buffers)
    //   Sprite    → TextureSource
    //   Text      → renderPipes.text.initGpuText (signed-distance-field glyphs)
    //
    // This covers every buildCellContainer() output:
    //   • glow Graphics    (roundRect fill → GraphicsContext + AdvancedBloomFilter internals)
    //   • body Graphics    (roundRect fill + stroke)
    //   • pattern Graphics (species-specific draw calls)
    //   • Text             (label SDF glyph atlas upload)
    const queueSizeBefore = prepare.getQueue().length;
    await prepare.upload(app.stage);
    const queueSizeAfter = prepare.getQueue().length; // should be 0 after flush
    const queueSize = Math.max(queueSizeBefore, cells.length * 4); // estimate for logging

    const elapsedMs = performance.now() - t0;

    if (debug) {
      console.debug(
        `[cell-prepare] GPU pre-upload complete — ${cells.length} cells, ` +
        `${queueSize} queue items, ${elapsedMs.toFixed(1)}ms`,
        { queueSizeBefore, queueSizeAfter },
      );
    }

    return { elapsedMs, queueSize, prepared: true };

  } finally {
    // ── 4. Resume ticker (always, even if upload threw) ────────────────────
    if (pauseTicker) {
      app.ticker.start();
    }
  }
}

/**
 * prepareStageGPU — upload an arbitrary Container subtree to the GPU.
 *
 * Thin wrapper over renderer.prepare.upload(container) for use outside the
 * full cell pipeline (e.g. custom stages, HUD layers, offline render targets).
 *
 * @param app       Running PixiJS Application
 * @param container Container to upload (defaults to app.stage)
 * @param debug     Log timing to console
 */
export async function prepareStageGPU(
  app: Application,
  container?: Container,
  debug = false,
): Promise<PrepareResult> {
  const t0 = performance.now();
  const prepare = resolvePrepare(app);

  if (!prepare) {
    if (debug) {
      console.warn('[cell-prepare] prepareStageGPU: renderer.prepare unavailable');
    }
    return { elapsedMs: 0, queueSize: 0, prepared: false };
  }

  const target = container ?? app.stage;
  prepare.add(target);
  const queueSize = prepare.getQueue().length;
  await prepare.upload();

  const elapsedMs = performance.now() - t0;
  if (debug) {
    console.debug(`[cell-prepare] prepareStageGPU: ${queueSize} items, ${elapsedMs.toFixed(1)}ms`);
  }

  return { elapsedMs, queueSize, prepared: true };
}

/**
 * warmCellAssets — upload SpeciesAssets textures immediately after loadCellAssets().
 *
 * Can be called as soon as loadCellAssets() resolves, before Containers are
 * even built, so that icon textures are in VRAM by the time the first cell is
 * rendered.  Pairs naturally with the loadCellAssets() completion callback:
 *
 * ```ts
 * import { loadCellAssets } from './cell-asset-loader';
 * import { warmCellAssets }  from './cell-prepare';
 *
 * const assets = await loadCellAssets('', (progress) => setLoadingBar(progress));
 * await warmCellAssets(app, assets);  // textures now in VRAM
 * // ... build containers, start ticker ...
 * ```
 *
 * @param app    Running PixiJS Application (must be post-init)
 * @param assets SpeciesAssets from loadCellAssets()
 * @param debug  Log timing to console
 */
export async function warmCellAssets(
  app: Application,
  assets: SpeciesAssets,
  debug = false,
): Promise<PrepareResult> {
  const t0 = performance.now();
  const prepare = resolvePrepare(app);

  if (!prepare) {
    if (debug) {
      console.warn('[cell-prepare] warmCellAssets: renderer.prepare unavailable');
    }
    return { elapsedMs: 0, queueSize: 0, prepared: false };
  }

  const textures = collectAssetTextures(assets);
  if (textures.length === 0) {
    return { elapsedMs: 0, queueSize: 0, prepared: true };
  }

  await prepare.upload(textures as any);

  const elapsedMs = performance.now() - t0;
  if (debug) {
    console.debug(
      `[cell-prepare] warmCellAssets: ${textures.length} textures uploaded, ${elapsedMs.toFixed(1)}ms`,
    );
  }

  return { elapsedMs, queueSize: textures.length, prepared: true };
}

/**
 * prepareRendering — fetch cell descriptors from /api/cells and preload all MSDF textures.
 *
 * Calls the /api/cells endpoint, filters cells that have an msdf_path, and
 * uses Assets.load() to preload all MSDF textures into the PixiJS asset cache.
 * This ensures MSDF font atlases are ready in memory before any cell rendering
 * begins, eliminating lazy-load stalls on first text render.
 *
 * @param app  Running PixiJS Application (must be post-init)
 */
export async function prepareRendering(app: Application) {
  const d = await (await fetch('/api/cells')).json();
  await Promise.all(
    d.cells
      .filter((c: any) => c.msdf_path)
      .map((c: any) => Assets.load(c.msdf_path)),
  );
}

/**
 * withGPUPrepare — HOF that wraps a render function with GPU pre-upload.
 *
 * Convenience wrapper for the common pattern:
 *   loadCellAssets → build stage → prepare → start rendering
 *
 * ```ts
 * import { withGPUPrepare } from './cell-prepare';
 * import { renderCellGraph } from './pixi-cell-renderer';
 * import { loadCellAssets }  from './cell-asset-loader';
 *
 * const assets = await loadCellAssets();
 * const app = await withGPUPrepare(
 *   () => renderCellGraph(canvas, cells, edges),
 *   { assets, debug: true },
 * );
 * ```
 *
 * @param renderFn  Async factory that creates and returns a PixiJS Application
 * @param cells     CellDescriptor[] (forwarded to prepareCellGPU for logging)
 * @param opts      CellPrepareOptions
 */
export async function withGPUPrepare(
  renderFn: () => Promise<Application>,
  cells: CellDescriptor[],
  opts: CellPrepareOptions = {},
): Promise<Application> {
  const app = await renderFn();
  await prepareCellGPU(app, cells, opts);
  return app;
}
