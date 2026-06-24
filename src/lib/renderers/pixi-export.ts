/**
 * pixi-export.ts — M162: PixiJS 离屏渲染导出 PNG / SVG
 *
 * 替代后端 assemble_final_svg()：
 *   1. 读取 channels/composite_params.json（M154 产出）或从各 cell/edge channel 聚合
 *   2. 用 PixiJS Application 离屏渲染所有 cell + edge
 *   3. 用 renderer.extract.canvas() 导出 Canvas → PNG blob / data‑URL
 *   4. exportAsSVG() 将 Canvas 内嵌为 <image> 元素，保留宽高语义
 *
 * Upstream refs:
 *   upstream/pixijs-engine/src/rendering/renderers/shared/extract/ExtractSystem.ts
 *   src/lib/renderers/pixi-cell-renderer.ts  (renderCellGraph / CellDescriptor / EdgeDescriptor)
 */

import { Application } from '../../../upstream/pixijs-engine/src/app/Application';
import {
  renderCellGraph,
  type CellDescriptor,
  type EdgeDescriptor,
  type ParamsJson,
} from './pixi-cell-renderer';

// ── composite_params.json schema (M154 output) ────────────────────────────────

// [orphan-precise] /**
// [orphan-precise]  * The shape written by M154 into channels/composite_params.json.
// [orphan-precise]  * Falls back to aggregating individual cell/edge channels when file is absent.
// [orphan-precise]  */
export interface CompositeParams {
  width: number;
  height: number;
  background?: string;
  cells: ParamsJson[];
  edges: EdgeRouteJson[];
}

/** Shape of channels/edge/<id>/route.json */
export interface EdgeRouteJson {
  edge_id: string;
  sources: string[];
  targets: string[];
  points: Array<{ x: number; y: number }>;
  z?: number;
  rerouted_epoch?: number;
  advanced?: Record<string, unknown>;
}

// ── PNG export options ────────────────────────────────────────────────────────

export interface ExportOptions {
  /**
   * Output pixel width.  Defaults to composite_params.width or 800.
   */
  width?: number;
  /**
   * Output pixel height.  Defaults to composite_params.height or 600.
   */
  height?: number;
  /**
   * Device pixel ratio for hi‑DPI export.  Default: 1.
   */
  resolution?: number;
  /**
   * JPEG / WebP quality 0‑1.  Ignored for PNG.  Default: 0.92.
   */
  quality?: number;
  /**
   * Canvas background colour (CSS colour string).  Default: transparent.
   */
  background?: string;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/**
 * Convert an EdgeRouteJson into the EdgeDescriptor shape expected by
 * pixi-cell-renderer's renderCellGraph.
 */
function edgeRouteToDescriptor(route: EdgeRouteJson): EdgeDescriptor {
  return {
    edge_id: route.edge_id,
    source: route.sources[0] ?? '',
    target: route.targets[0] ?? '',
    waypoints: route.points,
    z: route.z ?? 1,
  };
}

/**
 * Load composite_params.json from a URL (browser) or resolve it relative to
 * the Vite/Astro dev‑server root.  Returns null when the file cannot be
 * fetched so callers can fall back to per‑channel aggregation.
 */
async function fetchCompositeParams(url: string): Promise<CompositeParams | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return (await res.json()) as CompositeParams;
  } catch {
    return null;
  }
}

/**
 * Aggregate cells from channels/cell/<id>/params.json and
 * edges from channels/edge/<id>/route.json via the dev‑server API.
 *
 * Used as fallback when composite_params.json is absent.
 */
async function aggregateChannels(baseUrl: string): Promise<{
  cells: ParamsJson[];
  edges: EdgeRouteJson[];
  width: number;
  height: number;
  background: string;
}> {
  // Try /api/cells and /api/edges if available (pixi-cell-renderer convention)
  let cells: ParamsJson[] = [];
  let edges: EdgeRouteJson[] = [];

  try {
    const cr = await fetch(`${baseUrl}/api/cells`);
    if (cr.ok) cells = await cr.json();
  } catch { /* optional endpoint */ }

  try {
    const er = await fetch(`${baseUrl}/api/edges`);
    if (er.ok) edges = await er.json();
  } catch { /* optional endpoint */ }

  // Derive canvas bounds from cell bboxes
  let maxX = 800;
  let maxY = 600;
  for (const c of cells) {
    if (c.bbox) {
      maxX = Math.max(maxX, c.bbox.x + c.bbox.w + 40);
      maxY = Math.max(maxY, c.bbox.y + c.bbox.h + 40);
    }
  }

  return { cells, edges, width: maxX, height: maxY, background: '#0d0d0d' };
}

// ── PixiExporter ──────────────────────────────────────────────────────────────

/**
 * PixiExporter — orchestrates offscreen PixiJS rendering and provides
 * `exportAsPNG()` / `exportAsSVG()` methods.
 *
 * @example
 * ```ts
 * const exporter = new PixiExporter();
 * const pngBlob = await exporter.exportAsPNG();
 * const svgString = await exporter.exportAsSVG();
 * exporter.destroy();
 * ```
 */
export class PixiExporter {
  /**
   * URL base for fetching channel JSON files.
   * Default: '' (same origin, served by Astro/Vite dev‑server).
   */
  public channelBaseUrl: string;

  /**
   * Path to composite_params.json relative to channelBaseUrl.
   * M154 writes this; we read it first and fall back to per‑channel
   * aggregation when it is absent.
   */
  public compositeParamsPath: string;

  private _app: Application | null = null;
  private _canvas: HTMLCanvasElement | null = null;

  constructor(options?: {
    channelBaseUrl?: string;
    compositeParamsPath?: string;
  }) {
    this.channelBaseUrl = options?.channelBaseUrl ?? '';
    this.compositeParamsPath =
      options?.compositeParamsPath ?? '/channels/composite_params.json';
  }

  // ── Lifecycle ───────────────────────────────────────────────────────────────

  /**
   * Initialise an offscreen PixiJS application and render all cells + edges.
   * Must be called (and awaited) before export methods.
   */
  async init(exportOptions: ExportOptions = {}): Promise<void> {
    // 1. Load composite params
    const compositeUrl = `${this.channelBaseUrl}${this.compositeParamsPath}`;
    let composite = await fetchCompositeParams(compositeUrl);

    let cells: ParamsJson[];
    let edges: EdgeRouteJson[];
    let canvasWidth: number;
    let canvasHeight: number;
    let background: string;

    if (composite) {
      cells = composite.cells;
      edges = composite.edges;
      canvasWidth = composite.width;
      canvasHeight = composite.height;
      background = composite.background ?? '#0d0d0d';
    } else {
      // Fallback: aggregate per‑channel JSON files
      const agg = await aggregateChannels(this.channelBaseUrl);
      cells = agg.cells;
      edges = agg.edges;
      canvasWidth = agg.width;
      canvasHeight = agg.height;
      background = agg.background;
    }

    // Override with caller‑supplied dimensions
    const width = exportOptions.width ?? canvasWidth;
    const height = exportOptions.height ?? canvasHeight;
    const bg = exportOptions.background ?? background;

    // 2. Create an offscreen canvas
    const canvas = document.createElement('canvas');
    canvas.width = width * (exportOptions.resolution ?? 1);
    canvas.height = height * (exportOptions.resolution ?? 1);
    this._canvas = canvas;

    // 3. Convert EdgeRouteJson → EdgeDescriptor
    const edgeDescriptors: EdgeDescriptor[] = edges.map(edgeRouteToDescriptor);

    // 4. Convert ParamsJson → CellDescriptor (ParamsJson is a superset)
    const cellDescriptors: CellDescriptor[] = cells.map((p) => ({
      cell_id: p.cell_id,
      species: p.species,
      bbox: p.bbox,
      z: p.z,
      opacity: p.opacity,
      fill_color: p.fill_color,
      stroke_color: p.stroke_color,
      label: p.label,
      font_size: p.font_size,
      shadow: p.shadow,
      species_params: p.species_params,
      epoch: p.epoch,
    }));

    // 5. Render via pixi-cell-renderer (handles Application init internally)
    this._app = await renderCellGraph(canvas, cellDescriptors, edgeDescriptors);

    // Apply background fill on the renderer's canvas context as a best-effort
    // (PixiJS transparent background leaves compositing to the caller)
    if (bg && bg !== 'transparent') {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.save();
        ctx.globalCompositeOperation = 'destination-over';
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
      }
    }
  }

  /**
   * Destroy the PixiJS application and release GPU resources.
   */
  destroy(): void {
    if (this._app) {
      this._app.destroy(false, { children: true, texture: true });
      this._app = null;
    }
    this._canvas = null;
  }

  // ── Export methods ──────────────────────────────────────────────────────────

  /**
   * Export the rendered scene as a PNG Blob.
   *
   * Uses `renderer.extract.canvas()` when the PixiJS renderer exposes an
   * `extract` system; falls back to `HTMLCanvasElement.toBlob()` otherwise.
   *
   * @returns PNG Blob ready to be saved or uploaded.
   */
  async exportAsPNG(quality = 1): Promise<Blob> {
    const canvas = this._resolveCanvas();

    return new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(
        (blob) => {
          if (blob) resolve(blob);
          else reject(new Error('pixi-export: canvas.toBlob() returned null'));
        },
        'image/png',
        quality,
      );
    });
  }

  /**
   * Export the rendered scene as a PNG data‑URL string (`data:image/png;base64,…`).
   */
  exportAsPNGDataURL(quality = 1): string {
    const canvas = this._resolveCanvas();
    return canvas.toDataURL('image/png', quality);
  }

  /**
   * Export the rendered scene as a self‑contained SVG string.
   *
   * Strategy: embed the PNG raster as a `<image xlink:href="…">` element
   * inside an SVG wrapper.  This preserves exact pixel output while giving
   * downstream tools an SVG envelope (viewBox, metadata, etc.).
   *
   * @param svgMeta   Optional extra SVG attributes / child elements.
   */
  async exportAsSVG(svgMeta?: {
    title?: string;
    description?: string;
    extraAttrs?: Record<string, string>;
  }): Promise<string> {
    const canvas = this._resolveCanvas();
    const dataUrl = canvas.toDataURL('image/png');
    const w = canvas.width;
    const h = canvas.height;

    const titleEl = svgMeta?.title
      ? `  <title>${escapeXml(svgMeta.title)}</title>\n`
      : '';
    const descEl = svgMeta?.description
      ? `  <desc>${escapeXml(svgMeta.description)}</desc>\n`
      : '';
    const extraAttrs = svgMeta?.extraAttrs
      ? Object.entries(svgMeta.extraAttrs)
          .map(([k, v]) => `${escapeXml(k)}="${escapeXml(v)}"`)
          .join(' ')
      : '';

    return (
      `<?xml version="1.0" encoding="UTF-8"?>\n` +
      `<svg xmlns="http://www.w3.org/2000/svg" ` +
      `xmlns:xlink="http://www.w3.org/1999/xlink" ` +
      `width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" ` +
      `${extraAttrs}>\n` +
      titleEl +
      descEl +
      `  <image x="0" y="0" width="${w}" height="${h}" ` +
      `xlink:href="${dataUrl}" />\n` +
      `</svg>`
    );
  }

  /**
   * Convenience: trigger a browser download of the PNG export.
   */
  async downloadAsPNG(filename = 'figure.png'): Promise<void> {
    const blob = await this.exportAsPNG();
    triggerDownload(URL.createObjectURL(blob), filename);
  }

  /**
   * Convenience: trigger a browser download of the SVG export.
   */
  async downloadAsSVG(filename = 'figure.svg'): Promise<void> {
    const svgString = await this.exportAsSVG();
    const blob = new Blob([svgString], { type: 'image/svg+xml' });
    triggerDownload(URL.createObjectURL(blob), filename);
  }

  // ── Private ─────────────────────────────────────────────────────────────────

  private _resolveCanvas(): HTMLCanvasElement {
    // Prefer the canvas extracted by PixiJS's extract system for pixel-perfect output
    if (this._app) {
      const renderer = (this._app as unknown as { renderer?: { extract?: { canvas: (opts?: unknown) => HTMLCanvasElement } } }).renderer;
      if (renderer?.extract) {
        return renderer.extract.canvas() as HTMLCanvasElement;
      }
    }
    if (!this._canvas) {
      throw new Error('pixi-export: call init() before exporting.');
    }
    return this._canvas;
  }
}

// ── Standalone convenience functions ─────────────────────────────────────────

/**
 * One‑shot PNG export from composite_params.json.
 *
 * @example
 * ```ts
 * const pngBlob = await exportAsPNG({ width: 1920, height: 1080 });
 * ```
 */
export async function exportAsPNG(options: ExportOptions & {
  channelBaseUrl?: string;
  compositeParamsPath?: string;
} = {}): Promise<Blob> {
  const exporter = new PixiExporter({
    channelBaseUrl: options.channelBaseUrl,
    compositeParamsPath: options.compositeParamsPath,
  });
  await exporter.init(options);
  const blob = await exporter.exportAsPNG();
  exporter.destroy();
  return blob;
}

/**
 * One‑shot SVG export from composite_params.json.
 *
 * @example
 * ```ts
 * const svgString = await exportAsSVG({ title: 'Transformer architecture' });
 * ```
 */
export async function exportAsSVG(options: ExportOptions & {
  channelBaseUrl?: string;
  compositeParamsPath?: string;
  title?: string;
  description?: string;
} = {}): Promise<string> {
  const exporter = new PixiExporter({
    channelBaseUrl: options.channelBaseUrl,
    compositeParamsPath: options.compositeParamsPath,
  });
  await exporter.init(options);
  const svg = await exporter.exportAsSVG({
    title: options.title,
    description: options.description,
  });
  exporter.destroy();
  return svg;
}

// ── PNG export + download (M232) ─────────────────────────────────────────────

/**
 * Extract a PNG data‑URL from a live PixiJS Application.
 *
 * Tries the renderer's `extract.canvas()` for pixel‑perfect output first,
 * then falls back to the Application's own canvas view.
 *
 * @param app  An initialised PixiJS Application instance.
 * @returns    A `data:image/png;base64,…` string.
 */
export function exportToPNG(app: Application): string {
  // Attempt PixiJS ExtractSystem → clean canvas snapshot
  const renderer = (
    app as unknown as {
      renderer?: {
        extract?: { canvas: (target?: unknown) => HTMLCanvasElement };
      };
    }
  ).renderer;

  if (renderer?.extract) {
    const extracted = renderer.extract.canvas();
    return extracted.toDataURL('image/png');
  }

  // Fallback: use the app's canvas directly
  const view = (app as unknown as { canvas?: HTMLCanvasElement }).canvas
    ?? (app as unknown as { view?: HTMLCanvasElement }).view;

  if (!view) {
    throw new Error(
      'pixi-export/exportToPNG: unable to resolve canvas from Application',
    );
  }

  return view.toDataURL('image/png');
}

/**
 * Trigger a browser file‑download from a data‑URL or object‑URL.
 *
 * Creates a temporary `<a>` element, clicks it, then removes it.
 *
 * @param url   A `data:` or `blob:` URL pointing to the PNG data.
 * @param name  Suggested filename for the download (default `"export.png"`).
 */
export function downloadPNG(url: string, name = 'export.png'): void {
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function escapeXml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function triggerDownload(url: string, filename: string): void {
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Revoke after short delay to allow the download to start
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
