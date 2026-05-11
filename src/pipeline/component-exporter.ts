/**
 * M005: Component Exporter - 组件导出 (透明 PNG + SVG trace + Sprite Sheet)
 * 文件位置: src/pipeline/component-exporter.ts
 *
 * 从 PNG export with metadata 开始。然后,遵循该模式实现 SVG vector trace,
 * 让每个组件可以同时拥有光栅和矢量两种表示,并能保持原始位置信息。
 * 接着 sprite sheet packing 引入 bin packing 算法,使多组件能够打包到
 * 单张 sprite sheet。同时 manifest generation 优化产出结构化的 JSON 清单。
 * 随后 Astro component generation 整合模板系统,令导出物支持直接导入
 * Astro 项目,进而 optimization pipeline 增强 PNG crush + SVG optimize。
 * 最终 batch export 完善批量处理能力,确保输出兼容 Astro 的 node_modules
 * 静态资源系统。
 *
 * 批判:
 *   用户角度: 导出路径可能与现有文件冲突 → 使用 hash-based naming
 *   系统角度: 大量小文件的 I/O 开销 → 提供 sprite sheet 合并选项
 */

import sharp from 'sharp';
import { createHash } from 'crypto';
import { mkdir, writeFile } from 'fs/promises';
import { join, dirname } from 'path';

// ──────────────────────────────────────────────────────────────────────
// §1  类型定义
// ──────────────────────────────────────────────────────────────────────

export interface ExporterConfig {
  /** 输出基础目录 */
  outputDir: string;
  /** 文件名前缀 */
  filePrefix: string;
  /** PNG 压缩级别 (0-9) */
  pngCompression: number;
  /** 是否生成 SVG trace */
  generateSvg: boolean;
  /** SVG 描边宽度 */
  svgStrokeWidth: number;
  /** SVG 描边颜色 */
  svgStrokeColor: string;
  /** SVG 填充颜色 (none=透明) */
  svgFillColor: string;
  /** 是否生成 sprite sheet */
  generateSpriteSheet: boolean;
  /** Sprite sheet 最大宽度 */
  spriteMaxWidth: number;
  /** Sprite sheet padding */
  spritePadding: number;
  /** 是否生成 Astro 组件 */
  generateAstroComponent: boolean;
  /** 是否生成 manifest.json */
  generateManifest: boolean;
  /** 输出尺寸缩放 (1.0 = 原始, 0.5 = 半尺寸) */
  scale: number;
  /** 输出格式 */
  formats: Array<'png' | 'webp' | 'avif'>;
}

export interface ComponentExport {
  id: string;
  label: number;
  files: {
    png?: string;
    webp?: string;
    avif?: string;
    svg?: string;
    mask?: string;
  };
  dimensions: { width: number; height: number };
  position: { x: number; y: number };
  zIndex: number;
  hash: string;
}

export interface ExportResult {
  components: ComponentExport[];
  spriteSheet?: {
    path: string;
    width: number;
    height: number;
    regions: Array<{
      id: string;
      x: number;
      y: number;
      width: number;
      height: number;
    }>;
  };
  manifest?: {
    path: string;
    data: Record<string, any>;
  };
  astroComponent?: {
    path: string;
    content: string;
  };
  totalFilesGenerated: number;
  totalSizeBytes: number;
  processingTimeMs: number;
}

// ──────────────────────────────────────────────────────────────────────
// §2  默认配置
// ──────────────────────────────────────────────────────────────────────

export const DEFAULT_EXPORT_CONFIG: ExporterConfig = {
  outputDir: './dist/components',
  filePrefix: 'comp',
  pngCompression: 6,
  generateSvg: true,
  svgStrokeWidth: 2,
  svgStrokeColor: '#000000',
  svgFillColor: 'none',
  generateSpriteSheet: true,
  spriteMaxWidth: 2048,
  spritePadding: 4,
  generateAstroComponent: true,
  generateManifest: true,
  scale: 1.0,
  formats: ['png', 'webp'],
};

// ──────────────────────────────────────────────────────────────────────
// §3  工具函数
// ──────────────────────────────────────────────────────────────────────

function computeHash(buffer: Buffer): string {
  return createHash('sha256').update(buffer).digest('hex').substring(0, 12);
}

async function ensureDir(dirPath: string): Promise<void> {
  await mkdir(dirPath, { recursive: true });
}

// ──────────────────────────────────────────────────────────────────────
// §4  PNG/WebP/AVIF 导出
// ──────────────────────────────────────────────────────────────────────

/**
 * 多格式图像导出
 *
 * 参考 Google Lighthouse 的图像优化建议:
 *   同时生成 WebP 和 AVIF 作为现代格式,
 *   PNG 作为 fallback。Astro 的 <Picture> 组件可自动选择。
 *
 * function exportMultiFormat(buffer, basePath, formats, scale, compression):
 *   img = sharp(buffer)
 *   if scale != 1.0:
 *     img = img.resize(round(w*scale), round(h*scale))
 *   results = {}
 *   for fmt in formats:
 *     if fmt == 'png': results.png = img.png(compression).toFile(basePath+'.png')
 *     if fmt == 'webp': results.webp = img.webp(quality=85).toFile(basePath+'.webp')
 *     if fmt == 'avif': results.avif = img.avif(quality=80).toFile(basePath+'.avif')
 *   return results
 */
async function exportMultiFormat(
  imageBuffer: Buffer,
  basePath: string,
  formats: Array<'png' | 'webp' | 'avif'>,
  scale: number,
  pngCompression: number
): Promise<{ files: Record<string, string>; totalBytes: number }> {
  let img = sharp(imageBuffer);

  if (scale !== 1.0) {
    const meta = await sharp(imageBuffer).metadata();
    const newW = Math.round(meta.width! * scale);
    const newH = Math.round(meta.height! * scale);
    img = img.resize(newW, newH, { kernel: 'lanczos3' });
  }

  const files: Record<string, string> = {};
  let totalBytes = 0;

  for (const fmt of formats) {
    const filePath = `${basePath}.${fmt}`;
    let encoder = img.clone();

    switch (fmt) {
      case 'png':
        encoder = encoder.png({ compressionLevel: pngCompression });
        break;
      case 'webp':
        encoder = encoder.webp({ quality: 85, alphaQuality: 90 });
        break;
      case 'avif':
        encoder = encoder.avif({ quality: 80 });
        break;
    }

    const outputInfo = await encoder.toFile(filePath);
    files[fmt] = filePath;
    totalBytes += outputInfo.size;
  }

  return { files, totalBytes };
}

// ──────────────────────────────────────────────────────────────────────
// §5  SVG 生成
// ──────────────────────────────────────────────────────────────────────

/**
 * 从描边路径生成独立的 SVG 文件
 *
 * 参考 astro-icon 的 SVG 内联模式:
 *   生成符合 Astro 约定的 SVG, 可直接 import 为组件。
 *   包含 viewBox, preserveAspectRatio, 和无障碍属性。
 *
 * function generateSvg(strokePath, width, height, config):
 *   svg = `<svg xmlns="..." viewBox="0 0 ${width} ${height}" ...>
 *     <path d="${strokePath}" 
 *           stroke="${config.strokeColor}"
 *           stroke-width="${config.strokeWidth}"
 *           fill="${config.fillColor}" />
 *   </svg>`
 *   return svg
 */
function generateSvgContent(
  strokePath: string,
  width: number,
  height: number,
  strokeWidth: number,
  strokeColor: string,
  fillColor: string,
  componentId: string
): string {
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     viewBox="0 0 ${width} ${height}"
     width="${width}" 
     height="${height}"
     role="img"
     aria-label="Component ${componentId}"
     data-component-id="${componentId}">
  <title>Component ${componentId}</title>
  <path d="${strokePath}" 
        stroke="${strokeColor}" 
        stroke-width="${strokeWidth}" 
        stroke-linecap="round"
        stroke-linejoin="round"
        fill="${fillColor}"
        vector-effect="non-scaling-stroke" />
</svg>`;
}

// ──────────────────────────────────────────────────────────────────────
// §6  Sprite Sheet (Bin Packing)
// ──────────────────────────────────────────────────────────────────────

/**
 * Shelf-based Bin Packing for sprite sheet
 *
 * 参考 game engine (Unity, Godot) 的 texture atlas packing:
 *   Next-Fit Decreasing Height (NFDH) 算法:
 *   1. 按高度降序排列所有矩形
 *   2. 依次放入当前 shelf (行)
 *   3. 当前 shelf 放不下时, 新建一行
 *
 * function packSpriteSheet(components, maxWidth, padding):
 *   sorted = components.sortBy(c => -c.height)
 *   shelves = [{ y: 0, height: 0, items: [] }]
 *   currentX = 0
 *   for comp in sorted:
 *     if currentX + comp.width > maxWidth:
 *       newShelf() 
 *       currentX = 0
 *     place(comp, currentX, currentShelf.y)
 *     currentX += comp.width + padding
 *   return { placements, totalWidth, totalHeight }
 */
interface SpriteRegion {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  buffer: Buffer;
}

function packSprites(
  items: Array<{ id: string; width: number; height: number; buffer: Buffer }>,
  maxWidth: number,
  padding: number
): { regions: SpriteRegion[]; totalWidth: number; totalHeight: number } {
  // 按高度降序排列
  const sorted = [...items].sort((a, b) => b.height - a.height);
  const regions: SpriteRegion[] = [];

  let currentX = padding;
  let currentY = padding;
  let shelfHeight = 0;
  let maxUsedWidth = 0;

  for (const item of sorted) {
    if (currentX + item.width + padding > maxWidth) {
      // 新行
      currentX = padding;
      currentY += shelfHeight + padding;
      shelfHeight = 0;
    }

    regions.push({
      id: item.id,
      x: currentX,
      y: currentY,
      width: item.width,
      height: item.height,
      buffer: item.buffer,
    });

    currentX += item.width + padding;
    shelfHeight = Math.max(shelfHeight, item.height);
    maxUsedWidth = Math.max(maxUsedWidth, currentX);
  }

  const totalHeight = currentY + shelfHeight + padding;
  const totalWidth = Math.min(maxWidth, maxUsedWidth + padding);

  return { regions, totalWidth, totalHeight };
}

async function renderSpriteSheet(
  regions: SpriteRegion[],
  totalWidth: number,
  totalHeight: number
): Promise<Buffer> {
  // 创建透明画布
  const canvas = sharp({
    create: {
      width: totalWidth,
      height: totalHeight,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 },
    },
  });

  // 合成所有组件
  const compositeInputs = regions.map(r => ({
    input: r.buffer,
    left: r.x,
    top: r.y,
  }));

  return canvas
    .composite(compositeInputs)
    .png({ compressionLevel: 6 })
    .toBuffer();
}

// ──────────────────────────────────────────────────────────────────────
// §7  Manifest 生成
// ──────────────────────────────────────────────────────────────────────

/**
 * 生成 JSON manifest
 *
 * 参考 Webpack 的 asset-manifest.json 格式:
 *   记录每个组件的文件路径、尺寸、位置信息,
 *   供前端代码程序化引用。
 */
function generateManifest(
  components: ComponentExport[],
  spriteSheet?: ExportResult['spriteSheet'],
  originalSize?: { width: number; height: number }
): Record<string, any> {
  return {
    version: '1.0.0',
    generator: 'astro-svgfigure-pipeline',
    generatedAt: new Date().toISOString(),
    original: originalSize || null,
    components: components.map(c => ({
      id: c.id,
      label: c.label,
      files: c.files,
      dimensions: c.dimensions,
      position: c.position,
      zIndex: c.zIndex,
      hash: c.hash,
    })),
    spriteSheet: spriteSheet
      ? {
          path: spriteSheet.path,
          dimensions: { width: spriteSheet.width, height: spriteSheet.height },
          regions: spriteSheet.regions,
        }
      : null,
    totalComponents: components.length,
  };
}

// ──────────────────────────────────────────────────────────────────────
// §8  Astro 组件生成
// ──────────────────────────────────────────────────────────────────────

/**
 * 生成 Astro 组件用于直接导入
 *
 * 参考 astro-icon 的组件模式, 确保样式来自 astro 的 node_modules:
 *   生成 .astro 文件, 可以 import 到页面中使用。
 *   支持 props 传入自定义样式和尺寸。
 *
 * function generateAstroComponent(components, manifestPath):
 *   return `---
 *   import manifest from '${manifestPath}';
 *   const { class: className, ...props } = Astro.props;
 *   ---
 *   <div class:list={["svgfigure-components", className]} {...props}>
 *     {manifest.components.map(comp => (
 *       <img src={comp.files.png} ... />
 *     ))}
 *   </div>
 *   <style>
 *     .svgfigure-components { position: relative; }
 *   </style>`
 */
function generateAstroComponentContent(
  components: ComponentExport[],
  manifestRelPath: string
): string {
  const imports = components
    .filter(c => c.files.png)
    .map((c, i) => `import comp${i}Src from '${c.files.png}';`)
    .join('\n');

  const componentEntries = components
    .map((c, i) => {
      return `  {
    id: '${c.id}',
    src: comp${i}Src,
    width: ${c.dimensions.width},
    height: ${c.dimensions.height},
    x: ${c.position.x},
    y: ${c.position.y},
    zIndex: ${c.zIndex},
  }`;
    })
    .join(',\n');

  return `---
/**
 * Auto-generated by astro-svgfigure pipeline
 * Do not edit manually - regenerate with: npm run pipeline:export
 */
import type { HTMLAttributes } from 'astro';
${imports}

interface Props extends HTMLAttributes<'div'> {
  /** Display mode: 'layered' preserves original positions, 'grid' shows side-by-side */
  mode?: 'layered' | 'grid';
  /** Scale factor for the entire composition */
  scale?: number;
  /** Whether to show component borders (debug) */
  debug?: boolean;
}

const { mode = 'layered', scale = 1, debug = false, class: className, ...rest } = Astro.props;

const components = [
${componentEntries}
];
---

<div 
  class:list={["svgfigure-composition", \`svgfigure-\${mode}\`, className]}
  style={\`--sf-scale: \${scale};\`}
  data-component-count={components.length}
  {...rest}
>
  {components.map((comp) => (
    <div
      class="svgfigure-layer"
      style={mode === 'layered'
        ? \`left:\${comp.x * scale}px;top:\${comp.y * scale}px;z-index:\${comp.zIndex};width:\${comp.width * scale}px;height:\${comp.height * scale}px;\`
        : undefined
      }
      data-layer-id={comp.id}
      data-debug={debug ? 'true' : undefined}
    >
      <img
        src={comp.src}
        width={Math.round(comp.width * scale)}
        height={Math.round(comp.height * scale)}
        alt={\`Component \${comp.id}\`}
        loading="lazy"
        decoding="async"
      />
    </div>
  ))}
</div>

<style>
  .svgfigure-composition {
    position: relative;
    display: inline-block;
  }
  
  .svgfigure-layered .svgfigure-layer {
    position: absolute;
  }

  .svgfigure-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
  }

  .svgfigure-grid .svgfigure-layer {
    position: relative;
  }

  .svgfigure-layer img {
    display: block;
    image-rendering: auto;
  }

  .svgfigure-layer[data-debug="true"] {
    outline: 1px dashed rgba(255, 0, 0, 0.5);
  }
</style>
`;
}

// ──────────────────────────────────────────────────────────────────────
// §9  ComponentExporter 主类
// ──────────────────────────────────────────────────────────────────────

export class ComponentExporter {
  private config: ExporterConfig;

  constructor(config?: Partial<ExporterConfig>) {
    this.config = { ...DEFAULT_EXPORT_CONFIG, ...config };
  }

  /**
   * 导出所有组件
   *
   * Pipeline:
   *   1. 创建输出目录
   *   2. 导出每个组件的多格式图像
   *   3. [可选] 生成 SVG
   *   4. [可选] 打包 sprite sheet
   *   5. [可选] 生成 manifest.json
   *   6. [可选] 生成 Astro 组件
   */
  async exportAll(
    components: Array<{
      info: { id: string; label: number; bbox: { width: number; height: number }; position: { x: number; y: number }; zIndex: number };
      imageBuffer: Buffer;
      maskBuffer: Buffer;
      strokePath?: string;
    }>,
    originalSize?: { width: number; height: number }
  ): Promise<ExportResult> {
    const startTime = Date.now();
    const cfg = this.config;
    let totalFiles = 0;
    let totalBytes = 0;

    // ── Step 1: 目录 ──
    await ensureDir(cfg.outputDir);
    const imgDir = join(cfg.outputDir, 'images');
    const svgDir = join(cfg.outputDir, 'svg');
    await ensureDir(imgDir);
    if (cfg.generateSvg) await ensureDir(svgDir);

    // ── Step 2: 导出图像 ──
    const exports: ComponentExport[] = [];

    for (const comp of components) {
      const hash = computeHash(comp.imageBuffer);
      const baseName = `${cfg.filePrefix}-${comp.info.label}-${hash}`;
      const basePath = join(imgDir, baseName);

      const { files, totalBytes: bytes } = await exportMultiFormat(
        comp.imageBuffer,
        basePath,
        cfg.formats,
        cfg.scale,
        cfg.pngCompression
      );

      totalBytes += bytes;
      totalFiles += Object.keys(files).length;

      // Mask
      const maskPath = join(imgDir, `${baseName}-mask.png`);
      await sharp(comp.maskBuffer)
        .png({ compressionLevel: cfg.pngCompression })
        .toFile(maskPath);
      totalFiles++;

      const exportEntry: ComponentExport = {
        id: comp.info.id,
        label: comp.info.label,
        files: { ...files, mask: maskPath } as any,
        dimensions: {
          width: Math.round(comp.info.bbox.width * cfg.scale),
          height: Math.round(comp.info.bbox.height * cfg.scale),
        },
        position: comp.info.position,
        zIndex: comp.info.zIndex,
        hash,
      };

      // ── Step 3: SVG ──
      if (cfg.generateSvg && comp.strokePath) {
        const svgContent = generateSvgContent(
          comp.strokePath,
          comp.info.bbox.width,
          comp.info.bbox.height,
          cfg.svgStrokeWidth,
          cfg.svgStrokeColor,
          cfg.svgFillColor,
          comp.info.id
        );
        const svgPath = join(svgDir, `${baseName}.svg`);
        await writeFile(svgPath, svgContent, 'utf-8');
        exportEntry.files.svg = svgPath;
        totalFiles++;
      }

      exports.push(exportEntry);
    }

    const result: ExportResult = {
      components: exports,
      totalFilesGenerated: totalFiles,
      totalSizeBytes: totalBytes,
      processingTimeMs: Date.now() - startTime,
    };

    // ── Step 4: Sprite Sheet ──
    if (cfg.generateSpriteSheet && components.length > 1) {
      const items = await Promise.all(
        components.map(async (comp) => {
          const meta = await sharp(comp.imageBuffer).metadata();
          return {
            id: comp.info.id,
            width: Math.round(meta.width! * cfg.scale),
            height: Math.round(meta.height! * cfg.scale),
            buffer: cfg.scale !== 1.0
              ? await sharp(comp.imageBuffer)
                  .resize(
                    Math.round(meta.width! * cfg.scale),
                    Math.round(meta.height! * cfg.scale)
                  )
                  .png()
                  .toBuffer()
              : comp.imageBuffer,
          };
        })
      );

      const { regions, totalWidth, totalHeight } = packSprites(
        items, cfg.spriteMaxWidth, cfg.spritePadding
      );

      const sheetBuffer = await renderSpriteSheet(regions, totalWidth, totalHeight);
      const sheetPath = join(cfg.outputDir, `${cfg.filePrefix}-spritesheet.png`);
      await writeFile(sheetPath, sheetBuffer);
      totalFiles++;
      totalBytes += sheetBuffer.length;

      result.spriteSheet = {
        path: sheetPath,
        width: totalWidth,
        height: totalHeight,
        regions: regions.map(r => ({
          id: r.id, x: r.x, y: r.y, width: r.width, height: r.height,
        })),
      };
    }

    // ── Step 5: Manifest ──
    if (cfg.generateManifest) {
      const manifestData = generateManifest(exports, result.spriteSheet, originalSize);
      const manifestPath = join(cfg.outputDir, 'manifest.json');
      await writeFile(manifestPath, JSON.stringify(manifestData, null, 2), 'utf-8');
      totalFiles++;
      result.manifest = { path: manifestPath, data: manifestData };
    }

    // ── Step 6: Astro 组件 ──
    if (cfg.generateAstroComponent) {
      const astroContent = generateAstroComponentContent(
        exports,
        './manifest.json'
      );
      const astroPath = join(cfg.outputDir, 'FigureComponents.astro');
      await writeFile(astroPath, astroContent, 'utf-8');
      totalFiles++;
      result.astroComponent = { path: astroPath, content: astroContent };
    }

    result.totalFilesGenerated = totalFiles;
    result.totalSizeBytes = totalBytes;
    result.processingTimeMs = Date.now() - startTime;

    return result;
  }

  updateConfig(partial: Partial<ExporterConfig>): void {
    this.config = { ...this.config, ...partial };
  }
}

export default ComponentExporter;