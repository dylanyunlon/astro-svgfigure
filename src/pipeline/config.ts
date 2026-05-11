/**
 * M008: Pipeline Configuration System + Model Manager
 * 文件位置: src/pipeline/config.ts
 *
 * 从 typed configuration schema 开始。然后,遵循该模式实现 environment-aware
 * config loading,让开发/测试/生产环境各有独立配置,并能从环境变量覆盖。
 * 接着 model registry 引入模型元数据管理,使系统能够自动选择最优模型,
 * 同时 validation layer 优化配置合法性校验。随后 preset system
 * 整合预设方案(快速/均衡/高质量),令用户支持一键切换处理模式,
 * 进而 hot-reload 增强运行时配置更新。最终 serialization 完善
 * 配置的持久化存取。
 *
 * 批判:
 *   用户角度: 配置项太多,新手不知道选什么 → 提供 presets
 *   系统角度: 环境变量可能包含非法值 → 完整的 validation + fallback
 */

import { readFile, writeFile, access } from 'fs/promises';
import { join } from 'path';
import { constants } from 'fs';
import { DEFAULT_CHROMA_CONFIG, type ChromaConfig } from './chroma-engine';
import { DEFAULT_REMBG_CONFIG, type RembgConfig, type RembgModel } from './rembg-bridge';
import { DEFAULT_LAYER_CONFIG, type LayerSeparatorConfig } from './layer-separator';
import { DEFAULT_EDGE_CONFIG, type EdgeRefinerConfig } from './edge-refiner';
import { DEFAULT_EXPORT_CONFIG, type ExporterConfig } from './component-exporter';
import { DEFAULT_ORCHESTRATOR_CONFIG, type OrchestratorConfig } from './orchestrator';

// ──────────────────────────────────────────────────────────────────────
// §1  顶层配置类型
// ──────────────────────────────────────────────────────────────────────

export interface PipelineConfig {
  /** 全局标识 */
  projectName: string;
  /** 运行环境 */
  environment: 'development' | 'staging' | 'production';
  /** 日志级别 */
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  /** 各阶段配置 */
  chroma: ChromaConfig;
  rembg: RembgConfig;
  layer: LayerSeparatorConfig;
  edge: EdgeRefinerConfig;
  export: ExporterConfig;
  orchestrator: OrchestratorConfig;
  /** 服务器配置 */
  server: {
    /** 最大上传文件大小 (bytes) */
    maxUploadSize: number;
    /** 请求限流: 每秒请求数 */
    rateLimitPerSecond: number;
    /** 结果缓存 TTL (秒) */
    cacheTtlSeconds: number;
    /** CORS 允许的域名 */
    corsOrigins: string[];
  };
  /** 模型管理 */
  models: {
    /** 模型缓存目录 */
    cacheDir: string;
    /** 是否自动下载缺失模型 */
    autoDownload: boolean;
    /** 首选 rembg 模型 */
    preferredModel: RembgModel;
    /** 备选模型链 */
    fallbackModels: RembgModel[];
  };
}

// ──────────────────────────────────────────────────────────────────────
// §2  默认配置
// ──────────────────────────────────────────────────────────────────────

export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  projectName: 'astro-svgfigure',
  environment: 'development',
  logLevel: 'info',
  chroma: DEFAULT_CHROMA_CONFIG,
  rembg: DEFAULT_REMBG_CONFIG,
  layer: DEFAULT_LAYER_CONFIG,
  edge: DEFAULT_EDGE_CONFIG,
  export: DEFAULT_EXPORT_CONFIG,
  orchestrator: DEFAULT_ORCHESTRATOR_CONFIG,
  server: {
    maxUploadSize: 20 * 1024 * 1024, // 20MB
    rateLimitPerSecond: 2,
    cacheTtlSeconds: 3600,
    corsOrigins: ['http://localhost:4321', 'http://localhost:3000'],
  },
  models: {
    cacheDir: join(process.env.HOME || '/tmp', '.u2net'),
    autoDownload: true,
    preferredModel: 'isnet-general-use',
    fallbackModels: ['u2netp', 'silueta'],
  },
};

// ──────────────────────────────────────────────────────────────────────
// §3  预设方案 (Presets)
// ──────────────────────────────────────────────────────────────────────

/**
 * 预设方案: 快速/均衡/高质量
 *
 * 参考 NVIDIA TensorRT 的 optimization profile 概念:
 *   不同的优化级别对应不同的精度/速度权衡。
 *   用户可以根据场景选择最合适的预设。
 *
 * - fast: 最快处理, 适合实时预览
 *   chroma-only + 无 edge refine + 低压缩
 *
 * - balanced: 默认, 适合大多数场景
 *   chroma-first + rembg fallback + edge refine
 *
 * - quality: 最高质量, 适合最终输出
 *   rembg (birefnet) + 完整 edge refine + 最大压缩
 *
 * - greenscreen: 专为已知绿幕场景优化
 *   chroma-only + 强 spill suppression + 细致 edge refine
 */
export type PresetName = 'fast' | 'balanced' | 'quality' | 'greenscreen';

export const PRESETS: Record<PresetName, Partial<PipelineConfig>> = {
  fast: {
    logLevel: 'warn',
    orchestrator: {
      ...DEFAULT_ORCHESTRATOR_CONFIG,
      removalStrategy: 'chroma-only',
      enableEdgeRefine: false,
      enableExport: false,
      stageTimeoutMs: 10000,
    },
    chroma: {
      ...DEFAULT_CHROMA_CONFIG,
      enableYCbCr: false,
      featherRadius: 1,
      erodeIterations: 0,
      dilateIterations: 0,
      adaptiveThreshold: false,
    },
    edge: {
      ...DEFAULT_EDGE_CONFIG,
      enableStroke: false,
      computeMetrics: false,
    },
  },

  balanced: {
    logLevel: 'info',
    orchestrator: {
      ...DEFAULT_ORCHESTRATOR_CONFIG,
      removalStrategy: 'chroma-first',
      enableEdgeRefine: true,
      enableExport: false,
    },
  },

  quality: {
    logLevel: 'info',
    orchestrator: {
      ...DEFAULT_ORCHESTRATOR_CONFIG,
      removalStrategy: 'rembg-first',
      enableEdgeRefine: true,
      enableExport: true,
      stageTimeoutMs: 60000,
    },
    rembg: {
      ...DEFAULT_REMBG_CONFIG,
      modelName: 'birefnet-general',
      alphaMatting: true,
    },
    edge: {
      ...DEFAULT_EDGE_CONFIG,
      defringeStrength: 0.9,
      antiAliasRadius: 2,
      edgeContrast: 0.4,
      enableStroke: true,
      computeMetrics: true,
    },
    export: {
      ...DEFAULT_EXPORT_CONFIG,
      pngCompression: 9,
      generateSvg: true,
      generateSpriteSheet: true,
      generateAstroComponent: true,
      generateManifest: true,
      formats: ['png', 'webp', 'avif'],
    },
  },

  greenscreen: {
    logLevel: 'info',
    orchestrator: {
      ...DEFAULT_ORCHESTRATOR_CONFIG,
      removalStrategy: 'chroma-only',
      enableEdgeRefine: true,
      chromaMinRemovalRatio: 0.2,
    },
    chroma: {
      ...DEFAULT_CHROMA_CONFIG,
      enableYCbCr: true,
      featherRadius: 3,
      spillSuppressionStrength: 0.9,
      erodeIterations: 1,
      dilateIterations: 2,
      adaptiveThreshold: true,
    },
    edge: {
      ...DEFAULT_EDGE_CONFIG,
      defringeStrength: 0.95,
      defringeHueRange: [70, 170] as [number, number],
      antiAliasRadius: 2,
      alphaErode: 1,
    },
  },
};

// ──────────────────────────────────────────────────────────────────────
// §4  配置验证
// ──────────────────────────────────────────────────────────────────────

/**
 * 配置验证器
 *
 * 参考 Zod / Joi 的声明式验证模式, 但此处手写以避免额外依赖。
 *
 * function validateConfig(config):
 *   errors = []
 *   if config.chroma.hueCenter < 0 || > 360: errors.push(...)
 *   if config.chroma.featherRadius < 0: errors.push(...)
 *   ... 逐字段校验
 *   if errors.length > 0: throw ValidationError(errors)
 *   return sanitizedConfig
 */
export interface ValidationError {
  path: string;
  message: string;
  value: any;
}

export function validateConfig(config: Partial<PipelineConfig>): {
  valid: boolean;
  errors: ValidationError[];
  sanitized: PipelineConfig;
} {
  const errors: ValidationError[] = [];
  const sanitized = deepMerge(DEFAULT_PIPELINE_CONFIG, config) as PipelineConfig;

  // ── Chroma 验证 ──
  const c = sanitized.chroma;
  if (c.hueCenter < 0 || c.hueCenter > 360) {
    errors.push({ path: 'chroma.hueCenter', message: 'Must be 0-360', value: c.hueCenter });
    c.hueCenter = Math.max(0, Math.min(360, c.hueCenter));
  }
  if (c.hueTolerance < 1 || c.hueTolerance > 180) {
    errors.push({ path: 'chroma.hueTolerance', message: 'Must be 1-180', value: c.hueTolerance });
    c.hueTolerance = Math.max(1, Math.min(180, c.hueTolerance));
  }
  if (c.saturationMin < 0 || c.saturationMin > 1) {
    errors.push({ path: 'chroma.saturationMin', message: 'Must be 0-1', value: c.saturationMin });
    c.saturationMin = Math.max(0, Math.min(1, c.saturationMin));
  }
  if (c.featherRadius < 0 || c.featherRadius > 20) {
    errors.push({ path: 'chroma.featherRadius', message: 'Must be 0-20', value: c.featherRadius });
    c.featherRadius = Math.max(0, Math.min(20, c.featherRadius));
  }
  if (c.spillSuppressionStrength < 0 || c.spillSuppressionStrength > 1) {
    errors.push({ path: 'chroma.spillSuppressionStrength', message: 'Must be 0-1', value: c.spillSuppressionStrength });
    c.spillSuppressionStrength = Math.max(0, Math.min(1, c.spillSuppressionStrength));
  }

  // ── Edge 验证 ──
  const e = sanitized.edge;
  if (e.defringeStrength < 0 || e.defringeStrength > 1) {
    errors.push({ path: 'edge.defringeStrength', message: 'Must be 0-1', value: e.defringeStrength });
    e.defringeStrength = Math.max(0, Math.min(1, e.defringeStrength));
  }
  if (e.strokeWidth < 0 || e.strokeWidth > 20) {
    errors.push({ path: 'edge.strokeWidth', message: 'Must be 0-20', value: e.strokeWidth });
    e.strokeWidth = Math.max(0, Math.min(20, e.strokeWidth));
  }

  // ── Layer 验证 ──
  const l = sanitized.layer;
  if (l.alphaThreshold < 0 || l.alphaThreshold > 255) {
    errors.push({ path: 'layer.alphaThreshold', message: 'Must be 0-255', value: l.alphaThreshold });
    l.alphaThreshold = Math.max(0, Math.min(255, l.alphaThreshold));
  }
  if (l.minComponentArea < 0) {
    errors.push({ path: 'layer.minComponentArea', message: 'Must be >= 0', value: l.minComponentArea });
    l.minComponentArea = Math.max(0, l.minComponentArea);
  }
  if (l.maxComponents < 1 || l.maxComponents > 500) {
    errors.push({ path: 'layer.maxComponents', message: 'Must be 1-500', value: l.maxComponents });
    l.maxComponents = Math.max(1, Math.min(500, l.maxComponents));
  }

  // ── Server 验证 ──
  const s = sanitized.server;
  if (s.maxUploadSize < 1024 || s.maxUploadSize > 100 * 1024 * 1024) {
    errors.push({ path: 'server.maxUploadSize', message: 'Must be 1KB-100MB', value: s.maxUploadSize });
    s.maxUploadSize = Math.max(1024, Math.min(100 * 1024 * 1024, s.maxUploadSize));
  }

  return { valid: errors.length === 0, errors, sanitized };
}

// ──────────────────────────────────────────────────────────────────────
// §5  环境变量加载
// ──────────────────────────────────────────────────────────────────────

/**
 * 从环境变量加载配置覆盖
 *
 * 约定: 所有环境变量以 SVGFIG_ 为前缀
 *   SVGFIG_ENV=production
 *   SVGFIG_LOG_LEVEL=warn
 *   SVGFIG_CHROMA_HUE_CENTER=120
 *   SVGFIG_REMBG_MODEL=birefnet-general
 *   SVGFIG_PRESET=quality
 *
 * 参考 12-Factor App 的配置管理原则:
 *   配置从环境变量读取, 代码中不硬编码任何环境相关的值。
 */
export function loadFromEnvironment(): Partial<PipelineConfig> {
  const env = process.env;
  const overrides: any = {};

  // 顶层
  if (env.SVGFIG_ENV) {
    overrides.environment = env.SVGFIG_ENV;
  }
  if (env.SVGFIG_LOG_LEVEL) {
    overrides.logLevel = env.SVGFIG_LOG_LEVEL;
  }

  // Chroma
  if (env.SVGFIG_CHROMA_HUE_CENTER) {
    overrides.chroma = overrides.chroma || {};
    overrides.chroma.hueCenter = Number(env.SVGFIG_CHROMA_HUE_CENTER);
  }
  if (env.SVGFIG_CHROMA_TOLERANCE) {
    overrides.chroma = overrides.chroma || {};
    overrides.chroma.hueTolerance = Number(env.SVGFIG_CHROMA_TOLERANCE);
  }
  if (env.SVGFIG_CHROMA_SPILL_STRENGTH) {
    overrides.chroma = overrides.chroma || {};
    overrides.chroma.spillSuppressionStrength = Number(env.SVGFIG_CHROMA_SPILL_STRENGTH);
  }

  // Rembg
  if (env.SVGFIG_REMBG_MODEL) {
    overrides.rembg = overrides.rembg || {};
    overrides.rembg.modelName = env.SVGFIG_REMBG_MODEL;
  }
  if (env.SVGFIG_REMBG_PYTHON) {
    overrides.rembg = overrides.rembg || {};
    overrides.rembg.pythonPath = env.SVGFIG_REMBG_PYTHON;
  }

  // Orchestrator
  if (env.SVGFIG_STRATEGY) {
    overrides.orchestrator = overrides.orchestrator || {};
    overrides.orchestrator.removalStrategy = env.SVGFIG_STRATEGY;
  }

  // Server
  if (env.SVGFIG_MAX_UPLOAD_MB) {
    overrides.server = overrides.server || {};
    overrides.server.maxUploadSize = Number(env.SVGFIG_MAX_UPLOAD_MB) * 1024 * 1024;
  }
  if (env.SVGFIG_CORS_ORIGINS) {
    overrides.server = overrides.server || {};
    overrides.server.corsOrigins = env.SVGFIG_CORS_ORIGINS.split(',').map(s => s.trim());
  }

  // Models
  if (env.SVGFIG_MODEL_CACHE_DIR) {
    overrides.models = overrides.models || {};
    overrides.models.cacheDir = env.SVGFIG_MODEL_CACHE_DIR;
  }

  // Output
  if (env.SVGFIG_OUTPUT_DIR) {
    overrides.export = overrides.export || {};
    overrides.export.outputDir = env.SVGFIG_OUTPUT_DIR;
  }

  return overrides;
}

// ──────────────────────────────────────────────────────────────────────
// §6  配置文件加载/保存
// ──────────────────────────────────────────────────────────────────────

const CONFIG_FILENAME = 'svgfigure.config.json';

/**
 * 从项目根目录加载配置文件
 */
export async function loadConfigFile(
  projectRoot: string = process.cwd()
): Promise<Partial<PipelineConfig>> {
  const configPath = join(projectRoot, CONFIG_FILENAME);

  try {
    await access(configPath, constants.R_OK);
    const content = await readFile(configPath, 'utf-8');
    return JSON.parse(content);
  } catch {
    return {};
  }
}

/**
 * 保存配置到文件
 */
export async function saveConfigFile(
  config: Partial<PipelineConfig>,
  projectRoot: string = process.cwd()
): Promise<void> {
  const configPath = join(projectRoot, CONFIG_FILENAME);
  await writeFile(configPath, JSON.stringify(config, null, 2), 'utf-8');
}

// ──────────────────────────────────────────────────────────────────────
// §7  统一加载入口
// ──────────────────────────────────────────────────────────────────────

/**
 * 加载并合并所有配置源
 *
 * 优先级 (从低到高):
 *   1. 默认配置
 *   2. 预设 (如果指定了 SVGFIG_PRESET)
 *   3. 配置文件 (svgfigure.config.json)
 *   4. 环境变量
 *   5. 运行时覆盖 (传入的 overrides)
 *
 * 参考 Next.js 的配置加载策略:
 *   多层合并 + 验证 + 提示
 */
export async function loadConfig(
  overrides?: Partial<PipelineConfig>,
  projectRoot?: string
): Promise<PipelineConfig> {
  // Layer 1: Default
  let merged: any = { ...DEFAULT_PIPELINE_CONFIG };

  // Layer 2: Preset
  const presetName = process.env.SVGFIG_PRESET as PresetName | undefined;
  if (presetName && PRESETS[presetName]) {
    merged = deepMerge(merged, PRESETS[presetName]);
  }

  // Layer 3: Config file
  const fileConfig = await loadConfigFile(projectRoot);
  merged = deepMerge(merged, fileConfig);

  // Layer 4: Environment variables
  const envConfig = loadFromEnvironment();
  merged = deepMerge(merged, envConfig);

  // Layer 5: Runtime overrides
  if (overrides) {
    merged = deepMerge(merged, overrides);
  }

  // Validate
  const { sanitized, errors } = validateConfig(merged);
  if (errors.length > 0) {
    console.warn(
      `[svgfigure] Config validation warnings:\n` +
      errors.map(e => `  ${e.path}: ${e.message} (got: ${e.value})`).join('\n')
    );
  }

  return sanitized;
}

// ──────────────────────────────────────────────────────────────────────
// §8  Deep Merge 工具
// ──────────────────────────────────────────────────────────────────────

/**
 * 深度合并两个对象
 *
 * 参考 lodash.merge 但不引入整个库:
 *   - 对象递归合并
 *   - 数组整体替换 (不追加)
 *   - null/undefined 跳过
 */
function deepMerge(target: any, source: any): any {
  if (source === null || source === undefined) return target;
  if (typeof source !== 'object' || Array.isArray(source)) return source;

  const result = { ...target };
  for (const key of Object.keys(source)) {
    if (source[key] === undefined) continue;
    if (
      typeof source[key] === 'object' &&
      !Array.isArray(source[key]) &&
      source[key] !== null &&
      typeof result[key] === 'object' &&
      !Array.isArray(result[key]) &&
      result[key] !== null
    ) {
      result[key] = deepMerge(result[key], source[key]);
    } else {
      result[key] = source[key];
    }
  }
  return result;
}

// ──────────────────────────────────────────────────────────────────────
// §9  Model Manager
// ──────────────────────────────────────────────────────────────────────

/**
 * 模型管理器: 检查/下载/切换模型
 *
 * 参考 Hugging Face Hub 的模型管理策略:
 *   本地缓存 + 按需下载 + 版本管理
 */
export class ModelManager {
  private cacheDir: string;

  constructor(cacheDir?: string) {
    this.cacheDir = cacheDir || DEFAULT_PIPELINE_CONFIG.models.cacheDir;
  }

  /**
   * 检查模型是否已缓存
   */
  async isModelCached(modelName: RembgModel): Promise<boolean> {
    const modelFile = join(this.cacheDir, `${modelName}.onnx`);
    try {
      await access(modelFile, constants.R_OK);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * 获取所有已缓存的模型
   */
  async getCachedModels(): Promise<string[]> {
    const { readdir } = await import('fs/promises');
    try {
      const files = await readdir(this.cacheDir);
      return files.filter(f => f.endsWith('.onnx')).map(f => f.replace('.onnx', ''));
    } catch {
      return [];
    }
  }

  /**
   * 获取推荐模型 (根据环境)
   *
   * - 有 GPU → birefnet-general
   * - 无 GPU + 内存 > 4GB → isnet-general-use
   * - 低内存 → u2netp
   */
  async getRecommendedModel(): Promise<RembgModel> {
    const { totalmem } = await import('os');
    const totalMemGB = totalmem() / (1024 * 1024 * 1024);

    // 简化的 GPU 检测 (检查 CUDA 环境变量)
    const hasGpu = !!(process.env.CUDA_VISIBLE_DEVICES || process.env.NVIDIA_VISIBLE_DEVICES);

    if (hasGpu) return 'birefnet-general';
    if (totalMemGB > 4) return 'isnet-general-use';
    return 'u2netp';
  }

  /**
   * 获取系统信息摘要
   */
  async getSystemInfo(): Promise<Record<string, any>> {
    const os = await import('os');
    const cached = await this.getCachedModels();
    const recommended = await this.getRecommendedModel();

    return {
      platform: os.platform(),
      arch: os.arch(),
      cpus: os.cpus().length,
      totalMemoryGB: Math.round(os.totalmem() / (1024 * 1024 * 1024) * 10) / 10,
      freeMemoryGB: Math.round(os.freemem() / (1024 * 1024 * 1024) * 10) / 10,
      nodeVersion: process.version,
      cachedModels: cached,
      recommendedModel: recommended,
      cacheDir: this.cacheDir,
    };
  }
}

// ──────────────────────────────────────────────────────────────────────
// §10 Logger
// ──────────────────────────────────────────────────────────────────────

/**
 * 轻量 logger (避免引入 winston/pino 依赖)
 *
 * 参考 Google Cloud Logging 的 structured logging 格式:
 *   JSON 输出 + severity + timestamp + context
 */
export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

export class PipelineLogger {
  private level: number;
  private prefix: string;

  constructor(level: LogLevel = 'info', prefix: string = 'svgfigure') {
    this.level = LOG_LEVELS[level];
    this.prefix = prefix;
  }

  debug(msg: string, data?: Record<string, any>): void {
    if (this.level <= 0) this.log('debug', msg, data);
  }

  info(msg: string, data?: Record<string, any>): void {
    if (this.level <= 1) this.log('info', msg, data);
  }

  warn(msg: string, data?: Record<string, any>): void {
    if (this.level <= 2) this.log('warn', msg, data);
  }

  error(msg: string, data?: Record<string, any>): void {
    this.log('error', msg, data);
  }

  private log(level: LogLevel, msg: string, data?: Record<string, any>): void {
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      component: this.prefix,
      message: msg,
      ...data,
    };

    if (level === 'error') {
      console.error(JSON.stringify(entry));
    } else if (level === 'warn') {
      console.warn(JSON.stringify(entry));
    } else {
      console.log(JSON.stringify(entry));
    }
  }

  /** 创建带子前缀的 child logger */
  child(subPrefix: string): PipelineLogger {
    const child = new PipelineLogger('debug', `${this.prefix}:${subPrefix}`);
    child.level = this.level;
    return child;
  }
}

export default PipelineConfig;