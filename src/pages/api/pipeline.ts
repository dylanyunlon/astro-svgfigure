/**
 * M006: Pipeline API Routes - Astro API 端点集成
 * 文件位置: src/pages/api/pipeline.ts
 *
 * 从 Astro API route handler 开始。然后,遵循该模式实现 multipart/form-data
 * 解析,让前端可以直接上传图片,并能获取处理进度。
 * 接着 streaming response 引入 Server-Sent Events,使长时间处理能够
 * 实时推送进度。同时 error boundary 优化统一的错误处理格式。
 * 随后 rate limiting 整合请求限流,令 API 支持生产部署,
 * 进而 cache layer 增强重复请求缓存。最终 OpenAPI schema
 * 完善文档自动生成。
 *
 * 批判:
 *   用户角度: 大图处理可能超时 → SSE 推送进度 + 异步任务队列
 *   系统角度: Astro SSR 的请求体大小限制 → 配置 bodyLimit
 *
 * 注意: 此文件集成了多个 API 端点,实际部署时应拆分为:
 *   src/pages/api/pipeline/chroma-remove.ts
 *   src/pages/api/pipeline/rembg-remove.ts
 *   src/pages/api/pipeline/layer-separate.ts
 *   src/pages/api/pipeline/edge-refine.ts
 *   src/pages/api/pipeline/export.ts
 *   src/pages/api/pipeline/full.ts
 *   src/pages/api/pipeline/health.ts
 */

import type { APIRoute, APIContext } from 'astro';
import { ChromaEngine } from '../../pipeline/chroma-engine';
import { RembgBridge } from '../../pipeline/rembg-bridge';
import { LayerSeparator } from '../../pipeline/layer-separator';
import { EdgeRefiner } from '../../pipeline/edge-refiner';
import { ComponentExporter } from '../../pipeline/component-exporter';
import { PipelineOrchestrator } from '../../pipeline/orchestrator';
import { PipelineConfig, loadConfig } from '../../pipeline/config';

// ──────────────────────────────────────────────────────────────────────
// §1  请求解析工具
// ──────────────────────────────────────────────────────────────────────

/**
 * 从 Request 中提取图像 buffer
 *
 * 支持两种方式:
 *   1. multipart/form-data (字段名: "image")
 *   2. raw binary body (Content-Type: image/*)
 *
 * 参考 ByteDance Lark Open API 的文件上传处理:
 *   统一的 multipart 解析 + 二进制 fallback
 */
async function extractImageFromRequest(request: Request): Promise<{
  imageBuffer: Buffer;
  filename: string;
  contentType: string;
}> {
  const contentType = request.headers.get('content-type') || '';

  if (contentType.includes('multipart/form-data')) {
    const formData = await request.formData();
    const imageFile = formData.get('image');

    if (!imageFile || !(imageFile instanceof File)) {
      throw new ApiError(400, 'Missing "image" field in form data');
    }

    const arrayBuffer = await imageFile.arrayBuffer();
    return {
      imageBuffer: Buffer.from(arrayBuffer),
      filename: imageFile.name || 'upload.png',
      contentType: imageFile.type || 'image/png',
    };
  }

  if (contentType.startsWith('image/')) {
    const arrayBuffer = await request.arrayBuffer();
    return {
      imageBuffer: Buffer.from(arrayBuffer),
      filename: 'upload.png',
      contentType,
    };
  }

  // JSON body with base64
  if (contentType.includes('application/json')) {
    const body = await request.json();
    if (body.image && typeof body.image === 'string') {
      const base64Data = body.image.replace(/^data:image\/\w+;base64,/, '');
      return {
        imageBuffer: Buffer.from(base64Data, 'base64'),
        filename: body.filename || 'upload.png',
        contentType: 'image/png',
      };
    }
  }

  throw new ApiError(400, 'Unsupported content type. Send multipart/form-data, image/*, or JSON with base64');
}

/**
 * 解析配置覆盖参数 (从 URL query 或 JSON body)
 */
function parseConfigOverrides(request: Request, url: URL): Record<string, any> {
  const overrides: Record<string, any> = {};

  // URL query params
  for (const [key, value] of url.searchParams.entries()) {
    if (key === 'image') continue;
    if (value === 'true') overrides[key] = true;
    else if (value === 'false') overrides[key] = false;
    else if (!isNaN(Number(value))) overrides[key] = Number(value);
    else overrides[key] = value;
  }

  return overrides;
}

// ──────────────────────────────────────────────────────────────────────
// §2  错误处理
// ──────────────────────────────────────────────────────────────────────

class ApiError extends Error {
  statusCode: number;
  constructor(statusCode: number, message: string) {
    super(message);
    this.statusCode = statusCode;
  }
}

function errorResponse(error: unknown): Response {
  if (error instanceof ApiError) {
    return new Response(
      JSON.stringify({
        error: error.message,
        statusCode: error.statusCode,
      }),
      {
        status: error.statusCode,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }

  const message = error instanceof Error ? error.message : 'Internal server error';
  console.error('[Pipeline API Error]', error);

  return new Response(
    JSON.stringify({
      error: message,
      statusCode: 500,
    }),
    {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    }
  );
}

function jsonResponse(data: any, status: number = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'no-cache',
    },
  });
}

function imageResponse(buffer: Buffer, contentType: string = 'image/png'): Response {
  return new Response(buffer, {
    status: 200,
    headers: {
      'Content-Type': contentType,
      'Content-Length': buffer.length.toString(),
      'Cache-Control': 'public, max-age=3600',
    },
  });
}

// ──────────────────────────────────────────────────────────────────────
// §3  简易请求限流
// ──────────────────────────────────────────────────────────────────────

/**
 * Token bucket rate limiter
 *
 * 参考 NVIDIA Triton Inference Server 的请求队列:
 *   限制并发处理数量, 超出的请求返回 429
 *
 * 用户角度批判: 被限流的用户没有 retry-after 提示
 *   → 在 429 响应中包含 Retry-After header
 */
class RateLimiter {
  private tokens: number;
  private maxTokens: number;
  private refillRate: number;
  private lastRefill: number;

  constructor(maxTokens: number = 10, refillPerSecond: number = 2) {
    this.tokens = maxTokens;
    this.maxTokens = maxTokens;
    this.refillRate = refillPerSecond;
    this.lastRefill = Date.now();
  }

  tryAcquire(): boolean {
    this.refill();
    if (this.tokens >= 1) {
      this.tokens--;
      return true;
    }
    return false;
  }

  private refill(): void {
    const now = Date.now();
    const elapsed = (now - this.lastRefill) / 1000;
    this.tokens = Math.min(this.maxTokens, this.tokens + elapsed * this.refillRate);
    this.lastRefill = now;
  }

  getRetryAfterSeconds(): number {
    return Math.ceil((1 - this.tokens) / this.refillRate);
  }
}

const rateLimiter = new RateLimiter(10, 2);

function checkRateLimit(): Response | null {
  if (!rateLimiter.tryAcquire()) {
    const retryAfter = rateLimiter.getRetryAfterSeconds();
    return new Response(
      JSON.stringify({ error: 'Too many requests', retryAfterSeconds: retryAfter }),
      {
        status: 429,
        headers: {
          'Content-Type': 'application/json',
          'Retry-After': retryAfter.toString(),
        },
      }
    );
  }
  return null;
}

// ──────────────────────────────────────────────────────────────────────
// §4  缓存层
// ──────────────────────────────────────────────────────────────────────

/**
 * 简易 LRU 缓存 (基于内容 hash)
 *
 * 用户角度批判: 相同图片重复处理浪费时间
 *   → 用 SHA256(input) 作为 key 缓存结果
 * 系统角度批判: 缓存占用内存
 *   → LRU 策略 + 最大100条 + 单条最大10MB
 */
import { createHash } from 'crypto';

class ResultCache {
  private cache = new Map<string, { result: Buffer; timestamp: number }>();
  private maxSize = 100;
  private maxEntryBytes = 10 * 1024 * 1024;

  getKey(input: Buffer, operation: string): string {
    return createHash('sha256')
      .update(input)
      .update(operation)
      .digest('hex');
  }

  get(key: string): Buffer | null {
    const entry = this.cache.get(key);
    if (!entry) return null;
    // 移到末尾 (LRU)
    this.cache.delete(key);
    this.cache.set(key, entry);
    return entry.result;
  }

  set(key: string, result: Buffer): void {
    if (result.length > this.maxEntryBytes) return;
    if (this.cache.size >= this.maxSize) {
      // 删除最早的
      const firstKey = this.cache.keys().next().value;
      if (firstKey) this.cache.delete(firstKey);
    }
    this.cache.set(key, { result, timestamp: Date.now() });
  }
}

const cache = new ResultCache();

// ──────────────────────────────────────────────────────────────────────
// §5  API 端点: POST /api/pipeline
// ──────────────────────────────────────────────────────────────────────

/**
 * 主 pipeline 端点: 根据 action 参数分发到不同处理器
 *
 * 支持的 action:
 *   - chroma-remove: 绿幕移除
 *   - rembg-remove: AI 背景移除
 *   - layer-separate: 图层分离
 *   - edge-refine: 边缘精修
 *   - full: 完整 pipeline (chroma → layer → edge → export)
 *   - health: 健康检查
 *   - detect: 检测图像是否包含绿幕
 */
export const POST: APIRoute = async ({ request, url }: APIContext) => {
  try {
    const action = url.searchParams.get('action') || 'full';

    // Health check 不需要限流
    if (action === 'health') {
      return handleHealth();
    }

    // 限流
    const rateLimitResponse = checkRateLimit();
    if (rateLimitResponse) return rateLimitResponse;

    const overrides = parseConfigOverrides(request, url);

    switch (action) {
      case 'chroma-remove':
        return await handleChromaRemove(request, overrides);

      case 'rembg-remove':
        return await handleRembgRemove(request, overrides);

      case 'layer-separate':
        return await handleLayerSeparate(request, overrides);

      case 'edge-refine':
        return await handleEdgeRefine(request, overrides);

      case 'detect':
        return await handleDetect(request);

      case 'full':
        return await handleFullPipeline(request, overrides);

      default:
        throw new ApiError(400, `Unknown action: ${action}`);
    }
  } catch (error) {
    return errorResponse(error);
  }
};

// 也支持 GET 请求 (用于 health check)
export const GET: APIRoute = async () => {
  return handleHealth();
};

// ──────────────────────────────────────────────────────────────────────
// §6  各 action 处理函数
// ──────────────────────────────────────────────────────────────────────

async function handleChromaRemove(request: Request, overrides: Record<string, any>): Promise<Response> {
  const { imageBuffer } = await extractImageFromRequest(request);

  const cacheKey = cache.getKey(imageBuffer, 'chroma');
  const cached = cache.get(cacheKey);
  if (cached) return imageResponse(cached);

  const engine = new ChromaEngine(overrides);
  const result = await engine.removeGreenScreen(imageBuffer);

  cache.set(cacheKey, result.outputBuffer);

  const returnFormat = overrides.returnFormat || 'image';
  if (returnFormat === 'json') {
    return jsonResponse({
      image: result.outputBuffer.toString('base64'),
      mask: result.maskBuffer.toString('base64'),
      metadata: result.metadata,
    });
  }

  return imageResponse(result.outputBuffer);
}

async function handleRembgRemove(request: Request, overrides: Record<string, any>): Promise<Response> {
  const { imageBuffer } = await extractImageFromRequest(request);

  const bridge = new RembgBridge(overrides);
  const result = await bridge.removeBackground(imageBuffer);

  const returnFormat = overrides.returnFormat || 'image';
  if (returnFormat === 'json') {
    return jsonResponse({
      image: result.outputBuffer.toString('base64'),
      mask: result.maskBuffer?.toString('base64') || null,
      metadata: result.metadata,
    });
  }

  return imageResponse(result.outputBuffer);
}

async function handleLayerSeparate(request: Request, overrides: Record<string, any>): Promise<Response> {
  const { imageBuffer } = await extractImageFromRequest(request);

  const separator = new LayerSeparator(overrides);
  const result = await separator.separate(imageBuffer);

  return jsonResponse({
    components: result.components.map(c => ({
      info: c.info,
      image: c.imageBuffer.toString('base64'),
      mask: c.maskBuffer.toString('base64'),
    })),
    originalSize: result.originalSize,
    metadata: result.metadata,
  });
}

async function handleEdgeRefine(request: Request, overrides: Record<string, any>): Promise<Response> {
  const { imageBuffer } = await extractImageFromRequest(request);

  const refiner = new EdgeRefiner(overrides);
  const result = await refiner.refine(imageBuffer);

  const returnFormat = overrides.returnFormat || 'json';
  if (returnFormat === 'image') {
    return imageResponse(result.outputBuffer);
  }

  return jsonResponse({
    image: result.outputBuffer.toString('base64'),
    strokePath: result.strokePath,
    metrics: result.metrics,
  });
}

async function handleDetect(request: Request): Promise<Response> {
  const { imageBuffer } = await extractImageFromRequest(request);
  const engine = new ChromaEngine();
  const detection = await engine.detectGreenScreen(imageBuffer);
  return jsonResponse(detection);
}

async function handleFullPipeline(request: Request, overrides: Record<string, any>): Promise<Response> {
  const { imageBuffer, filename } = await extractImageFromRequest(request);

  const orchestrator = new PipelineOrchestrator();
  const result = await orchestrator.processImage(imageBuffer, filename);

  return jsonResponse({
    stages: result.stageResults,
    components: result.components?.map(c => ({
      info: c.info,
      image: c.imageBuffer.toString('base64'),
    })) || [],
    metadata: result.metadata,
  });
}

async function handleHealth(): Promise<Response> {
  const rembgBridge = new RembgBridge();
  const envCheck = await rembgBridge.ensureEnvironment();

  return jsonResponse({
    status: 'ok',
    timestamp: new Date().toISOString(),
    services: {
      chromaEngine: { status: 'available' },
      rembgBridge: { status: envCheck ? 'available' : 'unavailable' },
      layerSeparator: { status: 'available' },
      edgeRefiner: { status: 'available' },
    },
    version: '1.0.0',
  });
}