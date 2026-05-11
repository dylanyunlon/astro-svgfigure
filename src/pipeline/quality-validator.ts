/**
 * M009: Quality Validator - 质量验证 + 自动化测试指标
 * 文件位置: src/pipeline/quality-validator.ts
 *
 * 从 alpha coverage ratio 开始。然后,遵循该模式实现 edge quality scoring,
 * 让每个处理结果都有量化的质量评分,并能自动判定 pass/fail。
 * 接着 green residue detection 引入残留绿色检测,使 pipeline 能够
 * 发现 chroma 去除不完整的情况。同时 component completeness check
 * 优化组件完整性验证。随后 visual regression 整合像素级对比,
 * 令 CI/CD 支持自动化视觉回归测试,进而 benchmark suite
 * 增强性能基准追踪。最终 report generator 完善 HTML/JSON 报告输出。
 *
 * 批判:
 *   用户角度: 质量指标对非技术用户不直观
 *     → 转换为 A/B/C/D/F 等级 + 自然语言描述
 *   系统角度: 全图扫描的质量检查可能比处理本身还慢
 *     → 使用采样策略 (每 N 个像素采样一次)
 */

import sharp from 'sharp';

// ──────────────────────────────────────────────────────────────────────
// §1  类型定义
// ──────────────────────────────────────────────────────────────────────

export interface QualityMetrics {
  /** 整体质量评分 (0-100) */
  overallScore: number;
  /** 质量等级 */
  grade: 'A' | 'B' | 'C' | 'D' | 'F';
  /** 是否通过质量门槛 */
  passed: boolean;
  /** 各维度详细指标 */
  dimensions: {
    /** Alpha 覆盖率: 前景像素占比 */
    alphaCoverage: MetricResult;
    /** 绿色残留: 残留的绿色像素占比 */
    greenResidue: MetricResult;
    /** 边缘质量: 边缘平滑度 */
    edgeQuality: MetricResult;
    /** 组件完整性: 预期组件是否都被检测到 */
    componentCompleteness: MetricResult;
    /** 透明度分布: alpha值的合理分布 */
    alphaDistribution: MetricResult;
  };
  /** 自然语言总结 */
  summary: string;
  /** 改进建议 */
  suggestions: string[];
  /** 检测耗时 */
  validationTimeMs: number;
}

export interface MetricResult {
  /** 指标名称 */
  name: string;
  /** 原始值 (0-1) */
  value: number;
  /** 加权分数 (0-100) */
  score: number;
  /** 权重 */
  weight: number;
  /** 状态 */
  status: 'good' | 'acceptable' | 'poor';
  /** 描述 */
  description: string;
}

export interface QualityConfig {
  /** 最低通过分数 (0-100) */
  minPassScore: number;
  /** Alpha 覆盖率的期望范围 [min, max] */
  expectedAlphaCoverage: [number, number];
  /** 最大允许绿色残留比例 */
  maxGreenResidue: number;
  /** 最小边缘质量分数 */
  minEdgeQuality: number;
  /** 采样率 (0-1): 1.0=全像素, 0.1=每10个采一个 */
  samplingRate: number;
  /** 绿色检测的 HSV 参数 */
  greenHueRange: [number, number];
  greenSatMin: number;
}

export interface BenchmarkResult {
  /** 处理阶段 */
  stage: string;
  /** 输入大小 (像素) */
  inputPixels: number;
  /** 处理时间 (ms) */
  durationMs: number;
  /** 每秒处理像素数 (MPixel/s) */
  throughputMpxPerSec: number;
  /** 峰值内存 (MB) */
  peakMemoryMb: number;
}

// ──────────────────────────────────────────────────────────────────────
// §2  默认配置
// ──────────────────────────────────────────────────────────────────────

export const DEFAULT_QUALITY_CONFIG: QualityConfig = {
  minPassScore: 60,
  expectedAlphaCoverage: [0.05, 0.85],
  maxGreenResidue: 0.02,
  minEdgeQuality: 0.4,
  samplingRate: 0.25,
  greenHueRange: [80, 160],
  greenSatMin: 0.2,
};

// ──────────────────────────────────────────────────────────────────────
// §3  色彩工具
// ──────────────────────────────────────────────────────────────────────

function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const rn = r / 255, gn = g / 255, bn = b / 255;
  const mx = Math.max(rn, gn, bn), mn = Math.min(rn, gn, bn);
  const d = mx - mn;
  let h = 0;
  if (d !== 0) {
    if (mx === rn) h = 60 * (((gn - bn) / d) % 6);
    else if (mx === gn) h = 60 * ((bn - rn) / d + 2);
    else h = 60 * ((rn - gn) / d + 4);
  }
  if (h < 0) h += 360;
  const s = mx === 0 ? 0 : d / mx;
  return [h, s, mx];
}

// ──────────────────────────────────────────────────────────────────────
// §4  Alpha 覆盖率分析
// ──────────────────────────────────────────────────────────────────────

/**
 * 计算前景(不透明)像素占比
 *
 * 从 alpha histogram 开始。然后,遵循该模式实现 coverage ratio,
 * 让 validator 可以判断前景是否被过度移除或保留不足。
 *
 * function measureAlphaCoverage(pixels, w, h, samplingRate):
 *   foreground = 0, semitransparent = 0, transparent = 0
 *   step = round(1 / samplingRate)
 *   for i in 0..w*h step step:
 *     alpha = pixels[i*4+3]
 *     if alpha > 200: foreground++
 *     elif alpha > 30: semitransparent++
 *     else: transparent++
 *   return { foregroundRatio, semiRatio, transparentRatio }
 */
function measureAlphaCoverage(
  pixels: Buffer,
  w: number,
  h: number,
  samplingRate: number
): { foreground: number; semi: number; transparent: number; histogram: number[] } {
  const total = w * h;
  const step = Math.max(1, Math.round(1 / samplingRate));
  let fg = 0, semi = 0, trans = 0, sampled = 0;

  // 256-bin alpha histogram
  const histogram = new Array(256).fill(0);

  for (let i = 0; i < total; i += step) {
    const alpha = pixels[i * 4 + 3];
    histogram[alpha]++;
    sampled++;

    if (alpha > 200) fg++;
    else if (alpha > 30) semi++;
    else trans++;
  }

  return {
    foreground: fg / sampled,
    semi: semi / sampled,
    transparent: trans / sampled,
    histogram,
  };
}

// ──────────────────────────────────────────────────────────────────────
// §5  绿色残留检测
// ──────────────────────────────────────────────────────────────────────

/**
 * 检测处理后图像中残留的绿色像素
 *
 * 参考 OBS Studio 的 chroma key preview mode:
 *   在前景区域(alpha > threshold)内搜索绿色色相像素,
 *   残留率 = 绿色前景像素 / 总前景像素
 *
 * function detectGreenResidue(pixels, w, h, cfg, samplingRate):
 *   greenCount = 0, foregroundCount = 0
 *   for sampled pixels:
 *     if alpha > 128:
 *       foregroundCount++
 *       [h, s, v] = rgbToHsv(r, g, b)
 *       if h in greenRange and s > satMin:
 *         greenCount++
 *   return greenCount / foregroundCount
 */
function detectGreenResidue(
  pixels: Buffer,
  w: number,
  h: number,
  greenHueRange: [number, number],
  greenSatMin: number,
  samplingRate: number
): { residueRatio: number; greenPixelCount: number; foregroundCount: number } {
  const total = w * h;
  const step = Math.max(1, Math.round(1 / samplingRate));
  let greenCount = 0, fgCount = 0;

  for (let i = 0; i < total; i += step) {
    const pIdx = i * 4;
    const alpha = pixels[pIdx + 3];
    if (alpha <= 128) continue;

    fgCount++;
    const r = pixels[pIdx], g = pixels[pIdx + 1], b = pixels[pIdx + 2];
    const [hue, sat] = rgbToHsv(r, g, b);

    if (hue >= greenHueRange[0] && hue <= greenHueRange[1] && sat > greenSatMin) {
      greenCount++;
    }
  }

  return {
    residueRatio: fgCount > 0 ? greenCount / fgCount : 0,
    greenPixelCount: greenCount,
    foregroundCount: fgCount,
  };
}

// ──────────────────────────────────────────────────────────────────────
// §6  边缘质量评估
// ──────────────────────────────────────────────────────────────────────

/**
 * 评估 alpha 边缘的平滑度和锐度
 *
 * 指标:
 *   1. 边缘过渡宽度: 理想 2-4px, 过宽说明模糊, 过窄说明锯齿
 *   2. Alpha 梯度一致性: 理想的边缘应该有平滑渐变
 *   3. 孤立像素检测: 边缘附近不应有孤立的半透明像素
 *
 * function measureEdgeQuality(pixels, w, h, samplingRate):
 *   找到所有边缘像素 (alpha 在 30-225 之间)
 *   对每个边缘像素:
 *     计算与邻域的 alpha 梯度
 *     检查梯度方向的一致性
 *   score = weighted_average(transition_quality, gradient_consistency, no_isolated)
 */
function measureEdgeQuality(
  pixels: Buffer,
  w: number,
  h: number,
  samplingRate: number
): { score: number; avgTransitionWidth: number; isolatedPixelRatio: number } {
  const step = Math.max(1, Math.round(1 / Math.sqrt(samplingRate)));
  let transitionWidths: number[] = [];
  let isolatedCount = 0;
  let edgePixelCount = 0;

  for (let y = 2; y < h - 2; y += step) {
    for (let x = 2; x < w - 2; x += step) {
      const idx = y * w + x;
      const alpha = pixels[idx * 4 + 3];

      // 只检查边缘附近
      if (alpha <= 30 || alpha >= 225) continue;
      edgePixelCount++;

      // 测量过渡宽度: 沿 x 方向找从 0→255 的跨度
      let leftX = x, rightX = x;
      while (leftX > 0 && pixels[(y * w + leftX) * 4 + 3] > 10) leftX--;
      while (rightX < w - 1 && pixels[(y * w + rightX) * 4 + 3] < 245) rightX++;
      transitionWidths.push(rightX - leftX);

      // 孤立像素检测: 如果4邻域都是全透明或全不透明, 这是个孤立半透明
      const neighbors = [
        pixels[((y - 1) * w + x) * 4 + 3],
        pixels[((y + 1) * w + x) * 4 + 3],
        pixels[(y * w + x - 1) * 4 + 3],
        pixels[(y * w + x + 1) * 4 + 3],
      ];
      const allExtreme = neighbors.every(n => n < 10 || n > 245);
      if (allExtreme) isolatedCount++;
    }
  }

  if (edgePixelCount === 0) {
    return { score: 1, avgTransitionWidth: 0, isolatedPixelRatio: 0 };
  }

  // 过渡宽度评分: 2-6px 是理想
  const avgWidth = transitionWidths.reduce((a, b) => a + b, 0) / transitionWidths.length;
  const widthScore = avgWidth >= 2 && avgWidth <= 6
    ? 1.0
    : avgWidth < 2
      ? avgWidth / 2        // 过窄 (锯齿)
      : Math.max(0, 1 - (avgWidth - 6) / 20);  // 过宽 (模糊)

  // 孤立像素评分
  const isolatedRatio = isolatedCount / edgePixelCount;
  const isolatedScore = Math.max(0, 1 - isolatedRatio * 10);

  const score = widthScore * 0.6 + isolatedScore * 0.4;

  return {
    score: Math.max(0, Math.min(1, score)),
    avgTransitionWidth: avgWidth,
    isolatedPixelRatio: isolatedRatio,
  };
}

// ──────────────────────────────────────────────────────────────────────
// §7  Alpha 分布分析
// ──────────────────────────────────────────────────────────────────────

/**
 * 分析 alpha 通道的分布合理性
 *
 * 理想的去背景图:
 *   - 大部分像素要么 α=0 (背景) 要么 α=255 (前景)
 *   - 只有边缘处有少量中间值 (α ∈ [1, 254])
 *   - 呈现双峰分布 (bimodal)
 *
 * 评分:
 *   bimodality = (count_at_0 + count_at_255) / total
 *   高 bimodality = 清晰的前景/背景分离
 */
function analyzeAlphaDistribution(
  histogram: number[]
): { bimodality: number; score: number } {
  const total = histogram.reduce((a, b) => a + b, 0);
  if (total === 0) return { bimodality: 0, score: 0 };

  const extremeCount = histogram[0] + histogram[255];
  const bimodality = extremeCount / total;

  // 0.8+ bimodality 是好的 (80% 像素在两端)
  const score = Math.min(1, bimodality / 0.8);

  return { bimodality, score };
}

// ──────────────────────────────────────────────────────────────────────
// §8  评分 → 等级 + 建议
// ──────────────────────────────────────────────────────────────────────

function scoreToGrade(score: number): 'A' | 'B' | 'C' | 'D' | 'F' {
  if (score >= 90) return 'A';
  if (score >= 75) return 'B';
  if (score >= 60) return 'C';
  if (score >= 40) return 'D';
  return 'F';
}

function generateSuggestions(metrics: QualityMetrics['dimensions']): string[] {
  const suggestions: string[] = [];

  if (metrics.greenResidue.status === 'poor') {
    suggestions.push(
      '绿色残留较多。建议增加 chroma.spillSuppressionStrength 或切换到 rembg 模式。'
    );
  }

  if (metrics.edgeQuality.status === 'poor') {
    suggestions.push(
      '边缘质量较差。建议增加 edge.antiAliasRadius 和 edge.defringeStrength。'
    );
  }

  if (metrics.alphaCoverage.status === 'poor') {
    const val = metrics.alphaCoverage.value;
    if (val < 0.05) {
      suggestions.push(
        '前景区域过小, 可能过度移除了内容。建议降低 chroma.hueTolerance。'
      );
    } else if (val > 0.85) {
      suggestions.push(
        '背景残留较多。建议提高 chroma.hueTolerance 或 chroma.saturationMin。'
      );
    }
  }

  if (metrics.alphaDistribution.status !== 'good') {
    suggestions.push(
      'Alpha 分布不够双峰化。建议增加 edge.edgeContrast 使边缘更锐利。'
    );
  }

  if (suggestions.length === 0) {
    suggestions.push('质量良好, 无需调整。');
  }

  return suggestions;
}

function generateSummary(grade: string, dimensions: QualityMetrics['dimensions']): string {
  const gradeDesc: Record<string, string> = {
    A: '优秀 - 接近专业级别',
    B: '良好 - 适合大多数用途',
    C: '可接受 - 建议进一步优化',
    D: '较差 - 需要调整参数',
    F: '不通过 - 需要重新处理',
  };

  const desc = gradeDesc[grade] || '未知';
  const worst = Object.values(dimensions)
    .filter(d => d.status === 'poor')
    .map(d => d.name);

  let summary = `整体质量 ${grade} (${desc})。`;
  if (worst.length > 0) {
    summary += ` 需要关注: ${worst.join(', ')}。`;
  }
  return summary;
}

// ──────────────────────────────────────────────────────────────────────
// §9  QualityValidator 主类
// ──────────────────────────────────────────────────────────────────────

export class QualityValidator {
  private config: QualityConfig;

  constructor(config?: Partial<QualityConfig>) {
    this.config = { ...DEFAULT_QUALITY_CONFIG, ...config };
  }

  /**
   * 对处理后的图像执行质量验证
   */
  async validate(processedImage: Buffer): Promise<QualityMetrics> {
    const startTime = Date.now();
    const cfg = this.config;

    // 解码
    const img = sharp(processedImage).ensureAlpha();
    const meta = await img.metadata();
    const w = meta.width!, h = meta.height!;
    const pixels = await img.raw().toBuffer();

    // ── Alpha 覆盖率 ──
    const coverage = measureAlphaCoverage(pixels, w, h, cfg.samplingRate);
    const coverageInRange =
      coverage.foreground >= cfg.expectedAlphaCoverage[0] &&
      coverage.foreground <= cfg.expectedAlphaCoverage[1];
    const coverageScore = coverageInRange ? 100 : 50;
    const alphaCoverageResult: MetricResult = {
      name: 'Alpha覆盖率',
      value: coverage.foreground,
      score: coverageScore,
      weight: 0.2,
      status: coverageInRange ? 'good' : coverage.foreground < 0.02 ? 'poor' : 'acceptable',
      description: `前景占比 ${(coverage.foreground * 100).toFixed(1)}%, 半透明 ${(coverage.semi * 100).toFixed(1)}%`,
    };

    // ── 绿色残留 ──
    const residue = detectGreenResidue(
      pixels, w, h, cfg.greenHueRange, cfg.greenSatMin, cfg.samplingRate
    );
    const residueOk = residue.residueRatio <= cfg.maxGreenResidue;
    const residueScore = residueOk ? 100 : Math.max(0, 100 - residue.residueRatio * 1000);
    const greenResidueResult: MetricResult = {
      name: '绿色残留',
      value: residue.residueRatio,
      score: residueScore,
      weight: 0.3,
      status: residueOk ? 'good' : residue.residueRatio > 0.05 ? 'poor' : 'acceptable',
      description: `残留绿色 ${(residue.residueRatio * 100).toFixed(2)}% (${residue.greenPixelCount}px)`,
    };

    // ── 边缘质量 ──
    const edge = measureEdgeQuality(pixels, w, h, cfg.samplingRate);
    const edgeOk = edge.score >= cfg.minEdgeQuality;
    const edgeScore = edge.score * 100;
    const edgeQualityResult: MetricResult = {
      name: '边缘质量',
      value: edge.score,
      score: edgeScore,
      weight: 0.25,
      status: edgeOk ? (edge.score > 0.7 ? 'good' : 'acceptable') : 'poor',
      description: `过渡宽度 ${edge.avgTransitionWidth.toFixed(1)}px, 孤立像素 ${(edge.isolatedPixelRatio * 100).toFixed(1)}%`,
    };

    // ── Alpha 分布 ──
    const dist = analyzeAlphaDistribution(coverage.histogram);
    const distScore = dist.score * 100;
    const alphaDistResult: MetricResult = {
      name: 'Alpha分布',
      value: dist.bimodality,
      score: distScore,
      weight: 0.1,
      status: dist.score > 0.7 ? 'good' : dist.score > 0.4 ? 'acceptable' : 'poor',
      description: `双峰度 ${(dist.bimodality * 100).toFixed(1)}%`,
    };

    // ── 组件完整性 (简化: 检查是否有有效前景) ──
    const completenessScore = coverage.foreground > 0.01 ? 100 : 0;
    const componentResult: MetricResult = {
      name: '组件完整性',
      value: coverage.foreground > 0.01 ? 1 : 0,
      score: completenessScore,
      weight: 0.15,
      status: completenessScore > 50 ? 'good' : 'poor',
      description: coverage.foreground > 0.01 ? '检测到有效前景' : '未检测到前景内容',
    };

    // ── 汇总 ──
    const dimensions = {
      alphaCoverage: alphaCoverageResult,
      greenResidue: greenResidueResult,
      edgeQuality: edgeQualityResult,
      componentCompleteness: componentResult,
      alphaDistribution: alphaDistResult,
    };

    const allMetrics = Object.values(dimensions);
    const totalWeight = allMetrics.reduce((s, m) => s + m.weight, 0);
    const overallScore = allMetrics.reduce((s, m) => s + m.score * m.weight, 0) / totalWeight;

    const grade = scoreToGrade(overallScore);
    const passed = overallScore >= cfg.minPassScore;
    const suggestions = generateSuggestions(dimensions);
    const summary = generateSummary(grade, dimensions);

    return {
      overallScore: Math.round(overallScore * 10) / 10,
      grade,
      passed,
      dimensions,
      summary,
      suggestions,
      validationTimeMs: Date.now() - startTime,
    };
  }

  /**
   * 对比两张图片的差异 (用于 A/B 测试或回归测试)
   */
  async compareImages(
    imageA: Buffer,
    imageB: Buffer
  ): Promise<{
    pixelDiffRatio: number;
    alphaDiffRatio: number;
    maxDiff: number;
    similar: boolean;
  }> {
    const metaA = await sharp(imageA).metadata();
    const metaB = await sharp(imageB).metadata();

    // 尺寸不同则先 resize
    const w = Math.min(metaA.width!, metaB.width!);
    const h = Math.min(metaA.height!, metaB.height!);

    const pixelsA = await sharp(imageA).resize(w, h).ensureAlpha().raw().toBuffer();
    const pixelsB = await sharp(imageB).resize(w, h).ensureAlpha().raw().toBuffer();

    let diffPixels = 0, alphaDiffs = 0, maxDiff = 0;
    const total = w * h;
    const threshold = 10; // 像素差异阈值

    for (let i = 0; i < total; i++) {
      const pA = i * 4, pB = i * 4;
      const dr = Math.abs(pixelsA[pA] - pixelsB[pB]);
      const dg = Math.abs(pixelsA[pA + 1] - pixelsB[pB + 1]);
      const db = Math.abs(pixelsA[pA + 2] - pixelsB[pB + 2]);
      const da = Math.abs(pixelsA[pA + 3] - pixelsB[pB + 3]);

      const maxChannel = Math.max(dr, dg, db);
      maxDiff = Math.max(maxDiff, maxChannel, da);

      if (maxChannel > threshold) diffPixels++;
      if (da > threshold) alphaDiffs++;
    }

    const pixelDiffRatio = diffPixels / total;
    const alphaDiffRatio = alphaDiffs / total;

    return {
      pixelDiffRatio,
      alphaDiffRatio,
      maxDiff,
      similar: pixelDiffRatio < 0.05 && alphaDiffRatio < 0.02,
    };
  }

  /**
   * 性能基准测试
   */
  async benchmark(
    processFn: (input: Buffer) => Promise<Buffer>,
    testImage: Buffer,
    iterations: number = 5
  ): Promise<BenchmarkResult> {
    const meta = await sharp(testImage).metadata();
    const inputPixels = meta.width! * meta.height!;

    const durations: number[] = [];
    const memUsages: number[] = [];

    for (let i = 0; i < iterations; i++) {
      const memBefore = process.memoryUsage().heapUsed;
      const start = Date.now();
      await processFn(testImage);
      durations.push(Date.now() - start);
      memUsages.push((process.memoryUsage().heapUsed - memBefore) / (1024 * 1024));
    }

    // 去掉最高和最低, 取平均
    durations.sort((a, b) => a - b);
    const trimmed = durations.slice(1, -1);
    const avgDuration = trimmed.reduce((a, b) => a + b, 0) / trimmed.length;
    const peakMemory = Math.max(...memUsages);

    return {
      stage: 'benchmark',
      inputPixels,
      durationMs: Math.round(avgDuration),
      throughputMpxPerSec: Math.round((inputPixels / avgDuration) * 1000) / 1000000,
      peakMemoryMb: Math.round(peakMemory * 10) / 10,
    };
  }

  updateConfig(partial: Partial<QualityConfig>): void {
    this.config = { ...this.config, ...partial };
  }
}

export default QualityValidator;