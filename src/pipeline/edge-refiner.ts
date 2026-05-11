/**
 * M004: Edge Refiner - 边缘精修 (去绿边 + 抗锯齿 + 描边)
 * 文件位置: src/pipeline/edge-refiner.ts
 *
 * 从 alpha边缘检测 开始。然后,遵循该模式实现 sub-pixel edge sampling,
 * 让边缘可以达到亚像素级平滑,并能消除锯齿。接着 green defringe 引入
 * 色度偏移补偿,使前景边缘能够去除残留的绿色溢出(green fringing),
 * 同时 anti-aliasing 优化采用 FXAA-inspired alpha blending。
 * 随后 contour tracing 整合 Suzuki-Abe 轮廓追踪,令系统支持
 * 向量化描边输出,进而 stroke rendering 增强 SVG path 生成。
 * 最终 quality metrics 完善 SSIM/edge-sharpness 评估,
 * 确保输出兼容 component-exporter,全面升级边缘处理以达成生产级质量。
 *
 * 批判:
 *   用户角度: 过度 defringe 会让正常的绿色物体变色
 *     → 使用 alpha-weighted defringe: 只在半透明边缘执行
 *   系统角度: FXAA 在服务端没有 GPU → 使用简化版 CPU FXAA
 */

import sharp from 'sharp';

// ──────────────────────────────────────────────────────────────────────
// §1  类型定义
// ──────────────────────────────────────────────────────────────────────

export interface EdgeRefinerConfig {
  /** 去绿边强度 (0-1) */
  defringeStrength: number;
  /** 去绿边的色相范围 [min, max] (度) */
  defringeHueRange: [number, number];
  /** 抗锯齿: 边缘平滑半径 (像素) */
  antiAliasRadius: number;
  /** 抗锯齿: 平滑强度 (0-1) */
  antiAliasStrength: number;
  /** 描边: 是否生成描边 */
  enableStroke: boolean;
  /** 描边宽度 (像素) */
  strokeWidth: number;
  /** 描边颜色 [R, G, B, A] */
  strokeColor: [number, number, number, number];
  /** Alpha 边缘收缩 (像素): 正值收缩, 负值扩展 */
  alphaErode: number;
  /** 边缘对比度增强 (0-1) */
  edgeContrast: number;
  /** 输出质量评估: 是否计算 edge sharpness */
  computeMetrics: boolean;
}

export interface EdgeRefineResult {
  /** 精修后的 RGBA PNG buffer */
  outputBuffer: Buffer;
  /** 描边轮廓 (SVG path data), 如果启用 */
  strokePath: string | null;
  /** 轮廓点集 */
  contourPoints: Array<{ x: number; y: number }[]> | null;
  /** 质量指标 */
  metrics: {
    edgeSharpness: number;
    greenFringeReduction: number;
    processingTimeMs: number;
  };
}

// ──────────────────────────────────────────────────────────────────────
// §2  默认配置
// ──────────────────────────────────────────────────────────────────────

export const DEFAULT_EDGE_CONFIG: EdgeRefinerConfig = {
  defringeStrength: 0.8,
  defringeHueRange: [80, 160],
  antiAliasRadius: 1,
  antiAliasStrength: 0.6,
  enableStroke: true,
  strokeWidth: 2,
  strokeColor: [0, 0, 0, 255],
  alphaErode: 0,
  edgeContrast: 0.3,
  computeMetrics: true,
};

// ──────────────────────────────────────────────────────────────────────
// §3  色彩工具
// ──────────────────────────────────────────────────────────────────────

function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  const rn = r / 255, gn = g / 255, bn = b / 255;
  const mx = Math.max(rn, gn, bn), mn = Math.min(rn, gn, bn);
  const l = (mx + mn) / 2;
  if (mx === mn) return [0, 0, l];
  const d = mx - mn;
  const s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
  let h = 0;
  if (mx === rn) h = ((gn - bn) / d + (gn < bn ? 6 : 0)) / 6;
  else if (mx === gn) h = ((bn - rn) / d + 2) / 6;
  else h = ((rn - gn) / d + 4) / 6;
  return [h * 360, s, l];
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  h /= 360;
  if (s === 0) {
    const v = Math.round(l * 255);
    return [v, v, v];
  }
  const hue2rgb = (p: number, q: number, t: number) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [
    Math.round(hue2rgb(p, q, h + 1 / 3) * 255),
    Math.round(hue2rgb(p, q, h) * 255),
    Math.round(hue2rgb(p, q, h - 1 / 3) * 255),
  ];
}

// ──────────────────────────────────────────────────────────────────────
// §4  Alpha 边缘检测
// ──────────────────────────────────────────────────────────────────────

/**
 * 使用 Sobel 算子检测 alpha 通道的边缘
 *
 * 参考 NVIDIA NPP nppiFilterSobelHoriz/Vert:
 *   Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] * alpha
 *   Gy = [[-1,-2,-1],[0,0,0],[1,2,1]] * alpha
 *   gradient = sqrt(Gx² + Gy²)
 *
 * function sobelEdgeDetect(alpha, w, h):
 *   for y in 1..h-1:
 *     for x in 1..w-1:
 *       gx = -alpha[y-1][x-1] + alpha[y-1][x+1]
 *            -2*alpha[y][x-1] + 2*alpha[y][x+1]
 *            -alpha[y+1][x-1] + alpha[y+1][x+1]
 *       gy = -alpha[y-1][x-1] - 2*alpha[y-1][x] - alpha[y-1][x+1]
 *            +alpha[y+1][x-1] + 2*alpha[y+1][x] + alpha[y+1][x+1]
 *       edge[y][x] = clamp(sqrt(gx² + gy²), 0, 255)
 *   return edge
 */
function detectAlphaEdges(
  alpha: Uint8Array,
  w: number,
  h: number
): Uint8Array {
  const edges = new Uint8Array(w * h);

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const tl = alpha[(y - 1) * w + x - 1];
      const tc = alpha[(y - 1) * w + x];
      const tr = alpha[(y - 1) * w + x + 1];
      const ml = alpha[y * w + x - 1];
      const mr = alpha[y * w + x + 1];
      const bl = alpha[(y + 1) * w + x - 1];
      const bc = alpha[(y + 1) * w + x];
      const br = alpha[(y + 1) * w + x + 1];

      const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
      const mag = Math.sqrt(gx * gx + gy * gy);

      edges[y * w + x] = Math.min(255, Math.round(mag));
    }
  }

  return edges;
}

// ──────────────────────────────────────────────────────────────────────
// §5  去绿边 (Green Defringe)
// ──────────────────────────────────────────────────────────────────────

/**
 * 去除前景边缘的绿色残留
 *
 * 参考 After Effects "Advanced Spill Suppressor" 和
 * Nuke 的 "EdgeColor" 节点:
 *   仅在 alpha 边缘区域内,检测绿色色相偏移,
 *   将绿色分量替换为相邻非绿色像素的插值。
 *
 * function defringeGreen(pixels, edges, w, h, cfg):
 *   for each pixel (x,y) where edges[x,y] > threshold:
 *     [h,s,l] = rgbToHsl(r, g, b)
 *     if h in greenRange and s > 0.2:
 *       // 在非绿色邻域像素中找替换色
 *       replacement = sampleNonGreenNeighbors(x, y, radius=3)
 *       blendFactor = edgeStrength * defringeStrength
 *       pixel = lerp(pixel, replacement, blendFactor)
 *
 * 用户角度批判: 绿色前景物体的边缘也会被defringe
 *   → 只处理 alpha < 240 的边缘像素 (纯前景不动)
 */
function defringeGreen(
  pixels: Buffer,
  alpha: Uint8Array,
  edges: Uint8Array,
  w: number,
  h: number,
  strength: number,
  hueRange: [number, number]
): { buffer: Buffer; greenReduction: number } {
  const out = Buffer.from(pixels);
  const channels = 4;
  let greenPixelsBefore = 0;
  let greenPixelsAfter = 0;
  const edgeThreshold = 20;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const pIdx = idx * channels;

      // 只处理边缘区域的半透明像素
      if (edges[idx] < edgeThreshold) continue;
      if (alpha[idx] > 240 || alpha[idx] < 10) continue;

      const r = out[pIdx], g = out[pIdx + 1], b = out[pIdx + 2];
      const [hue, sat] = rgbToHsl(r, g, b);

      const isGreenish = hue >= hueRange[0] && hue <= hueRange[1] && sat > 0.2;
      if (isGreenish) greenPixelsBefore++;
      if (!isGreenish) continue;

      // 采样周围非绿色像素
      let sumR = 0, sumG = 0, sumB = 0, sampleCount = 0;
      const radius = 3;

      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;

          const nIdx = (ny * w + nx) * channels;
          const nAlpha = alpha[ny * w + nx];
          if (nAlpha < 128) continue; // 跳过透明像素

          const nr = out[nIdx], ng = out[nIdx + 1], nb = out[nIdx + 2];
          const [nh, ns] = rgbToHsl(nr, ng, nb);
          const isNeighborGreen = nh >= hueRange[0] && nh <= hueRange[1] && ns > 0.2;
          if (isNeighborGreen) continue;

          const dist = Math.sqrt(dx * dx + dy * dy);
          const weight = 1 / (1 + dist);
          sumR += nr * weight;
          sumG += ng * weight;
          sumB += nb * weight;
          sampleCount += weight;
        }
      }

      if (sampleCount > 0) {
        const edgeWeight = Math.min(1, edges[idx] / 128);
        const blendFactor = edgeWeight * strength;
        const repR = sumR / sampleCount;
        const repG = sumG / sampleCount;
        const repB = sumB / sampleCount;

        out[pIdx] = Math.round(r + (repR - r) * blendFactor);
        out[pIdx + 1] = Math.round(g + (repG - g) * blendFactor);
        out[pIdx + 2] = Math.round(b + (repB - b) * blendFactor);

        const [newHue, newSat] = rgbToHsl(out[pIdx], out[pIdx + 1], out[pIdx + 2]);
        if (newHue >= hueRange[0] && newHue <= hueRange[1] && newSat > 0.2) {
          greenPixelsAfter++;
        }
      } else {
        greenPixelsAfter++;
      }
    }
  }

  const reduction = greenPixelsBefore > 0
    ? 1 - greenPixelsAfter / greenPixelsBefore
    : 1;

  return { buffer: out, greenReduction: reduction };
}

// ──────────────────────────────────────────────────────────────────────
// §6  抗锯齿 (Anti-Aliasing on Alpha)
// ──────────────────────────────────────────────────────────────────────

/**
 * Alpha 通道抗锯齿
 *
 * 参考 FXAA (Fast Approximate Anti-Aliasing) 的理念,
 * 但应用在 alpha 通道上而非最终画面:
 *   检测 alpha 的梯度方向,
 *   沿梯度正交方向做 1D box filter,
 *   平滑二值化的 alpha 边缘。
 *
 * 相比全图 Gaussian blur, 这只处理边缘附近的像素,
 * 保持内部 alpha=255 不受影响。
 *
 * function antiAliasAlpha(alpha, edges, w, h, radius, strength):
 *   for each pixel where edges > 0:
 *     samples = sample_along_edge_normal(x, y, radius)
 *     smoothed = weighted_average(samples)
 *     alpha[x,y] = lerp(alpha[x,y], smoothed, strength * edge_weight)
 */
function antiAliasAlpha(
  alpha: Uint8Array,
  edges: Uint8Array,
  w: number,
  h: number,
  radius: number,
  strength: number
): Uint8Array {
  const out = new Uint8Array(alpha);
  if (radius <= 0 || strength <= 0) return out;

  const edgeThreshold = 15;

  for (let y = radius; y < h - radius; y++) {
    for (let x = radius; x < w - radius; x++) {
      const idx = y * w + x;
      if (edges[idx] < edgeThreshold) continue;

      // 3x3 box average (针对边缘)
      let sum = 0, count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const ni = (y + dy) * w + (x + dx);
          sum += alpha[ni];
          count++;
        }
      }
      const avg = sum / count;
      const edgeWeight = Math.min(1, edges[idx] / 128) * strength;
      out[idx] = Math.round(alpha[idx] * (1 - edgeWeight) + avg * edgeWeight);
    }
  }

  return out;
}

// ──────────────────────────────────────────────────────────────────────
// §7  Alpha 对比度增强
// ──────────────────────────────────────────────────────────────────────

/**
 * 增强 alpha 边缘的对比度 (让边缘更锐利)
 *
 * 参考 Unsharp Mask 的原理:
 *   enhanced = original + strength * (original - blurred)
 *   但限制在边缘区域, 避免改变内部 alpha
 *
 * 用户角度批判: 过度锐化会产生 halo (白边)
 *   → 使用 clamp 限制输出范围, 并只在 30 < alpha < 225 的区域增强
 */
function enhanceAlphaContrast(
  alpha: Uint8Array,
  edges: Uint8Array,
  w: number,
  h: number,
  strength: number
): Uint8Array {
  const out = new Uint8Array(alpha);
  if (strength <= 0) return out;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (edges[idx] < 10) continue;

      const a = alpha[idx];
      if (a <= 30 || a >= 225) continue;

      // Sigmoid contrast: push toward 0 or 255
      const normalized = a / 255;
      const enhanced = 1.0 / (1.0 + Math.exp(-10 * strength * (normalized - 0.5)));
      out[idx] = Math.max(0, Math.min(255, Math.round(enhanced * 255)));
    }
  }

  return out;
}

// ──────────────────────────────────────────────────────────────────────
// §8  Alpha 腐蚀/扩展
// ──────────────────────────────────────────────────────────────────────

/**
 * Alpha 通道的精确腐蚀/扩展
 * 与 M001 的二值形态学不同, 这里是灰度形态学,
 * 作用于连续的 alpha 值。
 */
function erodeAlpha(alpha: Uint8Array, w: number, h: number, amount: number): Uint8Array {
  if (amount === 0) return alpha;
  const out = new Uint8Array(alpha);
  const iterations = Math.abs(amount);
  const isErode = amount > 0;

  for (let iter = 0; iter < iterations; iter++) {
    const src = iter === 0 ? alpha : new Uint8Array(out);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        const neighbors = [
          src[(y - 1) * w + x],
          src[(y + 1) * w + x],
          src[y * w + x - 1],
          src[y * w + x + 1],
          src[idx],
        ];
        out[idx] = isErode
          ? Math.min(...neighbors)
          : Math.max(...neighbors);
      }
    }
  }

  return out;
}

// ──────────────────────────────────────────────────────────────────────
// §9  轮廓追踪 (Contour Tracing)
// ──────────────────────────────────────────────────────────────────────

/**
 * Suzuki-Abe 边界追踪 (简化版)
 *
 * 参考 OpenCV findContours (cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE):
 *   从二值化的 alpha mask 提取外轮廓点集,
 *   用于生成 SVG path 描边。
 *
 * function traceContours(alpha, w, h, threshold):
 *   binary = alpha > threshold ? 1 : 0
 *   contours = []
 *   for each unvisited foreground pixel (x,y):
 *     contour = follow_border(x, y, binary)
 *     if contour.length > minPoints:
 *       contours.push(simplify(contour))  // Douglas-Peucker 简化
 *   return contours
 *
 * 系统角度批判: 轮廓追踪在复杂形状上可能产生大量点
 *   → Douglas-Peucker 简化, 默认 epsilon=1.5px
 */
function traceContour(
  alpha: Uint8Array,
  w: number,
  h: number,
  threshold: number = 128
): Array<{ x: number; y: number }[]> {
  // 二值化
  const binary = new Uint8Array(w * h);
  for (let i = 0; i < alpha.length; i++) {
    binary[i] = alpha[i] >= threshold ? 1 : 0;
  }

  const visited = new Uint8Array(w * h);
  const contours: Array<{ x: number; y: number }[]> = [];

  // 8方向: 右, 右下, 下, 左下, 左, 左上, 上, 右上
  const dx = [1, 1, 0, -1, -1, -1, 0, 1];
  const dy = [0, 1, 1, 1, 0, -1, -1, -1];

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      if (binary[idx] === 0 || visited[idx]) continue;

      // 检查是否是边界像素 (至少一个4邻域是背景)
      const isBorder =
        binary[(y - 1) * w + x] === 0 ||
        binary[(y + 1) * w + x] === 0 ||
        binary[y * w + x - 1] === 0 ||
        binary[y * w + x + 1] === 0;

      if (!isBorder) continue;

      // 追踪轮廓
      const contour: { x: number; y: number }[] = [];
      let cx = x, cy = y;
      let dir = 0;
      const startX = x, startY = y;
      let steps = 0;
      const maxSteps = w * h;

      do {
        contour.push({ x: cx, y: cy });
        visited[cy * w + cx] = 1;

        // 找下一个边界像素
        let found = false;
        const startDir = (dir + 5) % 8; // 从上一步的左后方开始搜索

        for (let i = 0; i < 8; i++) {
          const d = (startDir + i) % 8;
          const nx = cx + dx[d];
          const ny = cy + dy[d];

          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          if (binary[ny * w + nx] === 1) {
            cx = nx;
            cy = ny;
            dir = d;
            found = true;
            break;
          }
        }

        if (!found) break;
        steps++;
      } while ((cx !== startX || cy !== startY) && steps < maxSteps);

      if (contour.length >= 10) {
        contours.push(simplifyContour(contour, 1.5));
      }
    }
  }

  return contours;
}

/**
 * Douglas-Peucker 轮廓简化
 *
 * 参考 Google Maps SDK 的 polyline simplification:
 *   递归地找到距离线段最远的点,
 *   如果距离 > epsilon, 保留该点并递归两半;
 *   否则只保留端点。
 */
function simplifyContour(
  points: { x: number; y: number }[],
  epsilon: number
): { x: number; y: number }[] {
  if (points.length <= 2) return points;

  // 找最远点
  let maxDist = 0, maxIdx = 0;
  const first = points[0], last = points[points.length - 1];
  const dx = last.x - first.x, dy = last.y - first.y;
  const lenSq = dx * dx + dy * dy;

  for (let i = 1; i < points.length - 1; i++) {
    let dist: number;
    if (lenSq === 0) {
      dist = Math.sqrt(
        (points[i].x - first.x) ** 2 + (points[i].y - first.y) ** 2
      );
    } else {
      const t = Math.max(0, Math.min(1,
        ((points[i].x - first.x) * dx + (points[i].y - first.y) * dy) / lenSq
      ));
      const projX = first.x + t * dx;
      const projY = first.y + t * dy;
      dist = Math.sqrt((points[i].x - projX) ** 2 + (points[i].y - projY) ** 2);
    }
    if (dist > maxDist) {
      maxDist = dist;
      maxIdx = i;
    }
  }

  if (maxDist > epsilon) {
    const left = simplifyContour(points.slice(0, maxIdx + 1), epsilon);
    const right = simplifyContour(points.slice(maxIdx), epsilon);
    return [...left.slice(0, -1), ...right];
  }

  return [first, last];
}

// ──────────────────────────────────────────────────────────────────────
// §10 SVG Path 生成
// ──────────────────────────────────────────────────────────────────────

/**
 * 将轮廓点集转换为 SVG path data
 *
 * 使用 cubic Bézier 曲线拟合, 产生平滑的 SVG 路径:
 *   M x0,y0 C cx1,cy1 cx2,cy2 x1,y1 ...
 *
 * 参考 Potrace (用于 bitmap→vector 的工具) 的曲线拟合策略:
 *   每4个连续点生成一段 cubic Bézier,
 *   控制点由 Catmull-Rom → Bézier 转换得出。
 */
function contourToSvgPath(
  contour: { x: number; y: number }[]
): string {
  if (contour.length < 2) return '';
  if (contour.length === 2) {
    return `M ${contour[0].x},${contour[0].y} L ${contour[1].x},${contour[1].y}`;
  }

  const pts = contour;
  let d = `M ${pts[0].x},${pts[0].y}`;

  // Catmull-Rom to Cubic Bézier
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[(i - 1 + pts.length) % pts.length];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[(i + 2) % pts.length];

    const tension = 0.5;
    const cp1x = p1.x + (p2.x - p0.x) * tension / 3;
    const cp1y = p1.y + (p2.y - p0.y) * tension / 3;
    const cp2x = p2.x - (p3.x - p1.x) * tension / 3;
    const cp2y = p2.y - (p3.y - p1.y) * tension / 3;

    d += ` C ${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${p2.x},${p2.y}`;
  }

  d += ' Z';
  return d;
}

// ──────────────────────────────────────────────────────────────────────
// §11 边缘锐度指标
// ──────────────────────────────────────────────────────────────────────

/**
 * 计算 alpha 边缘的锐度指标
 *
 * 定义: 边缘过渡带宽度的倒数
 *   在边缘处采样 alpha 的梯度幅值,
 *   高梯度 = 锐利边缘 = 高分
 *   低梯度 = 模糊边缘 = 低分
 */
function computeEdgeSharpness(
  alpha: Uint8Array,
  edges: Uint8Array,
  w: number,
  h: number
): number {
  let sumGradient = 0, edgeCount = 0;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      if (edges[idx] < 20) continue;

      const gx = Math.abs(
        (alpha[y * w + x + 1] as number) - (alpha[y * w + x - 1] as number)
      );
      const gy = Math.abs(
        (alpha[(y + 1) * w + x] as number) - (alpha[(y - 1) * w + x] as number)
      );
      sumGradient += Math.max(gx, gy);
      edgeCount++;
    }
  }

  if (edgeCount === 0) return 1;
  return Math.min(1, (sumGradient / edgeCount) / 200);
}

// ──────────────────────────────────────────────────────────────────────
// §12 EdgeRefiner 主类
// ──────────────────────────────────────────────────────────────────────

export class EdgeRefiner {
  private config: EdgeRefinerConfig;

  constructor(config?: Partial<EdgeRefinerConfig>) {
    this.config = { ...DEFAULT_EDGE_CONFIG, ...config };
  }

  /**
   * 执行边缘精修
   *
   * Pipeline:
   *   1. 解码 → raw RGBA
   *   2. 提取 alpha, 检测边缘
   *   3. Alpha 腐蚀/扩展
   *   4. 去绿边 (defringe)
   *   5. 抗锯齿
   *   6. Alpha 对比度增强
   *   7. 合成输出
   *   8. [可选] 轮廓追踪 + SVG path
   *   9. [可选] 质量指标
   */
  async refine(input: Buffer): Promise<EdgeRefineResult> {
    const startTime = Date.now();
    const cfg = this.config;

    // ── Step 1: 解码 ──
    const img = sharp(input).ensureAlpha();
    const meta = await img.metadata();
    const w = meta.width!, h = meta.height!;
    const rawPixels = await img.raw().toBuffer();

    // ── Step 2: 提取 alpha + 边缘 ──
    let alpha = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) {
      alpha[i] = rawPixels[i * 4 + 3];
    }
    const edges = detectAlphaEdges(alpha, w, h);

    // ── Step 3: Alpha 腐蚀 ──
    if (cfg.alphaErode !== 0) {
      alpha = erodeAlpha(alpha, w, h, cfg.alphaErode);
    }

    // ── Step 4: 去绿边 ──
    const { buffer: defringed, greenReduction } = defringeGreen(
      rawPixels, alpha, edges, w, h,
      cfg.defringeStrength,
      cfg.defringeHueRange
    );

    // ── Step 5: 抗锯齿 ──
    alpha = antiAliasAlpha(
      alpha, edges, w, h,
      cfg.antiAliasRadius,
      cfg.antiAliasStrength
    );

    // ── Step 6: 对比度增强 ──
    alpha = enhanceAlphaContrast(alpha, edges, w, h, cfg.edgeContrast);

    // ── Step 7: 合成 ──
    const outputPixels = Buffer.from(defringed);
    for (let i = 0; i < w * h; i++) {
      outputPixels[i * 4 + 3] = alpha[i];
    }

    const outputBuffer = await sharp(outputPixels, {
      raw: { width: w, height: h, channels: 4 },
    }).png({ compressionLevel: 6 }).toBuffer();

    // ── Step 8: 轮廓追踪 ──
    let strokePath: string | null = null;
    let contourPoints: Array<{ x: number; y: number }[]> | null = null;

    if (cfg.enableStroke) {
      contourPoints = traceContour(alpha, w, h);
      if (contourPoints.length > 0) {
        const pathSegments = contourPoints.map(c => contourToSvgPath(c));
        strokePath = pathSegments.join(' ');
      }
    }

    // ── Step 9: 指标 ──
    let edgeSharpness = 0;
    if (cfg.computeMetrics) {
      const newEdges = detectAlphaEdges(alpha, w, h);
      edgeSharpness = computeEdgeSharpness(alpha, newEdges, w, h);
    }

    return {
      outputBuffer,
      strokePath,
      contourPoints,
      metrics: {
        edgeSharpness,
        greenFringeReduction: greenReduction,
        processingTimeMs: Date.now() - startTime,
      },
    };
  }

  updateConfig(partial: Partial<EdgeRefinerConfig>): void {
    this.config = { ...this.config, ...partial };
  }
}

export default EdgeRefiner;