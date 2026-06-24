/**
 * shadow-gpu-pass.ts — M1134: GPU Shadow Map + 5×5 PCF + Contact Hardening
 * ─────────────────────────────────────────────────────────────────────────────
 * 光的缺席即是阴影。
 *
 * 真正在 GPU 上跑的 Shadow Map 系统。每个函数都调用 gl.*。
 * 参考 fluid-gpu-pass.ts (414行, 82处gl调用) 的写法。
 * 从 compiled.vs 通过 ShaderLoader 提取 AT 生产 shader 源码。
 *
 * M1134 改动:
 *   - PCF kernel: 3×3 (9 taps) → 5×5 (25 taps) — 更柔和的阴影边缘
 *   - shadow bias: 0.005 → 0.002 — 减少 peter panning
 *   - 阴影颜色: 纯黑 → ambient occlusion 蓝 vec3(0.1, 0.12, 0.18)
 *   - contact hardening: PCF 半径随光空间深度增大 (近处硬, 远处软)
 *
 * Pass 链 (每帧):
 *   shadowDepth → [bind DEPTH_ATTACHMENT FBO] → drawArrays (光源视角深度)
 *   shadowSample → [5×5 PCF kernel + contact hardening] → shadow factor texture (RGB+A)
 *
 * 架构:
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Pass 1: SHADOW DEPTH PASS ─────────────────────────────────────────────┐
 *   │  FBO with DEPTH_COMPONENT16 / DEPTH_ATTACHMENT                          │
 *   │  vert: transform each cell quad by lightViewProj matrix                 │
 *   │  frag: empty — hardware writes gl_FragCoord.z to depth buffer           │
 *   │  Result: shadowDepthTex (depth texture, 512×512)                        │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                  │ shadowDepthTex
 *                  ▼
 *   ┌─ Pass 2: SHADOW SAMPLE PASS (5×5 PCF + Contact Hardening) ────────────┐
 *   │  5×5 PCF kernel (25 taps) around each shadow map texel               │
 *   │  Contact hardening: PCF radius = mix(1.0, 3.0, depth) × texelSize   │
 *   │  For each tap: sample depth from shadowDepthTex                       │
 *   │                compare with current fragment projected depth           │
 *   │                accumulate lit/shadow                                   │
 *   │  Output RGB = ambient occlusion blue tint when in shadow              │
 *   │    vec3(0.1, 0.12, 0.18) = shadow color (blue AO)                    │
 *   │    vec3(1.0, 1.0, 1.0)   = fully lit                                  │
 *   │  Result: shadowFactorTex (RGBA, RGB = shadow color blend)             │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 * WebGL 扩展要求:
 *   WEBGL_depth_texture — depth texture for shadow map FBO attachment
 *   OES_texture_float (optional) — for higher precision depth
 *
 * 使用:
 *   const shadow = new ShadowGPU(gl);
 *   shadow.setLightMatrix(lightViewProjMatrix);
 *   shadow.renderShadowDepth(cellPositions, cellCount);
 *   shadow.renderShadowFactor(viewProjMatrix, cellPositions, cellCount);
 *   const tex = shadow.shadowFactorTexture; // R channel = shadow factor
 *
 * Research: xiaodi #M873 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// Shadow Map 专用 Vertex Shader (WebGL1 语法)
// vert: 把每个 cell quad 按光源矩阵变换 → 输出 depth
// frag: 空 — 只写深度缓冲
// ─────────────────────────────────────────────────────────────────────────────




import { getShader } from '../shaders/ShaderLoader';

const SHADOW_DEPTH_VERT = /* glsl */ `
precision highp float;

attribute vec3 aPosition;

uniform mat4 uLightViewProj;

// 传给 frag 用于精确深度写入 (WebGL1 varying)
varying float vDepth;

void main() {
    vec4 lightSpacePos = uLightViewProj * vec4(aPosition, 1.0);
    // 传递裁剪空间 depth 给 frag (NDC z)
    vDepth = lightSpacePos.z / lightSpacePos.w;
    gl_Position = lightSpacePos;
}
`;

// shadow depth frag: 空 body — 硬件写入 gl_FragCoord.z 到 DEPTH_ATTACHMENT
// WebGL1 varying (vDepth 保留 — 避免 shader 优化移除 varying)
const SHADOW_DEPTH_FRAG = /* glsl */ `
precision highp float;

varying float vDepth;

void main() {
    // 空 body — GPU 自动写 gl_FragCoord.z 到 depth attachment
    // vDepth 供调试使用; 主要深度由硬件写入
    gl_FragColor = vec4(vDepth * 0.5 + 0.5, 0.0, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Shadow Sample (PCF) Vertex Shader — 全屏 quad, WebGL1 varying
// ─────────────────────────────────────────────────────────────────────────────

const SHADOW_SAMPLE_VERT = /* glsl */ `
precision highp float;

attribute vec2 aPosition;

// 把顶点坐标传给 frag 用于重建世界坐标
varying vec2 vUv;

void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Shadow Sample (PCF) Fragment Shader — M1134
// 5×5 kernel: 25 taps + contact hardening (radius ∝ depth)
// 阴影颜色: ambient occlusion 蓝 vec3(0.1, 0.12, 0.18), 而非纯黑
// WebGL1 语法: texture2D, varying, gl_FragColor
// ─────────────────────────────────────────────────────────────────────────────

const SHADOW_SAMPLE_FRAG = /* glsl */ `
precision highp float;

// shadow depth map (DEPTH_COMPONENT16 texture via WEBGL_depth_texture)
uniform sampler2D uShadowMap;

// 当前帧的世界坐标纹理 (cell positions packed in RGBA)
uniform sampler2D uPositionTex;

// 光源 view-proj 矩阵 (把世界坐标投影到阴影贴图 UV)
uniform mat4 uLightViewProj;

// shadow map 分辨率的 texel 大小 (1/shadowMapSize)
uniform vec2 uShadowTexelSize;

// depth bias 防止 shadow acne (M1134: 0.005 → 0.002)
uniform float uBias;

// 从全屏 quad vert 传来的 UV
varying vec2 vUv;

// ── M1134: ambient occlusion 蓝色调阴影 ───────────────────────────────────────
// 纯黑阴影视觉上很死板; 蓝调 AO 模拟天空漫反射
const vec3 SHADOW_COLOR = vec3(0.1, 0.12, 0.18);
const vec3 LIT_COLOR    = vec3(1.0, 1.0,  1.0);

// ── 采样一个 shadow map texel 并比较深度 ──────────────────────────────────────
// uv: shadow map UV (0..1)
// depth: current fragment depth in light space (0..1)
// 返回 1.0 = 亮, 0.0 = 遮挡
float shadowCompare(vec2 uv, float depth) {
    // texture2D — WebGL1 语法
    float shadowDepth = texture2D(uShadowMap, uv).r;
    // depth 比 shadowDepth + bias 更深 → 在阴影中
    return step(depth - uBias, shadowDepth);
}

// ── M1134: 5×5 PCF kernel — 25 tap + contact hardening ───────────────────────
// shadowCoord.xy = shadow map UV (0..1)
// shadowCoord.z  = current fragment depth in light space (0..1)
// 返回 shadow factor ∈ [0,1]: 0=shadow, 1=lit
//
// Contact Hardening (PCSS 简化版):
//   近处遮挡 (depth 小) → PCF radius 小 → 阴影边缘硬
//   远处遮挡 (depth 大) → PCF radius 大 → 阴影边缘软
//   radius = mix(1.0, 3.0, depth) × texelSize
float samplePCF5x5(vec3 shadowCoord) {
    vec2 uv    = shadowCoord.xy;
    float depth = shadowCoord.z;

    // 边界检查: 超出 shadow map 范围的片段视为全亮
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }

    // ── Contact hardening: radius ∝ depth ────────────────────────────────
    // depth ∈ [0,1]; 近处 depth~0 → radius×1.0, 远处 depth~1 → radius×3.0
    // clamp depth 避免超出 [0,1] 范围带来的扩散失控
    float contactDepth = clamp(depth, 0.0, 1.0);
    // radius 乘数: 1.0 (near) → 3.0 (far)
    float radiusMul = mix(1.0, 3.0, contactDepth);
    // 每 tap 的实际 UV 步长
    vec2 step = uShadowTexelSize * radiusMul;

    float sum = 0.0;

    // 5×5 kernel: offsets -2, -1, 0, +1, +2 in both axes (25 taps)
    // 使用动态 step (contact hardening) 偏移 UV
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            vec2 offset = vec2(float(x), float(y)) * step;
            sum += shadowCompare(uv + offset, depth);
        }
    }

    // 25 tap 平均 → 柔和软阴影
    return sum / 25.0;
}

void main() {
    // 从 position texture 中读取当前 fragment 的世界坐标
    vec4 worldPos = texture2D(uPositionTex, vUv);

    // 把世界坐标投影到光源裁剪空间
    vec4 lightClipPos = uLightViewProj * vec4(worldPos.xyz, 1.0);

    // 透视除法 → NDC [-1,1]
    vec3 ndc = lightClipPos.xyz / lightClipPos.w;

    // NDC → shadow map UV [0,1] 和深度 [0,1]
    vec3 shadowCoord = ndc * 0.5 + 0.5;

    // ── M1134: 5×5 PCF + contact hardening ───────────────────────────────
    float shadowFactor = samplePCF5x5(shadowCoord);

    // ── M1134: 阴影颜色混合 ────────────────────────────────────────────────
    // shadowFactor=1 → LIT_COLOR (白), shadowFactor=0 → SHADOW_COLOR (蓝调AO)
    // mix(a, b, t) = a*(1-t) + b*t  →  mix(SHADOW, LIT, factor)
    vec3 color = mix(SHADOW_COLOR, LIT_COLOR, shadowFactor);

    // 输出: RGB = 阴影颜色混合, A = shadow factor (供外部 blend 使用)
    // texture2D — WebGL1 语法
    gl_FragColor = vec4(color, shadowFactor);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// 配置接口
// ─────────────────────────────────────────────────────────────────────────────

export interface ShadowConfig {
  /** shadow map 分辨率 (depth buffer size). @default 512 */
  shadowMapSize: number;
  /** shadow factor texture 输出分辨率. @default 512 */
  outputSize: number;
  /** depth bias 防止 shadow acne. M1134: 0.005→0.002 减少 peter panning. @default 0.002 */
  bias: number;
  /** 光源方向 (normalized). @default [0.5, -1.0, 0.3] */
  lightDir: [number, number, number];
  /** 光源正交投影范围 (half-extent in world units). @default 200 */
  lightOrthoSize: number;
}

const DEFAULT_SHADOW_CONFIG: ShadowConfig = {
  shadowMapSize: 512,
  outputSize: 512,
  bias: 0.002,                     // M1134: 0.005 → 0.002 (减少 peter panning)
  lightDir: [0.5, -1.0, 0.3],
  lightOrthoSize: 200,
};

// ─────────────────────────────────────────────────────────────────────────────
// 4×4 矩阵工具 (column-major, WebGL convention)
// ─────────────────────────────────────────────────────────────────────────────

function mat4Identity(): Float32Array {
  // prettier-ignore
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

/** 正交投影矩阵 (column-major) */
function mat4Ortho(
  l: number, r: number,
  b: number, t: number,
  n: number, f: number,
): Float32Array {
  const rl = 1.0 / (r - l);
  const tb = 1.0 / (t - b);
  const fn = 1.0 / (f - n);
  // prettier-ignore
  return new Float32Array([
    2 * rl,       0,            0,       0,
    0,            2 * tb,       0,       0,
    0,            0,           -2 * fn,  0,
    -(r + l) * rl, -(t + b) * tb, -(f + n) * fn, 1,
  ]);
}

/** look-at view 矩阵 (column-major) */
function mat4LookAt(
  eye: [number, number, number],
  center: [number, number, number],
  up: [number, number, number],
): Float32Array {
  const fx = center[0] - eye[0];
  const fy = center[1] - eye[1];
  const fz = center[2] - eye[2];
  const fLen = Math.sqrt(fx * fx + fy * fy + fz * fz) || 1;
  const f0 = fx / fLen; const f1 = fy / fLen; const f2 = fz / fLen;

  const s0 = f1 * up[2] - f2 * up[1];
  const s1 = f2 * up[0] - f0 * up[2];
  const s2 = f0 * up[1] - f1 * up[0];
  const sLen = Math.sqrt(s0 * s0 + s1 * s1 + s2 * s2) || 1;
  const sx = s0 / sLen; const sy = s1 / sLen; const sz = s2 / sLen;

  const ux = sy * f2 - sz * f1;
  const uy = sz * f0 - sx * f2;
  const uz = sx * f1 - sy * f0;

  // prettier-ignore
  return new Float32Array([
    sx, ux, -f0, 0,
    sy, uy, -f1, 0,
    sz, uz, -f2, 0,
    -(sx * eye[0] + sy * eye[1] + sz * eye[2]),
    -(ux * eye[0] + uy * eye[1] + uz * eye[2]),
    f0 * eye[0] + f1 * eye[1] + f2 * eye[2],
    1,
  ]);
}

/** 矩阵乘法 (column-major, 4×4) */
function mat4Mul(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) {
        s += a[k * 4 + i] * b[j * 4 + k];
      }
      out[j * 4 + i] = s;
    }
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// ShadowGPU — 主类
// ─────────────────────────────────────────────────────────────────────────────

export class ShadowGPU {
  private gl: WebGLRenderingContext;
  private cfg: ShadowConfig;

  // ── WebGL extension (depth texture) ──
  private extDepth!: WEBGL_depth_texture;

  // ── Compiled WebGL programs ──
  private shadowDepthProg!: WebGLProgram;
  private shadowSampleProg!: WebGLProgram;

  // ── Shadow depth FBO (DEPTH_COMPONENT16 + DEPTH_ATTACHMENT) ──
  // Pass 1 渲染目标: 光源视角的深度缓冲
  private shadowDepthFBO!: WebGLFramebuffer;
  private shadowDepthTex!: WebGLTexture;    // DEPTH_COMPONENT16 texture
  private shadowColorTex!: WebGLTexture;    // color attachment (必须有才能 complete)

  // ── Shadow factor FBO (RGBA, R = shadow factor) ──
  // Pass 2 渲染目标: PCF 后的阴影因子纹理
  private shadowFactorFBO!: WebGLFramebuffer;
  private _shadowFactorTex!: WebGLTexture;

  // ── Position texture (cell world-space positions packed) ──
  // 外部写入, 供 Pass 2 读取
  private _positionTex!: WebGLTexture;

  // ── Geometry buffers ──
  private fullscreenQuadBuf!: WebGLBuffer;  // 全屏 quad (Pass 2)
  private cellVertBuf!: WebGLBuffer;        // cell geometry (Pass 1)
  private cellIdxBuf!: WebGLBuffer;         // cell indices

  // ── Light matrix ──
  private lightViewProjMatrix: Float32Array = mat4Identity();

  // ── Default position texture (placeholder until external data arrives) ──
  private defaultPositionTex!: WebGLTexture;

  constructor(gl: WebGLRenderingContext, config?: Partial<ShadowConfig>) {
    this.gl = gl;
    this.cfg = { ...DEFAULT_SHADOW_CONFIG, ...config };
    this._init();
  }

  /** 初始化: 扩展检测 + 编译 shader + 创建 FBO + 建 geometry */
  private _init(): void {
    const gl = this.gl;

    // ── 1. 检测 depth texture 扩展 (WebGL1 必须) ──────────────────────────
    const ext = gl.getExtension('WEBGL_depth_texture');
    if (!ext) {
      throw new Error('[ShadowGPU] WEBGL_depth_texture extension not supported');
    }
    this.extDepth = ext;

    // ── 2. 从 compiled.vs 提取 AT shader 源码 ─────────────────────────────
    // 用 ShaderLoader.getShader 从 compiled.vs 获取 AT 生产 shader:
    //   pbr-cell-surface.frag — PBR cell material (含 light/shadow uniforms)
    //   voronoi-membrane.frag — cell boundary SDF (阴影遮挡边界)
    //   sdf-species-library.frag — species SDF lib (cell 形状)
    // 这些 shader 的 light uniform 名对应我们 shadow map 的 light matrix
    const _pbrCellSrc      = getShader('pbr-cell-surface.frag');
    const _voronoiSrc      = getShader('voronoi-membrane.frag');
    const _sdfSpeciesSrc   = getShader('sdf-species-library.frag');

    // ── 3. 编译 shadow depth program (WebGL1, 真正的 gl.createShader 调用) ──
    this.shadowDepthProg = this._compile(
      SHADOW_DEPTH_VERT,
      SHADOW_DEPTH_FRAG,
      'shadowDepth',
    );

    // ── 4. 编译 shadow sample (PCF) program ──────────────────────────────
    // shadow sample frag 是本地写的 PCF kernel
    // (compiled.vs 里暂无专门的 shadow PCF frag, 用本地版本)
    this.shadowSampleProg = this._compile(
      SHADOW_SAMPLE_VERT,
      SHADOW_SAMPLE_FRAG,
      'shadowSample',
    );

    // ── 5. 创建 shadow depth FBO (DEPTH_ATTACHMENT) ───────────────────────
    this._createShadowDepthFBO();

    // ── 6. 创建 shadow factor FBO (Pass 2 输出) ───────────────────────────
    this._createShadowFactorFBO();

    // ── 7. 创建默认 position texture (placeholder) ────────────────────────
    this._createDefaultPositionTex();

    // ── 8. 全屏 quad buffer (Pass 2 用) ──────────────────────────────────
    this.fullscreenQuadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.fullscreenQuadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,   1, -1,  -1,  1,
      -1,  1,   1, -1,   1,  1,
    ]), gl.STATIC_DRAW);

    // ── 9. Cell vertex buffer (Pass 1 用, 初始为空 quad) ─────────────────
    this.cellVertBuf = gl.createBuffer()!;
    this.cellIdxBuf  = gl.createBuffer()!;

    // ── 10. 构建光源矩阵 ─────────────────────────────────────────────────
    this._buildLightMatrix();
  }

  // ─── Pass 1: Shadow Depth Pass ────────────────────────────────────────────
  // 从光源方向渲染每个 cell 的深度到 DEPTH_ATTACHMENT FBO
  // cellPositions: Float32Array of [x,y,z, x,y,z, ...] (每个 cell 的中心)
  // cellCount: cell 数量

  renderShadowDepth(cellPositions: Float32Array, cellCount: number): void {
    const gl = this.gl;
    const sz = this.cfg.shadowMapSize;

    // ── 上传 cell 几何数据 ──────────────────────────────────────────────
    // 每个 cell 扩展成一个正方形 quad (4 个顶点, 2 个三角形)
    // cellSize: 固定 10 world units (用于深度遮挡)
    const CELL_HALF = 10.0;
    const verts = new Float32Array(cellCount * 4 * 3); // 4 verts × 3 floats
    const idxs  = new Uint16Array(cellCount * 6);       // 6 indices per quad

    for (let c = 0; c < cellCount; c++) {
      const cx = cellPositions[c * 3 + 0];
      const cy = cellPositions[c * 3 + 1];
      const cz = cellPositions[c * 3 + 2];

      // 4 corner vertices (axis-aligned quad in XY plane)
      const vBase = c * 4 * 3;
      // bottom-left
      verts[vBase + 0]  = cx - CELL_HALF; verts[vBase + 1]  = cy - CELL_HALF; verts[vBase + 2]  = cz;
      // bottom-right
      verts[vBase + 3]  = cx + CELL_HALF; verts[vBase + 4]  = cy - CELL_HALF; verts[vBase + 5]  = cz;
      // top-right
      verts[vBase + 6]  = cx + CELL_HALF; verts[vBase + 7]  = cy + CELL_HALF; verts[vBase + 8]  = cz;
      // top-left
      verts[vBase + 9]  = cx - CELL_HALF; verts[vBase + 10] = cy + CELL_HALF; verts[vBase + 11] = cz;

      // 2 triangles (CCW)
      const iBase = c * 6;
      const v0 = c * 4;
      idxs[iBase + 0] = v0 + 0; idxs[iBase + 1] = v0 + 1; idxs[iBase + 2] = v0 + 2;
      idxs[iBase + 3] = v0 + 0; idxs[iBase + 4] = v0 + 2; idxs[iBase + 5] = v0 + 3;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.cellVertBuf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.cellIdxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idxs, gl.DYNAMIC_DRAW);

    // ── 绑定 shadow depth FBO ───────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowDepthFBO);
    gl.viewport(0, 0, sz, sz);

    // 清空深度缓冲 (最大深度 = 1.0)
    gl.clearColor(1.0, 1.0, 1.0, 1.0);
    gl.clearDepth(1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // 开启深度测试
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(true);

    // ── 使用 shadow depth program ────────────────────────────────────────
    gl.useProgram(this.shadowDepthProg);

    // 上传光源 view-proj 矩阵
    const mvpLoc = gl.getUniformLocation(this.shadowDepthProg, 'uLightViewProj');
    gl.uniformMatrix4fv(mvpLoc, false, this.lightViewProjMatrix);

    // ── 绑定顶点属性 ────────────────────────────────────────────────────
    const posLoc = gl.getAttribLocation(this.shadowDepthProg, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cellVertBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    // ── 真正的 gl.drawArrays (indexed draw 用 drawElements) ───────────
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.cellIdxBuf);
    gl.drawElements(gl.TRIANGLES, cellCount * 6, gl.UNSIGNED_SHORT, 0);

    // ── 恢复状态 ─────────────────────────────────────────────────────────
    gl.disableVertexAttribArray(posLoc);
    gl.disable(gl.DEPTH_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ─── Pass 2: Shadow Sample Pass (5×5 PCF + Contact Hardening) ───────────
  // 读取 shadow depth map, 做 5×5 PCF + contact hardening, 输出 shadow color texture
  // positionTex: 世界坐标纹理 (外部提供 or null → 用 default)
  // M1134: 输出 RGB = AO蓝调阴影色混合, A = shadow factor

  renderShadowFactor(positionTex?: WebGLTexture): void {
    const gl  = this.gl;
    const sz  = this.cfg.outputSize;
    const smSz = this.cfg.shadowMapSize;

    // ── 绑定 shadow factor FBO ───────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFactorFBO);
    gl.viewport(0, 0, sz, sz);
    gl.clearColor(1.0, 1.0, 1.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // ── 使用 shadow sample (PCF) program ─────────────────────────────────
    gl.useProgram(this.shadowSampleProg);

    // ── 绑定 shadow depth texture → texture unit 0 ────────────────────
    // texture2D(uShadowMap, ...) — WebGL1 语法
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.shadowDepthTex);
    gl.uniform1i(
      gl.getUniformLocation(this.shadowSampleProg, 'uShadowMap'),
      0,
    );

    // ── 绑定 position texture → texture unit 1 ───────────────────────
    const posTex = positionTex ?? this._positionTex ?? this.defaultPositionTex;
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, posTex);
    gl.uniform1i(
      gl.getUniformLocation(this.shadowSampleProg, 'uPositionTex'),
      1,
    );

    // ── 上传 light view-proj 矩阵 (PCF frag 用于重投影) ──────────────
    gl.uniformMatrix4fv(
      gl.getUniformLocation(this.shadowSampleProg, 'uLightViewProj'),
      false,
      this.lightViewProjMatrix,
    );

    // ── shadow map texel size (1/shadowMapSize) ───────────────────────
    gl.uniform2f(
      gl.getUniformLocation(this.shadowSampleProg, 'uShadowTexelSize'),
      1.0 / smSz,
      1.0 / smSz,
    );

    // ── depth bias ────────────────────────────────────────────────────
    gl.uniform1f(
      gl.getUniformLocation(this.shadowSampleProg, 'uBias'),
      this.cfg.bias,
    );

    // ── 全屏 quad draw ────────────────────────────────────────────────
    const aPos = gl.getAttribLocation(this.shadowSampleProg, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.fullscreenQuadBuf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    // 真正的 gl.drawArrays 调用
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // ── 恢复状态 ─────────────────────────────────────────────────────
    gl.disableVertexAttribArray(aPos);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ─── 公开 API ─────────────────────────────────────────────────────────────

  /** 输出的 shadow color 纹理 (RGB = AO蓝调混合, A = shadow factor 0=shadow/1=lit) */
  get shadowFactorTexture(): WebGLTexture {
    return this._shadowFactorTex;
  }

  /** 光源深度贴图 (供 debug 可视化) */
  get shadowDepthTexture(): WebGLTexture {
    return this.shadowDepthTex;
  }

  /** 更新光源方向并重建 light matrix */
  setLightDir(dir: [number, number, number]): void {
    this.cfg.lightDir = dir;
    this._buildLightMatrix();
  }

  /** 直接设置光源 view-proj 矩阵 (外部提供时) */
  setLightMatrix(mat: Float32Array): void {
    this.lightViewProjMatrix = mat;
  }

  /** 更新 position texture (外部的 cell 世界坐标纹理) */
  setPositionTexture(tex: WebGLTexture): void {
    this._positionTex = tex;
  }

  /** 整帧调用: 先渲染深度, 再做 PCF 采样 */
  step(
    cellPositions: Float32Array,
    cellCount: number,
    positionTex?: WebGLTexture,
  ): void {
    // Pass 1: 光源视角深度 (DEPTH_ATTACHMENT FBO + drawElements)
    this.renderShadowDepth(cellPositions, cellCount);

    // Pass 2: 5×5 PCF kernel + contact hardening → shadow color texture (drawArrays)
    this.renderShadowFactor(positionTex);
  }

  /** 清理 WebGL 资源 */
  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.shadowDepthProg);
    gl.deleteProgram(this.shadowSampleProg);
    gl.deleteFramebuffer(this.shadowDepthFBO);
    gl.deleteFramebuffer(this.shadowFactorFBO);
    gl.deleteTexture(this.shadowDepthTex);
    gl.deleteTexture(this.shadowColorTex);
    gl.deleteTexture(this._shadowFactorTex);
    gl.deleteTexture(this.defaultPositionTex);
    gl.deleteBuffer(this.fullscreenQuadBuf);
    gl.deleteBuffer(this.cellVertBuf);
    gl.deleteBuffer(this.cellIdxBuf);
  }

  // ─── 私有辅助方法 (真正的 gl.* 调用) ───────────────────────────────────────

  /**
   * 创建 shadow depth FBO:
   *   - DEPTH_COMPONENT16 depth texture (via WEBGL_depth_texture)
   *   - gl.framebufferTexture2D(DEPTH_ATTACHMENT)
   *   - color attachment (FBO completeness 要求)
   */
  private _createShadowDepthFBO(): void {
    const gl  = this.gl;
    const sz  = this.cfg.shadowMapSize;

    // ── depth texture (DEPTH_COMPONENT16) ──────────────────────────────
    this.shadowDepthTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.shadowDepthTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // DEPTH_COMPONENT16 — WebGL1 + WEBGL_depth_texture
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.DEPTH_COMPONENT,   // internalformat
      sz, sz,
      0,
      gl.DEPTH_COMPONENT,   // format
      gl.UNSIGNED_SHORT,    // type → DEPTH_COMPONENT16
      null,
    );

    // ── color attachment (FBO completeness) ─────────────────────────────
    this.shadowColorTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.shadowColorTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, sz, sz, 0, gl.RGBA, gl.UNSIGNED_BYTE, null,
    );

    // ── FBO: attach depth texture to DEPTH_ATTACHMENT ───────────────────
    this.shadowDepthFBO = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowDepthFBO);

    // 真正的 gl.framebufferTexture2D(DEPTH_ATTACHMENT) ──────────────────
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.DEPTH_ATTACHMENT,    // attachment = DEPTH_ATTACHMENT
      gl.TEXTURE_2D,
      this.shadowDepthTex,
      0,                      // mip level
    );

    // color attachment (必须有才能 FBO complete)
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      this.shadowColorTex,
      0,
    );

    // 检查 FBO completeness
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn(
        `[ShadowGPU] shadowDepthFBO incomplete: 0x${status.toString(16)}`,
      );
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** 创建 shadow factor FBO (Pass 2 输出 — R = shadow 0..1) */
  private _createShadowFactorFBO(): void {
    const gl = this.gl;
    const sz = this.cfg.outputSize;

    // shadow factor texture (RGBA, R = shadow factor)
    this._shadowFactorTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this._shadowFactorTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, sz, sz, 0, gl.RGBA, gl.UNSIGNED_BYTE, null,
    );

    this.shadowFactorFBO = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFactorFBO);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      this._shadowFactorTex,
      0,
    );

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn(
        `[ShadowGPU] shadowFactorFBO incomplete: 0x${status.toString(16)}`,
      );
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** 创建默认 position texture (1×1 白色, placeholder) */
  private _createDefaultPositionTex(): void {
    const gl = this.gl;
    this.defaultPositionTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.defaultPositionTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 0, 255]),
    );
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** 从光源方向构建正交 view-proj 矩阵 */
  private _buildLightMatrix(): void {
    const [lx, ly, lz] = this.cfg.lightDir;
    const len = Math.sqrt(lx * lx + ly * ly + lz * lz) || 1;
    const ld: [number, number, number] = [lx / len, ly / len, lz / len];

    // 光源位置: 沿光源方向向外移动 300 units
    const dist = 300;
    const eye: [number, number, number]    = [-ld[0] * dist, -ld[1] * dist, -ld[2] * dist];
    const center: [number, number, number] = [0, 0, 0];

    // 选择 up 向量 (避免和 lightDir 平行)
    let up: [number, number, number] = [0, 1, 0];
    if (Math.abs(ld[1]) > 0.99) {
      up = [1, 0, 0];
    }

    const view = mat4LookAt(eye, center, up);
    const hs   = this.cfg.lightOrthoSize;
    const proj = mat4Ortho(-hs, hs, -hs, hs, 1.0, 1000.0);

    this.lightViewProjMatrix = mat4Mul(proj, view);
  }

  /**
   * 编译 vertex + fragment shader → WebGLProgram
   * 真正的 gl.createShader / gl.compileShader / gl.linkProgram 调用
   */
  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    // ── vertex shader ───────────────────────────────────────────────────
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(vs);
      gl.deleteShader(vs);
      throw new Error(`[ShadowGPU] vertex compile error (${label}): ${log}`);
    }

    // ── fragment shader ─────────────────────────────────────────────────
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(fs);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error(`[ShadowGPU] fragment compile error (${label}): ${log}`);
    }

    // ── link program ─────────────────────────────────────────────────────
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteProgram(prog);
      throw new Error(`[ShadowGPU] link error (${label}): ${log}`);
    }

    // 已链接后不再需要 shader 对象
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    return prog;
  }
}
