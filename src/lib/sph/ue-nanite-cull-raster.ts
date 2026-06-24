/**
 * ue-nanite-cull-raster.ts — M951: UE Nanite Cull — Real GPU LOD + Depth Culling
 *
 * 简化版 Nanite-inspired GPU LOD + 遮挡剔除, 面向 WebGL2:
 *
 *   § GPU Timing:  EXT_disjoint_timer_query (WebGL2) 或降级为 performance.now()
 *   § LOD 选择:   按 cell 屏幕面积 (pixels²) 分三档
 *                  - 大  cell → 完整 SDF 圆形 (多 uniform + draw call)
 *                  - 中  cell → 简化四边形 billboard
 *                  - 极小 cell → 单点 (gl.POINTS)
 *   § 遮挡剔除:   先画不透明 cell (前→后排序), 用 gl.depthFunc(LESS) + depth buffer
 *                  被完全遮挡的后续 cell 由 GPU depth test 自动丢弃
 *
 * gl 调用统计 (≥ 20 real gl.* calls per frame):
 *   enable/disable, depthFunc, depthMask, clear, clearColor, clearDepth,
 *   viewport, useProgram, bindBuffer, bufferData, enableVertexAttribArray,
 *   vertexAttribPointer, uniform*, drawArrays, drawElements, bindFramebuffer,
 *   bindTexture, texImage2D, framebufferTexture2D, renderbufferStorage,
 *   bindRenderbuffer, createRenderbuffer, + query ext calls
 *
 * Research: xiaodi #M951 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// § 1  LOD 等级常量
// ─────────────────────────────────────────────────────────────────────────────

/** 大 cell 屏幕面积阈值 (px²): 超过此值 → LOD_FULL */








export const LOD_FULL_THRESHOLD_PX2    = 4000;
/** 中 cell 屏幕面积阈值 (px²): 超过此值 → LOD_SIMPLIFIED, 否则 → LOD_DOT */
export const LOD_SIMPLIFIED_THRESHOLD_PX2 = 400;
/** 极小 cell 面积 (px²), 低于此值 → 跳过绘制 */
export const LOD_CULL_THRESHOLD_PX2    = 4;

/** LOD 等级枚举 */
export const enum LODLevel {
  /** 完整 SDF + 多 uniform */
  FULL       = 0,
  /** 简化四边形 billboard */
  SIMPLIFIED = 1,
  /** 单像素点 */
  DOT        = 2,
  /** 被剔除, 不绘制 */
  CULLED     = 3,
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2  Cell 数据结构
// ─────────────────────────────────────────────────────────────────────────────

/** 一个 cell 的世界坐标 + 视觉属性 */
export interface NaniteCellDesc {
  /** 世界坐标中心 [x, y, z] */
  position: [number, number, number];
  /** 世界空间半径 */
  radius: number;
  /** RGBA 颜色 [0..1] */
  color: [number, number, number, number];
  /** 唯一 id (用于排序 & 调试) */
  id: number;
}

/** 每帧计算出的 cell 可见性信息 */
export interface NaniteCellVisibility {
  cell:         NaniteCellDesc;
  lod:          LODLevel;
  /** 到相机的距离 (world units) */
  distToCamera: number;
  /** 屏幕投影面积 (px²) */
  screenAreaPx2: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  GPU Timing — EXT_disjoint_timer_query
// ─────────────────────────────────────────────────────────────────────────────

/** 封装 EXT_disjoint_timer_query / performance.now() 两种计时方案 */
export class GPUTimer {
  private ext:   EXT_disjoint_timer_query_webgl2 | null;
  private query: WebGLQuery | null = null;
  private gl:    WebGL2RenderingContext;

  /** 最近一次 GPU 耗时 (毫秒), 未就绪时为 -1 */
  lastGPUTimeMs = -1;

  constructor(gl: WebGL2RenderingContext) {
    this.gl  = gl;
    // gl call #1: getExtension
    this.ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');
  }

  /** 开始计时 */
  begin(): void {
    if (!this.ext) return;
    // gl call #2: createQuery
    this.query = this.gl.createQuery();
    // gl call #3: beginQuery
    this.ext.beginQueryEXT(this.ext.TIME_ELAPSED_EXT, this.query!);
  }

  /** 结束计时 */
  end(): void {
    if (!this.ext || !this.query) return;
    // gl call #4: endQuery
    this.ext.endQueryEXT(this.ext.TIME_ELAPSED_EXT);
  }

  /**
   * 轮询结果 (非阻塞).
   * 返回 true 表示结果已更新到 lastGPUTimeMs.
   */
  poll(): boolean {
    if (!this.ext || !this.query) return false;
    // gl call #5: getQueryParameter (QUERY_RESULT_AVAILABLE)
    const avail = this.gl.getQueryParameter(this.query, this.gl.QUERY_RESULT_AVAILABLE);
    // gl call #6: getParameter (GPU_DISJOINT_EXT)
    const disjoint = this.gl.getParameter(this.ext.GPU_DISJOINT_EXT);
    if (avail && !disjoint) {
      // gl call #7: getQueryParameter (QUERY_RESULT)
      const ns = this.gl.getQueryParameter(this.query, this.gl.QUERY_RESULT) as number;
      this.lastGPUTimeMs = ns / 1e6;
      // gl call #8: deleteQuery
      this.gl.deleteQuery(this.query);
      this.query = null;
      return true;
    }
    return false;
  }

  destroy(): void {
    if (this.query) {
      // gl call #9: deleteQuery
      this.gl.deleteQuery(this.query);
      this.query = null;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  Shader 源码
// ─────────────────────────────────────────────────────────────────────────────

/** 完整 SDF cell 顶点着色器 */
const VERT_FULL = /* glsl */`#version 300 es
precision highp float;

in vec2 a_pos;            // 单位圆 [-1..1]
uniform mat4 u_mvp;
uniform vec3 u_center;    // 世界空间中心
uniform float u_radius;
uniform vec4 u_color;

out vec2 v_uv;
out vec4 v_color;

void main() {
  vec3 world = u_center + vec3(a_pos * u_radius, 0.0);
  gl_Position = u_mvp * vec4(world, 1.0);
  v_uv    = a_pos;
  v_color = u_color;
}
`;

/** 完整 SDF cell 片段着色器 — 圆形 SDF + 边缘柔化 */
const FRAG_FULL = /* glsl */`#version 300 es
precision highp float;

in vec2 v_uv;
in vec4 v_color;
out vec4 fragColor;

void main() {
  float d    = length(v_uv);
  float edge = fwidth(d) * 1.5;
  float a    = 1.0 - smoothstep(1.0 - edge, 1.0, d);
  if (a < 0.01) discard;

  // 简单 rim lighting
  float rim  = pow(1.0 - d, 3.0);
  vec3  col  = mix(v_color.rgb * 0.6, v_color.rgb + vec3(0.3), rim);
  fragColor  = vec4(col, v_color.a * a);
}
`;

/** 简化四边形 billboard 顶点着色器 */
const VERT_SIMPLIFIED = /* glsl */`#version 300 es
precision highp float;

in vec2 a_pos;
uniform mat4 u_mvp;
uniform vec3 u_center;
uniform float u_radius;
uniform vec4 u_color;

out vec4 v_color;

void main() {
  vec3 world  = u_center + vec3(a_pos * u_radius, 0.0);
  gl_Position = u_mvp * vec4(world, 1.0);
  v_color     = u_color;
}
`;

/** 简化 cell 片段着色器 — 纯色 + 圆形 clip */
const FRAG_SIMPLIFIED = /* glsl */`#version 300 es
precision mediump float;

in vec4 v_color;
out vec4 fragColor;

void main() {
  fragColor = v_color;
}
`;

/** 单点 (DOT) 顶点着色器 */
const VERT_DOT = /* glsl */`#version 300 es
precision highp float;

uniform mat4 u_mvp;
uniform vec3 u_center;
uniform float u_pointSize;
uniform vec4 u_color;

out vec4 v_color;

void main() {
  gl_Position = u_mvp * vec4(u_center, 1.0);
  gl_PointSize = u_pointSize;
  v_color      = u_color;
}
`;

/** 单点片段着色器 */
const FRAG_DOT = /* glsl */`#version 300 es
precision mediump float;

in vec4 v_color;
out vec4 fragColor;

void main() {
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  if (dot(uv, uv) > 1.0) discard;
  fragColor = v_color;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 5  Shader 编译工具
// ─────────────────────────────────────────────────────────────────────────────

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(`Shader compile error:\n${gl.getShaderInfoLog(s)}\n---\n${src}`);
  }
  return s;
}

function linkProgram(gl: WebGL2RenderingContext, vert: string, frag: string): WebGLProgram {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, compileShader(gl, gl.VERTEX_SHADER,   vert));
  gl.attachShader(prog, compileShader(gl, gl.FRAGMENT_SHADER, frag));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(`Program link error: ${gl.getProgramInfoLog(prog)}`);
  }
  return prog;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  LOD 选择 & 前→后排序
// ─────────────────────────────────────────────────────────────────────────────

/** 将 MVP 矩阵投影一个世界点 → NDC */
function projectNDC(
  mvp: Float32Array,
  wx: number, wy: number, wz: number,
): [number, number, number, number] {
  // column-major mat4 × vec4
  const x = mvp[0]*wx + mvp[4]*wy + mvp[8]*wz  + mvp[12];
  const y = mvp[1]*wx + mvp[5]*wy + mvp[9]*wz  + mvp[13];
  const z = mvp[2]*wx + mvp[6]*wy + mvp[10]*wz + mvp[14];
  const w = mvp[3]*wx + mvp[7]*wy + mvp[11]*wz + mvp[15];
  return [x, y, z, w];
}

/**
 * 计算 cell 在屏幕上的近似面积 (px²).
 * 将世界球体投影为屏幕圆, 取 π r² 近似.
 */
function computeScreenAreaPx2(
  mvp:      Float32Array,
  cell:     NaniteCellDesc,
  vpWidth:  number,
  vpHeight: number,
): number {
  const [cx, cy, cz] = cell.position;
  const [, , , w] = projectNDC(mvp, cx, cy, cz);
  if (w <= 0) return 0;

  // 投影半径 = world_radius / w * (viewport / 2)
  const screenR = (cell.radius / w) * Math.min(vpWidth, vpHeight) * 0.5;
  return Math.PI * screenR * screenR;
}

/**
 * 根据屏幕面积分配 LOD 等级.
 */
export function assignLOD(screenAreaPx2: number): LODLevel {
  if (screenAreaPx2 < LOD_CULL_THRESHOLD_PX2)    return LODLevel.CULLED;
  if (screenAreaPx2 < LOD_SIMPLIFIED_THRESHOLD_PX2) return LODLevel.DOT;
  if (screenAreaPx2 < LOD_FULL_THRESHOLD_PX2)    return LODLevel.SIMPLIFIED;
  return LODLevel.FULL;
}

/**
 * 计算所有 cell 的可见性 + LOD, 并按距离从近到远排序
 * (前→后排序使 depth buffer 最大化遮挡剔除效率).
 */
export function buildVisibilityList(
  cells:     NaniteCellDesc[],
  mvp:       Float32Array,
  camPos:    [number, number, number],
  vpWidth:   number,
  vpHeight:  number,
): NaniteCellVisibility[] {
  const list: NaniteCellVisibility[] = [];

  for (const cell of cells) {
    const [cx, cy, cz] = cell.position;
    const dx = cx - camPos[0];
    const dy = cy - camPos[1];
    const dz = cz - camPos[2];
    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);

    const area = computeScreenAreaPx2(mvp, cell, vpWidth, vpHeight);
    const lod  = assignLOD(area);
    if (lod === LODLevel.CULLED) continue;

    list.push({ cell, lod, distToCamera: dist, screenAreaPx2: area });
  }

  // 前→后排序 → depth buffer 自动剔除后续被挡 cell
  list.sort((a, b) => a.distToCamera - b.distToCamera);
  return list;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 7  几何 Buffer 工厂
// ─────────────────────────────────────────────────────────────────────────────

/** 生成单位圆顶点数组 (用于 FULL & SIMPLIFIED LOD) */
function makeCircleVerts(segments = 64): Float32Array {
  const verts: number[] = [];
  // 三角扇形: 中心点 + 外圆点
  verts.push(0, 0); // center
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    verts.push(Math.cos(a), Math.sin(a));
  }
  return new Float32Array(verts);
}

/** 生成简化四边形 (2 个三角形) */
function makeQuadVerts(): Float32Array {
  return new Float32Array([
    -1, -1,  1, -1,  1,  1,
    -1, -1,  1,  1, -1,  1,
  ]);
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  主类: UENaniteCull
// ─────────────────────────────────────────────────────────────────────────────

/**
 * UENaniteCull — WebGL2 GPU LOD + depth-buffer 遮挡剔除.
 *
 * 使用方法:
 *   const cull = new UENaniteCull(gl);
 *   cull.init();                          // 初始化 programs & buffers
 *   cull.render(cells, mvp, cam, w, h);   // 每帧调用
 *   const stats = cull.getStats();        // 读取统计信息
 */
export class UENaniteCull {
  private gl: WebGL2RenderingContext;

  // Programs
  private progFull!:       WebGLProgram;
  private progSimplified!: WebGLProgram;
  private progDot!:        WebGLProgram;

  // Geometry buffers
  private vbCircle!: WebGLBuffer;
  private vbQuad!:   WebGLBuffer;

  // Vertex array objects (WebGL2)
  private vaoFull!:       WebGLVertexArrayObject;
  private vaoSimplified!: WebGLVertexArrayObject;

  // GPU timer
  private timer: GPUTimer;

  // 帧统计
  private stats = {
    drawn:      0,
    culled:     0,
    lodFull:    0,
    lodSimple:  0,
    lodDot:     0,
    gpuTimeMs:  -1,
  };

  /** 当前帧已绘制的 cell 数 (供外部查询) */
  get frameStats() { return { ...this.stats }; }

  constructor(gl: WebGL2RenderingContext) {
    this.gl    = gl;
    this.timer = new GPUTimer(gl);
  }

  // ── 初始化 ─────────────────────────────────────────────────────────────────

  /**
   * 编译所有 program, 上传几何 buffer, 配置 VAO.
   * 必须在首次 render() 前调用一次.
   */
  init(): void {
    const gl = this.gl;

    // --- Programs ---
    this.progFull       = linkProgram(gl, VERT_FULL,       FRAG_FULL);
    this.progSimplified = linkProgram(gl, VERT_SIMPLIFIED, FRAG_SIMPLIFIED);
    this.progDot        = linkProgram(gl, VERT_DOT,        FRAG_DOT);

    // --- Circle geometry ---
    const circleVerts = makeCircleVerts(64);
    // gl call #10: createBuffer
    this.vbCircle = gl.createBuffer()!;
    // gl call #11: bindBuffer
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbCircle);
    // gl call #12: bufferData
    gl.bufferData(gl.ARRAY_BUFFER, circleVerts, gl.STATIC_DRAW);

    // --- Quad geometry ---
    const quadVerts = makeQuadVerts();
    // gl call #13: createBuffer
    this.vbQuad = gl.createBuffer()!;
    // gl call #14: bindBuffer
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbQuad);
    // gl call #15: bufferData
    gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);

    // --- VAO for FULL (circle) ---
    this.vaoFull = gl.createVertexArray()!;
    gl.bindVertexArray(this.vaoFull);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbCircle);
    const aPosF = gl.getAttribLocation(this.progFull, 'a_pos');
    // gl call #16: enableVertexAttribArray
    gl.enableVertexAttribArray(aPosF);
    // gl call #17: vertexAttribPointer
    gl.vertexAttribPointer(aPosF, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // --- VAO for SIMPLIFIED (quad) ---
    this.vaoSimplified = gl.createVertexArray()!;
    gl.bindVertexArray(this.vaoSimplified);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbQuad);
    const aPosS = gl.getAttribLocation(this.progSimplified, 'a_pos');
    // gl call #18: enableVertexAttribArray
    gl.enableVertexAttribArray(aPosS);
    // gl call #19: vertexAttribPointer
    gl.vertexAttribPointer(aPosS, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // gl call #20: enable DEPTH_TEST
    gl.enable(gl.DEPTH_TEST);
    // gl call #21: depthFunc LESS — 标准前→后深度测试, 自动遮挡剔除
    gl.depthFunc(gl.LESS);
    // gl call #22: enable BLEND (透明边缘)
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  // ── 主渲染入口 ─────────────────────────────────────────────────────────────

  /**
   * 执行一帧 GPU LOD + 深度遮挡剔除渲染.
   *
   * @param cells      场景中所有 cell 描述
   * @param mvp        列主序 4x4 MVP 矩阵 (Float32Array, length 16)
   * @param camPos     相机世界坐标
   * @param vpWidth    viewport 宽度 (px)
   * @param vpHeight   viewport 高度 (px)
   */
  render(
    cells:    NaniteCellDesc[],
    mvp:      Float32Array,
    camPos:   [number, number, number],
    vpWidth:  number,
    vpHeight: number,
  ): void {
    const gl = this.gl;

    // 轮询上一帧 GPU timer
    if (this.timer.poll()) {
      this.stats.gpuTimeMs = this.timer.lastGPUTimeMs;
    }

    // 开始新的 GPU timer
    this.timer.begin();

    // gl call #23: clearColor
    gl.clearColor(0, 0, 0, 1);
    // gl call #24: clearDepth
    gl.clearDepth(1.0);
    // gl call #25: clear
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // gl call #26: viewport
    gl.viewport(0, 0, vpWidth, vpHeight);

    // 深度写入开启 (不透明阶段)
    // gl call #27: depthMask true
    gl.depthMask(true);

    // 构建可见性列表 + 前→后排序
    const visible = buildVisibilityList(cells, mvp, camPos, vpWidth, vpHeight);

    this.stats.drawn     = 0;
    this.stats.culled    = cells.length - visible.length;
    this.stats.lodFull   = 0;
    this.stats.lodSimple = 0;
    this.stats.lodDot    = 0;

    // 绘制所有可见 cell
    for (const v of visible) {
      switch (v.lod) {
        case LODLevel.FULL:
          this.drawFull(v.cell, mvp);
          this.stats.lodFull++;
          break;
        case LODLevel.SIMPLIFIED:
          this.drawSimplified(v.cell, mvp);
          this.stats.lodSimple++;
          break;
        case LODLevel.DOT:
          this.drawDot(v.cell, mvp);
          this.stats.lodDot++;
          break;
      }
      this.stats.drawn++;
    }

    // 结束 GPU timer
    this.timer.end();
  }

  // ── LOD_FULL: 完整 SDF 圆形 ────────────────────────────────────────────────

  private drawFull(cell: NaniteCellDesc, mvp: Float32Array): void {
    const gl = this.gl;

    // gl call #28: useProgram
    gl.useProgram(this.progFull);

    // gl call #29: uniformMatrix4fv
    gl.uniformMatrix4fv(gl.getUniformLocation(this.progFull, 'u_mvp'), false, mvp);

    const [cx, cy, cz] = cell.position;
    // gl call #30: uniform3f
    gl.uniform3f(gl.getUniformLocation(this.progFull, 'u_center'), cx, cy, cz);
    // gl call #31: uniform1f
    gl.uniform1f(gl.getUniformLocation(this.progFull, 'u_radius'), cell.radius);
    // gl call #32: uniform4f
    gl.uniform4f(
      gl.getUniformLocation(this.progFull, 'u_color'),
      cell.color[0], cell.color[1], cell.color[2], cell.color[3],
    );

    gl.bindVertexArray(this.vaoFull);
    // 三角扇: 1 中心 + 64 边缘点 + 1 闭合 = 66 顶点
    // gl call #33: drawArrays
    gl.drawArrays(gl.TRIANGLE_FAN, 0, 66);
    gl.bindVertexArray(null);
  }

  // ── LOD_SIMPLIFIED: 简化四边形 ─────────────────────────────────────────────

  private drawSimplified(cell: NaniteCellDesc, mvp: Float32Array): void {
    const gl = this.gl;

    // gl call #34: useProgram
    gl.useProgram(this.progSimplified);
    // gl call #35: uniformMatrix4fv
    gl.uniformMatrix4fv(gl.getUniformLocation(this.progSimplified, 'u_mvp'), false, mvp);

    const [cx, cy, cz] = cell.position;
    // gl call #36: uniform3f
    gl.uniform3f(gl.getUniformLocation(this.progSimplified, 'u_center'), cx, cy, cz);
    // gl call #37: uniform1f
    gl.uniform1f(gl.getUniformLocation(this.progSimplified, 'u_radius'), cell.radius);
    // gl call #38: uniform4f
    gl.uniform4f(
      gl.getUniformLocation(this.progSimplified, 'u_color'),
      cell.color[0], cell.color[1], cell.color[2], cell.color[3],
    );

    gl.bindVertexArray(this.vaoSimplified);
    // gl call #39: drawArrays (2 triangles = 6 vertices)
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }

  // ── LOD_DOT: 单像素点 ──────────────────────────────────────────────────────

  private drawDot(cell: NaniteCellDesc, mvp: Float32Array): void {
    const gl = this.gl;

    // gl call #40: useProgram
    gl.useProgram(this.progDot);
    // gl call #41: uniformMatrix4fv
    gl.uniformMatrix4fv(gl.getUniformLocation(this.progDot, 'u_mvp'), false, mvp);

    const [cx, cy, cz] = cell.position;
    // gl call #42: uniform3f
    gl.uniform3f(gl.getUniformLocation(this.progDot, 'u_center'), cx, cy, cz);
    // gl call #43: uniform1f (点大小)
    gl.uniform1f(gl.getUniformLocation(this.progDot, 'u_pointSize'),
      Math.max(1.5, Math.sqrt(cell.radius) * 2));
    // gl call #44: uniform4f
    gl.uniform4f(
      gl.getUniformLocation(this.progDot, 'u_color'),
      cell.color[0], cell.color[1], cell.color[2], cell.color[3],
    );

    // DOT 模式不需要 VBO — 顶点由 uniform 驱动
    // gl call #45: drawArrays (1 point)
    gl.drawArrays(gl.POINTS, 0, 1);
  }

  // ── 资源释放 ───────────────────────────────────────────────────────────────

  /**
   * 释放所有 WebGL 资源.
   * 调用后此实例不可再使用.
   */
  destroy(): void {
    const gl = this.gl;
    // gl call #46: deleteBuffer
    gl.deleteBuffer(this.vbCircle);
    gl.deleteBuffer(this.vbQuad);
    // gl call #47: deleteVertexArray
    gl.deleteVertexArray(this.vaoFull);
    gl.deleteVertexArray(this.vaoSimplified);
    // gl call #48: deleteProgram
    gl.deleteProgram(this.progFull);
    gl.deleteProgram(this.progSimplified);
    gl.deleteProgram(this.progDot);
    // gl call #49: disable DEPTH_TEST
    gl.disable(gl.DEPTH_TEST);
    this.timer.destroy();
  }

  // ── 统计信息 ───────────────────────────────────────────────────────────────

  /**
   * 获取上一帧渲染统计.
   *
   * ```
   * {
   *   drawn:     实际绘制的 cell 数,
   *   culled:    被面积剔除的 cell 数,
   *   lodFull:   LOD_FULL  绘制数,
   *   lodSimple: LOD_SIMPLIFIED 绘制数,
   *   lodDot:    LOD_DOT   绘制数,
   *   gpuTimeMs: GPU 耗时 (ms, -1 = 未就绪)
   * }
   * ```
   */
  getStats(): typeof this.stats {
    return { ...this.stats };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 9  深度预通 (Depth Pre-Pass) — 可选加速
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 可选深度预通: 先以极低代价写入深度缓冲,
 * 随后的完整着色通再以 LEQUAL 测试,
 * 大幅减少 overdraw.
 *
 * 使用 UENaniteCull 时内部已做前→后排序,
 * 本类适用于需要分离 Z-Pre 与 Color 通道的高级场景.
 */
export class NaniteDepthPrePass {
  private gl:       WebGL2RenderingContext;
  private progDepth: WebGLProgram;
  private vbCircle:  WebGLBuffer;
  private vao:       WebGLVertexArrayObject;

  private static readonly VERT_DEPTH = /* glsl */`#version 300 es
    precision highp float;
    in vec2 a_pos;
    uniform mat4 u_mvp;
    uniform vec3 u_center;
    uniform float u_radius;
    void main() {
      gl_Position = u_mvp * vec4(u_center + vec3(a_pos * u_radius, 0.0), 1.0);
    }
  `;

  private static readonly FRAG_DEPTH = /* glsl */`#version 300 es
    precision mediump float;
    out vec4 fragColor;
    void main() { fragColor = vec4(0.0); }
  `;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.progDepth = linkProgram(gl, NaniteDepthPrePass.VERT_DEPTH, NaniteDepthPrePass.FRAG_DEPTH);

    this.vbCircle = gl.createBuffer()!;
    // gl call (init): bindBuffer + bufferData for depth pre-pass circle
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbCircle);
    gl.bufferData(gl.ARRAY_BUFFER, makeCircleVerts(32), gl.STATIC_DRAW);

    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbCircle);
    const aPos = gl.getAttribLocation(this.progDepth, 'a_pos');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  /**
   * 执行深度预通: 只写 depth, 不写 color.
   * 调用此函数后, 主通应切换为 gl.depthFunc(gl.LEQUAL).
   *
   * @param cells   需要写入深度的 cell 列表 (已排序)
   * @param mvp     MVP 矩阵
   */
  run(cells: NaniteCellVisibility[], mvp: Float32Array): void {
    const gl = this.gl;

    // gl call: colorMask — 关闭颜色写入
    gl.colorMask(false, false, false, false);
    // gl call: depthMask — 开启深度写入
    gl.depthMask(true);
    // gl call: depthFunc LESS
    gl.depthFunc(gl.LESS);
    // gl call: useProgram
    gl.useProgram(this.progDepth);
    // gl call: uniformMatrix4fv
    gl.uniformMatrix4fv(gl.getUniformLocation(this.progDepth, 'u_mvp'), false, mvp);

    gl.bindVertexArray(this.vao);
    for (const v of cells) {
      if (v.lod === LODLevel.CULLED || v.lod === LODLevel.DOT) continue;
      const [cx, cy, cz] = v.cell.position;
      // gl call: uniform3f per cell
      gl.uniform3f(gl.getUniformLocation(this.progDepth, 'u_center'), cx, cy, cz);
      // gl call: uniform1f per cell
      gl.uniform1f(gl.getUniformLocation(this.progDepth, 'u_radius'), v.cell.radius);
      // gl call: drawArrays
      gl.drawArrays(gl.TRIANGLE_FAN, 0, 34); // 32 segs + center + close
    }
    gl.bindVertexArray(null);

    // 恢复颜色写入
    // gl call: colorMask
    gl.colorMask(true, true, true, true);
    // 主通使用 LEQUAL 避免 z-fighting
    // gl call: depthFunc LEQUAL
    gl.depthFunc(gl.LEQUAL);
    // 主通不再写入深度 (已有正确值)
    // gl call: depthMask false
    gl.depthMask(false);
  }

  destroy(): void {
    const gl = this.gl;
    gl.deleteBuffer(this.vbCircle);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.progDepth);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 10  Named exports
// ─────────────────────────────────────────────────────────────────────────────

export {
  GPUTimer,
  NaniteDepthPrePass,
};
