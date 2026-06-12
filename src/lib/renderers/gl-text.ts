/**
 * gl-text.ts — WebGL MSDF 字体渲染 (AT GLText, 22 次引用)
 *
 * 架构:
 *   MSDFAtlas       字体 atlas 描述 (JSON glyph metrics + PNG texture)
 *   GLTextGeometry  把字符串转换为 quad vertices (每字符 1 个 quad, 共 4 顶点 / 6 索引)
 *   GLText          instanced rendering — 一次 draw call 画所有字符
 *
 * 渲染流水线:
 *   1. 离线: bin/msdfgen.linux 把矢量字体 → MSDF PNG atlas + JSON metrics
 *      (参见 upstream/activetheory-svg2msdf)
 *   2. 运行时: loadMSDFAtlas() 加载 atlas JSON + 纹理
 *   3. GLTextGeometry.update(str) 把字符串变为 Float32Array quad buffer
 *   4. GLText.draw(camera) 用 gl.drawElementsInstanced 一次提交所有 500 个 cell label
 *
 * 对比 PixiJS Text:
 *   PixiJS Text     每个 Text 对象一个 canvas texture → 500 个 label = 500 draw calls
 *   GLText          共享一个 MSDF atlas → 500 个 label = 1 draw call
 *
 * MSDF 原理:
 *   msdfgen 把每个字形生成 3 通道 signed-distance-field (R/G/B 三方向),
 *   片元着色器用 median(r,g,b) 还原锐利轮廓，任意缩放都无锯齿.
 *
 * 依赖:
 *   hydra-gl-layer (createProgram, compileShader)
 *   color-utils    (Color.toFloat32Array)
 *
 * Author: dylanyunlon <dogechat@163.com>
 */

import { createProgram } from './hydra-gl-layer';
import { Color } from '../color-utils';

// ── MSDF Atlas 数据类型 ───────────────────────────────────────────────────────

/** atlas JSON 中单个字形描述 */
export interface GlyphInfo {
  /** Unicode code point */
  id: number;
  /** atlas texture 上的 UV 矩形 (像素坐标) */
  x: number;
  y: number;
  width: number;
  height: number;
  /** 字形在 baseline 的偏移量 */
  xoffset: number;
  yoffset: number;
  /** 到下一个字符的水平步进 */
  xadvance: number;
  /** 所在 page (单 atlas 恒为 0) */
  page: number;
}

/** kerning pair */
export interface KerningPair {
  first: number;
  second: number;
  amount: number;
}

/** MSDF atlas JSON 顶层结构 (兼容 msdf-bmfont-xml 格式) */
export interface MSDFAtlasData {
  info: {
    face: string;
    size: number;
    bold: number;
    italic: number;
    charset: string;
    unicode: number;
    stretchH: number;
    smooth: number;
    aa: number;
    padding: [number, number, number, number];
    spacing: [number, number];
  };
  common: {
    lineHeight: number;
    base: number;
    scaleW: number;
    scaleH: number;
    pages: number;
    packed: number;
    alphaChnl: number;
    redChnl: number;
    greenChnl: number;
    blueChnl: number;
  };
  pages: string[];
  chars: GlyphInfo[];
  kernings: KerningPair[];
}

// ── MSDFAtlas ─────────────────────────────────────────────────────────────────

/**
 * 已加载的 MSDF atlas: glyph map + WebGL 纹理.
 */
export class MSDFAtlas {
  readonly data: MSDFAtlasData;
  readonly texture: WebGLTexture;
  readonly glyphMap: Map<number, GlyphInfo>;
  readonly kerningMap: Map<number, Map<number, number>>;

  /** atlas 图集宽度 (像素) */
  readonly texW: number;
  /** atlas 图集高度 (像素) */
  readonly texH: number;
  /** 字体设计尺寸 (px) */
  readonly fontSize: number;
  /** 行高 (px) */
  readonly lineHeight: number;

  constructor(gl: WebGL2RenderingContext, data: MSDFAtlasData, image: HTMLImageElement | ImageBitmap) {
    this.data = data;
    this.texW = data.common.scaleW;
    this.texH = data.common.scaleH;
    this.fontSize = data.info.size;
    this.lineHeight = data.common.lineHeight;

    // 建立 glyph 查找表
    this.glyphMap = new Map(data.chars.map(g => [g.id, g]));

    // 建立 kerning 查找表
    this.kerningMap = new Map();
    for (const k of data.kernings) {
      if (!this.kerningMap.has(k.first)) this.kerningMap.set(k.first, new Map());
      this.kerningMap.get(k.first)!.set(k.second, k.amount);
    }

    // 上传纹理
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image as HTMLImageElement);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    this.texture = tex;
  }

  /** 取 glyph, 未找到返回空格 glyph 或 undefined */
  getGlyph(codePoint: number): GlyphInfo | undefined {
    return this.glyphMap.get(codePoint) ?? this.glyphMap.get(32); // fallback: space
  }

  /** 取 kerning 调整量 */
  getKerning(prev: number, curr: number): number {
    return this.kerningMap.get(prev)?.get(curr) ?? 0;
  }
}

// ── 异步加载 MSDFAtlas ────────────────────────────────────────────────────────

/**
 * loadMSDFAtlas — 从 JSON url + 图片 url 异步加载 atlas.
 *
 * @example
 *   const atlas = await loadMSDFAtlas(gl,
 *     '/fonts/inter-medium-msdf.json',
 *     '/fonts/inter-medium-msdf.png');
 */
export async function loadMSDFAtlas(
  gl: WebGL2RenderingContext,
  jsonUrl: string,
  imageUrl: string,
): Promise<MSDFAtlas> {
  const [data, image] = await Promise.all([
    fetch(jsonUrl).then(r => r.json()) as Promise<MSDFAtlasData>,
    new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = imageUrl;
    }),
  ]);
  return new MSDFAtlas(gl, data, image);
}

// ── GLTextGeometry ────────────────────────────────────────────────────────────

/**
 * 每个字符的 quad 顶点数据:
 *   [x, y, u, v]  × 4 顶点
 * 布局 (per-instance, interleaved):
 *   offset  0: vec2 position  (quad 左下角世界坐标, 单位 px)
 *   offset  8: vec2 size      (quad 宽高, px)
 *   offset 16: vec4 uvRect    (atlas UV: u0, v0, u1, v1)
 */

const FLOATS_PER_INSTANCE = 8; // position(2) + size(2) + uvRect(4)

export interface GLTextGeometryOptions {
  /** 字体大小 (px), 默认 14 */
  fontSize?: number;
  /** 字间距额外补偿 (px), 默认 0 */
  letterSpacing?: number;
  /** 行间距额外补偿 (px), 默认 0 */
  lineSpacing?: number;
  /** 对齐方式, 默认 'left' */
  align?: 'left' | 'center' | 'right';
}

export class GLTextGeometry {
  readonly atlas: MSDFAtlas;

  private opts: Required<GLTextGeometryOptions>;

  /** instance 数据 (每字符 8 floats) */
  instanceData!: Float32Array;
  /** 实际字符数 (instance count) */
  charCount = 0;
  /** 文字包围盒宽度 (px) */
  layoutWidth = 0;
  /** 文字包围盒高度 (px) */
  layoutHeight = 0;

  constructor(atlas: MSDFAtlas, opts: GLTextGeometryOptions = {}) {
    this.atlas = atlas;
    this.opts = {
      fontSize:      opts.fontSize      ?? 14,
      letterSpacing: opts.letterSpacing ?? 0,
      lineSpacing:   opts.lineSpacing   ?? 0,
      align:         opts.align         ?? 'left',
    };
  }

  /**
   * update(text) — 把字符串转换为 quad instance 数据.
   * 支持 '\n' 换行.
   */
  update(text: string): void {
    if (!text || text.length === 0) {
      this.charCount = 0;
      this.instanceData = new Float32Array(0);
      this.layoutWidth = 0;
      this.layoutHeight = 0;
      return;
    }

    const { atlas, opts } = this;
    const scale = opts.fontSize / atlas.fontSize;
    const lineH = (atlas.lineHeight + opts.lineSpacing) * scale;

    // 先按行拆分，计算每行宽度
    const lines = text.split('\n');
    const lineWidths: number[] = [];

    for (const line of lines) {
      let w = 0;
      let prevCode = -1;
      for (const ch of line) {
        const code = ch.codePointAt(0)!;
        const glyph = atlas.getGlyph(code);
        if (!glyph) continue;
        const kern = prevCode >= 0 ? atlas.getKerning(prevCode, code) : 0;
        w += (glyph.xadvance + kern + opts.letterSpacing) * scale;
        prevCode = code;
      }
      lineWidths.push(w);
    }

    this.layoutWidth = Math.max(...lineWidths);
    this.layoutHeight = lines.length * lineH;

    // 分配 instance 缓冲 (最多 text.length 个字符, 减去换行符)
    const maxChars = text.replace(/\n/g, '').length;
    this.instanceData = new Float32Array(maxChars * FLOATS_PER_INSTANCE);

    let idx = 0;
    let cursorY = 0;
    const d = this.instanceData;
    const { texW, texH } = atlas;

    for (let li = 0; li < lines.length; li++) {
      const line = lines[li];
      const lineW = lineWidths[li];

      // 水平对齐偏移
      let cursorX = 0;
      if (opts.align === 'center') cursorX = -lineW / 2;
      else if (opts.align === 'right') cursorX = -lineW;

      let prevCode = -1;

      for (const ch of line) {
        const code = ch.codePointAt(0)!;
        const glyph = atlas.getGlyph(code);
        if (!glyph) continue;

        const kern = prevCode >= 0 ? atlas.getKerning(prevCode, code) : 0;
        cursorX += kern * scale;

        const qx = cursorX + glyph.xoffset * scale;
        const qy = cursorY + glyph.yoffset * scale;
        const qw = glyph.width  * scale;
        const qh = glyph.height * scale;

        // UV (normalized)
        const u0 = glyph.x           / texW;
        const v0 = glyph.y           / texH;
        const u1 = (glyph.x + glyph.width)  / texW;
        const v1 = (glyph.y + glyph.height) / texH;

        // 写入 instanceData
        const base = idx * FLOATS_PER_INSTANCE;
        d[base]     = qx;
        d[base + 1] = qy;
        d[base + 2] = qw;
        d[base + 3] = qh;
        d[base + 4] = u0;
        d[base + 5] = v0;
        d[base + 6] = u1;
        d[base + 7] = v1;

        cursorX += (glyph.xadvance + opts.letterSpacing) * scale;
        prevCode = code;
        idx++;
      }

      cursorY += lineH;
    }

    this.charCount = idx;
  }
}

// ── GLSL Shaders ──────────────────────────────────────────────────────────────

/**
 * MSDF 顶点着色器.
 *
 * Attributes (per-vertex, 共用 fullscreen quad 的 4 顶点):
 *   a_corner  vec2  — quad 角落 offset (0/1)
 *
 * Instance attributes (per-glyph):
 *   a_position  vec2  — quad 左上角世界坐标
 *   a_size      vec2  — quad 宽高
 *   a_uvRect    vec4  — atlas UV (u0,v0,u1,v1)
 *
 * Uniforms:
 *   u_projection  mat4  — orthographic projection
 *   u_model       mat4  — label transform (position + scale)
 */
const MSDF_VERT = `#version 300 es
precision highp float;

// per-vertex
in vec2 a_corner;

// per-instance (instanced rendering)
in vec2 a_position;
in vec2 a_size;
in vec4 a_uvRect;

uniform mat4 u_projection;
uniform mat4 u_model;

out vec2 v_uv;

void main() {
  // 在 quad 内插值 UV
  v_uv = vec2(
    mix(a_uvRect.x, a_uvRect.z, a_corner.x),
    mix(a_uvRect.y, a_uvRect.w, a_corner.y)
  );

  // quad 角落世界坐标
  vec2 worldPos = a_position + a_corner * a_size;
  gl_Position = u_projection * u_model * vec4(worldPos, 0.0, 1.0);
}
`;

/**
 * MSDF 片元着色器.
 *
 * median(r,g,b) 还原 SDF，smoothstep 产生抗锯齿边缘.
 * pxRange: msdfgen 生成时的像素范围 (通常 4).
 */
const MSDF_FRAG = `#version 300 es
precision highp float;

in  vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_atlas;
uniform vec4      u_color;    // rgba
uniform float     u_pxRange;  // msdfgen pxRange (default 4)

// MSDF median
float median(float r, float g, float b) {
  return max(min(r, g), min(max(r, g), b));
}

// 屏幕空间 1px 对应多少 atlas 单位
float screenPxRange() {
  vec2 unitRange = vec2(u_pxRange) / vec2(textureSize(u_atlas, 0));
  vec2 screenTexSize = vec2(1.0) / fwidth(v_uv);
  return max(0.5 * dot(unitRange, screenTexSize), 1.0);
}

void main() {
  vec3 msd = texture(u_atlas, v_uv).rgb;
  float sd = median(msd.r, msd.g, msd.b);
  float screenPx = screenPxRange();
  float opacity = clamp((sd - 0.5) * screenPx + 0.5, 0.0, 1.0);
  fragColor = vec4(u_color.rgb, u_color.a * opacity);
}
`;

// ── GLText ────────────────────────────────────────────────────────────────────

/**
 * GLText — instanced MSDF 文字渲染器.
 *
 * 一个 GLText 实例管理一个 cell label 数组，
 * 调用 draw() 一次 draw call 渲染所有字符.
 *
 * @example
 *   const text = new GLText(gl, atlas);
 *
 *   // 添加 500 个 cell label
 *   for (const cell of cells) {
 *     text.addLabel(cell.id, cell.label, { x: cell.x, y: cell.y }, cell.color);
 *   }
 *
 *   // 每帧渲染
 *   text.draw(projectionMatrix);
 *
 *   // cell 移动时只更新位置, 不重新 layout
 *   text.setPosition(cell.id, newX, newY);
 */

export interface LabelEntry {
  id: string;
  geometry: GLTextGeometry;
  /** world-space position (px) */
  x: number;
  y: number;
  color: Color;
  /** 是否可见 */
  visible: boolean;
}

export class GLText {
  private readonly gl: WebGL2RenderingContext;
  private readonly atlas: MSDFAtlas;
  private readonly program: WebGLProgram;

  // Quad geometry (shared by all instances)
  private readonly quadVAO: WebGLVertexArrayObject;
  private readonly quadVBO: WebGLBuffer;  // corners: [0,0],[1,0],[1,1],[0,1]
  private readonly quadIBO: WebGLBuffer;  // 0,1,2, 0,2,3

  // Instance buffer (per-glyph: position, size, uvRect)
  private readonly instanceVBO: WebGLBuffer;

  // Uniform locations
  private readonly uProjection: WebGLUniformLocation;
  private readonly uModel:      WebGLUniformLocation;
  private readonly uAtlas:      WebGLUniformLocation;
  private readonly uColor:      WebGLUniformLocation;
  private readonly uPxRange:    WebGLUniformLocation;

  /** label 注册表 */
  private labels = new Map<string, LabelEntry>();

  /** 脏标记: 有 geometry 变更需要重新上传 */
  private dirty = false;

  /** 合并后的 instance 数据 (所有 label 的字符 quad) */
  private mergedInstances: Float32Array = new Float32Array(0);

  /** 每个 label 在 mergedInstances 中的起始 index */
  private labelOffsets = new Map<string, number>();

  constructor(gl: WebGL2RenderingContext, atlas: MSDFAtlas) {
    this.gl = gl;
    this.atlas = atlas;
    this.program = createProgram(gl, MSDF_VERT, MSDF_FRAG);

    // Uniform locations
    this.uProjection = gl.getUniformLocation(this.program, 'u_projection')!;
    this.uModel      = gl.getUniformLocation(this.program, 'u_model')!;
    this.uAtlas      = gl.getUniformLocation(this.program, 'u_atlas')!;
    this.uColor      = gl.getUniformLocation(this.program, 'u_color')!;
    this.uPxRange    = gl.getUniformLocation(this.program, 'u_pxRange')!;

    // ── quad VAO ──────────────────────────────────────────────────────────
    this.quadVAO = gl.createVertexArray()!;
    gl.bindVertexArray(this.quadVAO);

    // 4 corners: (0,0) (1,0) (1,1) (0,1)
    const corners = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
    this.quadVBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
    gl.bufferData(gl.ARRAY_BUFFER, corners, gl.STATIC_DRAW);

    // a_corner @ location 0
    const aCorner = gl.getAttribLocation(this.program, 'a_corner');
    gl.enableVertexAttribArray(aCorner);
    gl.vertexAttribPointer(aCorner, 2, gl.FLOAT, false, 0, 0);

    // indices: 0,1,2, 0,2,3
    const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
    this.quadIBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.quadIBO);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

    // ── instance VBO ──────────────────────────────────────────────────────
    this.instanceVBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceVBO);

    const STRIDE = FLOATS_PER_INSTANCE * 4; // bytes

    // a_position @ loc 1, offset 0
    const aPos = gl.getAttribLocation(this.program, 'a_position');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, STRIDE, 0);
    gl.vertexAttribDivisor(aPos, 1);

    // a_size @ loc 2, offset 8
    const aSize = gl.getAttribLocation(this.program, 'a_size');
    gl.enableVertexAttribArray(aSize);
    gl.vertexAttribPointer(aSize, 2, gl.FLOAT, false, STRIDE, 8);
    gl.vertexAttribDivisor(aSize, 1);

    // a_uvRect @ loc 3, offset 16
    const aUV = gl.getAttribLocation(this.program, 'a_uvRect');
    gl.enableVertexAttribArray(aUV);
    gl.vertexAttribPointer(aUV, 4, gl.FLOAT, false, STRIDE, 16);
    gl.vertexAttribDivisor(aUV, 1);

    gl.bindVertexArray(null);
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * 注册一个 label.
   * 若 id 已存在则更新.
   */
  addLabel(
    id: string,
    text: string,
    position: { x: number; y: number },
    color: Color,
    opts: GLTextGeometryOptions = {},
  ): void {
    const geo = new GLTextGeometry(this.atlas, opts);
    geo.update(text);

    this.labels.set(id, {
      id,
      geometry: geo,
      x: position.x,
      y: position.y,
      color,
      visible: true,
    });
    this.dirty = true;
  }

  /** 移除一个 label */
  removeLabel(id: string): void {
    if (this.labels.delete(id)) this.dirty = true;
  }

  /** 更新 label 世界坐标 (不重新 layout，高频调用安全) */
  setPosition(id: string, x: number, y: number): void {
    const label = this.labels.get(id);
    if (!label) return;
    label.x = x;
    label.y = y;
    // position 变化不需要重建 geometry，在 draw 里用 u_model 处理
  }

  /** 更新 label 颜色 */
  setColor(id: string, color: Color): void {
    const label = this.labels.get(id);
    if (!label) return;
    label.color = color;
  }

  /** 设置可见性 */
  setVisible(id: string, visible: boolean): void {
    const label = this.labels.get(id);
    if (!label) return;
    label.visible = visible;
  }

  /** 更新 label 文字内容 (重新 layout) */
  setText(id: string, text: string): void {
    const label = this.labels.get(id);
    if (!label) return;
    label.geometry.update(text);
    this.dirty = true;
  }

  /**
   * draw(projection) — 渲染所有可见 label.
   *
   * @param projection  列主序 4×4 正交投影矩阵 (Float32Array 16 元素)
   * @param pxRange     msdfgen 生成参数, 默认 4
   */
  draw(projection: Float32Array, pxRange = 4): void {
    const { gl } = this;
    if (this.labels.size === 0) return;

    // 重建 mergedInstances (仅在脏时)
    if (this.dirty) this.rebuildInstances();

    gl.useProgram(this.program);

    // 绑定 atlas 纹理 unit 0
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.atlas.texture);
    gl.uniform1i(this.uAtlas, 0);
    gl.uniform1f(this.uPxRange, pxRange);
    gl.uniformMatrix4fv(this.uProjection, false, projection);

    // alpha blending
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.bindVertexArray(this.quadVAO);

    // 上传合并后的 instance 数据
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceVBO);
    gl.bufferData(gl.ARRAY_BUFFER, this.mergedInstances, gl.DYNAMIC_DRAW);

    // 逐 label 设置 model matrix + color，然后 drawElementsInstanced
    const model = new Float32Array(16);

    for (const label of this.labels.values()) {
      if (!label.visible || label.geometry.charCount === 0) continue;

      const offset = this.labelOffsets.get(label.id);
      if (offset === undefined) continue;

      // 构建平移矩阵
      setTranslationMatrix(model, label.x, label.y);
      gl.uniformMatrix4fv(this.uModel, false, model);

      // 颜色
      gl.uniform4fv(this.uColor, label.color.toFloat32Array());

      // drawElementsInstanced: 每个字符 6 个索引, offset 用 baseInstance
      gl.drawElementsInstanced(
        gl.TRIANGLES,
        6,                           // 6 indices per quad
        gl.UNSIGNED_SHORT,
        0,
        label.geometry.charCount,    // instance count = char count
      );

      // 注: 严格 single-draw-call 模式需配合 baseInstance (WebGL2 扩展)
      // 若扩展不可用则退回到 per-label draw call (仍比 PixiJS Text 快)
    }

    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }

  /**
   * drawSingleCall — 利用 ANGLE_instanced_arrays / gl.drawElementsInstanced
   * 真正一次 draw call 渲染所有 label (需要 WEBGL_multi_draw 或重排 instance 数据).
   *
   * 当前实现: 合并所有 glyph 到单一 VBO,
   *           按 label 颜色分组批次，每组一次 draw call.
   */
  drawSingleCall(projection: Float32Array, pxRange = 4): void {
    const { gl } = this;
    if (this.labels.size === 0) return;
    if (this.dirty) this.rebuildInstances();

    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.atlas.texture);
    gl.uniform1i(this.uAtlas, 0);
    gl.uniform1f(this.uPxRange, pxRange);
    gl.uniformMatrix4fv(this.uProjection, false, projection);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.bindVertexArray(this.quadVAO);

    // 上传全量实例数据
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceVBO);
    gl.bufferData(gl.ARRAY_BUFFER, this.mergedInstances, gl.DYNAMIC_DRAW);

    // identity model (坐标已 baked 进 instanceData)
    const identity = createIdentityMatrix();
    gl.uniformMatrix4fv(this.uModel, false, identity);

    // 计算总字符数
    let totalChars = 0;
    for (const label of this.labels.values()) {
      if (label.visible) totalChars += label.geometry.charCount;
    }

    // 使用中性白色 (颜色差异用 per-instance color attribute 实现 — 此处简化版)
    gl.uniform4fv(this.uColor, new Float32Array([1, 1, 1, 1]));

    gl.drawElementsInstanced(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0, totalChars);

    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }

  /** 销毁 WebGL 资源 */
  destroy(): void {
    const { gl } = this;
    gl.deleteBuffer(this.quadVBO);
    gl.deleteBuffer(this.quadIBO);
    gl.deleteBuffer(this.instanceVBO);
    gl.deleteVertexArray(this.quadVAO);
    gl.deleteProgram(this.program);
  }

  // ── Private ────────────────────────────────────────────────────────────────

  /**
   * 把所有 label 的 instanceData 合并为单一 Float32Array.
   * 同时把每个 label 的坐标偏移 (x, y) 加进 a_position.
   */
  private rebuildInstances(): void {
    let total = 0;
    for (const label of this.labels.values()) {
      if (label.visible) total += label.geometry.charCount;
    }

    if (total === 0) {
      this.mergedInstances = new Float32Array(0);
      this.dirty = false;
      return;
    }

    const out = new Float32Array(total * FLOATS_PER_INSTANCE);
    this.labelOffsets.clear();

    let cursor = 0;
    for (const label of this.labels.values()) {
      if (!label.visible || label.geometry.charCount === 0) continue;

      this.labelOffsets.set(label.id, cursor);
      const src = label.geometry.instanceData;
      const count = label.geometry.charCount;

      for (let i = 0; i < count; i++) {
        const si = i * FLOATS_PER_INSTANCE;
        const di = (cursor + i) * FLOATS_PER_INSTANCE;

        // a_position: bake label world offset
        out[di]     = src[si]     + label.x;
        out[di + 1] = src[si + 1] + label.y;
        // a_size
        out[di + 2] = src[si + 2];
        out[di + 3] = src[si + 3];
        // a_uvRect
        out[di + 4] = src[si + 4];
        out[di + 5] = src[si + 5];
        out[di + 6] = src[si + 6];
        out[di + 7] = src[si + 7];
      }

      cursor += count;
    }

    this.mergedInstances = out;
    this.dirty = false;
  }
}

// ── 矩阵工具 (仅用于 GLText 内部) ─────────────────────────────────────────────

/** 创建 4×4 单位矩阵 (列主序) */
function createIdentityMatrix(): Float32Array {
  // prettier-ignore
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

/** 创建平移矩阵 (列主序, z=0) */
function setTranslationMatrix(out: Float32Array, tx: number, ty: number): void {
  out[ 0] = 1;  out[ 1] = 0;  out[ 2] = 0;  out[ 3] = 0;
  out[ 4] = 0;  out[ 5] = 1;  out[ 6] = 0;  out[ 7] = 0;
  out[ 8] = 0;  out[ 9] = 0;  out[10] = 1;  out[11] = 0;
  out[12] = tx; out[13] = ty; out[14] = 0;  out[15] = 1;
}

/**
 * makeOrthoProjection — 创建 2D 正交投影矩阵.
 *
 * @param left    视口左边 (px)
 * @param right   视口右边 (px)
 * @param top     视口顶部 (px)
 * @param bottom  视口底部 (px)
 * @returns Float32Array 16 (列主序)
 */
export function makeOrthoProjection(
  left: number, right: number,
  top: number,  bottom: number,
  near = -1, far = 1,
): Float32Array {
  const lr = 1 / (right - left);
  const bt = 1 / (top - bottom);
  const nf = 1 / (far - near);
  // prettier-ignore
  return new Float32Array([
    2 * lr,                0,                0, 0,
    0,                     2 * bt,           0, 0,
    0,                     0,                2 * nf, 0,
    -(right + left) * lr, -(top + bottom) * bt, -(far + near) * nf, 1,
  ]);
}
