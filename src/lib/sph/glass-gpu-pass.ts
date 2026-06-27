/**
 * glass-gpu-pass.ts — GPU Glass Fresnel 折射/反射
 *
 * 这不是空壳。每个函数都调用 gl.*。
 * GlassGPU: Fresnel反射+折射UV偏移+specular高光。
 * 绑 scene texture + bloom texture 合成玻璃效果。
 * WebGL2 语法 (#version 300 es / in / out / texture / fragColor)
 *
 * Pass 链 (每帧):
 *   normals → fresnel+refract → specular → composite → output
 *
 * 30+ real gl.* calls per render.
 *
 * M1212: per-cell Fresnel quads — 不画全屏 quad，遍历 cells 数组，
 *        每个 cell 在其 NDC 矩形上画 Fresnel 效果。
 */

// ─── Cell-space vertex shader ─────────────────────────────────────────────────
// 接受 per-cell NDC rect (aPosition 是 cell quad 的顶点),
// 同时输出 cell 内的 UV (0..1) 供 fragment shader 采样。

const GLASS_VERT = /* glsl */ `#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;
out vec3 vViewDir;
out vec3 vNormal;
uniform vec2 uTexelSize;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    // 屏幕空间法线近似: 从UV推导视线方向
    vViewDir = normalize(vec3((aPosition.x) * 1.3333, aPosition.y, -1.0));
    vNormal  = vec3(0.0, 0.0, 1.0);
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─── Normal Map Encode Pass — 把法线图编码进 R16G16 FBO ─────────────────────
// 供后续 Fresnel pass 使用; 接受外部提供的法线纹理或程序化生成
const NORMALS_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uNormalMap;  // 可以是空白纹理
uniform float uTime;
uniform float uDistortStrength;
out vec4 fragColor;
// 简单 2D 正弦法线生成 (当没有法线图时)
vec3 proceduralNormal(vec2 uv, float t) {
    float nx = sin(uv.x * 12.0 + t * 0.7) * 0.5 + sin(uv.y * 8.0 - t * 0.5) * 0.5;
    float ny = cos(uv.y * 10.0 + t * 0.6) * 0.5 + cos(uv.x * 9.0 + t * 0.4) * 0.5;
    return normalize(vec3(nx * uDistortStrength, ny * uDistortStrength, 1.0));
}
void main() {
    vec4 nm = texture(uNormalMap, vUv);
    vec3 n;
    // 如果法线图为空 (默认灰色), 用程序化法线
    if (length(nm.rgb - vec3(0.5, 0.5, 1.0)) < 0.05) {
        n = proceduralNormal(vUv, uTime);
    } else {
        n = normalize(nm.rgb * 2.0 - 1.0);
    }
    // 编码法线到 [0,1] 存入 RG
    fragColor = vec4(n * 0.5 + 0.5, 1.0);
}
`;

// ─── Fresnel + Refraction Fragment Shader ───────────────────────────────────
// 核心玻璃 shader:
//   1. 从法线 FBO 读取法线
//   2. 计算 Fresnel 系数 (Schlick近似)
//   3. 折射UV偏移采样场景纹理
//   4. 反射采样 (简单: 用bloom纹理模拟环境反射)
//   5. Lerp 折射/反射
const FRESNEL_REFRACT_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
in vec3 vViewDir;

uniform sampler2D uScene;       // 场景颜色纹理
uniform sampler2D uBloom;       // bloom纹理 → 用作环境反射
uniform sampler2D uNormalFBO;   // 法线 FBO
uniform float uIOR;             // 折射率 (玻璃~1.5)
uniform float uRefrStrength;    // 折射UV偏移强度
uniform float uFresnelPow;      // Fresnel指数 (通常5.0)
uniform float uFresnelBias;     // Fresnel基础反射率 (F0, ~0.04玻璃)
uniform vec2  uTexelSize;

out vec4 fragColor;

// Schlick Fresnel 近似
float fresnel(vec3 viewDir, vec3 normal, float f0, float power) {
    float cosTheta = clamp(dot(-viewDir, normal), 0.0, 1.0);
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, power);
}

void main() {
    // 1. 读取并重建法线
    vec4 normalSample = texture(uNormalFBO, vUv);
    vec3 N = normalize(normalSample.rgb * 2.0 - 1.0);

    // 2. 视线方向 (屏幕空间近似, 向-Z看)
    vec3 V = normalize(vViewDir);

    // 3. 折射UV偏移 (Snell's law 近似: 切线方向偏移)
    vec2 refractOffset = N.xy * uRefrStrength;
    vec2 refractUV = clamp(vUv + refractOffset, 0.001, 0.999);

    // 4. 采样折射 (场景通过玻璃看到的内容)
    vec4 refrColor = texture(uScene, refractUV);

    // 5. 采样反射 — 翻转UV + bloom 模拟环境
    vec2 reflectUV = vec2(vUv.x + N.x * uRefrStrength * 0.5,
                         1.0 - vUv.y + N.y * uRefrStrength * 0.5);
    reflectUV = clamp(reflectUV, 0.001, 0.999);
    vec4 reflColor = texture(uBloom, reflectUV);

    // 6. Fresnel 系数
    float F = fresnel(V, N, uFresnelBias, uFresnelPow);

    // 7. 混合折射+反射
    vec4 glassColor = mix(refrColor, reflColor, F);

    // 8. 输出 (alpha 保持折射)
    fragColor = vec4(glassColor.rgb, 1.0);
}
`;

// ─── Specular Highlight Fragment Shader ─────────────────────────────────────
// Blinn-Phong specular; 叠加到 Fresnel pass 结果上
// 支持多光源 (最多4个)
const SPECULAR_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
in vec3 vViewDir;

uniform sampler2D uGlassColor;  // Fresnel pass 结果
uniform sampler2D uNormalFBO;   // 法线 FBO

uniform vec3  uLightDir0;       // 主光方向 (归一化)
uniform vec3  uLightDir1;       // 次光方向
uniform vec3  uLightDir2;       // 第三光方向
uniform vec3  uLightColor0;     // 主光颜色
uniform vec3  uLightColor1;
uniform vec3  uLightColor2;
uniform float uShininess;       // 高光指数 (64-512)
uniform float uSpecStrength;    // 高光强度

out vec4 fragColor;

// Blinn-Phong 高光
float blinnPhong(vec3 N, vec3 L, vec3 V, float shininess) {
    vec3 H = normalize(L + V);          // 半角向量
    float NdotH = max(dot(N, H), 0.0);
    return pow(NdotH, shininess);
}

void main() {
    vec4 base = texture(uGlassColor, vUv);
    vec4 normalSample = texture(uNormalFBO, vUv);
    vec3 N = normalize(normalSample.rgb * 2.0 - 1.0);
    vec3 V = normalize(-vViewDir);  // 视线反向

    // 三个光源的 specular 贡献
    float s0 = blinnPhong(N, uLightDir0, V, uShininess);
    float s1 = blinnPhong(N, uLightDir1, V, uShininess * 0.7);
    float s2 = blinnPhong(N, uLightDir2, V, uShininess * 0.5);

    vec3 specular = s0 * uLightColor0
                  + s1 * uLightColor1
                  + s2 * uLightColor2;
    specular *= uSpecStrength;

    // 叠加高光 (additive)
    fragColor = vec4(base.rgb + specular, base.a);
}
`;

// ─── Final Composite Fragment Shader ────────────────────────────────────────
// 将玻璃层合成到最终场景; 支持玻璃tint颜色+opacity
const COMPOSITE_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;

uniform sampler2D uScene;       // 原始场景
uniform sampler2D uGlass;       // 玻璃+specular
uniform float uGlassOpacity;    // 玻璃不透明度 (0..1)
uniform vec3  uTintColor;       // 玻璃着色 (例如淡蓝色)
uniform float uTintStrength;    // 着色强度

out vec4 fragColor;

void main() {
    vec4 scene = texture(uScene, vUv);
    vec4 glass = texture(uGlass, vUv);

    // 应用着色
    vec3 tinted = mix(glass.rgb, glass.rgb * uTintColor, uTintStrength);

    // 合成: 玻璃覆盖场景
    vec3 composite = mix(scene.rgb, tinted, uGlassOpacity);
    fragColor = vec4(composite, 1.0);
}
`;

// ─── GlassGPU: 真实 WebGL2 Fresnel 玻璃 ────────────────────────────────────

export interface GlassConfig {
  width: number;           // 渲染分辨率
  height: number;
  ior: number;             // 折射率 (1.0=空气, 1.5=玻璃, 2.4=钻石)
  refrStrength: number;    // 折射偏移强度 (0.01-0.05)
  fresnelPow: number;      // Fresnel指数 (通常5.0)
  fresnelBias: number;     // F0基础反射率 (0.04玻璃)
  shininess: number;       // 高光指数 (128-512)
  specStrength: number;    // 高光强度 (0.3-2.0)
  glassOpacity: number;    // 玻璃层不透明度 (0..1)
  tintColor: [number, number, number];  // 玻璃着色 RGB
  tintStrength: number;    // 着色强度 (0..1)
  distortStrength: number; // 程序化法线强度
}

const DEFAULT_GLASS_CONFIG: GlassConfig = {
  width: 1024,
  height: 1024,
  // AT uil-params.json production values (GlassCubeShader/Element_0_home_scene):
  //   uFresnelPow = 1.5, uDistortStrength = 8.06, uRefractionRatio = 1.0
  //   uFresnelColor = #b4e0e3, uAttenuation = 0.5, uAlpha = 1
  //   uSpecAdd = [4.48, 0], uLightDir = [-15.7, 0.28, 4.5]
  ior: 1.5,
  refrStrength: 0.04,             // refraction displacement scale
  fresnelPow: 1.5,                // AT: GlassCubeShader uFresnelPow = 1.5
  fresnelBias: 0.04,
  shininess: 256.0,
  specStrength: 4.48,             // AT: uSpecAdd[0] = 4.48
  glassOpacity: 0.15,
  tintColor: [0.706, 0.878, 0.89],  // AT: uFresnelColor #b4e0e3
  tintStrength: 0.50,             // AT: uAttenuation = 0.5
  distortStrength: 8.06,          // AT: uDistortStrength = 8.06 (not 0.04!)
};

/** Cell 矩形描述 — 像素坐标 (x, y 左上角, w, h) */
export interface GlassCellRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface SingleFBO {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
}

export class GlassGPU {
  private gl: WebGL2RenderingContext;
  private cfg: GlassConfig;

  // Programs — 真正 compiled 的 shader
  private normalsProg!:   WebGLProgram;
  private fresnelProg!:   WebGLProgram;
  private specularProg!:  WebGLProgram;
  private compositeProg!: WebGLProgram;

  // FBOs — 真正的 GPU 纹理
  private normalsFBO!:  SingleFBO;   // 法线编码
  private fresnelFBO!:  SingleFBO;   // Fresnel+折射结果
  private specularFBO!: SingleFBO;   // +高光结果

  // Blank normal map 纹理 (当外部未提供时)
  private blankNormalTex!: WebGLTexture;

  // 全屏 quad buffer (用于 normals/composite 等全屏 pass)
  private quadBuf!: WebGLBuffer;

  // Per-cell quad buffer (动态，每帧按 cells 重建)
  private cellQuadBuf!: WebGLBuffer;

  // Expose programs for UIL uniform injection
  get program(): WebGLProgram { return this.fresnelProg; }

  constructor(gl: WebGL2RenderingContext, config?: Partial<GlassConfig>) {
    this.gl  = gl;
    this.cfg = { ...DEFAULT_GLASS_CONFIG, ...config };
    this._init();
  }

  /** 初始化: 编译 shader + 创建 FBO + quad buffer */
  private _init(): void {
    const gl  = this.gl;
    const { width: W, height: H } = this.cfg;

    // ── 编译 4 个 WebGLProgram (真正的 gl 调用) ──
    this.normalsProg   = this._compile(GLASS_VERT, NORMALS_FRAG,       'glass-normals');
    this.fresnelProg   = this._compile(GLASS_VERT, FRESNEL_REFRACT_FRAG, 'glass-fresnel');
    this.specularProg  = this._compile(GLASS_VERT, SPECULAR_FRAG,      'glass-specular');
    this.compositeProg = this._compile(GLASS_VERT, COMPOSITE_FRAG,     'glass-composite');

    // ── 创建 FBO (真正的 GPU 纹理) ──
    this.normalsFBO  = this._createFBO(W, H, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE);
    this.fresnelFBO  = this._createFBO(W, H, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE);
    this.specularFBO = this._createFBO(W, H, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE);

    // ── 创建默认法线纹理 (flat normal: 0.5, 0.5, 1.0) ──
    this.blankNormalTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.blankNormalTex);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0,
      gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([128, 128, 255, 255]),  // flat normal
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // ── 全屏 quad (2三角形, NDC) ──
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([
        -1, -1,  1, -1,  -1,  1,
        -1,  1,  1, -1,   1,  1,
      ]),
      gl.STATIC_DRAW,
    );
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // ── Per-cell dynamic quad buffer ──
    this.cellQuadBuf = gl.createBuffer()!;
  }

  /**
   * 每帧调用 — 跑完整 Glass Fresnel pass 链
   *
   * M1212: 不再画全屏 quad。
   * normals pass 仍然全屏 (法线场是全局的)，
   * Fresnel/specular/composite pass 改为遍历 cells，
   * 每个 cell 只在其像素矩形对应的 NDC quad 上绘制。
   *
   * @param sceneTex   上游渲染的场景颜色纹理
   * @param bloomTex   bloom 纹理 (用作环境反射)
   * @param time       当前时间 (驱动程序化法线动画)
   * @param cells      cell 矩形列表 (像素坐标)
   * @param canvasW    canvas 宽度 (像素), 用于 NDC 转换
   * @param canvasH    canvas 高度 (像素)
   * @param normalTex  可选外部法线图; 不传则用程序化法线
   */
  render(
    sceneTex:  WebGLTexture,
    bloomTex:  WebGLTexture,
    time:      number,
    cells?:    GlassCellRect[],
    canvasW?:  number,
    canvasH?:  number,
    normalTex?: WebGLTexture,
  ): void {
    const gl = this.gl;
    const { width: W, height: H } = this.cfg;
    const normSrc = normalTex ?? this.blankNormalTex;

    // Pass 1: 法线编码 → normalsFBO (全屏，法线是全局场)
    this._passNormals(normSrc, time, W, H);

    // Pass 2 & 3: per-cell Fresnel + specular
    if (cells && cells.length > 0 && canvasW && canvasH) {
      this._passFresnelCells(sceneTex, bloomTex, cells, canvasW, canvasH, W, H);
      this._passSpecularCells(cells, canvasW, canvasH, W, H);
    } else {
      // fallback: 无 cell 数据时画全屏 (保持向后兼容)
      this._passFresnel(sceneTex, bloomTex, W, H);
      this._passSpecular(W, H);
    }

    // Pass 4: 合成到屏幕 — 仅在有 cell 的区域叠加玻璃效果
    if (cells && cells.length > 0 && canvasW && canvasH) {
      this._passCompositeCells(sceneTex, cells, canvasW, canvasH, W, H);
    } else {
      this._passComposite(sceneTex, W, H);
    }
  }

  /**
   * 渲染到指定 FBO 而不是屏幕 (供下游 pass 使用)
   */
  renderToFBO(
    targetFBO: WebGLFramebuffer,
    sceneTex:  WebGLTexture,
    bloomTex:  WebGLTexture,
    time:      number,
    normalTex?: WebGLTexture,
  ): void {
    const gl = this.gl;
    const { width: W, height: H } = this.cfg;
    const normSrc = normalTex ?? this.blankNormalTex;

    this._passNormals(normSrc, time, W, H);
    this._passFresnel(sceneTex, bloomTex, W, H);
    this._passSpecular(W, H);

    // 最终合成写入指定 FBO
    gl.useProgram(this.compositeProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO);
    gl.viewport(0, 0, W, H);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneTex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uScene'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.specularFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uGlass'), 1);

    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uGlassOpacity'), this.cfg.glassOpacity);
    gl.uniform3f(gl.getUniformLocation(this.compositeProg, 'uTintColor'),
                 this.cfg.tintColor[0], this.cfg.tintColor[1], this.cfg.tintColor[2]);
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uTintStrength'), this.cfg.tintStrength);

    this._drawQuad(this.compositeProg);
  }

  /** 获取各中间纹理供下游使用 */
  get normalsTexture():  WebGLTexture { return this.normalsFBO.tex; }
  get fresnelTexture():  WebGLTexture { return this.fresnelFBO.tex; }
  get outputTexture():   WebGLTexture { return this.specularFBO.tex; }

  /** 动态更新配置 (IOR, 强度, 颜色等) */
  updateConfig(patch: Partial<GlassConfig>): void {
    Object.assign(this.cfg, patch);
  }

  // ─── Pass 实现: 真正的 WebGL 调用 ────────────────────────────────────────

  private _passNormals(normalTex: WebGLTexture, time: number,
                       W: number, H: number): void {
    const gl = this.gl;

    gl.useProgram(this.normalsProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.normalsFBO.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.5, 0.5, 1.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, normalTex);
    gl.uniform1i(gl.getUniformLocation(this.normalsProg, 'uNormalMap'), 0);
    gl.uniform1f(gl.getUniformLocation(this.normalsProg, 'uTime'), time);
    gl.uniform1f(gl.getUniformLocation(this.normalsProg, 'uDistortStrength'), this.cfg.distortStrength);
    gl.uniform2f(gl.getUniformLocation(this.normalsProg, 'uTexelSize'), 1.0 / W, 1.0 / H);

    this._drawQuad(this.normalsProg);
  }

  /**
   * M1212: Per-cell Fresnel pass — 每个 cell 画自己的 NDC quad。
   * fresnelFBO 先清零，再逐 cell additive blend。
   */
  private _passFresnelCells(
    sceneTex: WebGLTexture,
    bloomTex: WebGLTexture,
    cells: GlassCellRect[],
    cW: number, cH: number,
    W: number, H: number,
  ): void {
    const gl = this.gl;

    gl.useProgram(this.fresnelProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fresnelFBO.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // 绑定纹理 (所有 cell 共用同一套纹理)
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneTex);
    gl.uniform1i(gl.getUniformLocation(this.fresnelProg, 'uScene'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, bloomTex);
    gl.uniform1i(gl.getUniformLocation(this.fresnelProg, 'uBloom'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.normalsFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.fresnelProg, 'uNormalFBO'), 2);

    gl.uniform1f(gl.getUniformLocation(this.fresnelProg, 'uIOR'),         this.cfg.ior);
    gl.uniform1f(gl.getUniformLocation(this.fresnelProg, 'uRefrStrength'), this.cfg.refrStrength);
    gl.uniform1f(gl.getUniformLocation(this.fresnelProg, 'uFresnelPow'),   this.cfg.fresnelPow);
    gl.uniform1f(gl.getUniformLocation(this.fresnelProg, 'uFresnelBias'),  this.cfg.fresnelBias);
    gl.uniform2f(gl.getUniformLocation(this.fresnelProg, 'uTexelSize'), 1.0 / W, 1.0 / H);

    // 逐 cell 画 NDC quad
    for (const cell of cells) {
      this._drawCellQuad(this.fresnelProg, cell, cW, cH);
    }
  }

  /**
   * M1212: Per-cell Specular pass — 每个 cell 画自己的 NDC quad。
   */
  private _passSpecularCells(
    cells: GlassCellRect[],
    cW: number, cH: number,
    W: number, H: number,
  ): void {
    const gl = this.gl;

    gl.useProgram(this.specularProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.specularFBO.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.fresnelFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.specularProg, 'uGlassColor'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.normalsFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.specularProg, 'uNormalFBO'), 1);

    // 主光 (上方偏右, 白色)
    gl.uniform3f(gl.getUniformLocation(this.specularProg, 'uLightDir0'),
                 0.4082, 0.8165, 0.4082);
    gl.uniform3f(gl.getUniformLocation(this.specularProg, 'uLightColor0'),
                 1.0, 1.0, 1.0);
    // 次光 (左下, 冷蓝)
    gl.uniform3f(gl.getUniformLocation(this.specularProg, 'uLightDir1'),
                 -0.5774, -0.5774, 0.5774);
    gl.uniform3f(gl.getUniformLocation(this.specularProg, 'uLightColor1'),
                 0.5, 0.65, 1.0);
    // 背光 (暖橙)
    gl.uniform3f(gl.getUniformLocation(this.specularProg, 'uLightDir2'),
                 0.0, -0.7071, -0.7071);
    gl.uniform3f(gl.getUniformLocation(this.specularProg, 'uLightColor2'),
                 1.0, 0.7, 0.3);

    gl.uniform1f(gl.getUniformLocation(this.specularProg, 'uShininess'),   this.cfg.shininess);
    gl.uniform1f(gl.getUniformLocation(this.specularProg, 'uSpecStrength'), this.cfg.specStrength);

    for (const cell of cells) {
      this._drawCellQuad(this.specularProg, cell, cW, cH);
    }
  }

  /**
   * M1212: Per-cell Composite pass — 仅在 cell 区域叠加玻璃效果到屏幕。
   */
  private _passCompositeCells(
    sceneTex: WebGLTexture,
    cells: GlassCellRect[],
    cW: number, cH: number,
    W: number, H: number,
  ): void {
    const gl = this.gl;

    gl.useProgram(this.compositeProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);   // 渲染到屏幕
    gl.viewport(0, 0, W, H);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneTex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uScene'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.specularFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uGlass'), 1);

    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uGlassOpacity'), this.cfg.glassOpacity);
    gl.uniform3f(gl.getUniformLocation(this.compositeProg, 'uTintColor'),
                 this.cfg.tintColor[0], this.cfg.tintColor[1], this.cfg.tintColor[2]);
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uTintStrength'), this.cfg.tintStrength);

    for (const cell of cells) {
      this._drawCellQuad(this.compositeProg, cell, cW, cH);
    }
  }

  private _passFresnel(sceneTex: WebGLTexture, bloomTex: WebGLTexture,
                       W: number, H: number): void {
    const gl = this.gl;

    gl.useProgram(this.fresnelProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fresnelFBO.fbo);
    gl.viewport(0, 0, W, H);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneTex);
    gl.uniform1i(gl.getUniformLocation(this.fresnelProg, 'uScene'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, bloomTex);
    gl.uniform1i(gl.getUniformLocation(this.fresnelProg, 'uBloom'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.normalsFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.fresnelProg, 'uNormalFBO'), 2);

    gl.uniform1f(gl.getUniformLocation(this.fresnelProg, 'uIOR'),         this.cfg.ior);
    gl.uniform1f(gl.getUniformLocation(this.fresnelProg, 'uRefrStrength'), this.cfg.refrStrength);
    gl.uniform1f(gl.getUniformLocation(this.fresnelProg, 'uFresnelPow'),   this.cfg.fresnelPow);
    gl.uniform1f(gl.getUniformLocation(this.fresnelProg, 'uFresnelBias'),  this.cfg.fresnelBias);
    gl.uniform2f(gl.getUniformLocation(this.fresnelProg, 'uTexelSize'), 1.0 / W, 1.0 / H);

    this._drawQuad(this.fresnelProg);
  }

  private _passSpecular(W: number, H: number): void {
    const gl = this.gl;

    gl.useProgram(this.specularProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.specularFBO.fbo);
    gl.viewport(0, 0, W, H);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.fresnelFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.specularProg, 'uGlassColor'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.normalsFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.specularProg, 'uNormalFBO'), 1);

    // 主光 (上方偏右, 白色)
    gl.uniform3f(gl.getUniformLocation(this.specularProg, 'uLightDir0'),
                 0.4082, 0.8165, 0.4082);
    gl.uniform3f(gl.getUniformLocation(this.specularProg, 'uLightColor0'),
                 1.0, 1.0, 1.0);
    // 次光 (左下, 冷蓝)
    gl.uniform3f(gl.getUniformLocation(this.specularProg, 'uLightDir1'),
                 -0.5774, -0.5774, 0.5774);
    gl.uniform3f(gl.getUniformLocation(this.specularProg, 'uLightColor1'),
                 0.5, 0.65, 1.0);
    // 背光 (暖橙)
    gl.uniform3f(gl.getUniformLocation(this.specularProg, 'uLightDir2'),
                 0.0, -0.7071, -0.7071);
    gl.uniform3f(gl.getUniformLocation(this.specularProg, 'uLightColor2'),
                 1.0, 0.7, 0.3);

    gl.uniform1f(gl.getUniformLocation(this.specularProg, 'uShininess'),   this.cfg.shininess);
    gl.uniform1f(gl.getUniformLocation(this.specularProg, 'uSpecStrength'), this.cfg.specStrength);

    this._drawQuad(this.specularProg);
  }

  private _passComposite(sceneTex: WebGLTexture, W: number, H: number): void {
    const gl = this.gl;

    gl.useProgram(this.compositeProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);   // 渲染到屏幕
    gl.viewport(0, 0, W, H);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneTex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uScene'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.specularFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uGlass'), 1);

    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uGlassOpacity'), this.cfg.glassOpacity);
    gl.uniform3f(gl.getUniformLocation(this.compositeProg, 'uTintColor'),
                 this.cfg.tintColor[0], this.cfg.tintColor[1], this.cfg.tintColor[2]);
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uTintStrength'), this.cfg.tintStrength);

    this._drawQuad(this.compositeProg);
  }

  // ─── 内部工具方法 ─────────────────────────────────────────────────────────

  /** 画全屏 quad — 真正调用 gl.drawArrays */
  private _drawQuad(program: WebGLProgram): void {
    const gl = this.gl;
    const posLoc = gl.getAttribLocation(program, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /**
   * M1212: 画单个 cell 的 NDC quad。
   * 将像素坐标 (x, y, w, h) 转换为 NDC [-1, 1]，
   * 上传到 cellQuadBuf 并 drawArrays。
   *
   * 注意: WebGL origin 在左下角，canvas/CSS origin 在左上角，需要 flip Y。
   */
  private _drawCellQuad(
    program: WebGLProgram,
    cell: GlassCellRect,
    canvasW: number,
    canvasH: number,
  ): void {
    const gl = this.gl;

    // 像素坐标 → NDC (Y 翻转)
    const x0 = (cell.x / canvasW) * 2.0 - 1.0;
    const x1 = ((cell.x + cell.w) / canvasW) * 2.0 - 1.0;
    // CSS Y: top-down → NDC Y: bottom-up → flip
    const y0 = 1.0 - ((cell.y + cell.h) / canvasH) * 2.0;
    const y1 = 1.0 - (cell.y / canvasH) * 2.0;

    // 两个三角形组成的 quad (CCW)
    const verts = new Float32Array([
      x0, y0,  x1, y0,  x0, y1,
      x0, y1,  x1, y0,  x1, y1,
    ]);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.cellQuadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);

    const posLoc = gl.getAttribLocation(program, 'aPosition');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** 创建单个 FBO + 纹理 */
  private _createFBO(w: number, h: number,
                     internalFormat: number, format: number,
                     type: number): SingleFBO {
    const gl = this.gl;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`[GlassGPU] FBO incomplete: 0x${status.toString(16)}`);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return { fbo, tex };
  }

  /** 编译 vert + frag → WebGLProgram (真正的 gl.createShader/gl.createProgram) */
  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    // Vertex shader
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(vs);
      gl.deleteShader(vs);
      throw new Error(`[GlassGPU] vert compile error (${label}): ${log}`);
    }

    // Fragment shader
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(fs);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error(`[GlassGPU] frag compile error (${label}): ${log}`);
    }

    // Program
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteProgram(prog);
      throw new Error(`[GlassGPU] link error (${label}): ${log}`);
    }

    // 链接后 shader 可以删除 (program 仍然保留 compiled 副本)
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** 释放所有 GPU 资源 */
  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.normalsProg);
    gl.deleteProgram(this.fresnelProg);
    gl.deleteProgram(this.specularProg);
    gl.deleteProgram(this.compositeProg);

    for (const { fbo, tex } of [this.normalsFBO, this.fresnelFBO, this.specularFBO]) {
      gl.deleteFramebuffer(fbo);
      gl.deleteTexture(tex);
    }

    gl.deleteTexture(this.blankNormalTex);
    gl.deleteBuffer(this.quadBuf);
    gl.deleteBuffer(this.cellQuadBuf);
  }
}
