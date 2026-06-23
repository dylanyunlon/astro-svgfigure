/**
 * composite-gpu-pass.ts — M879: GPU Final Composite
 *
 * CompositeGPU: 把 6 个 G-Buffer 层在 GPU 上合成为最终画面。
 * 直接输出到 canvas (null framebuffer), 无额外 FBO。
 *
 * 输入层 (6 个 sampler2D):
 *   cell     — 细胞本体 (PBR/matcap 主色)
 *   edge     — 边/连接线
 *   particle — 粒子层 (additive)
 *   bloom    — bloom (screen blend)
 *   shadow   — 阴影遮罩
 *   fluid    — 流体 (subtle UV distortion)
 *
 * 合成顺序 (z-order, 低→高):
 *   cell * shadow  → edge → particle (additive) → bloom (screen) → fluid (distort)
 *
 * 后处理:
 *   vignette     — 1 - distance_from_center^2
 *   film grain   — fract(sin(dot(uv, seed)) * 43758.5453)
 *   color grading— simple RGB curves (shadows/midtones/highlights)
 *
 * 遵循 fluid-gpu-pass.ts 的写法:
 *   • 真实 gl.createShader / gl.createProgram / gl.compileShader
 *   • 真实 gl.bindTexture × 6 (units 0-5)
 *   • 真实 gl.bindFramebuffer(null) + gl.drawArrays
 *   • 从 compiled.vs 通过 getShader() 提取 AT shader
 *   • WebGL1 语法 (varying / texture2D)
 */

import { getShader } from '../shaders/ShaderLoader';

// ─── Vertex shader (WebGL1, fullscreen quad) ─────────────────────────────────

const COMPOSITE_VERT = /* glsl */ `
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─── Fragment shader (~50 行, WebGL1) ────────────────────────────────────────
// 手写合成 shader (非从 compiled.vs 提取, 因 compiled.vs 无 composite.fs entry).
// AT 惯例: 最终合成 shader 内联于 ts 文件 (参见 at-scene-composite-shaders.ts).

const COMPOSITE_FRAG = /* glsl */ `
precision highp float;

// 6 输入层
uniform sampler2D uCell;
uniform sampler2D uEdge;
uniform sampler2D uParticle;
uniform sampler2D uBloom;
uniform sampler2D uShadow;
uniform sampler2D uFluid;

// 后处理参数
uniform float uTime;
uniform float uGrainStrength;    // 胶片颗粒强度 (default 0.03)
uniform float uVignetteStrength; // 暗角强度     (default 0.6)
uniform vec3  uShadowColor;      // 暗部色调     (default vec3(0.05,0.02,0.08))
uniform vec3  uHighlightColor;   // 亮部色调     (default vec3(1.0,0.98,0.95))
uniform vec2  uResolution;       // 画布分辨率 (px)

varying vec2 vUv;

// ── Screen blend ─────────────────────────────────────────────────────────────
vec3 blendScreen(vec3 base, vec3 blend) {
    return 1.0 - (1.0 - base) * (1.0 - blend);
}

// ── Film grain (经典 fract/sin hash) ─────────────────────────────────────────
float filmGrain(vec2 uv, float t) {
    vec2 seed = uv * uResolution + vec2(t * 617.3, t * 131.7);
    return fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
}

// ── Simple RGB curve (shadows / highlights) ───────────────────────────────────
vec3 colorGrade(vec3 col) {
    // 亮度提升曲线: pow 轻微压暗中间调, 再 mix 阴影/高光色
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    // 暗部 → uShadowColor, 亮部 → uHighlightColor
    vec3 graded = mix(uShadowColor, uHighlightColor, luma);
    // 与原色混合 (保留饱和度)
    graded = mix(col, graded, 0.25);
    // Gamma-ish contrast toe
    graded = pow(clamp(graded, 0.0, 1.0), vec3(0.95));
    return graded;
}

void main() {
    vec2 uv = vUv;

    // ── 1. fluid UV distortion (subtle, 扭曲后续所有采样) ────────────────────
    vec2 fluidVel = texture2D(uFluid, uv).rg;           // velocity in [0,1]
    fluidVel = (fluidVel - 0.5) * 2.0;                  // remap → [-1, 1]
    vec2 distortedUv = uv + fluidVel * 0.004;           // 0.4% 最大偏移
    distortedUv = clamp(distortedUv, 0.0, 1.0);

    // ── 2. 采样所有层 (全部用 distortedUv 保持一致扭曲) ──────────────────────
    vec4 cellColor     = texture2D(uCell,     distortedUv);
    vec4 edgeColor     = texture2D(uEdge,     distortedUv);
    vec4 particleColor = texture2D(uParticle, distortedUv);
    vec4 bloomColor    = texture2D(uBloom,    distortedUv);
    float shadowMask   = texture2D(uShadow,   distortedUv).r; // 0=dark, 1=lit

    // ── 3. Z-order 合成 ──────────────────────────────────────────────────────
    // 底层: cell 乘以 shadow 遮罩
    vec3 composite = cellColor.rgb * shadowMask;

    // 边层: alpha-over 混合
    composite = mix(composite, edgeColor.rgb, edgeColor.a);

    // 粒子层: additive (premultiplied alpha)
    composite += particleColor.rgb * particleColor.a;

    // bloom: screen blend
    composite = blendScreen(composite, bloomColor.rgb * bloomColor.a);

    // ── 4. 后处理 ────────────────────────────────────────────────────────────

    // vignette: 1 - distance_from_center^2, 由 uVignetteStrength 控制强度
    vec2 centeredUv = uv * 2.0 - 1.0;
    float vignette  = 1.0 - dot(centeredUv, centeredUv) * uVignetteStrength;
    vignette        = clamp(vignette, 0.0, 1.0);
    composite      *= vignette;

    // film grain
    float grain = filmGrain(uv, uTime);
    composite  += grain * uGrainStrength;

    // color grading
    composite = colorGrade(composite);

    gl_FragColor = vec4(clamp(composite, 0.0, 1.0), 1.0);
}
`;

// ─── CompositeInputs: 每帧传入的 6 个纹理 ────────────────────────────────────

export interface CompositeInputs {
  cell:     WebGLTexture;
  edge:     WebGLTexture;
  particle: WebGLTexture;
  bloom:    WebGLTexture;
  shadow:   WebGLTexture;
  fluid:    WebGLTexture;
}

// ─── CompositeConfig: 后处理参数 ─────────────────────────────────────────────

export interface CompositeConfig {
  grainStrength:    number;   // 胶片颗粒强度 [0, 0.1]   default 0.03
  vignetteStrength: number;   // 暗角强度     [0, 1.5]  default 0.6
  shadowColor:      [number, number, number];   // 暗部色调 default [0.05,0.02,0.08]
  highlightColor:   [number, number, number];   // 亮部色调 default [1.0,0.98,0.95]
}

const DEFAULT_CONFIG: CompositeConfig = {
  grainStrength:    0.03,
  vignetteStrength: 0.6,
  shadowColor:      [0.05, 0.02, 0.08],
  highlightColor:   [1.0,  0.98, 0.95],
};

// ─── CompositeGPU ────────────────────────────────────────────────────────────

export class CompositeGPU {
  private gl:      WebGLRenderingContext;
  private config:  CompositeConfig;

  // WebGL resources — 真正 compiled 的 shader + quad geometry
  private prog!:   WebGLProgram;
  private quadBuf!: WebGLBuffer;

  // Uniform locations (cached for performance)
  private uCell!:             WebGLUniformLocation;
  private uEdge!:             WebGLUniformLocation;
  private uParticle!:         WebGLUniformLocation;
  private uBloom!:            WebGLUniformLocation;
  private uShadow!:           WebGLUniformLocation;
  private uFluid!:            WebGLUniformLocation;
  private uTime!:             WebGLUniformLocation;
  private uGrainStrength!:    WebGLUniformLocation;
  private uVignetteStrength!: WebGLUniformLocation;
  private uShadowColor!:      WebGLUniformLocation;
  private uHighlightColor!:   WebGLUniformLocation;
  private uResolution!:       WebGLUniformLocation;
  private aPosition!:         number;

  constructor(gl: WebGLRenderingContext, config?: Partial<CompositeConfig>) {
    this.gl     = gl;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this._init();
  }

  // ─── 初始化 ────────────────────────────────────────────────────────────────

  private _init(): void {
    const gl = this.gl;

    // ── 从 compiled.vs 尝试提取 AT 合成 shader (若无则用内联 FRAG) ──────────
    // AT 的 composite shader 以内联字符串形式存在于 at-scene-composite-shaders.ts;
    // compiled.vs 里没有独立的 composite.fs entry, 所以我们用内联 COMPOSITE_FRAG.
    // 但仍通过 getShader 尝试加载 (留给未来版本扩展).
    let fragSrc = COMPOSITE_FRAG;
    try {
      // 若将来 compiled.vs 中新增 composite.fs, 自动使用 AT 版本
      const atFrag = getShader('composite.fs');
      if (atFrag && atFrag.length > 0) {
        fragSrc = atFrag;
      }
    } catch (_) {
      // composite.fs 未找到 → 使用内联版本 (预期行为)
    }

    // ── 编译 composite shader (真正的 gl 调用) ──────────────────────────────
    this.prog = this._compile(COMPOSITE_VERT, fragSrc, 'composite');

    // ── 缓存 uniform / attrib locations ──────────────────────────────────────
    this.uCell             = gl.getUniformLocation(this.prog, 'uCell')!;
    this.uEdge             = gl.getUniformLocation(this.prog, 'uEdge')!;
    this.uParticle         = gl.getUniformLocation(this.prog, 'uParticle')!;
    this.uBloom            = gl.getUniformLocation(this.prog, 'uBloom')!;
    this.uShadow           = gl.getUniformLocation(this.prog, 'uShadow')!;
    this.uFluid            = gl.getUniformLocation(this.prog, 'uFluid')!;
    this.uTime             = gl.getUniformLocation(this.prog, 'uTime')!;
    this.uGrainStrength    = gl.getUniformLocation(this.prog, 'uGrainStrength')!;
    this.uVignetteStrength = gl.getUniformLocation(this.prog, 'uVignetteStrength')!;
    this.uShadowColor      = gl.getUniformLocation(this.prog, 'uShadowColor')!;
    this.uHighlightColor   = gl.getUniformLocation(this.prog, 'uHighlightColor')!;
    this.uResolution       = gl.getUniformLocation(this.prog, 'uResolution')!;
    this.aPosition         = gl.getAttribLocation(this.prog, 'aPosition');

    // ── 全屏 quad (2 个三角形, 同 fluid-gpu-pass.ts) ─────────────────────────
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);
  }

  // ─── 每帧调用: 把 6 个输入纹理合成到 canvas ────────────────────────────────

  render(
    inputs: CompositeInputs,
    canvasWidth: number,
    canvasHeight: number,
    time: number,
  ): void {
    const gl = this.gl;

    // 输出直接到 canvas (null framebuffer)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasWidth, canvasHeight);

    gl.useProgram(this.prog);

    // ── 上传 6 个 sampler2D (texture units 0-5) ───────────────────────────────

    // unit 0 — cell
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputs.cell);
    gl.uniform1i(this.uCell, 0);

    // unit 1 — edge
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, inputs.edge);
    gl.uniform1i(this.uEdge, 1);

    // unit 2 — particle
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, inputs.particle);
    gl.uniform1i(this.uParticle, 2);

    // unit 3 — bloom
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, inputs.bloom);
    gl.uniform1i(this.uBloom, 3);

    // unit 4 — shadow
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, inputs.shadow);
    gl.uniform1i(this.uShadow, 4);

    // unit 5 — fluid
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, inputs.fluid);
    gl.uniform1i(this.uFluid, 5);

    // ── 上传后处理 uniforms ───────────────────────────────────────────────────
    gl.uniform1f(this.uTime,             time);
    gl.uniform1f(this.uGrainStrength,    this.config.grainStrength);
    gl.uniform1f(this.uVignetteStrength, this.config.vignetteStrength);
    gl.uniform3f(this.uShadowColor,
      this.config.shadowColor[0],
      this.config.shadowColor[1],
      this.config.shadowColor[2]);
    gl.uniform3f(this.uHighlightColor,
      this.config.highlightColor[0],
      this.config.highlightColor[1],
      this.config.highlightColor[2]);
    gl.uniform2f(this.uResolution, canvasWidth, canvasHeight);

    // ── 画全屏 quad → gl.drawArrays ──────────────────────────────────────────
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(this.aPosition);
    gl.vertexAttribPointer(this.aPosition, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  // ─── 运行时更新后处理参数 ──────────────────────────────────────────────────

  setConfig(partial: Partial<CompositeConfig>): void {
    Object.assign(this.config, partial);
  }

  // ─── 销毁: 释放 GPU 资源 ────────────────────────────────────────────────────

  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.prog);
    gl.deleteBuffer(this.quadBuf);
  }

  // ─── 内部: 编译 WebGL1 shader ──────────────────────────────────────────────

  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    // vertex shader
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(
        `[CompositeGPU] vertex compile error (${label}): ${gl.getShaderInfoLog(vs)}`
      );
    }

    // fragment shader
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(
        `[CompositeGPU] fragment compile error (${label}): ${gl.getShaderInfoLog(fs)}`
      );
    }

    // link
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(
        `[CompositeGPU] link error (${label}): ${gl.getProgramInfoLog(prog)}`
      );
    }

    // 编译完成后即可删除 shader 对象 (AT 惯例)
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    return prog;
  }
}
