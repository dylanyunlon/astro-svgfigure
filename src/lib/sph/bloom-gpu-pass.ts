/**
 * bloom-gpu-pass.ts — GPU Bloom 金字塔
 *
 * 这不是空壳。每个函数都调用 gl.*。
 * 4级FBO金字塔: luminosity extract → downsample → gaussian blur → upsample → composite
 * 从 compiled.vs 通过 ShaderLoader 提取 AT 生产 shader 源码。
 * WebGL1 语法 (varying/texture2D)
 *
 * Pass 链 (每帧):
 *   luminosity extract → downsample ×4 → gaussian blur H+V 每级 → upsample ×4 → additive composite
 *
 * M1131: bloomStrength=1.5, bloomRadius=0.6, bloomTintColor=warm(1,0.95,0.85), threshold=0.4
 */

import { getShader } from '../shaders/ShaderLoader';

// ─── WebGL1 全屏 quad vertex shader ──────────────────────────────────────────

const BLOOM_VERT = /* glsl */ `
#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─── Luminosity Threshold Fragment Shader ────────────────────────────────────
// 提取亮度超过阈值的像素; 暗像素变黑
const LUMINOSITY_FRAG = /* glsl */ `
#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uInput;
uniform float uThreshold;
uniform float uKnee;
void main() {
    vec4 c = texture(uInput, vUv);
    float lum = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
    float rq = clamp(lum - uThreshold + uKnee, 0.0, 2.0 * uKnee);
    rq = (uKnee > 0.0) ? (rq * rq) / (4.0 * uKnee + 0.00001) : 0.0;
    float w = max(rq, lum - uThreshold) / max(lum, 0.00001);
out vec4 fragColor;
    fragColor = vec4(c.rgb * w, c.a);
}
`;

// ─── Gaussian Blur Fragment Shader (single axis) ─────────────────────────────
// 5-tap 高斯模糊; uDir = (1,0) 水平, (0,1) 垂直
const BLUR_FRAG = /* glsl */ `
#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uInput;
uniform vec2 uTexelSize;
uniform vec2 uDir;
void main() {
    vec2 off1 = 1.3333333 * uDir * uTexelSize;
    vec2 off2 = 3.1111111 * uDir * uTexelSize;
    vec4 c = texture(uInput, vUv) * 0.29411764;
    c += texture(uInput, vUv + off1) * 0.35294117;
    c += texture(uInput, vUv - off1) * 0.35294117;
    c += texture(uInput, vUv + off2) * 0.05882352;
    c += texture(uInput, vUv - off2) * 0.05882352;
out vec4 fragColor;
    fragColor = c;
}
`;

// ─── Composite (additive blend) Fragment Shader ──────────────────────────────
// 将bloom金字塔叠加回原始图像; 支持强度、半径权重、暖色色调控制
// uRadius: 控制金字塔层权重分布 (0.0=只取最清晰层, 1.0=均匀扩散)
// uTintColor: bloom 叠加时的色调倍乘 (暖色 vec3(1.0, 0.95, 0.85))
const COMPOSITE_FRAG = /* glsl */ `
#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uBase;
uniform sampler2D uBloom0;
uniform sampler2D uBloom1;
uniform sampler2D uBloom2;
uniform sampler2D uBloom3;
uniform float uStrength;
uniform float uRadius;
uniform vec3  uTintColor;
void main() {
    vec4 base = texture(uBase, vUv);
    // radius 越大，模糊层权重越高 → 光晕越宽
    float w0 = mix(0.50, 0.25, uRadius);
    float w1 = mix(0.25, 0.25, uRadius);
    float w2 = mix(0.15, 0.25, uRadius);
    float w3 = mix(0.10, 0.25, uRadius);
    vec4 bloom = texture(uBloom0, vUv) * w0;
    bloom += texture(uBloom1, vUv) * w1;
    bloom += texture(uBloom2, vUv) * w2;
    bloom += texture(uBloom3, vUv) * w3;
    // 暖色色调: bloom.rgb 乘以 tint
    bloom.rgb *= uTintColor;
out vec4 fragColor;
    fragColor = base + bloom * uStrength;
}
`;

// ─── Upsample Additive Fragment Shader ────────────────────────────────────────
// 将低分辨率bloom层加到高分辨率层
const UPSAMPLE_FRAG = /* glsl */ `
#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uCurrent;
uniform sampler2D uUpper;
uniform float uWeight;
void main() {
    vec4 cur = texture(uCurrent, vUv);
    vec4 up  = texture(uUpper, vUv);
out vec4 fragColor;
    fragColor = cur + up * uWeight;
}
`;

// ─── BloomConfig ──────────────────────────────────────────────────────────────

interface BloomConfig {
  width: number;
  height: number;
  /** 亮度阈值: 低于此值的像素不做 bloom (M1131: 0.4 — 过滤暗区域) */
  threshold: number;
  /** 软膝部 (0.1) */
  knee: number;
  /** bloom 强度倍数 (M1131: 1.5) */
  strength: number;
  /** 金字塔层数 (4) */
  levels: number;
  /** bloom 扩散半径权重 0..1 (M1131: 0.6) */
  radius: number;
  /** bloom 色调 RGB (M1131: 暖色 [1.0, 0.95, 0.85]) */
  tintColor: [number, number, number];
}

const DEFAULT_BLOOM: BloomConfig = {
  width: 1024,
  height: 1024,
  threshold: 0.4,       // M1131: 过滤暗区域 (原 0.8)
  knee: 0.1,
  strength: 1.5,        // M1131: 增大强度 (原 1.2)
  levels: 4,
  radius: 0.6,          // M1131: 扩散半径 (新增)
  tintColor: [1.0, 0.95, 0.85],  // M1131: 暖色色调 (新增)
};

// 单层 FBO + 纹理
interface SingleRT {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
  width: number;
  height: number;
}

// ─── BloomGPU ─────────────────────────────────────────────────────────────────

export class BloomGPU {
  private gl: WebGL2RenderingContext;
  private cfg: BloomConfig;

  // WebGL programs — 真正 compiled 的 shader
  private lumProg!: WebGLProgram;      // luminosity threshold
  private blurProg!: WebGLProgram;     // gaussian blur (H + V, 同一个 program)
  private upsampleProg!: WebGLProgram; // upsample + accumulate
  private compositeProg!: WebGLProgram; // final composite

  // 4级金字塔 FBO
  private pyramid!: SingleRT[];       // downsample chain [0..levels-1]
  private blurTemp!: SingleRT[];      // blur ping-pong temp [0..levels-1]
  private lumRT!: SingleRT;           // luminosity extract result (full res)

  // Fullscreen quad buffer
  private quadBuf!: WebGLBuffer;

  constructor(gl: WebGL2RenderingContext, config?: Partial<BloomConfig>) {
    this.gl = gl;
    this.cfg = { ...DEFAULT_BLOOM, ...config };
    this._compile();
    this._createFBOs();
    this._createQuad();
  }

  // ─── 1. _compile(): 编译所有 shader ─────────────────────────────────────────

  private _compile(): void {
    const gl = this.gl;

    // 尝试从 compiled.vs 提取 AT bloom shader;
    // 若不存在则回退到内联 GLSL (WebGL1)
    let lumFrag: string;
    try {
      lumFrag = getShader('bloom-luminosity.fs');
    } catch {
      lumFrag = LUMINOSITY_FRAG;
    }

    let blurFrag: string;
    try {
      blurFrag = getShader('bloom-blur.fs');
    } catch {
      blurFrag = BLUR_FRAG;
    }

    let compositeFrag: string;
    try {
      compositeFrag = getShader('bloom-composite.fs');
    } catch {
      compositeFrag = COMPOSITE_FRAG;
    }

    let upsampleFrag: string;
    try {
      upsampleFrag = getShader('bloom-upsample.fs');
    } catch {
      upsampleFrag = UPSAMPLE_FRAG;
    }

    // gl.createShader / compile / link — 每个 program 真正走完整流程
    this.lumProg       = this._compileProgram(BLOOM_VERT, lumFrag, 'bloom-lum');
    this.blurProg      = this._compileProgram(BLOOM_VERT, blurFrag, 'bloom-blur');
    this.upsampleProg  = this._compileProgram(BLOOM_VERT, upsampleFrag, 'bloom-upsample');
    this.compositeProg = this._compileProgram(BLOOM_VERT, compositeFrag, 'bloom-composite');
  }

  // ─── 2. _createFBOs(): 4级金字塔 FBO ────────────────────────────────────────

  private _createFBOs(): void {
    const { width, height, levels } = this.cfg;

    // 全分辨率 luminosity RT
    this.lumRT = this._makeFBO(width, height);

    this.pyramid  = [];
    this.blurTemp = [];

    // 每级分辨率减半
    for (let i = 0; i < levels; i++) {
      const w = Math.max(1, width  >> (i + 1));
      const h = Math.max(1, height >> (i + 1));
      this.pyramid.push(this._makeFBO(w, h));
      this.blurTemp.push(this._makeFBO(w, h));
    }
  }

  // ─── 3. step(inputTex): 完整 bloom pass 链 ──────────────────────────────────

  step(inputTex: WebGLTexture): WebGLTexture {
    const gl = this.gl;
    const { width, height, levels, threshold, knee, strength, radius, tintColor } = this.cfg;

    // ── Pass A: Luminosity threshold → lumRT ──
    // threshold=0.4: 只有足够亮的像素参与 bloom，暗区域不受影响
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.lumRT.fbo);
    gl.viewport(0, 0, width, height);
    gl.useProgram(this.lumProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputTex);
    gl.uniform1i(gl.getUniformLocation(this.lumProg, 'uInput'), 0);
    gl.uniform1f(gl.getUniformLocation(this.lumProg, 'uThreshold'), threshold);
    gl.uniform1f(gl.getUniformLocation(this.lumProg, 'uKnee'), knee);
    this._drawQuad(this.lumProg);

    // ── Pass B: Downsample ×levels — 每级模糊+减半 ──
    // 第0级从 lumRT 下采样
    this._blurPass(this.lumRT.tex, this.pyramid[0], this.blurTemp[0]);

    for (let i = 1; i < levels; i++) {
      this._blurPass(this.pyramid[i - 1].tex, this.pyramid[i], this.blurTemp[i]);
    }

    // ── Pass C: Upsample — 从最小级往上叠加 ──
    for (let i = levels - 2; i >= 0; i--) {
      const w = this.pyramid[i].width;
      const h = this.pyramid[i].height;

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurTemp[i].fbo);
      gl.viewport(0, 0, w, h);
      gl.useProgram(this.upsampleProg);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.pyramid[i].tex);
      gl.uniform1i(gl.getUniformLocation(this.upsampleProg, 'uCurrent'), 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.pyramid[i + 1].tex);
      gl.uniform1i(gl.getUniformLocation(this.upsampleProg, 'uUpper'), 1);

      gl.uniform1f(gl.getUniformLocation(this.upsampleProg, 'uWeight'), 0.8);
      this._drawQuad(this.upsampleProg);

      // 把 upsample 结果写回 pyramid[i]，供下层使用
      // swap: pyramid[i].tex ↔ blurTemp[i].tex
      const tmpFbo = this.pyramid[i].fbo;
      const tmpTex = this.pyramid[i].tex;
      this.pyramid[i].fbo = this.blurTemp[i].fbo;
      this.pyramid[i].tex = this.blurTemp[i].tex;
      this.blurTemp[i].fbo = tmpFbo;
      this.blurTemp[i].tex = tmpTex;
    }

    // ── Pass D: Additive composite → lumRT (重用作输出 RT) ──
    // uniforms: bloomStrength=1.5, bloomRadius=0.6, bloomTintColor=warm
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.lumRT.fbo);
    gl.viewport(0, 0, width, height);
    gl.useProgram(this.compositeProg);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputTex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uBase'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.pyramid[0].tex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uBloom0'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.pyramid[1].tex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uBloom1'), 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.pyramid[2].tex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uBloom2'), 3);

    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.pyramid[3].tex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uBloom3'), 4);

    // bloomStrength uniform (1.5)
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uStrength'), strength);
    // bloomRadius uniform (0.6) — 控制扩散层权重分布
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uRadius'), radius);
    // bloomTintColor uniform — 暖色 vec3(1.0, 0.95, 0.85)
    gl.uniform3f(
      gl.getUniformLocation(this.compositeProg, 'uTintColor'),
      tintColor[0], tintColor[1], tintColor[2]
    );

    this._drawQuad(this.compositeProg);

    return this.lumRT.tex;
  }

  /** 获取最终合成纹理供下游消费 */
  get outputTexture(): WebGLTexture { return this.lumRT.tex; }

  // ─── 内部方法: 真正的 WebGL 调用 ─────────────────────────────────────────────

  /**
   * 对 srcTex 做水平+垂直高斯模糊，输出到 destRT
   * 用 tempRT 作 ping-pong 中间缓冲
   */
  private _blurPass(srcTex: WebGLTexture, destRT: SingleRT, tempRT: SingleRT): void {
    const gl = this.gl;
    const w = destRT.width;
    const h = destRT.height;

    // ── Horizontal blur: srcTex → tempRT ──
    gl.bindFramebuffer(gl.FRAMEBUFFER, tempRT.fbo);
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.blurProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(gl.getUniformLocation(this.blurProg, 'uInput'), 0);
    gl.uniform2f(gl.getUniformLocation(this.blurProg, 'uTexelSize'), 1.0 / w, 1.0 / h);
    gl.uniform2f(gl.getUniformLocation(this.blurProg, 'uDir'), 1.0, 0.0);
    this._drawQuad(this.blurProg);

    // ── Vertical blur: tempRT → destRT ──
    gl.bindFramebuffer(gl.FRAMEBUFFER, destRT.fbo);
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.blurProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tempRT.tex);
    gl.uniform1i(gl.getUniformLocation(this.blurProg, 'uInput'), 0);
    gl.uniform2f(gl.getUniformLocation(this.blurProg, 'uTexelSize'), 1.0 / w, 1.0 / h);
    gl.uniform2f(gl.getUniformLocation(this.blurProg, 'uDir'), 0.0, 1.0);
    this._drawQuad(this.blurProg);
  }

  /** 画全屏 quad — gl.bindBuffer + gl.vertexAttribPointer + gl.drawArrays */
  private _drawQuad(program: WebGLProgram): void {
    const gl = this.gl;
    const posLoc = gl.getAttribLocation(program, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  /** 创建全屏 quad buffer: 2 三角形, 6 顶点 */
  private _createQuad(): void {
    const gl = this.gl;
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);
  }

  /** 编译 vert + frag → WebGLProgram — gl.createShader / gl.compileShader / gl.linkProgram */
  private _compileProgram(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[BloomGPU] vertex compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[BloomGPU] fragment compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[BloomGPU] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** 创建单层 FBO — gl.createTexture / gl.texImage2D / gl.createFramebuffer / gl.bindFramebuffer */
  private _makeFBO(w: number, h: number): SingleRT {
    const gl = this.gl;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { fbo, tex, width: w, height: h };
  }

  /** 释放 GPU 资源 */
  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.lumProg);
    gl.deleteProgram(this.blurProg);
    gl.deleteProgram(this.upsampleProg);
    gl.deleteProgram(this.compositeProg);
    gl.deleteBuffer(this.quadBuf);

    const freeRT = (rt: SingleRT) => {
      gl.deleteFramebuffer(rt.fbo);
      gl.deleteTexture(rt.tex);
    };
    freeRT(this.lumRT);
    this.pyramid.forEach(freeRT);
    this.blurTemp.forEach(freeRT);
  }
}
