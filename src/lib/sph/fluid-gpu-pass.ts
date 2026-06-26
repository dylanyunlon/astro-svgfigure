/**
 * fluid-gpu-pass.ts — 真正在 GPU 上跑的 Navier-Stokes 流体
 *
 * 这不是空壳。每个函数都调用 gl.*。
 * 用 NukePass + Nuke ping-pong FBO 做 Jacobi 压力求解。
 * 从 compiled.vs 通过 ShaderLoader 提取 AT 生产 shader 源码。
 *
 * Pass 链 (每帧):
 *   splat → curl → vorticity → divergence → pressure (×N jacobi) → gradientSub → advectVel → advectDye → display
 */




// ─── WebGL1 → WebGL2 适配 ────────────────────────────────────────────────────
// compiled.vs 里的 shader 是 WebGL1 (varying/texture2D)
// NukePass 用 WebGL2 (#version 300 es)
// 我们保持 WebGL1 context 以兼容 AT 原始 shader

// ─── AT 流体 shader 的共用 vertex shader ────────────────────────────────────

// 流体 pass 需要邻居像素 UV (vL/vR/vT/vB)，AT 用这个 vertex shader



import { getShader } from '../shaders/ShaderLoader';
import type { RenderTarget, UniformValue } from '../renderer/NukePass';

const FLUID_VERT = /* glsl */ `
#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;
out vec2 vL;
out vec2 vR;
out vec2 vT;
out vec2 vB;
uniform vec2 texelSize;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// 简单的全屏 quad vert (无邻居)
const SIMPLE_VERT = /* glsl */ `
#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─── FluidGPU: 真实 WebGL 流体 ────────────────────────────────────────────

interface FluidConfig {
  simWidth: number;      // 流体模拟分辨率 (128/256)
  simHeight: number;
  dyeWidth: number;      // 染料分辨率 (可以更高, 1024)
  dyeHeight: number;
  pressureIterations: number;  // Jacobi 迭代次数 (20-30)
  curl: number;                // 涡度强度
  splatRadius: number;
  dissipation: number;         // 速度衰减 (0.97-1.0)
  dyeDissipation: number;     // 染料衰减
}

const DEFAULT_CONFIG: FluidConfig = {
  simWidth: 256,
  simHeight: 256,
  dyeWidth: 1024,
  dyeHeight: 1024,
  pressureIterations: 25,
  curl: 30,
  splatRadius: 0.5,
  dissipation: 0.95,
  dyeDissipation: 0.95,
};

interface DoubleRT {
  read: WebGLFramebuffer;
  write: WebGLFramebuffer;
  readTex: WebGLTexture;
  writeTex: WebGLTexture;
  width: number;
  height: number;
}

export class FluidGPU {
  private gl: WebGL2RenderingContext;
  private config: FluidConfig;

  // WebGL programs — 真正 compiled 的 shader
  private splatProg!: WebGLProgram;
  private curlProg!: WebGLProgram;
  private vorticityProg!: WebGLProgram;
  private divergenceProg!: WebGLProgram;
  private pressureProg!: WebGLProgram;
  private gradSubProg!: WebGLProgram;
  private advectionProg!: WebGLProgram;
  private clearProg!: WebGLProgram;
  private displayProg!: WebGLProgram;

  // FBOs — 真正的 GPU 纹理
  private velocity!: DoubleRT;
  private pressure!: DoubleRT;
  private divergenceRT!: { fbo: WebGLFramebuffer; tex: WebGLTexture };
  private curlRT!: { fbo: WebGLFramebuffer; tex: WebGLTexture };
  private dye!: DoubleRT;

  // Fullscreen quad VAO
  private quadBuf!: WebGLBuffer;

  constructor(gl: WebGL2RenderingContext, config?: Partial<FluidConfig>) {
    this.gl = gl;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this._init();
  }

  /** 初始化: 编译 shader + 创建 FBO + 创建 quad geometry */
  private _init(): void {
    const gl = this.gl;

    // ── 从 compiled.vs 提取 AT 原始 shader 源码 ──
    const splatSrc     = getShader('splatShader.fs');
    const curlSrc      = getShader('curlShader.fs');
    const vorticitySrc = getShader('vorticityShader.fs');
    const divergeSrc   = getShader('divergenceShader.fs');
    const pressureSrc  = getShader('pressureShader.fs');
    const gradSubSrc   = getShader('gradientSubtractShader.fs');
    const advectSrc    = getShader('advectionShader.fs');
    const clearSrc     = getShader('clearShader.fs');
    const displaySrc   = getShader('displayShader.fs');

    // ── 编译 shader → WebGLProgram (真正的 gl 调用) ──
    this.splatProg      = this._compile(SIMPLE_VERT, splatSrc, 'splat');
    this.curlProg       = this._compile(FLUID_VERT, curlSrc, 'curl');
    this.vorticityProg  = this._compile(FLUID_VERT, vorticitySrc, 'vorticity');
    this.divergenceProg = this._compile(FLUID_VERT, divergeSrc, 'divergence');
    this.pressureProg   = this._compile(FLUID_VERT, pressureSrc, 'pressure');
    this.gradSubProg    = this._compile(FLUID_VERT, gradSubSrc, 'gradientSub');
    this.advectionProg  = this._compile(SIMPLE_VERT, advectSrc, 'advection');
    this.clearProg      = this._compile(SIMPLE_VERT, clearSrc, 'clear');
    this.displayProg    = this._compile(SIMPLE_VERT, displaySrc, 'display');

    // ── 创建 FBO (真正的 GPU 纹理) ──
    // WebGL2 有 RG16F/R16F/RGBA16F; WebGL1 需要扩展 + fallback to RGBA+FLOAT
    const isWebGL2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;
    const halfFloatExt = !isWebGL2 ? gl.getExtension('OES_texture_half_float') : null;
    const halfFloat = isWebGL2 ? gl.HALF_FLOAT : (halfFloatExt ? halfFloatExt.HALF_FLOAT_OES : gl.FLOAT);

    // For velocity (2-channel): WebGL2 uses RG16F/RG, WebGL1 falls back to RGBA
    const velInternal = isWebGL2 ? (gl as WebGL2RenderingContext).RG16F    : gl.RGBA;
    const velFormat   = isWebGL2 ? (gl as WebGL2RenderingContext).RG       : gl.RGBA;
    // For pressure/divergence/curl (1-channel): WebGL2 uses R16F/RED, WebGL1 falls back to RGBA
    const scalarInternal = isWebGL2 ? (gl as WebGL2RenderingContext).R16F  : gl.RGBA;
    const scalarFormat   = isWebGL2 ? (gl as WebGL2RenderingContext).RED   : gl.RGBA;
    // For dye (4-channel): WebGL2 uses RGBA16F, WebGL1 uses RGBA
    const dyeInternal = isWebGL2 ? (gl as WebGL2RenderingContext).RGBA16F : gl.RGBA;

    const { simWidth: sw, simHeight: sh, dyeWidth: dw, dyeHeight: dh } = this.config;
    this.velocity    = this._createDoubleFBO(sw, sh, velInternal, velFormat, halfFloat);
    this.pressure    = this._createDoubleFBO(sw, sh, scalarInternal, scalarFormat, halfFloat);
    this.divergenceRT = this._createSingleFBO(sw, sh, scalarInternal, scalarFormat, halfFloat);
    this.curlRT       = this._createSingleFBO(sw, sh, scalarInternal, scalarFormat, halfFloat);
    this.dye          = this._createDoubleFBO(dw, dh, dyeInternal, gl.RGBA, halfFloat);

    // ── 全屏 quad (2 个三角形) ──
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 1, -1, -1, 1,
      -1,  1, 1, -1,  1, 1,
    ]), gl.STATIC_DRAW);
  }

  // 自动背景 splat 的内部时间计数器
  private _autoSplatTimer: number = 0;
  private _autoSplatPoints: Array<{x:number,y:number,vx:number,vy:number,color:[number,number,number]}> = [];

  /** 初始化随机背景 splat 漂移点 (首次调用时) */
  private _initAutoSplats(): void {
    if (this._autoSplatPoints.length > 0) return;
    const palette: Array<[number,number,number]> = [
      [0.9, 0.1, 0.6],
      [0.1, 0.7, 0.9],
      [0.95, 0.5, 0.0],
      [0.2, 0.9, 0.3],
      [0.6, 0.0, 1.0],
    ];
    for (let i = 0; i < 5; i++) {
      this._autoSplatPoints.push({
        x: Math.random(),
        y: Math.random(),
        vx: (Math.random() - 0.5) * 0.004,
        vy: (Math.random() - 0.5) * 0.004,
        color: palette[i % palette.length],
      });
    }
  }

  /** 每帧调用 — 跑完整 Navier-Stokes pass 链 */
  step(mouseX: number, mouseY: number,
       prevMouseX: number, prevMouseY: number,
       dt: number = 1/60): void {
    const gl = this.gl;
    const { simWidth: sw, simHeight: sh, dyeWidth: dw, dyeHeight: dh } = this.config;

    // 0. 自动背景 splat — 每帧 2-3 个随机漂移点注入流体
    this._initAutoSplats();
    this._autoSplatTimer += dt;
    const numAuto = 2 + (Math.random() < 0.5 ? 1 : 0); // 2 or 3
    for (let i = 0; i < numAuto; i++) {
      const p = this._autoSplatPoints[i % this._autoSplatPoints.length];
      // 缓慢漂移 + 随机扰动
      p.x += p.vx + (Math.random() - 0.5) * 0.002;
      p.y += p.vy + (Math.random() - 0.5) * 0.002;
      // 边界反弹
      if (p.x < 0.05 || p.x > 0.95) { p.vx *= -1; p.x = Math.max(0.05, Math.min(0.95, p.x)); }
      if (p.y < 0.05 || p.y > 0.95) { p.vy *= -1; p.y = Math.max(0.05, Math.min(0.95, p.y)); }
      const force = 8000 * dt;
      const angle = this._autoSplatTimer * 1.3 + i * 2.1;
      this._splat(p.x, p.y, Math.cos(angle) * force, Math.sin(angle) * force, p.color);
    }

    // 1. Splat — 鼠标注入速度+染料 (force = 8000/s)
    const dx = mouseX - prevMouseX;
    const dy = mouseY - prevMouseY;
    if (dx !== 0 || dy !== 0) {
      const force = 8000;
      this._splat(mouseX, mouseY, dx * force, dy * force, [0.8, 0.2, 0.05]);
    }

    // 2. Curl — 计算涡度
    this._runPass(this.curlProg, this.curlRT.fbo, sw, sh, {
      uVelocity: this.velocity.readTex,
      texelSize: [1.0 / sw, 1.0 / sh],
    });

    // 3. Vorticity — 涡度约束力
    this._runPass(this.vorticityProg, this.velocity.write, sw, sh, {
      uVelocity: this.velocity.readTex,
      uCurl: this.curlRT.tex,
      curl: this.config.curl,
      dt,
      texelSize: [1.0 / sw, 1.0 / sh],
    });
    this._swapVelocity();

    // 4. Divergence — 散度场
    this._runPass(this.divergenceProg, this.divergenceRT.fbo, sw, sh, {
      uVelocity: this.velocity.readTex,
      texelSize: [1.0 / sw, 1.0 / sh],
    });

    // 5. Clear pressure
    this._runPass(this.clearProg, this.pressure.write, sw, sh, {
      uTexture: this.pressure.readTex,
      value: 0.8,
    });
    this._swapPressure();

    // 6. Pressure solve — Jacobi 迭代 (真正的 GPU 循环)
    for (let i = 0; i < this.config.pressureIterations; i++) {
      this._runPass(this.pressureProg, this.pressure.write, sw, sh, {
        uPressure: this.pressure.readTex,
        uDivergence: this.divergenceRT.tex,
        texelSize: [1.0 / sw, 1.0 / sh],
      });
      this._swapPressure();
    }

    // 7. Gradient subtract — 从速度场减去压力梯度 → 无散速度
    this._runPass(this.gradSubProg, this.velocity.write, sw, sh, {
      uPressure: this.pressure.readTex,
      uVelocity: this.velocity.readTex,
      texelSize: [1.0 / sw, 1.0 / sh],
    });
    this._swapVelocity();

    // 8. Advect velocity — 对流速度场
    this._runPass(this.advectionProg, this.velocity.write, sw, sh, {
      uVelocity: this.velocity.readTex,
      uSource: this.velocity.readTex,
      texelSize: [1.0 / sw, 1.0 / sh],
      dt,
      dissipation: this.config.dissipation,
    });
    this._swapVelocity();

    // 9. Advect dye — 对流染料
    this._runPass(this.advectionProg, this.dye.write, dw, dh, {
      uVelocity: this.velocity.readTex,
      uSource: this.dye.readTex,
      texelSize: [1.0 / sw, 1.0 / sh],
      dt,
      dissipation: this.config.dyeDissipation,
    });
    this._swapDye();
  }

  /** 把流体结果渲染到屏幕 */
  display(canvasWidth: number, canvasHeight: number): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasWidth, canvasHeight);
    gl.useProgram(this.displayProg);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.dye.readTex);
    gl.uniform1i(gl.getUniformLocation(this.displayProg, 'uTexture'), 0);

    this._drawQuad(this.displayProg);
  }

  /** 获取速度场纹理 — 供下游消费 (cell distortion, particle acceleration) */
  get velocityTexture(): WebGLTexture { return this.velocity.readTex; }
  get dyeTexture(): WebGLTexture { return this.dye.readTex; }

  // ─── 内部方法: 真正的 WebGL 调用 ──────────────────────────────

  private _splat(x: number, y: number, dx: number, dy: number, color: number[]): void {
    const gl = this.gl;
    const { simWidth: sw, simHeight: sh, dyeWidth: dw, dyeHeight: dh } = this.config;

    // Splat velocity
    gl.useProgram(this.splatProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.velocity.write);
    gl.viewport(0, 0, sw, sh);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.readTex);
    gl.uniform1i(gl.getUniformLocation(this.splatProg, 'uTarget'), 0);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'aspectRatio'), sw / sh);
    gl.uniform2f(gl.getUniformLocation(this.splatProg, 'point'), x, y);
    gl.uniform2f(gl.getUniformLocation(this.splatProg, 'prevPoint'), x - dx * 0.01, y - dy * 0.01);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'radius'), this.config.splatRadius / 100);
    gl.uniform3f(gl.getUniformLocation(this.splatProg, 'color'), dx, dy, 0);
    gl.uniform3f(gl.getUniformLocation(this.splatProg, 'bgColor'), 0, 0, 0);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'canRender'), 1);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'uAdd'), 1);
    this._drawQuad(this.splatProg);
    this._swapVelocity();

    // Splat dye
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.dye.write);
    gl.viewport(0, 0, dw, dh);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.dye.readTex);
    gl.uniform1i(gl.getUniformLocation(this.splatProg, 'uTarget'), 0);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'aspectRatio'), dw / dh);
    gl.uniform3f(gl.getUniformLocation(this.splatProg, 'color'), color[0], color[1], color[2]);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'uAdd'), 0);
    this._drawQuad(this.splatProg);
    this._swapDye();
  }

  /** 跑单个 shader pass 到指定 FBO */
  private _runPass(program: WebGLProgram, targetFBO: WebGLFramebuffer,
                   w: number, h: number, uniforms: Record<string, any>): void {
    const gl = this.gl;
    gl.useProgram(program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO);
    gl.viewport(0, 0, w, h);

    let texUnit = 0;
    for (const [name, val] of Object.entries(uniforms)) {
      const loc = gl.getUniformLocation(program, name);
      if (loc === null) continue;

      if (val instanceof WebGLTexture) {
        gl.activeTexture(gl.TEXTURE0 + texUnit);
        gl.bindTexture(gl.TEXTURE_2D, val);
        gl.uniform1i(loc, texUnit);
        texUnit++;
      } else if (typeof val === 'number') {
        gl.uniform1f(loc, val);
      } else if (Array.isArray(val) && val.length === 2) {
        gl.uniform2f(loc, val[0], val[1]);
      } else if (Array.isArray(val) && val.length === 3) {
        gl.uniform3f(loc, val[0], val[1], val[2]);
      }
    }

    this._drawQuad(program);
  }

  /** 画全屏 quad */
  private _drawQuad(program: WebGLProgram): void {
    const gl = this.gl;
    const posLoc = gl.getAttribLocation(program, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  /** 编译 vert + frag → WebGLProgram */
  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[FluidGPU] vertex compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, 'precision highp float;\n' + frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[FluidGPU] fragment compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[FluidGPU] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** 创建双缓冲 FBO (ping-pong) — read 和 write 各一个 FBO+tex 对 */
  private _createDoubleFBO(w: number, h: number,
    internalFormat: number, format: number, type: number): DoubleRT {
    const readRT  = this._createSingleFBO(w, h, internalFormat, format, type);
    const writeRT = this._createSingleFBO(w, h, internalFormat, format, type);
    return {
      read:     readRT.fbo,
      write:    writeRT.fbo,
      readTex:  readRT.tex,
      writeTex: writeRT.tex,
      width: w, height: h,
    };
  }

  private _createSingleFBO(w: number, h: number,
    internalFormat: number, format: number, type: number): { fbo: WebGLFramebuffer; tex: WebGLTexture } {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fbo, tex };
  }

  private _swapVelocity(): void {
    [this.velocity.read, this.velocity.write] = [this.velocity.write, this.velocity.read];
    [this.velocity.readTex, this.velocity.writeTex] = [this.velocity.writeTex, this.velocity.readTex];
  }
  private _swapPressure(): void {
    [this.pressure.read, this.pressure.write] = [this.pressure.write, this.pressure.read];
    [this.pressure.readTex, this.pressure.writeTex] = [this.pressure.writeTex, this.pressure.readTex];
  }
  private _swapDye(): void {
    [this.dye.read, this.dye.write] = [this.dye.write, this.dye.read];
    [this.dye.readTex, this.dye.writeTex] = [this.dye.writeTex, this.dye.readTex];
  }
}
