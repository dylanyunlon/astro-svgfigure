/**
 * fluid-fbo.ts — Navier-Stokes 2D fluid simulation via GPU passes
 *
 * AT Fluid 模块 (95 次引用) 移植:
 *   advection → divergence → pressure (Jacobi) → gradient subtract → curl → splat
 *
 * 架构:
 *   FluidFBO        双缓冲 FBO，velocity(RG) + pressure(B) + density(A)
 *   AdvectionPass   半拉格朗日粒子追踪
 *   DivergencePass  散度计算 ∇·u
 *   PressurePass    Jacobi 迭代求解压力泊松方程 ∇²p = ∇·u
 *   GradSubtractPass 速度修正 u -= ∇p，保证无散
 *   CurlPass        涡度计算 ω = ∇×u，用于涡度约束
 *   SplatPass       鼠标注入速度 + 密度
 *   MouseFluid      封装鼠标/触摸事件 → splat
 *
 * 分辨率: viewport 1/4 (性能)，双线性上采样输出至 canvas.
 */

import { RenderTarget, createProgram, FullscreenQuad } from './hydra-gl-layer';

// ── GLSL 公共头 ──────────────────────────────────────────────────────────────

const VERT = `#version 300 es
precision highp float;
out vec2 vUV;
void main() {
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  vUV = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  gl_Position = vec4(x, y, 0.0, 1.0);
}`;

// ── Advection fragment shader ────────────────────────────────────────────────
// 半拉格朗日法: 沿速度场反向追踪，双线性采样，带 dissipation 衰减
const ADVECTION_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uVelocity;   // RG = velocity
uniform sampler2D uSource;     // 被 advect 的场 (velocity 或 density)
uniform vec2      uTexelSize;  // 1/simRes
uniform float     uDt;
uniform float     uDissipation;

void main() {
  vec2 vel = texture(uVelocity, vUV).rg;
  // 反向追踪: 从当前格找到上一时刻位置
  vec2 prevUV = vUV - vel * uDt * uTexelSize;
  // 边界钳制
  prevUV = clamp(prevUV, uTexelSize * 0.5, 1.0 - uTexelSize * 0.5);
  fragColor = uDissipation * texture(uSource, prevUV);
}`;

// ── Divergence fragment shader ───────────────────────────────────────────────
// ∇·u = (u[i+1] - u[i-1]) / 2dx + (v[j+1] - v[j-1]) / 2dy
const DIVERGENCE_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uVelocity;
uniform vec2      uTexelSize;

void main() {
  float L = texture(uVelocity, vUV - vec2(uTexelSize.x, 0.0)).r;
  float R = texture(uVelocity, vUV + vec2(uTexelSize.x, 0.0)).r;
  float B = texture(uVelocity, vUV - vec2(0.0, uTexelSize.y)).g;
  float T = texture(uVelocity, vUV + vec2(0.0, uTexelSize.y)).g;
  float div = 0.5 * ((R - L) + (T - B));
  fragColor = vec4(div, 0.0, 0.0, 1.0);
}`;

// ── Pressure (Jacobi iteration) fragment shader ──────────────────────────────
// ∇²p = ∇·u  →  p[i,j] = (p[i+1] + p[i-1] + p[j+1] + p[j-1] - div) / 4
const PRESSURE_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uPressure;
uniform sampler2D uDivergence;
uniform vec2      uTexelSize;

void main() {
  float L   = texture(uPressure,   vUV - vec2(uTexelSize.x, 0.0)).r;
  float R   = texture(uPressure,   vUV + vec2(uTexelSize.x, 0.0)).r;
  float B   = texture(uPressure,   vUV - vec2(0.0, uTexelSize.y)).r;
  float T   = texture(uPressure,   vUV + vec2(0.0, uTexelSize.y)).r;
  float div = texture(uDivergence, vUV).r;
  float p   = (L + R + B + T - div) * 0.25;
  fragColor = vec4(p, 0.0, 0.0, 1.0);
}`;

// ── Gradient subtract fragment shader ────────────────────────────────────────
// u -= ∇p  →  无散速度场
const GRAD_SUBTRACT_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uVelocity;
uniform sampler2D uPressure;
uniform vec2      uTexelSize;

void main() {
  float pL = texture(uPressure, vUV - vec2(uTexelSize.x, 0.0)).r;
  float pR = texture(uPressure, vUV + vec2(uTexelSize.x, 0.0)).r;
  float pB = texture(uPressure, vUV - vec2(0.0, uTexelSize.y)).r;
  float pT = texture(uPressure, vUV + vec2(0.0, uTexelSize.y)).r;
  vec2  vel = texture(uVelocity, vUV).rg;
  vel -= 0.5 * vec2(pR - pL, pT - pB);
  fragColor = vec4(vel, 0.0, 1.0);
}`;

// ── Curl fragment shader ──────────────────────────────────────────────────────
// ω = ∂v/∂x - ∂u/∂y  (scalar curl for 2D)
const CURL_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uVelocity;
uniform vec2      uTexelSize;

void main() {
  float L = texture(uVelocity, vUV - vec2(uTexelSize.x, 0.0)).g;
  float R = texture(uVelocity, vUV + vec2(uTexelSize.x, 0.0)).g;
  float B = texture(uVelocity, vUV - vec2(0.0, uTexelSize.y)).r;
  float T = texture(uVelocity, vUV + vec2(0.0, uTexelSize.y)).r;
  float curl = 0.5 * ((R - L) - (T - B));
  fragColor = vec4(curl, 0.0, 0.0, 1.0);
}`;

// ── Vorticity confinement fragment shader ────────────────────────────────────
// 将涡度力注回速度场，增强细节
const VORTICITY_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform vec2      uTexelSize;
uniform float     uCurlStrength;
uniform float     uDt;

void main() {
  float L = texture(uCurl, vUV - vec2(uTexelSize.x, 0.0)).r;
  float R = texture(uCurl, vUV + vec2(uTexelSize.x, 0.0)).r;
  float B = texture(uCurl, vUV - vec2(0.0, uTexelSize.y)).r;
  float T = texture(uCurl, vUV + vec2(0.0, uTexelSize.y)).r;
  float C = texture(uCurl, vUV).r;

  // 涡度梯度方向 (η)
  vec2 force = vec2(abs(T) - abs(B), abs(R) - abs(L));
  float lenSq = dot(force, force);
  force = lenSq > 0.0 ? force * inversesqrt(lenSq) : vec2(0.0);
  // 叉乘 ω̂ × η 得到径向力
  force = vec2(force.y, -force.x) * C * uCurlStrength;

  vec2 vel = texture(uVelocity, vUV).rg + force * uDt;
  fragColor = vec4(vel, 0.0, 1.0);
}`;

// ── Splat fragment shader ─────────────────────────────────────────────────────
// 高斯注入: 在 uPoint 处注入速度 uSplatVel 和密度 uSplatColor
const SPLAT_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uTarget;   // 注入目标 (velocity 或 density)
uniform vec2      uPoint;    // 注入位置 [0,1]^2
uniform vec3      uSplatVel; // xy = 速度增量, z = 密度增量
uniform float     uRadius;   // 高斯半径 (归一化)
uniform float     uAspect;   // canvas width/height

void main() {
  vec2  p   = vUV - uPoint;
  p.x *= uAspect;
  float mag = exp(-dot(p, p) / (uRadius * uRadius));
  vec4  base = texture(uTarget, vUV);
  // velocity pass: add xy; density pass: add z
  fragColor = base + vec4(uSplatVel.xy, 0.0, uSplatVel.z) * mag;
}`;

// ── Display / blit fragment shader ───────────────────────────────────────────
// 将 density(alpha) 场以指定色调渲染到 canvas
const DISPLAY_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uDensity;
uniform vec3      uTint;    // base colour of the fluid

void main() {
  float d = texture(uDensity, vUV).a;
  // 以密度做发光叠加
  vec3 col = uTint * d;
  fragColor = vec4(col, d * 0.85);
}`;

// ── DoubleFBO: 双缓冲帮助类 ──────────────────────────────────────────────────

class DoubleFBO {
  read:  RenderTarget;
  write: RenderTarget;

  constructor(
    private gl: WebGL2RenderingContext,
    width: number,
    height: number,
    hdr = true,
  ) {
    this.read  = new RenderTarget(gl, { width, height, hdr });
    this.write = new RenderTarget(gl, { width, height, hdr });
  }

  swap(): void {
    [this.read, this.write] = [this.write, this.read];
  }

  destroy(): void {
    this.read.destroy();
    this.write.destroy();
  }
}

// ── FluidConfig ──────────────────────────────────────────────────────────────

export interface FluidConfig {
  /** 模拟分辨率相对于 viewport 的缩放 (default 0.25) */
  simScale?: number;
  /** Jacobi 迭代次数 (default 30) */
  pressureIterations?: number;
  /** 速度衰减 (default 0.98) */
  velocityDissipation?: number;
  /** 密度衰减 (default 0.97) */
  densityDissipation?: number;
  /** 涡度约束强度 (default 20) */
  curlStrength?: number;
  /** splat 高斯半径，归一化坐标 (default 0.25) */
  splatRadius?: number;
  /** 流体颜色 [r,g,b] 0-1 (default [0.2, 0.6, 1.0] 蓝色) */
  tint?: [number, number, number];
}

// ── FluidFBO ─────────────────────────────────────────────────────────────────

export class FluidFBO {
  private gl: WebGL2RenderingContext;
  private quad: FullscreenQuad;

  private simW: number;
  private simH: number;

  // 双缓冲 FBO
  private velocity:  DoubleFBO;
  private density:   DoubleFBO;
  private pressure:  DoubleFBO;
  private divergence: RenderTarget;
  private curl:       RenderTarget;

  // 各 pass program
  private progAdvect:      WebGLProgram;
  private progDivergence:  WebGLProgram;
  private progPressure:    WebGLProgram;
  private progGradSub:     WebGLProgram;
  private progCurl:        WebGLProgram;
  private progVorticity:   WebGLProgram;
  private progSplat:       WebGLProgram;
  private progDisplay:     WebGLProgram;

  // 配置
  private cfg: Required<FluidConfig>;

  constructor(gl: WebGL2RenderingContext, cfg: FluidConfig = {}) {
    this.gl = gl;
    this.cfg = {
      simScale:              cfg.simScale              ?? 0.25,
      pressureIterations:    cfg.pressureIterations    ?? 30,
      velocityDissipation:   cfg.velocityDissipation   ?? 0.98,
      densityDissipation:    cfg.densityDissipation    ?? 0.97,
      curlStrength:          cfg.curlStrength          ?? 20.0,
      splatRadius:           cfg.splatRadius           ?? 0.25,
      tint:                  cfg.tint                  ?? [0.2, 0.6, 1.0],
    };

    const cw = (gl.canvas as HTMLCanvasElement).width;
    const ch = (gl.canvas as HTMLCanvasElement).height;
    this.simW = Math.max(1, Math.floor(cw * this.cfg.simScale));
    this.simH = Math.max(1, Math.floor(ch * this.cfg.simScale));

    this.quad = new FullscreenQuad(gl);

    // 分配 FBO (HDR RGBA16F)
    this.velocity   = new DoubleFBO(gl, this.simW, this.simH, true);
    this.density    = new DoubleFBO(gl, this.simW, this.simH, true);
    this.pressure   = new DoubleFBO(gl, this.simW, this.simH, true);
    this.divergence = new RenderTarget(gl, { width: this.simW, height: this.simH, hdr: true });
    this.curl       = new RenderTarget(gl, { width: this.simW, height: this.simH, hdr: true });

    // 编译所有 shader program
    this.progAdvect     = createProgram(gl, VERT, ADVECTION_FRAG);
    this.progDivergence = createProgram(gl, VERT, DIVERGENCE_FRAG);
    this.progPressure   = createProgram(gl, VERT, PRESSURE_FRAG);
    this.progGradSub    = createProgram(gl, VERT, GRAD_SUBTRACT_FRAG);
    this.progCurl       = createProgram(gl, VERT, CURL_FRAG);
    this.progVorticity  = createProgram(gl, VERT, VORTICITY_FRAG);
    this.progSplat      = createProgram(gl, VERT, SPLAT_FRAG);
    this.progDisplay    = createProgram(gl, VERT, DISPLAY_FRAG);
  }

  // ── 内部 helpers ────────────────────────────────────────────────────────

  private get texelSize(): [number, number] {
    return [1.0 / this.simW, 1.0 / this.simH];
  }

  private u1f(prog: WebGLProgram, name: string, v: number): void {
    this.gl.uniform1f(this.gl.getUniformLocation(prog, name), v);
  }

  private u2f(prog: WebGLProgram, name: string, x: number, y: number): void {
    this.gl.uniform2f(this.gl.getUniformLocation(prog, name), x, y);
  }

  private u3f(prog: WebGLProgram, name: string, x: number, y: number, z: number): void {
    this.gl.uniform3f(this.gl.getUniformLocation(prog, name), x, y, z);
  }

  private u1i(prog: WebGLProgram, name: string, v: number): void {
    this.gl.uniform1i(this.gl.getUniformLocation(prog, name), v);
  }

  private bindTex(tex: WebGLTexture, unit: number): void {
    this.gl.activeTexture(this.gl.TEXTURE0 + unit);
    this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
  }

  private drawToRT(rt: RenderTarget): void {
    rt.bind();
    this.quad.draw();
    rt.unbind();
  }

  // ── advection pass ──────────────────────────────────────────────────────

  private advect(source: DoubleFBO, velocity: DoubleFBO, dissipation: number, dt: number): void {
    const gl = this.gl;
    const prog = this.progAdvect;
    gl.useProgram(prog);

    this.bindTex(velocity.read.texture, 0);
    this.u1i(prog, 'uVelocity', 0);

    this.bindTex(source.read.texture, 1);
    this.u1i(prog, 'uSource', 1);

    this.u2f(prog, 'uTexelSize', ...this.texelSize);
    this.u1f(prog, 'uDt', dt);
    this.u1f(prog, 'uDissipation', dissipation);

    this.drawToRT(source.write);
    source.swap();
  }

  // ── divergence pass ─────────────────────────────────────────────────────

  private computeDivergence(): void {
    const gl = this.gl;
    const prog = this.progDivergence;
    gl.useProgram(prog);

    this.bindTex(this.velocity.read.texture, 0);
    this.u1i(prog, 'uVelocity', 0);
    this.u2f(prog, 'uTexelSize', ...this.texelSize);

    this.drawToRT(this.divergence);
  }

  // ── pressure (Jacobi) pass ──────────────────────────────────────────────

  private solvePressure(): void {
    const gl = this.gl;
    const prog = this.progPressure;
    gl.useProgram(prog);

    this.u2f(prog, 'uTexelSize', ...this.texelSize);
    this.bindTex(this.divergence.texture, 1);
    this.u1i(prog, 'uDivergence', 1);

    for (let i = 0; i < this.cfg.pressureIterations; i++) {
      this.bindTex(this.pressure.read.texture, 0);
      this.u1i(prog, 'uPressure', 0);
      this.drawToRT(this.pressure.write);
      this.pressure.swap();
    }
  }

  // ── gradient subtract pass ──────────────────────────────────────────────

  private gradientSubtract(): void {
    const gl = this.gl;
    const prog = this.progGradSub;
    gl.useProgram(prog);

    this.bindTex(this.velocity.read.texture, 0);
    this.u1i(prog, 'uVelocity', 0);
    this.bindTex(this.pressure.read.texture, 1);
    this.u1i(prog, 'uPressure', 1);
    this.u2f(prog, 'uTexelSize', ...this.texelSize);

    this.drawToRT(this.velocity.write);
    this.velocity.swap();
  }

  // ── curl + vorticity confinement pass ───────────────────────────────────

  private applyVorticity(dt: number): void {
    const gl = this.gl;

    // Step 1: compute curl
    gl.useProgram(this.progCurl);
    this.bindTex(this.velocity.read.texture, 0);
    this.u1i(this.progCurl, 'uVelocity', 0);
    this.u2f(this.progCurl, 'uTexelSize', ...this.texelSize);
    this.drawToRT(this.curl);

    // Step 2: vorticity confinement (inject curl force back)
    gl.useProgram(this.progVorticity);
    this.bindTex(this.velocity.read.texture, 0);
    this.u1i(this.progVorticity, 'uVelocity', 0);
    this.bindTex(this.curl.texture, 1);
    this.u1i(this.progVorticity, 'uCurl', 1);
    this.u2f(this.progVorticity, 'uTexelSize', ...this.texelSize);
    this.u1f(this.progVorticity, 'uCurlStrength', this.cfg.curlStrength);
    this.u1f(this.progVorticity, 'uDt', dt);

    this.drawToRT(this.velocity.write);
    this.velocity.swap();
  }

  // ── public: splat velocity + density ────────────────────────────────────

  /**
   * 在归一化位置 (x,y) 注入速度增量 (dx,dy) 和密度增量 density.
   * 典型调用: 鼠标移动时每帧调用 1-2 次.
   */
  splat(x: number, y: number, dx: number, dy: number, density = 1.0): void {
    const gl = this.gl;
    const prog = this.progSplat;
    gl.useProgram(prog);

    const aspect = this.simW / this.simH;

    // 注入速度场
    this.bindTex(this.velocity.read.texture, 0);
    this.u1i(prog, 'uTarget', 0);
    this.u2f(prog, 'uPoint', x, y);
    this.u3f(prog, 'uSplatVel', dx, dy, 0.0);
    this.u1f(prog, 'uRadius', this.cfg.splatRadius);
    this.u1f(prog, 'uAspect', aspect);
    this.drawToRT(this.velocity.write);
    this.velocity.swap();

    // 注入密度场
    this.bindTex(this.density.read.texture, 0);
    this.u3f(prog, 'uSplatVel', 0.0, 0.0, density);
    this.drawToRT(this.density.write);
    this.density.swap();
  }

  // ── public: step (call each frame) ──────────────────────────────────────

  /**
   * 推进一个时间步:
   *   advect velocity → advect density → divergence → pressure → grad-sub
   *   → curl/vorticity
   *
   * dt: 帧间隔秒数 (典型 1/60 = 0.0167)
   */
  step(dt: number): void {
    const gl = this.gl;

    // 关闭混合，全写
    gl.disable(gl.BLEND);

    // 1. Advect velocity along itself
    this.advect(this.velocity, this.velocity, this.cfg.velocityDissipation, dt);

    // 2. Advect density along velocity
    this.advect(this.density, this.velocity, this.cfg.densityDissipation, dt);

    // 3. Vorticity confinement (before pressure, while curl still coherent)
    if (this.cfg.curlStrength > 0.0) {
      this.applyVorticity(dt);
    }

    // 4. Divergence
    this.computeDivergence();

    // 5. Pressure Jacobi solve
    this.solvePressure();

    // 6. Gradient subtract → divergence-free velocity
    this.gradientSubtract();
  }

  // ── public: render density to canvas (or any output RT) ─────────────────

  /**
   * 将密度场渲染到屏幕.
   * 通常在 step() 之后、PixiJS 之前 (作为背景层) 或之后 (叠加).
   * output = null → canvas; 否则写到指定 RenderTarget.
   */
  render(output: RenderTarget | null = null): void {
    const gl = this.gl;
    const prog = this.progDisplay;
    gl.useProgram(prog);

    this.bindTex(this.density.read.texture, 0);
    this.u1i(prog, 'uDensity', 0);
    this.u3f(prog, 'uTint', ...this.cfg.tint);

    // 启用 alpha blend 叠加
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    if (output) {
      output.bind();
      this.quad.draw();
      output.unbind();
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, (gl.canvas as HTMLCanvasElement).width, (gl.canvas as HTMLCanvasElement).height);
      this.quad.draw();
    }

    gl.disable(gl.BLEND);
  }

  /** 释放所有 WebGL 资源 */
  destroy(): void {
    const gl = this.gl;
    this.velocity.destroy();
    this.density.destroy();
    this.pressure.destroy();
    this.divergence.destroy();
    this.curl.destroy();
    this.quad.destroy();
    for (const prog of [
      this.progAdvect, this.progDivergence, this.progPressure,
      this.progGradSub, this.progCurl, this.progVorticity,
      this.progSplat, this.progDisplay,
    ]) {
      gl.deleteProgram(prog);
    }
  }
}

// ── MouseFluid ───────────────────────────────────────────────────────────────
// 封装鼠标/触摸事件 → FluidFBO.splat()
// 应用场景: cell 间力场可视化 — 鼠标拖动产生流体扰动, cell 移动产生尾流

export interface MouseFluidOptions {
  /** 速度缩放因子 (default 8.0) */
  velocityScale?: number;
  /** 注入密度强度 (default 1.0) */
  densityStrength?: number;
}

export class MouseFluid {
  private fluid: FluidFBO;
  private canvas: HTMLCanvasElement;

  private mouseX = 0.0;
  private mouseY = 0.0;
  private lastX  = 0.0;
  private lastY  = 0.0;
  private pressing = false;

  private opts: Required<MouseFluidOptions>;
  private _onMouseDown:  (e: MouseEvent) => void;
  private _onMouseMove:  (e: MouseEvent) => void;
  private _onMouseUp:    (e: MouseEvent) => void;
  private _onTouchStart: (e: TouchEvent) => void;
  private _onTouchMove:  (e: TouchEvent) => void;
  private _onTouchEnd:   (e: TouchEvent) => void;

  constructor(canvas: HTMLCanvasElement, fluid: FluidFBO, opts: MouseFluidOptions = {}) {
    this.canvas = canvas;
    this.fluid  = fluid;
    this.opts = {
      velocityScale:   opts.velocityScale   ?? 8.0,
      densityStrength: opts.densityStrength ?? 1.0,
    };

    // 绑定 handlers (保存引用以便 destroy() 时移除)
    this._onMouseDown  = this.onMouseDown.bind(this);
    this._onMouseMove  = this.onMouseMove.bind(this);
    this._onMouseUp    = this.onMouseUp.bind(this);
    this._onTouchStart = this.onTouchStart.bind(this);
    this._onTouchMove  = this.onTouchMove.bind(this);
    this._onTouchEnd   = this.onTouchEnd.bind(this);

    canvas.addEventListener('mousedown',  this._onMouseDown);
    canvas.addEventListener('mousemove',  this._onMouseMove);
    canvas.addEventListener('mouseup',    this._onMouseUp);
    canvas.addEventListener('mouseleave', this._onMouseUp);
    canvas.addEventListener('touchstart', this._onTouchStart, { passive: true });
    canvas.addEventListener('touchmove',  this._onTouchMove,  { passive: true });
    canvas.addEventListener('touchend',   this._onTouchEnd);
  }

  // ── 鼠标事件 ──────────────────────────────────────────────────────────

  private onMouseDown(e: MouseEvent): void {
    this.pressing = true;
    this.updatePos(e.clientX, e.clientY);
    this.lastX = this.mouseX;
    this.lastY = this.mouseY;
  }

  private onMouseMove(e: MouseEvent): void {
    this.updatePos(e.clientX, e.clientY);
    // 即使未按下也注入轻微扰动（hover 尾流）
    this.emitSplat(this.pressing ? 1.0 : 0.12);
  }

  private onMouseUp(): void {
    this.pressing = false;
  }

  // ── 触摸事件 ──────────────────────────────────────────────────────────

  private onTouchStart(e: TouchEvent): void {
    const t = e.touches[0];
    this.pressing = true;
    this.updatePos(t.clientX, t.clientY);
    this.lastX = this.mouseX;
    this.lastY = this.mouseY;
  }

  private onTouchMove(e: TouchEvent): void {
    const t = e.touches[0];
    this.updatePos(t.clientX, t.clientY);
    this.emitSplat(1.0);
  }

  private onTouchEnd(): void {
    this.pressing = false;
  }

  // ── 坐标转换 + splat 触发 ──────────────────────────────────────────────

  private updatePos(clientX: number, clientY: number): void {
    const rect = this.canvas.getBoundingClientRect();
    this.lastX = this.mouseX;
    this.lastY = this.mouseY;
    // 归一化 [0,1] 并修正 Y 轴 (canvas 坐标系 Y 向下 → UV Y 向上)
    this.mouseX = (clientX - rect.left) / rect.width;
    this.mouseY = 1.0 - (clientY - rect.top) / rect.height;
  }

  private emitSplat(densityMul: number): void {
    const dx = (this.mouseX - this.lastX) * this.opts.velocityScale;
    const dy = (this.mouseY - this.lastY) * this.opts.velocityScale;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 1e-5) return;   // 静止时不注入
    this.fluid.splat(
      this.mouseX,
      this.mouseY,
      dx,
      dy,
      this.opts.densityStrength * densityMul,
    );
  }

  /**
   * 外部主动注入 (e.g. cell 移动时产生尾流).
   * x, y: 归一化坐标 [0,1]
   * dx, dy: 归一化速度增量
   */
  injectAt(x: number, y: number, dx: number, dy: number, density = 0.5): void {
    this.fluid.splat(x, y, dx * this.opts.velocityScale, dy * this.opts.velocityScale, density);
  }

  /** 解绑所有事件监听 */
  destroy(): void {
    const c = this.canvas;
    c.removeEventListener('mousedown',  this._onMouseDown);
    c.removeEventListener('mousemove',  this._onMouseMove);
    c.removeEventListener('mouseup',    this._onMouseUp);
    c.removeEventListener('mouseleave', this._onMouseUp);
    c.removeEventListener('touchstart', this._onTouchStart);
    c.removeEventListener('touchmove',  this._onTouchMove);
    c.removeEventListener('touchend',   this._onTouchEnd);
  }
}
