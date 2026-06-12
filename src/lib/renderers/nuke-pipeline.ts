/**
 * nuke-pipeline.ts — AT Nuke-equivalent global post-process chain
 *
 * AT: Nuke=48, NukePass=37, Bloom=2, FXAA=4, MSAA=1
 * 我们之前: 0 次引用
 *
 * 3 个 pass:
 *   1. BloomPass: threshold extract → 4x Kawase blur → additive combine
 *   2. FXAAPass: fast approximate anti-aliasing (Lottes algorithm)
 *   3. TonemapPass: ACES filmic tone mapping (HDR → LDR)
 *
 * Usage:
 *   const hydra = new HydraGLLayer(canvas);
 *   const nuke = new NukePipeline(hydra);
 *   nuke.addPass(new BloomPass(hydra));
 *   nuke.addPass(new FXAAPass(hydra));
 *   nuke.addPass(new TonemapPass(hydra));
 *   // Each frame:
 *   nuke.render(sceneTexture);  // outputs to canvas
 */

import { HydraGLLayer, RenderTarget, createProgram } from './hydra-gl-layer';

// ── NukePass interface ──────────────────────────────────────────────────────

export interface NukePass {
  readonly name: string;
  enabled: boolean;
  /** Render this pass: read from input RT, write to output RT (null = canvas) */
  render(input: RenderTarget, output: RenderTarget | null): void;
  destroy(): void;
}

// ── BloomPass ───────────────────────────────────────────────────────────────
// AT: HydraBloom — threshold extraction + Kawase downSample/upSample + additive

const BLOOM_EXTRACT_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTexture;
uniform float uThreshold;
void main() {
  vec4 col = texture(uTexture, vUV);
  float brightness = dot(col.rgb, vec3(0.2126, 0.7152, 0.0722));
  fragColor = brightness > uThreshold ? col : vec4(0.0);
}
`;

const BLOOM_BLUR_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTexture;
uniform vec2 uTexelSize;
uniform vec2 uDirection;
void main() {
  // Kawase blur: 5-tap filter along direction
  vec4 sum = vec4(0.0);
  sum += texture(uTexture, vUV - 2.0 * uDirection * uTexelSize) * 0.06136;
  sum += texture(uTexture, vUV - 1.0 * uDirection * uTexelSize) * 0.24477;
  sum += texture(uTexture, vUV) * 0.38774;
  sum += texture(uTexture, vUV + 1.0 * uDirection * uTexelSize) * 0.24477;
  sum += texture(uTexture, vUV + 2.0 * uDirection * uTexelSize) * 0.06136;
  fragColor = sum;
}
`;

const BLOOM_COMBINE_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uBloomScale;
void main() {
  vec4 scene = texture(uScene, vUV);
  vec4 bloom = texture(uBloom, vUV);
  fragColor = scene + bloom * uBloomScale;
}
`;

export class BloomPass implements NukePass {
  readonly name = 'BloomPass';
  enabled = true;

  private hydra: HydraGLLayer;
  private extractProg: WebGLProgram;
  private blurProg: WebGLProgram;
  private combineProg: WebGLProgram;
  private rtExtract: RenderTarget;
  private rtBlurH: RenderTarget;
  private rtBlurV: RenderTarget;
  private threshold: number;
  private bloomScale: number;

  constructor(hydra: HydraGLLayer, threshold = 0.5, bloomScale = 1.5) {
    this.hydra = hydra;
    this.threshold = threshold;
    this.bloomScale = bloomScale;

    const gl = hydra.gl;
    const w = gl.canvas.width / 2;  // Half-res for blur
    const h = gl.canvas.height / 2;

    // Vertex shader is inside HydraGLLayer (FULLSCREEN_VERT)
    const VERT = `#version 300 es
precision highp float;
out vec2 vUV;
void main() {
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  vUV = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  gl_Position = vec4(x, y, 0.0, 1.0);
}`;

    this.extractProg = createProgram(gl, VERT, BLOOM_EXTRACT_FRAG);
    this.blurProg = createProgram(gl, VERT, BLOOM_BLUR_FRAG);
    this.combineProg = createProgram(gl, VERT, BLOOM_COMBINE_FRAG);

    this.rtExtract = hydra.createRenderTarget(w, h, true);
    this.rtBlurH = hydra.createRenderTarget(w, h, true);
    this.rtBlurV = hydra.createRenderTarget(w, h, true);
  }

  render(input: RenderTarget, output: RenderTarget | null): void {
    if (!this.enabled) return;
    const gl = this.hydra.gl;

    // Pass 1: Extract bright pixels
    gl.useProgram(this.extractProg);
    this.hydra.drawPass(this.extractProg, input.texture, this.rtExtract);
    gl.uniform1f(gl.getUniformLocation(this.extractProg, 'uThreshold'), this.threshold);

    // Pass 2: Horizontal Kawase blur
    gl.useProgram(this.blurProg);
    gl.uniform2f(gl.getUniformLocation(this.blurProg, 'uTexelSize'),
                 1.0 / this.rtExtract.width, 1.0 / this.rtExtract.height);
    gl.uniform2f(gl.getUniformLocation(this.blurProg, 'uDirection'), 1.0, 0.0);
    this.hydra.drawPass(this.blurProg, this.rtExtract.texture, this.rtBlurH);

    // Pass 3: Vertical Kawase blur
    gl.uniform2f(gl.getUniformLocation(this.blurProg, 'uDirection'), 0.0, 1.0);
    this.hydra.drawPass(this.blurProg, this.rtBlurH.texture, this.rtBlurV);

    // Pass 4: Additive combine scene + bloom
    gl.useProgram(this.combineProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, input.texture);
    gl.uniform1i(gl.getUniformLocation(this.combineProg, 'uScene'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.rtBlurV.texture);
    gl.uniform1i(gl.getUniformLocation(this.combineProg, 'uBloom'), 1);
    gl.uniform1f(gl.getUniformLocation(this.combineProg, 'uBloomScale'), this.bloomScale);

    if (output) {
      output.bind();
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    }
    this.hydra.quad.draw();
  }

  destroy(): void {
    const gl = this.hydra.gl;
    gl.deleteProgram(this.extractProg);
    gl.deleteProgram(this.blurProg);
    gl.deleteProgram(this.combineProg);
  }
}

// ── FXAAPass ────────────────────────────────────────────────────────────────
// Timothy Lottes' FXAA 3.11 — fast approximate anti-aliasing

const FXAA_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTexture;
uniform vec2 uTexelSize;

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
  vec3 rgbNW = texture(uTexture, vUV + vec2(-1.0, -1.0) * uTexelSize).rgb;
  vec3 rgbNE = texture(uTexture, vUV + vec2( 1.0, -1.0) * uTexelSize).rgb;
  vec3 rgbSW = texture(uTexture, vUV + vec2(-1.0,  1.0) * uTexelSize).rgb;
  vec3 rgbSE = texture(uTexture, vUV + vec2( 1.0,  1.0) * uTexelSize).rgb;
  vec3 rgbM  = texture(uTexture, vUV).rgb;

  float lumaNW = luma(rgbNW), lumaNE = luma(rgbNE);
  float lumaSW = luma(rgbSW), lumaSE = luma(rgbSE);
  float lumaM  = luma(rgbM);

  float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
  float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
  float lumaRange = lumaMax - lumaMin;

  if (lumaRange < max(0.0312, lumaMax * 0.125)) {
    fragColor = vec4(rgbM, 1.0);
    return;
  }

  vec2 dir;
  dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
  dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
  float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.25, 1.0/128.0);
  float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * uTexelSize;

  vec3 rgbA = 0.5 * (
    texture(uTexture, vUV + dir * (1.0/3.0 - 0.5)).rgb +
    texture(uTexture, vUV + dir * (2.0/3.0 - 0.5)).rgb);
  vec3 rgbB = rgbA * 0.5 + 0.25 * (
    texture(uTexture, vUV + dir * -0.5).rgb +
    texture(uTexture, vUV + dir *  0.5).rgb);

  float lumaB = luma(rgbB);
  fragColor = vec4((lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB, 1.0);
}
`;

export class FXAAPass implements NukePass {
  readonly name = 'FXAAPass';
  enabled = true;
  private hydra: HydraGLLayer;
  private prog: WebGLProgram;

  constructor(hydra: HydraGLLayer) {
    this.hydra = hydra;
    const VERT = `#version 300 es
precision highp float;
out vec2 vUV;
void main() {
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  vUV = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  gl_Position = vec4(x, y, 0.0, 1.0);
}`;
    this.prog = createProgram(hydra.gl, VERT, FXAA_FRAG);
  }

  render(input: RenderTarget, output: RenderTarget | null): void {
    if (!this.enabled) return;
    const gl = this.hydra.gl;
    gl.useProgram(this.prog);
    gl.uniform2f(gl.getUniformLocation(this.prog, 'uTexelSize'),
                 1.0 / input.width, 1.0 / input.height);
    this.hydra.drawPass(this.prog, input.texture, output);
  }

  destroy(): void { this.hydra.gl.deleteProgram(this.prog); }
}

// ── TonemapPass ─────────────────────────────────────────────────────────────
// ACES filmic tone mapping — maps HDR bloom values back to [0,1]

const TONEMAP_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTexture;

vec3 aces(vec3 x) {
  float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main() {
  vec3 col = texture(uTexture, vUV).rgb;
  col = aces(col);
  // Gamma correction
  col = pow(col, vec3(1.0 / 2.2));
  fragColor = vec4(col, 1.0);
}
`;

export class TonemapPass implements NukePass {
  readonly name = 'TonemapPass';
  enabled = true;
  private hydra: HydraGLLayer;
  private prog: WebGLProgram;

  constructor(hydra: HydraGLLayer) {
    this.hydra = hydra;
    const VERT = `#version 300 es
precision highp float;
out vec2 vUV;
void main() {
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  vUV = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  gl_Position = vec4(x, y, 0.0, 1.0);
}`;
    this.prog = createProgram(hydra.gl, VERT, TONEMAP_FRAG);
  }

  render(input: RenderTarget, output: RenderTarget | null): void {
    if (!this.enabled) return;
    this.hydra.drawPass(this.prog, input.texture, output);
  }

  destroy(): void { this.hydra.gl.deleteProgram(this.prog); }
}

// ── NukePipeline ────────────────────────────────────────────────────────────

export class NukePipeline {
  private hydra: HydraGLLayer;
  private passes: NukePass[] = [];
  private pingRT: RenderTarget;
  private pongRT: RenderTarget;

  constructor(hydra: HydraGLLayer) {
    this.hydra = hydra;
    const gl = hydra.gl;
    const w = gl.canvas.width;
    const h = gl.canvas.height;
    this.pingRT = hydra.createRenderTarget(w, h, true);
    this.pongRT = hydra.createRenderTarget(w, h, true);
  }

  addPass(pass: NukePass): void {
    this.passes.push(pass);
  }

  /** Run all passes: sceneTexture → pass chain → canvas */
  render(sceneTexture: WebGLTexture): void {
    const enabledPasses = this.passes.filter(p => p.enabled);
    if (enabledPasses.length === 0) return;

    // Copy scene texture into ping RT for first pass
    // (In practice, PixiJS would render directly into ping RT)
    let currentInput = this.pingRT;
    let currentOutput = this.pongRT;

    // First pass reads from sceneTexture
    const first = enabledPasses[0];
    const isLast = enabledPasses.length === 1;
    // Create a temporary RT wrapping the scene texture
    const sceneRT = { texture: sceneTexture, width: this.pingRT.width, height: this.pingRT.height } as RenderTarget;

    first.render(sceneRT, isLast ? null : this.pingRT);

    // Remaining passes ping-pong
    for (let i = 1; i < enabledPasses.length; i++) {
      const pass = enabledPasses[i];
      const isLastPass = i === enabledPasses.length - 1;
      const input = i % 2 === 1 ? this.pingRT : this.pongRT;
      const output = isLastPass ? null : (i % 2 === 1 ? this.pongRT : this.pingRT);
      pass.render(input, output);
    }
  }

  destroy(): void {
    for (const p of this.passes) p.destroy();
  }
}
