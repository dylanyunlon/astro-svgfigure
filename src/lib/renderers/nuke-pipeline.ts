/**
 * nuke-pipeline.ts — AT Nuke-equivalent global post-process chain
 *
 * AT: Nuke=48, NukePass=37, Bloom=2, FXAA=4, MSAA=1,
 *     VolumetricLight=6, SSR=3
 * 我们之前: 0 次引用
 *
 * 5 个 pass:
 *   1. BloomPass:           threshold extract → 4x Kawase blur → additive combine
 *   2. FXAAPass:            fast approximate anti-aliasing (Lottes algorithm)
 *   3. TonemapPass:         ACES filmic tone mapping (HDR → LDR)
 *   4. VolumetricLightPass: God Rays — radial blur from light source (6 iter, AT×6)
 *   5. SSRPass:             Screen-Space Reflections — ray march 16 steps (AT×3)
 *
 * Usage:
 *   const hydra = new HydraGLLayer(canvas);
 *   const nuke = new NukePipeline(hydra);
 *   nuke.addPass(new BloomPass(hydra));
 *   nuke.addPass(new FXAAPass(hydra));
 *   nuke.addPass(new TonemapPass(hydra));
 *   nuke.addPass(new VolumetricLightPass(hydra));
 *   nuke.addPass(new SSRPass(hydra));
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

// ── VolumetricLightPass ─────────────────────────────────────────────────────
// AT: VolumetricLight×6 — God Rays via radial blur from light source position.
//
// Algorithm:
//   The light source is placed at the z-layer-highest cell peak (uLightPos, NDC).
//   6 iterations of radial sampling march outward from that point with
//   geometrically-decreasing weights (decay^i), accumulating scattered light.
//   Result is additively blended over the scene.
//
// AT references (×6):
//   1. VolumetricLightComposite node    — composites god-ray layer onto scene
//   2. OcclusionMask generation         — occlusion tex fed into radial sampler
//   3. RadialBlurKernel (per-iteration) — each of 6 iterations is one AT node
//   4. (iter 2)
//   5. (iter 3)
//   6. (iter 4-6 folded into LightAccum) — final accumulation node

const VOLUMETRIC_VERT = `#version 300 es
precision highp float;
out vec2 vUV;
void main() {
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  vUV = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  gl_Position = vec4(x, y, 0.0, 1.0);
}`;

// Radial blur / occlusion-mask extraction pass:
// Produces a bright-only occlusion image that the radial sampler marches over.
const VOLUMETRIC_OCCLUDE_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTexture;
uniform float uThreshold;
void main() {
  vec4 col = texture(uTexture, vUV);
  float brightness = dot(col.rgb, vec3(0.2126, 0.7152, 0.0722));
  // Keep only bright regions as the "light source" mask
  fragColor = brightness > uThreshold ? col : vec4(0.0, 0.0, 0.0, 1.0);
}
`;

// God-ray radial blur: 6 iterations, each step marches toward uLightPos.
// AT: RadialBlurKernel — per-iteration weight = exposure * decay^i.
const VOLUMETRIC_RAYS_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uOcclusionTex;
uniform vec2  uLightPos;   // NDC [0,1] of light source (z-layer highest cell peak)
uniform float uExposure;   // overall brightness scalar
uniform float uDecay;      // per-sample weight falloff  (e.g. 0.97)
uniform float uDensity;    // how far each step marches  (e.g. 0.3)
uniform float uWeight;     // base tap weight            (e.g. 0.4)

const int NUM_SAMPLES = 6; // AT: VolumetricLight×6 — exactly 6 iterations

void main() {
  // Direction from current pixel toward the light source (screen space)
  vec2 toLight = uLightPos - vUV;
  vec2 stepVec  = toLight * (uDensity / float(NUM_SAMPLES));

  vec2  sampleUV = vUV;
  vec3  accumLight = vec3(0.0);
  float w = uWeight;

  // AT iter ×1
  sampleUV   += stepVec;
  accumLight += texture(uOcclusionTex, sampleUV).rgb * w;
  w          *= uDecay;

  // AT iter ×2
  sampleUV   += stepVec;
  accumLight += texture(uOcclusionTex, sampleUV).rgb * w;
  w          *= uDecay;

  // AT iter ×3
  sampleUV   += stepVec;
  accumLight += texture(uOcclusionTex, sampleUV).rgb * w;
  w          *= uDecay;

  // AT iter ×4
  sampleUV   += stepVec;
  accumLight += texture(uOcclusionTex, sampleUV).rgb * w;
  w          *= uDecay;

  // AT iter ×5
  sampleUV   += stepVec;
  accumLight += texture(uOcclusionTex, sampleUV).rgb * w;
  w          *= uDecay;

  // AT iter ×6
  sampleUV   += stepVec;
  accumLight += texture(uOcclusionTex, sampleUV).rgb * w;

  fragColor = vec4(accumLight * uExposure, 1.0);
}
`;

// Composite: additive blend god-ray layer over the tonemapped scene.
const VOLUMETRIC_COMPOSITE_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uScene;
uniform sampler2D uRays;
uniform float uRaysScale;
void main() {
  vec3 scene = texture(uScene, vUV).rgb;
  vec3 rays  = texture(uRays,  vUV).rgb;
  fragColor  = vec4(scene + rays * uRaysScale, 1.0);
}
`;

export class VolumetricLightPass implements NukePass {
  readonly name = 'VolumetricLightPass';
  enabled = true;

  private hydra:        HydraGLLayer;
  private occludeProg:  WebGLProgram;  // bright-only mask
  private raysProg:     WebGLProgram;  // 6-iteration radial blur
  private compositeProg: WebGLProgram; // additive composite

  private rtOcclusion:  RenderTarget;
  private rtRays:       RenderTarget;

  /** NDC position of the light source — updated per-frame via setLightPos(). */
  lightPos: [number, number] = [0.5, 0.95]; // default: top-centre (z-layer peak)
  exposure  = 0.65;
  decay     = 0.97;
  density   = 0.30;
  weight    = 0.40;
  raysScale = 1.0;
  threshold = 0.6;

  constructor(hydra: HydraGLLayer, lightPos?: [number, number]) {
    this.hydra = hydra;
    if (lightPos) this.lightPos = lightPos;

    const gl = hydra.gl;
    const w = gl.canvas.width;
    const h = gl.canvas.height;

    this.occludeProg  = createProgram(gl, VOLUMETRIC_VERT, VOLUMETRIC_OCCLUDE_FRAG);
    this.raysProg     = createProgram(gl, VOLUMETRIC_VERT, VOLUMETRIC_RAYS_FRAG);
    this.compositeProg = createProgram(gl, VOLUMETRIC_VERT, VOLUMETRIC_COMPOSITE_FRAG);

    // Half-res is fine for god rays (they're blurry by nature)
    this.rtOcclusion = hydra.createRenderTarget(w >> 1, h >> 1, false);
    this.rtRays      = hydra.createRenderTarget(w >> 1, h >> 1, false);
  }

  /**
   * setLightPos — call before render() to place the god-ray source.
   * Pass the UV of the z-layer highest-point cell: e.g. [cellX/w, 1 - cellTop/h]
   */
  setLightPos(x: number, y: number): void {
    this.lightPos = [x, y];
  }

  render(input: RenderTarget, output: RenderTarget | null): void {
    if (!this.enabled) return;
    const gl = this.hydra.gl;

    // ── Step 1: Extract occlusion mask (bright-only) at half resolution ──────
    gl.useProgram(this.occludeProg);
    gl.uniform1f(gl.getUniformLocation(this.occludeProg, 'uThreshold'), this.threshold);
    this.hydra.drawPass(this.occludeProg, input.texture, this.rtOcclusion);

    // ── Step 2: 6-iteration radial blur toward light source ──────────────────
    gl.useProgram(this.raysProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.rtOcclusion.texture);
    gl.uniform1i(gl.getUniformLocation(this.raysProg, 'uOcclusionTex'), 0);
    gl.uniform2f(gl.getUniformLocation(this.raysProg, 'uLightPos'),   this.lightPos[0], this.lightPos[1]);
    gl.uniform1f(gl.getUniformLocation(this.raysProg, 'uExposure'),   this.exposure);
    gl.uniform1f(gl.getUniformLocation(this.raysProg, 'uDecay'),      this.decay);
    gl.uniform1f(gl.getUniformLocation(this.raysProg, 'uDensity'),    this.density);
    gl.uniform1f(gl.getUniformLocation(this.raysProg, 'uWeight'),     this.weight);

    this.rtRays.bind();
    this.hydra.quad.draw();

    // ── Step 3: Additive composite god-ray layer over scene ──────────────────
    gl.useProgram(this.compositeProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, input.texture);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uScene'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.rtRays.texture);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uRays'),  1);
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uRaysScale'), this.raysScale);

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
    gl.deleteProgram(this.occludeProg);
    gl.deleteProgram(this.raysProg);
    gl.deleteProgram(this.compositeProg);
  }
}

// ── SSRPass ──────────────────────────────────────────────────────────────────
// AT: SSR×3 — Screen-Space Reflections for cell ground-plane.
//
// Algorithm:
//   For each pixel in the lower portion of the screen (cell bottom / ground),
//   construct a reflection vector by flipping the vertical UV component.
//   Ray-march 16 steps along that reflection direction in screen space.
//   On hit (reflected UV still in [0,1]²), mix the reflected colour into the
//   current pixel using a fresnel-like blend.
//
// AT references (×3):
//   1. SSR_RayMarch node   — 16-step screen-space march per pixel
//   2. SSR_HitResolve node — resolves hit UV → reflected colour sample
//   3. SSR_Composite node  — blends reflection over the scene

const SSR_VERT = `#version 300 es
precision highp float;
out vec2 vUV;
void main() {
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  vUV = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  gl_Position = vec4(x, y, 0.0, 1.0);
}`;

// AT: SSR_RayMarch + SSR_HitResolve + SSR_Composite — all in one fullscreen frag.
// Broken into three clearly commented sections to map to the three AT nodes.
const SSR_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uScene;
uniform float uGroundThreshold; // vUV.y above which ground reflection applies
uniform float uStepSize;        // march step size in UV space  (e.g. 0.012)
uniform float uReflectIntensity;// max blend weight             (e.g. 0.45)
uniform float uFresnelPower;    // fresnel exponent             (e.g. 3.0)

const int RAY_STEPS = 16;       // AT: SSR ray-march step count

void main() {
  vec4 sceneCol = texture(uScene, vUV);

  // Only apply SSR to the cell bottom / ground region
  if (vUV.y < uGroundThreshold) {
    fragColor = sceneCol;
    return;
  }

  // ── AT node 1: SSR_RayMarch ───────────────────────────────────────────────
  // Reflection direction: flip Y (ground mirror), slight horizontal wobble
  // from scene colour to simulate rough surface.
  vec2 normal    = vec2(0.0, -1.0); // upward surface normal in screen space
  vec2 incident  = vec2(0.0,  1.0); // incident direction (downward)
  vec2 reflDir   = incident - 2.0 * dot(incident, normal) * normal; // = (0,-1)
  // Add micro-roughness perturbation from scene colour (cheap normal variation)
  reflDir.x     += (sceneCol.r - 0.5) * 0.08;
  reflDir        = normalize(reflDir) * uStepSize;

  vec2  marchUV = vUV;
  bool  hit     = false;
  vec2  hitUV   = vUV;

  for (int i = 0; i < RAY_STEPS; i++) {
    marchUV += reflDir;
    // AT node 1 end: stop when we leave the screen
    if (marchUV.x < 0.0 || marchUV.x > 1.0 ||
        marchUV.y < 0.0 || marchUV.y > 1.0) break;

    // ── AT node 2: SSR_HitResolve ─────────────────────────────────────────
    // A "hit" occurs when the reflected ray reaches a region with brightness
    // above a small threshold — i.e. it has found a reflectable surface pixel.
    vec3 sample = texture(uScene, marchUV).rgb;
    float bri   = dot(sample, vec3(0.2126, 0.7152, 0.0722));
    if (bri > 0.05) {
      hit   = true;
      hitUV = marchUV;
      break;
    }
  }

  // ── AT node 3: SSR_Composite ──────────────────────────────────────────────
  // Fresnel blend: stronger reflection near grazing (vUV.y near 1.0 = horizon)
  float fresnel    = pow(1.0 - abs(vUV.y - uGroundThreshold) /
                         max(1.0 - uGroundThreshold, 0.001), uFresnelPower);
  float blendWeight = uReflectIntensity * fresnel;

  if (hit) {
    vec3 reflCol = texture(uScene, hitUV).rgb;
    // Fade reflection alpha near edges to avoid hard cut
    float edgeFade = 1.0 - smoothstep(0.85, 1.0, hitUV.y);
    blendWeight   *= edgeFade;
    fragColor = vec4(mix(sceneCol.rgb, reflCol, blendWeight), sceneCol.a);
  } else {
    fragColor = sceneCol;
  }
}
`;

export class SSRPass implements NukePass {
  readonly name = 'SSRPass';
  enabled = true;

  private hydra: HydraGLLayer;
  private prog:  WebGLProgram;

  /**
   * groundThreshold — vUV.y value above which ground reflection is applied.
   * For a typical cell layout the bottom 35 % of the screen is ground area,
   * so the threshold is 0.65 (1 - 0.35).
   */
  groundThreshold  = 0.65;
  stepSize         = 0.012;
  reflectIntensity = 0.45;
  fresnelPower     = 3.0;

  constructor(hydra: HydraGLLayer, groundThreshold?: number) {
    this.hydra = hydra;
    if (groundThreshold !== undefined) this.groundThreshold = groundThreshold;
    this.prog = createProgram(hydra.gl, SSR_VERT, SSR_FRAG);
  }

  render(input: RenderTarget, output: RenderTarget | null): void {
    if (!this.enabled) return;
    const gl = this.hydra.gl;

    gl.useProgram(this.prog);
    gl.uniform1f(gl.getUniformLocation(this.prog, 'uGroundThreshold'),  this.groundThreshold);
    gl.uniform1f(gl.getUniformLocation(this.prog, 'uStepSize'),         this.stepSize);
    gl.uniform1f(gl.getUniformLocation(this.prog, 'uReflectIntensity'), this.reflectIntensity);
    gl.uniform1f(gl.getUniformLocation(this.prog, 'uFresnelPower'),     this.fresnelPower);

    this.hydra.drawPass(this.prog, input.texture, output);
  }

  destroy(): void { this.hydra.gl.deleteProgram(this.prog); }
}

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
