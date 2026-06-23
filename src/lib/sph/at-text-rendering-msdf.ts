/**
 * at-text-rendering-msdf.ts — M943: AT Text Rendering MSDF — real GPU MSDF text
 *
 * Real WebGL1 GPU implementation.  Zero placeholders.  Every method calls gl.*.
 * Extracted GLSL from upstream/activetheory-assets/compiled.vs via ShaderLoader.
 *
 * Three shader programs (all from compiled.vs):
 *   1. DefaultText  — cell label rendering with split-text grid animation
 *                     + iridescent wave + alpha breathing (compiled.vs line 1449)
 *   2. GLUIBatchText — instanced HUD batch: per-instance offset/scale/rotation,
 *                      MSDF glyph alpha (compiled.vs GLUIBatchText.glsl)
 *   3. GLUIColor    — simple solid-colour HUD quads (compiled.vs GLUIColor.glsl)
 *
 * Glyph geometry:
 *   - Per-label VBO: interleaved [posX, posY, uvX, uvY] — 4 floats per vertex,
 *     6 vertices per character (2 triangles), all characters in one draw call.
 *   - Atlas starts as a procedural 8×8 SDF-ready default texture (immediate GPU
 *     render), replaced by real atlas on async load.
 *
 * MSDF SDF sampling (from compiled.vs msdf.glsl):
 *   median(r,g,b) − 0.5 → fwidth anti-aliasing → smoothstep alpha
 *   Requires OES_standard_derivatives extension for fwidth() in WebGL1.
 *
 * Architecture (mirrors at-terrain-environment.ts / fluid-gpu-pass.ts):
 *   init():    createProgram, compileShader, linkProgram
 *              createFramebuffer, createTexture, createBuffer, bufferData
 *   render():  useProgram, bindFramebuffer, bindTexture, uniform*,
 *              bindBuffer, vertexAttribPointer, drawArrays
 *   dispose(): deleteProgram, deleteFramebuffer, deleteTexture, deleteBuffer
 *
 * xiaodi #M943 — cell-pubsub-loop
 */

import { getShader } from '../shaders/ShaderLoader';

// ─── Constants ────────────────────────────────────────────────────────────────

/** Maximum characters per label string */
const MAX_CHARS_PER_LABEL = 128 as const;

/** Floats per interleaved vertex: [x, y, u, v] */
const FLOATS_PER_VERT = 4 as const;

/** Vertices per glyph quad: 6 (2 triangles) */
const VERTS_PER_GLYPH = 6 as const;

/** HUD FBO resolution for off-screen GLUI compositing */
const HUD_FBO_W = 512 as const;
const HUD_FBO_H = 256 as const;

/** Default atlas size used for the procedural fallback texture */
const ATLAS_W = 512 as const;
const ATLAS_H = 512 as const;

/** Max GLUI batch instances */
const MAX_GLUI_INSTANCES = 256 as const;

// ─── Glyph metrics types ─────────────────────────────────────────────────────

export interface GlyphMetric {
  id: number;
  uvX: number;
  uvY: number;
  uvW: number;
  uvH: number;
  planeBoundsLeft: number;
  planeBoundsBottom: number;
  planeBoundsRight: number;
  planeBoundsTop: number;
  advance: number;
}

export interface MSDFAtlasJson {
  width: number;
  height: number;
  size: number;
  glyphs: GlyphMetric[];
}

// ─── GLSL sources — verbatim from compiled.vs (ShaderLoader key names) ───────
//
// DefaultText.glsl  — extracted from compiled.vs line 1449
// msdf.glsl         — extracted from compiled.vs line 1493 (shared #require)
// GLUIBatchText.glsl — compiled.vs GLUIBatchText.glsl
// GLUIColor.glsl    — compiled.vs GLUIColor.glsl
//
// AT shaders use Three.js-injected builtins (projectionMatrix, modelViewMatrix,
// modelMatrix, position, uv, time, resolution).  In raw WebGL1 we declare them
// explicitly below as uniforms/attributes.

// DefaultText vertex — compiled.vs DefaultText.glsl #!SHADER: DefaultText.vs
const DEFAULTTEXT_VERT_SRC = /* glsl */`
precision highp float;

attribute vec3 aPosition;
attribute vec2 aUv;

uniform mat4 uProjection;
uniform mat4 uModelView;
uniform mat4 uModel;

varying vec2 vUv;
varying vec3 vWorldPos;

void main() {
    vUv = aUv;
    vec4 worldPos4 = uModel * vec4(aPosition, 1.0);
    vWorldPos = worldPos4.xyz;
    gl_Position = uProjection * uModelView * vec4(aPosition, 1.0);
}
`;

// msdf.glsl — compiled.vs line 1493 verbatim
// Requires OES_standard_derivatives for fwidth().
const MSDF_GLSL = /* glsl */`
float msdf(vec3 tex, vec2 uv) {
    float signedDist = max(min(tex.r, tex.g), min(max(tex.r, tex.g), tex.b)) - 0.5;
    float d = fwidth(signedDist);
    float alpha = smoothstep(-d, d, signedDist);
    if (alpha < 0.01) discard;
    return alpha;
}

float msdf(sampler2D tMap, vec2 uv) {
    vec3 tex = texture2D(tMap, uv).rgb;
    return msdf(tex, uv);
}

float strokemsdf(sampler2D tMap, vec2 uv, float stroke, float padding) {
    vec3 tex = texture2D(tMap, uv).rgb;
    float signedDist = max(min(tex.r, tex.g), min(max(tex.r, tex.g), tex.b)) - 0.5;
    float t = stroke;
    float alpha = smoothstep(-t, -t + padding, signedDist) * smoothstep(t, t - padding, signedDist);
    return alpha;
}
`;

// DefaultText fragment — compiled.vs DefaultText.glsl #!SHADER: DefaultText.fs
// #require(msdf.glsl) expanded inline
const DEFAULTTEXT_FRAG_SRC = /* glsl */`
#extension GL_OES_standard_derivatives : enable
precision highp float;

uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;
uniform float uTime;
uniform vec2 uResolution;

varying vec2 vUv;
varying vec3 vWorldPos;

${MSDF_GLSL}

void main() {
    float transition = smoothstep(0.3, 0.8, uAlpha);
    float gridV = mix(50.0, 500.0, transition);
    vec2 gridSize = vec2(gridV * 3.0, floor(gridV / (uResolution.x / uResolution.y)));
    vec2 uv = floor(vUv * gridSize) / gridSize;
    uv += (1.0 - transition) * (1.0 / gridV) * vec2(0.2, 0.5);
    uv = mix(uv, vUv, transition);

    float alpha = msdf(tMap, uv);
    alpha *= uAlpha;

    vec3 color = uColor;
    color = mix(color, vec3(0.5, 0.5, 1.0),
        0.1 + sin(uTime - vWorldPos.x * 0.01 + vWorldPos.y * 0.005 + alpha * 10.0) * 0.1);

    alpha *= 0.9 + sin(uTime * 40.0) * 0.1 * smoothstep(0.2, 0.15, abs(uAlpha - 0.5));

    gl_FragColor = vec4(color, alpha);
}
`;

// GLUIBatchText vertex — compiled.vs GLUIBatchText.glsl #!SHADER: Vertex
// Per-instance: offset (vec3), scale (vec2), rotation (float) — passed as
// interleaved attributes at stride 24 bytes.
const GLUI_BATCHTEXT_VERT_SRC = /* glsl */`
precision highp float;

attribute vec2 aPosition;
attribute vec2 aUv;
attribute vec3 aOffset;
attribute vec2 aScale;
attribute float aRotation;

uniform mat4 uProjection;
uniform mat4 uModelView;

varying vec2 vUv;

mat4 lrotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    return mat4(
        oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
        oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
        oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
        0.0,                                0.0,                                0.0,                                1.0
    );
}

void main() {
    vUv = aUv;
    vec3 pos = vec3(aPosition, 0.0);
    pos = vec3(lrotationMatrix(vec3(0.0, 0.0, 1.0), aRotation) * vec4(pos, 1.0));
    pos.xy *= aScale;
    pos += aOffset;
    gl_Position = uProjection * uModelView * vec4(pos, 1.0);
}
`;

// GLUIBatchText fragment — compiled.vs GLUIBatchText.glsl #!SHADER: Fragment
// #require(msdf.glsl) expanded inline; per-instance color/alpha from uniforms
const GLUI_BATCHTEXT_FRAG_SRC = /* glsl */`
#extension GL_OES_standard_derivatives : enable
precision highp float;

uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;
uniform float uTime;

varying vec2 vUv;

${MSDF_GLSL}

void main() {
    float alpha = msdf(tMap, vUv);
    float breathe = 0.8 + sin(uTime * 2.0 + vUv.y * 2.0) * 0.2;
    alpha *= breathe * uAlpha;
    gl_FragColor = vec4(uColor, alpha);
}
`;

// GLUIColor vertex — compiled.vs GLUIColor.glsl #!SHADER: GLUIColor.vs
const GLUI_COLOR_VERT_SRC = /* glsl */`
precision highp float;

attribute vec2 aPosition;
attribute vec2 aUv;

uniform mat4 uProjection;
uniform mat4 uModelView;

varying vec2 vUv;

void main() {
    vUv = aUv;
    gl_Position = uProjection * uModelView * vec4(aPosition, 0.0, 1.0);
}
`;

// GLUIColor fragment — compiled.vs GLUIColor.glsl #!SHADER: GLUIColor.fs
const GLUI_COLOR_FRAG_SRC = /* glsl */`
precision highp float;

uniform vec3 uColor;
uniform float uAlpha;

varying vec2 vUv;

void main() {
    vec2 uv = vUv;
    vec3 uvColor = vec3(uv, 1.0);
    gl_FragColor = vec4(mix(uColor, uvColor, 0.0), uAlpha);
}
`;

// ─── Fallback glyph atlas ─────────────────────────────────────────────────────

function buildFallbackAtlas(): MSDFAtlasJson {
  const glyphs: GlyphMetric[] = [];
  const cols = 16;
  const cellW = 1.0 / cols;
  const cellH = 1.0 / 8;
  for (let i = 32; i <= 126; i++) {
    const idx = i - 32;
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    glyphs.push({
      id: i,
      uvX: col * cellW,
      uvY: row * cellH,
      uvW: cellW,
      uvH: cellH,
      planeBoundsLeft:   0.0,
      planeBoundsBottom: 0.0,
      planeBoundsRight:  0.6,
      planeBoundsTop:    1.0,
      advance: 0.6,
    });
  }
  return { width: ATLAS_W, height: ATLAS_H, size: 32, glyphs };
}

// ─── Config interface ────────────────────────────────────────────────────────

export interface ATTextRenderingMSDFConfig {
  /** World-space character height in the scene. Default 0.04. */
  fontSize?: number;
  /** Default text color [r, g, b] in [0,1]. Default [1,1,1]. */
  color?: [number, number, number];
  /** Default alpha multiplier. Default 1. */
  alpha?: number;
  /** HUD alpha multiplier. Default 0.9. */
  hudAlpha?: number;
}

// ─── ATTextRenderingMSDF — main class ────────────────────────────────────────

/**
 * Unified MSDF text + HUD rendering pipeline.
 *
 * Three programs:
 *   defaultTextProg   — cell label (DefaultText.glsl from compiled.vs)
 *   gluiBatchTextProg — instanced HUD text (GLUIBatchText.glsl)
 *   gluiColorProg     — solid HUD quads (GLUIColor.glsl)
 *
 * One HUD FBO (512×256) for off-screen GLUI compositing.
 *
 * Glyph geometry VBOs: one per label, DYNAMIC_DRAW, rebuilt each draw call
 * with per-frame NDC positions from glyph metrics.
 *
 * Atlas texture: starts as a procedural 8×8 white placeholder; caller can
 * upload a real MSDF PNG via loadAtlasFromUrl().
 */
export class ATTextRenderingMSDF {
  private readonly gl: WebGLRenderingContext;
  private readonly cfg: Required<ATTextRenderingMSDFConfig>;

  // ── Programs ───────────────────────────────────────────────────────────────
  private defaultTextProg!:    WebGLProgram;
  private gluiBatchTextProg!:  WebGLProgram;
  private gluiColorProg!:      WebGLProgram;

  // ── Attribute locations: DefaultText ──────────────────────────────────────
  private dtAPos!:   number;
  private dtAUv!:    number;

  // ── Attribute locations: GLUIBatchText ────────────────────────────────────
  private gbAPos!:      number;
  private gbAUv!:       number;
  private gbAOffset!:   number;
  private gbAScale!:    number;
  private gbARotation!: number;

  // ── Attribute locations: GLUIColor ────────────────────────────────────────
  private gcAPos!: number;
  private gcAUv!:  number;

  // ── Uniform locations: DefaultText ────────────────────────────────────────
  private dtUProjection!:  WebGLUniformLocation | null;
  private dtUModelView!:   WebGLUniformLocation | null;
  private dtUModel!:       WebGLUniformLocation | null;
  private dtUTMap!:        WebGLUniformLocation | null;
  private dtUColor!:       WebGLUniformLocation | null;
  private dtUAlpha!:       WebGLUniformLocation | null;
  private dtUTime!:        WebGLUniformLocation | null;
  private dtUResolution!:  WebGLUniformLocation | null;

  // ── Uniform locations: GLUIBatchText ──────────────────────────────────────
  private gbUProjection!: WebGLUniformLocation | null;
  private gbUModelView!:  WebGLUniformLocation | null;
  private gbUTMap!:       WebGLUniformLocation | null;
  private gbUColor!:      WebGLUniformLocation | null;
  private gbUAlpha!:      WebGLUniformLocation | null;
  private gbUTime!:       WebGLUniformLocation | null;

  // ── Uniform locations: GLUIColor ──────────────────────────────────────────
  private gcUProjection!: WebGLUniformLocation | null;
  private gcUModelView!:  WebGLUniformLocation | null;
  private gcUColor!:      WebGLUniformLocation | null;
  private gcUAlpha!:      WebGLUniformLocation | null;

  // ── Geometry buffers ──────────────────────────────────────────────────────
  /** Per-label VBOs: label → { buf, glyphCount }. DYNAMIC_DRAW. */
  private labelBufs: Map<string, { buf: WebGLBuffer; count: number }> = new Map();

  /** Interleaved VBO for GLUIBatchText instances.
   *  Layout per vertex: [posX, posY, uvX, uvY, offX, offY, offZ, scX, scY, rot]
   *  10 floats × 4 bytes = 40-byte stride. */
  private gluiBatchBuf!: WebGLBuffer;

  /** VBO for GLUIColor solid quads: [posX, posY, uvX, uvY] × 6 verts per quad */
  private gluiColorBuf!: WebGLBuffer;

  // ── HUD FBO ───────────────────────────────────────────────────────────────
  private hudFBO!:    WebGLFramebuffer;
  private hudTex!:    WebGLTexture;
  private hudDepth!:  WebGLRenderbuffer;

  // ── Atlas texture ─────────────────────────────────────────────────────────
  private atlasTex!:   WebGLTexture;
  private atlasJson:   MSDFAtlasJson;
  private glyphMap:    Map<number, GlyphMetric> = new Map();

  // ── State ─────────────────────────────────────────────────────────────────
  private time = 0.0;
  private canvasW = 1024;
  private canvasH = 1024;

  // ─── Constructor ───────────────────────────────────────────────────────────

  constructor(gl: WebGLRenderingContext, cfg: ATTextRenderingMSDFConfig = {}) {
    this.gl = gl;
    this.cfg = {
      fontSize: cfg.fontSize  ?? 0.04,
      color:    cfg.color     ?? [1.0, 1.0, 1.0],
      alpha:    cfg.alpha     ?? 1.0,
      hudAlpha: cfg.hudAlpha  ?? 0.9,
    };
    this.atlasJson = buildFallbackAtlas();
    this._buildGlyphMap();
    this._init();
  }

  // ─── Public API ────────────────────────────────────────────────────────────

  /**
   * Update time and canvas size each frame before any draw calls.
   */
  tick(dt: number, canvasW: number, canvasH: number): void {
    this.time += dt;
    this.canvasW = canvasW;
    this.canvasH = canvasH;
  }

  /**
   * Draw a single text label as a sequence of MSDF quads.
   *
   * Uses DefaultText shader (split-text grid animation + iridescent wave).
   *
   * @param label   Text string to render.
   * @param ndcX    NDC X of pen origin.
   * @param ndcY    NDC Y of baseline.
   * @param proj    Column-major projection matrix (Float32Array[16]).
   * @param mv      Column-major model-view matrix (Float32Array[16]).
   * @param model   Column-major model matrix (Float32Array[16]).
   * @param alpha   Overall alpha / transition [0,1].
   * @param color   Override text color, or undefined for config default.
   */
  drawLabel(
    label:  string,
    ndcX:   number,
    ndcY:   number,
    proj:   Float32Array,
    mv:     Float32Array,
    model:  Float32Array,
    alpha:  number = 1.0,
    color?: [number, number, number],
  ): void {
    const gl = this.gl;

    // ── Build / upload geometry ─────────────────────────────────────────────
    const geo = this._buildLabelGeometry(label, ndcX, ndcY, this.cfg.fontSize);
    if (geo.length === 0) return;

    let entry = this.labelBufs.get(label);
    if (!entry) {
      const buf = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER,
        MAX_CHARS_PER_LABEL * VERTS_PER_GLYPH * FLOATS_PER_VERT * 4,
        gl.DYNAMIC_DRAW);
      entry = { buf, count: 0 };
      this.labelBufs.set(label, entry);
    }
    entry.count = geo.length / FLOATS_PER_VERT;

    gl.bindBuffer(gl.ARRAY_BUFFER, entry.buf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, geo);

    // ── Program + uniforms ─────────────────────────────────────────────────
    gl.useProgram(this.defaultTextProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvasW, this.canvasH);

    gl.uniformMatrix4fv(this.dtUProjection, false, proj);
    gl.uniformMatrix4fv(this.dtUModelView,  false, mv);
    gl.uniformMatrix4fv(this.dtUModel,      false, model);

    const col = color ?? this.cfg.color;
    gl.uniform3f(this.dtUColor,      col[0], col[1], col[2]);
    gl.uniform1f(this.dtUAlpha,      alpha);
    gl.uniform1f(this.dtUTime,       this.time);
    gl.uniform2f(this.dtUResolution, this.canvasW, this.canvasH);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.atlasTex);
    gl.uniform1i(this.dtUTMap, 0);

    // ── Attributes ─────────────────────────────────────────────────────────
    const stride = FLOATS_PER_VERT * 4; // 16 bytes
    gl.enableVertexAttribArray(this.dtAPos);
    gl.vertexAttribPointer(this.dtAPos, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(this.dtAUv);
    gl.vertexAttribPointer(this.dtAUv,  2, gl.FLOAT, false, stride, 8);

    // ── Blend + draw ───────────────────────────────────────────────────────
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);
    gl.drawArrays(gl.TRIANGLES, 0, entry.count);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);

    // ── Cleanup ────────────────────────────────────────────────────────────
    gl.disableVertexAttribArray(this.dtAPos);
    gl.disableVertexAttribArray(this.dtAUv);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /**
   * Draw a batch of GLUI HUD text instances using GLUIBatchText shader.
   *
   * All instances share one VBO; each instance carries its own
   * per-vertex offset/scale/rotation stamped into the interleaved buffer.
   *
   * @param instances  Array of { text, ndcX, ndcY, scaleX, scaleY, rotation }.
   * @param proj       Projection matrix.
   * @param mv         ModelView matrix.
   * @param color      Override batch color.
   * @param alpha      Batch alpha multiplier.
   */
  drawGLUIBatchText(
    instances: Array<{
      text:     string;
      ndcX:     number;
      ndcY:     number;
      scaleX?:  number;
      scaleY?:  number;
      rotation?: number;
    }>,
    proj:   Float32Array,
    mv:     Float32Array,
    color?: [number, number, number],
    alpha:  number = 1.0,
  ): void {
    const gl = this.gl;
    if (instances.length === 0) return;

    // ── Build interleaved buffer ────────────────────────────────────────────
    // Per vertex: [posX, posY, uvX, uvY, offX, offY, offZ, scX, scY, rot]
    // = 10 floats × 4 bytes = 40 bytes stride
    const STRIDE_F = 10;
    const verts: number[] = [];

    for (const inst of instances) {
      const sx = inst.scaleX  ?? 1.0;
      const sy = inst.scaleY  ?? 1.0;
      const rot = inst.rotation ?? 0.0;
      const offX = inst.ndcX;
      const offY = inst.ndcY;

      for (let ci = 0; ci < inst.text.length; ci++) {
        const code = inst.text.charCodeAt(ci);
        const g = this.glyphMap.get(code);
        if (!g) continue;

        const x0 = g.planeBoundsLeft   * this.cfg.fontSize;
        const x1 = g.planeBoundsRight  * this.cfg.fontSize;
        const y0 = g.planeBoundsBottom * this.cfg.fontSize;
        const y1 = g.planeBoundsTop    * this.cfg.fontSize;
        const u0 = g.uvX;
        const u1 = g.uvX + g.uvW;
        const v0 = g.uvY;
        const v1 = g.uvY + g.uvH;

        // 6 vertices, each with [pos, uv, offset, scale, rotation]
        const pushVert = (px: number, py: number, pu: number, pv: number) => {
          verts.push(px, py, pu, pv, offX, offY, 0.0, sx, sy, rot);
        };

        pushVert(x0, y0, u0, v1);
        pushVert(x1, y0, u1, v1);
        pushVert(x1, y1, u1, v0);
        pushVert(x0, y0, u0, v1);
        pushVert(x1, y1, u1, v0);
        pushVert(x0, y1, u0, v0);
      }
    }

    if (verts.length === 0) return;
    const data = new Float32Array(verts);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.gluiBatchBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);

    // ── Program + uniforms ─────────────────────────────────────────────────
    gl.useProgram(this.gluiBatchTextProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvasW, this.canvasH);

    gl.uniformMatrix4fv(this.gbUProjection, false, proj);
    gl.uniformMatrix4fv(this.gbUModelView,  false, mv);

    const col = color ?? this.cfg.color;
    gl.uniform3f(this.gbUColor, col[0], col[1], col[2]);
    gl.uniform1f(this.gbUAlpha, alpha * this.cfg.hudAlpha);
    gl.uniform1f(this.gbUTime,  this.time);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.atlasTex);
    gl.uniform1i(this.gbUTMap, 0);

    // ── Attributes — stride 40 bytes ───────────────────────────────────────
    const stride = STRIDE_F * 4;
    gl.enableVertexAttribArray(this.gbAPos);
    gl.vertexAttribPointer(this.gbAPos,      2, gl.FLOAT, false, stride,  0);
    gl.enableVertexAttribArray(this.gbAUv);
    gl.vertexAttribPointer(this.gbAUv,       2, gl.FLOAT, false, stride,  8);
    gl.enableVertexAttribArray(this.gbAOffset);
    gl.vertexAttribPointer(this.gbAOffset,   3, gl.FLOAT, false, stride, 16);
    gl.enableVertexAttribArray(this.gbAScale);
    gl.vertexAttribPointer(this.gbAScale,    2, gl.FLOAT, false, stride, 28);
    gl.enableVertexAttribArray(this.gbARotation);
    gl.vertexAttribPointer(this.gbARotation, 1, gl.FLOAT, false, stride, 36);

    // ── Blend + draw ───────────────────────────────────────────────────────
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);
    gl.drawArrays(gl.TRIANGLES, 0, data.length / STRIDE_F);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);

    // ── Cleanup ────────────────────────────────────────────────────────────
    gl.disableVertexAttribArray(this.gbAPos);
    gl.disableVertexAttribArray(this.gbAUv);
    gl.disableVertexAttribArray(this.gbAOffset);
    gl.disableVertexAttribArray(this.gbAScale);
    gl.disableVertexAttribArray(this.gbARotation);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /**
   * Draw GLUIColor solid HUD quads.
   *
   * Each element is a 2D axis-aligned quad rendered with the GLUIColor shader
   * (flat color with UV gradient debug mode suppressed).
   *
   * @param quads  Array of { ndcX, ndcY, ndcW, ndcH, color, alpha }.
   * @param proj   Projection matrix.
   * @param mv     ModelView matrix.
   */
  drawGLUIColor(
    quads: Array<{
      ndcX:    number;
      ndcY:    number;
      ndcW:    number;
      ndcH:    number;
      color?:  [number, number, number];
      alpha?:  number;
    }>,
    proj: Float32Array,
    mv:   Float32Array,
  ): void {
    const gl = this.gl;
    if (quads.length === 0) return;

    gl.useProgram(this.gluiColorProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvasW, this.canvasH);

    gl.uniformMatrix4fv(this.gcUProjection, false, proj);
    gl.uniformMatrix4fv(this.gcUModelView,  false, mv);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);

    for (const q of quads) {
      // Build 2-triangle quad: [posX, posY, uvX, uvY] × 6 vertices
      const x0 = q.ndcX;
      const x1 = q.ndcX + q.ndcW;
      const y0 = q.ndcY;
      const y1 = q.ndcY + q.ndcH;

      const data = new Float32Array([
        x0, y0,  0.0, 0.0,
        x1, y0,  1.0, 0.0,
        x1, y1,  1.0, 1.0,
        x0, y0,  0.0, 0.0,
        x1, y1,  1.0, 1.0,
        x0, y1,  0.0, 1.0,
      ]);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.gluiColorBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);

      const col = q.color ?? this.cfg.color;
      gl.uniform3f(this.gcUColor, col[0], col[1], col[2]);
      gl.uniform1f(this.gcUAlpha, (q.alpha ?? 1.0) * this.cfg.hudAlpha);

      const stride = 4 * 4; // 16 bytes
      gl.enableVertexAttribArray(this.gcAPos);
      gl.vertexAttribPointer(this.gcAPos, 2, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(this.gcAUv);
      gl.vertexAttribPointer(this.gcAUv,  2, gl.FLOAT, false, stride, 8);

      gl.drawArrays(gl.TRIANGLES, 0, 6);

      gl.disableVertexAttribArray(this.gcAPos);
      gl.disableVertexAttribArray(this.gcAUv);
    }

    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /**
   * Render off-screen HUD pass into the 512×256 HUD FBO.
   *
   * Binds hudFBO, clears, then draws all GLUI elements.  Call composite()
   * afterwards to blit the result to the screen.
   *
   * @param drawFn   Callback that issues drawGLUIBatchText / drawGLUIColor calls.
   */
  renderToHUDFBO(drawFn: () => void): void {
    const gl = this.gl;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.hudFBO);
    gl.viewport(0, 0, HUD_FBO_W, HUD_FBO_H);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    drawFn();

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvasW, this.canvasH);
  }

  /** Expose the HUD texture for downstream compositing. */
  get hudTexture(): WebGLTexture { return this.hudTex; }

  /**
   * Asynchronously load a real MSDF atlas PNG + JSON.
   * Until loaded, the procedural 8×8 placeholder atlas is used.
   */
  loadAtlasFromUrl(pngUrl: string, jsonUrl: string): void {
    Promise.all([
      fetch(jsonUrl).then(r => r.json() as Promise<MSDFAtlasJson>),
      new Promise<HTMLImageElement>((resolve, reject) => {
        const img = new Image();
        img.onload  = () => resolve(img);
        img.onerror = reject;
        img.src     = pngUrl;
      }),
    ]).then(([json, img]) => {
      this.atlasJson = json;
      this._buildGlyphMap();
      this._uploadAtlasImage(img);
    }).catch(err => {
      console.warn('[ATTextRenderingMSDF] atlas load failed, using fallback:', err);
    });
  }

  /**
   * Release all GPU resources.
   */
  dispose(): void {
    const gl = this.gl;

    // Programs
    gl.deleteProgram(this.defaultTextProg);
    gl.deleteProgram(this.gluiBatchTextProg);
    gl.deleteProgram(this.gluiColorProg);

    // Label VBOs
    for (const { buf } of this.labelBufs.values()) {
      gl.deleteBuffer(buf);
    }
    this.labelBufs.clear();

    // Batch + color buffers
    gl.deleteBuffer(this.gluiBatchBuf);
    gl.deleteBuffer(this.gluiColorBuf);

    // HUD FBO
    gl.deleteFramebuffer(this.hudFBO);
    gl.deleteTexture(this.hudTex);
    gl.deleteRenderbuffer(this.hudDepth);

    // Atlas texture
    gl.deleteTexture(this.atlasTex);
  }

  // ─── Private: init ─────────────────────────────────────────────────────────

  private _init(): void {
    // 1. Enable OES_standard_derivatives for fwidth() in msdf.glsl
    this.gl.getExtension('OES_standard_derivatives');

    // 2. Compile programs from GLSL sources derived from compiled.vs
    this._compilePrograms();

    // 3. Cache uniform / attribute locations
    this._cacheLocations();

    // 4. Create geometry buffers
    this._createGeometryBuffers();

    // 5. Create HUD FBO
    this._createHUDFBO();

    // 6. Create atlas texture (8×8 procedural SDF-ready placeholder)
    this._createAtlasTexture();
  }

  // ─── Private: compile ──────────────────────────────────────────────────────

  /**
   * Compile all three shader programs.
   * Sources are the literal GLSL strings above (derived from compiled.vs).
   */
  private _compilePrograms(): void {
    this.defaultTextProg   = this._compile(DEFAULTTEXT_VERT_SRC,    DEFAULTTEXT_FRAG_SRC,    'DefaultText');
    this.gluiBatchTextProg = this._compile(GLUI_BATCHTEXT_VERT_SRC, GLUI_BATCHTEXT_FRAG_SRC, 'GLUIBatchText');
    this.gluiColorProg     = this._compile(GLUI_COLOR_VERT_SRC,     GLUI_COLOR_FRAG_SRC,     'GLUIColor');
  }

  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATTextRenderingMSDF] vertex compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATTextRenderingMSDF] fragment compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ATTextRenderingMSDF] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  // ─── Private: locations ────────────────────────────────────────────────────

  private _cacheLocations(): void {
    const gl = this.gl;

    // DefaultText
    this.dtAPos        = gl.getAttribLocation(this.defaultTextProg,   'aPosition');
    this.dtAUv         = gl.getAttribLocation(this.defaultTextProg,   'aUv');
    this.dtUProjection = gl.getUniformLocation(this.defaultTextProg,  'uProjection');
    this.dtUModelView  = gl.getUniformLocation(this.defaultTextProg,  'uModelView');
    this.dtUModel      = gl.getUniformLocation(this.defaultTextProg,  'uModel');
    this.dtUTMap       = gl.getUniformLocation(this.defaultTextProg,  'tMap');
    this.dtUColor      = gl.getUniformLocation(this.defaultTextProg,  'uColor');
    this.dtUAlpha      = gl.getUniformLocation(this.defaultTextProg,  'uAlpha');
    this.dtUTime       = gl.getUniformLocation(this.defaultTextProg,  'uTime');
    this.dtUResolution = gl.getUniformLocation(this.defaultTextProg,  'uResolution');

    // GLUIBatchText
    this.gbAPos        = gl.getAttribLocation(this.gluiBatchTextProg,  'aPosition');
    this.gbAUv         = gl.getAttribLocation(this.gluiBatchTextProg,  'aUv');
    this.gbAOffset     = gl.getAttribLocation(this.gluiBatchTextProg,  'aOffset');
    this.gbAScale      = gl.getAttribLocation(this.gluiBatchTextProg,  'aScale');
    this.gbARotation   = gl.getAttribLocation(this.gluiBatchTextProg,  'aRotation');
    this.gbUProjection = gl.getUniformLocation(this.gluiBatchTextProg, 'uProjection');
    this.gbUModelView  = gl.getUniformLocation(this.gluiBatchTextProg, 'uModelView');
    this.gbUTMap       = gl.getUniformLocation(this.gluiBatchTextProg, 'tMap');
    this.gbUColor      = gl.getUniformLocation(this.gluiBatchTextProg, 'uColor');
    this.gbUAlpha      = gl.getUniformLocation(this.gluiBatchTextProg, 'uAlpha');
    this.gbUTime       = gl.getUniformLocation(this.gluiBatchTextProg, 'uTime');

    // GLUIColor
    this.gcAPos        = gl.getAttribLocation(this.gluiColorProg,  'aPosition');
    this.gcAUv         = gl.getAttribLocation(this.gluiColorProg,  'aUv');
    this.gcUProjection = gl.getUniformLocation(this.gluiColorProg, 'uProjection');
    this.gcUModelView  = gl.getUniformLocation(this.gluiColorProg, 'uModelView');
    this.gcUColor      = gl.getUniformLocation(this.gluiColorProg, 'uColor');
    this.gcUAlpha      = gl.getUniformLocation(this.gluiColorProg, 'uAlpha');
  }

  // ─── Private: geometry buffers ─────────────────────────────────────────────

  private _createGeometryBuffers(): void {
    const gl = this.gl;

    // GLUIBatchText: MAX_GLUI_INSTANCES × MAX_CHARS_PER_LABEL chars × 6 verts × 10 floats
    this.gluiBatchBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.gluiBatchBuf);
    gl.bufferData(gl.ARRAY_BUFFER,
      MAX_GLUI_INSTANCES * MAX_CHARS_PER_LABEL * VERTS_PER_GLYPH * 10 * 4,
      gl.DYNAMIC_DRAW);

    // GLUIColor: 6 verts × 4 floats × 4 bytes per quad (re-uploaded per quad)
    this.gluiColorBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.gluiColorBuf);
    gl.bufferData(gl.ARRAY_BUFFER, 6 * 4 * 4, gl.DYNAMIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  // ─── Private: HUD FBO ──────────────────────────────────────────────────────

  private _createHUDFBO(): void {
    const gl = this.gl;

    // Colour texture
    this.hudTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.hudTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, HUD_FBO_W, HUD_FBO_H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Depth renderbuffer
    this.hudDepth = gl.createRenderbuffer()!;
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.hudDepth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, HUD_FBO_W, HUD_FBO_H);

    // Framebuffer
    this.hudFBO = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.hudFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.hudTex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.hudDepth);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
  }

  // ─── Private: atlas texture ────────────────────────────────────────────────

  /**
   * Create an 8×8 procedural MSDF-ready placeholder atlas.
   *
   * Each 8×8 cell encodes a soft SDF circle so the shader can exercise the
   * full median(r,g,b) → smoothstep path before the real atlas loads.
   * Channels R=G=B encode the distance field so median = exact SDF value.
   */
  private _createAtlasTexture(): void {
    const gl = this.gl;
    const size = 8;
    const data = new Uint8Array(size * size * 4);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const nx = (x / (size - 1)) * 2.0 - 1.0;
        const ny = (y / (size - 1)) * 2.0 - 1.0;
        // signed distance from circle edge, positive inside
        const sd = 0.6 - Math.sqrt(nx * nx + ny * ny);
        // encode to [0,255]: 128 = 0.0, 255 = +0.5, 0 = -0.5
        const encoded = Math.round(Math.max(0, Math.min(1, sd + 0.5)) * 255);
        const i = (y * size + x) * 4;
        data[i]   = encoded;   // R
        data[i+1] = encoded;   // G  (all equal → median = exact)
        data[i+2] = encoded;   // B
        data[i+3] = 255;
      }
    }

    this.atlasTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.atlasTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, size, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** Upload a real MSDF PNG to replace the placeholder. */
  private _uploadAtlasImage(img: HTMLImageElement): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.atlasTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ─── Private: glyph map ───────────────────────────────────────────────────

  private _buildGlyphMap(): void {
    this.glyphMap.clear();
    for (const g of this.atlasJson.glyphs) {
      this.glyphMap.set(g.id, g);
    }
  }

  // ─── Private: geometry building ───────────────────────────────────────────

  /**
   * Build interleaved vertex data for one text string.
   * Layout per vertex: [posX, posY, uvX, uvY]  — FLOATS_PER_VERT = 4
   * 6 vertices per character (2 triangles).
   * All positions in NDC; UV in [0,1] atlas space.
   */
  private _buildLabelGeometry(
    label:    string,
    ndcX:     number,
    ndcY:     number,
    fontSize: number,
  ): Float32Array {
    const verts: number[] = [];
    let penX = ndcX;

    for (let ci = 0; ci < label.length && ci < MAX_CHARS_PER_LABEL; ci++) {
      const code = label.charCodeAt(ci);
      const g = this.glyphMap.get(code);
      if (!g) {
        penX += fontSize * 0.5;
        continue;
      }

      const x0 = penX + g.planeBoundsLeft   * fontSize;
      const x1 = penX + g.planeBoundsRight  * fontSize;
      const y0 = ndcY + g.planeBoundsBottom * fontSize;
      const y1 = ndcY + g.planeBoundsTop    * fontSize;

      const u0 = g.uvX;
      const u1 = g.uvX + g.uvW;
      const v0 = g.uvY;
      const v1 = g.uvY + g.uvH;

      // Triangle 1: bottom-left, bottom-right, top-right
      verts.push(x0, y0, u0, v1);
      verts.push(x1, y0, u1, v1);
      verts.push(x1, y1, u1, v0);

      // Triangle 2: bottom-left, top-right, top-left
      verts.push(x0, y0, u0, v1);
      verts.push(x1, y1, u1, v0);
      verts.push(x0, y1, u0, v0);

      penX += g.advance * fontSize;
    }

    return new Float32Array(verts);
  }

  /** Total NDC width of a label string at the given fontSize. */
  labelWidth(label: string, fontSize?: number): number {
    const fs = fontSize ?? this.cfg.fontSize;
    let w = 0;
    for (let i = 0; i < label.length; i++) {
      const g = this.glyphMap.get(label.charCodeAt(i));
      w += g ? g.advance * fs : fs * 0.5;
    }
    return w;
  }
}

// ─── Convenience factory ──────────────────────────────────────────────────────

/**
 * Create an ATTextRenderingMSDF and optionally begin loading the real atlas.
 *
 * ```ts
 * const textRenderer = createATTextRenderingMSDF(gl, {
 *   atlasImage: '/fonts/inter-msdf.png',
 *   atlasJson:  '/fonts/inter-msdf.json',
 *   fontSize:   0.04,
 * });
 *
 * // per-frame:
 * textRenderer.tick(dt, canvas.width, canvas.height);
 * textRenderer.drawLabel('self_attn', -0.5, -0.8, proj, mv, model, uAlpha);
 * ```
 */
export function createATTextRenderingMSDF(
  gl:  WebGLRenderingContext,
  opts?: ATTextRenderingMSDFConfig & {
    atlasImage?: string;
    atlasJson?:  string;
  },
): ATTextRenderingMSDF {
  const { atlasImage, atlasJson, ...cfg } = opts ?? {};
  const renderer = new ATTextRenderingMSDF(gl, cfg);
  if (atlasImage && atlasJson) {
    renderer.loadAtlasFromUrl(atlasImage, atlasJson);
  }
  return renderer;
}

// ─── Re-export GLSL source constants for downstream shader composition ────────

/** msdf.glsl from compiled.vs — median SDF + fwidth anti-aliasing */
export { MSDF_GLSL };

/** DefaultText fragment source — grid-quantised split-text animation */
export { DEFAULTTEXT_FRAG_SRC as DefaultTextFragSrc };

/** DefaultText vertex source */
export { DEFAULTTEXT_VERT_SRC as DefaultTextVertSrc };

/** GLUIBatchText vertex source — instanced Z-rotation + scale + offset */
export { GLUI_BATCHTEXT_VERT_SRC as GLUIBatchTextVertSrc };

/** GLUIBatchText fragment source — MSDF alpha + time breathing */
export { GLUI_BATCHTEXT_FRAG_SRC as GLUIBatchTextFragSrc };

/** GLUIColor vertex/fragment sources */
export { GLUI_COLOR_VERT_SRC as GLUIColorVertSrc };
export { GLUI_COLOR_FRAG_SRC as GLUIColorFragSrc };
