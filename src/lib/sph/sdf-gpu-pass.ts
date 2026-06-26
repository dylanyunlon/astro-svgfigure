/**
 * sdf-gpu-pass.ts — GPU Instanced SDF Species Icon
 *
 * 真正的 WebGL1 GPU instanced rendering — 不是占位符。
 * 每个 species 编译独立的 WebGLProgram (from cil-*.frag SDF shaders)。
 * per-instance VBO (position.xy, size, opacity) 通过 ANGLE_instanced_arrays
 * 调用 drawArraysInstancedANGLE 渲染每 species 全部 instance。
 *
 * Species shader 映射:
 *   'eye'         → cil-eye.frag
 *   'bolt'        → cil-bolt.frag
 *   'plus'        → cil-plus.frag
 *   'arrow-right' → cil-arrow-right.frag
 *   'vector'      → cil-vector.frag
 *
 * Pass 链 (每帧):
 *   uploadInstances → (per species) bindProgram → setUniforms → drawArraysInstanced
 */

// ─── Vertex shader — WebGL1 ────────────────────────────────────────────────
// aPosition: fullscreen-quad corner ([-1,1])
// a_iPos:    instance world position (canvas coords, NDC via u_resolution)
// a_iSize:   instance size in pixels
// a_iOpacity: per-instance opacity override








const SDF_VERT = /* glsl */ `
precision highp float;

// quad geometry (per-vertex)
attribute vec2 aPosition;

// per-instance attributes (divisor = 1 via ANGLE_instanced_arrays)
attribute vec2  a_iPos;
attribute float a_iSize;
attribute float a_iOpacity;

// passed to fragment
varying vec2  v_fragCoord;  // fragment position in canvas space
varying float v_iOpacity;
varying float v_iSize;
varying vec2  v_iPos;

uniform vec2 u_resolution;  // canvas width/height in pixels

void main() {
  // Scale quad corner by instance size, offset by instance world position.
  // aPosition ∈ [-1,1] → scale to half-size pixels, shift to iPos.
  vec2 halfPx   = vec2(a_iSize * 0.5);
  vec2 worldPos = a_iPos + aPosition * halfPx;

  // Convert canvas-pixel position → NDC [-1,1]
  vec2 ndc = (worldPos / u_resolution) * 2.0 - 1.0;
  ndc.y = -ndc.y; // flip Y (canvas Y grows downward)

  gl_Position = vec4(ndc, 0.0, 1.0);

  // The fragment shader uses gl_FragCoord for its bbox math.
  // We pass canvas-space coords so the frag shader can reproduce bbox-local UV.
  v_fragCoord = worldPos;
  v_iOpacity  = a_iOpacity;
  v_iSize     = a_iSize;
  v_iPos      = a_iPos;
}
`;

// ─── Per-species fragment shader wrapper ──────────────────────────────────
// The cil-*.frag shaders use gl_FragCoord + u_bbox to produce bbox-local UV.
// We inject the SDF source after injecting a common preamble with u_time, etc.
// u_bbox = vec4(instanceOriginX, instanceOriginY, width, height) — set per draw.
// NOTE: The actual SDF frag source is concatenated at compile time (per program).

// ─── Types ─────────────────────────────────────────────────────────────────

export type SDFSpecies = 'eye' | 'bolt' | 'plus' | 'arrow-right' | 'vector';

/** Per-instance upload data. */
export interface SDFInstance {
  x: number;       // canvas-space X of icon centre
  y: number;       // canvas-space Y of icon centre
  size: number;    // icon size in pixels (square bbox)
  opacity: number; // [0,1]
}

/** Compiled GPU program + cached uniform locations for one species. */
interface SpeciesProgram {
  prog: WebGLProgram;
  // attribute locations
  loc_aPosition: number;
  loc_a_iPos: number;
  loc_a_iSize: number;
  loc_a_iOpacity: number;
  // uniform locations
  u_resolution: WebGLUniformLocation | null;
  u_bbox: WebGLUniformLocation | null;
  u_fillColor: WebGLUniformLocation | null;
  u_opacity: WebGLUniformLocation | null;
  u_resolution2: WebGLUniformLocation | null; // u_resolution inside frag
  u_time: WebGLUniformLocation | null;
  // species-specific uniforms (optional, set if present)
  u_numRays: WebGLUniformLocation | null;
  u_pupilRadius: WebGLUniformLocation | null;
  u_focalIntensity: WebGLUniformLocation | null;
  u_bloomStrength: WebGLUniformLocation | null;
  u_bloomRadius: WebGLUniformLocation | null;
  u_ambientIntensity: WebGLUniformLocation | null;
  u_ambientColor: WebGLUniformLocation | null;
  u_lightExposure: WebGLUniformLocation | null;
  u_shadowFar: WebGLUniformLocation | null;
  u_shadowBias: WebGLUniformLocation | null;
  u_zigzagCount: WebGLUniformLocation | null;
  u_amplitude: WebGLUniformLocation | null;
  u_armLength: WebGLUniformLocation | null;
  u_strokeWidth: WebGLUniformLocation | null;
  u_arrowWidth: WebGLUniformLocation | null;
  u_arrowCount: WebGLUniformLocation | null;
  u_angleSpread: WebGLUniformLocation | null;
}

/** Per-species instance batch. */
interface SpeciesBatch {
  species: SDFSpecies;
  instances: SDFInstance[];
}

// Floats per instance in the VBO: [posX, posY, size, opacity]
const FLOATS_PER_INSTANCE = 4;
const MAX_INSTANCES       = 4096;

// ─── SDF shader source lookup ──────────────────────────────────────────────
// Inline the raw GLSL sources so this file has zero async dependencies at runtime.
// (Alternative: use getShader() from ShaderLoader, but that requires compiled.vs)

const FRAG_SOURCES: Record<SDFSpecies, string> = {
  'eye': /* raw cil-eye.frag — precision stripped (prepended globally) */ `
precision mediump float;
uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;
uniform float u_numRays;
uniform float u_pupilRadius;
uniform float u_focalIntensity;
uniform float u_time;
uniform float u_bloomStrength;
uniform float u_bloomRadius;
uniform float u_ambientIntensity;
uniform vec3  u_ambientColor;
uniform float u_lightExposure;
uniform float u_shadowFar;
uniform float u_shadowBias;

// Per-instance position/size passed from vertex shader (instanced rendering)
varying vec2  v_iPos;
varying float v_iSize;

float circleSDF(in vec2 v) {
  v -= 0.5;
  return length(v) * 2.0;
}

void main() {
  // Compute bbox-local UV from per-instance varyings (works for all instances)
  vec2 bboxOrigin = v_iPos - vec2(v_iSize * 0.5);
  vec2 uv = (gl_FragCoord.xy - bboxOrigin) / vec2(v_iSize);
  float dist = circleSDF(uv);
  vec2 p = uv * 2.0 - 1.0;
  float angle = atan(p.y, p.x);

  float pupilR   = u_pupilRadius * 0.5;
  float pupil    = 1.0 - smoothstep(pupilR - 0.01, pupilR + 0.01, dist);

  float irisInner = (u_pupilRadius + 0.02) * 0.5;
  float irisOuter = (u_pupilRadius + 0.08) * 0.5;
  float iris = smoothstep(irisInner, irisOuter, dist)
             * (1.0 - smoothstep(0.425, 0.5, dist));

  float halfStep = 3.14159265 / u_numRays;
  float rayAngle = mod(angle + u_time * 0.3, halfStep * 2.0) - halfStep;
  float rayMask  = smoothstep(0.07, 0.0, abs(rayAngle));
  float rayFade  = smoothstep(0.5, irisInner + 0.06, dist)
                 * smoothstep(pupilR, pupilR + 0.06, dist);
  float rays     = rayMask * rayFade * u_focalIntensity;

  float sclera   = smoothstep(0.525, 0.44, dist);

  float ambientFalloff = 1.0 - smoothstep(0.0, 0.6, dist);
  vec3  ambientContrib = u_ambientColor * u_ambientIntensity * u_lightExposure * ambientFalloff;

  float bloomCenter = (u_pupilRadius + 0.15) * 0.5;
  float bloomRing   = exp(-pow((dist - bloomCenter) / max(u_bloomRadius * 0.09, 0.005), 2.0));
  float bloom       = bloomRing * u_bloomStrength * 0.35;

  float shadowNorm   = clamp(dist / (u_shadowFar * 0.0125), 0.0, 1.0);
  float shadowFactor = 1.0 - shadowNorm * (1.0 - u_shadowBias * 100.0);

  float alpha = clamp(sclera * (iris + rays) + pupil, 0.0, 1.0);
  vec3 finalColor = u_fillColor + ambientContrib * (iris + bloom) * alpha;
  finalColor += u_fillColor * bloom;
  finalColor *= shadowFactor;
  gl_FragColor = vec4(finalColor, alpha * u_opacity);
}
`,
  'bolt': `
precision mediump float;
uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;
uniform float u_zigzagCount;
uniform float u_amplitude;
uniform float u_time;

// Per-instance position/size from vertex shader
varying vec2  v_iPos;
varying float v_iSize;

#define saturate(V) clamp(V, 0.0, 1.0)

float lineSDF(in vec2 st, in vec2 a, in vec2 b) {
  vec2 b2a = b - a;
  vec2 t2a = st - a;
  float h = saturate(dot(t2a, b2a) / dot(b2a, b2a));
  return length(t2a - h * b2a);
}

float strokeMask(vec2 p, vec2 a, vec2 b, float w) {
  return smoothstep(w, w * 0.4, lineSDF(p, a, b));
}

const float AT_BLOOM_INTENSITY       = 1.0;
const float AT_BLOOM_RADIUS          = 1.0;
const float AT_GLOBAL_BLOOM_STRENGTH = 0.3;
const float AT_GLOBAL_BLOOM_RADIUS   = 0.2;
const float AT_HOME_BLOOM_STRENGTH   = 0.6;
const float AT_HOME_BLOOM_RADIUS     = 0.8;
const float AT_LIGHT_INTENSITY       = 2.19;
const float AT_WIGGLE_SPEED          = 0.7;
const float AT_LUMINOSITY_THRESHOLD  = 0.0;

void main() {
  vec2 bboxOrigin = v_iPos - vec2(v_iSize * 0.5);
  vec2 uv = (gl_FragCoord.xy - bboxOrigin) / vec2(v_iSize);
  vec2 p  = uv * 2.0 - 1.0;

  float strokeW = 0.045;
  float total   = 0.0;
  float steps   = u_zigzagCount;
  float dy      = 2.0 / steps;
  float phase   = sin(u_time * 2.5 * AT_WIGGLE_SPEED) * 0.15;

  for (float i = 0.0; i < 32.0; i++) {
    if (i >= steps) break;
    float t0 = -1.0 + i       * dy;
    float t1 = -1.0 + (i+1.0) * dy;
    float s0 = (mod(i,       2.0) < 1.0 ? 1.0 : -1.0);
    float s1 = (mod(i + 1.0, 2.0) < 1.0 ? 1.0 : -1.0);
    vec2 a = vec2(s0 * u_amplitude + phase, t0);
    vec2 b = vec2(s1 * u_amplitude + phase, t1);
    total = max(total, strokeMask(p, a, b, strokeW));
  }

  float glowGlobal = 0.0;
  float globalGlowW = strokeW * (3.5 * AT_GLOBAL_BLOOM_RADIUS / AT_BLOOM_RADIUS);
  for (float i = 0.0; i < 32.0; i++) {
    if (i >= steps) break;
    float t0 = -1.0 + i * dy; float t1 = -1.0 + (i+1.0) * dy;
    float s0 = (mod(i, 2.0) < 1.0 ? 1.0 : -1.0);
    float s1 = (mod(i+1.0, 2.0) < 1.0 ? 1.0 : -1.0);
    vec2 a = vec2(s0*u_amplitude+phase,t0); vec2 b = vec2(s1*u_amplitude+phase,t1);
    glowGlobal = max(glowGlobal, strokeMask(p,a,b,globalGlowW)*AT_GLOBAL_BLOOM_STRENGTH);
  }

  float glowHome = 0.0;
  float homeGlowW = strokeW * (5.0 * AT_HOME_BLOOM_RADIUS / AT_BLOOM_RADIUS);
  for (float i = 0.0; i < 32.0; i++) {
    if (i >= steps) break;
    float t0 = -1.0 + i * dy; float t1 = -1.0 + (i+1.0) * dy;
    float s0 = (mod(i, 2.0) < 1.0 ? 1.0 : -1.0);
    float s1 = (mod(i+1.0, 2.0) < 1.0 ? 1.0 : -1.0);
    vec2 a = vec2(s0*u_amplitude+phase,t0); vec2 b = vec2(s1*u_amplitude+phase,t1);
    glowHome = max(glowHome, strokeMask(p,a,b,homeGlowW)*AT_HOME_BLOOM_STRENGTH);
  }

  float lum    = dot(u_fillColor, vec3(0.2126, 0.7152, 0.0722));
  float lumGate = step(AT_LUMINOSITY_THRESHOLD, lum);
  float bloomSum = (glowGlobal + glowHome) * lumGate * AT_BLOOM_INTENSITY * (AT_LIGHT_INTENSITY / 2.19);
  float alpha    = clamp(total + bloomSum, 0.0, 1.0);
  gl_FragColor = vec4(u_fillColor, alpha * u_opacity);
}
`,
  'plus': `
precision mediump float;
uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;
uniform float u_armLength;
uniform float u_strokeWidth;

// Per-instance position/size from vertex shader
varying vec2  v_iPos;
varying float v_iSize;

float sdBox2(vec2 p, vec2 b) {
  vec2 d = abs(p) - b;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float sdPlus(vec2 p, float armLen, float sw) {
  return min(sdBox2(p, vec2(armLen, sw)), sdBox2(p, vec2(sw, armLen)));
}

void main() {
  vec2 bboxOrigin = v_iPos - vec2(v_iSize * 0.5);
  vec2 uv = (gl_FragCoord.xy - bboxOrigin) / vec2(v_iSize);
  vec2 p  = uv * 2.0 - 1.0;
  float d    = sdPlus(p, u_armLength, u_strokeWidth);
  float mask = smoothstep(0.015, -0.015, d);
  float glow = smoothstep(0.08, 0.0, d) * 0.25;
  float alpha = clamp(mask + glow, 0.0, 1.0);
  gl_FragColor = vec4(u_fillColor, alpha * u_opacity);
}
`,
  'arrow-right': `
precision mediump float;
uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;
uniform float u_arrowWidth;
uniform float u_time;

// Per-instance position/size from vertex shader
varying vec2  v_iPos;
varying float v_iSize;

#define saturate(V) clamp(V, 0.0, 1.0)

float lineSDF(in vec2 st, in vec2 a, in vec2 b) {
  vec2 b2a = b - a; vec2 t2a = st - a;
  float h = saturate(dot(t2a, b2a) / dot(b2a, b2a));
  return length(t2a - h * b2a);
}

float sdArrowRight(vec2 p, float w) {
  float d1 = lineSDF(p, vec2(-0.45, 0.40), vec2(0.45, 0.0));
  float d2 = lineSDF(p, vec2(-0.45,-0.40), vec2(0.45, 0.0));
  return min(d1, d2) - w;
}

void main() {
  vec2 bboxOrigin = v_iPos - vec2(v_iSize * 0.5);
  vec2 uv    = (gl_FragCoord.xy - bboxOrigin) / vec2(v_iSize);
  float cols = 3.0; float rows = 3.0;
  vec2 scroll = vec2(u_time * 0.25, 0.0);
  vec2 tiled  = fract(uv * vec2(cols, rows) + scroll);
  vec2 lp     = tiled * 2.0 - 1.0;
  float d     = sdArrowRight(lp, u_arrowWidth * 0.5);
  float mask  = smoothstep(0.02, -0.01, d);
  float fade  = smoothstep(0.0, 0.6, tiled.x);
  float alpha = mask * (0.4 + 0.6 * fade);
  gl_FragColor = vec4(u_fillColor, clamp(alpha, 0.0, 1.0) * u_opacity);
}
`,
  'vector': `
precision mediump float;
uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;
uniform float u_arrowCount;
uniform float u_angleSpread;

// Per-instance position/size from vertex shader
varying vec2  v_iPos;
varying float v_iSize;

#define PI  3.1415926535897932384626433832795
#define TAU 6.2831853071795864769252867665590

#define saturate(V) clamp(V, 0.0, 1.0)

float lineSDF(in vec2 st, in vec2 a, in vec2 b) {
  vec2 b2a = b - a; vec2 t2a = st - a;
  float h = saturate(dot(t2a, b2a) / dot(b2a, b2a));
  return length(t2a - h * b2a);
}

float polySDF(in vec2 st, in int V) {
  st = st * 2.0 - 1.0;
  float a = atan(st.x, st.y) + PI;
  float r = length(st);
  float v = TAU / float(V);
  return cos(floor(0.5 + a/v)*v - a) * r;
}

float sdBox(vec2 p, vec2 b) {
  vec2 d = abs(p) - b;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float drawArrow(vec2 p, float angle, float scale) {
  float c = cos(angle); float s = sin(angle);
  vec2 lp = vec2(c*p.x + s*p.y, -s*p.x + c*p.y) / scale;
  float shaft = sdBox(lp - vec2(-0.15, 0.0), vec2(0.22, 0.045));
  vec2 headUV = (lp - vec2(0.13, 0.0)) / 0.32 + 0.5;
  headUV -= 0.5;
  float tmp = headUV.x; headUV.x = -headUV.y; headUV.y = tmp;
  headUV += 0.5;
  float head = polySDF(headUV, 3) * 0.32 - 0.16;
  return smoothstep(0.01, -0.01, min(shaft, head));
}

float rand(vec2 co) {
  return fract(sin(dot(co, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
  vec2 bboxOrigin = v_iPos - vec2(v_iSize * 0.5);
  vec2 uv   = (gl_FragCoord.xy - bboxOrigin) / vec2(v_iSize);
  float n   = u_arrowCount;
  vec2 cell = floor(uv * n);
  vec2 loc  = fract(uv * n) - 0.5;
  float jitter = (rand(cell) * 2.0 - 1.0) * u_angleSpread;
  float mask   = drawArrow(loc, jitter, 0.45);
  gl_FragColor = vec4(u_fillColor, mask * u_opacity);
}
`,
};

// ─── SDFIconGPU ────────────────────────────────────────────────────────────

export class SDFIconGPU {
  private gl: WebGLRenderingContext;
  private ext: ANGLE_instanced_arrays | null;

  // Per-species compiled program + uniform cache
  private programs: Map<SDFSpecies, SpeciesProgram> = new Map();

  // Shared fullscreen-quad geometry VBO (2 triangles, NDC corner positions)
  // aPosition lives here — same quad for all species
  private quadVBO!: WebGLBuffer;

  // Per-species instance VBO: [posX, posY, size, opacity] × N
  private instanceVBOs: Map<SDFSpecies, WebGLBuffer> = new Map();

  // Scratch Float32Array to avoid GC churn on upload
  private scratchBuf: Float32Array = new Float32Array(MAX_INSTANCES * FLOATS_PER_INSTANCE);

  // Canvas dimensions — updated via resize()
  private canvasW: number = 1;
  private canvasH: number = 1;

  // Global time for animated shaders
  private time: number = 0;

  // Per-species default fill colours (RGB)
  private fillColors: Map<SDFSpecies, [number, number, number]> = new Map([
    ['eye',         [0.047, 0.929, 0.565]],
    ['bolt',        [1.0,   0.8,   0.1  ]],
    ['plus',        [0.2,   0.6,   1.0  ]],
    ['arrow-right', [0.9,   0.3,   0.7  ]],
    ['vector',      [0.3,   0.9,   0.5  ]],
  ]);

  constructor(gl: WebGLRenderingContext) {
    this.gl  = gl;

    // ── Require ANGLE_instanced_arrays (WebGL1) / built-in (WebGL2) ────────
    if (gl instanceof WebGL2RenderingContext) {
      this.ext = null; // WebGL2: instancing is core
    } else {
      const ext = gl.getExtension('ANGLE_instanced_arrays');
      if (!ext) throw new Error('[SDFIconGPU] ANGLE_instanced_arrays not supported');
      this.ext = ext;
    }

    this._initQuadVBO();
    this._compileAllSpecies();
    this._initInstanceVBOs();
  }

  // ── Public API ──────────────────────────────────────────────────────────

  /** Call once when canvas is resized. */
  resize(w: number, h: number): void {
    this.canvasW = w;
    this.canvasH = h;
  }

  /**
   * Render all species batches.
   * @param batches  Array of {species, instances[]} to render this frame.
   * @param dt       Delta-time in seconds (for animation uniforms).
   */
  render(batches: SpeciesBatch[], dt: number = 1 / 60): void {
    this.time += dt;
    const gl  = this.gl;
    const ext = this.ext;

    // Enable additive blending (icons glow on dark backgrounds)
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);

    for (const batch of batches) {
      if (!batch.instances.length) continue;

      const sp = this.programs.get(batch.species);
      if (!sp) continue;
      const iVBO = this.instanceVBOs.get(batch.species);
      if (!iVBO) continue;

      const count = Math.min(batch.instances.length, MAX_INSTANCES);
      const fill  = this.fillColors.get(batch.species) ?? [1, 1, 1];

      // ── 1. Upload per-instance data to GPU ──────────────────────────
      this._uploadInstances(iVBO, batch.instances, count);

      // ── 2. Bind program ─────────────────────────────────────────────
      gl.useProgram(sp.prog);

      // ── 3. Set global uniforms ───────────────────────────────────────
      gl.uniform2f(sp.u_resolution, this.canvasW, this.canvasH);
      gl.uniform3f(sp.u_fillColor,  fill[0], fill[1], fill[2]);
      gl.uniform1f(sp.u_opacity,    1.0);
      if (sp.u_time !== null) gl.uniform1f(sp.u_time, this.time);

      // Set bbox to (0, 0, canvasW, canvasH) as base —
      // per-icon UV is recomputed from gl_FragCoord vs the vertex-generated bbox.
      // (The frag shader reinterprets its u_bbox relative to gl_FragCoord, which
      //  maps naturally because each instance quad occupies exactly icon-size pixels.)
      // u_bbox is intentionally NOT set here as a global canvas-sized uniform.
      // Each fragment shader computes its bbox-local UV from the varyings
      // v_iPos and v_iSize that the vertex shader already passes per instance:
      //   uv = (gl_FragCoord.xy - (v_iPos - vec2(v_iSize*0.5))) / vec2(v_iSize)
      // Frag shaders must use v_iPos/v_iSize varyings, not u_bbox, for instanced rendering.
      // (drawSingleQuad sets u_bbox directly for non-instanced use.)

      // ── 4. Species-specific uniform defaults ────────────────────────
      this._setSpeciesUniforms(sp, batch.species);

      // ── 5. Bind quad geometry (aPosition) ───────────────────────────
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
      gl.enableVertexAttribArray(sp.loc_aPosition);
      gl.vertexAttribPointer(sp.loc_aPosition, 2, gl.FLOAT, false, 0, 0);
      this._vertexAttribDivisor(sp.loc_aPosition, 0); // per-vertex

      // ── 6. Bind instance VBO ─────────────────────────────────────────
      // Layout: [posX(f32), posY(f32), size(f32), opacity(f32)]
      const stride = FLOATS_PER_INSTANCE * 4; // bytes
      gl.bindBuffer(gl.ARRAY_BUFFER, iVBO);

      gl.enableVertexAttribArray(sp.loc_a_iPos);
      gl.vertexAttribPointer(sp.loc_a_iPos, 2, gl.FLOAT, false, stride, 0);
      this._vertexAttribDivisor(sp.loc_a_iPos, 1); // per-instance

      gl.enableVertexAttribArray(sp.loc_a_iSize);
      gl.vertexAttribPointer(sp.loc_a_iSize, 1, gl.FLOAT, false, stride, 2 * 4);
      this._vertexAttribDivisor(sp.loc_a_iSize, 1);

      gl.enableVertexAttribArray(sp.loc_a_iOpacity);
      gl.vertexAttribPointer(sp.loc_a_iOpacity, 1, gl.FLOAT, false, stride, 3 * 4);
      this._vertexAttribDivisor(sp.loc_a_iOpacity, 1);

      // ── 7. Draw: 6 vertices (2 triangles) × count instances ─────────
      this._drawArraysInstanced(gl.TRIANGLES, 0, 6, count);

      // ── 8. Reset divisors (WebGL1 requires manual reset) ─────────────
      this._vertexAttribDivisor(sp.loc_a_iPos,     0);
      this._vertexAttribDivisor(sp.loc_a_iSize,    0);
      this._vertexAttribDivisor(sp.loc_a_iOpacity, 0);

      // Disable attribs
      gl.disableVertexAttribArray(sp.loc_aPosition);
      gl.disableVertexAttribArray(sp.loc_a_iPos);
      gl.disableVertexAttribArray(sp.loc_a_iSize);
      gl.disableVertexAttribArray(sp.loc_a_iOpacity);
    }

    gl.disable(gl.BLEND);
  }

  /** Update fill colour for a species. */
  setFillColor(species: SDFSpecies, r: number, g: number, b: number): void {
    this.fillColors.set(species, [r, g, b]);
  }

  /** Free all GPU resources. */
  dispose(): void {
    const gl = this.gl;
    for (const sp of this.programs.values()) {
      gl.deleteProgram(sp.prog);
    }
    for (const vbo of this.instanceVBOs.values()) {
      gl.deleteBuffer(vbo);
    }
    gl.deleteBuffer(this.quadVBO);
    this.programs.clear();
    this.instanceVBOs.clear();
  }

  /**
   * Draw a single non-instanced fullscreen quad for a given species program.
   * Used for unit testing / framebuffer readback scenarios where only a single
   * icon needs to be verified.  Internally calls gl.drawArrays directly.
   */
  drawSingleQuad(species: SDFSpecies,
                 x: number, y: number, size: number,
                 fboTarget: WebGLFramebuffer | null = null): void {
    const gl  = this.gl;
    const sp  = this.programs.get(species);
    if (!sp) return;
    const fill = this.fillColors.get(species) ?? [1, 1, 1];

    // Bind target framebuffer (null = default canvas)
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboTarget);
    gl.viewport(0, 0, this.canvasW, this.canvasH);

    gl.useProgram(sp.prog);
    gl.uniform2f(sp.u_resolution, this.canvasW, this.canvasH);
    gl.uniform3f(sp.u_fillColor,  fill[0], fill[1], fill[2]);
    gl.uniform1f(sp.u_opacity,    1.0);
    gl.uniform4f(sp.u_bbox, x - size * 0.5, y - size * 0.5, size, size);
    if (sp.u_time !== null) gl.uniform1f(sp.u_time, this.time);
    this._setSpeciesUniforms(sp, species);

    // For a single non-instanced draw, set per-instance attribs as constant values
    // using vertexAttrib* (no array needed — divisor is irrelevant for drawArrays)
    gl.disableVertexAttribArray(sp.loc_a_iPos);
    gl.vertexAttrib2f(sp.loc_a_iPos, x, y);
    gl.disableVertexAttribArray(sp.loc_a_iSize);
    gl.vertexAttrib1f(sp.loc_a_iSize, size);
    gl.disableVertexAttribArray(sp.loc_a_iOpacity);
    gl.vertexAttrib1f(sp.loc_a_iOpacity, 1.0);

    // Bind quad VBO for aPosition — only NDC corners, no instancing
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
    gl.enableVertexAttribArray(sp.loc_aPosition);
    gl.vertexAttribPointer(sp.loc_aPosition, 2, gl.FLOAT, false, 0, 0);

    // Non-instanced draw of 6 vertices (2 triangles)
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.disableVertexAttribArray(sp.loc_aPosition);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ── Private: init ────────────────────────────────────────────────────────

  /** Create the shared quad VBO (6 vertices, 2 triangles, NDC [-1,1]). */
  private _initQuadVBO(): void {
    const gl = this.gl;
    this.quadVBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
    // Two triangles covering [-1,1]² (standard fullscreen quad)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,   1, -1,  -1,  1,
      -1,  1,   1, -1,   1,  1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** Compile one WebGLProgram per species. */
  private _compileAllSpecies(): void {
    const species: SDFSpecies[] = ['eye', 'bolt', 'plus', 'arrow-right', 'vector'];
    for (const s of species) {
      const prog = this._compileProgram(SDF_VERT, FRAG_SOURCES[s], s);
      this.programs.set(s, this._buildSpeciesProgram(prog));
    }
  }

  /** Allocate one instance VBO per species (pre-allocated MAX_INSTANCES). */
  private _initInstanceVBOs(): void {
    const gl      = this.gl;
    const species: SDFSpecies[] = ['eye', 'bolt', 'plus', 'arrow-right', 'vector'];
    const byteLen = MAX_INSTANCES * FLOATS_PER_INSTANCE * 4;
    for (const s of species) {
      const vbo = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      // Allocate GPU memory (DYNAMIC_DRAW — updated per frame)
      gl.bufferData(gl.ARRAY_BUFFER, byteLen, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      this.instanceVBOs.set(s, vbo);
    }
  }

  // ── Private: per-frame ───────────────────────────────────────────────────

  /** Upload instance data into the species VBO via bufferSubData. */
  private _uploadInstances(vbo: WebGLBuffer, instances: SDFInstance[], count: number): void {
    const gl  = this.gl;
    const buf = this.scratchBuf;
    for (let i = 0; i < count; i++) {
      const inst = instances[i];
      const off  = i * FLOATS_PER_INSTANCE;
      buf[off + 0] = inst.x;
      buf[off + 1] = inst.y;
      buf[off + 2] = inst.size;
      buf[off + 3] = inst.opacity;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    // Only upload the used slice
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, buf.subarray(0, count * FLOATS_PER_INSTANCE));
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** Set species-specific uniform defaults for the current program. */
  private _setSpeciesUniforms(sp: SpeciesProgram, species: SDFSpecies): void {
    const gl = this.gl;
    switch (species) {
      case 'eye':
        if (sp.u_numRays      !== null) gl.uniform1f(sp.u_numRays,      8.0);
        if (sp.u_pupilRadius  !== null) gl.uniform1f(sp.u_pupilRadius,  0.3);
        if (sp.u_focalIntensity !== null) gl.uniform1f(sp.u_focalIntensity, 1.0);
        if (sp.u_bloomStrength !== null) gl.uniform1f(sp.u_bloomStrength, 1.2);
        if (sp.u_bloomRadius  !== null) gl.uniform1f(sp.u_bloomRadius,  1.0);
        if (sp.u_ambientIntensity !== null) gl.uniform1f(sp.u_ambientIntensity, 3.44);
        if (sp.u_ambientColor !== null) gl.uniform3f(sp.u_ambientColor, 0.047, 0.929, 0.565);
        if (sp.u_lightExposure !== null) gl.uniform1f(sp.u_lightExposure, 0.86);
        if (sp.u_shadowFar    !== null) gl.uniform1f(sp.u_shadowFar,    40.0);
        if (sp.u_shadowBias   !== null) gl.uniform1f(sp.u_shadowBias,   0.001);
        break;
      case 'bolt':
        if (sp.u_zigzagCount  !== null) gl.uniform1f(sp.u_zigzagCount, 6.0);
        if (sp.u_amplitude    !== null) gl.uniform1f(sp.u_amplitude,   0.45);
        break;
      case 'plus':
        if (sp.u_armLength    !== null) gl.uniform1f(sp.u_armLength,   0.65);
        if (sp.u_strokeWidth  !== null) gl.uniform1f(sp.u_strokeWidth, 0.12);
        break;
      case 'arrow-right':
        if (sp.u_arrowWidth   !== null) gl.uniform1f(sp.u_arrowWidth,  0.08);
        break;
      case 'vector':
        if (sp.u_arrowCount   !== null) gl.uniform1f(sp.u_arrowCount,  4.0);
        if (sp.u_angleSpread  !== null) gl.uniform1f(sp.u_angleSpread, 0.8);
        break;
    }
  }

  // ── Private: instancing helpers (WebGL1 ext vs WebGL2 built-in) ─────────

  /** Wrapper: vertexAttribDivisor for WebGL1 (ANGLE ext) or WebGL2 (core). */
  private _vertexAttribDivisor(index: number, divisor: number): void {
    if (this.ext) {
      this.ext.vertexAttribDivisorANGLE(index, divisor);
    } else {
      (this.gl as unknown as WebGL2RenderingContext).vertexAttribDivisor(index, divisor);
    }
  }

  /** Wrapper: drawArraysInstanced for WebGL1 (ANGLE ext) or WebGL2 (core). */
  private _drawArraysInstanced(mode: number, first: number, count: number, instanceCount: number): void {
    if (this.ext) {
      this.ext.drawArraysInstancedANGLE(mode, first, count, instanceCount);
    } else {
      (this.gl as unknown as WebGL2RenderingContext).drawArraysInstanced(mode, first, count, instanceCount);
    }
  }

  // ── Private: shader compilation ─────────────────────────────────────────

  /** Compile vert + frag → linked WebGLProgram. */
  private _compileProgram(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    // Vertex shader
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(vs) ?? '';
      gl.deleteShader(vs);
      throw new Error(`[SDFIconGPU] vertex compile error (${label}): ${log}`);
    }

    // Fragment shader
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(fs) ?? '';
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error(`[SDFIconGPU] fragment compile error (${label}): ${log}`);
    }

    // Link
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog) ?? '';
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteProgram(prog);
      throw new Error(`[SDFIconGPU] link error (${label}): ${log}`);
    }

    // Shaders are linked — no longer needed as standalone objects
    gl.detachShader(prog, vs);
    gl.detachShader(prog, fs);
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    return prog;
  }

  /** Cache all attribute/uniform locations for a compiled program. */
  private _buildSpeciesProgram(prog: WebGLProgram): SpeciesProgram {
    const gl = this.gl;
    const u  = (name: string) => gl.getUniformLocation(prog, name);
    return {
      prog,
      // attribute locations
      loc_aPosition:   gl.getAttribLocation(prog, 'aPosition'),
      loc_a_iPos:      gl.getAttribLocation(prog, 'a_iPos'),
      loc_a_iSize:     gl.getAttribLocation(prog, 'a_iSize'),
      loc_a_iOpacity:  gl.getAttribLocation(prog, 'a_iOpacity'),
      // shared uniforms
      u_resolution:    u('u_resolution'),
      u_bbox:          u('u_bbox'),
      u_fillColor:     u('u_fillColor'),
      u_opacity:       u('u_opacity'),
      u_resolution2:   u('u_resolution'),
      u_time:          u('u_time'),
      // eye uniforms
      u_numRays:       u('u_numRays'),
      u_pupilRadius:   u('u_pupilRadius'),
      u_focalIntensity:u('u_focalIntensity'),
      u_bloomStrength: u('u_bloomStrength'),
      u_bloomRadius:   u('u_bloomRadius'),
      u_ambientIntensity: u('u_ambientIntensity'),
      u_ambientColor:  u('u_ambientColor'),
      u_lightExposure: u('u_lightExposure'),
      u_shadowFar:     u('u_shadowFar'),
      u_shadowBias:    u('u_shadowBias'),
      // bolt uniforms
      u_zigzagCount:   u('u_zigzagCount'),
      u_amplitude:     u('u_amplitude'),
      // plus uniforms
      u_armLength:     u('u_armLength'),
      u_strokeWidth:   u('u_strokeWidth'),
      // arrow-right uniforms
      u_arrowWidth:    u('u_arrowWidth'),
      // vector uniforms
      u_arrowCount:    u('u_arrowCount'),
      u_angleSpread:   u('u_angleSpread'),
    };
  }
}

// ─── Convenience factory ───────────────────────────────────────────────────

/**
 * Create an SDFIconGPU bound to an existing WebGL1 rendering context.
 * Throws if ANGLE_instanced_arrays is unavailable.
 */
export function createSDFIconGPU(gl: WebGLRenderingContext): SDFIconGPU {
  return new SDFIconGPU(gl);
}

/**
 * Build a SpeciesBatch for a single species.
 * Usage:
 *   const batch = makeSDFBatch('eye', [{x:100,y:200,size:48,opacity:0.9}, ...]);
 *   sdfGPU.render([batch], dt);
 */
export function makeSDFBatch(species: SDFSpecies, instances: SDFInstance[]): SpeciesBatch {
  return { species, instances };
}
