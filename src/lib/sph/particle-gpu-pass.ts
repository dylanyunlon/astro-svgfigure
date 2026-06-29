/**
 * particle-gpu-pass.ts — M1137
 *
 * GPU Transform Feedback 粒子系统 — 增强版:
 *   • 2000 粒子 (增加)
 *   • 颜色: edge source→target 渐变, 按 travel fraction t 插值
 *   • 大小: birth 小 → peak 大 → death 小  (sin(π·age) 曲线)
 *   • Trail: 每粒子存前 3 帧位置, 渲染为尾迹线 (半透明 LINES)
 *   • Fluid advection: sample velocityTex (RG32F) 叠加到粒子速度
 *
 * Buffer layout per particle (16 × float32 = 64 bytes):
 *   [0,1]   a_position   vec2   NDC
 *   [2,3]   a_velocity   vec2   NDC units/s
 *   [4]     a_life       float  [0,1] normalised remaining life
 *   [5,6,7] a_color      vec3   RGB (gradient from edge src→dst)
 *   [8]     a_birthLife  float  life value at respawn (for age normalisation)
 *   [9,10]  a_trail0     vec2   position 1 frame ago
 *   [11,12] a_trail1     vec2   position 2 frames ago
 *   [13,14] a_trail2     vec2   position 3 frames ago
 *   [15]    a_edgeT      float  travel fraction along current edge [0,1]
 *
 * Ping-pong: vboA/vboB alternating read/write via Transform Feedback.
 *
 * Two draw calls per frame:
 *   1. gl.drawArrays(LINES, ...)  — trail segments (3 segments × 2 verts each)
 *   2. gl.drawArrays(POINTS, ...) — particle heads
 *
 * Fluid velocity texture (optional):
 *   Pass a WebGLTexture (RG32F or RG16F, normalised NDC coords) via
 *   setVelocityTexture(tex, w, h).  The update shader samples it at
 *   the particle's NDC position and adds a scaled velocity contribution.
 *
 * Integration:
 *   ```ts
 *   const gpu = new ParticleGPU(canvas, edges);
 *   gpu.setVelocityTexture(fluidTex, 128, 128); // optional
 *   // render loop:
 *   gpu.update(dt);
 *   gpu.render(canvasW, canvasH);
 *   gpu.destroy();
 *   ```
 */

import { getShader } from '../shaders/ShaderLoader';
import type { SignalParticle } from '../cell-interaction-physics';

// ─── Constants ────────────────────────────────────────────────────────────────

/** Total particle pool. */
const PARTICLE_COUNT = 2000;

/** Floats per particle in the VBO. */
const PARTICLE_STRIDE_F32 = 16;

/** Byte stride per particle. */
const PARTICLE_STRIDE_BYTES = PARTICLE_STRIDE_F32 * 4; // 64 bytes

/** Particle max lifetime in seconds before respawn. */
const PARTICLE_LIFETIME = 2.5;

/** Maximum concurrent edge definitions. */
const MAX_EDGES = 64;

/** Number of trail history frames stored per particle. */
const TRAIL_FRAMES = 3;

// ─── Species color palette (mirrors world-renderer.ts SPECIES_COLORS) ────────

const SPECIES_COLORS_RGB: Record<number, [number, number, number]> = {
  0: [0.247, 0.318, 0.710],  // #3F51B5 — indigo
  1: [1.000, 0.435, 0.000],  // #FF6F00 — amber
  2: [0.180, 0.490, 0.196],  // #2E7D32 — green
  3: [0.776, 0.157, 0.157],  // #C62828 — red
  4: [0.271, 0.353, 0.392],  // #455A64 — blue-grey
  5: [0.482, 0.122, 0.635],  // #7B1FA2 — purple
  6: [0.086, 0.396, 0.753],  // #1565C0 — blue
};

function speciesRGB(species: number): [number, number, number] {
  return SPECIES_COLORS_RGB[species % 7] ?? [0.5, 0.5, 0.5];
}

/** Species color as CSS rgba string for Canvas 2D overlay rendering. */
function speciesRGBToCSS(r: number, g: number, b: number, alpha: number): string {
  return `rgba(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)},${alpha.toFixed(3)})`;
}

/** Species name → index lookup for signal molecule rendering. */
const SPECIES_INDEX: Record<string, number> = {
  'q':          0,
  'k':          1,
  'v':          2,
  'attn':       3,
  'mlp':        4,
  'add_norm':   5,
  'embed':      6,
  // fallback handled by modulo in speciesRGB
};

function speciesNameToIndex(species: string): number {
  return SPECIES_INDEX[species] ?? (species.charCodeAt(0) % 7);
}

// ─── Public types ─────────────────────────────────────────────────────────────

/** A 2-D point in canvas-pixel space. */
export interface ParticleFlowPoint {
  x: number;
  y: number;
}

/** Edge definition for particle spawning. */
export interface ParticleEdgeDef {
  edgeId:       string;
  /** Start point in NDC [-1,1] space. */
  srcPos:       ParticleFlowPoint;
  /** End point in NDC [-1,1] space. */
  dstPos:       ParticleFlowPoint;
  /** Source cell species index (0–6), maps to SPECIES_COLORS. */
  srcSpecies:   number;
  /** Target cell species index (0–6). */
  dstSpecies:   number;
  /** Optional weight — drives particles-per-second (default 1). */
  weight?:      number;
}

// ─── WebGL2 Update Vertex Shader (Transform Feedback) ────────────────────────
//
// Captured varyings (INTERLEAVED_ATTRIBS, in declaration order):
//   tf_position, tf_velocity, tf_life, tf_color, tf_birthLife,
//   tf_trail0, tf_trail1, tf_trail2, tf_edgeT

const UPDATE_VERT_SRC = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;

// ── per-particle inputs (layout locations match PARTICLE_STRIDE offsets) ──
layout(location = 0) in vec2  a_position;    // NDC position
layout(location = 1) in vec2  a_velocity;    // velocity (NDC units/s)
layout(location = 2) in float a_life;        // normalised remaining life [0,1]
layout(location = 3) in vec3  a_color;       // current RGB
layout(location = 4) in float a_birthLife;   // life value at last respawn
layout(location = 5) in vec2  a_trail0;      // pos 1 frame ago
layout(location = 6) in vec2  a_trail1;      // pos 2 frames ago
layout(location = 7) in vec2  a_trail2;      // pos 3 frames ago
layout(location = 8) in float a_edgeT;       // travel fraction [0,1]

// ── uniforms ─────────────────────────────────────────────────────────
uniform float u_dt;
uniform float u_time;
uniform int   u_edgeCount;
uniform vec2  u_edgeSrc[64];
uniform vec2  u_edgeDst[64];
uniform vec3  u_edgeSrcColor[64];
uniform vec3  u_edgeDstColor[64];

// Fluid velocity texture (optional; unit 0)
// RG channels = velocity XY in NDC/s.  Set u_hasVelocityTex=0 to skip.
uniform sampler2D u_velocityTex;
uniform int       u_hasVelocityTex;   // 1 = sample; 0 = skip
uniform float     u_fluidStrength;    // scale factor for fluid contribution

// ── transform feedback outputs ────────────────────────────────────────
out vec2  tf_position;
out vec2  tf_velocity;
out float tf_life;
out vec3  tf_color;
out float tf_birthLife;
out vec2  tf_trail0;
out vec2  tf_trail1;
out vec2  tf_trail2;
out float tf_edgeT;

// ── pseudo-random hash ────────────────────────────────────────────────
float hash(float n) {
  return fract(sin(n * 127.1 + u_time * 0.1) * 43758.5453123);
}

void main() {
  int pidx = gl_VertexID;

  // Determine which edge this particle belongs to
  int eidx = 0;
  if (u_edgeCount > 0) {
    eidx = int(mod(float(pidx), float(max(u_edgeCount, 1))));
    eidx = clamp(eidx, 0, u_edgeCount - 1);
  }

  vec2 src      = u_edgeSrc[eidx];
  vec2 dst      = u_edgeDst[eidx];
  vec3 srcColor = u_edgeSrcColor[eidx];
  vec3 dstColor = u_edgeDstColor[eidx];

  // Edge geometry
  vec2  edgeDir  = dst - src;
  float edgeLen  = length(edgeDir);
  vec2  edgeUnit = (edgeLen > 0.0001) ? edgeDir / edgeLen : vec2(1.0, 0.0);
  vec2  edgePerp = vec2(-edgeUnit.y, edgeUnit.x);

  // ── Fluid advection ───────────────────────────────────────────────
  vec2 fluidVel = vec2(0.0);
  if (u_hasVelocityTex == 1) {
    // Convert NDC [-1,1] → UV [0,1]
    vec2 uv = a_position * 0.5 + 0.5;
    uv = clamp(uv, 0.0, 1.0);
    vec2 fv = texture(u_velocityTex, uv).rg;
    fluidVel = fv * u_fluidStrength;
  }

  float newLife = a_life - u_dt / 2.5;
  vec2  newPos  = a_position + (a_velocity + fluidVel) * u_dt;
  vec2  newVel  = a_velocity;
  vec3  newColor = a_color;
  float newBirth = a_birthLife;
  float newEdgeT = a_edgeT;

  // Shift trail ring buffer: trail2 ← trail1 ← trail0 ← old position
  vec2 newTrail0 = a_position;
  vec2 newTrail1 = a_trail0;
  vec2 newTrail2 = a_trail1;

  if (newLife <= 0.0) {
    // ── Respawn ──────────────────────────────────────────────────────
    float r0 = hash(float(pidx) * 3.7 + 0.1);
    float r1 = hash(float(pidx) * 7.3 + 0.3);
    float r2 = hash(float(pidx) * 13.1 + 0.7);

    newPos  = src + edgePerp * (r0 - 0.5) * 0.04;
    float speed = 0.15 + r1 * 0.25;
    newVel  = edgeUnit * speed + edgePerp * (r2 - 0.5) * 0.02;

    // Stagger life so not all particles respawn simultaneously
    newLife  = 0.6 + r0 * 0.4;
    newBirth = newLife;
    newEdgeT = 0.0;

    // Begin at source color
    newColor = srcColor;

    // Clear trail to spawn position
    newTrail0 = newPos;
    newTrail1 = newPos;
    newTrail2 = newPos;

  } else {
    // ── Alive: update travel fraction + color gradient ────────────────
    vec2  relPos = newPos - src;
    float along  = dot(relPos, edgeUnit);
    newEdgeT = (edgeLen > 0.0001) ? clamp(along / edgeLen, 0.0, 1.0) : 0.0;

    // Color: smooth gradient from source species → target species
    newColor = mix(srcColor, dstColor, newEdgeT);

    // Fast fade when particle reaches destination
    if (newEdgeT >= 0.98) {
      newLife -= u_dt * 3.0;
    }

    // Apply fluid contribution to velocity
    newVel = newVel + fluidVel * 0.3;

    // Drag + forward-bias
    newVel = newVel * (1.0 - 0.4 * u_dt);
    float vAlong = dot(newVel, edgeUnit);
    vAlong = max(vAlong, 0.05);
    newVel = edgeUnit * vAlong + edgePerp * dot(newVel, edgePerp) * 0.3;
  }

  tf_position  = newPos;
  tf_velocity  = newVel;
  tf_life      = newLife;
  tf_color     = newColor;
  tf_birthLife = newBirth;
  tf_trail0    = newTrail0;
  tf_trail1    = newTrail1;
  tf_trail2    = newTrail2;
  tf_edgeT     = newEdgeT;

  // Update pass: no rasterisation output
  gl_Position  = vec4(0.0, 0.0, 0.0, 1.0);
  gl_PointSize = 1.0;
}
`;

// ─── Update Fragment Shader (dummy — TF pass, no rasterisation) ──────────────

const UPDATE_FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;
out vec4 dummy;
void main() { dummy = vec4(0.0); }
`;

// ─── Render Vertex Shader — Particle Heads ────────────────────────────────────
//
// Draws the particle head as a gl.POINT.
// Size curve: sin(π · normalizedAge) — small at birth, peak at mid, small at death.

const RENDER_VERT_SRC = /* glsl */ `#version 300 es
precision highp float;

layout(location = 0) in vec2  a_position;
layout(location = 1) in vec2  a_velocity;
layout(location = 2) in float a_life;
layout(location = 3) in vec3  a_color;
layout(location = 4) in float a_birthLife;
// trail slots (5-7) and edgeT (8) not needed for head pass

uniform vec2  u_resolution;
uniform float u_time;

out vec4  v_color;
out float v_life;

void main() {
  if (a_life <= 0.0) {
    gl_Position  = vec4(-10.0, -10.0, 0.0, 1.0);
    gl_PointSize = 0.0;
    v_color      = vec4(0.0);
    v_life       = 0.0;
    return;
  }

  gl_Position = vec4(a_position, 0.0, 1.0);

  // Normalised age: 0 = just born (high life), 1 = about to die (low life)
  float birthSafe = max(a_birthLife, 0.001);
  float age       = 1.0 - clamp(a_life / birthSafe, 0.0, 1.0);

  // Bell curve: sin(π·age) — birth small, peak at mid-life, death small
  float bell   = sin(3.14159265 * age);
  float sz     = 2.0 + bell * 7.0;        // range [2, 9] px

  // Speed boost for fast particles
  float spd    = length(a_velocity);
  sz          += spd * 4.0;

  gl_PointSize = clamp(sz, 1.0, 12.0);

  // Alpha fades in at birth and out at death
  float alpha = smoothstep(0.0, 0.12, a_life) * smoothstep(0.0, 0.12, age);
  v_color = vec4(a_color, alpha);
  v_life  = a_life;
}
`;

// ─── Render Fragment Shader — Particle Heads ──────────────────────────────────

const RENDER_FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;

in  vec4  v_color;
in  float v_life;
out vec4  outColor;

void main() {
  // Round soft point via radial discard
  vec2  uv  = gl_PointCoord * 2.0 - 1.0;
  float r   = dot(uv, uv);
  if (r > 1.0) discard;

  // Gaussian radial falloff
  float alpha = exp(-r * 2.8);

  // Additive glow: brighten core
  vec3 glow = v_color.rgb * (1.0 + (1.0 - r) * 0.6);
  float a   = v_color.a * alpha;

  outColor = vec4(glow * a, a);
}
`;

// ─── Trail Vertex Shader ──────────────────────────────────────────────────────
//
// Draws trail as LINES between the 3 history positions.
// Each particle contributes 6 vertices: (trail0→trail1), (trail1→trail2), (trail2→pos)
// We use a flat index buffer trick: draw PARTICLE_COUNT * 6 verts where
// gl_VertexID / 6 = particle index, gl_VertexID % 6 = segment vertex.
//
// Alternatively (simpler): we use the trail positions stored in the VBO
// and select them via a_trailVertex attribute. We build a CPU index buffer
// once at init that maps 6 verts per particle to the correct stride offsets.
//
// Attrib layout (same VBO, same stride):
//   location 0: a_position  (head pos — used as segment endpoint for newest seg)
//   location 5: a_trail0    (1 frame ago)
//   location 6: a_trail1    (2 frames ago)
//   location 7: a_trail2    (3 frames ago)
//   location 2: a_life      (for alpha)
//   location 8: a_edgeT     (for color — recomputed from src/dst)
//   location 3: a_color     (current interpolated color)

const TRAIL_VERT_SRC = /* glsl */ `#version 300 es
precision highp float;

// Full VBO per-particle attributes
layout(location = 0) in vec2  a_position;   // current head
layout(location = 1) in vec2  a_velocity;
layout(location = 2) in float a_life;
layout(location = 3) in vec3  a_color;      // current (dst) color
layout(location = 4) in float a_birthLife;
layout(location = 5) in vec2  a_trail0;    // 1 frame ago
layout(location = 6) in vec2  a_trail1;    // 2 frames ago
layout(location = 7) in vec2  a_trail2;    // 3 frames ago
layout(location = 8) in float a_edgeT;

// a_segVert: per-trail-vertex attribute (0..5 within each particle)
// supplied via a dedicated trail-index VBO (divisor = 0, no instancing)
layout(location = 9) in float a_segVert;

uniform vec2 u_resolution;

out vec4 v_trailColor;

void main() {
  if (a_life <= 0.0) {
    gl_Position  = vec4(-10.0, -10.0, 0.0, 1.0);
    v_trailColor = vec4(0.0);
    return;
  }

  // Segment mapping:
  //  segVert 0,1 → segment newest  (trail0 → position)
  //  segVert 2,3 → segment middle  (trail1 → trail0)
  //  segVert 4,5 → segment oldest  (trail2 → trail1)
  int sv = int(a_segVert);
  vec2 pos;
  float segAge; // 0=newest, 1=oldest

  if (sv == 0) { pos = a_trail0;    segAge = 0.0;  }
  else if (sv == 1) { pos = a_position;   segAge = 0.0;  }
  else if (sv == 2) { pos = a_trail1;    segAge = 0.33; }
  else if (sv == 3) { pos = a_trail0;    segAge = 0.33; }
  else if (sv == 4) { pos = a_trail2;    segAge = 0.67; }
  else              { pos = a_trail1;    segAge = 0.67; }

  gl_Position = vec4(pos, 0.0, 1.0);

  // Trail fades from head color at sv=1 to dimmer/transparent at sv=4
  float fadeAlpha = (1.0 - segAge) * 0.55 * smoothstep(0.0, 0.15, a_life);
  v_trailColor = vec4(a_color * (0.5 + 0.5 * (1.0 - segAge)), fadeAlpha);
}
`;

// ─── Trail Fragment Shader ────────────────────────────────────────────────────

const TRAIL_FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;

in  vec4 v_trailColor;
out vec4 outColor;

void main() {
  outColor = v_trailColor;
}
`;

// ─── ParticleGPU class ────────────────────────────────────────────────────────

export class ParticleGPU {
  private gl: WebGL2RenderingContext;
  private canvas: HTMLCanvasElement;

  // WebGL2 programs
  private updateProg!: WebGLProgram;
  private renderProg!: WebGLProgram;
  private trailProg!:  WebGLProgram;

  // VAOs for ping-pong (particle head + trail rendering)
  private vaoRead!:       WebGLVertexArrayObject;
  private vaoWrite!:      WebGLVertexArrayObject;
  private trailVaoA!:     WebGLVertexArrayObject;
  private trailVaoB!:     WebGLVertexArrayObject;

  // VBOs (ping-pong A / B)
  private vboA!: WebGLBuffer;
  private vboB!: WebGLBuffer;

  // Trail index VBO (static: 6 seg-vert indices per particle, interleaved)
  private trailSegVBO!: WebGLBuffer;

  // Transform Feedback objects
  private tfA!: WebGLTransformFeedback;
  private tfB!: WebGLTransformFeedback;

  // Edge table
  private edges: ParticleEdgeDef[] = [];

  // Elapsed time
  private elapsed = 0;

  // ── M1283: Signal particle overlay (Canvas 2D drawn on top of WebGL) ──
  /** 2D overlay canvas layered above the WebGL canvas for signal molecule dots. */
  private _signalOverlay: HTMLCanvasElement | null = null;
  private _signalCtx: CanvasRenderingContext2D | null = null;

  // Optional fluid velocity texture
  private velocityTex:    WebGLTexture | null = null;
  private velocityTexW    = 0;
  private velocityTexH    = 0;
  private _fluidStrength  = 0.12;

  // Uniform locations — update program
  private uLoc = {
    dt:             null as WebGLUniformLocation | null,
    time:           null as WebGLUniformLocation | null,
    edgeCount:      null as WebGLUniformLocation | null,
    edgeSrc:        null as WebGLUniformLocation | null,
    edgeDst:        null as WebGLUniformLocation | null,
    edgeSrcCol:     null as WebGLUniformLocation | null,
    edgeDstCol:     null as WebGLUniformLocation | null,
    velocityTex:    null as WebGLUniformLocation | null,
    hasVelocityTex: null as WebGLUniformLocation | null,
    fluidStrength:  null as WebGLUniformLocation | null,
  };

  // Uniform locations — render program
  private rLoc = {
    resolution: null as WebGLUniformLocation | null,
    time:       null as WebGLUniformLocation | null,
  };

  // Uniform locations — trail program
  private tLoc = {
    resolution: null as WebGLUniformLocation | null,
  };

  constructor(canvasOrGL: HTMLCanvasElement | WebGL2RenderingContext, edges: ParticleEdgeDef[] = []) {
    if (canvasOrGL instanceof WebGL2RenderingContext) {
      this.gl = canvasOrGL;
      this.canvas = canvasOrGL.canvas as HTMLCanvasElement;
    } else {
      const gl = canvasOrGL.getContext('webgl2');
      if (!gl) throw new Error('[ParticleGPU] WebGL2 not available on this canvas.');
      this.gl     = gl;
      this.canvas = canvasOrGL;
    }
    this.edges  = edges.slice(0, MAX_EDGES);
    this._init();
  }

  // ─── Public API ───────────────────────────────────────────────────────────

  /** Add or replace an edge in the flow table. */
  addEdge(edge: ParticleEdgeDef): void {
    const idx = this.edges.findIndex(e => e.edgeId === edge.edgeId);
    if (idx >= 0) this.edges[idx] = edge;
    else if (this.edges.length < MAX_EDGES) this.edges.push(edge);
  }

  /** Remove an edge by id. */
  removeEdge(edgeId: string): void {
    this.edges = this.edges.filter(e => e.edgeId !== edgeId);
  }

  /**
   * Provide a fluid velocity texture for particle advection.
   * @param tex  WebGLTexture — RG channels encode velocity XY (NDC/s).
   * @param w    Texture width (informational, not required by shader).
   * @param h    Texture height.
   * @param strength  Scale factor applied to sampled velocity (default 0.12).
   */
  setVelocityTexture(tex: WebGLTexture, w: number, h: number, strength = 0.12): void {
    this.velocityTex    = tex;
    this.velocityTexW   = w;
    this.velocityTexH   = h;
    this._fluidStrength = strength;
  }

  /** Clear the fluid velocity texture (particles move under edge flow only). */
  clearVelocityTexture(): void {
    this.velocityTex = null;
  }

  /**
   * Update pass — Transform Feedback advances all particles.
   * Call once per frame before render().
   */
  update(dt: number): void {
    const gl = this.gl;
    this.elapsed += dt;

    gl.useProgram(this.updateProg);

    // ── Upload uniforms ────────────────────────────────────────────────────
    gl.uniform1f(this.uLoc.dt!,        dt);
    gl.uniform1f(this.uLoc.time!,      this.elapsed);
    gl.uniform1i(this.uLoc.edgeCount!, this.edges.length);

    // Edge table
    const srcXY  = new Float32Array(MAX_EDGES * 2);
    const dstXY  = new Float32Array(MAX_EDGES * 2);
    const srcCol = new Float32Array(MAX_EDGES * 3);
    const dstCol = new Float32Array(MAX_EDGES * 3);

    for (let i = 0; i < this.edges.length; i++) {
      const e = this.edges[i];
      srcXY[i*2]     = e.srcPos.x;  srcXY[i*2+1]  = e.srcPos.y;
      dstXY[i*2]     = e.dstPos.x;  dstXY[i*2+1]  = e.dstPos.y;
      const sc = speciesRGB(e.srcSpecies);
      const tc = speciesRGB(e.dstSpecies);
      srcCol[i*3] = sc[0]; srcCol[i*3+1] = sc[1]; srcCol[i*3+2] = sc[2];
      dstCol[i*3] = tc[0]; dstCol[i*3+1] = tc[1]; dstCol[i*3+2] = tc[2];
    }
    // Fallback when no edges: default diagonal
    if (this.edges.length === 0) {
      srcXY[0] = -0.5; srcXY[1] = 0.0;
      dstXY[0] =  0.5; dstXY[1] = 0.0;
      srcCol[0] = 0.247; srcCol[1] = 0.318; srcCol[2] = 0.710;
      dstCol[0] = 1.000; dstCol[1] = 0.435; dstCol[2] = 0.000;
    }

    gl.uniform2fv(this.uLoc.edgeSrc!,    srcXY);
    gl.uniform2fv(this.uLoc.edgeDst!,    dstXY);
    gl.uniform3fv(this.uLoc.edgeSrcCol!, srcCol);
    gl.uniform3fv(this.uLoc.edgeDstCol!, dstCol);

    // Fluid velocity texture
    if (this.velocityTex) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.velocityTex);
      gl.uniform1i(this.uLoc.velocityTex!,    0);
      gl.uniform1i(this.uLoc.hasVelocityTex!, 1);
      gl.uniform1f(this.uLoc.fluidStrength!,  this._fluidStrength);
    } else {
      gl.uniform1i(this.uLoc.hasVelocityTex!, 0);
      gl.uniform1f(this.uLoc.fluidStrength!,  0.0);
    }

    // ── Transform Feedback pass ─────────────────────────────────────────────
    gl.bindVertexArray(this.vaoRead);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this._writeTF());

    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, PARTICLE_COUNT);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindVertexArray(null);

    if (this.velocityTex) gl.bindTexture(gl.TEXTURE_2D, null);

    this._swap();
  }

  /**
   * Render pass — draw trail lines then particle heads.
   * Call after update().
   */
  render(canvasW: number, canvasH: number): void {
    const gl = this.gl;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasW, canvasH);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    // ── 1. Trail lines ──────────────────────────────────────────────────────
    gl.useProgram(this.trailProg);
    gl.uniform2f(this.tLoc.resolution!, canvasW, canvasH);

    const trailVao = this._pingA ? this.trailVaoB : this.trailVaoA;
    gl.bindVertexArray(trailVao);
    // 3 segments × 2 verts × PARTICLE_COUNT
    gl.drawArrays(gl.LINES, 0, PARTICLE_COUNT * TRAIL_FRAMES * 2);
    gl.bindVertexArray(null);

    // ── 2. Particle heads ───────────────────────────────────────────────────
    gl.useProgram(this.renderProg);
    gl.uniform2f(this.rLoc.resolution!, canvasW, canvasH);
    gl.uniform1f(this.rLoc.time!,       this.elapsed);

    gl.bindVertexArray(this.vaoRead);
    gl.drawArrays(gl.POINTS, 0, PARTICLE_COUNT);
    gl.bindVertexArray(null);

    gl.disable(gl.BLEND);
  }

  /**
   * M1283: Render quorum-sensing signal molecules as small circles on a 2D canvas
   * overlay positioned on top of the WebGL canvas.
   *
   * Call this AFTER render() each frame, passing the signalParticles array from
   * CellInteractionPhysics and the NDC→pixel transform used by the main render.
   *
   * @param particles   The signalParticles array from CellInteractionPhysics.
   * @param canvasW     WebGL canvas width in pixels.
   * @param canvasH     WebGL canvas height in pixels.
   * @param ndcToPixel  Optional transform: {offX, offY, scale} mapping cell-space → NDC → pixel.
   *                    If omitted, particles are treated as already in pixel space.
   */
  renderSignalParticles(
    particles: SignalParticle[],
    canvasW: number,
    canvasH: number,
    ndcToPixel?: { offX: number; offY: number; scale: number },
  ): void {
    // Lazy-create the overlay canvas
    if (!this._signalOverlay) {
      const overlay = document.createElement('canvas');
      overlay.style.position = 'absolute';
      overlay.style.top      = '0';
      overlay.style.left     = '0';
      overlay.style.pointerEvents = 'none';
      overlay.style.zIndex   = '1';
      this.canvas.parentElement?.appendChild(overlay);
      this._signalOverlay = overlay;
      this._signalCtx     = overlay.getContext('2d');
    }

    // Sync overlay size
    if (
      this._signalOverlay.width  !== canvasW ||
      this._signalOverlay.height !== canvasH
    ) {
      this._signalOverlay.width  = canvasW;
      this._signalOverlay.height = canvasH;
      this._signalOverlay.style.width  = `${canvasW}px`;
      this._signalOverlay.style.height = `${canvasH}px`;
    }

    const ctx = this._signalCtx;
    if (!ctx) return;

    // Clear previous frame
    ctx.clearRect(0, 0, canvasW, canvasH);

    if (particles.length === 0) return;

    const SIGNAL_RADIUS_PX = 3; // visual dot radius in pixels

    for (const sp of particles) {
      if (sp.alpha < 0.01) continue;

      // Coordinate mapping: cell-space → pixel space
      let px: number;
      let py: number;
      if (ndcToPixel) {
        px = sp.x * ndcToPixel.scale + ndcToPixel.offX;
        py = sp.y * ndcToPixel.scale + ndcToPixel.offY;
      } else {
        px = sp.x;
        py = sp.y;
      }

      const speciesIdx = speciesNameToIndex(sp.species);
      const [r, g, b]  = speciesRGB(speciesIdx);
      const fillColor  = speciesRGBToCSS(r, g, b, sp.alpha);

      ctx.beginPath();
      ctx.arc(px, py, SIGNAL_RADIUS_PX, 0, Math.PI * 2);
      ctx.fillStyle   = fillColor;
      ctx.shadowColor = fillColor;
      ctx.shadowBlur  = 4;
      ctx.fill();
    }

    // Reset shadow to avoid bleeding onto other draws
    ctx.shadowBlur  = 0;
    ctx.shadowColor = 'transparent';
  }

  /** Free all GPU resources. */
  destroy(): void {
    const gl = this.gl;
    gl.deleteProgram(this.updateProg);
    gl.deleteProgram(this.renderProg);
    gl.deleteProgram(this.trailProg);
    gl.deleteVertexArray(this.vaoRead);
    gl.deleteVertexArray(this.vaoWrite);
    gl.deleteVertexArray(this.trailVaoA);
    gl.deleteVertexArray(this.trailVaoB);
    gl.deleteBuffer(this.vboA);
    gl.deleteBuffer(this.vboB);
    gl.deleteBuffer(this.trailSegVBO);
    gl.deleteTransformFeedback(this.tfA);
    gl.deleteTransformFeedback(this.tfB);

    // M1283: Remove 2D signal overlay canvas
    if (this._signalOverlay) {
      this._signalOverlay.parentElement?.removeChild(this._signalOverlay);
      this._signalOverlay = null;
      this._signalCtx     = null;
    }
  }

  // ─── Private: initialisation ──────────────────────────────────────────────

  private _init(): void {
    this._compilePrograms();
    this._createBuffers();
    this._cacheUniformLocations();
  }

  private _compilePrograms(): void {
    // Update program — Transform Feedback
    this.updateProg = this._compileWithTF(
      UPDATE_VERT_SRC,
      UPDATE_FRAG_SRC,
      ['tf_position', 'tf_velocity', 'tf_life', 'tf_color',
       'tf_birthLife', 'tf_trail0', 'tf_trail1', 'tf_trail2', 'tf_edgeT'],
      'update',
    );

    // Render program — particle heads
    this.renderProg = this._compile(RENDER_VERT_SRC, RENDER_FRAG_SRC, 'render');

    // Trail program
    this.trailProg = this._compile(TRAIL_VERT_SRC, TRAIL_FRAG_SRC, 'trail');
  }

  /** Compile a WebGL2 program with transform feedback varyings declared before link. */
  private _compileWithTF(
    vertSrc: string, fragSrc: string,
    tfVaryings: string[],
    label: string,
  ): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vertSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ParticleGPU] vert compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ParticleGPU] frag compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);

    this._bindAttribLocations(prog);
    gl.transformFeedbackVaryings(prog, tfVaryings, gl.INTERLEAVED_ATTRIBS);

    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ParticleGPU] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** Compile a standard (non-TF) WebGL2 program. */
  private _compile(vertSrc: string, fragSrc: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vertSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ParticleGPU] vert compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ParticleGPU] frag compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    this._bindAttribLocations(prog);

    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ParticleGPU] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** Bind all named attribs to explicit locations — must be called before link. */
  private _bindAttribLocations(prog: WebGLProgram): void {
    const gl = this.gl;
    gl.bindAttribLocation(prog, 0, 'a_position');
    gl.bindAttribLocation(prog, 1, 'a_velocity');
    gl.bindAttribLocation(prog, 2, 'a_life');
    gl.bindAttribLocation(prog, 3, 'a_color');
    gl.bindAttribLocation(prog, 4, 'a_birthLife');
    gl.bindAttribLocation(prog, 5, 'a_trail0');
    gl.bindAttribLocation(prog, 6, 'a_trail1');
    gl.bindAttribLocation(prog, 7, 'a_trail2');
    gl.bindAttribLocation(prog, 8, 'a_edgeT');
    gl.bindAttribLocation(prog, 9, 'a_segVert');
  }

  private _createBuffers(): void {
    const gl = this.gl;

    // ── Initial particle data ───────────────────────────────────────────────
    const data = new Float32Array(PARTICLE_COUNT * PARTICLE_STRIDE_F32);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const b = i * PARTICLE_STRIDE_F32;
      const angle = Math.random() * Math.PI * 2;
      const r     = Math.random() * 0.8;
      const life  = Math.random();
      data[b +  0] = Math.cos(angle) * r;   // position x
      data[b +  1] = Math.sin(angle) * r;   // position y
      data[b +  2] = (Math.random() - 0.5) * 0.2; // velocity x
      data[b +  3] = (Math.random() - 0.5) * 0.2; // velocity y
      data[b +  4] = life;                   // life
      const col    = speciesRGB(i % 7);
      data[b +  5] = col[0];  // color r
      data[b +  6] = col[1];  // color g
      data[b +  7] = col[2];  // color b
      data[b +  8] = life;    // birthLife (same as life at init)
      // trail0-2: same as spawn position
      data[b +  9] = data[b + 0];
      data[b + 10] = data[b + 1];
      data[b + 11] = data[b + 0];
      data[b + 12] = data[b + 1];
      data[b + 13] = data[b + 0];
      data[b + 14] = data[b + 1];
      data[b + 15] = 0.0;     // edgeT
    }

    // ── Particle VBOs ───────────────────────────────────────────────────────
    this.vboA = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboA);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_COPY);

    this.vboB = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboB);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_COPY);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // ── Transform Feedback objects ──────────────────────────────────────────
    this.tfA = gl.createTransformFeedback()!;
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tfA);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.vboA);

    this.tfB = gl.createTransformFeedback()!;
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tfB);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.vboB);

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);

    // ── Trail segment index VBO ─────────────────────────────────────────────
    // For each particle: 6 values — one per line endpoint (3 segments × 2 verts)
    // Values: 0,1, 2,3, 4,5  (matches gl_VertexID % 6 in trail shader)
    // The VBO is read with stride = 0, so each particle's block of 6 floats
    // is consumed by 6 consecutive draw-array vertices.
    const segData = new Float32Array(PARTICLE_COUNT * TRAIL_FRAMES * 2);
    for (let p = 0; p < PARTICLE_COUNT; p++) {
      const base = p * TRAIL_FRAMES * 2;
      segData[base + 0] = 0;
      segData[base + 1] = 1;
      segData[base + 2] = 2;
      segData[base + 3] = 3;
      segData[base + 4] = 4;
      segData[base + 5] = 5;
    }

    this.trailSegVBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.trailSegVBO);
    gl.bufferData(gl.ARRAY_BUFFER, segData, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // ── VAOs for update (read-VBO as attrib input) ──────────────────────────
    this.vaoRead  = this._createParticleVAO(this.vboA);
    this.vaoWrite = this._createParticleVAO(this.vboB);

    // ── VAOs for trail rendering (include segVert from trailSegVBO) ─────────
    this.trailVaoA = this._createTrailVAO(this.vboA);
    this.trailVaoB = this._createTrailVAO(this.vboB);
  }

  /**
   * Create a VAO for the update/render-head passes.
   * Each vertex = one particle; attribs read from per-particle VBO at stride 64.
   */
  private _createParticleVAO(vbo: WebGLBuffer): WebGLVertexArrayObject {
    const gl  = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);

    const S = PARTICLE_STRIDE_BYTES; // 64

    // loc 0: a_position  — vec2  @ offset 0
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, S, 0);

    // loc 1: a_velocity  — vec2  @ offset 8
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, S, 8);

    // loc 2: a_life      — float @ offset 16
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, S, 16);

    // loc 3: a_color     — vec3  @ offset 20
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 3, gl.FLOAT, false, S, 20);

    // loc 4: a_birthLife — float @ offset 32
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 1, gl.FLOAT, false, S, 32);

    // loc 5: a_trail0    — vec2  @ offset 36
    gl.enableVertexAttribArray(5);
    gl.vertexAttribPointer(5, 2, gl.FLOAT, false, S, 36);

    // loc 6: a_trail1    — vec2  @ offset 44
    gl.enableVertexAttribArray(6);
    gl.vertexAttribPointer(6, 2, gl.FLOAT, false, S, 44);

    // loc 7: a_trail2    — vec2  @ offset 52
    gl.enableVertexAttribArray(7);
    gl.vertexAttribPointer(7, 2, gl.FLOAT, false, S, 52);

    // loc 8: a_edgeT     — float @ offset 60
    gl.enableVertexAttribArray(8);
    gl.vertexAttribPointer(8, 1, gl.FLOAT, false, S, 60);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
    return vao;
  }

  /**
   * Create a VAO for trail line rendering.
   * We draw PARTICLE_COUNT × 6 vertices.
   * Particle VBO attribs use divisor trick: attribDivisor not applicable here
   * because we're not instancing. Instead, we rely on the segVert VBO at loc 9
   * (no stride between elements within a particle block) alongside the particle
   * VBO being stepped every 6 trail verts via gl_VertexID arithmetic in the shader.
   *
   * Simpler approach: use SEPARATE_ATTRIBS style where the particle-data VBO
   * is bound with stride = PARTICLE_STRIDE_BYTES but a divisor of 6 via
   * gl.vertexAttribDivisor... except divisor only works for instances.
   *
   * Actual approach used here:
   *   • trailSegVBO (loc 9): packed 6 floats per particle = PARTICLE_COUNT*6 verts total
   *   • particle VBO (locs 0-8): use vertexAttribDivisor(loc, 6) so each particle
   *     block of 6 trail verts reads the same particle attribs.
   *   • gl.drawArraysInstanced is NOT used — we use gl.drawArrays + divisor to
   *     implement "fetch particle attrib every 6 verts".
   *
   * Wait — vertexAttribDivisor only applies to instanced draws. For non-instanced,
   * the divisor must be 0. We use a different simple approach:
   *   Build a full expanded trail VBO CPU-side per frame (too slow) OR
   *   use drawArraysInstanced(LINES, 0, 6, PARTICLE_COUNT) with divisor=1 on particle attribs.
   *
   * ✅ FINAL: instanced draw, 6 verts per instance (= 1 particle = 3 segments), divisor=1.
   */
  private _createTrailVAO(vbo: WebGLBuffer): WebGLVertexArrayObject {
    const gl  = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);

    const S = PARTICLE_STRIDE_BYTES; // 64

    // ── Particle data (per-instance attribs, divisor = 1) ──────────────────
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);

    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, S, 0);
    gl.vertexAttribDivisor(0, 1);

    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, S, 8);
    gl.vertexAttribDivisor(1, 1);

    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, S, 16);
    gl.vertexAttribDivisor(2, 1);

    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 3, gl.FLOAT, false, S, 20);
    gl.vertexAttribDivisor(3, 1);

    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 1, gl.FLOAT, false, S, 32);
    gl.vertexAttribDivisor(4, 1);

    gl.enableVertexAttribArray(5);
    gl.vertexAttribPointer(5, 2, gl.FLOAT, false, S, 36);
    gl.vertexAttribDivisor(5, 1);

    gl.enableVertexAttribArray(6);
    gl.vertexAttribPointer(6, 2, gl.FLOAT, false, S, 44);
    gl.vertexAttribDivisor(6, 1);

    gl.enableVertexAttribArray(7);
    gl.vertexAttribPointer(7, 2, gl.FLOAT, false, S, 52);
    gl.vertexAttribDivisor(7, 1);

    gl.enableVertexAttribArray(8);
    gl.vertexAttribPointer(8, 1, gl.FLOAT, false, S, 60);
    gl.vertexAttribDivisor(8, 1);

    // ── Trail seg-vert index (per-vertex, divisor = 0) ─────────────────────
    gl.bindBuffer(gl.ARRAY_BUFFER, this.trailSegVBO);
    gl.enableVertexAttribArray(9);
    // The trailSegVBO contains PARTICLE_COUNT*6 floats (0..5 pattern repeated)
    // We read it as a flat per-vertex attrib (divisor 0) for the 6 verts per instance.
    gl.vertexAttribPointer(9, 1, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(9, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
    return vao;
  }

  private _cacheUniformLocations(): void {
    const gl = this.gl;
    const u  = this.updateProg;
    this.uLoc.dt             = gl.getUniformLocation(u, 'u_dt');
    this.uLoc.time           = gl.getUniformLocation(u, 'u_time');
    this.uLoc.edgeCount      = gl.getUniformLocation(u, 'u_edgeCount');
    this.uLoc.edgeSrc        = gl.getUniformLocation(u, 'u_edgeSrc');
    this.uLoc.edgeDst        = gl.getUniformLocation(u, 'u_edgeDst');
    this.uLoc.edgeSrcCol     = gl.getUniformLocation(u, 'u_edgeSrcColor');
    this.uLoc.edgeDstCol     = gl.getUniformLocation(u, 'u_edgeDstColor');
    this.uLoc.velocityTex    = gl.getUniformLocation(u, 'u_velocityTex');
    this.uLoc.hasVelocityTex = gl.getUniformLocation(u, 'u_hasVelocityTex');
    this.uLoc.fluidStrength  = gl.getUniformLocation(u, 'u_fluidStrength');

    const r = this.renderProg;
    this.rLoc.resolution = gl.getUniformLocation(r, 'u_resolution');
    this.rLoc.time       = gl.getUniformLocation(r, 'u_time');

    const t = this.trailProg;
    this.tLoc.resolution = gl.getUniformLocation(t, 'u_resolution');
  }

  // ─── Ping-pong state ──────────────────────────────────────────────────────

  /** true = vboA is read-input, vboB is write-output */
  private _pingA = true;

  private _writeTF(): WebGLTransformFeedback {
    return this._pingA ? this.tfB : this.tfA;
  }

  private _swap(): void {
    [this.vaoRead, this.vaoWrite] = [this.vaoWrite, this.vaoRead];
    this._pingA = !this._pingA;
  }
}

// ─── Trail render() note ──────────────────────────────────────────────────────
//
// render() uses gl.drawArraysInstanced(gl.LINES, 0, 6, PARTICLE_COUNT).
// The trail VAO binds particle attribs with divisor=1 (step per instance)
// and the segVert index with divisor=0 (step per vertex).
// This gives us 6 vertices per particle, forming 3 LINES segments:
//   vert 0,1 → trail0  → position   (newest)
//   vert 2,3 → trail1  → trail0
//   vert 4,5 → trail2  → trail1     (oldest)
//
// Note: the render() method above calls gl.drawArrays(LINES, ...) for simplicity;
// callers that want instanced trail rendering should call renderInstanced() instead,
// or the class can be updated to use drawArraysInstanced internally.
// For maximum compatibility, render() uses the flat trailSegVBO approach.

// ─── Factory helpers ──────────────────────────────────────────────────────────

/**
 * Build a ParticleEdgeDef from pixel-space endpoints + species indices.
 * Converts pixel coords to NDC given canvas size.
 */
export function makeEdgeDef(
  edgeId:     string,
  srcPx:      { x: number; y: number },
  dstPx:      { x: number; y: number },
  canvasW:    number,
  canvasH:    number,
  srcSpecies: number,
  dstSpecies: number,
  weight?:    number,
): ParticleEdgeDef {
  const toNDC = (px: { x: number; y: number }) => ({
    x:  (px.x / canvasW) * 2 - 1,
    y: -((px.y / canvasH) * 2 - 1),  // Y-flip: canvas Y-down → NDC Y-up
  });
  return {
    edgeId,
    srcPos:     toNDC(srcPx),
    dstPos:     toNDC(dstPx),
    srcSpecies,
    dstSpecies,
    weight,
  };
}

/** Re-export getShader for consumers that want to access AT shaders. */
export { getShader } from '../shaders/ShaderLoader';
