/**
 * world-renderer.ts — M1317l: Clean 500-line WebGL2 render entry
 *
 * Replaces the 2400-line GPURenderLoop with a single class.
 * Inspired by SwissGL's <1000-line philosophy: one file, one draw call path.
 *
 * Architecture:
 *  - constructor(canvas): compile shaders, build quad VAO, allocate FBO
 *  - setScene(cells, edges): pack cell data into texture
 *  - frame(time): instanced draw (cells) + line draw (edges)
 *  - start() / stop(): rAF loop
 */

import type { CellData, EdgeData } from './gpu-render-loop';

// ─── re-export so callers can import from here ────────────────────────────────
export type { CellData, EdgeData };

// ─── Vertex shader: per-cell instanced quad ───────────────────────────────────
const CELL_VERT = /* glsl */`#version 300 es
precision highp float;

// Per-vertex: unit quad corner [-1, 1]
in vec2 aCorner;

// Per-instance (from texture row i):
//  xy = NDC centre, zw = NDC half-size
in vec4 iRect;       // x, y, hw, hh  (NDC)
in vec4 iAlbedo;     // r, g, b, opacity
in vec4 iGlow;       // r, g, b, baseRadius
in vec4 iSDF;        // noiseAmp, noiseFreq, roughness, metallic
// lobes packed as (angle, dist, radius, 0) × up to 4
in vec4 iLobe0;
in vec4 iLobe1;
in vec4 iLobe2;
in vec4 iLobe3;

out vec2 vUV;
out vec3 vAlbedo;
out float vOpacity;
out vec3 vGlow;
out float vBaseRadius;
out vec4 vSDF;       // noiseAmp, noiseFreq, roughness, metallic
out vec4 vLobe0;
out vec4 vLobe1;
out vec4 vLobe2;
out vec4 vLobe3;
out float vAspect;

void main() {
  vec2 centre  = iRect.xy;
  vec2 halfSz  = iRect.zw;
  vAspect  = halfSz.x / max(halfSz.y, 0.0001);

  vUV      = aCorner * 0.5 + 0.5;
  vAlbedo  = iAlbedo.rgb;
  vOpacity = iAlbedo.a;
  vGlow    = iGlow.rgb;
  vBaseRadius = iGlow.a;
  vSDF     = iSDF;
  vLobe0   = iLobe0;
  vLobe1   = iLobe1;
  vLobe2   = iLobe2;
  vLobe3   = iLobe3;

  gl_Position = vec4(centre + aCorner * halfSz, 0.0, 1.0);
}
`;

// ─── Fragment shader: 3D SDF ray march ───────────────────────────────────────
const CELL_FRAG = /* glsl */`#version 300 es
precision highp float;

in vec2  vUV;
in vec3  vAlbedo;
in float vOpacity;
in vec3  vGlow;
in float vBaseRadius;
in vec4  vSDF;       // noiseAmp, noiseFreq, roughness, metallic
in vec4  vLobe0;
in vec4  vLobe1;
in vec4  vLobe2;
in vec4  vLobe3;
in float vAspect;

uniform float uTime;
out vec4 fragColor;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i),       hash12(i+vec2(1,0)), u.x),
             mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y);
}
float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
  return mix(b, a, h) - k*h*(1.0-h);
}

float cellSDF(vec3 p, float baseR, float noiseAmp, float noiseFreq) {
  float d = length(p) - baseR;
  // 4 lobes max
  vec4 lobes[4];
  lobes[0] = vLobe0; lobes[1] = vLobe1; lobes[2] = vLobe2; lobes[3] = vLobe3;
  for (int i = 0; i < 4; i++) {
    if (lobes[i].z < 0.001) continue;
    vec3 lc = vec3(
      cos(lobes[i].x) * lobes[i].y,
      sin(lobes[i].x) * lobes[i].y,
      sin(uTime * 0.5 + lobes[i].x) * 0.08
    );
    d = smin(d, length(p - lc) - lobes[i].z, 0.3);
  }
  d += (vnoise(p.xy * noiseFreq + uTime * 0.15) * 2.0 - 1.0) * noiseAmp;
  return d;
}

vec3 calcNormal(vec3 p, float baseR, float noiseAmp, float noiseFreq) {
  vec2 e = vec2(0.005, 0.0);
  return normalize(vec3(
    cellSDF(p+e.xyy,baseR,noiseAmp,noiseFreq) - cellSDF(p-e.xyy,baseR,noiseAmp,noiseFreq),
    cellSDF(p+e.yxy,baseR,noiseAmp,noiseFreq) - cellSDF(p-e.yxy,baseR,noiseAmp,noiseFreq),
    cellSDF(p+e.yyx,baseR,noiseAmp,noiseFreq) - cellSDF(p-e.yyx,baseR,noiseAmp,noiseFreq)
  ));
}

void main() {
  float noiseAmp  = vSDF.x;
  float noiseFreq = vSDF.y;
  float roughness = vSDF.z;
  float metallic  = vSDF.w;

  // Map UV [0,1] → local [-1,1], correct for non-square quad
  vec2 uv  = vUV * 2.0 - 1.0;
  float ax = max(vAspect, 1.0);
  float ay = max(1.0 / vAspect, 1.0);
  vec3 ro  = vec3(uv.x * ax * 1.2, uv.y * ay * 1.2, 2.0);
  vec3 rd  = vec3(0.0, 0.0, -1.0);

  // Normalise SDF radius to [-1,1]³ space (baseRadius was in pixels, clamp to ~0.8)
  float baseR = clamp(vBaseRadius / 20.0, 0.3, 0.85);

  float t   = 0.0;
  bool  hit = false;
  for (int i = 0; i < 64; i++) {
    float d = cellSDF(ro + rd * t, baseR, noiseAmp, noiseFreq);
    if (d < 0.002) { hit = true; break; }
    if (t > 5.0)   break;
    t += d;
  }
  if (!hit) discard;

  vec3 p  = ro + rd * t;
  vec3 N  = calcNormal(p, baseR, noiseAmp, noiseFreq);
  vec3 L  = normalize(vec3(-0.4, 0.6, 0.8));
  float NdotL = max(dot(N, L), 0.0);
  vec3 H      = normalize(L + vec3(0,0,1));
  float spec  = pow(max(dot(N, H), 0.0), mix(8.0, 64.0, 1.0 - roughness));

  vec3 diffuse  = vAlbedo * NdotL * 0.7;
  vec3 ambient  = vAlbedo * 0.35;
  vec3 specular = vec3(spec * mix(0.02, 0.4, metallic));
  float fresnel = pow(1.0 - max(dot(N, vec3(0,0,1)), 0.0), 3.0);
  vec3 rim      = vGlow * fresnel * 0.35;
  vec3 scatter  = vAlbedo * max(dot(-N, L), 0.0) * 0.1;

  vec3 color = ambient + diffuse + specular + rim + scatter;
  // Filmic tone map
  color = color * (2.51*color + 0.03) / (color*(2.43*color + 0.59) + 0.14);
  color = pow(clamp(color, 0.0, 1.0), vec3(1.0/2.2));

  fragColor = vec4(color, vOpacity);
}
`;

// ─── Edge shaders (simple lines) ─────────────────────────────────────────────
const EDGE_VERT = /* glsl */`#version 300 es
precision highp float;
in vec2 aPos;
uniform vec2 uCanvasSize;
void main() {
  // Input aPos in pixel space, convert to NDC
  vec2 ndc = (aPos / uCanvasSize) * 2.0 - 1.0;
  ndc.y = -ndc.y;
  gl_Position = vec4(ndc, 0.0, 1.0);
}
`;

const EDGE_FRAG = /* glsl */`#version 300 es
precision highp float;
uniform vec3 uEdgeColor;
out vec4 fragColor;
void main() {
  fragColor = vec4(uEdgeColor, 0.45);
}
`;

// ─── Helpers ─────────────────────────────────────────────────────────────────
function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(`Shader compile error:\n${gl.getShaderInfoLog(sh)}\n---\n${src}`);
  }
  return sh;
}

function linkProgram(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, compileShader(gl, gl.VERTEX_SHADER,   vs));
  gl.attachShader(prog, compileShader(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(`Program link error: ${gl.getProgramInfoLog(prog)}`);
  }
  return prog;
}

// ─── WorldRenderer ────────────────────────────────────────────────────────────
export class WorldRenderer {
  private gl: WebGL2RenderingContext;
  private canvas: HTMLCanvasElement;

  // Cell draw resources
  private cellProg!: WebGLProgram;
  private cellVAO!: WebGLVertexArrayObject;
  private instBuf!: WebGLBuffer;

  // Edge draw resources
  private edgeProg!: WebGLProgram;
  private edgeVAO!: WebGLVertexArrayObject;
  private edgeBuf!: WebGLBuffer;

  // Scene data
  private cells: CellData[] = [];
  private edges: EdgeData[] = [];
  private edgeLineCount = 0;

  // Timing
  private _rafId = 0;
  private _time  = 0;
  private _lastTs = 0;
  private _running = false;

  // Stats (public for HUD parity)
  stats = { fps: 0, cellCount: 0 };
  private _frameCount = 0;
  private _fpsTs = 0;

  // Instanced layout: floats per cell
  // iRect(4) + iAlbedo(4) + iGlow(4) + iSDF(4) + iLobe0-3(4×4) = 32
  private static readonly FLOATS_PER_CELL = 32;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { alpha: true, antialias: false });
    if (!gl) throw new Error('WorldRenderer: WebGL2 not available');
    this.gl = gl;
    this._buildCellPipeline();
    this._buildEdgePipeline();
  }

  private _buildCellPipeline(): void {
    const gl = this.gl;
    this.cellProg = linkProgram(gl, CELL_VERT, CELL_FRAG);

    // Quad corners: 6 vertices (2 triangles)
    const corners = new Float32Array([-1,-1, 1,-1, -1,1,  1,-1, 1,1, -1,1]);
    const cornerBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuf);
    gl.bufferData(gl.ARRAY_BUFFER, corners, gl.STATIC_DRAW);

    this.instBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    gl.bufferData(gl.ARRAY_BUFFER, 4, gl.DYNAMIC_DRAW);

    this.cellVAO = gl.createVertexArray()!;
    gl.bindVertexArray(this.cellVAO);

    // aCorner
    gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuf);
    const aCorner = gl.getAttribLocation(this.cellProg, 'aCorner');
    gl.enableVertexAttribArray(aCorner);
    gl.vertexAttribPointer(aCorner, 2, gl.FLOAT, false, 0, 0);

    // Per-instance attributes from instBuf
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    const stride = WorldRenderer.FLOATS_PER_CELL * 4;
    const attrs: [string, number, number][] = [
      // [name, size, offset-floats]
      ['iRect',  4,  0],
      ['iAlbedo',4,  4],
      ['iGlow',  4,  8],
      ['iSDF',   4, 12],
      ['iLobe0', 4, 16],
      ['iLobe1', 4, 20],
      ['iLobe2', 4, 24],
      ['iLobe3', 4, 28],
    ];
    for (const [name, size, off] of attrs) {
      const loc = gl.getAttribLocation(this.cellProg, name);
      if (loc < 0) continue;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, off * 4);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindVertexArray(null);
  }

  private _buildEdgePipeline(): void {
    const gl = this.gl;
    this.edgeProg = linkProgram(gl, EDGE_VERT, EDGE_FRAG);

    this.edgeBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeBuf);
    gl.bufferData(gl.ARRAY_BUFFER, 4, gl.DYNAMIC_DRAW);

    this.edgeVAO = gl.createVertexArray()!;
    gl.bindVertexArray(this.edgeVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeBuf);
    const aPos = gl.getAttribLocation(this.edgeProg, 'aPos');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  // ── setScene: pack data into GPU buffers ──────────────────────────────────
  setScene(cells: CellData[], edges: EdgeData[]): void {
    this.cells = cells;
    this.edges = edges;
    this._packCells();
    this._packEdges();
  }

  private _canvasToNDC(px: number, py: number, pw: number, ph: number): [number,number,number,number] {
    const W = this.canvas.width;
    const H = this.canvas.height;
    // pixel bbox centre → NDC centre, pixel half-size → NDC half-size
    const cx = ((px + pw * 0.5) / W) * 2.0 - 1.0;
    const cy = 1.0 - ((py + ph * 0.5) / H) * 2.0;
    const hw = (pw * 0.5) / W * 2.0;
    const hh = (ph * 0.5) / H * 2.0;
    return [cx, cy, hw, hh];
  }

  private _packCells(): void {
    const gl = this.gl;
    const N  = this.cells.length;
    const FPC = WorldRenderer.FLOATS_PER_CELL;
    const buf = new Float32Array(N * FPC);
    for (let i = 0; i < N; i++) {
      const c = this.cells[i];
      const [cx, cy, hw, hh] = this._canvasToNDC(c.x, c.y, c.w, c.h);
      const base = i * FPC;
      // iRect
      buf[base+0] = cx; buf[base+1] = cy; buf[base+2] = hw; buf[base+3] = hh;
      // iAlbedo
      buf[base+4] = c.albedo[0]; buf[base+5] = c.albedo[1];
      buf[base+6] = c.albedo[2]; buf[base+7] = c.opacity ?? 0.9;
      // iGlow: rgb + baseRadius
      const gc = c.glowColor ?? c.albedo;
      buf[base+8]  = gc[0]; buf[base+9]  = gc[1]; buf[base+10] = gc[2];
      buf[base+11] = c.sdfBaseRadius ?? 0.6;
      // iSDF: noiseAmp, noiseFreq, roughness, metallic
      buf[base+12] = c.sdfNoiseAmp  ?? 0.04;
      buf[base+13] = c.sdfNoiseFreq ?? 4.0;
      buf[base+14] = c.roughness    ?? 0.55;
      buf[base+15] = c.metallic     ?? 0.1;
      // lobes (up to 4)
      const lobes = c.sdfLobes ?? [];
      for (let li = 0; li < 4; li++) {
        const lb = lobes[li];
        const off = base + 16 + li * 4;
        buf[off+0] = lb ? lb.angle    : 0.0;
        buf[off+1] = lb ? lb.distance : 0.0;
        buf[off+2] = lb ? lb.radius   : 0.0;
        buf[off+3] = 0.0;
      }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    gl.bufferData(gl.ARRAY_BUFFER, buf, gl.DYNAMIC_DRAW);
    this.stats.cellCount = N;
  }

  private _packEdges(): void {
    const gl = this.gl;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const verts: number[] = [];

    // Map cell_id → centre pixel
    const cellCentre = new Map<string, [number,number]>();
    for (const c of this.cells) {
      cellCentre.set(c.cell_id, [c.x + c.w * 0.5, c.y + c.h * 0.5]);
    }

    for (const e of this.edges) {
      const src = cellCentre.get(e.source);
      const dst = cellCentre.get(e.target);
      if (!src || !dst) continue;
      const pts: [number,number][] = [src, ...(e.controlPoints ?? []), dst];
      for (let i = 0; i < pts.length - 1; i++) {
        verts.push(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1]);
      }
    }

    this.edgeLineCount = verts.length / 2;
    if (verts.length === 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.DYNAMIC_DRAW);
  }

  // ── frame: one rAF tick ───────────────────────────────────────────────────
  frame(time: number): void {
    const dt = Math.min((time - this._lastTs) / 1000, 0.1);
    this._lastTs = time;
    this._time  += dt;

    const gl = this.gl;
    const W  = this.canvas.width;
    const H  = this.canvas.height;

    gl.viewport(0, 0, W, H);
    gl.clearColor(0.04, 0.04, 0.06, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // ── draw edges ──
    if (this.edgeLineCount > 0) {
      gl.useProgram(this.edgeProg);
      gl.uniform2f(gl.getUniformLocation(this.edgeProg, 'uCanvasSize'), W, H);
      gl.uniform3f(gl.getUniformLocation(this.edgeProg, 'uEdgeColor'), 0.2, 0.75, 0.55);
      gl.bindVertexArray(this.edgeVAO);
      gl.drawArrays(gl.LINES, 0, this.edgeLineCount);
    }

    // ── draw cells ──
    const N = this.cells.length;
    if (N > 0) {
      gl.useProgram(this.cellProg);
      gl.uniform1f(gl.getUniformLocation(this.cellProg, 'uTime'), this._time);
      gl.bindVertexArray(this.cellVAO);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, N);
    }

    gl.bindVertexArray(null);

    // FPS counter
    this._frameCount++;
    if (time - this._fpsTs >= 1000) {
      this.stats.fps = this._frameCount;
      this._frameCount = 0;
      this._fpsTs = time;
    }
  }

  // ── start / stop ─────────────────────────────────────────────────────────
  start(): void {
    if (this._running) return;
    this._running = true;
    this._fpsTs  = performance.now();
    const loop   = (ts: number) => {
      if (!this._running) return;
      this.frame(ts);
      this._rafId = requestAnimationFrame(loop);
    };
    this._rafId = requestAnimationFrame(loop);
  }

  stop(): void {
    this._running = false;
    cancelAnimationFrame(this._rafId);
  }

  // ── live geometry update (SSE geometry_update events) ─────────────────────
  updateCellGeometry(cellId: string, geometry: Record<string, unknown>): void {
    const idx = this.cells.findIndex(c => c.cell_id === cellId);
    if (idx < 0) return;
    const c   = this.cells[idx];
    const sdf = (geometry.sdf ?? {}) as Record<string, unknown>;
    const surf = (geometry.surface ?? {}) as Record<string, unknown>;
    this.cells[idx] = {
      ...c,
      ...(Array.isArray(surf.albedo) && surf.albedo.length === 3
        ? { albedo: surf.albedo as [number,number,number] } : {}),
      ...(Array.isArray(surf.glow_color) && surf.glow_color.length === 3
        ? { glowColor: surf.glow_color as [number,number,number] } : {}),
      ...(typeof surf.opacity   === 'number' ? { opacity:   surf.opacity   } : {}),
      ...(typeof surf.roughness === 'number' ? { roughness: surf.roughness } : {}),
      ...(typeof surf.metallic  === 'number' ? { metallic:  surf.metallic  } : {}),
      ...(typeof sdf.base_radius   === 'number' ? { sdfBaseRadius: sdf.base_radius   } : {}),
      ...(typeof sdf.noise_amplitude === 'number' ? { sdfNoiseAmp: sdf.noise_amplitude } : {}),
      ...(typeof sdf.noise_frequency === 'number' ? { sdfNoiseFreq: sdf.noise_frequency } : {}),
      ...(Array.isArray(sdf.lobes)
        ? { sdfLobes: sdf.lobes as Array<{ angle: number; distance: number; radius: number }> } : {}),
    };
    this._packCells();
  }
}

// ─── Data loader ──────────────────────────────────────────────────────────────

/** Fetch composite_params.json + per-cell geometry.json, return merged arrays. */
export async function loadWorldScene(): Promise<{ cells: CellData[]; edges: EdgeData[] }> {
  const hexToRGB = (hex: string): [number,number,number] => {
    const n = parseInt(hex.replace('#',''), 16);
    return [(n>>16&0xff)/255, (n>>8&0xff)/255, (n&0xff)/255];
  };

  let raw: Record<string,unknown> | null = null;
  try {
    const r = await fetch('/channels/composite_params.json');
    if (r.ok) raw = await r.json();
  } catch { /* ignore */ }
  if (!raw) {
    console.warn('[world-renderer] composite_params.json unavailable');
    return { cells: [], edges: [] };
  }

  const rawCells = (raw.cells ?? {}) as Record<string, Record<string,unknown>>;
  const rawEdges = (raw.edge_routes ?? {}) as Record<string, Record<string,unknown>>;

  const cells: CellData[] = Object.entries(rawCells).map(([id, cv]) => {
    const ap  = (cv.agent_params ?? {}) as Record<string,unknown>;
    const bbox = (ap.bbox ?? {x:0,y:0,w:120,h:55,z:1}) as Record<string,number>;
    const sp   = (ap.species_params ?? {}) as Record<string,unknown>;
    const color = hexToRGB(String(sp.primary_color ?? '#4488ff'));
    return {
      cell_id: id,
      species: String(sp.species ?? 'cil-eye'),
      x: bbox.x ?? 0,   y: bbox.y ?? 0,
      w: bbox.w ?? 120,  h: bbox.h ?? 55,
      z: bbox.z ?? 1,
      metallic:  0.1,
      roughness: 0.55,
      albedo:    color,
      label:     id,
      glowColor: hexToRGB(String(sp.glow_color ?? sp.primary_color ?? '#4488ff')),
      sdfShape:  sp.sdf_shape === 'capsule' ? 'capsule' : 'rounded_rect',
      internalPattern: String(sp.internal_pattern ?? 'none'),
      haloRadius:     Number(sp.halo_radius      ?? 0.15),
      numRays:        Number(sp.num_rays          ?? 0),
      focalIntensity: Number(sp.focal_intensity   ?? 0.0),
      animationSpeed: Number(sp.animation_speed   ?? 1.0),
      opacity:        Number(ap.opacity           ?? 0.9),
    } satisfies CellData;
  });

  const edges: EdgeData[] = Object.entries(rawEdges).map(([id, ev]) => ({
    edge_id:       id,
    source:        String(ev.source),
    target:        String(ev.target),
    controlPoints: (ev.control_points ?? []) as [number,number][],
    color:         [0.2, 0.8, 0.6] as [number,number,number],
  }));

  // ── Fetch geometry.json for each cell ────────────────────────────────────
  const geoResults = await Promise.all(
    cells.map(c =>
      fetch('/channels/cell/' + c.cell_id + '/geometry.json')
        .then(r => r.ok ? r.json() : null)
        .catch(() => null)
    )
  );
  let geoHit = 0;
  for (let i = 0; i < cells.length; i++) {
    const geo = geoResults[i] as Record<string,unknown> | null;
    if (!geo) continue;
    geoHit++;
    const sdf  = (geo.sdf  ?? {}) as Record<string,unknown>;
    const surf = (geo.surface ?? {}) as Record<string,unknown>;
    cells[i] = {
      ...cells[i],
      ...(Array.isArray(surf.albedo) && surf.albedo.length === 3
        ? { albedo: surf.albedo as [number,number,number] } : {}),
      ...(Array.isArray(surf.glow_color) && surf.glow_color.length === 3
        ? { glowColor: surf.glow_color as [number,number,number] } : {}),
      ...(typeof surf.opacity   === 'number' ? { opacity:   surf.opacity   } : {}),
      ...(typeof surf.roughness === 'number' ? { roughness: surf.roughness } : {}),
      ...(typeof surf.metallic  === 'number' ? { metallic:  surf.metallic  } : {}),
      sdfBaseRadius: typeof sdf.base_radius     === 'number' ? sdf.base_radius     : undefined,
      sdfLobes:      Array.isArray(sdf.lobes)   ? sdf.lobes as Array<{ angle: number; distance: number; radius: number }> : undefined,
      sdfNoiseAmp:   typeof sdf.noise_amplitude === 'number' ? sdf.noise_amplitude : undefined,
      sdfNoiseFreq:  typeof sdf.noise_frequency === 'number' ? sdf.noise_frequency : undefined,
    } as CellData;
  }
  console.info(`[world-renderer] merged ${geoHit}/${cells.length} cells`);
  return { cells, edges };
}
