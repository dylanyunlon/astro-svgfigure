/**
 * EdgeRenderer.ts — WebGL edge renderer for cell-to-cell connections
 *
 * Reads channels/edge/*/route.json at runtime, then builds a PixiJS Mesh for
 * every edge using the matching GLSL shader pair:
 *
 *   e1–e6   → edge-line.vert / edge-line.frag   (SDF straight line, width+dash)
 *   skip1/2 → edge-spline.vert / edge-spline.frag (instanced cubic Bézier)
 *
 * The topology.json curvature parameter (0.6 for skip connections) controls the
 * lateral offset of the two Bézier control points, producing the characteristic
 * arc that visually distinguishes skip connections from feed-forward edges.
 *
 * Usage:
 *   const er = new EdgeRenderer(pixiApp, { basePath: '/channels/edge' });
 *   await er.load();
 *   pixiApp.stage.addChild(er.container);
 *   // In your ticker:
 *   er.tick(elapsed);
 *
 * Upstream references:
 *   src/lib/renderers/sdf-cell-renderer.ts  (Mesh + UniformGroup pattern)
 *   upstream/pixijs-engine/src/scene/mesh/shared/Mesh.ts
 *   channels/skeleton/topology.json
 *   channels/edge/*/route.json
 */

import {
  Application,
  Container,
  Mesh,
  MeshGeometry,
  Shader,
  UniformGroup,
  Buffer,
} from 'pixi.js';

// ── Route data shapes ─────────────────────────────────────────────────────────

interface RoutePoint {
  x: number;
  y: number;
}

interface RouteAdvanced {
  semanticType?: string;
  routing?: string;
  curvature?: number;
}

interface EdgeRoute {
  edge_id: string;
  sources: string[];
  targets: string[];
  advanced: RouteAdvanced;
  points: RoutePoint[];
  z: number;
  rerouted_epoch: number;
}

// ── EdgeRenderer options ──────────────────────────────────────────────────────

export interface EdgeRendererOptions {
  /** URL prefix for route.json files, e.g. '/channels/edge'  */
  basePath?: string;
  /** Stroke width in pixels for straight edges (default: 2) */
  lineWidth?: number;
  /** Stroke width in pixels for skip connections (default: 2.5) */
  splineWidth?: number;
  /** Stroke colour for normal edges (default: 0x64B5F6 — light blue) */
  lineColor?: [number, number, number];
  /** Stroke colour for skip connections (default: 0xFFB74D — amber) */
  splineColor?: [number, number, number];
  /** Glow colour for skip connections (default: 0xFF8C00 — deep amber) */
  splineGlowColor?: [number, number, number];
  /** Dash length in pixels; 0 = solid (default: 0) */
  dashLength?: number;
  /** Gap  length in pixels (default: 6) */
  gapLength?: number;
  /** Bézier subdivision count per spline (default: 32) */
  splineSubdivisions?: number;
}

const DEFAULTS: Required<EdgeRendererOptions> = {
  basePath:          '/channels/edge',
  lineWidth:         2,
  splineWidth:       2.5,
  lineColor:         [0.39, 0.71, 0.96],   // #64B5F6
  splineColor:       [1.00, 0.72, 0.30],   // #FFB74D
  splineGlowColor:   [1.00, 0.55, 0.00],   // #FF8C00
  dashLength:        0,
  gapLength:         6,
  splineSubdivisions: 32,
};

// ── Straight-line edge IDs ────────────────────────────────────────────────────
const STRAIGHT_EDGES = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6'];
const SPLINE_EDGES   = ['skip1', 'skip2'];
const ALL_EDGES      = [...STRAIGHT_EDGES, ...SPLINE_EDGES];

// ── GLSL sources (inlined so no asset-pipeline dependency) ────────────────────
// These must stay byte-for-byte identical to the .vert/.frag files in
// src/lib/shaders/ — they are duplicated here so that EdgeRenderer can be
// imported by any bundler without a raw-loader plugin.

const EDGE_LINE_VERT = /* glsl */`#version 300 es
precision highp float;
in vec2  aPosition;
in vec2  aUV;
uniform vec2  uP0;
uniform vec2  uP1;
uniform float uLineWidth;
uniform vec2  uResolution;
out vec2  vUV;
out vec2  vP0;
out vec2  vP1;
out float vHalfWidth;
out vec2  vFragCoordPx;
vec2 pixelToNDC(vec2 px) { return (px / uResolution) * 2.0 - 1.0; }
void main() {
  float halfW = uLineWidth * 0.5 + 1.5;
  vec2  dir   = uP1 - uP0;
  float len   = length(dir);
  vec2  unit  = (len > 0.001) ? dir / len : vec2(1.0, 0.0);
  vec2  perp  = vec2(-unit.y, unit.x);
  vec2  longOff  = mix(-unit * halfW, unit  * halfW, aUV.x);
  vec2  perpOff  = mix(-perp * halfW, perp  * halfW, aUV.y);
  vec2  anchor   = mix(uP0, uP1, aUV.x);
  vec2  cornerPx = anchor + longOff + perpOff;
  vFragCoordPx = cornerPx;
  vP0          = uP0;
  vP1          = uP1;
  vHalfWidth   = halfW - 1.5;
  vUV          = aUV;
  vec2 ndc = pixelToNDC(cornerPx);
  ndc.y    = -ndc.y;
  gl_Position = vec4(ndc, 0.0, 1.0);
}`;

const EDGE_LINE_FRAG = /* glsl */`#version 300 es
precision highp float;
in vec2  vFragCoordPx;
in vec2  vP0;
in vec2  vP1;
in float vHalfWidth;
in vec2  vUV;
uniform vec3  uColor;
uniform float uAlpha;
uniform float uLineWidth;
uniform float uDashLength;
uniform float uGapLength;
uniform vec3  uGlowColor;
uniform float uGlowRadius;
uniform float uGlowAlpha;
uniform float uTime;
uniform float uDashOffset;
out vec4 finalColor;
vec2 sdSegment(vec2 p, vec2 a, vec2 b) {
  vec2 pa=p-a; vec2 ba=b-a;
  float h=clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0);
  return vec2(length(pa-ba*h),h);
}
float dashMask(float t_px) {
  if (uDashLength < 0.5) return 1.0;
  float period = uDashLength + uGapLength;
  float phase  = mod(t_px + uTime*40.0 + uDashOffset, period);
  float on = smoothstep(0.0,1.0,phase)*(1.0-smoothstep(uDashLength-1.0,uDashLength,phase));
  return clamp(on,0.0,1.0);
}
void main() {
  float segLen = length(vP1 - vP0);
  vec2  sd     = sdSegment(vFragCoordPx, vP0, vP1);
  float dist   = sd.x;
  float t_px   = sd.y * segLen;
  float strokeAlpha = (1.0 - smoothstep(vHalfWidth-0.75, vHalfWidth+0.75, dist)) * dashMask(t_px);
  float glowAlpha = 0.0;
  if (uGlowRadius > 0.5) {
    float gd = max(0.0, dist - vHalfWidth);
    glowAlpha = uGlowAlpha * exp(-gd*gd/(uGlowRadius*uGlowRadius*0.5)) * dashMask(t_px);
  }
  vec3  col   = mix(uGlowColor, uColor, strokeAlpha);
  float alpha = max(strokeAlpha, glowAlpha) * uAlpha;
  if (alpha < 0.004) discard;
  finalColor = vec4(col * alpha, alpha);
}`;

const EDGE_SPLINE_VERT = /* glsl */`#version 300 es
precision highp float;
in vec2  aUV;
in float aInstanceId;
uniform vec2  uP0;
uniform vec2  uP1;
uniform vec2  uCtrl0;
uniform vec2  uCtrl1;
uniform float uLineWidth;
uniform vec2  uResolution;
uniform float uSubdivisions;
out float vT;
out vec2  vTangentDir;
out vec2  vCurvePx;
out float vHalfWidth;
out vec2  vFragCoordPx;
vec2 bezier(float t) {
  float mt=1.0-t; float mt2=mt*mt; float t2=t*t;
  return mt2*mt*uP0+3.0*mt2*t*uCtrl0+3.0*mt*t2*uCtrl1+t2*t*uP1;
}
vec2 bezierTangent(float t) {
  float mt=1.0-t;
  return 3.0*(mt*mt*(uCtrl0-uP0)+2.0*mt*t*(uCtrl1-uCtrl0)+t*t*(uP1-uCtrl1));
}
vec2 pixelToNDC(vec2 px) { return (px/uResolution)*2.0-1.0; }
void main() {
  float tA=(aInstanceId)/uSubdivisions;
  float tB=(aInstanceId+1.0)/uSubdivisions;
  float tSide=mix(tA,tB,aUV.x);
  float tMid=(tA+tB)*0.5;
  float halfW=uLineWidth*0.5+1.5;
  vec2 posPx=bezier(tSide);
  vec2 tanRaw=bezierTangent(tSide);
  float tanLen=length(tanRaw);
  vec2 unit=(tanLen>0.001)?tanRaw/tanLen:vec2(1.0,0.0);
  vec2 perp=vec2(-unit.y,unit.x);
  float perpSign=mix(-1.0,1.0,aUV.y);
  vec2 cornerPx=posPx+perp*(halfW*perpSign);
  vFragCoordPx=cornerPx;
  vT=tSide;
  vTangentDir=unit;
  vCurvePx=bezier(tMid);
  vHalfWidth=halfW-1.5;
  vec2 ndc=pixelToNDC(cornerPx);
  ndc.y=-ndc.y;
  gl_Position=vec4(ndc,0.0,1.0);
}`;

const EDGE_SPLINE_FRAG = /* glsl */`#version 300 es
precision highp float;
in float vT;
in vec2  vTangentDir;
in vec2  vCurvePx;
in float vHalfWidth;
in vec2  vFragCoordPx;
uniform vec3  uColor;
uniform float uAlpha;
uniform float uLineWidth;
uniform float uDashLength;
uniform float uGapLength;
uniform vec3  uGlowColor;
uniform float uGlowRadius;
uniform float uGlowAlpha;
uniform float uTime;
uniform float uArcLength;
uniform float uCurvature;
out vec4 finalColor;
float approxDist() {
  vec2 diff=vFragCoordPx-vCurvePx;
  vec2 perp=vec2(-vTangentDir.y,vTangentDir.x);
  return abs(dot(diff,perp));
}
float dashMask(float t) {
  if(uDashLength<0.5) return 1.0;
  float t_px=t*uArcLength;
  float period=uDashLength+uGapLength;
  float phase=mod(t_px-uTime*50.0,period);
  float on=smoothstep(0.0,1.0,phase)*(1.0-smoothstep(uDashLength-1.0,uDashLength,phase));
  return clamp(on,0.0,1.0);
}
void main() {
  float dist=approxDist();
  float dash=dashMask(vT);
  float strokeAlpha=(1.0-smoothstep(vHalfWidth-0.75,vHalfWidth+0.75,dist))*dash;
  float glowAlpha=0.0;
  if(uGlowRadius>0.5){
    float gd=max(0.0,dist-vHalfWidth);
    glowAlpha=uGlowAlpha*exp(-gd*gd/(uGlowRadius*uGlowRadius*0.5))*dash;
  }
  vec3 strokeCol=mix(uColor,uGlowColor,uCurvature*0.25);
  vec3 col=mix(uGlowColor,strokeCol,strokeAlpha);
  float alpha=max(strokeAlpha,glowAlpha)*uAlpha;
  if(alpha<0.004) discard;
  finalColor=vec4(col*alpha,alpha);
}`;

// ── Geometry helpers ──────────────────────────────────────────────────────────

/** Build a simple quad (2 triangles) with UV. */
function buildQuadGeometry(): MeshGeometry {
  return new MeshGeometry({
    positions: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
    uvs:       new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
    indices:   new Uint32Array([0, 1, 2, 0, 2, 3]),
  });
}

/**
 * Build geometry for a spline with N subdivision instances.
 * Each instance gets 4 vertices; we bake instance IDs as an extra float attribute.
 *
 * Layout (4 verts × N instances):
 *   aUV        — [0,0], [1,0], [1,1], [0,1]  (quad UV)
 *   aInstanceId — float: 0, 0, 0, 0, 1, 1, 1, 1, …
 */
function buildSplineGeometry(subdivisions: number): MeshGeometry {
  const N    = subdivisions;
  const verts = N * 4;

  const uvs       = new Float32Array(verts * 2);
  const instanceIds = new Float32Array(verts);
  const positions = new Float32Array(verts * 2); // placeholder; vertex shader drives position
  const indices   = new Uint32Array(N * 6);

  const quadUVs = [0, 0,  1, 0,  1, 1,  0, 1];

  for (let i = 0; i < N; i++) {
    const base = i * 4;
    for (let v = 0; v < 4; v++) {
      uvs[(base + v) * 2]     = quadUVs[v * 2];
      uvs[(base + v) * 2 + 1] = quadUVs[v * 2 + 1];
      instanceIds[base + v]   = i;
      positions[(base + v) * 2]     = 0;
      positions[(base + v) * 2 + 1] = 0;
    }
    // Triangles
    const ib = i * 6;
    indices[ib]     = base;
    indices[ib + 1] = base + 1;
    indices[ib + 2] = base + 2;
    indices[ib + 3] = base;
    indices[ib + 4] = base + 2;
    indices[ib + 5] = base + 3;
  }

  const geom = new MeshGeometry({ positions, uvs, indices });
  // Attach instance IDs as a custom attribute
  (geom as any).addAttribute('aInstanceId', {
    buffer: new Buffer({ data: instanceIds }),
    format: 'float32',
    stride: 4,
    offset: 0,
  });
  return geom;
}

// ── Arc-length estimate (straight-chord sum over control polygon) ─────────────

function estimateBezierArcLength(
  p0: RoutePoint,
  ctrl0: RoutePoint,
  ctrl1: RoutePoint,
  p1: RoutePoint,
  samples = 20
): number {
  let len = 0;
  let prev = p0;
  for (let i = 1; i <= samples; i++) {
    const t  = i / samples;
    const mt = 1 - t;
    const x  = mt*mt*mt*p0.x + 3*mt*mt*t*ctrl0.x + 3*mt*t*t*ctrl1.x + t*t*t*p1.x;
    const y  = mt*mt*mt*p0.y + 3*mt*mt*t*ctrl0.y + 3*mt*t*t*ctrl1.y + t*t*t*p1.y;
    const dx = x - prev.x;
    const dy = y - prev.y;
    len += Math.sqrt(dx*dx + dy*dy);
    prev = { x, y };
  }
  return len;
}

// ── Bézier control-point derivation from curvature ───────────────────────────

function deriveControlPoints(
  p0: RoutePoint,
  p1: RoutePoint,
  midPt: RoutePoint,    // the middle waypoint from route.json (natural guide)
  curvature: number
): { ctrl0: RoutePoint; ctrl1: RoutePoint } {
  // Use the midpoint as the "apex" of the arc.
  // ctrl0 = lerp(p0, midPt, curvature)
  // ctrl1 = lerp(p1, midPt, curvature)
  const ctrl0: RoutePoint = {
    x: p0.x + (midPt.x - p0.x) * curvature,
    y: p0.y + (midPt.y - p0.y) * curvature,
  };
  const ctrl1: RoutePoint = {
    x: p1.x + (midPt.x - p1.x) * curvature,
    y: p1.y + (midPt.y - p1.y) * curvature,
  };
  return { ctrl0, ctrl1 };
}

// ── Main EdgeRenderer class ───────────────────────────────────────────────────

export class EdgeRenderer {
  readonly container: Container;
  private app: Application;
  private opts: Required<EdgeRendererOptions>;
  private meshes: Mesh[] = [];
  private uniformGroups: UniformGroup[] = [];
  private elapsed = 0;

  constructor(app: Application, opts: EdgeRendererOptions = {}) {
    this.app       = app;
    this.opts      = { ...DEFAULTS, ...opts };
    this.container = new Container();
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Fetch all route.json files and build WebGL meshes.
   * Call once after construction, before adding container to stage.
   */
  async load(): Promise<void> {
    const routes = await this._fetchAllRoutes();
    for (const route of routes) {
      const mesh = this._buildMesh(route);
      if (mesh) {
        this.container.addChild(mesh);
        this.meshes.push(mesh);
      }
    }
    this.container.sortableChildren = true;
  }

  /**
   * Call every frame with elapsed seconds to drive dash animation.
   */
  tick(dt: number): void {
    this.elapsed += dt;
    for (const ug of this.uniformGroups) {
      const u = (ug as any).uniforms;
      if (u.uTime !== undefined) u.uTime = this.elapsed;
    }
  }

  /**
   * Update resolution when canvas resizes.
   */
  resize(width: number, height: number): void {
    for (const ug of this.uniformGroups) {
      const u = (ug as any).uniforms;
      if (u.uResolution) {
        u.uResolution[0] = width;
        u.uResolution[1] = height;
      }
    }
  }

  // ── Private: fetch ──────────────────────────────────────────────────────────

  private async _fetchAllRoutes(): Promise<EdgeRoute[]> {
    const results: EdgeRoute[] = [];
    await Promise.all(
      ALL_EDGES.map(async (id) => {
        try {
          const url  = `${this.opts.basePath}/${id}/route.json`;
          const resp = await fetch(url);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
          const data: EdgeRoute = await resp.json();
          results.push(data);
        } catch (err) {
          console.warn(`[EdgeRenderer] Could not load route for ${id}:`, err);
        }
      })
    );
    // Restore declaration order (straight first, then splines)
    results.sort((a, b) => ALL_EDGES.indexOf(a.edge_id) - ALL_EDGES.indexOf(b.edge_id));
    return results;
  }

  // ── Private: mesh builders ──────────────────────────────────────────────────

  private _buildMesh(route: EdgeRoute): Mesh | null {
    const isSpline =
      route.advanced?.routing === 'SPLINES' ||
      route.advanced?.semanticType === 'skip_connection' ||
      SPLINE_EDGES.includes(route.edge_id);

    return isSpline
      ? this._buildSplineMesh(route)
      : this._buildLineMesh(route);
  }

  // ── Straight edge mesh ────────────────────────────────────────────────────

  private _buildLineMesh(route: EdgeRoute): Mesh | null {
    const pts = route.points;
    if (pts.length < 2) return null;

    const { app, opts } = this;
    const w = app.renderer.width;
    const h = app.renderer.height;

    // Use first and last point as P0/P1
    const p0 = pts[0];
    const p1 = pts[pts.length - 1];

    const geometry = buildQuadGeometry();

    const uniforms = new UniformGroup({
      uP0:         { value: new Float32Array([p0.x, p0.y]),       type: 'vec2<f32>' },
      uP1:         { value: new Float32Array([p1.x, p1.y]),       type: 'vec2<f32>' },
      uResolution: { value: new Float32Array([w, h]),              type: 'vec2<f32>' },
      uLineWidth:  { value: opts.lineWidth,                        type: 'f32'       },
      uColor:      { value: new Float32Array(opts.lineColor),      type: 'vec3<f32>' },
      uAlpha:      { value: 1.0,                                   type: 'f32'       },
      uDashLength: { value: opts.dashLength,                       type: 'f32'       },
      uGapLength:  { value: opts.gapLength,                        type: 'f32'       },
      uGlowColor:  { value: new Float32Array(opts.lineColor),      type: 'vec3<f32>' },
      uGlowRadius: { value: 0.0,                                   type: 'f32'       },
      uGlowAlpha:  { value: 0.0,                                   type: 'f32'       },
      uTime:       { value: 0.0,                                   type: 'f32'       },
      uDashOffset: { value: 0.0,                                   type: 'f32'       },
    });

    const shader = Shader.from({
      gl: { vertex: EDGE_LINE_VERT, fragment: EDGE_LINE_FRAG },
      resources: { uniforms },
    });

    const mesh = new Mesh({ geometry, shader });
    mesh.zIndex = route.z ?? 2;

    this.uniformGroups.push(uniforms);
    return mesh;
  }

  // ── Spline (skip connection) mesh ─────────────────────────────────────────

  private _buildSplineMesh(route: EdgeRoute): Mesh | null {
    const pts = route.points;
    if (pts.length < 2) return null;

    const { app, opts } = this;
    const w = app.renderer.width;
    const h = app.renderer.height;

    const p0  = pts[0];
    const p1  = pts[pts.length - 1];
    const mid = pts[Math.floor(pts.length / 2)] ?? { x: (p0.x+p1.x)*0.5, y: (p0.y+p1.y)*0.5 };

    const curvature = route.advanced?.curvature ?? 0.5;
    const { ctrl0, ctrl1 } = deriveControlPoints(p0, p1, mid, curvature);
    const arcLen = estimateBezierArcLength(p0, ctrl0, ctrl1, p1);

    const N = opts.splineSubdivisions;
    const geometry = buildSplineGeometry(N);

    const uniforms = new UniformGroup({
      uP0:           { value: new Float32Array([p0.x, p0.y]),          type: 'vec2<f32>' },
      uP1:           { value: new Float32Array([p1.x, p1.y]),          type: 'vec2<f32>' },
      uCtrl0:        { value: new Float32Array([ctrl0.x, ctrl0.y]),    type: 'vec2<f32>' },
      uCtrl1:        { value: new Float32Array([ctrl1.x, ctrl1.y]),    type: 'vec2<f32>' },
      uResolution:   { value: new Float32Array([w, h]),                 type: 'vec2<f32>' },
      uLineWidth:    { value: opts.splineWidth,                         type: 'f32'       },
      uSubdivisions: { value: N,                                        type: 'f32'       },
      uColor:        { value: new Float32Array(opts.splineColor),       type: 'vec3<f32>' },
      uAlpha:        { value: 1.0,                                      type: 'f32'       },
      uDashLength:   { value: opts.dashLength,                          type: 'f32'       },
      uGapLength:    { value: opts.gapLength,                           type: 'f32'       },
      uGlowColor:    { value: new Float32Array(opts.splineGlowColor),   type: 'vec3<f32>' },
      uGlowRadius:   { value: opts.splineWidth * 3.0,                   type: 'f32'       },
      uGlowAlpha:    { value: 0.4,                                      type: 'f32'       },
      uTime:         { value: 0.0,                                      type: 'f32'       },
      uArcLength:    { value: arcLen,                                   type: 'f32'       },
      uCurvature:    { value: curvature,                                type: 'f32'       },
    });

    const shader = Shader.from({
      gl: { vertex: EDGE_SPLINE_VERT, fragment: EDGE_SPLINE_FRAG },
      resources: { uniforms },
    });

    const mesh = new Mesh({ geometry, shader });
    mesh.zIndex = route.z ?? 5;

    this.uniformGroups.push(uniforms);
    return mesh;
  }
}
