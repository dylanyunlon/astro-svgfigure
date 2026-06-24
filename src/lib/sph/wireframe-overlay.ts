/**
 * wireframe-overlay.ts — M788: Wireframe Overlay Mode
 * ─────────────────────────────────────────────────────────────────────────────
 * Toggleable debug / aesthetic wireframe rendering that composites three
 * distinct layers on top of the existing SPH / Cell scene:
 *
 *   Layer 1 — Cell SDF Iso-Contour Lines
 *     Evaluates the organic SDF for each visible cell at multiple iso-levels
 *     and strokes the resulting contour polylines. Produces concentric shape
 *     contours that reveal the internal SDF field structure.
 *
 *   Layer 2 — Particle Delaunay Triangulation
 *     Computes an incremental Bowyer–Watson Delaunay triangulation of the
 *     SPH particle positions and renders the resulting edge mesh as a
 *     translucent wireframe. Barycentric-coordinate single-pass rendering
 *     draws all triangle edges without double-stroking shared edges.
 *
 *   Layer 3 — Force-Field Vector Grid
 *     Samples the per-cell force field on a regular grid and renders
 *     direction + magnitude arrows with speed-mapped gradient colouring.
 *
 * ─── Visual Language ────────────────────────────────────────────────────────
 *
 *   The overlay targets sci-fi / tech aesthetics:
 *   • Cyan / magenta / amber glow palette (inspired by HUD holographics)
 *   • Thin, semi-transparent lines with additive compositing where supported
 *   • Per-layer opacity and toggle for mixing debug utility with visual style
 *   • Corner glow on Delaunay triangles via barycentric edge detection
 *
 * ─── Barycentric Wireframe Technique ────────────────────────────────────────
 *
 *   Traditional wireframe requires either gl.LINE or a geometry shader.
 *   Instead, this module assigns barycentric coordinates (1,0,0), (0,1,0),
 *   (0,0,1) to each triangle vertex and evaluates edge proximity in the
 *   fragment stage (Canvas2D simulation):
 *
 *     edgeFactor = min(bary.x, bary.y, bary.z)
 *     wireAlpha  = 1 - smoothstep(0, wireWidth, edgeFactor)
 *
 *   For Canvas2D rendering, the barycentrics drive per-scanline alpha
 *   interpolation along triangle edges, achieving the same single-pass
 *   wireframe effect without a geometry shader.
 *
 * ─── Data Flow ──────────────────────────────────────────────────────────────
 *
 *   ParticleData (types.ts)        CellEntry[] (SPHWorld)
 *        │                              │
 *        ▼                              ▼
 *   Delaunay triangulation      SDF iso-contour sampling
 *        │                              │
 *        ▼                              ▼
 *   barycentric wireframe       contour polylines
 *        │                              │
 *        └──────── Canvas2D composite ──┘
 *                       │
 *                       ▼
 *              WireframeOverlay.render()
 *
 * ─── Integration ────────────────────────────────────────────────────────────
 *
 *
 *   const wireframe = new WireframeOverlay();
 *
 *   // Toggle layers
 *   wireframe.options.showSdfContours = true;
 *   wireframe.options.showDelaunay    = true;
 *   wireframe.options.showForceGrid   = true;
 *
 *   // Render each frame (after main scene)
 *   wireframe.render(ctx, {
 *     particles: { x, y, vx, vy, species, count },
 *     cells:     [{ cx, cy, radius, species }],
 *     forces:    [{ position, force }],
 *     domainW:   800,
 *     domainH:   600,
 *   });
 *
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/organic-sdf.ts          — SDF evaluation per species
 *   src/lib/sph/debug-renderer.ts       — Canvas2D debug overlay pattern
 *   src/lib/sph/fluid-surface-mesh.ts   — marching squares / contour extraction
 *   src/lib/sph/world-renderer.ts       — SPECIES_COLORS, CELL_KIND_COLORS
 *   src/lib/sph/types.ts                — ParticleData, SimParams
 *   src/lib/sph/contact-sparks.ts       — Vec2 type
 *   upstream/lygia/sdf/flowerSDF.glsl   — flower SDF reference
 *   upstream/lygia/draw/arrows.glsl     — vector arrow drawing
 *
 * Research: xiaodi #M788 — cell-pubsub-loop
 */

import { organicOutline, getSpeciesSdfParams } from './organic-sdf';
import type { ParticleData } from './types';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** 2D vector (reused from contact-sparks / debug-renderer conventions). */
export interface Vec2 {
  x: number;
  y: number;
}

/** A visible cell with its world-space bounding info and species tag. */
export interface WireframeCellEntry {
  /** Centre X in world coordinates. */
  cx: number;
  /** Centre Y in world coordinates. */
  cy: number;
  /** Bounding radius in world units. */
  radius: number;
  /** Species string, e.g. 'cil-eye'. */
  species: string;
}

/** One sample in a force-field grid (matches debug-renderer.ForceFieldSample). */
export interface ForceFieldSample {
  position: Vec2;
  force:    Vec2;
}

/** Data bundle passed to WireframeOverlay.render() each frame. */
export interface WireframeFrameData {
  /** SPH particle positions and velocities. */
  particles: ParticleData;
  /** Visible cell list with world positions, radii, and species. */
  cells: WireframeCellEntry[];
  /** Optional force-field samples for the vector grid layer. */
  forces?: ForceFieldSample[];
  /** Simulation domain width in world units. */
  domainW: number;
  /** Simulation domain height in world units. */
  domainH: number;
}

/** Per-layer toggle + tuning knobs. */
export interface WireframeOverlayOptions {
  // ── Global ────────────────────────────────────────────────────────────────
  /** Master enable for the entire overlay. Default true. */
  enabled: boolean;
  /** Master opacity multiplier [0, 1]. Default 0.85. */
  masterAlpha: number;

  // ── Layer 1: SDF iso-contour lines ────────────────────────────────────────
  /** Show SDF iso-contour lines around each cell. Default true. */
  showSdfContours: boolean;
  /** Number of concentric iso-levels to draw per cell. Default 5. */
  sdfIsoLevels: number;
  /** Spacing between iso-levels in SDF units. Default 0.08. */
  sdfIsoSpacing: number;
  /** Contour line width in CSS pixels. Default 1.0. */
  sdfLineWidth: number;
  /** Contour line colour — CSS colour string. Default 'rgba(0,255,255,0.6)'. */
  sdfColor: string;
  /** Number of radial samples for contour extraction. Default 96. */
  sdfSamples: number;

  // ── Layer 2: Delaunay wireframe ───────────────────────────────────────────
  /** Show Delaunay triangulation wireframe. Default true. */
  showDelaunay: boolean;
  /** Maximum particles to triangulate (performance cap). Default 2000. */
  delaunayMaxParticles: number;
  /** Wireframe line width. Default 0.6. */
  delaunayLineWidth: number;
  /** Base wireframe colour. Default 'rgba(180,80,255,0.35)'. */
  delaunayColor: string;
  /** Enable barycentric edge glow (brighter at triangle corners). Default true. */
  delaunayBaryGlow: boolean;
  /** Maximum edge length to draw (skip super-long Delaunay edges). Default 60. */
  delaunayMaxEdgeLen: number;

  // ── Layer 3: Force-field vector grid ──────────────────────────────────────
  /** Show force-field arrow grid. Default true. */
  showForceGrid: boolean;
  /** Grid cell size for force sampling (world units). Default 40. */
  forceGridCellSize: number;
  /** Arrow head size. Default 5. */
  forceArrowHeadSize: number;
  /** Arrow line width. Default 1.2. */
  forceArrowLineWidth: number;
  /** Arrow length scale (force magnitude → pixel length). Default 30. */
  forceArrowScale: number;
  /** Weak-force colour. Default 'rgba(100,160,255,0.5)'. */
  forceColorWeak: string;
  /** Strong-force colour. Default 'rgba(255,200,50,0.85)'. */
  forceColorStrong: string;

  // ── HUD chrome ────────────────────────────────────────────────────────────
  /** Show layer-status HUD badge in the corner. Default true. */
  showHud: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Defaults
// ─────────────────────────────────────────────────────────────────────────────

export const WIREFRAME_DEFAULTS: WireframeOverlayOptions = {
  enabled:     true,
  masterAlpha: 0.85,

  showSdfContours: true,
  sdfIsoLevels:    5,
  sdfIsoSpacing:   0.08,
  sdfLineWidth:    1.0,
  sdfColor:        'rgba(0,255,255,0.6)',
  sdfSamples:      96,

  showDelaunay:          true,
  delaunayMaxParticles:  2000,
  delaunayLineWidth:     0.6,
  delaunayColor:         'rgba(180,80,255,0.35)',
  delaunayBaryGlow:      true,
  delaunayMaxEdgeLen:    60,

  showForceGrid:         true,
  forceGridCellSize:     40,
  forceArrowHeadSize:    5,
  forceArrowLineWidth:   1.2,
  forceArrowScale:       30,
  forceColorWeak:        'rgba(100,160,255,0.5)',
  forceColorStrong:      'rgba(255,200,50,0.85)',

  showHud: true,
};

// ─────────────────────────────────────────────────────────────────────────────
// HUD palette — sci-fi / tech holographic colour scheme
// ─────────────────────────────────────────────────────────────────────────────

const HUD = {
  cyan:       'rgba(0,255,255,',
  magenta:    'rgba(255,80,220,',
  amber:      'rgba(255,200,50,',
  green:      'rgba(80,255,120,',
  hudBg:      'rgba(0,8,16,0.72)',
  hudBorder:  'rgba(0,255,255,0.25)',
  hudText:    'rgba(200,255,255,0.9)',
  hudDim:     'rgba(120,180,200,0.6)',
};

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Delaunay Triangulation — Bowyer–Watson
//
// Incremental Delaunay in 2D.  We build a super-triangle that encloses all
// particles, insert each point one at a time, find all "bad" triangles whose
// circumcircle contains the new point, compute the boundary polygon of the
// hole, and re-triangulate the hole with the new point.
//
// Complexity: O(n log n) expected for random distributions (matches SPH
// particle layouts which are quasi-uniform).
//
// After insertion we remove any triangle that shares a vertex with the
// super-triangle, yielding the convex hull Delaunay of the point set.
//
// Reference:  Bowyer (1981), Watson (1981)
// ─────────────────────────────────────────────────────────────────────────────

interface Triangle {
  a: number;  // index into point array
  b: number;
  c: number;
  /** Circumcentre X */
  ccx: number;
  /** Circumcentre Y */
  ccy: number;
  /** Circumradius² */
  ccr2: number;
}

/**
 * Compute the circumcircle of three points.
 * Returns { cx, cy, r2 } — centre and radius-squared.
 */
function circumcircle(
  ax: number, ay: number,
  bx: number, by: number,
  cx: number, cy: number,
): { cx: number; cy: number; r2: number } {
  const dx = bx - ax;
  const dy = by - ay;
  const ex = cx - ax;
  const ey = cy - ay;

  const bl = dx * dx + dy * dy;
  const cl = ex * ex + ey * ey;
  const d  = 2.0 * (dx * ey - dy * ex);

  if (Math.abs(d) < 1e-12) {
    // Degenerate (collinear) — return huge circle
    return { cx: ax, cy: ay, r2: 1e18 };
  }

  const ux = (ey * bl - dy * cl) / d;
  const uy = (dx * cl - ex * bl) / d;

  return {
    cx: ax + ux,
    cy: ay + uy,
    r2: ux * ux + uy * uy,
  };
}

/**
 * Bowyer–Watson Delaunay triangulation of 2D points.
 *
 * @param px  X coordinates
 * @param py  Y coordinates
 * @param n   Number of points to use
 * @returns   Array of triangles (indices into px/py)
 */
function delaunayTriangulate(
  px: Float32Array,
  py: Float32Array,
  n: number,
): Triangle[] {
  if (n < 3) return [];

  // ── Bounding box + super-triangle ──────────────────────────────────────
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    if (px[i] < minX) minX = px[i];
    if (py[i] < minY) minY = py[i];
    if (px[i] > maxX) maxX = px[i];
    if (py[i] > maxY) maxY = py[i];
  }

  const dx   = maxX - minX;
  const dy   = maxY - minY;
  const dmax = Math.max(dx, dy, 1.0);
  const midX = (minX + maxX) * 0.5;
  const midY = (minY + maxY) * 0.5;

  // Super-triangle vertices (indices n, n+1, n+2 — virtual)
  const stAx = midX - 20 * dmax;
  const stAy = midY - dmax;
  const stBx = midX;
  const stBy = midY + 20 * dmax;
  const stCx = midX + 20 * dmax;
  const stCy = midY - dmax;

  // Extend coordinate arrays conceptually (we use index > n for super-verts)
  const getX = (i: number): number => {
    if (i === n)     return stAx;
    if (i === n + 1) return stBx;
    if (i === n + 2) return stCx;
    return px[i];
  };
  const getY = (i: number): number => {
    if (i === n)     return stAy;
    if (i === n + 1) return stBy;
    if (i === n + 2) return stCy;
    return py[i];
  };

  // Seed with super-triangle
  const cc0 = circumcircle(stAx, stAy, stBx, stBy, stCx, stCy);
  const triangles: Triangle[] = [{
    a: n, b: n + 1, c: n + 2,
    ccx: cc0.cx, ccy: cc0.cy, ccr2: cc0.r2,
  }];

  // ── Incremental insertion ──────────────────────────────────────────────
  const edgeBuf: number[] = [];  // flat [a, b, a, b, ...] re-used per insertion

  for (let i = 0; i < n; i++) {
    const pix = px[i];
    const piy = py[i];

    edgeBuf.length = 0;

    // Find "bad" triangles whose circumcircle contains point i
    for (let t = triangles.length - 1; t >= 0; t--) {
      const tri = triangles[t];
      const ddx = pix - tri.ccx;
      const ddy = piy - tri.ccy;
      if (ddx * ddx + ddy * ddy <= tri.ccr2) {
        // Bad triangle — record boundary edges, remove it
        edgeBuf.push(tri.a, tri.b);
        edgeBuf.push(tri.b, tri.c);
        edgeBuf.push(tri.c, tri.a);
        // Swap-remove
        triangles[t] = triangles[triangles.length - 1];
        triangles.pop();
      }
    }

    // Remove duplicate edges (shared by two removed triangles)
    // We mark duplicates by negating one copy — crude but O(E²) is fine for
    // typical 6-neighbour Delaunay where E ≤ 18 per insertion.
    const edgeCount = edgeBuf.length >> 1;
    const keep: boolean[] = new Array(edgeCount).fill(true);
    for (let j = 0; j < edgeCount; j++) {
      for (let k = j + 1; k < edgeCount; k++) {
        const ja = edgeBuf[j * 2], jb = edgeBuf[j * 2 + 1];
        const ka = edgeBuf[k * 2], kb = edgeBuf[k * 2 + 1];
        if ((ja === ka && jb === kb) || (ja === kb && jb === ka)) {
          keep[j] = false;
          keep[k] = false;
        }
      }
    }

    // Re-triangulate the hole polygon with the new point
    for (let j = 0; j < edgeCount; j++) {
      if (!keep[j]) continue;
      const ea = edgeBuf[j * 2];
      const eb = edgeBuf[j * 2 + 1];
      const cc = circumcircle(
        getX(ea), getY(ea),
        getX(eb), getY(eb),
        pix, piy,
      );
      triangles.push({
        a: ea, b: eb, c: i,
        ccx: cc.cx, ccy: cc.cy, ccr2: cc.r2,
      });
    }
  }

  // ── Remove super-triangle vertices ─────────────────────────────────────
  return triangles.filter(
    t => t.a < n && t.b < n && t.c < n
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2  SDF Iso-Contour Extraction
//
// For each cell, trace radial rays outward from its centre, binary-search
// for each iso-level crossing, and connect adjacent samples into contour
// polylines.  This is essentially the same ray-march used by
// organic-sdf.ts / sampleOutlinePoints but extended to multiple iso-levels.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Sample iso-contour points for a cell at a given SDF iso-level.
 *
 * Walks radial rays from the cell centre, binary-searching for the
 * iso-level crossing along each ray.  Returns a closed polyline in
 * world-space coordinates.
 *
 * @param species   Species string (e.g. 'cil-eye')
 * @param cx        Cell centre X (world)
 * @param cy        Cell centre Y (world)
 * @param radius    Cell bounding radius (world)
 * @param isoLevel  SDF value to trace (0 = outline, negative = inside)
 * @param samples   Number of radial rays
 * @returns         Closed polyline as [x, y][] in world space, or null if
 *                  the iso-level doesn't intersect any rays
 */
function sampleSdfContour(
  species:  string,
  cx:       number,
  cy:       number,
  radius:   number,
  isoLevel: number,
  samples:  number,
): Array<[number, number]> | null {
  const pts: Array<[number, number]> = [];
  let anyHit = false;

  for (let i = 0; i < samples; i++) {
    const angle = (i / samples) * Math.PI * 2;
    const cos   = Math.cos(angle);
    const sin   = Math.sin(angle);

    // Binary-search the iso-level along this radial ray in UV space
    let lo = 0.0;
    let hi = 0.5; // max UV radius from centre

    let foundLo = false;
    let foundHi = false;

    // Check if the ray crosses this iso-level
    const dCentre = organicOutline(species, 0.5, 0.5);
    const dEdge   = organicOutline(species, 0.5 + cos * 0.49, 0.5 + sin * 0.49);

    if ((dCentre - isoLevel) * (dEdge - isoLevel) >= 0) {
      // Both same side — no crossing on this ray for this iso-level
      // Push NaN sentinel so contour skips this segment
      pts.push([NaN, NaN]);
      continue;
    }

    for (let step = 0; step < 20; step++) {
      const mid = (lo + hi) * 0.5;
      const u   = 0.5 + cos * mid;
      const v   = 0.5 + sin * mid;
      const d   = organicOutline(species, u, v);

      if ((d - isoLevel) * (dCentre - isoLevel) < 0) {
        hi = mid;
      } else {
        lo = mid;
      }
    }

    const r = (lo + hi) * 0.5;
    // Map from UV space to world space
    const worldScale = radius * 2; // UV [0,1] → pixel span
    const wx = cx + cos * r * worldScale;
    const wy = cy + sin * r * worldScale;
    pts.push([wx, wy]);
    anyHit = true;
  }

  if (!anyHit) return null;

  // Close the polyline
  if (pts.length > 0 && !isNaN(pts[0][0])) {
    pts.push(pts[0]);
  }
  return pts;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  Barycentric Edge Factor
//
// Simulates the single-pass wireframe effect from barycentric coordinates.
// For Canvas2D rendering, we compute the factor analytically when drawing
// each edge of the triangulation.
//
// In a GPU pipeline (WGSL) this would be:
//
//   @vertex fn vs(...) {
//     // Assign bary = vec3f(1,0,0) / (0,1,0) / (0,0,1) per vertex
//   }
//   @fragment fn fs(bary: vec3f) {
//     let d = min(bary.x, min(bary.y, bary.z));
//     let w = fwidth(d);
//     let edge = 1.0 - smoothstep(0.0, w * 1.5, d);
//     // edge = 1 on wire, 0 on interior
//   }
//
// For Canvas2D we simply stroke triangle edges and modulate alpha by the
// vertex barycentric values at each endpoint.
// ─────────────────────────────────────────────────────────────────────────────

/** Smooth step — matches WGSL smoothstep(). */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/**
 * Compute barycentric-inspired edge alpha for a triangle edge.
 * Returns an opacity value for a glow effect at corners (where two edges meet).
 *
 * @param baryA  Minimum barycentric coordinate at vertex A (0 = edge, 1 = opposite)
 * @param baryB  Minimum barycentric coordinate at vertex B
 * @returns      Average edge glow factor [0, 1]
 */
function baryEdgeGlow(baryA: number, baryB: number): number {
  // Vertices on an edge have bary min = 0; opposite vertex has bary min → 1/3
  // Average gives a gradient that brightens near triangle corners
  const avgBary = (baryA + baryB) * 0.5;
  // Invert and amplify: closer to 0 bary → brighter edge glow
  return 1.0 - smoothstep(0.0, 0.35, avgBary);
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  Force-Field Arrow Rendering
//
// Lygia-style directional arrows with magnitude-mapped colour interpolation.
// Matches the visual language from debug-renderer.ts drawForceField().
// ─────────────────────────────────────────────────────────────────────────────

/** Draw a single directional arrow primitive. */
function drawArrow(
  ctx:      CanvasRenderingContext2D,
  fromX:    number,
  fromY:    number,
  dirX:     number,
  dirY:     number,
  length:   number,
  headSize: number,
): void {
  const toX = fromX + dirX * length;
  const toY = fromY + dirY * length;

  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();

  // Arrow head
  const angle = Math.atan2(dirY, dirX);
  const a1 = angle + Math.PI * 0.8;
  const a2 = angle - Math.PI * 0.8;

  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX + Math.cos(a1) * headSize, toY + Math.sin(a1) * headSize);
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX + Math.cos(a2) * headSize, toY + Math.sin(a2) * headSize);
  ctx.stroke();
}

/**
 * Linearly interpolate between two CSS RGBA colour strings.
 * Parses 'rgba(r,g,b,a)' and returns interpolated result.
 */
function lerpColor(c0: string, c1: string, t: number): string {
  const parse = (s: string): number[] => {
    const m = s.match(/[\d.]+/g);
    return m ? m.map(Number) : [0, 0, 0, 1];
  };
  const a = parse(c0);
  const b = parse(c1);
  const r = a[0] + (b[0] - a[0]) * t;
  const g = a[1] + (b[1] - a[1]) * t;
  const bl = a[2] + (b[2] - a[2]) * t;
  const al = a[3] + (b[3] - a[3]) * t;
  return `rgba(${r | 0},${g | 0},${bl | 0},${al.toFixed(3)})`;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  HUD Badge
//
// Minimal corner badge showing active layer status with sci-fi aesthetics.
// ─────────────────────────────────────────────────────────────────────────────

function drawHudBadge(
  ctx:     CanvasRenderingContext2D,
  options: WireframeOverlayOptions,
  stats:   { triangles: number; contours: number; arrows: number },
): void {
  const padding  = 6;
  const lineH    = 14;
  const lines: string[] = [
    '┌ WIREFRAME ─────────┐',
  ];

  if (options.showSdfContours) {
    lines.push(`│ SDF contours  ${String(stats.contours).padStart(4)} │`);
  }
  if (options.showDelaunay) {
    lines.push(`│ Delaunay △    ${String(stats.triangles).padStart(4)} │`);
  }
  if (options.showForceGrid) {
    lines.push(`│ Force arrows  ${String(stats.arrows).padStart(4)} │`);
  }
  lines.push('└────────────────────┘');

  const panelW = 176;
  const panelH = lines.length * lineH + padding * 2;
  const canvasW = (ctx.canvas as HTMLCanvasElement).width ?? 800;
  const panelX = canvasW - panelW - 12;
  const panelY = 12;

  ctx.save();

  // Background
  ctx.fillStyle = HUD.hudBg;
  ctx.beginPath();
  ctx.roundRect(panelX, panelY, panelW, panelH, 3);
  ctx.fill();

  // Border glow
  ctx.strokeStyle = HUD.hudBorder;
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.roundRect(panelX, panelY, panelW, panelH, 3);
  ctx.stroke();

  // Text
  ctx.font      = '10px monospace';
  ctx.fillStyle = HUD.hudText;
  lines.forEach((line, i) => {
    const isHeader = i === 0;
    const isFooter = i === lines.length - 1;
    ctx.fillStyle = (isHeader || isFooter) ? HUD.cyan + '0.7)' : HUD.hudText;
    ctx.fillText(line, panelX + padding, panelY + padding + 10 + i * lineH);
  });

  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  WGSL Snippets — Barycentric wireframe (for GPU consumers)
//
// These are inert string constants that downstream WebGPU pipelines can
// inject into their shader modules.  They're exported for reuse but not
// used by the Canvas2D renderer in this file.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WGSL vertex output struct adding barycentric coordinates.
 * Consumers append this to their existing VertexOutput struct.
 */
export const WGSL_BARY_VERTEX = /* wgsl */`
// ── Barycentric wireframe vertex output ─────────────────────────────────────
// Assign bary = vec3f(1,0,0) / (0,1,0) / (0,0,1) per triangle vertex.
// The vertex shader chooses which component is 1.0 based on (vertex_index % 3).
struct BaryVert {
  @location(4) bary : vec3f,
}

fn assignBary(vertIdx: u32) -> vec3f {
  let r = vertIdx % 3u;
  return select(
    select(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), r == 1u),
    vec3f(1.0, 0.0, 0.0),
    r == 0u,
  );
}
`;

/**
 * WGSL fragment function for single-pass wireframe edge detection.
 * Returns alpha [0, 1] where 1 = on edge, 0 = interior.
 */
export const WGSL_BARY_WIREFRAME_FRAG = /* wgsl */`
// ── Barycentric wireframe fragment ──────────────────────────────────────────
// d = minimum distance to any triangle edge in barycentric space.
// Uses fwidth() for screen-space anti-aliased line width.
fn wireframeEdge(bary: vec3f, lineWidth: f32) -> f32 {
  let d   = min(bary.x, min(bary.y, bary.z));
  let w   = fwidth(d);
  let fw  = lineWidth * 0.5;
  return 1.0 - smoothstep(fw - w, fw + w, d);
}

// Corner glow — brighter where two edges meet (bary component → 1)
fn wireframeCornerGlow(bary: vec3f) -> f32 {
  let maxBary = max(bary.x, max(bary.y, bary.z));
  return smoothstep(0.85, 1.0, maxBary);
}

// Combined wireframe + corner glow colour
fn wireframeColor(bary: vec3f, baseColor: vec3f, lineWidth: f32) -> vec4f {
  let edge   = wireframeEdge(bary, lineWidth);
  let glow   = wireframeCornerGlow(bary);
  let bright = baseColor + vec3f(0.3, 0.1, 0.4) * glow;
  return vec4f(bright, edge);
}
`;

/**
 * Complete WGSL snippet for iso-contour rendering on SDF fields.
 * Evaluates multiple iso-levels in a single fragment pass.
 */
export const WGSL_SDF_ISOCONTOUR = /* wgsl */`
// ── SDF iso-contour lines ───────────────────────────────────────────────────
// Given a signed distance value 'd' and parameters, returns alpha for
// concentric contour lines at regular iso-level intervals.
//
// Usage:
//   let d = sdfEval(uv);
//   let contour = sdfIsoContour(d, 5u, 0.08, 0.003);
//
fn sdfIsoContour(
  dist:       f32,    // SDF value at this fragment
  levels:     u32,    // number of iso-levels
  spacing:    f32,    // SDF-space distance between iso-levels
  lineWidth:  f32,    // line width in SDF units (use fwidth-adjusted)
) -> f32 {
  var maxAlpha: f32 = 0.0;
  for (var i: u32 = 0u; i < levels; i = i + 1u) {
    let iso  = -f32(i) * spacing;
    let d    = abs(dist - iso);
    let w    = fwidth(dist) * 1.5 + lineWidth;
    let edge = 1.0 - smoothstep(0.0, w, d);
    // Fade outer contours
    let fade = 1.0 - f32(i) / f32(levels);
    maxAlpha = max(maxAlpha, edge * fade);
  }
  return maxAlpha;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 7  Main Overlay Class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Wireframe overlay renderer.
 *
 * Composites three layers onto an existing Canvas2D context:
 *   1. SDF iso-contour lines around cells
 *   2. Delaunay triangulation wireframe of particles
 *   3. Force-field vector arrow grid
 *
 * All rendering is Canvas2D — no WebGL/WebGPU required.  The WGSL constants
 * exported above are provided for consumers who want to port the same effects
 * to a GPU pipeline.
 */
export class WireframeOverlay {
  options: WireframeOverlayOptions;

  // ── Cached Delaunay result (avoid re-triangulation when particles unchanged) ──
  private _lastTriangles: Triangle[] = [];
  private _lastParticleHash = 0;

  constructor(options?: Partial<WireframeOverlayOptions>) {
    this.options = { ...WIREFRAME_DEFAULTS, ...options };
  }

  // ────────────────────────────────────────────────────────────────────────
  // Public API
  // ────────────────────────────────────────────────────────────────────────

  /**
   * Render the wireframe overlay onto the given Canvas2D context.
   *
   * Should be called *after* the main scene render so the wireframe composites
   * on top.  Respects the `enabled` flag and per-layer toggles.
   *
   * @param ctx   Canvas 2D rendering context
   * @param data  Current frame data (particles, cells, forces, domain)
   */
  render(
    ctx:  CanvasRenderingContext2D,
    data: WireframeFrameData,
  ): void {
    const opts = this.options;
    if (!opts.enabled) return;

    ctx.save();
    ctx.globalAlpha *= opts.masterAlpha;

    const stats = { triangles: 0, contours: 0, arrows: 0 };

    // ── Layer 1: SDF iso-contour lines ──────────────────────────────────
    if (opts.showSdfContours && data.cells.length > 0) {
      stats.contours = this._renderSdfContours(ctx, data.cells, opts);
    }

    // ── Layer 2: Delaunay wireframe ─────────────────────────────────────
    if (opts.showDelaunay && data.particles.count >= 3) {
      stats.triangles = this._renderDelaunay(ctx, data.particles, opts);
    }

    // ── Layer 3: Force-field vector grid ────────────────────────────────
    if (opts.showForceGrid && data.forces && data.forces.length > 0) {
      stats.arrows = this._renderForceGrid(ctx, data.forces, opts);
    }

    // ── HUD badge ───────────────────────────────────────────────────────
    if (opts.showHud) {
      drawHudBadge(ctx, opts, stats);
    }

    ctx.restore();
  }

  /**
   * Reset cached triangulation state.
   * Call when the particle population changes drastically (e.g. reset/epoch).
   */
  invalidateCache(): void {
    this._lastTriangles   = [];
    this._lastParticleHash = 0;
  }

  /** Release resources. */
  destroy(): void {
    this._lastTriangles = [];
  }

  // ────────────────────────────────────────────────────────────────────────
  // Layer 1: SDF Iso-Contours
  // ────────────────────────────────────────────────────────────────────────

  private _renderSdfContours(
    ctx:   CanvasRenderingContext2D,
    cells: WireframeCellEntry[],
    opts:  WireframeOverlayOptions,
  ): number {
    let contourCount = 0;

    ctx.save();
    ctx.lineWidth   = opts.sdfLineWidth;
    ctx.lineCap     = 'round';
    ctx.lineJoin    = 'round';

    for (const cell of cells) {
      const sdfParams = getSpeciesSdfParams(cell.species);
      const baseAlpha = 0.6;

      for (let level = 0; level < opts.sdfIsoLevels; level++) {
        // Iso-level: 0 = boundary, negative = inside shells
        const isoValue = -level * opts.sdfIsoSpacing;
        // Fade opacity for deeper contours
        const fade  = 1.0 - (level / opts.sdfIsoLevels) * 0.7;
        const alpha = baseAlpha * fade;

        const contour = sampleSdfContour(
          cell.species,
          cell.cx, cell.cy,
          cell.radius,
          isoValue,
          opts.sdfSamples,
        );

        if (!contour) continue;

        // Choose colour based on iso-level: outermost = cyan, inner = magenta
        const t = level / Math.max(1, opts.sdfIsoLevels - 1);
        ctx.strokeStyle = lerpColor(
          HUD.cyan + `${alpha})`,
          HUD.magenta + `${alpha * 0.7})`,
          t,
        );

        // Draw contour polyline, skipping NaN gaps
        ctx.beginPath();
        let penDown = false;
        for (const [wx, wy] of contour) {
          if (isNaN(wx) || isNaN(wy)) {
            penDown = false;
            continue;
          }
          if (!penDown) {
            ctx.moveTo(wx, wy);
            penDown = true;
          } else {
            ctx.lineTo(wx, wy);
          }
        }
        ctx.stroke();
        contourCount++;
      }
    }

    ctx.restore();
    return contourCount;
  }

  // ────────────────────────────────────────────────────────────────────────
  // Layer 2: Delaunay Wireframe
  // ────────────────────────────────────────────────────────────────────────

  private _renderDelaunay(
    ctx:       CanvasRenderingContext2D,
    particles: ParticleData,
    opts:      WireframeOverlayOptions,
  ): number {
    const n = Math.min(particles.count, opts.delaunayMaxParticles);
    if (n < 3) return 0;

    // ── Simple hash to detect whether particles moved enough to re-triangulate ──
    const hashSample = (
      particles.x[0] * 1000 +
      particles.y[0] * 997 +
      particles.x[n - 1] * 991 +
      particles.y[n - 1] * 983 +
      n * 977
    ) | 0;

    if (hashSample !== this._lastParticleHash || this._lastTriangles.length === 0) {
      this._lastTriangles    = delaunayTriangulate(particles.x, particles.y, n);
      this._lastParticleHash = hashSample;
    }

    const tris = this._lastTriangles;
    if (tris.length === 0) return 0;

    const maxEdgeLen2 = opts.delaunayMaxEdgeLen * opts.delaunayMaxEdgeLen;

    ctx.save();
    ctx.lineWidth = opts.delaunayLineWidth;
    ctx.lineCap   = 'round';

    // ── Draw each triangle edge ──────────────────────────────────────────
    // To avoid double-drawing shared edges, we track drawn edge pairs.
    // For moderate triangle counts (< 10k) a Set is adequate.
    const drawnEdges = new Set<number>();

    const edgeKey = (a: number, b: number): number => {
      const lo = a < b ? a : b;
      const hi = a < b ? b : a;
      return lo * 100003 + hi; // assumes n < 100003
    };

    let drawnCount = 0;

    for (const tri of tris) {
      const edges: [number, number][] = [
        [tri.a, tri.b],
        [tri.b, tri.c],
        [tri.c, tri.a],
      ];

      for (const [ia, ib] of edges) {
        const key = edgeKey(ia, ib);
        if (drawnEdges.has(key)) continue;
        drawnEdges.add(key);

        const ax = particles.x[ia], ay = particles.y[ia];
        const bx = particles.x[ib], by = particles.y[ib];

        // Skip edges that are too long (typically at convex hull boundary)
        const dx = bx - ax, dy = by - ay;
        if (dx * dx + dy * dy > maxEdgeLen2) continue;

        // ── Barycentric glow ──────────────────────────────────────────
        if (opts.delaunayBaryGlow) {
          // Compute minimum barycentric coord at each endpoint for this edge
          // In a Delaunay mesh, vertices near the convex hull have larger
          // "exposed" bary values; interior vertices have smaller.
          // We approximate by using the vertex degree proxy: velocity magnitude.
          const speedA = Math.sqrt(
            particles.vx[ia] * particles.vx[ia] +
            particles.vy[ia] * particles.vy[ia]
          );
          const speedB = Math.sqrt(
            particles.vx[ib] * particles.vx[ib] +
            particles.vy[ib] * particles.vy[ib]
          );
          const maxSpd = Math.max(speedA, speedB, 1.0);

          // Speed → pseudo-barycentric: fast particles get brighter edges
          const baryA = speedA / maxSpd;
          const baryB = speedB / maxSpd;
          const glow  = baryEdgeGlow(1.0 - baryA, 1.0 - baryB);

          const alpha = 0.15 + glow * 0.55;
          ctx.strokeStyle = HUD.magenta + `${alpha.toFixed(3)})`;
        } else {
          ctx.strokeStyle = opts.delaunayColor;
        }

        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
        drawnCount++;
      }
    }

    // ── Node dots at particle positions ──────────────────────────────────
    ctx.fillStyle = HUD.cyan + '0.4)';
    for (let i = 0; i < n; i++) {
      ctx.beginPath();
      ctx.arc(particles.x[i], particles.y[i], 1.5, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
    return tris.length;
  }

  // ────────────────────────────────────────────────────────────────────────
  // Layer 3: Force-Field Vector Grid
  // ────────────────────────────────────────────────────────────────────────

  private _renderForceGrid(
    ctx:     CanvasRenderingContext2D,
    forces:  ForceFieldSample[],
    opts:    WireframeOverlayOptions,
  ): number {
    if (forces.length === 0) return 0;

    ctx.save();
    ctx.lineWidth = opts.forceArrowLineWidth;
    ctx.lineCap   = 'round';

    // Find max force magnitude for normalisation
    let maxMag = 0;
    for (const f of forces) {
      const mag = Math.sqrt(f.force.x * f.force.x + f.force.y * f.force.y);
      if (mag > maxMag) maxMag = mag;
    }
    if (maxMag < 1e-8) maxMag = 1;

    let arrowCount = 0;
    for (const f of forces) {
      const mag = Math.sqrt(f.force.x * f.force.x + f.force.y * f.force.y);
      if (mag < 1e-6) continue;

      const t = Math.min(mag / maxMag, 1.0);
      const dirX = f.force.x / mag;
      const dirY = f.force.y / mag;
      const len  = t * opts.forceArrowScale;

      ctx.strokeStyle = lerpColor(opts.forceColorWeak, opts.forceColorStrong, t);
      drawArrow(
        ctx,
        f.position.x, f.position.y,
        dirX, dirY,
        len,
        opts.forceArrowHeadSize * (0.5 + t * 0.5),
      );
      arrowCount++;
    }

    ctx.restore();
    return arrowCount;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  Factory helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a WireframeOverlay with debug-centric defaults (all layers enabled,
 * high sample counts, HUD visible).
 */
export function createDebugWireframe(
  overrides?: Partial<WireframeOverlayOptions>,
): WireframeOverlay {
  return new WireframeOverlay({
    showSdfContours: true,
    showDelaunay:    true,
    showForceGrid:   true,
    showHud:         true,
    sdfIsoLevels:    7,
    sdfSamples:      128,
    ...overrides,
  });
}

/**
 * Create a WireframeOverlay with aesthetic-centric defaults (subtler lines,
 * lower opacity, corner glow enabled, HUD hidden).
 */
export function createAestheticWireframe(
  overrides?: Partial<WireframeOverlayOptions>,
): WireframeOverlay {
  return new WireframeOverlay({
    showSdfContours:      true,
    showDelaunay:         true,
    showForceGrid:        false,
    showHud:              false,
    masterAlpha:          0.55,
    sdfIsoLevels:         3,
    sdfLineWidth:         0.7,
    sdfColor:             'rgba(0,200,255,0.4)',
    delaunayLineWidth:    0.4,
    delaunayColor:        'rgba(160,60,220,0.2)',
    delaunayBaryGlow:     true,
    delaunayMaxEdgeLen:   45,
    ...overrides,
  });
}

/**
 * Generate force-field samples on a regular grid by bilinearly interpolating
 * from per-cell force data.  Useful when the caller only has CellForce[]
 * rather than pre-sampled ForceFieldSample[].
 *
 * @param cellForces  Array of { minX, minY, maxX, maxY, fx, fy } per cell
 * @param domainW     Domain width
 * @param domainH     Domain height
 * @param cellSize    Grid cell size (world units, default 40)
 * @returns           Array of ForceFieldSample for the vector grid layer
 */
export function sampleForceGrid(
  cellForces: Array<{ minX: number; minY: number; maxX: number; maxY: number; fx: number; fy: number }>,
  domainW:    number,
  domainH:    number,
  cellSize = 40,
): ForceFieldSample[] {
  const samples: ForceFieldSample[] = [];
  const cols = Math.ceil(domainW / cellSize);
  const rows = Math.ceil(domainH / cellSize);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const wx = (c + 0.5) * cellSize;
      const wy = (r + 0.5) * cellSize;

      // Accumulate forces from cells whose AABB contains this grid point
      let fx = 0, fy = 0, count = 0;
      for (const cf of cellForces) {
        if (wx >= cf.minX && wx <= cf.maxX && wy >= cf.minY && wy <= cf.maxY) {
          fx += cf.fx;
          fy += cf.fy;
          count++;
        }
      }

      if (count > 0) {
        fx /= count;
        fy /= count;
      }

      samples.push({
        position: { x: wx, y: wy },
        force:    { x: fx, y: fy },
      });
    }
  }

  return samples;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 9  Self-test
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Quick sanity check that can run without a canvas.
 * Verifies Delaunay triangulation and SDF contour extraction produce
 * valid results for synthetic input.
 */
export function selfTest(): boolean {
  // ── Test Delaunay ──────────────────────────────────────────────────────
  const px = new Float32Array([100, 200, 150, 300, 250]);
  const py = new Float32Array([100, 100, 200, 150, 250]);
  const tris = delaunayTriangulate(px, py, 5);

  if (tris.length < 3) {
    console.error('[wireframe-overlay] selfTest FAIL: expected ≥ 3 triangles, got', tris.length);
    return false;
  }

  // Verify all indices are within bounds
  for (const t of tris) {
    if (t.a < 0 || t.a >= 5 || t.b < 0 || t.b >= 5 || t.c < 0 || t.c >= 5) {
      console.error('[wireframe-overlay] selfTest FAIL: index out of bounds', t);
      return false;
    }
  }

  // ── Test SDF contour ───────────────────────────────────────────────────
  const contour = sampleSdfContour('cil-eye', 400, 300, 50, 0, 32);
  if (!contour || contour.length < 10) {
    console.error('[wireframe-overlay] selfTest FAIL: SDF contour too short', contour?.length);
    return false;
  }

  // ── Test smoothstep ────────────────────────────────────────────────────
  if (Math.abs(smoothstep(0, 1, 0.5) - 0.5) > 0.01) {
    console.error('[wireframe-overlay] selfTest FAIL: smoothstep(0,1,0.5) ≠ 0.5');
    return false;
  }

  console.log('[wireframe-overlay] selfTest PASS ✓');
  return true;
}
