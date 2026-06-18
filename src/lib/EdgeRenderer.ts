/**
 * EdgeRenderer.ts — edge渲染 (Graphics-based bezier edges with species_params)
 *
 * Reads edge topology from channels/physics/edge_routes.json and renders each
 * connection as a cubic Bézier curve using PixiJS Graphics.  Stroke colour is
 * derived from the source cell's species_params.primary_color (read from
 * channels/rendering/species/params.json), giving every edge a colour that
 * visually ties it to the cell it originates from.
 *
 * A lightweight "flow" animation shifts a dash-offset along each curve every
 * frame so the viewer perceives signal travelling from source → target.
 *
 * Usage:
 *   const er = new EdgeRenderer(stage, cellMap);
 *   // in ticker:
 *   er.update(dt);
 */

import { Container, Graphics } from 'pixi.js';
import routes from '../../channels/physics/edge_routes.json';
import speciesParamsArr from '../../channels/rendering/species/params.json';

// ── Species params lookup (species → primary_color hex number) ──────────────

interface SpeciesParamEntry {
  species: string;
  primary_color?: string;
  [key: string]: unknown;
}

function hexToNum(hex: string): number {
  return parseInt(hex.replace('#', ''), 16);
}

const DEFAULT_EDGE_COLOR = 0x90A4AE;

const speciesColorMap: Record<string, number> = {};
for (const entry of speciesParamsArr as SpeciesParamEntry[]) {
  if (entry.species && entry.primary_color) {
    speciesColorMap[entry.species] = hexToNum(entry.primary_color);
  }
}

// ── Route data shapes ────────────────────────────────────────────────────────

interface RoutePoint {
  x: number;
  y: number;
}

interface RouteEntry {
  edge_id: string;
  sources: string[];
  targets: string[];
  is_skip: boolean;
  advanced: {
    semanticType?: string;
    routing?: string;
    curvature?: number;
  };
  points: RoutePoint[];
  blocked_by: string[];
  m169: {
    crossings_before: number;
    crossings_after: number;
  };
}

// ── Internal edge descriptor (holds Graphics + animation state) ──────────────

interface EdgeEntry {
  gfx: Graphics;
  edgeId: string;
  sourceSpecies: string;
  paletteColor: number;
  isSkip: boolean;
  p0: RoutePoint;
  p1: RoutePoint;
  ctrl0: RoutePoint;
  ctrl1: RoutePoint;
  arcLength: number;
  flowOffset: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Derive cubic Bézier control points from the route's point list. */
function deriveControlPoints(
  p0: RoutePoint,
  p1: RoutePoint,
  midPt: RoutePoint | null,
  curvature: number,
): { ctrl0: RoutePoint; ctrl1: RoutePoint } {
  if (midPt) {
    // Use the explicit midpoint as arc apex, scaled by curvature
    return {
      ctrl0: {
        x: p0.x + (midPt.x - p0.x) * curvature,
        y: p0.y + (midPt.y - p0.y) * curvature,
      },
      ctrl1: {
        x: p1.x + (midPt.x - p1.x) * curvature,
        y: p1.y + (midPt.y - p1.y) * curvature,
      },
    };
  }
  // Straight-ish edge: controls at 1/3 and 2/3 along the segment
  return {
    ctrl0: { x: p0.x + (p1.x - p0.x) / 3, y: p0.y + (p1.y - p0.y) / 3 },
    ctrl1: { x: p0.x + (p1.x - p0.x) * 2 / 3, y: p0.y + (p1.y - p0.y) * 2 / 3 },
  };
}

/** Rough arc-length estimate by sampling the cubic Bézier. */
function estimateArcLength(
  p0: RoutePoint,
  c0: RoutePoint,
  c1: RoutePoint,
  p1: RoutePoint,
  samples = 20,
): number {
  let len = 0;
  let prev = p0;
  for (let i = 1; i <= samples; i++) {
    const t = i / samples;
    const mt = 1 - t;
    const x = mt * mt * mt * p0.x + 3 * mt * mt * t * c0.x + 3 * mt * t * t * c1.x + t * t * t * p1.x;
    const y = mt * mt * mt * p0.y + 3 * mt * mt * t * c0.y + 3 * mt * t * t * c1.y + t * t * t * p1.y;
    const dx = x - prev.x;
    const dy = y - prev.y;
    len += Math.sqrt(dx * dx + dy * dy);
    prev = { x, y };
  }
  return len;
}

// ── Constants ────────────────────────────────────────────────────────────────

const FLOW_SPEED        = 60;   // pixels per second
const EDGE_WIDTH        = 2;    // stroke width for feed-forward edges
const SKIP_EDGE_WIDTH   = 2.5;  // stroke width for skip connections
const EDGE_ALPHA        = 0.7;  // base alpha for feed-forward
const SKIP_ALPHA        = 0.85; // base alpha for skip connections
const DASH_LENGTH       = 12;   // drawn segment length (px)
const GAP_LENGTH        = 8;    // gap between dashes (px)
const DASH_SEGMENTS     = 40;   // how many sub-segments to approximate the dash pattern

// ── EdgeRenderer ─────────────────────────────────────────────────────────────

export class EdgeRenderer {
  private edges: Graphics[] = [];
  private entries: EdgeEntry[] = [];
  private stage: Container;

  constructor(
    stage: Container,
    cellMap: Map<string, { species: string }>,
  ) {
    this.stage = stage;
    const routeMap = routes as Record<string, RouteEntry>;

    for (const key of Object.keys(routeMap)) {
      const route = routeMap[key];
      const pts = route.points;
      if (pts.length < 2) continue;

      // Resolve source species for colour from species_params.primary_color
      const sourceId = route.sources[0] ?? '';
      const cellInfo = cellMap.get(sourceId);
      const species = cellInfo?.species ?? '';
      const strokeColor = speciesColorMap[species] ?? DEFAULT_EDGE_COLOR;

      const p0 = pts[0];
      const p1 = pts[pts.length - 1];
      const midPt = pts.length >= 3 ? pts[Math.floor(pts.length / 2)] : null;

      const isSkip = route.is_skip || route.advanced?.routing === 'SPLINES';
      const curvature = route.advanced?.curvature ?? (isSkip ? 0.6 : 1.0);

      const { ctrl0, ctrl1 } = deriveControlPoints(p0, p1, midPt, curvature);
      const arcLength = estimateArcLength(p0, ctrl0, ctrl1, p1);

      const gfx = new Graphics();

      // Initial draw
      const width = isSkip ? SKIP_EDGE_WIDTH : EDGE_WIDTH;
      const alpha = isSkip ? SKIP_ALPHA : EDGE_ALPHA;

      gfx.moveTo(p0.x, p0.y);
      gfx.bezierCurveTo(ctrl0.x, ctrl0.y, ctrl1.x, ctrl1.y, p1.x, p1.y);
      gfx.stroke({ color: strokeColor, width, alpha });

      stage.addChild(gfx);
      this.edges.push(gfx);

      this.entries.push({
        gfx,
        edgeId: route.edge_id,
        sourceSpecies: species,
        paletteColor: strokeColor,
        isSkip,
        p0,
        p1,
        ctrl0,
        ctrl1,
        arcLength,
        flowOffset: 0,
      });
    }
  }

  // ── Flow animation — redraws each edge with a shifting dash offset ────────

  update(dt: number): void {
    for (const entry of this.entries) {
      // Advance the flow offset (pixels along the curve)
      entry.flowOffset = (entry.flowOffset + FLOW_SPEED * dt) % (DASH_LENGTH + GAP_LENGTH);

      const { gfx, p0, p1, ctrl0, ctrl1, paletteColor, isSkip, arcLength, flowOffset } = entry;
      const width = isSkip ? SKIP_EDGE_WIDTH : EDGE_WIDTH;
      const alpha = isSkip ? SKIP_ALPHA : EDGE_ALPHA;
      const period = DASH_LENGTH + GAP_LENGTH;

      gfx.clear();

      // Draw dashed Bézier by sampling sub-segments and toggling visibility
      const totalSegments = DASH_SEGMENTS;
      for (let i = 0; i < totalSegments; i++) {
        const tStart = i / totalSegments;
        const tEnd = (i + 1) / totalSegments;

        // Distance along curve for this segment's midpoint
        const tMid = (tStart + tEnd) * 0.5;
        const distAlongCurve = tMid * arcLength;

        // Determine if this segment falls in the "dash" (drawn) or "gap" (hidden)
        const phase = ((distAlongCurve + flowOffset) % period);
        if (phase > DASH_LENGTH) continue; // gap — skip drawing

        // Sample start and end points on the Bézier
        const smt = 1 - tStart;
        const sx = smt * smt * smt * p0.x + 3 * smt * smt * tStart * ctrl0.x + 3 * smt * tStart * tStart * ctrl1.x + tStart * tStart * tStart * p1.x;
        const sy = smt * smt * smt * p0.y + 3 * smt * smt * tStart * ctrl0.y + 3 * smt * tStart * tStart * ctrl1.y + tStart * tStart * tStart * p1.y;

        const emt = 1 - tEnd;
        const ex = emt * emt * emt * p0.x + 3 * emt * emt * tEnd * ctrl0.x + 3 * emt * tEnd * tEnd * ctrl1.x + tEnd * tEnd * tEnd * p1.x;
        const ey = emt * emt * emt * p0.y + 3 * emt * emt * tEnd * ctrl0.y + 3 * emt * tEnd * tEnd * ctrl1.y + tEnd * tEnd * tEnd * p1.y;

        // Also compute a mid-control for a small sub-bezier → just draw a line segment
        gfx.moveTo(sx, sy);
        gfx.lineTo(ex, ey);
        gfx.stroke({ color: paletteColor, width, alpha });
      }
    }
  }
}
