/**
 * pixi-cell-motion.ts — Cell motion physics engine
 *
 * Physical model per frame:
 *   1. Spring force toward ELK target position   (k = 0.01)
 *   2. Pairwise repulsion when cells overlap      (radius = 200)
 *   3. Topology edge spring between connected     (k_edge = 0.005)
 *   4. Velocity damping                           (d = 0.95)
 *   5. Convergence check: stop when all |v| < ε
 *
 * Cells start at random positions and self-organise into the ELK layout.
 *
 * Upstream reference:
 *   skills/pixijs/pixijs-ticker/SKILL.md
 *   upstream/pixijs-engine/src/ticker/
 */

import { Application, Container, Graphics, Text, TextStyle, Ticker } from 'pixi.js';

// ── Types ───────────────────────────────────────────────────────────────────

export interface CellDescriptor {
  cell_id: string;
  label: string;
  species: string;
  bbox: { x: number; y: number; w: number; h: number };
  z: number;
  is_group: boolean;
}

export interface EdgeDescriptor {
  id: string;
  source: string;
  target: string;
  type: string;
}

// ── Physics constants ───────────────────────────────────────────────────────

const DAMPING         = 0.95;    // velocity damping per frame
const K_SPRING        = 0.01;    // spring stiffness — toward ELK target
const K_EDGE          = 0.005;   // spring stiffness — topology edges
const REPULSION_R     = 200;     // repulsion activation radius (px)
const REPULSION_K     = 3000;    // repulsion force magnitude
const CONVERGE_EPS    = 0.15;    // |v| threshold for convergence
const CONVERGE_FRAMES = 30;      // frames all cells must stay below eps

// ── Species colour palette (mirrors pixi-cell-renderer) ────────────────────

const SPECIES_COLOURS: Record<string, { fill: number; stroke: number; glow: number }> = {
  'cil-eye':         { fill: 0x5C6BC0, stroke: 0x3949AB, glow: 0x7986CB },
  'cil-vector':      { fill: 0x66BB6A, stroke: 0x388E3C, glow: 0x81C784 },
  'cil-bolt':        { fill: 0xFFA726, stroke: 0xF57C00, glow: 0xFFCC80 },
  'cil-plus':        { fill: 0xEC407A, stroke: 0xC62828, glow: 0xF48FB1 },
  'cil-arrow-right': { fill: 0x78909C, stroke: 0x455A64, glow: 0xB0BEC5 },
  'cil-filter':      { fill: 0xAB47BC, stroke: 0x7B1FA2, glow: 0xCE93D8 },
  'cil-code':        { fill: 0x29B6F6, stroke: 0x0277BD, glow: 0x4FC3F7 },
  'cil-graph':       { fill: 0x26A69A, stroke: 0x00695C, glow: 0x4DB6AC },
  'cil-settings':    { fill: 0x8D6E63, stroke: 0x4E342E, glow: 0xA1887F },
  'cil-layers':      { fill: 0xEF5350, stroke: 0xB71C1C, glow: 0xEF9A9A },
  'default':         { fill: 0x4A5568, stroke: 0x2D3748, glow: 0x718096 },
};

function gc(species: string) {
  return SPECIES_COLOURS[species] ?? SPECIES_COLOURS['default'];
}

// ── Physics particle ────────────────────────────────────────────────────────

interface Particle {
  desc: CellDescriptor;
  // current centre position
  cx: number;
  cy: number;
  // velocity
  vx: number;
  vy: number;
  // ELK target centre
  tx: number;
  ty: number;
  // half-extents for collision
  hw: number;
  hh: number;
  // pixi display objects
  gfx: Graphics;
  label: Text;
}

// ── Main exported function ──────────────────────────────────────────────────

/**
 * startCellMotion — initialise particles from random positions and
 * animate them toward ELK layout using spring + repulsion physics.
 *
 * @param app      Running PixiJS Application
 * @param cells    CellDescriptor[] from test_cells.json
 * @param edges    EdgeDescriptor[]  from test_cells.json
 * @param canvasW  Logical canvas width  (used to bound random spawn)
 * @param canvasH  Logical canvas height
 * @param scale    Stage scale (passed in from caller)
 * @returns        stop() function to cancel the animation early
 */
export function startCellMotion(
  app: Application,
  cells: CellDescriptor[],
  edges: EdgeDescriptor[],
  canvasW: number,
  canvasH: number,
  scale: number,
): () => void {

  // ── Build stage ────────────────────────────────────────────────────────
  const root = new Container();
  root.scale.set(scale);
  root.position.set(50, 50);
  app.stage.removeChildren();
  app.stage.addChild(root);

  // Draw edge lines layer (behind cells)
  const edgeGfx = new Graphics();
  edgeGfx.zIndex = 0;
  root.addChild(edgeGfx);

  // ── Build particles ────────────────────────────────────────────────────
  const leafCells = cells.filter(c => !c.is_group);

  // Cell lookup by id
  const byId = new Map<string, Particle>();

  const particles: Particle[] = leafCells.map(desc => {
    const col = gc(desc.species);

    // Random spawn anywhere on canvas
    const cx = Math.random() * canvasW;
    const cy = Math.random() * canvasH;

    // ELK target is bbox centre
    const tx = desc.bbox.x + desc.bbox.w / 2;
    const ty = desc.bbox.y + desc.bbox.h / 2;

    // Draw cell
    const gfx = new Graphics();
    gfx.roundRect(-desc.bbox.w / 2, -desc.bbox.h / 2, desc.bbox.w, desc.bbox.h, 8);
    gfx.fill({ color: col.fill, alpha: 0.85 });
    gfx.stroke({ color: col.stroke, width: 1.5 });
    gfx.zIndex = desc.z;
    gfx.position.set(cx, cy);
    root.addChild(gfx);

    // Label
    const label = new Text({
      text: desc.label,
      style: new TextStyle({
        fontSize: 10,
        fill: 0xFFFFFF,
        fontWeight: '500',
        fontFamily: 'system-ui',
        wordWrap: true,
        wordWrapWidth: desc.bbox.w - 8,
        align: 'center',
      }),
    });
    label.anchor.set(0.5);
    label.zIndex = desc.z + 1;
    label.position.set(cx, cy);
    root.addChild(label);

    const p: Particle = {
      desc,
      cx, cy,
      vx: (Math.random() - 0.5) * 4,
      vy: (Math.random() - 0.5) * 4,
      tx, ty,
      hw: desc.bbox.w / 2,
      hh: desc.bbox.h / 2,
      gfx,
      label,
    };
    byId.set(desc.cell_id, p);
    return p;
  });

  // Adjacency list for topology springs (leaf cells only)
  const adj = new Map<string, Set<string>>();
  for (const p of particles) adj.set(p.desc.cell_id, new Set());
  for (const e of edges) {
    adj.get(e.source)?.add(e.target);
    adj.get(e.target)?.add(e.source);
  }

  // ── Convergence state ──────────────────────────────────────────────────
  let quietFrames = 0;
  let stopped = false;

  // ── Ticker callback ────────────────────────────────────────────────────
  function tick(_ticker: Ticker) {
    if (stopped) return;

    // -- Forces --
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      let fx = 0;
      let fy = 0;

      // 1. Spring toward ELK target
      const dx = p.tx - p.cx;
      const dy = p.ty - p.cy;
      fx += K_SPRING * dx;
      fy += K_SPRING * dy;

      // 2. Pairwise repulsion
      for (let j = 0; j < particles.length; j++) {
        if (i === j) continue;
        const q = particles[j];
        const rx = p.cx - q.cx;
        const ry = p.cy - q.cy;
        const dist = Math.sqrt(rx * rx + ry * ry) + 0.001;
        if (dist < REPULSION_R) {
          const mag = REPULSION_K / (dist * dist);
          fx += (rx / dist) * mag;
          fy += (ry / dist) * mag;
        }
      }

      // 3. Topology edge springs
      const neighbours = adj.get(p.desc.cell_id);
      if (neighbours) {
        for (const nid of neighbours) {
          const q = byId.get(nid);
          if (!q) continue;
          const ex = q.cx - p.cx;
          const ey = q.cy - p.cy;
          fx += K_EDGE * ex;
          fy += K_EDGE * ey;
        }
      }

      // 4. Integrate
      p.vx = (p.vx + fx) * DAMPING;
      p.vy = (p.vy + fy) * DAMPING;
    }

    // -- Positions --
    let maxV = 0;
    for (const p of particles) {
      p.cx += p.vx;
      p.cy += p.vy;
      p.gfx.position.set(p.cx, p.cy);
      p.label.position.set(p.cx, p.cy);
      const v = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
      if (v > maxV) maxV = v;
    }

    // -- Redraw edges --
    edgeGfx.clear();
    for (const e of edges) {
      const src = byId.get(e.source);
      const tgt = byId.get(e.target);
      if (!src || !tgt) continue;
      const mx = (src.cx + tgt.cx) / 2;
      edgeGfx.setStrokeStyle({ width: 1, color: 0x388BFD, alpha: 0.5 });
      edgeGfx.moveTo(src.cx, src.cy);
      edgeGfx.bezierCurveTo(mx, src.cy, mx, tgt.cy, tgt.cx, tgt.cy);
      edgeGfx.stroke();
    }

    // -- Convergence --
    if (maxV < CONVERGE_EPS) {
      quietFrames++;
      if (quietFrames >= CONVERGE_FRAMES) {
        stopped = true;
        app.ticker.remove(tick);
        console.info('[pixi-cell-motion] converged — ticker stopped');
        // Snap all cells exactly onto target
        for (const p of particles) {
          p.gfx.position.set(p.tx, p.ty);
          p.label.position.set(p.tx, p.ty);
        }
      }
    } else {
      quietFrames = 0;
    }
  }

  app.ticker.add(tick);

  // Return manual stop handle
  return () => {
    if (!stopped) {
      stopped = true;
      app.ticker.remove(tick);
    }
  };
}
