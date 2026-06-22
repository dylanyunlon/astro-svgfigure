import { World, Particle, RigidBody } from './world';

export const SPECIES_COLORS: Record<number, string> = {
  0: '#3F51B5',
  1: '#FF6F00',
  2: '#2E7D32',
  3: '#C62828',
  4: '#455A64',
  5: '#7B1FA2',
  6: '#1565C0',
};

export interface RenderOptions {
  showTrails: boolean;
  showDensity: boolean;
  showVelocity: boolean;
  showGrid: boolean;
  showBoundaryParticles: boolean;
  showForces: boolean;
  showContacts: boolean;
  showContactNormals: boolean;
  /** Draw BVH node AABBs with depth-based colour coding */
  showBVH: boolean;
  /** Overlay density heatmap (cool → warm colormap) instead of species halo */
  showDensityHeatmap: boolean;
}

export const DEFAULT_OPTIONS: RenderOptions = {
  showTrails: false,
  showDensity: false,
  showVelocity: false,
  showGrid: true,
  showBoundaryParticles: false,
  showForces: false,
  showContacts: false,
  showContactNormals: false,
  showBVH: false,
  showDensityHeatmap: false,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

/**
 * Maps a normalised value t∈[0,1] to a "cool → warm" heatmap colour.
 * Palette: deep blue → cyan → green → yellow → orange → red
 */
function heatmapColor(t: number): string {
  // 5-stop palette (each stop covers 0.25 of the range)
  const stops = [
    [0.09, 0.22, 0.68],   // deep blue
    [0.09, 0.70, 0.85],   // cyan
    [0.18, 0.82, 0.24],   // green
    [1.00, 0.87, 0.00],   // yellow
    [1.00, 0.40, 0.00],   // orange
    [0.90, 0.05, 0.05],   // red
  ];
  const idx = Math.min(t * (stops.length - 1), stops.length - 1 - 1e-9);
  const lo  = Math.floor(idx);
  const hi  = lo + 1;
  const frac = idx - lo;
  const r = stops[lo][0] + (stops[hi][0] - stops[lo][0]) * frac;
  const g = stops[lo][1] + (stops[hi][1] - stops[lo][1]) * frac;
  const b = stops[lo][2] + (stops[hi][2] - stops[lo][2]) * frac;
  return `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;
}

// ---------------------------------------------------------------------------
// BVH overlay
// ---------------------------------------------------------------------------

export interface BVHNodeFlat {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  /** Depth in the tree (root = 0) */
  depth: number;
  /** True when this is a leaf holding an actual body */
  isLeaf: boolean;
}

/**
 * Draw every BVH node AABB.  Leaf nodes are drawn brighter and with a thicker
 * stroke; internal nodes fade with depth so the tree hierarchy is visually
 * obvious.
 */
function drawBVHBounds(
  ctx: CanvasRenderingContext2D,
  nodes: BVHNodeFlat[]
): void {
  if (!nodes.length) return;
  const maxDepth = nodes.reduce((m, n) => Math.max(m, n.depth), 0) || 1;

  ctx.save();
  for (const n of nodes) {
    const depthT = n.depth / maxDepth;          // 0 = root, 1 = deepest
    const alpha  = n.isLeaf ? 0.85 : 0.35 * (1 - depthT * 0.6);

    // Depth-coded hue: internal nodes are green, leaves shift toward cyan/white
    const h = n.isLeaf ? 180 : 120 + depthT * 40;  // green → cyan
    const fillAlpha  = alpha * 0.10;
    const strokeAlpha = alpha;

    ctx.strokeStyle = `hsla(${h},100%,65%,${strokeAlpha})`;
    ctx.fillStyle   = `hsla(${h},100%,65%,${fillAlpha})`;
    ctx.lineWidth   = n.isLeaf ? 1.5 : 0.75;

    const w = n.maxX - n.minX;
    const h_ = n.maxY - n.minY;
    ctx.fillRect(n.minX, n.minY, w, h_);
    ctx.strokeRect(n.minX, n.minY, w, h_);
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Contact points with red glow
// ---------------------------------------------------------------------------

export interface ContactPoint {
  x: number;
  y: number;
  nx: number;
  ny: number;
}

function drawContactPoints(
  ctx: CanvasRenderingContext2D,
  contacts: ContactPoint[],
  opts: RenderOptions
): void {
  if (!contacts.length) return;
  ctx.save();

  for (const cp of contacts) {
    if (opts.showContacts) {
      // Outer glow ring
      const grd = ctx.createRadialGradient(cp.x, cp.y, 0, cp.x, cp.y, 10);
      grd.addColorStop(0,   'rgba(255, 40,  40,  0.80)');
      grd.addColorStop(0.4, 'rgba(255, 80,  40,  0.40)');
      grd.addColorStop(1,   'rgba(255,  0,   0,  0.00)');
      ctx.beginPath();
      ctx.arc(cp.x, cp.y, 10, 0, Math.PI * 2);
      ctx.fillStyle = grd;
      ctx.fill();

      // Solid red dot
      ctx.beginPath();
      ctx.arc(cp.x, cp.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 30, 30, 0.97)';
      ctx.fill();

      // White highlight pip
      ctx.beginPath();
      ctx.arc(cp.x - 1.2, cp.y - 1.2, 1.2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.fill();
    }

    if (opts.showContactNormals) {
      const arrowLen = 20;
      const tx = cp.x + cp.nx * arrowLen;
      const ty = cp.y + cp.ny * arrowLen;
      const headLen = 6;
      const angle = Math.atan2(cp.ny, cp.nx);

      ctx.beginPath();
      ctx.moveTo(cp.x, cp.y);
      ctx.lineTo(tx, ty);
      ctx.strokeStyle = 'rgba(255, 220, 0, 0.95)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(
        tx + Math.cos(angle + Math.PI * 0.75) * headLen,
        ty + Math.sin(angle + Math.PI * 0.75) * headLen
      );
      ctx.moveTo(tx, ty);
      ctx.lineTo(
        tx + Math.cos(angle - Math.PI * 0.75) * headLen,
        ty + Math.sin(angle - Math.PI * 0.75) * headLen
      );
      ctx.strokeStyle = 'rgba(255, 220, 0, 0.95)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  ctx.restore();
}

// ---------------------------------------------------------------------------
// Velocity arrows (proper arrowhead, speed-mapped colour)
// ---------------------------------------------------------------------------

function drawVelocityArrow(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  vx: number,
  vy: number,
  maxSpeed: number
): void {
  const speed = Math.hypot(vx, vy);
  if (speed < 0.01) return;

  const scale = 5;
  const len   = Math.min(speed * scale, 40);
  const t     = Math.min(speed / maxSpeed, 1);

  // Colour: slow = cyan, fast = magenta
  const r = (t * 255) | 0;
  const g = ((1 - t) * 200) | 0;
  const b = 255;
  const color = `rgb(${r},${g},${b})`;

  const nx   = vx / speed;
  const ny   = vy / speed;
  const tx   = x + nx * len;
  const ty   = y + ny * len;
  const angle = Math.atan2(ny, nx);
  const headSize = Math.max(4, len * 0.22);

  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(tx, ty);
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.2;
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(tx, ty);
  ctx.lineTo(
    tx - headSize * Math.cos(angle - Math.PI / 6),
    ty - headSize * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    tx - headSize * Math.cos(angle + Math.PI / 6),
    ty - headSize * Math.sin(angle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

// ---------------------------------------------------------------------------
// Grid / boundary glow
// ---------------------------------------------------------------------------

function drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number, cellSize = 40): void {
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 0.5;
  for (let x = 0; x <= width; x += cellSize) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
  }
  for (let y = 0; y <= height; y += cellSize) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  ctx.restore();
}

function drawBoundaryGlow(ctx: CanvasRenderingContext2D, width: number, height: number): void {
  ctx.save();
  const glowSize = 32;
  const sides = [
    { x: 0, y: 0, w: width, h: glowSize, dir: 'top' },
    { x: 0, y: height - glowSize, w: width, h: glowSize, dir: 'bottom' },
    { x: 0, y: 0, w: glowSize, h: height, dir: 'left' },
    { x: width - glowSize, y: 0, w: glowSize, h: height, dir: 'right' },
  ];
  for (const s of sides) {
    let grad: CanvasGradient;
    if (s.dir === 'top')         grad = ctx.createLinearGradient(0, 0, 0, glowSize);
    else if (s.dir === 'bottom') grad = ctx.createLinearGradient(0, s.y, 0, s.y + glowSize);
    else if (s.dir === 'left')   grad = ctx.createLinearGradient(0, 0, glowSize, 0);
    else                         grad = ctx.createLinearGradient(s.x, 0, s.x + glowSize, 0);
    const inner = s.dir === 'bottom' || s.dir === 'right';
    grad.addColorStop(inner ? 1 : 0, 'rgba(100,160,255,0.18)');
    grad.addColorStop(inner ? 0 : 1, 'rgba(100,160,255,0.00)');
    ctx.fillStyle = grad;
    ctx.fillRect(s.x, s.y, s.w, s.h);
  }
  ctx.strokeStyle = 'rgba(100,160,255,0.35)';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(0.75, 0.75, width - 1.5, height - 1.5);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Trails
// ---------------------------------------------------------------------------

function drawTrails(ctx: CanvasRenderingContext2D, particles: Particle[]): void {
  ctx.save();
  for (const p of particles) {
    if (!p.trail || p.trail.length < 2) continue;
    const color = SPECIES_COLORS[p.species] ?? '#ffffff';
    ctx.beginPath();
    ctx.moveTo(p.trail[0].x, p.trail[0].y);
    for (let i = 1; i < p.trail.length; i++) {
      ctx.lineTo(p.trail[i].x, p.trail[i].y);
    }
    ctx.strokeStyle = hexToRgba(color, 0.25);
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Particles
// ---------------------------------------------------------------------------

function drawParticles(
  ctx: CanvasRenderingContext2D,
  particles: Particle[],
  opts: RenderOptions
): void {
  ctx.save();

  // Pre-compute max speed for velocity arrow normalisation
  let maxSpeed = 1;
  if (opts.showVelocity) {
    for (const p of particles) {
      const s = Math.hypot(p.vx ?? 0, p.vy ?? 0);
      if (s > maxSpeed) maxSpeed = s;
    }
  }

  // Density range for heatmap
  let minDensity = Infinity, maxDensity = 0;
  if (opts.showDensityHeatmap) {
    for (const p of particles) {
      if (p.density == null) continue;
      if (p.density < minDensity) minDensity = p.density;
      if (p.density > maxDensity) maxDensity = p.density;
    }
    if (maxDensity === minDensity) maxDensity = minDensity + 1;
  }

  for (const p of particles) {
    if (!opts.showBoundaryParticles && p.isBoundary) continue;

    const color  = SPECIES_COLORS[p.species] ?? '#ffffff';
    const speed  = Math.hypot(p.vx ?? 0, p.vy ?? 0);
    const radius = 3 + Math.min(speed * 0.4, 3);

    // ── Density heatmap overlay ──────────────────────────────────────────
    if (opts.showDensityHeatmap && p.density != null) {
      const t = (p.density - minDensity) / (maxDensity - minDensity);
      const hColor = heatmapColor(t);
      // Soft disk proportional to smoothing kernel influence
      const hr = radius * 3.2;
      const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, hr);
      grd.addColorStop(0, hColor.replace('rgb(', 'rgba(').replace(')', ',0.55)'));
      grd.addColorStop(1, hColor.replace('rgb(', 'rgba(').replace(')', ',0.00)'));
      ctx.beginPath();
      ctx.arc(p.x, p.y, hr, 0, Math.PI * 2);
      ctx.fillStyle = grd;
      ctx.fill();
    } else if (opts.showDensity && p.density != null) {
      // Legacy density halo (species-tinted)
      const d = Math.min(p.density / 20, 1);
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = hexToRgba(color, d * 0.15);
      ctx.fill();
    }

    // ── Glow halo ────────────────────────────────────────────────────────
    const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius * 2);
    grd.addColorStop(0, hexToRgba(color, 0.6));
    grd.addColorStop(1, hexToRgba(color, 0.0));
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius * 2, 0, Math.PI * 2);
    ctx.fillStyle = grd;
    ctx.fill();

    // ── Solid core ───────────────────────────────────────────────────────
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = opts.showDensityHeatmap && p.density != null
      ? heatmapColor(Math.min((p.density - minDensity) / (maxDensity - minDensity), 1))
      : color;
    ctx.fill();

    // ── Velocity arrow ───────────────────────────────────────────────────
    if (opts.showVelocity && (p.vx || p.vy)) {
      drawVelocityArrow(ctx, p.x, p.y, p.vx ?? 0, p.vy ?? 0, maxSpeed);
    }

    // ── Force arrow ──────────────────────────────────────────────────────
    if (opts.showForces && (p.fx || p.fy)) {
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x + (p.fx ?? 0) * 0.05, p.y + (p.fy ?? 0) * 0.05);
      ctx.strokeStyle = 'rgba(255,220,50,0.6)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  ctx.restore();
}

// ---------------------------------------------------------------------------
// Rigid bodies
// ---------------------------------------------------------------------------

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawRigidBodies(ctx: CanvasRenderingContext2D, bodies: RigidBody[]): void {
  ctx.save();
  for (const b of bodies) {
    const hw = b.width / 2, hh = b.height / 2;
    ctx.save();
    ctx.translate(b.x, b.y);
    ctx.rotate(b.angle ?? 0);
    roundRect(ctx, -hw, -hh, b.width, b.height, 6);
    ctx.fillStyle = 'rgba(255,255,255,0.07)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.50)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    if (b.pinned) {
      ctx.beginPath();
      ctx.arc(0, 0, 5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,220,50,0.9)';
      ctx.fill();
    }
    if (b.label) {
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(b.label, 0, 0);
    }
    ctx.restore();
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

function drawHUD(ctx: CanvasRenderingContext2D, world: World): void {
  ctx.save();
  const pad = 10, lineH = 16;
  const lines = [
    `particles: ${world.particles?.length ?? 0}`,
    `bodies:    ${world.rigidBodies?.length ?? 0}`,
    `tick:      ${world.tick ?? 0}`,
  ];
  const boxW = 130, boxH = lines.length * lineH + 12;
  roundRect(ctx, pad, pad, boxW, boxH, 5);
  ctx.fillStyle = 'rgba(0,0,0,0.45)';
  ctx.fill();
  ctx.fillStyle = 'rgba(180,200,255,0.85)';
  ctx.font = '11px monospace';
  ctx.textBaseline = 'top';
  lines.forEach((l, i) => ctx.fillText(l, pad + 8, pad + 6 + i * lineH));
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface WorldRenderExtras {
  /** Flat BVH node list for BVH bounds visualisation */
  bvhNodes?: BVHNodeFlat[];
  /** Contact points for the red-dot + normal-arrow overlay */
  contacts?: ContactPoint[];
}

export function renderWorld(
  ctx: CanvasRenderingContext2D,
  world: World,
  options: Partial<RenderOptions> = {},
  extras: WorldRenderExtras = {}
): void {
  const opts: RenderOptions = { ...DEFAULT_OPTIONS, ...options };
  const { width, height } = ctx.canvas;

  // 1. Dark background
  ctx.fillStyle = '#0d0f1a';
  ctx.fillRect(0, 0, width, height);

  // 2. Grid
  if (opts.showGrid) drawGrid(ctx, width, height);

  // 3. Boundary glow + border
  drawBoundaryGlow(ctx, width, height);

  // 4. BVH bounds (drawn before particles so they appear underneath)
  if (opts.showBVH && extras.bvhNodes?.length) {
    drawBVHBounds(ctx, extras.bvhNodes);
  }

  // 5. Trails
  if (opts.showTrails) drawTrails(ctx, world.particles ?? []);

  // 6. Particles: density heatmap / legacy halo → glow → core → arrows
  drawParticles(ctx, world.particles ?? [], opts);

  // 7. Rigid bodies: rounded rect + pin dot + label
  drawRigidBodies(ctx, world.rigidBodies ?? []);

  // 8. Contact points / normals
  if (opts.showContacts || opts.showContactNormals) {
    const contacts =
      extras.contacts ??
      (world as unknown as { contacts?: ContactPoint[] }).contacts ??
      [];
    drawContactPoints(ctx, contacts, opts);
  }

  // 9. HUD overlay
  drawHUD(ctx, world);
}
