// debug-renderer.ts
// Collision debug overlay for physics engine visualization

export interface DebugRenderOptions {
  showAABBs: boolean;
  showContacts: boolean;
  showContactNormals: boolean;
  showPenetration: boolean;
  showBVH: boolean;
  showEmitters: boolean;
  showStats: boolean;
  /** Velocity arrows with speed-mapped colour (cyan → magenta) */
  showVelocityArrows: boolean;
  /** Density heatmap splat per particle */
  showDensityHeatmap: boolean;
}

export interface Vec2 {
  x: number;
  y: number;
}

export interface AABB {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface ContactPoint {
  position: Vec2;
  normal: Vec2;
  penetrationDepth: number;
  bodyAId?: string;
  bodyBId?: string;
}

export interface CollisionManifold {
  contacts: ContactPoint[];
  normal: Vec2;
  penetrationDepth: number;
  bodyAId: string;
  bodyBId: string;
}

export interface BVHNode {
  aabb: AABB;
  isLeaf: boolean;
  depth: number;
  left?: BVHNode;
  right?: BVHNode;
  objectId?: string;
}

export interface ParticleEmitter {
  position: Vec2;
  radius: number;
  id: string;
}

export interface DebugParticle {
  position: Vec2;
  velocity: Vec2;
  density: number;
  /** Smoothing radius; used for heatmap splat size */
  smoothingRadius?: number;
}

export interface PhysicsWorld {
  bodies: Map<string, { aabb: AABB; position: Vec2 }>;
  emitters?: ParticleEmitter[];
  /** Optional flat particle list for velocity / density overlays */
  particles?: DebugParticle[];
  stats?: {
    bodyCount: number;
    activeContacts: number;
    broadphaseChecks: number;
    narrowphaseChecks: number;
    stepTimeMs: number;
  };
}

// ---------------------------------------------------------------------------
// Palette
// ---------------------------------------------------------------------------

const COLORS = {
  aabb:             'rgba(0, 255, 255, 0.3)',
  aabbStroke:       'rgba(0, 255, 255, 0.8)',
  contact:          'rgba(255, 0,   0,   0.9)',
  contactNormal:    'rgba(255, 255, 0,   0.9)',
  penetration:      'rgba(255, 128, 0,   0.85)',
  bvh:              'rgba(0, 255, 0,   0.15)',
  bvhStroke:        'rgba(0, 255, 0,   0.6)',
  emitter:          'rgba(180, 80, 255, 0.6)',
  emitterFill:      'rgba(180, 80, 255, 0.15)',
  statsBackground:  'rgba(0, 0,   0,   0.65)',
  statsText:        'rgba(220, 255, 220, 1.0)',
};

const CONTACT_DOT_RADIUS  = 4;
const NORMAL_ARROW_LENGTH = 24;
const ARROW_HEAD_SIZE     = 6;
const BVH_DEPTH_ALPHA_DECAY = 0.72;

// ---------------------------------------------------------------------------
// Heatmap colormap  (cool → warm: deep-blue → cyan → green → yellow → red)
// ---------------------------------------------------------------------------

function heatmapColor(t: number, alpha: number): string {
  const stops: [number, number, number][] = [
    [0.09, 0.22, 0.68],  // deep blue
    [0.09, 0.70, 0.85],  // cyan
    [0.18, 0.82, 0.24],  // green
    [1.00, 0.87, 0.00],  // yellow
    [1.00, 0.40, 0.00],  // orange
    [0.90, 0.05, 0.05],  // red
  ];
  const idx  = Math.min(t * (stops.length - 1), stops.length - 1 - 1e-9);
  const lo   = Math.floor(idx);
  const hi   = lo + 1;
  const frac = idx - lo;
  const r    = stops[lo][0] + (stops[hi][0] - stops[lo][0]) * frac;
  const g    = stops[lo][1] + (stops[hi][1] - stops[lo][1]) * frac;
  const b    = stops[lo][2] + (stops[hi][2] - stops[lo][2]) * frac;
  return `rgba(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0},${alpha})`;
}

// ---------------------------------------------------------------------------
// Arrow primitive
// ---------------------------------------------------------------------------

function drawArrow(
  ctx: CanvasRenderingContext2D,
  from: Vec2,
  dir: Vec2,
  length: number,
  headSize: number
): void {
  const toX = from.x + dir.x * length;
  const toY = from.y + dir.y * length;

  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(toX, toY);
  ctx.stroke();

  const angle = Math.atan2(dir.y, dir.x);
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(
    toX - headSize * Math.cos(angle - Math.PI / 6),
    toY - headSize * Math.sin(angle - Math.PI / 6)
  );
  ctx.moveTo(toX, toY);
  ctx.lineTo(
    toX - headSize * Math.cos(angle + Math.PI / 6),
    toY - headSize * Math.sin(angle + Math.PI / 6)
  );
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

function drawAABB(ctx: CanvasRenderingContext2D, aabb: AABB): void {
  const w = aabb.maxX - aabb.minX;
  const h = aabb.maxY - aabb.minY;
  ctx.fillStyle   = COLORS.aabb;
  ctx.fillRect(aabb.minX, aabb.minY, w, h);
  ctx.strokeStyle = COLORS.aabbStroke;
  ctx.lineWidth   = 1;
  ctx.strokeRect(aabb.minX, aabb.minY, w, h);
}

// ---------------------------------------------------------------------------
// BVH  — depth-coded colour, leaf vs internal distinction
// ---------------------------------------------------------------------------

function getBVHMaxDepth(node: BVHNode): number {
  if (node.isLeaf) return node.depth;
  let max = node.depth;
  if (node.left)  max = Math.max(max, getBVHMaxDepth(node.left));
  if (node.right) max = Math.max(max, getBVHMaxDepth(node.right));
  return max;
}

function drawBVHNode(
  ctx: CanvasRenderingContext2D,
  node: BVHNode,
  maxDepth: number
): void {
  const depthT      = maxDepth > 0 ? node.depth / maxDepth : 0;
  const alpha       = node.isLeaf
    ? 0.90
    : 0.45 * (1 - depthT * 0.55);
  const fillAlpha   = alpha * 0.12;
  const strokeAlpha = alpha;

  // Hue: root (green) → mid (cyan) → leaf (white-tinted cyan)
  const hue = node.isLeaf ? 180 : 120 + depthT * 40;
  const sat = node.isLeaf ? 90  : 100;
  const lit = node.isLeaf ? 75  : 65;

  ctx.strokeStyle = `hsla(${hue},${sat}%,${lit}%,${strokeAlpha})`;
  ctx.fillStyle   = `hsla(${hue},${sat}%,${lit}%,${fillAlpha})`;
  ctx.lineWidth   = node.isLeaf ? 1.5 : 0.75;

  const w  = node.aabb.maxX - node.aabb.minX;
  const h  = node.aabb.maxY - node.aabb.minY;
  ctx.fillRect  (node.aabb.minX, node.aabb.minY, w, h);
  ctx.strokeRect(node.aabb.minX, node.aabb.minY, w, h);

  // Label leaf with objectId
  if (node.isLeaf && node.objectId) {
    ctx.save();
    ctx.fillStyle = `hsla(${hue},${sat}%,${lit}%,0.65)`;
    ctx.font = '9px monospace';
    ctx.textBaseline = 'top';
    ctx.fillText(node.objectId, node.aabb.minX + 2, node.aabb.minY + 2);
    ctx.restore();
  }

  if (!node.isLeaf) {
    if (node.left)  drawBVHNode(ctx, node.left,  maxDepth);
    if (node.right) drawBVHNode(ctx, node.right, maxDepth);
  }
}

// ---------------------------------------------------------------------------
// Contact points — red glow dot + optional normal arrow
// ---------------------------------------------------------------------------

function drawContactDot(ctx: CanvasRenderingContext2D, pos: Vec2): void {
  // Glow ring
  const grd = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, 12);
  grd.addColorStop(0,   'rgba(255, 40,  40,  0.80)');
  grd.addColorStop(0.4, 'rgba(255, 80,  40,  0.35)');
  grd.addColorStop(1,   'rgba(255,  0,   0,  0.00)');
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, 12, 0, Math.PI * 2);
  ctx.fillStyle = grd;
  ctx.fill();

  // Solid dot
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, CONTACT_DOT_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.contact;
  ctx.fill();

  // Specular pip
  ctx.beginPath();
  ctx.arc(pos.x - 1.2, pos.y - 1.2, 1.3, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255,255,255,0.75)';
  ctx.fill();
}

// ---------------------------------------------------------------------------
// Velocity arrows  (speed-mapped colour: cyan → magenta)
// ---------------------------------------------------------------------------

function drawVelocityArrows(
  ctx: CanvasRenderingContext2D,
  particles: DebugParticle[]
): void {
  if (!particles.length) return;

  let maxSpeed = 1;
  for (const p of particles) {
    const s = Math.hypot(p.velocity.x, p.velocity.y);
    if (s > maxSpeed) maxSpeed = s;
  }

  ctx.save();
  for (const p of particles) {
    const speed = Math.hypot(p.velocity.x, p.velocity.y);
    if (speed < 0.01) continue;

    const t     = Math.min(speed / maxSpeed, 1);
    const scale = 5;
    const len   = Math.min(speed * scale, 40);
    const r     = (t * 255) | 0;
    const g     = ((1 - t) * 200) | 0;
    const color = `rgb(${r},${g},255)`;

    const nx    = p.velocity.x / speed;
    const ny    = p.velocity.y / speed;
    const tip: Vec2 = { x: p.position.x + nx * len, y: p.position.y + ny * len };
    const headSz = Math.max(4, len * 0.22);

    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.2;
    ctx.beginPath();
    ctx.moveTo(p.position.x, p.position.y);
    ctx.lineTo(tip.x, tip.y);
    ctx.stroke();

    const angle = Math.atan2(ny, nx);
    ctx.beginPath();
    ctx.moveTo(tip.x, tip.y);
    ctx.lineTo(
      tip.x - headSz * Math.cos(angle - Math.PI / 6),
      tip.y - headSz * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      tip.x - headSz * Math.cos(angle + Math.PI / 6),
      tip.y - headSz * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Density heatmap splat
// ---------------------------------------------------------------------------

function drawDensityHeatmap(
  ctx: CanvasRenderingContext2D,
  particles: DebugParticle[]
): void {
  if (!particles.length) return;

  let minD = Infinity, maxD = 0;
  for (const p of particles) {
    if (p.density < minD) minD = p.density;
    if (p.density > maxD) maxD = p.density;
  }
  if (maxD === minD) maxD = minD + 1;

  ctx.save();
  for (const p of particles) {
    const t  = (p.density - minD) / (maxD - minD);
    const hr = (p.smoothingRadius ?? 12) * 1.4;

    const grd = ctx.createRadialGradient(
      p.position.x, p.position.y, 0,
      p.position.x, p.position.y, hr
    );
    grd.addColorStop(0,   heatmapColor(t, 0.55));
    grd.addColorStop(0.6, heatmapColor(t, 0.20));
    grd.addColorStop(1,   heatmapColor(t, 0.00));

    ctx.beginPath();
    ctx.arc(p.position.x, p.position.y, hr, 0, Math.PI * 2);
    ctx.fillStyle = grd;
    ctx.fill();

    // Solid core dot coloured by density
    ctx.beginPath();
    ctx.arc(p.position.x, p.position.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = heatmapColor(t, 0.95);
    ctx.fill();
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Stats panel
// ---------------------------------------------------------------------------

function drawStats(
  ctx: CanvasRenderingContext2D,
  world: PhysicsWorld,
  manifolds: CollisionManifold[]
): void {
  const stats = world.stats;
  const lines: string[] = [
    '── Physics Debug ──',
    `Bodies      : ${stats?.bodyCount ?? world.bodies.size}`,
    `Manifolds   : ${manifolds.length}`,
    `Contacts    : ${stats?.activeContacts ?? manifolds.reduce((s, m) => s + m.contacts.length, 0)}`,
    `Broadphase  : ${stats?.broadphaseChecks ?? '—'}`,
    `Narrowphase : ${stats?.narrowphaseChecks ?? '—'}`,
    `Step (ms)   : ${stats?.stepTimeMs?.toFixed(2) ?? '—'}`,
    world.particles ? `Particles   : ${world.particles.length}` : null,
  ].filter(Boolean) as string[];

  const padding = 8;
  const lineHeight = 16;
  const panelW = 188;
  const panelH = lines.length * lineHeight + padding * 2;
  const x = 10;
  const y = 10;

  ctx.fillStyle = COLORS.statsBackground;
  ctx.beginPath();
  ctx.roundRect(x, y, panelW, panelH, 4);
  ctx.fill();

  ctx.font = '11px monospace';
  ctx.fillStyle = COLORS.statsText;
  lines.forEach((line, i) => {
    ctx.fillText(line, x + padding, y + padding + 11 + i * lineHeight);
  });
}

// ---------------------------------------------------------------------------
// Public entry-point
// ---------------------------------------------------------------------------

export function renderDebugOverlay(
  ctx: CanvasRenderingContext2D,
  world: PhysicsWorld,
  manifolds: CollisionManifold[],
  debugAABBs: AABB[],
  bvhNodes: BVHNode[],
  options: DebugRenderOptions
): void {
  ctx.save();

  // ── Density heatmap (lowest layer) ──────────────────────────────────────
  if (options.showDensityHeatmap && world.particles?.length) {
    drawDensityHeatmap(ctx, world.particles);
  }

  // ── BVH layer ────────────────────────────────────────────────────────────
  if (options.showBVH) {
    for (const root of bvhNodes) {
      const maxDepth = getBVHMaxDepth(root);
      drawBVHNode(ctx, root, maxDepth);
    }
  }

  // ── AABBs ────────────────────────────────────────────────────────────────
  if (options.showAABBs) {
    for (const aabb of debugAABBs) {
      drawAABB(ctx, aabb);
    }
    for (const [, body] of world.bodies) {
      drawAABB(ctx, body.aabb);
    }
  }

  // ── Emitters ─────────────────────────────────────────────────────────────
  if (options.showEmitters && world.emitters) {
    for (const emitter of world.emitters) {
      ctx.beginPath();
      ctx.arc(emitter.position.x, emitter.position.y, emitter.radius, 0, Math.PI * 2);
      ctx.fillStyle   = COLORS.emitterFill;
      ctx.fill();
      ctx.strokeStyle = COLORS.emitter;
      ctx.lineWidth   = 1.5;
      ctx.stroke();

      ctx.strokeStyle = COLORS.emitter;
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.moveTo(emitter.position.x - 5, emitter.position.y);
      ctx.lineTo(emitter.position.x + 5, emitter.position.y);
      ctx.moveTo(emitter.position.x,     emitter.position.y - 5);
      ctx.lineTo(emitter.position.x,     emitter.position.y + 5);
      ctx.stroke();
    }
  }

  // ── Velocity arrows ───────────────────────────────────────────────────────
  if (options.showVelocityArrows && world.particles?.length) {
    drawVelocityArrows(ctx, world.particles);
  }

  // ── Contacts, normals, penetration ───────────────────────────────────────
  for (const manifold of manifolds) {
    for (const contact of manifold.contacts) {
      if (options.showContacts) {
        drawContactDot(ctx, contact.position);
      }

      if (options.showContactNormals) {
        ctx.strokeStyle = COLORS.contactNormal;
        ctx.lineWidth   = 1.5;
        drawArrow(ctx, contact.position, contact.normal, NORMAL_ARROW_LENGTH, ARROW_HEAD_SIZE);
      }

      if (options.showPenetration && contact.penetrationDepth > 0) {
        ctx.strokeStyle = COLORS.penetration;
        ctx.lineWidth   = 2;
        const penVec: Vec2 = {
          x: -contact.normal.x,
          y: -contact.normal.y,
        };
        drawArrow(ctx, contact.position, penVec, contact.penetrationDepth * 10, ARROW_HEAD_SIZE);
      }
    }
  }

  // ── Stats overlay (topmost) ───────────────────────────────────────────────
  if (options.showStats) {
    drawStats(ctx, world, manifolds);
  }

  ctx.restore();
}
