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

export interface PhysicsWorld {
  bodies: Map<string, { aabb: AABB; position: Vec2 }>;
  emitters?: ParticleEmitter[];
  stats?: {
    bodyCount: number;
    activeContacts: number;
    broadphaseChecks: number;
    narrowphaseChecks: number;
    stepTimeMs: number;
  };
}

const COLORS = {
  aabb: 'rgba(0, 255, 255, 0.3)',
  aabbStroke: 'rgba(0, 255, 255, 0.8)',
  contact: 'rgba(255, 0, 0, 0.9)',
  contactNormal: 'rgba(255, 255, 0, 0.9)',
  penetration: 'rgba(255, 128, 0, 0.85)',
  bvh: 'rgba(0, 255, 0, 0.15)',
  bvhStroke: 'rgba(0, 255, 0, 0.6)',
  emitter: 'rgba(180, 80, 255, 0.6)',
  emitterFill: 'rgba(180, 80, 255, 0.15)',
  statsBackground: 'rgba(0, 0, 0, 0.65)',
  statsText: 'rgba(220, 255, 220, 1.0)',
};

const CONTACT_DOT_RADIUS = 4;
const NORMAL_ARROW_LENGTH = 24;
const ARROW_HEAD_SIZE = 6;
const BVH_DEPTH_ALPHA_DECAY = 0.72;

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

function drawAABB(ctx: CanvasRenderingContext2D, aabb: AABB): void {
  const w = aabb.maxX - aabb.minX;
  const h = aabb.maxY - aabb.minY;
  ctx.fillStyle = COLORS.aabb;
  ctx.fillRect(aabb.minX, aabb.minY, w, h);
  ctx.strokeStyle = COLORS.aabbStroke;
  ctx.lineWidth = 1;
  ctx.strokeRect(aabb.minX, aabb.minY, w, h);
}

function drawBVHNode(
  ctx: CanvasRenderingContext2D,
  node: BVHNode,
  maxDepth: number
): void {
  const alpha = Math.pow(BVH_DEPTH_ALPHA_DECAY, node.depth) * 0.85;
  const fillAlpha = 0.15 * alpha;
  const strokeAlpha = 0.6 * alpha;

  ctx.fillStyle = `rgba(0, 255, 0, ${fillAlpha})`;
  ctx.strokeStyle = `rgba(0, 255, 0, ${strokeAlpha})`;
  ctx.lineWidth = node.isLeaf ? 1.5 : 0.8;

  const w = node.aabb.maxX - node.aabb.minX;
  const h = node.aabb.maxY - node.aabb.minY;
  ctx.fillRect(node.aabb.minX, node.aabb.minY, w, h);
  ctx.strokeRect(node.aabb.minX, node.aabb.minY, w, h);

  if (!node.isLeaf) {
    if (node.left) drawBVHNode(ctx, node.left, maxDepth);
    if (node.right) drawBVHNode(ctx, node.right, maxDepth);
  }
}

function getBVHMaxDepth(node: BVHNode): number {
  if (node.isLeaf) return node.depth;
  let max = node.depth;
  if (node.left) max = Math.max(max, getBVHMaxDepth(node.left));
  if (node.right) max = Math.max(max, getBVHMaxDepth(node.right));
  return max;
}

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
  ];

  const padding = 8;
  const lineHeight = 16;
  const panelW = 180;
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

export function renderDebugOverlay(
  ctx: CanvasRenderingContext2D,
  world: PhysicsWorld,
  manifolds: CollisionManifold[],
  debugAABBs: AABB[],
  bvhNodes: BVHNode[],
  options: DebugRenderOptions
): void {
  ctx.save();

  // BVH layer (drawn first, lowest z-order)
  if (options.showBVH) {
    for (const root of bvhNodes) {
      const maxDepth = getBVHMaxDepth(root);
      drawBVHNode(ctx, root, maxDepth);
    }
  }

  // AABBs
  if (options.showAABBs) {
    for (const aabb of debugAABBs) {
      drawAABB(ctx, aabb);
    }
    for (const [, body] of world.bodies) {
      drawAABB(ctx, body.aabb);
    }
  }

  // Emitters
  if (options.showEmitters && world.emitters) {
    for (const emitter of world.emitters) {
      ctx.beginPath();
      ctx.arc(emitter.position.x, emitter.position.y, emitter.radius, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.emitterFill;
      ctx.fill();
      ctx.strokeStyle = COLORS.emitter;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // cross-hair center
      ctx.strokeStyle = COLORS.emitter;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(emitter.position.x - 5, emitter.position.y);
      ctx.lineTo(emitter.position.x + 5, emitter.position.y);
      ctx.moveTo(emitter.position.x, emitter.position.y - 5);
      ctx.lineTo(emitter.position.x, emitter.position.y + 5);
      ctx.stroke();
    }
  }

  // Contacts, normals, penetration
  for (const manifold of manifolds) {
    for (const contact of manifold.contacts) {
      if (options.showContacts) {
        ctx.beginPath();
        ctx.arc(contact.position.x, contact.position.y, CONTACT_DOT_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = COLORS.contact;
        ctx.fill();
      }

      if (options.showContactNormals) {
        ctx.strokeStyle = COLORS.contactNormal;
        ctx.lineWidth = 1.5;
        drawArrow(ctx, contact.position, contact.normal, NORMAL_ARROW_LENGTH, ARROW_HEAD_SIZE);
      }

      if (options.showPenetration && contact.penetrationDepth > 0) {
        ctx.strokeStyle = COLORS.penetration;
        ctx.lineWidth = 2;
        const penVec: Vec2 = {
          x: -contact.normal.x,
          y: -contact.normal.y,
        };
        drawArrow(ctx, contact.position, penVec, contact.penetrationDepth * 10, ARROW_HEAD_SIZE);
      }
    }
  }

  // Stats overlay (drawn last, topmost)
  if (options.showStats) {
    drawStats(ctx, world, manifolds);
  }

  ctx.restore();
}
