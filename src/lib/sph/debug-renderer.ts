// debug-renderer.ts
// Collision debug overlay for physics engine visualization
// M569: force field arrows + easing animations (lygia import)
// M1287: species interaction matrix debug overlay — attract/repel lines
//
// Lygia GLSL references ported to TypeScript:
//   upstream/lygia/draw/arrows.glsl     — arrowsTileCenterCoord, arrows()
//   upstream/lygia/draw/circle.glsl     — circle SDF fill
//   upstream/lygia/animation/easing.glsl — cubicOut, backOut, elasticOut

// ---------------------------------------------------------------------------
// M1287: Species interaction matrix (loaded from channels/physics/species_interaction_matrix.json)
// ---------------------------------------------------------------------------

interface SpeciesInteractionMatrix {
  interaction_radius: number;
  matrix: Record<string, Record<string, number>>;
  description?: Record<string, string>;
}

// Loaded lazily; null until first use or explicit preload.
let _speciesMatrix: SpeciesInteractionMatrix | null = null;

/**
 * Load the species interaction matrix from the JSON channel.
 * Safe to call multiple times — loads only once.
 */
async function loadSpeciesMatrix(): Promise<SpeciesInteractionMatrix> {
  if (_speciesMatrix) return _speciesMatrix;
  try {
    // Dynamic import so the JSON is available in both Node (tests) and browser builds.
    const mod = await import('../../../channels/physics/species_interaction_matrix.json');
    _speciesMatrix = (mod.default ?? mod) as SpeciesInteractionMatrix;
  } catch {
    // Fallback: empty matrix with default radius so the overlay degrades silently.
    _speciesMatrix = { interaction_radius: 300, matrix: {} };
  }
  return _speciesMatrix;
}

/** Pre-load the matrix eagerly on module init (fire-and-forget). */
loadSpeciesMatrix();

// ---------------------------------------------------------------------------
// M1287: Interaction debug overlay state
// ---------------------------------------------------------------------------

/** Whether the interaction-matrix debug overlay is currently enabled. */
let _interactionDebugEnabled = false;

/**
 * toggleInteractionDebug — toggle the species interaction matrix overlay.
 *
 * When enabled, renderDebugOverlay() draws a line between every pair of cells
 * that are within interaction_radius (300 px) of each other:
 *   • green line for attraction (G > 0), lineWidth = G * 2
 *   • red   line for repulsion  (G < 0), lineWidth = |G| * 2
 *
 * Returns the new enabled state.
 */
export function toggleInteractionDebug(): boolean {
  _interactionDebugEnabled = !_interactionDebugEnabled;
  return _interactionDebugEnabled;
}

/** Read the current enabled state without toggling. */
export function isInteractionDebugEnabled(): boolean {
  return _interactionDebugEnabled;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------









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
  /** Force-field vector grid (lygia arrows style) */
  showForceField?: boolean;
  /** Animate stats-panel slide-in/out (easing) */
  animatePanels?: boolean;
  /**
   * M1287: Draw species interaction lines between nearby cells.
   * Green = attraction (G>0), red = repulsion (G<0).
   * Controlled via toggleInteractionDebug() or by setting this flag.
   */
  showInteractionMatrix?: boolean;
}

/**
 * M1287: A positioned, species-tagged cell for the interaction debug overlay.
 * Pass an array of these to renderDebugOverlay() via PhysicsWorld.debugCells.
 */
export interface DebugCell {
  /** Unique cell identifier */
  id: string;
  /** Species key matching species_interaction_matrix.json (e.g. "cil-eye") */
  species: string;
  /** World-space centre X in canvas pixels */
  cx: number;
  /** World-space centre Y in canvas pixels */
  cy: number;
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

/** A single cell in a 2-D force-field grid */
export interface ForceFieldSample {
  position: Vec2;
  force: Vec2;    // world-space force vector
}

export interface PhysicsWorld {
  bodies: Map<string, { aabb: AABB; position: Vec2 }>;
  emitters?: ParticleEmitter[];
  /** Optional flat particle list for velocity / density overlays */
  particles?: DebugParticle[];
  /** Optional force-field samples for the arrow grid overlay */
  forceField?: ForceFieldSample[];
  /**
   * M1287: Optional positioned + species-tagged cells for the interaction
   * matrix debug overlay. When provided and showInteractionMatrix is true,
   * attraction/repulsion lines are drawn between pairs within interaction_radius.
   */
  debugCells?: DebugCell[];
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
  forceArrow:       'rgba(255, 200,  50, 0.85)',
  forceArrowWeak:   'rgba(100, 160, 255, 0.60)',
};

const CONTACT_DOT_RADIUS  = 4;
const NORMAL_ARROW_LENGTH = 24;
const ARROW_HEAD_SIZE     = 6;
const BVH_DEPTH_ALPHA_DECAY = 0.72;

// ---------------------------------------------------------------------------
// Lygia easing functions (ported from upstream/lygia/animation/easing/)
//
//   cubicOut   : upstream/lygia/animation/easing/cubicOut.glsl
//                  float f = t - 1.0; return f*f*f + 1.0;
//   backOut    : upstream/lygia/animation/easing/backOut.glsl
//                  return 1 - backIn(1-t);  backIn(t) = t^3 - t*sin(t*PI)
//   elasticOut : upstream/lygia/animation/easing/elasticOut.glsl
//                  sin(-13*(t+1)*PI/2) * 2^(-10t) + 1
// ---------------------------------------------------------------------------

/** cubicOut — smooth deceleration (upstream/lygia/animation/easing/cubicOut.glsl) */
function cubicOut(t: number): number {
  const f = t - 1.0;
  return f * f * f + 1.0;
}

/** backIn — slight overshoot going in (upstream/lygia/animation/easing/backIn.glsl) */
function backIn(t: number): number {
  return t * t * t - t * Math.sin(t * Math.PI);
}

/** backOut — slight overshoot on arrival (upstream/lygia/animation/easing/backOut.glsl) */
function backOut(t: number): number {
  return 1.0 - backIn(1.0 - t);
}

/** elasticOut — spring overshoot on arrival (upstream/lygia/animation/easing/elasticOut.glsl) */
function elasticOut(t: number): number {
  return (
    Math.sin(-13.0 * (t + 1.0) * (Math.PI / 2)) *
    Math.pow(2.0, -10.0 * t) +
    1.0
  );
}

// ---------------------------------------------------------------------------
// Panel animation state
// ---------------------------------------------------------------------------

interface PanelAnim {
  /** 0 = fully hidden, 1 = fully visible */
  progress: number;
  /** direction: +1 sliding in, -1 sliding out */
  direction: 1 | -1;
  lastTimestamp: number;
  /** duration in ms for one full transition */
  durationMs: number;
}

const _panelAnims = new Map<string, PanelAnim>();

function getPanelAnim(id: string, durationMs = 320): PanelAnim {
  if (!_panelAnims.has(id)) {
    _panelAnims.set(id, {
      progress: 0,
      direction: 1,
      lastTimestamp: performance.now(),
      durationMs,
    });
  }
  return _panelAnims.get(id)!;
}

/**
 * Advance the panel animation and return the eased [0..1] value.
 * Uses backOut for slide-in and cubicOut for slide-out.
 */
function tickPanel(
  id: string,
  visible: boolean,
  now: number,
  durationMs = 320
): number {
  const anim = getPanelAnim(id, durationMs);
  const dt   = (now - anim.lastTimestamp) / 1000;
  anim.lastTimestamp = now;

  anim.direction = visible ? 1 : -1;
  anim.progress = Math.max(
    0,
    Math.min(1, anim.progress + anim.direction * (dt / (anim.durationMs / 1000)))
  );

  // backOut for slide-in (spring feel), cubicOut for slide-out (smooth)
  return visible ? backOut(anim.progress) : cubicOut(anim.progress);
}

// ---------------------------------------------------------------------------
// Heatmap colormap  (cool → warm: deep-blue → cyan → green → yellow → red)
// ---------------------------------------------------------------------------

function heatmapColor(t: number, alpha: number): string {
  const stops: [number, number, number][] = [
    [0.09, 0.22, 0.68],
    [0.09, 0.70, 0.85],
    [0.18, 0.82, 0.24],
    [1.00, 0.87, 0.00],
    [1.00, 0.40, 0.00],
    [0.90, 0.05, 0.05],
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
// Arrow primitive (contact-normal / penetration use)
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
// Lygia-style force-field arrows
//
// Ported from upstream/lygia/draw/arrows.glsl
// (Morgan McGuire / Matthias Reitinger — MIT License)
//
// The GLSL version computes an SDF mask in a tiled grid.  Here we receive
// an explicit list of ForceFieldSample values (one per grid cell) and draw
// each arrow using Canvas 2-D primitives that reproduce the same visual
// style: shaft + V-head, magnitude clamped to tile radius.
//
// Constants match the GLSL #defines:
//   ARROWS_TILE_SIZE       = 32.0
//   ARROWS_HEAD_ANGLE      = 0.5   (radians)
//   ARROWS_HEAD_LENGTH     = TILE / 5
//   ARROWS_SHAFT_THICKNESS = 2.0
// ---------------------------------------------------------------------------

const ARROWS_TILE_SIZE    = 32.0;
const ARROWS_HEAD_ANGLE   = 0.5;          // radians
const ARROWS_HEAD_LENGTH  = ARROWS_TILE_SIZE / 5.0;
const ARROWS_SHAFT_THICKNESS = 2.0;

/**
 * arrowsTileCenterCoord — ported from arrows.glsl
 * Returns the pixel-space centre of the tile that contains `pos`.
 */
function arrowsTileCenterCoord(pos: Vec2): Vec2 {
  return {
    x: (Math.floor(pos.x / ARROWS_TILE_SIZE) + 0.5) * ARROWS_TILE_SIZE,
    y: (Math.floor(pos.y / ARROWS_TILE_SIZE) + 0.5) * ARROWS_TILE_SIZE,
  };
}

/**
 * Draw a single lygia-style force-field arrow at `center` pointing in the
 * direction of `force`.  Magnitude is clamped to [5, TILE/2] as in the GLSL.
 *
 * Visual style: solid filled shaft + filled V-head (ARROWS_STYLE_LINE_TRIANGLE
 * equivalent adapted to Canvas 2D).
 */
function drawForceArrow(
  ctx: CanvasRenderingContext2D,
  center: Vec2,
  force: Vec2,
  color: string
): void {
  const mag_v = Math.hypot(force.x, force.y);
  if (mag_v < 1e-6) return;

  // Clamp magnitude — mirrors GLSL: clamp(mag_v, 5, TILE/2)
  const clampedMag = Math.min(Math.max(mag_v, 5.0), ARROWS_TILE_SIZE / 2.0);
  const dirX = force.x / mag_v;
  const dirY = force.y / mag_v;

  // Arrow tip in tile-local coords (from centre)
  const tipX = center.x + dirX * clampedMag;
  const tipY = center.y + dirY * clampedMag;

  // Tail
  const tailX = center.x - dirX * clampedMag;
  const tailY = center.y - dirY * clampedMag;

  // Perpendicular unit vector
  const perpX = -dirY;
  const perpY =  dirX;

  const halfShaft = ARROWS_SHAFT_THICKNESS / 2.0;

  // Head dimensions matching ARROWS_HEAD_ANGLE / ARROWS_HEAD_LENGTH
  const headBaseOffset = clampedMag - ARROWS_HEAD_LENGTH;
  const headHalfWidth  = Math.tan(ARROWS_HEAD_ANGLE / 2.0) * ARROWS_HEAD_LENGTH;

  // --- shaft (rectangle) ---
  const shaftBaseX = center.x + dirX * headBaseOffset;
  const shaftBaseY = center.y + dirY * headBaseOffset;

  ctx.beginPath();
  ctx.moveTo(tailX + perpX * halfShaft, tailY + perpY * halfShaft);
  ctx.lineTo(shaftBaseX + perpX * halfShaft, shaftBaseY + perpY * halfShaft);
  ctx.lineTo(shaftBaseX - perpX * halfShaft, shaftBaseY - perpY * halfShaft);
  ctx.lineTo(tailX - perpX * halfShaft, tailY - perpY * halfShaft);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();

  // --- head (filled triangle) ---
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(
    shaftBaseX + perpX * headHalfWidth,
    shaftBaseY + perpY * headHalfWidth
  );
  ctx.lineTo(
    shaftBaseX - perpX * headHalfWidth,
    shaftBaseY - perpY * headHalfWidth
  );
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

/**
 * Draw the full force-field grid.
 * For each ForceFieldSample the arrow is snapped to the tile centre that
 * contains the sample's position (arrowsTileCenterCoord), matching the GLSL
 * tile-centring logic.
 */
function drawForceField(
  ctx: CanvasRenderingContext2D,
  samples: ForceFieldSample[]
): void {
  if (!samples.length) return;

  // Compute max magnitude for normalised colour mapping
  let maxMag = 1e-6;
  for (const s of samples) {
    const m = Math.hypot(s.force.x, s.force.y);
    if (m > maxMag) maxMag = m;
  }

  ctx.save();
  ctx.globalCompositeOperation = 'source-over';

  for (const s of samples) {
    const center = arrowsTileCenterCoord(s.position);
    const mag    = Math.hypot(s.force.x, s.force.y);
    const t      = Math.min(mag / maxMag, 1.0);

    // Colour: weak → cool blue, strong → warm yellow, interpolated
    const r = (t * 255)  | 0;
    const g = (t * 200)  | 0;
    const b = ((1 - t) * 255) | 0;
    const color = `rgba(${r},${g},${b},${0.55 + t * 0.35})`;

    drawForceArrow(ctx, center, s.force, color);
  }

  ctx.restore();
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
// BVH
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
  const grd = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, 12);
  grd.addColorStop(0,   'rgba(255, 40,  40,  0.80)');
  grd.addColorStop(0.4, 'rgba(255, 80,  40,  0.35)');
  grd.addColorStop(1,   'rgba(255,  0,   0,  0.00)');
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, 12, 0, Math.PI * 2);
  ctx.fillStyle = grd;
  ctx.fill();

  ctx.beginPath();
  ctx.arc(pos.x, pos.y, CONTACT_DOT_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.contact;
  ctx.fill();

  ctx.beginPath();
  ctx.arc(pos.x - 1.2, pos.y - 1.2, 1.3, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255,255,255,0.75)';
  ctx.fill();
}

// ---------------------------------------------------------------------------
// Velocity arrows — each arrow follows its particle (lygia arrows style)
//
// The arrow shaft + filled triangle-head mirrors the ARROWS_STYLE_LINE_TRIANGLE
// branch in arrows.glsl.  The tip position is p.position + dir * clampedLen,
// identical to "v = dir_v * mag_v" (the tip) in the GLSL.
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

    const t   = Math.min(speed / maxSpeed, 1.0);

    // Speed → colour: cyan (slow) to magenta (fast)
    const r = (t * 255)       | 0;
    const g = ((1 - t) * 200) | 0;
    const color = `rgb(${r},${g},255)`;

    // Clamp length in [5, TILE/2] — mirrors the GLSL clamp
    const rawLen     = speed * 5;
    const clampedLen = Math.min(Math.max(rawLen, 5.0), ARROWS_TILE_SIZE / 2.0);

    const nx = p.velocity.x / speed;
    const ny = p.velocity.y / speed;

    // Tip = particle position + dir * clamped length
    const tipX = p.position.x + nx * clampedLen;
    const tipY = p.position.y + ny * clampedLen;

    // Perpendicular
    const perpX = -ny;
    const perpY =  nx;

    const headLen  = clampedLen * 0.28;
    const halfHead = Math.tan(ARROWS_HEAD_ANGLE / 2.0) * headLen;
    const halfShaft = ARROWS_SHAFT_THICKNESS * 0.5;

    // Shaft base (where head meets shaft)
    const sbX = tipX - nx * headLen;
    const sbY = tipY - ny * headLen;

    ctx.fillStyle = color;

    // shaft
    ctx.beginPath();
    ctx.moveTo(p.position.x + perpX * halfShaft, p.position.y + perpY * halfShaft);
    ctx.lineTo(sbX           + perpX * halfShaft, sbY           + perpY * halfShaft);
    ctx.lineTo(sbX           - perpX * halfShaft, sbY           - perpY * halfShaft);
    ctx.lineTo(p.position.x  - perpX * halfShaft, p.position.y  - perpY * halfShaft);
    ctx.closePath();
    ctx.fill();

    // head (triangle)
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(sbX + perpX * halfHead, sbY + perpY * halfHead);
    ctx.lineTo(sbX - perpX * halfHead, sbY - perpY * halfHead);
    ctx.closePath();
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

    ctx.beginPath();
    ctx.arc(p.position.x, p.position.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = heatmapColor(t, 0.95);
    ctx.fill();
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Stats panel  — animated slide-in from left via backOut / cubicOut easing
//
// Panel uses backOut (spring overshoot) when appearing and cubicOut (smooth
// deceleration) when disappearing, both ported from lygia/animation/easing/.
// ---------------------------------------------------------------------------

function drawStats(
  ctx: CanvasRenderingContext2D,
  world: PhysicsWorld,
  manifolds: CollisionManifold[],
  easedT: number          // [0..1] from tickPanel()
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

  const padding    = 8;
  const lineHeight = 16;
  const panelW     = 188;
  const panelH     = lines.length * lineHeight + padding * 2;
  const panelX     = 10;
  const panelY     = 10;

  // Slide in/out: translate X from -(panelW + panelX) → panelX
  const slideRange = panelW + panelX + 10;
  const tx         = (easedT - 1.0) * slideRange;   // < 0 when hidden, 0 when fully in

  ctx.save();
  ctx.translate(tx, 0);
  ctx.globalAlpha = Math.max(0, easedT);

  ctx.fillStyle = COLORS.statsBackground;
  ctx.beginPath();
  ctx.roundRect(panelX, panelY, panelW, panelH, 4);
  ctx.fill();

  ctx.font      = '11px monospace';
  ctx.fillStyle = COLORS.statsText;
  lines.forEach((line, i) => {
    ctx.fillText(line, panelX + padding, panelY + padding + 11 + i * lineHeight);
  });

  ctx.restore();
}

// ---------------------------------------------------------------------------
// Force-field legend panel  — animated slide-in from right (elasticOut)
// ---------------------------------------------------------------------------

function drawForceFieldLegend(
  ctx: CanvasRenderingContext2D,
  canvasWidth: number,
  easedT: number
): void {
  const padding  = 8;
  const panelW   = 148;
  const panelH   = 52;
  const panelX   = canvasWidth - panelW - 10;
  const panelY   = 10;

  // Slide from right
  const slideRange = panelW + 20;
  const tx = (1.0 - easedT) * slideRange;

  ctx.save();
  ctx.translate(tx, 0);
  ctx.globalAlpha = Math.max(0, easedT);

  ctx.fillStyle = 'rgba(0,0,0,0.60)';
  ctx.beginPath();
  ctx.roundRect(panelX, panelY, panelW, panelH, 4);
  ctx.fill();

  // Colour gradient bar (weak → strong)
  const barX = panelX + padding;
  const barY = panelY + 28;
  const barW = panelW - padding * 2;
  const barH = 10;
  const grad = ctx.createLinearGradient(barX, 0, barX + barW, 0);
  grad.addColorStop(0,   'rgba(0,0,255,0.80)');
  grad.addColorStop(0.5, 'rgba(128,128,128,0.80)');
  grad.addColorStop(1,   'rgba(255,200,0,0.90)');
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.roundRect(barX, barY, barW, barH, 3);
  ctx.fill();

  ctx.font      = '10px monospace';
  ctx.fillStyle = 'rgba(220,255,220,0.90)';
  ctx.fillText('Force Field', panelX + padding, panelY + padding + 8);
  ctx.fillText('weak', barX, barY + barH + 10);
  ctx.fillText('strong', barX + barW - 36, barY + barH + 10);

  ctx.restore();
}

// ---------------------------------------------------------------------------
// M1287: Species interaction matrix overlay
//
// For each ordered pair (A, B) of cells within interaction_radius we look up
// G = matrix[A.species][B.species].  To avoid drawing the same line twice we
// only draw when A.id < B.id (lexicographic) and use the average of the two
// directional G values as the displayed strength.
//
//   G > 0 → attraction → green  line, lineWidth = G  * 2
//   G < 0 → repulsion  → red    line, lineWidth = |G| * 2
// ---------------------------------------------------------------------------

function drawInteractionLines(
  ctx: CanvasRenderingContext2D,
  cells: DebugCell[],
  matrix: SpeciesInteractionMatrix,
): void {
  if (!cells.length) return;

  const radiusSq = matrix.interaction_radius * matrix.interaction_radius;

  ctx.save();
  ctx.globalCompositeOperation = 'source-over';

  for (let i = 0; i < cells.length; i++) {
    const a = cells[i];
    for (let j = i + 1; j < cells.length; j++) {
      const b = cells[j];

      // Distance check — only draw pairs within interaction_radius
      const dx = b.cx - a.cx;
      const dy = b.cy - a.cy;
      if (dx * dx + dy * dy > radiusSq) continue;

      // Look up G for both directions; average for display
      const gAB = matrix.matrix[a.species]?.[b.species] ?? 0;
      const gBA = matrix.matrix[b.species]?.[a.species] ?? 0;
      const g = (gAB + gBA) / 2;

      if (Math.abs(g) < 1e-6) continue; // neutral — skip

      const lineWidth = Math.abs(g) * 2;
      // Fade line with distance: alpha 0.8 at touching → 0.2 at radius edge
      const dist = Math.sqrt(dx * dx + dy * dy);
      const alpha = 0.2 + 0.6 * (1 - dist / matrix.interaction_radius);

      if (g > 0) {
        // Attraction — green
        ctx.strokeStyle = `rgba(0, 220, 60, ${alpha.toFixed(3)})`;
      } else {
        // Repulsion — red
        ctx.strokeStyle = `rgba(220, 40, 40, ${alpha.toFixed(3)})`;
      }

      ctx.lineWidth = Math.max(0.5, lineWidth);
      ctx.beginPath();
      ctx.moveTo(a.cx, a.cy);
      ctx.lineTo(b.cx, b.cy);
      ctx.stroke();
    }
  }

  ctx.restore();
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
  options: DebugRenderOptions,
  /** Current timestamp (ms) — required for panel easing; defaults to performance.now() */
  nowMs?: number
): void {
  const now    = nowMs ?? performance.now();
  const animate = options.animatePanels !== false;

  ctx.save();

  // ── M1287: Species interaction matrix overlay (lowest layer — under BVH) ─
  // Enabled when: toggleInteractionDebug() is active OR options.showInteractionMatrix.
  if ((_interactionDebugEnabled || options.showInteractionMatrix) &&
      world.debugCells?.length &&
      _speciesMatrix) {
    drawInteractionLines(ctx, world.debugCells, _speciesMatrix);
  }

  // ── Density heatmap (lowest layer) ──────────────────────────────────────
  if (options.showDensityHeatmap && world.particles?.length) {
    drawDensityHeatmap(ctx, world.particles);
  }

  // ── Force-field vector grid (lygia arrows) ────────────────────────────
  if (options.showForceField && world.forceField?.length) {
    drawForceField(ctx, world.forceField);
  }

  // ── BVH layer ─────────────────────────────────────────────────────────
  if (options.showBVH) {
    for (const root of bvhNodes) {
      const maxDepth = getBVHMaxDepth(root);
      drawBVHNode(ctx, root, maxDepth);
    }
  }

  // ── AABBs ─────────────────────────────────────────────────────────────
  if (options.showAABBs) {
    for (const aabb of debugAABBs) drawAABB(ctx, aabb);
    for (const [, body] of world.bodies) drawAABB(ctx, body.aabb);
  }

  // ── Emitters ──────────────────────────────────────────────────────────
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

  // ── Velocity arrows (per-particle, lygia arrows geometry) ─────────────
  if (options.showVelocityArrows && world.particles?.length) {
    drawVelocityArrows(ctx, world.particles);
  }

  // ── Contacts, normals, penetration ────────────────────────────────────
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
        const penVec: Vec2 = { x: -contact.normal.x, y: -contact.normal.y };
        drawArrow(ctx, contact.position, penVec, contact.penetrationDepth * 10, ARROW_HEAD_SIZE);
      }
    }
  }

  // ── Stats panel  (backOut slide-in / cubicOut slide-out) ──────────────
  if (options.showStats || animate) {
    const statsT = animate
      ? tickPanel('stats', !!options.showStats, now, 320)
      : (options.showStats ? 1 : 0);

    if (statsT > 0.01) {
      drawStats(ctx, world, manifolds, statsT);
    }
  }

  // ── Force-field legend panel (elasticOut slide-in from right) ─────────
  if ((options.showForceField || animate) && world.forceField?.length) {
    const canvasW = (ctx.canvas as HTMLCanvasElement).width ?? 800;

    const legendT = animate
      ? (() => {
          const raw = tickPanel('forceFieldLegend', !!options.showForceField, now, 400);
          // Override with elasticOut for slide-in
          const anim = getPanelAnim('forceFieldLegend');
          return anim.direction === 1
            ? elasticOut(anim.progress)
            : cubicOut(anim.progress);
        })()
      : (options.showForceField ? 1 : 0);

    if (legendT > 0.01) {
      drawForceFieldLegend(ctx, canvasW, legendT);
    }
  }

  ctx.restore();
}

// ---------------------------------------------------------------------------
// Re-export lygia easing utilities so callers can drive their own UI
// ---------------------------------------------------------------------------
export { cubicOut, backOut, elasticOut };
