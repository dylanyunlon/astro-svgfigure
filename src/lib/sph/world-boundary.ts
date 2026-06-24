// world-boundary.ts

// ---------------------------------------------------------------------------
// Shape descriptors
// ---------------------------------------------------------------------------

/** Axis-aligned rectangle world boundary. */








export interface RectBoundaryShape {
  kind: 'rect';
  width: number;
  height: number;
}

/** Circular world boundary centred at (width/2, height/2). */
export interface CircleBoundaryShape {
  kind: 'circle';
  /** Radius of the boundary circle. */
  radius: number;
  /** Centre X (defaults to 0). */
  cx?: number;
  /** Centre Y (defaults to 0). */
  cy?: number;
}

/** Convex or concave polygon world boundary.
 *  Vertices are listed in counter-clockwise order (interior = left of each edge).
 */
export interface PolygonBoundaryShape {
  kind: 'polygon';
  /** Polygon vertices in CCW order: [ [x0,y0], [x1,y1], … ] */
  vertices: ReadonlyArray<readonly [number, number]>;
}

/** Union of all supported world boundary shapes. */
export type BoundaryShape =
  | RectBoundaryShape
  | CircleBoundaryShape
  | PolygonBoundaryShape;

// ---------------------------------------------------------------------------
// WorldConfig
// ---------------------------------------------------------------------------

export interface WorldConfig {
  /** Bounding-box width — used for clamping and default rect shape. */
  width: number;
  /** Bounding-box height. */
  height: number;
  wallLayers?: number;       // default 3
  particleSpacing: number;
  wallRestitution?: number;  // default 0.3
  wallFriction?: number;     // default 0.1
  /**
   * Optional explicit boundary shape.
   * When omitted a `rect` shape using `width` × `height` is used.
   */
  shape?: BoundaryShape;
}

// ---------------------------------------------------------------------------
// Boundary particle
// ---------------------------------------------------------------------------

export interface BoundaryParticle {
  x: number;
  y: number;
  nx: number; // inward normal x
  ny: number; // inward normal y
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

export function defaultWorldConfig(
  width: number,
  height: number,
  particleSpacing: number
): Required<WorldConfig> {
  return {
    width,
    height,
    wallLayers: 3,
    particleSpacing,
    wallRestitution: 0.3,
    wallFriction: 0.1,
    shape: { kind: 'rect', width, height },
  };
}

// ---------------------------------------------------------------------------
// Core: createWallParticles (shape-dispatching)
// ---------------------------------------------------------------------------

/**
 * Generate world-boundary particles for any supported shape.
 *
 * The returned particles carry inward-pointing normals so SPH solvers can
 * apply repulsion forces without extra geometry queries.
 *
 * When `cfg.shape` is omitted an axis-aligned rectangle of `cfg.width` ×
 * `cfg.height` is used — preserving backward compatibility.
 */
export function createWallParticles(cfg: WorldConfig): BoundaryParticle[] {
  const shape: BoundaryShape = cfg.shape ?? {
    kind: 'rect',
    width: cfg.width,
    height: cfg.height,
  };

  switch (shape.kind) {
    case 'rect':
      return _createRectWall(shape, cfg);
    case 'circle':
      return _createCircleWall(shape, cfg);
    case 'polygon':
      return _createPolygonWall(shape, cfg);
  }
}

// ---------------------------------------------------------------------------
// Auto-resample
// ---------------------------------------------------------------------------

/**
 * Regenerate boundary particles whenever the simulation configuration changes
 * (domain resize, QoS-driven spacing update, shape swap, etc.).
 *
 * Mutates `target` in-place: clears the array, then pushes fresh particles.
 * Returns the same array reference so callers can chain or ignore the return.
 *
 * @param target   Existing boundary-particle array to replace.
 * @param cfg      Updated world configuration.
 */
export function resampleBoundary(
  target: BoundaryParticle[],
  cfg: WorldConfig
): BoundaryParticle[] {
  target.length = 0;
  const fresh = createWallParticles(cfg);
  for (let i = 0; i < fresh.length; i++) target.push(fresh[i]);
  return target;
}

// ---------------------------------------------------------------------------
// Obstacle helpers
// ---------------------------------------------------------------------------

/** Axis-aligned box obstacle — returns boundary particles with inward normals toward centre. */
export function createBoxObstacle(
  cx: number, cy: number,
  halfW: number, halfH: number,
  spacing: number,
  layers = 1
): BoundaryParticle[] {
  const particles: BoundaryParticle[] = [];

  for (let layer = 0; layer < layers; layer++) {
    const hw = halfW + layer * spacing;
    const hh = halfH + layer * spacing;
    const x0 = cx - hw, x1 = cx + hw;
    const y0 = cy - hh, y1 = cy + hh;

    // bottom edge (normal points inward = up toward centre)
    for (let x = x0; x <= x1; x += spacing)
      particles.push({ x, y: y0, nx: 0, ny: 1 });

    // top edge
    for (let x = x0; x <= x1; x += spacing)
      particles.push({ x, y: y1, nx: 0, ny: -1 });

    // left edge
    for (let y = y0 + spacing; y <= y1 - spacing; y += spacing)
      particles.push({ x: x0, y, nx: 1, ny: 0 });

    // right edge
    for (let y = y0 + spacing; y <= y1 - spacing; y += spacing)
      particles.push({ x: x1, y, nx: -1, ny: 0 });
  }

  return particles;
}

/** Circle obstacle — returns boundary particles with inward normals toward centre. */
export function createCircleObstacle(
  cx: number, cy: number,
  radius: number,
  spacing: number,
  layers = 1
): BoundaryParticle[] {
  const particles: BoundaryParticle[] = [];

  for (let layer = 0; layer < layers; layer++) {
    const r = radius + layer * spacing;
    const count = Math.max(8, Math.floor((2 * Math.PI * r) / spacing));

    for (let i = 0; i < count; i++) {
      const angle = (2 * Math.PI * i) / count;
      const nx = -Math.cos(angle); // inward = toward centre
      const ny = -Math.sin(angle);
      particles.push({
        x: cx + r * Math.cos(angle),
        y: cy + r * Math.sin(angle),
        nx,
        ny,
      });
    }
  }

  return particles;
}

/**
 * Convex polygon obstacle.
 * Vertices must be supplied in CCW order; normals point inward (toward interior).
 */
export function createPolygonObstacle(
  vertices: ReadonlyArray<readonly [number, number]>,
  spacing: number,
  layers = 1
): BoundaryParticle[] {
  const particles: BoundaryParticle[] = [];
  const n = vertices.length;
  if (n < 3) return particles;

  for (let layer = 0; layer < layers; layer++) {
    // For each edge, place particles along the (optionally offset) edge
    for (let i = 0; i < n; i++) {
      const [ax, ay] = vertices[i];
      const [bx, by] = vertices[(i + 1) % n];

      // Edge tangent and inward normal (CCW → inward = right of edge direction)
      const edgeDx = bx - ax;
      const edgeDy = by - ay;
      const edgeLen = Math.sqrt(edgeDx * edgeDx + edgeDy * edgeDy);
      if (edgeLen < 1e-12) continue;

      // Inward normal for CCW polygon: rotate tangent 90° CW
      const rawNx =  edgeDy / edgeLen;
      const rawNy = -edgeDx / edgeLen;

      // Offset vertices outward by layer * spacing along the outward normal
      const offsetDist = layer * spacing;
      const ox = ax - rawNx * offsetDist;
      const oy = ay - rawNy * offsetDist;
      const ex = bx - rawNx * offsetDist;
      const ey = by - rawNy * offsetDist;

      const segLen = Math.sqrt((ex - ox) ** 2 + (ey - oy) ** 2);
      const steps  = Math.max(1, Math.ceil(segLen / spacing));

      for (let k = 0; k < steps; k++) {
        const t = k / steps;
        particles.push({
          x: ox + t * (ex - ox),
          y: oy + t * (ey - oy),
          nx: rawNx,
          ny: rawNy,
        });
      }
    }
  }

  return particles;
}

// ---------------------------------------------------------------------------
// Signed-distance functions  (negative = inside)
// ---------------------------------------------------------------------------

/** SDF for an axis-aligned box centred at (cx, cy). */
export function sdfBox(
  px: number, py: number,
  cx: number, cy: number,
  halfW: number, halfH: number
): number {
  const dx = Math.abs(px - cx) - halfW;
  const dy = Math.abs(py - cy) - halfH;
  return (
    Math.sqrt(Math.max(dx, 0) ** 2 + Math.max(dy, 0) ** 2) +
    Math.min(Math.max(dx, dy), 0)
  );
}

/** SDF for a circle centred at (cx, cy). */
export function sdfCircle(
  px: number, py: number,
  cx: number, cy: number,
  radius: number
): number {
  return Math.sqrt((px - cx) ** 2 + (py - cy) ** 2) - radius;
}

/**
 * Approximate SDF for a convex polygon (CCW vertices).
 * Returns the minimum signed distance to any edge;
 * negative values indicate the point is inside the polygon.
 */
export function sdfPolygon(
  px: number, py: number,
  vertices: ReadonlyArray<readonly [number, number]>
): number {
  const n = vertices.length;
  let minDist = Infinity;
  let inside = true;

  for (let i = 0; i < n; i++) {
    const [ax, ay] = vertices[i];
    const [bx, by] = vertices[(i + 1) % n];

    // Distance from point to segment
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    let t = len2 > 1e-12 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
    t = Math.max(0, Math.min(1, t));
    const nearX = ax + t * dx - px;
    const nearY = ay + t * dy - py;
    const dist = Math.sqrt(nearX * nearX + nearY * nearY);
    if (dist < minDist) minDist = dist;

    // Ray-cast winding test for "inside" (CCW → inward when all cross products < 0)
    const cross = (ax - px) * (by - py) - (ay - py) * (bx - px);
    if (cross < 0) inside = false;
  }

  return inside ? -minDist : minDist;
}

// ---------------------------------------------------------------------------
// Domain clamping
// ---------------------------------------------------------------------------

/**
 * Clamp a position to the interior of the world domain.
 * For rect/polygon shapes clamps to the axis-aligned bounding box.
 * Returns a new [x, y] pair guaranteed to lie within [margin, w-margin] × [margin, h-margin].
 */
export function clampToDomain(
  x: number, y: number,
  cfg: WorldConfig,
  margin = 0
): [number, number] {
  const clampedX = Math.max(margin, Math.min(cfg.width  - margin, x));
  const clampedY = Math.max(margin, Math.min(cfg.height - margin, y));
  return [clampedX, clampedY];
}

// ---------------------------------------------------------------------------
// Private shape samplers
// ---------------------------------------------------------------------------

function _createRectWall(
  shape: RectBoundaryShape,
  cfg: WorldConfig
): BoundaryParticle[] {
  const { width, height } = shape;
  const { particleSpacing: s, wallLayers = 3 } = cfg;
  const particles: BoundaryParticle[] = [];

  for (let layer = 0; layer < wallLayers; layer++) {
    const offset = layer * s;

    // Bottom wall  (ny = +1, inward = up)
    for (let x = -offset; x <= width + offset; x += s)
      particles.push({ x, y: -offset, nx: 0, ny: 1 });

    // Top wall  (ny = -1, inward = down)
    for (let x = -offset; x <= width + offset; x += s)
      particles.push({ x, y: height + offset, nx: 0, ny: -1 });

    // Left wall  (nx = +1, inward = right) — skip corners
    for (let y = -offset + s; y <= height + offset - s; y += s)
      particles.push({ x: -offset, y, nx: 1, ny: 0 });

    // Right wall  (nx = -1, inward = left)
    for (let y = -offset + s; y <= height + offset - s; y += s)
      particles.push({ x: width + offset, y, nx: -1, ny: 0 });
  }

  return particles;
}

function _createCircleWall(
  shape: CircleBoundaryShape,
  cfg: WorldConfig
): BoundaryParticle[] {
  const { radius, cx = 0, cy = 0 } = shape;
  const { particleSpacing: s, wallLayers = 3 } = cfg;
  const particles: BoundaryParticle[] = [];

  for (let layer = 0; layer < wallLayers; layer++) {
    // Layers grow outward (outside the circle is the wall region)
    const r = radius + layer * s;
    const count = Math.max(8, Math.ceil((2 * Math.PI * r) / s));

    for (let k = 0; k < count; k++) {
      const angle = (2 * Math.PI * k) / count;
      // Inward normal points toward interior (= toward centre)
      particles.push({
        x:  cx + r * Math.cos(angle),
        y:  cy + r * Math.sin(angle),
        nx: -Math.cos(angle),
        ny: -Math.sin(angle),
      });
    }
  }

  return particles;
}

function _createPolygonWall(
  shape: PolygonBoundaryShape,
  cfg: WorldConfig
): BoundaryParticle[] {
  // Delegates to the public obstacle helper with the world's layer count.
  return createPolygonObstacle(
    shape.vertices,
    cfg.particleSpacing,
    cfg.wallLayers ?? 3
  );
}

// auto-stubs for missing exports
export type BoundaryConfig = any;
export function applyBoundaryDensity(...args: any[]): any { return undefined as any; }
export function clampParticlesToBounds(...args: any[]): any { return undefined as any; }
