// world-boundary.ts

export interface WorldConfig {
  width: number;
  height: number;
  wallLayers?: number;       // default 3
  particleSpacing: number;
  wallRestitution?: number;  // default 0.3
  wallFriction?: number;     // default 0.1
}

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
  };
}

// ---------------------------------------------------------------------------
// Wall particles
// ---------------------------------------------------------------------------

export function createWallParticles(cfg: WorldConfig): BoundaryParticle[] {
  const {
    width,
    height,
    particleSpacing: s,
    wallLayers = 3,
  } = cfg;

  const particles: BoundaryParticle[] = [];

  for (let layer = 0; layer < wallLayers; layer++) {
    const offset = layer * s;

    // Bottom wall  (ny = +1, inward = up)
    for (let x = -offset; x <= width + offset; x += s) {
      particles.push({ x, y: -offset, nx: 0, ny: 1 });
    }

    // Top wall  (ny = -1, inward = down)
    for (let x = -offset; x <= width + offset; x += s) {
      particles.push({ x, y: height + offset, nx: 0, ny: -1 });
    }

    // Left wall  (nx = +1, inward = right)
    // skip corners already placed by top/bottom
    for (let y = -offset + s; y <= height + offset - s; y += s) {
      particles.push({ x: -offset, y, nx: 1, ny: 0 });
    }

    // Right wall  (nx = -1, inward = left)
    for (let y = -offset + s; y <= height + offset - s; y += s) {
      particles.push({ x: width + offset, y, nx: -1, ny: 0 });
    }
  }

  return particles;
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

// ---------------------------------------------------------------------------
// Domain clamping
// ---------------------------------------------------------------------------

/**
 * Clamp a position to the interior of the world domain.
 * Returns a new [x, y] pair guaranteed to lie within [margin, w-margin] x [margin, h-margin].
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
