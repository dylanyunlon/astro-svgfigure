// sat-solver.ts
// SAT (Separating Axis Theorem) for 2D OBB vs OBB collisions





export interface OBB {
  cx: number;    // center x
  cy: number;    // center y
  hw: number;    // half-width
  hh: number;    // half-height
  angle: number; // rotation in radians
}

export interface SATResult {
  intersecting: boolean;
  normal: { x: number; y: number };  // collision normal (unit vector)
  depth: number;                      // penetration depth
  axisIndex: number;                  // 0-3: which axis produced minimum overlap
}

/** Get the 4 corner vertices of an OBB */
function getVertices(obb: OBB): Array<{ x: number; y: number }> {
  const cos = Math.cos(obb.angle);
  const sin = Math.sin(obb.angle);

  // Local corners: (±hw, ±hh)
  const corners = [
    { x:  obb.hw, y:  obb.hh },
    { x: -obb.hw, y:  obb.hh },
    { x: -obb.hw, y: -obb.hh },
    { x:  obb.hw, y: -obb.hh },
  ];

  return corners.map(c => ({
    x: obb.cx + cos * c.x - sin * c.y,
    y: obb.cy + sin * c.x + cos * c.y,
  }));
}

/** Get the 2 unique edge normals (axes) of an OBB */
function getAxes(obb: OBB): Array<{ x: number; y: number }> {
  const cos = Math.cos(obb.angle);
  const sin = Math.sin(obb.angle);

  // Two perpendicular axes aligned with the box edges (already unit length)
  return [
    { x: cos, y: sin },   // local X axis (along width)
    { x: -sin, y: cos },  // local Y axis (along height)
  ];
}

/**
 * Project an OBB onto a given axis.
 * Returns [min, max] scalar projection values.
 */
export function projectOBB(
  obb: OBB,
  axis: { x: number; y: number }
): [number, number] {
  const vertices = getVertices(obb);
  let min = Infinity;
  let max = -Infinity;

  for (const v of vertices) {
    const proj = v.x * axis.x + v.y * axis.y;
    if (proj < min) min = proj;
    if (proj > max) max = proj;
  }

  return [min, max];
}

/**
 * Compute overlap of two 1D intervals [minA, maxA] and [minB, maxB].
 * Returns the signed overlap (negative means no overlap).
 */
function intervalOverlap(
  minA: number, maxA: number,
  minB: number, maxB: number
): number {
  // Overlap = min(maxA, maxB) - max(minA, minB)
  return Math.min(maxA, maxB) - Math.max(minA, minB);
}

/**
 * SAT test for two 2D OBBs.
 *
 * Tests exactly 4 separating axes:
 *   - Axis 0: OBB A local X (edge normal along width)
 *   - Axis 1: OBB A local Y (edge normal along height)
 *   - Axis 2: OBB B local X (edge normal along width)
 *   - Axis 3: OBB B local Y (edge normal along height)
 *
 * Returns the axis of minimum overlap as the collision normal.
 * If any axis has no overlap, the OBBs are not intersecting.
 */
export function satTest(a: OBB, b: OBB): SATResult {
  const axesA = getAxes(a); // 2 axes from box A
  const axesB = getAxes(b); // 2 axes from box B
  const allAxes = [...axesA, ...axesB]; // 4 axes total

  let minOverlap = Infinity;
  let minAxisIndex = 0;
  let minAxis = allAxes[0];

  for (let i = 0; i < allAxes.length; i++) {
    const axis = allAxes[i];

    const [minA, maxA] = projectOBB(a, axis);
    const [minB, maxB] = projectOBB(b, axis);

    const overlap = intervalOverlap(minA, maxA, minB, maxB);

    // If no overlap on this axis, we found a separating axis — no intersection
    if (overlap <= 0) {
      return {
        intersecting: false,
        normal: { x: axis.x, y: axis.y },
        depth: 0,
        axisIndex: i,
      };
    }

    // Track the axis with the smallest (minimum) overlap
    if (overlap < minOverlap) {
      minOverlap = overlap;
      minAxisIndex = i;
      minAxis = axis;
    }
  }

  // All axes overlapped — OBBs are intersecting
  // Ensure the normal points from A toward B
  const dx = b.cx - a.cx;
  const dy = b.cy - a.cy;
  const dot = dx * minAxis.x + dy * minAxis.y;

  // Flip normal so it points from A to B
  const normalX = dot < 0 ? -minAxis.x : minAxis.x;
  const normalY = dot < 0 ? -minAxis.y : minAxis.y;

  return {
    intersecting: true,
    normal: { x: normalX, y: normalY },
    depth: minOverlap,
    axisIndex: minAxisIndex,
  };
}

// ---------------------------------------------------------------------------
// Quick sanity test (runs when executed directly via ts-node / bun)
// ---------------------------------------------------------------------------
if (typeof require !== "undefined" && require.main === module) {
  // Two overlapping axis-aligned boxes
  const boxA: OBB = { cx: 0,   cy: 0,  hw: 2, hh: 2, angle: 0 };
  const boxB: OBB = { cx: 3,   cy: 0,  hw: 2, hh: 2, angle: 0 };
  const boxC: OBB = { cx: 10,  cy: 0,  hw: 2, hh: 2, angle: 0 };
  const boxD: OBB = { cx: 1.5, cy: 0,  hw: 2, hh: 2, angle: Math.PI / 6 };

  const r1 = satTest(boxA, boxB);
  console.log("A vs B (overlap ~1):", r1);
  // intersecting: true, depth ≈ 1

  const r2 = satTest(boxA, boxC);
  console.log("A vs C (separated):", r2);
  // intersecting: false

  const r3 = satTest(boxA, boxD);
  console.log("A vs D (rotated overlap):", r3);
  // intersecting: true
}
