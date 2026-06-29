/**
 * procedural-cell-geometries.ts — Procedural 3D meshes for 5 cell species
 *
 * Each function returns {positions, normals, uvs, indices} ready for
 * CellMeshRenderer._uploadGeometry(). Unit scale [-0.5, 0.5], normals out.
 *
 * These geometries are the 3D embodiment of the original 2D SVG generation
 * algorithms in channels/cell_component.py (initial commit). They are NOT
 * generic 3D solids — each shape *is* the algorithm's semantic signature:
 *
 *   cil-eye         (multi-head attention) → radial attention rays emanating
 *                     from a central pupil/focal point. Rays taper and shorten
 *                     with an intensity gradient (the attention "looking").
 *   cil-bolt        (FFN / activation)     → the ReLU zigzag profile, extruded:
 *                     flat on the left half, linear rising on the right half.
 *   cil-vector      (embedding)            → a bundle of direction+magnitude
 *                     arrows fanned across a small angular spread.
 *   cil-plus        (residual / add-norm)  → a plus cross with four corner
 *                     struts converging inward to the merge point.
 *   cil-arrow-right (forward dataflow)     → an extruded triangular arrowhead
 *                     pointing in the +X forward direction.
 */

interface ProceduralMesh {
  positions: Float32Array;
  normals:   Float32Array;
  uvs:       Float32Array;
  indices:   Uint16Array;
}

// ─── Utilities ──────────────────────────────────────────────────────────────

function pushVert(
  pos: number[], norm: number[], uv: number[],
  px: number, py: number, pz: number,
  nx: number, ny: number, nz: number,
  u: number, v: number,
): number {
  const idx = pos.length / 3;
  pos.push(px, py, pz);
  norm.push(nx, ny, nz);
  uv.push(u, v);
  return idx;
}

function normalise(x: number, y: number, z: number): [number, number, number] {
  const len = Math.sqrt(x * x + y * y + z * z) || 1;
  return [x / len, y / len, z / len];
}

function buildArrays(pos: number[], norm: number[], uv: number[], idx: number[]): ProceduralMesh {
  return {
    positions: new Float32Array(pos),
    normals:   new Float32Array(norm),
    uvs:       new Float32Array(uv),
    indices:   new Uint16Array(idx),
  };
}

/**
 * Append a tapered prism ("ray"/"strut") from `a` to `b` with square cross
 * section that shrinks from radius rA at the base to rB at the tip. Used to
 * build the attention rays, vector arrow shafts, and plus struts as real 3D
 * geometry. Cross-section is oriented in the plane perpendicular to (b-a).
 */
function pushTaperedPrism(
  pos: number[], norm: number[], uv: number[], idx: number[],
  a: [number, number, number], b: [number, number, number],
  rA: number, rB: number, sides = 6,
): void {
  const dir = normalise(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  // Pick an "up" not parallel to dir, then build an orthonormal frame.
  let up: [number, number, number] = Math.abs(dir[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const side = normalise(
    dir[1] * up[2] - dir[2] * up[1],
    dir[2] * up[0] - dir[0] * up[2],
    dir[0] * up[1] - dir[1] * up[0],
  );
  const upN = normalise(
    side[1] * dir[2] - side[2] * dir[1],
    side[2] * dir[0] - side[0] * dir[2],
    side[0] * dir[1] - side[1] * dir[0],
  );

  const ringStart = pos.length / 3;
  for (let ring = 0; ring < 2; ring++) {
    const base = ring === 0 ? a : b;
    const r = ring === 0 ? rA : rB;
    for (let s = 0; s <= sides; s++) {
      const ang = (s / sides) * Math.PI * 2;
      const ca = Math.cos(ang), sa = Math.sin(ang);
      const ox = (side[0] * ca + upN[0] * sa);
      const oy = (side[1] * ca + upN[1] * sa);
      const oz = (side[2] * ca + upN[2] * sa);
      const [nx, ny, nz] = normalise(ox, oy, oz);
      pushVert(pos, norm, uv,
        base[0] + ox * r, base[1] + oy * r, base[2] + oz * r,
        nx, ny, nz, s / sides, ring,
      );
    }
  }
  const stride = sides + 1;
  for (let s = 0; s < sides; s++) {
    const c = ringStart + s;
    const n = ringStart + stride + s;
    idx.push(c, n, c + 1, c + 1, n, n + 1);
  }
  // Tip cap (pointy when rB→0, but always closed)
  const tipCenter = pushVert(pos, norm, uv, b[0], b[1], b[2], dir[0], dir[1], dir[2], 0.5, 1);
  for (let s = 0; s < sides; s++) {
    idx.push(ringStart + stride + s, tipCenter, ringStart + stride + s + 1);
  }
  // Base cap
  const baseCenter = pushVert(pos, norm, uv, a[0], a[1], a[2], -dir[0], -dir[1], -dir[2], 0.5, 0);
  for (let s = 0; s < sides; s++) {
    idx.push(baseCenter, ringStart + s + 1, ringStart + s);
  }
}

/** Append a UV sphere centred at c with radius r. Normals point outward. */
function pushSphere(
  pos: number[], norm: number[], uv: number[], idx: number[],
  c: [number, number, number], r: number, rings = 10, segs = 14,
): void {
  const base = pos.length / 3;
  for (let ring = 0; ring <= rings; ring++) {
    const theta = (ring / rings) * Math.PI;
    const st = Math.sin(theta), ct = Math.cos(theta);
    for (let seg = 0; seg <= segs; seg++) {
      const phi = (seg / segs) * Math.PI * 2;
      const nx = st * Math.cos(phi), ny = ct, nz = st * Math.sin(phi);
      pushVert(pos, norm, uv,
        c[0] + nx * r, c[1] + ny * r, c[2] + nz * r,
        nx, ny, nz, seg / segs, ring / rings,
      );
    }
  }
  const stride = segs + 1;
  for (let ring = 0; ring < rings; ring++) {
    for (let seg = 0; seg < segs; seg++) {
      const cur = base + ring * stride + seg;
      const nxt = cur + stride;
      idx.push(cur, nxt, cur + 1, cur + 1, nxt, nxt + 1);
    }
  }
}

// ─── cil-eye: Radial attention rays + central pupil ─────────────────────────
// multi-head attention "observing". Mirrors generate_svg_cil_eye:
//   num_rays rays from an inner radius (r*0.3) outward, intensity gradient
//   (rays get shorter/lighter toward higher index), + central focal point.
// In 3D the rays radiate in the XY plane (the SVG plane) as tapered spikes,
// and the focal point is a pupil sphere with a small bright inner core.

export function createEyeSphere(): ProceduralMesh {
  const pos: number[] = [];
  const norm: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];

  const numRays = 12;                 // max(4, len(label)//2) → a representative value
  const rInner = 0.12;                // SVG: r_outer * 0.3 start radius
  const rOuterMax = 0.48;             // outer extent
  const rayBaseR = 0.018;

  for (let i = 0; i < numRays; i++) {
    const angle = (2 * Math.PI * i) / numRays;
    // Intensity gradient from the original: 0.3 + 0.7*(1 - i/num). Higher
    // intensity → longer, thicker ray (the "stronger attention" direction).
    const intensity = 0.3 + 0.7 * (1 - i / numRays);
    const rOuter = rInner + (rOuterMax - rInner) * intensity;
    const ca = Math.cos(angle), sa = Math.sin(angle);
    const a: [number, number, number] = [ca * rInner, sa * rInner, 0];
    const b: [number, number, number] = [ca * rOuter, sa * rOuter, 0];
    pushTaperedPrism(pos, norm, uv, idx, a, b, rayBaseR * intensity, 0.001, 5);
  }

  // Central focal point: outer pupil sphere + bright inner core sphere
  // (SVG drew two concentric circles: r_outer*0.2 dark, r_outer*0.08 light).
  pushSphere(pos, norm, uv, idx, [0, 0, 0], 0.13, 12, 18);
  pushSphere(pos, norm, uv, idx, [0, 0, 0.02], 0.055, 8, 12);

  return buildArrays(pos, norm, uv, idx);
}

// ─── cil-bolt: Extruded ReLU zigzag profile ─────────────────────────────────
// FFN activation. Mirrors generate_svg_cil_bolt: a polyline that is FLAT on the
// left half, then rises linearly on the right half — the ReLU(x) = max(0, x)
// curve. We extrude that exact 2D profile along Z into a 3D ribbon so the
// silhouette in the XY plane *is* the ReLU shape.

export function createBoltCrystal(): ProceduralMesh {
  const pos: number[] = [];
  const norm: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];

  const segments = 6;                 // matches SVG segment count
  const halfDepth = 0.12;             // extrusion ±Z
  const thickness = 0.08;             // ribbon thickness (band around the curve)

  // ReLU profile sampled left→right across [-0.5, 0.5] in X.
  // Flat (y=base) for the left half, then rising linearly on the right half.
  // The top of the rise is clamped to stay inside the [-0.5, 0.5] unit box.
  const baseY = -0.30;
  const topY = 0.42;                  // peak of the ReLU at the far right
  const activeSegs = segments - Math.floor(segments / 2); // rising-half count
  const profile: Array<[number, number]> = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const x = -0.45 + t * 0.9;
    let y: number;
    if (i < segments / 2) {
      y = baseY;                      // flat region: ReLU(x<0) = 0
    } else {
      const k = i - Math.floor(segments / 2);
      y = baseY + (k / activeSegs) * (topY - baseY); // linear rise to topY
    }
    profile.push([x, y]);
  }

  // Build a thick ribbon: for each profile point we have an upper edge (y) and
  // a lower edge (y - thickness), front face (+Z) and back face (-Z).
  // Vertex layout per profile point i, at front(z=+d)/back(z=-d):
  //   topFront, botFront, topBack, botBack
  function profNormal(i: number): [number, number] {
    // 2D tangent → outward (upward-left) normal of the top edge
    const a = profile[Math.max(0, i - 1)];
    const b = profile[Math.min(segments, i + 1)];
    const tx = b[0] - a[0], ty = b[1] - a[1];
    const [nx, ny] = [-ty, tx];
    const len = Math.hypot(nx, ny) || 1;
    return [nx / len, ny / len];
  }

  const cols: number[][] = []; // [topFront, botFront, topBack, botBack] per i
  for (let i = 0; i <= segments; i++) {
    const [x, y] = profile[i];
    const [tnx, tny] = profNormal(i);
    const u = i / segments;
    const topFront = pushVert(pos, norm, uv, x, y, halfDepth, tnx, tny, 0.4, u, 0);
    const botFront = pushVert(pos, norm, uv, x, y - thickness, halfDepth, -tnx, -tny, 0.4, u, 1);
    const topBack  = pushVert(pos, norm, uv, x, y, -halfDepth, tnx, tny, -0.4, u, 0);
    const botBack  = pushVert(pos, norm, uv, x, y - thickness, -halfDepth, -tnx, -tny, -0.4, u, 1);
    cols.push([topFront, botFront, topBack, botBack]);
  }

  for (let i = 0; i < segments; i++) {
    const [tf0, bf0, tb0, bb0] = cols[i];
    const [tf1, bf1, tb1, bb1] = cols[i + 1];
    // Front face (+Z)
    idx.push(tf0, bf0, tf1, tf1, bf0, bf1);
    // Back face (-Z)
    idx.push(tb1, bb0, tb0, tb1, bb1, bb0);
    // Top edge
    idx.push(tb0, tf0, tf1, tb0, tf1, tb1);
    // Bottom edge
    idx.push(bf0, bb0, bb1, bf0, bb1, bf1);
  }
  // End caps
  {
    const [tf, bf, tb, bb] = cols[0];
    idx.push(tf, tb, bb, tf, bb, bf);
  }
  {
    const [tf, bf, tb, bb] = cols[segments];
    idx.push(tb, tf, bf, tb, bf, bb);
  }

  return buildArrays(pos, norm, uv, idx);
}

// ─── cil-vector: Bundle of embedding arrows (direction + magnitude) ──────────
// Embedding. Mirrors generate_svg_cil_vector: num_arrows arrows fanned across a
// small angular spread (±0.4 rad), with varying stroke weight (magnitude). Each
// arrow is a 3D shaft (tapered prism) + a cone head pointing in its direction.

export function createVectorCapsule(): ProceduralMesh {
  const pos: number[] = [];
  const norm: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];

  const numArrows = 5;
  const arrowLen = 0.42;
  const spread = 0.4;                 // ±0.4 rad as in the original

  for (let i = 0; i < numArrows; i++) {
    // angle = -0.4 + 0.8*i/(n-1) : fan from -spread to +spread
    const angle = -spread + (2 * spread * i) / (numArrows - 1);
    const ca = Math.cos(angle), sa = Math.sin(angle);
    // weight = 1 + (i%3)*0.5  → encode magnitude as shaft thickness
    const weight = 1 + (i % 3) * 0.5;
    const shaftR = 0.012 * weight;
    const headR = 0.05 * (0.8 + 0.2 * weight);

    const tail: [number, number, number] = [-ca * arrowLen * 0.5, -sa * arrowLen * 0.5, 0];
    const neck: [number, number, number] = [ ca * arrowLen * 0.25,  sa * arrowLen * 0.25, 0];
    const tip:  [number, number, number] = [ ca * arrowLen * 0.5,   sa * arrowLen * 0.5,  0];

    // Shaft (uniform), then a cone head (taper to point) at the front.
    pushTaperedPrism(pos, norm, uv, idx, tail, neck, shaftR, shaftR, 6);
    pushTaperedPrism(pos, norm, uv, idx, neck, tip, headR, 0.001, 8);
  }

  return buildArrays(pos, norm, uv, idx);
}

// ─── cil-plus: Merge cross with converging corner struts ─────────────────────
// Residual / add-norm. Mirrors generate_svg_cil_plus: a plus cross (horizontal
// + vertical arms) PLUS four faint dashed lines from the corners converging to
// the center — the merge of skip-connection + main path into one point.

export function createPlusCross(): ProceduralMesh {
  const pos: number[] = [];
  const norm: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];

  const armLen = 0.42;
  const armW = 0.13;
  const armD = 0.11;

  function addBox(
    cx: number, cy: number, cz: number,
    sx: number, sy: number, sz: number,
  ) {
    const hx = sx / 2, hy = sy / 2, hz = sz / 2;
    const faces: Array<{ corners: [number, number, number][]; n: [number, number, number] }> = [
      { corners: [[-hx,-hy, hz],[ hx,-hy, hz],[ hx, hy, hz],[-hx, hy, hz]], n: [0,0,1] },
      { corners: [[-hx,-hy,-hz],[-hx, hy,-hz],[ hx, hy,-hz],[ hx,-hy,-hz]], n: [0,0,-1] },
      { corners: [[-hx, hy,-hz],[-hx, hy, hz],[ hx, hy, hz],[ hx, hy,-hz]], n: [0,1,0] },
      { corners: [[-hx,-hy,-hz],[ hx,-hy,-hz],[ hx,-hy, hz],[-hx,-hy, hz]], n: [0,-1,0] },
      { corners: [[ hx,-hy,-hz],[ hx, hy,-hz],[ hx, hy, hz],[ hx,-hy, hz]], n: [1,0,0] },
      { corners: [[-hx,-hy,-hz],[-hx,-hy, hz],[-hx, hy, hz],[-hx, hy,-hz]], n: [-1,0,0] },
    ];
    for (const face of faces) {
      const fi = pos.length / 3;
      for (let v = 0; v < 4; v++) {
        const [px, py, pz] = face.corners[v];
        pushVert(pos, norm, uv,
          cx + px, cy + py, cz + pz,
          face.n[0], face.n[1], face.n[2],
          v === 1 || v === 2 ? 1 : 0,
          v === 2 || v === 3 ? 1 : 0,
        );
      }
      idx.push(fi, fi + 1, fi + 2, fi, fi + 2, fi + 3);
    }
  }

  // The plus cross: horizontal (X) arm + vertical (Y) arm in the SVG plane.
  addBox(0, 0, 0, armLen * 2, armW, armD);   // horizontal
  addBox(0, 0, 0, armW, armLen * 2, armD);   // vertical

  // Four corner struts converging inward to the center (the merge point).
  // These are the dashed converging lines: thin tapered prisms from each
  // corner toward (0,0,0), thinning as they approach the junction.
  const corner = armLen * 0.95;
  for (const [dx, dy] of [[-1,-1],[1,-1],[-1,1],[1,1]] as const) {
    const a: [number, number, number] = [dx * corner, dy * corner, 0.02];
    const b: [number, number, number] = [0, 0, 0.02];
    pushTaperedPrism(pos, norm, uv, idx, a, b, 0.02, 0.004, 4);
  }

  return buildArrays(pos, norm, uv, idx);
}

// ─── cil-arrow-right: Extruded forward arrowhead ────────────────────────────
// Forward dataflow. Mirrors generate_svg_cil_arrow_right: a triangle polygon
//   (cx-aw, cy-8) (cx+aw, cy) (cx-aw, cy+8)  → a right-pointing arrowhead.
// We extrude that exact triangle along Z into a 3D wedge pointing +X.

export function createArrow(): ProceduralMesh {
  const pos: number[] = [];
  const norm: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];

  const halfDepth = 0.14;
  // Triangle in XY (forward = +X), matching the SVG arrowhead proportions.
  const back = -0.34;   // cx - aw
  const front = 0.48;   // cx + aw (tip)
  const halfH = 0.34;   // ± vertical extent at the back

  // Front (+Z) and back (-Z) faces of the triangular prism.
  const tri: [number, number][] = [
    [back, -halfH],   // bottom-back
    [front, 0],       // tip (front)
    [back,  halfH],   // top-back
  ];

  // Two flat triangle faces (front/back), normals ±Z.
  const fFront: number[] = [];
  const fBack: number[] = [];
  for (const [x, y] of tri) {
    fFront.push(pushVert(pos, norm, uv, x, y, halfDepth, 0, 0, 1, (x - back) / (front - back), (y + halfH) / (2 * halfH)));
  }
  for (const [x, y] of tri) {
    fBack.push(pushVert(pos, norm, uv, x, y, -halfDepth, 0, 0, -1, (x - back) / (front - back), (y + halfH) / (2 * halfH)));
  }
  idx.push(fFront[0], fFront[1], fFront[2]);
  idx.push(fBack[2], fBack[1], fBack[0]);

  // Three side faces connecting the two triangles, each with its own normal.
  function sideFace(i0: number, i1: number) {
    const a = tri[i0], b = tri[i1];
    // edge direction in XY → outward normal is perpendicular (rotate -90°)
    const ex = b[0] - a[0], ey = b[1] - a[1];
    let [nx, ny] = [ey, -ex];
    const len = Math.hypot(nx, ny) || 1; nx /= len; ny /= len;
    const v0 = pushVert(pos, norm, uv, a[0], a[1],  halfDepth, nx, ny, 0, 0, 0);
    const v1 = pushVert(pos, norm, uv, b[0], b[1],  halfDepth, nx, ny, 0, 1, 0);
    const v2 = pushVert(pos, norm, uv, b[0], b[1], -halfDepth, nx, ny, 0, 1, 1);
    const v3 = pushVert(pos, norm, uv, a[0], a[1], -halfDepth, nx, ny, 0, 0, 1);
    idx.push(v0, v1, v2, v0, v2, v3);
  }
  sideFace(0, 1); // bottom edge → tip
  sideFace(1, 2); // tip → top edge
  sideFace(2, 0); // back edge

  return buildArrays(pos, norm, uv, idx);
}

// ─── Species registry ───────────────────────────────────────────────────────

export const SPECIES_GEOMETRY: Record<string, () => ProceduralMesh> = {
  'cil-eye':         createEyeSphere,
  'cil-bolt':        createBoltCrystal,
  'cil-vector':      createVectorCapsule,
  'cil-plus':        createPlusCross,
  'cil-arrow-right': createArrow,
};
