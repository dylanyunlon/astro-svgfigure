/**
 * procedural-cell-geometries.ts — Procedural 3D meshes for 5 cell species
 *
 * Each function returns {positions, normals, uvs, indices} ready for
 * CellMeshRenderer._uploadGeometry(). Unit scale [-0.5, 0.5].
 *
 * Species visual design:
 *   cil-eye    (self-attention) → sphere with lens bumps — sensory perception
 *   cil-bolt   (FFN/activation) → faceted crystal with zigzag ridges — energy
 *   cil-vector (embedding)      → elongated capsule with groove lines — data stream
 *   cil-plus   (add-norm)       → cross/junction node — merging pathways
 *   cil-arrow-right (output)    → streamlined arrow — directional flow
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

// ─── cil-eye: Sphere with lens bumps ────────────────────────────────────────
// Self-attention = sensory perception. Sphere with protruding "eye" lenses.

export function createEyeSphere(rings = 16, segs = 24): ProceduralMesh {
  const pos: number[] = [];
  const norm: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];
  const R = 0.45;  // base radius

  // 4 "lens" positions on the sphere (tetrahedral arrangement)
  const lensAngles = [
    { theta: 0.8, phi: 0 },
    { theta: 0.8, phi: Math.PI * 2 / 3 },
    { theta: 0.8, phi: Math.PI * 4 / 3 },
    { theta: 2.2, phi: Math.PI / 3 },
  ];

  for (let ring = 0; ring <= rings; ring++) {
    const theta = (ring / rings) * Math.PI;
    const sinT = Math.sin(theta);
    const cosT = Math.cos(theta);

    for (let seg = 0; seg <= segs; seg++) {
      const phi = (seg / segs) * Math.PI * 2;
      const sinP = Math.sin(phi);
      const cosP = Math.cos(phi);

      let nx = sinT * cosP;
      let ny = cosT;
      let nz = sinT * sinP;

      // Check proximity to any lens position — bump outward
      let bump = 0;
      for (const lens of lensAngles) {
        const lx = Math.sin(lens.theta) * Math.cos(lens.phi);
        const ly = Math.cos(lens.theta);
        const lz = Math.sin(lens.theta) * Math.sin(lens.phi);
        const dot = nx * lx + ny * ly + nz * lz;
        if (dot > 0.85) {
          bump = Math.max(bump, (dot - 0.85) / 0.15 * 0.12);
        }
      }

      const r = R + bump;
      pushVert(pos, norm, uv,
        nx * r, ny * r, nz * r,
        nx, ny, nz,
        seg / segs, ring / rings,
      );

      if (ring < rings && seg < segs) {
        const cur = ring * (segs + 1) + seg;
        const nxt = cur + segs + 1;
        idx.push(cur, nxt, cur + 1);
        idx.push(cur + 1, nxt, nxt + 1);
      }
    }
  }

  return buildArrays(pos, norm, uv, idx);
}

// ─── cil-bolt: Faceted crystal with zigzag ridges ───────────────────────────
// FFN = rapid energy processing. Angular, faceted, with zigzag surface detail.

export function createBoltCrystal(facets = 8, heightSegs = 6): ProceduralMesh {
  const pos: number[] = [];
  const norm: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];

  const topY = 0.5;
  const botY = -0.5;

  for (let h = 0; h <= heightSegs; h++) {
    const t = h / heightSegs;
    const y = botY + t * (topY - botY);

    // Diamond profile: widest at middle, pointy at top/bottom
    const profile = 1.0 - Math.abs(t - 0.5) * 2.0;  // 0..1..0
    const radius = 0.15 + profile * 0.35;

    // Zigzag offset: alternating ring twist
    const twist = (h % 2 === 0) ? 0 : Math.PI / facets;

    for (let f = 0; f <= facets; f++) {
      const angle = (f / facets) * Math.PI * 2 + twist;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;

      // Faceted normal: perpendicular to the face (approximated by vertex normal)
      const [nx, ny, nz] = normalise(
        Math.cos(angle),
        (0.5 - t) * 0.5,  // slight upward at top, downward at bottom
        Math.sin(angle),
      );

      pushVert(pos, norm, uv, x, y, z, nx, ny, nz, f / facets, t);

      if (h < heightSegs && f < facets) {
        const cur = h * (facets + 1) + f;
        const nxt = cur + facets + 1;
        idx.push(cur, nxt, cur + 1);
        idx.push(cur + 1, nxt, nxt + 1);
      }
    }
  }

  // Top and bottom caps (pointy)
  const topIdx = pushVert(pos, norm, uv, 0, topY + 0.05, 0, 0, 1, 0, 0.5, 0);
  const botIdx = pushVert(pos, norm, uv, 0, botY - 0.05, 0, 0, -1, 0, 0.5, 1);
  const topRing = heightSegs * (facets + 1);
  const botRing = 0;
  for (let f = 0; f < facets; f++) {
    idx.push(topIdx, topRing + f, topRing + f + 1);
    idx.push(botIdx, botRing + f + 1, botRing + f);
  }

  return buildArrays(pos, norm, uv, idx);
}

// ─── cil-vector: Elongated capsule with groove lines ────────────────────────
// Embedding = data mapping. Smooth capsule with parallel grooves.

export function createVectorCapsule(segs = 20, rings = 12): ProceduralMesh {
  const pos: number[] = [];
  const norm: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];

  const bodyLen = 0.6;  // cylinder portion
  const capR = 0.2;     // hemisphere radius

  // Total: bottom cap + cylinder + top cap
  const totalRings = rings + rings + rings; // cap + body + cap

  for (let ring = 0; ring <= totalRings; ring++) {
    const t = ring / totalRings;
    let y: number, radius: number, ny: number;

    if (ring <= rings) {
      // Bottom hemisphere
      const capT = ring / rings;
      const angle = (1 - capT) * Math.PI * 0.5;  // π/2..0
      y = -bodyLen / 2 - Math.sin(angle) * capR;
      radius = Math.cos(angle) * capR;
      ny = -Math.sin(angle);
    } else if (ring <= rings * 2) {
      // Cylinder body
      const bodyT = (ring - rings) / rings;
      y = -bodyLen / 2 + bodyT * bodyLen;
      radius = capR;
      ny = 0;
    } else {
      // Top hemisphere
      const capT = (ring - rings * 2) / rings;
      const angle = capT * Math.PI * 0.5;  // 0..π/2
      y = bodyLen / 2 + Math.sin(angle) * capR;
      radius = Math.cos(angle) * capR;
      ny = Math.sin(angle);
    }

    for (let seg = 0; seg <= segs; seg++) {
      const phi = (seg / segs) * Math.PI * 2;
      const cosP = Math.cos(phi);
      const sinP = Math.sin(phi);

      // Groove: small radius modulation every 4 segments along body
      let groove = 0;
      if (ring > rings && ring <= rings * 2) {
        groove = Math.sin(phi * 6) * 0.015;
      }
      const r = radius + groove;

      const nx = cosP;
      const nz = sinP;
      const [nnx, nny, nnz] = normalise(nx, ny, nz);

      pushVert(pos, norm, uv,
        cosP * r, y, sinP * r,
        nnx, nny, nnz,
        seg / segs, t,
      );

      if (ring < totalRings && seg < segs) {
        const cur = ring * (segs + 1) + seg;
        const nxt = cur + segs + 1;
        idx.push(cur, nxt, cur + 1);
        idx.push(cur + 1, nxt, nxt + 1);
      }
    }
  }

  return buildArrays(pos, norm, uv, idx);
}

// ─── cil-plus: Cross-shaped junction node ───────────────────────────────────
// Add/Norm = merging pathways. 3D plus sign with rounded edges.

export function createPlusCross(): ProceduralMesh {
  const pos: number[] = [];
  const norm: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];

  // Build a plus from 3 boxes (X-arm, Y-arm, Z-arm) with slight rounding
  const armLen = 0.45;
  const armW = 0.15;
  const armD = 0.12;

  function addBox(
    cx: number, cy: number, cz: number,
    sx: number, sy: number, sz: number,
  ) {
    // 8 corners, 6 faces, 24 verts
    const baseIdx = pos.length / 3;
    const hx = sx / 2, hy = sy / 2, hz = sz / 2;

    // 6 face quads
    const faces: Array<{
      corners: [number, number, number][];
      n: [number, number, number];
    }> = [
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

  // X arm (horizontal)
  addBox(0, 0, 0, armLen * 2, armW, armD);
  // Y arm (vertical)
  addBox(0, 0, 0, armW, armLen * 2, armD);
  // Center sphere (small, to round the junction)
  // Approximate with a slightly larger box rotated 45°
  addBox(0, 0, 0, armW * 1.2, armW * 1.2, armD * 1.3);

  return buildArrays(pos, norm, uv, idx);
}

// ─── cil-arrow-right: Streamlined arrow ─────────────────────────────────────
// Output/forward = directional flow. Arrow shape pointing right (+X).

export function createArrow(segs = 12): ProceduralMesh {
  const pos: number[] = [];
  const norm: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];

  // Arrow profile: shaft + head
  // Shaft: cylinder from x=-0.4 to x=0.1
  // Head: cone from x=0.1 to x=0.5
  const shaftR = 0.12;
  const headR = 0.28;
  const shaftStart = -0.4;
  const shaftEnd = 0.1;
  const headEnd = 0.5;

  // Shaft cylinder
  for (let i = 0; i <= 1; i++) {
    const x = i === 0 ? shaftStart : shaftEnd;
    for (let s = 0; s <= segs; s++) {
      const angle = (s / segs) * Math.PI * 2;
      const y = Math.cos(angle) * shaftR;
      const z = Math.sin(angle) * shaftR;
      const [nx, ny, nz] = normalise(0, Math.cos(angle), Math.sin(angle));
      pushVert(pos, norm, uv, x, y, z, nx, ny, nz, i, s / segs);
    }
  }
  // Shaft indices
  for (let s = 0; s < segs; s++) {
    const a = s, b = s + 1, c = (segs + 1) + s, d = (segs + 1) + s + 1;
    idx.push(a, c, b, b, c, d);
  }

  // Head cone
  const headBase = pos.length / 3;
  // Base ring (at shaftEnd, radius = headR)
  for (let s = 0; s <= segs; s++) {
    const angle = (s / segs) * Math.PI * 2;
    const y = Math.cos(angle) * headR;
    const z = Math.sin(angle) * headR;
    // Cone normal: points outward and slightly forward
    const [nx, ny, nz] = normalise(headR, Math.cos(angle) * (headEnd - shaftEnd), Math.sin(angle) * (headEnd - shaftEnd));
    pushVert(pos, norm, uv, shaftEnd, y, z, nx, ny, nz, 0, s / segs);
  }
  // Tip vertex
  const tipIdx = pushVert(pos, norm, uv, headEnd, 0, 0, 1, 0, 0, 0.5, 0.5);
  for (let s = 0; s < segs; s++) {
    idx.push(headBase + s, tipIdx, headBase + s + 1);
  }

  // Shaft back cap
  const backCapIdx = pushVert(pos, norm, uv, shaftStart, 0, 0, -1, 0, 0, 0.5, 0.5);
  for (let s = 0; s < segs; s++) {
    idx.push(backCapIdx, s + 1, s);
  }

  // Head base cap (ring at shaftEnd, connecting shaft to head flare)
  const headCapIdx = pushVert(pos, norm, uv, shaftEnd, 0, 0, -1, 0, 0, 0.5, 0.5);
  for (let s = 0; s < segs; s++) {
    idx.push(headCapIdx, headBase + s + 1, headBase + s);
  }

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
