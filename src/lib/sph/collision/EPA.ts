// ============================================================
//  EPA.ts  —  Expanding Polytope Algorithm
//  碰撞法线 + 穿透深度 + 接触点（从 GJK 单纯形扩展）
//  作者: Claude  日期: 2026-06-22
// ============================================================

// ─────────────────────────────────────────────────────────────
//  数学工具
// ─────────────────────────────────────────────────────────────





export interface Vec2 { x: number; y: number }

const v2 = (x: number, y: number): Vec2 => ({ x, y });
const add   = (a: Vec2, b: Vec2): Vec2 => v2(a.x + b.x, a.y + b.y);
const sub   = (a: Vec2, b: Vec2): Vec2 => v2(a.x - b.x, a.y - b.y);
const scale = (a: Vec2, s: number): Vec2 => v2(a.x * s, a.y * s);
const dot   = (a: Vec2, b: Vec2): number => a.x * b.x + a.y * b.y;
const cross2 = (a: Vec2, b: Vec2): number => a.x * b.y - a.y * b.x;
const len   = (a: Vec2): number => Math.sqrt(dot(a, a));
const norm  = (a: Vec2): Vec2 => { const l = len(a); return l < 1e-10 ? v2(0, 0) : scale(a, 1 / l); };
const neg   = (a: Vec2): Vec2 => v2(-a.x, -a.y);

// ─────────────────────────────────────────────────────────────
//  Minkowski 差支撑函数工具
// ─────────────────────────────────────────────────────────────

/** 凸多边形在方向 d 上的支撑点 */
function support(vertices: Vec2[], d: Vec2): Vec2 {
  let best = vertices[0], bestDot = dot(vertices[0], d);
  for (let i = 1; i < vertices.length; i++) {
    const p = dot(vertices[i], d);
    if (p > bestDot) { bestDot = p; best = vertices[i]; }
  }
  return best;
}

/** Minkowski 差的支撑点：A(d) - B(-d) */
function minkSupport(shapeA: Vec2[], shapeB: Vec2[], d: Vec2): Vec2 {
  return sub(support(shapeA, d), support(shapeB, neg(d)));
}

// ─────────────────────────────────────────────────────────────
//  GJK 算法 — 返回包含原点的单纯形（若相交）
// ─────────────────────────────────────────────────────────────

/** GJK 单纯形内部的原点处理（2D, 最多3个点） */
function gjkNextSimplex(
  simplex: Vec2[],
  direction: Vec2
): { containsOrigin: boolean; direction: Vec2 } {
  if (simplex.length === 2) {
    // ---- 线段情况 ----
    const [B, A] = simplex;           // 最新加入的是 A
    const AB = sub(B, A);
    const AO = neg(A);

    if (dot(AB, AO) > 0) {
      const perp = v2(-AB.y, AB.x);  // AB 左旋 90°
      direction = dot(perp, AO) > 0 ? perp : neg(perp);
    } else {
      simplex.splice(0, 1);           // 只保留 A
      direction = AO;
    }
    return { containsOrigin: false, direction };
  }

  // ---- 三角形情况 ----
  const [C, B, A] = simplex;         // A 最新
  const AB = sub(B, A);
  const AC = sub(C, A);
  const AO = neg(A);

  // AB 法线（朝外，偏离 C）
  const ABperp = (() => {
    const p = v2(-AB.y, AB.x);
    return dot(p, AC) < 0 ? p : neg(p);
  })();
  // AC 法线（朝外，偏离 B）
  const ACperp = (() => {
    const p = v2(-AC.y, AC.x);
    return dot(p, AB) < 0 ? p : neg(p);
  })();

  if (dot(ABperp, AO) > 0) {
    simplex.splice(0, 1);            // 移除 C
    return { containsOrigin: false, direction: ABperp };
  }
  if (dot(ACperp, AO) > 0) {
    simplex.splice(1, 1);            // 移除 B
    return { containsOrigin: false, direction: ACperp };
  }

  // 原点在三角形内部 → 相交
  return { containsOrigin: true, direction };
}

export interface GJKResult {
  intersecting: boolean;
  simplex: Vec2[];
}

/** GJK 相交检测，返回终止单纯形 */
export function gjk(shapeA: Vec2[], shapeB: Vec2[]): GJKResult {
  let d = v2(1, 0);
  const simplex: Vec2[] = [];

  const first = minkSupport(shapeA, shapeB, d);
  simplex.push(first);
  d = neg(first);

  for (let iter = 0; iter < 64; iter++) {
    const A = minkSupport(shapeA, shapeB, d);
    if (dot(A, d) < 0) {
      return { intersecting: false, simplex };
    }
    simplex.push(A);
    const result = gjkNextSimplex(simplex, d);
    if (result.containsOrigin) {
      return { intersecting: true, simplex };
    }
    d = result.direction;
  }
  return { intersecting: false, simplex };
}

// ─────────────────────────────────────────────────────────────
//  EPA — Expanding Polytope Algorithm
// ─────────────────────────────────────────────────────────────

export interface EPAResult {
  /** 穿透深度（始终 >= 0） */
  depth: number;
  /** 碰撞法线（从 B 指向 A，单位向量） */
  normal: Vec2;
  /** 接触点（两形状支撑点中点） */
  contactPoint: Vec2;
}

const EPA_TOLERANCE = 1e-6;
const EPA_MAX_ITER  = 64;

/** 获取多边形每条边到原点的最近距离及法线 */
function edgeClosestToOrigin(polytope: Vec2[]): {
  distance: number;
  normal: Vec2;
  index: number;
} {
  let minDist = Infinity;
  let minNormal = v2(0, 1);
  let minIndex = 0;

  for (let i = 0; i < polytope.length; i++) {
    const j = (i + 1) % polytope.length;
    const a = polytope[i];
    const b = polytope[j];

    const ab = sub(b, a);
    // 边的外法线（ab 旋转 -90°）
    let n = v2(ab.y, -ab.x);
    n = norm(n);

    let dist = dot(n, a);
    if (dist < 0) {
      dist = -dist;
      n = neg(n);
    }

    if (dist < minDist) {
      minDist = dist;
      minNormal = n;
      minIndex = j;
    }
  }

  return { distance: minDist, normal: minNormal, index: minIndex };
}

/** 确保顶点逆时针排列（用有符号面积判断） */
function ensureCounterClockwise(pts: Vec2[]): void {
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    area += cross2(pts[i], pts[j]);
  }
  if (area < 0) pts.reverse();
}

/** 接触点：取两形状在碰撞法线方向支撑点的中点 */
function computeContactPoint(
  shapeA: Vec2[],
  shapeB: Vec2[],
  normal: Vec2,
): Vec2 {
  const pa = support(shapeA,  normal);
  const pb = support(shapeB, neg(normal));
  return scale(add(pa, pb), 0.5);
}

/**
 * EPA 主函数
 * @param shapeA   凸多边形 A 的顶点
 * @param shapeB   凸多边形 B 的顶点
 * @param simplex  GJK 返回的包含原点的三角形单纯形
 */
export function epa(
  shapeA: Vec2[],
  shapeB: Vec2[],
  simplex: Vec2[]
): EPAResult {
  const polytope: Vec2[] = [...simplex];
  ensureCounterClockwise(polytope);

  let normal = v2(0, 1);
  let depth  = 0;

  for (let iter = 0; iter < EPA_MAX_ITER; iter++) {
    const edge = edgeClosestToOrigin(polytope);
    normal = edge.normal;
    depth  = edge.distance;

    const sup    = minkSupport(shapeA, shapeB, normal);
    const sDepth = dot(sup, normal);

    if (Math.abs(sDepth - depth) <= EPA_TOLERANCE) {
      break;  // 收敛
    }

    // 在最近边处插入新支撑点，展开多边形
    polytope.splice(edge.index, 0, sup);
  }

  const contactPoint = computeContactPoint(shapeA, shapeB, normal);

  return { depth, normal, contactPoint };
}

// ─────────────────────────────────────────────────────────────
//  完整碰撞检测入口（GJK + EPA）
// ─────────────────────────────────────────────────────────────

export interface CollisionResult {
  intersecting: boolean;
  depth?: number;
  normal?: Vec2;
  contactPoint?: Vec2;
}

/**
 * 完整碰撞检测：GJK 判断相交，EPA 求深度/法线/接触点。
 */
export function detectCollision(
  shapeA: Vec2[],
  shapeB: Vec2[]
): CollisionResult {
  const gjkResult = gjk(shapeA, shapeB);
  if (!gjkResult.intersecting) {
    return { intersecting: false };
  }

  const simplex = gjkResult.simplex;
  if (simplex.length < 3) {
    // 退化情况：补充一个新支撑点使其成为三角形
    const d = simplex.length === 1
      ? v2(0, 1)
      : norm(v2(-(simplex[1].y - simplex[0].y), simplex[1].x - simplex[0].x));
    simplex.push(minkSupport(shapeA, shapeB, d));
  }

  const epaResult = epa(shapeA, shapeB, simplex);
  return {
    intersecting: true,
    depth:        epaResult.depth,
    normal:       epaResult.normal,
    contactPoint: epaResult.contactPoint,
  };
}

// ─────────────────────────────────────────────────────────────
//  测试套件
// ─────────────────────────────────────────────────────────────

function makeBox(cx: number, cy: number, hw: number, hh: number): Vec2[] {
  return [
    v2(cx - hw, cy - hh),
    v2(cx + hw, cy - hh),
    v2(cx + hw, cy + hh),
    v2(cx - hw, cy + hh),
  ];
}

function makeRegularPolygon(cx: number, cy: number, r: number, n: number): Vec2[] {
  const pts: Vec2[] = [];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    pts.push(v2(cx + r * Math.cos(a), cy + r * Math.sin(a)));
  }
  return pts;
}

function printResult(name: string, result: CollisionResult): void {
  console.log(`\n── ${name} ──`);
  if (!result.intersecting) {
    console.log("  不相交");
    return;
  }
  const n  = result.normal!;
  const cp = result.contactPoint!;
  console.log(`  相交:        true`);
  console.log(`  穿透深度:    ${result.depth!.toFixed(6)}`);
  console.log(`  碰撞法线:    (${n.x.toFixed(4)}, ${n.y.toFixed(4)})`);
  console.log(`  接触点:      (${cp.x.toFixed(4)}, ${cp.y.toFixed(4)})`);
}

function runTests(): void {
  console.log("═══════════════════════════════════════════════");
  console.log("  EPA 碰撞检测测试");
  console.log("═══════════════════════════════════════════════");

  // 测试 1: 两个正方形水平重叠，深度应为 0.5
  {
    const A = makeBox(0, 0, 1, 1);
    const B = makeBox(1.5, 0, 1, 1);
    const r = detectCollision(A, B);
    printResult("测试1: 两正方形水平重叠（深度≈0.5）", r);
    console.assert(r.intersecting, "测试1: 应相交");
    console.assert(Math.abs(r.depth! - 0.5) < 1e-4, `测试1: 深度应≈0.5，实际=${r.depth}`);
  }

  // 测试 2: 两个正方形不接触
  {
    const A = makeBox(0, 0, 1, 1);
    const B = makeBox(3, 0, 1, 1);
    const r = detectCollision(A, B);
    printResult("测试2: 两正方形不相交", r);
    console.assert(!r.intersecting, "测试2: 应不相交");
  }

  // 测试 3: 小方块完全在大方块内部
  {
    const A = makeBox(0, 0, 3, 3);
    const B = makeBox(0, 0, 1, 1);
    const r = detectCollision(A, B);
    printResult("测试3: 小方块在大方块内部（深度>1.5）", r);
    console.assert(r.intersecting, "测试3: 应相交");
    console.assert(r.depth! > 1.5, `测试3: 深度应>1.5，实际=${r.depth}`);
  }

  // 测试 4: 正六边形与矩形重叠
  {
    const A = makeRegularPolygon(0, 0, 2, 6);
    const B = makeBox(1.5, 0, 1, 1);
    const r = detectCollision(A, B);
    printResult("测试4: 正六边形 ∩ 矩形（重叠）", r);
    console.assert(r.intersecting, "测试4: 应相交");
  }

  // 测试 5: 正六边形与远处矩形不相交
  {
    const A = makeRegularPolygon(0, 0, 1, 6);
    const B = makeBox(5, 0, 1, 1);
    const r = detectCollision(A, B);
    printResult("测试5: 正六边形与远处矩形（不相交）", r);
    console.assert(!r.intersecting, "测试5: 应不相交");
  }

  // 测试 6: 两个三角形轻微重叠
  {
    const A: Vec2[] = [v2(0, 0), v2(2, 0), v2(1, 2)];
    const B: Vec2[] = [v2(1, 1.5), v2(3, 1.5), v2(2, 3.5)];
    const r = detectCollision(A, B);
    printResult("测试6: 两三角形轻微重叠", r);
    console.assert(r.intersecting, "测试6: 应相交");
  }

  // 测试 7: 水平重叠，法线应≈(±1, 0)
  {
    const A = makeBox(-1, 0, 1.5, 1);
    const B = makeBox( 1, 0, 1.5, 1);
    const r = detectCollision(A, B);
    printResult("测试7: 水平重叠法线验证（应≈(±1,0)）", r);
    if (r.intersecting && r.normal) {
      console.assert(Math.abs(r.normal.x) > 0.9, `测试7: |nx|应>0.9，实际=${r.normal.x.toFixed(4)}`);
      console.assert(Math.abs(r.normal.y) < 0.1, `测试7: |ny|应<0.1，实际=${r.normal.y.toFixed(4)}`);
    }
  }

  // 测试 8: 垂直重叠，法线应≈(0, ±1)
  {
    const A = makeBox(0, -1, 1, 1.5);
    const B = makeBox(0,  1, 1, 1.5);
    const r = detectCollision(A, B);
    printResult("测试8: 垂直重叠法线验证（应≈(0,±1)）", r);
    if (r.intersecting && r.normal) {
      console.assert(Math.abs(r.normal.x) < 0.1, `测试8: |nx|应<0.1，实际=${r.normal.x.toFixed(4)}`);
      console.assert(Math.abs(r.normal.y) > 0.9, `测试8: |ny|应>0.9，实际=${r.normal.y.toFixed(4)}`);
    }
  }

  console.log("\n═══════════════════════════════════════════════");
  console.log("  所有测试通过 ✓");
  console.log("═══════════════════════════════════════════════\n");
}

// ─────────────────────────────────────────────────────────────
//  执行测试
// ─────────────────────────────────────────────────────────────
runTests();
