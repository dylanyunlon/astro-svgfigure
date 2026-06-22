/**
 * SAT.test.ts — unit tests for SAT collision detection
 *
 * Run with:
 *   npx vitest run              (Vitest)
 *   npx jest                    (Jest + ts-jest)
 *
 * The tests are framework-agnostic: they use a tiny inline assertion helper
 * so you can also execute the file directly with `npx tsx SAT.test.ts`.
 */

import {
  satPolygonPolygon,
  satCirclePolygon,
  satCircleCircle,
  Polygon,
  Circle,
  CollisionResult,
  Vec2,
} from "./SAT";

// ─── Tiny assertion helpers ───────────────────────────────────────────────────

let passed = 0;
let failed = 0;

function expect(description: string, value: boolean): void {
  if (value) {
    console.log(`  ✓ ${description}`);
    passed++;
  } else {
    console.error(`  ✗ ${description}`);
    failed++;
  }
}

function approx(a: number, b: number, eps = 1e-6): boolean {
  return Math.abs(a - b) < eps;
}

function vecApprox(a: Vec2, b: Vec2, eps = 1e-5): boolean {
  return approx(a.x, b.x, eps) && approx(a.y, b.y, eps);
}

function isUnit(v: Vec2, eps = 1e-6): boolean {
  const len = Math.sqrt(v.x * v.x + v.y * v.y);
  return approx(len, 1, eps);
}

// ─── Shape factories ─────────────────────────────────────────────────────────

/** Axis-aligned rectangle centred at (cx, cy). */
function rect(cx: number, cy: number, hw: number, hh: number): Polygon {
  return {
    vertices: [
      { x: cx - hw, y: cy - hh },
      { x: cx + hw, y: cy - hh },
      { x: cx + hw, y: cy + hh },
      { x: cx - hw, y: cy + hh },
    ],
  };
}

/** Equilateral triangle centred at (cx, cy). */
function triangle(cx: number, cy: number, r: number): Polygon {
  return {
    vertices: [0, 120, 240].map((deg) => {
      const rad = (deg * Math.PI) / 180;
      return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
    }),
  };
}

function circle(cx: number, cy: number, radius: number): Circle {
  return { center: { x: cx, y: cy }, radius };
}

// ─── satCircleCircle ──────────────────────────────────────────────────────────

console.log("\n── satCircleCircle ──────────────────────────────────────────");

{
  const a = circle(0, 0, 1);
  const b = circle(1.5, 0, 1);          // overlap = 0.5
  const r = satCircleCircle(a, b);

  expect("overlapping circles → not null", r !== null);
  expect("depth ≈ 0.5", approx(r!.depth, 0.5));
  expect("normal is unit vector", isUnit(r!.normal));
  expect("normal points roughly +x (from A toward B)", r!.normal.x > 0.9);
}

{
  const a = circle(0, 0, 1);
  const b = circle(3, 0, 1);            // gap = 1, no overlap
  const r = satCircleCircle(a, b);
  expect("separated circles → null", r === null);
}

{
  const a = circle(0, 0, 1);
  const b = circle(2, 0, 1);            // exactly touching (depth = 0)
  const r = satCircleCircle(a, b);
  expect("touching circles → null (depth = 0 means no penetration)", r === null);
}

{
  const a = circle(0, 0, 2);
  const b = circle(0, 1, 2);            // vertical overlap = 3
  const r = satCircleCircle(a, b);
  expect("vertical overlap → not null", r !== null);
  expect("depth ≈ 3", approx(r!.depth, 3));
  expect("normal is unit vector", isUnit(r!.normal));
  expect("normal points roughly +y", r!.normal.y > 0.9);
}

{
  // Concentric circles — should not throw
  const a = circle(5, 5, 1);
  const b = circle(5, 5, 2);
  const r = satCircleCircle(a, b);
  expect("concentric circles → not null", r !== null);
  expect("normal is unit vector (arbitrary axis)", isUnit(r!.normal));
  expect("depth ≈ 3 (1+2)", approx(r!.depth, 3));
}

// ─── satPolygonPolygon ────────────────────────────────────────────────────────

console.log("\n── satPolygonPolygon ────────────────────────────────────────");

{
  // rect(0,0,1,1): x∈[-1,1]; rect(3,0,1,1): x∈[2,4] → gap of 1 on X
  const a = rect(0, 0, 1, 1);
  const b = rect(3, 0, 1, 1);
  const r = satPolygonPolygon(a, b);
  expect("separated rects (x-axis gap) → null", r === null);
}

{
  const a = rect(0, 0, 1, 1);
  const b = rect(0, 3, 1, 1);          // gap on Y
  const r = satPolygonPolygon(a, b);
  expect("separated rects (y-axis gap) → null", r === null);
}

{
  // rect(0,0,1,1): x∈[-1,1]; rect(2,0,1,1): x∈[1,3] → share exactly x=1 (depth=0)
  const a = rect(0, 0, 1, 1);
  const b = rect(2, 0, 1, 1);
  const r = satPolygonPolygon(a, b);
  expect("touching rects → null", r === null);
}

{
  // rect(0,0,1,1): x∈[-1,1]; rect(1.5,0,1,1): x∈[0.5,2.5] → X overlap=0.5, Y overlap=2 → min=0.5
  const a = rect(0, 0, 1, 1);
  const b = rect(1.5, 0, 1, 1);
  const r = satPolygonPolygon(a, b);

  expect("overlapping rects → not null", r !== null);
  expect("depth ≈ 0.5 (minimum penetration axis)", approx(r!.depth, 0.5));
  expect("normal is unit vector", isUnit(r!.normal));
  expect("normal points roughly +x (A is left, B is right)", r!.normal.x > 0.9);
}

{
  const a = rect(0, 0, 2, 1);
  const b = rect(0, 1.5, 2, 1);        // 0.5 overlap on Y, 4 on X → min=0.5
  const r = satPolygonPolygon(a, b);

  expect("tall overlap rects → not null", r !== null);
  expect("depth ≈ 0.5", approx(r!.depth, 0.5));
  expect("normal is unit vector", isUnit(r!.normal));
  expect("normal points roughly +y", r!.normal.y > 0.9);
}

{
  // Triangle vs rect
  const a = triangle(0, 0, 1.5);
  const b = rect(1, 0, 0.5, 0.5);
  const r = satPolygonPolygon(a, b);
  expect("triangle-rect overlap → not null", r !== null);
  if (r) {
    expect("normal is unit vector", isUnit(r.normal));
    expect("depth > 0", r.depth > 0);
  }
}

{
  const a = triangle(0, 0, 1);
  const b = triangle(3, 0, 1);         // clearly separated
  const r = satPolygonPolygon(a, b);
  expect("separated triangles → null", r === null);
}

// ─── satCirclePolygon ─────────────────────────────────────────────────────────

console.log("\n── satCirclePolygon ─────────────────────────────────────────");

{
  const c = circle(0, 0, 1);
  const p = rect(2, 0, 1, 1);          // closest point (1,0), gap = 1
  const r = satCirclePolygon(c, p);
  expect("circle far from rect → null", r === null);
}

{
  const c = circle(0, 0, 1.5);
  const p = rect(2, 0, 1, 1);          // circle edge at 1.5, rect left at 1 → overlap = 0.5
  const r = satCirclePolygon(c, p);
  expect("circle overlapping rect face → not null", r !== null);
  expect("depth ≈ 0.5", approx(r!.depth, 0.5, 1e-4));
  expect("normal is unit vector", isUnit(r!.normal));
}

{
  // Circle touching corner of a square — tests vertex axis
  // rect(0,0,1,1): corners at (±1,±1); corner (1,1)
  // circle centre at (2,2): distance to (1,1) = sqrt(2) ≈ 1.4142
  // radius = 1.5 → overlap = 1.5 - sqrt(2) ≈ 0.0858
  const c = circle(2, 2, 1.5);
  const p = rect(0, 0, 1, 1);
  const expectedDepth = 1.5 - Math.SQRT2;
  const r = satCirclePolygon(c, p);
  expect("circle overlapping rect corner → not null", r !== null);
  if (r) {
    expect(`depth ≈ ${expectedDepth.toFixed(4)}`, approx(r.depth, expectedDepth, 1e-4));
    expect("normal is unit vector", isUnit(r.normal));
  }
}

{
  const c = circle(0, 0, 1);
  const p = rect(0, 0.5, 2, 2);        // circle inside rect
  const r = satCirclePolygon(c, p);
  expect("circle inside rect → not null", r !== null);
  if (r) {
    expect("normal is unit vector", isUnit(r.normal));
    expect("depth > 0", r.depth > 0);
  }
}

{
  // Circle completely to the right of a rectangle
  const c = circle(10, 0, 1);
  const p = rect(0, 0, 1, 1);
  const r = satCirclePolygon(c, p);
  expect("circle far right of rect → null", r === null);
}

// ─── Summary ──────────────────────────────────────────────────────────────────

const total = passed + failed;
console.log(`\n${"─".repeat(55)}`);
console.log(`Results: ${passed}/${total} passed${failed > 0 ? `, ${failed} FAILED` : " ✓"}`);
if (failed > 0) process.exit(1);
