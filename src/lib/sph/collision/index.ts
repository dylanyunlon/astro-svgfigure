// src/lib/sph/collision/index.ts
// Barrel export for all collision sub-modules.
//
// kebab-case files are the canonical implementations.
// PascalCase files (AABB.ts, BVHTree.ts, etc.) are retained alongside them;
// unique classes from those files are exported here with no name conflicts.

// ── kebab-case modules ────────────────────────────────────────────────────────





export {
  AABB,
  computeAABB,
  expandAABB,
  mergeAABB,
  aabbArea,
  testAABB,
  containsPoint,
  raycastAABB,
} from './aabb-manager';

export {
// [orphan3]   BVHNode,
// [orphan3]   RaycastHit,
// [orphan3]   BVHTree,
} from './bvh-tree';

export {
// [orphan3]   SortAndSweep,
} from './sort-and-sweep';

export {
// [orphan3]   ConvexShape,
// [orphan3]   GJKResult,
// [orphan3]   EPAResult,
// [orphan3]   createBoxShape,
// [orphan3]   createCircleShape,
// [orphan3]   gjk,
// [orphan3]   epa,
} from './gjk-epa';

export {
// [orphan3]   OBB,
// [orphan3]   SATResult,
// [orphan3]   projectOBB,
// [orphan3]   satTest,
} from './sat-solver';

export {
// [orphan3]   ContactPoint,
// [orphan3]   Body,
// [orphan3]   ContactManifold,
// [orphan3]   generateContacts,
// [orphan3]   warmStartManifold,
// [orphan3]   combineFriction,
// [orphan3]   combineRestitution,
} from './contact-manifold';

export {
// [orphan3]   RigidBody,
// [orphan3]   Vec3,
// [orphan3]   Constraint,
// [orphan3]   NonPenetrationConstraint,
// [orphan3]   FrictionConstraint,
// [orphan3]   RestitutionConstraint,
} from './constraints';

export {
// [orphan3]   SolverBody,
// [orphan3]   SolverConfig,
// [orphan3]   defaultSolverConfig,
// [orphan3]   solveConstraints,
} from './impulse-solver';

export {
// [orphan3]   RayHit,
// [orphan3]   OverlapResult,
// [orphan3]   ClosestPointResult,
// [orphan3]   SceneQuery,
} from './scene-query';

export {
// [orphan3]   CollisionWorldConfig,
// [orphan3]   WorldStats,
// [orphan3]   CollisionWorld,
} from './collision-world';

// ── Collision Event System ───────────────────────────────────────────────────
export {
  CollisionEventDispatcher,
  CollisionCache,
  EventQueue,
  makePairKey,
} from './CollisionEvents';
export type {
// [orphan3]   CollisionPhase,
// [orphan3]   CollisionContactInfo,
// [orphan3]   CollisionEvent,
// [orphan3]   ActiveContactPair,
// [orphan3]   CollisionCallback,
} from './CollisionEvents';

// ── PascalCase modules (unique exports only, no name conflicts) ───────────────

// ContactSolver.ts — impulse-based contact resolution class
export { ContactSolver } from './ContactSolver';

// PositionSolver.ts — Baumgarte position correction class
export { PositionSolver } from './PositionSolver';

// ContactConstraint type (defined in ../types, surfaced via ContactSolver)
export type { ContactConstraint } from './ContactSolver';

// AABB.ts — standalone AABB utility functions (unique to this file)
export {
  aabbOverlap,
  aabbUnion,
  aabbPerimeter,
  aabbExpand,
  aabbFromCircle,
  aabbFromPoints,
  aabbContains,
  aabbCenter,
} from './AABB';

// CollisionWorld.ts — monolithic collision world (unique exports only)
export {
  vec2,
  aabbFromBox,
  computeContactInfo,
  createCircleBody,
  createBoxBody,
  resetIdCounter,
} from './CollisionWorld';
export type {
// [orphan3]   Vec2,
// [orphan3]   ShapeType,
// [orphan3]   BodyType,
// [orphan3]   BroadPhasePair,
// [orphan3]   ContactInfo,
// [orphan3]   SPHParticle,
// [orphan3]   SPHWorld,
} from './CollisionWorld';

// EPA.ts — standalone GJK/EPA on Vec2[] arrays (unique exports only)
export {
  detectCollision,
} from './EPA';
export type {
// [orphan3]   CollisionResult,
} from './EPA';

// GJK.ts — class-based shapes + collision (unique exports only)
export {
  Circle,
  Polygon,
  collide,
} from './GJK';
export type {
// [orphan3]   CollisionInfo,
} from './GJK';

// ── contact-generator ─────────────────────────────────────────────────────────
export type { GeneratedContact } from './contact-generator';
export {
  generateContacts  as generateContactsFromBodies,
  generateAllContacts,
} from './contact-generator';

// ── math utilities ────────────────────────────────────────────────────────────
export {
  vec2         as mathVec2,
  Vec2Zero,
  vec2Add, vec2Sub, vec2Scale, vec2Negate,
  vec2Dot, vec2Cross, vec2CrossSV, vec2CrossVS,
  vec2LengthSq, vec2Length, vec2Normalize,
  vec2Distance, vec2DistanceSq, vec2Lerp,
  vec2Perp, vec2Clamp, vec2Min, vec2Max, vec2Abs,
  mat2Rotation, Mat2x2Identity,
  mat2MulVec, mat2TransposeMulVec, mat2Mul,
  mat2Transpose, mat2Det, mat2Inverse,
  clamp, approxEqual, sign, wrapAngle,
} from './math';
export type { Vec2 as MathVec2, Mat2x2 } from './math';

// ── rigid-body (2-D rigid body system) ────────────────────────────────────────
export {
  RigidBody2D,
  resetBodyIdCounter,
  createCircleBody  as createCircleBody2D,
  createBoxBody     as createBoxBody2D,
  createPolygonBody,
  computeOutwardNormals,
  polygonArea, polygonInertia,
  bodyToWorld, worldToBody, bodyDirToWorld,
  getWorldVertices, getWorldNormals,
  velocityAtPoint,
  applyForce   as applyForce2D,
  applyImpulse as applyImpulse2D,
  applyCentralImpulse,
  integrateBody,
} from './rigid-body';
export type { CircleDef, PolygonDef, ShapeDef, RigidBody2DOptions } from './rigid-body';

// SceneQuery.ts — full-featured scene query (unique exports only)
export {
  closestPointOnSegment,
  closestPointOnConvex,
  closestPointOnShape,
  aabbFromShape,
  aabbOverlapsAABB,
  rayIntersectsAABB,
} from './SceneQuery';
export type {
// [orphan3]   AABB2,
// [orphan3]   Ray2,
// [orphan3]   CircleShape,
// [orphan3]   AABBShape,
// [orphan3]   CapsuleShape,
// [orphan3]   ConvexPolygonShape,
// [orphan3]   Shape,
// [orphan3]   UserData,
} from './SceneQuery';
