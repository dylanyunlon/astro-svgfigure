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
  BVHNode,
  RaycastHit,
  BVHTree,
} from './bvh-tree';

export {
  SortAndSweep,
} from './sort-and-sweep';

export {
  ConvexShape,
  GJKResult,
  EPAResult,
  createBoxShape,
  createCircleShape,
  gjk,
  epa,
} from './gjk-epa';

export {
  OBB,
  SATResult,
  projectOBB,
  satTest,
} from './sat-solver';

export {
  ContactPoint,
  Body,
  ContactManifold,
  generateContacts,
  warmStartManifold,
  combineFriction,
  combineRestitution,
} from './contact-manifold';

export {
  RigidBody,
  Vec3,
  Constraint,
  NonPenetrationConstraint,
  FrictionConstraint,
  RestitutionConstraint,
} from './constraints';

export {
  SolverBody,
  SolverConfig,
  defaultSolverConfig,
  solveConstraints,
} from './impulse-solver';

export {
  RayHit,
  OverlapResult,
  ClosestPointResult,
  SceneQuery,
} from './scene-query';

export {
  CollisionWorldConfig,
  WorldStats,
  CollisionWorld,
} from './collision-world';

// ── PascalCase modules (unique exports only, no name conflicts) ───────────────

// ContactSolver.ts — impulse-based contact resolution class
export { ContactSolver } from './ContactSolver';

// PositionSolver.ts — Baumgarte position correction class
export { PositionSolver } from './PositionSolver';

// ContactConstraint type (defined in ../types, surfaced via ContactSolver)
export type { ContactConstraint } from './ContactSolver';
