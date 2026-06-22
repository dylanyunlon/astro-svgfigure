// src/lib/sph/collision/index.ts
// Barrel export for all collision sub-modules.

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
