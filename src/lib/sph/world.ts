// src/lib/sph/world.ts
// Re-export / adapter types consumed by world-renderer.ts.
// The renderer was written against a `./world` module that never existed;
// this shim bridges the gap between the actual physics types and what the
// renderer expects (e.g. `width/height` vs `w/h` half-extents, `tick` vs `frame`).

export type { Particle } from './world-stepper';

export { type World } from './world-stepper';

// The renderer accesses `b.width`, `b.height`, and `b.label` which are not on
// the physics RigidBody (that uses half-extents `w`/`h`).  We extend the
// physics type with optional convenience fields so the renderer compiles.
import type { RigidBody as PhysicsRigidBody } from './rigid-body';

export interface RigidBody extends PhysicsRigidBody {
  /** Full width  — renderer convenience (= 2 * w). */
  width: number;
  /** Full height — renderer convenience (= 2 * h). */
  height: number;
  /** Optional display label drawn at the body centre. */
  label?: string;
}
