



import { AABB, aabbOverlap, computeAABB } from './AABB';
import { BVHNode, BVHTree, refitBVH, insertLeaf, removeLeaf } from './BVHTree';
import { sortAndSweep, BroadPair } from './SortAndSweep';
import { satNarrow, SATResult } from './SAT';
import { generateContacts, ContactManifold, ContactPoint } from './contact-generator';
import { warmStart, solveConstraints, ConstraintCache } from './ContactSolver';
import { SceneQuery, raycast, shapecast, overlapTest } from './SceneQuery';
import { RigidBody, BodyHandle, BodyType } from './rigid-body';
import {
} from './CollisionEvents';
import { Vec3, mat3Transpose, vec3Add, vec3Scale } from './math';

  CollisionEventDispatcher,
  type ActiveContactPair,

export interface CollisionWorldConfig {
  maxBodies: number;
  solverIterations: number;
  warmStartingFactor: number;
  broadPhaseMargin: number;
  enableContinuous: boolean;
  sleepThreshold: number;
  sleepFrames: number;
}

const DEFAULT_CONFIG: CollisionWorldConfig = {
  maxBodies: 4096,
  solverIterations: 10,
  warmStartingFactor: 0.85,
  broadPhaseMargin: 0.05,
  enableContinuous: false,
  sleepThreshold: 0.01,
  sleepFrames: 60,
};

export interface WorldStats {
  bodyCount: number;
  broadPairs: number;
  narrowTests: number;
  contactCount: number;
  solverIterations: number;
  bvhDepth: number;
  stepTimeMs: number;
  broadPhaseTimeMs: number;
  narrowPhaseTimeMs: number;
  solverTimeMs: number;
}

interface BodyEntry {
  body: RigidBody;
  handle: BodyHandle;
  aabb: AABB;
  bvhLeaf: BVHNode | null;
  sleepCounter: number;
  prevPosition: Vec3;
  prevVelocityMag: number;
}

export class CollisionWorld {
  private config: CollisionWorldConfig;
  private bodies: Map<BodyHandle, BodyEntry> = new Map();
  private nextHandle: BodyHandle = 1 as BodyHandle;
  private bvh: BVHTree;
  private constraintCache: ConstraintCache = new Map();
  private manifolds: ContactManifold[] = [];
  /** Collision event dispatcher — fires onCollisionEnter/Stay/Exit. */
  readonly events: CollisionEventDispatcher = new CollisionEventDispatcher();
  /** Monotone simulation time (seconds). */
  private _simTime = 0;

  private stats: WorldStats = {
    bodyCount: 0,
    broadPairs: 0,
    narrowTests: 0,
    contactCount: 0,
    solverIterations: 0,
    bvhDepth: 0,
    stepTimeMs: 0,
    broadPhaseTimeMs: 0,
    narrowPhaseTimeMs: 0,
    solverTimeMs: 0,
  };

  constructor(config: Partial<CollisionWorldConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.bvh = new BVHTree();
  }

  addBody(body: RigidBody): BodyHandle {
    const handle = this.nextHandle++ as BodyHandle;
    const aabb = computeAABB(body, this.config.broadPhaseMargin);
    const bvhLeaf = insertLeaf(this.bvh, aabb, handle);
    const entry: BodyEntry = {
      body,
      handle,
      aabb,
      bvhLeaf,
      sleepCounter: 0,
      prevPosition: { ...body.position },
      prevVelocityMag: 0,
    };
    this.bodies.set(handle, entry);
    this.stats.bodyCount = this.bodies.size;
    return handle;
  }

  removeBody(handle: BodyHandle): boolean {
    const entry = this.bodies.get(handle);
    if (!entry) return false;
    if (entry.bvhLeaf) {
      removeLeaf(this.bvh, entry.bvhLeaf);
    }
    this.constraintCache.forEach((_, key) => {
      if (key.includes(String(handle))) this.constraintCache.delete(key);
    });
    this.bodies.delete(handle);
    this.stats.bodyCount = this.bodies.size;
    // Evict from collision event cache
    this.events.evictBody(Number(handle));
    return true;
  }

  updateBody(handle: BodyHandle, body: RigidBody): void {
    const entry = this.bodies.get(handle);
    if (!entry) return;
    entry.body = body;
    const newAABB = computeAABB(body, this.config.broadPhaseMargin);
    entry.aabb = newAABB;
    if (entry.bvhLeaf) {
      removeLeaf(this.bvh, entry.bvhLeaf);
    }
    entry.bvhLeaf = insertLeaf(this.bvh, newAABB, handle);
  }

  step(bodies: RigidBody[], dt: number): ContactManifold[] {
    const stepStart = performance.now();

    // Sync external body array into registry
    const handles = Array.from(this.bodies.keys());
    for (let i = 0; i < handles.length && i < bodies.length; i++) {
      const entry = this.bodies.get(handles[i])!;
      entry.body = bodies[i];
    }

    // Refit AABBs and BVH
    this.refitAllAABBs();
    refitBVH(this.bvh);
    this.stats.bvhDepth = this.bvh.root ? this.computeBVHDepth(this.bvh.root) : 0;

    const broadStart = performance.now();

    // Build flat AABB list for sort-and-sweep
    const aabbList: Array<{ aabb: AABB; handleA: BodyHandle }> = [];
    this.bodies.forEach((entry) => {
      aabbList.push({ aabb: entry.aabb, handleA: entry.handle });
    });
    const broadPairs: BroadPair[] = sortAndSweep(aabbList);
    this.stats.broadPairs = broadPairs.length;
    this.stats.broadPhaseTimeMs = performance.now() - broadStart;

    const narrowStart = performance.now();

    // Narrow phase: SAT on each broad pair
    const newManifolds: ContactManifold[] = [];
    let narrowTests = 0;

    for (const pair of broadPairs) {
      const entryA = this.bodies.get(pair.handleA);
      const entryB = this.bodies.get(pair.handleB);
      if (!entryA || !entryB) continue;

      // Skip static-static pairs
      if (
        entryA.body.type === BodyType.Static &&
        entryB.body.type === BodyType.Static
      ) continue;

      // Skip sleeping pairs
      if (this.areBothSleeping(entryA, entryB)) continue;

      narrowTests++;
      const satResult: SATResult = satNarrow(entryA.body, entryB.body);
      if (!satResult.colliding) continue;

      const manifold = generateContacts(
        entryA.body,
        entryB.body,
        pair.handleA,
        pair.handleB,
        satResult
      );

      if (manifold.contacts.length > 0) {
        newManifolds.push(manifold);
      }
    }

    this.stats.narrowTests = narrowTests;
    this.stats.narrowPhaseTimeMs = performance.now() - narrowStart;

    const solverStart = performance.now();

    // Dispatch collision events (Enter / Stay / Exit)
    this._simTime += dt;
    {
      const activePairs: ActiveContactPair[] = [];
      for (const manifold of newManifolds) {
        for (const cp of manifold.contacts) {
          activePairs.push({
            bodyA: Number(manifold.handleA),
            bodyB: Number(manifold.handleB),
            contact: {
              normal: { x: cp.normal.x, y: cp.normal.y },
              depth:  cp.penetrationDepth ?? 0,
              pointA: { x: cp.positionA.x, y: cp.positionA.y },
              pointB: { x: cp.positionB.x, y: cp.positionB.y },
            },
          });
          break; // one representative pair per manifold is sufficient
        }
      }
      this.events.update(activePairs, this._simTime);
    }

    // Warm start from previous frame's impulse cache
    warmStart(newManifolds, this.constraintCache, this.config.warmStartingFactor);

    // Solve velocity constraints (PGS)
    for (let iter = 0; iter < this.config.solverIterations; iter++) {
      solveConstraints(newManifolds, bodies, dt);
    }
    this.stats.solverIterations = this.config.solverIterations;
    this.stats.solverTimeMs = performance.now() - solverStart;

    // Cache manifolds for next frame warm start
    this.updateConstraintCache(newManifolds);
    this.manifolds = newManifolds;

    // Update sleep state
    this.updateSleep(dt);

    this.stats.contactCount = newManifolds.reduce(
      (sum, m) => sum + m.contacts.length, 0
    );
    this.stats.stepTimeMs = performance.now() - stepStart;

    return this.manifolds;
  }

  getSceneQuery(): SceneQuery {
    const bvhSnapshot = this.bvh;
    const bodiesSnapshot = this.bodies;
    return {
      raycast: (origin: Vec3, direction: Vec3, maxDist: number) =>
        raycast(bvhSnapshot, bodiesSnapshot, origin, direction, maxDist),
      shapecast: (shape: RigidBody, maxResults: number) =>
        shapecast(bvhSnapshot, bodiesSnapshot, shape, maxResults),
      overlapTest: (aabb: AABB) =>
        overlapTest(bvhSnapshot, bodiesSnapshot, aabb),
    };
  }

  getStats(): Readonly<WorldStats> {
    return { ...this.stats };
  }

  private refitAllAABBs(): void {
    this.bodies.forEach((entry) => {
      entry.aabb = computeAABB(entry.body, this.config.broadPhaseMargin);
    });
  }

  private computeBVHDepth(node: BVHNode): number {
    if (!node.left && !node.right) return 1;
    const leftDepth = node.left ? this.computeBVHDepth(node.left) : 0;
    const rightDepth = node.right ? this.computeBVHDepth(node.right) : 0;
    return 1 + Math.max(leftDepth, rightDepth);
  }

  private areBothSleeping(a: BodyEntry, b: BodyEntry): boolean {
    return (
      a.sleepCounter >= this.config.sleepFrames &&
      b.sleepCounter >= this.config.sleepFrames
    );
  }

  private updateSleep(dt: number): void {
    this.bodies.forEach((entry) => {
      if (entry.body.type === BodyType.Static) {
        entry.sleepCounter = this.config.sleepFrames;
        return;
      }
      const vel = entry.body.linearVelocity;
      const speed = Math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
      if (speed < this.config.sleepThreshold) {
        entry.sleepCounter++;
      } else {
        entry.sleepCounter = 0;
      }
    });
  }

  private updateConstraintCache(manifolds: ContactManifold[]): void {
    const newCache: ConstraintCache = new Map();
    for (const manifold of manifolds) {
      const key = `${manifold.handleA}:${manifold.handleB}`;
      newCache.set(key, manifold);
    }
    this.constraintCache = newCache;
  }
}
