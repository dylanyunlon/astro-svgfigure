/**
 * physics-animation.ts — Physical synchronization, skin animation, mirroring, player model, bounce elastics
 */

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Vec3 { x: number; y: number; z: number; }
export interface Quat { x: number; y: number; z: number; w: number; }

export interface BoneTransform {
  position: Vec3;
  rotation: Quat;
  scale: Vec3;
}

export interface PhysicsBody {
  id: string;
  mass: number;
  position: Vec3;
  velocity: Vec3;
  angularVelocity: Vec3;
  restitution: number;
  friction: number;
}

// ─── PhysicalSync ─────────────────────────────────────────────────────────────

export interface PhysicalSyncOptions {
  fixedStep?: number;        // seconds, default 1/60
  maxSubsteps?: number;
  gravity?: Vec3;
  damping?: number;
}

export class PhysicalSync {
  private bodies = new Map<string, PhysicsBody>();
  private readonly fixedStep: number;
  private readonly maxSubsteps: number;
  private readonly gravity: Vec3;
  private readonly damping: number;
  private accumulator = 0;
  private lastTime = 0;

  constructor(opts: PhysicalSyncOptions = {}) {
    this.fixedStep = opts.fixedStep ?? 1 / 60;
    this.maxSubsteps = opts.maxSubsteps ?? 8;
    this.gravity = opts.gravity ?? { x: 0, y: -9.81, z: 0 };
    this.damping = opts.damping ?? 0.99;
  }

  addBody(body: PhysicsBody): void {
    this.bodies.set(body.id, { ...body });
  }

  removeBody(id: string): void {
    this.bodies.delete(id);
  }

  getBody(id: string): PhysicsBody | undefined {
    return this.bodies.get(id);
  }

  update(nowMs: number): void {
    if (this.lastTime === 0) { this.lastTime = nowMs; return; }
    const delta = Math.min((nowMs - this.lastTime) / 1000, 0.25);
    this.lastTime = nowMs;
    this.accumulator += delta;

    let steps = 0;
    while (this.accumulator >= this.fixedStep && steps < this.maxSubsteps) {
      this.step(this.fixedStep);
      this.accumulator -= this.fixedStep;
      steps++;
    }
  }

  private step(dt: number): void {
    for (const body of this.bodies.values()) {
      if (body.mass === 0) continue; // static
      // Semi-implicit Euler integration
      body.velocity.x = (body.velocity.x + this.gravity.x * dt) * this.damping;
      body.velocity.y = (body.velocity.y + this.gravity.y * dt) * this.damping;
      body.velocity.z = (body.velocity.z + this.gravity.z * dt) * this.damping;
      body.position.x += body.velocity.x * dt;
      body.position.y += body.velocity.y * dt;
      body.position.z += body.velocity.z * dt;
    }
  }

  syncTransform(id: string, target: { position: Vec3 }): void {
    const body = this.bodies.get(id);
    if (!body) return;
    target.position.x = body.position.x;
    target.position.y = body.position.y;
    target.position.z = body.position.z;
  }

  applyImpulse(id: string, impulse: Vec3): void {
    const body = this.bodies.get(id);
    if (!body || body.mass === 0) return;
    const invMass = 1 / body.mass;
    body.velocity.x += impulse.x * invMass;
    body.velocity.y += impulse.y * invMass;
    body.velocity.z += impulse.z * invMass;
  }

  dispose(): void {
    this.bodies.clear();
  }
}

// ─── SkinAnimation ────────────────────────────────────────────────────────────

export interface SkinClip {
  name: string;
  duration: number;     // seconds
  fps: number;
  frames: BoneTransform[][];  // [frameIndex][boneIndex]
  loop: boolean;
}

export interface SkinAnimationOptions {
  blendTime?: number;
}

export class SkinAnimation {
  private clips = new Map<string, SkinClip>();
  private currentClip: SkinClip | null = null;
  private previousClip: SkinClip | null = null;
  private time = 0;
  private previousTime = 0;
  private blendAlpha = 1;
  private readonly blendTime: number;
  private speed = 1;

  constructor(opts: SkinAnimationOptions = {}) {
    this.blendTime = opts.blendTime ?? 0.2;
  }

  addClip(clip: SkinClip): void {
    this.clips.set(clip.name, clip);
  }

  play(name: string, blendIn = true): void {
    const clip = this.clips.get(name);
    if (!clip) return;
    if (blendIn && this.currentClip) {
      this.previousClip = this.currentClip;
      this.previousTime = this.time;
      this.blendAlpha = 0;
    }
    this.currentClip = clip;
    this.time = 0;
  }

  setSpeed(s: number): void { this.speed = s; }

  update(dt: number): BoneTransform[] | null {
    if (!this.currentClip) return null;
    this.time += dt * this.speed;
    if (this.currentClip.loop) {
      this.time %= this.currentClip.duration;
    } else {
      this.time = Math.min(this.time, this.currentClip.duration);
    }

    const current = this.sampleClip(this.currentClip, this.time);

    if (this.previousClip && this.blendAlpha < 1) {
      this.blendAlpha = Math.min(1, this.blendAlpha + dt / this.blendTime);
      const previous = this.sampleClip(this.previousClip, this.previousTime);
      return this.blendPose(previous, current, this.blendAlpha);
    }

    return current;
  }

  private sampleClip(clip: SkinClip, t: number): BoneTransform[] {
    const frameF = (t / clip.duration) * (clip.frames.length - 1);
    const frameA = Math.floor(frameF);
    const frameB = Math.min(frameA + 1, clip.frames.length - 1);
    const alpha = frameF - frameA;
    return this.blendPose(clip.frames[frameA], clip.frames[frameB], alpha);
  }

  private blendPose(a: BoneTransform[], b: BoneTransform[], t: number): BoneTransform[] {
    return a.map((bone, i) => ({
      position: lerpVec3(bone.position, b[i].position, t),
      rotation: slerpQuat(bone.rotation, b[i].rotation, t),
      scale: lerpVec3(bone.scale, b[i].scale, t),
    }));
  }

  get currentClipName(): string | null { return this.currentClip?.name ?? null; }
  get progress(): number {
    if (!this.currentClip) return 0;
    return this.time / this.currentClip.duration;
  }
}

// ─── Mirror ───────────────────────────────────────────────────────────────────

export type MirrorAxis = 'x' | 'y' | 'z';

export interface MirrorOptions {
  axis?: MirrorAxis;
  mirrorRotation?: boolean;
}

export class Mirror {
  private readonly axis: MirrorAxis;
  private readonly mirrorRotation: boolean;

  constructor(opts: MirrorOptions = {}) {
    this.axis = opts.axis ?? 'x';
    this.mirrorRotation = opts.mirrorRotation ?? true;
  }

  applyToVec3(v: Vec3): Vec3 {
    return {
      x: this.axis === 'x' ? -v.x : v.x,
      y: this.axis === 'y' ? -v.y : v.y,
      z: this.axis === 'z' ? -v.z : v.z,
    };
  }

  applyToQuat(q: Quat): Quat {
    if (!this.mirrorRotation) return { ...q };
    switch (this.axis) {
      case 'x': return { x: q.x, y: -q.y, z: -q.z, w: q.w };
      case 'y': return { x: -q.x, y: q.y, z: -q.z, w: q.w };
      case 'z': return { x: -q.x, y: -q.y, z: q.z, w: q.w };
    }
  }

  applyToPose(pose: BoneTransform[]): BoneTransform[] {
    return pose.map(bone => ({
      position: this.applyToVec3(bone.position),
      rotation: this.applyToQuat(bone.rotation),
      scale: bone.scale, // scale is always positive
    }));
  }

  /** Mirror a bone name pair list to swap left/right */
  swapBones<T extends Record<string, unknown>>(pose: T[], leftRightPairs: [number, number][]): T[] {
    const result = [...pose];
    for (const [l, r] of leftRightPairs) {
      [result[l], result[r]] = [result[r], result[l]];
    }
    return result;
  }
}

// ─── PlayerModel ──────────────────────────────────────────────────────────────

export interface PlayerModelOptions {
  skinUrl?: string;
  boneCount?: number;
  scale?: number;
}

export interface PlayerState {
  position: Vec3;
  rotation: Quat;
  velocity: Vec3;
  isGrounded: boolean;
  currentAnimation: string;
  health: number;
}

export class PlayerModel {
  private readonly skin: SkinAnimation;
  private readonly physics: PhysicalSync;
  private readonly scale: number;
  private state: PlayerState;
  private boneTransforms: BoneTransform[] = [];

  constructor(id: string, opts: PlayerModelOptions = {}) {
    this.scale = opts.scale ?? 1;
    this.skin = new SkinAnimation({ blendTime: 0.15 });
    this.physics = new PhysicalSync({ gravity: { x: 0, y: -9.81, z: 0 } });
    this.state = {
      position: { x: 0, y: 0, z: 0 },
      rotation: { x: 0, y: 0, z: 0, w: 1 },
      velocity: { x: 0, y: 0, z: 0 },
      isGrounded: false,
      currentAnimation: 'idle',
      health: 100,
    };

    this.physics.addBody({
      id,
      mass: 70,
      position: { ...this.state.position },
      velocity: { ...this.state.velocity },
      angularVelocity: { x: 0, y: 0, z: 0 },
      restitution: 0.0,
      friction: 0.8,
    });
  }

  addAnimation(clip: SkinClip): void {
    this.skin.addClip(clip);
  }

  playAnimation(name: string): void {
    this.skin.play(name);
    this.state.currentAnimation = name;
  }

  update(dt: number, nowMs: number): void {
    this.physics.update(nowMs);
    const bones = this.skin.update(dt);
    if (bones) this.boneTransforms = bones;
  }

  move(dir: Vec3, speed: number): void {
    this.physics.applyImpulse('player', {
      x: dir.x * speed,
      y: 0,
      z: dir.z * speed,
    });
  }

  get currentBones(): BoneTransform[] { return this.boneTransforms; }
  get currentState(): Readonly<PlayerState> { return { ...this.state }; }

  dispose(): void {
    this.physics.dispose();
  }
}

// ─── Bounce / Elastic ─────────────────────────────────────────────────────────

export interface BounceOptions {
  stiffness?: number;    // spring stiffness k
  damping?: number;      // damping coefficient c
  mass?: number;
  clamp?: boolean;
}

/**
 * Spring-damper elastic animation for UI / physics bounce.
 * Uses critically-damped or under-damped spring depending on params.
 */
export class Bounce {
  private position = 0;
  private velocity = 0;
  private target = 0;
  private readonly stiffness: number;
  private readonly damping: number;
  private readonly mass: number;
  private readonly clamp: boolean;

  constructor(initialValue = 0, opts: BounceOptions = {}) {
    this.position = initialValue;
    this.target = initialValue;
    this.stiffness = opts.stiffness ?? 200;
    this.damping = opts.damping ?? 20;
    this.mass = opts.mass ?? 1;
    this.clamp = opts.clamp ?? false;
  }

  setTarget(t: number): void { this.target = t; }
  setPosition(p: number): void { this.position = p; this.velocity = 0; }

  /** Impulse-add velocity */
  impulse(v: number): void { this.velocity += v; }

  update(dt: number): number {
    // Spring: F = -k*(x - target) - c*v
    const force = -this.stiffness * (this.position - this.target) - this.damping * this.velocity;
    const acceleration = force / this.mass;
    this.velocity += acceleration * dt;
    this.position += this.velocity * dt;

    if (this.clamp) {
      const lo = Math.min(0, this.target);
      const hi = Math.max(0, this.target);
      if (this.position < lo) { this.position = lo; this.velocity = Math.abs(this.velocity) * 0.4; }
      if (this.position > hi) { this.position = hi; this.velocity = -Math.abs(this.velocity) * 0.4; }
    }

    return this.position;
  }

  get value(): number { return this.position; }
  get isSettled(): boolean { return Math.abs(this.velocity) < 0.001 && Math.abs(this.position - this.target) < 0.001; }
}

// ─── Math Helpers ─────────────────────────────────────────────────────────────

function lerpVec3(a: Vec3, b: Vec3, t: number): Vec3 {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t };
}

function slerpQuat(a: Quat, b: Quat, t: number): Quat {
  let dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  let bx = b.x, by = b.y, bz = b.z, bw = b.w;
  if (dot < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; dot = -dot; }
  if (dot > 0.9995) {
    return normalize({ x: a.x + (bx - a.x) * t, y: a.y + (by - a.y) * t, z: a.z + (bz - a.z) * t, w: a.w + (bw - a.w) * t });
  }
  const theta0 = Math.acos(dot);
  const theta = theta0 * t;
  const sinTheta = Math.sin(theta);
  const sinTheta0 = Math.sin(theta0);
  const s0 = Math.cos(theta) - dot * sinTheta / sinTheta0;
  const s1 = sinTheta / sinTheta0;
  return { x: s0 * a.x + s1 * bx, y: s0 * a.y + s1 * by, z: s0 * a.z + s1 * bz, w: s0 * a.w + s1 * bw };
}

function normalize(q: Quat): Quat {
  const l = Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  return { x: q.x / l, y: q.y / l, z: q.z / l, w: q.w / l };
}
