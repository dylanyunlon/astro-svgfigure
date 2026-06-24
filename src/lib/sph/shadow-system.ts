/**
 * shadow-system.ts — M784: Shadow System — Depth Map + PCF Soft Shadows
 * ─────────────────────────────────────────────────────────────────────────────
 * 没有阴影的世界是平的。
 *
 * 2D directional shadow system for the SPH world. Rigid bodies and obstacles
 * cast shadows onto particles and cells via a 1D depth map rendered from the
 * light's perspective (orthographic projection along the light direction).
 *
 * Inspired by AT's ShadowDepth.fs (440 lines, HAR archive) — ported to a
 * CPU-side 2D analogue suitable for Canvas2D compositing.
 *
 * Algorithm overview:
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Phase 1 ── Shadow Map Generation ──────────────────────────────────────┐
 *   │  Project all occluders (rigid bodies + obstacles) onto the light's 1D   │
 *   │  "depth axis". For each column in the shadow map, store the nearest     │
 *   │  occluder distance from the light. This is a 1D depth buffer scanned   │
 *   │  perpendicular to the light direction.                                  │
 *   │                                                                         │
 *   │  Occluder types:                                                        │
 *   │    • RigidBody (oriented rectangle: x, y, w, h, angle)                 │
 *   │    • ObstacleData (circle: cx, cy, r)                                   │
 *   │                                                                         │
 *   │  For each shadow-map texel (column along the tangent axis):             │
 *   │    – Raycast from light boundary toward the scene along lightDir        │
 *   │    – Record the nearest intersection depth                              │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                  │ depthMap: Float32Array[resolution]
 *                  ▼
 *   ┌─ Phase 2 ── Shadow Sampling ────────────────────────────────────────────┐
 *   │  sampleShadow(x, y):                                                    │
 *   │    Project world point onto light's tangent axis → texel index           │
 *   │    Project world point onto light's depth axis  → point depth            │
 *   │    Compare point depth vs depthMap[texel] + bias                         │
 *   │    → 0 = in shadow (point behind occluder), 1 = lit                     │
 *   │                                                                         │
 *   │  sampleShadowPCF(x, y, radius):                                         │
 *   │    Average N samples in a [texel-radius, texel+radius] window           │
 *   │    Each sample does the same depth comparison                            │
 *   │    → smooth value in [0, 1], soft shadow penumbra                       │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                  │ shadow factor per pixel
 *                  ▼
 *   ┌─ Phase 3 ── Composite (Canvas2D) ──────────────────────────────────────┐
 *   │  applyShadows(ctx, width, height):                                       │
 *   │    For each pixel, compute shadow factor via PCF                         │
 *   │    Multiply pixel luminance by lerp(shadowDarkness, 1.0, factor)        │
 *   │    Ambient occlusion tint for cells in shadow (slight blue shift)        │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 * Performance:
 *   Shadow map is 1D — O(resolution × numOccluders) per frame.
 *   PCF sampling is O(pcfSamples) per query point.
 *   applyShadows iterates all pixels once — O(width × height).
 *   Typical cost: <1ms for 512-texel shadow map + 800×600 composite.
 *
 * Usage:
 *   const shadows = new ShadowSystem(512, [-0.7, -1.0]);
 *   shadows.configure({ bias: 0.005, darkness: 0.35, pcfSamples: 9 });
 *   shadows.setWorldBounds(0, 0, domainW, domainH);
 *   shadows.renderShadowMap(rigidBodies, obstacles);
 *   const lit = shadows.sampleShadowPCF(px, py, 3.0); // per-particle
 *   shadows.applyShadows(ctx, canvas.width, canvas.height); // full-frame
 *
 * Research: xiaodi #M784 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/** Tweakable shadow parameters. */



import type { RigidBody } from './rigid-body';
import type { ObstacleData } from './types';

export interface ShadowConfig {
  /** Depth comparison bias to prevent shadow acne. @default 0.005 */
  bias: number;

  /** Minimum brightness in fully shadowed regions [0, 1]. @default 0.35 */
  darkness: number;

  /** Number of PCF taps for soft shadow edges. Must be odd. @default 9 */
  pcfSamples: number;

  /**
   * Ambient occlusion tint color applied to shadowed regions.
   * Slight cool shift simulates indirect sky illumination.
   * RGB in [0, 1]. @default [0.85, 0.88, 0.95]
   */
  aoTint: [number, number, number];

  /**
   * AO intensity — how strongly the tint is applied in shadow.
   * 0 = no tint, 1 = full tint replacement. @default 0.15
   */
  aoIntensity: number;

  /**
   * Shadow map "far plane" — maximum ray travel distance.
   * Set to the domain diagonal by default.
   * @default Infinity (auto-computed from world bounds)
   */
  farPlane: number;
}

const DEFAULT_CONFIG: ShadowConfig = {
  bias:        0.005,
  darkness:    0.35,
  pcfSamples:  9,
  aoTint:      [0.85, 0.88, 0.95],
  aoIntensity: 0.15,
  farPlane:    Infinity,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers — vector math
// ─────────────────────────────────────────────────────────────────────────────

/** Normalize a 2D vector in-place, return length. */
function normalize2(out: [number, number], v: [number, number]): number {
  const len = Math.sqrt(v[0] * v[0] + v[1] * v[1]);
  if (len < 1e-12) {
    out[0] = 0;
    out[1] = -1; // fallback: light from above
    return 0;
  }
  out[0] = v[0] / len;
  out[1] = v[1] / len;
  return len;
}

/** Dot product of two 2D vectors. */
function dot2(a: [number, number], b: [number, number]): number {
  return a[0] * b[0] + a[1] * b[1];
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers — ray-shape intersection
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Ray vs axis-aligned segment intersection.
 * Returns the parametric t along the ray, or Infinity if no hit.
 *
 * Ray: P = origin + t * dir, t >= 0
 */
function raySegmentT(
  ox: number, oy: number,  // ray origin
  dx: number, dy: number,  // ray direction (unit)
  ax: number, ay: number,  // segment start
  bx: number, by: number,  // segment end
): number {
  // Segment direction
  const sx = bx - ax;
  const sy = by - ay;
  const denom = dx * sy - dy * sx;
  if (Math.abs(denom) < 1e-12) return Infinity; // parallel

  const tx = ((ax - ox) * sy - (ay - oy) * sx) / denom;
  const u  = ((ax - ox) * dy - (ay - oy) * dx) / denom;

  if (tx >= 0 && u >= 0 && u <= 1) return tx;
  return Infinity;
}

/**
 * Ray vs circle intersection.
 * Returns the nearest positive t, or Infinity.
 */
function rayCircleT(
  ox: number, oy: number,
  dx: number, dy: number,
  cx: number, cy: number,
  r: number,
): number {
  const fx = ox - cx;
  const fy = oy - cy;
  const a = dx * dx + dy * dy;          // should be 1 for unit dir
  const b = 2 * (fx * dx + fy * dy);
  const c = fx * fx + fy * fy - r * r;
  let disc = b * b - 4 * a * c;
  if (disc < 0) return Infinity;
  disc = Math.sqrt(disc);
  const t0 = (-b - disc) / (2 * a);
  const t1 = (-b + disc) / (2 * a);
  if (t0 >= 0) return t0;
  if (t1 >= 0) return t1;
  return Infinity;
}

/**
 * Ray vs oriented rectangle (RigidBody).
 * The rectangle has centre (body.x, body.y), half-extents (body.w, body.h),
 * and rotation body.angle. We transform the ray into local body space, then
 * test against 4 axis-aligned edges.
 */
function rayRigidBodyT(
  ox: number, oy: number,
  dx: number, dy: number,
  body: RigidBody,
): number {
  const cos = Math.cos(-body.angle);
  const sin = Math.sin(-body.angle);

  // Transform ray origin into body-local coordinates
  const relX = ox - body.x;
  const relY = oy - body.y;
  const lox = cos * relX - sin * relY;
  const loy = sin * relX + cos * relY;

  // Transform ray direction into body-local coordinates
  const ldx = cos * dx - sin * dy;
  const ldy = sin * dx + cos * dy;

  const hw = body.w;   // half-width
  const hh = body.h;   // half-height

  // Test against 4 edges of the AABB [-hw, -hh] → [hw, hh]
  let tMin = Infinity;

  // Bottom edge: y = -hh, x ∈ [-hw, hw]
  tMin = Math.min(tMin, raySegmentT(lox, loy, ldx, ldy, -hw, -hh,  hw, -hh));
  // Top edge: y = hh, x ∈ [-hw, hw]
  tMin = Math.min(tMin, raySegmentT(lox, loy, ldx, ldy, -hw,  hh,  hw,  hh));
  // Left edge: x = -hw, y ∈ [-hh, hh]
  tMin = Math.min(tMin, raySegmentT(lox, loy, ldx, ldy, -hw, -hh, -hw,  hh));
  // Right edge: x = hw, y ∈ [-hh, hh]
  tMin = Math.min(tMin, raySegmentT(lox, loy, ldx, ldy,  hw, -hh,  hw,  hh));

  return tMin;
}

// ─────────────────────────────────────────────────────────────────────────────
// ShadowSystem
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 2D directional shadow system.
 *
 * Renders a 1D depth map from the light's perspective: each texel represents
 * one column perpendicular to the light direction. Rigid bodies and obstacles
 * cast shadows by occluding rays. World-space points are projected into this
 * 1D space to determine shadow/lit status, with PCF for soft penumbrae.
 */
export class ShadowSystem {
  // ── Resolution & configuration ──────────────────────────────────────────
  private readonly resolution: number;
  private config: ShadowConfig;

  // ── Light basis vectors ─────────────────────────────────────────────────
  /** Normalized light direction (pointing FROM light TOWARD scene). */
  private lightDir: [number, number] = [0, -1];
  /** Tangent axis (perpendicular to lightDir), defines shadow map columns. */
  private tangent:  [number, number] = [1,  0];

  // ── World bounds ────────────────────────────────────────────────────────
  private worldMinX = 0;
  private worldMinY = 0;
  private worldMaxX = 1;
  private worldMaxY = 1;

  /** Length along the tangent axis that the shadow map covers. */
  private tangentSpan = 1;
  /** Origin on the tangent axis (minimum projected coordinate). */
  private tangentOrigin = 0;
  /** Origin on the depth axis (light entry point — the "near plane"). */
  private depthOrigin = 0;

  // ── Shadow map (1D depth buffer) ────────────────────────────────────────
  /** Per-texel nearest occluder depth. Infinity means no occluder. */
  private depthMap: Float32Array;

  /**
   * Per-texel occluder exit depth — used for thickness-based shadow.
   * Allows thin objects to leak some light (semi-transparent shadow).
   */
  private exitMap: Float32Array;

  // ── Scratch buffers for composite ───────────────────────────────────────
  private shadowBuffer: Float32Array | null = null;

  // ─────────────────────────────────────────────────────────────────────────
  // Constructor
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * @param resolution  Number of texels in the 1D shadow map (512 is a good
   *                     default — each texel covers ~domainWidth/512 units).
   * @param lightDir    Direction vector FROM the light TOWARD the scene.
   *                     Does not need to be normalized.
   */
  constructor(resolution: number, lightDir: [number, number]) {
    this.resolution = Math.max(1, resolution | 0);
    this.depthMap   = new Float32Array(this.resolution);
    this.exitMap    = new Float32Array(this.resolution);
    this.config     = { ...DEFAULT_CONFIG };

    this.setLightDirection(lightDir);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public setters
  // ─────────────────────────────────────────────────────────────────────────

  /** Merge partial config into the current configuration. */
  configure(partial: Partial<ShadowConfig>): void {
    Object.assign(this.config, partial);
    // Ensure pcfSamples is odd and ≥ 1
    this.config.pcfSamples = Math.max(1, this.config.pcfSamples | 1);
  }

  /** Update the light direction (does not need to be normalized). */
  setLightDirection(dir: [number, number]): void {
    normalize2(this.lightDir, dir);
    // Tangent = 90° CCW rotation of lightDir
    this.tangent[0] = -this.lightDir[1];
    this.tangent[1] =  this.lightDir[0];
  }

  /**
   * Set the world-space bounding rectangle.
   * Must be called before renderShadowMap so the shadow map covers the scene.
   */
  setWorldBounds(minX: number, minY: number, maxX: number, maxY: number): void {
    this.worldMinX = minX;
    this.worldMinY = minY;
    this.worldMaxX = maxX;
    this.worldMaxY = maxY;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Phase 1 — Shadow Map Generation
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Render the 1D shadow depth map from the current light direction.
   *
   * For each texel column along the tangent axis, we cast a ray in the
   * light direction and record the nearest occluder depth.
   *
   * @returns  The raw depth map (readonly view — do not mutate).
   */
  renderShadowMap(rigidBodies: RigidBody[], obstacles: ObstacleData[]): Float32Array {
    // ── Compute light-space AABB of the world ──────────────────────────────
    this.computeLightSpaceBounds();

    const { resolution, depthMap, exitMap, lightDir, tangent } = this;
    const { farPlane } = this.config;
    const effectiveFar = Number.isFinite(farPlane)
      ? farPlane
      : this.computeDiagonal();

    // Clear depth map to far plane
    depthMap.fill(effectiveFar);
    exitMap.fill(effectiveFar);

    const texelSize = this.tangentSpan / resolution;

    // ── Per-texel raycast ──────────────────────────────────────────────────
    for (let i = 0; i < resolution; i++) {
      // Tangent position for this texel (center of texel)
      const tPos = this.tangentOrigin + (i + 0.5) * texelSize;

      // Ray origin: start at the "near plane" (depthOrigin) along the
      // tangent offset, then march in lightDir.
      const rox = this.tangent[0] * tPos + this.lightDir[0] * this.depthOrigin;
      const roy = this.tangent[1] * tPos + this.lightDir[1] * this.depthOrigin;

      let nearestEntry = effectiveFar;
      let nearestExit  = effectiveFar;

      // ── Test rigid bodies ──────────────────────────────────────────────
      for (let b = 0; b < rigidBodies.length; b++) {
        const body = rigidBodies[b];
        const t = rayRigidBodyT(rox, roy, lightDir[0], lightDir[1], body);
        if (t < nearestEntry) {
          nearestEntry = t;
          // Estimate exit depth: cast backward from far side
          // For a convex body the second hit is the exit
          const exitT = this.computeExitDepth(
            rox, roy, lightDir[0], lightDir[1], body, t,
          );
          nearestExit = exitT;
        }
      }

      // ── Test obstacles (circles) ───────────────────────────────────────
      for (let o = 0; o < obstacles.length; o++) {
        const obs = obstacles[o];
        const t = rayCircleT(
          rox, roy, lightDir[0], lightDir[1],
          obs.cx, obs.cy, obs.r,
        );
        if (t < nearestEntry) {
          nearestEntry = t;
          // Circle exit: the far intersection
          const exitT = this.circleExitT(
            rox, roy, lightDir[0], lightDir[1], obs,
          );
          nearestExit = exitT;
        }
      }

      depthMap[i] = nearestEntry;
      exitMap[i]  = nearestExit;
    }

    return depthMap;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Phase 2 — Shadow Sampling
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Hard shadow sample — returns 0 (full shadow) or 1 (full light).
   *
   * Projects (x, y) into light space, then compares its depth against the
   * shadow map with a small bias to prevent self-shadowing artifacts.
   */
  sampleShadow(x: number, y: number): number {
    const { texel, depth } = this.projectToLightSpace(x, y);
    const idx = this.texelIndex(texel);
    if (idx < 0) return 1; // outside shadow map → lit

    const storedDepth = this.depthMap[idx];
    return depth > storedDepth + this.config.bias ? 0 : 1;
  }

  /**
   * PCF (Percentage Closer Filtering) — soft shadow with configurable radius.
   *
   * Takes multiple samples across neighbouring texels and averages the binary
   * shadow results, producing a smooth gradient at shadow edges (penumbra).
   *
   * @param x      World X
   * @param y      World Y
   * @param radius Kernel radius in texels (e.g. 3.0 for a ±3-texel window).
   * @returns      Value in [0, 1] — 0 = full shadow, 1 = full light.
   */
  sampleShadowPCF(x: number, y: number, radius: number): number {
    const { texel, depth } = this.projectToLightSpace(x, y);
    const { pcfSamples, bias } = this.config;
    const { resolution, depthMap } = this;

    // PCF kernel: spread samples evenly across [-radius, +radius]
    const halfN = (pcfSamples - 1) / 2;
    let litSum = 0;

    for (let s = 0; s < pcfSamples; s++) {
      const offset = pcfSamples === 1 ? 0 : ((s - halfN) / halfN) * radius;
      const sampleTexel = texel + offset;
      const idx = this.texelIndex(sampleTexel);
      if (idx < 0) {
        litSum += 1; // outside map → lit
        continue;
      }
      litSum += depth > depthMap[idx] + bias ? 0 : 1;
    }

    return litSum / pcfSamples;
  }

  /**
   * Extended PCF with thickness-aware attenuation.
   *
   * Thin occluders let some light bleed through based on the entry-exit
   * thickness stored in exitMap. This prevents razor-thin rigid bodies from
   * casting unrealistically dark shadows.
   *
   * @param x         World X
   * @param y         World Y
   * @param radius    PCF kernel radius in texels
   * @param thinLimit Maximum occluder thickness (world units) below which
   *                  shadow intensity is reduced. @default 0.1
   * @returns Value in [0, 1].
   */
  sampleShadowPCFThickness(
    x: number, y: number,
    radius: number,
    thinLimit: number = 0.1,
  ): number {
    const { texel, depth } = this.projectToLightSpace(x, y);
    const { pcfSamples, bias } = this.config;
    const { resolution, depthMap, exitMap } = this;

    const halfN = (pcfSamples - 1) / 2;
    let litSum = 0;

    for (let s = 0; s < pcfSamples; s++) {
      const offset = pcfSamples === 1 ? 0 : ((s - halfN) / halfN) * radius;
      const sampleTexel = texel + offset;
      const idx = this.texelIndex(sampleTexel);

      if (idx < 0) {
        litSum += 1;
        continue;
      }

      const entryD = depthMap[idx];
      const exitD  = exitMap[idx];

      if (depth <= entryD + bias) {
        // Point is in front of the occluder → lit
        litSum += 1;
      } else {
        // In shadow — but thin objects attenuate
        const thickness = exitD - entryD;
        if (thickness < thinLimit && thickness > 0) {
          litSum += 1 - (thickness / thinLimit);
        }
        // else: fully shadowed → litSum += 0
      }
    }

    return litSum / pcfSamples;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Phase 3 — Canvas2D Composite
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Apply shadows to an existing Canvas2D rendering.
   *
   * Reads the current framebuffer via getImageData, computes per-pixel shadow
   * factors using PCF, darkens shadowed pixels, and writes back via putImageData.
   *
   * For performance, the shadow is computed at quarter resolution and bilinearly
   * upsampled before compositing (configurable via `downsample`).
   *
   * @param ctx        Canvas2D rendering context (must have content already drawn)
   * @param width      Canvas width in pixels
   * @param height     Canvas height in pixels
   * @param downsample Downsampling factor for shadow computation. @default 2
   */
  applyShadows(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    downsample: number = 2,
  ): void {
    const ds = Math.max(1, downsample | 0);
    const sw = Math.ceil(width / ds);
    const sh = Math.ceil(height / ds);
    const totalShadowPixels = sw * sh;

    // Allocate or resize shadow buffer
    if (!this.shadowBuffer || this.shadowBuffer.length < totalShadowPixels) {
      this.shadowBuffer = new Float32Array(totalShadowPixels);
    }
    const shadow = this.shadowBuffer;

    const { darkness, aoTint, aoIntensity } = this.config;
    const pcfRadius = 3.0; // texels

    // ── Compute shadow map at reduced resolution ──────────────────────────
    const invW = (this.worldMaxX - this.worldMinX) / width;
    const invH = (this.worldMaxY - this.worldMinY) / height;

    for (let sy = 0; sy < sh; sy++) {
      const wy = this.worldMinY + (sy * ds + ds * 0.5) * invH;
      for (let sx = 0; sx < sw; sx++) {
        const wx = this.worldMinX + (sx * ds + ds * 0.5) * invW;
        shadow[sy * sw + sx] = this.sampleShadowPCF(wx, wy, pcfRadius);
      }
    }

    // ── Read framebuffer ──────────────────────────────────────────────────
    const imageData = ctx.getImageData(0, 0, width, height);
    const pixels = imageData.data;

    // ── Composite: darken shadowed pixels with AO tint ────────────────────
    for (let py = 0; py < height; py++) {
      // Shadow-buffer Y (clamped)
      const sy0 = Math.min(sh - 1, (py / ds) | 0);
      const sy1 = Math.min(sh - 1, sy0 + 1);
      const fy  = (py / ds) - sy0;

      for (let px = 0; px < width; px++) {
        // Bilinear fetch from shadow buffer
        const sx0 = Math.min(sw - 1, (px / ds) | 0);
        const sx1 = Math.min(sw - 1, sx0 + 1);
        const fx  = (px / ds) - sx0;

        const s00 = shadow[sy0 * sw + sx0];
        const s10 = shadow[sy0 * sw + sx1];
        const s01 = shadow[sy1 * sw + sx0];
        const s11 = shadow[sy1 * sw + sx1];

        const sMix = s00 * (1 - fx) * (1 - fy)
                   + s10 *      fx  * (1 - fy)
                   + s01 * (1 - fx) *      fy
                   + s11 *      fx  *      fy;

        // Shadow multiplier: 1.0 in full light, `darkness` in full shadow
        const shadowMul = darkness + (1 - darkness) * sMix;

        const idx = (py * width + px) * 4;
        let r = pixels[idx];
        let g = pixels[idx + 1];
        let b = pixels[idx + 2];
        // alpha stays unchanged

        // Apply darkening
        r *= shadowMul;
        g *= shadowMul;
        b *= shadowMul;

        // Apply AO tint in shadow (subtle cool shift)
        const aoFactor = (1 - sMix) * aoIntensity;
        if (aoFactor > 0.001) {
          r = r * (1 - aoFactor) + r * aoTint[0] * aoFactor;
          g = g * (1 - aoFactor) + g * aoTint[1] * aoFactor;
          b = b * (1 - aoFactor) + b * aoTint[2] * aoFactor;
        }

        pixels[idx]     = Math.min(255, r + 0.5) | 0;
        pixels[idx + 1] = Math.min(255, g + 0.5) | 0;
        pixels[idx + 2] = Math.min(255, b + 0.5) | 0;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal — light-space projection
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Compute the light-space bounding box from the world bounds.
   * Sets tangentOrigin, tangentSpan, and depthOrigin so the shadow map
   * covers the entire scene.
   */
  private computeLightSpaceBounds(): void {
    // Project all 4 corners of the world AABB onto the tangent and depth axes
    const corners: [number, number][] = [
      [this.worldMinX, this.worldMinY],
      [this.worldMaxX, this.worldMinY],
      [this.worldMinX, this.worldMaxY],
      [this.worldMaxX, this.worldMaxY],
    ];

    let tMin = Infinity,  tMax = -Infinity;
    let dMin = Infinity,  dMax = -Infinity;

    for (const [cx, cy] of corners) {
      const t = cx * this.tangent[0]  + cy * this.tangent[1];
      const d = cx * this.lightDir[0] + cy * this.lightDir[1];
      if (t < tMin) tMin = t;
      if (t > tMax) tMax = t;
      if (d < dMin) dMin = d;
      if (d > dMax) dMax = d;
    }

    // Add a small margin to avoid edge clipping
    const margin = (tMax - tMin) * 0.02;
    this.tangentOrigin = tMin - margin;
    this.tangentSpan   = (tMax - tMin) + 2 * margin;
    this.depthOrigin   = dMin - margin;

    // Update farPlane if set to auto
    if (!Number.isFinite(this.config.farPlane)) {
      this.config.farPlane = (dMax - dMin) + 2 * margin;
    }
  }

  /** Compute the world diagonal (for far plane default). */
  private computeDiagonal(): number {
    const dx = this.worldMaxX - this.worldMinX;
    const dy = this.worldMaxY - this.worldMinY;
    return Math.sqrt(dx * dx + dy * dy);
  }

  /**
   * Project a world-space point into light space.
   * Returns the tangent-axis texel coordinate and the depth along lightDir.
   */
  private projectToLightSpace(x: number, y: number): { texel: number; depth: number } {
    const t = x * this.tangent[0]  + y * this.tangent[1];
    const d = x * this.lightDir[0] + y * this.lightDir[1];

    // Convert tangent position to texel coordinate
    const texel = ((t - this.tangentOrigin) / this.tangentSpan) * this.resolution;
    const depth = d - this.depthOrigin;

    return { texel, depth };
  }

  /**
   * Convert a floating-point texel coordinate to a clamped integer index.
   * Returns -1 if the texel is out of range.
   */
  private texelIndex(texel: number): number {
    const i = texel | 0;
    if (i < 0 || i >= this.resolution) return -1;
    return i;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal — exit depth helpers
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Compute the exit depth for a rigid body hit.
   * We cast from beyond the body back toward the light to find the far edge.
   */
  private computeExitDepth(
    ox: number, oy: number,
    dx: number, dy: number,
    body: RigidBody,
    entryT: number,
  ): number {
    // Start from well past the body and cast in the reverse direction
    const farT = entryT + (body.w + body.h) * 4; // generous overshoot
    const rox = ox + dx * farT;
    const roy = oy + dy * farT;
    const exitBackT = rayRigidBodyT(rox, roy, -dx, -dy, body);

    if (Number.isFinite(exitBackT)) {
      return farT - exitBackT;
    }
    // Fallback: estimate using body extent
    return entryT + Math.max(body.w, body.h) * 2;
  }

  /**
   * Compute the far intersection of a ray with a circle (exit point).
   */
  private circleExitT(
    ox: number, oy: number,
    dx: number, dy: number,
    obs: ObstacleData,
  ): number {
    const fx = ox - obs.cx;
    const fy = oy - obs.cy;
    const a = dx * dx + dy * dy;
    const b = 2 * (fx * dx + fy * dy);
    const c = fx * fx + fy * fy - obs.r * obs.r;
    let disc = b * b - 4 * a * c;
    if (disc < 0) return Infinity;
    disc = Math.sqrt(disc);
    const t1 = (-b + disc) / (2 * a); // far intersection
    return t1 >= 0 ? t1 : Infinity;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Diagnostics
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Render a visual representation of the shadow map onto a Canvas2D context.
   * Useful for debugging — draws a horizontal bar showing depth values.
   *
   * @param ctx    Canvas2D context
   * @param x      Left edge of the debug bar
   * @param y      Top edge of the debug bar
   * @param width  Bar width in pixels
   * @param height Bar height in pixels
   */
  debugDrawShadowMap(
    ctx: CanvasRenderingContext2D,
    x: number, y: number,
    width: number, height: number,
  ): void {
    const { resolution, depthMap } = this;
    const farPlane = Number.isFinite(this.config.farPlane)
      ? this.config.farPlane
      : this.computeDiagonal();

    const texelW = width / resolution;

    ctx.save();
    for (let i = 0; i < resolution; i++) {
      const d = depthMap[i];
      // Normalize depth to [0, 1]
      const norm = Number.isFinite(d) ? Math.min(1, d / farPlane) : 1;
      const brightness = Math.round((1 - norm) * 255);
      ctx.fillStyle = `rgb(${brightness},${brightness},${brightness})`;
      ctx.fillRect(x + i * texelW, y, Math.ceil(texelW), height);
    }

    // Border
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);

    // Label
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '10px monospace';
    ctx.fillText('shadow depth map', x + 4, y + height - 4);
    ctx.restore();
  }

  /**
   * Render a 2D shadow overlay showing lit/shadowed regions.
   * Draws semi-transparent black rectangles for shadowed areas.
   *
   * @param ctx    Canvas2D context
   * @param width  Canvas width
   * @param height Canvas height
   * @param step   Sample step in pixels (higher = faster but blockier). @default 4
   */
  debugDrawShadowOverlay(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    step: number = 4,
  ): void {
    const { darkness } = this.config;
    const invW = (this.worldMaxX - this.worldMinX) / width;
    const invH = (this.worldMaxY - this.worldMinY) / height;

    ctx.save();
    for (let py = 0; py < height; py += step) {
      const wy = this.worldMinY + (py + step * 0.5) * invH;
      for (let px = 0; px < width; px += step) {
        const wx = this.worldMinX + (px + step * 0.5) * invW;
        const lit = this.sampleShadowPCF(wx, wy, 2.0);
        const alpha = (1 - lit) * (1 - darkness);
        if (alpha > 0.01) {
          ctx.fillStyle = `rgba(0,0,20,${alpha.toFixed(3)})`;
          ctx.fillRect(px, py, step, step);
        }
      }
    }
    ctx.restore();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Read-only accessors
  // ─────────────────────────────────────────────────────────────────────────

  /** Current shadow map resolution. */
  getResolution(): number { return this.resolution; }

  /** Current light direction (normalized, read-only copy). */
  getLightDirection(): [number, number] {
    return [this.lightDir[0], this.lightDir[1]];
  }

  /** Current tangent axis (normalized, read-only copy). */
  getTangent(): [number, number] {
    return [this.tangent[0], this.tangent[1]];
  }

  /** Raw depth map (read-only view). */
  getDepthMap(): Float32Array { return this.depthMap; }

  /** Current shadow configuration (read-only copy). */
  getConfig(): Readonly<ShadowConfig> { return { ...this.config }; }
}
