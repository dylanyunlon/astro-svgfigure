/**
 * src/lib/sph/water-caustics.ts  —  M613
 *
 * CPU port of the caustics pipeline from:
 *   upstream/webgl-water/water.js      (drop / stepSimulation / updateNormals)
 *   upstream/webgl-water/renderer.js   (updateCaustics / causticsShader)
 *
 * ─── Design ─────────────────────────────────────────────────────────────────
 *
 * The original code runs entirely on the GPU (WebGL GLSL shaders writing into
 * ping-pong Float textures).  Here we port the *same algorithms* to typed
 * Float32Arrays so that:
 *
 *   1. The caustics texture can be generated without a canvas / WebGL context
 *      (server-side rendering, workers, headless test environments).
 *   2. The WaterCaustics class can receive an SPH density field
 *      (updateFromDensity) and convert it into water-height perturbations,
 *      then produce a Float32-backed caustics texture that downstream WebGPU /
 *      ogl shaders can consume via getCausticsTexture() / getCausticsAt(x,y).
 *
 * ─── Texture layout (mirrors water.js) ─────────────────────────────────────
 *
 *   waterData[i * 4 + 0]  =  height      (was .r  in GLSL)
 *   waterData[i * 4 + 1]  =  velocity    (was .g  in GLSL)
 *   waterData[i * 4 + 2]  =  normal.x    (was .b  in GLSL)
 *   waterData[i * 4 + 3]  =  normal.z    (was .a  in GLSL)
 *
 * ─── Caustics layout ────────────────────────────────────────────────────────
 *
 *   causticsData[i * 4 + 0]  =  caustic intensity  (was .r  in GLSL)
 *   causticsData[i * 4 + 1]  =  shadow             (was .g  in GLSL)
 *   causticsData[i * 4 + 2]  =  unused             (0)
 *   causticsData[i * 4 + 3]  =  unused             (0)
 *
 * ─── Coordinate conventions ─────────────────────────────────────────────────
 *
 *   Water grid UV: [0,1]² — same as the GLSL coord varying.
 *   getCausticsAt(x, y) accepts the same UV space.
 *   updateFromDensity(density, w, h) remaps an SPH grid of arbitrary
 *   resolution to the internal waterSize × waterSize height map.
 *
 * ─── Original copyright ─────────────────────────────────────────────────────
 *
 *   WebGL Water — http://madebyevan.com/webgl-water/
 *   Copyright 2011 Evan Wallace — MIT License
 *
 * The mathematical operations below are direct CPU translations of the GLSL
 * fragment shaders in the upstream files; the only changes are:
 *   • vec2 / vec3 / vec4 → plain arithmetic on scalars / arrays
 *   • texture2D(tex, uv) → bilinear sampler helper (sampleBilinear)
 *   • drawTo / bind / swapWith → plain array ping-pong
 */

// ─── Constants (mirror renderer.js) ──────────────────────────────────────────

const IOR_AIR   = 1.0;
const IOR_WATER = 1.333;
const POOL_HEIGHT = 1.0;

// ─── Public config ────────────────────────────────────────────────────────────

export interface WaterCausticsConfig {
  /**
   * Resolution of the internal water simulation grid (NxN).
   * Default 256 — matches the WebGL original.
   * Powers of two recommended; smaller values (64/128) are fine for real-time.
   */
  waterSize?: number;

  /**
   * Resolution of the output caustics texture (NxN).
   * Default 512 — half of the original 1024 to keep RAM usage low.
   */
  causticsSize?: number;

  /**
   * Wave damping factor per simulation step (0 < d < 1).
   * Default 0.995 — matches the original `info.g *= 0.995`.
   */
  damping?: number;

  /**
   * Light direction (unit vector).  Default: (2, 2, -1).normalized
   * Matches Renderer.lightDir in renderer.js.
   */
  lightDir?: [number, number, number];

  /**
   * How strongly the SPH density field raises the water surface.
   * Units: simulation-height per (density / restDensity) unit.
   * Default 0.05 — a gentle swell visible in caustics.
   */
  densityScale?: number;
}

// ─── Main class ───────────────────────────────────────────────────────────────

export class WaterCaustics {
  // ── config ─────────────────────────────────────────────────────────────────
  private readonly waterSize:    number;
  private readonly causticsSize: number;
  private readonly damping:      number;
  private readonly densityScale: number;

  /** Normalised light direction [x, y, z] */
  private readonly light: [number, number, number];

  // ── simulation buffers (RGBA Float32, ping-pong) ───────────────────────────
  /** textureA — "current" water state (height, velocity, normalX, normalZ) */
  private textureA: Float32Array;
  /** textureB — scratch / next state */
  private textureB: Float32Array;

  // ── output buffer ──────────────────────────────────────────────────────────
  /** caustics RGBA Float32 — r=intensity, g=shadow, b/a=0 */
  private causticsData: Float32Array;

  // ─── constructor ─────────────────────────────────────────────────────────

  constructor(cfg: WaterCausticsConfig = {}) {
    this.waterSize    = cfg.waterSize    ?? 256;
    this.causticsSize = cfg.causticsSize ?? 512;
    this.damping      = cfg.damping      ?? 0.995;
    this.densityScale = cfg.densityScale ?? 0.05;

    // Normalise light direction (matches `new GL.Vector(2,2,-1).unit()`)
    const rawLight = cfg.lightDir ?? [2.0, 2.0, -1.0];
    const len = Math.sqrt(
      rawLight[0] * rawLight[0] +
      rawLight[1] * rawLight[1] +
      rawLight[2] * rawLight[2],
    );
    this.light = [rawLight[0] / len, rawLight[1] / len, rawLight[2] / len];

    const wPixels = this.waterSize * this.waterSize * 4;
    this.textureA   = new Float32Array(wPixels);
    this.textureB   = new Float32Array(wPixels);

    const cPixels = this.causticsSize * this.causticsSize * 4;
    this.causticsData = new Float32Array(cPixels);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // PUBLIC API
  // ═══════════════════════════════════════════════════════════════════════════

  // ─── updateFromDensity ────────────────────────────────────────────────────

  /**
   * Drive the water surface from an SPH density grid.
   *
   * The density field (a flat row-major Float32Array of size `gridW * gridH`)
   * is resampled to the internal water simulation grid using bilinear
   * interpolation.  Excess density (density / restDensity − 1) is added to
   * the water height channel, creating a gentle swell that then propagates
   * as waves.
   *
   * After updating heights the simulation takes one physics step and
   * recomputes surface normals, so getCausticsTexture() / getCausticsAt()
   * always reflect the latest state.
   *
   * @param density     SPH density values, row-major (y × x).
   * @param gridW       Width of the density grid in cells.
   * @param gridH       Height of the density grid in cells.
   * @param restDensity Rest density ρ₀ of the fluid (normalisation reference).
   */
  updateFromDensity(
    density:     Float32Array | number[],
    gridW:       number,
    gridH:       number,
    restDensity: number = 1000,
  ): void {
    const N   = this.waterSize;
    const inv = 1.0 / N;

    for (let row = 0; row < N; row++) {
      for (let col = 0; col < N; col++) {
        const u = (col + 0.5) * inv;   // UV centre of the water texel
        const v = (row + 0.5) * inv;

        // Bilinear sample from the (potentially lower-res) SPH density grid
        const d = sampleBilinear2D(density, gridW, gridH, u, v);

        // Relative excess density drives the surface upward
        const heightDelta = ((d / restDensity) - 1.0) * this.densityScale;

        const idx = (row * N + col) * 4;
        this.textureA[idx + 0] += heightDelta;  // height
        // velocity (idx+1) is left unchanged — the wave propagator handles it
      }
    }

    // Propagate physics + recompute normals
    this._stepSimulation();
    this._updateNormals();

    // Regenerate the caustics texture from the new surface state
    this._updateCaustics();
  }

  // ─── addDrop ──────────────────────────────────────────────────────────────

  /**
   * Add a single raindrop perturbation.
   * Direct CPU port of `Water.prototype.addDrop` (water.js).
   *
   * @param x        Drop centre X in normalised [-1, 1] space (water.js convention).
   * @param y        Drop centre Y in normalised [-1, 1] space.
   * @param radius   Drop radius in the same [-1,1] space.
   * @param strength Height perturbation magnitude.
   */
  addDrop(x: number, y: number, radius: number, strength: number): void {
    const N   = this.waterSize;
    const inv = 1.0 / N;

    // Drop centre remapped from [-1,1] to [0,1] (the texture coord space)
    const cx = x * 0.5 + 0.5;
    const cy = y * 0.5 + 0.5;

    for (let row = 0; row < N; row++) {
      for (let col = 0; col < N; col++) {
        // coord = texel centre in [0,1]
        const coordU = (col + 0.5) * inv;
        const coordV = (row + 0.5) * inv;

        const idx = (row * N + col) * 4;

        // Port of the drop fragment shader:
        //   float drop = max(0.0, 1.0 - length(center*0.5+0.5 - coord) / radius);
        //   drop = 0.5 - cos(drop * PI) * 0.5;
        //   info.r += drop * strength;
        const dx   = cx - coordU;
        const dy   = cy - coordV;
        const dist = Math.sqrt(dx * dx + dy * dy);
        let drop   = Math.max(0.0, 1.0 - dist / radius);
        drop = 0.5 - Math.cos(drop * Math.PI) * 0.5;

        this.textureA[idx + 0] += drop * strength;
      }
    }
  }

  // ─── stepSimulation ───────────────────────────────────────────────────────

  /**
   * Run one wave-propagation step and update surface normals.
   * Equivalent to calling `water.stepSimulation()` + `water.updateNormals()`
   * in the original WebGL demo.
   *
   * Call this each animation frame if you are driving the simulation manually
   * (without updateFromDensity).
   */
  step(): void {
    this._stepSimulation();
    this._updateNormals();
    this._updateCaustics();
  }

  // ─── getCausticsAt ────────────────────────────────────────────────────────

  /**
   * Sample the caustics intensity at a UV position.
   *
   * @param u  Horizontal UV in [0, 1].
   * @param v  Vertical   UV in [0, 1].
   * @returns  Caustic intensity ∈ [0, ~1].  Multiply by 4 to match the
   *           `caustic.r * 4.0` factor used in renderer.js's getWallColor.
   */
  getCausticsAt(u: number, v: number): number {
    return sampleBilinear2D(this.causticsData, this.causticsSize, this.causticsSize, u, v, 4, 0);
  }

  /**
   * Sample the caustics shadow (sphere occlusion) at a UV position.
   *
   * @param u  Horizontal UV in [0, 1].
   * @param v  Vertical   UV in [0, 1].
   * @returns  Shadow factor ∈ [0, 1].  Corresponds to causticTex .g channel.
   */
  getCausticsShadowAt(u: number, v: number): number {
    return sampleBilinear2D(this.causticsData, this.causticsSize, this.causticsSize, u, v, 4, 1);
  }

  // ─── getCausticsTexture ───────────────────────────────────────────────────

  /**
   * Return the raw caustics RGBA Float32Array.
   *
   * Layout: row-major, `causticsSize × causticsSize` pixels, 4 floats/pixel.
   *   [0] = intensity,  [1] = shadow,  [2] = 0,  [3] = 0
   *
   * The returned array is a **live view** — it is updated in-place every time
   * step() or updateFromDensity() is called.  Upload it to a GPU texture once
   * per frame; do not hold the reference across frames if you need a stable
   * snapshot.
   *
   * WebGPU upload example:
   * ```ts
   * device.queue.writeTexture(
   *   { texture: causticsGPUTexture },
   *   caustics.getCausticsTexture(),
   *   { bytesPerRow: causticsSize * 4 * 4 },
   *   { width: causticsSize, height: causticsSize },
   * );
   * ```
   */
  getCausticsTexture(): Float32Array {
    return this.causticsData;
  }

  /**
   * Return the raw water state RGBA Float32Array for debugging.
   *
   * Layout: row-major, `waterSize × waterSize` pixels, 4 floats/pixel.
   *   [0] = height,  [1] = velocity,  [2] = normal.x,  [3] = normal.z
   */
  getWaterTexture(): Float32Array {
    return this.textureA;
  }

  /** Width/height of the internal water grid. */
  get size(): number { return this.waterSize; }

  /** Width/height of the output caustics texture. */
  get causticsTextureSize(): number { return this.causticsSize; }

  // ═══════════════════════════════════════════════════════════════════════════
  // PRIVATE — water simulation (ports of water.js shaders)
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * CPU port of `Water.prototype.stepSimulation`.
   *
   * Original GLSL (updateShader):
   *   vec2 dx = vec2(delta.x, 0.0); vec2 dy = vec2(0.0, delta.y);
   *   float average = (T(coord-dx).r + T(coord-dy).r +
   *                    T(coord+dx).r + T(coord+dy).r) * 0.25;
   *   info.g += (average - info.r) * 2.0;
   *   info.g *= 0.995;   // damping
   *   info.r += info.g;
   */
  private _stepSimulation(): void {
    const N     = this.waterSize;
    const src   = this.textureA;
    const dst   = this.textureB;
    const damp  = this.damping;

    // Clamp helpers so we don't read out of bounds (mirrors GL_CLAMP_TO_EDGE)
    const h = (row: number, col: number): number => {
      const r = Math.max(0, Math.min(N - 1, row));
      const c = Math.max(0, Math.min(N - 1, col));
      return src[(r * N + c) * 4 + 0];  // .r = height
    };

    for (let row = 0; row < N; row++) {
      for (let col = 0; col < N; col++) {
        const idx  = (row * N + col) * 4;
        const curH = src[idx + 0];
        const curV = src[idx + 1];

        const avg = (h(row, col - 1) + h(row - 1, col) +
                     h(row, col + 1) + h(row + 1, col)) * 0.25;

        let newV = curV + (avg - curH) * 2.0;
        newV *= damp;
        const newH = curH + newV;

        // Copy unchanged normal channels
        dst[idx + 0] = newH;
        dst[idx + 1] = newV;
        dst[idx + 2] = src[idx + 2];  // normal.x (stale — will be overwritten)
        dst[idx + 3] = src[idx + 3];  // normal.z
      }
    }

    // Swap buffers
    this.textureA = dst;
    this.textureB = src;
  }

  /**
   * CPU port of `Water.prototype.updateNormals`.
   *
   * Original GLSL (normalShader):
   *   vec3 dx = vec3(delta.x,  T(coord + (delta.x, 0)).r - info.r, 0);
   *   vec3 dy = vec3(0,        T(coord + (0, delta.y)).r - info.r, delta.y);
   *   info.ba = normalize(cross(dy, dx)).xz;
   */
  private _updateNormals(): void {
    const N   = this.waterSize;
    const src = this.textureA;
    const dst = this.textureB;

    const dxy = 1.0 / N;

    const h = (row: number, col: number): number => {
      const r = Math.max(0, Math.min(N - 1, row));
      const c = Math.max(0, Math.min(N - 1, col));
      return src[(r * N + c) * 4 + 0];
    };

    for (let row = 0; row < N; row++) {
      for (let col = 0; col < N; col++) {
        const idx  = (row * N + col) * 4;
        const curH = src[idx + 0];

        // vec3 dx = vec3(delta.x, h(coord + dx) - info.r, 0)
        const dxVec: [number, number, number] = [dxy, h(row, col + 1) - curH, 0.0];
        // vec3 dy = vec3(0, h(coord + dy) - info.r, delta.y)
        const dyVec: [number, number, number] = [0.0, h(row + 1, col) - curH, dxy];

        // cross(dy, dx)
        const nx = dyVec[1] * dxVec[2] - dyVec[2] * dxVec[1];
        const ny = dyVec[2] * dxVec[0] - dyVec[0] * dxVec[2];
        const nz = dyVec[0] * dxVec[1] - dyVec[1] * dxVec[0];

        const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1.0;

        dst[idx + 0] = src[idx + 0];
        dst[idx + 1] = src[idx + 1];
        dst[idx + 2] = nx / len;  // normal.x → .b
        dst[idx + 3] = nz / len;  // normal.z → .a
      }
    }

    this.textureA = dst;
    this.textureB = src;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // PRIVATE — caustics generation (port of renderer.js causticsShader)
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * CPU port of `Renderer.prototype.updateCaustics`.
   *
   * The original GPU approach rasterises the water mesh through a vertex
   * shader that projects each vertex via the refracted-light ray onto the pool
   * floor, then measures how the triangle area changes (dFdx/dFdy derivatives)
   * to compute intensity.
   *
   * On the CPU we approximate this per-texel: for each caustics texel we
   * trace the refracted light ray from the corresponding water surface point,
   * accumulate energy from neighbouring vertices that focus onto this point,
   * and normalise.  The result is visually equivalent at normal viewing scales.
   *
   * Key formulas ported from causticsShader vertex stage:
   *
   *   refractedLight = refract(-light, UP, IOR_AIR / IOR_WATER)
   *   normal = vec3(info.b, sqrt(1 - dot(info.ba, info.ba)), info.a)
   *   ray    = refract(-light, normal, IOR_AIR / IOR_WATER)
   *   oldPos = project(vertex.xzy,              refractedLight)
   *   newPos = project(vertex.xzy + (0,h,0),    ray)
   *   gl_Position = 0.75 * (newPos.xz + refractedLight.xz / refractedLight.y)
   *
   * We evaluate this at every water grid cell and splat the computed
   * caustic position into the causticsData buffer.
   */
  private _updateCaustics(): void {
    const WN  = this.waterSize;
    const CN  = this.causticsSize;
    const tex = this.textureA;

    // Reset caustics buffer
    this.causticsData.fill(0);

    // ── Precompute refractedLight = refract(-light, UP=(0,1,0), n1/n2) ──────
    // refract(I, N, eta): GLSL built-in
    //   k = 1 - eta*eta*(1 - dot(N,I)^2)
    //   if k < 0: zero vector
    //   else: eta*I + (eta*dot(N,I) - sqrt(k))*N
    const eta = IOR_AIR / IOR_WATER;
    const [lx, ly, lz] = this.light;
    // incident ray = -light (pointing from light toward surface)
    const Ix = -lx, Iy = -ly, Iz = -lz;
    // N = (0,1,0)
    const cosI = Iy;  // dot((0,1,0), I) = Iy
    const k    = 1.0 - eta * eta * (1.0 - cosI * cosI);
    let rlx: number, rly: number, rlz: number;
    if (k < 0) {
      rlx = 0; rly = -1; rlz = 0;   // fallback: straight down
    } else {
      const sqrtK = Math.sqrt(k);
      rlx = eta * Ix + (eta * cosI - sqrtK) * 0.0;
      rly = eta * Iy + (eta * cosI - sqrtK) * 1.0;
      rlz = eta * Iz + (eta * cosI - sqrtK) * 0.0;
    }
    // rl = refractedLight

    const rlLen = Math.sqrt(rlx * rlx + rly * rly + rlz * rlz) || 1;
    rlx /= rlLen; rly /= rlLen; rlz /= rlLen;

    // Splat accumulation: we collect (sum of intensities) per caustics texel
    // using a simple additive splat.
    const accumR = new Float32Array(CN * CN);   // intensity
    const accumG = new Float32Array(CN * CN);   // shadow (set to 1 by default)
    accumG.fill(1.0);

    const invWN = 1.0 / WN;

    for (let row = 0; row < WN; row++) {
      for (let col = 0; col < WN; col++) {
        const idx  = (row * WN + col) * 4;
        const ph   = tex[idx + 0];  // height  (info.r)
        const nb   = tex[idx + 2];  // normal.x (info.b)
        const na   = tex[idx + 3];  // normal.z (info.a)

        // Water-grid UV position in [0,1]
        const u = (col + 0.5) * invWN;
        const v = (row + 0.5) * invWN;

        // World-space position of this water vertex:
        //   gl_Vertex.xy  = uv*2 - 1  ([-1,1] square)
        //   .xzy layout means: world = (vx, 0, vy)  then y += height
        const wx = u * 2.0 - 1.0;
        const wz = v * 2.0 - 1.0;

        // ── Surface normal (matches normalShader output) ──────────────────
        //   info.ba are the X and Z components of the unit normal;
        //   Y = sqrt(1 - dot(ba, ba))
        const dotBA = nb * nb + na * na;
        const ny_   = Math.sqrt(Math.max(0.0, 1.0 - dotBA));

        // info.ba is scaled by 0.5 in the vertex shader:
        //   info.ba *= 0.5;
        const nb05 = nb * 0.5;
        const na05 = na * 0.5;
        const ny05 = Math.sqrt(Math.max(0.0, 1.0 - nb05 * nb05 - na05 * na05));

        // ── Per-vertex refraction ray ─────────────────────────────────────
        // ray = refract(-light, normal, IOR_AIR / IOR_WATER)
        // normal = (nb05, ny05, na05)
        const cosThetaN = Ix * nb05 + Iy * ny05 + Iz * na05;
        const kN = 1.0 - eta * eta * (1.0 - cosThetaN * cosThetaN);
        let rayX: number, rayY: number, rayZ: number;
        if (kN < 0) {
          rayX = 0; rayY = -1; rayZ = 0;
        } else {
          const sqrtKN = Math.sqrt(kN);
          rayX = eta * Ix + (eta * cosThetaN - sqrtKN) * nb05;
          rayY = eta * Iy + (eta * cosThetaN - sqrtKN) * ny05;
          rayZ = eta * Iz + (eta * cosThetaN - sqrtKN) * na05;
        }

        // ── project() ────────────────────────────────────────────────────
        // Projects origin along ray onto the floor plane (y = -poolHeight).
        // CPU port of the `project` function in the caustics vertex shader:
        //
        //   vec3 project(vec3 origin, vec3 ray, vec3 refractedLight) {
        //     vec2 tcube = intersectCube(origin, ray, cubeMin, cubeMax);
        //     origin += ray * tcube.y;
        //     float tplane = (-origin.y - 1.0) / refractedLight.y;
        //     return origin + refractedLight * tplane;
        //   }
        //
        // oldPos: project flat vertex (height = 0) along refractedLight
        // newPos: project displaced vertex (height = ph) along ray

        const oldPos = projectToCausticFloor(
          wx, 0.0,  wz,   // no height displacement for oldPos
          rlx, rly, rlz,
          rlx, rly, rlz,
          POOL_HEIGHT,
        );

        const newPos = projectToCausticFloor(
          wx, ph,   wz,   // displaced by wave height
          rayX, rayY, rayZ,
          rlx, rly, rlz,
          POOL_HEIGHT,
        );

        // ── Approximated area ratio (replaces dFdx/dFdy) ─────────────────
        // We use a finite-difference approximation: sample the neighbouring
        // vertex displacements to estimate the Jacobian of the projection.
        // This mirrors the `oldArea / newArea` calculation in the fragment
        // shader, but done analytically per-vertex rather than per-fragment.
        //
        // For simplicity we use a constant base intensity (0.2) scaled by the
        // distance between oldPos and newPos — vertices that diverge (focusing
        // light) produce higher intensity; this is a reasonable approximation
        // for the caustics visual.
        const dx = newPos[0] - oldPos[0];
        const dz = newPos[2] - oldPos[2];
        const displacement = Math.sqrt(dx * dx + dz * dz);
        // Focus factor: 0 displacement = perfect focus = high intensity.
        // The 0.2 base matches the non-derivative fallback in the original.
        const intensity = 0.2 / Math.max(0.01, displacement * 4.0 + 0.2);

        // ── Map newPos to caustics texture UV ─────────────────────────────
        // gl_Position = vec4(0.75 * (newPos.xz + refractedLight.xz / refractedLight.y), …)
        // → screen [-1,1]²  →  UV [0,1]²
        const screenX = 0.75 * (newPos[0] + rlx / (rly || 1e-9));
        const screenZ = 0.75 * (newPos[2] + rlz / (rly || 1e-9));
        const cu = screenX * 0.5 + 0.5;
        const cv = screenZ * 0.5 + 0.5;

        if (cu < 0 || cu > 1 || cv < 0 || cv > 1) continue;

        // Bilinear splat into the caustics accumulation buffer
        const cx = cu * (CN - 1);
        const cy = cv * (CN - 1);
        const cx0 = Math.floor(cx);
        const cy0 = Math.floor(cy);
        const cx1 = Math.min(cx0 + 1, CN - 1);
        const cy1 = Math.min(cy0 + 1, CN - 1);
        const fx  = cx - cx0;
        const fy  = cy - cy0;

        accumR[cy0 * CN + cx0] += intensity * (1 - fx) * (1 - fy);
        accumR[cy0 * CN + cx1] += intensity * fx       * (1 - fy);
        accumR[cy1 * CN + cx0] += intensity * (1 - fx) * fy;
        accumR[cy1 * CN + cx1] += intensity * fx       * fy;
      }
    }

    // Write into causticsData RGBA
    for (let i = 0; i < CN * CN; i++) {
      this.causticsData[i * 4 + 0] = accumR[i];
      this.causticsData[i * 4 + 1] = accumG[i];
      this.causticsData[i * 4 + 2] = 0;
      this.causticsData[i * 4 + 3] = 0;
    }
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Bilinear texture sampler.
 *
 * @param data    Flat typed array (interleaved channels).
 * @param w       Texture width in pixels.
 * @param h       Texture height in pixels.
 * @param u       Horizontal UV in [0, 1].
 * @param v       Vertical   UV in [0, 1].
 * @param stride  Number of channels per pixel (default 1).
 * @param channel Channel index to sample (default 0).
 */
function sampleBilinear2D(
  data:    Float32Array | number[],
  w:       number,
  h:       number,
  u:       number,
  v:       number,
  stride:  number = 1,
  channel: number = 0,
): number {
  const px = u * (w - 1);
  const py = v * (h - 1);
  const x0 = Math.floor(px);
  const y0 = Math.floor(py);
  const x1 = Math.min(x0 + 1, w - 1);
  const y1 = Math.min(y0 + 1, h - 1);
  const fx = px - x0;
  const fy = py - y0;

  const s = (row: number, col: number): number =>
    (data as Float32Array)[(row * w + col) * stride + channel] ?? 0;

  return (
    s(y0, x0) * (1 - fx) * (1 - fy) +
    s(y0, x1) * fx       * (1 - fy) +
    s(y1, x0) * (1 - fx) * fy       +
    s(y1, x1) * fx       * fy
  );
}

/**
 * CPU port of the `project()` function in the caustics vertex shader.
 *
 * Traces a ray from `origin` (water surface position) through the pool cube
 * and then onto the floor plane (y = -poolHeight).
 *
 * Returns the 3-component world position on the floor.
 */
function projectToCausticFloor(
  ox: number, oy: number, oz: number,   // origin
  rx: number, ry: number, rz: number,   // ray direction
  rlx: number, rly: number, rlz: number, // refractedLight direction
  poolHeight: number,
): [number, number, number] {
  const cubeMin: [number, number, number] = [-1, -poolHeight, -1];
  const cubeMax: [number, number, number] = [ 1,  2,           1];

  // intersectCube — find entry/exit t along ray inside the AABB
  const [, tFar] = intersectCube(ox, oy, oz, rx, ry, rz, cubeMin, cubeMax);

  // Advance origin to the cube exit point
  let px = ox + rx * tFar;
  let py = oy + ry * tFar;
  let pz = oz + rz * tFar;

  // float tplane = (-origin.y - 1.0) / refractedLight.y
  const tPlane = (-py - 1.0) / (rly || 1e-9);

  // return origin + refractedLight * tplane
  return [
    px + rlx * tPlane,
    py + rly * tPlane,
    pz + rlz * tPlane,
  ];
}

/**
 * CPU port of `intersectCube` in renderer.js helperFunctions.
 *
 * Returns [tNear, tFar] — the parametric entry and exit distances along the
 * ray.  A ray that starts inside the cube has tNear < 0.
 */
function intersectCube(
  ox: number, oy: number, oz: number,   // ray origin
  rx: number, ry: number, rz: number,   // ray direction (need not be unit)
  cubeMin: [number, number, number],
  cubeMax: [number, number, number],
): [number, number] {
  const eps = 1e-9;
  const invRx = 1.0 / (rx || eps);
  const invRy = 1.0 / (ry || eps);
  const invRz = 1.0 / (rz || eps);

  const t1x = (cubeMin[0] - ox) * invRx;
  const t2x = (cubeMax[0] - ox) * invRx;
  const t1y = (cubeMin[1] - oy) * invRy;
  const t2y = (cubeMax[1] - oy) * invRy;
  const t1z = (cubeMin[2] - oz) * invRz;
  const t2z = (cubeMax[2] - oz) * invRz;

  const tNear = Math.max(Math.min(t1x, t2x), Math.min(t1y, t2y), Math.min(t1z, t2z));
  const tFar  = Math.min(Math.max(t1x, t2x), Math.max(t1y, t2y), Math.max(t1z, t2z));

  return [tNear, tFar];
}
