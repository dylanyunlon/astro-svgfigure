/**
 * at-navier-stokes.ts — M715: AT Navier-Stokes Fluid Compute — WGSL Port
 *
 * Ports the Active Theory mouse-fluid / Navier-Stokes grid pipeline to
 * WebGPU compute shaders (WGSL).  The shader vocabulary is reverse-engineered
 * from `src/lib/shaders/compiled.vs` (fluid-surface.frag, curl-trail.frag)
 * and the upstream LYGIA `simulate/simpleAndFastFluid.glsl` reference which
 * AT uses as its fluid base.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Pipeline — five compute passes per frame
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  Pass 0 — SPLAT
 *    Injects a Gaussian force / dye impulse at the mouse position.
 *    Writes to the velocity ping-pong texture.  Mirrors AT "mousefluid"
 *    splat behaviour found in SceneLayoutPresets ("mousefluid_scale").
 *
 *  Pass 1 — ADVECTION  (semi-Lagrangian)
 *    Back-projects every grid cell along its own velocity to find the
 *    value one timestep ago (MacCormack-style bilinear fetch).
 *    Handles both velocity self-advection and dye advection in one shader.
 *
 *  Pass 2 — VORTICITY
 *    Computes 2-D curl  ω = ∂vy/∂x − ∂vx/∂y  and stores it in the
 *    w-channel of the velocity texture (matching lygia vorticity convention).
 *    Then applies vorticity confinement to re-energise small-scale eddies
 *    that numerical diffusion smears out.
 *
 *  Pass 3 — DIVERGENCE
 *    Computes the velocity divergence  div = ∂vx/∂x + ∂vy/∂y  and stores
 *    the result in a dedicated half-float texture for the pressure solver.
 *
 *  Pass 4 — PRESSURE  (Jacobi iteration × N)
 *    Solves the Poisson equation  ∇²p = div  via Jacobi relaxation.
 *    Default 40 iterations (configurable).  Ping-pong pressure textures A↔B.
 *
 *  Pass 5 — GRADIENT SUBTRACT  (projection)
 *    Subtracts the pressure gradient from the velocity field to enforce
 *    incompressibility: v ← v − ∇p.  This is the "divergence-free" step.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Texture layout
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  velTex[2]   rgba16float   W×H   XY=velocity, Z=unused, W=curl ω
 *  dyeTex[2]   rgba16float   W×H   RGB=dye colour, W=density
 *  divTex      r16float      W×H   divergence scalar
 *  preTex[2]   r16float      W×H   pressure (Jacobi ping-pong)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Sources / lineage
 * ─────────────────────────────────────────────────────────────────────────────
 *  • upstream/lygia/simulate/simpleAndFastFluid.glsl  — Patricio Gonzalez Vivo
 *  • src/lib/shaders/compiled.vs :: fluid-surface.frag (M553)
 *  • src/lib/shaders/compiled.vs :: curl-trail.frag
 *  • src/lib/sph/physics-uniform-bridge.ts :: estimateVorticity
 *  • src/lib/sph/lattice-boltzmann-bg.ts   (WebGPU ping-pong pattern)
 *  • src/lib/sph/boids-compute.ts          (WebGPU orchestrator pattern)
 *
 * Research: xiaodi #M715 — cell-pubsub-loop
 * ─────────────────────────────────────────────────────────────────────────────
 */

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Default fluid grid resolution (square). */
export const NS_GRID = 512;

/** Compute workgroup tile side length (16×16 = 256 threads). */
const WG = 16;

/** Default number of Jacobi pressure iterations per frame. */
export const NS_PRESSURE_ITERS_DEFAULT = 40;

/** Velocity texture format — RGBA16Float: XY=vel, Z=unused, W=curl. */
const VEL_FORMAT: GPUTextureFormat  = 'rgba16float';

/** Dye texture format — RGBA16Float: RGB=colour, W=density. */
const DYE_FORMAT: GPUTextureFormat  = 'rgba16float';

/** Divergence / pressure texture format. */
const DIV_FORMAT: GPUTextureFormat  = 'r16float';
const PRE_FORMAT: GPUTextureFormat  = 'r16float';

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

export interface NavierStokesParams {
  /** Grid resolution (square).  Default 512. */
  grid?: number;
  /** Simulation timestep in seconds.  Default 1/60. */
  dt?: number;
  /** Fluid viscosity (higher = more diffusion).  Default 0.16. */
  viscosity?: number;
  /** Vorticity confinement strength (0 = off, 0.3 = strong).  Default 0.15. */
  vorticityStrength?: number;
  /** Velocity dissipation per frame (multiplicative, < 1).  Default 0.999. */
  velocityDissipation?: number;
  /** Dye dissipation per frame (multiplicative, < 1).  Default 0.995. */
  dyeDissipation?: number;
  /** Splat radius in normalised [0,1] UV space.  Default 0.012. */
  splatRadius?: number;
  /** Number of Jacobi pressure iterations.  Default 40. */
  pressureIters?: number;
}

export interface NavierStokesSplat {
  /** Normalised X position [0,1]. */
  x: number;
  /** Normalised Y position [0,1]. */
  y: number;
  /** Velocity impulse X (world units / second). */
  vx: number;
  /** Velocity impulse Y (world units / second). */
  vy: number;
  /** Dye colour RGB [0,1] each. */
  color: [number, number, number];
}

// ─────────────────────────────────────────────────────────────────────────────
// Uniform struct — 16-byte aligned, keep in sync with WGSL
// ─────────────────────────────────────────────────────────────────────────────

// struct NSUniforms {                  offset  size
//   pixel            : vec2f,         0       8
//   dt               : f32,           8       4
//   viscosity        : f32,           12      4
//   velocityDissipation : f32,        16      4
//   dyeDissipation   : f32,           20      4
//   vorticityStrength: f32,           24      4
//   _pad0            : f32,           28      4   → total 32
// }
const UNIFORMS_SIZE = 32; // bytes

// struct SplatUniforms {              offset  size
//   pos    : vec2f,                   0       8
//   vel    : vec2f,                   8       8
//   color  : vec4f,                   16      16
//   radius : f32,                     32      4
//   _pad   : vec3f,                   36      12  → total 48
// }
const SPLAT_UNIFORMS_SIZE = 48; // bytes

// struct PressureUniforms {           offset  size
//   alpha  : f32,                     0       4
//   beta   : f32,                     4       4
//   _pad   : vec2f,                   8       8   → total 16
// }
const PRESSURE_UNIFORMS_SIZE = 16; // bytes

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — shared grid uniform (all passes)
// ─────────────────────────────────────────────────────────────────────────────

const GRID_UNIFORM_WGSL = /* wgsl */`
struct NSUniforms {
  pixel             : vec2f,   // vec2f(1.0/W, 1.0/H) — texel size
  dt                : f32,
  viscosity         : f32,
  velocityDissipation : f32,
  dyeDissipation    : f32,
  vorticityStrength : f32,
  _pad0             : f32,
}
@group(0) @binding(0) var<uniform> u : NSUniforms;
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 0 — SPLAT WGSL
// ─────────────────────────────────────────────────────────────────────────────
//
// Injects a Gaussian force/dye splat at a mouse position.
// Mirrors AT "mousefluid" splat (SceneLayoutPresets mousefluid_scale).
// The Gaussian kernel: exp( -|uv-pos|² / radius² ).
//
// layout:
//   group(0) bind(0): NSUniforms
//   group(1) bind(0): SplatUniforms
//   group(1) bind(1): velWrite (texture_storage_2d rgba16float write)
//   group(1) bind(2): dyeWrite (texture_storage_2d rgba16float write)
//   group(1) bind(3): velRead  (texture_2d f32)
//   group(1) bind(4): dyeRead  (texture_2d f32)

const SPLAT_WGSL = /* wgsl */`
${GRID_UNIFORM_WGSL}

struct SplatUniforms {
  pos    : vec2f,
  vel    : vec2f,
  color  : vec4f,
  radius : f32,
  _pad   : vec3f,
}
@group(1) @binding(0) var<uniform> sp : SplatUniforms;
@group(1) @binding(1) var velWrite : texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var dyeWrite : texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var velRead  : texture_2d<f32>;
@group(1) @binding(4) var dyeRead  : texture_2d<f32>;

@compute @workgroup_size(${WG}, ${WG})
fn splat_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dim  = vec2i(textureDimensions(velWrite));
  let st   = vec2i(gid.xy);
  if (st.x >= dim.x || st.y >= dim.y) { return; }

  let uv   = (vec2f(st) + 0.5) * u.pixel;           // normalised [0,1] UV
  let diff = uv - sp.pos;
  let r2   = sp.radius * sp.radius;
  // Gaussian weight — matches lygia splat convention
  let weight = exp(-dot(diff, diff) / r2);

  // Read existing velocity + dye
  let vel0 = textureLoad(velRead, st, 0);
  let dye0 = textureLoad(dyeRead, st, 0);

  // Accumulate splat impulse on top of existing values
  let vNew = vec4f(vel0.xy + sp.vel * weight, vel0.z, vel0.w);
  let dNew = vec4f(dye0.rgb + sp.color.rgb * weight, dye0.w + weight * 0.5);

  textureStore(velWrite, st, vNew);
  textureStore(dyeWrite, st, dNew);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1 — ADVECTION WGSL
// ─────────────────────────────────────────────────────────────────────────────
//
// Semi-Lagrangian advection — back-project along velocity to find
// the value at the previous timestep, then write to the write texture.
// Handles both velocity self-advection and dye advection.
//
// From LYGIA simpleAndFastFluid.glsl:
//   vec2 was = st - dt * d.xy * pixel;
//   d.xyw = sampler(tex, was).xyw;
//
// layout:
//   group(0) bind(0): NSUniforms
//   group(1) bind(0): velRead  (texture_2d<f32>)  — velocity source
//   group(1) bind(1): velWrite (texture_storage_2d rgba16float write)
//   group(1) bind(2): dyeRead  (texture_2d<f32>)
//   group(1) bind(3): dyeWrite (texture_storage_2d rgba16float write)

const ADVECTION_WGSL = /* wgsl */`
${GRID_UNIFORM_WGSL}

@group(1) @binding(0) var velRead  : texture_2d<f32>;
@group(1) @binding(1) var velWrite : texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var dyeRead  : texture_2d<f32>;
@group(1) @binding(3) var dyeWrite : texture_storage_2d<rgba16float, write>;

// Bilinear texture sample from a texture_2d<f32> at a float UV.
fn sampleBilinear(tex: texture_2d<f32>, uv: vec2f) -> vec4f {
  let dim  = vec2f(textureDimensions(tex));
  let px   = uv * dim - 0.5;          // shift to texel-centre space
  let i    = vec2i(px);
  let f    = fract(px);
  // four corners (clamped to boundary)
  let i00  = clamp(i,              vec2i(0), vec2i(dim) - vec2i(1));
  let i10  = clamp(i + vec2i(1,0), vec2i(0), vec2i(dim) - vec2i(1));
  let i01  = clamp(i + vec2i(0,1), vec2i(0), vec2i(dim) - vec2i(1));
  let i11  = clamp(i + vec2i(1,1), vec2i(0), vec2i(dim) - vec2i(1));
  let v00  = textureLoad(tex, i00, 0);
  let v10  = textureLoad(tex, i10, 0);
  let v01  = textureLoad(tex, i01, 0);
  let v11  = textureLoad(tex, i11, 0);
  return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}

@compute @workgroup_size(${WG}, ${WG})
fn advect_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dim = vec2i(textureDimensions(velWrite));
  let st  = vec2i(gid.xy);
  if (st.x >= dim.x || st.y >= dim.y) { return; }

  let uv  = (vec2f(st) + 0.5) * u.pixel;
  let vel = textureLoad(velRead, st, 0);

  // Back-project: where was this parcel one timestep ago?
  // Velocity is in pixel-space (multiply by pixel to go back to UV).
  let backUV = uv - u.dt * vel.xy * u.pixel;

  // Sample velocity and dye at the back-projected position
  var vAdv = sampleBilinear(velRead, backUV);
  var dAdv = sampleBilinear(dyeRead, backUV);

  // Preserve the w (curl) channel — vorticity recomputed in pass 2
  vAdv.w = vel.w;

  // Apply dissipation (viscous decay)
  vAdv   = vec4f(vAdv.xy * u.velocityDissipation, vAdv.z, vAdv.w);
  dAdv   = vec4f(dAdv.rgb * u.dyeDissipation, dAdv.w * u.dyeDissipation);

  textureStore(velWrite, st, vAdv);
  textureStore(dyeWrite, st, dAdv);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2 — VORTICITY WGSL
// ─────────────────────────────────────────────────────────────────────────────
//
// Two sub-passes encoded sequentially:
//   2a. curl_compute  — computes ω = ∂vy/∂x − ∂vx/∂y, stores in velTex.w
//   2b. curl_confine  — reads the stored curl and applies vorticity confinement
//                       force to the velocity field (re-energises eddies).
//
// Lineage: physics-uniform-bridge.ts :: estimateVorticity
//   ω = ∂vy/∂x − ∂vx/∂y  (2-D curl / vorticity)
// LYGIA simpleAndFastFluid.glsl:
//   d.w = (dB.x - dT.x + dR.y - dL.y)   // curl stored in w channel
//   vorticity = vec2(abs(dT.w)-abs(dB.w), abs(dL.w)-abs(dR.w))
//   vorticity *= STRENGTH / (length + ε) * d.w
//
// layout:
//   group(0) bind(0): NSUniforms
//   group(1) bind(0): velRead  (texture_2d<f32>)
//   group(1) bind(1): velWrite (texture_storage_2d rgba16float write)

const VORTICITY_WGSL = /* wgsl */`
${GRID_UNIFORM_WGSL}

@group(1) @binding(0) var velRead  : texture_2d<f32>;
@group(1) @binding(1) var velWrite : texture_storage_2d<rgba16float, write>;

fn loadVel(st: vec2i) -> vec4f {
  let dim = vec2i(textureDimensions(velRead));
  let c   = clamp(st, vec2i(0), dim - vec2i(1));
  return textureLoad(velRead, c, 0);
}

// ── 2a: Curl (vorticity) computation ─────────────────────────────────────────
@compute @workgroup_size(${WG}, ${WG})
fn curl_compute(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dim = vec2i(textureDimensions(velWrite));
  let st  = vec2i(gid.xy);
  if (st.x >= dim.x || st.y >= dim.y) { return; }

  // Neighbours — central finite differences
  let L = loadVel(st + vec2i(-1,  0));
  let R = loadVel(st + vec2i( 1,  0));
  let B = loadVel(st + vec2i( 0, -1));
  let T = loadVel(st + vec2i( 0,  1));

  // ω = ∂vy/∂x − ∂vx/∂y  (2-D curl; matches lygia convention)
  // Using central differences at half-pixel spacing (×0.5 implicit in sign):
  //   ∂vy/∂x ≈ (R.y - L.y) / (2·dx)
  //   ∂vx/∂y ≈ (T.x - B.x) / (2·dy)
  // Central-difference factor absorbed into vorticity confinement pass.
  let curl = (R.y - L.y) - (T.x - B.x);

  var v = loadVel(st);
  v.w   = curl;  // store curl in w channel
  textureStore(velWrite, st, v);
}

// ── 2b: Vorticity confinement ─────────────────────────────────────────────────
@compute @workgroup_size(${WG}, ${WG})
fn curl_confine(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dim = vec2i(textureDimensions(velWrite));
  let st  = vec2i(gid.xy);
  if (st.x >= dim.x || st.y >= dim.y) { return; }

  // Read curl from w-channel of neighbours (already written by curl_compute)
  let L = loadVel(st + vec2i(-1,  0));
  let R = loadVel(st + vec2i( 1,  0));
  let B = loadVel(st + vec2i( 0, -1));
  let T = loadVel(st + vec2i( 0,  1));

  // ∇|ω| — gradient of curl magnitude (points toward vortex centre)
  var eta = vec2f(abs(R.w) - abs(L.w), abs(T.w) - abs(B.w));
  let lenEta = length(eta) + 1e-5;
  eta = eta / lenEta;  // normalised curl-gradient direction

  // Vorticity confinement force: F = ε · (η × ω · ẑ)
  // In 2-D, ẑ cross (eta.x, eta.y) = (eta.y, -eta.x)
  let d   = loadVel(st);
  let fvc = u.vorticityStrength * vec2f(eta.y, -eta.x) * d.w;

  // Add force to velocity (scaled by dt)
  let vNew = vec4f(d.xy + fvc * u.dt, d.z, d.w);
  textureStore(velWrite, st, vNew);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3 — DIVERGENCE WGSL
// ─────────────────────────────────────────────────────────────────────────────
//
// Computes  div = 0.5 · (∂vx/∂x + ∂vy/∂y)  using central differences.
// From LYGIA simpleAndFastFluid.glsl:
//   divergence = (ddx.x + ddy.y) / (2.0 * dx * dx)
//
// layout:
//   group(0) bind(0): NSUniforms
//   group(1) bind(0): velRead  (texture_2d<f32>)
//   group(1) bind(1): divWrite (texture_storage_2d r16float write)

const DIVERGENCE_WGSL = /* wgsl */`
${GRID_UNIFORM_WGSL}

@group(1) @binding(0) var velRead  : texture_2d<f32>;
@group(1) @binding(1) var divWrite : texture_storage_2d<r16float, write>;

fn loadVelD(st: vec2i) -> vec4f {
  let dim = vec2i(textureDimensions(velRead));
  let c   = clamp(st, vec2i(0), dim - vec2i(1));
  return textureLoad(velRead, c, 0);
}

@compute @workgroup_size(${WG}, ${WG})
fn divergence_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dim = vec2i(textureDimensions(divWrite));
  let st  = vec2i(gid.xy);
  if (st.x >= dim.x || st.y >= dim.y) { return; }

  let L = loadVelD(st + vec2i(-1,  0));
  let R = loadVelD(st + vec2i( 1,  0));
  let B = loadVelD(st + vec2i( 0, -1));
  let T = loadVelD(st + vec2i( 0,  1));

  // Central-difference divergence (half-step factor ×0.5)
  // Matches LYGIA: divergence = (ddx.x + ddy.y) * 0.5
  let div = 0.5 * ((R.x - L.x) + (T.y - B.y));

  textureStore(divWrite, st, vec4f(div, 0.0, 0.0, 1.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 4 — PRESSURE (Jacobi) WGSL
// ─────────────────────────────────────────────────────────────────────────────
//
// Solves  ∇²p = div  iteratively.  Each Jacobi step:
//   p_new(x,y) = (p(x-1,y) + p(x+1,y) + p(x,y-1) + p(x,y+1) − div·α) / β
// where α = dx² = 1.0,  β = 4.0  (standard discrete Laplacian weights).
//
// From LYGIA simpleAndFastFluid.glsl:
//   float a = 1.0 / (dx * dx);
//   d.z = 1.0 / (-4.0*a) * (divergence - a*(dT.z+dR.z+dB.z+dL.z))
//
// layout:
//   group(0) bind(0): NSUniforms
//   group(1) bind(0): preRead  (texture_2d<f32>)   — pressure (ping)
//   group(1) bind(1): preWrite (texture_storage_2d r16float write) — (pong)
//   group(1) bind(2): divRead  (texture_2d<f32>)   — divergence

const PRESSURE_WGSL = /* wgsl */`
${GRID_UNIFORM_WGSL}

@group(1) @binding(0) var preRead  : texture_2d<f32>;
@group(1) @binding(1) var preWrite : texture_storage_2d<r16float, write>;
@group(1) @binding(2) var divRead  : texture_2d<f32>;

fn loadPre(st: vec2i) -> f32 {
  let dim = vec2i(textureDimensions(preRead));
  let c   = clamp(st, vec2i(0), dim - vec2i(1));
  return textureLoad(preRead, c, 0).r;
}

@compute @workgroup_size(${WG}, ${WG})
fn pressure_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dim = vec2i(textureDimensions(preWrite));
  let st  = vec2i(gid.xy);
  if (st.x >= dim.x || st.y >= dim.y) { return; }

  let div = textureLoad(divRead, clamp(st, vec2i(0), dim - vec2i(1)), 0).r;

  let pL  = loadPre(st + vec2i(-1,  0));
  let pR  = loadPre(st + vec2i( 1,  0));
  let pB  = loadPre(st + vec2i( 0, -1));
  let pT  = loadPre(st + vec2i( 0,  1));

  // Standard Jacobi iteration: p_new = (pL+pR+pB+pT - div) / 4
  // (dx=1, α=dx²=1, β=4; from LYGIA: p = (sum - div·a) / (4·a))
  let pNew = (pL + pR + pB + pT - div) * 0.25;

  textureStore(preWrite, st, vec4f(pNew, 0.0, 0.0, 1.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 5 — GRADIENT SUBTRACT (projection) WGSL
// ─────────────────────────────────────────────────────────────────────────────
//
// Subtracts the pressure gradient from the velocity field to enforce
// incompressibility:  v ← v − 0.5·∇p
//
// layout:
//   group(0) bind(0): NSUniforms
//   group(1) bind(0): velRead  (texture_2d<f32>)
//   group(1) bind(1): velWrite (texture_storage_2d rgba16float write)
//   group(1) bind(2): preRead  (texture_2d<f32>)   — converged pressure

const GRADIENT_WGSL = /* wgsl */`
${GRID_UNIFORM_WGSL}

@group(1) @binding(0) var velRead  : texture_2d<f32>;
@group(1) @binding(1) var velWrite : texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var preRead  : texture_2d<f32>;

fn loadPreG(st: vec2i) -> f32 {
  let dim = vec2i(textureDimensions(preRead));
  let c   = clamp(st, vec2i(0), dim - vec2i(1));
  return textureLoad(preRead, c, 0).r;
}

@compute @workgroup_size(${WG}, ${WG})
fn gradient_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dim = vec2i(textureDimensions(velWrite));
  let st  = vec2i(gid.xy);
  if (st.x >= dim.x || st.y >= dim.y) { return; }

  let pL  = loadPreG(st + vec2i(-1,  0));
  let pR  = loadPreG(st + vec2i( 1,  0));
  let pB  = loadPreG(st + vec2i( 0, -1));
  let pT  = loadPreG(st + vec2i( 0,  1));

  // Pressure gradient via central differences
  let gradP = vec2f(pR - pL, pT - pB) * 0.5;

  var vel = textureLoad(velRead, st, 0);
  vel = vec4f(vel.xy - gradP, vel.z, vel.w);

  // Boundary no-slip: zero velocity at grid edges
  let fst  = vec2f(st);
  let fdim = vec2f(dim);
  if (st.x == 0 || st.x == dim.x - 1 || st.y == 0 || st.y == dim.y - 1) {
    vel = vec4f(0.0, 0.0, vel.z, vel.w);
  }

  textureStore(velWrite, st, vel);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// NavierStokesFluid — WebGPU orchestrator class
// ─────────────────────────────────────────────────────────────────────────────

export class NavierStokesFluid {
  private readonly device: GPUDevice;
  private readonly grid  : number;
  readonly params: Required<NavierStokesParams>;

  // ── Textures (ping-pong pairs where noted) ───────────────────────────────

  /** Velocity field: XY=vel, Z=unused, W=curl ω */
  private velTex  : [GPUTexture, GPUTexture];
  private velView : [GPUTextureView, GPUTextureView];

  /** Dye / colour field: RGB=colour, W=density */
  private dyeTex  : [GPUTexture, GPUTexture];
  private dyeView : [GPUTextureView, GPUTextureView];

  /** Divergence (scalar). */
  private divTex  : GPUTexture;
  private divView : GPUTextureView;

  /** Pressure (Jacobi ping-pong). */
  private preTex  : [GPUTexture, GPUTexture];
  private preView : [GPUTextureView, GPUTextureView];

  // ── Uniform buffers ───────────────────────────────────────────────────────

  private gridUniformBuf  : GPUBuffer;
  private splatUniformBuf : GPUBuffer;

  // ── Pipelines ─────────────────────────────────────────────────────────────

  private splatPipeline     : GPUComputePipeline;
  private advectPipeline    : GPUComputePipeline;
  private curlComputePipeline : GPUComputePipeline;
  private curlConfinePipeline : GPUComputePipeline;
  private divergencePipeline: GPUComputePipeline;
  private pressurePipeline  : GPUComputePipeline;
  private gradientPipeline  : GPUComputePipeline;

  // ── Bind-group layouts ────────────────────────────────────────────────────

  private gridBGL   : GPUBindGroupLayout;
  private splatBGL  : GPUBindGroupLayout;
  private advectBGL : GPUBindGroupLayout;
  private curlBGL   : GPUBindGroupLayout;
  private divBGL    : GPUBindGroupLayout;
  private preBGL    : GPUBindGroupLayout;
  private gradBGL   : GPUBindGroupLayout;

  // ── Cached bind groups ────────────────────────────────────────────────────

  private gridBG       : GPUBindGroup;
  // advect / curl / gradient use same vel ping-pong, rebuilt per frame swap
  private advectBG     : [GPUBindGroup, GPUBindGroup] = [null!, null!];
  private curlComputeBG: [GPUBindGroup, GPUBindGroup] = [null!, null!];
  private curlConfineBG: [GPUBindGroup, GPUBindGroup] = [null!, null!];
  private divBG        : [GPUBindGroup, GPUBindGroup] = [null!, null!];
  // pressure BGs rebuilt per Jacobi iteration ping-pong
  private preBG        : [GPUBindGroup, GPUBindGroup] = [null!, null!];
  private gradBG       : [GPUBindGroup, GPUBindGroup] = [null!, null!];

  /** Current ping-pong index (0 or 1). */
  private ping = 0;
  /** Current pressure ping-pong. */
  private prePing = 0;

  private destroyed = false;

  // ─────────────────────────────────────────────────────────────────────────
  // Constructor
  // ─────────────────────────────────────────────────────────────────────────

  constructor(device: GPUDevice, params: NavierStokesParams = {}) {
    this.device = device;

    this.params = {
      grid               : params.grid               ?? NS_GRID,
      dt                 : params.dt                 ?? 1 / 60,
      viscosity          : params.viscosity          ?? 0.16,
      vorticityStrength  : params.vorticityStrength  ?? 0.15,
      velocityDissipation: params.velocityDissipation ?? 0.999,
      dyeDissipation     : params.dyeDissipation     ?? 0.995,
      splatRadius        : params.splatRadius        ?? 0.012,
      pressureIters      : params.pressureIters      ?? NS_PRESSURE_ITERS_DEFAULT,
    };

    this.grid = this.params.grid;

    // ── Allocate textures ────────────────────────────────────────────────────

    const mkTex = (fmt: GPUTextureFormat, label: string): GPUTexture =>
      device.createTexture({
        label,
        size   : [this.grid, this.grid, 1],
        format : fmt,
        usage  :
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.COPY_SRC        |
          GPUTextureUsage.COPY_DST,
      });

    this.velTex  = [mkTex(VEL_FORMAT, 'ns:vel:A'),   mkTex(VEL_FORMAT, 'ns:vel:B')];
    this.dyeTex  = [mkTex(DYE_FORMAT, 'ns:dye:A'),   mkTex(DYE_FORMAT, 'ns:dye:B')];
    this.preTex  = [mkTex(PRE_FORMAT, 'ns:pre:A'),   mkTex(PRE_FORMAT, 'ns:pre:B')];
    this.divTex  = mkTex(DIV_FORMAT, 'ns:div');

    const mkView = (t: GPUTexture) => t.createView();

    this.velView = [mkView(this.velTex[0]), mkView(this.velTex[1])];
    this.dyeView = [mkView(this.dyeTex[0]), mkView(this.dyeTex[1])];
    this.preView = [mkView(this.preTex[0]), mkView(this.preTex[1])];
    this.divView = mkView(this.divTex);

    // ── Uniform buffers ──────────────────────────────────────────────────────

    this.gridUniformBuf = device.createBuffer({
      label: 'ns:uniforms:grid',
      size : UNIFORMS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.splatUniformBuf = device.createBuffer({
      label: 'ns:uniforms:splat',
      size : SPLAT_UNIFORMS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.writeGridUniforms();

    // ── Build pipelines and bind groups ─────────────────────────────────────

    const [
      gridBGL, splatBGL, advectBGL, curlBGL, divBGL, preBGL, gradBGL,
    ] = this.buildLayouts();

    this.gridBGL  = gridBGL;
    this.splatBGL = splatBGL;
    this.advectBGL = advectBGL;
    this.curlBGL   = curlBGL;
    this.divBGL    = divBGL;
    this.preBGL    = preBGL;
    this.gradBGL   = gradBGL;

    const mk = (label: string, wgsl: string, entry: string, bgls: GPUBindGroupLayout[]) =>
      device.createComputePipeline({
        label,
        layout: device.createPipelineLayout({ bindGroupLayouts: bgls }),
        compute: { module: device.createShaderModule({ code: wgsl }), entryPoint: entry },
      });

    this.splatPipeline      = mk('ns:splat',        SPLAT_WGSL,      'splat_main',      [gridBGL, splatBGL]);
    this.advectPipeline     = mk('ns:advect',       ADVECTION_WGSL,  'advect_main',     [gridBGL, advectBGL]);
    this.curlComputePipeline= mk('ns:curl:compute', VORTICITY_WGSL,  'curl_compute',    [gridBGL, curlBGL]);
    this.curlConfinePipeline= mk('ns:curl:confine', VORTICITY_WGSL,  'curl_confine',    [gridBGL, curlBGL]);
    this.divergencePipeline = mk('ns:divergence',   DIVERGENCE_WGSL, 'divergence_main', [gridBGL, divBGL]);
    this.pressurePipeline   = mk('ns:pressure',     PRESSURE_WGSL,   'pressure_main',   [gridBGL, preBGL]);
    this.gradientPipeline   = mk('ns:gradient',     GRADIENT_WGSL,   'gradient_main',   [gridBGL, gradBGL]);

    this.gridBG = this.buildGridBG();
    this.rebuildSwapBGs();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Bind-group layout factory
  // ─────────────────────────────────────────────────────────────────────────

  private buildLayouts(): [
    GPUBindGroupLayout, GPUBindGroupLayout, GPUBindGroupLayout,
    GPUBindGroupLayout, GPUBindGroupLayout, GPUBindGroupLayout,
    GPUBindGroupLayout,
  ] {
    const dev = this.device;

    const tex2d   = (binding: number): GPUBindGroupLayoutEntry => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      texture: { sampleType: 'unfilterable-float' },
    });
    const storTex = (binding: number, fmt: GPUTextureFormat): GPUBindGroupLayoutEntry => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: 'write-only', format: fmt },
    });
    const ubo = (binding: number): GPUBindGroupLayoutEntry => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'uniform' },
    });

    const gridBGL = dev.createBindGroupLayout({
      label  : 'ns:bgl:grid',
      entries: [ubo(0)],
    });

    // splat: 0=SplatUniforms, 1=velWrite, 2=dyeWrite, 3=velRead, 4=dyeRead
    const splatBGL = dev.createBindGroupLayout({
      label  : 'ns:bgl:splat',
      entries: [
        ubo(0),
        storTex(1, VEL_FORMAT),
        storTex(2, DYE_FORMAT),
        tex2d(3),
        tex2d(4),
      ],
    });

    // advect: 0=velRead, 1=velWrite, 2=dyeRead, 3=dyeWrite
    const advectBGL = dev.createBindGroupLayout({
      label  : 'ns:bgl:advect',
      entries: [tex2d(0), storTex(1, VEL_FORMAT), tex2d(2), storTex(3, DYE_FORMAT)],
    });

    // curl (shared for curl_compute and curl_confine): 0=velRead, 1=velWrite
    const curlBGL = dev.createBindGroupLayout({
      label  : 'ns:bgl:curl',
      entries: [tex2d(0), storTex(1, VEL_FORMAT)],
    });

    // divergence: 0=velRead, 1=divWrite
    const divBGL = dev.createBindGroupLayout({
      label  : 'ns:bgl:div',
      entries: [tex2d(0), storTex(1, DIV_FORMAT)],
    });

    // pressure: 0=preRead, 1=preWrite, 2=divRead
    const preBGL = dev.createBindGroupLayout({
      label  : 'ns:bgl:pre',
      entries: [tex2d(0), storTex(1, PRE_FORMAT), tex2d(2)],
    });

    // gradient: 0=velRead, 1=velWrite, 2=preRead
    const gradBGL = dev.createBindGroupLayout({
      label  : 'ns:bgl:grad',
      entries: [tex2d(0), storTex(1, VEL_FORMAT), tex2d(2)],
    });

    return [gridBGL, splatBGL, advectBGL, curlBGL, divBGL, preBGL, gradBGL];
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Bind-group builders
  // ─────────────────────────────────────────────────────────────────────────

  private buildGridBG(): GPUBindGroup {
    return this.device.createBindGroup({
      label  : 'ns:bg:grid',
      layout : this.gridBGL,
      entries: [{ binding: 0, resource: { buffer: this.gridUniformBuf } }],
    });
  }

  private rebuildSwapBGs(): void {
    const dev = this.device;
    // ping=0: src=A dst=B, ping=1: src=B dst=A
    for (const p of [0, 1] as const) {
      const q = 1 - p;  // destination

      this.advectBG[p] = dev.createBindGroup({
        label  : `ns:bg:advect:${p}`,
        layout : this.advectBGL,
        entries: [
          { binding: 0, resource: this.velView[p] },
          { binding: 1, resource: this.velView[q] },
          { binding: 2, resource: this.dyeView[p] },
          { binding: 3, resource: this.dyeView[q] },
        ],
      });

      this.curlComputeBG[p] = dev.createBindGroup({
        label  : `ns:bg:curlCompute:${p}`,
        layout : this.curlBGL,
        entries: [
          { binding: 0, resource: this.velView[p] },
          { binding: 1, resource: this.velView[q] },
        ],
      });

      this.curlConfineBG[p] = dev.createBindGroup({
        label  : `ns:bg:curlConfine:${p}`,
        layout : this.curlBGL,
        entries: [
          { binding: 0, resource: this.velView[q] },   // read the result of curlCompute
          { binding: 1, resource: this.velView[p] },   // write back to original
        ],
      });

      this.divBG[p] = dev.createBindGroup({
        label  : `ns:bg:div:${p}`,
        layout : this.divBGL,
        entries: [
          { binding: 0, resource: this.velView[p] },
          { binding: 1, resource: this.divView },
        ],
      });

      this.gradBG[p] = dev.createBindGroup({
        label  : `ns:bg:grad:${p}`,
        layout : this.gradBGL,
        entries: [
          { binding: 0, resource: this.velView[p] },
          { binding: 1, resource: this.velView[q] },
          { binding: 2, resource: this.preView[this.prePing] },
        ],
      });
    }

    // pressure ping-pong (both directions)
    for (const pp of [0, 1] as const) {
      const qq = 1 - pp;
      this.preBG[pp] = dev.createBindGroup({
        label  : `ns:bg:pre:${pp}`,
        layout : this.preBGL,
        entries: [
          { binding: 0, resource: this.preView[pp] },
          { binding: 1, resource: this.preView[qq] },
          { binding: 2, resource: this.divView },
        ],
      });
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Uniform writers
  // ─────────────────────────────────────────────────────────────────────────

  private writeGridUniforms(): void {
    const buf = new Float32Array(UNIFORMS_SIZE / 4);
    const p   = this.params;
    const inv = 1.0 / this.grid;
    buf[0]  = inv;                    // pixel.x
    buf[1]  = inv;                    // pixel.y
    buf[2]  = p.dt;
    buf[3]  = p.viscosity;
    buf[4]  = p.velocityDissipation;
    buf[5]  = p.dyeDissipation;
    buf[6]  = p.vorticityStrength;
    buf[7]  = 0;                      // pad
    this.device.queue.writeBuffer(this.gridUniformBuf, 0, buf);
  }

  private writeSplatUniforms(s: NavierStokesSplat): void {
    const buf = new Float32Array(SPLAT_UNIFORMS_SIZE / 4);
    buf[0]  = s.x;
    buf[1]  = s.y;
    buf[2]  = s.vx;
    buf[3]  = s.vy;
    buf[4]  = s.color[0];
    buf[5]  = s.color[1];
    buf[6]  = s.color[2];
    buf[7]  = 1.0;                    // alpha
    buf[8]  = this.params.splatRadius;
    // buf[9..11] pad
    this.device.queue.writeBuffer(this.splatUniformBuf, 0, buf);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // dispatch helper
  // ─────────────────────────────────────────────────────────────────────────

  private dispatchGrid(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline): void {
    pass.setPipeline(pipeline);
    const tiles = Math.ceil(this.grid / WG);
    pass.dispatchWorkgroups(tiles, tiles);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API — splat (mouse fluid impulse)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Encodes a single mouse-fluid splat impulse into the given command encoder.
   * Call this BEFORE step() each frame for any pending mouse events.
   *
   * @param encoder  — active GPUCommandEncoder
   * @param splat    — position, velocity impulse and dye colour
   */
  splat(encoder: GPUCommandEncoder, splat: NavierStokesSplat): void {
    if (this.destroyed) return;
    this.writeSplatUniforms(splat);

    // splat bind group — writes to current ping destination, reads from ping source
    const p = this.ping;
    const q = 1 - p;
    const splatBG = this.device.createBindGroup({
      label  : 'ns:bg:splat:frame',
      layout : this.splatBGL,
      entries: [
        { binding: 0, resource: { buffer: this.splatUniformBuf } },
        { binding: 1, resource: this.velView[q] },   // write dst
        { binding: 2, resource: this.dyeView[q] },
        { binding: 3, resource: this.velView[p] },   // read src
        { binding: 4, resource: this.dyeView[p] },
      ],
    });

    const pass = encoder.beginComputePass({ label: 'ns:splat' });
    pass.setPipeline(this.splatPipeline);
    pass.setBindGroup(0, this.gridBG);
    pass.setBindGroup(1, splatBG);
    const tiles = Math.ceil(this.grid / WG);
    pass.dispatchWorkgroups(tiles, tiles);
    pass.end();

    // Splat writes to [q]; swap so subsequent passes see [q] as source
    this.ping = q;
    this.rebuildSwapBGs();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API — step (one full simulation frame)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Encodes a full Navier-Stokes simulation step.
   * Pipeline: advection → vorticity → divergence → pressure(×N) → gradient subtract.
   *
   * @param encoder  — active GPUCommandEncoder
   */
  step(encoder: GPUCommandEncoder): void {
    if (this.destroyed) return;

    const p = this.ping;
    const q = 1 - p;

    // ── Pass 1: Advection ──────────────────────────────────────────────────
    {
      const pass = encoder.beginComputePass({ label: 'ns:advect' });
      pass.setPipeline(this.advectPipeline);
      pass.setBindGroup(0, this.gridBG);
      pass.setBindGroup(1, this.advectBG[p]);
      const tiles = Math.ceil(this.grid / WG);
      pass.dispatchWorkgroups(tiles, tiles);
      pass.end();
    }
    // After advection: vel/dye are in [q]  →  swap
    this.ping = q;

    // ── Pass 2a: Curl compute ──────────────────────────────────────────────
    // Read from new ping (q), write curl into (p)
    {
      const pass = encoder.beginComputePass({ label: 'ns:curl:compute' });
      pass.setPipeline(this.curlComputePipeline);
      pass.setBindGroup(0, this.gridBG);
      pass.setBindGroup(1, this.curlComputeBG[this.ping]);
      const tiles = Math.ceil(this.grid / WG);
      pass.dispatchWorkgroups(tiles, tiles);
      pass.end();
    }

    // ── Pass 2b: Vorticity confinement ────────────────────────────────────
    {
      const pass = encoder.beginComputePass({ label: 'ns:curl:confine' });
      pass.setPipeline(this.curlConfinePipeline);
      pass.setBindGroup(0, this.gridBG);
      pass.setBindGroup(1, this.curlConfineBG[this.ping]);
      const tiles = Math.ceil(this.grid / WG);
      pass.dispatchWorkgroups(tiles, tiles);
      pass.end();
    }
    // Confinement writes back to original [this.ping]; ping unchanged

    // ── Pass 3: Divergence ────────────────────────────────────────────────
    {
      const pass = encoder.beginComputePass({ label: 'ns:divergence' });
      pass.setPipeline(this.divergencePipeline);
      pass.setBindGroup(0, this.gridBG);
      pass.setBindGroup(1, this.divBG[this.ping]);
      const tiles = Math.ceil(this.grid / WG);
      pass.dispatchWorkgroups(tiles, tiles);
      pass.end();
    }

    // ── Pass 4: Pressure Jacobi (N iterations) ────────────────────────────
    for (let i = 0; i < this.params.pressureIters; i++) {
      const pp  = this.prePing;
      const pp2 = 1 - pp;
      const pass = encoder.beginComputePass({ label: `ns:pressure:${i}` });
      pass.setPipeline(this.pressurePipeline);
      pass.setBindGroup(0, this.gridBG);
      pass.setBindGroup(1, this.preBG[pp]);
      const tiles = Math.ceil(this.grid / WG);
      pass.dispatchWorkgroups(tiles, tiles);
      pass.end();
      this.prePing = pp2;
    }

    // ── Pass 5: Gradient subtract (projection) ────────────────────────────
    // Rebuild gradBG to point at the final converged pressure
    this.rebuildSwapBGs();
    {
      const pass = encoder.beginComputePass({ label: 'ns:gradient' });
      pass.setPipeline(this.gradientPipeline);
      pass.setBindGroup(0, this.gridBG);
      pass.setBindGroup(1, this.gradBG[this.ping]);
      const tiles = Math.ceil(this.grid / WG);
      pass.dispatchWorkgroups(tiles, tiles);
      pass.end();
    }
    // gradient writes to [1-ping]; swap
    this.ping = 1 - this.ping;
    this.rebuildSwapBGs();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API — parameter update
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Updates simulation parameters and rewrites the GPU uniform buffer.
   * Safe to call every frame (e.g. for live UIL tweaking).
   */
  updateParams(partial: Partial<NavierStokesParams>): void {
    Object.assign(this.params, partial);
    this.writeGridUniforms();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API — texture accessors (for downstream renders)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * The current velocity texture view (XY=vel, W=curl).
   * Bind to fragment shaders or other compute passes for advection overlays.
   */
  get velocityTextureView(): GPUTextureView {
    return this.velView[this.ping];
  }

  /**
   * The current dye / colour texture view (RGB=colour, W=density).
   * Bind to the fluid-surface.frag uDensityTex / uVelocityTex inputs
   * (via WebGL→WebGPU interop or a blit pass).
   */
  get dyeTextureView(): GPUTextureView {
    return this.dyeView[this.ping];
  }

  /**
   * The converged pressure texture view.
   */
  get pressureTextureView(): GPUTextureView {
    return this.preView[this.prePing];
  }

  /**
   * The divergence texture view.
   */
  get divergenceTextureView(): GPUTextureView {
    return this.divView;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Lifecycle
  // ─────────────────────────────────────────────────────────────────────────

  /** Release all GPU resources. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;

    for (const t of [...this.velTex, ...this.dyeTex, ...this.preTex]) t.destroy();
    this.divTex.destroy();
    this.gridUniformBuf.destroy();
    this.splatUniformBuf.destroy();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory helper — matches project createXxx() convention
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Creates a NavierStokesFluid and returns it, or null if the device is
 * unavailable (e.g. WebGPU not supported).
 */
export function createNavierStokesFluid(
  device: GPUDevice | null | undefined,
  params?: NavierStokesParams,
): NavierStokesFluid | null {
  if (!device) return null;
  return new NavierStokesFluid(device, params);
}
