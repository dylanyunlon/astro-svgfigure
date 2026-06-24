/**
 * subsurface-scattering.ts — M780: Subsurface Scattering
 * ─────────────────────────────────────────────────────────────────────────────
 * 次表面散射——Cell 内部光线穿透（蜡烛/皮肤/树叶效果）。基于 depth difference +
 * blur + color absorption。不同 species 不同散射颜色和强度。
 *
 * 算法概览
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Pass 0 ── Thickness Map (compute) ──────────────────────────────────────┐
 *   │  depthTex (front-face) + backDepthTex (back-face) → thicknessTex         │
 *   │  Per-pixel: thickness = linearize(backDepth) - linearize(frontDepth)     │
 *   │  Thin regions → high transmittance; thick regions → low transmittance.   │
 *   │  speciesIdTex provides per-pixel species lookup for absorption profile.  │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                │ thicknessTex (r16float — view-space thickness, world units)
 *                ▼
 *   ┌─ Pass 1 ── Diffusion Blur (compute, separable) ─────────────────────────┐
 *   │  thicknessTex → diffusedTex                                              │
 *   │  Separable Gaussian blur whose kernel width scales with local thickness: │
 *   │  thicker regions → wider blur → softer scatter spread.                   │
 *   │  Two dispatches: horizontal then vertical (classic separable pattern).   │
 *   │  Depth-aware: taps whose depth differs by > depthThreshold are rejected  │
 *   │  to preserve cell boundary edges (same strategy as SSAO bilateral blur). │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                │ diffusedTex (r16float — blurred thickness)
 *                ▼
 *   ┌─ Pass 2 ── Transmittance + Absorption (compute) ────────────────────────┐
 *   │  diffusedTex + sceneTex + speciesIdTex + lightDir → sssTex               │
 *   │  Per-pixel:                                                              │
 *   │    1. Sample blurred thickness d                                         │
 *   │    2. Look up species absorption profile σ_a (RGB extinction coeff.)     │
 *   │    3. Transmittance T = exp(-σ_a · d)  (Beer-Lambert law)                │
 *   │    4. Wrap-lighting NdotL with configurable wrap factor                  │
 *   │    5. Forward scatter: pow(VdotL, power) × species scatter tint          │
 *   │    6. SSS color = lightColor × T × (wrapNdotL + forwardScatter)          │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                │ sssTex (rgba16float — scattered light contribution)
 *                ▼
 *   ┌─ Pass 3 ── Composite (compute) ──────────────────────────────────────────┐
 *   │  sceneTex + sssTex → dst                                                 │
 *   │  Additive blend: output = scene + sss × globalIntensity                  │
 *   │  Optional: curvature-driven intensity boost at convex regions.           │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 * Per-Species Scattering Profiles
 * ─────────────────────────────────────────────────────────────────────────────
 * Each CellSpecies carries a distinct SSSProfile that encodes the material's
 * optical properties — based on real-world analogies:
 *
 *   │ Species      │ Analogy     │ Scatter Tint         │ σ_a (extinction)       │
 *   ├──────────────┼─────────────┼──────────────────────┼────────────────────────┤
 *   │ attention    │ Wax/candle  │ warm amber           │ low R, mid G, high B   │
 *   │ ffn          │ Glass/amber │ golden-orange        │ very low uniform       │
 *   │ layernorm    │ Marble/jade │ cool grey-green      │ high uniform (opaque)  │
 *   │ embedding    │ Leaf/tissue │ green chlorophyll     │ low G, high R/B        │
 *   │ softmax      │ Skin/plasma │ warm red-orange      │ low R, mid G, high B   │
 *
 * The σ_a vectors are tuned so that short-wavelength (blue) light is absorbed
 * faster in organic materials (wax, skin, leaf), matching the real-world
 * observation that subsurface-scattered light shifts toward warm/red tones.
 *
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/ambient-occlusion.ts         — bilateral blur pattern, depth reconstruction
 *   src/lib/sph/god-rays.ts                  — compute post-process pipeline pattern
 *   src/lib/sph/dof-bokeh.ts                 — separable blur + staging pattern
 *   src/lib/sph/cell-material-system.ts      — CellSpecies type, SpeciesMaterialDef
 *   src/lib/sph/species-shader-registry.ts   — SpeciesShaderConfig, PhysicsBindings
 *   src/lib/sph/physics-uniform-bridge.ts    — PhysicsUniforms (density modulates scatter)
 *   src/lib/sph/at-pbr-material.ts           — PBRParams (albedo, roughness for SSS blend)
 *   src/lib/sph/at-render-pipeline.ts        — FBO chain orchestration
 *
 * Research references:
 *   - Jimenez et al. 2015  — "Separable Subsurface Scattering" (SSSS)
 *   - Penner & Borshukov   — "Pre-Integrated Skin Shading" (GDC 2011)
 *   - Barré-Brisebois 2011 — "Approximating Translucency for a Fast, Cheap,
 *                             and Convincing Subsurface Scattering Look" (GDC)
 *   - d'Eon & Luebke 2007  — "Advanced Techniques for Realistic Real-Time
 *                             Skin Rendering" (GPU Gems 3, Ch. 14)
 *
 * Research: xiaodi #M780 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────


import type { CellSpecies } from './cell-material-system';
import type { PhysicsUniforms } from './physics-uniform-bridge';

<<<<<<< HEAD
// [orphan-precise] /** Workgroup size for compute dispatches (16×16 = 256 threads). */
=======
/** Workgroup size for compute dispatches (16×16 = 256 threads). */




>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
const WG_SIZE = 16;

/** Maximum number of Gaussian blur taps per axis (must match WGSL constant). */
const MAX_BLUR_TAPS = 25;

/** Maximum species count supported by the species LUT texture. */
const MAX_SPECIES = 16;

/** Thickness scale factor for depth difference → world units conversion. */
const THICKNESS_SCALE = 1.0;

// ─────────────────────────────────────────────────────────────────────────────
// Per-Species SSS Profile
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Subsurface scattering optical profile for a single species.
 *
 * Encodes the material's light-transport characteristics using a physically
 * motivated (but artistically tuneable) parameterisation:
 *
 *   σ_a  — RGB volumetric absorption (extinction) coefficients.
 *           Higher values → faster absorption of that wavelength.
 *           Beer-Lambert: transmittance T_c = exp(-σ_a_c · thickness)
 *
 *   scatterTint — RGB tint of the scattered light contribution after
 *                 absorption.  This is the "glow colour" visible when
 *                 back-lighting the cell (e.g. warm red for skin, green for leaf).
 *
 *   scatterWidth — controls the Gaussian blur radius multiplier.
 *                  Higher → light diffuses further inside the medium → softer.
 *
 *   wrapFactor — wrap-lighting factor for NdotL (0 = Lambertian, 1 = fully
 *                wrapped hemisphere lighting).  Higher values simulate more
 *                light wrapping around curvature.
 *
 *   forwardScatterPower — exponent for the VdotL forward-scatter lobe.
 *                         Lower → broader forward scatter; higher → tighter.
 *
 *   forwardScatterStrength — intensity multiplier for the forward-scatter term.
 *
 *   translucency — overall translucency blend (0 = opaque, 1 = fully translucent).
 *                  Modulates the final SSS contribution before compositing.
 */
export interface SSSProfile {
  /** RGB volumetric absorption coefficients (Beer-Lambert σ_a). */
  absorption: [number, number, number];

  /** RGB tint of scattered light emerging from the back face. */
  scatterTint: [number, number, number];

  /** Blur radius multiplier — controls scatter spread distance. */
  scatterWidth: number;

  /** Wrap-lighting factor for diffuse NdotL (0–1). */
  wrapFactor: number;

  /** Exponent for the forward-scatter lobe (VdotL term). */
  forwardScatterPower: number;

  /** Intensity multiplier for the forward-scatter term. */
  forwardScatterStrength: number;

  /** Overall translucency (0 = opaque stone, 1 = thin leaf/wax). */
  translucency: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Species SSS Registry
// ─────────────────────────────────────────────────────────────────────────────

/**
 * SSS_PROFILE_REGISTRY
 *
 * Per-species subsurface scattering profiles.  Each entry encodes how light
 * penetrates, scatters inside, and re-emerges from the cell body.
 *
 * Design rationale mirrors cell-material-system.ts Transformer analogies:
 *
 *   attention  → 蜡烛 (candle/wax):
 *     Warm amber translucency.  Low R absorption lets red light pass through
 *     easily; high B absorption eats blue rapidly.  Wide scatter spread mimics
 *     the soft warm glow of a candle viewed against light.  Moderate wrap
 *     lighting simulates the spherical curvature of a molten wax body.
 *
 *   ffn → 琥珀玻璃 (amber glass):
 *     Very low, nearly uniform absorption — glass is almost transparent.
 *     Golden-orange scatter tint from the amber substrate.  Tight forward
 *     scatter (high power) produces focused caustic-like transmission.
 *     Low scatter width because glass doesn't diffuse much laterally.
 *
 *   layernorm → 大理石/玉石 (marble/jade):
 *     High uniform absorption — marble is mostly opaque, but thin edges let
 *     some cool grey-green light bleed through.  Narrow scatter width: light
 *     doesn't travel far in dense stone.  Weak translucency overall, but the
 *     effect is visible at thin shell edges and bevels.
 *
 *   embedding → 树叶/组织 (leaf/tissue):
 *     Classic chlorophyll transmission: green passes easily (low σ_a_G),
 *     red and blue are strongly absorbed.  Wide scatter width for the soft
 *     internal glow of a back-lit leaf.  High wrap factor simulates the
 *     thin, rounded curvature of a cell membrane.  Strong forward scatter
 *     for the distinctive back-lit leaf halo.
 *
 *   softmax → 皮肤/等离子 (skin/plasma):
 *     Human-skin-like scattering: red passes most easily (low σ_a_R) due to
 *     hemoglobin spectral properties; blue is absorbed fastest.  The warm
 *     red-orange scatter tint is the hallmark of skin SSS (ears, fingertips
 *     back-lit by sunlight).  Medium scatter width for natural skin softness.
 *     Moderate forward scatter for the rim-lit translucency of plasma edges.
 */
export const SSS_PROFILE_REGISTRY: Readonly<Record<CellSpecies, SSSProfile>> = {

  // ── attention — 蜡烛 (candle wax) ─────────────────────────────────────────
  attention: {
    absorption:             [0.15, 0.60, 1.80],  // R passes easily; B absorbed fast
    scatterTint:            [1.00, 0.78, 0.35],  // warm amber glow
    scatterWidth:           2.8,                  // wide — soft wax diffusion
    wrapFactor:             0.45,                 // moderate wrap (spherical body)
    forwardScatterPower:    3.5,                  // medium-broad forward lobe
    forwardScatterStrength: 0.40,                 // visible but not dominant
    translucency:           0.75,                 // quite translucent
  },

  // ── ffn — 琥珀玻璃 (amber glass) ──────────────────────────────────────────
  ffn: {
    absorption:             [0.08, 0.12, 0.35],  // nearly transparent; slight blue absorption
    scatterTint:            [1.00, 0.88, 0.50],  // golden-orange transmission
    scatterWidth:           1.2,                  // narrow — glass doesn't diffuse much
    wrapFactor:             0.20,                 // low wrap (hard dielectric surface)
    forwardScatterPower:    8.0,                  // tight focused caustic lobe
    forwardScatterStrength: 0.65,                 // strong forward transmission
    translucency:           0.85,                 // very translucent (glass)
  },

  // ── layernorm — 大理石/玉石 (marble / jade) ────────────────────────────────
  layernorm: {
    absorption:             [1.40, 1.20, 1.00],  // high, nearly uniform — mostly opaque
    scatterTint:            [0.80, 0.88, 0.82],  // cool grey-green
    scatterWidth:           0.8,                  // narrow — dense stone
    wrapFactor:             0.15,                 // minimal wrap (hard polished surface)
    forwardScatterPower:    5.0,                  // medium lobe
    forwardScatterStrength: 0.15,                 // weak — marble barely transmits
    translucency:           0.25,                 // mostly opaque, thin edges only
  },

  // ── embedding — 树叶/组织 (leaf / biological tissue) ──────────────────────
  embedding: {
    absorption:             [1.20, 0.18, 1.50],  // green passes (low G); R/B absorbed
    scatterTint:            [0.45, 0.92, 0.38],  // chlorophyll green
    scatterWidth:           3.2,                  // wide — soft internal leaf glow
    wrapFactor:             0.55,                 // high wrap (thin curved membrane)
    forwardScatterPower:    2.5,                  // broad forward lobe
    forwardScatterStrength: 0.55,                 // strong back-lit halo
    translucency:           0.80,                 // highly translucent (thin tissue)
  },

  // ── softmax — 皮肤/等离子 (skin / plasma) ─────────────────────────────────
  softmax: {
    absorption:             [0.10, 0.55, 1.60],  // red passes (hemoglobin); blue absorbed
    scatterTint:            [1.00, 0.60, 0.30],  // warm red-orange (skin SSS hallmark)
    scatterWidth:           2.2,                  // medium — natural skin softness
    wrapFactor:             0.40,                 // moderate wrap (organic curvature)
    forwardScatterPower:    4.0,                  // medium forward lobe
    forwardScatterStrength: 0.45,                 // moderate rim translucency
    translucency:           0.65,                 // partially translucent
  },

} as const;

// ─────────────────────────────────────────────────────────────────────────────
// SSS Params (runtime-adjustable global parameters)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Runtime-adjustable parameters for the SSS post-process pipeline.
 * These control the overall behaviour; per-species tuning is in SSSProfile.
 */
export interface SSSParams {
  /** Global intensity multiplier for the final SSS contribution. Default: 1.0 */
  globalIntensity: number;

  /** Depth threshold for bilateral blur edge preservation (view-space). Default: 0.1 */
  depthThreshold: number;

  /** Base blur radius in pixels before species scatterWidth scaling. Default: 8.0 */
  baseBlurRadius: number;

  /** Near plane distance for depth linearisation. Default: 0.1 */
  nearPlane: number;

  /** Far plane distance for depth linearisation. Default: 100.0 */
  farPlane: number;

  /** Light direction (view-space, normalised). Default: [0.4, 0.7, 0.6] */
  lightDir: [number, number, number];

  /** Light colour (linear HDR). Default: [1.0, 0.98, 0.92] (warm sunlight) */
  lightColor: [number, number, number];

  /**
   * Ambient scatter boost — adds a view-independent minimum scatter term so
   * that even unlit sides of cells get a faint SSS halo.  Default: 0.08
   */
  ambientScatter: number;

  /**
   * Curvature intensity boost — amplifies SSS at convex regions where the
   * surface curves away from the viewer (thin shell edges).  Default: 0.3
   */
  curvatureBoost: number;
}

/** Sensible defaults for SSSParams. */
export const DEFAULT_SSS_PARAMS: Readonly<SSSParams> = {
  globalIntensity: 1.0,
  depthThreshold:  0.1,
  baseBlurRadius:  8.0,
  nearPlane:       0.1,
  farPlane:        100.0,
  lightDir:        [0.4, 0.7, 0.6],
  lightColor:      [1.0, 0.98, 0.92],
  ambientScatter:  0.08,
  curvatureBoost:  0.3,
};

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Common helpers (shared across passes)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_SSS_COMMON = /* wgsl */ `
const PI      : f32 = 3.14159265358979323846;
const TWO_PI  : f32 = 6.28318530717958647693;
const INV_PI  : f32 = 0.31830988618379067154;
const EPSILON : f32 = 1e-6;

fn saturate_f(v: f32) -> f32 { return clamp(v, 0.0, 1.0); }
fn saturate_v3(v: vec3f) -> vec3f { return clamp(v, vec3f(0.0), vec3f(1.0)); }

// ── Reverse-Z depth linearisation (matches ambient-occlusion.ts) ─────────────
fn linearizeDepth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

// ── Beer-Lambert absorption ──────────────────────────────────────────────────
// T = exp(-σ_a · d)  — per-channel RGB transmittance through thickness d
fn beerLambert(absorption: vec3f, thickness: f32) -> vec3f {
    return exp(-absorption * max(thickness, 0.0));
}

// ── Gaussian weight ──────────────────────────────────────────────────────────
fn gaussianWeight(offset: f32, sigma: f32) -> f32 {
    let inv2s2 = 1.0 / (2.0 * sigma * sigma + EPSILON);
    return exp(-offset * offset * inv2s2);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 0: Thickness Map
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_THICKNESS_MAP = /* wgsl */ `
${WGSL_SSS_COMMON}

// ── Uniforms ──────────────────────────────────────────────────────────────────
struct ThicknessUniforms {
    nearPlane      : f32,
    farPlane       : f32,
    thicknessScale : f32,
    _pad0          : f32,
    resolution     : vec2f,
    _pad1          : vec2f,
}

@group(0) @binding(0) var<uniform>  u_params    : ThicknessUniforms;
@group(0) @binding(1) var           t_frontDepth: texture_2d<f32>;
@group(0) @binding(2) var           t_backDepth : texture_2d<f32>;
@group(0) @binding(3) var           t_output    : texture_storage_2d<r16float, write>;

@compute @workgroup_size(${WG_SIZE}, ${WG_SIZE})
fn cs_thickness(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2u(u_params.resolution);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coord = vec2i(gid.xy);
    let frontRaw = textureLoad(t_frontDepth, coord, 0).r;
    let backRaw  = textureLoad(t_backDepth,  coord, 0).r;

    // Linearise both depth samples
    let frontZ = linearizeDepth(frontRaw, u_params.nearPlane, u_params.farPlane);
    let backZ  = linearizeDepth(backRaw,  u_params.nearPlane, u_params.farPlane);

    // Thickness = back - front (clamped to non-negative)
    let thickness = max(backZ - frontZ, 0.0) * u_params.thicknessScale;

    textureStore(t_output, coord, vec4f(thickness, 0.0, 0.0, 0.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 1: Depth-Aware Separable Gaussian Blur (Diffusion)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_DIFFUSION_BLUR = /* wgsl */ `
${WGSL_SSS_COMMON}

struct BlurUniforms {
    direction      : vec2f,    // (1,0) for horizontal, (0,1) for vertical
    baseRadius     : f32,      // base blur radius in pixels
    depthThreshold : f32,      // bilateral depth rejection threshold
    nearPlane      : f32,
    farPlane       : f32,
    resolution     : vec2f,
}

@group(0) @binding(0) var<uniform>  u_blur     : BlurUniforms;
@group(0) @binding(1) var           t_thickness: texture_2d<f32>;
@group(0) @binding(2) var           t_depth    : texture_2d<f32>;
@group(0) @binding(3) var           t_output   : texture_storage_2d<r16float, write>;

@compute @workgroup_size(${WG_SIZE}, ${WG_SIZE})
fn cs_diffusion_blur(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2u(u_blur.resolution);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let center = vec2i(gid.xy);
    let centerThickness = textureLoad(t_thickness, center, 0).r;
    let centerDepthRaw  = textureLoad(t_depth, center, 0).r;
    let centerDepth     = linearizeDepth(centerDepthRaw, u_blur.nearPlane, u_blur.farPlane);

    // Adaptive sigma: thicker regions → wider blur (more internal diffusion)
    let adaptiveScale = saturate_f(centerThickness * 0.5) + 0.3;
    let sigma         = u_blur.baseRadius * adaptiveScale;
    let kernelRadius  = i32(ceil(sigma * 2.5));
    let maxTaps       = min(kernelRadius, ${MAX_BLUR_TAPS});

    var accumThickness : f32 = 0.0;
    var totalWeight    : f32 = 0.0;

    let step = vec2i(u_blur.direction);

    for (var i = -maxTaps; i <= maxTaps; i++) {
        let sampleCoord = center + step * i;

        // Bounds check
        if (sampleCoord.x < 0 || sampleCoord.y < 0 ||
            sampleCoord.x >= i32(dims.x) || sampleCoord.y >= i32(dims.y)) {
            continue;
        }

        let sampleThickness = textureLoad(t_thickness, sampleCoord, 0).r;
        let sampleDepthRaw  = textureLoad(t_depth, sampleCoord, 0).r;
        let sampleDepth     = linearizeDepth(sampleDepthRaw, u_blur.nearPlane, u_blur.farPlane);

        // Bilateral depth rejection — preserves cell boundary edges
        let depthDiff = abs(sampleDepth - centerDepth);
        if (depthDiff > u_blur.depthThreshold) { continue; }

        // Spatial Gaussian weight
        let w = gaussianWeight(f32(i), sigma);

        // Depth closeness weight (soft falloff instead of hard reject)
        let depthW = exp(-depthDiff * depthDiff / (u_blur.depthThreshold * u_blur.depthThreshold + EPSILON));

        let combinedW    = w * depthW;
        accumThickness  += sampleThickness * combinedW;
        totalWeight     += combinedW;
    }

    let result = select(centerThickness, accumThickness / totalWeight, totalWeight > EPSILON);
    textureStore(t_output, center, vec4f(result, 0.0, 0.0, 0.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 2: Transmittance + Absorption
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_TRANSMITTANCE = /* wgsl */ `
${WGSL_SSS_COMMON}

struct TransmittanceUniforms {
    lightDir         : vec3f,      // view-space, normalised
    ambientScatter   : f32,
    lightColor       : vec3f,      // linear HDR
    curvatureBoost   : f32,
    resolution       : vec2f,
    nearPlane        : f32,
    farPlane         : f32,
    // inverse projection params for view-space reconstruction
    invProjX         : f32,
    invProjY         : f32,
    _pad             : vec2f,
}

// ── Species LUT ──────────────────────────────────────────────────────────────
// Each species occupies one texel row in a ${MAX_SPECIES}×2 RGBA32Float texture:
//   row 0: absorption.rgb + scatterWidth
//   row 1: scatterTint.rgb + translucency
//   row 2: wrapFactor, forwardScatterPower, forwardScatterStrength, 0
struct SpeciesSSS {
    absorption              : vec3f,
    scatterWidth            : f32,
    scatterTint             : vec3f,
    translucency            : f32,
    wrapFactor              : f32,
    forwardScatterPower     : f32,
    forwardScatterStrength  : f32,
}

fn loadSpeciesSSS(speciesId: u32) -> SpeciesSSS {
    var sss: SpeciesSSS;
    let row0 = textureLoad(t_speciesLUT, vec2i(i32(speciesId), 0), 0);
    let row1 = textureLoad(t_speciesLUT, vec2i(i32(speciesId), 1), 0);
    let row2 = textureLoad(t_speciesLUT, vec2i(i32(speciesId), 2), 0);
    sss.absorption             = row0.rgb;
    sss.scatterWidth           = row0.a;
    sss.scatterTint            = row1.rgb;
    sss.translucency           = row1.a;
    sss.wrapFactor             = row2.r;
    sss.forwardScatterPower    = row2.g;
    sss.forwardScatterStrength = row2.b;
    return sss;
}

@group(0) @binding(0) var<uniform>  u_trans      : TransmittanceUniforms;
@group(0) @binding(1) var           t_thickness  : texture_2d<f32>;  // blurred thickness
@group(0) @binding(2) var           t_normal     : texture_2d<f32>;  // view-space normals
@group(0) @binding(3) var           t_depth      : texture_2d<f32>;  // front-face depth
@group(0) @binding(4) var           t_speciesId  : texture_2d<u32>;  // per-pixel species index
@group(0) @binding(5) var           t_speciesLUT : texture_2d<f32>;  // ${MAX_SPECIES}×3 RGBA32F
@group(0) @binding(6) var           t_output     : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WG_SIZE}, ${WG_SIZE})
fn cs_transmittance(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2u(u_trans.resolution);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coord = vec2i(gid.xy);

    // ── Read inputs ──────────────────────────────────────────────────────────
    let thickness = textureLoad(t_thickness, coord, 0).r;
    let N         = normalize(textureLoad(t_normal, coord, 0).rgb * 2.0 - 1.0);
    let speciesId = textureLoad(t_speciesId, coord, 0).r;

    // Skip background pixels (species 0xFFFFFFFF or thickness ≈ 0)
    if (speciesId >= ${MAX_SPECIES}u || thickness < EPSILON) {
        textureStore(t_output, coord, vec4f(0.0));
        return;
    }

    let sss = loadSpeciesSSS(speciesId);

    // ── View vector (reconstruct from UV + depth) ────────────────────────────
    let uv      = (vec2f(gid.xy) + 0.5) / u_trans.resolution;
    let depthRaw = textureLoad(t_depth, coord, 0).r;
    let linearZ  = linearizeDepth(depthRaw, u_trans.nearPlane, u_trans.farPlane);
    let ndcX     = uv.x * 2.0 - 1.0;
    let ndcY     = (1.0 - uv.y) * 2.0 - 1.0;
    let viewPos  = vec3f(ndcX * u_trans.invProjX * linearZ,
                         ndcY * u_trans.invProjY * linearZ,
                         linearZ);
    let V = normalize(-viewPos);
    let L = normalize(u_trans.lightDir);

    // ── Beer-Lambert transmittance ───────────────────────────────────────────
    let T = beerLambert(sss.absorption, thickness);

    // ── Wrap lighting (diffuse with hemisphere wrap) ─────────────────────────
    // NdotL with wrap: (dot(N,L) + wrap) / (1 + wrap)
    // Using -N for back-face lighting (light penetrating from behind)
    let backN    = -N;
    let NdotL    = dot(backN, L);
    let wrapNdotL = saturate_f((NdotL + sss.wrapFactor) / (1.0 + sss.wrapFactor));

    // ── Forward scatter (view-dependent translucency) ────────────────────────
    // VdotL: when viewer looks into the light through the object → max scatter
    let VdotL        = saturate_f(dot(-V, L));
    let forwardScat  = pow(VdotL, sss.forwardScatterPower) * sss.forwardScatterStrength;

    // ── Curvature-enhanced scatter ───────────────────────────────────────────
    // Convex regions (normal facing away from viewer) get boosted SSS because
    // they represent thin shell edges where more light passes through.
    let curvatureTerm = (1.0 - saturate_f(dot(N, V))) * u_trans.curvatureBoost;

    // ── Ambient scatter (view-independent minimum) ───────────────────────────
    let ambientTerm = u_trans.ambientScatter;

    // ── Combine ──────────────────────────────────────────────────────────────
    let scatterIntensity = wrapNdotL + forwardScat + curvatureTerm + ambientTerm;
    let sssColor = u_trans.lightColor * T * sss.scatterTint * scatterIntensity * sss.translucency;

    textureStore(t_output, coord, vec4f(sssColor, 1.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 3: Composite
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_COMPOSITE = /* wgsl */ `
struct CompositeUniforms {
    globalIntensity : f32,
    resolution      : vec2f,
    _pad            : f32,
}

@group(0) @binding(0) var<uniform>  u_comp  : CompositeUniforms;
@group(0) @binding(1) var           t_scene : texture_2d<f32>;
@group(0) @binding(2) var           t_sss   : texture_2d<f32>;
@group(0) @binding(3) var           t_output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WG_SIZE}, ${WG_SIZE})
fn cs_composite(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2u(u_comp.resolution);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coord = vec2i(gid.xy);
    let scene = textureLoad(t_scene, coord, 0);
    let sss   = textureLoad(t_sss,   coord, 0);

    // Additive blend: SSS light is additional transmitted light
    let result = scene.rgb + sss.rgb * u_comp.globalIntensity;

    textureStore(t_output, coord, vec4f(result, scene.a));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Species LUT Builder
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Order in which CellSpecies values are packed into the species LUT texture.
 * Index 0 = attention, 1 = ffn, etc.  Must stay in sync with the species ID
 * written into speciesIdTex by the G-buffer pass.
 */
export const SPECIES_LUT_ORDER: readonly CellSpecies[] = [
  'attention',
  'ffn',
  'layernorm',
  'embedding',
  'softmax',
] as const;

/**
 * Build the CPU-side Float32Array for the species SSS lookup texture.
 *
 * Layout: ${MAX_SPECIES} columns × 3 rows × 4 channels (RGBA32Float)
 *   row 0: absorption.rgb + scatterWidth
 *   row 1: scatterTint.rgb + translucency
 *   row 2: wrapFactor, forwardScatterPower, forwardScatterStrength, 0
 *
 * @returns Float32Array of size MAX_SPECIES × 3 × 4
 */
export function buildSpeciesLUTData(): Float32Array {
  const data = new Float32Array(MAX_SPECIES * 3 * 4);

  for (let i = 0; i < SPECIES_LUT_ORDER.length; i++) {
    const species = SPECIES_LUT_ORDER[i];
    const profile = SSS_PROFILE_REGISTRY[species];
    const base    = i * 4; // column offset (one texel = 4 floats)

    // Row 0 (y=0): absorption.rgb + scatterWidth
    const r0 = 0 * MAX_SPECIES * 4 + base;
    data[r0 + 0] = profile.absorption[0];
    data[r0 + 1] = profile.absorption[1];
    data[r0 + 2] = profile.absorption[2];
    data[r0 + 3] = profile.scatterWidth;

    // Row 1 (y=1): scatterTint.rgb + translucency
    const r1 = 1 * MAX_SPECIES * 4 + base;
    data[r1 + 0] = profile.scatterTint[0];
    data[r1 + 1] = profile.scatterTint[1];
    data[r1 + 2] = profile.scatterTint[2];
    data[r1 + 3] = profile.translucency;

    // Row 2 (y=2): wrapFactor, forwardScatterPower, forwardScatterStrength, 0
    const r2 = 2 * MAX_SPECIES * 4 + base;
    data[r2 + 0] = profile.wrapFactor;
    data[r2 + 1] = profile.forwardScatterPower;
    data[r2 + 2] = profile.forwardScatterStrength;
    data[r2 + 3] = 0.0; // padding
  }

  return data;
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian Kernel Builder (CPU-side, for potential uniform upload)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a 1D Gaussian kernel with the given sigma.
 *
 * @param sigma   Standard deviation in pixels.
 * @param maxTaps Maximum number of taps on one side (total = 2*maxTaps + 1).
 * @returns       Normalised weight array of length 2*radius + 1.
 */
export function buildGaussianKernel(sigma: number, maxTaps: number = MAX_BLUR_TAPS): Float32Array {
  const radius  = Math.min(Math.ceil(sigma * 2.5), maxTaps);
  const size    = 2 * radius + 1;
  const weights = new Float32Array(size);
  let sum       = 0;

  for (let i = 0; i < size; i++) {
    const offset = i - radius;
    const w      = Math.exp(-(offset * offset) / (2 * sigma * sigma + 1e-6));
    weights[i]   = w;
    sum         += w;
  }

  // Normalise
  for (let i = 0; i < size; i++) {
    weights[i] /= sum;
  }

  return weights;
}

// ─────────────────────────────────────────────────────────────────────────────
// SSSPass — WebGPU compute pipeline orchestrator
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Intermediate GPU resources created and owned by SSSPass.
 */
interface SSSResources {
  thicknessTex:     GPUTexture;
  thicknessView:    GPUTextureView;
  diffuseTempTex:   GPUTexture;   // horizontal blur intermediate
  diffuseTempView:  GPUTextureView;
  diffusedTex:      GPUTexture;   // final blurred thickness
  diffusedView:     GPUTextureView;
  sssTex:           GPUTexture;
  sssView:          GPUTextureView;
  speciesLUTTex:    GPUTexture;
  speciesLUTView:   GPUTextureView;
}

/**
 * SSSPass — full subsurface scattering post-process pipeline.
 *
 * Manages four compute passes (thickness → blur H → blur V → transmittance)
 * plus a final composite pass. All intermediate textures are owned by the pass
 * and resized on demand.
 *
 * Usage:
 *   const sss = await SSSPass.create(device, width, height, format);
 *   sss.setParams({ globalIntensity: 1.2, lightDir: [0.3, 0.8, 0.5] });
 *   // Per frame:
 *   sss.dispatch(encoder, {
 *     frontDepthView, backDepthView, normalView, speciesIdView,
 *     sceneView, outputView, cameraUBO,
 *   });
 */
export class SSSPass {
  private device:    GPUDevice;
  private width:     number;
  private height:    number;

  // Pipelines
  private thicknessPipeline:     GPUComputePipeline | null = null;
  private blurPipeline:          GPUComputePipeline | null = null;
  private transmittancePipeline: GPUComputePipeline | null = null;
  private compositePipeline:     GPUComputePipeline | null = null;

  // Uniform buffers
  private thicknessUBO:     GPUBuffer | null = null;
  private blurHUBO:         GPUBuffer | null = null;
  private blurVUBO:         GPUBuffer | null = null;
  private transmittanceUBO: GPUBuffer | null = null;
  private compositeUBO:     GPUBuffer | null = null;

  // Internal textures
  private res: SSSResources | null = null;

  // Current params
  private params: SSSParams = { ...DEFAULT_SSS_PARAMS };

  private constructor(device: GPUDevice, width: number, height: number) {
    this.device = device;
    this.width  = width;
    this.height = height;
  }

  /**
   * Create and initialise the SSSPass.
   *
   * @param device  WebGPU device.
   * @param width   Render target width in pixels.
   * @param height  Render target height in pixels.
   * @param format  Swap-chain / output texture format (unused directly by compute,
   *                but kept for API symmetry with other post-process passes).
   */
  static async create(
    device: GPUDevice,
    width: number,
    height: number,
    _format?: GPUTextureFormat,
  ): Promise<SSSPass> {
    const pass = new SSSPass(device, width, height);
    pass.initPipelines();
    pass.initResources();
    pass.initUBOs();
    pass.uploadSpeciesLUT();
    return pass;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /** Update runtime SSS parameters. Partial update supported. */
  setParams(partial: Partial<SSSParams>): void {
    Object.assign(this.params, partial);
    this.writeUBOs();
  }

  /** Get a copy of the current params. */
  getParams(): SSSParams {
    return { ...this.params };
  }

  /** Resize internal textures when the viewport changes. */
  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width  = width;
    this.height = height;
    this.destroyResources();
    this.initResources();
    this.writeUBOs();
  }

  /**
   * Dispatch all SSS compute passes.
   *
   * @param encoder         Active GPUCommandEncoder.
   * @param inputs          Required texture views from the G-buffer / scene.
   * @param inputs.frontDepthView  Front-face depth buffer view.
   * @param inputs.backDepthView   Back-face depth buffer view (separate pass).
   * @param inputs.normalView      View-space normal G-buffer view.
   * @param inputs.speciesIdView   Per-pixel species index (r32uint).
   * @param inputs.sceneView       Lit scene colour view (pre-SSS).
   * @param inputs.outputView      Output texture storage view (rgba16float).
   * @param inputs.invProjX        1/projection[0][0] for view-space reconstruction.
   * @param inputs.invProjY        1/projection[1][1] for view-space reconstruction.
   */
  dispatch(
    encoder: GPUCommandEncoder,
    inputs: {
      frontDepthView: GPUTextureView;
      backDepthView:  GPUTextureView;
      normalView:     GPUTextureView;
      speciesIdView:  GPUTextureView;
      sceneView:      GPUTextureView;
      outputView:     GPUTextureView;
      invProjX:       number;
      invProjY:       number;
    },
  ): void {
    if (!this.res || !this.thicknessPipeline || !this.blurPipeline ||
        !this.transmittancePipeline || !this.compositePipeline) {
      return;
    }

    const wgX = Math.ceil(this.width  / WG_SIZE);
    const wgY = Math.ceil(this.height / WG_SIZE);

    // Write invProj into transmittance UBO (may change per frame)
    this.writeTransmittanceInvProj(inputs.invProjX, inputs.invProjY);

    // ── Pass 0: Thickness Map ────────────────────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        layout: this.thicknessPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.thicknessUBO! } },
          { binding: 1, resource: inputs.frontDepthView },
          { binding: 2, resource: inputs.backDepthView },
          { binding: 3, resource: this.res.thicknessView },
        ],
      });

      const cp = encoder.beginComputePass({ label: 'SSS Pass 0 — Thickness Map' });
      cp.setPipeline(this.thicknessPipeline);
      cp.setBindGroup(0, bg);
      cp.dispatchWorkgroups(wgX, wgY);
      cp.end();
    }

    // ── Pass 1a: Blur Horizontal ─────────────────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        layout: this.blurPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.blurHUBO! } },
          { binding: 1, resource: this.res.thicknessView },
          { binding: 2, resource: inputs.frontDepthView },
          { binding: 3, resource: this.res.diffuseTempView },
        ],
      });

      const cp = encoder.beginComputePass({ label: 'SSS Pass 1a — Blur H' });
      cp.setPipeline(this.blurPipeline);
      cp.setBindGroup(0, bg);
      cp.dispatchWorkgroups(wgX, wgY);
      cp.end();
    }

    // ── Pass 1b: Blur Vertical ───────────────────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        layout: this.blurPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.blurVUBO! } },
          { binding: 1, resource: this.res.diffuseTempView },
          { binding: 2, resource: inputs.frontDepthView },
          { binding: 3, resource: this.res.diffusedView },
        ],
      });

      const cp = encoder.beginComputePass({ label: 'SSS Pass 1b — Blur V' });
      cp.setPipeline(this.blurPipeline);
      cp.setBindGroup(0, bg);
      cp.dispatchWorkgroups(wgX, wgY);
      cp.end();
    }

    // ── Pass 2: Transmittance + Absorption ───────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        layout: this.transmittancePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.transmittanceUBO! } },
          { binding: 1, resource: this.res.diffusedView },
          { binding: 2, resource: inputs.normalView },
          { binding: 3, resource: inputs.frontDepthView },
          { binding: 4, resource: inputs.speciesIdView },
          { binding: 5, resource: this.res.speciesLUTView },
          { binding: 6, resource: this.res.sssView },
        ],
      });

      const cp = encoder.beginComputePass({ label: 'SSS Pass 2 — Transmittance' });
      cp.setPipeline(this.transmittancePipeline);
      cp.setBindGroup(0, bg);
      cp.dispatchWorkgroups(wgX, wgY);
      cp.end();
    }

    // ── Pass 3: Composite ────────────────────────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        layout: this.compositePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.compositeUBO! } },
          { binding: 1, resource: inputs.sceneView },
          { binding: 2, resource: this.res.sssView },
          { binding: 3, resource: inputs.outputView },
        ],
      });

      const cp = encoder.beginComputePass({ label: 'SSS Pass 3 — Composite' });
      cp.setPipeline(this.compositePipeline);
      cp.setBindGroup(0, bg);
      cp.dispatchWorkgroups(wgX, wgY);
      cp.end();
    }
  }

  /** Release all GPU resources. */
  destroy(): void {
    this.destroyResources();
    this.thicknessUBO?.destroy();
    this.blurHUBO?.destroy();
    this.blurVUBO?.destroy();
    this.transmittanceUBO?.destroy();
    this.compositeUBO?.destroy();
    this.thicknessUBO     = null;
    this.blurHUBO         = null;
    this.blurVUBO         = null;
    this.transmittanceUBO = null;
    this.compositeUBO     = null;
  }

  // ── Internal: Pipeline Creation ───────────────────────────────────────────

  private initPipelines(): void {
    // Pass 0: Thickness
    this.thicknessPipeline = this.device.createComputePipeline({
      label:   'SSS — Thickness Pipeline',
      layout:  'auto',
      compute: {
        module:     this.device.createShaderModule({ code: WGSL_THICKNESS_MAP, label: 'SSS Thickness Shader' }),
        entryPoint: 'cs_thickness',
      },
    });

    // Pass 1: Diffusion Blur (shared for H and V)
    this.blurPipeline = this.device.createComputePipeline({
      label:   'SSS — Diffusion Blur Pipeline',
      layout:  'auto',
      compute: {
        module:     this.device.createShaderModule({ code: WGSL_DIFFUSION_BLUR, label: 'SSS Blur Shader' }),
        entryPoint: 'cs_diffusion_blur',
      },
    });

    // Pass 2: Transmittance
    this.transmittancePipeline = this.device.createComputePipeline({
      label:   'SSS — Transmittance Pipeline',
      layout:  'auto',
      compute: {
        module:     this.device.createShaderModule({ code: WGSL_TRANSMITTANCE, label: 'SSS Transmittance Shader' }),
        entryPoint: 'cs_transmittance',
      },
    });

    // Pass 3: Composite
    this.compositePipeline = this.device.createComputePipeline({
      label:   'SSS — Composite Pipeline',
      layout:  'auto',
      compute: {
        module:     this.device.createShaderModule({ code: WGSL_COMPOSITE, label: 'SSS Composite Shader' }),
        entryPoint: 'cs_composite',
      },
    });
  }

  // ── Internal: Resource Creation ───────────────────────────────────────────

  private initResources(): void {
    const { device, width, height } = this;

    const mkTex = (label: string, format: GPUTextureFormat, w = width, h = height): GPUTexture =>
      device.createTexture({
        label,
        size:   [w, h],
        format,
        usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      });

    const thicknessTex   = mkTex('SSS thicknessTex',   'r16float');
    const diffuseTempTex = mkTex('SSS diffuseTempTex', 'r16float');
    const diffusedTex    = mkTex('SSS diffusedTex',    'r16float');
    const sssTex         = mkTex('SSS sssTex',         'rgba16float');

    // Species LUT: MAX_SPECIES columns × 3 rows
    const speciesLUTTex  = device.createTexture({
      label:  'SSS speciesLUT',
      size:   [MAX_SPECIES, 3],
      format: 'rgba32float',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.res = {
      thicknessTex,
      thicknessView:    thicknessTex.createView(),
      diffuseTempTex,
      diffuseTempView:  diffuseTempTex.createView(),
      diffusedTex,
      diffusedView:     diffusedTex.createView(),
      sssTex,
      sssView:          sssTex.createView(),
      speciesLUTTex,
      speciesLUTView:   speciesLUTTex.createView(),
    };
  }

  private destroyResources(): void {
    if (!this.res) return;
    this.res.thicknessTex.destroy();
    this.res.diffuseTempTex.destroy();
    this.res.diffusedTex.destroy();
    this.res.sssTex.destroy();
    this.res.speciesLUTTex.destroy();
    this.res = null;
  }

  // ── Internal: Uniform Buffer Creation ─────────────────────────────────────

  private initUBOs(): void {
    const mkUBO = (label: string, size: number): GPUBuffer =>
      this.device.createBuffer({
        label,
        size:  Math.ceil(size / 16) * 16, // 16-byte alignment
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

    // ThicknessUniforms: nearPlane(4) + farPlane(4) + thicknessScale(4) + pad(4) + resolution(8) + pad(8) = 32
    this.thicknessUBO     = mkUBO('SSS thicknessUBO', 32);
    // BlurUniforms: direction(8) + baseRadius(4) + depthThreshold(4) + nearPlane(4) + farPlane(4) + resolution(8) = 32
    this.blurHUBO         = mkUBO('SSS blurH UBO', 32);
    this.blurVUBO         = mkUBO('SSS blurV UBO', 32);
    // TransmittanceUniforms: lightDir(12) + ambientScatter(4) + lightColor(12) + curvatureBoost(4) + resolution(8) + near(4) + far(4) + invProjX(4) + invProjY(4) + pad(8) = 64
    this.transmittanceUBO = mkUBO('SSS transmittanceUBO', 64);
    // CompositeUniforms: globalIntensity(4) + resolution(8) + pad(4) = 16
    this.compositeUBO     = mkUBO('SSS compositeUBO', 16);

    this.writeUBOs();
  }

  private writeUBOs(): void {
    const { device, params, width, height } = this;

    // Thickness UBO
    {
      const d = new Float32Array([
        params.nearPlane, params.farPlane, THICKNESS_SCALE, 0, // pad
        width, height, 0, 0, // resolution + pad
      ]);
      device.queue.writeBuffer(this.thicknessUBO!, 0, d);
    }

    // Blur H UBO
    {
      const d = new Float32Array([
        1, 0,                       // direction horizontal
        params.baseBlurRadius,
        params.depthThreshold,
        params.nearPlane,
        params.farPlane,
        width, height,              // resolution
      ]);
      device.queue.writeBuffer(this.blurHUBO!, 0, d);
    }

    // Blur V UBO
    {
      const d = new Float32Array([
        0, 1,                       // direction vertical
        params.baseBlurRadius,
        params.depthThreshold,
        params.nearPlane,
        params.farPlane,
        width, height,              // resolution
      ]);
      device.queue.writeBuffer(this.blurVUBO!, 0, d);
    }

    // Transmittance UBO (invProjX/Y written per-frame via writeTransmittanceInvProj)
    {
      const L = params.lightDir;
      const len = Math.sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]) || 1;
      const d = new Float32Array([
        L[0] / len, L[1] / len, L[2] / len,   // lightDir (normalised)
        params.ambientScatter,
        params.lightColor[0], params.lightColor[1], params.lightColor[2],
        params.curvatureBoost,
        width, height,                          // resolution
        params.nearPlane, params.farPlane,
        1.0, 1.0,                               // invProjX, invProjY (placeholder)
        0, 0,                                   // pad
      ]);
      device.queue.writeBuffer(this.transmittanceUBO!, 0, d);
    }

    // Composite UBO
    {
      const d = new Float32Array([
        params.globalIntensity,
        width, height,
        0, // pad
      ]);
      device.queue.writeBuffer(this.compositeUBO!, 0, d);
    }
  }

  /** Write per-frame inverse projection params into the transmittance UBO. */
  private writeTransmittanceInvProj(invProjX: number, invProjY: number): void {
    const d = new Float32Array([invProjX, invProjY]);
    // Offset: 12 floats in = byte 48
    this.device.queue.writeBuffer(this.transmittanceUBO!, 48, d);
  }

  // ── Internal: Species LUT Upload ──────────────────────────────────────────

  private uploadSpeciesLUT(): void {
    if (!this.res) return;
    const data = buildSpeciesLUTData();
    this.device.queue.writeTexture(
      { texture: this.res.speciesLUTTex },
      data,
      { bytesPerRow: MAX_SPECIES * 4 * 4, rowsPerImage: 3 }, // 4 channels × 4 bytes/float
      { width: MAX_SPECIES, height: 3 },
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Physics-Driven SSS Modulation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Modulated SSS parameters for a single cell, derived from its SSSProfile +
 * live PhysicsUniforms.
 *
 * Consumers (e.g. per-cell material uniform writers) can use this to adjust
 * the SSS contribution per-cell based on the cell's local physics environment.
 */
export interface ModulatedSSS {
  /** Effective translucency after physics modulation. */
  translucency: number;

  /** Effective scatter width after physics modulation. */
  scatterWidth: number;

  /** Effective absorption (may be shifted by density). */
  absorption: [number, number, number];
}

/**
 * Compute physics-modulated SSS parameters for a given species and physics state.
 *
 * Modulation rules:
 *   - Density drives absorption: higher fluid density around the cell → denser
 *     medium → increased absorption (thicker effective path length).
 *   - Pressure modulates translucency: compressed cells become thinner / more
 *     translucent (membrane stretching under pressure).
 *   - Kinetic energy modulates scatter width: fast-moving cells exhibit broader
 *     scatter due to motion blur analogy (dynamic diffusion).
 *   - Neighbor count boosts translucency: cells in dense clusters scatter more
 *     light internally (multiple-scattering approximation).
 *
 * @param species  Target species identifier.
 * @param physics  Live PhysicsUniforms snapshot.
 * @param restDensity  World rest density for normalisation.
 * @returns Modulated SSS parameters for uniform upload.
 */
export function modulateSSSByPhysics(
  species: CellSpecies,
  physics: PhysicsUniforms,
  restDensity: number = 1000,
): ModulatedSSS {
  const profile = SSS_PROFILE_REGISTRY[species];

  // Normalised density ratio (0 = vacuum, 1 = rest, >1 = compressed)
  const densityRatio = restDensity > 0
    ? physics.u_density / restDensity
    : 1.0;

  // Normalised speed (magnitude of velocity / reference max speed)
  const MAX_SPEED = 10.0;
  const [vx, vy]  = physics.u_velocity;
  const speed     = Math.sqrt(vx * vx + vy * vy) / MAX_SPEED;

  // Normalised neighbor influence (0 = isolated, 1 = crowded)
  const MAX_NEIGHBORS = 64;
  const neighborRatio = Math.min(physics.u_neighborCount / MAX_NEIGHBORS, 1.0);

  // ── Absorption modulation ──────────────────────────────────────────────────
  // Dense fluid → higher effective absorption (Beer-Lambert path is longer)
  const densityAbsorptionScale = 0.8 + 0.4 * clamp01(densityRatio);
  const absorption: [number, number, number] = [
    profile.absorption[0] * densityAbsorptionScale,
    profile.absorption[1] * densityAbsorptionScale,
    profile.absorption[2] * densityAbsorptionScale,
  ];

  // ── Translucency modulation ────────────────────────────────────────────────
  // Pressure → membrane stretch → more translucent
  const pressureNorm  = clamp01(physics.u_pressure / 500); // normalise to sensible range
  const pressureBoost = 1.0 + pressureNorm * 0.15;

  // Neighbor cluster → multiple scattering → slight translucency boost
  const neighborBoost = 1.0 + neighborRatio * 0.10;

  const translucency = clamp01(profile.translucency * pressureBoost * neighborBoost);

  // ── Scatter width modulation ───────────────────────────────────────────────
  // Speed → dynamic broadening of scatter kernel
  const speedBroadening = 1.0 + speed * 0.3;
  const scatterWidth    = profile.scatterWidth * speedBroadening;

  return { translucency, scatterWidth, absorption };
}

// ─────────────────────────────────────────────────────────────────────────────
// SSS-Enhanced Material Compositing (CPU-side soft-body shader helpers)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compute a CPU-side approximation of the subsurface-scattered light colour
 * for a given species + thickness + light angle.
 *
 * This is useful for:
 *   - Preview renderers that don't run the full GPU pipeline.
 *   - 2D canvas fallback rendering with SSS-tinted fill colours.
 *   - Debug visualisation overlays.
 *
 * @param species     Target species.
 * @param thickness   Estimated thickness in world units (e.g. from bounding sphere diameter).
 * @param NdotL       Cosine of angle between surface normal and light direction.
 * @param VdotL       Cosine of angle between view direction and light direction.
 * @param lightColor  RGB light colour (linear).
 * @returns           RGB scattered light colour (linear, may exceed 1.0 for HDR).
 */
export function cpuSubsurfaceScatter(
  species:    CellSpecies,
  thickness:  number,
  NdotL:      number,
  VdotL:      number,
  lightColor: [number, number, number] = [1.0, 0.98, 0.92],
): [number, number, number] {
  const p = SSS_PROFILE_REGISTRY[species];

  // Beer-Lambert transmittance
  const T: [number, number, number] = [
    Math.exp(-p.absorption[0] * Math.max(thickness, 0)),
    Math.exp(-p.absorption[1] * Math.max(thickness, 0)),
    Math.exp(-p.absorption[2] * Math.max(thickness, 0)),
  ];

  // Wrap lighting (using back-face normal, so negate NdotL)
  const backNdotL = -NdotL;
  const wrap      = clamp01((backNdotL + p.wrapFactor) / (1 + p.wrapFactor));

  // Forward scatter
  const fwd = Math.pow(clamp01(VdotL), p.forwardScatterPower) * p.forwardScatterStrength;

  // Combine
  const intensity = wrap + fwd;

  return [
    lightColor[0] * T[0] * p.scatterTint[0] * intensity * p.translucency,
    lightColor[1] * T[1] * p.scatterTint[1] * intensity * p.translucency,
    lightColor[2] * T[2] * p.scatterTint[2] * intensity * p.translucency,
  ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API Summary
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Look up the SSS profile for a given species.
 *
 * @param species  Target species identifier.
 * @returns        Readonly SSSProfile with the species' optical properties.
 *
 * @example
 *   const profile = getSSSProfile('embedding');
 *   // profile.scatterTint === [0.45, 0.92, 0.38]  — chlorophyll green
 */
export function getSSSProfile(species: CellSpecies): SSSProfile {
  return SSS_PROFILE_REGISTRY[species];
}

/**
 * Return all species identifiers that have non-trivial translucency
 * (translucency > 0.1), i.e. species that would visibly benefit from SSS.
 * Useful for render-loop optimisation: skip SSS passes when the scene contains
 * only opaque species.
 *
 * @example
 *   getTranslucentSpecies();
 *   // → ['attention', 'ffn', 'embedding', 'softmax']
 *   // layernorm (0.25) is included; its thin edges still scatter slightly.
 */
export function getTranslucentSpecies(): CellSpecies[] {
  return (Object.keys(SSS_PROFILE_REGISTRY) as CellSpecies[])
    .filter(s => SSS_PROFILE_REGISTRY[s].translucency > 0.1);
}

/**
 * Check whether a given species has enough translucency to warrant SSS.
 *
 * @param species  Target species.
 * @param minTranslucency  Threshold (default 0.1).
 * @returns true if the species' translucency exceeds the threshold.
 */
export function speciesNeedsSSS(species: CellSpecies, minTranslucency = 0.1): boolean {
  return SSS_PROFILE_REGISTRY[species].translucency > minTranslucency;
}
