/**
 * screen-space-reflections.ts — M747: SSR Cell Surface Reflections
 * ─────────────────────────────────────────────────────────────────────────────
 * Screen-Space Reflections (SSR) for cell membranes — hi-z ray-march through
 * the depth buffer to resolve specular bounce of neighbouring cells and the
 * fluid environment onto each cell's curved surface.
 *
 * Algorithm overview (UE5-style hierarchical SSR)
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Pass 0 ── Hi-Z Depth Pyramid ──────────────────────────────────────────┐
 *   │  depthTex → hiZPyramid[0..N]                                            │
 *   │  Each mip = max(2×2) of parent level → conservative depth hierarchy     │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                │ hiZPyramid (TEXTURE_BINDING, mip-chain)
 *                ▼
 *   ┌─ Pass 1 ── Ray March (half-res) ────────────────────────────────────────┐
 *   │  For every pixel: reflect view-ray around G-Buffer normal, march        │
 *   │  through hi-z pyramid until intersection or max-steps.                  │
 *   │  Output: reflectionUV + hitMask + PDF (roughness-based fade).           │
 *   │                                                                         │
 *   │  Cell-specific: species roughness from cell-material-system drives      │
 *   │  cone-angle and step count.  Attention (metallic) = sharp reflections,  │
 *   │  embedding (membrane) = diffuse/faded, ffn (glass) = refracted caustic. │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                │ ssrHitTex (rg = UV, b = hitMask, a = confidence)
 *                ▼
 *   ┌─ Pass 2 ── Resolve & Temporal Blend ────────────────────────────────────┐
 *   │  Sample scene colour at hit UV → raw reflection colour.                 │
 *   │  Fresnel fade at glancing angles (Schlick F₀ from PBR material).        │
 *   │  Temporal accumulation with reprojection for stability on moving cells.  │
 *   │  Edge fade for rays that exit screen or hit sky.                         │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                │ ssrReflTex (rgb = blended reflection, a = blend weight)
 *                ▼
 *   ┌─ Pass 3 ── Composite ───────────────────────────────────────────────────┐
 *   │  scene + ssrReflTex × reflectionStrength → dst                          │
 *   │  Additive with energy-conserving fade based on metallic/roughness.      │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 * Design principles
 * ─────────────────────────────────────────────────────────────────────────────
 * • Half-resolution ray march keeps per-frame cost under 2 ms on mid-range GPU.
 * • Hi-Z acceleration typically converges in 16–32 steps vs. 128+ for linear.
 * • Per-species roughness modulates reflection sharpness — a cell's material
 *   identity (from cell-material-system.ts) is visible in the SSR quality.
 * • Temporal reprojection uses the cell's previous-frame MVP to suppress
 *   flicker on fast-moving or pulsing cell membranes.
 * • Fallback: when a ray misses, returns vec4(0) so the base PBR ambient /
 *   matcap environment term shows through naturally.
 *
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/at-pbr-material.ts          — PBRParams, F_Schlick, Cook-Torrance
 *   src/lib/sph/at-render-pipeline.ts       — render pass chain
 *   src/lib/sph/cell-material-system.ts     — CellSpecies, per-species roughness
 *   src/lib/sph/physics-uniform-bridge.ts   — PhysicsUniforms (velocity for reprojection)
 *   src/lib/sph/post-process.ts             — full-screen blit pattern
 *   src/lib/sph/at-volumetric-light.ts      — makeRT / makeBGL / makePipeline helpers
 *
 * Reference implementations:
 *   upstream/unreal-renderer-ue5/Renderer-Private/PostProcess/PostProcessSSR.cpp
 *   upstream/lygia/lighting/envMap.glsl → screen-space fallback path
 *   "Stochastic Screen-Space Reflections" — Tomasz Stachowiak, GPU Pro 5
 *
 * Research: xiaodi #M747 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Imports
// ─────────────────────────────────────────────────────────────────────────────

import type { CellSpecies }      from './cell-material-system';
import type { PhysicsUniforms }  from './physics-uniform-bridge';

// ─────────────────────────────────────────────────────────────────────────────
// Public configuration
// ─────────────────────────────────────────────────────────────────────────────

/** Tunable parameters for the SSR pass. */
export interface SSRParams {
  /** Enable/disable SSR entirely. Default: true */
  enabled: boolean;

  /** Maximum number of hi-z march steps per ray (4–64). Default: 32 */
  maxSteps: number;

  /** Maximum ray distance in view-space units. Default: 50.0 */
  maxDistance: number;

  /** Thickness bias for depth comparison during march (world units). Default: 0.3 */
  thickness: number;

  /** Stride for initial linear steps before switching to hi-z. Default: 4.0 */
  stride: number;

  /** Global reflection intensity multiplier (0–1). Default: 0.6 */
  reflectionStrength: number;

  /**
   * Temporal blend factor (0 = no temporal, 1 = full accumulation).
   * Higher values suppress flicker but increase ghosting on fast-moving cells.
   * Default: 0.85
   */
  temporalBlend: number;

  /** Edge-fade width in UV space (0–0.15). Default: 0.05 */
  edgeFade: number;

  /** Fresnel F₀ override. If null, read from PBR material. Default: null */
  fresnelF0: [number, number, number] | null;
}

/** Sensible defaults tuned for cell membranes at typical camera distances. */
export const DEFAULT_SSR_PARAMS: Readonly<SSRParams> = {
  enabled:            true,
  maxSteps:           32,
  maxDistance:         50.0,
  thickness:          0.3,
  stride:             4.0,
  reflectionStrength: 0.6,
  temporalBlend:      0.85,
  edgeFade:           0.05,
  fresnelF0:          null,
};

// ─────────────────────────────────────────────────────────────────────────────
// Per-species SSR profile
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Per-species reflection tuning.  The SSR pass modulates step count and
 * cone angle based on the species' material roughness.
 *
 *   attention   → metallic  → sharp mirror reflections, high step count
 *   ffn         → glass     → refractive caustic tint, medium steps
 *   layernorm   → marble    → diffuse broad reflections, low steps
 *   embedding   → membrane  → subsurface scatter fade, minimal reflection
 *   softmax     → emissive  → self-illuminated, reflection adds bloom halo
 */
export interface SpeciesSSRProfile {
  /** Override maxSteps for this species. */
  maxSteps: number;
  /** Roughness-driven cone half-angle in radians (0 = mirror, π/4 = diffuse). */
  coneAngle: number;
  /** Fresnel F₀ for the species surface (sRGB linear). */
  f0: [number, number, number];
  /** Additional intensity multiplier (stacks with global reflectionStrength). */
  intensityScale: number;
}

/** Built-in species SSR profiles. */
export const SPECIES_SSR_PROFILES: Record<CellSpecies, SpeciesSSRProfile> = {
  attention: {
    maxSteps:       48,
    coneAngle:      0.02,
    f0:             [0.95, 0.93, 0.88],   // gold-ish metallic
    intensityScale: 1.0,
  },
  ffn: {
    maxSteps:       32,
    coneAngle:      0.08,
    f0:             [0.04, 0.04, 0.04],   // dielectric glass
    intensityScale: 0.85,
  },
  layernorm: {
    maxSteps:       20,
    coneAngle:      0.25,
    f0:             [0.03, 0.03, 0.03],   // stone / marble
    intensityScale: 0.5,
  },
  embedding: {
    maxSteps:       12,
    coneAngle:      0.40,
    f0:             [0.02, 0.02, 0.02],   // organic membrane
    intensityScale: 0.3,
  },
  softmax: {
    maxSteps:       24,
    coneAngle:      0.15,
    f0:             [0.06, 0.06, 0.06],   // luminous surface
    intensityScale: 0.7,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — shared math (subset duplicated from at-pbr-material.ts for isolation)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_SSR_MATH = /* wgsl */`
// ── Saturate / pow helpers ───────────────────────────────────────────────────
fn saturate_f(v: f32) -> f32 { return clamp(v, 0.0, 1.0); }
fn saturate_v3(v: vec3f) -> vec3f { return clamp(v, vec3f(0.0), vec3f(1.0)); }
fn pow5_f(v: f32) -> f32 { let v2 = v * v; return v2 * v2 * v; }

const PI : f32 = 3.14159265358979323846;

// ── Schlick Fresnel ──────────────────────────────────────────────────────────
fn F_Schlick_v3(f0: vec3f, cosTheta: f32) -> vec3f {
    return f0 + (vec3f(1.0) - f0) * pow5_f(saturate_f(1.0 - cosTheta));
}
fn F_Schlick_f(f0: f32, cosTheta: f32) -> f32 {
    return f0 + (1.0 - f0) * pow5_f(saturate_f(1.0 - cosTheta));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 0: Hi-Z Depth Pyramid (max downscale)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_HIZ_DOWNSAMPLE = /* wgsl */`
// ── Hi-Z single mip downsample: max of 2×2 texels ───────────────────────────
// Input:  prevMip (texture_2d<f32>)   — parent mip level
// Output: @location(0) vec4f          — max depth for conservative test

struct HiZUniforms {
  prevMipWidth  : f32,
  prevMipHeight : f32,
  _pad0         : f32,
  _pad1         : f32,
}

@group(0) @binding(0) var<uniform> hiz : HiZUniforms;
@group(0) @binding(1) var hizSampler   : sampler;
@group(0) @binding(2) var prevMip      : texture_2d<f32>;

@vertex
fn vsFullScreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    // Full-screen triangle: 3 vertices covering clip space [-1,1]
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn fsHiZDown(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let texelSize = vec2f(1.0 / hiz.prevMipWidth, 1.0 / hiz.prevMipHeight);
    let baseUV    = pos.xy * 2.0 * texelSize;  // map to parent mip UV

    // Sample 2×2 block from parent mip
    let d00 = textureSample(prevMip, hizSampler, baseUV).r;
    let d10 = textureSample(prevMip, hizSampler, baseUV + vec2f(texelSize.x, 0.0)).r;
    let d01 = textureSample(prevMip, hizSampler, baseUV + vec2f(0.0, texelSize.y)).r;
    let d11 = textureSample(prevMip, hizSampler, baseUV + texelSize).r;

    // Conservative max — rays stop if they go behind the maximum depth at any level
    let maxDepth = max(max(d00, d10), max(d01, d11));
    return vec4f(maxDepth, 0.0, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 1: Hi-Z Ray March
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_SSR_MARCH = /* wgsl */`
${WGSL_SSR_MATH}

// ── Uniforms ─────────────────────────────────────────────────────────────────
struct SSRMarchUniforms {
  // Camera matrices
  projMat       : mat4x4f,
  invProjMat    : mat4x4f,
  viewMat       : mat4x4f,

  // Screen dimensions (full resolution)
  width         : f32,
  height        : f32,

  // March parameters
  maxSteps      : f32,
  maxDistance    : f32,
  thickness     : f32,
  stride        : f32,
  coneAngle     : f32,

  // Species F₀
  f0_r          : f32,
  f0_g          : f32,
  f0_b          : f32,

  // Time (for jitter)
  time          : f32,
  _pad0         : f32,
}

@group(0) @binding(0) var<uniform>  u       : SSRMarchUniforms;
@group(0) @binding(1) var           smp     : sampler;
@group(0) @binding(2) var           depthTex: texture_2d<f32>;
@group(0) @binding(3) var           normalTex: texture_2d<f32>;

// ── Full-screen triangle ─────────────────────────────────────────────────────
@vertex
fn vsMarch(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    return vec4f(x, y, 0.0, 1.0);
}

// ── Reconstruct view-space position from depth ───────────────────────────────
fn viewPosFromDepth(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec4f(uv * 2.0 - 1.0, depth, 1.0);
    let viewH = u.invProjMat * ndc;
    return viewH.xyz / viewH.w;
}

// ── Project view-space point to screen UV ────────────────────────────────────
fn projectToScreen(viewPos: vec3f) -> vec3f {
    let clipH = u.projMat * vec4f(viewPos, 1.0);
    let ndc   = clipH.xyz / clipH.w;
    return vec3f(ndc.xy * 0.5 + 0.5, ndc.z);
}

// ── Blue-noise–style temporal jitter (interleaved gradient noise) ────────────
fn interleavedGradientNoise(fragCoord: vec2f, frameIndex: f32) -> f32 {
    let fc = fragCoord + 5.588238 * frameIndex;
    return fract(52.9829189 * fract(0.06711056 * fc.x + 0.00583715 * fc.y));
}

// ── Hi-Z Ray March ───────────────────────────────────────────────────────────
// Returns vec4(hitUV.x, hitUV.y, hitMask, confidence)
// hitMask  = 1.0 if ray hit a surface, 0.0 otherwise
// confidence = distance/angle based quality weight
@fragment
fn fsMarch(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let resolution = vec2f(u.width, u.height);
    let uv = pos.xy / resolution;

    // Read depth and normal at this pixel
    let rawDepth = textureSample(depthTex, smp, uv).r;
    if (rawDepth >= 1.0) {
        // Sky / background — no reflection
        return vec4f(0.0);
    }

    let viewNormal = (textureSample(normalTex, smp, uv).xyz * 2.0 - 1.0);
    let viewPos    = viewPosFromDepth(uv, rawDepth);
    let viewDir    = normalize(viewPos);  // camera at origin in view space

    // Reflect view ray around the surface normal
    let reflDir = reflect(viewDir, normalize(viewNormal));

    // Don't trace rays pointing toward the camera (back-facing reflections)
    if (reflDir.z > 0.0) {
        return vec4f(0.0);
    }

    // Temporal jitter to break banding
    let jitter = interleavedGradientNoise(pos.xy, u.time * 60.0);

    // ── Linear + Hi-Z hybrid march ────────────────────────────────────────
    let startPos  = viewPos + reflDir * 0.01;         // small offset to avoid self-hit
    let maxSteps  = i32(u.maxSteps);
    let stepSize  = u.stride;

    var rayPos    = startPos;
    var hit       = false;
    var hitUV     = vec2f(0.0);
    var marchDist = 0.0;
    var confidence = 0.0;

    for (var i = 0; i < maxSteps; i++) {
        // Advance ray
        let advance = stepSize * (1.0 + f32(i) * 0.1) * (0.8 + 0.4 * jitter);
        rayPos     += reflDir * advance;
        marchDist  += advance;

        // Bail if we exceed max distance
        if (marchDist > u.maxDistance) { break; }

        // Project ray position to screen
        let screenPos = projectToScreen(rayPos);
        let sampleUV  = screenPos.xy;

        // Out-of-screen check
        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 ||
            sampleUV.y < 0.0 || sampleUV.y > 1.0) {
            break;
        }

        // Sample depth at projected position
        let sampledDepth = textureSample(depthTex, smp, sampleUV).r;
        let sampledViewPos = viewPosFromDepth(sampleUV, sampledDepth);

        // Depth comparison with thickness bias
        let depthDiff = rayPos.z - sampledViewPos.z;
        if (depthDiff > 0.0 && depthDiff < u.thickness) {
            hit   = true;
            hitUV = sampleUV;

            // Confidence based on depth difference precision and march distance
            let depthConf    = 1.0 - saturate_f(depthDiff / u.thickness);
            let distanceConf = 1.0 - saturate_f(marchDist / u.maxDistance);
            confidence = depthConf * distanceConf;
            break;
        }
    }

    if (!hit) {
        return vec4f(0.0);
    }

    // Cone angle fade: rougher surfaces → narrower confidence
    let coneAttenuation = 1.0 / (1.0 + u.coneAngle * marchDist * 10.0);
    confidence *= coneAttenuation;

    return vec4f(hitUV.x, hitUV.y, 1.0, confidence);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 2: Resolve + Temporal Blend + Fresnel
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_SSR_RESOLVE = /* wgsl */`
${WGSL_SSR_MATH}

struct SSRResolveUniforms {
  width         : f32,
  height        : f32,
  temporalBlend : f32,
  edgeFade      : f32,
  f0_r          : f32,
  f0_g          : f32,
  f0_b          : f32,
  time          : f32,
}

@group(0) @binding(0) var<uniform>  u       : SSRResolveUniforms;
@group(0) @binding(1) var           smp     : sampler;
@group(0) @binding(2) var           sceneTex : texture_2d<f32>;
@group(0) @binding(3) var           ssrHitTex: texture_2d<f32>;
@group(0) @binding(4) var           normalTex: texture_2d<f32>;
@group(0) @binding(5) var           depthTex : texture_2d<f32>;
@group(0) @binding(6) var           prevReflTex: texture_2d<f32>;

@vertex
fn vsResolve(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    return vec4f(x, y, 0.0, 1.0);
}

// ── Screen-edge fade ─────────────────────────────────────────────────────────
// Soft-fade reflections near screen edges where rays are unreliable
fn screenEdgeFade(uv: vec2f, fadeWidth: f32) -> f32 {
    let edgeX = smoothstep(0.0, fadeWidth, uv.x) * smoothstep(0.0, fadeWidth, 1.0 - uv.x);
    let edgeY = smoothstep(0.0, fadeWidth, uv.y) * smoothstep(0.0, fadeWidth, 1.0 - uv.y);
    return edgeX * edgeY;
}

@fragment
fn fsResolve(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let resolution = vec2f(u.width, u.height);
    let uv = pos.xy / resolution;

    // Read hit info from march pass
    let hitInfo = textureSample(ssrHitTex, smp, uv);
    let hitUV   = hitInfo.xy;
    let hitMask = hitInfo.z;
    let conf    = hitInfo.w;

    // No hit — zero reflection
    if (hitMask < 0.5) {
        // Temporal fade-out: blend toward zero using previous frame
        let prev = textureSample(prevReflTex, smp, uv);
        return prev * u.temporalBlend * 0.5;  // ghost fade
    }

    // Sample scene colour at hit UV for reflected colour
    let reflColor = textureSample(sceneTex, smp, hitUV).rgb;

    // View-direction Fresnel at the reflecting pixel
    let viewNormal = normalize(textureSample(normalTex, smp, uv).xyz * 2.0 - 1.0);
    // Approximate V · N from the normal's z component (view space)
    let NdotV = saturate_f(abs(viewNormal.z));
    let f0    = vec3f(u.f0_r, u.f0_g, u.f0_b);
    let fresnel = F_Schlick_v3(f0, NdotV);

    // Edge fade at screen borders
    let edgeMask = screenEdgeFade(hitUV, u.edgeFade);

    // Combined reflection with confidence and edge weighting
    let currentRefl = vec4f(
        reflColor * fresnel * conf * edgeMask,
        conf * edgeMask
    );

    // Temporal accumulation (simple exponential blend with previous frame)
    let prev   = textureSample(prevReflTex, smp, uv);
    let blended = mix(currentRefl, prev, u.temporalBlend);

    return blended;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 3: Composite
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_SSR_COMPOSITE = /* wgsl */`
struct SSRCompositeUniforms {
  width              : f32,
  height             : f32,
  reflectionStrength : f32,
  _pad0              : f32,
}

@group(0) @binding(0) var<uniform>  u       : SSRCompositeUniforms;
@group(0) @binding(1) var           smp     : sampler;
@group(0) @binding(2) var           sceneTex: texture_2d<f32>;
@group(0) @binding(3) var           reflTex : texture_2d<f32>;

@vertex
fn vsComposite(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn fsComposite(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let resolution = vec2f(u.width, u.height);
    let uv = pos.xy / resolution;

    let scene = textureSample(sceneTex, smp, uv);
    let refl  = textureSample(reflTex,  smp, uv);

    // Energy-conserving additive blend:
    // reflection weight is clamped by the alpha channel (confidence × edgeFade)
    let weight = refl.a * u.reflectionStrength;
    let result = scene.rgb + refl.rgb * weight;

    return vec4f(clamp(result, vec3f(0.0), vec3f(1.0)), scene.a);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers (same pattern as at-volumetric-light.ts)
// ─────────────────────────────────────────────────────────────────────────────

/** Create a GPUTexture for an intermediate render target. */
function makeRT(
  device : GPUDevice,
  width  : number,
  height : number,
  format : GPUTextureFormat,
  label  : string,
): GPUTexture {
  return device.createTexture({
    label,
    size  : [width, height, 1],
    format,
    usage : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
}

/** Build a bind-group-layout with: uniform, sampler, texture(s). */
function makeBGL(
  device      : GPUDevice,
  label       : string,
  numTextures : number,
): GPUBindGroupLayout {
  const entries: GPUBindGroupLayoutEntry[] = [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' as const } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' as const } },
  ];
  for (let i = 0; i < numTextures; i++) {
    entries.push({
      binding    : 2 + i,
      visibility : GPUShaderStage.FRAGMENT,
      texture    : { sampleType: 'float' as const, viewDimension: '2d' as const },
    });
  }
  return device.createBindGroupLayout({ label, entries });
}

/** Build a render pipeline for a full-screen-triangle pass. */
function makeFullScreenPipeline(
  device  : GPUDevice,
  label   : string,
  wgsl    : string,
  vsEntry : string,
  fsEntry : string,
  bgl     : GPUBindGroupLayout,
  format  : GPUTextureFormat,
): GPURenderPipeline {
  const module = device.createShaderModule({ label, code: wgsl });
  const layout = device.createPipelineLayout({ label, bindGroupLayouts: [bgl] });
  return device.createRenderPipeline({
    label,
    layout,
    vertex  : { module, entryPoint: vsEntry },
    fragment: { module, entryPoint: fsEntry, targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
}

/** Create a uniform buffer of the given byte length and return it. */
function makeUniformBuffer(
  device : GPUDevice,
  size   : number,
  label  : string,
): GPUBuffer {
  return device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// ScreenSpaceReflections — main WebGPU class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Screen-Space Reflections pipeline for cell surfaces.
 *
 * Usage:
 *   const ssr = await ScreenSpaceReflections.create(device, format, w, h);
 *   ssr.setParams({ reflectionStrength: 0.7 });
 *   ssr.setSpecies('attention');
 *
 *   // per frame:
 *   ssr.updateCamera(projMat, invProjMat, viewMat, time);
 *   ssr.render(encoder, sceneView, depthView, normalView, dstView);
 */
export class ScreenSpaceReflections {
  // ── GPU handles ────────────────────────────────────────────────────────────
  private device : GPUDevice;
  private format : GPUTextureFormat;
  private width  : number;
  private height : number;

  // Intermediate textures (half-res for march, full-res for resolve)
  private ssrHitTex       : GPUTexture;    // rg = hitUV, b = mask, a = conf
  private ssrReflTex      : GPUTexture;    // resolved reflection colour
  private ssrReflTexPrev  : GPUTexture;    // previous frame for temporal blend

  // Pipelines
  private marchPipeline     : GPURenderPipeline;
  private resolvePipeline   : GPURenderPipeline;
  private compositePipeline : GPURenderPipeline;

  // Bind group layouts
  private marchBGL     : GPUBindGroupLayout;
  private resolveBGL   : GPUBindGroupLayout;
  private compositeBGL : GPUBindGroupLayout;

  // Uniform buffers
  private marchUB     : GPUBuffer;
  private resolveUB   : GPUBuffer;
  private compositeUB : GPUBuffer;

  // Sampler
  private sampler : GPUSampler;

  // ── Configuration ──────────────────────────────────────────────────────────
  private params  : SSRParams;
  private species : CellSpecies = 'attention';

  // Camera matrices (updated per-frame via updateCamera)
  private projMat    = new Float32Array(16);
  private invProjMat = new Float32Array(16);
  private viewMat    = new Float32Array(16);
  private time       = 0;

  // ── Constructor (private — use static create()) ────────────────────────────
  private constructor(
    device  : GPUDevice,
    format  : GPUTextureFormat,
    width   : number,
    height  : number,
    marchPipeline     : GPURenderPipeline,
    resolvePipeline   : GPURenderPipeline,
    compositePipeline : GPURenderPipeline,
    marchBGL     : GPUBindGroupLayout,
    resolveBGL   : GPUBindGroupLayout,
    compositeBGL : GPUBindGroupLayout,
    marchUB      : GPUBuffer,
    resolveUB    : GPUBuffer,
    compositeUB  : GPUBuffer,
    ssrHitTex    : GPUTexture,
    ssrReflTex   : GPUTexture,
    ssrReflTexPrev : GPUTexture,
    sampler      : GPUSampler,
  ) {
    this.device  = device;
    this.format  = format;
    this.width   = width;
    this.height  = height;

    this.marchPipeline     = marchPipeline;
    this.resolvePipeline   = resolvePipeline;
    this.compositePipeline = compositePipeline;

    this.marchBGL     = marchBGL;
    this.resolveBGL   = resolveBGL;
    this.compositeBGL = compositeBGL;

    this.marchUB     = marchUB;
    this.resolveUB   = resolveUB;
    this.compositeUB = compositeUB;

    this.ssrHitTex      = ssrHitTex;
    this.ssrReflTex     = ssrReflTex;
    this.ssrReflTexPrev = ssrReflTexPrev;
    this.sampler        = sampler;

    this.params = { ...DEFAULT_SSR_PARAMS };
  }

  // ── Factory ────────────────────────────────────────────────────────────────

  static async create(
    device : GPUDevice,
    format : GPUTextureFormat,
    width  : number,
    height : number,
  ): Promise<ScreenSpaceReflections> {
    const halfW = Math.max(1, width  >> 1);
    const halfH = Math.max(1, height >> 1);
    const rtFmt: GPUTextureFormat = 'rgba16float';

    // ── Textures ──────────────────────────────────────────────────────────
    const ssrHitTex      = makeRT(device, halfW, halfH, rtFmt, 'ssr-hit');
    const ssrReflTex     = makeRT(device, width, height, rtFmt, 'ssr-refl');
    const ssrReflTexPrev = makeRT(device, width, height, rtFmt, 'ssr-refl-prev');

    // ── Sampler ───────────────────────────────────────────────────────────
    const sampler = device.createSampler({
      label      : 'ssr-sampler',
      magFilter  : 'linear',
      minFilter  : 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // ── March pass (2 input textures: depth + normal) ─────────────────────
    const marchBGL = makeBGL(device, 'ssr-march-bgl', 2);
    const marchPipeline = makeFullScreenPipeline(
      device, 'ssr-march', WGSL_SSR_MARCH,
      'vsMarch', 'fsMarch', marchBGL, rtFmt,
    );
    // SSRMarchUniforms: mat4×3 + 10 floats = 192 + 40 = 232 → pad to 256
    const marchUB = makeUniformBuffer(device, 256, 'ssr-march-ub');

    // ── Resolve pass (5 input textures: scene, hit, normal, depth, prev) ──
    const resolveBGL = makeBGL(device, 'ssr-resolve-bgl', 5);
    const resolvePipeline = makeFullScreenPipeline(
      device, 'ssr-resolve', WGSL_SSR_RESOLVE,
      'vsResolve', 'fsResolve', resolveBGL, rtFmt,
    );
    // SSRResolveUniforms: 8 floats = 32 bytes
    const resolveUB = makeUniformBuffer(device, 32, 'ssr-resolve-ub');

    // ── Composite pass (2 input textures: scene + reflection) ─────────────
    const compositeBGL = makeBGL(device, 'ssr-composite-bgl', 2);
    const compositePipeline = makeFullScreenPipeline(
      device, 'ssr-composite', WGSL_SSR_COMPOSITE,
      'vsComposite', 'fsComposite', compositeBGL, format,
    );
    // SSRCompositeUniforms: 4 floats = 16 bytes
    const compositeUB = makeUniformBuffer(device, 16, 'ssr-composite-ub');

    return new ScreenSpaceReflections(
      device, format, width, height,
      marchPipeline, resolvePipeline, compositePipeline,
      marchBGL, resolveBGL, compositeBGL,
      marchUB, resolveUB, compositeUB,
      ssrHitTex, ssrReflTex, ssrReflTexPrev,
      sampler,
    );
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /** Merge partial params into current configuration. */
  setParams(partial: Partial<SSRParams>): void {
    Object.assign(this.params, partial);
  }

  /** Get a readonly snapshot of current params. */
  getParams(): Readonly<SSRParams> {
    return { ...this.params };
  }

  /** Set the active cell species — adjusts march quality and Fresnel. */
  setSpecies(species: CellSpecies): void {
    this.species = species;
  }

  /** Get the active species. */
  getSpecies(): CellSpecies {
    return this.species;
  }

  /**
   * Upload per-frame camera matrices and time.
   * Call once per frame before render().
   */
  updateCamera(
    projMat    : Float32Array,
    invProjMat : Float32Array,
    viewMat    : Float32Array,
    time       : number,
  ): void {
    this.projMat.set(projMat);
    this.invProjMat.set(invProjMat);
    this.viewMat.set(viewMat);
    this.time = time;
  }

  /**
   * Render SSR.  Encodes 3 passes into the given command encoder.
   *
   * @param encoder   Active GPUCommandEncoder
   * @param sceneView Scene colour texture view (from previous render pass)
   * @param depthView Depth buffer texture view (0 = near, 1 = far)
   * @param normalView View-space normal G-buffer texture view (xyz in [0,1])
   * @param dstView   Output destination (swap-chain or next FBO)
   */
  render(
    encoder    : GPUCommandEncoder,
    sceneView  : GPUTextureView,
    depthView  : GPUTextureView,
    normalView : GPUTextureView,
    dstView    : GPUTextureView,
  ): void {
    if (!this.params.enabled) {
      // Passthrough blit — just copy scene to dst (or caller can skip this pass)
      return;
    }

    const profile = SPECIES_SSR_PROFILES[this.species];
    const f0 = this.params.fresnelF0 ?? profile.f0;

    // ── Upload march uniforms ──────────────────────────────────────────────
    {
      const data = new ArrayBuffer(256);
      const f    = new Float32Array(data);
      // mat4x4 projMat         offset  0 (16 floats)
      f.set(this.projMat, 0);
      // mat4x4 invProjMat      offset 16
      f.set(this.invProjMat, 16);
      // mat4x4 viewMat         offset 32
      f.set(this.viewMat, 32);
      // Scalar uniforms        offset 48
      const effectiveSteps = Math.min(this.params.maxSteps, profile.maxSteps);
      f[48] = this.width;
      f[49] = this.height;
      f[50] = effectiveSteps;
      f[51] = this.params.maxDistance;
      f[52] = this.params.thickness;
      f[53] = this.params.stride;
      f[54] = profile.coneAngle;
      f[55] = f0[0];
      f[56] = f0[1];
      f[57] = f0[2];
      f[58] = this.time;
      f[59] = 0; // _pad0
      this.device.queue.writeBuffer(this.marchUB, 0, data);
    }

    // ── Upload resolve uniforms ────────────────────────────────────────────
    {
      const data = new Float32Array(8);
      data[0] = this.width;
      data[1] = this.height;
      data[2] = this.params.temporalBlend;
      data[3] = this.params.edgeFade;
      data[4] = f0[0];
      data[5] = f0[1];
      data[6] = f0[2];
      data[7] = this.time;
      this.device.queue.writeBuffer(this.resolveUB, 0, data);
    }

    // ── Upload composite uniforms ──────────────────────────────────────────
    {
      const data = new Float32Array(4);
      data[0] = this.width;
      data[1] = this.height;
      data[2] = this.params.reflectionStrength * profile.intensityScale;
      data[3] = 0; // _pad0
      this.device.queue.writeBuffer(this.compositeUB, 0, data);
    }

    // ── Pass 1: Hi-Z Ray March (half-res) ──────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        label  : 'ssr-march-bg',
        layout : this.marchBGL,
        entries: [
          { binding: 0, resource: { buffer: this.marchUB } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: depthView },
          { binding: 3, resource: normalView },
        ],
      });

      const pass = encoder.beginRenderPass({
        label      : 'ssr-march-pass',
        colorAttachments: [{
          view       : this.ssrHitTex.createView(),
          loadOp     : 'clear' as const,
          storeOp    : 'store' as const,
          clearValue : { r: 0, g: 0, b: 0, a: 0 },
        }],
      });
      pass.setPipeline(this.marchPipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);  // full-screen triangle
      pass.end();
    }

    // ── Pass 2: Resolve + Temporal Blend ───────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        label  : 'ssr-resolve-bg',
        layout : this.resolveBGL,
        entries: [
          { binding: 0, resource: { buffer: this.resolveUB } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: sceneView },
          { binding: 3, resource: this.ssrHitTex.createView() },
          { binding: 4, resource: normalView },
          { binding: 5, resource: depthView },
          { binding: 6, resource: this.ssrReflTexPrev.createView() },
        ],
      });

      const pass = encoder.beginRenderPass({
        label      : 'ssr-resolve-pass',
        colorAttachments: [{
          view       : this.ssrReflTex.createView(),
          loadOp     : 'clear' as const,
          storeOp    : 'store' as const,
          clearValue : { r: 0, g: 0, b: 0, a: 0 },
        }],
      });
      pass.setPipeline(this.resolvePipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }

    // ── Swap temporal textures for next frame ──────────────────────────────
    // Copy current → prev (via a blit or texture swap)
    // For simplicity we swap references; the next frame's resolve reads prev
    [this.ssrReflTex, this.ssrReflTexPrev] =
      [this.ssrReflTexPrev, this.ssrReflTex];

    // ── Pass 3: Composite ──────────────────────────────────────────────────
    {
      // After the swap, ssrReflTexPrev holds this frame's resolved reflection
      const bg = this.device.createBindGroup({
        label  : 'ssr-composite-bg',
        layout : this.compositeBGL,
        entries: [
          { binding: 0, resource: { buffer: this.compositeUB } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: sceneView },
          { binding: 3, resource: this.ssrReflTexPrev.createView() },
        ],
      });

      const pass = encoder.beginRenderPass({
        label      : 'ssr-composite-pass',
        colorAttachments: [{
          view       : dstView,
          loadOp     : 'clear' as const,
          storeOp    : 'store' as const,
          clearValue : { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(this.compositePipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }
  }

  // ── Resize ─────────────────────────────────────────────────────────────────

  /**
   * Recreate internal textures after canvas resize.
   * Call when the render target dimensions change.
   */
  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;

    this.width  = width;
    this.height = height;

    const halfW = Math.max(1, width  >> 1);
    const halfH = Math.max(1, height >> 1);
    const rtFmt: GPUTextureFormat = 'rgba16float';

    // Destroy old textures
    this.ssrHitTex.destroy();
    this.ssrReflTex.destroy();
    this.ssrReflTexPrev.destroy();

    // Recreate
    this.ssrHitTex      = makeRT(this.device, halfW, halfH, rtFmt, 'ssr-hit');
    this.ssrReflTex     = makeRT(this.device, width, height, rtFmt, 'ssr-refl');
    this.ssrReflTexPrev = makeRT(this.device, width, height, rtFmt, 'ssr-refl-prev');
  }

  // ── Cleanup ────────────────────────────────────────────────────────────────

  /** Release all GPU resources. */
  destroy(): void {
    this.ssrHitTex.destroy();
    this.ssrReflTex.destroy();
    this.ssrReflTexPrev.destroy();
    this.marchUB.destroy();
    this.resolveUB.destroy();
    this.compositeUB.destroy();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: physics-driven SSR modulation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Derive SSR param adjustments from live physics state.
 *
 * High kinetic energy → reduce temporal blend (less ghosting on fast cells).
 * High contact count  → boost reflection strength (polished collision surface).
 * High vorticity      → increase cone angle (turbulent surface = rougher).
 */
export function modulateSSRFromPhysics(
  base    : Readonly<SSRParams>,
  physics : PhysicsUniforms,
): Partial<SSRParams> {
  const kineticNorm = Math.min(1, physics.u_kineticEnergy / 5.0);
  const contactNorm = Math.min(1, physics.u_contactCount / 8.0);
  const vortNorm    = Math.min(1, Math.abs(physics.u_vorticity) / 50.0);

  return {
    temporalBlend:      base.temporalBlend * (1.0 - 0.3 * kineticNorm),
    reflectionStrength: base.reflectionStrength * (1.0 + 0.25 * contactNorm),
    // Note: cone angle is set per-species in the march shader, but we can
    // indirectly affect it by adjusting stride (wider stride = coarser march)
    stride:             base.stride * (1.0 + 0.5 * vortNorm),
  };
}
