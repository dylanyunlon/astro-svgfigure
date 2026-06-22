/**
 * at-volumetric-light.ts — M716: AT VolumetricLight WGSL Port — WebGPU/WGSL
 *
 * 移植自 ActiveTheory 的体积光 / God Rays / Light Shafts 系统：
 *
 *   AT 原始数据 (src/lib/shaders/compiled.vs + channels/physics/xiaodi_options_table.json):
 *     VolumetricLight_home fExposure = 0.86
 *     VolumetricLight_home fDensity  = 0.22
 *     VolumetricLight×6 — 6次迭代径向模糊 (nuke-pipeline.ts VolumetricLightPass)
 *
 *   参考源码：
 *     upstream/pixijs-filters/src/godray/god-ray.wgsl      — PixiJS God Ray WGSL
 *     upstream/pixijs-filters/src/godray/perlin.wgsl        — Perlin 湍流噪声
 *     upstream/lygia/lighting/volumetricLightScattering.wgsl — Lygia 光散射
 *     upstream/unreal-renderer-ue5/Renderer-Private/LightShaftRendering.cpp — UE5 算法
 *     src/lib/renderers/nuke-pipeline.ts VolumetricLightPass — WebGL 原型
 *
 * 管线结构（每帧 render()）：
 *   ┌─ Pass 1: OCCLUSION MASK ─────────────────────────────────────────────┐
 *   │  scene → occlusionTex (半分辨率)                                      │
 *   │  保留超过 threshold 的高亮像素 → 构成"光源遮挡掩码"                     │
 *   └──────────────────────────────────────────────────────────────────────┘
 *   ┌─ Pass 2: GOD RAY RADIAL BLUR (×NUM_SAMPLES iterations) ─────────────┐
 *   │  occlusionTex → raysTex (半分辨率)                                    │
 *   │  从每像素向光源 UV 方向径向采样, 每步乘衰减系数 decay^i               │
 *   │  AT 原始: 6 次迭代 (VolumetricLight×6 节点链)                        │
 *   │  + 可选 PixiJS Perlin 湍流噪声叠加（噪声光束效果）                    │
 *   └──────────────────────────────────────────────────────────────────────┘
 *   ┌─ Pass 3: MIE SCATTERING (Lygia volumetricLightScattering) ──────────┐
 *   │  raysTex → scatterTex (全分辨率)                                     │
 *   │  Mie 相函数 g = VOLUMETRICLIGHTSCATTERING_FACTOR                    │
 *   │  沿光线步进累积散射强度（阴影贴图/遮挡版）                             │
 *   └──────────────────────────────────────────────────────────────────────┘
 *   ┌─ Pass 4: COMPOSITE ──────────────────────────────────────────────────┐
 *   │  scene + scatterTex → dst (用户指定 GPUTextureView)                   │
 *   │  加性混合: output = scene + rays * raysScale                          │
 *   └──────────────────────────────────────────────────────────────────────┘
 *
 * 快速使用：
 *   const vl = await ATVolumetricLight.create(device, format, width, height);
 *   vl.setParams({ exposure: 0.86, density: 0.22, lightPos: [0.5, 0.1] });
 *   // 每帧:
 *   vl.render(encoder, sceneTex, dstView);
 *
 * Research: xiaodi #M716 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Public parameter types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Tweakable parameters for ATVolumetricLight.
 * Defaults match the AT UIL table (VolumetricLight_home).
 */
export interface ATVolumetricLightParams {
  /**
   * Overall brightness scalar.
   * AT: VolumetricLight_home fExposure = 0.86
   * @default 0.86
   */
  exposure?: number;

  /**
   * Controls how far each radial step marches (0–1, fraction of screen).
   * AT: VolumetricLight_home fDensity = 0.22
   * @default 0.22
   */
  density?: number;

  /**
   * Per-sample exponential decay.  Lower = faster fade-out.
   * AT: ~0.97 (derived from 6-iteration radial blur tapering)
   * @default 0.97
   */
  decay?: number;

  /**
   * Base tap weight (multiplied by decay^i per iteration).
   * @default 0.4
   */
  weight?: number;

  /**
   * Luminance threshold below which pixels are excluded from the occlusion mask.
   * @default 0.6
   */
  occlusionThreshold?: number;

  /**
   * UV position of the light source in [0,1]² screen space.
   * [0,0] = top-left; [1,1] = bottom-right.
   * @default [0.5, 0.05]
   */
  lightPos?: [number, number];

  /**
   * Multiplier for the god-ray layer before additive composite.
   * @default 1.0
   */
  raysScale?: number;

  /**
   * Enable PixiJS-style Perlin turbulence modulation on the rays.
   * Adds organic noise to the light shaft appearance.
   * @default false
   */
  enableNoise?: boolean;

  /**
   * Noise animation speed (used as uTime increment per frame; call setTime()).
   * @default 0.05
   */
  noiseSpeed?: number;

  /**
   * Fractal noise lacunarity (density of noise frequencies).
   * PixiJS GodrayFilter: lacunarity = 2.5
   * @default 2.5
   */
  lacunarity?: number;

  /**
   * Noise gain (controls contrast / intensity of each octave).
   * PixiJS GodrayFilter: gain = 0.5
   * @default 0.5
   */
  gain?: number;

  /**
   * Mie scattering asymmetry factor g ∈ (−1, 1).
   * Lygia: VOLUMETRICLIGHTSCATTERING_FACTOR = 0.25
   * 0 = isotropic; >0 = forward-scattering (toward camera); <0 = backward
   * @default 0.25
   */
  mieG?: number;

  /**
   * Number of radial blur samples per ray.
   * AT: exactly 6 (VolumetricLight×6 node chain).
   * Increase for smoother shafts at cost of GPU time.
   * @default 6
   */
  numSamples?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Perlin turbulence noise (移植自 pixijs-filters/src/godray/perlin.wgsl)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_PERLIN_NOISE = /* wgsl */`
// ── Perlin / turbulence noise (from pixijs-filters/src/godray/perlin.wgsl) ──
// Munrocket WGSL port: https://gist.github.com/munrocket/236ed5ba7e409b8bdf1ff6eca5dcdc39

fn vl_moduloVec3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
  return x - y * floor(x / y);
}
fn vl_mod289Vec3(x: vec3<f32>) -> vec3<f32> {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}
fn vl_mod289Vec4(x: vec4<f32>) -> vec4<f32> {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}
fn vl_permute4(x: vec4<f32>) -> vec4<f32> {
  return vl_mod289Vec4(((x * 34.0) + 1.0) * x);
}
fn vl_taylorInvSqrt(r: vec4<f32>) -> vec4<f32> {
  return 1.79284291400159 - 0.85373472095314 * r;
}
fn vl_fade3(t: vec3<f32>) -> vec3<f32> {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Classic 3D Perlin noise, periodic variant
fn vl_perlinNoise3(P: vec3<f32>, rep: vec3<f32>) -> f32 {
  var Pi0: vec3<f32> = vl_moduloVec3(floor(P), rep);
  var Pi1: vec3<f32> = vl_moduloVec3(Pi0 + vec3<f32>(1.0), rep);
  Pi0 = vl_mod289Vec3(Pi0);
  Pi1 = vl_mod289Vec3(Pi1);
  let Pf0: vec3<f32> = fract(P);
  let Pf1: vec3<f32> = Pf0 - vec3<f32>(1.0);
  let ix: vec4<f32> = vec4<f32>(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  let iy: vec4<f32> = vec4<f32>(Pi0.yy, Pi1.yy);
  let iz0: vec4<f32> = Pi0.zzzz;
  let iz1: vec4<f32> = Pi1.zzzz;
  let ixy: vec4<f32>  = vl_permute4(vl_permute4(ix) + iy);
  let ixy0: vec4<f32> = vl_permute4(ixy + iz0);
  let ixy1: vec4<f32> = vl_permute4(ixy + iz1);
  var gx0: vec4<f32> = ixy0 * (1.0 / 7.0);
  var gy0: vec4<f32> = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  let gz0: vec4<f32> = vec4<f32>(0.5) - abs(gx0) - abs(gy0);
  let sz0: vec4<f32> = step(gz0, vec4<f32>(0.0));
  gx0 -= sz0 * (step(vec4<f32>(0.0), gx0) - 0.5);
  gy0 -= sz0 * (step(vec4<f32>(0.0), gy0) - 0.5);
  var gx1: vec4<f32> = ixy1 * (1.0 / 7.0);
  var gy1: vec4<f32> = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  let gz1: vec4<f32> = vec4<f32>(0.5) - abs(gx1) - abs(gy1);
  let sz1: vec4<f32> = step(gz1, vec4<f32>(0.0));
  gx1 -= sz1 * (step(vec4<f32>(0.0), gx1) - 0.5);
  gy1 -= sz1 * (step(vec4<f32>(0.0), gy1) - 0.5);
  var g000: vec3<f32> = vec3<f32>(gx0.x, gy0.x, gz0.x);
  var g100: vec3<f32> = vec3<f32>(gx0.y, gy0.y, gz0.y);
  var g010: vec3<f32> = vec3<f32>(gx0.z, gy0.z, gz0.z);
  var g110: vec3<f32> = vec3<f32>(gx0.w, gy0.w, gz0.w);
  var g001: vec3<f32> = vec3<f32>(gx1.x, gy1.x, gz1.x);
  var g101: vec3<f32> = vec3<f32>(gx1.y, gy1.y, gz1.y);
  var g011: vec3<f32> = vec3<f32>(gx1.z, gy1.z, gz1.z);
  var g111: vec3<f32> = vec3<f32>(gx1.w, gy1.w, gz1.w);
  let norm0: vec4<f32> = vl_taylorInvSqrt(vec4<f32>(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x; g010 *= norm0.y; g100 *= norm0.z; g110 *= norm0.w;
  let norm1: vec4<f32> = vl_taylorInvSqrt(vec4<f32>(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x; g011 *= norm1.y; g101 *= norm1.z; g111 *= norm1.w;
  let n000 = dot(g000, Pf0);
  let n100 = dot(g100, vec3<f32>(Pf1.x, Pf0.yz));
  let n010 = dot(g010, vec3<f32>(Pf0.x, Pf1.y, Pf0.z));
  let n110 = dot(g110, vec3<f32>(Pf1.xy, Pf0.z));
  let n001 = dot(g001, vec3<f32>(Pf0.xy, Pf1.z));
  let n101 = dot(g101, vec3<f32>(Pf1.x, Pf0.y, Pf1.z));
  let n011 = dot(g011, vec3<f32>(Pf0.x, Pf1.yz));
  let n111 = dot(g111, Pf1);
  let fade_xyz = vl_fade3(Pf0);
  let n_z  = mix(vec4<f32>(n000, n100, n010, n110), vec4<f32>(n001, n101, n011, n111), fade_xyz.z);
  let n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  return 2.2 * mix(n_yz.x, n_yz.y, fade_xyz.x);
}

// Fractional Brownian Motion turbulence (6 octaves)
// Source: pixijs-filters perlin.wgsl turb()
fn vl_turb(P: vec3<f32>, rep: vec3<f32>, lacunarity: f32, gain: f32) -> f32 {
  var sum: f32 = 0.0;
  var sc: f32 = 1.0;
  var totalGain: f32 = 1.0;
  for (var i = 0.0; i < 6.0; i += 1.0) {
    sum += totalGain * vl_perlinNoise3(P * sc, rep);
    sc *= lacunarity;
    totalGain *= gain;
  }
  return abs(sum);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Uniform structs (shared across all passes)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_VL_UNIFORMS = /* wgsl */`
struct VLUniforms {
  // AT VolumetricLight_home parameters
  exposure  : f32,   // fExposure  default 0.86
  density   : f32,   // fDensity   default 0.22
  decay     : f32,   // per-sample decay  default 0.97
  weight    : f32,   // base tap weight   default 0.40
  // light source
  lightPosX : f32,   // UV [0,1]
  lightPosY : f32,   // UV [0,1]
  raysScale : f32,   // composite multiplier  default 1.0
  occlusionThreshold: f32,  // luminance cutoff default 0.6
  // noise (PixiJS GodrayFilter params)
  enableNoise : f32,  // 0.0 = off, 1.0 = on
  time        : f32,  // animation time
  lacunarity  : f32,  // default 2.5
  gain        : f32,  // default 0.5
  // Mie scattering (Lygia volumetricLightScattering.wgsl)
  mieG        : f32,  // default 0.25
  numSamples  : f32,  // default 6.0 (AT VolumetricLight×6)
  // padding to 16-byte alignment
  _pad0       : f32,
  _pad1       : f32,
};
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Full-screen triangle vertex shader (shared)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_FULLSCREEN_VERT = /* wgsl */`
@vertex
fn vsMain(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  // Emit a full-screen triangle without a vertex buffer.
  // vid=0 → (-1,-1); vid=1 → (3,-1); vid=2 → (-1, 3)
  let x = f32(i32(vid & 1u) * 4 - 1);
  let y = f32(i32(vid & 2u) * 2 - 1);
  return vec4<f32>(x, y, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 1: Occlusion Mask
//
// Retains only pixels whose luminance exceeds occlusionThreshold.
// Equivalent to AT VolumetricLightComposite mask pass + nuke-pipeline
//   VOLUMETRIC_OCCLUDE_FRAG (WebGL original).
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_OCCLUSION = /* wgsl */`
${WGSL_VL_UNIFORMS}

@group(0) @binding(0) var<uniform> u : VLUniforms;
@group(0) @binding(1) var uSampler   : sampler;
@group(0) @binding(2) var uScene     : texture_2d<f32>;

${WGSL_FULLSCREEN_VERT}

@fragment
fn fsOcclusion(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dims    = vec2<f32>(textureDimensions(uScene));
  let uv      = pos.xy / dims;
  let color   = textureSample(uScene, uSampler, uv);
  // Rec.709 luminance
  let lum     = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
  // Keep only bright "light source" regions; black everything else
  let keep    = step(u.occlusionThreshold, lum);
  return vec4<f32>(color.rgb * keep, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 2: God Ray Radial Blur
//
// Algorithm:
//   Each fragment marches from its UV toward uLightPos in numSamples steps.
//   Each step accumulates the occlusionTex sample weighted by decay^i.
//   Result is the "light shaft accumulation" image.
//
// Sources:
//   nuke-pipeline.ts VOLUMETRIC_RAYS_FRAG (WebGL)
//   pixijs-filters/src/godray/god-ray.wgsl (PixiJS WGSL)
//   unreal-renderer-ue5/Renderer-Private/LightShaftRendering.cpp (UE5)
//
// AT parallel: VolumetricLight×6 nodes = 6 explicit RadialBlurKernel iterations.
// Here we generalize to numSamples (default 6, matching AT).
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_GOD_RAYS = /* wgsl */`
${WGSL_VL_UNIFORMS}

@group(0) @binding(0) var<uniform> u          : VLUniforms;
@group(0) @binding(1) var uSampler            : sampler;
@group(0) @binding(2) var uOcclusionTex       : texture_2d<f32>;

// Perlin noise functions (conditionally used when u.enableNoise > 0.5)
${WGSL_PERLIN_NOISE}

${WGSL_FULLSCREEN_VERT}

@fragment
fn fsGodRays(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dims  = vec2<f32>(textureDimensions(uOcclusionTex));
  let uv    = pos.xy / dims;

  // Direction from fragment toward light source (screen space)
  let lightPos = vec2<f32>(u.lightPosX, u.lightPosY);
  let toLight  = lightPos - uv;
  let n        = max(u.numSamples, 1.0);
  let stepVec  = toLight * (u.density / n);

  var sampleUV   = uv;
  var accumLight = vec3<f32>(0.0);
  var w          = u.weight;

  // ── AT VolumetricLight×6: unrolled for numSamples iterations ─────────────
  // We loop up to 16 iterations max (WGSL requires a constant loop bound
  // or a dynamic loop; we use dynamic here which is valid in WGSL 2024).
  let iMax = i32(clamp(u.numSamples, 1.0, 32.0));
  for (var i = 0; i < iMax; i++) {
    sampleUV   += stepVec;
    let sample  = textureSample(uOcclusionTex, uSampler, sampleUV).rgb;
    accumLight += sample * w;
    w          *= u.decay;
  }

  // Exposure scale (AT: fExposure = 0.86)
  var rayColor = accumLight * u.exposure;

  // ── Optional: PixiJS Perlin turbulence noise modulation ─────────────────
  // god-ray.wgsl: noise modulates the ray intensity with organic variability.
  // AT parallel: the "organic" quality of god rays vs pure radial blur.
  if (u.enableNoise > 0.5) {
    // Project uv to a 1D "d" value along ray direction (PixiJS technique)
    let aspect = dims.y / dims.x;
    let dx     = uv.x - u.lightPosX;
    let dy     = (uv.y - u.lightPosY) * aspect;
    let dis    = sqrt(dx * dx + dy * dy) + 0.00001;
    let d      = dy / dis;

    let noiseDir = vec3<f32>(d, d, 0.0);
    let noisePos = noiseDir + vec3<f32>(u.time, 0.0, 62.1 + u.time) * 0.05;
    let noise    = vl_turb(noisePos, vec3<f32>(480.0, 320.0, 480.0), u.lacunarity, u.gain);
    let noiseMod = mix(noise, 0.0, 0.3);
    // Vertical fade (PixiJS: mist *= (1 - coord.y))
    let fade     = max(1.0 - uv.y, 0.0);
    rayColor    += vec3<f32>(noiseMod) * fade * u.exposure * 0.4;
  }

  return vec4<f32>(rayColor, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 3: Mie Scattering (Lygia volumetricLightScattering.wgsl port)
//
// Applies the Henyey-Greenstein Mie phase function to the accumulated rays.
// This gives the characteristic forward-scattering "glow cone" appearance.
//
// Source: upstream/lygia/lighting/volumetricLightScattering.wgsl
//   Henyey-Greenstein phase:
//     scattering_g = g * g
//     scattering   = (1 - scattering_g) / (4π × (1 + scattering_g - 2g×cosθ)^1.5)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_MIE_SCATTER = /* wgsl */`
${WGSL_VL_UNIFORMS}

@group(0) @binding(0) var<uniform> u      : VLUniforms;
@group(0) @binding(1) var uSampler        : sampler;
@group(0) @binding(2) var uRaysTex        : texture_2d<f32>;

const VL_PI: f32 = 3.14159265358979323846;

// Henyey-Greenstein Mie phase function
// g: asymmetry parameter (0 = isotropic, >0 = forward-scatter)
// cosTheta: cos of angle between view ray and light direction
fn vl_henyeyGreenstein(g: f32, cosTheta: f32) -> f32 {
  let g2  = g * g;
  let num = 1.0 - g2;
  let den = pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
  return num / (4.0 * VL_PI * den + 1e-7);
}

${WGSL_FULLSCREEN_VERT}

@fragment
fn fsMieScatter(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dims     = vec2<f32>(textureDimensions(uRaysTex));
  let uv       = pos.xy / dims;

  // Fetch accumulated god-ray color
  let rays     = textureSample(uRaysTex, uSampler, uv).rgb;

  // Compute cos(θ) between view ray (away from screen) and light direction
  // Simplified: use dot of (uv → lightPos) direction vs (0,0,1) view vector.
  let lightPos  = vec2<f32>(u.lightPosX, u.lightPosY);
  let toLightUV = normalize(lightPos - uv);
  // In screen space the "view ray" projected 2D points away from center.
  // cosTheta approximated as the normalized overlap factor.
  let centerDir = normalize(uv - vec2<f32>(0.5));
  let cosTheta  = dot(toLightUV, -centerDir) * 0.5 + 0.5;  // map [-1,1]→[0,1]

  // Apply Mie phase weighting
  let phase     = vl_henyeyGreenstein(u.mieG, cosTheta);
  // Normalize phase contribution to keep output range stable
  let phaseNorm = clamp(phase / (1.0 / (4.0 * VL_PI) + 1e-7), 0.0, 3.0);

  // Scale scattered light (blend between raw rays and Mie-modulated)
  let scattered = rays * mix(1.0, phaseNorm, 0.35);

  return vec4<f32>(scattered, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 4: Composite
//
// Additive composite: output = scene + scatteredRays × raysScale
// AT: VolumetricLightComposite node — additive blend over scene.
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_COMPOSITE = /* wgsl */`
${WGSL_VL_UNIFORMS}

@group(0) @binding(0) var<uniform> u          : VLUniforms;
@group(0) @binding(1) var uSampler            : sampler;
@group(0) @binding(2) var uScene              : texture_2d<f32>;
@group(0) @binding(3) var uScatterTex         : texture_2d<f32>;

${WGSL_FULLSCREEN_VERT}

@fragment
fn fsComposite(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let sceneDims   = vec2<f32>(textureDimensions(uScene));
  let uv          = pos.xy / sceneDims;
  let scene       = textureSample(uScene,       uSampler, uv);
  // scatterTex may be half-res; sample with bilinear interpolation
  let scatter     = textureSample(uScatterTex,  uSampler, uv);
  // Additive blend — AT VolumetricLightComposite
  let result      = scene.rgb + scatter.rgb * u.raysScale;
  return vec4<f32>(clamp(result, vec3<f32>(0.0), vec3<f32>(1.0)), scene.a);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
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
  device  : GPUDevice,
  label   : string,
  numTextures: number,
): GPUBindGroupLayout {
  const entries: GPUBindGroupLayoutEntry[] = [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
  ];
  for (let i = 0; i < numTextures; i++) {
    entries.push({
      binding    : 2 + i,
      visibility : GPUShaderStage.FRAGMENT,
      texture    : { sampleType: 'float', viewDimension: '2d' },
    });
  }
  return device.createBindGroupLayout({ label, entries });
}

/** Build a render pipeline with the given vertex + fragment code. */
function makePipeline(
  device     : GPUDevice,
  label      : string,
  wgsl       : string,
  vsEntry    : string,
  fsEntry    : string,
  bgl        : GPUBindGroupLayout,
  format     : GPUTextureFormat,
): GPURenderPipeline {
  const module = device.createShaderModule({ label, code: wgsl });
  const layout = device.createPipelineLayout({
    label,
    bindGroupLayouts: [bgl],
  });
  return device.createRenderPipeline({
    label,
    layout,
    vertex  : { module, entryPoint: vsEntry },
    fragment: { module, entryPoint: fsEntry, targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// ATVolumetricLight — main WebGPU class
// ─────────────────────────────────────────────────────────────────────────────

/** Default parameter values (AT UIL table, VolumetricLight_home). */
const DEFAULTS: Required<ATVolumetricLightParams> = {
  exposure          : 0.86,
  density           : 0.22,
  decay             : 0.97,
  weight            : 0.40,
  occlusionThreshold: 0.60,
  lightPos          : [0.5, 0.05],
  raysScale         : 1.0,
  enableNoise       : false,
  noiseSpeed        : 0.05,
  lacunarity        : 2.5,
  gain              : 0.5,
  mieG              : 0.25,
  numSamples        : 6,
};

/**
 * ATVolumetricLight — WebGPU God Rays / Light Shafts post-process.
 *
 * Implements the AT VolumetricLight×6 radial blur pipeline as a 4-pass
 * WebGPU renderer:
 *   1. Occlusion mask (bright pixel extraction)
 *   2. Radial blur toward light source (AT ×6 iterations)
 *   3. Mie scattering phase modulation (Lygia WGSL)
 *   4. Additive composite over scene
 */
export class ATVolumetricLight {
  private readonly device : GPUDevice;
  private readonly format : GPUTextureFormat;
  private readonly width  : number;
  private readonly height : number;

  // ── GPU resources ──────────────────────────────────────────────────────────
  private uniformBuf   : GPUBuffer;
  private sampler      : GPUSampler;

  // Half-resolution intermediate textures (god rays are blurry; half-res is fine)
  private occlusionTex : GPUTexture;  // bright-pixel mask
  private raysTex      : GPUTexture;  // radial blur result
  private scatterTex   : GPUTexture;  // Mie scatter result (full-res)

  // ── Pipeline state ─────────────────────────────────────────────────────────
  private occlusionBGL : GPUBindGroupLayout;
  private raysBGL      : GPUBindGroupLayout;
  private mieScatterBGL: GPUBindGroupLayout;
  private compositeBGL : GPUBindGroupLayout;

  private occlusionPipeline  : GPURenderPipeline;
  private raysPipeline       : GPURenderPipeline;
  private mieScatterPipeline : GPURenderPipeline;
  private compositePipeline  : GPURenderPipeline;

  // ── Runtime state ──────────────────────────────────────────────────────────
  private params : Required<ATVolumetricLightParams>;
  private time   = 0.0;

  /** Uniform buffer layout: 16 f32 values = 64 bytes */
  private static readonly UNIFORM_SIZE = 64;

  private constructor(
    device : GPUDevice,
    format : GPUTextureFormat,
    width  : number,
    height : number,
  ) {
    this.device = device;
    this.format = format;
    this.width  = width;
    this.height = height;
    this.params = { ...DEFAULTS };

    const halfW = Math.max(1, width  >> 1);
    const halfH = Math.max(1, height >> 1);

    // Intermediate textures
    this.occlusionTex = makeRT(device, halfW, halfH, format, 'at-vl-occlusion');
    this.raysTex      = makeRT(device, halfW, halfH, format, 'at-vl-rays');
    this.scatterTex   = makeRT(device, width,  height, format, 'at-vl-scatter');

    // Uniform buffer
    this.uniformBuf = device.createBuffer({
      label : 'at-vl-uniforms',
      size  : ATVolumetricLight.UNIFORM_SIZE,
      usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Bilinear sampler (used in all passes for smooth interpolation)
    this.sampler = device.createSampler({
      label        : 'at-vl-sampler',
      magFilter    : 'linear',
      minFilter    : 'linear',
      addressModeU : 'clamp-to-edge',
      addressModeV : 'clamp-to-edge',
    });

    // Bind group layouts
    this.occlusionBGL   = makeBGL(device, 'at-vl-occlusion-bgl',    1); // scene
    this.raysBGL        = makeBGL(device, 'at-vl-rays-bgl',         1); // occlusionTex
    this.mieScatterBGL  = makeBGL(device, 'at-vl-mie-bgl',          1); // raysTex
    this.compositeBGL   = makeBGL(device, 'at-vl-composite-bgl',    2); // scene + scatterTex

    // Render pipelines
    this.occlusionPipeline  = makePipeline(device, 'at-vl-occlusion-pipeline',
      WGSL_OCCLUSION,  'vsMain', 'fsOcclusion',  this.occlusionBGL,  format);

    this.raysPipeline       = makePipeline(device, 'at-vl-rays-pipeline',
      WGSL_GOD_RAYS,   'vsMain', 'fsGodRays',    this.raysBGL,       format);

    this.mieScatterPipeline = makePipeline(device, 'at-vl-mie-pipeline',
      WGSL_MIE_SCATTER,'vsMain', 'fsMieScatter', this.mieScatterBGL, format);

    this.compositePipeline  = makePipeline(device, 'at-vl-composite-pipeline',
      WGSL_COMPOSITE,  'vsMain', 'fsComposite',  this.compositeBGL,  format);

    // Sync initial params to GPU
    this._uploadUniforms();
  }

  /**
   * Factory — asynchronous constructor (mirrors ATBloomPostProcess.create()).
   *
   * @param device - WebGPU device.
   * @param format - Swapchain / render target texture format.
   * @param width  - Viewport width in pixels.
   * @param height - Viewport height in pixels.
   */
  static async create(
    device : GPUDevice,
    format : GPUTextureFormat,
    width  : number,
    height : number,
  ): Promise<ATVolumetricLight> {
    return new ATVolumetricLight(device, format, width, height);
  }

  // ── Parameter API ──────────────────────────────────────────────────────────

  /**
   * Update volumetric light parameters.
   * Writes the new values to the GPU uniform buffer.
   *
   * @param p - Partial parameter overrides.
   */
  setParams(p: ATVolumetricLightParams): void {
    Object.assign(this.params, p);
    this._uploadUniforms();
  }

  /**
   * Advance the noise animation clock.
   * Call once per frame when enableNoise is true.
   *
   * @param dt - Delta time in seconds (or arbitrary frame increment).
   */
  tick(dt = 0.016): void {
    this.time += dt * this.params.noiseSpeed;
    this._uploadUniforms();
  }

  /** Directly set the animation time (e.g. for scrubbing). */
  setTime(t: number): void {
    this.time = t;
    this._uploadUniforms();
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  /**
   * Execute the full 4-pass volumetric light pipeline.
   *
   * @param encoder  - Active GPUCommandEncoder.
   * @param sceneTex - Source scene texture (GPUTexture, not a view).
   * @param dstView  - Destination GPUTextureView (e.g. swapchain current texture).
   */
  render(
    encoder  : GPUCommandEncoder,
    sceneTex : GPUTexture,
    dstView  : GPUTextureView,
  ): void {
    const sceneView      = sceneTex.createView();
    const occlusionView  = this.occlusionTex.createView();
    const raysView       = this.raysTex.createView();
    const scatterView    = this.scatterTex.createView();

    // ── Pass 1: Occlusion mask ───────────────────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        label  : 'at-vl-occlusion-bg',
        layout : this.occlusionBGL,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: sceneView },
        ],
      });
      const pass = encoder.beginRenderPass({
        label: 'at-vl-occlusion-pass',
        colorAttachments: [{
          view      : occlusionView,
          loadOp    : 'clear',
          storeOp   : 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(this.occlusionPipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }

    // ── Pass 2: God Ray radial blur (AT VolumetricLight×6) ───────────────────
    {
      const bg = this.device.createBindGroup({
        label  : 'at-vl-rays-bg',
        layout : this.raysBGL,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: occlusionView },
        ],
      });
      const pass = encoder.beginRenderPass({
        label: 'at-vl-rays-pass',
        colorAttachments: [{
          view      : raysView,
          loadOp    : 'clear',
          storeOp   : 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(this.raysPipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }

    // ── Pass 3: Mie scattering phase modulation (Lygia) ──────────────────────
    {
      const bg = this.device.createBindGroup({
        label  : 'at-vl-mie-bg',
        layout : this.mieScatterBGL,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: raysView },
        ],
      });
      const pass = encoder.beginRenderPass({
        label: 'at-vl-mie-pass',
        colorAttachments: [{
          view      : scatterView,
          loadOp    : 'clear',
          storeOp   : 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(this.mieScatterPipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }

    // ── Pass 4: Composite — additive blend over scene ────────────────────────
    {
      const bg = this.device.createBindGroup({
        label  : 'at-vl-composite-bg',
        layout : this.compositeBGL,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: sceneView },
          { binding: 3, resource: scatterView },
        ],
      });
      const pass = encoder.beginRenderPass({
        label: 'at-vl-composite-pass',
        colorAttachments: [{
          view      : dstView,
          loadOp    : 'clear',
          storeOp   : 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(this.compositePipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }
  }

  // ── Resource management ────────────────────────────────────────────────────

  /**
   * Release all GPU resources.
   * The instance must not be used after calling this method.
   */
  destroy(): void {
    this.uniformBuf.destroy();
    this.occlusionTex.destroy();
    this.raysTex.destroy();
    this.scatterTex.destroy();
  }

  // ── Diagnostics ────────────────────────────────────────────────────────────

  /** Returns the four WGSL shader sources for inspection / hot-reload. */
  get wgslSources(): Readonly<Record<string, string>> {
    return {
      occlusion : WGSL_OCCLUSION,
      godRays   : WGSL_GOD_RAYS,
      mieScatter: WGSL_MIE_SCATTER,
      composite : WGSL_COMPOSITE,
    };
  }

  /** Current resolved parameters (including defaults). */
  get currentParams(): Readonly<Required<ATVolumetricLightParams>> {
    return { ...this.params };
  }

  // ── Private ────────────────────────────────────────────────────────────────

  /**
   * Write current params to the GPU uniform buffer.
   * Layout must match VLUniforms struct (16 × f32 = 64 bytes).
   */
  private _uploadUniforms(): void {
    const p   = this.params;
    const buf = new Float32Array(16);
    buf[ 0] = p.exposure;
    buf[ 1] = p.density;
    buf[ 2] = p.decay;
    buf[ 3] = p.weight;
    buf[ 4] = p.lightPos[0];
    buf[ 5] = p.lightPos[1];
    buf[ 6] = p.raysScale;
    buf[ 7] = p.occlusionThreshold;
    buf[ 8] = p.enableNoise ? 1.0 : 0.0;
    buf[ 9] = this.time;
    buf[10] = p.lacunarity;
    buf[11] = p.gain;
    buf[12] = p.mieG;
    buf[13] = p.numSamples;
    buf[14] = 0;  // _pad0
    buf[15] = 0;  // _pad1
    this.device.queue.writeBuffer(this.uniformBuf, 0, buf);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factory — species-params integration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build an ATVolumetricLight from BloomVariants-style species params.
 * Bridges the SPH species parameter schema to the AT volumetric light API.
 *
 * @param device      - WebGPU device.
 * @param format      - Swapchain texture format.
 * @param width       - Viewport width.
 * @param height      - Viewport height.
 * @param speciesOpts - Optional species-specific overrides.
 */
export async function createATVolumetricLightForSpecies(
  device      : GPUDevice,
  format      : GPUTextureFormat,
  width       : number,
  height      : number,
  speciesOpts?: {
    lightExposure?  : number;  // VolumetricLight_home fExposure
    lightDensity?   : number;  // VolumetricLight_home fDensity
    lightPos?       : [number, number];
    raysScale?      : number;
    enableNoise?    : boolean;
  },
): Promise<ATVolumetricLight> {
  const vl = await ATVolumetricLight.create(device, format, width, height);
  if (speciesOpts) {
    vl.setParams({
      exposure   : speciesOpts.lightExposure ?? DEFAULTS.exposure,
      density    : speciesOpts.lightDensity  ?? DEFAULTS.density,
      lightPos   : speciesOpts.lightPos      ?? DEFAULTS.lightPos,
      raysScale  : speciesOpts.raysScale     ?? DEFAULTS.raysScale,
      enableNoise: speciesOpts.enableNoise   ?? DEFAULTS.enableNoise,
    });
  }
  return vl;
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-export WGSL fragments — other shaders may embed them
// ─────────────────────────────────────────────────────────────────────────────

/** WGSL source fragments for embedding in other shader modules. */
export const AT_VOLUMETRIC_LIGHT_WGSL = {
  /** Shared uniform struct (VLUniforms). */
  uniforms    : WGSL_VL_UNIFORMS,
  /** Full-screen triangle vertex shader. */
  fullscreenVs: WGSL_FULLSCREEN_VERT,
  /** Perlin turbulence noise helpers. */
  perlinNoise : WGSL_PERLIN_NOISE,
  /** Occlusion mask fragment shader. */
  occlusion   : WGSL_OCCLUSION,
  /** God ray radial blur fragment shader. */
  godRays     : WGSL_GOD_RAYS,
  /** Mie scattering phase fragment shader. */
  mieScatter  : WGSL_MIE_SCATTER,
  /** Additive composite fragment shader. */
  composite   : WGSL_COMPOSITE,
} as const;
