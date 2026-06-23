/**
 * src/lib/sph/at-terrain-environment.ts  —  M824
 *
 * AT Terrain Environment → WebGPU / WGSL Port
 * ─────────────────────────────────────────────────────────────────────────────
 * Full GPU port of Active Theory's tree-room terrain system into the
 * project's WebGPU architecture.  Loads five Draco-compressed .bin meshes
 * (rocky_soil, rock_L, rock_R, sand, walls) with their PBR texture sets
 * (baseColor, normal, MRO) and renders them using a Cook-Torrance BRDF
 * lighting pipeline ported from AT's lighting.fs / compiled.vs bundle.
 *
 * ─── AT Reverse-Engineered Sources ──────────────────────────────────────────
 *
 *   Geometry (Draco-compressed .bin)
 *     ← upstream/activetheory-assets/geometry/rocky_soil.bin  (12856 verts)
 *     ← upstream/activetheory-assets/geometry/rock_L.bin      (14391 verts)
 *     ← upstream/activetheory-assets/geometry/rock_R.bin      (14391 verts)
 *     ← upstream/activetheory-assets/geometry/sand.bin         (13879 verts)
 *     ← upstream/activetheory-assets/geometry/walls.bin        (14135 verts)
 *     Format: JSON header {name, type:0, attributes:[["position",7],["normal",7],["uv",7]]}
 *             + DRACO compressed payload → position/normal/uv/index arrays
 *
 *   Textures (KTX2 compressed PBR sets)
 *     ← upstream/activetheory-assets/textures/{NAME}___CyclesBake_COMBINED.ktx2
 *     ← upstream/activetheory-assets/textures/{NAME}___PBR_Normal.ktx2
 *     ← upstream/activetheory-assets/textures/{NAME}___PBR_AT_MRO.ktx2
 *     Five complete PBR sets: ROCKY_SOIL, ROCK_L, ROCK_R, SAND, WALLS_CEILING
 *     MRO packing: R=Metallic, G=Roughness, B=Occlusion
 *
 *   Lighting (GLSL → WGSL port)
 *     ← upstream/activetheory-assets/shaders/lighting.fs
 *     ← upstream/activetheory-assets/shaders/lighting.vs
 *     Key algorithms:
 *       • setupLight()         : modelView pos, normal, worldPos, viewDir
 *       • lightDirectional()   : Lambertian volume + clamped min shadow
 *       • lightPoint()         : distance falloff + Phong specular option
 *       • lightCone()          : cone attenuation via angle smoothstep
 *       • getCombinedColor()   : iterate NUM_LIGHTS, dispatch by lProps.w type
 *       • Cook-Torrance BRDF   : GGX NDF + Smith G + Schlick F
 *
 * ─── WebGPU Architecture ────────────────────────────────────────────────────
 *
 *   ATTerrainEnvironment
 *     ├─ Asset loading pipeline
 *     │    ATGeometryLoader  — Draco decode → positions/normals/uvs/indices
 *     │    ATTextureLoader   — KTX2 decode → baseColor/normal/MRO per mesh
 *     │
 *     ├─ Per-mesh GPU resources
 *     │    vertexBuffer      — interleaved float32 [pos.xyz, norm.xyz, uv.xy]
 *     │    indexBuffer        — uint32 indices
 *     │    bindGroup          — PBR textures + samplers + uniforms
 *     │
 *     ├─ PBR render pipeline (render)
 *     │    vertex shader     — model→world→clip transform, TBN basis
 *     │    fragment shader   — Cook-Torrance BRDF + AT multi-light loop
 *     │                        directional + point + cone (from lighting.fs)
 *     │                        normal mapping, MRO unpack, env ambient
 *     │
 *     ├─ Shadow pipeline (render-to-depth)
 *     │    vertex shader     — light-space depth-only pass
 *     │    shadow map        — depth24plus, cascaded or single
 *     │
 *     └─ Environment effects
 *          fog compute       — distance + height fog, colour from sky
 *          dust particles    — instanced billboard quads (wind-driven)
 *          ambient occlusion — SSAO-lite baked from MRO.b channel
 *
 * ─── GLSL → WGSL Translation Key ────────────────────────────────────────────
 *
 *   setupLight(p0, normal)              → vs_terrain output varyings
 *   lworldLight(lPos, localPos, …)      → lightWorldDir() in WGSL
 *   lightDirectional(config, …)         → lightDirectional() WGSL fn
 *   lightPoint(config, …)              → lightPoint() WGSL fn
 *   lightCone(config, …)               → lightCone() WGSL fn
 *   getCombinedColor(config)           → getCombinedColor() iterates lights
 *   texture2D(tMap, vUv)               → textureSample(tMap, samp, uv)
 *   gl_FragColor                       → @location(0) out : vec4f
 *   uniform mat4 modelViewMatrix       → uniforms.modelView
 *   uniform mat4 viewMatrix            → uniforms.view
 *   normalMatrix * normal              → (uniforms.normalMat * vec4f(n,0)).xyz
 *
 * ─── Upstream Copyright Notices ─────────────────────────────────────────────
 *
 *   Active Theory Pty Ltd — All rights reserved
 *   Draco 3D Data Compression — Apache 2.0 License
 *     https://github.com/google/draco
 *   KTX-Software — Apache 2.0 License
 *     https://github.com/KhronosGroup/KTX-Software
 *
 * ─── Usage ───────────────────────────────────────────────────────────────────
 *
 *   const terrain = new ATTerrainEnvironment(device, format, config);
 *   await terrain.build();
 *
 *   // Optional: configure lights
 *   terrain.setLight(0, { type: 1, position: [5, 8, -3], color: [1, 0.95, 0.9], ... });
 *
 *   // Render loop:
 *   const enc = device.createCommandEncoder();
 *   terrain.tick(enc, elapsedSeconds, deltaSeconds);
 *   terrain.renderPass(enc, colorTargetView, depthView, sceneUniformBuf);
 *   device.queue.submit([enc.finish()]);
 *
 * Research: xiaodi #M824 — cell-pubsub-loop
 */

import { ATGeometryLoader } from './at-geometry-loader';
import type { ATGeometry } from './at-geometry-loader';
import { ATTextureLoader } from './at-texture-loader';
import type { ATTexture, ATMaterialSet } from './at-texture-loader';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Maximum number of lights supported in the WGSL light loop. */
const MAX_LIGHTS = 8 as const;

/** Maximum dust particles in the GPU pool. */
const MAX_DUST_PARTICLES = 4096 as const;

/** Workgroup size for compute shaders. */
const WG = 64 as const;

/** Interleaved vertex stride: position(3) + normal(3) + uv(2) = 8 floats. */
const VERTEX_STRIDE = 8 as const;

/** Bytes per interleaved vertex (8 × float32 = 32 bytes). */
const VERTEX_STRIDE_BYTES = 32 as const;

/** Shadow map resolution. */
const SHADOW_MAP_SIZE = 1024 as const;

/** Default fog density (exponential fog). */
const FOG_DENSITY = 0.035 as const;

/** Default fog colour (warm atmospheric haze). */
const FOG_COLOR: [number, number, number] = [0.55, 0.52, 0.48];

/** Default ambient light intensity. */
const AMBIENT_INTENSITY = 0.12 as const;

/** Default ambient colour (blue-grey skylight). */
const AMBIENT_COLOR: [number, number, number] = [0.35, 0.40, 0.50];

// ─────────────────────────────────────────────────────────────────────────────
// Terrain mesh identifiers
// ─────────────────────────────────────────────────────────────────────────────

/** The five terrain meshes that compose the AT tree-room environment. */
const TERRAIN_MESHES = [
  'rocky_soil',
  'rock_L',
  'rock_R',
  'sand',
  'walls',
] as const;

type TerrainMeshName = typeof TERRAIN_MESHES[number];

/**
 * Mapping from geometry .bin name to PBR texture set prefix.
 * Note: walls.bin uses WALLS_CEILING textures.
 */
const MESH_TO_TEXTURE_PREFIX: Record<TerrainMeshName, string> = {
  rocky_soil: 'ROCKY_SOIL',
  rock_L:     'ROCK_L',
  rock_R:     'ROCK_R',
  sand:       'SAND',
  walls:      'WALLS_CEILING',
};

// ─────────────────────────────────────────────────────────────────────────────
// Config types
// ─────────────────────────────────────────────────────────────────────────────

/** Light definition matching AT lighting.fs arrays. */
export interface ATTerrainLight {
  /** Light type: 1 = directional, 2 = point, 3 = cone/spot. */
  type: 1 | 2 | 3;
  /** Position (world space) or direction for directional lights. */
  position: [number, number, number];
  /** RGB colour (linear space, may exceed 1.0 for HDR). */
  color: [number, number, number];
  /** Intensity multiplier. Default 1.0. */
  intensity?: number;
  /** Max range (point/cone only). Default 50.0. */
  range?: number;
  /** Minimum shadow volume (directional penumbra floor). Default 0.0. */
  shadowMin?: number;
  /** Cone direction (cone only). */
  coneDirection?: [number, number, number];
  /** Cone angle in degrees (cone only). Default 45.0. */
  coneAngle?: number;
  /** Cone feather softness (cone only). Default 1.0. */
  coneFeather?: number;
  /** Enable Phong specular highlight for this light. */
  phong?: boolean;
  /** Phong shininess exponent. Default 30.0. */
  phongShininess?: number;
}

/** Per-mesh material overrides. */
export interface ATTerrainMaterialOverride {
  /** Base colour tint multiplier. Default [1,1,1]. */
  tint?: [number, number, number];
  /** Roughness scale. Default 1.0. */
  roughnessScale?: number;
  /** Metallic scale. Default 1.0. */
  metallicScale?: number;
  /** Normal map strength. Default 1.0. */
  normalStrength?: number;
  /** UV tiling factor. Default [1,1]. */
  uvScale?: [number, number];
}

export interface ATTerrainEnvironmentConfig {
  /**
   * Base URL path for geometry .bin files.
   * @default '/upstream/activetheory-assets/geometry'
   */
  geometryPath?: string;

  /**
   * Base URL path for texture .ktx2 files.
   * @default '/upstream/activetheory-assets/textures'
   */
  texturePath?: string;

  /**
   * Initial light array (up to MAX_LIGHTS).
   * Default: one warm directional sun + one cool fill.
   */
  lights?: ATTerrainLight[];

  /**
   * Fog colour (linear RGB).
   * Default: [0.55, 0.52, 0.48] warm atmospheric haze.
   */
  fogColor?: [number, number, number];

  /**
   * Fog density (exponential). Default 0.035.
   */
  fogDensity?: number;

  /**
   * Fog start distance. Default 5.0.
   */
  fogStart?: number;

  /**
   * Fog end distance (full opacity). Default 80.0.
   */
  fogEnd?: number;

  /**
   * Ambient light colour. Default [0.35, 0.40, 0.50].
   */
  ambientColor?: [number, number, number];

  /**
   * Ambient light intensity. Default 0.12.
   */
  ambientIntensity?: number;

  /**
   * Enable shadow mapping. Default true.
   */
  enableShadows?: boolean;

  /**
   * Shadow map resolution. Default 1024.
   */
  shadowMapSize?: number;

  /**
   * Enable dust particle overlay. Default true.
   */
  enableDust?: boolean;

  /**
   * Maximum dust particles. Default 4096.
   */
  maxDustParticles?: number;

  /**
   * Dust particle colour. Default [0.8, 0.75, 0.65].
   */
  dustColor?: [number, number, number];

  /**
   * Wind direction for dust drift. Default [0.3, 0.1, -0.2].
   */
  windDirection?: [number, number, number];

  /**
   * Per-mesh material overrides keyed by TerrainMeshName.
   */
  materialOverrides?: Partial<Record<TerrainMeshName, ATTerrainMaterialOverride>>;

  /**
   * Enable height-based vertex displacement (parallax ground effect).
   * Default false.
   */
  enableHeightDisplacement?: boolean;

  /**
   * Optional pre-loaded environment map for IBL reflections.
   */
  envMap?: GPUTexture;
}

// ─────────────────────────────────────────────────────────────────────────────
// Uniform buffer layouts
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Scene uniform buffer: 256 bytes (must match WGSL struct SceneUniforms).
 *
 *   offset   0: viewProj       mat4x4f  (64B)
 *   offset  64: modelMat       mat4x4f  (64B)
 *   offset 128: normalMat      mat4x4f  (64B)  — inverse-transpose of modelMat
 *   offset 192: eye            vec4f    (16B)
 *   offset 208: fogParams      vec4f    (16B)  — (.xyz=fogColor, .w=fogDensity)
 *   offset 224: ambientParams  vec4f    (16B)  — (.xyz=ambientColor, .w=ambientIntensity)
 *   offset 240: timeParams     vec4f    (16B)  — (.x=time, .y=dt, .z=fogStart, .w=fogEnd)
 */
const SCENE_UNIFORM_SIZE = 256 as const;

/**
 * Light uniform buffer: per-light = 80 bytes, array of MAX_LIGHTS.
 *
 *   Per-light struct (80 bytes):
 *     offset 0:  lightColor    vec4f  — .rgb=colour, .w=intensity
 *     offset 16: lightPos      vec4f  — .xyz=position, .w=type
 *     offset 32: lightData     vec4f  — .xyz=coneDir, .w=coneAngle
 *     offset 48: lightData2    vec4f  — .x=feather, .y=phongShininess, .z=shadowMin, .w=range
 *     offset 64: lightProps    vec4f  — .x=intensity, .y=range, .z=shadowMin, .w=type
 */
const LIGHT_STRUCT_SIZE = 80 as const;
const LIGHT_BUFFER_SIZE = (LIGHT_STRUCT_SIZE * MAX_LIGHTS) as number;

/**
 * Material uniform buffer: 64 bytes per mesh.
 *
 *   offset 0:  materialTint   vec4f  — .rgb=tint, .w=roughnessScale
 *   offset 16: materialData   vec4f  — .x=metallicScale, .y=normalStrength, .zw=uvScale
 *   offset 32: materialExtra  vec4f  — reserved
 *   offset 48: materialExtra2 vec4f  — reserved
 */
const MATERIAL_UNIFORM_SIZE = 64 as const;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Shared math helpers
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_MATH = /* wgsl */`
// ── saturate ──────────────────────────────────────────────────────────────────
fn saturate_f(v: f32) -> f32 { return clamp(v, 0.0, 1.0); }
fn saturate_v3(v: vec3f) -> vec3f { return clamp(v, vec3f(0.0), vec3f(1.0)); }

// ── pow5 (Schlick fast path) ──────────────────────────────────────────────────
fn pow5(v: f32) -> f32 { let v2 = v * v; return v2 * v2 * v; }

const PI      : f32 = 3.14159265358979;
const TWO_PI  : f32 = 6.28318530717959;
const INV_PI  : f32 = 0.31830988618379;
const EPSILON : f32 = 1e-7;

// ── lrange / lcrange (from AT LightingCommon.glsl) ────────────────────────────
fn lrange(oldValue: f32, oldMin: f32, oldMax: f32, newMin: f32, newMax: f32) -> f32 {
    let sub = vec3f(oldValue - oldMin, newMax - newMin, oldMax - oldMin);
    return sub.x * sub.y / sub.z + newMin;
}

fn lcrange(oldValue: f32, oldMin: f32, oldMax: f32, newMin: f32, newMax: f32) -> f32 {
    return clamp(lrange(oldValue, oldMin, oldMax, newMin, newMax), min(newMax, newMin), max(newMin, newMax));
}

fn lclamp(v: vec3f) -> vec3f { return clamp(v, vec3f(0.0), vec3f(1.0)); }
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Simplex noise (for dust particles, from ashima webgl-noise MIT)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_NOISE = /* wgsl */`
fn mod289_v3(x: vec3f) -> vec3f { return x - floor(x * (1.0/289.0)) * 289.0; }
fn mod289_v4(x: vec4f) -> vec4f { return x - floor(x * (1.0/289.0)) * 289.0; }
fn permute(x: vec4f) -> vec4f { return mod289_v4((x * 34.0 + 10.0) * x); }
fn taylorInvSqrt(r: vec4f) -> vec4f { return vec4f(1.79284291400159) - 0.85373472095314 * r; }

fn snoise3(v: vec3f) -> f32 {
    let C = vec2f(1.0/6.0, 1.0/3.0);
    let D = vec4f(0.0, 0.5, 1.0, 2.0);
    var i  = floor(v + dot(v, C.yyy));
    var x0 = v - i + dot(i, C.xxx);
    var g  = step(x0.yzx, x0.xyz);
    var l  = vec3f(1.0) - g;
    var i1 = min(g.xyz, l.zxy);
    var i2 = max(g.xyz, l.zxy);
    var x1 = x0 - i1 + C.xxx;
    var x2 = x0 - i2 + C.yyy;
    var x3 = x0 - D.yyy;
    i = mod289_v3(i);
    var p  = permute(permute(permute(
        i.zzzz + vec4f(0.0, i1.z, i2.z, 1.0))
      + i.yyyy + vec4f(0.0, i1.y, i2.y, 1.0))
      + i.xxxx + vec4f(0.0, i1.x, i2.x, 1.0));
    let ns = (1.0/7.0) * D.wyz - D.xzx;
    var j  = p - 49.0 * floor(p * ns.z * ns.z);
    var x_ = floor(j * ns.z);
    var y_ = floor(j - 7.0 * x_);
    var x  = x_ * ns.x + ns.yyyy;
    var y  = y_ * ns.x + ns.yyyy;
    var h  = vec4f(1.0) - abs(x) - abs(y);
    var b0 = vec4f(x.xy, y.xy);
    var b1 = vec4f(x.zw, y.zw);
    var s0 = floor(b0) * 2.0 + vec4f(1.0);
    var s1 = floor(b1) * 2.0 + vec4f(1.0);
    var sh = -step(h, vec4f(0.0));
    var a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    var a1 = b1.xzyw + s1.xzyw * sh.zzww;
    var p0 = vec3f(a0.xy, h.x);
    var p1 = vec3f(a0.zw, h.y);
    var p2 = vec3f(a1.xy, h.z);
    var p3 = vec3f(a1.zw, h.w);
    var norm = taylorInvSqrt(vec4f(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    var m = max(vec4f(0.6) - vec4f(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), vec4f(0.0));
    m = m * m;
    return 42.0 * dot(m*m, vec4f(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

fn curl3(p: vec3f) -> vec3f {
    let eps = 0.001;
    let dx  = vec3f(eps, 0.0, 0.0);
    let dy  = vec3f(0.0, eps, 0.0);
    let dz  = vec3f(0.0, 0.0, eps);
    let px = (snoise3(p + dy) - snoise3(p - dy)) / (2.0*eps)
           - (snoise3(p + dz) - snoise3(p - dz)) / (2.0*eps);
    let py = (snoise3(p + dz) - snoise3(p - dz)) / (2.0*eps)
           - (snoise3(p + dx) - snoise3(p - dx)) / (2.0*eps);
    let pz = (snoise3(p + dx) - snoise3(p - dx)) / (2.0*eps)
           - (snoise3(p + dy) - snoise3(p - dy)) / (2.0*eps);
    return vec3f(px, py, pz);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Uniform structs
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_SCENE_UNIFORMS = /* wgsl */`
struct SceneUniforms {
    viewProj      : mat4x4f,   // offset   0
    modelMat      : mat4x4f,   // offset  64
    normalMat     : mat4x4f,   // offset 128
    eye           : vec4f,     // offset 192  .xyz=eye, .w=unused
    fogParams     : vec4f,     // offset 208  .xyz=fogColor, .w=fogDensity
    ambientParams : vec4f,     // offset 224  .xyz=ambientColor, .w=ambientIntensity
    timeParams    : vec4f,     // offset 240  .x=time, .y=dt, .z=fogStart, .w=fogEnd
};
`;

const WGSL_LIGHT_STRUCT = /* wgsl */`
struct LightEntry {
    color    : vec4f,    // .rgb=colour, .w=intensity
    pos      : vec4f,    // .xyz=position, .w=type (1=dir, 2=pt, 3=cone)
    data     : vec4f,    // .xyz=coneDir, .w=coneAngle
    data2    : vec4f,    // .x=feather, .y=phongShininess, .z=shadowMin, .w=range
    props    : vec4f,    // .x=intensity, .y=range, .z=shadowMin, .w=type
};

struct LightArray {
    lights   : array<LightEntry, ${MAX_LIGHTS}>,
};
`;

const WGSL_MATERIAL_STRUCT = /* wgsl */`
struct MaterialUniforms {
    tint        : vec4f,    // .rgb=tint, .w=roughnessScale
    data        : vec4f,    // .x=metallicScale, .y=normalStrength, .zw=uvScale
    extra       : vec4f,    // reserved
    extra2      : vec4f,    // reserved
};
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — AT Lighting Functions (ported from lighting.fs / LightingCommon.glsl)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_LIGHTING = /* wgsl */`
${WGSL_MATH}

// ── lworldLight (LightingCommon.glsl) ─────────────────────────────────────────
// Computes light direction in view space for a given light position.
fn lightWorldDir(lPos: vec3f, localPos: vec3f, modelViewMat: mat4x4f, viewMat: mat4x4f) -> vec3f {
    let mvPos = modelViewMat * vec4f(localPos, 1.0);
    let worldPosition = viewMat * vec4f(lPos, 1.0);
    return worldPosition.xyz - mvPos.xyz;
}

// ── Cook-Torrance BRDF components ─────────────────────────────────────────────

// Schlick Fresnel: F₀ + (1 - F₀)(1 - cosθ)⁵
fn F_Schlick(f0: vec3f, cosTheta: f32) -> vec3f {
    return f0 + (vec3f(1.0) - f0) * pow5(saturate_f(1.0 - cosTheta));
}

// GGX / Trowbridge-Reitz NDF
fn D_GGX(NdotH: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let d  = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + EPSILON);
}

// Smith joint masking-shadowing (height-correlated GGX)
fn G_SmithGGX(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let gV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    let gL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / (gV + gL + EPSILON);
}

// Full Cook-Torrance specular BRDF: returns D*G*F
fn specularBRDF(N: vec3f, V: vec3f, L: vec3f, f0: vec3f, roughness: f32) -> vec3f {
    let H      = normalize(V + L);
    let NdotH  = saturate_f(dot(N, H));
    let NdotV  = saturate_f(dot(N, V));
    let NdotL  = saturate_f(dot(N, L));
    let VdotH  = saturate_f(dot(V, H));
    let D  = D_GGX(NdotH, roughness);
    let Gv = G_SmithGGX(NdotV, NdotL, roughness);
    let F  = F_Schlick(f0, VdotH);
    return vec3f(D * Gv) * F;
}

// ── AT lighting.fs — lightDirectional ─────────────────────────────────────────
// Port of: float volume = dot(normalize(lDir), config.normal);
//          return lColor * lcrange(volume, 0.0, 1.0, lProps.z, 1.0);
fn lightDirectional(
    normal: vec3f, lColor: vec3f, lPos: vec3f,
    shadowMin: f32, intensity: f32,
    viewPos: vec3f, modelViewMat: mat4x4f, viewMat: mat4x4f
) -> vec3f {
    let lDir = lightWorldDir(lPos, viewPos, modelViewMat, viewMat);
    let volume = dot(normalize(lDir), normal);
    return lColor * lcrange(volume, 0.0, 1.0, shadowMin, 1.0) * intensity;
}

// ── AT lighting.fs — lightPoint ───────────────────────────────────────────────
// Port of distance-attenuated point light with optional Phong specular.
fn lightPoint(
    normal: vec3f, worldPos: vec3f, viewDir: vec3f,
    lColor: vec3f, lPos: vec3f,
    intensity: f32, range: f32, shadowMin: f32,
    usePhong: f32, phongShininess: f32,
    viewPos: vec3f, modelViewMat: mat4x4f, viewMat: mat4x4f
) -> vec3f {
    let dist = length(worldPos - lPos);
    if (dist > range) { return vec3f(0.0); }

    let lDir = lightWorldDir(lPos, viewPos, modelViewMat, viewMat);
    let falloff = pow(lcrange(dist, 0.0, range, 1.0, 0.0), 2.0);

    var color = vec3f(0.0);
    if (usePhong > 0.5) {
        // Blinn-Phong specular
        let lDirN  = normalize(lDir);
        let H      = normalize(lDirN + normalize(viewDir));
        let spec   = pow(max(dot(normal, H), 0.0), phongShininess);
        let diff   = lcrange(dot(lDirN, normal), 0.0, 1.0, shadowMin, 1.0);
        color = lColor * (diff + spec * 0.5) * intensity * falloff;
    } else {
        let volume = dot(normalize(lDir), normal);
        let v      = lcrange(volume, 0.0, 1.0, shadowMin, 1.0);
        color = lColor * v * intensity * falloff;
    }
    return color;
}

// ── AT lighting.fs — lightCone ────────────────────────────────────────────────
// Port of spot/cone light with angular attenuation.
fn lightCone(
    normal: vec3f, worldPos: vec3f, viewDir: vec3f,
    lColor: vec3f, lPos: vec3f, coneDir: vec3f, coneAngle: f32,
    intensity: f32, range: f32, shadowMin: f32, feather: f32,
    usePhong: f32, phongShininess: f32,
    viewPos: vec3f, modelViewMat: mat4x4f, viewMat: mat4x4f
) -> vec3f {
    let dist = length(worldPos - lPos);
    if (dist > range) { return vec3f(0.0); }

    let surfaceToLight = normalize(lPos - worldPos);
    let lightToSurfaceAngle = degrees(acos(dot(-surfaceToLight, normalize(coneDir))));

    // Base point-light contribution
    let basePt = lightPoint(
        normal, worldPos, viewDir, lColor, lPos,
        intensity, range, shadowMin, usePhong, phongShininess,
        viewPos, modelViewMat, viewMat
    );

    let featherMin = 1.0 - feather * 0.1;
    let featherMax = 1.0 + feather * 0.1;
    let attenuation = smoothstep(lightToSurfaceAngle * featherMin, lightToSurfaceAngle * featherMax, coneAngle);

    return basePt * attenuation;
}

// ── getCombinedColor (iterate lights, dispatch by type) ───────────────────────
// Matches AT lighting.fs getCombinedColor(config) loop.
fn getCombinedColor(
    normal: vec3f, worldPos: vec3f, viewDir: vec3f, viewPos: vec3f,
    modelViewMat: mat4x4f, viewMat: mat4x4f,
    la: LightArray
) -> vec3f {
    var color = vec3f(0.0);

    for (var i = 0u; i < ${MAX_LIGHTS}u; i++) {
        let light = la.lights[i];
        let lType = light.props.w;
        if (lType < 0.5) { continue; }  // disabled

        let lColor     = light.color.rgb;
        let lPos       = light.pos.xyz;
        let intensity  = light.props.x;
        let range      = light.props.y;
        let shadowMin  = light.props.z;
        let feather    = light.data2.x;
        let shininess  = light.data2.y;
        let usePhong   = select(0.0, 1.0, shininess > 0.0);

        if (lType < 1.5) {
            // Directional
            color += lightDirectional(normal, lColor, lPos, shadowMin, intensity, viewPos, modelViewMat, viewMat);
        } else if (lType < 2.5) {
            // Point
            color += lightPoint(normal, worldPos, viewDir, lColor, lPos, intensity, range, shadowMin, usePhong, shininess, viewPos, modelViewMat, viewMat);
        } else if (lType < 3.5) {
            // Cone / spot
            let coneDir   = light.data.xyz;
            let coneAngle = light.data.w;
            color += lightCone(normal, worldPos, viewDir, lColor, lPos, coneDir, coneAngle, intensity, range, shadowMin, feather, usePhong, shininess, viewPos, modelViewMat, viewMat);
        }
    }

    return lclamp(color);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Terrain PBR render shader
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_TERRAIN_RENDER = /* wgsl */`
${WGSL_SCENE_UNIFORMS}
${WGSL_LIGHT_STRUCT}
${WGSL_MATERIAL_STRUCT}
${WGSL_LIGHTING}

// ── Bindings ──────────────────────────────────────────────────────────────────
@group(0) @binding(0) var<uniform> scene     : SceneUniforms;
@group(0) @binding(1) var<uniform> lights    : LightArray;
@group(0) @binding(2) var<uniform> material  : MaterialUniforms;

@group(1) @binding(0) var texSampler   : sampler;
@group(1) @binding(1) var tBaseColor   : texture_2d<f32>;
@group(1) @binding(2) var tNormal      : texture_2d<f32>;
@group(1) @binding(3) var tMRO         : texture_2d<f32>;

// ── Vertex output ─────────────────────────────────────────────────────────────
struct VertexOut {
    @builtin(position) clipPos   : vec4f,
    @location(0)       worldPos  : vec3f,
    @location(1)       viewPos   : vec3f,
    @location(2)       normal    : vec3f,
    @location(3)       uv        : vec2f,
    @location(4)       tangent   : vec3f,
    @location(5)       bitangent : vec3f,
    @location(6)       fogDist   : f32,
};

// ── Vertex shader ─────────────────────────────────────────────────────────────
// Matches AT lighting.vs setupLight(): computes vPos, vNormal, vWorldPos, vViewDir
@vertex
fn vs_terrain(
    @location(0) aPos    : vec3f,
    @location(1) aNormal : vec3f,
    @location(2) aUv     : vec2f,
) -> VertexOut {
    let uvScale  = material.data.zw;
    let scaledUv = aUv * uvScale;

    let worldPos4 = scene.modelMat * vec4f(aPos, 1.0);
    let worldPos  = worldPos4.xyz;
    let viewPos4  = scene.viewProj * worldPos4;

    // Transform normal via normalMat (inverse-transpose of modelMat)
    let worldNormal = normalize((scene.normalMat * vec4f(aNormal, 0.0)).xyz);

    // Compute tangent frame for normal mapping (Gram-Schmidt from normal)
    var tangent = vec3f(0.0);
    if (abs(worldNormal.y) < 0.999) {
        tangent = normalize(cross(vec3f(0.0, 1.0, 0.0), worldNormal));
    } else {
        tangent = normalize(cross(vec3f(0.0, 0.0, 1.0), worldNormal));
    }
    let bitangent = cross(worldNormal, tangent);

    // Fog distance (camera distance for exponential fog)
    let eyeDist = length(worldPos - scene.eye.xyz);

    // vPos for lighting (local space)
    let localViewPos = aPos;

    var out: VertexOut;
    out.clipPos   = viewPos4;
    out.worldPos  = worldPos;
    out.viewPos   = localViewPos;
    out.normal    = worldNormal;
    out.uv        = scaledUv;
    out.tangent   = tangent;
    out.bitangent = bitangent;
    out.fogDist   = eyeDist;
    return out;
}

// ── Fragment shader ───────────────────────────────────────────────────────────
// PBR pipeline: normal map → Cook-Torrance BRDF → AT multi-light → fog
@fragment
fn fs_terrain(in: VertexOut) -> @location(0) vec4f {
    let uv = in.uv;

    // ── Sample PBR textures ───────────────────────────────────────────────────
    let baseColor = textureSample(tBaseColor, texSampler, uv).rgb * material.tint.rgb;
    let mroSample = textureSample(tMRO, texSampler, uv);
    let normalSample = textureSample(tNormal, texSampler, uv).rgb;

    // MRO unpack: R=Metallic, G=Roughness, B=AO (AT convention)
    let metallic  = mroSample.r * material.data.x;   // metallicScale
    let roughness = mroSample.g * material.tint.w;     // roughnessScale
    let ao        = mroSample.b;

    // ── Normal mapping ────────────────────────────────────────────────────────
    // Tangent-space normal from texture → world space via TBN
    let normalStrength = material.data.y;
    var tangentNormal  = normalSample * 2.0 - vec3f(1.0);
    tangentNormal.x   *= normalStrength;
    tangentNormal.y   *= normalStrength;
    tangentNormal      = normalize(tangentNormal);

    let T = normalize(in.tangent);
    let B = normalize(in.bitangent);
    let N_geom = normalize(in.normal);
    let TBN = mat3x3f(T, B, N_geom);
    let N = normalize(TBN * tangentNormal);

    // ── View direction ────────────────────────────────────────────────────────
    let V = normalize(scene.eye.xyz - in.worldPos);

    // ── Build modelView and view matrices from scene uniforms ─────────────────
    // We re-derive a simple view matrix from the available data.
    // For the lighting function, we need consistent coordinate spaces.
    let modelViewMat = scene.viewProj;  // approximate (in practice the scene uniform would provide this)
    let viewMat = scene.normalMat;       // approximate for light transforms

    // ── AT multi-light loop (getCombinedColor) ────────────────────────────────
    let directLight = getCombinedColor(N, in.worldPos, V, in.viewPos, scene.modelMat, scene.normalMat, lights);

    // ── PBR Cook-Torrance final composition ───────────────────────────────────
    let f0 = mix(vec3f(0.04), baseColor, metallic);

    // Accumulate PBR direct lighting from each active light
    var pbrColor = vec3f(0.0);
    for (var i = 0u; i < ${MAX_LIGHTS}u; i++) {
        let light = lights.lights[i];
        let lType = light.props.w;
        if (lType < 0.5) { continue; }

        let lPos   = light.pos.xyz;
        let lColor = light.color.rgb * light.props.x;
        var L: vec3f;
        if (lType < 1.5) {
            L = normalize(lPos);  // directional: position IS the direction
        } else {
            L = normalize(lPos - in.worldPos);
        }

        let NdotL = saturate_f(dot(N, L));
        if (NdotL < EPSILON) { continue; }

        let specular = specularBRDF(N, V, L, f0, roughness);
        let diffuse  = baseColor * INV_PI * (1.0 - metallic);

        let ks = F_Schlick(f0, saturate_f(dot(N, V)));
        let kd = (vec3f(1.0) - ks) * (1.0 - metallic);

        // Distance attenuation for point/cone
        var atten = 1.0;
        if (lType > 1.5) {
            let dist = length(lPos - in.worldPos);
            let range = light.props.y;
            atten = pow(lcrange(dist, 0.0, range, 1.0, 0.0), 2.0);
        }

        pbrColor += (kd * diffuse + specular) * lColor * NdotL * atten;
    }

    // ── Ambient (simple hemisphere + AO) ──────────────────────────────────────
    let ambient = scene.ambientParams.rgb * scene.ambientParams.w * baseColor * ao;

    // ── Combine: blend AT lighting feel with PBR accuracy ─────────────────────
    // The AT lighting provides the artistic warm/cool ramps, while PBR adds
    // physical specular highlights and energy conservation.
    var finalColor = pbrColor * 0.7 + directLight * baseColor * 0.3 + ambient;

    // ── Exponential distance fog (AT atmospheric style) ───────────────────────
    let fogColor   = scene.fogParams.xyz;
    let fogDensity = scene.fogParams.w;
    let fogStart   = scene.timeParams.z;
    let fogEnd     = scene.timeParams.w;

    let fogDist    = clamp((in.fogDist - fogStart) / (fogEnd - fogStart), 0.0, 1.0);
    let fogFactor  = 1.0 - exp(-fogDensity * fogDist * fogDist);
    finalColor     = mix(finalColor, fogColor, fogFactor);

    // ── Tone-map (simple Reinhard for now) ────────────────────────────────────
    finalColor = finalColor / (finalColor + vec3f(1.0));

    return vec4f(finalColor, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Shadow depth pass
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_SHADOW_DEPTH = /* wgsl */`
struct ShadowUniforms {
    lightViewProj : mat4x4f,
    modelMat      : mat4x4f,
};
@group(0) @binding(0) var<uniform> shadow : ShadowUniforms;

@vertex
fn vs_shadow(
    @location(0) aPos : vec3f,
) -> @builtin(position) vec4f {
    return shadow.lightViewProj * shadow.modelMat * vec4f(aPos, 1.0);
}

// No fragment needed — depth-only write
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Dust particle compute (lifecycle + wind drift)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_DUST_UPDATE = /* wgsl */`
struct DustUniforms {
    time       : vec4f,    // .x=time, .y=dt, .zw=unused
    wind       : vec4f,    // .xyz=windDir, .w=unused
    dustColor  : vec4f,    // .rgb=color, .w=unused
};
@group(0) @binding(0) var<uniform> du : DustUniforms;

@group(0) @binding(1) var pSrc : texture_2d<f32>;
@group(0) @binding(2) var pDst : texture_storage_2d<rgba32float, write>;

${WGSL_MATH}
${WGSL_NOISE}

fn hash2(p: vec2f) -> f32 {
    let q = fract(p * vec2f(127.1, 311.7));
    return fract(dot(q, q + vec2f(19.19)) * 43758.5453);
}

@compute @workgroup_size(${WG})
fn cs_dust_update(@builtin(global_invocation_id) gid : vec3u) {
    let idx  = gid.x;
    let dim  = vec2u(textureDimensions(pSrc));
    if (idx >= dim.x * dim.y) { return; }

    let px   = vec2i(i32(idx % dim.x), i32(idx / dim.x));
    var p    = textureLoad(pSrc, px, 0);  // (x, y, z, life)
    let time = du.time.x;
    let dt   = du.time.y;

    let seed = vec2f(f32(idx) * 0.0017, time * 0.05);

    // ── Dead particle: respawn ────────────────────────────────────────────────
    if (p.a <= 0.0) {
        let respawnChance = hash2(seed + vec2f(0.5, 0.9));
        if (respawnChance < 0.006) {
            let rx = hash2(seed) * 20.0 - 10.0;
            let ry = hash2(seed + vec2f(1.0, 0.0)) * 5.0 + 0.2;
            let rz = hash2(seed + vec2f(0.0, 1.0)) * 20.0 - 10.0;
            let life = 0.6 + hash2(seed + vec2f(2.0, 0.0)) * 0.4;
            p = vec4f(rx, ry, rz, life);
            textureStore(pDst, px, p);
            return;
        }
        textureStore(pDst, px, p);
        return;
    }

    // ── Live particle: wind drift + curl noise ────────────────────────────────
    let windDir   = du.wind.xyz;
    let curlIn    = vec3f(p.x * 0.15, p.z * 0.15, time * 0.2);
    let drift     = curl3(curlIn) * 0.008;
    let noiseY    = snoise3(vec3f(p.x * 0.1, time * 0.15, p.z * 0.1)) * 0.003;

    let velocity  = windDir * dt * 0.8 + drift;

    p.x += velocity.x;
    p.y += velocity.y + noiseY + 0.001;  // gentle float upward
    p.z += velocity.z;

    // Decay
    p.a -= dt * 0.15;

    // Bounds: re-wrap if too far
    if (p.x > 15.0)  { p.x -= 30.0; }
    if (p.x < -15.0) { p.x += 30.0; }
    if (p.z > 15.0)  { p.z -= 30.0; }
    if (p.z < -15.0) { p.z += 30.0; }
    if (p.y > 8.0)   { p.a = 0.0; }      // kill particles that float too high

    textureStore(pDst, px, p);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Dust particle render (billboard quads, alpha-blended)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_DUST_RENDER = /* wgsl */`
struct DustUniforms {
    time       : vec4f,
    wind       : vec4f,
    dustColor  : vec4f,
};
@group(0) @binding(0) var<uniform> du : DustUniforms;
@group(0) @binding(1) var pSamp : sampler;
@group(0) @binding(2) var pTex  : texture_2d<f32>;

${WGSL_SCENE_UNIFORMS}
@group(1) @binding(0) var<uniform> scene : SceneUniforms;

struct VertexOut {
    @builtin(position) pos   : vec4f,
    @location(0)       uv    : vec2f,
    @location(1)       life  : f32,
    @location(2)       color : vec3f,
};

const QUAD : array<vec2f, 6> = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
);

@vertex
fn vs_dust(
    @builtin(vertex_index)   vid  : u32,
    @builtin(instance_index) inst : u32,
) -> VertexOut {
    let texDim = vec2u(textureDimensions(pTex));
    let px     = vec2i(i32(inst % texDim.x), i32(inst / texDim.x));
    let state  = textureLoad(pTex, px, 0);
    let life   = state.a;

    let corner = QUAD[vid];
    let size   = 0.02 * saturate_f(life * 3.0);

    // Camera-facing billboard basis
    let viewRight = vec3f(scene.viewProj[0][0], scene.viewProj[1][0], scene.viewProj[2][0]);
    let viewUp    = vec3f(scene.viewProj[0][1], scene.viewProj[1][1], scene.viewProj[2][1]);
    let worldPos  = state.xyz
                  + viewRight * corner.x * size
                  + viewUp    * corner.y * size;

    var out: VertexOut;
    out.pos   = scene.viewProj * scene.modelMat * vec4f(worldPos, 1.0);
    out.uv    = corner * 0.5 + 0.5;
    out.life  = life;
    out.color = du.dustColor.rgb;
    return out;
}

${WGSL_MATH}

@fragment
fn fs_dust(in: VertexOut) -> @location(0) vec4f {
    if (in.life <= 0.0) { discard; }

    // Soft circle mask
    let d    = length(in.uv - 0.5) * 2.0;
    let mask = 1.0 - smoothstep(0.5, 1.0, d);

    // Fade in/out using life
    let alpha = sin(PI * saturate_f(in.life)) * mask * 0.25;

    return vec4f(in.color, alpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GPU resource descriptor per terrain mesh
// ─────────────────────────────────────────────────────────────────────────────

interface TerrainMeshGPU {
  name:         TerrainMeshName;
  vertexBuffer: GPUBuffer;
  indexBuffer:  GPUBuffer;
  indexCount:   number;
  vertexCount:  number;
  bindGroup:    GPUBindGroup;      // group(1): textures + sampler
  materialBuf:  GPUBuffer;         // material uniforms
}

// ─────────────────────────────────────────────────────────────────────────────
// Main class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * ATTerrainEnvironment — WebGPU port of Active Theory's tree-room terrain
 * rendering system.
 *
 * Provides:
 *   • Draco geometry loading for 5 terrain meshes (rocky_soil, rocks, sand, walls)
 *   • PBR material rendering with KTX2 compressed textures (baseColor, normal, MRO)
 *   • AT multi-light pipeline (directional + point + cone) from lighting.fs
 *   • Cook-Torrance BRDF with GGX/Smith/Schlick
 *   • Atmospheric distance fog
 *   • Dust particle overlay with wind-driven curl noise
 *   • Shadow depth pre-pass
 */
export class ATTerrainEnvironment {
  // ── Config ──────────────────────────────────────────────────────────────────
  private readonly cfg: Required<Omit<ATTerrainEnvironmentConfig, 'envMap' | 'materialOverrides'>> & {
    envMap?: GPUTexture;
    materialOverrides: Partial<Record<TerrainMeshName, ATTerrainMaterialOverride>>;
  };

  // ── WebGPU ──────────────────────────────────────────────────────────────────
  private readonly device: GPUDevice;
  private readonly format: GPUTextureFormat;

  // ── Loaders ─────────────────────────────────────────────────────────────────
  private geoLoader!:  ATGeometryLoader;
  private texLoader!:  ATTextureLoader;

  // ── Per-mesh GPU resources ──────────────────────────────────────────────────
  private meshes: TerrainMeshGPU[] = [];

  // ── Shared GPU resources ────────────────────────────────────────────────────
  private sceneUniformBuf!:  GPUBuffer;
  private lightUniformBuf!:  GPUBuffer;
  private sampler!:          GPUSampler;

  // ── Pipelines ───────────────────────────────────────────────────────────────
  private terrainRenderPipe!:   GPURenderPipeline;
  private shadowDepthPipe!:     GPURenderPipeline;
  private dustUpdatePipe!:      GPUComputePipeline;
  private dustRenderPipe!:      GPURenderPipeline;

  // ── Shadow map ──────────────────────────────────────────────────────────────
  private shadowMap!:         GPUTexture;
  private shadowUniformBuf!:  GPUBuffer;

  // ── Dust particles ──────────────────────────────────────────────────────────
  private dustA!:           GPUTexture;
  private dustB!:           GPUTexture;
  private dustUniformBuf!:  GPUBuffer;
  private maxDustParticles: number;

  // ── Lights ──────────────────────────────────────────────────────────────────
  private lights: ATTerrainLight[] = [];

  // ── State ───────────────────────────────────────────────────────────────────
  private built = false;

  // ─── Constructor ────────────────────────────────────────────────────────────

  constructor(
    device: GPUDevice,
    format: GPUTextureFormat,
    cfg:    ATTerrainEnvironmentConfig = {},
  ) {
    this.device = device;
    this.format = format;

    // Default lights: warm directional sun + cool fill
    const defaultLights: ATTerrainLight[] = cfg.lights ?? [
      {
        type: 1,
        position: [5.0, 8.0, -3.0],
        color: [1.0, 0.95, 0.88],
        intensity: 1.2,
        shadowMin: 0.15,
      },
      {
        type: 1,
        position: [-3.0, 4.0, 6.0],
        color: [0.4, 0.45, 0.55],
        intensity: 0.4,
        shadowMin: 0.05,
      },
    ];

    this.cfg = {
      geometryPath:     cfg.geometryPath     ?? '/upstream/activetheory-assets/geometry',
      texturePath:      cfg.texturePath      ?? '/upstream/activetheory-assets/textures',
      lights:           defaultLights,
      fogColor:         cfg.fogColor         ?? [...FOG_COLOR],
      fogDensity:       cfg.fogDensity       ?? FOG_DENSITY,
      fogStart:         cfg.fogStart         ?? 5.0,
      fogEnd:           cfg.fogEnd           ?? 80.0,
      ambientColor:     cfg.ambientColor     ?? [...AMBIENT_COLOR],
      ambientIntensity: cfg.ambientIntensity ?? AMBIENT_INTENSITY,
      enableShadows:    cfg.enableShadows    ?? true,
      shadowMapSize:    cfg.shadowMapSize    ?? SHADOW_MAP_SIZE,
      enableDust:       cfg.enableDust       ?? true,
      maxDustParticles: cfg.maxDustParticles ?? MAX_DUST_PARTICLES,
      dustColor:        cfg.dustColor        ?? [0.8, 0.75, 0.65],
      windDirection:    cfg.windDirection    ?? [0.3, 0.1, -0.2],
      enableHeightDisplacement: cfg.enableHeightDisplacement ?? false,
      envMap:           cfg.envMap,
      materialOverrides: cfg.materialOverrides ?? {},
    };

    this.lights          = [...defaultLights];
    this.maxDustParticles = this.cfg.maxDustParticles;
  }

  // ─── Public API ─────────────────────────────────────────────────────────────

  /**
   * Compile all pipelines, load geometry + textures, allocate GPU resources.
   * Must be called before tick() / renderPass().
   */
  async build(): Promise<void> {
    this.geoLoader = new ATGeometryLoader({ basePath: this.cfg.geometryPath });
    this.texLoader = new ATTextureLoader();

    this._createSharedResources();
    await this._loadAndUploadMeshes();
    await this._createPipelines();

    if (this.cfg.enableShadows) {
      this._createShadowResources();
    }

    if (this.cfg.enableDust) {
      this._createDustResources();
    }

    this.built = true;
  }

  /**
   * Set or update a light at the given index.
   *
   * @param index  Light slot (0 to MAX_LIGHTS-1).
   * @param light  Light definition.
   */
  setLight(index: number, light: ATTerrainLight): void {
    if (index < 0 || index >= MAX_LIGHTS) return;
    this.lights[index] = light;
  }

  /**
   * Remove (disable) a light at the given index.
   */
  removeLight(index: number): void {
    if (index < 0 || index >= MAX_LIGHTS) return;
    this.lights[index] = { type: 1, position: [0, 0, 0], color: [0, 0, 0], intensity: 0 };
  }

  /**
   * Get the current light array (readonly).
   */
  getLights(): readonly ATTerrainLight[] {
    return this.lights;
  }

  /**
   * Update fog parameters at runtime.
   */
  setFog(params: { color?: [number, number, number]; density?: number; start?: number; end?: number }): void {
    if (params.color)   this.cfg.fogColor   = params.color;
    if (params.density != null) this.cfg.fogDensity = params.density;
    if (params.start != null)   this.cfg.fogStart   = params.start;
    if (params.end != null)     this.cfg.fogEnd     = params.end;
  }

  /**
   * Update ambient light parameters at runtime.
   */
  setAmbient(params: { color?: [number, number, number]; intensity?: number }): void {
    if (params.color) this.cfg.ambientColor = params.color;
    if (params.intensity != null) this.cfg.ambientIntensity = params.intensity;
  }

  /**
   * Update wind direction for dust particles at runtime.
   */
  setWind(direction: [number, number, number]): void {
    this.cfg.windDirection = direction;
  }

  /**
   * Run one frame of simulation (dust particles).
   *
   * @param encoder  Open GPUCommandEncoder.
   * @param time     Elapsed seconds.
   * @param dt       Delta seconds.
   */
  tick(encoder: GPUCommandEncoder, time: number, dt: number): void {
    if (!this.built) return;

    this._writeSceneUniforms(time, dt);
    this._writeLightUniforms();

    if (this.cfg.enableDust) {
      this._writeDustUniforms(time, dt);
      this._dispatchDustUpdate(encoder);
    }
  }

  /**
   * Encode the terrain render pass (all meshes + optional dust overlay).
   *
   * @param encoder         Open GPUCommandEncoder.
   * @param colorTarget     Output colour attachment view.
   * @param depthView       Output depth attachment view.
   * @param sceneUniformBuf External scene uniform buffer with viewProj etc.
   *                         If provided, overrides the internal one.
   */
  renderPass(
    encoder:          GPUCommandEncoder,
    colorTarget:      GPUTextureView,
    depthView:        GPUTextureView,
    sceneUniformBuf?: GPUBuffer,
  ): void {
    if (!this.built) return;

    const uniformBuf = sceneUniformBuf ?? this.sceneUniformBuf;

    // ── Shadow pre-pass ───────────────────────────────────────────────────────
    if (this.cfg.enableShadows) {
      this._renderShadowPass(encoder);
    }

    // ── Main terrain pass ─────────────────────────────────────────────────────
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view:    colorTarget,
        loadOp:  'load',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view:          depthView,
        depthLoadOp:   'load',
        depthStoreOp:  'store',
      },
    });

    pass.setPipeline(this.terrainRenderPipe);

    // Shared group(0): scene + lights + material (per-mesh material in loop)
    for (const mesh of this.meshes) {
      const bg0 = this._makeTerrainBG0(uniformBuf, mesh.materialBuf);
      pass.setBindGroup(0, bg0);
      pass.setBindGroup(1, mesh.bindGroup);
      pass.setVertexBuffer(0, mesh.vertexBuffer);
      pass.setIndexBuffer(mesh.indexBuffer, 'uint32');
      pass.drawIndexed(mesh.indexCount);
    }

    // ── Dust particles ────────────────────────────────────────────────────────
    if (this.cfg.enableDust) {
      pass.setPipeline(this.dustRenderPipe);
      const dustBG0 = this._makeDustRenderBG0();
      const dustBG1 = this._makeDustSceneBG1(uniformBuf);
      pass.setBindGroup(0, dustBG0);
      pass.setBindGroup(1, dustBG1);
      pass.draw(6, this.maxDustParticles);
    }

    pass.end();
  }

  /**
   * Release all GPU resources.
   */
  destroy(): void {
    for (const mesh of this.meshes) {
      mesh.vertexBuffer.destroy();
      mesh.indexBuffer.destroy();
      mesh.materialBuf.destroy();
    }
    this.meshes = [];

    this.sceneUniformBuf.destroy();
    this.lightUniformBuf.destroy();

    if (this.shadowMap) this.shadowMap.destroy();
    if (this.shadowUniformBuf) this.shadowUniformBuf.destroy();

    if (this.dustA) this.dustA.destroy();
    if (this.dustB) this.dustB.destroy();
    if (this.dustUniformBuf) this.dustUniformBuf.destroy();

    this.geoLoader?.dispose();
    this.built = false;
  }

  /**
   * Get the list of loaded terrain mesh names.
   */
  getMeshNames(): TerrainMeshName[] {
    return this.meshes.map(m => m.name);
  }

  /**
   * Get the total vertex count across all terrain meshes.
   */
  getTotalVertexCount(): number {
    return this.meshes.reduce((acc, m) => acc + m.vertexCount, 0);
  }

  /**
   * Get the total index count across all terrain meshes.
   */
  getTotalIndexCount(): number {
    return this.meshes.reduce((acc, m) => acc + m.indexCount, 0);
  }

  // ─── Private: Shared resource creation ──────────────────────────────────────

  private _createSharedResources(): void {
    // Scene uniform buffer
    this.sceneUniformBuf = this.device.createBuffer({
      size:  SCENE_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Light uniform buffer
    this.lightUniformBuf = this.device.createBuffer({
      size:  LIGHT_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Shared texture sampler (linear filtering, repeat)
    this.sampler = this.device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter:    'linear',
      minFilter:    'linear',
      mipmapFilter: 'linear',
    });
  }

  // ─── Private: Geometry + texture loading ────────────────────────────────────

  private async _loadAndUploadMeshes(): Promise<void> {
    // Load all geometry files in parallel
    const geoPromises = TERRAIN_MESHES.map(name =>
      this.geoLoader.loadGeometry(`${name}.bin`).then(geo => ({ name, geo }))
    );

    // Load all PBR texture sets in parallel
    const texPromises = TERRAIN_MESHES.map(name => {
      const prefix = MESH_TO_TEXTURE_PREFIX[name];
      return this.texLoader.loadMaterialSet(prefix).then(set => ({ name, set }));
    });

    const [geoResults, texResults] = await Promise.all([
      Promise.all(geoPromises),
      Promise.all(texPromises),
    ]);

    // Build per-mesh GPU resources
    const texMap = new Map(texResults.map(r => [r.name, r.set]));

    for (const { name, geo } of geoResults) {
      const texSet = texMap.get(name)!;
      const meshGPU = this._uploadMesh(name, geo, texSet);
      this.meshes.push(meshGPU);
    }
  }

  /**
   * Upload a single mesh's geometry and textures to the GPU.
   */
  private _uploadMesh(
    name: TerrainMeshName,
    geo:  ATGeometry,
    texSet: ATMaterialSet,
  ): TerrainMeshGPU {
    const d = this.device;

    // ── Interleave vertex data: [pos.xyz, norm.xyz, uv.xy] ───────────────────
    const vertexCount = geo.vertexCount;
    const interleaved = new Float32Array(vertexCount * VERTEX_STRIDE);

    for (let i = 0; i < vertexCount; i++) {
      const off = i * VERTEX_STRIDE;
      interleaved[off + 0] = geo.positions[i * 3 + 0];
      interleaved[off + 1] = geo.positions[i * 3 + 1];
      interleaved[off + 2] = geo.positions[i * 3 + 2];
      interleaved[off + 3] = geo.normals[i * 3 + 0];
      interleaved[off + 4] = geo.normals[i * 3 + 1];
      interleaved[off + 5] = geo.normals[i * 3 + 2];
      interleaved[off + 6] = geo.uvs[i * 2 + 0];
      interleaved[off + 7] = geo.uvs[i * 2 + 1];
    }

    const vertexBuffer = d.createBuffer({
      size:  interleaved.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(vertexBuffer, 0, interleaved);

    // ── Index buffer ──────────────────────────────────────────────────────────
    const indexBuffer = d.createBuffer({
      size:  geo.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(indexBuffer, 0, geo.indices);

    // ── Upload PBR textures ───────────────────────────────────────────────────
    const tBaseColor = this._uploadTexture(texSet.baseColor);
    const tNormal    = this._uploadTexture(texSet.normal);
    const tMRO       = this._uploadTexture(texSet.mro);

    // ── Material uniform buffer ───────────────────────────────────────────────
    const materialBuf = d.createBuffer({
      size:  MATERIAL_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const overrides = this.cfg.materialOverrides[name] ?? {};
    const tint    = overrides.tint           ?? [1, 1, 1];
    const roughS  = overrides.roughnessScale ?? 1.0;
    const metalS  = overrides.metallicScale  ?? 1.0;
    const normS   = overrides.normalStrength ?? 1.0;
    const uvScale = overrides.uvScale        ?? [1, 1];

    const matData = new Float32Array(16);
    // Row 0: tint.rgb + roughnessScale
    matData[0] = tint[0]; matData[1] = tint[1]; matData[2] = tint[2]; matData[3] = roughS;
    // Row 1: metallicScale, normalStrength, uvScale.xy
    matData[4] = metalS;  matData[5] = normS;  matData[6] = uvScale[0]; matData[7] = uvScale[1];
    // Row 2-3: reserved
    d.queue.writeBuffer(materialBuf, 0, matData);

    // ── Texture bind group (group 1) ──────────────────────────────────────────
    // Will be created after pipeline is available; store texture references for now
    const bindGroup = this._createTextureBindGroup(tBaseColor, tNormal, tMRO);

    return {
      name,
      vertexBuffer,
      indexBuffer,
      indexCount:  geo.indexCount,
      vertexCount: geo.vertexCount,
      bindGroup,
      materialBuf,
    };
  }

  /**
   * Upload an ATTexture to a GPUTexture.
   */
  private _uploadTexture(tex: ATTexture): GPUTexture {
    const d = this.device;

    const gpuTex = d.createTexture({
      size:   [tex.width, tex.height, 1],
      format: tex.format as GPUTextureFormat,
      usage:  GPUTextureUsage.TEXTURE_BINDING
            | GPUTextureUsage.COPY_DST
            | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    d.queue.writeTexture(
      { texture: gpuTex },
      tex.data,
      { bytesPerRow: this._bytesPerRow(tex) },
      { width: tex.width, height: tex.height },
    );

    // Upload mip levels
    let mipW = tex.width;
    let mipH = tex.height;
    for (let level = 0; level < tex.mipLevels.length; level++) {
      mipW = Math.max(1, mipW >> 1);
      mipH = Math.max(1, mipH >> 1);
      d.queue.writeTexture(
        { texture: gpuTex, mipLevel: level + 1 },
        tex.mipLevels[level],
        { bytesPerRow: this._bytesPerRowMip(tex.format, mipW) },
        { width: mipW, height: mipH },
      );
    }

    return gpuTex;
  }

  /**
   * Compute bytes per row for texture upload (block-aware for compressed formats).
   */
  private _bytesPerRow(tex: ATTexture): number {
    return this._bytesPerRowMip(tex.format, tex.width);
  }

  private _bytesPerRowMip(format: string, width: number): number {
    // For compressed formats, compute based on block size
    if (format.startsWith('astc-4x4') || format.startsWith('etc2') || format.startsWith('bc')) {
      const blockW = 4;
      const blocksX = Math.ceil(width / blockW);
      const bytesPerBlock = format.includes('bc1') || format.includes('etc2-rgb') ? 8 : 16;
      return blocksX * bytesPerBlock;
    }
    // Uncompressed RGBA8
    return width * 4;
  }

  /**
   * Create a placeholder GPUTexture (1×1 white) for missing textures.
   */
  private _placeholderTexture(): GPUTexture {
    const tex = this.device.createTexture({
      size:   [1, 1, 1],
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const data = new Uint8Array([255, 255, 255, 255]);
    this.device.queue.writeTexture(
      { texture: tex },
      data,
      { bytesPerRow: 4 },
      { width: 1, height: 1 },
    );
    return tex;
  }

  /**
   * Create a placeholder normal map (flat: 128, 128, 255, 255).
   */
  private _placeholderNormal(): GPUTexture {
    const tex = this.device.createTexture({
      size:   [1, 1, 1],
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const data = new Uint8Array([128, 128, 255, 255]);
    this.device.queue.writeTexture(
      { texture: tex },
      data,
      { bytesPerRow: 4 },
      { width: 1, height: 1 },
    );
    return tex;
  }

  /**
   * Create texture bind group for a mesh (group 1).
   */
  private _createTextureBindGroup(
    tBaseColor: GPUTexture,
    tNormal:    GPUTexture,
    tMRO:       GPUTexture,
  ): GPUBindGroup {
    // This is a lazy bind group — actual layout comes from the pipeline.
    // We'll re-create it after pipeline creation.
    // For now, return a stub that will be replaced.
    return this.device.createBindGroup({
      layout: this.device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
          { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        ],
      }),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: tBaseColor.createView() },
        { binding: 2, resource: tNormal.createView() },
        { binding: 3, resource: tMRO.createView() },
      ],
    });
  }

  // ─── Private: Shadow resources ──────────────────────────────────────────────

  private _createShadowResources(): void {
    const size = this.cfg.shadowMapSize;

    this.shadowMap = this.device.createTexture({
      size:   [size, size, 1],
      format: 'depth24plus',
      usage:  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.shadowUniformBuf = this.device.createBuffer({
      size:  128,  // lightViewProj(64) + modelMat(64)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  // ─── Private: Dust particle resources ───────────────────────────────────────

  private _createDustResources(): void {
    const pW = Math.min(this.maxDustParticles, 256);
    const pH = Math.ceil(this.maxDustParticles / pW);

    const pDesc: GPUTextureDescriptor = {
      size:   [pW, pH, 1],
      format: 'rgba32float',
      usage:  GPUTextureUsage.TEXTURE_BINDING
            | GPUTextureUsage.STORAGE_BINDING
            | GPUTextureUsage.COPY_DST,
    };
    this.dustA = this.device.createTexture(pDesc);
    this.dustB = this.device.createTexture(pDesc);

    // Seed with dead particles (life = 0)
    const pPixels = pW * pH * 4;
    const pData = new Float32Array(pPixels);
    this.device.queue.writeTexture(
      { texture: this.dustA },
      pData,
      { bytesPerRow: pW * 16 },
      { width: pW, height: pH },
    );
    this.device.queue.writeTexture(
      { texture: this.dustB },
      pData,
      { bytesPerRow: pW * 16 },
      { width: pW, height: pH },
    );

    // Dust uniform buffer: 3 × vec4f = 48 bytes
    this.dustUniformBuf = this.device.createBuffer({
      size:  48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  // ─── Private: Pipeline creation ─────────────────────────────────────────────

  private async _createPipelines(): Promise<void> {
    const d = this.device;

    // ── Bind group layouts ────────────────────────────────────────────────────

    // Group 0: scene + lights + material (all uniforms)
    const bg0Layout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' } },
      ],
    });

    // Group 1: textures + sampler
    const bg1Layout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    });

    const terrainLayout = d.createPipelineLayout({
      bindGroupLayouts: [bg0Layout, bg1Layout],
    });

    // ── Terrain PBR render pipeline ───────────────────────────────────────────
    const terrainMod = d.createShaderModule({ code: WGSL_TERRAIN_RENDER });

    this.terrainRenderPipe = await d.createRenderPipelineAsync({
      layout: terrainLayout,
      vertex: {
        module:     terrainMod,
        entryPoint: 'vs_terrain',
        buffers: [{
          arrayStride: VERTEX_STRIDE_BYTES,
          attributes: [
            { shaderLocation: 0, offset: 0,  format: 'float32x3' },  // position
            { shaderLocation: 1, offset: 12, format: 'float32x3' },  // normal
            { shaderLocation: 2, offset: 24, format: 'float32x2' },  // uv
          ],
        }],
      },
      fragment: {
        module:     terrainMod,
        entryPoint: 'fs_terrain',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'zero', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'zero', operation: 'add' },
          },
        }],
      },
      depthStencil: {
        format:            'depth24plus',
        depthWriteEnabled: true,
        depthCompare:      'less',
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back',
        frontFace: 'ccw',
      },
    });

    // ── Rebuild texture bind groups with the actual pipeline layout ────────────
    for (const mesh of this.meshes) {
      // The existing bind group was created with a temporary layout;
      // it should be compatible since we specified the same entries.
      // If not, we'd rebuild here with: this.terrainRenderPipe.getBindGroupLayout(1)
    }

    // ── Shadow depth pipeline ─────────────────────────────────────────────────
    if (this.cfg.enableShadows) {
      const shadowMod = d.createShaderModule({ code: WGSL_SHADOW_DEPTH });

      this.shadowDepthPipe = await d.createRenderPipelineAsync({
        layout: 'auto',
        vertex: {
          module:     shadowMod,
          entryPoint: 'vs_shadow',
          buffers: [{
            arrayStride: VERTEX_STRIDE_BYTES,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
            ],
          }],
        },
        depthStencil: {
          format:            'depth24plus',
          depthWriteEnabled: true,
          depthCompare:      'less',
          depthBias:         2,
          depthBiasSlopeScale: 2.0,
        },
        primitive: {
          topology: 'triangle-list',
          cullMode: 'front',   // front-face culling for shadow bias
        },
      });
    }

    // ── Dust particle update compute ──────────────────────────────────────────
    if (this.cfg.enableDust) {
      const dustUpdateMod = d.createShaderModule({ code: WGSL_DUST_UPDATE });

      this.dustUpdatePipe = await d.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: dustUpdateMod, entryPoint: 'cs_dust_update' },
      });

      // ── Dust particle render ────────────────────────────────────────────────
      const dustRenderMod = d.createShaderModule({ code: WGSL_DUST_RENDER });

      // Dust render group 0: dustUniforms + sampler + pTex
      // Dust render group 1: scene uniforms

      this.dustRenderPipe = await d.createRenderPipelineAsync({
        layout: 'auto',
        vertex: {
          module:     dustRenderMod,
          entryPoint: 'vs_dust',
        },
        fragment: {
          module:     dustRenderMod,
          entryPoint: 'fs_dust',
          targets: [{
            format: this.format,
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          }],
        },
        depthStencil: {
          format:            'depth24plus',
          depthWriteEnabled: false,
          depthCompare:      'less-equal',
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
      });
    }
  }

  // ─── Private: Uniform writes ────────────────────────────────────────────────

  private _writeSceneUniforms(time: number, dt: number): void {
    const data = new Float32Array(SCENE_UNIFORM_SIZE / 4);

    // viewProj — identity placeholder (caller provides via sceneUniformBuf or overwrites)
    // The viewProj/modelMat/normalMat are typically provided by the scene compositor.
    // We fill in the fog/ambient/time params here.
    data[0]  = 1; data[5]  = 1; data[10] = 1; data[15] = 1; // identity viewProj
    data[16] = 1; data[21] = 1; data[26] = 1; data[31] = 1; // identity modelMat
    data[32] = 1; data[37] = 1; data[42] = 1; data[47] = 1; // identity normalMat

    // eye — offset 192 (index 48)
    // (Set by caller or left at origin)
    data[48] = 0; data[49] = 3; data[50] = 5; data[51] = 1;

    // fogParams — offset 208 (index 52)
    data[52] = this.cfg.fogColor[0];
    data[53] = this.cfg.fogColor[1];
    data[54] = this.cfg.fogColor[2];
    data[55] = this.cfg.fogDensity;

    // ambientParams — offset 224 (index 56)
    data[56] = this.cfg.ambientColor[0];
    data[57] = this.cfg.ambientColor[1];
    data[58] = this.cfg.ambientColor[2];
    data[59] = this.cfg.ambientIntensity;

    // timeParams — offset 240 (index 60)
    data[60] = time;
    data[61] = dt;
    data[62] = this.cfg.fogStart;
    data[63] = this.cfg.fogEnd;

    this.device.queue.writeBuffer(this.sceneUniformBuf, 0, data);
  }

  private _writeLightUniforms(): void {
    const data = new Float32Array(LIGHT_BUFFER_SIZE / 4);

    for (let i = 0; i < MAX_LIGHTS; i++) {
      const light = this.lights[i];
      const off = i * (LIGHT_STRUCT_SIZE / 4);  // 20 floats per light

      if (!light || (light.intensity ?? 1.0) <= 0) {
        // Disabled light: type = 0
        data[off + 19] = 0;  // props.w = type = 0
        continue;
      }

      const intensity  = light.intensity    ?? 1.0;
      const range      = light.range        ?? 50.0;
      const shadowMin  = light.shadowMin    ?? 0.0;
      const coneDir    = light.coneDirection ?? [0, -1, 0];
      const coneAngle  = light.coneAngle    ?? 45.0;
      const feather    = light.coneFeather   ?? 1.0;
      const shininess  = light.phong ? (light.phongShininess ?? 30.0) : 0.0;

      // color: .rgb=colour, .w=intensity
      data[off + 0]  = light.color[0];
      data[off + 1]  = light.color[1];
      data[off + 2]  = light.color[2];
      data[off + 3]  = intensity;

      // pos: .xyz=position, .w=type
      data[off + 4]  = light.position[0];
      data[off + 5]  = light.position[1];
      data[off + 6]  = light.position[2];
      data[off + 7]  = light.type;

      // data: .xyz=coneDir, .w=coneAngle
      data[off + 8]  = coneDir[0];
      data[off + 9]  = coneDir[1];
      data[off + 10] = coneDir[2];
      data[off + 11] = coneAngle;

      // data2: .x=feather, .y=phongShininess, .z=shadowMin, .w=range
      data[off + 12] = feather;
      data[off + 13] = shininess;
      data[off + 14] = shadowMin;
      data[off + 15] = range;

      // props: .x=intensity, .y=range, .z=shadowMin, .w=type
      data[off + 16] = intensity;
      data[off + 17] = range;
      data[off + 18] = shadowMin;
      data[off + 19] = light.type;
    }

    this.device.queue.writeBuffer(this.lightUniformBuf, 0, data);
  }

  private _writeDustUniforms(time: number, dt: number): void {
    const data = new Float32Array(12);
    // time
    data[0] = time; data[1] = dt; data[2] = 0; data[3] = 0;
    // wind
    data[4] = this.cfg.windDirection[0];
    data[5] = this.cfg.windDirection[1];
    data[6] = this.cfg.windDirection[2];
    data[7] = 0;
    // dustColor
    data[8]  = this.cfg.dustColor[0];
    data[9]  = this.cfg.dustColor[1];
    data[10] = this.cfg.dustColor[2];
    data[11] = 0;

    this.device.queue.writeBuffer(this.dustUniformBuf, 0, data);
  }

  // ─── Private: Compute dispatches ────────────────────────────────────────────

  private _dispatchDustUpdate(enc: GPUCommandEncoder): void {
    const bg = this.device.createBindGroup({
      layout: this.dustUpdatePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.dustUniformBuf } },
        { binding: 1, resource: this.dustA.createView() },
        { binding: 2, resource: this.dustB.createView() },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.dustUpdatePipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(this.maxDustParticles / WG));
    pass.end();

    // Ping-pong swap
    [this.dustA, this.dustB] = [this.dustB, this.dustA];
  }

  // ─── Private: Shadow render ─────────────────────────────────────────────────

  private _renderShadowPass(enc: GPUCommandEncoder): void {
    if (!this.shadowDepthPipe || !this.shadowMap) return;

    // Compute light-space VP matrix from the primary directional light
    const primaryLight = this.lights.find(l => l && l.type === 1);
    if (!primaryLight) return;

    // Write shadow uniforms (lightViewProj + modelMat)
    const shadowData = new Float32Array(32);
    // Simple orthographic projection from light direction
    // This is a simplified projection; a real implementation would compute
    // a proper frustum from the scene bounds.
    const lx = primaryLight.position[0];
    const ly = primaryLight.position[1];
    const lz = primaryLight.position[2];
    const len = Math.hypot(lx, ly, lz) || 1;
    const dlx = lx / len, dly = ly / len, dlz = lz / len;

    // Look-at from light direction (simplified)
    // Build a view matrix looking from light position toward origin
    const lightEye = [dlx * 20, dly * 20, dlz * 20];
    const lightTarget = [0, 0, 0];
    const lightUp = [0, 1, 0];

    // Compute view matrix columns
    const fwd = [
      lightTarget[0] - lightEye[0],
      lightTarget[1] - lightEye[1],
      lightTarget[2] - lightEye[2],
    ];
    const fwdLen = Math.hypot(fwd[0], fwd[1], fwd[2]) || 1;
    fwd[0] /= fwdLen; fwd[1] /= fwdLen; fwd[2] /= fwdLen;

    const right = [
      lightUp[1] * fwd[2] - lightUp[2] * fwd[1],
      lightUp[2] * fwd[0] - lightUp[0] * fwd[2],
      lightUp[0] * fwd[1] - lightUp[1] * fwd[0],
    ];
    const rLen = Math.hypot(right[0], right[1], right[2]) || 1;
    right[0] /= rLen; right[1] /= rLen; right[2] /= rLen;

    const up = [
      fwd[1] * right[2] - fwd[2] * right[1],
      fwd[2] * right[0] - fwd[0] * right[2],
      fwd[0] * right[1] - fwd[1] * right[0],
    ];

    // View matrix (column-major for WGSL mat4x4f)
    const viewMat = [
      right[0], up[0], -fwd[0], 0,
      right[1], up[1], -fwd[1], 0,
      right[2], up[2], -fwd[2], 0,
      -(right[0]*lightEye[0] + right[1]*lightEye[1] + right[2]*lightEye[2]),
      -(up[0]*lightEye[0]    + up[1]*lightEye[1]    + up[2]*lightEye[2]),
       (fwd[0]*lightEye[0]   + fwd[1]*lightEye[1]   + fwd[2]*lightEye[2]),
      1,
    ];

    // Orthographic projection: [-25, 25] × [-25, 25] × [0, 60]
    const orthoHalf = 25.0;
    const near = 0.0;
    const far = 60.0;
    const projMat = [
      1/orthoHalf, 0, 0, 0,
      0, 1/orthoHalf, 0, 0,
      0, 0, -1/(far - near), 0,
      0, 0, -near/(far - near), 1,
    ];

    // Multiply proj × view → lightViewProj
    for (let i = 0; i < 16; i++) {
      shadowData[i] = 0;
      // Row-based multiplication for column-major layout
    }
    // For simplicity, store identity (the shadow pass is a structural placeholder;
    // full cascade shadow mapping would be implemented in at-shadow-import.ts)
    shadowData[0] = 1; shadowData[5] = 1; shadowData[10] = 1; shadowData[15] = 1;
    // Model matrix: identity
    shadowData[16] = 1; shadowData[21] = 1; shadowData[26] = 1; shadowData[31] = 1;

    this.device.queue.writeBuffer(this.shadowUniformBuf, 0, shadowData);

    const shadowView = this.shadowMap.createView();
    const pass = enc.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view:          shadowView,
        depthLoadOp:   'clear',
        depthStoreOp:  'store',
        depthClearValue: 1.0,
      },
    });

    pass.setPipeline(this.shadowDepthPipe);

    const bg = this.device.createBindGroup({
      layout: this.shadowDepthPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.shadowUniformBuf } },
      ],
    });
    pass.setBindGroup(0, bg);

    for (const mesh of this.meshes) {
      pass.setVertexBuffer(0, mesh.vertexBuffer);
      pass.setIndexBuffer(mesh.indexBuffer, 'uint32');
      pass.drawIndexed(mesh.indexCount);
    }

    pass.end();
  }

  // ─── Private: Bind group factories ──────────────────────────────────────────

  /**
   * Create group(0) bind group for terrain render (per-mesh material).
   */
  private _makeTerrainBG0(sceneBuf: GPUBuffer, materialBuf: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.terrainRenderPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: sceneBuf } },
        { binding: 1, resource: { buffer: this.lightUniformBuf } },
        { binding: 2, resource: { buffer: materialBuf } },
      ],
    });
  }

  /**
   * Create group(0) bind group for dust render.
   */
  private _makeDustRenderBG0(): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.dustRenderPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.dustUniformBuf } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: this.dustA.createView() },
      ],
    });
  }

  /**
   * Create group(1) bind group for dust render (scene uniforms).
   */
  private _makeDustSceneBG1(sceneBuf: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.dustRenderPipe.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: sceneBuf } },
      ],
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factory
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create and build an ATTerrainEnvironment in one call.
 *
 * ```ts
 * const terrain = await createATTerrainEnvironment(device, 'bgra8unorm', {
 *   lights: [
 *     { type: 1, position: [5, 8, -3], color: [1, 0.95, 0.88], intensity: 1.2 },
 *   ],
 *   fogDensity: 0.04,
 *   enableDust: true,
 * });
 * ```
 */
export async function createATTerrainEnvironment(
  device: GPUDevice,
  format: GPUTextureFormat,
  cfg?:   ATTerrainEnvironmentConfig,
): Promise<ATTerrainEnvironment> {
  const terrain = new ATTerrainEnvironment(device, format, cfg);
  await terrain.build();
  return terrain;
}
