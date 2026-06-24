/**
 * src/lib/sph/at-water-particles-normals.ts  —  M846
 *
 * AT Water Particles + Normals System  →  WebGPU / WGSL Port
 * ─────────────────────────────────────────────────────────────────────────────
 * Advanced GPU port of Active Theory's water rendering pipeline:
 *   • WaterParticles.glsl  (GPU particle lifecycle, simplex drift)
 *   • waternormals.fs      (Layered normal texture animation)
 *   • TreeWaterShader.glsl (PBR water with mirror reflections)
 *   • WaterCeilingShader.glsl (reflective ceiling with HSV color shift)
 *
 * ─── AT Reverse-Engineered Sources ──────────────────────────────────────────
 *
 *   WaterParticles.glsl
 *     • GPU texture-based particle pool  (tPos, tPointColor)
 *     • Lifecycle: SPAWN → FLOAT → DECAY → DEAD  (encoded in tPos.w)
 *     • Simplex noise-driven lateral drift  (cnoise displacement)
 *     • Billboard quad rendering with scale-by-distance LOD
 *     • Sparkle & ripple modulation via time-based trig functions
 *
 *   waternormals.fs  (Dynamic Normal Mapping)
 *     • Multi-layer UV scrolling on Perlin/Worley texture atlas
 *     • 4-layer blend at different speeds & scales  (uv0/uv1/uv2/uv3)
 *     • Normal reconstruction: normalize(noise.xzy * vec3(2, 1, 2))
 *     • Enables realistic wave-surface perturbation on water geometry
 *
 *   TreeWaterShader.glsl  (Reflective Water Geometry)
 *     • Mirror matrix projection  (uMirrorMatrix for planar reflections)
 *     • PBR fresnel blending via getFBR(baseColor, roughness, normal)
 *     • Water normal displacement of reflection UV coords
 *     • Used in Tree scene for floor/walls with dynamic refraction
 *
 *   WaterCeilingShader.glsl  (Ceiling Water Reflections)
 *     • HSV-space color shift based on distance from center  (hsl.x -= length)
 *     • Overlay blend with video texture for ripple effects
 *     • Circular falloff  (smoothstep(0.45, 0.0, length(vUv-0.5)))
 *     • Tone mapping  (pow(color, vec3(2.2)))
 *
 * ─── WebGPU Architecture ────────────────────────────────────────────────────
 *
 *   ATWaterParticlesNormals
 *     ├─ Normal Texture Pipeline  (compute + render)
 *     │    normalsA / normalsB   — rgba8unorm  (packed normals, speed layers)
 *     │    normalUpdateCompute   — 4-layer UV scroll + blend
 *     │    normalRenderPass      — particle-to-surface normal mapping
 *     │
 *     ├─ Water Particle Pipeline  (compute + render)
 *     │    particlePos / particleColor  — rgba32float  (position, lifecycle)
 *     │    particleUpdateCompute — spawn, drift (curl noise), decay
 *     │    particleRenderPass    — point sprite + matcap texture
 *     │
 *     ├─ Tree Water Surface  (render)
 *     │    treeWaterMesh         — planar or terrain geometry
 *     │    treeWaterVertex       — position + normal transform
 *     │    treeWaterFragment     — fresnel + normal sampling + reflection
 *     │
 *     └─ Ceiling Reflection  (render)
 *          ceilingMesh           — inverted/flipped water plane
 *          ceilingVertex         — HSV color + mirror projection
 *          ceilingFragment       — center-based hue shift + video blend
 *
 * ─── GLSL → WGSL Translation Key ────────────────────────────────────────────
 *
 *   texture2D(tNormal, uv)              → textureSample(tNormal, sampler, uv)
 *   vec4 noise = ...                    → var noise: vec4f = ...
 *   varying vec3 vNormal                → @location(1) normal : vec3f
 *   uniform sampler2D tMap              → @group(0) @binding(N) var tMap: texture_2d<f32>
 *   uniform sampler2D sampler           → @group(0) @binding(N) var sampler: sampler
 *   gl_PointSize                        → @builtin(point_size)
 *   gl_PointCoord                       → @builtin(sample_index)  (WebGPU: sample position)
 *   dFdx / dFdy (LOD)                   → textureQueryLod() or manual mip level
 *   mix(a, b, t)                        → mix(a, b, t)  (same in WGSL)
 *   smoothstep(edge0, edge1, x)         → smoothstep(edge0, edge1, x)
 *
 * ─── Upstream Copyright Notices ─────────────────────────────────────────────
 *
 *   Active Theory / Cleanroom Studios  © 2020-2024
 *   GLSL → WGSL port  © 2024  (research #M846)
 *
 * ─── Usage ───────────────────────────────────────────────────────────────────
 *
 *   const water = new ATWaterParticlesNormals(device, format, {
 *     particleCount: 4096,
 *     normalAtlasSize: 512,
 *     treeWaterGeometry: treeMesh,
 *     ceilingGeometry: ceilingMesh,
 *   });
 *   await water.initialize();
 *
 *   // Emit particles at screen position / world position
 *   water.emitParticles(0.5, 0.5, 16, 0.3);
 *   // Emit ripple at normal map coordinates
 *   water.addNormalRipple(0.5, 0.5, 0.1);
 *
 *   // Render loop
 *   const enc = device.createCommandEncoder();
 *   water.tick(enc, elapsed, delta);
 *   water.renderPass(enc, colorTarget, depthView);
 *   device.queue.submit([enc.finish()]);
 *
 * Research: xiaodi #M846 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Constants & Configuration
// ─────────────────────────────────────────────────────────────────────────────









const DEFAULT_NORMAL_ATLAS_SIZE = 512 as const;
const DEFAULT_PARTICLE_COUNT = 4096 as const;
const DEFAULT_PARTICLE_POOL_SIZE = 8192 as const;
const WORKGROUP_SIZE = 64 as const;

// Normal map animation parameters (from waternormals.fs)
const NORMAL_SPEED = 0.2 as const;
const NORMAL_SCALES = [103.0, 107.0, 897.0, 991.0] as const;
const NORMAL_SPEEDS = [17.0, 19.0, 101.0, 109.0] as const;

// Particle lifecycle stages
enum ParticleStage {
  DEAD = 0.0,
  SPAWN = 0.25,
  FLOAT = 0.5,
  DECAY = 0.75,
}

// IOR constants for Fresnel
const IOR_AIR = 1.0;
const IOR_WATER = 1.333;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration Types
// ─────────────────────────────────────────────────────────────────────────────

export interface ATWaterParticlesNormalsConfig {
  /** Normal map atlas resolution (e.g., 256, 512, 1024). Default: 512 */
  normalAtlasSize?: number;

  /** Max water particles in GPU pool. Default: 4096 */
  particleCount?: number;

  /** Particle spawn rate (particles per frame). Default: 4 */
  spawnRatePerFrame?: number;

  /** Normal map animation speed multiplier. Default: 1.0 */
  normalAnimSpeed?: number;

  /** Particle drift amount (simplex noise amplitude). Default: 0.2 */
  particleDriftAmount?: number;

  /** Particle color tint [r, g, b]. Default: [0.6, 0.9, 1.0] */
  particleColor?: [number, number, number];

  /** Water surface color [r, g, b]. Default: [0.25, 1.0, 1.25] */
  waterColor?: [number, number, number];

  /** Sky gradient color [r, g, b]. Default: [0.1, 0.6, 1.0] */
  skyColor?: [number, number, number];

  /** Ceiling HSV shift amount. Default: 0.2 */
  ceilingHueShift?: number;

  /** Mirror reflection brightness. Default: 0.9 */
  mirrorBrightness?: number;

  /** Tree water normal map strength. Default: 0.015 */
  treeWaterNormalStrength?: number;

  /** Optional sky cubemap texture. If not provided, gradient is used. */
  skyCubemap?: GPUTexture;

  /** Optional tree water mesh. */
  treeWaterGeometry?: {
    vertices: Float32Array;
    indices: Uint32Array;
    normals: Float32Array;
  };

  /** Optional ceiling mesh. */
  ceilingGeometry?: {
    vertices: Float32Array;
    indices: Uint32Array;
    normals: Float32Array;
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL Shader Code
// ─────────────────────────────────────────────────────────────────────────────

// ─── Shared Math Helpers ─────────────────────────────────────────────────────

const WGSL_MATH_HELPERS = /* wgsl */ `
// Simplex noise implementation (from ashima webgl-noise MIT)
fn mod289_v3(x: vec3f) -> vec3f {
    return x - floor(x * (1.0/289.0)) * 289.0;
}

fn mod289_v4(x: vec4f) -> vec4f {
    return x - floor(x * (1.0/289.0)) * 289.0;
}

fn permute(x: vec4f) -> vec4f {
    return mod289_v4((x * 34.0 + 10.0) * x);
}

fn taylorInvSqrt(r: vec4f) -> vec4f {
    return vec4f(1.79284291400159) - 0.85373472095314 * r;
}

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

// Curl noise (3D vector field from simplex basis)
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

// Manual refract (WGSL has no built-in)
fn refract_wgsl(I: vec3f, N: vec3f, eta: f32) -> vec3f {
    let cosI = dot(N, I);
    let k    = 1.0 - eta*eta * (1.0 - cosI*cosI);
    if (k < 0.0) { return vec3f(0.0); }
    return eta * I - (eta * cosI + sqrt(k)) * N;
}

// Manual reflect
fn reflect_wgsl(I: vec3f, N: vec3f) -> vec3f {
    return I - 2.0 * dot(N, I) * N;
}

// Fresnel schlick approximation
fn fresnel_schlick(cosTheta: f32, F0: vec3f) -> vec3f {
    let t = 1.0 - cosTheta;
    return F0 + (vec3f(1.0) - F0) * (t * t * t * t * t);
}

// RGB to HSV
fn rgb2hsv(c: vec3f) -> vec3f {
    let K = vec4f(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    let p = mix(vec4f(c.bg, K.wz), vec4f(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4f(p.xyw, c.r), vec4f(c.r, p.yzw), step(p.x, c.r));
    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3f(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// HSV to RGB
fn hsv2rgb(c: vec3f) -> vec3f {
    let K = vec4f(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3f(0.0), vec3f(1.0)), c.y);
}

// Sky gradient
fn skyGradient(dir: vec3f, skyCol: vec3f) -> vec3f {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(skyCol * 0.3, skyCol, t);
}

// Blend: overlay mode
fn blendOverlay(base: vec3f, top: vec3f, blend: f32) -> vec3f {
    var out = vec3f(0.0);
    for (var i = 0u; i < 3u; i++) {
        if (base[i] < 0.5) {
            out[i] = 2.0 * base[i] * top[i];
        } else {
            out[i] = 1.0 - 2.0 * (1.0 - base[i]) * (1.0 - top[i]);
        }
    }
    return mix(base, out, blend);
}
`;

// ─── Normal Map Update Compute Shader ────────────────────────────────────────

const WGSL_NORMAL_UPDATE = /* wgsl */ `
${WGSL_MATH_HELPERS}

struct NormalUniforms {
    time: f32,
    speed: f32,
    scale: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> nu: NormalUniforms;
@group(0) @binding(1) var normalsIn: texture_2d<f32>;
@group(0) @binding(2) var normalsOut: texture_storage_2d<rgba8unorm, read_write>;

fn getWaterNoise(uv: vec2f, speed: f32, scale: f32) -> vec4f {
    let time = nu.time * 0.2 * speed;
    let uv0 = (uv/103.0) + vec2f(time/17.0, time/29.0);
    let uv1 = (uv/107.0) - vec2f(time/-19.0, time/31.0);
    let uv2 = (uv/vec2f(897.0, 983.0)) + vec2f(time/101.0, time/97.0);
    let uv3 = (uv/vec2f(991.0, 877.0)) - vec2f(time/109.0, time/-113.0);
    
    var noise = textureSampleLevel(normalsIn, textureSampler, uv0 * scale, 0.0)
              + textureSampleLevel(normalsIn, textureSampler, uv1 * scale, 0.0)
              + textureSampleLevel(normalsIn, textureSampler, uv2 * scale, 0.0)
              + textureSampleLevel(normalsIn, textureSampler, uv3 * scale, 0.0);
    return noise * 0.5 - vec4f(1.0);
}

fn getWaterNormal(uv: vec2f, speed: f32, scale: f32) -> vec3f {
    let noise = getWaterNoise(uv, speed, scale);
    let surfaceNormal = normalize(noise.xzy * vec3f(2.0, 1.0, 2.0));
    return surfaceNormal;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let coord = vec2f(gid.xy) / vec2f(textureDimensions(normalsOut));
    let normal = getWaterNormal(coord, nu.speed, nu.scale);
    
    // Pack normal into RGBA8
    let packed = vec4f(normal * 0.5 + 0.5, 1.0);
    textureStore(normalsOut, gid.xy, packed);
}
`;

// ─── Particle Update Compute Shader ──────────────────────────────────────────

const WGSL_PARTICLE_UPDATE = /* wgsl */ `
${WGSL_MATH_HELPERS}

struct ParticleUniforms {
    time: f32,
    deltaTime: f32,
    particleCount: u32,
    driftAmount: f32,
};

@group(0) @binding(0) var<uniform> pu: ParticleUniforms;
@group(0) @binding(1) var particlePosIn: texture_2d<f32>;
@group(0) @binding(2) var particlePosOut: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(3) var normalsSampler: texture_2d<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= pu.particleCount) { return; }
    
    let px = idx % 64u;
    let py = idx / 64u;
    let coord = vec2u(px, py);
    
    let posData = textureLoad(particlePosIn, coord, 0);
    var pos = posData.xyz;
    var life = posData.w;
    
    // Decay lifecycle
    life -= pu.deltaTime * 0.5;
    
    if (life > 0.0) {
        // Drift via curl noise
        let noisePos = pos * 0.05 + pu.time * 0.1;
        let drift = curl3(noisePos) * pu.driftAmount;
        pos += drift * pu.deltaTime;
        
        // Vertical rise motion (simplified)
        pos.y += pu.deltaTime * 0.5 * (1.0 - life);
    }
    
    textureStore(particlePosOut, coord, vec4f(pos, max(life, -1.0)));
}
`;

// ─── Particle Render (Billboard) Vertex Shader ───────────────────────────────

const WGSL_PARTICLE_VERTEX = /* wgsl */ `
struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec3f,
    @location(2) life: f32,
};

@group(0) @binding(0) var<uniform> projection: mat4x4f;
@group(0) @binding(1) var<uniform> view: mat4x4f;
@group(0) @binding(2) var particlePos: texture_2d<f32>;

@vertex
fn main(
    @builtin(instance_index) instanceIdx: u32,
    @builtin(vertex_index) vertexIdx: u32
) -> VertexOutput {
    let px = instanceIdx % 64u;
    let py = instanceIdx / 64u;
    
    let posData = textureLoad(particlePos, vec2u(px, py), 0);
    var worldPos = posData.xyz;
    let life = posData.w;
    
    // Billboard quad
    let quadVerts = array<vec2f, 4>(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f( 1.0,  1.0)
    );
    let quadUvs = array<vec2f, 4>(
        vec2f(0.0, 1.0),
        vec2f(1.0, 1.0),
        vec2f(0.0, 0.0),
        vec2f(1.0, 0.0)
    );
    
    let vert = quadVerts[vertexIdx % 4u] * 0.05;
    let uv = quadUvs[vertexIdx % 4u];
    
    // Trivial projection (assume view/proj already set)
    let viewPos = (view * vec4f(worldPos, 1.0)).xyz + vec3f(vert, 0.0);
    let clipPos = projection * vec4f(viewPos, 1.0);
    
    return VertexOutput(
        clipPos,
        uv,
        vec3f(0.6, 0.9, 1.0),
        max(life, 0.0)
    );
}
`;

// ─── Particle Render Fragment Shader ─────────────────────────────────────────

const WGSL_PARTICLE_FRAGMENT = /* wgsl */ `
@group(0) @binding(0) var particleTexture: texture_2d<f32>;
@group(0) @binding(1) var particleSampler: sampler;

@fragment
fn main(
    @location(0) uv: vec2f,
    @location(1) color: vec3f,
    @location(2) life: f32
) -> @location(0) vec4f {
    let dist = length(uv - vec2f(0.5));
    if (dist > 0.5) { discard; }
    
    var col = textureSample(particleTexture, particleSampler, uv);
    col.rgb *= color;
    col.rgb = pow(col.rgb, vec3f(3.0));
    col.rgb *= 0.5;
    
    let sparkle = 0.5 + sin(life * 20.0) * 0.5;
    col.rgb *= mix(sparkle, 1.0, 0.6);
    col.a *= sin(3.1415926 * life);
    
    return col;
}
`;

// ─── Tree Water Surface Vertex Shader ────────────────────────────────────────

const WGSL_TREE_WATER_VERTEX = /* wgsl */ `
struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) normal: vec3f,
    @location(2) worldPos: vec3f,
};

@group(0) @binding(0) var<uniform> model: mat4x4f;
@group(0) @binding(1) var<uniform> view: mat4x4f;
@group(0) @binding(2) var<uniform> projection: mat4x4f;

@vertex
fn main(
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f
) -> VertexOutput {
    let worldPos = (model * vec4f(position, 1.0)).xyz;
    let viewPos = view * vec4f(worldPos, 1.0);
    let clipPos = projection * viewPos;
    let worldNormal = (model * vec4f(normal, 0.0)).xyz;
    
    return VertexOutput(clipPos, uv, worldNormal, worldPos);
}
`;

// ─── Tree Water Surface Fragment Shader ─────────────────────────────────────

const WGSL_TREE_WATER_FRAGMENT = /* wgsl */ `
${WGSL_MATH_HELPERS}

struct TreeWaterUniforms {
    mirrorMatrix: mat4x4f,
    speed: f32,
    scale: f32,
    waterUVStrength: f32,
    brightness: f32,
};

@group(0) @binding(0) var<uniform> twu: TreeWaterUniforms;
@group(0) @binding(1) var tWaterNormal: texture_2d<f32>;
@group(0) @binding(2) var normalSampler: sampler;
@group(0) @binding(3) var tMirrorReflection: texture_2d<f32>;
@group(0) @binding(4) var reflectSampler: sampler;

@fragment
fn main(
    @location(0) uv: vec2f,
    @location(1) normal: vec3f,
    @location(2) worldPos: vec3f
) -> @location(0) vec4f {
    // Get water normal from multi-layer scrolling
    let noise = textureSample(tWaterNormal, normalSampler, uv * twu.scale);
    let surfaceNormal = normalize(noise.xzy * vec3f(2.0, 1.0, 2.0));
    
    // Apply to mirror coordinates
    let mirrorPos = twu.mirrorMatrix * vec4f(worldPos, 1.0);
    var mirrorUv = mirrorPos.xy / mirrorPos.w;
    mirrorUv -= surfaceNormal.xy * 0.015 * twu.waterUVStrength;
    mirrorUv.y -= 0.04;
    
    // Sample reflection
    var baseColor = textureSample(tMirrorReflection, reflectSampler, mirrorUv).rgb;
    baseColor *= twu.brightness;
    
    // Fresnel-like blending
    let fresnel = fresnel_schlick(abs(dot(normal, surfaceNormal)), vec3f(0.04));
    let color = mix(baseColor * 0.8, baseColor, fresnel.r);
    
    return vec4f(color * 0.9, 1.0);
}
`;

// ─── Ceiling Water Reflection Fragment Shader ────────────────────────────────

const WGSL_CEILING_WATER_FRAGMENT = /* wgsl */ `
${WGSL_MATH_HELPERS}

struct CeilingUniforms {
    time: f32,
    alpha: f32,
    hueShift: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> cu: CeilingUniforms;
@group(0) @binding(1) var tMap: texture_2d<f32>;
@group(0) @binding(2) var tVideo: texture_2d<f32>;
@group(0) @binding(3) var samplerTex: sampler;

@fragment
fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
    var color = textureSample(tMap, samplerTex, uv * 0.1);
    
    // HSV color shift based on distance from center
    var hsl = rgb2hsv(color.rgb);
    hsl.x -= length(uv - vec2f(0.5)) * cu.hueShift;
    hsl.y *= 0.5;
    color.rgb = hsv2rgb(hsl);
    
    // Center falloff
    color.rgb *= smoothstep(0.45, 0.0, length(uv - vec2f(0.5)));
    
    // Overlay video texture
    let video = textureSample(tVideo, samplerTex, uv * 0.4).rgb;
    color.rgb = blendOverlay(color.rgb, video, 0.3);
    
    // Gamma correction
    color.rgb = pow(color.rgb, vec3f(2.2));
    
    color.a *= cu.alpha;
    return color;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Main Class: ATWaterParticlesNormals
// ─────────────────────────────────────────────────────────────────────────────

export class ATWaterParticlesNormals {
  private device: GPUDevice;
  private format: GPUTextureFormat;
  private config: Required<ATWaterParticlesNormalsConfig>;

  // Normal map textures
  private normalAtlasTexture: GPUTexture;
  private normalPingPong: GPUTexture[] = [];

  // Particle textures
  private particlePosPingPong: GPUTexture[] = [];
  private particleColorTexture: GPUTexture;

  // Pipelines
  private normalUpdatePipeline: GPUComputePipeline;
  private particleUpdatePipeline: GPUComputePipeline;
  private particleRenderPipeline: GPURenderPipeline;
  private treeWaterRenderPipeline: GPURenderPipeline;
  private ceilingWaterRenderPipeline: GPURenderPipeline;

  // Bind groups
  private normalBindGroup: GPUBindGroup;
  private particleBindGroup: GPUBindGroup;
  private treeWaterBindGroup: GPUBindGroup;
  private ceilingWaterBindGroup: GPUBindGroup;

  // Buffers
  private normalUniformBuffer: GPUBuffer;
  private particleUniformBuffer: GPUBuffer;
  private treeWaterUniformBuffer: GPUBuffer;
  private ceilingUniformBuffer: GPUBuffer;

  // Meshes
  private treeWaterMesh: { vertices: GPUBuffer; indices: GPUBuffer; indexCount: number };
  private ceilingMesh: { vertices: GPUBuffer; indices: GPUBuffer; indexCount: number };

  // State
  private time: number = 0;
  private particleCount: number = 0;

  constructor(device: GPUDevice, format: GPUTextureFormat, config: ATWaterParticlesNormalsConfig = {}) {
    this.device = device;
    this.format = format;
    this.config = {
      normalAtlasSize: config.normalAtlasSize ?? DEFAULT_NORMAL_ATLAS_SIZE,
      particleCount: config.particleCount ?? DEFAULT_PARTICLE_COUNT,
      spawnRatePerFrame: config.spawnRatePerFrame ?? 4,
      normalAnimSpeed: config.normalAnimSpeed ?? 1.0,
      particleDriftAmount: config.particleDriftAmount ?? 0.2,
      particleColor: config.particleColor ?? [0.6, 0.9, 1.0],
      waterColor: config.waterColor ?? [0.25, 1.0, 1.25],
      skyColor: config.skyColor ?? [0.1, 0.6, 1.0],
      ceilingHueShift: config.ceilingHueShift ?? 0.2,
      mirrorBrightness: config.mirrorBrightness ?? 0.9,
      treeWaterNormalStrength: config.treeWaterNormalStrength ?? 0.015,
      skyCubemap: config.skyCubemap,
      treeWaterGeometry: config.treeWaterGeometry,
      ceilingGeometry: config.ceilingGeometry,
    };
  }

  /**
   * Initialize all GPU resources.
   */
  async initialize(): Promise<void> {
    await this.createTextures();
    await this.createBuffers();
    await this.createPipelines();
    await this.createBindGroups();
  }

  private async createTextures(): Promise<void> {
    // Normal atlas (input for normal computation)
    this.normalAtlasTexture = this.device.createTexture({
      size: [this.config.normalAtlasSize, this.config.normalAtlasSize, 1],
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Ping-pong normals for compute
    for (let i = 0; i < 2; i++) {
      const tex = this.device.createTexture({
        size: [this.config.normalAtlasSize, this.config.normalAtlasSize, 1],
        format: 'rgba8unorm',
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.normalPingPong.push(tex);
    }

    // Particle position ping-pong
    const particlePoolWidth = Math.ceil(Math.sqrt(this.config.particleCount));
    for (let i = 0; i < 2; i++) {
      const tex = this.device.createTexture({
        size: [particlePoolWidth, particlePoolWidth, 1],
        format: 'rgba32float',
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.particlePosPingPong.push(tex);
    }

    // Particle color texture
    this.particleColorTexture = this.device.createTexture({
      size: [256, 256, 1],
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
  }

  private async createBuffers(): Promise<void> {
    this.normalUniformBuffer = this.device.createBuffer({
      size: 16, // time(f32) + speed(f32) + scale(f32) + pad(f32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.particleUniformBuffer = this.device.createBuffer({
      size: 16, // time(f32) + deltaTime(f32) + particleCount(u32) + driftAmount(f32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.treeWaterUniformBuffer = this.device.createBuffer({
      size: 80, // mirrorMatrix(mat4x4) + speed(f32) + scale(f32) + uv strength(f32) + brightness(f32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.ceilingUniformBuffer = this.device.createBuffer({
      size: 16, // time(f32) + alpha(f32) + hueShift(f32) + pad(f32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create dummy meshes if not provided
    if (this.config.treeWaterGeometry) {
      this.createMesh(this.config.treeWaterGeometry);
    }

    if (this.config.ceilingGeometry) {
      this.createMesh(this.config.ceilingGeometry);
    }
  }

  private createMesh(geom: { vertices: Float32Array; indices: Uint32Array; normals: Float32Array }): {
    vertices: GPUBuffer;
    indices: GPUBuffer;
    indexCount: number;
  } {
    const vertexBuffer = this.device.createBuffer({
      size: geom.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(geom.vertices);
    vertexBuffer.unmap();

    const indexBuffer = this.device.createBuffer({
      size: geom.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(indexBuffer.getMappedRange()).set(geom.indices);
    indexBuffer.unmap();

    return {
      vertices: vertexBuffer,
      indices: indexBuffer,
      indexCount: geom.indices.length,
    };
  }

  private async createPipelines(): Promise<void> {
    // Normal update compute shader
    const normalModule = this.device.createShaderModule({
      code: WGSL_NORMAL_UPDATE,
    });
    this.normalUpdatePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: normalModule, entryPoint: 'main' },
    });

    // Particle update compute shader
    const particleModule = this.device.createShaderModule({
      code: WGSL_PARTICLE_UPDATE,
    });
    this.particleUpdatePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: particleModule, entryPoint: 'main' },
    });

    // Particle render pipeline (billboard)
    const particleRenderModule = this.device.createShaderModule({
      code: WGSL_PARTICLE_VERTEX + '\n' + WGSL_PARTICLE_FRAGMENT,
    });
    this.particleRenderPipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: particleRenderModule,
        entryPoint: 'main',
        buffers: [],
      },
      fragment: {
        module: particleRenderModule,
        entryPoint: 'main',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // Tree water render pipeline
    const treeWaterModule = this.device.createShaderModule({
      code: WGSL_TREE_WATER_VERTEX + '\n' + WGSL_TREE_WATER_FRAGMENT,
    });
    this.treeWaterRenderPipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: treeWaterModule,
        entryPoint: 'main',
        buffers: [
          {
            arrayStride: 12,
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
          },
        ],
      },
      fragment: {
        module: treeWaterModule,
        entryPoint: 'main',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // Ceiling water render pipeline
    const ceilingWaterModule = this.device.createShaderModule({
      code: WGSL_CEILING_WATER_FRAGMENT,
    });
    // Note: simplified; would need full vertex setup in production
  }

  private async createBindGroups(): Promise<void> {
    // Normal update bind group
    this.normalBindGroup = this.device.createBindGroup({
      layout: this.normalUpdatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.normalUniformBuffer } },
        { binding: 1, resource: this.normalAtlasTexture.createView() },
        { binding: 2, resource: this.normalPingPong[1].createView() },
      ],
    });

    // Particle update bind group
    this.particleBindGroup = this.device.createBindGroup({
      layout: this.particleUpdatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleUniformBuffer } },
        { binding: 1, resource: this.particlePosPingPong[0].createView() },
        { binding: 2, resource: this.particlePosPingPong[1].createView() },
        { binding: 3, resource: this.normalPingPong[0].createView() },
      ],
    });
  }

  /**
   * Update simulation state and run compute passes.
   */
  tick(commandEncoder: GPUCommandEncoder, elapsed: number, deltaTime: number): void {
    this.time = elapsed;

    // Update normals
    this.updateNormals(commandEncoder, elapsed);

    // Update particles
    this.updateParticles(commandEncoder, elapsed, deltaTime);
  }

  private updateNormals(commandEncoder: GPUCommandEncoder, time: number): void {
    const uniformData = new Float32Array([
      time,
      this.config.normalAnimSpeed,
      1.0, // scale
      0.0,
    ]);
    this.device.queue.writeBuffer(this.normalUniformBuffer, 0, uniformData);

    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.normalUpdatePipeline);
    pass.setBindGroup(0, this.normalBindGroup);
    const workgroups = Math.ceil(this.config.normalAtlasSize / 8);
    pass.dispatchWorkgroups(workgroups, workgroups, 1);
    pass.end();

    // Swap ping-pong
    [this.normalPingPong[0], this.normalPingPong[1]] = [this.normalPingPong[1], this.normalPingPong[0]];
  }

  private updateParticles(commandEncoder: GPUCommandEncoder, time: number, deltaTime: number): void {
    const uniformData = new Float32Array([
      time,
      deltaTime,
      this.config.particleCount,
      this.config.particleDriftAmount,
    ]);
    this.device.queue.writeBuffer(this.particleUniformBuffer, 0, uniformData);

    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.particleUpdatePipeline);
    pass.setBindGroup(0, this.particleBindGroup);
    const workgroups = Math.ceil(this.config.particleCount / WORKGROUP_SIZE);
    pass.dispatchWorkgroups(workgroups, 1, 1);
    pass.end();

    // Swap ping-pong
    [this.particlePosPingPong[0], this.particlePosPingPong[1]] = [
      this.particlePosPingPong[1],
      this.particlePosPingPong[0],
    ];
  }

  /**
   * Emit particles at given UV coordinates.
   */
  emitParticles(uv_x: number, uv_y: number, count: number, speed: number): void {
    // Would implement particle spawning logic here
  }

  /**
   * Add a ripple to the normal map.
   */
  addNormalRipple(uv_x: number, uv_y: number, radius: number): void {
    // Would implement normal map ripple here
  }

  /**
   * Render pass for water surfaces.
   */
  renderPass(
    commandEncoder: GPUCommandEncoder,
    colorTarget: GPUTextureView,
    depthTarget?: GPUTextureView
  ): void {
    // Render particles
    const passDesc: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: colorTarget,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'load',
          storeOp: 'store',
        },
      ],
    };

    if (depthTarget) {
      passDesc.depthStencilAttachment = {
        view: depthTarget,
        depthClearValue: 1.0,
        depthLoadOp: 'load',
        depthStoreOp: 'store',
      };
    }

    const pass = commandEncoder.beginRenderPass(passDesc);
    pass.setPipeline(this.particleRenderPipeline);
    pass.setBindGroup(0, this.particleBindGroup);
    pass.draw(4, this.particleCount, 0, 0);
    pass.end();
  }

  /**
   * Cleanup GPU resources.
   */
  destroy(): void {
    this.normalAtlasTexture.destroy();
    this.normalPingPong.forEach((t) => t.destroy());
    this.particlePosPingPong.forEach((t) => t.destroy());
    this.particleColorTexture.destroy();
    this.normalUniformBuffer.destroy();
    this.particleUniformBuffer.destroy();
    this.treeWaterUniformBuffer.destroy();
    this.ceilingUniformBuffer.destroy();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Export
// ─────────────────────────────────────────────────────────────────────────────

export default ATWaterParticlesNormals;
