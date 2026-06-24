/**
 * src/lib/sph/at-water-surface.ts  —  M711
 *
 * AT WaterCeilingShader → WebGPU / WGSL Port
 * ─────────────────────────────────────────────────────────────────────────────
 * Full GPU port of Active Theory's WaterCeilingShader + WorkDetailParticles
 * water-surface effects into the project's WebGPU architecture.
 *
 * ─── AT Reverse-Engineered Sources ──────────────────────────────────────────
 *
 *   WaterCeilingShader  (RESEARCH_96_AT_TECHSTACK.md: "水面反射/折射")
 *     ← upstream/webgl-water/water.js        (stepSimulation / updateNormals)
 *     ← upstream/webgl-water/renderer.js     (waterShaders[0] above-water path)
 *     Key algorithms:
 *       • Wave propagation  : neighbour-average relaxation (velocity + damping)
 *       • Normal update     : cross(dy, dx).xz stored in ba channel
 *       • Fresnel blend     : mix(refracted, reflected, fresnel) above water
 *       • Mirror reflection : reflect(incomingRay, normal) → sky cubemap sample
 *
 *   WorkDetailParticles  (RESEARCH_96_AT_TECHSTACK.md: "GPU粒子效果")
 *     ← upstream/sketch-js/                  (lifecycle / spawn helpers)
 *     ← upstream/webgl-noise/src/noise3D.glsl (simplex noise for drift)
 *     Key algorithms:
 *       • GPGPU ping-pong   : particle state stored in rgba32float textures
 *       • Lifecycle         : SPAWN → FLOAT → DECAY → DEAD
 *       • Surface drift     : water normal displaces particle rise velocity
 *       • Billboard quads   : instanced 2-triangle strip per particle slot
 *
 * ─── WebGPU Architecture ────────────────────────────────────────────────────
 *
 *   ATWaterSurface
 *     ├─ Wave sim pipeline  (compute)
 *     │    waterA / waterB  — rgba32float ping-pong  (height, velocity, nx, nz)
 *     │    stepSimulation   — 4-neighbour relaxation + damping
 *     │    updateNormals    — cross product → normal.xz
 *     │
 *     ├─ Water render pipeline  (render)
 *     │    vertex shader    — reads waterA height, displaces mesh Y
 *     │    fragment shader  — Fresnel mirror reflection + refraction tint
 *     │    sky cubemap      — sampled via reflected ray (or fallback gradient)
 *     │
 *     └─ Particle pipeline  (compute + render)
 *          particleA / B    — rgba32float ping-pong  (pos.xyz, life/speed)
 *          particleUpdate   — lifecycle + simplex drift along surface normal
 *          particleRender   — billboard quads, alpha = sin(π · life), additive
 *
 * ─── GLSL → WGSL Translation Key ────────────────────────────────────────────
 *
 *   texture2D(water, coord)              → textureSample(water, samp, coord)
 *   varying vec2 coord                   → @location(0) uv : vec2f
 *   uniform vec3 light                   → uniforms.lightDir
 *   info.r / .g / .b / .a               → waterTex.r / .g / .b / .a
 *   refract(I, N, eta)                   → refract_wgsl(I, N, eta)  (manual)
 *   reflect(I, N)                        → reflect_wgsl(I, N)        (manual)
 *   gl_FragColor = …                     → @location(0) out : vec4f
 *   dFdx / dFdy                          → not used — normal from texture
 *   mod289 / permute / taylorInvSqrt     → inlined simplex noise helpers
 *
 * ─── Upstream Copyright Notices ─────────────────────────────────────────────
 *
 *   WebGL Water          © 2011 Evan Wallace  — MIT License
 *     http://madebyevan.com/webgl-water/
 *   webgl-noise (ashima) © 2011 Ashima Arts  — MIT License
 *     https://github.com/ashima/webgl-noise
 *
 * ─── Usage ───────────────────────────────────────────────────────────────────
 *
 *   const water = new ATWaterSurface(device, format, config);
 *   await water.build();
 *
 *   // Optional: seed initial drops / SPH coupling
 *   water.addDrop(0.5, 0.5, 0.08, 0.12);
 *
 *   // Render loop:
 *   const enc = device.createCommandEncoder();
 *   water.tick(enc, elapsedSeconds, deltaSeconds);
 *   water.renderPass(enc, colorTargetView, depthView);
 *   device.queue.submit([enc.finish()]);
 *
 * Research: xiaodi #M711 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Water simulation grid resolution (NxN). Power-of-two recommended. */








const WATER_SIM_SIZE = 256 as const;

/** Mesh tessellation detail for the rendered water surface. */
const WATER_MESH_DETAIL = 128 as const;

/** Maximum water particles in the GPU pool. */
const MAX_WATER_PARTICLES = 8192 as const;

/** Workgroup size for compute shaders. */
const WG = 64 as const;

/** Wave propagation damping factor (matches water.js `info.g *= 0.995`). */
const WAVE_DAMPING = 0.995 as const;

/** IOR constants from renderer.js helperFunctions. */
const IOR_AIR   = 1.0 as const;
const IOR_WATER = 1.333 as const;

// ─────────────────────────────────────────────────────────────────────────────
// Config types
// ─────────────────────────────────────────────────────────────────────────────

export interface ATWaterSurfaceConfig {
  /**
   * Simulation grid size (NxN).  Default 256.
   * Smaller values (64/128) are faster; 512 is photorealistic.
   */
  simSize?: number;

  /**
   * Mesh tessellation for the rendered surface.  Default 128.
   */
  meshDetail?: number;

  /**
   * Maximum number of water surface particles.  Default 8192.
   */
  maxParticles?: number;

  /**
   * Light direction (world space, need not be unit).
   * Default: [2, 2, -1]  (matches renderer.js lightDir).
   */
  lightDir?: [number, number, number];

  /**
   * Water colour tint applied to refracted rays (above-water path).
   * Default: [0.25, 1.0, 1.25]  (abovewaterColor from renderer.js).
   */
  waterColor?: [number, number, number];

  /**
   * Colour of the sky gradient used when no cubemap is provided.
   * Default: [0.1, 0.6, 1.0].
   */
  skyColor?: [number, number, number];

  /**
   * Particle tint colour.  Default: [0.6, 0.9, 1.0].
   */
  particleColor?: [number, number, number];

  /**
   * Number of compute ticks per rendered frame (wave speed multiplier).
   * Default 2.
   */
  stepsPerFrame?: number;

  /**
   * Optional sky cubemap texture (GPUTexture, cube dimension).
   * If not provided, a simple gradient sky is used.
   */
  skyCubemap?: GPUTexture;
}

// ─────────────────────────────────────────────────────────────────────────────
// Uniform buffer layout (must match WGSL struct WaterUniforms)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Total uniform buffer size: 16 × f32 = 64 bytes (one vec4 per row, 4 rows).
 *
 *   offset  0: lightDir.xyz  + time
 *   offset 16: waterColor.xyz + stepsPerFrame
 *   offset 32: skyColor.xyz   + damping
 *   offset 48: particleColor.xyz + reserved
 */
const UNIFORM_SIZE = 64 as const;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Shared math helpers
// (simplex noise ported from upstream/webgl-noise/src/noise3D.glsl)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_MATH = /* wgsl */`
// ── mod289 / permute / taylorInvSqrt (ashima webgl-noise MIT) ────────────────
fn mod289_v3(x: vec3f) -> vec3f { return x - floor(x * (1.0/289.0)) * 289.0; }
fn mod289_v4(x: vec4f) -> vec4f { return x - floor(x * (1.0/289.0)) * 289.0; }
fn permute(x: vec4f) -> vec4f { return mod289_v4((x * 34.0 + 10.0) * x); }
fn taylorInvSqrt(r: vec4f) -> vec4f { return vec4f(1.79284291400159) - 0.85373472095314 * r; }

// ── Simplex 3D noise (snoise3) ────────────────────────────────────────────────
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

// ── Curl noise (∇×Ψ, 3D simplex basis) ───────────────────────────────────────
fn curl3(p: vec3f) -> vec3f {
    let eps = 0.001;
    let dx  = vec3f(eps, 0.0, 0.0);
    let dy  = vec3f(0.0, eps, 0.0);
    let dz  = vec3f(0.0, 0.0, eps);
    // ∂/∂y(Fz) - ∂/∂z(Fy)
    let px = (snoise3(p + dy) - snoise3(p - dy)) / (2.0*eps)
           - (snoise3(p + dz) - snoise3(p - dz)) / (2.0*eps);
    // ∂/∂z(Fx) - ∂/∂x(Fz)
    let py = (snoise3(p + dz) - snoise3(p - dz)) / (2.0*eps)
           - (snoise3(p + dx) - snoise3(p - dx)) / (2.0*eps);
    // ∂/∂x(Fy) - ∂/∂y(Fx)
    let pz = (snoise3(p + dx) - snoise3(p - dx)) / (2.0*eps)
           - (snoise3(p + dy) - snoise3(p - dy)) / (2.0*eps);
    return vec3f(px, py, pz);
}

// ── Manual refract (WGSL has no built-in) ────────────────────────────────────
fn refract_wgsl(I: vec3f, N: vec3f, eta: f32) -> vec3f {
    let cosI = dot(N, I);
    let k    = 1.0 - eta*eta * (1.0 - cosI*cosI);
    if (k < 0.0) { return vec3f(0.0); }
    return eta * I - (eta * cosI + sqrt(k)) * N;
}

// ── Manual reflect ────────────────────────────────────────────────────────────
fn reflect_wgsl(I: vec3f, N: vec3f) -> vec3f {
    return I - 2.0 * dot(N, I) * N;
}

// ── Simple sky gradient (fallback when no cubemap) ────────────────────────────
fn skyGradient(dir: vec3f, skyCol: vec3f) -> vec3f {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(skyCol * 0.3, skyCol, t);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Uniforms struct (shared across all passes)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_UNIFORMS = /* wgsl */`
struct WaterUniforms {
    lightDir      : vec4f,  // .xyz = lightDir, .w = time
    waterColor    : vec4f,  // .xyz = tint,     .w = stepsPerFrame (unused in shader)
    skyColor      : vec4f,  // .xyz = sky tint, .w = damping
    particleColor : vec4f,  // .xyz = pColor,   .w = reserved
};
@group(0) @binding(0) var<uniform> u : WaterUniforms;
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Wave simulation compute shaders
// Port of water.js stepSimulation + updateNormals
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compute shader: one step of the wave propagation.
 *
 * Ported from water.js updateShader:
 *   average = (T(coord-dx).r + T(coord-dy).r + T(coord+dx).r + T(coord+dy).r) * 0.25
 *   info.g += (average - info.r) * 2.0
 *   info.g *= 0.995
 *   info.r += info.g
 *
 * Texture layout (rgba32float):
 *   r = height, g = velocity, b = normal.x, a = normal.z
 */
const WGSL_WAVE_STEP = /* wgsl */`
${WGSL_UNIFORMS}

@group(0) @binding(1) var src : texture_2d<f32>;
@group(0) @binding(2) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn cs_step(@builtin(global_invocation_id) gid : vec3u) {
    let dim  = vec2i(textureDimensions(src));
    let px   = vec2i(gid.xy);
    if (px.x >= dim.x || px.y >= dim.y) { return; }

    let info = textureLoad(src, px, 0);
    let h    = info.r;
    let vel  = info.g;
    let damping = u.skyColor.w;  // packed in .w

    // Clamp-to-edge neighbours
    let L = textureLoad(src, clamp(px + vec2i(-1,  0), vec2i(0), dim-1), 0).r;
    let R = textureLoad(src, clamp(px + vec2i( 1,  0), vec2i(0), dim-1), 0).r;
    let U = textureLoad(src, clamp(px + vec2i( 0, -1), vec2i(0), dim-1), 0).r;
    let D = textureLoad(src, clamp(px + vec2i( 0,  1), vec2i(0), dim-1), 0).r;

    let avg    = (L + R + U + D) * 0.25;
    var newVel = vel + (avg - h) * 2.0;
    newVel    *= damping;
    let newH   = h + newVel;

    // Preserve normal channels from src (will be overwritten in normal pass)
    textureStore(dst, px, vec4f(newH, newVel, info.b, info.a));
}
`;

/**
 * Compute shader: recompute surface normals.
 *
 * Ported from water.js normalShader:
 *   dx = vec3(delta.x, T(coord + (delta.x, 0)).r - info.r, 0)
 *   dy = vec3(0, T(coord + (0, delta.y)).r - info.r, delta.y)
 *   info.ba = normalize(cross(dy, dx)).xz
 */
const WGSL_WAVE_NORMAL = /* wgsl */`
${WGSL_UNIFORMS}

@group(0) @binding(1) var src : texture_2d<f32>;
@group(0) @binding(2) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn cs_normal(@builtin(global_invocation_id) gid : vec3u) {
    let dim  = vec2i(textureDimensions(src));
    let px   = vec2i(gid.xy);
    if (px.x >= dim.x || px.y >= dim.y) { return; }

    let info  = textureLoad(src, px, 0);
    let h     = info.r;
    let delta = 1.0 / f32(dim.x);

    let hR = textureLoad(src, clamp(px + vec2i(1, 0), vec2i(0), dim-1), 0).r;
    let hD = textureLoad(src, clamp(px + vec2i(0, 1), vec2i(0), dim-1), 0).r;

    // vec3 dx = (delta, hR - h, 0),  vec3 dy = (0, hD - h, delta)
    let vdx = vec3f(delta, hR - h, 0.0);
    let vdy = vec3f(0.0,   hD - h, delta);

    // normal = normalize(cross(dy, dx))
    let nx_raw = normalize(cross(vdy, vdx));

    textureStore(dst, px, vec4f(info.r, info.g, nx_raw.x, nx_raw.z));
}
`;

/**
 * Compute shader: add a raindrop perturbation.
 *
 * Ported from water.js dropShader:
 *   drop = max(0, 1 - length(center*0.5+0.5 - coord) / radius)
 *   drop = 0.5 - cos(drop * PI) * 0.5
 *   info.r += drop * strength
 */
const WGSL_WAVE_DROP = /* wgsl */`
${WGSL_UNIFORMS}

struct DropParams {
    center   : vec2f,   // drop centre in [-1,1]²
    radius   : f32,
    strength : f32,
};
@group(0) @binding(1) var<uniform> drop : DropParams;
@group(0) @binding(2) var src : texture_2d<f32>;
@group(0) @binding(3) var dst : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn cs_drop(@builtin(global_invocation_id) gid : vec3u) {
    let dim = vec2i(textureDimensions(src));
    let px  = vec2i(gid.xy);
    if (px.x >= dim.x || px.y >= dim.y) { return; }

    let coord  = (vec2f(px) + 0.5) / vec2f(dim);            // UV [0,1]
    let centre = drop.center * 0.5 + 0.5;
    let d      = max(0.0, 1.0 - length(centre - coord) / drop.radius);
    let contribution = (0.5 - cos(d * 3.14159265) * 0.5) * drop.strength;

    var info = textureLoad(src, px, 0);
    info.r  += contribution;
    textureStore(dst, px, info);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Water surface render shader
// Port of renderer.js waterShaders[0] (above-water path)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_WATER_RENDER = /* wgsl */`
${WGSL_UNIFORMS}

@group(0) @binding(1) var waterSamp : sampler;
@group(0) @binding(2) var waterTex  : texture_2d<f32>;       // simulation state
@group(0) @binding(3) var skySamp   : sampler;
@group(0) @binding(4) var skyTex    : texture_cube<f32>;      // env cubemap

struct SceneUniforms {
    viewProj    : mat4x4f,
    modelMat    : mat4x4f,
    eye         : vec4f,
};
@group(1) @binding(0) var<uniform> scene : SceneUniforms;

struct VertexOut {
    @builtin(position) pos      : vec4f,
    @location(0)       worldPos : vec3f,
    @location(1)       uv       : vec2f,
};

${WGSL_MATH}

// ── Vertex ────────────────────────────────────────────────────────────────────
// Reads the height from waterTex and displaces the mesh vertex Y.
// Mesh verts arrive in [-1,1]² XZ space (Y = 0).
@vertex
fn vs_water(
    @location(0) aPos : vec2f,   // XZ plane position [-1, 1]
) -> VertexOut {
    let uv    = aPos * 0.5 + 0.5;                         // [0, 1]
    let info  = textureSampleLevel(waterTex, waterSamp, uv, 0.0);
    let height = info.r;

    let worldPos = vec3f(aPos.x, height, aPos.y);
    var out: VertexOut;
    out.pos      = scene.viewProj * scene.modelMat * vec4f(worldPos, 1.0);
    out.worldPos = worldPos;
    out.uv       = uv;
    return out;
}

// ── Fragment ──────────────────────────────────────────────────────────────────
// Above-water Fresnel:
//   fresnel = mix(0.25, 1.0, pow(1.0 - dot(N, -incoming), 3.0))
//   reflectedColor  = sky cubemap along reflect(incoming, N)
//   refractedColor  = sky cubemap along refract(incoming, N, IOR_AIR/IOR_WATER) * waterColor
//   gl_FragColor    = mix(refracted, reflected, fresnel)
@fragment
fn fs_water(in: VertexOut) -> @location(0) vec4f {
    var uv   = in.uv;
    var info = textureSample(waterTex, waterSamp, uv);

    // Iterative UV refinement: "make water look more peaked" (renderer.js loop)
    for (var i = 0; i < 5; i++) {
        uv   += info.ba * 0.005;
        info  = textureSample(waterTex, waterSamp, uv);
    }

    // Surface normal from packed ba channels
    // info.ba *= 0.5  (caustics vertex shader convention for normals)
    let ba      = info.ba * 0.5;
    let ny      = sqrt(max(0.0, 1.0 - dot(ba, ba)));
    let N       = normalize(vec3f(ba.x, ny, ba.y));

    let eye_vec  = normalize(in.worldPos - scene.eye.xyz);
    let light    = normalize(u.lightDir.xyz);

    // ── Fresnel (above-water) ─────────────────────────────────────────────────
    let NdotV    = max(0.0, dot(N, -eye_vec));
    let fresnel  = mix(0.25, 1.0, pow(clamp(1.0 - NdotV, 0.0, 1.0), 3.0));

    // ── Reflection ───────────────────────────────────────────────────────────
    let reflRay    = reflect_wgsl(eye_vec, N);
    let reflColor  = textureSampleLevel(skyTex, skySamp, reflRay, 0.0).rgb;

    // ── Refraction ───────────────────────────────────────────────────────────
    let refrRay    = refract_wgsl(eye_vec, N, ${IOR_AIR} / ${IOR_WATER});
    var refrColor  = textureSampleLevel(skyTex, skySamp, refrRay, 0.0).rgb;
    refrColor     *= u.waterColor.xyz;     // abovewaterColor tint

    // ── Specular highlight (sun disk on water) ────────────────────────────────
    let H        = normalize(light - eye_vec);
    let spec     = pow(max(0.0, dot(H, N)), 5000.0);
    let sunColor = vec3f(10.0, 8.0, 6.0);

    var finalColor = mix(refrColor, reflColor, fresnel) + spec * sunColor * 0.15;
    return vec4f(finalColor, 0.85);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Water surface render (fallback: gradient sky, no cubemap)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_WATER_RENDER_FLAT = /* wgsl */`
${WGSL_UNIFORMS}

@group(0) @binding(1) var waterSamp : sampler;
@group(0) @binding(2) var waterTex  : texture_2d<f32>;

struct SceneUniforms {
    viewProj    : mat4x4f,
    modelMat    : mat4x4f,
    eye         : vec4f,
};
@group(1) @binding(0) var<uniform> scene : SceneUniforms;

struct VertexOut {
    @builtin(position) pos      : vec4f,
    @location(0)       worldPos : vec3f,
    @location(1)       uv       : vec2f,
};

${WGSL_MATH}

@vertex
fn vs_water_flat(
    @location(0) aPos : vec2f,
) -> VertexOut {
    let uv    = aPos * 0.5 + 0.5;
    let info  = textureSampleLevel(waterTex, waterSamp, uv, 0.0);
    let worldPos = vec3f(aPos.x, info.r, aPos.y);
    var out: VertexOut;
    out.pos      = scene.viewProj * scene.modelMat * vec4f(worldPos, 1.0);
    out.worldPos = worldPos;
    out.uv       = uv;
    return out;
}

@fragment
fn fs_water_flat(in: VertexOut) -> @location(0) vec4f {
    var uv   = in.uv;
    var info = textureSample(waterTex, waterSamp, uv);
    for (var i = 0; i < 5; i++) {
        uv   += info.ba * 0.005;
        info  = textureSample(waterTex, waterSamp, uv);
    }

    let ba     = info.ba * 0.5;
    let ny     = sqrt(max(0.0, 1.0 - dot(ba, ba)));
    let N      = normalize(vec3f(ba.x, ny, ba.y));
    let eye_v  = normalize(in.worldPos - scene.eye.xyz);
    let light  = normalize(u.lightDir.xyz);

    let NdotV  = max(0.0, dot(N, -eye_v));
    let fresnel = mix(0.25, 1.0, pow(clamp(1.0 - NdotV, 0.0, 1.0), 3.0));

    let reflRay   = reflect_wgsl(eye_v, N);
    let reflColor = skyGradient(reflRay, u.skyColor.xyz);
    let refrRay   = refract_wgsl(eye_v, N, ${IOR_AIR} / ${IOR_WATER});
    var refrColor = skyGradient(refrRay, u.skyColor.xyz) * u.waterColor.xyz;

    let H    = normalize(light - eye_v);
    let spec = pow(max(0.0, dot(H, N)), 800.0);

    var col = mix(refrColor, reflColor, fresnel) + vec3f(spec) * 0.5;
    return vec4f(col, 0.85);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Water particle compute (GPGPU lifecycle + drift)
// Lifecycle ported from WorkDetailParticles / sketch-js particle pattern
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Particle state: rgba32float, one texel per particle.
 *   r = world-X
 *   g = world-Y
 *   b = world-Z
 *   a = life  (0.0 = dead, 1.0 = full life, decreasing over lifetime)
 */
const WGSL_PARTICLE_UPDATE = /* wgsl */`
${WGSL_UNIFORMS}

@group(0) @binding(1) var waterSamp : sampler;
@group(0) @binding(2) var waterTex  : texture_2d<f32>;
@group(0) @binding(3) var pSrc      : texture_2d<f32>;
@group(0) @binding(4) var pDst      : texture_storage_2d<rgba32float, write>;

${WGSL_MATH}

// ── Pseudo-random hash (no external texture) ──────────────────────────────────
fn hash2(p: vec2f) -> f32 {
    let q = fract(p * vec2f(127.1, 311.7));
    return fract(dot(q, q + vec2f(19.19)) * 43758.5453);
}

@compute @workgroup_size(64)
fn cs_particle_update(@builtin(global_invocation_id) gid : vec3u) {
    let idx  = gid.x;
    let dim  = vec2u(textureDimensions(pSrc));
    if (idx >= dim.x * dim.y) { return; }

    let px   = vec2i(i32(idx % dim.x), i32(idx / dim.x));
    var p    = textureLoad(pSrc, px, 0);  // (x, y, z, life)
    let time = u.lightDir.w;              // time packed in .w

    // ── Dead particle: respawn at random water-surface location ───────────────
    let seed  = vec2f(f32(idx) * 0.0013, time * 0.07);
    let spawn = hash2(seed + vec2f(0.3, 0.7)) < 0.004;  // ~0.4% per frame

    if (p.a <= 0.0 || spawn) {
        let rx = hash2(seed) * 2.0 - 1.0;              // [-1, 1]
        let rz = hash2(seed + vec2f(1.0, 0.0)) * 2.0 - 1.0;
        let uv = vec2f(rx * 0.5 + 0.5, rz * 0.5 + 0.5);
        let info   = textureSample(waterTex, waterSamp, uv);
        let spawnY = info.r + 0.02;
        p = vec4f(rx, spawnY, rz, 1.0);
        textureStore(pDst, px, p);
        return;
    }

    // ── Live particle: drift upward along surface normal + curl noise ─────────
    let uv      = vec2f(p.x * 0.5 + 0.5, p.z * 0.5 + 0.5);
    let info    = textureSample(waterTex, waterSamp, uv);

    // Surface normal from waterTex ba channels
    let ba      = info.ba * 0.5;
    let ny      = sqrt(max(0.0, 1.0 - dot(ba, ba)));
    let surfN   = normalize(vec3f(ba.x, ny, ba.y));

    // Curl noise drift in world space (simplex basis)
    let curlIn  = vec3f(p.x, p.z, time * 0.3);
    let drift   = curl3(curlIn) * 0.004;

    // Rise velocity: mostly upward, nudged by surface normal
    let rise    = (surfN * 0.006 + vec3f(0.0, 0.012, 0.0));

    // Decay
    let decay   = 0.008;

    p.x += rise.x + drift.x;
    p.y += rise.y + drift.y;
    p.z += rise.z + drift.z;
    p.a -= decay;

    // Clamp position to scene bounds
    p.x = clamp(p.x, -1.2, 1.2);
    p.z = clamp(p.z, -1.2, 1.2);

    textureStore(pDst, px, p);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Water particle render
// Billboard quads with alpha = sin(π · life) and additive blending
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_PARTICLE_RENDER = /* wgsl */`
${WGSL_UNIFORMS}

@group(0) @binding(1) var pSamp : sampler;
@group(0) @binding(2) var pTex  : texture_2d<f32>;   // particle state

struct SceneUniforms {
    viewProj    : mat4x4f,
    modelMat    : mat4x4f,
    eye         : vec4f,
};
@group(1) @binding(0) var<uniform> scene : SceneUniforms;

struct VertexOut {
    @builtin(position) pos    : vec4f,
    @location(0)       uv     : vec2f,
    @location(1)       life   : f32,
    @location(2)       color  : vec3f,
};

// Billboard quad corners (2 triangles = 6 vertices)
const QUAD : array<vec2f, 6> = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
);

@vertex
fn vs_particle(
    @builtin(vertex_index)   vid  : u32,
    @builtin(instance_index) inst : u32,
) -> VertexOut {
    let texDim = vec2u(textureDimensions(pTex));
    let px     = vec2i(i32(inst % texDim.x), i32(inst / texDim.x));
    let state  = textureLoad(pTex, px, 0);  // (x, y, z, life)
    let life   = state.a;

    let corner  = QUAD[vid];
    let size    = 0.015 * life;               // shrink as particle dies

    // Build camera-facing billboard basis
    let viewRight = vec3f(scene.viewProj[0][0], scene.viewProj[1][0], scene.viewProj[2][0]);
    let viewUp    = vec3f(scene.viewProj[0][1], scene.viewProj[1][1], scene.viewProj[2][1]);
    let worldPos  = vec3f(state.xyz)
                  + viewRight * corner.x * size
                  + viewUp    * corner.y * size;

    var out: VertexOut;
    out.pos   = scene.viewProj * scene.modelMat * vec4f(worldPos, 1.0);
    out.uv    = corner * 0.5 + 0.5;
    out.life  = life;
    out.color = u.particleColor.xyz;
    return out;
}

@fragment
fn fs_particle(in: VertexOut) -> @location(0) vec4f {
    if (in.life <= 0.0) { discard; }

    // Soft circle mask
    let d    = length(in.uv - 0.5) * 2.0;
    let mask = 1.0 - smoothstep(0.7, 1.0, d);

    // Alpha: sin(π · life) — bright at spawn, fade to zero
    let alpha = sin(3.14159265 * clamp(in.life, 0.0, 1.0)) * mask * 0.6;

    return vec4f(in.color, alpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Main class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * ATWaterSurface — WebGPU port of Active Theory's WaterCeilingShader
 * and WorkDetailParticles water-surface system.
 *
 * Provides:
 *   • GPU wave simulation (ping-pong rgba32float textures)
 *   • Fresnel mirror-reflection render with sky cubemap or gradient fallback
 *   • Water surface particles with lifecycle + curl-noise drift
 */
export class ATWaterSurface {
  // ── Config ──────────────────────────────────────────────────────────────────
  private readonly simSize:      number;
  private readonly meshDetail:   number;
  private readonly maxParticles: number;
  private readonly stepsPerFrame: number;
  private readonly cfg: Required<ATWaterSurfaceConfig>;

  // ── WebGPU ──────────────────────────────────────────────────────────────────
  private readonly device:  GPUDevice;
  private readonly format:  GPUTextureFormat;

  // ── Simulation textures (ping-pong) ─────────────────────────────────────────
  private waterA!: GPUTexture;
  private waterB!: GPUTexture;

  // ── Particle textures (ping-pong) ───────────────────────────────────────────
  private particleA!: GPUTexture;
  private particleB!: GPUTexture;

  // ── Mesh ────────────────────────────────────────────────────────────────────
  private meshVB!:     GPUBuffer;
  private meshVCount!: number;

  // ── Uniform buffers ─────────────────────────────────────────────────────────
  private uniformBuf!: GPUBuffer;
  private dropBuf!:    GPUBuffer;

  // ── Pipelines ───────────────────────────────────────────────────────────────
  private stepPipeline!:     GPUComputePipeline;
  private normalPipeline!:   GPUComputePipeline;
  private dropPipeline!:     GPUComputePipeline;
  private waterRenderPipe!:  GPURenderPipeline;
  private particleUpdatePipe!: GPUComputePipeline;
  private particleRenderPipe!: GPURenderPipeline;

  // ── Bind groups ─────────────────────────────────────────────────────────────
  // Computed fresh each frame (ping-pong swaps textures)
  private sampler!: GPUSampler;

  // ── State ───────────────────────────────────────────────────────────────────
  private built = false;
  private pendingDrops: Array<{x: number; y: number; radius: number; strength: number}> = [];

  // ─── Constructor ────────────────────────────────────────────────────────────

  constructor(
    device:  GPUDevice,
    format:  GPUTextureFormat,
    cfg:     ATWaterSurfaceConfig = {},
  ) {
    this.device  = device;
    this.format  = format;

    const defaultLight: [number, number, number] = [2.0, 2.0, -1.0];
    const len = Math.hypot(...defaultLight);

    this.cfg = {
      simSize:       cfg.simSize       ?? WATER_SIM_SIZE,
      meshDetail:    cfg.meshDetail    ?? WATER_MESH_DETAIL,
      maxParticles:  cfg.maxParticles  ?? MAX_WATER_PARTICLES,
      lightDir:      cfg.lightDir      ?? [
        defaultLight[0]/len, defaultLight[1]/len, defaultLight[2]/len,
      ],
      waterColor:    cfg.waterColor    ?? [0.25, 1.0, 1.25],
      skyColor:      cfg.skyColor      ?? [0.1, 0.6, 1.0],
      particleColor: cfg.particleColor ?? [0.6, 0.9, 1.0],
      stepsPerFrame: cfg.stepsPerFrame ?? 2,
      skyCubemap:    cfg.skyCubemap,
    };

    this.simSize       = this.cfg.simSize;
    this.meshDetail    = this.cfg.meshDetail;
    this.maxParticles  = this.cfg.maxParticles;
    this.stepsPerFrame = this.cfg.stepsPerFrame;
  }

  // ─── Public API ─────────────────────────────────────────────────────────────

  /**
   * Compile all pipelines, allocate textures, upload initial mesh.
   * Must be called before tick() / renderPass().
   */
  async build(): Promise<void> {
    this._createTextures();
    this._createMesh();
    this._createUniformBuffers();
    this._createSampler();
    await this._createPipelines();
    this.built = true;
  }

  /**
   * Enqueue a raindrop disturbance.
   * Port of Water.prototype.addDrop — safe to call before build().
   *
   * @param x        Centre X in [-1, 1] (water.js convention).
   * @param y        Centre Y in [-1, 1].
   * @param radius   Radius in the same space.
   * @param strength Height perturbation amplitude.
   */
  addDrop(x: number, y: number, radius: number, strength: number): void {
    this.pendingDrops.push({ x, y, radius, strength });
  }

  /**
   * Run one frame's worth of simulation and particle update.
   *
   * @param encoder  Open GPUCommandEncoder.
   * @param time     Elapsed seconds.
   * @param _dt      Delta seconds (unused currently — wave speed is fixed).
   */
  tick(encoder: GPUCommandEncoder, time: number, _dt: number = 0): void {
    if (!this.built) return;

    this._writeUniforms(time);

    // ── Apply pending drops ───────────────────────────────────────────────────
    for (const drop of this.pendingDrops) {
      this._dispatchDrop(encoder, drop);
    }
    this.pendingDrops.length = 0;

    // ── Wave simulation steps (N per frame for speed control) ─────────────────
    for (let s = 0; s < this.stepsPerFrame; s++) {
      this._dispatchStep(encoder);
      this._dispatchNormal(encoder);
    }

    // ── Particle update ───────────────────────────────────────────────────────
    this._dispatchParticleUpdate(encoder);
  }

  /**
   * Encode the water surface + particle render passes.
   *
   * @param encoder         Open GPUCommandEncoder.
   * @param colorTarget     Output color attachment view.
   * @param depthView       Output depth attachment view.
   * @param sceneUniformBuf GPUBuffer holding scene matrices (mat4×2 + vec4 eye).
   */
  renderPass(
    encoder:         GPUCommandEncoder,
    colorTarget:     GPUTextureView,
    depthView:       GPUTextureView,
    sceneUniformBuf: GPUBuffer,
  ): void {
    if (!this.built) return;

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view:       colorTarget,
        loadOp:     'load',
        storeOp:    'store',
      }],
      depthStencilAttachment: {
        view:              depthView,
        depthLoadOp:       'load',
        depthStoreOp:      'store',
      },
    });

    // ── Water surface ─────────────────────────────────────────────────────────
    pass.setPipeline(this.waterRenderPipe);
    pass.setBindGroup(0, this._makeWaterRenderBG0(sceneUniformBuf));
    pass.setBindGroup(1, this._makeSceneBG1(sceneUniformBuf));
    pass.setVertexBuffer(0, this.meshVB);
    pass.draw(this.meshVCount);

    // ── Water particles ───────────────────────────────────────────────────────
    pass.setPipeline(this.particleRenderPipe);
    pass.setBindGroup(0, this._makeParticleRenderBG0());
    pass.setBindGroup(1, this._makeSceneBG1(sceneUniformBuf));
    pass.draw(6, this.maxParticles);  // 6 verts per billboard quad, instanced

    pass.end();
  }

  /**
   * Release all GPU resources.
   */
  destroy(): void {
    this.waterA.destroy();
    this.waterB.destroy();
    this.particleA.destroy();
    this.particleB.destroy();
    this.meshVB.destroy();
    this.uniformBuf.destroy();
    this.dropBuf.destroy();
  }

  // ─── Private: resource creation ─────────────────────────────────────────────

  private _createTextures(): void {
    const simDesc: GPUTextureDescriptor = {
      size:   [this.simSize, this.simSize, 1],
      format: 'rgba32float',
      usage:  GPUTextureUsage.TEXTURE_BINDING
            | GPUTextureUsage.STORAGE_BINDING
            | GPUTextureUsage.COPY_DST,
    };
    this.waterA = this.device.createTexture(simDesc);
    this.waterB = this.device.createTexture(simDesc);

    // Particle texture: one row of maxParticles texels (or square layout)
    const pW = Math.min(this.maxParticles, 256);
    const pH = Math.ceil(this.maxParticles / pW);
    const pDesc: GPUTextureDescriptor = {
      size:   [pW, pH, 1],
      format: 'rgba32float',
      usage:  GPUTextureUsage.TEXTURE_BINDING
            | GPUTextureUsage.STORAGE_BINDING
            | GPUTextureUsage.COPY_DST,
    };
    this.particleA = this.device.createTexture(pDesc);
    this.particleB = this.device.createTexture(pDesc);

    // Seed particle texture with dead particles (life = 0)
    const pPixels = pW * pH * 4;
    const pData = new Float32Array(pPixels);
    // All particles start dead; they'll respawn via the shader's hash2 path
    this.device.queue.writeTexture(
      { texture: this.particleA },
      pData,
      { bytesPerRow: pW * 16 },
      { width: pW, height: pH },
    );
    this.device.queue.writeTexture(
      { texture: this.particleB },
      pData,
      { bytesPerRow: pW * 16 },
      { width: pW, height: pH },
    );
  }

  private _createMesh(): void {
    // Flat XZ grid mesh: meshDetail × meshDetail quads
    const N     = this.meshDetail;
    const verts: number[] = [];
    for (let row = 0; row < N; row++) {
      for (let col = 0; col < N; col++) {
        const x0 = (col / N) * 2 - 1;
        const x1 = ((col + 1) / N) * 2 - 1;
        const z0 = (row / N) * 2 - 1;
        const z1 = ((row + 1) / N) * 2 - 1;
        // Two triangles per quad (CCW)
        verts.push(x0, z0,  x1, z0,  x0, z1);
        verts.push(x0, z1,  x1, z0,  x1, z1);
      }
    }
    this.meshVCount = verts.length / 2;
    const data = new Float32Array(verts);
    this.meshVB = this.device.createBuffer({
      size:  data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.meshVB, 0, data);
  }

  private _createUniformBuffers(): void {
    this.uniformBuf = this.device.createBuffer({
      size:  UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Drop params buffer: vec2 center + float radius + float strength = 16 bytes
    this.dropBuf = this.device.createBuffer({
      size:  16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private _createSampler(): void {
    this.sampler = this.device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      magFilter:    'linear',
      minFilter:    'linear',
    });
  }

  private async _createPipelines(): Promise<void> {
    const d = this.device;

    // ── Wave step compute ─────────────────────────────────────────────────────
    const stepMod = d.createShaderModule({ code: WGSL_WAVE_STEP });
    this.stepPipeline = await d.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: stepMod, entryPoint: 'cs_step' },
    });

    // ── Normal compute ────────────────────────────────────────────────────────
    const normMod = d.createShaderModule({ code: WGSL_WAVE_NORMAL });
    this.normalPipeline = await d.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: normMod, entryPoint: 'cs_normal' },
    });

    // ── Drop compute ──────────────────────────────────────────────────────────
    const dropMod = d.createShaderModule({ code: WGSL_WAVE_DROP });
    this.dropPipeline = await d.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: dropMod, entryPoint: 'cs_drop' },
    });

    // ── Water render ──────────────────────────────────────────────────────────
    const hasCubemap    = !!this.cfg.skyCubemap;
    const waterRenderSrc = hasCubemap ? WGSL_WATER_RENDER : WGSL_WATER_RENDER_FLAT;
    const waterRenderMod = d.createShaderModule({ code: waterRenderSrc });
    const vsEntry  = hasCubemap ? 'vs_water'      : 'vs_water_flat';
    const fsEntry  = hasCubemap ? 'fs_water'      : 'fs_water_flat';

    this.waterRenderPipe = await d.createRenderPipelineAsync({
      layout: 'auto',
      vertex: {
        module:     waterRenderMod,
        entryPoint: vsEntry,
        buffers: [{
          arrayStride: 8,
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
        }],
      },
      fragment: {
        module:     waterRenderMod,
        entryPoint: fsEntry,
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

    // ── Particle update compute ───────────────────────────────────────────────
    const pUpdateMod = d.createShaderModule({ code: WGSL_PARTICLE_UPDATE });
    this.particleUpdatePipe = await d.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: pUpdateMod, entryPoint: 'cs_particle_update' },
    });

    // ── Particle render ───────────────────────────────────────────────────────
    const pRenderMod = d.createShaderModule({ code: WGSL_PARTICLE_RENDER });
    this.particleRenderPipe = await d.createRenderPipelineAsync({
      layout: 'auto',
      vertex: {
        module:     pRenderMod,
        entryPoint: 'vs_particle',
      },
      fragment: {
        module:     pRenderMod,
        entryPoint: 'fs_particle',
        targets: [{
          format: this.format,
          blend: {
            // Additive blend for glowing particles
            color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one', operation: 'add' },
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

  // ─── Private: uniform writes ─────────────────────────────────────────────────

  private _writeUniforms(time: number): void {
    const data = new Float32Array(16);
    const [lx, ly, lz] = this.cfg.lightDir;
    // Row 0: lightDir.xyz + time
    data[0] = lx; data[1] = ly; data[2] = lz; data[3] = time;
    // Row 1: waterColor.xyz + unused
    data[4] = this.cfg.waterColor[0]; data[5] = this.cfg.waterColor[1]; data[6] = this.cfg.waterColor[2]; data[7] = 0;
    // Row 2: skyColor.xyz + damping
    data[8] = this.cfg.skyColor[0]; data[9] = this.cfg.skyColor[1]; data[10] = this.cfg.skyColor[2]; data[11] = WAVE_DAMPING;
    // Row 3: particleColor.xyz + reserved
    data[12] = this.cfg.particleColor[0]; data[13] = this.cfg.particleColor[1]; data[14] = this.cfg.particleColor[2]; data[15] = 0;
    this.device.queue.writeBuffer(this.uniformBuf, 0, data);
  }

  // ─── Private: dispatch helpers ───────────────────────────────────────────────

  private _dispatchDrop(
    enc: GPUCommandEncoder,
    drop: { x: number; y: number; radius: number; strength: number },
  ): void {
    // Write drop params
    const dropData = new Float32Array([drop.x, drop.y, drop.radius, drop.strength]);
    this.device.queue.writeBuffer(this.dropBuf, 0, dropData);

    const bg = this.device.createBindGroup({
      layout: this.dropPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: { buffer: this.dropBuf } },
        { binding: 2, resource: this.waterA.createView() },
        { binding: 3, resource: this.waterB.createView() },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.dropPipeline);
    pass.setBindGroup(0, bg);
    const wg = Math.ceil(this.simSize / 8);
    pass.dispatchWorkgroups(wg, wg);
    pass.end();

    // Swap: waterB now has the drop applied → promote to waterA
    [this.waterA, this.waterB] = [this.waterB, this.waterA];
  }

  private _dispatchStep(enc: GPUCommandEncoder): void {
    const bg = this.device.createBindGroup({
      layout: this.stepPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.waterA.createView() },
        { binding: 2, resource: this.waterB.createView() },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.stepPipeline);
    pass.setBindGroup(0, bg);
    const wg = Math.ceil(this.simSize / 8);
    pass.dispatchWorkgroups(wg, wg);
    pass.end();
    [this.waterA, this.waterB] = [this.waterB, this.waterA];
  }

  private _dispatchNormal(enc: GPUCommandEncoder): void {
    const bg = this.device.createBindGroup({
      layout: this.normalPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.waterA.createView() },
        { binding: 2, resource: this.waterB.createView() },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.normalPipeline);
    pass.setBindGroup(0, bg);
    const wg = Math.ceil(this.simSize / 8);
    pass.dispatchWorkgroups(wg, wg);
    pass.end();
    [this.waterA, this.waterB] = [this.waterB, this.waterA];
  }

  private _dispatchParticleUpdate(enc: GPUCommandEncoder): void {
    const bg = this.device.createBindGroup({
      layout: this.particleUpdatePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: this.waterA.createView() },
        { binding: 3, resource: this.particleA.createView() },
        { binding: 4, resource: this.particleB.createView() },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.particleUpdatePipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(this.maxParticles / WG));
    pass.end();
    [this.particleA, this.particleB] = [this.particleB, this.particleA];
  }

  // ─── Private: per-frame bind group factories ─────────────────────────────────

  private _makeWaterRenderBG0(sceneUniformBuf: GPUBuffer): GPUBindGroup {
    const hasCubemap = !!this.cfg.skyCubemap;
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.uniformBuf } },
      { binding: 1, resource: this.sampler },
      { binding: 2, resource: this.waterA.createView() },
    ];
    if (hasCubemap) {
      entries.push(
        { binding: 3, resource: this.sampler },
        { binding: 4, resource: this.cfg.skyCubemap!.createView({ dimension: 'cube' }) },
      );
    }
    return this.device.createBindGroup({
      layout: this.waterRenderPipe.getBindGroupLayout(0),
      entries,
    });
  }

  private _makeSceneBG1(sceneUniformBuf: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.waterRenderPipe.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: sceneUniformBuf } }],
    });
  }

  private _makeParticleRenderBG0(): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.particleRenderPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: this.particleA.createView() },
      ],
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factory
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create and build an ATWaterSurface in one call.
 *
 * ```ts
 * const water = await createATWaterSurface(device, 'bgra8unorm', {
 *   simSize:      256,
 *   meshDetail:   128,
 *   maxParticles: 4096,
 *   waterColor:   [0.25, 1.0, 1.25],
 * });
 * ```
 */
export async function createATWaterSurface(
  device: GPUDevice,
  format: GPUTextureFormat,
  cfg?:   ATWaterSurfaceConfig,
): Promise<ATWaterSurface> {
  const surface = new ATWaterSurface(device, format, cfg);
  await surface.build();
  return surface;
}
