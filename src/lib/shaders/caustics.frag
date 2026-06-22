#version 300 es
// ── caustics.frag ─────────────────────────────────────────────────────────────
// Water caustic light-pattern renderer.
//
// Technique:
//   Caustics are the bright, reticulated patterns cast on a pool floor when
//   sunlight refracts through a moving water surface.  We approximate them
//   with a two-pass analytic method that avoids the full ray-marching cost:
//
//   1. SURFACE HEIGHT FIELD — Four independent sine-wave octaves are summed to
//      build a time-evolving, turbulent water height h(uv,t).  Each octave
//      carries its own direction vector, frequency, amplitude, and phase speed
//      so the interference pattern never repeats visually.
//
//   2. SURFACE NORMAL — The 2-D gradient ∇h is estimated analytically (closed-
//      form derivative of the sine sum) to give a cheap surface normal N used
//      for the refraction step.  No texture lookups needed.
//
//   3. REFRACTION DISPLACEMENT — Snell's law in the thin-lens approximation:
//        d = (n_air / n_water) · dot(N, L̂)
//      displaces the floor UV by the refracted light direction projected onto
//      the XZ plane scaled by uWaterDepth.  The classical webgl-water caustic
//      trick (upstream/webgl-water/renderer.js:59) is used to advance the
//      sample point along the refracted ray to the pool bottom.
//
//   4. CAUSTIC INTENSITY — The Jacobian of the UV displacement map approximates
//      photon convergence.  Bright caustics form where light rays converge
//      (positive divergence of the refracted ray bundle).  We compute a 5-tap
//      finite-difference Jacobian (central differences in x and y) and map
//      it to luminance via a power curve.
//
//   5. MULTI-LAYER BLEND — Three independently displaced Jacobian samples at
//      different depth planes are blended to mimic volumetric caustic depth.
//      A fourth, large-scale low-frequency pass provides the broad illumination
//      envelope, preventing the pattern from becoming too busy.
//
//   6. FLOOR TINT + FRESNEL EDGE — The caustic intensity modulates a configurable
//      floor colour.  A Fresnel-style edge darkening is applied at the quad
//      boundary to frame the effect.  Premultiplied alpha is emitted.
//
// Inlined LYGIA-style helpers (self-contained, no external includes required):
//   mod289 / permute / taylorInvSqrt — from lygia/math/*.glsl
//   saturate                          — from lygia/math/saturate.glsl
//
// Uniforms:
//   uTime             — elapsed seconds                            float
//   uTexelSize        — 1 / render-target resolution              vec2
//   uWaterDepth       — virtual pool depth (refraction scale)     float  ≈ 0.5
//   uIOR              — water index of refraction                 float  ≈ 1.333
//   uWaveAmp          — master wave amplitude                     float  ≈ 0.08
//   uWaveSpeed        — master wave phase velocity                float  ≈ 0.4
//   uCausticSharpness — Jacobian power curve exponent             float  ≈ 2.5
//   uCausticBrightness— output luminance scale                    float  ≈ 3.0
//   uFloorColor       — RGB floor tint                            vec3
//   uCausticColor     — RGB caustic highlight tint                vec3
//   uLightDir         — normalised world-space light direction     vec3
//   uFresnelEdge      — edge darkening half-width [0,1]           float  ≈ 0.12
//
// Varyings (from fullscreen-quad vertex shader):
//   vUV               — [0,1]² texture coordinates
//
// Output:
//   fragColor         — premultiplied RGBA
//
// Research: xiaodi #M605 — cell-pubsub-loop
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── Varyings ──────────────────────────────────────────────────────────────────

in  vec2  vUV;

out vec4  fragColor;

// ── Uniforms ──────────────────────────────────────────────────────────────────

uniform float     uTime;
uniform vec2      uTexelSize;

// Refraction / geometry
uniform float     uWaterDepth;        // e.g. 0.5
uniform float     uIOR;               // e.g. 1.333  (air→water)

// Wave synthesis
uniform float     uWaveAmp;           // e.g. 0.08
uniform float     uWaveSpeed;         // e.g. 0.40

// Caustic shading
uniform float     uCausticSharpness;  // e.g. 2.5
uniform float     uCausticBrightness; // e.g. 3.0

// Colour
uniform vec3      uFloorColor;        // e.g. vec3(0.06, 0.18, 0.32)
uniform vec3      uCausticColor;      // e.g. vec3(0.75, 0.92, 1.00)
uniform vec3      uLightDir;          // e.g. normalize(vec3(0.3, 1.0, 0.5))
uniform float     uFresnelEdge;       // e.g. 0.12

// ─────────────────────────────────────────────────────────────────────────────
// ── Inlined LYGIA math helpers ────────────────────────────────────────────────
// Source: upstream/lygia/math/saturate.glsl, mod289.glsl, permute.glsl,
//         taylorInvSqrt.glsl
// License: Patricio Gonzalez Vivo — Prosperity / Patron License
// ─────────────────────────────────────────────────────────────────────────────

#if !defined(FNC_SATURATE) && !defined(saturate)
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

#ifndef FNC_MOD289
#define FNC_MOD289
float mod289(const in float x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2  mod289(const in vec2  x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3  mod289(const in vec3  x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4  mod289(const in vec4  x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
#endif

#ifndef FNC_PERMUTE
#define FNC_PERMUTE
float permute(const in float v) { return mod289(((v * 34.0) + 1.0) * v); }
vec3  permute(const in vec3  v) { return mod289(((v * 34.0) + 1.0) * v); }
vec4  permute(const in vec4  v) { return mod289(((v * 34.0) + 1.0) * v); }
#endif

#ifndef FNC_TAYLORINVSQRT
#define FNC_TAYLORINVSQRT
float taylorInvSqrt(in float r) { return 1.79284291400159 - 0.85373472095314 * r; }
vec4  taylorInvSqrt(in vec4  r) { return 1.79284291400159 - 0.85373472095314 * r; }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── Wave octave descriptors ───────────────────────────────────────────────────
// Four independent sine-wave octaves, each with a unique:
//   dir   — normalised propagation direction
//   freq  — spatial frequency (radians / uv-unit)
//   amp   — amplitude relative to uWaveAmp
//   speed — phase velocity relative to uWaveSpeed
//   phase — static phase offset (breaks symmetry)
// ─────────────────────────────────────────────────────────────────────────────

#define NUM_OCTAVES 4

// Packing: vec4(dir.x, dir.y, freq_scale, amp_scale)
const vec4 WAVE_DA[NUM_OCTAVES] = vec4[NUM_OCTAVES](
    vec4( 0.97, 0.24, 1.00, 1.00),   // octave 0 — primary swell, NE
    vec4(-0.50, 0.87, 2.13, 0.55),   // octave 1 — secondary, NW
    vec4( 0.21,-0.98, 3.97, 0.28),   // octave 2 — high-freq capillary, S
    vec4(-0.83,-0.56, 6.71, 0.14)    // octave 3 — micro-ripple, SW
);

// Packing: vec2(speed_scale, phase_offset)
const vec2 WAVE_SP[NUM_OCTAVES] = vec2[NUM_OCTAVES](
    vec2(1.00, 0.000),
    vec2(1.32, 1.047),
    vec2(0.81, 2.094),
    vec2(1.61, 3.665)
);

// ─────────────────────────────────────────────────────────────────────────────
// ── waveHeight — sum all octaves to get surface height h(p) ─────────────────
// Returns the scalar water height at position p ∈ ℝ².
// ─────────────────────────────────────────────────────────────────────────────

float waveHeight(vec2 p) {
    float h = 0.0;
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        vec2  dir   = WAVE_DA[i].xy;
        float freq  = WAVE_DA[i].z;
        float amp   = WAVE_DA[i].w * uWaveAmp;
        float speed = WAVE_SP[i].x * uWaveSpeed;
        float phase = WAVE_SP[i].y;

        float arg = dot(dir, p) * freq - uTime * speed + phase;
        h += amp * sin(arg);
    }
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// ── waveNormal — analytic XZ gradient → surface normal ───────────────────────
// Returns the unnormalised surface normal N = (-dh/dx, 1, -dh/dz) in
// view-aligned coords where Y is "up" (out of the water surface).
// ─────────────────────────────────────────────────────────────────────────────

vec3 waveNormal(vec2 p) {
    // Analytic partial derivatives of the sine sum.
    float dhdx = 0.0;
    float dhdy = 0.0;
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        vec2  dir   = WAVE_DA[i].xy;
        float freq  = WAVE_DA[i].z;
        float amp   = WAVE_DA[i].w * uWaveAmp;
        float speed = WAVE_SP[i].x * uWaveSpeed;
        float phase = WAVE_SP[i].y;

        float arg  = dot(dir, p) * freq - uTime * speed + phase;
        float darg = amp * freq * cos(arg); // d/dp of sin(arg)

        dhdx += darg * dir.x;
        dhdy += darg * dir.y;
    }
    // N points away from the surface (into air), unnormalised.
    return vec3(-dhdx, 1.0, -dhdy);
}

// ─────────────────────────────────────────────────────────────────────────────
// ── refractedFloorUV — project refracted ray to virtual pool floor ────────────
// Implements the thin-lens / flat-surface Snell projection used in
// upstream/webgl-water/renderer.js (causticsShader, vertex stage lines ~247-252).
//
// Given:
//   uv        — surface UV ∈ [0,1]²
//   N         — normalised surface normal (pointing up)
//   lightDir  — normalised incoming light direction (pointing toward surface)
//   depth     — virtual pool depth in UV units
//   ior       — n_air / n_water  (≈ 0.75 for air→water)
//
// Returns the displaced floor UV where the refracted ray hits.
// ─────────────────────────────────────────────────────────────────────────────

vec2 refractedFloorUV(vec2 uv, vec3 N, vec3 lightDir, float depth, float ior) {
    // GLSL built-in refract: refract(I, N, eta) uses N pointing away from
    // the interface surface (into the medium the ray is entering from).
    // Our N points upward (into air), so we negate lightDir for the incidence
    // vector I (pointing from air toward the surface).
    vec3 I        = normalize(-lightDir);
    vec3 refRay   = refract(I, N, ior);          // direction in water

    // Project the refracted ray to the pool floor.
    // Flat-surface approximation: the floor sits at y = -depth in normalised
    // UV space.  Parametric hit: origin + t*refRay where origin.y = 0, so
    //   t = depth / max(refRay.y, 1e-4)   (refRay.y should be negative → down)
    // Here we drive depth along the xz displacement only.
    float t = depth / max(abs(refRay.y), 1e-4);
    vec2  displacement = refRay.xz * t;
    return uv + displacement;
}

// ─────────────────────────────────────────────────────────────────────────────
// ── causticJacobian — finite-difference photon convergence map ────────────────
// Computes the Jacobian determinant of the floor UV mapping at a given surface
// UV.  Where light converges (det > 1) we get bright caustics; where it
// diverges (det < 1) we get shadows.
//
// A 5-tap central-difference stencil is used for numerical stability.
// The Jacobian det = |∂(fUV)/∂(uv)| approximates the local photon density.
// ─────────────────────────────────────────────────────────────────────────────

float causticJacobian(vec2 uv, vec3 lightDir, float depth, float ior, float eps) {
    // ±1-texel offsets for the stencil
    vec2 ex = vec2(eps, 0.0);
    vec2 ey = vec2(0.0, eps);

    // Evaluate displaced floor UV at the four stencil points.
    // We need the wave normal at each offset sample too.
    vec3 Nx = normalize(waveNormal(uv + ex));
    vec3 Px = refractedFloorUV(uv + ex, Nx, lightDir, depth, ior);
    vec3 Nx2= normalize(waveNormal(uv - ex));
    vec3 Px2= refractedFloorUV(uv - ex, Nx2, lightDir, depth, ior);

    vec3 Ny = normalize(waveNormal(uv + ey));
    vec3 Py = refractedFloorUV(uv + ey, Ny, lightDir, depth, ior);
    vec3 Ny2= normalize(waveNormal(uv - ey));
    vec3 Py2= refractedFloorUV(uv - ey, Ny2, lightDir, depth, ior);

    // 2D Jacobian columns via central differences.
    vec2 dFdx = (Px - Px2) / (2.0 * eps);
    vec2 dFdy = (Py - Py2) / (2.0 * eps);

    // Determinant: dFdx.x * dFdy.y - dFdx.y * dFdy.x
    // A value > 1 means photons converge → bright spot.
    float det = dFdx.x * dFdy.y - dFdx.y * dFdy.x;

    // Clamp to [0, ∞); shadows are handled separately by dimming the floor.
    return max(0.0, det);
}

// ─────────────────────────────────────────────────────────────────────────────
// ── singleCausticLayer — full caustic intensity for one depth plane ───────────
// ─────────────────────────────────────────────────────────────────────────────

float singleCausticLayer(vec2 uv, vec3 lightDir, float depth, float ior) {
    // Stencil step — 2 texels for reasonable finite-difference accuracy.
    float eps = uTexelSize.x * 2.0;

    float raw = causticJacobian(uv, lightDir, depth, ior, eps);

    // Map Jacobian to [0,1] caustic intensity via power curve.
    // Values just above 1.0 correspond to mild convergence; we want them
    // visually bright, so we subtract 1 (neutral) and apply a power.
    float intensity = saturate(pow(max(raw - 0.5, 0.0) * 0.8, uCausticSharpness));
    return intensity;
}

// ─────────────────────────────────────────────────────────────────────────────
// ── fresnelEdge — vignette / edge fade helper ─────────────────────────────────
// Darkens the quad at its UV boundary to frame the caustic pool.
// ─────────────────────────────────────────────────────────────────────────────

float fresnelEdge(vec2 uv, float halfWidth) {
    vec2 edge = smoothstep(0.0, halfWidth, uv) *
                smoothstep(0.0, halfWidth, 1.0 - uv);
    return edge.x * edge.y;
}

// ─────────────────────────────────────────────────────────────────────────────
// ── main ──────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

void main() {

    // ── 1. Normalised light direction (user-supplied or default) ──────────────
    vec3 L = normalize(uLightDir);

    // ── 2. Surface normal at this UV ─────────────────────────────────────────
    vec3 N = normalize(waveNormal(vUV));

    // ── 3. Snell's law ratio for air → water ─────────────────────────────────
    // eta = n_air / n_water  (≈ 1.0 / 1.333 ≈ 0.7502)
    float eta = 1.0 / max(uIOR, 1.0);

    // ── 4. Multi-layer caustic accumulation ───────────────────────────────────
    // Layer 0 — shallow (strong, sharp, fast-moving)
    float c0 = singleCausticLayer(vUV, L, uWaterDepth * 0.60, eta);

    // Layer 1 — mid depth (moderate, slightly blurred by using larger stencil)
    float c1 = singleCausticLayer(vUV, L, uWaterDepth * 1.00, eta);

    // Layer 2 — deep (dim, broad, gives volumetric depth impression)
    float c2 = singleCausticLayer(vUV, L, uWaterDepth * 1.55, eta);

    // Layer 3 — large-scale low-frequency envelope (prevents busy appearance)
    // Evaluated at a down-sampled UV with a single sine to get the broad swell.
    float envelope = 0.5 + 0.5 * sin(
        dot(vUV - 0.5, vec2(1.3, 0.7)) * 3.1 - uTime * uWaveSpeed * 0.25
    );

    // Weighted blend: sharper shallower layers dominate, envelope softens.
    float caustic = c0 * 0.50
                  + c1 * 0.30
                  + c2 * 0.12
                  + envelope * 0.08;

    // Master brightness
    caustic = saturate(caustic * uCausticBrightness);

    // ── 5. Diffuse floor lighting (Lambertian from above) ─────────────────────
    // N is the water surface normal; for the floor, use the un-displaced floor
    // normal (0,1,0) with the refracted light direction.
    vec3  refRay        = refract(normalize(-L), N, eta);
    float floorDiffuse  = max(0.0, -refRay.y);    // dot with floor normal (0,-1,0) flipped

    // ── 6. Floor base colour ──────────────────────────────────────────────────
    vec3 floorCol = uFloorColor * (0.35 + 0.65 * floorDiffuse);

    // ── 7. Add caustic highlight ──────────────────────────────────────────────
    // Caustic bright spots are tinted with uCausticColor (typically pale cyan).
    vec3 litColor = floorCol + uCausticColor * caustic;

    // ── 8. Specular glint on the water surface itself ─────────────────────────
    // A very thin Blinn-Phong specular pass mimics sun-glint at grazing angles.
    vec3  viewDir  = vec3(0.0, 0.0, 1.0);          // orthographic top-down view
    vec3  halfVec  = normalize(L + viewDir);
    float specular = pow(max(dot(N, halfVec), 0.0), 96.0) * 0.45;
    litColor += uCausticColor * specular;

    // ── 9. Fresnel-style edge vignette ────────────────────────────────────────
    float edge = fresnelEdge(vUV, uFresnelEdge);
    litColor  *= edge;

    // ── 10. Alpha — fully opaque pool floor with edge fade ────────────────────
    float alpha = edge;

    // ── 11. Pre-multiplied RGBA output ────────────────────────────────────────
    fragColor = vec4(litColor * alpha, alpha);
}
