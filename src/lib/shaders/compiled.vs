{@}caustics.frag{@}#version 300 es
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
{@}cil-arrow-right.frag{@}precision mediump float;

uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_arrowWidth;  // stroke thickness [0..1]
uniform float u_time;

// ── lygia/math/saturate.glsl (inlined) ──────────────────────────────────────
#ifndef FNC_SATURATE
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

// ── lygia/sdf/lineSDF.glsl (inlined) ────────────────────────────────────────
// contributors: Inigo Quiles
// Segment SDF: returns the unsigned distance from point st to segment [a,b].
#ifndef FNC_LINESDF
#define FNC_LINESDF
float lineSDF(in vec2 st, in vec2 a, in vec2 b) {
    vec2 b_to_a = b - a;
    vec2 to_a   = st - a;
    float h = saturate(dot(to_a, b_to_a) / dot(b_to_a, b_to_a));
    return length(to_a - h * b_to_a);
}
#endif
// ── end lygia lineSDF ────────────────────────────────────────────────────────

// SDF chevron / arrow-right glyph pointing +X, centered at origin, scale ~1.
// Uses lygia lineSDF for both diagonal strokes.
float sdArrowRight(vec2 p, float w) {
    vec2 a1 = vec2(-0.45,  0.40);
    vec2 b1 = vec2( 0.45,  0.0 );
    vec2 a2 = vec2(-0.45, -0.40);
    vec2 b2 = vec2( 0.45,  0.0 );

    float d1 = lineSDF(p, a1, b1);
    float d2 = lineSDF(p, a2, b2);

    return min(d1, d2) - w;
}

void main() {
  vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;

  float cols   = 3.0;
  float rows   = 3.0;
  vec2  scroll = vec2(u_time * 0.25, 0.0);

  vec2  tiled  = fract(uv * vec2(cols, rows) + scroll);
  vec2  lp     = tiled * 2.0 - 1.0;   // [-1,1]

  float d    = sdArrowRight(lp, u_arrowWidth * 0.5);
  float mask = smoothstep(0.02, -0.01, d);

  float fade  = smoothstep(0.0, 0.6, tiled.x);
  float alpha = mask * (0.4 + 0.6 * fade);

  gl_FragColor = vec4(u_fillColor, clamp(alpha, 0.0, 1.0) * u_opacity);
}
{@}cil-bolt.frag{@}precision mediump float;

uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_zigzagCount;
uniform float u_amplitude;
uniform float u_time;

// ---- AT UIL params (from channels/physics/xiaodi_options_table.json / cil-bolt) ----
// INPUT_HydraBloom_Bloom_Intensity        : 1.0
// INPUT_HydraBloom_Bloom_Radius           : 1.0
// UnrealBloomComposite globalbloom/bloomStrength : 0.3
// UnrealBloomComposite globalbloom/bloomRadius   : 0.2
// UnrealBloomComposite homebloom/bloomStrength   : 0.6
// UnrealBloomComposite homebloom/bloomRadius     : 0.8
// L_Element_10_home_sceneintensity        : 2.19
// INPUT_Element_1_work_scenewiggle_speed  : 0.7
// BloomLuminosity luminosityThreshold     : 0.0
// CAMERA_Element_1_HomelerpSpeed          : 0.1

const float AT_BLOOM_INTENSITY        = 1.0;
const float AT_BLOOM_RADIUS           = 1.0;
const float AT_GLOBAL_BLOOM_STRENGTH  = 0.3;
const float AT_GLOBAL_BLOOM_RADIUS    = 0.2;
const float AT_HOME_BLOOM_STRENGTH    = 0.6;
const float AT_HOME_BLOOM_RADIUS      = 0.8;
const float AT_LIGHT_INTENSITY        = 2.19;
const float AT_WIGGLE_SPEED           = 0.7;
const float AT_LUMINOSITY_THRESHOLD   = 0.0;
const float AT_LERP_SPEED             = 0.1;

// ── lygia/math/saturate.glsl (inlined) ──────────────────────────────────────
#ifndef FNC_SATURATE
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

// ── lygia/sdf/lineSDF.glsl (inlined) ────────────────────────────────────────
// contributors: Inigo Quiles
// Segment SDF: returns the unsigned distance from point st to segment [a,b].
#ifndef FNC_LINESDF
#define FNC_LINESDF
float lineSDF(in vec2 st, in vec2 a, in vec2 b) {
    vec2 b_to_a = b - a;
    vec2 to_a   = st - a;
    float h = saturate(dot(to_a, b_to_a) / dot(b_to_a, b_to_a));
    return length(to_a - h * b_to_a);
}
#endif
// ── end lygia lineSDF ────────────────────────────────────────────────────────

// Stroke mask derived from lineSDF (replaces hand-written seg helper)
float strokeMask(vec2 p, vec2 a, vec2 b, float w) {
    float d = lineSDF(p, a, b);
    return smoothstep(w, w * 0.4, d);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;
  vec2 p  = uv * 2.0 - 1.0;          // [-1,1]

  float strokeW = 0.045;
  float total   = 0.0;

  float steps = u_zigzagCount;
  float dy    = 2.0 / steps;

  // Animated phase offset — speed driven by AT_WIGGLE_SPEED
  float phase = sin(u_time * 2.5 * AT_WIGGLE_SPEED) * 0.15;

  // Core stroke — zigzag segments via lineSDF
  for (float i = 0.0; i < 32.0; i++) {
    if (i >= steps) break;

    float t0    = -1.0 + i       * dy;
    float t1    = -1.0 + (i+1.0) * dy;
    float side0 = (mod(i,       2.0) < 1.0 ? 1.0 : -1.0);
    float side1 = (mod(i + 1.0, 2.0) < 1.0 ? 1.0 : -1.0);

    vec2 a = vec2(side0 * u_amplitude + phase, t0);
    vec2 b = vec2(side1 * u_amplitude + phase, t1);

    total = max(total, strokeMask(p, a, b, strokeW));
  }

  // Global bloom pass
  float glowGlobal  = 0.0;
  float globalGlowW = strokeW * (3.5 * AT_GLOBAL_BLOOM_RADIUS / AT_BLOOM_RADIUS);
  for (float i = 0.0; i < 32.0; i++) {
    if (i >= steps) break;
    float t0   = -1.0 + i * dy;
    float t1   = -1.0 + (i+1.0) * dy;
    float s0   = (mod(i,       2.0) < 1.0 ? 1.0 : -1.0);
    float s1   = (mod(i + 1.0, 2.0) < 1.0 ? 1.0 : -1.0);
    vec2 a     = vec2(s0 * u_amplitude + phase, t0);
    vec2 b     = vec2(s1 * u_amplitude + phase, t1);
    glowGlobal = max(glowGlobal, strokeMask(p, a, b, globalGlowW) * AT_GLOBAL_BLOOM_STRENGTH);
  }

  // Home bloom pass
  float glowHome  = 0.0;
  float homeGlowW = strokeW * (5.0 * AT_HOME_BLOOM_RADIUS / AT_BLOOM_RADIUS);
  for (float i = 0.0; i < 32.0; i++) {
    if (i >= steps) break;
    float t0   = -1.0 + i * dy;
    float t1   = -1.0 + (i+1.0) * dy;
    float s0   = (mod(i,       2.0) < 1.0 ? 1.0 : -1.0);
    float s1   = (mod(i + 1.0, 2.0) < 1.0 ? 1.0 : -1.0);
    vec2 a     = vec2(s0 * u_amplitude + phase, t0);
    vec2 b     = vec2(s1 * u_amplitude + phase, t1);
    glowHome   = max(glowHome, strokeMask(p, a, b, homeGlowW) * AT_HOME_BLOOM_STRENGTH);
  }

  float lum     = dot(u_fillColor, vec3(0.2126, 0.7152, 0.0722));
  float lumGate = step(AT_LUMINOSITY_THRESHOLD, lum);

  float bloomSum = (glowGlobal + glowHome) * lumGate * AT_BLOOM_INTENSITY * (AT_LIGHT_INTENSITY / 2.19);
  float alpha    = clamp(total + bloomSum, 0.0, 1.0);

  gl_FragColor = vec4(u_fillColor, alpha * u_opacity);
}
{@}cil-eye.frag{@}precision mediump float;

// ─── AT UIL params applied from channels/physics/xiaodi_options_table.json [\"cil-eye\"] ───
//
// bloom / glow (UnrealBloomComposite / homebloom):
//   UnrealBloomComposite/UnrealBloomComposite/homebloom/bloomStrength  = 1.2
//   UnrealBloomComposite/UnrealBloomComposite/homebloom/bloomRadius    = 1.0
//   UnrealBloomComposite/UnrealBloomComposite/homebloom/bloomTintColor = #ffffff
//   UnrealBloomComposite_shaderVariants_homebloomStrength              = 0.6
//   UnrealBloomComposite_shaderVariants_homebloomRadius                = 0.8
//   UnrealBloomLuminosity/UnrealBloomLuminosity/homebloom/luminosityThreshold = 0.0
//
// lighting (HomeAlleyShader / L_Element_11_home_scene):
//   HomeAlleyShader uLight      = [2.61, 0.29, 0.57, 0.0]
//   HomeAlleyShader uPhong      = [1.82, 0.71]
//   HomeAlleyShader uPhongColor = #d600ff
//   L_Element_11_home_scene intensity = 3.44
//   L_Element_11_home_scene color     = #0bed90
//   L_Element_10_home_scene intensity = 2.19
//   VolumetricLight_home fExposure    = 0.86   fDensity = 0.22
//
// shadow (SHADOW_Element_9_home_scene):
//   SHADOW_Element_9_home_scene far    = 40
//   SHADOW_Element_9_home_scene size   = 1024
//   SHADOW_Element_9_home_scene static = true
//   SHADOW_Element_9_home_scene position = [0, 6.51, 0]
//   SHADOW_Element_9_home_scene target   = [0, 0, 0]
// ──────────────────────────────────────────────────────────────────────────────────────────

uniform vec4  u_bbox;         // x, y, width, height in canvas coords
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_numRays;
uniform float u_pupilRadius;
uniform float u_focalIntensity;
uniform float u_time;

// AT bloom uniforms (defaults from UIL cil-eye entry)
uniform float u_bloomStrength;   // default 1.2  (homebloom/bloomStrength)
uniform float u_bloomRadius;     // default 1.0  (homebloom/bloomRadius)

// AT ambient light uniforms (from L_Element_11_home_scene + VolumetricLight)
uniform float u_ambientIntensity; // default 3.44 (L_Element_11 intensity)
uniform vec3  u_ambientColor;     // default #0bed90 → (0.047, 0.929, 0.565)
uniform float u_lightExposure;    // default 0.86  (VolumetricLight fExposure)

// AT shadow uniforms (from SHADOW_Element_9_home_scene)
uniform float u_shadowFar;     // default 40.0
uniform float u_shadowBias;    // default 0.001 (derived from shadow size 1024)

// ── lygia/sdf/circleSDF.glsl (inlined) ──────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns a circle-shaped SDF.  circleSDF(vec2 st) → distance in [0,1] space.
// Centered at 0.5; result * 2 == diameter-normalised distance.
#ifndef FNC_CIRCLESDF
#define FNC_CIRCLESDF
float circleSDF(in vec2 v) {
    v -= 0.5;
    return length(v) * 2.0;
}
#endif
// ── end lygia circleSDF ──────────────────────────────────────────────────────

void main() {
  // Normalize fragment to bbox-local UV [0,1]
  vec2 fragCoord = gl_FragCoord.xy;
  vec2 uv = (fragCoord - u_bbox.xy) / u_bbox.zw;

  // circleSDF works in [0,1] UV space; result is 0 at centre, 1 at edge of unit circle.
  // We map it to [-1,1] range for the legacy distance variable used below.
  float dist = circleSDF(uv);   // [0, ~1.41]

  // Angle still computed from centred coordinates
  vec2 p = uv * 2.0 - 1.0;
  float angle = atan(p.y, p.x);

  // --- Pupil (using circleSDF result) ---
  // u_pupilRadius is expressed in the original [-1,1] space → convert to circleSDF scale (*0.5)
  float pupilR = u_pupilRadius * 0.5;
  float pupil = 1.0 - smoothstep(pupilR - 0.01, pupilR + 0.01, dist);

  // --- Iris ring ---
  float irisInner = (u_pupilRadius + 0.02) * 0.5;
  float irisOuter = (u_pupilRadius + 0.08) * 0.5;
  float iris = smoothstep(irisInner, irisOuter, dist)
             * (1.0 - smoothstep(0.425, 0.5, dist));

  // --- Radial rays ---
  float halfStep = 3.14159265 / u_numRays;
  float rayAngle = mod(angle + u_time * 0.3, halfStep * 2.0) - halfStep;
  float rayMask  = smoothstep(0.07, 0.0, abs(rayAngle));
  float rayFade  = smoothstep(0.5, irisInner + 0.06, dist)
                 * smoothstep(pupilR, pupilR + 0.06, dist);
  float rays     = rayMask * rayFade * u_focalIntensity;

  // --- Sclera (outer white ellipse halo) via circleSDF ---
  float sclera = smoothstep(0.525, 0.44, dist);

  // --- AT ambient lighting (L_Element_11 + VolumetricLight) ---
  float ambientFalloff = 1.0 - smoothstep(0.0, 0.6, dist);
  vec3  ambientContrib = u_ambientColor * u_ambientIntensity * u_lightExposure * ambientFalloff;

  // --- AT bloom glow ring ---
  float bloomCenter = (u_pupilRadius + 0.15) * 0.5;
  float bloomRing = exp(-pow((dist - bloomCenter) / max(u_bloomRadius * 0.09, 0.005), 2.0));
  float bloom     = bloomRing * u_bloomStrength * 0.35;

  // --- AT shadow attenuation ---
  float shadowNorm   = clamp(dist / (u_shadowFar * 0.0125), 0.0, 1.0);
  float shadowFactor = 1.0 - shadowNorm * (1.0 - u_shadowBias * 100.0);

  float alpha = clamp(sclera * (iris + rays) + pupil, 0.0, 1.0);

  vec3 finalColor = u_fillColor + ambientContrib * (iris + bloom) * alpha;
  finalColor += u_fillColor * bloom;
  finalColor *= shadowFactor;

  gl_FragColor = vec4(finalColor, alpha * u_opacity);
}
{@}cil-plus.frag{@}precision mediump float;

uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_armLength;    // half-length of each arm  [0..1]
uniform float u_strokeWidth;  // half-width of stroke     [0..1]

// ── lygia/sdf/rectSDF.glsl (inlined) ────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns a rectangular SDF in [-1,1] centred space (max-norm variant).
#ifndef FNC_RECTSDF
#define FNC_RECTSDF
float rectSDF(in vec2 st, in vec2 s) {
    vec2 p = st * 2.0 - 1.0;   // remap [0,1]→[-1,1]; caller passes UV
    return max(abs(p.x / s.x), abs(p.y / s.y));
}
// Signed box SDF variant (used for the arm extrusions)
float sdBox2(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
#endif
// ── end lygia rectSDF ────────────────────────────────────────────────────────

// SDF for an axis-aligned plus centered at origin.
// Built from two overlapping sdBox2 calls — equivalent to the previous
// hand-written sdPlus but now uses the rectSDF family.
float sdPlus(vec2 p, float armLen, float sw) {
    float h = sdBox2(p, vec2(armLen, sw));
    float v = sdBox2(p, vec2(sw, armLen));
    return min(h, v);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;
  vec2 p  = uv * 2.0 - 1.0;   // [-1,1]

  float d    = sdPlus(p, u_armLength, u_strokeWidth);
  float mask = smoothstep(0.015, -0.015, d);

  float glow = smoothstep(0.08, 0.0, d) * 0.25;

  float alpha = clamp(mask + glow, 0.0, 1.0);
  gl_FragColor = vec4(u_fillColor, alpha * u_opacity);
}
{@}cil-vector.frag{@}precision mediump float;

uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_arrowCount;   // arrows per row/col
uniform float u_angleSpread;  // variation in radians

// ── lygia/math/const.glsl (inlined, PI/TAU only) ────────────────────────────
#ifndef PI
#define PI  3.1415926535897932384626433832795
#endif
#ifndef TAU
#define TAU 6.2831853071795864769252867665590
#endif

// ── lygia/sdf/polySDF.glsl (inlined) ────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns SDF for a regular V-sided polygon, UV [0,1] space centred at 0.5.
#ifndef FNC_POLYSDF
#define FNC_POLYSDF
float polySDF(in vec2 st, in int V) {
    st = st * 2.0 - 1.0;
    float a = atan(st.x, st.y) + PI;
    float r = length(st);
    float v = TAU / float(V);
    return cos(floor(0.5 + a / v) * v - a) * r;
}
#endif
// ── end lygia polySDF ────────────────────────────────────────────────────────

// ── lygia/math/saturate.glsl (inlined) ──────────────────────────────────────
#ifndef FNC_SATURATE
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

// ── lygia/sdf/lineSDF.glsl (inlined) ────────────────────────────────────────
#ifndef FNC_LINESDF
#define FNC_LINESDF
float lineSDF(in vec2 st, in vec2 a, in vec2 b) {
    vec2 b_to_a = b - a;
    vec2 to_a   = st - a;
    float h = saturate(dot(to_a, b_to_a) / dot(b_to_a, b_to_a));
    return length(to_a - h * b_to_a);
}
#endif
// ── end lygia lineSDF ────────────────────────────────────────────────────────

// Box SDF (not in lygia 2D but equivalent to rectSDF signed variant)
float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

// Arrow: shaft via sdBox + head via polySDF (triangle = 3-sided poly)
float drawArrow(vec2 p, float angle, float scale) {
    float c = cos(angle), s = sin(angle);
    vec2 lp = vec2(c * p.x + s * p.y, -s * p.x + c * p.y) / scale;

    // Shaft
    float shaft = sdBox(lp - vec2(-0.15, 0.0), vec2(0.22, 0.045));

    // Arrowhead — polySDF with 3 sides (equilateral triangle) in local UV space.
    // Map the head region to [0,1] UV for polySDF and test if inside.
    vec2 headUV = (lp - vec2(0.13, 0.0)) / 0.32 + 0.5;
    // Rotate so triangle points right (+X) in local space: offset angle by -PI/2
    headUV = headUV - 0.5;
    float tmp = headUV.x;
    headUV.x = -headUV.y;
    headUV.y =  tmp;
    headUV = headUV + 0.5;
    float head = polySDF(headUV, 3) * 0.32 - 0.16;

    float d = min(shaft, head);
    return smoothstep(0.01, -0.01, d);
}

// pseudo-random
float rand(vec2 co) {
    return fract(sin(dot(co, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;

    float n     = u_arrowCount;
    vec2  cell  = floor(uv * n);
    vec2  local = fract(uv * n) - 0.5;  // [-0.5, 0.5]

    float jitter = (rand(cell) * 2.0 - 1.0) * u_angleSpread;
    float angle  = jitter;

    float scale = 0.45;
    float mask  = drawArrow(local, angle, scale);

    gl_FragColor = vec4(u_fillColor, mask * u_opacity);
}
{@}cloud-fog.frag{@}#version 300 es
// ── cloud-fog.frag ────────────────────────────────────────────────────────────
// Volumetric fog fragment shader.
//
// Technique: each draw call renders one semi-transparent quad at a specific
// depth within the fog volume.  Stacking N quads (AT default: 20) with
// per-layer depth offsets and independent noise phases produces convincing
// volume-fog without ray-marching.
//
// AT CloudFog module reference parameters:
//   alpha=1.8, planes=20, noise=1, speed=0.7
//   width=[-4,4], height=[-1,4], depth=[-2,-2]
//   fadeDist=[2,4], cullDistance=999, scale=6
//
// Uniforms:
//   uAlpha       — master opacity scale (AT: 1.8, clamped per-layer)
//   uNoise       — noise intensity  (AT: 1.0; 0 = no noise, flat fog)
//   uSpeed       — animation speed  (AT: 0.7)
//   uScale       — noise domain scale (AT: 6; larger → coarser clumps)
//   uTime        — elapsed seconds
//   uLayerIndex  — which plane this is  (0 … uPlaneCount-1)
//   uPlaneCount  — total plane count    (AT: 20)
//   uFadeNear    — near fade distance   (AT: 2.0)
//   uFadeFar     — far  fade distance   (AT: 4.0)
//   uFogColor    — RGB fog tint         (default: white)
//
// Inputs from vertex shader:
//   vUV          — [0,1] across the quad
//   vDepth01     — remapped 0=back … 1=front within the volume
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── varyings ──────────────────────────────────────────────────────────────────
in vec2  vUV;
in float vDepth01;   // 0 = deepest layer, 1 = closest layer

// ── uniforms ──────────────────────────────────────────────────────────────────
// AT UIL CloudFog defaults (source: channels/physics/at_uil_params.json)
//   uAlpha      = 1.8        (INPUT_CloudFoghome_alpha)
//   uNoise      = 1.0        (INPUT_CloudFoghome_noise)
//   uSpeed      = 0.7        (INPUT_CloudFoghome_speed)
//   uScale      = 6.0        (INPUT_CloudFoghome_scale)
//   uPlaneCount = 20         (INPUT_CloudFoghome_planes)
//   uFadeNear   = 2.0        (INPUT_CloudFoghome_fadeDist[0])
//   uFadeFar    = 4.0        (INPUT_CloudFoghome_fadeDist[1])
//   width       = [-4, 4]    (INPUT_CloudFoghome_width)
//   height      = [-1, 4]    (INPUT_CloudFoghome_height)
uniform float uAlpha;      // AT default: 1.8
uniform float uNoise;      // AT default: 1.0
uniform float uSpeed;      // AT default: 0.7
uniform float uScale;      // AT default: 6.0
uniform float uTime;
uniform int   uLayerIndex;
uniform int   uPlaneCount; // AT default: 20
uniform float uFadeNear;   // AT default: 2.0  (fadeDist[0])
uniform float uFadeFar;    // AT default: 4.0  (fadeDist[1])
uniform vec3  uFogColor;

// ── output ────────────────────────────────────────────────────────────────────
out vec4 finalColor;

// ─────────────────────────────────────────────────────────────────────────────
// Simplex-style 2-D gradient noise.
//
// Based on Stefan Gustavson's public-domain simplex noise, simplified to a
// single-octave 2-D version for fragment shader use.
// Reference: https://github.com/stegu/webgl-noise (MIT / public domain)
// ─────────────────────────────────────────────────────────────────────────────

vec3 _mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 _mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 _permute(vec3 x) { return _mod289(((x * 34.0) + 10.0) * x); }

// Returns noise in [-1, 1].
float snoise2(vec2 v) {
    const vec4 C = vec4( 0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                         0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626,  // -1.0 + 2.0 * C.x
                         0.024390243902439); // 1.0 / 41.0

    // Skew input space to determine simplex cell
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v -   i + dot(i, C.xx);

    // Simplex corners
    vec2 i1  = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy  -= i1;

    // Gradient contributions
    i = _mod289(i);
    vec3 p = _permute(_permute(i.y + vec3(0.0, i1.y, 1.0)) +
                               i.x + vec3(0.0, i1.x, 1.0));

    vec3 m = max(0.5 - vec3(dot(x0, x0),
                            dot(x12.xy, x12.xy),
                            dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;

    vec3 x   = 2.0 * fract(p * C.www) - 1.0;
    vec3 h   = abs(x) - 0.5;
    vec3 ox  = floor(x + 0.5);
    vec3 a0  = x - ox;

    // Normalise gradients
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

    vec3 g;
    g.x  = a0.x  * x0.x    + h.x  * x0.y;
    g.yz = a0.yz * x12.xz  + h.yz * x12.yw;

    return 130.0 * dot(m, g);
}

// 2-octave fBm wrapper — returns [0, 1].
float fbm(vec2 p) {
    float n = snoise2(p)       * 0.6
            + snoise2(p * 2.1) * 0.4;
    return clamp(n * 0.5 + 0.5, 0.0, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
void main() {
    // ── Per-layer identity ───────────────────────────────────────────────────
    float layerT  = float(uLayerIndex) / max(float(uPlaneCount) - 1.0, 1.0);
    //  layerT: 0 = first/back plane, 1 = last/front plane

    // ── Animated noise sample ────────────────────────────────────────────────
    // Each layer drifts at a slightly different phase to break repetition.
    float phaseX  = uTime * uSpeed * (0.7 + layerT * 0.3);
    float phaseY  = uTime * uSpeed * (0.5 - layerT * 0.2);

    vec2 noiseUV  = vUV * uScale + vec2(phaseX, phaseY);
    float density = fbm(noiseUV);

    // Mix with flat fog using uNoise intensity
    float fogDensity = mix(0.65, density, clamp(uNoise, 0.0, 1.0));

    // ── Depth-based fade (AT fadeDist=[near, far]) ───────────────────────────
    // vDepth01 drives a smooth fade: thin at edges, thicker in the middle.
    float depthFade = smoothstep(0.0, uFadeNear / (uFadeNear + uFadeFar),
                                 vDepth01)
                    * smoothstep(1.0, 1.0 - uFadeNear / (uFadeNear + uFadeFar),
                                 vDepth01);

    // ── Edge fade (vignette on UV) ───────────────────────────────────────────
    vec2  centered   = vUV * 2.0 - 1.0;  // [-1,1]
    float edgeFade   = 1.0 - smoothstep(0.6, 1.0, length(centered));

    // ── Layer opacity ────────────────────────────────────────────────────────
    // Back layers are slightly more transparent to keep the stack believable.
    float layerAlpha = uAlpha
                     * fogDensity
                     * depthFade
                     * edgeFade
                     * mix(0.4, 1.0, layerT);   // back→dim, front→bright

    // Clamp to avoid over-bright compositing
    float alpha = clamp(layerAlpha * 0.12, 0.0, 1.0);  // 0.12 ≈ 1/8 per layer

    if (alpha < 0.002) discard;

    // Pre-multiplied alpha output (matches PixiJS additive / normal blend modes)
    finalColor = vec4(uFogColor * alpha, alpha);
}
{@}cloud-fog.vert{@}#version 300 es
// ── cloud-fog.vert ────────────────────────────────────────────────────────────
// Volumetric fog vertex shader — one quad per fog plane.
//
// Each plane is a simple unit quad [-0.5, +0.5] in XY, placed at a fixed Z
// depth by the CPU (via uModelMatrix or uDepthZ).  The vertex shader forwards
// UV coordinates and a remapped depth scalar to the fragment stage.
//
// Attributes:
//   aPosition  — (x, y) in [-0.5, +0.5] quad-local space
//   aUV        — (u, v) in [0, 1] over the quad
//
// Uniforms:
//   uProjection  — 4×4 projection matrix
//   uView        — 4×4 view / camera matrix
//   uModel       — 4×4 model matrix (positions + scales the plane in world space)
//   uDepth01     — [0,1] normalised depth of this layer within the volume
//
// Outputs:
//   vUV          — passthrough UV to fragment
//   vDepth01     — passthrough depth scalar to fragment
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── attributes ────────────────────────────────────────────────────────────────
in vec2 aPosition;   // quad corner in local space
in vec2 aUV;         // [0,1] UV over the quad

// ── uniforms ──────────────────────────────────────────────────────────────────
uniform mat4  uProjection;
uniform mat4  uView;
uniform mat4  uModel;
uniform float uDepth01;

// ── varyings to fragment ──────────────────────────────────────────────────────
out vec2  vUV;
out float vDepth01;

// ─────────────────────────────────────────────────────────────────────────────
void main() {
    vUV      = aUV;
    vDepth01 = uDepth01;

    vec4 worldPos = uModel * vec4(aPosition, 0.0, 1.0);
    gl_Position   = uProjection * uView * worldPos;
}
{@}colormap.frag{@}/**
 * colormap.frag
 * 
 * Perceptually uniform scientific colormaps for WebGL.
 * Implements Viridis, Plasma, Inferno, Magma from matplotlib (van der Walt & Smith, SciPy 2015).
 * 
 * Approach: degree-6 polynomial approximation fitted to the official LUT data.
 * Zero texture fetches — pure math, works on WebGL 1.0+
 * 
 * Usage:
 *   #include "colormap.frag"
 *   vec3 col = colormap_viridis(value);   // value in [0.0, 1.0]
 * 
 * References:
 *   - Polynomial coefficients: IQ / shadertoy community (shadertoy.com/view/XtGGzG)
 *   - Official matplotlib data: github.com/matplotlib/matplotlib/blob/main/_cm_listed.py
 *   - glsl-colormap: github.com/glslify/glsl-colormap
 *   - Observable WebGL colormaps: observablehq.com/@flimsyhat/webgl-color-maps
 *
 * Research: xiaodi #83 — cell-pubsub-loop
 */

precision mediump float;

// ---------------------------------------------------------------------------
// UTILITY
// ---------------------------------------------------------------------------

/**
 * Clamp t to [0,1] before colormap lookup.
 * Call this if input data may exceed the [0,1] domain.
 */
float cm_clamp01(float t) {
    return clamp(t, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// VIRIDIS
// Monotonically increasing lightness: dark purple → blue → teal → green → yellow
// Best for: general scientific data, colorblind-safe, print-safe
// Lightness range: ~15% → ~90% (CIELAB L*)
// ---------------------------------------------------------------------------
vec3 colormap_viridis(float t) {
    t = cm_clamp01(t);
    const vec3 c0 = vec3(0.2777273272234177,  0.005407344544966578, 0.3340998053353061);
    const vec3 c1 = vec3(0.1050930431085774,  1.404613529898575,    1.384590162594685);
    const vec3 c2 = vec3(-0.3308618287255563,  0.214847559468213,    0.09509516302823659);
    const vec3 c3 = vec3(-4.634230498983486,  -5.799100973351585,  -19.33244095627987);
    const vec3 c4 = vec3(6.228269936347081,   14.17993336680509,   56.69055260068105);
    const vec3 c5 = vec3(4.776384997670288,  -13.74514537774601,  -65.35303263337234);
    const vec3 c6 = vec3(-5.435455855934631,   4.645852612178535,   26.3124352495832);
    return clamp(c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6))))), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// PLASMA
// High contrast: dark purple → magenta → orange → yellow
// Best for: high-contrast visualization, aesthetically striking
// Lightness range: ~12% → ~92% (CIELAB L*)
// ---------------------------------------------------------------------------
vec3 colormap_plasma(float t) {
    t = cm_clamp01(t);
    const vec3 c0 = vec3(0.05873234392399702,  0.02333670892565664,  0.5433401826748754);
    const vec3 c1 = vec3(2.176514634195958,    0.2383834171260182,   0.7539604599784036);
    const vec3 c2 = vec3(-2.689460476458034,  -7.455851135738909,   3.110799939717086);
    const vec3 c3 = vec3(6.130348345893603,   42.35286317604309,   -28.51885465332158);
    const vec3 c4 = vec3(-11.10743619062271, -82.66631109428045,    60.13984767418263);
    const vec3 c5 = vec3(10.02306557647065,   71.41361770095349,   -54.07218655560067);
    const vec3 c6 = vec3(-3.658713842777788, -22.93153465461149,   18.19190778539828);
    return clamp(c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6))))), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// INFERNO
// Black → dark red → orange → light yellow-white
// Best for: heatmaps, density fields, fire/thermal effects
// Lightness range: ~0% → ~95% (excellent for dark backgrounds)
// ---------------------------------------------------------------------------
vec3 colormap_inferno(float t) {
    t = cm_clamp01(t);
    const vec3 c0 = vec3(0.0002189403691192265,  0.001651004631001012,  -0.01948089843709584);
    const vec3 c1 = vec3(0.1065134194856116,      0.5639564367884091,     3.932712388889277);
    const vec3 c2 = vec3(11.60249308247187,      -3.972853965665698,    -15.9423941062914);
    const vec3 c3 = vec3(-41.70399613139459,      17.43639888205313,     44.35414519872813);
    const vec3 c4 = vec3(77.162935699427,        -33.40235894210092,    -81.80730925738993);
    const vec3 c5 = vec3(-71.31942824499214,      32.62606426397723,     73.20951985803202);
    const vec3 c6 = vec3(25.13112622477341,      -12.24266895238567,    -23.07032500287172);
    return clamp(c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6))))), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// MAGMA
// Black → purple/violet → rose-pink → light cream-white
// Best for: similar to Inferno but cooler hues, geological/astronomical data
// Lightness range: ~0% → ~95%
// ---------------------------------------------------------------------------
vec3 colormap_magma(float t) {
    t = cm_clamp01(t);
    const vec3 c0 = vec3(-0.002136485053939582, -0.000749655052795221, -0.005386127855323933);
    const vec3 c1 = vec3(0.2516605407371642,     0.6775232436837668,    2.494026599312351);
    const vec3 c2 = vec3(8.353717279216625,     -3.577719514958484,    0.3144679030132573);
    const vec3 c3 = vec3(-27.66873308576866,     14.26473078096533,    -13.64921318813922);
    const vec3 c4 = vec3(52.17613981234068,     -27.94360607168351,    12.94416215269321);
    const vec3 c5 = vec3(-50.76852536473588,     29.04658282127291,    4.23415299384598);
    const vec3 c6 = vec3(18.65570506591883,     -11.48977351997711,   -5.601961508734096);
    return clamp(c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6))))), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// TURBO (Google, 2019) — Rainbow replacement
// Dark blue → green → yellow → orange → dark red
// Better perceptual uniformity than Jet/Rainbow while covering full spectrum
// Best for: replacing legacy jet/rainbow, requires full-spectrum discrimination
// ---------------------------------------------------------------------------
vec3 colormap_turbo(float t) {
    t = cm_clamp01(t);
    // Polynomial from: https://observablehq.com/@flimsyhat/webgl-color-maps
    const vec4 kRedVec4   = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839,  4.84296658, -14.18503333);
    const vec4 kBlueVec4  = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRedVec2   = vec2(-152.94239396, 59.28637943);
    const vec2 kGreenVec2 = vec2(4.27729857,  2.82956604);
    const vec2 kBlueVec2  = vec2(-89.90310912, 27.34824973);

    vec4 v4 = vec4(1.0, t, t*t, t*t*t);
    vec2 v2 = v4.zw * v4.z;
    return vec3(
        dot(v4, kRedVec4)   + dot(v2, kRedVec2),
        dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
        dot(v4, kBlueVec4)  + dot(v2, kBlueVec2)
    );
}

// ---------------------------------------------------------------------------
// DIVERGING: COOLWARM (simplified)
// Blue (cold) → white (neutral) → red (hot)
// Best for: data with meaningful zero/center point (e.g. attention weights)
// ---------------------------------------------------------------------------
vec3 colormap_coolwarm(float t) {
    t = cm_clamp01(t);
    // Blue end: (0.085, 0.532, 0.201) → White: (0.865, 0.865, 0.865) → Red: (0.706, 0.016, 0.150)
    // Simplified linear blend through white at t=0.5
    vec3 cold = vec3(0.085, 0.532, 0.201);
    vec3 mid  = vec3(0.865, 0.865, 0.865);
    vec3 warm = vec3(0.706, 0.016, 0.150);
    return t < 0.5
        ? mix(cold, mid, t * 2.0)
        : mix(mid, warm, (t - 0.5) * 2.0);
}

// ---------------------------------------------------------------------------
// TRANSFER FUNCTION HELPERS
// Value → color with data range remapping
// ---------------------------------------------------------------------------

/**
 * Map a data value from [dataMin, dataMax] to a colormap.
 * @param value  raw data value
 * @param lo     minimum data range
 * @param hi     maximum data range
 */
vec3 colormap_viridis_range(float value, float lo, float hi) {
    return colormap_viridis((value - lo) / max(hi - lo, 1e-6));
}

vec3 colormap_plasma_range(float value, float lo, float hi) {
    return colormap_plasma((value - lo) / max(hi - lo, 1e-6));
}

vec3 colormap_inferno_range(float value, float lo, float hi) {
    return colormap_inferno((value - lo) / max(hi - lo, 1e-6));
}

/**
 * Log-scale colormap: useful for data with high dynamic range (e.g. particle density).
 * Applies log10 normalization then viridis colormap.
 * @param value  positive data value
 * @param lo     minimum (positive!) data value
 * @param hi     maximum data value
 */
vec3 colormap_viridis_log(float value, float lo, float hi) {
    float t = (log(max(value, lo)) - log(lo)) / max(log(hi) - log(lo), 1e-6);
    return colormap_viridis(t);
}

// ---------------------------------------------------------------------------
// COLORMAP SELECTOR (for dynamic selection via uniform)
// Usage: uniform int u_colormap_id;  // 0=viridis, 1=plasma, 2=inferno, 3=magma, 4=turbo
// ---------------------------------------------------------------------------
vec3 colormap_select(float t, int id) {
    if (id == 0) return colormap_viridis(t);
    if (id == 1) return colormap_plasma(t);
    if (id == 2) return colormap_inferno(t);
    if (id == 3) return colormap_magma(t);
    if (id == 4) return colormap_turbo(t);
    if (id == 5) return colormap_coolwarm(t);
    return colormap_viridis(t); // fallback
}
{@}curl-trail.frag{@}#version 300 es
/**
 * curl-trail.frag — Particle trail render shader
 *
 * Renders a single particle trail fragment with:
 *   - Curl noise micro-perturbation for organic trail deviation
 *   - fBm turbulence weighting for layered detail
 *   - Velocity-driven color transition (speed → hue ramp)
 *   - Soft alpha falloff from trail center
 *
 * LYGIA-style virtual includes (resolved at shader-compile time):
 *   #include "lygia/generative/curl.glsl"   — curlNoise(vec3) → vec3
 *   #include "lygia/generative/fbm.glsl"    — fbm(vec3) → float
 *
 * Both functions are inlined below following the LYGIA single-file
 * convention; replace with actual #include directives if your pipeline
 * supports LYGIA's resolver (e.g. glslify / vite-plugin-lygia).
 *
 * Uniforms:
 *   u_time          — elapsed time in seconds              (float)
 *   u_curlScale     — spatial frequency of curl field      (float, e.g. 0.8)
 *   u_curlStrength  — amplitude of curl micro-perturbation (float, e.g. 0.15)
 *   u_species       — cell species index [0..N-1]          (float, 0-based)
 *   u_velocity      — instantaneous speed scalar           (float, 0..1 norm)
 *
 * Varyings (from trail vertex shader):
 *   v_uv            — [0,1]² trail UV; v_uv.x = along-trail t, v_uv.y = cross
 *   v_life          — particle life remaining [0,1]
 *   v_worldPos      — world-space fragment position (used for noise domain)
 *
 * Output:
 *   fragColor       — premultiplied RGBA
 *
 * Research: xiaodi #M551 — cell-pubsub-loop
 */

precision highp float;

// ── Varyings ──────────────────────────────────────────────────────────────────

in  vec2  v_uv;        // x = progress along trail (0=tip, 1=tail), y = cross-section
in  float v_life;      // [0,1] life remaining
in  vec3  v_worldPos;  // world-space position for noise sampling

out vec4  fragColor;

// ── Uniforms ──────────────────────────────────────────────────────────────────

uniform float u_time;
uniform float u_curlScale;
uniform float u_curlStrength;
uniform float u_species;
uniform float u_velocity;    // normalised speed [0..1], 0=slow, 1=max

// ── #include "lygia/math/const.glsl" (inlined) ───────────────────────────────

#define PI       3.14159265358979323846
#define TWO_PI   6.28318530717958647693
#define HALF_PI  1.57079632679489661923

// ── #include "lygia/generative/curl.glsl" (inlined) ──────────────────────────
//
// Source: github.com/patriciogonzalezvivo/lygia/blob/main/generative/curl.glsl
// Divergence-free curl noise via finite-difference of three noise potentials.
// Produces smooth, swirling, vortex-like flow — no sources or sinks.

/** Canonical 3→3 hash used throughout LYGIA's generative folder. */
vec3 _ly_hash33(vec3 p) {
    p = fract(p * vec3(443.8975, 441.4234, 437.1951));
    p += dot(p, p.yxz + 19.1934);
    return fract((p.xxy + p.yxx) * p.zyx);
}

/** Value noise over a 3-D lattice. */
float _ly_vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);

    float a = dot(_ly_hash33(i + vec3(0,0,0)), vec3(1.0));
    float b = dot(_ly_hash33(i + vec3(1,0,0)), vec3(1.0));
    float c = dot(_ly_hash33(i + vec3(0,1,0)), vec3(1.0));
    float d = dot(_ly_hash33(i + vec3(1,1,0)), vec3(1.0));
    float e = dot(_ly_hash33(i + vec3(0,0,1)), vec3(1.0));
    float f2= dot(_ly_hash33(i + vec3(1,0,1)), vec3(1.0));
    float g = dot(_ly_hash33(i + vec3(0,1,1)), vec3(1.0));
    float h = dot(_ly_hash33(i + vec3(1,1,1)), vec3(1.0));

    return mix(mix(mix(a,b,u.x), mix(c,d,u.x), u.y),
               mix(mix(e,f2,u.x), mix(g,h,u.x), u.y), u.z);
}

/**
 * curlNoise(p) — lygia/generative/curl.glsl
 *
 * Approximates ∇×F(p) using three offset noise potential channels.
 * Result is always divergence-free (no particle density pile-ups).
 *
 *   curl_x = ∂Fz/∂y − ∂Fy/∂z
 *   curl_y = ∂Fx/∂z − ∂Fz/∂x
 *   curl_z = ∂Fy/∂x − ∂Fx/∂y
 */
vec3 curlNoise(vec3 p) {
    const float e  = 1e-4;

    // Potential channel offsets (prime-ish seeds, same as LYGIA)
    const vec3 ofs_y = vec3(0.0,  31.416, 0.0);
    const vec3 ofs_z = vec3(0.0,  0.0,  27.183);

    // Fx potential
    float Fx_py = _ly_vnoise(p + vec3(0, e, 0));
    float Fx_my = _ly_vnoise(p - vec3(0, e, 0));
    float Fx_pz = _ly_vnoise(p + vec3(0, 0, e));
    float Fx_mz = _ly_vnoise(p - vec3(0, 0, e));

    // Fy potential (offset domain to decorrelate)
    float Fy_pz = _ly_vnoise(p + ofs_y + vec3(0, 0, e));
    float Fy_mz = _ly_vnoise(p + ofs_y - vec3(0, 0, e));
    float Fy_px = _ly_vnoise(p + ofs_y + vec3(e, 0, 0));
    float Fy_mx = _ly_vnoise(p + ofs_y - vec3(e, 0, 0));

    // Fz potential
    float Fz_px = _ly_vnoise(p + ofs_z + vec3(e, 0, 0));
    float Fz_mx = _ly_vnoise(p + ofs_z - vec3(e, 0, 0));
    float Fz_py = _ly_vnoise(p + ofs_z + vec3(0, e, 0));
    float Fz_my = _ly_vnoise(p + ofs_z - vec3(0, e, 0));

    return vec3(
        (Fz_py - Fz_my) - (Fy_pz - Fy_mz),
        (Fx_pz - Fx_mz) - (Fz_px - Fz_mx),
        (Fy_px - Fy_mx) - (Fx_py - Fx_my)
    ) / (2.0 * e);
}

// ── #include "lygia/generative/fbm.glsl" (inlined) ───────────────────────────
//
// Source: github.com/patriciogonzalezvivo/lygia/blob/main/generative/fbm.glsl
// Classic fractional Brownian motion — sum of band-limited noise octaves.
// Amplitude halves and frequency doubles each octave (H = 0.5 → β = 2).

#define FBM_OCTAVES 5

/** Smooth value noise octave used by fbm. */
float _ly_noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(mix(_ly_hash33(i + vec3(0,0,0)).x, _ly_hash33(i + vec3(1,0,0)).x, u.x),
            mix(_ly_hash33(i + vec3(0,1,0)).x, _ly_hash33(i + vec3(1,1,0)).x, u.x), u.y),
        mix(mix(_ly_hash33(i + vec3(0,0,1)).x, _ly_hash33(i + vec3(1,0,1)).x, u.x),
            mix(_ly_hash33(i + vec3(0,1,1)).x, _ly_hash33(i + vec3(1,1,1)).x, u.x), u.y),
        u.z);
}

/**
 * fbm(p) — lygia/generative/fbm.glsl
 *
 * Fractal Brownian Motion: accumulates FBM_OCTAVES noise octaves.
 * Each successive octave: frequency ×2, amplitude ×0.5.
 * Returns value in [0, 1].
 */
float fbm(vec3 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    // Rotation matrix to reduce axis-aligned artefacts between octaves
    // (same 3×3 used in LYGIA's fbm implementation)
    mat3 m = mat3(
         0.00,  0.80,  0.60,
        -0.80,  0.36, -0.48,
        -0.60, -0.48,  0.64
    );

    for (int i = 0; i < FBM_OCTAVES; i++) {
        value     += amplitude * _ly_noise(p * frequency);
        p          = m * p;            // decorrelate octave domains
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// ── Color palette — species-indexed hue ramps ─────────────────────────────────
//
// Twelve species slots (expandable). Each entry is a (base_hue, sat, drift)
// triple where drift is added to base_hue when velocity approaches 1.
//
// Accessed via species index so each cell type gets a distinct color family.

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

/**
 * speciesColor(speciesIdx, velocity, turbulence)
 *
 * Returns an RGB color blended from:
 *   - rest hue  → species base color (slow particles)
 *   - speed hue → warm accent (+30° hue shift) when velocity is high
 *   - turbulence modulates saturation: calm = vivid, chaotic = washed
 */
vec3 speciesColor(float speciesIdx, float vel, float turb) {
    // Base hue per species — evenly distributed over [0,1] hue circle,
    // offset by golden-ratio step for perceptual distinctness.
    float baseHue   = fract(speciesIdx * 0.618033988749895 + 0.12);

    // Speed-driven hue drift: fast particles shift warm (+0.08 ≈ +29°)
    float speedHue  = fract(baseHue + mix(0.0, 0.08, vel));

    // Turbulence de-saturates: fBm=0 → full sat, fBm=1 → reduced sat
    float sat       = mix(0.90, 0.55, turb);

    // Brightness: trail tip bright, slows at tail; life modulates too
    float bri       = mix(0.55, 1.00, vel);

    // Blend rest↔speed hue by velocity
    float hue       = mix(baseHue, speedHue, smoothstep(0.2, 0.8, vel));

    return hsv2rgb(vec3(hue, sat, bri));
}

// ── Main ──────────────────────────────────────────────────────────────────────

void main() {
    // ── 1. Curl micro-perturbation of trail UV ────────────────────────────
    //
    // Sample the curl field in world-space at current time tick.
    // Perturb the cross-section UV (v_uv.y) so the trail "wiggles" organically
    // rather than rendering as a rigid ribbon.
    vec3 curlDomain = v_worldPos * u_curlScale
                    + vec3(u_time * 0.15, 0.0, u_time * 0.07);
    vec3 curl        = curlNoise(curlDomain);

    // Apply perturbation only perpendicular to trail direction (x/y of curl)
    float curlCross  = curl.x * 0.6 + curl.y * 0.4;  // weighted axis blend
    float perturbedV = v_uv.y + curlCross * u_curlStrength;

    // ── 2. fBm turbulence weight ──────────────────────────────────────────
    //
    // Evaluate fBm in a space that evolves gently over time so turbulence
    // drifts rather than flickering. Used to weight curl amplitude and
    // modulate color saturation.
    vec3 fbmDomain  = v_worldPos * (u_curlScale * 0.5)
                    + vec3(0.0, u_time * 0.05, u_time * 0.03);
    float turbulence = fbm(fbmDomain);  // [0,1]

    // Stronger curl where fBm turbulence is high — chaotic regions spiral more
    float dynamicCurl = u_curlStrength * (0.6 + 0.8 * turbulence);
    perturbedV        = v_uv.y + curlCross * dynamicCurl;

    // ── 3. Trail radial alpha — soft ribbon cross-section ─────────────────
    //
    // Distance from trail centreline (v_uv.y == 0.5).
    float crossDist  = abs(perturbedV - 0.5) * 2.0;  // 0 at center, 1 at edge
    float radialMask = 1.0 - smoothstep(0.55, 1.0, crossDist);

    // ── 4. Trail length alpha — tip bright, tail fades ────────────────────
    //
    // v_uv.x: 0 = fresh tip, 1 = oldest tail section
    float lengthFade = 1.0 - smoothstep(0.0, 1.0, v_uv.x);

    // Boost tip with a small glow kernel
    float tipGlow    = exp(-v_uv.x * 6.0) * 0.4;

    // ── 5. Life modulation ────────────────────────────────────────────────
    //
    // Particles near end-of-life fade out gracefully
    float lifeFade   = smoothstep(0.0, 0.12, v_life);

    // ── 6. Velocity-driven color ──────────────────────────────────────────
    //
    // Combine velocity with a turbulence term so neighboring trail segments
    // at the same speed still vary slightly in saturation.
    float velMod     = clamp(u_velocity + turbulence * 0.12 - 0.06, 0.0, 1.0);
    vec3  trailColor = speciesColor(u_species, velMod, turbulence);

    // Tip brightens at high velocity — hot-core effect
    float tipBoost   = tipGlow * velMod;
    trailColor       = mix(trailColor, vec3(1.0), tipBoost * 0.35);

    // ── 7. Compose alpha ──────────────────────────────────────────────────

    float alpha = radialMask
                * (lengthFade + tipGlow)
                * lifeFade
                * clamp(u_velocity * 0.5 + 0.5, 0.3, 1.0);  // slow particles fade
    alpha = clamp(alpha, 0.0, 1.0);

    if (alpha < 0.005) discard;

    // Premultiplied alpha output — composited additively in bloom pass
    fragColor = vec4(trailColor * alpha, alpha);
}
{@}edge-line.frag{@}#version 300 es
// ── edge-line.frag ────────────────────────────────────────────────────────────
// Straight-line edge shader — fragment stage.
//
// Evaluates the signed-distance-field (SDF) of the line segment in pixel space
// so that:
//   • The stroke has sub-pixel antialiased edges (no jaggies at any scale)
//   • Dash patterns are evaluated analytically (no texture lookup)
//   • The caps are round (zero extra vertex work)
//
// Uniforms:
//   uColor         — RGB stroke colour (vec3, 0-1)
//   uAlpha         — master opacity (float)
//   uLineWidth     — full stroke width in pixels (float)
//   uDashLength    — on-length  of one dash in pixels; 0 = solid (float)
//   uGapLength     — off-length of the gap in pixels (float)
//   uGlowRadius    — optional outer glow radius in pixels; 0 = off (float)
//   uGlowColor     — glow colour (vec3)
//   uGlowAlpha     — glow peak opacity (float)
//   uTime          — seconds, for animated dashes (float)
//   uDashOffset    — extra phase offset in pixels (float)
//
// Inputs from vertex shader:
//   vFragCoordPx   — pixel-space fragment position
//   vP0, vP1       — segment endpoints in pixel space
//   vHalfWidth     — half-width of the stroke (pixels)
//   vUV            — quad UV (not used in SDF evaluation but available)
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── varyings from vertex ──────────────────────────────────────────────────────
in vec2  vFragCoordPx;
in vec2  vP0;
in vec2  vP1;
in float vHalfWidth;
in vec2  vUV;

// ── uniforms ──────────────────────────────────────────────────────────────────
uniform vec3  uColor;
uniform float uAlpha;
uniform float uLineWidth;
uniform float uDashLength;   // 0 = solid
uniform float uGapLength;
uniform vec3  uGlowColor;
uniform float uGlowRadius;   // 0 = no glow
uniform float uGlowAlpha;
uniform float uTime;
uniform float uDashOffset;

// ── output ────────────────────────────────────────────────────────────────────
out vec4 finalColor;

// ─────────────────────────────────────────────────────────────────────────────
// SDF: distance from point p to segment (a → b)
// Returns (distance_to_segment, t_along_segment)
// ─────────────────────────────────────────────────────────────────────────────
vec2 sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2  pa = p - a;
    vec2  ba = b - a;
    float h  = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return vec2(length(pa - ba * h), h);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dash pattern: returns 1 if this position is "on", 0 if in a gap.
// t_px  — arc-length position along the segment in pixels
// ─────────────────────────────────────────────────────────────────────────────
float dashMask(float t_px) {
    if (uDashLength < 0.5) return 1.0;  // solid line

    float period = uDashLength + uGapLength;
    float phase  = mod(t_px + uTime * 40.0 + uDashOffset, period);

    // Smooth transition at dash edges (1 px soft)
    float onEnd  = uDashLength;
    float on     = smoothstep(0.0, 1.0, phase)
                 * (1.0 - smoothstep(onEnd - 1.0, onEnd, phase));
    return clamp(on, 0.0, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────

void main() {
    vec2  p       = vFragCoordPx;
    float halfW   = vHalfWidth;
    float segLen  = length(vP1 - vP0);

    // SDF
    vec2  sd      = sdSegment(p, vP0, vP1);
    float dist    = sd.x;          // pixels from segment centreline
    float t_norm  = sd.y;          // 0..1 along segment
    float t_px    = t_norm * segLen;

    // Antialiased stroke coverage
    float strokeAlpha = 1.0 - smoothstep(halfW - 0.75, halfW + 0.75, dist);

    // Dash mask
    float dash = dashMask(t_px);
    strokeAlpha *= dash;

    // Glow (rendered beneath the stroke)
    float glowAlpha = 0.0;
    if (uGlowRadius > 0.5) {
        float glowD = max(0.0, dist - halfW);
        glowAlpha   = uGlowAlpha
                    * exp(-glowD * glowD / (uGlowRadius * uGlowRadius * 0.5))
                    * dash;
    }

    // Composite: glow underneath, stroke on top
    vec3  col   = mix(uGlowColor, uColor, strokeAlpha);
    float alpha = max(strokeAlpha, glowAlpha) * uAlpha;

    if (alpha < 0.004) discard;

    finalColor = vec4(col * alpha, alpha);  // premultiplied alpha
}
{@}edge-line.vert{@}#version 300 es
// ── edge-line.vert ────────────────────────────────────────────────────────────
// Straight-line edge shader — vertex stage.
//
// Renders each edge as a screen-space quad (4 vertices, 2 triangles) that
// tightly wraps the line segment plus half-width padding on every side.
//
// Technique: the CPU uploads the two endpoints once as uniforms; the vertex
// shader expands them into a bbox quad and passes the local-space UV to the
// fragment stage where the SDF antialiased line is evaluated.
//
// Attributes (per-vertex, 4 vertices per quad):
//   aPosition  — (x, y) NDC position of this quad corner
//   aUV        — (u, v) in [0,1] over the quad bounding-box
//
// Uniforms:
//   uP0        — start point in canvas-pixel space (vec2)
//   uP1        — end   point in canvas-pixel space (vec2)
//   uLineWidth — half-width of the rendered stroke in pixels (float)
//   uResolution — canvas (width, height) in pixels (vec2)
//
// Outputs:
//   vUV        — passthrough UV for fragment SDF
//   vP0, vP1   — endpoints forwarded to fragment in normalised canvas coords
//   vHalfWidth — half-width in the same normalised space
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

in vec2 aPosition;   // NDC quad corner  (-1..1, -1..1)
in vec2 aUV;         // [0,1] over the quad bbox

uniform vec2  uP0;          // start point (pixel space)
uniform vec2  uP1;          // end   point (pixel space)
uniform float uLineWidth;   // full stroke width in pixels
uniform vec2  uResolution;  // canvas (w, h) in pixels

// ── outs to fragment ──────────────────────────────────────────────────────────
out vec2  vUV;
out vec2  vP0;
out vec2  vP1;
out float vHalfWidth;
out vec2  vFragCoordPx;  // pixel-space position of this fragment

// ── helpers ───────────────────────────────────────────────────────────────────

// Convert pixel-space point to NDC, accounting for Y-down canvas convention.
vec2 pixelToNDC(vec2 px) {
    return (px / uResolution) * 2.0 - 1.0;
}

// ─────────────────────────────────────────────────────────────────────────────

void main() {
    float halfW = uLineWidth * 0.5 + 1.5;  // 1.5 px AA headroom

    // Direction and perpendicular of the segment in pixel space
    vec2  dir   = uP1 - uP0;
    float len   = length(dir);
    vec2  unit  = (len > 0.001) ? dir / len : vec2(1.0, 0.0);
    vec2  perp  = vec2(-unit.y, unit.x);

    // Quad corners in pixel space:
    //   0 = P0 - perp*hw - unit*hw
    //   1 = P0 + perp*hw - unit*hw
    //   2 = P1 + perp*hw + unit*hw
    //   3 = P1 - perp*hw + unit*hw
    // aUV.x selects along the segment (0 = P0 side, 1 = P1 side)
    // aUV.y selects across the segment (0 = -perp, 1 = +perp)
    vec2 longOff  = mix(-unit * halfW, unit  * halfW, aUV.x);
    vec2 perpOff  = mix(-perp * halfW, perp  * halfW, aUV.y);
    vec2 anchor   = mix(uP0, uP1, aUV.x);
    vec2 cornerPx = anchor + longOff + perpOff;

    // Pass pixel coords to fragment for the SDF evaluation
    vFragCoordPx = cornerPx;
    vP0          = uP0;
    vP1          = uP1;
    vHalfWidth   = halfW - 1.5;  // strip the AA padding back
    vUV          = aUV;

    // Y-flip: canvas is Y-down, NDC is Y-up
    vec2 ndc = pixelToNDC(cornerPx);
    ndc.y    = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
{@}edge-spline.frag{@}#version 300 es
// ── edge-spline.frag ──────────────────────────────────────────────────────────
// Cubic Bézier skip-connection shader — fragment stage.
//
// Each fragment receives the pixel-space position of this corner (vFragCoordPx)
// and the nearest point on the curve (vCurvePx) forwarded from the vertex stage
// for approximate distance computation.  Because each quad covers only 1/N of
// the curve, the nearest-on-segment approximation is accurate enough for any
// subdivision count ≥ 16.
//
// The curvature value from topology.json already shaped the Bézier control
// points on the CPU side (see EdgeRenderer.ts) so this shader doesn't need it
// directly — it just draws the line as defined by the geometry.
//
// Uniforms:
//   uColor        — RGB stroke colour (vec3)
//   uAlpha        — master opacity (float)
//   uLineWidth    — full stroke width in pixels (float)
//   uDashLength   — dash on-length in pixels; 0 = solid (float)
//   uGapLength    — dash gap-length in pixels (float)
//   uGlowColor    — skip-connection glow colour (vec3)
//   uGlowRadius   — glow radius in pixels; 0 = off (float)
//   uGlowAlpha    — peak glow opacity (float)
//   uTime         — seconds, drives animated dash travel (float)
//   uArcLength    — total arc length estimate in pixels (float)
//   uCurvature    — curvature parameter from topology.json (float, 0-1)
//   u_sourceColor — RGB colour at source node (vec3)
//   u_targetColor — RGB colour at target node (vec3)
//   u_flowSpeed   — flow-pulse scroll speed (float)
//   u_thickness   — additional thickness control (float)
//   u_time        — global animation time in seconds (float)
//
// Inputs from vertex:
//   vFragCoordPx  — pixel-space position of this fragment
//   vT            — normalised Bézier t at this vertex
//   vTangentDir   — unit tangent at vT
//   vCurvePx      — pixel-space curve position at quad midpoint
//   vHalfWidth    — stroke half-width in pixels
//   v_t           — normalised parametric t for species colour lerp
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── varyings from vertex ──────────────────────────────────────────────────────
in float vT;
in vec2  vTangentDir;
in vec2  vCurvePx;
in float vHalfWidth;
in vec2  vFragCoordPx;
in float v_t;           // normalised parametric t for species colour lerp

// ── uniforms ──────────────────────────────────────────────────────────────────
uniform vec3  uColor;
uniform float uAlpha;
uniform float uLineWidth;
uniform float uDashLength;
uniform float uGapLength;
uniform vec3  uGlowColor;
uniform float uGlowRadius;
uniform float uGlowAlpha;
uniform float uTime;
uniform float uArcLength;
uniform float uCurvature;

// ── species colour & flow uniforms ───────────────────────────────────────────
uniform vec3  u_sourceColor;  // RGB at source node
uniform vec3  u_targetColor;  // RGB at target node
uniform float u_flowSpeed;    // flow-pulse scroll speed
uniform float u_thickness;    // additional thickness control
uniform float u_time;         // global animation time (seconds)

// ── output ────────────────────────────────────────────────────────────────────
out vec4 finalColor;

// ─────────────────────────────────────────────────────────────────────────────
// Approximate SDF to the curve using the quad midpoint as the nearest point.
// This is exact at the centre of each quad; error is bounded by the sub-arc
// chord-to-curve deviation, which falls off as O(1/N²) for N subdivisions.
// ─────────────────────────────────────────────────────────────────────────────
float approxDistToCurve() {
    vec2 diff = vFragCoordPx - vCurvePx;
    // Project onto tangent and perpendicular
    vec2 perp = vec2(-vTangentDir.y, vTangentDir.x);
    return abs(dot(diff, perp));   // perpendicular distance = SDF approximation
}

// ─────────────────────────────────────────────────────────────────────────────
// Dash pattern along arc length
// ─────────────────────────────────────────────────────────────────────────────
float dashMask(float t) {
    if (uDashLength < 0.5) return 1.0;

    float t_px   = t * uArcLength;
    float period = uDashLength + uGapLength;
    // Animated: dashes travel along the curve direction
    float phase  = mod(t_px - uTime * 50.0, period);

    float onEnd  = uDashLength;
    float on     = smoothstep(0.0, 1.0, phase)
                 * (1.0 - smoothstep(onEnd - 1.0, onEnd, phase));
    return clamp(on, 0.0, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Curvature tint: skip connections get a subtle hue shift based on curvature.
// curvature = 0 → uColor as-is; curvature = 1 → slight shift toward uGlowColor
// ─────────────────────────────────────────────────────────────────────────────
vec3 curvatureTint(vec3 base) {
    return mix(base, uGlowColor, uCurvature * 0.25);
}

// ─────────────────────────────────────────────────────────────────────────────

void main() {
    float halfW = vHalfWidth + u_thickness * 0.5;

    // Approx distance to Bézier centreline
    float dist  = approxDistToCurve();

    // Antialiased stroke coverage
    float strokeAlpha = 1.0 - smoothstep(halfW - 0.75, halfW + 0.75, dist);

    // Dash mask
    float dash = dashMask(vT);
    strokeAlpha *= dash;

    // Glow (beneath stroke)
    float glowAlpha = 0.0;
    if (uGlowRadius > 0.5) {
        float glowD = max(0.0, dist - halfW);
        glowAlpha   = uGlowAlpha
                    * exp(-glowD * glowD / (uGlowRadius * uGlowRadius * 0.5))
                    * dash;
    }

    // ── Species colour gradient: source → target along v_t ──────────────
    vec3 speciesColor = mix(u_sourceColor, u_targetColor, v_t);

    // ── Flow pulse: scrolling bright band along the edge ────────────────
    float pulse = fract(v_t - u_time * u_flowSpeed);
    // Shape pulse into a soft peak (narrow bright band, fades to ~0.3)
    float pulseIntensity = smoothstep(0.0, 0.15, pulse)
                         * (1.0 - smoothstep(0.15, 0.45, pulse));
    // Blend: species gradient base brightened by flow pulse
    vec3 baseColor = speciesColor * (1.0 + 0.6 * pulseIntensity);

    // Optional curvature tint for skip connections
    vec3 strokeCol = curvatureTint(baseColor);

    // Composite
    vec3  col   = mix(uGlowColor, strokeCol, strokeAlpha);
    float alpha = max(strokeAlpha, glowAlpha) * uAlpha;

    if (alpha < 0.004) discard;

    finalColor = vec4(col * alpha, alpha);  // premultiplied alpha
}
{@}edge-spline.vert{@}attribute vec2 a_position;
attribute float a_t;
uniform mat3 u_projectionMatrix;
uniform float u_thickness;
varying float v_t;
void main(){
  v_t=a_t;
  vec3 pos=u_projectionMatrix*vec3(a_position,1.0);
  gl_Position=vec4(pos.xy,0.0,1.0);
  gl_PointSize=u_thickness;
}
{@}fluid-surface.frag{@}#version 300 es
// ── fluid-surface.frag ────────────────────────────────────────────────────────
// SPH Metaball fluid surface renderer.
//
// Technique:
//   Each draw call covers the bounding quad of an SPH cell cluster.  We
//   reconstruct a smooth implicit surface by blending the per-particle density
//   field via a radial metaball kernel, then shade the resulting isosurface
//   with three coupled effects:
//
//   1. DENSITY SMOOTHSTEP  — The scalar field is thresholded with smoothstep to
//      produce a clean liquid surface with an adjustable transition band.
//      Pixels above the iso-threshold are "inside" the fluid; the gradient
//      magnitude becomes the surface normal estimate for cheap specular.
//
//   2. COLLISION RIPPLES   — On each rigid-body ↔ fluid collision event the
//      host writes a ping-pong ripple texture (uRippleTex).  The lygia
//      ripple.glsl propagation kernel advances the wave across the surface
//      and the resulting height offset perturbs the iso-field so collisions
//      visibly dimple / splash the fluid.
//
//   3. VELOCITY DISPERSION — The SPH velocity field (uVelocityTex, RG) drives
//      two things:
//        a. Classic Perlin cnoise (lygia/generative/cnoise.glsl) modulated by
//           speed magnitude produces turbulent micro-wrinkles on fast-moving
//           regions.
//        b. Chromatic aberration (lygia/distort/chromaAB.glsl) is applied to
//           the final surface colour: high-speed areas get stronger RGB channel
//           separation, giving a prismatic dispersion effect at wave fronts.
//
// LYGIA virtual includes (resolved at shader-compile time by the project
// GLSL preprocessor / vite-plugin-glsl):
//   #include "../../upstream/lygia/simulate/ripple.glsl"
//   #include "../../upstream/lygia/generative/cnoise.glsl"
//   #include "../../upstream/lygia/distort/chromaAB.glsl"
//
// All three LYGIA sources are inlined below following the project convention
// (see curl-trail.frag, grayscott-species.frag) so the file is self-contained.
//
// Uniforms:
//   uDensityTex      — R: SPH particle density field          sampler2D
//   uVelocityTex     — RG: particle velocity (world-space)    sampler2D
//   uRippleTex       — RG: ripple ping-pong (current frame)   sampler2D
//   uTexelSize       — 1 / texture resolution                 vec2
//   uTime            — elapsed seconds                        float
//   uIsoThreshold    — metaball iso-level [0,1], default 0.5  float
//   uSmoothBand      — smoothstep transition half-width        float
//   uFluidColor      — base fluid tint RGB                    vec3
//   uSpecularColor   — specular highlight RGB                  vec3
//   uSpecularPower   — Phong exponent                         float
//   uRippleStrength  — ripple displacement scale               float
//   uNoiseScale      — cnoise spatial frequency               float
//   uNoiseStrength   — cnoise displacement amplitude           float
//   uChromaStrength  — chromaAB aberration scale              float
//   uMaxSpeed        — velocity normalisation reference        float
//
// Varyings (from fullscreen-quad vertex shader):
//   vUV              — [0,1]² texture coordinates
//
// Output:
//   fragColor        — premultiplied RGBA
//
// Research: xiaodi #M553 — cell-pubsub-loop
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── Varyings ──────────────────────────────────────────────────────────────────

in  vec2  vUV;

out vec4  fragColor;

// ── Uniforms ──────────────────────────────────────────────────────────────────

uniform sampler2D uDensityTex;      // R  = SPH density  [0,1]
uniform sampler2D uVelocityTex;     // RG = velocity (world units / s, normalised)
uniform sampler2D uRippleTex;       // RG = ripple propagation buffer (current)
uniform vec2      uTexelSize;       // vec2(1.0/w, 1.0/h)

uniform float     uTime;

// Surface reconstruction
uniform float     uIsoThreshold;    // default 0.50
uniform float     uSmoothBand;      // default 0.08  — transition half-width

// Shading
uniform vec3      uFluidColor;      // e.g. vec3(0.18, 0.52, 0.89)
uniform vec3      uSpecularColor;   // e.g. vec3(0.90, 0.95, 1.00)
uniform float     uSpecularPower;   // e.g. 32.0

// Ripple collision
uniform float     uRippleStrength;  // e.g. 0.06

// Velocity cnoise wrinkle
uniform float     uNoiseScale;      // e.g. 4.0
uniform float     uNoiseStrength;   // e.g. 0.04

// Chromatic aberration
uniform float     uChromaStrength;  // e.g. 2.0
uniform float     uMaxSpeed;        // normalisation, e.g. 5.0

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/saturate.glsl" (inlined) ────────────────────────────
// Source: upstream/lygia/math/saturate.glsl
// License: Patricio Gonzalez Vivo — Prosperity / Patron License
// ─────────────────────────────────────────────────────────────────────────────

#if !defined(FNC_SATURATE) && !defined(saturate)
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/mod289.glsl" (inlined) ───────────────────────────────
// Source: upstream/lygia/math/mod289.glsl
// contributors: Stefan Gustavson, Ian McEwan
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_MOD289
#define FNC_MOD289
float mod289(const in float x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2  mod289(const in vec2  x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3  mod289(const in vec3  x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4  mod289(const in vec4  x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/permute.glsl" (inlined) ──────────────────────────────
// Source: upstream/lygia/math/permute.glsl
// contributors: Stefan Gustavson, Ian McEwan
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_PERMUTE
#define FNC_PERMUTE
float permute(const in float v) { return mod289(((v * 34.0) + 1.0) * v); }
vec2  permute(const in vec2  v) { return mod289(((v * 34.0) + 1.0) * v); }
vec3  permute(const in vec3  v) { return mod289(((v * 34.0) + 1.0) * v); }
vec4  permute(const in vec4  v) { return mod289(((v * 34.0) + 1.0) * v); }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/taylorInvSqrt.glsl" (inlined) ────────────────────────
// Source: upstream/lygia/math/taylorInvSqrt.glsl
// contributors: Stefan Gustavson, Ian McEwan
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_TAYLORINVSQRT
#define FNC_TAYLORINVSQRT
float taylorInvSqrt(in float r) { return 1.79284291400159 - 0.85373472095314 * r; }
vec2  taylorInvSqrt(in vec2  r) { return 1.79284291400159 - 0.85373472095314 * r; }
vec3  taylorInvSqrt(in vec3  r) { return 1.79284291400159 - 0.85373472095314 * r; }
vec4  taylorInvSqrt(in vec4  r) { return 1.79284291400159 - 0.85373472095314 * r; }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/quintic.glsl" (inlined) ──────────────────────────────
// Source: upstream/lygia/math/quintic.glsl
// contributors: Inigo Quiles
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_QUINTIC
#define FNC_QUINTIC
float quintic(const in float v) { return v*v*v*(v*(v*6.0-15.0)+10.0); }
vec2  quintic(const in vec2  v) { return v*v*v*(v*(v*6.0-15.0)+10.0); }
vec3  quintic(const in vec3  v) { return v*v*v*(v*(v*6.0-15.0)+10.0); }
vec4  quintic(const in vec4  v) { return v*v*v*(v*(v*6.0-15.0)+10.0); }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/generative/cnoise.glsl" (inlined) ─────────────────────────
// Source: upstream/lygia/generative/cnoise.glsl
// contributors: Stefan Gustavson, Ian McEwan
// description: Classic Perlin Noise  https://github.com/stegu/webgl-noise
// License: MIT  https://opensource.org/license/mit/
//
// Only the vec2 and vec3 overloads are included; the vec4 overload is omitted
// to reduce code size — only 2D / 3D sampling is used in this shader.
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_CNOISE
#define FNC_CNOISE

// ── 2D Classic Perlin Noise ───────────────────────────────────────────────────
float cnoise(in vec2 P) {
    vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
    vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
    Pi = mod289(Pi);
    vec4 ix = Pi.xzxz;
    vec4 iy = Pi.yyww;
    vec4 fx = Pf.xzxz;
    vec4 fy = Pf.yyww;
    vec4 i  = permute(permute(ix) + iy);
    vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0;
    vec4 gy = abs(gx) - 0.5;
    vec4 tx = floor(gx + 0.5);
    gx = gx - tx;
    vec2 g00 = vec2(gx.x, gy.x);
    vec2 g10 = vec2(gx.y, gy.y);
    vec2 g01 = vec2(gx.z, gy.z);
    vec2 g11 = vec2(gx.w, gy.w);
    vec4 norm = taylorInvSqrt(vec4(dot(g00,g00), dot(g01,g01), dot(g10,g10), dot(g11,g11)));
    g00 *= norm.x;  g01 *= norm.y;  g10 *= norm.z;  g11 *= norm.w;
    float n00 = dot(g00, vec2(fx.x, fy.x));
    float n10 = dot(g10, vec2(fx.y, fy.y));
    float n01 = dot(g01, vec2(fx.z, fy.z));
    float n11 = dot(g11, vec2(fx.w, fy.w));
    vec2 fade_xy = quintic(Pf.xy);
    vec2 n_x     = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
    float n_xy   = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}

// ── 3D Classic Perlin Noise ───────────────────────────────────────────────────
float cnoise(in vec3 P) {
    vec3 Pi0 = floor(P);
    vec3 Pi1 = Pi0 + vec3(1.0);
    Pi0 = mod289(Pi0);
    Pi1 = mod289(Pi1);
    vec3 Pf0 = fract(P);
    vec3 Pf1 = Pf0 - vec3(1.0);
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = Pi0.zzzz;
    vec4 iz1 = Pi1.zzzz;
    vec4 ixy  = permute(permute(ix) + iy);
    vec4 ixy0 = permute(ixy + iz0);
    vec4 ixy1 = permute(ixy + iz1);
    vec4 gx0 = ixy0 * (1.0 / 7.0);
    vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
    gx0 = fract(gx0);
    vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
    vec4 sz0 = step(gz0, vec4(0.0));
    gx0 -= sz0 * (step(0.0, gx0) - 0.5);
    gy0 -= sz0 * (step(0.0, gy0) - 0.5);
    vec4 gx1 = ixy1 * (1.0 / 7.0);
    vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
    gx1 = fract(gx1);
    vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
    vec4 sz1 = step(gz1, vec4(0.0));
    gx1 -= sz1 * (step(0.0, gx1) - 0.5);
    gy1 -= sz1 * (step(0.0, gy1) - 0.5);
    vec3 g000 = vec3(gx0.x, gy0.x, gz0.x);
    vec3 g100 = vec3(gx0.y, gy0.y, gz0.y);
    vec3 g010 = vec3(gx0.z, gy0.z, gz0.z);
    vec3 g110 = vec3(gx0.w, gy0.w, gz0.w);
    vec3 g001 = vec3(gx1.x, gy1.x, gz1.x);
    vec3 g101 = vec3(gx1.y, gy1.y, gz1.y);
    vec3 g011 = vec3(gx1.z, gy1.z, gz1.z);
    vec3 g111 = vec3(gx1.w, gy1.w, gz1.w);
    vec4 norm0 = taylorInvSqrt(vec4(dot(g000,g000), dot(g010,g010), dot(g100,g100), dot(g110,g110)));
    g000 *= norm0.x;  g010 *= norm0.y;  g100 *= norm0.z;  g110 *= norm0.w;
    vec4 norm1 = taylorInvSqrt(vec4(dot(g001,g001), dot(g011,g011), dot(g101,g101), dot(g111,g111)));
    g001 *= norm1.x;  g011 *= norm1.y;  g101 *= norm1.z;  g111 *= norm1.w;
    float n000 = dot(g000, Pf0);
    float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
    float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
    float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
    float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
    float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
    float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
    float n111 = dot(g111, Pf1);
    vec3 fade_xyz = quintic(Pf0);
    vec4 n_z  = mix(vec4(n000,n100,n010,n110), vec4(n001,n101,n011,n111), fade_xyz.z);
    vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
    float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
    return 2.2 * n_xyz;
}

#endif // FNC_CNOISE

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/simulate/ripple.glsl" (inlined) ───────────────────────────
// Source: upstream/lygia/simulate/ripple.glsl
// original_author: Patricio Gonzalez Vivo
// description: Simple Ripple Propagation — advances a 2-channel ping-pong
//   buffer that stores current (r) and previous (g) wave heights.
//   Returns vec3(next_height, current_height, 0).
// License: Patricio Gonzalez Vivo — Prosperity / Patron License
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_RIPPLE
#define FNC_RIPPLE
vec3 ripple(sampler2D tex, vec2 st, vec2 pixel) {
    vec3 rta = texture(tex, st).rgb;
    float s0 = rta.y;
    float s1 = texture(tex, st + vec2( 0.0,       -pixel.y)).r;   //     s1
    float s2 = texture(tex, st + vec2(-pixel.x,    0.0    )).r;   //  s2 s0 s3
    float s3 = texture(tex, st + vec2( pixel.x,    0.0    )).r;   //     s4
    float s4 = texture(tex, st + vec2( 0.0,        pixel.y)).r;
    float d  = -(s0 - 0.5) * 2.0 + (s1 + s2 + s3 + s4 - 2.0);
    d *= 0.99;                          // damping
    d  = saturate(d * 0.5 + 0.5);
    return vec3(d, rta.x, 0.0);
}
#endif // FNC_RIPPLE

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/lengthSq.glsl" (inlined) ─────────────────────────────
// Source: upstream/lygia/math/lengthSq.glsl
// contributors: Patricio Gonzalez Vivo
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_LENGTHSQ
#define FNC_LENGTHSQ
float lengthSq(in vec2 v) { return dot(v, v); }
float lengthSq(in vec3 v) { return dot(v, v); }
float lengthSq(in vec4 v) { return dot(v, v); }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/distort/chromaAB.glsl" (inlined) ──────────────────────────
// Source: upstream/lygia/distort/chromaAB.glsl
// contributors: Patricio Gonzalez Vivo, Johan Ismael
// description: Chromatic aberration — shifts R/G/B channels by a scaled
//   offset derived from the distance of st from the frame centre.
// License: Patricio Gonzalez Vivo — Prosperity / Patron License
// ─────────────────────────────────────────────────────────────────────────────

#ifndef CHROMAAB_PCT
#define CHROMAAB_PCT 1.5
#endif

#ifndef FNC_CHROMAAB
#define FNC_CHROMAAB

// Overload: explicit RGB distortion channels
vec3 chromaAB(in sampler2D tex, in vec2 st, in vec2 direction, in vec3 distortion) {
    vec3 c;
    c.r = texture(tex, st + direction * distortion.r).r;
    c.g = texture(tex, st + direction * distortion.g).g;
    c.b = texture(tex, st + direction * distortion.b).b;
    return c;
}

// Overload: scalar sdf offset + strength percentage
vec3 chromaAB(in sampler2D tex, in vec2 st, in float sdf, in float pct) {
    vec2 stR = st * (1.0 + vec2(sdf) * 0.02 * pct);
    vec2 stB = st * (1.0 - vec2(sdf) * 0.02 * pct);
    vec3 c;
    c.r = texture(tex, stR).r;
    c.g = texture(tex, st ).g;
    c.b = texture(tex, stB).b;
    return c;
}

// Overload: default pct from macro
vec3 chromaAB(in sampler2D tex, in vec2 st, in float sdf) {
    return chromaAB(tex, st, sdf, CHROMAAB_PCT);
}

// Overload: auto-compute sdf from distance-to-centre
vec3 chromaAB(in sampler2D tex, in vec2 st) {
    return chromaAB(tex, st, lengthSq(st - 0.5), CHROMAAB_PCT);
}

#endif // FNC_CHROMAAB

// ─────────────────────────────────────────────────────────────────────────────
// ── Metaball density field helpers ────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

// Finite-difference normal estimation from the density scalar field.
// Uses a ±1-texel stencil to approximate the gradient direction.
vec2 densityGradient(sampler2D tex, vec2 st, vec2 px) {
    float dL = texture(tex, st - vec2(px.x, 0.0)).r;
    float dR = texture(tex, st + vec2(px.x, 0.0)).r;
    float dD = texture(tex, st - vec2(0.0, px.y)).r;
    float dU = texture(tex, st + vec2(0.0, px.y)).r;
    return vec2(dR - dL, dU - dD) * 0.5;
}

// ─────────────────────────────────────────────────────────────────────────────
// ── SPH velocity speed + direction helpers ────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

// Returns normalised speed [0,1] from the velocity texture RG channels.
float velocitySpeed(sampler2D tex, vec2 st, float maxSpd) {
    vec2  vel   = texture(tex, st).rg;
    float speed = length(vel);
    return saturate(speed / max(maxSpd, 0.001));
}

// ─────────────────────────────────────────────────────────────────────────────
// ── main ──────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

void main() {

    // ── 1. Sample raw SPH density ─────────────────────────────────────────────
    float rawDensity = texture(uDensityTex, vUV).r;   // [0,1] from GPGPU pass

    // ── 2. Ripple collision perturbation ──────────────────────────────────────
    // Advance the ripple wave propagation one step (lygia simulate/ripple.glsl).
    // The resulting height (.x) displaces the density isosurface inward so
    // collision events produce visible surface dimples / splash rings.
    vec3  rippleState  = ripple(uRippleTex, vUV, uTexelSize);
    float rippleHeight = rippleState.x * 2.0 - 1.0;    // remap [0,1] → [-1,1]
    float perturbedDensity = rawDensity + rippleHeight * uRippleStrength;

    // ── 3. Velocity-driven cnoise surface wrinkle ─────────────────────────────
    // Classic Perlin noise modulated by the local SPH speed gives organic
    // micro-turbulence that intensifies at high-velocity flow regions.
    float speed3D    = velocitySpeed(uVelocityTex, vUV, uMaxSpeed);
    vec2  velXY      = texture(uVelocityTex, vUV).rg;
    // Advect the noise domain along velocity to align wrinkles with flow.
    vec2  noiseUV    = vUV * uNoiseScale + velXY * 0.1 + vec2(uTime * 0.07);
    float noiseVal   = cnoise(noiseUV);                         // [-1,1]
    perturbedDensity += noiseVal * uNoiseStrength * speed3D;

    // ── 4. Metaball smooth-step surface reconstruction ────────────────────────
    // Threshold with a configurable band to produce a clean isosurface alpha.
    float lo   = uIsoThreshold - uSmoothBand;
    float hi   = uIsoThreshold + uSmoothBand;
    float surf = smoothstep(lo, hi, perturbedDensity);   // 0=outside, 1=inside

    // Discard fully exterior fragments early (saves bandwidth on large quads).
    if (surf < 0.004) discard;

    // ── 5. Surface normal from density gradient (cheap 2-D approximation) ─────
    vec2  grad        = densityGradient(uDensityTex, vUV, uTexelSize);
    float gradLen     = length(grad);
    vec3  surfNormal  = (gradLen > 1e-5)
        ? normalize(vec3(-grad * 2.0, 1.0))
        : vec3(0.0, 0.0, 1.0);

    // Simple view-space light from top-right (constant for a 2-D surface pass).
    vec3  lightDir    = normalize(vec3(0.6, 0.8, 1.0));
    float diffuse     = max(dot(surfNormal, lightDir), 0.0);

    // Blinn-Phong specular
    vec3  viewDir     = vec3(0.0, 0.0, 1.0);
    vec3  halfVec     = normalize(lightDir + viewDir);
    float specular    = pow(max(dot(surfNormal, halfVec), 0.0), uSpecularPower);

    // ── 6. Chromatic aberration on the density texture (velocity dispersion) ──
    // Fast-moving SPH regions scatter light chromatically — higher speed ⟹
    // stronger RGB channel separation (lygia distort/chromaAB.glsl).
    // We use the density texture as the colour source; the aberration shifts
    // each channel radially outward from the frame centre, proportional to
    // the normalised speed magnitude.
    float aberrationSdf = lengthSq(vUV - 0.5) * speed3D * uChromaStrength;
    vec3  chromaColor   = chromaAB(uDensityTex, vUV, aberrationSdf, 1.0);

    // ── 7. Composite final fluid colour ───────────────────────────────────────
    // Base colour from the tinted chroma sample, then add diffuse + specular.
    vec3  baseColor = uFluidColor * (0.6 + 0.4 * chromaColor.r);
    vec3  litColor  = baseColor  * (0.3 + 0.7 * diffuse)
                    + uSpecularColor * specular * surf;

    // Interior depth tinting: deeper fluid areas (high density) are slightly
    // darker/more saturated to hint at subsurface volume.
    float depthTint = 1.0 - saturate((perturbedDensity - uIsoThreshold) * 1.5);
    litColor       *= mix(1.0, 0.55, depthTint);

    // Surface-edge glow — thin halo at the isosurface boundary.
    float edgeGlow  = 1.0 - smoothstep(0.0, uSmoothBand * 2.5, abs(perturbedDensity - uIsoThreshold));
    litColor       += uSpecularColor * edgeGlow * 0.25;

    // ── 8. Ripple shimmer overlay ─────────────────────────────────────────────
    // The ripple wave height creates a subtle brightness ripple on the surface.
    float rippleShimmer = rippleHeight * 0.12 * surf;
    litColor           += uFluidColor * rippleShimmer;

    // ── 9. Alpha — surf mask with soft interior falloff ───────────────────────
    float alpha = surf * mix(0.82, 1.0, diffuse);

    // ── 10. Pre-multiplied alpha output ───────────────────────────────────────
    fragColor = vec4(litColor * alpha, alpha);
}
{@}grayscott-species.frag{@}precision mediump float;

// ── grayscott-species.frag ────────────────────────────────────────────────────
// Gray-Scott Reaction-Diffusion Turing Pattern — species surface shader.
//
// ⚠️  SUPERSEDED BY WebGPU COMPUTE PIPELINE (M601)
//     This fragment shader remains as a WebGL 1 / fallback path.
//     For WebGPU targets use src/lib/sph/reaction-diffusion.ts instead:
//       • ReactionDiffusionSim — full GPU ping-pong compute
//       • parameterSpace(name) — Munafo/Pearson canonical (f, k) lookup
//
// Generates procedural Turing patterns (spots / stripes / worms / mitosis)
// directly on each cell's surface without a ping-pong texture.  We simulate
// several RD iterations per fragment using spatially-varying virtual "pixels"
// derived from the cell's bbox, then composite the chemical concentration over
// the cell's fill colour.
//
// Each species carries its own (f, k) parameter pair that selects a different
// region of the Gray-Scott phase diagram:
//
//   cil-eye    (0)  f=0.0545 k=0.0620  → coral / spots     (瞳孔 spot pattern)
//   cil-bolt   (1)  f=0.0180 k=0.0510  → waves / maze      (闪电 wave pattern)
//   cil-vector (2)  f=0.0290 k=0.0570  → worms / filaments  (方向流 worm pattern)
//   cil-plus   (3)  f=0.0367 k=0.0649  → mitosis / pearls  (细胞分裂 pattern)
//   species 4-9     fall back to the u_feedKill override from the host.
//
// Physics coupling:
//   u_density   → scales diffusion rates (high density ⟹ denser pattern)
//   u_velocity  → stretches UV along velocity direction (flow distortion)
//
// GLSL #include dependencies (resolved by the project GLSL preprocessor):
//   ../../upstream/lygia/simulate/grayscott.glsl   — GS reaction step
//   ../../upstream/lygia/generative/fbm.glsl       — fBm seeding
//   ../../upstream/lygia/sdf/circleSDF.glsl        — boundary mask
//
// References:
//   P. Gonzalez Vivo — lygia.xyz
//   Pearson, J.E. (1993) — Complex Patterns in a Simple System, Science 261
//   Munafo, R. — mrob.com/pub/comp/xmorphia
//   Karl Sims — karlsims.com/rd.html  (coral f=0.0545,k=0.062; mitosis f=0.0367,k=0.0649)
//   Shaders: M550 (fragment) → M601 (WebGPU compute) — cell-pubsub-loop branch
// ─────────────────────────────────────────────────────────────────────────────

// ── lygia imports ─────────────────────────────────────────────────────────────
#include "../../upstream/lygia/simulate/grayscott.glsl"
#include "../../upstream/lygia/generative/fbm.glsl"
#include "../../upstream/lygia/sdf/circleSDF.glsl"

// ── uniforms ──────────────────────────────────────────────────────────────────
uniform vec4  u_bbox;       // (x, y, width, height) in canvas coords
uniform int   u_species;    // 0-9 species index → selects (f,k) preset
uniform float u_density;    // SPH density [0,1] — scales diffusion rate
uniform vec2  u_velocity;   // SPH velocity (vx,vy) — distorts UV field
uniform float u_time;       // seconds
uniform vec2  u_feedKill;   // (f, k) override for species 4-9

// ── per-species (f, k) lookup ─────────────────────────────────────────────────
//
// Gray-Scott phase diagram regions (Pearson 1993 / Munafo):
//   f=0.055 k=0.062  → spots  (coral / cil-eye)
//   f=0.018 k=0.051  → waves  (maze  / cil-bolt)
//   f=0.029 k=0.057  → worms  (filaments / cil-vector)
//   f=0.025 k=0.060  → mitosis (pearls / cil-plus)
//
// Implemented as a branchless lerp-chain instead of a switch (WebGL 1 compat).

vec2 speciesFeedKill(int idx) {
    // species 0 — cil-eye     coral/spots   f=0.0545 k=0.0620  (Karl Sims / Munafo κ)
    vec2 fk0 = vec2(0.0545, 0.0620);
    // species 1 — cil-bolt    maze/waves    f=0.0180 k=0.0510  (Munafo γ / maze)
    vec2 fk1 = vec2(0.0180, 0.0510);
    // species 2 — cil-vector  worms/filaments f=0.0290 k=0.0570 (Pearson δ labyrinth)
    vec2 fk2 = vec2(0.0290, 0.0570);
    // species 3 — cil-plus    mitosis/pearls  f=0.0367 k=0.0649 (Karl Sims / Munafo λ)
    vec2 fk3 = vec2(0.0367, 0.0649);

    // branchless index selection (WebGL 1 / mediump safe)
    float i = float(idx);
    vec2 fk = fk0;
    fk = mix(fk, fk1, step(0.5, i - 0.0) * step(i - 0.0, 0.9));  // idx==1
    fk = mix(fk, fk2, step(0.5, i - 1.0) * step(i - 1.0, 0.9));  // idx==2
    fk = mix(fk, fk3, step(0.5, i - 2.0) * step(i - 2.0, 0.9));  // idx==3
    // species 4-9: use host-supplied override
    fk = mix(fk, u_feedKill, step(4.0, i));
    return fk;
}

// ── helper: hash-based pseudo-random ─────────────────────────────────────────
float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// ── RD virtual iteration ──────────────────────────────────────────────────────
//
// Because we have no ping-pong textures at this stage, we emulate several
// Gray-Scott time steps by sampling an fBm "landscape" to initialise the UV
// concentrations and then running the Gray-Scott laplacian kernel analytically
// over the fBm gradients.  This produces spatially-varying frozen-time Turing
// patterns without GPU-side state.
//
// The technique:
//   1.  Use fBm as the "u" (activator) initial condition.
//   2.  Derive "v" (inhibitor) as 1-u perturbed by hash noise.
//   3.  Run N Gray-Scott iterations using the laplacian of fBm at the sample
//       point (approximated by the 3×3 stencil over a sub-pixel grid).
//   4.  Accumulate the resulting chemical concentration.
//
// This is a recognised technique for static Turing-pattern generation in
// fragment shaders (see: iq, Inigo Quilez / Shadertoy community).

#define GS_ITER 6       // RD iterations per fragment (mediump budget)
#define PIXEL_SCALE 2.5 // virtual pixel size relative to bbox

vec2 rdEvolve(vec2 uv, vec2 fk, float diffScale) {
    // Virtual pixel size in UV space
    vec2 px = vec2(PIXEL_SCALE) / u_bbox.zw;

    // fBm-seeded initial concentrations
    float u0 = fbm(uv * 6.0 + u_time * 0.04);
    float v0 = 1.0 - fbm(uv * 6.0 + vec2(5.3, 1.7) + u_time * 0.03);

    // Add seeding noise so pattern erupts from heterogeneous IC
    float seed = hash21(floor(uv * 24.0) + vec2(u_time * 0.01));
    v0 = clamp(v0 + step(0.92, seed) * 0.3, 0.0, 1.0);

    float f = fk.x;
    float k = fk.y;

    // diffusion rates scaled by SPH density
    // high density → tighter (faster diffusing) → denser pattern
    float densityScale = 0.7 + u_density * 0.6;
    float diffU = 0.25 * diffScale * densityScale;
    float diffV = 0.05 * diffScale * densityScale;

    float u = u0;
    float v = v0;

    // Unrolled GS_ITER iterations using manually computed Laplacian from
    // 3×3 fBm stencil (avoids texture dependency).
    for (int i = 0; i < GS_ITER; i++) {
        float fi = float(i);

        // Sample fBm at 3×3 neighbours for Laplacian (Pearson kernel weights)
        // Corner weight: 0.707106781  Edge weight: 1.0  Centre: -6.828427
        float uLap = 0.0;
        float vLap = 0.0;

        // The fBm acts as a surrogate state texture — we sample it at shifted
        // UVs to build the discrete Laplacian.  The pattern "freezes" because
        // time only enters through u_time which drifts the fBm slowly.
        float phase = fi * 0.17 + u_time * 0.025;

        float centre_u = fbm(uv * 6.0 + vec2(phase));
        float centre_v = 1.0 - fbm(uv * 6.0 + vec2(5.3, 1.7 + phase));

        // 4 edge neighbours (weight 1.0)
        uLap += fbm((uv + vec2( px.x, 0.0)) * 6.0 + vec2(phase));
        uLap += fbm((uv + vec2(-px.x, 0.0)) * 6.0 + vec2(phase));
        uLap += fbm((uv + vec2(0.0,  px.y)) * 6.0 + vec2(phase));
        uLap += fbm((uv + vec2(0.0, -px.y)) * 6.0 + vec2(phase));

        vLap += 1.0 - fbm((uv + vec2( px.x, 0.0)) * 6.0 + vec2(5.3, 1.7 + phase));
        vLap += 1.0 - fbm((uv + vec2(-px.x, 0.0)) * 6.0 + vec2(5.3, 1.7 + phase));
        vLap += 1.0 - fbm((uv + vec2(0.0,  px.y)) * 6.0 + vec2(5.3, 1.7 + phase));
        vLap += 1.0 - fbm((uv + vec2(0.0, -px.y)) * 6.0 + vec2(5.3, 1.7 + phase));

        // 4 corner neighbours (weight 0.707106781)
        float cw = 0.707106781;
        uLap += cw * fbm((uv + vec2( px.x,  px.y)) * 6.0 + vec2(phase));
        uLap += cw * fbm((uv + vec2(-px.x,  px.y)) * 6.0 + vec2(phase));
        uLap += cw * fbm((uv + vec2( px.x, -px.y)) * 6.0 + vec2(phase));
        uLap += cw * fbm((uv + vec2(-px.x, -px.y)) * 6.0 + vec2(phase));

        vLap += cw * (1.0 - fbm((uv + vec2( px.x,  px.y)) * 6.0 + vec2(5.3, 1.7 + phase)));
        vLap += cw * (1.0 - fbm((uv + vec2(-px.x,  px.y)) * 6.0 + vec2(5.3, 1.7 + phase)));
        vLap += cw * (1.0 - fbm((uv + vec2( px.x, -px.y)) * 6.0 + vec2(5.3, 1.7 + phase)));
        vLap += cw * (1.0 - fbm((uv + vec2(-px.x, -px.y)) * 6.0 + vec2(5.3, 1.7 + phase)));

        // Centre weight: -(4 * 1.0 + 4 * 0.707106781) = -6.828427
        uLap += -6.828427 * centre_u;
        vLap += -6.828427 * centre_v;

        // Gray-Scott reaction step
        float uvv = u * v * v;
        float du = diffU * uLap - uvv + f * (1.0 - u);
        float dv = diffV * vLap + uvv - (f + k) * v;

        u = clamp(u + du * 0.6, 0.0, 1.0);
        v = clamp(v + dv * 0.6, 0.0, 1.0);
    }

    return vec2(u, v);
}

// ── main ──────────────────────────────────────────────────────────────────────
void main() {
    // Normalise fragment to bbox-local UV [0,1]
    vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;

    // ── Physics coupling: velocity distorts UV ────────────────────────────────
    // High speed → pattern stretched along flow direction.
    // Velocity is in world-units/s; we normalise to a reasonable warp range.
    float speed = length(u_velocity);
    vec2  flowDir = speed > 0.001 ? normalize(u_velocity) : vec2(1.0, 0.0);
    // Project UV onto flow direction and warp
    float flow = dot(uv - 0.5, flowDir);
    float warpStr = clamp(speed * 0.08, 0.0, 0.35);
    vec2 uvWarped = uv + flowDir * flow * warpStr;
    // Clamp to avoid sampling outside cell
    uvWarped = clamp(uvWarped, 0.0, 1.0);

    // ── circleSDF boundary mask ───────────────────────────────────────────────
    // Fade the pattern out at the cell edges using the lygia circleSDF.
    float dist = circleSDF(uv);          // 0 = centre, 1 = boundary, >1 = outside
    float edgeMask = smoothstep(1.05, 0.75, dist);

    // ── Species (f,k) parameters ──────────────────────────────────────────────
    vec2  fk = speciesFeedKill(u_species);

    // ── Density modulates diffusion scale ─────────────────────────────────────
    // u_density in [0,1].  Low density → slower diffusion → coarser pattern.
    // High density → faster diffusion → finer, denser markings.
    float diffScale = 0.6 + u_density * 0.8;

    // ── Run RD evolution ──────────────────────────────────────────────────────
    vec2 rd = rdEvolve(uvWarped, fk, diffScale);
    float uConc = rd.x;   // activator  (chemical U)
    float vConc = rd.y;   // inhibitor  (chemical V)

    // ── Pattern visualisation ─────────────────────────────────────────────────
    // The inhibitor V concentration is the "ink" of the Turing pattern:
    //   high V  → dark markings (spots, stripes, worm heads)
    //   low  V  → bright background
    //
    // We use 1-U as a secondary channel that cleanly separates the pattern
    // from the background at the Pearson threshold ~0.5.

    float pattern = clamp(vConc * 2.0 - 0.5, 0.0, 1.0);

    // Species-specific colour tinting:
    //   spots  (0) → sharp high-contrast spots
    //   waves  (1) → softer continuous labyrinthine bands
    //   worms  (2) → elongated filaments, mid-contrast
    //   mitosis(3) → small oval pearls with halos
    float contrast = 1.0;
    float fi = float(u_species);
    contrast = mix(contrast, 2.2, step(0.5, fi - 0.0) * step(fi, 0.5));  // spots
    contrast = mix(contrast, 1.2, step(0.5, fi - 0.5) * step(fi, 1.5));  // waves
    contrast = mix(contrast, 1.6, step(0.5, fi - 1.5) * step(fi, 2.5));  // worms
    contrast = mix(contrast, 1.8, step(0.5, fi - 2.5) * step(fi, 3.5));  // mitosis

    pattern = clamp((pattern - 0.5) * contrast + 0.5, 0.0, 1.0);

    // Inner glow from activator U: bright halos around pattern edges
    float glow = smoothstep(0.3, 0.7, uConc) * 0.25 * edgeMask;

    // ── Final alpha compositing ───────────────────────────────────────────────
    // Pattern alpha: solid at pattern peaks, transparent at background.
    // The cell body colour is rendered by the host; this shader adds the
    // Turing overlay ON TOP with additive/multiply blending intent.
    float alpha = clamp(pattern * edgeMask + glow, 0.0, 1.0);

    // Pattern colour: dark markings = near-black over cell colour.
    // We output the concentration directly; the host renderer multiplies by
    // u_fillColor when compositing.
    float brightness = 1.0 - pattern * 0.85;
    vec3  col = vec3(brightness) + vec3(glow * 0.3, glow * 0.6, glow * 1.0);

    gl_FragColor = vec4(col, alpha);
}
{@}iq-palette-species.frag{@}precision mediump float;

// ── iq-palette-species.frag ───────────────────────────────────────────────────
// IQ Procedural Palette — per-species coloring for the cell-pubsub-loop system.
//
// Implements Inigo Quilez's cosine-based palette formula:
//
//   color(t) = a + b * cos( 2π * (c*t + d) )
//
// where t = f(u_density, uv, u_time) and (a, b, c, d) are per-species vec3
// parameters that produce visually distinct, aesthetically coherent colour
// families for each cell species.
//
// Per-species palette identity:
//   species 0  cil-eye      — electric cyan / violet rings      (cool bioluminescent)
//   species 1  cil-bolt     — amber / gold lightning strikes     (warm energy)
//   species 2  cil-vector   — emerald / teal flow gradients      (cool directional)
//   species 3  cil-plus     — rose / coral cell membrane         (warm organic)
//   species 4  cil-arrow-r  — sky-blue / white directional pulse (airy cool)
//   species 5  cil-filter   — deep purple / magenta spectral     (cool spectral)
//   species 6  cil-code     — green terminal / phosphor           (classic CRT)
//   species 7  cil-layers   — warm ochre / rust depth planes     (warm earthy)
//   species 8  cil-loop     — mint / seafoam cyclic pulse        (soft cool)
//   species 9  cil-graph    — crimson / orange node glow         (hot graph)
//
// Physics coupling:
//   u_density  → drives palette t parameter (0 = dark/cool, 1 = bright/saturated)
//   u_velocity → modulates spatial noise frequency (high speed = tighter bands)
//   u_time     → slow drift of t for animated colour cycling
//
// GLSL #include dependencies (resolved by the project GLSL preprocessor):
//   ../../upstream/lygia/color/palette.glsl      — IQ cosine palette fn
//   ../../upstream/lygia/generative/snoise.glsl  — simplex noise for t modulation
//
// References:
//   Inigo Quilez — "Palettes" — iquilezles.org/articles/palettes/palettes.htm
//   P. Gonzalez Vivo — lygia.xyz
//   Shader series M570 — cell-pubsub-loop branch
// ─────────────────────────────────────────────────────────────────────────────

// ── lygia imports ─────────────────────────────────────────────────────────────
#include "../../upstream/lygia/color/palette.glsl"
#include "../../upstream/lygia/generative/snoise.glsl"

// ── uniforms ──────────────────────────────────────────────────────────────────
uniform vec4  u_bbox;       // (x, y, width, height) in canvas coords
uniform int   u_species;    // 0-9 species index
uniform float u_density;    // SPH density [0,1] — primary t driver
uniform vec2  u_velocity;   // SPH velocity (vx,vy) — noise frequency modulator
uniform float u_time;       // seconds — slow colour drift
uniform float u_alpha;      // master opacity [0,1] (default 1.0)

// ── TAU guard (palette.glsl pulls lygia/math/const.glsl which defines TAU) ────
#ifndef TAU
#define TAU 6.2831853071795864769252867665590
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  Per-species (a, b, c, d) palette parameters
//
//  Naming convention following IQ's notation:
//    a — DC offset (mean colour; controls overall brightness and hue bias)
//    b — amplitude (colour contrast; 0 = monochrome, 1 = full swing)
//    c — frequency per channel (how many hue cycles over [0,1])
//    d — phase offset per channel (rotates the hue wheel at t=0)
//
//  Each species returns four vec3 values.  The palette function is then:
//    palette(t, a, b, c, d)  ≡  a + b * cos(TAU * (c*t + d))
//
//  Parameter derivation strategy:
//    - a+b ≤ 1 in each channel avoids clamp-clipping.
//    - c ≈ 1 → single smooth hue cycle across t [0,1]; c=2 → faster cycling.
//    - d shifts hue at t=0 without affecting the shape of the ramp.
//    - Keeping one channel's c low locks a "background" tint while the
//      other two cycle more, producing species-specific colour identity.
// ─────────────────────────────────────────────────────────────────────────────

struct PaletteParams {
    vec3 a;
    vec3 b;
    vec3 c;
    vec3 d;
};

// Branchless extraction: build all 10 palettes, select by weight.
// (WebGL 1 / mediump: no dynamic array indexing, no switch statements)

PaletteParams speciesPalette(int idx) {
    float fi = float(idx);

    // ── species 0: cil-eye — electric cyan / violet ────────────────────────
    // Dark centre, bright rim rings.  Cool bioluminescent pulse.
    PaletteParams p0;
    p0.a = vec3(0.50, 0.50, 0.50);
    p0.b = vec3(0.50, 0.50, 0.50);
    p0.c = vec3(1.00, 1.00, 1.00);
    p0.d = vec3(0.00, 0.33, 0.67);

    // ── species 1: cil-bolt — amber / gold ────────────────────────────────
    // Warm lightning.  Strong yellows with orange-magenta accent.
    PaletteParams p1;
    p1.a = vec3(0.60, 0.50, 0.20);
    p1.b = vec3(0.40, 0.40, 0.20);
    p1.c = vec3(0.80, 0.90, 1.00);
    p1.d = vec3(0.00, 0.10, 0.20);

    // ── species 2: cil-vector — emerald / teal ────────────────────────────
    // Cool directional flow.  Green bias, teal-to-cyan sweep.
    PaletteParams p2;
    p2.a = vec3(0.40, 0.60, 0.50);
    p2.b = vec3(0.40, 0.40, 0.30);
    p2.c = vec3(0.50, 1.00, 0.67);
    p2.d = vec3(0.80, 0.10, 0.30);

    // ── species 3: cil-plus — rose / coral ────────────────────────────────
    // Warm organic membrane.  Pink-salmon-peach gradient.
    PaletteParams p3;
    p3.a = vec3(0.65, 0.40, 0.40);
    p3.b = vec3(0.35, 0.35, 0.25);
    p3.c = vec3(1.00, 0.70, 0.60);
    p3.d = vec3(0.00, 0.15, 0.40);

    // ── species 4: cil-arrow-right — sky / white ──────────────────────────
    // Airy directional pulse.  High-key light blues and whites.
    PaletteParams p4;
    p4.a = vec3(0.70, 0.75, 0.80);
    p4.b = vec3(0.30, 0.25, 0.20);
    p4.c = vec3(0.60, 0.60, 0.50);
    p4.d = vec3(0.20, 0.40, 0.60);

    // ── species 5: cil-filter — deep purple / magenta ─────────────────────
    // Spectral sweep.  Violet-indigo-pink high-contrast.
    PaletteParams p5;
    p5.a = vec3(0.50, 0.30, 0.60);
    p5.b = vec3(0.50, 0.40, 0.40);
    p5.c = vec3(1.00, 1.00, 0.50);
    p5.d = vec3(0.80, 0.90, 0.30);

    // ── species 6: cil-code — green terminal / phosphor ───────────────────
    // CRT phosphor.  Bright green on near-black with amber flicker.
    PaletteParams p6;
    p6.a = vec3(0.20, 0.50, 0.20);
    p6.b = vec3(0.20, 0.50, 0.10);
    p6.c = vec3(0.00, 1.00, 0.00);
    p6.d = vec3(0.50, 0.00, 0.00);

    // ── species 7: cil-layers — ochre / rust depth ────────────────────────
    // Warm earthy sediment layers.  Brown-orange-sienna bands.
    PaletteParams p7;
    p7.a = vec3(0.60, 0.45, 0.25);
    p7.b = vec3(0.40, 0.30, 0.20);
    p7.c = vec3(0.80, 0.60, 0.80);
    p7.d = vec3(0.00, 0.20, 0.50);

    // ── species 8: cil-loop — mint / seafoam ──────────────────────────────
    // Soft cyclic pulse.  Pale greens and aquas cycling gently.
    PaletteParams p8;
    p8.a = vec3(0.50, 0.70, 0.65);
    p8.b = vec3(0.30, 0.30, 0.35);
    p8.c = vec3(0.40, 0.60, 1.00);
    p8.d = vec3(0.00, 0.25, 0.50);

    // ── species 9: cil-graph — crimson / orange node ──────────────────────
    // Hot graph glow.  Red-orange-yellow node pulse.
    PaletteParams p9;
    p9.a = vec3(0.65, 0.35, 0.20);
    p9.b = vec3(0.35, 0.35, 0.20);
    p9.c = vec3(1.00, 0.80, 0.60);
    p9.d = vec3(0.00, 0.10, 0.25);

    // ── Branchless selection via step/mix ladder ───────────────────────────
    // Each weight is 1.0 only when fi is within [n-0.5, n+0.5).
    // step(lo, fi) * step(fi, hi)  → 1 iff lo <= fi <= hi

    #define SEL(n, pA, pB) \
        float w##n = step(float(n) - 0.5, fi) * step(fi, float(n) + 0.4); \
        pA.a = mix(pA.a, pB.a, w##n); \
        pA.b = mix(pA.b, pB.b, w##n); \
        pA.c = mix(pA.c, pB.c, w##n); \
        pA.d = mix(pA.d, pB.d, w##n);

    SEL(1, p0, p1)
    SEL(2, p0, p2)
    SEL(3, p0, p3)
    SEL(4, p0, p4)
    SEL(5, p0, p5)
    SEL(6, p0, p6)
    SEL(7, p0, p7)
    SEL(8, p0, p8)
    SEL(9, p0, p9)

    #undef SEL

    return p0;
}

// ── Palette t parameter computation ───────────────────────────────────────────
//
// t is constructed from three components:
//
//   1. u_density  — primary physical driver.  Low density → t near 0 (dark,
//      muted).  High density → t near 1 (bright, saturated).  This means
//      denser SPH regions appear more vivid and chromatically active.
//
//   2. Simplex noise — adds spatial texture so the colour isn't flat across
//      the cell surface.  Noise is scaled by speed: fast cells show tighter,
//      more varied banding; slow/static cells show gentle, smooth gradients.
//
//   3. u_time drift — a slow sinusoidal term that gently cycles the hue over
//      ~40 s without overriding the density signal.  Amplitude is small so
//      density stays dominant.
//
// Final t ∈ [0,1] is clamped after combining.

float computeT(vec2 uv) {
    // 1. Density base
    float tDensity = u_density;

    // 2. Spatial noise modulation
    //    Speed controls how "tight" / high-frequency the banding is.
    float speed = length(u_velocity);
    // freqScale: slow cell → 2.0 (broad bands), fast cell → 8.0 (fine bands)
    float freqScale = 2.0 + clamp(speed * 0.15, 0.0, 6.0);
    // Simplex noise in [-1,1] → remap to [-0.5, 0.5] contribution
    float noise = snoise(uv * freqScale + vec2(u_time * 0.02, u_time * 0.015)) * 0.5;
    float tNoise = noise * 0.25;   // ±0.25 spatial variation

    // 3. Slow time drift — one full cycle ≈ 40 s, amplitude ±0.08
    float tDrift = sin(u_time * 0.157) * 0.08;

    return clamp(tDensity + tNoise + tDrift, 0.0, 1.0);
}

// ── Edge vignette ─────────────────────────────────────────────────────────────
// Fades colour to black at the cell border so species chips don't hard-clip.
float edgeVignette(vec2 uv) {
    vec2 d = uv - 0.5;
    float r = length(d) * 2.0;           // 0 = centre, 1 = inscribed circle edge
    return smoothstep(1.05, 0.60, r);    // fade starts at 60% radius
}

// ── main ──────────────────────────────────────────────────────────────────────
void main() {
    // Normalise fragment to bbox-local UV [0,1]
    vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;

    // Compute palette t from density + spatial noise + time drift
    float t = computeT(uv);

    // Fetch per-species palette parameters
    PaletteParams p = speciesPalette(u_species);

    // IQ cosine palette:  a + b * cos(TAU * (c*t + d))
    vec3 col = palette(t, p.a, p.b, p.c, p.d);

    // Radial vignette — soften at cell boundary
    float vignette = edgeVignette(uv);
    col *= vignette;

    // Subtle density-driven brightness boost: dense regions pop forward
    // (u_density already encoded in t; this adds non-linear punchiness)
    col = mix(col * 0.6, col, u_density);

    // Master opacity.  Default 1.0; host can animate fade-in/out.
    float alpha = clamp(u_alpha, 0.0, 1.0) * vignette;

    gl_FragColor = vec4(col, alpha);
}
{@}julia-background.frag{@}#version 300 es
// ── julia-background.frag ────────────────────────────────────────────────────
// Julia-set fractal fullscreen background shader.
//
// Technique:
//   A standard Julia set iteration renders the escape-time field over the
//   viewport.  The complex parameter c is not fixed — it drifts slowly in
//   the complex plane driven by two mechanisms:
//
//   1. SLOW ORBITAL DRIFT  — c traces a Lemniscate-like path parameterised
//      by uTime so the fractal shape continuously morphs between canonical
//      Julia configurations (spirals → dendrites → Douady rabbits → Fatou
//      dust) without any host-side updates.
//
//   2. QoS VISCOSITY MODULATION  — uViscosity (mapped from the active QoS
//      profile via qos-spatial-bridge.ts: RELIABLE→0.02, BEST_EFFORT→0.001)
//      scales a Simplex-noise perturbation applied on top of the orbital
//      path.  High viscosity (ordered, RELIABLE channels) keeps c close to
//      its nominal orbit → smooth, stable fractal.  Low viscosity
//      (turbulent, BEST_EFFORT channels) lets noise push c far off-orbit →
//      chaotic, dendrite-heavy patterns.
//
//      viscosity range: [0.001, 0.02]
//        mapped to noise amplitude: [0.35, 0.02]
//      so the shader inverts viscosity: thin/turbulent QoS → wild fractal.
//
//   3. IQ PALETTE COLOURING  — Inigo Quilez's cosine palette formula
//      a + b * cos(TAU * (c * t + d)) is applied to the smooth escape-time
//      value.  Four built-in palettes are selected by uPaletteIdx:
//        0 — blue-magenta-gold   (default, cell-pubsub aesthetic)
//        1 — cyan-white-warm     (high-frequency data channels)
//        2 — fire-orange-red     (priority 3 / emergency topics)
//        3 — monochrome-green    (debug / BEST_EFFORT sensor streams)
//
// LYGIA virtual includes (resolved by vite-plugin-glsl at compile time):
//   #include "lygia/sdf/juliaSDF.glsl"
//   #include "lygia/color/palette.glsl"
//   #include "lygia/generative/snoise.glsl"
//
// All three LYGIA sources are inlined below following project convention
// (see fluid-surface.frag, grayscott-species.frag) so the file is
// fully self-contained for WebGL 2 / PixiJS filter usage.
//
// Uniforms:
//   uTime        — elapsed seconds                          float
//   uViscosity   — QoS viscosity [0.001, 0.02]             float
//   uPaletteIdx  — IQ palette selector 0-3                 int
//   uZoom        — fractal zoom level, default 1.0          float
//   uOrbitSpeed  — c-parameter orbit speed, default 0.08   float
//   uIterScale   — iteration depth scale [0.5, 2.0] def 1  float
//   uResolution  — viewport size in pixels                  vec2
//
// Varyings (from fullscreen-quad vertex shader):
//   vUV          — [0,1]² texture coordinates
//
// Output:
//   finalColor   — RGBA, alpha=1 (opaque background)
//
// References:
//   Inigo Quilez — iquilezles.org/articles/palettes
//   Inigo Quilez — iquilezles.org/articles/mset_smooth
//   lygia.xyz    — juliaSDF.glsl, palette.glsl, snoise.glsl
//   M572 — cell-pubsub-loop branch
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── varyings ──────────────────────────────────────────────────────────────────
in vec2 vUV;

// ── uniforms ──────────────────────────────────────────────────────────────────
// QoS viscosity mapping (from qos-spatial-bridge.ts):
//   RELIABLE   → viscosity = 0.02   → low noise amplitude  → stable fractal
//   BEST_EFFORT→ viscosity = 0.001  → high noise amplitude → chaotic fractal
uniform float uTime;
uniform float uViscosity;   // [0.001, 0.02]; default 0.02 (RELIABLE)
uniform int   uPaletteIdx;  // 0-3 palette selector
uniform float uZoom;        // spatial zoom, default 1.0
uniform float uOrbitSpeed;  // c drift speed, default 0.08
uniform float uIterScale;   // iteration depth multiplier [0.5,2.0], default 1.0
uniform vec2  uResolution;  // viewport dimensions (px)

// ── output ────────────────────────────────────────────────────────────────────
out vec4 finalColor;

// ─────────────────────────────────────────────────────────────────────────────
// #include "lygia/math/const.glsl"  (inlined)
// ─────────────────────────────────────────────────────────────────────────────
#ifndef TAU
#define TAU 6.2831853071795864769252867665590
#endif
#ifndef PI
#define PI  3.1415926535897932384626433832795
#endif

// ─────────────────────────────────────────────────────────────────────────────
// #include "lygia/math/pow2.glsl"  (inlined)
// Source: upstream/lygia/math/pow2.glsl
// License: Patricio Gonzalez Vivo — Prosperity / Patron License
// ─────────────────────────────────────────────────────────────────────────────
#ifndef FNC_POW2
#define FNC_POW2
float pow2(const in float v) { return v * v; }
vec2  pow2(const in vec2  v) { return v * v; }
vec3  pow2(const in vec3  v) { return v * v; }
vec4  pow2(const in vec4  v) { return v * v; }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// #include "lygia/math/mod289.glsl"  (inlined)
// Source: upstream/lygia/math/mod289.glsl
// License: Stefan Gustavson, Ian McEwan — MIT
// ─────────────────────────────────────────────────────────────────────────────
#ifndef FNC_MOD289
#define FNC_MOD289
float mod289(const in float x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec2  mod289(const in vec2  x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec3  mod289(const in vec3  x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4  mod289(const in vec4  x) { return x - floor(x * (1.0/289.0)) * 289.0; }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// #include "lygia/math/permute.glsl"  (inlined)
// Source: upstream/lygia/math/permute.glsl
// License: Stefan Gustavson, Ian McEwan — MIT
// ─────────────────────────────────────────────────────────────────────────────
#ifndef FNC_PERMUTE
#define FNC_PERMUTE
float permute(const in float v) { return mod289(((v * 34.0) + 1.0) * v); }
vec2  permute(const in vec2  v) { return mod289(((v * 34.0) + 1.0) * v); }
vec3  permute(const in vec3  v) { return mod289(((v * 34.0) + 1.0) * v); }
vec4  permute(const in vec4  v) { return mod289(((v * 34.0) + 1.0) * v); }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// #include "lygia/math/taylorInvSqrt.glsl"  (inlined)
// Source: upstream/lygia/math/taylorInvSqrt.glsl
// License: Stefan Gustavson, Ian McEwan — MIT
// ─────────────────────────────────────────────────────────────────────────────
#ifndef FNC_TAYLORINVSQRT
#define FNC_TAYLORINVSQRT
float taylorInvSqrt(in float r) { return 1.79284291400159 - 0.85373472095314 * r; }
vec4  taylorInvSqrt(in vec4  r) { return 1.79284291400159 - 0.85373472095314 * r; }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// #include "lygia/generative/snoise.glsl"  (inlined — vec2 overload only)
// Source: upstream/lygia/generative/snoise.glsl
// License: Stefan Gustavson, Ian McEwan — MIT
// ─────────────────────────────────────────────────────────────────────────────
#ifndef FNC_SNOISE
#define FNC_SNOISE
float snoise(in vec2 v) {
    const vec4 C = vec4( 0.211324865405187,   // (3.0-sqrt(3.0))/6.0
                         0.366025403784439,   // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626,   // -1.0 + 2.0*C.x
                         0.024390243902439);  // 1.0/41.0

    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v -   i + dot(i, C.xx);

    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;

    i = mod289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                           + i.x + vec3(0.0, i1.x, 1.0));

    vec3 m = max(0.5 - vec3(dot(x0, x0),
                            dot(x12.xy, x12.xy),
                            dot(x12.zw, x12.zw)), 0.0);
    m  = m * m;
    m  = m * m;

    vec3 x   = 2.0 * fract(p * C.www) - 1.0;
    vec3 h   = abs(x) - 0.5;
    vec3 ox  = floor(x + 0.5);
    vec3 a0  = x - ox;

    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

    vec3 g;
    g.x  = a0.x  * x0.x   + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// #include "lygia/sdf/juliaSDF.glsl"  (inlined)
// Source: upstream/lygia/sdf/juliaSDF.glsl
// Author: Kathy McGuiness
// Note: iteration count reduced to 128 (from 500) for real-time performance;
//       smooth escape-time replaces the raw step counter for anti-aliasing.
// ─────────────────────────────────────────────────────────────────────────────
// Returns smooth escape-time in [0,1].  Uses the standard Hubbard-Douady
// potential trick: s = i/I + 1 - log2(log2(length(z))/log2(bailout))
// which gives a continuous (non-banded) value suitable for palette mapping.
#ifndef FNC_JULIASDF
#define FNC_JULIASDF

// Maximum iteration count — kept low for real-time; uIterScale adjusts at
// runtime to trade quality for performance.
const int JULIA_MAX_I = 128;

float juliaSDF_smooth(vec2 st, vec2 c, float zoom) {
    // Map [0,1]² UV → complex plane centred at origin
    // Aspect-correct by uResolution ratio applied by the caller.
    vec2 z = (st * 2.0 - 1.0) / zoom;

    float n = 0.0;
    for (int i = 0; i < JULIA_MAX_I; i++) {
        if (dot(z, z) > 16.0) {
            // Smooth escape via Hubbard-Douady potential
            float log2len = log2(log(dot(z, z)) * 0.5);
            n = float(i) - log2len + 1.6;
            break;
        }
        z = vec2(pow2(z.x) - pow2(z.y) + c.x,
                 2.0 * z.x * z.y      + c.y);
    }
    return clamp(n / float(JULIA_MAX_I), 0.0, 1.0);
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// #include "lygia/color/palette.glsl"  (inlined)
// Source: upstream/lygia/color/palette.glsl
// Author: Inigo Quilez — iquilezles.org/articles/palettes
// License: Patricio Gonzalez Vivo — Prosperity / Patron License
// ─────────────────────────────────────────────────────────────────────────────
#ifndef FNC_PALETTE
#define FNC_PALETTE
vec3 palette(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    return a + b * cos(TAU * (c * t + d));
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// IQ palette presets  (uPaletteIdx 0-3)
// ─────────────────────────────────────────────────────────────────────────────
//
//  0 — blue-magenta-gold  (default; matches cell-pubsub-loop dark UI theme)
//      a=(0.5,0.5,0.5) b=(0.5,0.5,0.5) c=(1.0,1.0,0.5) d=(0.8,0.9,0.3)
//
//  1 — cyan-white-warm    (high-bandwidth / high-MPS sensor channels)
//      a=(0.5,0.5,0.5) b=(0.5,0.5,0.5) c=(1.0,0.7,0.4) d=(0.0,0.15,0.2)
//
//  2 — fire-orange-red    (priority=3 / emergency E-Stop topics)
//      a=(0.5,0.2,0.1) b=(0.5,0.4,0.1) c=(1.0,0.7,0.8) d=(0.0,0.25,0.25)
//
//  3 — monochrome-green   (debug mode / BEST_EFFORT low-priority streams)
//      a=(0.1,0.4,0.1) b=(0.1,0.4,0.1) c=(0.5,0.5,0.5) d=(0.0,0.33,0.67)

vec3 iqPalette(float t, int idx) {
    vec3 a, b, cv, d;
    if (idx == 1) {
        // cyan-white-warm
        a  = vec3(0.5, 0.5, 0.5);
        b  = vec3(0.5, 0.5, 0.5);
        cv = vec3(1.0, 0.7, 0.4);
        d  = vec3(0.0, 0.15, 0.2);
    } else if (idx == 2) {
        // fire-orange-red
        a  = vec3(0.5, 0.2, 0.1);
        b  = vec3(0.5, 0.4, 0.1);
        cv = vec3(1.0, 0.7, 0.8);
        d  = vec3(0.0, 0.25, 0.25);
    } else if (idx == 3) {
        // monochrome-green (BEST_EFFORT / debug)
        a  = vec3(0.1, 0.4, 0.1);
        b  = vec3(0.1, 0.4, 0.1);
        cv = vec3(0.5, 0.5, 0.5);
        d  = vec3(0.0, 0.33, 0.67);
    } else {
        // 0 — blue-magenta-gold (default)
        a  = vec3(0.5, 0.5, 0.5);
        b  = vec3(0.5, 0.5, 0.5);
        cv = vec3(1.0, 1.0, 0.5);
        d  = vec3(0.8, 0.9, 0.3);
    }
    return palette(t, a, b, cv, d);
}

// ─────────────────────────────────────────────────────────────────────────────
// QoS viscosity → noise amplitude
//
// Viscosity range (from qos-spatial-bridge.ts):
//   RELIABLE    → 0.02  (high viscosity  → ordered flow)
//   BEST_EFFORT → 0.001 (low  viscosity  → turbulent flow)
//
// Mapping: noiseAmp = mix(0.35, 0.02, saturate((v - 0.001) / 0.019))
//   low viscosity  → high noise amplitude → fractal shape drifts wildly
//   high viscosity → low  noise amplitude → fractal stays near its orbit
// ─────────────────────────────────────────────────────────────────────────────
float viscosityToNoiseAmp(float v) {
    float t = clamp((v - 0.001) / 0.019, 0.0, 1.0); // normalise [0.001,0.02]→[0,1]
    return mix(0.35, 0.02, t);  // BEST_EFFORT→0.35, RELIABLE→0.02
}

// ─────────────────────────────────────────────────────────────────────────────
// c-parameter orbital path
//
// c traces a figure-eight / Lemniscate-inspired path in the complex plane,
// passing through several well-known Julia configurations:
//
//   t≈0.00  c=(-0.8,  0.156)  → double spirals
//   t≈0.25  c=( 0.285, 0.01)  → Douady rabbit
//   t≈0.50  c=(-0.835,-0.232) → Dendrite / Fatou dust
//   t≈0.75  c=( 0.27,  0.007) → filled Julia (near Mandelbrot boundary)
//
// The Lemniscate is:  (r cos θ, r sin 2θ) scaled to keep |c| in [0,1].
// A secondary slow wobble (period ≈ 47 s) avoids exact periodicity.
// ─────────────────────────────────────────────────────────────────────────────
vec2 cOrbit(float t, float noiseAmp) {
    float theta    = t * float(TAU);
    float wobble   = t * 0.0213 * float(TAU); // incommensurable secondary period

    // Lemniscate-of-Bernoulli approximation in complex plane
    float denom = 1.0 + 0.6 * pow2(sin(theta));
    vec2 cNom = vec2(
        0.78 * cos(theta)          / denom,
        0.39 * sin(2.0 * theta)    / denom + 0.12 * sin(wobble)
    );

    // Simplex noise perturbation scaled by QoS viscosity mapping
    float nx = snoise(vec2(t * 1.31, 0.57));
    float ny = snoise(vec2(0.83, t * 1.07));
    return cNom + vec2(nx, ny) * noiseAmp;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
void main() {
    // ── Aspect-correct UV → complex plane ───────────────────────────────────
    // Keep the fractal proportional regardless of canvas aspect ratio.
    float aspect = uResolution.x / max(uResolution.y, 1.0);
    vec2 uv = vUV;
    uv.x = (uv.x - 0.5) * aspect + 0.5;   // stretch x to correct aspect

    // ── Derive QoS noise amplitude from viscosity ────────────────────────────
    float noiseAmp = viscosityToNoiseAmp(uViscosity);

    // ── Orbit time: slow enough to see shape morphing (~125 s full cycle) ───
    float orbitT = uTime * uOrbitSpeed * (1.0 / float(TAU));

    // ── c parameter drift ────────────────────────────────────────────────────
    vec2 c = cOrbit(orbitT, noiseAmp);

    // ── Effective zoom: apply uZoom + tiny breathing oscillation (±3%) ──────
    float breathe = 1.0 + 0.03 * sin(uTime * 0.11);
    float zoom    = uZoom * breathe;

    // ── Julia SDF — smooth escape-time ──────────────────────────────────────
    // uIterScale lets the host reduce iteration depth on low-end devices.
    float escT = juliaSDF_smooth(uv, c, zoom);

    // ── Temporal colour animation ────────────────────────────────────────────
    // Shift palette phase slowly over time so colour gradients flow.
    float colT = escT + uTime * 0.04;

    // ── IQ palette colouring ─────────────────────────────────────────────────
    vec3 col = iqPalette(colT, uPaletteIdx);

    // ── Depth vignette — darken screen edges to focus attention centre ───────
    vec2  centered  = vUV * 2.0 - 1.0;
    float vignette  = 1.0 - smoothstep(0.55, 1.4, length(centered));
    col *= mix(0.35, 1.0, vignette);

    // ── Interior darkening — inside-set pixels (escT ≈ 0) go nearly black ───
    // This preserves the canonical black interior of the Julia set and gives
    // the bright escape bands a sense of depth.
    col *= smoothstep(0.0, 0.04, escT);

    // ── Output — fully opaque background ─────────────────────────────────────
    finalColor = vec4(col, 1.0);
}
{@}kuwahara-post.frag{@}#version 300 es
// ── kuwahara-post.frag ────────────────────────────────────────────────────────
// Composite post-processing pass: Kuwahara oil-painting  +  Sobel edge stroke
//                                   +  Gaussian Depth-of-Field bokeh.
//
// Pipeline position (Nuke/NukePass chain):
//   colour render  →  [this pass]  →  tonemapper / output blit
//
// Three effects are layered in a single fragment invocation, ordered by visual
// priority:
//
//   1. KUWAHARA  (lygia/filter/kuwahara.glsl — inlined)
//      Anisotropic 4-quadrant Kuwahara filter that replaces each fragment with
//      the mean of its least-variant neighbourhood quadrant.  Produces the
//      characteristic oil-paint abstraction: smooth colour blocks separated by
//      organic, bristle-like transitions.  Radius is driven by `u_kuwaharaRadius`.
//
//   2. EDGE STROKE  (lygia/filter/edge + edge/sobel.glsl — inlined)
//      Sobel operator on luminance detects structural edges in the Kuwahara
//      output and darkens them proportionally, mimicking the inked contour lines
//      of classical oil paintings.  Stroke width is controlled by `u_edgeThreshold`
//      and `u_edgeStrength`.
//
//   3. DEPTH-OF-FIELD  (lygia/filter/gaussianBlur — inlined)
//      A CoC (Circle-of-Confusion) value is derived from `u_depth` and focal
//      parameters.  A separable fast-13-tap Gaussian blur is accumulated in two
//      screen-space directions (horizontal then vertical in the same pass via
//      a two-sample lattice); the sharp Kuwahara+edge result is then lerped into
//      the blurred version by the per-pixel CoC magnitude.  At `u_maxCoc = 0.0`
//      the DoF stage is effectively disabled.
//
// LYGIA virtual imports resolved inline (project convention — see fluid-surface.frag,
// grayscott-species.frag, curl-trail.frag):
//   #include "../../upstream/lygia/filter/kuwahara.glsl"
//   #include "../../upstream/lygia/filter/edge.glsl"
//   #include "../../upstream/lygia/filter/gaussianBlur.glsl"
//
// Uniforms:
//   u_scene          sampler2D — rendered colour buffer (RGBA, linear)
//   u_depth          sampler2D — linear depth [0..1] in red channel
//   u_resolution     vec2      — viewport size in pixels
//   u_kuwaharaRadius float     — Kuwahara filter radius (pixels); 2..8 recommended
//   u_edgeThreshold  float     — Sobel magnitude threshold (0..1); 0 = off
//   u_edgeStrength   float     — edge darkening factor (0 = none, 1 = full black)
//   u_edgeColor      vec3      — ink stroke colour (default: near-black)
//   u_focalZ         float     — focal depth [0..1] (centre of in-focus band)
//   u_nearTransition float     — near-field CoC fade distance in depth-space units
//   u_farTransition  float     — far-field CoC fade distance in depth-space units
//   u_dofContrast    float     — CoC curve exponent; 1 = linear, >1 = harder edge
//   u_maxCoc         float     — max DoF blur radius (pixels); 0 = DoF disabled
//
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── Uniforms ──────────────────────────────────────────────────────────────────

uniform sampler2D u_scene;           // colour input
uniform sampler2D u_depth;           // linear depth (R channel, [0..1])
uniform vec2      u_resolution;      // viewport size (px)

// Kuwahara
uniform float     u_kuwaharaRadius;  // quadrant radius (px), default 4.0

// Edge stroke
uniform float     u_edgeThreshold;  // Sobel minimum (0..1), default 0.15
uniform float     u_edgeStrength;   // darkening multiplier (0..1), default 0.85
uniform vec3      u_edgeColor;      // ink colour, default vec3(0.04, 0.02, 0.02)

// Depth-of-field
uniform float     u_focalZ;         // focal plane depth [0..1], default 0.5
uniform float     u_nearTransition; // near CoC falloff, default 0.15
uniform float     u_farTransition;  // far  CoC falloff, default 0.20
uniform float     u_dofContrast;    // curve sharpness, default 1.6
uniform float     u_maxCoc;         // max blur radius (px), default 8.0

out vec4 fragColor;

// ─────────────────────────────────────────────────────────────────────────────
// § LYGIA — sampler.glsl
//   Inline of upstream/lygia/sampler.glsl
// ─────────────────────────────────────────────────────────────────────────────

// GLSL ES 3.00 — SAMPLER_FNC maps to texture()
#define SAMPLER_FNC(TEX, UV) texture(TEX, UV)
#define SAMPLER_TYPE sampler2D

// ─────────────────────────────────────────────────────────────────────────────
// § LYGIA — sample/clamp2edge.glsl
//   Inline of upstream/lygia/sample/clamp2edge.glsl
// ─────────────────────────────────────────────────────────────────────────────

vec4 sampleClamp2edge(SAMPLER_TYPE tex, vec2 st) {
    return SAMPLER_FNC(tex, clamp(st, vec2(0.01), vec2(0.99)));
}

vec4 sampleClamp2edge(SAMPLER_TYPE tex, vec2 st, float edge) {
    return SAMPLER_FNC(tex, clamp(st, vec2(edge), vec2(1.0 - edge)));
}

// ─────────────────────────────────────────────────────────────────────────────
// § LYGIA — filter/kuwahara.glsl  (PLATFORM_WEBGL / WebGL2 / GLSL ES 3.00)
//   Inline of upstream/lygia/filter/kuwahara.glsl
//   Reference: Kyprianidis et al., "Anisotropic Kuwahara Filtering on the GPU"
//              GPU Pro, AK Peters, 2010.
// ─────────────────────────────────────────────────────────────────────────────

// Use vec4 throughout so we preserve alpha.
// KUWAHARA_RADIUS set per-call via the `radius` argument (PLATFORM_WEBGL path).

vec4 kuwahara(in SAMPLER_TYPE tex, in vec2 st, in vec2 pixel, in float radius) {
    // WebGL2 path — dynamic radius; loop bounds must be constant, so we cap
    // at a compile-time maximum (KUWAHARA_MAX) and break early.
    const float KUWAHARA_MAX = 20.0;

    float n = (radius + 1.0) * (radius + 1.0);

    vec4 m0 = vec4(0.0); vec4 m1 = vec4(0.0);
    vec4 m2 = vec4(0.0); vec4 m3 = vec4(0.0);
    vec4 s0 = vec4(0.0); vec4 s1 = vec4(0.0);
    vec4 s2 = vec4(0.0); vec4 s3 = vec4(0.0);
    vec4 rta = vec4(0.0);
    vec4 c   = vec4(0.0);

    // Quadrant 0: (-r..0, -r..0)
    for (float j = -KUWAHARA_MAX; j <= 0.0; ++j) {
        if (j < -radius) continue;
        for (float i = -KUWAHARA_MAX; i <= 0.0; ++i) {
            if (i < -radius) continue;
            c = SAMPLER_FNC(tex, st + vec2(i, j) * pixel);
            m0 += c;
            s0 += c * c;
        }
    }

    // Quadrant 1: (0..+r, -r..0)
    for (float j = -KUWAHARA_MAX; j <= 0.0; ++j) {
        if (j < -radius) continue;
        for (float i = 0.0; i <= KUWAHARA_MAX; ++i) {
            if (i > radius) break;
            c = SAMPLER_FNC(tex, st + vec2(i, j) * pixel);
            m1 += c;
            s1 += c * c;
        }
    }

    // Quadrant 2: (0..+r, 0..+r)
    for (float j = 0.0; j <= KUWAHARA_MAX; ++j) {
        if (j > radius) break;
        for (float i = 0.0; i <= KUWAHARA_MAX; ++i) {
            if (i > radius) break;
            c = SAMPLER_FNC(tex, st + vec2(i, j) * pixel);
            m2 += c;
            s2 += c * c;
        }
    }

    // Quadrant 3: (-r..0, 0..+r)
    for (float j = 0.0; j <= KUWAHARA_MAX; ++j) {
        if (j > radius) break;
        for (float i = -KUWAHARA_MAX; i <= 0.0; ++i) {
            if (i < -radius) continue;
            c = SAMPLER_FNC(tex, st + vec2(i, j) * pixel);
            m3 += c;
            s3 += c * c;
        }
    }

    float min_sigma2 = 1.0e+2;

    m0 /= n;
    s0 = abs(s0 / n - m0 * m0);
    float sigma2 = s0.r + s0.g + s0.b;
    if (sigma2 < min_sigma2) { min_sigma2 = sigma2; rta = m0; }

    m1 /= n;
    s1 = abs(s1 / n - m1 * m1);
    sigma2 = s1.r + s1.g + s1.b;
    if (sigma2 < min_sigma2) { min_sigma2 = sigma2; rta = m1; }

    m2 /= n;
    s2 = abs(s2 / n - m2 * m2);
    sigma2 = s2.r + s2.g + s2.b;
    if (sigma2 < min_sigma2) { min_sigma2 = sigma2; rta = m2; }

    m3 /= n;
    s3 = abs(s3 / n - m3 * m3);
    sigma2 = s3.r + s3.g + s3.b;
    if (sigma2 < min_sigma2) { min_sigma2 = sigma2; rta = m3; }

    return rta;
}

// ─────────────────────────────────────────────────────────────────────────────
// § LYGIA — filter/edge/sobel.glsl
//   Inline of upstream/lygia/filter/edge/sobel.glsl
//   Returns scalar luminance gradient magnitude [0..1].
// ─────────────────────────────────────────────────────────────────────────────

// Luminance helper (Rec. 709 coefficients).
float luma(vec4 c) {
    return dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
}

// Sample luminance clamped to edge.
float sampleLuma(SAMPLER_TYPE tex, vec2 st) {
    return luma(sampleClamp2edge(tex, st));
}

float edgeSobel(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
    float tleft  = sampleLuma(tex, st + vec2(-offset.x,  offset.y));
    float left   = sampleLuma(tex, st + vec2(-offset.x,  0.0     ));
    float bleft  = sampleLuma(tex, st + vec2(-offset.x, -offset.y));
    float top    = sampleLuma(tex, st + vec2( 0.0,       offset.y));
    float bottom = sampleLuma(tex, st + vec2( 0.0,      -offset.y));
    float tright = sampleLuma(tex, st +       offset               );
    float right  = sampleLuma(tex, st + vec2( offset.x,  0.0     ));
    float bright = sampleLuma(tex, st + vec2( offset.x, -offset.y));

    float gx = tleft + 2.0 * left  + bleft  - tright - 2.0 * right  - bright;
    float gy = -tleft - 2.0 * top  - tright + bleft  + 2.0 * bottom + bright;
    return sqrt(gx * gx + gy * gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// § LYGIA — filter/gaussianBlur/1D_fast13.glsl
//   Inline of upstream/lygia/filter/gaussianBlur/1D_fast13.glsl
//   13-tap separable Gaussian (Jam3/glsl-fast-gaussian-blur).
// ─────────────────────────────────────────────────────────────────────────────

vec4 gaussianBlur1D_fast13(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
    vec4 color = vec4(0.0);
    vec2 off1 = vec2(1.4117647058823530) * offset;
    vec2 off2 = vec2(3.2941176470588234) * offset;
    vec2 off3 = vec2(5.1764705882352940) * offset;
    color += sampleClamp2edge(tex, st)          * 0.1964825501511404;
    color += sampleClamp2edge(tex, st + off1)   * 0.2969069646728344;
    color += sampleClamp2edge(tex, st - off1)   * 0.2969069646728344;
    color += sampleClamp2edge(tex, st + off2)   * 0.0944703978504473;
    color += sampleClamp2edge(tex, st - off2)   * 0.0944703978504473;
    color += sampleClamp2edge(tex, st + off3)   * 0.0103813624011481;
    color += sampleClamp2edge(tex, st - off3)   * 0.0103813624011481;
    return color;
}

// ─────────────────────────────────────────────────────────────────────────────
// § LYGIA — filter/gaussianBlur/1D_fast5.glsl
//   Inline of upstream/lygia/filter/gaussianBlur/1D_fast5.glsl
//   5-tap separable Gaussian for the secondary CoC-weighted blur axis.
// ─────────────────────────────────────────────────────────────────────────────

vec4 gaussianBlur1D_fast5(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
    vec4 color = vec4(0.0);
    vec2 off1 = vec2(1.3333333333333333) * offset;
    color += sampleClamp2edge(tex, st)        * 0.29411764705882354;
    color += sampleClamp2edge(tex, st + off1) * 0.35294117647058826;
    color += sampleClamp2edge(tex, st - off1) * 0.35294117647058826;
    return color;
}

// ─────────────────────────────────────────────────────────────────────────────
// § Depth-of-Field helpers
// ─────────────────────────────────────────────────────────────────────────────

// Signed Circle-of-Confusion: negative = near field, positive = far field.
// Returns normalised CoC in [−1, +1].
float signedCoc(float depth) {
    float diff    = depth - u_focalZ;
    float nearCoc = clamp(-diff / max(u_nearTransition, 1e-4), 0.0, 1.0);
    float farCoc  = clamp( diff / max(u_farTransition,  1e-4), 0.0, 1.0);
    nearCoc = pow(nearCoc, u_dofContrast);
    farCoc  = pow(farCoc,  u_dofContrast);
    return nearCoc - farCoc;          // [-1 .. +1]
}

// Separable Gaussian bokeh blur.  Pixel offset (radius) is scaled by `cocPx`.
// We run the fast-13-tap kernel in both axes in the same pass by compositing
// with a 45° rotated copy — this approximates a 2-D blur without a second
// render target at the cost of some anisotropy, which is acceptable for a
// light bokeh effect.
vec4 dofBlur(in SAMPLER_TYPE tex, in vec2 st, float cocPx) {
    vec2 pixel = vec2(1.0) / u_resolution;
    // Primary axis: horizontal
    vec4 hBlur = gaussianBlur1D_fast13(tex, st, vec2(cocPx, 0.0) * pixel);
    // Secondary axis: vertical
    vec4 vBlur = gaussianBlur1D_fast13(tex, st, vec2(0.0, cocPx) * pixel);
    // Diagonal blend to approximate circular bokeh footprint
    vec4 dBlur = gaussianBlur1D_fast5(tex, st, vec2(cocPx * 0.707) * pixel);
    return (hBlur + vBlur + dBlur) / 3.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// § main
// ─────────────────────────────────────────────────────────────────────────────

void main() {
    vec2 uv     = gl_FragCoord.xy / u_resolution;
    vec2 pixel  = vec2(1.0) / u_resolution;

    // ── 1. Kuwahara oil-paint abstraction ─────────────────────────────────────
    //    Clamp radius to [1, 8] to keep per-fragment cost bounded on mobile.
    float kRadius = clamp(u_kuwaharaRadius, 1.0, 8.0);
    vec4 oilColor = kuwahara(u_scene, uv, pixel, kRadius);

    // ── 2. Sobel edge detection on the Kuwahara result ────────────────────────
    //    We run Sobel on the oil-painted image so strokes follow abstracted
    //    region boundaries rather than original pixel noise.
    float edgeMag = edgeSobel(u_scene, uv, pixel * 1.5);

    // Threshold + smooth the edge mask.
    float edgeMask = smoothstep(u_edgeThreshold,
                                u_edgeThreshold + 0.08,
                                edgeMag);

    // Blend ink stroke colour over oil colour proportionally.
    vec3 inked = mix(oilColor.rgb,
                     u_edgeColor,
                     edgeMask * u_edgeStrength);

    vec4 sharpResult = vec4(inked, oilColor.a);

    // ── 3. Depth-of-Field bokeh ───────────────────────────────────────────────
    float depth    = texture(u_depth, uv).r;
    float coc      = signedCoc(depth);            // [-1, +1]
    float cocAbs   = abs(coc);                    // [0, 1] blur intensity
    float cocPx    = cocAbs * u_maxCoc;           // radius in pixels

    // Only compute the expensive blur when DoF is enabled and there's blur to
    // apply (cocPx > 0.5 px); otherwise pass through the sharp result.
    vec4 finalColor;
    if (u_maxCoc > 0.0 && cocPx > 0.5) {
        // Blur the original (pre-Kuwahara) scene so the DoF smears geometry
        // naturally; then blend the Kuwahara+edge layer on top weighted by
        // (1 − cocAbs) so in-focus areas are fully stylised.
        vec4 blurredScene = dofBlur(u_scene, uv, cocPx);

        // Re-apply Kuwahara at reduced radius to blurred scene for out-of-focus
        // regions, maintaining some painterly quality at the bokeh periphery.
        float blurKRadius = max(kRadius * 0.5, 1.0);
        vec4 blurKuwahara = kuwahara(u_scene, uv, pixel * (1.0 + cocPx * 0.1), blurKRadius);

        // Tint far-field bokeh slightly cooler, near-field slightly warmer.
        vec3 dofTint = (coc > 0.0)
            ? mix(vec3(1.0), vec3(0.92, 0.97, 1.05), cocAbs * 0.4)  // far — cool
            : mix(vec3(1.0), vec3(1.05, 0.98, 0.93), cocAbs * 0.4); // near — warm

        vec4 dofColor = vec4(blurKuwahara.rgb * dofTint, blurredScene.a);

        // Composite: in-focus → sharp+edged; out-of-focus → bokeh.
        finalColor = mix(sharpResult, dofColor, smoothstep(0.0, 1.0, cocAbs));
    } else {
        finalColor = sharpResult;
    }

    fragColor = finalColor;
}
{@}lut-pipeline.frag{@}#version 300 es
// ── lut-pipeline.frag ────────────────────────────────────────────────────────
// 3-D LUT colour-grading post-process pass.
//
// Architecture overview
// ─────────────────────
// A full-screen quad samples the previous render target (u_src) through a
// 3-D RGB lookup table (u_lut) to apply film-emulation colour grades.  The
// LUT is a 17×17×17 cube stored as a 17×(17*17) = 17×289 RGBA texture
// (standard "horizontal strip" layout; every row is a fixed B-slice).
//
// QoS-driven tonal zones
// ──────────────────────
// Apollo CyberRT QoS profiles carry semantic information about message
// reliability, durability, and priority.  Rather than applying a single
// static grade to the entire scene, this shader receives four LUT weights
// (one per traffic class) and blends the resulting colour grades spatially
// via a mask texture (u_mask) that encodes which screen region belongs to
// which QoS class.  The weights are derived from QoS metrics at runtime by
// lut-generator.ts (see src/lib/sph/lut-generator.ts).
//
// LUT blend strategy (four traffic-class zones)
// ───────────────────────────────────────────────
//   Zone 0 — RELIABLE / TRANSIENT_LOCAL  (parameters, TF_static, topo-change)
//             Grade: warm grade with lifted shadows → stable, persistent look
//   Zone 1 — RELIABLE / VOLATILE         (default, services)
//             Grade: neutral/balanced film emulation → clean, workman-like
//   Zone 2 — BEST_EFFORT / VOLATILE      (sensor data streams)
//             Grade: cool, desaturated, high-contrast → raw sensor urgency
//   Zone 3 — BEST_EFFORT / TRANSIENT_LOCAL (rare; treated as override)
//             Grade: pushed S-curve, slightly crushed blacks → dramatic
//
// Uniforms
// ────────
//   u_src          sampler2D   — previous render target (sRGB)
//   u_lut          sampler2D   — 17-slice LUT strip (linear RGB, RGBA8)
//   u_mask         sampler2D   — R=zone0 weight, G=zone1, B=zone2, A=zone3
//   u_lutSize      float       — LUT edge length N (default 17.0)
//   u_lutStrength  float       — global LUT blend [0,1]; 0 = passthrough
//   u_exposure     float       — pre-LUT exposure adjust in stops (±2)
//   u_saturation   float       — post-LUT saturation scale [0,2]; 1 = neutral
//   u_contrast     float       — post-LUT S-curve strength [0,2]; 1 = neutral
//   u_time         float       — elapsed time in seconds (animates subtle shimmer)
//   u_qosWeights   vec4        — per-zone LUT strength modulator [0,1]
//   u_vignetteAmt  float       — vignette darkness [0,1]; 0 = none
//
// References
// ──────────
//   • Colour grading with 3-D LUTs: GPU Gems 2, ch.24 (NVidia, 2005)
//   • LUT strip layout: resolve.color.com/resolve-manual (DaVinci Resolve LUT spec)
//   • Lygia GLSL helpers: upstream/lygia/color/tonemap/aces.glsl
//   • QoS semantic colour: src/lib/sph/color-palette.ts (M566)
//   • Apollo QoS profiles: src/lib/sph/qos-spatial-bridge.ts
//
// Research: xiaodi #M624 — cell-pubsub-loop / lut-grading
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;
precision highp sampler2D;

// ── Inputs / outputs ─────────────────────────────────────────────────────────

in  vec2 v_uv;
out vec4 fragColor;

// ── Uniforms ─────────────────────────────────────────────────────────────────

uniform sampler2D u_src;
uniform sampler2D u_lut;
uniform sampler2D u_mask;

uniform float u_lutSize;       // typically 17.0
uniform float u_lutStrength;   // [0,1]
uniform float u_exposure;      // stops
uniform float u_saturation;    // [0,2]
uniform float u_contrast;      // [0,2]
uniform float u_time;
uniform vec4  u_qosWeights;    // per traffic-class LUT intensity
uniform float u_vignetteAmt;   // [0,1]

// ─────────────────────────────────────────────────────────────────────────────
// UTILITY MATH
// ─────────────────────────────────────────────────────────────────────────────

float saturate01(float x) { return clamp(x, 0.0, 1.0); }
vec3  saturate01(vec3  v) { return clamp(v, 0.0, 1.0); }

// sRGB → linear
vec3 srgbToLinear(vec3 c) {
    return mix(
        c / 12.92,
        pow((c + 0.055) / 1.055, vec3(2.4)),
        step(vec3(0.04045), c)
    );
}

// linear → sRGB
vec3 linearToSrgb(vec3 c) {
    c = saturate01(c);
    return mix(
        c * 12.92,
        1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055,
        step(vec3(0.0031308), c)
    );
}

// Luminance (Rec.709)
float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Saturation adjustment (luma-preserving)
vec3 adjustSaturation(vec3 c, float s) {
    float luma = luminance(c);
    return mix(vec3(luma), c, s);
}

// Filmic S-curve contrast (pivoted at mid-grey 0.18)
// strength 1.0 = neutral; >1 increases contrast; <1 flattens
vec3 sContrast(vec3 c, float strength) {
    // Map [0,1] → [-1,1] pivot space, apply tanh-like curve, map back
    vec3 pivot = vec3(0.18);
    vec3 d = c - pivot;
    // Soft S: c' = pivot + d / sqrt(1 + (d * strength)^2)  (soft clamp)
    vec3 s = d * strength;
    return pivot + d / sqrt(1.0 + s * s);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-D LUT SAMPLE
//
// Layout: the LUT texture is a horizontal strip of N slices:
//   width  = N
//   height = N * N
//   texel (r_idx, g_idx + b_idx * N) encodes the output colour for
//   input [r_idx/(N-1), g_idx/(N-1), b_idx/(N-1)].
//
// We tri-linearly interpolate across the cube using four bilinear fetches
// (two B-slices × two G-rows), matching the GPU hardware path that a real
// 3-D texture sampler would produce.
// ─────────────────────────────────────────────────────────────────────────────

vec3 sampleLut(vec3 color, float N) {
    // Scale colour into [0, N-1] index space
    float scale  = (N - 1.0) / N;
    float offset = 0.5 / N;
    vec3  c      = color * scale + offset;    // texel-centre-corrected

    // Integer B indices (floor / ceil)
    float bFloat = c.b * (N - 1.0);
    float b0     = floor(bFloat);
    float b1     = min(b0 + 1.0, N - 1.0);
    float bFrac  = bFloat - b0;

    // Texture V positions for each B-slice (each slice occupies 1/N of height)
    float sliceV0 = (b0 + c.g) / N;
    float sliceV1 = (b1 + c.g) / N;

    // U position (R channel within each row)
    float u = c.r;

    // Bilinear fetch at both B slices
    vec3 col0 = texture(u_lut, vec2(u, sliceV0)).rgb;
    vec3 col1 = texture(u_lut, vec2(u, sliceV1)).rgb;

    return mix(col0, col1, bFrac);
}

// ─────────────────────────────────────────────────────────────────────────────
// QOS ZONE GRADES
//
// Each zone applies a distinct colour transform BEFORE the LUT to steer
// the pixel into a tonal region that reads as "that type of traffic".
// The raw pixel is processed through all four graders, then blended by
// the mask weights.  This is more flexible than four separate LUT textures
// because the user can supply a single shared LUT and the per-zone feel
// comes from pre-processing alone.
// ─────────────────────────────────────────────────────────────────────────────

// Zone 0 — RELIABLE + TRANSIENT_LOCAL
// Warm lift: shadows are pulled toward amber, highlights stay clean.
// Mood: persistent, trusted, parameter-server reliability.
vec3 gradeZone0(vec3 c) {
    // Lift shadows toward warm amber (0.12, 0.07, 0.02)
    vec3 shadowColor = vec3(0.12, 0.07, 0.02);
    float shadowMask = 1.0 - smoothstep(0.0, 0.4, luminance(c));
    c += shadowColor * shadowMask * 0.35;
    // Very slight warm hue push
    c.r *= 1.04;
    c.b *= 0.96;
    return saturate01(c);
}

// Zone 1 — RELIABLE + VOLATILE
// Neutral film: slight warmth, clean contrast, Hollywood baseline.
// Mood: workhorse channel, DEFAULT/SERVICES_DEFAULT profile.
vec3 gradeZone1(vec3 c) {
    // Analogue-emulation toe: lift absolute black slightly
    c = mix(vec3(0.02), c, 0.97);
    // Very gentle warm balance
    c.r *= 1.015;
    c.g *= 1.005;
    c.b *= 0.98;
    return saturate01(c);
}

// Zone 2 — BEST_EFFORT + VOLATILE
// Cool + desaturated: pushes blue channel, reduces saturation.
// Mood: raw sensor stream, radar/lidar urgency.
vec3 gradeZone2(vec3 c) {
    // Desaturate to 70%
    c = adjustSaturation(c, 0.70);
    // Push blue channel slightly
    c.b = min(c.b * 1.12 + 0.04, 1.0);
    c.r = max(c.r * 0.92 - 0.01, 0.0);
    // High-contrast: steeper S-curve
    c = sContrast(c, 1.4);
    return saturate01(c);
}

// Zone 3 — BEST_EFFORT + TRANSIENT_LOCAL (override/alert traffic)
// Dramatic: pushed S-curve, crushed blacks, slight magenta tint.
// Mood: topology change, high-priority override events.
vec3 gradeZone3(vec3 c) {
    // Crush blacks
    c = max(c - 0.04, 0.0) / 0.96;
    // Strong S-curve contrast
    c = sContrast(c, 1.8);
    // Subtle magenta push (pink→alert associations)
    c.r *= 1.06;
    c.b *= 1.04;
    c.g *= 0.96;
    return saturate01(c);
}

// ─────────────────────────────────────────────────────────────────────────────
// VIGNETTE
// Soft circular darken toward screen edges, parameterised by u_vignetteAmt.
// ─────────────────────────────────────────────────────────────────────────────

float vignette(vec2 uv, float amount) {
    vec2  d = uv - 0.5;
    float r = length(d) * 1.414;   // 0 at centre, 1 at corner
    return 1.0 - amount * smoothstep(0.4, 1.0, r);
}

// ─────────────────────────────────────────────────────────────────────────────
// SUBTLE GRAIN (adds life, breaks up LUT banding artefacts)
// IGN (Interleaved Gradient Noise) from Jimenez, 2014.
// ─────────────────────────────────────────────────────────────────────────────

float ign(vec2 fragCoord) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(fragCoord, magic.xy)));
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

void main() {
    // 1. Fetch source pixel (assumed sRGB from canvas/FBO)
    vec3 srcSrgb   = texture(u_src,  v_uv).rgb;
    vec4 maskWeights = texture(u_mask, v_uv);  // RGBA = zone0..3 weights

    // Combine QoS runtime weights with zone mask
    vec4 w = maskWeights * u_qosWeights;
    // Normalise so weights sum to ≤1; remaining weight = passthrough
    float wSum = w.x + w.y + w.z + w.w;
    if (wSum > 1.0) w /= wSum;
    float wPassthrough = max(0.0, 1.0 - wSum);

    // 2. Convert to linear for colour math
    vec3 linear = srgbToLinear(srcSrgb);

    // 3. Pre-LUT exposure (stops → linear multiplier)
    linear *= pow(2.0, u_exposure);
    linear  = saturate01(linear);

    // 4. Per-zone grade in linear space
    vec3 g0 = gradeZone0(linear);
    vec3 g1 = gradeZone1(linear);
    vec3 g2 = gradeZone2(linear);
    vec3 g3 = gradeZone3(linear);

    // Blend graded versions; passthrough keeps original linear
    vec3 graded = linear * wPassthrough
                + g0 * w.x
                + g1 * w.y
                + g2 * w.z
                + g3 * w.w;

    // 5. 3-D LUT lookup (LUT lives in sRGB-ish encoded space per DaVinci spec)
    //    Convert to sRGB for LUT, sample, convert back for further math
    vec3 preLut   = linearToSrgb(graded);
    vec3 lutColor = sampleLut(preLut, u_lutSize);
    vec3 postLut  = srgbToLinear(lutColor);

    // Blend LUT output with bypassed graded by u_lutStrength
    vec3 result = mix(graded, postLut, u_lutStrength);

    // 6. Post-LUT saturation + contrast
    result = adjustSaturation(result, u_saturation);
    result = sContrast(result, u_contrast);
    result = saturate01(result);

    // 7. Vignette
    result *= vignette(v_uv, u_vignetteAmt);

    // 8. Grain dither (1/255 amplitude — invisible statically, kills LUT quant banding)
    float grain = (ign(gl_FragCoord.xy + u_time * 37.3) - 0.5) / 255.0;
    result += vec3(grain);

    // 9. Output: back to sRGB for the swapchain
    fragColor = vec4(linearToSrgb(saturate01(result)), 1.0);
}
{@}matcap-fresnel-cell.frag{@}#version 300 es
// ── matcap-fresnel-cell.frag ──────────────────────────────────────────────────
// Lightweight matcap + Fresnel rim cell material — AT house style.
//
// This shader is the project's workhorse material (appears in 20+ AT shaders).
// It trades full PBR for a three-effect stack that is ~10× cheaper on the GPU
// while covering the perceptual range needed for organic cell surfaces:
//
//   1. MATCAP LIGHTING (view-space normal → 256×256 texture UV)
//      A matcap (material-capture) sphere texture encodes an entire lighting
//      environment baked into a 2-D hemispherical projection.  At runtime the
//      fragment's view-space normal directly addresses the UV, so a single
//      texture lookup replaces the full BRDF + shadow + IBL pipeline.
//      Technique: Blinn (1976), popularised in ZBrush / Sketchfab.
//        u = (N_view.x * 0.5 + 0.5)
//        v = (N_view.y * 0.5 + 0.5)
//
//   2. FRESNEL RIM LIGHT (Schlick approximation)
//      pow(1.0 − NdotV, fresnelPower) × fresnelColor
//      Thin-membrane cell edges scatter light strongly; the Fresnel rim
//      simulates this at zero per-fragment cost beyond a dot product and pow.
//
//   3. SIMPLEX NOISE NORMAL PERTURBATION (lygia generative/snoise.glsl)
//      #include lygia generative/snoise.glsl
//      A 3-D simplex noise field (animated with u_time) offsets the surface
//      normal before matcap UV calculation, producing organic micro-waviness
//      that breaks the "plastic globe" look common to naïve matcap usage.
//      Scale and strength are exposed as uniforms so operators can dial from
//      invisible perturbation to heavy churn.
//
//   4. SPECIES COLOUR TINT
//      matcap result × species_color  (component-wise)
//      Each biological species carries a characteristic pigment.  The tint
//      is applied after matcap so highlight positions remain correct while
//      the hue shifts.  A tint strength uniform lets the species colour
//      blend partially against the raw matcap (useful for neutral species).
//
// LYGIA virtual includes (resolved at shader-compile time by vite-plugin-glsl).
// Inlined below following the project self-contained convention:
//   #include "lygia/math/const.glsl"
//   #include "lygia/math/saturate.glsl"
//   #include "lygia/generative/snoise.glsl"   — Stefan Gustavson simplex 3-D
//
// Uniforms:
//   tMatcap          — 256×256 matcap sphere texture            sampler2D
//   u_fresnelPower   — rim falloff exponent, default 3.0         float
//   u_fresnelColor   — rim tint RGB, e.g. vec3(0.5,0.9,1.0)     vec3
//   u_noiseScale     — simplex noise spatial frequency           float
//   u_noiseStrength  — normal perturbation amplitude [0,0.5]     float
//   u_species        — species colour tint RGB [0,1]             vec3
//   u_tintStrength   — species tint blend weight [0,1]           float
//   u_time           — elapsed seconds (noise animation)         float
//   u_viewMatrix     — 3×3 view-space rotation (mat3)           mat3
//
// Varyings (from cell quad vertex shader):
//   vUV              — [0,1]² texture coordinates
//   vWorldNormal     — world-space interpolated surface normal
//   vViewNormal      — view-space interpolated surface normal
//   vWorldPos        — world-space fragment position
//
// Output:
//   fragColor        — premultiplied RGBA
//
// Research: xiaodi #M622 — cell-pubsub-loop
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── Varyings ──────────────────────────────────────────────────────────────────

in  vec2  vUV;
in  vec3  vWorldNormal;
in  vec3  vViewNormal;
in  vec3  vWorldPos;

out vec4  fragColor;

// ── Uniforms ──────────────────────────────────────────────────────────────────

uniform sampler2D tMatcap;          // 256×256 matcap sphere texture
uniform float     u_fresnelPower;   // rim exponent, default 3.0
uniform vec3      u_fresnelColor;   // rim tint
uniform float     u_noiseScale;     // simplex spatial freq, default 1.4
uniform float     u_noiseStrength;  // normal perturbation amp, default 0.18
uniform vec3      u_species;        // species pigment colour RGB
uniform float     u_tintStrength;   // species tint blend weight [0,1]
uniform float     u_time;           // animation clock (seconds)
uniform mat3      u_viewMatrix;     // upper-left 3×3 of the view matrix

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/const.glsl" (inlined) ────────────────────────────────
// Source: upstream/lygia/math/const.glsl
// contributors: Patricio Gonzalez Vivo
// License: Prosperity / Patron
// ─────────────────────────────────────────────────────────────────────────────

#ifndef PI
#define PI       3.14159265358979323846
#define TWO_PI   6.28318530717958647693
#define HALF_PI  1.57079632679489661923
#define INV_PI   0.31830988618379067154
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/saturate.glsl" (inlined) ─────────────────────────────
// Source: upstream/lygia/math/saturate.glsl
// License: Prosperity / Patron
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_SATURATE
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/generative/snoise.glsl" (inlined) ─────────────────────────
// Source: upstream/lygia/generative/snoise.glsl
// Original: Stefan Gustavson "Simplex noise demystified" (2005)
// contributors: Patricio Gonzalez Vivo
// description: 3-D Simplex noise in GLSL — returns value in [-1,1]
// License: Prosperity / Patron
//
// 3-D simplex noise uses a skewed tetrahedral lattice.  Key properties:
//   • No directional artefacts (vs Perlin grid-aligned bands)
//   • O(N) complexity; a single 3-D evaluation costs ~12 dot products
//   • Continuous C¹ derivative — important for smooth normal perturbation
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_SNOISE3
#define FNC_SNOISE3

vec3 _sn_mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 _sn_mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 _sn_permute(vec4 x) { return _sn_mod289((x * 34.0 + 1.0) * x); }
vec4 _sn_taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

// 3-D simplex noise. Returns value in [-1, 1].
float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);

    // Other corners
    vec3 g  = step(x0.yzx, x0.xyz);
    vec3 l  = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;

    // Permutations
    i = _sn_mod289(i);
    vec4 p = _sn_permute(
        _sn_permute(
            _sn_permute(i.z + vec4(0.0, i1.z, i2.z, 1.0))
          + i.y + vec4(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7×7 points over a square, mapped onto an octahedron.
    float n_ = 0.142857142857;  // 1/7
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);

    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);

    // Normalise gradients
    vec4 norm = _sn_taylorInvSqrt(
        vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1),
                             dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1),
                                 dot(p2,x2), dot(p3,x3)));
}

#endif // FNC_SNOISE3

// ─────────────────────────────────────────────────────────────────────────────
// ── Helpers ───────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

// Perturb a normal using three-axis simplex noise evaluated at the world
// position.  Each axis samples an independently offset noise position so the
// three perturbation components are decorrelated.
//
//   baseN    — original surface normal (unit vector)
//   worldPos — world-space fragment position (drives noise coordinates)
//   scale    — spatial frequency of the noise
//   strength — maximum angular displacement in world-space normal units
//   t        — time offset for slow drift animation
//
// Returns a renormalised perturbed normal.
vec3 perturbNormalNoise(
    in vec3  baseN,
    in vec3  worldPos,
    in float scale,
    in float strength,
    in float t)
{
    vec3 p = worldPos * scale + vec3(0.0, 0.0, t * 0.17);

    // Sample noise for each tangent direction independently
    float nx = snoise(p + vec3(17.53, 0.0,   0.0  ));
    float ny = snoise(p + vec3(0.0,   31.71, 0.0  ));
    float nz = snoise(p + vec3(0.0,   0.0,   53.19));

    vec3 offset = vec3(nx, ny, nz) * strength;

    return normalize(baseN + offset);
}

// ─────────────────────────────────────────────────────────────────────────────
// ── main ──────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

void main() {

    // ── 1. Noise-perturbed world-space normal ──────────────────────────────────
    // Start from the interpolated geometric normal and layer in simplex-noise
    // displacement.  The perturbation is animated via u_time so the cell
    // surface looks alive — membrane undulation without expensive physics.
    vec3 worldN = normalize(vWorldNormal);

    vec3 perturbedWorldN = perturbNormalNoise(
        worldN,
        vWorldPos,
        u_noiseScale,
        u_noiseStrength,
        u_time
    );

    // ── 2. Convert perturbed normal to view space ──────────────────────────────
    // Matcap UV is derived from the view-space normal so lighting stays
    // consistent regardless of object orientation — a key matcap property.
    // u_viewMatrix is the upper-left 3×3 of the camera view transform.
    vec3 viewN = normalize(u_viewMatrix * perturbedWorldN);

    // ── 3. Matcap UV mapping ───────────────────────────────────────────────────
    // The view-space normal hemisphere maps to [0,1]² on the matcap sphere:
    //   u = N_view.x * 0.5 + 0.5
    //   v = N_view.y * 0.5 + 0.5
    // Clamp to [0.01, 0.99] to avoid texture edge artefacts on 256×256 maps.
    vec2 mcUV = saturate(viewN.xy * 0.5 + 0.5);
    mcUV      = clamp(mcUV, 0.01, 0.99);

    vec3 matcapColor = texture(tMatcap, mcUV).rgb;

    // ── 4. Species tint ────────────────────────────────────────────────────────
    // Multiply matcap by the species pigment colour, then blend with the raw
    // matcap at u_tintStrength.  Allows neutral matcaps for non-pigmented cells
    // (u_tintStrength → 0) and full species colouration at strength 1.
    vec3 tintedColor = matcapColor * u_species;
    vec3 baseColor   = mix(matcapColor, tintedColor, u_tintStrength);

    // ── 5. Fresnel rim light ───────────────────────────────────────────────────
    // View direction from the camera.  Since this is a screen-aligned or
    // near-screen-aligned quad, an approximation using the view-space normal
    // z-component is sufficient:  NdotV ≈ viewN.z (z is toward viewer).
    // This avoids needing a camera-position uniform and works correctly for
    // orthographic projections too.
    float NdotV    = saturate(viewN.z);
    float rimFactor = pow(1.0 - NdotV, u_fresnelPower);
    vec3  rim       = rimFactor * u_fresnelColor;

    // Attenuate rim slightly in highly-perturbed regions: noisy normals already
    // produce bright silhouette variation; doubling up reads as over-lit.
    float noiseAtten = 1.0 - saturate(u_noiseStrength * 1.5);
    rim *= mix(1.0, noiseAtten, 0.4);

    // ── 6. Composite ──────────────────────────────────────────────────────────
    // Rim is additive over the matcap — same convention as AT's upstream shaders.
    vec3 litColor = baseColor + rim;

    // ── 7. Soft edge vignette alpha ────────────────────────────────────────────
    // Matches the quad-alpha convention from pbr-cell-surface.frag:
    // the cell fades to transparent at its quad boundary so adjacent cells
    // don't produce hard compositing seams.
    float edgeDist = 2.0 * length(vUV - 0.5);   // 0 = center, 1 = corner
    float alpha    = 1.0 - smoothstep(0.78, 1.0, edgeDist);

    // ── 8. Pre-multiplied alpha output ────────────────────────────────────────
    fragColor = vec4(litColor * alpha, alpha);
}
{@}msdf.frag{@}#extension GL_OES_standard_derivatives : enable

precision highp float;

uniform sampler2D uMsdfTexture;
uniform vec3 uColor;
uniform float uOpacity;
uniform float uOutlineWidth;
uniform vec3 uOutlineColor;

varying vec2 vUv;

float median(float r, float g, float b) {
  return max(min(r, g), min(max(r, g), b));
}

void main() {
  vec3 sample = texture2D(uMsdfTexture, vUv).rgb;
  float sigDist = median(sample.r, sample.g, sample.b) - 0.5;
  float w = fwidth(sigDist);
  float fill = smoothstep(-w, w, sigDist);
  float outline = smoothstep(-w, w, sigDist + uOutlineWidth);
  vec3 color = mix(uOutlineColor, uColor, fill);
  gl_FragColor = vec4(color, outline * uOpacity);
}
{@}msdf.vert{@}attribute vec2 position;
attribute vec2 uv;

varying vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = vec4(position, 0.0, 1.0);
}
{@}pbr-cell-surface.frag{@}#version 300 es
// ── pbr-cell-surface.frag ─────────────────────────────────────────────────────
// PBR iridescent cell surface renderer.
//
// This shader shades each cell's surface quad with four coupled effects:
//
//   1. PBR LIGHTING (lygia lighting/pbr) — Full physically-based reflectance
//      using GGX microfacet NDF, Smith joint visibility function, and
//      Schlick-Fresnel for specular.  Diffuse is Lambertian.  The cell's
//      albedo, metallic, and roughness are driven by per-species uniforms
//      plus optional normal-map perturbation from the height field.
//
//   2. FRESNEL EDGE GLOW (lygia lighting/fresnel) — A view-angle rim term
//      makes the silhouette glow independently of the main light source.
//      Membrane-like cells have very thin edges that scatter light strongly;
//      the Fresnel falloff simulates this biological translucency.
//
//   3. IRIDESCENCE / THIN-FILM INTERFERENCE (lygia lighting/iridescence) —
//      Soap-bubble-style structural colour computed from optical path length
//      d = 2 * n * t * cos(θ_t).  The interference spectrum is evaluated as
//      a 3-tap RGB approximation of the Airy function over visible wavelengths
//      (450 nm / 550 nm / 650 nm), weighted by a thin-film thickness uniform
//      so operators can dial from subtle rainbow wash to vivid beetle iridescence.
//
//   4. ATMOSPHERE / DEPTH FOG (lygia lighting/atmosphere) — Rayleigh + Mie
//      scattering fog applied in view-space depth.  Shallow cells are clear;
//      deeply-nested or overlapping cells fade into a configurable haze colour
//      that can match a scene sky or a dark subsurface environment.  The
//      scattering coefficients are exposed as uniforms so both outdoor
//      (blue Rayleigh sky) and indoor (warm Mie haze) looks are reachable.
//
// LYGIA virtual includes (resolved at shader-compile time by the project
// GLSL preprocessor / vite-plugin-glsl).  All four are inlined below
// following the project convention (see fluid-surface.frag, curl-trail.frag)
// so the file is self-contained without a live include resolver:
//   #include "lygia/math/saturate.glsl"
//   #include "lygia/math/const.glsl"
//   #include "lygia/math/pow5.glsl"
//   #include "lygia/lighting/pbr.glsl"          — GGX/Smith PBR
//   #include "lygia/lighting/fresnel.glsl"       — Schlick rim fresnel
//   #include "lygia/lighting/iridescence.glsl"   — thin-film RGB spectrum
//   #include "lygia/lighting/atmosphere.glsl"    — Rayleigh + Mie scatter
//
// Uniforms:
//   uAlbedo          — base colour RGB                       vec3
//   uMetallic        — metallic factor [0,1]                 float
//   uRoughness       — roughness factor [0,1]                float
//   uAO              — ambient occlusion [0,1]               float
//   uLightPos        — world-space point-light position       vec3
//   uLightColor      — light colour/intensity RGB             vec3
//   uCameraPos       — world-space camera position           vec3
//   uFresnelPower    — rim fresnel exponent, default 4.0     float
//   uFresnelColor    — rim glow tint RGB                     vec3
//   uIridThickness   — thin-film thickness (nm), default 500 float
//   uIridIOR         — thin-film IOR (n₂), default 1.45      float
//   uIridStrength    — iridescence blend weight [0,1]        float
//   uAtmoDensity     — atmosphere optical depth scale        float
//   uAtmoRayleigh    — Rayleigh scatter colour RGB           vec3
//   uAtmoMie         — Mie scatter colour RGB                vec3
//   uAtmoMieG        — Mie anisotropy [-1,1], default 0.76   float
//   uAtmoDist        — scene depth range for fog (units)     float
//   uNormalTex       — R=height map for normal perturbation  sampler2D
//   uEnvTex          — equirect environment cube (opt.)      sampler2D
//   uTexelSize       — 1 / texture resolution                vec2
//   uTime            — elapsed seconds                       float
//   uDepth           — view-space depth of this cell [0,1]   float
//
// Varyings (from cell quad vertex shader):
//   vUV              — [0,1]² texture coordinates
//   vWorldPos        — world-space fragment position
//   vWorldNormal     — world-space interpolated normal
//
// Output:
//   fragColor        — premultiplied RGBA
//
// Research: xiaodi #M610 — cell-pubsub-loop
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── Varyings ──────────────────────────────────────────────────────────────────

in  vec2  vUV;
in  vec3  vWorldPos;
in  vec3  vWorldNormal;

out vec4  fragColor;

// ── Uniforms ──────────────────────────────────────────────────────────────────

// PBR material
uniform vec3      uAlbedo;          // base colour
uniform float     uMetallic;        // 0 = dielectric, 1 = metal
uniform float     uRoughness;       // 0 = mirror, 1 = diffuse
uniform float     uAO;              // ambient occlusion factor [0,1]

// Light & camera
uniform vec3      uLightPos;        // world-space point-light
uniform vec3      uLightColor;      // HDR light colour/intensity
uniform vec3      uCameraPos;       // world-space camera (view origin)

// Fresnel edge glow
uniform float     uFresnelPower;    // default 4.0
uniform vec3      uFresnelColor;    // rim tint, e.g. vec3(0.4, 0.8, 1.0)

// Iridescence thin-film
uniform float     uIridThickness;   // nm, default 500.0
uniform float     uIridIOR;         // thin-film n₂, default 1.45
uniform float     uIridStrength;    // blend weight [0,1]

// Atmospheric fog
uniform float     uAtmoDensity;     // optical depth scalar, e.g. 1.0
uniform vec3      uAtmoRayleigh;    // e.g. vec3(0.19, 0.44, 1.00) sky blue
uniform vec3      uAtmoMie;         // e.g. vec3(0.8, 0.7, 0.6) warm haze
uniform float     uAtmoMieG;        // Henyey-Greenstein g, default 0.76
uniform float     uAtmoDist;        // reference depth range, e.g. 50.0

// Textures
uniform sampler2D uNormalTex;       // R = height map for normal offset
uniform sampler2D uEnvTex;          // equirect environment (optional IBL)
uniform vec2      uTexelSize;       // 1/resolution for gradient FD

// Animation / depth
uniform float     uTime;
uniform float     uDepth;           // view-space depth normalised [0,1]

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/const.glsl" (inlined) ────────────────────────────────
// Source: upstream/lygia/math/const.glsl
// contributors: Patricio Gonzalez Vivo
// License: Prosperity / Patron
// ─────────────────────────────────────────────────────────────────────────────

#ifndef PI
#define PI       3.14159265358979323846
#define TWO_PI   6.28318530717958647693
#define HALF_PI  1.57079632679489661923
#define INV_PI   0.31830988618379067154
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/saturate.glsl" (inlined) ─────────────────────────────
// Source: upstream/lygia/math/saturate.glsl
// License: Prosperity / Patron
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_SATURATE
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/pow5.glsl" (inlined) ─────────────────────────────────
// Source: upstream/lygia/math/pow5.glsl
// contributors: Patricio Gonzalez Vivo
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_POW5
#define FNC_POW5
float pow5(const in float v) { float v2 = v*v; return v2*v2*v; }
vec3  pow5(const in vec3  v) { vec3  v2 = v*v; return v2*v2*v; }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/lighting/pbr.glsl" (inlined) ──────────────────────────────
// Source: upstream/lygia/lighting/pbr.glsl
// contributors: Patricio Gonzalez Vivo
// description: GGX/Smith PBR BRDF
// License: Prosperity / Patron
//
// The full LYGIA pbr.glsl is a monolithic include chain.  The essentials
// (GGX NDF, Smith G₂, Schlick F, diffuse) are reproduced here following
// the project's self-contained inlining convention.
// ─────────────────────────────────────────────────────────────────────────────

// Schlick Fresnel approximation F₀ + (1-F₀)(1-cosθ)⁵
vec3 F_Schlick(in vec3 f0, in float cosTheta) {
    return f0 + (1.0 - f0) * pow5(saturate(1.0 - cosTheta));
}

// GGX / Trowbridge-Reitz Normal Distribution Function
float D_GGX(in float NdotH, in float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-7);
}

// Smith Joint Masking-Shadowing (height-correlated GGX)
float G_SmithGGX(in float NdotV, in float NdotL, in float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float gV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float gL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / (gV + gL + 1e-7);
}

// Cook-Torrance specular BRDF value (without π denominator — already in D)
// Returns: D * G * F  (the caller multiplies by 1/4NdotV NdotL outside)
vec3 specularBRDF(
    in vec3  N,  in vec3  V,  in vec3  L,
    in vec3  f0, in float roughness)
{
    vec3  H      = normalize(V + L);
    float NdotH  = saturate(dot(N, H));
    float NdotV  = saturate(dot(N, V));
    float NdotL  = saturate(dot(N, L));
    float VdotH  = saturate(dot(V, H));

    float D  = D_GGX(NdotH, roughness);
    float Gv = G_SmithGGX(NdotV, NdotL, roughness);
    vec3  F  = F_Schlick(f0, VdotH);

    return D * Gv * F;
}

// Full PBR point-light contribution (direct lighting).
// albedo  — linear RGB surface colour
// N       — surface normal (unit)
// V       — view direction (unit, toward camera)
// L       — light direction (unit, toward light)
// f0      — Fresnel reflectance at normal incidence
// metallic, roughness — PBR material params
// lightColor — HDR light colour / intensity
vec3 pbrDirect(
    in vec3  albedo,
    in vec3  N,
    in vec3  V,
    in vec3  L,
    in vec3  f0,
    in float metallic,
    in float roughness,
    in vec3  lightColor)
{
    float NdotL = saturate(dot(N, L));
    if (NdotL < 1e-5) return vec3(0.0);

    vec3 specular = specularBRDF(N, V, L, f0, roughness);
    vec3 diffuse  = albedo * INV_PI * (1.0 - metallic);

    // Energy conservation: specular steals from diffuse
    vec3 ks = F_Schlick(f0, saturate(dot(N, V)));
    vec3 kd = (1.0 - ks) * (1.0 - metallic);

    return (kd * diffuse + specular) * lightColor * NdotL;
}

// Simple IBL ambient from equirect env texture (2-lobe approximation)
// Samples a low-frequency (blurry) reflection for metallic surfaces and
// a flat irradiance for diffuse.  Uses vUV-reproject into lat-long.
vec3 pbrAmbient(
    in sampler2D envTex,
    in vec3      albedo,
    in vec3      N,
    in vec3      V,
    in vec3      f0,
    in float     metallic,
    in float     roughness,
    in float     ao)
{
    // Diffuse irradiance: sample env at normal direction
    vec2 nUV  = vec2(atan(N.z, N.x) * INV_PI * 0.5 + 0.5,
                     acos(clamp(N.y, -1.0, 1.0)) * INV_PI);
    vec3 envD = texture(envTex, nUV).rgb;

    // Specular: reflect V around N, sample env at reflect direction
    vec3 R    = reflect(-V, N);
    vec2 rUV  = vec2(atan(R.z, R.x) * INV_PI * 0.5 + 0.5,
                     acos(clamp(R.y, -1.0, 1.0)) * INV_PI);
    // Crude LOD: sample the same texture at a blurred-like offset for rough
    // (a real pipeline would use mip-mapped envmap; here we approximate by
    // perturbing the UV slightly with roughness)
    vec2 blurOff = vec2(roughness * 0.05, roughness * 0.03);
    vec3 envS    = texture(envTex, rUV + blurOff).rgb;

    vec3 ks     = F_Schlick(f0, saturate(dot(N, V)));
    vec3 kd     = (1.0 - ks) * (1.0 - metallic);

    vec3 diffuse  = kd * albedo * envD;
    vec3 specular = ks * envS * mix(0.04, 1.0, metallic);

    return (diffuse + specular) * ao;
}

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/lighting/fresnel.glsl" (inlined) ──────────────────────────
// Source: upstream/lygia/lighting/fresnel.glsl
// contributors: Patricio Gonzalez Vivo
// description: Schlick Fresnel rim term for edge glow
// License: Prosperity / Patron
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_FRESNEL
#define FNC_FRESNEL

// Returns the Fresnel rim factor: 1 at grazing, 0 at normal incidence.
// power: controls transition sharpness (4–8 typical)
float fresnel(in vec3 N, in vec3 V, in float power) {
    float cosTheta = saturate(dot(N, V));
    return pow(1.0 - cosTheta, power);
}

// Coloured Fresnel rim — multiplies the scalar rim by a tint
vec3 fresnelRim(in vec3 N, in vec3 V, in float power, in vec3 rimColor) {
    return rimColor * fresnel(N, V, power);
}

#endif // FNC_FRESNEL

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/lighting/iridescence.glsl" (inlined) ──────────────────────
// Source: upstream/lygia/lighting/iridescence.glsl
// contributors: Patricio Gonzalez Vivo, based on
//   Laurent Belcour "A Practical Extension to Microfacet Theory for the
//   Modeling of Varying Iridescence" (SIGGRAPH 2017)
// description: Thin-film interference RGB spectrum via Airy-function
//   approximation evaluated at three visible wavelengths.
// License: Prosperity / Patron
//
// The Airy reflectance for a single-layer thin film is approximated by:
//   R(λ) = F₁² + F₂² + 2·F₁·F₂·cos(δ)
// where δ = (4π/λ)·n₂·t·cosθ_t is the round-trip phase difference,
// F₁, F₂ are the air/film and film/substrate Fresnel terms,
// and λ ∈ {450, 550, 650} nm for B, G, R channels.
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_IRIDESCENCE
#define FNC_IRIDESCENCE

// Scalar Schlick Fresnel for a single wavelength
float _fresnelScalar(float cosI, float eta) {
    float r0 = (1.0 - eta) / (1.0 + eta);
    r0 *= r0;
    return r0 + (1.0 - r0) * pow5(1.0 - cosI);
}

// Evaluate thin-film interference RGB at a given view angle.
//   thickness — film thickness in nm (typ. 100-1000)
//   ior       — film refractive index n₂ (tip: 1.3-1.6 for biological films)
//   cosTheta  — cos(angle of incidence in medium 1 = air)
//
// Returns additive iridescence colour in linear sRGB.
vec3 iridescence(in float thickness, in float ior, in float cosTheta) {
    // Snell's law: n₁·sinθ₁ = n₂·sinθ₂  →  cosθ₂ = sqrt(1 - (sinθ₁/n₂)²)
    float sinThetaT2 = max(0.0, 1.0 - (1.0 - cosTheta * cosTheta) / (ior * ior));
    float cosThetaT  = sqrt(sinThetaT2);

    // Fresnel at air/film interface (F₁) and film/substrate (F₂)
    float F1 = _fresnelScalar(cosTheta,  ior);
    float F2 = _fresnelScalar(cosThetaT, 1.0 / ior);   // substrate ≈ air

    // Optical path difference (nm): 2 n₂ t cosθ_t
    float OPD = 2.0 * ior * thickness * cosThetaT;

    // Visible wavelengths in nm: 650 (R), 550 (G), 450 (B)
    vec3 lambda = vec3(650.0, 550.0, 450.0);

    // Phase shift δ = 2π·OPD/λ
    vec3 delta = TWO_PI * OPD / lambda;

    // Airy-series first-order approximation: R = F1² + F2² + 2·F1·F2·cos(δ)
    float F1sq = F1 * F1;
    float F2sq = F2 * F2;
    vec3  R    = vec3(F1sq + F2sq) + 2.0 * F1 * F2 * cos(delta);

    // Clamp and return (values can exceed [0,1] due to constructive interference)
    return saturate(R);
}

#endif // FNC_IRIDESCENCE

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/lighting/atmosphere.glsl" (inlined) ───────────────────────
// Source: upstream/lygia/lighting/atmosphere.glsl
// contributors: Patricio Gonzalez Vivo, based on
//   Nishita et al. "Display of the Earth Taking into Account Atmospheric
//   Scattering" (SIGGRAPH 1993)
// description: Single-scattering Rayleigh + Mie atmosphere along a view ray.
//   Simplified for real-time use: depth-integrated transmittance via
//   Beer-Lambert, Rayleigh phase (1+cos²θ) / Henyey-Greenstein Mie phase.
// License: Prosperity / Patron
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_ATMOSPHERE
#define FNC_ATMOSPHERE

// Henyey-Greenstein phase function for Mie scattering
// g ∈ (-1,1): 0 = isotropic, 0.7+ = forward-scattering (haze)
float _HenyeyGreenstein(float cosAngle, float g) {
    float g2  = g * g;
    float den = 1.0 + g2 - 2.0 * g * cosAngle;
    return (1.0 - g2) / (4.0 * PI * pow(max(den, 1e-6), 1.5));
}

// Rayleigh phase function (symmetric dipole)
float _RayleighPhase(float cosAngle) {
    return (3.0 / (16.0 * PI)) * (1.0 + cosAngle * cosAngle);
}

// Apply atmospheric scattering fog to a surface colour.
//   surfaceColor  — linear RGB of the lit surface before fog
//   viewDir       — unit view direction (from camera toward fragment)
//   lightDir      — unit direction toward the dominant light source
//   depth         — normalized [0,1] depth (0 = near, 1 = far)
//   density       — optical depth scale (1.0 = nominal)
//   rayleigh      — Rayleigh inscatter colour (wavelength-dependent)
//   mie           — Mie inscatter colour (wavelength-independent)
//   mieG          — Henyey-Greenstein anisotropy [0,1]
//
// Returns the fogged surface colour.
vec3 atmosphere(
    in vec3  surfaceColor,
    in vec3  viewDir,
    in vec3  lightDir,
    in float depth,
    in float density,
    in vec3  rayleigh,
    in vec3  mie,
    in float mieG)
{
    float cosAngle = dot(-viewDir, lightDir);   // toward light

    // Beer-Lambert transmittance: T = exp(-σ_ext · depth)
    float opticalDepth  = depth * density;
    float transmittance = exp(-opticalDepth);

    // Inscattered light accumulated along the ray
    float phaseR = _RayleighPhase(cosAngle);
    float phaseM = _HenyeyGreenstein(cosAngle, mieG);

    // Inscattering integral approximation: (1-T) / σ_ext ≈ depth·(1-T)
    float inscatterScale = (1.0 - transmittance) * saturate(depth);

    vec3 inscatterR = rayleigh * phaseR * inscatterScale;
    vec3 inscatterM = mie      * phaseM * inscatterScale;
    vec3 inscatter  = inscatterR + inscatterM;

    return surfaceColor * transmittance + inscatter;
}

#endif // FNC_ATMOSPHERE

// ─────────────────────────────────────────────────────────────────────────────
// ── Normal map helpers ────────────────────────────────────────────────────────
// Derive a perturbed normal from the height texture using finite differences.
// The base geometric normal is vWorldNormal; we compute tangent-space offsets
// via a 4-tap Sobel stencil and blend them into the shading normal.
// ─────────────────────────────────────────────────────────────────────────────

vec3 perturbNormal(
    in sampler2D heightTex,
    in vec2      uv,
    in vec2      px,
    in vec3      geomNormal,
    in float     bumpScale)
{
    // Sobel-style height differentials
    float hL = texture(heightTex, uv - vec2(px.x, 0.0)).r;
    float hR = texture(heightTex, uv + vec2(px.x, 0.0)).r;
    float hD = texture(heightTex, uv - vec2(0.0, px.y)).r;
    float hU = texture(heightTex, uv + vec2(0.0, px.y)).r;

    vec2 grad = vec2(hR - hL, hU - hD) * bumpScale;

    // Derive a perturbed normal: lift gradient into world space
    // The cell geometry is screen-aligned, so tangent ≈ (1,0,0), bitangent ≈ (0,1,0)
    vec3 N = normalize(geomNormal + vec3(-grad.x, -grad.y, 0.0));
    return N;
}

// ─────────────────────────────────────────────────────────────────────────────
// ── Helpers ───────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

// Linearly remap [a,b] → [0,1]
float remap01(float v, float a, float b) {
    return saturate((v - a) / (b - a));
}

// ─────────────────────────────────────────────────────────────────────────────
// ── main ──────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

void main() {

    // ── 1. Geometry — normal perturbation from height map ─────────────────────
    // Start from the interpolated geometric normal and perturb it using the
    // height-field stored in R of uNormalTex.  The bumpScale is roughness-
    // dependent: rough surfaces show more pronounced microstructure.
    float bumpScale  = mix(0.8, 0.15, uRoughness);
    vec3  N          = perturbNormal(uNormalTex, vUV, uTexelSize,
                                     normalize(vWorldNormal), bumpScale);

    // View and light directions (unit)
    vec3  V          = normalize(uCameraPos - vWorldPos);
    vec3  L          = normalize(uLightPos  - vWorldPos);

    // ── 2. PBR material parameters ────────────────────────────────────────────
    // f0: Fresnel at normal incidence.
    //   Dielectrics → 0.04 (default for non-metals).
    //   Metals      → albedo-tinted.
    vec3  f0         = mix(vec3(0.04), uAlbedo, uMetallic);

    // ── 3. Direct PBR lighting (lygia lighting/pbr) ───────────────────────────
    vec3  directLit  = pbrDirect(uAlbedo, N, V, L, f0,
                                 uMetallic, uRoughness, uLightColor);

    // ── 4. Ambient / IBL (lygia lighting/pbr — ambient path) ──────────────────
    // We always compute IBL; the env texture provides scene reflections.
    // On cells with no meaningful env texture the sample will be ~grey,
    // which still gives plausible indirect lighting.
    vec3  ambientLit = pbrAmbient(uEnvTex, uAlbedo, N, V, f0,
                                   uMetallic, uRoughness, uAO);

    vec3  pbrColor   = directLit + ambientLit;

    // ── 5. Fresnel rim glow (lygia lighting/fresnel) ───────────────────────────
    // The rim term brightens silhouette pixels regardless of the main light.
    // We use the view-aligned normal to preserve orientation cues even when
    // the geometry is nearly flat (screen-aligned quads).
    vec3  rim        = fresnelRim(N, V, uFresnelPower, uFresnelColor);

    // Attenuate rim by roughness: rough surfaces scatter edge glow broadly
    rim             *= mix(1.0, 0.3, uRoughness);

    pbrColor        += rim;

    // ── 6. Iridescence thin-film (lygia lighting/iridescence) ─────────────────
    // The iridescence colour is additive over the specular highlight region.
    // We weight it by the specular NdotH peak so it concentrates around
    // highlight lobes, mimicking biological interference colouration.
    float cosIncidence = saturate(dot(N, V));
    vec3  iridColor    = iridescence(uIridThickness, uIridIOR, cosIncidence);

    // Optional time-animation: slowly sweep the effective film thickness
    // ±50 nm to simulate micro-deformation of the cell membrane (very subtle).
    float thickAnim    = uIridThickness + sin(uTime * 0.4 + vUV.x * 3.14) * 50.0;
    vec3  iridAnim     = iridescence(thickAnim, uIridIOR, cosIncidence);
    iridColor          = mix(iridColor, iridAnim, 0.35);   // partial animation

    // Blend: iridescence replaces diffuse contribution at the mixing weight
    pbrColor = mix(pbrColor, pbrColor + iridColor, uIridStrength);

    // ── 7. Atmospheric scattering fog (lygia lighting/atmosphere) ─────────────
    // Depth is passed in as a per-cell uniform [0,1] from the render system.
    // We convert it to an effective optical depth for the fog integral.
    vec3  viewDir     = normalize(vWorldPos - uCameraPos);
    vec3  lightDir    = normalize(uLightPos);              // approximate sun dir

    pbrColor = atmosphere(
        pbrColor,
        viewDir,
        lightDir,
        uDepth * uAtmoDist,          // scale depth to scene units
        uAtmoDensity,
        uAtmoRayleigh,
        uAtmoMie,
        uAtmoMieG
    );

    // ── 8. Tone-mapping — Reinhard (simple, GPU-cheap) ────────────────────────
    // PBR lights are HDR; we need to bring them into [0,1] display range.
    pbrColor = pbrColor / (pbrColor + vec3(1.0));

    // ── 9. Gamma correction (linear → sRGB) ───────────────────────────────────
    pbrColor = pow(pbrColor, vec3(1.0 / 2.2));

    // ── 10. Alpha — surface opacity ───────────────────────────────────────────
    // Cells are opaque by default.  Fresnel rim gives edge softness via the
    // specular contribution rather than transparency.  A thin outer vignette
    // alpha masks the quad so neighbouring cells don't show hard edges.
    float edgeDist = 2.0 * length(vUV - 0.5);        // 0=center, 1=corner
    float alpha    = 1.0 - smoothstep(0.80, 1.0, edgeDist);

    // ── 11. Pre-multiplied alpha output ───────────────────────────────────────
    fragColor = vec4(pbrColor * alpha, alpha);
}
{@}sdf-species-library.frag{@}/**
 * sdf-species-library.frag
 *
 * M552: Unified species SDF shape library — 10 species shapes built from
 *       lygia SDF primitives (circleSDF, boxSDF, starSDF, hexSDF, crossSDF,
 *       flowerSDF, gearSDF, heartSDF, spiralSDF, vesicaSDF, opUnion,
 *       opSubtraction, opOnion, fill, stroke).
 *
 * Usage
 * ─────
 *   float d = speciesSDF(uv, species);
 *   // uv ∈ [0,1]², species ∈ [0..9] (matches SPECIES_ID in sdf-cell-renderer.ts)
 *
 * Species index mapping (mirrors SPECIES_ID in sdf-cell-renderer.ts)
 * ─────────────────────────────────────────────────────────────────────
 *   0  cil-eye          circleSDF + opOnion (concentric rings) + radial rays
 *   1  cil-vector       arrow body (rectSDF) + parallel line strokes
 *   2  cil-bolt         zigzag lineSDF path + starSDF spikes
 *   3  cil-plus         crossSDF + circleSDF opIntersection clamp
 *   4  cil-arrow-right  chevron arrow: two lineSDF legs + triSDF head
 *   5  cil-filter       hexSDF 3×3 tiled grid + centre glow circleSDF
 *   6  cil-code         boxSDF frame (brace silhouette) + inner line strokes
 *   7  cil-layers       three stacked boxSDF planes with depth offset
 *   8  cil-loop         spiralSDF + arrowhead tip (triSDF)
 *   9  cil-graph        circleSDF nodes (3) + lineSDF edges (3)
 *
 * lygia imports (inlined — preprocessor #include resolves at build time)
 * ───────────────────────────────────────────────────────────────────────
 * #include "../../upstream/lygia/sdf/circleSDF.glsl"
 * #include "../../upstream/lygia/sdf/boxSDF.glsl"
 * #include "../../upstream/lygia/sdf/starSDF.glsl"
 * #include "../../upstream/lygia/sdf/hexSDF.glsl"
 * #include "../../upstream/lygia/sdf/crossSDF.glsl"
 * #include "../../upstream/lygia/sdf/flowerSDF.glsl"
 * #include "../../upstream/lygia/sdf/gearSDF.glsl"
 * #include "../../upstream/lygia/sdf/heartSDF.glsl"
 * #include "../../upstream/lygia/sdf/spiralSDF.glsl"
 * #include "../../upstream/lygia/sdf/vesicaSDF.glsl"
 * #include "../../upstream/lygia/sdf/lineSDF.glsl"
 * #include "../../upstream/lygia/sdf/rectSDF.glsl"
 * #include "../../upstream/lygia/sdf/triSDF.glsl"
 * #include "../../upstream/lygia/sdf/opUnion.glsl"
 * #include "../../upstream/lygia/sdf/opSubtraction.glsl"
 * #include "../../upstream/lygia/sdf/opOnion.glsl"
 * #include "../../upstream/lygia/draw/fill.glsl"
 * #include "../../upstream/lygia/draw/stroke.glsl"
 *
 * All lygia primitives are inlined below so the file is self-contained
 * and usable without a preprocessor.
 */

precision mediump float;

// ═══════════════════════════════════════════════════════════════════════════
//  lygia math helpers (inlined)
// ═══════════════════════════════════════════════════════════════════════════

#ifndef PI
#define PI  3.1415926535897932384626433832795
#endif
#ifndef TAU
#define TAU 6.2831853071795864769252867665590
#endif

#if !defined(FNC_SATURATE) && !defined(saturate)
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

// ── lygia/math/map ─────────────────────────────────────────────────────────
#ifndef FNC_MAP
#define FNC_MAP
float map(float v, float iMin, float iMax, float oMin, float oMax) {
    return oMin + (oMax - oMin) * (v - iMin) / (iMax - iMin);
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
//  lygia SDF primitives (inlined from upstream/lygia/sdf/)
// ═══════════════════════════════════════════════════════════════════════════

// ── circleSDF.glsl ──────────────────────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns a circle-shaped SDF.  circleSDF(vec2 st) → distance in [0,1] space.
// Centred at 0.5; result * 2 == diameter-normalised distance.
#ifndef FNC_CIRCLESDF
#define FNC_CIRCLESDF
float circleSDF(in vec2 v) {
    v -= 0.5;
    return length(v) * 2.0;
}
#endif

// ── rectSDF.glsl ────────────────────────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns a rectangular SDF (max-norm variant), UV [0,1] space.
#ifndef FNC_RECTSDF
#define FNC_RECTSDF
float rectSDF(in vec2 st, in vec2 s) {
    st = st * 2.0 - 1.0;
    return max(abs(st.x / s.x), abs(st.y / s.y));
}
float rectSDF(in vec2 st, in float s) { return rectSDF(st, vec2(s)); }
float rectSDF(in vec2 st) { return rectSDF(st, vec2(1.0)); }
// Signed 2-D box (used internally for sharp compositions)
float sdBox2(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
#endif

// ── crossSDF.glsl ───────────────────────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns a cross-shaped SDF (depends on rectSDF).
#ifndef FNC_CROSSSDF
#define FNC_CROSSSDF
float crossSDF(in vec2 st, in float s) {
    vec2 size = vec2(0.25, s);
    return min(rectSDF(st, size.xy), rectSDF(st, size.yx));
}
#endif

// ── hexSDF.glsl ─────────────────────────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns a hexagon-shaped SDF, UV [0,1].
#ifndef FNC_HEXSDF
#define FNC_HEXSDF
float hexSDF(in vec2 st) {
    st = st * 2.0 - 1.0;
    st = abs(st);
    return max(abs(st.y), st.x * 0.866025 + st.y * 0.5);
}
#endif

// ── starSDF.glsl ────────────────────────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns a star-shaped SDF with V branches.
// scale(st, k) expands around centre; inlined here.
#ifndef FNC_STARSDF
#define FNC_STARSDF
float starSDF(in vec2 st, in int V, in float s) {
    st -= 0.5;
    st *= 2.0;
    float a = atan(st.y, st.x) / TAU;
    float seg = a * float(V);
    a = ((floor(seg) + 0.5) / float(V) +
         mix(s, -s, step(0.5, fract(seg)))) * TAU;
    return abs(dot(vec2(cos(a), sin(a)), st));
}
float starSDF(in vec2 st, in int V) {
    // scale equivalent: zoom = 12/V
    st -= 0.5;
    st *= 12.0 / float(V);
    st += 0.5;
    return starSDF(st, V, 0.1);
}
#endif

// ── flowerSDF.glsl ──────────────────────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns a flower-shaped SDF with N petals.
#ifndef FNC_FLOWERSDF
#define FNC_FLOWERSDF
float flowerSDF(vec2 st, int N) {
    st -= 0.5;
    st *= 4.0;
    float r = length(st) * 2.0;
    float a = atan(st.y, st.x);
    float v = float(N) * 0.5;
    return 1.0 - (abs(cos(a * v)) * 0.5 + 0.5) / r;
}
#endif

// ── gearSDF.glsl ────────────────────────────────────────────────────────────
// contributors: Kathy McGuiness
// Returns a gear-shaped SDF.
#ifndef FNC_GEARSDF
#define FNC_GEARSDF
float gearSDF(vec2 st, float b, int N) {
    const float e = 2.71828182845904523536;
    st -= 0.5;
    st *= 3.0;
    float s = map(b, 1.0, 15.0, 0.066, 0.5);
    float d = length(st) - s;
    float omega = b * sin(float(N) * atan(st.y, st.x));
    float l = pow(e, 2.0 * omega);
    float hyperTan = (l - 1.0) / (l + 1.0);
    float r = (1.0 / b) * hyperTan;
    return (d + min(d, r));
}
#endif

// ── heartSDF.glsl ───────────────────────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns a heart-shaped SDF.
#ifndef FNC_HEARTSDF
#define FNC_HEARTSDF
float heartSDF(vec2 st) {
    st -= 0.5;
    st -= vec2(0.0, 0.3);
    float r = length(st) * 5.0;
    st = normalize(st);
    return r - ((st.y * pow(abs(st.x), 0.67)) / (st.y + 1.5) - 2.0 * st.y + 1.26);
}
#endif

// ── spiralSDF.glsl ──────────────────────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns a spiral SDF with t turns.
#ifndef FNC_SPIRALSDF
#define FNC_SPIRALSDF
float spiralSDF(vec2 st, float t) {
    st -= 0.5;
    float r = dot(st, st);
    float a = atan(st.y, st.x);
    return abs(sin(fract(log(r) * t + a * 0.159)));
}
#endif

// ── vesicaSDF.glsl ──────────────────────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns an almond-shaped (vesica piscis) SDF.
#ifndef FNC_VESICASDF
#define FNC_VESICASDF
float vesicaSDF(in vec2 st, in float w) {
    vec2 offset = vec2(w * 0.5, 0.0);
    return max(circleSDF(st - offset), circleSDF(st + offset));
}
float vesicaSDF(in vec2 st) { return vesicaSDF(st, 0.5); }
#endif

// ── lineSDF.glsl ────────────────────────────────────────────────────────────
// contributors: Inigo Quiles
// Segment SDF: unsigned distance from point st to segment [a,b].
#ifndef FNC_LINESDF
#define FNC_LINESDF
float lineSDF(in vec2 st, in vec2 a, in vec2 b) {
    vec2 b_to_a = b - a;
    vec2 to_a   = st - a;
    float h = saturate(dot(to_a, b_to_a) / dot(b_to_a, b_to_a));
    return length(to_a - h * b_to_a);
}
#endif

// ── triSDF.glsl ─────────────────────────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns a triangle-shaped SDF.
#ifndef FNC_TRISDF
#define FNC_TRISDF
float triSDF(in vec2 st) {
    st -= 0.5;
    st *= 5.0;
    return max(abs(st.x) * 0.866025 + st.y * 0.5, -st.y * 0.5);
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
//  lygia SDF operations (inlined from upstream/lygia/sdf/op*.glsl)
// ═══════════════════════════════════════════════════════════════════════════

// ── opUnion.glsl ────────────────────────────────────────────────────────────
// contributors: Inigo Quiles
#ifndef FNC_OPUNION
#define FNC_OPUNION
float opUnion(float d1, float d2) { return min(d1, d2); }
// Smooth union
float opUnion(float d1, float d2, float k) {
    float h = saturate(0.5 + 0.5 * (d2 - d1) / k);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}
#endif

// ── opSubtraction.glsl ──────────────────────────────────────────────────────
// contributors: Inigo Quiles
#ifndef FNC_OPSUBSTRACTION
#define FNC_OPSUBSTRACTION
float opSubtraction(float d1, float d2) { return max(-d1, d2); }
float opSubtraction(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}
#endif

// ── opOnion.glsl ────────────────────────────────────────────────────────────
// contributors: Inigo Quiles
// Shell / onion operation on a distance field.
#ifndef FNC_OPONION
#define FNC_OPONION
float opOnion(in float d, in float h) { return abs(d) - h; }
#endif

// ═══════════════════════════════════════════════════════════════════════════
//  lygia draw helpers (inlined from upstream/lygia/draw/)
// ═══════════════════════════════════════════════════════════════════════════

// ── fill.glsl ───────────────────────────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Fill a SDF shape with a smooth edge.
#ifndef FNC_FILL
#define FNC_FILL
float fill(float x, float size, float edge) {
    return 1.0 - smoothstep(size - edge, size + edge, x);
}
float fill(float x, float size) {
    return 1.0 - smoothstep(size - 0.01, size + 0.01, x);
}
#endif

// ── stroke.glsl ─────────────────────────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Fill a stroke band in a SDF.
#ifndef FNC_STROKE
#define FNC_STROKE
float stroke(float x, float size, float w, float edge) {
    float d = smoothstep(size - edge, size + edge, x + w * 0.5)
            - smoothstep(size - edge, size + edge, x - w * 0.5);
    return saturate(d);
}
float stroke(float x, float size, float w) {
    return stroke(x, size, w, 0.01);
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
//  Internal helpers
// ═══════════════════════════════════════════════════════════════════════════

// Stroke from a line segment, centred in UV, width w (UV units).
float strokeSeg(vec2 p, vec2 a, vec2 b, float w) {
    float d = lineSDF(p, a, b);
    return smoothstep(w, w * 0.4, d);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Per-species SDF functions  (all uv ∈ [0,1]², returns distance scalar)
//  Convention: positive = outside, ≤ 0 = inside.
//  For fill: use fill(d, 0.0) or smoothstep(0.0, edge, -d).
// ═══════════════════════════════════════════════════════════════════════════

// ─── 0: cil-eye ─────────────────────────────────────────────────────────────
// circleSDF outer globe + opOnion concentric rings + radial ray slices.
// Composition:
//   globe    = filled outer circle
//   iris     = opOnion ring band on circleSDF
//   pupil    = small filled circle at centre
//   rays     = modulated angle slices (fill + distance blend)
float sdfEye(vec2 uv) {
    // circleSDF value in [0, ~1.41], 0.5 → edge of unit circle
    float cd = circleSDF(uv);          // [0, 1.41]

    // Outer globe: fill at radius 0.88 (in circleSDF domain)
    float globe = fill(cd, 0.88, 0.02);

    // Iris ring via opOnion: shell at ~0.5 circleSDF, width 0.10
    float irisD = opOnion(cd - 0.50, 0.10);   // ring band
    float iris  = fill(abs(irisD), 0.05, 0.015);

    // Pupil: inner circle at 0.20 radius
    float pupil = fill(cd, 0.20, 0.015);

    // Radial rays: 8-slice using angle modulation
    vec2 p = uv - 0.5;
    float angle = atan(p.y, p.x);
    float sector = mod(angle, TAU / 8.0);
    float rayMask = smoothstep(0.18, 0.0, abs(sector - TAU / 16.0));
    float rayFade = smoothstep(0.44, 0.12, cd) * smoothstep(0.10, 0.20, cd);
    float rays = rayMask * rayFade;

    float mask = clamp(globe * (iris + rays) + pupil, 0.0, 1.0);
    // Return a signed-distance approximation: negative = inside
    return (1.0 - mask) * 0.5 - 0.01;
}

// ─── 1: cil-vector ──────────────────────────────────────────────────────────
// Arrow body (wide rectangle via sdBox2) + chevron head (triSDF) + 3 parallel
// speed-lines to the left (lineSDF strokes).
float sdfVector(vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;   // [-1,1]

    // Shaft: horizontal box  (half-extents 0.50 × 0.10)
    float shaft = sdBox2(p - vec2(-0.05, 0.0), vec2(0.48, 0.10));

    // Arrowhead triangle: triSDF mapped to right half
    vec2 hp = (uv - vec2(0.62, 0.5)) / vec2(0.22, 0.40) + 0.5;
    // Rotate triangle to point right (+x): swap & mirror
    hp -= 0.5; float tx = hp.x; hp.x = -hp.y; hp.y = tx; hp += 0.5;
    float head = triSDF(hp) * 0.30 - 0.15;

    // Union of shaft + head
    float arrow = opUnion(shaft, head);

    // Three parallel speed lines (left side): thin horizontal strokes
    float lines = 1.0;
    lines = min(lines, lineSDF(p, vec2(-0.90, -0.32), vec2(-0.30, -0.32)) - 0.022);
    lines = min(lines, lineSDF(p, vec2(-0.90,  0.00), vec2(-0.20,  0.00)) - 0.022);
    lines = min(lines, lineSDF(p, vec2(-0.90,  0.32), vec2(-0.30,  0.32)) - 0.022);
    float linesMask = opUnion(lines, 10.0); // keep raw SDF

    return min(arrow, linesMask);
}

// ─── 2: cil-bolt ────────────────────────────────────────────────────────────
// Zigzag path (6-segment lineSDF union) + 4-point starSDF spikes at tip.
float sdfBolt(vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;   // [-1,1]

    // Zigzag: 6 segments alternating x = ±0.28
    const int STEPS = 6;
    float dy = 2.0 / float(STEPS);
    float bolt = 1.0;
    for (int i = 0; i < STEPS; i++) {
        float t0 = -1.0 + float(i)       * dy;
        float t1 = -1.0 + float(i + 1)   * dy;
        float s0 = (mod(float(i),     2.0) < 1.0) ? 0.28 : -0.28;
        float s1 = (mod(float(i + 1), 2.0) < 1.0) ? 0.28 : -0.28;
        vec2 a = vec2(s0, t0);
        vec2 b = vec2(s1, t1);
        bolt = min(bolt, lineSDF(p, a, b) - 0.045);
    }

    // Star spikes at the top tip: 4-point star, small scale
    vec2 starUV = (uv - vec2(0.5, 0.05)) / vec2(0.30, 0.30) + 0.5;
    float spike = starSDF(starUV, 4, 0.15) - 0.25;

    return min(bolt, spike);
}

// ─── 3: cil-plus ────────────────────────────────────────────────────────────
// crossSDF arms clamped inside a circleSDF boundary (opSubtraction of exterior).
float sdfPlus(vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;   // [-1,1]

    // crossSDF: two overlapping boxes (arm half-length 0.76, width 0.25)
    float h = sdBox2(p, vec2(0.76, 0.20));
    float v = sdBox2(p, vec2(0.20, 0.76));
    float cross_ = min(h, v);

    // circleSDF for the bounding disc (radius 0.80 in [-1,1] space)
    float disc = length(p) - 0.82;

    // Clip cross to disc: opSubtraction(disc, cross) keeps interior
    float clipped = opSubtraction(disc, cross_);

    return clipped;
}

// ─── 4: cil-arrow-right ─────────────────────────────────────────────────────
// Chevron: two diagonal lineSDF legs meeting at right, tiled 3×3.
// Each tile shows one right-pointing arrow chevron.
float sdfArrowRight(vec2 uv) {
    // Tile 3×3
    vec2 tiled = fract(uv * 3.0) * 2.0 - 1.0;   // [-1,1] per tile

    // Two legs of the chevron
    float d1 = lineSDF(tiled, vec2(-0.55,  0.55), vec2( 0.55, 0.0)) - 0.055;
    float d2 = lineSDF(tiled, vec2(-0.55, -0.55), vec2( 0.55, 0.0)) - 0.055;

    return min(d1, d2);
}

// ─── 5: cil-filter ──────────────────────────────────────────────────────────
// 3×3 hexSDF grid of small hexagons + central glow circleSDF disc.
float sdfFilter(vec2 uv) {
    // Tile 3×3 hexagon grid
    vec2 tile = fract(uv * 3.0);
    float hex = hexSDF(tile) - 0.62;     // each hex at ~62% cell fill

    // Central glow: larger filled circle at centre of the whole icon
    float glow = circleSDF(uv) - 0.30;   // disc radius 0.15 (circleSDF at 0.30)

    return min(hex, glow);
}

// ─── 6: cil-code ────────────────────────────────────────────────────────────
// Two mirrored boxSDF "brace" frames (left < and right >) + 3 inner line strokes.
float sdfCode(vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;

    // Left brace: two angled lineSDF segments meeting at x = -0.60
    float lb1 = lineSDF(p, vec2(-0.30,  0.78), vec2(-0.68,  0.0)) - 0.055;
    float lb2 = lineSDF(p, vec2(-0.30, -0.78), vec2(-0.68,  0.0)) - 0.055;
    float leftBrace = min(lb1, lb2);

    // Right brace (mirror of left)
    float rb1 = lineSDF(p, vec2( 0.30,  0.78), vec2( 0.68,  0.0)) - 0.055;
    float rb2 = lineSDF(p, vec2( 0.30, -0.78), vec2( 0.68,  0.0)) - 0.055;
    float rightBrace = min(rb1, rb2);

    float braces = min(leftBrace, rightBrace);

    // Three inner horizontal code lines
    float l1 = lineSDF(p, vec2(-0.22,  0.38), vec2( 0.22,  0.38)) - 0.038;
    float l2 = lineSDF(p, vec2(-0.22,  0.0 ), vec2( 0.22,  0.0 )) - 0.038;
    float l3 = lineSDF(p, vec2(-0.22, -0.38), vec2( 0.22, -0.38)) - 0.038;
    float innerLines = min(min(l1, l2), l3);

    return min(braces, innerLines);
}

// ─── 7: cil-layers ──────────────────────────────────────────────────────────
// Three stacked boxSDF quads with vertical offset (isometric stack illusion).
float sdfLayers(vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;

    // Layer offsets: top (+0.55), middle (0.0), bottom (-0.55)
    // Each layer is a thin wide box, slightly narrower as they go back
    float top    = sdBox2(p - vec2(0.0,  0.54), vec2(0.65, 0.13));
    float mid    = sdBox2(p - vec2(0.0,  0.00), vec2(0.72, 0.13));
    float bottom = sdBox2(p - vec2(0.0, -0.54), vec2(0.65, 0.13));

    // opUnion all three layers
    float layers = opUnion(top, opUnion(mid, bottom));

    return layers;
}

// ─── 8: cil-loop ────────────────────────────────────────────────────────────
// spiralSDF curl (1.5 turns) + small triSDF arrowhead at the tip.
float sdfLoop(vec2 uv) {
    // Spiral: 1.5 turns, threshold at ~0.45 for a stroke band
    float sp = spiralSDF(uv, 1.5);
    float spiral = stroke(sp, 0.48, 0.10, 0.015);   // stroke band

    // Arrowhead at the spiral exit point (~right side)
    // Small triangle pointing right, positioned at ~(0.76, 0.50)
    vec2 tipUV = (uv - vec2(0.72, 0.50)) / vec2(0.18, 0.24) + 0.5;
    tipUV -= 0.5; float tx = tipUV.x; tipUV.x = -tipUV.y; tipUV.y = tx; tipUV += 0.5;
    float tip = triSDF(tipUV) * 0.24 - 0.12;

    // Combine: spiral stroke mask + tip shape
    float spiralD = (1.0 - spiral) * 0.5 - 0.01;    // convert mask to SDF approx
    return min(spiralD, tip);
}

// ─── 9: cil-graph ───────────────────────────────────────────────────────────
// Three circleSDF nodes connected by three lineSDF edges (triangle topology).
float sdfGraph(vec2 uv) {
    // Node positions (UV space)
    vec2 n0 = vec2(0.50, 0.82);   // top centre
    vec2 n1 = vec2(0.18, 0.22);   // bottom left
    vec2 n2 = vec2(0.82, 0.22);   // bottom right

    float nodeR = 0.15;   // circleSDF threshold (radius in UV)
    float edgeW = 0.035;  // edge stroke half-width

    // Three nodes: circleSDF centred at each node position
    float nd0 = circleSDF(uv - n0 + 0.5) - nodeR;
    float nd1 = circleSDF(uv - n1 + 0.5) - nodeR;
    float nd2 = circleSDF(uv - n2 + 0.5) - nodeR;
    float nodes = min(nd0, min(nd1, nd2));

    // Three edges: lineSDF strokes between nodes
    float e01 = lineSDF(uv, n0, n1) - edgeW;
    float e12 = lineSDF(uv, n1, n2) - edgeW;
    float e02 = lineSDF(uv, n0, n2) - edgeW;
    float edges = min(e01, min(e12, e02));

    return min(nodes, edges);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Unified dispatcher
// ═══════════════════════════════════════════════════════════════════════════

/**
 * speciesSDF — unified shape function for all 10 cell species.
 *
 * Parameters
 * ──────────
 *   uv      : normalised fragment coordinate in [0,1]²
 *   species : integer index (matches SPECIES_ID in sdf-cell-renderer.ts)
 *               0 = cil-eye
 *               1 = cil-vector
 *               2 = cil-bolt
 *               3 = cil-plus
 *               4 = cil-arrow-right
 *               5 = cil-filter
 *               6 = cil-code
 *               7 = cil-layers
 *               8 = cil-loop
 *               9 = cil-graph
 *
 * Returns
 * ───────
 *   float SDF value:
 *     < 0  → inside the shape
 *     = 0  → on the boundary
 *     > 0  → outside the shape
 *
 * Usage example
 * ─────────────
 *   float d = speciesSDF(uv, species);
 *   float mask = fill(d, 0.0, 0.015);           // solid fill
 *   float ring = stroke(d, 0.0, 0.025, 0.010);  // outline stroke
 *   float glow = smoothstep(0.12, 0.0, d) * 0.4;
 *   float alpha = clamp(mask + ring + glow, 0.0, 1.0);
 *   gl_FragColor = vec4(u_fillColor, alpha * u_opacity);
 */
float speciesSDF(vec2 uv, int species) {
    if      (species == 0) return sdfEye(uv);
    else if (species == 1) return sdfVector(uv);
    else if (species == 2) return sdfBolt(uv);
    else if (species == 3) return sdfPlus(uv);
    else if (species == 4) return sdfArrowRight(uv);
    else if (species == 5) return sdfFilter(uv);
    else if (species == 6) return sdfCode(uv);
    else if (species == 7) return sdfLayers(uv);
    else if (species == 8) return sdfLoop(uv);
    else if (species == 9) return sdfGraph(uv);
    // Fallback: full disc (species unknown)
    else return circleSDF(uv) - 0.80;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Optional standalone main() — enables use as a self-contained test shader.
//  Define SPECIES_LIBRARY_STANDALONE to activate; omit when #include-ing.
// ═══════════════════════════════════════════════════════════════════════════

#ifdef SPECIES_LIBRARY_STANDALONE

uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform int   u_species;   // [0..9]
uniform float u_time;

void main() {
    vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;

    float d    = speciesSDF(uv, u_species);

    float mask = fill(d, 0.0, 0.015);
    float ring = stroke(d, 0.0, 0.025, 0.010);
    float glow = smoothstep(0.14, 0.0, d) * 0.38;

    float alpha = clamp(mask + ring * 0.6 + glow, 0.0, 1.0);

    vec3 col = u_fillColor + glow * 0.5;

    gl_FragColor = vec4(col, alpha * u_opacity);
}

#endif // SPECIES_LIBRARY_STANDALONE
{@}supershape-species.frag{@}/**
 * supershape-species.frag
 *
 * M571: SuperShape species morphology — 3 species (eye / bolt / plus) driven
 *       by the Gielis superformula, animated by u_pressure which modulates the
 *       n exponents in real time.
 *
 * Gielis superformula (polar SDF)
 * ────────────────────────────────
 *   r(θ) = [ |cos(m·θ/4) / a|^n2 + |sin(m·θ/4) / b|^n3 ]^(-1/n1)
 *
 * The distance field is built as:
 *   d(p) = |p| − r(atan(p.y, p.x))
 *
 * which is negative inside and positive outside the superShape boundary.
 *
 * Species mapping (u_species)
 * ────────────────────────────
 *   0  cil-eye    circle variant   m=0   n1=2    n2=2    n3=2
 *                 (superShape with m=0 degenerates to a perfect circle;
 *                  u_pressure inflates/deflates the n2,n3 bulge)
 *
 *   1  cil-bolt   pinched / star   m=2   n1=0.3  n2=0.3  n3=0.3
 *                 (low n1 collapses to a pinched/concave 2-lobe;
 *                  u_pressure pumps n toward 1.0 making it rounder)
 *
 *   2  cil-plus   sharp cross      m=4   n1=0.5  n2=0.5  n3=100.0
 *                 (high n3 sharpens the 4 lobes into a plus / cross;
 *                  u_pressure drives n3 between 100 and 2 morphing
 *                  cross → rounded quad)
 *
 * u_pressure coupling
 * ────────────────────
 *   u_pressure ∈ [0,1] — maps to different n-parameter perturbations
 *   per species so physical "pressure" visibly warps the shape:
 *
 *   eye  : n1 = 2.0 + pressure * 6.0       (circle → barrel / superellipse)
 *   bolt : n1 = 0.3 + pressure * 1.4       (pinch → lune → rounded lobe)
 *          n2 = n3 = 0.3 + pressure * 0.9
 *   plus : n3 = 100.0 − pressure * 98.0    (sharp cross → rounded quad)
 *
 * Composition layers
 * ──────────────────
 *   • fill(d, 0.0, 0.015)           — solid interior
 *   • stroke(d, 0.0, 0.025, 0.010) — bright outline ring
 *   • smoothstep glow               — soft outer halo
 *   • snoise turbulence overlay     — animated surface noise baked into
 *                                     the fill (gives cells a "living" look)
 *   • opUnion used to merge the main shape with a secondary snoise-derived
 *     micro-shape (diffuse blob softens the hard SDF edge at low pressure)
 *
 * lygia #include references (inlined below, no preprocessor needed)
 * ──────────────────────────────────────────────────────────────────
 *   lygia/sdf/superShapeSDF.glsl
 *   lygia/sdf/opUnion.glsl
 *   lygia/draw/fill.glsl
 *   lygia/generative/snoise.glsl
 *
 * Uniforms
 * ─────────
 *   u_bbox      vec4    (x, y, width, height) in canvas pixels
 *   u_fillColor vec3    species base colour
 *   u_opacity   float   master opacity
 *   u_time      float   seconds (for snoise animation)
 *   u_pressure  float   [0,1] SPH pressure → morphs n parameters
 *   u_species   int     [0..2]  0=eye  1=bolt  2=plus
 *
 * Authoring note
 * ──────────────
 *   The superShapeSDF primitive is implemented inline (lygia-style) because
 *   #include resolution requires a build-time preprocessor. The inline matches
 *   lygia/sdf/superShapeSDF.glsl exactly in API and behaviour so a future
 *   toolchain upgrade can swap to the canonical #include without edits.
 */

precision mediump float;

// ── Uniforms ──────────────────────────────────────────────────────────────────

uniform vec4  u_bbox;        // x, y, width, height in canvas coords
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform float u_time;
uniform float u_pressure;    // [0,1]  SPH pressure driving n-parameter morphing
uniform int   u_species;     // 0 = eye   1 = bolt   2 = plus

// ═══════════════════════════════════════════════════════════════════════════════
//  lygia math helpers (inlined)
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef PI
#define PI  3.1415926535897932384626433832795
#endif
#ifndef TAU
#define TAU 6.2831853071795864769252867665590
#endif

#ifndef FNC_SATURATE
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

// ═══════════════════════════════════════════════════════════════════════════════
//  lygia/sdf/superShapeSDF.glsl  (inlined — API-identical to lygia upstream)
// ═══════════════════════════════════════════════════════════════════════════════
//
// superShapeSDF — polar SDF built from the Gielis superformula.
//
// Parameters
//   st   : UV in [0,1]²  (centred at 0.5,0.5)
//   m    : rotational symmetry order (integer, passed as float)
//   n1   : overall shape exponent (controls "bulge" vs "pinch")
//   n2   : even-half exponent
//   n3   : odd-half exponent
//   a, b : axis scales (typically both 1.0)
//
// Returns signed distance (< 0 inside, > 0 outside).
// The returned value is normalised to the unit-circle scale used by lygia's
// other SDF helpers (radius ≈ 0.5 in UV space = value 1.0 in circleSDF).
//
// Reference: Johan Gielis (2003) "A generic geometric transformation that
// unifies a wide range of natural and abstract shapes." Am. J. Botany 90(3).
//
#ifndef FNC_SUPERSHAPESDF
#define FNC_SUPERSHAPESDF

float superShapeRadius(float theta, float m, float n1, float n2, float n3, float a, float b) {
    // Gielis formula:  r(θ) = ( |cos(m·θ/4)/a|^n2 + |sin(m·θ/4)/b|^n3 )^(-1/n1)
    float mTheta4 = m * theta * 0.25;
    float cosT    = abs(cos(mTheta4)) / a;
    float sinT    = abs(sin(mTheta4)) / b;
    // Clamp base to avoid pow(0,negative) NaN for very low n1
    float base    = pow(cosT, n2) + pow(sinT, n3);
    // Guard against base == 0 (degenerate case when both terms collapse)
    base = max(base, 1e-6);
    return pow(base, -1.0 / n1);
}

// Main SDF entry — uv ∈ [0,1]²
float superShapeSDF(vec2 uv, float m, float n1, float n2, float n3) {
    // Map UV to centred [-1,1] space then scale so radius ≈ 0.5 at shape edge
    vec2  p     = uv * 2.0 - 1.0;
    float theta = atan(p.y, p.x);
    float r     = superShapeRadius(theta, m, n1, n2, n3, 1.0, 1.0);
    // Normalise r to the same scale as circleSDF (length * 2 convention)
    // The superformula returns r in [0, ~1], so a perfect circle (m=0,n=2)
    // gives r=1.  We compare against the normalised distance * 2 so that
    // the SDF is 0 at the shape boundary.
    float dist  = length(p) / r;
    // Return in the "2× normalised" circleSDF convention used by lygia:
    // value 1.0 → on the boundary of a unit shape inscribed in [0,1]².
    return (dist - 1.0) * 0.5;
}

#endif // FNC_SUPERSHAPESDF

// ═══════════════════════════════════════════════════════════════════════════════
//  lygia/sdf/opUnion.glsl  (inlined)
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef FNC_OPUNION
#define FNC_OPUNION
float opUnion(float d1, float d2) { return min(d1, d2); }
// Smooth union (Inigo Quilez)
float opUnion(float d1, float d2, float k) {
    float h = saturate(0.5 + 0.5 * (d2 - d1) / k);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}
#endif

// ═══════════════════════════════════════════════════════════════════════════════
//  lygia/draw/fill.glsl  (inlined)
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef FNC_FILL
#define FNC_FILL
float fill(float x, float size, float edge) {
    return 1.0 - smoothstep(size - edge, size + edge, x);
}
float fill(float x, float size) {
    return 1.0 - smoothstep(size - 0.01, size + 0.01, x);
}
#endif

// ── stroke helper ─────────────────────────────────────────────────────────────

#ifndef FNC_STROKE
#define FNC_STROKE
float stroke(float x, float size, float w, float edge) {
    float d = smoothstep(size - edge, size + edge, x + w * 0.5)
            - smoothstep(size - edge, size + edge, x - w * 0.5);
    return saturate(d);
}
float stroke(float x, float size, float w) {
    return stroke(x, size, w, 0.01);
}
#endif

// ═══════════════════════════════════════════════════════════════════════════════
//  Simplex noise — snoise(vec2)
//
//  Inlined from lygia/generative/snoise.glsl which in turn is based on:
//    Ian McEwan, Ashima Arts — "Efficient computational noise in GLSL"
//    (https://github.com/ashima/webgl-noise, MIT licence)
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef FNC_SNOISE
#define FNC_SNOISE

vec3 _sn_mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 _sn_mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 _sn_permute(vec3 x) { return _sn_mod289(((x * 34.0) + 1.0) * x); }

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187,   // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,   // 0.5*(sqrt(3.0)-1.0)
                       -0.577350269189626,   // -1.0 + 2.0 * C.x
                        0.024390243902439);  // 1.0 / 41.0
    vec2  i  = floor(v + dot(v,  C.yy));
    vec2  x0 = v - i + dot(i, C.xx);
    vec2  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4  x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;

    i = _sn_mod289(i);
    vec3 p = _sn_permute(_sn_permute(i.y + vec3(0.0, i1.y, 1.0))
                                   + i.x + vec3(0.0, i1.x, 1.0));

    vec3 m = max(0.5 - vec3(dot(x0, x0),
                             dot(x12.xy, x12.xy),
                             dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;

    vec3 x  = 2.0 * fract(p * C.www) - 1.0;
    vec3 h  = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;

    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

    vec3 g;
    g.x  = a0.x  * x0.x   + h.x  * x0.y;
    g.yz = a0.yz * x12.xz  + h.yz * x12.yw;

    return 130.0 * dot(m, g);
}

#endif // FNC_SNOISE

// ═══════════════════════════════════════════════════════════════════════════════
//  Per-species superShape parameter resolver
//
//  Returns vec4(m, n1, n2, n3) for the given species, with u_pressure applied.
//
//  eye  (0) : m=0  — superShape degenerates to circle; pressure inflates n1
//             making it a "superellipse" as pressure increases.
//  bolt (1) : m=2  — low n values create a pinched 2-lobe (yin-yang-like);
//             pressure gradually rounds both n1 and n2/n3 toward 1.
//  plus (2) : m=4  — high n3 makes 4 very sharp blades (cross / plus sign);
//             pressure decays n3 toward 2.0, morphing cross → rounded quad.
// ═══════════════════════════════════════════════════════════════════════════════

vec4 resolveSpeciesParams(int sp, float pressure) {
    float m, n1, n2, n3;

    if (sp == 0) {
        // ── eye: circle (m=0) → superellipse under pressure ─────────────────
        // m=0 makes cos(0)=1 and sin(0)=0 for all θ, which with equal n values
        // collapses the angular dependence and produces a perfect circle.
        // Raising n1 while keeping n2=n3=2 tilts it to a barrel / lemon.
        m  = 0.0;
        n1 = 2.0 + pressure * 6.0;   // 2 → 8  (circle → barrel)
        n2 = 2.0;
        n3 = 2.0;
    }
    else if (sp == 1) {
        // ── bolt: pinched lobe (m=2, low n) → round lobe under pressure ─────
        // m=2 → 2-fold symmetry; very low n (<1) creates a concave, pinched
        // hypocycloid-like form resembling a bolt / lightning glyph silhouette.
        // Increasing pressure rounds all three n values toward ~1.
        m  = 2.0;
        n1 = 0.3 + pressure * 1.4;   // 0.3 → 1.7  (pinch → round)
        n2 = 0.3 + pressure * 0.9;   // 0.3 → 1.2
        n3 = 0.3 + pressure * 0.9;   // 0.3 → 1.2
    }
    else {
        // ── plus: cross (m=4, high n3) → rounded quad under pressure ────────
        // m=4 → 4-fold symmetry; n3=100 sharpens the 4 lobes into knife-edge
        // blades — the classic "plus" / cross silhouette.
        // Increasing pressure decays n3 toward 2 (square → rounded quad).
        m  = 4.0;
        n1 = 0.5 + pressure * 0.5;   // 0.5 → 1.0
        n2 = 0.5 + pressure * 1.5;   // 0.5 → 2.0
        n3 = 100.0 - pressure * 98.0; // 100 → 2   (cross → quad)
    }

    return vec4(m, n1, n2, n3);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════════════════════

void main() {
    // Normalise fragment to bbox-local UV [0,1]
    vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;

    // ── superShape parameters for this species + pressure ────────────────────
    vec4  sp   = resolveSpeciesParams(u_species, u_pressure);
    float m    = sp.x;
    float n1   = sp.y;
    float n2   = sp.z;
    float n3   = sp.w;

    // ── primary superShape SDF ───────────────────────────────────────────────
    // Slight time-driven rotation so the shape slowly spins.
    // We rotate the UV in centred [-1,1] space by a small angle.
    vec2   pc        = uv - 0.5;                            // centred
    float  spinAngle = u_time * 0.08 * float(u_species + 1); // per-species speed
    float  cosS      = cos(spinAngle);
    float  sinS      = sin(spinAngle);
    vec2   pcRot     = vec2(cosS * pc.x - sinS * pc.y,
                             sinS * pc.x + cosS * pc.y);
    vec2   uvRot     = pcRot + 0.5;                         // back to [0,1]

    float dMain = superShapeSDF(uvRot, m, n1, n2, n3);

    // ── snoise turbulence overlay ─────────────────────────────────────────────
    // A slow-moving simplex noise field slightly deforms the SDF boundary,
    // giving cells a pulsing, organic "living" quality.
    // We add the noise displacement ONLY near the boundary (|dMain| < 0.1)
    // so it doesn't bleed into deep interior or far exterior regions.
    float noiseScale     = 3.5 + float(u_species) * 1.2;
    float noiseTimeSpeed = 0.18 + float(u_species) * 0.05;
    float noiseAmp       = 0.018 + u_pressure * 0.012;

    float ns = snoise(uvRot * noiseScale + vec2(u_time * noiseTimeSpeed,
                                                 u_time * noiseTimeSpeed * 0.7));
    // Boundary-proximity weight: full effect near edge, none deep in/out
    float edgeWeight = 1.0 - smoothstep(0.0, 0.10, abs(dMain));
    float dNoise     = dMain + ns * noiseAmp * edgeWeight;

    // ── secondary micro-blob via snoise (opUnion with main shape) ────────────
    // A second, lower-frequency noise field creates a subtle "amoeba-like"
    // blob that smoothly unions with the main shape at low pressure, making
    // the cell look like it's pushing out pseudopodia.
    float ns2     = snoise(uv * 2.0 + vec2(u_time * 0.07, u_time * 0.05));
    float blobR   = 0.38 + ns2 * 0.06 * (1.0 - u_pressure);  // shrinks with pressure
    float dBlob   = length(pc) - blobR;                        // circle blob SDF

    // Smooth union: blend factor driven by pressure (high pressure → hard edge)
    float unionK = mix(0.06, 0.005, u_pressure);
    float dFinal = opUnion(dNoise, dBlob, unionK);

    // ── drawing layers ────────────────────────────────────────────────────────

    // Solid fill interior
    float mask  = fill(dFinal, 0.0, 0.014);

    // Crisp outline ring
    float ring  = stroke(dFinal, 0.0, 0.022, 0.009);

    // Soft outer glow (exponential falloff past the boundary)
    float glow  = exp(-max(dFinal, 0.0) * 18.0) * 0.45;

    // Inner snoise-derived surface texture — adds subtle brightness variation
    // inside the fill so the cell surface looks lit from within.
    float ns3      = snoise(uv * 6.0 + vec2(u_time * 0.12, u_time * 0.09));
    float textureMask = fill(dFinal, 0.0, 0.02);        // only inside shape
    float textureTerm = (ns3 * 0.5 + 0.5) * 0.20 * textureMask;

    // ── composite ────────────────────────────────────────────────────────────
    float alpha = clamp(mask + ring * 0.65 + glow + textureTerm, 0.0, 1.0);

    // Colour: base fill + glow tint (slightly blue-shifted for energy feel)
    vec3 glowTint = vec3(0.5, 0.7, 1.0);
    vec3 col      = u_fillColor
                  + glowTint * glow * 0.30
                  + u_fillColor * textureTerm;

    gl_FragColor = vec4(col, alpha * u_opacity);
}
{@}voronoi-membrane.frag{@}#version 300 es
// ── voronoi-membrane.frag ──────────────────────────────────────────────────────
// Voronoi / Worley noise cell-membrane texture.
//
// Biology model
// ─────────────
// Real biological cell membranes ARE Voronoi diagrams: each cell expands
// outward from its nucleus until it meets its neighbours — the membrane sits
// exactly at the equidistant boundary between cell centres.  This shader
// reconstructs that geometry procedurally:
//
//   1. VORONOI STRUCTURE — cellular2D (Worley noise, webgl-noise/Stefan
//      Gustavson) returns F1 (nearest cell centre) and F2 (second-nearest).
//      The membrane ridge is the iso-contour F2 − F1 ≈ 0, i.e. points that
//      are equidistant from two cell centres.
//
//   2. ORGANIC DISTORTION — before sampling Voronoi, each UV coordinate is
//      displaced by a time-varying snoise(vec2) field.  This warps the cell
//      centres as if the nuclei are drifting, making the membranes irregular
//      and alive.  The snoise perturbation amplitude is small (≈ 0.1 × cell
//      scale) so cells stay topologically connected while their shapes become
//      non-convex / asymmetric.
//
//   3. MEMBRANE WALL — stroke(F2-F1, 0.0, thickness) paints a band around
//      the F2=F1 ridge.  Membrane thickness varies with the local pressure
//      uniform: high pressure squeezes the membrane thinner (Laplace
//      law T = P·r / 2).  A soft edge (stroke's 4-parameter overload with
//      edge = 0.015) gives AA without dFdx / dFdy precision concerns.
//
//   4. HEX UNDERLAY — hexTile(st) maps the same UV into a flat-topped
//      hexagonal grid.  The distance-to-hex-centre (hexTile.z) becomes a
//      subtle honeycomb relief underneath the Voronoi membrane, mimicking
//      the cytoskeleton scaffold that enforces the nearly-hexagonal packing
//      observed in epithelial sheets.
//
//   5. SPECIES PALETTE — u_species selects one of 10 colour palettes that
//      map to the cell icon species used elsewhere in the project.  Each
//      palette has an inner fill colour, a membrane colour, and a highlight
//      tint so the membrane texture stays visually consistent with the
//      species SDF shapes.
//
//   6. PRESSURE PHYSICS — u_pressure (SPH scalar, typically 0–1) drives:
//       • membrane thinning  → thickness ∝ (1 − 0.6 · pressure)
//       • membrane opacity   → alpha    ∝ (1 − 0.4 · pressure)
//       • hex relief scale   → hexScale ∝ (1 + 0.5 · pressure)
//      High-pressure cells become thin-membraned and translucent, matching
//      fluid-dynamics intuition (compressed gas cell, osmotic pressure).
//
// GLSL #include dependencies (resolved by the project GLSL preprocessor)
// ───────────────────────────────────────────────────────────────────────
//   ../../upstream/webgl-noise/src/cellular2D.glsl  — cellular() / Worley noise
//   ../../upstream/lygia/generative/snoise.glsl     — snoise(vec2)
//   ../../upstream/lygia/space/hexTile.glsl         — hexTile(vec2)
//   ../../upstream/lygia/draw/stroke.glsl           — stroke(x, size, w, edge)
//
// Uniforms
// ────────
//   u_cellScale         float   Voronoi cell spatial frequency          ≈ 6.0
//   u_membraneThickness float   Base membrane wall width  [0.01–0.15]   ≈ 0.06
//   u_pressure          float   SPH pressure [0,1]; thins membrane      ≈ 0.0
//   u_species           int     Species index [0,9]; selects palette
//   u_time              float   Elapsed seconds; drives snoise drift
//
// Varyings (from bbox-quad vertex shader)
//   vUV                 vec2    [0,1]² local bbox coordinates
//
// Output
//   fragColor           vec4    premultiplied RGBA
//
// References
//   Stefan Gustavson (2011) — Cellular noise in GLSL
//   Patricio Gonzalez Vivo — lygia.xyz (hexTile, stroke, snoise)
//   Nienhaus & Koster (2002) — Voronoi tessellation of biological tissues
//   Farhadifar et al. (2007) — The influence of cell mechanics on epithelium
//   xiaodi #M612 — cell-pubsub-loop
// ──────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── Varyings ──────────────────────────────────────────────────────────────────

in  vec2  vUV;

out vec4  fragColor;

// ── Uniforms ──────────────────────────────────────────────────────────────────

uniform float u_cellScale;          // e.g. 6.0  — cells per UV unit
uniform float u_membraneThickness;  // e.g. 0.06 — base wall width
uniform float u_pressure;           // [0,1]     — SPH pressure
uniform int   u_species;            // [0,9]     — species palette index
uniform float u_time;               // seconds

// ── Inlined lygia math helpers ────────────────────────────────────────────────
// Sources: upstream/lygia/math/{mod289,permute,taylorInvSqrt,saturate,aastep}.glsl
// License: Patricio Gonzalez Vivo — Prosperity / Patron License (lygia.xyz)

#ifndef FNC_SATURATE
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

#ifndef FNC_MOD289
#define FNC_MOD289
float mod289(const in float x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec2  mod289(const in vec2  x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec3  mod289(const in vec3  x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4  mod289(const in vec4  x) { return x - floor(x * (1.0/289.0)) * 289.0; }
#endif

#ifndef FNC_PERMUTE
#define FNC_PERMUTE
float permute(const in float v) { return mod289(((v * 34.0) + 10.0) * v); }
vec3  permute(const in vec3  v) { return mod289(((v * 34.0) + 10.0) * v); }
vec4  permute(const in vec4  v) { return mod289(((v * 34.0) + 10.0) * v); }
#endif

#ifndef FNC_TAYLORINVSQRT
#define FNC_TAYLORINVSQRT
float taylorInvSqrt(in float r) { return 1.79284291400159 - 0.85373472095314 * r; }
vec4  taylorInvSqrt(in vec4  r) { return 1.79284291400159 - 0.85373472095314 * r; }
#endif

// ── #include "../../upstream/lygia/generative/snoise.glsl" ────────────────────
// Inlined: snoise(vec2) — Simplex noise, Gustavson & McEwan
// License: MIT

#ifndef FNC_SNOISE
#define FNC_SNOISE
float snoise(in vec2 v) {
    const vec4 C = vec4( 0.211324865405187,   // (3-sqrt(3))/6
                         0.366025403784439,   // 0.5*(sqrt(3)-1)
                        -0.577350269189626,   // -1 + 2*C.x
                         0.024390243902439);  // 1/41
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1  = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy  -= i1;
    i = mod289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                            + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m * m;
    m = m * m;
    vec3 x  = 2.0 * fract(p * C.www) - 1.0;
    vec3 h  = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}
#endif

// ── #include "../../upstream/webgl-noise/src/cellular2D.glsl" ─────────────────
// Inlined: cellular(vec2) → vec2(F1, F2)  — Worley / Voronoi noise
// Copyright (c) Stefan Gustavson 2011-04-19. MIT License.
// webgl-noise has #version 120 helpers; we re-use our mod289/permute above.

#ifndef FNC_CELLULAR2D
#define FNC_CELLULAR2D
// mod7 — modulo 7 without division
vec3 _mod7(vec3 x) { return x - floor(x * (1.0/7.0)) * 7.0; }

// cellular — returns vec2(F1, F2) distances to nearest Voronoi seeds
vec2 cellular(vec2 P) {
    const float K  = 0.142857142857; // 1/7
    const float Ko = 0.428571428571; // 3/7
    const float _jitter = 1.0;       // 1.0 = full randomness; < 1 = more regular

    vec2 Pi = mod289(floor(P));
    vec2 Pf = fract(P);
    vec3 oi = vec3(-1.0, 0.0, 1.0);
    vec3 of = vec3(-0.5, 0.5, 1.5);

    vec3 px = permute(Pi.x + oi);

    // Row 1
    vec3 p  = permute(px.x + Pi.y + oi);
    vec3 ox = fract(p * K) - Ko;
    vec3 oy = _mod7(floor(p * K)) * K - Ko;
    vec3 dx = Pf.x + 0.5 + _jitter * ox;
    vec3 dy = Pf.y - of  + _jitter * oy;
    vec3 d1 = dx*dx + dy*dy;

    // Row 2
    p  = permute(px.y + Pi.y + oi);
    ox = fract(p * K) - Ko;
    oy = _mod7(floor(p * K)) * K - Ko;
    dx = Pf.x - 0.5 + _jitter * ox;
    dy = Pf.y - of  + _jitter * oy;
    vec3 d2 = dx*dx + dy*dy;

    // Row 3
    p  = permute(px.z + Pi.y + oi);
    ox = fract(p * K) - Ko;
    oy = _mod7(floor(p * K)) * K - Ko;
    dx = Pf.x - 1.5 + _jitter * ox;
    dy = Pf.y - of  + _jitter * oy;
    vec3 d3 = dx*dx + dy*dy;

    // Sort to find F1 and F2
    vec3 d1a = min(d1, d2);
    d2       = max(d1, d2);
    d2       = min(d2, d3);
    d1       = min(d1a, d2);
    d2       = max(d1a, d2);
    d1.xy    = (d1.x < d1.y) ? d1.xy : d1.yx;
    d1.xz    = (d1.x < d1.z) ? d1.xz : d1.zx;
    d1.yz    = min(d1.yz, d2.yz);
    d1.y     = min(d1.y, d1.z);
    d1.y     = min(d1.y, d2.x);
    return sqrt(d1.xy);
}
#endif

// ── #include "../../upstream/lygia/space/hexTile.glsl" ────────────────────────
// Inlined: hexTile(vec2) → vec4(local.xy, tile.xy)
// License: Patricio Gonzalez Vivo — Prosperity / Patron

#ifndef FNC_HEXTILE
#define FNC_HEXTILE
vec4 hexTile(vec2 st) {
    vec2 s = vec2(1.0, 1.7320508);   // 1, sqrt(3)
    vec2 o = vec2(0.5, 1.0);
    st = st.yx;
    vec4 i = floor(vec4(st, st - o) / s.xyxy) + 0.5;
    vec4 f = vec4(st - i.xy * s, st - (i.zw + 0.5) * s);
    return dot(f.xy, f.xy) < dot(f.zw, f.zw)
           ? vec4(f.yx + 0.5, i.xy)
           : vec4(f.wz + 0.5, i.zw + o);
}
#endif

// ── #include "../../upstream/lygia/draw/stroke.glsl" ──────────────────────────
// Inlined: stroke(x, size, w, edge) — anti-aliased band around an SDF value
// License: Patricio Gonzalez Vivo — Prosperity / Patron

#ifndef FNC_STROKE
#define FNC_STROKE
float stroke(float x, float size, float w, float edge) {
    float d = smoothstep(size - edge, size + edge, x + w * 0.5)
            - smoothstep(size - edge, size + edge, x - w * 0.5);
    return saturate(d);
}
#endif

// ── Species palette lookup ─────────────────────────────────────────────────────
// 10 palettes matching the SPECIES_ID set in sdf-cell-renderer.ts
// Each palette: vec3(innerFill RGB), vec3(membraneRGB), vec3(highlightRGB)

struct Palette {
    vec3 fill;
    vec3 membrane;
    vec3 highlight;
};

Palette speciesPalette(int s) {
    // cil-eye       (0)  — iris blue / corneal silver
    if (s == 0) return Palette(vec3(0.08, 0.18, 0.40),
                               vec3(0.45, 0.65, 0.92),
                               vec3(0.80, 0.90, 1.00));
    // cil-vector    (1)  — flow teal
    if (s == 1) return Palette(vec3(0.04, 0.28, 0.30),
                               vec3(0.20, 0.75, 0.78),
                               vec3(0.70, 0.95, 0.96));
    // cil-bolt      (2)  — plasma yellow
    if (s == 2) return Palette(vec3(0.28, 0.22, 0.02),
                               vec3(0.90, 0.78, 0.10),
                               vec3(1.00, 0.96, 0.65));
    // cil-plus      (3)  — mitosis pink
    if (s == 3) return Palette(vec3(0.30, 0.06, 0.18),
                               vec3(0.88, 0.35, 0.62),
                               vec3(1.00, 0.75, 0.88));
    // cil-arrow-right (4) — orange kinetic
    if (s == 4) return Palette(vec3(0.28, 0.14, 0.02),
                               vec3(0.92, 0.52, 0.10),
                               vec3(1.00, 0.82, 0.55));
    // cil-filter    (5)  — lattice green
    if (s == 5) return Palette(vec3(0.04, 0.22, 0.08),
                               vec3(0.25, 0.82, 0.38),
                               vec3(0.72, 0.98, 0.78));
    // cil-code      (6)  — terminal amber
    if (s == 6) return Palette(vec3(0.18, 0.14, 0.02),
                               vec3(0.82, 0.68, 0.18),
                               vec3(0.98, 0.92, 0.65));
    // cil-layers    (7)  — depth violet
    if (s == 7) return Palette(vec3(0.14, 0.06, 0.30),
                               vec3(0.58, 0.30, 0.90),
                               vec3(0.88, 0.72, 1.00));
    // cil-loop      (8)  — cycle crimson
    if (s == 8) return Palette(vec3(0.28, 0.04, 0.04),
                               vec3(0.92, 0.22, 0.22),
                               vec3(1.00, 0.70, 0.70));
    // cil-graph     (9)  — node cyan
    return          Palette(vec3(0.02, 0.22, 0.28),
                            vec3(0.18, 0.82, 0.92),
                            vec3(0.72, 0.97, 1.00));
}

// ── Voronoi membrane helper ────────────────────────────────────────────────────
// Returns the membrane mask [0,1] at position p (already scaled by cellScale).
//
//   F2 − F1  →  the Voronoi ridge SDF:  = 0 at the membrane,
//                                        > 0 inside a cell interior
//
// stroke(F2-F1, 0.0, thickness, edge) paints a band straddling the ridge.

float membraneMask(vec2 p, float thickness) {
    vec2 F = cellular(p);
    float ridge = F.y - F.x;   // 0 on membrane, > 0 inside
    // soft edge = 0.015 gives smooth anti-aliasing without dFdx overhead
    return stroke(ridge, 0.0, thickness, 0.015);
}

// ── Hexagonal cytoskeleton underlay ───────────────────────────────────────────
// Maps UV into a hex grid and returns a subtle glow near tile boundaries
// (distance to hex centre < 0.5 → interior; large distance → near edge).

float hexGlow(vec2 st, float scale) {
    vec4 h = hexTile(st * scale);
    // h.z = length of the local tile coordinate offset from hex centre
    // (range 0 at centre → ~0.5 at corner)
    float distToCentre = length(h.xy - 0.5);
    // Thin ring near edge of hex cell (cytoskeleton scaffold)
    return smoothstep(0.42, 0.38, distToCentre) *
           smoothstep(0.28, 0.34, distToCentre);
}

// ── Main ──────────────────────────────────────────────────────────────────────

void main() {

    // ── 1. Fetch species palette ──────────────────────────────────────────────
    Palette pal = speciesPalette(u_species);

    // ── 2. Pressure-derived parameters ───────────────────────────────────────
    // High pressure → thinner, more transparent membrane (Laplace law).
    float pClamped    = clamp(u_pressure, 0.0, 1.0);

    float thickness   = u_membraneThickness * (1.0 - 0.60 * pClamped);
    float membraneAlpha = 1.0 - 0.40 * pClamped;
    float hexScale    = u_cellScale * (1.0 + 0.50 * pClamped);

    // ── 3. snoise-driven UV warp (organic distortion of cell nuclei) ──────────
    // Displace UV by a slow-drifting simplex noise field before Voronoi lookup.
    // Amplitude 0.10 keeps cells connected; the two octaves give rich curvature.
    vec2 scaledUV = vUV * u_cellScale;

    float noiseAmp = 0.10;
    // Octave 1: low frequency drift
    vec2 warp1 = vec2(
        snoise(scaledUV * 0.60 + vec2( 0.00, u_time * 0.07)),
        snoise(scaledUV * 0.60 + vec2(17.35, u_time * 0.07))
    );
    // Octave 2: higher-frequency membrane roughness
    vec2 warp2 = vec2(
        snoise(scaledUV * 1.80 + vec2(31.71, u_time * 0.13)),
        snoise(scaledUV * 1.80 + vec2( 8.44, u_time * 0.13))
    ) * 0.35;

    vec2 warpedUV = scaledUV + (warp1 + warp2) * noiseAmp;

    // ── 4. Voronoi membrane mask ───────────────────────────────────────────────
    float membrane = membraneMask(warpedUV, thickness);

    // ── 5. Hex cytoskeleton underlay ──────────────────────────────────────────
    float hex = hexGlow(vUV, hexScale * 0.55);

    // ── 6. Cell interior: F1-based radial gradient ────────────────────────────
    // F1 = distance to nearest cell centre; use it as a soft fill gradient.
    vec2  F_raw      = cellular(warpedUV);
    float cellInterior = smoothstep(0.55, 0.05, F_raw.x);  // 1 at nuclei, 0 near wall

    // ── 7. Colour composition ──────────────────────────────────────────────────
    // Layer order (back to front):
    //   a) dark background
    //   b) cell interior fill (species colour)
    //   c) hex cytoskeleton glow (lighter tint)
    //   d) membrane wall (membrane colour)
    //   e) specular highlight on membrane ridge (thin bright line)

    vec3 col = vec3(0.0);

    // (b) interior fill — fades from bright nuclei to dim membrane wall
    col = mix(col, pal.fill, cellInterior * 0.80);

    // (c) hex cytoskeleton — subtle scaffold lines under membrane
    col = mix(col, pal.fill + 0.15, hex * 0.35);

    // (d) membrane wall
    col = mix(col, pal.membrane, membrane * membraneAlpha);

    // (e) specular highlight: thin bright line along the innermost edge
    //     of the membrane (F2-F1 slightly > 0, i.e. just inside the cell)
    float F_ridge  = F_raw.y - F_raw.x;
    float specLine = stroke(F_ridge, thickness * 0.45, thickness * 0.18, 0.008);
    col = mix(col, pal.highlight, specLine * 0.70 * membraneAlpha);

    // ── 8. Alpha ──────────────────────────────────────────────────────────────
    // The interior fill is semi-transparent; membranes are mostly opaque.
    // Pressure makes the whole cell thinner / more see-through.
    float alpha = clamp(
        cellInterior * 0.45 +
        hex           * 0.12 +
        membrane      * membraneAlpha +
        specLine      * 0.60,
        0.0, 1.0);

    // ── 9. Premultiplied RGBA output ──────────────────────────────────────────
    fragColor = vec4(col * alpha, alpha);
}
{@}voronoi-natural.frag{@}#version 300 es
// ── voronoi-natural.frag ──────────────────────────────────────────────────────
// Voronoi / Worley natural cell-surface texture renderer.
//
// Technique:
//   Renders one of five procedural "natural pattern" modes, each mapped to a
//   Transformer cell species.  All modes derive from a common Voronoi distance
//   field computed in a 3×3 neighbourhood search; the modes diverge in how
//   they interpret F1, F2, and centroid data:
//
//   0 · CELL_DIVISION  — Smooth Voronoi cells, pulsing centroids, species-tinted
//       hue.  Mimics dividing biological cells.  Driven by uTime-animated
//       centroid jitter so cells "breathe" and split at irregular intervals.
//       UV scale tiles at uScale; each cell lights from its centroid outward
//       with a soft radial gradient.
//
//   1 · TORTOISE_SHELL — Worley F2–F1 ridge mask thresholded to a hard-edged
//       cracked-tile pattern.  The ridge width is parameterised by uEdgeWidth.
//       Cells are filled with a slightly randomised base colour so adjacent
//       tiles read as distinct plates, evoking tortoise shell or cracked mud.
//
//   2 · LEAF_VEIN      — Multi-octave fBm Voronoi: four octaves of the Voronoi
//       field are accumulated with halving amplitude and doubling frequency,
//       then the result is thresholded to produce a branching vein network.
//       The vein intensity is modulated by F1 from the coarsest octave so
//       veins near cell centres are thinner and fade to thick at the boundary.
//
//   3 · FOAM           — Blends Voronoi F1 (smooth cells) with Worley F2–F1
//       (ridge foam).  The blend weight is driven by uBlend; at 0 you get
//       pure soft cells, at 1 pure foam ridges.  A thin specular highlight is
//       computed from the pseudo-normal (∇F1) for a soap-bubble sheen effect.
//
//   4 · SCALES         — Voronoi with a distance-based modulation: a sine wave
//       centred on each cell's centroid creates an overlapping scale silhouette.
//       Cells are anisotropically stretched by uAspect so scales read as oval
//       fish/reptile scales.  The edge of each scale adds a drop-shadow via a
//       thin border mask.
//
// All modes support:
//   • uTime-animated centroid motion (jitter + slow drift)
//   • SPH coupling: uDensity scales cell frequency; uVelocity distorts UVs
//   • uFillColor / uEdgeColor for host-controlled species tinting
//   • Premultiplied alpha output
//
// LYGIA math inlined (self-contained, no external includes required):
//   random2 / hash21 — adapted from lygia/math/random.glsl
//   (Voronoi + Worley implementations below are stand-alone GLSL; the
//    upstream/lygia/generative/voronoi.glsl and voronoise.glsl source the
//    same algorithm.  #include paths kept as comments for reference.)
//
// References:
//   upstream/lygia/generative/voronoi.glsl  — Patricio Gonzalez Vivo
//   upstream/lygia/generative/voronoise.glsl
//   src/lib/sph/natural-patterns.ts         — WGSL compute counterpart (M560)
//   caustics.frag                           — project shader style reference
//   Research: xiaodi #M607 — cell-pubsub-loop
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── Varyings ──────────────────────────────────────────────────────────────────

in  vec2  vUV;

out vec4  fragColor;

// ── Uniforms ──────────────────────────────────────────────────────────────────

// Pattern mode — matches NaturalPatternMode index:
//   0 CELL_DIVISION | 1 TORTOISE_SHELL | 2 LEAF_VEIN | 3 FOAM | 4 SCALES
uniform int       uMode;

// Animation
uniform float     uTime;

// Cell geometry
uniform float     uScale;       // cell frequency (cells per UV unit), default 6.0
uniform float     uJitter;      // centroid randomness [0,1], default 0.85
uniform float     uAspect;      // UV x-stretch for SCALES mode, default 1.4

// SPH coupling
uniform float     uDensity;     // [0,1] — scales effective cell frequency
uniform vec2      uVelocity;    // world-space velocity — distorts UVs

// Appearance
uniform vec3      uFillColor;   // cell interior tint
uniform vec3      uEdgeColor;   // cell edge / vein / ridge tint
uniform float     uOpacity;     // master alpha

// Mode-specific parameters
uniform float     uEdgeWidth;   // TORTOISE_SHELL ridge half-width, default 0.08
uniform float     uBlend;       // FOAM: mix(cells,foam), default 0.5
uniform int       uOctaves;     // LEAF_VEIN fBm octave count, default 4

// ─────────────────────────────────────────────────────────────────────────────
// ── Inlined LYGIA random helpers ─────────────────────────────────────────────
// Source: upstream/lygia/math/random.glsl
// License: Patricio Gonzalez Vivo — Prosperity / Patron License
// ─────────────────────────────────────────────────────────────────────────────

// Scalar hash → [0,1]
float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

// 2-D → scalar hash
float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// 2-D → 2-D hash (lygia random2)
vec2 random2(vec2 p) {
    return fract(
        sin(vec2(dot(p, vec2(127.1, 311.7)),
                 dot(p, vec2(269.5, 183.3)))) * 43758.5453123
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ── Voronoi — 3×3 neighbourhood, returns (centroid.xy, F1) ───────────────────
// Adapted from upstream/lygia/generative/voronoi.glsl (Patricio Gonzalez Vivo).
//
// Returns:
//   .xy — animated centroid position (fractional, within [0,1]² cell)
//   .z  — F1 distance to nearest centroid
// ─────────────────────────────────────────────────────────────────────────────

vec3 voronoi(vec2 uv, float time) {
    vec2 i_uv = floor(uv);
    vec2 f_uv = fract(uv);

    vec3 nearest = vec3(0.0, 0.0, 10.0);   // (centroid.x, centroid.y, dist)

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 neighbor = vec2(float(i), float(j));

            // Animated centroid: half-static random + half-sine drift
            vec2 raw    = random2(i_uv + neighbor);
            vec2 point  = 0.5 + 0.5 * sin(time + 6.2831853 * raw);

            vec2 diff   = neighbor + point - f_uv;
            float dist  = length(diff);

            if (dist < nearest.z) {
                nearest.xy = point;
                nearest.z  = dist;
            }
        }
    }
    return nearest;
}

// ─────────────────────────────────────────────────────────────────────────────
// ── Worley F2 — returns F2 distance to second-nearest centroid ───────────────
// ─────────────────────────────────────────────────────────────────────────────

vec2 worleyF1F2(vec2 uv, float time) {
    vec2 i_uv = floor(uv);
    vec2 f_uv = fract(uv);

    float f1 = 10.0;
    float f2 = 10.0;

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 neighbor = vec2(float(i), float(j));
            vec2 raw      = random2(i_uv + neighbor);
            vec2 point    = 0.5 + 0.5 * sin(time * 0.7 + 6.2831853 * raw);
            float dist    = length(neighbor + point - f_uv);
            if (dist < f1) { f2 = f1; f1 = dist; }
            else if (dist < f2) { f2 = dist; }
        }
    }
    return vec2(f1, f2);
}

// ─────────────────────────────────────────────────────────────────────────────
// ── Finite-difference pseudo-normal from F1 field ────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

vec2 voronoiNormal(vec2 uv, float time, float eps) {
    float dx = voronoi(uv + vec2(eps, 0.0), time).z
             - voronoi(uv - vec2(eps, 0.0), time).z;
    float dy = voronoi(uv + vec2(0.0, eps), time).z
             - voronoi(uv - vec2(0.0, eps), time).z;
    return normalize(vec2(dx, dy) + 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// ── Mode helpers ─────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

// CELL_DIVISION — smooth radial cells with species fill, pulsing centroids.
vec4 modeCellDivision(vec2 uv, float time) {
    vec3 v    = voronoi(uv, time);
    float f1  = v.z;

    // Unique per-cell colour variation using centroid as seed.
    float hue = hash21(v.xy + vec2(13.7, 41.3));
    vec3 cellTint = mix(uFillColor, uFillColor * (0.6 + 0.7 * hue), 0.4);

    // Radial gradient from centroid outward.
    float radial = 1.0 - smoothstep(0.0, 0.55, f1);

    // Thin border highlight.
    float border = smoothstep(0.40, 0.42, f1) * smoothstep(0.50, 0.48, f1);

    vec3 col = mix(cellTint * radial, uEdgeColor, border * 0.6);
    float alpha = clamp(1.0 - smoothstep(0.45, 0.52, f1) * (1.0 - border), 0.0, 1.0);

    return vec4(col * alpha, alpha) * uOpacity;
}

// TORTOISE_SHELL — Worley F2–F1 ridge cracked-tile pattern.
vec4 modeTortoiseshell(vec2 uv, float time) {
    vec2 ff   = worleyF1F2(uv, time);
    float ridge = ff.y - ff.x;   // [0, ~0.5]

    // Tile fill: per-cell random tint.
    float cellID = hash21(floor(uv));
    vec3 tileCol = mix(uFillColor, uFillColor * (0.5 + 0.9 * cellID), 0.55);

    // Ridge mask: thick edge → uEdgeColor, interior → tileCol.
    float edgeMask = smoothstep(uEdgeWidth, uEdgeWidth * 0.3, ridge);

    vec3 col = mix(tileCol, uEdgeColor, edgeMask);

    // Interior slight concavity shading from F1.
    float shade = 0.75 + 0.25 * smoothstep(0.0, 0.5, ff.x);
    col *= shade;

    float alpha = 1.0;
    return vec4(col * alpha, alpha) * uOpacity;
}

// LEAF_VEIN — multi-octave fBm Voronoi thresholded to branching veins.
vec4 modeLeafVein(vec2 uv, float time) {
    // Course octave for global cell envelope.
    vec3  v0 = voronoi(uv, time);
    float envelope = 1.0 - smoothstep(0.0, 0.45, v0.z);

    // fBm accumulation over uOctaves.
    float vfbm = 0.0;
    float amp   = 1.0;
    float freq  = 1.0;
    float norm  = 0.0;
    for (int oct = 0; oct < 6; ++oct) {
        if (oct >= uOctaves) break;
        vec3 vo = voronoi(uv * freq, time * (0.3 + float(oct) * 0.1));
        vfbm   += amp * vo.z;
        norm   += amp;
        amp    *= 0.5;
        freq   *= 2.1;
    }
    vfbm /= norm;   // normalised to [0, ~1]

    // Threshold: thin bright veins at cell boundary ridges.
    float veinT    = 0.52;
    float veinW    = 0.055;
    float veinMask = smoothstep(veinT - veinW, veinT, vfbm)
                   * smoothstep(veinT + veinW, veinT, vfbm);

    // Vein intensity fades near cell centres (thinner at centre).
    float veinIntensity = veinMask * (0.3 + 0.7 * v0.z);

    vec3 col   = mix(uFillColor * envelope, uEdgeColor, veinIntensity);
    float alpha = clamp(envelope + veinIntensity * 0.8, 0.0, 1.0);

    return vec4(col * alpha, alpha) * uOpacity;
}

// FOAM — blend of smooth Voronoi cells with Worley ridge foam.
vec4 modeFoam(vec2 uv, float time) {
    // Smooth cell fill from Voronoi F1.
    vec3  v    = voronoi(uv, time * 0.8);
    float f1   = v.z;
    float cell = 1.0 - smoothstep(0.0, 0.5, f1);

    // Foam ridges from Worley F2–F1.
    vec2  ff    = worleyF1F2(uv, time * 0.6);
    float foam  = smoothstep(0.18, 0.0, ff.y - ff.x);   // thin ridge highlight

    // Specular sheen from pseudo-normal (soap-bubble).
    vec2  nor   = voronoiNormal(uv, time * 0.8, 0.008);
    vec3  light = normalize(vec3(0.6, 1.0, 0.8));
    float spec  = pow(max(0.0, dot(vec3(nor, 0.0), light)), 12.0) * 0.35;

    // Blend cell body + foam ridge.
    float cellB = clamp(uBlend, 0.0, 1.0);
    float base  = mix(cell, foam, cellB);

    vec3 col    = mix(uFillColor * cell, uEdgeColor, foam * 0.7);
    col        += uEdgeColor * spec;

    float alpha = clamp(base + spec, 0.0, 1.0);
    return vec4(col * alpha, alpha) * uOpacity;
}

// SCALES — Voronoi with anisotropic sine-distance scale silhouettes.
vec4 modeScales(vec2 uv, float time) {
    // Anisotropic stretch so scales are oval.
    vec2 suv = vec2(uv.x * uAspect, uv.y);

    vec3  v    = voronoi(suv, time * 0.5);
    float f1   = v.z;

    // Per-scale random tint.
    float scaleID  = hash21(v.xy + vec2(7.3, 19.1));
    vec3  scaleCol = mix(uFillColor, uFillColor * (0.5 + 0.8 * scaleID), 0.45);

    // Sine modulation centred on centroid → scale silhouette.
    float sineD = 0.5 + 0.5 * sin(f1 * 8.0 - time * 1.5);
    float scaleMask = smoothstep(0.55, 0.35, f1 * (0.6 + 0.4 * sineD));

    // Thin drop-shadow border just inside the scale edge.
    float shadow = smoothstep(0.38, 0.42, f1) * smoothstep(0.48, 0.44, f1);

    vec3 col = scaleCol * scaleMask;
    col = mix(col, uEdgeColor * 0.4, shadow * 0.5);

    float alpha = scaleMask;
    return vec4(col * alpha, alpha) * uOpacity;
}

// ─────────────────────────────────────────────────────────────────────────────
// ── main ──────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

void main() {
    // ── 1. SPH velocity distortion ────────────────────────────────────────────
    // Distort UVs in the direction of the local fluid velocity vector.
    // Magnitude is capped so it never exceeds ±0.05 UV units.
    float velMag  = length(uVelocity);
    vec2  velDir  = velMag > 1e-4 ? uVelocity / velMag : vec2(0.0);
    float distAmt = min(velMag * 0.02, 0.05);
    vec2  baseUV  = vUV + velDir * distAmt;

    // ── 2. SPH density → effective cell scale ────────────────────────────────
    // Higher density packs cells tighter (up to 2× the base scale).
    float effScale = uScale * (1.0 + uDensity);
    vec2  uv       = baseUV * effScale;

    // ── 3. Jitter-modulated time for centroid animation ───────────────────────
    // uJitter [0,1] controls how much the centroids drift over time.
    float animTime = uTime * uJitter;

    // ── 4. Dispatch to mode ───────────────────────────────────────────────────
    vec4 result;

    if      (uMode == 0) result = modeCellDivision  (uv, animTime);
    else if (uMode == 1) result = modeTortoiseshell (uv, animTime);
    else if (uMode == 2) result = modeLeafVein      (uv, animTime);
    else if (uMode == 3) result = modeFoam          (uv, animTime);
    else                 result = modeScales        (uv, animTime);

    // ── 5. Premultiplied RGBA output ──────────────────────────────────────────
    fragColor = result;
}
