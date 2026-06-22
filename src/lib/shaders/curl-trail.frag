#version 300 es
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
