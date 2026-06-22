#version 300 es
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
