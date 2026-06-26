#version 300 es
precision mediump float;
out vec4 fragColor;

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

    fragColor = vec4(col, alpha);
}
