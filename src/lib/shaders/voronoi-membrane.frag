#version 300 es
// ── voronoi-membrane.frag ──────────────────────────────────────────────────────
// M747: Voronoi cell membrane — soft F2-F1 translucent membrane rendering.
//
// Biology model
// ─────────────
// Real biological cell membranes ARE Voronoi diagrams: each cell expands
// outward from its nucleus until it meets its neighbours — the membrane sits
// exactly at the equidistant boundary between cell centres.  This shader
// reconstructs that geometry procedurally:
//
//   1. VORONOI F2-F1 MEMBRANE — No hard borders.  The membrane is the
//      Voronoi ridge field F2 − F1, rendered as a semi-transparent band
//      with smooth falloff on both sides.  The opacity of the membrane is
//      proportional to exp(−(F2−F1)²/σ²), giving a soft, diffuse wall
//      that fades to nothing at cell interiors.  σ (membrane width)
//      breathes with sin(time) to simulate living cell pulsation.
//
//   2. DOMAIN-WARPED MEMBRANE TEXTURE — The membrane surface is not
//      featureless: a multi-octave snoise warp displaces a secondary
//      Voronoi lookup to produce an organic lipid-bilayer / endoplasmic
//      reticulum texture visible only where the membrane is opaque.
//      This gives each membrane wall a unique veined, fibrous character.
//
//   3. BREATHING PULSATION — The membrane width σ is modulated by
//      sin(u_time * u_breathRate), so the cell boundary gently expands
//      and contracts like a breathing diaphragm.  The amplitude is small
//      (±15% of base width) to keep topology stable.
//
//   4. COLLISION SPARKS — When u_contactCount > 0 (cells are touching),
//      the membrane brightens at the ridge with a hot highlight flash.
//      The spark intensity is proportional to contact count and inversely
//      proportional to F2−F1 (strongest exactly on the boundary).  This
//      mimics junction potential / membrane depolarisation at contact.
//
//   5. HEX UNDERLAY — hexTile(st) maps the same UV into a flat-topped
//      hexagonal grid.  The distance-to-hex-centre (hexTile.z) becomes a
//      subtle honeycomb relief underneath the Voronoi membrane, mimicking
//      the cytoskeleton scaffold that enforces the nearly-hexagonal packing
//      observed in epithelial sheets.
//
//   6. SPECIES PALETTE — u_species selects one of 10 colour palettes that
//      map to the cell icon species used elsewhere in the project.  Each
//      palette has an inner fill colour, a membrane colour, and a highlight
//      tint so the membrane texture stays visually consistent with the
//      species SDF shapes.
//
//   7. PRESSURE PHYSICS — u_pressure (SPH scalar, typically 0–1) drives:
//       • membrane thinning  → σ ∝ (1 − 0.5 · pressure)
//       • membrane opacity   → alpha ∝ (1 − 0.3 · pressure)
//       • hex relief scale   → hexScale ∝ (1 + 0.5 · pressure)
//      High-pressure cells become thin-membraned and translucent, matching
//      fluid-dynamics intuition (compressed gas cell, osmotic pressure).
//
// GLSL #include dependencies (resolved by the project GLSL preprocessor)
// ───────────────────────────────────────────────────────────────────────
//   ../../upstream/webgl-noise/src/cellular2D.glsl  — cellular() / Worley noise
//   ../../upstream/lygia/generative/snoise.glsl     — snoise(vec2)
//   ../../upstream/lygia/space/hexTile.glsl         — hexTile(vec2)
//
// Uniforms
// ────────
//   u_cellScale         float   Voronoi cell spatial frequency          ≈ 6.0
//   u_membraneThickness float   Base membrane Gaussian σ [0.01–0.20]    ≈ 0.08
//   u_pressure          float   SPH pressure [0,1]; thins membrane      ≈ 0.0
//   u_species           int     Species index [0,9]; selects palette
//   u_time              float   Elapsed seconds; drives all animation
//   u_contactCount      float   SPH contact count [0,N]; collision sparks
//   u_breathRate        float   Breathing frequency (rad/s)             ≈ 1.2
//
// Varyings (from bbox-quad vertex shader)
//   vUV                 vec2    [0,1]² local bbox coordinates
//
// Output
//   fragColor           vec4    premultiplied RGBA
//
// References
//   Stefan Gustavson (2011) — Cellular noise in GLSL
//   Patricio Gonzalez Vivo — lygia.xyz (hexTile, snoise)
//   Nienhaus & Koster (2002) — Voronoi tessellation of biological tissues
//   Farhadifar et al. (2007) — The influence of cell mechanics on epithelium
//   xiaodi #M747 — voronoi cell membrane
// ──────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── Varyings ──────────────────────────────────────────────────────────────────

in  vec2  vUV;

out vec4  fragColor;

// ── Uniforms ──────────────────────────────────────────────────────────────────

uniform float u_cellScale;          // e.g. 6.0  — cells per UV unit
uniform float u_membraneThickness;  // e.g. 0.08 — base membrane Gaussian σ
uniform float u_pressure;           // [0,1]     — SPH pressure
uniform int   u_species;            // [0,9]     — species palette index
uniform float u_time;               // seconds
uniform float u_contactCount;       // [0,N]     — SPH contact count for sparks
uniform float u_breathRate;         // rad/s     — breathing pulsation speed

// ── Inlined lygia math helpers ────────────────────────────────────────────────
// Sources: upstream/lygia/math/{mod289,permute,taylorInvSqrt,saturate}.glsl
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

// ── Hexagonal cytoskeleton underlay ───────────────────────────────────────────
// Maps UV into a hex grid and returns a subtle glow near tile boundaries.

float hexGlow(vec2 st, float scale) {
    vec4 h = hexTile(st * scale);
    float distToCentre = length(h.xy - 0.5);
    return smoothstep(0.42, 0.38, distToCentre) *
           smoothstep(0.28, 0.34, distToCentre);
}

// ── Domain warp helper ────────────────────────────────────────────────────────
// Three-octave domain warp: displaces input coordinates by layered snoise
// to produce an organic, fibrous distortion field for the membrane texture.
//
// Returns the warped coordinate.  Each octave doubles frequency and halves
// amplitude, with time-dependent phase offsets for slow drift.

vec2 domainWarp(vec2 p, float time) {
    // Octave 1 — large-scale cell drift
    vec2 w1 = vec2(
        snoise(p * 0.55 + vec2( 0.00, time * 0.06)),
        snoise(p * 0.55 + vec2(17.35, time * 0.06))
    );
    // Octave 2 — mid-scale membrane roughness
    vec2 w2 = vec2(
        snoise(p * 1.40 + vec2(31.71, time * 0.12) + w1 * 0.4),
        snoise(p * 1.40 + vec2( 8.44, time * 0.12) + w1 * 0.4)
    ) * 0.45;
    // Octave 3 — fine grain fibrillation (visible only on membrane)
    vec2 w3 = vec2(
        snoise(p * 3.50 + vec2(53.12, time * 0.18) + w2 * 0.3),
        snoise(p * 3.50 + vec2(22.78, time * 0.18) + w2 * 0.3)
    ) * 0.20;

    return p + (w1 + w2 + w3) * 0.12;
}

// ── Membrane texture ──────────────────────────────────────────────────────────
// A secondary Voronoi lookup through the domain-warped coordinate space
// produces a veined, lipid-bilayer-like texture.  This texture is only
// visible where the membrane opacity is non-zero.
//
// Returns a [0,1] greyscale texture value: 0 = vein, 1 = smooth membrane.

float membraneTexture(vec2 p, float time) {
    // Domain-warp the coordinate before secondary Voronoi lookup
    vec2 warped = domainWarp(p * 2.8, time);
    vec2 F = cellular(warped);
    float ridge = F.y - F.x;

    // Map the secondary F2-F1 into a vein pattern:
    // thin dark veins where ridge ≈ 0, smooth between.
    float vein = smoothstep(0.0, 0.12, ridge);

    // Add a subtle granularity from snoise for lipid texture
    float grain = snoise(warped * 4.0 + vec2(time * 0.05)) * 0.12 + 0.88;

    return vein * grain;
}

// ── Main ──────────────────────────────────────────────────────────────────────

void main() {

    // ── 1. Fetch species palette ──────────────────────────────────────────────
    Palette pal = speciesPalette(u_species);

    // ── 2. Pressure & breathing parameters ───────────────────────────────────
    float pClamped = clamp(u_pressure, 0.0, 1.0);

    // Breathing: sin(time) modulates membrane width ±15%
    float breathPhase  = sin(u_time * u_breathRate);
    float breathFactor = 1.0 + 0.15 * breathPhase;

    // Pressure thins the membrane (Laplace law); breathing widens/narrows it.
    float sigma = u_membraneThickness * (1.0 - 0.50 * pClamped) * breathFactor;

    // Membrane base opacity: pressure reduces, breathing subtly modulates.
    float membraneBaseAlpha = (1.0 - 0.30 * pClamped) * (0.92 + 0.08 * breathPhase);

    float hexScale = u_cellScale * (1.0 + 0.50 * pClamped);

    // ── 3. Domain-warped UV for primary Voronoi ──────────────────────────────
    vec2 scaledUV = vUV * u_cellScale;
    vec2 warpedUV = domainWarp(scaledUV, u_time);

    // ── 4. Voronoi F1 / F2 — primary cell structure ─────────────────────────
    vec2 F = cellular(warpedUV);
    float f1    = F.x;       // distance to nearest cell centre
    float f2    = F.y;       // distance to second-nearest
    float ridge = f2 - f1;   // 0 on the membrane, > 0 inside cell

    // ── 5. Soft F2-F1 membrane mask (Gaussian-like falloff) ──────────────────
    // Instead of a hard stroke, the membrane is a smooth Gaussian band
    // centred on the F2=F1 ridge.  The width σ controls how diffuse the
    // membrane appears.  The exp(-x²/σ²) shape gives soft edges that fade
    // naturally — no hard borders.
    float ridgeNorm = ridge / max(sigma, 0.001);
    float membraneMask = exp(-ridgeNorm * ridgeNorm);

    // Boost the membrane core slightly (the very centre of the ridge is
    // maximally opaque; edges are translucent).
    membraneMask = pow(membraneMask, 0.7);

    // ── 6. Membrane surface texture (domain-warped secondary Voronoi) ────────
    float memTex = membraneTexture(scaledUV, u_time);

    // ── 7. Collision sparks ──────────────────────────────────────────────────
    // When u_contactCount > 0, the membrane ridge brightens with a hot flash.
    // Spark intensity: strongest at the ridge (small F2-F1), fades into interior.
    // A high-frequency flicker (sin²) adds visual energy to the spark.
    float contacts = clamp(u_contactCount, 0.0, 8.0);
    float sparkBase = contacts / 8.0;  // normalised [0, 1]
    float sparkRidge = exp(-ridge * ridge / (sigma * sigma * 0.25));  // tight Gaussian on ridge
    float sparkFlicker = 0.7 + 0.3 * sin(u_time * 18.0 + ridge * 40.0);
    float spark = sparkBase * sparkRidge * sparkFlicker;

    // ── 8. Hex cytoskeleton underlay ─────────────────────────────────────────
    float hex = hexGlow(vUV, hexScale * 0.55);

    // ── 9. Cell interior: F1-based radial gradient ───────────────────────────
    float cellInterior = smoothstep(0.55, 0.05, f1);

    // ── 10. Colour composition ───────────────────────────────────────────────
    // Layer order (back to front):
    //   a) dark background
    //   b) cell interior fill (species colour, semi-transparent)
    //   c) hex cytoskeleton glow (lighter tint, subtle)
    //   d) membrane wall — textured, translucent F2-F1 band
    //   e) membrane highlight — thin bright inner edge
    //   f) collision sparks — hot highlight on contact

    vec3 col = vec3(0.0);

    // (b) interior fill — fades from bright nuclei to dim membrane wall
    col = mix(col, pal.fill, cellInterior * 0.75);

    // (c) hex cytoskeleton — subtle scaffold lines under membrane
    col = mix(col, pal.fill + 0.15, hex * 0.30);

    // (d) membrane wall — textured, translucent
    //     The membrane colour is modulated by the secondary Voronoi texture:
    //     veins appear as darker channels within the membrane surface.
    vec3 memCol = pal.membrane * (0.6 + 0.4 * memTex);
    float memOpacity = membraneMask * membraneBaseAlpha;
    col = mix(col, memCol, memOpacity);

    // (e) membrane highlight — thin bright inner edge (just inside the ridge)
    //     Uses a tighter Gaussian than the main membrane for a specular-like line.
    float highlightSigma = sigma * 0.35;
    float highlightNorm  = (ridge - sigma * 0.3) / max(highlightSigma, 0.001);
    float highlight      = exp(-highlightNorm * highlightNorm) * 0.65;
    col = mix(col, pal.highlight, highlight * membraneBaseAlpha);

    // (f) collision sparks — hot white/highlight flash at contact points
    vec3 sparkCol = mix(pal.highlight, vec3(1.0), 0.5);  // push toward white
    col = mix(col, sparkCol, spark * 0.85);

    // ── 11. Alpha ────────────────────────────────────────────────────────────
    // Cell interior is semi-transparent; membrane is the dominant alpha source.
    // Sparks add extra opacity.
    float alpha = clamp(
        cellInterior   * 0.40 +
        hex            * 0.10 +
        memOpacity     * 1.00 +
        highlight      * membraneBaseAlpha * 0.55 +
        spark          * 0.90,
        0.0, 1.0);

    // ── 12. Premultiplied RGBA output ────────────────────────────────────────
    fragColor = vec4(col * alpha, alpha);
}
