#version 300 es
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
