#version 300 es
/**
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
out vec4 fragColor;

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
 *   fragColor = vec4(u_fillColor, alpha * u_opacity);
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

    fragColor = vec4(col, alpha * u_opacity);
}

#endif // SPECIES_LIBRARY_STANDALONE
