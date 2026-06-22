#version 300 es
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
