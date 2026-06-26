#version 300 es
/**
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
out vec4 fragColor;

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

    fragColor = vec4(col, alpha * u_opacity);
}
