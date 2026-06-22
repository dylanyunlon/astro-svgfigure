#version 300 es
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
