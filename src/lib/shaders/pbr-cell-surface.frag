#version 300 es
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
