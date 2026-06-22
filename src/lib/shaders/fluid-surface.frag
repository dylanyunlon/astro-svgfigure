#version 300 es
// ── fluid-surface.frag ────────────────────────────────────────────────────────
// SPH Metaball fluid surface renderer.
//
// Technique:
//   Each draw call covers the bounding quad of an SPH cell cluster.  We
//   reconstruct a smooth implicit surface by blending the per-particle density
//   field via a radial metaball kernel, then shade the resulting isosurface
//   with three coupled effects:
//
//   1. DENSITY SMOOTHSTEP  — The scalar field is thresholded with smoothstep to
//      produce a clean liquid surface with an adjustable transition band.
//      Pixels above the iso-threshold are "inside" the fluid; the gradient
//      magnitude becomes the surface normal estimate for cheap specular.
//
//   2. COLLISION RIPPLES   — On each rigid-body ↔ fluid collision event the
//      host writes a ping-pong ripple texture (uRippleTex).  The lygia
//      ripple.glsl propagation kernel advances the wave across the surface
//      and the resulting height offset perturbs the iso-field so collisions
//      visibly dimple / splash the fluid.
//
//   3. VELOCITY DISPERSION — The SPH velocity field (uVelocityTex, RG) drives
//      two things:
//        a. Classic Perlin cnoise (lygia/generative/cnoise.glsl) modulated by
//           speed magnitude produces turbulent micro-wrinkles on fast-moving
//           regions.
//        b. Chromatic aberration (lygia/distort/chromaAB.glsl) is applied to
//           the final surface colour: high-speed areas get stronger RGB channel
//           separation, giving a prismatic dispersion effect at wave fronts.
//
// LYGIA virtual includes (resolved at shader-compile time by the project
// GLSL preprocessor / vite-plugin-glsl):
//   #include "../../upstream/lygia/simulate/ripple.glsl"
//   #include "../../upstream/lygia/generative/cnoise.glsl"
//   #include "../../upstream/lygia/distort/chromaAB.glsl"
//
// All three LYGIA sources are inlined below following the project convention
// (see curl-trail.frag, grayscott-species.frag) so the file is self-contained.
//
// Uniforms:
//   uDensityTex      — R: SPH particle density field          sampler2D
//   uVelocityTex     — RG: particle velocity (world-space)    sampler2D
//   uRippleTex       — RG: ripple ping-pong (current frame)   sampler2D
//   uTexelSize       — 1 / texture resolution                 vec2
//   uTime            — elapsed seconds                        float
//   uIsoThreshold    — metaball iso-level [0,1], default 0.5  float
//   uSmoothBand      — smoothstep transition half-width        float
//   uFluidColor      — base fluid tint RGB                    vec3
//   uSpecularColor   — specular highlight RGB                  vec3
//   uSpecularPower   — Phong exponent                         float
//   uRippleStrength  — ripple displacement scale               float
//   uNoiseScale      — cnoise spatial frequency               float
//   uNoiseStrength   — cnoise displacement amplitude           float
//   uChromaStrength  — chromaAB aberration scale              float
//   uMaxSpeed        — velocity normalisation reference        float
//
// Varyings (from fullscreen-quad vertex shader):
//   vUV              — [0,1]² texture coordinates
//
// Output:
//   fragColor        — premultiplied RGBA
//
// Research: xiaodi #M553 — cell-pubsub-loop
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── Varyings ──────────────────────────────────────────────────────────────────

in  vec2  vUV;

out vec4  fragColor;

// ── Uniforms ──────────────────────────────────────────────────────────────────

uniform sampler2D uDensityTex;      // R  = SPH density  [0,1]
uniform sampler2D uVelocityTex;     // RG = velocity (world units / s, normalised)
uniform sampler2D uRippleTex;       // RG = ripple propagation buffer (current)
uniform vec2      uTexelSize;       // vec2(1.0/w, 1.0/h)

uniform float     uTime;

// Surface reconstruction
uniform float     uIsoThreshold;    // default 0.50
uniform float     uSmoothBand;      // default 0.08  — transition half-width

// Shading
uniform vec3      uFluidColor;      // e.g. vec3(0.18, 0.52, 0.89)
uniform vec3      uSpecularColor;   // e.g. vec3(0.90, 0.95, 1.00)
uniform float     uSpecularPower;   // e.g. 32.0

// Ripple collision
uniform float     uRippleStrength;  // e.g. 0.06

// Velocity cnoise wrinkle
uniform float     uNoiseScale;      // e.g. 4.0
uniform float     uNoiseStrength;   // e.g. 0.04

// Chromatic aberration
uniform float     uChromaStrength;  // e.g. 2.0
uniform float     uMaxSpeed;        // normalisation, e.g. 5.0

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/saturate.glsl" (inlined) ────────────────────────────
// Source: upstream/lygia/math/saturate.glsl
// License: Patricio Gonzalez Vivo — Prosperity / Patron License
// ─────────────────────────────────────────────────────────────────────────────

#if !defined(FNC_SATURATE) && !defined(saturate)
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/mod289.glsl" (inlined) ───────────────────────────────
// Source: upstream/lygia/math/mod289.glsl
// contributors: Stefan Gustavson, Ian McEwan
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_MOD289
#define FNC_MOD289
float mod289(const in float x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2  mod289(const in vec2  x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3  mod289(const in vec3  x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4  mod289(const in vec4  x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/permute.glsl" (inlined) ──────────────────────────────
// Source: upstream/lygia/math/permute.glsl
// contributors: Stefan Gustavson, Ian McEwan
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_PERMUTE
#define FNC_PERMUTE
float permute(const in float v) { return mod289(((v * 34.0) + 1.0) * v); }
vec2  permute(const in vec2  v) { return mod289(((v * 34.0) + 1.0) * v); }
vec3  permute(const in vec3  v) { return mod289(((v * 34.0) + 1.0) * v); }
vec4  permute(const in vec4  v) { return mod289(((v * 34.0) + 1.0) * v); }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/taylorInvSqrt.glsl" (inlined) ────────────────────────
// Source: upstream/lygia/math/taylorInvSqrt.glsl
// contributors: Stefan Gustavson, Ian McEwan
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_TAYLORINVSQRT
#define FNC_TAYLORINVSQRT
float taylorInvSqrt(in float r) { return 1.79284291400159 - 0.85373472095314 * r; }
vec2  taylorInvSqrt(in vec2  r) { return 1.79284291400159 - 0.85373472095314 * r; }
vec3  taylorInvSqrt(in vec3  r) { return 1.79284291400159 - 0.85373472095314 * r; }
vec4  taylorInvSqrt(in vec4  r) { return 1.79284291400159 - 0.85373472095314 * r; }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/quintic.glsl" (inlined) ──────────────────────────────
// Source: upstream/lygia/math/quintic.glsl
// contributors: Inigo Quiles
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_QUINTIC
#define FNC_QUINTIC
float quintic(const in float v) { return v*v*v*(v*(v*6.0-15.0)+10.0); }
vec2  quintic(const in vec2  v) { return v*v*v*(v*(v*6.0-15.0)+10.0); }
vec3  quintic(const in vec3  v) { return v*v*v*(v*(v*6.0-15.0)+10.0); }
vec4  quintic(const in vec4  v) { return v*v*v*(v*(v*6.0-15.0)+10.0); }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/generative/cnoise.glsl" (inlined) ─────────────────────────
// Source: upstream/lygia/generative/cnoise.glsl
// contributors: Stefan Gustavson, Ian McEwan
// description: Classic Perlin Noise  https://github.com/stegu/webgl-noise
// License: MIT  https://opensource.org/license/mit/
//
// Only the vec2 and vec3 overloads are included; the vec4 overload is omitted
// to reduce code size — only 2D / 3D sampling is used in this shader.
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_CNOISE
#define FNC_CNOISE

// ── 2D Classic Perlin Noise ───────────────────────────────────────────────────
float cnoise(in vec2 P) {
    vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
    vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
    Pi = mod289(Pi);
    vec4 ix = Pi.xzxz;
    vec4 iy = Pi.yyww;
    vec4 fx = Pf.xzxz;
    vec4 fy = Pf.yyww;
    vec4 i  = permute(permute(ix) + iy);
    vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0;
    vec4 gy = abs(gx) - 0.5;
    vec4 tx = floor(gx + 0.5);
    gx = gx - tx;
    vec2 g00 = vec2(gx.x, gy.x);
    vec2 g10 = vec2(gx.y, gy.y);
    vec2 g01 = vec2(gx.z, gy.z);
    vec2 g11 = vec2(gx.w, gy.w);
    vec4 norm = taylorInvSqrt(vec4(dot(g00,g00), dot(g01,g01), dot(g10,g10), dot(g11,g11)));
    g00 *= norm.x;  g01 *= norm.y;  g10 *= norm.z;  g11 *= norm.w;
    float n00 = dot(g00, vec2(fx.x, fy.x));
    float n10 = dot(g10, vec2(fx.y, fy.y));
    float n01 = dot(g01, vec2(fx.z, fy.z));
    float n11 = dot(g11, vec2(fx.w, fy.w));
    vec2 fade_xy = quintic(Pf.xy);
    vec2 n_x     = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
    float n_xy   = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}

// ── 3D Classic Perlin Noise ───────────────────────────────────────────────────
float cnoise(in vec3 P) {
    vec3 Pi0 = floor(P);
    vec3 Pi1 = Pi0 + vec3(1.0);
    Pi0 = mod289(Pi0);
    Pi1 = mod289(Pi1);
    vec3 Pf0 = fract(P);
    vec3 Pf1 = Pf0 - vec3(1.0);
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = Pi0.zzzz;
    vec4 iz1 = Pi1.zzzz;
    vec4 ixy  = permute(permute(ix) + iy);
    vec4 ixy0 = permute(ixy + iz0);
    vec4 ixy1 = permute(ixy + iz1);
    vec4 gx0 = ixy0 * (1.0 / 7.0);
    vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
    gx0 = fract(gx0);
    vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
    vec4 sz0 = step(gz0, vec4(0.0));
    gx0 -= sz0 * (step(0.0, gx0) - 0.5);
    gy0 -= sz0 * (step(0.0, gy0) - 0.5);
    vec4 gx1 = ixy1 * (1.0 / 7.0);
    vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
    gx1 = fract(gx1);
    vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
    vec4 sz1 = step(gz1, vec4(0.0));
    gx1 -= sz1 * (step(0.0, gx1) - 0.5);
    gy1 -= sz1 * (step(0.0, gy1) - 0.5);
    vec3 g000 = vec3(gx0.x, gy0.x, gz0.x);
    vec3 g100 = vec3(gx0.y, gy0.y, gz0.y);
    vec3 g010 = vec3(gx0.z, gy0.z, gz0.z);
    vec3 g110 = vec3(gx0.w, gy0.w, gz0.w);
    vec3 g001 = vec3(gx1.x, gy1.x, gz1.x);
    vec3 g101 = vec3(gx1.y, gy1.y, gz1.y);
    vec3 g011 = vec3(gx1.z, gy1.z, gz1.z);
    vec3 g111 = vec3(gx1.w, gy1.w, gz1.w);
    vec4 norm0 = taylorInvSqrt(vec4(dot(g000,g000), dot(g010,g010), dot(g100,g100), dot(g110,g110)));
    g000 *= norm0.x;  g010 *= norm0.y;  g100 *= norm0.z;  g110 *= norm0.w;
    vec4 norm1 = taylorInvSqrt(vec4(dot(g001,g001), dot(g011,g011), dot(g101,g101), dot(g111,g111)));
    g001 *= norm1.x;  g011 *= norm1.y;  g101 *= norm1.z;  g111 *= norm1.w;
    float n000 = dot(g000, Pf0);
    float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
    float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
    float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
    float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
    float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
    float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
    float n111 = dot(g111, Pf1);
    vec3 fade_xyz = quintic(Pf0);
    vec4 n_z  = mix(vec4(n000,n100,n010,n110), vec4(n001,n101,n011,n111), fade_xyz.z);
    vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
    float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
    return 2.2 * n_xyz;
}

#endif // FNC_CNOISE

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/simulate/ripple.glsl" (inlined) ───────────────────────────
// Source: upstream/lygia/simulate/ripple.glsl
// original_author: Patricio Gonzalez Vivo
// description: Simple Ripple Propagation — advances a 2-channel ping-pong
//   buffer that stores current (r) and previous (g) wave heights.
//   Returns vec3(next_height, current_height, 0).
// License: Patricio Gonzalez Vivo — Prosperity / Patron License
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_RIPPLE
#define FNC_RIPPLE
vec3 ripple(sampler2D tex, vec2 st, vec2 pixel) {
    vec3 rta = texture(tex, st).rgb;
    float s0 = rta.y;
    float s1 = texture(tex, st + vec2( 0.0,       -pixel.y)).r;   //     s1
    float s2 = texture(tex, st + vec2(-pixel.x,    0.0    )).r;   //  s2 s0 s3
    float s3 = texture(tex, st + vec2( pixel.x,    0.0    )).r;   //     s4
    float s4 = texture(tex, st + vec2( 0.0,        pixel.y)).r;
    float d  = -(s0 - 0.5) * 2.0 + (s1 + s2 + s3 + s4 - 2.0);
    d *= 0.99;                          // damping
    d  = saturate(d * 0.5 + 0.5);
    return vec3(d, rta.x, 0.0);
}
#endif // FNC_RIPPLE

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/math/lengthSq.glsl" (inlined) ─────────────────────────────
// Source: upstream/lygia/math/lengthSq.glsl
// contributors: Patricio Gonzalez Vivo
// ─────────────────────────────────────────────────────────────────────────────

#ifndef FNC_LENGTHSQ
#define FNC_LENGTHSQ
float lengthSq(in vec2 v) { return dot(v, v); }
float lengthSq(in vec3 v) { return dot(v, v); }
float lengthSq(in vec4 v) { return dot(v, v); }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ── #include "lygia/distort/chromaAB.glsl" (inlined) ──────────────────────────
// Source: upstream/lygia/distort/chromaAB.glsl
// contributors: Patricio Gonzalez Vivo, Johan Ismael
// description: Chromatic aberration — shifts R/G/B channels by a scaled
//   offset derived from the distance of st from the frame centre.
// License: Patricio Gonzalez Vivo — Prosperity / Patron License
// ─────────────────────────────────────────────────────────────────────────────

#ifndef CHROMAAB_PCT
#define CHROMAAB_PCT 1.5
#endif

#ifndef FNC_CHROMAAB
#define FNC_CHROMAAB

// Overload: explicit RGB distortion channels
vec3 chromaAB(in sampler2D tex, in vec2 st, in vec2 direction, in vec3 distortion) {
    vec3 c;
    c.r = texture(tex, st + direction * distortion.r).r;
    c.g = texture(tex, st + direction * distortion.g).g;
    c.b = texture(tex, st + direction * distortion.b).b;
    return c;
}

// Overload: scalar sdf offset + strength percentage
vec3 chromaAB(in sampler2D tex, in vec2 st, in float sdf, in float pct) {
    vec2 stR = st * (1.0 + vec2(sdf) * 0.02 * pct);
    vec2 stB = st * (1.0 - vec2(sdf) * 0.02 * pct);
    vec3 c;
    c.r = texture(tex, stR).r;
    c.g = texture(tex, st ).g;
    c.b = texture(tex, stB).b;
    return c;
}

// Overload: default pct from macro
vec3 chromaAB(in sampler2D tex, in vec2 st, in float sdf) {
    return chromaAB(tex, st, sdf, CHROMAAB_PCT);
}

// Overload: auto-compute sdf from distance-to-centre
vec3 chromaAB(in sampler2D tex, in vec2 st) {
    return chromaAB(tex, st, lengthSq(st - 0.5), CHROMAAB_PCT);
}

#endif // FNC_CHROMAAB

// ─────────────────────────────────────────────────────────────────────────────
// ── Metaball density field helpers ────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

// Finite-difference normal estimation from the density scalar field.
// Uses a ±1-texel stencil to approximate the gradient direction.
vec2 densityGradient(sampler2D tex, vec2 st, vec2 px) {
    float dL = texture(tex, st - vec2(px.x, 0.0)).r;
    float dR = texture(tex, st + vec2(px.x, 0.0)).r;
    float dD = texture(tex, st - vec2(0.0, px.y)).r;
    float dU = texture(tex, st + vec2(0.0, px.y)).r;
    return vec2(dR - dL, dU - dD) * 0.5;
}

// ─────────────────────────────────────────────────────────────────────────────
// ── SPH velocity speed + direction helpers ────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

// Returns normalised speed [0,1] from the velocity texture RG channels.
float velocitySpeed(sampler2D tex, vec2 st, float maxSpd) {
    vec2  vel   = texture(tex, st).rg;
    float speed = length(vel);
    return saturate(speed / max(maxSpd, 0.001));
}

// ─────────────────────────────────────────────────────────────────────────────
// ── main ──────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

void main() {

    // ── 1. Sample raw SPH density ─────────────────────────────────────────────
    float rawDensity = texture(uDensityTex, vUV).r;   // [0,1] from GPGPU pass

    // ── 2. Ripple collision perturbation ──────────────────────────────────────
    // Advance the ripple wave propagation one step (lygia simulate/ripple.glsl).
    // The resulting height (.x) displaces the density isosurface inward so
    // collision events produce visible surface dimples / splash rings.
    vec3  rippleState  = ripple(uRippleTex, vUV, uTexelSize);
    float rippleHeight = rippleState.x * 2.0 - 1.0;    // remap [0,1] → [-1,1]
    float perturbedDensity = rawDensity + rippleHeight * uRippleStrength;

    // ── 3. Velocity-driven cnoise surface wrinkle ─────────────────────────────
    // Classic Perlin noise modulated by the local SPH speed gives organic
    // micro-turbulence that intensifies at high-velocity flow regions.
    float speed3D    = velocitySpeed(uVelocityTex, vUV, uMaxSpeed);
    vec2  velXY      = texture(uVelocityTex, vUV).rg;
    // Advect the noise domain along velocity to align wrinkles with flow.
    vec2  noiseUV    = vUV * uNoiseScale + velXY * 0.1 + vec2(uTime * 0.07);
    float noiseVal   = cnoise(noiseUV);                         // [-1,1]
    perturbedDensity += noiseVal * uNoiseStrength * speed3D;

    // ── 4. Metaball smooth-step surface reconstruction ────────────────────────
    // Threshold with a configurable band to produce a clean isosurface alpha.
    float lo   = uIsoThreshold - uSmoothBand;
    float hi   = uIsoThreshold + uSmoothBand;
    float surf = smoothstep(lo, hi, perturbedDensity);   // 0=outside, 1=inside

    // Discard fully exterior fragments early (saves bandwidth on large quads).
    if (surf < 0.004) discard;

    // ── 5. Surface normal from density gradient (cheap 2-D approximation) ─────
    vec2  grad        = densityGradient(uDensityTex, vUV, uTexelSize);
    float gradLen     = length(grad);
    vec3  surfNormal  = (gradLen > 1e-5)
        ? normalize(vec3(-grad * 2.0, 1.0))
        : vec3(0.0, 0.0, 1.0);

    // Simple view-space light from top-right (constant for a 2-D surface pass).
    vec3  lightDir    = normalize(vec3(0.6, 0.8, 1.0));
    float diffuse     = max(dot(surfNormal, lightDir), 0.0);

    // Blinn-Phong specular
    vec3  viewDir     = vec3(0.0, 0.0, 1.0);
    vec3  halfVec     = normalize(lightDir + viewDir);
    float specular    = pow(max(dot(surfNormal, halfVec), 0.0), uSpecularPower);

    // ── 6. Chromatic aberration on the density texture (velocity dispersion) ──
    // Fast-moving SPH regions scatter light chromatically — higher speed ⟹
    // stronger RGB channel separation (lygia distort/chromaAB.glsl).
    // We use the density texture as the colour source; the aberration shifts
    // each channel radially outward from the frame centre, proportional to
    // the normalised speed magnitude.
    float aberrationSdf = lengthSq(vUV - 0.5) * speed3D * uChromaStrength;
    vec3  chromaColor   = chromaAB(uDensityTex, vUV, aberrationSdf, 1.0);

    // ── 7. Composite final fluid colour ───────────────────────────────────────
    // Base colour from the tinted chroma sample, then add diffuse + specular.
    vec3  baseColor = uFluidColor * (0.6 + 0.4 * chromaColor.r);
    vec3  litColor  = baseColor  * (0.3 + 0.7 * diffuse)
                    + uSpecularColor * specular * surf;

    // Interior depth tinting: deeper fluid areas (high density) are slightly
    // darker/more saturated to hint at subsurface volume.
    float depthTint = 1.0 - saturate((perturbedDensity - uIsoThreshold) * 1.5);
    litColor       *= mix(1.0, 0.55, depthTint);

    // Surface-edge glow — thin halo at the isosurface boundary.
    float edgeGlow  = 1.0 - smoothstep(0.0, uSmoothBand * 2.5, abs(perturbedDensity - uIsoThreshold));
    litColor       += uSpecularColor * edgeGlow * 0.25;

    // ── 8. Ripple shimmer overlay ─────────────────────────────────────────────
    // The ripple wave height creates a subtle brightness ripple on the surface.
    float rippleShimmer = rippleHeight * 0.12 * surf;
    litColor           += uFluidColor * rippleShimmer;

    // ── 9. Alpha — surf mask with soft interior falloff ───────────────────────
    float alpha = surf * mix(0.82, 1.0, diffuse);

    // ── 10. Pre-multiplied alpha output ───────────────────────────────────────
    fragColor = vec4(litColor * alpha, alpha);
}
