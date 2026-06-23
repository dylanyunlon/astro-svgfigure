// splineshader.glsl — AT Spline Shader (edge 数据流样条线渲染核心)
// Ported from Active Theory Hydra engine / FindingLove spline path render.
// Augments edge-spline.frag with AT-quality Catmull-Rom pressure + taper.
// ─────────────────────────────────────────────────────────────────────────────
// Uniforms (additional to edge-spline.frag):
//   uPressure   — flow pressure scalar [0,1], thickens stroke at high load
//   uTaper      — taper exponent: 0=uniform, 1=point-to-tail taper
//   uSpecies    — species index [0–7], drives hue rotation on curvature nodes
// ─────────────────────────────────────────────────────────────────────────────
#ifndef SPLINESHADER_GLSL
#define SPLINESHADER_GLSL

precision highp float;

uniform float uPressure;   // [0,1]  flow load → stroke width modulation
uniform float uTaper;      // [0,1]  taper exponent along arc length
uniform float uSpecies;    // [0,7]  species hue shift index

// ── Catmull-Rom knot weight at parameter t ────────────────────────────────
vec4 catmullRomWeights(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return 0.5 * vec4(
        -t3 + 2.0*t2 - t,
         3.0*t3 - 5.0*t2 + 2.0,
        -3.0*t3 + 4.0*t2 + t,
         t3 - t2
    );
}

// ── Pressure-driven stroke width modulation ───────────────────────────────
float pressureWidth(float baseHalfW, float pressure, float t, float taper) {
    float taperScale = mix(1.0, pow(1.0 - t, taper + 0.5) * 2.0, taper);
    float pressureMod = 1.0 + pressure * 0.45;
    return baseHalfW * taperScale * pressureMod;
}

// ── Species hue rotation (0–7 → 0°–315° in HSV space) ────────────────────
vec3 speciesHueShift(vec3 rgb, float speciesIdx) {
    float hueAngle = speciesIdx * 0.125 * 6.28318;  // 45° steps
    float s = sin(hueAngle), c = cos(hueAngle);
    // Rodrigues rotation in cone-mapped RGB
    vec3 k   = vec3(0.57735);
    return rgb * c + cross(k, rgb) * s + k * dot(k, rgb) * (1.0 - c);
}

#endif // SPLINESHADER_GLSL
