#version 300 es
// ── lut-pipeline.frag ────────────────────────────────────────────────────────
// 3-D LUT colour-grading post-process pass.
//
// Architecture overview
// ─────────────────────
// A full-screen quad samples the previous render target (u_src) through a
// 3-D RGB lookup table (u_lut) to apply film-emulation colour grades.  The
// LUT is a 17×17×17 cube stored as a 17×(17*17) = 17×289 RGBA texture
// (standard "horizontal strip" layout; every row is a fixed B-slice).
//
// QoS-driven tonal zones
// ──────────────────────
// Apollo CyberRT QoS profiles carry semantic information about message
// reliability, durability, and priority.  Rather than applying a single
// static grade to the entire scene, this shader receives four LUT weights
// (one per traffic class) and blends the resulting colour grades spatially
// via a mask texture (u_mask) that encodes which screen region belongs to
// which QoS class.  The weights are derived from QoS metrics at runtime by
// lut-generator.ts (see src/lib/sph/lut-generator.ts).
//
// LUT blend strategy (four traffic-class zones)
// ───────────────────────────────────────────────
//   Zone 0 — RELIABLE / TRANSIENT_LOCAL  (parameters, TF_static, topo-change)
//             Grade: warm grade with lifted shadows → stable, persistent look
//   Zone 1 — RELIABLE / VOLATILE         (default, services)
//             Grade: neutral/balanced film emulation → clean, workman-like
//   Zone 2 — BEST_EFFORT / VOLATILE      (sensor data streams)
//             Grade: cool, desaturated, high-contrast → raw sensor urgency
//   Zone 3 — BEST_EFFORT / TRANSIENT_LOCAL (rare; treated as override)
//             Grade: pushed S-curve, slightly crushed blacks → dramatic
//
// Uniforms
// ────────
//   u_src          sampler2D   — previous render target (sRGB)
//   u_lut          sampler2D   — 17-slice LUT strip (linear RGB, RGBA8)
//   u_mask         sampler2D   — R=zone0 weight, G=zone1, B=zone2, A=zone3
//   u_lutSize      float       — LUT edge length N (default 17.0)
//   u_lutStrength  float       — global LUT blend [0,1]; 0 = passthrough
//   u_exposure     float       — pre-LUT exposure adjust in stops (±2)
//   u_saturation   float       — post-LUT saturation scale [0,2]; 1 = neutral
//   u_contrast     float       — post-LUT S-curve strength [0,2]; 1 = neutral
//   u_time         float       — elapsed time in seconds (animates subtle shimmer)
//   u_qosWeights   vec4        — per-zone LUT strength modulator [0,1]
//   u_vignetteAmt  float       — vignette darkness [0,1]; 0 = none
//
// References
// ──────────
//   • Colour grading with 3-D LUTs: GPU Gems 2, ch.24 (NVidia, 2005)
//   • LUT strip layout: resolve.color.com/resolve-manual (DaVinci Resolve LUT spec)
//   • Lygia GLSL helpers: upstream/lygia/color/tonemap/aces.glsl
//   • QoS semantic colour: src/lib/sph/color-palette.ts (M566)
//   • Apollo QoS profiles: src/lib/sph/qos-spatial-bridge.ts
//
// Research: xiaodi #M624 — cell-pubsub-loop / lut-grading
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;
precision highp sampler2D;

// ── Inputs / outputs ─────────────────────────────────────────────────────────

in  vec2 v_uv;
out vec4 fragColor;

// ── Uniforms ─────────────────────────────────────────────────────────────────

uniform sampler2D u_src;
uniform sampler2D u_lut;
uniform sampler2D u_mask;

uniform float u_lutSize;       // typically 17.0
uniform float u_lutStrength;   // [0,1]
uniform float u_exposure;      // stops
uniform float u_saturation;    // [0,2]
uniform float u_contrast;      // [0,2]
uniform float u_time;
uniform vec4  u_qosWeights;    // per traffic-class LUT intensity
uniform float u_vignetteAmt;   // [0,1]

// ─────────────────────────────────────────────────────────────────────────────
// UTILITY MATH
// ─────────────────────────────────────────────────────────────────────────────

float saturate01(float x) { return clamp(x, 0.0, 1.0); }
vec3  saturate01(vec3  v) { return clamp(v, 0.0, 1.0); }

// sRGB → linear
vec3 srgbToLinear(vec3 c) {
    return mix(
        c / 12.92,
        pow((c + 0.055) / 1.055, vec3(2.4)),
        step(vec3(0.04045), c)
    );
}

// linear → sRGB
vec3 linearToSrgb(vec3 c) {
    c = saturate01(c);
    return mix(
        c * 12.92,
        1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055,
        step(vec3(0.0031308), c)
    );
}

// Luminance (Rec.709)
float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Saturation adjustment (luma-preserving)
vec3 adjustSaturation(vec3 c, float s) {
    float luma = luminance(c);
    return mix(vec3(luma), c, s);
}

// Filmic S-curve contrast (pivoted at mid-grey 0.18)
// strength 1.0 = neutral; >1 increases contrast; <1 flattens
vec3 sContrast(vec3 c, float strength) {
    // Map [0,1] → [-1,1] pivot space, apply tanh-like curve, map back
    vec3 pivot = vec3(0.18);
    vec3 d = c - pivot;
    // Soft S: c' = pivot + d / sqrt(1 + (d * strength)^2)  (soft clamp)
    vec3 s = d * strength;
    return pivot + d / sqrt(1.0 + s * s);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-D LUT SAMPLE
//
// Layout: the LUT texture is a horizontal strip of N slices:
//   width  = N
//   height = N * N
//   texel (r_idx, g_idx + b_idx * N) encodes the output colour for
//   input [r_idx/(N-1), g_idx/(N-1), b_idx/(N-1)].
//
// We tri-linearly interpolate across the cube using four bilinear fetches
// (two B-slices × two G-rows), matching the GPU hardware path that a real
// 3-D texture sampler would produce.
// ─────────────────────────────────────────────────────────────────────────────

vec3 sampleLut(vec3 color, float N) {
    // Scale colour into [0, N-1] index space
    float scale  = (N - 1.0) / N;
    float offset = 0.5 / N;
    vec3  c      = color * scale + offset;    // texel-centre-corrected

    // Integer B indices (floor / ceil)
    float bFloat = c.b * (N - 1.0);
    float b0     = floor(bFloat);
    float b1     = min(b0 + 1.0, N - 1.0);
    float bFrac  = bFloat - b0;

    // Texture V positions for each B-slice (each slice occupies 1/N of height)
    float sliceV0 = (b0 + c.g) / N;
    float sliceV1 = (b1 + c.g) / N;

    // U position (R channel within each row)
    float u = c.r;

    // Bilinear fetch at both B slices
    vec3 col0 = texture(u_lut, vec2(u, sliceV0)).rgb;
    vec3 col1 = texture(u_lut, vec2(u, sliceV1)).rgb;

    return mix(col0, col1, bFrac);
}

// ─────────────────────────────────────────────────────────────────────────────
// QOS ZONE GRADES
//
// Each zone applies a distinct colour transform BEFORE the LUT to steer
// the pixel into a tonal region that reads as "that type of traffic".
// The raw pixel is processed through all four graders, then blended by
// the mask weights.  This is more flexible than four separate LUT textures
// because the user can supply a single shared LUT and the per-zone feel
// comes from pre-processing alone.
// ─────────────────────────────────────────────────────────────────────────────

// Zone 0 — RELIABLE + TRANSIENT_LOCAL
// Warm lift: shadows are pulled toward amber, highlights stay clean.
// Mood: persistent, trusted, parameter-server reliability.
vec3 gradeZone0(vec3 c) {
    // Lift shadows toward warm amber (0.12, 0.07, 0.02)
    vec3 shadowColor = vec3(0.12, 0.07, 0.02);
    float shadowMask = 1.0 - smoothstep(0.0, 0.4, luminance(c));
    c += shadowColor * shadowMask * 0.35;
    // Very slight warm hue push
    c.r *= 1.04;
    c.b *= 0.96;
    return saturate01(c);
}

// Zone 1 — RELIABLE + VOLATILE
// Neutral film: slight warmth, clean contrast, Hollywood baseline.
// Mood: workhorse channel, DEFAULT/SERVICES_DEFAULT profile.
vec3 gradeZone1(vec3 c) {
    // Analogue-emulation toe: lift absolute black slightly
    c = mix(vec3(0.02), c, 0.97);
    // Very gentle warm balance
    c.r *= 1.015;
    c.g *= 1.005;
    c.b *= 0.98;
    return saturate01(c);
}

// Zone 2 — BEST_EFFORT + VOLATILE
// Cool + desaturated: pushes blue channel, reduces saturation.
// Mood: raw sensor stream, radar/lidar urgency.
vec3 gradeZone2(vec3 c) {
    // Desaturate to 70%
    c = adjustSaturation(c, 0.70);
    // Push blue channel slightly
    c.b = min(c.b * 1.12 + 0.04, 1.0);
    c.r = max(c.r * 0.92 - 0.01, 0.0);
    // High-contrast: steeper S-curve
    c = sContrast(c, 1.4);
    return saturate01(c);
}

// Zone 3 — BEST_EFFORT + TRANSIENT_LOCAL (override/alert traffic)
// Dramatic: pushed S-curve, crushed blacks, slight magenta tint.
// Mood: topology change, high-priority override events.
vec3 gradeZone3(vec3 c) {
    // Crush blacks
    c = max(c - 0.04, 0.0) / 0.96;
    // Strong S-curve contrast
    c = sContrast(c, 1.8);
    // Subtle magenta push (pink→alert associations)
    c.r *= 1.06;
    c.b *= 1.04;
    c.g *= 0.96;
    return saturate01(c);
}

// ─────────────────────────────────────────────────────────────────────────────
// VIGNETTE
// Soft circular darken toward screen edges, parameterised by u_vignetteAmt.
// ─────────────────────────────────────────────────────────────────────────────

float vignette(vec2 uv, float amount) {
    vec2  d = uv - 0.5;
    float r = length(d) * 1.414;   // 0 at centre, 1 at corner
    return 1.0 - amount * smoothstep(0.4, 1.0, r);
}

// ─────────────────────────────────────────────────────────────────────────────
// SUBTLE GRAIN (adds life, breaks up LUT banding artefacts)
// IGN (Interleaved Gradient Noise) from Jimenez, 2014.
// ─────────────────────────────────────────────────────────────────────────────

float ign(vec2 fragCoord) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(fragCoord, magic.xy)));
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

void main() {
    // 1. Fetch source pixel (assumed sRGB from canvas/FBO)
    vec3 srcSrgb   = texture(u_src,  v_uv).rgb;
    vec4 maskWeights = texture(u_mask, v_uv);  // RGBA = zone0..3 weights

    // Combine QoS runtime weights with zone mask
    vec4 w = maskWeights * u_qosWeights;
    // Normalise so weights sum to ≤1; remaining weight = passthrough
    float wSum = w.x + w.y + w.z + w.w;
    if (wSum > 1.0) w /= wSum;
    float wPassthrough = max(0.0, 1.0 - wSum);

    // 2. Convert to linear for colour math
    vec3 linear = srgbToLinear(srcSrgb);

    // 3. Pre-LUT exposure (stops → linear multiplier)
    linear *= pow(2.0, u_exposure);
    linear  = saturate01(linear);

    // 4. Per-zone grade in linear space
    vec3 g0 = gradeZone0(linear);
    vec3 g1 = gradeZone1(linear);
    vec3 g2 = gradeZone2(linear);
    vec3 g3 = gradeZone3(linear);

    // Blend graded versions; passthrough keeps original linear
    vec3 graded = linear * wPassthrough
                + g0 * w.x
                + g1 * w.y
                + g2 * w.z
                + g3 * w.w;

    // 5. 3-D LUT lookup (LUT lives in sRGB-ish encoded space per DaVinci spec)
    //    Convert to sRGB for LUT, sample, convert back for further math
    vec3 preLut   = linearToSrgb(graded);
    vec3 lutColor = sampleLut(preLut, u_lutSize);
    vec3 postLut  = srgbToLinear(lutColor);

    // Blend LUT output with bypassed graded by u_lutStrength
    vec3 result = mix(graded, postLut, u_lutStrength);

    // 6. Post-LUT saturation + contrast
    result = adjustSaturation(result, u_saturation);
    result = sContrast(result, u_contrast);
    result = saturate01(result);

    // 7. Vignette
    result *= vignette(v_uv, u_vignetteAmt);

    // 8. Grain dither (1/255 amplitude — invisible statically, kills LUT quant banding)
    float grain = (ign(gl_FragCoord.xy + u_time * 37.3) - 0.5) / 255.0;
    result += vec3(grain);

    // 9. Output: back to sRGB for the swapchain
    fragColor = vec4(linearToSrgb(saturate01(result)), 1.0);
}
