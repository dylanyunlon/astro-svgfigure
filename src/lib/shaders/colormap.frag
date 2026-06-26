#version 300 es
/**
 * colormap.frag
 * 
 * Perceptually uniform scientific colormaps for WebGL.
 * Implements Viridis, Plasma, Inferno, Magma from matplotlib (van der Walt & Smith, SciPy 2015).
 * 
 * Approach: degree-6 polynomial approximation fitted to the official LUT data.
 * Zero texture fetches — pure math, works on WebGL 1.0+
 * 
 * Usage:
 *   #include "colormap.frag"
 *   vec3 col = colormap_viridis(value);   // value in [0.0, 1.0]
 * 
 * References:
 *   - Polynomial coefficients: IQ / shadertoy community (shadertoy.com/view/XtGGzG)
 *   - Official matplotlib data: github.com/matplotlib/matplotlib/blob/main/_cm_listed.py
 *   - glsl-colormap: github.com/glslify/glsl-colormap
 *   - Observable WebGL colormaps: observablehq.com/@flimsyhat/webgl-color-maps
 *
 * Research: xiaodi #83 — cell-pubsub-loop
 */

precision mediump float;

// ---------------------------------------------------------------------------
// UTILITY
// ---------------------------------------------------------------------------

/**
 * Clamp t to [0,1] before colormap lookup.
 * Call this if input data may exceed the [0,1] domain.
 */
float cm_clamp01(float t) {
    return clamp(t, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// VIRIDIS
// Monotonically increasing lightness: dark purple → blue → teal → green → yellow
// Best for: general scientific data, colorblind-safe, print-safe
// Lightness range: ~15% → ~90% (CIELAB L*)
// ---------------------------------------------------------------------------
vec3 colormap_viridis(float t) {
    t = cm_clamp01(t);
    const vec3 c0 = vec3(0.2777273272234177,  0.005407344544966578, 0.3340998053353061);
    const vec3 c1 = vec3(0.1050930431085774,  1.404613529898575,    1.384590162594685);
    const vec3 c2 = vec3(-0.3308618287255563,  0.214847559468213,    0.09509516302823659);
    const vec3 c3 = vec3(-4.634230498983486,  -5.799100973351585,  -19.33244095627987);
    const vec3 c4 = vec3(6.228269936347081,   14.17993336680509,   56.69055260068105);
    const vec3 c5 = vec3(4.776384997670288,  -13.74514537774601,  -65.35303263337234);
    const vec3 c6 = vec3(-5.435455855934631,   4.645852612178535,   26.3124352495832);
    return clamp(c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6))))), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// PLASMA
// High contrast: dark purple → magenta → orange → yellow
// Best for: high-contrast visualization, aesthetically striking
// Lightness range: ~12% → ~92% (CIELAB L*)
// ---------------------------------------------------------------------------
vec3 colormap_plasma(float t) {
    t = cm_clamp01(t);
    const vec3 c0 = vec3(0.05873234392399702,  0.02333670892565664,  0.5433401826748754);
    const vec3 c1 = vec3(2.176514634195958,    0.2383834171260182,   0.7539604599784036);
    const vec3 c2 = vec3(-2.689460476458034,  -7.455851135738909,   3.110799939717086);
    const vec3 c3 = vec3(6.130348345893603,   42.35286317604309,   -28.51885465332158);
    const vec3 c4 = vec3(-11.10743619062271, -82.66631109428045,    60.13984767418263);
    const vec3 c5 = vec3(10.02306557647065,   71.41361770095349,   -54.07218655560067);
    const vec3 c6 = vec3(-3.658713842777788, -22.93153465461149,   18.19190778539828);
    return clamp(c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6))))), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// INFERNO
// Black → dark red → orange → light yellow-white
// Best for: heatmaps, density fields, fire/thermal effects
// Lightness range: ~0% → ~95% (excellent for dark backgrounds)
// ---------------------------------------------------------------------------
vec3 colormap_inferno(float t) {
    t = cm_clamp01(t);
    const vec3 c0 = vec3(0.0002189403691192265,  0.001651004631001012,  -0.01948089843709584);
    const vec3 c1 = vec3(0.1065134194856116,      0.5639564367884091,     3.932712388889277);
    const vec3 c2 = vec3(11.60249308247187,      -3.972853965665698,    -15.9423941062914);
    const vec3 c3 = vec3(-41.70399613139459,      17.43639888205313,     44.35414519872813);
    const vec3 c4 = vec3(77.162935699427,        -33.40235894210092,    -81.80730925738993);
    const vec3 c5 = vec3(-71.31942824499214,      32.62606426397723,     73.20951985803202);
    const vec3 c6 = vec3(25.13112622477341,      -12.24266895238567,    -23.07032500287172);
    return clamp(c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6))))), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// MAGMA
// Black → purple/violet → rose-pink → light cream-white
// Best for: similar to Inferno but cooler hues, geological/astronomical data
// Lightness range: ~0% → ~95%
// ---------------------------------------------------------------------------
vec3 colormap_magma(float t) {
    t = cm_clamp01(t);
    const vec3 c0 = vec3(-0.002136485053939582, -0.000749655052795221, -0.005386127855323933);
    const vec3 c1 = vec3(0.2516605407371642,     0.6775232436837668,    2.494026599312351);
    const vec3 c2 = vec3(8.353717279216625,     -3.577719514958484,    0.3144679030132573);
    const vec3 c3 = vec3(-27.66873308576866,     14.26473078096533,    -13.64921318813922);
    const vec3 c4 = vec3(52.17613981234068,     -27.94360607168351,    12.94416215269321);
    const vec3 c5 = vec3(-50.76852536473588,     29.04658282127291,    4.23415299384598);
    const vec3 c6 = vec3(18.65570506591883,     -11.48977351997711,   -5.601961508734096);
    return clamp(c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6))))), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// TURBO (Google, 2019) — Rainbow replacement
// Dark blue → green → yellow → orange → dark red
// Better perceptual uniformity than Jet/Rainbow while covering full spectrum
// Best for: replacing legacy jet/rainbow, requires full-spectrum discrimination
// ---------------------------------------------------------------------------
vec3 colormap_turbo(float t) {
    t = cm_clamp01(t);
    // Polynomial from: https://observablehq.com/@flimsyhat/webgl-color-maps
    const vec4 kRedVec4   = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839,  4.84296658, -14.18503333);
    const vec4 kBlueVec4  = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRedVec2   = vec2(-152.94239396, 59.28637943);
    const vec2 kGreenVec2 = vec2(4.27729857,  2.82956604);
    const vec2 kBlueVec2  = vec2(-89.90310912, 27.34824973);

    vec4 v4 = vec4(1.0, t, t*t, t*t*t);
    vec2 v2 = v4.zw * v4.z;
    return vec3(
        dot(v4, kRedVec4)   + dot(v2, kRedVec2),
        dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
        dot(v4, kBlueVec4)  + dot(v2, kBlueVec2)
    );
}

// ---------------------------------------------------------------------------
// DIVERGING: COOLWARM (simplified)
// Blue (cold) → white (neutral) → red (hot)
// Best for: data with meaningful zero/center point (e.g. attention weights)
// ---------------------------------------------------------------------------
vec3 colormap_coolwarm(float t) {
    t = cm_clamp01(t);
    // Blue end: (0.085, 0.532, 0.201) → White: (0.865, 0.865, 0.865) → Red: (0.706, 0.016, 0.150)
    // Simplified linear blend through white at t=0.5
    vec3 cold = vec3(0.085, 0.532, 0.201);
    vec3 mid  = vec3(0.865, 0.865, 0.865);
    vec3 warm = vec3(0.706, 0.016, 0.150);
    return t < 0.5
        ? mix(cold, mid, t * 2.0)
        : mix(mid, warm, (t - 0.5) * 2.0);
}

// ---------------------------------------------------------------------------
// TRANSFER FUNCTION HELPERS
// Value → color with data range remapping
// ---------------------------------------------------------------------------

/**
 * Map a data value from [dataMin, dataMax] to a colormap.
 * @param value  raw data value
 * @param lo     minimum data range
 * @param hi     maximum data range
 */
vec3 colormap_viridis_range(float value, float lo, float hi) {
    return colormap_viridis((value - lo) / max(hi - lo, 1e-6));
}

vec3 colormap_plasma_range(float value, float lo, float hi) {
    return colormap_plasma((value - lo) / max(hi - lo, 1e-6));
}

vec3 colormap_inferno_range(float value, float lo, float hi) {
    return colormap_inferno((value - lo) / max(hi - lo, 1e-6));
}

/**
 * Log-scale colormap: useful for data with high dynamic range (e.g. particle density).
 * Applies log10 normalization then viridis colormap.
 * @param value  positive data value
 * @param lo     minimum (positive!) data value
 * @param hi     maximum data value
 */
vec3 colormap_viridis_log(float value, float lo, float hi) {
    float t = (log(max(value, lo)) - log(lo)) / max(log(hi) - log(lo), 1e-6);
    return colormap_viridis(t);
}

// ---------------------------------------------------------------------------
// COLORMAP SELECTOR (for dynamic selection via uniform)
// Usage: uniform int u_colormap_id;  // 0=viridis, 1=plasma, 2=inferno, 3=magma, 4=turbo
// ---------------------------------------------------------------------------
vec3 colormap_select(float t, int id) {
    if (id == 0) return colormap_viridis(t);
    if (id == 1) return colormap_plasma(t);
    if (id == 2) return colormap_inferno(t);
    if (id == 3) return colormap_magma(t);
    if (id == 4) return colormap_turbo(t);
    if (id == 5) return colormap_coolwarm(t);
    return colormap_viridis(t); // fallback
}
