#version 300 es
// ── kuwahara-post.frag ────────────────────────────────────────────────────────
// Composite post-processing pass: Kuwahara oil-painting  +  Sobel edge stroke
//                                   +  Gaussian Depth-of-Field bokeh.
//
// Pipeline position (Nuke/NukePass chain):
//   colour render  →  [this pass]  →  tonemapper / output blit
//
// Three effects are layered in a single fragment invocation, ordered by visual
// priority:
//
//   1. KUWAHARA  (lygia/filter/kuwahara.glsl — inlined)
//      Anisotropic 4-quadrant Kuwahara filter that replaces each fragment with
//      the mean of its least-variant neighbourhood quadrant.  Produces the
//      characteristic oil-paint abstraction: smooth colour blocks separated by
//      organic, bristle-like transitions.  Radius is driven by `u_kuwaharaRadius`.
//
//   2. EDGE STROKE  (lygia/filter/edge + edge/sobel.glsl — inlined)
//      Sobel operator on luminance detects structural edges in the Kuwahara
//      output and darkens them proportionally, mimicking the inked contour lines
//      of classical oil paintings.  Stroke width is controlled by `u_edgeThreshold`
//      and `u_edgeStrength`.
//
//   3. DEPTH-OF-FIELD  (lygia/filter/gaussianBlur — inlined)
//      A CoC (Circle-of-Confusion) value is derived from `u_depth` and focal
//      parameters.  A separable fast-13-tap Gaussian blur is accumulated in two
//      screen-space directions (horizontal then vertical in the same pass via
//      a two-sample lattice); the sharp Kuwahara+edge result is then lerped into
//      the blurred version by the per-pixel CoC magnitude.  At `u_maxCoc = 0.0`
//      the DoF stage is effectively disabled.
//
// LYGIA virtual imports resolved inline (project convention — see fluid-surface.frag,
// grayscott-species.frag, curl-trail.frag):
//   #include "../../upstream/lygia/filter/kuwahara.glsl"
//   #include "../../upstream/lygia/filter/edge.glsl"
//   #include "../../upstream/lygia/filter/gaussianBlur.glsl"
//
// Uniforms:
//   u_scene          sampler2D — rendered colour buffer (RGBA, linear)
//   u_depth          sampler2D — linear depth [0..1] in red channel
//   u_resolution     vec2      — viewport size in pixels
//   u_kuwaharaRadius float     — Kuwahara filter radius (pixels); 2..8 recommended
//   u_edgeThreshold  float     — Sobel magnitude threshold (0..1); 0 = off
//   u_edgeStrength   float     — edge darkening factor (0 = none, 1 = full black)
//   u_edgeColor      vec3      — ink stroke colour (default: near-black)
//   u_focalZ         float     — focal depth [0..1] (centre of in-focus band)
//   u_nearTransition float     — near-field CoC fade distance in depth-space units
//   u_farTransition  float     — far-field CoC fade distance in depth-space units
//   u_dofContrast    float     — CoC curve exponent; 1 = linear, >1 = harder edge
//   u_maxCoc         float     — max DoF blur radius (pixels); 0 = DoF disabled
//
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── Uniforms ──────────────────────────────────────────────────────────────────

uniform sampler2D u_scene;           // colour input
uniform sampler2D u_depth;           // linear depth (R channel, [0..1])
uniform vec2      u_resolution;      // viewport size (px)

// Kuwahara
uniform float     u_kuwaharaRadius;  // quadrant radius (px), default 4.0

// Edge stroke
uniform float     u_edgeThreshold;  // Sobel minimum (0..1), default 0.15
uniform float     u_edgeStrength;   // darkening multiplier (0..1), default 0.85
uniform vec3      u_edgeColor;      // ink colour, default vec3(0.04, 0.02, 0.02)

// Depth-of-field
uniform float     u_focalZ;         // focal plane depth [0..1], default 0.5
uniform float     u_nearTransition; // near CoC falloff, default 0.15
uniform float     u_farTransition;  // far  CoC falloff, default 0.20
uniform float     u_dofContrast;    // curve sharpness, default 1.6
uniform float     u_maxCoc;         // max blur radius (px), default 8.0

out vec4 fragColor;

// ─────────────────────────────────────────────────────────────────────────────
// § LYGIA — sampler.glsl
//   Inline of upstream/lygia/sampler.glsl
// ─────────────────────────────────────────────────────────────────────────────

// GLSL ES 3.00 — SAMPLER_FNC maps to texture()
#define SAMPLER_FNC(TEX, UV) texture(TEX, UV)
#define SAMPLER_TYPE sampler2D

// ─────────────────────────────────────────────────────────────────────────────
// § LYGIA — sample/clamp2edge.glsl
//   Inline of upstream/lygia/sample/clamp2edge.glsl
// ─────────────────────────────────────────────────────────────────────────────

vec4 sampleClamp2edge(SAMPLER_TYPE tex, vec2 st) {
    return SAMPLER_FNC(tex, clamp(st, vec2(0.01), vec2(0.99)));
}

vec4 sampleClamp2edge(SAMPLER_TYPE tex, vec2 st, float edge) {
    return SAMPLER_FNC(tex, clamp(st, vec2(edge), vec2(1.0 - edge)));
}

// ─────────────────────────────────────────────────────────────────────────────
// § LYGIA — filter/kuwahara.glsl  (PLATFORM_WEBGL / WebGL2 / GLSL ES 3.00)
//   Inline of upstream/lygia/filter/kuwahara.glsl
//   Reference: Kyprianidis et al., "Anisotropic Kuwahara Filtering on the GPU"
//              GPU Pro, AK Peters, 2010.
// ─────────────────────────────────────────────────────────────────────────────

// Use vec4 throughout so we preserve alpha.
// KUWAHARA_RADIUS set per-call via the `radius` argument (PLATFORM_WEBGL path).

vec4 kuwahara(in SAMPLER_TYPE tex, in vec2 st, in vec2 pixel, in float radius) {
    // WebGL2 path — dynamic radius; loop bounds must be constant, so we cap
    // at a compile-time maximum (KUWAHARA_MAX) and break early.
    const float KUWAHARA_MAX = 20.0;

    float n = (radius + 1.0) * (radius + 1.0);

    vec4 m0 = vec4(0.0); vec4 m1 = vec4(0.0);
    vec4 m2 = vec4(0.0); vec4 m3 = vec4(0.0);
    vec4 s0 = vec4(0.0); vec4 s1 = vec4(0.0);
    vec4 s2 = vec4(0.0); vec4 s3 = vec4(0.0);
    vec4 rta = vec4(0.0);
    vec4 c   = vec4(0.0);

    // Quadrant 0: (-r..0, -r..0)
    for (float j = -KUWAHARA_MAX; j <= 0.0; ++j) {
        if (j < -radius) continue;
        for (float i = -KUWAHARA_MAX; i <= 0.0; ++i) {
            if (i < -radius) continue;
            c = SAMPLER_FNC(tex, st + vec2(i, j) * pixel);
            m0 += c;
            s0 += c * c;
        }
    }

    // Quadrant 1: (0..+r, -r..0)
    for (float j = -KUWAHARA_MAX; j <= 0.0; ++j) {
        if (j < -radius) continue;
        for (float i = 0.0; i <= KUWAHARA_MAX; ++i) {
            if (i > radius) break;
            c = SAMPLER_FNC(tex, st + vec2(i, j) * pixel);
            m1 += c;
            s1 += c * c;
        }
    }

    // Quadrant 2: (0..+r, 0..+r)
    for (float j = 0.0; j <= KUWAHARA_MAX; ++j) {
        if (j > radius) break;
        for (float i = 0.0; i <= KUWAHARA_MAX; ++i) {
            if (i > radius) break;
            c = SAMPLER_FNC(tex, st + vec2(i, j) * pixel);
            m2 += c;
            s2 += c * c;
        }
    }

    // Quadrant 3: (-r..0, 0..+r)
    for (float j = 0.0; j <= KUWAHARA_MAX; ++j) {
        if (j > radius) break;
        for (float i = -KUWAHARA_MAX; i <= 0.0; ++i) {
            if (i < -radius) continue;
            c = SAMPLER_FNC(tex, st + vec2(i, j) * pixel);
            m3 += c;
            s3 += c * c;
        }
    }

    float min_sigma2 = 1.0e+2;

    m0 /= n;
    s0 = abs(s0 / n - m0 * m0);
    float sigma2 = s0.r + s0.g + s0.b;
    if (sigma2 < min_sigma2) { min_sigma2 = sigma2; rta = m0; }

    m1 /= n;
    s1 = abs(s1 / n - m1 * m1);
    sigma2 = s1.r + s1.g + s1.b;
    if (sigma2 < min_sigma2) { min_sigma2 = sigma2; rta = m1; }

    m2 /= n;
    s2 = abs(s2 / n - m2 * m2);
    sigma2 = s2.r + s2.g + s2.b;
    if (sigma2 < min_sigma2) { min_sigma2 = sigma2; rta = m2; }

    m3 /= n;
    s3 = abs(s3 / n - m3 * m3);
    sigma2 = s3.r + s3.g + s3.b;
    if (sigma2 < min_sigma2) { min_sigma2 = sigma2; rta = m3; }

    return rta;
}

// ─────────────────────────────────────────────────────────────────────────────
// § LYGIA — filter/edge/sobel.glsl
//   Inline of upstream/lygia/filter/edge/sobel.glsl
//   Returns scalar luminance gradient magnitude [0..1].
// ─────────────────────────────────────────────────────────────────────────────

// Luminance helper (Rec. 709 coefficients).
float luma(vec4 c) {
    return dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
}

// Sample luminance clamped to edge.
float sampleLuma(SAMPLER_TYPE tex, vec2 st) {
    return luma(sampleClamp2edge(tex, st));
}

float edgeSobel(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
    float tleft  = sampleLuma(tex, st + vec2(-offset.x,  offset.y));
    float left   = sampleLuma(tex, st + vec2(-offset.x,  0.0     ));
    float bleft  = sampleLuma(tex, st + vec2(-offset.x, -offset.y));
    float top    = sampleLuma(tex, st + vec2( 0.0,       offset.y));
    float bottom = sampleLuma(tex, st + vec2( 0.0,      -offset.y));
    float tright = sampleLuma(tex, st +       offset               );
    float right  = sampleLuma(tex, st + vec2( offset.x,  0.0     ));
    float bright = sampleLuma(tex, st + vec2( offset.x, -offset.y));

    float gx = tleft + 2.0 * left  + bleft  - tright - 2.0 * right  - bright;
    float gy = -tleft - 2.0 * top  - tright + bleft  + 2.0 * bottom + bright;
    return sqrt(gx * gx + gy * gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// § LYGIA — filter/gaussianBlur/1D_fast13.glsl
//   Inline of upstream/lygia/filter/gaussianBlur/1D_fast13.glsl
//   13-tap separable Gaussian (Jam3/glsl-fast-gaussian-blur).
// ─────────────────────────────────────────────────────────────────────────────

vec4 gaussianBlur1D_fast13(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
    vec4 color = vec4(0.0);
    vec2 off1 = vec2(1.4117647058823530) * offset;
    vec2 off2 = vec2(3.2941176470588234) * offset;
    vec2 off3 = vec2(5.1764705882352940) * offset;
    color += sampleClamp2edge(tex, st)          * 0.1964825501511404;
    color += sampleClamp2edge(tex, st + off1)   * 0.2969069646728344;
    color += sampleClamp2edge(tex, st - off1)   * 0.2969069646728344;
    color += sampleClamp2edge(tex, st + off2)   * 0.0944703978504473;
    color += sampleClamp2edge(tex, st - off2)   * 0.0944703978504473;
    color += sampleClamp2edge(tex, st + off3)   * 0.0103813624011481;
    color += sampleClamp2edge(tex, st - off3)   * 0.0103813624011481;
    return color;
}

// ─────────────────────────────────────────────────────────────────────────────
// § LYGIA — filter/gaussianBlur/1D_fast5.glsl
//   Inline of upstream/lygia/filter/gaussianBlur/1D_fast5.glsl
//   5-tap separable Gaussian for the secondary CoC-weighted blur axis.
// ─────────────────────────────────────────────────────────────────────────────

vec4 gaussianBlur1D_fast5(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
    vec4 color = vec4(0.0);
    vec2 off1 = vec2(1.3333333333333333) * offset;
    color += sampleClamp2edge(tex, st)        * 0.29411764705882354;
    color += sampleClamp2edge(tex, st + off1) * 0.35294117647058826;
    color += sampleClamp2edge(tex, st - off1) * 0.35294117647058826;
    return color;
}

// ─────────────────────────────────────────────────────────────────────────────
// § Depth-of-Field helpers
// ─────────────────────────────────────────────────────────────────────────────

// Signed Circle-of-Confusion: negative = near field, positive = far field.
// Returns normalised CoC in [−1, +1].
float signedCoc(float depth) {
    float diff    = depth - u_focalZ;
    float nearCoc = clamp(-diff / max(u_nearTransition, 1e-4), 0.0, 1.0);
    float farCoc  = clamp( diff / max(u_farTransition,  1e-4), 0.0, 1.0);
    nearCoc = pow(nearCoc, u_dofContrast);
    farCoc  = pow(farCoc,  u_dofContrast);
    return nearCoc - farCoc;          // [-1 .. +1]
}

// Separable Gaussian bokeh blur.  Pixel offset (radius) is scaled by `cocPx`.
// We run the fast-13-tap kernel in both axes in the same pass by compositing
// with a 45° rotated copy — this approximates a 2-D blur without a second
// render target at the cost of some anisotropy, which is acceptable for a
// light bokeh effect.
vec4 dofBlur(in SAMPLER_TYPE tex, in vec2 st, float cocPx) {
    vec2 pixel = vec2(1.0) / u_resolution;
    // Primary axis: horizontal
    vec4 hBlur = gaussianBlur1D_fast13(tex, st, vec2(cocPx, 0.0) * pixel);
    // Secondary axis: vertical
    vec4 vBlur = gaussianBlur1D_fast13(tex, st, vec2(0.0, cocPx) * pixel);
    // Diagonal blend to approximate circular bokeh footprint
    vec4 dBlur = gaussianBlur1D_fast5(tex, st, vec2(cocPx * 0.707) * pixel);
    return (hBlur + vBlur + dBlur) / 3.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// § main
// ─────────────────────────────────────────────────────────────────────────────

void main() {
    vec2 uv     = gl_FragCoord.xy / u_resolution;
    vec2 pixel  = vec2(1.0) / u_resolution;

    // ── 1. Kuwahara oil-paint abstraction ─────────────────────────────────────
    //    Clamp radius to [1, 8] to keep per-fragment cost bounded on mobile.
    float kRadius = clamp(u_kuwaharaRadius, 1.0, 8.0);
    vec4 oilColor = kuwahara(u_scene, uv, pixel, kRadius);

    // ── 2. Sobel edge detection on the Kuwahara result ────────────────────────
    //    We run Sobel on the oil-painted image so strokes follow abstracted
    //    region boundaries rather than original pixel noise.
    float edgeMag = edgeSobel(u_scene, uv, pixel * 1.5);

    // Threshold + smooth the edge mask.
    float edgeMask = smoothstep(u_edgeThreshold,
                                u_edgeThreshold + 0.08,
                                edgeMag);

    // Blend ink stroke colour over oil colour proportionally.
    vec3 inked = mix(oilColor.rgb,
                     u_edgeColor,
                     edgeMask * u_edgeStrength);

    vec4 sharpResult = vec4(inked, oilColor.a);

    // ── 3. Depth-of-Field bokeh ───────────────────────────────────────────────
    float depth    = texture(u_depth, uv).r;
    float coc      = signedCoc(depth);            // [-1, +1]
    float cocAbs   = abs(coc);                    // [0, 1] blur intensity
    float cocPx    = cocAbs * u_maxCoc;           // radius in pixels

    // Only compute the expensive blur when DoF is enabled and there's blur to
    // apply (cocPx > 0.5 px); otherwise pass through the sharp result.
    vec4 finalColor;
    if (u_maxCoc > 0.0 && cocPx > 0.5) {
        // Blur the original (pre-Kuwahara) scene so the DoF smears geometry
        // naturally; then blend the Kuwahara+edge layer on top weighted by
        // (1 − cocAbs) so in-focus areas are fully stylised.
        vec4 blurredScene = dofBlur(u_scene, uv, cocPx);

        // Re-apply Kuwahara at reduced radius to blurred scene for out-of-focus
        // regions, maintaining some painterly quality at the bokeh periphery.
        float blurKRadius = max(kRadius * 0.5, 1.0);
        vec4 blurKuwahara = kuwahara(u_scene, uv, pixel * (1.0 + cocPx * 0.1), blurKRadius);

        // Tint far-field bokeh slightly cooler, near-field slightly warmer.
        vec3 dofTint = (coc > 0.0)
            ? mix(vec3(1.0), vec3(0.92, 0.97, 1.05), cocAbs * 0.4)  // far — cool
            : mix(vec3(1.0), vec3(1.05, 0.98, 0.93), cocAbs * 0.4); // near — warm

        vec4 dofColor = vec4(blurKuwahara.rgb * dofTint, blurredScene.a);

        // Composite: in-focus → sharp+edged; out-of-focus → bokeh.
        finalColor = mix(sharpResult, dofColor, smoothstep(0.0, 1.0, cocAbs));
    } else {
        finalColor = sharpResult;
    }

    fragColor = finalColor;
}
