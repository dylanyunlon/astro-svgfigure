#version 300 es
// ── edge-spline.frag ──────────────────────────────────────────────────────────
// Cubic Bézier skip-connection shader — fragment stage.
//
// Each fragment receives the pixel-space position of this corner (vFragCoordPx)
// and the nearest point on the curve (vCurvePx) forwarded from the vertex stage
// for approximate distance computation.  Because each quad covers only 1/N of
// the curve, the nearest-on-segment approximation is accurate enough for any
// subdivision count ≥ 16.
//
// The curvature value from topology.json already shaped the Bézier control
// points on the CPU side (see EdgeRenderer.ts) so this shader doesn't need it
// directly — it just draws the line as defined by the geometry.
//
// Uniforms:
//   uColor        — RGB stroke colour (vec3)
//   uAlpha        — master opacity (float)
//   uLineWidth    — full stroke width in pixels (float)
//   uDashLength   — dash on-length in pixels; 0 = solid (float)
//   uGapLength    — dash gap-length in pixels (float)
//   uGlowColor    — skip-connection glow colour (vec3)
//   uGlowRadius   — glow radius in pixels; 0 = off (float)
//   uGlowAlpha    — peak glow opacity (float)
//   uTime         — seconds, drives animated dash travel (float)
//   uArcLength    — total arc length estimate in pixels (float)
//   uCurvature    — curvature parameter from topology.json (float, 0-1)
//   u_sourceColor — RGB colour at source node (vec3)
//   u_targetColor — RGB colour at target node (vec3)
//   u_flowSpeed   — flow-pulse scroll speed (float)
//   u_thickness   — additional thickness control (float)
//   u_time        — global animation time in seconds (float)
//
// Inputs from vertex:
//   vFragCoordPx  — pixel-space position of this fragment
//   vT            — normalised Bézier t at this vertex
//   vTangentDir   — unit tangent at vT
//   vCurvePx      — pixel-space curve position at quad midpoint
//   vHalfWidth    — stroke half-width in pixels
//   v_t           — normalised parametric t for species colour lerp
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── varyings from vertex ──────────────────────────────────────────────────────
in float vT;
in vec2  vTangentDir;
in vec2  vCurvePx;
in float vHalfWidth;
in vec2  vFragCoordPx;
in float v_t;           // normalised parametric t for species colour lerp

// ── uniforms ──────────────────────────────────────────────────────────────────
uniform vec3  uColor;
uniform float uAlpha;
uniform float uLineWidth;
uniform float uDashLength;
uniform float uGapLength;
uniform vec3  uGlowColor;
uniform float uGlowRadius;
uniform float uGlowAlpha;
uniform float uTime;
uniform float uArcLength;
uniform float uCurvature;

// ── species colour & flow uniforms ───────────────────────────────────────────
uniform vec3  u_sourceColor;  // RGB at source node
uniform vec3  u_targetColor;  // RGB at target node
uniform float u_flowSpeed;    // flow-pulse scroll speed
uniform float u_thickness;    // additional thickness control
uniform float u_time;         // global animation time (seconds)

// ── output ────────────────────────────────────────────────────────────────────
out vec4 finalColor;

// ─────────────────────────────────────────────────────────────────────────────
// Approximate SDF to the curve using the quad midpoint as the nearest point.
// This is exact at the centre of each quad; error is bounded by the sub-arc
// chord-to-curve deviation, which falls off as O(1/N²) for N subdivisions.
// ─────────────────────────────────────────────────────────────────────────────
float approxDistToCurve() {
    vec2 diff = vFragCoordPx - vCurvePx;
    // Project onto tangent and perpendicular
    vec2 perp = vec2(-vTangentDir.y, vTangentDir.x);
    return abs(dot(diff, perp));   // perpendicular distance = SDF approximation
}

// ─────────────────────────────────────────────────────────────────────────────
// Dash pattern along arc length
// ─────────────────────────────────────────────────────────────────────────────
float dashMask(float t) {
    if (uDashLength < 0.5) return 1.0;

    float t_px   = t * uArcLength;
    float period = uDashLength + uGapLength;
    // Animated: dashes travel along the curve direction
    float phase  = mod(t_px - uTime * 50.0, period);

    float onEnd  = uDashLength;
    float on     = smoothstep(0.0, 1.0, phase)
                 * (1.0 - smoothstep(onEnd - 1.0, onEnd, phase));
    return clamp(on, 0.0, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Curvature tint: skip connections get a subtle hue shift based on curvature.
// curvature = 0 → uColor as-is; curvature = 1 → slight shift toward uGlowColor
// ─────────────────────────────────────────────────────────────────────────────
vec3 curvatureTint(vec3 base) {
    return mix(base, uGlowColor, uCurvature * 0.25);
}

// ─────────────────────────────────────────────────────────────────────────────

void main() {
    float halfW = vHalfWidth + u_thickness * 0.5;

    // Approx distance to Bézier centreline
    float dist  = approxDistToCurve();

    // Antialiased stroke coverage
    float strokeAlpha = 1.0 - smoothstep(halfW - 0.75, halfW + 0.75, dist);

    // Dash mask
    float dash = dashMask(vT);
    strokeAlpha *= dash;

    // Glow (beneath stroke)
    float glowAlpha = 0.0;
    if (uGlowRadius > 0.5) {
        float glowD = max(0.0, dist - halfW);
        glowAlpha   = uGlowAlpha
                    * exp(-glowD * glowD / (uGlowRadius * uGlowRadius * 0.5))
                    * dash;
    }

    // ── Species colour gradient: source → target along v_t ──────────────
    vec3 speciesColor = mix(u_sourceColor, u_targetColor, v_t);

    // ── Flow pulse: scrolling bright band along the edge ────────────────
    float pulse = fract(v_t - u_time * u_flowSpeed);
    // Shape pulse into a soft peak (narrow bright band, fades to ~0.3)
    float pulseIntensity = smoothstep(0.0, 0.15, pulse)
                         * (1.0 - smoothstep(0.15, 0.45, pulse));
    // Blend: species gradient base brightened by flow pulse
    vec3 baseColor = speciesColor * (1.0 + 0.6 * pulseIntensity);

    // Optional curvature tint for skip connections
    vec3 strokeCol = curvatureTint(baseColor);

    // Composite
    vec3  col   = mix(uGlowColor, strokeCol, strokeAlpha);
    float alpha = max(strokeAlpha, glowAlpha) * uAlpha;

    if (alpha < 0.004) discard;

    finalColor = vec4(col * alpha, alpha);  // premultiplied alpha
}
