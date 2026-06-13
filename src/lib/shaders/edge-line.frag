#version 300 es
// ── edge-line.frag ────────────────────────────────────────────────────────────
// Straight-line edge shader — fragment stage.
//
// Evaluates the signed-distance-field (SDF) of the line segment in pixel space
// so that:
//   • The stroke has sub-pixel antialiased edges (no jaggies at any scale)
//   • Dash patterns are evaluated analytically (no texture lookup)
//   • The caps are round (zero extra vertex work)
//
// Uniforms:
//   uColor         — RGB stroke colour (vec3, 0-1)
//   uAlpha         — master opacity (float)
//   uLineWidth     — full stroke width in pixels (float)
//   uDashLength    — on-length  of one dash in pixels; 0 = solid (float)
//   uGapLength     — off-length of the gap in pixels (float)
//   uGlowRadius    — optional outer glow radius in pixels; 0 = off (float)
//   uGlowColor     — glow colour (vec3)
//   uGlowAlpha     — glow peak opacity (float)
//   uTime          — seconds, for animated dashes (float)
//   uDashOffset    — extra phase offset in pixels (float)
//
// Inputs from vertex shader:
//   vFragCoordPx   — pixel-space fragment position
//   vP0, vP1       — segment endpoints in pixel space
//   vHalfWidth     — half-width of the stroke (pixels)
//   vUV            — quad UV (not used in SDF evaluation but available)
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── varyings from vertex ──────────────────────────────────────────────────────
in vec2  vFragCoordPx;
in vec2  vP0;
in vec2  vP1;
in float vHalfWidth;
in vec2  vUV;

// ── uniforms ──────────────────────────────────────────────────────────────────
uniform vec3  uColor;
uniform float uAlpha;
uniform float uLineWidth;
uniform float uDashLength;   // 0 = solid
uniform float uGapLength;
uniform vec3  uGlowColor;
uniform float uGlowRadius;   // 0 = no glow
uniform float uGlowAlpha;
uniform float uTime;
uniform float uDashOffset;

// ── output ────────────────────────────────────────────────────────────────────
out vec4 finalColor;

// ─────────────────────────────────────────────────────────────────────────────
// SDF: distance from point p to segment (a → b)
// Returns (distance_to_segment, t_along_segment)
// ─────────────────────────────────────────────────────────────────────────────
vec2 sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2  pa = p - a;
    vec2  ba = b - a;
    float h  = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return vec2(length(pa - ba * h), h);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dash pattern: returns 1 if this position is "on", 0 if in a gap.
// t_px  — arc-length position along the segment in pixels
// ─────────────────────────────────────────────────────────────────────────────
float dashMask(float t_px) {
    if (uDashLength < 0.5) return 1.0;  // solid line

    float period = uDashLength + uGapLength;
    float phase  = mod(t_px + uTime * 40.0 + uDashOffset, period);

    // Smooth transition at dash edges (1 px soft)
    float onEnd  = uDashLength;
    float on     = smoothstep(0.0, 1.0, phase)
                 * (1.0 - smoothstep(onEnd - 1.0, onEnd, phase));
    return clamp(on, 0.0, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────

void main() {
    vec2  p       = vFragCoordPx;
    float halfW   = vHalfWidth;
    float segLen  = length(vP1 - vP0);

    // SDF
    vec2  sd      = sdSegment(p, vP0, vP1);
    float dist    = sd.x;          // pixels from segment centreline
    float t_norm  = sd.y;          // 0..1 along segment
    float t_px    = t_norm * segLen;

    // Antialiased stroke coverage
    float strokeAlpha = 1.0 - smoothstep(halfW - 0.75, halfW + 0.75, dist);

    // Dash mask
    float dash = dashMask(t_px);
    strokeAlpha *= dash;

    // Glow (rendered beneath the stroke)
    float glowAlpha = 0.0;
    if (uGlowRadius > 0.5) {
        float glowD = max(0.0, dist - halfW);
        glowAlpha   = uGlowAlpha
                    * exp(-glowD * glowD / (uGlowRadius * uGlowRadius * 0.5))
                    * dash;
    }

    // Composite: glow underneath, stroke on top
    vec3  col   = mix(uGlowColor, uColor, strokeAlpha);
    float alpha = max(strokeAlpha, glowAlpha) * uAlpha;

    if (alpha < 0.004) discard;

    finalColor = vec4(col * alpha, alpha);  // premultiplied alpha
}
