#version 300 es
precision mediump float;
out vec4 fragColor;

uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_armLength;    // half-length of each arm  [0..1]
uniform float u_strokeWidth;  // half-width of stroke     [0..1]

// ── lygia/sdf/rectSDF.glsl (inlined) ────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns a rectangular SDF in [-1,1] centred space (max-norm variant).
#ifndef FNC_RECTSDF
#define FNC_RECTSDF
float rectSDF(in vec2 st, in vec2 s) {
    vec2 p = st * 2.0 - 1.0;   // remap [0,1]→[-1,1]; caller passes UV
    return max(abs(p.x / s.x), abs(p.y / s.y));
}
// Signed box SDF variant (used for the arm extrusions)
float sdBox2(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
#endif
// ── end lygia rectSDF ────────────────────────────────────────────────────────

// SDF for an axis-aligned plus centered at origin.
// Built from two overlapping sdBox2 calls — equivalent to the previous
// hand-written sdPlus but now uses the rectSDF family.
float sdPlus(vec2 p, float armLen, float sw) {
    float h = sdBox2(p, vec2(armLen, sw));
    float v = sdBox2(p, vec2(sw, armLen));
    return min(h, v);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;
  vec2 p  = uv * 2.0 - 1.0;   // [-1,1]

  float d    = sdPlus(p, u_armLength, u_strokeWidth);
  float mask = smoothstep(0.015, -0.015, d);

  float glow = smoothstep(0.08, 0.0, d) * 0.25;

  float alpha = clamp(mask + glow, 0.0, 1.0);
  fragColor = vec4(u_fillColor, alpha * u_opacity);
}
