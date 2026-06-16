precision mediump float;

uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_arrowWidth;  // stroke thickness [0..1]
uniform float u_time;

// ── lygia/math/saturate.glsl (inlined) ──────────────────────────────────────
#ifndef FNC_SATURATE
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

// ── lygia/sdf/lineSDF.glsl (inlined) ────────────────────────────────────────
// contributors: Inigo Quiles
// Segment SDF: returns the unsigned distance from point st to segment [a,b].
#ifndef FNC_LINESDF
#define FNC_LINESDF
float lineSDF(in vec2 st, in vec2 a, in vec2 b) {
    vec2 b_to_a = b - a;
    vec2 to_a   = st - a;
    float h = saturate(dot(to_a, b_to_a) / dot(b_to_a, b_to_a));
    return length(to_a - h * b_to_a);
}
#endif
// ── end lygia lineSDF ────────────────────────────────────────────────────────

// SDF chevron / arrow-right glyph pointing +X, centered at origin, scale ~1.
// Uses lygia lineSDF for both diagonal strokes.
float sdArrowRight(vec2 p, float w) {
    vec2 a1 = vec2(-0.45,  0.40);
    vec2 b1 = vec2( 0.45,  0.0 );
    vec2 a2 = vec2(-0.45, -0.40);
    vec2 b2 = vec2( 0.45,  0.0 );

    float d1 = lineSDF(p, a1, b1);
    float d2 = lineSDF(p, a2, b2);

    return min(d1, d2) - w;
}

void main() {
  vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;

  float cols   = 3.0;
  float rows   = 3.0;
  vec2  scroll = vec2(u_time * 0.25, 0.0);

  vec2  tiled  = fract(uv * vec2(cols, rows) + scroll);
  vec2  lp     = tiled * 2.0 - 1.0;   // [-1,1]

  float d    = sdArrowRight(lp, u_arrowWidth * 0.5);
  float mask = smoothstep(0.02, -0.01, d);

  float fade  = smoothstep(0.0, 0.6, tiled.x);
  float alpha = mask * (0.4 + 0.6 * fade);

  gl_FragColor = vec4(u_fillColor, clamp(alpha, 0.0, 1.0) * u_opacity);
}
