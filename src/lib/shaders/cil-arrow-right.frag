precision mediump float;

uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_arrowWidth;  // stroke thickness [0..1]
uniform float u_time;

// SDF chevron / arrow-right glyph pointing +X, centered at origin, scale ~1
float sdArrowRight(vec2 p, float w) {
  // Two diagonal strokes meeting at tip
  // Top stroke: goes from (-0.4, 0.4) to (0.4, 0.0)
  // Bot stroke: goes from (-0.4,-0.4) to (0.4, 0.0)

  vec2 a1 = vec2(-0.45,  0.40);
  vec2 b1 = vec2( 0.45,  0.0 );
  vec2 a2 = vec2(-0.45, -0.40);
  vec2 b2 = vec2( 0.45,  0.0 );

  // Segment distance helper inline
  vec2 ab1 = b1 - a1;
  float t1 = clamp(dot(p - a1, ab1) / dot(ab1, ab1), 0.0, 1.0);
  float d1 = length(p - a1 - t1 * ab1);

  vec2 ab2 = b2 - a2;
  float t2 = clamp(dot(p - a2, ab2) / dot(ab2, ab2), 0.0, 1.0);
  float d2 = length(p - a2 - t2 * ab2);

  return min(d1, d2) - w;
}

void main() {
  vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;

  // Scrolling flow: tile horizontally, shift over time
  float cols   = 3.0;
  float rows   = 3.0;
  vec2  scroll = vec2(u_time * 0.25, 0.0);

  vec2  tiled  = fract(uv * vec2(cols, rows) + scroll);
  vec2  lp     = tiled * 2.0 - 1.0;   // [-1,1]

  float d    = sdArrowRight(lp, u_arrowWidth * 0.5);
  float mask = smoothstep(0.02, -0.01, d);

  // Trailing fade: arrows dim to the left
  float fade = smoothstep(0.0, 0.6, tiled.x);
  float alpha = mask * (0.4 + 0.6 * fade);

  gl_FragColor = vec4(u_fillColor, clamp(alpha, 0.0, 1.0) * u_opacity);
}
