precision mediump float;

uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_armLength;    // half-length of each arm  [0..1]
uniform float u_strokeWidth;  // half-width of stroke     [0..1]

// SDF for an axis-aligned plus centered at origin
float sdPlus(vec2 p, float armLen, float sw) {
  // horizontal bar
  vec2 dH = abs(p) - vec2(armLen, sw);
  float h  = length(max(dH, 0.0)) + min(max(dH.x, dH.y), 0.0);

  // vertical bar
  vec2 dV = abs(p) - vec2(sw, armLen);
  float v  = length(max(dV, 0.0)) + min(max(dV.x, dV.y), 0.0);

  return min(h, v);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;
  vec2 p  = uv * 2.0 - 1.0;   // [-1,1]

  float d    = sdPlus(p, u_armLength, u_strokeWidth);
  float mask = smoothstep(0.015, -0.015, d);

  // Soft glow
  float glow = smoothstep(0.08, 0.0, d) * 0.25;

  float alpha = clamp(mask + glow, 0.0, 1.0);
  gl_FragColor = vec4(u_fillColor, alpha * u_opacity);
}
