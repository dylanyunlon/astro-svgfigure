precision mediump float;

uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_zigzagCount;
uniform float u_amplitude;
uniform float u_time;

// ---- helpers ----
float seg(vec2 p, vec2 a, vec2 b, float w) {
  vec2 ab = b - a;
  vec2 ap = p - a;
  float t  = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
  float d  = length(ap - t * ab);
  return smoothstep(w, w * 0.4, d);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;
  vec2 p  = uv * 2.0 - 1.0;          // [-1,1]

  float strokeW = 0.045;
  float total   = 0.0;

  // Build zigzag path from top to bottom
  float steps = u_zigzagCount;
  float dy    = 2.0 / steps;

  // Animated phase offset
  float phase = sin(u_time * 2.5) * 0.15;

  for (float i = 0.0; i < 32.0; i++) {
    if (i >= steps) break;

    float t0 = -1.0 + i       * dy;
    float t1 = -1.0 + (i+1.0) * dy;

    // Alternate left/right with amplitude + phase
    float side0 = (mod(i,       2.0) < 1.0 ? 1.0 : -1.0);
    float side1 = (mod(i + 1.0, 2.0) < 1.0 ? 1.0 : -1.0);

    vec2 a = vec2(side0 * u_amplitude + phase, t0);
    vec2 b = vec2(side1 * u_amplitude + phase, t1);

    total = max(total, seg(p, a, b, strokeW));
  }

  // Glow pass (wider, dimmer)
  float glow = 0.0;
  for (float i = 0.0; i < 32.0; i++) {
    if (i >= steps) break;
    float t0   = -1.0 + i * dy;
    float t1   = -1.0 + (i+1.0) * dy;
    float s0   = (mod(i,       2.0) < 1.0 ? 1.0 : -1.0);
    float s1   = (mod(i + 1.0, 2.0) < 1.0 ? 1.0 : -1.0);
    vec2 a     = vec2(s0 * u_amplitude + phase, t0);
    vec2 b     = vec2(s1 * u_amplitude + phase, t1);
    glow       = max(glow, seg(p, a, b, strokeW * 3.5) * 0.3);
  }

  float alpha = clamp(total + glow, 0.0, 1.0);
  gl_FragColor = vec4(u_fillColor, alpha * u_opacity);
}
