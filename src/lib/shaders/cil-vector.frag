precision mediump float;

uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_arrowCount;   // arrows per row/col
uniform float u_angleSpread;  // variation in radians

// ---- SDF helpers ----
float sdBox(vec2 p, vec2 b) {
  vec2 d = abs(p) - b;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float sdTriangle(vec2 p, float r) {
  // equilateral triangle pointing right
  const float k = 1.7320508; // sqrt(3)
  p.x = abs(p.x) - r;
  p.y = p.y + r / k;
  if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
  p.x -= clamp(p.x, -2.0 * r, 0.0);
  return -length(p) * sign(p.y);
}

float drawArrow(vec2 p, float angle, float scale) {
  float c = cos(angle), s = sin(angle);
  // rotate p into arrow local space
  vec2 lp = vec2(c * p.x + s * p.y, -s * p.x + c * p.y) / scale;

  // shaft
  float shaft = sdBox(lp - vec2(-0.15, 0.0), vec2(0.22, 0.045));

  // head (triangle)
  float head  = sdTriangle(lp - vec2(0.13, 0.0), 0.16);

  float d = min(shaft, head);
  return smoothstep(0.01, -0.01, d);
}

// pseudo-random
float rand(vec2 co) {
  return fract(sin(dot(co, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw; // [0,1]

  float n     = u_arrowCount;
  vec2  cell  = floor(uv * n);
  vec2  local = fract(uv * n) - 0.5; // [-0.5,0.5] within each cell

  // Per-cell angle variation
  float baseAngle = 0.0; // pointing right by default
  float jitter    = (rand(cell) * 2.0 - 1.0) * u_angleSpread;
  float angle     = baseAngle + jitter;

  float scale = 0.45;
  float mask  = drawArrow(local, angle, scale);

  gl_FragColor = vec4(u_fillColor, mask * u_opacity);
}
