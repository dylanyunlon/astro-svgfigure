// Auto-generated — do not edit by hand
// Exports raw GLSL fragment shader strings for all cell species.

export const CIL_EYE_FRAG = `
precision mediump float;

uniform vec4  u_bbox;         // x, y, width, height in canvas coords
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_numRays;
uniform float u_pupilRadius;
uniform float u_focalIntensity;
uniform float u_time;

void main() {
  // Normalize fragment to bbox-local UV [0,1]
  vec2 fragCoord = gl_FragCoord.xy;
  vec2 uv = (fragCoord - u_bbox.xy) / u_bbox.zw;

  // Map to [-1,1] centered space
  vec2 p = uv * 2.0 - 1.0;

  float dist  = length(p);
  float angle = atan(p.y, p.x);

  // --- Pupil ---
  float pupil = 1.0 - smoothstep(u_pupilRadius - 0.02, u_pupilRadius + 0.02, dist);

  // --- Iris ring ---
  float iris = smoothstep(u_pupilRadius + 0.02, u_pupilRadius + 0.08, dist)
             * (1.0 - smoothstep(0.85, 1.0, dist));

  // --- Radial rays ---
  float halfStep  = 3.14159265 / u_numRays;
  float rayAngle  = mod(angle + u_time * 0.3, halfStep * 2.0) - halfStep;
  float rayMask   = smoothstep(0.07, 0.0, abs(rayAngle));
  float rayFade   = smoothstep(1.0, u_pupilRadius + 0.1, dist)
                  * smoothstep(u_pupilRadius, u_pupilRadius + 0.12, dist);
  float rays      = rayMask * rayFade * u_focalIntensity;

  // --- Sclera (outer white ellipse halo) ---
  float sclera = smoothstep(1.05, 0.88, dist);

  float alpha = clamp(sclera * (iris + rays) + pupil, 0.0, 1.0);

  gl_FragColor = vec4(u_fillColor, alpha * u_opacity);
}
`;

export const CIL_BOLT_FRAG = `
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
`;

export const CIL_VECTOR_FRAG = `
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
`;

export const CIL_PLUS_FRAG = `
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
`;

export const CIL_ARROW_RIGHT_FRAG = `
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
`;

