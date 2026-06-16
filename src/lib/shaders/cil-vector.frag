precision mediump float;

uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_arrowCount;   // arrows per row/col
uniform float u_angleSpread;  // variation in radians

// ── lygia/math/const.glsl (inlined, PI/TAU only) ────────────────────────────
#ifndef PI
#define PI  3.1415926535897932384626433832795
#endif
#ifndef TAU
#define TAU 6.2831853071795864769252867665590
#endif

// ── lygia/sdf/polySDF.glsl (inlined) ────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns SDF for a regular V-sided polygon, UV [0,1] space centred at 0.5.
#ifndef FNC_POLYSDF
#define FNC_POLYSDF
float polySDF(in vec2 st, in int V) {
    st = st * 2.0 - 1.0;
    float a = atan(st.x, st.y) + PI;
    float r = length(st);
    float v = TAU / float(V);
    return cos(floor(0.5 + a / v) * v - a) * r;
}
#endif
// ── end lygia polySDF ────────────────────────────────────────────────────────

// ── lygia/math/saturate.glsl (inlined) ──────────────────────────────────────
#ifndef FNC_SATURATE
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

// ── lygia/sdf/lineSDF.glsl (inlined) ────────────────────────────────────────
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

// Box SDF (not in lygia 2D but equivalent to rectSDF signed variant)
float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

// Arrow: shaft via sdBox + head via polySDF (triangle = 3-sided poly)
float drawArrow(vec2 p, float angle, float scale) {
    float c = cos(angle), s = sin(angle);
    vec2 lp = vec2(c * p.x + s * p.y, -s * p.x + c * p.y) / scale;

    // Shaft
    float shaft = sdBox(lp - vec2(-0.15, 0.0), vec2(0.22, 0.045));

    // Arrowhead — polySDF with 3 sides (equilateral triangle) in local UV space.
    // Map the head region to [0,1] UV for polySDF and test if inside.
    vec2 headUV = (lp - vec2(0.13, 0.0)) / 0.32 + 0.5;
    // Rotate so triangle points right (+X) in local space: offset angle by -PI/2
    headUV = headUV - 0.5;
    float tmp = headUV.x;
    headUV.x = -headUV.y;
    headUV.y =  tmp;
    headUV = headUV + 0.5;
    float head = polySDF(headUV, 3) * 0.32 - 0.16;

    float d = min(shaft, head);
    return smoothstep(0.01, -0.01, d);
}

// pseudo-random
float rand(vec2 co) {
    return fract(sin(dot(co, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;

    float n     = u_arrowCount;
    vec2  cell  = floor(uv * n);
    vec2  local = fract(uv * n) - 0.5;  // [-0.5, 0.5]

    float jitter = (rand(cell) * 2.0 - 1.0) * u_angleSpread;
    float angle  = jitter;

    float scale = 0.45;
    float mask  = drawArrow(local, angle, scale);

    gl_FragColor = vec4(u_fillColor, mask * u_opacity);
}
