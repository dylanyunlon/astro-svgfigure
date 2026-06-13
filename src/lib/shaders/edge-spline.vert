#version 300 es
// ── edge-spline.vert ──────────────────────────────────────────────────────────
// Cubic Bézier skip-connection shader — vertex stage.
//
// Uses instanced rendering: each instance is one Bézier curve segment (one of
// SUBDIVISIONS quads).  The vertex shader evaluates the curve parametrically,
// expands it into a screen-space quad strip, and passes local state to the
// fragment stage for SDF antialiasing.
//
// Geometry per instance:
//   A single quad (4 vertices, 2 triangles) covering the arc from t=tA to t=tB.
//   The quad is oriented along the tangent so the perpendicular always faces
//   away from the curve centre — ensuring correct coverage at sharp bends.
//
// Per-vertex attributes:
//   aUV         — (u, v): u selects t-side (0=tA, 1=tB), v selects perp side
//   aInstanceId — which subdivision index this instance covers (float, instanced)
//
// Uniforms:
//   uP0, uP1          — Bézier endpoints in pixel space (vec2 each)
//   uCtrl0, uCtrl1    — Bézier control points  (vec2 each)
//   uLineWidth        — full stroke width in pixels (float)
//   uResolution       — canvas (w, h) in pixels (vec2)
//   uSubdivisions     — total number of quad instances along curve (float)
//
// The CPU derives uCtrl0/uCtrl1 from curvature:
//   ctrl0 = P0 + perp * curvature * segLen * 0.5
//   ctrl1 = P1 + perp * curvature * segLen * 0.5
// where perp is the unit normal of the P0→P1 chord, and curvature is the
// value from topology.json (0.6 for skip connections).
//
// Outputs to fragment:
//   vT         — normalised arc parameter [0,1] at this vertex's t-side
//   vDist      — always 0 at centreline; fragment computes actual distance
//   vTangent   — unit tangent at tMid for SDF orientation
//   vCurvePx   — pixel-space curve position at tMid (centre of this quad)
//   vHalfWidth — half-width in pixels
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── per-vertex ────────────────────────────────────────────────────────────────
in vec2  aUV;          // (u∈[0,1] along t-axis, v∈[0,1] across)
in float aInstanceId;  // which quad (0..uSubdivisions-1)

// ── uniforms ──────────────────────────────────────────────────────────────────
uniform vec2  uP0;
uniform vec2  uP1;
uniform vec2  uCtrl0;
uniform vec2  uCtrl1;
uniform float uLineWidth;
uniform vec2  uResolution;
uniform float uSubdivisions;

// ── outs ──────────────────────────────────────────────────────────────────────
out float vT;
out vec2  vTangentDir;
out vec2  vCurvePx;
out float vHalfWidth;
out vec2  vFragCoordPx;

// ── helpers ───────────────────────────────────────────────────────────────────

// Cubic Bézier position
vec2 bezier(float t) {
    float mt  = 1.0 - t;
    float mt2 = mt  * mt;
    float t2  = t   * t;
    return mt2*mt*uP0 + 3.0*mt2*t*uCtrl0 + 3.0*mt*t2*uCtrl1 + t2*t*uP1;
}

// Cubic Bézier derivative (tangent), not normalised
vec2 bezierTangent(float t) {
    float mt = 1.0 - t;
    return 3.0*(mt*mt*(uCtrl0 - uP0)
              + 2.0*mt*t*(uCtrl1 - uCtrl0)
              + t*t*(uP1 - uCtrl1));
}

vec2 pixelToNDC(vec2 px) {
    return (px / uResolution) * 2.0 - 1.0;
}

// ─────────────────────────────────────────────────────────────────────────────

void main() {
    float subs  = uSubdivisions;
    float idx   = aInstanceId;

    // t-range for this quad: [tA, tB]
    float tA    = (idx       ) / subs;
    float tB    = (idx + 1.0 ) / subs;
    float tSide = mix(tA, tB, aUV.x);     // t at this vertex's longitudinal end
    float tMid  = (tA + tB) * 0.5;

    float halfW = uLineWidth * 0.5 + 1.5;  // 1.5 px AA headroom

    // Curve position and tangent at this vertex's t
    vec2 posPx  = bezier(tSide);
    vec2 tanRaw = bezierTangent(tSide);
    float tanLen = length(tanRaw);
    vec2 unit   = (tanLen > 0.001) ? tanRaw / tanLen : vec2(1.0, 0.0);
    vec2 perp   = vec2(-unit.y, unit.x);

    // Expand quad across the perpendicular
    float perpSign = mix(-1.0, 1.0, aUV.y);
    vec2  cornerPx = posPx + perp * (halfW * perpSign);

    // Outputs
    vFragCoordPx = cornerPx;
    vT           = tSide;
    vTangentDir  = unit;
    vCurvePx     = bezier(tMid);
    vHalfWidth   = halfW - 1.5;

    // Y-flip for NDC (canvas Y-down → NDC Y-up)
    vec2 ndc = pixelToNDC(cornerPx);
    ndc.y    = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
