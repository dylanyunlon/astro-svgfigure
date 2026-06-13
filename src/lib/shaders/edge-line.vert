#version 300 es
// ── edge-line.vert ────────────────────────────────────────────────────────────
// Straight-line edge shader — vertex stage.
//
// Renders each edge as a screen-space quad (4 vertices, 2 triangles) that
// tightly wraps the line segment plus half-width padding on every side.
//
// Technique: the CPU uploads the two endpoints once as uniforms; the vertex
// shader expands them into a bbox quad and passes the local-space UV to the
// fragment stage where the SDF antialiased line is evaluated.
//
// Attributes (per-vertex, 4 vertices per quad):
//   aPosition  — (x, y) NDC position of this quad corner
//   aUV        — (u, v) in [0,1] over the quad bounding-box
//
// Uniforms:
//   uP0        — start point in canvas-pixel space (vec2)
//   uP1        — end   point in canvas-pixel space (vec2)
//   uLineWidth — half-width of the rendered stroke in pixels (float)
//   uResolution — canvas (width, height) in pixels (vec2)
//
// Outputs:
//   vUV        — passthrough UV for fragment SDF
//   vP0, vP1   — endpoints forwarded to fragment in normalised canvas coords
//   vHalfWidth — half-width in the same normalised space
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

in vec2 aPosition;   // NDC quad corner  (-1..1, -1..1)
in vec2 aUV;         // [0,1] over the quad bbox

uniform vec2  uP0;          // start point (pixel space)
uniform vec2  uP1;          // end   point (pixel space)
uniform float uLineWidth;   // full stroke width in pixels
uniform vec2  uResolution;  // canvas (w, h) in pixels

// ── outs to fragment ──────────────────────────────────────────────────────────
out vec2  vUV;
out vec2  vP0;
out vec2  vP1;
out float vHalfWidth;
out vec2  vFragCoordPx;  // pixel-space position of this fragment

// ── helpers ───────────────────────────────────────────────────────────────────

// Convert pixel-space point to NDC, accounting for Y-down canvas convention.
vec2 pixelToNDC(vec2 px) {
    return (px / uResolution) * 2.0 - 1.0;
}

// ─────────────────────────────────────────────────────────────────────────────

void main() {
    float halfW = uLineWidth * 0.5 + 1.5;  // 1.5 px AA headroom

    // Direction and perpendicular of the segment in pixel space
    vec2  dir   = uP1 - uP0;
    float len   = length(dir);
    vec2  unit  = (len > 0.001) ? dir / len : vec2(1.0, 0.0);
    vec2  perp  = vec2(-unit.y, unit.x);

    // Quad corners in pixel space:
    //   0 = P0 - perp*hw - unit*hw
    //   1 = P0 + perp*hw - unit*hw
    //   2 = P1 + perp*hw + unit*hw
    //   3 = P1 - perp*hw + unit*hw
    // aUV.x selects along the segment (0 = P0 side, 1 = P1 side)
    // aUV.y selects across the segment (0 = -perp, 1 = +perp)
    vec2 longOff  = mix(-unit * halfW, unit  * halfW, aUV.x);
    vec2 perpOff  = mix(-perp * halfW, perp  * halfW, aUV.y);
    vec2 anchor   = mix(uP0, uP1, aUV.x);
    vec2 cornerPx = anchor + longOff + perpOff;

    // Pass pixel coords to fragment for the SDF evaluation
    vFragCoordPx = cornerPx;
    vP0          = uP0;
    vP1          = uP1;
    vHalfWidth   = halfW - 1.5;  // strip the AA padding back
    vUV          = aUV;

    // Y-flip: canvas is Y-down, NDC is Y-up
    vec2 ndc = pixelToNDC(cornerPx);
    ndc.y    = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
