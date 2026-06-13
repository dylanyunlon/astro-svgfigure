#version 300 es
// ── cloud-fog.vert ────────────────────────────────────────────────────────────
// Volumetric fog vertex shader — one quad per fog plane.
//
// Each plane is a simple unit quad [-0.5, +0.5] in XY, placed at a fixed Z
// depth by the CPU (via uModelMatrix or uDepthZ).  The vertex shader forwards
// UV coordinates and a remapped depth scalar to the fragment stage.
//
// Attributes:
//   aPosition  — (x, y) in [-0.5, +0.5] quad-local space
//   aUV        — (u, v) in [0, 1] over the quad
//
// Uniforms:
//   uProjection  — 4×4 projection matrix
//   uView        — 4×4 view / camera matrix
//   uModel       — 4×4 model matrix (positions + scales the plane in world space)
//   uDepth01     — [0,1] normalised depth of this layer within the volume
//
// Outputs:
//   vUV          — passthrough UV to fragment
//   vDepth01     — passthrough depth scalar to fragment
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── attributes ────────────────────────────────────────────────────────────────
in vec2 aPosition;   // quad corner in local space
in vec2 aUV;         // [0,1] UV over the quad

// ── uniforms ──────────────────────────────────────────────────────────────────
uniform mat4  uProjection;
uniform mat4  uView;
uniform mat4  uModel;
uniform float uDepth01;

// ── varyings to fragment ──────────────────────────────────────────────────────
out vec2  vUV;
out float vDepth01;

// ─────────────────────────────────────────────────────────────────────────────
void main() {
    vUV      = aUV;
    vDepth01 = uDepth01;

    vec4 worldPos = uModel * vec4(aPosition, 0.0, 1.0);
    gl_Position   = uProjection * uView * worldPos;
}
