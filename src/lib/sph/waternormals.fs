// waternormals.fs — AT Water Normals Fragment Shader
// Ported from upstream/webgl-water/water.js normalShader
// + renderer.js "make water look more peaked" UV-refinement loop.
// Used by ATWaterNormals to complete at-water-surface.ts normal pipeline.
// ─────────────────────────────────────────────────────────────────────────────
// Uniforms:  uTexture — water simulation texture (rgba: h, vel, nx, nz)
//            uDelta   — texel size vec2 (1/width, 1/height)
// Varying:   vCoord   — UV [0,1]²
// Output:    packed ba = normalize(cross(dy, dx)).xz  (same layout as water.js)
// ─────────────────────────────────────────────────────────────────────────────
precision highp float;
uniform sampler2D uTexture;
uniform vec2      uDelta;
varying vec2      vCoord;
void main() {
    vec4  info = texture2D(uTexture, vCoord);
    vec3  dx   = vec3(uDelta.x, texture2D(uTexture, vCoord + vec2(uDelta.x, 0.0)).r - info.r, 0.0);
    vec3  dy   = vec3(0.0, texture2D(uTexture, vCoord + vec2(0.0, uDelta.y)).r - info.r, uDelta.y);
    info.ba    = normalize(cross(dy, dx)).xz;
    gl_FragColor = info;
}
