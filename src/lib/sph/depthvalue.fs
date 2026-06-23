// depthvalue.fs — AT Depth Value Encode / Decode
// Based on upstream/lygia/space/depth2viewZ.glsl + viewZ2depth.glsl +
// linearizeDepth.glsl (Patricio Gonzalez Vivo, Prosperity/Patron License).
// Supplies linearised eye-space Z to SSAO, SSR, and DoF post-process passes.
// ─────────────────────────────────────────────────────────────────────────────
// Uniforms:  uDepthTex — hardware depth sampler2D
//            uNear, uFar — camera clip planes
// Varying:   vUV — fullscreen quad UV [0,1]²
// Output:    r = linear depth (eye-space Z, positive = toward camera)
// ─────────────────────────────────────────────────────────────────────────────
precision highp float;
uniform sampler2D uDepthTex;
uniform float     uNear;
uniform float     uFar;
varying vec2      vUV;
float linearizeDepth(float d, float near, float far) {
    float ndc = 2.0 * d - 1.0;
    return (2.0 * near * far) / (far + near - ndc * (far - near));
}
float depth2viewZ(float depth, float near, float far) {
    return (near * far) / ((far - near) * depth - far);
}
void main() {
    float raw  = texture2D(uDepthTex, vUV).r;
    float linD = linearizeDepth(raw, uNear, uFar);   // world-space distance
    float viewZ = depth2viewZ(raw, uNear, uFar);      // signed eye-space Z
    // Pack: r=linearDepth (0→1 remapped), g=raw, b=viewZ (normalised), a=1
    gl_FragColor = vec4(linD / uFar, raw, (-viewZ) / uFar, 1.0);
}
