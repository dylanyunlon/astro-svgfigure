precision mediump float;

// LUT Color Grading Pipeline
// 3D LUT encoded as 2D texture (standard 16x16 grid = 256 color entries)
// QoS profile drives the color tone of the entire world

uniform sampler2D tScene;        // input scene texture
uniform sampler2D tLUT;          // LUT texture (256x16 strip)
uniform float u_lutIntensity;    // blend factor 0..1
uniform float u_lutSize;         // LUT dimension (default 16)
uniform vec2 u_resolution;

varying vec2 vUv;

// Standard 3D LUT lookup from 2D strip texture
// LUT is arranged as 16 slices of 16x16, laid out horizontally = 256x16 px
vec3 lutLookup(sampler2D lutTex, vec3 color, float size) {
    float sliceSize = 1.0 / size;
    float slicePixelSize = sliceSize / size;
    float sliceInnerSize = slicePixelSize * (size - 1.0);

    float zSlice0 = min(floor(color.b * (size - 1.0)), size - 2.0);
    float zSlice1 = zSlice0 + 1.0;

    float xOffset = slicePixelSize * 0.5 + color.r * sliceInnerSize;
    float yOffset = slicePixelSize * 0.5 + color.g * sliceInnerSize;

    vec2 uv0 = vec2(zSlice0 * sliceSize + xOffset, yOffset);
    vec2 uv1 = vec2(zSlice1 * sliceSize + xOffset, yOffset);

    vec3 c0 = texture2D(lutTex, uv0).rgb;
    vec3 c1 = texture2D(lutTex, uv1).rgb;

    float zFract = fract(color.b * (size - 1.0));
    return mix(c0, c1, zFract);
}

void main() {
    vec3 scene = texture2D(tScene, vUv).rgb;
    
    // Clamp to valid LUT range
    vec3 clamped = clamp(scene, 0.0, 1.0);
    
    // Apply LUT
    vec3 graded = lutLookup(tLUT, clamped, u_lutSize);
    
    // Blend original with graded
    gl_FragColor = vec4(mix(scene, graded, u_lutIntensity), 1.0);
}
