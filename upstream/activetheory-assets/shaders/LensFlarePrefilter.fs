
#require(transformUV.glsl)
#require(luma.fs)

void main() {

    vec2 uv = vUv;

    uv = rotateUV(uv, -uRotate);

    vec4 c = texture2D(tMap, vec2(uv.x, uv.y));

    // threshold the brightness

    float brightness = luma(c.rgb);
    if (brightness < uThreshold) {
        c = vec4(0.);
    }

    gl_FragColor = vec4(c.rgb, 1.0);
}{@}LensFlareUp.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tHigh;
uniform sampler2D tScene;
uniform float uStretch;
uniform float uSoftenEdge;
uniform vec2 uResolution;

#!VARYINGS
varying vec2 vUv;