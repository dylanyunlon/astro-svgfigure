
#require(transformUV.glsl)

void main() {
    // float transition = smoothstep(0.0, 0.8, uAlpha);
    // float gridV = mix(20.0, 100.0, transition);
    // vec2 gridSize = vec2(gridV, floor(gridV/(resolution.x/resolution.y)));
    // vec2 uv = floor(vUv * gridSize) / gridSize;
    // uv += (1.0-transition) * (1.0/gridV) * 0.4;
    // uv = mix(uv, vUv,transition);

    vec4 color = texture2D(tMap, vUv);
    color.a *= 0.8 + sin(time * 2.0 + vUv.y * 2.0 - vWorldPos.x * 0.02) * 0.2;
    color.a *= uAlpha;
    gl_FragColor = color;
}{@}gluimask.fs{@}uniform vec4 uMaskValues;

#require(range.glsl)

vec2 getMaskUV() {
    vec2 ores = gl_FragCoord.xy / resolution;
    vec2 uv;
    uv.x = range(ores.x, uMaskValues.x, uMaskValues.z, 0.0, 1.0);
    uv.y = 1.0 - range(1.0 - ores.y, uMaskValues.y, uMaskValues.w, 0.0, 1.0);
    return uv;
}{@}LightVolume.glsl{@}#!ATTRIBUTES
attribute vec3 offset;
attribute vec4 attribs;

#!UNIFORMS
uniform sampler2D tMap;
uniform sampler2D tMask;

uniform float uScale;
uniform float uSeparation;
uniform float uAlpha;
uniform float uMaskScale;
uniform float uRotateSpeed;
uniform float uRotateTexture;
uniform float uNoiseScale;
uniform float uNoiseSpeed;
uniform float uNoiseRange;
uniform float uOffset;
uniform float uScrollX;
uniform float uScrollY;
uniform float uHueShift;
uniform vec3 uColor;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;
varying vec4 vAttribs;
varying float vOffset;