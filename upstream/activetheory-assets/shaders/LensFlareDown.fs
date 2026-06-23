#require(transformUV.glsl)
void main() {

    vec2 uv = vUv;

    float dx = 1. / uResolution.x;

    float stretch = uStretch;

    float u0 = uv.x - ((dx * 5.) * stretch);
    float u1 = uv.x - ((dx * 3.) * stretch);
    float u2 = uv.x - ((dx * 1.) * stretch);
    float u3 = uv.x + ((dx * 1.) * stretch);
    float u4 = uv.x + ((dx * 3.) * stretch);
    float u5 = uv.x + ((dx * 5.) * stretch);

    vec3 c0 = texture2D(tMap, vec2(u0, uv.y)).rgb;
    vec3 c1 = texture2D(tMap, vec2(u1, uv.y)).rgb;
    vec3 c2 = texture2D(tMap, vec2(u2, uv.y)).rgb;
    vec3 c3 = texture2D(tMap, vec2(u3, uv.y)).rgb;
    vec3 c4 = texture2D(tMap, vec2(u4, uv.y)).rgb;
    vec3 c5 = texture2D(tMap, vec2(u5, uv.y)).rgb;

    vec3 col =  vec3((c0 + c1 * 2. + c2 * 3. + c3 * 3. + c4 * 2. + c5) / 12.);

    gl_FragColor = vec4( col.rgb, 1.0 );
}{@}LensFlarePrefilter.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uThreshold;
uniform float uRotate;

#!VARYINGS
varying vec2 vUv;