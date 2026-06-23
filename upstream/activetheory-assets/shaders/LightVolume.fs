
#require(rgb2hsv.fs)
#require(range.glsl)
#require(transformUV.glsl)
#require(simplenoise.glsl)

void main() {
    vec3 color = rgb2hsv(uColor);
    color += vOffset * uHueShift * 0.01;
    color = hsv2rgb(color);

    vec2 auv = vUv;
    if (uRotateTexture > 0.0) {
        auv = rotateUV(vUv, time * uRotateTexture * 0.1);
    }

    float alpha = texture2D(tMap, auv).r;

    vec2 uv = scaleUV(vUv, vec2(uMaskScale));

    if (uNoiseSpeed > 0.0) {
        float noise = cnoise(vPos * uNoiseScale + (time * uNoiseSpeed));
        uv += noise * uNoiseRange * 0.1;
        uv = scaleUV(uv, vec2(range(noise, -1.0, 0.0, 0.96, 1.02)));
        uv.x += sin(time * 0.04) * 0.3;
    }

    if (uRotateSpeed > 0.0) {
        uv = rotateUV(uv, uRotateSpeed * time * range(vAttribs.x, 0.0, 1.0, 0.5, 1.5));
        uv.x += time * uScrollX * 0.1 * range(vAttribs.y, 0.0, 1.0, 0.5, 1.5);
        uv.y += time * uScrollY * 0.1 * range(vAttribs.z, 0.0, 1.0, 0.5, 1.5);
    }

    float mask = texture2D(tMask, uv).r;
    alpha *= mask;

    #drawbuffer Color gl_FragColor = vec4(color, alpha * uAlpha);
    #drawbuffer VolumetricLight gl_FragColor = vec4(color, alpha * uAlpha);
}
{@}luma.fs{@}float luma(vec3 color) {
  return dot(color, vec3(0.299, 0.587, 0.114));
}

float luma(vec4 color) {
  return dot(color.rgb, vec3(0.299, 0.587, 0.114));
}{@}matcap.vs{@}vec2 reflectMatcap(vec3 position, mat4 modelMatrix, vec3 normal) {
    vec3 worldNormal = mat3(modelMatrix[0].xyz, modelMatrix[1].xyz, modelMatrix[2].xyz) * normal;
    vec3 worldPos = (modelMatrix * vec4(position, 1.0)).xyz;
    vec3 viewDir = normalize(cameraPosition - worldPos);
    vec3 x = normalize(vec3(viewDir.z, 0.0, - viewDir.x));
    vec3 y = cross(viewDir, x);
    vec2 uv = vec2(dot(x, worldNormal), dot(y, worldNormal)) * 0.495 + 0.5; // 0.495 to remove artifacts caused by undersized matcap disks
    return uv;
}

vec2 reflectMatcap(vec3 worldPos, vec3 worldNormal) {
    vec3 viewDir = normalize(cameraPosition - worldPos);
    vec3 x = normalize(vec3(viewDir.z, 0.0, - viewDir.x));
    vec3 y = cross(viewDir, x);
    vec2 uv = vec2(dot(x, worldNormal), dot(y, worldNormal)) * 0.495 + 0.5; // 0.495 to remove artifacts caused by undersized matcap disks
    return uv;
}
{@}BasicMirror.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMirrorReflection;
uniform mat4 uMirrorMatrix;

#!VARYINGS
varying vec4 vMirrorCoord;