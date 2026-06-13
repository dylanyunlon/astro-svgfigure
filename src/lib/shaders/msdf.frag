#extension GL_OES_standard_derivatives : enable

precision highp float;

uniform sampler2D uMsdfTexture;
uniform vec3 uColor;
uniform float uOpacity;
uniform float uOutlineWidth;
uniform vec3 uOutlineColor;

varying vec2 vUv;

float median(float r, float g, float b) {
  return max(min(r, g), min(max(r, g), b));
}

void main() {
  vec3 sample = texture2D(uMsdfTexture, vUv).rgb;
  float sigDist = median(sample.r, sample.g, sample.b) - 0.5;
  float w = fwidth(sigDist);
  float fill = smoothstep(-w, w, sigDist);
  float outline = smoothstep(-w, w, sigDist + uOutlineWidth);
  vec3 color = mix(uOutlineColor, uColor, fill);
  gl_FragColor = vec4(color, outline * uOpacity);
}
