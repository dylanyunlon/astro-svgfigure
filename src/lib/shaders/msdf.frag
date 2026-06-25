#version 300 es
precision highp float;

uniform sampler2D uMsdfTexture;
uniform vec3  uColor;
uniform float uOpacity;
uniform float uOutlineWidth;
uniform vec3  uOutlineColor;

// M1138 — shadow + glow
uniform float uShadowMode;
uniform vec3  uGlowColor;
uniform float uGlowRadius;

in vec2 vUv;
out vec4 fragColor;

float median(float r, float g, float b) {
  return max(min(r, g), min(max(r, g), b));
}

void main() {
  vec3  samp    = texture(uMsdfTexture, vUv).rgb;
  float sigDist = median(samp.r, samp.g, samp.b) - 0.5;
  float w       = fwidth(sigDist);

  // Shadow pass
  if (uShadowMode > 0.5) {
    float fill = smoothstep(-w, w, sigDist);
    fragColor = vec4(0.0, 0.0, 0.0, fill * uOpacity * 0.45);
    return;
  }

  // Normal pass
  float fill    = smoothstep(-w, w, sigDist);
  float outline = smoothstep(-w, w, sigDist + uOutlineWidth);

  float glowRadius = max(uGlowRadius, w * 2.0);
  float glowFactor = smoothstep(-glowRadius, 0.0, sigDist) * (1.0 - fill);
  vec3 color = mix(uOutlineColor, uColor, fill);
  color      = mix(color, color + uGlowColor, glowFactor);

  fragColor = vec4(color, outline * uOpacity);
}
