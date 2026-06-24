#extension GL_OES_standard_derivatives : enable

precision highp float;

uniform sampler2D uMsdfTexture;
uniform vec3  uColor;
uniform float uOpacity;
uniform float uOutlineWidth;
uniform vec3  uOutlineColor;

// M1138 — shadow + glow
// uShadowMode: 0 = normal pass, 1 = shadow pass (render shifted silhouette)
uniform float uShadowMode;
// uGlowColor: species colour * 0.3, added in the SDF halo region
uniform vec3  uGlowColor;
// uGlowRadius: how far outside the fill edge the glow reaches (in SDF units)
uniform float uGlowRadius;

varying vec2 vUv;

float median(float r, float g, float b) {
  return max(min(r, g), min(max(r, g), b));
}

void main() {
  vec3  samp    = texture2D(uMsdfTexture, vUv).rgb;
  float sigDist = median(samp.r, samp.g, samp.b) - 0.5;
  float w       = fwidth(sigDist);

  // ── Shadow pass: render a semi-transparent black silhouette ──────────────
  if (uShadowMode > 0.5) {
    float fill = smoothstep(-w, w, sigDist);
    gl_FragColor = vec4(0.0, 0.0, 0.0, fill * uOpacity * 0.45);
    return;
  }

  // ── Normal pass ──────────────────────────────────────────────────────────
  float fill    = smoothstep(-w, w, sigDist);
  float outline = smoothstep(-w, w, sigDist + uOutlineWidth);

  // Glow: smoothly lights up the halo region just outside the fill edge.
  // sigDist ∈ [-glowRadius, 0] → glow peaks at sigDist=0 (edge), falls to 0 at -glowRadius.
  float glowRadius = max(uGlowRadius, w * 2.0);
  float glowFactor = smoothstep(-glowRadius, 0.0, sigDist) * (1.0 - fill);
  // Glow is additive over the outline colour, then we blend the fill colour on top.
  vec3 color = mix(uOutlineColor, uColor, fill);
  color      = mix(color, color + uGlowColor, glowFactor);

  gl_FragColor = vec4(color, outline * uOpacity);
}
