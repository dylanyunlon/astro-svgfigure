precision mediump float;

uniform vec4  u_bbox;
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_zigzagCount;
uniform float u_amplitude;
uniform float u_time;

// ---- AT UIL params (from channels/physics/xiaodi_options_table.json / cil-bolt) ----
// INPUT_HydraBloom_Bloom_Intensity        : 1.0
// INPUT_HydraBloom_Bloom_Radius           : 1.0
// UnrealBloomComposite globalbloom/bloomStrength : 0.3
// UnrealBloomComposite globalbloom/bloomRadius   : 0.2
// UnrealBloomComposite homebloom/bloomStrength   : 0.6
// UnrealBloomComposite homebloom/bloomRadius     : 0.8
// L_Element_10_home_sceneintensity        : 2.19
// INPUT_Element_1_work_scenewiggle_speed  : 0.7
// BloomLuminosity luminosityThreshold     : 0.0
// CAMERA_Element_1_HomelerpSpeed          : 0.1

const float AT_BLOOM_INTENSITY        = 1.0;   // INPUT_HydraBloom_Bloom_Intensity
const float AT_BLOOM_RADIUS           = 1.0;   // INPUT_HydraBloom_Bloom_Radius
const float AT_GLOBAL_BLOOM_STRENGTH  = 0.3;   // globalbloom/bloomStrength
const float AT_GLOBAL_BLOOM_RADIUS    = 0.2;   // globalbloom/bloomRadius
const float AT_HOME_BLOOM_STRENGTH    = 0.6;   // homebloom/bloomStrength
const float AT_HOME_BLOOM_RADIUS      = 0.8;   // homebloom/bloomRadius
const float AT_LIGHT_INTENSITY        = 2.19;  // L_Element_10_home_sceneintensity
const float AT_WIGGLE_SPEED           = 0.7;   // INPUT_Element_1_work_scenewiggle_speed
const float AT_LUMINOSITY_THRESHOLD   = 0.0;   // BloomLuminosity luminosityThreshold
const float AT_LERP_SPEED             = 0.1;   // CAMERA_Element_1_HomelerpSpeed

// ---- helpers ----
float seg(vec2 p, vec2 a, vec2 b, float w) {
  vec2 ab = b - a;
  vec2 ap = p - a;
  float t  = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
  float d  = length(ap - t * ab);
  return smoothstep(w, w * 0.4, d);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;
  vec2 p  = uv * 2.0 - 1.0;          // [-1,1]

  float strokeW = 0.045;
  float total   = 0.0;

  // Build zigzag path from top to bottom
  float steps = u_zigzagCount;
  float dy    = 2.0 / steps;

  // Animated phase offset — speed driven by AT_WIGGLE_SPEED
  float phase = sin(u_time * 2.5 * AT_WIGGLE_SPEED) * 0.15;

  for (float i = 0.0; i < 32.0; i++) {
    if (i >= steps) break;

    float t0 = -1.0 + i       * dy;
    float t1 = -1.0 + (i+1.0) * dy;

    float side0 = (mod(i,       2.0) < 1.0 ? 1.0 : -1.0);
    float side1 = (mod(i + 1.0, 2.0) < 1.0 ? 1.0 : -1.0);

    vec2 a = vec2(side0 * u_amplitude + phase, t0);
    vec2 b = vec2(side1 * u_amplitude + phase, t1);

    total = max(total, seg(p, a, b, strokeW));
  }

  // Global bloom pass — radius/strength from AT UIL globalbloom params
  float glowGlobal = 0.0;
  float globalGlowW = strokeW * (3.5 * AT_GLOBAL_BLOOM_RADIUS / AT_BLOOM_RADIUS);
  for (float i = 0.0; i < 32.0; i++) {
    if (i >= steps) break;
    float t0   = -1.0 + i * dy;
    float t1   = -1.0 + (i+1.0) * dy;
    float s0   = (mod(i,       2.0) < 1.0 ? 1.0 : -1.0);
    float s1   = (mod(i + 1.0, 2.0) < 1.0 ? 1.0 : -1.0);
    vec2 a     = vec2(s0 * u_amplitude + phase, t0);
    vec2 b     = vec2(s1 * u_amplitude + phase, t1);
    glowGlobal = max(glowGlobal, seg(p, a, b, globalGlowW) * AT_GLOBAL_BLOOM_STRENGTH);
  }

  // Home bloom pass — radius/strength from AT UIL homebloom params
  float glowHome = 0.0;
  float homeGlowW = strokeW * (5.0 * AT_HOME_BLOOM_RADIUS / AT_BLOOM_RADIUS);
  for (float i = 0.0; i < 32.0; i++) {
    if (i >= steps) break;
    float t0   = -1.0 + i * dy;
    float t1   = -1.0 + (i+1.0) * dy;
    float s0   = (mod(i,       2.0) < 1.0 ? 1.0 : -1.0);
    float s1   = (mod(i + 1.0, 2.0) < 1.0 ? 1.0 : -1.0);
    vec2 a     = vec2(s0 * u_amplitude + phase, t0);
    vec2 b     = vec2(s1 * u_amplitude + phase, t1);
    glowHome   = max(glowHome, seg(p, a, b, homeGlowW) * AT_HOME_BLOOM_STRENGTH);
  }

  // Luminosity gate — pixels below threshold don't contribute bloom (AT_LUMINOSITY_THRESHOLD = 0)
  float lum      = dot(u_fillColor, vec3(0.2126, 0.7152, 0.0722));
  float lumGate  = step(AT_LUMINOSITY_THRESHOLD, lum);

  // Combine: core stroke + bloom passes scaled by AT_BLOOM_INTENSITY and AT_LIGHT_INTENSITY
  float bloomSum = (glowGlobal + glowHome) * lumGate * AT_BLOOM_INTENSITY * (AT_LIGHT_INTENSITY / 2.19);
  float alpha    = clamp(total + bloomSum, 0.0, 1.0);

  gl_FragColor = vec4(u_fillColor, alpha * u_opacity);
}
