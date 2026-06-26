#version 300 es
precision mediump float;
out vec4 fragColor;

// ─── AT UIL params applied from channels/physics/xiaodi_options_table.json [\"cil-eye\"] ───
//
// bloom / glow (UnrealBloomComposite / homebloom):
//   UnrealBloomComposite/UnrealBloomComposite/homebloom/bloomStrength  = 1.2
//   UnrealBloomComposite/UnrealBloomComposite/homebloom/bloomRadius    = 1.0
//   UnrealBloomComposite/UnrealBloomComposite/homebloom/bloomTintColor = #ffffff
//   UnrealBloomComposite_shaderVariants_homebloomStrength              = 0.6
//   UnrealBloomComposite_shaderVariants_homebloomRadius                = 0.8
//   UnrealBloomLuminosity/UnrealBloomLuminosity/homebloom/luminosityThreshold = 0.0
//
// lighting (HomeAlleyShader / L_Element_11_home_scene):
//   HomeAlleyShader uLight      = [2.61, 0.29, 0.57, 0.0]
//   HomeAlleyShader uPhong      = [1.82, 0.71]
//   HomeAlleyShader uPhongColor = #d600ff
//   L_Element_11_home_scene intensity = 3.44
//   L_Element_11_home_scene color     = #0bed90
//   L_Element_10_home_scene intensity = 2.19
//   VolumetricLight_home fExposure    = 0.86   fDensity = 0.22
//
// shadow (SHADOW_Element_9_home_scene):
//   SHADOW_Element_9_home_scene far    = 40
//   SHADOW_Element_9_home_scene size   = 1024
//   SHADOW_Element_9_home_scene static = true
//   SHADOW_Element_9_home_scene position = [0, 6.51, 0]
//   SHADOW_Element_9_home_scene target   = [0, 0, 0]
// ──────────────────────────────────────────────────────────────────────────────────────────

uniform vec4  u_bbox;         // x, y, width, height in canvas coords
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_numRays;
uniform float u_pupilRadius;
uniform float u_focalIntensity;
uniform float u_time;

// AT bloom uniforms (defaults from UIL cil-eye entry)
uniform float u_bloomStrength;   // default 1.2  (homebloom/bloomStrength)
uniform float u_bloomRadius;     // default 1.0  (homebloom/bloomRadius)

// AT ambient light uniforms (from L_Element_11_home_scene + VolumetricLight)
uniform float u_ambientIntensity; // default 3.44 (L_Element_11 intensity)
uniform vec3  u_ambientColor;     // default #0bed90 → (0.047, 0.929, 0.565)
uniform float u_lightExposure;    // default 0.86  (VolumetricLight fExposure)

// AT shadow uniforms (from SHADOW_Element_9_home_scene)
uniform float u_shadowFar;     // default 40.0
uniform float u_shadowBias;    // default 0.001 (derived from shadow size 1024)

// ── lygia/sdf/circleSDF.glsl (inlined) ──────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
// Returns a circle-shaped SDF.  circleSDF(vec2 st) → distance in [0,1] space.
// Centered at 0.5; result * 2 == diameter-normalised distance.
#ifndef FNC_CIRCLESDF
#define FNC_CIRCLESDF
float circleSDF(in vec2 v) {
    v -= 0.5;
    return length(v) * 2.0;
}
#endif
// ── end lygia circleSDF ──────────────────────────────────────────────────────

void main() {
  // Normalize fragment to bbox-local UV [0,1]
  vec2 fragCoord = gl_FragCoord.xy;
  vec2 uv = (fragCoord - u_bbox.xy) / u_bbox.zw;

  // circleSDF works in [0,1] UV space; result is 0 at centre, 1 at edge of unit circle.
  // We map it to [-1,1] range for the legacy distance variable used below.
  float dist = circleSDF(uv);   // [0, ~1.41]

  // Angle still computed from centred coordinates
  vec2 p = uv * 2.0 - 1.0;
  float angle = atan(p.y, p.x);

  // --- Pupil (using circleSDF result) ---
  // u_pupilRadius is expressed in the original [-1,1] space → convert to circleSDF scale (*0.5)
  float pupilR = u_pupilRadius * 0.5;
  float pupil = 1.0 - smoothstep(pupilR - 0.01, pupilR + 0.01, dist);

  // --- Iris ring ---
  float irisInner = (u_pupilRadius + 0.02) * 0.5;
  float irisOuter = (u_pupilRadius + 0.08) * 0.5;
  float iris = smoothstep(irisInner, irisOuter, dist)
             * (1.0 - smoothstep(0.425, 0.5, dist));

  // --- Radial rays ---
  float halfStep = 3.14159265 / u_numRays;
  float rayAngle = mod(angle + u_time * 0.3, halfStep * 2.0) - halfStep;
  float rayMask  = smoothstep(0.07, 0.0, abs(rayAngle));
  float rayFade  = smoothstep(0.5, irisInner + 0.06, dist)
                 * smoothstep(pupilR, pupilR + 0.06, dist);
  float rays     = rayMask * rayFade * u_focalIntensity;

  // --- Sclera (outer white ellipse halo) via circleSDF ---
  float sclera = smoothstep(0.525, 0.44, dist);

  // --- AT ambient lighting (L_Element_11 + VolumetricLight) ---
  float ambientFalloff = 1.0 - smoothstep(0.0, 0.6, dist);
  vec3  ambientContrib = u_ambientColor * u_ambientIntensity * u_lightExposure * ambientFalloff;

  // --- AT bloom glow ring ---
  float bloomCenter = (u_pupilRadius + 0.15) * 0.5;
  float bloomRing = exp(-pow((dist - bloomCenter) / max(u_bloomRadius * 0.09, 0.005), 2.0));
  float bloom     = bloomRing * u_bloomStrength * 0.35;

  // --- AT shadow attenuation ---
  float shadowNorm   = clamp(dist / (u_shadowFar * 0.0125), 0.0, 1.0);
  float shadowFactor = 1.0 - shadowNorm * (1.0 - u_shadowBias * 100.0);

  float alpha = clamp(sclera * (iris + rays) + pupil, 0.0, 1.0);

  vec3 finalColor = u_fillColor + ambientContrib * (iris + bloom) * alpha;
  finalColor += u_fillColor * bloom;
  finalColor *= shadowFactor;

  fragColor = vec4(finalColor, alpha * u_opacity);
}
