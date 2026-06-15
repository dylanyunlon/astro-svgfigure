precision mediump float;

// ─── AT UIL params applied from channels/physics/xiaodi_options_table.json ["cil-eye"] ───
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

void main() {
  // Normalize fragment to bbox-local UV [0,1]
  vec2 fragCoord = gl_FragCoord.xy;
  vec2 uv = (fragCoord - u_bbox.xy) / u_bbox.zw;

  // Map to [-1,1] centered space
  vec2 p = uv * 2.0 - 1.0;

  float dist  = length(p);
  float angle = atan(p.y, p.x);

  // --- Pupil ---
  float pupil = 1.0 - smoothstep(u_pupilRadius - 0.02, u_pupilRadius + 0.02, dist);

  // --- Iris ring ---
  float iris = smoothstep(u_pupilRadius + 0.02, u_pupilRadius + 0.08, dist)
             * (1.0 - smoothstep(0.85, 1.0, dist));

  // --- Radial rays ---
  float halfStep  = 3.14159265 / u_numRays;
  float rayAngle  = mod(angle + u_time * 0.3, halfStep * 2.0) - halfStep;
  float rayMask   = smoothstep(0.07, 0.0, abs(rayAngle));
  float rayFade   = smoothstep(1.0, u_pupilRadius + 0.1, dist)
                  * smoothstep(u_pupilRadius, u_pupilRadius + 0.12, dist);
  float rays      = rayMask * rayFade * u_focalIntensity;

  // --- Sclera (outer white ellipse halo) ---
  float sclera = smoothstep(1.05, 0.88, dist);

  // --- AT ambient lighting (L_Element_11 + VolumetricLight) ---
  // Radial fall-off for ambient: bright at centre, fades at edge
  float ambientFalloff = 1.0 - smoothstep(0.0, 1.2, dist);
  vec3  ambientContrib = u_ambientColor * u_ambientIntensity * u_lightExposure * ambientFalloff;

  // --- AT bloom glow ring ---
  // A soft additive glow at the iris boundary, scaled by bloomStrength / bloomRadius
  float bloomRing = exp(-pow((dist - (u_pupilRadius + 0.15)) / max(u_bloomRadius * 0.18, 0.01), 2.0));
  float bloom     = bloomRing * u_bloomStrength * 0.35;

  // --- AT shadow attenuation ---
  // Simulate a shadow that darkens the outer sclera region using far-plane distance
  float shadowNorm   = clamp(dist / (u_shadowFar * 0.025), 0.0, 1.0);
  float shadowFactor = 1.0 - shadowNorm * (1.0 - u_shadowBias * 100.0);

  float alpha = clamp(sclera * (iris + rays) + pupil, 0.0, 1.0);

  // Combine base color with AT ambient contribution
  vec3 finalColor = u_fillColor + ambientContrib * (iris + bloom) * alpha;

  // Apply bloom additive on top
  finalColor += u_fillColor * bloom;

  // Apply shadow attenuation
  finalColor *= shadowFactor;

  gl_FragColor = vec4(finalColor, alpha * u_opacity);
}
