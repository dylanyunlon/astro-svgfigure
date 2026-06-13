precision mediump float;

uniform vec4  u_bbox;         // x, y, width, height in canvas coords
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform vec2  u_resolution;

uniform float u_numRays;
uniform float u_pupilRadius;
uniform float u_focalIntensity;
uniform float u_time;

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

  float alpha = clamp(sclera * (iris + rays) + pupil, 0.0, 1.0);

  gl_FragColor = vec4(u_fillColor, alpha * u_opacity);
}
