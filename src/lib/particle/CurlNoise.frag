#version 300 es
/**
 * CurlNoise.frag — GPGPU curl noise particle update pass
 *
 * Active Theory 粒子系统核心驱动力。每帧并行更新每个粒子（像素）的
 * position + velocity，curl noise 作为无散度向量场驱动流体感运动。
 *
 * 纹理布局 (RGBA float):
 *   tPosition : RGB = world position (x,y,z),  A = life [0,1]
 *   tVelocity : RGB = velocity (vx,vy,vz),     A = speed scalar
 *
 * 参数来源: channels/physics/at_uil_params.json
 *   am_ProtonAntimatter_P_Element_0_WorkDetailParticles uSCurlNoiseSpeed = 5
 *   am_ProtonAntimatter_P_Element_0_TubesInteraction    uCurlNoiseSpeed  = 5
 *   am_ProtonAntimatter_P_Element_4_work_page           uCurlNoiseSpeed  = 10
 *   am_ProtonAntimatter_P_Element_0_WorkDetailParticles uSCurlNoiseScale = 2
 *
 * Uniforms (外部设置，不在此处声明默认值):
 *   uCurlNoiseSpeed  — primary curl field strength   (AT: 5 / 10)
 *   uSCurlNoiseSpeed — spline-curl field strength    (AT: 5)
 *   uCurlNoiseScale  — spatial frequency of field
 *   uSCurlNoiseScale — spline-curl spatial frequency (AT: 2)
 *   uCurlTimeScale   — temporal drift speed
 *   uTime            — elapsed time (seconds)
 *   uDelta           — frame delta (HZ equivalent)
 *   tPosition        — current particle position texture
 *   tVelocity        — current particle velocity texture
 */

precision highp float;
precision highp sampler2D;

// ── Uniforms ─────────────────────────────────────────────────────────────────

uniform sampler2D tPosition;
uniform sampler2D tVelocity;

/** AT: uCurlNoiseSpeed — primary field (5 for WorkDetail/Tubes, 10 for work_page) */
uniform float uCurlNoiseSpeed;

/** AT: uSCurlNoiseSpeed — spline curl field strength (5) */
uniform float uSCurlNoiseSpeed;

/** AT: uCurlNoiseScale — spatial frequency */
uniform float uCurlNoiseScale;

/** AT: uSCurlNoiseScale — spline curl spatial frequency (2) */
uniform float uSCurlNoiseScale;

/** AT: uCurlTimeScale — temporal drift */
uniform float uCurlTimeScale;

/** AT: uSCurlTimeScale — spline curl temporal drift (2) */
uniform float uSCurlTimeScale;

uniform float uTime;
uniform float uDelta;   /* HZ: target-normalised delta, ~1.0 at 60fps */

in  vec2 vUv;
out vec4 fragColor;

// ── Noise primitives ──────────────────────────────────────────────────────────

/** Classic 3-D hash — identical to AT's curl.glsl */
vec3 hash33(vec3 p) {
  p = fract(p * vec3(443.897, 441.423, 437.195));
  p += dot(p, p.yxz + 19.19);
  return fract((p.xxy + p.yxx) * p.zyx);
}

/** Gradient noise (Perlin-style) over a 3-D domain */
float gradNoise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);           /* smoothstep */

  return mix(
    mix(
      mix(dot(hash33(i + vec3(0,0,0)) * 2.0 - 1.0, f - vec3(0,0,0)),
          dot(hash33(i + vec3(1,0,0)) * 2.0 - 1.0, f - vec3(1,0,0)), u.x),
      mix(dot(hash33(i + vec3(0,1,0)) * 2.0 - 1.0, f - vec3(0,1,0)),
          dot(hash33(i + vec3(1,1,0)) * 2.0 - 1.0, f - vec3(1,1,0)), u.x),
      u.y),
    mix(
      mix(dot(hash33(i + vec3(0,0,1)) * 2.0 - 1.0, f - vec3(0,0,1)),
          dot(hash33(i + vec3(1,0,1)) * 2.0 - 1.0, f - vec3(1,0,1)), u.x),
      mix(dot(hash33(i + vec3(0,1,1)) * 2.0 - 1.0, f - vec3(0,1,1)),
          dot(hash33(i + vec3(1,1,1)) * 2.0 - 1.0, f - vec3(1,1,1)), u.x),
      u.y),
    u.z);
}

/**
 * curlNoise — curl of a noise potential field → divergence-free vector.
 * Matches AT's #require(curl.glsl) implementation.
 *
 * Approximates ∇×F by finite differences:
 *   curl.x = dFz/dy − dFy/dz
 *   curl.y = dFx/dz − dFz/dx
 *   curl.z = dFy/dx − dFx/dy
 */
vec3 curlNoise(vec3 p) {
  const float e = 0.0001;

  vec3 dx = vec3(e,   0.0, 0.0);
  vec3 dy = vec3(0.0, e,   0.0);
  vec3 dz = vec3(0.0, 0.0, e  );

  // Three independent noise potentials (x-,y-,z-components of F)
  float Fx_py = gradNoise(p + dy);  float Fx_my = gradNoise(p - dy);
  float Fx_pz = gradNoise(p + dz);  float Fx_mz = gradNoise(p - dz);

  float Fy_pz = gradNoise(p + dz + vec3(31.41, 0.0, 0.0));
  float Fy_mz = gradNoise(p - dz + vec3(31.41, 0.0, 0.0));
  float Fy_px = gradNoise(p + dx + vec3(31.41, 0.0, 0.0));
  float Fy_mx = gradNoise(p - dx + vec3(31.41, 0.0, 0.0));

  float Fz_px = gradNoise(p + dx + vec3(0.0, 27.18, 0.0));
  float Fz_mx = gradNoise(p - dx + vec3(0.0, 27.18, 0.0));
  float Fz_py = gradNoise(p + dy + vec3(0.0, 27.18, 0.0));
  float Fz_my = gradNoise(p - dy + vec3(0.0, 27.18, 0.0));

  vec3 curl;
  curl.x = ((Fz_py - Fz_my) - (Fy_pz - Fy_mz)) / (2.0 * e);
  curl.y = ((Fx_pz - Fx_mz) - (Fz_px - Fz_mx)) / (2.0 * e);
  curl.z = ((Fy_px - Fy_mx) - (Fx_py - Fx_my)) / (2.0 * e);
  return curl;
}

// ── Main ──────────────────────────────────────────────────────────────────────

void main() {
  vec4 posLife = texture(tPosition, vUv);
  vec4 velSpd  = texture(tVelocity, vUv);

  vec3 pos  = posLife.xyz;
  float life = posLife.w;

  vec3 vel  = velSpd.xyz;

  // ── Primary curl field (AT: uCurlNoiseSpeed=5/10) ────────────────────────
  vec3 curlSample = pos * uCurlNoiseScale * 0.1
                  + vec3(uTime * uCurlTimeScale * 0.1);
  vec3 curlForce  = curlNoise(curlSample);
  pos += curlForce * uCurlNoiseSpeed * 0.01 * uDelta;

  // ── Secondary spline-curl field (AT: uSCurlNoiseSpeed=5) ─────────────────
  // Applied when particles follow splines (SplineEmitter handles routing)
  vec3 sCurlSample = pos * uSCurlNoiseScale * 0.1
                   + vec3(uTime * uSCurlTimeScale * 0.1);
  vec3 sCurlForce  = curlNoise(sCurlSample);
  pos += sCurlForce * uSCurlNoiseSpeed * 0.005 * uDelta;

  // ── Velocity damping ──────────────────────────────────────────────────────
  vel  = curlForce * uCurlNoiseSpeed * 0.01;

  // ── Life decay ────────────────────────────────────────────────────────────
  life -= 0.001 * uDelta;
  if (life <= 0.0) {
    // Respawn at origin jitter — SplineEmitter overrides this
    pos  = (hash33(vec3(vUv, uTime)) - 0.5) * 2.0;
    life = 1.0;
  }

  fragColor = vec4(pos, life);
  // velocity written to separate attachment via MRT if needed
  // velocity magnitude stored in alpha
  fragColor.w = life; // overwrite alpha with life (pos.w)
  // vel available for MRT second attachment
}
