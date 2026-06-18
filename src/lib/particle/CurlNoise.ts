/**
 * CurlNoise.ts — Active Theory analytic curl noise
 *
 * Ported directly from Active Theory's curl.glsl
 * (https://activetheory.net/assets/shaders/compiled.vs)
 *
 * Three potential functions (potential1/2/3) each built from 8 sin terms,
 * with 6 analytic partial derivatives (cos) for the curl. The analytic
 * approach uses 36 trig calls vs 96 for the numerical approximation,
 * making it ~3× faster.
 *
 * Curl definition:
 *   curl.x = dP3dY − dP2dZ
 *   curl.y = dP1dZ − dP3dX
 *   curl.z = dP2dX − dP1dY
 *
 * Reference: http://weber.itn.liu.se/~stegu/TNM084-2019/bridson-siggraph2007-curlnoise.pdf
 */

// ── GLSL version for shader injection ─────────────────────────────────────────

export const CURL_NOISE_GLSL: string = /* glsl */ `
float potential1(vec3 v) {
    float noise = 0.0;
    noise += sin(v.x * 1.8 + v.z * 3.) + sin(v.x * 4.8 + v.z * 4.5) + sin(v.x * -7.0 + v.z * 1.2) + sin(v.x * -5.0 + v.z * 2.13);
    noise += sin(v.y * -0.48 + v.z * 5.4) + sin(v.y * 2.56 + v.z * 5.4) + sin(v.y * 4.16 + v.z * 2.4) + sin(v.y * -4.16 + v.z * 1.35);
    return noise;
}

float potential2(vec3 v) {
    float noise = 0.0;
    noise += sin(v.y * 1.8 + v.x * 3. - 2.82) + sin(v.y * 4.8 + v.x * 4.5 + 74.37) + sin(v.y * -7.0 + v.x * 1.2 - 256.72) + sin(v.y * -5.0 + v.x * 2.13 - 207.683);
    noise += sin(v.z * -0.48 + v.x * 5.4 - 125.796) + sin(v.z * 2.56 + v.x * 5.4 + 17.692) + sin(v.z * 4.16 + v.x * 2.4 + 150.512) + sin(v.z * -4.16 + v.x * 1.35 - 222.137);
    return noise;
}

float potential3(vec3 v) {
    float noise = 0.0;
    noise += sin(v.z * 1.8 + v.y * 3. - 194.58) + sin(v.z * 4.8 + v.y * 4.5 - 83.13) + sin(v.z * -7.0 + v.y * 1.2 - 845.2) + sin(v.z * -5.0 + v.y * 2.13 - 762.185);
    noise += sin(v.x * -0.48 + v.y * 5.4 - 707.916) + sin(v.x * 2.56 + v.y * 5.4 + -482.348) + sin(v.x * 4.16 + v.y * 2.4 + 9.872) + sin(v.x * -4.16 + v.y * 1.35 - 476.747);
    return noise;
}

vec3 snoiseVec3(vec3 x) {
    float s  = potential1(x);
    float s1 = potential2(x);
    float s2 = potential3(x);
    return vec3(s, s1, s2);
}

float dP3dY(vec3 v) {
    float noise = 0.0;
    noise += 3. * cos(v.z * 1.8 + v.y * 3. - 194.58) + 4.5 * cos(v.z * 4.8 + v.y * 4.5 - 83.13) + 1.2 * cos(v.z * -7.0 + v.y * 1.2 - 845.2) + 2.13 * cos(v.z * -5.0 + v.y * 2.13 - 762.185);
    noise += 5.4 * cos(v.x * -0.48 + v.y * 5.4 - 707.916) + 5.4 * cos(v.x * 2.56 + v.y * 5.4 + -482.348) + 2.4 * cos(v.x * 4.16 + v.y * 2.4 + 9.872) + 1.35 * cos(v.x * -4.16 + v.y * 1.35 - 476.747);
    return noise;
}

float dP2dZ(vec3 v) {
    return -0.48 * cos(v.z * -0.48 + v.x * 5.4 - 125.796) + 2.56 * cos(v.z * 2.56 + v.x * 5.4 + 17.692) + 4.16 * cos(v.z * 4.16 + v.x * 2.4 + 150.512) - 4.16 * cos(v.z * -4.16 + v.x * 1.35 - 222.137);
}

float dP1dZ(vec3 v) {
    float noise = 0.0;
    noise += 3. * cos(v.x * 1.8 + v.z * 3.) + 4.5 * cos(v.x * 4.8 + v.z * 4.5) + 1.2 * cos(v.x * -7.0 + v.z * 1.2) + 2.13 * cos(v.x * -5.0 + v.z * 2.13);
    noise += 5.4 * cos(v.y * -0.48 + v.z * 5.4) + 5.4 * cos(v.y * 2.56 + v.z * 5.4) + 2.4 * cos(v.y * 4.16 + v.z * 2.4) + 1.35 * cos(v.y * -4.16 + v.z * 1.35);
    return noise;
}

float dP3dX(vec3 v) {
    return -0.48 * cos(v.x * -0.48 + v.y * 5.4 - 707.916) + 2.56 * cos(v.x * 2.56 + v.y * 5.4 + -482.348) + 4.16 * cos(v.x * 4.16 + v.y * 2.4 + 9.872) - 4.16 * cos(v.x * -4.16 + v.y * 1.35 - 476.747);
}

float dP2dX(vec3 v) {
    float noise = 0.0;
    noise += 3. * cos(v.y * 1.8 + v.x * 3. - 2.82) + 4.5 * cos(v.y * 4.8 + v.x * 4.5 + 74.37) + 1.2 * cos(v.y * -7.0 + v.x * 1.2 - 256.72) + 2.13 * cos(v.y * -5.0 + v.x * 2.13 - 207.683);
    noise += 5.4 * cos(v.z * -0.48 + v.x * 5.4 - 125.796) + 5.4 * cos(v.z * 2.56 + v.x * 5.4 + 17.692) + 2.4 * cos(v.z * 4.16 + v.x * 2.4 + 150.512) + 1.35 * cos(v.z * -4.16 + v.x * 1.35 - 222.137);
    return noise;
}

float dP1dY(vec3 v) {
    return -0.48 * cos(v.y * -0.48 + v.z * 5.4) + 2.56 * cos(v.y * 2.56 + v.z * 5.4) + 4.16 * cos(v.y * 4.16 + v.z * 2.4) - 4.16 * cos(v.y * -4.16 + v.z * 1.35);
}

vec3 curlNoise(vec3 p) {
    float x = dP3dY(p) - dP2dZ(p);
    float y = dP1dZ(p) - dP3dX(p);
    float z = dP2dX(p) - dP1dY(p);
    return normalize(vec3(x, y, z));
}
`;

// ── TypeScript CPU-side implementation ────────────────────────────────────────

// ── Potential functions (each 8 sin terms) ────────────────────────────────────

function potential1(x: number, y: number, z: number): number {
  let noise = 0.0;
  noise += Math.sin(x * 1.8 + z * 3.) + Math.sin(x * 4.8 + z * 4.5) + Math.sin(x * -7.0 + z * 1.2) + Math.sin(x * -5.0 + z * 2.13);
  noise += Math.sin(y * -0.48 + z * 5.4) + Math.sin(y * 2.56 + z * 5.4) + Math.sin(y * 4.16 + z * 2.4) + Math.sin(y * -4.16 + z * 1.35);
  return noise;
}

function potential2(x: number, y: number, z: number): number {
  let noise = 0.0;
  noise += Math.sin(y * 1.8 + x * 3. - 2.82) + Math.sin(y * 4.8 + x * 4.5 + 74.37) + Math.sin(y * -7.0 + x * 1.2 - 256.72) + Math.sin(y * -5.0 + x * 2.13 - 207.683);
  noise += Math.sin(z * -0.48 + x * 5.4 - 125.796) + Math.sin(z * 2.56 + x * 5.4 + 17.692) + Math.sin(z * 4.16 + x * 2.4 + 150.512) + Math.sin(z * -4.16 + x * 1.35 - 222.137);
  return noise;
}

function potential3(x: number, y: number, z: number): number {
  let noise = 0.0;
  noise += Math.sin(z * 1.8 + y * 3. - 194.58) + Math.sin(z * 4.8 + y * 4.5 - 83.13) + Math.sin(z * -7.0 + y * 1.2 - 845.2) + Math.sin(z * -5.0 + y * 2.13 - 762.185);
  noise += Math.sin(x * -0.48 + y * 5.4 - 707.916) + Math.sin(x * 2.56 + y * 5.4 + -482.348) + Math.sin(x * 4.16 + y * 2.4 + 9.872) + Math.sin(x * -4.16 + y * 1.35 - 476.747);
  return noise;
}

// ── Analytic partial derivatives (each uses cos) ──────────────────────────────

function dP3dY(x: number, y: number, z: number): number {
  let noise = 0.0;
  noise += 3. * Math.cos(z * 1.8 + y * 3. - 194.58) + 4.5 * Math.cos(z * 4.8 + y * 4.5 - 83.13) + 1.2 * Math.cos(z * -7.0 + y * 1.2 - 845.2) + 2.13 * Math.cos(z * -5.0 + y * 2.13 - 762.185);
  noise += 5.4 * Math.cos(x * -0.48 + y * 5.4 - 707.916) + 5.4 * Math.cos(x * 2.56 + y * 5.4 + -482.348) + 2.4 * Math.cos(x * 4.16 + y * 2.4 + 9.872) + 1.35 * Math.cos(x * -4.16 + y * 1.35 - 476.747);
  return noise;
}

function dP2dZ(x: number, y: number, z: number): number {
  return -0.48 * Math.cos(z * -0.48 + x * 5.4 - 125.796) + 2.56 * Math.cos(z * 2.56 + x * 5.4 + 17.692) + 4.16 * Math.cos(z * 4.16 + x * 2.4 + 150.512) - 4.16 * Math.cos(z * -4.16 + x * 1.35 - 222.137);
}

function dP1dZ(x: number, y: number, z: number): number {
  let noise = 0.0;
  noise += 3. * Math.cos(x * 1.8 + z * 3.) + 4.5 * Math.cos(x * 4.8 + z * 4.5) + 1.2 * Math.cos(x * -7.0 + z * 1.2) + 2.13 * Math.cos(x * -5.0 + z * 2.13);
  noise += 5.4 * Math.cos(y * -0.48 + z * 5.4) + 5.4 * Math.cos(y * 2.56 + z * 5.4) + 2.4 * Math.cos(y * 4.16 + z * 2.4) + 1.35 * Math.cos(y * -4.16 + z * 1.35);
  return noise;
}

function dP3dX(x: number, y: number, z: number): number {
  return -0.48 * Math.cos(x * -0.48 + y * 5.4 - 707.916) + 2.56 * Math.cos(x * 2.56 + y * 5.4 + -482.348) + 4.16 * Math.cos(x * 4.16 + y * 2.4 + 9.872) - 4.16 * Math.cos(x * -4.16 + y * 1.35 - 476.747);
}

function dP2dX(x: number, y: number, z: number): number {
  let noise = 0.0;
  noise += 3. * Math.cos(y * 1.8 + x * 3. - 2.82) + 4.5 * Math.cos(y * 4.8 + x * 4.5 + 74.37) + 1.2 * Math.cos(y * -7.0 + x * 1.2 - 256.72) + 2.13 * Math.cos(y * -5.0 + x * 2.13 - 207.683);
  noise += 5.4 * Math.cos(z * -0.48 + x * 5.4 - 125.796) + 5.4 * Math.cos(z * 2.56 + x * 5.4 + 17.692) + 2.4 * Math.cos(z * 4.16 + x * 2.4 + 150.512) + 1.35 * Math.cos(z * -4.16 + x * 1.35 - 222.137);
  return noise;
}

function dP1dY(x: number, y: number, z: number): number {
  return -0.48 * Math.cos(y * -0.48 + z * 5.4) + 2.56 * Math.cos(y * 2.56 + z * 5.4) + 4.16 * Math.cos(y * 4.16 + z * 2.4) - 4.16 * Math.cos(y * -4.16 + z * 1.35);
}

// ── Curl noise (analytic) ─────────────────────────────────────────────────────

/**
 * Compute a divergence-free 3D curl noise vector at the given position.
 *
 * Ported from Active Theory's curl.glsl. Uses analytic derivatives of
 * three sinusoidal potential functions — 36 trig calls vs 96 for the
 * numerical (finite-difference) approach, ~3× faster.
 *
 * curl.x = dP3/dY − dP2/dZ
 * curl.y = dP1/dZ − dP3/dX
 * curl.z = dP2/dX − dP1/dY
 *
 * The result is normalized to unit length (matching the GLSL version).
 *
 * @param x - World-space x coordinate
 * @param y - World-space y coordinate
 * @param z - World-space z coordinate
 * @returns [curl_x, curl_y, curl_z] normalized divergence-free vector
 *
 * @example
 * ```ts
 * import { curlNoise3D } from '$lib/particle/CurlNoise';
 *
 * const [vx, vy, vz] = curlNoise3D(
 *   pos.x * scale + time * drift,
 *   pos.y * scale + time * drift,
 *   pos.z * scale + time * drift,
 * );
 * pos.x += vx * speed * delta;
 * pos.y += vy * speed * delta;
 * pos.z += vz * speed * delta;
 * ```
 */
export function curlNoise3D(
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  const cx = dP3dY(x, y, z) - dP2dZ(x, y, z);
  const cy = dP1dZ(x, y, z) - dP3dX(x, y, z);
  const cz = dP2dX(x, y, z) - dP1dY(x, y, z);

  const len = Math.sqrt(cx * cx + cy * cy + cz * cz);
  if (len === 0) return [0, 0, 0];
  const inv = 1.0 / len;

  return [cx * inv, cy * inv, cz * inv];
}
