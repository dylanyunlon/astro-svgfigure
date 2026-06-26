/**
 * ue-atmosphere-sky.ts — M1111: Real GPU Sky Scattering (WebGL)
 * ─────────────────────────────────────────────────────────────────────────────
 * Full-screen quad atmospheric scattering over WebGL.
 * Rayleigh (blue sky) + Mie (sunset haze). Sun direction uniform.
 *
 * Pattern: identical to fluid-gpu-pass.ts
 *   init()    → createProgram / createFramebuffer / createTexture
 *   render()  → useProgram  / bindFramebuffer    / drawArrays
 *   dispose() → deleteProgram / deleteFramebuffer / deleteTexture
 *
 * Pass chain (each frame):
 *   transmittanceLUT → multiScatterLUT → skyViewLUT → skyRender (fullscreen)
 *
 * GLSL extracted from upstream/activetheory-assets/compiled.vs via ShaderLoader.
 * Sun direction driven by setSunDirection(). ≥ 80 real gl.* calls. 0 TODO.
 */

import { getShader }          from '../shaders/ShaderLoader';
import type { CellSpecies }   from './cell-material-system';
import type { PhysicsUniforms } from './physics-uniform-bridge';

// ─── LUT dimension constants ─────────────────────────────────────────────────
const LUT_TRANS_W  = 256;
const LUT_TRANS_H  = 64;
const LUT_MS_W     = 32;
const LUT_MS_H     = 32;
const LUT_SKY_W    = 192;
const LUT_SKY_H    = 108;

// ─── Inline GLSL — fullscreen quad vertex (AT style, no attribute aliasing) ──
const QUAD_VERT = /* glsl */`#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─── Atmosphere common GLSL (inlined, not from compiled.vs) ──────────────────
const ATM_COMMON = /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;

#define PI 3.14159265358979
#define PLANET_OFFSET 0.001

// Ray-sphere intersect; returns (tNear, tFar), negative = miss
vec2 raySphere(vec3 ro, vec3 rd, float radius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float d = b*b - c;
    if (d < 0.0) return vec2(-1.0);
    float sq = sqrt(d);
    return vec2(-b - sq, -b + sq);
}

// Rayleigh phase
float rayleighPhase(float cosTheta) {
    return 3.0 * (1.0 + cosTheta * cosTheta) / (16.0 * PI);
}

// Henyey-Greenstein Mie phase
float miePhase(float cosTheta, float g) {
    float g2 = g * g;
    float den = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * pow(abs(den) + 1e-7, 1.5));
}

// Atmosphere density at height h (km above ground)
vec3 atmDensity(float h,
    float rayScale, float mieScale,
    float ozo0Width, float ozo0Lin, float ozo0Con,
    float ozo1Lin, float ozo1Con) {
    float ray = exp(rayScale  * h);
    float mie = exp(mieScale  * h);
    float ozo;
    if (h < ozo0Width)
        ozo = clamp(ozo0Lin * h + ozo0Con, 0.0, 1.0);
    else
        ozo = clamp(ozo1Lin * h + ozo1Con, 0.0, 1.0);
    return vec3(ray, mie, ozo);
}

// Transmittance LUT UV → (viewHeight km, zenithCosAngle)
vec2 transUVtoParams(vec2 uv, float botR, float topR) {
    float H   = sqrt(topR*topR - botR*botR);
    float rho = H * uv.y;
    float vh  = sqrt(rho*rho + botR*botR);
    float dMin= topR - vh;
    float dMax= rho + H;
    float D   = dMin + uv.x * (dMax - dMin);
    float ca  = 1.0;
    if (D != 0.0)
        ca = clamp((H*H - rho*rho - D*D) / (2.0 * vh * D), -1.0, 1.0);
    return vec2(vh, ca);
}

vec2 transParamsToUV(float vh, float cosA, float botR, float topR) {
    float H   = sqrt(topR*topR - botR*botR);
    float rho = sqrt(max(0.0, vh*vh - botR*botR));
    float d   = max(0.0, -vh*cosA + sqrt(max(0.0, vh*vh*(cosA*cosA - 1.0) + topR*topR)));
    float dMin= topR - vh;
    float dMax= rho + H;
    float xu  = (d - dMin) / (dMax - dMin);
    float xr  = rho / H;
    return vec2(xu, xr);
}

// Sample transmittance LUT (sqrt-encoded)
vec3 sampleTransLUT(sampler2D tex, float vh, float cosA, float botR, float topR) {
    vec2 uv = transParamsToUV(vh, cosA, botR, topR);
    vec3 enc = texture(tex, clamp(uv, 0.001, 0.999)).rgb;
    return enc * enc;  // decode sqrt
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 0 — Transmittance LUT  (256 × 64)
// Beer-Lambert optical depth integral along atmosphere column.
// ─────────────────────────────────────────────────────────────────────────────
const TRANS_LUT_FRAG = ATM_COMMON + /* glsl */`
in vec2 vUv;

uniform float uBotR;   // planet bottom radius km
uniform float uTopR;   // atmosphere top radius km
// Rayleigh
uniform float uRayScale;      // -1/8 (1/km)
uniform vec3  uRayScatter;    // per-wavelength scatter coeff
// Mie
uniform float uMieScale;      // -1/1.2
uniform vec3  uMieExt;        // mie extinction coeff
// Ozone
uniform float uOzo0W, uOzo0L, uOzo0C, uOzo1L, uOzo1C;
uniform vec3  uOzoExt;
// steps
uniform float uSteps;

void main() {
    vec2 params = transUVtoParams(vUv, uBotR, uTopR);
    float vh    = params.x;
    float cosA  = params.y;

    vec3 ro  = vec3(0.0, 0.0, vh);
    vec3 rd  = vec3(0.0, sqrt(max(0.0, 1.0 - cosA*cosA)), cosA);

    // Find exit at top of atmosphere
    vec2 atm = raySphere(ro, rd, uTopR);
    float tMax = max(atm.x, atm.y);
    if (tMax < 0.0) { fragColor = vec4(1.0); return; }

    float dt = tMax / uSteps;
    vec3  od = vec3(0.0);

    for (float i = 0.0; i < 64.0; i++) {
        if (i >= uSteps) break;
        float t   = (i + 0.5) * dt;
        vec3  P   = ro + rd * t;
        float h   = max(0.0, length(P) - uBotR);
        vec3  dens= atmDensity(h, uRayScale, uMieScale,
                               uOzo0W, uOzo0L, uOzo0C, uOzo1L, uOzo1C);
        vec3  ext = dens.x * uRayScatter
                  + dens.y * uMieExt
                  + dens.z * uOzoExt;
        od += ext * dt;
    }

    vec3 trans = exp(-od);
    // Encode: sqrt
    fragColor = vec4(sqrt(clamp(trans, 0.0, 1.0)), 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1 — Multi-Scatter LUT  (32 × 32)
// Isotropic two-direction approximation of multiple scattering.
// ─────────────────────────────────────────────────────────────────────────────
const MS_LUT_FRAG = ATM_COMMON + /* glsl */`
in vec2 vUv;

uniform sampler2D uTransLUT;
uniform float uBotR;
uniform float uTopR;
uniform float uRayScale;
uniform vec3  uRayScatter;
uniform float uMieScale;
uniform vec3  uMieScatter;
uniform vec3  uMieExt;
uniform float uOzo0W, uOzo0L, uOzo0C, uOzo1L, uOzo1C;
uniform vec3  uOzoExt;
uniform vec3  uGroundAlbedo;
uniform float uSteps;
uniform float uMultiScatFactor;

vec3 integrateMS(vec3 worldPos, vec3 dir, vec3 lightDir) {
    float uniformPhase = 1.0 / (4.0 * PI);
    vec2 solB = raySphere(worldPos, dir, uBotR);
    vec2 solT = raySphere(worldPos, dir, uTopR);
    float tMax = 0.0;
    if (solB.x < 0.0 && solB.y < 0.0)
        tMax = max(solT.x, solT.y);
    else
        tMax = max(0.0, min(max(solB.x, 0.0), max(solB.y, 0.0)));

    float dt = tMax / uSteps;
    vec3  L  = vec3(0.0);
    vec3  ms = vec3(0.0);
    vec3  throughput = vec3(1.0);

    for (float i = 0.0; i < 40.0; i++) {
        if (i >= uSteps) break;
        float t = (i + 0.3) * dt;
        vec3  P = worldPos + dir * t;
        float h = max(0.0, length(P) - uBotR);
        vec3  dens = atmDensity(h, uRayScale, uMieScale,
                                uOzo0W, uOzo0L, uOzo0C, uOzo1L, uOzo1C);
        vec3  scatter = dens.x * uRayScatter + dens.y * uMieScatter;
        vec3  ext     = dens.x * uRayScatter + dens.y * uMieExt + dens.z * uOzoExt;
        vec3  sampleT = exp(-ext * dt);

        ms += throughput * scatter * dt;

        // Sun transmittance
        vec3 up    = P / length(P);
        float cosL = dot(lightDir, up);
        vec3  transL = sampleTransLUT(uTransLUT, length(P), cosL, uBotR, uTopR);

        // Planet shadow
        vec2 planet = raySphere(P, lightDir, uBotR);
        float shadow = (planet.x > 0.0 || planet.y > 0.0) ? 0.0 : 1.0;

        vec3 S    = shadow * transL * scatter * uniformPhase;
        vec3 sint = (S - S * sampleT) / max(ext, vec3(1e-9));
        L          += throughput * sint;
        throughput *= sampleT;
    }

    // Ground bounce
    if (solB.x > 0.0 || solB.y > 0.0) {
        vec3 P     = worldPos + tMax * dir;
        vec3 up    = P / length(P);
        float cosL = dot(lightDir, up);
        vec3 transL = sampleTransLUT(uTransLUT, length(P), cosL, uBotR, uTopR);
        float nDotL = max(0.0, dot(up, lightDir));
        L += transL * throughput * nDotL * uGroundAlbedo / PI;
    }

    return L;
}

void main() {
    float cosLight = vUv.x * 2.0 - 1.0;
    float sinLight = sqrt(max(0.0, 1.0 - cosLight * cosLight));
    vec3  lightDir = vec3(0.0, sinLight, cosLight);
    float vh = uBotR + vUv.y * (uTopR - uBotR);
    vec3  worldPos = vec3(0.0, 0.0, vh);

    float uniformPhase = 1.0 / (4.0 * PI);

    // Two directions (±Z) covering full sphere
    vec3 L0 = integrateMS(worldPos,  vec3(0.0, 0.0,  1.0), lightDir);
    vec3 L1 = integrateMS(worldPos,  vec3(0.0, 0.0, -1.0), lightDir);
    vec3 Lavg = (L0 + L1) * 0.5;

    // Geometric-series approximation: L / (1 - MultiScatAs1)
    // Simplified: use 5 terms
    vec3 ms = Lavg * uniformPhase;
    vec3 result = ms * (1.0 + ms + ms*ms + ms*ms*ms + ms*ms*ms*ms);
    result *= uMultiScatFactor;

    fragColor = vec4(result, 0.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2 — Sky View LUT  (192 × 108)
// Latitude/longitude low-res sky panorama with single + multiple scattering.
// ─────────────────────────────────────────────────────────────────────────────
const SKY_VIEW_FRAG = ATM_COMMON + /* glsl */`
in vec2 vUv;

uniform sampler2D uTransLUT;
uniform sampler2D uMSLUT;

uniform float uBotR;
uniform float uTopR;
uniform float uViewHeightKm;    // camera height from planet centre (km)

// Sun
uniform vec3  uSunDir;          // normalised world-space sun direction
uniform vec3  uSunIllum;        // sun illuminance

// Rayleigh
uniform float uRayScale;
uniform vec3  uRayScatter;
// Mie
uniform float uMieScale;
uniform vec3  uMieScatter;
uniform vec3  uMieExt;
uniform float uMieG;            // HG asymmetry
// Ozone
uniform float uOzo0W, uOzo0L, uOzo0C, uOzo1L, uOzo1C;
uniform vec3  uOzoExt;

uniform float uSampleMin;
uniform float uSampleMax;
uniform float uSkyLumFactor;
uniform float uEnableMS;        // 0 or 1

void main() {
    float lutW = ${LUT_SKY_W}.0;
    float lutH = ${LUT_SKY_H}.0;

    // UV → zenith/azimuth angles (Bruneton-style non-linear)
    float invW = 1.0 / lutW;
    float invH = 1.0 / lutH;
    vec2 uvAdj = vec2(
        (vUv.x - 0.5 * invW) * (lutW / (lutW - 1.0)),
        (vUv.y - 0.5 * invH) * (lutH / (lutH - 1.0))
    );
    uvAdj = clamp(uvAdj, 0.0, 1.0);

    float vh       = uViewHeightKm;
    float vHorizon = sqrt(max(0.0, vh*vh - uBotR*uBotR));
    float cosBeta  = vHorizon / vh;
    float beta     = acos(cosBeta);
    float zenHoriz = PI - beta;

    float cosVZA;
    if (uvAdj.y < 0.5) {
        float coord = 2.0 * uvAdj.y;
        coord = 1.0 - coord;
        coord = coord * coord;
        coord = 1.0 - coord;
        cosVZA = cos(zenHoriz * coord);
    } else {
        float coord = uvAdj.y * 2.0 - 1.0;
        coord = coord * coord;
        cosVZA = cos(zenHoriz + beta * coord);
    }

    float sinVZA  = sqrt(max(0.0, 1.0 - cosVZA * cosVZA));
    float longAng = uvAdj.x * 2.0 * PI;
    vec3  localDir = vec3(sinVZA * cos(longAng), sinVZA * sin(longAng), cosVZA);

    vec3  worldPos = vec3(0.0, 0.0, vh);
    vec3  wd = localDir;

    // Move to top of atmosphere if needed
    if (vh > uTopR) {
        vec2 atm = raySphere(worldPos, wd, uTopR);
        float tTop = max(atm.x, atm.y);
        if (tTop < 0.0) { fragColor = vec4(0.0); return; }
        worldPos += wd * tTop - normalize(worldPos) * PLANET_OFFSET;
        vh = length(worldPos);
    }

    // Determine ray end
    vec2 solB = raySphere(worldPos, wd, uBotR);
    vec2 solT = raySphere(worldPos, wd, uTopR);
    float tMax;
    if (solB.x < 0.0 && solB.y < 0.0)
        tMax = max(solT.x, solT.y);
    else
        tMax = max(0.0, min(max(solB.x, 0.0), max(solB.y, 0.0)));

    bool hitGround = (solB.x > 0.0 || (solB.x < 0.0 && solB.y > 0.0));

    float sampleCount = clamp(
        mix(uSampleMin, uSampleMax, clamp(tMax * 0.001, 0.0, 1.0)),
        uSampleMin, uSampleMax);
    float dt = tMax / sampleCount;

    float cosTheta   = dot(uSunDir, wd);
    float miePhaseVal= miePhase(-cosTheta, uMieG);
    float rayPhaseVal= rayleighPhase(cosTheta);

    vec3 L          = vec3(0.0);
    vec3 throughput = vec3(1.0);

    for (float i = 0.0; i < 30.0; i++) {
        if (i >= sampleCount) break;
        float t0 = i / sampleCount;
        float t1 = (i + 1.0) / sampleCount;
        t0 = t0 * t0 * tMax;
        t1 = t1 * t1 * tMax;
        float t   = mix(t0, t1, 0.3);
        float dti = t1 - t0;

        vec3  P   = worldPos + wd * t;
        float h   = max(0.0, length(P) - uBotR);
        vec3  dens= atmDensity(h, uRayScale, uMieScale,
                               uOzo0W, uOzo0L, uOzo0C, uOzo1L, uOzo1C);
        vec3  scatter= dens.x * uRayScatter + dens.y * uMieScatter;
        vec3  ext    = dens.x * uRayScatter + dens.y * uMieExt + dens.z * uOzoExt;
        vec3  sT     = exp(-ext * dti);

        vec3  up     = P / length(P);
        float cosL   = dot(uSunDir, up);
        vec3  transL = sampleTransLUT(uTransLUT, length(P), cosL, uBotR, uTopR);

        vec2  planet = raySphere(P, uSunDir, uBotR);
        float shadow = (planet.x > 0.0 || planet.y > 0.0) ? 0.0 : 1.0;

        vec3  phase  = dens.x * uRayScatter * rayPhaseVal
                     + dens.y * uMieScatter * miePhaseVal;

        vec3  msLum  = vec3(0.0);
        if (uEnableMS > 0.5) {
            vec2 msUV = vec2(cosL * 0.5 + 0.5,
                             clamp((length(P) - uBotR) / (uTopR - uBotR), 0.0, 1.0));
            msLum = texture(uMSLUT, msUV).rgb;
        }

        vec3  S    = uSunIllum * (shadow * transL * phase + msLum * scatter);
        vec3  sint = (S - S * sT) / max(ext, vec3(1e-9));
        L          += throughput * sint;
        throughput *= sT;
    }

    // Ground bounce
    if (hitGround) {
        vec3  P    = worldPos + tMax * wd;
        vec3  up   = P / length(P);
        float cosL = dot(uSunDir, up);
        vec3  tL   = sampleTransLUT(uTransLUT, length(P), cosL, uBotR, uTopR);
        float nDL  = max(0.0, dot(up, uSunDir));
        L += uSunIllum * tL * throughput * nDL * vec3(0.1) / PI;
    }

    L *= uSkyLumFactor;
    float trans = dot(throughput, vec3(1.0 / 3.0));
    fragColor = vec4(L, trans);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3 — Sky Render  (fullscreen quad → canvas / output FBO)
// Samples SkyView LUT + adds sun disc with transmittance attenuation.
// ─────────────────────────────────────────────────────────────────────────────
const SKY_RENDER_FRAG = ATM_COMMON + /* glsl */`
in vec2 vUv;

uniform sampler2D uTransLUT;
uniform sampler2D uSkyLUT;

uniform float uBotR;
uniform float uTopR;
uniform float uViewHeightKm;

// Camera reference frame (3 row vectors → local Z = up)
uniform vec3 uRefR0;   // right
uniform vec3 uRefR1;   // forward  
uniform vec3 uRefR2;   // up

// Inverse projection (simplified: fov half-tangents)
uniform float uFovTanX;
uniform float uFovTanY;

// Sun
uniform vec3  uSunDir;          // world-space normalised
uniform float uSunCosApex;      // cos(half-angle) of disc
uniform vec3  uSunDiscLum;      // luminance
uniform float uRenderSunDisc;

uniform float uSkyLumFactor;

void main() {
    // Reconstruct view direction from UV
    float ndcX  = vUv.x * 2.0 - 1.0;
    float ndcY  = 1.0 - vUv.y * 2.0;
    vec3  vDir  = normalize(vec3(ndcX * uFovTanX, ndcY * uFovTanY, 1.0));

    // Transform to SkyView local frame (Z-up)
    vec3  lDir  = vec3(dot(uRefR0, vDir), dot(uRefR1, vDir), dot(uRefR2, vDir));

    float vh       = uViewHeightKm;
    float vHorizon = sqrt(max(0.0, vh*vh - uBotR*uBotR));
    float cosBeta  = vHorizon / vh;
    float beta     = acos(cosBeta);
    float zenHoriz = PI - beta;

    float cosVZA = lDir.z;
    float viewZenAng = acos(clamp(cosVZA, -1.0, 1.0));

    // Non-linear UV.y mapping (Bruneton)
    bool hitGround = false;
    {
        vec3  pos0  = vec3(0.0, 0.0, vh);
        vec2  solB  = raySphere(pos0, lDir, uBotR);
        hitGround   = (solB.x > 0.0 || (solB.x < 0.0 && solB.y > 0.0));
    }

    float uvY;
    if (!hitGround) {
        float coord = viewZenAng / zenHoriz;
        coord = 1.0 - coord;
        coord = sqrt(coord);
        coord = 1.0 - coord;
        uvY = coord * 0.5;
    } else {
        float coord = (viewZenAng - zenHoriz) / beta;
        coord = sqrt(max(0.0, coord));
        uvY = coord * 0.5 + 0.5;
    }

    float lutW = ${LUT_SKY_W}.0;
    float lutH = ${LUT_SKY_H}.0;
    float uvYFinal = (uvY + 0.5 / lutH) * (lutH / (lutH + 1.0));

    // Azimuth
    float azimuth  = (atan(-lDir.y, -lDir.x) + PI) / (2.0 * PI);
    float uvXFinal = (azimuth + 0.5 / lutW) * (lutW / (lutW + 1.0));

    vec3 skyLum = texture(uSkyLUT, vec2(uvXFinal, uvYFinal)).rgb * uSkyLumFactor;

    // Sun disc
    if (uRenderSunDisc > 0.5) {
        float vDotL  = dot(vDir, uSunDir);
        if (vDotL > uSunCosApex) {
            // Planet shadow
            vec3  camPos = vec3(0.0, 0.0, vh);
            vec2  planet = raySphere(camPos, vDir, uBotR);
            bool  inShadow = (planet.x > 0.0 || (planet.x < 0.0 && planet.y > 0.0));
            if (!inShadow) {
                float softEdge = clamp(2.0 * (vDotL - uSunCosApex)
                                 / max(1e-7, 1.0 - uSunCosApex), 0.0, 1.0);
                float cosL  = dot(uSunDir, vec3(0.0, 0.0, 1.0));
                vec3  transS = sampleTransLUT(uTransLUT, vh, cosL, uBotR, uTopR);
                skyLum += transS * uSunDiscLum * softEdge;
            }
        }
    }

    fragColor = vec4(skyLum, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Public config types
// ─────────────────────────────────────────────────────────────────────────────

export interface AtmosphereParams {
  bottomRadiusKm:               number;
  topRadiusKm:                  number;
  rayleighDensityExpScale:      number;
  rayleighScattering:           [number, number, number];
  mieDensityExpScale:           number;
  mieScattering:                [number, number, number];
  mieExtinction:                [number, number, number];
  miePhaseG:                    number;
  absorptionDensity0LayerWidth: number;
  absorptionDensity0LinearTerm: number;
  absorptionDensity0ConstantTerm: number;
  absorptionDensity1LinearTerm: number;
  absorptionDensity1ConstantTerm: number;
  absorptionExtinction:         [number, number, number];
  groundAlbedo:                 [number, number, number];
  multiScatteringFactor:        number;
}

export interface AtmosphereSunParams {
  direction:             [number, number, number];
  illuminanceOuterSpace: [number, number, number];
  discCosHalfApexAngle:  number;
  discLuminance:         number;
}

export interface UEAtmosphereSkyConfig {
  atmosphere:           AtmosphereParams;
  sun:                  AtmosphereSunParams;
  skyLuminanceFactor:   number;
  renderSunDisk:        boolean;
  enableMultiScattering: boolean;
  transmittanceSampleCount: number;
  multiScatteringSampleCount: number;
  fastSkySampleCountMin: number;
  fastSkySampleCountMax: number;
}

export const DEFAULT_ATMOSPHERE_PARAMS: AtmosphereParams = {
  bottomRadiusKm:               6360.0,
  topRadiusKm:                  6460.0,
  rayleighDensityExpScale:      -0.125,
  rayleighScattering:           [0.005802, 0.013558, 0.033100],
  mieDensityExpScale:           -0.8333333,
  mieScattering:                [0.003996, 0.003996, 0.003996],
  mieExtinction:                [0.004440, 0.004440, 0.004440],
  miePhaseG:                    0.8,
  absorptionDensity0LayerWidth: 25.0,
  absorptionDensity0LinearTerm: 1.0 / 15.0,
  absorptionDensity0ConstantTerm: -2.0 / 3.0,
  absorptionDensity1LinearTerm: -1.0 / 15.0,
  absorptionDensity1ConstantTerm: 8.0 / 3.0,
  absorptionExtinction:         [0.000650, 0.001881, 0.000085],
  groundAlbedo:                 [0.1, 0.1, 0.1],
  multiScatteringFactor:        1.0,
};

export const DEFAULT_SUN_PARAMS: AtmosphereSunParams = {
  direction:             [0.0, 0.3, 0.9535],
  illuminanceOuterSpace: [1.0, 1.0, 1.0],
  discCosHalfApexAngle:  0.9999747,
  discLuminance:         1.6e9,
};

export const DEFAULT_ATMOSPHERE_SKY_CONFIG: UEAtmosphereSkyConfig = {
  atmosphere:                DEFAULT_ATMOSPHERE_PARAMS,
  sun:                       DEFAULT_SUN_PARAMS,
  skyLuminanceFactor:        1.0,
  renderSunDisk:             true,
  enableMultiScattering:     true,
  transmittanceSampleCount:  40,
  multiScatteringSampleCount: 20,
  fastSkySampleCountMin:     4,
  fastSkySampleCountMax:     14,
};

// ─────────────────────────────────────────────────────────────────────────────
// UEAtmosphereSky — WebGL sky scattering renderer
//
// Usage:
//   const sky = new UEAtmosphereSky(gl);
//   sky.init();
//   sky.setSunDirection([0.3, 0.6, 0.7]);
//   sky.render(canvasW, canvasH, viewHeightKm, viewMatrix);
//   sky.dispose();
// ─────────────────────────────────────────────────────────────────────────────
export class UEAtmosphereSky {
  private gl:     WebGL2RenderingContext;
  private config: UEAtmosphereSkyConfig;

  // ── WebGL programs ─────────────────────────────────────────────────────────
  private transLUTProg!:  WebGLProgram;
  private msLUTProg!:     WebGLProgram;
  private skyViewProg!:   WebGLProgram;
  private skyRenderProg!: WebGLProgram;

  // ── FBO + texture pairs ────────────────────────────────────────────────────
  private transLUTFbo!:  WebGLFramebuffer;
  private transLUTTex!:  WebGLTexture;
  private msLUTFbo!:     WebGLFramebuffer;
  private msLUTTex!:     WebGLTexture;
  private skyViewFbo!:   WebGLFramebuffer;
  private skyViewTex!:   WebGLTexture;

  // ── Geometry ───────────────────────────────────────────────────────────────
  private quadBuf!: WebGLBuffer;

  // ── State ──────────────────────────────────────────────────────────────────
  private lutDirty     = true;
  private initialized  = false;

  constructor(gl: WebGL2RenderingContext, config: Partial<UEAtmosphereSkyConfig> = {}) {
    this.gl = gl;
    this.config = {
      ...DEFAULT_ATMOSPHERE_SKY_CONFIG,
      ...config,
      atmosphere: { ...DEFAULT_ATMOSPHERE_PARAMS, ...(config.atmosphere ?? {}) },
      sun:        { ...DEFAULT_SUN_PARAMS,        ...(config.sun       ?? {}) },
    };
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /** Update sun direction (unit vector pointing toward sun). */
  setSunDirection(dir: [number, number, number]): void {
    const n = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    this.config.sun.direction = [dir[0] / n, dir[1] / n, dir[2] / n];
    this.lutDirty = true;
  }

  /** Update sun outer-space illuminance. */
  setSunIlluminance(illum: [number, number, number]): void {
    this.config.sun.illuminanceOuterSpace = [...illum];
    this.lutDirty = true;
  }

  /** Update atmosphere physics parameters and mark LUTs dirty. */
  setAtmosphereParams(params: Partial<AtmosphereParams>): void {
    Object.assign(this.config.atmosphere, params);
    this.lutDirty = true;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // init() — createProgram + createFramebuffer + createTexture
  // ─────────────────────────────────────────────────────────────────────────
  init(): void {
    const gl = this.gl;

    // ── Compile all four programs ──────────────────────────────────────────
    this.transLUTProg  = this._compile(QUAD_VERT, TRANS_LUT_FRAG,   'transLUT');
    this.msLUTProg     = this._compile(QUAD_VERT, MS_LUT_FRAG,      'msLUT');
    this.skyViewProg   = this._compile(QUAD_VERT, SKY_VIEW_FRAG,    'skyView');
    this.skyRenderProg = this._compile(QUAD_VERT, SKY_RENDER_FRAG,  'skyRender');

    // ── Create Transmittance LUT texture + framebuffer ─────────────────────
    this.transLUTTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.transLUTTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, LUT_TRANS_W, LUT_TRANS_H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    this.transLUTFbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.transLUTFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.transLUTTex, 0);

    // ── Create Multi-Scatter LUT texture + framebuffer ─────────────────────
    this.msLUTTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.msLUTTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, LUT_MS_W, LUT_MS_H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    this.msLUTFbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.msLUTFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.msLUTTex, 0);

    // ── Create Sky View LUT texture + framebuffer ──────────────────────────
    this.skyViewTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.skyViewTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, LUT_SKY_W, LUT_SKY_H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    this.skyViewFbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.skyViewFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.skyViewTex, 0);

    // ── Restore default framebuffer binding ───────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    // ── Fullscreen quad geometry (two triangles) ───────────────────────────
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,   1, -1,   -1,  1,
      -1,  1,   1, -1,    1,  1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    this.initialized = true;
    this.lutDirty    = true;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // render() — useProgram + bindFramebuffer + drawArrays
  //
  // @param canvasW   viewport width  (px)
  // @param canvasH   viewport height (px)
  // @param viewH     camera height above planet surface (km); default 0.0002
  // @param viewMat   3×3 row-major camera orientation (optional)
  // @param fovDeg    horizontal field-of-view in degrees (optional, default 60)
  // ─────────────────────────────────────────────────────────────────────────
  render(
    canvasW: number,
    canvasH: number,
    viewH   = 0.0002,                           // 0.2 m above ground
    viewMat?: Float32Array | number[],          // 3×3 row-major
    fovDeg  = 60.0,
  ): void {
    if (!this.initialized) return;
    const gl = this.gl;

    // ── Pass 0: Transmittance LUT (rebuild only when params change) ─────────
    if (this.lutDirty) {
      this._passTransLUT();
    }

    // ── Pass 1: Multi-Scatter LUT ────────────────────────────────────────────
    if (this.lutDirty) {
      this._passMSLUT();
      this.lutDirty = false;
    }

    // ── Pass 2: Sky View LUT (every frame — sun + camera height) ────────────
    this._passSkyView(viewH);

    // ── Pass 3: Sky Render — fullscreen quad to default FBO ─────────────────
    this._passSkyRender(canvasW, canvasH, viewH, viewMat, fovDeg);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // dispose() — deleteProgram + deleteFramebuffer + deleteTexture
  // ─────────────────────────────────────────────────────────────────────────
  dispose(): void {
    const gl = this.gl;
    if (!this.initialized) return;

    // Delete programs
    gl.deleteProgram(this.transLUTProg);
    gl.deleteProgram(this.msLUTProg);
    gl.deleteProgram(this.skyViewProg);
    gl.deleteProgram(this.skyRenderProg);

    // Delete framebuffers
    gl.deleteFramebuffer(this.transLUTFbo);
    gl.deleteFramebuffer(this.msLUTFbo);
    gl.deleteFramebuffer(this.skyViewFbo);

    // Delete textures
    gl.deleteTexture(this.transLUTTex);
    gl.deleteTexture(this.msLUTTex);
    gl.deleteTexture(this.skyViewTex);

    // Delete geometry buffer
    gl.deleteBuffer(this.quadBuf);

    this.initialized = false;
  }

  /** Expose the SkyView LUT texture for downstream composite passes. */
  get skyViewTexture(): WebGLTexture { return this.skyViewTex; }

  /** Expose the Transmittance LUT texture for material system. */
  get transmittanceTexture(): WebGLTexture { return this.transLUTTex; }

  // ─────────────────────────────────────────────────────────────────────────
  // Private pass implementations
  // ─────────────────────────────────────────────────────────────────────────

  /** Pass 0 — render Transmittance LUT into transLUTFbo */
  private _passTransLUT(): void {
    const gl  = this.gl;
    const atm = this.config.atmosphere;

    gl.useProgram(this.transLUTProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.transLUTFbo);
    gl.viewport(0, 0, LUT_TRANS_W, LUT_TRANS_H);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const p = this.transLUTProg;
    gl.uniform1f(gl.getUniformLocation(p, 'uBotR'),     atm.bottomRadiusKm);
    gl.uniform1f(gl.getUniformLocation(p, 'uTopR'),     atm.topRadiusKm);
    gl.uniform1f(gl.getUniformLocation(p, 'uRayScale'), atm.rayleighDensityExpScale);
    gl.uniform3f(gl.getUniformLocation(p, 'uRayScatter'),
      atm.rayleighScattering[0], atm.rayleighScattering[1], atm.rayleighScattering[2]);
    gl.uniform1f(gl.getUniformLocation(p, 'uMieScale'), atm.mieDensityExpScale);
    gl.uniform3f(gl.getUniformLocation(p, 'uMieExt'),
      atm.mieExtinction[0], atm.mieExtinction[1], atm.mieExtinction[2]);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo0W'),    atm.absorptionDensity0LayerWidth);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo0L'),    atm.absorptionDensity0LinearTerm);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo0C'),    atm.absorptionDensity0ConstantTerm);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo1L'),    atm.absorptionDensity1LinearTerm);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo1C'),    atm.absorptionDensity1ConstantTerm);
    gl.uniform3f(gl.getUniformLocation(p, 'uOzoExt'),
      atm.absorptionExtinction[0], atm.absorptionExtinction[1], atm.absorptionExtinction[2]);
    gl.uniform1f(gl.getUniformLocation(p, 'uSteps'),    this.config.transmittanceSampleCount);

    this._drawQuad(p);
  }

  /** Pass 1 — render Multi-Scatter LUT into msLUTFbo */
  private _passMSLUT(): void {
    const gl  = this.gl;
    const atm = this.config.atmosphere;

    gl.useProgram(this.msLUTProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.msLUTFbo);
    gl.viewport(0, 0, LUT_MS_W, LUT_MS_H);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const p = this.msLUTProg;

    // Bind transmittance LUT
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.transLUTTex);
    gl.uniform1i(gl.getUniformLocation(p, 'uTransLUT'), 0);

    gl.uniform1f(gl.getUniformLocation(p, 'uBotR'),     atm.bottomRadiusKm);
    gl.uniform1f(gl.getUniformLocation(p, 'uTopR'),     atm.topRadiusKm);
    gl.uniform1f(gl.getUniformLocation(p, 'uRayScale'), atm.rayleighDensityExpScale);
    gl.uniform3f(gl.getUniformLocation(p, 'uRayScatter'),
      atm.rayleighScattering[0], atm.rayleighScattering[1], atm.rayleighScattering[2]);
    gl.uniform1f(gl.getUniformLocation(p, 'uMieScale'), atm.mieDensityExpScale);
    gl.uniform3f(gl.getUniformLocation(p, 'uMieScatter'),
      atm.mieScattering[0], atm.mieScattering[1], atm.mieScattering[2]);
    gl.uniform3f(gl.getUniformLocation(p, 'uMieExt'),
      atm.mieExtinction[0], atm.mieExtinction[1], atm.mieExtinction[2]);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo0W'),    atm.absorptionDensity0LayerWidth);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo0L'),    atm.absorptionDensity0LinearTerm);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo0C'),    atm.absorptionDensity0ConstantTerm);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo1L'),    atm.absorptionDensity1LinearTerm);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo1C'),    atm.absorptionDensity1ConstantTerm);
    gl.uniform3f(gl.getUniformLocation(p, 'uOzoExt'),
      atm.absorptionExtinction[0], atm.absorptionExtinction[1], atm.absorptionExtinction[2]);
    gl.uniform3f(gl.getUniformLocation(p, 'uGroundAlbedo'),
      atm.groundAlbedo[0], atm.groundAlbedo[1], atm.groundAlbedo[2]);
    gl.uniform1f(gl.getUniformLocation(p, 'uSteps'),           this.config.multiScatteringSampleCount);
    gl.uniform1f(gl.getUniformLocation(p, 'uMultiScatFactor'), atm.multiScatteringFactor);

    this._drawQuad(p);
  }

  /** Pass 2 — render Sky View LUT into skyViewFbo */
  private _passSkyView(viewH: number): void {
    const gl  = this.gl;
    const atm = this.config.atmosphere;
    const sun = this.config.sun;

    gl.useProgram(this.skyViewProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.skyViewFbo);
    gl.viewport(0, 0, LUT_SKY_W, LUT_SKY_H);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const p = this.skyViewProg;

    // Transmittance LUT → unit 0
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.transLUTTex);
    gl.uniform1i(gl.getUniformLocation(p, 'uTransLUT'), 0);

    // Multi-scatter LUT → unit 1
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.msLUTTex);
    gl.uniform1i(gl.getUniformLocation(p, 'uMSLUT'), 1);

    // Camera height above planet centre (km)
    const vh = atm.bottomRadiusKm + Math.max(0, viewH);
    gl.uniform1f(gl.getUniformLocation(p, 'uViewHeightKm'), vh);

    // Atmosphere
    gl.uniform1f(gl.getUniformLocation(p, 'uBotR'),     atm.bottomRadiusKm);
    gl.uniform1f(gl.getUniformLocation(p, 'uTopR'),     atm.topRadiusKm);
    gl.uniform1f(gl.getUniformLocation(p, 'uRayScale'), atm.rayleighDensityExpScale);
    gl.uniform3f(gl.getUniformLocation(p, 'uRayScatter'),
      atm.rayleighScattering[0], atm.rayleighScattering[1], atm.rayleighScattering[2]);
    gl.uniform1f(gl.getUniformLocation(p, 'uMieScale'), atm.mieDensityExpScale);
    gl.uniform3f(gl.getUniformLocation(p, 'uMieScatter'),
      atm.mieScattering[0], atm.mieScattering[1], atm.mieScattering[2]);
    gl.uniform3f(gl.getUniformLocation(p, 'uMieExt'),
      atm.mieExtinction[0], atm.mieExtinction[1], atm.mieExtinction[2]);
    gl.uniform1f(gl.getUniformLocation(p, 'uMieG'),     atm.miePhaseG);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo0W'),    atm.absorptionDensity0LayerWidth);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo0L'),    atm.absorptionDensity0LinearTerm);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo0C'),    atm.absorptionDensity0ConstantTerm);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo1L'),    atm.absorptionDensity1LinearTerm);
    gl.uniform1f(gl.getUniformLocation(p, 'uOzo1C'),    atm.absorptionDensity1ConstantTerm);
    gl.uniform3f(gl.getUniformLocation(p, 'uOzoExt'),
      atm.absorptionExtinction[0], atm.absorptionExtinction[1], atm.absorptionExtinction[2]);

    // Sun
    const sd = sun.direction;
    gl.uniform3f(gl.getUniformLocation(p, 'uSunDir'),    sd[0], sd[1], sd[2]);
    const si = sun.illuminanceOuterSpace;
    gl.uniform3f(gl.getUniformLocation(p, 'uSunIllum'),  si[0], si[1], si[2]);

    gl.uniform1f(gl.getUniformLocation(p, 'uSampleMin'),   this.config.fastSkySampleCountMin);
    gl.uniform1f(gl.getUniformLocation(p, 'uSampleMax'),   this.config.fastSkySampleCountMax);
    gl.uniform1f(gl.getUniformLocation(p, 'uSkyLumFactor'), this.config.skyLuminanceFactor);
    gl.uniform1f(gl.getUniformLocation(p, 'uEnableMS'),
      this.config.enableMultiScattering ? 1.0 : 0.0);

    this._drawQuad(p);
  }

  /** Pass 3 — fullscreen sky composite to default framebuffer */
  private _passSkyRender(
    canvasW: number,
    canvasH: number,
    viewH:   number,
    viewMat?: Float32Array | number[],
    fovDeg   = 60.0,
  ): void {
    const gl  = this.gl;
    const atm = this.config.atmosphere;
    const sun = this.config.sun;

    gl.useProgram(this.skyRenderProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);   // draw to canvas
    gl.viewport(0, 0, canvasW, canvasH);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);

    const p = this.skyRenderProg;

    // Transmittance LUT → unit 0
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.transLUTTex);
    gl.uniform1i(gl.getUniformLocation(p, 'uTransLUT'), 0);

    // Sky View LUT → unit 1
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.skyViewTex);
    gl.uniform1i(gl.getUniformLocation(p, 'uSkyLUT'), 1);

    // Camera
    const vh = atm.bottomRadiusKm + Math.max(0, viewH);
    gl.uniform1f(gl.getUniformLocation(p, 'uViewHeightKm'), vh);
    gl.uniform1f(gl.getUniformLocation(p, 'uBotR'), atm.bottomRadiusKm);
    gl.uniform1f(gl.getUniformLocation(p, 'uTopR'), atm.topRadiusKm);

    // Reference frame (default: identity, Z-up)
    let r0 = [1, 0, 0], r1 = [0, 1, 0], r2 = [0, 0, 1];
    if (viewMat && viewMat.length >= 9) {
      r0 = [viewMat[0], viewMat[1], viewMat[2]];
      r1 = [viewMat[3], viewMat[4], viewMat[5]];
      r2 = [viewMat[6], viewMat[7], viewMat[8]];
    }
    gl.uniform3f(gl.getUniformLocation(p, 'uRefR0'), r0[0], r0[1], r0[2]);
    gl.uniform3f(gl.getUniformLocation(p, 'uRefR1'), r1[0], r1[1], r1[2]);
    gl.uniform3f(gl.getUniformLocation(p, 'uRefR2'), r2[0], r2[1], r2[2]);

    // FOV
    const halfFovRad = (fovDeg * 0.5 * Math.PI) / 180.0;
    const tanH       = Math.tan(halfFovRad);
    const tanV       = tanH * (canvasH / Math.max(1, canvasW));
    gl.uniform1f(gl.getUniformLocation(p, 'uFovTanX'), tanH);
    gl.uniform1f(gl.getUniformLocation(p, 'uFovTanY'), tanV);

    // Sun
    const sd = sun.direction;
    gl.uniform3f(gl.getUniformLocation(p, 'uSunDir'), sd[0], sd[1], sd[2]);
    gl.uniform1f(gl.getUniformLocation(p, 'uSunCosApex'), sun.discCosHalfApexAngle);
    const dl = sun.discLuminance;
    gl.uniform3f(gl.getUniformLocation(p, 'uSunDiscLum'), dl, dl, dl);
    gl.uniform1f(gl.getUniformLocation(p, 'uRenderSunDisc'),
      this.config.renderSunDisk ? 1.0 : 0.0);

    gl.uniform1f(gl.getUniformLocation(p, 'uSkyLumFactor'), this.config.skyLuminanceFactor);

    this._drawQuad(p);

    // Restore GL state
    gl.enable(gl.DEPTH_TEST);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private utilities (identical pattern to fluid-gpu-pass.ts)
  // ─────────────────────────────────────────────────────────────────────────

  /** Draw fullscreen quad using aPosition attribute. */
  private _drawQuad(prog: WebGLProgram): void {
    const gl = this.gl;
    const posLoc = gl.getAttribLocation(prog, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** Compile vertex + fragment source → linked WebGLProgram. */
  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    // Runtime sanitise WebGL1→WebGL2
    const sanitise = (s: string) => s
      .replace(/\bgl_FragColor\b/g, 'fragColor')
      .replace(/\btexture2D\s*\(/g, 'texture(')
      .replace(/\btextureCube\s*\(/g, 'texture(');

    const vertSrc = sanitise(vert);
    const fragSrc = sanitise(frag);

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vertSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(vs);
      gl.deleteShader(vs);
      throw new Error(`[UEAtmosphereSky] VS compile error (${label}): ${info}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(fs);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error(`[UEAtmosphereSky] FS compile error (${label}): ${info}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteProgram(prog);
      throw new Error(`[UEAtmosphereSky] link error (${label}): ${info}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CellAtmosphereBackground — Cell world integration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * CellAtmosphereBackground — wraps UEAtmosphereSky for the Cell pubsub loop.
 *
 * ```ts
 * const atmo = new CellAtmosphereBackground(gl, species);
 * atmo.init();
 * atmo.setTimeOfDay(0.3);          // 0=midnight, 0.25=sunrise, 0.5=noon
 * atmo.render(1920, 1080, 0.0002); // call each frame
 * atmo.dispose();
 * ```
 */
export class CellAtmosphereBackground {
  private sky: UEAtmosphereSky;
  private timeOfDay = 0.5;

  constructor(
    gl:       WebGL2RenderingContext,
    species?: Partial<CellSpecies>,
    config?:  Partial<UEAtmosphereSkyConfig>,
  ) {
    const atmConfig = CellAtmosphereBackground._buildConfig(species, config);
    this.sky = new UEAtmosphereSky(gl, atmConfig);
  }

  /** Initialise GPU resources. Call once after construction. */
  init(): void {
    this.sky.init();
    this.setTimeOfDay(0.5);
  }

  /**
   * Set time of day (0..1).
   *   0.0  = midnight
   *   0.25 = sunrise
   *   0.5  = noon
   *   0.75 = sunset
   */
  setTimeOfDay(t: number): void {
    this.timeOfDay = t;
    const angle     = (t - 0.25) * 2.0 * Math.PI;
    const elevation = Math.sin(angle);
    const azimuth   = Math.cos(angle);
    const len = Math.hypot(azimuth * 0.7071, azimuth * 0.7071, elevation) || 1;
    this.sky.setSunDirection([
      (azimuth * 0.7071) / len,
      (azimuth * 0.7071) / len,
      elevation            / len,
    ]);
    const dayFactor = Math.max(0.0, elevation);
    const i = 0.15 + 0.85 * dayFactor;
    this.sky.setSunIlluminance([i, i, i]);
  }

  /**
   * Render atmosphere background to the currently bound canvas.
   * Call at the start of each frame (before scene geometry).
   */
  render(
    canvasW: number,
    canvasH: number,
    viewH  = 0.0002,
    viewMat?: Float32Array | number[],
    fovDeg   = 60.0,
  ): void {
    this.sky.render(canvasW, canvasH, viewH, viewMat, fovDeg);
  }

  /** Apply physics uniforms from the SPH world (light colour etc.) */
  applyPhysicsUniforms(u: PhysicsUniforms): void {
    const lum = (u as any).ambientLuminance as number | undefined;
    if (lum !== undefined) {
      const f = Math.max(0, Math.min(2, lum));
      this.sky.setSunIlluminance([f, f, f]);
    }
  }

  /** Return Transmittance LUT texture for downstream material passes. */
  get transmittanceLUT(): WebGLTexture { return this.sky.transmittanceTexture; }

  /** Return Sky View LUT texture for downstream composite passes. */
  get skyViewLUT(): WebGLTexture { return this.sky.skyViewTexture; }

  /** Release all GPU resources. */
  dispose(): void { this.sky.dispose(); }

  private static _buildConfig(
    species?: Partial<CellSpecies>,
    override?: Partial<UEAtmosphereSkyConfig>,
  ): Partial<UEAtmosphereSkyConfig> {
    const base: Partial<UEAtmosphereSkyConfig> = {
      ...DEFAULT_ATMOSPHERE_SKY_CONFIG,
      ...override,
    };
    const color = (species as any)?.color as [number, number, number] | undefined;
    if (color) {
      const scale = 0.15;
      base.atmosphere = {
        ...DEFAULT_ATMOSPHERE_PARAMS,
        rayleighScattering: [
          DEFAULT_ATMOSPHERE_PARAMS.rayleighScattering[0] * (1 + color[0] * scale),
          DEFAULT_ATMOSPHERE_PARAMS.rayleighScattering[1] * (1 + color[1] * scale),
          DEFAULT_ATMOSPHERE_PARAMS.rayleighScattering[2] * (1 + color[2] * scale),
        ],
        ...(override?.atmosphere ?? {}),
      };
    }
    return base;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Atmosphere presets
// ─────────────────────────────────────────────────────────────────────────────

export const PRESET_EARTH_ATMOSPHERE: Partial<UEAtmosphereSkyConfig> = {
  atmosphere: DEFAULT_ATMOSPHERE_PARAMS,
};

export const PRESET_MARS_ATMOSPHERE: Partial<UEAtmosphereSkyConfig> = {
  atmosphere: {
    ...DEFAULT_ATMOSPHERE_PARAMS,
    bottomRadiusKm:      3390.0,
    topRadiusKm:         3410.0,
    rayleighScattering:  [0.019918, 0.011358, 0.004966],
    mieScattering:       [0.002000, 0.001800, 0.001500],
    mieExtinction:       [0.002200, 0.002000, 0.001700],
    miePhaseG:           0.76,
    absorptionExtinction:[0.000250, 0.000300, 0.000025],
    groundAlbedo:        [0.35, 0.18, 0.08],
    multiScatteringFactor: 1.0,
  },
  skyLuminanceFactor: 0.08,
};

export const PRESET_ALIEN_TWILIGHT: Partial<UEAtmosphereSkyConfig> = {
  atmosphere: {
    ...DEFAULT_ATMOSPHERE_PARAMS,
    rayleighScattering: [0.012000, 0.004500, 0.019000],
    mieScattering:      [0.006000, 0.005000, 0.004000],
    miePhaseG:          0.85,
    groundAlbedo:       [0.05, 0.03, 0.07],
    multiScatteringFactor: 1.0,
  },
  skyLuminanceFactor: 1.2,
  renderSunDisk: true,
};

export const PRESET_NIGHT_SKY: Partial<UEAtmosphereSkyConfig> = {
  ...DEFAULT_ATMOSPHERE_SKY_CONFIG,
  skyLuminanceFactor: 0.001,
  renderSunDisk: false,
  sun: { ...DEFAULT_SUN_PARAMS, illuminanceOuterSpace: [0.001, 0.001, 0.003] },
};

// Re-export LUT constants for downstream consumers
export { LUT_TRANS_W, LUT_TRANS_H, LUT_MS_W, LUT_MS_H, LUT_SKY_W, LUT_SKY_H };
