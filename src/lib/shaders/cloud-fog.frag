#version 300 es
// ── cloud-fog.frag ────────────────────────────────────────────────────────────
// Volumetric fog fragment shader.
//
// Technique: each draw call renders one semi-transparent quad at a specific
// depth within the fog volume.  Stacking N quads (AT default: 20) with
// per-layer depth offsets and independent noise phases produces convincing
// volume-fog without ray-marching.
//
// AT CloudFog module reference parameters:
//   alpha=1.8, planes=20, noise=1, speed=0.7
//   width=[-4,4], height=[-1,4], depth=[-2,-2]
//   fadeDist=[2,4], cullDistance=999, scale=6
//
// Uniforms:
//   uAlpha       — master opacity scale (AT: 1.8, clamped per-layer)
//   uNoise       — noise intensity  (AT: 1.0; 0 = no noise, flat fog)
//   uSpeed       — animation speed  (AT: 0.7)
//   uScale       — noise domain scale (AT: 6; larger → coarser clumps)
//   uTime        — elapsed seconds
//   uLayerIndex  — which plane this is  (0 … uPlaneCount-1)
//   uPlaneCount  — total plane count    (AT: 20)
//   uFadeNear    — near fade distance   (AT: 2.0)
//   uFadeFar     — far  fade distance   (AT: 4.0)
//   uFogColor    — RGB fog tint         (default: white)
//
// Inputs from vertex shader:
//   vUV          — [0,1] across the quad
//   vDepth01     — remapped 0=back … 1=front within the volume
// ─────────────────────────────────────────────────────────────────────────────

precision highp float;

// ── varyings ──────────────────────────────────────────────────────────────────
in vec2  vUV;
in float vDepth01;   // 0 = deepest layer, 1 = closest layer

// ── uniforms ──────────────────────────────────────────────────────────────────
// AT UIL CloudFog defaults (source: channels/physics/at_uil_params.json)
//   uAlpha      = 1.8        (INPUT_CloudFoghome_alpha)
//   uNoise      = 1.0        (INPUT_CloudFoghome_noise)
//   uSpeed      = 0.7        (INPUT_CloudFoghome_speed)
//   uScale      = 6.0        (INPUT_CloudFoghome_scale)
//   uPlaneCount = 20         (INPUT_CloudFoghome_planes)
//   uFadeNear   = 2.0        (INPUT_CloudFoghome_fadeDist[0])
//   uFadeFar    = 4.0        (INPUT_CloudFoghome_fadeDist[1])
//   width       = [-4, 4]    (INPUT_CloudFoghome_width)
//   height      = [-1, 4]    (INPUT_CloudFoghome_height)
uniform float uAlpha;      // AT default: 1.8
uniform float uNoise;      // AT default: 1.0
uniform float uSpeed;      // AT default: 0.7
uniform float uScale;      // AT default: 6.0
uniform float uTime;
uniform int   uLayerIndex;
uniform int   uPlaneCount; // AT default: 20
uniform float uFadeNear;   // AT default: 2.0  (fadeDist[0])
uniform float uFadeFar;    // AT default: 4.0  (fadeDist[1])
uniform vec3  uFogColor;

// ── output ────────────────────────────────────────────────────────────────────
out vec4 finalColor;

// ─────────────────────────────────────────────────────────────────────────────
// Simplex-style 2-D gradient noise.
//
// Based on Stefan Gustavson's public-domain simplex noise, simplified to a
// single-octave 2-D version for fragment shader use.
// Reference: https://github.com/stegu/webgl-noise (MIT / public domain)
// ─────────────────────────────────────────────────────────────────────────────

vec3 _mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 _mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 _permute(vec3 x) { return _mod289(((x * 34.0) + 10.0) * x); }

// Returns noise in [-1, 1].
float snoise2(vec2 v) {
    const vec4 C = vec4( 0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                         0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626,  // -1.0 + 2.0 * C.x
                         0.024390243902439); // 1.0 / 41.0

    // Skew input space to determine simplex cell
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v -   i + dot(i, C.xx);

    // Simplex corners
    vec2 i1  = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy  -= i1;

    // Gradient contributions
    i = _mod289(i);
    vec3 p = _permute(_permute(i.y + vec3(0.0, i1.y, 1.0)) +
                               i.x + vec3(0.0, i1.x, 1.0));

    vec3 m = max(0.5 - vec3(dot(x0, x0),
                            dot(x12.xy, x12.xy),
                            dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;

    vec3 x   = 2.0 * fract(p * C.www) - 1.0;
    vec3 h   = abs(x) - 0.5;
    vec3 ox  = floor(x + 0.5);
    vec3 a0  = x - ox;

    // Normalise gradients
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

    vec3 g;
    g.x  = a0.x  * x0.x    + h.x  * x0.y;
    g.yz = a0.yz * x12.xz  + h.yz * x12.yw;

    return 130.0 * dot(m, g);
}

// 2-octave fBm wrapper — returns [0, 1].
float fbm(vec2 p) {
    float n = snoise2(p)       * 0.6
            + snoise2(p * 2.1) * 0.4;
    return clamp(n * 0.5 + 0.5, 0.0, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
void main() {
    // ── Per-layer identity ───────────────────────────────────────────────────
    float layerT  = float(uLayerIndex) / max(float(uPlaneCount) - 1.0, 1.0);
    //  layerT: 0 = first/back plane, 1 = last/front plane

    // ── Animated noise sample ────────────────────────────────────────────────
    // Each layer drifts at a slightly different phase to break repetition.
    float phaseX  = uTime * uSpeed * (0.7 + layerT * 0.3);
    float phaseY  = uTime * uSpeed * (0.5 - layerT * 0.2);

    vec2 noiseUV  = vUV * uScale + vec2(phaseX, phaseY);
    float density = fbm(noiseUV);

    // Mix with flat fog using uNoise intensity
    float fogDensity = mix(0.65, density, clamp(uNoise, 0.0, 1.0));

    // ── Depth-based fade (AT fadeDist=[near, far]) ───────────────────────────
    // vDepth01 drives a smooth fade: thin at edges, thicker in the middle.
    float depthFade = smoothstep(0.0, uFadeNear / (uFadeNear + uFadeFar),
                                 vDepth01)
                    * smoothstep(1.0, 1.0 - uFadeNear / (uFadeNear + uFadeFar),
                                 vDepth01);

    // ── Edge fade (vignette on UV) ───────────────────────────────────────────
    vec2  centered   = vUV * 2.0 - 1.0;  // [-1,1]
    float edgeFade   = 1.0 - smoothstep(0.6, 1.0, length(centered));

    // ── Layer opacity ────────────────────────────────────────────────────────
    // Back layers are slightly more transparent to keep the stack believable.
    float layerAlpha = uAlpha
                     * fogDensity
                     * depthFade
                     * edgeFade
                     * mix(0.4, 1.0, layerT);   // back→dim, front→bright

    // Clamp to avoid over-bright compositing
    float alpha = clamp(layerAlpha * 0.12, 0.0, 1.0);  // 0.12 ≈ 1/8 per layer

    if (alpha < 0.002) discard;

    // Pre-multiplied alpha output (matches PixiJS additive / normal blend modes)
    finalColor = vec4(uFogColor * alpha, alpha);
}
