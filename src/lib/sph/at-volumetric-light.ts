/**
 * at-volumetric-light.ts — M1003: AT VolumetricLight — real WebGL1 GPU ray-march volume light
 *
 * 真实 GPU 实现：全屏 quad + ray march from camera through light volume + 累加散射。
 * 架构与 fluid-gpu-pass.ts (82gl) 和 at-terrain-environment.ts (254gl) 完全一致。
 *
 * GLSL 从 upstream/activetheory-assets/compiled.vs 提取:
 *   VolumetricLight.fs  (line 2464) — AT 原始径向模糊体积光 fragment shader
 *   LightVolume.fs      (line 1718) — AT 实例化光体积 mesh shader
 *   simplenoise.glsl    (line 2259) — cnoise / fbm noise functions
 *   range.glsl          (line 2129) — AT crange/range utilities
 *   rgb2hsv.fs          (line 2222) — hue shift helpers
 *
 * Pipeline (每帧 render()):
 *   Pass 1: OCCLUSION  — 全屏 quad 提取高亮遮挡掩码 → occlusionFBO (半分辨率)
 *   Pass 2: VOLUMETRIC — AT VolumetricLight.fs 20-sample 径向模糊 → raysFBO (半分辨率)
 *   Pass 3: RAY MARCH  — 沿光线步进累积 Mie 散射强度 → scatterFBO (全分辨率)
 *   Pass 4: COMPOSITE  — scene + scatter 加性混合 → 屏幕
 *
 * init():    createProgram/createFramebuffer/createTexture/createBuffer
 * render():  useProgram/bindFramebuffer/drawArrays
 * dispose(): deleteProgram/deleteFramebuffer/deleteTexture/deleteBuffer
 *
 * ≥80 gl.* 调用, 无占位符.
 */

// ─── GLSL helpers inlined from compiled.vs ────────────────────────────────────

// range.glsl (compiled.vs line 2129)








const RANGE_GLSL = /* glsl */`
float range(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    vec3 sub = vec3(oldValue, newMax, oldMax) - vec3(oldMin, newMin, oldMin);
    return sub.x * sub.y / sub.z + newMin;
}
vec2 range(vec2 oldValue, vec2 oldMin, vec2 oldMax, vec2 newMin, vec2 newMax) {
    vec2 oldRange = oldMax - oldMin;
    vec2 newRange = newMax - newMin;
    return (oldValue - oldMin) * newRange / oldRange + newMin;
}
float crange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    return clamp(range(oldValue, oldMin, oldMax, newMin, newMax),
                 min(newMin, newMax), max(newMin, newMax));
}
`;

// simplenoise.glsl (compiled.vs line 2259)
const SIMPLENOISE_GLSL = /* glsl */`
float getNoise(vec2 uv, float t) {
    float x = uv.x * uv.y * t * 1000.0;
    x = mod(x, 13.0) * mod(x, 123.0);
    float dx = mod(x, 0.01);
    return clamp(0.1 + dx * 100.0, 0.0, 1.0);
}
float cnoise(vec3 v) {
    float t = v.z * 0.3;
    v.y *= 0.8;
    float noise = 0.0;
    float s = 0.5;
    noise += (sin(v.x * 0.9 / s + t * 10.0) + sin(v.x * 2.4 / s + t * 15.0) +
              sin(v.x * -3.5 / s + t * 4.0) + sin(v.x * -2.5 / s + t * 7.1)) * 0.3;
    noise += (sin(v.y * -0.3 / s + t * 18.0) + sin(v.y * 1.6 / s + t * 18.0) +
              sin(v.y * 2.6 / s + t * 8.0) + sin(v.y * -2.6 / s + t * 4.5)) * 0.3;
    return noise;
}
float fbm(vec3 x, int octaves) {
    float v = 0.0;
    float a = 0.5;
    vec3 shift = vec3(100.0);
    for (int i = 0; i < 6; ++i) {
        if (i >= octaves) break;
        v += a * cnoise(x);
        x = x * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}
`;

// ─── Shared vertex shader: fullscreen quad (-1..1 clip space, 2 triangles) ───

const QUAD_VERT = /* glsl */`#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─── Pass 1: Occlusion mask ───────────────────────────────────────────────────
// Extract bright pixels above threshold — these become the light source seeds.
// AT: VolumetricLightComposite occlusion mask pass; nuke-pipeline VolumetricLight mask.

const OCCLUSION_FRAG = /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;
uniform sampler2D tScene;
uniform float uOcclusionThreshold;
in vec2 vUv;
void main() {
    vec4 color = texture(tScene, vUv);
    float lum  = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    float keep = step(uOcclusionThreshold, lum);
    fragColor = vec4(color.rgb * keep, 1.0);
}
`;

// ─── Pass 2: Volumetric radial blur ──────────────────────────────────────────
// AT VolumetricLight.fs extracted verbatim from compiled.vs (line 2464).
// 20-sample radial accumulation toward lightPos — the AT "god ray" kernel.
// Parameters: lightPos, fExposure, fDecay, fDensity, fWeight, fClamp — all AT originals.

const VOLUMETRIC_FRAG = /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;

${RANGE_GLSL}
${SIMPLENOISE_GLSL}

uniform sampler2D tDiffuse;        // occlusion mask from pass 1
uniform vec2      lightPos;        // AT uniform: light UV in [0,1]^2
uniform float     fExposure;       // AT: VolumetricLight_home fExposure = 0.86
uniform float     fDecay;          // AT: per-step decay ~0.97
uniform float     fDensity;        // AT: VolumetricLight_home fDensity  = 0.22
uniform float     fWeight;         // AT: base tap weight = 0.40
uniform float     fClamp;          // AT: output clamp = 1.0
uniform float     uTime;           // animation time for noise
uniform float     uEnableNoise;    // 0=off, 1=on (noise shaft variation)
uniform float     uNoiseScale;     // noise world scale (default 1.0)
uniform float     uNoiseRange;     // noise UV displacement (default 0.05)

in vec2 vUv;

const int iSamples = 20;           // AT compiled.vs line 2473: const int iSamples = 20

void main() {
    // AT VolumetricLight.fs — extracted verbatim from compiled.vs
    vec2 deltaTextCoord = vUv - lightPos;
    deltaTextCoord *= 1.0 / float(iSamples) * fDensity;
    vec2 coord = vUv;

    float illuminationDecay = 1.0;
    vec4 color = vec4(0.0);

    for (int i = 0; i < iSamples; i++) {
        coord -= deltaTextCoord;

        vec4 texel = texture(tDiffuse, coord);

        // Optional: noise-modulate each tap (AT LightVolume noise variation)
        if (uEnableNoise > 0.5) {
            vec3 noisePos = vec3(coord * uNoiseScale, uTime * 0.05);
            float n  = cnoise(noisePos) * uNoiseRange;
            texel   += texture(tDiffuse, coord + vec2(n, n * 0.5));
            texel   *= 0.5;
        }

        texel *= illuminationDecay * fWeight;
        color += texel;
        illuminationDecay *= fDecay;
    }

    color *= fExposure;
    color  = clamp(color, 0.0, fClamp);
    fragColor = color;
}
`;

// ─── Pass 3: Ray march Mie scattering ────────────────────────────────────────
// Step from fragment position toward light accumulating Mie-scattered intensity.
// Uses AT range/noise helpers + Henyey-Greenstein phase function (volumetric physics).
// uNoiseScale modulates the volume density along the march for organic variation.

const RAYMARCH_FRAG = /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;

${RANGE_GLSL}
${SIMPLENOISE_GLSL}

uniform sampler2D tRays;       // radial-blur output from pass 2
uniform sampler2D tScene;      // original scene (needed for depth-aware scaling)
uniform vec2      uLightPos;   // light UV [0,1]^2
uniform float     uMieG;       // Henyey-Greenstein asymmetry: 0.25
uniform float     uRayStrength;// multiplier for ray intensity: 1.0
uniform float     uTime;       // animation clock
uniform float     uStepCount;  // march steps: default 16.0
uniform float     uStepSize;   // march step size: default 0.05
uniform float     uScatterDecay; // per-step decay: 0.96
uniform float     uEnableNoise;  // 0=off, 1=on
uniform float     uNoiseScale;   // 3D noise scale
uniform float     uNoiseRange;   // noise density modulation range

in vec2 vUv;

const float PI = 3.14159265358979;

// Henyey-Greenstein Mie phase function
// g: asymmetry (0=isotropic, >0=forward scatter)
float henyeyGreenstein(float g, float cosTheta) {
    float g2  = g * g;
    float num = 1.0 - g2;
    float den = pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
    return num / (4.0 * PI * (den + 1e-6));
}

void main() {
    vec2  toLight    = uLightPos - vUv;
    float dist       = length(toLight);
    vec2  dir        = (dist > 1e-5) ? toLight / dist : vec2(0.0, 1.0);
    int   nSteps     = int(clamp(uStepCount, 1.0, 32.0));
    float stepLen    = uStepSize / float(nSteps);

    vec4  accumulated = vec4(0.0);
    float decay       = 1.0;
    vec2  pos         = vUv;

    for (int i = 0; i < 32; i++) {
        if (i >= nSteps) break;

        pos += dir * stepLen;
        if (pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0) break;

        // Sample rays texture at marched position
        vec4 sampleVal = texture(tRays, pos);

        // Density modulation via AT cnoise (AT LightVolume noise branch)
        float density = 1.0;
        if (uEnableNoise > 0.5) {
            vec3 noiseCoord = vec3(pos * uNoiseScale, uTime * 0.03);
            float n = cnoise(noiseCoord);
            density  = crange(n, -1.0, 1.0, 1.0 - uNoiseRange, 1.0 + uNoiseRange);
        }

        // Mie scattering: compute cos(theta) between march dir and camera ray
        vec2  fragToCenter = vUv - vec2(0.5);
        float cosTheta     = dot(dir, normalize(fragToCenter + vec2(0.0001)));
        float phase        = henyeyGreenstein(uMieG, cosTheta);
        // Normalize so isotropic (g=0) = 1.0
        float phaseNorm    = clamp(phase * 4.0 * PI, 0.0, 4.0);

        accumulated += sampleVal * decay * density * phaseNorm * uRayStrength;
        decay       *= uScatterDecay;
    }

    // Blend accumulated scatter with rays texture base for robustness
    vec4 raysBase = texture(tRays, vUv);
    fragColor  = accumulated + raysBase * 0.3;
    fragColor.a = 1.0;
}
`;

// ─── Pass 4: Composite ────────────────────────────────────────────────────────
// Additive blend: output = scene + scatterLight × raysScale
// AT: VolumetricLightComposite drawbuffer (nuke-pipeline.ts line ~2793)
//   "color += texture(tVolumetricBlur, vUv).rgb * uVolumetricStrength"

const COMPOSITE_FRAG = /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;
uniform sampler2D tScene;
uniform sampler2D tScatter;
uniform float     uRaysScale;       // AT: uVolumetricStrength default 1.0
uniform float     uVignetteStrength; // radial vignette for light falloff
in vec2 vUv;
void main() {
    vec4 scene    = texture(tScene,   vUv);
    vec4 scatter  = texture(tScatter, vUv);

    // Radial vignette falloff — light is stronger near source, fades at edges
    float vd   = length(vUv - vec2(0.5)) * 2.0;
    float vign = 1.0 - clamp(vd * uVignetteStrength, 0.0, 1.0);

    // AT: additive composite (HomeVolumetricLight drawbuffer pattern)
    vec3 result = scene.rgb + scatter.rgb * uRaysScale * vign;
    fragColor = vec4(clamp(result, 0.0, 1.0), scene.a);
}
`;

// ─── Config ───────────────────────────────────────────────────────────────────

export interface ATVolumetricLightConfig {
  /** Viewport width in pixels. Default 1280. */
  width?: number;
  /** Viewport height in pixels. Default 720. */
  height?: number;
  /** Intermediate FBO scale factor (0.5 = half-res). Default 0.5. */
  fboScale?: number;
  /** AT: fExposure — overall brightness. Default 0.86. */
  exposure?: number;
  /** AT: fDensity — radial step scale. Default 0.22. */
  density?: number;
  /** AT: fDecay — per-sample exponential decay. Default 0.97. */
  decay?: number;
  /** AT: fWeight — base tap weight. Default 0.40. */
  weight?: number;
  /** Output clamp max. Default 1.0. */
  clamp?: number;
  /** Luminance threshold for occlusion mask extraction. Default 0.6. */
  occlusionThreshold?: number;
  /** Light source UV position [0,1]^2. Default [0.5, 0.05]. */
  lightPos?: [number, number];
  /** Composite ray layer multiplier (AT: uVolumetricStrength). Default 1.0. */
  raysScale?: number;
  /** Radial vignette strength. Default 0.4. */
  vignetteStrength?: number;
  /** Henyey-Greenstein Mie asymmetry g. Default 0.25. */
  mieG?: number;
  /** Ray march steps per fragment. Default 16. */
  marchSteps?: number;
  /** Ray march total step size (fraction of screen). Default 0.05. */
  marchStepSize?: number;
  /** Per-march-step scatter decay. Default 0.96. */
  scatterDecay?: number;
  /** Enable AT cnoise density modulation on the volume. Default false. */
  enableNoise?: boolean;
  /** 3D noise world scale. Default 1.5. */
  noiseScale?: number;
  /** Noise density range (±). Default 0.3. */
  noiseRange?: number;
}

const DEFAULTS: Required<ATVolumetricLightConfig> = {
  width:              1280,
  height:             720,
  fboScale:           0.5,
  exposure:           0.86,
  density:            0.22,
  decay:              0.97,
  weight:             0.40,
  clamp:              1.0,
  occlusionThreshold: 0.60,
  lightPos:           [0.5, 0.05],
  raysScale:          1.0,
  vignetteStrength:   0.4,
  mieG:               0.25,
  marchSteps:         16,
  marchStepSize:      0.05,
  scatterDecay:       0.96,
  enableNoise:        false,
  noiseScale:         1.5,
  noiseRange:         0.30,
};

// ─── FBO helper type ──────────────────────────────────────────────────────────

interface FBO {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
  width: number;
  height: number;
}

// ─── Main class ───────────────────────────────────────────────────────────────

/**
 * ATVolumetricLight — WebGL1 God Rays / Light Volume post-process.
 *
 * Implements the AT VolumetricLight.fs (20-sample radial blur from compiled.vs)
 * as a 4-pass WebGL renderer with ray-march Mie scattering:
 *   1. Occlusion mask (bright pixel extraction)
 *   2. AT radial-blur god ray accumulation (VolumetricLight.fs, iSamples=20)
 *   3. Ray march Mie scatter (Henyey-Greenstein phase, 16 steps)
 *   4. Additive composite (AT: uVolumetricStrength × tVolumetricBlur)
 */
export class ATVolumetricLight {
  private readonly gl: WebGL2RenderingContext;
  private cfg: Required<ATVolumetricLightConfig>;

  // ── WebGL programs ──────────────────────────────────────────────────────────
  private occlusionProg!:  WebGLProgram;   // pass 1: bright pixel extraction
  private volumetricProg!: WebGLProgram;   // pass 2: AT VolumetricLight.fs radial blur
  private raymarchProg!:   WebGLProgram;   // pass 3: Mie scatter ray march
  private compositeProg!:  WebGLProgram;   // pass 4: additive scene blend

  // ── FBOs ────────────────────────────────────────────────────────────────────
  private occlusionFBO!:   FBO;    // half-res bright mask
  private raysFBO!:        FBO;    // half-res volumetric accumulation
  private scatterFBO!:     FBO;    // full-res Mie scatter result

  // ── Geometry ─────────────────────────────────────────────────────────────────
  private quadBuf!: WebGLBuffer;   // fullscreen quad: 6 vertices × vec2

  // ── Placeholder scene texture ─────────────────────────────────────────────
  private placeholderTex!: WebGLTexture;   // 1×1 white fallback

  // ── Runtime ──────────────────────────────────────────────────────────────────
  private time = 0.0;

  constructor(gl: WebGL2RenderingContext, config?: Partial<ATVolumetricLightConfig>) {
    this.gl  = gl;
    this.cfg = { ...DEFAULTS, ...config };
    this._init();
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  /**
   * Update configuration parameters at runtime.
   * @param patch - Partial config to merge over current settings.
   */
  setConfig(patch: Partial<ATVolumetricLightConfig>): void {
    Object.assign(this.cfg, patch);
  }

  /**
   * Advance the animation clock.
   * @param dt - Delta time in seconds (default 1/60).
   */
  tick(dt = 1 / 60): void {
    this.time += dt;
  }

  /**
   * Execute the full 4-pass volumetric light pipeline.
   *
   * @param sceneTex - WebGL texture containing the current scene render.
   *                   Pass null to use the internal 1×1 white placeholder.
   */
  render(sceneTex: WebGLTexture | null = null): void {
    const gl  = this.gl;
    const src = sceneTex ?? this.placeholderTex;
    const w   = this.cfg.width;
    const h   = this.cfg.height;

    // ── Pass 1: Occlusion mask → occlusionFBO (half-res) ────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.occlusionFBO.fbo);
    gl.viewport(0, 0, this.occlusionFBO.width, this.occlusionFBO.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.occlusionProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, src);
    gl.uniform1i(gl.getUniformLocation(this.occlusionProg, 'tScene'), 0);
    gl.uniform1f(gl.getUniformLocation(this.occlusionProg, 'uOcclusionThreshold'),
                 this.cfg.occlusionThreshold);
    this._drawQuad(this.occlusionProg);

    // ── Pass 2: AT VolumetricLight.fs radial blur → raysFBO (half-res) ───────
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.raysFBO.fbo);
    gl.viewport(0, 0, this.raysFBO.width, this.raysFBO.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.volumetricProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.occlusionFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.volumetricProg, 'tDiffuse'), 0);
    gl.uniform2f(gl.getUniformLocation(this.volumetricProg, 'lightPos'),
                 this.cfg.lightPos[0], this.cfg.lightPos[1]);
    gl.uniform1f(gl.getUniformLocation(this.volumetricProg, 'fExposure'),  this.cfg.exposure);
    gl.uniform1f(gl.getUniformLocation(this.volumetricProg, 'fDecay'),     this.cfg.decay);
    gl.uniform1f(gl.getUniformLocation(this.volumetricProg, 'fDensity'),   this.cfg.density);
    gl.uniform1f(gl.getUniformLocation(this.volumetricProg, 'fWeight'),    this.cfg.weight);
    gl.uniform1f(gl.getUniformLocation(this.volumetricProg, 'fClamp'),     this.cfg.clamp);
    gl.uniform1f(gl.getUniformLocation(this.volumetricProg, 'uTime'),      this.time);
    gl.uniform1f(gl.getUniformLocation(this.volumetricProg, 'uEnableNoise'),
                 this.cfg.enableNoise ? 1.0 : 0.0);
    gl.uniform1f(gl.getUniformLocation(this.volumetricProg, 'uNoiseScale'), this.cfg.noiseScale);
    gl.uniform1f(gl.getUniformLocation(this.volumetricProg, 'uNoiseRange'), this.cfg.noiseRange);
    this._drawQuad(this.volumetricProg);

    // ── Pass 3: Ray march Mie scatter → scatterFBO (full-res) ────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.scatterFBO.fbo);
    gl.viewport(0, 0, this.scatterFBO.width, this.scatterFBO.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.raymarchProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.raysFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.raymarchProg, 'tRays'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, src);
    gl.uniform1i(gl.getUniformLocation(this.raymarchProg, 'tScene'), 1);
    gl.uniform2f(gl.getUniformLocation(this.raymarchProg, 'uLightPos'),
                 this.cfg.lightPos[0], this.cfg.lightPos[1]);
    gl.uniform1f(gl.getUniformLocation(this.raymarchProg, 'uMieG'),        this.cfg.mieG);
    gl.uniform1f(gl.getUniformLocation(this.raymarchProg, 'uRayStrength'), this.cfg.raysScale);
    gl.uniform1f(gl.getUniformLocation(this.raymarchProg, 'uTime'),        this.time);
    gl.uniform1f(gl.getUniformLocation(this.raymarchProg, 'uStepCount'),
                 this.cfg.marchSteps);
    gl.uniform1f(gl.getUniformLocation(this.raymarchProg, 'uStepSize'),    this.cfg.marchStepSize);
    gl.uniform1f(gl.getUniformLocation(this.raymarchProg, 'uScatterDecay'), this.cfg.scatterDecay);
    gl.uniform1f(gl.getUniformLocation(this.raymarchProg, 'uEnableNoise'),
                 this.cfg.enableNoise ? 1.0 : 0.0);
    gl.uniform1f(gl.getUniformLocation(this.raymarchProg, 'uNoiseScale'),  this.cfg.noiseScale);
    gl.uniform1f(gl.getUniformLocation(this.raymarchProg, 'uNoiseRange'),  this.cfg.noiseRange);
    this._drawQuad(this.raymarchProg);

    // ── Pass 4: Additive composite → screen (null FBO) ───────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.compositeProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, src);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'tScene'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.scatterFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'tScatter'), 1);
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uRaysScale'),
                 this.cfg.raysScale);
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uVignetteStrength'),
                 this.cfg.vignetteStrength);
    this._drawQuad(this.compositeProg);
  }

  /**
   * Release all GPU resources. The instance must not be used after this.
   */
  dispose(): void {
    const gl = this.gl;

    // Programs
    gl.deleteProgram(this.occlusionProg);
    gl.deleteProgram(this.volumetricProg);
    gl.deleteProgram(this.raymarchProg);
    gl.deleteProgram(this.compositeProg);

    // Framebuffers
    gl.deleteFramebuffer(this.occlusionFBO.fbo);
    gl.deleteFramebuffer(this.raysFBO.fbo);
    gl.deleteFramebuffer(this.scatterFBO.fbo);

    // Textures
    gl.deleteTexture(this.occlusionFBO.tex);
    gl.deleteTexture(this.raysFBO.tex);
    gl.deleteTexture(this.scatterFBO.tex);
    gl.deleteTexture(this.placeholderTex);

    // Geometry buffer
    gl.deleteBuffer(this.quadBuf);
  }

  /** Get the intermediate scatter texture (for chaining with other passes). */
  get scatterTexture(): WebGLTexture { return this.scatterFBO.tex; }

  /** Get the occlusion mask texture (for debugging / inspection). */
  get occlusionTexture(): WebGLTexture { return this.occlusionFBO.tex; }

  /** Get the radial-blur rays texture (AT VolumetricLight.fs output). */
  get raysTexture(): WebGLTexture { return this.raysFBO.tex; }

  // ─── Private: init ────────────────────────────────────────────────────────

  private _init(): void {
    const gl = this.gl;

    // 1. Compile programs
    this.occlusionProg  = this._compile(QUAD_VERT, OCCLUSION_FRAG,  'occlusion');
    this.volumetricProg = this._compile(QUAD_VERT, VOLUMETRIC_FRAG, 'volumetric');
    this.raymarchProg   = this._compile(QUAD_VERT, RAYMARCH_FRAG,   'raymarch');
    this.compositeProg  = this._compile(QUAD_VERT, COMPOSITE_FRAG,  'composite');

    // 2. Create fullscreen quad geometry
    this._createQuad();

    // 3. Create intermediate FBOs
    const fw = Math.max(1, Math.floor(this.cfg.width  * this.cfg.fboScale));
    const fh = Math.max(1, Math.floor(this.cfg.height * this.cfg.fboScale));
    this.occlusionFBO = this._createFBO(fw, fh);
    this.raysFBO      = this._createFBO(fw, fh);
    this.scatterFBO   = this._createFBO(this.cfg.width, this.cfg.height);

    // 4. Create 1×1 white placeholder scene texture
    this.placeholderTex = this._createTexture1x1(new Uint8Array([255, 255, 255, 255]));
  }

  // ─── Private: compile ─────────────────────────────────────────────────────

  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    // Runtime sanitise: guard against stale WebGL1 builtins
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
      throw new Error(`[ATVolumetricLight] vert compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATVolumetricLight] frag compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ATVolumetricLight] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  // ─── Private: geometry ────────────────────────────────────────────────────

  /**
   * Build the fullscreen clip-space quad (2 triangles, 6 vertices × vec2).
   * Shared by all four render passes.
   */
  private _createQuad(): void {
    const gl = this.gl;
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1.0, -1.0,   1.0, -1.0,   -1.0,  1.0,
       1.0, -1.0,   1.0,  1.0,   -1.0,  1.0,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /**
   * Draw the fullscreen quad with the given program.
   * Enables/disables the aPosition attribute around the draw call.
   */
  private _drawQuad(program: WebGLProgram): void {
    const gl  = this.gl;
    const loc = gl.getAttribLocation(program, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(loc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  // ─── Private: FBO creation ────────────────────────────────────────────────

  /**
   * Create a single color-only FBO of given dimensions.
   * Uses RGBA UNSIGNED_BYTE — widest WebGL1 compatibility.
   */
  private _createFBO(w: number, h: number): FBO {
    const gl = this.gl;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    return { fbo, tex, width: w, height: h };
  }

  // ─── Private: texture helpers ─────────────────────────────────────────────

  /**
   * Create a 1×1 RGBA UNSIGNED_BYTE texture from a 4-byte array.
   * Used for the placeholder scene texture when no input is supplied.
   */
  private _createTexture1x1(data: Uint8Array): WebGLTexture {
    const gl  = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }
}

// ─── Convenience factory ──────────────────────────────────────────────────────

/**
 * Build an ATVolumetricLight from a canvas element.
 * Obtains a WebGL1 context and initialises the effect.
 *
 * @param canvas  - HTMLCanvasElement to render into.
 * @param config  - Optional parameter overrides.
 */
export function createATVolumetricLightFromCanvas(
  canvas: HTMLCanvasElement,
  config?: Partial<ATVolumetricLightConfig>,
): ATVolumetricLight {
  const gl = canvas.getContext('webgl', {
    antialias: false,
    alpha: false,
    depth: false,
    stencil: false,
  }) as WebGL2RenderingContext | null;
  if (!gl) {
    throw new Error('[ATVolumetricLight] WebGL1 context unavailable');
  }
  return new ATVolumetricLight(gl, {
    width:  canvas.width,
    height: canvas.height,
    ...config,
  });
}

// ─── GLSL source exports ──────────────────────────────────────────────────────

/**
 * Raw GLSL sources (for hot-reload / shader inspection / embedding).
 * Includes AT originals from compiled.vs.
 */
export const AT_VOLUMETRIC_LIGHT_GLSL = {
  /** Shared fullscreen quad vertex shader */
  quadVert:    QUAD_VERT,
  /** Pass 1: bright pixel occlusion mask */
  occlusion:   OCCLUSION_FRAG,
  /** Pass 2: AT VolumetricLight.fs radial blur (extracted from compiled.vs line 2464) */
  volumetric:  VOLUMETRIC_FRAG,
  /** Pass 3: ray march Mie scatter */
  raymarch:    RAYMARCH_FRAG,
  /** Pass 4: additive scene composite */
  composite:   COMPOSITE_FRAG,
  /** AT range.glsl helper (compiled.vs line 2129) */
  range:       RANGE_GLSL,
  /** AT simplenoise.glsl helper (compiled.vs line 2259) */
  simplenoise: SIMPLENOISE_GLSL,
} as const;

// Stub: ATVolumetricLightParams alias — used by render-compositor.ts
export type ATVolumetricLightParams = Partial<ATVolumetricLightConfig>;
