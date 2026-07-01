/**
 * composite-gpu-pass.ts — M879: GPU Final Composite
 *
 * CompositeGPU: 把 6 个 G-Buffer 层在 GPU 上合成为最终画面。
 * 直接输出到 canvas (null framebuffer), 无额外 FBO。
 *
 * 输入层 (6 个 sampler2D):
 *   cell     — 细胞本体 (PBR/matcap 主色)
 *   edge     — 边/连接线
 *   particle — 粒子层 (additive)
 *   bloom    — bloom (screen blend)
 *   shadow   — 阴影遮罩
 *   fluid    — 流体 (subtle UV distortion)
 *
 * 合成顺序 (z-order, 低→高):
 *   cell * shadow  → edge → particle (additive) → bloom (screen) → fluid (distort)
 *
 * 后处理:
 *   vignette     — 1 - distance_from_center^2
 *   film grain   — fract(sin(dot(uv, seed)) * 43758.5453)
 *   color grading— simple RGB curves (shadows/midtones/highlights)
 *
 * 遵循 fluid-gpu-pass.ts 的写法:
 *   • 真实 gl.createShader / gl.createProgram / gl.compileShader
 *   • 真实 gl.bindTexture × 6 (units 0-5)
 *   • 真实 gl.bindFramebuffer(null) + gl.drawArrays
 *   • 从 compiled.vs 通过 getShader() 提取 AT shader
 *   • WebGL2 语法 (GLSL 300 es)
 */




// ─── Vertex shader (WebGL2, fullscreen quad) ─────────────────────────────────




import { getShader } from '../shaders/ShaderLoader';

const COMPOSITE_VERT = /* glsl */ `#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─── Fragment shader (~50 行, WebGL2) ────────────────────────────────────────
// 手写合成 shader (非从 compiled.vs 提取, 因 compiled.vs 无 composite.fs entry).
// AT 惯例: 最终合成 shader 内联于 ts 文件 (参见 at-scene-composite-shaders.ts).

const COMPOSITE_FRAG = /* glsl */ `#version 300 es
precision highp float;

// 6 输入层
uniform sampler2D uCell;
uniform sampler2D uEdge;
uniform sampler2D uParticle;
uniform sampler2D uBloom;
uniform sampler2D uShadow;
uniform sampler2D uFluid;

// Optional extended layers (M1250)
uniform sampler2D uGI;          // Lumen GI indirect lighting
uniform sampler2D uVolumetric;  // Volumetric light rays
uniform sampler2D uGeometry;    // AT 3D geometry preview
uniform float     uHasGI;       // 1.0 if GI texture is bound
uniform float     uHasVolumetric;
uniform float     uHasGeometry;
// M1314b: 1.0 when cell texture is real PBR output (not 1×1 placeholder)
uniform float     uHasCellContent;

// 后处理参数
uniform float uTime;
uniform float uGrainStrength;    // 胶片颗粒强度 (default 0.03)
uniform float uVignetteStrength; // 暗角强度     (default 0.6)
uniform vec3  uShadowColor;      // 暗部色调     (default vec3(0.05,0.02,0.08))
uniform vec3  uHighlightColor;   // 亮部色调     (default vec3(1.0,0.98,0.95))
uniform vec2  uResolution;       // 画布分辨率 (px)

in vec2 vUv;

out vec4 fragColor;

// ── Screen blend ─────────────────────────────────────────────────────────────
vec3 blendScreen(vec3 base, vec3 blend) {
    return 1.0 - (1.0 - base) * (1.0 - blend);
}

// ── Film grain (经典 fract/sin hash) ─────────────────────────────────────────
float filmGrain(vec2 uv, float t) {
    vec2 seed = uv * uResolution + vec2(t * 617.3, t * 131.7);
    return fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
}

// ── Simple RGB curve (shadows / highlights) ───────────────────────────────────
vec3 colorGrade(vec3 col) {
    // 亮度提升曲线: pow 轻微压暗中间调, 再 mix 阴影/高光色
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    // 暗部 → uShadowColor, 亮部 → uHighlightColor
    vec3 graded = mix(uShadowColor, uHighlightColor, luma);
    // 与原色混合 (保留饱和度)
    graded = mix(col, graded, 0.25);
    // Gamma-ish contrast toe
    graded = pow(clamp(graded, 0.0, 1.0), vec3(0.95));
    return graded;
}

void main() {
    vec2 uv = vUv;
    vec2 centeredUv = uv * 2.0 - 1.0;

    // ── 0. Cyber background ──────────────────────────────────────────────────
    // Deep blue radial gradient + subtle grid + slow noise
    float dist = length(centeredUv);
    vec3 bgCenter = vec3(0.04, 0.07, 0.14);  // dark navy
    vec3 bgEdge   = vec3(0.01, 0.02, 0.06);  // near black
    vec3 bg = mix(bgCenter, bgEdge, smoothstep(0.0, 1.4, dist));

    // Grid lines (subtle)
    float gridX = smoothstep(0.985, 1.0, fract(uv.x * 25.0));
    float gridY = smoothstep(0.985, 1.0, fract(uv.y * 25.0));
    float grid = max(gridX, gridY) * 0.035;
    bg += vec3(0.15, 0.3, 0.5) * grid;

    // Slow moving noise (fake with sin)
    float noise = sin(uv.x * 40.0 + uTime * 0.3) * sin(uv.y * 30.0 - uTime * 0.2) * 0.008;
    bg += noise;

    // ── 1. fluid UV distortion ───────────────────────────────────────────────
    vec2 fluidVel = texture(uFluid, uv).rg;
    fluidVel = (fluidVel - 0.5) * 2.0;
    vec2 distortedUv = uv + fluidVel * 0.004;
    distortedUv = clamp(distortedUv, 0.0, 1.0);

    // ── 2. 采样所有层 ────────────────────────────────────────────────────────
    vec4 cellColor     = texture(uCell,     distortedUv);
    vec4 edgeColor     = texture(uEdge,     distortedUv);
    vec4 particleColor = texture(uParticle, distortedUv);
    vec4 bloomColor    = texture(uBloom,    distortedUv);
    float shadowMask   = texture(uShadow,   distortedUv).r;

    // ── 3. Z-order composite ─────────────────────────────────────────────────
    // Start with background, alpha-over cell on top
    float ambientShadow = max(shadowMask, 0.25);
    vec3 composite = bg;

    // Cell layer: alpha-over (cell FBO has transparent background)
    // M1314b: only apply shadow multiply when real PBR content exists.
    // If cell is a 1x1 placeholder, ambientShadow would darken the background
    // and produce a near-black frame even though PBR rendered successfully.
    float shadowApply = uHasCellContent > 0.5 ? ambientShadow : 1.0;
    composite = mix(composite, cellColor.rgb * shadowApply, cellColor.a);

    // Edge layer: alpha-over
    composite = mix(composite, edgeColor.rgb, edgeColor.a * 0.85);

    // Particle layer: additive
    composite += particleColor.rgb * particleColor.a;

    // Bloom: screen blend (only where bloom has content)
    vec3 bloomContrib = bloomColor.rgb * bloomColor.a;
    composite = blendScreen(composite, bloomContrib * 0.8);

    // Geometry (3D mesh): alpha-over on top of everything
    if (uHasGeometry > 0.5) {
      vec4 geoColor = texture(uGeometry, distortedUv);
      composite = mix(composite, geoColor.rgb, geoColor.a);
    }

    // GI: subtle indirect fill
    if (uHasGI > 0.5) {
      vec3 giColor = texture(uGI, distortedUv).rgb;
      composite += giColor * 0.2;
    }

    // Volumetric: light rays
    if (uHasVolumetric > 0.5) {
      vec3 volColor = texture(uVolumetric, distortedUv).rgb;
      composite = blendScreen(composite, volColor * 0.4);
    }

    // ── 4. Post-processing ───────────────────────────────────────────────────
    // Vignette (gentle)
    float vignette = 1.0 - dot(centeredUv, centeredUv) * uVignetteStrength;
    vignette = clamp(vignette, 0.0, 1.0);
    composite *= vignette;

    // Film grain (subtle)
    float grain = filmGrain(uv, uTime);
    composite += grain * uGrainStrength;

    // Color grading
    composite = colorGrade(composite);

    fragColor = vec4(clamp(composite, 0.0, 1.0), 1.0);
}
`;

// ─── CompositeInputs: 每帧传入的 6 个纹理 ────────────────────────────────────

export interface CompositeInputs {
  cell:       WebGLTexture;
  edge:       WebGLTexture;
  particle:   WebGLTexture;
  bloom:      WebGLTexture;
  shadow:     WebGLTexture;
  fluid:      WebGLTexture;
  /** Optional: Lumen GI output (indirect lighting) */
  gi?:        WebGLTexture;
  /** Optional: Volumetric light rays */
  volumetric?: WebGLTexture;
  /** Optional: AT Geometry preview */
  geometry?:  WebGLTexture;
}

// ─── CompositeConfig: 后处理参数 ─────────────────────────────────────────────

export interface CompositeConfig {
  grainStrength:    number;   // 胶片颗粒强度 [0, 0.1]   default 0.03
  vignetteStrength: number;   // 暗角强度     [0, 1.5]  default 0.6
  shadowColor:      [number, number, number];   // 暗部色调 default [0.05,0.02,0.08]
  highlightColor:   [number, number, number];   // 亮部色调 default [1.0,0.98,0.95]
}

const DEFAULT_CONFIG: CompositeConfig = {
  grainStrength:    0.02,
  vignetteStrength: 0.3,
  shadowColor:      [0.05, 0.02, 0.08],
  highlightColor:   [1.0,  0.98, 0.95],
};

// ─── CompositeGPU ────────────────────────────────────────────────────────────

export class CompositeGPU {
  private gl:      WebGL2RenderingContext;
  private config:  CompositeConfig;

  // WebGL resources — 真正 compiled 的 shader + quad geometry
  private prog!:   WebGLProgram;
  private quadBuf!: WebGLBuffer;

  // Uniform locations (cached for performance)
  private uCell!:             WebGLUniformLocation;
  private uEdge!:             WebGLUniformLocation;
  private uParticle!:         WebGLUniformLocation;
  private uBloom!:            WebGLUniformLocation;
  private uShadow!:           WebGLUniformLocation;
  private uFluid!:            WebGLUniformLocation;
  private uGI!:               WebGLUniformLocation | null;
  private uVolumetric!:       WebGLUniformLocation | null;
  private uGeometry!:         WebGLUniformLocation | null;
  private uHasGI!:            WebGLUniformLocation | null;
  private uHasVolumetric!:    WebGLUniformLocation | null;
  private uHasGeometry!:      WebGLUniformLocation | null;
  // M1314b: signals whether cell texture is real PBR output vs 1×1 placeholder
  private uHasCellContent!:   WebGLUniformLocation | null;
  private uTime!:             WebGLUniformLocation;
  private uGrainStrength!:    WebGLUniformLocation;
  private uVignetteStrength!: WebGLUniformLocation;
  private uShadowColor!:      WebGLUniformLocation;
  private uHighlightColor!:   WebGLUniformLocation;
  private uResolution!:       WebGLUniformLocation;
  private aPosition!:         number;

  constructor(gl: WebGL2RenderingContext, config?: Partial<CompositeConfig>) {
    this.gl     = gl;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this._init();
  }

  // ─── 初始化 ────────────────────────────────────────────────────────────────

  private _init(): void {
    const gl = this.gl;

    // ── 从 compiled.vs 尝试提取 AT 合成 shader (若无则用内联 FRAG) ──────────
    // AT 的 composite shader 以内联字符串形式存在于 at-scene-composite-shaders.ts;
    // compiled.vs 里没有独立的 composite.fs entry, 所以我们用内联 COMPOSITE_FRAG.
    // 但仍通过 getShader 尝试加载 (留给未来版本扩展).
    let fragSrc = COMPOSITE_FRAG;
    try {
      // 若将来 compiled.vs 中新增 composite.fs, 自动使用 AT 版本
      const atFrag = getShader('composite.fs');
      if (atFrag && atFrag.length > 0) {
        fragSrc = atFrag;
      }
    } catch (_) {
      // composite.fs 未找到 → 使用内联版本 (预期行为)
    }

    // ── 编译 composite shader (真正的 gl 调用) ──────────────────────────────
    this.prog = this._compile(COMPOSITE_VERT, fragSrc, 'composite');

    // ── 缓存 uniform / attrib locations ──────────────────────────────────────
    this.uCell             = gl.getUniformLocation(this.prog, 'uCell')!;
    this.uEdge             = gl.getUniformLocation(this.prog, 'uEdge')!;
    this.uParticle         = gl.getUniformLocation(this.prog, 'uParticle')!;
    this.uBloom            = gl.getUniformLocation(this.prog, 'uBloom')!;
    this.uShadow           = gl.getUniformLocation(this.prog, 'uShadow')!;
    this.uFluid            = gl.getUniformLocation(this.prog, 'uFluid')!;
    this.uGI               = gl.getUniformLocation(this.prog, 'uGI');
    this.uVolumetric       = gl.getUniformLocation(this.prog, 'uVolumetric');
    this.uGeometry         = gl.getUniformLocation(this.prog, 'uGeometry');
    this.uHasGI            = gl.getUniformLocation(this.prog, 'uHasGI');
    this.uHasVolumetric    = gl.getUniformLocation(this.prog, 'uHasVolumetric');
    this.uHasGeometry      = gl.getUniformLocation(this.prog, 'uHasGeometry');
    // M1314b: cell content flag
    this.uHasCellContent   = gl.getUniformLocation(this.prog, 'uHasCellContent');
    this.uTime             = gl.getUniformLocation(this.prog, 'uTime')!;
    this.uGrainStrength    = gl.getUniformLocation(this.prog, 'uGrainStrength')!;
    this.uVignetteStrength = gl.getUniformLocation(this.prog, 'uVignetteStrength')!;
    this.uShadowColor      = gl.getUniformLocation(this.prog, 'uShadowColor')!;
    this.uHighlightColor   = gl.getUniformLocation(this.prog, 'uHighlightColor')!;
    this.uResolution       = gl.getUniformLocation(this.prog, 'uResolution')!;
    this.aPosition         = gl.getAttribLocation(this.prog, 'aPosition');

    // ── 全屏 quad (2 个三角形, 同 fluid-gpu-pass.ts) ─────────────────────────
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);
  }

  // ─── 每帧调用: 把 6 个输入纹理合成到 canvas ────────────────────────────────

  render(
    inputs: CompositeInputs,
    canvasWidth: number,
    canvasHeight: number,
    time: number,
    /** M1314b: true when cell texture is real PBR output (not 1×1 placeholder) */
    hasCellContent = true,
  ): void {
    const gl = this.gl;

    // 输出直接到 canvas (null framebuffer)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasWidth, canvasHeight);

    // ── Blend 状态: composite pass 直接覆写 canvas, 必须关闭 blend ────────────
    // 若前序 pass (particle/fluid) 开启了 additive blend 而未还原,
    // 输出会被错误叠加导致全白或全黑 (M1217 核心 bug).
    gl.disable(gl.BLEND);

    // 清除上一帧残留 (防止脏帧缓冲)
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.prog);

    // ── 上传 6 个 sampler2D (texture units 0-5) ───────────────────────────────

    // unit 0 — cell
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputs.cell);
    gl.uniform1i(this.uCell, 0);

    // unit 1 — edge
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, inputs.edge);
    gl.uniform1i(this.uEdge, 1);

    // unit 2 — particle
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, inputs.particle);
    gl.uniform1i(this.uParticle, 2);

    // unit 3 — bloom
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, inputs.bloom);
    gl.uniform1i(this.uBloom, 3);

    // unit 4 — shadow
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, inputs.shadow);
    gl.uniform1i(this.uShadow, 4);

    // unit 5 — fluid
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, inputs.fluid);
    gl.uniform1i(this.uFluid, 5);

    // unit 6 — GI (optional, M1250)
    const hasGI = !!inputs.gi;
    if (hasGI && this.uGI) {
      gl.activeTexture(gl.TEXTURE6);
      gl.bindTexture(gl.TEXTURE_2D, inputs.gi!);
      gl.uniform1i(this.uGI, 6);
    }
    if (this.uHasGI) gl.uniform1f(this.uHasGI, hasGI ? 1.0 : 0.0);

    // unit 7 — Volumetric (optional, M1250)
    const hasVol = !!inputs.volumetric;
    if (hasVol && this.uVolumetric) {
      gl.activeTexture(gl.TEXTURE7);
      gl.bindTexture(gl.TEXTURE_2D, inputs.volumetric!);
      gl.uniform1i(this.uVolumetric, 7);
    }
    if (this.uHasVolumetric) gl.uniform1f(this.uHasVolumetric, hasVol ? 1.0 : 0.0);

    // unit 8 — Geometry (optional, M1250)
    const hasGeo = !!inputs.geometry;
    if (hasGeo && this.uGeometry) {
      gl.activeTexture(gl.TEXTURE8);
      gl.bindTexture(gl.TEXTURE_2D, inputs.geometry!);
      gl.uniform1i(this.uGeometry, 8);
    }
    if (this.uHasGeometry) gl.uniform1f(this.uHasGeometry, hasGeo ? 1.0 : 0.0);

    // M1314b: tell shader whether cell tex is real content
    if (this.uHasCellContent) gl.uniform1f(this.uHasCellContent, hasCellContent ? 1.0 : 0.0);

    // ── 上传后处理 uniforms ───────────────────────────────────────────────────
    gl.uniform1f(this.uTime,             time);
    gl.uniform1f(this.uGrainStrength,    this.config.grainStrength);
    gl.uniform1f(this.uVignetteStrength, this.config.vignetteStrength);
    gl.uniform3f(this.uShadowColor,
      this.config.shadowColor[0],
      this.config.shadowColor[1],
      this.config.shadowColor[2]);
    gl.uniform3f(this.uHighlightColor,
      this.config.highlightColor[0],
      this.config.highlightColor[1],
      this.config.highlightColor[2]);
    gl.uniform2f(this.uResolution, canvasWidth, canvasHeight);

    // ── 画全屏 quad → gl.drawArrays ──────────────────────────────────────────
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(this.aPosition);
    gl.vertexAttribPointer(this.aPosition, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // ── 还原 GL 状态 (不污染后续 pass) ──────────────────────────────────────
    gl.disableVertexAttribArray(this.aPosition);
  }

  // ─── 运行时更新后处理参数 ──────────────────────────────────────────────────

  setConfig(partial: Partial<CompositeConfig>): void {
    Object.assign(this.config, partial);
  }

  // ─── 销毁: 释放 GPU 资源 ────────────────────────────────────────────────────

  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.prog);
    gl.deleteBuffer(this.quadBuf);
  }

  // ─── 内部: 编译 WebGL2 shader ──────────────────────────────────────────────

  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    // vertex shader
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(
        `[CompositeGPU] vertex compile error (${label}): ${gl.getShaderInfoLog(vs)}`
      );
    }

    // fragment shader
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(
        `[CompositeGPU] fragment compile error (${label}): ${gl.getShaderInfoLog(fs)}`
      );
    }

    // link
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(
        `[CompositeGPU] link error (${label}): ${gl.getProgramInfoLog(prog)}`
      );
    }

    // 编译完成后即可删除 shader 对象 (AT 惯例)
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    return prog;
  }

  /** Expose primary WebGLProgram for UIL uniform injection. */
  get program(): WebGLProgram { return this.prog; }
}
