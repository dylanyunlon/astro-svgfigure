/**
 * ParticleSystem.ts — GPGPU Particle System
 *
 * Active Theory 风格：每个粒子 = 一个像素，存储在 RenderTexture 里。
 * Ping-pong 双缓冲：tPosition/tVelocity 每帧交替读写。
 *
 * 参数来源: channels/physics/at_uil_params.json
 *
 *  uCurlNoiseSpeed  : WorkDetailParticles=5, TubesInteraction=5, work_page=10
 *  uSCurlNoiseSpeed : WorkDetailParticles=5
 *  uSCurlNoiseScale : WorkDetailParticles=2
 *  uThicknessSpeed  : WorkDetailParticles=1  (AT: uThicknessSpeed)
 *  uSplineSpeed     : [0.82, 1.21]           (AT: uSplineSpeed)
 *
 * 架构参考:
 *   src/lib/gpgpu/constraint-texture.ts   (ping-pong pattern)
 *   src/lib/gpgpu/constraint-shader.frag  (GPGPU update shader)
 *   src/lib/renderers/fluid-fbo.ts        (FBO double-buffer)
 */

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ParticleSystemConfig {
  gl: WebGL2RenderingContext;
  /** Particle count — must be a perfect square for square texture layout */
  particleCount: number;
  /**
   * AT参数: uCurlNoiseSpeed
   * WorkDetailParticles=5, TubesInteraction=5, work_page=10
   */
  curlNoiseSpeed?: number;
  /**
   * AT参数: uSCurlNoiseSpeed (spline curl)
   * WorkDetailParticles=5
   */
  sCurlNoiseSpeed?: number;
  /**
   * AT参数: uCurlNoiseScale
   * WorkDetailParticles=2, BodyCores=7
   */
  curlNoiseScale?: number;
  /**
   * AT参数: uSCurlNoiseScale
   * WorkDetailParticles=2
   */
  sCurlNoiseScale?: number;
  /** AT参数: uCurlTimeScale */
  curlTimeScale?: number;
  /** AT参数: uSCurlTimeScale */
  sCurlTimeScale?: number;
  /**
   * AT参数: uThicknessSpeed = 1
   * Controls spline thickness animation speed
   */
  thicknessSpeed?: number;
  /**
   * AT参数: uSplineSpeed = [0.82, 1.21]
   * Min/max particle travel speed along spline
   */
  splineSpeed?: [number, number];
  /** Optional GLSL update shader src override */
  updateShaderSrc?: string;
}

export interface ParticleSystemUniforms {
  uCurlNoiseSpeed: number;
  uSCurlNoiseSpeed: number;
  uCurlNoiseScale: number;
  uSCurlNoiseScale: number;
  uCurlTimeScale: number;
  uSCurlTimeScale: number;
  uThicknessSpeed: number;
  uSplineSpeedMin: number;
  uSplineSpeedMax: number;
  uTime: number;
  uDelta: number;
}

/** RGBA float texture: RGB=position, A=life */
export interface GPGPURenderTexture {
  texture: WebGLTexture;
  framebuffer: WebGLFramebuffer;
  width: number;
  height: number;
}

// ── AT parameter defaults (from at_uil_params.json) ──────────────────────────

const AT_DEFAULTS = {
  curlNoiseSpeed:  5,     // am_ProtonAntimatter_P_Element_0_WorkDetailParticles uSCurlNoiseSpeed
  sCurlNoiseSpeed: 5,     // am_ProtonAntimatter_P_Element_0_WorkDetailParticles uSCurlNoiseSpeed
  curlNoiseScale:  2,     // am_ProtonAntimatter_P_Element_0_WorkDetailParticles uSCurlNoiseScale
  sCurlNoiseScale: 2,     // am_ProtonAntimatter_P_Element_0_WorkDetailParticles uSCurlNoiseScale
  curlTimeScale:   2,     // am_ProtonAntimatter_P_Element_0_WorkDetailParticles uSCurlTimeScale
  sCurlTimeScale:  2,
  thicknessSpeed:  1,     // am_ProtonAntimatter_P_Element_0_WorkDetailParticles uThicknessSpeed
  splineSpeedMin:  0.82,  // am_SplineParticleLife_Element_0_WorkDetailParticles uSplineSpeed[0]
  splineSpeedMax:  1.21,  // am_SplineParticleLife_Element_0_WorkDetailParticles uSplineSpeed[1]
} as const;

// ── Vertex shader for full-screen quad (GPGPU pass) ──────────────────────────

const VERT_SRC = /* glsl */`#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

// ── Render vertex shader (point sprites) ─────────────────────────────────────

const RENDER_VERT_SRC = /* glsl */`#version 300 es
precision highp float;

uniform sampler2D tPosition;
uniform sampler2D tVelocity;
uniform mat4 uProjection;
uniform mat4 uModelView;
uniform float uPointSize;

in vec2 aParticleUv;    /* per-particle UV into state textures */
out float vLife;
out vec3  vVelocity;

void main() {
  vec4 posLife = texture(tPosition, aParticleUv);
  vLife     = posLife.w;
  vVelocity = texture(tVelocity, aParticleUv).xyz;

  vec4 mvPos = uModelView * vec4(posLife.xyz, 1.0);
  gl_Position  = uProjection * mvPos;
  gl_PointSize = uPointSize * (1.0 / -mvPos.z);
}`;

// ── Render fragment shader (point sprite) ────────────────────────────────────

const RENDER_FRAG_SRC = /* glsl */`#version 300 es
precision highp float;

uniform vec3  uColor;
uniform float uAlpha;

in float vLife;
in vec3  vVelocity;

out vec4 fragColor;

void main() {
  /* Soft circle */
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float r = dot(uv, uv);
  if (r > 1.0) discard;

  float alpha = (1.0 - r) * vLife * uAlpha;
  float speed = length(vVelocity);
  vec3  col   = uColor + vec3(speed * 0.1);
  fragColor   = vec4(col, alpha);
}`;

// ── Helpers ──────────────────────────────────────────────────────────────────

function createShader(
  gl: WebGL2RenderingContext,
  type: GLenum,
  src: string,
): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(`[ParticleSystem] Shader compile error:\n${gl.getShaderInfoLog(s)}`);
  }
  return s;
}

function createProgram(
  gl: WebGL2RenderingContext,
  vertSrc: string,
  fragSrc: string,
): WebGLProgram {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, createShader(gl, gl.VERTEX_SHADER,   vertSrc));
  gl.attachShader(prog, createShader(gl, gl.FRAGMENT_SHADER, fragSrc));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(`[ParticleSystem] Program link error:\n${gl.getProgramInfoLog(prog)}`);
  }
  return prog;
}

function createFloatTexture(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  data?: Float32Array,
): WebGLTexture {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(
    gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0,
    gl.RGBA, gl.FLOAT,
    data ?? null,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

function createFBO(
  gl: WebGL2RenderingContext,
  texture: WebGLTexture,
): WebGLFramebuffer {
  const fbo = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error(`[ParticleSystem] FBO incomplete: 0x${status.toString(16)}`);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return fbo;
}

function createRenderTexture(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  data?: Float32Array,
): GPGPURenderTexture {
  const texture     = createFloatTexture(gl, w, h, data);
  const framebuffer = createFBO(gl, texture);
  return { texture, framebuffer, width: w, height: h };
}

// ── ParticleSystem ────────────────────────────────────────────────────────────

export class ParticleSystem {
  private gl: WebGL2RenderingContext;
  private texSize: number;
  readonly particleCount: number;

  // ── AT uniforms (loaded from at_uil_params.json) ───────────────────────────
  readonly uniforms: ParticleSystemUniforms;

  // ── GPGPU ping-pong state ──────────────────────────────────────────────────
  private posRead:  GPGPURenderTexture;
  private posWrite: GPGPURenderTexture;
  private velRead:  GPGPURenderTexture;
  private velWrite: GPGPURenderTexture;

  // ── GL resources ──────────────────────────────────────────────────────────
  private updateProgram: WebGLProgram;
  private renderProgram: WebGLProgram;
  private quadVAO:       WebGLVertexArrayObject;
  private particleVAO:   WebGLVertexArrayObject;

  private time = 0;

  constructor(config: ParticleSystemConfig) {
    const {
      gl,
      particleCount,
      updateShaderSrc,
    } = config;

    this.gl            = gl;
    this.particleCount = particleCount;

    // Texture must be square — round up to nearest perfect square
    const side    = Math.ceil(Math.sqrt(particleCount));
    this.texSize  = side;

    // ── AT parameter defaults ─────────────────────────────────────────────
    this.uniforms = {
      uCurlNoiseSpeed:  config.curlNoiseSpeed  ?? AT_DEFAULTS.curlNoiseSpeed,
      uSCurlNoiseSpeed: config.sCurlNoiseSpeed ?? AT_DEFAULTS.sCurlNoiseSpeed,
      uCurlNoiseScale:  config.curlNoiseScale  ?? AT_DEFAULTS.curlNoiseScale,
      uSCurlNoiseScale: config.sCurlNoiseScale ?? AT_DEFAULTS.sCurlNoiseScale,
      uCurlTimeScale:   config.curlTimeScale   ?? AT_DEFAULTS.curlTimeScale,
      uSCurlTimeScale:  config.sCurlTimeScale  ?? AT_DEFAULTS.sCurlTimeScale,
      uThicknessSpeed:  config.thicknessSpeed  ?? AT_DEFAULTS.thicknessSpeed,
      uSplineSpeedMin:  config.splineSpeed?.[0] ?? AT_DEFAULTS.splineSpeedMin,
      uSplineSpeedMax:  config.splineSpeed?.[1] ?? AT_DEFAULTS.splineSpeedMax,
      uTime:  0,
      uDelta: 1,
    };

    // ── Initialise particle state ─────────────────────────────────────────
    const total   = side * side;
    const posData = new Float32Array(total * 4);
    const velData = new Float32Array(total * 4);
    for (let i = 0; i < total; i++) {
      posData[i * 4 + 0] = (Math.random() - 0.5) * 2;   // x
      posData[i * 4 + 1] = (Math.random() - 0.5) * 2;   // y
      posData[i * 4 + 2] = (Math.random() - 0.5) * 2;   // z
      posData[i * 4 + 3] = Math.random();                // life
    }

    this.posRead  = createRenderTexture(gl, side, side, posData);
    this.posWrite = createRenderTexture(gl, side, side);
    this.velRead  = createRenderTexture(gl, side, side, velData);
    this.velWrite = createRenderTexture(gl, side, side);

    // ── Load update shader (CurlNoise.frag) ──────────────────────────────
    // In production: import curlFrag from './CurlNoise.frag?raw';
    // Here we accept override or expect caller to inject
    if (!updateShaderSrc) {
      console.warn(
        '[ParticleSystem] No updateShaderSrc provided. ' +
        'Import CurlNoise.frag as raw string and pass via config.updateShaderSrc.',
      );
    }
    const fragSrc = updateShaderSrc ?? `#version 300 es
precision highp float;
uniform sampler2D tPosition;
uniform float uTime;
in vec2 vUv;
out vec4 fragColor;
void main() { fragColor = texture(tPosition, vUv); }`;

    this.updateProgram = createProgram(gl, VERT_SRC, fragSrc);
    this.renderProgram = createProgram(gl, RENDER_VERT_SRC, RENDER_FRAG_SRC);

    // ── Full-screen quad VAO (GPGPU pass) ─────────────────────────────────
    const quadVerts = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
    const quadVBO   = gl.createBuffer()!;
    this.quadVAO    = gl.createVertexArray()!;
    gl.bindVertexArray(this.quadVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
    gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
    const aPos = gl.getAttribLocation(this.updateProgram, 'aPosition');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // ── Particle UV VAO (render pass) ────────────────────────────────────
    const uvData = new Float32Array(total * 2);
    for (let iy = 0; iy < side; iy++) {
      for (let ix = 0; ix < side; ix++) {
        const idx = (iy * side + ix) * 2;
        uvData[idx + 0] = (ix + 0.5) / side;
        uvData[idx + 1] = (iy + 0.5) / side;
      }
    }
    const uvVBO   = gl.createBuffer()!;
    this.particleVAO = gl.createVertexArray()!;
    gl.bindVertexArray(this.particleVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, uvVBO);
    gl.bufferData(gl.ARRAY_BUFFER, uvData, gl.STATIC_DRAW);
    const aUV = gl.getAttribLocation(this.renderProgram, 'aParticleUv');
    gl.enableVertexAttribArray(aUV);
    gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    console.log(
      `[ParticleSystem] Initialised ${particleCount} particles ` +
      `in ${side}×${side} texture. ` +
      `curlNoiseSpeed=${this.uniforms.uCurlNoiseSpeed} ` +
      `splineSpeed=[${this.uniforms.uSplineSpeedMin},${this.uniforms.uSplineSpeedMax}]`,
    );
  }

  // ── Update (GPGPU compute pass) ──────────────────────────────────────────

  update(delta: number): void {
    const gl = this.gl;
    this.time += delta;
    this.uniforms.uTime  = this.time;
    this.uniforms.uDelta = delta * 60; // normalise to 60fps (AT's HZ)

    const prog = this.updateProgram;
    gl.useProgram(prog);

    // Bind uniforms
    const u = this.uniforms;
    const setF = (name: string, v: number) => {
      const loc = gl.getUniformLocation(prog, name);
      if (loc !== null) gl.uniform1f(loc, v);
    };
    setF('uCurlNoiseSpeed',  u.uCurlNoiseSpeed);
    setF('uSCurlNoiseSpeed', u.uSCurlNoiseSpeed);
    setF('uCurlNoiseScale',  u.uCurlNoiseScale);
    setF('uSCurlNoiseScale', u.uSCurlNoiseScale);
    setF('uCurlTimeScale',   u.uCurlTimeScale);
    setF('uSCurlTimeScale',  u.uSCurlTimeScale);
    setF('uTime',            u.uTime);
    setF('uDelta',           u.uDelta);

    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.posRead.texture);
    gl.uniform1i(gl.getUniformLocation(prog, 'tPosition'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.velRead.texture);
    gl.uniform1i(gl.getUniformLocation(prog, 'tVelocity'), 1);

    // Render to write FBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.posWrite.framebuffer);
    gl.viewport(0, 0, this.texSize, this.texSize);
    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Swap ping-pong
    [this.posRead, this.posWrite] = [this.posWrite, this.posRead];
    [this.velRead, this.velWrite] = [this.velWrite, this.velRead];
  }

  // ── Render (point sprites) ───────────────────────────────────────────────

  render(
    projection: Float32Array,
    modelView:  Float32Array,
    options: { pointSize?: number; color?: [number,number,number]; alpha?: number } = {},
  ): void {
    const gl   = this.gl;
    const prog = this.renderProgram;
    gl.useProgram(prog);

    gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProjection'), false, projection);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uModelView'),  false, modelView);
    gl.uniform1f(gl.getUniformLocation(prog, 'uPointSize'), options.pointSize ?? 2.0);
    gl.uniform3fv(gl.getUniformLocation(prog, 'uColor'),
      options.color ? new Float32Array(options.color) : new Float32Array([0.4, 0.7, 1.0]));
    gl.uniform1f(gl.getUniformLocation(prog, 'uAlpha'), options.alpha ?? 0.8);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.posRead.texture);
    gl.uniform1i(gl.getUniformLocation(prog, 'tPosition'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.velRead.texture);
    gl.uniform1i(gl.getUniformLocation(prog, 'tVelocity'), 1);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive — AT standard for particles
    gl.depthMask(false);

    gl.bindVertexArray(this.particleVAO);
    gl.drawArrays(gl.POINTS, 0, this.particleCount);
    gl.bindVertexArray(null);

    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }

  // ── Accessors ────────────────────────────────────────────────────────────

  /** Current position texture (for external sampling / SplineEmitter) */
  get positionTexture(): WebGLTexture { return this.posRead.texture; }

  /** Current velocity texture */
  get velocityTexture(): WebGLTexture { return this.velRead.texture; }

  /** Set a uniform at runtime (e.g. driven by UIL tweaks) */
  setUniform<K extends keyof ParticleSystemUniforms>(
    key: K,
    value: ParticleSystemUniforms[K],
  ): void {
    (this.uniforms as Record<string, unknown>)[key] = value;
  }

  dispose(): void {
    const gl = this.gl;
    for (const rt of [this.posRead, this.posWrite, this.velRead, this.velWrite]) {
      gl.deleteTexture(rt.texture);
      gl.deleteFramebuffer(rt.framebuffer);
    }
    gl.deleteProgram(this.updateProgram);
    gl.deleteProgram(this.renderProgram);
    gl.deleteVertexArray(this.quadVAO);
    gl.deleteVertexArray(this.particleVAO);
  }
}
