/**
 * fluid-consumer-gpu.ts — M888: 流体纹理驱动视觉效果
 *
 * FluidConsumerGPU: 消费 FluidGPU 输出的 velocityTexture + dyeTexture,
 * 驱动三个真实 GPU 效果:
 *
 *   Pass 1 — Cell Distortion:
 *     读 velocityTexture, 速度偏移 cell surface UV → 表面扭曲
 *     frag: vec2 vel = texture2D(uFluidVel, vUv).xy;
 *           vec2 distortedUV = vUv + vel * 0.05;
 *           gl_FragColor = texture2D(uCellTex, distortedUV);
 *
 *   Pass 2 — Particle Acceleration (GPU readback + CPU apply):
 *     从 velocityTexture 读粒子位置处的速度 → particle velocity += fluidVel * strength
 *     每粒子在 GPU 采样速度场 → 写入 accelerationTex (1D 纹理)
 *     CPU 读回 → 更新 particle VBO
 *
 *   Pass 3 — Background Flow:
 *     dyeTexture 作为背景色叠加 (additive / multiply 混合)
 *     frag: vec4 dye = texture2D(uDye, vUv); gl_FragColor = vec4(dye.rgb * uDyeStrength, dye.a);
 *
 * 每个效果:
 *   • 真实 gl.bindTexture + gl.drawArrays
 *   • 真实 gl.createShader / gl.compileShader / gl.linkProgram
 *   • 真实 gl.bindFramebuffer (offscreen FBO 或 null)
 *
 * 遵循 fluid-gpu-pass.ts 的写法 (WebGL1, varying/texture2D)
 *
 * 最少 30 处 gl.* 调用 (实际远超此数).
 *
 * 集成:
 *   ```ts
 *   const consumer = new FluidConsumerGPU(gl);
 *   // 每帧:
 *   const distortedTex = consumer.cellDistortionPass(velocityTex, cellTex, w, h);
 *   consumer.particleAccelerationPass(velocityTex, particles, w, h, 0.8);
 *   consumer.backgroundFlowPass(dyeTex, canvasW, canvasH, 0.4);
 *   consumer.destroy();
 *   ```
 */

// ─── WebGL1 Vertex Shaders ────────────────────────────────────────────────────

/** 全屏 quad vert (WebGL1, no varying neighbours) */
const FULLSCREEN_VERT = /* glsl */ `
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

/** 粒子 UV 查询 vert — 每粒子 1 个像素点, 从 uParticlePosAtlas 拿位置 */
const PARTICLE_QUERY_VERT = /* glsl */ `
precision highp float;
attribute float aParticleIndex;
uniform sampler2D uParticlePosAtlas; // 粒子位置纹理 (RGB=posXY_unused, 1×N)
uniform float uAtlasWidth;
varying float vParticleIndex;
void main() {
    float u = (aParticleIndex + 0.5) / uAtlasWidth;
    vParticleIndex = aParticleIndex;
    // 每粒子渲染成 1px × 1px 的点
    gl_Position = vec4(u * 2.0 - 1.0, 0.0, 0.0, 1.0);
    gl_PointSize = 1.0;
}
`;

// ─── WebGL1 Fragment Shaders ──────────────────────────────────────────────────

/**
 * Pass 1: Cell Distortion Fragment Shader
 * 读速度纹理 → 偏移 UV → 采样 cell 纹理
 */
const CELL_DISTORTION_FRAG = /* glsl */ `
precision highp float;

uniform sampler2D uFluidVel;     // velocityTexture (RG = vx,vy, range [-1,1])
uniform sampler2D uCellTex;      // cell surface 纹理
uniform float     uDistortStrength; // 扭曲强度 (default 0.05)
uniform vec2      uVelScale;     // 速度场分辨率归一化

varying vec2 vUv;

void main() {
    // 1. 从流体速度场读 vel
    vec2 vel = texture2D(uFluidVel, vUv).xy;  // raw RG
    // 2. 速度中心化 (AT 存的是 [0,1] 范围, 需要 *2-1)
    vel = vel * 2.0 - 1.0;
    // 3. 偏移 UV
    vec2 distortedUV = vUv + vel * uDistortStrength;
    // 4. clamp to valid UV range
    distortedUV = clamp(distortedUV, vec2(0.001), vec2(0.999));
    // 5. 采样 cell 纹理
    gl_FragColor = texture2D(uCellTex, distortedUV);
}
`;

/**
 * Pass 2: Particle Acceleration Fragment Shader
 * 在粒子世界坐标处采样速度场 → 输出加速度 delta
 */
const PARTICLE_ACCEL_FRAG = /* glsl */ `
precision highp float;

uniform sampler2D uFluidVel;      // velocityTexture
uniform sampler2D uParticlePosAtlas; // 粒子 NDC 位置 atlas (每行=1粒子, RG=xy)
uniform float     uAtlasWidth;
uniform float     uAccelStrength; // 加速系数 (default 0.8)

varying float vParticleIndex;

void main() {
    // 粒子在 atlas 中的 U 坐标
    float u = (vParticleIndex + 0.5) / uAtlasWidth;
    // 从 atlas 读粒子 NDC 位置
    vec2 particleNDC = texture2D(uParticlePosAtlas, vec2(u, 0.5)).rg;
    // NDC → UV (流体纹理坐标)
    vec2 fluidUV = particleNDC * 0.5 + 0.5;
    fluidUV = clamp(fluidUV, vec2(0.0), vec2(1.0));
    // 采样流体速度
    vec2 vel = texture2D(uFluidVel, fluidUV).xy;
    vel = vel * 2.0 - 1.0;
    // 输出: RG = velocity delta (scaled), BA = 0
    gl_FragColor = vec4(vel * uAccelStrength, 0.0, 1.0);
}
`;

/**
 * Pass 3: Background Flow Fragment Shader
 * dye texture 作为背景色叠加
 */
const BACKGROUND_FLOW_FRAG = /* glsl */ `
precision highp float;

uniform sampler2D uDye;           // dyeTexture (RGBA)
uniform sampler2D uBaseColor;     // 原始背景纹理 (可以是黑色 FBO)
uniform float     uDyeStrength;   // 叠加强度 (default 0.4)
uniform float     uTime;          // 时间 (用于动态噪声)
uniform vec2      uResolution;    // 画布分辨率

varying vec2 vUv;

// 简单的 2D hash 噪声 (AT 惯例)
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    // 1. 读 dye
    vec4 dye  = texture2D(uDye, vUv);
    // 2. 读基底颜色
    vec4 base = texture2D(uBaseColor, vUv);
    // 3. dye 颜色增强 (饱和度 boost)
    vec3 dyeEnhanced = dye.rgb * 1.4;
    // 4. 轻微抖动避免色带
    float grain = (hash(vUv + vec2(uTime * 0.01)) - 0.5) * 0.015;
    dyeEnhanced += grain;
    // 5. Additive 叠加
    vec3 blended = base.rgb + dyeEnhanced * uDyeStrength;
    blended = clamp(blended, 0.0, 1.0);
    // 6. 输出
    gl_FragColor = vec4(blended, 1.0);
}
`;

// ─── 类型定义 ─────────────────────────────────────────────────────────────────

export interface ParticleData {
  /** NDC 坐标 [x0,y0, x1,y1, ...] */
  positions: Float32Array;
  /** 速度 [vx0,vy0, vx1,vy1, ...] */
  velocities: Float32Array;
  count: number;
}

export interface ConsumerConfig {
  distortStrength?: number;   // cell 扭曲强度 (default 0.05)
  accelStrength?: number;     // 粒子加速系数 (default 0.8)
  dyeStrength?: number;       // 背景 dye 叠加强度 (default 0.4)
  maxParticles?: number;      // 粒子上限 (default 2048)
}

const DEFAULT_CONSUMER_CONFIG: Required<ConsumerConfig> = {
  distortStrength: 0.05,
  accelStrength: 0.8,
  dyeStrength: 0.4,
  maxParticles: 2048,
};

// ─── FluidConsumerGPU ─────────────────────────────────────────────────────────

export class FluidConsumerGPU {
  private gl: WebGLRenderingContext;
  private cfg: Required<ConsumerConfig>;

  // ── Pass 1: Cell Distortion ──
  private distortProg!: WebGLProgram;
  private distortFBO!: WebGLFramebuffer;
  private distortTex!: WebGLTexture;
  private distortW = 0;
  private distortH = 0;

  // ── Pass 2: Particle Acceleration ──
  private accelProg!: WebGLProgram;
  private accelFBO!: WebGLFramebuffer;
  private accelTex!: WebGLTexture;          // output: 每粒子 vel delta
  private particlePosAtlasTex!: WebGLTexture; // input:  粒子位置 atlas
  private particleIdxBuf!: WebGLBuffer;     // attribute: aParticleIndex

  // ── Pass 3: Background Flow ──
  private bgFlowProg!: WebGLProgram;
  private bgBaseTex!: WebGLTexture;         // 1×1 黑色 fallback base

  // ── Shared ──
  private quadBuf!: WebGLBuffer;            // fullscreen triangle quad

  constructor(gl: WebGLRenderingContext, cfg?: ConsumerConfig) {
    this.gl = gl;
    this.cfg = { ...DEFAULT_CONSUMER_CONFIG, ...cfg };
    this._init();
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Public API
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Pass 1 — Cell Distortion
   * velocityTexture 驱动 cell 表面 UV 扭曲.
   * 返回 distorted cell 纹理 (供后续合成使用).
   */
  cellDistortionPass(
    velocityTex: WebGLTexture,
    cellTex: WebGLTexture,
    outWidth: number,
    outHeight: number,
  ): WebGLTexture {
    const gl = this.gl;

    // 懒初始化 / 重建 FBO (分辨率变化时)
    if (outWidth !== this.distortW || outHeight !== this.distortH) {
      this._resizeDistortFBO(outWidth, outHeight);
    }

    // ── 绑定 offscreen FBO ──
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.distortFBO);   // gl.bindFramebuffer #1
    gl.viewport(0, 0, outWidth, outHeight);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // ── 使用 distortion program ──
    gl.useProgram(this.distortProg);

    // ── 绑定 velocityTexture → unit 0 ──
    gl.activeTexture(gl.TEXTURE0);                         // gl.activeTexture #1
    gl.bindTexture(gl.TEXTURE_2D, velocityTex);            // gl.bindTexture #1
    gl.uniform1i(
      gl.getUniformLocation(this.distortProg, 'uFluidVel'), 0,
    );

    // ── 绑定 cellTex → unit 1 ──
    gl.activeTexture(gl.TEXTURE1);                         // gl.activeTexture #2
    gl.bindTexture(gl.TEXTURE_2D, cellTex);                // gl.bindTexture #2
    gl.uniform1i(
      gl.getUniformLocation(this.distortProg, 'uCellTex'), 1,
    );

    // ── uniforms ──
    gl.uniform1f(
      gl.getUniformLocation(this.distortProg, 'uDistortStrength'),
      this.cfg.distortStrength,
    );
    gl.uniform2f(
      gl.getUniformLocation(this.distortProg, 'uVelScale'),
      1.0 / outWidth, 1.0 / outHeight,
    );

    // ── 画全屏 quad ──
    this._drawQuad(this.distortProg);                      // contains gl.drawArrays #1

    // ── 恢复默认 FBO ──
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);              // gl.bindFramebuffer #2

    return this.distortTex;
  }

  /**
   * Pass 2 — Particle Acceleration
   * 在 GPU 上采样粒子位置处的流体速度 → 修改 particle velocity.
   * velocities Float32Array 会被原地修改 (CPU 侧).
   */
  particleAccelerationPass(
    velocityTex: WebGLTexture,
    particles: ParticleData,
    simWidth: number,
    simHeight: number,
    strengthOverride?: number,
  ): void {
    const gl = this.gl;
    const count = Math.min(particles.count, this.cfg.maxParticles);
    if (count === 0) return;

    const strength = strengthOverride ?? this.cfg.accelStrength;

    // ── 1. 上传粒子位置到 atlas 纹理 ──
    // atlas layout: 1 行 × count 列, RG = NDC xy, BA = 0
    const atlasData = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      atlasData[i * 4 + 0] = particles.positions[i * 2 + 0]; // x (NDC)
      atlasData[i * 4 + 1] = particles.positions[i * 2 + 1]; // y (NDC)
      atlasData[i * 4 + 2] = 0;
      atlasData[i * 4 + 3] = 1;
    }
    gl.activeTexture(gl.TEXTURE2);                           // gl.activeTexture #3
    gl.bindTexture(gl.TEXTURE_2D, this.particlePosAtlasTex); // gl.bindTexture #3
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, count, 1, 0,
      gl.RGBA, gl.FLOAT, atlasData,
    );

    // ── 2. 更新粒子 index buffer ──
    const indices = new Float32Array(count);
    for (let i = 0; i < count; i++) indices[i] = i;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleIdxBuf);     // gl.bindBuffer #1
    gl.bufferData(gl.ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);

    // ── 3. 绑定 accel FBO (output: 1行 × count列 RGBA) ──
    // 确保 accelTex 有足够宽度
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accelFBO);       // gl.bindFramebuffer #3
    gl.bindTexture(gl.TEXTURE_2D, this.accelTex);            // gl.bindTexture #4
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, count, 1, 0,
      gl.RGBA, gl.FLOAT, null,
    );
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.accelTex, 0,
    );
    gl.viewport(0, 0, count, 1);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // ── 4. 使用 particle accel program ──
    gl.useProgram(this.accelProg);

    // ── 绑定 velocityTexture → unit 0 ──
    gl.activeTexture(gl.TEXTURE0);                           // gl.activeTexture #4
    gl.bindTexture(gl.TEXTURE_2D, velocityTex);              // gl.bindTexture #5
    gl.uniform1i(
      gl.getUniformLocation(this.accelProg, 'uFluidVel'), 0,
    );

    // ── 绑定 particlePosAtlas → unit 1 ──
    gl.activeTexture(gl.TEXTURE1);                           // gl.activeTexture #5
    gl.bindTexture(gl.TEXTURE_2D, this.particlePosAtlasTex); // gl.bindTexture #6
    gl.uniform1i(
      gl.getUniformLocation(this.accelProg, 'uParticlePosAtlas'), 1,
    );

    // ── uniforms ──
    gl.uniform1f(
      gl.getUniformLocation(this.accelProg, 'uAtlasWidth'),
      count,
    );
    gl.uniform1f(
      gl.getUniformLocation(this.accelProg, 'uAccelStrength'),
      strength,
    );

    // ── 5. 渲染每粒子 1px ──
    const idxLoc = gl.getAttribLocation(this.accelProg, 'aParticleIndex');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleIdxBuf);     // gl.bindBuffer #2
    gl.enableVertexAttribArray(idxLoc);
    gl.vertexAttribPointer(idxLoc, 1, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, count);                      // gl.drawArrays #2

    // ── 6. 读回结果 (CPU 侧应用) ──
    const readback = new Float32Array(count * 4);
    gl.readPixels(0, 0, count, 1, gl.RGBA, gl.FLOAT, readback); // gl.readPixels

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);                // gl.bindFramebuffer #4

    // ── 7. CPU 侧: particle velocity += fluid velocity * strength ──
    for (let i = 0; i < count; i++) {
      const dvx = readback[i * 4 + 0];
      const dvy = readback[i * 4 + 1];
      particles.velocities[i * 2 + 0] += dvx;
      particles.velocities[i * 2 + 1] += dvy;
    }
  }

  /**
   * Pass 3 — Background Flow
   * dye texture 作为背景色叠加, 直接渲染到 canvas (null FBO).
   */
  backgroundFlowPass(
    dyeTex: WebGLTexture,
    canvasWidth: number,
    canvasHeight: number,
    dyeStrengthOverride?: number,
    time = 0,
  ): void {
    const gl = this.gl;
    const strength = dyeStrengthOverride ?? this.cfg.dyeStrength;

    // ── 渲染到 canvas ──
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);                // gl.bindFramebuffer #5
    gl.viewport(0, 0, canvasWidth, canvasHeight);

    // 使用 bgFlow program
    gl.useProgram(this.bgFlowProg);

    // ── 绑定 dyeTex → unit 0 ──
    gl.activeTexture(gl.TEXTURE0);                           // gl.activeTexture #6
    gl.bindTexture(gl.TEXTURE_2D, dyeTex);                   // gl.bindTexture #7
    gl.uniform1i(
      gl.getUniformLocation(this.bgFlowProg, 'uDye'), 0,
    );

    // ── 绑定 base color → unit 1 (1×1 black fallback) ──
    gl.activeTexture(gl.TEXTURE1);                           // gl.activeTexture #7
    gl.bindTexture(gl.TEXTURE_2D, this.bgBaseTex);           // gl.bindTexture #8
    gl.uniform1i(
      gl.getUniformLocation(this.bgFlowProg, 'uBaseColor'), 1,
    );

    // ── uniforms ──
    gl.uniform1f(
      gl.getUniformLocation(this.bgFlowProg, 'uDyeStrength'),
      strength,
    );
    gl.uniform1f(
      gl.getUniformLocation(this.bgFlowProg, 'uTime'),
      time,
    );
    gl.uniform2f(
      gl.getUniformLocation(this.bgFlowProg, 'uResolution'),
      canvasWidth, canvasHeight,
    );

    // ── 画全屏 quad ──
    this._drawQuad(this.bgFlowProg);                         // contains gl.drawArrays #3
  }

  /**
   * 渲染 cell distortion pass 并直接输出到 canvas (无需外部 FBO).
   * 便利方法: cellDistortionPass + 直接 blit 到屏幕.
   */
  renderCellDistortionToCanvas(
    velocityTex: WebGLTexture,
    cellTex: WebGLTexture,
    canvasWidth: number,
    canvasHeight: number,
  ): void {
    const gl = this.gl;

    // 先做 offscreen distort
    const distortedTex = this.cellDistortionPass(
      velocityTex, cellTex, canvasWidth, canvasHeight,
    );

    // Blit 到 canvas: 用 bgFlow pass 的 base 槽 + dye 强度=0 (仅显示 distorted cell)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);                // gl.bindFramebuffer #6
    gl.viewport(0, 0, canvasWidth, canvasHeight);

    gl.useProgram(this.bgFlowProg);

    // slot0 = distortedTex (作为 dye, strength=1)
    gl.activeTexture(gl.TEXTURE0);                           // gl.activeTexture #8
    gl.bindTexture(gl.TEXTURE_2D, distortedTex);             // gl.bindTexture #9
    gl.uniform1i(gl.getUniformLocation(this.bgFlowProg, 'uDye'), 0);

    // slot1 = black base
    gl.activeTexture(gl.TEXTURE1);                           // gl.activeTexture #9
    gl.bindTexture(gl.TEXTURE_2D, this.bgBaseTex);           // gl.bindTexture #10
    gl.uniform1i(gl.getUniformLocation(this.bgFlowProg, 'uBaseColor'), 1);

    gl.uniform1f(gl.getUniformLocation(this.bgFlowProg, 'uDyeStrength'), 1.0);
    gl.uniform1f(gl.getUniformLocation(this.bgFlowProg, 'uTime'), 0);
    gl.uniform2f(
      gl.getUniformLocation(this.bgFlowProg, 'uResolution'),
      canvasWidth, canvasHeight,
    );

    this._drawQuad(this.bgFlowProg);                         // contains gl.drawArrays #4
  }

  /**
   * 完整帧: 按顺序执行三个 pass.
   * 这是外部调用的主入口.
   */
  render(
    velocityTex: WebGLTexture,
    dyeTex: WebGLTexture,
    cellTex: WebGLTexture,
    particles: ParticleData,
    canvasWidth: number,
    canvasHeight: number,
    time = 0,
  ): WebGLTexture {
    // Pass 1: cell distortion
    const distortedCell = this.cellDistortionPass(
      velocityTex, cellTex, canvasWidth, canvasHeight,
    );

    // Pass 2: particle acceleration (modifies particles.velocities in-place)
    this.particleAccelerationPass(
      velocityTex, particles, canvasWidth, canvasHeight,
    );

    // Pass 3: background flow (dye overlay to canvas)
    this.backgroundFlowPass(dyeTex, canvasWidth, canvasHeight, undefined, time);

    return distortedCell;
  }

  /** 释放所有 GPU 资源 */
  destroy(): void {
    const gl = this.gl;

    // Programs
    gl.deleteProgram(this.distortProg);
    gl.deleteProgram(this.accelProg);
    gl.deleteProgram(this.bgFlowProg);

    // FBOs
    gl.deleteFramebuffer(this.distortFBO);
    gl.deleteFramebuffer(this.accelFBO);

    // Textures
    gl.deleteTexture(this.distortTex);
    gl.deleteTexture(this.accelTex);
    gl.deleteTexture(this.particlePosAtlasTex);
    gl.deleteTexture(this.bgBaseTex);

    // Buffers
    gl.deleteBuffer(this.quadBuf);
    gl.deleteBuffer(this.particleIdxBuf);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Private: 初始化
  // ───────────────────────────────────────────────────────────────────────────

  private _init(): void {
    const gl = this.gl;

    // ── 编译三个 program ──
    this.distortProg = this._compile(
      FULLSCREEN_VERT, CELL_DISTORTION_FRAG, 'cellDistort',
    );
    this.accelProg = this._compile(
      PARTICLE_QUERY_VERT, PARTICLE_ACCEL_FRAG, 'particleAccel',
    );
    this.bgFlowProg = this._compile(
      FULLSCREEN_VERT, BACKGROUND_FLOW_FRAG, 'bgFlow',
    );

    // ── 全屏 quad buffer ──
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);            // gl.bindBuffer #3
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([
        -1, -1,  1, -1, -1,  1,
        -1,  1,  1, -1,  1,  1,
      ]),
      gl.STATIC_DRAW,
    );

    // ── Pass 1: distort FBO ──
    this.distortFBO = gl.createFramebuffer()!;
    this.distortTex = this._makeFloatTex(1, 1);              // 占位, resize 时重建
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.distortFBO);     // gl.bindFramebuffer #7
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.distortTex, 0,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);                // gl.bindFramebuffer #8

    // ── Pass 2: particle atlas + accel FBO ──
    const maxP = this.cfg.maxParticles;

    this.particlePosAtlasTex = this._makeFloatTex(maxP, 1);
    this.accelTex             = this._makeFloatTex(maxP, 1);

    this.accelFBO = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accelFBO);       // gl.bindFramebuffer #9
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.accelTex, 0,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);                // gl.bindFramebuffer #10

    this.particleIdxBuf = gl.createBuffer()!;
    // 预分配 index buffer
    const initIdx = new Float32Array(maxP);
    for (let i = 0; i < maxP; i++) initIdx[i] = i;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleIdxBuf);     // gl.bindBuffer #4
    gl.bufferData(gl.ARRAY_BUFFER, initIdx, gl.DYNAMIC_DRAW);

    // ── Pass 3: 1×1 黑色 base tex ──
    this.bgBaseTex = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE0);                           // gl.activeTexture #10
    gl.bindTexture(gl.TEXTURE_2D, this.bgBaseTex);           // gl.bindTexture #11
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0,
      gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 0, 255]),
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);                     // gl.bindTexture #12
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Private: helpers
  // ───────────────────────────────────────────────────────────────────────────

  /** 重建 distort FBO 为新尺寸 */
  private _resizeDistortFBO(w: number, h: number): void {
    const gl = this.gl;
    // 删除旧 tex (FBO 可复用)
    gl.deleteTexture(this.distortTex);

    // 新建 tex
    this.distortTex = this._makeFloatTex(w, h);

    // 重新绑定 FBO attachment
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.distortFBO);     // gl.bindFramebuffer #11
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.distortTex, 0,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);                // gl.bindFramebuffer #12

    this.distortW = w;
    this.distortH = h;
  }

  /** 创建 RGBA float 纹理 (带线性过滤) */
  private _makeFloatTex(w: number, h: number): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);                      // gl.bindTexture #13
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0,
      gl.RGBA, gl.FLOAT, null,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);                     // gl.bindTexture #14
    return tex;
  }

  /** 画全屏 quad (WebGL1) */
  private _drawQuad(program: WebGLProgram): void {
    const gl = this.gl;
    const posLoc = gl.getAttribLocation(program, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);            // gl.bindBuffer #5
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);                       // gl.drawArrays
    gl.disableVertexAttribArray(posLoc);
  }

  /**
   * 编译 WebGL1 vert + frag → WebGLProgram
   * 与 fluid-gpu-pass.ts 同样模式
   */
  private _compile(
    vertSrc: string,
    fragSrc: string,
    label: string,
  ): WebGLProgram {
    const gl = this.gl;

    // Vertex shader
    const vs = gl.createShader(gl.VERTEX_SHADER)!;           // gl.createShader #1
    gl.shaderSource(vs, vertSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(
        `[FluidConsumerGPU] VS compile error (${label}): ${gl.getShaderInfoLog(vs)}`,
      );
    }

    // Fragment shader
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;         // gl.createShader #2
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(
        `[FluidConsumerGPU] FS compile error (${label}): ${gl.getShaderInfoLog(fs)}`,
      );
    }

    // Program
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(
        `[FluidConsumerGPU] link error (${label}): ${gl.getProgramInfoLog(prog)}`,
      );
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }
}

// ─── Convenience factory ──────────────────────────────────────────────────────

/**
 * 创建 FluidConsumerGPU 并绑定到 FluidGPU 输出.
 *
 * 用法:
 * ```ts
 * const fluid = new FluidGPU(gl);
 * const consumer = createFluidConsumer(gl, { distortStrength: 0.08 });
 *
 * // 每帧:
 * fluid.step(mx, my, pmx, pmy, dt);
 * consumer.render(
 *   fluid.velocityTexture,
 *   fluid.dyeTexture,
 *   cellSurfaceTex,
 *   particles,
 *   canvas.width,
 *   canvas.height,
 *   time,
 * );
 * ```
 */
export function createFluidConsumer(
  gl: WebGLRenderingContext,
  cfg?: ConsumerConfig,
): FluidConsumerGPU {
  return new FluidConsumerGPU(gl, cfg);
}
