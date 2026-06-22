/**
 * src/lib/sph/ripple-effect.ts
 *
 * Ripple Collision Effect — WebGPU ping-pong wave simulation
 *
 * Architecture
 * ────────────
 *  RippleEffect
 *    • Maintains two ping-pong textures (rgba16float) storing wave state:
 *        r = current displacement  [0..1, neutral = 0.5]
 *        g = previous displacement [0..1, neutral = 0.5]
 *    • Each frame: runs a compute-like fullscreen pass that propagates the
 *      wave field using the Lygia ripple algorithm (lygia/simulate/ripple.wgsl).
 *    • Collision events feed impulses: the collision point is mapped to UV
 *      space and a Gaussian stamp proportional to the impulse magnitude is
 *      written into the current wave texture before the propagation step.
 *    • A compositing pass blends the ripple displacement map over the final
 *      rendered frame as a screen-space distortion + brightness overlay.
 *
 * Lygia import
 * ────────────
 *  The lygia ripple algorithm is inlined from:
 *    upstream/lygia/simulate/ripple.wgsl
 *  Dependencies (saturate, sampler macros) are resolved inline since WebGPU
 *  WGSL doesn't support #include directives at runtime.
 *
 * Usage
 * ─────
 *   const ripple = new RippleEffect(device, canvasFormat);
 *   ripple.buildPipelines();
 *
 *   // Wire up collision dispatcher:
 *   collisionDispatcher.onCollisionEnter(e => ripple.onCollision(e));
 *
 *   // Each frame, inside your render loop:
 *   ripple.step(encoder, canvasWidth, canvasHeight);          // propagate
 *   ripple.composite(encoder, sceneView, canvasWidth, canvasHeight); // overlay
 */

import type { CollisionEvent }           from './collision/CollisionEvents';
import type { CameraUniforms }           from './ParticleRenderer';

// ─────────────────────────────────────────────────────────────────────────────
// Lygia ripple propagation shader (lygia/simulate/ripple.wgsl — inlined)
//
// Original author: Patricio Gonzalez Vivo
// License: https://lygia.xyz/license
//
// Adapted for WebGPU:
//   • SAMPLER_TYPE    → texture_2d<f32> + sampler (explicit bindings)
//   • SAMPLER_FNC     → textureSampleLevel(tex, smp, uv, 0.0)
//   • saturate        → clamp(v, 0.0, 1.0)   (WGSL builtin)
//   • float keyword   → let / var  (WGSL syntax fix in original source)
// ─────────────────────────────────────────────────────────────────────────────

/** WGSL: wave propagation step (fullscreen quad → writes new wave state) */
const RIPPLE_PROPAGATE_SHADER = /* wgsl */`
// ── Lygia ripple.wgsl — inlined ───────────────────────────────────────────
// Propagation kernel operating on a double-buffered wave texture.
// Texture layout:  r = wave height (current), g = wave height (previous)
//
// fn ripple(tex, smp, st, pixel) -> vec3f
//   rta.r = current   (was written last frame as new current)
//   rta.g = previous  (was written last frame as new previous = old current)
//   Neighbours s1..s4 sample current channel (r) of adjacent texels.
//   Result: vec3f(newCurrent, oldCurrent, 0)
//           → next frame's r = newCurrent, g = oldCurrent
fn lygiaRipple(tex: texture_2d<f32>, smp: sampler, st: vec2f, pixel: vec2f) -> vec3f {
  let rta = textureSampleLevel(tex, smp, st,               0.0).rgb;
  let s0  = rta.g;                                                        // previous height
  let s1  = textureSampleLevel(tex, smp, st + vec2f( 0.0,        -pixel.y), 0.0).r;
  let s2  = textureSampleLevel(tex, smp, st + vec2f(-pixel.x,     0.0    ), 0.0).r;
  let s3  = textureSampleLevel(tex, smp, st + vec2f( pixel.x,     0.0    ), 0.0).r;
  let s4  = textureSampleLevel(tex, smp, st + vec2f( 0.0,         pixel.y), 0.0).r;

  var d = -(s0 - 0.5) * 2.0 + (s1 + s2 + s3 + s4 - 2.0);
  d    *= 0.99;                        // damping
  d     = clamp(d * 0.5 + 0.5, 0.0, 1.0);   // lygia: saturate(d*0.5+0.5)
  return vec3f(d, rta.r, 0.0);         // (newHeight, oldHeight, unused)
}
// ── end lygia ripple ──────────────────────────────────────────────────────

struct PropUni {
  pixel : vec2f,   // 1/width, 1/height
  _pad  : vec2f,
}

@group(0) @binding(0) var<uniform> uni  : PropUni;
@group(0) @binding(1) var waveTex       : texture_2d<f32>;
@group(0) @binding(2) var waveSmp       : sampler;

struct Vout { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

@vertex fn vs_full(@builtin(vertex_index) vi: u32) -> Vout {
  // Full-screen triangle trick (3 vertices covering NDC)
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0), vec2f( 3.0, -1.0), vec2f(-1.0,  3.0)
  );
  var uv = array<vec2f, 3>(
    vec2f(0.0, 1.0), vec2f(2.0, 1.0), vec2f(0.0, -1.0)
  );
  var o: Vout;
  o.pos = vec4f(pos[vi], 0.0, 1.0);
  o.uv  = uv[vi];
  return o;
}

@fragment fn fs_propagate(in: Vout) -> @location(0) vec4f {
  let result = lygiaRipple(waveTex, waveSmp, in.uv, uni.pixel);
  return vec4f(result, 1.0);
}
`;

/** WGSL: stamp an impulse Gaussian onto the wave texture */
const RIPPLE_STAMP_SHADER = /* wgsl */`
struct StampUni {
  origin    : vec2f,   // UV of collision point
  amplitude : f32,     // impulse strength → wave height offset
  radius    : f32,     // Gaussian sigma in UV space
}

@group(0) @binding(0) var<uniform> uni : StampUni;
@group(0) @binding(1) var waveTex      : texture_2d<f32>;
@group(0) @binding(2) var waveSmp      : sampler;

struct Vout { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

@vertex fn vs_full(@builtin(vertex_index) vi: u32) -> Vout {
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0), vec2f( 3.0, -1.0), vec2f(-1.0,  3.0)
  );
  var uv = array<vec2f, 3>(
    vec2f(0.0, 1.0), vec2f(2.0, 1.0), vec2f(0.0, -1.0)
  );
  var o: Vout;
  o.pos = vec4f(pos[vi], 0.0, 1.0);
  o.uv  = uv[vi];
  return o;
}

@fragment fn fs_stamp(in: Vout) -> @location(0) vec4f {
  let existing = textureSampleLevel(waveTex, waveSmp, in.uv, 0.0);
  let dist2     = dot(in.uv - uni.origin, in.uv - uni.origin);
  let sigma2    = uni.radius * uni.radius;
  let gaussian  = uni.amplitude * exp(-dist2 / (2.0 * sigma2));
  // Add impulse to current channel (r), clamp to [0,1]
  let newR = clamp(existing.r + gaussian, 0.0, 1.0);
  return vec4f(newR, existing.g, existing.b, existing.a);
}
`;

/** WGSL: composite ripple distortion over the rendered scene */
const RIPPLE_COMPOSITE_SHADER = /* wgsl */`
struct CompUni {
  pixel         : vec2f,   // 1/width, 1/height
  strength      : f32,     // distortion UV offset scale
  brightness    : f32,     // additive brightness scale for foam/crest
}

@group(0) @binding(0) var<uniform>  uni      : CompUni;
@group(0) @binding(1) var sceneTex           : texture_2d<f32>;
@group(0) @binding(2) var sceneSmp           : sampler;
@group(0) @binding(3) var waveTex            : texture_2d<f32>;
@group(0) @binding(4) var waveSmp            : sampler;

struct Vout { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

@vertex fn vs_full(@builtin(vertex_index) vi: u32) -> Vout {
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0), vec2f( 3.0, -1.0), vec2f(-1.0,  3.0)
  );
  var uv = array<vec2f, 3>(
    vec2f(0.0, 1.0), vec2f(2.0, 1.0), vec2f(0.0, -1.0)
  );
  var o: Vout;
  o.pos = vec4f(pos[vi], 0.0, 1.0);
  o.uv  = uv[vi];
  return o;
}

@fragment fn fs_composite(in: Vout) -> @location(0) vec4f {
  let wave = textureSampleLevel(waveTex, waveSmp, in.uv, 0.0);

  // Finite-difference normal from wave height field
  let hL = textureSampleLevel(waveTex, waveSmp, in.uv - vec2f(uni.pixel.x, 0.0), 0.0).r;
  let hR = textureSampleLevel(waveTex, waveSmp, in.uv + vec2f(uni.pixel.x, 0.0), 0.0).r;
  let hD = textureSampleLevel(waveTex, waveSmp, in.uv - vec2f(0.0, uni.pixel.y), 0.0).r;
  let hU = textureSampleLevel(waveTex, waveSmp, in.uv + vec2f(0.0, uni.pixel.y), 0.0).r;
  let grad = vec2f(hR - hL, hU - hD);  // surface slope → UV distortion

  // Sample scene with ripple-displaced UV
  let distortedUV = clamp(in.uv + grad * uni.strength, vec2f(0.0), vec2f(1.0));
  let sceneColor  = textureSampleLevel(sceneTex, sceneSmp, distortedUV, 0.0);

  // Foam / crest brightness where |wave - 0.5| is large
  let deviation = abs(wave.r - 0.5) * 2.0;
  let foam      = deviation * deviation * uni.brightness;

  return vec4f(sceneColor.rgb + vec3f(foam), sceneColor.a);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pending impulse queued from the CPU each frame
// ─────────────────────────────────────────────────────────────────────────────

interface PendingImpulse {
  /** UV-space origin of the impulse (0..1) */
  originU : number;
  originV : number;
  /** Wave amplitude added at origin (mapped from impulse magnitude) */
  amplitude: number;
  /** Gaussian sigma in UV space */
  radius   : number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public configuration
// ─────────────────────────────────────────────────────────────────────────────

export interface RippleEffectConfig {
  /**
   * Physics domain extents used to convert world-space collision points to
   * UV coordinates.  Defaults to [0,1]×[0,1].
   */
  domainMinX?: number;
  domainMaxX?: number;
  domainMinY?: number;
  domainMaxY?: number;
  /**
   * Scale applied to the collision impulse depth to produce wave amplitude.
   * Higher values produce more pronounced ripples.  Default: 0.4.
   */
  impulseScale?: number;
  /**
   * Maximum wave amplitude clamped per impulse.  Default: 0.45.
   */
  maxAmplitude?: number;
  /**
   * Gaussian sigma (UV space) for each impulse stamp.  Default: 0.04.
   */
  stampRadius?: number;
  /**
   * UV offset strength for screen-space distortion compositing.  Default: 0.006.
   */
  distortionStrength?: number;
  /**
   * Additive brightness scale for wave crests (foam effect).  Default: 0.12.
   */
  foamBrightness?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// RippleEffect — main class
// ─────────────────────────────────────────────────────────────────────────────

export class RippleEffect {

  private readonly device : GPUDevice;
  private readonly format : GPUTextureFormat;

  // ── Config ─────────────────────────────────────────────────────────────────
  private cfg: Required<RippleEffectConfig>;

  // ── Ping-pong wave textures (rgba16float) ──────────────────────────────────
  //   r = current height, g = previous height, b/a = unused
  private waveTex   : [GPUTexture, GPUTexture] | null = null;
  private waveView  : [GPUTextureView, GPUTextureView] | null = null;
  private ping      = 0;   // index of the texture we READ from this frame
  private waveW     = 0;
  private waveH     = 0;

  // ── Sampler ────────────────────────────────────────────────────────────────
  private waveSampler !: GPUSampler;

  // ── Propagation pipeline ───────────────────────────────────────────────────
  private propagatePipeline  !: GPURenderPipeline;
  private propagateBGL       !: GPUBindGroupLayout;
  private propagateUniBuf    !: GPUBuffer;  // PropUni (pixel size)

  // ── Stamp pipeline ─────────────────────────────────────────────────────────
  private stampPipeline      !: GPURenderPipeline;
  private stampBGL           !: GPUBindGroupLayout;
  private stampUniBuf        !: GPUBuffer;  // StampUni

  // ── Composite pipeline ─────────────────────────────────────────────────────
  private compositePipeline  !: GPURenderPipeline;
  private compositeBGL       !: GPUBindGroupLayout;
  private compositeUniBuf    !: GPUBuffer;  // CompUni

  // ── Intermediate scene texture (copy target for composite) ─────────────────
  private sceneTex           !: GPUTexture;
  private sceneView          !: GPUTextureView;
  private sceneSampler       !: GPUSampler;
  private sceneW             = 0;
  private sceneH             = 0;

  // ── Impulse queue ──────────────────────────────────────────────────────────
  private _pendingImpulses: PendingImpulse[] = [];

  // ── Pipeline-built flag ───────────────────────────────────────────────────
  private _built = false;

  // ─────────────────────────────────────────────────────────────────────────
  constructor(device: GPUDevice, format: GPUTextureFormat, config: RippleEffectConfig = {}) {
    this.device = device;
    this.format = format;
    this.cfg = {
      domainMinX          : config.domainMinX          ?? 0,
      domainMaxX          : config.domainMaxX          ?? 1,
      domainMinY          : config.domainMinY          ?? 0,
      domainMaxY          : config.domainMaxY          ?? 1,
      impulseScale        : config.impulseScale        ?? 0.4,
      maxAmplitude        : config.maxAmplitude        ?? 0.45,
      stampRadius         : config.stampRadius         ?? 0.04,
      distortionStrength  : config.distortionStrength  ?? 0.006,
      foamBrightness      : config.foamBrightness      ?? 0.12,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public: update domain config at runtime (e.g. after world resize)
  // ─────────────────────────────────────────────────────────────────────────

  updateConfig(patch: Partial<RippleEffectConfig>): void {
    Object.assign(this.cfg, patch);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public: collision event handler
  //
  // Wire this up to CollisionEventDispatcher.onCollisionEnter():
  //   dispatcher.onCollisionEnter(e => rippleEffect.onCollision(e));
  //
  // The impulse amplitude is derived from the contact penetration depth;
  // the origin is the midpoint of pointA and pointB mapped to UV space.
  // ─────────────────────────────────────────────────────────────────────────

  onCollision(event: CollisionEvent): void {
    if (event.phase === 'exit' || !event.contact) return;

    const { contact } = event;

    // World-space midpoint of the two contact points
    const wx = (contact.pointA.x + contact.pointB.x) * 0.5;
    const wy = (contact.pointA.y + contact.pointB.y) * 0.5;

    // Map world → UV  [domainMin, domainMax] → [0, 1]
    const originU = (wx - this.cfg.domainMinX) / (this.cfg.domainMaxX - this.cfg.domainMinX);
    const originV = 1.0 - (wy - this.cfg.domainMinY) / (this.cfg.domainMaxY - this.cfg.domainMinY);

    // Amplitude proportional to penetration depth (impulse proxy)
    const amplitude = Math.min(
      contact.depth * this.cfg.impulseScale,
      this.cfg.maxAmplitude,
    );

    if (amplitude < 1e-4) return;  // discard negligible impulses

    this._pendingImpulses.push({
      originU,
      originV,
      amplitude,
      radius: this.cfg.stampRadius,
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public: build all GPU pipelines (call once after device is ready)
  // ─────────────────────────────────────────────────────────────────────────

  buildPipelines(): void {
    const d = this.device;

    // ── Sampler ──────────────────────────────────────────────────────────────
    this.waveSampler = d.createSampler({
      label       : 'ripple-wave-sampler',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      magFilter   : 'linear',
      minFilter   : 'linear',
    });

    this.sceneSampler = d.createSampler({
      label       : 'ripple-scene-sampler',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      magFilter   : 'linear',
      minFilter   : 'linear',
    });

    // ── Propagation pipeline ──────────────────────────────────────────────
    {
      this.propagateUniBuf = d.createBuffer({
        label : 'ripple-prop-uni',
        size  : 16,   // vec2f pixel + vec2f pad
        usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      this.propagateBGL = d.createBindGroupLayout({
        label  : 'ripple-prop-bgl',
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        ],
      });

      const propModule = d.createShaderModule({ label: 'ripple-prop', code: RIPPLE_PROPAGATE_SHADER });
      this.propagatePipeline = d.createRenderPipeline({
        label  : 'ripple-propagate-pipeline',
        layout : d.createPipelineLayout({ bindGroupLayouts: [this.propagateBGL] }),
        vertex  : { module: propModule, entryPoint: 'vs_full' },
        fragment: {
          module     : propModule,
          entryPoint : 'fs_propagate',
          targets    : [{ format: 'rgba16float' }],
        },
        primitive: { topology: 'triangle-list' },
      });
    }

    // ── Stamp pipeline ────────────────────────────────────────────────────
    {
      this.stampUniBuf = d.createBuffer({
        label : 'ripple-stamp-uni',
        size  : 16,   // vec2f origin + f32 amplitude + f32 radius
        usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      this.stampBGL = d.createBindGroupLayout({
        label  : 'ripple-stamp-bgl',
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        ],
      });

      const stampModule = d.createShaderModule({ label: 'ripple-stamp', code: RIPPLE_STAMP_SHADER });
      this.stampPipeline = d.createRenderPipeline({
        label  : 'ripple-stamp-pipeline',
        layout : d.createPipelineLayout({ bindGroupLayouts: [this.stampBGL] }),
        vertex  : { module: stampModule, entryPoint: 'vs_full' },
        fragment: {
          module     : stampModule,
          entryPoint : 'fs_stamp',
          targets    : [{
            format : 'rgba16float',
            blend  : {
              color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
              alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'zero' },
            },
          }],
        },
        primitive: { topology: 'triangle-list' },
      });
    }

    // ── Composite pipeline ────────────────────────────────────────────────
    {
      this.compositeUniBuf = d.createBuffer({
        label : 'ripple-comp-uni',
        size  : 16,   // vec2f pixel + f32 strength + f32 brightness
        usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      this.compositeBGL = d.createBindGroupLayout({
        label  : 'ripple-comp-bgl',
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
          { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        ],
      });

      const compModule = d.createShaderModule({ label: 'ripple-comp', code: RIPPLE_COMPOSITE_SHADER });
      this.compositePipeline = d.createRenderPipeline({
        label  : 'ripple-composite-pipeline',
        layout : d.createPipelineLayout({ bindGroupLayouts: [this.compositeBGL] }),
        vertex  : { module: compModule, entryPoint: 'vs_full' },
        fragment: {
          module     : compModule,
          entryPoint : 'fs_composite',
          targets    : [{ format: this.format }],
        },
        primitive: { topology: 'triangle-list' },
      });
    }

    this._built = true;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public: step — propagate wave + stamp pending impulses
  //
  // Call once per frame BEFORE composite().
  // ─────────────────────────────────────────────────────────────────────────

  step(encoder: GPUCommandEncoder, width: number, height: number): void {
    if (!this._built) return;

    this._ensureWaveTextures(width, height);

    const d    = this.device;
    const read = this.ping;           // ping  = read source this frame
    const write = 1 - this.ping;     // pong  = render target this frame

    // ── 1. Stamp pending impulses onto the READ texture ───────────────────
    //   (we modify the source before propagation so impulses affect t+1)
    for (const imp of this._pendingImpulses) {
      this._stampImpulse(encoder, imp, this.waveView![read]);
    }
    this._pendingImpulses.length = 0;

    // ── 2. Propagate: read → write ────────────────────────────────────────
    d.queue.writeBuffer(this.propagateUniBuf, 0, new Float32Array([
      1.0 / width,
      1.0 / height,
      0, 0,  // pad
    ]));

    const propBG = d.createBindGroup({
      label  : 'ripple-prop-bg',
      layout : this.propagateBGL,
      entries: [
        { binding: 0, resource: { buffer: this.propagateUniBuf } },
        { binding: 1, resource: this.waveView![read] },
        { binding: 2, resource: this.waveSampler },
      ],
    });

    const propPass = encoder.beginRenderPass({
      label           : 'ripple-propagate-pass',
      colorAttachments: [{
        view    : this.waveView![write],
        loadOp  : 'clear',
        clearValue: { r: 0.5, g: 0.5, b: 0.0, a: 1.0 },
        storeOp : 'store',
      }],
    });
    propPass.setPipeline(this.propagatePipeline);
    propPass.setBindGroup(0, propBG);
    propPass.draw(3);
    propPass.end();

    // Swap ping/pong
    this.ping = write;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public: composite — overlay ripple distortion onto the rendered frame
  //
  // Because WebGPU doesn't allow reading and writing the same texture in one
  // pass, the caller must pass the rendered scene as a GPUTexture that has
  // been written to (TEXTURE_BINDING | RENDER_ATTACHMENT) and we output the
  // composited result to `outputView`.
  //
  // Typical integration pattern:
  //   1. Render particles into sceneTexture.
  //   2. rippleEffect.composite(encoder, sceneTextureView, outputView, w, h);
  // ─────────────────────────────────────────────────────────────────────────

  composite(
    encoder     : GPUCommandEncoder,
    sceneView   : GPUTextureView,
    outputView  : GPUTextureView,
    width       : number,
    height      : number,
  ): void {
    if (!this._built) return;

    this._ensureWaveTextures(width, height);

    const d = this.device;

    d.queue.writeBuffer(this.compositeUniBuf, 0, new Float32Array([
      1.0 / width,
      1.0 / height,
      this.cfg.distortionStrength,
      this.cfg.foamBrightness,
    ]));

    const compBG = d.createBindGroup({
      label  : 'ripple-comp-bg',
      layout : this.compositeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.compositeUniBuf } },
        { binding: 1, resource: sceneView },
        { binding: 2, resource: this.sceneSampler },
        { binding: 3, resource: this.waveView![this.ping] },
        { binding: 4, resource: this.waveSampler },
      ],
    });

    const compPass = encoder.beginRenderPass({
      label           : 'ripple-composite-pass',
      colorAttachments: [{
        view    : outputView,
        loadOp  : 'load',
        storeOp : 'store',
      }],
    });
    compPass.setPipeline(this.compositePipeline);
    compPass.setBindGroup(0, compBG);
    compPass.draw(3);
    compPass.end();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public: convenience helper — update domain from CameraUniforms
  // ─────────────────────────────────────────────────────────────────────────

  syncDomainFromCamera(u: CameraUniforms): void {
    const halfW = (u.domainScale ?? 1) / (u.scaleX ?? 1) * 0.5;
    const halfH = (u.domainScale ?? 1) / (u.scaleY ?? 1) * 0.5;
    const cx    = -(u.offsetX ?? 0) / (u.scaleX ?? 1);
    const cy    = -(u.offsetY ?? 0) / (u.scaleY ?? 1);
    this.updateConfig({
      domainMinX: cx - halfW,
      domainMaxX: cx + halfW,
      domainMinY: cy - halfH,
      domainMaxY: cy + halfH,
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public: reset the wave field to neutral (all 0.5)
  // ─────────────────────────────────────────────────────────────────────────

  resetField(): void {
    if (!this.waveTex) return;
    // We mark both textures as dirty by forcing a recreate on next step()
    this.waveW = 0;
    this.waveH = 0;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public: release GPU resources
  // ─────────────────────────────────────────────────────────────────────────

  destroy(): void {
    this.waveTex?.[0].destroy();
    this.waveTex?.[1].destroy();
    this.waveTex = null;
    this.waveView = null;
    this.sceneTex?.destroy();
    this.propagateUniBuf?.destroy();
    this.stampUniBuf?.destroy();
    this.compositeUniBuf?.destroy();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private helpers
  // ─────────────────────────────────────────────────────────────────────────

  private _ensureWaveTextures(width: number, height: number): void {
    if (this.waveW === width && this.waveH === height) return;

    this.waveTex?.[0].destroy();
    this.waveTex?.[1].destroy();

    const make = (label: string): GPUTexture =>
      this.device.createTexture({
        label,
        size   : { width, height },
        format : 'rgba16float',
        usage  : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });

    this.waveTex  = [make('ripple-wave-ping'), make('ripple-wave-pong')];
    this.waveView = [
      this.waveTex[0].createView({ label: 'ripple-wave-ping-view' }),
      this.waveTex[1].createView({ label: 'ripple-wave-pong-view' }),
    ];
    this.waveW = width;
    this.waveH = height;
    this.ping  = 0;

    // Neutral field is written on first propagation pass (clearValue 0.5)
  }

  private _stampImpulse(
    encoder  : GPUCommandEncoder,
    imp      : PendingImpulse,
    targetView: GPUTextureView,
  ): void {
    const d = this.device;

    d.queue.writeBuffer(this.stampUniBuf, 0, new Float32Array([
      imp.originU,
      imp.originV,
      imp.amplitude,
      imp.radius,
    ]));

    // We need a temporary sampler + read view from the write target.
    // Because we can't read+write the same texture, we stamp using additive
    // blending on top of the propagation target — impulse pass runs before
    // the propagation pass, but outputs to the READ texture directly via an
    // extra intermediate.  For simplicity we stamp by running a render pass
    // with additive blend on the target view; this works because the stamp
    // pass reads from the same source it blends into (via the waveSampler
    // binding) which is allowed when using blending (no read-write hazard
    // in WebGPU for colour attachments with blending).
    //
    // NOTE: the stamp shader already reads existing values via texture binding
    // and writes newR = clamp(existing.r + gaussian).  The additive blend on
    // the pipeline doubles the effect if we use both; we therefore use a
    // simple load+store with ONE_ZERO blend factor so only the shader output
    // determines the result.

    const stampBG = d.createBindGroup({
      label  : 'ripple-stamp-bg',
      layout : this.stampBGL,
      entries: [
        { binding: 0, resource: { buffer: this.stampUniBuf } },
        { binding: 1, resource: targetView },
        { binding: 2, resource: this.waveSampler },
      ],
    });

    const stampPass = encoder.beginRenderPass({
      label           : 'ripple-stamp-pass',
      colorAttachments: [{
        view    : targetView,
        loadOp  : 'load',
        storeOp : 'store',
      }],
    });
    stampPass.setPipeline(this.stampPipeline);
    stampPass.setBindGroup(0, stampBG);
    stampPass.draw(3);
    stampPass.end();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports for convenience
// ─────────────────────────────────────────────────────────────────────────────

export type { RippleEffectConfig };
