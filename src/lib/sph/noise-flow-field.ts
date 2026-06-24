// === src/lib/sph/noise-flow-field.ts ===
// noise-flow-field.ts --- Curl-noise + FBM flow field for SPH particles
//
// Imports lygia's fbm.wgsl and curl.wgsl concepts and inlines a self-contained
// WGSL implementation that can be appended into the SPH force shader.
//
// Effects achievable via NoiseFlowFieldConfig:
//   'smoke'  --- gentle FBM-modulated curl, low frequency, soft eddies
//   'aurora' --- layered octave FBM, high vertical drift, colour-sweep time warp
//   'water'  --- fast curl, medium frequency, gentle gravity-aligned drift
//
// Integration with SPHGPUOrchestrator:
//   Call NoiseFlowField.encodeForceOverlay(encoder, n) *after* encodeForces()
//   in your sim loop.  It reads posX/posY from the GPUBufferSet and writes an
//   additive force contribution into forceX/forceY.




// ---------------------------------------------------------------------------
// Effect presets
// ---------------------------------------------------------------------------




import type { GPUBufferSet, SimParams } from './types';
import { WORKGROUP_SIZE } from './types';

export type NoiseEffect = "smoke" | "aurora" | "water";

export interface NoiseFlowFieldConfig {
  /** Visual effect preset. Controls frequency, octaves, strength, drift. */
  effect: NoiseEffect;
  /** Global strength multiplier (default 1). */
  strength?: number;
  /** Simulation time in seconds --- drives the noise animation. */
  time?: number;
}

// Per-effect parameter packs shipped to the GPU via a uniform buffer.
const EFFECT_PARAMS: Record<
  NoiseEffect,
  { freq: number; octaves: number; baseStrength: number; driftX: number; driftY: number; timeScale: number }
> = {
  smoke:  { freq: 0.8,  octaves: 4, baseStrength: 0.6,  driftX:  0.05, driftY: -0.15, timeScale: 0.25 },
  aurora: { freq: 0.4,  octaves: 6, baseStrength: 1.2,  driftX:  0.2,  driftY: -0.60, timeScale: 0.40 },
  water:  { freq: 1.4,  octaves: 3, baseStrength: 0.9,  driftX:  0.0,  driftY: -0.05, timeScale: 0.60 },
};

// ---------------------------------------------------------------------------
// WGSL: simplex noise basis (based on lygia snoise2 --- self-contained)
// ---------------------------------------------------------------------------
// Inlined from upstream/lygia/generative/snoise.wgsl + permute helpers so
// the shader compiles standalone without the lygia preprocessor.

const WGSL_SIMPLEX_BASIS = /* wgsl */`
// ------ Lygia-derived simplex permute helpers ---------------------------------------------------------------------------------------------------------
fn sn_mod289_2(x: vec2f) -> vec2f { return x - floor(x / 289.0) * 289.0; }
fn sn_mod289_3(x: vec3f) -> vec3f { return x - floor(x / 289.0) * 289.0; }
fn sn_mod289_4(x: vec4f) -> vec4f { return x - floor(x / 289.0) * 289.0; }
fn sn_permute3(x: vec3f) -> vec3f { return sn_mod289_3((x * 34.0 + 1.0) * x); }
fn sn_permute4(x: vec4f) -> vec4f { return sn_mod289_4((x * 34.0 + 1.0) * x); }
fn sn_taylorInvSqrt4(r: vec4f) -> vec4f { return 1.79284291400159 - 0.85373472095314 * r; }

/// 2-D simplex noise in [-1, 1] --- direct port of lygia snoise2.
fn snoise2(v: vec2f) -> f32 {
  let C = vec4f( 0.211324865405187,   // (3.0 - sqrt(3.0)) / 6.0
                 0.366025403784439,   // 0.5 * (sqrt(3.0) - 1.0)
                -0.577350269189626,   // -1.0 + 2.0 * C.x
                 0.024390243902439);  // 1.0 / 41.0

  var i  = floor(v + dot(v, C.yy));
  let x0 = v - i + dot(i, C.xx);

  let i1 = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), x0.x > x0.y);
  let x12 = x0.xyxy + C.xxzz - vec4f(i1, 1.0, 1.0);

  i = sn_mod289_2(i);
  let p = sn_permute3(
    sn_permute3(i.y + vec3f(0.0, i1.y, 1.0)) + i.x + vec3f(0.0, i1.x, 1.0));

  var m = max(0.5 - vec3f(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3f(0.0));
  m = m * m;
  m = m * m;

  let x = 2.0 * fract(p * C.www) - 1.0;
  let h = abs(x) - 0.5;
  let ox = floor(x + 0.5);
  let a0 = x - ox;

  m *= sn_taylorInvSqrt4(vec4f(dot(a0.xy, a0.xy) + dot(h.xy, h.xy),
                                dot(a0.z,  a0.z)  + dot(h.z,  h.z),
                                0.0, 0.0)).xyz;  // only first 3 used

  let g = vec3f(a0.x * x0.x  + h.x * x0.y,
                a0.y * x12.x + h.y * x12.y,
                a0.z * x12.z + h.z * x12.w);

  return 130.0 * dot(m, g);
}
`;

// ---------------------------------------------------------------------------
// WGSL: FBM over 2-D simplex (lygia fbm.wgsl, translated to WGSL proper)
// Supports 2---6 octaves via a runtime uniform rather than compile-time const.
// ---------------------------------------------------------------------------

const WGSL_FBM = /* wgsl */`
// ------ Lygia fbm --- runtime octave count ------------------------------------------------------------------------------------------------------------------------
fn fbm2(st_in: vec2f, octaves: i32) -> f32 {
  var value: f32    = 0.0;
  var amplitude: f32 = 0.5;
  var st = st_in;

  for (var i = 0; i < octaves; i++) {
    value     += amplitude * snoise2(st);
    st        *= 2.0;           // FBM_SCALE_SCALAR = 2
    amplitude *= 0.5;           // FBM_AMPLITUDE_SCALAR = 0.5
  }
  return value;
}
`;

// ---------------------------------------------------------------------------
// WGSL: 2-D curl noise using FBM potential (lygia curl.wgsl variant)
//
// Curl is computed from the numerical gradient of a scalar FBM potential -:
//   curl(-)(x,y) = (----/---y, -------/---x)
// This guarantees a divergence-free field, perfect for smoke / aurora / water.
// ---------------------------------------------------------------------------

const WGSL_CURL = /* wgsl */`
// ------ Lygia curl2 --- FBM-potential divergence-free field ---------------------------------------------------------------------
fn curlFBM(p: vec2f, octaves: i32) -> vec2f {
  let e = 0.1;
  let dx = vec2f(e, 0.0);
  let dy = vec2f(0.0, e);

  let p_x0 = fbm2(p - dx, octaves);
  let p_x1 = fbm2(p + dx, octaves);
  let p_y0 = fbm2(p - dy, octaves);
  let p_y1 = fbm2(p + dy, octaves);

  // central-difference gradient --- 90- rotation for curl
  let gx = (p_x1 - p_x0) / (2.0 * e);
  let gy = (p_y1 - p_y0) / (2.0 * e);

  // curl of scalar field: rotate gradient 90-
  return vec2f(gy, -gx);
}
`;

// ---------------------------------------------------------------------------
// WGSL: the force-overlay compute shader
// ---------------------------------------------------------------------------
// Uniform layout (NoiseUniforms, 32 bytes = 8 -- f32):
//   0  freq        --- noise space frequency
//   1  timeScale   --- time warp speed
//   2  time        --- current simulation time (animated)
//   3  strength    --- force magnitude
//   4  driftX      --- constant drift force X
//   5  driftY      --- constant drift force Y
//   6  count_f     --- particle count (as f32 for padding)
//   7  octaves_f   --- FBM octave count (as f32, cast to i32 in shader)

const NOISE_FORCE_SHADER = /* wgsl */`
${WGSL_SIMPLEX_BASIS}
${WGSL_FBM}
${WGSL_CURL}

struct NoiseUniforms {
  freq      : f32,
  timeScale : f32,
  time      : f32,
  strength  : f32,
  driftX    : f32,
  driftY    : f32,
  count_f   : f32,
  octaves_f : f32,
}

@group(0) @binding(0) var<uniform>            noiseParams : NoiseUniforms;
@group(1) @binding(0) var<storage, read>       posX        : array<f32>;
@group(1) @binding(1) var<storage, read>       posY        : array<f32>;
@group(1) @binding(2) var<storage, read_write> forceX      : array<f32>;
@group(1) @binding(3) var<storage, read_write> forceY      : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = u32(noiseParams.count_f);
  if (i >= n) { return; }

  let octaves = i32(noiseParams.octaves_f);
  let t       = noiseParams.time * noiseParams.timeScale;

  // Sample curl-noise in noise-space (scaled by freq, animated by t)
  let p = vec2f(posX[i], posY[i]) * noiseParams.freq + vec2f(t, t * 0.37);

  let curl   = curlFBM(p, octaves);         // divergence-free direction
  let fbmMag = abs(fbm2(p * 0.5, octaves)); // amplitude modulation

  // Compose final force: curl direction -- FBM magnitude -- strength + drift
  let fx = (curl.x * fbmMag + noiseParams.driftX) * noiseParams.strength;
  let fy = (curl.y * fbmMag + noiseParams.driftY) * noiseParams.strength;

  // Additive overlay onto existing SPH forces
  forceX[i] += fx;
  forceY[i] += fy;
}
`;

// ---------------------------------------------------------------------------
// NoiseFlowField --- TypeScript class
// ---------------------------------------------------------------------------

export class NoiseFlowField {
  private readonly device: any /*GPUDevice*/;
  private readonly bufs: GPUBufferSet;

  private pipeline!: any /*GPUComputePipeline*/;
  private noiseUniformBuf!: any /*GPUBuffer*/;
  private noiseUniformBGL!: GPUBindGroupLayout;
  private particleBGL!: GPUBindGroupLayout;
  private noiseUniformBG!: any /*GPUBindGroup*/;
  private particleBG!: any /*GPUBindGroup*/;

  /** Current active effect. */
  private _effect: NoiseEffect = "smoke";
  /** Accumulated simulation time (seconds). */
  private _time: number = 0;

  constructor(device: any /*GPUDevice*/, bufs: GPUBufferSet) {
    this.device = device;
    this.bufs   = bufs;
    this._init();
  }

  // -------------------------------------------------------------------------
  private _init(): void {
    const dev = this.device;

    // 32-byte noise uniform buffer
    this.noiseUniformBuf = dev.createBuffer({
      label: "noise-flow-uniform",
      size : 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // group 0 --- NoiseUniforms
    this.noiseUniformBGL = dev.createBindGroupLayout({
      label  : "noise-uniform-bgl",
      entries: [{
        binding   : 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer    : { type: "uniform" },
      }],
    });

    // group 1 --- posX(r), posY(r), forceX(rw), forceY(rw)
    this.particleBGL = dev.createBindGroupLayout({
      label  : "noise-particle-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    const layout = dev.createPipelineLayout({
      label           : "noise-flow-layout",
      bindGroupLayouts: [this.noiseUniformBGL, this.particleBGL],
    });

    const module = dev.createShaderModule({
      label: "noise-flow-shader",
      code : NOISE_FORCE_SHADER,
    });

    this.pipeline = dev.createComputePipeline({
      label  : "noise-flow-pipeline",
      layout,
      compute: { module, entryPoint: "main" },
    });

    // Bind groups (static --- buffers don't change)
    this.noiseUniformBG = dev.createBindGroup({
      label  : "noise-uniform-bg",
      layout : this.noiseUniformBGL,
      entries: [{ binding: 0, resource: { buffer: this.noiseUniformBuf } }],
    });

    this.particleBG = dev.createBindGroup({
      label  : "noise-particle-bg",
      layout : this.particleBGL,
      entries: [
        { binding: 0, resource: { buffer: this.bufs.posX   } },
        { binding: 1, resource: { buffer: this.bufs.posY   } },
        { binding: 2, resource: { buffer: this.bufs.forceX } },
        { binding: 3, resource: { buffer: this.bufs.forceY } },
      ],
    });
  }

  // -------------------------------------------------------------------------
  /** Upload NoiseUniforms to GPU from the current config + accumulated time. */
  private _uploadUniforms(cfg: Required<NoiseFlowFieldConfig>, count: number): void {
    const ep = EFFECT_PARAMS[cfg.effect];
    const data = new Float32Array(8);
    data[0] = ep.freq;
    data[1] = ep.timeScale;
    data[2] = cfg.time;
    data[3] = ep.baseStrength * cfg.strength;
    data[4] = ep.driftX;
    data[5] = ep.driftY;
    data[6] = count;
    data[7] = ep.octaves;
    this.device.queue.writeBuffer(this.noiseUniformBuf, 0, data);
  }

  // -------------------------------------------------------------------------
  /**
   * Encode the noise force overlay pass into an existing GPUCommandEncoder.
   *
   * Call this *after* SPHGPUOrchestrator.encodeForces() so the noise force
   * is added on top of the SPH pressure/viscosity forces.
   *
   * @param encoder - command encoder to append the pass into
   * @param n       - number of fluid particles (must match SPH sim)
   * @param cfg     - effect config; if omitted uses last call's settings
   */
  encodeForceOverlay(
    encoder: any /*GPUCommandEncoder*/,
    n      : number,
    cfg    : NoiseFlowFieldConfig = { effect: this._effect },
  ): void {
    const resolved: Required<NoiseFlowFieldConfig> = {
      effect  : cfg.effect   ?? this._effect,
      strength: cfg.strength ?? 1.0,
      time    : cfg.time     ?? this._time,
    };

    this._effect = resolved.effect;
    this._time   = resolved.time;

    this._uploadUniforms(resolved, n);

    const wg   = Math.ceil(n / WORKGROUP_SIZE);
    const pass = encoder.beginComputePass({ label: "noise-flow-pass" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.noiseUniformBG);
    pass.setBindGroup(1, this.particleBG);
    pass.dispatchWorkgroups(wg);
    pass.end();
  }

  // -------------------------------------------------------------------------
  /**
   * Convenience tick: creates its own encoder, submits it immediately.
   * Suitable for standalone use without DFSPH encode-style loop.
   *
   * @param n    - particle count
   * @param dt   - frame delta-time in seconds (advances internal time)
   * @param cfg  - optional effect config override
   */
  tick(n: number, dt: number, cfg?: NoiseFlowFieldConfig): void {
    this._time += dt;
    const effect = cfg?.effect ?? this._effect;
    const encoder = this.device.createCommandEncoder({ label: "noise-flow-tick" });
    this.encodeForceOverlay(encoder, n, { ...cfg, effect, time: this._time });
    this.device.queue.submit([encoder.finish()]);
  }

  // -------------------------------------------------------------------------
  /** Change effect preset without resetting accumulated time. */
  setEffect(effect: NoiseEffect): void {
    this._effect = effect;
  }

  /** Reset internal time (restarts noise animation). */
  resetTime(): void {
    this._time = 0;
  }

  /** Release GPU resources. */
  destroy(): void {
    this.noiseUniformBuf.destroy();
  }
}

// ---------------------------------------------------------------------------
// Helper: build a NoiseFlowField and wire it into an SPHGPUOrchestrator
// encode loop.  Import and call from your world-stepper or simulation setup.
// ---------------------------------------------------------------------------

/**
 * Returns a per-frame callback that encodes the noise force overlay.
 * Drop the returned function into your existing encode loop, e.g.:
 *
 * ```ts
 * const noiseOverlay = createNoiseOverlay(device, bufs, 'aurora');
 *
 * // Inside sim loop:
 * orchestrator.encodeForces(encoder, n);
 * noiseOverlay(encoder, n, dt);            // --- additive noise on top
 * orchestrator.encodeIntegrate(encoder, n, dt);
 * ```
 */
export function createNoiseOverlay(
  device: any /*GPUDevice*/,
  bufs  : GPUBufferSet,
  effect: NoiseEffect = "smoke",
): (encoder: any /*GPUCommandEncoder*/, n: number, dt: number) => void {
  const field = new NoiseFlowField(device, bufs);
  field.setEffect(effect);
  let time = 0;

  return (encoder: any /*GPUCommandEncoder*/, n: number, dt: number) => {
    time += dt;
    field.encodeForceOverlay(encoder, n, { effect, time });
  };
}
