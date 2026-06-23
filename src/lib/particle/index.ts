/**
 * src/lib/particle/index.ts
 *
 * Particle system module — GPGPU emitter for Active Theory.
 *
 * AT参数来源: channels/physics/at_uil_params.json
 *
 * Quick start:
 *
 *   import { ParticleSystem, SplineEmitter } from '$lib/particle';
 *   import curlFrag from '$lib/particle/CurlNoise.frag?raw';
 *
 *   const ps = new ParticleSystem({
 *     gl,
 *     particleCount: 16384,
 *     // AT: uCurlNoiseSpeed=5, uSCurlNoiseSpeed=5 (WorkDetailParticles)
 *     curlNoiseSpeed:  5,
 *     sCurlNoiseSpeed: 5,
 *     // AT: uSCurlNoiseScale=2, uThicknessSpeed=1
 *     curlNoiseScale:  2,
 *     thicknessSpeed:  1,
 *     // AT: uSplineSpeed=[0.82,1.21]
 *     splineSpeed: [0.82, 1.21],
 *     updateShaderSrc: curlFrag,
 *   });
 *
 *   const emitter = new SplineEmitter(16384, {
 *     splineSpeed:    [0.82, 1.21],
 *     timeMultiplier: 0.17,
 *     infinite:       true,
 *   });
 *
 *   emitter.loadFromJSON(splineJSON);
 *
 *   // Per frame:
 *   emitter.update(delta);
 *   emitter.uploadToGPU(gl, ps);
 *   ps.update(delta);
 *   ps.render(projection, modelView);
 */

export { ParticleSystem }    from './ParticleSystem';
export type { ParticleSystemConfig, ParticleSystemUniforms, GPGPURenderTexture } from './ParticleSystem';

export { SplineEmitter }     from './SplineEmitter';
export type { SplineEmitterConfig, SplinePoint, SplineJSON, SplineParticleState } from './SplineEmitter';

export { EdgeParticleSystem, createEdgeParticleSystem } from './EdgeParticleSystem';
export type { EdgeParticleSystemConfig, EdgeRoute as EdgeParticleRoute } from './EdgeParticleSystem';
