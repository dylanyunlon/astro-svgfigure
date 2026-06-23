export { ATFlowerParticleRenderer, edgeRouteToFlowerSpline, createATFlowerForSPH, AT_FLOWER_DEFAULTS } from './at-flower-particle';
export type { FlowerEdgeSpline, FlowerPoint3, ATFlowerConfig } from './at-flower-particle';
export { SplineParticleLife, edgeRouteToSplineData, createSplineParticleLifeForSPH } from './spline-particle-life';
export type { EdgeSplineData, SplineParticle, SplineParticleLifeConfig, ParticlePhase, SplinePoint3 } from './spline-particle-life';
export { AtmospherePass, ATMOSPHERE_PRESETS } from './atmosphere';
export type { AtmosphereParams } from './atmosphere';
export { SpatialHashGrid } from './SpatialHashGrid';
export { NeighborListBuilder } from './NeighborListBuilder';
export { BoundaryModel } from './BoundaryModel';
export { SPHGPUOrchestrator } from './SPHGPUOrchestrator';
export { ParticleRenderer } from './ParticleRenderer';
export { SPHWorld } from './SPHWorld';
export type { EffectModule, EffectName } from './SPHWorld';
export { ATRenderPipeline } from './ATRenderPipeline';
export type { ATBackend, ATRenderPipelineOptions } from './ATRenderPipeline';
export { qosToSpatial, interpolateConfigs, QOS_PRESETS, qosSpatial } from './qosSpatial';
export { BoidsCompute, BOIDS_WORKGROUP, BOIDS_MAX } from './boids-compute';
export type { BoidsParams, BoidsSnapshot } from './boids-compute';
export * from './types';
export * as collision from './collision';


// ─── Auto-generated barrel exports (M730) ───
// at-bloom-postprocess
export { ATBloomParams, ATBloomPostProcess, createATBloomForSpecies, AT_BLOOM_WGSL } from './at-bloom-postprocess';
// at-navier-stokes
export { NS_GRID, NS_PRESSURE_ITERS_DEFAULT, NavierStokesParams, NavierStokesSplat, NavierStokesFluid, createNavierStokesFluid } from './at-navier-stokes';
// at-pbr-material
export { PBRParams, MatcapParams, DEFAULT_PBR_PARAMS, DEFAULT_MATCAP_PARAMS, packPBRUniforms, packMatcapUniforms, ATPBRMaterial, ATMatcapFresnel /* +1 more */ } from './at-pbr-material';
// at-render-pipeline
export { SceneMatrices, ATPassFlags, ATRenderPipelineConfig, ATRenderPipeline } from './at-render-pipeline';
// at-scene-compositor
export { CellBBox, SPHWorldView, ATSceneCompositorConfig, CompositorPassFlags, ATSceneCompositor } from './at-scene-compositor';
// at-shader-utils
export { WGSL_EASES, WGSL_RANGE, WGSL_BLEND_MODES, AT_SHADER_UTILS_WGSL, range, crange, rangeNorm, rangeMirror /* +3 more */ } from './at-shader-utils';
// at-spline-particle
export { SplinePoint3, EdgeSpline, SplineParticleInstance, ATSplineParticleConfig, SplineParticlePreset, ATSplineParticleLife, createATSplineParticleForSPH, canvasRouteToEdgeSpline /* +1 more */ } from './at-spline-particle';
// at-volumetric-light
export { ATVolumetricLightParams, ATVolumetricLight, createATVolumetricLightForSpecies, AT_VOLUMETRIC_LIGHT_WGSL } from './at-volumetric-light';
// at-water-surface
export { ATWaterSurfaceConfig, ATWaterSurface, createATWaterSurface } from './at-water-surface';
// cell-body-bridge
export { CellPhysicsConfig, speciesToIndex, cellsToBodies, initCellBodies, defaultFluidLayout } from './cell-body-bridge';
// cell-material-system
export { CellSpecies, MaterialPhysicsModulator, SpeciesMaterialDef, ATTENTION_MATERIAL, FFN_MATERIAL, LAYERNORM_MATERIAL, EMBEDDING_MATERIAL, SOFTMAX_MATERIAL /* +6 more */ } from './cell-material-system';
// cell-visual-identity
export { Morphology, ForceInput, ContactSummary, VisualProfile, CellVisualIdentity } from './cell-visual-identity';
// chromatic-adaptation
export { ChromaticMode, ParticlePhysics, ChromaticColor, resolveChromatic, CHROMATIC_STRIDE, batchResolveChromatic, chromaticToCss, chromaticToGlowCss /* +3 more */ } from './chromatic-adaptation';
// color-palette
export { RGB, RGBA, ThemePalette, QOS_THEME, ParticleColorInput, resolveParticleColor, rgbaToCss, rgbaToU8 /* +1 more */ } from './color-palette';
// collision-fx-system
export { CollisionFXConfig, CollisionFXSystem, estimateFlowerImpulse } from './collision-fx-system';
// collision-shockwave
export { ShockwaveConfig, ShockwaveRingGPU, CollisionShockwaveSystem, CollisionShockwavePipeline, estimateShockwaveImpulse } from './collision-shockwave';
// contact-sparks
export { Vec2, SparkConfig, ContactSparkSystem, estimateImpulse } from './contact-sparks';
// curl-aura
export { AURA_STRIDE, CurlAuraOptions, CurlAuraRenderer, CellAuraDescriptor, PackedAuraResult, packAuraData, packAuraDataSOA } from './curl-aura';
// curl-flow-field
export { CurlFlowFieldParams, ResolvedCurlParams, CurlSample, CurlFlowField, createCurlFlowField } from './curl-flow-field';
// debug-renderer
export { DebugRenderOptions, Vec2, AABB, ContactPoint, CollisionManifold, BVHNode, ParticleEmitter, DebugParticle /* +3 more */ } from './debug-renderer';
// dfsph-solver
export { Particle, createParticle, computeDFSPHFactor, pressureSolve, divergenceSolve, stepDFSPH } from './dfsph-solver';
// differential-growth
export { DifferentialGrowthConfig, DifferentialGrowth } from './differential-growth';
// emitter-strategy
export { EmitterConfig, EmissionPattern, ContinuousPattern, HighFreqStreamPattern, LowFreqPulsePattern, ConstantFieldPattern, BurstWavePattern, patternForProfile /* +7 more */ } from './emitter-strategy';
// environment-fx
export { EnvironmentFxConfig, EnvironmentFxParams, EnvironmentFx, EnvironmentFxPresets } from './environment-fx';
// flowmap-bridge
export { SPHParticleAOS, SPHParticleSOA, FlowmapBridgeOptions, rasterizeVelocityField, rasterizeVelocityFieldSOA, dissipateField, normalizeVelocityField, FlowmapBridge } from './flowmap-bridge';
// fluid-rigid-coupling
export { BoundaryVolumeTable, computeBoundaryVolumes, refreshBoundaryState, addBoundaryDensity, computeCouplingForces, stepFluidRigidCoupling, buildBoundaryNeighbors } from './fluid-rigid-coupling';
// fluid-surface-mesh
export { FluidSurfaceMeshConfig, FluidSurfaceMesh, Vertex2D, ContourSegment, SurfaceTriangleMesh, drawContourToCanvas, drawFieldHeatmap, selfTest as fluidSurfaceMeshSelfTest } from './fluid-surface-mesh';
// lattice-boltzmann-bg
export { LBM_GRID, SPH_PARTICLE_COUNT, LBMConfig, LBMBuffers, LatticeBoltzmannBackground, createDualLayerFluid } from './lattice-boltzmann-bg';
// lut-generator
export { LutStyleName, LutCube, LutPipelineState, LutGeneratorOptions, LutStateOverrides, classifyQoSZone, classifyQoSProfileName, LutGenerator /* +4 more */ } from './lut-generator';
// morphogenesis
export { LSystemPreset, LSystemDef, MorphogenesisConfig, Morphogenesis, fromPreset, PRESET_NAMES } from './morphogenesis';
// natural-patterns
export { NaturalPatternMode, SpeciesFbmParams, SpeciesParams, speciesParams, speciesPatternMode, NaturalPatternParams, NaturalPatternGenerator } from './natural-patterns';
// noise-flow-field
export { NoiseEffect, NoiseFlowFieldConfig, NoiseFlowField, createNoiseOverlay } from './noise-flow-field';
// ocean-background
export { OceanConfig, OceanUniforms, OceanBackground } from './ocean-background';
// ocean-bridge
export { SimMode, PresetIndex, OceanBridgeOptions, OceanFrameEvent, OceanBridge, createOceanBridge, isWebGPUSupported } from './ocean-bridge';
// ogl-flowmap-bridge
export { FlowParticle, SPHFlowmap, SPHFlowmapOptions, createSPHFlowmap } from './ogl-flowmap-bridge';
// organic-growth-animator (M773)
export { OrganicGrowthAnimator, GROWTH_PRESETS, getGlobalGrowthAnimator, setGlobalGrowthAnimator } from './organic-growth-animator';
export type { GrowthPhase, GrowthConfig, BatchGrowthOptions, VeinStrand, GrowthState } from './organic-growth-animator';
// organic-sdf
export { Vec2, flowerSDF, kochSDF, juliaSDF, OrganicSdfKind, SpeciesSdfParams, getSpeciesSdfParams, organicOutline /* +3 more */ } from './organic-sdf';
// particle-compositor
export { enum, LayerDescriptor, CompositorConfig, ParticleCompositor, createCompositorForATRenderers, COMPOSITOR_DEFAULTS } from './particle-compositor';
// performance-budget
export { Tier, TierConfig, TierChangeHandler, PerformanceBudget, getGlobalBudget, setGlobalBudget } from './performance-budget';
// phyllotaxis
export { GOLDEN_ANGLE_DEG, GOLDEN_ANGLE_RAD, PhyllotaxisPoint, PhyllotaxisOptions, generatePhyllotaxis, polarAt, cartesianAt, SvgCircleOptions /* +1 more */ } from './phyllotaxis';
// physarum-sim
export { PhysarumParams, PhysarumSimulation } from './physarum-sim';
// physarum-edge-bridge
export { PhysarumEdgeBridgeConfig, SpeedModulatorFn, DriftVector, PhysarumEdgeBridge, createPhysarumEdgeBridge, PHYSARUM_EDGE_BRIDGE_DEFAULTS } from './physarum-edge-bridge';
// edge-flow-renderer
export { FlowPoint, FlowEdge, FlowPhase, FlowParticle, OnArrivalFn, EdgeFlowRendererConfig, EdgeFlowRenderer, createEdgeFlowRenderer, createEdgeFlowForSPH, evalCatmullRom, splineTangent, EDGE_FLOW_DEFAULTS } from './edge-flow-renderer';
// physics-uniform-bridge
export { PhysicsUniforms, samplePhysicsForBody, sampleAllBodies } from './physics-uniform-bridge';
// post-process
export { PostProcessStyle, PostProcessParams, PostProcessPipeline, PostProcessPresets } from './post-process';
// qos-spatial-bridge
export { Reliability, Durability, QoSProfile, SpatialPhysics, qosToSpatial, APOLLO_PROFILES, PROFILE_DESCRIPTIONS } from './qos-spatial-bridge';
// reaction-diffusion
export { RD_DEFAULT_SIZE, RD_DEFAULT_SUBSTEPS, RD_MAX_SPECIES, GrayScottSpecies, GrayScottParams, RDSimConfig, SpeciesParamEntry, SpeciesRegion, parameterSpace, speciesGrayScottParams, buildDefaultSpeciesLUT, SPECIES_GRAYSCOTT_MAP, ReactionDiffusionSim } from './reaction-diffusion';
// rigid-body
export { RigidBody, RigidBodyOptions, createRigidBody, sampleBoundaryParticles, applyForce, integrateRigidBodies, resolveRigidRigidCollisions, resetForces } from './rigid-body';
// ripple-effect
export { RippleEffectConfig, RippleEffect } from './ripple-effect';
// spatial-hash
export { SpatialHashGrid, buildNeighborLists } from './spatial-hash';
// species-shader-registry
export { SdfShape, SdfShapeParams, MaterialType, MaterialParams, PatternShader, PhysicsBinding, PhysicsBindings, SpeciesShaderConfig /* +5 more */ } from './species-shader-registry';
// species-visual-dna
export { RuntimeBloom, RuntimeMaterial, RuntimePattern, RuntimeSdf, VisualDNA, initVisualDNA, getVisualDNA, getAllVisualDNA /* +1 more */ } from './species-visual-dna';
// sph-bridge
export { SPHFrameSnapshot, initSPHWorld, addFluid, addBody, stepSPH, setQoS, raycast, terminateSPHWorker } from './sph-bridge';
// sph-epoch-bridge
export { CellDescriptor, TopologyPayload, EpochPayload, CellParamsUpdatedPayload, SPHEpochBridgeOptions, SPHEpochBridge } from './sph-epoch-bridge';
// sph-kernels
export { SPHConfig, defaultConfig, cubicW, cubicGradW, spikyGradW, poly6W, viscLaplacianW, selfTest } from './sph-kernels';
// sph-worker
export { InitOptions, FluidParams, BodyParams, EmitterParams, WorkerSnapshot, RaycastHit, SPHWorkerAPI } from './sph-worker';
// turing-pattern
export { TuringPatternSpecies, speciesTuringMode, TuringPatternParams, TuringPatternGenerator } from './turing-pattern';
// uil-species-live
export { UniformValue, SpeciesUniformBag, PhysicsState, initSpeciesLive, getSpeciesUniforms, physicsUniformsToState, getLoadedSpecies } from './uil-species-live';
// density-field-texture (M763)
export { DensityFieldTextureConfig, DensityFieldTexture, ParticleSource, selfTest as densityFieldTextureSelfTest } from './density-field-texture';
// velocity-field-texture (M764)
export { VelocityFieldTextureConfig, VelocityFieldTexture, VelocityParticleSource, selfTest as velocityFieldTextureSelfTest } from './velocity-field-texture';
// water-caustics
export { WaterCausticsConfig, WaterCaustics } from './water-caustics';
// fluid-caustics (M781)
export { FluidCausticsConfig, CausticWaveOctave, FluidCaustics, FluidParticleSource, CellSurfaceTarget, selfTest as fluidCausticsSelfTest } from './fluid-caustics';
// world-boundary
export { RectBoundaryShape, CircleBoundaryShape, PolygonBoundaryShape, BoundaryShape, WorldConfig, BoundaryParticle, defaultWorldConfig, createWallParticles /* +8 more */ } from './world-boundary';
// world-renderer
export { SPECIES_COLORS, CELL_KIND_COLORS, RenderOptions, DEFAULT_OPTIONS, BVHNodeFlat, ContactPoint, WorldRenderExtras, renderWorld } from './world-renderer';
// world-serializer
export { serializeWorld, deserializeWorld } from './world-serializer';
// world-stepper
export { Particle, Emitter, WorldConfig, World, createWorld, addFluidBlock, addRigidBody, addEmitter /* +7 more */ } from './world-stepper';
// epoch-physics-recorder (M768)
export { EpochPhysicsRecorder, Recording } from './epoch-physics-recorder';
export type { EpochSnapshot } from './epoch-physics-recorder';
// transition-system (M748)
export { TransitionSystem, TRANSITION_PRESETS, getGlobalTransitionSystem, setGlobalTransitionSystem } from './transition-system';
export type { TransitionDirection, TransitionMode, TransitionPhase, DissolveParticle, CellShapeSnapshot, TransitionConfig, TransitionState } from './transition-system';
// vfx-timeline (M768)
export { VFXTimelinePlayer, VFXTimelineBuilder, VFX_PRESETS, wireCollisionVFX, CanvasScreenFlash, getGlobalVFXPlayer, setGlobalVFXPlayer } from './vfx-timeline';
export type { VFXEventKind, ShockwaveParams, BloomSpikeParams, ParticleBurstParams, ScreenFlashParams, CameraShakeParams, CustomVFXParams, VFXEventParams, VFXKeyframe, VFXTimeline, VFXHandler, CollisionVFXWiringConfig } from './vfx-timeline';
// dynamic-environment (M765)
export { DynamicEnvironment, DYNAMIC_ENV_PRESETS, createDynamicEnvironment } from './dynamic-environment';
export type { SkyPhase, Color3 as DynEnvColor3, EnvironmentSnapshot, DynamicEnvironmentConfig } from './dynamic-environment';
// physics-render-bridge (M781)
export { PhysicsRenderBridge, snapshotToWorldView, createPhysicsRenderBridge } from './physics-render-bridge';
export type { PhysicsWorldView, RigidBodyRenderState, ContactEvent, PhysicsFrameStats, PhysicsRenderSnapshot, PhysicsRenderConsumer, PhysicsRenderBridgeOptions } from './physics-render-bridge';
// magnetic-field-lines (M782)
export { MagneticFieldLines, evaluateField, fieldMagnitude, traceStreamline, generateSeeds, subdivideStreamline, computeArcLengths, catmullRom, SPECIES_CHARGE_SIGN, QOS_FIELD_STYLES } from './magnetic-field-lines';
export type { FieldCharge, FieldCellInput, MagneticFieldLinesConfig, FieldLineStyle, StreamPoint, Streamline, FieldVec2 } from './magnetic-field-lines';
// render-loop (M782)
export { RenderLoop, UPDATE_PRIORITY, FPSMonitor, CellPubSubInbox, RENDER_LOOP_DEFAULTS } from './render-loop';
export type { RenderLoopConfig, TickerState, TickerCallback, ListenerHandle, CellEventKind, CellEvent, CellEventHandler, UpdatePriority } from './render-loop';
// particle-effect-system (M783)
export { ParticleEffectSystem, EFFECT_DEFAULTS, EFFECT_TYPE_INDEX, GPU_STRIDE, PARTICLE_FX_WGSL } from './particle-effect-system';
export type { ParticleEffectType, Vec2 as ParticleFxVec2, CollisionSparkParams, FlowTrailParams, AmbientDustParams, QosTransitionParams, CellBirthParams, CellDeathParams, EmitParams, ParticleEffectSystemConfig } from './particle-effect-system';
// pic-flip-solver (M780)
export { createMACGrid, stepPICFLIP, createDamBreak } from './pic-flip-solver';
export type { PICFLIPConfig, FLIPParticle, MACGrid } from './pic-flip-solver';
// wireframe-overlay (M788)
export { WireframeOverlay, WIREFRAME_DEFAULTS, createDebugWireframe, createAestheticWireframe, sampleForceGrid, selfTest as wireframeOverlaySelfTest, WGSL_BARY_VERTEX, WGSL_BARY_WIREFRAME_FRAG, WGSL_SDF_ISOCONTOUR } from './wireframe-overlay';
export type { Vec2 as WireframeVec2, WireframeCellEntry, ForceFieldSample as WireframeForceFieldSample, WireframeFrameData, WireframeOverlayOptions } from './wireframe-overlay';
// heat-transfer (M783)
export { HeatTransferSolver, defaultHeatConfig, sampleThermalRamp, rgbaToCss as heatRgbaToCss, rgbaToU8 as heatRgbaToU8, selfTest as heatTransferSelfTest } from './heat-transfer';
export type { HeatTransferConfig, HeatParticle, HeatParticleSOA, RGBA as HeatRGBA } from './heat-transfer';
// destruction-system (M789)
export { DestructionSystem, estimateDestructionImpulse } from './destruction-system';
export type { DestructionConfig, DebrisShard } from './destruction-system';
// decal-projector (M792)
export { DecalProjector, DecalGPUPipeline, estimateDecalImpulse } from './decal-projector';
export type { DecalType, DecalProjectorConfig, DecalGPU, CellTransform } from './decal-projector';
