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
// at-texture-loader
export { ATTextureLoader, AT_MATERIAL_NAMES, estimateGPUBytes, estimateMaterialSetGPUBytes, formatBytes } from './at-texture-loader';
export type { ATTexture, ATMaterialSet, ATMaterialName } from './at-texture-loader';
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
// emissive-glow (M791)
export { EmissiveGlowSystem, createEmissiveGlowSystem, getGlobalEmissiveGlow, setGlobalEmissiveGlow, EMISSIVE_STRIDE, EMISSIVE_GLOW_DEFAULTS, EMISSIVE_BUFFER_WGSL } from './emissive-glow';
export type { GlowMode, EmissiveProfile, EmissiveGlowConfig, CellBloomFeedback } from './emissive-glow';
// rain-snow-system (M796)
export { RainSnowSystem, WEATHER_PRESETS, WEATHER_WORKGROUP, WEATHER_MAX, WIND_FIELD_SIZE, PARTICLE_STRIDE, RAIN, SNOW, createWeatherSystem } from './rain-snow-system';
export type { WeatherKind, WeatherMode, RainSnowConfig, WeatherSnapshot } from './rain-snow-system';
// holographic-display (M798)
export { HolographicDisplay, HOLO_PRESETS, HOLOGRAPHIC_DEFAULTS, HOLOGRAPHIC_WGSL, createHolographicDisplay, createDebugHolographic, createSubtleHolographic, selfTest as holographicDisplaySelfTest } from './holographic-display';
export type { HoloCellEntry, HoloFrameData, HoloPresetName, RGBA as HoloRGBA, HoloPalette, HolographicDisplayOptions } from './holographic-display';

// ─── M813: barrel exports 100% ──────────────────────────────────────────────

// adaptive-lod
export { AdaptiveLOD, getGlobalLOD, setGlobalLOD } from './adaptive-lod';
export type { CellRenderHint, CellLODAssignment, LODCamera, LODCellInput, AdaptiveLODConfig, AdaptiveLODSnapshot } from './adaptive-lod';

// ambient-occlusion
export { DEFAULT_SSAO_PARAMS, DEFAULT_BLUR_PARAMS, DEFAULT_COMPOSITE_PARAMS, generateSSAOKernel, generateNoiseTexture, packSSAOUniforms, packBlurUniforms, packCompositeUniforms, SSAOPass, SSAO_WGSL, _selfTest as ssaoSelfTest } from './ambient-occlusion';
export type { SSAOParams, BlurParams, CompositeParams, SSAOConfig } from './ambient-occlusion';

// at-geometry-loader
export { ATGeometryLoader } from './at-geometry-loader';
export type { ATGeometry, ATGeometryName, ATGeometryLoaderOptions } from './at-geometry-loader';

// at-jellyfish-cell
export { ATJellyfishCell, createATJellyfishCell } from './at-jellyfish-cell';
export type { JellyfishVariantConfig, JellyfishInstance } from './at-jellyfish-cell';

// at-lighting-import
export { ATLightingImport, createDirectionalLight, createPointLight, createConeLight, createAreaLight } from './at-lighting-import';
export type { ATUniformInfo, ATLight, ATShadowConfig as ATLightingShadowConfig, ATPBRTextures, ATLightingConfig } from './at-lighting-import';

// at-postprocess-import
export { AT_FXAA_VERT, AT_FXAA_FRAG, AT_LENS_PREFILTER_FRAG, AT_LENS_DOWN_FRAG, AT_LENS_UP_FRAG, AT_COMPOSITE_FRAG, AT_FULLSCREEN_VERT, AT_LIGHT_VOLUME_VERT, AT_LIGHT_VOLUME_FRAG, ATPostProcessPipeline } from './at-postprocess-import';
export type { ATPostProcessParams } from './at-postprocess-import';

// at-shader-loader
export { ATShaderLoader } from './at-shader-loader';

// at-shadow-import
export { ATShadowSystem, computeCascadeSplits, mat4LookAt, mat4Ortho, mat4Multiply, mat4Identity, DEPTH_VERTEX_SHADER, DEPTH_FRAGMENT_STANDARD, DEPTH_FRAGMENT_VSM, VSM_BLUR_FRAGMENT, FULLSCREEN_QUAD_VERTEX, POISSON_DISK_25, POISSON_DISK_16, MAX_CASCADES, CASCADE_SPLIT_LAMBDA, DEFAULT_RESOLUTION } from './at-shadow-import';
export type { ShadowMode, ATShadowConfig as ATShadowImportConfig, DepthShaderSource, ShadowMapResource } from './at-shadow-import';

// audio-physics-bridge
export { AudioPhysicsBridge } from './audio-physics-bridge';

// audio-reactive-visuals
export { AudioReactiveVisuals, applyBreathToInstanceBuffer, applyBreathWaveToInstanceBuffer, updateAudioReactiveFrame } from './audio-reactive-visuals';
export type { AudioReactiveSnapshot, AudioReactiveConfig } from './audio-reactive-visuals';

// cell-aura
export { AURA_FLOATS_PER_CELL, CellAuraSystem, CellAuraPass, sdfShapeToKind } from './cell-aura';
export type { CellAuraState, CellAuraInput, PackedAuraBuffer, CellAuraPassOptions } from './cell-aura';

// cinematic-camera
export { CinematicCamera, pathPoint, CINEMATIC_PRESETS } from './cinematic-camera';
export type { CameraMode, ShakeProfile, PathLoopMode, FollowConfig, OrbitConfig, DollyZoomConfig, ShakeEvent, PathPoint, AutoFrameConfig, CameraSnapshot, CameraTransition } from './cinematic-camera';

// dof-bokeh
export { DOFBokehPipeline, DOFBokehPresets, createDOFBokehForScene, DOF_COC_WGSL, DOF_HEX_BLUR_A_WGSL, DOF_HEX_BLUR_B_WGSL, DOF_COMPOSITE_WGSL, DOF_UNIFORMS_WGSL } from './dof-bokeh';
export type { DOFBokehParams } from './dof-bokeh';

// domain
export { clampToDomain } from './domain';

// edge-data-flow-viz
export { EdgeDataFlowViz, createEdgeDataFlowViz, EDGE_DATA_FLOW_DEFAULTS } from './edge-data-flow-viz';
export type { PulseStyle, VizPoint, VizEdge, EdgeDataFlowVizConfig } from './edge-data-flow-viz';

// edge-energy-flow
export { EdgeEnergyFlow, createEdgeEnergyFlowWithBus, EDGE_ENERGY_FLOW_DEFAULTS, FLUID_PROFILES } from './edge-energy-flow';
export type { FluidProfile, EnergyPoint, EnergyEdge, TrafficMetric, EdgeEnergyFlowConfig } from './edge-energy-flow';

// environment-fog
export { EnvironmentFog, createEnvironmentFogForSpecies, EnvironmentFogPresets, ENVIRONMENT_FOG_WGSL } from './environment-fog';
export type { FogMode, EnvironmentFogParams } from './environment-fog';

// epoch-visual-sync
export { EpochVisualSync, diffEpochSnapshots, qualityToRenderConfig } from './epoch-visual-sync';
export type { VisualBbox, CellVisualSnapshot, EpochSnapshot as VisualEpochSnapshot, QualityState, BboxMorphTarget, TimelineState, EpochCompletedPayload, CellParamsPayload, RollbackPayload, CellApiDescriptor, OnSpeciesTransition, OnBboxMorph, OnQualityChange, OnTimelineProgress, OnRollback, OnEpochSnapshot, OnCellEnter, OnCellExit, EpochVisualSyncOptions, EpochVisualDiff, RenderQualityConfig } from './epoch-visual-sync';

// god-rays
export { GodRaysCompute, createGodRaysForSpecies, GOD_RAYS_WGSL, GOD_RAYS_MAX_LIGHTS } from './god-rays';
export type { GodRayLight, GodRaysParams } from './god-rays';

// gpu-culling
export { extractFrustumPlanes, buildViewProjFromCamera, cellBBoxToAABB, GPUCullingPipeline, prepareCellCullDispatch } from './gpu-culling';
export type { FrustumPlane, CullAABB, GPUCullingConfig } from './gpu-culling';

// gpu-particle-sort
export { GPUParticleRadixSort, RADIX_SORT_CONSTANTS } from './gpu-particle-sort';
export type { SortKeyMode, RadixSortConfig, RadixSortMetrics } from './gpu-particle-sort';

// heat-distortion
export { HeatDistortionSystem, HeatDistortionPipeline, evaluateEnergyField, nearestEmitter } from './heat-distortion';
export type { HeatEmitter, HeatEmitterGPU, HeatDistortionConfig } from './heat-distortion';

// instanced-cell-renderer
export { FLOATS_PER_CELL, CELL_INSTANCED_VERT, CELL_INSTANCED_FRAG, hexToLinearRGBA, InstancedCellRenderer, createInstancedCellRenderer, visualProfilesToDescriptors, computeRDGridAssignment } from './instanced-cell-renderer';
export type { CellBBox as InstancedCellBBox, CellInstanceDescriptor } from './instanced-cell-renderer';

// integration-test — no public exports (side-effect test runner)

// interactive-fluid
export { InteractiveFluid, createInteractiveFluid } from './interactive-fluid';
export type { FluidRenderCallback, InteractiveFluidOptions } from './interactive-fluid';

// lens-flare
export { LensFlareCompute, createLensFlareForSpecies, LENS_FLARE_WGSL, MAX_FLARE_LIGHTS, MAX_GHOST_LAYERS, LENS_FLARE_DEFAULTS } from './lens-flare';
export type { FlareLightSource, GhostLayerConfig, LensFlareParams } from './lens-flare';

// minimap-renderer
export { defaultMinimapConfig, renderMinimap } from './minimap-renderer';
export type { MinimapConfig } from './minimap-renderer';

// nature-texture-manager
export { NatureTextureManager } from './nature-texture-manager';
export type { NatureTextureKind, NatureTextureInstance, NatureTextureConfigMap, PhysarumCreateConfig, NatureTextureInstanceMap, GeneratorEntry } from './nature-texture-manager';

// neural-pathway-renderer
export { NeuralPathwayRenderer, createNeuralPathwayRenderer, createNeuralPathwayForSPH, NEURAL_PATHWAY_DEFAULTS } from './neural-pathway-renderer';
export type { NeuralStyle, NeuralPoint, NeuralEdge, OnVesicleArrivalFn, OnPulseArrivalFn, NeuralPathwayConfig } from './neural-pathway-renderer';

// particle-instancing
export { INSTANCE_STRIDE, ParticleInstancer, packParticleData, packParticleDataSOA, ortho } from './particle-instancing';
export type { ParticleInstancerOptions, PackedParticleResult } from './particle-instancing';

// particle-life-color
export { getLifecyclePhase, SPECIES_BASE_COLOR, resolveLifecycleColor, lifecycleColorToCss, lifecycleColorToU8, speciesIndexToId, batchResolveLifecycleColors, batchResolveLifecycleColorsIndexed, generateLifecycleRamp, generateLifecycleLUT } from './particle-life-color';
export type { LifecyclePhase, SpeciesBaseColor, LifecycleColorInput, LifecycleColor } from './particle-life-color';

// portal-effect
export { PortalEffectSystem, createPortalEffectRenderer, shouldRenderPortal, PORTAL_DEFAULTS, QOS_PORTAL_STYLE } from './portal-effect';
export type { PortalStyle, PortalPoint, PortalFlowPoint, PortalEdge, PortalEffectConfig } from './portal-effect';

// procedural-texture-atlas
export { speciesTextureKind, buildAtlasConfig, speciesAtlasUV, ATLAS_UV_WGSL, ProceduralTextureAtlas, PROCEDURAL_ATLAS_WGSL } from './procedural-texture-atlas';
export type { ProceduralTextureKind, TileParams, AtlasConfig } from './procedural-texture-atlas';

// reaction-diffusion-surface
export { CELL_RD_SIZE, CELL_RD_DEFAULT_SUBSTEPS, CELL_RD_MAX_CELLS, CellRDSurface, CellRDSurfaceManager, registerCellSurfaces } from './reaction-diffusion-surface';
export type { CellRDSurfaceConfig, CellRDManagerConfig, CellRDSnapshot } from './reaction-diffusion-surface';

// render-compositor
export { RenderCompositor, createRenderCompositor } from './render-compositor';
export type { CellBBox as CompositorCellBBox, SPHWorldView as CompositorSPHWorldView, SceneMatrices as CompositorSceneMatrices, RenderPassFlags, RenderCompositorConfig, FrameTimings } from './render-compositor';

// screen-space-reflection
export { DEFAULT_SSR_REFLECTION_PARAMS, SPECIES_REFLECTION_PROFILES, SSRReflectionPass, modulateSSRReflectionFromPhysics, selectSSRTier } from './screen-space-reflection';
export type { SSRReflectionParams, SpeciesReflectionProfile } from './screen-space-reflection';

// screen-space-reflections
export { DEFAULT_SSR_PARAMS, SPECIES_SSR_PROFILES, ScreenSpaceReflections, modulateSSRFromPhysics } from './screen-space-reflections';
export type { SSRParams, SpeciesSSRProfile } from './screen-space-reflections';

// shader-tuning-presets
export { SPECIES_TO_CIL, SHADER_PRESETS, getShaderPreset, getCilSpecies, SPECIES_PIPELINE_ORDER, lerpPreset } from './shader-tuning-presets';
export type { ShaderPreset, TransformerSpecies } from './shader-tuning-presets';

// shadow-map
export { ShadowMap } from './shadow-map';
export type { CellCaster, ShadowMapConfig } from './shadow-map';

// shadow-system
export { ShadowSystem } from './shadow-system';
export type { ShadowConfig } from './shadow-system';

// subsurface-scattering
export { SSS_PROFILE_REGISTRY, DEFAULT_SSS_PARAMS, SPECIES_LUT_ORDER, buildSpeciesLUTData, buildGaussianKernel, SSSPass, modulateSSSByPhysics, cpuSubsurfaceScatter, getSSSProfile, getTranslucentSpecies, speciesNeedsSSS } from './subsurface-scattering';
export type { SSSProfile, SSSParams, ModulatedSSS } from './subsurface-scattering';

// temporal-reprojection
export { TemporalReprojection, applyJitterToProjection, TAAPresets } from './temporal-reprojection';
export type { TAAParams } from './temporal-reprojection';

// test-scenes
export { createTestWorld, TEST_SCENES, runAllTestScenes } from './test-scenes';
export type { TestWorld, TestScene } from './test-scenes';

// tone-mapping
export { ACESInputMat, ACESOutputMat, acesFilm, acesNarkowicz, acesNarkowiczColor, expose } from './tone-mapping';
export type { Color3 as ToneMappingColor3, Mat3 } from './tone-mapping';

// topology-physics-sync
export { TopologyPhysicsSync, buildEdgeEmitters } from './topology-physics-sync';
export type { TopoNode, TopoEdge, TopologyPayload as TopoPhysicsPayload, EdgeRoute, TopologyPhysicsSyncOptions, PhysicsWorldDelegate, EdgeFlowDelegate, OnSyncCallback, SyncStats } from './topology-physics-sync';

// topology-transition-fx
export { TopologyTransitionFX, createTopologyTransitionFX, wireTopologyTransitionFX } from './topology-transition-fx';
export type { SplinePoint as TopoSplinePoint, EdgeCreatedOptions, EdgeRemovedOptions, CellCreatedOptions, CellRemovedOptions, TopologyTransitionFXConfig, TopologyFXEvent, OnTopologyFXCallback } from './topology-transition-fx';

// trails
export { updateTrails } from './trails';

// webgpu-sph-compute
export { WebGPUSPHCompute } from './webgpu-sph-compute';

// world-orchestrator
export { WorldOrchestrator } from './world-orchestrator';
export type { OrchestratorStats, WorldOrchestratorConfig } from './world-orchestrator';

// world-preset-scenes
export { PRESET_SCENES, getPresetNames, setupPreset } from './world-preset-scenes';
export type { PresetScene } from './world-preset-scenes';
