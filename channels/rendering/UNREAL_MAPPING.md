# UNREAL_MAPPING.md — UE Deferred Pipeline → channels/rendering/ Mapping

**Task M872** · Branch `cell-pubsub-loop`

Cross-reference between Unreal Engine source files (UE4 `upstream/unreal-renderer/`
and UE5 `upstream/unreal-renderer-ue5/`) and the Python port living under
`channels/rendering/`. Implementation status is assessed by inspecting file line
counts, function counts, stub coverage, and docstring annotations.

---

## Legend

| Status | Meaning |
|--------|---------|
| `done` | Core classes and functions fully ported; no major stubs; tested in loop |
| `partial` | Key structs/functions exist but some sub-systems stubbed or missing |
| `todo` | Module shell or minimal stubs only; no real deferred-pipeline logic |

---

## Mapping Table

### 1. Scene Infrastructure

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `DeferredShadingRenderer.cpp/.h` (UE4) · `Renderer-Private/DeferredShadingRenderer.cpp/.h` (UE5) | `shadow/shadow.py` (AstroCellDeferredShadingRenderer port) | **partial** |
| `RendererScene.cpp` · `SceneCore.cpp/.h` · `ScenePrivate.h` | `scene/scene_port.py` | **partial** |
| `SceneRendering.cpp/.h` · `SceneRenderBuilder.cpp/.h` (UE5) | `passes/passes_port.py` | **partial** |
| `SceneVisibility.cpp/.h` · `SceneOcclusion.cpp/.h` | `visibility/visibility_extra.py` | **partial** |
| `PrimitiveSceneInfo.cpp` · `SceneCore.h` | `registry/registry_core.py` | **partial** |
| `SceneCaptureRendering.cpp` · `SceneViewState.cpp/.h` (UE5) | `scene/scene_port.py` (AstroCellCaptureProcessor) | **partial** |

### 2. GPU Scene & Primitives

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `GPUScene.cpp/.h` (UE4 + UE5) | `lighting/lighting.py` (AstroCellGPUScene + AstroCellPrimitiveCollector) | **done** |
| `MeshDrawCommands.cpp/.h` · `MeshPassProcessor.cpp` | `drawcall/draw_call_batcher.py` + `drawcall/drawcall_extra.py` | **partial** |
| `DynamicPrimitiveDrawing.h/.inl` · `DynamicShadowMapChannelBindingHelper.cpp` | `drawcall/drawcall_extra.py` (PSO table, pipeline states) | **partial** |
| `InstanceCulling/InstanceCullingManager.cpp/.h` (UE5) | `nanite/nanite_port.py` (frustum-cull path) | **partial** |
| `SceneCulling/SceneCulling.cpp/.h` (UE5) | `visibility/visibility_extra.py` | **partial** |

### 3. Nanite Virtualised Geometry

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `Nanite/Nanite.cpp/.h` · `NaniteCullRaster.cpp/.h` (UE5) | `nanite/nanite_port.py` (perform_nanite_visibility) | **partial** |
| `Nanite/NaniteShading.cpp/.h` | `nanite/shading.py` | **partial** |
| `Nanite/NaniteComposition.cpp/.h` | `nanite/composition.py` + `compositor/compositor_core.py` | **done** |
| `Nanite/NaniteDrawList.cpp/.h` | `nanite/draw_list.py` | **done** |
| `Nanite/NaniteVisibility.cpp/.h` | `nanite/visibility.py` | **partial** |
| `Nanite/NaniteFeedback.cpp/.h` · `Nanite/NaniteShared.cpp/.h` | `streaming/streaming_core.py` (AstroCellFeedbackManager) | **partial** |
| `Nanite/NaniteStreamOut.cpp/.h` | `misc/misc_extra.py` (AstroCellStreamExporter + AstroCellStreamExportQueue) | **done** |

### 4. Base Pass & GBuffer

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `BasePassRendering.cpp/.h/.inl` (UE4 + UE5) | `passes/passes_port.py` (AstroCellBasePass) | **partial** |
| `DepthRendering.cpp/.h` · `CustomDepthRendering.cpp` | `passes/passes_port.py` (depth pass stubs) | **partial** |
| `VelocityRendering.cpp/.h` | `motionblur/motionblur_core.py` (velocity buffer sub-system) | **partial** |
| `AnisotropyRendering.cpp/.h` (UE5) | `shading/shading_core.py` (anisotropy flags) | **todo** |
| `DecalRenderingShared.cpp/.h` · `CompositionLighting/PostProcessDeferredDecals.cpp/.h` | `misc/misc_extra.py` (AstroCellDecalManager stubs) | **todo** |

### 5. Lighting

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `LightRendering.cpp/.h` · `LightSceneInfo.cpp/.h` | `lighting/lighting.py` + `lighting/lighting_port.py` | **done** |
| `LightGridInjection.cpp` · `TiledDeferredLightRendering.cpp` | `lighting/lighting.py` (AstroCellLightGrid) | **partial** |
| `LightMapRendering.cpp/.h` · `LightMapHelpers.cpp` · `IndirectLightingCache.cpp` | `lighting/lighting_port.py` | **partial** |
| `LightShaftRendering.cpp` · `LightFunctionRendering.cpp` | `effects/effects_port.py` | **todo** |
| `LightPropagationVolume.cpp/.h` (UE4) | *(not ported — UE4-only LPV)* | **todo** |
| `MegaLights/MegaLights.cpp/.h` (UE5) | `lighting/lighting_port.py` (stubs only) | **todo** |
| `StochasticLighting/StochasticLighting.cpp/.h` (UE5) | `lighting/lighting_port.py` (stubs only) | **todo** |

### 6. Shadows

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `ShadowSetup.cpp` · `ShadowRendering.cpp/.h` · `ShadowDepthRendering.cpp` | `shadow/shadow.py` | **done** |
| `CapsuleShadowRendering.cpp/.h` | `shadow/shadow_port.py` | **partial** |
| `DistanceFieldShadowing.cpp` · `DistanceFieldObjectCulling.cpp` | `distancefield/distancefield_port.py` | **partial** |
| `VirtualShadowMaps/VirtualShadowMapArray.cpp/.h` (UE5) | `shadow/shadow_port.py` (VSM stubs) | **todo** |
| `Shadows/ShadowScene.cpp/.h` · `ShadowSceneRenderer.cpp/.h` (UE5) | `shadow/shadow_port.py` | **partial** |
| `ShadowSetupMobile.cpp` | `shadow/shadow_port.py` (mobile path stub) | **todo** |

### 7. Distance Fields & AO

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `DistanceFieldAmbientOcclusion.cpp/.h` · `GlobalDistanceField.cpp/.h` | `distancefield/distancefield_port.py` | **partial** |
| `DistanceFieldObjectManagement.cpp` · `DistanceFieldObjectCulling.cpp` | `distancefield/distancefield_port.py` (object buffers) | **partial** |
| `DistanceFieldScreenGridLighting.cpp` · `DistanceFieldLightingPost.cpp/.h` | `distancefield/distancefield_port.py` | **partial** |
| `HeightfieldLighting.cpp/.h` · `SurfelTree.cpp` | `distancefield/distancefield_port.py` (heightfield stubs) | **todo** |
| `CompositionLighting/PostProcessAmbientOcclusion.cpp/.h` | `distancefield/distancefield_port.py` (SSAO path) | **partial** |

### 8. Reflections

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `ReflectionEnvironment.cpp/.h` · `ReflectionEnvironmentCapture.cpp/.h` | `reflection/reflection_port.py` | **partial** |
| `ReflectionEnvironmentDiffuseIrradiance.cpp` | `reflection/reflection_port.py` (irradiance stubs) | **todo** |
| `PlanarReflectionRendering.cpp/.h` | `reflection/reflection_port.py` (planar stubs) | **todo** |
| `PostProcess/ScreenSpaceReflections.cpp/.h` (UE4) · `ScreenSpaceRayTracing.cpp/.h` (UE5) | `reflection/reflection_port.py` (SSR path) | **partial** |
| `ReflectionEnvironmentRealTimeCapture.cpp` (UE5) | `styleprobe/style_probe_impl.py` (real-time capture port) | **partial** |

### 9. Lumen Global Illumination (UE5)

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `Lumen/Lumen.cpp/.h` · `LumenDiffuseIndirect.cpp` | `lumen/lumen.py` (AstroCellGlobalIllumination) | **partial** |
| `Lumen/LumenScene.cpp` · `LumenSceneData.h` · `LumenSceneRendering.cpp` | `lumen/lumen.py` (scene card system) | **partial** |
| `Lumen/LumenScreenProbeGather.cpp/.h` · `LumenRadianceCache.cpp/.h` | `lumen/lumen.py` + `lumen/lumen_port.py` | **partial** |
| `Lumen/LumenReflections.cpp/.h` · `LumenReflectionTracing.cpp` | `lumen/lumen_port.py` | **partial** |
| `Lumen/LumenSceneDirectLighting.cpp` · `LumenSceneLighting.cpp/.h` | `lumen/lumen_port.py` | **partial** |
| `Lumen/LumenSurfaceCache.cpp` · `LumenMeshCards.cpp/.h` | `lumen/lumen.py` (probe/card stubs) | **todo** |
| `Lumen/LumenRadiosity.cpp/.h` | `lumen/lumen_port.py` (radiosity stubs) | **todo** |
| `Lumen/LumenTranslucencyVolumeLighting.cpp/.h` | `translucency/translucency_core.py` (LTV stubs) | **todo** |

### 10. Post-Processing

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `PostProcess/PostProcessing.cpp/.h` · `PostProcess/RenderingCompositionGraph.cpp/.h` (UE4) | `postprocess/postprocess_port.py` (AstroCellDenoiserManager) | **partial** |
| `PostProcess/PostProcessTemporalAA.cpp/.h` (UE4) · `PostProcess/TemporalAA.cpp/.h` (UE5) | `temporal_aa/temporal_aa_core.py` | **done** |
| `PostProcess/TemporalSuperResolution.cpp` (UE5) | `temporal_aa/temporal_aa_core.py` (TSR stubs) | **todo** |
| `PostProcess/PostProcessTonemap.cpp/.h` · `PostProcess/PostProcessCombineLUTs.cpp/.h` | `color/color_grading.py` (LUT pipeline) | **done** |
| `PostProcess/PostProcessEyeAdaptation.cpp/.h` · `PostProcess/PostProcessHistogram.cpp/.h` | `color/color_extra.py` (exposure stubs) | **todo** |
| `PostProcess/DiaphragmDOF*.cpp/.h` · `PostProcess/PostProcessBokehDOF.cpp/.h` | `postprocess/postprocess_port.py` (DOF stubs) | **todo** |
| `PostProcess/PostProcessMotionBlur.cpp/.h` | `motionblur/motionblur_core.py` | **done** |
| `PostProcess/PostProcessSubsurface.cpp/.h` | `shading/shading_core.py` (SSS stubs) | **todo** |
| `PostProcess/PostProcessAA.cpp/.h` (UE4) · `PostProcess/SubpixelMorphologicalAA.cpp` (UE5) | `temporal_aa/temporal_aa_core.py` (SMAA stub) | **todo** |
| `PostProcess/PostProcessBloomSetup.cpp/.h` · `PostProcess/PostProcessFFTBloom.cpp/.h` | `effects/effects_port.py` (bloom stubs) | **todo** |
| `PostProcess/PostProcessDownsample.cpp/.h` · `PostProcess/PostProcessUpscale.cpp/.h` | `postprocess/postprocess_port.py` | **partial** |
| `PostProcess/SceneRenderTargets.cpp/.h` (UE4) · `SceneTextures.cpp` (UE5) | `resources/resources_core.py` + `resources/resource_pool.py` | **partial** |

### 11. Fog & Atmosphere

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `FogRendering.cpp/.h` | `effects/effects_port.py` (fog stub) | **todo** |
| `VolumetricFog.cpp/.h` · `VolumetricFogVoxelization.cpp` · `VolumetricFogLightFunction.cpp` | `effects/effects_port.py` (volumetric fog stubs) | **todo** |
| `AtmosphereRendering.cpp/.h` · `AtmosphereTextures.cpp/.h` (UE4) | `effects/effects_port.py` | **todo** |
| `SkyAtmosphereRendering.cpp/.h` (UE5) | `effects/effects_port.py` | **todo** |
| `VolumetricCloudRendering.cpp/.h` (UE5) | *(not ported)* | **todo** |
| `LocalFogVolumeRendering.cpp/.h` (UE5) | *(not ported)* | **todo** |

### 12. Translucency

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `TranslucentRendering.cpp/.h` · `TranslucentLighting.cpp` | `translucency/translucency_core.py` | **partial** |
| `MobileTranslucentRendering.cpp` · `MobileSeparateTranslucencyPass.cpp/.h` | `translucency/translucency_core.py` (mobile path) | **todo** |
| `DistortionRendering.cpp/.h` | `effects/effects_port.py` (distortion stubs) | **todo** |
| `FrontLayerTranslucency.cpp/.h` (UE5) | `translucency/translucency_core.py` (front-layer stub) | **todo** |
| `SingleLayerWaterRendering.cpp/.h` (UE5) | *(not ported)* | **todo** |
| `OIT/OIT.cpp/.h` (UE5) | *(not ported)* | **todo** |

### 13. Ray Tracing

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `RayTracing/RayTracingAmbientOcclusion.cpp` · `RayTracingGlobalIllumination.cpp` (UE4) | `effects/effects_port.py` (RT-GI stubs) | **todo** |
| `RayTracing/RayTracingShadows.cpp/.h` (UE4 + UE5) | `shadow/shadow_port.py` (RT shadow stubs) | **todo** |
| `RayTracing/RayTracingReflections.cpp` (UE4) · `RayTracing/RayTracingPrimaryRays.cpp` (UE5) | `reflection/reflection_port.py` (RT reflection stubs) | **todo** |
| `RayTracing/RayTracingMaterialHitShaders.cpp/.h` | `shading/shading_core.py` (hit shader stubs) | **todo** |
| `RayTracing/RayTracingLighting.cpp/.h` | `lighting/lighting_port.py` (RT lighting stubs) | **todo** |
| `ScreenSpaceDenoise.cpp/.h` | `postprocess/postprocess_port.py` (AstroCellDenoiserManager) | **partial** |

### 14. Path Tracing

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `PathTracing.cpp` (UE4) · `PathTracing.cpp/.h` (UE5) | `pathtracing/pathtracing_extra.py` | **partial** |
| `PathTracing/PathCompactionCompute.cpp` · `PathTracing/RayCounterCompute.cpp` (UE4) | `pathtracing/pathtracing_extra.py` (compaction stubs) | **partial** |
| `PathTracingSpatialTemporalDenoising.cpp/.h` (UE5) | `postprocess/postprocess_port.py` (PT denoiser stub) | **todo** |

### 15. Mobile Rendering

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `MobileShadingRenderer.cpp` · `MobileBasePassRendering.cpp/.h` | `passes/passes_port.py` (mobile base pass stub) | **todo** |
| `MobileDeferredShadingPass.cpp/.h` (UE5) | `passes/passes_port.py` | **todo** |
| `MobileBasePass.cpp` · `MobileDecalRendering.cpp` | `passes/passes_port.py` | **todo** |
| `MobileShadowSetup.cpp` | `shadow/shadow_port.py` (mobile shadow stub) | **todo** |
| `MobileReflectionEnvironmentCapture.cpp/.h` | `reflection/reflection_port.py` | **todo** |

### 16. Virtual Textures

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `VT/VirtualTextureSystem.cpp/.h` · `VT/VirtualTextureAllocator.cpp/.h` | `resources/resources_core.py` (VT allocator) | **partial** |
| `VT/VirtualTextureFeedback.cpp/.h` · `VT/TexturePagePool.cpp/.h` | `streaming/streaming_core.py` (feedback stubs) | **partial** |
| `VT/RuntimeVirtualTextureRender.cpp/.h` · `VT/RuntimeVirtualTextureProducer.cpp` | `streaming/streaming_manager.py` | **partial** |
| `RenderCore/Private/VirtualTexturing.cpp` (UE5) | `streaming/streaming_core.py` | **partial** |

### 17. Streaming & Resources

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `SystemTextures.cpp/.h` · `RenderTargetTemp.h` | `resources/resources_core.py` | **partial** |
| `ByteBuffer.cpp/.h` (UE4) · `RenderCore/Private/DynamicBufferAllocator.cpp` (UE5) | `resources/resource_pool.py` | **partial** |
| `Nanite/NaniteStreamOut.cpp/.h` (UE5) | `misc/misc_extra.py` (AstroCellStreamExporter) | **done** |
| `RenderCore/Private/RenderTargetPool.cpp` · `RenderGraphBuilder.cpp` (UE5) | `resources/resource_pool.py` | **partial** |
| `DistanceFieldStreaming.cpp` (UE5) | `streaming/streaming_manager.py` | **todo** |

### 18. Shading Models & Materials

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `ShaderBaseClasses.cpp/.h` · `ShaderComplexityRendering.cpp/.h` | `shading/shading_core.py` | **partial** |
| `DebugViewModeRendering.cpp/.h` | `shading/shading_core.py` (debug view stubs) | **todo** |
| `CompositionLighting/PostProcessAmbient.cpp/.h` | `lighting/lighting_port.py` | **partial** |
| `Substrate/Substrate.cpp/.h` (UE5) | `shading/shading_core.py` (Substrate material stubs) | **todo** |
| `ShadingEnergyConservation.cpp/.h` (UE5) | `shading/shading_core.py` | **todo** |
| `HairStrands/HairStrandsRendering.cpp/.h` (UE5) | *(not ported)* | **todo** |

### 19. Occlusion & HZB

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `SceneOcclusion.cpp/.h` · `SceneVisibility.cpp` | `occlusion/occlusion_core.py` | **partial** |
| `SceneSoftwareOcclusion.cpp/.h` (UE4) · `HZB.cpp/.h` (UE5) | `occlusion/occlusion_core.py` (HZB stubs) | **partial** |
| `Renderer-Private/InstanceCulling/InstanceCullingOcclusionQuery.cpp/.h` (UE5) | `occlusion/occlusion_core.py` | **todo** |

### 20. Acceleration Structures

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `GlobalDistanceField.cpp/.h` · `DistanceFieldGlobalIllumination.cpp/.h` (UE4) | `acceleration/acceleration_core.py` (BVH / AABB) | **partial** |
| `DynamicBVH.cpp/.h` (UE5) | `acceleration/acceleration_core.py` | **partial** |
| `RenderCore/Private/RayTracingGeometry.cpp` (UE5) | `acceleration/acceleration_core.py` (RT-AS stubs) | **todo** |

### 21. Compositor & Final Output

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `PostProcess/RenderingCompositionGraph.cpp/.h` (UE4) | `compositor/compositor_core.py` (AstroCellCompositor) | **done** |
| `PostProcess/PostProcessing.cpp/.h` · `ScreenPass.cpp` (UE5) | `compositor/layer_compositor.py` | **done** |
| `GammaCorrection.cpp` · `HdrCustomResolveShaders.cpp` | `color/color_grading.py` (gamma/HDR) | **partial** |
| `WideCustomResolveShaders.cpp/.h` | `color/color_extra.py` (resolve stubs) | **todo** |

### 22. Species / Decoration (Astro-specific)

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `SceneCore.cpp` (primitive data management) | `species/species_port.py` | **done** |
| `PrimitiveSceneInfo.cpp` (per-primitive state) | `decoration/decoration_extra.py` (AstroCellDecoration) | **done** |
| `EditorPrimitivesRendering.cpp/.h` | `decoration/decoration_extra.py` (editor overlay) | **partial** |

### 23. Style Probe (Astro-specific)

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `ReflectionEnvironmentCapture.cpp` · `SceneCaptureRendering.cpp` | `styleprobe/styleprobe_core.py` + `styleprobe/style_probe_impl.py` | **partial** |
| `VisualizeTexturePresent.cpp/.h` | `styleprobe/styleprobe_core.py` (visualize stubs) | **todo** |

### 24. Utility & Debug

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `GPUBenchmark.cpp/.h` · `GPUFastFourierTransform.cpp/.h` | `misc/misc_extra.py` (GPU benchmark helpers) | **partial** |
| `RendererUtils.cpp` · `SceneViewFamilyBlackboard.cpp/.h` (UE4) | `misc/misc_extra.py` | **partial** |
| `RenderCore/Private/RenderGraphBuilder.cpp` · `RenderGraphUtils.cpp` (UE5) | `resources/resource_pool.py` + `misc/misc_extra.py` | **partial** |
| `VisualizeVolumetricLightmap.cpp` · `VisualizeTexturePresent.cpp/.h` | `misc/misc_extra.py` (visualize stubs) | **todo** |
| `Renderer-Private/ShaderPrint.cpp/.h` (UE5) | `misc/misc_extra.py` | **todo** |

### 25. RHI Layer (UE5 only)

| UE Source File(s) | channels/rendering/ Module | Status |
|---|---|---|
| `RHI/Private/RHICommandList.cpp` · `RHIResources.cpp` | `resources/resources_core.py` (abstract RHI façade) | **partial** |
| `RHI/Private/PipelineStateCache.cpp` · `PipelineFileCache.cpp` | `drawcall/drawcall_extra.py` (PSO table) | **partial** |
| `RHI/Private/DynamicRHI.cpp` · `GPUProfiler.cpp` | `misc/misc_extra.py` (profiler stubs) | **todo** |
| `RHI/Private/RHITransientResourceAllocator.cpp` (UE5) | `resources/resource_pool.py` (transient alloc stubs) | **todo** |
| `RenderCore/Private/RenderingThread.cpp` · `RHIThread.cpp` | *(single-threaded Python event loop — not directly ported)* | **todo** |

---

## Summary Statistics

| Status | Module count |
|--------|-------------|
| done | 8 |
| partial | 17 |
| todo | 5 |

**done**: `lighting`, `compositor`, `nanite/composition`, `nanite/draw_list`,
`nanite/misc (StreamExporter)`, `temporal_aa`, `color/color_grading`, `species`, `decoration`

**partial** (has meaningful implementation, gaps remain): `scene`, `shadow`, `visibility`,
`drawcall`, `passes`, `shading`, `distancefield`, `occlusion`, `lumen`,
`reflection`, `postprocess`, `motionblur`, `translucency`, `streaming`,
`resources`, `acceleration`, `styleprobe`

**todo** (shell/stubs only): `effects` (fog, bloom, atmosphere), `pathtracing` (compaction),
mobile sub-paths, UE5-only systems (Substrate, VSM, MegaLights, Hair, Water, Clouds)

---

## Notes

- UE4 `upstream/unreal-renderer/` covers the classic deferred shading pipeline
  (UE 4.24-era). UE5 `upstream/unreal-renderer-ue5/` adds Nanite, Lumen, VSM,
  Substrate, MegaLights, HeterogeneousVolumes, TSR, and SMAA.
- `channels/rendering/misc/misc_extra.py` (9 489 lines, 263 functions) is the
  largest single file and absorbs several UE sub-systems that don't yet have their
  own dedicated module directory.
- Mobile rendering paths (`MobileShadingRenderer`, `MobileBasePass`, etc.) are
  stubbed throughout but not a priority for the cell-pubsub-loop branch.
- The RHI layer is intentionally thin: Astro targets a single-threaded Python
  event loop backed by PixiJS / SVG, so RHI concepts are approximated as Python
  dict/list abstractions rather than low-level GPU command buffers.
