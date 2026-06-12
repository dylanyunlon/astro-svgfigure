// Copyright Epic Games, Inc. All Rights Reserved.

#include "CoreMinimal.h"
#include "Containers/ArrayView.h"
#include "DataDrivenShaderPlatformInfo.h"
#include "DecalRenderingShared.h"
#include "DeferredShadingRenderer.h"
#include "DistanceFieldLightingShared.h"
#include "GlobalShader.h"
#include "HairStrands/HairStrandsData.h"
#include "HAL/IConsoleManager.h"
#include "Logging/LogMacros.h"
#include "Lumen/Lumen.h"
#include "Lumen/LumenHardwareRayTracingCommon.h"
#include "MegaLights/MegaLights.h"
#include "MegaLightsInternal.h"
#include "MegaLightsViewState.h"
#include "Misc/LargeWorldCoordinates.h"
#include "Nanite/NaniteRayTracing.h"
#include "RayTracing/RaytracingOptions.h"
#include "RayTracing/RayTracingScene.h"
#include "RayTracingPayloadType.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "Rendering/NaniteInterface.h"
#include "RenderUtils.h"
#include "RHI.h"
#include "RHIResources.h"
#include "SceneInterface.h"
#include "ScenePrivate.h"
#include "SceneRendering.h"
#include "SceneTextures.h"
#include "SceneView.h"
#include "Shader.h"
#include "ShaderParameterMacros.h"
#include "ShaderParameterStruct.h"
#include "ShaderParameterUtils.h"
#include "ShaderPrintParameters.h"
#include "ShowFlags.h"
#include "Stats/Stats.h"
#include "VirtualShadowMaps/VirtualShadowMapArray.h"
#include "LogRenderer.h"

static TAutoConsoleVariable<int32> CVarMegaLightsTraceCompactionThreadGroupSize(
	TEXT("r.MegaLights.TraceCompaction.ThreadGroupSize"),
	32,
	TEXT("Size of the trace compaction threadgroup. Larger group is slower to compact, but cam make tracing faster due to improved coherency. Values: 8, 16, 32."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsVolumeTraceCompactionThreadGroupSize(
	TEXT("r.MegaLights.Volume.TraceCompaction.ThreadGroupSize"),
	8,
	TEXT("Size of the volume trace compaction threadgroup. Larger group is slower to compact, but cam make tracing faster due to improved coherency. Values: 4, 8."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsScreenTraces(
	TEXT("r.MegaLights.ScreenTraces"),
	true,
	TEXT("Whether to trace screen space shadow rays."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsScreenTracesQuality(
	TEXT("r.MegaLights.ScreenTraces.Quality"),
	1,
	TEXT("Screen trace quality.\n")
	TEXT("0 - Fast half-resolution traces. Can miss small detail and can have self-occlusion artifacts on some flat surfaces parallel to the ray.\n")
	TEXT("1 - High quality full-resolution traces. Better contact shadows and less artifacts, but also more expensive."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsScreenTracesMaxIterations(
	TEXT("r.MegaLights.ScreenTraces.MaxIterations"),
	50,
	TEXT("Max iterations for HZB tracing."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsScreenTracesMaxDistance(
	TEXT("r.MegaLights.ScreenTraces.MaxDistance"),
	100,
	TEXT("Max distance in world space for screen space tracing."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsScreenTracesMinimumOccupancy(
	TEXT("r.MegaLights.ScreenTraces.MinimumOccupancy"),
	0,
	TEXT("Minimum number of threads still tracing before aborting the trace. Can be used for scalability to abandon traces that have a disproportionate cost."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsScreenTraceRelativeDepthThreshold(
	TEXT("r.MegaLights.ScreenTraces.RelativeDepthThickness"),
	0.005f,
	TEXT("Determines depth thickness of objects hit by HZB tracing, as a relative depth threshold."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsScreenTracesRelativeDepthThicknessWhenNoFallback(
	TEXT("r.MegaLights.ScreenTraces.RelativeDepthThicknessWhenNoFallback"),
	0.01f,
	TEXT("Determines depth thickness of objects hit by HZB tracing, as a relative depth threshold, when there is no world space representation to resume the occluded ray.\n")
	TEXT("(Only used when this value is larger than the main RelativeDepthThickness value)."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsScreenTracesNormalBias(
	TEXT("r.MegaLights.ScreenTraces.NormalBias"),
	0.05f,
	TEXT("Normal bias for screen space shadow rays."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsDistantScreenTraces(
	TEXT("r.MegaLights.DistantScreenTraces"),
	2,
	TEXT("Whether to do a linear screen trace starting where Ray Tracing Scene ends to handle distant shadows.\n")
	TEXT("0 - Off\n")
	TEXT("1 - Enable when not using far field\n")
	TEXT("2 - Enable (default)"),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> GVarMegaLightsDistantScreenTraceDepthThreshold(
	TEXT("r.MegaLights.DistantScreenTraces.DepthThreshold"),
	2.0f,
	TEXT("Depth threshold for the linear screen traces done where other traces have missed."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> GVarMegaLightsDistantScreenTraceLength(
	TEXT("r.MegaLights.DistantScreenTraces.Length"),
	0.2f,
	TEXT("Length of distant screen traces (in screen percentage)."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsWorldSpaceTraces(
	TEXT("r.MegaLights.WorldSpaceTraces"),
	true,
	TEXT("Whether to trace world space shadow rays for samples. Useful for debugging."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsSoftwareRayTracingAllow(
	TEXT("r.MegaLights.SoftwareRayTracing.Allow"),
	0,
	TEXT("Whether to allow using software ray tracing when hardware ray tracing is not supported."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsHardwareRayTracing(
	TEXT("r.MegaLights.HardwareRayTracing"),
	1,
	TEXT("Whether to use hardware ray tracing for shadow rays."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsHardwareRayTracingInline(
	TEXT("r.MegaLights.HardwareRayTracing.Inline"),
	1,
	TEXT("Uses hardware inline ray tracing for ray traced lighting, when available."),
	ECVF_RenderThreadSafe | ECVF_Scalability
);

static TAutoConsoleVariable<int32> CVarMegaLightsHardwareRayTracingEvaluateMaterialMode(
	TEXT("r.MegaLights.HardwareRayTracing.EvaluateMaterialMode"),
	0,
	TEXT("Which mode to use for material evaluation to support alpha masked materials.\n")
	TEXT("0 - Don't evaluate materials (default)\n")
	TEXT("1 - Evaluate materials\n")
	TEXT("2 - Evaluate materials in a separate pass (may be faster on certain platforms without dedicated ray tracing hardware)"),
	ECVF_RenderThreadSafe | ECVF_Scalability
);

static TAutoConsoleVariable<float> CVarMegaLightsHardwareRayTracingBias(
	TEXT("r.MegaLights.HardwareRayTracing.Bias"),
	1.0f,
	TEXT("Constant bias for hardware ray traced shadow rays."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsHardwareRayTracingEndBias(
	TEXT("r.MegaLights.HardwareRayTracing.EndBias"),
	1.0f,
	TEXT("Bias for the end of a shadow ray. Can be used to fix self-occlusion artifacts between a light and nearby geometry like an enclosure mesh."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsHardwareRayTracingNormalBias(
	TEXT("r.MegaLights.HardwareRayTracing.NormalBias"),
	0.1f,
	TEXT("Normal bias for hardware ray traced shadow rays."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsHardwareRayTracingPullbackBias(
	TEXT("r.MegaLights.HardwareRayTracing.PullbackBias"),
	1.0f,
	TEXT("Determines the pull-back bias when resuming a screen-trace ray."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsHardwareRayTracingMaxIterations(
	TEXT("r.MegaLights.HardwareRayTracing.MaxIterations"),
	8192,
	TEXT("Limit number of ray tracing traversal iterations on supported platfoms. Improves performance, but may add over-occlusion."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsHardwareRayTracingMeshSectionVisibilityTest(
	TEXT("r.MegaLights.HardwareRayTracing.MeshSectionVisibilityTest"),
	false,
	TEXT("Whether to test mesh section visibility at runtime.\n")
	TEXT("When enabled translucent mesh sections are automatically hidden based on the material, but it slows down performance due to extra visibility tests per intersection.\n")
	TEXT("When disabled translucent meshes can be hidden only if they are fully translucent. Individual mesh sections need to be hidden upfront inside the static mesh editor."),
	ECVF_RenderThreadSafe | ECVF_Scalability
);

static TAutoConsoleVariable<bool> CVarMegaLightsHardwareRayTracingForceTwoSided(
	TEXT("r.MegaLights.HardwareRayTracing.ForceTwoSided"),
	true,
	TEXT("Whether to force two-sided on all meshes. This greatly speedups ray tracing, but may cause mismatches with rasterization."),
	ECVF_Scalability | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<bool> CVarMegaLightsHardwareRayTracingFarField(
	TEXT("r.MegaLights.HardwareRayTracing.FarField"),
	false,
	TEXT("Determines whether a second trace will be fired for far-field shadowing."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsHardwareRayTracingFarFieldMaxDistance(
	TEXT("r.MegaLights.HardwareRayTracing.FarField.MaxDistance"),
	1.0e8f,
	TEXT("Maximum distance in world space for far-field ray tracing."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsHardwareRayTracingFarFieldBias(
	TEXT("r.MegaLights.HardwareRayTracing.FarField.Bias"),
	200.0f,
	TEXT("Determines bias for the far field traces."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsScreenTracesFirstPerson(
	TEXT("r.MegaLights.ScreenTraces.FirstPerson"),
	true,
	TEXT("Whether to trace screen space shadow rays from First Person Geometry pixels."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsScreenTracesFirstPersonRelativeDepthThickness(
	TEXT("r.MegaLights.ScreenTraces.FirstPerson.RelativeDepthThickness"),
	0.2f,
	TEXT("Determines depth thickness of objects hit by HZB tracing, as a relative depth threshold, when screen space ray hits First Person geometry."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsScreenTracesFirstPersonMinimumHitDistance(
	TEXT("r.MegaLights.ScreenTraces.FirstPerson.MinimumHitDistance"),
	4.0f,
	TEXT("The minimum hit distance when handing off a ray to HWRT after missing in screen space but potentially intersecting first person world space representation primitives.\n")
	TEXT("These primitives are not visible on screen, so we must ensure HWRT has a chance to intersect them, but we also want to ensure that screen tracing covers a minimum distance to avoid self intersection artifacts between Nanite geometry and the fallback meshes in the BVH."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsScreenTracesHair(
	TEXT("r.MegaLights.ScreenTraces.Hair"),
	true,
	TEXT("Whether to trace screen space shadow rays from pixels with a hair shading model (usually hair cards)."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsScreenTracesCompact(
	TEXT("r.MegaLights.ScreenTraces.Compact"),
	true,
	TEXT("Whether to compact for screen traces so all threads do useful work."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsScreenTracesHairBias(
	TEXT("r.MegaLights.ScreenTraces.Hair.Bias"),
	1.0f,
	TEXT("Extra ray bias for rays starting from hair pixels. Increasing this value can reduce screen space trace noise on hair, but also removes some contact shadows."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsHairVoxelTraces(
	TEXT("r.MegaLights.HairVoxelTraces"),
	1,
	TEXT("Whether to trace hair voxels."),
	ECVF_Scalability | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarMegaLightsVolumeWorldSpaceTraces(
	TEXT("r.MegaLights.Volume.WorldSpaceTraces"),
	1,
	TEXT("Whether to trace world space shadow rays for volume samples. Useful for debugging."),
	ECVF_Scalability | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<bool> CVarMegaLightsScreenTracesHairStrands(
	TEXT("r.MegaLights.HairStrands.ScreenTraces"),
	false,
	TEXT("Whether to trace screen space shadow rays in a dedicated Hair Strands rendering pass."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsScreenTracesFrontLayerTranslucency(
	TEXT("r.MegaLights.FrontLayerTranslucency.ScreenTraces"),
	false,
	TEXT("Whether to trace screen space shadow rays in a dedicated Front Layer Translucency rendering pass."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsDebugTraceStats(
	TEXT("r.MegaLights.Debug.TraceStats"),
	0,
	TEXT("Whether to print ray tracing stats on screen."),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<float> CVarMegaLightsHairNonShadowedLightMaxTraceDistance(
	TEXT("r.MegaLights.Hair.NonShadowedLightMaxDistance"),
	20.f,
	TEXT("Maximum distance in world space for non-shadowed lights ray tracing."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsVisualizeRays(
	TEXT("r.MegaLights.VisualizeRays"),
	0,
	TEXT("Whether to visualize traced rays."),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarMegaLightsVisualizeRaysMinIterations(
	TEXT("r.MegaLights.VisualizeRays.MinIterations"),
	160,
	TEXT("Min number of iterations to visualize a ray. Increase this to filter out cheap rays and only show the costliest rays."),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarMegaLightsVisualizeRaysHeatmapMin(
	TEXT("r.MegaLights.VisualizeRays.Heatmap.Min"),
	50,
	TEXT("Min ray heatmap range (blue)."),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarMegaLightsVisualizeRaysHeatmapMax(
	TEXT("r.MegaLights.VisualizeRays.Heatmap.Max"),
	300,
	TEXT("Max ray heatmap range (red)."),
	ECVF_RenderThreadSafe);

namespace MegaLights
{
	bool IsVisualizeRaysEnabled()
	{
		return CVarMegaLightsVisualizeRays.GetValueOnRenderThread() != 0;
	}

	float GetHairNonShadowedLightMaxTraceDistance()
	{
		return CVarMegaLightsHairNonShadowedLightMaxTraceDistance.GetValueOnRenderThread();
	}

	bool IsSoftwareRayTracingSupported(const FSceneViewFamily& ViewFamily)
	{
		return DoesProjectSupportDistanceFields() && CVarMegaLightsSoftwareRayTracingAllow.GetValueOnRenderThread() != 0;
	}

	bool IsHardwareRayTracingSupported(const FSceneViewFamily& ViewFamily)
	{
#if RHI_RAYTRACING
		{
			// Update MegaLights::WriteWarnings(...) when conditions below are changed
			if (IsRayTracingEnabled()
				&& CVarMegaLightsHardwareRayTracing.GetValueOnRenderThread() != 0
				&& (GRHISupportsRayTracingShaders || (GRHISupportsInlineRayTracing && CVarMegaLightsHardwareRayTracingInline.GetValueOnRenderThread() != 0))
				&& ViewFamily.Views[0]->IsRayTracingAllowedForView())
			{
				return true;
			}
		}
#endif

		return false;
	}

	bool UseHardwareRayTracing(const FSceneViewFamily& ViewFamily)
	{
		return MegaLights::IsEnabled(ViewFamily) && IsHardwareRayTracingSupported(ViewFamily);
	}

	bool UseInlineHardwareRayTracing(const FSceneViewFamily& ViewFamily)
	{
		#if RHI_RAYTRACING
		{
			if (UseHardwareRayTracing(ViewFamily)
				&& GRHISupportsInlineRayTracing
				&& CVarMegaLightsHardwareRayTracingInline.GetValueOnRenderThread() != 0)
			{
				return true;
			}
		}
		#endif

		return false;
	}

	bool UseWorldSpaceTraces(const FEngineShowFlags& EngineShowFlags)
	{
		return EngineShowFlags.MegaLightsWorldSpaceTraces
			&& CVarMegaLightsWorldSpaceTraces.GetValueOnRenderThread();
	}

	bool UseHairVoxelTraces(const FViewInfo& View)
	{
		return View.Family->EngineShowFlags.MegaLightsHairVoxelTraces
			&& HairStrands::HasViewHairStrandsData(View)
			&& HairStrands::HasViewHairStrandsVoxelData(View)
			&& CVarMegaLightsHairVoxelTraces.GetValueOnRenderThread() != 0;
	}

	bool UseFarField(const FSceneViewFamily& ViewFamily)
	{
		// #ml_todo: check if far field has any instances
		return ViewFamily.EngineShowFlags.MegaLightsFarFieldTraces 
			&& UseHardwareRayTracing(ViewFamily) 
			&& CVarMegaLightsHardwareRayTracingFarField.GetValueOnRenderThread();
	}

	bool UseScreenTraces(const FEngineShowFlags& EngineShowFlags, EMegaLightsInput InputType)
	{
		const bool bValidMaxDistance = CVarMegaLightsScreenTracesMaxDistance.GetValueOnRenderThread() > 0.0f;

		if (EngineShowFlags.MegaLightsScreenTraces && bValidMaxDistance)
		{
			switch (InputType)
			{
				case EMegaLightsInput::GBuffer: return CVarMegaLightsScreenTraces.GetValueOnRenderThread() != 0;
				case EMegaLightsInput::HairStrands: return CVarMegaLightsScreenTracesHairStrands.GetValueOnRenderThread() != 0;
				case EMegaLightsInput::FrontLayerTranslucency: return CVarMegaLightsScreenTracesFrontLayerTranslucency.GetValueOnRenderThread() != 0;
				default: checkf(false, TEXT("MegaLight::UseScreenTraces not implemented")); return false;
			}
		}

		return false;
	}

	bool UseFirstPersonScreenTraces(const FEngineShowFlags& EngineShowFlags, EMegaLightsInput InputType)
	{
		if (EngineShowFlags.MegaLightsScreenTraces)
		{
			switch (InputType)
			{
				case EMegaLightsInput::GBuffer: return CVarMegaLightsScreenTracesFirstPerson.GetValueOnRenderThread();
				case EMegaLightsInput::HairStrands: return false;
				case EMegaLightsInput::FrontLayerTranslucency: return false;
				default: checkf(false, TEXT("MegaLight::UseFirstPersonScreenTraces not implemented")); return false;
			}
		}

		return false;
	}

	bool UseHairScreenTraces(const FEngineShowFlags& EngineShowFlags, EMegaLightsInput InputType)
	{
		if (EngineShowFlags.MegaLightsScreenTraces)
		{
			switch (InputType)
			{
				case EMegaLightsInput::GBuffer: return CVarMegaLightsScreenTracesHair.GetValueOnRenderThread();
				case EMegaLightsInput::HairStrands: return true;
				case EMegaLightsInput::FrontLayerTranslucency: return false;
				default: checkf(false, TEXT("MegaLight::UseHairScreenTraces not implemented")); return false;
			}
		}

		return false;
	}

	bool UseCompactedScreenTraces(bool bTraceStats)
	{
		return bTraceStats || CVarMegaLightsScreenTracesCompact.GetValueOnRenderThread();
	}

	bool IsUsingClosestHZB(const FSceneViewFamily& ViewFamily)
	{
		if (!IsEnabled(ViewFamily))
		{
			return false;
		}

		for (int32 InputType = 0; InputType < int32(EMegaLightsInput::Count); ++InputType)
		{
			if (UseScreenTraces(ViewFamily.EngineShowFlags, EMegaLightsInput(InputType)))
			{
				return true;
			}
		}

		return false;
	}

	bool IsUsingGlobalSDF(const FSceneViewFamily& ViewFamily)
	{
		return IsEnabled(ViewFamily)
			&& UseWorldSpaceTraces(ViewFamily.EngineShowFlags)
			&& IsSoftwareRayTracingSupported(ViewFamily)
			&& !UseHardwareRayTracing(ViewFamily);
	}

#if RHI_RAYTRACING
	bool IsUsingLightingChannels(const FViewInfo& View, const FRayTracingScene& RayTracingScene)
	{
		return IsUsingLightingChannels() 
			// If all instances and lights share a lighting channel then we can skip testing it
			&& (View.LightGridLightingChannelAndMask & RayTracingScene.LightingChannelAndMask) == 0;
	}
#endif

	bool IsUsingRayTracingRepresentationBit()
	{
		return CVarMegaLightsScreenTraceRelativeDepthThreshold.GetValueOnRenderThread() < CVarMegaLightsScreenTracesRelativeDepthThicknessWhenNoFallback.GetValueOnRenderThread();
	}

	bool ShouldForceTwoSided()
	{
		return CVarMegaLightsHardwareRayTracingForceTwoSided.GetValueOnAnyThread() != 0;
	}

	bool UseDistantScreenTraces(const FViewInfo& View, bool bUseFarField)
	{
		const int32 DistantScreenTraces = CVarMegaLightsDistantScreenTraces.GetValueOnRenderThread();

		return (DistantScreenTraces == 2 || (DistantScreenTraces != 0 && !bUseFarField))
			&& RayTracing::GetCullingMode(View.Family->EngineShowFlags) != RayTracing::ECullingMode::Disabled;
	}

	uint32 GetTraceCompactionWaveOpSize(EShaderPlatform ShaderPlatform)
	{
		uint32 WaveOpWaveSize = 0;

		if (MegaLights::UseWaveOps(ShaderPlatform))
		{
			// 64 wave size is preferred
			if (GRHIMinimumWaveSize <= 64 && GRHIMaximumWaveSize >= 64)
			{
				WaveOpWaveSize = 64;
			}
			else if (GRHIMinimumWaveSize <= 32 && GRHIMaximumWaveSize >= 32)
			{
				WaveOpWaveSize = 32;
			}
		}

		return WaveOpWaveSize;
	}

	enum class ECompactedTraceIndirectArgs
	{
		NumTracesDiv128 = 0 * sizeof(FRHIDispatchIndirectParameters),
		NumTracesDiv64 = 1 * sizeof(FRHIDispatchIndirectParameters),
		NumTracesDiv32 = 2 * sizeof(FRHIDispatchIndirectParameters),
		NumTraces = 3 * sizeof(FRHIDispatchIndirectParameters),
		MAX = 4
	};

	FCompactedTraceParameters CompactMegaLightsTraces(
		const FViewInfo& View,
		FRDGBuilder& GraphBuilder,
		const FIntPoint SampleBufferSize,
		FRDGTextureRef LightSamples,
		const FMegaLightsParameters& MegaLightsParameters,
		EMegaLightsInput InputType,
		bool bCompactForScreenSpaceTraces,
		ERDGPassFlags ComputePassFlags);

	FCompactedTraceParameters CompactMegaLightsVolumeTraces(
		const FViewInfo& View,
		FRDGBuilder& GraphBuilder,
		const FIntVector VolumeSampleBufferSize,
		FRDGTextureRef VolumeLightSampleRays,
		const FMegaLightsParameters& MegaLightsParameters,
		const FMegaLightsVolumeParameters& MegaLightsVolumeParameters,
		ERDGPassFlags ComputePassFlags);

	EMaterialMode GetMaterialMode()
	{
		EMaterialMode MaterialMode = (EMaterialMode)FMath::Clamp(CVarMegaLightsHardwareRayTracingEvaluateMaterialMode.GetValueOnAnyThread(), 0, 2);

		if (!GRHISupportsRayTracingShaders)
		{
			static bool bWarnOnce = true;

			if (bWarnOnce && MaterialMode != EMaterialMode::Disabled)
			{
				UE_LOGF(LogRenderer, Warning, "Ignoring r.MegaLights.HardwareRayTracing.EvaluateMaterialMode because RHI doesn't support ray tracing shaders. Check platform settings.");
				bWarnOnce = false;
			}

			return EMaterialMode::Disabled;
		}

		return MaterialMode;
	}

	bool ShouldUseMeshSectionVisibilityTest()
	{
		return CVarMegaLightsHardwareRayTracingMeshSectionVisibilityTest.GetValueOnRenderThread();
	}

	uint32 GetMaxTraversalInterations()
	{
		return FMath::Max(CVarMegaLightsHardwareRayTracingMaxIterations.GetValueOnRenderThread(), 1);
	}
};

class FCompactLightSampleTracesCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FCompactLightSampleTracesCS)
	SHADER_USE_PARAMETER_STRUCT(FCompactLightSampleTracesCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, RWCompactedTraceTexelData)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, RWCompactedTraceTexelAllocator)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<uint>, LightSampleRays)
		SHADER_PARAMETER(uint32, CompactForScreenSpaceTraces)
		SHADER_PARAMETER(uint32, UseHairScreenTraces)
		SHADER_PARAMETER(uint32, UseFirstPersonScreenTraces)
	END_SHADER_PARAMETER_STRUCT()

	static uint32 GetThreadGroupSize(const uint32 RequestedSize)
	{
		if (RequestedSize <= 8)
		{
			return 8;
		}
		else if (RequestedSize <= 16)
		{
			return 16;
		}
		else
		{
			return 32;
		}
	}

	class FThreadGroupSize : SHADER_PERMUTATION_SPARSE_INT("THREADGROUP_SIZE", 8, 16, 32);
	class FWaveOpWaveSize : SHADER_PERMUTATION_SPARSE_INT("WAVE_OP_WAVE_SIZE", 0, 32, 64);
	class FFastClear : SHADER_PERMUTATION_BOOL("FAST_CLEAR");
	using FPermutationDomain = TShaderPermutationDomain<FThreadGroupSize, FWaveOpWaveSize, FFastClear>;

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (!UE::ShaderPermutationUtils::ShouldCompileWithWaveSize(Parameters, PermutationVector.Get<FWaveOpWaveSize>()))
		{
			return false;
		}

		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FShaderPermutationParameters& Parameters)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (!UE::ShaderPermutationUtils::ShouldPrecacheWithWaveSize(Parameters, PermutationVector.Get<FWaveOpWaveSize>()))
		{
			return EShaderPermutationPrecacheRequest::NotUsed;
		}

		return FGlobalShader::ShouldPrecachePermutation(Parameters);
	}

	FORCENOINLINE static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (PermutationVector.Get<FWaveOpWaveSize>() > 0)
		{
			OutEnvironment.CompilerFlags.Add(CFLAG_WaveOperations);
		}

		if (PermutationVector.Get<FWaveOpWaveSize>() == 32)
		{
			OutEnvironment.CompilerFlags.Add(CFLAG_Wave32);
		}
	}
};

IMPLEMENT_GLOBAL_SHADER(FCompactLightSampleTracesCS, "/Engine/Private/MegaLights/MegaLightsRayTracing.usf", "CompactLightSampleTracesCS", SF_Compute);

class FVolumeCompactLightSampleTracesCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FVolumeCompactLightSampleTracesCS)
	SHADER_USE_PARAMETER_STRUCT(FVolumeCompactLightSampleTracesCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsVolumeParameters, MegaLightsVolumeParameters)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, RWCompactedTraceTexelData)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, RWCompactedTraceTexelAllocator)
		SHADER_PARAMETER_RDG_TEXTURE(Texture3D<uint>, VolumeLightSampleRays)
	END_SHADER_PARAMETER_STRUCT()

	static uint32 GetThreadGroupSize(const uint32 RequestedSize)
	{
		if (RequestedSize <= 4)
		{
			return 4;
		}
		else
		{
			return 8;
		}
	}

	class FThreadGroupSize : SHADER_PERMUTATION_SPARSE_INT("THREADGROUP_SIZE", 4, 8);
	class FWaveOpWaveSize : SHADER_PERMUTATION_SPARSE_INT("WAVE_OP_WAVE_SIZE", 0, 32, 64);
	using FPermutationDomain = TShaderPermutationDomain<FThreadGroupSize, FWaveOpWaveSize>;

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (!UE::ShaderPermutationUtils::ShouldCompileWithWaveSize(Parameters, PermutationVector.Get<FWaveOpWaveSize>()))
		{
			return false;
		}

		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FShaderPermutationParameters& Parameters)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (!UE::ShaderPermutationUtils::ShouldPrecacheWithWaveSize(Parameters, PermutationVector.Get<FWaveOpWaveSize>()))
		{
			return EShaderPermutationPrecacheRequest::NotUsed;
		}

		return FGlobalShader::ShouldPrecachePermutation(Parameters);
	}

	FORCENOINLINE static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (PermutationVector.Get<FWaveOpWaveSize>() > 0)
		{
			OutEnvironment.CompilerFlags.Add(CFLAG_WaveOperations);
		}

		if (PermutationVector.Get<FWaveOpWaveSize>() == 32)
		{
			OutEnvironment.CompilerFlags.Add(CFLAG_Wave32);
		}
	}
};

IMPLEMENT_GLOBAL_SHADER(FVolumeCompactLightSampleTracesCS, "/Engine/Private/MegaLights/MegaLightsVolumeRayTracing.usf", "VolumeCompactLightSampleTracesCS", SF_Compute);

class FInitCompactedTraceTexelIndirectArgsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FInitCompactedTraceTexelIndirectArgsCS)
	SHADER_USE_PARAMETER_STRUCT(FInitCompactedTraceTexelIndirectArgsCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, RWIndirectArgs)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, CompactedTraceTexelAllocator)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}

	static int32 GetGroupSize()
	{
		return 64;
	}
};

IMPLEMENT_GLOBAL_SHADER(FInitCompactedTraceTexelIndirectArgsCS, "/Engine/Private/MegaLights/MegaLightsRayTracing.usf", "InitCompactedTraceTexelIndirectArgsCS", SF_Compute);

class FPrintTraceStatsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FPrintTraceStatsCS)
	SHADER_USE_PARAMETER_STRUCT(FPrintTraceStatsCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, VSMIndirectArgs)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, ScreenIndirectArgs)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, WorldIndirectArgs)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, WorldMaterialRetraceIndirectArgs)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, VolumeIndirectArgs)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, TranslucencyVolume0IndirectArgs)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, TranslucencyVolume1IndirectArgs)
		SHADER_PARAMETER_STRUCT_INCLUDE(ShaderPrint::FShaderParameters, ShaderPrintUniformBuffer)
		END_SHADER_PARAMETER_STRUCT()

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		ShaderPrint::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
		OutEnvironment.SetDefine(TEXT("DEBUG_MODE"), 1);
	}

	static int32 GetGroupSize()
	{
		return 64;
	}
};

IMPLEMENT_GLOBAL_SHADER(FPrintTraceStatsCS, "/Engine/Private/MegaLights/MegaLightsRayTracing.usf", "PrintTraceStatsCS", SF_Compute);

#if RHI_RAYTRACING

class FHardwareRayTraceLightSamples : public FLumenHardwareRayTracingShaderBase
{
	DECLARE_LUMEN_RAYTRACING_SHADER(FHardwareRayTraceLightSamples)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FRayTracingParameters, RayTracingParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FHairVoxelTraceParameters, HairVoxelTraceParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FCompactedTraceParameters, CompactedTraceParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWLightSamples)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWLightSampleRays)
	END_SHADER_PARAMETER_STRUCT()

	class FEvaluateMaterials : SHADER_PERMUTATION_BOOL("MEGA_LIGHTS_EVALUATE_MATERIALS");
	class FLightingChannels : SHADER_PERMUTATION_BOOL("MEGA_LIGHTS_LIGHTING_CHANNELS");
	class FSupportContinuation : SHADER_PERMUTATION_BOOL("SUPPORT_CONTINUATION");
	class FEnableFarFieldTracing : SHADER_PERMUTATION_BOOL("ENABLE_FAR_FIELD_TRACING");
	class FForceTwoSided : SHADER_PERMUTATION_BOOL("FORCE_TWO_SIDED");
	class FHairVoxelTraces : SHADER_PERMUTATION_BOOL("HAIR_VOXEL_TRACES");
	class FDistantScreenTraces : SHADER_PERMUTATION_BOOL("DISTANT_SCREEN_TRACES");
	class FDebugMode : SHADER_PERMUTATION_BOOL("DEBUG_MODE");
	using FPermutationDomain = TShaderPermutationDomain<
		FLumenHardwareRayTracingShaderBase::FBasePermutationDomain,
		FEvaluateMaterials,
		FLightingChannels,
		FSupportContinuation,
		FEnableFarFieldTracing,
		FForceTwoSided,
		FHairVoxelTraces,
		FDistantScreenTraces,
		FDebugMode>;

	static FPermutationDomain RemapPermutation(FPermutationDomain PermutationVector)
	{
		if (PermutationVector.Get<FEvaluateMaterials>())
		{
			PermutationVector.Set<FLightingChannels>(false);
		}

		return PermutationVector;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters, Lumen::ERayTracingShaderDispatchType ShaderDispatchType)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);

		if (RemapPermutation(PermutationVector) != PermutationVector)
		{
			return false;
		}

		if (ShaderDispatchType == Lumen::ERayTracingShaderDispatchType::Inline && PermutationVector.Get<FEvaluateMaterials>())
		{
			return false;
		}

		return MegaLights::ShouldCompileShaders(Parameters.Platform)  
			&& FLumenHardwareRayTracingShaderBase::ShouldCompilePermutation(Parameters, ShaderDispatchType);
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);

		if (PermutationVector.Get<FDebugMode>())
		{
			return EShaderPermutationPrecacheRequest::NotPrecached;
		}

		if (PermutationVector.Get<FForceTwoSided>() != MegaLights::ShouldForceTwoSided())
		{
			return EShaderPermutationPrecacheRequest::NotPrecached;
		}

		bool bMegaLightFarField = CVarMegaLightsHardwareRayTracingFarField.GetValueOnAnyThread();
		if (PermutationVector.Get<FEnableFarFieldTracing>() != bMegaLightFarField)
		{
			return EShaderPermutationPrecacheRequest::NotPrecached;
		}

		// inline code path (no materials)
		if (!PermutationVector.Get<FEvaluateMaterials>())
		{
			if (PermutationVector.Get<FSupportContinuation>())
			{
				return EShaderPermutationPrecacheRequest::NotPrecached;
			}
		}

		return FGlobalShader::ShouldPrecachePermutation(Parameters);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, Lumen::ERayTracingShaderDispatchType ShaderDispatchType, FShaderCompilerEnvironment& OutEnvironment)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (PermutationVector.Get<FDebugMode>())
		{
			// Required for ray iterations in visualize rays
			OutEnvironment.SetDefine(TEXT("ENABLE_TRACE_RAY_INLINE_TRAVERSAL_STATISTICS"), 1);
		}

		FLumenHardwareRayTracingShaderBase::ModifyCompilationEnvironment(Parameters, ShaderDispatchType, Lumen::ESurfaceCacheSampling::AlwaysResidentPagesWithoutFeedback, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
	}

	static ERayTracingPayloadType GetRayTracingPayloadType(const int32 PermutationId)
	{
		FPermutationDomain PermutationVector(PermutationId);
		if (PermutationVector.Get<FEvaluateMaterials>())
		{
			return ERayTracingPayloadType::RayTracingMaterial;
		}
		else
		{
			return ERayTracingPayloadType::LumenMinimal;
		}
	}
};

IMPLEMENT_MEGALIGHT_RAYGEN_AND_COMPUTE_RAYTRACING_SHADERS(FHardwareRayTraceLightSamples)

IMPLEMENT_GLOBAL_SHADER(FHardwareRayTraceLightSamplesCS, "/Engine/Private/MegaLights/MegaLightsHardwareRayTracing.usf", "HardwareRayTraceLightSamplesCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FHardwareRayTraceLightSamplesRGS, "/Engine/Private/MegaLights/MegaLightsHardwareRayTracing.usf", "HardwareRayTraceLightSamplesRGS", SF_RayGen);

class FVolumeHardwareRayTraceLightSamples : public FLumenHardwareRayTracingShaderBase
{
	DECLARE_LUMEN_RAYTRACING_SHADER(FVolumeHardwareRayTraceLightSamples)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FRayTracingParameters, RayTracingParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FCompactedTraceParameters, CompactedTraceParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsVolumeParameters, MegaLightsVolumeParameters)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture3D<uint>, RWVolumeLightSamples)
	END_SHADER_PARAMETER_STRUCT()

	class FTranslucencyLightingVolume : SHADER_PERMUTATION_BOOL("TRANSLUCENCY_LIGHTING_VOLUME");
	class FLightingChannels : SHADER_PERMUTATION_BOOL("MEGA_LIGHTS_LIGHTING_CHANNELS");
	class FEnableFarFieldTracing : SHADER_PERMUTATION_BOOL("ENABLE_FAR_FIELD_TRACING");
	class FForceTwoSided : SHADER_PERMUTATION_BOOL("FORCE_TWO_SIDED");
	class FDebugMode : SHADER_PERMUTATION_BOOL("DEBUG_MODE");
	using FPermutationDomain = TShaderPermutationDomain<FLumenHardwareRayTracingShaderBase::FBasePermutationDomain, FTranslucencyLightingVolume, FLightingChannels, FEnableFarFieldTracing, FForceTwoSided, FDebugMode>;

	static FPermutationDomain RemapPermutation(FPermutationDomain PermutationVector)
	{
		return PermutationVector;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters, Lumen::ERayTracingShaderDispatchType ShaderDispatchType)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);

		if (RemapPermutation(PermutationVector) != PermutationVector)
		{
			return false;
		}

		return MegaLights::ShouldCompileShaders(Parameters.Platform)
			&& FLumenHardwareRayTracingShaderBase::ShouldCompilePermutation(Parameters, ShaderDispatchType);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, Lumen::ERayTracingShaderDispatchType ShaderDispatchType, FShaderCompilerEnvironment& OutEnvironment)
	{
		FLumenHardwareRayTracingShaderBase::ModifyCompilationEnvironment(Parameters, ShaderDispatchType, Lumen::ESurfaceCacheSampling::AlwaysResidentPagesWithoutFeedback, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);
				
		if (PermutationVector.Get<FDebugMode>())
		{
			return EShaderPermutationPrecacheRequest::NotPrecached;
		}
		
		bool bEnableFarField = CVarMegaLightsHardwareRayTracingFarField.GetValueOnAnyThread();
		if (PermutationVector.Get<FEnableFarFieldTracing>() && !bEnableFarField)
		{
			return EShaderPermutationPrecacheRequest::NotPrecached;
		}

		if (PermutationVector.Get<FForceTwoSided>() != MegaLights::ShouldForceTwoSided())
		{
			return EShaderPermutationPrecacheRequest::NotPrecached;
		}

		return CVarMegaLightsVolumeWorldSpaceTraces.GetValueOnAnyThread() != 0 ? FLumenHardwareRayTracingShaderBase::ShouldPrecachePermutation(Parameters) : EShaderPermutationPrecacheRequest::NotPrecached;;
	}

	static ERayTracingPayloadType GetRayTracingPayloadType(const int32 PermutationId)
	{
		return ERayTracingPayloadType::LumenMinimal;
	}
};

IMPLEMENT_MEGALIGHT_RAYGEN_AND_COMPUTE_RAYTRACING_SHADERS(FVolumeHardwareRayTraceLightSamples)

IMPLEMENT_GLOBAL_SHADER(FVolumeHardwareRayTraceLightSamplesCS, "/Engine/Private/MegaLights/MegaLightsVolumeHardwareRayTracing.usf", "VolumeHardwareRayTraceLightSamplesCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FVolumeHardwareRayTraceLightSamplesRGS, "/Engine/Private/MegaLights/MegaLightsVolumeHardwareRayTracing.usf", "VolumeHardwareRayTraceLightSamplesRGS", SF_RayGen);

#endif // RHI_RAYTRACING

class FSoftwareRayTraceLightSamplesCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FSoftwareRayTraceLightSamplesCS)
	SHADER_USE_PARAMETER_STRUCT(FSoftwareRayTraceLightSamplesCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FHairVoxelTraceParameters, HairVoxelTraceParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FCompactedTraceParameters, CompactedTraceParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWLightSamples)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<uint>, LightSampleRays)
	END_SHADER_PARAMETER_STRUCT()

	static int32 GetGroupSize()
	{
		return 64;
	}

	class FHairVoxelTraces : SHADER_PERMUTATION_BOOL("HAIR_VOXEL_TRACES");
	class FDistantScreenTraces : SHADER_PERMUTATION_BOOL("DISTANT_SCREEN_TRACES");
	class FDebugMode : SHADER_PERMUTATION_BOOL("DEBUG_MODE");
	using FPermutationDomain = TShaderPermutationDomain<FHairVoxelTraces, FDistantScreenTraces, FDebugMode>;

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return MegaLights::ShouldCompileShaders(Parameters.Platform) && ShouldCompileDistanceFieldShaders(Parameters.Platform);
	}

	FORCENOINLINE static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		OutEnvironment.CompilerFlags.Add(CFLAG_Wave32);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());

		// GPU Scene definitions
		OutEnvironment.SetDefine(TEXT("VF_SUPPORTS_PRIMITIVE_SCENE_DATA"), 1);
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (PermutationVector.Get<FDebugMode>())
		{
			return EShaderPermutationPrecacheRequest::NotPrecached;
		}
		return FGlobalShader::ShouldPrecachePermutation(Parameters);
	}
};

IMPLEMENT_GLOBAL_SHADER(FSoftwareRayTraceLightSamplesCS, "/Engine/Private/MegaLights/MegaLightsRayTracing.usf", "SoftwareRayTraceLightSamplesCS", SF_Compute);

class FVolumeSoftwareRayTraceLightSamplesCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FVolumeSoftwareRayTraceLightSamplesCS)
	SHADER_USE_PARAMETER_STRUCT(FVolumeSoftwareRayTraceLightSamplesCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FCompactedTraceParameters, CompactedTraceParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsVolumeParameters, MegaLightsVolumeParameters)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture3D<uint>, RWVolumeLightSamples)
	END_SHADER_PARAMETER_STRUCT()

	static int32 GetGroupSize()
	{
		return 64;
	}

	class FDebugMode : SHADER_PERMUTATION_BOOL("DEBUG_MODE");
	using FPermutationDomain = TShaderPermutationDomain<FDebugMode>;

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return MegaLights::ShouldCompileShaders(Parameters.Platform) && ShouldCompileDistanceFieldShaders(Parameters.Platform);
	}

	FORCENOINLINE static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		OutEnvironment.CompilerFlags.Add(CFLAG_Wave32);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());

		// GPU Scene definitions
		OutEnvironment.SetDefine(TEXT("VF_SUPPORTS_PRIMITIVE_SCENE_DATA"), 1);
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (PermutationVector.Get<FDebugMode>())
		{
			return EShaderPermutationPrecacheRequest::NotPrecached;
		}
		return FGlobalShader::ShouldPrecachePermutation(Parameters);
	}
};

IMPLEMENT_GLOBAL_SHADER(FVolumeSoftwareRayTraceLightSamplesCS, "/Engine/Private/MegaLights/MegaLightsVolumeRayTracing.usf", "VolumeSoftwareRayTraceLightSamplesCS", SF_Compute);

class FScreenSpaceRayTraceLightSamplesCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FScreenSpaceRayTraceLightSamplesCS)
	SHADER_USE_PARAMETER_STRUCT(FScreenSpaceRayTraceLightSamplesCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FCompactedTraceParameters, CompactedTraceParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWLightSamples)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWLightSampleRays)
		SHADER_PARAMETER(float, RayTracingNormalBias)
		SHADER_PARAMETER(float, HairScreenTraceBias)
		SHADER_PARAMETER(float, MaxHierarchicalScreenTraceIterations)
		SHADER_PARAMETER(float, MaxTraceDistance)
		SHADER_PARAMETER(float, HairNonShadowedLightMaxTraceDistance)
		SHADER_PARAMETER(float, RelativeDepthThickness)
		SHADER_PARAMETER(float, RelativeDepthThicknessNoFallback)
		SHADER_PARAMETER(float, RelativeDepthThicknessForFirstPerson)
		SHADER_PARAMETER(uint32, MinimumTracingThreadOccupancy)
		SHADER_PARAMETER(uint32, UseRayTracingRepresentationBit)
		SHADER_PARAMETER(uint32, UseHairScreenTraces)
		SHADER_PARAMETER(uint32, UseFirstPersonScreenTraces)
		SHADER_PARAMETER(FVector4f, FirstPersonWorldSpaceRepresentationBounds)
		SHADER_PARAMETER(float, FirstPersonMinimumHitDistanceOnScreenTraceMiss)
	END_SHADER_PARAMETER_STRUCT()

	class FScreenTraceQuality : SHADER_PERMUTATION_INT("SCREEN_TRACE_QUALITY", 2);
	class FUseCompactedTraces : SHADER_PERMUTATION_BOOL("USE_COMPACTED_TRACES");
	class FFastClear : SHADER_PERMUTATION_BOOL("FAST_CLEAR");
	class FDebugMode : SHADER_PERMUTATION_BOOL("DEBUG_MODE");
	using FPermutationDomain = TShaderPermutationDomain<FScreenTraceQuality, FUseCompactedTraces, FFastClear, FDebugMode>;

	static FIntPoint GetGroupSize(const FPermutationDomain& PermutationVector, EShaderPlatform ShaderPlatform)
	{
		const bool bUseCompactedTraces = PermutationVector.Get<FUseCompactedTraces>();
		const int32 ThreadGroupSizeMode = FDataDrivenShaderPlatformInfo::GetThreadGroupSizeMode(ShaderPlatform);

		if (bUseCompactedTraces)
		{
			switch (ThreadGroupSizeMode)
			{
			case 1:
				return FIntPoint(128, 1);
			case 0:
			default:
				return FIntPoint(64, 1);
			}
		}
		else
		{
			switch (ThreadGroupSizeMode)
			{
			case 1:
				return FIntPoint(16, 8);
			case 0:
			default:
				return FIntPoint(8, 8);
			}
		}
	}

	static int32 GetIndirectArgsOffset(const FPermutationDomain& PermutationVector, EShaderPlatform ShaderPlatform)
	{
		const FIntPoint GroupSize = GetGroupSize(PermutationVector, ShaderPlatform);
		check(GroupSize.Y == 1);

		switch (GroupSize.X)
		{
		case 128:
			return (int32)MegaLights::ECompactedTraceIndirectArgs::NumTracesDiv128;
		case 64:
			return (int32)MegaLights::ECompactedTraceIndirectArgs::NumTracesDiv64;
		default:
			checkNoEntry();
			return (int32)MegaLights::ECompactedTraceIndirectArgs::NumTracesDiv64;
		}
	}

	static FPermutationDomain RemapPermutation(FPermutationDomain PermutationVector)
	{
		if (PermutationVector.Get<FUseCompactedTraces>())
		{
			PermutationVector.Set<FFastClear>(false);
		}

		return PermutationVector;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		const FPermutationDomain PermutationVector(Parameters.PermutationId);

		if (RemapPermutation(PermutationVector) != PermutationVector)
		{
			return false;
		}

		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	FORCENOINLINE static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		
		const FPermutationDomain PermutationVector(Parameters.PermutationId);
		const FIntPoint GroupSize = GetGroupSize(PermutationVector, Parameters.Platform);

		OutEnvironment.CompilerFlags.Add(CFLAG_Wave32);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_X"), GroupSize.X);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_Y"), GroupSize.Y);
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (PermutationVector.Get<FDebugMode>())
		{
			return EShaderPermutationPrecacheRequest::NotPrecached;
		}
		return FGlobalShader::ShouldPrecachePermutation(Parameters);
	}
};

IMPLEMENT_GLOBAL_SHADER(FScreenSpaceRayTraceLightSamplesCS, "/Engine/Private/MegaLights/MegaLightsRayTracing.usf", "ScreenSpaceRayTraceLightSamplesCS", SF_Compute);

class FVirtualShadowMapTraceLightSamplesCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FVirtualShadowMapTraceLightSamplesCS)
	SHADER_USE_PARAMETER_STRUCT(FVirtualShadowMapTraceLightSamplesCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FCompactedTraceParameters, CompactedTraceParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FVirtualShadowMapSamplingParameters, VirtualShadowMapSamplingParameters)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWLightSamples)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWLightSampleRays)
	END_SHADER_PARAMETER_STRUCT()

	static int32 GetGroupSize()
	{
		return 64;
	}

	class FDebugMode : SHADER_PERMUTATION_BOOL("DEBUG_MODE");
	using FPermutationDomain = TShaderPermutationDomain<FDebugMode>;

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	FORCENOINLINE static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		FVirtualShadowMapArray::SetShaderDefines(OutEnvironment);
		OutEnvironment.CompilerFlags.Add(CFLAG_Wave32);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (PermutationVector.Get<FDebugMode>())
		{
			return EShaderPermutationPrecacheRequest::NotPrecached;
		}
		return FGlobalShader::ShouldPrecachePermutation(Parameters);
	}
};

IMPLEMENT_GLOBAL_SHADER(FVirtualShadowMapTraceLightSamplesCS, "/Engine/Private/MegaLights/MegaLightsVSMTracing.usf", "VirtualShadowMapTraceLightSamplesCS", SF_Compute);

class FVirtualShadowMapMarkLightSamplesCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FVirtualShadowMapMarkLightSamplesCS)
	SHADER_USE_PARAMETER_STRUCT(FVirtualShadowMapMarkLightSamplesCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FCompactedTraceParameters, CompactedTraceParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FVirtualShadowMapMarkingParameters, VirtualShadowMapMarkingParameters)
		// TODO: These can probably be SRVs in this pass
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWLightSamples)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWLightSampleRays)
	END_SHADER_PARAMETER_STRUCT()

	static int32 GetGroupSize()
	{
		return 64;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	FORCENOINLINE static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		FVirtualShadowMapArray::SetShaderDefines(OutEnvironment);
		//OutEnvironment.CompilerFlags.Add(CFLAG_Wave32);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}
};

IMPLEMENT_GLOBAL_SHADER(FVirtualShadowMapMarkLightSamplesCS, "/Engine/Private/MegaLights/MegaLightsVSMMarking.usf", "VirtualShadowMapMarkLightSamplesCS", SF_Compute);

namespace MegaLights
{
	bool ShouldPrepareDebugModeForRayTracing()
	{
		return MegaLights::IsVisualizeRaysEnabled()
			|| MegaLights::IsDebugEnabled(EMegaLightsDebugMode::GBuffer)
			|| MegaLights::IsDebugEnabled(EMegaLightsDebugMode::HairStrands);
	}
}

#if RHI_RAYTRACING
void FDeferredShadingSceneRenderer::PrepareMegaLightsHardwareRayTracing(const FViewInfo& View, const FScene& Scene, TArray<FRHIRayTracingShader*>& OutRayGenShaders)
{
	const MegaLights::EMaterialMode MaterialMode = MegaLights::GetMaterialMode();
	const bool bUseFarField = MegaLights::UseFarField(*View.Family);

	if (MegaLights::UseHardwareRayTracing(*View.Family) && MaterialMode != MegaLights::EMaterialMode::Disabled)
	{
		// Check if we need additional debug permutation
		const bool bPrepareDebugMode = MegaLights::ShouldPrepareDebugModeForRayTracing();

		for (int32 DebugModeIt = 0; DebugModeIt < (bPrepareDebugMode ? 2 : 1); ++DebugModeIt)
		{
			for (int32 HairVoxelTraces = 0; HairVoxelTraces < 2; ++HairVoxelTraces)
			{
				FHardwareRayTraceLightSamplesRGS::FPermutationDomain PermutationVector;
				PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FEvaluateMaterials>(true);
				PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FLightingChannels>(MegaLights::IsUsingLightingChannels(View, Scene.RayTracingScene));
				PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FSupportContinuation>(false);
				PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FEnableFarFieldTracing>(bUseFarField);
				PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FForceTwoSided>(MegaLights::ShouldForceTwoSided());
				PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FHairVoxelTraces>(HairVoxelTraces != 0);
				PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FDistantScreenTraces>(MegaLights::UseDistantScreenTraces(View, bUseFarField));
				PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FDebugMode>(DebugModeIt != 0);
				PermutationVector = FHardwareRayTraceLightSamplesRGS::RemapPermutation(PermutationVector);

				TShaderRef<FHardwareRayTraceLightSamplesRGS> RayGenerationShader = View.ShaderMap->GetShader<FHardwareRayTraceLightSamplesRGS>(PermutationVector);

				OutRayGenShaders.Add(RayGenerationShader.GetRayTracingShader());
			}
		}
	}
}

void FDeferredShadingSceneRenderer::PrepareMegaLightsHardwareRayTracingMaterial(const FViewInfo& View, const FScene& Scene, TArray<FRHIRayTracingShader*>& OutRayGenShaders)
{
	const MegaLights::EMaterialMode MaterialMode = MegaLights::GetMaterialMode();
	const bool bUseFarField = MegaLights::UseFarField(*View.Family);

	if (MegaLights::UseHardwareRayTracing(*View.Family) && !MegaLights::UseInlineHardwareRayTracing(*View.Family) && MaterialMode != MegaLights::EMaterialMode::AHS)
	{
		// GBuffer
		{
			// Check if we need additional debug permutation
			const bool bPrepareDebugMode = MegaLights::ShouldPrepareDebugModeForRayTracing();

			for (int32 DebugModeIt = 0; DebugModeIt < (bPrepareDebugMode ? 2 : 1); ++DebugModeIt)
			{
				for (int32 HairVoxelTraces = 0; HairVoxelTraces < 2; ++HairVoxelTraces)
				{
					FHardwareRayTraceLightSamplesRGS::FPermutationDomain PermutationVector;
					PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FEvaluateMaterials>(false);
					PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FLightingChannels>(MegaLights::IsUsingLightingChannels(View, Scene.RayTracingScene));
					PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FSupportContinuation>(MaterialMode == MegaLights::EMaterialMode::RetraceAHS);
					PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FEnableFarFieldTracing>(bUseFarField);
					PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FForceTwoSided>(MegaLights::ShouldForceTwoSided());
					PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FHairVoxelTraces>(HairVoxelTraces != 0);
					PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FDistantScreenTraces>(MegaLights::UseDistantScreenTraces(View, bUseFarField));
					PermutationVector.Set<FHardwareRayTraceLightSamplesRGS::FDebugMode>(DebugModeIt != 0);
					PermutationVector = FHardwareRayTraceLightSamplesRGS::RemapPermutation(PermutationVector);

					TShaderRef<FHardwareRayTraceLightSamplesRGS> RayGenerationShader = View.ShaderMap->GetShader<FHardwareRayTraceLightSamplesRGS>(PermutationVector);

					OutRayGenShaders.Add(RayGenerationShader.GetRayTracingShader());
				}
			}
		}

		// Volume
		{
			const bool bVolumeDebug = MegaLights::IsDebugEnabled(EMegaLightsDebugMode::Volume);

			FVolumeHardwareRayTraceLightSamplesRGS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FVolumeHardwareRayTraceLightSamplesRGS::FTranslucencyLightingVolume>(false);
			PermutationVector.Set<FVolumeHardwareRayTraceLightSamplesRGS::FLightingChannels>(MegaLights::IsUsingLightingChannels(View, Scene.RayTracingScene));
			PermutationVector.Set<FVolumeHardwareRayTraceLightSamplesRGS::FEnableFarFieldTracing>(bUseFarField);
			PermutationVector.Set<FVolumeHardwareRayTraceLightSamplesRGS::FForceTwoSided>(MegaLights::ShouldForceTwoSided());
			PermutationVector.Set<FVolumeHardwareRayTraceLightSamplesRGS::FDebugMode>(bVolumeDebug);
			PermutationVector = FVolumeHardwareRayTraceLightSamplesRGS::RemapPermutation(PermutationVector);

			TShaderRef<FVolumeHardwareRayTraceLightSamplesRGS> RayGenerationShader = View.ShaderMap->GetShader<FVolumeHardwareRayTraceLightSamplesRGS>(PermutationVector);

			OutRayGenShaders.Add(RayGenerationShader.GetRayTracingShader());
		}

		// Translucency Volume
		{
			const bool bTranslucencyVolumeDebug = MegaLights::IsDebugEnabled(EMegaLightsDebugMode::TranslucencyVolume);

			FVolumeHardwareRayTraceLightSamplesRGS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FVolumeHardwareRayTraceLightSamplesRGS::FTranslucencyLightingVolume>(true);
			PermutationVector.Set<FVolumeHardwareRayTraceLightSamplesRGS::FLightingChannels>(MegaLights::IsUsingLightingChannels(View, Scene.RayTracingScene));
			PermutationVector.Set<FVolumeHardwareRayTraceLightSamplesRGS::FEnableFarFieldTracing>(bUseFarField);
			PermutationVector.Set<FVolumeHardwareRayTraceLightSamplesRGS::FForceTwoSided>(MegaLights::ShouldForceTwoSided());
			PermutationVector.Set<FVolumeHardwareRayTraceLightSamplesRGS::FDebugMode>(bTranslucencyVolumeDebug);
			PermutationVector = FVolumeHardwareRayTraceLightSamplesRGS::RemapPermutation(PermutationVector);

			TShaderRef<FVolumeHardwareRayTraceLightSamplesRGS> RayGenerationShader = View.ShaderMap->GetShader<FVolumeHardwareRayTraceLightSamplesRGS>(PermutationVector);

			OutRayGenShaders.Add(RayGenerationShader.GetRayTracingShader());
		}
	}
}

#endif

MegaLights::FCompactedTraceParameters MegaLights::CompactMegaLightsTraces(
	const FViewInfo& View,
	FRDGBuilder& GraphBuilder,
	const FIntPoint SampleBufferSize,
	FRDGTextureRef LightSampleRays,
	const FMegaLightsParameters& MegaLightsParameters,
	EMegaLightsInput InputType,
	bool bCompactForScreenSpaceTraces,
	ERDGPassFlags ComputePassFlags)
{
	FRDGBufferRef CompactedTraceTexelData = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateBufferDesc(sizeof(uint32), SampleBufferSize.X * SampleBufferSize.Y),
		MEGALIGHTS_RESOURCE_NAME("CompactedTraceTexelData"));

	FRDGBufferRef CompactedTraceTexelAllocator = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateBufferDesc(sizeof(uint32), 1),
		MEGALIGHTS_RESOURCE_NAME("CompactedTraceTexelAllocator"));

	FRDGBufferRef CompactedTraceTexelIndirectArgs = GraphBuilder.CreateBuffer(
		FRDGBufferDesc::CreateIndirectDesc<FRHIDispatchIndirectParameters>((int32)ECompactedTraceIndirectArgs::MAX),
		MEGALIGHTS_RESOURCE_NAME("CompactedTraceTexelIndirectArgs"));

	AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(CompactedTraceTexelAllocator, PF_R32_UINT), 0, ComputePassFlags);

	// Compact light sample traces before tracing
	{
		FCompactLightSampleTracesCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FCompactLightSampleTracesCS::FParameters>();
		PassParameters->RWCompactedTraceTexelData = GraphBuilder.CreateUAV(CompactedTraceTexelData, PF_R32_UINT);
		PassParameters->RWCompactedTraceTexelAllocator = GraphBuilder.CreateUAV(CompactedTraceTexelAllocator, PF_R32_UINT);
		PassParameters->MegaLightsParameters = MegaLightsParameters;
		PassParameters->LightSampleRays = LightSampleRays;
		PassParameters->CompactForScreenSpaceTraces = bCompactForScreenSpaceTraces ? 1 : 0;
		PassParameters->UseHairScreenTraces = UseHairScreenTraces(View.Family->EngineShowFlags, InputType) ? 1 : 0;
		PassParameters->UseFirstPersonScreenTraces = UseFirstPersonScreenTraces(View.Family->EngineShowFlags, InputType) ? 1 : 0;

		const uint32 TraceCompactionThreadGroupSize = FCompactLightSampleTracesCS::GetThreadGroupSize(CVarMegaLightsTraceCompactionThreadGroupSize.GetValueOnRenderThread());
		const uint32 WaveOpWaveSize = MegaLights::GetTraceCompactionWaveOpSize(View.GetShaderPlatform());

		FCompactLightSampleTracesCS::FPermutationDomain PermutationVector;
		PermutationVector.Set<FCompactLightSampleTracesCS::FThreadGroupSize>(TraceCompactionThreadGroupSize);
		PermutationVector.Set<FCompactLightSampleTracesCS::FWaveOpWaveSize>(WaveOpWaveSize);
		PermutationVector.Set<FCompactLightSampleTracesCS::FFastClear>(MegaLights::UseFastClear(InputType));
		auto ComputeShader = View.ShaderMap->GetShader<FCompactLightSampleTracesCS>(PermutationVector);

		const FIntVector GroupCount = FComputeShaderUtils::GetGroupCount(MegaLightsParameters.SampleViewSize, TraceCompactionThreadGroupSize);

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("CompactLightSampleTraces WaveOpSize:%d", WaveOpWaveSize),
			ComputePassFlags,
			ComputeShader,
			PassParameters,
			GroupCount);
	}

	// Setup indirect args for tracing
	{
		FInitCompactedTraceTexelIndirectArgsCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FInitCompactedTraceTexelIndirectArgsCS::FParameters>();
		PassParameters->MegaLightsParameters = MegaLightsParameters;
		PassParameters->RWIndirectArgs = GraphBuilder.CreateUAV(CompactedTraceTexelIndirectArgs);
		PassParameters->CompactedTraceTexelAllocator = GraphBuilder.CreateSRV(CompactedTraceTexelAllocator, PF_R32_UINT);

		auto ComputeShader = View.ShaderMap->GetShader<FInitCompactedTraceTexelIndirectArgsCS>();

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("InitCompactedTraceTexelIndirectArgs"),
			ComputePassFlags,
			ComputeShader,
			PassParameters,
			FIntVector(1, 1, 1));
	}

	FCompactedTraceParameters Parameters;
	Parameters.CompactedTraceTexelAllocator = GraphBuilder.CreateSRV(CompactedTraceTexelAllocator, PF_R32_UINT);
	Parameters.CompactedTraceTexelData = GraphBuilder.CreateSRV(CompactedTraceTexelData, PF_R32_UINT);
	Parameters.IndirectArgs = CompactedTraceTexelIndirectArgs;
	return Parameters;
}

MegaLights::FCompactedTraceParameters MegaLights::CompactMegaLightsVolumeTraces(
	const FViewInfo& View,
	FRDGBuilder& GraphBuilder,
	const FIntVector VolumeSampleBufferSize,
	FRDGTextureRef VolumeLightSampleRays,
	const FMegaLightsParameters& MegaLightsParameters,
	const FMegaLightsVolumeParameters& MegaLightsVolumeParameters,
	ERDGPassFlags ComputePassFlags)
{
	FRDGBufferRef CompactedTraceTexelData = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateBufferDesc(sizeof(uint32), VolumeSampleBufferSize.X * VolumeSampleBufferSize.Y * VolumeSampleBufferSize.Z),
		TEXT("MegaLights.CompactedVolumeTraceTexelData"));

	FRDGBufferRef CompactedTraceTexelAllocator = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateBufferDesc(sizeof(uint32), 1),
		TEXT("MegaLights.CompactedVolumeTraceTexelAllocator"));

	FRDGBufferRef CompactedTraceTexelIndirectArgs = GraphBuilder.CreateBuffer(
		FRDGBufferDesc::CreateIndirectDesc<FRHIDispatchIndirectParameters>((int32)ECompactedTraceIndirectArgs::MAX),
		TEXT("MegaLights.CompactedVolumeTraceTexelIndirectArgs"));

	AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(CompactedTraceTexelAllocator, PF_R32_UINT), 0, ComputePassFlags);

	// Compact light sample traces before tracing
	{
		FVolumeCompactLightSampleTracesCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FVolumeCompactLightSampleTracesCS::FParameters>();
		PassParameters->RWCompactedTraceTexelData = GraphBuilder.CreateUAV(CompactedTraceTexelData, PF_R32_UINT);
		PassParameters->RWCompactedTraceTexelAllocator = GraphBuilder.CreateUAV(CompactedTraceTexelAllocator, PF_R32_UINT);
		PassParameters->MegaLightsParameters = MegaLightsParameters;
		PassParameters->MegaLightsVolumeParameters = MegaLightsVolumeParameters;
		PassParameters->VolumeLightSampleRays = VolumeLightSampleRays;

		const uint32 TraceCompactionThreadGroupSize = FVolumeCompactLightSampleTracesCS::GetThreadGroupSize(CVarMegaLightsVolumeTraceCompactionThreadGroupSize.GetValueOnRenderThread());
		const uint32 WaveOpWaveSize = MegaLights::GetTraceCompactionWaveOpSize(View.GetShaderPlatform());

		FVolumeCompactLightSampleTracesCS::FPermutationDomain PermutationVector;
		PermutationVector.Set<FVolumeCompactLightSampleTracesCS::FThreadGroupSize>(TraceCompactionThreadGroupSize);
		PermutationVector.Set<FVolumeCompactLightSampleTracesCS::FWaveOpWaveSize>(WaveOpWaveSize);
		auto ComputeShader = View.ShaderMap->GetShader<FVolumeCompactLightSampleTracesCS>(PermutationVector);

		const FIntVector GroupCount = FComputeShaderUtils::GetGroupCount(MegaLightsVolumeParameters.VolumeSampleViewSize, TraceCompactionThreadGroupSize);

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("CompactVolumeLightSampleTraces WaveOpSize:%d", WaveOpWaveSize),
			ComputePassFlags,
			ComputeShader,
			PassParameters,
			GroupCount);
	}

	// Setup indirect args for tracing
	{
		FInitCompactedTraceTexelIndirectArgsCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FInitCompactedTraceTexelIndirectArgsCS::FParameters>();
		PassParameters->MegaLightsParameters = MegaLightsParameters;
		PassParameters->RWIndirectArgs = GraphBuilder.CreateUAV(CompactedTraceTexelIndirectArgs);
		PassParameters->CompactedTraceTexelAllocator = GraphBuilder.CreateSRV(CompactedTraceTexelAllocator, PF_R32_UINT);

		auto ComputeShader = View.ShaderMap->GetShader<FInitCompactedTraceTexelIndirectArgsCS>();

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("InitCompactedVolumeTraceTexelIndirectArgs"),
			ComputePassFlags,
			ComputeShader,
			PassParameters,
			FIntVector(1, 1, 1));
	}

	FCompactedTraceParameters Parameters;
	Parameters.CompactedTraceTexelAllocator = GraphBuilder.CreateSRV(CompactedTraceTexelAllocator, PF_R32_UINT);
	Parameters.CompactedTraceTexelData = GraphBuilder.CreateSRV(CompactedTraceTexelData, PF_R32_UINT);
	Parameters.IndirectArgs = CompactedTraceTexelIndirectArgs;
	return Parameters;
}

void MegaLights::MarkVSMPages(
	const FViewInfo& View,
	int32 ViewIndex,
	FRDGBuilder& GraphBuilder,
	const FVirtualShadowMapArray& VirtualShadowMapArray,
	const FIntPoint SampleBufferSize,
	FRDGTextureRef LightSamples,
	FRDGTextureRef LightSampleRays,
	const FMegaLightsParameters& MegaLightsParameters,
	EMegaLightsInput InputType)
{
	// TODO: This pass doesn't remove any traces so we should perhaps convert the compaction to a lazy dirty bit
	FCompactedTraceParameters CompactedTraceParameters = MegaLights::CompactMegaLightsTraces(
		View,
		GraphBuilder,
		SampleBufferSize,
		LightSampleRays,
		MegaLightsParameters,
		InputType,
		/*bCompactForScreenSpaceTraces*/ false,
		ERDGPassFlags::Compute);

	FVirtualShadowMapMarkLightSamplesCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FVirtualShadowMapMarkLightSamplesCS::FParameters>();
	PassParameters->CompactedTraceParameters = CompactedTraceParameters;
	PassParameters->MegaLightsParameters = MegaLightsParameters;
	PassParameters->RWLightSamples = GraphBuilder.CreateUAV(LightSamples);
	PassParameters->RWLightSampleRays = GraphBuilder.CreateUAV(LightSampleRays);
	PassParameters->VirtualShadowMapMarkingParameters = VirtualShadowMapArray.GetMarkingParameters(GraphBuilder, ViewIndex);

	FVirtualShadowMapMarkLightSamplesCS::FPermutationDomain PermutationVector;
	//PermutationVector.Set<FVirtualShadowMapMarkLightSamplesCS::FDebugMode>(bDebug);
	auto ComputeShader = View.ShaderMap->GetShader<FVirtualShadowMapMarkLightSamplesCS>(PermutationVector);

	FComputeShaderUtils::AddPass(
		GraphBuilder,
		RDG_EVENT_NAME("VirtualShadowMapMarkLightSamples"),
		ComputeShader,
		PassParameters,
		CompactedTraceParameters.IndirectArgs,
		(int32)MegaLights::ECompactedTraceIndirectArgs::NumTracesDiv64);
}

template <typename TraceStatsType>
MegaLights::FVolumeCompactedTraceParameters CompactVolumeTraces(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	const FMegaLightsParameters& MegaLightsParameters,
	const FMegaLightsVolumeParameters& MegaLightsVolumeParameters,
	const FMegaLightsVolumeParameters& MegaLightsTranslucencyVolumeParameters,
	const MegaLights::FVolumeLightSampleParameters& LightSampleParameters,
	ERDGPassFlags ComputePassFlags,
	FRDGBufferAccessArray& OutBatchBarrierBuffers,
	TraceStatsType& OutTraceStats)
{
	MegaLights::FVolumeCompactedTraceParameters CompactedTraceParameters;

	if (LightSampleParameters.VolumeLightSamples)
	{
		CompactedTraceParameters.Volume = MegaLights::CompactMegaLightsVolumeTraces(
			View,
			GraphBuilder,
			LightSampleParameters.VolumeSampleBufferSize,
			LightSampleParameters.VolumeLightSampleRays,
			MegaLightsParameters,
			MegaLightsVolumeParameters,
			ComputePassFlags);

		OutBatchBarrierBuffers.Emplace(CompactedTraceParameters.Volume.IndirectArgs, ERHIAccess::SRVCompute);
		OutTraceStats.Volume = CompactedTraceParameters.Volume.IndirectArgs;
	}

	if (!LightSampleParameters.TranslucencyVolumeLightSamples.IsEmpty())
	{
		FMegaLightsVolumeParameters CascadeMegaLightsParameters = MegaLightsTranslucencyVolumeParameters;

		for (int32 CascadeIndex = 0; CascadeIndex < TVC_MAX; ++CascadeIndex)
		{
			CascadeMegaLightsParameters.TranslucencyVolumeCascadeIndex = CascadeIndex;

			CompactedTraceParameters.TranslucencyVolume[CascadeIndex] = MegaLights::CompactMegaLightsVolumeTraces(
				View,
				GraphBuilder,
				LightSampleParameters.TranslucencyVolumeSampleBufferSize,
				LightSampleParameters.TranslucencyVolumeLightSampleRays[CascadeIndex],
				MegaLightsParameters,
				CascadeMegaLightsParameters,
				ComputePassFlags);

			OutBatchBarrierBuffers.Emplace(CompactedTraceParameters.TranslucencyVolume[CascadeIndex].IndirectArgs, ERHIAccess::SRVCompute);
			OutTraceStats.TranslucencyVolume[CascadeIndex] = CompactedTraceParameters.TranslucencyVolume[CascadeIndex].IndirectArgs;
		}
	}

	return CompactedTraceParameters;
}

MegaLights::FTraceStats MegaLights::RayTraceVolumeLightSamples(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	const FSceneViewFamily& ViewFamily,
	const FMegaLightsParameters& MegaLightsParameters,
	const FMegaLightsVolumeParameters& MegaLightsVolumeParameters,
	const FMegaLightsVolumeParameters& MegaLightsTranslucencyVolumeParameters,
	const FVolumeLightSampleParameters& LightSampleParameters,
	const FVolumeCompactedTraceParameters* CompactedTraceParameters,
	ERDGPassFlags ComputePassFlags)
{
	const bool bVolumeDebug = MegaLights::IsDebugEnabled(EMegaLightsDebugMode::Volume);
	const bool bTranslucencyVolumeDebug = MegaLights::IsDebugEnabled(EMegaLightsDebugMode::TranslucencyVolume);

	FTraceStats TraceStats;

	if (UseWorldSpaceTraces(ViewFamily.EngineShowFlags)
		&& CVarMegaLightsVolumeWorldSpaceTraces.GetValueOnRenderThread() != 0)
	{
		if (CompactedTraceParameters != nullptr)
		{
			TraceStats.Volume = CompactedTraceParameters->Volume.IndirectArgs;

			for (int32 CascadeIndex = 0; CascadeIndex < TVC_MAX; ++CascadeIndex)
			{
				TraceStats.TranslucencyVolume[CascadeIndex] = CompactedTraceParameters->TranslucencyVolume[CascadeIndex].IndirectArgs;
			}
		}
		else
		{
			FVolumeCompactedTraceParameters* LocalCompactedTraceParameters = GraphBuilder.AllocParameters<FVolumeCompactedTraceParameters>();
			FRDGBufferAccessArray BatchBarrierBuffers;

			*LocalCompactedTraceParameters = CompactVolumeTraces(
				GraphBuilder,
				View,
				MegaLightsParameters,
				MegaLightsVolumeParameters,
				MegaLightsTranslucencyVolumeParameters,
				LightSampleParameters,
				ComputePassFlags,
				BatchBarrierBuffers,
				TraceStats);

			AddBarrierPass(GraphBuilder, ComputePassFlags, BatchBarrierBuffers);

			CompactedTraceParameters = LocalCompactedTraceParameters;
		}

		if (MegaLights::UseHardwareRayTracing(ViewFamily))
		{
#if RHI_RAYTRACING
			const FScene* Scene = static_cast<const FScene*>(ViewFamily.Scene);
			const FRayTracingScene& RayTracingScene = Scene->RayTracingScene;

			const bool bUseFarField = MegaLights::UseFarField(ViewFamily);

			if (LightSampleParameters.VolumeLightSamples)
			{
				FVolumeHardwareRayTraceLightSamples::FParameters* PassParameters = GraphBuilder.AllocParameters<FVolumeHardwareRayTraceLightSamples::FParameters>();
				SetRayTracingParameters(GraphBuilder, View, PassParameters->RayTracingParameters);
				PassParameters->CompactedTraceParameters = CompactedTraceParameters->Volume;
				PassParameters->MegaLightsParameters = MegaLightsParameters;
				PassParameters->MegaLightsVolumeParameters = MegaLightsVolumeParameters;
				PassParameters->RWVolumeLightSamples = GraphBuilder.CreateUAV(LightSampleParameters.VolumeLightSamples);

				FVolumeHardwareRayTraceLightSamples::FPermutationDomain PermutationVector;
				PermutationVector.Set<FVolumeHardwareRayTraceLightSamples::FTranslucencyLightingVolume>(false);
				PermutationVector.Set<FVolumeHardwareRayTraceLightSamples::FLightingChannels>(MegaLights::IsUsingLightingChannels(View, RayTracingScene));
				PermutationVector.Set<FVolumeHardwareRayTraceLightSamples::FEnableFarFieldTracing>(bUseFarField);
				PermutationVector.Set<FVolumeHardwareRayTraceLightSamples::FForceTwoSided>(MegaLights::ShouldForceTwoSided());
				PermutationVector.Set<FVolumeHardwareRayTraceLightSamples::FDebugMode>(bVolumeDebug);
				PermutationVector = FVolumeHardwareRayTraceLightSamples::RemapPermutation(PermutationVector);

				if (MegaLights::UseInlineHardwareRayTracing(ViewFamily))
				{
					FVolumeHardwareRayTraceLightSamplesCS::AddMegaLightRayTracingDispatchIndirect(
						GraphBuilder,
						RDG_EVENT_NAME("VolumeHardwareRayTraceLightSamples Inline"),
						View,
						PermutationVector,
						PassParameters,
						CompactedTraceParameters->Volume.IndirectArgs,
						(int32)MegaLights::ECompactedTraceIndirectArgs::NumTracesDiv32,
						ComputePassFlags);
				}
				else
				{
					FVolumeHardwareRayTraceLightSamplesRGS::AddMegaLightRayTracingDispatchIndirect(
						GraphBuilder,
						RDG_EVENT_NAME("VolumeHardwareRayTraceLightSamples RayGen"),
						View,
						PermutationVector,
						PassParameters,
						PassParameters->CompactedTraceParameters.IndirectArgs,
						(int32)MegaLights::ECompactedTraceIndirectArgs::NumTraces,
						/*bUseMinimalPayload*/ true,
						ComputePassFlags);
				}
			}

			// Translucency Volume
			if (!LightSampleParameters.TranslucencyVolumeLightSamples.IsEmpty())
			{
				for (uint32 CascadeIndex = 0; CascadeIndex < TVC_MAX; ++CascadeIndex)
				{
					FMegaLightsVolumeParameters CascadeMegaLightsParameters = MegaLightsTranslucencyVolumeParameters;
					CascadeMegaLightsParameters.TranslucencyVolumeCascadeIndex = CascadeIndex;

					FVolumeHardwareRayTraceLightSamples::FParameters* PassParameters = GraphBuilder.AllocParameters<FVolumeHardwareRayTraceLightSamples::FParameters>();
					SetRayTracingParameters(GraphBuilder, View, PassParameters->RayTracingParameters);
					PassParameters->CompactedTraceParameters = CompactedTraceParameters->TranslucencyVolume[CascadeIndex];
					PassParameters->MegaLightsParameters = MegaLightsParameters;
					PassParameters->MegaLightsVolumeParameters = CascadeMegaLightsParameters;
					PassParameters->RWVolumeLightSamples = GraphBuilder.CreateUAV(LightSampleParameters.TranslucencyVolumeLightSamples[CascadeIndex]);

					FVolumeHardwareRayTraceLightSamples::FPermutationDomain PermutationVector;
					PermutationVector.Set<FVolumeHardwareRayTraceLightSamples::FTranslucencyLightingVolume>(true);
					PermutationVector.Set<FVolumeHardwareRayTraceLightSamples::FLightingChannels>(MegaLights::IsUsingLightingChannels(View, RayTracingScene));
					PermutationVector.Set<FVolumeHardwareRayTraceLightSamples::FEnableFarFieldTracing>(bUseFarField);
					PermutationVector.Set<FVolumeHardwareRayTraceLightSamples::FForceTwoSided>(MegaLights::ShouldForceTwoSided());
					PermutationVector.Set<FVolumeHardwareRayTraceLightSamples::FDebugMode>(bTranslucencyVolumeDebug);
					PermutationVector = FVolumeHardwareRayTraceLightSamples::RemapPermutation(PermutationVector);

					if (MegaLights::UseInlineHardwareRayTracing(ViewFamily))
					{
						FVolumeHardwareRayTraceLightSamplesCS::AddMegaLightRayTracingDispatchIndirect(
							GraphBuilder,
							RDG_EVENT_NAME("TranslucencyVolumeHardwareRayTraceLightSamples Inline"),
							View,
							PermutationVector,
							PassParameters,
							CompactedTraceParameters->TranslucencyVolume[CascadeIndex].IndirectArgs,
							(int32)MegaLights::ECompactedTraceIndirectArgs::NumTracesDiv32,
							ComputePassFlags);
					}
					else
					{
						FVolumeHardwareRayTraceLightSamplesRGS::AddMegaLightRayTracingDispatchIndirect(
							GraphBuilder,
							RDG_EVENT_NAME("TranslucencyVolumeHardwareRayTraceLightSamples RayGen"),
							View,
							PermutationVector,
							PassParameters,
							PassParameters->CompactedTraceParameters.IndirectArgs,
							(int32)MegaLights::ECompactedTraceIndirectArgs::NumTraces,
							/*bUseMinimalPayload*/ true,
							ComputePassFlags);
					}
				}
			}
#endif // RHI_RAYTRACING
		}
		else
		{
			ensure(MegaLights::IsUsingGlobalSDF(ViewFamily));

			// Volume
			if (LightSampleParameters.VolumeLightSamples)
			{
				FVolumeSoftwareRayTraceLightSamplesCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FVolumeSoftwareRayTraceLightSamplesCS::FParameters>();
				PassParameters->CompactedTraceParameters = CompactedTraceParameters->Volume;
				PassParameters->MegaLightsParameters = MegaLightsParameters;
				PassParameters->MegaLightsVolumeParameters = MegaLightsVolumeParameters;
				PassParameters->RWVolumeLightSamples = GraphBuilder.CreateUAV(LightSampleParameters.VolumeLightSamples);

				FVolumeSoftwareRayTraceLightSamplesCS::FPermutationDomain PermutationVector;
				PermutationVector.Set<FVolumeSoftwareRayTraceLightSamplesCS::FDebugMode>(bVolumeDebug);
				auto ComputeShader = View.ShaderMap->GetShader<FVolumeSoftwareRayTraceLightSamplesCS>(PermutationVector);

				FComputeShaderUtils::AddPass(
					GraphBuilder,
					RDG_EVENT_NAME("VolumeSoftwareRayTraceLightSamples"),
					ComputePassFlags,
					ComputeShader,
					PassParameters,
					CompactedTraceParameters->Volume.IndirectArgs,
					(int32)MegaLights::ECompactedTraceIndirectArgs::NumTracesDiv64);
			}

			// TODO: Translucency Volume
		}
	}

	return TraceStats;
}

void MegaLights::SetHairVoxelTraceParameters(const FViewInfo& View, MegaLights::FHairVoxelTraceParameters& Parameters)
{
	Parameters.HairStrands = HairStrands::BindHairStrandsViewUniformParameters(View);
	Parameters.VirtualVoxel = HairStrands::BindHairStrandsVoxelUniformParameters(View);
}

void MegaLights::SetRayTracingParameters(FRDGBuilder& GraphBuilder, const FViewInfo& View, MegaLights::FRayTracingParameters& Parameters)
{
#if RHI_RAYTRACING
	const FScene* Scene = View.Family->Scene->GetRenderScene();
	const bool bUseFarField = MegaLights::UseFarField(*View.Family);

	// Bias
	Parameters.RayTracingBias = CVarMegaLightsHardwareRayTracingBias.GetValueOnRenderThread();
	Parameters.RayTracingNormalBias = CVarMegaLightsHardwareRayTracingNormalBias.GetValueOnRenderThread();
	Parameters.RayTracingPullbackBias = CVarMegaLightsHardwareRayTracingPullbackBias.GetValueOnRenderThread();
	Parameters.HairNonShadowedLightMaxTraceDistance = MegaLights::GetHairNonShadowedLightMaxTraceDistance();

	// Distant Screen Traces
	Parameters.DistantScreenTraceSlopeCompareTolerance = GVarMegaLightsDistantScreenTraceDepthThreshold.GetValueOnRenderThread();
	Parameters.DistantScreenTraceStartDistance = RayTracing::GetCullingMode(View.Family->EngineShowFlags) != RayTracing::ECullingMode::Disabled ? GetRayTracingCullingRadius() : FLT_MAX;
	Parameters.DistantScreenTraceLength = GVarMegaLightsDistantScreenTraceLength.GetValueOnRenderThread();

	// Visualize rays
	Parameters.VisualizeRays = MegaLights::IsVisualizeRaysEnabled() ? 1 : 0;
	Parameters.VisualizeRaysMinIterations = FMath::Max(CVarMegaLightsVisualizeRaysMinIterations.GetValueOnRenderThread(), 0);
	Parameters.VisualizeRaysHeatmapMin = FMath::Max(CVarMegaLightsVisualizeRaysHeatmapMin.GetValueOnRenderThread(), 0);
	Parameters.VisualizeRaysHeatmapMax = FMath::Max(CVarMegaLightsVisualizeRaysHeatmapMax.GetValueOnRenderThread(), 0);

	// #ml_todo: should use MegaLights specific far field tracing configuration instead of sharing Lumen config?
	Parameters.NearFieldSceneRadius = Lumen::GetNearFieldSceneRadius(View, bUseFarField);
	Parameters.NearFieldMaxTraceDistance = Lumen::MaxTraceDistance;
	Parameters.NearFieldMaxTraceDistanceDitherScale = Lumen::GetNearFieldMaxTraceDistanceDitherScale(bUseFarField);
	Parameters.FarFieldBias = CVarMegaLightsHardwareRayTracingFarFieldBias.GetValueOnRenderThread();
	Parameters.FarFieldMaxTraceDistance = CVarMegaLightsHardwareRayTracingFarFieldMaxDistance.GetValueOnRenderThread();

	Parameters.UseFarField = bUseFarField ? 1 : 0;
	Parameters.ForceTwoSided = MegaLights::ShouldForceTwoSided() ? 1 : 0;
	Parameters.MaxTraversalIterations = MegaLights::GetMaxTraversalInterations();
	Parameters.MeshSectionVisibilityTest = MegaLights::ShouldUseMeshSectionVisibilityTest() ? 1 : 0;
	Parameters.RayTracingSceneLightingChannelAndMask = Scene ? Scene->RayTracingScene.LightingChannelAndMask : 0xFFFFFFFF;

	checkf(View.HasRayTracingScene(), TEXT("TLAS does not exist. Verify that the current pass is represented in MegaLights::UseHardwareRayTracing()"));
	Parameters.TLAS = View.GetRayTracingSceneLayerViewChecked(ERayTracingSceneLayer::Base);
	Parameters.FarFieldTLAS = View.GetRayTracingSceneLayerViewChecked(ERayTracingSceneLayer::FarField);
	Parameters.HitGroupData = View.LumenHardwareRayTracingHitDataBuffer ? GraphBuilder.CreateSRV(View.LumenHardwareRayTracingHitDataBuffer) : nullptr;
	Parameters.LumenHardwareRayTracingUniformBuffer = View.LumenHardwareRayTracingUniformBuffer;
	Parameters.RayTracingSceneMetadata = View.InlineRayTracingBindingDataBuffer ? GraphBuilder.CreateSRV(View.InlineRayTracingBindingDataBuffer) : nullptr;
	Parameters.NaniteRayTracing = Nanite::GRayTracingManager.GetUniformBuffer();
	Parameters.RWInstanceHitCountBuffer = View.GetRayTracingInstanceHitCountUAV(GraphBuilder);
#endif
}

/**
 * Ray trace light samples using a variety of tracing methods depending on the feature configuration.
 */
void MegaLights::RayTraceLightSamples(
	const FSceneViewFamily& ViewFamily,
	const FViewInfo& View,
	int32 ViewIndex,
	FRDGBuilder& GraphBuilder,
	const FSceneTextures& SceneTextures,
	const FVirtualShadowMapArray* VirtualShadowMapArray,
	const FBoxSphereBounds& FirstPersonWorldSpaceRepresentationBounds,
	const FIntPoint SampleBufferSize,
	FRDGTextureRef LightSamples,
	FRDGTextureRef LightSampleRays,
	const FMegaLightsParameters& MegaLightsParameters,
	const FMegaLightsVolumeParameters& MegaLightsVolumeParameters,
	const FMegaLightsVolumeParameters& MegaLightsTranslucencyVolumeParameters,
	const FVolumeLightSampleParameters* VolumeLightSampleParameters,
	const FTraceStats* VolumeTraceStats,
	EMegaLightsInput InputType,
	bool bDebug,
	ERDGPassFlags ComputePassFlags)
{
	const bool bTraceStats = bDebug && CVarMegaLightsDebugTraceStats.GetValueOnRenderThread();

	const FScene* Scene = static_cast<const FScene*>(ViewFamily.Scene);
#if RHI_RAYTRACING
	const FRayTracingScene& RayTracingScene = Scene->RayTracingScene;
#endif

	FTraceStats TraceStats;

	if (VirtualShadowMapArray)
	{
		FCompactedTraceParameters CompactedTraceParameters = MegaLights::CompactMegaLightsTraces(
			View,
			GraphBuilder,
			SampleBufferSize,
			LightSampleRays,
			MegaLightsParameters,
			InputType,
			/*bCompactForScreenSpaceTraces*/ false,
			ComputePassFlags);

		TraceStats.VSM = CompactedTraceParameters.IndirectArgs;

		FVirtualShadowMapTraceLightSamplesCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FVirtualShadowMapTraceLightSamplesCS::FParameters>();
		PassParameters->CompactedTraceParameters = CompactedTraceParameters;
		PassParameters->MegaLightsParameters = MegaLightsParameters;
		PassParameters->RWLightSamples = GraphBuilder.CreateUAV(LightSamples);
		PassParameters->RWLightSampleRays = GraphBuilder.CreateUAV(LightSampleRays);
		PassParameters->VirtualShadowMapSamplingParameters = VirtualShadowMapArray->GetSamplingParameters(GraphBuilder, ViewIndex);

		FVirtualShadowMapTraceLightSamplesCS::FPermutationDomain PermutationVector;
		PermutationVector.Set<FVirtualShadowMapTraceLightSamplesCS::FDebugMode>(bDebug);
		auto ComputeShader = View.ShaderMap->GetShader<FVirtualShadowMapTraceLightSamplesCS>(PermutationVector);

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("VirtualShadowMapTraceLightSamples"),
			ComputePassFlags,
			ComputeShader,
			PassParameters,
			CompactedTraceParameters.IndirectArgs,
			(int32)MegaLights::ECompactedTraceIndirectArgs::NumTracesDiv64);
	}

	if (MegaLights::UseScreenTraces(View.Family->EngineShowFlags, InputType))
	{
		const bool bUseCompactedTraces = MegaLights::UseCompactedScreenTraces(bTraceStats);
		FScreenSpaceRayTraceLightSamplesCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FScreenSpaceRayTraceLightSamplesCS::FParameters>();

		if (bUseCompactedTraces)
		{
			FCompactedTraceParameters CompactedTraceParameters = MegaLights::CompactMegaLightsTraces(
				View,
				GraphBuilder,
				SampleBufferSize,
				LightSampleRays,
				MegaLightsParameters,
				InputType,
				/*bCompactForScreenSpaceTraces*/ true,
				ComputePassFlags);

			PassParameters->CompactedTraceParameters = CompactedTraceParameters;
			TraceStats.Screen = CompactedTraceParameters.IndirectArgs;
		}

		const float RelativeDepthThickness = CVarMegaLightsScreenTraceRelativeDepthThreshold.GetValueOnRenderThread() * View.ViewMatrices.GetPerProjectionDepthThicknessScale();
		const float RelativeDepthThicknessNoFallback = CVarMegaLightsScreenTracesRelativeDepthThicknessWhenNoFallback.GetValueOnRenderThread() * View.ViewMatrices.GetPerProjectionDepthThicknessScale();
		const float RelativeDepthThicknessForFirstPerson = CVarMegaLightsScreenTracesFirstPersonRelativeDepthThickness.GetValueOnRenderThread() * View.ViewMatrices.GetPerProjectionDepthThicknessScale();

		PassParameters->MegaLightsParameters = MegaLightsParameters;
		PassParameters->RWLightSamples = GraphBuilder.CreateUAV(LightSamples);
		PassParameters->RWLightSampleRays = GraphBuilder.CreateUAV(LightSampleRays);
		PassParameters->RayTracingNormalBias = CVarMegaLightsScreenTracesNormalBias.GetValueOnRenderThread();
		PassParameters->HairScreenTraceBias = CVarMegaLightsScreenTracesHairBias.GetValueOnRenderThread();
		PassParameters->MaxHierarchicalScreenTraceIterations = CVarMegaLightsScreenTracesMaxIterations.GetValueOnRenderThread();
		PassParameters->MaxTraceDistance = CVarMegaLightsScreenTracesMaxDistance.GetValueOnRenderThread();
		PassParameters->HairNonShadowedLightMaxTraceDistance = GetHairNonShadowedLightMaxTraceDistance();
		PassParameters->RelativeDepthThickness = RelativeDepthThickness;
		PassParameters->RelativeDepthThicknessNoFallback = RelativeDepthThicknessNoFallback;
		PassParameters->RelativeDepthThicknessForFirstPerson = RelativeDepthThicknessForFirstPerson;
		PassParameters->MinimumTracingThreadOccupancy = CVarMegaLightsScreenTracesMinimumOccupancy.GetValueOnRenderThread();
		PassParameters->UseRayTracingRepresentationBit = IsUsingRayTracingRepresentationBit() ? 1 : 0;
		PassParameters->UseHairScreenTraces = UseHairScreenTraces(View.Family->EngineShowFlags, InputType) ? 1 : 0;
		PassParameters->UseFirstPersonScreenTraces = UseFirstPersonScreenTraces(View.Family->EngineShowFlags, InputType) ? 1 : 0;
		PassParameters->FirstPersonWorldSpaceRepresentationBounds = FVector4f::Zero();
		PassParameters->FirstPersonMinimumHitDistanceOnScreenTraceMiss = FMath::Max(CVarMegaLightsScreenTracesFirstPersonMinimumHitDistance.GetValueOnRenderThread(), UE_SMALL_NUMBER);
		// If there are relevant primitives in the scene, the bounds will not be zero sized, in which case they are valid.
		if (FirstPersonWorldSpaceRepresentationBounds.SphereRadius > 0.0)
		{
			PassParameters->FirstPersonWorldSpaceRepresentationBounds = FVector4f(FVector3f(FirstPersonWorldSpaceRepresentationBounds.Origin + View.ViewMatrices.GetPreViewTranslation()), FirstPersonWorldSpaceRepresentationBounds.SphereRadius);
		}

		const int32 ScreenTraceQuality = FMath::Clamp(CVarMegaLightsScreenTracesQuality.GetValueOnRenderThread(), 0, 1);

		FScreenSpaceRayTraceLightSamplesCS::FPermutationDomain PermutationVector;
		PermutationVector.Set<FScreenSpaceRayTraceLightSamplesCS::FScreenTraceQuality>(ScreenTraceQuality);
		PermutationVector.Set<FScreenSpaceRayTraceLightSamplesCS::FUseCompactedTraces>(bUseCompactedTraces);
		PermutationVector.Set<FScreenSpaceRayTraceLightSamplesCS::FFastClear>(MegaLights::UseFastClear(InputType));
		PermutationVector.Set<FScreenSpaceRayTraceLightSamplesCS::FDebugMode>(bDebug);
		PermutationVector = FScreenSpaceRayTraceLightSamplesCS::RemapPermutation(PermutationVector);
		auto ComputeShader = View.ShaderMap->GetShader<FScreenSpaceRayTraceLightSamplesCS>(PermutationVector);

		if (bUseCompactedTraces)
		{
			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("ScreenSpaceRayTraceLightSamples Quality:%d Compact:1", ScreenTraceQuality),
				ComputePassFlags,
				ComputeShader,
				PassParameters,
				PassParameters->CompactedTraceParameters.IndirectArgs,
				FScreenSpaceRayTraceLightSamplesCS::GetIndirectArgsOffset(PermutationVector, View.GetShaderPlatform()));
		}
		else
		{
			const FIntPoint GroupSize = FScreenSpaceRayTraceLightSamplesCS::GetGroupSize(PermutationVector, View.GetShaderPlatform());

			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("ScreenSpaceRayTraceLightSamples Quality:%d Compact:0", ScreenTraceQuality),
				ComputePassFlags,
				ComputeShader,
				PassParameters,
				FComputeShaderUtils::GetGroupCount(MegaLightsParameters.SampleViewSize, GroupSize));
		}
	}

	const bool bHairVoxelTraces = UseHairVoxelTraces(View) && InputType != EMegaLightsInput::HairStrands;

	FHairVoxelTraceParameters HairVoxelTraceParameters;
	if (bHairVoxelTraces)
	{
		HairVoxelTraceParameters.HairStrands = HairStrands::BindHairStrandsViewUniformParameters(View);
		HairVoxelTraceParameters.VirtualVoxel = HairStrands::BindHairStrandsVoxelUniformParameters(View);
	}

	if (MegaLights::UseWorldSpaceTraces(View.Family->EngineShowFlags))
	{
		FRDGBufferAccessArray BatchBarrierBuffers;

		FCompactedTraceParameters CompactedTraceParameters = MegaLights::CompactMegaLightsTraces(
			View,
			GraphBuilder,
			SampleBufferSize,
			LightSampleRays,
			MegaLightsParameters,
			InputType,
			/*bCompactForScreenSpaceTraces*/ false,
			ComputePassFlags);

		BatchBarrierBuffers.Emplace(CompactedTraceParameters.IndirectArgs, ERHIAccess::SRVCompute);
		TraceStats.World = CompactedTraceParameters.IndirectArgs;

		FVolumeCompactedTraceParameters VolumeCompactedTraceParameters;
		if (VolumeLightSampleParameters && CVarMegaLightsVolumeWorldSpaceTraces.GetValueOnRenderThread() != 0)
		{
			VolumeCompactedTraceParameters = CompactVolumeTraces(
				GraphBuilder,
				View,
				MegaLightsParameters,
				MegaLightsVolumeParameters,
				MegaLightsTranslucencyVolumeParameters,
				*VolumeLightSampleParameters,
				ComputePassFlags,
				BatchBarrierBuffers,
				TraceStats);
		}

		AddBarrierPass(GraphBuilder, ComputePassFlags, BatchBarrierBuffers);

		if (MegaLights::UseHardwareRayTracing(ViewFamily))
		{
#if RHI_RAYTRACING
			const EMaterialMode MaterialMode = MegaLights::GetMaterialMode();
			const bool bUseFarField = MegaLights::UseFarField(*View.Family);

			const bool bDistantScreenTraces = MegaLights::UseDistantScreenTraces(View, bUseFarField);

			{
				FHardwareRayTraceLightSamples::FParameters* PassParameters = GraphBuilder.AllocParameters<FHardwareRayTraceLightSamples::FParameters>();
				SetRayTracingParameters(GraphBuilder, View, PassParameters->RayTracingParameters);
				PassParameters->HairVoxelTraceParameters = HairVoxelTraceParameters;
				PassParameters->CompactedTraceParameters = CompactedTraceParameters;
				PassParameters->MegaLightsParameters = MegaLightsParameters;
				PassParameters->RWLightSamples = GraphBuilder.CreateUAV(LightSamples);
				PassParameters->RWLightSampleRays = GraphBuilder.CreateUAV(LightSampleRays);

				FHardwareRayTraceLightSamples::FPermutationDomain PermutationVector;
				PermutationVector.Set<FHardwareRayTraceLightSamples::FEvaluateMaterials>(MaterialMode == EMaterialMode::AHS);
				PermutationVector.Set<FHardwareRayTraceLightSamples::FSupportContinuation>(MaterialMode == EMaterialMode::RetraceAHS);
				PermutationVector.Set<FHardwareRayTraceLightSamples::FLightingChannels>(MegaLights::IsUsingLightingChannels(View, RayTracingScene));
				PermutationVector.Set<FHardwareRayTraceLightSamples::FEnableFarFieldTracing>(bUseFarField);
				PermutationVector.Set<FHardwareRayTraceLightSamples::FForceTwoSided>(MegaLights::ShouldForceTwoSided());
				PermutationVector.Set<FHardwareRayTraceLightSamples::FHairVoxelTraces>(bHairVoxelTraces);
				PermutationVector.Set<FHardwareRayTraceLightSamples::FDistantScreenTraces>(bDistantScreenTraces);
				PermutationVector.Set<FHardwareRayTraceLightSamples::FDebugMode>(bDebug);
				PermutationVector = FHardwareRayTraceLightSamples::RemapPermutation(PermutationVector);

				if (MegaLights::UseInlineHardwareRayTracing(ViewFamily) && !PermutationVector.Get<FHardwareRayTraceLightSamples::FEvaluateMaterials>())
				{
					FHardwareRayTraceLightSamplesCS::AddMegaLightRayTracingDispatchIndirect(
						GraphBuilder,
						RDG_EVENT_NAME("HardwareRayTraceLightSamples Inline FarField:%d HairVoxel:%d", bUseFarField ? 1 : 0, bHairVoxelTraces ? 1 : 0),
						View,
						PermutationVector,
						PassParameters,
						CompactedTraceParameters.IndirectArgs,
						(int32)MegaLights::ECompactedTraceIndirectArgs::NumTracesDiv32,
						ComputePassFlags);
				}
				else
				{
					FHardwareRayTraceLightSamplesRGS::AddMegaLightRayTracingDispatchIndirect(
						GraphBuilder,
						RDG_EVENT_NAME("HardwareRayTraceLightSamples RayGen FarField:%d HairVoxel:%d", bUseFarField ? 1 : 0, bHairVoxelTraces ? 1 : 0),
						View,
						PermutationVector,
						PassParameters,
						PassParameters->CompactedTraceParameters.IndirectArgs,
						(int32)MegaLights::ECompactedTraceIndirectArgs::NumTraces,
						/*bUseMinimalPayload*/ MaterialMode != EMaterialMode::AHS,
						ComputePassFlags);
				}
			}

			if (VolumeLightSampleParameters)
			{
				RayTraceVolumeLightSamples(
					GraphBuilder,
					View,
					ViewFamily,
					MegaLightsParameters,
					MegaLightsVolumeParameters,
					MegaLightsTranslucencyVolumeParameters,
					*VolumeLightSampleParameters,
					&VolumeCompactedTraceParameters,
					ComputePassFlags);
			}

			if (MaterialMode == EMaterialMode::RetraceAHS)
			{
				FCompactedTraceParameters RetraceCompactedTraceParameters = MegaLights::CompactMegaLightsTraces(
					View,
					GraphBuilder,
					SampleBufferSize,
					LightSampleRays,
					MegaLightsParameters,
					InputType,
					/*bCompactForScreenSpaceTraces*/ false,
					ComputePassFlags);

				TraceStats.WorldMaterialRetrace = RetraceCompactedTraceParameters.IndirectArgs;

				FHardwareRayTraceLightSamples::FParameters* PassParameters = GraphBuilder.AllocParameters<FHardwareRayTraceLightSamples::FParameters>();
				SetRayTracingParameters(GraphBuilder, View, PassParameters->RayTracingParameters);
				PassParameters->HairVoxelTraceParameters = HairVoxelTraceParameters;
				PassParameters->CompactedTraceParameters = RetraceCompactedTraceParameters;
				PassParameters->MegaLightsParameters = MegaLightsParameters;
				PassParameters->RWLightSamples = GraphBuilder.CreateUAV(LightSamples);
				PassParameters->RWLightSampleRays = GraphBuilder.CreateUAV(LightSampleRays);

				FHardwareRayTraceLightSamples::FPermutationDomain PermutationVector;
				PermutationVector.Set<FHardwareRayTraceLightSamples::FEvaluateMaterials>(true);
				PermutationVector.Set<FHardwareRayTraceLightSamples::FSupportContinuation>(false);
				PermutationVector.Set<FHardwareRayTraceLightSamples::FLightingChannels>(false);
				PermutationVector.Set<FHardwareRayTraceLightSamples::FEnableFarFieldTracing>(bUseFarField);
				PermutationVector.Set<FHardwareRayTraceLightSamples::FForceTwoSided>(MegaLights::ShouldForceTwoSided());
				PermutationVector.Set<FHardwareRayTraceLightSamples::FHairVoxelTraces>(bHairVoxelTraces);
				PermutationVector.Set<FHardwareRayTraceLightSamples::FDistantScreenTraces>(MegaLights::UseDistantScreenTraces(View, bUseFarField));
				PermutationVector.Set<FHardwareRayTraceLightSamples::FDebugMode>(bDebug);
				PermutationVector = FHardwareRayTraceLightSamples::RemapPermutation(PermutationVector);

				FHardwareRayTraceLightSamplesRGS::AddMegaLightRayTracingDispatchIndirect(
					GraphBuilder,
					RDG_EVENT_NAME("HardwareRayTraceLightSamples RayGen (material retrace)"),
					View,
					PermutationVector,
					PassParameters,
					PassParameters->CompactedTraceParameters.IndirectArgs,
					(int32)MegaLights::ECompactedTraceIndirectArgs::NumTraces,
					/*bUseMinimalPayload*/ false,
					ComputePassFlags);
			}
#endif // RHI_RAYTRACING
		}
		else
		{
			ensure(MegaLights::IsUsingGlobalSDF(ViewFamily));

			// Opaque/GBuffer pass
			{
				FSoftwareRayTraceLightSamplesCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FSoftwareRayTraceLightSamplesCS::FParameters>();
				PassParameters->HairVoxelTraceParameters = HairVoxelTraceParameters;
				PassParameters->CompactedTraceParameters = CompactedTraceParameters;
				PassParameters->MegaLightsParameters = MegaLightsParameters;
				PassParameters->RWLightSamples = GraphBuilder.CreateUAV(LightSamples);
				PassParameters->LightSampleRays = LightSampleRays;

				FSoftwareRayTraceLightSamplesCS::FPermutationDomain PermutationVector;
				PermutationVector.Set<FSoftwareRayTraceLightSamplesCS::FHairVoxelTraces>(bHairVoxelTraces);
				PermutationVector.Set<FSoftwareRayTraceLightSamplesCS::FDebugMode>(bDebug);
				auto ComputeShader = View.ShaderMap->GetShader<FSoftwareRayTraceLightSamplesCS>(PermutationVector);

				FComputeShaderUtils::AddPass(
					GraphBuilder,
					RDG_EVENT_NAME("SoftwareRayTraceLightSamples"),
					ComputePassFlags,
					ComputeShader,
					PassParameters,
					CompactedTraceParameters.IndirectArgs,
					0);
			}

			if (VolumeLightSampleParameters)
			{
				RayTraceVolumeLightSamples(
					GraphBuilder,
					View,
					ViewFamily,
					MegaLightsParameters,
					MegaLightsVolumeParameters,
					MegaLightsTranslucencyVolumeParameters,
					*VolumeLightSampleParameters,
					&VolumeCompactedTraceParameters,
					ComputePassFlags);
			}
		}
	}

	if (bTraceStats)
	{
		if (!VolumeLightSampleParameters && VolumeTraceStats)
		{
			TraceStats.Volume = VolumeTraceStats->Volume;

			for (int32 CasecadeIndex = 0; CasecadeIndex < TVC_MAX; ++CasecadeIndex)
			{
				TraceStats.TranslucencyVolume[CasecadeIndex] = VolumeTraceStats->TranslucencyVolume[CasecadeIndex];
			}
		}

		FRDGBufferRef NullIndirectArgs = GraphBuilder.CreateBuffer(
			FRDGBufferDesc::CreateIndirectDesc<FRHIDispatchIndirectParameters>((int32)ECompactedTraceIndirectArgs::MAX),
			MEGALIGHTS_RESOURCE_NAME("NullIndirectArgs"));

		AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(NullIndirectArgs, PF_R32_UINT), 0, ComputePassFlags);

		FPrintTraceStatsCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FPrintTraceStatsCS::FParameters>();
		PassParameters->VSMIndirectArgs = GraphBuilder.CreateSRV(TraceStats.VSM ? TraceStats.VSM : NullIndirectArgs, PF_R32_UINT);
		PassParameters->ScreenIndirectArgs = GraphBuilder.CreateSRV(TraceStats.Screen ? TraceStats.Screen : NullIndirectArgs, PF_R32_UINT);
		PassParameters->WorldIndirectArgs = GraphBuilder.CreateSRV(TraceStats.World ? TraceStats.World : NullIndirectArgs, PF_R32_UINT);
		PassParameters->WorldMaterialRetraceIndirectArgs = GraphBuilder.CreateSRV(TraceStats.WorldMaterialRetrace ? TraceStats.WorldMaterialRetrace : NullIndirectArgs, PF_R32_UINT);
		PassParameters->VolumeIndirectArgs = GraphBuilder.CreateSRV(TraceStats.Volume ? TraceStats.Volume : NullIndirectArgs, PF_R32_UINT);
		PassParameters->TranslucencyVolume0IndirectArgs = GraphBuilder.CreateSRV(TraceStats.TranslucencyVolume[0] ? TraceStats.TranslucencyVolume[0] : NullIndirectArgs, PF_R32_UINT);
		PassParameters->TranslucencyVolume1IndirectArgs = GraphBuilder.CreateSRV(TraceStats.TranslucencyVolume[1] ? TraceStats.TranslucencyVolume[1] : NullIndirectArgs, PF_R32_UINT);

		ShaderPrint::SetParameters(GraphBuilder, View.ShaderPrintData, PassParameters->ShaderPrintUniformBuffer);

		auto ComputeShader = View.ShaderMap->GetShader<FPrintTraceStatsCS>();

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("PrintTraceStats"),
			ComputePassFlags,
			ComputeShader,
			PassParameters,
			FIntVector(1, 1, 1));
	}
}
