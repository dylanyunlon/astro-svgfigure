// Copyright Epic Games, Inc. All Rights Reserved.

#include "Lumen.h"
#include "HAL/IConsoleManager.h"
#include "RHI.h"
#include "SceneRendering.h"
#include "ScenePrivate.h"
#include "SceneProxies/SkyLightSceneProxy.h"
#include "DataDrivenShaderPlatformInfo.h"
#include "LumenSceneData.h"
#include "RayTracedTranslucency.h"
#include "RenderUtils.h"
#include "Substrate/Substrate.h"
#include "Quantization.h"
#include "PostProcess/PostProcessEyeAdaptation.h"

static TAutoConsoleVariable<int32> CVarLumenSupported(
	TEXT("r.Lumen.Supported"),
	1,
	TEXT("Whether Lumen is supported at all for the project, regardless of platform. This can be used to avoid compiling shaders and other load time overhead."),
	ECVF_ReadOnly | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarLumenAsyncCompute(
	TEXT("r.Lumen.AsyncCompute"),
	1,
	TEXT("Whether Lumen should use async compute if supported."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarLumenWaveOps(
	TEXT("r.Lumen.WaveOps"),
	1,
	TEXT("Whether Lumen should use wave ops if supported."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarLumenThreadGroupSize32(
	TEXT("r.Lumen.ThreadGroupSize32"),
	1,
	TEXT("Whether to prefer dispatches in groups of 32 threads on HW which supports it (instead of standard 64)."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarLumenLightingDataFormat(
	TEXT("r.Lumen.LightingDataFormat"),
	0,
	TEXT("Data format for surfaces storing lighting information (e.g. radiance, irradiance).\n")
	TEXT("0 - Float_R11G11B10 (fast default)\n")
	TEXT("1 - Float16_RGBA (slow, but higher precision, mostly for testing)\n")
	TEXT("2 - Float32_RGBA (reference for testing)"),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarLumenFinalGatherMethod(
	TEXT("r.Lumen.FinalGatherMethod"),
	1,
	TEXT("Lumen Final Gather Method\n")
	TEXT("0 - Irradiance Field Gather - places World Space Radiance Cache probes around pixels, pre-calculates their irradiance, and interpolates to pixels with probe occlusion. Faster but lower quality GI. Targeted at mid range PC and Switch 2.\n")
	TEXT("1 - Screen Probe Gather - traces from Screen Probes placed on pixels, giving adaptive downsampling, with World Space Radiance Cache for distant lighting. Requires its own temporal accumulation. Higher quality GI that can scale from consoles to enterprise."),
	ECVF_Scalability | ECVF_RenderThreadSafe);

bool DoesRuntimePlatformSupportLumen()
{
	return UE::PixelFormat::HasCapabilities(PF_R16_UINT, EPixelFormatCapabilities::TypedUAVLoad);
}

bool Lumen::UseAsyncCompute(const FViewFamilyInfo& ViewFamily)
{
	bool bUseAsync = GSupportsEfficientAsyncCompute
		&& CVarLumenAsyncCompute.GetValueOnRenderThread() != 0;
		
	if (Lumen::UseHardwareRayTracing(ViewFamily))
	{
		// Async for Lumen HWRT path can only be used by inline ray tracing or if RHI support async ray trace dispatch calls
		bUseAsync &= CVarLumenAsyncCompute.GetValueOnRenderThread() != 0 && (GRHIGlobals.RayTracing.SupportsAsyncRayTraceDispatch || Lumen::UseHardwareInlineRayTracing(ViewFamily));
	}

	return bUseAsync;
}

bool Lumen::UseWaveOps(EShaderPlatform ShaderPlatform)
{
	return CVarLumenWaveOps.GetValueOnRenderThread() != 0
		&& GRHISupportsWaveOperations
		&& RHISupportsWaveOperations(ShaderPlatform);
}

bool Lumen::UseThreadGroupSize32()
{
	return GRHISupportsWaveOperations && GRHIMinimumWaveSize <= 32 && CVarLumenThreadGroupSize32.GetValueOnAnyThread() != 0;
}

EPixelFormat Lumen::GetLightingDataFormat()
{
	if (CVarLumenLightingDataFormat.GetValueOnRenderThread() == 2)
	{
		return PF_A32B32G32R32F;
	}
	else if (CVarLumenLightingDataFormat.GetValueOnRenderThread() == 1)
	{
		return PF_FloatRGBA;
	}
	else
	{
		return PF_FloatR11G11B10;
	}
}

FVector3f Lumen::GetLightingQuantizationError()
{
	return ComputePixelFormatQuantizationError(GetLightingDataFormat());
}

float Lumen::GetCachedLightingPreExposure()
{
	return EyeAdaptation::GetCachedLightingPreExposure();
}

namespace Lumen
{
	bool AnyLumenHardwareRayTracingPassEnabled(const FScene* Scene, const FViewInfo& View)
	{
#if RHI_RAYTRACING
		// Set to true if any view has Lumen GI because some ray tracing passes in Lumen scene
		// lighting dispatch for all views if at least one of the views has Lumen GI
		bool bLumenGI = false;
		for (int32 ViewIndex = 0; ViewIndex < View.Family->Views.Num(); ++ViewIndex)
		{
			const FSceneView* SceneView = View.Family->Views[ViewIndex];

			if (SceneView && SceneView->bIsViewInfo)
			{
				const FViewInfo& OtherView = *(const FViewInfo*)SceneView;

				if (ShouldRenderLumenDiffuseGI(Scene, OtherView))
				{
					bLumenGI = true;
					break;
				}
			}
		}

		const bool bLumenReflections = ShouldRenderLumenReflections(View);

		if (bLumenGI
			&& (UseHardwareRayTracedScreenProbeGather(*View.Family) 
				|| UseHardwareRayTracedRadianceCache(*View.Family) 
				|| UseHardwareRayTracedDirectLighting(*View.Family)
				|| UseHardwareRayTracedTranslucencyVolume(*View.Family)))
		{
			return true;
		}

		if (bLumenReflections
			&& UseHardwareRayTracedReflections(*View.Family))
		{
			return true;
		}

		if ((bLumenGI || bLumenReflections) && Lumen::ShouldVisualizeHardwareRayTracing(*View.Family))
		{
			return true;
		}

		if ((bLumenGI || bLumenReflections) && Lumen::ShouldRenderRadiosityHardwareRayTracing(*View.Family))
		{
			return true;
		}

		if (RayTracedTranslucency::IsEnabled(View))
		{
			return true;
		}
#endif
		return false;
	}

	bool AnyLumenHardwareInlineRayTracingPassEnabled(const FScene* Scene, const FViewInfo& View)
	{
		if (!AnyLumenHardwareRayTracingPassEnabled(Scene, View))
		{
			return false;
		}

		return Lumen::UseHardwareInlineRayTracing(*View.Family);
	}

	bool SupportsMultipleClosureEvaluation(EShaderPlatform ShaderPlatform)
	{
		return Substrate::IsSubstrateEnabled() && !Substrate::IsSubstrateBlendableGBufferEnabled(ShaderPlatform);
	}

	bool SupportsMultipleClosureEvaluation(const FViewInfo& View)
	{
		return SupportsMultipleClosureEvaluation(View.GetShaderPlatform()) && View.SubstrateViewData.SceneData->ViewsMaxClosurePerPixel > 1;
	}

	ELumenFinalGatherMethod GetFinalGatherMethod(const FSceneViewFamily& ViewFamily, EShaderPlatform ShaderPlatform)
	{
		if (UseIrradianceFieldGather())
		{
			return ELumenFinalGatherMethod::IrradianceFieldGather;
		}
		else if (Lumen::UseReSTIRGather(ViewFamily, ShaderPlatform))
		{
			return ELumenFinalGatherMethod::ReSTIRGather;
		}
		else
		{
			return ELumenFinalGatherMethod::ScreenProbeGather;
		}
	}

	bool UseIrradianceFieldGather()
	{
		return CVarLumenFinalGatherMethod.GetValueOnRenderThread() == 0;
	}
}

bool Lumen::IsUsingDistanceFieldRepresentationBit(const FViewInfo& View)
{
	// TODO: When HWRT is enabled, we mark pixels based on whether instance is represented in ray tracing scene.
	// So this could be extended to also work with Lumen HWRT.
	return !Lumen::UseHardwareRayTracedScreenProbeGather(*View.Family) && !IsRayTracingEnabled();
}

bool Lumen::ShouldHandleSkyLight(const FScene* Scene, const FSceneViewFamily& ViewFamily)
{
	return Scene->SkyLight
		&& (Scene->SkyLight->ProcessedTexture || Scene->SkyLight->bRealTimeCaptureEnabled)
		&& ViewFamily.EngineShowFlags.SkyLighting
		&& DoesPlatformSupportLumenGI(Scene->GetShaderPlatform(), /*bSkipProjectCheck*/ false)
		&& !ViewFamily.EngineShowFlags.VisualizeLightCulling;
}

bool ShouldRenderLumenForViewFamily(const FScene* Scene, const FSceneViewFamily& ViewFamily, bool bSkipProjectCheck)
{
	return Scene
		&& Scene->DefaultLumenSceneData != nullptr
		&& (ViewFamily.Views.Num() <= LUMEN_MAX_VIEWS || ViewFamily.Views[0]->bIsSceneCaptureCube)
		&& DoesPlatformSupportLumenGI(Scene->GetShaderPlatform(), bSkipProjectCheck);
}

bool Lumen::IsSoftwareRayTracingSupported(EShaderPlatform ShaderPlatform)
{
	return DoesProjectSupportDistanceFields() && ShouldCompileDistanceFieldShaders(ShaderPlatform);
}

bool Lumen::IsLumenFeatureAllowedForView(const FScene* Scene, const FSceneView& View, bool bSkipTracingDataCheck, bool bSkipProjectCheck)
{
	return View.Family
		&& DoesRuntimePlatformSupportLumen()
		&& ShouldRenderLumenForViewFamily(Scene, *View.Family, bSkipProjectCheck)
		// Don't update scene lighting for secondary views
		&& !View.bIsPlanarReflection
		&& !View.bIsReflectionCapture
		&& View.State
		&& (bSkipTracingDataCheck || Lumen::UseHardwareRayTracing(*View.Family) || IsSoftwareRayTracingSupported(View.GetShaderPlatform()));
}

bool Lumen::UseGlobalSDFObjectGrid(const FSceneViewFamily& ViewFamily)
{
	if (!Lumen::IsSoftwareRayTracingSupported(ViewFamily.GetShaderPlatform()))
	{
		return false;
	}

	// All features use Hardware RayTracing, no need to update voxel lighting
	if (Lumen::UseHardwareRayTracedSceneLighting(ViewFamily)
		&& Lumen::UseHardwareRayTracedScreenProbeGather(ViewFamily)
		&& Lumen::UseHardwareRayTracedReflections(ViewFamily)
		&& Lumen::UseHardwareRayTracedRadianceCache(ViewFamily)
		&& Lumen::UseHardwareRayTracedTranslucencyVolume(ViewFamily)
		&& Lumen::UseHardwareRayTracedVisualize(ViewFamily))
	{
		return false;
	}

	return true;
}

uint32 Lumen::GetMeshCardDistanceBin(float Distance)
{
	uint32 OffsetDistance = FMath::Max(1, (int32)(Distance - 1000));
	uint32 Bin = FMath::Min(FMath::FloorLog2(OffsetDistance), Lumen::NumDistanceBuckets - 1);
	return Bin;
}

bool Lumen::WriteWarnings(const FScene* Scene, const FEngineShowFlags& ShowFlags, const TArray<FViewInfo>& Views, FScreenMessageWriter* Writer)
{
	bool bEnabledButHasNoDataForTracing = false;
	bool bEnabledButDisabledForTheProject = false;
	bool bVisualizeButDisabled = false;

	for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ViewIndex++)
	{
		const FViewInfo& View = Views[ViewIndex];

		bEnabledButHasNoDataForTracing = bEnabledButHasNoDataForTracing
			|| (!ShouldRenderLumenDiffuseGI(Scene, View) && ShouldRenderLumenDiffuseGI(Scene, View, /*bSkipTracingDataCheck*/ true))
			|| (!ShouldRenderLumenReflections(View) && ShouldRenderLumenReflections(View, /*bSkipTracingDataCheck*/ true));

		bEnabledButDisabledForTheProject = bEnabledButDisabledForTheProject
			|| (!ShouldRenderLumenDiffuseGI(Scene, View) && ShouldRenderLumenDiffuseGI(Scene, View, /*bSkipTracingDataCheck*/ false, /*bSkipProjectCheck*/ true))
			|| (!ShouldRenderLumenReflections(View) && ShouldRenderLumenReflections(View, /*bSkipTracingDataCheck*/ false, /*bSkipProjectCheck*/ true));

		bVisualizeButDisabled = ShouldVisualizeScene(ShowFlags) && !(ShouldRenderLumenDiffuseGI(Scene, View) || ShouldRenderLumenReflections(View));
	}

	bool bSurfaceCacheAtlasOversubscribed = false;
	int32 PhysicalAtlasTexels = 0;
	int32 DesiredAtlasTexels = 0;

	if (Views.Num() > 0)
	{
		const FLumenSceneData* LumenSceneData = Scene->GetLumenSceneData(Views[0]);

		if (LumenSceneData)
		{
			const FIntPoint PhysicalAtlasSize = LumenSceneData->GetPhysicalAtlasSize();

			PhysicalAtlasTexels = PhysicalAtlasSize.X * PhysicalAtlasSize.Y;
			DesiredAtlasTexels = LumenSceneData->NumDesiredSurfaceCacheTexels;
			bSurfaceCacheAtlasOversubscribed = DesiredAtlasTexels > PhysicalAtlasTexels;
		}
	}

	if (Writer)
	{
		if (bEnabledButHasNoDataForTracing)
		{
			static const FText Message = NSLOCTEXT("Renderer", "LumenCantDisplay", "Lumen is enabled, but has no ray tracing data and won't operate correctly.\nEither configure Lumen to use software distance field ray tracing and enable 'Generate Mesh Distancefields' in project settings\nor configure Lumen to use Hardware Ray Tracing and enable 'Support Hardware Ray Tracing' in project settings.");
			Writer->DrawLine(Message);
		}

		if (bEnabledButDisabledForTheProject)
		{
			static const FText Message = NSLOCTEXT("Renderer", "LumenDisabledForProject", "Lumen is enabled but cannot render, because the project has Lumen disabled in an ini (r.Lumen.Supported = 0)");
			Writer->DrawLine(Message);
		}

		if (bVisualizeButDisabled)
		{
			static const FText Message = NSLOCTEXT("Renderer", "LumenCantVisualize", "Lumen visualization is enabled but cannot render, because Lumen is disabled.");
			Writer->DrawLine(Message);
		}

		if (bSurfaceCacheAtlasOversubscribed)
		{
			const float OverSubscribePercent = float(DesiredAtlasTexels - PhysicalAtlasTexels) / PhysicalAtlasTexels * 100.0f;
			const FString OverSubscribePercentStr = FString::Printf(TEXT("%.2f"), OverSubscribePercent);
			const FText Message = FText::Format(NSLOCTEXT("Renderer", "LumenAtlasFull", "Lumen surface cache atlas oversubscribed by {0}%. Consider increasing r.LumenScene.SurfaceCache.AtlasSize."), FText::FromString(OverSubscribePercentStr));
			Writer->DrawLine(Message);
		}
	}

	return bEnabledButHasNoDataForTracing || bEnabledButDisabledForTheProject || bVisualizeButDisabled || bSurfaceCacheAtlasOversubscribed;
}