// Copyright Epic Games, Inc. All Rights Reserved.

#include "MegaLightsVisualize.h"
#include "MegaLights.h"
#include "MegaLightsInternal.h"
#include "MegaLightsVisualizationData.h"
#include "Lumen/LumenHardwareRayTracingCommon.h"
#include "Nanite/NaniteRayTracing.h"
#include "StochasticLighting/StochasticLightingVisualize.h"
#include "DeferredShadingRenderer.h"
#include "UnrealEngine.h"

static TAutoConsoleVariable<int32> CVarMegaLightsVisualize(
	TEXT("r.MegaLights.Visualize"),
	0,
	TEXT("MegaLights visualization mode:")
	TEXT("0  - Disable\n")
	TEXT("1  - ## Overview ##\n")
	TEXT("2  - Shadow Casters\n")
	TEXT("3  - Shadow Caster Quality\n")
	TEXT("4  - ## Denoiser Overview ##\n")
	TEXT("5  - ## Denoiser Overview 2 ##\n")
	TEXT("6  - Shading Confidence\n")
	TEXT("7  - Num Frames Accumulated\n")
	TEXT("8  - Upsample Mask\n")
	TEXT("9  - Diffuse Demodulate\n")
	TEXT("10 - Specular Demodulate\n")
	TEXT("11 - Diffuse Sdt Dev\n")
	TEXT("12 - Specular Sdt Dev\n")
	TEXT("13 - Diffuse History Spatial Sdt Dev\n")
	TEXT("14 - Specular History Spatial Sdt Dev\n")
	TEXT("15 - Diffuse History Clamp\n")
	TEXT("16 - Specular History Clamp"),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsVisualizeShadowCasterQualityMissingProxyDistanceThreshold(
	TEXT("r.MegaLights.Visualize.ShadowCasterQuality.MissingProxyDistanceThreshold"),
	50.0f,
	TEXT("The distance in cm from the nanite mesh at which we should the raytracing proxy missing."));

static TAutoConsoleVariable<float> CVarMegaLightsVisualizeShadowCasterQualityPoorQualityDistanceThreshold(
	TEXT("r.MegaLights.Visualize.ShadowCasterQuality.PoorQualityDistanceThreshold"),
	25.0f,
	TEXT("The distance in cm at which we should consider meshes not representative of the nanite mesh."));

namespace MegaLights
{
	bool IsVisualizeModeValid(int32 VisualizeMode)
	{
		return VisualizeMode >= MEGA_LIGHTS_VISUALIZE_MIN && VisualizeMode <= MEGA_LIGHTS_VISUALIZE_MAX;
	}

	int32 GetVisualizeMode(const FViewInfo& View)
	{
		int32 VisualizeMode = 0;

		if (IsVisualizeModeValid(CVarMegaLightsVisualize.GetValueOnRenderThread()))
		{
			VisualizeMode = CVarMegaLightsVisualize.GetValueOnRenderThread();
		}
		else if (View.Family->EngineShowFlags.VisualizeMegaLights)
		{
			const FMegaLightsVisualizationData& VisualizationData = GetMegaLightsVisualizationData();
			const FMegaLightsVisualizationData::EModeID ModeId = VisualizationData.GetModeID(View.CurrentMegaLightsVisualizationMode);

			if (ModeId >= FMegaLightsVisualizationData::EModeID::MinVisualizeMode && ModeId <= FMegaLightsVisualizationData::EModeID::MaxVisualizeMode)
			{
				VisualizeMode = (int32)ModeId;
			}
		}

		return VisualizeMode;
	}

	bool IsOverviewVisualizeMode(int32 VisualizeMode)
	{
		return VisualizeMode == MEGA_LIGHTS_VISUALIZE_OVERVIEW 
			|| VisualizeMode == MEGA_LIGHTS_VISUALIZE_DENOISER_OVERVIEW
			|| VisualizeMode == MEGA_LIGHTS_VISUALIZE_DENOISER_OVERVIEW2;
	}

	bool ShouldAddVisualizePostProcessingPass(const FViewInfo& View)
	{
		if (View.Family && IsEnabled(*View.Family))
		{
			const int32 VisualizeMode = GetVisualizeMode(View);
			return VisualizeMode > 0;
		}

		return false;
	}

	bool ShouldAddVisualizePostProcessingPass(const FViewInfo& View, int32 ViewIndex, const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries, bool bOverview)
	{
		if (View.Family && IsEnabled(*View.Family) && MegaLightsFrameTemporaries.IsValid())
		{
			const int32 VisualizeMode = GetVisualizeMode(View);
			if (VisualizeMode > 0)
			{
				return IsOverviewVisualizeMode(VisualizeMode) == bOverview;
			}
		}

		return false;
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FVisualizeOutputParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(StochasticLightingVisualize::FTonemappingParameters, TonemappingParameters)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FReflectionUniformParameters, ReflectionStruct)
		SHADER_PARAMETER(FIntPoint, InputViewSize)
		SHADER_PARAMETER(FIntPoint, InputViewOffset)
		SHADER_PARAMETER(FIntPoint, OutputViewSize)
		SHADER_PARAMETER(FIntPoint, OutputViewOffset)
		SHADER_PARAMETER(uint32, VisualizeMode)
	END_SHADER_PARAMETER_STRUCT()
}

class FMegaLightsVisualizeCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FMegaLightsVisualizeCS)
	SHADER_USE_PARAMETER_STRUCT(FMegaLightsVisualizeCS, FGlobalShader)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FVisualizeParameters, VisualizeParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FVisualizeOutputParameters, VisualizeOutputParameters)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float3>, RWSceneColor)
	END_SHADER_PARAMETER_STRUCT()

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return FGlobalShader::ShouldPrecachePermutation(Parameters);
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static int32 GetGroupSize()
	{
		return 8;
	}
};

IMPLEMENT_GLOBAL_SHADER(FMegaLightsVisualizeCS, "/Engine/Private/MegaLights/MegaLightsVisualize.usf", "MegaLightsVisualizeCS", SF_Compute);

#if RHI_RAYTRACING

class FMegaLightsVisualizeHardwareRayTracing : public FLumenHardwareRayTracingShaderBase
{
	DECLARE_LUMEN_RAYTRACING_SHADER(FMegaLightsVisualizeHardwareRayTracing)

	class FEvaluateMaterials : SHADER_PERMUTATION_BOOL("MEGA_LIGHTS_EVALUATE_MATERIALS");
	class FHairVoxelTraces : SHADER_PERMUTATION_BOOL("HAIR_VOXEL_TRACES");
	using FPermutationDomain = TShaderPermutationDomain<FLumenHardwareRayTracingShaderBase::FBasePermutationDomain, FEvaluateMaterials, FHairVoxelTraces>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FRayTracingParameters, RayTracingParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FHairVoxelTraceParameters, HairVoxelTraceParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FVisualizeOutputParameters, VisualizeOutputParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(MegaLights::FVisualizeParameters, VisualizeParameters)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float3>, RWSceneColor)
		SHADER_PARAMETER(float, ShadowCasterQualityMissingProxyDistanceThreshold)
		SHADER_PARAMETER(float, ShadowCasterQualityPoorQualityDistanceThreshold)
	END_SHADER_PARAMETER_STRUCT()

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

		if (ShaderDispatchType == Lumen::ERayTracingShaderDispatchType::Inline && PermutationVector.Get<FEvaluateMaterials>())
		{
			return false;
		}

		return MegaLights::ShouldCompileShaders(Parameters.Platform)
			&& FLumenHardwareRayTracingShaderBase::ShouldCompilePermutation(Parameters, ShaderDispatchType);
	}

	static EShaderPermutationPrecacheRequest ShouldPrecachePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return EShaderPermutationPrecacheRequest::NotPrecached;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, Lumen::ERayTracingShaderDispatchType ShaderDispatchType, FShaderCompilerEnvironment& OutEnvironment)
	{
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

	static uint32 GetGroupSize()
	{
		return 64;
	}
};

IMPLEMENT_MEGALIGHT_RAYGEN_AND_COMPUTE_RAYTRACING_SHADERS(FMegaLightsVisualizeHardwareRayTracing)

IMPLEMENT_GLOBAL_SHADER(FMegaLightsVisualizeHardwareRayTracingRGS, "/Engine/Private/MegaLights/MegaLightsVisualizeHardwareRayTracing.usf", "MegaLightsVisualizeHardwareRayTracingRGS", SF_RayGen);
IMPLEMENT_GLOBAL_SHADER(FMegaLightsVisualizeHardwareRayTracingCS, "/Engine/Private/MegaLights/MegaLightsVisualizeHardwareRayTracing.usf", "MegaLightsVisualizeHardwareRayTracingCS", SF_Compute);

void FDeferredShadingSceneRenderer::PrepareMegaLightsHardwareRayTracingVisualize(const FViewInfo& View, const FScene& Scene, TArray<FRHIRayTracingShader*>& OutRayGenShaders)
{
	const MegaLights::EMaterialMode MaterialMode = MegaLights::GetMaterialMode();

	if (MegaLights::UseHardwareRayTracing(*View.Family) && MaterialMode != MegaLights::EMaterialMode::Disabled)
	{
		for (int32 HairVoxelTraces = 0; HairVoxelTraces < 2; ++HairVoxelTraces)
		{
			FMegaLightsVisualizeHardwareRayTracingRGS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FMegaLightsVisualizeHardwareRayTracingRGS::FHairVoxelTraces>(HairVoxelTraces != 0);
			PermutationVector.Set<FMegaLightsVisualizeHardwareRayTracingRGS::FEvaluateMaterials>(true);
			PermutationVector = FMegaLightsVisualizeHardwareRayTracingRGS::RemapPermutation(PermutationVector);

			TShaderRef<FMegaLightsVisualizeHardwareRayTracingRGS> RayGenerationShader = View.ShaderMap->GetShader<FMegaLightsVisualizeHardwareRayTracingRGS>(PermutationVector);
			OutRayGenShaders.Add(RayGenerationShader.GetRayTracingShader());
		}
	}
}

void FDeferredShadingSceneRenderer::PrepareMegaLightsHardwareRayTracingVisualizeMaterial(const FViewInfo& View, const FScene& Scene, TArray<FRHIRayTracingShader*>& OutRayGenShaders)
{
	const MegaLights::EMaterialMode MaterialMode = MegaLights::GetMaterialMode();

	if (MegaLights::UseHardwareRayTracing(*View.Family) && !MegaLights::UseInlineHardwareRayTracing(*View.Family) && MaterialMode != MegaLights::EMaterialMode::Disabled)
	{
		for (int32 HairVoxelTraces = 0; HairVoxelTraces < 2; ++HairVoxelTraces)
		{
			FMegaLightsVisualizeHardwareRayTracingRGS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FMegaLightsVisualizeHardwareRayTracingRGS::FHairVoxelTraces>(HairVoxelTraces != 0);
			PermutationVector.Set<FMegaLightsVisualizeHardwareRayTracingRGS::FEvaluateMaterials>(true);
			PermutationVector = FMegaLightsVisualizeHardwareRayTracingRGS::RemapPermutation(PermutationVector);

			TShaderRef<FMegaLightsVisualizeHardwareRayTracingRGS> RayGenerationShader = View.ShaderMap->GetShader<FMegaLightsVisualizeHardwareRayTracingRGS>(PermutationVector);
			OutRayGenShaders.Add(RayGenerationShader.GetRayTracingShader());
		}
	}
}

#endif // RHI_RAYTRACING

void FMegaLightsViewContext::Visualize(FScreenPassTexture Output, int32 VisualizeMode, int32 VisualizeTileIndex, uint32 NumOverviewTilesPerRow)
{
	bool bEvaluateMaterials = false;
	bool bUseFarField = false;
	bool bHairVoxelTraces = false;

	// Setup common visualization output shader parameters
	MegaLights::FVisualizeOutputParameters VisualizeOutputParameters;
	{
		VisualizeOutputParameters.TonemappingParameters = StochasticLightingVisualize::GetTonemappingParameters(GraphBuilder, View, /*bAllowTonemapping*/ true);
		VisualizeOutputParameters.ReflectionStruct = CreateReflectionUniformBuffer(GraphBuilder, View);
		VisualizeOutputParameters.VisualizeMode = VisualizeMode;
		VisualizeOutputParameters.InputViewOffset = Output.ViewRect.Min;
		VisualizeOutputParameters.OutputViewOffset = Output.ViewRect.Min;
		VisualizeOutputParameters.InputViewSize = Output.ViewRect.Size();
		VisualizeOutputParameters.OutputViewSize = Output.ViewRect.Size();
		StochasticLightingVisualize::GetTileOutputView(Output.ViewRect, VisualizeTileIndex, VisualizeOutputParameters.OutputViewOffset, VisualizeOutputParameters.OutputViewSize, NumOverviewTilesPerRow);
	}

	if (MegaLights::UseHardwareRayTracing(*View.Family) 
		&& (VisualizeMode == MEGA_LIGHTS_VISUALIZE_SHADOW_CASTERS || VisualizeMode == MEGA_LIGHTS_VISUALIZE_SHADOW_CASTER_QUALITY))
	{
#if RHI_RAYTRACING
		bEvaluateMaterials = MegaLights::GetMaterialMode() != MegaLights::EMaterialMode::Disabled;
		bUseFarField = MegaLights::UseFarField(*View.Family);
		bHairVoxelTraces = MegaLights::UseHairVoxelTraces(View);

		FMegaLightsVisualizeHardwareRayTracing::FParameters* PassParameters = GraphBuilder.AllocParameters<FMegaLightsVisualizeHardwareRayTracing::FParameters>();
		PassParameters->MegaLightsParameters = MegaLightsParameters;
		PassParameters->VisualizeOutputParameters = VisualizeOutputParameters;
		PassParameters->VisualizeParameters = MegaLightsVisualizeParameters;
		MegaLights::SetRayTracingParameters(GraphBuilder, View, PassParameters->RayTracingParameters);
		PassParameters->RWSceneColor = GraphBuilder.CreateUAV(Output.Texture);
		if (bHairVoxelTraces)
		{
			MegaLights::SetHairVoxelTraceParameters(View, PassParameters->HairVoxelTraceParameters);
		}
		PassParameters->ShadowCasterQualityMissingProxyDistanceThreshold = CVarMegaLightsVisualizeShadowCasterQualityMissingProxyDistanceThreshold.GetValueOnRenderThread();
		PassParameters->ShadowCasterQualityPoorQualityDistanceThreshold = CVarMegaLightsVisualizeShadowCasterQualityPoorQualityDistanceThreshold.GetValueOnRenderThread();

		FMegaLightsVisualizeHardwareRayTracing::FPermutationDomain PermutationVector;
		PermutationVector.Set<FMegaLightsVisualizeHardwareRayTracing::FEvaluateMaterials>(bEvaluateMaterials);
		PermutationVector.Set<FMegaLightsVisualizeHardwareRayTracing::FHairVoxelTraces>(bHairVoxelTraces);
		PermutationVector = FMegaLightsVisualizeHardwareRayTracing::RemapPermutation(PermutationVector);

		const FIntPoint NumTiles = FIntPoint::DivideAndRoundUp(VisualizeOutputParameters.OutputViewSize, 8);
		const FIntPoint DispatchResolution = FIntPoint(
			NumTiles.X * NumTiles.Y,
			FMegaLightsVisualizeHardwareRayTracing::GetGroupSize());

		if (MegaLights::UseInlineHardwareRayTracing(*View.Family) && !PermutationVector.Get<FMegaLightsVisualizeHardwareRayTracing::FEvaluateMaterials>())
		{
			const FIntVector GroupCount = FComputeShaderUtils::GetGroupCount(DispatchResolution, FMegaLightsVisualizeHardwareRayTracingCS::GetThreadGroupSize(View.GetShaderPlatform()));

			FMegaLightsVisualizeHardwareRayTracingCS::AddMegaLightRayTracingDispatch(
				GraphBuilder,
				RDG_EVENT_NAME("MegaLightsVisualize Inline FarField:%d HairVoxel:%d", bUseFarField ? 1 : 0, bHairVoxelTraces ? 1 : 0),
				View,
				PermutationVector,
				PassParameters,
				GroupCount,
				ERDGPassFlags::Compute);
		}
		else
		{
			FMegaLightsVisualizeHardwareRayTracingRGS::AddMegaLightRayTracingDispatch(
				GraphBuilder,
				RDG_EVENT_NAME("MegaLightsVisualize RayGen FarField:%d HairVoxel:%d", bUseFarField ? 1 : 0, bHairVoxelTraces ? 1 : 0),
				View,
				PermutationVector,
				PassParameters,
				DispatchResolution,
				/*bUseMinimalPayload*/ !bEvaluateMaterials,
				ERDGPassFlags::Compute);
		}
#endif // RHI_RAYTRACING
	}
	else
	{
		// Render target visualization without any ray tracing
		FMegaLightsVisualizeCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FMegaLightsVisualizeCS::FParameters>();
		PassParameters->MegaLightsParameters = MegaLightsParameters;
		PassParameters->VisualizeParameters = MegaLightsVisualizeParameters;
		PassParameters->VisualizeOutputParameters = VisualizeOutputParameters;
		PassParameters->RWSceneColor = GraphBuilder.CreateUAV(Output.Texture);

		auto ComputeShader = View.ShaderMap->GetShader<FMegaLightsVisualizeCS>();

		const FIntVector GroupCount = FComputeShaderUtils::GetGroupCount(PassParameters->VisualizeOutputParameters.OutputViewSize, FMegaLightsVisualizeCS::GetGroupSize());

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("MegaLightsVisualize"),
			ComputeShader,
			PassParameters,
			GroupCount);
	}

	FString LabelText;
	{
		switch (VisualizeMode)
		{
			case MEGA_LIGHTS_VISUALIZE_SHADOW_CASTERS:
				LabelText = FString::Printf(TEXT("Shadow Casters, Materials:%d, FarField:%d, HairVoxels:%d"), bEvaluateMaterials ? 1 : 0, bUseFarField ? 1 : 0, bHairVoxelTraces ? 1 : 0);
				break;

			case MEGA_LIGHTS_VISUALIZE_SHADOW_CASTER_QUALITY:
				LabelText = TEXT("Shadow Caster Quality, Red - missing shadow caster, Blue - mismatching shadow caster");
				break;

			case MEGA_LIGHTS_VISUALIZE_SHADING_CONFIDENCE:
				LabelText = TEXT("Shading Confidence");
				break;

			case MEGA_LIGHTS_VISUALIZE_NUM_FRAMES_ACCUMULATED:
				LabelText = TEXT("Num Frames Accumulated, Red - invalid history, Yellow - first frame");
				break;

			case MEGA_LIGHTS_VISUALIZE_UPSAMPLE_MASK:
				LabelText = TEXT("Upsample Mask, Yellow - invalid pixels");
				break;

			case MEGA_LIGHTS_VISUALIZE_DIFFUSE_DEMODULATE:
				LabelText = TEXT("Diffuse Demodulate");
				break;

			case MEGA_LIGHTS_VISUALIZE_SPECULAR_DEMODULATE:
				LabelText = TEXT("Specular Demodulate");
				break;

			case MEGA_LIGHTS_VISUALIZE_DIFFUSE_STD_DEV:
				LabelText = TEXT("Diffuse Std Dev, Red - disocclusion, Blue - spatial filtering");
				break;

			case MEGA_LIGHTS_VISUALIZE_SPECULAR_STD_DEV:
				LabelText = TEXT("Specular Std Dev, Red - disocclusion, Blue - spatial filtering");
				break;

			case MEGA_LIGHTS_VISUALIZE_DIFFUSE_HISTORY_SPATIAL_STD_DEV:
				LabelText = TEXT("Diffuse History Var, Blue - spatial filtering");
				break;

			case MEGA_LIGHTS_VISUALIZE_SPECULAR_HISTORY_SPATIAL_STD_DEV:
				LabelText = TEXT("Specular History Var, Blue - spatial filtering");
				break;

			case MEGA_LIGHTS_VISUALIZE_DIFFUSE_HISTORY_CLAMP:
				LabelText = TEXT("Diffuse History Clamp");
				break;

			case MEGA_LIGHTS_VISUALIZE_SPECULAR_HISTORY_CLAMP:
				LabelText = TEXT("Specular History Clamp");
				break;
		}
	}

	if (!LabelText.IsEmpty())
	{
		AddDrawCanvasPass(GraphBuilder, RDG_EVENT_NAME("MegaLightsVisualizeLabel"), View, FScreenPassRenderTarget(Output, ERenderTargetLoadAction::ELoad),
			[&ViewRect = Output.ViewRect, VisualizeTileIndex, NumOverviewTilesPerRow, LabelText](FCanvas& Canvas)
			{
				const float DPIScale = Canvas.GetDPIScale();
				Canvas.SetBaseTransform(FMatrix(FScaleMatrix(DPIScale) * Canvas.CalcBaseTransform2D(Canvas.GetViewRect().Width(), Canvas.GetViewRect().Height())));

				const FLinearColor LabelColor(1, 1, 0);

				FIntPoint OutputViewSize;
				FIntPoint OutputViewOffset;
				StochasticLightingVisualize::GetTileOutputView(ViewRect, VisualizeTileIndex, OutputViewOffset, OutputViewSize, NumOverviewTilesPerRow);

				FIntPoint LabelLocation(OutputViewOffset.X + 2 * StochasticLightingVisualize::OverviewTileMargin, OutputViewOffset.Y + OutputViewSize.Y - 20);
				Canvas.DrawShadowedString(LabelLocation.X / DPIScale, LabelLocation.Y / DPIScale, LabelText, GetStatsFont(), LabelColor);
			});
	}
}

FScreenPassTexture MegaLights::AddVisualizePass(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	int32 ViewIndex,
	FScreenPassTexture SceneColor,
	FScreenPassTexture SceneDepth,
	FScreenPassRenderTarget OverrideOutput,
	const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries)
{
	check(SceneColor.IsValid());

	FScreenPassTexture Output = SceneColor;
	const FScene* Scene = (const FScene*)View.Family->Scene;
	const FSceneViewFamily& ViewFamily = *View.Family;

	if (MegaLights::IsEnabled(ViewFamily))
	{
		RDG_EVENT_SCOPE(GraphBuilder, "MegaLightsVisualize");

		const int32 VisualizeMode = MegaLights::GetVisualizeMode(View);
		check(VisualizeMode > 0);

		// Create a new output just to make sure the right flags are set
		FRDGTextureDesc VisualizeOutputDesc = SceneColor.Texture->Desc;
		VisualizeOutputDesc.Flags |= TexCreate_UAV | TexCreate_RenderTargetable;
		Output = FScreenPassTexture(GraphBuilder.CreateTexture(VisualizeOutputDesc, TEXT("VisualizeMegaLights")), SceneColor.ViewRect);

		// In the overview mode we don't fully overwrite, copy the old Scene Color
		if (IsOverviewVisualizeMode(VisualizeMode))
		{
			FRHICopyTextureInfo CopyInfo;

			AddCopyTexturePass(
				GraphBuilder,
				SceneColor.Texture,
				Output.Texture,
				CopyInfo);
		}

		if (VisualizeMode == MEGA_LIGHTS_VISUALIZE_OVERVIEW)
		{
			const int32 VisualizeTiles[] = { MEGA_LIGHTS_VISUALIZE_SHADOW_CASTERS , MEGA_LIGHTS_VISUALIZE_SHADOW_CASTER_QUALITY };

			for (int32 TileIndex = 0; TileIndex < UE_ARRAY_COUNT(VisualizeTiles); ++TileIndex)
			{
				MegaLightsFrameTemporaries->ViewContexts[ViewIndex].Visualize(
					Output,
					VisualizeTiles[TileIndex],
					TileIndex,
					/*NumOverviewTilesPerRow*/ 3);
			}
		}
		else if (VisualizeMode == MEGA_LIGHTS_VISUALIZE_DENOISER_OVERVIEW || VisualizeMode == MEGA_LIGHTS_VISUALIZE_DENOISER_OVERVIEW2)
		{
			int32 VisualizeTiles[] = { 
				MEGA_LIGHTS_VISUALIZE_UPSAMPLE_MASK, MEGA_LIGHTS_VISUALIZE_DIFFUSE_DEMODULATE, MEGA_LIGHTS_VISUALIZE_SPECULAR_DEMODULATE, MEGA_LIGHTS_VISUALIZE_DIFFUSE_HISTORY_CLAMP,
				MEGA_LIGHTS_VISUALIZE_SHADING_CONFIDENCE, -1, -1, MEGA_LIGHTS_VISUALIZE_SPECULAR_HISTORY_CLAMP,
				MEGA_LIGHTS_VISUALIZE_NUM_FRAMES_ACCUMULATED, -1, -1, MEGA_LIGHTS_VISUALIZE_DIFFUSE_STD_DEV,
				-1, -1, -1, MEGA_LIGHTS_VISUALIZE_SPECULAR_STD_DEV
			};

			if (VisualizeMode == MEGA_LIGHTS_VISUALIZE_DENOISER_OVERVIEW2)
			{
				VisualizeTiles[1] = MEGA_LIGHTS_VISUALIZE_DIFFUSE_HISTORY_SPATIAL_STD_DEV;
				VisualizeTiles[2] = MEGA_LIGHTS_VISUALIZE_SPECULAR_HISTORY_SPATIAL_STD_DEV;
			}

			for (int32 TileIndex = 0; TileIndex < UE_ARRAY_COUNT(VisualizeTiles); ++TileIndex)
			{
				if (VisualizeTiles[TileIndex] >= 0)
				{
					MegaLightsFrameTemporaries->ViewContexts[ViewIndex].Visualize(
						Output,
						VisualizeTiles[TileIndex],
						TileIndex,
						/*NumOverviewTilesPerRow*/ 4);
				}
			}
		}
		else
		{
			MegaLightsFrameTemporaries->ViewContexts[ViewIndex].Visualize(
				Output,
				VisualizeMode,
				/*VisualizeTileIndex*/ -1,
				/*NumOverviewTilesPerRow*/ 1);
		}
	}

	if (OverrideOutput.IsValid())
	{
		AddDrawTexturePass(GraphBuilder, View, Output, OverrideOutput);
		return OverrideOutput;
	}

	return MoveTemp(Output);
}		