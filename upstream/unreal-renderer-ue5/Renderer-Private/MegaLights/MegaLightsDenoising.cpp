// Copyright Epic Games, Inc. All Rights Reserved.

#include "MegaLights.h"
#include "MegaLightsInternal.h"
#include "Quantization.h"
#include "../PostProcess/TemporalAA.h"
#include "TextureFallbacksRDG.h"
#include "../PostProcess/PostProcessSubsurface.h"

static TAutoConsoleVariable<bool> CVarMegaLightsDenoiser(
	TEXT("r.MegaLights.Denoiser"),
	true,
	TEXT("Whether to use denoiser. Useful in case of using joint denoising and upsampling at the end of the frame."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsTemporal(
	TEXT("r.MegaLights.Temporal"),
	true,
	TEXT("Whether to use temporal accumulation for shadow mask."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

TAutoConsoleVariable<float> CVarMegaLightsTemporalHistoryDistanceThreshold(
	TEXT("r.MegaLights.Temporal.HistoryDistanceThreshold"),
	0.03f,
	TEXT("Relative distance threshold needed to discard last frame's lighting results. Lower values reduce ghosting from characters when near a wall but increase flickering artifacts."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsTemporalMinFramesAccumulatedForHistoryMiss(
	TEXT("r.MegaLights.Temporal.MinFramesAccumulatedForHistoryMiss"),
	4,
	TEXT("Minimal amount of history length when reducing history length due to a history miss. Higher values than 1 soften and slowdown transitions."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsTemporalMaxFramesAccumulated(
	TEXT("r.MegaLights.Temporal.MaxFramesAccumulated"),
	12,
	TEXT("Max history length when accumulating frames. Lower values have less ghosting, but more noise."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsTemporalNeighborhoodClampScale(
	TEXT("r.MegaLights.Temporal.NeighborhoodClampScale"),
	1.0f,
	TEXT("Scales how permissive is neighborhood clamp. Higher values increase ghosting, but reduce noise and instability."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsSpatial(
	TEXT("r.MegaLights.Spatial"),
	true,
	TEXT("Whether denoiser should run spatial filter."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsSpatialDepthWeightScale(
	TEXT("r.MegaLights.Spatial.DepthWeightScale"),
	10000.0f,
	TEXT("Scales the depth weight of the spatial filter. Smaller values allow for more sample reuse, but also introduce more bluriness between unrelated surfaces."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsSpatialKernelRadius(
	TEXT("r.MegaLights.Spatial.KernelRadius"),
	8.0f,
	TEXT("Spatial filter kernel radius in pixels"),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsSpatialNumSamples(
	TEXT("r.MegaLights.Spatial.NumSamples"),
	4,
	TEXT("Number of spatial filter samples."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarMegaLightsSpatialMaxDisocclusionFrames(
	TEXT("r.MegaLights.Spatial.MaxDisocclusionFrames"),
	2,
	TEXT("Number of of history frames to boost spatial filtering in order to minimize noise after disocclusion."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsSpatialDisocclusionDiffuseSpatialVarianceScale(
	TEXT("r.MegaLights.Spatial.DisocclusionDiffuseSpatialVarianceScale"),
	2.0f,
	TEXT("Scale of diffuse spatial variance on disocclusion. Higher values reduce noise on disocclusion, but reduce sharpness."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsSpatialDisocclusionSpecularSpatialVarianceScale(
	TEXT("r.MegaLights.Spatial.DisocclusionSpecularSpatialVarianceScale"),
	1.0f,
	TEXT("Scale of specular spatial variance on disocclusion. Higher values reduce noise on disocclusion, but reduce sharpness."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsSpatialUseHistorySpatialVariance(
	TEXT("r.MegaLights.Spatial.UseHistorySpatialVariance"),
	true,
	TEXT("Whether history spatial variance should be used to drive spatial filtering."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarMegaLightsSpatialHistorySpatialStdDevThreshold(
	TEXT("r.MegaLights.Spatial.HistorySpatialStdDevThreshold"),
	0.15f,
	TEXT("A threshold of history spatial std dev above which spatial filtering will be applied."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsFrontLayerTranslucencyTemporal(
	TEXT("r.MegaLights.FrontLayerTranslucency.Temporal"),
	true,
	TEXT("Whether to use temporal accumulation for shadow mask on front layer translucency pass."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<bool> CVarMegaLightsFrontLayerTranslucencySpatial(
	TEXT("r.MegaLights.FrontLayerTranslucency.Spatial"),
	true,
	TEXT("Whether denoiser should run spatial filter on front layer translucency pass."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

namespace MegaLights
{
	float GetSpatialKernelRadius(EMegaLightsInput InputType)
	{
		float KernelRadius = CVarMegaLightsSpatialKernelRadius.GetValueOnRenderThread();

		if (MegaLights::UseFastClear(InputType))
		{
			// Limit max spatial kernel radius as we only clear one tile border around valid area
			KernelRadius = FMath::Min(KernelRadius, TILE_SIZE);
		}

		return KernelRadius;
	}

	float GetTemporalHistoryDistanceThreshold()
	{
		return CVarMegaLightsTemporalHistoryDistanceThreshold.GetValueOnRenderThread();
	}

	bool UseDenoiser()
	{
		return CVarMegaLightsDenoiser.GetValueOnRenderThread();
	}

	bool UseTemporalFilter(EMegaLightsInput InputType)
	{
		if (!UseDenoiser())
		{
			return false;
		}

		switch (InputType)
		{
		case EMegaLightsInput::GBuffer: return CVarMegaLightsTemporal.GetValueOnRenderThread();
		case EMegaLightsInput::HairStrands: return CVarMegaLightsTemporal.GetValueOnRenderThread();
		case EMegaLightsInput::FrontLayerTranslucency: return CVarMegaLightsFrontLayerTranslucencyTemporal.GetValueOnRenderThread();
		default: checkf(false, TEXT("MegaLight::UseTemporalFilter not implemented")); return false;
		};
	}

	bool UseSpatialFilter(EMegaLightsInput InputType)
	{
		if (!SupportsSpatialFilter(InputType))
		{
			return false;
		}

		if (!UseTemporalFilter(InputType))
		{
			// Spatial filter is driven by temporal filter's output
			return false;
		}

		switch (InputType)
		{
		case EMegaLightsInput::GBuffer: return CVarMegaLightsSpatial.GetValueOnRenderThread();
		case EMegaLightsInput::HairStrands: return CVarMegaLightsSpatial.GetValueOnRenderThread();
		case EMegaLightsInput::FrontLayerTranslucency: return CVarMegaLightsFrontLayerTranslucencySpatial.GetValueOnRenderThread();
		default: checkf(false, TEXT("MegaLight::UseSpatialFilter not implemented")); return false;
		};
	}

	bool UseHistorySpatialVariance(EMegaLightsInput InputType)
	{
		if (!UseSpatialFilter(InputType))
		{
			return false;
		}

		switch (InputType)
		{
		case EMegaLightsInput::GBuffer: return CVarMegaLightsSpatialUseHistorySpatialVariance.GetValueOnRenderThread();
		case EMegaLightsInput::HairStrands: return false;
		case EMegaLightsInput::FrontLayerTranslucency: return CVarMegaLightsSpatialUseHistorySpatialVariance.GetValueOnRenderThread();
		default: checkf(false, TEXT("MegaLight::UseHistorySpatialVariance not implemented")); return false;
		};
	}

	float GetHistorySpatialStdDevThreshold()
	{
		return FMath::Clamp(CVarMegaLightsSpatialHistorySpatialStdDevThreshold.GetValueOnRenderThread(), 0.0f, 1.0f);
	}

	float GetTemporalMaxFramesAccumulated()
	{
		return FMath::Max(CVarMegaLightsTemporalMaxFramesAccumulated.GetValueOnRenderThread(), 1.0f);
	}

	float GetSpatialFilterMaxDisocclusionFrames()
	{
		return FMath::Max(FMath::Min(CVarMegaLightsSpatialMaxDisocclusionFrames.GetValueOnRenderThread(), GetTemporalMaxFramesAccumulated() - 1.0f), 0.0f);
	}
}

class FClearTemporalAccumulationCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FClearTemporalAccumulationCS)
	SHADER_USE_PARAMETER_STRUCT(FClearTemporalAccumulationCS, FGlobalShader)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		RDG_BUFFER_ACCESS(IndirectArgs, ERHIAccess::IndirectArgs)
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float3>, RWDiffuseLighting)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float3>, RWSpecularLighting)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<UNORM float2>, RWHistoryConfidence)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWNumFramesAccumulated)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, TileAllocator)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, TileData)
	END_SHADER_PARAMETER_STRUCT()

	class FOutputHistoryConfidence : SHADER_PERMUTATION_BOOL("OUTPUT_HISTORY_CONFIDENCE");
	using FPermutationDomain = TShaderPermutationDomain<FOutputHistoryConfidence>;

	static int32 GetGroupSize()
	{
		return 8;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		MegaLights::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}
};

IMPLEMENT_GLOBAL_SHADER(FClearTemporalAccumulationCS, "/Engine/Private/MegaLights/MegaLightsDenoiserTemporal.usf", "ClearTemporalAccumulationCS", SF_Compute);

class FDenoiserTemporalCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FDenoiserTemporalCS)
	SHADER_USE_PARAMETER_STRUCT(FDenoiserTemporalCS, FGlobalShader)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(StochasticLighting::FHistoryScreenParameters, HistoryScreenParameters)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float4>, ResolvedDiffuseLighting)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float4>, ResolvedSpecularLighting)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, ShadingConfidenceTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float3>, DiffuseLightingHistoryTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float3>, SpecularLightingHistoryTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float4>, LightingMomentsHistoryTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<uint>, NumFramesAccumulatedHistoryTexture)
		SHADER_PARAMETER(FVector3f, TargetFormatQuantizationError)
		SHADER_PARAMETER(float, PrevSceneColorPreExposureCorrection)
		SHADER_PARAMETER(float, TemporalMaxFramesAccumulated)
		SHADER_PARAMETER(float, TemporalNeighborhoodClampScale)
		SHADER_PARAMETER(float, MinFramesAccumulatedForHistoryMiss)
		SHADER_PARAMETER(uint32, DebugModeEnabled)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float3>, RWDiffuseLighting)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float3>, RWSpecularLighting)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, RWLightingMoments)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<UNORM float2>, RWHistoryConfidence)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWNumFramesAccumulated)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<UNORM float4>, RWVisualizeTemporalFilterTexture)
	END_SHADER_PARAMETER_STRUCT()

	class FValidHistory : SHADER_PERMUTATION_BOOL("VALID_HISTORY");
	class FFastClear : SHADER_PERMUTATION_BOOL("FAST_CLEAR");
	class FOutputHistoryConfidence : SHADER_PERMUTATION_BOOL("OUTPUT_HISTORY_CONFIDENCE");
	class FDebugMode : SHADER_PERMUTATION_BOOL("DEBUG_MODE");
	using FPermutationDomain = TShaderPermutationDomain<FValidHistory, FFastClear, FOutputHistoryConfidence, FDebugMode>;

	static FIntPoint GetGroupSize(EShaderPlatform ShaderPlatform)
	{
		switch (FDataDrivenShaderPlatformInfo::GetThreadGroupSizeMode(ShaderPlatform))
		{
		case 1:
			return FIntPoint(8, 16);
		case 0:
		default:
			return FIntPoint(8, 8);
		}
	}

	static FPermutationDomain RemapPermutation(FPermutationDomain PermutationVector)
	{
		return PermutationVector;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (RemapPermutation(PermutationVector) != PermutationVector)
		{
			return false;
		}

		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

		const FIntPoint GroupSize = GetGroupSize(Parameters.Platform);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_X"), GroupSize.X);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_Y"), GroupSize.Y);

		if (FDataDrivenShaderPlatformInfo::GetSupportsRealTypes(Parameters.Platform) == ERHIFeatureSupport::RuntimeGuaranteed)
		{
			OutEnvironment.CompilerFlags.Add(CFLAG_AllowRealTypes);
		}
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

IMPLEMENT_GLOBAL_SHADER(FDenoiserTemporalCS, "/Engine/Private/MegaLights/MegaLightsDenoiserTemporal.usf", "DenoiserTemporalCS", SF_Compute);

class FDenoiserSpatialCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FDenoiserSpatialCS)
	SHADER_USE_PARAMETER_STRUCT(FDenoiserSpatialCS, FGlobalShader)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FMegaLightsParameters, MegaLightsParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FSubsurfaceDiffuseWriteParameters, SubsurfaceDiffuseParameters)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, RWSceneColor)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<UNORM float4>, RWVisualizeSpatialFilterTexture)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<UNORM float4>, RWVisualizeSpatialFilterTexture2)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float4>, DiffuseLightingTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float4>, SpecularLightingTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float4>, LightingMomentsTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float4>, NormalAndShadingInfoTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float2>, HistoryConfidenceTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, ShadingConfidenceTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<uint>, NumFramesAccumulatedTexture)
		SHADER_PARAMETER(float, TemporalMaxFramesAccumulated)
		SHADER_PARAMETER(float, SpatialFilterDepthWeightScale)
		SHADER_PARAMETER(float, SpatialFilterKernelRadius)
		SHADER_PARAMETER(uint32, SpatialFilterNumSamples)
		SHADER_PARAMETER(float, SpatialFilterMaxDisocclusionFrames)
		SHADER_PARAMETER(float, SpatialFilterDisocclusionDiffuseSpatialVarianceScale)
		SHADER_PARAMETER(float, SpatialFilterDisocclusionSpecularSpatialVarianceScale)
		SHADER_PARAMETER(float, HistorySpatialStdDevThreshold)
		SHADER_PARAMETER(uint32, bSubPixelShading)
		SHADER_PARAMETER(uint32, DebugModeEnabled)
	END_SHADER_PARAMETER_STRUCT()

	class FSpatialFilter : SHADER_PERMUTATION_BOOL("SPATIAL_FILTER");
	class FInputType : SHADER_PERMUTATION_INT("INPUT_TYPE", int32(EMegaLightsInput::Count));
	class FFastClear : SHADER_PERMUTATION_BOOL("FAST_CLEAR");
	class FUseHistorySpatialVariance : SHADER_PERMUTATION_BOOL("USE_HISTORY_SPATIAL_VARIANCE");
	class FDebugMode : SHADER_PERMUTATION_BOOL("DEBUG_MODE");
	using FPermutationDomain = TShaderPermutationDomain<FSpatialFilter, FInputType, FFastClear, FUseHistorySpatialVariance, FDebugMode>;

	static FIntPoint GetGroupSize(EShaderPlatform ShaderPlatform)
	{
		switch (FDataDrivenShaderPlatformInfo::GetThreadGroupSizeMode(ShaderPlatform))
		{
		case 1:
			return FIntPoint(8, 16);
		case 0:
		default:
			return FIntPoint(8, 8);
		}
	}

	static FPermutationDomain RemapPermutation(FPermutationDomain PermutationVector)
	{
		const EMegaLightsInput InputType = EMegaLightsInput(PermutationVector.Get<FInputType>());

		if (!MegaLights::SupportsSpatialFilter(InputType))
		{
			PermutationVector.Set<FSpatialFilter>(false);
		}

		if (!PermutationVector.Get<FSpatialFilter>())
		{
			PermutationVector.Set<FUseHistorySpatialVariance>(false);
		}

		return PermutationVector;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);
		
		if (RemapPermutation(PermutationVector) != PermutationVector)
		{
			return false;
		}
		
		return MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		const FIntPoint GroupSize = GetGroupSize(Parameters.Platform);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_X"), GroupSize.X);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_Y"), GroupSize.Y);
		OutEnvironment.SetDefine(TEXT("SSS_SUPPORT_SEPARATED_SUBSURFACE_DIFFUSE"), PlatformSupportsSeparatedSubsurfaceDiffuse(Parameters.Platform) ? 1 : 0);
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

IMPLEMENT_GLOBAL_SHADER(FDenoiserSpatialCS, "/Engine/Private/MegaLights/MegaLightsDenoiserSpatial.usf", "DenoiserSpatialCS", SF_Compute);

void FMegaLightsViewContext::DenoiseLighting(FRDGTextureRef OutputColorTarget, ERDGPassFlags ComputePassFlags)
{
	// Demodulated lighting components with second luminance moments stored in alpha channel for temporal variance tracking
	// This will be passed to the next frame
	const EPixelFormat LightingDataFormat = MegaLights::GetLightingDataFormat();

	FRDGTextureRef DiffuseLighting = nullptr;
	FRDGTextureRef SpecularLighting = nullptr;
	FRDGTextureRef LightingMoments = nullptr;
	FRDGTextureRef NumFramesAccumulated = nullptr;
	FRDGTextureRef VisualizeTemporalFilterTexture = nullptr;
	FRDGTextureRef VisualizeSpatialFilterTexture = nullptr;
	FRDGTextureRef VisualizeSpatialFilterTexture2 = nullptr;
	FRDGTextureRef HistoryConfidence = nullptr;
	FRDGTextureRef SceneColorBeforeDenoising = nullptr;

	const bool bUseHistorySpatialVariance = MegaLights::UseHistorySpatialVariance(InputType);
	const bool bVisualize = MegaLights::ShouldAddVisualizePostProcessingPass(View);

	if (bTemporal)
	{
		DiffuseLighting = GraphBuilder.CreateTexture(
			FRDGTextureDesc::Create2D(View.GetSceneTexturesConfig().Extent, LightingDataFormat, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
			MEGALIGHTS_RESOURCE_NAME("DiffuseLighting"));

		SpecularLighting = GraphBuilder.CreateTexture(
			FRDGTextureDesc::Create2D(View.GetSceneTexturesConfig().Extent, LightingDataFormat, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
			MEGALIGHTS_RESOURCE_NAME("SpecularLighting"));

		LightingMoments = GraphBuilder.CreateTexture(
			FRDGTextureDesc::Create2D(View.GetSceneTexturesConfig().Extent, PF_FloatRGBA, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
			MEGALIGHTS_RESOURCE_NAME("LightingMoments"));

		NumFramesAccumulated = GraphBuilder.CreateTexture(
			FRDGTextureDesc::Create2D(View.GetSceneTexturesConfig().Extent, PF_R8_UINT, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
			MEGALIGHTS_RESOURCE_NAME("NumFramesAccumulated"));

		if (bUseHistorySpatialVariance)
		{
			HistoryConfidence = GraphBuilder.CreateTexture(
				FRDGTextureDesc::Create2D(View.GetSceneTexturesConfig().Extent, PF_R8G8, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
				MEGALIGHTS_RESOURCE_NAME("HistoryConfidence"));
		}

		if (bDebug || bVisualize)
		{
			VisualizeTemporalFilterTexture = GraphBuilder.CreateTexture(
				FRDGTextureDesc::Create2D(SceneTextures.Config.Extent, PF_R8G8B8A8, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
				MEGALIGHTS_RESOURCE_NAME("Visualize.TemporalFilter"));
		}
	}

	// Temporal filter
	if (bTemporal)
	{
		FRDGTextureUAVRef DiffuseLightingUAV = GraphBuilder.CreateUAV(DiffuseLighting, ERDGUnorderedAccessViewFlags::SkipBarrier);
		FRDGTextureUAVRef SpecularLightingUAV = GraphBuilder.CreateUAV(SpecularLighting, ERDGUnorderedAccessViewFlags::SkipBarrier);
		FRDGTextureUAVRef HistoryConfidenceUAV = HistoryConfidence ? GraphBuilder.CreateUAV(HistoryConfidence, ERDGUnorderedAccessViewFlags::SkipBarrier) : nullptr;
		FRDGTextureUAVRef NumFramesAccumulatedUAV = GraphBuilder.CreateUAV(NumFramesAccumulated, ERDGUnorderedAccessViewFlags::SkipBarrier);

		// Clear tiles which won't be processed by FDenoiserTemporalCS
		if (MegaLights::UseFastClear(InputType))
		{
			FClearTemporalAccumulationCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FClearTemporalAccumulationCS::FParameters>();
			PassParameters->IndirectArgs = TileIndirectArgs;
			PassParameters->RWDiffuseLighting = DiffuseLightingUAV;
			PassParameters->RWSpecularLighting = SpecularLightingUAV;
			PassParameters->RWHistoryConfidence = HistoryConfidenceUAV;
			PassParameters->RWNumFramesAccumulated = NumFramesAccumulatedUAV;
			PassParameters->MegaLightsParameters = MegaLightsParameters;
			PassParameters->TileAllocator = GraphBuilder.CreateSRV(TileAllocator);
			PassParameters->TileData = GraphBuilder.CreateSRV(TileData);

			FClearTemporalAccumulationCS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FClearTemporalAccumulationCS::FOutputHistoryConfidence>(bUseHistorySpatialVariance);
			auto ComputeShader = View.ShaderMap->GetShader<FClearTemporalAccumulationCS>(PermutationVector);

			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("ClearTemporalAccumulation"),
				ComputePassFlags,
				ComputeShader,
				PassParameters,
				TileIndirectArgs,
				(int32)MegaLights::ETileType::Empty * sizeof(FRHIDispatchIndirectParameters));
		}

		{
			const bool bValidHistory = DiffuseLightingHistory && SceneDepthHistory && SceneNormalAndShadingHistory && NumFramesAccumulatedHistory;

			FDenoiserTemporalCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FDenoiserTemporalCS::FParameters>();
			PassParameters->MegaLightsParameters = MegaLightsParameters;
			PassParameters->HistoryScreenParameters = HistoryScreenParameters;
			PassParameters->ResolvedDiffuseLighting = ResolvedDiffuseLighting;
			PassParameters->ResolvedSpecularLighting = ResolvedSpecularLighting;
			PassParameters->ShadingConfidenceTexture = ShadingConfidence;
			PassParameters->DiffuseLightingHistoryTexture = DiffuseLightingHistory;
			PassParameters->SpecularLightingHistoryTexture = SpecularLightingHistory;
			PassParameters->LightingMomentsHistoryTexture = LightingMomentsHistory;
			PassParameters->NumFramesAccumulatedHistoryTexture = NumFramesAccumulatedHistory;
			PassParameters->TargetFormatQuantizationError = ComputePixelFormatQuantizationError(LightingDataFormat);
			PassParameters->PrevSceneColorPreExposureCorrection = View.PreExposure / View.PrevViewInfo.SceneColorPreExposure;
			PassParameters->TemporalMaxFramesAccumulated = MegaLights::GetTemporalMaxFramesAccumulated();
			PassParameters->TemporalNeighborhoodClampScale = CVarMegaLightsTemporalNeighborhoodClampScale.GetValueOnRenderThread();
			PassParameters->MinFramesAccumulatedForHistoryMiss = FMath::Clamp(CVarMegaLightsTemporalMinFramesAccumulatedForHistoryMiss.GetValueOnRenderThread(), 1.0f, MegaLights::GetTemporalMaxFramesAccumulated());
			PassParameters->DebugModeEnabled = bDebug ? 1u : 0u;
			PassParameters->RWDiffuseLighting = DiffuseLightingUAV;
			PassParameters->RWSpecularLighting = SpecularLightingUAV;
			PassParameters->RWLightingMoments = GraphBuilder.CreateUAV(LightingMoments);
			PassParameters->RWHistoryConfidence = HistoryConfidenceUAV;
			PassParameters->RWNumFramesAccumulated = NumFramesAccumulatedUAV;
			PassParameters->RWVisualizeTemporalFilterTexture = VisualizeTemporalFilterTexture ? GraphBuilder.CreateUAV(VisualizeTemporalFilterTexture) : nullptr;

			FDenoiserTemporalCS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FDenoiserTemporalCS::FValidHistory>(bValidHistory);
			PermutationVector.Set<FDenoiserTemporalCS::FFastClear>(MegaLights::UseFastClear(InputType));
			PermutationVector.Set<FDenoiserTemporalCS::FOutputHistoryConfidence>(bUseHistorySpatialVariance);
			PermutationVector.Set<FDenoiserTemporalCS::FDebugMode>(bDebug || bVisualize);
			PermutationVector = FDenoiserTemporalCS::RemapPermutation(PermutationVector);

			auto ComputeShader = View.ShaderMap->GetShader<FDenoiserTemporalCS>(PermutationVector);

			const FIntVector GroupCount = FComputeShaderUtils::GetGroupCount(View.ViewRect.Size(), FDenoiserTemporalCS::GetGroupSize(View.GetShaderPlatform()));

			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("Temporal Filter"),
				ComputePassFlags,
				ComputeShader,
				PassParameters,
				GroupCount);
		}
	}
	
	// Copy SceneColor before it's overwritten by Spatial Filtering (only when MegaFilters is enabled)
	SceneColorBeforeDenoising = AddMegaFiltersTextureCopy(GraphBuilder, View, OutputColorTarget, UE::Renderer::Private::EMegaFiltersPassType::MegaLights, MEGALIGHTS_RESOURCE_NAME("SceneColorBeforeDenoising"));

	if (bDebug || bVisualize)
	{
		VisualizeSpatialFilterTexture = GraphBuilder.CreateTexture(
			FRDGTextureDesc::Create2D(SceneTextures.Config.Extent, PF_R8G8B8A8, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
			MEGALIGHTS_RESOURCE_NAME("Visualize.SpatialFilter"));

		VisualizeSpatialFilterTexture2 = GraphBuilder.CreateTexture(
			FRDGTextureDesc::Create2D(SceneTextures.Config.Extent, PF_R8G8B8A8, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
			MEGALIGHTS_RESOURCE_NAME("Visualize.SpatialFilter2"));
	}

	// Spatial filter
	{
		FDenoiserSpatialCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FDenoiserSpatialCS::FParameters>();
		PassParameters->MegaLightsParameters = MegaLightsParameters;
		SetupSubsurfaceDiffuseWriteParameters(GraphBuilder, View, PassParameters->SubsurfaceDiffuseParameters);
		PassParameters->RWSceneColor = GraphBuilder.CreateUAV(OutputColorTarget);
		PassParameters->RWVisualizeSpatialFilterTexture = VisualizeSpatialFilterTexture ? GraphBuilder.CreateUAV(VisualizeSpatialFilterTexture) : nullptr;
		PassParameters->RWVisualizeSpatialFilterTexture2 = VisualizeSpatialFilterTexture2 ? GraphBuilder.CreateUAV(VisualizeSpatialFilterTexture2) : nullptr;
		PassParameters->DiffuseLightingTexture = bTemporal ? DiffuseLighting : ResolvedDiffuseLighting;
		PassParameters->SpecularLightingTexture = bTemporal ? SpecularLighting : ResolvedSpecularLighting;
		PassParameters->LightingMomentsTexture = LightingMoments;
		PassParameters->NormalAndShadingInfoTexture = SceneWorldNormal;
		PassParameters->HistoryConfidenceTexture = HistoryConfidence;
		PassParameters->ShadingConfidenceTexture = ShadingConfidence;
		PassParameters->NumFramesAccumulatedTexture = NumFramesAccumulated;
		PassParameters->TemporalMaxFramesAccumulated = MegaLights::GetTemporalMaxFramesAccumulated();
		PassParameters->SpatialFilterDepthWeightScale = CVarMegaLightsSpatialDepthWeightScale.GetValueOnRenderThread();
		PassParameters->SpatialFilterKernelRadius = MegaLights::GetSpatialKernelRadius(InputType);
		PassParameters->SpatialFilterNumSamples = FMath::Clamp(CVarMegaLightsSpatialNumSamples.GetValueOnRenderThread(), 0, 1024);
		PassParameters->SpatialFilterMaxDisocclusionFrames = MegaLights::GetSpatialFilterMaxDisocclusionFrames();
		PassParameters->SpatialFilterDisocclusionDiffuseSpatialVarianceScale = CVarMegaLightsSpatialDisocclusionDiffuseSpatialVarianceScale.GetValueOnRenderThread();
		PassParameters->SpatialFilterDisocclusionSpecularSpatialVarianceScale = CVarMegaLightsSpatialDisocclusionSpecularSpatialVarianceScale.GetValueOnRenderThread();
		PassParameters->HistorySpatialStdDevThreshold = MegaLights::GetHistorySpatialStdDevThreshold();
		PassParameters->bSubPixelShading = bSubPixelShading ? 1u : 0u;
		PassParameters->DebugModeEnabled = bDebug ? 1u : 0u;

		FDenoiserSpatialCS::FPermutationDomain PermutationVector;
		PermutationVector.Set<FDenoiserSpatialCS::FSpatialFilter>(bSpatial);
		PermutationVector.Set<FDenoiserSpatialCS::FInputType>(uint32(InputType));
		PermutationVector.Set<FDenoiserSpatialCS::FUseHistorySpatialVariance>(bUseHistorySpatialVariance);
		PermutationVector.Set<FDenoiserSpatialCS::FFastClear>(MegaLights::UseFastClear(InputType));
		PermutationVector.Set<FDenoiserSpatialCS::FDebugMode>(bDebug || bVisualize);
		auto ComputeShader = View.ShaderMap->GetShader<FDenoiserSpatialCS>(PermutationVector);

		const FIntVector GroupCount = FComputeShaderUtils::GetGroupCount(View.ViewRect.Size(), FDenoiserSpatialCS::GetGroupSize(View.GetShaderPlatform()));

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("Spatial Filter"),
			ComputePassFlags,
			ComputeShader,
			PassParameters,
			GroupCount);
	}

	// Fill parameters for MegaLights visualization
	{
		MegaLightsVisualizeParameters.ShadingConfidenceTexture = OrBlack2DIfNull(GraphBuilder, ShadingConfidence);
		MegaLightsVisualizeParameters.NumFramesAccumulatedTexture = OrBlack2DIfNull(GraphBuilder, NumFramesAccumulated);
		MegaLightsVisualizeParameters.VisualizeTemporalFilterTexture = OrBlack2DIfNull(GraphBuilder, VisualizeTemporalFilterTexture);
		MegaLightsVisualizeParameters.VisualizeSpatialFilterTexture = OrBlack2DIfNull(GraphBuilder, VisualizeSpatialFilterTexture);
		MegaLightsVisualizeParameters.VisualizeSpatialFilterTexture2 = OrBlack2DIfNull(GraphBuilder, VisualizeSpatialFilterTexture2);
		MegaLightsVisualizeParameters.TemporalMaxFramesAccumulated = MegaLights::GetTemporalMaxFramesAccumulated();
		MegaLightsVisualizeParameters.HistorySpatialStdDevThreshold = MegaLights::GetHistorySpatialStdDevThreshold();
	}

	{
		using UE::Renderer::Private::IMegaFilters;

		IMegaFilters::FInputs MegaFiltersInputs;
		{
			using UE::Renderer::Private::EMegaFiltersPassType;
			MegaFiltersInputs.MegaFiltersPassType = EMegaFiltersPassType::MegaLights;
			MegaFiltersInputs.OutputViewRects.Add(View.ViewRect);
			MegaFiltersInputs.Variables.Emplace(TInPlaceType<float>(), View.PreExposure);

			MegaFiltersInputs.Textures.Add(OutputColorTarget); // 0
			MegaFiltersInputs.Textures.Add(ResolvedDiffuseLighting);  // 1
			MegaFiltersInputs.Textures.Add(ResolvedSpecularLighting); // 2
			MegaFiltersInputs.Textures.Add(HistoryScreenParameters.EncodedHistoryScreenCoordTexture);  // 3
			MegaFiltersInputs.Textures.Add(SceneColorBeforeDenoising);  // 4
		}

		// Output is written to OutputColorTarget directly, so no postprocessing for output.
		AddMegaFiltersPass(GraphBuilder, View, TEXT("MegaLights"), MegaFiltersInputs);
	}

	if (View.ViewState && !View.bStatePrevViewInfoIsReadOnly)
	{
		FMegaLightsViewState::FResources& MegaLightsViewState = MegaLights::GetViewState(View, InputType);

		MegaLightsViewState.HistoryVisibleLightHashViewMinInTiles = VisibleLightHashViewMinInTiles;
		MegaLightsViewState.HistoryVisibleLightHashViewSizeInTiles = VisibleLightHashViewSizeInTiles;

		MegaLightsViewState.HistoryVolumeVisibleLightHashViewSizeInTiles = VolumeVisibleLightHashViewSizeInTiles;
		MegaLightsViewState.HistoryTranslucencyVolumeVisibleLightHashSizeInTiles = TranslucencyVolumeVisibleLightHashSizeInTiles;

		if (DiffuseLighting && SpecularLighting && LightingMoments && NumFramesAccumulated && bTemporal)
		{
			GraphBuilder.QueueTextureExtraction(DiffuseLighting, &MegaLightsViewState.DiffuseLightingHistory);
			GraphBuilder.QueueTextureExtraction(SpecularLighting, &MegaLightsViewState.SpecularLightingHistory);
			GraphBuilder.QueueTextureExtraction(LightingMoments, &MegaLightsViewState.LightingMomentsHistory);
			GraphBuilder.QueueTextureExtraction(NumFramesAccumulated, &MegaLightsViewState.NumFramesAccumulatedHistory);
		}
		else
		{
			MegaLightsViewState.DiffuseLightingHistory = nullptr;
			MegaLightsViewState.SpecularLightingHistory = nullptr;
			MegaLightsViewState.LightingMomentsHistory = nullptr;
			MegaLightsViewState.NumFramesAccumulatedHistory = nullptr;
		}

		if (bGuideByHistory)
		{
			GraphBuilder.QueueBufferExtraction(VisibleLightHash, &MegaLightsViewState.VisibleLightHashHistory);
			GraphBuilder.QueueBufferExtraction(HiddenLightHash, &MegaLightsViewState.HiddenLightHashHistory);
		}
		else
		{
			MegaLightsViewState.VisibleLightHashHistory = nullptr;
			MegaLightsViewState.HiddenLightHashHistory = nullptr;
		}

		if (bVolumeGuideByHistory
			&& VolumeVisibleLightHash != nullptr
			&& VolumeHiddenLightHash != nullptr)
		{
			GraphBuilder.QueueBufferExtraction(VolumeVisibleLightHash, &MegaLightsViewState.VolumeVisibleLightHashHistory);
			GraphBuilder.QueueBufferExtraction(VolumeHiddenLightHash, &MegaLightsViewState.VolumeHiddenLightHashHistory);
		}
		else
		{
			MegaLightsViewState.VolumeVisibleLightHashHistory = nullptr;
			MegaLightsViewState.VolumeHiddenLightHashHistory = nullptr;
		}

		if (bTranslucencyVolumeGuideByHistory 
			&& TranslucencyVolumeVisibleLightHash[0] != nullptr 
			&& TranslucencyVolumeVisibleLightHash[1] != nullptr
			&& TranslucencyVolumeHiddenLightHash[0] != nullptr
			&& TranslucencyVolumeHiddenLightHash[1] != nullptr)
		{
			GraphBuilder.QueueBufferExtraction(TranslucencyVolumeVisibleLightHash[0], &MegaLightsViewState.TranslucencyVolume0VisibleLightHashHistory);
			GraphBuilder.QueueBufferExtraction(TranslucencyVolumeVisibleLightHash[1], &MegaLightsViewState.TranslucencyVolume1VisibleLightHashHistory);
			GraphBuilder.QueueBufferExtraction(TranslucencyVolumeHiddenLightHash[0], &MegaLightsViewState.TranslucencyVolume0HiddenLightHashHistory);
			GraphBuilder.QueueBufferExtraction(TranslucencyVolumeHiddenLightHash[1], &MegaLightsViewState.TranslucencyVolume1HiddenLightHashHistory);
		}
		else
		{
			MegaLightsViewState.TranslucencyVolume0VisibleLightHashHistory = nullptr;
			MegaLightsViewState.TranslucencyVolume1VisibleLightHashHistory = nullptr;
			MegaLightsViewState.TranslucencyVolume0HiddenLightHashHistory = nullptr;
			MegaLightsViewState.TranslucencyVolume1HiddenLightHashHistory = nullptr;
		}

		// Extract Scene Depth/Normal history
		if (InputType == EMegaLightsInput::GBuffer || InputType == EMegaLightsInput::FrontLayerTranslucency)
		{
			// GBuffer and FrontLayerTranslucency uses StochasticLightingViewState so we can skip extracting history resources here
		}
		else if (InputType == EMegaLightsInput::HairStrands)
		{
			if (SceneDepth)
			{
				GraphBuilder.QueueTextureExtraction(SceneDepth, &MegaLightsViewState.SceneDepthHistory);
			}
			else
			{
				MegaLightsViewState.SceneDepthHistory = nullptr;
			}

			if (SceneWorldNormal)
			{
				GraphBuilder.QueueTextureExtraction(SceneWorldNormal, &MegaLightsViewState.SceneNormalHistory);
			}
			else
			{
				MegaLightsViewState.SceneNormalHistory = nullptr;
			}
		}
		else
		{
			unimplemented();
		}
	}
}