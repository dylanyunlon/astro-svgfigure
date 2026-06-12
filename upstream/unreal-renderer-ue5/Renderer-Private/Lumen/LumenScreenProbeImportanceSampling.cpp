// Copyright Epic Games, Inc. All Rights Reserved.

#include "LumenScreenProbeGather.h"
#include "SceneUtils.h"
#include "PipelineStateCache.h"
#include "ShaderParameterStruct.h"
#include "PixelShaderUtils.h"
#include "ReflectionEnvironment.h"
#include "DistanceFieldAmbientOcclusion.h"
#include "SceneViewState.h"

int32 GLumenScreenProbeImportanceSampling = 1;
FAutoConsoleVariableRef GVarLumenScreenProbeGatherImportanceSampling(
	TEXT("r.Lumen.ScreenProbeGather.ImportanceSample"),
	GLumenScreenProbeImportanceSampling,
	TEXT("Whether to use Importance Sampling to generate probe trace directions."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenScreenProbeImportanceSamplingNumLevels = 1;
FAutoConsoleVariableRef GVarLumenScreenProbeImportanceSamplingNumLevels(
	TEXT("r.Lumen.ScreenProbeGather.ImportanceSample.NumLevels"),
	GLumenScreenProbeImportanceSamplingNumLevels,
	TEXT("Number of refinement levels to use for screen probe importance sampling.  Currently only supported by the serial reference path in ScreenProbeGenerateRaysCS."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenScreenProbeImportanceSampleIncomingLighting = 1;
FAutoConsoleVariableRef GVarLumenScreenProbeImportanceSampleIncomingLighting(
	TEXT("r.Lumen.ScreenProbeGather.ImportanceSample.IncomingLighting"),
	GLumenScreenProbeImportanceSampleIncomingLighting,
	TEXT("Whether to Importance Sample incoming lighting to generate probe trace directions.  When disabled, only the BRDF will be importance sampled."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenScreenProbeImportanceSampleProbeRadianceHistory = 1;
FAutoConsoleVariableRef GVarLumenScreenProbeImportanceSampleProbeRadianceHistory(
	TEXT("r.Lumen.ScreenProbeGather.ImportanceSample.ProbeRadianceHistory"),
	GLumenScreenProbeImportanceSampleProbeRadianceHistory,
	TEXT("Whether to Importance Sample incoming lighting from last frame's filtered traces to generate probe trace directions.  When disabled, the Radiance Cache will be used instead."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenScreenProbeBRDFOctahedronResolution = 8;
FAutoConsoleVariableRef CVarLumenScreenProbeBRDFOctahedronResolution(
	TEXT("r.Lumen.ScreenProbeGather.ImportanceSample.BRDFOctahedronResolution"),
	GLumenScreenProbeBRDFOctahedronResolution,
	TEXT("Resolution of the BRDF PDF octahedron per probe."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

float GLumenScreenProbeImportanceSamplingMinPDFToTrace = .1f;
FAutoConsoleVariableRef GVarLumenScreenProbeImportanceSamplingMinPDFToTrace(
	TEXT("r.Lumen.ScreenProbeGather.ImportanceSample.MinPDFToTrace"),
	GLumenScreenProbeImportanceSamplingMinPDFToTrace,
	TEXT("Minimum normalized BRDF PDF to trace rays for.  Larger values cause black corners, but reduce noise as more rays are able to be reassigned to an important direction."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

float GLumenScreenProbeImportanceSamplingHistoryDistanceThreshold = 30;
FAutoConsoleVariableRef CVarLumenScreenProbeImportanceSamplingHistoryDistanceThreshold(
	TEXT("r.Lumen.ScreenProbeGather.ImportanceSample.HistoryDistanceThreshold"),
	GLumenScreenProbeImportanceSamplingHistoryDistanceThreshold,
	TEXT(""),
	ECVF_RenderThreadSafe
	);

class FScreenProbeGatherScreenDataCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FScreenProbeGatherScreenDataCS)
	SHADER_USE_PARAMETER_STRUCT(FScreenProbeGatherScreenDataCS, FGlobalShader)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<UNORM float>, RWScreenProbeDisocclusion)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float>, RWBRDFProbabilityDensityFunction)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float>, RWBRDFProbabilityDensityFunctionSH)
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneTextureUniformParameters, SceneTexturesStruct)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSubstrateGlobalUniformParameters, Substrate)
		SHADER_PARAMETER_STRUCT_INCLUDE(FScreenProbeParameters, ScreenProbeParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(StochasticLighting::FHistoryScreenParameters, HistoryScreenParameters)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2DArray, HistoryFastUpdateMode_NumFramesAccumulated)
		SHADER_PARAMETER(float, MaxFramesAccumulated)
		SHADER_PARAMETER(float, SpatialFilterDisocclusionMaxFrames)
		SHADER_PARAMETER(float, SpatialFilterDisocclusionFraction)
	END_SHADER_PARAMETER_STRUCT()

	class FScreenProbeDisocclusion : SHADER_PERMUTATION_BOOL("SCREEN_PROBE_DISOCCLUSION");
	class FValidHistory : SHADER_PERMUTATION_BOOL("VALID_HISTORY");
	using FPermutationDomain = TShaderPermutationDomain<FScreenProbeDisocclusion, FValidHistory>;

	static FPermutationDomain RemapPermutation(FPermutationDomain PermutationVector)
	{
		if (!PermutationVector.Get<FScreenProbeDisocclusion>())
		{
			PermutationVector.Set<FValidHistory>(false);
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

		return DoesPlatformSupportLumenGI(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
	}
};

IMPLEMENT_GLOBAL_SHADER(FScreenProbeGatherScreenDataCS, "/Engine/Private/Lumen/LumenScreenProbeGatherScreenData.usf", "ScreenProbeGatherScreenDataCS", SF_Compute);


class FScreenProbeComputeLightingProbabilityDensityFunctionCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FScreenProbeComputeLightingProbabilityDensityFunctionCS)
	SHADER_USE_PARAMETER_STRUCT(FScreenProbeComputeLightingProbabilityDensityFunctionCS, FGlobalShader)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float>, RWLightingProbabilityDensityFunction)
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_STRUCT_INCLUDE(FScreenProbeParameters, ScreenProbeParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(LumenRadianceCache::FRadianceCacheInterpolationParameters, RadianceCacheParameters)
		SHADER_PARAMETER(FVector4f, ProbeHistoryScreenPositionScaleBias)
		SHADER_PARAMETER(FVector4f, ImportanceSamplingHistoryUVMinMax)
		SHADER_PARAMETER(float, ImportanceSamplingHistoryDistanceThreshold)
		SHADER_PARAMETER(float, PrevInvPreExposure)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float3>, HistoryScreenProbeRadiance)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, HistoryScreenProbeSceneDepth)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float3>, HistoryScreenProbeTranslatedWorldPosition)
		SHADER_PARAMETER_STRUCT_INCLUDE(FSceneTextureParameters, SceneTextures)
	END_SHADER_PARAMETER_STRUCT()

	static uint32 GetThreadGroupSize(uint32 TracingResolution)
	{
		if (TracingResolution == 4)
		{
			return 4;
		}
		else if (TracingResolution == 8)
		{
			return 8;
		}
		else if (TracingResolution == 16)
		{
			return 16;
		}
		else
		{
			return MAX_uint32;
		}
	}

	class FThreadGroupSize : SHADER_PERMUTATION_SPARSE_INT("LIGHTING_PDF_THREADGROUP_SIZE", 4, 8, 16);
	class FProbeRadianceHistory : SHADER_PERMUTATION_BOOL("PROBE_RADIANCE_HISTORY");
	class FRadianceCache : SHADER_PERMUTATION_BOOL("RADIANCE_CACHE");
	using FPermutationDomain = TShaderPermutationDomain<FThreadGroupSize, FProbeRadianceHistory, FRadianceCache>;

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportLumenGI(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.CompilerFlags.Add(CFLAG_Wave32);
	}
};

IMPLEMENT_GLOBAL_SHADER(FScreenProbeComputeLightingProbabilityDensityFunctionCS, "/Engine/Private/Lumen/LumenScreenProbeImportanceSampling.usf", "ScreenProbeComputeLightingProbabilityDensityFunctionCS", SF_Compute);



class FScreenProbeGenerateRaysCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FScreenProbeGenerateRaysCS)
	SHADER_USE_PARAMETER_STRUCT(FScreenProbeGenerateRaysCS, FGlobalShader)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWStructuredImportanceSampledRayInfosForTracing)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint2>, RWStructuredImportanceSampledRayCoordForComposite)
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, BRDFProbabilityDensityFunction)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float>, BRDFProbabilityDensityFunctionSH)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, LightingProbabilityDensityFunction)
		SHADER_PARAMETER(float, MinPDFToTrace)
		SHADER_PARAMETER_STRUCT_INCLUDE(FScreenProbeParameters, ScreenProbeParameters)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportLumenGI(Parameters.Platform);
	}

	static uint32 GetThreadGroupSize(uint32 TracingResolution)
	{
		if (TracingResolution == 4)
		{
			return 4;
		}
		else if (TracingResolution == 8)
		{
			return 8;
		}
		else if (TracingResolution == 16)
		{
			return 16;
		}
		else
		{
			return MAX_uint32;
		}
	}

	class FThreadGroupSize : SHADER_PERMUTATION_SPARSE_INT("GENERATE_RAYS_THREADGROUP_SIZE", 4, 8, 16);
	class FImportanceSampleLighting : SHADER_PERMUTATION_BOOL("IMPORTANCE_SAMPLE_LIGHTING");
	using FPermutationDomain = TShaderPermutationDomain<FThreadGroupSize, FImportanceSampleLighting>;

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.CompilerFlags.Add(CFLAG_Wave32);
	}
};

IMPLEMENT_GLOBAL_SHADER(FScreenProbeGenerateRaysCS, "/Engine/Private/Lumen/LumenScreenProbeImportanceSampling.usf", "ScreenProbeGenerateRaysCS", SF_Compute);

extern int32 GLumenScreenProbeGatherReferenceMode;

namespace LumenScreenProbeGather
{
	int32 IsProbeTracingResolutionSupportedForImportanceSampling(int32 TracingResolution)
	{
		return FScreenProbeGenerateRaysCS::GetThreadGroupSize(TracingResolution) != MAX_uint32;
	}

	bool UseImportanceSampling(const FViewInfo& View)
	{
		if (GLumenScreenProbeGatherReferenceMode)
		{
			return false;
		}

		// Shader permutations only created for these resolutions
		const int32 TracingResolution = GetTracingOctahedronResolution(View);
		return GLumenScreenProbeImportanceSampling != 0 && IsProbeTracingResolutionSupportedForImportanceSampling(TracingResolution);
	}
}

/**
 * Gather 64 pixels from the screen probe's interpolation footprint in order to compute ProbeBRDF_PDF and ProbeDisocclusion
 */
void ScreenProbeGatherScreenData(
	FRDGBuilder& GraphBuilder, 
	const FViewInfo& View, 
	const FSceneTextures& SceneTextures,
	FLumenSceneFrameTemporaries& FrameTemporaries,
	FRDGTextureRef& BRDFProbabilityDensityFunction,
	FRDGBufferSRVRef& BRDFProbabilityDensityFunctionSH,
	FScreenProbeParameters& ScreenProbeParameters,
	bool bIsHistoryValid,
	ERDGPassFlags ComputePassFlags)
{
	const FRDGSystemTextures& SystemTextures = FRDGSystemTextures::Get(GraphBuilder);

	const uint32 BRDFOctahedronResolution = GLumenScreenProbeBRDFOctahedronResolution;
	ScreenProbeParameters.ImportanceSampling.ScreenProbeBRDFOctahedronResolution = BRDFOctahedronResolution;

	FIntPoint PDFBufferSize = ScreenProbeParameters.ScreenProbeAtlasBufferSize * BRDFOctahedronResolution;
	FRDGTextureDesc BRDFProbabilityDensityFunctionDesc(FRDGTextureDesc::Create2D(PDFBufferSize, PF_R16F, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV));
	BRDFProbabilityDensityFunction = GraphBuilder.CreateTexture(BRDFProbabilityDensityFunctionDesc, TEXT("Lumen.ScreenProbeGather.BRDFProbabilityDensityFunction"));
	
	const int32 BRDF_SHBufferSize = ScreenProbeParameters.ScreenProbeAtlasBufferSize.X * ScreenProbeParameters.ScreenProbeAtlasBufferSize.Y * 9;
	FRDGBufferDesc BRDFProbabilityDensityFunctionSHDesc = FRDGBufferDesc::CreateBufferDesc(sizeof(FFloat16), BRDF_SHBufferSize);
	FRDGBufferRef BRDFProbabilityDensityFunctionSHBuffer = GraphBuilder.CreateBuffer(BRDFProbabilityDensityFunctionSHDesc, TEXT("Lumen.ScreenProbeGather.BRDFProbabilityDensityFunctionSH"));

	const bool bProbeDisocclusion = LumenScreenProbeGather::UseDisocclusionSpatialFilter();
	if (bProbeDisocclusion)
	{
		ScreenProbeParameters.ScreenProbeDisocclusion = GraphBuilder.CreateTexture(
			FRDGTextureDesc::Create2D(ScreenProbeParameters.ScreenProbeAtlasBufferSize, PF_R8, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
			TEXT("Lumen.ScreenProbeGather.ScreenProbeDisocclusion"));
	}

	StochasticLighting::FHistoryScreenParameters HistoryScreenParameters = StochasticLighting::GetHistoryScreenParameters(View);
	HistoryScreenParameters.EncodedHistoryScreenCoordTexture = FrameTemporaries.StochasticLighting.EncodedHistoryScreenCoord.GetRenderTarget();
	HistoryScreenParameters.PackedPixelDataTexture = FrameTemporaries.StochasticLighting.LumenPackedPixelData.GetRenderTarget();

	// Load last frame's history
	FRDGTextureRef OldFastUpdateMode_NumFramesAccumulatedHistory = nullptr;
	if (bIsHistoryValid)
	{
		const FScreenProbeGatherTemporalState& ScreenProbeGatherState = View.ViewState->Lumen.ScreenProbeGatherState;
		if (ScreenProbeGatherState.FastUpdateMode_NumFramesAccumulatedHistoryRT)
		{
			OldFastUpdateMode_NumFramesAccumulatedHistory = GraphBuilder.RegisterExternalTexture(ScreenProbeGatherState.FastUpdateMode_NumFramesAccumulatedHistoryRT);
		}
	}

	{
		FScreenProbeGatherScreenDataCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FScreenProbeGatherScreenDataCS::FParameters>();
		PassParameters->Substrate = Substrate::BindSubstrateGlobalUniformParameters(View);
		PassParameters->RWScreenProbeDisocclusion = bProbeDisocclusion ? GraphBuilder.CreateUAV(ScreenProbeParameters.ScreenProbeDisocclusion) : nullptr;
		PassParameters->RWBRDFProbabilityDensityFunction = GraphBuilder.CreateUAV(FRDGTextureUAVDesc(BRDFProbabilityDensityFunction));
		PassParameters->RWBRDFProbabilityDensityFunctionSH = GraphBuilder.CreateUAV(FRDGBufferUAVDesc(BRDFProbabilityDensityFunctionSHBuffer, PF_R16F));
		PassParameters->View = View.ViewUniformBuffer;
		PassParameters->SceneTexturesStruct = SceneTextures.UniformBuffer;
		PassParameters->ScreenProbeParameters = ScreenProbeParameters;
		PassParameters->HistoryScreenParameters = HistoryScreenParameters;
		PassParameters->HistoryFastUpdateMode_NumFramesAccumulated = OldFastUpdateMode_NumFramesAccumulatedHistory;
		PassParameters->MaxFramesAccumulated = LumenScreenProbeGather::GetMaxFramesAccumulated(View);
		PassParameters->SpatialFilterDisocclusionMaxFrames = LumenScreenProbeGather::GetSpatialFilterDisocclusionMaxFrames();
		PassParameters->SpatialFilterDisocclusionFraction = LumenScreenProbeGather::GetSpatialFilterDisocclusionFraction();

		FScreenProbeGatherScreenDataCS::FPermutationDomain PermutationVector;
		PermutationVector.Set<FScreenProbeGatherScreenDataCS::FScreenProbeDisocclusion>(bProbeDisocclusion);
		PermutationVector.Set<FScreenProbeGatherScreenDataCS::FValidHistory>(OldFastUpdateMode_NumFramesAccumulatedHistory != nullptr);
		PermutationVector = FScreenProbeGatherScreenDataCS::RemapPermutation(PermutationVector);

		auto ComputeShader = View.ShaderMap->GetShader<FScreenProbeGatherScreenDataCS>(PermutationVector);

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("ScreenProbeGatherScreenData"),
			ComputePassFlags,
			ComputeShader,
			PassParameters,
			ScreenProbeParameters.ProbeIndirectArgs,
			(uint32)EScreenProbeIndirectArgs::GroupPerProbe * sizeof(FRHIDispatchIndirectParameters));
	}

	BRDFProbabilityDensityFunctionSH = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(BRDFProbabilityDensityFunctionSHBuffer, PF_R16F));
}

void GenerateImportanceSamplingRays(
	FRDGBuilder& GraphBuilder, 
	const FViewInfo& View,
	const FSceneTextures& SceneTextures,
	const LumenRadianceCache::FRadianceCacheInterpolationParameters& RadianceCacheParameters,
	FRDGTextureRef BRDFProbabilityDensityFunction,
	FRDGBufferSRVRef BRDFProbabilityDensityFunctionSH,
	FScreenProbeParameters& ScreenProbeParameters,
	ERDGPassFlags ComputePassFlags)
{
	const uint32 MaxImportanceSamplingOctahedronResolution = ScreenProbeParameters.ScreenProbeTracingOctahedronResolution * (1 << GLumenScreenProbeImportanceSamplingNumLevels);
	ScreenProbeParameters.ImportanceSampling.MaxImportanceSamplingOctahedronResolution = MaxImportanceSamplingOctahedronResolution;

	const bool bImportanceSampleLighting = GLumenScreenProbeImportanceSampleIncomingLighting != 0;

	FRDGTextureRef LightingProbabilityDensityFunction = nullptr;

	if (bImportanceSampleLighting)
	{
		FIntPoint LightingProbabilityDensityFunctionBufferSize = ScreenProbeParameters.ScreenProbeAtlasBufferSize * ScreenProbeParameters.ScreenProbeTracingOctahedronResolution;
		FRDGTextureDesc LightingProbabilityDensityFunctionDesc(FRDGTextureDesc::Create2D(LightingProbabilityDensityFunctionBufferSize, PF_R16F, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV));
		LightingProbabilityDensityFunction = GraphBuilder.CreateTexture(LightingProbabilityDensityFunctionDesc, TEXT("Lumen.ScreenProbeGather.LightingProbabilityDensityFunction"));

		const FScreenProbeGatherTemporalState& ScreenProbeGatherState = View.ViewState->Lumen.ScreenProbeGatherState;

		const FIntPoint ScreenProbeGatherBufferSize = ScreenProbeParameters.ScreenProbeAtlasBufferSize * ScreenProbeParameters.ScreenProbeGatherOctahedronResolution;

		const bool bUseProbeRadianceHistory = GLumenScreenProbeImportanceSampleProbeRadianceHistory != 0 
			&& ScreenProbeGatherState.ImportanceSamplingHistoryScreenProbeRadiance.IsValid()
			&& !View.bCameraCut 
			&& !View.bPrevTransformsReset
			&& ScreenProbeGatherState.ImportanceSamplingHistoryScreenProbeRadiance->GetDesc().Extent == ScreenProbeGatherBufferSize;

		{
			FScreenProbeComputeLightingProbabilityDensityFunctionCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FScreenProbeComputeLightingProbabilityDensityFunctionCS::FParameters>();
			PassParameters->RWLightingProbabilityDensityFunction = GraphBuilder.CreateUAV(FRDGTextureUAVDesc(LightingProbabilityDensityFunction));
			PassParameters->ScreenProbeParameters = ScreenProbeParameters;
			PassParameters->RadianceCacheParameters = RadianceCacheParameters;
			PassParameters->View = View.ViewUniformBuffer;
			PassParameters->SceneTextures = GetSceneTextureParameters(GraphBuilder, SceneTextures.UniformBuffer);

			if (bUseProbeRadianceHistory)
			{
				const FRDGSystemTextures& SystemTextures = FRDGSystemTextures::Get(GraphBuilder);

				PassParameters->PrevInvPreExposure = 1.0f / View.PrevViewInfo.SceneColorPreExposure;
				PassParameters->ProbeHistoryScreenPositionScaleBias = ScreenProbeGatherState.ProbeHistoryScreenPositionScaleBias;

				const FIntPoint SceneTexturesExtent = View.GetSceneTexturesConfig().Extent;
				const FVector2D InvBufferSize(1.0f / SceneTexturesExtent.X, 1.0f / SceneTexturesExtent.Y);

				// Pull in the max UV to exclude the region which will read outside the viewport due to bilinear filtering
				PassParameters->ImportanceSamplingHistoryUVMinMax = FVector4f(
					(ScreenProbeGatherState.ProbeHistoryViewRect.Min.X + 0.5f) * InvBufferSize.X,
					(ScreenProbeGatherState.ProbeHistoryViewRect.Min.Y + 0.5f) * InvBufferSize.Y,
					(ScreenProbeGatherState.ProbeHistoryViewRect.Max.X - 0.5f) * InvBufferSize.X,
					(ScreenProbeGatherState.ProbeHistoryViewRect.Max.Y - 0.5f) * InvBufferSize.Y);

				PassParameters->ImportanceSamplingHistoryDistanceThreshold = GLumenScreenProbeImportanceSamplingHistoryDistanceThreshold;
				PassParameters->HistoryScreenProbeRadiance = GraphBuilder.RegisterExternalTexture(ScreenProbeGatherState.ImportanceSamplingHistoryScreenProbeRadiance);
				PassParameters->HistoryScreenProbeSceneDepth = GraphBuilder.RegisterExternalTexture(ScreenProbeGatherState.HistoryScreenProbeSceneDepth);
				PassParameters->HistoryScreenProbeTranslatedWorldPosition = GraphBuilder.RegisterExternalTexture(ScreenProbeGatherState.HistoryScreenProbeTranslatedWorldPosition);
			}

			const uint32 ComputeLightingPDFGroupSize = FScreenProbeComputeLightingProbabilityDensityFunctionCS::GetThreadGroupSize(ScreenProbeParameters.ScreenProbeTracingOctahedronResolution);
			check(ComputeLightingPDFGroupSize != MAX_uint32);

			FScreenProbeComputeLightingProbabilityDensityFunctionCS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FScreenProbeComputeLightingProbabilityDensityFunctionCS::FThreadGroupSize>(ComputeLightingPDFGroupSize);
			PermutationVector.Set<FScreenProbeComputeLightingProbabilityDensityFunctionCS::FProbeRadianceHistory>(bUseProbeRadianceHistory);
			PermutationVector.Set<FScreenProbeComputeLightingProbabilityDensityFunctionCS::FRadianceCache>(LumenScreenProbeGather::UseRadianceCache());
			auto ComputeShader = View.ShaderMap->GetShader<FScreenProbeComputeLightingProbabilityDensityFunctionCS>(PermutationVector);

			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("ComputeLightingPDF"),
				ComputePassFlags,
				ComputeShader,
				PassParameters,
				ScreenProbeParameters.ProbeIndirectArgs,
				// Spawn a group on every probe
				(uint32)EScreenProbeIndirectArgs::GroupPerProbe * sizeof(FRHIDispatchIndirectParameters));
		}
	}

	FIntPoint RayInfosForTracingBufferSize = ScreenProbeParameters.ScreenProbeAtlasBufferSize * ScreenProbeParameters.ScreenProbeTracingOctahedronResolution;
	FRDGTextureDesc StructuredImportanceSampledRayInfosForTracingDesc(FRDGTextureDesc::Create2D(RayInfosForTracingBufferSize, PF_R16_UINT, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV));
	ScreenProbeParameters.ImportanceSampling.StructuredImportanceSampledRayInfosForTracing = GraphBuilder.CreateTexture(StructuredImportanceSampledRayInfosForTracingDesc, TEXT("Lumen.ScreenProbeGather.RayInfosForTracing"));

	{
		FScreenProbeGenerateRaysCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FScreenProbeGenerateRaysCS::FParameters>();
		PassParameters->RWStructuredImportanceSampledRayInfosForTracing = GraphBuilder.CreateUAV(FRDGTextureUAVDesc(ScreenProbeParameters.ImportanceSampling.StructuredImportanceSampledRayInfosForTracing));
		PassParameters->View = View.ViewUniformBuffer;
		PassParameters->BRDFProbabilityDensityFunction = BRDFProbabilityDensityFunction;
		PassParameters->LightingProbabilityDensityFunction = LightingProbabilityDensityFunction;
		PassParameters->BRDFProbabilityDensityFunctionSH = BRDFProbabilityDensityFunctionSH;
		PassParameters->MinPDFToTrace = GLumenScreenProbeImportanceSamplingMinPDFToTrace;
		PassParameters->ScreenProbeParameters = ScreenProbeParameters;

		const uint32 GenerateRaysGroupSize = FScreenProbeGenerateRaysCS::GetThreadGroupSize(ScreenProbeParameters.ScreenProbeTracingOctahedronResolution);
		check(GenerateRaysGroupSize != MAX_uint32);

		FScreenProbeGenerateRaysCS::FPermutationDomain PermutationVector;
		PermutationVector.Set< FScreenProbeGenerateRaysCS::FThreadGroupSize >(GenerateRaysGroupSize);
		PermutationVector.Set< FScreenProbeGenerateRaysCS::FImportanceSampleLighting >(bImportanceSampleLighting);
		auto ComputeShader = View.ShaderMap->GetShader<FScreenProbeGenerateRaysCS>(PermutationVector);

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("GenerateRays %ux%u", GenerateRaysGroupSize, GenerateRaysGroupSize),
			ComputePassFlags,
			ComputeShader,
			PassParameters,
			ScreenProbeParameters.ProbeIndirectArgs,
			// Spawn a group on every probe
			(uint32)EScreenProbeIndirectArgs::GroupPerProbe * sizeof(FRHIDispatchIndirectParameters));
	}
}
