// Copyright Epic Games, Inc. All Rights Reserved.

#include "SceneUtils.h"
#include "PipelineStateCache.h"
#include "ShaderParameterStruct.h"
#include "PixelShaderUtils.h"
#include "ReflectionEnvironment.h"
#include "DistanceFieldAmbientOcclusion.h"
#include "ScreenSpaceDenoise.h"
#include "LumenRadianceCache.h"
#include "LumenTracingUtils.h"
#include "LumenReflections.h"
#include "LumenScreenProbeGather.h"
#include "LumenShortRangeAO.h"
#include "SceneViewState.h"
#include "DeferredShadingRenderer.h"

#if RHI_RAYTRACING
#include "RayTracing/RaytracingOptions.h"
#endif // RHI_RAYTRACING

int32 GLumenIrradianceFieldNumClipmaps = 6;
FAutoConsoleVariableRef CVarLumenIrradianceFieldNumClipmaps(
	TEXT("r.Lumen.IrradianceFieldGather.NumClipmaps"),
	GLumenIrradianceFieldNumClipmaps,
	TEXT("Number of radiance cache clipmaps."),
	ECVF_RenderThreadSafe
);

float GLumenIrradianceFieldClipmapWorldExtent = 5000.0f;
FAutoConsoleVariableRef CVarLumenIrradianceFieldClipmapWorldExtent(
	TEXT("r.Lumen.IrradianceFieldGather.ClipmapWorldExtent"),
	GLumenIrradianceFieldClipmapWorldExtent,
	TEXT("World space extent of the first clipmap"),
	ECVF_RenderThreadSafe
);

float GLumenIrradianceFieldClipmapDistributionBase = 2.0f;
FAutoConsoleVariableRef CVarLumenIrradianceFieldClipmapDistributionBase(
	TEXT("r.Lumen.IrradianceFieldGather.ClipmapDistributionBase"),
	GLumenIrradianceFieldClipmapDistributionBase,
	TEXT("Base of the Pow() that controls the size of each successive clipmap relative to the first."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarLumenIrradianceFieldNumProbesToTraceBudget(
	TEXT("r.Lumen.IrradianceFieldGather.NumProbesToTraceBudget"),
	100,
	TEXT("Number of radiance cache probes that can be updated per frame."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldGridResolution = 64;
FAutoConsoleVariableRef CVarLumenIrradianceFieldResolution(
	TEXT("r.Lumen.IrradianceFieldGather.GridResolution"),
	GLumenIrradianceFieldGridResolution,
	TEXT("Resolution of the probe placement grid within each clipmap"),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldProbeResolution = 16;
FAutoConsoleVariableRef CVarLumenIrradianceFieldProbeResolution(
	TEXT("r.Lumen.IrradianceFieldGather.ProbeResolution"),
	GLumenIrradianceFieldProbeResolution,
	TEXT("Resolution of the probe's 2d radiance layout.  The number of rays traced for the probe will be ProbeResolution ^ 2. This must match occlusion probe resolution currently."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldProbeIrradianceResolution = 6;
FAutoConsoleVariableRef CVarLumenIrradianceFieldProbeIrradianceResolution(
	TEXT("r.Lumen.IrradianceFieldGather.IrradianceProbeResolution"),
	GLumenIrradianceFieldProbeIrradianceResolution,
	TEXT("Resolution of the probe's 2d irradiance layout."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldProbeOcclusionResolution = 16;
FAutoConsoleVariableRef CVarLumenIrradianceFieldProbeOcclusionResolution(
	TEXT("r.Lumen.IrradianceFieldGather.OcclusionProbeResolution"),
	GLumenIrradianceFieldProbeOcclusionResolution,
	TEXT("Resolution of the probe's 2d occlusion layout. This must match radiance probe resolution currently."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarLumenIrradianceFieldNumMipmaps(
	TEXT("r.Lumen.IrradianceFieldGather.NumMipmaps"),
	3,
	TEXT("Number of radiance cache mipmaps."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldProbeAtlasResolutionInProbes = 128;
FAutoConsoleVariableRef CVarLumenIrradianceFieldProbeAtlasResolutionInProbes(
	TEXT("r.Lumen.IrradianceFieldGather.ProbeAtlasResolutionInProbes"),
	GLumenIrradianceFieldProbeAtlasResolutionInProbes,
	TEXT("Number of probes along one dimension of the probe atlas cache texture.  This controls the memory usage of the cache.  Overflow currently results in incorrect rendering."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

float GLumenIrradianceFieldProbeOcclusionBias = .8f;
FAutoConsoleVariableRef CVarLumenIrradianceFieldProbeOcclusionBias(
	TEXT("r.Lumen.IrradianceFieldGather.ProbeOcclusionBias"),
	GLumenIrradianceFieldProbeOcclusionBias,
	TEXT("Bias along the normal and toward the viewer to reduce self-occlusion artifacts from Probe Occlusion"),
	ECVF_RenderThreadSafe
);

float GLumenIrradianceFieldShortRangeAORadius = 32.0f;
FAutoConsoleVariableRef CVarLumenIrradianceFieldShortRangeAORadius(
	TEXT("r.Lumen.IrradianceFieldGather.ShortRangeAORadius"),
	GLumenIrradianceFieldShortRangeAORadius,
	TEXT("Radius of ShortRangeAO, in pixels"),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldStats = 0;
FAutoConsoleVariableRef CVarLumenIrradianceFieldStats(
	TEXT("r.Lumen.IrradianceFieldGather.RadianceCache.Stats"),
	GLumenIrradianceFieldStats,
	TEXT("GPU print out Radiance Cache update stats."),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldInterpolateDownsampleFactor = 2;
FAutoConsoleVariableRef CVarLumenIrradianceFieldInterpolateDownsampleFactor(
	TEXT("r.Lumen.IrradianceFieldGather.InterpolateDownsampleFactor"),
	GLumenIrradianceFieldInterpolateDownsampleFactor,
	TEXT("Downsample factor for Irradiance Field interpolation and shading."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldAdaptivePlacement = 0;
FAutoConsoleVariableRef CVarLumenIrradianceFieldAdaptivePlacement(
	TEXT("r.Lumen.IrradianceFieldGather.AdaptivePlacement"),
	GLumenIrradianceFieldAdaptivePlacement,
	TEXT("Whether to place new probes where interpolation fails due to occlusion."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldAdaptivePlacementInterpolationFeedbackTileSize = 8;
FAutoConsoleVariableRef CVarLumenIrradianceFieldAdaptivePlacementInterpolationFeedbackTileSize(
	TEXT("r.Lumen.IrradianceFieldGather.AdaptivePlacement.InterpolationFeedbackTileSize"),
	GLumenIrradianceFieldAdaptivePlacementInterpolationFeedbackTileSize,
	TEXT("One pixel in every screen tile will process interpolation failures for adaptive probe placement."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

float GLumenIrradianceFieldMaxRoughnessToEvaluateRoughSpecular = .8f;
FAutoConsoleVariableRef CVarLumenIrradianceFieldMaxRoughnessToEvaluateRoughSpecular(
	TEXT("r.Lumen.IrradianceFieldGather.MaxRoughnessToEvaluateRoughSpecular"),
	GLumenIrradianceFieldMaxRoughnessToEvaluateRoughSpecular,
	TEXT("Maximum material roughness to evaluate rough specular for. Pixels with roughness larger than this will reuse diffuse GI as their specular GI."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldMarkProbesDownsampleFactor = 2;
FAutoConsoleVariableRef CVarLumenIrradianceFieldMarkProbesDownsampleFactor(
	TEXT("r.Lumen.IrradianceFieldGather.MarkProbesDownsampleFactor"),
	GLumenIrradianceFieldMarkProbesDownsampleFactor,
	TEXT("Downsample factor (applied on top of InterpolateDownsampleFactor) used to mark the irradiance probes as used by the GBuffer. Downsampling too high can cause small geometry to flicker black."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

namespace LumenIrradianceFieldGather
{
	int32 GetNumMipmaps()
	{
		return FMath::Clamp(CVarLumenIrradianceFieldNumMipmaps.GetValueOnRenderThread(), 1, LumenRadianceCache::MaxMipmaps);
	}

	float GetSceneRadius(const FViewInfo& View, const FSceneViewFamily& ViewFamily)
	{
		float SceneRadius = 2000000.0f;
		const bool bHWRT = Lumen::UseHardwareRayTracedRadianceCache(ViewFamily);

		if (bHWRT)
		{
			#if RHI_RAYTRACING
			// Limit to ray tracing culling radius if ray tracing is used
			if (RayTracing::GetCullingMode(View.Family->EngineShowFlags) != RayTracing::ECullingMode::Disabled)
			{
				SceneRadius = GetRayTracingCullingRadius();

				if (Lumen::UseFarField(ViewFamily))
				{
					SceneRadius = Lumen::GetFarFieldMaxTraceDistance();
				}
			}
			#endif
		}
		else
		{
			const float LastClipmapExtent = Lumen::GetGlobalDFClipmapExtent(Lumen::GetNumGlobalDFClipmaps(View) - 1);
			SceneRadius = LastClipmapExtent;
		}
		
		return SceneRadius;
	}

	LumenRadianceCache::FRadianceCacheInputs SetupRadianceCacheInputs(const FViewInfo& View)
	{
		LumenRadianceCache::FRadianceCacheInputs Parameters = LumenRadianceCache::GetDefaultRadianceCacheInputs();
		Parameters.ReprojectionRadiusScale = 1.5f;
		Parameters.ClipmapWorldExtent = GLumenIrradianceFieldClipmapWorldExtent;
		Parameters.ClipmapDistributionBase = FMath::Clamp<float>(GLumenIrradianceFieldClipmapDistributionBase, .1f, 10.0f);
		Parameters.RadianceProbeClipmapResolution = FMath::Clamp(GLumenIrradianceFieldGridResolution, 1, 256);
		Parameters.ProbeAtlasResolutionInProbes = LumenRadianceCache::GetProbeAtlasResolutionInProbes(GLumenIrradianceFieldProbeAtlasResolutionInProbes);

		{
			const bool bTranslucencyVolumeShareRadianceCacheWithOpaque = LumenTranslucencyVolumeRadianceCache::ShareWithOpaque();
			// Don't cull clipmaps that will be used by the translucency volume
			const int32 MinClipmapIndex = bTranslucencyVolumeShareRadianceCacheWithOpaque ? LumenTranslucencyVolumeRadianceCache::GetClipmapBias() : 0;
			const int32 NumClipmapsCVar = FMath::Clamp<int32>(GLumenIrradianceFieldNumClipmaps, MinClipmapIndex + 1, LumenRadianceCache::MaxClipmaps);
			const float SceneRadius = GetSceneRadius(View, *View.Family);

			// Solve for the number of clipmaps needed to cover the scene radius, rounded up
			float MaxClipmapFloat = FMath::Log2(SceneRadius / Parameters.ClipmapWorldExtent) / FMath::Log2(Parameters.ClipmapDistributionBase);
			Parameters.NumRadianceProbeClipmaps = FMath::Clamp<int32>(FMath::CeilToInt(MaxClipmapFloat) + 1, MinClipmapIndex + 1, NumClipmapsCVar);
		}

		Parameters.RadianceProbeResolution = FMath::Max(GLumenIrradianceFieldProbeResolution, LumenRadianceCache::MinRadianceProbeResolution);
		Parameters.FinalProbeResolution = GLumenIrradianceFieldProbeResolution + 2 * (1 << (GetNumMipmaps() - 1));
		Parameters.FinalRadianceAtlasMaxMip = GetNumMipmaps() - 1;
		Parameters.CalculateIrradiance = 1;
		Parameters.UseAdaptiveProbes = GLumenIrradianceFieldAdaptivePlacement != 0 ? 1 : 0;
		Parameters.IrradianceProbeResolution = GLumenIrradianceFieldProbeIrradianceResolution;
		Parameters.OcclusionProbeResolution = GLumenIrradianceFieldProbeOcclusionResolution;
		Parameters.NumProbesToTraceBudget = CVarLumenIrradianceFieldNumProbesToTraceBudget.GetValueOnRenderThread();
		Parameters.RadianceCacheStats = GLumenIrradianceFieldStats;
		// Avoid overflow with quick camera movements
		Parameters.NumFramesToKeepCachedProbes = 2;
		return Parameters;
	}

	bool NeedsStochasticLightingDownsample()
	{
		return LumenShortRangeAO::GetDownsampleFactor() != 1;
	}

	float GetProbeOcclusionBias()
	{
		return GLumenIrradianceFieldProbeOcclusionBias;
	}

	float GetMaxRoughnessToEvaluateRoughSpecular()
	{
		return GLumenIrradianceFieldMaxRoughnessToEvaluateRoughSpecular;
	}
}

class FMarkProbesUsedByGBufferCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FMarkProbesUsedByGBufferCS)
	SHADER_USE_PARAMETER_STRUCT(FMarkProbesUsedByGBufferCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneTextureUniformParameters, SceneTexturesStruct)
		SHADER_PARAMETER_STRUCT_INCLUDE(LumenRadianceCache::FRadianceCacheMarkParameters, RadianceCacheMarkParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FLumenUpsampleParameters, UpsampleParameters)
		SHADER_PARAMETER(uint32, MarkProbesDownsampleFactor)
		SHADER_PARAMETER(float, ProbeOcclusionBias)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float>, RWDownsampledSceneDepth)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<UNORM float3>, RWDownsampledNormal)
	END_SHADER_PARAMETER_STRUCT()

	class FInterpolateDownsampleFactor : SHADER_PERMUTATION_RANGE_INT("INTERPOLATE_DOWNSAMPLE_FACTOR", 1, 2);
	class FDownsampleDepthNormal : SHADER_PERMUTATION_BOOL("DOWNSAMPLE_DEPTH_NORMAL");
	using FPermutationDomain = TShaderPermutationDomain<FInterpolateDownsampleFactor, FDownsampleDepthNormal>;

public:

	static FPermutationDomain RemapPermutation(FPermutationDomain PermutationVector)
	{
		if (PermutationVector.Get<FInterpolateDownsampleFactor>() == 1)
		{
			PermutationVector.Set<FDownsampleDepthNormal>(false);
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

		return DoesPlatformSupportLumenGI(Parameters.Platform);
	}

	static int32 GetGroupSize()
	{
		return 8;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}
};

IMPLEMENT_GLOBAL_SHADER(FMarkProbesUsedByGBufferCS, "/Engine/Private/Lumen/LumenIrradianceFieldGather.usf", "MarkProbesUsedByGBufferCS", SF_Compute);


class FInterpolateIrradianceProbesToPixelsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FInterpolateIrradianceProbesToPixelsCS)
	SHADER_USE_PARAMETER_STRUCT(FInterpolateIrradianceProbesToPixelsCS, FGlobalShader)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2DArray<float3>, RWDiffuseIndirect)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2DArray<float3>, RWRoughSpecularIndirect)
		SHADER_PARAMETER_STRUCT_INCLUDE(LumenRadianceCache::FRadianceCacheInterpolationParameters, RadianceCacheParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(LumenReflections::FCompositeParameters, ReflectionsCompositeParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FLumenUpsampleParameters, UpsampleParameters)
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneTextureUniformParameters, SceneTexturesStruct)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSubstrateGlobalUniformParameters, Substrate)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FReflectionUniformParameters, ReflectionStruct)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, ProbeAdaptiveIndices)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, ProbeValid)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, RWProbeLastUsedFrame)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, RWProbeInterpolationMisses)
		SHADER_PARAMETER(uint32, InterpolationFeedbackTileWrapMask)
		SHADER_PARAMETER(FIntPoint, InterpolationFeedbackTileJitter)
		SHADER_PARAMETER(uint32, MaxNumProbes)
		SHADER_PARAMETER(float, ProbeOcclusionBias)
		SHADER_PARAMETER(float, MaxRoughnessToEvaluateRoughSpecular)
		SHADER_PARAMETER(float, SkylightPrefilterMipBias)
	END_SHADER_PARAMETER_STRUCT()

	class FInterpolateDownsampleFactor : SHADER_PERMUTATION_RANGE_INT("INTERPOLATE_DOWNSAMPLE_FACTOR", 1, 2);
	class FAdaptiveProbes : SHADER_PERMUTATION_BOOL("ADAPTIVE_PROBES");
	using FPermutationDomain = TShaderPermutationDomain<FInterpolateDownsampleFactor, FAdaptiveProbes>;

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportLumenGI(Parameters.Platform);
	}

	static int32 GetGroupSize() 
	{
		return 8;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}
};

IMPLEMENT_GLOBAL_SHADER(FInterpolateIrradianceProbesToPixelsCS, "/Engine/Private/Lumen/LumenIrradianceFieldGather.usf", "InterpolateIrradianceProbesToPixelsCS", SF_Compute);

void RenderHairStrandsLumenLighting(
	FRDGBuilder& GraphBuilder,
	const FScene* Scene,
	const FViewInfo& View);

void HairStrandsMarkUsedProbes(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	const LumenRadianceCache::FRadianceCacheMarkParameters& RadianceCacheMarkParameters,
	ERDGPassFlags ComputePassFlags);	

static void IrradianceFieldMarkUsedProbes(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	const FSceneTextures& SceneTextures,
	const LumenRadianceCache::FRadianceCacheMarkParameters& RadianceCacheMarkParameters,
	const FLumenUpsampleParameters& UpsampleParameters,
	bool bDownsampleDepthNormal,
	ERDGPassFlags ComputePassFlags)
{
	const uint32 MarkDownsampleFactor = FMath::Clamp<uint32>(GLumenIrradianceFieldMarkProbesDownsampleFactor, 1, 8);

	FMarkProbesUsedByGBufferCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FMarkProbesUsedByGBufferCS::FParameters>();
	PassParameters->View = View.ViewUniformBuffer;
	PassParameters->SceneTexturesStruct = SceneTextures.UniformBuffer;
	PassParameters->RadianceCacheMarkParameters = RadianceCacheMarkParameters;
	PassParameters->UpsampleParameters = UpsampleParameters;
	PassParameters->MarkProbesDownsampleFactor = MarkDownsampleFactor;
	PassParameters->ProbeOcclusionBias = LumenIrradianceFieldGather::GetProbeOcclusionBias();

	if (bDownsampleDepthNormal)
	{
		PassParameters->UpsampleParameters.DownsampledSceneDepth = nullptr;
		PassParameters->UpsampleParameters.DownsampledSceneWorldNormal = nullptr;
		PassParameters->RWDownsampledSceneDepth = GraphBuilder.CreateUAV(UpsampleParameters.DownsampledSceneDepth);
		PassParameters->RWDownsampledNormal = GraphBuilder.CreateUAV(UpsampleParameters.DownsampledSceneWorldNormal);
	}

	FMarkProbesUsedByGBufferCS::FPermutationDomain PermutationVector;
	PermutationVector.Set<FMarkProbesUsedByGBufferCS::FInterpolateDownsampleFactor>(UpsampleParameters.DownsampleFactor);
	PermutationVector.Set<FMarkProbesUsedByGBufferCS::FDownsampleDepthNormal>(bDownsampleDepthNormal);
	auto ComputeShader = View.ShaderMap->GetShader<FMarkProbesUsedByGBufferCS>(PermutationVector);

	FIntVector GroupCount;
	if (bDownsampleDepthNormal)
	{
		GroupCount = FComputeShaderUtils::GetGroupCount(UpsampleParameters.IntegrateViewSize, FMarkProbesUsedByGBufferCS::GetGroupSize());
	}
	else
	{
		GroupCount = FComputeShaderUtils::GetGroupCount(FIntPoint::DivideAndRoundUp(UpsampleParameters.IntegrateViewSize, MarkDownsampleFactor), FMarkProbesUsedByGBufferCS::GetGroupSize());
	}

	FComputeShaderUtils::AddPass(
		GraphBuilder,
		RDG_EVENT_NAME("MarkProbesUsedByGBuffer InterpFactor=%u, MarkFactor=%u, DepthNormal=%u", UpsampleParameters.DownsampleFactor, MarkDownsampleFactor, bDownsampleDepthNormal ? 1 : 0),
		ComputePassFlags,
		ComputeShader,
		PassParameters,
		GroupCount);

	// When view has some hair strands, mark probes
	if (HairStrands::HasViewHairStrandsData(View))
	{
		HairStrandsMarkUsedProbes(GraphBuilder, View, RadianceCacheMarkParameters, ComputePassFlags);
	}
}

FIntPoint GetFeedbackBufferTileJitter(uint32 TileSize, uint32 FrameIndex) 
{
	const uint32 TileSizeLog2 = FMath::CeilLogTwo(TileSize);
	const uint32 SequenceSize = FMath::Square(TileSize);
	const uint32 PixelIndex = FrameIndex % SequenceSize;
	const uint32 PixelAddress = ReverseBits(PixelIndex) >> (32U - 2 * TileSizeLog2);

	FIntPoint TileJitter;
	TileJitter.X = FMath::ReverseMortonCode2(PixelAddress);
	TileJitter.Y = FMath::ReverseMortonCode2(PixelAddress >> 1);

	return TileJitter;
}

DECLARE_GPU_STAT(LumenIrradianceFieldGather);

FSSDSignalTextures FDeferredShadingSceneRenderer::RenderLumenIrradianceFieldGather(
	FRDGBuilder& GraphBuilder,
	const FSceneTextures& SceneTextures,
	const FLumenSceneFrameTemporaries& FrameTemporaries,
	FRDGTextureRef LightingChannelsTexture,
	FViewInfo& View,
	FLumenScreenSpaceBentNormalParameters& ScreenSpaceBentNormalParameters,
	FLumenUpsampleParameters& UpsampleParameters,
	LumenRadianceCache::FRadianceCacheInterpolationParameters& TranslucencyVolumeRadianceCacheParameters,
	ERDGPassFlags ComputePassFlags)
{
	RDG_EVENT_SCOPE_STAT(GraphBuilder, LumenIrradianceFieldGather, "LumenIrradianceFieldGather");

	const LumenRadianceCache::FRadianceCacheInputs RadianceCacheInputs = LumenIrradianceFieldGather::SetupRadianceCacheInputs(View);
	const int32 MaxNumProbes = RadianceCacheInputs.ProbeAtlasResolutionInProbes.X * RadianceCacheInputs.ProbeAtlasResolutionInProbes.Y;
	const int32 InterpolateDownsampleFactor = FMath::Clamp<int32>(GLumenIrradianceFieldInterpolateDownsampleFactor, 1, 2);
	const FIntPoint DownsampledBufferSize = FIntPoint::DivideAndRoundUp(SceneTextures.Config.Extent, InterpolateDownsampleFactor);

	{
		UpsampleParameters.DownsampleFactor = InterpolateDownsampleFactor;
		UpsampleParameters.IntegrateViewMin = FIntPoint::DivideAndRoundUp(View.ViewRect.Min, InterpolateDownsampleFactor);
		UpsampleParameters.IntegrateViewSize = FIntPoint::DivideAndRoundUp(View.ViewRect.Size(), InterpolateDownsampleFactor);
		UpsampleParameters.DownsampledBufferInvSize = FVector2f(1.0f) / DownsampledBufferSize;
		UpsampleParameters.ScreenProbeGatherStateFrameIndex = LumenScreenProbeGather::GetStateFrameIndex(View.ViewState);
	}

	bool bMarkByGBufferDownsampleDepthNormal = false;

	if (InterpolateDownsampleFactor != 1)
	{
		FRDGTextureRef DownsampledSceneDepth = nullptr;
		FRDGTextureRef DownsampledWorldNormal = nullptr;

		// If the Stochastic Lighting downsample pass ran then reuse that
		if (FrameTemporaries.StochasticLighting.DownsampledSceneDepth2x2.GetRenderTarget() != nullptr
			&& FrameTemporaries.StochasticLighting.DownsampledWorldNormal2x2.GetRenderTarget() != nullptr)
		{
			DownsampledSceneDepth = FrameTemporaries.StochasticLighting.DownsampledSceneDepth2x2.GetRenderTarget();
			DownsampledWorldNormal = FrameTemporaries.StochasticLighting.DownsampledWorldNormal2x2.GetRenderTarget();
		}
		else
		{
			ensureMsgf(!LumenIrradianceFieldGather::NeedsStochasticLightingDownsample(), TEXT("InternalRequiresStochasticLightingPass needs to be updated"));

			// Otherwise run our own dedicated downsample (which is faster since it runs only on downsampled pixels)
			FRDGTextureDesc DownsampledSceneDepthDesc = FRDGTextureDesc::Create2D(DownsampledBufferSize, PF_R32_FLOAT, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV);
			DownsampledSceneDepth = GraphBuilder.CreateTexture(DownsampledSceneDepthDesc, TEXT("r.Lumen.IrradianceFieldGather.DownsampledSceneDepth"));

			FRDGTextureDesc DownsampledWorldNormalDesc = FRDGTextureDesc::Create2D(DownsampledBufferSize, PF_A2B10G10R10, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV);
			DownsampledWorldNormal = GraphBuilder.CreateTexture(DownsampledWorldNormalDesc, TEXT("r.Lumen.IrradianceFieldGather.DownsampledWorldNormal"));

			bMarkByGBufferDownsampleDepthNormal = true;
		}

		UpsampleParameters.DownsampledSceneDepth = DownsampledSceneDepth;
		UpsampleParameters.DownsampledSceneWorldNormal = DownsampledWorldNormal;
	}

	const bool bTranslucencyVolumeShareRadianceCacheWithOpaque = LumenTranslucencyVolumeRadianceCache::ShareWithOpaque();

	FMarkUsedRadianceCacheProbes GraphicsMarkUsedRadianceCacheProbesCallbacks;
	FMarkUsedRadianceCacheProbes ComputeMarkUsedRadianceCacheProbesCallbacks;

	ComputeMarkUsedRadianceCacheProbesCallbacks.AddLambda([&SceneTextures, UpsampleParameters, bMarkByGBufferDownsampleDepthNormal, ComputePassFlags](
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		const LumenRadianceCache::FRadianceCacheMarkParameters& RadianceCacheMarkParameters)
		{
			IrradianceFieldMarkUsedProbes(
				GraphBuilder,
				View,
				SceneTextures,
				RadianceCacheMarkParameters,
				UpsampleParameters,
				bMarkByGBufferDownsampleDepthNormal,
				ComputePassFlags);
		});

	if (CVarLumenTranslucencyVolume.GetValueOnRenderThread() && bTranslucencyVolumeShareRadianceCacheWithOpaque)
	{
		const FLumenTranslucencyLightingVolumeParameters VolumeParameters = GetTranslucencyLightingVolumeParameters(GraphBuilder, View);

		ComputeMarkUsedRadianceCacheProbesCallbacks.AddLambda([&SceneTextures, VolumeParameters, ComputePassFlags](
			FRDGBuilder& GraphBuilder,
			const FViewInfo& View,
			const LumenRadianceCache::FRadianceCacheMarkParameters& RadianceCacheMarkParameters)
			{
				MarkRadianceProbesUsedByTranslucencyVolume(
					GraphBuilder,
					View,
					VolumeParameters,
					RadianceCacheMarkParameters,
					LumenTranslucencyVolumeRadianceCache::GetClipmapBias(),
					ComputePassFlags);
			});
	}

	if (Lumen::UseLumenTranslucencyRadianceCacheReflections(ViewFamily))
	{
		const FSceneRenderer& SceneRenderer = *this;
		FViewInfo& ViewNonConst = View;

		GraphicsMarkUsedRadianceCacheProbesCallbacks.AddLambda([&SceneTextures, &SceneRenderer, &ViewNonConst](
			FRDGBuilder& GraphBuilder,
			const FViewInfo& View,
			const LumenRadianceCache::FRadianceCacheMarkParameters& RadianceCacheMarkParameters)
			{
				LumenTranslucencyReflectionsMarkUsedProbes(
					GraphBuilder,
					SceneRenderer,
					ViewNonConst,
					SceneTextures,
					&RadianceCacheMarkParameters);
			});
	}

	LumenRadianceCache::FRadianceCacheInterpolationParameters RadianceCacheParameters;

	LumenRadianceCache::TInlineArray<LumenRadianceCache::FUpdateInputs> InputArray;
	LumenRadianceCache::TInlineArray<LumenRadianceCache::FUpdateOutputs> OutputArray;

	InputArray.Add(LumenRadianceCache::FUpdateInputs(
		RadianceCacheInputs,
		FRadianceCacheConfiguration(),
		View,
		nullptr,
		nullptr,
		MoveTemp(GraphicsMarkUsedRadianceCacheProbesCallbacks),
		MoveTemp(ComputeMarkUsedRadianceCacheProbesCallbacks)));

	FRadianceCacheState& RadianceCacheState = View.ViewState->Lumen.RadianceCacheState;

	OutputArray.Add(LumenRadianceCache::FUpdateOutputs(
		RadianceCacheState,
		RadianceCacheParameters));

	if (!bTranslucencyVolumeShareRadianceCacheWithOpaque)
	{
		LumenRadianceCache::FUpdateInputs TranslucencyVolumeRadianceCacheUpdateInputs = GetLumenTranslucencyGIVolumeRadianceCacheInputs(
			GraphBuilder,
			View,
			FrameTemporaries,
			ComputePassFlags);

		if (TranslucencyVolumeRadianceCacheUpdateInputs.IsAnyCallbackBound())
		{
			InputArray.Add(TranslucencyVolumeRadianceCacheUpdateInputs);
			OutputArray.Add(LumenRadianceCache::FUpdateOutputs(
				View.ViewState->Lumen.TranslucencyVolumeRadianceCacheState,
				TranslucencyVolumeRadianceCacheParameters));
		}
	}

	LumenRadianceCache::UpdateRadianceCaches(
		GraphBuilder, 
		FrameTemporaries,
		InputArray,
		OutputArray,
		Scene,
		ViewFamily,
		SceneTextures,
		LumenCardRenderer.bPropagateGlobalLightingChange,
		ComputePassFlags);

	if (bTranslucencyVolumeShareRadianceCacheWithOpaque)
	{
		TranslucencyVolumeRadianceCacheParameters = RadianceCacheParameters;
	}

	if (Lumen::UseLumenTranslucencyRadianceCacheReflections(ViewFamily))
	{
		View.GetOwnLumenTranslucencyGIVolume().RadianceCacheInterpolationParameters = RadianceCacheParameters;

		extern float GLumenTranslucencyReflectionsRadianceCacheReprojectionRadiusScale;
		extern float GLumenTranslucencyVolumeRadianceCacheClipmapFadeSize;
		View.GetOwnLumenTranslucencyGIVolume().RadianceCacheInterpolationParameters.RadianceCacheInputs.ReprojectionRadiusScale = GLumenTranslucencyReflectionsRadianceCacheReprojectionRadiusScale;
		View.GetOwnLumenTranslucencyGIVolume().RadianceCacheInterpolationParameters.RadianceCacheInputs.InvClipmapFadeSize = 1.0f / FMath::Clamp(GLumenTranslucencyVolumeRadianceCacheClipmapFadeSize, .001f, 16.0f);
	}

	if (LumenScreenProbeGather::UseShortRangeAmbientOcclusion(ViewFamily.EngineShowFlags, View.FinalPostProcessSettings.LumenAmbientOcclusionIntensity))
	{
		FVector2f MaxScreenTraceFraction = FVector2f(GLumenIrradianceFieldShortRangeAORadius * 2.0f) / FVector2f(View.ViewRect.Size());
		FBlueNoise BlueNoise = GetBlueNoiseGlobalParameters();
		const uint32 DownsampleFactor = LumenShortRangeAO::GetRequestedDownsampleFactor();
		ScreenSpaceBentNormalParameters = ComputeScreenSpaceShortRangeAO(GraphBuilder, Scene, View, FrameTemporaries, SceneTextures, LightingChannelsTexture, BlueNoise, MaxScreenTraceFraction, DownsampleFactor, 1.0f, ComputePassFlags);
	}

	FRDGTextureDesc DiffuseIndirectDesc = FRDGTextureDesc::Create2DArray(DownsampledBufferSize, PF_FloatRGB, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV, 1);
	FRDGTextureRef DiffuseIndirect = GraphBuilder.CreateTexture(DiffuseIndirectDesc, TEXT("r.Lumen.IrradianceFieldGather.DiffuseIndirect"));

	FRDGTextureDesc RoughSpecularIndirectDesc = FRDGTextureDesc::Create2DArray(DownsampledBufferSize, PF_FloatRGB, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV, 1);
	FRDGTextureRef RoughSpecularIndirect = GraphBuilder.CreateTexture(RoughSpecularIndirectDesc, TEXT("r.Lumen.IrradianceFieldGather.RoughSpecularIndirect"));

	const bool bAdaptivePlacement = RadianceCacheInputs.UseAdaptiveProbes != 0;
	FRDGBufferRef ProbeAdaptiveIndices = bAdaptivePlacement ? GraphBuilder.RegisterExternalBuffer(RadianceCacheState.ProbeAdaptiveIndices) : nullptr;
	FRDGBufferRef ProbeInterpolationMisses = bAdaptivePlacement ? GraphBuilder.RegisterExternalBuffer(RadianceCacheState.ProbeInterpolationMisses) : nullptr;
	FRDGBufferRef ProbeLastUsedFrame = bAdaptivePlacement ? GraphBuilder.RegisterExternalBuffer(RadianceCacheState.ProbeLastUsedFrame) : nullptr;
	FRDGBufferRef ProbeValid = GraphBuilder.RegisterExternalBuffer(RadianceCacheState.ProbeValid);

	{
		FInterpolateIrradianceProbesToPixelsCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FInterpolateIrradianceProbesToPixelsCS::FParameters>();
		PassParameters->RWDiffuseIndirect = GraphBuilder.CreateUAV(FRDGTextureUAVDesc(DiffuseIndirect));
		PassParameters->RWRoughSpecularIndirect = GraphBuilder.CreateUAV(FRDGTextureUAVDesc(RoughSpecularIndirect));
		PassParameters->RadianceCacheParameters = RadianceCacheParameters;
		PassParameters->View = View.ViewUniformBuffer;
		PassParameters->SceneTexturesStruct = SceneTextures.UniformBuffer;
		PassParameters->Substrate = Substrate::BindSubstrateGlobalUniformParameters(View);
		PassParameters->ReflectionStruct = CreateReflectionUniformBuffer(GraphBuilder, View);
		LumenReflections::SetupCompositeParameters(View, GetViewPipelineState(View).ReflectionsMethod, PassParameters->ReflectionsCompositeParameters);
		PassParameters->UpsampleParameters = UpsampleParameters;
		PassParameters->ProbeAdaptiveIndices = bAdaptivePlacement ? GraphBuilder.CreateSRV(FRDGBufferSRVDesc(ProbeAdaptiveIndices, PF_R32_UINT)) : nullptr;
		PassParameters->ProbeValid = LumenRadianceCache::GetProbeValidSRV(GraphBuilder, ProbeValid);
		PassParameters->RWProbeLastUsedFrame = bAdaptivePlacement ? GraphBuilder.CreateUAV(FRDGBufferUAVDesc(ProbeLastUsedFrame, PF_R32_UINT)) : nullptr;
		PassParameters->RWProbeInterpolationMisses = bAdaptivePlacement ? GraphBuilder.CreateUAV(FRDGBufferUAVDesc(ProbeInterpolationMisses, PF_R32_UINT)) : nullptr;
		const uint32 InterpolationFeedbackTileSize = FMath::RoundUpToPowerOfTwo(FMath::Clamp(GLumenIrradianceFieldAdaptivePlacementInterpolationFeedbackTileSize, 1, 256));
		PassParameters->InterpolationFeedbackTileWrapMask = InterpolationFeedbackTileSize - 1;
		PassParameters->InterpolationFeedbackTileJitter = GetFeedbackBufferTileJitter(InterpolationFeedbackTileSize, View.ViewState->GetFrameIndex());
		PassParameters->MaxNumProbes = MaxNumProbes;
		PassParameters->ProbeOcclusionBias = LumenIrradianceFieldGather::GetProbeOcclusionBias();
		PassParameters->MaxRoughnessToEvaluateRoughSpecular = LumenIrradianceFieldGather::GetMaxRoughnessToEvaluateRoughSpecular();
		PassParameters->SkylightPrefilterMipBias = FMath::Clamp<float>(CVarLumenSkylightPrefilterMipBias.GetValueOnRenderThread(), 0.0f, 16.0f);

		FInterpolateIrradianceProbesToPixelsCS::FPermutationDomain PermutationVector;
		PermutationVector.Set<FInterpolateIrradianceProbesToPixelsCS::FInterpolateDownsampleFactor>(InterpolateDownsampleFactor);
		PermutationVector.Set<FInterpolateIrradianceProbesToPixelsCS::FAdaptiveProbes>(bAdaptivePlacement);
		auto ComputeShader = View.ShaderMap->GetShader<FInterpolateIrradianceProbesToPixelsCS>(PermutationVector);

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("InterpolateIrradianceProbesToPixels DownsampleFactor=%u, %ux%u", InterpolateDownsampleFactor, UpsampleParameters.IntegrateViewSize.X, UpsampleParameters.IntegrateViewSize.Y),
			ComputePassFlags,
			ComputeShader,
			PassParameters,
			FComputeShaderUtils::GetGroupCount(UpsampleParameters.IntegrateViewSize, FInterpolateIrradianceProbesToPixelsCS::GetGroupSize()));
	}

	const FRDGSystemTextures& SystemTextures = FRDGSystemTextures::Get(GraphBuilder);

	FSSDSignalTextures DenoiserOutputs;
	DenoiserOutputs.Textures[0] = DiffuseIndirect;
	DenoiserOutputs.Textures[1] = SystemTextures.Black;
	DenoiserOutputs.Textures[2] = RoughSpecularIndirect;


	// Sample radiance caches for hair strands lighting.
	if (HairStrands::HasViewHairStrandsData(View))
	{
		RenderHairStrandsLumenLighting(GraphBuilder, Scene, View);
	}

	return DenoiserOutputs;
}

