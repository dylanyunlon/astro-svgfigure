// Copyright Epic Games, Inc. All Rights Reserved.

#include "LumenRadianceCache.h"
#include "DeferredShadingRenderer.h"
#include "LumenScreenProbeGather.h"
#include "LumenTranslucencyVolumeLighting.h"
#include "SceneViewState.h"

static TAutoConsoleVariable<int32> CVarLumenRadianceCacheVisualize(
	TEXT("r.Lumen.RadianceCache.Visualize"),
	0,
	TEXT("Whether to visualize radiance cache probes.\n")
	TEXT("0 - Disabled\n")
	TEXT("1 - Radiance\n")
	TEXT("2 - Sky Visibility"),
	ECVF_RenderThreadSafe
);

int32 GLumenVisualizeTranslucencyVolumeRadianceCache = 0;
FAutoConsoleVariableRef CVarLumenRadianceCacheVisualizeTranslucencyVolume(
	TEXT("r.Lumen.TranslucencyVolume.RadianceCache.Visualize"),
	GLumenVisualizeTranslucencyVolumeRadianceCache,
	TEXT(""),
	ECVF_RenderThreadSafe
);

float GLumenRadianceCacheVisualizeRadiusScale = 1.0f;
FAutoConsoleVariableRef CVarLumenRadianceCacheVisualizeRadiusScale(
	TEXT("r.Lumen.RadianceCache.VisualizeRadiusScale"),
	GLumenRadianceCacheVisualizeRadiusScale,
	TEXT("Scales the size of the spheres used to visualize radiance cache samples."),
	ECVF_RenderThreadSafe
);

int32 GLumenRadianceCacheVisualizeClipmapIndex = -1;
FAutoConsoleVariableRef CVarLumenRadianceCacheVisualizeClipmapIndex(
	TEXT("r.Lumen.RadianceCache.VisualizeClipmapIndex"),
	GLumenRadianceCacheVisualizeClipmapIndex,
	TEXT("Selects which radiance cache clipmap should be visualized. -1 visualizes all clipmaps at once."),
	ECVF_RenderThreadSafe
);

BEGIN_SHADER_PARAMETER_STRUCT(FVisualizeRadianceCacheCommonParameters, )
	SHADER_PARAMETER_STRUCT_INCLUDE(LumenRadianceCache::FRadianceCacheInterpolationParameters, RadianceCacheParameters)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FReflectionUniformParameters, ReflectionStruct)
	SHADER_PARAMETER(FVector4f, ClipmapCornerTWSAndCellSizeForVisualization)
	SHADER_PARAMETER(float, VisualizeProbeRadiusScale)
	SHADER_PARAMETER(uint32, ProbeClipmapIndex)
END_SHADER_PARAMETER_STRUCT()

class FClearProbeVisualizeBufferCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FClearProbeVisualizeBufferCS)
	SHADER_USE_PARAMETER_STRUCT(FClearProbeVisualizeBufferCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, RWProbeVisualizeBufferIndirectArguments)
	END_SHADER_PARAMETER_STRUCT()

public:

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportLumenGI(Parameters.Platform);
	}

	static uint32 GetGroupSize()
	{
		return 64;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}
};

IMPLEMENT_GLOBAL_SHADER(FClearProbeVisualizeBufferCS, "/Engine/Private/Lumen/LumenVisualizeRadianceCache.usf", "ClearProbeVisualizeBufferCS", SF_Compute);

class FBuildProbeVisualizeBufferCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FBuildProbeVisualizeBufferCS)
	SHADER_USE_PARAMETER_STRUCT(FBuildProbeVisualizeBufferCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, RWProbeVisualizeBufferIndirectArguments)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint4>, RWProbeVisualizeBuffer)
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_STRUCT_INCLUDE(FVisualizeRadianceCacheCommonParameters, VisualizeCommonParameters)
	END_SHADER_PARAMETER_STRUCT()

	class FAdaptiveProbes : SHADER_PERMUTATION_BOOL("ADAPTIVE_PROBES");
	using FPermutationDomain = TShaderPermutationDomain<FAdaptiveProbes>;

public:

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportLumenGI(Parameters.Platform);
	}

	static uint32 GetGroupSize()
	{
		return 8;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}
};

IMPLEMENT_GLOBAL_SHADER(FBuildProbeVisualizeBufferCS, "/Engine/Private/Lumen/LumenVisualizeRadianceCache.usf", "BuildProbeVisualizeBufferCS", SF_Compute);

class FVisualizeRadianceCacheVS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FVisualizeRadianceCacheVS);
	SHADER_USE_PARAMETER_STRUCT(FVisualizeRadianceCacheVS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_STRUCT_INCLUDE(FVisualizeRadianceCacheCommonParameters, VisualizeCommonParameters)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint4>, ProbeVisualizeBuffer)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, LastFrameProbeInterpolationMisses)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, ProbeAdaptiveIndices)
		SHADER_PARAMETER(uint32, MaxNumProbes)
		RDG_BUFFER_ACCESS(IndirectDrawParameter, ERHIAccess::IndirectArgs)
	END_SHADER_PARAMETER_STRUCT()

	class FAdaptiveProbes : SHADER_PERMUTATION_BOOL("ADAPTIVE_PROBES");
	using FPermutationDomain = TShaderPermutationDomain<FAdaptiveProbes>;

public:
	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportLumenGI(Parameters.Platform);
	}
};

IMPLEMENT_GLOBAL_SHADER(FVisualizeRadianceCacheVS,"/Engine/Private/Lumen/LumenVisualizeRadianceCache.usf", "VisualizeRadianceCacheVS", SF_Vertex);

class FVisualizeRadianceCachePS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FVisualizeRadianceCachePS);
	SHADER_USE_PARAMETER_STRUCT(FVisualizeRadianceCachePS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_STRUCT_INCLUDE(FVisualizeRadianceCacheCommonParameters, VisualizeCommonParameters)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, ProbeAdaptiveIndices)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, ProbeValid)
	END_SHADER_PARAMETER_STRUCT()

public:

	class FVisualizeMode : SHADER_PERMUTATION_RANGE_INT("VISUALIZE_MODE", 1, 2);
	class FRadianceCacheIrradiance : SHADER_PERMUTATION_BOOL("RADIANCE_CACHE_IRRADIANCE");
	class FRadianceCacheSkyVisibility : SHADER_PERMUTATION_BOOL("RADIANCE_CACHE_SKY_VISIBILITY");
	class FAdaptiveProbes : SHADER_PERMUTATION_BOOL("ADAPTIVE_PROBES");
	using FPermutationDomain = TShaderPermutationDomain<FVisualizeMode, FRadianceCacheIrradiance, FRadianceCacheSkyVisibility, FAdaptiveProbes>;
	
	static FPermutationDomain RemapPermutation(FPermutationDomain PermutationVector)
	{
		if (PermutationVector.Get<FVisualizeMode>() != 1)
		{
			PermutationVector.Set<FRadianceCacheIrradiance>(false);
		}

		if (!PermutationVector.Get<FRadianceCacheIrradiance>())
		{
			PermutationVector.Set<FAdaptiveProbes>(false);
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
};

IMPLEMENT_GLOBAL_SHADER(FVisualizeRadianceCachePS, "/Engine/Private/Lumen/LumenVisualizeRadianceCache.usf", "VisualizeRadianceCachePS", SF_Pixel);

BEGIN_SHADER_PARAMETER_STRUCT(FVisualizeRadianceCacheParameters, )
	SHADER_PARAMETER_STRUCT_INCLUDE(FVisualizeRadianceCacheVS::FParameters, VS)
	SHADER_PARAMETER_STRUCT_INCLUDE(FVisualizeRadianceCachePS::FParameters, PS)
	RENDER_TARGET_BINDING_SLOTS()
END_SHADER_PARAMETER_STRUCT()

LumenRadianceCache::FRadianceCacheInputs GetFinalGatherRadianceCacheInputs(const FViewInfo& View)
{
	if (GLumenVisualizeTranslucencyVolumeRadianceCache)
	{
		return LumenTranslucencyVolumeRadianceCache::SetupRadianceCacheInputs(View);
	}
	else
	{
		if (Lumen::UseIrradianceFieldGather())
		{
			return LumenIrradianceFieldGather::SetupRadianceCacheInputs(View);
		}
		else
		{
			return LumenScreenProbeGatherRadianceCache::SetupRadianceCacheInputs(View);
		}
	}
}

extern int32 GLumenVisualizeTranslucencyVolumeRadianceCache;

void FDeferredShadingSceneRenderer::RenderLumenRadianceCacheVisualization(FRDGBuilder& GraphBuilder, const FMinimalSceneTextures& SceneTextures)
{
	for (const FViewInfo& View : Views)
	{
		const FPerViewPipelineState& ViewPipelineState = GetViewPipelineState(View);
		const bool bAnyLumenActive = ViewPipelineState.DiffuseIndirectMethod == EDiffuseIndirectMethod::Lumen;

		if (View.ViewState
			&& bAnyLumenActive
			&& (LumenScreenProbeGather::UseRadianceCache()
				|| (GLumenVisualizeTranslucencyVolumeRadianceCache && CVarLumenTranslucencyVolume.GetValueOnRenderThread())
				|| Lumen::UseIrradianceFieldGather())
			&& CVarLumenRadianceCacheVisualize.GetValueOnRenderThread() != 0)
		{
			RDG_EVENT_SCOPE(GraphBuilder, "VisualizeLumenRadianceCache");

			const FRadianceCacheState& RadianceCacheState = GLumenVisualizeTranslucencyVolumeRadianceCache != 0 ? View.ViewState->Lumen.TranslucencyVolumeRadianceCacheState : View.ViewState->Lumen.RadianceCacheState;

			FRDGTextureRef SceneColor = SceneTextures.Color.Resolve;
			FRDGTextureRef SceneDepth = SceneTextures.Depth.Resolve;

			const LumenRadianceCache::FRadianceCacheInputs RadianceCacheInputs = GetFinalGatherRadianceCacheInputs(View);

			const int32 VisualizationClipmapIndex = FMath::Clamp(GLumenRadianceCacheVisualizeClipmapIndex, -1, RadianceCacheState.Clipmaps.Num() - 1);
			for (int32 ClipmapIndex = 0; ClipmapIndex < RadianceCacheState.Clipmaps.Num(); ++ClipmapIndex)
			{
				if (VisualizationClipmapIndex != -1 && VisualizationClipmapIndex != ClipmapIndex)
				{
					continue;
				}

				const FRadianceCacheClipmap& Clipmap = RadianceCacheState.Clipmaps[ClipmapIndex];

				FVisualizeRadianceCacheCommonParameters VisualizeCommonParameters;
				LumenRadianceCache::GetInterpolationParameters(View, GraphBuilder, RadianceCacheState, RadianceCacheInputs, VisualizeCommonParameters.RadianceCacheParameters);
				VisualizeCommonParameters.VisualizeProbeRadiusScale = GLumenRadianceCacheVisualizeRadiusScale * 0.05f;
				VisualizeCommonParameters.ProbeClipmapIndex = ClipmapIndex;
				VisualizeCommonParameters.ClipmapCornerTWSAndCellSizeForVisualization = FVector4f(Clipmap.CornerTranslatedWorldSpace, Clipmap.CellSize);
				VisualizeCommonParameters.ReflectionStruct = CreateReflectionUniformBuffer(GraphBuilder, View);

				const int32 MaxNumProbes = RadianceCacheInputs.ProbeAtlasResolutionInProbes.X * RadianceCacheInputs.ProbeAtlasResolutionInProbes.Y;
				FRDGBufferRef ProbeVisualizeBufferIndirectArguments = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateIndirectDesc<FRHIDrawIndexedIndirectParameters>(1), TEXT("Lumen.RadianceCache.Visualize.ProbeVisualizeBufferIndirectArguments"));
				FRDGBufferRef ProbeVisualizeBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateBufferDesc(sizeof(FUintVector4), MaxNumProbes), TEXT("Lumen.RadianceCache.Visualize.ProbeVisualizeBuffer"));
				FRDGBufferUAVRef ProbeVisualizeBufferIndirectArgumentsUAV = GraphBuilder.CreateUAV(FRDGBufferUAVDesc(ProbeVisualizeBufferIndirectArguments));
				FRDGBufferUAVRef ProbeVisualizeBufferUAV = GraphBuilder.CreateUAV(FRDGBufferUAVDesc(ProbeVisualizeBuffer, PF_R32G32B32A32_UINT));

				{
					FClearProbeVisualizeBufferCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FClearProbeVisualizeBufferCS::FParameters>();
					PassParameters->RWProbeVisualizeBufferIndirectArguments = ProbeVisualizeBufferIndirectArgumentsUAV;

					auto ComputeShader = View.ShaderMap->GetShader<FClearProbeVisualizeBufferCS>();
					const FIntVector GroupSize = FComputeShaderUtils::GetGroupCount(1, FClearProbeVisualizeBufferCS::GetGroupSize());

					FComputeShaderUtils::AddPass(
						GraphBuilder,
						RDG_EVENT_NAME("ClearProbeVisualizeBuffer"),
						ERDGPassFlags::Compute,
						ComputeShader,
						PassParameters,
						GroupSize);
				}

				{
					FBuildProbeVisualizeBufferCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FBuildProbeVisualizeBufferCS::FParameters>();
					PassParameters->RWProbeVisualizeBufferIndirectArguments = ProbeVisualizeBufferIndirectArgumentsUAV;
					PassParameters->RWProbeVisualizeBuffer = ProbeVisualizeBufferUAV;
					PassParameters->View = GetShaderBinding(View.ViewUniformBuffer);
					PassParameters->VisualizeCommonParameters = VisualizeCommonParameters;

					FBuildProbeVisualizeBufferCS::FPermutationDomain PermutationVector;
					PermutationVector.Set<FBuildProbeVisualizeBufferCS::FAdaptiveProbes>(RadianceCacheInputs.UseAdaptiveProbes != 0);
					auto ComputeShader = View.ShaderMap->GetShader<FBuildProbeVisualizeBufferCS>(PermutationVector);

					const FIntVector GroupSize = FComputeShaderUtils::GetGroupCount(FIntVector(RadianceCacheInputs.RadianceProbeClipmapResolution), FBuildProbeVisualizeBufferCS::GetGroupSize());

					FComputeShaderUtils::AddPass(
						GraphBuilder,
						RDG_EVENT_NAME("BuildProbeVisualizeBuffer"),
						ERDGPassFlags::Compute,
						ComputeShader,
						PassParameters,
						GroupSize);
				}

				const bool bAdaptiveProbes = RadianceCacheInputs.UseAdaptiveProbes != 0;
				const bool bIrradiance = RadianceCacheInputs.CalculateIrradiance != 0;
				const bool bSkyVisibility = RadianceCacheInputs.UseSkyVisibility != 0;
				const int32 VisualizationMode = FMath::Clamp(CVarLumenRadianceCacheVisualize.GetValueOnRenderThread(), 1, 2);
				FRDGBufferRef ProbeValid = bIrradiance ? GraphBuilder.RegisterExternalBuffer(RadianceCacheState.ProbeValid) : nullptr;

				FVisualizeRadianceCacheParameters* PassParameters = GraphBuilder.AllocParameters<FVisualizeRadianceCacheParameters>();
				PassParameters->VS.VisualizeCommonParameters = VisualizeCommonParameters;
				PassParameters->VS.View = GetShaderBinding(View.ViewUniformBuffer);
				PassParameters->VS.IndirectDrawParameter = ProbeVisualizeBufferIndirectArguments;
				PassParameters->VS.ProbeVisualizeBuffer = GraphBuilder.CreateSRV(ProbeVisualizeBuffer, PF_R32G32B32A32_UINT);
				PassParameters->VS.LastFrameProbeInterpolationMisses = bAdaptiveProbes ? GraphBuilder.CreateSRV(FRDGBufferSRVDesc(GraphBuilder.RegisterExternalBuffer(RadianceCacheState.ProbeInterpolationMisses), PF_R32_UINT)) : nullptr;
				PassParameters->VS.MaxNumProbes = MaxNumProbes;
				PassParameters->VS.ProbeAdaptiveIndices = bAdaptiveProbes ? GraphBuilder.CreateSRV(FRDGBufferSRVDesc(GraphBuilder.RegisterExternalBuffer(RadianceCacheState.ProbeAdaptiveIndices), PF_R32_UINT)) : nullptr;

				PassParameters->PS.VisualizeCommonParameters = VisualizeCommonParameters;
				PassParameters->PS.View = GetShaderBinding(View.ViewUniformBuffer);
				PassParameters->PS.ProbeAdaptiveIndices = bAdaptiveProbes ? GraphBuilder.CreateSRV(FRDGBufferSRVDesc(GraphBuilder.RegisterExternalBuffer(RadianceCacheState.ProbeAdaptiveIndices), PF_R32_UINT)) : nullptr;
				PassParameters->PS.ProbeValid = bIrradiance ? LumenRadianceCache::GetProbeValidSRV(GraphBuilder, ProbeValid) : nullptr;

				FVisualizeRadianceCacheVS::FPermutationDomain PermutationVectorVS;
				PermutationVectorVS.Set<FVisualizeRadianceCacheVS::FAdaptiveProbes>(bAdaptiveProbes);
				auto VertexShader = View.ShaderMap->GetShader<FVisualizeRadianceCacheVS>(PermutationVectorVS);

				FVisualizeRadianceCachePS::FPermutationDomain PermutationVector;
				PermutationVector.Set<FVisualizeRadianceCachePS::FVisualizeMode>(VisualizationMode);
				PermutationVector.Set<FVisualizeRadianceCachePS::FRadianceCacheIrradiance>(bIrradiance);
				PermutationVector.Set<FVisualizeRadianceCachePS::FRadianceCacheSkyVisibility>(bSkyVisibility);
				PermutationVector.Set<FVisualizeRadianceCachePS::FAdaptiveProbes>(bAdaptiveProbes);
				PermutationVector = FVisualizeRadianceCachePS::RemapPermutation(PermutationVector);
				auto PixelShader = View.ShaderMap->GetShader<FVisualizeRadianceCachePS>(PermutationVector);

				PassParameters->RenderTargets.DepthStencil = FDepthStencilBinding(
					SceneDepth,
					ERenderTargetLoadAction::ENoAction,
					ERenderTargetLoadAction::ELoad,
					FExclusiveDepthStencil::DepthWrite_StencilWrite);
				PassParameters->RenderTargets[0] = FRenderTargetBinding(SceneColor, ERenderTargetLoadAction::ELoad);

				ClearUnusedGraphResources(VertexShader, &PassParameters->VS);
				ClearUnusedGraphResources(PixelShader, &PassParameters->PS);

				GraphBuilder.AddPass(
					RDG_EVENT_NAME("Visualize Radiance Cache Clipmap:%d", ClipmapIndex),
					PassParameters,
					ERDGPassFlags::Raster,
					[PassParameters, &View, VertexShader, PixelShader](FRDGAsyncTask, FRHICommandList& RHICmdList)
					{
						RHICmdList.SetViewport(View.ViewRect.Min.X, View.ViewRect.Min.Y, 0.0f, View.ViewRect.Max.X, View.ViewRect.Max.Y, 1.0f);

						FGraphicsPipelineStateInitializer GraphicsPSOInit;
						RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);
						GraphicsPSOInit.BlendState = TStaticBlendState<CW_RGB>::GetRHI();
						GraphicsPSOInit.RasterizerState = TStaticRasterizerState<FM_Solid, CM_None>::GetRHI();
						GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<true, CF_DepthNear>::GetRHI();
						GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI = GEmptyVertexDeclaration.VertexDeclarationRHI;
						GraphicsPSOInit.BoundShaderState.VertexShaderRHI = VertexShader.GetVertexShader();
						GraphicsPSOInit.BoundShaderState.PixelShaderRHI = PixelShader.GetPixelShader();
						GraphicsPSOInit.PrimitiveType = PT_TriangleList;

						SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit, 0);

						SetShaderParameters(RHICmdList, VertexShader, VertexShader.GetVertexShader(), PassParameters->VS);
						SetShaderParameters(RHICmdList, PixelShader, PixelShader.GetPixelShader(), PassParameters->PS);

						RHICmdList.SetStreamSource(0, NULL, 0);

						// Marks the indirect draw parameter as used by the pass, given it's not used directly by any of the shaders.
						//PassParameters->VS.IndirectDrawParameter->MarkResourceAsUsed();

						RHICmdList.DrawIndexedPrimitiveIndirect(
							GCubeIndexBuffer.IndexBufferRHI,
							PassParameters->VS.IndirectDrawParameter->GetIndirectRHICallBuffer(),
							0);
					});
			}
		}
	}
}
