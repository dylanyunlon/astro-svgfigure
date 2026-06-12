// Copyright Epic Games, Inc. All Rights Reserved.

#include "PostProcess/PostProcessVisualizeShadowCasters.h"
#include "DataDrivenShaderPlatformInfo.h"
#include "GlobalShader.h"
#include "Rendering/NaniteStreamingManager.h"
#include "RenderingVisualizationUtils.h"
#include "SceneRendering.h"
#include "SceneUniformBuffer.h"
#include "VirtualShadowMaps/VirtualShadowMapArray.h"
#include "VisualizationData/VisualizationDataShared.h"

namespace
{
	enum class EShadowCastersMode : uint8
	{
		Far = 0,
		Cached = 1,
		MAX,
	};

	// Renders the DVSM_FarShadowCasters legend
	static void RenderFarShadowCastersLegend(FRDGBuilder& GraphBuilder, const FViewInfo& View, FScreenPassRenderTarget OutputTarget, const FIntRect& OutputRect)
	{
		const FVector2f LegendAnchorPosition(OutputRect.Min.X + 8, OutputRect.Max.Y - 100);
		const FVector2f LegendSize(100.0f, 30.0f);
		const FString HeaderLabel(TEXT("Far Shadow Casters"));

		TArray<FVisualizationDataLegendEntry, SceneRenderingAllocator> LegendEntries
		{
			FVisualizationDataLegendEntry{ NSLOCTEXT("RenderingVisualizationUtils", "FarShadowCastersCasts",  "Casts Far Shadow"),         FLinearColor(0.0f, 0.7f, 0.0f)    },
			FVisualizationDataLegendEntry{ NSLOCTEXT("RenderingVisualizationUtils", "FarShadowCastersNoCast", "Does Not Cast Far Shadow"), FLinearColor(0.35f, 0.35f, 0.35f) },
		};
		AddLegendCanvasPass(GraphBuilder, View, OutputTarget, HeaderLabel, LegendAnchorPosition, LegendSize, LegendEntries);
	}

	// Renders the DVSM_ShadowCasters legend
	static void RenderShadowCastersLegend(FRDGBuilder& GraphBuilder, const FViewInfo& View, FScreenPassRenderTarget OutputTarget, const FIntRect& OutputRect)
	{
		const FVector2f LegendAnchorPosition(OutputRect.Min.X + 8, OutputRect.Max.Y - 100);
		const FVector2f LegendSize(100.0f, 30.0f);
		const FString HeaderLabel(TEXT("Shadow Casters"));

		TArray<FVisualizationDataLegendEntry, SceneRenderingAllocator> LegendEntries
		{
			FVisualizationDataLegendEntry{ NSLOCTEXT("RenderingVisualizationUtils", "ShadowCastersStatic",              "Static, Not Invalidating"),    FLinearColor(0.0f, 1.0f, 0.0f) },
			FVisualizationDataLegendEntry{ NSLOCTEXT("RenderingVisualizationUtils", "ShadowCastersStaticInvalidating",  "Static, Invalidating"),        FLinearColor(1.0f, 0.0f, 0.0f) },
			FVisualizationDataLegendEntry{ NSLOCTEXT("RenderingVisualizationUtils", "ShadowCastersDynamic",             "Dynamic, Not Invalidating"),   FLinearColor(0.0f, 1.0f, 1.0f) },
			FVisualizationDataLegendEntry{ NSLOCTEXT("RenderingVisualizationUtils", "ShadowCastersDynamicInvalidating", "Dynamic, Invalidating"),       FLinearColor(0.5f, 0.0f, 0.5f) },
			FVisualizationDataLegendEntry{ NSLOCTEXT("RenderingVisualizationUtils", "ShadowCastersDynamicMaybe",        "Dynamic, Maybe Invalidating"), FLinearColor(0.0f, 0.0f, 1.0f) },
			FVisualizationDataLegendEntry{ NSLOCTEXT("RenderingVisualizationUtils", "ShadowCastersContact",             "Contact Shadow"),              FLinearColor(1.0f, 1.0f, 0.0f) },
		};
		AddLegendCanvasPass(GraphBuilder, View, OutputTarget, HeaderLabel, LegendAnchorPosition, LegendSize, LegendEntries);
	}

	// Renders a simplified version of DVSM_ShadowCasters legend
	static void RenderShadowCastersSimplifiedLegend(FRDGBuilder& GraphBuilder, const FViewInfo& View, FScreenPassRenderTarget OutputTarget, const FIntRect& OutputRect)
	{
		const FVector2f LegendAnchorPosition(OutputRect.Min.X + 8, OutputRect.Max.Y - 100);
		const FVector2f LegendSize(100.0f, 30.0f);
		const FString HeaderLabel(TEXT("Shadow Casters (VSM disabled - degraded)"));

		TArray<FVisualizationDataLegendEntry, SceneRenderingAllocator> LegendEntries
		{
			FVisualizationDataLegendEntry{ NSLOCTEXT("RenderingVisualizationUtils", "ShadowCastersDegradedCasts",   "Casts Shadow"),       FLinearColor(0.0f, 1.0f, 0.0f) },
			FVisualizationDataLegendEntry{ NSLOCTEXT("RenderingVisualizationUtils", "ShadowCastersDegradedContact", "Contact Shadow Only"), FLinearColor(1.0f, 1.0f, 0.0f) },
			FVisualizationDataLegendEntry{ NSLOCTEXT("RenderingVisualizationUtils", "ShadowCastersDegradedNone",    "No Shadow"),          FLinearColor(0.5f, 0.5f, 0.5f) },
		};
		AddLegendCanvasPass(GraphBuilder, View, OutputTarget, HeaderLabel, LegendAnchorPosition, LegendSize, LegendEntries);
	}
}

class FVisualizeShadowCastersPS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FVisualizeShadowCastersPS);
	SHADER_USE_PARAMETER_STRUCT(FVisualizeShadowCastersPS, FGlobalShader);

	class FModeDim                : SHADER_PERMUTATION_ENUM_CLASS("SHADOW_CASTERS_MODE", EShadowCastersMode);
	class FUseNaniteVisBufferDim  : SHADER_PERMUTATION_BOOL("USE_NANITE_VISBUFFER");
	class FUseVsmRichClassifierDim: SHADER_PERMUTATION_BOOL("USE_VSM_RICH_CLASSIFIER");
	using FPermutationDomain = TShaderPermutationDomain<FModeDim, FUseNaniteVisBufferDim, FUseVsmRichClassifierDim>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FViewShaderParameters, View)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneUniformParameters, Scene)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FVirtualShadowMapUniformParameters, VirtualShadowMap)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<uint>, DebugAuxTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, SceneColorTexture)
		SHADER_PARAMETER_SAMPLER(SamplerState, SceneColorSampler)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<UlongType>, VisBuffer64)
		SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, VisibleClustersSWHW)
		SHADER_PARAMETER(FIntVector4, PageConstants) // Required by the Nanite cluster decode (NaniteDataDecode.ush) on the USE_NANITE_VISBUFFER path.
		SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, ClusterPageData)
		RENDER_TARGET_BINDING_SLOTS()
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		const FPermutationDomain PermutationVector(Parameters.PermutationId);

		if (PermutationVector.Get<FUseNaniteVisBufferDim>() && !DoesPlatformSupportNanite(Parameters.Platform))
		{
			return false;
		}

		// DVSM_ShadowCasters never runs on  on mobile so skip the permutation
		if (PermutationVector.Get<FModeDim>() == EShadowCastersMode::Cached	&& IsMobilePlatform(Parameters.Platform))
		{
			return false;
		}

		if (PermutationVector.Get<FUseVsmRichClassifierDim>())
		{
			if (PermutationVector.Get<FModeDim>() != EShadowCastersMode::Cached)
			{
				return false;
			}
			if (!DoesPlatformSupportVirtualShadowMaps(Parameters.Platform))
			{
				return false;
			}
		}

		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("VF_SUPPORTS_PRIMITIVE_SCENE_DATA"), 1);

		const FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (PermutationVector.Get<FUseVsmRichClassifierDim>())
		{
			// Required for FNaniteView / FInstanceViewData.
			OutEnvironment.SetDefine(TEXT("NANITE_USE_VIEW_UNIFORM_BUFFER"), 1);
		}
	}
};

IMPLEMENT_GLOBAL_SHADER(FVisualizeShadowCastersPS, "/Engine/Private/PostProcessVisualizeShadowCasters.usf", "MainPS", SF_Pixel);

static FScreenPassTexture AddVisualizeShadowCastersPassInternal(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	EShadowCastersMode Mode,
	FScreenPassRenderTarget OverrideOutput,
	FScreenPassTexture SceneColor,
	FRDGTextureRef DebugAuxTexture,
	FRDGTextureRef NaniteVisBuffer64,
	FRDGBufferRef NaniteVisibleClustersSWHW,
	const FIntVector4& NanitePageConstants,
	TRDGUniformBufferRef<FVirtualShadowMapUniformParameters> VirtualShadowMap,
	const TCHAR* OutputTextureName,
	const TCHAR* PassEventName)
{
	check(SceneColor.IsValid());
	check(DebugAuxTexture);

	FScreenPassRenderTarget Output = OverrideOutput;
	if (!Output.IsValid())
	{
		Output = FScreenPassRenderTarget::CreateFromInput(GraphBuilder, SceneColor, View.GetOverwriteLoadAction(), OutputTextureName);
	}

	const FScreenPassTextureViewport InputViewport(SceneColor);
	const FScreenPassTextureViewport OutputViewport(Output);

	const bool bUseNaniteVisBuffer = NaniteVisBuffer64 != nullptr && NaniteVisibleClustersSWHW != nullptr && DoesPlatformSupportNanite(View.GetShaderPlatform());

	const bool bUseVsmRichClassifier = Mode == EShadowCastersMode::Cached && VirtualShadowMap != nullptr && DoesPlatformSupportVirtualShadowMaps(View.GetShaderPlatform());

	FRDGTextureRef VisBufferBinding = NaniteVisBuffer64;
	if (!bUseNaniteVisBuffer)
	{
		FRDGTextureDesc DummyDesc = FRDGTextureDesc::Create2D(FIntPoint(1, 1), PF_R32G32_UINT, FClearValueBinding::Black, TexCreate_ShaderResource);
		VisBufferBinding = GraphBuilder.CreateTexture(DummyDesc, TEXT("VisualizeShadowCasters.VisBuffer64Dummy"));
	}

	FVisualizeShadowCastersPS::FParameters* PassParameters = GraphBuilder.AllocParameters<FVisualizeShadowCastersPS::FParameters>();
	PassParameters->View = View.GetShaderParameters();
	PassParameters->Scene = View.GetSceneUniforms().GetBuffer(GraphBuilder);
	PassParameters->VirtualShadowMap = VirtualShadowMap; // null when the rich-classifier permutation is not used; shader does not reference it
	PassParameters->DebugAuxTexture = DebugAuxTexture;
	PassParameters->SceneColorTexture = SceneColor.Texture;
	PassParameters->SceneColorSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	PassParameters->VisBuffer64 = VisBufferBinding;
	PassParameters->VisibleClustersSWHW = GraphBuilder.CreateSRV(bUseNaniteVisBuffer ? NaniteVisibleClustersSWHW : GSystemTextures.GetDefaultByteAddressBuffer(GraphBuilder, 4u));
	PassParameters->PageConstants = bUseNaniteVisBuffer ? NanitePageConstants : FIntVector4(ForceInitToZero);
	PassParameters->ClusterPageData = bUseNaniteVisBuffer
		? Nanite::GStreamingManager.GetClusterPageDataSRV(GraphBuilder)
		: GraphBuilder.CreateSRV(GSystemTextures.GetDefaultByteAddressBuffer(GraphBuilder, 4u));
	PassParameters->RenderTargets[0] = Output.GetRenderTargetBinding();

	FVisualizeShadowCastersPS::FPermutationDomain PermutationVector;
	PermutationVector.Set<FVisualizeShadowCastersPS::FModeDim>(Mode);
	PermutationVector.Set<FVisualizeShadowCastersPS::FUseNaniteVisBufferDim>(bUseNaniteVisBuffer);
	PermutationVector.Set<FVisualizeShadowCastersPS::FUseVsmRichClassifierDim>(bUseVsmRichClassifier);
	TShaderMapRef<FVisualizeShadowCastersPS> PixelShader(View.ShaderMap, PermutationVector);

	AddDrawScreenPass(GraphBuilder, RDG_EVENT_NAME("%s", PassEventName), View, OutputViewport, InputViewport, PixelShader, PassParameters);

	Output.LoadAction = ERenderTargetLoadAction::ELoad;
	if (Mode == EShadowCastersMode::Far)
	{
		RenderFarShadowCastersLegend(GraphBuilder, View, Output, OutputViewport.Rect);
	}
	else if (bUseVsmRichClassifier)
	{
		RenderShadowCastersLegend(GraphBuilder, View, Output, OutputViewport.Rect);
	}
	else
	{
		// When VSM data unavailable fall back to GetCachedShadowCasterColorSimplified()
		RenderShadowCastersSimplifiedLegend(GraphBuilder, View, Output, OutputViewport.Rect);
	}

	return MoveTemp(Output);
}

FScreenPassTexture AddVisualizeShadowCastersPass(FRDGBuilder& GraphBuilder, const FViewInfo& View, const FVisualizeShadowCastersInputs& Inputs)
{
	const EShadowCastersMode Mode = Inputs.bFarShadowCasters ? EShadowCastersMode::Far : EShadowCastersMode::Cached;
	const TCHAR* PassName = Inputs.bFarShadowCasters ? TEXT("VisualizeFarShadowCasters") : TEXT("VisualizeShadowCasters");

	return AddVisualizeShadowCastersPassInternal(
		GraphBuilder,
		View,
		Mode,
		Inputs.OverrideOutput,
		Inputs.SceneColor,
		Inputs.DebugAuxTexture,
		Inputs.NaniteVisBuffer64,
		Inputs.NaniteVisibleClustersSWHW,
		Inputs.NanitePageConstants,
		Inputs.VirtualShadowMap,
		PassName,
		PassName);
}
