// Copyright Epic Games, Inc. All Rights Reserved.

#include "NaniteEditor.h"

#include "Algo/Sort.h"
#include "Async/ParallelFor.h"
#include "RHI.h"
#include "SceneUtils.h"
#include "ScenePrivate.h"
#include "Rendering/NaniteStreamingManager.h"
#include "PixelShaderUtils.h"
#include "ScreenPass.h"
#include "SystemTextures.h"
#include "InstanceDataSceneProxy.h"
#include "Nanite/NaniteMaterialsSceneExtension.h"

DEFINE_GPU_STAT(NaniteEditor);

#if WITH_EDITOR
static TAutoConsoleVariable<bool> CVarNaniteEditorDrawListsAsyncUpdates(
	TEXT("r.Nanite.Editor.DrawLists.AsyncUpdates"),
	true,
	TEXT("When true, the Nanite editor scene extension builds its draw lists on an async RDG setup task and accumulates primitives in parallel. Set to false to run inline on the render thread with a serial primitive iteration for debugging."),
	ECVF_RenderThreadSafe
);
#endif

BEGIN_SHADER_PARAMETER_STRUCT(FNaniteSelectionOutlineParameters, )
	SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneUniformParameters, Scene)
	SHADER_PARAMETER(FVector2f, OutputToInputScale)
	SHADER_PARAMETER(FVector2f, OutputToInputBias)
	SHADER_PARAMETER(uint32, MaxVisibleClusters)

	SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FVisibleCluster>, VisibleClustersSWHW)
	SHADER_PARAMETER(FIntVector4, PageConstants)
	SHADER_PARAMETER(FIntVector4, ViewRect)
	SHADER_PARAMETER_RDG_BUFFER_SRV( ByteAddressBuffer, ClusterPageData )
	SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, HierarchyBuffer)

	SHADER_PARAMETER_RDG_TEXTURE(Texture2D<uint2>, VisBuffer64)

	SHADER_PARAMETER_RDG_TEXTURE(Texture2D<UlongType>, SceneVisBuffer64)
	SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FVisibleCluster>, SceneVisibleClustersSWHW)
	SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, SceneDepthTexture)
	SHADER_PARAMETER(FVector2f, SceneOutputToInputScale)
	SHADER_PARAMETER(FVector2f, SceneOutputToInputBias)
	SHADER_PARAMETER(FVector2f, SceneDepthOutputToInputScale)
	SHADER_PARAMETER(FVector2f, SceneDepthOutputToInputBias)

	SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, MaterialHitProxyTable)
	SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, EditorSelectedHitProxyIds)
	SHADER_PARAMETER(uint32, NumEditorSelectedHitProxyIds)
	RENDER_TARGET_BINDING_SLOTS()
END_SHADER_PARAMETER_STRUCT()

class FEmitHitProxyIdPS : public FNaniteGlobalShader
{
	DECLARE_GLOBAL_SHADER(FEmitHitProxyIdPS);
	SHADER_USE_PARAMETER_STRUCT(FEmitHitProxyIdPS, FNaniteGlobalShader);

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportNanite(Parameters.Platform);
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneUniformParameters, Scene)

		SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, VisibleClustersSWHW)
		SHADER_PARAMETER(FIntVector4, PageConstants)
		SHADER_PARAMETER(FIntVector4, ViewRect)
		SHADER_PARAMETER_RDG_BUFFER_SRV( ByteAddressBuffer, ClusterPageData )
		SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, HierarchyBuffer)

		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<UlongType>, VisBuffer64)

		SHADER_PARAMETER_RDG_BUFFER_SRV(ByteAddressBuffer, MaterialHitProxyTable)
	
		RENDER_TARGET_BINDING_SLOTS()
	END_SHADER_PARAMETER_STRUCT()
};
IMPLEMENT_GLOBAL_SHADER(FEmitHitProxyIdPS, "/Engine/Private/Nanite/NaniteExportGBuffer.usf", "EmitHitProxyIdPS", SF_Pixel);

// Selection-outline pass. ONLY_SELECTED restricts to selected instances.
class FEmitEditorSelectionDepthPS : public FNaniteGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FEmitEditorSelectionDepthPS);
	SHADER_USE_PARAMETER_STRUCT(FEmitEditorSelectionDepthPS, FNaniteGlobalShader);

	using FParameters = FNaniteSelectionOutlineParameters;

	class FOnlySelectedDim : SHADER_PERMUTATION_BOOL("ONLY_SELECTED");
	using FPermutationDomain = TShaderPermutationDomain<FOnlySelectedDim>;
	
	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportNanite(Parameters.Platform);
	}
};
IMPLEMENT_GLOBAL_SHADER(FEmitEditorSelectionDepthPS, "/Engine/Private/Nanite/NaniteExportGBuffer.usf", "EmitEditorSelectionDepthPS", SF_Pixel);

// Overlay-color pass
class FEmitEditorOverlayColorPS : public FNaniteGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FEmitEditorOverlayColorPS);
	SHADER_USE_PARAMETER_STRUCT(FEmitEditorOverlayColorPS, FNaniteGlobalShader);

	using FParameters = FNaniteSelectionOutlineParameters;

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportNanite(Parameters.Platform);
	}
};
IMPLEMENT_GLOBAL_SHADER(FEmitEditorOverlayColorPS, "/Engine/Private/Nanite/NaniteExportGBuffer.usf", "EmitEditorOverlayColorPS", SF_Pixel);

// Shader that seeds the editor depth buffer with Nanite scene depth.
// This ensures non-Nanite editor selection draws are depth-tested against Nanite geometry.
class FEmitEditorNaniteSceneDepthPS : public FNaniteGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FEmitEditorNaniteSceneDepthPS);
	SHADER_USE_PARAMETER_STRUCT(FEmitEditorNaniteSceneDepthPS, FNaniteGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<UlongType>, NaniteSceneVisBuffer)
		SHADER_PARAMETER(FVector2f, NaniteSceneToOutputScale)
		SHADER_PARAMETER(FVector2f, NaniteSceneToOutputBias)
		RENDER_TARGET_BINDING_SLOTS()
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportNanite(Parameters.Platform);
	}
};
IMPLEMENT_GLOBAL_SHADER(FEmitEditorNaniteSceneDepthPS, "/Engine/Private/Nanite/NaniteExportGBuffer.usf", "EmitEditorNaniteSceneDepthPS", SF_Pixel);


namespace Nanite
{

void EmitEditorNaniteSceneDepth(
	FRDGBuilder& GraphBuilder,
	FRDGTextureRef EditorDepthTarget,
	FScene& Scene,
	const FViewInfo& SceneView,
	const FViewInfo& EditorView,
	const FRasterResults& NaniteRasterResults
)
{
#if WITH_EDITOR
	if (!NaniteRasterResults.VisBuffer64)
	{
		return;
	}

	auto* PassParameters = GraphBuilder.AllocParameters<FEmitEditorNaniteSceneDepthPS::FParameters>();

	PassParameters->NaniteSceneVisBuffer = NaniteRasterResults.VisBuffer64;

	// Transform from editor output coordinates to scene VisBuffer coordinates
	const FScreenTransform SceneTransform =
		FScreenTransform::ChangeRectFromTo(EditorView.ViewRect, SceneView.ViewRect);
	PassParameters->NaniteSceneToOutputScale = SceneTransform.Scale;
	PassParameters->NaniteSceneToOutputBias  = SceneTransform.Bias;

	PassParameters->RenderTargets.DepthStencil = FDepthStencilBinding(
		EditorDepthTarget,
		ERenderTargetLoadAction::EClear,
		ERenderTargetLoadAction::ENoAction,
		FExclusiveDepthStencil::DepthWrite_StencilNop);

	auto PixelShader = SceneView.ShaderMap->GetShader<FEmitEditorNaniteSceneDepthPS>();

	FPixelShaderUtils::AddFullscreenPass(
		GraphBuilder,
		SceneView.ShaderMap,
		RDG_EVENT_NAME("EmitEditorNaniteSceneDepth"),
		PixelShader,
		PassParameters,
		EditorView.ViewRect,
		TStaticBlendState<CW_NONE>::GetRHI(),
		TStaticRasterizerState<>::GetRHI(),
		TStaticDepthStencilState<true, CF_DepthNearOrEqual>::GetRHI()
	);
#endif
}

void DrawHitProxies(
	FRDGBuilder& GraphBuilder,
	const FScene& Scene,
	const FViewInfo& View, 
	const FRasterResults& RasterResults,
	FRDGTextureRef HitProxyTexture,
	FRDGTextureRef HitProxyDepthTexture
	)
{
#if WITH_EDITOR
	LLM_SCOPE_BYTAG(Nanite);

	RDG_EVENT_SCOPE_STAT(GraphBuilder, NaniteEditor, "NaniteHitProxyPass");

	const FRDGSystemTextures& SystemTextures = FRDGSystemTextures::Get(GraphBuilder);

	FRDGTextureRef VisBuffer64 = RasterResults.VisBuffer64 ? RasterResults.VisBuffer64 : SystemTextures.Black;

	FRDGBufferRef VisibleClustersSWHW = RasterResults.VisibleClustersSWHW;

	const FIntRect& ViewRect = View.ViewRect;
	{
		auto& MaterialsExtension = Scene.GetExtension<Nanite::FMaterialsSceneExtension>();
		auto* PassParameters = GraphBuilder.AllocParameters<FEmitHitProxyIdPS::FParameters>();

		PassParameters->View = View.ViewUniformBuffer;
		PassParameters->Scene = View.GetSceneUniforms().GetBuffer(GraphBuilder);
		PassParameters->VisibleClustersSWHW = GraphBuilder.CreateSRV(VisibleClustersSWHW);
		PassParameters->PageConstants = RasterResults.PageConstants;
		PassParameters->ViewRect = FInt32Vector4(ViewRect.Min.X, ViewRect.Min.Y, ViewRect.Max.X, ViewRect.Max.Y);
		PassParameters->ClusterPageData = Nanite::GStreamingManager.GetClusterPageDataSRV(GraphBuilder);
		PassParameters->HierarchyBuffer = Nanite::GStreamingManager.GetHierarchySRV(GraphBuilder);
		PassParameters->VisBuffer64 = VisBuffer64;
		PassParameters->MaterialHitProxyTable = GraphBuilder.CreateSRV(MaterialsExtension.CreateHitProxyIDBuffer(GraphBuilder));

		PassParameters->RenderTargets[0]			= FRenderTargetBinding(HitProxyTexture, ERenderTargetLoadAction::ELoad);
		PassParameters->RenderTargets.DepthStencil	= FDepthStencilBinding(HitProxyDepthTexture, ERenderTargetLoadAction::ELoad, FExclusiveDepthStencil::DepthWrite_StencilWrite);

		auto PixelShader = View.ShaderMap->GetShader<FEmitHitProxyIdPS>();

		FPixelShaderUtils::AddFullscreenPass(
			GraphBuilder,
			View.ShaderMap,
			RDG_EVENT_NAME("Nanite::EmitHitProxyId"),
			PixelShader,
			PassParameters,
			ViewRect,
			TStaticBlendState<>::GetRHI(),
			TStaticRasterizerState<>::GetRHI(),
			TStaticDepthStencilState<true, CF_DepthNearOrEqual>::GetRHI()
			);
	}
#endif
}

FRDGBufferSRVRef GetEditorSelectedHitProxyIdsSRV(FRDGBuilder& GraphBuilder, TArrayView<const uint32> HitProxyIds)
{
	FRDGBufferRef HitProxyIdsBuffer = nullptr;
	const uint32 BufferCount = HitProxyIds.Num();
	if (BufferCount > 0)
	{
		HitProxyIdsBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateUploadDesc(sizeof(uint32), BufferCount), TEXT("EditorSelectedNaniteHitProxyIds"));
		GraphBuilder.QueueBufferUpload(HitProxyIdsBuffer, HitProxyIds);
	}
	else
	{
		HitProxyIdsBuffer = GSystemTextures.GetDefaultBuffer<uint32>(GraphBuilder);
	}

	return GraphBuilder.CreateSRV(HitProxyIdsBuffer, PF_R32_UINT);
}

FRDGBufferSRVRef GetEditorSelectedHitProxyIdsSRV(FRDGBuilder& GraphBuilder, const FScene& Scene)
{
#if WITH_EDITOR
	if (const FEditorSceneExtension* EditorExtension = Scene.GetExtensionPtr<FEditorSceneExtension>())
	{
		return GetEditorSelectedHitProxyIdsSRV(
			GraphBuilder,
			EditorExtension->GetSelectionGroup(FEditorSceneExtension::ESelectionGroup::Selected).HitProxyIds);
	}
#endif
	return GetEditorSelectedHitProxyIdsSRV(GraphBuilder, TArrayView<const uint32>());
}

#if WITH_EDITOR
	
static void GetEditorSelectionVisBuffer(
	FRDGBuilder& GraphBuilder,
	FScene& Scene,
	const FViewInfo& SceneView,
	const FViewInfo& EditorView,
	FSceneUniformBuffer &SceneUniformBuffer,
	const FRasterResults& NaniteRasterResults,
	const FEditorSceneExtension::FInstanceDrawList& DrawList,
	FNaniteSelectionOutlineParameters& OutParameters
)
{
	const FRDGSystemTextures& SystemTextures = FRDGSystemTextures::Get(GraphBuilder);
	const FIntPoint RasterTextureSize = EditorView.ViewRect.Size();
	const FIntRect RasterViewRect = FIntRect(FIntPoint(0, 0), RasterTextureSize);

	Nanite::FSharedContext SharedContext{};
	SharedContext.FeatureLevel = Scene.GetFeatureLevel();
	SharedContext.ShaderPlatform = Scene.GetShaderPlatform();
	SharedContext.ShaderMap = GetGlobalShaderMap(SharedContext.ShaderPlatform);
	SharedContext.Pipeline = Nanite::ERasterPipeline::Primary;

	const Nanite::FRasterContextInitParams RasterInitParams =
	{
		.TextureSize = RasterTextureSize,
		.TextureRect = RasterViewRect,
		.RasterMode = Nanite::EOutputBufferMode::VisBuffer
	};
	Nanite::FRasterContext RasterContext = Nanite::InitRasterContext(
		GraphBuilder,
		SharedContext,
		*(const FViewFamilyInfo*)SceneView.Family,
		RasterInitParams
	);

	// Rasterize the view
	{
		Nanite::FConfiguration CullingConfig = { 0 };
		CullingConfig.bUpdateStreaming = true;
		CullingConfig.bEditorShowFlag = true;

		Nanite::FPackedViewParams NaniteViewParams;
		NaniteViewParams.ViewMatrices = EditorView.ViewMatrices;
		NaniteViewParams.PrevViewMatrices = EditorView.PrevViewInfo.ViewMatrices;
		NaniteViewParams.ViewRect = RasterViewRect;
		NaniteViewParams.RasterContextSize = RasterTextureSize;
		NaniteViewParams.HZBTestViewRect = EditorView.PrevViewInfo.ViewRect;
		NaniteViewParams.GlobalClippingPlane = EditorView.GlobalClippingPlane;
		NaniteViewParams.SceneRendererPrimaryViewId = EditorView.SceneRendererPrimaryViewId;
		Nanite::SetViewFrameIndex(&EditorView, NaniteViewParams);

		const Nanite::FPackedView NaniteView = Nanite::CreatePackedView(NaniteViewParams);

		auto NaniteRenderer = Nanite::IRenderer::Create(
			GraphBuilder,
			Scene,
			EditorView,
			SceneUniformBuffer,
			SharedContext,
			RasterContext,
			CullingConfig,
			RasterViewRect,
			/* PrevHZB = */ nullptr
		);

		NaniteRenderer->DrawGeometry(
			Scene.NaniteRasterPipelines[ENaniteMeshPass::BasePass],
			NaniteRasterResults.VisibilityQuery,
			*Nanite::FPackedViewArray::Create(GraphBuilder, NaniteView),
			DrawList
		);

		Nanite::FRasterResults RasterResults;
		NaniteRenderer->ExtractResults( RasterResults );

		const FScreenTransform OutputToInputTransform = FScreenTransform::ChangeRectFromTo(EditorView.ViewRect, RasterViewRect);

		OutParameters.VisBuffer64			= RasterResults.VisBuffer64 ? RasterResults.VisBuffer64 : SystemTextures.Black;
		OutParameters.VisibleClustersSWHW	= GraphBuilder.CreateSRV(RasterResults.VisibleClustersSWHW);
		OutParameters.OutputToInputScale	= OutputToInputTransform.Scale;
		OutParameters.OutputToInputBias		= OutputToInputTransform.Bias;
	}
}

static void AddEditorSelectionDepthPass(
	FRDGBuilder& GraphBuilder,
	FRDGTextureRef DepthTarget,
	bool bSelectionOutline,
	FScene& Scene,
	const FViewInfo& SceneView,
	const FViewInfo& EditorView,
	FSceneUniformBuffer &SceneUniformBuffer,
	const FRasterResults& NaniteRasterResults,
	const FEditorSceneExtension::FInstanceDrawList& DrawList,
	TArrayView<const uint32> HitProxyIds,
	int32 StencilRefValue,
	FRDGTextureRef SceneDepthTexture,
	const FIntRect& SceneDepthViewRect
)
{
	LLM_SCOPE_BYTAG(Nanite);
	RDG_EVENT_SCOPE_STAT(GraphBuilder, NaniteEditor, "NaniteEditor");

	auto& MaterialsExtension = Scene.GetExtension<Nanite::FMaterialsSceneExtension>();
	auto PassParameters = GraphBuilder.AllocParameters<FNaniteSelectionOutlineParameters>();

	GetEditorSelectionVisBuffer(
		GraphBuilder,
		Scene,
		SceneView,
		EditorView,
		SceneUniformBuffer,
		NaniteRasterResults,
		DrawList,
		*PassParameters);

	PassParameters->View					= EditorView.ViewUniformBuffer;
	PassParameters->Scene					= SceneView.GetSceneUniforms().GetBuffer(GraphBuilder);
	PassParameters->MaxVisibleClusters		= Nanite::FGlobalResources::GetMaxVisibleClusters();
	PassParameters->PageConstants			= NaniteRasterResults.PageConstants;
	PassParameters->ViewRect				= FInt32Vector4(EditorView.ViewRect.Min.X, EditorView.ViewRect.Min.Y, EditorView.ViewRect.Max.X, EditorView.ViewRect.Max.Y);
	PassParameters->ClusterPageData			= Nanite::GStreamingManager.GetClusterPageDataSRV(GraphBuilder);
	PassParameters->HierarchyBuffer			= Nanite::GStreamingManager.GetHierarchySRV(GraphBuilder);
	PassParameters->SceneVisBuffer64		= NaniteRasterResults.VisBuffer64;
	PassParameters->SceneVisibleClustersSWHW= GraphBuilder.CreateSRV(NaniteRasterResults.VisibleClustersSWHW);
	PassParameters->SceneDepthTexture		= SceneDepthTexture;
	PassParameters->MaterialHitProxyTable	= GraphBuilder.CreateSRV(MaterialsExtension.CreateHitProxyIDBuffer(GraphBuilder));
	PassParameters->RenderTargets.DepthStencil = FDepthStencilBinding(
		DepthTarget,
		ERenderTargetLoadAction::ELoad,
		ERenderTargetLoadAction::ELoad,
		FExclusiveDepthStencil::DepthWrite_StencilWrite);

	const FScreenTransform SceneOutputToInputTransform =
		FScreenTransform::ChangeRectFromTo(EditorView.ViewRect, SceneView.ViewRect);
	PassParameters->SceneOutputToInputScale = SceneOutputToInputTransform.Scale;
	PassParameters->SceneOutputToInputBias = SceneOutputToInputTransform.Bias;

	const FScreenTransform DepthOutputToInputTransform = FScreenTransform::ChangeRectFromTo(EditorView.ViewRect, SceneDepthViewRect);
	PassParameters->SceneDepthOutputToInputScale = DepthOutputToInputTransform.Scale;
	PassParameters->SceneDepthOutputToInputBias = DepthOutputToInputTransform.Bias;
	
	auto AddPass = [&GraphBuilder, &SceneView, &EditorView](
		FNaniteSelectionOutlineParameters* PassParameters,
		int32 StencilValue,
		bool bOnlySelected,
		FRHIDepthStencilState* DepthStencilState,
		FRHIBlendState* BlendState)
	{
		FEmitEditorSelectionDepthPS::FPermutationDomain PermutationVectorPS;
		PermutationVectorPS.Set<FEmitEditorSelectionDepthPS::FOnlySelectedDim>(bOnlySelected);
		
		auto PixelShader = SceneView.ShaderMap->GetShader<FEmitEditorSelectionDepthPS>(PermutationVectorPS);

		FPixelShaderUtils::AddFullscreenPass(
			GraphBuilder,
			SceneView.ShaderMap,
			RDG_EVENT_NAME("EditorSelectionDepth"),
			PixelShader,
			PassParameters,
			EditorView.ViewRect,
			BlendState,
			TStaticRasterizerState<>::GetRHI(),
			DepthStencilState,
			StencilValue
		);
	};

	FRHIDepthStencilState* const OutlineDepthStencilState =
		TStaticDepthStencilState<true, CF_DepthNearOrEqual, true, CF_Always, SO_Keep, SO_Keep, SO_Replace>::GetRHI();
	FRHIDepthStencilState* const DefaultDepthStencilState =
		TStaticDepthStencilState<true, CF_DepthNearOrEqual, true, CF_Always, SO_Keep, SO_Keep, SO_Replace>::GetRHI();

	FRHIBlendState* const OpaqueBlendState = TStaticBlendState<>::GetRHI();

	if (bSelectionOutline)
	{
		PassParameters->EditorSelectedHitProxyIds = GetEditorSelectedHitProxyIdsSRV(GraphBuilder, HitProxyIds);
		PassParameters->NumEditorSelectedHitProxyIds = HitProxyIds.Num();

		// Two passes for correct inter-object borders: NotSelected for all outline prims, then the
		// selection's stencil value for the selected subset.
		auto* OutlineParams = GraphBuilder.AllocParameters<FNaniteSelectionOutlineParameters>();
		*OutlineParams = *PassParameters;

		auto* OutlineParamsSelected = GraphBuilder.AllocParameters<FNaniteSelectionOutlineParameters>();
		*OutlineParamsSelected = *OutlineParams;

		AddPass(OutlineParams, EEditorSelectionStencilValues::NotSelected, /*bOnlySelected*/ false, OutlineDepthStencilState, OpaqueBlendState);
		AddPass(OutlineParamsSelected, StencilRefValue, /*bOnlySelected*/ true, OutlineDepthStencilState, OpaqueBlendState);
	}
	else
	{
		AddPass(PassParameters, StencilRefValue, /*bOnlySelected*/ false, DefaultDepthStencilState, OpaqueBlendState);
	}
}

// Renders editor overlay color for Nanite primitives that have an overlay color.
static void AddEditorOverlayColorPass(
	FRDGBuilder& GraphBuilder,
	FRDGTextureRef OverlayTarget,
	FScene& Scene,
	const FViewInfo& SceneView,
	const FViewInfo& EditorView,
	FSceneUniformBuffer& SceneUniformBuffer,
	const FRasterResults& NaniteRasterResults,
	const FEditorSceneExtension::FInstanceDrawList& OverlayDrawList,
	FRDGTextureRef SceneDepthTexture,
	const FIntRect& SceneDepthViewRect
)
{
	LLM_SCOPE_BYTAG(Nanite);
	RDG_EVENT_SCOPE_STAT(GraphBuilder, NaniteEditor, "NaniteEditorOverlay");

	auto& MaterialsExtension = Scene.GetExtension<Nanite::FMaterialsSceneExtension>();
	auto PassParameters = GraphBuilder.AllocParameters<FNaniteSelectionOutlineParameters>();

	GetEditorSelectionVisBuffer(
		GraphBuilder, Scene, SceneView, EditorView, SceneUniformBuffer, NaniteRasterResults, OverlayDrawList, *PassParameters);

	PassParameters->View					= EditorView.ViewUniformBuffer;
	PassParameters->Scene					= SceneView.GetSceneUniforms().GetBuffer(GraphBuilder);
	PassParameters->MaxVisibleClusters		= Nanite::FGlobalResources::GetMaxVisibleClusters();
	PassParameters->PageConstants			= NaniteRasterResults.PageConstants;
	PassParameters->ViewRect				= FInt32Vector4(EditorView.ViewRect.Min.X, EditorView.ViewRect.Min.Y, EditorView.ViewRect.Max.X, EditorView.ViewRect.Max.Y);
	PassParameters->ClusterPageData			= Nanite::GStreamingManager.GetClusterPageDataSRV(GraphBuilder);
	PassParameters->HierarchyBuffer			= Nanite::GStreamingManager.GetHierarchySRV(GraphBuilder);
	PassParameters->SceneVisBuffer64		= NaniteRasterResults.VisBuffer64;
	PassParameters->SceneVisibleClustersSWHW= GraphBuilder.CreateSRV(NaniteRasterResults.VisibleClustersSWHW);
	PassParameters->SceneDepthTexture		= SceneDepthTexture;
	PassParameters->MaterialHitProxyTable	= GraphBuilder.CreateSRV(MaterialsExtension.CreateHitProxyIDBuffer(GraphBuilder));

	const FScreenTransform SceneOutputToInputTransform =
		FScreenTransform::ChangeRectFromTo(EditorView.ViewRect, SceneView.ViewRect);
	PassParameters->SceneOutputToInputScale = SceneOutputToInputTransform.Scale;
	PassParameters->SceneOutputToInputBias = SceneOutputToInputTransform.Bias;

	const FScreenTransform DepthOutputToInputTransform = FScreenTransform::ChangeRectFromTo(EditorView.ViewRect, SceneDepthViewRect);
	PassParameters->SceneDepthOutputToInputScale = DepthOutputToInputTransform.Scale;
	PassParameters->SceneDepthOutputToInputBias = DepthOutputToInputTransform.Bias;

	// Color target only, no depth/stencil.
	PassParameters->RenderTargets[0] = FRenderTargetBinding(OverlayTarget, ERenderTargetLoadAction::ELoad);

	auto PixelShader = SceneView.ShaderMap->GetShader<FEmitEditorOverlayColorPS>();

	FPixelShaderUtils::AddFullscreenPass(
		GraphBuilder,
		SceneView.ShaderMap,
		RDG_EVENT_NAME("EditorOverlayColor"),
		PixelShader,
		PassParameters,
		EditorView.ViewRect,
		TStaticBlendState<CW_RGBA, BO_Add, BF_SourceAlpha, BF_InverseSourceAlpha>::GetRHI(),
		TStaticRasterizerState<>::GetRHI(),
		TStaticDepthStencilState<false, CF_Always>::GetRHI());
}

void DrawEditorSelection(
	FRDGBuilder& GraphBuilder,
	FRDGTextureRef DepthTarget,
	FRDGTextureRef OverlayTarget,
	FScene& Scene,
	const FViewInfo& SceneView,
	const FViewInfo& EditorView,
	FSceneUniformBuffer &SceneUniformBuffer,
	const FRasterResults* NaniteRasterResults,
	FRDGTextureRef SceneDepthTexture,
	const FIntRect SceneDepthViewRect
)
{
	if (NaniteRasterResults == nullptr)
	{
		return;
	}

	const FEditorSceneExtension* EditorExtension = Scene.GetExtensionPtr<FEditorSceneExtension>();
	if (EditorExtension == nullptr)
	{
		return;
	}

	// Ensure the extension's build task has finished before we read any of its draw lists.
	EditorExtension->SyncOnDrawLists();

	RDG_EVENT_SCOPE(GraphBuilder, "NaniteEditorSelection");

	// Each selection group must composite individually because each writes a different stencil value
	for (uint32 Index = 0; Index < uint32(FEditorSceneExtension::ESelectionGroup::Num); ++Index)
	{
		const FEditorSceneExtension::FSelectionGroup& SelectionGroup =
			EditorExtension->GetSelectionGroup(FEditorSceneExtension::ESelectionGroup(Index));
		if (!SelectionGroup.DrawList.IsEmpty())
		{
			// Encode the selection group index in the high 3 bits of the stencil value to get the shader to draw it with that color
			const int32 StencilValue = EEditorSelectionStencilValues::Nanite | int32(Index << 5);
			AddEditorSelectionDepthPass(
				GraphBuilder,
				DepthTarget,
				/*bSelectionOutline*/ true,
				Scene,
				SceneView,
				EditorView,
				SceneUniformBuffer,
				*NaniteRasterResults,
				SelectionGroup.DrawList,
				SelectionGroup.HitProxyIds,
				StencilValue,
				SceneDepthTexture,
				SceneDepthViewRect
			);
		}
	}

	// Overlay color pass
	const FEditorSceneExtension::FInstanceDrawList& OverlayDrawList = EditorExtension->GetOverlayDrawList();
	if (OverlayTarget && !OverlayDrawList.IsEmpty())
	{
		AddEditorOverlayColorPass(
			GraphBuilder,
			OverlayTarget,
			Scene,
			SceneView,
			EditorView,
			SceneUniformBuffer,
			*NaniteRasterResults,
			OverlayDrawList,
			SceneDepthTexture,
			SceneDepthViewRect);
	}
}

void DrawEditorVisualizeLevelInstance(
	FRDGBuilder& GraphBuilder,
	FRDGTextureRef DepthTarget,
	FScene& Scene,
	const FViewInfo& SceneView,
	const FViewInfo& EditorView,
	FSceneUniformBuffer &SceneUniformBuffer,
	const FRasterResults* NaniteRasterResults,
	FRDGTextureRef SceneDepthTexture,
	const FIntRect SceneDepthViewRect
)
{
	if (NaniteRasterResults == nullptr)
	{
		return;
	}

	const FEditorSceneExtension* EditorExtension = Scene.GetExtensionPtr<FEditorSceneExtension>();
	if (EditorExtension == nullptr)
	{
		return;
	}

	const FEditorSceneExtension::FInstanceDrawList& DrawList = EditorExtension->GetVisualizeLevelInstancesDrawList();
	if (DrawList.Num() == 0)
	{
		return;
	}

	RDG_EVENT_SCOPE(GraphBuilder, "NaniteVisualizeLevelInstances");
	AddEditorSelectionDepthPass(
		GraphBuilder,
		DepthTarget,
		/*bSelectionOutline*/ false,
		Scene,
		SceneView,
		EditorView,
		SceneUniformBuffer,
		*NaniteRasterResults,
		DrawList,
		TConstArrayView<uint32>(),
		EEditorSelectionStencilValues::VisualizeLevelInstances,
		SceneDepthTexture,
		SceneDepthViewRect
	);
}

///////////////////////////////////////////////////////////////////////////////
// FEditorSceneExtension
///////////////////////////////////////////////////////////////////////////////

IMPLEMENT_SCENE_EXTENSION(FEditorSceneExtension);

bool FEditorSceneExtension::ShouldCreateExtension(FScene& InScene)
{
	return DoesRuntimeSupportNanite(GetFeatureLevelShaderPlatform(InScene.GetFeatureLevel()), true, true);
}

ISceneExtensionUpdater* FEditorSceneExtension::CreateUpdater()
{
	return new FUpdater(*this);
}

const FEditorSceneExtension::FSelectionGroup& FEditorSceneExtension::GetSelectionGroup(ESelectionGroup Group) const
{
	check(uint32(Group) < uint32(ESelectionGroup::Num));
	SyncOnDrawLists();
	return SelectionGroups[uint32(Group)];
}

const FEditorSceneExtension::FInstanceDrawList& FEditorSceneExtension::GetVisualizeLevelInstancesDrawList() const
{
	SyncOnDrawLists();
	return VisualizeLevelInstancesDrawList;
}

const FEditorSceneExtension::FInstanceDrawList& FEditorSceneExtension::GetOverlayDrawList() const
{
	SyncOnDrawLists();
	return OverlayDrawList;
}

void FEditorSceneExtension::SyncOnDrawLists() const
{
	TaskHandles[BuildDrawListsTask].Wait();
}

FEditorSceneExtension::FUpdater::FUpdater(FEditorSceneExtension& InSceneData)
	: SceneData(&InSceneData)
	, bEnableAsync(CVarNaniteEditorDrawListsAsyncUpdates.GetValueOnRenderThread())
{
}

void FEditorSceneExtension::FUpdater::End()
{
	// Ensure the build task finishes before we fall out of scope, since its lambda captures `this`.
	SceneData->SyncOnDrawLists();
}

namespace EditorSceneExtensionImpl
{
	static constexpr uint32 NumSelectionGroups = uint32(FEditorSceneExtension::ESelectionGroup::Num);

	/** Per-parallel-task accumulation buffers. Merged into the scene extension after ParallelFor completes. */
	struct FBuildContext
	{
		FEditorSceneExtension::FInstanceDrawList VisualizeLevelInstancesDrawList;
		FEditorSceneExtension::FInstanceDrawList OverlayDrawList;
		FEditorSceneExtension::FSelectionGroup   SelectionGroups[NumSelectionGroups];
	};

	/** Collect editor draw data from a single Nanite primitive into a context. */
	static void AccumulatePrimitive(
		FBuildContext& Context,
		const FPrimitiveSceneInfo* PrimitiveSceneInfo,
		bool bHasSelectedComponents)
	{
		check(PrimitiveSceneInfo);
		check(PrimitiveSceneInfo->Proxy->IsNaniteMesh());
		const auto* NaniteProxy = static_cast<const Nanite::FSceneProxyBase*>(PrimitiveSceneInfo->Proxy);

		const bool bIsSelected = NaniteProxy->IsSelected();
		const bool bWantsEditorEffects = NaniteProxy->WantsEditorEffects();
		const bool bEditingLevelInstanceChild = NaniteProxy->IsEditingLevelInstanceChild();
		const bool bHasOverlay = NaniteProxy->GetOverlayColor().A != 0;
		const bool bWantsOutline = bIsSelected || (bWantsEditorEffects && !bHasOverlay);

		// Early out for proxies that don't contribute to any editor draw list.		
		if (!bWantsOutline && !bHasOverlay && !bEditingLevelInstanceChild)
		{
			return;
		}

		const int32 MaxInstances = PrimitiveSceneInfo->GetNumInstanceSceneDataEntries();
		const int32 InstanceSceneDataOffset = PrimitiveSceneInfo->GetInstanceSceneDataOffset();

		// Overlay color 
		if (bHasOverlay)
		{
			FEditorSceneExtension::FInstanceDrawList& OverlayList = Context.OverlayDrawList;
			OverlayList.Reserve(OverlayList.Num() + MaxInstances);
			for (int32 Idx = 0; Idx < MaxInstances; ++Idx)
			{
				OverlayList.Add(Nanite::FInstanceDraw{ uint32(InstanceSceneDataOffset + Idx), 0u });
			}
		}

		// Selection outline.
		if (bWantsOutline)
		{
			const bool bParentOnlySelected = bIsSelected
				&& bHasSelectedComponents
				&& NaniteProxy->IsParentSelected()
				&& !NaniteProxy->IsIndividuallySelected();
			const FEditorSceneExtension::ESelectionGroup SelectionGroupId = bParentOnlySelected
				? FEditorSceneExtension::ESelectionGroup::ParentOnlySelected
				: FEditorSceneExtension::ESelectionGroup::Selected;
			FEditorSceneExtension::FSelectionGroup& SelectionGroup = Context.SelectionGroups[uint32(SelectionGroupId)];

			if (NaniteProxy->HasSelectedInstances())
			{
				// Collect instance draws and per-instance hit proxies for selected instances
				const FInstanceSceneDataBuffers* InstanceSceneDataBuffers = PrimitiveSceneInfo->GetInstanceSceneDataBuffers();
				if (InstanceSceneDataBuffers != nullptr && !InstanceSceneDataBuffers->IsInstanceDataGPUOnly())
				{
					FInstanceSceneDataBuffers::FReadView ProxyData = InstanceSceneDataBuffers->GetReadView();
					SelectionGroup.DrawList.Reserve(SelectionGroup.DrawList.Num() + MaxInstances);
					SelectionGroup.HitProxyIds.Reserve(SelectionGroup.HitProxyIds.Num() + MaxInstances);
					for (int32 Idx = 0; Idx < MaxInstances; ++Idx)
					{
						// If we have per-instance editor data, exclude instance draws of unselected instances
						if (ProxyData.InstanceEditorData.IsValidIndex(Idx))
						{
							FColor HitProxyColor;
							bool bInstanceSelected;
							FInstanceEditorData::Unpack(ProxyData.InstanceEditorData[Idx], HitProxyColor, bInstanceSelected);
							if (!bInstanceSelected)
							{
								continue;
							}

							const uint32 HitProxyID = HitProxyColor.ToPackedABGR();
							SelectionGroup.HitProxyIds.Add(HitProxyID);
							SelectionGroup.DrawList.Add(
								Nanite::FInstanceDraw{ uint32(InstanceSceneDataOffset + Idx), 0u }
							);
						}
					}
				}
			}
			else
			{
				if (bIsSelected)
				{
					// Primitive is selected but not individual instances, so add the primitive's hit proxy IDs
					TArrayView<const FHitProxyId> PrimitiveHitProxyIds = NaniteProxy->GetHitProxyIds();
					SelectionGroup.HitProxyIds.Reserve(SelectionGroup.HitProxyIds.Num() + PrimitiveHitProxyIds.Num());
					for (const FHitProxyId& HitProxyId : PrimitiveHitProxyIds)
					{
						const uint32 HitProxyID = HitProxyId.GetColor().ToPackedABGR();
						SelectionGroup.HitProxyIds.Add(HitProxyID);
					}
				}

				// Add all instances to the draw list, because all are either selected or want editor effects
				// (Nanite meshes with editor effects are also handled in the selection outline rendering pass)
				SelectionGroup.DrawList.Reserve(SelectionGroup.DrawList.Num() + MaxInstances);
				for (int32 Idx = 0; Idx < MaxInstances; ++Idx)
				{
					SelectionGroup.DrawList.Add(
						Nanite::FInstanceDraw{ uint32(InstanceSceneDataOffset + Idx), 0u }
					);
				}
			}
		}

		// Level instance visualization draws.
		if (bEditingLevelInstanceChild)
		{
			FEditorSceneExtension::FInstanceDrawList& DrawList = Context.VisualizeLevelInstancesDrawList;
			DrawList.Reserve(DrawList.Num() + MaxInstances);
			for (int32 Idx = 0; Idx < MaxInstances; ++Idx)
			{
				DrawList.Add(Nanite::FInstanceDraw{ uint32(InstanceSceneDataOffset + Idx), 0u });
			}
		}
	}
}

void FEditorSceneExtension::FUpdater::PreSceneUpdate(FRDGBuilder& GraphBuilder, const FScenePreUpdateChangeSet& ChangeSet)
{
	// Sync on any draw list task (shouldn't be necessary, but protects against re-entry)
	SceneData->SyncOnDrawLists();

	// Update whether or not we are enabled based on in Nanite is enabled
	SceneData->bEnabled = UseNanite(GetFeatureLevelShaderPlatform(SceneData->Scene.GetFeatureLevel()));
	if (!SceneData->bEnabled)
	{
		SceneData->RelevantPrimitives.Reset();
		return;
	}

	// Remove primitive IDs from our list of relevant primitives
	for (const auto& PersistentIndex : ChangeSet.RemovedPrimitiveIds)
	{
		SceneData->RelevantPrimitives.Remove(PersistentIndex);
	}
}

void FEditorSceneExtension::FUpdater::PostSceneUpdate(FRDGBuilder& GraphBuilder, const FScenePostUpdateChangeSet& ChangeSet)
{
	if (!SceneData->bEnabled)
	{
		return;
	}

	// JWH-TODO - HACK! `bHasSelectedComponents` comes from GEditor->GetSelectedComponentCount() and is therefore
	// never actually per-view. Can't read GEditor from this module because of the direction of dependencies. This
	// seems to work, but would be cleaner if this boolean were set from some logic at a higher, editor-only level.
	bool bHasSelectedComponents = false;
	if (const FSceneRendererBase* ActiveSceneRenderer = FSceneRendererBase::GetActiveInstance(GraphBuilder))
	{
		if (ActiveSceneRenderer->Views.Num() > 0)
		{
			bHasSelectedComponents = ActiveSceneRenderer->Views[0].bHasSelectedComponents;
		}
	}

	SceneData->TaskHandles[BuildDrawListsTask] = GraphBuilder.AddSetupTask(
		[this, bHasSelectedComponents, AddedList = ChangeSet.AddedPrimitiveSceneInfos]
		{
			using namespace EditorSceneExtensionImpl;

			// Reset the draw lists and hit proxy ids, we'll fill them from merged contexts below.
			SceneData->VisualizeLevelInstancesDrawList.Reset();
			SceneData->OverlayDrawList.Reset();
			for (uint32 Index = 0; Index < uint32(ESelectionGroup::Num); ++Index)
			{
				SceneData->SelectionGroups[Index].DrawList.Reset();
				SceneData->SelectionGroups[Index].HitProxyIds.Reset();
			}

			// First add all relevant new primitives to our relevant primitive list
			// JWH-TODO: get this relevant list from the editor so as not to have to visit all Nanite primitives
			for (const FPrimitiveSceneInfo* AddedPrimitive : AddedList)
			{
				if (AddedPrimitive->Proxy->IsNaniteMesh())
				{
					SceneData->RelevantPrimitives.Add(AddedPrimitive->GetPersistentIndex());
				}				
			}
			
			const FScene& Scene = SceneData->Scene;
			const int32 NumPrimitives = SceneData->RelevantPrimitives.Num();
			if (NumPrimitives == 0)
			{
				return;
			}

			// Get a flat array of relevant primitive infos
			TArray<const FPrimitiveSceneInfo*, SceneRenderingAllocator> PrimitiveInfos;
			PrimitiveInfos.Reserve(NumPrimitives);
			for (FPersistentPrimitiveIndex PersistentIndex : SceneData->RelevantPrimitives)
			{
				PrimitiveInfos.Add(Scene.GetPrimitiveSceneInfo(PersistentIndex));
			}
				
			TArray<FBuildContext, SceneRenderingAllocator> Contexts;
			ParallelForWithTaskContext(
				TEXT("Nanite::FEditorSceneExtension::BuildDrawLists"),
				Contexts,
				NumPrimitives,
				10,
				[this, &PrimitiveInfos, bHasSelectedComponents](FBuildContext& Context, int32 PrimitiveIndex)
				{
					AccumulatePrimitive(Context, PrimitiveInfos[PrimitiveIndex], bHasSelectedComponents);
				},
				bEnableAsync ? EParallelForFlags::None : EParallelForFlags::ForceSingleThread
			);

			// Merge per-task contexts into the extension-owned arrays.
			int32 TotalSelectionDraws[NumSelectionGroups] = {};
			int32 TotalHitProxyIds[NumSelectionGroups] = {};
			int32 TotalVisualizeLevelInstanceDraws = 0;
			int32 TotalOverlayDraws = 0;
			for (const FBuildContext& Context : Contexts)
			{
				TotalVisualizeLevelInstanceDraws += Context.VisualizeLevelInstancesDrawList.Num();
				TotalOverlayDraws += Context.OverlayDrawList.Num();
				for (uint32 Index = 0; Index < NumSelectionGroups; ++Index)
				{
					TotalSelectionDraws[Index] += Context.SelectionGroups[Index].DrawList.Num();
					TotalHitProxyIds[Index]    += Context.SelectionGroups[Index].HitProxyIds.Num();
				}
			}
			SceneData->VisualizeLevelInstancesDrawList.Reserve(TotalVisualizeLevelInstanceDraws);
			SceneData->OverlayDrawList.Reserve(TotalOverlayDraws);
			for (uint32 Index = 0; Index < NumSelectionGroups; ++Index)
			{
				SceneData->SelectionGroups[Index].DrawList.Reserve(TotalSelectionDraws[Index]);
				SceneData->SelectionGroups[Index].HitProxyIds.Reserve(TotalHitProxyIds[Index]);
			}
			for (FBuildContext& Context : Contexts)
			{
				SceneData->VisualizeLevelInstancesDrawList.Append(MoveTemp(Context.VisualizeLevelInstancesDrawList));
				SceneData->OverlayDrawList.Append(MoveTemp(Context.OverlayDrawList));
				for (uint32 Index = 0; Index < NumSelectionGroups; ++Index)
				{
					SceneData->SelectionGroups[Index].DrawList.Append(MoveTemp(Context.SelectionGroups[Index].DrawList));
					SceneData->SelectionGroups[Index].HitProxyIds.Append(MoveTemp(Context.SelectionGroups[Index].HitProxyIds));
				}
			}

			// Sort each selection group's hit proxy ID list so the shader's binary search works.
			for (uint32 Index = 0; Index < NumSelectionGroups; ++Index)
			{
				Algo::Sort(SceneData->SelectionGroups[Index].HitProxyIds);
			}
		},
		UE::Tasks::ETaskPriority::Normal,
		bEnableAsync
	);
}

#endif // WITH_EDITOR

} // namespace Nanite
