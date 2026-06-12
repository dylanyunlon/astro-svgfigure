// Copyright Epic Games, Inc. All Rights Reserved.

#include "NaniteTranslucency.h"
#include "NaniteCullRaster.h"
#include "RenderingThread.h"
#include "PrimitiveViewRelevance.h"
#include "MaterialDomain.h"
#include "MeshDrawShaderBindings.h"
#include "MeshBatch.h"
#include "MeshMaterialShader.h"
#include "RenderUtils.h"
#include "SceneView.h"
#include "ScenePrivate.h"
#include "SceneUtils.h"
#include "SystemTextures.h"
#include "ShadowRendering.h"
#include "TranslucentRendering.h"

static TAutoConsoleVariable<int32> CVarNaniteMeshShaderTranslucency(
	TEXT("r.Nanite.MeshShaderTranslucency"),
	1,
	TEXT("If available, use mesh shaders for hardware translucency."),
	ECVF_RenderThreadSafe
);

bool UseNaniteMeshShader(EShaderPlatform ShaderPlatform)
{
	if (!FDataDrivenShaderPlatformInfo::GetSupportsMeshShadersTier0(ShaderPlatform))
	{
		return false;
	}

	if (!NaniteMeshShadersSupported(ShaderPlatform))
	{
		return false;
	}

	// Disable mesh shaders if global clip planes are enabled and the platform cannot support MS with clip distance output
	static const auto AllowGlobalClipPlaneVar = IConsoleManager::Get().FindTConsoleVariableDataInt(TEXT("r.AllowGlobalClipPlane"));
	static const bool bAllowGlobalClipPlane = (AllowGlobalClipPlaneVar && AllowGlobalClipPlaneVar->GetValueOnAnyThread() != 0);
	const bool bMSSupportsClipDistance = FDataDrivenShaderPlatformInfo::GetSupportsMeshShadersWithClipDistance(ShaderPlatform);

	const bool bSupported = CVarNaniteMeshShaderTranslucency.GetValueOnAnyThread() != 0 && GRHISupportsMeshShadersTier0 && (!bAllowGlobalClipPlane || bMSSupportsClipDistance);
	return bSupported;
}

FNaniteTranslucencyFactory::FNaniteTranslucencyFactory(ERHIFeatureLevel::Type FeatureLevel)
: FVertexFactory(FeatureLevel)
{
	bNeedsDeclaration = false;
}

FNaniteTranslucencyFactory::~FNaniteTranslucencyFactory()
{
	ReleaseResource();
}

void FNaniteTranslucencyFactory::InitRHI(FRHICommandListBase& RHICmdList)
{
	//LLM_SCOPE_BYTAG(Nanite);
}

bool FNaniteTranslucencyFactory::ShouldCompilePermutation(const FVertexFactoryShaderPermutationParameters& Parameters)
{
	bool bShouldCompile =
		NaniteTranslucencySupported() &&
		Parameters.MaterialParameters.bIsUsedWithNanite &&
		Parameters.MaterialParameters.MaterialDomain == EMaterialDomain::MD_Surface &&
		IsTranslucentBlendMode(Parameters.MaterialParameters.BlendMode) &&
		DoesPlatformSupportNanite(Parameters.Platform);

	return bShouldCompile;
}

void FNaniteTranslucencyFactory::ModifyCompilationEnvironment(const FVertexFactoryShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
{
	FVertexFactory::ModifyCompilationEnvironment(Parameters, OutEnvironment);

	const bool bUseNaniteUniformBuffers = Parameters.ShaderType->GetFrequency() != SF_RayHitGroup;

	OutEnvironment.SetDefine(TEXT("IS_NANITE_SHADING_PASS"), 1);
	OutEnvironment.SetDefine(TEXT("IS_NANITE_PASS"), 1);
	OutEnvironment.SetDefine(TEXT("IS_NANITE_TRANSLUCENCY"), 1);
	OutEnvironment.SetDefine(TEXT("USE_ANALYTIC_DERIVATIVES"), 0);
	OutEnvironment.SetDefine(TEXT("VF_SUPPORTS_PRIMITIVE_SCENE_DATA"), 1);
	OutEnvironment.SetDefine(TEXT("NANITE_USE_RASTER_UNIFORM_BUFFER"), bUseNaniteUniformBuffers);
	OutEnvironment.SetDefine(TEXT("NANITE_USE_SHADING_UNIFORM_BUFFER"), bUseNaniteUniformBuffers);
	OutEnvironment.SetDefine(TEXT("NANITE_USE_RAYTRACING_UNIFORM_BUFFER"), !bUseNaniteUniformBuffers);
	OutEnvironment.SetDefine(TEXT("NANITE_USE_VIEW_UNIFORM_BUFFER"), 0);
	OutEnvironment.SetDefine(TEXT("NANITE_COMPUTE_SHADE"), 0);
	OutEnvironment.SetDefine(TEXT("ALWAYS_EVALUATE_WORLD_POSITION_OFFSET"),
							 Parameters.MaterialParameters.bAlwaysEvaluateWorldPositionOffset ? 1 : 0);
	OutEnvironment.SetDefine(TEXT("VIRTUAL_TEXTURE_TARGET"), 0);

	OutEnvironment.SetCompileArgument(TEXT("USING_MESH_SHADING"), true);

	if (NaniteSplineMeshesSupported())
	{
		if (Parameters.MaterialParameters.bIsUsedWithSplineMeshes || Parameters.MaterialParameters.bIsDefaultMaterial)
		{
			// NOTE: This effectively means the logic to deform vertices will be added to the barycentrics calculation in the
			// Nanite shading CS, but will be branched over on instances that do not supply spline mesh parameters. If that
			// frequently causes occupancy issues, we may want to consider ways to split the spline meshes into their own
			// shading bin and permute the CS.
			OutEnvironment.SetDefine(TEXT("USE_SPLINEDEFORM"), 1);
			OutEnvironment.SetDefine(TEXT("USE_SPLINE_MESH_SCENE_RESOURCES"), UseSplineMeshSceneResources(Parameters.Platform));
		}
	}

	if (NaniteSkinnedMeshesSupported())
	{
		if (Parameters.MaterialParameters.bIsUsedWithSkeletalMesh || Parameters.MaterialParameters.bIsDefaultMaterial)
		{
			OutEnvironment.SetDefine(TEXT("USE_SKINNING"), 1);
		}
	}

	OutEnvironment.CompilerFlags.Add(CFLAG_ForceDXC);
	OutEnvironment.CompilerFlags.Add(CFLAG_HLSL2021);
	//OutEnvironment.CompilerFlags.Add(CFLAG_Wave32);
	OutEnvironment.CompilerFlags.Add(CFLAG_RootConstants);
	OutEnvironment.CompilerFlags.Add(CFLAG_ShaderBundle);
	//OutEnvironment.CompilerFlags.Add(CFLAG_CheckForDerivativeOps);
	OutEnvironment.CompilerFlags.Add(CFLAG_SupportsMinimalBindless);
}

IMPLEMENT_VERTEX_FACTORY_TYPE(FNaniteTranslucencyFactory, "/Engine/Private/Nanite/NaniteTranslucencyFactory.ush",
	  EVertexFactoryFlags::UsedWithMaterials
	| EVertexFactoryFlags::SupportsStaticLighting
	| EVertexFactoryFlags::SupportsGPUScene
	| EVertexFactoryFlags::SupportsNaniteRendering
	| EVertexFactoryFlags::SupportsManualVertexFetch
	| EVertexFactoryFlags::SupportsMeshShading
);

class FTranscodeRasterizerArgs_CS : public FNaniteGlobalShader
{
	DECLARE_GLOBAL_SHADER(FTranscodeRasterizerArgs_CS);
	SHADER_USE_PARAMETER_STRUCT(FTranscodeRasterizerArgs_CS, FNaniteGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, InRasterizerArgsSWHW)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, OutRasterizerArgsSWHW)
		SHADER_PARAMETER(uint32, RasterBinCount)
		SHADER_PARAMETER(uint32, InputFormat)
		SHADER_PARAMETER(uint32, OutputFormat)
	END_SHADER_PARAMETER_STRUCT()
};
IMPLEMENT_GLOBAL_SHADER(FTranscodeRasterizerArgs_CS, "/Engine/Private/Nanite/NaniteTranslucency.usf", "TranscodeRasterizerArgs", SF_Compute);

namespace Nanite
{

void FTranslucencyFactoryResource::InitRHI(FRHICommandListBase& RHICmdList)
{
	if (DoesPlatformSupportNanite(GMaxRHIShaderPlatform))
	{
		//LLM_SCOPE_BYTAG(Nanite);
		VertexFactory = new FNaniteTranslucencyFactory(ERHIFeatureLevel::SM5);
		VertexFactory->InitResource(RHICmdList);
	}
}

void FTranslucencyFactoryResource::ReleaseRHI()
{
	if (DoesPlatformSupportNanite(GMaxRHIShaderPlatform))
	{
		//LLM_SCOPE_BYTAG(Nanite);
		delete VertexFactory;
		VertexFactory = nullptr;
	}
}

TGlobalResource<FTranslucencyFactoryResource> GTranslucencyFactoryResource;

void SetTranslucencyParameters(FRDGBuilder& GraphBuilder, const FScene* Scene, const FViewInfo& View, const Nanite::FRasterResults* RasterResults, FNaniteTranslucencyParameters& Parameters)
{
	if (RasterResults)
	{
		Parameters.InViews = GraphBuilder.CreateSRV(RasterResults->ViewsBuffer);
		Parameters.RasterBinMeta = GraphBuilder.CreateSRV(RasterResults->RasterBinMeta);
		Parameters.RasterBinData = GraphBuilder.CreateSRV(RasterResults->RasterBinData);
		Parameters.RasterGroupMeta = GraphBuilder.CreateSRV(RasterResults->RasterGroupMeta);
	}
	else
	{
		Parameters.InViews = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(GSystemTextures.GetDefaultStructuredBuffer<FPackedNaniteView>(GraphBuilder)));
		Parameters.RasterBinMeta = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(GSystemTextures.GetDefaultStructuredBuffer<FNaniteRasterBinMeta>(GraphBuilder)));
		Parameters.RasterBinData = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(GSystemTextures.GetDefaultStructuredBuffer<FUintVector2>(GraphBuilder)));
		Parameters.RasterGroupMeta = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(GSystemTextures.GetDefaultStructuredBuffer<FNaniteRasterGroupMeta>(GraphBuilder)));
	}
}

void RenderTranslucency(
	FRDGBuilder& GraphBuilder,
	const FSceneRenderer& SceneRenderer,
	FViewInfo& View, int32 ViewIndex,
	FScreenPassTextureViewport Viewport,
	float ViewportScale,
	FRDGTextureMSAA SceneColorTexture,
	ERenderTargetLoadAction SceneColorLoadAction,
	FRDGTextureRef SceneDepthTexture,
	FTranslucentBasePassParameters* PassParameters,
	TRDGUniformBufferRef<FTranslucentBasePassUniformParameters> BasePassParameters,
	ETranslucencyPass::Type TranslucencyPass,
	FRDGBufferRef RasterBinArgs,
	const FNaniteTranslucencyContext& TranslucencyContext
)
{
	if (!NaniteTranslucencySupported() || TranslucencyContext.TranslucencyBins.Num() == 0)
	{
		return;
	}

	const ERHIFeatureLevel::Type FeatureLevel = SceneRenderer.FeatureLevel;
	const ERasterHardwarePath RasterHardwarePath = GetRasterHardwarePath(SceneRenderer.ShaderPlatform, ERasterPipeline::Primary);

	const bool bAllowHardwareVRS = HardwareVariableRateShadingSupportedByPlatform(SceneRenderer.ShaderPlatform);
	const bool bUseMeshShaders = UseNaniteMeshShader(SceneRenderer.ShaderPlatform);
	const bool bRemapPrimitiveToMeshShader = (RasterHardwarePath == ERasterHardwarePath::PrimitiveShader);
	const ERasterHardwarePath HardwarePath = bUseMeshShaders ? (bRemapPrimitiveToMeshShader ? ERasterHardwarePath::MeshShaderWrapped : RasterHardwarePath) : ERasterHardwarePath::VertexShader;

	FNaniteTranslucencyContext& Context = *GraphBuilder.AllocObject<FNaniteTranslucencyContext>();
	Context = TranslucencyContext; // Need to take a copy for RDG thread safety

	const bool bShowDrawEvents = GShowMaterialDrawEvents != 0;

	// Transcode the indirect arguments when the input and output indirect layouts differ
	if (RasterHardwarePath != HardwarePath)
	{
		FTranscodeRasterizerArgs_CS::FParameters* TranscodeParameters = GraphBuilder.AllocParameters<FTranscodeRasterizerArgs_CS::FParameters>();

		TranscodeParameters->RasterBinCount = 0;
		for (const FNaniteTranslucencyBin& TranslucencyBin : Context.TranslucencyBins)
		{
			TranscodeParameters->RasterBinCount = FMath::Max<uint32>(TranscodeParameters->RasterBinCount, TranslucencyBin.RasterBin + 1u);
		}

		switch (RasterHardwarePath)
		{
		case ERasterHardwarePath::PrimitiveShader:
			TranscodeParameters->InputFormat = NANITE_RENDER_FLAG_PRIMITIVE_SHADER;
			break;

		case ERasterHardwarePath::MeshShader:
		case ERasterHardwarePath::MeshShaderWrapped:
		case ERasterHardwarePath::MeshShaderNV:
			TranscodeParameters->InputFormat = NANITE_RENDER_FLAG_MESH_SHADER;
			break;

		case ERasterHardwarePath::VertexShader:
			TranscodeParameters->InputFormat = 0;
			break;

		default:
			check(false); // Not implemented
			TranscodeParameters->InputFormat = 0;
			break;
		}

		switch (HardwarePath)
		{
		case ERasterHardwarePath::PrimitiveShader:
			TranscodeParameters->OutputFormat = NANITE_RENDER_FLAG_PRIMITIVE_SHADER;
			break;

		case ERasterHardwarePath::MeshShader:
		case ERasterHardwarePath::MeshShaderWrapped:
			TranscodeParameters->OutputFormat = NANITE_RENDER_FLAG_MESH_SHADER;
			break;

		case ERasterHardwarePath::VertexShader:
			TranscodeParameters->OutputFormat = 0;
			break;

		default:
			check(false); // Not implemented
			TranscodeParameters->OutputFormat = 0;
			break;
		}

		TranscodeParameters->InRasterizerArgsSWHW = GraphBuilder.CreateSRV(RasterBinArgs);

		FRDGBufferRef OutRasterizerArgsSWHW = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateIndirectDesc(TranscodeParameters->RasterBinCount * NANITE_RASTERIZER_ARG_COUNT), TEXT("Nanite.TranscodedIndirectArgs"));
		TranscodeParameters->OutRasterizerArgsSWHW = GraphBuilder.CreateUAV(OutRasterizerArgsSWHW);

		FGlobalShaderMap* GlobalShaderMap = View.ShaderMap;
		auto ComputeShader = GlobalShaderMap->GetShader<FTranscodeRasterizerArgs_CS>();

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("TranscodeRasterizerArgs"),
			ComputeShader,
			TranscodeParameters,
			FComputeShaderUtils::GetGroupCount(TranscodeParameters->RasterBinCount, 64)
		);

		PassParameters->NaniteIndirectArgs = OutRasterizerArgsSWHW;
	}

	GraphBuilder.AddPass(
		RDG_EVENT_NAME("NaniteTranslucency(%s) %dx%d",
			TranslucencyPassToString(TranslucencyPass),
			int32(View.ViewRect.Width() * ViewportScale),
			int32(View.ViewRect.Height() * ViewportScale)),
		PassParameters,
		ERDGPassFlags::Raster,
		[&View, &Context, TranslucencyPass, PassParameters, ViewportScale, HardwarePath, bRemapPrimitiveToMeshShader, bAllowHardwareVRS, bShowDrawEvents](FRDGAsyncTask, FRHICommandList& RHICmdList)
		{
			PassParameters->NaniteIndirectArgs->MarkResourceAsUsed();

			FSceneRenderer::SetStereoViewport(RHICmdList, View, ViewportScale);
			RHICmdList.SetStreamSource(0, nullptr, 0);

			const bool bNeedsPointList = (HardwarePath == ERasterHardwarePath::PrimitiveShader) || (bRemapPrimitiveToMeshShader && HardwarePath != ERasterHardwarePath::VertexShader);
			EPrimitiveType PrimitiveType = bNeedsPointList ? PT_PointList : PT_TriangleList;
			FRHIVertexDeclaration* VertexDeclaration = (HardwarePath == ERasterHardwarePath::VertexShader) ? GEmptyVertexDeclaration.VertexDeclarationRHI : nullptr;

			FGraphicsPipelineStateInitializer GraphicsPSOInit;
			RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);

			FRHIDepthStencilState* PassDepthStencilState = nullptr;
			switch (TranslucencyPass)
			{
			default:
				break;

			case ETranslucencyPass::Type::TPT_AllTranslucency:
			case ETranslucencyPass::Type::TPT_TranslucencyStandard:
			case ETranslucencyPass::Type::TPT_TranslucencyHoldout:
			case ETranslucencyPass::Type::TPT_TranslucencyStandardModulate:
			case ETranslucencyPass::Type::TPT_TranslucencyAfterDOF:
			case ETranslucencyPass::Type::TPT_TranslucencyAfterDOFModulate:
				PassDepthStencilState = TStaticDepthStencilState<false, CF_DepthNearOrEqual>::GetRHI();
				break;

			case ETranslucencyPass::Type::TPT_TranslucencyAfterMotionBlur:
				PassDepthStencilState = TStaticDepthStencilState<false, CF_Always>::GetRHI();
				// GW-TODO: SetDepthStencilAccess(FExclusiveDepthStencil::DepthNop_StencilNop);
				break;
			}

			GraphicsPSOInit.PrimitiveType = PrimitiveType;
			GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI = VertexDeclaration;

			for (const FNaniteTranslucencyBin& TranslucencyBin : Context.TranslucencyBins)
			{
				FNaniteTranslucencyPassData* PassData = TranslucencyBin.RasterPipeline.TranslucencyPassData.Get();
				const FMaterial* Material = PassData->Material;

				const bool bShouldDraw = ShouldDrawInTranslucentBasePass(*Material, TranslucencyPass, View.AutoBeforeDOFTranslucencyBoundary);
				if (!bShouldDraw)
				{
					continue;
				}

			#if WANTS_DRAW_MESH_EVENTS
				SCOPED_CONDITIONAL_DRAW_EVENTF(RHICmdList, Translucency, bShowDrawEvents != 0, TEXT("%s"), Material->GetFriendlyName());
			#endif

				struct FStateCache
				{
					FShaderBindingState ShaderBindings[SF_NumStandardFrequencies];
				} StateCache;

				if (IsMeshShaderRasterPath(HardwarePath))
				{
					GraphicsPSOInit.BoundShaderState.SetMeshShader(PassData->TypedMeshShader.GetMeshShader());
				}
				else
				{
					GraphicsPSOInit.BoundShaderState.VertexShaderRHI = PassData->TypedVertexShader.GetVertexShader();
				}

				GraphicsPSOInit.BoundShaderState.PixelShaderRHI = PassData->TypedPixelShader.GetPixelShader();

				// NOTE: We do *not* use any CullMode overrides here because HWRasterize[VS/MS] already
				// changes the index order in cases where the culling should be flipped.
				// The exception is if CM_None is specified for two sided materials, or if the entire raster pass has CM_None specified.
				const bool bCullModeNone = false;// RasterizerPass.RasterPipeline.bIsTwoSided;
				GraphicsPSOInit.RasterizerState = GetStaticRasterizerState<true /* Enable MSAA */>(FM_Solid, bCullModeNone ? CM_None : CM_CW);

				GraphicsPSOInit.BlendState = GetTranslucentBlendState(*Material, TranslucencyPass);

				const bool bDisableDepthTest = Material->ShouldDisableDepthTest();
				const bool bEnableResponsiveAA = Material->ShouldEnableResponsiveAA();
				const bool bIsPostMotionBlur = Material->IsTranslucencyAfterMotionBlurEnabled();

				// When separate standard translucent are used, we must mark the distortion bit for the composition to happen correctly for any BeforeDoF translucent.
				const bool bSeparatedStandardTranslucent = (TranslucencyPass == ETranslucencyPass::TPT_TranslucencyStandard ||TranslucencyPass == ETranslucencyPass::TPT_AllTranslucency) && IsStandardTranslucencyPassSeparated();

				uint32 StencilRef = 0;
				FRHIDepthStencilState* DepthStencilState = nullptr;

				if (bEnableResponsiveAA && !bIsPostMotionBlur)
				{
					if (bSeparatedStandardTranslucent)
					{
						StencilRef = STENCIL_TEMPORAL_RESPONSIVE_AA_MASK | DISTORTION_STENCIL_MASK_BIT;
						DepthStencilState = GetTranslucentPassDepthStencilState<STENCIL_TEMPORAL_RESPONSIVE_AA_MASK | DISTORTION_STENCIL_MASK_BIT>(bDisableDepthTest);
					}
					else
					{
						StencilRef = STENCIL_TEMPORAL_RESPONSIVE_AA_MASK;
						DepthStencilState = GetTranslucentPassDepthStencilState<STENCIL_TEMPORAL_RESPONSIVE_AA_MASK>(bDisableDepthTest);
					}
				}
				else if (bSeparatedStandardTranslucent)
				{
					StencilRef = DISTORTION_STENCIL_MASK_BIT;
					DepthStencilState = GetTranslucentPassDepthStencilState<DISTORTION_STENCIL_MASK_BIT>(bDisableDepthTest);
				}
				
				if (bDisableDepthTest)
				{
					DepthStencilState = TStaticDepthStencilState<false, CF_Always>::GetRHI();
				}

				GraphicsPSOInit.DepthStencilState = DepthStencilState ? DepthStencilState : PassDepthStencilState;

				GraphicsPSOInit.bAllowVariableRateShading = bAllowHardwareVRS && Material->IsVariableRateShadingAllowed();

				SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit, StencilRef);
				
				if (IsMeshShaderRasterPath(HardwarePath))
				{
					PassData->ShaderBindingsMSPS.SetOnCommandList(RHICmdList, GraphicsPSOInit.BoundShaderState, StateCache.ShaderBindings);

					FRHIBatchedShaderParameters& ShaderParameters = RHICmdList.GetScratchShaderParameters();
					PassData->TypedMeshShader->SetParameters(ShaderParameters, PassData->MaterialProxy, *PassData->Material, View);
					RHICmdList.SetBatchedShaderParameters(GraphicsPSOInit.BoundShaderState.GetMeshShader(), ShaderParameters);
				}
				else
				{
					PassData->ShaderBindingsVSPS.SetOnCommandList(RHICmdList, GraphicsPSOInit.BoundShaderState, StateCache.ShaderBindings);

					FRHIBatchedShaderParameters& ShaderParameters = RHICmdList.GetScratchShaderParameters();
					PassData->TypedVertexShader->SetParameters(ShaderParameters, PassData->MaterialProxy, *PassData->Material, View);
					RHICmdList.SetBatchedShaderParameters(GraphicsPSOInit.BoundShaderState.VertexShaderRHI, ShaderParameters);
				}

				{
					FRHIBatchedShaderParameters& ShaderParameters = RHICmdList.GetScratchShaderParameters();
					PassData->TypedPixelShader->SetParameters(ShaderParameters, PassData->MaterialProxy, *PassData->Material, View);
					RHICmdList.SetBatchedShaderParameters(GraphicsPSOInit.BoundShaderState.PixelShaderRHI, ShaderParameters);
				}

				const FUintVector4 RootConstants(TranslucencyBin.RasterGroup, 0, 0, 0);

				if (GRHISupportsShaderRootConstants)
				{
					RHICmdList.SetShaderRootConstants(RootConstants);
				}

				if (IsMeshShaderRasterPath(HardwarePath))
				{
					RHICmdList.DispatchIndirectMeshShader(PassParameters->NaniteIndirectArgs->GetIndirectRHICallBuffer(), TranslucencyBin.IndirectOffset + 16);
				}
				else
				{
					RHICmdList.DrawPrimitiveIndirect(PassParameters->NaniteIndirectArgs->GetIndirectRHICallBuffer(), TranslucencyBin.IndirectOffset + 16);
				}
			}
		}
	);
}

} // Nanite

