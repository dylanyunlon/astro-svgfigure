// Copyright Epic Games, Inc. All Rights Reserved.

#include "PostProcess/PostProcessVisualizeCSMOverlay.h"
#include "ScenePrivate.h"
#include "SceneView.h"
#include "RenderGraphUtils.h"
#include "PixelShaderUtils.h"
#include "SceneRendering.h"
#include "ShadowRendering.h"
#include "Shadows/CSMSceneExtension.h"
#include "ScreenPass.h"
#include "RHIStaticStates.h"
#include "CanvasTypes.h"
#include "Engine/Engine.h"
#include "DebugViewModeHelpers.h"

#if WITH_DEBUG_VIEW_MODES

static TAutoConsoleVariable<int32> CVarVisualizeCSMOverlay(
	TEXT("r.Shadow.VisualizeCSMOverlay"),
	0,
	TEXT("Visualizes CSM + far shadow cascades with a color overlay per cascade.\n")
	TEXT(" 0: Off (default)\n")
	TEXT(" 1: On  - CSM 0..3 = Red/Yellow/Green/Blue, Far 0..3 = Cyan/Orange/Violet/White, Pink = Beyond all cascades\n"),
	ECVF_RenderThreadSafe);

bool IsVisualizeCSMOverlayEnabled(EShaderPlatform Platform)
{
	if (CVarVisualizeCSMOverlay.GetValueOnRenderThread() == 0)
	{
		return false;
	}

	// For now, this shader is never compiled for mobile devices(Android/iOS) so we don't allow it to run there
	return !IsMobilePlatform(Platform) || FDataDrivenShaderPlatformInfo::GetIsPreviewPlatform(Platform);
}

class FVisualizeCSMOverlayPS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FVisualizeCSMOverlayPS);
	SHADER_USE_PARAMETER_STRUCT(FVisualizeCSMOverlayPS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, ViewUniformBuffer)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FForwardDirectionalLightShadowMapParameters, ForwardDirLightShadowStruct)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, SceneDepthTexture)
		SHADER_PARAMETER_SAMPLER(SamplerState, SceneDepthSampler)
		SHADER_PARAMETER(FScreenTransform, PassSvPositionToViewportUV)
		SHADER_PARAMETER(FScreenTransform, ViewportUVToDepthUV)
		SHADER_PARAMETER(FVector4f, CascadeAlphaFractions)
		SHADER_PARAMETER(FVector4f, FarCascadeAlphaFractions)
		SHADER_PARAMETER(FVector3f, FrozenTranslatedViewOrigin)
		SHADER_PARAMETER(FVector3f, FrozenViewForward)
		SHADER_PARAMETER(FVector4f, FrozenSplitFarValues)
		SHADER_PARAMETER(FVector4f, FarFrozenSplitFarValues)
		SHADER_PARAMETER_ARRAY(FMatrix44f, LiveToFrozenClipMatrices, [4])
		SHADER_PARAMETER_ARRAY(FMatrix44f, LiveToFrozenFarClipMatrices, [4])
		RENDER_TARGET_BINDING_SLOTS()
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return AllowDebugViewmodes(Parameters.Platform) && (!IsMobilePlatform(Parameters.Platform) || FDataDrivenShaderPlatformInfo::GetIsPreviewPlatform(Parameters.Platform));
	}
};

IMPLEMENT_GLOBAL_SHADER(FVisualizeCSMOverlayPS, "/Engine/Private/PostProcessVisualizeCSMOverlay.usf", "MainPS", SF_Pixel);

void RenderVisualizeCSMOverlay( FRDGBuilder& GraphBuilder, TArrayView<FViewInfo> Views, const FMinimalSceneTextures& SceneTextures,	const FScene* Scene, const FVisibleLightInfoArray& VisibleLightInfos)
{
	RDG_EVENT_SCOPE(GraphBuilder, "VisualizeCSMOverlay");

	FRHIBlendState* BlendState = TStaticBlendState<CW_RGBA, BO_Add, BF_SourceAlpha, BF_InverseSourceAlpha, BO_Add, BF_Zero, BF_One>::GetRHI();
	FVector4f CascadeAlphaFractions(1.0f, 1.0f, 1.0f, 1.0f);
	FVector4f FarCascadeAlphaFractions(1.0f, 1.0f, 1.0f, 1.0f);

	struct FCSMFrozenCascadeEntry
	{
		float      FrozenSplitFar                  = 0.0f;
		FVector    ViewOrigin                      = FVector::ZeroVector;
		FVector    ViewForward                     = FVector::ZeroVector;
		FVector    PreShadowTranslation            = FVector::ZeroVector;
		FMatrix44f TranslatedWorldToClipInnerMatrix = FMatrix44f::Identity;
	};
	TStaticArray<FCSMFrozenCascadeEntry, 4> FrozenCascadeEntries;
	TStaticArray<FCSMFrozenCascadeEntry, 4> FrozenFarCascadeEntries;

	if (Scene && CVarCSMShadowDepthReuse.GetValueOnRenderThread())
	{
		const int32 StaticInterval = FMath::Max(1, CVarCSMShadowDepthReuseInterval.GetValueOnRenderThread());
		// Far cascades fade on the interval the renderer actually uses for their static pass: FarInterval when set (>=0), otherwise the near Interval
		const int32 FarIntervalCVar = CVarCSMShadowDepthReuseFarInterval.GetValueOnRenderThread();
		const int32 FarInterval = FMath::Max(1, FarIntervalCVar >= 0 ? FarIntervalCVar : CVarCSMShadowDepthReuseInterval.GetValueOnRenderThread());
		const uint32 CurrentFrame = Scene->GetFrameNumberRenderThread();

		const FCSMSceneExtension& CSMExt = Scene->GetExtension<FCSMSceneExtension>();
		const TMap<FLightSceneId, TArray<FCSMDepthReuse>>& AllDepthReuseData = CSMExt.GetAllCSMDepthReuseData();

		if (Scene->DirectionalLights.Num() > 0)
		{
			const FLightSceneId LightId = Scene->DirectionalLights[0]->Id;
			const TArray<FCSMDepthReuse>* DepthReuseArrayPtr = AllDepthReuseData.Find(LightId);

			if (DepthReuseArrayPtr && VisibleLightInfos.IsValidIndex(LightId))
			{
				const TArray<FCSMDepthReuse>& DepthReuseArray = *DepthReuseArrayPtr;

				for (const FProjectedShadowInfo* ShadowInfo : VisibleLightInfos[LightId].AllProjectedShadows)
				{
					if (!ShadowInfo->IsWholeSceneDirectionalShadow()
						|| ShadowInfo->HasVirtualShadowMap()
						|| !ShadowInfo->bAllocated
						|| ShadowInfo->CascadeSettings.bFarShadowCascade
						|| ShadowInfo->bIsDynamicOnly)
					{
						continue;
					}

					const int32 CascadeIdx = ShadowInfo->CascadeSettings.ShadowSplitIndex;
					if (CascadeIdx < 0 || CascadeIdx >= 4)
					{
						continue;
					}

					const int32 ReuseIndex = ShadowInfo->GetCSMShadowDepthReuseIndex();
					if (!DepthReuseArray.IsValidIndex(ReuseIndex))
					{
						continue;
					}

					const FCSMDepthReuse& DepthReuse = DepthReuseArray[ReuseIndex];
					if (!DepthReuse.PreviousAtlas.IsValid() || DepthReuse.FrozenSplitFar <= 0.0f)
					{
						continue;
					}

					FrozenCascadeEntries[CascadeIdx].FrozenSplitFar                   = DepthReuse.FrozenSplitFar;
					FrozenCascadeEntries[CascadeIdx].ViewOrigin                       = DepthReuse.FrozenViewOrigin;
					FrozenCascadeEntries[CascadeIdx].ViewForward                      = DepthReuse.FrozenViewForward;
					FrozenCascadeEntries[CascadeIdx].PreShadowTranslation             = DepthReuse.PreShadowTranslation;
					FrozenCascadeEntries[CascadeIdx].TranslatedWorldToClipInnerMatrix = DepthReuse.TranslatedWorldToClipInnerMatrix;

					if (StaticInterval > 1)
					{
						// Alpha fade: 1.0 = just rendered, 0.0 = about to re-render (i.e stale).
						const uint32 FrameAge        = CurrentFrame - DepthReuse.PreviousRenderDepthFrameNumber;
						const uint32 FramesUsed      = FrameAge % (uint32)StaticInterval;
						const uint32 FramesRemaining = (uint32)StaticInterval - FramesUsed;
						CascadeAlphaFractions[CascadeIdx] = float(FramesRemaining - 1) / float(StaticInterval - 1);
					}
				}
			}

			if (DepthReuseArrayPtr && VisibleLightInfos.IsValidIndex(LightId))
			{
				const TArray<FCSMDepthReuse>& DepthReuseArray = *DepthReuseArrayPtr;

				int32 FarShadowSplitIndexBase = MAX_int32;
				for (const FProjectedShadowInfo* ShadowInfo : VisibleLightInfos[LightId].AllProjectedShadows)
				{
					if (ShadowInfo->IsWholeSceneDirectionalShadow()
						&& !ShadowInfo->HasVirtualShadowMap()
						&& ShadowInfo->bAllocated
						&& ShadowInfo->CascadeSettings.bFarShadowCascade
						&& !ShadowInfo->bIsDynamicOnly)
					{
						FarShadowSplitIndexBase = FMath::Min(FarShadowSplitIndexBase, ShadowInfo->CascadeSettings.ShadowSplitIndex);
					}
				}

				for (const FProjectedShadowInfo* ShadowInfo : VisibleLightInfos[LightId].AllProjectedShadows)
				{
					if (ShadowInfo->IsWholeSceneDirectionalShadow()
						&& !ShadowInfo->HasVirtualShadowMap()
						&& ShadowInfo->bAllocated
						&& ShadowInfo->CascadeSettings.bFarShadowCascade
						&& !ShadowInfo->bIsDynamicOnly)
					{
						const int32 FarIdx = ShadowInfo->CascadeSettings.ShadowSplitIndex - FarShadowSplitIndexBase;
						if (FarIdx < 0 || FarIdx >= 4)
						{
							continue;
						}
						const int32 ReuseIndex = ShadowInfo->GetCSMShadowDepthReuseIndex();
						if (!DepthReuseArray.IsValidIndex(ReuseIndex))
						{
							continue;
						}
						const FCSMDepthReuse& DepthReuse = DepthReuseArray[ReuseIndex];
						if (!DepthReuse.PreviousAtlas.IsValid() || DepthReuse.FrozenSplitFar <= 0.0f)
						{
							continue;
						}

						FrozenFarCascadeEntries[FarIdx].FrozenSplitFar                   = DepthReuse.FrozenSplitFar;
						FrozenFarCascadeEntries[FarIdx].ViewOrigin                       = DepthReuse.FrozenViewOrigin;
						FrozenFarCascadeEntries[FarIdx].ViewForward                      = DepthReuse.FrozenViewForward;
						FrozenFarCascadeEntries[FarIdx].PreShadowTranslation             = DepthReuse.PreShadowTranslation;
						FrozenFarCascadeEntries[FarIdx].TranslatedWorldToClipInnerMatrix = DepthReuse.TranslatedWorldToClipInnerMatrix;

						if (FarInterval > 1)
						{
							// Alpha fade: 1.0 = just rendered, 0.0 = about to re-render (i.e stale)
							const uint32 FrameAge        = CurrentFrame - DepthReuse.PreviousRenderDepthFrameNumber;
							const uint32 FramesUsed      = FrameAge % (uint32)FarInterval;
							const uint32 FramesRemaining = (uint32)FarInterval - FramesUsed;
							FarCascadeAlphaFractions[FarIdx] = float(FramesRemaining - 1) / float(FarInterval - 1);
						}
					}
				}
			}
		}
	}

	for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ++ViewIndex)
	{
		const FViewInfo& View = Views[ViewIndex];

		FVector3f FrozenTranslatedViewOrigin(0.0f);
		FVector3f FrozenViewFwd(0.0f);
		FVector4f FrozenSplitFarValues(0.0f, 0.0f, 0.0f, 0.0f);
		FVector4f FarFrozenSplitFarValues(0.0f, 0.0f, 0.0f, 0.0f);
		FMatrix44f LiveToFrozenClipMatrices[4]    = { FMatrix44f::Identity, FMatrix44f::Identity, FMatrix44f::Identity, FMatrix44f::Identity };
		FMatrix44f LiveToFrozenFarClipMatrices[4] = { FMatrix44f::Identity, FMatrix44f::Identity, FMatrix44f::Identity, FMatrix44f::Identity };
		{
			const FVector PreViewTranslation = View.ViewMatrices.GetPreViewTranslation();
			bool bHasFrozenOrigin = false;

			auto BuildLiveToFrozenClip = [&PreViewTranslation](const FCSMFrozenCascadeEntry& Entry)
			{
				const FVector TranslationDelta = Entry.PreShadowTranslation - PreViewTranslation;
				return FMatrix44f(FTranslationMatrix(TranslationDelta) * FMatrix(Entry.TranslatedWorldToClipInnerMatrix));
			};

			for (int32 CascadeIdx = 0; CascadeIdx < 4; ++CascadeIdx)
			{
				const FCSMFrozenCascadeEntry& Entry = FrozenCascadeEntries[CascadeIdx];
				if (Entry.FrozenSplitFar > 0.0f)
				{
					if (!bHasFrozenOrigin)
					{
						FrozenTranslatedViewOrigin = FVector3f(Entry.ViewOrigin + PreViewTranslation);
						FrozenViewFwd              = FVector3f(Entry.ViewForward);
						bHasFrozenOrigin           = true;
					}
					FrozenSplitFarValues[CascadeIdx] = Entry.FrozenSplitFar;
					LiveToFrozenClipMatrices[CascadeIdx] = BuildLiveToFrozenClip(Entry);
				}
			}

			for (int32 FarCascadeIdx = 0; FarCascadeIdx < 4; ++FarCascadeIdx)
			{
				const FCSMFrozenCascadeEntry& Entry = FrozenFarCascadeEntries[FarCascadeIdx];
				if (Entry.FrozenSplitFar > 0.0f)
				{
					// Fall back to far cascade frozen view if no CSM cascade was frozen yet.
					if (!bHasFrozenOrigin)
					{
						FrozenTranslatedViewOrigin = FVector3f(Entry.ViewOrigin + PreViewTranslation);
						FrozenViewFwd              = FVector3f(Entry.ViewForward);
						bHasFrozenOrigin           = true;
					}
					FarFrozenSplitFarValues[FarCascadeIdx]    = Entry.FrozenSplitFar;
					LiveToFrozenFarClipMatrices[FarCascadeIdx] = BuildLiveToFrozenClip(Entry);
				}
			}
		}

		FVisualizeCSMOverlayPS::FParameters* PassParameters = GraphBuilder.AllocParameters<FVisualizeCSMOverlayPS::FParameters>();

		PassParameters->ViewUniformBuffer = View.ViewUniformBuffer;
		PassParameters->ForwardDirLightShadowStruct = View.ForwardLightingResources.ForwardDirLightShadowUniformBuffer
			? View.ForwardLightingResources.ForwardDirLightShadowUniformBuffer
			: CreateDummyForwardDirLightShadowUniformBuffer(GraphBuilder);
		PassParameters->SceneDepthTexture = SceneTextures.Depth.Resolve;
		PassParameters->SceneDepthSampler = TStaticSamplerState<SF_Point>::GetRHI();
		PassParameters->PassSvPositionToViewportUV = FScreenTransform::SvPositionToViewportUV(View.ViewRect);
		PassParameters->ViewportUVToDepthUV = FScreenTransform::ChangeTextureBasisFromTo(
			FScreenPassTextureViewport(SceneTextures.Depth.Resolve, View.ViewRect),
			FScreenTransform::ETextureBasis::ViewportUV,
			FScreenTransform::ETextureBasis::TextureUV);
		PassParameters->CascadeAlphaFractions      = CascadeAlphaFractions;
		PassParameters->FarCascadeAlphaFractions   = FarCascadeAlphaFractions;
		PassParameters->FrozenTranslatedViewOrigin = FrozenTranslatedViewOrigin;
		PassParameters->FrozenViewForward          = FrozenViewFwd;
		PassParameters->FrozenSplitFarValues       = FrozenSplitFarValues;
		PassParameters->FarFrozenSplitFarValues    = FarFrozenSplitFarValues;
		for (int32 K = 0; K < 4; ++K)
		{
			PassParameters->LiveToFrozenClipMatrices[K]    = LiveToFrozenClipMatrices[K];
			PassParameters->LiveToFrozenFarClipMatrices[K] = LiveToFrozenFarClipMatrices[K];
		}
		PassParameters->RenderTargets[0] = FRenderTargetBinding(SceneTextures.Color.Target, ERenderTargetLoadAction::ELoad);

		TShaderMapRef<FVisualizeCSMOverlayPS> PixelShader(View.ShaderMap);
		FPixelShaderUtils::AddFullscreenPass<FVisualizeCSMOverlayPS>(
			GraphBuilder, View.ShaderMap,
			RDG_EVENT_NAME("VisualizeCSMOverlay View:%d", ViewIndex),
			PixelShader, PassParameters, View.ViewRect, BlendState);

		// Diagnostic snapshot of the GPU-bound shadow params to show as part of an on screen legend
		uint32 DiagNumCascades = 0;
		FVector4f DiagCascadeEndDepths(0.0f);
		uint32 DiagNumFarCascades = 0;
		FVector4f DiagFarShadowEndDepths(0.0f);
		if (const FForwardDirectionalLightShadowMapParameters* DirShadowParams = View.ForwardLightingResources.ForwardDirLightShadowParameters)
		{
			DiagNumCascades = DirShadowParams->NumDirectionalLightCascades;
			DiagCascadeEndDepths = DirShadowParams->CascadeEndDepths;
			DiagNumFarCascades = DirShadowParams->NumFarShadowCascades;
			DiagFarShadowEndDepths = DirShadowParams->FarShadowEndDepths;
		}

		const FScreenPassRenderTarget LegendOutput(SceneTextures.Color.Target, View.UnscaledViewRect, ERenderTargetLoadAction::ELoad);
		AddDrawCanvasPass(GraphBuilder, RDG_EVENT_NAME("CSMOverlayLegend View:%d", ViewIndex), View, LegendOutput,
			[DiagNumCascades, DiagCascadeEndDepths, DiagNumFarCascades, DiagFarShadowEndDepths](FCanvas& Canvas)
			{
				static const FLinearColor Colors[9] = 
				{
					FLinearColor(1, 0, 0),    FLinearColor(1, 1, 0),   FLinearColor(0, 1, 0),    FLinearColor(0, 0, 1),
					FLinearColor(0, 1, 1),    FLinearColor(1, 0.5, 0), FLinearColor(0.5, 0, 1),  FLinearColor(1, 1, 1),
					FLinearColor(1, 0, 1)
				};
				static const TCHAR* Labels[9] = 
				{
					TEXT("CSM Cascade 0"), TEXT("CSM Cascade 1"), TEXT("CSM Cascade 2"), TEXT("CSM Cascade 3"),
					TEXT("Far Cascade 0"), TEXT("Far Cascade 1"), TEXT("Far Cascade 2"), TEXT("Far Cascade 3"),
					TEXT("Beyond Cascades")
				};

				UFont* Font = GEngine->GetSmallFont();
				const int32 LineHeight = 16;
				for (int32 i = 0; i < 9; ++i)
				{
					Canvas.DrawShadowedString(20, 20 + i * LineHeight, Labels[i], Font, Colors[i]);
				}

				// Unused cascade slots are initialized to MAX_FLT but display them as 0.0 to keep the legend readable
				auto SanitizeForDisplay = [](float V) { return V >= MAX_FLT ? 0.0f : V; };

				const FString DiagLine1 = FString::Printf(TEXT("NumDirCascades=%u  NumFarCascades=%u"), DiagNumCascades, DiagNumFarCascades);
				const FString DiagLine2 = FString::Printf(TEXT("CSMEndDepths=[%.1f, %.1f, %.1f, %.1f]"),
					SanitizeForDisplay(DiagCascadeEndDepths.X), SanitizeForDisplay(DiagCascadeEndDepths.Y),
					SanitizeForDisplay(DiagCascadeEndDepths.Z), SanitizeForDisplay(DiagCascadeEndDepths.W));
				const FString DiagLine3 = FString::Printf(TEXT("FarShadowEndDepths=[%.1f, %.1f, %.1f, %.1f]"),
					SanitizeForDisplay(DiagFarShadowEndDepths.X), SanitizeForDisplay(DiagFarShadowEndDepths.Y),
					SanitizeForDisplay(DiagFarShadowEndDepths.Z), SanitizeForDisplay(DiagFarShadowEndDepths.W));
				Canvas.DrawShadowedString(20, 20 + 9 * LineHeight + 8,  *DiagLine1, Font, FLinearColor::White);
				Canvas.DrawShadowedString(20, 20 + 10 * LineHeight + 8, *DiagLine2, Font, FLinearColor::White);
				Canvas.DrawShadowedString(20, 20 + 11 * LineHeight + 8, *DiagLine3, Font, FLinearColor::White);
			});
	}
}

#endif // WITH_DEBUG_VIEW_MODES
