// Copyright Epic Games, Inc. All Rights Reserved.

#include "PostProcess/PostProcessVisualizeCSMDepthBuffers.h"
#include "SceneView.h"
#include "RenderGraphUtils.h"
#include "PixelShaderUtils.h"
#include "CanvasTypes.h"
#include "Engine/Engine.h"
#include "SceneRendering.h"
#include "ShadowRendering.h"
#include "DebugViewModeHelpers.h"

#if WITH_DEBUG_VIEW_MODES

static TAutoConsoleVariable<int32> CVarVisualizeCSMDepthBuffers(
	TEXT("r.Shadow.VisualizeCSMDepthBuffers"),
	0,
	TEXT("Visualize CSM depth buffers as a row of tiles at the top of the screen.\n")
	TEXT("0: disabled \n")
	TEXT("1: enabled"),
	ECVF_RenderThreadSafe);

bool IsVisualizeCSMDepthBuffersEnabled(EShaderPlatform Platform)
{
	if (CVarVisualizeCSMDepthBuffers.GetValueOnRenderThread() == 0)
	{
		return false;
	}

	// For now, this shader is never compiled for mobile devices(Android/iOS) so we don't allow it to run there
	return !IsMobilePlatform(Platform) || FDataDrivenShaderPlatformInfo::GetIsPreviewPlatform(Platform);
}

TArray<FVisualizeCSMDepthBufferTile> CollectVisualizeCSMDepthBufferTiles(TArrayView<const FSortedShadowMapAtlas> ShadowMapAtlases)
{
	int32 FarSplitIndexBase = MAX_int32;
	for (const FSortedShadowMapAtlas& Atlas : ShadowMapAtlases)
	{
		if (!Atlas.RenderTargets.DepthTarget) 
		{
			continue;
		}
		for (const FProjectedShadowInfo* Shadow : Atlas.Shadows)
		{
			if (Shadow->bWholeSceneShadow && Shadow->CascadeSettings.bFarShadowCascade)
				FarSplitIndexBase = FMath::Min(FarSplitIndexBase, Shadow->CascadeSettings.ShadowSplitIndex);
		}
	}
	if (FarSplitIndexBase == MAX_int32) 
	{
		FarSplitIndexBase = 0;
	}

	TArray<FVisualizeCSMDepthBufferTile> Tiles;

	for (const FSortedShadowMapAtlas& Atlas : ShadowMapAtlases)
	{
		if (!Atlas.RenderTargets.DepthTarget) 
		{
			continue;
		}

		for (const FProjectedShadowInfo* Shadow : Atlas.Shadows)
		{
			if (!Shadow->bWholeSceneShadow)
			{
				continue;
			}
			const bool bFar = Shadow->CascadeSettings.bFarShadowCascade;
			if (!bFar && Shadow->CascadeSettings.ShadowSplitIndex < 0)
			{
				continue;
			}

			FVisualizeCSMDepthBufferTile& Tile = Tiles.AddDefaulted_GetRef();
			Tile.DepthTarget  = Atlas.RenderTargets.DepthTarget;
			const int32 ContentX = Shadow->X + (int32)Shadow->BorderSize;
			const int32 ContentY = Shadow->Y + (int32)Shadow->BorderSize;
			Tile.AtlasSubRect = FIntRect(ContentX,
										ContentY,
										ContentX + Shadow->ResolutionX,
										ContentY + Shadow->ResolutionY);

			// When r.Shadow.CSMShadowDepthReuse.SeparateDynamic=1, each split cascade has a static and a dynamic FProjectedShadowInfo sharing the same ShadowSplitIndex 
			// Show both but distinguish them in the label
			const TCHAR* DynSuffix = Shadow->bIsDynamicOnly ? TEXT(" (dynamic)") : TEXT("");
			const int32  DynBias   = Shadow->bIsDynamicOnly ? 1 : 0;

			if (bFar)
			{
				// Order far shadows after all CSM cascades, but preserve relative order between them
				const int32 FarOrder = FMath::Max(0, Shadow->CascadeSettings.ShadowSplitIndex - FarSplitIndexBase);
				Tile.Label   = FString::Printf(TEXT("Far Cascade %d%s"), FarOrder, DynSuffix);
				Tile.SortKey = INT32_MAX - 256 + FarOrder * 2 + DynBias;
			}
			else
			{
				Tile.Label   = FString::Printf(TEXT("Cascade %d%s"), Shadow->CascadeSettings.ShadowSplitIndex, DynSuffix);
				Tile.SortKey = Shadow->CascadeSettings.ShadowSplitIndex * 2 + DynBias;
			}
		}
	}

	Tiles.Sort([](const FVisualizeCSMDepthBufferTile& A, const FVisualizeCSMDepthBufferTile& B)
	{
		return A.SortKey < B.SortKey;
	});
	return Tiles;
}

class FVisualizeCSMDepthBufferPS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FVisualizeCSMDepthBufferPS);
	SHADER_USE_PARAMETER_STRUCT(FVisualizeCSMDepthBufferPS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, AtlasTexture)
		SHADER_PARAMETER_SAMPLER(SamplerState, AtlasSampler)
		SHADER_PARAMETER(FVector4f, UVTransform)    // xy=scale, zw=bias ([0,1] tile-local → atlas UV)
		SHADER_PARAMETER(FVector2f, TileScreenMin)  
		SHADER_PARAMETER(FVector2f, TileScreenMax)
		RENDER_TARGET_BINDING_SLOTS()
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return AllowDebugViewmodes(Parameters.Platform) && (!IsMobilePlatform(Parameters.Platform) || FDataDrivenShaderPlatformInfo::GetIsPreviewPlatform(Parameters.Platform));
	}
};

IMPLEMENT_GLOBAL_SHADER(FVisualizeCSMDepthBufferPS, "/Engine/Private/PostProcessVisualizeCSMDepthBuffers.usf", "MainPS", SF_Pixel);

FScreenPassTexture AddVisualizeCSMDepthBuffersPass(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	FScreenPassTexture                              SceneColor,
	TArrayView<const FVisualizeCSMDepthBufferTile>  Tiles)
{
	if (Tiles.IsEmpty()) 
	{
		return SceneColor;
	}

	const FIntRect ViewRect = SceneColor.ViewRect;
	FScreenPassRenderTarget Output(SceneColor.Texture, ViewRect, ERenderTargetLoadAction::ELoad);

	constexpr int32 TileRowHeightFraction      = 5;
	constexpr int32 GapDefaultDivisor          = 10;
	constexpr int32 MinGapDivisor              = 11;
	constexpr int32 MaxGapDivisor              = 2;

	constexpr int32 LabelLeftInsetDivisor      = 80;
	constexpr int32 LabelLeftInsetMinPx        = 2;
	constexpr int32 LabelBottomInsetDivisor    = 15;
	constexpr int32 LabelBottomInsetMinPx      = 8;

	const int32 NumTiles   = FMath::Max(1, Tiles.Num());
	const int32 SlotWidth  = FMath::Max(1, ViewRect.Width() / NumTiles);
	const int32 Gap        = FMath::Clamp(SlotWidth / GapDefaultDivisor, SlotWidth / MinGapDivisor, SlotWidth / MaxGapDivisor);
	const int32 TileWidth  = SlotWidth - Gap;
	const int32 TileHeight = ViewRect.Height() / TileRowHeightFraction;

	TArray<FIntRect> TileRects;
	TileRects.Reserve(NumTiles);
	for (int32 i = 0; i < NumTiles; ++i)
	{
		const int32 TileOriginX = ViewRect.Min.X + i * SlotWidth + Gap / 2;
		TileRects.Add(FIntRect(TileOriginX, ViewRect.Min.Y, TileOriginX + TileWidth, ViewRect.Min.Y + TileHeight));
	}

	TMap<IPooledRenderTarget*, FRDGTextureRef> RegisteredAtlases;

	const int32 NumToRender = Tiles.Num();
	for (int32 i = 0; i < NumToRender; ++i)
	{
		const FVisualizeCSMDepthBufferTile& Tile = Tiles[i];
		const FIntRect& TileRect = TileRects[i];

		FRDGTextureRef* Cached = RegisteredAtlases.Find(Tile.DepthTarget.GetReference());
		FRDGTextureRef AtlasTex = Cached
			? *Cached
			: RegisteredAtlases.Add(Tile.DepthTarget.GetReference(), GraphBuilder.RegisterExternalTexture(Tile.DepthTarget));

		const FIntPoint AtlasSize = AtlasTex->Desc.Extent;

		// Maps tile-local [0,1] UV → cascade sub-rect UV in the atlas
		const FVector4f UVTransform(
			float(Tile.AtlasSubRect.Width()) / AtlasSize.X,
			float(Tile.AtlasSubRect.Height()) / AtlasSize.Y,
			float(Tile.AtlasSubRect.Min.X) / AtlasSize.X,
			float(Tile.AtlasSubRect.Min.Y) / AtlasSize.Y);

		auto* Params = GraphBuilder.AllocParameters<FVisualizeCSMDepthBufferPS::FParameters>();
		Params->AtlasTexture = AtlasTex;
		Params->AtlasSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp>::GetRHI();
		Params->UVTransform = UVTransform;
		Params->TileScreenMin = FVector2f(TileRect.Min);
		Params->TileScreenMax = FVector2f(TileRect.Max);
		Params->RenderTargets[0] = Output.GetRenderTargetBinding();

		FPixelShaderUtils::AddFullscreenPass(
			GraphBuilder,
			View.ShaderMap,
			RDG_EVENT_NAME("VisualizeCSMDepthBuffer[%s]", *Tile.Label),
			TShaderMapRef<FVisualizeCSMDepthBufferPS>(View.ShaderMap),
			Params,
			TileRect);
	}


	TArray<FVisualizeCSMDepthBufferTile> TilesCopy(Tiles);
	TArray<FIntRect> RectsCopy(TileRects.GetData(), NumToRender);

	AddDrawCanvasPass(GraphBuilder, RDG_EVENT_NAME("CSMDepthBufferLabels"), View, Output,
		[TilesCopy = MoveTemp(TilesCopy), RectsCopy = MoveTemp(RectsCopy)](FCanvas& Canvas)
		{
			for (int32 i = 0; i < RectsCopy.Num(); ++i)
			{
				const FIntRect& CurrentTileRect = RectsCopy[i];
				const int32 LabelLeftInset      = FMath::Max(LabelLeftInsetMinPx,   CurrentTileRect.Width()  / LabelLeftInsetDivisor);
				const int32 LabelBottomInset    = FMath::Max(LabelBottomInsetMinPx, CurrentTileRect.Height() / LabelBottomInsetDivisor);
				Canvas.DrawShadowedString(
					CurrentTileRect.Min.X + LabelLeftInset,
					CurrentTileRect.Max.Y - LabelBottomInset,
					*TilesCopy[i].Label,
					GEngine->GetSmallFont(),
					FLinearColor::White);
			}
		});

	return MoveTemp(SceneColor);
}

#endif // WITH_DEBUG_VIEW_MODES
