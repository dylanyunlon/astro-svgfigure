// Copyright Epic Games, Inc. All Rights Reserved.

#include "PostProcess/PostProcessing.h"
#include "ScreenPass.h"
#include "PixelShaderUtils.h"
#include "UnrealEngine.h"
#include "VisualizeTexture.h"
#include "SceneRendering.h"
#include "Engine/ToonProfile.h"


FScreenPassTexture AddToonProfileAtlasVisualizationPasses(FRDGBuilder& GraphBuilder, const FViewInfo& View, FScreenPassTexture& ScreenPassSceneColor, FScreenPassRenderTarget& OverrideOutput)
{
	FScreenPassRenderTarget OutputRT = OverrideOutput;

#if WITH_EDITOR

	if (!OutputRT.IsValid())
	{
		OutputRT = FScreenPassRenderTarget::CreateFromInput(GraphBuilder, ScreenPassSceneColor, View.GetOverwriteLoadAction(), TEXT("VisualizeToonProfileAtlas"));
	}

	const FIntPoint EntryStart = FIntPoint(128, 128);
	int32 EntryOffset = 140;
	const FIntPoint EntryPixelSize = FIntPoint(128, 128);

	ToonProfile::FToonProfileAtlasVisualizationData VizData = ToonProfile::GetToonProfileAtlasVisualizationData();

	FIntPoint AtlasResolution = FIntPoint::ZeroValue;
	if (VizData.ToonProfileAtlasTexture)
	{
		AtlasResolution = VizData.ToonProfileAtlasTexture->GetDesc().Extent;
		EntryOffset = AtlasResolution.X * 2 + 10;
		FRDGTextureRef ToonAtlasTextureRDG = GraphBuilder.RegisterExternalTexture(VizData.ToonProfileAtlasTexture);

		FIntRect SavedViewRect = OutputRT.ViewRect;

		for (auto& AtlasEntry : VizData.ToonProfileEntries)
		{
			FScreenPassTextureSlice TextureSrc;
			TextureSrc.TextureSRV = GraphBuilder.CreateSRV(FRDGTextureSRVDesc::CreateForSlice(ToonAtlasTextureRDG, AtlasEntry.SliceIndex));
			TextureSrc.ViewRect = FIntRect(FIntPoint::ZeroValue, ToonAtlasTextureRDG->Desc.Extent);

			const FIntPoint EntryPixelOrigin = FIntPoint(EntryStart.X + EntryOffset * AtlasEntry.SliceIndex, EntryStart.Y);

			OutputRT.ViewRect = FIntRect(EntryPixelOrigin, EntryPixelOrigin + AtlasResolution*2);

			AddDrawTexturePass(
				GraphBuilder,
				View,
				TextureSrc,
				OutputRT);
		}

		OutputRT.ViewRect = SavedViewRect;
	}


		// Now debug print
	AddDrawCanvasPass(GraphBuilder, {}, View, FScreenPassRenderTarget(OutputRT, ERenderTargetLoadAction::ELoad),
		[&View, EntryStart, EntryOffset, EntryPixelSize, VizData, AtlasResolution](FCanvas& Canvas)
		{
			FString Text;

			const float ViewPortWidth = float(View.ViewRect.Width());
			const float ViewPortHeight = float(View.ViewRect.Height());

			const float DPIScale = Canvas.GetDPIScale();
			Canvas.SetBaseTransform(FMatrix(FScaleMatrix(DPIScale) * Canvas.CalcBaseTransform2D(Canvas.GetViewRect().Width(), Canvas.GetViewRect().Height())));

			Canvas.DrawShadowedString(180.0f, 90.0f, TEXT("TOON PROFILE ATLAS"), GEngine->GetLargeFont(), FLinearColor::White);

			for (auto& AtlasEntry : VizData.ToonProfileEntries)
			{
				const FIntPoint EntryPixelOrigin = FIntPoint(EntryStart.X + EntryOffset * AtlasEntry.SliceIndex, EntryStart.Y - 15);

				Text = FString::Printf(TEXT("#%d = %s"), AtlasEntry.SliceIndex, *AtlasEntry.AssetName);
				Canvas.DrawShadowedString(EntryPixelOrigin.X, EntryPixelOrigin.Y, Text, GEngine->GetLargeFont(), FLinearColor::White);
			}

		});

#endif // WITH_EDITOR

	return MoveTemp(OutputRT);
}
