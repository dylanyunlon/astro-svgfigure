// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once
class FRDGBuilder;
class FScene;
class FSceneView;
class FViewInfo;
struct FMinimalSceneTextures;

#include "Containers/ArrayView.h"
#include "SceneRendering.h"
#include "DebugViewModeHelpers.h"

#if WITH_DEBUG_VIEW_MODES
bool IsVisualizeCSMOverlayEnabled(EShaderPlatform Platform);

// Renders a fullscreen overlay coloring pixels by which CSM cascade they fall into
// An alpha fade is used to indicate depth-reuse staleness
void RenderVisualizeCSMOverlay(
	FRDGBuilder& GraphBuilder,
	TArrayView<FViewInfo> Views,
	const FMinimalSceneTextures& SceneTextures,
	const FScene* Scene,
	const FVisibleLightInfoArray& VisibleLightInfos);
#else
inline bool IsVisualizeCSMOverlayEnabled(EShaderPlatform) { return false; }
inline void RenderVisualizeCSMOverlay(FRDGBuilder&, TArrayView<FViewInfo>, const FMinimalSceneTextures&, const FScene*, const FVisibleLightInfoArray&) {}
#endif // WITH_DEBUG_VIEW_MODES

inline bool ShouldVisualizeCSMOverlay(const FSceneView& View)
{
	return IsVisualizeCSMOverlayEnabled(View.GetShaderPlatform()) && View.Family->EngineShowFlags.DynamicShadows;
}
