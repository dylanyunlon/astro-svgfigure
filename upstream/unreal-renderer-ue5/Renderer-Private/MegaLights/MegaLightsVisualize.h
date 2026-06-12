// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "RenderGraphFwd.h"
#include "SceneTexturesConfig.h"
#include "ScreenPass.h"
#include "ShaderParameterMacros.h"

struct FMegaLightsFrameTemporaries;

namespace MegaLights
{
	FScreenPassTexture AddVisualizePass(
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		int32 ViewIndex,
		FScreenPassTexture SceneColor,
		FScreenPassTexture SceneDepth,
		FScreenPassRenderTarget OverrideOutput,
		const TSharedPtr<FMegaLightsFrameTemporaries>& MegaLightsFrameTemporaries);
}