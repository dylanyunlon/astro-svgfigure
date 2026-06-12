// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "ScreenPass.h"

class FViewInfo;
class FVirtualShadowMapUniformParameters;

struct FVisualizeShadowCastersInputs
{
	FScreenPassRenderTarget OverrideOutput;
	FScreenPassTexture SceneColor;
	FRDGTextureRef DebugAuxTexture = nullptr;
	FRDGTextureRef NaniteVisBuffer64 = nullptr;
	FRDGBufferRef NaniteVisibleClustersSWHW = nullptr;
	FIntVector4 NanitePageConstants = FIntVector4(ForceInitToZero);
	// Only consumed by the (non-far) Shadow Casters VSM rich classifier; leave null for Far Shadow Casters.
	TRDGUniformBufferRef<FVirtualShadowMapUniformParameters> VirtualShadowMap = nullptr;
	bool bFarShadowCasters = false;
};

FScreenPassTexture AddVisualizeShadowCastersPass(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	const FVisualizeShadowCastersInputs& Inputs);
