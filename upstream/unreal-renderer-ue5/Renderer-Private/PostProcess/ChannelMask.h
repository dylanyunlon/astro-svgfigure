// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "ScreenPass.h"

namespace ChannelMask
{
	struct FChannelMaskInputs
	{
		// Input scene color
		FScreenPassTexture SceneColor;

		// Optional output render target, if not provided a new render target is created and returned
		FScreenPassRenderTarget OverrideOutput;
	};
	
	FScreenPassTexture AddChannelMaskPass(class FRDGBuilder& GraphBuilder, const class FViewInfo& View, const struct FChannelMaskInputs& Inputs);
}
