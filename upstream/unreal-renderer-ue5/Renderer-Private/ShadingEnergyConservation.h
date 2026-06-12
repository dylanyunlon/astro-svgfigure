// Copyright Epic Games, Inc. All Rights Reserved.

/*=============================================================================
	ShadingEnergyConservation.h: private energy conservation related data
=============================================================================*/

#pragma once

#include "CoreMinimal.h"
#include "RendererInterface.h"

class FViewInfo;

struct FShadingEnergyConservationData
{
	bool bEnergyConservation = false;
	bool bEnergyPreservation = false;

	TRefCountPtr<IPooledRenderTarget> GGXSpecEnergyTexture = nullptr;
	TRefCountPtr<IPooledRenderTarget> GGXGlassEnergyTexture = nullptr;
	TRefCountPtr<IPooledRenderTarget> ClothEnergyTexture = nullptr;
	TRefCountPtr<IPooledRenderTarget> DiffuseEnergyTexture = nullptr;
	TRefCountPtr<IPooledRenderTarget> GGXMaxSpecEnergyTexture = nullptr;
};

namespace ShadingEnergyConservation
{
	void Init(FRDGBuilder& GraphBuilder, TArray<FViewInfo>& Views);
	void Debug(FRDGBuilder& GraphBuilder, const FViewInfo& View, FSceneTextures& SceneTextures);
	FShadingEnergyConservationData GetData(const FViewInfo& View);
}
