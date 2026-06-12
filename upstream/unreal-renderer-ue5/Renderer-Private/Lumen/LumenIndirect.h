// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Containers/ContainersFwd.h"
#include "Misc/EnumClassFlags.h"
#include "ScreenSpaceDenoise.h"
#include "Lumen/LumenTracingUtils.h"
#include "Lumen/LumenRadianceCacheInterpolation.h"

enum class ELumenIndirectLightingSteps
{
	None = 0,
	ScreenProbeGather = 1u << 0,
	Reflections = 1u << 1,
	Composite = 1u << 3,
	All = ScreenProbeGather | Reflections | Composite
};
ENUM_CLASS_FLAGS(ELumenIndirectLightingSteps)

struct FAsyncLumenIndirectLightingOutputs
{
	struct FViewOutputs
	{
		FSSDSignalTextures IndirectLightingTextures;
		FLumenMeshSDFGridParameters MeshSDFGridParameters;
		LumenRadianceCache::FRadianceCacheInterpolationParameters RadianceCacheParameters;
		FLumenScreenSpaceBentNormalParameters ScreenBentNormalParameters;
		FLumenUpsampleParameters LumenUpsampleParameters;
	};

	TArray<FViewOutputs, TInlineAllocator<1>> ViewOutputs;
	ELumenIndirectLightingSteps StepsLeft = ELumenIndirectLightingSteps::All;
	bool bHasDrawnBeforeLightingDecals = false;

	void Resize(int32 NewNum)
	{
		ViewOutputs.SetNumZeroed(NewNum);
	}

	void DoneAsync(bool bAsyncReflections)
	{
		check(StepsLeft == ELumenIndirectLightingSteps::All);

		EnumRemoveFlags(StepsLeft, ELumenIndirectLightingSteps::ScreenProbeGather);
		if (bAsyncReflections)
		{
			EnumRemoveFlags(StepsLeft, ELumenIndirectLightingSteps::Reflections);
		}
	}

	void DonePreLights()
	{
		if (StepsLeft == ELumenIndirectLightingSteps::All)
		{
			StepsLeft = ELumenIndirectLightingSteps::None;
		}
		else
		{
			StepsLeft = ELumenIndirectLightingSteps::Composite;
		}
	}

	void DoneComposite()
	{
		StepsLeft = ELumenIndirectLightingSteps::None;
	}
};