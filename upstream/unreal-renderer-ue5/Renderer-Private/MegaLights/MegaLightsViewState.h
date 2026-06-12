// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Math/Vector4.h"
#include "RendererInterface.h"
#include "RenderGraphResources.h"

enum class EMegaLightsInput : uint8;
enum class EMegaLightsDebugTarget : uint32;
class FRHIGPUBufferReadback;
class FGPUBufferReadbackCollection;

class FMegaLightsViewState
{
public:
	struct FResources
	{
		TRefCountPtr<IPooledRenderTarget> DiffuseLightingHistory;
		TRefCountPtr<IPooledRenderTarget> SpecularLightingHistory;
		TRefCountPtr<IPooledRenderTarget> LightingMomentsHistory;
		TRefCountPtr<IPooledRenderTarget> NumFramesAccumulatedHistory;
		TRefCountPtr<FRDGPooledBuffer> VisibleLightHashHistory;
		TRefCountPtr<FRDGPooledBuffer> HiddenLightHashHistory;
		TRefCountPtr<FRDGPooledBuffer> VolumeVisibleLightHashHistory;
		TRefCountPtr<FRDGPooledBuffer> VolumeHiddenLightHashHistory;
		TRefCountPtr<FRDGPooledBuffer> TranslucencyVolume0VisibleLightHashHistory;
		TRefCountPtr<FRDGPooledBuffer> TranslucencyVolume1VisibleLightHashHistory;
		TRefCountPtr<FRDGPooledBuffer> TranslucencyVolume0HiddenLightHashHistory;
		TRefCountPtr<FRDGPooledBuffer> TranslucencyVolume1HiddenLightHashHistory;

		// Optionally used, default is StochasticLightingViewState.SceneXxxHistory
		TRefCountPtr<IPooledRenderTarget> SceneDepthHistory;
		TRefCountPtr<IPooledRenderTarget> SceneNormalHistory;

		// Only used when visualizing light complexity
		TRefCountPtr<FRDGPooledBuffer> FrozenLightData;
		TRefCountPtr<FRDGPooledBuffer> GeneralPurposeState;
		FGPUBufferReadbackCollection* GeneralPurposeReadback = nullptr;
		FRHIGPUBufferReadback* DumpLightDataReadback = nullptr;
		uint32 PendingDumpLightDataReadbackSize = 0;
		EMegaLightsDebugTarget LastVisualizeLightComplexityTarget = (EMegaLightsDebugTarget)0;

		FIntPoint HistoryVisibleLightHashViewMinInTiles = 0;
		FIntPoint HistoryVisibleLightHashViewSizeInTiles = 0;

		FIntVector HistoryVolumeVisibleLightHashViewSizeInTiles = FIntVector::ZeroValue;
		FIntVector HistoryTranslucencyVolumeVisibleLightHashSizeInTiles = FIntVector::ZeroValue;

		void SafeRelease();

		uint64 GetGPUSizeBytes(bool bLogSizes) const;
	};

	void SafeRelease()
	{
		GBuffer.SafeRelease();
		HairStrands.SafeRelease();
		FrontLayerTranslucency.SafeRelease();
		PrevLightPower.SafeRelease();
	}

	uint64 GetGPUSizeBytes(bool bLogSizes) const;

	FResources GBuffer;
	FResources HairStrands;
	FResources FrontLayerTranslucency;

	TRefCountPtr<FRDGPooledBuffer> PrevLightPower;
};

