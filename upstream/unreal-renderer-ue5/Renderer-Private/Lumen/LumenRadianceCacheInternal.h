// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Lumen/LumenRadianceCache.h"
#include "Lumen/LumenViewState.h"

namespace LumenRadianceCache
{
	// Must match *.usf
	const int32 TRACE_TILE_SIZE_2D = 8;
	const int32 TRACE_TILE_ATLAS_STRITE_IN_TILES = 512;

	BEGIN_SHADER_PARAMETER_STRUCT(FRadianceCacheUpdateParameters, )
		SHADER_PARAMETER(FIntPoint, TempProbeAtlasSizeInProbes)
		SHADER_PARAMETER(uint32, TempProbeAtlasResolutionModuloMask)
		SHADER_PARAMETER(uint32, TempProbeAtlasResolutionDivideShift)
		SHADER_PARAMETER(uint32, FinalRadianceProbeBorder)
	END_SHADER_PARAMETER_STRUCT()

	class FRadianceCacheSetup
	{
	public:
		TArray<FRadianceCacheClipmap> LastFrameClipmaps;
		FRDGTextureRef DepthProbeAtlasTexture;
		FRDGTextureRef FinalIrradianceAtlas;
		FRDGTextureRef ProbeOcclusionAtlas;
		FRDGTextureRef FinalRadianceAtlas;
		FRDGTextureRef FinalSkyVisibilityAtlas;
		FRDGTextureRef RadianceProbeAtlasTextureSource;
		FRDGTextureRef SkyVisibilityProbeAtlasTextureSource;

		FRDGTextureRef TempRadianceAtlas = nullptr;
		FRDGTextureRef TempSkyVisibilityAtlas = nullptr;

		FRadianceCacheUpdateParameters UpdateParameters;

		bool bPersistentCache;
		bool bAdaptiveProbes;
	};

	void RenderLumenHardwareRayTracingRadianceCache(
		FRDGBuilder& GraphBuilder,
		const FScene* Scene,
		const FLumenSceneFrameTemporaries& FrameTemporaries,
		const TInlineArray<FUpdateInputs>& InputArray,
		TInlineArray<FUpdateOutputs>& OutputArray,
		const TInlineArray<FRadianceCacheSetup>& SetupOutputArray,
		const TInlineArray<FRDGBufferRef>& ProbeTraceTileAllocatorArray,
		const TInlineArray<FRDGBufferRef>& ProbeTraceTileDataArray,
		const TInlineArray<FRDGBufferRef>& ProbeTraceDataArray,
		const TInlineArray<FRDGBufferRef>& ProbeWorldOffsetArray,
		const TInlineArray<FRDGBufferRef>& HardwareRayTracingRayAllocatorBufferArray,
		const TInlineArray<FRDGBufferRef>& TraceProbesIndirectArgsArray,
		ERDGPassFlags ComputePassFlags);

	bool ShouldFilterProbes();
};