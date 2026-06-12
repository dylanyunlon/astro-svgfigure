// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreTypes.h"
#include "MeshCardRepresentation.h"
#include "MeshPassProcessor.h"
#include "RenderGraphFwd.h"

class FScene;
class FLumenPrimitiveGroup;
class FLumenCard;
struct FNaniteShadingBin;
struct FNaniteShadingCommand;
class FViewInfo;
struct FViewMatrices;

struct FCardCaptureAtlas
{
	FIntPoint Size;
	FRDGTextureRef Albedo = nullptr;
	FRDGTextureRef Normal = nullptr;
	FRDGTextureRef Emissive = nullptr;
	FRDGTextureRef DepthStencil = nullptr;
};

struct FResampledCardCaptureAtlas
{
	FIntPoint Size;
	FRDGTextureRef DirectLighting = nullptr;
	FRDGTextureRef IndirectLighting = nullptr;
	FRDGTextureRef NumFramesAccumulated = nullptr;
	FRDGBufferRef TileShadowDownsampleFactor = nullptr;
};

class FCardPageRenderData
{
public:
	int32 PrimitiveGroupIndex = INDEX_NONE;

	// CardData
	const int32 CardIndex = INDEX_NONE;
	const int32 PageTableIndex = INDEX_NONE;
	FVector4f CardUVRect;
	FIntRect CardCaptureAtlasRect;
	FIntRect SurfaceCacheAtlasRect;

	FLumenCardOBBd CardWorldOBB;

	FMatrix ProjectionMatrixUnadjustedForRHI;

	int32 StartMeshDrawCommandIndex = 0;
	int32 NumMeshDrawCommands = 0;

	TArray<uint32, SceneRenderingAllocator> NaniteInstanceIds;
	TArray<FNaniteShadingBin, SceneRenderingAllocator> NaniteShadingBins;
	float NaniteLODScaleFactor = 1.0f;

	bool bHeightField = false;
	bool bResampleLastLighting = false;
	ELumenCardDilationMode DilationMode = ELumenCardDilationMode::Disabled;

	bool bAxisXFlipped;
	int32 CopyCardIndex;

	// Non-Nanite mesh inclusive instance ranges to draw
	TArray<uint32, SceneRenderingAllocator> InstanceRuns;

	FCardPageRenderData(
		const FLumenCard& InLumenCard,
		FVector4f InCardUVRect,
		FIntRect InCardCaptureAtlasRect,
		FIntRect InSurfaceCacheAtlasRect,
		int32 InPrimitiveGroupIndex,
		int32 InCardIndex,
		int32 InCardPageIndex,
		bool bResampleLastLighting,
		bool bInAxisXFlipped = false,
		int32 InCopyCardIndex = INDEX_NONE);

	~FCardPageRenderData();

	void GetViewMatrices(const FViewInfo& MainView, FViewMatrices& OutViewMatrices) const;

	void PatchView(const FScene* Scene, FViewInfo* View) const;

	inline bool HasNanite() const
	{
		return NaniteShadingBins.Num() > 0 && NaniteInstanceIds.Num() > 0;
	}

	bool NeedsRender() const
	{
		return CopyCardIndex == INDEX_NONE;
	}

private:
	void CacheProjectionMatrix();
};

namespace LumenScene
{

bool HasPrimitiveNaniteMeshBatches(const FPrimitiveSceneProxy* Proxy);
bool AllowSurfaceCacheCardSharing();
bool CullUndergroundTexels();

void AllocateCardCaptureAtlas(
	FRDGBuilder& GraphBuilder,
	FIntPoint CardCaptureAtlasSize,
	FCardCaptureAtlas& CardCaptureAtlas,
	EShaderPlatform ShaderPlatform
);

void AddCardCaptureDraws(
	const FScene* Scene,
	FCardPageRenderData& CardPageRenderData,
	const FLumenPrimitiveGroup& PrimitiveGroup,
	TConstArrayView<const FPrimitiveSceneInfo*> SceneInfoPrimitives,
	FMeshCommandOneFrameArray& VisibleMeshCommands,
	TArray<int32, SceneRenderingAllocator>& PrimitiveIds
);

}

/**   
 * Data for rendering meshes into Surface Cache
 */
class FLumenCardRenderer
{
public:
	TArray<FCardPageRenderData, SceneRenderingAllocator> CardPagesToRender;

	int32 NumCardTexelsToCapture;
	FMeshCommandOneFrameArray MeshDrawCommands;
	TArray<int32, SceneRenderingAllocator> MeshDrawPrimitiveIds;

	FResampledCardCaptureAtlas ResampledCardCaptureAtlas;

	/** Whether Lumen should propagate a global lighting change this frame. */
	bool bPropagateGlobalLightingChange = false;

	// If true, at least one card page is copied instead of being captured. A copy can be downsampling
	// from self or copying from another matching card with the same or higher resolution
	bool bHasAnyCardCopy = false;

	void Reset()
	{
		CardPagesToRender.Reset();
		MeshDrawCommands.Reset();
		MeshDrawPrimitiveIds.Reset();
		NumCardTexelsToCapture = 0;
		bHasAnyCardCopy = false;
	}
};

BEGIN_UNIFORM_BUFFER_STRUCT(FLumenCardOutputs, )
	SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, OutTarget0)
	SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, OutTarget1)
	SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, OutTarget2)
END_UNIFORM_BUFFER_STRUCT()

namespace Nanite
{
	void RecordLumenCardParameters(
		FRHIBatchedShaderParameters& ShaderParameters,
		FNaniteShadingCommand& ShadingCommand,
		TUniformBufferRef<FLumenCardOutputs> Outputs
	);
}
