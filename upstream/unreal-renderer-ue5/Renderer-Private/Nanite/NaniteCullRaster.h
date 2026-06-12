// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "NaniteShared.h"
#include "NaniteSceneProxy.h"
#include "NaniteVisibility.h"
#include "PSOPrecacheMaterial.h"

class FVirtualShadowMapArray;
class FViewFamilyInfo;
class FSceneInstanceCullingQuery;

BEGIN_SHADER_PARAMETER_STRUCT(FRasterParameters,)
	SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>,			OutDepthBuffer)
	SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2DArray<uint>,	OutDepthBufferArray)
	SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<UlongType>,	OutVisBuffer64)
	SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<UlongType>,	OutDbgBuffer64)
	SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>,			OutDbgBuffer32)
END_SHADER_PARAMETER_STRUCT()

namespace Nanite
{

enum class ERasterScheduling : uint8
{
	// Only rasterize using fixed function hardware.
	HardwareOnly = 0,

	// Rasterize large triangles with hardware, small triangles with software (compute).
	HardwareThenSoftware = 1,

	// Rasterize large triangles with hardware, overlapped with rasterizing small triangles with software (compute).
	HardwareAndSoftwareOverlap = 2,
};

/**
 * Used to select raster mode when creating the context.
 */
enum class EOutputBufferMode : uint8
{
	// Default mode outputting both ID and depth
	VisBuffer,

	// Rasterize only depth to 32 bit buffer
	DepthOnly,
};

struct FSharedContext
{
	FGlobalShaderMap* ShaderMap;
	ERHIFeatureLevel::Type FeatureLevel;
	EShaderPlatform ShaderPlatform;
	ERasterPipeline Pipeline;
};

struct FRasterContext
{
	FVector2f			RcpViewSize;
	FIntPoint			TextureSize;
	EOutputBufferMode	RasterMode;
	ERasterScheduling	RasterScheduling;

	FRasterParameters	Parameters;

	FRDGTextureRef		DepthBuffer;
	FRDGTextureRef		VisBuffer64;
	FRDGTextureRef		DbgBuffer64;
	FRDGTextureRef		DbgBuffer32;

	bool				VisualizeActive : 1;
	bool				VisualizeModeOverdraw : 1;
	bool				bCustomPass : 1;
	bool				bEnableAssemblyMeta : 1;
	bool 				bAllowTessellation : 1;
};

struct FRasterResults
{
	FIntVector4		PageConstants;
	uint32			MaxVisibleClusters;
	uint32			MaxCandidatePatches;
	uint32			MaxNodes;
	uint32			MaxPatchesPerGroup;
	uint32			MeshPass;
	float			InvDiceRate;
	uint32			RenderFlags;
	uint32			DebugFlags;

	FRDGBufferRef	ViewsBuffer			= nullptr;
	FRDGBufferRef	VisibleClustersSWHW	= nullptr;
	FRDGBufferRef	AssemblyTransforms	= nullptr;
	FRDGBufferRef	AssemblyMeta		= nullptr;
	FRDGBufferRef	RasterBinMeta		= nullptr;
	FRDGBufferRef	RasterBinData		= nullptr;
	FRDGBufferRef	RasterBinArgs		= nullptr;
	FRDGBufferRef	RasterGroupMeta		= nullptr;

	FRDGTextureRef	VisBuffer64			= nullptr;
	FRDGTextureRef	DbgBuffer64			= nullptr;
	FRDGTextureRef	DbgBuffer32			= nullptr;

	FRDGTextureRef	ShadingMask			= nullptr;

	FRDGBufferRef	ClearTileArgs		= nullptr;
	FRDGBufferRef	ClearTileBuffer		= nullptr;

	FNaniteTranslucencyContext TranslucencyContext;

	FNaniteVisibilityQuery* VisibilityQuery = nullptr;

	TArray<FVisualizeResult, TInlineAllocator<32>> Visualizations;
};

void CollectRasterPSOInitializers(
	const FSceneTexturesConfig& SceneTexturesConfig,
	const FMaterial& Material,
	const FPSOPrecacheParams& PreCacheParams,
	EShaderPlatform ShaderPlatform,
	int32 PSOCollectorIndex,
	TArray<FPSOPrecacheData>& PSOInitializers);

void CollectRasterPSOInitializersForRasterPipeline(
	const FNaniteRasterPipeline& RasterPipeline,
	ERHIFeatureLevel::Type FeatureLevel,
	TArray<FPSOPrecacheData>& OutPSOPrecacheData);

struct FRasterContextInitParams
{
	FIntPoint TextureSize;
	FIntRect TextureRect;
	EOutputBufferMode RasterMode = EOutputBufferMode::VisBuffer;
	FRDGBufferSRVRef RectMinMaxBufferSRV = nullptr;
	FRDGTextureRef ExternalDepthBuffer = nullptr;
	uint32 NumRects = 0;
	bool bClearTarget : 1 = true;
	bool bAsyncCompute : 1 = true;
	bool bCustomPass : 1 = false;
	bool bVisualize : 1 = false;
	bool bVisualizeOverdraw : 1 = false;
	bool bEnableAssemblyMeta : 1 = false;
	bool bAllowTessellation : 1 = true;
};

FRasterContext InitRasterContext(
	FRDGBuilder& GraphBuilder,
	const FSharedContext& SharedContext,
	const FViewFamilyInfo& ViewFamily,
	const FRasterContextInitParams& InitParams
);

struct FConfiguration
{
	uint32 bTwoPassOcclusion : 1;
	uint32 bUpdateStreaming : 1;
	uint32 bDrawOnlyRayTracingFarField : 1;
	uint32 bSupportsMultiplePasses : 1;
	uint32 bForceHWRaster : 1;
	uint32 bPrimaryContext : 1;
	uint32 bDrawOnlyRootGeometry : 1;
	uint32 bIsShadowPass : 1;
	uint32 bIsSceneCapture : 1;
	uint32 bIsReflectionCapture : 1;
	uint32 bIsLumenCapture : 1;
	uint32 bIsMaterialCache : 1;
	uint32 bIsGameView : 1;
	uint32 bEditorShowFlag : 1;
	uint32 bGameShowFlag : 1;
	uint32 bDisableProgrammable : 1;
	uint32 bExtractVSMPerformanceFeedback : 1;
	uint32 bExtractStats : 1;
	EFilterFlags HiddenFilterFlags;

	void SetViewFlags(const FViewInfo& View);
};

/**
 * Used to supply an explicit list of chunks of draws.
 */
struct FExplicitChunkDrawInfo
{
	uint32 NumChunks = 0;
	FRDGBufferRef ExplicitChunkDraws = nullptr; // Buffer of FInstanceCullingGroupWork
	FRDGBufferRef InstanceIds = nullptr; // Buffer of instance ids indexed by ExplicitChunkDraws elements.
};

class IRenderer
{
public:
	static TUniquePtr< IRenderer > Create(
		FRDGBuilder&			GraphBuilder,
		const FScene&			Scene,
		const FViewInfo&		SceneView,
		FSceneUniformBuffer&	SceneUniformBuffer,
		const FSharedContext&	SharedContext,
		const FRasterContext&	RasterContext,
		const FConfiguration&	Configuration,
		const FIntRect&			ViewRect,
		const FRDGTextureRef	PrevHZB,
		FVirtualShadowMapArray*	VirtualShadowMapArray = nullptr );

	IRenderer() = default;
	virtual ~IRenderer() = default;

	// NOTE: NumViews is not used if InViewDrawRanges is provided; can be set to 0
	virtual void DrawGeometry(
		FNaniteRasterPipelines& RasterPipelines,
		const FNaniteVisibilityQuery* VisibilityQuery,
		FRDGBufferRef ViewsBuffer,
		FRDGBufferRef InViewDrawRanges,
		int32 NumViews,
		FSceneInstanceCullingQuery* OptionalSceneInstanceCullingQuery,
		const TConstArrayView<FInstanceDraw>* OptionalInstanceDraws,
		const FExplicitChunkDrawInfo* OptionalExplicitChunkDrawInfo) = 0;

	/**
	* Draw scene geometry with a CPU-provided view array
	*/
	virtual void DrawGeometry(
		FNaniteRasterPipelines& RasterPipelines,
		const FNaniteVisibilityQuery* VisibilityQuery,
		const FPackedViewArray& ViewArray,
		FSceneInstanceCullingQuery* OptionalSceneInstanceCullingQuery,
		const TConstArrayView<FInstanceDraw>* OptionalInstanceDraws) = 0;

	/**
	 * Draw scene geometry by brute-force culling against all instances in the scene.
	 */
	inline void DrawGeometry(FNaniteRasterPipelines& RasterPipelines,
		const FNaniteVisibilityQuery* VisibilityQuery,
		const FPackedViewArray& ViewArray)
	{
		DrawGeometry(RasterPipelines, VisibilityQuery, ViewArray, nullptr, nullptr);
	}

	/**
	 * Draw scene geometry driven by an explicit list FInstanceDraw (instance-id / view-id pairs).
	 */
	inline void DrawGeometry(FNaniteRasterPipelines& RasterPipelines,
		const FNaniteVisibilityQuery* VisibilityQuery,
		const FPackedViewArray& ViewArray,
		const TConstArrayView<FInstanceDraw> &InstanceDraws)
	{
		DrawGeometry(RasterPipelines, VisibilityQuery, ViewArray, nullptr, &InstanceDraws);
	}

	/**
	 * Draw scene geometry with and optional scene instance culling query. If non-null, the culling result is used to drive rendering, 
	 * otherwise falls back to brute-force culling (as above). 
	 */
	inline void DrawGeometry(FNaniteRasterPipelines& RasterPipelines,
		const FNaniteVisibilityQuery* VisibilityQuery,
		const FPackedViewArray& ViewArray,
		FSceneInstanceCullingQuery* OptionalSceneInstanceCullingQuery)
	{
		DrawGeometry(RasterPipelines, VisibilityQuery, ViewArray, OptionalSceneInstanceCullingQuery, nullptr);
	}

	virtual void ExtractResults( FRasterResults& RasterResults ) = 0;
};

} // namespace Nanite
