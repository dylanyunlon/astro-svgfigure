// Copyright Epic Games, Inc. All Rights Reserved.

#include "NaniteShading.h"
#include "CoreMinimal.h"
#include "Async/ParallelFor.h"
#include "Async/TaskGraphInterfaces.h"
#include "BasePassRendering.h"
#include "ComponentRecreateRenderStateContext.h"
#include "Containers/ArrayView.h"
#include "Containers/SetUtilities.h"
#include "Containers/SparseSet.h"
#include "Containers/StaticArray.h"
#include "DataDrivenShaderPlatformInfo.h"
#include "DecalRenderingShared.h"
#include "DynamicRHI.h"
#include "Engine/EngineTypes.h"
#include "Experimental/ConcurrentLinearAllocator.h"
#include "Experimental/Containers/RobinHoodHashTable.h"
#include "ExtensibleUniformBuffer.h"
#include "GBufferInfo.h"
#include "GlobalShader.h"
#include "HAL/IConsoleManager.h"
#include "HAL/LowLevelMemTracker.h"
#include "LightMapRendering.h"
#include "Logging/LogMacros.h"
#include "Lumen/LumenSceneCardCapture.h"
#include "MaterialDomain.h"
#include "Materials/Material.h"
#include "Materials/MaterialInterface.h"
#include "Materials/MaterialRelevance.h"
#include "Materials/MaterialRenderProxy.h"
#include "MaterialShaderType.h"
#include "MaterialShared.h"
#include "Math/Color.h"
#include "MeshDrawCommands.h"
#include "MeshMaterialShader.h"
#include "MeshPassProcessor.h"
#include "MeshPassUtils.h"
#include "Misc/App.h"
#include "Misc/LargeWorldCoordinates.h"
#include "Nanite/Nanite.h"
#include "Nanite/NaniteComposition.h"
#include "Nanite/NaniteMaterialsSceneExtension.h"
#include "Nanite/NaniteShared.h"
#include "NaniteDefinitions.h"
#include "NaniteSceneProxy.h"
#include "NaniteTranslucency.h"
#include "NaniteVertexFactory.h"
#include "NaniteVisualizationData.h"
#include "PipelineStateCache.h"
#include "PrimitiveSceneInfo.h"
#include "PrimitiveSceneProxy.h"
#include "PrimitiveViewRelevance.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"
#include "ProfilingDebugging/TagTrace.h"
#include "PSOPrecache.h"
#include "PSOPrecacheMaterial.h"
#include "RendererInterface.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "Rendering/NaniteInterface.h"
#include "Rendering/NaniteResources.h"
#include "Rendering/NaniteStreamingManager.h"
#include "RenderUtils.h"
#include "RHI.h"
#include "RHIBreadcrumbs.h"
#include "RHICommandList.h"
#include "RHIResources.h"
#include "RHIShaderParameters.h"
#include "RHIStaticStates.h"
#include "SceneInterface.h"
#include "SceneManagement.h"
#include "ScenePrivate.h"
#include "SceneRendering.h"
#include "SceneTextures.h"
#include "SceneTexturesConfig.h"
#include "SceneUniformBuffer.h"
#include "SceneUtils.h"
#include "SceneView.h"
#include "Shader.h"
#include "ShaderParameterMacros.h"
#include "ShaderParameterStruct.h"
#include "ShaderParameterUtils.h"
#include "ShaderPlatformConfig.h"
#include "Stats/Stats.h"
#include "Substrate/Substrate.h"
#include "SystemTextures.h"
#include "Tasks/Task.h"
#include "UnrealEngine.h"
#include "VariableRateShadingImageManager.h"
#include "VertexFactory.h"
#include "VirtualShadowMaps/VirtualShadowMapArray.h"
#include "Lumen/LumenViewState.h"
#include "PSOPrecacheValidation.h"

extern TAutoConsoleVariable<int32> CVarNaniteShowDrawEvents;
extern TAutoConsoleVariable<int32> CVarRHICmdMinDrawsPerParallelCmdList;

extern int32 GSkipDrawOnPSOPrecaching;
extern int32 GNaniteShowStats;

#if WANTS_DRAW_MESH_EVENTS
static FORCEINLINE const FString& GetShadingMaterialName(const FMaterialRenderProxy* InShadingMaterial)
{
	if (InShadingMaterial == nullptr)
	{
		static FString Invalid = "<Invalid>";
		return Invalid;
	}

	return InShadingMaterial->GetMaterialName();
}
#endif

TAutoConsoleVariable<int32> CVarParallelBasePassBuild(
	TEXT("r.Nanite.ParallelBasePassBuild"),
	1,
	TEXT(""),
	ECVF_RenderThreadSafe
);

static int32 GNaniteFastTileClear = 1;
static FAutoConsoleVariableRef CVarNaniteFastTileClear(
	TEXT("r.Nanite.FastTileClear"),
	GNaniteFastTileClear,
	TEXT("Whether to enable Nanite fast tile clearing"),
	ECVF_RenderThreadSafe
);

static int32 GNaniteFastTileClearSubTiles = 1;
static FAutoConsoleVariableRef CVarNaniteFastTileClearSubTiles(
	TEXT("r.Nanite.FastTileClear.SubTiles"),
	GNaniteFastTileClearSubTiles,
	TEXT("Whether to enable Nanite fast tile clearing (for 4x4 sub tiles)"),
	ECVF_RenderThreadSafe
);

static int32 GNaniteFastTileClearDynResOpt = 0;
static FAutoConsoleVariableRef CVarNaniteFastTileClearDynResOpt(
	TEXT("r.Nanite.FastTileClear.DynResOpt"),
	GNaniteFastTileClearDynResOpt,
	TEXT("Skip clearing anything outside the dynres viewport (Experimental)."),	
	ECVF_RenderThreadSafe
);


static int32 GNaniteValidateShadingBinUB = 0;
static FAutoConsoleVariableRef CVarNaniteValidateShadingBinUB(
	TEXT("r.Nanite.Debug.ValidateShadingBinUB"),
	GNaniteValidateShadingBinUB,
	TEXT("Validate that all Nanite shading pipeline MaterialUB pointers are up-to-date at dispatch time."),
	ECVF_RenderThreadSafe
);

static int32 GNaniteFastTileVis = INDEX_NONE;
static FAutoConsoleVariableRef CVarNaniteFastTileVis(
	TEXT("r.Nanite.FastTileVis"),
	GNaniteFastTileVis,
	TEXT("Allows for just showing a single target in the visualization, or -1 to show all accumulated"),
	ECVF_RenderThreadSafe
);

TAutoConsoleVariable<int32> CVarNaniteBundleEmulation(
	TEXT("r.Nanite.Bundle.Emulation"),
	0,
	TEXT("Whether to force shader bundle dispatch emulation"),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
	{
		// We need to recreate scene proxies so that BuildShadingCommands can be re-evaluated.
		FGlobalComponentRecreateRenderStateContext Context;
	}),
	ECVF_RenderThreadSafe
);

static int32 GNaniteBundleShading = 0;
static FAutoConsoleVariableRef CVarNaniteBundleShading(
	TEXT("r.Nanite.Bundle.Shading"),
	GNaniteBundleShading,
	TEXT("Whether to enable Nanite shader bundle dispatch for shading"),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
	{
		// We need to recreate scene proxies so that BuildShadingCommands can be re-evaluated.
		FGlobalComponentRecreateRenderStateContext Context;
	}),
	ECVF_RenderThreadSafe
);

static int32 GNaniteComputeMaterialsSort = 1;
static FAutoConsoleVariableRef CVarNaniteComputeMaterialsSort(
	TEXT("r.Nanite.ComputeMaterials.Sort"),
	GNaniteComputeMaterialsSort,
	TEXT(""),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
	{
		// We need to recreate scene proxies so that BuildShadingCommands can be re-evaluated.
		FGlobalComponentRecreateRenderStateContext Context;
	}),
	ECVF_RenderThreadSafe
);

static int32 GBinningTechnique = 0;
static FAutoConsoleVariableRef CVarNaniteBinningTechnique(
	TEXT("r.Nanite.BinningTechnique"),
	GBinningTechnique,
	TEXT(""),
	ECVF_RenderThreadSafe
);

static int32 GNaniteShadeBinningMode = 0;
static FAutoConsoleVariableRef CVarNaniteShadeBinningMode(
	TEXT("r.Nanite.ShadeBinningMode"),
	GNaniteShadeBinningMode,
	TEXT("0: Auto\n")
	TEXT("1: Force to Pixel Mode\n")
	TEXT("2: Force to Quad Mode\n"),
	FConsoleVariableDelegate::CreateLambda([](IConsoleVariable* InVariable)
	{
		// We need to recreate scene proxies so that BuildShadingCommands can be re-evaluated.
		FGlobalComponentRecreateRenderStateContext Context;
	}),
	ECVF_RenderThreadSafe
);

static int32 GNaniteSoftwareVRS = 1;
static FAutoConsoleVariableRef CVarNaniteSoftwareVRS(
	TEXT("r.Nanite.SoftwareVRS"),
	GNaniteSoftwareVRS,
	TEXT("Whether to enable Nanite software variable rate shading in compute."),
	ECVF_RenderThreadSafe
);

int32 GNaniteValidateShadeBinning = 0;
static FAutoConsoleVariableRef CVarNaniteValidateShadeBinning(
	TEXT("r.Nanite.Debug.ValidateShadeBinning"),
	GNaniteValidateShadeBinning,
	TEXT(""),
	ECVF_RenderThreadSafe
);

static int32 GNaniteCacheRelevanceParallel = 1;
static FAutoConsoleVariableRef CVarNaniteCacheRelevanceParallel(
	TEXT("r.Nanite.CacheRelevanceParallel"),
	GNaniteCacheRelevanceParallel,
	TEXT("Enable parallel caching of Nanite material relevance. 0=disabled, 1=enabled (default)"),
	ECVF_RenderThreadSafe
);

inline bool UsingHighPrecisionGBuffer()
{
	static const auto CVarFormat = IConsoleManager::Get().FindTConsoleVariableDataInt(TEXT("r.GBufferFormat"));
	static const int32 EGBufferFormat_Force16BitsPerChannel = 5; // TODO: Refactor GBufferInfo.cpp to cleanly expose this
	const bool bHighPrecisionGBuffer = CVarFormat && CVarFormat->GetValueOnRenderThread() >= EGBufferFormat_Force16BitsPerChannel;
	return bHighPrecisionGBuffer;
}

bool CanUseShaderBundleWorkGraph(EShaderPlatform Platform)
{
	static bool bNaniteBundleSupportWorkGraphs = NaniteWorkGraphMaterialsSupported();
	return bNaniteBundleSupportWorkGraphs && !!GRHISupportsShaderBundleWorkGraphDispatch && RHISupportsWorkGraphs(Platform);
}

static bool UseShadingShaderBundle(EShaderPlatform Platform)
{
	return  GNaniteBundleShading != 0 && (!!GRHISupportsShaderBundleDispatch || CanUseShaderBundleWorkGraph(Platform));
}

static ERHIShaderBundleMethod GetShadingShaderBundleMethod(EShaderPlatform Platform)
{
	// Enabling emulation always overrides the other cvars/methods.
	ERHIShaderBundleMethod Method = ERHIShaderBundleMethod::Emulation;

	if (UseShadingShaderBundle(Platform) && CVarNaniteBundleEmulation.GetValueOnRenderThread() == 0)
	{
		Method = ERHIShaderBundleMethod::Native;
	}

	return Method;
}

static bool UseWorkGraphForNaniteShading(EShaderPlatform Platform)
{
	const ERHIShaderBundleMethod BundleMethod = GetShadingShaderBundleMethod(Platform);
	return CanUseShaderBundleWorkGraph(Platform) && BundleMethod == ERHIShaderBundleMethod::Native;
}

static uint32 GetShadingRateTileSizeBits()
{
	uint32 TileSizeBits = 0;

	if (GNaniteSoftwareVRS != 0 && GVRSImageManager.IsVRSEnabledForFrame() /* HW or SW VRS enabled? */)
	{
		bool bUseSoftwareImage = GVRSImageManager.IsSoftwareVRSEnabledForFrame();
		if (!bUseSoftwareImage)
		{
			// Technically these could be different, but currently never in practice
			// 8x8, 16x16, or 32x32 for DX12 Tier2 HW VRS
			ensure
			(
				GRHIVariableRateShadingImageTileMinWidth == GRHIVariableRateShadingImageTileMinHeight &&
				GRHIVariableRateShadingImageTileMinWidth == GRHIVariableRateShadingImageTileMaxWidth &&
				GRHIVariableRateShadingImageTileMinWidth == GRHIVariableRateShadingImageTileMaxHeight &&
				FMath::IsPowerOfTwo(GRHIVariableRateShadingImageTileMinWidth)
			);
		}

		uint32 TileSize = GVRSImageManager.GetSRITileSize(bUseSoftwareImage).X;
		TileSizeBits = FMath::FloorLog2(TileSize);
	}

	return TileSizeBits;
}

static FRDGTextureRef GetShadingRateImage(FRDGBuilder& GraphBuilder, const FViewInfo& ViewInfo)
{
	FRDGTextureRef ShadingRateImage = nullptr;

	if (GetShadingRateTileSizeBits() != 0)
	{
		bool bUseSoftwareImage = GVRSImageManager.IsSoftwareVRSEnabledForFrame();
		ShadingRateImage = GVRSImageManager.GetVariableRateShadingImage(GraphBuilder, ViewInfo, FVariableRateShadingImageManager::EVRSPassType::NaniteEmitGBufferPass, bUseSoftwareImage);
	}

	if (ShadingRateImage == nullptr)
	{
		const FRDGSystemTextures& SystemTextures = FRDGSystemTextures::Get(GraphBuilder);
		ShadingRateImage = SystemTextures.Black;
	}

	return ShadingRateImage;
}

class FVisualizeClearTilesCS : public FNaniteGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FVisualizeClearTilesCS);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(FUint32Vector4, ViewRect)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTextureMetadata, OutCMaskBuffer)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, OutVisualized)
	END_SHADER_PARAMETER_STRUCT()

	FVisualizeClearTilesCS() = default;
	FVisualizeClearTilesCS(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
	: FNaniteGlobalShader(Initializer)
	{
		PlatformDataParam.Bind(Initializer.ParameterMap, TEXT("PlatformData"), SPF_Mandatory);
		BindForLegacyShaderParameters<FParameters>(this, Initializer.PermutationId, Initializer.ParameterMap);
	}

	// Shader parameter structs don't have a way to push variable sized data yet. So the we use the old shader parameter API.
	void SetParameters(FRHIBatchedShaderParameters& BatchedParameters, const void* PlatformDataPtr, uint32 PlatformDataSize)
	{
		BatchedParameters.SetShaderParameter(PlatformDataParam.GetBufferIndex(), PlatformDataParam.GetBaseIndex(), PlatformDataSize, PlatformDataPtr);
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return RHISupportsRenderTargetWriteMask(Parameters.Platform) && DoesPlatformSupportNanite(Parameters.Platform);
	}

private:
	LAYOUT_FIELD(FShaderParameter, PlatformDataParam);
};
IMPLEMENT_GLOBAL_SHADER(FVisualizeClearTilesCS, "/Engine/Private/Nanite/NaniteFastClear.usf", "VisualizeClearTilesCS", SF_Compute);

class FShadingBinBuildCS : public FNaniteGlobalShader
{
	DECLARE_GLOBAL_SHADER(FShadingBinBuildCS);

	class FBuildPassDim : SHADER_PERMUTATION_SPARSE_INT("SHADING_BIN_PASS", NANITE_SHADING_BIN_COUNT, NANITE_SHADING_BIN_SCATTER);
	class FTechniqueDim : SHADER_PERMUTATION_INT("BINNING_TECHNIQUE", 2);
	class FGatherStatsDim : SHADER_PERMUTATION_BOOL("GATHER_STATS");
	class FVariableRateDim : SHADER_PERMUTATION_BOOL("VARIABLE_SHADING_RATE");
	class FOptimizeWriteMaskDim : SHADER_PERMUTATION_BOOL("OPTIMIZE_WRITE_MASK");
	class FNumExportsDim : SHADER_PERMUTATION_RANGE_INT("NUM_EXPORTS", 1, MaxSimultaneousRenderTargets);
	class FBindlessDim : SHADER_PERMUTATION_BOOL("BINDLESS");
	using FPermutationDomain = TShaderPermutationDomain<FBuildPassDim, FTechniqueDim, FGatherStatsDim, FVariableRateDim, FOptimizeWriteMaskDim, FNumExportsDim, FBindlessDim>;

	FShadingBinBuildCS() = default;
	FShadingBinBuildCS(const ShaderMetaType::CompiledShaderInitializerType & Initializer)
	: FNaniteGlobalShader(Initializer)
	{
		PlatformDataParam.Bind(Initializer.ParameterMap, TEXT("PlatformData"), SPF_Optional);
		SubTileMatchParam.Bind(Initializer.ParameterMap, TEXT("SubTileMatch"), SPF_Optional);
		BindForLegacyShaderParameters<FParameters>(this, Initializer.PermutationId, Initializer.ParameterMap);
	}

	// Shader parameter structs don't have a way to push variable sized data yet. So the we use the old shader parameter API.
	void SetParameters(FRHIBatchedShaderParameters& BatchedParameters, const void* PlatformDataPtr, uint32 PlatformDataSize, bool bSubTileMatch)
	{
		BatchedParameters.SetShaderParameter(PlatformDataParam.GetBufferIndex(), PlatformDataParam.GetBaseIndex(), PlatformDataSize, PlatformDataPtr);

		uint32 SubTileMatch = bSubTileMatch ? 1u : 0u;
		BatchedParameters.SetShaderParameter(SubTileMatchParam.GetBufferIndex(), SubTileMatchParam.GetBaseIndex(), sizeof(SubTileMatch), &SubTileMatch);
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		if (!DoesPlatformSupportNanite(Parameters.Platform))
		{
			return false;
		}

		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (PermutationVector.Get<FOptimizeWriteMaskDim>() && !RHISupportsRenderTargetWriteMask(Parameters.Platform))
		{
			return false;
		}

		if (PermutationVector.Get<FOptimizeWriteMaskDim>() && PermutationVector.Get<FBuildPassDim>() != NANITE_SHADING_BIN_COUNT)
		{
			// We only want one of the build passes to export out cmask, so we choose the 
			// counting pass because it touches less memory already than scatter.
			return false;
		}

		if (!PermutationVector.Get<FOptimizeWriteMaskDim>() && PermutationVector.Get<FNumExportsDim>() > 1)
		{
			// The NUM_EXPORTS perm is only valid when optimizing the write mask.
			return false;
		}

		if (PermutationVector.Get<FBindlessDim>() && !UseNaniteBindlessShading(Parameters.Platform))
		{
			return false;
		}

		return true;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (PermutationVector.Get<FBindlessDim>())
		{
			OutEnvironment.CompilerFlags.Add(CFLAG_SupportsMinimalBindless);
		}
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(FUint32Vector4, ViewRect)
		SHADER_PARAMETER(uint32, ValidWriteMask)
		SHADER_PARAMETER(FUint32Vector2, DispatchOffsetTL)
		SHADER_PARAMETER(uint32, ShadingBinCount)
		SHADER_PARAMETER(uint32, ShadingGroupCount)
		SHADER_PARAMETER(uint32, ShadingRateTileSizeBits)
		SHADER_PARAMETER(uint32, DummyZero)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<uint>, ShadingRateImage)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<uint>, ShadingMask)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FNaniteShadingBinScatterRanges>, ShadingBinScatterRanges)
		SHADER_PARAMETER_SAMPLER(SamplerState, ShadingMaskSampler)
		SHADER_PARAMETER_RDG_TEXTURE_UAV_ARRAY(RWTextureMetadata, OutCMaskBuffer, [MaxSimultaneousRenderTargets])
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<FNaniteShadingBinStats>, OutShadingBinStats)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, OutShadingBinData)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, OutShadingDispatchArgs)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<FNaniteShadingBinScatterCounters>, OutShadingBinScatterCounters)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, OutThreadGroupData)
	END_SHADER_PARAMETER_STRUCT()

private:
	LAYOUT_FIELD(FShaderParameter, PlatformDataParam);
	LAYOUT_FIELD(FShaderParameter, SubTileMatchParam);
};
IMPLEMENT_GLOBAL_SHADER(FShadingBinBuildCS, "/Engine/Private/Nanite/NaniteShadeBinning.usf", "ShadingBinBuildCS", SF_Compute);

class FShadingGroupCS : public FNaniteGlobalShader
{
	DECLARE_GLOBAL_SHADER(FShadingGroupCS);
	SHADER_USE_PARAMETER_STRUCT(FShadingGroupCS, FNaniteGlobalShader);

	class FBinPassDim : SHADER_PERMUTATION_SPARSE_INT("SHADING_BIN_PASS", NANITE_SHADING_BIN_GROUP_COUNT, NANITE_SHADING_BIN_GROUP_RESERVE);
	using FPermutationDomain = TShaderPermutationDomain<FBinPassDim>;

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportNanite(Parameters.Platform) && UseNaniteBindlessShading(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

		OutEnvironment.CompilerFlags.Add(CFLAG_SupportsMinimalBindless);
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(uint32, ShadingBinCount)
		SHADER_PARAMETER(uint32, ShadingGroupCount)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, OutShadingBinData)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<FNaniteShadingBinScatterCounters>, OutShadingBinScatterCounters)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<FNaniteShadingGroupMeta>, OutShadingGroupMeta)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<FNaniteShadingGroupAllocator>, OutShadingGroupAllocator)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, OutShadingDispatchArgs)
	END_SHADER_PARAMETER_STRUCT()
};
IMPLEMENT_GLOBAL_SHADER(FShadingGroupCS, "/Engine/Private/Nanite/NaniteShadeBinning.usf", "ShadingGroupCS", SF_Compute);


class FShadingBinReserveCS : public FNaniteGlobalShader
{
	DECLARE_GLOBAL_SHADER(FShadingBinReserveCS);
	SHADER_USE_PARAMETER_STRUCT(FShadingBinReserveCS, FNaniteGlobalShader);

	class FGatherStatsDim : SHADER_PERMUTATION_BOOL("GATHER_STATS");
	class FBindlessDim : SHADER_PERMUTATION_BOOL("BINDLESS");
	using FPermutationDomain = TShaderPermutationDomain<FGatherStatsDim, FBindlessDim>;

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		if (!DoesPlatformSupportNanite(Parameters.Platform))
		{
			return false;
		}

		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (PermutationVector.Get<FBindlessDim>() && !UseNaniteBindlessShading(Parameters.Platform))
		{
			return false;
		}

		return true;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("SHADING_BIN_PASS"), NANITE_SHADING_BIN_RESERVE);

		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (PermutationVector.Get<FBindlessDim>())
		{
			OutEnvironment.CompilerFlags.Add(CFLAG_SupportsMinimalBindless);
		}
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(uint32, ShadingBinCount)
		SHADER_PARAMETER(uint32, ShadingGroupCount)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<FNaniteShadingBinStats>, OutShadingBinStats)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, OutShadingBinData)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, OutShadingBinAllocator)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, OutShadingDispatchArgs)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<FNaniteShadingBinScatterCounters>, OutShadingBinScatterCounters)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<FNaniteShadingBinScatterRanges>, OutShadingBinScatterRanges)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<FNaniteShadingGroupMeta>, OutShadingGroupMeta)
	END_SHADER_PARAMETER_STRUCT()
};
IMPLEMENT_GLOBAL_SHADER(FShadingBinReserveCS, "/Engine/Private/Nanite/NaniteShadeBinning.usf", "ShadingBinReserveCS", SF_Compute);

class FShadingBinValidateCS : public FNaniteGlobalShader
{
	DECLARE_GLOBAL_SHADER(FShadingBinValidateCS);
	SHADER_USE_PARAMETER_STRUCT(FShadingBinValidateCS, FNaniteGlobalShader);

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportNanite(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("SHADING_BIN_PASS"), NANITE_SHADING_BIN_VALIDATE);
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(uint32, ShadingBinCount)
		SHADER_PARAMETER(uint32, ShadingGroupCount)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWByteAddressBuffer, OutShadingBinData)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<FNaniteShadingBinScatterCounters>, OutShadingBinScatterCounters)
	END_SHADER_PARAMETER_STRUCT()
};
IMPLEMENT_GLOBAL_SHADER(FShadingBinValidateCS, "/Engine/Private/Nanite/NaniteShadeBinning.usf", "ShadingBinValidateCS", SF_Compute);

IMPLEMENT_UNIFORM_BUFFER_STRUCT_EX(FComputeShadingOutputs, "ComputeShadingOutputs", FShaderParametersMetadata::EUsageFlags::NeedsReflectedMembers|FShaderParametersMetadata::EUsageFlags::ManuallyBoundByPass);

BEGIN_SHADER_PARAMETER_STRUCT(FNaniteShadingPassParameters, )
	RDG_BUFFER_ACCESS(ShadingDispatchArgs, ERHIAccess::IndirectArgs)

	SHADER_PARAMETER_STRUCT_INCLUDE(FViewShaderParameters, View)	// To access VTFeedbackBuffer
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneUniformParameters, Scene)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FNaniteRasterUniformParameters, NaniteRaster)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FNaniteShadingUniformParameters, NaniteShading)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FOpaqueBasePassUniformParameters, BasePass)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FLumenCardPassUniformParameters, CardPass)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FComputeShadingOutputs, ComputeShadingOutputs)
END_SHADER_PARAMETER_STRUCT()

class FClearCMaskRectCS : public FNaniteGlobalShader
{
	DECLARE_GLOBAL_SHADER(FClearCMaskRectCS);

	class FNumExportsDim : SHADER_PERMUTATION_RANGE_INT("NUM_EXPORTS", 1, MaxSimultaneousRenderTargets);
	using FPermutationDomain = TShaderPermutationDomain<FNumExportsDim>;

	FClearCMaskRectCS() = default;
	FClearCMaskRectCS(const ShaderMetaType::CompiledShaderInitializerType & Initializer)
	: FNaniteGlobalShader(Initializer)
	{
		PlatformDataParam.Bind(Initializer.ParameterMap, TEXT("PlatformData"), SPF_Mandatory);
		BindForLegacyShaderParameters<FParameters>(this, Initializer.PermutationId, Initializer.ParameterMap);
	}

	void SetParameters(FRHIBatchedShaderParameters & BatchedParameters, const void* PlatformDataPtr, uint32 PlatformDataSize)
	{
		BatchedParameters.SetShaderParameter(PlatformDataParam.GetBufferIndex(), PlatformDataParam.GetBaseIndex(), PlatformDataSize, PlatformDataPtr);
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportNanite(Parameters.Platform) && RHISupportsRenderTargetWriteMask(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FNaniteGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("SHADING_BIN_PASS"), NANITE_SHADING_BIN_CMASK_CLEAR);
		OutEnvironment.SetDefine(TEXT("OPTIMIZE_WRITE_MASK"), 1);
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(FIntPoint, ClearTileRectMin)
		SHADER_PARAMETER(FIntPoint, ClearTileRectSize)
		SHADER_PARAMETER_RDG_TEXTURE_UAV_ARRAY(RWTextureMetadata, OutCMaskBuffer, [MaxSimultaneousRenderTargets])
	END_SHADER_PARAMETER_STRUCT()
private:
	LAYOUT_FIELD(FShaderParameter, PlatformDataParam);
	LAYOUT_FIELD(FShaderParameter, SubTileMatchParam);
};
IMPLEMENT_GLOBAL_SHADER(FClearCMaskRectCS, "/Engine/Private/Nanite/NaniteShadeBinning.usf", "ClearCMaskRectCS", SF_Compute);

namespace Nanite
{

bool HasNoDerivativeOps(FRHIComputeShader* ComputeShaderRHI)
{
	if (GNaniteShadeBinningMode == 1)
	{
		return true;
	}
	else if (GNaniteShadeBinningMode == 2)
	{
		return false;
	}
	else
	{
		return ComputeShaderRHI ? ComputeShaderRHI->HasNoDerivativeOps() : false;
	}
}

void BuildShadingCommands(FRDGBuilder& GraphBuilder, FScene& Scene, ENaniteMeshPass::Type MeshPass, FNaniteShadingCommands& ShadingCommands, EBuildShadingCommandsMode Mode)
{
	FNaniteShadingPipelines& ShadingPipelines = Scene.NaniteShadingPipelines[MeshPass];
	if (ShadingPipelines.bBuildCommands || Mode == EBuildShadingCommandsMode::Custom)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE(Nanite::BuildShadingCommands);
		const auto& Pipelines = ShadingPipelines.GetShadingPipelineMap();
		const auto& PipelineGroups = ShadingPipelines.GetShadingPipelineGroupMap();
		const EShaderPlatform ShaderPlatform = Scene.GetShaderPlatform();

		const bool bBindless = ShadingPipelines.UseBindless();

		ShadingCommands.SetupTask = GraphBuilder.AddSetupTask([&ShadingCommands, &Pipelines, &PipelineGroups, ShaderPlatform, bBindless]
		{
			TRACE_CPUPROFILER_EVENT_SCOPE(Nanite::BuildShadingCommandsMetadata);
			ShadingCommands.MaxShadingBin = 0u;
			ShadingCommands.BoundTargetMask = 0x0u;
			ShadingCommands.NumCommands = bBindless ? PipelineGroups.Num() : Pipelines.Num();
			ShadingCommands.bBindless = bBindless;

			for (const auto& Iter : Pipelines)
			{
				const FNaniteShadingEntry& Entry = Iter.Value;
				ShadingCommands.MaxShadingBin = FMath::Max<uint32>(ShadingCommands.MaxShadingBin, uint32(Entry.BinIndex));
				ShadingCommands.BoundTargetMask |= Entry.ShadingPipeline->BoundTargetMask;
			}
			const uint32 NumShadingBins = ShadingCommands.MaxShadingBin + 1u;

			ShadingCommands.MaxShadingGroup = 0u;
			for (const auto& Iter : PipelineGroups)
			{
				const FNaniteShadingGroupEntry& Entry = Iter.Value;
				ShadingCommands.MaxShadingGroup = FMath::Max<uint32>(ShadingCommands.MaxShadingGroup, uint32(Entry.GroupIndex));
			}
			const uint32 NumShadingGroups = ShadingCommands.MaxShadingGroup + 1u;

			// Beware: ShadingCommands.BinMetaData is not necessarily empty here.
			check(NANITE_SHADING_BIN_META_BYTES % 16 == 0);
			ShadingCommands.BinMetaData.SetNumUninitialized(NumShadingGroups + NumShadingBins * NANITE_SHADING_BIN_META_BYTES / 16);
			FUintVector4* GroupMetaData = ShadingCommands.BinMetaData.GetData();
			FNaniteShadingBinMeta* BinMetaData = (FNaniteShadingBinMeta*)(ShadingCommands.BinMetaData.GetData() + NumShadingGroups);

			for (uint32 GroupIdx = 0; GroupIdx < NumShadingGroups; GroupIdx++)
			{
				GroupMetaData[GroupIdx] = FUintVector4(0xFFFFFFFFu, 0u, 0u, 0u);
			}

			for (uint32 BinIdx = 0; BinIdx < NumShadingBins; BinIdx++)
			{
				BinMetaData[BinIdx] = FNaniteShadingBinMeta{.ShadingGroup = NANITE_INVALID_SHADING_GROUP };
			}

			for (const auto& Iter : Pipelines)
			{
				const FNaniteShadingEntry& Entry = Iter.Value;
				FNaniteShadingBinMeta& MetaEntry = BinMetaData[Entry.BinIndex];
				
				MetaEntry.MaterialFlags = Entry.ShadingPipeline->MaterialBitFlags;
				MetaEntry.ShadingGroup = Entry.GroupIndex;
				MetaEntry.MaterialCBIndex = Entry.ShadingPipeline->MaterialCBIndex;
				MetaEntry.QuadBinning = Entry.ShadingPipeline->bNoDerivativeOps ? 0u : 1u;
			}

			// Create Shader Bundle
			if (UseShadingShaderBundle(ShaderPlatform) && ShadingCommands.NumCommands > 0)
			{
				FShaderBundleCreateInfo CreateInfo;
				CreateInfo.ArgOffset = 0u;
				CreateInfo.ArgStride = 16u;
				CreateInfo.NumRecords = bBindless ? NumShadingGroups : NumShadingBins;

				CreateInfo.Mode = ERHIShaderBundleMode::CS;
				ShadingCommands.ShaderBundle = RHICreateShaderBundle(CreateInfo);
				check(ShadingCommands.ShaderBundle != nullptr);
			}
			else
			{
				ShadingCommands.ShaderBundle = nullptr;
			}
		});

		ShadingCommands.BuildCommandsTask = GraphBuilder.AddSetupTask([&Commands = ShadingCommands.Commands, &CommandLookup = ShadingCommands.CommandLookup, &Pipelines, &PipelineGroups, bBindless]
		{
			TRACE_CPUPROFILER_EVENT_SCOPE(Nanite::BuildShadingCommandsTask);
			Commands.Reset();
		
			if (bBindless)
			{
				Commands.Reserve(PipelineGroups.Num());

				uint32 MaxShadingGroup = 0;
				for (const auto& Iter : PipelineGroups)
				{
					FNaniteShadingCommand& ShadingCommand = Commands.AddDefaulted_GetRef();
					const FNaniteShadingGroupEntry& Entry = Iter.Value;
					ShadingCommand.PipelineGroup = Entry.ShadingPipelineGroup;
					ShadingCommand.Pipeline = *Entry.ShadingPipelineGroup->Pipelines.CreateConstIterator();
					ShadingCommand.ArgsIndex = Entry.GroupIndex;

					MaxShadingGroup = FMath::Max<uint32>(MaxShadingGroup, uint32(Entry.GroupIndex));
				}
				CommandLookup.SetNumZeroed(MaxShadingGroup + 1);
			}
			else
			{
				Commands.Reserve(Pipelines.Num());

				uint32 MaxShadingBin = 0;
				for (const auto& Iter : Pipelines)
				{
					FNaniteShadingCommand& ShadingCommand = Commands.AddDefaulted_GetRef();
					const FNaniteShadingEntry& Entry = Iter.Value;
					ShadingCommand.Pipeline = Entry.ShadingPipeline;
					ShadingCommand.ArgsIndex = Entry.BinIndex;

					MaxShadingBin = FMath::Max<uint32>(MaxShadingBin, uint32(Entry.BinIndex));
				}

				CommandLookup.SetNumZeroed(MaxShadingBin + 1);
			}

			if (GNaniteComputeMaterialsSort != 0)
			{
				Commands.Sort([](auto& A, auto& B)
				{
					const FNaniteShadingPipeline& PipelineA = *A.Pipeline.Get();
					const FNaniteShadingPipeline& PipelineB = *B.Pipeline.Get();

					// First group all shaders with the same bound target mask (UAV exports)
					if (PipelineA.BoundTargetMask != PipelineB.BoundTargetMask)
					{
						return PipelineA.BoundTargetMask < PipelineB.BoundTargetMask;
					}

					// Then group up all shading bins using same shader but different bindings
					if (PipelineA.ComputeShader != PipelineB.ComputeShader)
					{
						return PipelineA.ComputeShader < PipelineB.ComputeShader;
					}

					// Sort indirect arg memory location in ascending order to help minimize cache misses on the indirect args
					return A.ArgsIndex < B.ArgsIndex;
				});
			}

			for (int32 CommandIndex = 0; CommandIndex < Commands.Num(); ++CommandIndex)
			{
				const FNaniteShadingCommand& ShadingCommand = Commands[CommandIndex];
				CommandLookup[ShadingCommand.ArgsIndex] = CommandIndex;
			}

		}, ShadingCommands.SetupTask);

		if (Mode == EBuildShadingCommandsMode::Default)
		{
			ShadingPipelines.bBuildCommands = false;

			if (auto MaterialsExtension = Scene.GetExtensionPtr<Nanite::FMaterialsSceneExtension>())
			{
				MaterialsExtension->PostBuildNaniteShadingCommands(GraphBuilder, ShadingCommands.BuildCommandsTask, MeshPass);
			}
		}
	}
}

uint32 PackMaterialBitFlags(const FMaterial& Material, uint32 BoundTargetMask, bool bNoDerivativeOps, bool bVoxel, bool bCurve)
{
	const bool bMaterialHasProgrammableVertexUVs = Material.MaterialUsesVertexInterpolator_RenderThread() || Material.GetNumCustomizedUVs() > 0;

	FNaniteMaterialFlags Flags = { 0 };
	Flags.bPixelDiscard = Material.IsMasked();
	Flags.bPixelDepthOffset = Material.MaterialUsesPixelDepthOffset_RenderThread();
	Flags.bWorldPositionOffset = Material.MaterialUsesWorldPositionOffset_RenderThread();
	Flags.bAllowVRS = Material.IsVariableRateShadingAllowed();
	Flags.bDisplacement = UseNaniteTessellation() && Material.MaterialUsesDisplacement_RenderThread();
	Flags.bNoDerivativeOps = bNoDerivativeOps;
	Flags.bTwoSided = Material.IsTwoSided();
	Flags.bVoxel = bVoxel;
	Flags.bCurve = bCurve;

	const bool bPixelProgrammable = IsNaniteMaterialPixelProgrammable(Flags);
	Flags.bVertexUVs = bMaterialHasProgrammableVertexUVs && bPixelProgrammable;
	Flags.bFirstPersonLerp = Material.HasFirstPersonOutput();

	const uint32 PackedFlags = PackNaniteMaterialBitFlags(Flags);
	return ((BoundTargetMask & 0xFFu) << 24u) | (PackedFlags & 0x00FFFFFFu);
}

bool LoadTranslucencyPassPipeline(
	const FScene& Scene,
	FSceneProxyBase* SceneProxy,
	FSceneProxyBase::FMaterialSection& Section,
	FNaniteRasterPipeline& Pipeline
)
{
	static const bool bAllowStaticLighting = IsStaticLightingAllowed();

	const ERHIFeatureLevel::Type FeatureLevel = Scene.GetFeatureLevel();

	FNaniteTranslucencyFactory* NaniteTranslucencyFactory = Nanite::GTranslucencyFactoryResource.GetVertexFactory();
	FVertexFactoryType* NaniteTranslucencyFactoryType = NaniteTranslucencyFactory->GetType();

	// NOTE: We don't want to fallback to default material (compiler error, etc..), because that will not be a translucent material.
#if 1
	const FMaterialRenderProxy* MaterialProxy = Section.ShadingMaterialProxy;
	if (MaterialProxy)
	{
		const FMaterial* Material = MaterialProxy->GetMaterialNoFallback(FeatureLevel);
		if (!Material)
		{
			MaterialProxy = nullptr;
		}
	}

	if (!MaterialProxy)
	{
		return false;
	}
#else
	const FMaterialRenderProxy* MaterialProxy = Section.ShadingMaterialProxy;
	while (MaterialProxy)
	{
		const FMaterial* Material = MaterialProxy->GetMaterialNoFallback(FeatureLevel);
		if (Material)
		{
			break;
		}
		MaterialProxy = MaterialProxy->GetFallback(FeatureLevel);
	}
	check(MaterialProxy);
#endif

	ELightMapPolicyType LightMapPolicyType = ELightMapPolicyType::LMP_NO_LIGHTMAP;

	FLightCacheInterface* LightCacheInterface = nullptr;
	if (bAllowStaticLighting)
	{
		FPrimitiveSceneProxy::FLCIArray LCIs;
		SceneProxy->GetLCIs(LCIs);

		// We expect a Nanite scene proxy can only ever have a single LCI, or none in cases like skeletal meshes
		check(LCIs.Num() <= 1u);
		if (LCIs.Num() == 1u)
		{
			LightCacheInterface = LCIs[0];
		}
	}

	TShaderRef<TBasePassMeshShaderPolicyParamType<FUniformLightMapPolicy>> TranslucencyMeshShader;
	TShaderRef<TBasePassVertexShaderPolicyParamType<FUniformLightMapPolicy>> TranslucencyVertexShader;
	TShaderRef<TBasePassPixelShaderPolicyParamType<FUniformLightMapPolicy>> TranslucencyPixelShader;

	const bool bMSSupported = GRHISupportsMeshShadersTier0;

	auto LoadTranslucentMaterial = [&](const FMaterialRenderProxy* MaterialProxyPtr)
	{
		const FMaterial& ShadingMaterial = MaterialProxy->GetIncompleteMaterialWithFallback(FeatureLevel);
		checkf(Nanite::IsSupportedMaterialDomain(ShadingMaterial.GetMaterialDomain()), TEXT("Material '%s' uses unsupported material domain '%s'."),
			*ShadingMaterial.GetFriendlyName(), *MaterialDomainString(ShadingMaterial.GetMaterialDomain()));
		checkf(IsTranslucentOnlyBlendMode(ShadingMaterial), TEXT("Material '%s' uses invalid blend mode '%s'."),
			*ShadingMaterial.GetFriendlyName(), *GetBlendModeString(ShadingMaterial.GetBlendMode()));

		const FMaterialShadingModelField ShadingModels = ShadingMaterial.GetShadingModels();		

		if (LightCacheInterface)
		{
			LightMapPolicyType = FBasePassMeshProcessor::GetUniformLightMapPolicyType(FeatureLevel, &Scene, LightCacheInterface, SceneProxy, ShadingMaterial);
		}

		bool bRequiresExplicit128bitRT = false; // TODO: bRequiresExplicit128bitRT((Flags & EFlags::bRequires128bitRT) == EFlags::bRequires128bitRT)
		// Use project-level check so the OIT permutation is always cached when OIT is project-supported.
		// Runtime enable/disable is handled by bOITEnable in FTranslucentBasePassUniformParameters::OIT,
		// avoiding stale non-OIT shaders in the pipeline cache when r.OIT.SortedPixels.Enable is toggled.
		bool bOITBasePass = OIT::IsSortedPixelsEnabledForProject(GetFeatureLevelShaderPlatform(FeatureLevel));

		bool bShadersValid = GetBasePassShaders<FUniformLightMapPolicy>(
			ShadingMaterial,
			NaniteTranslucencyFactoryType,
			FUniformLightMapPolicy(LightMapPolicyType),
			FeatureLevel,
			bRequiresExplicit128bitRT,
			Scene.RequiresDebugMaterials(),
			GBL_Default,
			bMSSupported ? &TranslucencyMeshShader : nullptr,
			&TranslucencyVertexShader,
			&TranslucencyPixelShader,
			bOITBasePass
		);

		return bShadersValid;
	};

	bool bLoaded = LoadTranslucentMaterial(MaterialProxy);
	if (!bLoaded)
	{
		//MaterialProxy = UMaterial::GetDefaultMaterial(MD_Surface)->GetRenderProxy();
		//bLoaded = LoadTranslucentMaterial(MaterialProxy);
	}

	if (bLoaded)
	{
		Pipeline.TranslucencyPassData = MakePimpl<FNaniteTranslucencyPassData, EPimplPtrMode::DeepCopy>();
		Pipeline.TranslucencyPassData->TypedMeshShader = TranslucencyMeshShader;
		Pipeline.TranslucencyPassData->TypedVertexShader = TranslucencyVertexShader;
		Pipeline.TranslucencyPassData->TypedPixelShader = TranslucencyPixelShader;
		Pipeline.TranslucencyPassData->MaterialProxy = MaterialProxy;
		Pipeline.TranslucencyPassData->Material = MaterialProxy->GetMaterialNoFallback(FeatureLevel);
		check(Pipeline.TranslucencyPassData->Material != nullptr);

		TBasePassShaderElementData<FUniformLightMapPolicy> ShaderElementData(LightCacheInterface);
		ShaderElementData.InitializeMeshMaterialData();

		// Mesh Shader Path
		if (Pipeline.TranslucencyPassData->TypedMeshShader.IsValid())
		{
			FMeshProcessorShaders MeshProcessorShaders;
			MeshProcessorShaders.MeshShader = Pipeline.TranslucencyPassData->TypedMeshShader;
			MeshProcessorShaders.PixelShader = Pipeline.TranslucencyPassData->TypedPixelShader;

			Pipeline.TranslucencyPassData->ShaderBindingsMSPS.Initialize(MeshProcessorShaders);

			int32 BindingDataOffset = 0;

			// Mesh
			{
				FMeshDrawSingleShaderBindings SingleShaderBindings = Pipeline.TranslucencyPassData->ShaderBindingsMSPS.GetSingleShaderBindings(SF_Mesh, BindingDataOffset);
				Pipeline.TranslucencyPassData->TypedMeshShader->GetShaderBindings(
					&Scene,
					FeatureLevel,
					SceneProxy,
					*Pipeline.TranslucencyPassData->MaterialProxy,
					*Pipeline.TranslucencyPassData->Material,
					ShaderElementData,
					SingleShaderBindings
				);
			}

			// Pixel
			{
				FMeshDrawSingleShaderBindings SingleShaderBindings = Pipeline.TranslucencyPassData->ShaderBindingsMSPS.GetSingleShaderBindings(SF_Pixel, BindingDataOffset);
				Pipeline.TranslucencyPassData->TypedPixelShader->GetShaderBindings(
					&Scene,
					FeatureLevel,
					SceneProxy,
					*Pipeline.TranslucencyPassData->MaterialProxy,
					*Pipeline.TranslucencyPassData->Material,
					ShaderElementData,
					SingleShaderBindings
				);
			}

			Pipeline.TranslucencyPassData->ShaderBindingsMSPS.Finalize(&MeshProcessorShaders);
		}

		// Vertex Shader Path
		if (Pipeline.TranslucencyPassData->TypedVertexShader.IsValid())
		{
			FMeshProcessorShaders MeshProcessorShaders;
			MeshProcessorShaders.VertexShader = Pipeline.TranslucencyPassData->TypedVertexShader;
			MeshProcessorShaders.PixelShader = Pipeline.TranslucencyPassData->TypedPixelShader;

			Pipeline.TranslucencyPassData->ShaderBindingsVSPS.Initialize(MeshProcessorShaders);

			int32 BindingDataOffset = 0;

			// Vertex
			{
				FMeshDrawSingleShaderBindings SingleShaderBindings = Pipeline.TranslucencyPassData->ShaderBindingsVSPS.GetSingleShaderBindings(SF_Vertex, BindingDataOffset);
				Pipeline.TranslucencyPassData->TypedVertexShader->GetShaderBindings(
					&Scene,
					FeatureLevel,
					SceneProxy,
					*Pipeline.TranslucencyPassData->MaterialProxy,
					*Pipeline.TranslucencyPassData->Material,
					ShaderElementData,
					SingleShaderBindings
				);
			}

			// Pixel
			{
				FMeshDrawSingleShaderBindings SingleShaderBindings = Pipeline.TranslucencyPassData->ShaderBindingsVSPS.GetSingleShaderBindings(SF_Pixel, BindingDataOffset);
				Pipeline.TranslucencyPassData->TypedPixelShader->GetShaderBindings(
					&Scene,
					FeatureLevel,
					SceneProxy,
					*Pipeline.TranslucencyPassData->MaterialProxy,
					*Pipeline.TranslucencyPassData->Material,
					ShaderElementData,
					SingleShaderBindings
				);
			}

			Pipeline.TranslucencyPassData->ShaderBindingsVSPS.Finalize(&MeshProcessorShaders);
		}
	}

	return bLoaded;
}

bool LoadBasePassPipeline(
	const FScene& Scene,
	FSceneProxyBase* SceneProxy,
	FSceneProxyBase::FMaterialSection& Section,
	bool bVoxel,
	bool bCurve,
	FNaniteShadingPipeline& ShadingPipeline
)
{
	static const bool bAllowStaticLighting = IsStaticLightingAllowed();

	const ERHIFeatureLevel::Type FeatureLevel = Scene.GetFeatureLevel();

	FNaniteVertexFactory* NaniteVertexFactory = Nanite::GVertexFactoryResource.GetVertexFactory();
	FVertexFactoryType* NaniteVertexFactoryType = NaniteVertexFactory->GetType();

	const FMaterialRenderProxy* MaterialProxy = Section.ShadingMaterialProxy;
	while (MaterialProxy)
	{
		const FMaterial* Material = MaterialProxy->GetMaterialNoFallback(FeatureLevel);
		if (Material)
		{
			break;
		}
		MaterialProxy = MaterialProxy->GetFallback(FeatureLevel);
	}

	check(MaterialProxy);

	ELightMapPolicyType LightMapPolicyType = ELightMapPolicyType::LMP_NO_LIGHTMAP;

	FLightCacheInterface* LightCacheInterface = nullptr;
	if (bAllowStaticLighting)
	{
		FPrimitiveSceneProxy::FLCIArray LCIs;
		SceneProxy->GetLCIs(LCIs);

		// We expect a Nanite scene proxy can only ever have a single LCI, or none in cases like skeletal meshes
		check(LCIs.Num() <= 1u);
		if (LCIs.Num() == 1u)
		{
			LightCacheInterface = LCIs[0];
		}
	}

	const bool bUseWorkGraphShaders = UseWorkGraphForNaniteShading(Scene.GetShaderPlatform());
	TShaderRef<TBasePassComputeShaderPolicyParamType<FUniformLightMapPolicy>> BasePassShader;

	auto LoadShadingMaterial = [&](const FMaterialRenderProxy* MaterialProxyPtr)
	{
		const FMaterial& ShadingMaterial = MaterialProxy->GetIncompleteMaterialWithFallback(FeatureLevel);
		checkf(Nanite::IsSupportedMaterialDomain(ShadingMaterial.GetMaterialDomain()), TEXT("Material '%s' uses unsupported material domain '%s'."),
				*ShadingMaterial.GetFriendlyName(), *MaterialDomainString(ShadingMaterial.GetMaterialDomain()));
		checkf(Nanite::IsSupportedBlendMode(ShadingMaterial), TEXT("Material '%s' uses unsupported blend mode '%s'."),
				*ShadingMaterial.GetFriendlyName(), *GetBlendModeString(ShadingMaterial.GetBlendMode()));

		const FMaterialShadingModelField ShadingModels = ShadingMaterial.GetShadingModels();

		if (LightCacheInterface)
		{
			LightMapPolicyType = FBasePassMeshProcessor::GetUniformLightMapPolicyType(FeatureLevel, &Scene, LightCacheInterface, SceneProxy, ShadingMaterial);
		}

		bool bShadersValid = GetBasePassShader<FUniformLightMapPolicy>(
				ShadingMaterial,
				NaniteVertexFactoryType,
				FUniformLightMapPolicy(LightMapPolicyType),
				FeatureLevel,
				bVoxel,
				bCurve,
				Scene.RequiresDebugMaterials(),
				bUseWorkGraphShaders ? SF_WorkGraphComputeNode : SF_Compute,
				&BasePassShader
				);

		return bShadersValid;
	};

	bool bLoaded = LoadShadingMaterial(MaterialProxy);
	if (!bLoaded)
	{
		MaterialProxy = UMaterial::GetDefaultMaterial(MD_Surface)->GetRenderProxy();
		bLoaded = LoadShadingMaterial(MaterialProxy);
	}

	if (bLoaded)
	{
		ShadingPipeline.MaterialProxy		= MaterialProxy;
		ShadingPipeline.Material			= MaterialProxy->GetMaterialNoFallback(FeatureLevel);
		ShadingPipeline.BoundTargetMask		= BasePassShader->GetBoundTargetMask();
		ShadingPipeline.ComputeShader		= bUseWorkGraphShaders ? nullptr : BasePassShader.GetComputeShader();
		ShadingPipeline.WorkGraphShader		= bUseWorkGraphShaders ? BasePassShader.GetWorkGraphShader() : nullptr;
		ShadingPipeline.bIsTwoSided			= !!Section.MaterialRelevance.bTwoSided;
		ShadingPipeline.bIsMasked			= !!Section.MaterialRelevance.bMasked;
		ShadingPipeline.bNoDerivativeOps	= HasNoDerivativeOps(ShadingPipeline.ComputeShader);
		ShadingPipeline.bVoxel				= bVoxel;
		ShadingPipeline.bCurve				= bCurve;
		ShadingPipeline.MaterialBitFlags	= PackMaterialBitFlags(*ShadingPipeline.Material, ShadingPipeline.BoundTargetMask, ShadingPipeline.bNoDerivativeOps, bVoxel, bCurve);

		ShadingPipeline.BasePassData = MakePimpl<FNaniteBasePassData, EPimplPtrMode::DeepCopy>();
		ShadingPipeline.BasePassData->TypedShader = BasePassShader;

#if WITH_DEBUG_VIEW_MODES
		ShadingPipeline.InstructionCount = BasePassShader->GetNumInstructions();
		ShadingPipeline.LWCComplexity = 0;
#if WITH_EDITOR
		FMaterialShaderMap* MaterialShaderMap = ShadingPipeline.Material->GetRenderingThreadShaderMap();
		if (ensure(MaterialShaderMap))
		{
			uint32 LWCComplexityVS = 0;
			uint32 LWCComplexityPS = 0;
			uint32 LWCComplexityCS = 0;

			MaterialShaderMap->GetEstimatedLWCFuncUsageComplexity(LWCComplexityVS, LWCComplexityPS, LWCComplexityCS);

			ShadingPipeline.LWCComplexity = static_cast<uint16>(FMath::Clamp(LWCComplexityCS + LWCComplexityPS, 1, TNumericLimits<uint16>::Max()));
		}
#endif
#endif

		TBasePassShaderElementData<FUniformLightMapPolicy> ShaderElementData(LightCacheInterface);
		ShaderElementData.InitializeMeshMaterialData();

		ShadingPipeline.ShaderBindings = MakePimpl<FMeshDrawShaderBindings, EPimplPtrMode::DeepCopy>();

		UE::MeshPassUtils::SetupComputeBindings(BasePassShader, &Scene, FeatureLevel, SceneProxy, *MaterialProxy, *ShadingPipeline.Material, ShaderElementData, *ShadingPipeline.ShaderBindings);

		ShadingPipeline.ShaderBindingsHash = ShadingPipeline.ShaderBindings->GetDynamicInstancingHash();

		// TODO: Is there a more sensible way to get to the Material CB?
		ShadingPipeline.MaterialUB = MaterialProxy->UniformExpressionCache[FeatureLevel].UniformBuffer;
		FRHIDescriptorHandle Handle = ShadingPipeline.MaterialUB->GetBindlessHandle();
		ShadingPipeline.MaterialCBIndex = Handle.IsValid() ? Handle.GetIndex() : 0xFFFFFFFFu;
		ShadingPipeline.MaterialUBSerialNumber = MaterialProxy->GetExpressionCacheSerialNumber();
	}

	return bLoaded;
}

struct FShadingConfig
{
	ERHIShaderBundleMethod BundleMethod;
	uint8 bBundleShading	: 1;
	uint8 bHighPrecision	: 1;
	uint8 bShowDrawEvents	: 1;
};

inline void RecordShadingParameters(
	FRHIBatchedShaderParameters& BatchedParameters,
	FNaniteShadingCommand& ShadingCommand,
	const FShadingConfig& ShadingConfig,
	const uint32 NumShadingBins,
	const uint32 NumShadingGroups,
	const FUint32Vector4& ViewRect,
	TUniformBufferRef<FComputeShadingOutputs> OutputTargetsBuffer
)
{
	ShadingCommand.PassData.X = ShadingCommand.ArgsIndex;	// Shading Bin / Shading Group
	ShadingCommand.PassData.Y = NumShadingGroups;
	ShadingCommand.PassData.Z = NumShadingBins;
	ShadingCommand.PassData.W = (ShadingCommand.Pipeline->bNoDerivativeOps != 0u ? 0u : 1u) | (ShadingConfig.bHighPrecision ? 2u : 0u);

	const bool bBindless = ShadingCommand.IsBindless();
	if (bBindless)
	{
		for (const auto& GroupPipeline : ShadingCommand.PipelineGroup->Pipelines)
		{
			if (FRHIUniformBuffer* MaterialUB = GroupPipeline->MaterialUB)
			{
				if (MaterialUB->GetBindlessHandle().IsValid())
				{
					BatchedParameters.SetBindlessShaderUniformBuffer(MaterialUB);
				}
			}
		}
	}

	ShadingCommand.Pipeline->ShaderBindings->SetParameters(BatchedParameters);

	if (ShadingCommand.Pipeline->ComputeShader || ShadingCommand.Pipeline->WorkGraphShader)
	{
		ShadingCommand.Pipeline->BasePassData->TypedShader->SetPassParameters(	//RUNETODO: Is it safe to use BasePassData from the pipeline?
			BatchedParameters,
			ViewRect,
			ShadingCommand.PassData,
			OutputTargetsBuffer.GetReference()
		);
	}
}

inline void RecordShadingCommand(
	FRHIComputeCommandList& RHICmdList,
	FRHIBuffer* IndirectArgsBuffer,
	const uint32 IndirectArgStride,
	const FShadingConfig& ShadingConfig,
	FRHIBatchedShaderParameters& ShadingParameters,
	FNaniteShadingCommand& ShadingCommand
)
{
#if WANTS_DRAW_MESH_EVENTS
	SCOPED_CONDITIONAL_DRAW_EVENTF(RHICmdList, SWShading, !!ShadingConfig.bShowDrawEvents && !ShadingCommand.Pipeline->bVoxel && !ShadingCommand.Pipeline->bCurve, TEXT("%s"),			GetShadingMaterialName(ShadingCommand.Pipeline->MaterialProxy));
	SCOPED_CONDITIONAL_DRAW_EVENTF(RHICmdList, SWShading, !!ShadingConfig.bShowDrawEvents &&  ShadingCommand.Pipeline->bVoxel && !ShadingCommand.Pipeline->bCurve, TEXT("%s (Voxels)"),	GetShadingMaterialName(ShadingCommand.Pipeline->MaterialProxy));
	SCOPED_CONDITIONAL_DRAW_EVENTF(RHICmdList, SWShading, !!ShadingConfig.bShowDrawEvents && !ShadingCommand.Pipeline->bVoxel &&  ShadingCommand.Pipeline->bCurve, TEXT("%s (Curves)"),	GetShadingMaterialName(ShadingCommand.Pipeline->MaterialProxy));
#endif

	const uint32 IndirectOffset = (ShadingCommand.ArgsIndex * IndirectArgStride);

	FRHIComputeShader* ComputeShaderRHI = ShadingCommand.Pipeline->ComputeShader;
	SetComputePipelineState(RHICmdList, ComputeShaderRHI);

	if (GRHISupportsShaderRootConstants)
	{
		RHICmdList.SetShaderRootConstants(ShadingCommand.PassData);
	}

	RHICmdList.SetBatchedShaderParameters(ComputeShaderRHI, ShadingParameters);
	RHICmdList.DispatchIndirectComputeShader(IndirectArgsBuffer, IndirectOffset);
}

inline bool PrepareShadingCommand(FNaniteShadingCommand& ShadingCommand, int32 PSOCollectorIndex)
{
	if (!PipelineStateCache::IsPSOPrecachingEnabled())
	{
		ShadingCommand.PSOPrecacheState = EPSOPrecacheResult::Unknown;
		return true;
	}

	EPSOPrecacheResult PSOPrecacheResult = ShadingCommand.PSOPrecacheState;
	bool bShouldCheckPrecacheResult = false;

	// If PSO precache validation is on, we need to check the state for stats tracking purposes.
#if PSO_PRECACHING_VALIDATE
	if (PSOCollectorStats::IsPrecachingValidationEnabled() && PSOPrecacheResult == EPSOPrecacheResult::Unknown)
	{
		bShouldCheckPrecacheResult = true;
	}
#endif

	// If we are skipping commands when the PSO is being precached but is not ready, we
	// need to keep checking the state until it's not marked active anymore.
	const bool bAllowSkip = true;
	if (bAllowSkip && GSkipDrawOnPSOPrecaching)
	{
		if (PSOPrecacheResult == EPSOPrecacheResult::Unknown ||
			PSOPrecacheResult == EPSOPrecacheResult::Active)
		{
			bShouldCheckPrecacheResult = true;
		}
	}

	if (bShouldCheckPrecacheResult)
	{
		// Cache the state so that it's only checked again if necessary.
		PSOPrecacheResult = PipelineStateCache::CheckPipelineStateInCache(ShadingCommand.Pipeline->ComputeShader);
		ShadingCommand.PSOPrecacheState = PSOPrecacheResult;
	}

#if PSO_PRECACHING_VALIDATE
	PSOCollectorStats::CheckComputePipelineStateInCache(*ShadingCommand.Pipeline->ComputeShader, PSOPrecacheResult, ShadingCommand.Pipeline->MaterialProxy, PSOCollectorIndex);
#endif

	// Try and skip draw if the PSO is not precached yet.
	const bool bSkipped = (bAllowSkip && GSkipDrawOnPSOPrecaching && PSOPrecacheResult == EPSOPrecacheResult::Active);
	return !bSkipped;
}

struct FNaniteShadingPassIntermediates
{
	TUniformBufferRef<FComputeShadingOutputs> ShadingOutputs;
	TBitArray<SceneRenderingBitArrayAllocator> VisibilityData;
	FRHIBuffer* IndirectArgsBuffer = nullptr;
	FUint32Vector4 ViewRect;
};

static TSharedPtr<FNaniteShadingPassIntermediates> CreateNaniteShadingPassIntermediates(
	const FNaniteShadingPassParameters* ShadingPassParameters,
	const FNaniteShadingCommands& ShadingCommands,
	const FNaniteVisibilityQuery* VisibilityQuery,
	FIntRect ViewRect)
{
	// This is processed within the RDG pass lambda, so the setup task should be complete by now.
	check(ShadingCommands.BuildCommandsTask.IsCompleted());

	TSharedPtr<FNaniteShadingPassIntermediates> Intermediates = MakeShared<FNaniteShadingPassIntermediates>();

	ShadingPassParameters->ShadingDispatchArgs->MarkResourceAsUsed();
	Intermediates->IndirectArgsBuffer = ShadingPassParameters->ShadingDispatchArgs->GetIndirectRHICallBuffer();

	const auto GetOutputTargetRHI = [](const FRDGTextureUAVRef OutputTarget)
	{
		FRHIUnorderedAccessView* OutputTargetRHI = nullptr;
		if (OutputTarget != nullptr)
		{
			OutputTarget->MarkResourceAsUsed();
			OutputTargetRHI = OutputTarget->GetRHI();
		}
		return OutputTargetRHI;
	};

	const FNaniteVisibilityResults* VisibilityResults = Nanite::GetVisibilityResults(VisibilityQuery);

	TSharedPtr<TBitArray<SceneRenderingBitArrayAllocator>> VisibilityData;
	if (VisibilityResults && VisibilityResults->IsShadingTestValid())
	{
		Intermediates->VisibilityData = VisibilityResults->GetShadingBinVisibility();
	}

	TRDGUniformBufferRef<FComputeShadingOutputs> ShadingOutputs = ShadingPassParameters->ComputeShadingOutputs.GetUniformBuffer();
	Intermediates->ShadingOutputs = ShadingOutputs->GetRHIRef();

	Intermediates->ViewRect = FUint32Vector4(
		(uint32)ViewRect.Min.X,
		(uint32)ViewRect.Min.Y,
		(uint32)ViewRect.Max.X,
		(uint32)ViewRect.Max.Y
	);

	return Intermediates;
};

inline bool IsCommandVisible(FNaniteShadingCommand& ShadingCommand, const FNaniteShadingPassIntermediates& Intermediates)
{
	if (Intermediates.VisibilityData.IsEmpty())
	{
		return true;
	}

	if (ShadingCommand.IsBindless())
	{
		for (const uint32 ShadingBin : ShadingCommand.PipelineGroup->PipelineShadingBins)
		{
			if (Intermediates.VisibilityData.AccessCorrespondingBit(FRelativeBitReference(ShadingBin)))
			{
				return true;
			}
		}
		return false;
	}
	else
	{
		const uint32 ShadingBin = ShadingCommand.ArgsIndex;
		return Intermediates.VisibilityData.AccessCorrespondingBit(FRelativeBitReference(ShadingBin));
	}
}


static void DispatchComputeShaderBundle(
	FRHIComputeCommandList& RHICmdList,
	FNaniteShadingCommands& ShadingCommands,
	const FShadingConfig& ShadingConfig,
	const FShaderBundleRHIRef& ShaderBundle,
	const FNaniteShadingPassIntermediates& Intermediates,
	int32 PSOCollectorIndex,
	EParallelForFlags ParallelForFlags = EParallelForFlags::None)
{
	RHICmdList.DispatchComputeShaderBundle([&](FRHICommandDispatchComputeShaderBundle& Command)
	{
		Command.ShaderBundle		= ShaderBundle;
		Command.Method				= ShadingConfig.BundleMethod;
		Command.RecordArgBuffer		= Intermediates.IndirectArgsBuffer;
		Command.Dispatches.SetNum(ShaderBundle->NumRecords);

		std::atomic<uint32> PendingPSOs{ 0u };

		const bool bBindless = ShadingCommands.bBindless;

		TArray<FRHIBatchedShaderParametersAllocator*, SceneRenderingAllocator> Allocators;

		ParallelForWithTaskContext(TEXT("RecordShadingCommands"), Allocators, ShadingCommands.Commands.Num(), 1,
			[&] (int32, int32)
			{
				// Use the large page size for the allocator to reduce allocations
				return RHICmdList.CreateBatchedShaderParameterAllocator(ERHIBatchedShaderParameterAllocatorPageSize::Large);
			},
			[&](FRHIBatchedShaderParametersAllocator* ParameterAllocator, int32 CommandIndex)
			{
				FNaniteShadingCommand& ShadingCommand = ShadingCommands.Commands[CommandIndex];
				ShadingCommand.bVisible = IsCommandVisible(ShadingCommand, Intermediates);

				const uint32 DispatchIndex = ShadingCommand.ArgsIndex;
								
				if (ShadingCommand.bVisible && PrepareShadingCommand(ShadingCommand, PSOCollectorIndex))
				{
					FRHIShaderBundleComputeDispatch& Dispatch = Command.Dispatches[DispatchIndex];
					const uint32 NumShadingBins = ShadingCommands.MaxShadingBin + 1;
					const uint32 NumShadingGroups = ShadingCommands.MaxShadingGroup + 1;

					Dispatch.RecordIndex = DispatchIndex;
					Dispatch.Parameters.Emplace(*ParameterAllocator);
					RecordShadingParameters(*Dispatch.Parameters, ShadingCommand, ShadingConfig, NumShadingBins, NumShadingGroups, Intermediates.ViewRect, Intermediates.ShadingOutputs);
					Dispatch.Parameters->Finish();
					Dispatch.Shader				= bBindless ? ShadingCommand.PipelineGroup->ComputeShader : ShadingCommand.Pipeline->ComputeShader;
					Dispatch.WorkGraphShader	= bBindless ? ShadingCommand.PipelineGroup->WorkGraphShader : ShadingCommand.Pipeline->WorkGraphShader;
					Dispatch.Constants = ShadingCommand.PassData;
					Dispatch.PipelineState = Dispatch.Shader ? FindComputePipelineState(Dispatch.Shader) : nullptr;

					if (Dispatch.Shader)
					{
						PendingPSOs.fetch_add(1u, std::memory_order_relaxed);
					}
				}
				else
				{
					// TODO: Optimization: Send partial dispatch lists, but for now we'll leave the record index invalid so bundle dispatch skips it
					Command.Dispatches[DispatchIndex].RecordIndex = ~uint32(0u);
				}
			}
		);

		// Resolve invalid pipeline states
		if (PendingPSOs.load(std::memory_order_relaxed) > 0)
		{
			for (FRHIShaderBundleComputeDispatch& Dispatch : Command.Dispatches)
			{
				if (!Dispatch.IsValid() || Dispatch.PipelineState != nullptr)
				{
					continue;
				}

				// If we don't have precaching, then GetComputePipelineState() might return a PipelineState that isn't ready.
				const bool bSkipDraw = !PipelineStateCache::IsPSOPrecachingEnabled();

				// This cache lookup cannot be parallelized due to the possibility of a fence insertion into the command list during a miss.
				Dispatch.PipelineState = GetComputePipelineState(RHICmdList, Dispatch.Shader, !bSkipDraw);

				if (bSkipDraw)
				{
					Dispatch.RecordIndex = ~uint32(0u);
					continue;
				}

				if (Dispatch.Shader && RHICmdList.Bypass())
				{
					Dispatch.RHIPipeline = ExecuteSetComputePipelineState(Dispatch.PipelineState);
				}
			}
		}
	});
}

FNaniteShadingPassParameters CreateNaniteShadingPassParams(
	FRDGBuilder& GraphBuilder,
	const FSceneRenderer& SceneRenderer,
	const FSceneTextures& SceneTextures,
	const FViewInfo& View,
	const FIntRect ViewRect,
	const FRasterResults& RasterResults,
	FRDGTextureRef ShadingMask,
	FRDGTextureRef VisBuffer64,
	FRDGTextureRef DbgBuffer64,
	FRDGTextureRef DbgBuffer32,
	FRDGBufferRef VisibleClustersSWHW,
	FRDGBufferRef AssemblyTransforms,
	FRDGBufferRef MultiViewIndices,
	FRDGBufferRef MultiViewRectScaleOffsets,
	FRDGBufferRef ViewsBuffer,
	const FRenderTargetBindingSlots& BasePassRenderTargets,
	const uint32 BoundTargetMask,
	const FShadeBinning& ShadeBinning,
	const bool SubstrateAdaptativeGbufferEnabled
)
{
	FNaniteShadingPassParameters Result;

	Result.ShadingDispatchArgs = ShadeBinning.ShadingDispatchArgs;

	// NaniteRaster Uniform Buffer
	{
		FNaniteRasterUniformParameters* UniformParameters = GraphBuilder.AllocParameters<FNaniteRasterUniformParameters>();
		UniformParameters->PageConstants = RasterResults.PageConstants;
		UniformParameters->MaxNodes = RasterResults.MaxNodes;
		UniformParameters->MaxVisibleClusters = RasterResults.MaxVisibleClusters;
		UniformParameters->MaxCandidatePatches = RasterResults.MaxCandidatePatches;
		UniformParameters->MaxPatchesPerGroup = RasterResults.MaxPatchesPerGroup;
		UniformParameters->MeshPass = RasterResults.MeshPass;
		UniformParameters->InvDiceRate = RasterResults.InvDiceRate;
		UniformParameters->RenderFlags = RasterResults.RenderFlags;
		UniformParameters->DebugFlags = RasterResults.DebugFlags;
		UniformParameters->CurveSubdivisionThreshold = Nanite::FGlobalResources::GetCurveSubdivisionThreshold();
		UniformParameters->CurveSubdivisionMaxSegments = Nanite::FGlobalResources::GetCurveSubdivisionMaxSegments();
		Result.NaniteRaster = GraphBuilder.CreateUniformBuffer(UniformParameters);
	}

	// NaniteShading Uniform Buffer
	{
		FNaniteShadingUniformParameters* UniformParameters = GraphBuilder.AllocParameters<FNaniteShadingUniformParameters>();
	
		UniformParameters->ClusterPageData = Nanite::GStreamingManager.GetClusterPageDataSRV(GraphBuilder);
		UniformParameters->HierarchyBuffer = Nanite::GStreamingManager.GetHierarchySRV(GraphBuilder);
		UniformParameters->VisibleClustersSWHW = GraphBuilder.CreateSRV(VisibleClustersSWHW);
		UniformParameters->AssemblyTransforms = GraphBuilder.CreateSRV(AssemblyTransforms);
		
		UniformParameters->VisBuffer64 = VisBuffer64;
		UniformParameters->DbgBuffer64 = DbgBuffer64;
		UniformParameters->DbgBuffer32 = DbgBuffer32;

		UniformParameters->ShadingMask = ShadingMask;

		UniformParameters->MultiViewEnabled = 0;
		UniformParameters->MultiViewIndices = GraphBuilder.CreateSRV(MultiViewIndices);
		UniformParameters->MultiViewRectScaleOffsets = GraphBuilder.CreateSRV(MultiViewRectScaleOffsets);
		UniformParameters->InViews = GraphBuilder.CreateSRV(ViewsBuffer);

		UniformParameters->ShadingBinData = GraphBuilder.CreateSRV(ShadeBinning.ShadingBinData);

		if(ShadeBinning.ThreadGroupData)
		{
			UniformParameters->ThreadGroupData = GraphBuilder.CreateSRV(ShadeBinning.ThreadGroupData);
		}
		else
		{
			UniformParameters->ThreadGroupData = GraphBuilder.CreateSRV(GSystemTextures.GetDefaultStructuredBuffer<uint32>(GraphBuilder));
		}

		Result.NaniteShading = GraphBuilder.CreateUniformBuffer(UniformParameters);
	}

	Result.View = View.GetShaderParameters(); // To get VTFeedbackBuffer
	Result.Scene = View.GetSceneUniforms().GetBuffer(GraphBuilder);
	
	// Skip base pass skylight if Lumen GI is enabled, as Lumen handles the skylight.
	const FScene* Scene = View.Family->Scene ? View.Family->Scene->GetRenderScene() : nullptr;
	const bool bShouldRenderSkylightInBasePass = Scene ? Scene->ShouldRenderSkylightInBasePass(/*bIsTranslucent*/false) : true;
	const bool bLumenGIEnabled = SceneRenderer.IsLumenGIEnabled(View);
	const bool bEnableSkylight = bShouldRenderSkylightInBasePass && !bLumenGIEnabled;
	
	Result.BasePass = CreateOpaqueBasePassUniformBuffer(GraphBuilder, View, 0, {}, *SceneTextures.DBufferTextures, bEnableSkylight);

	FComputeShadingOutputs* ShadingOutputs = GraphBuilder.AllocParameters<FComputeShadingOutputs>();

	// No possibility of read/write hazard due to fully resolved vbuffer/materials
	const ERDGUnorderedAccessViewFlags OutTargetFlags = ERDGUnorderedAccessViewFlags::SkipBarrier;

	FRDGTextureUAVRef DummyUAV{};
	auto GetDummyUAV = [&DummyUAV, &GraphBuilder, OutTargetFlags]()
	{
		if (!DummyUAV)
		{
			FRDGTextureDesc DummyDesc = FRDGTextureDesc::Create2D(
				FIntPoint(1u, 1u),
				PF_R32_UINT,
				FClearValueBinding::Transparent,
				TexCreate_ShaderResource | TexCreate_UAV
			);

			DummyUAV = GraphBuilder.CreateUAV(GraphBuilder.CreateTexture(DummyDesc, TEXT("Nanite.TargetDummy")), OutTargetFlags);
		}
		return DummyUAV;
	};

	if (SubstrateAdaptativeGbufferEnabled)
	{
		const FSubstrateSceneData& SubstrateSceneData = *SceneTextures.SubstrateSceneData;
		ShadingOutputs->OutTargets = GraphBuilder.CreateUAV(SubstrateSceneData.MaterialTextureArray, OutTargetFlags);
		ShadingOutputs->OutTopLayerTarget = GraphBuilder.CreateUAV(SubstrateSceneData.TopLayerTexture, OutTargetFlags);
	}
	else
	{
		ShadingOutputs->OutTargets = GetDummyUAV();
		ShadingOutputs->OutTopLayerTarget = GetDummyUAV();
	}

#if NANITE_USE_ANALYTIC_SGGX
	// Optional SGGX distribution output
	ShadingOutputs->OutSGGX = GraphBuilder.CreateUAV(SceneRenderer.GetActiveSceneTextures().GBufferSGGX);
#else
	ShadingOutputs->OutSGGX = GetDummyUAV();
#endif

	const bool bMaintainCompression = (GNaniteFastTileClear == 2) && RHISupportsRenderTargetWriteMask(GMaxRHIShaderPlatform);

	FRDGTextureUAVRef* OutTargets[MaxSimultaneousRenderTargets] =
	{
		&ShadingOutputs->OutTarget0,
		&ShadingOutputs->OutTarget1,
		&ShadingOutputs->OutTarget2,
		&ShadingOutputs->OutTarget3,
		&ShadingOutputs->OutTarget4,
		&ShadingOutputs->OutTarget5,
		&ShadingOutputs->OutTarget6,
		&ShadingOutputs->OutTarget7
	};

	for (uint32 TargetIndex = 0; TargetIndex < MaxSimultaneousRenderTargets; ++TargetIndex)
	{
		if (FRDGTexture* TargetTexture = BasePassRenderTargets.Output[TargetIndex].GetTexture())
		{
			if ((BoundTargetMask & (1u << TargetIndex)) == 0u)
			{
				*OutTargets[TargetIndex] = GetDummyUAV();
			}
			else if (bMaintainCompression)
			{
				*OutTargets[TargetIndex] = GraphBuilder.CreateUAV(FRDGTextureUAVDesc::CreateForMetaData(TargetTexture, ERDGTextureMetaDataAccess::PrimaryCompressed), OutTargetFlags);
			}
			else
			{
				*OutTargets[TargetIndex] = GraphBuilder.CreateUAV(TargetTexture, OutTargetFlags);
			}
		}
		else
		{
			*OutTargets[TargetIndex] = GetDummyUAV();
		}
	}

	Result.ComputeShadingOutputs = GraphBuilder.CreateUniformBuffer(ShadingOutputs);

	return Result;
}

void DispatchBasePass(
	FRDGBuilder& GraphBuilder,
	FNaniteShadingCommands& ShadingCommands,
	const FSceneRenderer& SceneRenderer,
	const FSceneTextures& SceneTextures,
	const FRenderTargetBindingSlots& BasePassRenderTargets,
	const FScene& Scene,
	const FViewInfo& View,
	const uint32 ViewIndex,
	const FRasterResults& RasterResults
)
{
	checkSlow(DoesPlatformSupportNanite(GMaxRHIShaderPlatform));

	LLM_SCOPE_BYTAG(Nanite);
	RDG_EVENT_SCOPE(GraphBuilder, "Nanite::BasePass");
	SCOPED_NAMED_EVENT(DispatchBasePass, FColor::Emerald);

	ShadingCommands.SetupTask.Wait();
	ShadingCommands.BuildCommandsTask.Wait();

	if (ShadingCommands.NumCommands == 0u)
	{
		return;
	}

	// Validate that all shading pipelines have up-to-date MaterialUB pointers.
	if (GNaniteValidateShadingBinUB && ShadingCommands.bBindless)
	{
		const ERHIFeatureLevel::Type FeatureLevel = Scene.GetFeatureLevel();
		for (const FNaniteShadingCommand& ShadingCommand : ShadingCommands.Commands)
		{
			if (ShadingCommand.PipelineGroup)
			{
				for (const auto& Pipeline : ShadingCommand.PipelineGroup->Pipelines)
				{
					FRHIUniformBuffer* CachedUB = Pipeline->MaterialUB;
					FRHIUniformBuffer* CurrentUB = Pipeline->MaterialProxy->UniformExpressionCache[FeatureLevel].UniformBuffer;
					ensureMsgf(CachedUB == CurrentUB,
						TEXT("Nanite shading pipeline has stale MaterialUB for material '%s'. Cached=%p, Current=%p. SerialNumber at capture: %d, current: %d"),
						*Pipeline->MaterialProxy->GetMaterialName(), CachedUB, CurrentUB,
						Pipeline->MaterialUBSerialNumber, Pipeline->MaterialProxy->GetExpressionCacheSerialNumber());
				}
			}
		}
	}

	FShaderBundleRHIRef ShaderBundle = ShadingCommands.ShaderBundle;

	const bool bDrawSceneViewsInOneNanitePass = ShouldDrawSceneViewsInOneNanitePass(View);
	FIntRect ViewRect = bDrawSceneViewsInOneNanitePass ? View.GetFamilyViewRect() : View.ViewRect;

	const int32 ViewWidth = ViewRect.Max.X - ViewRect.Min.X;
	const int32 ViewHeight = ViewRect.Max.Y - ViewRect.Min.Y;
	const FIntPoint ViewSize = FIntPoint(ViewWidth, ViewHeight);

	const FRDGSystemTextures& SystemTextures = FRDGSystemTextures::Get(GraphBuilder);

	FRDGTextureRef VisBuffer64 = RasterResults.VisBuffer64 ? RasterResults.VisBuffer64 : SystemTextures.Black;
	FRDGTextureRef DbgBuffer64 = RasterResults.DbgBuffer64 ? RasterResults.DbgBuffer64 : SystemTextures.Black;
	FRDGTextureRef DbgBuffer32 = RasterResults.DbgBuffer32 ? RasterResults.DbgBuffer32 : SystemTextures.Black;

	FRDGBufferRef VisibleClustersSWHW = RasterResults.VisibleClustersSWHW;
	FRDGBufferRef AssemblyTransforms = RasterResults.AssemblyTransforms;

	const uint32 IndirectArgsStride = sizeof(FUint32Vector4);

	FRDGBufferRef MultiViewIndices = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), 1), TEXT("Nanite.DummyMultiViewIndices"));
	FRDGBufferRef MultiViewRectScaleOffsets = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(FVector4f), 1), TEXT("Nanite.DummyMultiViewRectScaleOffsets"));
	FRDGBufferRef ViewsBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(FVector4f), 1), TEXT("Nanite.PackedViews"));

	AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(MultiViewIndices), 0);
	AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(MultiViewRectScaleOffsets), 0);
	AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(ViewsBuffer), 0);

	const FNaniteVisibilityQuery* VisibilityQuery = RasterResults.VisibilityQuery;
	const EShaderPlatform ShaderPlatform = Scene.GetShaderPlatform();
	const bool SubstrateAdaptativeGbufferEnabled = Substrate::IsSubstrateEnabled() && !Substrate::IsSubstrateBlendableGBufferEnabled(ShaderPlatform);

	TStaticArray<FTextureRenderTargetBinding, MaxSimultaneousRenderTargets> BasePassTextures;

	// NOTE: Always use a GBuffer layout with velocity output (It won't be written to unless the material has WPO or IsUsingBasePassVelocity())
	uint32 BasePassTextureCount = SceneTextures.GetGBufferRenderTargets(BasePassTextures, GBL_ForceVelocity);

	// We don't want to have Substrate MRTs appended to the list, except for the top layer data
	if (SubstrateAdaptativeGbufferEnabled && SceneRenderer.Scene)
	{
		// Add another MRT for Substrate top layer information. We want to follow the usual clear process which can leverage fast clear.
		{
			BasePassTextures[BasePassTextureCount] = FTextureRenderTargetBinding(SceneTextures.SubstrateSceneData->TopLayerTexture);
			BasePassTextureCount++;
		};
	}

	TArrayView<FTextureRenderTargetBinding> BasePassTexturesView = MakeArrayView(BasePassTextures.GetData(), BasePassTextureCount);

	// Render targets bindings should remain constant at this point.
	FRenderTargetBindingSlots BasePassBindings = GetRenderTargetBindings(ERenderTargetLoadAction::ELoad, BasePassTexturesView);
	BasePassBindings.DepthStencil = BasePassRenderTargets.DepthStencil;

	TArray<FRDGTextureRef, TInlineAllocator<MaxSimultaneousRenderTargets>> ClearTargetList;

	// Fast tile clear prior to fast clear eliminate
	const bool bFastTileClear = GNaniteFastTileClear != 0 && RHISupportsRenderTargetWriteMask(GMaxRHIShaderPlatform);
	if (bFastTileClear)
	{
		for (uint32 TargetIndex = 0; TargetIndex < MaxSimultaneousRenderTargets; ++TargetIndex)
		{
			if (FRDGTexture* TargetTexture = BasePassRenderTargets.Output[TargetIndex].GetTexture())
			{
				if (!EnumHasAnyFlags(TargetTexture->Desc.Flags, TexCreate_DisableDCC))
				{
					// Skip any targets that do not explicitly disable DCC, as this clear would not work correctly for DCC
					ClearTargetList.Add(nullptr);
					continue;
				}

				if (EnumHasAnyFlags(TargetTexture->Desc.Flags, TexCreate_NoFastClear))
				{
					// Skip any targets that explicitly disable fast clear optimization
					ClearTargetList.Add(nullptr);
					continue;
				}

				if ((ShadingCommands.BoundTargetMask & (1u << TargetIndex)) == 0u)
				{
					// Skip any targets that are not written by at least one shading command
					ClearTargetList.Add(nullptr);
					continue;
				}

				ClearTargetList.Add(TargetTexture);
			}
		}
	}

	FShadeBinning Binning = ShadeBinning(GraphBuilder, Scene, View, ViewRect, ShadingCommands, RasterResults, ClearTargetList);

	FNaniteShadingPassParameters* ShadingPassParameters = GraphBuilder.AllocParameters<FNaniteShadingPassParameters>();
	*ShadingPassParameters = CreateNaniteShadingPassParams(
		GraphBuilder,
		SceneRenderer,
		SceneTextures,
		View,
		ViewRect,
		RasterResults,
		RasterResults.ShadingMask,
		VisBuffer64,
		DbgBuffer64,
		DbgBuffer32,
		VisibleClustersSWHW,
		AssemblyTransforms,
		MultiViewIndices,
		MultiViewRectScaleOffsets,
		ViewsBuffer,
		BasePassBindings,
		ShadingCommands.BoundTargetMask,
		Binning,
		SubstrateAdaptativeGbufferEnabled
	);

	FShadingConfig ShadingConfig{};

	ShadingConfig.bHighPrecision	= UsingHighPrecisionGBuffer();
	ShadingConfig.bBundleShading	= ShaderBundle != nullptr && UseShadingShaderBundle(ShaderPlatform);
	ShadingConfig.bShowDrawEvents	= GShowMaterialDrawEvents != 0;
	ShadingConfig.BundleMethod		= GetShadingShaderBundleMethod(ShaderPlatform);
		
	static int32 PSOCollectorIndex = FPSOCollectorCreateManager::GetIndex(EShadingPath::Deferred, TEXT("NaniteShading"));

	const bool bParallelDispatch = GRHICommandList.UseParallelAlgorithms() && CVarParallelBasePassBuild.GetValueOnRenderThread() != 0 &&
								   FParallelMeshDrawCommandPass::IsOnDemandShaderCreationEnabled();
	if (bParallelDispatch)
	{
		GraphBuilder.AddDispatchPass(
			RDG_EVENT_NAME("ShadeGBufferCS"),
			ShadingPassParameters,
			ERDGPassFlags::Compute,
			[ShadingPassParameters, &ShadingCommands, ShadingConfig, ShaderBundle, IndirectArgsStride, VisibilityQuery, &View, ViewRect]
			(FRDGDispatchPassBuilder& DispatchPassBuilder)
		{
			TSharedPtr<FNaniteShadingPassIntermediates> Intermediates = CreateNaniteShadingPassIntermediates(ShadingPassParameters, ShadingCommands, VisibilityQuery, ViewRect);

			if (ShadingConfig.bBundleShading)
			{
				FRHICommandList* RHICmdListTask = DispatchPassBuilder.CreateCommandList();

				UE::Tasks::Launch(UE_SOURCE_LOCATION, [RHICmdListTask, Intermediates = MoveTemp(Intermediates), &ShadingCommands, ShaderBundle, ViewRect, ShadingConfig]
				{
					FTaskTagScope Scope(ETaskTag::EParallelRenderingThread);
					TRACE_CPUPROFILER_EVENT_SCOPE(RecordBundleShadingCommandsTask);
					DispatchComputeShaderBundle(*RHICmdListTask, ShadingCommands, ShadingConfig, ShaderBundle, *Intermediates, PSOCollectorIndex);
					RHICmdListTask->FinishRecording();
				});
			}
			else
			{
				// Distribute work evenly to the available task graph workers based on NumPassCommands.
				const int32 NumPassCommands = ShadingCommands.Commands.Num();
				const int32 NumThreads = FMath::Min<int32>(FTaskGraphInterface::Get().GetNumWorkerThreads(), CVarRHICmdWidth.GetValueOnRenderThread());
				const int32 NumTasks = FMath::Min<int32>(NumThreads, FMath::DivideAndRoundUp(NumPassCommands, CVarRHICmdMinDrawsPerParallelCmdList.GetValueOnRenderThread()));
				const int32 NumCommandsPerTask = FMath::DivideAndRoundUp(NumPassCommands, NumTasks);

				for (int32 TaskIndex = 0; TaskIndex < NumTasks; TaskIndex++)
				{
					const int32 StartIndex = TaskIndex * NumCommandsPerTask;
					const int32 NumCommands = FMath::Min(NumCommandsPerTask, NumPassCommands - StartIndex);
					checkSlow(NumCommands > 0);

					FRHICommandList* RHICmdListTask = DispatchPassBuilder.CreateCommandList();

					UE::Tasks::Launch(UE_SOURCE_LOCATION, [RHICmdListTask, &ShadingCommands, Intermediates = Intermediates, IndirectArgsStride, StartIndex, NumCommands, ShadingConfig]
					{
						FTaskTagScope Scope(ETaskTag::EParallelRenderingThread);
						TRACE_CPUPROFILER_EVENT_SCOPE(RecordShadingCommandsTask);

						for (int32 CommandIndex = 0; CommandIndex < NumCommands; ++CommandIndex)
						{
							FNaniteShadingCommand& ShadingCommand = ShadingCommands.Commands[StartIndex + CommandIndex];
							ShadingCommand.bVisible = IsCommandVisible(ShadingCommand, *Intermediates);

							if (ShadingCommand.bVisible && PrepareShadingCommand(ShadingCommand, PSOCollectorIndex))
							{
								FRHIBatchedShaderParameters& ShadingParameters = RHICmdListTask->GetScratchShaderParameters();

								RecordShadingParameters(
									ShadingParameters,
									ShadingCommand,
									ShadingConfig,
									ShadingCommands.MaxShadingBin + 1,
									ShadingCommands.MaxShadingGroup + 1,
									Intermediates->ViewRect,
									Intermediates->ShadingOutputs
								);

								RecordShadingCommand(
									*RHICmdListTask,
									Intermediates->IndirectArgsBuffer,
									IndirectArgsStride,
									ShadingConfig,
									ShadingParameters,
									ShadingCommand
								);
							}
						}

						RHICmdListTask->FinishRecording();
					});
				}
			}
		});
	}
	else
	{
		GraphBuilder.AddPass(
			RDG_EVENT_NAME("ShadeGBufferCS"),
			ShadingPassParameters,
			ERDGPassFlags::Compute,
			[ShadingPassParameters, &ShadingCommands, ShadingConfig, ShaderBundle, IndirectArgsStride, VisibilityQuery, &View, ViewRect]
			(FRDGAsyncTask, FRHIComputeCommandList& RHICmdList)
		{
			TSharedPtr<FNaniteShadingPassIntermediates> Intermediates = CreateNaniteShadingPassIntermediates(ShadingPassParameters, ShadingCommands, VisibilityQuery, ViewRect);

			if (ShadingConfig.bBundleShading)
			{
				TRACE_CPUPROFILER_EVENT_SCOPE(RecordBundleShadingCommands);
				DispatchComputeShaderBundle(RHICmdList, ShadingCommands, ShadingConfig, ShaderBundle, *Intermediates, PSOCollectorIndex, EParallelForFlags::ForceSingleThread);
			}
			else
			{
				TRACE_CPUPROFILER_EVENT_SCOPE(RecordShadingCommands);
				for (FNaniteShadingCommand& ShadingCommand : ShadingCommands.Commands)
				{
					ShadingCommand.bVisible = IsCommandVisible(ShadingCommand, *Intermediates);
					if (ShadingCommand.bVisible && PrepareShadingCommand(ShadingCommand, PSOCollectorIndex))
					{
						FRHIBatchedShaderParameters& ShadingParameters = RHICmdList.GetScratchShaderParameters();
						RecordShadingParameters(ShadingParameters, ShadingCommand, ShadingConfig, ShadingCommands.MaxShadingBin + 1, ShadingCommands.MaxShadingGroup + 1, Intermediates->ViewRect, Intermediates->ShadingOutputs);
						RecordShadingCommand(RHICmdList, Intermediates->IndirectArgsBuffer, IndirectArgsStride, ShadingConfig, ShadingParameters, ShadingCommand);
					}
				}
			}
		});
	}

	ExtractShadingDebug(GraphBuilder, View, Binning, ShadingCommands.MaxShadingBin + 1, ShadingCommands.MaxShadingGroup + 1);
}

FShadeBinning ShadeBinning(
	FRDGBuilder& GraphBuilder,
	const FScene& Scene,
	const FViewInfo& View,
	const FIntRect InViewRect,
	const FNaniteShadingCommands& ShadingCommands,
	const FRasterResults& RasterResults,
	const TConstArrayView<FRDGTextureRef> ClearTargets
)
{
	FShadeBinning Binning = {};

	LLM_SCOPE_BYTAG(Nanite);
	RDG_EVENT_SCOPE(GraphBuilder, "Nanite::ShadeBinning");

	if (!ShadingCommands.NumCommands)
	{
		return Binning;
	}

	const FSceneTexturesConfig& Config = View.GetSceneTexturesConfig();
	const EShaderPlatform ShaderPlatform = View.GetShaderPlatform();

	const bool bBindless = ShadingCommands.bBindless;

	const FNaniteShadingCommands::FMetaBufferArray& BinMetaData = ShadingCommands.BinMetaData;

	TArray<FRDGTextureRef, TInlineAllocator<MaxSimultaneousRenderTargets>> ValidClearTargets;

	uint32 ValidWriteMask = 0x0u;
	FIntPoint OuterRectSize = FIntPoint(MAX_int32, MAX_int32);		//TODO: Figure out how to actually get the correct rect here
	if (ClearTargets.Num() > 0)
	{
		for (int32 TargetIndex = 0; TargetIndex < ClearTargets.Num(); ++TargetIndex)
		{
			if (ClearTargets[TargetIndex] != nullptr)
			{
				// Compute a mask containing only set bits for MRT targets that are suitable for meta data optimization.
				ValidWriteMask |= (1u << uint32(TargetIndex));
				ValidClearTargets.Add(ClearTargets[TargetIndex]);

				const FIntVector Size = ClearTargets[TargetIndex]->Desc.GetSize();
				OuterRectSize.X = FMath::Min(OuterRectSize.X, Size.X);
				OuterRectSize.Y = FMath::Min(OuterRectSize.Y, Size.Y);
			}
		}
	}

	const uint32 ShadingBinCount = ShadingCommands.MaxShadingBin + 1u;
	const uint32 ShadingBinCountPow2 = FMath::RoundUpToPowerOfTwo(ShadingBinCount);

	const uint32 ShadingGroupCount = ShadingCommands.MaxShadingGroup + 1u;
	const uint32 ShadingGroupCountPow2 = FMath::RoundUpToPowerOfTwo(ShadingGroupCount);

	const bool bGatherStats = GNaniteShowStats != 0;

	const FUintVector4 ViewRect = FUintVector4(uint32(InViewRect.Min.X), uint32(InViewRect.Min.Y), uint32(InViewRect.Max.X), uint32(InViewRect.Max.Y));

	const uint32 PixelCount = InViewRect.Width() * InViewRect.Height();

	const FIntPoint GroupDim = GBinningTechnique == 0 ? FIntPoint(8u, 8u) : FIntPoint(32u, 32u);
	const FIntVector   BinDispatchDim = FComputeShaderUtils::GetGroupCount(ShadingBinCount, 64u);

	const FUint32Vector2 DispatchOffsetTL = FUint32Vector2(InViewRect.Min.X, InViewRect.Min.Y);

	const uint32 NumBytes_Meta = sizeof(FUintVector4) * ShadingGroupCount + sizeof(FNaniteShadingBinMeta) * ShadingBinCount;
	const uint32 NumBytes_Data = PixelCount * 8;

	FRDGBufferRef ShadingBinMeta = CreateStructuredBuffer(
		GraphBuilder,
		TEXT("Nanite.ShadingBinMeta"),
		sizeof(FUintVector4),
		NumBytes_Meta / sizeof(FUintVector4),
		BinMetaData.GetData(),
		BinMetaData.NumBytes()
	);

	Binning.ShadingBinData	= GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateByteAddressDesc(NumBytes_Meta + NumBytes_Data), TEXT("Nanite.ShadingBinData"));

	AddCopyBufferPass(GraphBuilder, Binning.ShadingBinData, 0, ShadingBinMeta, 0, NumBytes_Meta);

	const uint32 ShadingDispatchCountPow2 = bBindless ? ShadingGroupCountPow2 : ShadingBinCountPow2;

	Binning.ShadingDispatchArgs = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateRawIndirectDesc(sizeof(FUint32Vector4) * ShadingDispatchCountPow2), TEXT("Nanite.ShadingDispatchArgs"));
	Binning.ShadingBinStats  = bGatherStats ? GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(FNaniteShadingBinStats), 1u), TEXT("Nanite.ShadingBinStats")) : nullptr;

	FRDGBufferUAVRef ShadingDispatchArgsUAV = GraphBuilder.CreateUAV(FRDGBufferUAVDesc(Binning.ShadingDispatchArgs, PF_R32_UINT));
	FRDGBufferUAVRef ShadingBinDataUAV  = GraphBuilder.CreateUAV(Binning.ShadingBinData);
	FRDGBufferUAVRef ShadingBinStatsUAV = bGatherStats ? GraphBuilder.CreateUAV(Binning.ShadingBinStats) : nullptr;

	FRDGBufferRef ShadingBinScatterCountersBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(FNaniteShadingBinScatterCounters), ShadingBinCountPow2), TEXT("Nanite.ShadingBinScatterCounters"));
	FRDGBufferUAVRef ShadingBinScatterCountersUAV = GraphBuilder.CreateUAV(ShadingBinScatterCountersBuffer);

	FRDGBufferRef ShadingBinScatterRangesBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(FNaniteShadingBinScatterRanges), ShadingBinCountPow2), TEXT("Nanite.ShadingBinScatterRanges"));

	AddClearUAVPass(GraphBuilder, ShadingBinScatterCountersUAV, 0);
	if (bGatherStats)
	{
		AddClearUAVPass(GraphBuilder, ShadingBinStatsUAV, 0);
	}

	const bool bOptimizeWriteMask = (ValidClearTargets.Num() > 0);

	const uint32 ShadingRateTileSizeBits = GetShadingRateTileSizeBits();
	const bool bVariableRateShading = (ShadingRateTileSizeBits != 0);

	const uint32 TargetAlignment =	bOptimizeWriteMask ? 8 :	// 8x8 for optimized write mask
									bVariableRateShading ? 4 :	// 4x4 for VRS
									2;							// 2x2 for just quad processing

	const FUint32Vector2 AlignedDispatchOffsetTL = FUint32Vector2(AlignDown(InViewRect.Min.X, TargetAlignment), AlignDown(InViewRect.Min.Y, TargetAlignment));
	const FIntVector AlignedDispatchDim = FComputeShaderUtils::GetGroupCount(FIntPoint(InViewRect.Max.X - AlignedDispatchOffsetTL.X, InViewRect.Max.Y - AlignedDispatchOffsetTL.Y), GroupDim * 2);

	// Shading Bin Count
	{
		FShadingBinBuildCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FShadingBinBuildCS::FParameters>();
		PassParameters->ViewRect = ViewRect;
		PassParameters->ValidWriteMask = ValidWriteMask;
		PassParameters->DispatchOffsetTL = bOptimizeWriteMask ? AlignedDispatchOffsetTL : DispatchOffsetTL;
		PassParameters->ShadingBinCount = ShadingBinCount;
		PassParameters->ShadingGroupCount = ShadingGroupCount;
		PassParameters->ShadingRateTileSizeBits = GetShadingRateTileSizeBits();
		PassParameters->DummyZero = 0;
		PassParameters->ShadingRateImage = GetShadingRateImage(GraphBuilder, View);
		PassParameters->ShadingMaskSampler = TStaticSamplerState<SF_Point>::GetRHI();
		PassParameters->ShadingMask = RasterResults.ShadingMask;
		PassParameters->OutShadingBinData = ShadingBinDataUAV;
		PassParameters->OutShadingDispatchArgs = ShadingDispatchArgsUAV;
		PassParameters->OutShadingBinScatterCounters = ShadingBinScatterCountersUAV;

		FShadingBinBuildCS::FPermutationDomain PermutationVector;
		PermutationVector.Set<FShadingBinBuildCS::FBuildPassDim>(NANITE_SHADING_BIN_COUNT);
		PermutationVector.Set<FShadingBinBuildCS::FTechniqueDim>(FMath::Clamp<int32>(GBinningTechnique, 0, 1));
		PermutationVector.Set<FShadingBinBuildCS::FGatherStatsDim>(bGatherStats);
		PermutationVector.Set<FShadingBinBuildCS::FVariableRateDim>(bVariableRateShading);
		PermutationVector.Set<FShadingBinBuildCS::FOptimizeWriteMaskDim>(bOptimizeWriteMask);
		PermutationVector.Set<FShadingBinBuildCS::FNumExportsDim>(FMath::Max(1, ValidClearTargets.Num()));
		PermutationVector.Set<FShadingBinBuildCS::FBindlessDim>(bBindless);
		auto ComputeShader = View.ShaderMap->GetShader<FShadingBinBuildCS>(PermutationVector);

		if (bOptimizeWriteMask)
		{
			for (int32 TargetIndex = 0; TargetIndex < ValidClearTargets.Num(); ++TargetIndex)
			{
				PassParameters->OutCMaskBuffer[TargetIndex] = GraphBuilder.CreateUAV(FRDGTextureUAVDesc::CreateForMetaData(ValidClearTargets[TargetIndex], ERDGTextureMetaDataAccess::CMask));
			}

			const bool bWriteSubTiles = GNaniteFastTileClearSubTiles != 0u;

			FClearCMaskRectCS::FPermutationDomain CMaskPermutationVector;
			CMaskPermutationVector.Set<FClearCMaskRectCS::FNumExportsDim>(FMath::Max(1, ValidClearTargets.Num()));
			auto CMaskClearComputeShader = View.ShaderMap->GetShader<FClearCMaskRectCS>(CMaskPermutationVector);

			const FIntPoint InnerTileEnd = FIntPoint(FMath::DivideAndRoundUp(InViewRect.Max.X, 8), FMath::DivideAndRoundUp(InViewRect.Max.Y, 8));
			const FIntPoint OuterTileEnd = FIntPoint(FMath::DivideAndRoundDown(OuterRectSize.X, 8), FMath::DivideAndRoundDown(OuterRectSize.Y, 8));


			// TODO: The dynres clear optimization currently only works for a single view that starts at (0,0). Make this best more general and robust.
			// 
			//       Longer term, it seems the more optimal path would be explicitly clearing the few tiles that need clearing instead of relying on FCE,
			//       which still has a cost even when no work needs to be done, but that requires some tighter cooperation with the RHI.
			const bool bFastDynResClear =	GNaniteFastTileClearDynResOpt != 0 && View.IsFirstInFamily() && View.IsLastInFamily() && 
											InViewRect.Min.X == 0 && InViewRect.Min.Y == 0 && 
											(View.UnscaledViewRect.Width() > View.ViewRect.Width() || View.UnscaledViewRect.Height() > View.ViewRect.Height()) &&
											(OuterTileEnd.X > InnerTileEnd.X || OuterTileEnd.Y > InnerTileEnd.Y);

			GraphBuilder.AddPass(
				RDG_EVENT_NAME("ShadingCount"),
				PassParameters,
				ERDGPassFlags::Compute,
				[AlignedDispatchDim, ComputeShader, PassParameters, TargetCount = ValidClearTargets.Num(), bWriteSubTiles, bFastDynResClear, CMaskClearComputeShader, InnerTileEnd, OuterTileEnd](FRDGAsyncTask, FRHIComputeCommandList& RHICmdList)
				{
					void* PlatformDataPtr = nullptr;
					uint32 PlatformDataSize = 0;

					// Note: Assumes all targets match in resolution (which they should)
					if (PassParameters->OutCMaskBuffer[0] != nullptr)
					{
						FRHITexture* TargetTextureRHI = PassParameters->OutCMaskBuffer[0]->GetParentRHI();

						// Retrieve the platform specific data that the decode shader needs.
						TargetTextureRHI->GetWriteMaskProperties(PlatformDataPtr, PlatformDataSize);
						check(PlatformDataSize > 0);

						if (PlatformDataPtr == nullptr)
						{
							// If the returned pointer was null, the platform RHI wants us to allocate the memory instead.
							PlatformDataPtr = alloca(PlatformDataSize);
							TargetTextureRHI->GetWriteMaskProperties(PlatformDataPtr, PlatformDataSize);
						}
					}

					check(PlatformDataPtr != nullptr && PlatformDataSize > 0);

					bool bSubTileMatch = bWriteSubTiles;

					// If we want to write 4x4 subtiles, ensure platform specific data matches across all MRTs (tile modes, etc..)
					if (bWriteSubTiles)
					{
						TArray<uint8, TInlineAllocator<8>> Scratch;

						for (int32 TargetIndex = 1; TargetIndex < TargetCount; ++TargetIndex)
						{
							void* TestPlatformDataPtr = nullptr;
							uint32 TestPlatformDataSize = 0;

							// We want to enforce that the platform metadata is bit exact across all MRTs
							if (PassParameters->OutCMaskBuffer[TargetIndex] != nullptr)
							{
								FRHITexture* TargetTextureRHI = PassParameters->OutCMaskBuffer[TargetIndex]->GetParentRHI();

								TargetTextureRHI->GetWriteMaskProperties(TestPlatformDataPtr, TestPlatformDataSize);
								check(TestPlatformDataSize > 0);

								if (TestPlatformDataPtr == nullptr)
								{
									// If the returned pointer was null, the platform RHI wants us to allocate the memory instead.
									Scratch.SetNumZeroed(TestPlatformDataSize);
									TestPlatformDataPtr = Scratch.GetData();
									TargetTextureRHI->GetWriteMaskProperties(TestPlatformDataPtr, TestPlatformDataSize);
								}

								check(TestPlatformDataPtr != nullptr && TestPlatformDataSize == PlatformDataSize);

								if (FMemory::Memcmp(PlatformDataPtr, TestPlatformDataPtr, PlatformDataSize) != 0)
								{
									bSubTileMatch = false;
									break;
								}
							}
						}
					}

					if (bFastDynResClear)
					{
						FClearCMaskRectCS::FParameters Parameters = FClearCMaskRectCS::FParameters();
						for (int32 i = 0; i < TargetCount; i++)
						{
							Parameters.OutCMaskBuffer[i] = PassParameters->OutCMaskBuffer[i];
						}

						SetComputePipelineState(RHICmdList, CMaskClearComputeShader.GetComputeShader());

						// Split clear into a right rect and a bottom rect
						const FIntPoint RightRectMin = FIntPoint(InnerTileEnd.X, 0);
						const FIntPoint RightRectSize = FIntPoint(OuterTileEnd.X - InnerTileEnd.X, InnerTileEnd.Y);

						const FIntPoint BottomRectMin = FIntPoint(0, InnerTileEnd.Y);
						const FIntPoint BottomRectSize = FIntPoint(OuterTileEnd.X, OuterTileEnd.Y - InnerTileEnd.Y);

						if (RightRectSize.X > 0 && RightRectSize.Y > 0)
						{
							// Right rect
							const FIntVector RightRectDispatchDim = FComputeShaderUtils::GetGroupCount(RightRectSize, 8);

							Parameters.ClearTileRectMin = RightRectMin;
							Parameters.ClearTileRectSize = RightRectSize;
							
							SetShaderParametersMixedCS(RHICmdList, CMaskClearComputeShader, Parameters, PlatformDataPtr, PlatformDataSize);
							RHICmdList.DispatchComputeShader(RightRectDispatchDim.X, RightRectDispatchDim.Y, RightRectDispatchDim.Z);
						}

						if (BottomRectSize.X > 0 &&  BottomRectSize.Y > 0)
						{
							// Bottom rect
							const FIntVector BottomRectDispatchDim = FComputeShaderUtils::GetGroupCount(BottomRectSize, 8);

							Parameters.ClearTileRectMin = BottomRectMin;
							Parameters.ClearTileRectSize = BottomRectSize;

							SetShaderParametersMixedCS(RHICmdList, CMaskClearComputeShader, Parameters, PlatformDataPtr, PlatformDataSize);
							RHICmdList.DispatchComputeShader(BottomRectDispatchDim.X, BottomRectDispatchDim.Y, BottomRectDispatchDim.Z);
						}
					}

					SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
					SetShaderParametersMixedCS(RHICmdList, ComputeShader, *PassParameters, PlatformDataPtr, PlatformDataSize, bSubTileMatch);

					RHICmdList.DispatchComputeShader(AlignedDispatchDim.X, AlignedDispatchDim.Y, AlignedDispatchDim.Z);
				}
			);
		}
		else
		{
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("ShadingCount"), ComputeShader, PassParameters, AlignedDispatchDim);
		}
	}

	FRDGBufferRef ShadingGroupMetaBuffer = nullptr;
	if(bBindless)
	{	
		const FIntVector   GroupDispatchDim = FComputeShaderUtils::GetGroupCount(ShadingGroupCount, 64u);

		// Worst case number of thread groups
		// 8x4 pixel groups with every quad getting emitted 4 times. Each bin gets one extra group for potential padding.
		const uint32 MaxThreadGroups = FMath::DivideAndRoundUp(InViewRect.Width(), 8) * FMath::DivideAndRoundUp(InViewRect.Height(), 4) * 4 + ShadingBinCount;

		ShadingGroupMetaBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(FNaniteShadingGroupMeta), ShadingGroupCountPow2), TEXT("Nanite.ShadingGroupMeta"));
		AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(ShadingGroupMetaBuffer), 0);

		FRDGBufferRef ShadingGroupAllocatorBuffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(FNaniteShadingGroupAllocator), 1), TEXT("Nanite.ShadingGroupAllocator"));
		AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(ShadingGroupAllocatorBuffer), 0);

		Binning.ThreadGroupData = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), MaxThreadGroups), TEXT("Nanite.ThreadGroupData"));

		// Shading Group Allocate
		{
			FShadingGroupCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FShadingGroupCS::FParameters>();
			PassParameters->ShadingBinCount = ShadingBinCount;
			PassParameters->ShadingGroupCount = ShadingGroupCount;
			PassParameters->OutShadingBinData = ShadingBinDataUAV;
			PassParameters->OutShadingBinScatterCounters = ShadingBinScatterCountersUAV;
			PassParameters->OutShadingGroupMeta = GraphBuilder.CreateUAV(ShadingGroupMetaBuffer);

			FShadingGroupCS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FShadingGroupCS::FBinPassDim>(NANITE_SHADING_BIN_GROUP_COUNT);
			auto ComputeShader = View.ShaderMap->GetShader<FShadingGroupCS>(PermutationVector);

			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("ShadingGroupCount"), ComputeShader, PassParameters, BinDispatchDim);
		}

		// Shading Group Reserve
		{
			FShadingGroupCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FShadingGroupCS::FParameters>();
			PassParameters->ShadingGroupCount = ShadingGroupCount;
			PassParameters->OutShadingBinData = ShadingBinDataUAV;
			PassParameters->OutShadingGroupMeta = GraphBuilder.CreateUAV(ShadingGroupMetaBuffer);
			PassParameters->OutShadingGroupAllocator = GraphBuilder.CreateUAV(ShadingGroupAllocatorBuffer);
			PassParameters->OutShadingDispatchArgs = ShadingDispatchArgsUAV;

			FShadingGroupCS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FShadingGroupCS::FBinPassDim>(NANITE_SHADING_BIN_GROUP_RESERVE);
			auto ComputeShader = View.ShaderMap->GetShader<FShadingGroupCS>(PermutationVector);

			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("ShadingGroupReserve"), ComputeShader, PassParameters, GroupDispatchDim);
		}
	}

	// Shading Bin Reserve
	{
		FRDGBufferRef ShadingBinAllocator = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), 1), TEXT("Nanite.ShadingBinAllocator"));
		FRDGBufferUAVRef ShadingBinAllocatorUAV = GraphBuilder.CreateUAV(FRDGBufferUAVDesc(ShadingBinAllocator, PF_R32_UINT));
		AddClearUAVPass(GraphBuilder, ShadingBinAllocatorUAV, 0);

		FShadingBinReserveCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FShadingBinReserveCS::FParameters>();
		PassParameters->ShadingBinCount = ShadingBinCount;
		PassParameters->ShadingGroupCount = ShadingGroupCount;
		PassParameters->OutShadingBinStats = ShadingBinStatsUAV;
		PassParameters->OutShadingBinData = ShadingBinDataUAV;
		PassParameters->OutShadingBinAllocator = ShadingBinAllocatorUAV;
		PassParameters->OutShadingDispatchArgs = ShadingDispatchArgsUAV;
		PassParameters->OutShadingBinStats = ShadingBinStatsUAV;
		PassParameters->OutShadingBinScatterCounters = ShadingBinScatterCountersUAV;
		PassParameters->OutShadingBinScatterRanges = GraphBuilder.CreateUAV(ShadingBinScatterRangesBuffer);
		if (ShadingGroupMetaBuffer)
		{
			PassParameters->OutShadingGroupMeta = GraphBuilder.CreateUAV(ShadingGroupMetaBuffer);
		}

		FShadingBinReserveCS::FPermutationDomain PermutationVector;
		PermutationVector.Set<FShadingBinReserveCS::FGatherStatsDim>(bGatherStats);
		PermutationVector.Set<FShadingBinReserveCS::FBindlessDim>(bBindless);
		auto ComputeShader = View.ShaderMap->GetShader<FShadingBinReserveCS>(PermutationVector);

		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("ShadingReserve"), ComputeShader, PassParameters, BinDispatchDim);
	}

	// Shading Bin Scatter
	{
		FShadingBinBuildCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FShadingBinBuildCS::FParameters>();
		PassParameters->ViewRect = ViewRect;
		PassParameters->DispatchOffsetTL = AlignedDispatchOffsetTL;
		PassParameters->ShadingBinCount = ShadingBinCount;
		PassParameters->ShadingGroupCount = ShadingGroupCount;
		PassParameters->ShadingRateTileSizeBits = GetShadingRateTileSizeBits();
		PassParameters->DummyZero = 0;
		PassParameters->ShadingRateImage = GetShadingRateImage(GraphBuilder, View);
		PassParameters->ShadingMaskSampler = TStaticSamplerState<SF_Point>::GetRHI();
		PassParameters->ShadingMask = RasterResults.ShadingMask;
		PassParameters->ShadingBinScatterRanges = GraphBuilder.CreateSRV(ShadingBinScatterRangesBuffer);
		PassParameters->OutShadingBinStats = ShadingBinStatsUAV;
		PassParameters->OutShadingBinData = ShadingBinDataUAV;
		PassParameters->OutShadingBinScatterCounters = ShadingBinScatterCountersUAV;
		if(Binning.ThreadGroupData)
		{
			PassParameters->OutThreadGroupData = GraphBuilder.CreateUAV(Binning.ThreadGroupData);
		}

		FShadingBinBuildCS::FPermutationDomain PermutationVector;
		PermutationVector.Set<FShadingBinBuildCS::FBuildPassDim>(NANITE_SHADING_BIN_SCATTER);
		PermutationVector.Set<FShadingBinBuildCS::FTechniqueDim>(FMath::Clamp<int32>(GBinningTechnique, 0, 1));
		PermutationVector.Set<FShadingBinBuildCS::FGatherStatsDim>(bGatherStats);
		PermutationVector.Set<FShadingBinBuildCS::FVariableRateDim>(bVariableRateShading);
		PermutationVector.Set<FShadingBinBuildCS::FOptimizeWriteMaskDim>(false);
		PermutationVector.Set<FShadingBinBuildCS::FNumExportsDim>(1);
		PermutationVector.Set<FShadingBinBuildCS::FBindlessDim>(bBindless);
		auto ComputeShader = View.ShaderMap->GetShader<FShadingBinBuildCS>(PermutationVector);

		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("ShadingScatter"), ComputeShader, PassParameters, AlignedDispatchDim);
	}

	// Shading Bin Validate
	if (GNaniteValidateShadeBinning)
	{
		FShadingBinValidateCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FShadingBinValidateCS::FParameters>();
		PassParameters->ShadingBinCount = ShadingBinCount;
		PassParameters->ShadingGroupCount = ShadingGroupCount;
		PassParameters->OutShadingBinData = ShadingBinDataUAV;
		PassParameters->OutShadingBinScatterCounters = ShadingBinScatterCountersUAV;

		auto ComputeShader = View.ShaderMap->GetShader<FShadingBinValidateCS>();
		FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("ShadingValidate"), ERDGPassFlags::Compute | ERDGPassFlags::NeverCull, ComputeShader, PassParameters, BinDispatchDim);
	}

	const FNaniteVisualizationData& VisualizationData = GetNaniteVisualizationData();
	if (bOptimizeWriteMask && VisualizationData.IsActive())
	{
		auto ComputeShader = View.ShaderMap->GetShader<FVisualizeClearTilesCS>();

		FRDGTextureDesc VisClearMaskDesc = FRDGTextureDesc::Create2D(
			FIntPoint(InViewRect.Width(), InViewRect.Height()),
			PF_R32_UINT,
			FClearValueBinding::Transparent,
			TexCreate_ShaderResource | TexCreate_UAV
		);

		Binning.FastClearVisualize = GraphBuilder.CreateTexture(VisClearMaskDesc, TEXT("Nanite.VisClearMask"));
		AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(Binning.FastClearVisualize), FUintVector4(ForceInitToZero));

		for (int32 TargetIndex = 0; TargetIndex < ValidClearTargets.Num(); ++TargetIndex)
		{
			if (TargetIndex != GNaniteFastTileVis && GNaniteFastTileVis != INDEX_NONE)
			{
				continue;
			}

			FVisualizeClearTilesCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FVisualizeClearTilesCS::FParameters>();
			PassParameters->ViewRect		= ViewRect;
			PassParameters->OutCMaskBuffer	= GraphBuilder.CreateUAV(FRDGTextureUAVDesc::CreateForMetaData(ValidClearTargets[TargetIndex], ERDGTextureMetaDataAccess::CMask));
			PassParameters->OutVisualized	= GraphBuilder.CreateUAV(Binning.FastClearVisualize);

			GraphBuilder.AddPass(
				RDG_EVENT_NAME("VisualizeFastClear"),
				PassParameters,
				ERDGPassFlags::Compute,
				[InViewRect, ComputeShader, PassParameters](FRDGAsyncTask, FRHIComputeCommandList& RHICmdList)
				{
					void* PlatformDataPtr = nullptr;
					uint32 PlatformDataSize = 0;

					if (PassParameters->OutCMaskBuffer != nullptr)
					{
						FRHITexture* TargetTextureRHI = PassParameters->OutCMaskBuffer->GetParentRHI();

						// Retrieve the platform specific data that the decode shader needs.
						TargetTextureRHI->GetWriteMaskProperties(PlatformDataPtr, PlatformDataSize);
						check(PlatformDataSize > 0);

						if (PlatformDataPtr == nullptr)
						{
							// If the returned pointer was null, the platform RHI wants us to allocate the memory instead.
							PlatformDataPtr = alloca(PlatformDataSize);
							TargetTextureRHI->GetWriteMaskProperties(PlatformDataPtr, PlatformDataSize);
						}
					}

					SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
					SetShaderParametersMixedCS(RHICmdList, ComputeShader, *PassParameters, PlatformDataPtr, PlatformDataSize);

					const FIntVector DispatchDim = FComputeShaderUtils::GetGroupCount(FIntPoint(InViewRect.Width(), InViewRect.Height()), FIntPoint(8u, 8u));
					RHICmdList.DispatchComputeShader(DispatchDim.X, DispatchDim.Y, DispatchDim.Z);
				}
			);
		}
	}

	return Binning;
}

void CollectBasePassShadingPSOInitializers(
	const FSceneTexturesConfig& SceneTexturesConfig,
	const FPSOPrecacheVertexFactoryData& VertexFactoryData,
	const FMaterial& Material,
	const FPSOPrecacheParams& PreCacheParams,
	ERHIFeatureLevel::Type FeatureLevel,
	EShaderPlatform ShaderPlatform,
	int32 PSOCollectorIndex,
	TArray<FPSOPrecacheData>& PSOInitializers)
{
	TArray<ELightMapPolicyType, TInlineAllocator<2>> UniformLightMapPolicyTypes = FBasePassMeshProcessor::GetUniformLightMapPolicyTypeForPSOCollection(FeatureLevel, Material);

	auto CollectBasePass = [&]()
	{
		for (uint32 Curve = 0; Curve < 2; Curve++)
		for (uint32 Voxel = 0; Voxel < 2; Voxel++)
		{
			if (Curve == 1 && Voxel == 1)
			{
				continue;
			}
			const bool bVoxel = (Voxel == 1);
			const bool bCurve = (Curve == 1);
			for (ELightMapPolicyType UniformLightMapPolicyType : UniformLightMapPolicyTypes)
			{
				TShaderRef<TBasePassComputeShaderPolicyParamType<FUniformLightMapPolicy>> BasePassComputeShader;

				bool bShadersValid = GetBasePassShader<FUniformLightMapPolicy>(
					Material,
					VertexFactoryData.VertexFactoryType,
					FUniformLightMapPolicy(UniformLightMapPolicyType),
					FeatureLevel,
					bVoxel,
					bCurve,
					false, // bIsDebug
					SF_Compute,
					&BasePassComputeShader
				);

				if (!bShadersValid)
				{
					continue;
				}

				FPSOPrecacheData ComputePSOPrecacheData;
				ComputePSOPrecacheData.Type = FPSOPrecacheData::EType::Compute;
				if (ComputePSOPrecacheData.SetComputeShader(BasePassComputeShader))
				{
#if PSO_PRECACHING_VALIDATE
					ComputePSOPrecacheData.ResourceName = Material.GetAssetName();
					ComputePSOPrecacheData.PSOCollectorIndex = PSOCollectorIndex;
					ComputePSOPrecacheData.VertexFactoryType = VertexFactoryData.VertexFactoryType;
					if (PSOCollectorStats::IsFullPrecachingValidationEnabled())
					{
						ComputePSOPrecacheData.bDefaultMaterial = Material.IsDefaultMaterial();
						ConditionalBreakOnPSOPrecacheShader(ComputePSOPrecacheData.ComputeInitializer);
					}
#endif // PSO_PRECACHING_VALIDATE
					PSOInitializers.Add(MoveTemp(ComputePSOPrecacheData));
				}
			}
		}
	};

	CollectBasePass();
}

} // Nanite

FNaniteRasterPipeline FNaniteRasterPipeline::GetFixedFunctionPipeline(uint8 BinMask)
{
	FNaniteRasterPipeline Pipeline;
	Pipeline.RasterMaterial = UMaterial::GetDefaultMaterial(MD_Surface)->GetRenderProxy();
	Pipeline.bIsTwoSided = (BinMask & NANITE_FIXED_FUNCTION_BIN_TWOSIDED) != 0;
	Pipeline.bVoxel = (BinMask & NANITE_FIXED_FUNCTION_BIN_VOXEL) != 0;
	Pipeline.bCurve = (BinMask & NANITE_FIXED_FUNCTION_BIN_CURVE) != 0;
	Pipeline.bSplineMesh = (BinMask & NANITE_FIXED_FUNCTION_BIN_SPLINE) != 0;
	Pipeline.bSkinnedMesh = (BinMask & NANITE_FIXED_FUNCTION_BIN_SKINNED) != 0;
	Pipeline.bCastShadow = (BinMask & NANITE_FIXED_FUNCTION_BIN_CAST_SHADOW) != 0;
	Pipeline.bTranslucent = false;
	return Pipeline;
}

uint32 FNaniteRasterPipeline::GetPipelineHash() const
{
	struct FHashKey
	{
		uint32 MaterialFlags;
		uint32 MaterialHash;

		FDisplacementScaling DisplacementScaling;
		FDisplacementFadeRange DisplacementFadeRange;

		static inline uint32 PointerHash(const void* Key)
		{
			// Ignoring the lower 4 bits since they are likely zero anyway.
			// Higher bits are more significant in 64 bit builds.
			return reinterpret_cast<UPTRINT>(Key) >> 4;
		};

	} HashKey;
	
	FMemory::Memzero(HashKey);

	HashKey.MaterialFlags  = 0;
	HashKey.MaterialFlags |= bIsTwoSided				? 0x1u : 0x0u;
	HashKey.MaterialFlags |= bWPOEnabled				? 0x2u : 0x0u;
	HashKey.MaterialFlags |= bDisplacementEnabled		? 0x4u : 0x0u;
	HashKey.MaterialFlags |= bPerPixelEval				? 0x8u : 0x0u;
	HashKey.MaterialFlags |= bSplineMesh				? 0x10u : 0x0u;
	HashKey.MaterialFlags |= bSkinnedMesh				? 0x20u : 0x0u;
	HashKey.MaterialFlags |= bCastShadow				? 0x40u : 0x0u;
	HashKey.MaterialFlags |= bIsDisplacementFallback	? 0x80u : 0x0u;
	HashKey.MaterialFlags |= bVertexUVs					? 0x100u : 0x0u;
	HashKey.MaterialFlags |= bVoxel						? 0x200u : 0x0u;
	HashKey.MaterialFlags |= bFirstPersonLerp			? 0x400u : 0x0u;
	HashKey.MaterialFlags |= bCurve						? 0x800u : 0x0u;
	HashKey.MaterialFlags |= bTranslucent				? 0x1000u : 0x0u;
	HashKey.MaterialHash   = FHashKey::PointerHash(RasterMaterial);

	if (bDisplacementEnabled || bIsDisplacementFallback)
	{
		HashKey.DisplacementScaling = DisplacementScaling;
	}
	if (bDisplacementEnabled && bHasDisplacementFadeOut)
	{
		HashKey.DisplacementFadeRange = DisplacementFadeRange;
	}

	if (bTranslucent)
	{
		// GW-TODO: Hash TranslucencyPassData!
	}

	const uint64 PipelineHash = CityHash64((char*)&HashKey, sizeof(FHashKey));
	return HashCombineFast(uint32(PipelineHash & 0xFFFFFFFF), uint32((PipelineHash >> 32) & 0xFFFFFFFF));
}

uint32 FNaniteRasterPipeline::GetFixedFunctionBinMask() const
{
	// Get a mask of the required fixed function features for this pipeline to fall back to a fixed function bin.
	return (bIsTwoSided  ? NANITE_FIXED_FUNCTION_BIN_TWOSIDED    : 0) |
		(bSplineMesh  ? NANITE_FIXED_FUNCTION_BIN_SPLINE      : 0) |
		(bSkinnedMesh ? NANITE_FIXED_FUNCTION_BIN_SKINNED     : 0) |
		(bCastShadow  ? NANITE_FIXED_FUNCTION_BIN_CAST_SHADOW : 0) |
		(bVoxel       ? NANITE_FIXED_FUNCTION_BIN_VOXEL       : 0) |
		(bCurve       ? NANITE_FIXED_FUNCTION_BIN_CURVE       : 0);
}

bool FNaniteRasterPipeline::IsPSOPrecaching(ERHIFeatureLevel::Type FeatureLevel, FGraphEventArray& OutAsyncPSOCompileEvents) const
{	
	if (!PipelineStateCache::IsPSOPrecachingEnabled())
	{
		return false;
	}
	
	// Collect the PSO precache data to make sure the raster material is only used when it's not compiling anymore (otherwise fallback to fixed function)		
	TArray<FPSOPrecacheData> PSOPrecacheData;
	Nanite::CollectRasterPSOInitializersForRasterPipeline(*this, FeatureLevel, PSOPrecacheData);
	
	const uint32 FixedBinMask = GetFixedFunctionBinMask();

	// Check all the precache requests and collect all the async compile events for active compiling PSOs (and boost priority)
	bool bIsPSOPrecaching = false;
	for (const FPSOPrecacheData& Entry : PSOPrecacheData)
	{
		EPSOPrecacheResult PSOPrecacheResult = EPSOPrecacheResult::Unknown;
		FGraphEventRef CompletionEvent;		
		FPSOPrecacheRequestID PSOPrecacheRequestID;
		if (Entry.Type == FPSOPrecacheData::EType::Graphics)
		{
			PSOPrecacheResult = PipelineStateCache::CheckPipelineStateInCache(Entry.GraphicsPSOInitializer, CompletionEvent, PSOPrecacheRequestID);
		}
		else
		{
			PSOPrecacheResult = PipelineStateCache::CheckPipelineStateInCache(Entry.ComputeInitializer.ComputeShader, CompletionEvent, PSOPrecacheRequestID);
		}

		if (PSOPrecacheResult == EPSOPrecacheResult::Active)
		{
			check(CompletionEvent.IsValid());
			OutAsyncPSOCompileEvents.Add(CompletionEvent);
			PipelineStateCache::BoostPrecachePriority(EPSOPrecachePriority::High, PSOPrecacheRequestID);
			bIsPSOPrecaching = true;
		}
	}

	return bIsPSOPrecaching;
}

bool FNaniteRasterPipeline::GetFallbackPipeline(FNaniteRasterPipeline& OutFallback) const
{
	const uint32 FixedBinMask = GetFixedFunctionBinMask();

	// NOTE: Ordering matters here. We don't want to have to create many bins to handle enabled/disabled state of
	// pixel programmable, WPO, and displacement, so when we have overlap, WPO disabled clusters rely on branching
	// rather than using simpler shaders until either pixel programmable distance or displacement fade-out occurs,
	// and when either pixel programmable or displacement is disabled, both are.
	const bool bNeedsVertexProgrammableFallback = bWPOEnabled && bHasWPODistance;
	const bool bNeedsPixelProgrammableFallback = bPerPixelEval && bHasPixelDistance;
	const bool bNeedsDisplacementFallback = bDisplacementEnabled && (bHasDisplacementFadeOut || bRequiresDisplacementFallback);
	if (bNeedsPixelProgrammableFallback || bNeedsDisplacementFallback)
	{
		if (bWPOEnabled)
		{
			// The fallback bin must still be a programmable bin, but with pixel programmable and displacement disabled
			OutFallback = *this;
			OutFallback.bHasWPODistance = false;
			OutFallback.bHasPixelDistance = false;
			OutFallback.bHasDisplacementFadeOut = false;
			OutFallback.bRequiresDisplacementFallback = false;
			OutFallback.bPerPixelEval = false;
			OutFallback.bDisplacementEnabled = false;
			OutFallback.bVertexUVs = false;
		}
		else
		{
			// The fallback bin can be a non-programmable, fixed-function bin
			OutFallback = GetFixedFunctionPipeline(FixedBinMask);
		}

		if (bDisplacementEnabled)
		{
			// NOTE: We do something special for displacement fallback bins. The displacement scaling still has to be unique
			// per bin, so it can't strictly be a "fixed function bin", though it does use default material permutations if
			// the fallback does not have WPO (and is therefore not itself programmable in any way).
			OutFallback.bIsDisplacementFallback = true;
			OutFallback.DisplacementScaling = DisplacementScaling;
			OutFallback.DisplacementFadeRange = FDisplacementFadeRange::Invalid();
		}

		return true;
	}
	else if (bNeedsVertexProgrammableFallback)
	{
		if (bPerPixelEval || bDisplacementEnabled)
		{
			// The fallback bin must still be a programmable bin, but with WPO force disabled.
			OutFallback = *this;
			OutFallback.bHasWPODistance = false;
			OutFallback.bWPOEnabled = false;
		}
		else
		{
			// The fallback bin can be a non-programmable, fixed-function bin
			OutFallback = GetFixedFunctionPipeline(FixedBinMask);
		}

		if (bDisplacementEnabled)
		{
			// Make sure the fallback bin preserves the displacement scaling
			OutFallback.DisplacementScaling = DisplacementScaling;
			OutFallback.DisplacementFadeRange = FDisplacementFadeRange::Invalid();
		}

		return true;
	}

	return false;
}

FNaniteRasterPipelines::FNaniteRasterPipelines()
{
	PipelineBins.Reserve(256);
	PerPixelEvalPipelineBins.Reserve(256);
	PipelineMap.Reserve(256);

	AllocateFixedFunctionBins();
}

FNaniteRasterPipelines::~FNaniteRasterPipelines()
{
	ReleaseFixedFunctionBins();

	PipelineBins.Reset();
	PerPixelEvalPipelineBins.Reset();
	PipelineMap.Empty();
}

void FNaniteRasterPipelines::AllocateFixedFunctionBins()
{
	check(FixedFunctionBins.Num() == 0);

	// Note: Invalid mutually exclusive permutation: NANITE_FIXED_FUNCTION_BIN_SKINNED | NANITE_FIXED_FUNCTION_BIN_SPLINE
	// We let the registration succeed because permutations are not actually fetched for the fixed function material here.
	// When caching the raster passes we remap skinned | spline => skinned permutation and also skip launching these bins.

	// When Curve is activated and Voxel is not activated, we allocate more bins than necesssary as the bins are allocated 
	// contiguously, while Voxel's bins are inbetween triangles' & curves' bins.
	//   Curve bin
	//   v 
	// 0010 1111 1111
	//    ^
	//    Voxel bin
	for (uint32 BinMask = 0; BinMask <= Nanite::FGlobalResources::GetFixedFunctionBinMask(); ++BinMask)
	{
		FFixedFunctionBin Bin;
		FNaniteRasterPipeline Pipeline = FNaniteRasterPipeline::GetFixedFunctionPipeline(BinMask);
		Bin.RasterBin = Register(Pipeline);
		Bin.BinMask = BinMask;
		check(Bin.RasterBin.BinIndex == BinMask);

		FixedFunctionBins.Emplace(Bin);
	}
}

void FNaniteRasterPipelines::ReleaseFixedFunctionBins()
{
	for (const FFixedFunctionBin& FixedFunctionBin : FixedFunctionBins)
	{
		Unregister(FixedFunctionBin.RasterBin);
	}

	FixedFunctionBins.Reset();
}

void FNaniteRasterPipelines::ReloadFixedFunctionBins()
{
	for (const FFixedFunctionBin& FixedFunctionBin : FixedFunctionBins)
	{
		FNaniteRasterPipeline Pipeline = FNaniteRasterPipeline::GetFixedFunctionPipeline(FixedFunctionBin.BinMask);
		FNaniteRasterEntry* RasterEntry = PipelineMap.Find(Pipeline);
		check(RasterEntry != nullptr);
		RasterEntry->RasterPipeline = Pipeline;
	}

	// Reset the entire raster setup cache
	for (const auto& Pair : PipelineMap)
	{
		Pair.Value.CacheMap.Reset();
	}
}

uint16 FNaniteRasterPipelines::AllocateBin(bool bPerPixelEval)
{
	TBitArray<>& BinUsageMask = bPerPixelEval ? PerPixelEvalPipelineBins : PipelineBins;
	
	int32 BinIndex = BinUsageMask.FindAndSetFirstZeroBit();
	if (BinIndex == INDEX_NONE)
	{
		BinIndex = BinUsageMask.Add(true);
	}

	check(int32(uint16(BinIndex)) == BinIndex && PipelineBins.Num() + PerPixelEvalPipelineBins.Num() < int32(NANITE_INVALID_RASTER_BIN));
	return bPerPixelEval ? FNaniteRasterBinIndexTranslator::RevertBinIndex(BinIndex) : uint16(BinIndex);
}

void FNaniteRasterPipelines::ReleaseBin(uint16 BinIndex)
{
	check(IsBinAllocated(BinIndex));
	if (BinIndex < PipelineBins.Num())
	{
		PipelineBins[BinIndex] = false;
	}
	else
	{
		PerPixelEvalPipelineBins[FNaniteRasterBinIndexTranslator::RevertBinIndex(BinIndex)] = false;
	}
}

bool FNaniteRasterPipelines::IsBinAllocated(uint16 BinIndex) const
{
	return BinIndex < PipelineBins.Num() ? PipelineBins[BinIndex] : PerPixelEvalPipelineBins[FNaniteRasterBinIndexTranslator::RevertBinIndex(BinIndex)];
}

uint32 FNaniteRasterPipelines::GetRegularBinCount() const
{
	return PipelineBins.FindLast(true) + 1;
}

uint32 FNaniteRasterPipelines::GetBinCount() const
{
	return GetRegularBinCount() + PerPixelEvalPipelineBins.FindLast(true) + 1;
}

FNaniteRasterBin FNaniteRasterPipelines::Register(const FNaniteRasterPipeline& InRasterPipeline)
{
	FNaniteRasterBin RasterBin;

	const FRasterHash RasterPipelineHash = PipelineMap.ComputeHash(InRasterPipeline);
	FRasterId RasterBinId = PipelineMap.FindOrAddIdByHash(RasterPipelineHash, InRasterPipeline, FNaniteRasterEntry());
	RasterBin.BinId = RasterBinId.GetIndex();

	FNaniteRasterEntry& RasterEntry = PipelineMap.GetByElementId(RasterBinId).Value;
	if (RasterEntry.ReferenceCount == 0)
	{
		// First reference
		RasterEntry.RasterPipeline = InRasterPipeline;
		RasterEntry.BinIndex = AllocateBin(InRasterPipeline.bPerPixelEval);
	}

	++RasterEntry.ReferenceCount;

	RasterBin.BinIndex = RasterEntry.BinIndex;
	return RasterBin;
}

void FNaniteRasterPipelines::Unregister(const FNaniteRasterBin& InRasterBin)
{
	FRasterId RasterBinId(InRasterBin.BinId);
	check(RasterBinId.IsValid());

	FNaniteRasterEntry& RasterEntry = PipelineMap.GetByElementId(RasterBinId).Value;

	check(RasterEntry.ReferenceCount > 0);
	--RasterEntry.ReferenceCount;
	if (RasterEntry.ReferenceCount == 0)
	{
		checkf(!ShouldBinRenderInCustomPass(InRasterBin.BinIndex), TEXT("A raster bin has dangling references to Custom Pass on final release."));
		ReleaseBin(RasterEntry.BinIndex);
		PipelineMap.RemoveByElementId(RasterBinId);
	}
}

void FNaniteRasterPipelines::RegisterBinForCustomPass(uint16 BinIndex)
{
	check(IsBinAllocated(BinIndex));

	const bool bPerPixelEval = BinIndex >= PipelineBins.Num();
	TArray<uint32>& RefCounts = bPerPixelEval ? PerPixelEvalCustomPassRefCounts : CustomPassRefCounts;
	const uint16 ArrayIndex = bPerPixelEval ? FNaniteRasterBinIndexTranslator::RevertBinIndex(BinIndex) : BinIndex;

	if (RefCounts.Num() <= ArrayIndex)
	{
		RefCounts.AddZeroed(ArrayIndex - RefCounts.Num() + 1);
	}
	RefCounts[ArrayIndex]++;
}

void FNaniteRasterPipelines::UnregisterBinForCustomPass(uint16 BinIndex)
{
	check(IsBinAllocated(BinIndex));

	const bool bPerPixelEval = BinIndex >= PipelineBins.Num();
	TArray<uint32>& RefCounts = bPerPixelEval ? PerPixelEvalCustomPassRefCounts : CustomPassRefCounts;
	const uint16 ArrayIndex = bPerPixelEval ? FNaniteRasterBinIndexTranslator::RevertBinIndex(BinIndex) : BinIndex;

	checkf(RefCounts.IsValidIndex(ArrayIndex), TEXT("Attempting to unregister a bin that was never registered for Custom Pass"));
	checkf(RefCounts[ArrayIndex] > 0, TEXT("Mismatched calls to RegisterBinForCustomPass/UnregisterBinForCustomPass"));

	RefCounts[ArrayIndex]--;
}

bool FNaniteRasterPipelines::ShouldBinRenderInCustomPass(uint16 BinIndex) const
{
	check(IsBinAllocated(BinIndex));

	const bool bPerPixelEval = BinIndex >= PipelineBins.Num();
	const TArray<uint32>& RefCounts = bPerPixelEval ? PerPixelEvalCustomPassRefCounts : CustomPassRefCounts;
	const uint16 ArrayIndex = bPerPixelEval ? FNaniteRasterBinIndexTranslator::RevertBinIndex(BinIndex) : BinIndex;

	return RefCounts.IsValidIndex(ArrayIndex) ? RefCounts[ArrayIndex] > 0 : false;
}


void FNaniteShadingPipelines::Init(bool bInBindless)
{
	bBindless = bInBindless;

	PipelineBins.Reserve(256);
	PipelineMap.Reserve(256);

	if (bBindless)
{
		PipelineGroups.Reserve(256);
		PipelineGroupMap.Reserve(256);
	}
}

uint16 FNaniteShadingPipelines::AllocateBin()
{
	TBitArray<>& BinUsageMask = PipelineBins;
	int32 BinIndex = BinUsageMask.FindAndSetFirstZeroBit();
	if (BinIndex == INDEX_NONE)
	{
		BinIndex = BinUsageMask.Add(true);
	}

	check(int32(uint16(BinIndex)) == BinIndex && PipelineBins.Num() <= int32(MAX_uint16));
	return uint16(BinIndex);
}

void FNaniteShadingPipelines::ReleaseBin(uint16 BinIndex)
{
	check(IsBinAllocated(BinIndex));
	if (BinIndex < PipelineBins.Num())
	{
		PipelineBins[BinIndex] = false;
	}
}

bool FNaniteShadingPipelines::IsBinAllocated(uint16 BinIndex) const
{
	return BinIndex < PipelineBins.Num() ? PipelineBins[BinIndex] : false;
}

uint32 FNaniteShadingPipelines::GetBinCount() const
{
	return PipelineBins.FindLast(true) + 1;
}

uint16 FNaniteShadingPipelines::AllocateGroup()
{
	int32 GroupIndex = PipelineGroups.FindAndSetFirstZeroBit();
	if (GroupIndex == INDEX_NONE)
	{
		GroupIndex = PipelineGroups.Add(true);
	}

	check(int32(uint16(GroupIndex)) == GroupIndex && PipelineGroups.Num() <= int32(MAX_uint16));
	return uint16(GroupIndex);
}

void FNaniteShadingPipelines::ReleaseGroup(uint16 GroupIndex)
{
	check(IsGroupAllocated(GroupIndex));
	if (GroupIndex < PipelineGroups.Num())
	{
		PipelineGroups[GroupIndex] = false;
	}
}

bool FNaniteShadingPipelines::IsGroupAllocated(uint16 GroupIndex) const
{
	return GroupIndex < PipelineGroups.Num() ? PipelineGroups[GroupIndex] : false;
}


void FNaniteShadingPipelines::RegisterGroup(const FNaniteShadingPipelineGroup& InShadingPipelineGroup, int32& GroupId, uint16& GroupIndex)
{
	const FShadingHash ShadingPipelineGroupHash = PipelineGroupMap.ComputeHash(InShadingPipelineGroup);
	FShadingId ShadingGroupId = PipelineGroupMap.FindOrAddIdByHash(ShadingPipelineGroupHash, InShadingPipelineGroup, FNaniteShadingGroupEntry());

	FNaniteShadingGroupEntry& ShadingGroupEntry = PipelineGroupMap.GetByElementId(ShadingGroupId).Value;
	if (ShadingGroupEntry.ReferenceCount == 0)
	{
		// First reference
		ShadingGroupEntry.ShadingPipelineGroup = MakeShared<FNaniteShadingPipelineGroup>(InShadingPipelineGroup);
		ShadingGroupEntry.GroupIndex = AllocateGroup();
	}

	++ShadingGroupEntry.ReferenceCount;
	
	GroupId = ShadingGroupId.GetIndex();
	GroupIndex = ShadingGroupEntry.GroupIndex;
}

void FNaniteShadingPipelines::UnregisterGroup(int32 GroupId)
{
	FShadingId ShadingGroupId(GroupId);
	check(ShadingGroupId.IsValid());

	FNaniteShadingGroupEntry& ShadingGroupEntry = PipelineGroupMap.GetByElementId(ShadingGroupId).Value;

	check(ShadingGroupEntry.ReferenceCount > 0);
	const uint32 GroupIndex = ShadingGroupEntry.GroupIndex;

	--ShadingGroupEntry.ReferenceCount;
	if (ShadingGroupEntry.ReferenceCount == 0)
	{
		ReleaseGroup(GroupIndex);
		PipelineGroupMap.RemoveByElementId(ShadingGroupId);
	}
}

FNaniteShadingBin FNaniteShadingPipelines::Register(const FNaniteShadingPipeline& InShadingPipeline)
{
	FNaniteShadingBin ShadingBin;

	const FShadingHash ShadingPipelineHash = PipelineMap.ComputeHash(InShadingPipeline);
	FShadingId ShadingBinId = PipelineMap.FindOrAddIdByHash(ShadingPipelineHash, InShadingPipeline, FNaniteShadingEntry());
	ShadingBin.BinId = ShadingBinId.GetIndex();

	FNaniteShadingEntry& ShadingEntry = PipelineMap.GetByElementId(ShadingBinId).Value;
	if (ShadingEntry.ReferenceCount == 0)
	{
		// First reference
		ShadingEntry.ShadingPipeline = MakeShared<FNaniteShadingPipeline>(InShadingPipeline);
		ShadingEntry.BinIndex = AllocateBin();

		if(bBindless)
		{
			FNaniteShadingPipelineGroup PipelineGroup;
			PipelineGroup.ShaderBindings = InShadingPipeline.ShaderBindings;
			PipelineGroup.ShaderBindingsHash = InShadingPipeline.ShaderBindingsHash;
			PipelineGroup.ComputeShader = InShadingPipeline.ComputeShader;
			PipelineGroup.WorkGraphShader = InShadingPipeline.WorkGraphShader;

			RegisterGroup(PipelineGroup, ShadingEntry.GroupId, ShadingEntry.GroupIndex);
		}

		bBuildIdList = true;
	}

	if(bBindless)
	{
		FNaniteShadingGroupEntry& ShadingGroupEntry = PipelineGroupMap.GetByElementId(ShadingEntry.GroupId).Value;

		checkf(ShadingGroupEntry.ShadingPipelineGroup->ComputeShader == InShadingPipeline.ComputeShader
			&& ShadingGroupEntry.ShadingPipelineGroup->WorkGraphShader == InShadingPipeline.WorkGraphShader,
			TEXT("Nanite shading pipeline shader mismatch with group for material '%s'. Pipeline CS=%p, Group CS=%p, Pipeline WG=%p, Group WG=%p"),
			InShadingPipeline.MaterialProxy ? *InShadingPipeline.MaterialProxy->GetMaterialName() : TEXT("unknown"),
			InShadingPipeline.ComputeShader, ShadingGroupEntry.ShadingPipelineGroup->ComputeShader,
			InShadingPipeline.WorkGraphShader, ShadingGroupEntry.ShadingPipelineGroup->WorkGraphShader);

		ShadingGroupEntry.ShadingPipelineGroup->Pipelines.Add(ShadingEntry.ShadingPipeline);
		ShadingGroupEntry.ShadingPipelineGroup->PipelineShadingBins.Add(ShadingEntry.BinIndex);
	}

	++ShadingEntry.ReferenceCount;

	ShadingBin.BinIndex = ShadingEntry.BinIndex;
	return ShadingBin;
}

void FNaniteShadingPipelines::Unregister(const FNaniteShadingBin& InShadingBin)
{
	FShadingId ShadingBinId(InShadingBin.BinId);
	check(ShadingBinId.IsValid());

	FNaniteShadingEntry& ShadingEntry = PipelineMap.GetByElementId(ShadingBinId).Value;

	check(ShadingEntry.ReferenceCount > 0);
	--ShadingEntry.ReferenceCount;
	if (ShadingEntry.ReferenceCount == 0)
	{
		if(bBindless)
		{
			FNaniteShadingGroupEntry& ShadingGroupEntry = PipelineGroupMap.GetByElementId(ShadingEntry.GroupId).Value;
			ShadingGroupEntry.ShadingPipelineGroup->Pipelines.Remove(ShadingEntry.ShadingPipeline);
			ShadingGroupEntry.ShadingPipelineGroup->PipelineShadingBins.Remove(ShadingEntry.BinIndex);
			UnregisterGroup(ShadingEntry.GroupId);
		}

		ReleaseBin(ShadingEntry.BinIndex);
		PipelineMap.RemoveByElementId(ShadingBinId);
		bBuildIdList = true;
	}
}

void FNaniteShadingPipelines::BuildIdList()
{
	if (bBuildIdList)
	{
		ShadingIdList.Reset(PipelineMap.Num());

		for (auto Iter = PipelineMap.begin(); Iter != PipelineMap.end(); ++Iter)
		{
			ShadingIdList.Add(Iter.GetElementId());
		}

		bBuildIdList = false;
	}
}

const TConstArrayView<const FNaniteShadingPipelines::FShadingId> FNaniteShadingPipelines::GetIdList() const
{
	check(!bBuildIdList);
	return ShadingIdList;
}

static void ComputeMaterialRelevance_Thread(
	const ERHIFeatureLevel::Type InFeatureLevel,
	const FNaniteShadingPipelineMap& InPipelineMap,
	const FNaniteShadingPipelines::FShadingId& InShadingId,
	FMaterialRelevance& OutMaterialRelevance
)
{
	const FNaniteShadingEntry& ShadingEntry = InPipelineMap.GetByElementId(InShadingId).Value;

	if (ShadingEntry.ShadingPipeline.IsValid())
	{
		const FMaterialRenderProxy* MaterialProxy = ShadingEntry.ShadingPipeline->MaterialProxy;
		const FMaterial* Material = ShadingEntry.ShadingPipeline->Material;
		if (MaterialProxy && Material)
		{
			const UMaterialInterface* MaterialInterface = MaterialProxy->GetMaterialInterface();
			if (MaterialInterface)
			{
				OutMaterialRelevance |= MaterialInterface->GetRelevance_Concurrent(GetFeatureLevelShaderPlatform_Checked(InFeatureLevel));
			}
		}
	}
}

void FNaniteShadingPipelines::ComputeRelevance(ERHIFeatureLevel::Type InFeatureLevel)
{
	// Reset relevance
	CombinedRelevance = FPrimitiveViewRelevance();

	struct FRelevanceContext
	{
		FMaterialRelevance MaterialRelevance{};
	};

	TArray<FRelevanceContext, TInlineAllocator<8>> RelevanceContexts;

	BuildIdList();

	if (ShadingIdList.Num() > 0)
	{
		CombinedRelevance.bDrawRelevance	= true;
		CombinedRelevance.bStaticRelevance	= true;
		CombinedRelevance.bRenderInMainPass	= true;
		CombinedRelevance.bShadowRelevance	= true;

		// Nanite::GetSupportsCustomDepthRendering() && ShouldRenderCustomDepth();
		CombinedRelevance.bRenderCustomDepth = false; // TODO: Unsupported in fast path

		// GetLightingChannelMask() != GetDefaultLightingChannelMask();
		CombinedRelevance.bUsesLightingChannels = false; // TODO: Unsupported in fast path

		if (GNaniteCacheRelevanceParallel && FApp::ShouldUseThreadingForPerformance())
		{
			ParallelForWithTaskContext(
				RelevanceContexts,
				ShadingIdList.Num(),
				[this, InFeatureLevel](FRelevanceContext& Context, int32 Index)
				{
					FTaskTagScope Scope(ETaskTag::EParallelRenderingThread);
					const FNaniteShadingPipelines::FShadingId& ShadingId = ShadingIdList[Index];
					ComputeMaterialRelevance_Thread(InFeatureLevel, PipelineMap, ShadingId, Context.MaterialRelevance);
				}
			);

			for (int32 MergeIndex = 1; MergeIndex < RelevanceContexts.Num(); ++MergeIndex)
			{
				// Update combined material relevance
				RelevanceContexts[0].MaterialRelevance |= RelevanceContexts[MergeIndex].MaterialRelevance;
			}

			// Apply combined material relevance to combined primitive view relevance
			RelevanceContexts[0].MaterialRelevance.SetPrimitiveViewRelevance(CombinedRelevance);
		}
		else
		{
			FMaterialRelevance MaterialRelevance{};

			for (const FNaniteShadingPipelines::FShadingId& ShadingId : ShadingIdList)
			{
				// Update combined material relevance
				ComputeMaterialRelevance_Thread(InFeatureLevel, PipelineMap, ShadingId, MaterialRelevance);
			}

			// Apply combined material relevance to combined primitive view relevance
			MaterialRelevance.SetPrimitiveViewRelevance(CombinedRelevance);
		}
	}
}

struct FLumenShadingBinEntry
{
	FLumenShadingBinEntry(int32 InBuildIndex, const FNaniteShadingBin& InShadingBin)
	: BuildIndex(InBuildIndex)
	, ShadingBin(InShadingBin)
	{
	}

	inline friend uint32 GetTypeHash(const FLumenShadingBinEntry& InEntry)
	{
		return uint32(InEntry.ShadingBin.BinId);
	}

	inline bool operator==(const FLumenShadingBinEntry& Other) const
	{
		return ShadingBin == Other.ShadingBin;
	}

	int32 BuildIndex = INDEX_NONE;
	FNaniteShadingBin ShadingBin;
};

BEGIN_SHADER_PARAMETER_STRUCT(FLumenMeshCapturePassParameters, )
	SHADER_PARAMETER_STRUCT_INCLUDE(FViewShaderParameters, View)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneUniformParameters, Scene)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FNaniteRasterUniformParameters, NaniteRaster)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FNaniteShadingUniformParameters, NaniteShading)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FOpaqueBasePassUniformParameters, BasePass)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FLumenCardPassUniformParameters, CardPass)
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FLumenCardOutputs, LumenCardOutputs)
END_SHADER_PARAMETER_STRUCT()

void DispatchLumenMeshCapturePass(
	FRDGBuilder& GraphBuilder,
	FScene& Scene,
	FViewInfo* SharedView,
	TArrayView<const FCardPageRenderData> CardPagesToRender,
	const Nanite::FRasterResults& RasterResults,
	const Nanite::FRasterContext& RasterContext,
	FLumenCardPassUniformParameters* PassUniformParameters,
	FRDGBufferSRVRef RectMinMaxBufferSRV,
	uint32 NumRects,
	FIntPoint ViewportSize,
	FRDGTextureRef AlbedoAtlasTexture,
	FRDGTextureRef NormalAtlasTexture,
	FRDGTextureRef EmissiveAtlasTexture,
	FRDGTextureRef DepthAtlasTexture
)
{
	checkSlow(DoesPlatformSupportNanite(GMaxRHIShaderPlatform));
	checkSlow(DoesPlatformSupportLumenGI(GMaxRHIShaderPlatform));

	LLM_SCOPE_BYTAG(Nanite);
	RDG_EVENT_SCOPE(GraphBuilder, "Nanite::LumenMeshCapturePass");
	TRACE_CPUPROFILER_EVENT_SCOPE(Nanite_LumenMeshCapturePass);

	const FRDGSystemTextures& SystemTextures = FRDGSystemTextures::Get(GraphBuilder);

	FNaniteShadingCommands& ShadingCommands = Scene.NaniteShadingCommands[ENaniteMeshPass::LumenCardCapture];
	ShadingCommands.SetupTask.Wait();
	ShadingCommands.BuildCommandsTask.Wait();

	struct FLumenCaptureTile
	{
		// Top Left X: 8 bits (tile x in card atlas) - multiplied by 8 and added to card view rect min.x in shader
		// Top Left Y: 8 bits (tile y in card atlas) - multiplied by 8 and added to card view rect min.y in shader
		// Card Index: 16 bits
		uint32 Packed;
	};

	struct FLumenCapturePass
	{
		FNaniteShadingBin ShadingBin;
		TArray<uint16, TInlineAllocator<64>> ViewIndices;
		uint32 TotalTileCount = 0;

		bool operator<(const FLumenCapturePass& Other) const
		{
			return ShadingBin.BinIndex < Other.ShadingBin.BinIndex;
		}
	};

	struct FLumenShadingBinMeta
	{
		uint32 DataByteOffset;
	};

	struct FLumenCaptureContext
	{
		uint32 TotalPassCount = 0;
		uint32 TotalTileCount = 0;

		TArray<FLumenCapturePass, SceneRenderingAllocator> Passes;
		TArray<uint32, SceneRenderingAllocator> ViewIndices;
		TArray<Nanite::FPackedView, SceneRenderingAllocator> PackedViews;

		uint32 ShadingBinCount = 0;

		uint32 NumBytes_Meta = 0;
		uint32 NumBytes_Data = 0;

		uint32 MaxShadingBin = 0u;

		TArray<uint32, SceneRenderingAllocator> ShadingBinData;
	};

	FLumenCaptureContext& CaptureContext = *GraphBuilder.AllocObject<FLumenCaptureContext>();

	GraphBuilder.AddSetupTask([&CaptureContext, CardPagesToRender, &Scene, ViewportSize, SharedView]
	{
		TRACE_CPUPROFILER_EVENT_SCOPE(BuildLumenMeshCaptureMaterialPasses);

		CaptureContext.Passes.Reserve(CardPagesToRender.Num());
		CaptureContext.PackedViews.Reserve(CardPagesToRender.Num());
		CaptureContext.MaxShadingBin = 0u;

		CaptureContext.TotalTileCount = 0;

		// Determine unique list of shading bins
		Experimental::TRobinHoodHashSet<FLumenShadingBinEntry> CapturePassSet;

		for (int32 CardPageIndex = 0; CardPageIndex < CardPagesToRender.Num(); ++CardPageIndex)
		{
			const FCardPageRenderData& CardPageRenderData = CardPagesToRender[CardPageIndex];
			check((CardPageRenderData.CardCaptureAtlasRect.Min.X & 7u) == 0 &&
				  (CardPageRenderData.CardCaptureAtlasRect.Min.Y & 7u) == 0);

			if (!CardPageRenderData.NeedsRender())
			{
				continue;
			}

			const uint32 CardWidth  = CardPageRenderData.CardCaptureAtlasRect.Width();
			const uint32 CardHeight = CardPageRenderData.CardCaptureAtlasRect.Height();
			check((CardWidth & 7u) == 0 && (CardHeight & 7u) == 0);

			const uint32 TilesWide = CardWidth  >> 3u;
			const uint32 TilesTall = CardHeight >> 3u;
			check(TilesWide <= 256 && TilesTall <= 256);

			const uint32 TileCount = TilesWide * TilesTall;

			for (const FNaniteShadingBin& ShadingBin : CardPageRenderData.NaniteShadingBins)
			{
				const FLumenShadingBinEntry& ShadingBinEntry = *CapturePassSet.FindOrAdd(FLumenShadingBinEntry(CaptureContext.Passes.Num(), ShadingBin));

				if (ShadingBinEntry.BuildIndex >= CaptureContext.Passes.Num())
				{
					FLumenCapturePass CapturePass;
					CapturePass.ShadingBin = ShadingBin;
					CaptureContext.Passes.Emplace(CapturePass);
					CaptureContext.MaxShadingBin = FMath::Max<uint32>(CaptureContext.MaxShadingBin, uint32(ShadingBin.BinIndex));
				}

				CaptureContext.Passes[ShadingBinEntry.BuildIndex].ViewIndices.Add(CardPageIndex);
				CaptureContext.Passes[ShadingBinEntry.BuildIndex].TotalTileCount += TileCount;
				
				CaptureContext.TotalTileCount += TileCount;
				++CaptureContext.TotalPassCount;
			}

			//check(CaptureContext.Passes.Num() > 0);
		}

		if (CaptureContext.Passes.Num() > 0)
		{
			TRACE_CPUPROFILER_EVENT_SCOPE(Sort);
			CaptureContext.Passes.Sort();
		}

		CaptureContext.ShadingBinCount = CaptureContext.MaxShadingBin + 1u;

		CaptureContext.NumBytes_Meta = CaptureContext.ShadingBinCount * sizeof(FLumenShadingBinMeta);
		CaptureContext.NumBytes_Data = CaptureContext.TotalTileCount * sizeof(FLumenCaptureTile);

		CaptureContext.ShadingBinData.SetNumUninitialized((CaptureContext.NumBytes_Meta + CaptureContext.NumBytes_Data) >> 2u);
		uint8* ShadingBinDataPtr = reinterpret_cast<uint8*>(CaptureContext.ShadingBinData.GetData());

		uint32 DataWriteOffset = CaptureContext.NumBytes_Meta;

		// We only need to zero the shading bin meta data headers
		FMemory::Memzero(ShadingBinDataPtr, CaptureContext.NumBytes_Meta);

		for (FLumenCapturePass& CapturePass : CaptureContext.Passes)
		{
			FLumenShadingBinMeta& MetaEntry = reinterpret_cast<FLumenShadingBinMeta*>(ShadingBinDataPtr)[CapturePass.ShadingBin.BinIndex];
			MetaEntry.DataByteOffset = DataWriteOffset;

			DataWriteOffset += (sizeof(FLumenCaptureTile) * CapturePass.TotalTileCount);

			FLumenCaptureTile* TileData = reinterpret_cast<FLumenCaptureTile*>(ShadingBinDataPtr + MetaEntry.DataByteOffset);

			for (uint32 ViewIndex : CapturePass.ViewIndices)
			{
				const FCardPageRenderData& CardPageRenderData = CardPagesToRender[ViewIndex];
				const uint32 TilesWide = CardPageRenderData.CardCaptureAtlasRect.Width()  >> 3u;
				const uint32 TilesTall = CardPageRenderData.CardCaptureAtlasRect.Height() >> 3u;
				for (uint32 TileX = 0; TileX < TilesWide; ++TileX)
				{
					for (uint32 TileY = 0; TileY < TilesTall; ++TileY)
					{
						FLumenCaptureTile* Tile = new(TileData) FLumenCaptureTile;
						Tile->Packed = (TileX & 0xFFu) | ((TileY & 0xFFu) << 8u) | ((ViewIndex & 0xFFFFu) << 16u);
						++TileData;
					}
				}
			}
		}

		for (const FCardPageRenderData& CardPageRenderData : CardPagesToRender)
		{
			Nanite::FPackedViewParams Params;
			CardPageRenderData.GetViewMatrices(*SharedView, Params.ViewMatrices);
			Params.PrevViewMatrices = Params.ViewMatrices;
			Params.ViewRect = CardPageRenderData.CardCaptureAtlasRect;
			Params.RasterContextSize = ViewportSize;
			Params.MaxPixelsPerEdgeMultipler = 1.0f;

			CaptureContext.PackedViews.Add(Nanite::CreatePackedView(Params));
		}
	});

	FRDGBuffer* PackedViewBuffer = CreateStructuredBuffer(
		GraphBuilder,
		TEXT("Nanite.PackedViews"),
		CaptureContext.PackedViews.GetTypeSize(),
		[&PackedViews = CaptureContext.PackedViews] { return FMath::RoundUpToPowerOfTwo(PackedViews.Num()); },
		[&PackedViews = CaptureContext.PackedViews] { return PackedViews.GetData(); },
		[&PackedViews = CaptureContext.PackedViews] { return PackedViews.Num() * PackedViews.GetTypeSize(); }
	);

	FRDGBuffer* ShadingBinData = CreateByteAddressBuffer(
		GraphBuilder,
		TEXT("Nanite.ShadingBinData"),
		[&BinData = CaptureContext.ShadingBinData]() -> auto& { return BinData; }
	);

	FLumenMeshCapturePassParameters* LumenCardPassParameters = GraphBuilder.AllocParameters<FLumenMeshCapturePassParameters>();

	{
		// NaniteRaster Uniform Buffer
		{
			FNaniteRasterUniformParameters* UniformParameters	= GraphBuilder.AllocParameters<FNaniteRasterUniformParameters>();

			UniformParameters->PageConstants				= RasterResults.PageConstants;
			UniformParameters->MaxNodes						= Nanite::FGlobalResources::GetMaxNodes();
			UniformParameters->MaxVisibleClusters			= Nanite::FGlobalResources::GetMaxVisibleClusters();
			UniformParameters->MaxCandidatePatches			= Nanite::FGlobalResources::GetMaxCandidatePatches();
			UniformParameters->MaxPatchesPerGroup			= RasterResults.MaxPatchesPerGroup;
			UniformParameters->MeshPass						= RasterResults.MeshPass;
			UniformParameters->InvDiceRate					= RasterResults.InvDiceRate;
			UniformParameters->RenderFlags					= RasterResults.RenderFlags;
			UniformParameters->DebugFlags					= RasterResults.DebugFlags;
			UniformParameters->CurveSubdivisionThreshold 	= Nanite::FGlobalResources::GetCurveSubdivisionThreshold();
			UniformParameters->CurveSubdivisionMaxSegments	= Nanite::FGlobalResources::GetCurveSubdivisionMaxSegments();

			LumenCardPassParameters->NaniteRaster				= GraphBuilder.CreateUniformBuffer(UniformParameters);
		}

		// NaniteShading Uniform Buffer
		{
			FNaniteShadingUniformParameters* UniformParameters	= GraphBuilder.AllocParameters<FNaniteShadingUniformParameters>();

			UniformParameters->ClusterPageData				= Nanite::GStreamingManager.GetClusterPageDataSRV(GraphBuilder);
			UniformParameters->HierarchyBuffer				= Nanite::GStreamingManager.GetHierarchySRV(GraphBuilder);
			UniformParameters->VisibleClustersSWHW			= GraphBuilder.CreateSRV(RasterResults.VisibleClustersSWHW);
			UniformParameters->AssemblyTransforms			= GraphBuilder.CreateSRV(RasterResults.AssemblyTransforms);
			
			UniformParameters->VisBuffer64					= RasterContext.VisBuffer64;
			UniformParameters->DbgBuffer64					= SystemTextures.Black;
			UniformParameters->DbgBuffer32					= SystemTextures.Black;
			UniformParameters->ShadingMask					= SystemTextures.Black;

			UniformParameters->ShadingBinData				= GraphBuilder.CreateSRV(ShadingBinData);
			UniformParameters->ThreadGroupData				= GraphBuilder.CreateSRV(GSystemTextures.GetDefaultStructuredBuffer<uint32>(GraphBuilder));

			UniformParameters->MultiViewEnabled				= 1;
			UniformParameters->MultiViewIndices				= GraphBuilder.CreateSRV(GSystemTextures.GetDefaultStructuredBuffer<uint32>(GraphBuilder));
			UniformParameters->MultiViewRectScaleOffsets	= GraphBuilder.CreateSRV(GSystemTextures.GetDefaultStructuredBuffer<FVector4>(GraphBuilder));
			UniformParameters->InViews						= GraphBuilder.CreateSRV(PackedViewBuffer);

			LumenCardPassParameters->NaniteShading			= GraphBuilder.CreateUniformBuffer(UniformParameters);
		}
	}

	CardPagesToRender[0].PatchView(&Scene, SharedView);
	LumenCardPassParameters->View = SharedView->GetShaderParameters();
	LumenCardPassParameters->Scene = SharedView->GetSceneUniforms().GetBuffer(GraphBuilder);
	LumenCardPassParameters->CardPass = GraphBuilder.CreateUniformBuffer(PassUniformParameters);

	{
		FLumenCardOutputs* Outputs = GraphBuilder.AllocParameters<FLumenCardOutputs>();

		// No possibility of read/write hazard due to fully resolved vbuffer/materials
		const ERDGUnorderedAccessViewFlags OutTargetFlags = ERDGUnorderedAccessViewFlags::SkipBarrier;

		Outputs->OutTarget0 = GraphBuilder.CreateUAV(AlbedoAtlasTexture, OutTargetFlags);
		Outputs->OutTarget1 = GraphBuilder.CreateUAV(NormalAtlasTexture, OutTargetFlags);
		Outputs->OutTarget2 = GraphBuilder.CreateUAV(EmissiveAtlasTexture, OutTargetFlags);

		LumenCardPassParameters->LumenCardOutputs = GraphBuilder.CreateUniformBuffer(Outputs);
	}

	GraphBuilder.AddPass(
		RDG_EVENT_NAME("LumenShadeCS"),
		LumenCardPassParameters,
		ERDGPassFlags::Compute,
		[LumenCardPassParameters, SharedView, &ShadingCommands, &CapturePasses = CaptureContext.Passes]
		(FRDGAsyncTask, FRHIComputeCommandList& RHICmdList)
		{
			// This is processed within the RDG pass lambda, so the setup task should be complete by now.
			check(ShadingCommands.BuildCommandsTask.IsCompleted());

			TRACE_CPUPROFILER_EVENT_SCOPE(LumenEmitGBuffer);
			SCOPED_DRAW_EVENTF(RHICmdList, LumenEmitGBuffer, TEXT("%d materials"), CapturePasses.Num());

			FRHIBatchedShaderParameters& BatchedParameters = RHICmdList.GetScratchShaderParameters();
			check(!BatchedParameters.HasParameters());

			static int32 PSOCollectorIndex = FPSOCollectorCreateManager::GetIndex(EShadingPath::Deferred, TEXT("NaniteLumenCard"));

			for (const FLumenCapturePass& CapturePass : CapturePasses)
			{
				const int32 CommandIndex = ShadingCommands.CommandLookup[CapturePass.ShadingBin.BinIndex];
				FNaniteShadingCommand& ShadingCommand = ShadingCommands.Commands[CommandIndex];
				check(ShadingCommand.ArgsIndex == CapturePass.ShadingBin.BinIndex);
				
				if (!Nanite::PrepareShadingCommand(ShadingCommand, PSOCollectorIndex))
				{
					break;
				}

				const FNaniteShadingPipeline& ShadingPipeline = *ShadingCommand.Pipeline;
			#if WANTS_DRAW_MESH_EVENTS
				SCOPED_CONDITIONAL_DRAW_EVENTF(RHICmdList, LumenCS, GShowMaterialDrawEvents != 0 && !ShadingPipeline.bVoxel && !ShadingPipeline.bCurve, TEXT("%s [%d tiles]"),			GetShadingMaterialName(ShadingPipeline.MaterialProxy), CapturePass.TotalTileCount);
				SCOPED_CONDITIONAL_DRAW_EVENTF(RHICmdList, LumenCS, GShowMaterialDrawEvents != 0 &&  ShadingPipeline.bVoxel && !ShadingPipeline.bCurve, TEXT("%s [%d tiles] (Voxels)"),	GetShadingMaterialName(ShadingPipeline.MaterialProxy), CapturePass.TotalTileCount);
				SCOPED_CONDITIONAL_DRAW_EVENTF(RHICmdList, LumenCS, GShowMaterialDrawEvents != 0 && !ShadingPipeline.bVoxel &&  ShadingPipeline.bCurve, TEXT("%s [%d tiles] (Curves)"),	GetShadingMaterialName(ShadingPipeline.MaterialProxy), CapturePass.TotalTileCount);
			#endif

				TRDGUniformBufferRef<FLumenCardOutputs> LumenCardOutputs = LumenCardPassParameters->LumenCardOutputs.GetUniformBuffer();

				// Record parameters
				FRHIBatchedShaderParameters& ShadingParameters = RHICmdList.GetScratchShaderParameters();
				Nanite::RecordLumenCardParameters(ShadingParameters, ShadingCommand, LumenCardPassParameters->LumenCardOutputs->GetRHIRef());

				// Record dispatch
				{
					FRHIComputeShader* ComputeShaderRHI = ShadingPipeline.ComputeShader;
					SetComputePipelineState(RHICmdList, ComputeShaderRHI);

					if (GRHISupportsShaderRootConstants)
					{
						RHICmdList.SetShaderRootConstants(ShadingCommand.PassData);
					}

					RHICmdList.SetBatchedShaderParameters(ComputeShaderRHI, ShadingParameters);
					RHICmdList.DispatchComputeShader(CapturePass.TotalTileCount, 1, 1);
				}
			}
		}
	);

	// Mark scene stencil for all Nanite pixels
	{
		MarkSceneStencilRects(
			GraphBuilder,
			RasterContext,
			Scene,
			SharedView,
			ViewportSize,
			NumRects,
			RectMinMaxBufferSRV,
			DepthAtlasTexture
		);
	}

	// Emit scene depth values for all Nanite pixels
	{
		EmitSceneDepthRects(
			GraphBuilder,
			RasterContext,
			Scene,
			SharedView,
			ViewportSize,
			NumRects,
			RectMinMaxBufferSRV,
			DepthAtlasTexture
		);
	}
}
